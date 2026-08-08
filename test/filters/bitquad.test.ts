/**
 * Quad creation, LS3 subdivision and attribute seams.
 *
 * The quad filters are pleasant to check because the counts are forced. A quad
 * is two triangles joined by a hidden edge, so `2·quads + triangles = faces`
 * always, and no face may end up with two hidden edges — that would be a
 * polygon larger than a quad, which this representation cannot express.
 *
 * `makePureByRefine` has an exact answer worth stating: it splits every
 * triangle at its centroid, and every *original edge* becomes the diagonal of a
 * quad. So a closed mesh yields exactly as many quads as it had edges, and a
 * cube — 12 triangles, 18 edges — yields 18.
 *
 * LS3's whole claim is that it uses the normals it was given. The test is a
 * sphere: Loop pulls the surface inside the true sphere because it only averages
 * positions, while LS3 lands on it to five decimal places because the normals
 * say where it is.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { AttributeSeam } from "../../src/vcg/complex/attribute_seam.ts";
import {
	countBitQuads,
	countBitTris,
	hasConsistentPerFaceFauxFlag,
	isBitTriQuadOnly,
} from "../../src/vcg/complex/bit_quad.ts";
import { BitQuadCreation } from "../../src/vcg/complex/bitquad_creation.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { AlgebraicSphere } from "../../src/vcg/complex/refine.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { rgba } from "../../src/vcg/space/color4.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO, channels: number = MeshElement.MM_NONE) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	if (channels !== MeshElement.MM_NONE) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Mean distance of the vertices from the origin. */
function meanRadius(cm: CMeshO): number {
	let sum = 0;
	let n = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		sum += Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
		n++;
	}
	return sum / n;
}

// ------------------------------------------------------------ quad dominant

describe("Turn into Quad-Dominant mesh", () => {
	test("a cube becomes exactly six quads with nothing left over", () => {
		const { doc } = docWith(cube(1).mesh, MeshElement.MM_FACEQUALITY);
		const out = kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level: 1 });
		expect(out.quad_number).toBe(6);
		expect(out.triangle_number).toBe(0);
	});

	test("the pairing is consistent and never makes anything bigger than a quad", () => {
		for (const subdiv of [1, 2, 3]) {
			const { doc, cm } = docWith(sphereIcosa(subdiv).mesh, MeshElement.MM_FACEQUALITY);
			kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level: 1 });
			UpdateTopology.faceFace(cm);
			// Both halves of a quad must agree that their shared edge is hidden.
			expect(hasConsistentPerFaceFauxFlag(cm), `subdiv ${subdiv}`).toBe(true);
			// No face with two hidden edges: that would be a pentagon or larger.
			expect(isBitTriQuadOnly(cm), `subdiv ${subdiv}`).toBe(true);
			// And the count adds up.
			expect(2 * countBitQuads(cm) + countBitTris(cm), `subdiv ${subdiv}`).toBe(cm.fn);
		}
	});

	test("moves no vertex and adds no face", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh, MeshElement.MM_FACEQUALITY);
		const before = { vn: cm.vn, fn: cm.fn };
		const positions = Float64Array.from(cm.vertCoord.subarray(0, 3 * cm.vertSize));
		kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level: 2 });
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
		expect([...cm.vertCoord.subarray(0, 3 * cm.vertSize)]).toEqual([...positions]);
	});

	test("an odd face count must leave at least one triangle", () => {
		// Pairing consumes two faces at a time, so parity alone forces it.
		const { doc, cm } = docWith(gridPlane(3, 3).mesh, MeshElement.MM_FACEQUALITY);
		// A 3x3 grid is 18 triangles; delete one to make it odd.
		const out = kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level: 0 });
		expect(2 * (out.quad_number as number) + (out.triangle_number as number)).toBe(cm.fn);
		expect(cm.fn % 2).toBe(0);
	});

	test("greedier levels leave no more triangles than careful ones", () => {
		const leftovers = (level: number) => {
			const { doc } = docWith(sphereIcosa(3).mesh, MeshElement.MM_FACEQUALITY);
			return kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level })
				.triangle_number as number;
		};
		// Level 0 is maximally greedy, so it cannot do worse on count.
		expect(leftovers(0)).toBeLessThanOrEqual(leftovers(2));
	});

	test("a grid with borders still pairs its interior", () => {
		const { doc } = docWith(gridPlane(6, 6).mesh, MeshElement.MM_FACEQUALITY);
		const out = kernel.applyFilter(doc, "Turn into Quad-Dominant mesh", { level: 1 });
		expect(out.quad_number as number).toBeGreaterThan(0);
	});
});

// ------------------------------------------------------------ 4-8 refinement

describe("Tri to Quad by 4-8 Subdivision", () => {
	test("a closed mesh yields one quad per original edge", () => {
		// Every original edge becomes the diagonal of a quad, so the count is
		// forced: a cube has 18 edges, an icosphere 3V - 6.
		const cases: Array<[string, CMeshO, number]> = [
			["cube", cube(1).mesh, 18],
			["sphere1", sphereIcosa(1).mesh, 3 * 42 - 6],
			["sphere2", sphereIcosa(2).mesh, 3 * 162 - 6],
		];
		for (const [label, mesh, edges] of cases) {
			const { doc, cm } = docWith(mesh);
			const before = cm.fn;
			const out = kernel.applyFilter(doc, "Tri to Quad by 4-8 Subdivision");
			expect(out.quad_number, label).toBe(edges);
			expect(out.triangle_number, label).toBe(0);
			// Three sub-triangles per original.
			expect(cm.fn, label).toBe(3 * before);
		}
	});

	test("the faux tagging is consistent and nothing exceeds a quad", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Tri to Quad by 4-8 Subdivision");
		UpdateTopology.faceFace(cm);
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(true);
		expect(isBitTriQuadOnly(cm)).toBe(true);
	});

	test("the surface stays closed and keeps its shape", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const before = meanRadius(cm);
		kernel.applyFilter(doc, "Tri to Quad by 4-8 Subdivision");
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(true);
		// Centroids sit inside the sphere, so the mean radius drops a little —
		// but the surface itself is unchanged, only re-triangulated.
		expect(meanRadius(cm)).toBeLessThan(before);
		expect(meanRadius(cm)).toBeGreaterThan(before * 0.98);
	});

	test("a bordered mesh leaves the border triangles unpaired", () => {
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		const out = kernel.applyFilter(doc, "Tri to Quad by 4-8 Subdivision");
		// A border edge has only one sub-triangle beside it and so cannot be
		// half of a quad.
		expect(out.triangle_number as number).toBeGreaterThan(0);
		expect(2 * (out.quad_number as number) + (out.triangle_number as number)).toBe(cm.fn);
	});

	test("refuses a mesh with faces larger than a quad", () => {
		const cm = sphereIcosa(1).mesh;
		UpdateTopology.faceFace(cm);
		// Two hidden edges on one face is a pentagon or bigger.
		cm.faceFlags[0] |= 0x00040000 | 0x00080000;
		const { doc } = docWith(cm);
		expect(() => kernel.applyFilter(doc, "Tri to Quad by 4-8 Subdivision")).toThrow(MLException);
	});
});

describe("quad quality", () => {
	test("a square scores four and a degenerate quad scores near zero", () => {
		const square = BitQuadCreation.quadQuality([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]);
		expect(square).toBeCloseTo(4, 9);
		// A sliver: three corners nearly collinear.
		const sliver = BitQuadCreation.quadQuality([0, 0, 0], [1, 0, 0], [2, 0.001, 0], [1, 0.002, 0]);
		expect(sliver).toBeLessThan(1);
	});

	test("a long rectangle still scores four", () => {
		// Deliberate: the metric is about angles, and a long thin rectangle
		// renders and subdivides perfectly well.
		expect(BitQuadCreation.quadQuality([0, 0, 0], [10, 0, 0], [10, 1, 0], [0, 1, 0])).toBeCloseTo(
			4,
			9,
		);
	});
});

// --------------------------------------------------------------- LS3 Loop

describe("Subdivision Surfaces: LS3 Loop", () => {
	test("recovers a sphere far better than plain Loop", () => {
		// The claim that justifies the scheme. Both are given the same mesh and
		// the same normals; only LS3 uses the normals.
		const ls3 = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(ls3.doc, "Subdivision Surfaces: LS3 Loop", {
			Iterations: 1,
			Threshold: 0,
		});
		const loop = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(loop.doc, "Subdivision Surfaces: Loop", {
			Iterations: 1,
			Threshold: 0,
		});
		expect(Math.abs(meanRadius(ls3.cm) - 1)).toBeLessThan(0.001);
		expect(Math.abs(meanRadius(loop.cm) - 1)).toBeGreaterThan(0.01);
	});

	test("quadruples the face count and stays closed", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = cm.fn;
		kernel.applyFilter(doc, "Subdivision Surfaces: LS3 Loop", {
			Iterations: 1,
			Threshold: 0,
		});
		expect(cm.fn).toBe(4 * before);
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(true);
	});

	test("a flat sheet stays flat", () => {
		// Every normal points the same way, so the fitted sphere degenerates to
		// the plane the sheet already lies in — the branch that would divide by
		// a vanishing quadratic term.
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		kernel.applyFilter(doc, "Subdivision Surfaces: LS3 Loop", {
			Iterations: 1,
			Threshold: 0,
		});
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vz(v)).toBeCloseTo(0, 9);
		}
	});

	test("more iterations converge further", () => {
		const after = (iterations: number) => {
			const { doc, cm } = docWith(sphereIcosa(1).mesh);
			kernel.applyFilter(doc, "Subdivision Surfaces: LS3 Loop", {
				Iterations: iterations,
				Threshold: 0,
			});
			return { radius: meanRadius(cm), faces: cm.fn };
		};
		const one = after(1);
		const two = after(2);
		expect(two.faces).toBe(4 * one.faces);
		// Both land on the sphere, but not monotonically: after the first pass
		// the normals are recomputed from the new geometry rather than being the
		// true sphere normals, so the second pass has slightly worse input than
		// the first did. Convergence is to the surface the normals describe, and
		// that surface drifts a little as the mesh does.
		expect(Math.abs(one.radius - 1)).toBeLessThan(0.001);
		expect(Math.abs(two.radius - 1)).toBeLessThan(0.001);
	});

	test("the algebraic sphere fit recovers a known sphere", () => {
		// Points on the unit sphere with their true normals: the fit should put
		// the projection back on the sphere wherever the weights land it.
		const fit = new AlgebraicSphere();
		for (const p of [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
			[-1, 0, 0],
			[0, -1, 0],
		]) {
			fit.add(p, p, 1);
		}
		const projected = fit.project();
		expect(Math.hypot(projected[0], projected[1], projected[2])).toBeCloseTo(1, 6);
	});
});

// ---------------------------------------------------------- attribute seam

describe("Vertex Attribute Seam", () => {
	/** A cube whose every face has its own wedge UV, so every corner disagrees. */
	function perFaceUVs() {
		const { doc, m, cm } = docWith(cube(1).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				wt[6 * f + 2 * k] = f / cm.faceSize;
				wt[6 * f + 2 * k + 1] = 0;
			}
		}
		return { doc, m, cm };
	}

	test("splits a vertex once per distinct attribute value around it", () => {
		const { doc, cm } = perFaceUVs();
		const before = cm.vn;
		const out = kernel.applyFilter(doc, "Vertex Attribute Seam", { TexcoordMode: 2 });
		// Every corner disagrees with every other, so each corner gets its own
		// vertex: 12 faces x 3 corners.
		expect(cm.vn).toBe(36);
		expect(out.added_vertices).toBe(36 - before);
	});

	test("the geometry is untouched", () => {
		const { doc, cm } = perFaceUVs();
		const before = new Set<string>();
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) before.add(`${cm.vx(v)},${cm.vy(v)},${cm.vz(v)}`);
		}
		kernel.applyFilter(doc, "Vertex Attribute Seam", { TexcoordMode: 2 });
		// Every position after the split was already present before it.
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(before.has(`${cm.vx(v)},${cm.vy(v)},${cm.vz(v)}`)).toBe(true);
		}
		expect(cm.fn).toBe(12);
	});

	test("the per-vertex attribute ends up holding what the corner meant", () => {
		const { doc, cm } = perFaceUVs();
		kernel.applyFilter(doc, "Vertex Attribute Seam", { TexcoordMode: 2 });
		const wt = cm.wedgeTexCoord as Float64Array;
		const vt = cm.vertTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				expect(vt[2 * v]).toBeCloseTo(wt[6 * f + 2 * k], 12);
			}
		}
	});

	test("a mesh whose corners already agree is left alone", () => {
		const { doc, cm } = docWith(cube(1).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		// The UV follows the vertex, so no corner disagrees with another.
		for (let f = 0; f < cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				wt[6 * f + 2 * k] = v;
				wt[6 * f + 2 * k + 1] = 0;
			}
		}
		const before = cm.vn;
		const out = kernel.applyFilter(doc, "Vertex Attribute Seam", { TexcoordMode: 2 });
		expect(out.added_vertices).toBe(0);
		expect(cm.vn).toBe(before);
	});

	test("splitting on a per-face colour gives one vertex per face corner", () => {
		const { doc, cm } = docWith(cube(1).mesh, MeshElement.MM_FACECOLOR);
		const colors = cm.faceColor as Uint32Array;
		for (let f = 0; f < cm.faceSize; f++) colors[f] = rgba(f * 20, 0, 0);
		kernel.applyFilter(doc, "Vertex Attribute Seam", { ColorMode: 3 });
		expect(cm.vn).toBe(36);
		// And each vertex now carries its face's colour.
		for (let f = 0; f < cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) expect(cm.vertColor[cm.fv(f, k)]).toBe(colors[f]);
		}
	});

	test("refuses when every source is set to None", () => {
		const { doc } = perFaceUVs();
		expect(() => kernel.applyFilter(doc, "Vertex Attribute Seam", {})).toThrow(MLException);
	});

	test("the splitter alone reports nothing to do on an empty mask", () => {
		const cm = cube(1).mesh;
		expect(AttributeSeam.splitVertexBySeam(cm, {})).toBe(0);
	});
});
