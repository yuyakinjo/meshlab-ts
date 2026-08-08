/**
 * `Make mesh developable`.
 *
 * The energy has a property that makes it testable without a golden file: it
 * is *exactly* zero on a surface that really is developable. A plane folded
 * along a crease, a grid rolled into a cylinder and a grid swept into a cone
 * are all developable, and every interior vertex of each one splits into two
 * runs of coplanar faces. A sphere cannot be flattened anywhere, so its energy
 * is large. That gives both a zero to hit and a direction to move in.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import {
	combinatorialEnergy,
	normalsAndAreas,
	orderedStars,
	removeSmallAngles,
	vertexBorderFromFaceAdj,
} from "../../src/vcg/complex/developability.ts";
import { faceFace } from "../../src/vcg/complex/update/topology.ts";
import {
	bowtieVertex,
	gridPlane,
	nonManifoldEdgeFan,
	sphereIcosa,
} from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();
const NAME = "Make mesh developable";

function docWith(cm: CMeshO) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** The mesh's developability energy, with everything it needs rebuilt first. */
function energyOf(cm: CMeshO): number {
	faceFace(cm);
	vertexBorderFromFaceAdj(cm);
	return combinatorialEnergy(cm, orderedStars(cm), normalsAndAreas(cm).normals);
}

/** A grid reshaped by a map applied to each vertex. */
function shaped(nu: number, nv: number, f: (x: number, y: number) => readonly number[]): CMeshO {
	const cm = gridPlane(nu, nv).mesh;
	for (let v = 0; v < cm.vertSize; v++) {
		const [x, y, z] = f(cm.vx(v), cm.vy(v));
		cm.setVert(v, x, y, z);
	}
	return cm;
}

const tent = () => shaped(8, 8, (x, y) => [x, y, Math.abs(x) * 0.6]);
const cylinder = () => shaped(8, 8, (x, y) => [Math.cos(x * 2), y, Math.sin(x * 2)]);
const cone = () =>
	shaped(8, 8, (x, y) => [(y + 2) * Math.cos(x * 2), y, (y + 2) * Math.sin(x * 2)]);

describe("the developability energy", () => {
	test("is exactly zero on surfaces that can be flattened", () => {
		// Not "small": zero. Every interior star of these three splits into two
		// runs whose faces are coplanar, which is the paper's definition.
		expect(energyOf(tent())).toBe(0);
		expect(energyOf(cylinder())).toBe(0);
		expect(energyOf(cone())).toBeLessThan(1e-20);
	});

	test("is large on a sphere, which cannot be flattened anywhere", () => {
		expect(energyOf(sphereIcosa(2).mesh)).toBeGreaterThan(1);
	});

	test("is zero wherever the star is too small to say anything", () => {
		// Three faces around a vertex are always readable as a hinge, so a
		// tetrahedron's every vertex is free however sharp the corner looks.
		const cm = gridPlane(3, 3).mesh;
		faceFace(cm);
		vertexBorderFromFaceAdj(cm);
		const stars = orderedStars(cm);
		const { normals } = normalsAndAreas(cm);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v) || stars[v].length > 3) continue;
			expect(combinatorialEnergy(cm, stars, normals)).toBe(0);
		}
	});

	test("the star of every vertex is its whole set of incident faces, once each", () => {
		// The ring walk is the piece most likely to be quietly wrong: a fan cut
		// short still produces a plausible number, just from fewer normals.
		for (const cm of [sphereIcosa(2).mesh, gridPlane(5, 4).mesh, cylinder()]) {
			faceFace(cm);
			const stars = orderedStars(cm);
			const incident: number[][] = Array.from({ length: cm.vertSize }, () => []);
			for (let f = 0; f < cm.faceSize; f++) {
				if (cm.isFaceD(f)) continue;
				for (let k = 0; k < 3; k++) incident[cm.fv(f, k)].push(f);
			}
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				expect(new Set(stars[v]).size, `vertex ${v}`).toBe(stars[v].length);
				expect([...stars[v]].sort((a, b) => a - b)).toEqual(incident[v].sort((a, b) => a - b));
			}
		}
	});

	test("consecutive faces in a star share an edge", () => {
		// What "ordered" has to mean, checked rather than assumed.
		const cm = sphereIcosa(2).mesh;
		faceFace(cm);
		const stars = orderedStars(cm);
		for (let v = 0; v < cm.vertSize; v++) {
			const star = stars[v];
			for (let i = 0; i + 1 < star.length; i++) {
				const a = new Set([0, 1, 2].map((k) => cm.fv(star[i], k)));
				const shared = [0, 1, 2].map((k) => cm.fv(star[i + 1], k)).filter((w) => a.has(w));
				expect(shared.length, `vertex ${v}, step ${i}`).toBe(2);
			}
		}
	});
});

describe(NAME, () => {
	test("leaves a developable surface where it found it", () => {
		const { doc, cm } = docWith(cylinder());
		const before = Array.from({ length: cm.vertSize }, (_, v) => [cm.vx(v), cm.vy(v), cm.vz(v)]);
		const out = kernel.applyFilter(doc, NAME, { MaxFunEvals: 50 });
		// Zero energy means zero gradient, so it should converge immediately.
		expect(out.converged).toBe(true);
		for (let v = 0; v < cm.vertSize; v++) {
			expect(cm.vx(v)).toBeCloseTo(before[v][0], 9);
			expect(cm.vy(v)).toBeCloseTo(before[v][1], 9);
			expect(cm.vz(v)).toBeCloseTo(before[v][2], 9);
		}
	});

	test("lowers the energy of a sphere", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const before = energyOf(cm);
		const out = kernel.applyFilter(doc, NAME, {
			MaxFunEvals: 60,
			EdgeFlips: false,
			EdgeCollapses: false,
		});
		expect(energyOf(cm)).toBeLessThan(before);
		expect(out.energy as number).toBeLessThan(before);
	});

	test("the fixed-step method also descends", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = energyOf(cm);
		kernel.applyFilter(doc, NAME, {
			OptMethod: 0,
			MaxFunEvals: 40,
			StepSize: 1e-4,
			EdgeFlips: false,
			EdgeCollapses: false,
		});
		expect(energyOf(cm)).toBeLessThan(before);
	});

	test("returns the mesh to the scale and place it came in at", () => {
		// The optimization runs on a unit-diagonal copy at the origin, because
		// the step size is an absolute length. A mesh that came back 40x smaller
		// would still pass an energy test.
		const { doc, cm } = docWith(shaped(6, 6, (x, y) => [x * 40 + 100, y * 40 - 7, 3]));
		const before = { x: cm.vx(0), y: cm.vy(0), z: cm.vz(0) };
		kernel.applyFilter(doc, NAME, { MaxFunEvals: 5 });
		expect(cm.vx(0)).toBeCloseTo(before.x, 6);
		expect(cm.vy(0)).toBeCloseTo(before.y, 6);
		expect(cm.vz(0)).toBeCloseTo(before.z, 6);
	});

	test("keeps every vertex and face when remeshing is off", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, NAME, { MaxFunEvals: 30, EdgeFlips: false, EdgeCollapses: false });
		expect({ vn: cm.vn, fn: cm.fn }).toEqual(before);
	});

	test("refuses a non-manifold mesh, by edge and by vertex", () => {
		expect(() => kernel.applyFilter(docWith(nonManifoldEdgeFan(1).mesh).doc, NAME)).toThrow(
			/non manifold edges/,
		);
		expect(() => kernel.applyFilter(docWith(bowtieVertex().mesh).doc, NAME)).toThrow(
			/non manifold vert/,
		);
	});

	test("is registered as MeshLab registers it", () => {
		const action = kernel.pluginManager.filterAction(NAME);
		expect(action).toBeDefined();
		expect(action?.plugin.pluginName()).toBe("FilterDevelopability");
		expect(action?.pythonName).toBe("apply_coord_developability_of_mesh");
	});

	test("rejects a parameter it does not have", () => {
		expect(() =>
			kernel.applyFilter(docWith(gridPlane(3, 3).mesh).doc, NAME, { Steps: 5 }),
		).toThrow();
	});
});

describe("the sliver-removal pass", () => {
	/** A grid with one vertex dragged almost onto its neighbour. */
	function withSliver(): CMeshO {
		const cm = gridPlane(5, 5).mesh;
		// Pull an interior vertex nearly onto the one beside it, which leaves
		// the faces between them with two very sharp corners each.
		cm.setVert(12, cm.vx(11) + 1e-3, cm.vy(11), cm.vz(11));
		return cm;
	}

	test("changes the mesh when a corner is sharper than the threshold", () => {
		const cm = withSliver();
		faceFace(cm);
		expect(removeSmallAngles(cm, true, true, (18 * Math.PI) / 180)).toBe(true);
	});

	test("leaves a well-shaped mesh alone", () => {
		const cm = sphereIcosa(2).mesh;
		faceFace(cm);
		expect(removeSmallAngles(cm, true, true, (18 * Math.PI) / 180)).toBe(false);
	});

	test("does nothing when both operations are switched off", () => {
		const cm = withSliver();
		faceFace(cm);
		const before = { vn: cm.vn, fn: cm.fn };
		expect(removeSmallAngles(cm, false, false, (18 * Math.PI) / 180)).toBe(false);
		expect({ vn: cm.vn, fn: cm.fn }).toEqual(before);
	});

	test("the filter reports the remeshing it did", () => {
		const { doc } = docWith(withSliver());
		const out = kernel.applyFilter(doc, NAME, { MaxFunEvals: 10 });
		expect(out.remeshing_rounds as number).toBeGreaterThan(0);
	});
});
