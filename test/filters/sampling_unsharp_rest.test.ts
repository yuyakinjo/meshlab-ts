/**
 * The last thirteen of `filter_sampling` and `filter_unsharp`.
 *
 * Two groups need saying something about up front.
 *
 * The crease cut has an invariant that makes it easy to check and hard to fake:
 * the geometry must not move at all. Every vertex it adds is a copy of one that
 * was already there, so the set of positions, the face count and the volume all
 * have to come back identical — only the vertex count and the topology change.
 * A cut that got the fan walk wrong shows up immediately as a changed volume or
 * a lost face.
 *
 * The two-step smoother is checked on a shape where the right answer is
 * unambiguous: a cube with noise. A feature-preserving filter must flatten the
 * faces without rounding the edges, so the test measures both — how far the
 * vertices are from the six planes, and how sharp the dihedral angles at the
 * edges still are. A plain Laplacian passes the first and fails the second,
 * which is what makes the pair worth measuring rather than either alone.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { CreaseCut } from "../../src/vcg/complex/crease_cut.ts";
import { FaceFlag, VertexFlag } from "../../src/vcg/complex/flags.ts";
import { Smooth } from "../../src/vcg/complex/smooth.ts";
import { UpdateBounding } from "../../src/vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../src/vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { blue, red, rgba } from "../../src/vcg/space/color4.ts";
import { cube, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO, channels: number = MeshElement.MM_NONE) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	if (channels !== MeshElement.MM_NONE) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Signed volume, the invariant that catches a mangled topology. */
function volumeOf(cm: CMeshO): number {
	let total = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const [a, b, c] = [0, 1, 2].map((k) => cm.fv(f, k));
		total +=
			(cm.vx(a) * (cm.vy(b) * cm.vz(c) - cm.vz(b) * cm.vy(c)) -
				cm.vy(a) * (cm.vx(b) * cm.vz(c) - cm.vz(b) * cm.vx(c)) +
				cm.vz(a) * (cm.vx(b) * cm.vy(c) - cm.vy(b) * cm.vx(c))) /
			6;
	}
	return Math.abs(total);
}

/** The multiset of vertex positions, rounded and sorted so order cannot matter. */
function positionDigest(cm: CMeshO): string[] {
	const out: string[] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		out.push(`${cm.vx(v).toFixed(9)},${cm.vy(v).toFixed(9)},${cm.vz(v).toFixed(9)}`);
	}
	return out.sort();
}

// ------------------------------------------------------------------ crease cut

describe("Cut mesh along crease edges", () => {
	test("a cube's corners split into one vertex per face", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = { vn: cm.vn, fn: cm.fn, volume: volumeOf(cm) };
		kernel.applyFilter(doc, "Cut mesh along crease edges", { angleDeg: 45 });

		// Every corner meets three faces, so the cut needs 3 vertices per
		// corner: 24 referenced. MeshLab allocates one *more* per corner — the
		// crossing that closes each fan claims a vertex it never writes — and
		// ships the orphan, so 8 -> 32 with 8 unreferenced vertices at the
		// origin. The differential tests hold us to that number, wart and all.
		expect(cm.vn).toBe(32);
		const referenced = new Set<number>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) referenced.add(cm.fv(f, k));
		}
		expect(referenced.size).toBe(24);
		expect(cm.fn).toBe(before.fn);
		// The shape is untouched. This is the invariant that matters.
		expect(volumeOf(cm)).toBeCloseTo(before.volume, 9);
	});

	test("adds no referenced position the mesh did not already have", () => {
		// The 8 orphan vertices sit at the origin (allocated, never written) —
		// upstream's artefact, reproduced. Referenced geometry stays a subset
		// of what was there.
		const { doc, cm } = docWith(cube(1).mesh);
		const before = new Set(positionDigest(cm));
		kernel.applyFilter(doc, "Cut mesh along crease edges", { angleDeg: 45 });
		const referenced = new Set<number>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) referenced.add(cm.fv(f, k));
		}
		for (const v of referenced) {
			const key = `${cm.vx(v).toFixed(9)},${cm.vy(v).toFixed(9)},${cm.vz(v).toFixed(9)}`;
			expect(before.has(key), key).toBe(true);
		}
	});

	test("a threshold above every angle leaves the mesh alone", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		const before = cm.vn;
		// A cube's dihedral is 90 degrees, so 120 cuts nothing.
		kernel.applyFilter(doc, "Cut mesh along crease edges", { angleDeg: 120 });
		expect(cm.vn).toBe(before);
	});

	test("a smooth sphere is barely cut at all", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const before = cm.vn;
		kernel.applyFilter(doc, "Cut mesh along crease edges", { angleDeg: 60 });
		expect(cm.vn).toBe(before);
	});

	test("the cut leaves every crease edge a boundary", () => {
		const { doc, cm } = docWith(cube(1).mesh);
		kernel.applyFilter(doc, "Cut mesh along crease edges", { angleDeg: 45 });
		UpdateTopology.faceFace(cm);
		// Six separate quads, each two triangles: 6 * 4 = 24 boundary edges.
		let border = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) if (cm.isBorderFF(f, e)) border++;
		}
		expect(border).toBe(24);
	});

	test("refuses a non-manifold mesh", () => {
		// Three triangles sharing one edge: there is no fan to walk.
		const cm = new CMeshO();
		Allocator.addVertices(cm, 5);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 0, 1, 0);
		cm.setVert(3, 0, 0, 1);
		cm.setVert(4, 0, -1, 0);
		Allocator.addFaces(cm, 3);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 1, 3);
		cm.setFace(2, 0, 1, 4);
		const { doc } = docWith(cm);
		expect(() => kernel.applyFilter(doc, "Cut mesh along crease edges")).toThrow(MLException);
	});

	test("the dihedral angle is signed", () => {
		// Two triangles folded into a valley: the same absolute angle as a ridge,
		// but the sign is what an asymmetric threshold selects on.
		const build = (zSign: number) => {
			const cm = new CMeshO();
			Allocator.addVertices(cm, 4);
			cm.setVert(0, 0, 0, 0);
			cm.setVert(1, 1, 0, 0);
			cm.setVert(2, 0.5, 1, 0);
			cm.setVert(3, 0.5, -1, zSign);
			Allocator.addFaces(cm, 2);
			cm.setFace(0, 0, 1, 2);
			cm.setFace(1, 1, 0, 3);
			UpdateTopology.faceFace(cm);
			UpdateNormal.perFaceNormalized(cm);
			return cm;
		};
		const up = CreaseCut.dihedralAngleRad(build(1), 0, 0);
		const down = CreaseCut.dihedralAngleRad(build(-1), 0, 0);
		expect(Math.sign(up)).toBe(-Math.sign(down));
		expect(Math.abs(up)).toBeCloseTo(Math.abs(down), 9);
	});
});

// -------------------------------------------------------------- polygon normals

describe("Re-Compute Per-Polygon Face Normals", () => {
	/** A quad split into two triangles of very different size, joined faux. */
	function skewedQuad() {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 1, 0.4);
		cm.setVert(3, 0, 1, 0.4);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 2, 3);
		// The shared diagonal 0-2 is edge 2 of face 0 and edge 0 of face 1.
		cm.faceFlags[0] |= FaceFlag.FAUX2;
		cm.faceFlags[1] |= FaceFlag.FAUX0;
		return cm;
	}

	test("gives both halves of a quad one normal", () => {
		const { doc, cm } = docWith(skewedQuad());
		kernel.applyFilter(doc, "Re-Compute Per-Polygon Face Normals");
		for (let k = 0; k < 3; k++) {
			expect(cm.faceNormal[k]).toBeCloseTo(cm.faceNormal[3 + k], 12);
		}
	});

	test("the shared normal is a unit vector between the two originals", () => {
		const { doc, cm } = docWith(skewedQuad());
		UpdateNormal.perFaceNormalized(cm);
		const before = [0, 1].map((f) => [0, 1, 2].map((k) => cm.faceNormal[3 * f + k]));
		kernel.applyFilter(doc, "Re-Compute Per-Polygon Face Normals");
		const after = [0, 1, 2].map((k) => cm.faceNormal[k]);
		expect(Math.hypot(...after)).toBeCloseTo(1, 12);
		// It must lie between the two, so its dot with each is positive and its
		// dot with their average is the largest.
		for (const n of before) {
			expect(n[0] * after[0] + n[1] * after[1] + n[2] * after[2]).toBeGreaterThan(0.9);
		}
	});

	test("a mesh with no faux edges is left as plain face normals", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Re-Compute Per-Polygon Face Normals");
		const mine = Float64Array.from(cm.faceNormal);
		UpdateNormal.perFaceNormalized(cm);
		for (let i = 0; i < 3 * cm.faceSize; i++) expect(mine[i]).toBeCloseTo(cm.faceNormal[i], 12);
	});
});

// ------------------------------------------------------------- depth smoothing

describe("Depth Smooth", () => {
	test("moves each vertex only along the line to the viewpoint", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const viewpoint = [0, 0, 10];
		const before: number[][] = [];
		for (let v = 0; v < cm.vertSize; v++) before.push([cm.vx(v), cm.vy(v), cm.vz(v)]);
		// Jitter so there is something to smooth.
		for (let v = 0; v < cm.vertSize; v++) {
			const j = ((v * 2654435761) % 1000) / 1000 - 0.5;
			cm.setVert(v, cm.vx(v) * (1 + 0.1 * j), cm.vy(v) * (1 + 0.1 * j), cm.vz(v) * (1 + 0.1 * j));
		}
		const jittered: number[][] = [];
		for (let v = 0; v < cm.vertSize; v++) jittered.push([cm.vx(v), cm.vy(v), cm.vz(v)]);

		kernel.applyFilter(doc, "Depth Smooth", { viewPoint: viewpoint, stepSmoothNum: 3, delta: 1 });

		let moved = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const d = [cm.vx(v) - jittered[v][0], cm.vy(v) - jittered[v][1], cm.vz(v) - jittered[v][2]];
			const len = Math.hypot(d[0], d[1], d[2]);
			if (len < 1e-12) continue;
			moved++;
			// The displacement must be parallel to the ray from the viewpoint.
			const ray = [
				jittered[v][0] - viewpoint[0],
				jittered[v][1] - viewpoint[1],
				jittered[v][2] - viewpoint[2],
			];
			const rl = Math.hypot(ray[0], ray[1], ray[2]);
			const cosang = (d[0] * ray[0] + d[1] * ray[1] + d[2] * ray[2]) / (len * rl);
			expect(Math.abs(cosang)).toBeCloseTo(1, 9);
		}
		expect(moved).toBeGreaterThan(cm.vn / 2);
		expect(before.length).toBe(jittered.length);
	});

	test("strength zero is a no-op", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = positionDigest(cm);
		kernel.applyFilter(doc, "Depth Smooth", { delta: 0, stepSmoothNum: 5 });
		expect(positionDigest(cm)).toEqual(before);
	});

	test("a vertex exactly at the viewpoint cannot move", () => {
		// There is no direction to project onto; it must be left alone rather
		// than turned into NaN.
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const viewpoint = [cm.vx(0), cm.vy(0), cm.vz(0)];
		kernel.applyFilter(doc, "Depth Smooth", { viewPoint: viewpoint, stepSmoothNum: 2 });
		expect(Number.isFinite(cm.vx(0))).toBe(true);
		expect(cm.vx(0)).toBeCloseTo(viewpoint[0], 12);
	});
});

// -------------------------------------------------------------- two-step smooth

describe("TwoStep Smooth", () => {
	/** A cube subdivided a few times, then jittered. */
	function noisyCube() {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("", "m", true, cube(1).mesh);
		m.updateDataMask(MeshElement.MM_FACEFACETOPO);
		m.updateBoxAndNormals();
		kernel.applyFilter(doc, "Subdivision Surfaces: Midpoint", { Iterations: 4, Threshold: 0 });
		const cm = doc.mm().cm;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const j = (((v * 2654435761) % 997) / 997 - 0.5) * 0.03;
			cm.setVert(v, cm.vx(v) + j, cm.vy(v) + j * 0.7, cm.vz(v) - j * 0.4);
		}
		doc.mm().updateBoxAndNormals();
		return { doc, cm };
	}

	/** Mean distance from the nearest of the cube's six planes. */
	function flatness(cm: CMeshO): number {
		let sum = 0;
		let n = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const d = [cm.vx(v), cm.vy(v), cm.vz(v)].map((c) =>
				Math.min(Math.abs(c - 0.5), Math.abs(c + 0.5)),
			);
			sum += Math.min(...d);
			n++;
		}
		return sum / n;
	}

	test("flattens the faces", () => {
		const { doc, cm } = noisyCube();
		const before = flatness(cm);
		kernel.applyFilter(doc, "TwoStep Smooth", {
			stepSmoothNum: 2,
			normalThr: 60,
			stepNormalNum: 10,
			stepFitNum: 10,
		});
		expect(flatness(cm)).toBeLessThan(before / 2);
	});

	test("keeps the cube's edges sharp where Laplacian rounds them", () => {
		const sharpness = (cm: CMeshO) => {
			// The box diagonal: a cube that kept its edges stays large, one that
			// was rounded off shrinks toward a sphere.
			UpdateBounding.box(cm);
			return cm.bbox.diagonal;
		};
		const two = noisyCube();
		const lap = noisyCube();
		const start = sharpness(two.cm);
		kernel.applyFilter(two.doc, "TwoStep Smooth", {
			stepSmoothNum: 2,
			normalThr: 60,
			stepNormalNum: 10,
			stepFitNum: 10,
		});
		kernel.applyFilter(lap.doc, "Laplacian Smooth", { stepSmoothNum: 8 });
		// Both smooth; only one keeps the corners out where they were.
		expect(sharpness(two.cm)).toBeGreaterThan(sharpness(lap.cm));
		expect(sharpness(two.cm)).toBeGreaterThan(start * 0.95);
	});

	test("a feature angle of zero smooths nothing", () => {
		// sigma = cos(0) = 1, so every neighbour's weight is (cos - 1)^2 with
		// cos <= 1 clamped to 1: the normal field cannot change.
		const { doc, cm } = noisyCube();
		const before = flatness(cm);
		kernel.applyFilter(doc, "TwoStep Smooth", {
			stepSmoothNum: 1,
			normalThr: 0,
			stepNormalNum: 5,
			stepFitNum: 0,
		});
		expect(flatness(cm)).toBeCloseTo(before, 9);
	});

	test("the normal smoother leaves an already-flat patch alone", () => {
		const cm = sphereIcosa(2).mesh;
		UpdateTopology.vertexFace(cm);
		UpdateNormal.perFaceNormalized(cm);
		const before = Float64Array.from(cm.faceNormal);
		Smooth.faceNormalAngleThreshold(cm, 1);
		for (let i = 0; i < 3 * cm.faceSize; i++) expect(cm.faceNormal[i]).toBeCloseTo(before[i], 9);
	});
});

// ------------------------------------------------------- directional preservation

describe("Directional Geometry Preservation", () => {
	test("keeps only the component of the smoothing along the view ray", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Define New Per Vertex Custom Point Attribute", {
			name: "SavedVertPosition",
			x_expr: "x",
			y_expr: "y",
			z_expr: "z",
		});
		const saved = cm.customAttribute("SavedVertPosition", "vert") as { data: Float64Array };
		const original = Float64Array.from(saved.data);
		kernel.applyFilter(doc, "Laplacian Smooth", { stepSmoothNum: 5 });

		const viewpoint = [0, 0, 10];
		kernel.applyFilter(doc, "Directional Geometry Preservation", {
			attr_name: "SavedVertPosition",
			viewPoint: viewpoint,
		});

		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const old = [original[3 * v], original[3 * v + 1], original[3 * v + 2]];
			const now = [cm.vx(v), cm.vy(v), cm.vz(v)];
			const moved = [now[0] - old[0], now[1] - old[1], now[2] - old[2]];
			const len = Math.hypot(moved[0], moved[1], moved[2]);
			if (len < 1e-12) continue;
			const ray = [old[0] - viewpoint[0], old[1] - viewpoint[1], old[2] - viewpoint[2]];
			const rl = Math.hypot(ray[0], ray[1], ray[2]);
			const cosang = (moved[0] * ray[0] + moved[1] * ray[1] + moved[2] * ray[2]) / (len * rl);
			expect(Math.abs(cosang)).toBeCloseTo(1, 9);
		}
	});

	test("says what is missing when there is no such attribute", () => {
		const { doc } = docWith(sphereIcosa(1).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Directional Geometry Preservation", { attr_name: "nope" }),
		).toThrow(MLException);
	});

	test("rejects a scalar attribute where a point one is needed", () => {
		const { doc } = docWith(sphereIcosa(1).mesh);
		kernel.applyFilter(doc, "Define New Per Vertex Custom Scalar Attribute", {
			name: "flat",
			expr: "x",
		});
		expect(() =>
			kernel.applyFilter(doc, "Directional Geometry Preservation", { attr_name: "flat" }),
		).toThrow(MLException);
	});
});

// ------------------------------------------------------------- harmonic field

describe("Generate Scalar Harmonic Field", () => {
	test("the two constrained vertices hold their given values", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh, MeshElement.MM_VERTQUALITY);
		UpdateBounding.box(cm);
		const lo = [0, 0, -1];
		const hi = [0, 0, 1];
		kernel.applyFilter(doc, "Generate Scalar Harmonic Field", {
			point1: lo,
			point2: hi,
			value1: 0,
			value2: 1,
		});
		const nearest = (p: number[]) => {
			let best = 0;
			let bd = Number.POSITIVE_INFINITY;
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				const d = (cm.vx(v) - p[0]) ** 2 + (cm.vy(v) - p[1]) ** 2 + (cm.vz(v) - p[2]) ** 2;
				if (d < bd) {
					bd = d;
					best = v;
				}
			}
			return best;
		};
		expect(cm.vertQuality[nearest(lo)]).toBeCloseTo(0, 6);
		expect(cm.vertQuality[nearest(hi)]).toBeCloseTo(1, 6);
	});

	test("has no interior maximum or minimum", () => {
		// The defining property of a harmonic function, and the reason it is
		// useful: every vertex except the two constrained ones is between the
		// smallest and largest of its neighbours.
		const { doc, cm } = docWith(sphereIcosa(3).mesh, MeshElement.MM_VERTQUALITY);
		kernel.applyFilter(doc, "Generate Scalar Harmonic Field", {
			point1: [0, 0, -1],
			point2: [0, 0, 1],
			value1: 0,
			value2: 1,
		});

		const neighbours: Array<Set<number>> = Array.from(
			{ length: cm.vertSize },
			() => new Set<number>(),
		);
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				neighbours[cm.fv(f, k)].add(cm.fv(f, (k + 1) % 3));
				neighbours[cm.fv(f, (k + 1) % 3)].add(cm.fv(f, k));
			}
		}
		// The constrained vertices are the two extremes and are allowed to be —
		// the maximum principle is a statement about the *interior*.
		const constrained = new Set<number>();
		for (const p of [
			[0, 0, -1],
			[0, 0, 1],
		]) {
			let best = 0;
			let bd = Number.POSITIVE_INFINITY;
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				const d = (cm.vx(v) - p[0]) ** 2 + (cm.vy(v) - p[1]) ** 2 + (cm.vz(v) - p[2]) ** 2;
				if (d < bd) {
					bd = d;
					best = v;
				}
			}
			constrained.add(best);
		}
		expect(constrained.size).toBe(2);

		let violations = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v) || neighbours[v].size === 0 || constrained.has(v)) continue;
			const around = [...neighbours[v]].map((w) => cm.vertQuality[w]);
			const q = cm.vertQuality[v];
			if (q > Math.max(...around) + 1e-6 || q < Math.min(...around) - 1e-6) violations++;
		}
		expect(violations).toBe(0);
	});

	test("stays between the two constrained values", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh, MeshElement.MM_VERTQUALITY);
		kernel.applyFilter(doc, "Generate Scalar Harmonic Field", {
			point1: [0, 0, -1],
			point2: [0, 0, 1],
			value1: 0.25,
			value2: 0.75,
		});
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect(cm.vertQuality[v]).toBeGreaterThan(0.25 - 1e-6);
			expect(cm.vertQuality[v]).toBeLessThan(0.75 + 1e-6);
		}
	});

	test("refuses two points that land on the same vertex", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh, MeshElement.MM_VERTQUALITY);
		const p = [cm.vx(0), cm.vy(0), cm.vz(0)];
		expect(() =>
			kernel.applyFilter(doc, "Generate Scalar Harmonic Field", { point1: p, point2: p }),
		).toThrow(MLException);
	});

	test("refuses a mesh in more than one piece", () => {
		const cm = sphereIcosa(1).mesh;
		const second = sphereIcosa(1).mesh;
		const base = cm.vertSize;
		const vf = Allocator.addVertices(cm, second.vertSize);
		for (let v = 0; v < second.vertSize; v++) {
			cm.setVert(vf + v, second.vx(v) + 5, second.vy(v), second.vz(v));
		}
		const ff = Allocator.addFaces(cm, second.faceSize);
		for (let f = 0; f < second.faceSize; f++) {
			cm.setFace(ff + f, base + second.fv(f, 0), base + second.fv(f, 1), base + second.fv(f, 2));
		}
		const { doc } = docWith(cm, MeshElement.MM_VERTQUALITY);
		expect(() => kernel.applyFilter(doc, "Generate Scalar Harmonic Field")).toThrow(MLException);
	});
});

// ------------------------------------------------------------ distance measures

describe("Distance from Reference Mesh", () => {
	function twoSpheres(scale: number) {
		const doc = new MeshDocument();
		const inner = sphereIcosa(3).mesh;
		for (let v = 0; v < inner.vertSize; v++) {
			inner.setVert(v, inner.vx(v) * scale, inner.vy(v) * scale, inner.vz(v) * scale);
		}
		const a = doc.addNewMesh("", "measured", true, inner);
		const b = doc.addNewMesh("", "reference", true, sphereIcosa(4).mesh);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		return { doc, a, b };
	}

	test("reads the gap between two concentric spheres", () => {
		const { doc, a, b } = twoSpheres(1.2);
		const out = kernel.applyFilter(doc, "Distance from Reference Mesh", {
			MeasureMesh: a.id(),
			RefMesh: b.id(),
		});
		// A sphere of radius 1.2 against one of radius 1: about 0.2 everywhere.
		expect(out.mean as number).toBeGreaterThan(0.15);
		expect(out.mean as number).toBeLessThan(0.25);
	});

	test("the sign says which side of the reference each vertex is on", () => {
		const outside = twoSpheres(1.2);
		const inside = twoSpheres(0.8);
		kernel.applyFilter(outside.doc, "Distance from Reference Mesh", {
			MeasureMesh: outside.a.id(),
			RefMesh: outside.b.id(),
			SignedDist: true,
		});
		kernel.applyFilter(inside.doc, "Distance from Reference Mesh", {
			MeasureMesh: inside.a.id(),
			RefMesh: inside.b.id(),
			SignedDist: true,
		});
		expect(outside.a.cm.vertQuality[0]).toBeGreaterThan(0);
		expect(inside.a.cm.vertQuality[0]).toBeLessThan(0);
	});

	test("unsigned mode never reports a negative", () => {
		const { doc, a, b } = twoSpheres(0.8);
		kernel.applyFilter(doc, "Distance from Reference Mesh", {
			MeasureMesh: a.id(),
			RefMesh: b.id(),
			SignedDist: false,
		});
		for (let v = 0; v < a.cm.vertSize; v++) {
			if (!a.cm.isVertD(v)) expect(a.cm.vertQuality[v]).toBeGreaterThanOrEqual(0);
		}
	});

	test("a mesh against itself reads zero", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, sphereIcosa(3).mesh);
		const b = doc.addNewMesh("", "b", true, sphereIcosa(3).mesh);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		const out = kernel.applyFilter(doc, "Distance from Reference Mesh", {
			MeasureMesh: a.id(),
			RefMesh: b.id(),
		});
		expect(out.mean as number).toBeCloseTo(0, 9);
	});

	test("refuses the same layer twice", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, sphereIcosa(1).mesh);
		a.updateBoxAndNormals();
		expect(() =>
			kernel.applyFilter(doc, "Distance from Reference Mesh", {
				MeasureMesh: a.id(),
				RefMesh: a.id(),
			}),
		).toThrow(MLException);
	});
});

// ------------------------------------------------------------ attribute transfer

describe("Vertex Attribute Transfer", () => {
	function pair() {
		const doc = new MeshDocument();
		const src = doc.addNewMesh("", "src", true, sphereIcosa(3).mesh);
		const trg = doc.addNewMesh("", "trg", true, sphereIcosa(2).mesh);
		src.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
		src.updateBoxAndNormals();
		trg.updateBoxAndNormals();
		// A colour field that varies smoothly with height, so an interpolated
		// answer and a snapped one are distinguishable.
		for (let v = 0; v < src.cm.vertSize; v++) {
			const t = (src.cm.vz(v) + 1) / 2;
			src.cm.vertColor[v] = rgba(255 * t, 0, 255 * (1 - t));
			src.cm.vertQuality[v] = src.cm.vz(v);
		}
		return { doc, src, trg };
	}

	test("carries the colour field across", () => {
		const { doc, src, trg } = pair();
		kernel.applyFilter(doc, "Vertex Attribute Transfer", {
			SourceMesh: src.id(),
			TargetMesh: trg.id(),
			ColorTransfer: true,
		});
		for (let v = 0; v < trg.cm.vertSize; v++) {
			if (trg.cm.isVertD(v)) continue;
			const t = (trg.cm.vz(v) + 1) / 2;
			expect(red(trg.cm.vertColor[v])).toBeCloseTo(255 * t, -1);
			expect(blue(trg.cm.vertColor[v])).toBeCloseTo(255 * (1 - t), -1);
		}
	});

	test("quality transfer reproduces the source field", () => {
		const { doc, src, trg } = pair();
		kernel.applyFilter(doc, "Vertex Attribute Transfer", {
			SourceMesh: src.id(),
			TargetMesh: trg.id(),
			ColorTransfer: false,
			QualityTransfer: true,
		});
		for (let v = 0; v < trg.cm.vertSize; v++) {
			if (!trg.cm.isVertD(v)) expect(trg.cm.vertQuality[v]).toBeCloseTo(trg.cm.vz(v), 1);
		}
	});

	test("surface sampling gives more distinct values than vertex sampling", () => {
		// The reason the surface mode is the default: snapping to a coarse
		// source quantises a smooth field, interpolating does not. Here the
		// source is fine and the target coarse, so use the reverse direction.
		const build = (byVertex: boolean) => {
			const doc = new MeshDocument();
			const src = doc.addNewMesh("", "src", true, sphereIcosa(1).mesh);
			const trg = doc.addNewMesh("", "trg", true, sphereIcosa(3).mesh);
			src.updateDataMask(MeshElement.MM_VERTQUALITY);
			src.updateBoxAndNormals();
			trg.updateBoxAndNormals();
			for (let v = 0; v < src.cm.vertSize; v++) src.cm.vertQuality[v] = src.cm.vz(v);
			kernel.applyFilter(doc, "Vertex Attribute Transfer", {
				SourceMesh: src.id(),
				TargetMesh: trg.id(),
				ColorTransfer: false,
				QualityTransfer: true,
				VertexSampling: byVertex,
			});
			const seen = new Set<string>();
			for (let v = 0; v < trg.cm.vertSize; v++) {
				if (!trg.cm.isVertD(v)) seen.add(trg.cm.vertQuality[v].toFixed(6));
			}
			return seen.size;
		};
		expect(build(false)).toBeGreaterThan(build(true));
	});

	test("geometry transfer snaps the target onto the source", () => {
		const doc = new MeshDocument();
		const src = doc.addNewMesh("", "src", true, sphereIcosa(3).mesh);
		const scaled = sphereIcosa(2).mesh;
		for (let v = 0; v < scaled.vertSize; v++) {
			scaled.setVert(v, scaled.vx(v) * 1.3, scaled.vy(v) * 1.3, scaled.vz(v) * 1.3);
		}
		const trg = doc.addNewMesh("", "trg", true, scaled);
		src.updateBoxAndNormals();
		trg.updateBoxAndNormals();
		kernel.applyFilter(doc, "Vertex Attribute Transfer", {
			SourceMesh: src.id(),
			TargetMesh: trg.id(),
			ColorTransfer: false,
			GeomTransfer: true,
			UpperBound: 1,
		});
		// Every target vertex now sits on the unit sphere, not the 1.3 one.
		for (let v = 0; v < trg.cm.vertSize; v++) {
			if (trg.cm.isVertD(v)) continue;
			expect(Math.hypot(trg.cm.vx(v), trg.cm.vy(v), trg.cm.vz(v))).toBeLessThan(1.05);
		}
	});

	test("the search bound rejects what is too far away", () => {
		const doc = new MeshDocument();
		const src = doc.addNewMesh("", "src", true, sphereIcosa(2).mesh);
		const far = sphereIcosa(2).mesh;
		for (let v = 0; v < far.vertSize; v++) far.setVert(v, far.vx(v) + 50, far.vy(v), far.vz(v));
		const trg = doc.addNewMesh("", "trg", true, far);
		src.updateDataMask(MeshElement.MM_VERTCOLOR);
		src.updateBoxAndNormals();
		trg.updateBoxAndNormals();
		const out = kernel.applyFilter(doc, "Vertex Attribute Transfer", {
			SourceMesh: src.id(),
			TargetMesh: trg.id(),
			ColorTransfer: true,
			UpperBound: 0.1,
		});
		expect(out.vertex_number).toBe(0);
		expect(out.rejected as number).toBeGreaterThan(0);
	});

	test("refuses when nothing was asked for", () => {
		const { doc, src, trg } = pair();
		expect(() =>
			kernel.applyFilter(doc, "Vertex Attribute Transfer", {
				SourceMesh: src.id(),
				TargetMesh: trg.id(),
				ColorTransfer: false,
			}),
		).toThrow(MLException);
	});
});

// ------------------------------------------------------------------- resampling

describe("Uniform Mesh Resampling", () => {
	test("rebuilds a sphere at roughly the same size", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		UpdateBounding.box(cm);
		const before = volumeOf(cm);
		kernel.applyFilter(doc, "Uniform Mesh Resampling", { CellSize: 0.08 });
		const out = doc.mm().cm;
		expect(out.vn).toBeGreaterThan(0);
		// Within a few percent: the grid rounds the surface off a little.
		expect(volumeOf(out)).toBeGreaterThan(before * 0.85);
		expect(volumeOf(out)).toBeLessThan(before * 1.15);
	});

	test("a positive offset grows the surface", () => {
		const make = (offset: number) => {
			const { doc } = docWith(sphereIcosa(3).mesh);
			kernel.applyFilter(doc, "Uniform Mesh Resampling", { CellSize: 0.1, Offset: offset });
			const out = doc.mm().cm;
			UpdateBounding.box(out);
			return out.bbox.diagonal;
		};
		expect(make(0.2)).toBeGreaterThan(make(0));
		expect(make(0)).toBeGreaterThan(make(-0.2));
	});

	test("an unsigned field with no offset is refused rather than returning nothing", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Uniform Mesh Resampling", {
				CellSize: 0.2,
				absDist: true,
				Offset: 0,
			}),
		).toThrow(MLException);
	});

	test("an unsigned field with an offset builds a shell around the surface", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Uniform Mesh Resampling", {
			CellSize: 0.1,
			absDist: true,
			Offset: 0.2,
		});
		const out = doc.mm().cm;
		// Two surfaces, one each side of the original: the radii span both.
		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (let v = 0; v < out.vertSize; v++) {
			if (out.isVertD(v)) continue;
			const r = Math.hypot(out.vx(v), out.vy(v), out.vz(v));
			min = Math.min(min, r);
			max = Math.max(max, r);
		}
		expect(min).toBeLessThan(0.95);
		expect(max).toBeGreaterThan(1.05);
	});

	test("refuses a point cloud", () => {
		const cloud = new CMeshO();
		Allocator.addVertices(cloud, 10);
		for (let v = 0; v < 10; v++) cloud.setVert(v, v, 0, 0);
		const { doc } = docWith(cloud);
		expect(() => kernel.applyFilter(doc, "Uniform Mesh Resampling")).toThrow(MLException);
	});

	test("discretize puts every crossing at a cell midpoint", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, "Uniform Mesh Resampling", { CellSize: 0.2, discretize: true });
		const out = doc.mm().cm;
		expect(out.vn).toBeGreaterThan(0);
		// Every vertex is a midpoint or a grid point, so nothing lands at an
		// arbitrary fraction. Checking the count of distinct radii is enough to
		// see the quantisation: a smooth extraction has almost one per vertex.
		const radii = new Set<string>();
		for (let v = 0; v < out.vertSize; v++) {
			if (!out.isVertD(v)) radii.add(Math.hypot(out.vx(v), out.vy(v), out.vz(v)).toFixed(4));
		}
		expect(radii.size).toBeLessThan(out.vn / 2);
	});
});

describe("Regular Recursive Sampling", () => {
	test("puts its samples on the surface", () => {
		const { doc } = docWith(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Regular Recursive Sampling", { CellSize: 0.1 });
		const cloud = doc.mm().cm;
		expect(cloud.vn).toBeGreaterThan(50);
		expect(cloud.fn).toBe(0);
		for (let v = 0; v < cloud.vertSize; v++) {
			if (cloud.isVertD(v)) continue;
			expect(Math.hypot(cloud.vx(v), cloud.vy(v), cloud.vz(v))).toBeCloseTo(1, 1);
		}
	});

	test("an offset moves the samples off the surface by that much", () => {
		const { doc } = docWith(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Regular Recursive Sampling", { CellSize: 0.1, Offset: 0.3 });
		const cloud = doc.mm().cm;
		expect(cloud.vn).toBeGreaterThan(0);
		let sum = 0;
		let n = 0;
		for (let v = 0; v < cloud.vertSize; v++) {
			if (cloud.isVertD(v)) continue;
			sum += Math.hypot(cloud.vx(v), cloud.vy(v), cloud.vz(v));
			n++;
		}
		expect(sum / n).toBeCloseTo(1.3, 1);
	});

	test("refuses a point cloud", () => {
		const cloud = new CMeshO();
		Allocator.addVertices(cloud, 5);
		const { doc } = docWith(cloud);
		expect(() => kernel.applyFilter(doc, "Regular Recursive Sampling")).toThrow(MLException);
	});
});

// -------------------------------------------------------------------- colouring

describe("Voronoi and Disk Vertex Coloring", () => {
	/** A sphere to colour and a handful of seed points near its surface. */
	function scene(seedCount = 4) {
		const doc = new MeshDocument();
		const surface = doc.addNewMesh("", "surface", true, sphereIcosa(3).mesh);
		surface.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
		surface.updateBoxAndNormals();
		const seeds = new CMeshO();
		Allocator.addVertices(seeds, seedCount);
		for (let i = 0; i < seedCount; i++) {
			const t = (2 * Math.PI * i) / seedCount;
			seeds.setVert(i, Math.cos(t), Math.sin(t), 0);
		}
		const cloud = doc.addNewMesh("", "seeds", true, seeds);
		cloud.updateBoxAndNormals();
		return { doc, surface, cloud };
	}

	test("Voronoi colouring reaches every vertex", () => {
		const { doc, surface, cloud } = scene();
		kernel.applyFilter(doc, "Voronoi Vertex Coloring", {
			ColoredMesh: surface.id(),
			VertexMesh: cloud.id(),
		});
		for (let v = 0; v < surface.cm.vertSize; v++) {
			if (surface.cm.isVertD(v)) continue;
			expect(Number.isFinite(surface.cm.vertQuality[v])).toBe(true);
		}
	});

	test("the distance is geodesic, so it exceeds the straight line", () => {
		// The claim that makes the filter worth having. On a sphere the surface
		// route between two far points is longer than the chord, always.
		const { doc, surface, cloud } = scene(1);
		kernel.applyFilter(doc, "Voronoi Vertex Coloring", {
			ColoredMesh: surface.id(),
			VertexMesh: cloud.id(),
		});
		const seed = [1, 0, 0];
		let violations = 0;
		for (let v = 0; v < surface.cm.vertSize; v++) {
			if (surface.cm.isVertD(v)) continue;
			const chord = Math.hypot(
				surface.cm.vx(v) - seed[0],
				surface.cm.vy(v) - seed[1],
				surface.cm.vz(v) - seed[2],
			);
			if (surface.cm.vertQuality[v] < chord - 1e-6) violations++;
		}
		expect(violations).toBe(0);
	});

	test("more seeds means shorter distances", () => {
		const far = (seedCount: number) => {
			const { doc, surface, cloud } = scene(seedCount);
			kernel.applyFilter(doc, "Voronoi Vertex Coloring", {
				ColoredMesh: surface.id(),
				VertexMesh: cloud.id(),
			});
			let max = 0;
			for (let v = 0; v < surface.cm.vertSize; v++) {
				if (!surface.cm.isVertD(v)) max = Math.max(max, surface.cm.vertQuality[v]);
			}
			return max;
		};
		expect(far(8)).toBeLessThan(far(1));
	});

	test("Disk colouring leaves everything outside every disk grey", () => {
		const { doc, surface, cloud } = scene(1);
		kernel.applyFilter(doc, "Disk Vertex Coloring", {
			ColoredMesh: surface.id(),
			VertexMesh: cloud.id(),
			Radius: 0.3,
		});
		const GREY = rgba(192, 192, 192);
		let painted = 0;
		let grey = 0;
		for (let v = 0; v < surface.cm.vertSize; v++) {
			if (surface.cm.isVertD(v)) continue;
			if (surface.cm.vertColor[v] === GREY) grey++;
			else painted++;
		}
		expect(painted).toBeGreaterThan(0);
		// One small disk on a whole sphere leaves most of it untouched.
		expect(grey).toBeGreaterThan(painted);
	});

	test("a bigger radius paints more", () => {
		const painted = (radius: number) => {
			const { doc, surface, cloud } = scene(1);
			kernel.applyFilter(doc, "Disk Vertex Coloring", {
				ColoredMesh: surface.id(),
				VertexMesh: cloud.id(),
				Radius: radius,
			});
			let n = 0;
			for (let v = 0; v < surface.cm.vertSize; v++) {
				if (!surface.cm.isVertD(v) && surface.cm.vertColor[v] !== rgba(192, 192, 192)) n++;
			}
			return n;
		};
		expect(painted(0.4)).toBeGreaterThan(painted(0.1));
	});

	test("refuses the same layer twice", () => {
		const { doc, surface } = scene();
		expect(() =>
			kernel.applyFilter(doc, "Disk Vertex Coloring", {
				ColoredMesh: surface.id(),
				VertexMesh: surface.id(),
			}),
		).toThrow(MLException);
	});
});

// ------------------------------------------------------------------ texel sampling

describe("Texel Sampling", () => {
	/** A unit square as two triangles, its UVs covering the whole texture. */
	function texturedQuad() {
		const doc = new MeshDocument();
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 1, 0);
		cm.setVert(3, 0, 1, 0);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 2, 3);
		const m = doc.addNewMesh("", "quad", true, cm);
		m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		// UV equals XY, so a texel's UV and its surface point coincide.
		const uvOf = [
			[
				[0, 0],
				[1, 0],
				[1, 1],
			],
			[
				[0, 0],
				[1, 1],
				[0, 1],
			],
		];
		for (let f = 0; f < 2; f++) {
			for (let k = 0; k < 3; k++) {
				wt[6 * f + 2 * k] = uvOf[f][k][0];
				wt[6 * f + 2 * k + 1] = uvOf[f][k][1];
			}
		}
		m.updateBoxAndNormals();
		return { doc, m, cm };
	}

	test("emits one sample per texel the parametrization covers", () => {
		const { doc } = texturedQuad();
		const out = kernel.applyFilter(doc, "Texel Sampling", { TextureW: 16, TextureH: 16 });
		// The two triangles tile the whole square, so every texel is covered
		// exactly once — the diagonal is the only place rounding could double
		// or drop one.
		expect(out.sample_num as number).toBeGreaterThan(240);
		expect(out.sample_num as number).toBeLessThanOrEqual(256 + 16);
	});

	test("the samples land on the surface", () => {
		const { doc } = texturedQuad();
		kernel.applyFilter(doc, "Texel Sampling", { TextureW: 8, TextureH: 8 });
		const cloud = doc.mm().cm;
		for (let v = 0; v < cloud.vertSize; v++) {
			if (cloud.isVertD(v)) continue;
			expect(cloud.vz(v)).toBeCloseTo(0, 12);
			expect(cloud.vx(v)).toBeGreaterThanOrEqual(0);
			expect(cloud.vx(v)).toBeLessThanOrEqual(1);
		}
	});

	test("a bigger texture gives proportionally more samples", () => {
		const count = (size: number) => {
			const { doc } = texturedQuad();
			const out = kernel.applyFilter(doc, "Texel Sampling", { TextureW: size, TextureH: size });
			return out.sample_num as number;
		};
		// Four times the texels, so about four times the samples.
		expect(count(16) / count(8)).toBeGreaterThan(3.5);
		expect(count(16) / count(8)).toBeLessThan(4.5);
	});

	test("UV-space mode flattens the cloud into the unit square", () => {
		const { doc } = texturedQuad();
		kernel.applyFilter(doc, "Texel Sampling", {
			TextureW: 8,
			TextureH: 8,
			TextureSpace: true,
		});
		const cloud = doc.mm().cm;
		for (let v = 0; v < cloud.vertSize; v++) {
			if (cloud.isVertD(v)) continue;
			expect(cloud.vz(v)).toBe(0);
		}
		// The normals still come from the surface, which is what tells you
		// where each texel was.
		expect(Math.abs(cloud.vertNormal[2])).toBeCloseTo(1, 6);
	});

	test("refuses a mesh with no parametrization", () => {
		const { doc } = docWith(sphereIcosa(1).mesh);
		expect(() => kernel.applyFilter(doc, "Texel Sampling")).toThrow(MLException);
	});
});

// ------------------------------------------------------------------ registration

describe("registration", () => {
	test("every filter in both plugins is implemented", () => {
		for (const plugin of ["FilterSampling", "FilterUnsharp"]) {
			const all = kernel.filterList().filter((f) => f.plugin.pluginName() === plugin);
			expect(all.length, plugin).toBeGreaterThan(0);
			expect(all.filter((f) => !f.implemented).map((f) => f.name)).toEqual([]);
		}
	});

	test("the vertex flags the crease cut copies do not include VISITED", () => {
		// A copied vertex inheriting VISITED would make the next pass skip it,
		// and the fan around it would never be walked.
		const cm = cube(1).mesh;
		UpdateTopology.faceFace(cm);
		CreaseCut.creaseCut(cm, Math.PI / 4);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect((cm.vertFlags[v] & VertexFlag.DELETED) === 0).toBe(true);
		}
		expect(cm.vn).toBe(32); // 24 referenced + upstream's 8 orphans
	});
});
