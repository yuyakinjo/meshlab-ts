/**
 * The last six of `filter_clean` and `filter_select`.
 *
 * Ball pivoting gets the most attention, because it is the only filter here
 * that builds a surface out of nothing and so the only one where "did it work"
 * needs a real answer. The test is Euler's formula: a closed triangulated
 * surface of genus zero has exactly `2V - 4` faces and `3V - 6` edges, with no
 * boundary and no non-manifold edge. Reconstructing a sphere's point cloud and
 * getting anything else back means the front closed wrongly somewhere, and the
 * count says so without needing a reference mesh to compare against.
 *
 * Self-intersection is the other one worth a note. The interesting cases are
 * all about what should *not* be reported: two faces sharing an edge, two
 * sharing only a corner, and any face against itself. A test that only checks
 * a genuine crossing is found would pass with a predicate that answers "yes"
 * to everything.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { ballPivoting } from "../../src/vcg/complex/create/ball_pivoting.ts";
import { VertexFlag } from "../../src/vcg/complex/flags.ts";
import { UpdateTexture } from "../../src/vcg/complex/update/texture.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { intersectionTriangleTriangle } from "../../src/vcg/space/intersection3.ts";
import { gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** The vertices of a mesh, with its faces thrown away. */
function cloudOf(src: CMeshO): CMeshO {
	const cm = new CMeshO();
	Allocator.addVertices(cm, src.vn);
	for (let v = 0; v < src.vn; v++) cm.setVert(v, src.vx(v), src.vy(v), src.vz(v));
	return cm;
}

function docWith(cm: CMeshO, channels: number = MeshElement.MM_NONE) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	if (channels !== MeshElement.MM_NONE) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

function countSelectedFaces(cm: CMeshO): number {
	let n = 0;
	for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f) && cm.isFaceS(f)) n++;
	return n;
}

function countSelectedVerts(cm: CMeshO): number {
	let n = 0;
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v) && cm.isVertS(v)) n++;
	return n;
}

// ----------------------------------------------------------- ball pivoting

describe("Surface Reconstruction: Ball Pivoting", () => {
	test("closes a sphere's point cloud exactly", () => {
		for (const subdiv of [2, 3, 4]) {
			const cm = cloudOf(sphereIcosa(subdiv).mesh);
			const before = cm.vn;
			ballPivoting(cm, {});

			// Every input point is used and none is invented — the interpolating
			// property that distinguishes this from Poisson.
			expect(cm.vn, `subdiv ${subdiv}`).toBe(before);
			// A closed genus-zero surface: V - E + F = 2, so F = 2V - 4.
			expect(cm.fn, `subdiv ${subdiv}`).toBe(2 * cm.vn - 4);
			const edges = Clean.countEdgeNum(cm);
			expect(edges.total, `subdiv ${subdiv}`).toBe(3 * cm.vn - 6);
			expect(edges.boundary, `subdiv ${subdiv}`).toBe(0);
			expect(edges.nonManifold, `subdiv ${subdiv}`).toBe(0);
		}
	});

	test("the result is coherently oriented", () => {
		const cm = cloudOf(sphereIcosa(3).mesh);
		ballPivoting(cm, {});
		UpdateTopology.faceFace(cm);
		expect(Clean.isCoherentlyOrientedMesh(cm)).toBe(true);
	});

	test("a flat grid comes back as a disc with one boundary loop", () => {
		const cm = cloudOf(gridPlane(10, 10).mesh);
		ballPivoting(cm, {});
		UpdateTopology.faceFace(cm);
		expect(cm.fn).toBeGreaterThan(0);
		// One hole, because a disc's boundary is a single loop. More than one
		// would mean the front tore the sheet.
		expect(Clean.countHoles(cm)).toBe(1);
		expect(Clean.countEdgeNum(cm).nonManifold).toBe(0);
	});

	test("a radius smaller than the point spacing builds nothing", () => {
		const cm = cloudOf(sphereIcosa(3).mesh);
		// The points are about 0.08 apart; a ball a tenth of that can never
		// rest on three of them.
		const result = ballPivoting(cm, { radius: 0.005 });
		expect(result.addedFaces).toBe(0);
	});

	test("the automatic radius scales with the mesh", () => {
		const small = ballPivoting(cloudOf(sphereIcosa(3).mesh), {});
		const scaled = sphereIcosa(3).mesh;
		for (let v = 0; v < scaled.vertSize; v++) {
			scaled.setVert(v, scaled.vx(v) * 10, scaled.vy(v) * 10, scaled.vz(v) * 10);
		}
		const big = ballPivoting(cloudOf(scaled), {});
		expect(big.radius / small.radius).toBeCloseTo(10, 6);
		expect(big.addedFaces).toBe(small.addedFaces);
	});

	test("existing faces are kept and their border seeds the front", () => {
		// Half a sphere already triangulated, the rest bare points: the filter
		// should fill in around what is there rather than start over.
		const src = sphereIcosa(3).mesh;
		const cm = cloudOf(src);
		const keep: number[] = [];
		for (let f = 0; f < src.faceSize; f++) {
			if (src.isFaceD(f)) continue;
			const z = [0, 1, 2].map((k) => src.vz(src.fv(f, k)));
			if (Math.min(...z) > 0.2) keep.push(f);
		}
		const first = Allocator.addFaces(cm, keep.length);
		keep.forEach((f, i) => {
			cm.setFace(first + i, src.fv(f, 0), src.fv(f, 1), src.fv(f, 2));
		});

		const startingFaces = cm.fn;
		expect(startingFaces).toBeGreaterThan(0);
		const result = ballPivoting(cm, {});
		expect(result.addedFaces).toBeGreaterThan(0);
		expect(cm.fn).toBe(startingFaces + result.addedFaces);
	});

	test("fewer than three points is a no-op rather than an error", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 2);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		expect(ballPivoting(cm, {}).addedFaces).toBe(0);
	});

	test("through the filter, with the initial faces deleted", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const points = cm.vn;
		kernel.applyFilter(doc, "Surface Reconstruction: Ball Pivoting", { DeleteFaces: true });
		expect(cm.vn).toBe(points);
		expect(cm.fn).toBe(2 * points - 4);
	});

	test("refuses a clustering radius outside 0..100", () => {
		const { doc } = docWith(cloudOf(sphereIcosa(2).mesh));
		for (const bad of [0, 100, 150]) {
			expect(() =>
				kernel.applyFilter(doc, "Surface Reconstruction: Ball Pivoting", { Clustering: bad }),
			).toThrow(MLException);
		}
	});
});

// ------------------------------------------------------ remove wrt quality

describe("Remove Vertices wrt Quality", () => {
	function scored() {
		const { doc, m, cm } = docWith(sphereIcosa(3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vz(v);
		return { doc, m, cm };
	}

	test("deletes every vertex below the threshold and the faces using them", () => {
		const { doc, cm } = scored();
		kernel.applyFilter(doc, "Remove Vertices wrt Quality", { MaxQualityThr: 0 });
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) expect(cm.vertQuality[v]).toBeGreaterThanOrEqual(0);
		}
		// And nothing left refers to a deleted vertex.
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) expect(cm.isVertD(cm.fv(f, k))).toBe(false);
		}
	});

	test("a threshold at the minimum removes nothing", () => {
		// The slider is clamped to the quality range, so -1 is as low as it goes
		// on a unit sphere — and the test is `q < threshold`, which no vertex
		// satisfies at the minimum.
		const { doc, cm } = scored();
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, "Remove Vertices wrt Quality", { MaxQualityThr: -1 });
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
	});

	test("a threshold at the maximum leaves no faces", () => {
		const { doc, cm } = scored();
		kernel.applyFilter(doc, "Remove Vertices wrt Quality", { MaxQualityThr: 1 });
		// Only the single topmost vertex can survive `q < 1`, and one vertex
		// carries no face.
		expect(cm.vn).toBeLessThanOrEqual(1);
		expect(cm.fn).toBe(0);
	});

	test("reports what it deleted", () => {
		const { doc, cm } = scored();
		const before = cm.vn;
		const out = kernel.applyFilter(doc, "Remove Vertices wrt Quality", { MaxQualityThr: 0 });
		expect(out.deleted_vertices as number).toBe(before - cm.vn);
		expect(out.deleted_faces as number).toBeGreaterThan(0);
	});
});

// ---------------------------------------------------------- wedge tex merge

describe("Merge Wedge Texture Coord", () => {
	/** A sphere whose wedge UVs carry a tiny per-face rounding error. */
	function jittered(spread: number) {
		const { doc, m, cm } = docWith(sphereIcosa(2).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				// The same UV per vertex, plus a per-face wobble: exactly the
				// fake seam the filter exists to remove.
				wt[6 * f + 2 * k] = (v % 17) / 17 + ((f % 3) - 1) * spread;
				wt[6 * f + 2 * k + 1] = (v % 13) / 13;
			}
		}
		return { doc, m, cm };
	}

	function distinctUVsPerVertex(cm: CMeshO): number {
		const seen = new Map<number, Set<string>>();
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				if (!seen.has(v)) seen.set(v, new Set());
				(seen.get(v) as Set<string>).add(`${wt[6 * f + 2 * k]},${wt[6 * f + 2 * k + 1]}`);
			}
		}
		let total = 0;
		for (const s of seen.values()) total += s.size;
		return total;
	}

	test("collapses UVs that differ by less than the threshold", () => {
		const { doc, cm } = jittered(1e-6);
		const before = distinctUVsPerVertex(cm);
		const out = kernel.applyFilter(doc, "Merge Wedge Texture Coord", { MergeThr: 1e-4 });
		expect(out.merged as number).toBeGreaterThan(0);
		expect(distinctUVsPerVertex(cm)).toBeLessThan(before);
	});

	test("leaves a genuine seam alone", () => {
		// A wobble far wider than the threshold is a real chart boundary.
		const { doc, cm } = jittered(0.1);
		const before = distinctUVsPerVertex(cm);
		const out = kernel.applyFilter(doc, "Merge Wedge Texture Coord", { MergeThr: 1e-4 });
		expect(out.merged).toBe(0);
		expect(distinctUVsPerVertex(cm)).toBe(before);
	});

	test("is idempotent", () => {
		const { doc, cm } = jittered(1e-6);
		kernel.applyFilter(doc, "Merge Wedge Texture Coord", { MergeThr: 1e-4 });
		const after = Float64Array.from(cm.wedgeTexCoord as Float64Array);
		const out = kernel.applyFilter(doc, "Merge Wedge Texture Coord", { MergeThr: 1e-4 });
		expect(out.merged).toBe(0);
		expect([...(cm.wedgeTexCoord as Float64Array)]).toEqual([...after]);
	});

	test("merges within a vertex, not across the whole mesh", () => {
		// Two vertices given the same pair of near-equal UVs. Merging is
		// per-vertex, so each keeps its own representative and no coordinate
		// travels from one vertex to the other.
		const cm = new CMeshO();
		Allocator.addVertices(cm, 4);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 1, 0);
		cm.setVert(3, 0, 1, 0);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 2, 3);
		const { doc } = docWith(cm, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		wt[0] = 0.5;
		wt[1] = 0.5; // face 0, corner 0 -> vertex 0
		wt[6] = 0.5 + 1e-7;
		wt[7] = 0.5; // face 1, corner 0 -> vertex 0, nearly the same
		const out = kernel.applyFilter(doc, "Merge Wedge Texture Coord", { MergeThr: 1e-4 });
		expect(out.merged).toBe(1);
		// Which of the two values wins is whichever the VF chain reaches first,
		// so the claim is that they now agree, not what they agree on.
		expect(wt[6]).toBe(wt[0]);
		expect(wt[7]).toBe(wt[1]);
	});

	test("the algorithm alone leaves an untextured mesh untouched", () => {
		const cm = sphereIcosa(1).mesh;
		expect(UpdateTexture.wedgeTexMergeClose(cm, 1e-4)).toBe(0);
	});
});

// ------------------------------------------------------------ snap borders

describe("Snap Mismatched Borders", () => {
	/**
	 * Two strips meeting along y = 0, where the upper strip's vertices fall at
	 * the midpoints of the lower strip's border edges. Exactly the mismatch a
	 * pair of range maps produces.
	 */
	function mismatchedSeam() {
		const cm = new CMeshO();
		// Lower sheet: border vertices at x = 0, 1, 2.
		const lower = [
			[0, 0, 0],
			[1, 0, 0],
			[2, 0, 0],
			[0, -1, 0],
			[1, -1, 0],
			[2, -1, 0],
		];
		// Upper sheet: border vertices at x = 0.5, 1.5 — the midpoints.
		const upper = [
			[0.5, 0, 0],
			[1.5, 0, 0],
			[0.5, 1, 0],
			[1.5, 1, 0],
		];
		const all = [...lower, ...upper];
		Allocator.addVertices(cm, all.length);
		all.forEach((p, i) => {
			cm.setVert(i, p[0], p[1], p[2]);
		});
		const faces = [
			[0, 3, 4],
			[0, 4, 1],
			[1, 4, 5],
			[1, 5, 2],
			[6, 7, 9],
			[6, 9, 8],
		];
		const first = Allocator.addFaces(cm, faces.length);
		faces.forEach((f, i) => {
			cm.setFace(first + i, f[0], f[1], f[2]);
		});
		return docWith(cm);
	}

	test("splits the edge a mismatched vertex lands on", () => {
		const { doc, cm } = mismatchedSeam();
		const before = { vn: cm.vn, fn: cm.fn };
		const out = kernel.applyFilter(doc, "Snap Mismatched Borders", {
			EdgeDistRatio: 0.01,
			UnifyVertices: false,
		});
		// Two upper-sheet vertices sit on two lower-sheet edges.
		expect(out.split_faces as number).toBeGreaterThan(0);
		expect(cm.fn).toBe(before.fn + (out.split_faces as number));
		expect(cm.vn).toBe(before.vn + (out.split_faces as number));
	});

	test("the split adds no area and moves nothing", () => {
		// Splitting an edge at a point already on it cannot change the surface.
		const area = (cm: CMeshO) => {
			let total = 0;
			for (let f = 0; f < cm.faceSize; f++) {
				if (cm.isFaceD(f)) continue;
				const p = [0, 1, 2].map((k) => {
					const v = cm.fv(f, k);
					return [cm.vx(v), cm.vy(v), cm.vz(v)];
				});
				const u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
				const w = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
				total +=
					Math.hypot(
						u[1] * w[2] - u[2] * w[1],
						u[2] * w[0] - u[0] * w[2],
						u[0] * w[1] - u[1] * w[0],
					) / 2;
			}
			return total;
		};
		const { doc, cm } = mismatchedSeam();
		const before = area(cm);
		kernel.applyFilter(doc, "Snap Mismatched Borders", {
			EdgeDistRatio: 0.01,
			UnifyVertices: false,
		});
		expect(area(cm)).toBeCloseTo(before, 9);
	});

	test("welding after the split joins the two sheets", () => {
		const { doc, cm } = mismatchedSeam();
		UpdateTopology.faceFace(cm);
		const before = Clean.countConnectedComponents(cm);
		expect(before).toBe(2);
		kernel.applyFilter(doc, "Snap Mismatched Borders", {
			EdgeDistRatio: 0.01,
			UnifyVertices: true,
		});
		UpdateTopology.faceFace(cm);
		expect(Clean.countConnectedComponents(cm)).toBe(1);
	});

	test("a mesh whose borders already match is left alone", () => {
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		const out = kernel.applyFilter(doc, "Snap Mismatched Borders", {
			EdgeDistRatio: 0.01,
			UnifyVertices: false,
		});
		expect(out.split_faces).toBe(0);
		expect(cm.vn).toBe(before.vn);
		expect(cm.fn).toBe(before.fn);
	});

	test("a closed mesh has no borders to snap", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = cm.fn;
		const out = kernel.applyFilter(doc, "Snap Mismatched Borders", { UnifyVertices: false });
		expect(out.split_faces).toBe(0);
		expect(cm.fn).toBe(before);
	});
});

// ------------------------------------------------------- self-intersection

describe("Select Self Intersecting Faces", () => {
	/** Two triangles that genuinely pass through each other. */
	function crossing() {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 6);
		cm.setVert(0, -1, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 0, 1, 0);
		cm.setVert(3, 0, 0.5, -1);
		cm.setVert(4, 0, 0.5, 1);
		cm.setVert(5, 0, -0.5, 0);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 3, 4, 5);
		return cm;
	}

	test("finds a genuine crossing", () => {
		const { doc, cm } = docWith(crossing());
		const out = kernel.applyFilter(doc, "Select Self Intersecting Faces");
		expect(out.selected_faces).toBe(2);
		expect(countSelectedFaces(cm)).toBe(2);
	});

	test("a clean sphere has none", () => {
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Select Self Intersecting Faces");
		expect(out.selected_faces).toBe(0);
		expect(countSelectedFaces(cm)).toBe(0);
	});

	test("neighbours sharing an edge are not intersections", () => {
		// Every face of a grid shares an edge with a neighbour; reporting those
		// would make the filter useless on any real mesh.
		const { doc } = docWith(gridPlane(6, 6).mesh);
		const out = kernel.applyFilter(doc, "Select Self Intersecting Faces");
		expect(out.selected_faces).toBe(0);
	});

	test("two triangles hinged at a single vertex are not intersections", () => {
		// The case upstream's half-way offset exists for: without it, the two
		// opposite edges lie in a common plane and read as touching.
		const cm = new CMeshO();
		Allocator.addVertices(cm, 5);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 1, 1, 0);
		cm.setVert(3, -1, 0, 0);
		cm.setVert(4, -1, -1, 0);
		Allocator.addFaces(cm, 2);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 3, 4);
		const { doc } = docWith(cm);
		expect(kernel.applyFilter(doc, "Select Self Intersecting Faces").selected_faces).toBe(0);
	});

	test("clears a previous selection rather than adding to it", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= 0x0020;
		kernel.applyFilter(doc, "Select Self Intersecting Faces");
		expect(countSelectedFaces(cm)).toBe(0);
	});

	test("the triangle-triangle predicate is symmetric", () => {
		const a = [
			[0, 0, 0],
			[1, 0, 0],
			[0, 1, 0],
		];
		const b = [
			[0.2, 0.2, -1],
			[0.2, 0.2, 1],
			[0.9, 0.9, 0],
		];
		expect(intersectionTriangleTriangle(a, b)).toBe(true);
		expect(intersectionTriangleTriangle(b, a)).toBe(true);
		const far = b.map((p) => [p[0] + 10, p[1], p[2]]);
		expect(intersectionTriangleTriangle(a, far)).toBe(false);
		expect(intersectionTriangleTriangle(far, a)).toBe(false);
	});
});

// --------------------------------------------------------- texture seams

describe("Select Vertex Texture Seams", () => {
	/** A grid whose UVs are split down the middle into two charts. */
	function twoCharts() {
		const { doc, m, cm } = docWith(gridPlane(4, 4).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				// u jumps by a whole unit across x = 0.5, so the faces either
				// side disagree about the UV on their shared edge.
				const left = cm.vx(v) <= 0.5;
				wt[6 * f + 2 * k] = cm.vx(v) + (left ? 0 : 1);
				wt[6 * f + 2 * k + 1] = cm.vy(v);
			}
		}
		return { doc, m, cm };
	}

	test("selects the vertices along the chart boundary", () => {
		const { doc, cm } = twoCharts();
		const out = kernel.applyFilter(doc, "Select Vertex Texture Seams");
		expect(out.selected_vertices as number).toBeGreaterThan(0);
		// The seam runs along x = 0.5, and the mesh boundary is a seam too.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v) || !cm.isVertS(v)) continue;
			const onSeam = Math.abs(cm.vx(v) - 0.5) < 1e-9;
			const onBorder = cm.vx(v) === 0 || cm.vx(v) === 1 || cm.vy(v) === 0 || cm.vy(v) === 1;
			expect(onSeam || onBorder).toBe(true);
		}
	});

	test("a single chart selects only the mesh's own boundary", () => {
		const { doc, m, cm } = docWith(gridPlane(4, 4).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				wt[6 * f + 2 * k] = cm.vx(v);
				wt[6 * f + 2 * k + 1] = cm.vy(v);
			}
		}
		kernel.applyFilter(doc, "Select Vertex Texture Seams");
		const seams = countSelectedVerts(cm);

		// The same count the ordinary border gives, because with one chart the
		// two notions of border coincide.
		UpdateTopology.faceFace(m.cm);
		kernel.applyFilter(doc, "Select Border");
		expect(seams).toBe(countSelectedVerts(cm));
	});

	test("restores the ordinary topology afterwards", () => {
		// The filter detaches faces to build the texture adjacency. Leaving it
		// that way would have every later filter see the charts as components.
		const { doc, cm } = twoCharts();
		kernel.applyFilter(doc, "Select Vertex Texture Seams");
		expect(Clean.countConnectedComponents(cm)).toBe(1);
		let border = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) if (cm.isBorderFF(f, e)) border++;
		}
		// A 4x4 grid's outline is 16 edges; the seam is no longer one of them.
		expect(border).toBe(16);
	});

	test("refuses a mesh with no texture coordinates", () => {
		const { doc } = docWith(sphereIcosa(1).mesh);
		expect(() => kernel.applyFilter(doc, "Select Vertex Texture Seams")).toThrow(MLException);
	});
});

// ---------------------------------------------------------- registration

describe("registration", () => {
	test("every filter in both plugins is implemented", () => {
		for (const plugin of ["FilterClean", "FilterSelect"]) {
			const all = kernel.filterList().filter((f) => f.plugin.pluginName() === plugin);
			expect(all.length, plugin).toBeGreaterThan(0);
			expect(all.filter((f) => !f.implemented).map((f) => f.name)).toEqual([]);
		}
	});

	test("the ball-pivoting flags do not leak into the result", () => {
		// The algorithm uses VISITED and BORDER as working state. A vertex left
		// marked would make the next filter believe the mesh has a boundary it
		// does not have.
		const cm = cloudOf(sphereIcosa(3).mesh);
		ballPivoting(cm, {});
		let stillBorder = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v) && (cm.vertFlags[v] & VertexFlag.BORDER) !== 0) stillBorder++;
		}
		expect(stillBorder).toBe(0);
	});
});
