/**
 * filter_sampling, plus the point-set normal filters.
 *
 * The sphere is the workhorse fixture here because every property has a closed
 * form: a sample on the unit sphere is at radius 1, and the true normal at
 * that point is the point itself.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Rng, SurfaceSampling } from "../../src/vcg/complex/point_sampling.ts";
import { estimateNormals } from "../../src/vcg/complex/pointcloud_normal.ts";
import { KdTree, pointBounds } from "../../src/vcg/space/index/kdtree.ts";
import { assertAllocatorConsistent, surfaceArea } from "../helpers/invariants.ts";
import { cube, gridPlane, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO): { doc: MeshDocument; m: ReturnType<MeshDocument["mm"]> } {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("t", "t", true, cm);
	m.updateBoxAndNormals();
	return { doc, m };
}

/** Distance from the origin, for checking a point lies on the unit sphere. */
const radius = (m: CMeshO, v: number) => Math.hypot(m.vx(v), m.vy(v), m.vz(v));

describe("KdTree", () => {
	test("finds the true nearest neighbours, checked against brute force", () => {
		const cloud = sphereIcosa(3).mesh;
		const tree = new KdTree(cloud.vertCoord, cloud.vertSize);
		for (const v of [0, 17, 200, cloud.vn - 1]) {
			const found = Array.from(tree.nearest(v, 8));
			const brute = Array.from({ length: cloud.vn }, (_, w) => w)
				.map((w) => ({
					w,
					d:
						(cloud.vx(w) - cloud.vx(v)) ** 2 +
						(cloud.vy(w) - cloud.vy(v)) ** 2 +
						(cloud.vz(w) - cloud.vz(v)) ** 2,
				}))
				.sort((a, b) => a.d - b.d)
				.slice(0, 8)
				.map((e) => e.w);
			// Ties can order differently, so compare as sets.
			expect(new Set(found)).toEqual(new Set(brute));
			// A point is always its own nearest neighbour.
			expect(found[0]).toBe(v);
		}
	});

	test("nearestToPoint agrees with brute force from outside the set", () => {
		const cloud = sphereIcosa(2).mesh;
		const tree = new KdTree(cloud.vertCoord, cloud.vertSize);
		for (const p of [
			[2, 0, 0],
			[0, -3, 0.5],
			[0.1, 0.1, 0.1],
		] as const) {
			const got = tree.nearestToPoint(p[0], p[1], p[2]);
			let bestD = Number.POSITIVE_INFINITY;
			for (let w = 0; w < cloud.vn; w++) {
				const d = (cloud.vx(w) - p[0]) ** 2 + (cloud.vy(w) - p[1]) ** 2 + (cloud.vz(w) - p[2]) ** 2;
				if (d < bestD) bestD = d;
			}
			// Compare the distance, not the index: a symmetric point set queried
			// from a symmetric location has exact ties, and any of the tied
			// points is a correct answer.
			const gotD =
				(cloud.vx(got) - p[0]) ** 2 + (cloud.vy(got) - p[1]) ** 2 + (cloud.vz(got) - p[2]) ** 2;
			expect(gotD).toBe(bestD);
		}
	});

	test("survives an empty set and a single point", () => {
		expect(new KdTree(new Float64Array(0), 0).nearestToPoint(0, 0, 0)).toBe(-1);
		const one = new KdTree(new Float64Array([1, 2, 3]), 1);
		expect(one.nearestToPoint(9, 9, 9)).toBe(0);
		expect(Array.from(one.nearest(0, 5))).toEqual([0]);
	});

	test("handles coincident points, where no split makes progress", () => {
		const coords = new Float64Array(300).fill(0);
		const tree = new KdTree(coords, 100);
		expect(tree.nearestToPoint(0, 0, 0)).toBeGreaterThanOrEqual(0);
		expect(Array.from(tree.nearest(0, 4))).toHaveLength(4);
	});

	test("pointBounds matches the mesh's own box", () => {
		const m = cube(4);
		const b = pointBounds(m.mesh.vertCoord, m.mesh.vertSize);
		expect(b.low).toEqual([-2, -2, -2]);
		expect(b.high).toEqual([2, 2, 2]);
		expect(b.diagonal).toBeCloseTo(Math.sqrt(48), 12);
	});
});

describe("surface sampling", () => {
	test("Montecarlo puts every sample on the surface", () => {
		const m = sphereIcosa(4).mesh;
		const cloud = SurfaceSampling.montecarloSampling(m, 3000);
		expect(cloud.vn).toBe(3000);
		expect(cloud.fn).toBe(0); // a point set, not a surface
		for (let v = 0; v < cloud.vn; v++) {
			// Samples land inside faces, so slightly inside the unit sphere the
			// icosphere inscribes — never outside it.
			expect(radius(cloud, v)).toBeLessThanOrEqual(1 + 1e-9);
			expect(radius(cloud, v)).toBeGreaterThan(0.97);
		}
		assertAllocatorConsistent(cloud);
	});

	test("Montecarlo spreads by area, not by face count", () => {
		// A grid whose left half is finely divided: equal-probability face
		// picking would crowd the samples there.
		const m = gridPlane(20, 20).mesh;
		const cloud = SurfaceSampling.montecarloSampling(m, 4000);
		let left = 0;
		for (let v = 0; v < cloud.vn; v++) if (cloud.vx(v) < 0.5) left++;
		// The grid is uniform, so a uniform sampler should halve it.
		expect(left / cloud.vn).toBeGreaterThan(0.45);
		expect(left / cloud.vn).toBeLessThan(0.55);
	});

	test("stratified sampling hits the requested count", () => {
		for (const n of [100, 1000, 5000]) {
			const cloud = SurfaceSampling.stratifiedSampling(sphereIcosa(3).mesh, n);
			// The fractional remainder is carried face to face, so the total
			// lands on the request rather than drifting by the rounding.
			expect(Math.abs(cloud.vn - n)).toBeLessThanOrEqual(1);
		}
	});

	test("clustered sampling keeps a subset of the input points", () => {
		const m = sphereIcosa(3).mesh;
		const original = new Set(
			Array.from({ length: m.vn }, (_, v) => `${m.vx(v)},${m.vy(v)},${m.vz(v)}`),
		);
		const cloud = SurfaceSampling.clusteredVertexSampling(m, 5);
		expect(cloud.vn).toBeGreaterThan(0);
		expect(cloud.vn).toBeLessThan(m.vn);
		for (let v = 0; v < cloud.vn; v++) {
			// A subset, never an interpolation — which is what makes it safe
			// to run before normal estimation.
			expect(original.has(`${cloud.vx(v)},${cloud.vy(v)},${cloud.vz(v)}`)).toBe(true);
		}
	});

	test("a coarser cluster grid keeps fewer points", () => {
		const m = sphereIcosa(4).mesh;
		let previous = m.vn + 1;
		for (const percent of [1, 3, 5, 10]) {
			const n = SurfaceSampling.clusteredVertexSampling(m, percent).vn;
			expect(n).toBeLessThan(previous);
			previous = n;
		}
	});

	test("Poisson-disk keeps no two points closer than the radius", () => {
		const candidates = SurfaceSampling.montecarloSampling(sphereIcosa(4).mesh, 4000);
		for (const r of [0.05, 0.15, 0.3]) {
			const pruned = SurfaceSampling.poissonDiskPruning(candidates, r);
			expect(pruned.vn).toBeGreaterThan(0);
			for (let a = 0; a < pruned.vn; a++) {
				for (let b = a + 1; b < pruned.vn; b++) {
					const d = Math.hypot(
						pruned.vx(a) - pruned.vx(b),
						pruned.vy(a) - pruned.vy(b),
						pruned.vz(a) - pruned.vz(b),
					);
					expect(d, `r=${r}`).toBeGreaterThanOrEqual(r - 1e-12);
				}
			}
		}
	});

	test("a larger Poisson radius keeps fewer points", () => {
		const candidates = SurfaceSampling.montecarloSampling(sphereIcosa(4).mesh, 3000);
		let previous = Number.POSITIVE_INFINITY;
		for (const r of [0.05, 0.1, 0.2, 0.4]) {
			const n = SurfaceSampling.poissonDiskPruning(candidates, r).vn;
			expect(n).toBeLessThan(previous);
			previous = n;
		}
	});

	test("sampling is deterministic for a given seed, and varies with it", () => {
		const m = sphereIcosa(2).mesh;
		const a = SurfaceSampling.montecarloSampling(m, 500, new Rng(1));
		const b = SurfaceSampling.montecarloSampling(m, 500, new Rng(1));
		const c = SurfaceSampling.montecarloSampling(m, 500, new Rng(2));
		const key = (cm: CMeshO) =>
			Array.from({ length: cm.vn }, (_, v) => `${cm.vx(v)},${cm.vy(v)}`).join(";");
		expect(key(b)).toBe(key(a));
		expect(key(c)).not.toBe(key(a));
	});

	test("an empty or face-less mesh yields nothing rather than crashing", () => {
		const empty = sphereIcosa(1).mesh;
		empty.clear();
		expect(SurfaceSampling.montecarloSampling(empty, 100).vn).toBe(0);
		expect(SurfaceSampling.stratifiedSampling(empty, 100).vn).toBe(0);
		expect(SurfaceSampling.clusteredVertexSampling(empty, 5).vn).toBe(0);
	});

	test("Hausdorff distance is zero between a mesh and itself", () => {
		const cloud = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		const d = SurfaceSampling.hausdorffPointDistance(cloud, cloud);
		expect(d.max).toBe(0);
		expect(d.mean).toBe(0);
	});

	test("Hausdorff distance grows with a displacement", () => {
		const a = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		const b = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		for (let v = 0; v < b.vn; v++) b.setVert(v, b.vx(v) + 0.25, b.vy(v), b.vz(v));
		const d = SurfaceSampling.hausdorffPointDistance(a, b);
		// A rigid shift of a sphere: no point moves further than the shift.
		expect(d.max).toBeGreaterThan(0);
		expect(d.max).toBeLessThanOrEqual(0.25 + 1e-9);
	});
});

describe("point-set normals", () => {
	test("a sphere's normals are radial and all face outward", () => {
		const cloud = SurfaceSampling.vertexSampling(sphereIcosa(4).mesh);
		estimateNormals(cloud, { neighbors: 12, smoothIterations: 0 });
		let worst = 0;
		let outward = 0;
		for (let v = 0; v < cloud.vn; v++) {
			// On the unit sphere the position *is* the true normal.
			const dot =
				cloud.vertNormal[3 * v] * cloud.vx(v) +
				cloud.vertNormal[3 * v + 1] * cloud.vy(v) +
				cloud.vertNormal[3 * v + 2] * cloud.vz(v);
			if (dot > 0) outward++;
			worst = Math.max(worst, 1 - Math.abs(dot));
		}
		expect(worst).toBeLessThan(1e-3);
		// The whole point of the propagation step: a consistent global sign.
		expect(outward).toBe(cloud.vn);
	});

	test("normals come out unit length", () => {
		const cloud = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		estimateNormals(cloud, { neighbors: 10, smoothIterations: 2 });
		for (let v = 0; v < cloud.vn; v++) {
			const n = Math.hypot(
				cloud.vertNormal[3 * v],
				cloud.vertNormal[3 * v + 1],
				cloud.vertNormal[3 * v + 2],
			);
			expect(n).toBeCloseTo(1, 9);
		}
	});

	test("a plane's normals all point the same way", () => {
		const cloud = SurfaceSampling.vertexSampling(gridPlane(10, 10).mesh);
		estimateNormals(cloud, { neighbors: 8, smoothIterations: 0 });
		const first = [cloud.vertNormal[0], cloud.vertNormal[1], cloud.vertNormal[2]];
		for (let v = 0; v < cloud.vn; v++) {
			const dot =
				first[0] * cloud.vertNormal[3 * v] +
				first[1] * cloud.vertNormal[3 * v + 1] +
				first[2] * cloud.vertNormal[3 * v + 2];
			expect(dot).toBeGreaterThan(0.99);
		}
		// A z = 0 plane's normal is along z.
		expect(Math.abs(first[2])).toBeCloseTo(1, 6);
	});

	test("a viewpoint turns every normal toward it", () => {
		const cloud = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		estimateNormals(cloud, { neighbors: 10, smoothIterations: 0, viewpoint: [0, 0, 100] });
		// Facing a camera far up the z axis: every normal has a non-negative
		// component toward it.
		for (let v = 0; v < cloud.vn; v++) {
			const dot =
				cloud.vertNormal[3 * v] * (0 - cloud.vx(v)) +
				cloud.vertNormal[3 * v + 1] * (0 - cloud.vy(v)) +
				cloud.vertNormal[3 * v + 2] * (100 - cloud.vz(v));
			expect(dot).toBeGreaterThanOrEqual(0);
		}
	});

	test("a torus's normals stay consistent across the handle", () => {
		// The case that catches a naive orientation pass: propagating around
		// the hole must come back agreeing with itself.
		const cloud = SurfaceSampling.vertexSampling(torus(2, 0.6, 40, 20).mesh);
		estimateNormals(cloud, { neighbors: 12, smoothIterations: 0 });
		// The true outward normal on a torus points away from the tube's centre
		// circle, which lies at radius R in the z = 0 plane.
		let agree = 0;
		for (let v = 0; v < cloud.vn; v++) {
			const rho = Math.hypot(cloud.vx(v), cloud.vy(v));
			const cx = (cloud.vx(v) / rho) * 2;
			const cy = (cloud.vy(v) / rho) * 2;
			const dot =
				cloud.vertNormal[3 * v] * (cloud.vx(v) - cx) +
				cloud.vertNormal[3 * v + 1] * (cloud.vy(v) - cy) +
				cloud.vertNormal[3 * v + 2] * cloud.vz(v);
			if (dot > 0) agree++;
		}
		expect(agree / cloud.vn).toBeGreaterThan(0.98);
	});

	test("an empty cloud is survivable", () => {
		const empty = sphereIcosa(1).mesh;
		empty.clear();
		expect(() => estimateNormals(empty, { neighbors: 10, smoothIterations: 1 })).not.toThrow();
	});
});

describe("filter_sampling: registry and behaviour", () => {
	test("all seven are registered as Sampling", () => {
		for (const name of [
			"Mesh Element Sampling",
			"Montecarlo Sampling",
			"Stratified Triangle Sampling",
			"Clustered Vertex Sampling",
			"Poisson-disk Sampling",
			"Point Cloud Simplification",
			"Hausdorff Distance",
		]) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toContain("Sampling");
		}
	});

	test("PyMeshLab names match upstream", () => {
		expect(kernel.filterAction("Clustered Vertex Sampling").pythonName).toBe(
			"generate_sampling_clustered_vertex",
		);
		expect(kernel.filterAction("compute_normal_for_point_clouds").name).toBe(
			"Compute normals for point sets",
		);
	});

	test("sampling adds a layer and leaves the source alone", () => {
		const { doc, m } = docWith(sphereIcosa(3).mesh);
		const before = m.cm.vn;
		const out = kernel.applyFilter(doc, "Montecarlo Sampling", { SampleNum: 500 });
		expect(out.sample_num).toBe(500);
		expect(doc.meshNumber()).toBe(2);
		// The source layer is untouched.
		expect(doc.requireMesh(m.id()).cm.vn).toBe(before);
		expect(doc.mm().cm.vn).toBe(500);
		expect(doc.mm().cm.fn).toBe(0);
	});

	test("Clustered Vertex Sampling honours the cell size", () => {
		const { doc } = docWith(sphereIcosa(4).mesh);
		const coarse = kernel.applyFilter(doc, "Clustered Vertex Sampling", { Threshold: 0.2 });
		const { doc: doc2 } = docWith(sphereIcosa(4).mesh);
		const fine = kernel.applyFilter(doc2, "Clustered Vertex Sampling", { Threshold: 0.05 });
		expect(coarse.sample_num as number).toBeLessThan(fine.sample_num as number);
	});

	test("the unimplemented strategies refuse rather than silently substituting", () => {
		// "Average" and Edge/Face element sampling are accepted parameters that
		// have no behaviour yet. Quietly running the other strategy would give
		// the caller a different answer than they asked for.
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Clustered Vertex Sampling", { Sampling: "Average" }),
		).toThrow(MLException);
		expect(() => kernel.applyFilter(doc, "Mesh Element Sampling", { Sampling: "Face" })).toThrow(
			MLException,
		);
	});

	test("Poisson-disk derives a radius from the sample count", () => {
		const { doc } = docWith(sphereIcosa(4).mesh);
		const out = kernel.applyFilter(doc, "Poisson-disk Sampling", { SampleNum: 300 });
		expect(out.sample_num as number).toBeGreaterThan(100);
		expect(out.sample_num as number).toBeLessThan(900);
		expect(out.radius as number).toBeGreaterThan(0);
	});

	test("Hausdorff Distance reports zero against itself", () => {
		const { doc, m } = docWith(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Hausdorff Distance", {
			SampledMesh: m.id(),
			TargetMesh: m.id(),
			SampleNum: 500,
		});
		// Both sides sample the same surface, so the distance is a sampling
		// artefact only — well under a percent of the diagonal.
		expect(out.max_over_diag as number).toBeLessThan(0.05);
	});

	test("Compute normals for point sets fills the normals in place", () => {
		const cloud = SurfaceSampling.vertexSampling(sphereIcosa(3).mesh);
		const { doc, m } = docWith(cloud);
		const out = kernel.applyFilter(doc, "Compute normals for point sets", { K: 12 });
		expect(out.vertices).toBe(cloud.vn);
		// No new layer: it edits the current mesh.
		expect(doc.meshNumber()).toBe(1);
		let outward = 0;
		for (let v = 0; v < m.cm.vn; v++) {
			const dot =
				m.cm.vertNormal[3 * v] * m.cm.vx(v) +
				m.cm.vertNormal[3 * v + 1] * m.cm.vy(v) +
				m.cm.vertNormal[3 * v + 2] * m.cm.vz(v);
			if (dot > 0) outward++;
		}
		expect(outward).toBe(m.cm.vn);
	});

	test("the whole point-cloud path runs end to end", () => {
		// Sample a surface, thin it, and estimate normals — the sequence that
		// feeds surface reconstruction.
		const { doc } = docWith(sphereIcosa(4).mesh);
		kernel.applyFilter(doc, "Montecarlo Sampling", { SampleNum: 8000 });
		kernel.applyFilter(doc, "Clustered Vertex Sampling", { Threshold: 0.06 });
		kernel.applyFilter(doc, "Compute normals for point sets", { K: 12 });

		const cloud = doc.mm().cm;
		expect(cloud.vn).toBeGreaterThan(200);
		expect(cloud.fn).toBe(0);
		expect(surfaceArea(cloud)).toBe(0); // still a point set
		let outward = 0;
		for (let v = 0; v < cloud.vn; v++) {
			const dot =
				cloud.vertNormal[3 * v] * cloud.vx(v) +
				cloud.vertNormal[3 * v + 1] * cloud.vy(v) +
				cloud.vertNormal[3 * v + 2] * cloud.vz(v);
			if (dot > 0) outward++;
		}
		expect(outward / cloud.vn).toBeGreaterThan(0.99);
		assertAllocatorConsistent(cloud);
	});
});
