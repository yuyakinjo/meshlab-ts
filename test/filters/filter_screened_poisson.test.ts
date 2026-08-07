/**
 * `filter_screened_poisson`.
 *
 * Poisson reconstruction has no closed form to compare against, but it has a
 * very strong contract: whatever comes out is a closed surface. So the tests
 * here check topology exactly — watertight, one component, the genus of the
 * shape that was sampled — and geometry to a tolerance, against solids whose
 * volume is known on paper.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import {
	POISSON_DEFAULTS,
	quantile,
	reconstructScreenedPoisson,
	trimByDensity,
} from "../../src/meshlabplugins/filter_screened_poisson/poisson_recon.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { SurfaceSampling } from "../../src/vcg/complex/point_sampling.ts";
import { estimateNormals } from "../../src/vcg/complex/pointcloud_normal.ts";
import { assertAllocatorConsistent, computeFacts } from "../helpers/invariants.ts";
import { cube, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();
const FILTER = "Surface Reconstruction: Screened Poisson";

/** Samples a surface and gives every sample the outward normal of its face. */
function orientedCloud(m: CMeshO, sampleNum: number, normalAt: (p: number[]) => number[]): CMeshO {
	const cloud = SurfaceSampling.montecarloSampling(m, sampleNum);
	for (let v = 0; v < cloud.vn; v++) {
		const n = normalAt([cloud.vx(v), cloud.vy(v), cloud.vz(v)]);
		for (let a = 0; a < 3; a++) cloud.vertNormal[3 * v + a] = n[a];
	}
	return cloud;
}

const sphereNormal = (p: number[]) => {
	const l = Math.hypot(p[0], p[1], p[2]) || 1;
	return [p[0] / l, p[1] / l, p[2] / l];
};

/** Outward normal of a torus of major radius 2: away from the nearest point of the core circle. */
const torusNormal = (p: number[]) => {
	const l = Math.hypot(p[0], p[1]) || 1;
	const n = [p[0] - (2 * p[0]) / l, p[1] - (2 * p[1]) / l, p[2]];
	const s = Math.hypot(n[0], n[1], n[2]) || 1;
	return [n[0] / s, n[1] / s, n[2] / s];
};

/** Face normal of an axis-aligned box: the dominant axis wins. */
const boxNormal = (p: number[]) => {
	let axis = 0;
	for (let a = 1; a < 3; a++) if (Math.abs(p[a]) > Math.abs(p[axis])) axis = a;
	return [0, 1, 2].map((a) => (a === axis ? Math.sign(p[axis]) : 0));
};

describe("Screened Poisson reconstruction", () => {
	test("a sampled sphere comes back closed, genus 0, and the right size", () => {
		const out = reconstructScreenedPoisson([
			orientedCloud(sphereIcosa(4).mesh, 20000, sphereNormal),
		]);
		const facts = computeFacts(out);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.nonManifoldEdges).toBe(0);
		expect(facts.genus).toBe(0);
		expect(facts.coherentlyOriented).toBe(true);
		assertAllocatorConsistent(out);

		// 4/3 pi for the unit sphere, within a couple of percent — the level set
		// is a first-order approximation on a finite grid, not the ideal surface.
		expect(volumeOf(out)).toBeGreaterThan((4 / 3) * Math.PI * 0.97);
		expect(volumeOf(out)).toBeLessThan((4 / 3) * Math.PI * 1.03);
		for (let v = 0; v < out.vn; v++) {
			expect(Math.hypot(out.vx(v), out.vy(v), out.vz(v))).toBeCloseTo(1, 1);
		}
	});

	test("a sampled torus keeps its hole", () => {
		const out = reconstructScreenedPoisson([
			orientedCloud(torus(2, 0.7, 60, 30).mesh, 30000, torusNormal),
		]);
		const facts = computeFacts(out);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		// The one property a volumetric method can plausibly get wrong: filling
		// the hole in, or shattering the surface into extra handles.
		expect(facts.genus).toBe(1);
		// 2 pi^2 R r^2 for the ideal torus; the sampled polygonal one is a little
		// smaller, so this is a loose sanity bound rather than a tight check.
		expect(volumeOf(out)).toBeGreaterThan(17);
		expect(volumeOf(out)).toBeLessThan(20);
	});

	test("a sampled box comes back closed, at the right volume", () => {
		const out = reconstructScreenedPoisson([orientedCloud(cube(2).mesh, 20000, boxNormal)]);
		const facts = computeFacts(out);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(0);
		// Side 2, so volume 8. Poisson rounds the corners, which costs a little.
		expect(volumeOf(out)).toBeGreaterThan(7.7);
		expect(volumeOf(out)).toBeLessThan(8.3);
	});

	test("estimated normals work as well as exact ones", () => {
		// The whole point of the point-cloud path: nobody hands you true normals.
		const cloud = SurfaceSampling.montecarloSampling(sphereIcosa(4).mesh, 20000);
		estimateNormals(cloud, { neighbors: 16, smoothIterations: 0 });
		const facts = computeFacts(reconstructScreenedPoisson([cloud]));
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(0);
	});

	test("two point sets reconstruct as one solid", () => {
		// Splitting one sphere's samples across two layers must not change the
		// answer in kind: `visibleLayer` merges them into a single point stream.
		const all = orientedCloud(sphereIcosa(4).mesh, 20000, sphereNormal);
		const halves = [new CMeshO(), new CMeshO()];
		for (const [i, half] of halves.entries()) {
			const mine: number[] = [];
			for (let v = 0; v < all.vn; v++) if (v % 2 === i) mine.push(v);
			const first = Allocator.addVertices(half, mine.length);
			for (let k = 0; k < mine.length; k++) {
				const v = mine[k];
				half.setVert(first + k, all.vx(v), all.vy(v), all.vz(v));
				for (let a = 0; a < 3; a++)
					half.vertNormal[3 * (first + k) + a] = all.vertNormal[3 * v + a];
			}
		}
		const facts = computeFacts(reconstructScreenedPoisson(halves));
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(0);
	});

	test("the same input always gives the same surface", () => {
		const cloud = orientedCloud(sphereIcosa(3).mesh, 5000, sphereNormal);
		const a = reconstructScreenedPoisson([cloud]);
		const b = reconstructScreenedPoisson([cloud]);
		expect(a.vn).toBe(b.vn);
		expect(a.fn).toBe(b.fn);
		expect(Array.from(a.vertCoord.subarray(0, 3 * a.vertSize))).toEqual(
			Array.from(b.vertCoord.subarray(0, 3 * b.vertSize)),
		);
	});

	test("depth is an upper bound, not a demand", () => {
		// MeshLab's own wording. A sparse cloud cannot support a 2^8 grid: the
		// splats stop touching and the level set shatters, so the depth is
		// capped by the sample count and every setting above the cap agrees.
		const cloud = orientedCloud(sphereIcosa(3).mesh, 4000, sphereNormal);
		const coarse = reconstructScreenedPoisson([cloud], { depth: 5 });
		for (const depth of [6, 8, 11]) {
			const out = reconstructScreenedPoisson([cloud], { depth });
			expect(computeFacts(out).components, `depth ${depth}`).toBe(1);
			expect(out.fn, `depth ${depth}`).toBe(coarse.fn);
		}
	});

	test("a lower depth gives a coarser surface", () => {
		const cloud = orientedCloud(sphereIcosa(4).mesh, 40000, sphereNormal);
		let previous = 0;
		for (const depth of [3, 4, 5, 6]) {
			const out = reconstructScreenedPoisson([cloud], { depth });
			expect(out.fn, `depth ${depth}`).toBeGreaterThan(previous);
			previous = out.fn;
		}
	});

	test("rejects a depth outside the supported range", () => {
		const cloud = orientedCloud(sphereIcosa(2).mesh, 500, sphereNormal);
		for (const depth of [0, -1, 13, 2.5]) {
			expect(() => reconstructScreenedPoisson([cloud], { depth })).toThrow(MLException);
		}
	});

	test("refuses a point set with no normals, and cleans it when asked", () => {
		const cloud = SurfaceSampling.montecarloSampling(sphereIcosa(3).mesh, 3000);
		// montecarloSampling leaves normals zeroed; nothing can be reconstructed
		// from unoriented points, and MeshLab says so rather than guessing.
		expect(() => reconstructScreenedPoisson([cloud])).toThrow(MLException);
		expect(() => reconstructScreenedPoisson([cloud], { preClean: true })).toThrow(MLException);

		// One bad sample among good ones: refused by default, dropped by preClean.
		const mostly = orientedCloud(sphereIcosa(3).mesh, 3000, sphereNormal);
		for (let a = 0; a < 3; a++) mostly.vertNormal[3 * 7 + a] = 0;
		expect(() => reconstructScreenedPoisson([mostly])).toThrow(MLException);
		expect(computeFacts(reconstructScreenedPoisson([mostly], { preClean: true })).watertight).toBe(
			true,
		);
	});

	test("refuses an empty or flat point set", () => {
		expect(() => reconstructScreenedPoisson([new CMeshO()])).toThrow(MLException);
		const dot = new CMeshO();
		const first = Allocator.addVertices(dot, 4);
		for (let i = 0; i < 4; i++) {
			dot.setVert(first + i, 1, 2, 3);
			dot.vertNormal[3 * (first + i)] = 1;
		}
		// Every point in the same place: no extent, so no volume to enclose.
		expect(() => reconstructScreenedPoisson([dot])).toThrow(MLException);
	});

	test("confidence weights the samples by quality", () => {
		const cloud = orientedCloud(sphereIcosa(3).mesh, 8000, sphereNormal);
		for (let v = 0; v < cloud.vn; v++) cloud.vertQuality[v] = cloud.vx(v) > 0 ? 4 : 1;
		const plain = reconstructScreenedPoisson([cloud]);
		const weighted = reconstructScreenedPoisson([cloud], { confidence: true });
		// Both still close, but the field is not the same one.
		expect(computeFacts(weighted).watertight).toBe(true);
		expect(weighted.fn).not.toBe(plain.fn);
	});
});

describe("density", () => {
	test("output vertices carry the sample density in their quality", () => {
		const out = reconstructScreenedPoisson([
			orientedCloud(sphereIcosa(3).mesh, 8000, sphereNormal),
		]);
		let positive = 0;
		for (let v = 0; v < out.vn; v++) {
			expect(out.vertQuality[v]).toBeGreaterThanOrEqual(0);
			if (out.vertQuality[v] > 0) positive++;
		}
		// A cut running between two nodes no sample splatted onto reads zero,
		// which is the honest answer and the one trimming acts on. The bulk of
		// the surface sits on splatted nodes, though.
		expect(positive / out.vn).toBeGreaterThan(0.9);
	});

	test("quantile interpolates between order statistics, like numpy", () => {
		const v = [1, 2, 3, 4];
		expect(quantile(v, 0)).toBe(1);
		expect(quantile(v, 1)).toBe(4);
		expect(quantile(v, 0.5)).toBeCloseTo(2.5, 12);
		expect(quantile(v, 0.25)).toBeCloseTo(1.75, 12);
		expect(Number.isNaN(quantile([], 0.5))).toBe(true);
	});

	test("trimming drops the sparsest vertices and their faces", () => {
		const out = reconstructScreenedPoisson([
			orientedCloud(sphereIcosa(3).mesh, 8000, sphereNormal),
		]);
		const before = out.vn;
		const dropped = trimByDensity(out, 20);
		expect(dropped).toBeGreaterThan(0);
		expect(out.vn).toBe(before - dropped);
		// Roughly the requested fraction, allowing for ties at the threshold.
		expect(dropped / before).toBeGreaterThan(0.15);
		expect(dropped / before).toBeLessThan(0.25);
		// No face may survive pointing at a deleted vertex.
		for (let f = 0; f < out.faceSize; f++) {
			if (out.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) expect(out.isVertD(out.fv(f, k))).toBe(false);
		}
		assertAllocatorConsistent(out);
	});

	test("trimming nothing is a no-op", () => {
		const out = reconstructScreenedPoisson([
			orientedCloud(sphereIcosa(2).mesh, 2000, sphereNormal),
		]);
		const before = out.vn;
		expect(trimByDensity(out, 0)).toBe(0);
		expect(trimByDensity(out, -5)).toBe(0);
		expect(out.vn).toBe(before);
	});
});

describe("the filter", () => {
	test("is registered exactly as MeshLab registers it", () => {
		const action = kernel.pluginManager.filterAction(FILTER);
		expect(action).toBeDefined();
		if (!action) return;
		expect(action.pythonName).toBe("generate_surface_reconstruction_screened_poisson");
		expect(filterClassToString(action.filterClass)).toBe("Remeshing");
		expect(action.plugin.pluginName()).toBe("FilterScreenedPoisson");
	});

	test("has MeshLab's parameters, at MeshLab's defaults", () => {
		const list = kernel.initParameterList(FILTER);
		const defaults: Record<string, number | boolean> = {
			visibleLayer: false,
			depth: 8,
			fullDepth: 5,
			cgDepth: 0,
			scale: 1.1,
			samplesPerNode: 1.5,
			pointWeight: 4,
			iters: 8,
			confidence: false,
			preClean: false,
		};
		for (const [name, value] of Object.entries(defaults)) {
			expect(list.getParameterByName(name).defaultValue.value, name).toBe(value);
		}
		// The algorithm's own defaults must not drift from the registered ones.
		expect(POISSON_DEFAULTS.depth).toBe(8);
		expect(POISSON_DEFAULTS.scale).toBe(1.1);
		expect(POISSON_DEFAULTS.pointWeight).toBe(4);
	});

	test("rejects a parameter it does not have", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "cloud", true, orientedCloud(sphereIcosa(2).mesh, 1000, sphereNormal));
		expect(() => kernel.applyFilter(doc, FILTER, { Depth: 6 })).toThrow();
	});

	test("adds a hidden layer and leaves the point cloud alone", () => {
		const doc = new MeshDocument();
		const cloud = orientedCloud(sphereIcosa(3).mesh, 6000, sphereNormal);
		const source = doc.addNewMesh("", "cloud", true, cloud);
		const before = { vn: source.cm.vn, fn: source.cm.fn };

		const out = kernel.applyFilter(doc, FILTER, { depth: 5 });
		expect(doc.meshNumber()).toBe(2);
		expect(source.cm.vn).toBe(before.vn);
		expect(source.cm.fn).toBe(before.fn);

		const pm = doc.requireMesh(out.new_mesh_id as number);
		expect(pm.label()).toBe("Poisson mesh");
		// MeshLab hides it so the cloud stays on screen, and marks the quality
		// channel because that is where the density went.
		expect(pm.isVisible()).toBe(false);
		expect(pm.hasDataMask(MeshElement.MM_VERTQUALITY)).toBe(true);
		expect(out.face_number as number).toBe(pm.cm.fn);
		expect(computeFacts(pm.cm).watertight).toBe(true);
	});

	test("visibleLayer reads every visible layer", () => {
		const doc = new MeshDocument();
		// Two spheres, far apart. Reading only the current layer can only ever
		// find one of them; reading both must find two separate solids.
		for (const shift of [0, 6]) {
			const cloud = orientedCloud(sphereIcosa(3).mesh, 6000, sphereNormal);
			for (let v = 0; v < cloud.vn; v++) {
				cloud.setVert(v, cloud.vx(v) + shift, cloud.vy(v), cloud.vz(v));
			}
			doc.addNewMesh("", `sphere at ${shift}`, true, cloud);
		}

		const single = kernel.applyFilter(doc, FILTER, { visibleLayer: false, depth: 6 });
		expect(computeFacts(doc.requireMesh(single.new_mesh_id as number).cm).components).toBe(1);

		doc.setCurrentMesh(0);
		const merged = kernel.applyFilter(doc, FILTER, { visibleLayer: true, depth: 6 });
		const mergedMesh = doc.requireMesh(merged.new_mesh_id as number).cm;
		const facts = computeFacts(mergedMesh);
		expect(facts.components).toBe(2);
		expect(facts.watertight).toBe(true);
		expect(facts.genus).toBe(0);
	});
});

/** Signed volume by the divergence theorem, positive when faces point outward. */
function volumeOf(m: CMeshO): number {
	let total = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		total +=
			(m.vx(a) * (m.vy(b) * m.vz(c) - m.vz(b) * m.vy(c)) -
				m.vy(a) * (m.vx(b) * m.vz(c) - m.vz(b) * m.vx(c)) +
				m.vz(a) * (m.vx(b) * m.vy(c) - m.vy(b) * m.vx(c))) /
			6;
	}
	return total;
}
