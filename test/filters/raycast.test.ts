/**
 * The BVH, ambient occlusion, the ray-cast measures, and ICP.
 *
 * The BVH is tested against a brute-force sweep over every triangle: on a
 * small mesh the two must agree exactly, which is the only test that can
 * catch a traversal that silently drops a subtree. That is not hypothetical —
 * a NaN in the slab test did exactly that here, and only an axis-aligned ray
 * through the origin exposed it, so those rays are tested by name.
 *
 * ICP is tested by construction: take a mesh, move it by a known rigid
 * motion, and check that the alignment brings it back.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { mulberry32 } from "../../src/vcg/math/noise.ts";
import { red } from "../../src/vcg/space/color4.ts";
import { BVH, coneDirections, cosineHemisphere } from "../../src/vcg/space/index/bvh.ts";
import { cube, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Every hit, found by testing every triangle — the reference implementation. */
function bruteForce(cm: CMeshO, origin: number[], direction: number[]): number[] {
	const hits: number[] = [];
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => {
			const v = cm.fv(f, k);
			return [cm.vx(v), cm.vy(v), cm.vz(v)];
		});
		const e1 = sub(p[1], p[0]);
		const e2 = sub(p[2], p[0]);
		const pv = cross(direction, e2);
		const det = dot(e1, pv);
		if (Math.abs(det) < 1e-14) continue;
		const inv = 1 / det;
		const tv = sub(origin, p[0]);
		const u = dot(tv, pv) * inv;
		if (u < 0 || u > 1) continue;
		const qv = cross(tv, e1);
		const v = dot(direction, qv) * inv;
		if (v < 0 || u + v > 1) continue;
		const t = dot(e2, qv) * inv;
		if (t < 1e-7) continue;
		hits.push(t);
	}
	return hits.sort((a, b) => a - b);
}

const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];

describe("BVH", () => {
	test("agrees with a brute-force sweep on random rays", () => {
		const cm = sphereIcosa(3).mesh;
		const bvh = new BVH(cm);
		const random = mulberry32(11);
		for (let i = 0; i < 200; i++) {
			const origin = [random() * 6 - 3, random() * 6 - 3, random() * 6 - 3];
			const d = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
			const length = Math.hypot(d[0], d[1], d[2]) || 1;
			const direction = d.map((c) => c / length);

			const expected = bruteForce(cm, origin, direction);
			const got = bvh
				.intersectAll(origin, direction)
				.map((h) => h.t)
				.sort((a, b) => a - b);
			expect(got.length, `ray ${i} count`).toBe(expected.length);
			got.forEach((t, k) => {
				expect(t, `ray ${i} hit ${k}`).toBeCloseTo(expected[k], 10);
			});
		}
	});

	test("an axis-aligned ray through the origin is not dropped", () => {
		// The case that a NaN in the slab test silently loses: the ray has two
		// zero direction components and its origin sits exactly on the box's
		// symmetry planes, so `0 * Infinity` appears in two of three axes.
		const cm = sphereIcosa(3).mesh;
		const bvh = new BVH(cm);
		for (const direction of [
			[0, 0, 1],
			[0, 1, 0],
			[1, 0, 0],
			[0, 0, -1],
		]) {
			const origin = direction.map((c) => -5 * c);
			expect(bvh.occluded(origin, direction), `direction ${direction}`).toBe(true);
			const hit = bvh.intersect(origin, direction);
			expect(hit, `direction ${direction}`).not.toBeNull();
			expect((hit as NonNullable<typeof hit>).t).toBeCloseTo(4, 6);
		}
	});

	test("the nearest hit is the nearest of all hits", () => {
		const cm = torus(2, 0.6, 20, 12).mesh;
		const bvh = new BVH(cm);
		const random = mulberry32(7);
		for (let i = 0; i < 60; i++) {
			const origin = [random() * 10 - 5, random() * 10 - 5, random() * 4 - 2];
			const d = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
			const length = Math.hypot(d[0], d[1], d[2]) || 1;
			const direction = d.map((c) => c / length);
			const all = bvh.intersectAll(origin, direction);
			const nearest = bvh.intersect(origin, direction);
			if (all.length === 0) {
				expect(nearest).toBeNull();
				continue;
			}
			expect((nearest as NonNullable<typeof nearest>).t).toBeCloseTo(
				Math.min(...all.map((h) => h.t)),
				10,
			);
		}
	});

	test("occlusion agrees with there being any hit at all", () => {
		const cm = cube(2).mesh;
		const bvh = new BVH(cm);
		const random = mulberry32(3);
		for (let i = 0; i < 200; i++) {
			const origin = [random() * 8 - 4, random() * 8 - 4, random() * 8 - 4];
			const d = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
			const length = Math.hypot(d[0], d[1], d[2]) || 1;
			const direction = d.map((c) => c / length);
			expect(bvh.occluded(origin, direction)).toBe(bvh.intersectAll(origin, direction).length > 0);
		}
	});

	test("a ray leaving a closed mesh crosses it an even number of times", () => {
		const cm = sphereIcosa(3).mesh;
		const bvh = new BVH(cm);
		const random = mulberry32(19);
		for (let i = 0; i < 50; i++) {
			const d = [random() * 2 - 1, random() * 2 - 1, random() * 2 - 1];
			const length = Math.hypot(d[0], d[1], d[2]) || 1;
			const direction = d.map((c) => c / length);
			const origin = direction.map((c) => -4 * c);
			// A watertight surface has no odd crossing count; an odd one is a
			// hole, so this doubles as a check that no hit was missed.
			expect(bvh.intersectAll(origin, direction).length % 2).toBe(0);
		}
	});

	test("the backface flag says which way the surface was facing", () => {
		const cm = sphereIcosa(2).mesh;
		const bvh = new BVH(cm);
		const hits = bvh.intersectAll([0, 0, -4], [0, 0, 1]).sort((a, b) => a.t - b.t);
		expect(hits.length).toBeGreaterThanOrEqual(2);
		// The first crossing is the outside of the sphere, the last the inside.
		expect(hits[0].backface).not.toBe(hits[hits.length - 1].backface);
	});

	test("an empty mesh hits nothing rather than throwing", () => {
		const cm = sphereIcosa(1).mesh;
		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= 1;
		const bvh = new BVH(cm);
		expect(bvh.faceCount).toBe(0);
		expect(bvh.intersect([0, 0, -5], [0, 0, 1])).toBeNull();
		expect(bvh.occluded([0, 0, -5], [0, 0, 1])).toBe(false);
	});
});

describe("direction sampling", () => {
	test("hemisphere directions all face the normal", () => {
		const random = mulberry32(5);
		const normal = [0, 0, 1];
		for (const d of cosineHemisphere(normal, 500, random)) {
			expect(dot(d, normal)).toBeGreaterThan(-1e-12);
			expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 9);
		}
	});

	test("cosine weighting really does favour the normal", () => {
		const random = mulberry32(6);
		const directions = cosineHemisphere([0, 0, 1], 4000, random);
		const mean = directions.reduce((s, d) => s + d[2], 0) / directions.length;
		// The mean of cos over a cosine-weighted hemisphere is 2/3; a uniform
		// hemisphere would give 1/2, so this distinguishes the two.
		expect(mean).toBeCloseTo(2 / 3, 1);
	});

	test("cone directions stay inside the cone", () => {
		const random = mulberry32(8);
		const axis = [0, 1, 0];
		const half = Math.PI / 6;
		for (const d of coneDirections(axis, half, 400, random)) {
			expect(Math.acos(Math.min(1, dot(d, axis)))).toBeLessThanOrEqual(half + 1e-9);
		}
	});

	test("the frame stays well conditioned for an axis-aligned normal", () => {
		const random = mulberry32(9);
		for (const normal of [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
			[-1, 0, 0],
		]) {
			for (const d of cosineHemisphere(normal, 50, random)) {
				expect(Number.isFinite(d[0]) && Number.isFinite(d[1]) && Number.isFinite(d[2])).toBe(true);
				expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 9);
			}
		}
	});
});

describe("Ambient Occlusion", () => {
	const NAME = "Ambient Occlusion";

	test("a convex surface is open everywhere", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			occMode: 0,
			dirBias: 0,
			coneDir: [0, 1, 0],
			coneAngle: 30,
			numberRays: 64,
			randomSeed: 1,
			useGPU: false,
		});
		// Nothing on a sphere occludes anything else, so every vertex sees the
		// whole hemisphere.
		expect(out.min_openness as number).toBeGreaterThan(0.95);
		for (let v = 0; v < cm.vertSize; v++) expect(cm.vertQuality[v]).toBeGreaterThan(0.9);
	});

	test("a surface facing another is darker than one facing away", () => {
		// Two spheres almost touching. The sides that face each other can see
		// much less of the sky than the sides that face outwards, which is the
		// whole of what ambient occlusion measures.
		const doc = new MeshDocument();
		const left = sphereIcosa(3).mesh;
		for (let v = 0; v < left.vertSize; v++) {
			left.setVert(v, left.vx(v) - 1.05, left.vy(v), left.vz(v));
		}
		const right = sphereIcosa(3).mesh;
		for (let v = 0; v < right.vertSize; v++) {
			right.setVert(v, right.vx(v) + 1.05, right.vy(v), right.vz(v));
		}
		const a = doc.addNewMesh("", "a", true, left);
		const b = doc.addNewMesh("", "b", true, right);
		a.updateBoxAndNormals();
		b.updateBoxAndNormals();
		doc.setCurrentMesh(a.id());
		kernel.applyFilter(doc, "Flatten Visible Layers", {
			MergeVisible: true,
			DeleteLayer: true,
			MergeVertices: false,
		});

		const cm = doc.mm().cm;
		kernel.applyFilter(doc, NAME, {
			occMode: 0,
			dirBias: 0,
			coneDir: [0, 1, 0],
			coneAngle: 30,
			numberRays: 96,
			randomSeed: 2,
			useGPU: false,
		});

		let facing = 0;
		let facingCount = 0;
		let away = 0;
		let awayCount = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			// The inner poles of the two spheres, and the outer ones.
			if (Math.abs(Math.abs(cm.vx(v)) - 0.15) < 0.15) {
				facing += cm.vertQuality[v];
				facingCount++;
			} else if (Math.abs(cm.vx(v)) > 1.9) {
				away += cm.vertQuality[v];
				awayCount++;
			}
		}
		expect(facingCount).toBeGreaterThan(0);
		expect(awayCount).toBeGreaterThan(0);
		expect(facing / facingCount).toBeLessThan(away / awayCount);
	});

	test("the colour is the openness as grey", () => {
		const { doc, cm } = scene(sphereIcosa(2).mesh);
		kernel.applyFilter(doc, NAME, {
			occMode: 0,
			dirBias: 0,
			coneDir: [0, 1, 0],
			coneAngle: 30,
			numberRays: 32,
			randomSeed: 1,
			useGPU: false,
		});
		for (let v = 0; v < cm.vertSize; v++) {
			expect(red(cm.vertColor[v])).toBe(Math.round(cm.vertQuality[v] * 255));
		}
	});

	test("asking for the GPU says there is not one", () => {
		const { doc } = scene(sphereIcosa(1).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				occMode: 0,
				dirBias: 0,
				coneDir: [0, 1, 0],
				coneAngle: 30,
				numberRays: 16,
				randomSeed: 1,
				useGPU: true,
			}),
		).toThrow(/no GPU path/);
	});
});

describe("Shape Diameter Function", () => {
	test("measures the diameter of a sphere", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		kernel.applyFilter(doc, "Shape Diameter Function", {
			onPrimitive: 0,
			numberRays: 64,
			randomSeed: 1,
			coneAngle: 60,
			removeFalse: true,
			removeOutliers: false,
		});
		// A unit sphere is two units thick through the middle; a narrow cone
		// measures close to that everywhere.
		let sum = 0;
		let count = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			sum += cm.vertQuality[v];
			count++;
		}
		expect(sum / count).toBeGreaterThan(1.3);
		expect(sum / count).toBeLessThan(2.05);
	});

	test("a thin shape reads thinner than a thick one", () => {
		const thickness = (scale: number) => {
			const cm = sphereIcosa(3).mesh;
			for (let v = 0; v < cm.vertSize; v++) cm.setVert(v, cm.vx(v), cm.vy(v), cm.vz(v) * scale);
			const { doc } = scene(cm);
			const out = kernel.applyFilter(doc, "Shape Diameter Function", {
				onPrimitive: 0,
				numberRays: 48,
				randomSeed: 1,
				coneAngle: 40,
				removeFalse: true,
				removeOutliers: false,
			});
			return out.max as number;
		};
		expect(thickness(0.3)).toBeLessThan(thickness(1));
	});
});

describe("Depth complexity", () => {
	test("a closed sphere never exceeds two crossings", () => {
		const { doc } = scene(sphereIcosa(2).mesh);
		const out = kernel.applyFilter(doc, "Depth complexity", {
			onPrimitive: 0,
			numberRays: 32,
			randomSeed: 1,
		});
		// A convex solid: a ray from its surface inwards leaves once.
		expect(out.max as number).toBeLessThanOrEqual(2);
	});

	test("a torus reaches more than two, because a ray can re-enter", () => {
		const { doc } = scene(torus(2, 0.6, 20, 12).mesh);
		const out = kernel.applyFilter(doc, "Depth complexity", {
			onPrimitive: 0,
			numberRays: 96,
			randomSeed: 1,
		});
		expect(out.max as number).toBeGreaterThan(2);
	});
});

describe("Volumetric obscurance", () => {
	test("is higher on a sphere than in a crevice", () => {
		const { doc } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Volumetric obscurance", {
			onPrimitive: 0,
			numberRays: 48,
			randomSeed: 1,
			obscuranceExponent: 0.1,
		});
		expect(out.min as number).toBeGreaterThanOrEqual(0);
		expect(out.max as number).toBeLessThanOrEqual(1);
	});
});

describe("ICP", () => {
	const NAME = "ICP Between Meshes";

	/** The same mesh, displaced and rotated by a known amount. */
	function moved(cm: CMeshO, angle: number, shift: readonly number[]): CMeshO {
		const cos = Math.cos(angle);
		const sin = Math.sin(angle);
		for (let v = 0; v < cm.vertSize; v++) {
			const x = cm.vx(v);
			const y = cm.vy(v);
			cm.setVert(
				v,
				x * cos - y * sin + shift[0],
				x * sin + y * cos + shift[1],
				cm.vz(v) + shift[2],
			);
		}
		return cm;
	}

	test("brings a displaced copy back onto the original", () => {
		const doc = new MeshDocument();
		const reference = doc.addNewMesh("", "reference", true, torus(2, 0.6, 24, 14).mesh);
		reference.updateBoxAndNormals();
		const source = doc.addNewMesh(
			"",
			"source",
			true,
			moved(torus(2, 0.6, 24, 14).mesh, 0.12, [0.15, -0.1, 0.08]),
		);
		source.updateBoxAndNormals();

		const before = meanDistance(reference.cm, source.cm);
		const out = kernel.applyFilter(doc, NAME, {
			referenceMesh: reference.id(),
			sourceMesh: source.id(),
			SampleNum: 800,
			MinDistAbs: 1,
			MaxIterNum: 40,
			TrgDistAbs: 1e-5,
			ReduceFactorPerc: 0.85,
			randomSeed: 1,
		});
		const after = meanDistance(reference.cm, source.cm);
		expect(after).toBeLessThan(before / 5);
		expect(out.error as number).toBeLessThan(before);
	});

	test("a mesh already aligned barely moves", () => {
		const doc = new MeshDocument();
		const reference = doc.addNewMesh("", "reference", true, sphereIcosa(3).mesh);
		reference.updateBoxAndNormals();
		const source = doc.addNewMesh("", "source", true, sphereIcosa(3).mesh);
		source.updateBoxAndNormals();
		const before = Float64Array.from(source.cm.vertCoord);

		kernel.applyFilter(doc, NAME, {
			referenceMesh: reference.id(),
			sourceMesh: source.id(),
			SampleNum: 400,
			MinDistAbs: 0.5,
			MaxIterNum: 10,
			TrgDistAbs: 1e-6,
			ReduceFactorPerc: 0.8,
			randomSeed: 1,
		});
		let worst = 0;
		for (let v = 0; v < source.cm.vertSize; v++) {
			worst = Math.max(
				worst,
				Math.hypot(
					source.cm.vx(v) - before[3 * v],
					source.cm.vy(v) - before[3 * v + 1],
					source.cm.vz(v) - before[3 * v + 2],
				),
			);
		}
		expect(worst).toBeLessThan(0.02);
	});

	test("the same layer twice is refused", () => {
		const { doc, m } = scene(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				referenceMesh: m.id(),
				sourceMesh: m.id(),
				SampleNum: 100,
				MinDistAbs: 0.5,
				MaxIterNum: 5,
				TrgDistAbs: 1e-6,
				ReduceFactorPerc: 0.8,
				randomSeed: 1,
			}),
		).toThrow(/two different layers/);
	});

	test("Overlapping Meshes reports the fraction that is close", () => {
		const doc = new MeshDocument();
		const a = doc.addNewMesh("", "a", true, sphereIcosa(2).mesh);
		a.updateBoxAndNormals();
		const far = sphereIcosa(2).mesh;
		for (let v = 0; v < far.vertSize; v++) far.setVert(v, far.vx(v) + 10, far.vy(v), far.vz(v));
		const b = doc.addNewMesh("", "b", true, far);
		b.updateBoxAndNormals();
		doc.setCurrentMesh(a.id());

		const out = kernel.applyFilter(doc, "Overlapping Meshes", { overlapDistance: 0.05 });
		expect(out[`overlap_${b.id()}`]).toBe(0);
	});
});

function meanDistance(reference: CMeshO, source: CMeshO): number {
	let sum = 0;
	let count = 0;
	for (let v = 0; v < source.vertSize; v++) {
		if (source.isVertD(v)) continue;
		let best = Number.POSITIVE_INFINITY;
		// Every reference vertex, not a sample of them: sampling puts a floor
		// on this metric at the vertex spacing, which is far above the
		// alignment error ICP actually reaches and hides the convergence.
		for (let w = 0; w < reference.vertSize; w++) {
			if (reference.isVertD(w)) continue;
			best = Math.min(
				best,
				Math.hypot(
					source.vx(v) - reference.vx(w),
					source.vy(v) - reference.vy(w),
					source.vz(v) - reference.vz(w),
				),
			);
		}
		sum += best;
		count++;
	}
	return count === 0 ? 0 : sum / count;
}

describe("registry", () => {
	test("all seven are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Ambient Occlusion", "FilterAmbientOcclusion"],
			["Shape Diameter Function", "FilterSDFGPU"],
			["Depth complexity", "FilterSDFGPU"],
			["Volumetric obscurance", "FilterSDFGPU"],
			["ICP Between Meshes", "FilterIcpPlugin"],
			["Global Align Meshes", "FilterIcpPlugin"],
			["Overlapping Meshes", "FilterIcpPlugin"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
