/**
 * Stage A of the texture-defragmentation port: the pieces with exact answers.
 *
 * These three modules are the only part of that algorithm that can be checked
 * against something other than "it looks plausible", so they are checked hard:
 *
 * - a matching fit must recover a *known* transform exactly from noiseless
 *   points, and must be the least-squares optimum when the points are noisy;
 * - segment intersection must agree with brute force on random inputs, which
 *   is what the grid acceleration could silently break;
 * - ARAP energy must be exactly zero on a map that is already an isometry, and
 *   its fitting energy must decrease on every single iteration.
 */
import { describe, expect, test } from "bun:test";
import {
	arap2D,
	arapEnergy,
	arapFittingEnergy,
	localIsometry,
	targetShapesFrom2D,
	targetShapesFrom3D,
} from "../../src/vcg/complex/parametrization/arap2d.ts";
import {
	applyMatching,
	type MatchingTransform,
	matchAffine,
	matchingError,
	matchRigid,
	matchSimilarity,
} from "../../src/vcg/complex/parametrization/matching2.ts";
import {
	closestRotation2,
	determinant2,
	invert2,
	multiply2,
	rotation2,
	svd2,
} from "../../src/vcg/math/mat2.ts";
import {
	crossIntersections,
	type Segment2,
	segmentIntersection,
	selfIntersections,
} from "../../src/vcg/space/intersection2.ts";

/** A deterministic generator, so a failure can be reproduced from its seed. */
function rng(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state * 1664525 + 1013904223) >>> 0;
		return state / 4294967296;
	};
}

// ------------------------------------------------------------------- mat2

describe("2x2 linear algebra", () => {
	const random = rng(7);
	const randomMat = (): [number, number, number, number] => [
		random() * 4 - 2,
		random() * 4 - 2,
		random() * 4 - 2,
		random() * 4 - 2,
	];

	test("the SVD reconstructs the matrix it decomposed", () => {
		for (let trial = 0; trial < 200; trial++) {
			const m = randomMat();
			const { u, v, s0, s1 } = svd2(m);
			// M = U · diag(s0, s1) · Vᵀ
			const reconstructed = multiply2(multiply2(rotation2(u), [s0, 0, 0, s1]), [
				rotation2(v)[0],
				rotation2(v)[2],
				rotation2(v)[1],
				rotation2(v)[3],
			]);
			for (let i = 0; i < 4; i++) expect(reconstructed[i]).toBeCloseTo(m[i], 10);
		}
	});

	test("the second singular value is negative exactly when orientation flips", () => {
		for (let trial = 0; trial < 200; trial++) {
			const m = randomMat();
			const { s1 } = svd2(m);
			expect(Math.sign(s1)).toBe(Math.sign(determinant2(m)));
		}
	});

	test("the closest rotation to a rotation is itself", () => {
		for (const angle of [0, 0.3, 1.2, -2.5, Math.PI]) {
			const r = rotation2(angle);
			const found = closestRotation2(r);
			for (let i = 0; i < 4; i++) expect(found[i]).toBeCloseTo(r[i], 12);
		}
	});

	test("the closest rotation is a rotation, and is the closest one", () => {
		for (let trial = 0; trial < 100; trial++) {
			const m = randomMat();
			const r = closestRotation2(m);
			expect(determinant2(r)).toBeCloseTo(1, 12);
			// Nothing nearby does better. Frobenius distance to m, swept.
			const distance = (q: readonly number[]): number =>
				Math.hypot(q[0] - m[0], q[1] - m[1], q[2] - m[2], q[3] - m[3]);
			const best = distance(r);
			const base = Math.atan2(r[2], r[0]);
			for (let k = -20; k <= 20; k++) {
				if (k === 0) continue;
				expect(distance(rotation2(base + k * 0.01))).toBeGreaterThanOrEqual(best - 1e-12);
			}
		}
	});

	test("the inverse inverts, and singular matrices report themselves", () => {
		for (let trial = 0; trial < 100; trial++) {
			const m = randomMat();
			const inverse = invert2(m);
			expect(inverse).not.toBeNull();
			const product = multiply2(m, inverse as [number, number, number, number]);
			for (const [i, want] of [1, 0, 0, 1].entries()) {
				expect(product[i]).toBeCloseTo(want, 9);
			}
		}
		expect(invert2([1, 2, 2, 4])).toBeNull();
	});
});

// --------------------------------------------------------------- matching

describe("plane matching", () => {
	const random = rng(99);
	const randomPoints = (n: number): Float64Array => {
		const p = new Float64Array(2 * n);
		for (let i = 0; i < 2 * n; i++) p[i] = random() * 10 - 5;
		return p;
	};
	const transformed = (t: MatchingTransform, p: Float64Array): Float64Array => {
		const out = new Float64Array(p.length);
		for (let i = 0; i < p.length / 2; i++) {
			const [x, y] = applyMatching(t, p[2 * i], p[2 * i + 1]);
			out[2 * i] = x;
			out[2 * i + 1] = y;
		}
		return out;
	};

	test("the rigid fit recovers a known rotation and translation exactly", () => {
		for (const angle of [0.4, -1.1, 2.9]) {
			const known: MatchingTransform = { m: rotation2(angle), tx: 3.5, ty: -2 };
			const source = randomPoints(12);
			const target = transformed(known, source);
			const found = matchRigid(target, source);
			expect(matchingError(found, target, source)).toBeLessThan(1e-12);
			for (let i = 0; i < 4; i++) expect(found.m[i]).toBeCloseTo(known.m[i], 10);
			expect(found.tx).toBeCloseTo(known.tx, 10);
			expect(found.ty).toBeCloseTo(known.ty, 10);
		}
	});

	test("the similarity fit recovers a known scale too", () => {
		const r = rotation2(0.8);
		const scale = 2.75;
		const known: MatchingTransform = {
			m: [scale * r[0], scale * r[1], scale * r[2], scale * r[3]],
			tx: -4,
			ty: 9,
		};
		const source = randomPoints(20);
		const target = transformed(known, source);
		const found = matchSimilarity(target, source);
		expect(matchingError(found, target, source)).toBeLessThan(1e-12);
		for (let i = 0; i < 4; i++) expect(found.m[i]).toBeCloseTo(known.m[i], 9);
	});

	test("the affine fit recovers a known shear, which the others cannot", () => {
		const known: MatchingTransform = { m: [1.4, 0.7, -0.2, 0.9], tx: 1, ty: 1 };
		const source = randomPoints(30);
		const target = transformed(known, source);
		expect(matchingError(matchAffine(target, source), target, source)).toBeLessThan(1e-12);
		// A shear is not a similarity, so the rigid fits must leave error behind.
		expect(matchingError(matchSimilarity(target, source), target, source)).toBeGreaterThan(1e-3);
	});

	test("a reflection is not fitted by a rotation", () => {
		// The one case where a sign slip would go unnoticed: fitting a mirrored
		// point set must leave error rather than silently producing det = -1.
		const source = randomPoints(15);
		const target = transformed({ m: [1, 0, 0, -1], tx: 0, ty: 0 }, source);
		const found = matchRigid(target, source);
		expect(determinant2(found.m)).toBeCloseTo(1, 10);
		expect(matchingError(found, target, source)).toBeGreaterThan(1e-3);
	});

	test("each fit is the best of its own kind under noise", () => {
		const known: MatchingTransform = { m: rotation2(0.6), tx: 2, ty: -1 };
		const source = randomPoints(40);
		const target = transformed(known, source);
		for (let i = 0; i < target.length; i++) target[i] += (random() - 0.5) * 0.2;

		const rigid = matchRigid(target, source);
		const best = matchingError(rigid, target, source);
		// Perturbing the answer in any direction makes it worse.
		const base = Math.atan2(rigid.m[2], rigid.m[0]);
		for (const d of [-0.05, -0.01, 0.01, 0.05]) {
			const nudged = { ...rigid, m: rotation2(base + d) };
			expect(matchingError(nudged, target, source)).toBeGreaterThan(best);
		}
		for (const d of [-0.05, 0.05]) {
			expect(matchingError({ ...rigid, tx: rigid.tx + d }, target, source)).toBeGreaterThan(best);
		}
		// And the freer fits are never worse than the more constrained ones.
		expect(matchingError(matchSimilarity(target, source), target, source)).toBeLessThanOrEqual(
			best + 1e-12,
		);
		expect(matchingError(matchAffine(target, source), target, source)).toBeLessThanOrEqual(
			matchingError(matchSimilarity(target, source), target, source) + 1e-9,
		);
	});

	test("collinear points give a finite answer rather than an enormous one", () => {
		// The degenerate case that would otherwise divide by a determinant near
		// zero and return a transform with 1e15 in it.
		const source = new Float64Array([0, 0, 1, 0, 2, 0, 3, 0]);
		const target = new Float64Array([0, 0, 2, 0, 4, 0, 6, 0]);
		for (const fit of [matchAffine, matchSimilarity, matchRigid]) {
			const found = fit(target, source);
			for (const value of [...found.m, found.tx, found.ty]) {
				expect(Number.isFinite(value)).toBe(true);
				expect(Math.abs(value)).toBeLessThan(1e6);
			}
		}
	});

	test("refuses mismatched or too-short input", () => {
		expect(() => matchRigid(new Float64Array(4), new Float64Array(6))).toThrow();
		expect(() => matchRigid(new Float64Array(2), new Float64Array(2))).toThrow();
	});
});

// ----------------------------------------------------------- intersection

describe("segment intersection", () => {
	test("crossing, touching and missing are told apart", () => {
		const a: Segment2 = { x0: 0, y0: 0, x1: 2, y1: 2 };
		expect(segmentIntersection(a, { x0: 0, y0: 2, x1: 2, y1: 0 })).not.toBeNull();
		expect(segmentIntersection(a, { x0: 3, y0: 0, x1: 4, y1: 1 })).toBeNull();
		// Parallel and collinear are both "no proper crossing".
		expect(segmentIntersection(a, { x0: 1, y0: 0, x1: 3, y1: 2 })).toBeNull();
		expect(segmentIntersection(a, { x0: 1, y0: 1, x1: 3, y1: 3 })).toBeNull();
	});

	test("the crossing point is on both segments", () => {
		const random = rng(11);
		for (let trial = 0; trial < 500; trial++) {
			const seg = (): Segment2 => ({
				x0: random() * 10,
				y0: random() * 10,
				x1: random() * 10,
				y1: random() * 10,
			});
			const a = seg();
			const b = seg();
			const hit = segmentIntersection(a, b);
			if (hit === null) continue;
			for (const s of [a, b]) {
				const t =
					Math.abs(s.x1 - s.x0) > Math.abs(s.y1 - s.y0)
						? (hit[0] - s.x0) / (s.x1 - s.x0)
						: (hit[1] - s.y0) / (s.y1 - s.y0);
				expect(t).toBeGreaterThanOrEqual(-1e-9);
				expect(t).toBeLessThanOrEqual(1 + 1e-9);
			}
		}
	});

	test("the grid finds exactly what brute force finds", () => {
		// The whole point of the acceleration is that it changes nothing. A
		// segment straddling a cell boundary is where it would go wrong.
		const random = rng(2024);
		for (const count of [4, 20, 80]) {
			const segments: Segment2[] = [];
			for (let i = 0; i < count; i++) {
				const x0 = random() * 10;
				const y0 = random() * 10;
				segments.push({ x0, y0, x1: x0 + random() * 4 - 2, y1: y0 + random() * 4 - 2 });
			}
			const brute = new Set<string>();
			for (let i = 0; i < count; i++) {
				for (let j = i + 1; j < count; j++) {
					const a = segments[i];
					const b = segments[j];
					const shares =
						(a.x0 === b.x0 && a.y0 === b.y0) ||
						(a.x0 === b.x1 && a.y0 === b.y1) ||
						(a.x1 === b.x0 && a.y1 === b.y0) ||
						(a.x1 === b.x1 && a.y1 === b.y1);
					if (!shares && segmentIntersection(a, b) !== null) brute.add(`${i},${j}`);
				}
			}
			const found = new Set(selfIntersections(segments).map(([i, j]) => `${i},${j}`));
			expect(found, `count ${count}`).toEqual(brute);
		}
	});

	test("each crossing pair is reported once", () => {
		// Upstream reports a pair once per shared grid cell; the callers count
		// these, so a duplicate is a wrong answer, not noise.
		const long: Segment2[] = [
			{ x0: -100, y0: 0, x1: 100, y1: 1 },
			{ x0: 0, y0: -100, x1: 1, y1: 100 },
		];
		expect(selfIntersections(long).length).toBe(1);
	});

	test("segments meeting at a shared endpoint are not crossings", () => {
		// Mesh edges share endpoints constantly; treating that as an overlap
		// would reject every merge.
		const fan: Segment2[] = [
			{ x0: 0, y0: 0, x1: 1, y1: 0 },
			{ x0: 0, y0: 0, x1: 0, y1: 1 },
			{ x0: 0, y0: 0, x1: -1, y1: -1 },
		];
		expect(selfIntersections(fan)).toEqual([]);
	});

	test("cross intersection ignores crossings inside either list", () => {
		const first: Segment2[] = [
			{ x0: 0, y0: 0, x1: 2, y1: 2 },
			{ x0: 0, y0: 2, x1: 2, y1: 0 }, // crosses the one above, but same list
		];
		const second: Segment2[] = [{ x0: 1, y0: -1, x1: 1, y1: 3 }];
		const pairs = crossIntersections(first, second);
		expect(pairs.length).toBe(2);
		for (const [a, b] of pairs) {
			expect(a).toBeLessThan(first.length);
			expect(b).toBeLessThan(second.length);
		}
	});

	test("empty and single-segment inputs are answered, not crashed on", () => {
		expect(selfIntersections([])).toEqual([]);
		expect(selfIntersections([{ x0: 0, y0: 0, x1: 1, y1: 1 }])).toEqual([]);
		expect(crossIntersections([], [{ x0: 0, y0: 0, x1: 1, y1: 1 }])).toEqual([]);
	});
});

// ------------------------------------------------------------------ ARAP

describe("2D ARAP", () => {
	/** A regular triangulated grid, with its 3D positions flat in the z=0 plane. */
	function flatGrid(n: number) {
		const positions = new Float64Array(3 * (n + 1) * (n + 1));
		const uv = new Float64Array(2 * (n + 1) * (n + 1));
		const index = (i: number, j: number): number => j * (n + 1) + i;
		for (let j = 0; j <= n; j++) {
			for (let i = 0; i <= n; i++) {
				const v = index(i, j);
				positions[3 * v] = i;
				positions[3 * v + 1] = j;
				uv[2 * v] = i;
				uv[2 * v + 1] = j;
			}
		}
		const faces = new Int32Array(6 * n * n);
		let f = 0;
		for (let j = 0; j < n; j++) {
			for (let i = 0; i < n; i++) {
				faces.set([index(i, j), index(i + 1, j), index(i + 1, j + 1)], 3 * f++);
				faces.set([index(i, j), index(i + 1, j + 1), index(i, j + 1)], 3 * f++);
			}
		}
		const boundary = new Map<number, readonly [number, number]>();
		for (let j = 0; j <= n; j++) {
			for (let i = 0; i <= n; i++) {
				if (i === 0 || j === 0 || i === n || j === n) {
					boundary.set(index(i, j), [i, j]);
				}
			}
		}
		return { positions, uv, faces, boundary, vertexCount: (n + 1) * (n + 1) };
	}

	test("the local isometry preserves both edge lengths and the angle", () => {
		const v1 = [1, 2, 3];
		const v2 = [-2, 0.5, 1];
		const [x1, y1, x2, y2] = localIsometry(v1, v2);
		expect(Math.hypot(x1, y1)).toBeCloseTo(Math.hypot(...v1), 12);
		expect(Math.hypot(x2, y2)).toBeCloseTo(Math.hypot(...v2), 12);
		const dot3 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
		expect(x1 * x2 + y1 * y2).toBeCloseTo(dot3, 10);
	});

	test("the energy is exactly zero on a map that is already an isometry", () => {
		const { faces, positions, uv } = flatGrid(4);
		const target = targetShapesFrom3D(faces, positions);
		expect(arapEnergy(faces, target, uv).energy).toBeLessThan(1e-20);
		expect(arapFittingEnergy(faces, target, uv)).toBeLessThan(1e-20);
	});

	test("the energy is still zero after a rigid motion of the whole layout", () => {
		// Rotating and translating a parametrization changes no distortion, and
		// an energy that noticed would be measuring the wrong thing.
		const { faces, positions, uv } = flatGrid(4);
		const target = targetShapesFrom3D(faces, positions);
		const r = rotation2(0.7);
		for (let v = 0; v < uv.length / 2; v++) {
			const [x, y] = [uv[2 * v], uv[2 * v + 1]];
			uv[2 * v] = r[0] * x + r[1] * y + 12;
			uv[2 * v + 1] = r[2] * x + r[3] * y - 5;
		}
		expect(arapEnergy(faces, target, uv).energy).toBeLessThan(1e-18);
	});

	test("a uniform scale costs the expected amount", () => {
		// Both singular values become s, so the energy must be exactly 2(s-1)².
		const { faces, positions, uv } = flatGrid(3);
		const target = targetShapesFrom3D(faces, positions);
		const s = 1.5;
		for (let i = 0; i < uv.length; i++) uv[i] *= s;
		expect(arapEnergy(faces, target, uv).energy).toBeCloseTo(2 * (s - 1) ** 2, 10);
	});

	test("the fitting energy decreases on every iteration", () => {
		// The property that makes local/global a descent — checked step by step,
		// not just between the endpoints.
		const { faces, positions, uv, boundary, vertexCount } = flatGrid(5);
		const target = targetShapesFrom3D(faces, positions);
		const random = rng(3);
		for (let v = 0; v < vertexCount; v++) {
			if (boundary.has(v)) continue;
			uv[2 * v] += random() - 0.5;
			uv[2 * v + 1] += random() - 0.5;
		}

		let previous = arapFittingEnergy(faces, target, uv);
		for (let step = 0; step < 8; step++) {
			arap2D(faces, vertexCount, target, uv, boundary, { maxIterations: 1, tolerance: 0 });
			const current = arapFittingEnergy(faces, target, uv);
			expect(current, `step ${step}`).toBeLessThanOrEqual(previous + 1e-9);
			previous = current;
		}
	});

	test("a scrambled interior is pulled back to the isometry it came from", () => {
		const { faces, positions, uv, boundary, vertexCount } = flatGrid(5);
		const target = targetShapesFrom3D(faces, positions);
		const random = rng(21);
		for (let v = 0; v < vertexCount; v++) {
			if (boundary.has(v)) continue;
			uv[2 * v] += (random() - 0.5) * 0.8;
			uv[2 * v + 1] += (random() - 0.5) * 0.8;
		}
		const before = arapEnergy(faces, target, uv).energy;
		// Run to the iteration cap rather than the early-out, so what is measured
		// is where the algorithm converges rather than where it stopped early.
		const result = arap2D(faces, vertexCount, target, uv, boundary, {
			maxIterations: 200,
			tolerance: 0,
		});
		expect(result.finalEnergy).toBeLessThan(before);
		// The boundary pins the exact original layout, and the interior of a flat
		// grid has one isometric completion, so it should come back to it. The
		// floor is the solver's, not the algorithm's: energy is quadratic in the
		// position error, so vertices right to 1e-6 leave energy around 1e-9.
		expect(result.finalEnergy).toBeLessThan(1e-8);
		for (let v = 0; v < vertexCount; v++) {
			const j = Math.floor(v / 6);
			const i = v % 6;
			expect(uv[2 * v]).toBeCloseTo(i, 6);
			expect(uv[2 * v + 1]).toBeCloseTo(j, 6);
		}
	});

	test("pinned vertices do not move", () => {
		const { faces, positions, uv, boundary, vertexCount } = flatGrid(4);
		const target = targetShapesFrom3D(faces, positions);
		for (let i = 0; i < uv.length; i++) uv[i] += 0.3;
		arap2D(faces, vertexCount, target, uv, boundary, { maxIterations: 5 });
		for (const [v, position] of boundary) {
			expect(uv[2 * v]).toBeCloseTo(position[0], 12);
			expect(uv[2 * v + 1]).toBeCloseTo(position[1], 12);
		}
	});

	test("a curved surface cannot reach zero, and says so", () => {
		// A patch of a sphere has Gaussian curvature, so no isometry into the
		// plane exists. The energy must settle above zero rather than pretend.
		const n = 5;
		const { faces, uv, boundary, vertexCount } = flatGrid(n);
		const positions = new Float64Array(3 * vertexCount);
		for (let v = 0; v < vertexCount; v++) {
			const i = (v % (n + 1)) / n - 0.5;
			const j = Math.floor(v / (n + 1)) / n - 0.5;
			positions[3 * v] = i;
			positions[3 * v + 1] = j;
			positions[3 * v + 2] = Math.sqrt(Math.max(0, 1 - i * i - j * j));
		}
		const target = targetShapesFrom3D(faces, positions);
		const result = arap2D(faces, vertexCount, target, uv, boundary);
		expect(result.finalEnergy).toBeGreaterThan(1e-6);
		expect(Number.isFinite(result.finalEnergy)).toBe(true);
	});

	test("target shapes taken from a 2D layout describe that layout exactly", () => {
		const { faces, uv } = flatGrid(3);
		const target = targetShapesFrom2D(faces, uv);
		expect(arapEnergy(faces, target, uv).energy).toBeLessThan(1e-20);
	});

	test("refuses to run with nothing pinned", () => {
		const { faces, positions, uv, vertexCount } = flatGrid(2);
		const target = targetShapesFrom3D(faces, positions);
		expect(() => arap2D(faces, vertexCount, target, uv, new Map())).toThrow(/pinned/);
	});
});
