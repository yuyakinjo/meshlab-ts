/**
 * As-rigid-as-possible flattening, in the plane.
 *
 * Every triangle has a shape it "wants": the shape it has on the surface,
 * carried into 2D by an isometry. A parametrization gives it a different one.
 * ARAP alternates between two steps — find the rotation that best explains each
 * triangle's current shape given the one it wants (local), then move the
 * vertices to best satisfy all those rotations at once (global) — and each step
 * decreases the same energy, so the alternation converges.
 *
 * The atlas work uses this to answer "if these two charts were merged and the
 * seam between them relaxed, how much distortion would that cost?". The
 * boundary of the patch is pinned, the inside is allowed to move, and the
 * energy afterwards is the price of the merge.
 *
 * This is deliberately free of `CMeshO`: the input is a triangle list, a target
 * shape per triangle and a UV per vertex. Charts are cut meshes with their own
 * vertex numbering, and threading a mesh type through would mean building one
 * for every candidate merge.
 *
 * **Two divergences from upstream, both in how the linear system is posed.**
 *
 * 1. Upstream leaves the pinned vertices in the matrix as identity rows, which
 *    makes it non-symmetric, and factorises with a general LU. Here the pinned
 *    values are eliminated into the right-hand side by `SparseMatrix.pin`,
 *    which keeps the system symmetric positive definite so conjugate gradients
 *    applies. Same solution, better conditioning, no factorisation.
 * 2. Upstream reports a singular-value energy, `Σ area·((σ₀-1)² + (σ₁-1)²)`,
 *    which is *not* the quadratic energy the local/global iteration descends.
 *    Both are reported here and named for what they are: {@link arapEnergy} is
 *    upstream's measure of distortion, {@link arapFittingEnergy} is the one
 *    that provably decreases every iteration. They vanish together, exactly
 *    when the map is an isometry.
 */
import { closestRotation2, invert2, type Mat2, multiply2, svd2 } from "../../math/mat2.ts";
import { SparseMatrix, solveCG } from "../../math/sparse.ts";

/**
 * A triangle's target shape: the two edge vectors `x₁₀` and `x₂₀` of the source
 * triangle, laid flat. Four numbers per face.
 */
export type TargetShapes = Float64Array;

/**
 * Lays a triangle's two edge vectors flat, preserving both lengths and the
 * angle between them.
 *
 * Any isometric placement will do — the energy only ever sees the triangle's
 * shape — so the first edge is put along the x axis.
 */
export function localIsometry(
	v1: readonly number[],
	v2: readonly number[],
): [number, number, number, number] {
	const n1 = Math.hypot(v1[0], v1[1], v1[2]);
	const n2 = Math.hypot(v2[0], v2[1], v2[2]);
	// A degenerate edge has no direction; upstream substitutes a tiny length
	// rather than propagating a zero into the cotangent weights below.
	const l1 = n1 === 0 ? 1e-6 : n1;
	const l2 = n2 === 0 ? 1e-6 : n2;
	const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	let theta = Math.acos(Math.max(-1, Math.min(1, dot / (l1 * l2))));
	// A flat or reversed triangle would give a zero-area target and an infinite
	// cotangent; nudged off the boundary, as upstream does.
	if (!(theta > 0 && theta < Math.PI)) theta = theta === 0 ? 1e-3 : Math.PI - 1e-3;
	return [l1, 0, l2 * Math.cos(theta), l2 * Math.sin(theta)];
}

/** The target shapes of a triangle list, taken from its 3D positions. */
export function targetShapesFrom3D(faces: Int32Array, positions: Float64Array): TargetShapes {
	const faceCount = faces.length / 3;
	const out = new Float64Array(4 * faceCount);
	for (let f = 0; f < faceCount; f++) {
		const [a, b, c] = [0, 1, 2].map((k) => faces[3 * f + k]);
		const e1 = [0, 1, 2].map((i) => positions[3 * b + i] - positions[3 * a + i]);
		const e2 = [0, 1, 2].map((i) => positions[3 * c + i] - positions[3 * a + i]);
		out.set(localIsometry(e1, e2), 4 * f);
	}
	return out;
}

/** The target shapes of a triangle list, taken from an existing 2D layout. */
export function targetShapesFrom2D(faces: Int32Array, uv: Float64Array): TargetShapes {
	const faceCount = faces.length / 3;
	const out = new Float64Array(4 * faceCount);
	for (let f = 0; f < faceCount; f++) {
		const [a, b, c] = [0, 1, 2].map((k) => faces[3 * f + k]);
		out[4 * f] = uv[2 * b] - uv[2 * a];
		out[4 * f + 1] = uv[2 * b + 1] - uv[2 * a + 1];
		out[4 * f + 2] = uv[2 * c] - uv[2 * a];
		out[4 * f + 3] = uv[2 * c + 1] - uv[2 * a + 1];
	}
	return out;
}

/** The Jacobian of the map from a face's target shape to its current one. */
function jacobian(target: TargetShapes, uv: Float64Array, faces: Int32Array, f: number): Mat2 {
	const [a, b, c] = [0, 1, 2].map((k) => faces[3 * f + k]);
	// Columns are the edge vectors, so the map is `current · target⁻¹`.
	const x: Mat2 = [target[4 * f], target[4 * f + 2], target[4 * f + 1], target[4 * f + 3]];
	const u: Mat2 = [
		uv[2 * b] - uv[2 * a],
		uv[2 * c] - uv[2 * a],
		uv[2 * b + 1] - uv[2 * a + 1],
		uv[2 * c + 1] - uv[2 * a + 1],
	];
	const inverse = invert2(x);
	if (inverse === null) return [1, 0, 0, 1];
	return multiply2(u, inverse);
}

/** Twice the signed area of a target triangle. */
function targetArea(target: TargetShapes, f: number): number {
	return Math.abs(target[4 * f] * target[4 * f + 3] - target[4 * f + 1] * target[4 * f + 2]) / 2;
}

/**
 * The area-weighted mean distortion, `Σ area·((σ₀-1)² + (σ₁-1)²) / Σ area`.
 *
 * Zero exactly when every triangle is a rigid copy of its target — this is what
 * upstream reports and what the merge tolerances are expressed in.
 */
export function arapEnergy(
	faces: Int32Array,
	target: TargetShapes,
	uv: Float64Array,
): { energy: number; numerator: number; denominator: number } {
	let numerator = 0;
	let denominator = 0;
	for (let f = 0; f < faces.length / 3; f++) {
		const area = targetArea(target, f);
		if (area <= 0) continue;
		const { s0, s1 } = svd2(jacobian(target, uv, faces, f));
		numerator += area * ((s0 - 1) ** 2 + (Math.abs(s1) - 1) ** 2);
		denominator += area;
	}
	return {
		energy: denominator > 0 ? numerator / denominator : 0,
		numerator,
		denominator,
	};
}

/** Half the cotangent of each corner angle of a target triangle. */
function cotangents(target: TargetShapes, f: number): [number, number, number] {
	const p: Array<[number, number]> = [
		[0, 0],
		[target[4 * f], target[4 * f + 1]],
		[target[4 * f + 2], target[4 * f + 3]],
	];
	const out: [number, number, number] = [0, 0, 0];
	for (let i = 0; i < 3; i++) {
		const j = (i + 1) % 3;
		const k = (i + 2) % 3;
		const ux = p[j][0] - p[i][0];
		const uy = p[j][1] - p[i][1];
		const vx = p[k][0] - p[i][0];
		const vy = p[k][1] - p[i][1];
		const dot = ux * vx + uy * vy;
		const area2 = Math.abs(ux * vy - uy * vx);
		// A zero-area corner has an infinite cotangent. Upstream substitutes
		// 1e-8, which turns an infinite spring into a nearly absent one — the
		// choice that lets a degenerate triangle be ignored rather than dominate.
		out[i] = area2 > 0 ? (0.5 * dot) / area2 : 1e-8;
		if (!Number.isFinite(out[i])) out[i] = 1e-8;
	}
	return out;
}

/**
 * The quadratic energy the local/global iteration actually descends:
 * `Σ_f Σ_i w_ij |(u_i - u_j) - R_f (x_i - x_j)|²`.
 *
 * Not normalised and not comparable across meshes — its only use is that it
 * decreases every half-step, which is what makes the alternation a descent.
 */
export function arapFittingEnergy(
	faces: Int32Array,
	target: TargetShapes,
	uv: Float64Array,
): number {
	let total = 0;
	for (let f = 0; f < faces.length / 3; f++) {
		const r = closestRotation2(jacobian(target, uv, faces, f));
		const cot = cotangents(target, f);
		const idx = [faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]];
		const t: Array<[number, number]> = [
			[0, 0],
			[target[4 * f], target[4 * f + 1]],
			[target[4 * f + 2], target[4 * f + 3]],
		];
		for (let i = 0; i < 3; i++) {
			const j = (i + 1) % 3;
			const k = (i + 2) % 3;
			for (const [other, weight] of [
				[j, cot[k]],
				[k, cot[j]],
			] as const) {
				const dx = uv[2 * idx[i]] - uv[2 * idx[other]];
				const dy = uv[2 * idx[i] + 1] - uv[2 * idx[other] + 1];
				const tx = t[i][0] - t[other][0];
				const ty = t[i][1] - t[other][1];
				const rx = r[0] * tx + r[1] * ty;
				const ry = r[2] * tx + r[3] * ty;
				total += weight * ((dx - rx) ** 2 + (dy - ry) ** 2);
			}
		}
	}
	return total;
}

export interface Arap2DOptions {
	readonly maxIterations?: number;
	/** Stops once the fitting energy improves by less than this, relatively. */
	readonly tolerance?: number;
}

export interface Arap2DResult {
	readonly initialEnergy: number;
	readonly finalEnergy: number;
	readonly iterations: number;
	readonly converged: boolean;
	/** True when the solver produced something non-finite and was rolled back. */
	readonly numericalError: boolean;
}

/**
 * Relaxes `uv` toward an isometry of `target`, in place.
 *
 * `fixed` pins vertices at given positions; at least one must be pinned, or the
 * system is only determined up to a translation. In practice callers pin the
 * whole boundary of the patch they are relaxing.
 *
 * Returns upstream's singular-value energy before and after, which is the
 * number the merge tolerances compare against.
 */
export function arap2D(
	faces: Int32Array,
	vertexCount: number,
	target: TargetShapes,
	uv: Float64Array,
	fixed: ReadonlyMap<number, readonly [number, number]>,
	options: Arap2DOptions = {},
): Arap2DResult {
	const maxIterations = options.maxIterations ?? 100;
	const tolerance = options.tolerance ?? 1e-6;
	const faceCount = faces.length / 3;
	const initial = arapEnergy(faces, target, uv).energy;

	if (fixed.size === 0) {
		throw new Error("arap2D needs at least one pinned vertex to fix the translation");
	}
	for (const [v, position] of fixed) {
		uv[2 * v] = position[0];
		uv[2 * v + 1] = position[1];
	}

	// The cotangent weights come from the target shapes, which never change, so
	// the system matrix is the same every iteration. Only the right-hand side
	// moves — but `pin` mutates the matrix, so it is rebuilt per iteration.
	const cot: Array<[number, number, number]> = [];
	for (let f = 0; f < faceCount; f++) cot.push(cotangents(target, f));

	let previous = arapFittingEnergy(faces, target, uv);
	let iterations = 0;
	let numericalError = false;

	for (let iter = 0; iter < maxIterations; iter++) {
		iterations = iter + 1;

		// Local step: the rotation each triangle would need.
		const rotations: Mat2[] = [];
		for (let f = 0; f < faceCount; f++) {
			rotations.push(closestRotation2(jacobian(target, uv, faces, f)));
		}

		// Global step: one symmetric system, solved twice — once per coordinate,
		// since the two are coupled only through the shared matrix.
		const a = new SparseMatrix(vertexCount);
		const bu = new Float64Array(vertexCount);
		const bv = new Float64Array(vertexCount);

		for (let f = 0; f < faceCount; f++) {
			const idx = [faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]];
			const r = rotations[f];
			const t: Array<[number, number]> = [
				[0, 0],
				[target[4 * f], target[4 * f + 1]],
				[target[4 * f + 2], target[4 * f + 3]],
			];
			for (let i = 0; i < 3; i++) {
				const j = (i + 1) % 3;
				const k = (i + 2) % 3;
				const wij = cot[f][k];
				const wik = cot[f][j];
				a.add(idx[i], idx[j], -wij);
				a.add(idx[i], idx[k], -wik);
				a.add(idx[i], idx[i], wij + wik);
				for (const [other, weight] of [
					[j, wij],
					[k, wik],
				] as const) {
					const tx = t[i][0] - t[other][0];
					const ty = t[i][1] - t[other][1];
					bu[idx[i]] += weight * (r[0] * tx + r[1] * ty);
					bv[idx[i]] += weight * (r[2] * tx + r[3] * ty);
				}
			}
		}

		// Both coordinates share this matrix, so they must be pinned together —
		// pinning is destructive and the second call would find nothing left to
		// eliminate.
		for (const [v, position] of fixed) {
			a.pinMulti(v, [position[0], position[1]], [bu, bv]);
		}

		const solvedU = solveCG(a, bu, { tolerance: 1e-12 });
		const solvedV = solveCG(a, bv, { tolerance: 1e-12 });

		// A non-finite solve means the system was degenerate — a fold that
		// collapsed a triangle, most often. Keeping the last good layout is more
		// useful to the caller than a UV array full of NaN.
		let finite = true;
		for (let v = 0; v < vertexCount; v++) {
			if (!Number.isFinite(solvedU.x[v]) || !Number.isFinite(solvedV.x[v])) {
				finite = false;
				break;
			}
		}
		if (!finite) {
			numericalError = true;
			break;
		}

		for (let v = 0; v < vertexCount; v++) {
			uv[2 * v] = solvedU.x[v];
			uv[2 * v + 1] = solvedV.x[v];
		}

		const current = arapFittingEnergy(faces, target, uv);
		const improvement = previous - current;
		previous = current;
		if (improvement <= tolerance * Math.max(1, Math.abs(current))) {
			return {
				initialEnergy: initial,
				finalEnergy: arapEnergy(faces, target, uv).energy,
				iterations,
				converged: true,
				numericalError: false,
			};
		}
	}

	return {
		initialEnergy: initial,
		finalEnergy: arapEnergy(faces, target, uv).energy,
		iterations,
		converged: false,
		numericalError,
	};
}

export const Arap2D = {
	localIsometry,
	targetShapesFrom3D,
	targetShapesFrom2D,
	arapEnergy,
	arapFittingEnergy,
	arap2D,
} as const;
