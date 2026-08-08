/**
 * 2×2 matrices, in closed form.
 *
 * Everything the texture-atlas work needs from linear algebra in the plane —
 * the closest rotation to a linear map, the singular values that say how much
 * it stretches, the inverse — has a closed form at this size. No iteration and
 * no library: the 2×2 SVD is four `atan2`s and two square roots.
 *
 * A matrix is four numbers in row-major order, `[a, b, c, d]` meaning
 * `[[a, b], [c, d]]`.
 */

/** `[a, b, c, d]` for `[[a, b], [c, d]]`. */
export type Mat2 = readonly [number, number, number, number];

export const IDENTITY2: Mat2 = [1, 0, 0, 1];

export function determinant2(m: Mat2): number {
	return m[0] * m[3] - m[1] * m[2];
}

export function multiply2(a: Mat2, b: Mat2): Mat2 {
	return [
		a[0] * b[0] + a[1] * b[2],
		a[0] * b[1] + a[1] * b[3],
		a[2] * b[0] + a[3] * b[2],
		a[2] * b[1] + a[3] * b[3],
	];
}

export function transpose2(m: Mat2): Mat2 {
	return [m[0], m[2], m[1], m[3]];
}

export function apply2(m: Mat2, x: number, y: number): [number, number] {
	return [m[0] * x + m[1] * y, m[2] * x + m[3] * y];
}

/** The inverse, or null when the matrix is singular to within `eps`. */
export function invert2(m: Mat2, eps = 0): Mat2 | null {
	const det = determinant2(m);
	if (Math.abs(det) <= eps) return null;
	return [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det];
}

/** A rotation by `angle`, anticlockwise. */
export function rotation2(angle: number): Mat2 {
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	return [c, -s, s, c];
}

/**
 * The singular value decomposition, `M = U · diag(σ₀, σ₁) · Vᵀ`.
 *
 * `U` and `V` come back as rotation *angles* rather than matrices, because
 * both are proper rotations in the 2×2 case and every caller here wants either
 * the product `U Vᵀ` or the singular values alone.
 *
 * The second singular value is returned **signed**: negative exactly when the
 * map reverses orientation. Callers wanting the conventional non-negative
 * singular values take absolute values; callers asking "did this triangle
 * flip" read the sign, which is otherwise lost.
 */
export interface Svd2 {
	/** The angle of `U`. */
	readonly u: number;
	/** The angle of `V`. */
	readonly v: number;
	/** Always non-negative, and the larger in magnitude. */
	readonly s0: number;
	/** Signed: negative when the map reverses orientation. */
	readonly s1: number;
}

export function svd2(m: Mat2): Svd2 {
	const [a, b, c, d] = m;
	// Writing out `U(φ) diag(σ₀, σ₁) V(θ)ᵀ` entry by entry gives four sums that
	// separate into the two angles:
	//   a + d = (σ₀ + σ₁) cos(φ − θ)      b − c = (σ₀ + σ₁) sin(θ − φ)
	//   a − d = (σ₀ − σ₁) cos(φ + θ)      b + c = (σ₀ − σ₁) sin(φ + θ)
	// so one `atan2` recovers `φ + θ`, the other `θ − φ`, and the two moduli
	// give the singular values.
	const e = (a + d) / 2;
	const f = (a - d) / 2;
	const g = (b + c) / 2;
	const h = (b - c) / 2;
	const q = Math.hypot(e, h);
	const r = Math.hypot(f, g);
	const sum = Math.atan2(g, f); // φ + θ
	const difference = Math.atan2(h, e); // θ − φ
	return {
		u: (sum - difference) / 2,
		v: (sum + difference) / 2,
		s0: q + r,
		s1: q - r,
	};
}

/**
 * The rotation closest to `m` — the rotational factor of its polar
 * decomposition.
 *
 * In two dimensions this is unconditional: with `M = U diag(σ₀, σ₁) Vᵀ` and
 * `U`, `V` proper rotations, `U Vᵀ` maximises `tr(Rᵀ M)` whenever `σ₀ + σ₁ ≥ 0`,
 * which the decomposition above always gives. So unlike the 3×3 case there is
 * no reflection to detect and correct — the correction is already inside the
 * sign of `σ₁`.
 *
 * This one function is what ARAP's local step, the rigid matching fit and the
 * similarity fit all reduce to.
 */
export function closestRotation2(m: Mat2): Mat2 {
	const { u, v } = svd2(m);
	return rotation2(u - v);
}

/**
 * Solves the symmetric 2×2 system `A x = b`, in the least-squares sense when
 * `A` is singular.
 *
 * The singular case is real: it is what a set of matching points strung out
 * along a line looks like. Rather than dividing by a determinant near zero and
 * returning something enormous, the rank-deficient direction is dropped — the
 * minimum-norm solution, which for a collinear point set means "any transform
 * consistent with the line", not "a transform with 10¹² in it".
 */
export function solveSymmetric2(a: Mat2, bx: number, by: number, eps = 1e-12): [number, number] {
	const det = determinant2(a);
	const scale = Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]), Math.abs(a[3]), 1e-300);
	if (Math.abs(det) > eps * scale * scale) {
		return [(a[3] * bx - a[1] * by) / det, (a[0] * by - a[2] * bx) / det];
	}

	// Rank-deficient: project onto the one direction the matrix does span.
	// For a symmetric 2×2 that direction is the eigenvector of the larger
	// eigenvalue, and both are available in closed form.
	const tr = a[0] + a[3];
	const disc = Math.sqrt(Math.max(0, ((a[0] - a[3]) / 2) ** 2 + a[1] * a[2]));
	const lambda = tr / 2 + disc;
	if (Math.abs(lambda) <= eps * scale) return [0, 0];
	let ex = a[1];
	let ey = lambda - a[0];
	if (Math.hypot(ex, ey) < eps * scale) {
		ex = lambda - a[3];
		ey = a[2];
	}
	const norm = Math.hypot(ex, ey);
	if (norm === 0) return [0, 0];
	ex /= norm;
	ey /= norm;
	const coeff = (ex * bx + ey * by) / lambda;
	return [coeff * ex, coeff * ey];
}

export const Mat2Ops = {
	determinant2,
	multiply2,
	transpose2,
	apply2,
	invert2,
	rotation2,
	svd2,
	closestRotation2,
	solveSymmetric2,
} as const;
