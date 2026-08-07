/**
 * `Quadric` — the error metric behind QEM decimation, mirroring
 * `vcg::math::Quadric<double>`.
 *
 * A quadric is the quadratic form measuring squared distance to a set of
 * planes. Summing the quadrics of the planes around a vertex gives a function
 * whose value at any point is the total squared distance to that vertex's
 * original surface — which is exactly the error a collapse would introduce.
 *
 * Stored as the ten distinct entries of the symmetric 4×4 matrix
 * `pᵀ p` for a plane `p = (a, b, c, d)` with `a² + b² + c² = 1`, laid out flat
 * so a per-vertex quadric is ten slots in a `Float64Array` rather than an
 * object.
 */

/** Numbers per quadric. */
export const QUADRIC_STRIDE = 10;

/** Offsets within one quadric: the upper triangle of the symmetric matrix. */
const A2 = 0; // a·a
const AB = 1;
const AC = 2;
const AD = 3;
const B2 = 4; // b·b
const BC = 5;
const BD = 6;
const C2 = 7; // c·c
const CD = 8;
const D2 = 9; // d·d

/** Zeroes the quadric at slot `i`. */
export function quadricZero(q: Float64Array, i: number): void {
	q.fill(0, i * QUADRIC_STRIDE, (i + 1) * QUADRIC_STRIDE);
}

/**
 * Adds `weight · pᵀp` for the plane `(a, b, c, d)` into the quadric at `i`.
 *
 * The plane must be normalised, or the metric stops measuring distance.
 */
export function quadricAddPlane(
	q: Float64Array,
	i: number,
	a: number,
	b: number,
	c: number,
	d: number,
	weight = 1,
): void {
	const o = i * QUADRIC_STRIDE;
	q[o + A2] += weight * a * a;
	q[o + AB] += weight * a * b;
	q[o + AC] += weight * a * c;
	q[o + AD] += weight * a * d;
	q[o + B2] += weight * b * b;
	q[o + BC] += weight * b * c;
	q[o + BD] += weight * b * d;
	q[o + C2] += weight * c * c;
	q[o + CD] += weight * c * d;
	q[o + D2] += weight * d * d;
}

/** `dst += src`. */
export function quadricAdd(q: Float64Array, dst: number, src: number): void {
	const a = dst * QUADRIC_STRIDE;
	const b = src * QUADRIC_STRIDE;
	for (let k = 0; k < QUADRIC_STRIDE; k++) q[a + k] += q[b + k];
}

/** Copies the quadric at `src` over the one at `dst`. */
export function quadricCopy(q: Float64Array, dst: number, src: number): void {
	q.copyWithin(dst * QUADRIC_STRIDE, src * QUADRIC_STRIDE, (src + 1) * QUADRIC_STRIDE);
}

/**
 * The quadric's value at `(x, y, z)`: the weighted sum of squared distances
 * to the planes that built it.
 *
 * Clamped at zero — the true value cannot be negative, and rounding on a
 * near-planar neighbourhood produces small negatives that would otherwise
 * sort ahead of every legitimate collapse.
 */
export function quadricEval(q: Float64Array, i: number, x: number, y: number, z: number): number {
	const o = i * QUADRIC_STRIDE;
	const v =
		q[o + A2] * x * x +
		2 * q[o + AB] * x * y +
		2 * q[o + AC] * x * z +
		2 * q[o + AD] * x +
		q[o + B2] * y * y +
		2 * q[o + BC] * y * z +
		2 * q[o + BD] * y +
		q[o + C2] * z * z +
		2 * q[o + CD] * z +
		q[o + D2];
	return v > 0 ? v : 0;
}

/**
 * The point minimising the quadric, or null when the system is singular.
 *
 * Solves `A v = -b` for the 3×3 upper block `A` and the linear part `b`.
 * Singular means the neighbourhood does not pin the point down in every
 * direction — a flat region, or a straight edge — and the caller should fall
 * back to one of the endpoints or the midpoint rather than invent a position.
 */
export function quadricMinimum(
	q: Float64Array,
	i: number,
	out: Float64Array | number[],
	epsilon = 1e-15,
): boolean {
	const o = i * QUADRIC_STRIDE;
	const a11 = q[o + A2];
	const a12 = q[o + AB];
	const a13 = q[o + AC];
	const a22 = q[o + B2];
	const a23 = q[o + BC];
	const a33 = q[o + C2];

	// Cofactors of the symmetric 3×3 block.
	const c11 = a22 * a33 - a23 * a23;
	const c12 = a13 * a23 - a12 * a33;
	const c13 = a12 * a23 - a13 * a22;
	const det = a11 * c11 + a12 * c12 + a13 * c13;

	// Scale-aware singularity test: comparing the determinant against an
	// absolute epsilon would call every small-but-well-conditioned patch
	// singular, and every huge-but-degenerate one solvable.
	const scale = Math.max(Math.abs(a11), Math.abs(a22), Math.abs(a33), 1e-300);
	if (Math.abs(det) <= epsilon * scale * scale * scale) return false;

	const c22 = a11 * a33 - a13 * a13;
	const c23 = a12 * a13 - a11 * a23;
	const c33 = a11 * a22 - a12 * a12;

	const b1 = -q[o + AD];
	const b2 = -q[o + BD];
	const b3 = -q[o + CD];
	const inv = 1 / det;

	out[0] = (c11 * b1 + c12 * b2 + c13 * b3) * inv;
	out[1] = (c12 * b1 + c22 * b2 + c23 * b3) * inv;
	out[2] = (c13 * b1 + c23 * b2 + c33 * b3) * inv;
	return Number.isFinite(out[0]) && Number.isFinite(out[1]) && Number.isFinite(out[2]);
}
