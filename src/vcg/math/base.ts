/**
 * Scalar type of the mesh kernel.
 *
 * MeshLab's `Scalarm` is a compile-time alias for either `float` or `double`.
 * We are always the double build.
 */
export type Scalarm = number;

/** Comparison epsilon for coordinates that ought to be identical. */
export const EPSILON = 1e-12;

/** Relative epsilon used when comparing lengths and areas. */
export const REL_EPSILON = 1e-9;

export function clamp(x: number, lo: number, hi: number): number {
	return x < lo ? lo : x > hi ? hi : x;
}

export function toRad(deg: number): number {
	return (deg * Math.PI) / 180;
}

export function toDeg(rad: number): number {
	return (rad * 180) / Math.PI;
}

/**
 * `Math.acos` that tolerates arguments a hair outside [-1, 1], which dot
 * products of normalised vectors routinely produce.
 */
export function safeAcos(x: number): number {
	return Math.acos(clamp(x, -1, 1));
}

/** True when `a` and `b` agree to within an absolute or relative tolerance. */
export function nearlyEqual(a: number, b: number, tol = REL_EPSILON): boolean {
	const diff = Math.abs(a - b);
	if (diff <= tol) return true;
	return diff <= tol * Math.max(Math.abs(a), Math.abs(b));
}
