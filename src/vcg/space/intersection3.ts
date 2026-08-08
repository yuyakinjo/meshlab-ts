/**
 * `vcg/space/intersection3.h` — the predicates that answer "do these two
 * pieces of surface pass through each other".
 *
 * Used by self-intersection detection, where the question is asked once per
 * pair of nearby faces and so has to be both cheap and exact enough not to
 * report a face as crossing its own neighbour.
 */

type Vec = readonly number[];

const sub = (a: Vec, b: Vec): number[] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec, b: Vec): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec, b: Vec): number[] => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];

/**
 * Möller–Trumbore: where a segment meets a triangle.
 *
 * Returns the two barycentric parameters of the hit, or null. The segment is
 * treated as bounded, so a plane crossing beyond either endpoint is not a hit.
 */
export function intersectionSegmentTriangle(
	from: Vec,
	to: Vec,
	p0: Vec,
	p1: Vec,
	p2: Vec,
): { a: number; b: number; t: number } | null {
	const dir = sub(to, from);
	const e1 = sub(p1, p0);
	const e2 = sub(p2, p0);
	const pvec = cross(dir, e2);
	const det = dot(e1, pvec);
	// A zero determinant means the segment lies in the triangle's plane. VCG
	// reports no hit there, and so does this: a coplanar overlap is a distinct
	// question that the caller's shared-vertex handling already covers.
	if (Math.abs(det) < 1e-20) return null;
	const inv = 1 / det;
	const tvec = sub(from, p0);
	const a = dot(tvec, pvec) * inv;
	if (a < 0 || a > 1) return null;
	const qvec = cross(tvec, e1);
	const b = dot(dir, qvec) * inv;
	if (b < 0 || a + b > 1) return null;
	const t = dot(e2, qvec) * inv;
	if (t < 0 || t > 1) return null;
	return { a, b, t };
}

/**
 * Do two triangles cross?
 *
 * Each of the six edges is tested against the other triangle. That misses only
 * the coplanar case, where two triangles overlap without any edge piercing the
 * other's interior — which is exactly the case VCG also declines to report,
 * because two coplanar faces of a real mesh are far more often a legitimate
 * shared edge than a defect.
 */
export function intersectionTriangleTriangle(a: readonly Vec[], b: readonly Vec[]): boolean {
	for (let k = 0; k < 3; k++) {
		if (intersectionSegmentTriangle(a[k], a[(k + 1) % 3], b[0], b[1], b[2]) !== null) return true;
		if (intersectionSegmentTriangle(b[k], b[(k + 1) % 3], a[0], a[1], a[2]) !== null) return true;
	}
	return false;
}
