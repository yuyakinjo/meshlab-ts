/**
 * The 3D convex hull, standing in for the part of Qhull that MeshLab's
 * `filter_qhull` uses.
 *
 * Quickhull: start from a tetrahedron known to be inside the hull, then
 * repeatedly take the point furthest outside some face, delete every face that
 * point can see, and cone the resulting horizon back to it. Each round removes
 * at least one point from consideration, so it terminates, and every face it
 * keeps has all remaining points behind it — which is the definition of the
 * hull.
 *
 * The whole file is exact-arithmetic-free on purpose. Degeneracy is handled by
 * a scale-relative epsilon rather than by predicates, which is enough for the
 * point clouds these filters see and is what keeps the code readable. A hull
 * of nearly-cospherical points may come back with a slightly different
 * triangulation of a coplanar patch than Qhull would give; the *set* of hull
 * vertices is the same either way, and that is what all four callers use.
 */

export interface HullFace {
	/** Indices into the point array, wound anticlockwise seen from outside. */
	readonly v: readonly [number, number, number];
	/** Outward unit normal. */
	readonly normal: readonly number[];
	/** Plane offset: a point `p` is outside when `dot(normal, p) > offset`. */
	readonly offset: number;
}

export interface Hull {
	readonly faces: readonly HullFace[];
	/** Every point index that ended up on the hull, ascending. */
	readonly vertices: readonly number[];
}

const dot = (a: readonly number[], b: readonly number[]): number =>
	a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: readonly number[], b: readonly number[]): number[] => [
	a[0] - b[0],
	a[1] - b[1],
	a[2] - b[2],
];
const cross = (a: readonly number[], b: readonly number[]): number[] => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];

/**
 * The convex hull of `points`, given as a flat xyz array.
 *
 * Returns null when the points are degenerate — fewer than four, or all on one
 * line or plane — because there is no closed hull to return in those cases and
 * the callers all need to say so rather than proceed with an empty one.
 */
export function convexHull(points: Float64Array, count: number): Hull | null {
	if (count < 4) return null;
	const at = (i: number): number[] => [points[3 * i], points[3 * i + 1], points[3 * i + 2]];

	// A scale-relative tolerance: absolute epsilons are meaningless when the
	// same algorithm has to work on a 1mm bracket and a 100m scan.
	let extent = 0;
	for (let axis = 0; axis < 3; axis++) {
		let lo = Number.POSITIVE_INFINITY;
		let hi = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < count; i++) {
			lo = Math.min(lo, points[3 * i + axis]);
			hi = Math.max(hi, points[3 * i + axis]);
		}
		extent = Math.max(extent, hi - lo);
	}
	if (extent === 0) return null;
	const eps = extent * 1e-10;

	const seed = initialTetrahedron(at, count, eps);
	if (seed === null) return null;

	// Faces are mutable while building; `dead` marks the ones a round removed.
	interface Work {
		v: [number, number, number];
		normal: number[];
		offset: number;
		outside: number[];
		dead: boolean;
	}

	const makeFace = (a: number, b: number, c: number, interior: readonly number[]): Work => {
		const pa = at(a);
		let n = cross(sub(at(b), pa), sub(at(c), pa));
		let len = Math.hypot(n[0], n[1], n[2]);
		if (len === 0) {
			n = [0, 0, 0];
			len = 1;
		}
		n = [n[0] / len, n[1] / len, n[2] / len];
		let off = dot(n, pa);
		// Point the normal away from the seed's interior, so "outside" is
		// unambiguous without tracking winding through the horizon rebuild.
		if (dot(n, interior) > off) {
			n = [-n[0], -n[1], -n[2]];
			off = -off;
			return { v: [a, c, b], normal: n, offset: off, outside: [], dead: false };
		}
		return { v: [a, b, c], normal: n, offset: off, outside: [], dead: false };
	};

	const interior = [0, 1, 2].map(
		(k) => (at(seed[0])[k] + at(seed[1])[k] + at(seed[2])[k] + at(seed[3])[k]) / 4,
	);
	let faces: Work[] = [
		makeFace(seed[0], seed[1], seed[2], interior),
		makeFace(seed[0], seed[1], seed[3], interior),
		makeFace(seed[0], seed[2], seed[3], interior),
		makeFace(seed[1], seed[2], seed[3], interior),
	];

	// Assign each point to a face it lies outside of. A point behind every
	// face is inside the hull and is never looked at again.
	const assigned = new Uint8Array(count);
	for (const s of seed) assigned[s] = 1;
	const assign = (candidates: readonly number[], group: readonly Work[]) => {
		for (const i of candidates) {
			if (assigned[i] === 1) continue;
			const p = at(i);
			for (const f of group) {
				if (dot(f.normal, p) - f.offset > eps) {
					f.outside.push(i);
					break;
				}
			}
		}
	};
	const all: number[] = [];
	for (let i = 0; i < count; i++) all.push(i);
	assign(all, faces);

	// The main loop. Bounded because each round permanently accepts one point.
	let guard = 0;
	const limit = 10 * count + 64;
	for (;;) {
		if (++guard > limit) break;
		const seedFace = faces.find((f) => !f.dead && f.outside.length > 0);
		if (seedFace === undefined) break;

		// The point furthest outside this face. Taking the furthest is what
		// keeps the hull from being rebuilt once per point in the worst case.
		let apex = seedFace.outside[0];
		let best = Number.NEGATIVE_INFINITY;
		for (const i of seedFace.outside) {
			const d = dot(seedFace.normal, at(i)) - seedFace.offset;
			if (d > best) {
				best = d;
				apex = i;
			}
		}
		const p = at(apex);

		// Everything the apex can see comes off; the boundary of what came off
		// is the horizon.
		const visible: Work[] = [];
		for (const f of faces) {
			if (!f.dead && dot(f.normal, p) - f.offset > eps) {
				f.dead = true;
				visible.push(f);
			}
		}
		if (visible.length === 0) {
			// Numerically the apex is not outside anything after all; drop it
			// rather than spin.
			seedFace.outside = seedFace.outside.filter((i) => i !== apex);
			assigned[apex] = 1;
			continue;
		}

		// An edge of the visible set that appears once is on the horizon; one
		// that appears twice is interior to it.
		const edgeCount = new Map<string, [number, number]>();
		for (const f of visible) {
			for (let k = 0; k < 3; k++) {
				const a = f.v[k];
				const b = f.v[(k + 1) % 3];
				const key = a < b ? `${a}_${b}` : `${b}_${a}`;
				if (edgeCount.has(key)) edgeCount.delete(key);
				else edgeCount.set(key, [a, b]);
			}
		}

		const orphans: number[] = [];
		for (const f of visible) orphans.push(...f.outside);
		const fresh: Work[] = [];
		for (const [a, b] of edgeCount.values()) fresh.push(makeFace(a, b, apex, interior));
		assigned[apex] = 1;
		faces = faces.filter((f) => !f.dead);
		faces.push(...fresh);
		assign(orphans, fresh);
	}

	const live = faces.filter((f) => !f.dead);
	if (live.length < 4) return null;
	const used = new Set<number>();
	for (const f of live) for (const v of f.v) used.add(v);
	return {
		faces: live.map((f) => ({
			v: f.v as [number, number, number],
			normal: f.normal,
			offset: f.offset,
		})),
		vertices: [...used].sort((a, b) => a - b),
	};
}

/** Four points that span three dimensions, or null if none do. */
function initialTetrahedron(
	at: (i: number) => number[],
	count: number,
	eps: number,
): [number, number, number, number] | null {
	// The two extremes along x are certainly on the hull and certainly distinct
	// unless everything is coplanar in x.
	let a = 0;
	let b = 0;
	for (let i = 1; i < count; i++) {
		if (at(i)[0] < at(a)[0]) a = i;
		if (at(i)[0] > at(b)[0]) b = i;
	}
	if (a === b) return null;
	const pa = at(a);
	const ab = sub(at(b), pa);

	let c = -1;
	let bestArea = eps;
	for (let i = 0; i < count; i++) {
		if (i === a || i === b) continue;
		const n = cross(ab, sub(at(i), pa));
		const area = Math.hypot(n[0], n[1], n[2]);
		if (area > bestArea) {
			bestArea = area;
			c = i;
		}
	}
	if (c < 0) return null; // collinear

	const normal = cross(ab, sub(at(c), pa));
	let d = -1;
	let bestVolume = eps;
	for (let i = 0; i < count; i++) {
		if (i === a || i === b || i === c) continue;
		const volume = Math.abs(dot(normal, sub(at(i), pa)));
		if (volume > bestVolume) {
			bestVolume = volume;
			d = i;
		}
	}
	if (d < 0) return null; // coplanar
	return [a, b, c, d];
}

export const ConvexHull = { convexHull } as const;
