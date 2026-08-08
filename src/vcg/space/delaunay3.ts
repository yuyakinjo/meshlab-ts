/**
 * 3D Delaunay tetrahedralization, standing in for the `qhull d` call MeshLab's
 * alpha-shape and Voronoi-filtering filters make.
 *
 * Bowyer–Watson: hold a tetrahedralization of everything inserted so far, and
 * to add a point, delete every tetrahedron whose circumsphere contains it and
 * re-fill the resulting cavity by coning its boundary to the new point. The
 * invariant that makes this Delaunay is exactly the one being restored —
 * no tetrahedron's circumsphere contains any point.
 *
 * The two things worth knowing before reading:
 *
 * - **The circumcentre is the Voronoi vertex.** The Delaunay tetrahedralization
 *   and the Voronoi diagram are duals, so a tetrahedron's circumcentre is a
 *   vertex of the Voronoi cell of each of its four corners. That identity is
 *   the whole basis of the Amenta–Bern pole selection, and it is why one
 *   structure serves both filters.
 * - **The bounding simplex is not deleted, it is filtered.** Insertion starts
 *   from a huge tetrahedron containing every point; any output tetrahedron
 *   still touching one of its corners is outside the point set's own hull and
 *   is dropped at the end.
 */

export interface Tetrahedron {
	/** Indices into the caller's point array. */
	readonly v: readonly [number, number, number, number];
	/** Circumcentre — a Voronoi vertex of all four corners. */
	readonly centre: readonly number[];
	/** Circumradius. For alpha shapes this is the simplex's alpha value. */
	readonly radius: number;
}

/** The four triangular faces of a tetrahedron, as sorted index triples. */
export function tetraFaces(t: Tetrahedron): Array<[number, number, number]> {
	const [a, b, c, d] = t.v;
	return [sorted3(a, b, c), sorted3(a, b, d), sorted3(a, c, d), sorted3(b, c, d)];
}

const sorted3 = (a: number, b: number, c: number): [number, number, number] => {
	const s = [a, b, c].sort((x, y) => x - y);
	return [s[0], s[1], s[2]];
};

/**
 * The Delaunay tetrahedralization of `points`, as a flat xyz array.
 *
 * Returns an empty list when the points are degenerate — fewer than four, or
 * all coplanar — since there is no tetrahedron to build in either case.
 */
export function delaunay3(points: Float64Array, count: number): Tetrahedron[] {
	if (count < 4) return [];

	// Work on a copy extended with the four corners of a containing simplex,
	// so the point indices the caller passed in keep their meaning.
	const lo = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
	const hi = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
	for (let i = 0; i < count; i++) {
		for (let a = 0; a < 3; a++) {
			lo[a] = Math.min(lo[a], points[3 * i + a]);
			hi[a] = Math.max(hi[a], points[3 * i + a]);
		}
	}
	const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
	if (!(span > 0)) return [];
	const centre = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2);
	// Large enough that the bounding simplex's circumspheres cannot cut into
	// the real point set, which would leave spurious tetrahedra behind.
	const big = span * 1000;

	const work = new Float64Array(3 * (count + 4));
	work.set(points.subarray(0, 3 * count));
	const superCorners = [
		[centre[0] - big, centre[1] - big, centre[2] - big],
		[centre[0] + big * 3, centre[1] - big, centre[2] - big],
		[centre[0] - big, centre[1] + big * 3, centre[2] - big],
		[centre[0] - big, centre[1] - big, centre[2] + big * 3],
	];
	superCorners.forEach((p, i) => {
		work[3 * (count + i)] = p[0];
		work[3 * (count + i) + 1] = p[1];
		work[3 * (count + i) + 2] = p[2];
	});
	const at = (i: number): number[] => [work[3 * i], work[3 * i + 1], work[3 * i + 2]];

	let tetra: Tetrahedron[] = [];
	const seed = makeTetra(at, count, count + 1, count + 2, count + 3);
	if (seed === null) return [];
	tetra.push(seed);

	const eps = span * 1e-12;
	for (let i = 0; i < count; i++) {
		const p = at(i);
		// Every tetrahedron whose circumsphere swallows the new point loses its
		// claim to being Delaunay.
		const bad: Tetrahedron[] = [];
		const keep: Tetrahedron[] = [];
		for (const t of tetra) {
			const d = Math.hypot(p[0] - t.centre[0], p[1] - t.centre[1], p[2] - t.centre[2]);
			if (d < t.radius - eps) bad.push(t);
			else keep.push(t);
		}
		if (bad.length === 0) continue;

		// The cavity's boundary: a face shared by two doomed tetrahedra is
		// interior to the hole, one that appears once is on its wall.
		const wall = new Map<string, [number, number, number]>();
		for (const t of bad) {
			for (const f of tetraFaces(t)) {
				const key = `${f[0]}_${f[1]}_${f[2]}`;
				if (wall.has(key)) wall.delete(key);
				else wall.set(key, f);
			}
		}

		tetra = keep;
		for (const f of wall.values()) {
			const made = makeTetra(at, f[0], f[1], f[2], i);
			// A degenerate cone — the new point coplanar with a wall face —
			// contributes nothing and must not be kept with a NaN circumsphere.
			if (made !== null) tetra.push(made);
		}
	}

	// Anything still touching the bounding simplex lies outside the real hull.
	return tetra.filter((t) => t.v.every((v) => v < count));
}

/** A tetrahedron with its circumsphere, or null when the four are coplanar. */
function makeTetra(
	at: (i: number) => number[],
	a: number,
	b: number,
	c: number,
	d: number,
): Tetrahedron | null {
	const pa = at(a);
	const pb = at(b);
	const pc = at(c);
	const pd = at(d);
	const centre = circumcentre(pa, pb, pc, pd);
	if (centre === null) return null;
	return {
		v: [a, b, c, d],
		centre,
		radius: Math.hypot(centre[0] - pa[0], centre[1] - pa[1], centre[2] - pa[2]),
	};
}

/**
 * The centre of the sphere through four points, or null when they are coplanar.
 *
 * Solving the three planes that bisect `pa`–`pb`, `pa`–`pc` and `pa`–`pd`.
 * Coplanar points give a singular system, which is exactly the case with no
 * finite circumsphere.
 */
export function circumcentre(
	pa: readonly number[],
	pb: readonly number[],
	pc: readonly number[],
	pd: readonly number[],
): number[] | null {
	const row = (p: readonly number[]): number[] => [
		2 * (p[0] - pa[0]),
		2 * (p[1] - pa[1]),
		2 * (p[2] - pa[2]),
		sq(p) - sq(pa),
	];
	const m = [row(pb), row(pc), row(pd)];

	// Gaussian elimination with partial pivoting on a 3x4 system.
	const scale = Math.max(1e-300, ...m.flatMap((r) => r.slice(0, 3).map(Math.abs)));
	for (let col = 0; col < 3; col++) {
		let pivot = col;
		for (let r = col + 1; r < 3; r++) {
			if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
		}
		if (Math.abs(m[pivot][col]) < scale * 1e-14) return null;
		[m[col], m[pivot]] = [m[pivot], m[col]];
		for (let r = 0; r < 3; r++) {
			if (r === col) continue;
			const factor = m[r][col] / m[col][col];
			for (let k = col; k < 4; k++) m[r][k] -= factor * m[col][k];
		}
	}
	return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

const sq = (p: readonly number[]): number => p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

/**
 * The circumradius of a triangle — its own alpha value.
 *
 * A face of the Delaunay tetrahedralization joins the alpha complex when its
 * smallest empty circumsphere is no larger than alpha, and for a triangle on
 * the boundary that sphere is the one through its three corners.
 */
export function triangleCircumradius(
	pa: readonly number[],
	pb: readonly number[],
	pc: readonly number[],
): number {
	const a = Math.hypot(pb[0] - pc[0], pb[1] - pc[1], pb[2] - pc[2]);
	const b = Math.hypot(pa[0] - pc[0], pa[1] - pc[1], pa[2] - pc[2]);
	const c = Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]);
	const u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
	const v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
	const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
	const area = Math.hypot(n[0], n[1], n[2]) / 2;
	// A degenerate triangle has no finite circumcircle; reporting infinity
	// keeps it out of every alpha complex rather than into all of them.
	if (area <= 0) return Number.POSITIVE_INFINITY;
	return (a * b * c) / (4 * area);
}

export const Delaunay3 = { delaunay3, circumcentre, triangleCircumradius, tetraFaces } as const;
