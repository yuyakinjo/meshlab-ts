/**
 * The two local operations that rewrite connectivity in place: collapsing an
 * edge to a point, and flipping the diagonal of the two triangles across one.
 *
 * Both are used by more than one algorithm — QEM decimation collapses,
 * isotropic remeshing does both — and both have a correctness condition that
 * is easy to get subtly wrong, so there is exactly one copy of each here.
 *
 * These work over an explicit vertex→faces incidence rather than FF adjacency,
 * because FF is invalidated by the first operation and rebuilding it per step
 * would dominate the cost. {@link buildVertexFaces} makes the table; the caller
 * keeps it up to date through the return values.
 */

import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";

/** For each vertex slot, the set of live faces using it. */
export function buildVertexFaces(m: CMeshO): Array<Set<number>> {
	const table: Array<Set<number>> = Array.from({ length: m.vertSize }, () => new Set<number>());
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) table[m.faceVert[3 * f + k]].add(f);
	}
	return table;
}

/** The live faces using both `u` and `v` — one for a border edge, two inside. */
export function sharedFaces(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	u: number,
	v: number,
): number[] {
	const out: number[] = [];
	for (const f of vertFaces[u]) if (!m.isFaceD(f) && vertFaces[v].has(f)) out.push(f);
	return out;
}

/** 2·area / longest-edge² for a triangle given as nine coordinates. */
export function triQuality(
	ax: number,
	ay: number,
	az: number,
	bx: number,
	by: number,
	bz: number,
	cx: number,
	cy: number,
	cz: number,
): number {
	const ux = bx - ax;
	const uy = by - ay;
	const uz = bz - az;
	const vx = cx - ax;
	const vy = cy - ay;
	const vz = cz - az;
	const nx = uy * vz - uz * vy;
	const ny = uz * vx - ux * vz;
	const nz = ux * vy - uy * vx;
	const doubleArea = Math.hypot(nx, ny, nz);
	if (doubleArea === 0) return 0;
	const e0 = ux * ux + uy * uy + uz * uz;
	const e1 = vx * vx + vy * vy + vz * vz;
	const dx = bx - cx;
	const dy = by - cy;
	const dz = bz - cz;
	const e2 = dx * dx + dy * dy + dz * dz;
	return doubleArea / Math.max(e0, e1, e2);
}

/**
 * The link condition: collapsing an edge is safe exactly when the only
 * vertices adjacent to both endpoints are the ones opposite the edge.
 *
 * A common neighbour that is *not* part of a shared face means the collapse
 * would fuse two pieces of surface that only met at that vertex — closing a
 * hole or pinching a handle, either of which changes the genus.
 */
export function linkCondition(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	u: number,
	v: number,
	shared: readonly number[],
): boolean {
	const neighboursOf = (a: number): Set<number> => {
		const out = new Set<number>();
		for (const f of vertFaces[a]) {
			if (m.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const w = m.faceVert[3 * f + k];
				if (w !== a) out.add(w);
			}
		}
		return out;
	};

	const allowed = new Set<number>();
	for (const f of shared) {
		for (let k = 0; k < 3; k++) {
			const w = m.faceVert[3 * f + k];
			if (w !== u && w !== v) allowed.add(w);
		}
	}

	const nu = neighboursOf(u);
	const nv = neighboursOf(v);
	for (const w of nu) {
		if (w === v) continue;
		if (nv.has(w) && !allowed.has(w)) return false;
	}
	return true;
}

/**
 * Collapses the edge `(u, v)` onto `(x, y, z)`, keeping `u` and deleting `v`.
 *
 * Assumes the caller has already checked that this is allowed. Returns the
 * vertices whose neighbourhood changed, which is what a priority-queue caller
 * needs in order to know whose costs went stale.
 */
export function collapseEdge(
	m: CMeshO,
	vertFaces: Array<Set<number>>,
	u: number,
	v: number,
	x: number,
	y: number,
	z: number,
): number[] {
	const shared = sharedFaces(m, vertFaces, u, v);
	const touched = new Set<number>();

	// The faces on the collapsed edge become degenerate, so they go.
	for (const f of shared) {
		for (let k = 0; k < 3; k++) {
			const w = m.faceVert[3 * f + k];
			touched.add(w);
			vertFaces[w].delete(f);
		}
		if (!m.isFaceD(f)) Allocator.deleteFace(m, f);
	}

	// Everything else that referenced v now references u.
	for (const f of [...vertFaces[v]]) {
		if (m.isFaceD(f)) {
			vertFaces[v].delete(f);
			continue;
		}
		for (let k = 0; k < 3; k++) {
			if (m.faceVert[3 * f + k] === v) m.faceVert[3 * f + k] = u;
			touched.add(m.faceVert[3 * f + k]);
		}
		vertFaces[u].add(f);
	}
	vertFaces[v].clear();

	if (!m.isVertD(v)) Allocator.deleteVertex(m, v);
	m.setVert(u, x, y, z);

	touched.delete(u);
	touched.delete(v);
	return [...touched];
}

/** The two faces across an interior edge, and the corners opposite it. */
export interface EdgePair {
	/** The face carrying the edge, and the one on the other side. */
	readonly f0: number;
	readonly f1: number;
	/** The edge's endpoints. */
	readonly a: number;
	readonly b: number;
	/** The vertex of each face that is not an endpoint. */
	readonly o0: number;
	readonly o1: number;
}

/** Resolves the edge `(u, v)` into the pair of faces across it, or null. */
export function edgePairOf(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	u: number,
	v: number,
): EdgePair | null {
	const shared = sharedFaces(m, vertFaces, u, v);
	if (shared.length !== 2) return null;
	const opposite = (f: number): number => {
		for (let k = 0; k < 3; k++) {
			const w = m.faceVert[3 * f + k];
			if (w !== u && w !== v) return w;
		}
		return -1;
	};
	const o0 = opposite(shared[0]);
	const o1 = opposite(shared[1]);
	if (o0 < 0 || o1 < 0) return null;
	return { f0: shared[0], f1: shared[1], a: u, b: v, o0, o1 };
}

/**
 * Replaces the edge `(a, b)` with `(o0, o1)`, the other diagonal of the
 * quadrilateral the two faces form.
 *
 * Refuses when the flip would be illegal rather than merely ugly: a boundary
 * edge has no second face to work with, and if `o0` and `o1` are already
 * joined the flip would produce a second edge between them, which no
 * manifold mesh may have. Returns whether it happened.
 */
export function flipEdge(m: CMeshO, vertFaces: Array<Set<number>>, pair: EdgePair): boolean {
	const { f0, f1, a, b, o0, o1 } = pair;
	if (o0 === o1) return false;
	// A duplicate edge would be created, which is exactly the case that turns a
	// valid mesh non-manifold.
	if (sharedFaces(m, vertFaces, o0, o1).length > 0) return false;

	// Substituting a single vertex leaves each face's cyclic order — and so its
	// orientation — untouched: f0 gives up b for o1, f1 gives up a for o0. The
	// two then traverse the new edge in opposite directions, whichever way round
	// they held the old one.
	substitute(m, f0, b, o1);
	substitute(m, f1, a, o0);

	vertFaces[a].delete(f1);
	vertFaces[b].delete(f0);
	vertFaces[o0].add(f1);
	vertFaces[o1].add(f0);
	return true;
}

/** Replaces one vertex of a face in place, leaving the other two where they are. */
function substitute(m: CMeshO, f: number, gone: number, arrived: number): void {
	for (let k = 0; k < 3; k++) {
		if (m.faceVert[3 * f + k] === gone) m.faceVert[3 * f + k] = arrived;
	}
}

export const EdgeOps = {
	buildVertexFaces,
	sharedFaces,
	triQuality,
	linkCondition,
	collapseEdge,
	edgePairOf,
	flipEdge,
} as const;
