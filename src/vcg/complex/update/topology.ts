/**
 * `UpdateTopology` — building the adjacency relations.
 *
 * Mirrors `vcg::tri::UpdateTopology`, including the two encodings that most
 * surprise newcomers to VCGLib:
 *
 * - **FF is a ring, not a pair.** All faces sharing an edge are linked into a
 *   cycle. A manifold edge gives a 2-cycle, a border edge a self-loop, and a
 *   non-manifold edge shared by k faces a k-cycle. Code that assumes
 *   `ffp(ffp(f,e), ffi(f,e)) === f` is only correct on a manifold mesh; use
 *   {@link isManifoldEdge} to find out. The self-loop that marks a border
 *   requires *both* `ffp(f,e) === f` and `ffi(f,e) === e` — see
 *   `CMeshO.isBorderFF` for why the face alone is not enough.
 * - **VF is an intrusive linked list.** Each vertex holds the head of a chain
 *   threading every face corner that touches it, with the links stored on the
 *   corners themselves.
 */
import { MeshElement } from "../../../common/ml_document/mesh_element.ts";
import type { CMeshO } from "../cmesho.ts";
import { enableChannels } from "../components.ts";

/** One directed use of an undirected edge by a face corner. */
export interface EdgeUse {
	/** Lower vertex index of the edge. */
	v0: number;
	/** Higher vertex index of the edge. */
	v1: number;
	/** Face that uses it. */
	f: number;
	/** Which of that face's three edges (0..2) this is. */
	e: number;
}

/**
 * Every (face, edge) pair in the mesh, sorted by the edge's vertex pair.
 *
 * This is `UpdateTopology::FillEdgeVector` followed by its sort — the two are
 * never used apart, so they are one function here. Equal edges end up adjacent,
 * which is what makes the grouping loops below linear.
 */
export function fillSortedEdgeVector(m: CMeshO): EdgeUse[] {
	const uses: EdgeUse[] = [];
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = m.faceVert[3 * f + e];
			const b = m.faceVert[3 * f + ((e + 1) % 3)];
			uses.push(a < b ? { v0: a, v1: b, f, e } : { v0: b, v1: a, f, e });
		}
	}
	uses.sort((p, q) => (p.v0 !== q.v0 ? p.v0 - q.v0 : p.v1 !== q.v1 ? p.v1 - q.v1 : p.f - q.f));
	return uses;
}

/**
 * The distinct undirected edges, each with the faces that use it.
 *
 * `UpdateTopology::FillUniqueEdgeVector` in VCGLib.
 */
export function fillUniqueEdgeVector(
	m: CMeshO,
): Array<{ v0: number; v1: number; uses: EdgeUse[] }> {
	const sorted = fillSortedEdgeVector(m);
	const out: Array<{ v0: number; v1: number; uses: EdgeUse[] }> = [];
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		out.push({ v0: sorted[i].v0, v1: sorted[i].v1, uses: sorted.slice(i, j) });
		i = j;
	}
	return out;
}

/** Resets FF so that every edge is a border (a self-loop). */
export function clearFaceFace(m: CMeshO): void {
	enableChannels(m, MeshElement.MM_FACEFACETOPO);
	const ffFace = m.ffFace as Int32Array;
	const ffEdge = m.ffEdge as Uint8Array;
	for (let f = 0; f < m.faceSize; f++) {
		for (let e = 0; e < 3; e++) {
			ffFace[3 * f + e] = f;
			ffEdge[3 * f + e] = e;
		}
	}
}

/**
 * Builds face-face adjacency.
 *
 * Allocates the storage first if it is missing. VCGLib asserts instead; being
 * forgiving here costs nothing and keeps every caller from having to remember
 * to enable the channel.
 */
export function faceFace(m: CMeshO): void {
	clearFaceFace(m);
	if (m.fn === 0) return;
	const ffFace = m.ffFace as Int32Array;
	const ffEdge = m.ffEdge as Uint8Array;

	const sorted = fillSortedEdgeVector(m);
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		// Link the group [i, j) into a cycle. A single-element group links to
		// itself, which is exactly the border encoding clearFaceFace left.
		for (let k = i; k < j; k++) {
			const cur = sorted[k];
			const next = sorted[k + 1 < j ? k + 1 : i];
			ffFace[3 * cur.f + cur.e] = next.f;
			ffEdge[3 * cur.f + cur.e] = next.e;
		}
		i = j;
	}
	m.currentDataMask |= MeshElement.MM_FACEFACETOPO;
}

/** Builds vertex-face adjacency: one chain per vertex over its face corners. */
export function vertexFace(m: CMeshO): void {
	enableChannels(m, MeshElement.MM_VERTFACETOPO);
	const head = m.vfHeadFace as Int32Array;
	const headIdx = m.vfHeadIndex as Uint8Array;
	const next = m.vfNextFace as Int32Array;
	const nextIdx = m.vfNextIndex as Uint8Array;

	head.fill(-1);
	headIdx.fill(0);
	next.fill(-1);
	nextIdx.fill(0);

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = m.faceVert[3 * f + k];
			// Prepend, as VCGLib does: the chain ends up in reverse face order.
			next[3 * f + k] = head[v];
			nextIdx[3 * f + k] = headIdx[v];
			head[v] = f;
			headIdx[v] = k;
		}
	}
	m.currentDataMask |= MeshElement.MM_VERTFACETOPO;
}

/**
 * Calls `fn(face, cornerIndex)` for every face corner touching vertex `v`.
 *
 * Requires VF adjacency.
 */
export function forEachVFCorner(m: CMeshO, v: number, fn: (f: number, k: number) => void): void {
	let f = (m.vfHeadFace as Int32Array)[v];
	let k = (m.vfHeadIndex as Uint8Array)[v];
	while (f !== -1) {
		const nf = (m.vfNextFace as Int32Array)[3 * f + k];
		const nk = (m.vfNextIndex as Uint8Array)[3 * f + k];
		fn(f, k);
		f = nf;
		k = nk;
	}
}

/** How many faces share edge `e` of face `f`, counted by walking the FF ring. */
export function faceRingSize(m: CMeshO, f: number, e: number): number {
	let n = 1;
	let cf = m.ffp(f, e);
	let ce = m.ffi(f, e);
	while (cf !== f || ce !== e) {
		n++;
		const nf = m.ffp(cf, ce);
		const ne = m.ffi(cf, ce);
		cf = nf;
		ce = ne;
	}
	return n;
}

/** True when edge `e` of face `f` is shared by at most two faces. */
export function isManifoldEdge(m: CMeshO, f: number, e: number): boolean {
	// The common cases without walking: a border is a 1-ring, and a manifold
	// edge is a 2-ring that comes straight back.
	if (m.isBorderFF(f, e)) return true;
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	return m.ffp(g, ge) === f && m.ffi(g, ge) === e;
}

export const UpdateTopology = {
	fillSortedEdgeVector,
	fillUniqueEdgeVector,
	clearFaceFace,
	faceFace,
	vertexFace,
	forEachVFCorner,
	faceRingSize,
	isManifoldEdge,
	faceFaceFromTexCoord,
	detachFF,
} as const;

/**
 * FF adjacency that treats a texture seam as a boundary.
 *
 * Builds the ordinary adjacency and then detaches every edge whose two faces
 * disagree about the UVs along it. The result answers "what are the charts"
 * with the same border machinery that answers "what are the holes" — which is
 * why the seam selection filter is three lines rather than its own traversal.
 */
export function faceFaceFromTexCoord(m: CMeshO): void {
	faceFace(m);
	const wt = m.wedgeTexCoord;
	if (wt === null) return;
	const uv = (f: number, k: number): [number, number] => [wt[6 * f + 2 * k], wt[6 * f + 2 * k + 1]];
	const same = (a: [number, number], b: [number, number]) => a[0] === b[0] && a[1] === b[1];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(f, e)) continue;
			const g = m.ffp(f, e);
			const ge = m.ffi(f, e);
			// The two faces may traverse the shared edge in either direction, so
			// which of the neighbour's corners to compare against depends on
			// whether the first endpoints already agree.
			const aligned = m.fv(f, e) === m.fv(g, ge);
			const seam = aligned
				? !same(uv(f, e), uv(g, ge)) || !same(uv(f, (e + 1) % 3), uv(g, (ge + 1) % 3))
				: !same(uv(f, e), uv(g, (ge + 1) % 3)) || !same(uv(f, (e + 1) % 3), uv(g, ge));
			if (seam) detachFF(m, f, e);
		}
	}
}

/**
 * Removes face `f`'s edge `e` from its adjacency ring, leaving it a border.
 *
 * VCG's `face::FFDetach`. The ring is walked to find the face pointing back at
 * this one so its link can be redirected, which is what keeps a non-manifold
 * ring consistent rather than merely dropping one of its members.
 */
export function detachFF(m: CMeshO, f: number, e: number): void {
	if (m.isBorderFF(f, e)) return;
	let curF = m.ffp(f, e);
	let curE = m.ffi(f, e);
	// Walk to the ring member immediately before `f`.
	while (m.ffp(curF, curE) !== f || m.ffi(curF, curE) !== e) {
		const nf = m.ffp(curF, curE);
		const ne = m.ffi(curF, curE);
		curF = nf;
		curE = ne;
	}
	const ff = m.ffFace as Int32Array;
	const fe = m.ffEdge as Uint8Array;
	if (curF === f && curE === e) {
		// A two-face ring: the neighbour becomes a border too.
		const g = ff[3 * f + e];
		const ge = fe[3 * f + e];
		ff[3 * g + ge] = g;
		fe[3 * g + ge] = ge;
	} else {
		ff[3 * curF + curE] = ff[3 * f + e];
		fe[3 * curF + curE] = fe[3 * f + e];
	}
	ff[3 * f + e] = f;
	fe[3 * f + e] = e;
}
