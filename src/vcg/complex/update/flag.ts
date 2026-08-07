/**
 * `UpdateFlags` — clearing and recomputing the per-simplex bits.
 *
 * Mirrors `vcg::tri::UpdateFlags`. The DELETED bit is deliberately never
 * touched by the bulk clears: it is the allocator's business, and wiping it
 * would resurrect deleted elements while leaving `vn`/`fn` stale.
 */
import type { CMeshO } from "../cmesho.ts";
import { borderBit, FaceFlag, VertexFlag } from "../flags.ts";
import { fillSortedEdgeVector } from "./topology.ts";

const KEEP_VERT = VertexFlag.DELETED;
const KEEP_FACE = FaceFlag.DELETED;

/** Clears the given vertex bits (never DELETED). Defaults to all of them. */
export function vertexClear(m: CMeshO, flagMask = 0xffffffff): void {
	const mask = ~(flagMask & ~KEEP_VERT);
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= mask;
}

/** Clears the given face bits (never DELETED). Defaults to all of them. */
export function faceClear(m: CMeshO, flagMask = 0xffffffff): void {
	const mask = ~(flagMask & ~KEEP_FACE);
	for (let f = 0; f < m.faceSize; f++) m.faceFlags[f] &= mask;
}

export function clear(m: CMeshO): void {
	vertexClear(m);
	faceClear(m);
}

export function vertexClearV(m: CMeshO): void {
	vertexClear(m, VertexFlag.VISITED);
}
export function vertexClearS(m: CMeshO): void {
	vertexClear(m, VertexFlag.SELECTED);
}
export function vertexClearB(m: CMeshO): void {
	vertexClear(m, VertexFlag.BORDER);
}
export function faceClearV(m: CMeshO): void {
	faceClear(m, FaceFlag.VISITED);
}
export function faceClearS(m: CMeshO): void {
	faceClear(m, FaceFlag.SELECTED);
}
export function faceClearB(m: CMeshO): void {
	faceClear(m, FaceFlag.BORDER012);
}

/** Sets each face's border bits from FF adjacency. Requires FF. */
export function faceBorderFromFF(m: CMeshO): void {
	faceClearB(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(f, e)) m.faceFlags[f] |= borderBit(e);
		}
	}
}

/**
 * Sets each face's border bits without any adjacency, by sorting the edges.
 *
 * An edge used by exactly one face is a border; note that this treats a
 * non-manifold edge (three or more uses) as *not* a border, matching VCGLib.
 */
export function faceBorderFromNone(m: CMeshO): void {
	faceClearB(m);
	const sorted = fillSortedEdgeVector(m);
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		if (j - i === 1) {
			const u = sorted[i];
			m.faceFlags[u.f] |= borderBit(u.e);
		}
		i = j;
	}
}

/** Marks a vertex as border when any incident edge is a border edge. */
export function vertexBorderFromNone(m: CMeshO): void {
	vertexClearB(m);
	const sorted = fillSortedEdgeVector(m);
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		if (j - i === 1) {
			m.vertFlags[sorted[i].v0] |= VertexFlag.BORDER;
			m.vertFlags[sorted[i].v1] |= VertexFlag.BORDER;
		}
		i = j;
	}
}

/** Propagates the faces' border bits onto their vertices. */
export function vertexBorderFromFace(m: CMeshO): void {
	vertexClearB(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if ((m.faceFlags[f] & borderBit(e)) === 0) continue;
			m.vertFlags[m.fv(f, e)] |= VertexFlag.BORDER;
			m.vertFlags[m.fv(f, (e + 1) % 3)] |= VertexFlag.BORDER;
		}
	}
}

export const UpdateFlags = {
	clear,
	vertexClear,
	faceClear,
	vertexClearV,
	vertexClearS,
	vertexClearB,
	faceClearV,
	faceClearS,
	faceClearB,
	faceBorderFromFF,
	faceBorderFromNone,
	vertexBorderFromNone,
	vertexBorderFromFace,
} as const;
