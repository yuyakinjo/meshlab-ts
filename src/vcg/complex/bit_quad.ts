/**
 * Bit-quads: polygonal meshes stored as triangles.
 *
 * VCGLib represents a quad-dominant mesh as an ordinary triangle mesh in which
 * the diagonals introduced by triangulation are tagged "faux". A quad is then
 * two triangles sharing a faux edge, and an n-gon is a fan sharing n-3 of
 * them. Nothing else about the mesh changes, which is why every algorithm in
 * this library keeps working on a quad mesh without knowing it is one.
 *
 * The invariant that makes the counting work is reciprocity: an edge is faux
 * from both sides or from neither, and a border edge is never faux. A mesh
 * that breaks it will still be counted, but the answers stop meaning anything
 * — hence {@link hasConsistentPerFaceFauxFlag}, which callers run first.
 */

import type { CMeshO } from "./cmesho.ts";
import { FaceFlag, fauxBit } from "./flags.ts";
import { faceDoubleArea, perFaceNormalized } from "./update/normal.ts";
import { faceFace } from "./update/topology.ts";

function fauxMask(m: CMeshO, f: number): number {
	return m.faceFlags[f] & FaceFlag.FAUX012;
}

function isFaux(m: CMeshO, f: number, e: number): boolean {
	return (m.faceFlags[f] & fauxBit(e)) !== 0;
}

/**
 * True when every face is a triangle or half of a quad.
 *
 * A face with two or three faux edges belongs to a larger polygon, so this is
 * the test that says "quads and triangles, nothing bigger".
 */
export function isBitTriQuadOnly(m: CMeshO): boolean {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const tmp = fauxMask(m, f);
		if (tmp !== 0 && tmp !== FaceFlag.FAUX0 && tmp !== FaceFlag.FAUX1 && tmp !== FaceFlag.FAUX2) {
			return false;
		}
	}
	return true;
}

/**
 * The number of quads: triangles with exactly one faux edge, halved.
 *
 * Assumes the faux bits are consistent, so each quad is counted from both of
 * its halves.
 */
export function countBitQuads(m: CMeshO): number {
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const tmp = fauxMask(m, f);
		if (tmp === FaceFlag.FAUX0 || tmp === FaceFlag.FAUX1 || tmp === FaceFlag.FAUX2) count++;
	}
	return Math.floor(count / 2);
}

/** The number of faces that are genuine triangles — no faux edge at all. */
export function countBitTris(m: CMeshO): number {
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (!m.isFaceD(f) && fauxMask(m, f) === 0) count++;
	}
	return count;
}

/**
 * Polygons of any size, triangles included.
 *
 * Each faux edge hides exactly one triangle boundary, so the count is faces
 * minus faux edges. Assumes there are no faux *vertices* — see
 * {@link countBitLargePolygons}, which corrects for them.
 */
export function countBitPolygons(m: CMeshO): number {
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) if (isFaux(m, f, e)) count++;
	}
	return m.fn - Math.floor(count / 2);
}

/**
 * Polygons, counted so that faux vertices are handled correctly.
 *
 * A vertex surrounded entirely by faux edges sits in a polygon's interior. It
 * was never a corner, so the "faces minus faux edges" identity over-subtracts
 * by one for each of them, and they are added back.
 *
 * The difference from {@link countBitPolygons} is exactly the number of such
 * vertices, which is why MeshLab reports both and treats a non-zero gap as a
 * sign that the quad structure is more complicated than it looks.
 */
export function countBitLargePolygons(m: CMeshO): number {
	// `touched` means "this vertex is on the boundary of a polygon". Start
	// with every vertex touched, clear the referenced ones, then re-touch the
	// endpoints of every real edge. What is left untouched and referenced is
	// interior to a polygon — an unreferenced vertex is inside nothing and
	// keeps its initial true.
	const touched = new Uint8Array(m.vertSize).fill(1);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) touched[m.fv(f, k)] = 0;
	}

	let fauxEdges = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (isFaux(m, f, e)) {
				fauxEdges++;
			} else {
				touched[m.fv(f, e)] = 1;
				touched[m.fv(f, (e + 1) % 3)] = 1;
			}
		}
	}

	let fauxVerts = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v) && touched[v] === 0) fauxVerts++;
	return m.fn - Math.floor(fauxEdges / 2) + fauxVerts;
}

/**
 * True when faux bits are reciprocated and no border edge is faux.
 *
 * Requires FF adjacency.
 */
export function hasConsistentPerFaceFauxFlag(m: CMeshO): boolean {
	if (m.ffFace === null) return false;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const mine = isFaux(m, f, e);
			if (m.isBorderFF(f, e)) {
				if (mine) return false;
				continue;
			}
			if (mine !== isFaux(m, m.ffp(f, e), m.ffi(f, e))) return false;
		}
	}
	return true;
}

/**
 * True when every FF link is well formed: a border points at itself, a
 * manifold edge points back, and a non-manifold edge closes into a ring.
 *
 * `vcg::tri::Clean::IsFFAdjacencyConsistent`.
 */
export function isFFAdjacencyConsistent(m: CMeshO): boolean {
	if (m.ffFace === null) return false;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!ffCorrectness(m, f, e)) return false;
		}
	}
	return true;
}

function ffCorrectness(m: CMeshO, f: number, e: number): boolean {
	const nf = m.ffp(f, e);
	const ne = m.ffi(f, e);
	if (nf < 0) return false;
	if (nf === f) return ne === e; // border: both indices must agree
	if (m.ffp(nf, ne) === f) return m.ffi(nf, ne) === e; // plain two-manifold

	// Non-manifold: the faces on this edge must form one closed ring, and no
	// step of it may look like a border or like a manifold pair — either would
	// mean the ring is really two separate structures wearing one edge.
	let curF = f;
	let curE = e;
	let steps = 0;
	do {
		const backF = m.ffp(curF, curE);
		const backE = m.ffi(curF, curE);
		if (backF === curF) return false;
		if (m.ffp(backF, backE) === curF && m.ffi(backF, backE) === curE) return false;
		curF = backF;
		curE = backE;
		if (++steps > 3 * m.faceSize + 3) return false;
	} while (curF !== f || curE !== e);
	return true;
}

/**
 * The faces of the polygon containing `f`, found by crossing only faux edges.
 *
 * A polygon in this representation is a connected group of triangles joined by
 * faux ("invisible") edges — a quad is two, a pentagon three, and so on.
 */
export function extractPolygon(m: CMeshO, f: number, out: number[]): void {
	out.length = 0;
	const stack = [f];
	const seen = new Set<number>([f]);
	while (stack.length > 0) {
		const cur = stack.pop() as number;
		out.push(cur);
		for (let e = 0; e < 3; e++) {
			if (!isFaux(m, cur, e)) continue;
			if (m.ffFace === null || m.isBorderFF(cur, e)) continue;
			const g = m.ffp(cur, e);
			if (m.isFaceD(g) || seen.has(g)) continue;
			seen.add(g);
			stack.push(g);
		}
	}
}

/**
 * Gives every triangle of a polygon the polygon's own normal.
 *
 * Area-weighted, so that a quad split into one large and one sliver triangle
 * is not dominated by the sliver's noisier normal. Without this, a quad mesh
 * shaded flat shows the diagonal of every quad, which is exactly the artefact
 * faux edges exist to hide.
 */
export function perBitPolygonFaceNormalized(m: CMeshO): void {
	perFaceNormalized(m);
	if (m.ffFace === null) faceFace(m);
	const done = new Uint8Array(m.faceSize);
	const group: number[] = [];
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || done[f] === 1) continue;
		extractPolygon(m, f, group);
		const n = [0, 0, 0];
		for (const g of group) {
			const area = faceDoubleArea(m, g);
			for (let k = 0; k < 3; k++) n[k] += m.faceNormal[3 * g + k] * area;
		}
		const len = Math.hypot(n[0], n[1], n[2]);
		for (const g of group) {
			done[g] = 1;
			if (len === 0) continue;
			for (let k = 0; k < 3; k++) m.faceNormal[3 * g + k] = n[k] / len;
		}
	}
}
