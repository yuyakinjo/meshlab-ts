/**
 * `vcg/complex/algorithms/crease_cut.h` — opening a mesh along its sharp
 * edges.
 *
 * The mesh keeps exactly the same shape; what changes is that a vertex on a
 * crease stops being one vertex shared by both sides and becomes one per
 * smooth patch around it. That is what a renderer needs to shade a cube as a
 * cube — a single shared normal at a corner averages three perpendicular faces
 * and gives the rounded look every exporter is trying to avoid.
 */
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag, VertexFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { faceNormalOf } from "./update/normal.ts";
import { faceFace } from "./update/topology.ts";

const faceEdgeSelBit = (e: number): number => FaceFlag.FACEEDGESEL0 << e;

const isEdgeSelected = (m: CMeshO, f: number, e: number): boolean =>
	(m.faceFlags[f] & faceEdgeSelBit(e)) !== 0;

/**
 * The signed angle between the two faces sharing edge `e` of face `f`.
 *
 * Positive is convex, negative concave. The sign comes from which side of the
 * neighbour's plane the far vertex falls on, which is why this needs more than
 * the dot product of the two normals — that alone cannot tell a ridge from a
 * valley, and `FaceEdgeSelSignedCrease` selects on an asymmetric range.
 */
export function dihedralAngleRad(m: CMeshO, f: number, e: number): number {
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	const nf = [0, 0, 0];
	const ng = [0, 0, 0];
	faceNormalOf(m, f, nf);
	faceNormalOf(m, g, ng);
	const normalise = (n: number[]) => {
		const len = Math.hypot(n[0], n[1], n[2]);
		if (len > 0) for (let k = 0; k < 3; k++) n[k] /= len;
	};
	normalise(nf);
	normalise(ng);
	const cosang = Math.min(1, Math.max(-1, nf[0] * ng[0] + nf[1] * ng[1] + nf[2] * ng[2]));
	const angle = Math.acos(cosang);

	// The vertex of the neighbour that is not on the shared edge. If it lies
	// on the *outside* of this face's plane the pair is concave.
	const opposite = m.fv(g, (ge + 2) % 3);
	const a = m.fv(f, e);
	const d = [m.vx(opposite) - m.vx(a), m.vy(opposite) - m.vy(a), m.vz(opposite) - m.vz(a)];
	const side = d[0] * nf[0] + d[1] * nf[1] + d[2] * nf[2];
	return side > 0 ? -angle : angle;
}

/**
 * Marks every edge whose dihedral angle falls outside `[negative, positive]`.
 *
 * Needs FF adjacency.
 */
export function faceEdgeSelSignedCrease(
	m: CMeshO,
	negative: number,
	positive: number,
	markBorder = false,
): void {
	for (let f = 0; f < m.faceSize; f++) m.faceFlags[f] &= ~FaceFlag.FACEEDGESEL012;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(f, e)) {
				if (markBorder) m.faceFlags[f] |= faceEdgeSelBit(e);
				continue;
			}
			const angle = dihedralAngleRad(m, f, e);
			if (angle < negative || angle > positive) m.faceFlags[f] |= faceEdgeSelBit(e);
		}
	}
}

/**
 * Duplicates vertices so that every selected edge becomes a boundary.
 *
 * Walks the fan of faces around each vertex and starts a new copy whenever it
 * steps over a marked edge. On a boundary vertex the walk has to begin at the
 * boundary and on an interior one at a crease — starting anywhere else would
 * cross the same crease twice and split the fan into one piece too many.
 *
 * Assumes FF adjacency and a two-manifold mesh; the caller checks the latter.
 */
export function cutMeshAlongSelectedFaceEdges(m: CMeshO): void {
	Allocator.compactEveryVector(m);
	if (m.ffFace === null) faceFace(m);

	const wedgeVert = new Int32Array(3 * m.faceSize).fill(-1);
	let newVertexCounter = m.vn;
	const startVn = m.vn;
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.VISITED;

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let j = 0; j < 3; j++) {
			const centre = m.fv(f, j);
			if ((m.vertFlags[centre] & VertexFlag.VISITED) !== 0) continue;
			m.vertFlags[centre] |= VertexFlag.VISITED;

			const start = new Pos(m, f, j, centre);
			const cur = start.clone();
			let onBorder = false;
			// Rotate backwards looking for a boundary. If there is one, the fan
			// is an arc and we must start at its end.
			do {
				cur.flipF();
				cur.flipE();
				if (cur.isBorder()) {
					onBorder = true;
					break;
				}
			} while (!cur.equals(start));

			let from = cur.clone();
			if (!onBorder) {
				// A closed fan: start at a crease instead, so that the walk ends
				// where it began rather than in the middle of a smooth patch.
				do {
					cur.flipF();
					cur.flipE();
					if (isEdgeSelected(m, cur.f, cur.z)) break;
				} while (!cur.equals(from));
				from = cur.clone();
			}

			let current = centre;
			// Crossing a crease means the *next* wedge gets a fresh vertex, and
			// the allocation happens on the crossing itself — exactly as
			// upstream does it. On a closed fan the final crease closes the
			// loop after the counter has already moved on, so every cut corner
			// leaves one allocated-but-unreferenced vertex behind: eight on a
			// cube. Deliberately reproduced. An earlier version deferred the
			// allocation to the moment a wedge was written, which is tidier and
			// leaves no orphans — and the differential tests flagged it,
			// because real MeshLab ships the orphans (32 vertices on that cube,
			// not 24) and a caller counting vertices sees the difference.
			const walk = cur.clone();
			do {
				wedgeVert[3 * walk.f + cornerOf(m, walk)] = current;
				walk.flipE();
				if (isEdgeSelected(m, walk.f, walk.z)) {
					current = newVertexCounter;
					newVertexCounter++;
				}
				walk.flipF();
			} while (!from.equals(walk) && !walk.isBorder());
		}
	}

	const added = newVertexCounter - m.vn;
	if (added > 0) Allocator.addVertices(m, added);

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let j = 0; j < 3; j++) {
			const target = wedgeVert[3 * f + j];
			// A face can be missed entirely when the mesh is not manifold; the
			// caller rejects that case, so leaving the corner alone is enough.
			if (target < 0 || target < startVn) continue;
			const source = m.fv(f, j);
			copyVertex(m, source, target);
			m.faceVert[3 * f + j] = target;
		}
	}
	m.imark++;
}

/** Which corner of `p.f` holds `p.v`. */
function cornerOf(m: CMeshO, p: Pos): number {
	for (let k = 0; k < 3; k++) if (m.fv(p.f, k) === p.v) return k;
	return p.z;
}

function copyVertex(m: CMeshO, from: number, to: number): void {
	m.setVert(to, m.vx(from), m.vy(from), m.vz(from));
	for (let k = 0; k < 3; k++) m.vertNormal[3 * to + k] = m.vertNormal[3 * from + k];
	m.vertQuality[to] = m.vertQuality[from];
	m.vertColor[to] = m.vertColor[from];
	// Deliberately not the flags: the copy must not inherit VISITED, or a
	// later pass would skip a vertex that has never been walked.
	if (m.vertTexCoord !== null) {
		m.vertTexCoord[2 * to] = m.vertTexCoord[2 * from];
		m.vertTexCoord[2 * to + 1] = m.vertTexCoord[2 * from + 1];
	}
	if (m.vertRadius !== null) m.vertRadius[to] = m.vertRadius[from];
	for (const attr of m.customAttrs) {
		if (attr.domain !== "vert") continue;
		for (let k = 0; k < attr.arity; k++) {
			attr.data[attr.arity * to + k] = attr.data[attr.arity * from + k];
		}
	}
}

/** Cuts the mesh wherever two faces meet at more than `angleRad`. */
export function creaseCut(m: CMeshO, angleRad: number): void {
	if (m.ffFace === null) faceFace(m);
	faceEdgeSelSignedCrease(m, -angleRad, angleRad);
	cutMeshAlongSelectedFaceEdges(m);
}

export const CreaseCut = {
	creaseCut,
	cutMeshAlongSelectedFaceEdges,
	faceEdgeSelSignedCrease,
	dihedralAngleRad,
} as const;
