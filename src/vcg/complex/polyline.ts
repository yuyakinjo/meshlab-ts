/**
 * Extracting polylines from a surface.
 *
 * All three callers want the same shape of answer — a set of line segments,
 * delivered as an edge mesh with no faces — and differ only in which segments.
 * A perimeter is the boundary of a face selection, a crease polyline is the
 * marked edges, and a planar section is where a plane crosses the surface.
 *
 * The segments come out unordered and with duplicated endpoints; welding them
 * into shared vertices is left to the caller, which does it with the same
 * `RemoveDuplicateVertex` any other mesh would use. That keeps this file about
 * geometry rather than about bookkeeping.
 */
import { Allocator } from "./allocator.ts";
import { CMeshO } from "./cmesho.ts";
import { FaceFlag } from "./flags.ts";
import { faceFace } from "./update/topology.ts";

/** Appends one segment, giving it two fresh vertices. */
function addSegment(out: CMeshO, a: readonly number[], b: readonly number[]): void {
	const v = Allocator.addVertices(out, 2);
	out.setVert(v, a[0], a[1], a[2]);
	out.setVert(v + 1, b[0], b[1], b[2]);
	Allocator.addEdge(out, v, v + 1);
}

const cornerOf = (m: CMeshO, f: number, k: number): number[] => {
	const v = m.fv(f, k);
	return [m.vx(v), m.vy(v), m.vz(v)];
};

/**
 * The boundary of the selected faces.
 *
 * An edge is on the perimeter when the face across it is missing or not
 * selected — which is the same test that finds a mesh's own boundary, applied
 * to the selection instead of to the mesh.
 */
export function selectionPerimeter(m: CMeshO): CMeshO {
	if (m.ffFace === null) faceFace(m);
	const out = new CMeshO();
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || !m.isFaceS(f)) continue;
		for (let e = 0; e < 3; e++) {
			const across = m.ffp(f, e);
			if (!m.isBorderFF(f, e) && !m.isFaceD(across) && m.isFaceS(across)) continue;
			addSegment(out, cornerOf(m, f, e), cornerOf(m, f, (e + 1) % 3));
		}
	}
	return out;
}

/**
 * The edges carrying the per-face-edge selection bit.
 *
 * That bit is what `Select Crease Edges` sets, so this turns a crease
 * selection into something that can be exported and looked at. Each interior
 * crease is marked from both of its faces, so the lower face index emits it
 * and the other skips — otherwise every segment would appear twice.
 */
export function polylineFromFaceEdgeSelection(m: CMeshO): CMeshO {
	if (m.ffFace === null) faceFace(m);
	const out = new CMeshO();
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if ((m.faceFlags[f] & (FaceFlag.FACEEDGESEL0 << e)) === 0) continue;
			if (!m.isBorderFF(f, e) && m.ffp(f, e) < f) continue;
			addSegment(out, cornerOf(m, f, e), cornerOf(m, f, (e + 1) % 3));
		}
	}
	return out;
}

/**
 * Where a plane crosses the surface.
 *
 * Each face is examined on its own: the plane meets it in a segment joining
 * the two edges whose endpoints straddle the plane. A vertex exactly on the
 * plane is treated as being on the positive side, which is what stops a face
 * touching the plane at one corner from emitting a zero-length segment.
 */
export function planarSection(m: CMeshO, normal: readonly number[], offset: number): CMeshO {
	const out = new CMeshO();
	const len = Math.hypot(normal[0], normal[1], normal[2]);
	if (len === 0) return out;
	const n = [normal[0] / len, normal[1] / len, normal[2] / len];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => cornerOf(m, f, k));
		const d = p.map((q) => n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - offset);
		const crossings: number[][] = [];
		for (let k = 0; k < 3; k++) {
			const a = d[k];
			const b = d[(k + 1) % 3];
			// Straddling, counting "exactly on the plane" as positive so each
			// crossing is found by exactly one of the face's edges.
			if (a >= 0 === b >= 0) continue;
			const t = a / (a - b);
			const pa = p[k];
			const pb = p[(k + 1) % 3];
			crossings.push([0, 1, 2].map((c) => pa[c] + (pb[c] - pa[c]) * t));
		}
		// A triangle can only be crossed in zero or two of its edges; anything
		// else is the degenerate case the epsilon-free test above avoids.
		if (crossings.length === 2) addSegment(out, crossings[0], crossings[1]);
	}
	return out;
}

export const Polyline = {
	selectionPerimeter,
	polylineFromFaceEdgeSelection,
	planarSection,
} as const;
