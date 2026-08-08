/**
 * Snapping mismatched borders, from MeshLab's `cleanfilter.cpp`.
 *
 * The problem: two range scans of the same object meet along a seam, but their
 * triangulations do not line up — a vertex of one sheet sits partway along an
 * edge of the other. The two borders are geometrically coincident and
 * topologically unrelated, so no amount of vertex merging will join them.
 *
 * The fix is to split the offending edge at that vertex. That alone does not
 * weld anything; it makes the two borders *compatible*, so that a subsequent
 * "Merge Close Vertices" can. Splitting and welding are deliberately separate
 * steps, because the split is safe and the weld is not.
 */

import { SurfaceLookup } from "../space/index/surface_lookup.ts";
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { VertexFlag } from "./flags.ts";
import { UpdateBounding } from "./update/bounding.ts";
import { faceBorderFromFF, vertexBorderFromFace } from "./update/flag.ts";
import { UpdateNormal } from "./update/normal.ts";
import { faceFace } from "./update/topology.ts";

/**
 * Splits the border edges that a border vertex of another sheet falls on.
 *
 * `threshold` is a ratio, not a distance: a vertex is snapped onto an edge
 * when its distance to that edge is below `threshold` times the edge's length.
 * Scale-free on purpose — a seam between two fine patches and one between two
 * coarse patches need the same setting.
 *
 * Returns how many faces were split.
 */
export function snapVertexBorder(m: CMeshO, threshold: number): number {
	Allocator.compactEveryVector(m);
	faceFace(m);
	faceBorderFromFF(m);
	vertexBorderFromFace(m);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
	UpdateBounding.box(m);

	const maxDist = (m.bbox.diagonal || 1) / 20;
	const lookup = new SurfaceLookup(m, maxDist);
	// One split per face at most: a face that has already been chosen has its
	// edge indices about to change, so a second vertex claiming the same face
	// would split an edge that no longer means what it did.
	const claimed = new Uint8Array(m.faceSize);

	const splits: Array<{ face: number; edge: number; x: number; y: number; z: number }> = [];
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || (m.vertFlags[v] & VertexFlag.BORDER) === 0) continue;
		const x = m.vx(v);
		const y = m.vy(v);
		const z = m.vz(v);

		let bestFace = -1;
		let bestEdge = -1;
		let bestDist = Number.POSITIVE_INFINITY;
		for (const hit of lookup.closestWithin(x, y, z, maxDist)) {
			const f = hit.face;
			if (claimed[f] === 1) continue;
			// The vertex must land on the *interior of a border edge*: two
			// barycentric coordinates clearly positive and the third at zero.
			// Landing near a corner is not a mismatch, it is the same vertex.
			for (let e = 0; e < 3; e++) {
				if (!m.isFaceB(f, e)) continue;
				const bary = hit.bary;
				if (bary[e] > BIG && bary[(e + 1) % 3] > BIG && bary[(e + 2) % 3] < SMALL) {
					const d = Math.hypot(hit.x - x, hit.y - y, hit.z - z);
					if (d < bestDist) {
						bestDist = d;
						bestFace = f;
						bestEdge = e;
					}
				}
			}
		}
		if (bestFace < 0) continue;

		const a = m.fv(bestFace, bestEdge);
		const b = m.fv(bestFace, (bestEdge + 1) % 3);
		const edgeLen = Math.hypot(m.vx(a) - m.vx(b), m.vy(a) - m.vy(b), m.vz(a) - m.vz(b));
		if (bestDist >= threshold * edgeLen) continue;
		claimed[bestFace] = 1;
		m.vertFlags[v] |= VertexFlag.SELECTED;
		splits.push({ face: bestFace, edge: bestEdge, x, y, z });
	}

	if (splits.length === 0) return 0;

	//         ^                    ^
	//       /   \                /  |  \
	//     /  fp   \            /  fp | ff \
	//   V0 ------- V2        V0 ---- fv --- V2
	//        e
	const firstVert = Allocator.addVertices(m, splits.length);
	const firstFace = Allocator.addFaces(m, splits.length);
	splits.forEach((s, i) => {
		const nv = firstVert + i;
		m.setVert(nv, s.x, s.y, s.z);
		const v0 = m.fv(s.face, s.edge);
		const v2 = m.fv(s.face, (s.edge + 2) % 3);
		m.setFace(firstFace + i, nv, v2, v0);
		m.faceVert[3 * s.face + s.edge] = nv;
	});
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
	m.imark++;
	return splits.length;
}

// Upstream's epsilons: clearly inside the edge, and clearly on it.
const BIG = 1e-2;
const SMALL = 1e-5;

export const SnapBorder = { snapVertexBorder } as const;
