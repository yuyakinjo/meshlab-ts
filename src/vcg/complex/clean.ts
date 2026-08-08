/**
 * `Clean` — mesh analysis and repair, mirroring `vcg::tri::Clean`.
 *
 * Two halves: the analysis functions (counting edges, holes, components and
 * non-manifold features; deciding orientability) and the removal and repair
 * functions behind `filter_clean`.
 *
 * A convention worth knowing before reading any of the removal functions: they
 * *mark* elements deleted and leave compaction to the caller, exactly as
 * VCGLib does. That is what lets several of them run in sequence over stable
 * indices. The filters call `Allocator.compactEveryVector` at the end.
 */
import { MLInternalException } from "../../common/utilities/ml_exception.ts";
import { KdTree } from "../space/index/kdtree.ts";
import {
	intersectionSegmentTriangle,
	intersectionTriangleTriangle,
} from "../space/intersection3.ts";
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { faceClearV } from "./update/flag.ts";
import { faceNormalOf } from "./update/normal.ts";
import { faceFace, fillSortedEdgeVector, isManifoldEdge } from "./update/topology.ts";

export interface EdgeCounts {
	/** Distinct undirected edges. */
	total: number;
	/** Edges used by exactly one face. */
	boundary: number;
	/** Edges used by three or more faces. */
	nonManifold: number;
}

/**
 * Counts the edges by how many faces use each one.
 *
 * Needs no adjacency — it sorts the (face, edge) pairs, which is why VCGLib
 * uses it as the basis of `IsWaterTight`.
 */
export function countEdgeNum(m: CMeshO): EdgeCounts {
	const sorted = fillSortedEdgeVector(m);
	let total = 0;
	let boundary = 0;
	let nonManifold = 0;
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		total++;
		const uses = j - i;
		if (uses === 1) boundary++;
		else if (uses > 2) nonManifold++;
		i = j;
	}
	return { total, boundary, nonManifold };
}

/**
 * No boundary edges and no non-manifold edges.
 *
 * Orientability is deliberately not part of the test, matching VCGLib: it
 * would require FF adjacency, and this check requires none.
 */
export function isWaterTight(m: CMeshO): boolean {
	const { boundary, nonManifold } = countEdgeNum(m);
	return boundary === 0 && nonManifold === 0;
}

/** Counts edges used by three or more faces, optionally selecting their faces. */
export function countNonManifoldEdgeFF(m: CMeshO, selectFlag = false): number {
	const sorted = fillSortedEdgeVector(m);
	let count = 0;
	let i = 0;
	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		if (j - i > 2) {
			count++;
			if (selectFlag) for (let k = i; k < j; k++) m.faceFlags[sorted[k].f] |= FaceFlag.SELECTED;
		}
		i = j;
	}
	return count;
}

/**
 * Counts vertices whose incident faces do not form a single fan — the
 * "bowtie" case, where a vertex joins two otherwise separate patches.
 *
 * Works from the face-vertex table rather than from adjacency: for each
 * vertex, the incident face corners are grouped by the edges they share, and
 * the vertex is non-manifold when more than one group results.
 */
export function countNonManifoldVertexFF(m: CMeshO): number {
	// vertex -> the faces touching it
	const incident: number[][] = Array.from({ length: m.vertSize }, () => []);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) incident[m.faceVert[3 * f + k]].push(f);
	}

	let count = 0;
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const faces = incident[v];
		if (faces.length < 2) continue;

		// Union faces that share an edge *through v*.
		const index = new Map<number, number>(faces.map((f, i) => [f, i]));
		const parent = faces.map((_, i) => i);
		const find = (x: number): number => {
			let r = x;
			while (parent[r] !== r) r = parent[r];
			let cur = x;
			while (parent[cur] !== r) {
				const nxt = parent[cur];
				parent[cur] = r;
				cur = nxt;
			}
			return r;
		};
		const edgeOwner = new Map<number, number>();
		for (const f of faces) {
			for (let k = 0; k < 3; k++) {
				const a = m.faceVert[3 * f + k];
				const b = m.faceVert[3 * f + ((k + 1) % 3)];
				if (a !== v && b !== v) continue;
				const other = a === v ? b : a;
				const owner = edgeOwner.get(other);
				const me = index.get(f) as number;
				if (owner === undefined) edgeOwner.set(other, me);
				else {
					const ra = find(owner);
					const rb = find(me);
					if (ra !== rb) parent[ra] = rb;
				}
			}
		}
		const groups = new Set(faces.map((_, i) => find(i)));
		if (groups.size > 1) count++;
	}
	return count;
}

/**
 * Counts boundary loops by walking each one with a {@link Pos}. Requires FF.
 *
 * VCGLib names this `CountHoles`; a "hole" here is a boundary loop, so a disk
 * has one and a closed surface has none.
 */
export function countHoles(m: CMeshO): number {
	if (m.ffFace === null) {
		throw new MLInternalException("countHoles requires FF adjacency (MM_FACEFACETOPO)");
	}
	faceClearV(m);
	let loops = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if ((m.faceFlags[f] & FaceFlag.VISITED) !== 0) break;
			if (!m.isBorderFF(f, e)) continue;
			const start = Pos.onEdge(m, f, e);
			const cur = start.clone();
			do {
				cur.nextB();
				m.faceFlags[cur.f] |= FaceFlag.VISITED;
			} while (!cur.equals(start));
			loops++;
		}
	}
	return loops;
}

/**
 * The connected components, as `[faceCount, seedFace]` pairs. Requires FF.
 *
 * Faces are connected when they share an edge; a bowtie vertex therefore does
 * *not* join two components, which is the same convention VCGLib uses.
 */
export function connectedComponents(m: CMeshO): Array<[number, number]> {
	if (m.ffFace === null) {
		throw new MLInternalException("connectedComponents requires FF adjacency (MM_FACEFACETOPO)");
	}
	faceClearV(m);
	const out: Array<[number, number]> = [];
	const stack: number[] = [];
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || (m.faceFlags[f] & FaceFlag.VISITED) !== 0) continue;
		m.faceFlags[f] |= FaceFlag.VISITED;
		let size = 0;
		stack.push(f);
		while (stack.length > 0) {
			const cur = stack.pop() as number;
			size++;
			for (let e = 0; e < 3; e++) {
				// Walk the whole FF ring so that non-manifold edges still join
				// every face that meets along them.
				let nf = m.ffp(cur, e);
				let ne = m.ffi(cur, e);
				while (nf !== cur || ne !== e) {
					if (!m.isFaceD(nf) && (m.faceFlags[nf] & FaceFlag.VISITED) === 0) {
						m.faceFlags[nf] |= FaceFlag.VISITED;
						stack.push(nf);
					}
					const tf = m.ffp(nf, ne);
					const te = m.ffi(nf, ne);
					nf = tf;
					ne = te;
				}
			}
		}
		out.push([size, f]);
	}
	return out;
}

export function countConnectedComponents(m: CMeshO): number {
	return connectedComponents(m).length;
}

/**
 * Genus from the Euler characteristic.
 *
 * `V - E + F = 2C - 2G - B`, so `G = -(V - E + F + B - 2C) / 2`, where B is
 * the number of boundary loops and C the number of connected components.
 *
 * Only meaningful for orientable surfaces; on a Möbius strip it returns a
 * half-integer, which is the formula honestly reporting that it does not
 * apply.
 */
export function meshGenus(
	nvert: number,
	nedges: number,
	nfaces: number,
	numholes: number,
	numcomponents: number,
): number {
	const g = -(nvert + nfaces - nedges + numholes - 2 * numcomponents) / 2;
	// Negating zero yields -0, which compares unequal to 0 under Object.is and
	// therefore under most test matchers. Genus 0 is by far the common answer,
	// so hand back a plain zero.
	return g === 0 ? 0 : g;
}

/** True when the two faces along edge `e` of `f` traverse it in opposite directions. */
export function checkOrientation(m: CMeshO, f: number, e: number): boolean {
	if (m.isBorderFF(f, e)) return true; // a border edge is vacuously fine
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	// Coherent means face g sees the same edge with its endpoints swapped.
	return m.fv(f, e) === m.fv(g, (ge + 1) % 3) && m.fv(f, (e + 1) % 3) === m.fv(g, ge);
}

/** True when every manifold internal edge is traversed coherently. Requires FF. */
export function isCoherentlyOrientedMesh(m: CMeshO): boolean {
	if (m.ffFace === null) {
		throw new MLInternalException(
			"isCoherentlyOrientedMesh requires FF adjacency (MM_FACEFACETOPO)",
		);
	}
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!isManifoldEdge(m, f, e)) continue;
			if (!checkOrientation(m, f, e)) return false;
		}
	}
	return true;
}

/** Reverses the winding of every face, or only the selected ones. */
export function flipMesh(m: CMeshO, selected = false): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (selected && !m.isFaceS(f)) continue;
		const b = m.faceVert[3 * f + 1];
		m.faceVert[3 * f + 1] = m.faceVert[3 * f + 2];
		m.faceVert[3 * f + 2] = b;
	}
	m.imark++;
}

/**
 * The volume the mesh encloses, signed by its winding.
 *
 * Positive when the faces are wound outward. Only meaningful for a closed,
 * coherently oriented surface.
 */
export function signedVolume(m: CMeshO): number {
	let total = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		total +=
			m.vx(a) * (m.vy(b) * m.vz(c) - m.vz(b) * m.vy(c)) -
			m.vy(a) * (m.vx(b) * m.vz(c) - m.vz(b) * m.vx(c)) +
			m.vz(a) * (m.vx(b) * m.vy(c) - m.vy(b) * m.vx(c));
	}
	return total / 6;
}

/**
 * Flips the whole mesh if it is inside out, and reports whether it did.
 *
 * Decided by the sign of the enclosed volume, which is exact for a closed
 * coherently oriented surface. On an open surface "outside" is not defined and
 * the volume is meaningless, so nothing is flipped — the caller should reorient
 * and close the holes first.
 */
export function flipNormalOutside(m: CMeshO): boolean {
	if (m.fn === 0) return false;
	if (!isWaterTight(m)) return false;
	if (signedVolume(m) >= 0) return false;
	flipMesh(m);
	return true;
}

/** Swaps two of a face's vertices, and repairs the FF links that move with them. */
function swapEdge(m: CMeshO, f: number): void {
	// Swapping corners 1 and 2 reverses the winding; edges 0 and 2 exchange
	// places while edge 1 keeps its endpoints.
	const v1 = m.faceVert[3 * f + 1];
	m.faceVert[3 * f + 1] = m.faceVert[3 * f + 2];
	m.faceVert[3 * f + 2] = v1;

	const ff = m.ffFace as Int32Array;
	const fe = m.ffEdge as Uint8Array;
	const f0 = ff[3 * f];
	const e0 = fe[3 * f];
	ff[3 * f] = ff[3 * f + 2];
	fe[3 * f] = fe[3 * f + 2];
	ff[3 * f + 2] = f0;
	fe[3 * f + 2] = e0;

	// Fix the back-links of the two neighbours we just moved.
	for (const e of [0, 2]) {
		const g = ff[3 * f + e];
		const ge = fe[3 * f + e];
		if (g === f) {
			// A self-loop must point at its own new edge index.
			ff[3 * f + e] = f;
			fe[3 * f + e] = e;
			continue;
		}
		ff[3 * g + ge] = f;
		fe[3 * g + ge] = e;
	}
}

export interface OrientResult {
	/** The mesh was already coherently oriented; nothing was changed. */
	isOriented: boolean;
	/** A coherent orientation exists at all. False for a Möbius strip. */
	isOrientable: boolean;
}

/**
 * Makes the mesh coherently oriented by flooding an orientation across shared
 * edges, and reports whether that was possible. Requires FF.
 *
 * When the surface is non-orientable the flood reaches a face that is already
 * fixed and disagrees; `isOrientable` comes back false and the mesh is left
 * partially reoriented, exactly as VCGLib leaves it.
 */
export function orientCoherentlyMesh(m: CMeshO): OrientResult {
	if (m.ffFace === null) {
		throw new MLInternalException("orientCoherentlyMesh requires FF adjacency (MM_FACEFACETOPO)");
	}
	let isOriented = true;
	let isOrientable = true;
	faceClearV(m);
	const stack: number[] = [];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || (m.faceFlags[f] & FaceFlag.VISITED) !== 0) continue;
		m.faceFlags[f] |= FaceFlag.VISITED;
		stack.push(f);
		while (stack.length > 0) {
			const fp = stack.pop() as number;
			for (let e = 0; e < 3; e++) {
				if (m.isBorderFF(fp, e)) continue; // border
				if (!isManifoldEdge(m, fp, e)) continue;
				const g = m.ffp(fp, e);
				const ge = m.ffi(fp, e);
				if (!checkOrientation(m, g, ge)) {
					isOriented = false;
					if ((m.faceFlags[g] & FaceFlag.VISITED) === 0) {
						swapEdge(m, g);
					} else {
						// Already fixed and still disagreeing: no consistent
						// orientation exists on this component.
						isOrientable = false;
						break;
					}
				}
				if ((m.faceFlags[g] & FaceFlag.VISITED) === 0) {
					m.faceFlags[g] |= FaceFlag.VISITED;
					stack.push(g);
				}
			}
		}
	}
	m.imark++;
	return { isOriented, isOrientable };
}

// ---------------------------------------------------------------------------
// Removal and repair
// ---------------------------------------------------------------------------

/**
 * Merges vertices that share the *exact* same coordinates, rewriting the faces
 * that referenced the duplicates.
 *
 * Exact equality, not a tolerance — that is what `mergeCloseVertex` is for.
 * This is the filter an STL always needs first, since the format stores every
 * triangle's corners separately and a welded cube arrives as 36 vertices.
 *
 * Returns the number of vertices deleted. With `removeDegenerateFlag`, faces
 * that collapsed to a line or a point as a result are deleted too, matching
 * VCGLib.
 */
export function removeDuplicateVertex(m: CMeshO, removeDegenerateFlag = true): number {
	if (m.vertSize === 0 || m.vn === 0) return 0;

	// Group by coordinate. VCGLib sorts an array of pointers; a hash on the
	// coordinate triple is the same idea without the O(n log n).
	const survivorOf = new Map<string, number>();
	const remap = new Int32Array(m.vertSize).fill(-1);
	let deleted = 0;

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const key = `${m.vx(v)},${m.vy(v)},${m.vz(v)}`;
		const first = survivorOf.get(key);
		if (first === undefined) {
			survivorOf.set(key, v);
			continue;
		}
		remap[v] = first;
		Allocator.deleteVertex(m, v);
		deleted++;
	}
	if (deleted === 0) return 0;

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const survivor = remap[m.faceVert[3 * f + k]];
			if (survivor >= 0) m.faceVert[3 * f + k] = survivor;
		}
	}
	// Edges reference vertices as well, and welding a polyline is exactly what
	// turns a heap of disconnected segments into one. Leaving them unremapped
	// would point every edge at a vertex that had just been deleted.
	for (let e = 0; e < m.edgeSize; e++) {
		if (m.isEdgeD(e)) continue;
		for (let k = 0; k < 2; k++) {
			const survivor = remap[m.edgeVert[2 * e + k]];
			if (survivor >= 0) m.edgeVert[2 * e + k] = survivor;
		}
	}
	// A segment whose two ends welded together has no length left.
	for (let e = 0; e < m.edgeSize; e++) {
		if (!m.isEdgeD(e) && m.ev(e, 0) === m.ev(e, 1)) Allocator.deleteEdge(m, e);
	}

	if (removeDegenerateFlag) removeDegenerateFace(m);
	m.imark++;
	return deleted;
}

/**
 * Deletes faces that reference the same vertex more than once.
 *
 * Topologically degenerate rather than geometrically: a face with a repeated
 * index owns a self-edge, which makes adjacency ill-defined. See
 * {@link removeFaceOutOfRangeArea} for the zero-area case.
 */
export function removeDegenerateFace(m: CMeshO): number {
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.faceVert[3 * f];
		const b = m.faceVert[3 * f + 1];
		const c = m.faceVert[3 * f + 2];
		if (a === b || b === c || a === c) {
			Allocator.deleteFace(m, f);
			count++;
		}
	}
	return count;
}

/** Deletes faces whose area falls outside `[minAreaThr, maxAreaThr]`. */
export function removeFaceOutOfRangeArea(
	m: CMeshO,
	minAreaThr = 0,
	maxAreaThr = Number.POSITIVE_INFINITY,
	onlyOnSelected = false,
): number {
	const n = new Float64Array(3);
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (onlyOnSelected && !m.isFaceS(f)) continue;
		faceNormalOf(m, f, n);
		const area = Math.hypot(n[0], n[1], n[2]) / 2;
		if (area <= minAreaThr || area >= maxAreaThr) {
			Allocator.deleteFace(m, f);
			count++;
		}
	}
	return count;
}

/**
 * Deletes faces of exactly zero area.
 *
 * Note the boundary condition inherited from VCGLib: the test is `area <= 0`,
 * so a face of area exactly zero goes, and nothing else does.
 */
export function removeZeroAreaFace(m: CMeshO): number {
	return removeFaceOutOfRangeArea(m, 0);
}

/** Deletes faces that repeat another face's three vertices, in any order. */
export function removeDuplicateFace(m: CMeshO): number {
	const seen = new Set<string>();
	let count = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const key = [m.faceVert[3 * f], m.faceVert[3 * f + 1], m.faceVert[3 * f + 2]]
			.sort((a, b) => a - b)
			.join("_");
		if (seen.has(key)) {
			Allocator.deleteFace(m, f);
			count++;
		} else {
			seen.add(key);
		}
	}
	return count;
}

/** Counts vertices no live face references. */
export function countUnreferencedVertex(m: CMeshO): number {
	return removeUnreferencedVertex(m, false);
}

/**
 * Deletes vertices no live face references, or just counts them when
 * `deleteVertexFlag` is false.
 */
export function removeUnreferencedVertex(m: CMeshO, deleteVertexFlag = true): number {
	const referenced = new Uint8Array(m.vertSize);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) referenced[m.faceVert[3 * f + k]] = 1;
	}
	let count = 0;
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || referenced[v] === 1) continue;
		count++;
		if (deleteVertexFlag) Allocator.deleteVertex(m, v);
	}
	return count;
}

/**
 * Snaps vertices closer than `radius` onto a shared position, then merges
 * them.
 *
 * Uses a uniform spatial hash sized to the radius, so each vertex only
 * compares against the 27 cells that could hold a neighbour within range.
 * Unlike {@link removeDuplicateVertex} this moves geometry, which is why the
 * threshold defaults so small in the filter.
 */
export function mergeCloseVertex(m: CMeshO, radius: number): number {
	if (radius <= 0 || m.vn === 0) return 0;
	const cell = radius;
	const buckets = new Map<string, number[]>();
	const keyOf = (x: number, y: number, z: number) =>
		`${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const key = keyOf(m.vx(v), m.vy(v), m.vz(v));
		const hit = buckets.get(key);
		if (hit === undefined) buckets.set(key, [v]);
		else hit.push(v);
	}

	const r2 = radius * radius;
	const snapped = new Uint8Array(m.vertSize);
	let merged = 0;

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || snapped[v] === 1) continue;
		const cx = Math.floor(m.vx(v) / cell);
		const cy = Math.floor(m.vy(v) / cell);
		const cz = Math.floor(m.vz(v) / cell);
		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				for (let dz = -1; dz <= 1; dz++) {
					for (const w of buckets.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
						if (w === v || snapped[w] === 1 || m.isVertD(w)) continue;
						const ex = m.vx(w) - m.vx(v);
						const ey = m.vy(w) - m.vy(v);
						const ez = m.vz(w) - m.vz(v);
						if (ex * ex + ey * ey + ez * ez >= r2) continue;
						// Snap onto v's position so the pair becomes an exact
						// duplicate, which removeDuplicateVertex then welds.
						m.setVert(w, m.vx(v), m.vy(v), m.vz(v));
						snapped[w] = 1;
						merged++;
					}
				}
			}
		}
		snapped[v] = 1;
	}

	if (merged > 0) removeDuplicateVertex(m);
	return merged;
}

/** The faces of the component seeded at `seed`, walked over FF adjacency. */
function componentFaces(m: CMeshO, seed: number): number[] {
	const out: number[] = [];
	const seen = new Set<number>([seed]);
	const stack = [seed];
	while (stack.length > 0) {
		const f = stack.pop() as number;
		out.push(f);
		for (let e = 0; e < 3; e++) {
			let nf = m.ffp(f, e);
			let ne = m.ffi(f, e);
			while (nf !== f || ne !== e) {
				if (!m.isFaceD(nf) && !seen.has(nf)) {
					seen.add(nf);
					stack.push(nf);
				}
				const tf = m.ffp(nf, ne);
				const te = m.ffi(nf, ne);
				nf = tf;
				ne = te;
			}
		}
	}
	return out;
}

export interface ComponentRemovalResult {
	/** Components the mesh had before anything was removed. */
	total: number;
	/** Components deleted. */
	deleted: number;
}

/** Deletes connected components with fewer than `maxCCSize` faces. Requires FF. */
export function removeSmallConnectedComponentsSize(
	m: CMeshO,
	maxCCSize: number,
): ComponentRemovalResult {
	const components = connectedComponents(m);
	let deleted = 0;
	for (const [size, seed] of components) {
		if (size >= maxCCSize) continue;
		deleted++;
		for (const f of componentFaces(m, seed)) {
			if (!m.isFaceD(f)) Allocator.deleteFace(m, f);
		}
	}
	return { total: components.length, deleted };
}

/**
 * Deletes connected components whose bounding-box diagonal is under
 * `maxDiameter`. Requires FF.
 *
 * Size in space rather than in face count, which is what you want for
 * scanner noise: a dense speck can have plenty of triangles.
 */
export function removeSmallConnectedComponentsDiameter(
	m: CMeshO,
	maxDiameter: number,
): ComponentRemovalResult {
	const components = connectedComponents(m);
	let deleted = 0;
	for (const [, seed] of components) {
		const faces = componentFaces(m, seed);
		let minX = Number.POSITIVE_INFINITY;
		let minY = Number.POSITIVE_INFINITY;
		let minZ = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		let maxY = Number.NEGATIVE_INFINITY;
		let maxZ = Number.NEGATIVE_INFINITY;
		for (const f of faces) {
			for (let k = 0; k < 3; k++) {
				const v = m.fv(f, k);
				minX = Math.min(minX, m.vx(v));
				minY = Math.min(minY, m.vy(v));
				minZ = Math.min(minZ, m.vz(v));
				maxX = Math.max(maxX, m.vx(v));
				maxY = Math.max(maxY, m.vy(v));
				maxZ = Math.max(maxZ, m.vz(v));
			}
		}
		const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
		if (diag >= maxDiameter) continue;
		deleted++;
		for (const f of faces) if (!m.isFaceD(f)) Allocator.deleteFace(m, f);
	}
	return { total: components.length, deleted };
}

/**
 * Deletes faces incident on a non-manifold edge, smallest first, stopping as
 * soon as the edge becomes manifold. Requires FF.
 *
 * Smallest-first is VCGLib's heuristic and a good one: the extra sliver
 * glued onto a surface is usually the mistake, and the large faces around it
 * are the real geometry. Each deletion re-checks, so an edge shared by three
 * faces loses exactly one.
 */
export function removeNonManifoldFace(m: CMeshO): number {
	const candidates: Array<{ f: number; area: number }> = [];
	const n = new Float64Array(3);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (isManifoldEdge(m, f, 0) && isManifoldEdge(m, f, 1) && isManifoldEdge(m, f, 2)) continue;
		faceNormalOf(m, f, n);
		candidates.push({ f, area: Math.hypot(n[0], n[1], n[2]) / 2 });
	}
	if (candidates.length === 0) return 0;
	candidates.sort((a, b) => a.area - b.area || a.f - b.f);

	let count = 0;
	for (const { f } of candidates) {
		if (m.isFaceD(f)) continue;
		// Re-test: an earlier deletion may already have made this face's edges
		// manifold, in which case it is legitimate geometry and stays.
		if (isManifoldEdge(m, f, 0) && isManifoldEdge(m, f, 1) && isManifoldEdge(m, f, 2)) continue;
		Allocator.deleteFace(m, f);
		count++;
		// The rings have to be rebuilt for the next re-test to mean anything.
		faceFace(m);
	}
	return count;
}

/**
 * Splits vertices whose incident faces form more than one fan, so that each
 * fan gets its own copy.
 *
 * This is the "bowtie" repair. `moveThreshold` nudges each new vertex toward
 * the barycentre of its own fan by that fraction, which separates the copies
 * visibly; zero leaves them coincident.
 *
 * Returns the number of vertices split.
 */
export function splitNonManifoldVertex(m: CMeshO, moveThreshold = 0): number {
	// vertex -> the (face, corner) pairs touching it
	const corners: Array<Array<[number, number]>> = Array.from({ length: m.vertSize }, () => []);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) corners[m.faceVert[3 * f + k]].push([f, k]);
	}

	// Splitting appends vertices, so `m.vertSize` grows underneath the loop.
	// Only the vertices that existed when the table was built are candidates;
	// the copies are by construction single-fan.
	const originalVertSize = m.vertSize;
	let split = 0;
	for (let v = 0; v < originalVertSize; v++) {
		if (m.isVertD(v)) continue;
		const inc = corners[v];
		if (inc.length < 2) continue;

		// Group the incident faces into fans: two faces belong to the same fan
		// when they share an edge *through v*.
		const index = new Map<number, number>(inc.map(([f], i) => [f, i]));
		const parent = inc.map((_, i) => i);
		const find = (x: number): number => {
			let r = x;
			while (parent[r] !== r) r = parent[r];
			let cur = x;
			while (parent[cur] !== r) {
				const nxt = parent[cur];
				parent[cur] = r;
				cur = nxt;
			}
			return r;
		};
		const edgeOwner = new Map<number, number>();
		for (const [f] of inc) {
			for (let k = 0; k < 3; k++) {
				const a = m.faceVert[3 * f + k];
				const b = m.faceVert[3 * f + ((k + 1) % 3)];
				if (a !== v && b !== v) continue;
				const other = a === v ? b : a;
				const me = index.get(f) as number;
				const owner = edgeOwner.get(other);
				if (owner === undefined) edgeOwner.set(other, me);
				else {
					const ra = find(owner);
					const rb = find(me);
					if (ra !== rb) parent[ra] = rb;
				}
			}
		}

		const fans = new Map<number, Array<[number, number]>>();
		for (let i = 0; i < inc.length; i++) {
			const root = find(i);
			const hit = fans.get(root);
			if (hit === undefined) fans.set(root, [inc[i]]);
			else hit.push(inc[i]);
		}
		if (fans.size < 2) continue;

		// The first fan keeps the original vertex; every other gets a copy.
		let first = true;
		for (const fan of fans.values()) {
			if (first) {
				first = false;
				if (moveThreshold > 0) displaceTowardFan(m, v, v, fan, moveThreshold);
				continue;
			}
			const copy = Allocator.addVertex(m, m.vx(v), m.vy(v), m.vz(v));
			m.vertQuality[copy] = m.vertQuality[v];
			m.vertColor[copy] = m.vertColor[v];
			for (const [f, k] of fan) m.faceVert[3 * f + k] = copy;
			if (moveThreshold > 0) displaceTowardFan(m, copy, v, fan, moveThreshold);
		}
		split++;
	}
	if (split > 0) m.imark++;
	return split;
}

/** Moves `target` a fraction of the way toward its fan's barycentre. */
function displaceTowardFan(
	m: CMeshO,
	target: number,
	origin: number,
	fan: ReadonlyArray<[number, number]>,
	alpha: number,
): void {
	let bx = 0;
	let by = 0;
	let bz = 0;
	let n = 0;
	for (const [f] of fan) {
		for (let k = 0; k < 3; k++) {
			const w = m.fv(f, k);
			bx += m.vx(w);
			by += m.vy(w);
			bz += m.vz(w);
			n++;
		}
	}
	if (n === 0) return;
	bx /= n;
	by /= n;
	bz /= n;
	m.setVert(
		target,
		m.vx(origin) + (bx - m.vx(origin)) * alpha,
		m.vy(origin) + (by - m.vy(origin)) * alpha,
		m.vz(origin) + (bz - m.vz(origin)) * alpha,
	);
}

/**
 * Splits the mesh into edge-manifold components by duplicating the vertices
 * along non-manifold edges, so no face is deleted.
 *
 * The alternative to {@link removeNonManifoldFace}: it keeps every triangle
 * and pays for it by pulling the sheets apart. Returns the number of
 * components afterwards.
 */
export function splitManifoldComponents(m: CMeshO, moveThreshold = 0): number {
	// Faces meeting along a non-manifold edge are given their own copies of
	// that edge's endpoints, which detaches the sheets.
	const sorted = fillSortedEdgeVector(m);
	let i = 0;
	const duplicated = new Map<string, number>();

	while (i < sorted.length) {
		let j = i + 1;
		while (j < sorted.length && sorted[j].v0 === sorted[i].v0 && sorted[j].v1 === sorted[i].v1) j++;
		if (j - i > 2) {
			// Leave the first two uses on the original vertices; give each
			// further use its own pair, so every sheet ends up separate.
			for (let k = i + 2; k < j; k++) {
				const use = sorted[k];
				for (const endpoint of [sorted[i].v0, sorted[i].v1]) {
					const key = `${use.f}_${endpoint}`;
					let copy = duplicated.get(key);
					if (copy === undefined) {
						copy = Allocator.addVertex(m, m.vx(endpoint), m.vy(endpoint), m.vz(endpoint));
						m.vertQuality[copy] = m.vertQuality[endpoint];
						m.vertColor[copy] = m.vertColor[endpoint];
						duplicated.set(key, copy);
					}
					for (let c = 0; c < 3; c++) {
						if (m.faceVert[3 * use.f + c] === endpoint) m.faceVert[3 * use.f + c] = copy;
					}
				}
			}
		}
		i = j;
	}

	splitNonManifoldVertex(m, moveThreshold);
	faceFace(m);
	m.imark++;
	return countConnectedComponents(m);
}

/**
 * Removes T-vertices — a vertex sitting partway along another face's edge,
 * making a sliver whose base-to-height ratio exceeds `ratio`.
 *
 * `method` selects VCGLib's two strategies: collapsing the sliver's short
 * edge, or flipping it. Collapse removes geometry, flip only rewires, so
 * flip is the conservative choice on a mesh whose vertices matter.
 *
 * Requires FF. Returns the number of slivers dealt with.
 */
export function removeTVertexByCollapse(m: CMeshO, ratio = 40, repeat = true): number {
	let total = 0;
	let pass = 0;
	for (;;) {
		let done = 0;
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			const k = sliverApex(m, f, ratio);
			if (k < 0) continue;
			// Collapse the apex onto the nearest of the base's endpoints.
			const apex = m.fv(f, k);
			const b0 = m.fv(f, (k + 1) % 3);
			const b1 = m.fv(f, (k + 2) % 3);
			const d0 = distanceSquared(m, apex, b0);
			const d1 = distanceSquared(m, apex, b1);
			const keep = d0 <= d1 ? b0 : b1;
			if (keep === apex) continue;
			for (let g = 0; g < m.faceSize; g++) {
				if (m.isFaceD(g)) continue;
				for (let c = 0; c < 3; c++)
					if (m.faceVert[3 * g + c] === apex) m.faceVert[3 * g + c] = keep;
			}
			done++;
		}
		removeDegenerateFace(m);
		removeUnreferencedVertex(m);
		total += done;
		if (done === 0 || !repeat || ++pass > 10) break;
		faceFace(m);
	}
	if (total > 0) {
		faceFace(m);
		m.imark++;
	}
	return total;
}

/**
 * Removes T-vertices by flipping the sliver's base edge to the opposite
 * diagonal of the quad it shares with its neighbour. Requires FF.
 */
export function removeTVertexByFlip(m: CMeshO, ratio = 40, repeat = true): number {
	let total = 0;
	let pass = 0;
	for (;;) {
		let done = 0;
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			const k = sliverApex(m, f, ratio);
			if (k < 0) continue;
			// The base is the edge opposite the apex; flipping needs a
			// manifold neighbour across it.
			const base = (k + 1) % 3;
			if (m.isBorderFF(f, base) || !isManifoldEdge(m, f, base)) continue;
			if (!flipEdge(m, f, base)) continue;
			done++;
			faceFace(m);
		}
		total += done;
		if (done === 0 || !repeat || ++pass > 10) break;
	}
	if (total > 0) m.imark++;
	return total;
}

/**
 * The corner of `f` opposite its longest edge, when the face is a sliver whose
 * base-to-height ratio exceeds `ratio`. -1 when the face is well shaped.
 */
function sliverApex(m: CMeshO, f: number, ratio: number): number {
	let longest = -1;
	let longestLen = -1;
	for (let e = 0; e < 3; e++) {
		const len = distanceSquared(m, m.fv(f, e), m.fv(f, (e + 1) % 3));
		if (len > longestLen) {
			longestLen = len;
			longest = e;
		}
	}
	if (longestLen <= 0) return -1;
	const n = new Float64Array(3);
	faceNormalOf(m, f, n);
	const doubleArea = Math.hypot(n[0], n[1], n[2]);
	const base = Math.sqrt(longestLen);
	const height = doubleArea / base;
	if (height === 0) return -1;
	return base / height > ratio ? (longest + 2) % 3 : -1;
}

function distanceSquared(m: CMeshO, a: number, b: number): number {
	const dx = m.vx(a) - m.vx(b);
	const dy = m.vy(a) - m.vy(b);
	const dz = m.vz(a) - m.vz(b);
	return dx * dx + dy * dy + dz * dz;
}

/**
 * Replaces edge `e` of face `f` with the other diagonal of the quad formed by
 * `f` and its neighbour. Returns false when the flip is not legal.
 */
function flipEdge(m: CMeshO, f: number, e: number): boolean {
	if (m.isBorderFF(f, e) || !isManifoldEdge(m, f, e)) return false;
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	if (g === f) return false;

	const a = m.fv(f, e);
	const b = m.fv(f, (e + 1) % 3);
	const cf = m.fv(f, (e + 2) % 3); // f's opposite corner
	const cg = m.fv(g, (ge + 2) % 3); // g's opposite corner
	if (cf === cg) return false;

	// The flipped edge must not already exist, or the result is non-manifold.
	for (let h = 0; h < m.faceSize; h++) {
		if (m.isFaceD(h) || h === f || h === g) continue;
		for (let k = 0; k < 3; k++) {
			const p = m.faceVert[3 * h + k];
			const q = m.faceVert[3 * h + ((k + 1) % 3)];
			if ((p === cf && q === cg) || (p === cg && q === cf)) return false;
		}
	}

	m.setFace(f, cf, a, cg);
	m.setFace(g, cg, b, cf);
	return true;
}

/**
 * Flips the shared edge of pairs of faces that fold back on each other by more
 * than `normalThresholdDeg`. Requires FF.
 *
 * A fold is two adjacent triangles facing nearly opposite directions — the
 * classic artefact of a reconstruction that punched through itself.
 */
export function removeFaceFoldByFlip(m: CMeshO, normalThresholdDeg = 175, repeat = true): number {
	const cosThreshold = Math.cos((normalThresholdDeg * Math.PI) / 180);
	const nf = new Float64Array(3);
	const ng = new Float64Array(3);
	let total = 0;
	let pass = 0;

	for (;;) {
		let done = 0;
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) {
				if (m.isBorderFF(f, e) || !isManifoldEdge(m, f, e)) continue;
				const g = m.ffp(f, e);
				if (g <= f || m.isFaceD(g)) continue;
				faceNormalOf(m, f, nf);
				faceNormalOf(m, g, ng);
				const lf = Math.hypot(nf[0], nf[1], nf[2]);
				const lg = Math.hypot(ng[0], ng[1], ng[2]);
				if (lf === 0 || lg === 0) continue;
				const cos = (nf[0] * ng[0] + nf[1] * ng[1] + nf[2] * ng[2]) / (lf * lg);
				if (cos > cosThreshold) continue;
				if (!flipEdge(m, f, e)) continue;
				done++;
				faceFace(m);
				break;
			}
		}
		total += done;
		if (done === 0 || !repeat || ++pass > 10) break;
	}
	if (total > 0) m.imark++;
	return total;
}

/**
 * The faces that pass through some other face.
 *
 * Pairs are found with a k-d tree over face centroids rather than the uniform
 * grid VCG uses — same purpose, and the tree is already here. Faces that share
 * a vertex get the offset treatment upstream applies: their opposite edges are
 * pulled halfway toward the shared corner before testing, so two triangles that
 * merely fold about a common vertex are not reported as crossing.
 */
export function selfIntersections(m: CMeshO): number[] {
	const live: number[] = [];
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) live.push(f);
	if (live.length === 0) return [];

	const centres = new Float64Array(3 * live.length);
	let maxRadius = 0;
	live.forEach((f, i) => {
		const p = corners(m, f);
		for (let k = 0; k < 3; k++) {
			centres[3 * i + k] = (p[0][k] + p[1][k] + p[2][k]) / 3;
		}
		const c = [centres[3 * i], centres[3 * i + 1], centres[3 * i + 2]];
		for (const q of p) {
			maxRadius = Math.max(maxRadius, Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]));
		}
	});
	const tree = new KdTree(centres, live.length);

	const hits = new Set<number>();
	// Two faces can only meet if their centres are within the sum of their
	// radii, and 2 * maxRadius bounds that for every pair.
	const reach = 2 * maxRadius;
	for (let i = 0; i < live.length; i++) {
		const f = live[i];
		for (const near of tree.withinRadius(
			centres[3 * i],
			centres[3 * i + 1],
			centres[3 * i + 2],
			reach,
		)) {
			const g = live[near.index];
			// Each unordered pair once.
			if (g <= f) continue;
			if (testFaceFaceIntersection(m, f, g)) {
				hits.add(f);
				hits.add(g);
			}
		}
	}
	return [...hits].sort((a, b) => a - b);
}

const EPSIL = 1e-8;

/** Whether two faces genuinely cross, discounting shared vertices. */
export function testFaceFaceIntersection(m: CMeshO, f0: number, f1: number): boolean {
	const shared: Array<[number, number]> = [];
	for (let i = 0; i < 3; i++) {
		for (let j = 0; j < 3; j++) if (m.fv(f0, i) === m.fv(f1, j)) shared.push([i, j]);
	}
	// Same three vertices: a duplicate face, which counts as intersecting.
	if (shared.length === 3) return true;
	const a = corners(m, f0);
	const b = corners(m, f1);
	if (shared.length === 0) return intersectionTriangleTriangle(a, b);
	// Two shared vertices is a shared edge — neighbours, not a defect.
	if (shared.length !== 1) return false;

	// One shared corner. Shrink each triangle's opposite edge halfway toward
	// that corner and test it against the other face: a genuine crossing still
	// pierces, while two triangles merely hinged at the corner no longer touch.
	const [i0, i1] = shared[0];
	const shP = a[i0].map((c) => c * 0.5);
	const halfway = (p: readonly number[]) => [0, 1, 2].map((k) => p[k] * 0.5 + shP[k]);
	const probe = (
		from: readonly number[],
		to: readonly number[],
		tri: readonly (readonly number[])[],
	): boolean => {
		const hit = intersectionSegmentTriangle(from, to, tri[0], tri[1], tri[2]);
		if (hit === null) return false;
		return !(hit.a + hit.b >= 1 || hit.a <= EPSIL || hit.b <= EPSIL);
	};
	if (probe(halfway(a[(i0 + 1) % 3]), halfway(a[(i0 + 2) % 3]), b)) return true;
	if (probe(halfway(b[(i1 + 1) % 3]), halfway(b[(i1 + 2) % 3]), a)) return true;
	return false;
}

function corners(m: CMeshO, f: number): number[][] {
	return [0, 1, 2].map((k) => {
		const v = m.fv(f, k);
		return [m.vx(v), m.vy(v), m.vz(v)];
	});
}

export const Clean = {
	selfIntersections,
	testFaceFaceIntersection,
	removeDuplicateVertex,
	removeDuplicateFace,
	removeDegenerateFace,
	removeFaceOutOfRangeArea,
	removeZeroAreaFace,
	countUnreferencedVertex,
	removeUnreferencedVertex,
	mergeCloseVertex,
	removeSmallConnectedComponentsSize,
	removeSmallConnectedComponentsDiameter,
	removeNonManifoldFace,
	splitNonManifoldVertex,
	splitManifoldComponents,
	removeTVertexByCollapse,
	removeTVertexByFlip,
	removeFaceFoldByFlip,
	countEdgeNum,
	isWaterTight,
	countNonManifoldEdgeFF,
	countNonManifoldVertexFF,
	countHoles,
	connectedComponents,
	countConnectedComponents,
	meshGenus,
	checkOrientation,
	isCoherentlyOrientedMesh,
	flipMesh,
	flipNormalOutside,
	signedVolume,
	orientCoherentlyMesh,
} as const;
