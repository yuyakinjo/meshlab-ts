/**
 * `Clean` — mesh analysis and repair, mirroring `vcg::tri::Clean`.
 *
 * This file currently holds the *analysis* half: counting edges, holes,
 * components and non-manifold features, deciding orientability, and making a
 * mesh coherently oriented. The removal and repair operations
 * (`RemoveDuplicateVertex`, `RemoveNonManifoldFace`, …) land alongside them
 * when `filter_clean` is implemented.
 */
import { MLInternalException } from "../../common/utilities/ml_exception.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { faceClearV } from "./update/flag.ts";
import { fillSortedEdgeVector, isManifoldEdge } from "./update/topology.ts";

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

export const Clean = {
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
	orientCoherentlyMesh,
} as const;
