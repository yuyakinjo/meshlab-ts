/**
 * Quadric edge-collapse decimation — Garland & Heckbert's QEM, driven by the
 * lazy priority queue, mirroring `vcg::tri::TriEdgeCollapseQuadric`.
 *
 * Each vertex carries a quadric: the sum, over the planes of its incident
 * faces, of the squared-distance form to that plane. Collapsing an edge merges
 * the two quadrics, and the cost of the collapse is the merged quadric's value
 * at wherever the surviving vertex ends up. Repeatedly performing the cheapest
 * collapse is what keeps the silhouette while the triangle count falls.
 *
 * The pieces that matter beyond the textbook version, all of them upstream's:
 * boundary quadrics so open edges are not eaten away, a shape penalty so the
 * result is not a mesh of slivers, a normal check so faces do not fold over,
 * and the link condition so the genus cannot change.
 */

import {
	QUADRIC_STRIDE,
	quadricAdd,
	quadricAddPlane,
	quadricEval,
	quadricMinimum,
	quadricZero,
} from "../../math/quadric.ts";
import { Allocator } from "../allocator.ts";
import type { CMeshO } from "../cmesho.ts";
import { LazyPriorityQueue, type OptimizationResult } from "../local_optimization.ts";

/**
 * `TriEdgeCollapseQuadricParameter`. Defaults are upstream's.
 */
export interface QuadricParameters {
	/** Collapses producing faces below this shape quality are penalised. */
	qualityThr: number;
	/** Below this, a collapse is rejected outright rather than penalised. */
	hardQualityThr: number;
	/** Keep boundary edges by giving them their own quadrics. */
	preserveBoundary: boolean;
	/** How heavily those boundary quadrics count. */
	boundaryQuadricWeight: number;
	/** Reject collapses that would flip a face normal. */
	normalCheck: boolean;
	/** How far a normal may turn before the collapse is rejected. */
	normalThrRad: number;
	/** Forbid collapses that would change the genus. */
	preserveTopology: boolean;
	/** Place the survivor where the quadric is minimal, not at an endpoint. */
	optimalPlacement: boolean;
	/** Add per-edge quadrics everywhere, to hold triangle shape in flat areas. */
	qualityQuadric: boolean;
	qualityQuadricWeight: number;
	/** Weight the error by per-vertex quality. */
	qualityWeight: boolean;
	qualityWeightFactor: number;
	/** Weight each face's plane by its area. */
	useArea: boolean;
	/** Normalise the error by the mesh's size, so thresholds are scale-free. */
	scaleIndependent: boolean;
}

export function defaultQuadricParameters(): QuadricParameters {
	return {
		qualityThr: 0.3,
		hardQualityThr: 0.1,
		preserveBoundary: false,
		boundaryQuadricWeight: 0.5,
		normalCheck: false,
		normalThrRad: Math.PI / 2,
		preserveTopology: false,
		optimalPlacement: true,
		qualityQuadric: false,
		qualityQuadricWeight: 0.001,
		qualityWeight: false,
		qualityWeightFactor: 100,
		useArea: true,
		scaleIndependent: true,
	};
}

interface CollapseEntry {
	readonly priority: number;
	readonly tieBreak: number;
	readonly u: number;
	readonly v: number;
	readonly x: number;
	readonly y: number;
	readonly z: number;
	/** Versions of the two endpoints when the cost was computed. */
	readonly vu: number;
	readonly vv: number;
}

export interface DecimateOptions {
	readonly targetFaceNum: number;
	readonly params?: Partial<QuadricParameters>;
	/** Only collapse edges whose incident faces are all selected. */
	readonly selected?: boolean;
	/** Called with progress in 0..100; return false to stop. */
	readonly callback?: (pos: number, message: string) => boolean;
}

export interface DecimateResult extends OptimizationResult {
	/** Faces before and after. */
	initialFaces: number;
	finalFaces: number;
}

/**
 * Decimates `m` to about `targetFaceNum` faces.
 *
 * The mesh is left with deleted slots; the caller compacts. Requires nothing
 * of the input beyond being a triangle mesh, but a non-manifold one will
 * simply have fewer legal collapses.
 */
export function quadricSimplification(m: CMeshO, options: DecimateOptions): DecimateResult {
	const p: QuadricParameters = { ...defaultQuadricParameters(), ...options.params };
	const initialFaces = m.fn;
	const callback = options.callback;

	if (m.fn === 0 || m.fn <= options.targetFaceNum) {
		return {
			performed: 0,
			discarded: 0,
			reason: "goalReached",
			initialFaces,
			finalFaces: m.fn,
		};
	}

	// ---- incidence -------------------------------------------------------
	// A per-vertex set of incident faces, maintained through the run. VCGLib
	// uses its VF adjacency here; keeping a set is far easier to hold correct
	// across collapses and costs little at these sizes.
	const vertFaces: Array<Set<number>> = Array.from({ length: m.vertSize }, () => new Set());
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) vertFaces[m.faceVert[3 * f + k]].add(f);
	}

	const version = new Int32Array(m.vertSize);
	const quadrics = new Float64Array(m.vertSize * QUADRIC_STRIDE);
	initQuadrics(m, quadrics, p);

	// Errors scale with the square of the mesh's size, so a fixed threshold
	// would mean something different on a millimetre part and a metre one.
	const diag = m.bbox.isEmpty ? 1 : m.bbox.diagonal || 1;
	const scaleFactor = p.scaleIndependent ? 1 / (diag * diag) : 1;

	const queue = new LazyPriorityQueue<CollapseEntry>();
	const scratch = new Float64Array(3);

	const enqueue = (u: number, v: number): void => {
		if (u === v) return;
		const a = Math.min(u, v);
		const b = Math.max(u, v);
		if (m.isVertD(a) || m.isVertD(b)) return;
		const cost = collapseCost(m, quadrics, a, b, p, scaleFactor, scratch);
		if (cost === null) return;
		queue.push({
			priority: cost.priority,
			// Deterministic second key, so equal costs always break the same
			// way and a run is reproducible.
			tieBreak: a * (m.vertSize + 1) + b,
			u: a,
			v: b,
			x: cost.x,
			y: cost.y,
			z: cost.z,
			vu: version[a],
			vv: version[b],
		});
	};

	// Seed with every edge.
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = m.faceVert[3 * f + e];
			const b = m.faceVert[3 * f + ((e + 1) % 3)];
			if (a < b) enqueue(a, b);
		}
	}

	// ---- the collapse loop -------------------------------------------------
	let performed = 0;
	let discarded = 0;
	let reason: OptimizationResult["reason"] = "exhausted";
	const startFaces = m.fn;

	while (m.fn > options.targetFaceNum) {
		const entry = queue.pop();
		if (entry === undefined) break;

		// Lazy invalidation: the cost was computed against a mesh that has
		// since changed under it.
		if (entry.vu !== version[entry.u] || entry.vv !== version[entry.v]) {
			discarded++;
			continue;
		}
		if (m.isVertD(entry.u) || m.isVertD(entry.v)) {
			discarded++;
			continue;
		}
		if (
			!isFeasible(
				m,
				vertFaces,
				entry.u,
				entry.v,
				entry.x,
				entry.y,
				entry.z,
				p,
				options.selected ?? false,
			)
		) {
			discarded++;
			continue;
		}

		const touched = collapse(m, vertFaces, entry.u, entry.v, entry.x, entry.y, entry.z);
		quadricAdd(quadrics, entry.u, entry.v);
		version[entry.u]++;
		version[entry.v]++;
		for (const w of touched) version[w]++;
		performed++;

		// Re-price every edge now incident on the survivor.
		for (const f of vertFaces[entry.u]) {
			if (m.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const a = m.faceVert[3 * f + k];
				const b = m.faceVert[3 * f + ((k + 1) % 3)];
				enqueue(a, b);
			}
		}

		if (callback !== undefined && performed % 128 === 0) {
			const done = startFaces - m.fn;
			const total = Math.max(1, startFaces - options.targetFaceNum);
			if (!callback(Math.min(100, (100 * done) / total), "Simplifying")) {
				reason = "canceled";
				break;
			}
		}
	}

	if (m.fn <= options.targetFaceNum) reason = "goalReached";
	m.imark++;

	return { performed, discarded, reason, initialFaces, finalFaces: m.fn };
}

/**
 * Builds the per-vertex quadrics from the incident face planes.
 *
 * Boundary edges additionally contribute a plane perpendicular to the surface
 * along the edge. Without it the error of pulling a boundary inward is zero —
 * the boundary lies in its own faces' planes — and an open mesh gets eaten
 * from the edges in.
 */
function initQuadrics(m: CMeshO, quadrics: Float64Array, p: QuadricParameters): void {
	for (let v = 0; v < m.vertSize; v++) quadricZero(quadrics, v);

	// How many faces use each edge, to find the boundary without adjacency.
	const edgeUses = new Map<string, number>();
	const key = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const k = key(m.faceVert[3 * f + e], m.faceVert[3 * f + ((e + 1) % 3)]);
			edgeUses.set(k, (edgeUses.get(k) ?? 0) + 1);
		}
	}

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.faceVert[3 * f];
		const b = m.faceVert[3 * f + 1];
		const c = m.faceVert[3 * f + 2];

		const ux = m.vx(b) - m.vx(a);
		const uy = m.vy(b) - m.vy(a);
		const uz = m.vz(b) - m.vz(a);
		const vx = m.vx(c) - m.vx(a);
		const vy = m.vy(c) - m.vy(a);
		const vz = m.vz(c) - m.vz(a);
		let nx = uy * vz - uz * vy;
		let ny = uz * vx - ux * vz;
		let nz = ux * vy - uy * vx;
		const len = Math.hypot(nx, ny, nz);
		if (len === 0) continue; // a degenerate face defines no plane
		const area = len / 2;
		nx /= len;
		ny /= len;
		nz /= len;
		const d = -(nx * m.vx(a) + ny * m.vy(a) + nz * m.vz(a));
		const w = p.useArea ? area : 1;

		for (const v of [a, b, c]) quadricAddPlane(quadrics, v, nx, ny, nz, d, w);

		// Constraint planes along the edges that need holding.
		for (let e = 0; e < 3; e++) {
			const p0 = m.faceVert[3 * f + e];
			const p1 = m.faceVert[3 * f + ((e + 1) % 3)];
			const isBorder = (edgeUses.get(key(p0, p1)) ?? 0) === 1;
			const wantBoundary = p.preserveBoundary && isBorder;
			if (!wantBoundary && !p.qualityQuadric) continue;

			// A plane through the edge, perpendicular to the face.
			let ex = m.vx(p1) - m.vx(p0);
			let ey = m.vy(p1) - m.vy(p0);
			let ez = m.vz(p1) - m.vz(p0);
			const elen = Math.hypot(ex, ey, ez);
			if (elen === 0) continue;
			ex /= elen;
			ey /= elen;
			ez /= elen;
			let px = ny * ez - nz * ey;
			let py = nz * ex - nx * ez;
			let pz = nx * ey - ny * ex;
			const plen = Math.hypot(px, py, pz);
			if (plen === 0) continue;
			px /= plen;
			py /= plen;
			pz /= plen;
			const pd = -(px * m.vx(p0) + py * m.vy(p0) + pz * m.vz(p0));

			const weight =
				(wantBoundary ? p.boundaryQuadricWeight : p.qualityQuadricWeight) * (p.useArea ? area : 1);
			quadricAddPlane(quadrics, p0, px, py, pz, pd, weight);
			quadricAddPlane(quadrics, p1, px, py, pz, pd, weight);
		}
	}
}

/** Where the survivor goes, and what that costs. */
function collapseCost(
	m: CMeshO,
	quadrics: Float64Array,
	u: number,
	v: number,
	p: QuadricParameters,
	scaleFactor: number,
	scratch: Float64Array,
): { priority: number; x: number; y: number; z: number } | null {
	// The merged quadric, built in a scratch slot at the end of the array
	// rather than allocated per candidate — this runs millions of times.
	const merged = mergedQuadric(quadrics, u, v);

	let x: number;
	let y: number;
	let z: number;
	if (p.optimalPlacement && quadricMinimum(merged, 0, scratch)) {
		x = scratch[0];
		y = scratch[1];
		z = scratch[2];
	} else {
		// Two quite different situations land here, and they get different
		// candidate sets.
		//
		// With optimalPlacement off, the parameter's promise is that "the
		// final mesh is composed by a subset of the original vertices", so
		// only the two endpoints are eligible — including the midpoint would
		// introduce a point that was never in the input and quietly break
		// that contract.
		//
		// With it on but the quadric singular (a flat patch, or a straight
		// edge, where the minimum is a whole line or plane rather than a
		// point) the midpoint is a reasonable extra candidate.
		const candidates: Array<[number, number, number]> = [
			[m.vx(u), m.vy(u), m.vz(u)],
			[m.vx(v), m.vy(v), m.vz(v)],
		];
		if (p.optimalPlacement) {
			candidates.push([(m.vx(u) + m.vx(v)) / 2, (m.vy(u) + m.vy(v)) / 2, (m.vz(u) + m.vz(v)) / 2]);
		}
		let best = Number.POSITIVE_INFINITY;
		let pick = candidates[0];
		for (const c of candidates) {
			const e = quadricEval(merged, 0, c[0], c[1], c[2]);
			if (e < best) {
				best = e;
				pick = c;
			}
		}
		[x, y, z] = pick;
	}

	if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;

	let error = quadricEval(merged, 0, x, y, z) * scaleFactor;
	if (p.qualityWeight) {
		// A high-quality vertex is one the caller wants kept, so its error is
		// amplified rather than reduced.
		const q = Math.max(m.vertQuality[u], m.vertQuality[v]);
		error *= 1 + q * p.qualityWeightFactor;
	}
	return { priority: error, x, y, z };
}

// One scratch quadric, reused. mergedQuadric writes into slot 0 of this.
const mergeScratch = new Float64Array(QUADRIC_STRIDE);

function mergedQuadric(quadrics: Float64Array, u: number, v: number): Float64Array {
	const a = u * QUADRIC_STRIDE;
	const b = v * QUADRIC_STRIDE;
	for (let k = 0; k < QUADRIC_STRIDE; k++) mergeScratch[k] = quadrics[a + k] + quadrics[b + k];
	return mergeScratch;
}

/** 2·area / longest-edge² for a triangle given as nine coordinates. */
function triQuality(
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
 * Whether collapsing `(u, v)` onto `(x, y, z)` is allowed.
 *
 * Four separate reasons to refuse, each of which corresponds to a visible
 * defect if skipped: a topology change, a face folding over, a sliver, or
 * touching geometry the caller excluded.
 */
function isFeasible(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	u: number,
	v: number,
	x: number,
	y: number,
	z: number,
	p: QuadricParameters,
	selectedOnly: boolean,
): boolean {
	const shared: number[] = [];
	for (const f of vertFaces[u]) if (vertFaces[v].has(f)) shared.push(f);
	// An edge with no face, or more than two, is not a manifold edge; leave it.
	if (shared.length === 0 || shared.length > 2) return false;

	if (selectedOnly) {
		for (const f of vertFaces[u]) if (!m.isFaceD(f) && !m.isFaceS(f)) return false;
		for (const f of vertFaces[v]) if (!m.isFaceD(f) && !m.isFaceS(f)) return false;
	}

	if (p.preserveTopology && !linkCondition(m, vertFaces, u, v, shared)) return false;

	// Shape and orientation of every face that will survive the collapse.
	for (const [keep, gone] of [
		[u, v],
		[v, u],
	] as const) {
		for (const f of vertFaces[keep]) {
			if (m.isFaceD(f) || shared.includes(f)) continue;
			const verts: Array<[number, number, number]> = [];
			for (let k = 0; k < 3; k++) {
				const w = m.faceVert[3 * f + k];
				// Both endpoints end up at the new position.
				if (w === u || w === v) verts.push([x, y, z]);
				else verts.push([m.vx(w), m.vy(w), m.vz(w)]);
			}
			void gone;

			const q = triQuality(
				verts[0][0],
				verts[0][1],
				verts[0][2],
				verts[1][0],
				verts[1][1],
				verts[1][2],
				verts[2][0],
				verts[2][1],
				verts[2][2],
			);
			if (q < p.hardQualityThr) return false;

			if (p.normalCheck) {
				const before = faceNormalUnit(m, f);
				const after = triNormalUnit(verts);
				if (before !== null && after !== null) {
					const dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2];
					if (dot < Math.cos(p.normalThrRad)) return false;
				}
			}
		}
	}
	return true;
}

/**
 * The link condition: collapsing an edge is safe exactly when the only
 * vertices adjacent to both endpoints are the ones opposite the edge.
 *
 * A common neighbour that is *not* part of a shared face means the collapse
 * would fuse two pieces of surface that only met at that vertex — closing a
 * hole or pinching a handle, either of which changes the genus.
 */
function linkCondition(
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

function faceNormalUnit(m: CMeshO, f: number): [number, number, number] | null {
	const a = m.faceVert[3 * f];
	const b = m.faceVert[3 * f + 1];
	const c = m.faceVert[3 * f + 2];
	return triNormalUnit([
		[m.vx(a), m.vy(a), m.vz(a)],
		[m.vx(b), m.vy(b), m.vz(b)],
		[m.vx(c), m.vy(c), m.vz(c)],
	]);
}

function triNormalUnit(
	v: ReadonlyArray<readonly [number, number, number]>,
): [number, number, number] | null {
	const ux = v[1][0] - v[0][0];
	const uy = v[1][1] - v[0][1];
	const uz = v[1][2] - v[0][2];
	const wx = v[2][0] - v[0][0];
	const wy = v[2][1] - v[0][1];
	const wz = v[2][2] - v[0][2];
	const nx = uy * wz - uz * wy;
	const ny = uz * wx - ux * wz;
	const nz = ux * wy - uy * wx;
	const len = Math.hypot(nx, ny, nz);
	if (len === 0) return null;
	return [nx / len, ny / len, nz / len];
}

/**
 * Performs the collapse: `v` merges into `u`, which moves to `(x, y, z)`.
 *
 * Returns the other vertices whose neighbourhood changed, so the caller can
 * invalidate their queued costs.
 */
function collapse(
	m: CMeshO,
	vertFaces: Array<Set<number>>,
	u: number,
	v: number,
	x: number,
	y: number,
	z: number,
): number[] {
	const shared: number[] = [];
	for (const f of vertFaces[u]) if (vertFaces[v].has(f)) shared.push(f);

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
