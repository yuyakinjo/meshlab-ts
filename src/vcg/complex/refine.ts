/**
 * `vcg/complex/algorithms/refine.h` — edge-split refinement, and the
 * subdivision schemes built on it.
 *
 * The framework is two functions handed in by the caller: a predicate saying
 * which edges to split, and an interpolator saying where the new vertex goes.
 * Everything else — deciding how a triangle with 1, 2 or 3 split edges falls
 * apart, keeping the two sides of a split edge agreeing on the vertex they
 * share — is the same regardless of scheme, and lives here.
 */

import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag, VertexFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { UpdateFlags } from "./update/flag.ts";
import { UpdateNormal } from "./update/normal.ts";
import { UpdateTopology } from "./update/topology.ts";

/** Where a new vertex lands, given the edge it splits. */
export type Interpolator = (m: CMeshO, pos: Pos) => [number, number, number];

/** Whether the edge under `pos` should be split. */
export type EdgePredicate = (m: CMeshO, pos: Pos) => boolean;

/**
 * How a triangle falls apart, indexed by which of its edges were split.
 *
 * The index is a 3-bit mask: bit 0 for the 0-1 edge, bit 1 for 1-2, bit 2 for
 * 2-0. Vertices 0..2 are the triangle's own; 3, 4 and 5 are the midpoints of
 * those three edges in that order.
 *
 * When exactly two edges are split the result is a triangle and a trapezoid,
 * and the trapezoid can be cut along either diagonal. `swap` names the two
 * candidate diagonals: if the first is shorter than the second, the last two
 * triangles trade their second vertices, which flips the cut. `edge` records
 * which edge of the original each new edge came from, so border flags survive;
 * 3 means the edge is new and interior.
 */
interface SplitRule {
	readonly count: number;
	readonly tri: ReadonlyArray<readonly [number, number, number]>;
	readonly swap: readonly [readonly [number, number], readonly [number, number]];
	readonly edge: ReadonlyArray<readonly [number, number, number]>;
}

// biome-ignore format: one row per split mask, as the C++ table is laid out
const SPLIT_TAB: readonly SplitRule[] = [
	/* --- */ { count: 1, tri: [[0, 1, 2]],                              swap: [[0, 0], [0, 0]], edge: [[0, 1, 2]] },
	/* 01  */ { count: 2, tri: [[0, 3, 2], [3, 1, 2]],                   swap: [[0, 0], [0, 0]], edge: [[0, 3, 2], [0, 1, 3]] },
	/* 12  */ { count: 2, tri: [[0, 1, 4], [0, 4, 2]],                   swap: [[0, 0], [0, 0]], edge: [[0, 1, 3], [3, 1, 2]] },
	/* 01+12 */ { count: 3, tri: [[3, 1, 4], [0, 3, 2], [4, 2, 3]],      swap: [[0, 4], [3, 2]], edge: [[0, 1, 3], [0, 3, 2], [1, 3, 3]] },
	/* 20  */ { count: 2, tri: [[0, 1, 5], [5, 1, 2]],                   swap: [[0, 0], [0, 0]], edge: [[0, 3, 2], [3, 1, 2]] },
	/* 01+20 */ { count: 3, tri: [[0, 3, 5], [3, 1, 5], [2, 5, 1]],      swap: [[3, 2], [5, 1]], edge: [[0, 3, 2], [0, 3, 3], [2, 3, 1]] },
	/* 12+20 */ { count: 3, tri: [[2, 5, 4], [0, 1, 5], [4, 5, 1]],      swap: [[0, 4], [5, 1]], edge: [[2, 3, 1], [0, 3, 2], [3, 3, 1]] },
	/* all */ { count: 4, tri: [[3, 4, 5], [0, 3, 5], [3, 1, 4], [5, 4, 2]], swap: [[0, 0], [0, 0]], edge: [[3, 3, 3], [0, 3, 2], [0, 1, 3], [3, 1, 2]] },
];

const squaredDistance = (m: CMeshO, a: number, b: number): number =>
	(m.vx(a) - m.vx(b)) ** 2 + (m.vy(a) - m.vy(b)) ** 2 + (m.vz(a) - m.vz(b)) ** 2;

/** Splits every edge longer than `threshold`. VCGLib's `EdgeLen`. */
export function longerThan(threshold: number): EdgePredicate {
	const t2 = threshold * threshold;
	return (m, pos) => squaredDistance(m, pos.v, pos.vFlip) > t2;
}

/** Splits every edge. */
export const everyEdge: EdgePredicate = () => true;

/** The plain midpoint. */
export const midPoint: Interpolator = (m, pos) => {
	const a = pos.v;
	const b = pos.vFlip;
	return [(m.vx(a) + m.vx(b)) / 2, (m.vy(a) + m.vy(b)) / 2, (m.vz(a) + m.vz(b)) / 2];
};

export interface RefineOptions {
	/** Split only inside the current selection, as MeshLab's filters offer. */
	readonly selectedOnly?: boolean;
}

/**
 * Splits the edges the predicate names and retriangulates the faces they touch.
 *
 * Returns false when the predicate matched nothing, which is how the callers
 * that iterate to convergence know to stop. Requires FF adjacency, and leaves
 * it stale — the caller rebuilds if it needs it.
 */
export function refineE(
	m: CMeshO,
	interpolate: Interpolator,
	predicate: EdgePredicate,
	options: RefineOptions = {},
): boolean {
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	UpdateFlags.faceBorderFromFF(m);
	const selectedOnly = options.selectedOnly === true;

	const originalFaceSize = m.faceSize;
	// Per (face, edge): whether it splits, and which new vertex it splits at.
	const splits = new Uint8Array(originalFaceSize * 3);
	const splitVert = new Int32Array(originalFaceSize * 3).fill(-1);

	let newVerts = 0;
	let newFaces = 0;
	for (let f = 0; f < originalFaceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (selectedOnly && !m.isFaceS(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (splits[3 * f + e]) continue;
			const pos = Pos.onEdge(m, f, e);
			// A selected-only pass must not split an edge onto an unselected
			// neighbour, or that neighbour is left with a T-vertex.
			if (selectedOnly && !pos.isBorder() && !m.isFaceS(m.ffp(f, e))) continue;
			if (!pos.isManifold()) continue;
			if (!predicate(m, pos)) continue;

			splits[3 * f + e] = 1;
			newVerts++;
			newFaces++;
			if (!pos.isBorder()) {
				splits[3 * m.ffp(f, e) + m.ffi(f, e)] = 1;
				newFaces++;
			}
		}
	}
	if (newVerts === 0) return false;

	// Second pass: give each split edge its vertex, and make the far side of a
	// shared edge point at the same one.
	const firstVert = Allocator.addVertices(m, newVerts);
	let nextVert = firstVert;
	for (let f = 0; f < originalFaceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!splits[3 * f + e] || splitVert[3 * f + e] >= 0) continue;
			const pos = Pos.onEdge(m, f, e);
			const p = interpolate(m, pos);
			const v = nextVert++;
			m.setVert(v, p[0], p[1], p[2]);
			carryVertexData(m, v, pos.v, pos.vFlip);
			splitVert[3 * f + e] = v;
			if (!pos.isBorder()) splitVert[3 * m.ffp(f, e) + m.ffi(f, e)] = v;
		}
	}

	const firstFace = Allocator.addFaces(m, newFaces);
	let nextFace = firstFace;
	for (let f = 0; f < originalFaceSize; f++) {
		if (m.isFaceD(f)) continue;
		const vv = [
			m.fv(f, 0),
			m.fv(f, 1),
			m.fv(f, 2),
			splitVert[3 * f],
			splitVert[3 * f + 1],
			splitVert[3 * f + 2],
		];
		const mask = (vv[3] >= 0 ? 1 : 0) | (vv[4] >= 0 ? 2 : 0) | (vv[5] >= 0 ? 4 : 0);
		if (mask === 0) continue;
		const rule = SPLIT_TAB[mask];

		const originalFlags = m.faceFlags[f];
		const target: number[] = [f];
		for (let i = 1; i < rule.count; i++) {
			const nf = nextFace++;
			m.faceFlags[nf] = originalFlags & ~FaceFlag.BORDER012;
			// The pieces inherit whatever the face they came from carried.
			if (m.faceQuality !== null) m.faceQuality[nf] = m.faceQuality[f];
			if (m.faceColor !== null) m.faceColor[nf] = m.faceColor[f];
			target.push(nf);
		}

		for (let i = 0; i < rule.count; i++) {
			const t = rule.tri[i];
			m.setFace(target[i], vv[t[0]], vv[t[1]], vv[t[2]]);
			// Carry the original border bits onto the pieces that inherited an
			// original edge; new interior edges (coded 3) are never borders.
			let flags = m.faceFlags[target[i]] & ~FaceFlag.BORDER012;
			for (let j = 0; j < 3; j++) {
				const from = rule.edge[i][j];
				if (from !== 3 && (originalFlags & (FaceFlag.BORDER0 << from)) !== 0) {
					flags |= FaceFlag.BORDER0 << j;
				}
			}
			m.faceFlags[target[i]] = flags;
		}

		// With two edges split, cut the trapezoid along its shorter diagonal.
		if (
			rule.count === 3 &&
			squaredDistance(m, vv[rule.swap[0][0]], vv[rule.swap[0][1]]) <
				squaredDistance(m, vv[rule.swap[1][0]], vv[rule.swap[1][1]])
		) {
			const a = target[1];
			const b = target[2];
			// Each takes the other's *first* vertex as its second. Both reads
			// have to happen before either write, or the second one picks up
			// the value the first just replaced and the triangle collapses.
			const av0 = m.fv(a, 0);
			const bv0 = m.fv(b, 0);
			m.setFace(b, bv0, av0, m.fv(b, 2));
			m.setFace(a, av0, bv0, m.fv(a, 2));
			// The swap moved edge 0 of each into slot 1 of the other.
			const aBorder0 = (m.faceFlags[a] & FaceFlag.BORDER0) !== 0;
			const bBorder0 = (m.faceFlags[b] & FaceFlag.BORDER0) !== 0;
			m.faceFlags[a] &= ~(FaceFlag.BORDER0 | FaceFlag.BORDER1);
			m.faceFlags[b] &= ~(FaceFlag.BORDER0 | FaceFlag.BORDER1);
			if (aBorder0) m.faceFlags[b] |= FaceFlag.BORDER1;
			if (bBorder0) m.faceFlags[a] |= FaceFlag.BORDER1;
		}
	}

	// Every face index moved or gained neighbours, so nothing built on the old
	// adjacency is true any more.
	UpdateTopology.clearFaceFace(m);
	return true;
}

/** Averages the two endpoints' per-vertex channels onto a new midpoint vertex. */
function carryVertexData(m: CMeshO, v: number, a: number, b: number): void {
	m.vertQuality[v] = (m.vertQuality[a] + m.vertQuality[b]) / 2;
	m.vertColor[v] = lerpColor(m.vertColor[a], m.vertColor[b]);
	for (let k = 0; k < 3; k++) {
		m.vertNormal[3 * v + k] = (m.vertNormal[3 * a + k] + m.vertNormal[3 * b + k]) / 2;
	}
}

/** Halfway between two packed RGBA colours, channel by channel. */
function lerpColor(a: number, b: number): number {
	let out = 0;
	for (let shift = 0; shift < 32; shift += 8) {
		const mid = (((a >>> shift) & 0xff) + ((b >>> shift) & 0xff)) >> 1;
		out |= mid << shift;
	}
	return out >>> 0;
}

/**
 * The modified butterfly rule: an interpolating scheme, so original vertices
 * stay exactly where they are and the limit surface passes through them.
 *
 * Reads the two triangles across the edge and the four beyond them; on a
 * border it falls back to the four-point rule along the boundary curve.
 */
export const midPointButterfly: Interpolator = (m, ep) => {
	const he = new Pos(m, ep.f, ep.z, m.fv(ep.f, ep.z));
	const l = he.v;
	he.flipV();
	const r = he.v;

	const p = (v: number): [number, number, number] => [m.vx(v), m.vy(v), m.vz(v)];
	const combine = (terms: Array<[number, readonly number[]]>): [number, number, number] => {
		const out: [number, number, number] = [0, 0, 0];
		for (const [w, q] of terms) for (let k = 0; k < 3; k++) out[k] += w * q[k];
		return out;
	};

	if (he.isBorder()) {
		he.nextB();
		const r0 = he.v;
		he.flipV();
		he.nextB();
		he.nextB();
		const l0 = he.v;
		return combine([
			[9 / 16, p(l)],
			[9 / 16, p(r)],
			[-1 / 16, p(l0)],
			[-1 / 16, p(r0)],
		]);
	}

	// The eight-point stencil: up and down across the edge, then the four
	// shoulders beyond them.
	he.flipE();
	he.flipV();
	const u = he.v;
	he.flipF();
	he.flipE();
	he.flipV();
	const ur = he.v;
	he.flipV();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipV();
	const ul = he.v;
	he.flipV();
	he.flipE();
	he.flipF();
	he.flipV();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipV();
	const d = he.v;
	he.flipF();
	he.flipE();
	he.flipV();
	const dl = he.v;
	he.flipV();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipV();
	const dr = he.v;

	return combine([
		[1 / 2, p(l)],
		[1 / 2, p(r)],
		[1 / 8, p(u)],
		[1 / 8, p(d)],
		[-1 / 16, p(ul)],
		[-1 / 16, p(ur)],
		[-1 / 16, p(dl)],
		[-1 / 16, p(dr)],
	]);
};

/** Loop's odd (new-vertex) rule: 3/8 on the edge, 1/8 on the two opposite corners. */
export const oddPointLoop: Interpolator = (m, ep) => {
	const he = new Pos(m, ep.f, ep.z, m.fv(ep.f, ep.z));
	const l = he.v;
	he.flipV();
	const r = he.v;

	const at = (v: number, w: number, into: [number, number, number]) => {
		into[0] += w * m.vx(v);
		into[1] += w * m.vy(v);
		into[2] += w * m.vz(v);
	};
	const out: [number, number, number] = [0, 0, 0];

	if (he.isBorder()) {
		at(l, 0.5, out);
		at(r, 0.5, out);
		return out;
	}
	he.flipE();
	he.flipV();
	const u = he.v;
	he.flipV();
	he.flipE();
	he.flipF();
	he.flipE();
	he.flipV();
	const d = he.v;

	at(l, 3 / 8, out);
	at(r, 3 / 8, out);
	at(u, 1 / 8, out);
	at(d, 1 / 8, out);
	return out;
};

/** Loop's beta: how much of each neighbour an interior vertex of valence `k` takes. */
export function loopBeta(k: number): number {
	if (k <= 3) return 3 / 16;
	return (5 / 8 - (3 / 8 + Math.cos((2 * Math.PI) / k) / 4) ** 2) / k;
}

/**
 * Loop subdivision: split every edge by the odd rule, then move every original
 * vertex by the even rule.
 *
 * Approximating rather than interpolating — the original vertices move, which
 * is exactly why the surface comes out smooth instead of merely denser. The
 * even positions are all computed from the mesh as it stands before any split,
 * then written afterwards, so the two rules never see each other's output.
 */
export function refineLoop(
	m: CMeshO,
	predicate: EdgePredicate = everyEdge,
	options: RefineOptions = {},
): boolean {
	if (m.ffFace === null) UpdateTopology.faceFace(m);

	const originalVertSize = m.vertSize;
	const moved = new Uint8Array(originalVertSize);
	const even = new Float64Array(originalVertSize * 3);

	const seen = new Uint8Array(originalVertSize);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (options.selectedOnly === true && !m.isFaceS(f)) continue;
		for (let i = 0; i < 3; i++) {
			const v = m.fv(f, i);
			if (seen[v] || m.isVertD(v)) continue;
			seen[v] = 1;
			const p = evenPointLoop(m, f, i);
			even[3 * v] = p[0];
			even[3 * v + 1] = p[1];
			even[3 * v + 2] = p[2];
			moved[v] = 1;
		}
	}

	if (!refineE(m, oddPointLoop, predicate, options)) return false;
	for (let v = 0; v < originalVertSize; v++) {
		if (moved[v] && !m.isVertD(v)) m.setVert(v, even[3 * v], even[3 * v + 1], even[3 * v + 2]);
	}
	return true;
}

/**
 * Where an original vertex moves under Loop.
 *
 * Interior: it keeps `1 - k*beta` of itself and spreads `beta` over each of
 * its `k` neighbours. On a boundary the interior is ignored entirely and the
 * vertex follows the boundary curve alone — 3/4 itself, 1/8 each way — which
 * is what stops an open mesh from pulling its own edge inwards.
 */
function evenPointLoop(m: CMeshO, face: number, corner: number): [number, number, number] {
	const start = new Pos(m, face, corner, m.fv(face, corner));
	const centre = start.v;
	const he = start.clone();

	let k = 0;
	do {
		he.nextE();
		k++;
	} while (!he.isBorder() && !he.equals(start) && k < m.faceSize * 3 + 3);

	const out: [number, number, number] = [0, 0, 0];
	const at = (v: number, w: number) => {
		out[0] += w * m.vx(v);
		out[1] += w * m.vy(v);
		out[2] += w * m.vz(v);
	};

	if (he.isBorder()) {
		// Walk to the boundary neighbour on each side.
		const right = he.clone();
		right.flipV();
		const r = right.v;
		const other = he.clone();
		other.nextB();
		const l = other.v;
		at(centre, 3 / 4);
		at(l, 1 / 8);
		at(r, 1 / 8);
		return out;
	}

	const beta = loopBeta(k);
	at(centre, 1 - k * beta);
	const walk = start.clone();
	do {
		at(walk.vFlip, beta);
		walk.nextE();
	} while (!walk.equals(start));
	return out;
}

/** Marks every vertex of a selected face, for filters that report what moved. */
export function selectVerticesFromFaces(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.SELECTED;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || !m.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) m.vertFlags[m.fv(f, k)] |= VertexFlag.SELECTED;
	}
}

export const Refine = {
	refineE,
	refineLoop,
	refineLS3Loop,
	midPoint,
	midPointButterfly,
	oddPointLoop,
	loopBeta,
	longerThan,
	everyEdge,
} as const;

/**
 * An algebraic sphere fitted to weighted points-with-normals, and the
 * projection onto it.
 *
 * The LS3 scheme of Boyé, Guennebaud and Schlick: run Loop's own weights, but
 * instead of averaging the *positions* with them, use them to fit a sphere to
 * the neighbourhood's positions *and normals* and take the point on that
 * sphere. Loop converges to a surface that ignores the normals it was given;
 * LS3 uses them, so a coarse mesh with good normals subdivides into something
 * much closer to the surface those normals describe.
 *
 * The sphere is `u₄|p|² + u⃗·p + u₀ = 0`, which degenerates gracefully: a zero
 * quadratic term is a plane, and the code takes that branch rather than
 * dividing by it.
 */
export class AlgebraicSphere {
	private sumP = [0, 0, 0];
	private sumN = [0, 0, 0];
	private sumDotPN = 0;
	private sumDotPP = 0;
	private sumW = 0;

	add(p: readonly number[], n: readonly number[], w: number): void {
		for (let k = 0; k < 3; k++) {
			this.sumP[k] += p[k] * w;
			this.sumN[k] += n[k] * w;
		}
		this.sumDotPN += w * (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
		this.sumDotPP += w * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
		this.sumW += w;
	}

	/** The projection of the weighted mean onto the fitted sphere. */
	project(beta = 1): [number, number, number] {
		const invW = this.sumW === 0 ? 0 : 1 / this.sumW;
		const origin = this.sumP.map((x) => x * invW) as [number, number, number];
		if (invW === 0) return origin;

		const dotPN =
			this.sumP[0] * this.sumN[0] + this.sumP[1] * this.sumN[1] + this.sumP[2] * this.sumN[2];
		const spread =
			this.sumDotPP - invW * (this.sumP[0] ** 2 + this.sumP[1] ** 2 + this.sumP[2] ** 2);
		// Every neighbour at the same place: no sphere to speak of.
		if (Math.abs(spread) < 1e-300) return origin;
		const quad = (beta * 0.5 * (this.sumDotPN - invW * dotPN)) / spread;
		const linear = [0, 1, 2].map((k) => (this.sumN[k] - this.sumP[k] * 2 * quad) * invW);
		const constant =
			-invW *
			(linear[0] * this.sumP[0] +
				linear[1] * this.sumP[1] +
				linear[2] * this.sumP[2] +
				this.sumDotPP * quad);

		if (Math.abs(quad) > 1e-7) {
			const b = 1 / quad;
			const centre = linear.map((x) => x * -0.5 * b);
			const inside = centre[0] ** 2 + centre[1] ** 2 + centre[2] ** 2 - b * constant;
			if (!(inside > 0)) return origin;
			const radius = Math.sqrt(inside);
			const dir = [0, 1, 2].map((k) => origin[k] - centre[k]);
			const len = Math.hypot(dir[0], dir[1], dir[2]);
			if (len === 0) return origin;
			return [0, 1, 2].map((k) => centre[k] + (dir[k] / len) * radius) as [number, number, number];
		}

		// The quadratic term vanished: the fit is a plane, so project onto it.
		const len = Math.hypot(linear[0], linear[1], linear[2]);
		if (len === 0) return origin;
		const unit = linear.map((x) => x / len);
		const offset = constant / len;
		const d = unit[0] * origin[0] + unit[1] * origin[1] + unit[2] * origin[2] + offset;
		return [0, 1, 2].map((k) => origin[k] - unit[k] * d) as [number, number, number];
	}
}

/**
 * Loop subdivision with the new points projected onto a locally fitted
 * algebraic sphere rather than placed by the positional average.
 *
 * Same connectivity as {@link refineLoop} and the same weights; what differs is
 * only where each new vertex lands. Needs per-vertex normals, and reads them
 * as given rather than recomputing — using the mesh's own normals is the whole
 * point of the scheme.
 */
export function refineLS3Loop(
	m: CMeshO,
	predicate: EdgePredicate = everyEdge,
	options: RefineOptions = {},
): boolean {
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);

	const originalVertSize = m.vertSize;
	const moved = new Uint8Array(originalVertSize);
	const even = new Float64Array(originalVertSize * 3);
	const seen = new Uint8Array(originalVertSize);

	const normalOf = (v: number): number[] => [
		m.vertNormal[3 * v],
		m.vertNormal[3 * v + 1],
		m.vertNormal[3 * v + 2],
	];
	const pointOf = (v: number): number[] => [m.vx(v), m.vy(v), m.vz(v)];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (options.selectedOnly === true && !m.isFaceS(f)) continue;
		for (let i = 0; i < 3; i++) {
			const v = m.fv(f, i);
			if (seen[v] || m.isVertD(v)) continue;
			seen[v] = 1;
			const ring = oneRing(m, f, i);
			if (ring === null) continue;
			// Loop's even rule as weights: beta on each neighbour, the rest on
			// the vertex itself.
			const beta = loopBeta(ring.length);
			const sphere = new AlgebraicSphere();
			sphere.add(pointOf(v), normalOf(v), 1 - ring.length * beta);
			for (const w of ring) sphere.add(pointOf(w), normalOf(w), beta);
			const p = sphere.project();
			even[3 * v] = p[0];
			even[3 * v + 1] = p[1];
			even[3 * v + 2] = p[2];
			moved[v] = 1;
		}
	}

	// Loop's odd rule as weights: 3/8 on the edge's ends, 1/8 on the two
	// vertices opposite it.
	const ls3Odd: Interpolator = (mesh, pos) => {
		const sphere = new AlgebraicSphere();
		sphere.add(pointOf(pos.v), normalOf(pos.v), 3 / 8);
		sphere.add(pointOf(pos.vFlip), normalOf(pos.vFlip), 3 / 8);
		for (const w of oppositeVertices(mesh, pos)) {
			sphere.add(pointOf(w), normalOf(w), 1 / 8);
		}
		return sphere.project();
	};

	if (!refineE(m, ls3Odd, predicate, options)) return false;
	for (let v = 0; v < originalVertSize; v++) {
		if (moved[v] && !m.isVertD(v)) m.setVert(v, even[3 * v], even[3 * v + 1], even[3 * v + 2]);
	}
	return true;
}

/** The neighbours of corner `i` of `face`, or null on a non-manifold fan. */
function oneRing(m: CMeshO, face: number, i: number): number[] | null {
	const start = new Pos(m, face, i, m.fv(face, i));
	const he = start.clone();
	const out: number[] = [];
	let guard = 0;
	do {
		out.push(he.vFlip);
		he.nextE();
		if (++guard > 1000) return null;
	} while (!he.equals(start));
	return out.length === 0 ? null : out;
}

/** The one or two vertices opposite the edge `pos` sits on. */
function oppositeVertices(m: CMeshO, pos: Pos): number[] {
	const out: number[] = [];
	const here = m.fv(pos.f, (pos.z + 2) % 3);
	out.push(here);
	if (!pos.isBorder()) {
		const g = m.ffp(pos.f, pos.z);
		out.push(m.fv(g, (m.ffi(pos.f, pos.z) + 2) % 3));
	} else {
		// On a border the odd rule has only one opposite vertex, so it counts
		// double to keep the weights summing to one.
		out.push(here);
	}
	return out;
}
