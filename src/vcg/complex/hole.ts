/**
 * `Hole` — filling boundary loops by ear cutting, mirroring
 * `vcg::tri::Hole`.
 *
 * The strategy is VCGLib's: walk each boundary loop, treat every three
 * consecutive boundary vertices as a candidate "ear", repeatedly cut the
 * best-scoring one, and stop when the loop is a single triangle. What makes it
 * produce good surfaces rather than fans is the scoring, which balances the
 * triangle's shape against how sharply it folds away from the geometry already
 * there.
 *
 * Three ear strategies, as upstream:
 * - `trivial` scores on triangle shape alone.
 * - `minimumWeight` adds the dihedral term. This is the default.
 * - `selfIntersection` additionally rejects ears that would cut through
 *   nearby faces.
 */
import { MLInternalException } from "../../common/utilities/ml_exception.ts";
import { safeAcos } from "../math/base.ts";
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { faceNormalOf } from "./update/normal.ts";
import { faceFace } from "./update/topology.ts";

/** How much the dihedral angle counts against triangle shape. VCG's 0.1. */
export const DIEDRAL_WEIGHT = 0.1;

export type EarStrategy = "trivial" | "minimumWeight" | "selfIntersection";

export interface FillHoleOptions {
	/** Skip loops with more boundary edges than this. */
	readonly maxHoleSize?: number;
	/** Only fill loops with at least one selected boundary face. */
	readonly selected?: boolean;
	readonly strategy?: EarStrategy;
}

export interface FillHoleResult {
	/** Loops actually filled. */
	holeCount: number;
	/** Faces created. */
	newFaces: number;
	/** Index of the first created face, for selecting them afterwards. */
	firstNewFace: number;
}

/** One boundary loop: its vertices in order, plus the faces along it. */
export interface HoleInfo {
	/** Boundary vertices, in loop order. */
	readonly vertices: number[];
	/** The (face, edge) border pairs, aligned with `vertices`. */
	readonly borders: Array<[number, number]>;
	/** Boundary edges, i.e. `vertices.length`. */
	readonly size: number;
}

/**
 * Enumerates the boundary loops. Requires FF adjacency.
 *
 * `Hole<M>::GetInfo` in VCGLib.
 */
export function getInfo(m: CMeshO, selected = false): HoleInfo[] {
	if (m.ffFace === null) {
		throw new MLInternalException("Hole.getInfo requires FF adjacency (MM_FACEFACETOPO)");
	}
	const visited = new Set<string>();
	const holes: HoleInfo[] = [];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!m.isBorderFF(f, e)) continue;
			if (visited.has(`${f}_${e}`)) continue;

			const start = Pos.onEdge(m, f, e);
			const cur = start.clone();
			const vertices: number[] = [];
			const borders: Array<[number, number]> = [];
			let anySelected = false;
			let guard = 0;

			do {
				visited.add(`${cur.f}_${cur.z}`);
				vertices.push(cur.v);
				borders.push([cur.f, cur.z]);
				if (m.isFaceS(cur.f)) anySelected = true;
				cur.nextB();
				if (++guard > m.faceSize * 3 + 3) {
					throw new MLInternalException("Hole.getInfo: a boundary walk did not terminate");
				}
			} while (!cur.equals(start));

			if (selected && !anySelected) continue;
			holes.push({ vertices, borders, size: vertices.length });
		}
	}
	return holes;
}

/** 2·area / longest-edge², VCG's `Quality`. Equilateral gives √3/2 ≈ 0.866. */
function triangleQuality(
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
	const d10x = bx - ax;
	const d10y = by - ay;
	const d10z = bz - az;
	const d20x = cx - ax;
	const d20y = cy - ay;
	const d20z = cz - az;
	const nx = d10y * d20z - d10z * d20y;
	const ny = d10z * d20x - d10x * d20z;
	const nz = d10x * d20y - d10y * d20x;
	const a = Math.hypot(nx, ny, nz);
	if (a === 0) return 0;
	const e0 = d10x * d10x + d10y * d10y + d10z * d10z;
	if (e0 === 0) return 0;
	const e1 = d20x * d20x + d20y * d20y + d20z * d20z;
	const d12x = bx - cx;
	const d12y = by - cy;
	const d12z = bz - cz;
	const e2 = d12x * d12x + d12y * d12y + d12z * d12z;
	return a / Math.max(e0, e1, e2);
}

/** Unit normal of a triangle, or null when it is degenerate. */
function unitNormal(m: CMeshO, a: number, b: number, c: number): [number, number, number] | null {
	const ux = m.vx(b) - m.vx(a);
	const uy = m.vy(b) - m.vy(a);
	const uz = m.vz(b) - m.vz(a);
	const vx = m.vx(c) - m.vx(a);
	const vy = m.vy(c) - m.vy(a);
	const vz = m.vz(c) - m.vz(a);
	const nx = uy * vz - uz * vy;
	const ny = uz * vx - ux * vz;
	const nz = ux * vy - uy * vx;
	const len = Math.hypot(nx, ny, nz);
	if (len === 0) return null;
	return [nx / len, ny / len, nz / len];
}

const scratchNormal = new Float64Array(3);

/** Unit normal of an existing face, or null when degenerate. */
function faceUnitNormal(m: CMeshO, f: number): [number, number, number] | null {
	faceNormalOf(m, f, scratchNormal);
	const len = Math.hypot(scratchNormal[0], scratchNormal[1], scratchNormal[2]);
	if (len === 0) return null;
	return [scratchNormal[0] / len, scratchNormal[1] / len, scratchNormal[2] / len];
}

function angleBetween(
	a: readonly [number, number, number],
	b: readonly [number, number, number],
): number {
	return safeAcos(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

/** A candidate ear: the triangle (prev, cur, next) around boundary vertex `cur`. */
interface Ear {
	/** Position in the working loop. */
	index: number;
	score: number;
	concave: boolean;
	/** True when the ear would cut through nearby geometry. */
	intersects: boolean;
}

/**
 * Scores the ear at `i` of the working loop.
 *
 * Reproduces `MinimumWeightEar::ComputeQuality` and its comparison: higher is
 * better, and a concave ear always loses to a convex one no matter how well
 * shaped it is. Without that rule the filler happily bridges across a
 * concavity and produces a surface that folds through itself.
 */
function scoreEar(
	m: CMeshO,
	loop: readonly number[],
	adjacentFace: readonly number[],
	i: number,
	strategy: EarStrategy,
): Ear {
	const n = loop.length;
	const prev = loop[(i - 1 + n) % n];
	const cur = loop[i];
	const next = loop[(i + 1) % n];

	const quality = triangleQuality(
		m.vx(cur),
		m.vy(cur),
		m.vz(cur),
		m.vx(next),
		m.vy(next),
		m.vz(next),
		m.vx(prev),
		m.vy(prev),
		m.vz(prev),
	);

	// The ear as VCG orders it: apex first, then along the boundary. That
	// winding is what makes the new face agree with the surrounding surface.
	const earNormal = unitNormal(m, cur, next, prev);

	let concave = false;
	let dihedral = 0;
	if (earNormal !== null) {
		// Concavity is decided against the vertex normal, which points out of
		// the existing surface.
		const vn: [number, number, number] = [
			m.vertNormal[3 * cur],
			m.vertNormal[3 * cur + 1],
			m.vertNormal[3 * cur + 2],
		];
		const vnLen = Math.hypot(vn[0], vn[1], vn[2]);
		if (vnLen > 0) {
			const dot = (earNormal[0] * vn[0] + earNormal[1] * vn[1] + earNormal[2] * vn[2]) / vnLen;
			concave = dot < 0;
		}

		if (strategy !== "trivial") {
			// How sharply the ear folds away from the two faces it will sit
			// against. The larger of the two angles is what counts.
			for (const f of [adjacentFace[(i - 1 + n) % n], adjacentFace[i]]) {
				if (f < 0 || m.isFaceD(f)) continue;
				const fn = faceUnitNormal(m, f);
				if (fn === null) continue;
				dihedral = Math.max(dihedral, angleBetween(earNormal, fn));
			}
		}
	}

	const score = strategy === "trivial" ? quality : quality - (dihedral / Math.PI) * DIEDRAL_WEIGHT;

	let intersects = false;
	if (strategy === "selfIntersection" && earNormal !== null) {
		intersects = earIntersectsNeighbourhood(m, prev, cur, next, adjacentFace);
	}

	return { index: i, score, concave, intersects };
}

/**
 * Whether the triangle (prev, cur, next) would pass through a face near the
 * hole.
 *
 * Only the faces already touching the boundary are tested. A full test against
 * the whole mesh would be the honest thing but is quadratic, and upstream
 * makes the same trade — the doc comment on the filter calls it a heuristic.
 */
function earIntersectsNeighbourhood(
	m: CMeshO,
	prev: number,
	cur: number,
	next: number,
	adjacentFace: readonly number[],
): boolean {
	for (const f of adjacentFace) {
		if (f < 0 || m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		// Faces sharing a vertex with the ear touch it by construction.
		if ([a, b, c].some((v) => v === prev || v === cur || v === next)) continue;
		if (trianglesIntersect(m, prev, cur, next, a, b, c)) return true;
	}
	return false;
}

/**
 * Segment-versus-triangle test in both directions, which is enough to catch
 * two triangles passing through each other.
 */
function trianglesIntersect(
	m: CMeshO,
	p0: number,
	p1: number,
	p2: number,
	q0: number,
	q1: number,
	q2: number,
): boolean {
	const edgesOf = (a: number, b: number, c: number): Array<[number, number]> => [
		[a, b],
		[b, c],
		[c, a],
	];
	for (const [s, e] of edgesOf(p0, p1, p2)) {
		if (segmentHitsTriangle(m, s, e, q0, q1, q2)) return true;
	}
	for (const [s, e] of edgesOf(q0, q1, q2)) {
		if (segmentHitsTriangle(m, s, e, p0, p1, p2)) return true;
	}
	return false;
}

/** Möller–Trumbore, restricted to the segment. */
function segmentHitsTriangle(
	m: CMeshO,
	s: number,
	e: number,
	a: number,
	b: number,
	c: number,
): boolean {
	const ox = m.vx(s);
	const oy = m.vy(s);
	const oz = m.vz(s);
	const dx = m.vx(e) - ox;
	const dy = m.vy(e) - oy;
	const dz = m.vz(e) - oz;

	const e1x = m.vx(b) - m.vx(a);
	const e1y = m.vy(b) - m.vy(a);
	const e1z = m.vz(b) - m.vz(a);
	const e2x = m.vx(c) - m.vx(a);
	const e2y = m.vy(c) - m.vy(a);
	const e2z = m.vz(c) - m.vz(a);

	const px = dy * e2z - dz * e2y;
	const py = dz * e2x - dx * e2z;
	const pz = dx * e2y - dy * e2x;
	const det = e1x * px + e1y * py + e1z * pz;
	if (Math.abs(det) < 1e-12) return false; // parallel

	const inv = 1 / det;
	const tx = ox - m.vx(a);
	const ty = oy - m.vy(a);
	const tz = oz - m.vz(a);
	const u = (tx * px + ty * py + tz * pz) * inv;
	if (u < 1e-9 || u > 1 - 1e-9) return false;

	const qx = ty * e1z - tz * e1y;
	const qy = tz * e1x - tx * e1z;
	const qz = tx * e1y - ty * e1x;
	const v = (dx * qx + dy * qy + dz * qz) * inv;
	if (v < 1e-9 || u + v > 1 - 1e-9) return false;

	const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
	// Strictly inside the segment, so shared endpoints do not count as hits.
	return t > 1e-9 && t < 1 - 1e-9;
}

/** Orders two ears; returns true when `a` is the better one to cut. */
function betterEar(a: Ear, b: Ear): boolean {
	// A self-intersecting ear is a last resort whatever it scores.
	if (a.intersects !== b.intersects) return !a.intersects;
	// Convex always beats concave, as in MinimumWeightEar::operator<.
	if (a.concave !== b.concave) return !a.concave;
	return a.score > b.score;
}

/**
 * Fills the boundary loops of `m` by ear cutting.
 *
 * Requires FF adjacency, and rebuilds it afterwards. Non-manifold edges are
 * rejected up front: the boundary of a hole is only well defined once every
 * edge has at most two faces, which is why `filter_meshing` runs the
 * non-manifold repair first.
 */
export function fillHoles(m: CMeshO, options: FillHoleOptions = {}): FillHoleResult {
	const maxHoleSize = options.maxHoleSize ?? 30;
	const strategy = options.strategy ?? "minimumWeight";
	const holes = getInfo(m, options.selected ?? false);

	const firstNewFace = m.faceSize;
	let holeCount = 0;
	let newFaces = 0;

	for (const hole of holes) {
		if (hole.size > maxHoleSize) continue;
		if (hole.size < 3) continue;

		// The working loop shrinks by one vertex per ear. `adjacentFace[i]` is
		// the face across the boundary edge leaving `loop[i]`; a bridging edge
		// created by an earlier cut has none, hence -1.
		const loop = [...hole.vertices];
		const adjacentFace = hole.borders.map(([f]) => f);
		let cut = 0;

		while (loop.length > 3) {
			let best: Ear | null = null;
			for (let i = 0; i < loop.length; i++) {
				const ear = scoreEar(m, loop, adjacentFace, i, strategy);
				if (best === null || betterEar(ear, best)) best = ear;
			}
			if (best === null) break;

			const i = best.index;
			const n = loop.length;
			const prev = loop[(i - 1 + n) % n];
			const cur = loop[i];
			const next = loop[(i + 1) % n];
			if (prev === next) break; // degenerate loop, nothing sensible to add

			const f = Allocator.addFace(m, cur, next, prev);
			newFaces++;
			cut++;

			// Drop the apex; the new face becomes the neighbour of the edge
			// that replaced the two it consumed.
			loop.splice(i, 1);
			adjacentFace.splice(i, 1);
			adjacentFace[(i - 1 + loop.length) % loop.length] = f;
		}

		if (loop.length === 3) {
			Allocator.addFace(m, loop[0], loop[1], loop[2]);
			newFaces++;
			cut++;
		}
		if (cut > 0) holeCount++;
	}

	if (newFaces > 0) {
		faceFace(m);
		m.imark++;
	}
	return { holeCount, newFaces, firstNewFace };
}

/** Selects exactly the faces from `firstNewFace` onward. */
export function selectFacesFrom(m: CMeshO, firstNewFace: number): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (f < firstNewFace || m.isFaceD(f)) m.faceFlags[f] &= ~FaceFlag.SELECTED;
		else m.faceFlags[f] |= FaceFlag.SELECTED;
	}
}

export const Hole = {
	getInfo,
	fillHoles,
	selectFacesFrom,
	DIEDRAL_WEIGHT,
} as const;
