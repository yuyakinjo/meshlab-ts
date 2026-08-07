/**
 * `SurfaceSampling` — drawing a point set from a mesh, mirroring
 * `vcg/complex/algorithms/point_sampling.h`.
 *
 * Every function here returns a *new* `CMeshO` holding only vertices: no
 * faces, because a sample set is not a surface. That is how MeshLab models it
 * too — the sampling filters add a new point-cloud layer rather than editing
 * the mesh they sampled.
 *
 * Sampling is either of the vertices (clustered) or of the surface itself
 * (Montecarlo, stratified, Poisson-disk); the latter interpolate inside faces
 * and so produce points the input never contained.
 */

import { KdTree, pointBounds } from "../space/index/kdtree.ts";
import { Allocator } from "./allocator.ts";
import { CMeshO } from "./cmesho.ts";
import { faceNormalOf } from "./update/normal.ts";

/**
 * A deterministic pseudo-random source.
 *
 * MeshLab seeds from the clock, so two runs of a sampling filter differ. That
 * is unhelpful in a pipeline whose outputs are hashed, so sampling here is
 * seeded and reproducible by default; pass a different seed for a different
 * draw.
 */
export class Rng {
	private state: number;

	constructor(seed = 0x2f6e2b1) {
		// Zero is a fixed point of the generator, so keep it out.
		this.state = seed >>> 0 || 0x2f6e2b1;
	}

	/** A float in [0, 1). xorshift32, which is fast and good enough for sampling. */
	next(): number {
		let x = this.state;
		x ^= x << 13;
		x >>>= 0;
		x ^= x >> 17;
		x ^= x << 5;
		x >>>= 0;
		this.state = x;
		return x / 4294967296;
	}
}

/** Copies the given vertices of `src` into a new vertex-only mesh. */
function gatherVertices(src: CMeshO, kept: ArrayLike<number>): CMeshO {
	const out = new CMeshO();
	const n = kept.length;
	if (n === 0) return out;
	const first = Allocator.addVertices(out, n);
	for (let i = 0; i < n; i++) {
		const v = kept[i];
		out.setVert(first + i, src.vx(v), src.vy(v), src.vz(v));
		out.vertQuality[first + i] = src.vertQuality[v];
		out.vertColor[first + i] = src.vertColor[v];
		for (let k = 0; k < 3; k++) {
			out.vertNormal[3 * (first + i) + k] = src.vertNormal[3 * v + k];
		}
	}
	return out;
}

/**
 * `Clustered Vertex Sampling`, with MeshLab's "Closest to center" rule: lay a
 * grid whose cells are `percent` of the bounding-box diagonal and keep, per
 * occupied cell, the input vertex nearest that cell's centre.
 *
 * A subset of the input points, never an interpolation — which is what makes
 * it the right thinning step before normal estimation, where an invented point
 * would have no neighbourhood to fit a plane to.
 */
export function clusteredVertexSampling(m: CMeshO, percent: number): CMeshO {
	if (percent <= 0 || m.vn === 0) return gatherVertices(m, liveVertices(m));
	const bounds = pointBounds(m.vertCoord, m.vertSize);
	const cell = (bounds.diagonal * percent) / 100;
	if (!(cell > 0)) return gatherVertices(m, liveVertices(m));

	const chosen = new Map<string, { point: number; distance: number }>();
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		let distance = 0;
		let key = "";
		for (let axis = 0; axis < 3; axis++) {
			const value = m.vertCoord[3 * v + axis];
			const cellIndex = Math.floor((value - bounds.low[axis]) / cell);
			const centre = bounds.low[axis] + (cellIndex + 0.5) * cell;
			distance += (value - centre) ** 2;
			key += `${cellIndex},`;
		}
		const current = chosen.get(key);
		if (current === undefined || distance < current.distance) {
			chosen.set(key, { point: v, distance });
		}
	}
	// Sorted, so the result does not depend on Map iteration order.
	const kept = Int32Array.from([...chosen.values()], (e) => e.point).sort();
	return gatherVertices(m, kept);
}

function liveVertices(m: CMeshO): number[] {
	const out: number[] = [];
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) out.push(v);
	return out;
}

/** Twice the area of face `f`. */
function faceDoubleArea(m: CMeshO, f: number, scratch: Float64Array): number {
	faceNormalOf(m, f, scratch);
	return Math.hypot(scratch[0], scratch[1], scratch[2]);
}

/**
 * A uniformly random barycentric coordinate.
 *
 * The square root is what makes it uniform over the triangle: sampling `u` and
 * `v` directly and rejecting `u + v > 1` also works but throws away half the
 * draws, and simply clamping would bunch the points along the diagonal.
 */
function randomBarycentric(rng: Rng): [number, number, number] {
	const r1 = Math.sqrt(rng.next());
	const r2 = rng.next();
	return [1 - r1, r1 * (1 - r2), r1 * r2];
}

function pointOnFace(
	m: CMeshO,
	f: number,
	bary: readonly [number, number, number],
): [number, number, number] {
	const a = m.fv(f, 0);
	const b = m.fv(f, 1);
	const c = m.fv(f, 2);
	return [
		bary[0] * m.vx(a) + bary[1] * m.vx(b) + bary[2] * m.vx(c),
		bary[0] * m.vy(a) + bary[1] * m.vy(b) + bary[2] * m.vy(c),
		bary[0] * m.vz(a) + bary[1] * m.vz(b) + bary[2] * m.vz(c),
	];
}

/**
 * `Montecarlo Sampling`: scatter `sampleNum` points over the surface with
 * probability proportional to face area.
 *
 * Area weighting is what makes it uniform over the *surface*. Picking faces
 * with equal probability would crowd the small ones, which on a mesh whose
 * triangles vary in size is very visible.
 */
export function montecarloSampling(m: CMeshO, sampleNum: number, rng = new Rng()): CMeshO {
	const out = new CMeshO();
	if (m.fn === 0 || sampleNum <= 0) return out;

	// Cumulative area, so a single uniform draw picks a face by binary search.
	const faces: number[] = [];
	const cumulative: number[] = [];
	const scratch = new Float64Array(3);
	let total = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		total += faceDoubleArea(m, f, scratch) / 2;
		faces.push(f);
		cumulative.push(total);
	}
	if (total <= 0) return out;

	const first = Allocator.addVertices(out, sampleNum);
	for (let i = 0; i < sampleNum; i++) {
		const target = rng.next() * total;
		let lo = 0;
		let hi = cumulative.length - 1;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (cumulative[mid] < target) lo = mid + 1;
			else hi = mid;
		}
		const p = pointOnFace(m, faces[lo], randomBarycentric(rng));
		out.setVert(first + i, p[0], p[1], p[2]);
	}
	return out;
}

/**
 * `Stratified Triangle Sampling`: every face gets a share of the samples in
 * proportion to its area, rather than being drawn for independently.
 *
 * Lower variance than Montecarlo for the same count — no face can happen to
 * receive none — at the cost of a slightly regular look.
 */
export function stratifiedSampling(m: CMeshO, sampleNum: number, rng = new Rng()): CMeshO {
	const out = new CMeshO();
	if (m.fn === 0 || sampleNum <= 0) return out;

	const scratch = new Float64Array(3);
	let total = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (!m.isFaceD(f)) total += faceDoubleArea(m, f, scratch) / 2;
	}
	if (total <= 0) return out;

	const points: Array<[number, number, number]> = [];
	// Carry the fractional remainder from face to face, so the total lands on
	// sampleNum instead of drifting by the rounding of every face.
	let debt = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const share = (faceDoubleArea(m, f, scratch) / 2 / total) * sampleNum + debt;
		const n = Math.floor(share);
		debt = share - n;
		for (let i = 0; i < n; i++) points.push(pointOnFace(m, f, randomBarycentric(rng)));
	}

	if (points.length === 0) return out;
	const first = Allocator.addVertices(out, points.length);
	for (let i = 0; i < points.length; i++) {
		out.setVert(first + i, points[i][0], points[i][1], points[i][2]);
	}
	return out;
}

/**
 * `Poisson-disk Sampling`: a subset of `candidates` in which no two points are
 * closer than `radius`.
 *
 * Dart throwing over a spatial hash — take candidates in order and keep one
 * only when nothing already kept is within the radius. Simpler than the
 * hierarchical scheme upstream uses and gives the same guarantee; what it does
 * not give is upstream's tight *coverage* bound, so it can leave slightly
 * larger gaps.
 */
export function poissonDiskPruning(candidates: CMeshO, radius: number, _rng = new Rng()): CMeshO {
	if (radius <= 0 || candidates.vn === 0)
		return gatherVertices(candidates, liveVertices(candidates));

	const cell = radius;
	const buckets = new Map<string, number[]>();
	const keyOf = (x: number, y: number, z: number) =>
		`${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
	const r2 = radius * radius;
	const kept: number[] = [];

	for (let v = 0; v < candidates.vertSize; v++) {
		if (candidates.isVertD(v)) continue;
		const x = candidates.vx(v);
		const y = candidates.vy(v);
		const z = candidates.vz(v);
		const cx = Math.floor(x / cell);
		const cy = Math.floor(y / cell);
		const cz = Math.floor(z / cell);

		let tooClose = false;
		for (let dx = -1; dx <= 1 && !tooClose; dx++) {
			for (let dy = -1; dy <= 1 && !tooClose; dy++) {
				for (let dz = -1; dz <= 1 && !tooClose; dz++) {
					for (const w of buckets.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
						const ex = candidates.vx(w) - x;
						const ey = candidates.vy(w) - y;
						const ez = candidates.vz(w) - z;
						if (ex * ex + ey * ey + ez * ez < r2) {
							tooClose = true;
							break;
						}
					}
				}
			}
		}
		if (tooClose) continue;

		kept.push(v);
		const key = keyOf(x, y, z);
		const hit = buckets.get(key);
		if (hit === undefined) buckets.set(key, [v]);
		else hit.push(v);
	}
	return gatherVertices(candidates, kept);
}

/** `Mesh Element Sampling` over vertices: every live vertex, as a point set. */
export function vertexSampling(m: CMeshO): CMeshO {
	return gatherVertices(m, liveVertices(m));
}

/**
 * `meshing_remove_duplicate_vertices` for a point set: exactly equal
 * coordinates only.
 */
export function removeDuplicatePoints(m: CMeshO): CMeshO {
	const seen = new Set<string>();
	const kept: number[] = [];
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const key = `${m.vx(v)}:${m.vy(v)}:${m.vz(v)}`;
		if (seen.has(key)) continue;
		seen.add(key);
		kept.push(v);
	}
	return gatherVertices(m, kept);
}

/**
 * The one-sided Hausdorff distance from `from`'s vertices to `to`'s vertices,
 * plus the mean.
 *
 * Point-to-point, not point-to-surface: the caller is expected to sample `to`
 * densely enough first, which is exactly what MeshLab's Hausdorff filter does
 * with its sampling parameters.
 */
export function hausdorffPointDistance(
	from: CMeshO,
	to: CMeshO,
): { max: number; mean: number; rms: number } {
	if (from.vn === 0 || to.vn === 0) return { max: 0, mean: 0, rms: 0 };
	const index = new KdTree(to.vertCoord, to.vertSize);
	let max = 0;
	let sum = 0;
	let sumSq = 0;
	let n = 0;
	for (let v = 0; v < from.vertSize; v++) {
		if (from.isVertD(v)) continue;
		// The tree indexes `to`, so query by coordinate through a temporary.
		const nearest = index.nearestToPoint(from.vx(v), from.vy(v), from.vz(v));
		if (nearest < 0) continue;
		const d = Math.hypot(
			from.vx(v) - to.vx(nearest),
			from.vy(v) - to.vy(nearest),
			from.vz(v) - to.vz(nearest),
		);
		if (d > max) max = d;
		sum += d;
		sumSq += d * d;
		n++;
	}
	return n === 0 ? { max: 0, mean: 0, rms: 0 } : { max, mean: sum / n, rms: Math.sqrt(sumSq / n) };
}

export const SurfaceSampling = {
	clusteredVertexSampling,
	montecarloSampling,
	stratifiedSampling,
	poissonDiskPruning,
	vertexSampling,
	removeDuplicatePoints,
	hausdorffPointDistance,
	Rng,
} as const;
