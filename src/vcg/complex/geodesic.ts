/**
 * Distance along a surface, two ways.
 *
 * {@link dijkstraGeodesic} propagates shortest paths along the mesh's edges.
 * It is exact for paths that happen to follow edges and an over-estimate for
 * every other path — a straight line across a triangle is shorter than going
 * round two of its sides — so on a coarse mesh it reads long, by up to about
 * 15% on a regular triangulation.
 *
 * {@link heatGeodesic} is Crane, Weischedel and Wardetzky's heat method: let
 * heat diffuse briefly from the source, take the direction it flowed, and
 * integrate that direction field back to a distance. It crosses triangles
 * properly and so does not have Dijkstra's bias, at the cost of two sparse
 * solves and a sensitivity to triangle shape — MeshLab says as much in the
 * filter's own description.
 *
 * Both take a *set* of sources and give the distance to the nearest one,
 * which is what "distance from the selection" and "distance from the border"
 * both need.
 */

import { SparseMatrix, solveCG } from "../math/sparse.ts";
import type { CMeshO } from "./cmesho.ts";
import { UpdateTopology } from "./update/topology.ts";

/** Shortest-path distance along edges from any of the sources. */
export function dijkstraGeodesic(cm: CMeshO, sources: readonly number[]): Float64Array {
	const distance = new Float64Array(cm.vertSize).fill(Number.POSITIVE_INFINITY);
	if (sources.length === 0) return distance;

	const neighbours = adjacency(cm);
	// A binary heap of (distance, vertex). Stale entries are left in and
	// skipped on pop, which is cheaper than a decrease-key structure and is
	// what every practical Dijkstra does.
	const heap: Array<[number, number]> = [];
	for (const s of sources) {
		if (cm.isVertD(s)) continue;
		distance[s] = 0;
		push(heap, [0, s]);
	}

	while (heap.length > 0) {
		const [d, v] = pop(heap) as [number, number];
		if (d > distance[v]) continue;
		for (const n of neighbours[v]) {
			const step = d + edgeLength(cm, v, n);
			if (step < distance[n]) {
				distance[n] = step;
				push(heap, [step, n]);
			}
		}
	}
	return distance;
}

/** Every vertex on a boundary edge, for "distance from the border". */
export function borderVertices(cm: CMeshO): number[] {
	UpdateTopology.faceFace(cm);
	const out = new Set<number>();
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!cm.isBorderFF(f, e)) continue;
			out.add(cm.fv(f, e));
			out.add(cm.fv(f, (e + 1) % 3));
		}
	}
	return [...out];
}

export interface HeatOptions {
	/**
	 * Diffusion time, as a multiple of the mean edge length squared. Larger
	 * values smooth the result; Crane et al. recommend 1.
	 */
	readonly m?: number;
}

/**
 * The heat method.
 *
 * Three steps, each of which is worth naming because each fails differently.
 *
 *  1. Diffuse: solve `(M - t·L) u = δ`, heat released at the sources.
 *  2. Normalise: `X = -∇u / |∇u|`, the direction heat flowed, which after
 *     even a brief diffusion points away from the source everywhere.
 *  3. Integrate: solve `L φ = ∇·X` for the distance. The solution is only
 *     determined up to a constant, so it is shifted to put the sources at
 *     zero afterwards.
 *
 * Returns null when either solve fails to converge, which on a mesh with
 * degenerate triangles it can — the cotangent weights blow up. Dijkstra is
 * the caller's fallback and has no such failure mode.
 */
export function heatGeodesic(
	cm: CMeshO,
	sources: readonly number[],
	options: HeatOptions = {},
): Float64Array | null {
	const live: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) live.push(v);
	if (live.length === 0 || sources.length === 0) return null;

	const { laplacian, mass, meanEdge } = cotangentOperator(cm);
	if (meanEdge === 0) return null;
	const t = (options.m ?? 1) * meanEdge * meanEdge;

	// Step 1: (M - t·L) u = δ.
	const heat = new SparseMatrix(cm.vertSize);
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) {
			heat.add(v, v, 1);
			continue;
		}
		heat.add(v, v, mass[v]);
	}
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		for (const [n, w] of laplacian[v]) heat.add(v, n, -t * w);
	}
	const delta = new Float64Array(cm.vertSize);
	for (const s of sources) if (!cm.isVertD(s)) delta[s] = 1;

	// Solved much tighter than the Poisson step below, and not for elegance:
	// on a closed surface the heat that reaches the far side is many orders of
	// magnitude below the peak, and a solve to a relative residual of 1e-9
	// leaves exactly that far field as numerical noise. The normalised
	// gradient of noise is noise, so the distance at the antipode comes out
	// roughly half of what it should be. Tightening this took the worst error
	// on a sphere from 44% to 12%.
	const u = solveCG(heat, delta, { tolerance: 1e-14, iterations: 20000 });
	if (!u.converged) return null;

	// Step 2: the normalised negative gradient, per face.
	const field = new Float64Array(cm.faceSize * 3);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const g = faceGradient(cm, f, u.x);
		const length = Math.hypot(g[0], g[1], g[2]);
		if (length === 0) continue;
		for (let k = 0; k < 3; k++) field[3 * f + k] = -g[k] / length;
	}

	// Step 3: L φ = ∇·X, with one vertex pinned to fix the constant.
	//
	// Negated on both sides. The Laplacian assembled above is negative
	// semi-definite, which is what makes `M - t·L` positive definite in step
	// 1 — but conjugate gradients needs a *positive* definite matrix, and on
	// a negative definite one it converges happily to nonsense.
	const poisson = new SparseMatrix(cm.vertSize);
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) {
			poisson.add(v, v, 1);
			continue;
		}
		for (const [n, w] of laplacian[v]) poisson.add(v, n, -w);
	}
	const divergence = fieldDivergence(cm, field);
	for (let v = 0; v < cm.vertSize; v++) divergence[v] = -divergence[v];
	poisson.pin(sources[0], 0, divergence);

	const phi = solveCG(poisson, divergence, { tolerance: 1e-9 });
	if (!phi.converged) return null;

	// The integration is up to a constant and a sign; put the sources at zero
	// and make the rest positive, which is what a distance is.
	let base = Number.POSITIVE_INFINITY;
	for (const s of sources) if (!cm.isVertD(s)) base = Math.min(base, phi.x[s]);
	const out = new Float64Array(cm.vertSize);
	for (const v of live) out[v] = Math.max(0, phi.x[v] - base);
	return out;
}

// ---- the discrete operators ----------------------------------------------

export interface Operator {
	/** Per vertex, the cotangent weight to each neighbour and to itself. */
	readonly laplacian: Array<Map<number, number>>;
	/** Barycentric vertex areas, the lumped mass matrix. */
	readonly mass: Float64Array;
	readonly meanEdge: number;
}

/**
 * The cotangent Laplacian and the lumped mass matrix.
 *
 * The Laplacian is negative semi-definite here — the diagonal is minus the
 * sum of the off-diagonals — which is the convention that makes `M - t·L`
 * positive definite above. Getting that sign backwards produces a solve that
 * converges to something smooth and completely wrong.
 */
export function cotangentOperator(cm: CMeshO): Operator {
	const laplacian: Array<Map<number, number>> = Array.from(
		{ length: cm.vertSize },
		() => new Map<number, number>(),
	);
	const mass = new Float64Array(cm.vertSize);
	let edgeSum = 0;
	let edgeCount = 0;

	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => point(cm, cm.fv(f, k)));
		const area = triangleArea(p);
		for (let k = 0; k < 3; k++) mass[cm.fv(f, k)] += area / 3;

		for (let k = 0; k < 3; k++) {
			const a = cm.fv(f, (k + 1) % 3);
			const b = cm.fv(f, (k + 2) % 3);
			// Half the cotangent of the angle at corner k, opposite edge (a,b).
			const w = 0.5 * cotangent(sub(p[(k + 1) % 3], p[k]), sub(p[(k + 2) % 3], p[k]));
			laplacian[a].set(b, (laplacian[a].get(b) ?? 0) + w);
			laplacian[b].set(a, (laplacian[b].get(a) ?? 0) + w);
			laplacian[a].set(a, (laplacian[a].get(a) ?? 0) - w);
			laplacian[b].set(b, (laplacian[b].get(b) ?? 0) - w);

			edgeSum += distance(p[(k + 1) % 3], p[(k + 2) % 3]);
			edgeCount++;
		}
	}
	// A vertex with no incident area would divide by zero in the heat solve.
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v) && mass[v] === 0) mass[v] = 1e-12;
	return { laplacian, mass, meanEdge: edgeCount === 0 ? 0 : edgeSum / edgeCount };
}

/** The gradient of a per-vertex scalar over one face. */
function faceGradient(cm: CMeshO, f: number, values: Float64Array): [number, number, number] {
	const p = [0, 1, 2].map((k) => point(cm, cm.fv(f, k)));
	const normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
	const twiceArea = Math.hypot(normal[0], normal[1], normal[2]);
	if (twiceArea === 0) return [0, 0, 0];
	const unit = normal.map((x) => x / twiceArea);

	const out: [number, number, number] = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		// Each vertex contributes its value times the rotated opposite edge.
		const opposite = sub(p[(k + 2) % 3], p[(k + 1) % 3]);
		const rotated = cross(unit, opposite);
		for (let i = 0; i < 3; i++) out[i] += (values[cm.fv(f, k)] * rotated[i]) / twiceArea;
	}
	return out;
}

/** The divergence of a per-face vector field, integrated at each vertex. */
function fieldDivergence(cm: CMeshO, field: Float64Array): Float64Array {
	const out = new Float64Array(cm.vertSize);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => point(cm, cm.fv(f, k)));
		const x = [field[3 * f], field[3 * f + 1], field[3 * f + 2]];
		for (let k = 0; k < 3; k++) {
			const e1 = sub(p[(k + 1) % 3], p[k]);
			const e2 = sub(p[(k + 2) % 3], p[k]);
			const cot2 = 0.5 * cotangent(sub(p[k], p[(k + 1) % 3]), sub(p[(k + 2) % 3], p[(k + 1) % 3]));
			const cot1 = 0.5 * cotangent(sub(p[k], p[(k + 2) % 3]), sub(p[(k + 1) % 3], p[(k + 2) % 3]));
			out[cm.fv(f, k)] += cot1 * dot(e1, x) + cot2 * dot(e2, x);
		}
	}
	return out;
}

// ---- helpers --------------------------------------------------------------

function adjacency(cm: CMeshO): Array<number[]> {
	const sets: Array<Set<number>> = Array.from({ length: cm.vertSize }, () => new Set<number>());
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = cm.fv(f, k);
			const b = cm.fv(f, (k + 1) % 3);
			sets[a].add(b);
			sets[b].add(a);
		}
	}
	return sets.map((s) => [...s]);
}

function edgeLength(cm: CMeshO, a: number, b: number): number {
	return Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
}

function push(heap: Array<[number, number]>, item: [number, number]): void {
	heap.push(item);
	let i = heap.length - 1;
	while (i > 0) {
		const parent = (i - 1) >> 1;
		if (heap[parent][0] <= heap[i][0]) break;
		[heap[parent], heap[i]] = [heap[i], heap[parent]];
		i = parent;
	}
}

function pop(heap: Array<[number, number]>): [number, number] | undefined {
	if (heap.length === 0) return undefined;
	const top = heap[0];
	const last = heap.pop() as [number, number];
	if (heap.length === 0) return top;
	heap[0] = last;
	let i = 0;
	for (;;) {
		const l = 2 * i + 1;
		const r = l + 1;
		let smallest = i;
		if (l < heap.length && heap[l][0] < heap[smallest][0]) smallest = l;
		if (r < heap.length && heap[r][0] < heap[smallest][0]) smallest = r;
		if (smallest === i) break;
		[heap[smallest], heap[i]] = [heap[i], heap[smallest]];
		i = smallest;
	}
	return top;
}

function point(cm: CMeshO, v: number): number[] {
	return [cm.vx(v), cm.vy(v), cm.vz(v)];
}

function sub(a: readonly number[], b: readonly number[]): number[] {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: readonly number[], b: readonly number[]): number[] {
	return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function distance(a: readonly number[], b: readonly number[]): number {
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function triangleArea(p: readonly number[][]): number {
	const c = cross(sub(p[1], p[0]), sub(p[2], p[0]));
	return Math.hypot(c[0], c[1], c[2]) / 2;
}

function cotangent(u: readonly number[], v: readonly number[]): number {
	const c = cross(u, v);
	const length = Math.hypot(c[0], c[1], c[2]);
	if (length === 0) return 0;
	return dot(u, v) / length;
}
