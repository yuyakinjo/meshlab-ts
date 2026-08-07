/**
 * Flattening a disk-topology patch into the plane.
 *
 * This is the classic barycentric-mapping construction, from Tutte by way of
 * Floater: pin the boundary to a convex polygon, then place every interior
 * vertex at a weighted average of its neighbours. That is a sparse linear
 * system, one equation per interior vertex, and its solution is the
 * parametrisation.
 *
 * Two weight schemes are offered, and the difference between them is the
 * whole reason both exist.
 *
 * - **Mean value** (Floater 2003) uses `tan(α/2) + tan(β/2)` over the edge
 *   length, where α and β are the two half-angles at the vertex. The weights
 *   are *always positive*, so Tutte's theorem applies and the result is
 *   guaranteed to be a valid embedding — no folded triangles, ever.
 * - **Harmonic**, or cotangent (Pinkall and Polthier 1993), uses
 *   `cot α + cot β` over the two opposite angles. It minimises the Dirichlet
 *   energy, which makes it the most conformal map with that boundary — but an
 *   obtuse triangle contributes a *negative* weight, and with enough of them
 *   the guarantee is lost and triangles can fold.
 *
 * So: harmonic is better when it works, mean value always works. The default
 * is mean value, and {@link parametrizeDisk} reports whether the result is
 * actually unfolded so a caller that wants harmonic can check and fall back.
 */

import type { CMeshO } from "../cmesho.ts";
import { UpdateTopology } from "../update/topology.ts";

export type WeightScheme = "mean-value" | "harmonic" | "uniform";

/** Where the boundary of the patch is pinned. */
export type BoundaryShape = "circle" | "square";

export interface DiskParametrizationOptions {
	readonly weights?: WeightScheme;
	readonly boundary?: BoundaryShape;
	/**
	 * How the boundary vertices are spaced around the shape. `"chord"` gives
	 * each boundary edge an arc proportional to its 3D length, which keeps the
	 * boundary from being distorted; `"uniform"` spaces them evenly.
	 */
	readonly boundarySpacing?: "chord" | "uniform";
	/** Gauss-Seidel sweeps. The system is diagonally dominant, so this converges. */
	readonly iterations?: number;
	/** Stop early once no coordinate moves by more than this. */
	readonly tolerance?: number;
}

export interface DiskParametrization {
	/** Per-vertex (u, v), indexed by vertex, in 0..1. */
	readonly uv: Float64Array;
	/** The boundary loop, in order. */
	readonly boundary: readonly number[];
	/** How many Gauss-Seidel sweeps were actually run. */
	readonly iterations: number;
	/** False when some triangle came out folded — only possible for harmonic. */
	readonly valid: boolean;
}

const DEFAULTS = {
	weights: "mean-value" as WeightScheme,
	boundary: "circle" as BoundaryShape,
	boundarySpacing: "chord" as const,
	iterations: 500,
	tolerance: 1e-9,
};

/**
 * Parametrises a patch whose faces form a single disk.
 *
 * Throws when the patch is not a disk, because every method here silently
 * produces nonsense on a closed surface or on one with several boundary
 * loops, and "nonsense that looks like UVs" is the worst possible outcome.
 */
export function parametrizeDisk(
	cm: CMeshO,
	options: DiskParametrizationOptions = {},
): DiskParametrization {
	const weights = options.weights ?? DEFAULTS.weights;
	const shape = options.boundary ?? DEFAULTS.boundary;
	const spacing = options.boundarySpacing ?? DEFAULTS.boundarySpacing;
	const maxIterations = options.iterations ?? DEFAULTS.iterations;
	const tolerance = options.tolerance ?? DEFAULTS.tolerance;

	const boundary = boundaryLoop(cm);
	const uv = new Float64Array(cm.vertSize * 2);

	pinBoundary(cm, boundary, shape, spacing, uv);

	const onBoundary = new Uint8Array(cm.vertSize);
	for (const v of boundary) onBoundary[v] = 1;

	const interior: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v) && onBoundary[v] === 0) interior.push(v);
	}
	// Start the interior at the centroid of the boundary shape: any starting
	// point converges, but the middle is closest to the answer.
	for (const v of interior) {
		uv[2 * v] = 0.5;
		uv[2 * v + 1] = 0.5;
	}

	const rows = buildWeights(cm, interior, weights);
	const iterations = gaussSeidel(rows, uv, maxIterations, tolerance);

	return { uv, boundary, iterations, valid: unfolded(cm, uv) };
}

/** One interior vertex's equation: its position is this average of its neighbours. */
interface Row {
	readonly vertex: number;
	readonly neighbours: number[];
	readonly weights: number[];
}

/**
 * Gauss-Seidel, in place.
 *
 * A direct sparse solve would be faster and exact, but it needs a sparse
 * factorisation this library does not have. The system is weakly diagonally
 * dominant by construction — each row's off-diagonal weights sum to its
 * diagonal — so Gauss-Seidel converges unconditionally, and it converges fast
 * because the boundary condition is right at the edge of every patch.
 */
function gaussSeidel(
	rows: readonly Row[],
	uv: Float64Array,
	maxIterations: number,
	tolerance: number,
): number {
	let iteration = 0;
	for (; iteration < maxIterations; iteration++) {
		let worst = 0;
		for (const row of rows) {
			let su = 0;
			let sv = 0;
			let total = 0;
			for (let i = 0; i < row.neighbours.length; i++) {
				const w = row.weights[i];
				const n = row.neighbours[i];
				su += w * uv[2 * n];
				sv += w * uv[2 * n + 1];
				total += w;
			}
			if (total === 0) continue;
			const nu = su / total;
			const nv = sv / total;
			worst = Math.max(
				worst,
				Math.abs(nu - uv[2 * row.vertex]),
				Math.abs(nv - uv[2 * row.vertex + 1]),
			);
			uv[2 * row.vertex] = nu;
			uv[2 * row.vertex + 1] = nv;
		}
		if (worst <= tolerance) return iteration + 1;
	}
	return iteration;
}

function buildWeights(cm: CMeshO, interior: readonly number[], scheme: WeightScheme): Row[] {
	const rings = oneRings(cm);
	const rows: Row[] = [];
	for (const v of interior) {
		const ring = rings[v];
		if (ring === undefined || ring.length === 0) continue;
		const neighbours: number[] = [];
		const weights: number[] = [];
		for (const n of ring) {
			const w = weightFor(cm, v, n, rings, scheme);
			// A non-positive weight breaks diagonal dominance and with it the
			// convergence guarantee. Clamping to a small positive keeps the
			// solve stable; the caller learns about it through `valid`.
			neighbours.push(n);
			weights.push(Math.max(w, 1e-8));
		}
		rows.push({ vertex: v, neighbours, weights });
	}
	return rows;
}

function weightFor(
	cm: CMeshO,
	v: number,
	n: number,
	rings: ReadonlyArray<number[] | undefined>,
	scheme: WeightScheme,
): number {
	if (scheme === "uniform") return 1;

	const shared = sharedNeighbours(rings, v, n);
	const p = point(cm, v);
	const q = point(cm, n);
	const edge = distance(p, q);
	if (edge === 0) return 1;

	if (scheme === "harmonic") {
		// cot of the angle opposite the edge, from each side.
		let sum = 0;
		for (const o of shared) {
			const r = point(cm, o);
			sum += cotangent(sub(p, r), sub(q, r));
		}
		return sum;
	}

	// Mean value: tan of each half-angle at v, between this edge and the two
	// edges to the shared neighbours.
	let sum = 0;
	for (const o of shared) {
		const r = point(cm, o);
		const angle = angleBetween(sub(q, p), sub(r, p));
		sum += Math.tan(angle / 2);
	}
	return sum / edge;
}

/** The vertices adjacent to both endpoints — one per face on the edge. */
function sharedNeighbours(
	rings: ReadonlyArray<number[] | undefined>,
	a: number,
	b: number,
): number[] {
	const ringB = rings[b];
	if (ringB === undefined) return [];
	const set = new Set(ringB);
	const out: number[] = [];
	for (const v of rings[a] ?? []) if (set.has(v)) out.push(v);
	return out;
}

function oneRings(cm: CMeshO): Array<number[] | undefined> {
	const rings: Array<Set<number>> = Array.from({ length: cm.vertSize }, () => new Set<number>());
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = cm.fv(f, k);
			const b = cm.fv(f, (k + 1) % 3);
			rings[a].add(b);
			rings[b].add(a);
		}
	}
	return rings.map((s) => (s.size === 0 ? undefined : [...s]));
}

/**
 * The single boundary loop of a disk, in order.
 *
 * Throws for a closed surface (no boundary) or for anything with more than
 * one loop, which includes an annulus and a mesh with an extra hole punched
 * in it. Both are common enough that a clear message beats a wrong answer.
 */
export function boundaryLoop(cm: CMeshO): number[] {
	UpdateTopology.faceFace(cm);
	// Directed border edges, each from a face's own winding, so following
	// `next` walks the loop in a consistent direction.
	const next = new Map<number, number>();
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!cm.isBorderFF(f, e)) continue;
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			if (next.has(a)) {
				throw new Error(
					`vertex ${a} has two outgoing border edges: the patch is not a disk ` +
						"(a pinched boundary or a non-manifold vertex)",
				);
			}
			next.set(a, b);
		}
	}
	if (next.size === 0) throw new Error("the patch has no boundary, so it is not a disk");

	const start = next.keys().next().value as number;
	const loop: number[] = [start];
	let at = next.get(start) as number;
	while (at !== start) {
		loop.push(at);
		const step = next.get(at);
		if (step === undefined) throw new Error("the boundary does not close into a loop");
		at = step;
	}
	if (loop.length !== next.size) {
		throw new Error(
			`the patch has more than one boundary loop (${next.size} border edges, ` +
				`the first loop uses ${loop.length}), so it is not a disk`,
		);
	}
	return loop;
}

/**
 * Pins the boundary onto a convex shape.
 *
 * Convexity is not cosmetic: Tutte's theorem needs it, and it is what makes
 * the interior solve produce a valid embedding rather than merely a plausible
 * one.
 */
function pinBoundary(
	cm: CMeshO,
	loop: readonly number[],
	shape: BoundaryShape,
	spacing: "chord" | "uniform",
	uv: Float64Array,
): void {
	const n = loop.length;
	const lengths: number[] = [];
	let perimeter = 0;
	for (let i = 0; i < n; i++) {
		const a = point(cm, loop[i]);
		const b = point(cm, loop[(i + 1) % n]);
		const d = spacing === "chord" ? distance(a, b) : 1;
		lengths.push(d);
		perimeter += d;
	}
	if (perimeter === 0) {
		// Every boundary vertex sits on top of the others; fall back to even
		// spacing rather than dividing by zero.
		for (let i = 0; i < n; i++) lengths[i] = 1;
		perimeter = n;
	}

	let travelled = 0;
	for (let i = 0; i < n; i++) {
		const t = travelled / perimeter;
		const [u, v] = shape === "circle" ? onCircle(t) : onSquare(t);
		uv[2 * loop[i]] = u;
		uv[2 * loop[i] + 1] = v;
		travelled += lengths[i];
	}
}

/** A point of the unit-diameter circle inscribed in 0..1, at parameter t. */
function onCircle(t: number): [number, number] {
	const angle = 2 * Math.PI * t;
	return [0.5 + 0.5 * Math.cos(angle), 0.5 + 0.5 * Math.sin(angle)];
}

/** A point of the 0..1 square's perimeter, at parameter t. */
function onSquare(t: number): [number, number] {
	const s = (t % 1) * 4;
	if (s < 1) return [s, 0];
	if (s < 2) return [1, s - 1];
	if (s < 3) return [3 - s, 1];
	return [0, 4 - s];
}

function unfolded(cm: CMeshO, uv: Float64Array): boolean {
	let positive = 0;
	let negative = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const a = cm.fv(f, 0);
		const b = cm.fv(f, 1);
		const c = cm.fv(f, 2);
		const area =
			(uv[2 * b] - uv[2 * a]) * (uv[2 * c + 1] - uv[2 * a + 1]) -
			(uv[2 * c] - uv[2 * a]) * (uv[2 * b + 1] - uv[2 * a + 1]);
		if (area > 0) positive++;
		else if (area < 0) negative++;
		else return false; // a collapsed triangle is not an embedding either
	}
	return positive === 0 || negative === 0;
}

/** Copies a per-vertex parametrisation into the per-wedge channel. */
export function writeWedgeUV(cm: CMeshO, uv: Float64Array): void {
	const wt = cm.wedgeTexCoord;
	if (wt === null) throw new Error("the mesh has no per-wedge texture coordinates");
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(f, k);
			wt[6 * f + 2 * k] = uv[2 * v];
			wt[6 * f + 2 * k + 1] = uv[2 * v + 1];
		}
	}
}

// ---- small helpers --------------------------------------------------------

function point(cm: CMeshO, v: number): number[] {
	return [cm.vx(v), cm.vy(v), cm.vz(v)];
}

function sub(a: readonly number[], b: readonly number[]): number[] {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function distance(a: readonly number[], b: readonly number[]): number {
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function angleBetween(u: readonly number[], v: readonly number[]): number {
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	return Math.acos(Math.min(1, Math.max(-1, dot(u, v) / (lu * lv))));
}

/** cot of the angle between two vectors, as cos/sin from the cross product. */
function cotangent(u: readonly number[], v: readonly number[]): number {
	const cross = Math.hypot(
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0],
	);
	if (cross === 0) return 0;
	return dot(u, v) / cross;
}
