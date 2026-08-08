/**
 * The greedy driver: merging charts until no merge is worth making.
 *
 * Every seam in the atlas is a candidate move. Removing one means placing the
 * two charts against each other, welding the seam, and relaxing the
 * neighbourhood so the join is not a crease — and then deciding whether the
 * result is better than what was there. The driver keeps the candidates in a
 * priority queue, takes the cheapest, and either commits it or puts it back
 * with a penalty.
 *
 * The cost, following the paper, is not the seam's length. It is
 *
 * ```
 * cost = matchingError · min(borderUV_a / seamUV_a, borderUV_b / seamUV_b)^expb
 *        · min(areaUV_a, areaUV_b)
 * ```
 *
 * — how badly the two sides disagree about where the seam is, scaled by how
 * little of each chart's outline the seam accounts for, scaled by the smaller
 * chart's area. A seam that two charts already agree on costs nearly nothing; a
 * seam that is most of a small chart's outline is cheap to remove because
 * removing it eliminates most of that outline; a seam that is a sliver of two
 * large charts is expensive because it buys almost nothing.
 *
 * **The structural divergence from upstream.** MeshLab performs each move
 * destructively on the mesh — moving vertex references, rewiring face-face and
 * vertex-face topology incrementally — and unwinds all of it when a move is
 * rejected, through `RejectMove` and `RestoreChartAttributes`. That is a large
 * amount of machinery whose only purpose is speed, and its failure mode is a
 * partially-unwound atlas that still looks valid. Here a move is computed
 * against a copy of the affected coordinates and committed only if it passes,
 * so a rejected move provably leaves nothing behind. The price is recomputing
 * what upstream updates in place.
 *
 * **The second divergence: what gets to move.** Upstream grows the optimization
 * area by an offset threshold derived from how far each welded vertex had to
 * travel. Here it is a fixed number of face rings around the seam. The rule is
 * cruder, but it is a rule the reader can predict, and the thing that actually
 * matters — that everything outside the area stays pinned, so the rest of the
 * atlas cannot be disturbed — is identical.
 */

import { crossIntersections, type Segment2, selfIntersections } from "../../space/intersection2.ts";
import { arap2D, arapEnergy, type TargetShapes } from "./arap2d.ts";
import { type AtlasMesh, type ChartGraph, computeChartGraph, isBorder } from "./chart_graph.ts";
import {
	applyMatching,
	IDENTITY_MATCHING,
	type MatchingTransform,
	matchingErrorTotal,
	matchRigid,
} from "./matching2.ts";
import {
	buildSeamMesh,
	type ClusteredSeam,
	clusterSeamsByChartPair,
	generateSeams,
	type SeamMesh,
} from "./seams.ts";

export interface DefragParameters {
	/** How far the two sides of a seam may disagree, relative to its length. */
	readonly matchingThreshold?: number;
	/** A seam accounting for less of a chart's outline than this is skipped. */
	readonly boundaryTolerance?: number;
	/** The most distortion one relaxed neighbourhood may end up with. */
	readonly distortionTolerance?: number;
	/** The most distortion the whole atlas may end up with. */
	readonly globalDistortionThreshold?: number;
	/** Stop once the atlas outline has shrunk to this fraction of its original. */
	readonly uvBorderLengthReduction?: number;
	/** The exponent on the seam-to-outline ratio in the cost. */
	readonly expb?: number;
	/** Face rings around a seam that are free to move. */
	readonly optimizationRings?: number;
	readonly arapIterations?: number;
	/** Give up after this many moves, accepted or not. 0 means no limit. */
	readonly maxMoves?: number;
}

/** Why a move was not made. */
export type RejectReason =
	| "zero-area"
	| "unfeasible-boundary"
	| "unfeasible-matching"
	| "global-overlap-before"
	| "global-overlap-after"
	| "local-overlap"
	| "distortion-local"
	| "distortion-global"
	| "numerical-error";

export interface DefragResult {
	readonly chartsBefore: number;
	readonly chartsAfter: number;
	readonly merges: number;
	readonly rejected: Record<RejectReason, number>;
	readonly borderUVBefore: number;
	readonly borderUVAfter: number;
	readonly seamEdgesRemoved: number;
	readonly iterations: number;
	readonly stopped: "queue-empty" | "border-target" | "move-limit";
	/**
	 * The final UVs, per atlas vertex.
	 *
	 * Welded vertices all carry their group's coordinates, so this can be read
	 * straight off any vertex index without consulting {@link vertexRep} first.
	 */
	readonly uv: Float64Array;
	/**
	 * The surviving vertex of each weld group, per atlas vertex.
	 *
	 * A merge fuses the two sides of a seam, so vertices that were distinct in
	 * the cut atlas become one. A caller writing coordinates back to a mesh does
	 * not need this — {@link uv} is already normalised — but a caller rebuilding
	 * the mesh's topology does.
	 */
	readonly vertexRep: Int32Array;
	/** Which chart each face ended up in. */
	readonly chartOf: Int32Array;
}

const noRejections = (): Record<RejectReason, number> => ({
	"zero-area": 0,
	"unfeasible-boundary": 0,
	"unfeasible-matching": 0,
	"global-overlap-before": 0,
	"global-overlap-after": 0,
	"local-overlap": 0,
	"distortion-local": 0,
	"distortion-global": 0,
	"numerical-error": 0,
});

/** A candidate move, and what it would cost. */
interface Candidate {
	readonly cluster: ClusteredSeam;
	/** The charts as they stand now, which merging renames. */
	charts: [number, number];
	cost: number;
	matching: MatchingTransform;
	reason: RejectReason | null;
	penalty: number;
	/** Bumped whenever the cost is recomputed, to spot stale queue entries. */
	version: number;
}

/**
 * Merges charts across the atlas until nothing is worth merging.
 *
 * `am` is not modified; the new coordinates come back in the result.
 */
export function defragmentAtlas(am: AtlasMesh, params: DefragParameters = {}): DefragResult {
	const matchingThreshold = params.matchingThreshold ?? 2;
	const boundaryTolerance = params.boundaryTolerance ?? 0.2;
	const distortionTolerance = params.distortionTolerance ?? 0.5;
	const globalDistortionThreshold = params.globalDistortionThreshold ?? 0.025;
	const borderReduction = params.uvBorderLengthReduction ?? 0;
	const expb = params.expb ?? 1;
	const rings = params.optimizationRings ?? 3;
	const arapIterations = params.arapIterations ?? 20;
	const maxMoves = params.maxMoves ?? 0;

	const graph = computeChartGraph(am);
	const sm = buildSeamMesh(am, graph);
	const clusters = clusterSeamsByChartPair(sm, generateSeams(sm, graph), graph);

	const state = new AtlasState(am, graph, sm);
	const borderUVBefore = state.totalBorderUV();
	const chartsBefore = state.chartCount();
	const rejected = noRejections();

	// Every cluster is a candidate; the ones between a chart and itself are
	// skipped, because closing them is a different operation (it changes the
	// chart's topology rather than joining two of them).
	const candidates: Candidate[] = [];
	for (const cluster of clusters) {
		if (cluster.charts[0] === cluster.charts[1]) continue;
		const candidate: Candidate = {
			cluster,
			charts: [cluster.charts[0], cluster.charts[1]],
			cost: Number.POSITIVE_INFINITY,
			matching: IDENTITY_MATCHING,
			reason: null,
			penalty: 1,
			version: 0,
		};
		computeCost(state, candidate, { matchingThreshold, boundaryTolerance, expb });
		candidates.push(candidate);
	}

	// A plain array used as a queue: the candidate count is the number of chart
	// adjacencies, and a linear scan for the cheapest costs less than keeping a
	// heap correct through the invalidations a merge causes.
	const live = candidates.filter((c) => Number.isFinite(c.cost));
	for (const c of candidates) if (c.reason !== null) rejected[c.reason]++;

	let merges = 0;
	let seamEdgesRemoved = 0;
	let iterations = 0;
	let stopped: DefragResult["stopped"] = "queue-empty";

	while (live.length > 0) {
		if (maxMoves > 0 && iterations >= maxMoves) {
			stopped = "move-limit";
			break;
		}
		if (borderReduction > 0 && state.totalBorderUV() <= borderReduction * borderUVBefore) {
			stopped = "border-target";
			break;
		}

		let best = 0;
		for (let i = 1; i < live.length; i++) if (live[i].cost < live[best].cost) best = i;
		const candidate = live[best];
		live.splice(best, 1);
		iterations++;

		const outcome = attemptMerge(state, candidate, {
			distortionTolerance,
			globalDistortionThreshold,
			rings,
			arapIterations,
		});

		if (outcome === null) {
			merges++;
			seamEdgesRemoved += countEdges(candidate.cluster);
			// The merge renamed a chart, so every candidate touching either of
			// them is costed against something that no longer exists.
			for (let i = live.length - 1; i >= 0; i--) {
				const other = live[i];
				const a = state.chartRep(other.charts[0]);
				const b = state.chartRep(other.charts[1]);
				if (a !== other.charts[0] || b !== other.charts[1]) {
					if (a === b) {
						// Both sides ended up in the same chart: this seam is now
						// interior and there is nothing left to join.
						live.splice(i, 1);
						continue;
					}
					other.charts = [a, b];
					computeCost(state, other, { matchingThreshold, boundaryTolerance, expb });
					if (!Number.isFinite(other.cost)) {
						if (other.reason !== null) rejected[other.reason]++;
						live.splice(i, 1);
					}
				}
			}
		} else {
			rejected[outcome]++;
			// Penalised rather than discarded: the charts around it will change
			// as other merges land, and the move may become viable.
			candidate.penalty *= 4;
			computeCost(state, candidate, { matchingThreshold, boundaryTolerance, expb });
			if (Number.isFinite(candidate.cost) && candidate.penalty < 1e4) live.push(candidate);
		}
	}

	return {
		chartsBefore,
		chartsAfter: state.chartCount(),
		merges,
		rejected,
		borderUVBefore,
		borderUVAfter: state.totalBorderUV(),
		seamEdgesRemoved,
		iterations,
		stopped,
		uv: state.normalisedUV(),
		vertexRep: state.vertexReps(),
		chartOf: state.faceChart(),
	};
}

const countEdges = (cluster: ClusteredSeam): number =>
	cluster.seams.reduce((sum, seam) => sum + seam.edges.length, 0);

// ------------------------------------------------------------------ state

/**
 * The atlas as it is being changed: coordinates, which charts have merged, and
 * which vertices have been welded together.
 *
 * Both merging relations are union-find, which is what makes an accepted move
 * cheap: a merge is one link, not a relabelling of every face.
 */
class AtlasState {
	readonly uv: Float64Array;
	private readonly chartParent: Int32Array;
	private readonly vertexParent: Int32Array;
	private readonly chartFaces: Map<number, number[]>;
	/** Seam edges that have been welded away and are no longer borders. */
	readonly closed: Uint8Array;

	constructor(
		readonly am: AtlasMesh,
		readonly graph: ChartGraph,
		readonly sm: SeamMesh,
	) {
		this.uv = Float64Array.from(am.uv);
		this.chartParent = new Int32Array(graph.charts.size);
		for (let i = 0; i < this.chartParent.length; i++) this.chartParent[i] = i;
		this.vertexParent = new Int32Array(am.vertexCount);
		for (let i = 0; i < this.vertexParent.length; i++) this.vertexParent[i] = i;
		this.chartFaces = new Map();
		for (const chart of graph.charts.values()) this.chartFaces.set(chart.id, [...chart.faces]);
		this.closed = new Uint8Array(3 * am.faceCount);
	}

	chartRep(id: number): number {
		let root = id;
		while (this.chartParent[root] !== root) root = this.chartParent[root];
		let cursor = id;
		while (this.chartParent[cursor] !== cursor) {
			const next = this.chartParent[cursor];
			this.chartParent[cursor] = root;
			cursor = next;
		}
		return root;
	}

	vertexRep(v: number): number {
		let root = v;
		while (this.vertexParent[root] !== root) root = this.vertexParent[root];
		let cursor = v;
		while (this.vertexParent[cursor] !== cursor) {
			const next = this.vertexParent[cursor];
			this.vertexParent[cursor] = root;
			cursor = next;
		}
		return root;
	}

	weldVertices(a: number, b: number): void {
		const ra = this.vertexRep(a);
		const rb = this.vertexRep(b);
		if (ra !== rb) this.vertexParent[rb] = ra;
	}

	/** Links two charts and returns the surviving id. */
	mergeCharts(a: number, b: number): number {
		const ra = this.chartRep(a);
		const rb = this.chartRep(b);
		if (ra === rb) return ra;
		const [keep, gone] =
			(this.chartFaces.get(ra) as number[]).length >= (this.chartFaces.get(rb) as number[]).length
				? [ra, rb]
				: [rb, ra];
		this.chartParent[gone] = keep;
		(this.chartFaces.get(keep) as number[]).push(...(this.chartFaces.get(gone) as number[]));
		this.chartFaces.delete(gone);
		return keep;
	}

	facesOf(id: number): number[] {
		return this.chartFaces.get(this.chartRep(id)) as number[];
	}

	chartCount(): number {
		return this.chartFaces.size;
	}

	/** Every vertex's coordinates, with welded ones sharing their group's. */
	normalisedUV(): Float64Array {
		const out = Float64Array.from(this.uv);
		for (let v = 0; v < this.am.vertexCount; v++) {
			const root = this.vertexRep(v);
			out[2 * v] = this.uv[2 * root];
			out[2 * v + 1] = this.uv[2 * root + 1];
		}
		return out;
	}

	vertexReps(): Int32Array {
		const out = new Int32Array(this.am.vertexCount);
		for (let v = 0; v < this.am.vertexCount; v++) out[v] = this.vertexRep(v);
		return out;
	}

	faceChart(): Int32Array {
		const out = new Int32Array(this.am.faceCount);
		for (let f = 0; f < this.am.faceCount; f++) out[f] = this.chartRep(this.graph.chartOf[f]);
		return out;
	}

	/** The UV of a corner, after welding. */
	cornerUV(f: number, k: number): [number, number] {
		const v = this.vertexRep(this.am.faces[3 * f + k]);
		return [this.uv[2 * v], this.uv[2 * v + 1]];
	}

	/** True when this edge is still an outline edge of the atlas. */
	isOpen(f: number, k: number): boolean {
		return isBorder(this.am.ffFace, f, k) && this.closed[3 * f + k] === 0;
	}

	totalBorderUV(): number {
		let total = 0;
		for (let f = 0; f < this.am.faceCount; f++) {
			for (let k = 0; k < 3; k++) {
				if (!this.isOpen(f, k)) continue;
				const [x0, y0] = this.cornerUV(f, k);
				const [x1, y1] = this.cornerUV(f, (k + 1) % 3);
				total += Math.hypot(x1 - x0, y1 - y0);
			}
		}
		return total;
	}

	/** Area measures of a chart, over its current coordinates. */
	measure(id: number): { areaUV: number; area3D: number; borderUV: number } {
		let areaUV = 0;
		let area3D = 0;
		let borderUV = 0;
		for (const f of this.facesOf(id)) {
			const t = [0, 1, 2].map((k) => this.cornerUV(f, k));
			areaUV +=
				Math.abs(
					(t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]),
				) / 2;
			const p = [0, 1, 2].map((k) => {
				const v = this.am.faces[3 * f + k];
				return [
					this.am.positions[3 * v],
					this.am.positions[3 * v + 1],
					this.am.positions[3 * v + 2],
				];
			});
			const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
			const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
			area3D +=
				Math.hypot(
					e1[1] * e2[2] - e1[2] * e2[1],
					e1[2] * e2[0] - e1[0] * e2[2],
					e1[0] * e2[1] - e1[1] * e2[0],
				) / 2;
			for (let k = 0; k < 3; k++) {
				if (!this.isOpen(f, k)) continue;
				const [x0, y0] = this.cornerUV(f, k);
				const [x1, y1] = this.cornerUV(f, (k + 1) % 3);
				borderUV += Math.hypot(x1 - x0, y1 - y0);
			}
		}
		return { areaUV, area3D, borderUV };
	}
}

// ------------------------------------------------------------------- cost

/** The two sides' coordinates along a cluster, under the current state. */
function seamCoordinates(
	state: AtlasState,
	cluster: ClusteredSeam,
	firstChart: number,
): { a: Float64Array; b: Float64Array; lengthA: number; lengthB: number } {
	const a: number[] = [];
	const b: number[] = [];
	const seen = new Set<number>();
	let lengthA = 0;
	let lengthB = 0;

	for (const seam of cluster.seams) {
		for (const index of seam.edges) {
			const edge = state.sm.edges[index];
			let [fa, ea, fb, eb] = [edge.fa, edge.ea, edge.fb, edge.eb];
			if (state.chartRep(state.graph.chartOf[fa]) !== firstChart) {
				[fa, ea, fb, eb] = [fb, eb, fa, ea];
			}
			const ua = [0, 1].map((k) => state.cornerUV(fa, (ea + k) % 3));
			const ub = [0, 1].map((k) => state.cornerUV(fb, (eb + k) % 3));
			lengthA += Math.hypot(ua[1][0] - ua[0][0], ua[1][1] - ua[0][1]);
			lengthB += Math.hypot(ub[1][0] - ub[0][0], ub[1][1] - ub[0][1]);

			// The shared edge runs the other way in the neighbouring face.
			for (const [ka, kb] of [
				[0, 1],
				[1, 0],
			] as const) {
				const va = state.vertexRep(state.am.faces[3 * fa + ((ea + ka) % 3)]);
				const vb = state.vertexRep(state.am.faces[3 * fb + ((eb + kb) % 3)]);
				const id = va * state.am.vertexCount + vb;
				if (seen.has(id)) continue;
				seen.add(id);
				a.push(state.uv[2 * va], state.uv[2 * va + 1]);
				b.push(state.uv[2 * vb], state.uv[2 * vb + 1]);
			}
		}
	}
	return { a: Float64Array.from(a), b: Float64Array.from(b), lengthA, lengthB };
}

function computeCost(
	state: AtlasState,
	candidate: Candidate,
	params: { matchingThreshold: number; boundaryTolerance: number; expb: number },
): void {
	candidate.version++;
	candidate.reason = null;
	const [ida, idb] = candidate.charts;
	const a = state.measure(ida);
	const b = state.measure(idb);

	if (a.areaUV === 0 || b.areaUV === 0 || a.area3D === 0 || b.area3D === 0) {
		candidate.cost = Number.POSITIVE_INFINITY;
		candidate.reason = "zero-area";
		return;
	}

	const { a: pa, b: pb, lengthA, lengthB } = seamCoordinates(state, candidate.cluster, ida);
	if (pa.length < 4) {
		candidate.cost = Number.POSITIVE_INFINITY;
		candidate.reason = "unfeasible-matching";
		return;
	}

	// How much of each chart's outline this seam accounts for. A seam that is a
	// sliver of both outlines buys almost nothing, so it is not attempted.
	const ratio = Math.max(
		a.borderUV > 0 ? lengthA / a.borderUV : 0,
		b.borderUV > 0 ? lengthB / b.borderUV : 0,
	);
	if (ratio < params.boundaryTolerance) {
		candidate.cost = Number.POSITIVE_INFINITY;
		candidate.reason = "unfeasible-boundary";
		return;
	}

	const matching = matchRigid(pa, pb);
	const averageError = matchingErrorTotal(matching, pa, pb) / (pa.length / 2);
	if (averageError > params.matchingThreshold * ((lengthA + lengthB) / 2)) {
		candidate.cost = Number.POSITIVE_INFINITY;
		candidate.reason = "unfeasible-matching";
		return;
	}

	const loss =
		averageError *
		(lengthA > 0 && lengthB > 0
			? Math.min(a.borderUV / lengthA, b.borderUV / lengthB) ** params.expb
			: 1);
	let cost = loss * Math.min(a.areaUV, b.areaUV);
	// A penalised move must not stay free, or it would be retried forever.
	if (cost === 0 && candidate.penalty > 1) cost = 1;

	candidate.matching = matching;
	candidate.cost = cost * candidate.penalty;
}

// ------------------------------------------------------------------ moves

/** Attempts one merge. Returns null on success, or why it was rejected. */
function attemptMerge(
	state: AtlasState,
	candidate: Candidate,
	params: {
		distortionTolerance: number;
		globalDistortionThreshold: number;
		rings: number;
		arapIterations: number;
	},
): RejectReason | null {
	const [ida, idb] = candidate.charts;
	const facesA = state.facesOf(ida);
	const facesB = state.facesOf(idb);
	const region = [...facesA, ...facesB];

	// 1. Place b against a, on a copy. Nothing below touches the live state
	// until the move is known to be good.
	const trial = new Map<number, [number, number]>();
	const uvOf = (v: number): [number, number] => {
		const root = state.vertexRep(v);
		return trial.get(root) ?? [state.uv[2 * root], state.uv[2 * root + 1]];
	};
	const setUV = (v: number, x: number, y: number): void => {
		trial.set(state.vertexRep(v), [x, y]);
	};

	// Each vertex once: a vertex is shared between its incident faces, and
	// transforming it per face would apply the matching several times over.
	const movedVertices = new Set<number>();
	for (const f of facesB) {
		for (let k = 0; k < 3; k++) movedVertices.add(state.vertexRep(state.am.faces[3 * f + k]));
	}
	for (const v of movedVertices) {
		const [x, y] = uvOf(v);
		const moved = applyMatching(candidate.matching, x, y);
		setUV(v, moved[0], moved[1]);
	}

	// 2. Weld the seam: each pair of coincident vertices moves to their midpoint.
	const welds: Array<[number, number]> = [];
	for (const seam of candidate.cluster.seams) {
		for (const index of seam.edges) {
			const edge = state.sm.edges[index];
			for (const [ka, kb] of [
				[0, 1],
				[1, 0],
			] as const) {
				const va = state.vertexRep(state.am.faces[3 * edge.fa + ((edge.ea + ka) % 3)]);
				const vb = state.vertexRep(state.am.faces[3 * edge.fb + ((edge.eb + kb) % 3)]);
				if (va === vb) continue;
				welds.push([va, vb]);
				const [xa, ya] = uvOf(va);
				const [xb, yb] = uvOf(vb);
				const mid: [number, number] = [(xa + xb) / 2, (ya + yb) / 2];
				trial.set(va, mid);
				trial.set(vb, mid);
			}
		}
	}
	if (welds.length === 0) return "unfeasible-matching";

	// 3. The neighbourhood that is allowed to move: the seam's faces and the
	// rings around them. Everything else in both charts stays put.
	const free = ringsAround(state, candidate.cluster, region, params.rings);

	// 4. Relax it, against the shapes the faces had before the move — so the
	// measured distortion is the distortion the move introduced.
	const patch = buildPatch(state, region, free, welds, uvOf);
	const before = arapEnergy(patch.faces, patch.target, patch.uv);
	const result = arap2D(patch.faces, patch.vertexCount, patch.target, patch.uv, patch.pins, {
		maxIterations: params.arapIterations,
	});
	if (result.numericalError) return "numerical-error";
	for (const value of patch.uv) if (!Number.isFinite(value)) return "numerical-error";

	// 5. The checks, cheapest first.
	if (result.finalEnergy > params.distortionTolerance) return "distortion-local";
	if (result.finalEnergy - before.energy > params.globalDistortionThreshold) {
		return "distortion-global";
	}

	// Read the relaxed coordinates back into the trial map.
	for (const [v, slot] of patch.slotOf) {
		trial.set(v, [patch.uv[2 * slot], patch.uv[2 * slot + 1]]);
	}

	const foldedBefore = foldedRatio(state, free, (v) => {
		const root = state.vertexRep(v);
		return [state.uv[2 * root], state.uv[2 * root + 1]];
	});
	const foldedAfter = foldedRatio(state, free, uvOf);
	if (foldedAfter > foldedBefore + 1e-12) return "local-overlap";

	if (overlapsAfterMove(state, region, free, uvOf)) return "global-overlap-after";

	// 6. Commit.
	for (const [v, position] of trial) {
		state.uv[2 * v] = position[0];
		state.uv[2 * v + 1] = position[1];
	}
	for (const [va, vb] of welds) state.weldVertices(va, vb);
	for (const seam of candidate.cluster.seams) {
		for (const index of seam.edges) {
			const edge = state.sm.edges[index];
			state.closed[3 * edge.fa + edge.ea] = 1;
			state.closed[3 * edge.fb + edge.eb] = 1;
		}
	}
	state.mergeCharts(ida, idb);
	// The weld may have made two vertices one after their coordinates were
	// written, so the surviving representative takes the welded position.
	for (const [va] of welds) {
		const root = state.vertexRep(va);
		const position = trial.get(va);
		if (position !== undefined) {
			state.uv[2 * root] = position[0];
			state.uv[2 * root + 1] = position[1];
		}
	}
	return null;
}

/** The faces within `rings` steps of the cluster's seam, inside `region`. */
function ringsAround(
	state: AtlasState,
	cluster: ClusteredSeam,
	region: readonly number[],
	rings: number,
): Set<number> {
	const inRegion = new Set(region);
	const free = new Set<number>();
	let frontier: number[] = [];
	for (const seam of cluster.seams) {
		for (const index of seam.edges) {
			const edge = state.sm.edges[index];
			for (const f of [edge.fa, edge.fb]) {
				if (!inRegion.has(f) || free.has(f)) continue;
				free.add(f);
				frontier.push(f);
			}
		}
	}
	for (let step = 0; step < rings; step++) {
		const next: number[] = [];
		for (const f of frontier) {
			// The seam's own edges are still borders in the stored adjacency, so
			// the walk crosses them through the seam's other side instead — which
			// is already in the set from the seeding above.
			for (let k = 0; k < 3; k++) {
				const g = state.am.ffFace[3 * f + k];
				if (g === f || free.has(g) || !inRegion.has(g)) continue;
				free.add(g);
				next.push(g);
			}
		}
		frontier = next;
	}
	return free;
}

interface Patch {
	readonly faces: Int32Array;
	readonly vertexCount: number;
	readonly target: TargetShapes;
	readonly uv: Float64Array;
	readonly pins: Map<number, readonly [number, number]>;
	/** Atlas vertex (welded representative) to patch slot. */
	readonly slotOf: Map<number, number>;
}

/**
 * The ARAP problem for one move.
 *
 * The target shape of each face is the UV triangle it had *before* the move, so
 * the energy measures what the move cost rather than what the parametrization
 * cost. Every vertex outside the free set is pinned.
 */
function buildPatch(
	state: AtlasState,
	region: readonly number[],
	free: ReadonlySet<number>,
	welds: ReadonlyArray<readonly [number, number]>,
	uvOf: (v: number) => [number, number],
): Patch {
	// Welded pairs must land in one slot, or ARAP would leave the seam open.
	const weldRoot = new Map<number, number>();
	const findRoot = (v: number): number => {
		let cursor = v;
		while (weldRoot.has(cursor) && weldRoot.get(cursor) !== cursor) {
			cursor = weldRoot.get(cursor) as number;
		}
		return cursor;
	};
	for (const [a, b] of welds) {
		const ra = findRoot(state.vertexRep(a));
		const rb = findRoot(state.vertexRep(b));
		if (ra !== rb) weldRoot.set(rb, ra);
	}

	const slotOf = new Map<number, number>();
	const faces: number[] = [];
	const target: number[] = [];
	const uv: number[] = [];

	const slot = (v: number): number => {
		const root = findRoot(state.vertexRep(v));
		let found = slotOf.get(root);
		if (found === undefined) {
			found = uv.length / 2;
			slotOf.set(root, found);
			const position = uvOf(root);
			uv.push(position[0], position[1]);
		}
		return found;
	};

	for (const f of region) {
		const corners = [0, 1, 2].map((k) => slot(state.am.faces[3 * f + k]));
		faces.push(...corners);
		// The shape before the move, from the live state rather than the trial.
		const t = [0, 1, 2].map((k) => state.cornerUV(f, k));
		target.push(t[1][0] - t[0][0], t[1][1] - t[0][1], t[2][0] - t[0][0], t[2][1] - t[0][1]);
	}

	// Everything not in a free face is pinned where it is.
	const freeVertices = new Set<number>();
	for (const f of free) {
		for (let k = 0; k < 3; k++) freeVertices.add(slot(state.am.faces[3 * f + k]));
	}
	const pins = new Map<number, readonly [number, number]>();
	for (const [, index] of slotOf) {
		if (freeVertices.has(index)) continue;
		pins.set(index, [uv[2 * index], uv[2 * index + 1]]);
	}
	// A patch entirely free has nothing holding it in place; its outer ring is
	// pinned instead so the relaxation cannot drift the whole thing away.
	if (pins.size === 0) {
		for (const f of region) {
			for (let k = 0; k < 3; k++) {
				if (!state.isOpen(f, k)) continue;
				for (const kk of [k, (k + 1) % 3]) {
					const index = slot(state.am.faces[3 * f + kk]);
					pins.set(index, [uv[2 * index], uv[2 * index + 1]]);
				}
			}
		}
	}
	if (pins.size === 0) pins.set(0, [uv[0], uv[1]]);

	// Fold the welded roots back so a caller reading `slotOf` gets every vertex.
	const full = new Map<number, number>();
	for (const f of region) {
		for (let k = 0; k < 3; k++) {
			const v = state.vertexRep(state.am.faces[3 * f + k]);
			full.set(v, slotOf.get(findRoot(v)) as number);
		}
	}

	return {
		faces: Int32Array.from(faces),
		vertexCount: uv.length / 2,
		target: Float64Array.from(target),
		uv: Float64Array.from(uv),
		pins,
		slotOf: full,
	};
}

/** How much of a face set is folded over, as a fraction of its total area. */
function foldedRatio(
	state: AtlasState,
	faces: ReadonlySet<number>,
	uvOf: (v: number) => [number, number],
): number {
	let negative = 0;
	let absolute = 0;
	for (const f of faces) {
		const t = [0, 1, 2].map((k) => uvOf(state.am.faces[3 * f + k]));
		const signed =
			((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])) / 2;
		if (signed < 0) negative += -signed;
		absolute += Math.abs(signed);
	}
	return absolute > 0 ? negative / absolute : 0;
}

/**
 * Whether the relaxed neighbourhood crosses anything it should not.
 *
 * Two things would be wrong: the free area's own outline crossing itself, and
 * it crossing the part of the charts that was pinned. Both mean two pieces of
 * surface now claim the same texels.
 */
function overlapsAfterMove(
	state: AtlasState,
	region: readonly number[],
	free: ReadonlySet<number>,
	uvOf: (v: number) => [number, number],
): boolean {
	const segment = (f: number, k: number): Segment2 => {
		const [x0, y0] = uvOf(state.am.faces[3 * f + k]);
		const [x1, y1] = uvOf(state.am.faces[3 * f + ((k + 1) % 3)]);
		return { x0, y0, x1, y1 };
	};

	// The outline of the free area: its borders, plus where it meets the pinned
	// part of the same charts.
	const freeOutline: Segment2[] = [];
	for (const f of free) {
		for (let k = 0; k < 3; k++) {
			const g = state.am.ffFace[3 * f + k];
			if (g !== f && free.has(g)) continue;
			freeOutline.push(segment(f, k));
		}
	}
	if (freeOutline.length === 0) return false;
	if (selfIntersections(freeOutline).length > 0) return true;

	const fixedOutline: Segment2[] = [];
	for (const f of region) {
		if (free.has(f)) continue;
		for (let k = 0; k < 3; k++) {
			const g = state.am.ffFace[3 * f + k];
			if (g !== f && !free.has(g)) continue;
			fixedOutline.push(segment(f, k));
		}
	}
	if (fixedOutline.length === 0) return false;
	return crossIntersections(freeOutline, fixedOutline).length > 0;
}

export const Defragment = { defragmentAtlas } as const;
