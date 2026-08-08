/**
 * The seam network of a texture atlas.
 *
 * A seam edge is one the surface joins and the atlas cuts. Individually they
 * are not useful: the driver does not remove one edge at a time, it removes a
 * whole *seam* — a maximal run of seam edges between the same two charts,
 * running from one junction to the next. Those are what get costed, queued and
 * merged.
 *
 * So the network is built in three steps:
 *
 * 1. **The seam mesh** — a graph whose nodes are 3D positions and whose arcs
 *    are the seam edges, each remembering the two faces it separates. Two seam
 *    edges meeting at the same *point on the surface* are joined here even
 *    though the cut mesh gave them different vertices, which is the whole point
 *    of building it: the atlas cut the network apart and this puts it back.
 * 2. **Seams** — connected runs through that graph, stopping wherever the path
 *    is not simply two-in-two-out or where the pair of charts changes. Those
 *    stopping points are the seam's endpoints.
 * 3. **Clusters** — seams grouped by which pair of charts they separate, since
 *    two charts may meet along several disconnected runs and merging them deals
 *    with all of those at once.
 */
import type { AtlasMesh, ChartGraph } from "./chart_graph.ts";
import { isSeamEdge } from "./chart_graph.ts";

/** One seam edge: the two half-edges it separates, and its two endpoints. */
export interface SeamEdge {
	/** The face and edge on one side, and on the other. */
	readonly fa: number;
	readonly ea: number;
	readonly fb: number;
	readonly eb: number;
	/** Nodes of the seam graph, identified by position. */
	readonly v0: number;
	readonly v1: number;
}

/** The seam edges of an atlas as a graph over 3D positions. */
export interface SeamMesh {
	readonly edges: SeamEdge[];
	/** Position per seam node, three numbers each. */
	readonly positions: Float64Array;
	/** The edges meeting at each node. */
	readonly star: number[][];
}

/**
 * Builds the seam graph.
 *
 * Nodes are quantised positions rather than vertex indices: the two sides of a
 * seam have different vertices in the cut mesh but the same point on the
 * surface, and it is the surface that decides whether two seam edges are
 * connected.
 */
export function buildSeamMesh(am: AtlasMesh, graph: ChartGraph): SeamMesh {
	const nodeOf = new Map<string, number>();
	const positions: number[] = [];
	const star: number[][] = [];

	const node = (v: number): number => {
		const x = am.positions[3 * v];
		const y = am.positions[3 * v + 1];
		const z = am.positions[3 * v + 2];
		// Exact position equality: the two sides of a cut hold literally the same
		// coordinates, because the cut copied them. Rounding here would fuse
		// distinct points that merely sit close together.
		const id = `${x},${y},${z}`;
		let found = nodeOf.get(id);
		if (found === undefined) {
			found = positions.length / 3;
			nodeOf.set(id, found);
			positions.push(x, y, z);
			star.push([]);
		}
		return found;
	};

	const edges: SeamEdge[] = [];
	const done = new Set<string>();
	for (let f = 0; f < am.faceCount; f++) {
		for (let k = 0; k < 3; k++) {
			if (!isSeamEdge(am, f, k)) continue;
			const g = am.ff3DFace[3 * f + k];
			const h = am.ff3DEdge[3 * f + k];
			// Each seam edge is seen from both sides; take it once, from the
			// side with the lower chart id so the two sides are ordered.
			const id = f < g ? `${f}:${k}` : `${g}:${h}`;
			if (done.has(id)) continue;
			done.add(id);

			let [fa, ea, fb, eb] = [f, k, g, h];
			if (graph.chartOf[fa] > graph.chartOf[fb]) [fa, ea, fb, eb] = [fb, eb, fa, ea];

			const v0 = node(am.faces[3 * fa + ea]);
			const v1 = node(am.faces[3 * fa + ((ea + 1) % 3)]);
			const index = edges.length;
			edges.push({ fa, ea, fb, eb, v0, v1 });
			star[v0].push(index);
			star[v1].push(index);
		}
	}

	return { edges, positions: Float64Array.from(positions), star };
}

/** A maximal run of seam edges between one pair of charts. */
export interface Seam {
	/** Edge indices into the seam mesh, in path order. */
	readonly edges: number[];
	/** The seam graph nodes it starts and ends at. */
	readonly endpoints: number[];
}

const chartPairOf = (graph: ChartGraph, edge: SeamEdge): [number, number] => {
	const a = graph.chartOf[edge.fa];
	const b = graph.chartOf[edge.fb];
	return a <= b ? [a, b] : [b, a];
};

const samePair = (x: readonly [number, number], y: readonly [number, number]): boolean =>
	x[0] === y[0] && x[1] === y[1];

/**
 * Splits the seam graph into seams.
 *
 * A run continues through a node only when that node has exactly two seam
 * edges and both separate the same pair of charts. Anywhere else — a junction
 * of three seams, a point where a third chart arrives — the run stops and the
 * node becomes an endpoint. A closed loop with no junction at all gets its
 * starting node recorded twice, so that every seam has two endpoints and the
 * ordering below has somewhere to begin.
 */
export function generateSeams(sm: SeamMesh, graph: ChartGraph): Seam[] {
	const visited = new Uint8Array(sm.edges.length);
	const out: Seam[] = [];

	for (let start = 0; start < sm.edges.length; start++) {
		if (visited[start] === 1) continue;
		const pair = chartPairOf(graph, sm.edges[start]);
		const edges: number[] = [];
		const endpoints: number[] = [];
		visited[start] = 1;
		const stack = [start];

		while (stack.length > 0) {
			const e = stack.pop() as number;
			edges.push(e);
			for (const v of [sm.edges[e].v0, sm.edges[e].v1]) {
				const incident = sm.star[v];
				const uniform = incident.every((other) =>
					samePair(chartPairOf(graph, sm.edges[other]), pair),
				);
				if (incident.length !== 2 || !uniform) {
					endpoints.push(v);
					continue;
				}
				const next = incident[0] === e ? incident[1] : incident[0];
				if (visited[next] === 1) continue;
				visited[next] = 1;
				stack.push(next);
			}
		}

		if (endpoints.length === 0) {
			// A closed loop: it has no junction, so its start stands in for both
			// ends and the sort below has an anchor.
			endpoints.push(sm.edges[start].v0, sm.edges[start].v0);
		}
		out.push({ edges: sortSeam(sm, edges, endpoints[0]), endpoints });
	}

	return out;
}

/**
 * Puts a seam's edges in path order, walking from one endpoint.
 *
 * The order matters downstream: shortening a seam means dropping edges from an
 * end, which is only meaningful if the list is a path.
 */
function sortSeam(sm: SeamMesh, edges: readonly number[], from: number): number[] {
	const remaining = new Set(edges);
	const sorted: number[] = [];
	let cursor = from;

	while (sorted.length < edges.length) {
		let taken = -1;
		for (const e of remaining) {
			if (sm.edges[e].v0 === cursor || sm.edges[e].v1 === cursor) {
				taken = e;
				break;
			}
		}
		// A seam that is not a single path — which the endpoint rule should
		// prevent — is left in the order it was collected rather than dropped.
		if (taken < 0) {
			for (const e of remaining) sorted.push(e);
			return sorted;
		}
		remaining.delete(taken);
		sorted.push(taken);
		cursor = sm.edges[taken].v0 === cursor ? sm.edges[taken].v1 : sm.edges[taken].v0;
	}
	return sorted;
}

/** Every seam between one pair of charts, considered as one move. */
export interface ClusteredSeam {
	readonly charts: readonly [number, number];
	readonly seams: Seam[];
}

/**
 * Groups seams by the pair of charts they separate.
 *
 * A seam whose two sides belong to the *same* chart is left on its own: it is
 * a cut inside one chart, and merging it is a different operation from joining
 * two charts.
 */
export function clusterSeamsByChartPair(
	sm: SeamMesh,
	seams: readonly Seam[],
	graph: ChartGraph,
): ClusteredSeam[] {
	const byPair = new Map<string, ClusteredSeam>();
	const out: ClusteredSeam[] = [];

	for (const seam of seams) {
		const pair = chartPairOf(graph, sm.edges[seam.edges[0]]);
		if (pair[0] === pair[1]) {
			out.push({ charts: pair, seams: [seam] });
			continue;
		}
		const id = `${pair[0]}:${pair[1]}`;
		let cluster = byPair.get(id);
		if (cluster === undefined) {
			cluster = { charts: pair, seams: [] };
			byPair.set(id, cluster);
			out.push(cluster);
		}
		cluster.seams.push(seam);
	}
	return out;
}

/** The length of a seam, measured on the surface. */
export function seamLength3D(sm: SeamMesh, seam: Seam): number {
	let total = 0;
	for (const e of seam.edges) {
		const { v0, v1 } = sm.edges[e];
		total += Math.hypot(
			sm.positions[3 * v0] - sm.positions[3 * v1],
			sm.positions[3 * v0 + 1] - sm.positions[3 * v1 + 1],
			sm.positions[3 * v0 + 2] - sm.positions[3 * v1 + 2],
		);
	}
	return total;
}

export function clusterLength3D(sm: SeamMesh, cluster: ClusteredSeam): number {
	let total = 0;
	for (const seam of cluster.seams) total += seamLength3D(sm, seam);
	return total;
}

/**
 * The nodes where a cluster meets the rest of the network.
 *
 * A node listed by exactly one of the cluster's seams is where the cluster
 * ends; a node listed twice is interior to it, an artefact of the run having
 * been split there. Only the former are the cluster's own endpoints.
 */
export function clusterEndpoints(cluster: ClusteredSeam): number[] {
	const count = new Map<number, number>();
	for (const seam of cluster.seams) {
		for (const v of seam.endpoints) count.set(v, (count.get(v) ?? 0) + 1);
	}
	return [...count.entries()].filter(([, n]) => n === 1).map(([v]) => v);
}

/**
 * The two sides' UV coordinates along a cluster, paired up.
 *
 * This is what the matching fit consumes: for each seam vertex, where chart `a`
 * put it and where chart `b` did. The pairing follows the shared edge, and each
 * pair of positions is taken once however many edges meet there.
 */
export function extractSeamUV(
	am: AtlasMesh,
	sm: SeamMesh,
	cluster: ClusteredSeam,
	graph: ChartGraph,
	firstChart: number,
): { a: Float64Array; b: Float64Array } {
	const a: number[] = [];
	const b: number[] = [];
	const seen = new Set<number>();

	for (const seam of cluster.seams) {
		for (const index of seam.edges) {
			const edge = sm.edges[index];
			let [fa, ea, fb, eb] = [edge.fa, edge.ea, edge.fb, edge.eb];
			if (graph.chartOf[fa] !== firstChart) [fa, ea, fb, eb] = [fb, eb, fa, ea];

			// The shared edge runs one way in one face and the other way in its
			// neighbour, so corner 0 on one side matches corner 1 on the other.
			for (const [ka, kb] of [
				[0, 1],
				[1, 0],
			] as const) {
				const va = am.faces[3 * fa + ((ea + ka) % 3)];
				const vb = am.faces[3 * fb + ((eb + kb) % 3)];
				if (seen.has(va) && seen.has(vb)) continue;
				seen.add(va);
				seen.add(vb);
				a.push(am.uv[2 * va], am.uv[2 * va + 1]);
				b.push(am.uv[2 * vb], am.uv[2 * vb + 1]);
			}
		}
	}
	return { a: Float64Array.from(a), b: Float64Array.from(b) };
}

export const Seams = {
	buildSeamMesh,
	generateSeams,
	clusterSeamsByChartPair,
	seamLength3D,
	clusterLength3D,
	clusterEndpoints,
	extractSeamUV,
} as const;
