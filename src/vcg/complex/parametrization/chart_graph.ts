/**
 * Reading a texture atlas as a graph of charts.
 *
 * The whole defragmentation algorithm rests on one structure: the mesh
 * carrying **two** adjacency relations at once.
 *
 * - **3D adjacency** is the surface's own — which triangles actually touch.
 * - **UV adjacency** is the atlas's — which triangles touch *in the texture*.
 *
 * They differ exactly along the seams. An edge that neighbours another face in
 * 3D but is a border in UV is a cut: the two sides of it were placed in
 * different parts of the image. The charts are then the connected components
 * under UV adjacency, and the seam network is the set of edges where the two
 * relations disagree. Everything downstream — which charts could be merged,
 * how long the seam between them is, how much border would disappear — is read
 * off those two arrays.
 *
 * The structure here is deliberately not a `CMeshO`. Cutting along seams
 * renumbers vertices, and the filter has to write its results back to the
 * original mesh's wedges; keeping the atlas as a separate index structure with
 * a face numbering *identical* to the source mesh's makes that mapping the
 * identity rather than a bookkeeping problem.
 */
import { MLException } from "../../../common/utilities/ml_exception.ts";
import type { CMeshO } from "../cmesho.ts";

/** A mesh cut along its UV seams, with both adjacency relations kept. */
export interface AtlasMesh {
	/** Faces after cutting: three vertex indices each. */
	readonly faces: Int32Array;
	readonly faceCount: number;
	readonly vertexCount: number;
	/** Position per cut vertex. */
	readonly positions: Float64Array;
	/** UV per cut vertex — after the cut, UV is per-vertex, not per-wedge. */
	readonly uv: Float64Array;
	/** UV adjacency: the face across each edge, or the face itself on a border. */
	readonly ffFace: Int32Array;
	readonly ffEdge: Int32Array;
	/** 3D adjacency, in the same encoding. */
	readonly ff3DFace: Int32Array;
	readonly ff3DEdge: Int32Array;
	/** The source vertex each cut vertex came from. */
	readonly sourceVertex: Int32Array;
}

const key = (a: number, b: number): string => (a < b ? `${a}:${b}` : `${b}:${a}`);

/** Builds adjacency over a face list, matching edges by their vertex pair. */
function adjacency(
	faceCount: number,
	vertexOf: (f: number, k: number) => number,
): { face: Int32Array; edge: Int32Array; nonManifold: number } {
	const face = new Int32Array(3 * faceCount);
	const edge = new Int32Array(3 * faceCount);
	// A border is encoded as VCGLib does it: pointing at itself.
	for (let f = 0; f < faceCount; f++) {
		for (let k = 0; k < 3; k++) {
			face[3 * f + k] = f;
			edge[3 * f + k] = k;
		}
	}

	const seen = new Map<string, [number, number]>();
	let nonManifold = 0;
	for (let f = 0; f < faceCount; f++) {
		for (let k = 0; k < 3; k++) {
			const a = vertexOf(f, k);
			const b = vertexOf(f, (k + 1) % 3);
			const id = key(a, b);
			const previous = seen.get(id);
			if (previous === undefined) {
				seen.set(id, [f, k]);
				continue;
			}
			if (face[3 * previous[0] + previous[1]] !== previous[0]) {
				// A third face on this edge. It gets no neighbour rather than
				// displacing one of the two already paired.
				nonManifold++;
				continue;
			}
			face[3 * f + k] = previous[0];
			edge[3 * f + k] = previous[1];
			face[3 * previous[0] + previous[1]] = f;
			edge[3 * previous[0] + previous[1]] = k;
		}
	}
	return { face, edge, nonManifold };
}

/**
 * Cuts a mesh along its UV seams and records both adjacencies.
 *
 * Two wedges at the same vertex with different UVs become two vertices. The
 * face numbering is untouched, so face `i` here is face `i` of `m`.
 */
export function buildAtlasMesh(m: CMeshO): AtlasMesh {
	const wt = m.wedgeTexCoord;
	if (wt === null) throw new MLException("the mesh has no per-wedge texture coordinates");

	// Compact face numbering over the live faces, and the cut vertices with it.
	const liveFaces: number[] = [];
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) liveFaces.push(f);
	const faceCount = liveFaces.length;

	const faces = new Int32Array(3 * faceCount);
	const originalCorner = new Int32Array(3 * faceCount);
	const slots = new Map<string, number>();
	const positions: number[] = [];
	const uvs: number[] = [];
	const sourceVertex: number[] = [];

	liveFaces.forEach((f, index) => {
		for (let k = 0; k < 3; k++) {
			const v = m.fv(f, k);
			const u = wt[6 * f + 2 * k];
			const w = wt[6 * f + 2 * k + 1];
			// One cut vertex per (source vertex, UV) pair: exactly the condition
			// under which two wedges may stay joined.
			const id = `${v}|${u}|${w}`;
			let slot = slots.get(id);
			if (slot === undefined) {
				slot = positions.length / 3;
				slots.set(id, slot);
				positions.push(m.vx(v), m.vy(v), m.vz(v));
				uvs.push(u, w);
				sourceVertex.push(v);
			}
			faces[3 * index + k] = slot;
			originalCorner[3 * index + k] = v;
		}
	});

	const uvAdj = adjacency(faceCount, (f, k) => faces[3 * f + k]);
	const threeD = adjacency(faceCount, (f, k) => originalCorner[3 * f + k]);

	return {
		faces,
		faceCount,
		vertexCount: positions.length / 3,
		positions: Float64Array.from(positions),
		uv: Float64Array.from(uvs),
		ffFace: uvAdj.face,
		ffEdge: uvAdj.edge,
		ff3DFace: threeD.face,
		ff3DEdge: threeD.edge,
		sourceVertex: Int32Array.from(sourceVertex),
	};
}

/** True when edge `k` of face `f` has no neighbour in the given relation. */
export function isBorder(face: Int32Array, f: number, k: number): boolean {
	return face[3 * f + k] === f;
}

/**
 * True when this edge is a seam: joined on the surface, cut in the atlas.
 *
 * This is the single predicate the rest of the algorithm is built on.
 */
export function isSeamEdge(am: AtlasMesh, f: number, k: number): boolean {
	return isBorder(am.ffFace, f, k) && !isBorder(am.ff3DFace, f, k);
}

/** A connected component of the atlas: one island of the texture. */
export interface Chart {
	readonly id: number;
	readonly faces: number[];
	/** The charts this one shares a seam with. */
	readonly adjacent: Set<number>;
}

export interface ChartGraph {
	readonly charts: Map<number, Chart>;
	/** The chart each face belongs to. */
	readonly chartOf: Int32Array;
}

/**
 * The charts of an atlas, and which of them touch.
 *
 * Two charts are adjacent when some edge is a border between them in UV but
 * not in 3D — that is, when there is a seam to consider removing.
 */
export function computeChartGraph(am: AtlasMesh): ChartGraph {
	const chartOf = new Int32Array(am.faceCount).fill(-1);
	const charts = new Map<number, Chart>();

	let next = 0;
	for (let start = 0; start < am.faceCount; start++) {
		if (chartOf[start] >= 0) continue;
		const id = next++;
		const faces: number[] = [];
		const stack = [start];
		chartOf[start] = id;
		while (stack.length > 0) {
			const f = stack.pop() as number;
			faces.push(f);
			for (let k = 0; k < 3; k++) {
				const g = am.ffFace[3 * f + k];
				if (g === f || chartOf[g] >= 0) continue;
				chartOf[g] = id;
				stack.push(g);
			}
		}
		charts.set(id, { id, faces, adjacent: new Set() });
	}

	for (let f = 0; f < am.faceCount; f++) {
		for (let k = 0; k < 3; k++) {
			if (!isSeamEdge(am, f, k)) continue;
			const other = chartOf[am.ff3DFace[3 * f + k]];
			if (other === chartOf[f]) continue;
			(charts.get(chartOf[f]) as Chart).adjacent.add(other);
			(charts.get(other) as Chart).adjacent.add(chartOf[f]);
		}
	}

	return { charts, chartOf };
}

/** What the driver needs to know about a chart to rank and check merges. */
export interface ChartMeasures {
	readonly area3D: number;
	readonly areaUV: number;
	/** Signed, so a chart mirrored in the atlas can be spotted. */
	readonly signedAreaUV: number;
	/** The length of the chart's outline, in UV. */
	readonly borderUV: number;
	/** The same outline, measured on the surface. */
	readonly border3D: number;
	readonly uvBox: { minU: number; minV: number; maxU: number; maxV: number };
	/** True when the parametrization reverses orientation over most of the chart. */
	readonly flipped: boolean;
}

export function measureChart(am: AtlasMesh, chart: Chart): ChartMeasures {
	let area3D = 0;
	let signedAreaUV = 0;
	let areaUV = 0;
	let borderUV = 0;
	let border3D = 0;
	let minU = Number.POSITIVE_INFINITY;
	let minV = Number.POSITIVE_INFINITY;
	let maxU = Number.NEGATIVE_INFINITY;
	let maxV = Number.NEGATIVE_INFINITY;

	for (const f of chart.faces) {
		const v = [0, 1, 2].map((k) => am.faces[3 * f + k]);
		const p = v.map((i) => [am.positions[3 * i], am.positions[3 * i + 1], am.positions[3 * i + 2]]);
		const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
		const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
		area3D +=
			Math.hypot(
				e1[1] * e2[2] - e1[2] * e2[1],
				e1[2] * e2[0] - e1[0] * e2[2],
				e1[0] * e2[1] - e1[1] * e2[0],
			) / 2;

		const t = v.map((i) => [am.uv[2 * i], am.uv[2 * i + 1]]);
		const signed =
			((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1])) / 2;
		signedAreaUV += signed;
		areaUV += Math.abs(signed);
		for (const [u, w] of t) {
			if (u < minU) minU = u;
			if (w < minV) minV = w;
			if (u > maxU) maxU = u;
			if (w > maxV) maxV = w;
		}

		for (let k = 0; k < 3; k++) {
			if (!isBorder(am.ffFace, f, k)) continue;
			const a = am.faces[3 * f + k];
			const b = am.faces[3 * f + ((k + 1) % 3)];
			borderUV += Math.hypot(am.uv[2 * a] - am.uv[2 * b], am.uv[2 * a + 1] - am.uv[2 * b + 1]);
			border3D += Math.hypot(
				am.positions[3 * a] - am.positions[3 * b],
				am.positions[3 * a + 1] - am.positions[3 * b + 1],
				am.positions[3 * a + 2] - am.positions[3 * b + 2],
			);
		}
	}

	return {
		area3D,
		areaUV,
		signedAreaUV,
		borderUV,
		border3D,
		uvBox: { minU, minV, maxU, maxV },
		flipped: signedAreaUV < 0,
	};
}

/** The total length of every chart outline in the atlas. */
export function totalBorderUV(am: AtlasMesh, graph: ChartGraph): number {
	let total = 0;
	for (const chart of graph.charts.values()) total += measureChart(am, chart).borderUV;
	return total;
}

export const ChartGraphOps = {
	buildAtlasMesh,
	computeChartGraph,
	measureChart,
	totalBorderUV,
	isSeamEdge,
	isBorder,
} as const;
