/**
 * Catmull–Clark and Doo–Sabin, the two classical polygon subdivision schemes.
 *
 * Both read the mesh as polygons and write polygons back, so neither touches a
 * triangle; {@link polygon_support} does the translation. They differ in what
 * the new faces correspond to:
 *
 * - **Catmull–Clark** makes one quad per *corner*. Every face contributes a
 *   face point, every edge an edge point, and every original vertex moves; a
 *   quad joins vertex, edge point, face point, edge point. The result is
 *   always pure quads however irregular the input, which is why it is the
 *   scheme every modelling package uses.
 * - **Doo–Sabin** makes one face per original *face*, *edge* and *vertex*
 *   instead. It shrinks each face toward its centroid and fills the gaps, so
 *   an n-valent vertex becomes an n-gon. The result is not pure quads, and the
 *   surface is only C¹ rather than C² — but every new vertex is a simple
 *   weighted average of one face's corners, so it is far cheaper and it
 *   handles arbitrary polygons without special cases.
 */
import type { CMeshO } from "./cmesho.ts";
import { edgeKey, extractPolygons, meshFromPolygons, type Polygon } from "./polygon_support.ts";

const centroidOf = (points: ReadonlyArray<readonly number[]>): number[] => {
	const c = [0, 0, 0];
	for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k];
	return c.map((x) => x / points.length);
};

/** The live vertex positions of `m`, indexed as the mesh indexes them. */
function positionsOf(m: CMeshO): number[][] {
	const out: number[][] = [];
	for (let v = 0; v < m.vertSize; v++) out.push([m.vx(v), m.vy(v), m.vz(v)]);
	return out;
}

/**
 * One step of Catmull–Clark.
 *
 * The three rules, in the order they depend on each other:
 *
 * 1. A **face point** is the centroid of the face's corners.
 * 2. An **edge point** is the average of the edge's two ends and the face
 *    points either side of it — or just the edge's midpoint on a boundary,
 *    where there is only one face and the interior rule would pull the border
 *    inwards.
 * 3. An **original vertex** moves to `(F + 2R + (n−3)P) / n`, where `F` is the
 *    average of the surrounding face points, `R` the average of the
 *    surrounding edge midpoints, and `n` the valence. A boundary vertex
 *    instead uses `(P + R₁ + R₂) / 3` over its two boundary edges, which keeps
 *    the border a curve of its own rather than letting the interior drag it.
 */
export function catmullClark(m: CMeshO): CMeshO {
	const polygons = extractPolygons(m);
	if (polygons.length === 0) return meshFromPolygons([], []);
	const p = positionsOf(m);

	// 1. Face points.
	const facePoint = polygons.map((ring) => centroidOf(ring.map((v) => p[v])));

	// Which faces touch each edge, and which edges and faces touch each vertex.
	const facesOfEdge = new Map<string, number[]>();
	const edgesOfVertex = new Map<number, Set<string>>();
	const facesOfVertex = new Map<number, number[]>();
	polygons.forEach((ring, f) => {
		for (let i = 0; i < ring.length; i++) {
			const a = ring[i];
			const b = ring[(i + 1) % ring.length];
			const key = edgeKey(a, b);
			const list = facesOfEdge.get(key);
			if (list === undefined) facesOfEdge.set(key, [f]);
			else list.push(f);
			for (const v of [a, b]) {
				const set = edgesOfVertex.get(v);
				if (set === undefined) edgesOfVertex.set(v, new Set([key]));
				else set.add(key);
			}
			const fs = facesOfVertex.get(a);
			if (fs === undefined) facesOfVertex.set(a, [f]);
			else fs.push(f);
		}
	});

	const endsOf = (key: string): [number, number] => {
		const [a, b] = key.split("_").map(Number);
		return [a, b];
	};

	// 2. Edge points.
	const edgePoint = new Map<string, number[]>();
	for (const [key, faces] of facesOfEdge) {
		const [a, b] = endsOf(key);
		const mid = centroidOf([p[a], p[b]]);
		// A boundary edge has one face, and the interior rule would drag the
		// border toward the surface's inside.
		edgePoint.set(
			key,
			faces.length < 2 ? mid : centroidOf([p[a], p[b], ...faces.map((f) => facePoint[f])]),
		);
	}

	// 3. Moved originals.
	const moved = new Map<number, number[]>();
	for (const [v, keys] of edgesOfVertex) {
		const boundary = [...keys].filter((key) => (facesOfEdge.get(key) as number[]).length < 2);
		if (boundary.length > 0) {
			// A boundary vertex follows the border curve alone.
			const ends = boundary.map((key) => {
				const [a, b] = endsOf(key);
				return centroidOf([p[a], p[b]]);
			});
			const sum = [0, 1, 2].map((k) => p[v][k] + ends.reduce((s, e) => s + e[k], 0));
			moved.set(
				v,
				sum.map((x) => x / (1 + ends.length)),
			);
			continue;
		}
		const n = keys.size;
		const F = centroidOf((facesOfVertex.get(v) as number[]).map((f) => facePoint[f]));
		const R = centroidOf(
			[...keys].map((key) => {
				const [a, b] = endsOf(key);
				return centroidOf([p[a], p[b]]);
			}),
		);
		moved.set(
			v,
			[0, 1, 2].map((k) => (F[k] + 2 * R[k] + (n - 3) * p[v][k]) / n),
		);
	}

	// Assemble: one quad per corner of every original face.
	const positions: number[][] = [];
	const indexOfMoved = new Map<number, number>();
	const indexOfEdge = new Map<string, number>();
	const indexOfFace = new Map<number, number>();
	const claim = (point: number[]): number => {
		positions.push(point);
		return positions.length - 1;
	};
	for (const [v, point] of moved) indexOfMoved.set(v, claim(point));
	for (const [key, point] of edgePoint) indexOfEdge.set(key, claim(point));
	facePoint.forEach((point, f) => indexOfFace.set(f, claim(point)));

	const out: Polygon[] = [];
	polygons.forEach((ring, f) => {
		for (let i = 0; i < ring.length; i++) {
			const v = ring[i];
			const prev = ring[(i - 1 + ring.length) % ring.length];
			const next = ring[(i + 1) % ring.length];
			out.push([
				indexOfMoved.get(v) as number,
				indexOfEdge.get(edgeKey(v, next)) as number,
				indexOfFace.get(f) as number,
				indexOfEdge.get(edgeKey(prev, v)) as number,
			]);
		}
	});
	return meshFromPolygons(positions, out);
}

/**
 * One step of Doo–Sabin.
 *
 * Every corner of every face spawns a new vertex at
 * `(P + F + E₁ + E₂) / 4` — the corner, the face's centroid, and the midpoints
 * of the two edges at that corner. Then three families of face are built from
 * those points: a shrunk copy of each original face, a quad across each
 * original interior edge, and an n-gon around each interior vertex.
 *
 * Boundary faces and vertices are skipped rather than guessed at: a border has
 * no second face to build the connecting quad from, and inventing one would
 * close a hole that the mesh genuinely has.
 */
export function dooSabin(m: CMeshO): CMeshO {
	const polygons = extractPolygons(m);
	if (polygons.length === 0) return meshFromPolygons([], []);
	const p = positionsOf(m);

	const positions: number[][] = [];
	// Per (face, corner): the index of the new vertex it spawned.
	const spawned: number[][] = polygons.map(() => []);

	polygons.forEach((ring, f) => {
		const centre = centroidOf(ring.map((v) => p[v]));
		for (let i = 0; i < ring.length; i++) {
			const v = ring[i];
			const prev = ring[(i - 1 + ring.length) % ring.length];
			const next = ring[(i + 1) % ring.length];
			const e1 = centroidOf([p[v], p[next]]);
			const e2 = centroidOf([p[prev], p[v]]);
			positions.push(centroidOf([p[v], centre, e1, e2]));
			spawned[f][i] = positions.length - 1;
		}
	});

	const out: Polygon[] = [];
	// 1. The shrunk copy of each original face.
	for (const ring of spawned) out.push([...ring]);

	// Where each (vertex, face) corner is, and which corners each edge has.
	const cornerAt = new Map<string, number>();
	polygons.forEach((ring, f) => {
		ring.forEach((v, i) => cornerAt.set(`${f}_${v}`, spawned[f][i]));
	});
	const facesOfEdge = new Map<string, number[]>();
	polygons.forEach((ring, f) => {
		for (let i = 0; i < ring.length; i++) {
			const key = edgeKey(ring[i], ring[(i + 1) % ring.length]);
			const list = facesOfEdge.get(key);
			if (list === undefined) facesOfEdge.set(key, [f]);
			else list.push(f);
		}
	});

	// 2. A quad across each interior edge.
	for (const [key, faces] of facesOfEdge) {
		if (faces.length !== 2) continue;
		const [a, b] = key.split("_").map(Number);
		const [f, g] = faces;
		out.push([
			cornerAt.get(`${f}_${a}`) as number,
			cornerAt.get(`${f}_${b}`) as number,
			cornerAt.get(`${g}_${b}`) as number,
			cornerAt.get(`${g}_${a}`) as number,
		]);
	}

	// 3. An n-gon around each interior vertex, its corners taken in the order
	// the faces sit around that vertex — which the edges give directly.
	const edgesOfVertex = new Map<number, string[]>();
	for (const key of facesOfEdge.keys()) {
		const [a, b] = key.split("_").map(Number);
		for (const v of [a, b]) {
			const list = edgesOfVertex.get(v);
			if (list === undefined) edgesOfVertex.set(v, [key]);
			else list.push(key);
		}
	}
	for (const [v, keys] of edgesOfVertex) {
		if (keys.some((key) => (facesOfEdge.get(key) as number[]).length !== 2)) continue;
		const fan = faceRingAround(keys, facesOfEdge);
		if (fan === null || fan.length < 3) continue;
		out.push(fan.map((f) => cornerAt.get(`${f}_${v}`) as number));
	}

	return meshFromPolygons(positions, out);
}

/**
 * The faces around `v`, in the order they meet each other.
 *
 * Walks face to face across the shared edges. Returns null when the fan does
 * not close, which is what a non-manifold vertex looks like from here.
 */
function faceRingAround(
	keys: readonly string[],
	facesOfEdge: ReadonlyMap<string, number[]>,
): number[] | null {
	const startKey = keys[0];
	const order: number[] = [];
	let currentKey = startKey;
	let currentFace = (facesOfEdge.get(startKey) as number[])[0];
	const guard = keys.length + 1;

	for (let step = 0; step < guard; step++) {
		order.push(currentFace);
		// The other edge of this face at v.
		const nextKey = keys.find(
			(key) => key !== currentKey && (facesOfEdge.get(key) as number[]).includes(currentFace),
		);
		if (nextKey === undefined) return null;
		const pair = facesOfEdge.get(nextKey) as number[];
		const nextFace = pair[0] === currentFace ? pair[1] : pair[0];
		currentKey = nextKey;
		currentFace = nextFace;
		if (currentKey === startKey) return order.length === keys.length ? order : null;
	}
	return null;
}

export const SubdivisionPoly = { catmullClark, dooSabin } as const;
