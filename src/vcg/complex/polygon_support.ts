/**
 * `vcg/complex/algorithms/polygon_support.h` — reading a triangle mesh as a
 * polygon mesh and writing one back.
 *
 * Upstream keeps a second mesh type for polygons and converts between the two.
 * Here the polygon *is* the faux-edge representation: a group of triangles
 * joined by hidden edges, whose outer boundary is the polygon's own. So there
 * is no second type, only two functions — one that walks a group's boundary
 * into an ordered ring, and one that fans a ring back into triangles with the
 * interior edges hidden.
 *
 * Both subdivision schemes above this file work the same way: read the polygons
 * out, compute a new set of polygons from them, write those back. Neither
 * touches a triangle directly.
 */
import { Allocator } from "./allocator.ts";
import { extractPolygon } from "./bit_quad.ts";
import { CMeshO } from "./cmesho.ts";
import { fauxBit } from "./flags.ts";
import { faceFace } from "./update/topology.ts";

/** A polygon as an ordered ring of vertex indices, anticlockwise. */
export type Polygon = number[];

/**
 * Every polygon of `m`, as ordered vertex rings.
 *
 * A face with no hidden edges is a triangle and comes back as one; a group
 * joined by hidden edges comes back as the ring around it. Throws when a group
 * has no single closed boundary, which means the faux tagging describes
 * something that is not a polygon.
 */
export function extractPolygons(m: CMeshO): Polygon[] {
	if (m.ffFace === null) faceFace(m);
	const done = new Uint8Array(m.faceSize);
	const group: number[] = [];
	const out: Polygon[] = [];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || done[f] === 1) continue;
		extractPolygon(m, f, group);
		for (const g of group) done[g] = 1;

		// The boundary is every edge of the group that is not hidden. Walking
		// it in order turns the set into a ring.
		const next = new Map<number, number>();
		for (const g of group) {
			for (let e = 0; e < 3; e++) {
				if ((m.faceFlags[g] & fauxBit(e)) !== 0) continue;
				const a = m.fv(g, e);
				const b = m.fv(g, (e + 1) % 3);
				if (next.has(a)) throw new Error("polygon boundary visits a vertex twice");
				next.set(a, b);
			}
		}
		if (next.size === 0) continue;

		const start = next.keys().next().value as number;
		const ring: number[] = [];
		let cursor = start;
		do {
			ring.push(cursor);
			const step = next.get(cursor);
			if (step === undefined) throw new Error("polygon boundary is not closed");
			cursor = step;
		} while (cursor !== start && ring.length <= next.size);
		if (ring.length !== next.size) throw new Error("polygon boundary is not a single ring");
		out.push(ring);
	}
	return out;
}

/**
 * Builds a mesh from polygons, fanning each into triangles.
 *
 * A ring of `n` vertices becomes `n - 2` triangles sharing the ring's first
 * vertex, with the `n - 3` interior diagonals hidden. So a quad becomes two
 * triangles and one hidden edge, exactly the representation the rest of the
 * library expects, and a triangle becomes itself with nothing hidden.
 */
export function meshFromPolygons(
	positions: ReadonlyArray<readonly number[]>,
	polygons: ReadonlyArray<readonly number[]>,
): CMeshO {
	const out = new CMeshO();
	if (positions.length === 0) return out;
	Allocator.addVertices(out, positions.length);
	positions.forEach((p, i) => out.setVert(i, p[0], p[1], p[2]));

	let faceCount = 0;
	for (const ring of polygons) faceCount += Math.max(0, ring.length - 2);
	if (faceCount === 0) return out;
	const first = Allocator.addFaces(out, faceCount);

	let cursor = first;
	for (const ring of polygons) {
		const n = ring.length;
		if (n < 3) continue;
		for (let k = 1; k + 1 < n; k++) {
			const f = cursor++;
			out.setFace(f, ring[0], ring[k], ring[k + 1]);
			// Every fan diagonal except the ring's own two outer edges is
			// interior to the polygon and so hidden. In face (r0, rk, rk+1),
			// edge 0 is r0-rk and edge 2 is rk+1-r0.
			if (k > 1) out.faceFlags[f] |= fauxBit(0);
			if (k + 2 < n) out.faceFlags[f] |= fauxBit(2);
		}
	}
	return out;
}

/** Rings sharing each undirected edge, keyed `min_max`. */
export function edgeKey(a: number, b: number): string {
	return a < b ? `${a}_${b}` : `${b}_${a}`;
}

/**
 * Replaces `target`'s geometry with `source`'s, in place.
 *
 * The subdivision schemes build a whole new mesh rather than editing one, but
 * the filter has to leave the *same layer* holding the result — a caller's
 * reference to `doc.mm().cm` must still be the mesh it gets back.
 */
export function replaceGeometry(target: CMeshO, source: CMeshO): void {
	target.clear();
	if (source.vn === 0) return;
	Allocator.addVertices(target, source.vertSize);
	for (let v = 0; v < source.vertSize; v++) {
		target.setVert(v, source.vx(v), source.vy(v), source.vz(v));
		target.vertFlags[v] = source.vertFlags[v];
	}
	if (source.faceSize > 0) {
		Allocator.addFaces(target, source.faceSize);
		for (let f = 0; f < source.faceSize; f++) {
			target.setFace(f, source.fv(f, 0), source.fv(f, 1), source.fv(f, 2));
			// The faux bits are the whole point: they carry which edges are the
			// polygons' interiors.
			target.faceFlags[f] = source.faceFlags[f];
		}
	}
	target.imark++;
}

export const PolygonSupport = {
	extractPolygons,
	meshFromPolygons,
	edgeKey,
	replaceGeometry,
} as const;
