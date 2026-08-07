/**
 * Marching tetrahedra over a regular grid.
 *
 * Each cell is cut into six tetrahedra around one diagonal, and each
 * tetrahedron contributes at most two triangles. It is the plainer cousin of
 * marching cubes: the case table has four entries instead of 256, and because
 * neighbouring cells share the diagonal the output is watertight by
 * construction — no ambiguous-face resolution and no cracks.
 *
 * The price is roughly twice the triangles for the same grid, which a
 * decimation pass takes back if it matters.
 */

import { Allocator } from "../allocator.ts";
import { Clean } from "../clean.ts";
import { CMeshO } from "../cmesho.ts";
import { UpdateNormal } from "../update/normal.ts";
import { UpdateTopology } from "../update/topology.ts";

/** The 6 tetrahedra of a cube, all sharing the 0-7 diagonal so faces stay matched. */
const TETRAHEDRA: ReadonlyArray<readonly [number, number, number, number]> = [
	[0, 1, 3, 7],
	[0, 3, 2, 7],
	[0, 2, 6, 7],
	[0, 6, 4, 7],
	[0, 4, 5, 7],
	[0, 5, 1, 7],
];

/** Corner offsets of a cell, indexed the way {@link TETRAHEDRA} expects. */
const CORNERS: ReadonlyArray<readonly [number, number, number]> = [
	[0, 0, 0],
	[1, 0, 0],
	[0, 1, 0],
	[1, 1, 0],
	[0, 0, 1],
	[1, 0, 1],
	[0, 1, 1],
	[1, 1, 1],
];

export function marchingTetrahedra(
	values: Float64Array,
	counts: readonly number[],
	coord: (axis: number, i: number) => number,
	index: (i: number, j: number, k: number) => number,
): CMeshO {
	const px: number[] = [];
	const py: number[] = [];
	const pz: number[] = [];
	const faces: number[] = [];
	const onEdge = new Map<number, number>();
	const total = values.length;

	const cut = (a: number, b: number, ai: readonly number[], bi: readonly number[]): number => {
		const key = a < b ? a * total + b : b * total + a;
		const seen = onEdge.get(key);
		if (seen !== undefined) return seen;
		const va = values[a];
		const vb = values[b];
		const gap = vb - va;
		const t = gap === 0 ? 0.5 : -va / gap;
		const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
		const p = [0, 1, 2].map((axis) => {
			const from = coord(axis, ai[axis]);
			return from + (coord(axis, bi[axis]) - from) * clamped;
		});
		const made = px.length;
		px.push(p[0]);
		py.push(p[1]);
		pz.push(p[2]);
		onEdge.set(key, made);
		return made;
	};

	const corner = new Int32Array(8);
	const cornerAt: Array<readonly number[]> = new Array(8);
	for (let k = 0; k + 1 < counts[2]; k++) {
		for (let j = 0; j + 1 < counts[1]; j++) {
			for (let i = 0; i + 1 < counts[0]; i++) {
				for (let c = 0; c < 8; c++) {
					const o = CORNERS[c];
					cornerAt[c] = [i + o[0], j + o[1], k + o[2]];
					corner[c] = index(i + o[0], j + o[1], k + o[2]);
				}
				for (const tet of TETRAHEDRA) {
					const inside: number[] = [];
					const outside: number[] = [];
					for (const c of tet) (values[corner[c]] < 0 ? inside : outside).push(c);
					if (inside.length === 0 || inside.length === 4) continue;
					const at = (c: number) => cornerAt[c];
					const edge = (a: number, b: number) => cut(corner[a], corner[b], at(a), at(b));
					if (inside.length === 1) {
						faces.push(
							edge(inside[0], outside[0]),
							edge(inside[0], outside[1]),
							edge(inside[0], outside[2]),
						);
					} else if (outside.length === 1) {
						faces.push(
							edge(outside[0], inside[0]),
							edge(outside[0], inside[1]),
							edge(outside[0], inside[2]),
						);
					} else {
						const quad = [
							edge(inside[0], outside[0]),
							edge(inside[0], outside[1]),
							edge(inside[1], outside[1]),
							edge(inside[1], outside[0]),
						];
						faces.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
					}
				}
			}
		}
	}

	const out = new CMeshO();
	if (px.length === 0) return out;
	const firstVert = Allocator.addVertices(out, px.length);
	for (let v = 0; v < px.length; v++) out.setVert(firstVert + v, px[v], py[v], pz[v]);
	const faceCount = faces.length / 3;
	if (faceCount > 0) {
		const firstFace = Allocator.addFaces(out, faceCount);
		for (let f = 0; f < faceCount; f++) {
			out.setFace(firstFace + f, faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]);
		}
	}
	Clean.removeDegenerateFace(out);
	Clean.removeUnreferencedVertex(out);
	Allocator.compactEveryVector(out);
	// Marching tetrahedra winds each triangle from whichever corners fell
	// inside, so the sheet is only consistent after a propagation pass.
	UpdateTopology.faceFace(out);
	Clean.orientCoherentlyMesh(out);
	Clean.flipNormalOutside(out);
	UpdateTopology.faceFace(out);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(out);
	return out;
}
