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
	orientAgainstField(out, values, counts, coord, index);
	UpdateTopology.faceFace(out);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(out);
	return out;
}

/**
 * Turns each connected component so its normals point towards *increasing*
 * field value — outwards, since the field is negative inside.
 *
 * The obvious alternative, orienting by signed volume, is wrong the moment
 * there is more than one component: a hollow shell's inner surface encloses a
 * cavity, so pointing it "outwards" in the volume sense turns it inside out
 * and the two surfaces stop describing one solid. Reading the field is
 * unambiguous however many pieces there are.
 */
function orientAgainstField(
	out: CMeshO,
	values: Float64Array,
	counts: readonly number[],
	coord: (axis: number, i: number) => number,
	index: (i: number, j: number, k: number) => number,
): void {
	const components = componentsOf(out);
	const step = [0, 1, 2].map((a) => (counts[a] > 1 ? coord(a, 1) - coord(a, 0) : 1));
	const origin = [0, 1, 2].map((a) => coord(a, 0));

	for (const faces of components) {
		// Vote across the component rather than trusting one face: a single
		// triangle can sit where the gradient is numerically zero.
		let agree = 0;
		let disagree = 0;
		for (const f of faces) {
			const p = [0, 0, 0];
			for (let k = 0; k < 3; k++) {
				const v = out.fv(f, k);
				p[0] += out.vx(v) / 3;
				p[1] += out.vy(v) / 3;
				p[2] += out.vz(v) / 3;
			}
			const gradient = sampleGradient(values, counts, origin, step, index, p);
			const n = faceNormal(out, f);
			const dot = n[0] * gradient[0] + n[1] * gradient[1] + n[2] * gradient[2];
			if (dot > 0) agree++;
			else if (dot < 0) disagree++;
		}
		if (disagree <= agree) continue;
		for (const f of faces) out.setFace(f, out.fv(f, 0), out.fv(f, 2), out.fv(f, 1));
	}
}

function componentsOf(out: CMeshO): number[][] {
	const seen = new Uint8Array(out.faceSize);
	const components: number[][] = [];
	for (let start = 0; start < out.faceSize; start++) {
		if (out.isFaceD(start) || seen[start] === 1) continue;
		const component: number[] = [];
		const stack = [start];
		seen[start] = 1;
		while (stack.length > 0) {
			const f = stack.pop() as number;
			component.push(f);
			for (let e = 0; e < 3; e++) {
				if (out.isBorderFF(f, e)) continue;
				const g = out.ffp(f, e);
				if (g < 0 || out.isFaceD(g) || seen[g] === 1) continue;
				seen[g] = 1;
				stack.push(g);
			}
		}
		components.push(component);
	}
	return components;
}

/** The field's gradient at a point, by central differences on the grid. */
function sampleGradient(
	values: Float64Array,
	counts: readonly number[],
	origin: readonly number[],
	step: readonly number[],
	index: (i: number, j: number, k: number) => number,
	p: readonly number[],
): number[] {
	const cell = [0, 1, 2].map((a) =>
		Math.max(1, Math.min(counts[a] - 2, Math.round((p[a] - origin[a]) / step[a]))),
	);
	const at = (di: number, dj: number, dk: number) =>
		values[index(cell[0] + di, cell[1] + dj, cell[2] + dk)];
	return [
		(at(1, 0, 0) - at(-1, 0, 0)) / (2 * step[0]),
		(at(0, 1, 0) - at(0, -1, 0)) / (2 * step[1]),
		(at(0, 0, 1) - at(0, 0, -1)) / (2 * step[2]),
	];
}

function faceNormal(out: CMeshO, f: number): number[] {
	const p = [0, 1, 2].map((k) => {
		const v = out.fv(f, k);
		return [out.vx(v), out.vy(v), out.vz(v)];
	});
	const u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
	const w = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
	return [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
}
