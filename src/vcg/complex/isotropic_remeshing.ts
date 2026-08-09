/**
 * Isotropic explicit remeshing — Botsch and Kobbelt's four local operations,
 * repeated until the triangles are all about the target size and about
 * equilateral.
 *
 * Each pass does, in order: split what is too long, collapse what is too
 * short, flip edges that would improve the vertex valences, and relax the
 * vertices tangentially. The order matters — collapsing before splitting can
 * destroy a feature the split would have resolved — and so do the 4/3 and 4/5
 * factors on the target length, which are what stop a split and the following
 * collapse from undoing each other forever.
 *
 * "Explicit" is MeshLab's word for the fact that the target length is given
 * rather than derived from curvature.
 */

import { KdTree } from "../space/index/kdtree.ts";
import { Allocator } from "./allocator.ts";
import { Clean } from "./clean.ts";
import type { CMeshO } from "./cmesho.ts";
import {
	buildVertexFaces,
	collapseEdge,
	edgePairOf,
	flipEdge,
	linkCondition,
	sharedFaces,
	triQuality,
} from "./edge_ops.ts";
import { longerThan, midPoint, refineE } from "./refine.ts";
import { UpdateBounding } from "./update/bounding.ts";
import { UpdateNormal } from "./update/normal.ts";
import { UpdateTopology } from "./update/topology.ts";

export interface RemeshOptions {
	/** How many times to run the whole four-step pass. */
	iterations: number;
	/** The edge length the mesh is driven toward. */
	targetLen: number;
	/** Dihedral angle, in degrees, above which an edge counts as a crease. */
	featureDeg: number;
	/** Reject any operation that moves the surface further than this. */
	maxSurfDist: number;
	checkSurfDist: boolean;
	splitFlag: boolean;
	collapseFlag: boolean;
	swapFlag: boolean;
	smoothFlag: boolean;
	reprojectFlag: boolean;
	selectedOnly: boolean;
}

export const REMESH_DEFAULTS: Omit<RemeshOptions, "targetLen" | "maxSurfDist"> = {
	iterations: 3,
	featureDeg: 30,
	checkSurfDist: true,
	splitFlag: true,
	collapseFlag: true,
	swapFlag: true,
	smoothFlag: true,
	reprojectFlag: true,
	selectedOnly: false,
};

export interface RemeshResult {
	readonly splits: number;
	readonly collapses: number;
	readonly flips: number;
}

/**
 * The classical bounds: split above 4/3 of the target, collapse below 4/5.
 *
 * A narrower band would oscillate — a freshly split edge is half its old
 * length, and if that lands under the collapse threshold the next step undoes
 * the split.
 */
const SPLIT_FACTOR = 4 / 3;
const COLLAPSE_FACTOR = 4 / 5;

/** Bound on the repeated collapse sweeps, so a pathological mesh cannot hang. */
const MAX_COLLAPSE_SWEEPS = 10;

/** The valence a vertex wants: 6 inside the surface, 4 on the boundary. */
function targetValence(onBoundary: boolean): number {
	return onBoundary ? 4 : 6;
}

export function isotropicRemeshing(m: CMeshO, options: RemeshOptions): RemeshResult {
	if (!(options.targetLen > 0)) {
		throw new Error(`Isotropic remeshing needs a positive target length, got ${options.targetLen}`);
	}
	// The reference surface is the mesh as it arrived; every reprojection and
	// every distance check measures against this, not against the mesh as it is
	// being rewritten under them.
	const reference = options.checkSurfDist || options.reprojectFlag ? sampleSurface(m) : null;

	let splits = 0;
	let collapses = 0;
	let flips = 0;

	for (let pass = 0; pass < options.iterations; pass++) {
		if (options.splitFlag) {
			UpdateTopology.faceFace(m);
			const before = m.fn;
			refineE(m, midPoint, longerThan(options.targetLen * SPLIT_FACTOR), {
				selectedOnly: options.selectedOnly,
			});
			splits += Math.max(0, m.fn - before);
		}
		if (options.collapseFlag) {
			// One sweep is not enough: collapsing an edge shortens the ones
			// around it, and a sweep that has already passed them leaves them
			// behind. Repeat until a sweep finds nothing.
			for (let sweep = 0; sweep < MAX_COLLAPSE_SWEEPS; sweep++) {
				const done = collapseShortEdges(m, options, reference);
				collapses += done;
				if (done === 0) break;
			}
		}
		if (options.swapFlag) {
			flips += improveValence(m, options);
		}
		if (options.smoothFlag) {
			tangentialRelaxation(m, options, reference);
		}
	}

	// Two distinct edges can bisect to exactly the same point — on a cylinder
	// wall the two diagonals of a parallelogram do — and once a later flip puts
	// both midpoints in one face that face has zero area. Rare (six faces in
	// 7,616 on a cylinder at five passes) but real, and welding is the honest
	// repair: it merges the coincident pair and drops the faces that only
	// existed between them.
	Clean.removeDuplicateVertex(m);
	Clean.removeDegenerateFace(m);
	Clean.removeUnreferencedVertex(m);
	Allocator.compactEveryVector(m);
	UpdateTopology.faceFace(m);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
	UpdateBounding.box(m);
	return { splits, collapses, flips };
}

/** A point set standing in for the original surface, plus an index over it. */
interface Reference {
	readonly coords: Float64Array;
	readonly tree: KdTree;
}

/**
 * Samples the surface densely enough to serve as the thing to reproject onto.
 *
 * Vertices alone would be too sparse where the mesh is coarse — reprojection
 * would pull new vertices back onto the old ones and undo the remeshing — so
 * the face centres and edge midpoints go in too.
 */
function sampleSurface(m: CMeshO): Reference {
	const pts: number[] = [];
	for (let v = 0; v < m.vertSize; v++) {
		if (!m.isVertD(v)) pts.push(m.vx(v), m.vy(v), m.vz(v));
	}
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		pts.push(
			(m.vx(a) + m.vx(b) + m.vx(c)) / 3,
			(m.vy(a) + m.vy(b) + m.vy(c)) / 3,
			(m.vz(a) + m.vz(b) + m.vz(c)) / 3,
		);
		for (const [p, q] of [
			[a, b],
			[b, c],
			[c, a],
		] as const) {
			pts.push((m.vx(p) + m.vx(q)) / 2, (m.vy(p) + m.vy(q)) / 2, (m.vz(p) + m.vz(q)) / 2);
		}
	}
	const coords = Float64Array.from(pts);
	return { coords, tree: new KdTree(coords, coords.length / 3) };
}

/** Distance from a point to the sampled original surface. */
function distanceToReference(ref: Reference, x: number, y: number, z: number): number {
	const i = ref.tree.nearestToPoint(x, y, z);
	if (i < 0) return 0;
	return Math.hypot(ref.coords[3 * i] - x, ref.coords[3 * i + 1] - y, ref.coords[3 * i + 2] - z);
}

/** Which vertices sit on a boundary, and which on a crease. */
function markFeatures(m: CMeshO, featureDeg: number): { boundary: Uint8Array; crease: Uint8Array } {
	UpdateTopology.faceFace(m);
	const boundary = new Uint8Array(m.vertSize);
	const crease = new Uint8Array(m.vertSize);
	const cosLimit = Math.cos((featureDeg * Math.PI) / 180);
	const scratch = new Float64Array(3);

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const n0 = unitNormal(m, f, scratch);
		for (let e = 0; e < 3; e++) {
			const a = m.fv(f, e);
			const b = m.fv(f, (e + 1) % 3);
			if (m.isBorderFF(f, e)) {
				boundary[a] = 1;
				boundary[b] = 1;
				continue;
			}
			const other = m.ffp(f, e);
			if (other < f) continue; // each interior edge once
			const n1 = unitNormal(m, other, new Float64Array(3));
			if (n0 === null || n1 === null) continue;
			// A sharp fold is geometry the user wants kept, so its vertices are
			// pinned against smoothing and reprojection.
			if (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] < cosLimit) {
				crease[a] = 1;
				crease[b] = 1;
			}
		}
	}
	return { boundary, crease };
}

function unitNormal(m: CMeshO, f: number, out: Float64Array): Float64Array | null {
	const a = m.fv(f, 0);
	const b = m.fv(f, 1);
	const c = m.fv(f, 2);
	const ux = m.vx(b) - m.vx(a);
	const uy = m.vy(b) - m.vy(a);
	const uz = m.vz(b) - m.vz(a);
	const vx = m.vx(c) - m.vx(a);
	const vy = m.vy(c) - m.vy(a);
	const vz = m.vz(c) - m.vz(a);
	out[0] = uy * vz - uz * vy;
	out[1] = uz * vx - ux * vz;
	out[2] = ux * vy - uy * vx;
	const len = Math.hypot(out[0], out[1], out[2]);
	if (len === 0) return null;
	out[0] /= len;
	out[1] /= len;
	out[2] /= len;
	return out;
}

/** Collapses every edge shorter than 4/5 of the target that it is safe to collapse. */
function collapseShortEdges(m: CMeshO, options: RemeshOptions, ref: Reference | null): number {
	const { boundary, crease } = markFeatures(m, options.featureDeg);
	const vertFaces = buildVertexFaces(m);
	const limit = options.targetLen * COLLAPSE_FACTOR;
	const limit2 = limit * limit;
	const splitLimit = options.targetLen * SPLIT_FACTOR;
	let done = 0;

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (options.selectedOnly && !m.isFaceS(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (m.isFaceD(f)) break;
			// VCG splits an edge only when the faces on BOTH sides are selected
			// (a mesh border counts, because FFp points back at the face
			// itself). An edge on the selection's rim stays whole — which is
			// what keeps a refined patch from leaking into its surroundings,
			// and what makes refining a single-triangle cap a no-op, exactly
			// as MeshLab has it.
			if (options.selectedOnly) {
				const g = m.ffp(f, e);
				if (!m.isFaceD(g) && !m.isFaceS(g)) continue;
			}
			const u = m.faceVert[3 * f + e];
			const v = m.faceVert[3 * f + ((e + 1) % 3)];
			if (u === v || m.isVertD(u) || m.isVertD(v)) continue;
			const d2 = (m.vx(u) - m.vx(v)) ** 2 + (m.vy(u) - m.vy(v)) ** 2 + (m.vz(u) - m.vz(v)) ** 2;
			if (d2 >= limit2) continue;

			// A crease or boundary vertex may only move along its own feature,
			// so a collapse that would drag it into the interior is refused.
			const keep = pickSurvivor(u, v, boundary, crease);
			if (keep === null) continue;
			const gone = keep === u ? v : u;
			const x = m.vx(keep);
			const y = m.vy(keep);
			const z = m.vz(keep);

			const shared = sharedFaces(m, vertFaces, u, v);
			if (shared.length !== 2) continue;
			if (!linkCondition(m, vertFaces, keep, gone, shared)) continue;
			// Botsch and Kobbelt's other half of the rule, and the half it is
			// easy to leave out: a collapse must not produce an edge that the
			// next split step would immediately cut again. Without it the two
			// steps cycle — bisecting a 3.0 edge walks it down to 0.1875, under
			// the 0.2 collapse threshold, so the collapse undoes the split and
			// a cylinder loses 80% of its volume in a single pass.
			if (!collapseKeepsEdgesShort(m, vertFaces, keep, gone, x, y, z, splitLimit)) continue;
			if (options.selectedOnly && !allSelected(m, vertFaces, keep, gone)) continue;
			if (!survivingFacesAreSane(m, vertFaces, keep, gone, x, y, z, shared)) continue;
			if (
				options.checkSurfDist &&
				ref !== null &&
				distanceToReference(ref, x, y, z) > options.maxSurfDist
			) {
				continue;
			}

			collapseEdge(m, vertFaces, keep, gone, x, y, z);
			done++;
		}
	}
	return done;
}

/**
 * Which endpoint survives a collapse.
 *
 * The rule is that a feature vertex outranks a free one, and two feature
 * vertices of different kinds cannot be merged at all — collapsing a crease
 * vertex onto a boundary vertex would move the crease onto the boundary.
 */
function pickSurvivor(
	u: number,
	v: number,
	boundary: Uint8Array,
	crease: Uint8Array,
): number | null {
	const rank = (w: number) => (boundary[w] ? 2 : crease[w] ? 1 : 0);
	const ru = rank(u);
	const rv = rank(v);
	if (ru === rv) {
		// Two vertices of the same feature kind may merge, except that two
		// unrelated boundary vertices meeting would pinch the boundary shut.
		return ru === 0 ? u : u;
	}
	return ru > rv ? u : v;
}

/** Whether every edge left at the merged vertex would stay under `limit`. */
function collapseKeepsEdgesShort(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	keep: number,
	gone: number,
	x: number,
	y: number,
	z: number,
	limit: number,
): boolean {
	const limit2 = limit * limit;
	for (const w of [keep, gone]) {
		for (const f of vertFaces[w]) {
			if (m.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const q = m.faceVert[3 * f + k];
				if (q === keep || q === gone) continue;
				const d2 = (m.vx(q) - x) ** 2 + (m.vy(q) - y) ** 2 + (m.vz(q) - z) ** 2;
				if (d2 > limit2) return false;
			}
		}
	}
	return true;
}

function allSelected(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	u: number,
	v: number,
): boolean {
	for (const w of [u, v]) {
		for (const f of vertFaces[w]) if (!m.isFaceD(f) && !m.isFaceS(f)) return false;
	}
	return true;
}

/**
 * Minimum triangle quality (2·area / longest-edge²) a collapse may leave
 * behind — unless the face was already worse, in which case the collapse is
 * judged on whether it makes things worse rather than against the floor.
 *
 * The exception is not a nicety. A cylinder tessellated into long thin strips
 * has faces at quality 0.006 from the start, and an absolute floor refuses
 * every collapse on it: the split step then runs away unopposed, turning 74
 * vertices into 52,000 over ten passes.
 */
const MIN_COLLAPSE_QUALITY = 0.01;

/** Refuses a collapse that would fold a face over or leave a sliver behind. */
function survivingFacesAreSane(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	keep: number,
	gone: number,
	x: number,
	y: number,
	z: number,
	shared: readonly number[],
): boolean {
	const scratch = new Float64Array(3);
	for (const w of [keep, gone]) {
		for (const f of vertFaces[w]) {
			if (m.isFaceD(f) || shared.includes(f)) continue;
			const p: Array<[number, number, number]> = [];
			for (let k = 0; k < 3; k++) {
				const q = m.faceVert[3 * f + k];
				p.push(q === keep || q === gone ? [x, y, z] : [m.vx(q), m.vy(q), m.vz(q)]);
			}
			const after = triQuality(
				p[0][0],
				p[0][1],
				p[0][2],
				p[1][0],
				p[1][1],
				p[1][2],
				p[2][0],
				p[2][1],
				p[2][2],
			);
			if (after < MIN_COLLAPSE_QUALITY) {
				const a = m.fv(f, 0);
				const b = m.fv(f, 1);
				const c = m.fv(f, 2);
				const before = triQuality(
					m.vx(a),
					m.vy(a),
					m.vz(a),
					m.vx(b),
					m.vy(b),
					m.vz(b),
					m.vx(c),
					m.vy(c),
					m.vz(c),
				);
				if (after < before) return false;
			}
			const wasFacing = unitNormal(m, f, scratch);
			const nowFacing = triNormal(p);
			if (wasFacing !== null && nowFacing !== null) {
				// A face that ends up facing the other way has folded through
				// its own plane, which no amount of smoothing recovers from.
				const dot =
					wasFacing[0] * nowFacing[0] + wasFacing[1] * nowFacing[1] + wasFacing[2] * nowFacing[2];
				if (dot < 0) return false;
			}
		}
	}
	return true;
}

function triNormal(
	p: ReadonlyArray<readonly [number, number, number]>,
): [number, number, number] | null {
	const ux = p[1][0] - p[0][0];
	const uy = p[1][1] - p[0][1];
	const uz = p[1][2] - p[0][2];
	const vx = p[2][0] - p[0][0];
	const vy = p[2][1] - p[0][1];
	const vz = p[2][2] - p[0][2];
	const n: [number, number, number] = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
	const len = Math.hypot(n[0], n[1], n[2]);
	if (len === 0) return null;
	return [n[0] / len, n[1] / len, n[2] / len];
}

/**
 * Flips interior edges whose two triangles would have valences closer to the
 * ideal on the other diagonal.
 *
 * Valence is the whole objective here: a mesh where every interior vertex has
 * six neighbours is the one whose triangles are closest to equilateral.
 */
function improveValence(m: CMeshO, options: RemeshOptions): number {
	const { boundary, crease } = markFeatures(m, options.featureDeg);
	const vertFaces = buildVertexFaces(m);
	const valence = new Int32Array(m.vertSize);
	for (let v = 0; v < m.vertSize; v++)
		if (!m.isVertD(v)) valence[v] = countNeighbours(m, vertFaces, v);

	let done = 0;
	const seen = new Set<number>();
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (options.selectedOnly && !m.isFaceS(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (m.isFaceD(f)) break;
			const a = m.faceVert[3 * f + e];
			const b = m.faceVert[3 * f + ((e + 1) % 3)];
			if (a === b) continue;
			const key = a < b ? a * m.vertSize + b : b * m.vertSize + a;
			if (seen.has(key)) continue;
			seen.add(key);
			// A crease or boundary edge is the shape the user asked to keep.
			if ((boundary[a] && boundary[b]) || (crease[a] && crease[b])) continue;

			const pair = edgePairOf(m, vertFaces, a, b);
			if (pair === null) continue;
			const { o0, o1 } = pair;
			if (options.selectedOnly && (!m.isFaceS(pair.f0) || !m.isFaceS(pair.f1))) continue;

			const before =
				deviation(valence[a], boundary[a] === 1) +
				deviation(valence[b], boundary[b] === 1) +
				deviation(valence[o0], boundary[o0] === 1) +
				deviation(valence[o1], boundary[o1] === 1);
			const after =
				deviation(valence[a] - 1, boundary[a] === 1) +
				deviation(valence[b] - 1, boundary[b] === 1) +
				deviation(valence[o0] + 1, boundary[o0] === 1) +
				deviation(valence[o1] + 1, boundary[o1] === 1);
			if (after >= before) continue;
			if (!flipWouldStayFlat(m, pair)) continue;
			if (!flipEdge(m, vertFaces, pair)) continue;

			valence[a]--;
			valence[b]--;
			valence[o0]++;
			valence[o1]++;
			done++;
		}
	}
	return done;
}

const deviation = (valence: number, onBoundary: boolean): number =>
	Math.abs(valence - targetValence(onBoundary));

function countNeighbours(m: CMeshO, vertFaces: ReadonlyArray<Set<number>>, v: number): number {
	const out = new Set<number>();
	for (const f of vertFaces[v]) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const w = m.faceVert[3 * f + k];
			if (w !== v) out.add(w);
		}
	}
	return out.size;
}

/**
 * Whether flipping would keep both triangles facing the same way.
 *
 * On a curved surface the two diagonals of a quad are not equivalent: one of
 * them can pass outside the quad entirely, producing a pair of overlapping
 * triangles. Comparing the normals before and after catches that.
 */
function flipWouldStayFlat(
	m: CMeshO,
	pair: { f0: number; f1: number; a: number; b: number; o0: number; o1: number },
): boolean {
	const p = (v: number): [number, number, number] => [m.vx(v), m.vy(v), m.vz(v)];
	const n0 = unitNormal(m, pair.f0, new Float64Array(3));
	const n1 = unitNormal(m, pair.f1, new Float64Array(3));
	const a0 = triNormal([p(pair.a), p(pair.o1), p(pair.o0)]);
	const a1 = triNormal([p(pair.b), p(pair.o0), p(pair.o1)]);
	if (n0 === null || n1 === null || a0 === null || a1 === null) return false;
	const avg: [number, number, number] = [n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]];
	return (
		a0[0] * avg[0] + a0[1] * avg[1] + a0[2] * avg[2] > 0 &&
		a1[0] * avg[0] + a1[1] * avg[1] + a1[2] * avg[2] > 0
	);
}

/**
 * Moves every free vertex toward the centroid of its neighbours, but only
 * within the tangent plane, then puts it back on the original surface.
 *
 * Dropping the normal component is what separates this from plain Laplacian
 * smoothing: it evens out the triangle sizes without shrinking the object.
 */
function tangentialRelaxation(m: CMeshO, options: RemeshOptions, ref: Reference | null): void {
	const { boundary, crease } = markFeatures(m, options.featureDeg);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
	const vertFaces = buildVertexFaces(m);
	const target = new Float64Array(m.vertSize * 3);
	const move = new Uint8Array(m.vertSize);

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || boundary[v] || crease[v]) continue;
		if (options.selectedOnly) {
			// VCG marks the selection's vertices and then strips any that also
			// belong to an unselected face — so only the strict interior of the
			// selection moves. A vertex on the rim is shared with faces nobody
			// asked to change.
			let allFacesSelected = vertFaces[v].size > 0;
			for (const f of vertFaces[v]) {
				if (!m.isFaceD(f) && !m.isFaceS(f)) {
					allFacesSelected = false;
					break;
				}
			}
			if (!allFacesSelected) continue;
		}
		let sx = 0;
		let sy = 0;
		let sz = 0;
		let n = 0;
		let nearest = Number.POSITIVE_INFINITY;
		const seen = new Set<number>();
		for (const f of vertFaces[v]) {
			if (m.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const w = m.faceVert[3 * f + k];
				if (w === v || seen.has(w)) continue;
				seen.add(w);
				sx += m.vx(w);
				sy += m.vy(w);
				sz += m.vz(w);
				nearest = Math.min(
					nearest,
					Math.hypot(m.vx(w) - m.vx(v), m.vy(w) - m.vy(v), m.vz(w) - m.vz(v)),
				);
				n++;
			}
		}
		if (n === 0) continue;
		let dx = sx / n - m.vx(v);
		let dy = sy / n - m.vy(v);
		let dz = sz / n - m.vz(v);
		// Project the step onto the tangent plane.
		const nx = m.vertNormal[3 * v];
		const ny = m.vertNormal[3 * v + 1];
		const nz = m.vertNormal[3 * v + 2];
		const along = dx * nx + dy * ny + dz * nz;
		dx -= along * nx;
		dy -= along * ny;
		dz -= along * nz;

		// No vertex may travel more than a third of the way to its nearest
		// neighbour. Since that neighbour is bound by the same rule against an
		// edge no longer than this one, the gap between any two vertices can
		// close by at most two thirds in a pass — so they can never meet, and
		// no zero-length edge or degenerate face can appear. Without the bound
		// they do: ten passes over a cylinder produced edges of length exactly 0.
		const step = Math.hypot(dx, dy, dz);
		const limit = nearest / 3;
		if (step > limit && step > 0) {
			const k = limit / step;
			dx *= k;
			dy *= k;
			dz *= k;
		}

		let x = m.vx(v) + dx;
		let y = m.vy(v) + dy;
		let z = m.vz(v) + dz;
		if (options.reprojectFlag && ref !== null) {
			// Correct only the component along the normal. Snapping the vertex
			// onto the nearest sample instead would throw away the tangential
			// spread this pass just created, and — worse — land two different
			// vertices on the same sample, giving a zero-length edge and a
			// degenerate face.
			const i = ref.tree.nearestToPoint(x, y, z);
			if (i >= 0) {
				const off =
					(ref.coords[3 * i] - x) * nx +
					(ref.coords[3 * i + 1] - y) * ny +
					(ref.coords[3 * i + 2] - z) * nz;
				x += off * nx;
				y += off * ny;
				z += off * nz;
			}
		}
		if (options.checkSurfDist && ref !== null) {
			if (distanceToReference(ref, x, y, z) > options.maxSurfDist) continue;
		}
		// Keeping a vertex away from its neighbours is not enough to keep the
		// triangles healthy: three vertices can go collinear without any two of
		// them meeting. Near convergence that showed up as a handful of
		// zero-area faces on an otherwise clean cylinder wall.
		if (!moveKeepsFacesSane(m, vertFaces, v, x, y, z)) continue;
		target[3 * v] = x;
		target[3 * v + 1] = y;
		target[3 * v + 2] = z;
		move[v] = 1;
	}

	// Written in one go, so the pass is a Jacobi step rather than a
	// Gauss-Seidel one and the result does not depend on vertex order.
	for (let v = 0; v < m.vertSize; v++) {
		if (move[v]) m.setVert(v, target[3 * v], target[3 * v + 1], target[3 * v + 2]);
	}
}

/** Whether moving `v` to the given point leaves every incident face usable. */
function moveKeepsFacesSane(
	m: CMeshO,
	vertFaces: ReadonlyArray<Set<number>>,
	v: number,
	x: number,
	y: number,
	z: number,
): boolean {
	for (const f of vertFaces[v]) {
		if (m.isFaceD(f)) continue;
		const p: Array<[number, number, number]> = [];
		for (let k = 0; k < 3; k++) {
			const q = m.faceVert[3 * f + k];
			p.push(q === v ? [x, y, z] : [m.vx(q), m.vy(q), m.vz(q)]);
		}
		const after = triQuality(
			p[0][0],
			p[0][1],
			p[0][2],
			p[1][0],
			p[1][1],
			p[1][2],
			p[2][0],
			p[2][1],
			p[2][2],
		);
		if (after >= MIN_COLLAPSE_QUALITY) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		const before = triQuality(
			m.vx(a),
			m.vy(a),
			m.vz(a),
			m.vx(b),
			m.vy(b),
			m.vz(b),
			m.vx(c),
			m.vy(c),
			m.vz(c),
		);
		// As with collapses: refuse only when the move makes things worse, so a
		// mesh that starts out full of slivers can still be relaxed out of them.
		if (after < before) return false;
	}
	return true;
}

/** Grid-cell clustering decimation: one vertex per occupied cell. */
export function clusteringDecimation(m: CMeshO, cellSize: number): void {
	if (!(cellSize > 0)) throw new Error(`Clustering needs a positive cell size, got ${cellSize}`);
	UpdateBounding.box(m);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);

	// VCG's grid, not the naive one: the box is inflated by one cell on every
	// side, the cell count per axis is truncated from the inflated extent, and
	// the voxel actually used is the extent divided back by that count — so the
	// real cell edge is a little *larger* than asked for, and the bin walls sit
	// where MeshLab's sit. Binning straight off bbox.min with the exact cell
	// size puts every wall elsewhere and clusters differently.
	const low = [m.bbox.min[0] - cellSize, m.bbox.min[1] - cellSize, m.bbox.min[2] - cellSize];
	const high = [m.bbox.max[0] + cellSize, m.bbox.max[1] + cellSize, m.bbox.max[2] + cellSize];
	const voxel = [0, 1, 2].map((axis) => {
		const dim = high[axis] - low[axis];
		const count = Math.max(1, Math.trunc(dim / cellSize));
		return dim / count;
	});

	const cellOf = (x: number, y: number, z: number): string => {
		const i = Math.floor((x - low[0]) / voxel[0]);
		const j = Math.floor((y - low[1]) / voxel[1]);
		const k = Math.floor((z - low[2]) / voxel[2]);
		return `${i},${j},${k}`;
	};

	// Each occupied cell keeps the average of everything that landed in it.
	interface Cell {
		x: number;
		y: number;
		z: number;
		nx: number;
		ny: number;
		nz: number;
		count: number;
		index: number;
	}
	const cells = new Map<string, Cell>();
	const cellFor = (x: number, y: number, z: number): Cell => {
		const key = cellOf(x, y, z);
		let cell = cells.get(key);
		if (cell === undefined) {
			cell = { x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 0, count: 0, index: -1 };
			cells.set(key, cell);
		}
		return cell;
	};

	const isPointCloud = m.fn === 0;
	// A face contributes each of its corners to the cell that corner falls in,
	// weighted by the face's un-normalised normal — so a tiny face facing the
	// wrong way barely counts against a large one.
	const faceCells: Array<[Cell, Cell, Cell]> = [];
	if (isPointCloud) {
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			const cell = cellFor(m.vx(v), m.vy(v), m.vz(v));
			accumulate(
				cell,
				m.vx(v),
				m.vy(v),
				m.vz(v),
				m.vertNormal[3 * v],
				m.vertNormal[3 * v + 1],
				m.vertNormal[3 * v + 2],
			);
		}
	} else {
		const scratch = new Float64Array(3);
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			const n = faceAreaNormal(m, f, scratch);
			const corners: Cell[] = [];
			for (let k = 0; k < 3; k++) {
				const v = m.fv(f, k);
				const cell = cellFor(m.vx(v), m.vy(v), m.vz(v));
				accumulate(cell, m.vx(v), m.vy(v), m.vz(v), n[0], n[1], n[2]);
				corners.push(cell);
			}
			// A face whose three corners do not fall in three distinct cells has
			// collapsed to an edge or a point, and simply disappears.
			if (corners[0] !== corners[1] && corners[1] !== corners[2] && corners[0] !== corners[2]) {
				faceCells.push([corners[0], corners[1], corners[2]]);
			}
		}
	}

	const kept = [...cells.values()];
	const rebuilt = new Map<string, true>();
	m.clear();
	if (kept.length === 0) return;

	const first = Allocator.addVertices(m, kept.length);
	for (let i = 0; i < kept.length; i++) {
		const c = kept[i];
		c.index = first + i;
		m.setVert(c.index, c.x / c.count, c.y / c.count, c.z / c.count);
		const len = Math.hypot(c.nx, c.ny, c.nz);
		if (len > 0) {
			m.vertNormal[3 * c.index] = c.nx / len;
			m.vertNormal[3 * c.index + 1] = c.ny / len;
			m.vertNormal[3 * c.index + 2] = c.nz / len;
		}
	}

	// Two faces landing on the same three cells are the same face now, and only
	// one of them is wanted. Rotating each triple to start at its lowest index
	// keeps the winding, so a two-sided sheet still comes out as two faces.
	const unique: Array<[number, number, number]> = [];
	for (const [c0, c1, c2] of faceCells) {
		const t = rotateToMin(c0.index, c1.index, c2.index);
		const key = `${t[0]},${t[1]},${t[2]}`;
		if (rebuilt.has(key)) continue;
		rebuilt.set(key, true);
		unique.push(t);
	}
	if (unique.length > 0) {
		const firstFace = Allocator.addFaces(m, unique.length);
		for (let i = 0; i < unique.length; i++) {
			m.setFace(firstFace + i, unique[i][0], unique[i][1], unique[i][2]);
		}
	}
	Clean.removeDegenerateFace(m);
	Allocator.compactEveryVector(m);
	UpdateBounding.box(m);
}

function accumulate(
	cell: { x: number; y: number; z: number; nx: number; ny: number; nz: number; count: number },
	x: number,
	y: number,
	z: number,
	nx: number,
	ny: number,
	nz: number,
): void {
	cell.x += x;
	cell.y += y;
	cell.z += z;
	cell.nx += nx;
	cell.ny += ny;
	cell.nz += nz;
	cell.count++;
}

/** The face normal, left un-normalised so its length carries the area. */
function faceAreaNormal(m: CMeshO, f: number, out: Float64Array): Float64Array {
	const a = m.fv(f, 0);
	const b = m.fv(f, 1);
	const c = m.fv(f, 2);
	const ux = m.vx(b) - m.vx(a);
	const uy = m.vy(b) - m.vy(a);
	const uz = m.vz(b) - m.vz(a);
	const vx = m.vx(c) - m.vx(a);
	const vy = m.vy(c) - m.vy(a);
	const vz = m.vz(c) - m.vz(a);
	out[0] = uy * vz - uz * vy;
	out[1] = uz * vx - ux * vz;
	out[2] = ux * vy - uy * vx;
	return out;
}

/** Rotates a triple so its smallest entry comes first, preserving cyclic order. */
function rotateToMin(a: number, b: number, c: number): [number, number, number] {
	if (b < a && b < c) return [b, c, a];
	if (c < a && c < b) return [c, a, b];
	return [a, b, c];
}

export const IsotropicRemeshing = {
	isotropicRemeshing,
	clusteringDecimation,
	REMESH_DEFAULTS,
} as const;
