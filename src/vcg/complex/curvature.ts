/**
 * `vcg/complex/algorithms/update/curvature.h` — curvature on a triangle mesh.
 *
 * Two things a discrete surface can mean by "curvature", and they are computed
 * differently:
 *
 *  - The **scalar** mean and Gaussian curvatures, from Desbrun's cotangent
 *    formulation. Cheap, and what `Discrete Curvatures` reports.
 *  - The **tensor**: two principal curvatures and the two directions they act
 *    along, by Taubin's method. What anisotropic remeshing and crease
 *    detection want, and what `Compute curvature principal directions` writes
 *    into the `vertCurvDir` channel.
 *
 * Both are one-ring computations, so both need the mesh's adjacency and both
 * are meaningless at an unreferenced vertex — a vertex with no incident face
 * has no neighbourhood to be curved in, and comes back at zero rather than
 * at NaN.
 */

import type { CMeshO } from "./cmesho.ts";
import { UpdateNormal } from "./update/normal.ts";
import { UpdateTopology } from "./update/topology.ts";

/** Per-vertex mean and Gaussian curvature. */
export interface ScalarCurvature {
	/** Mean curvature H. */
	readonly mean: Float64Array;
	/** Gaussian curvature K. */
	readonly gaussian: Float64Array;
}

const HALF_PI = Math.PI / 2;

/**
 * Mean and Gaussian curvature by the cotangent formula of Desbrun et al.
 *
 * Each vertex gets a share of every incident triangle's area — the "mixed"
 * area, which is the Voronoi region where the triangle is acute and a
 * fallback split where it is obtuse, because the Voronoi cell of an obtuse
 * triangle spills outside it and would make the area negative.
 *
 * The Gaussian curvature is the angle deficit `2π - Σθ` over that area. On a
 * boundary vertex the surface does not close, so the deficit is measured
 * against the boundary's own opening angle instead of a full turn.
 */
export function meanAndGaussian(m: CMeshO): ScalarCurvature {
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(m);

	const area = new Float64Array(m.vertSize);
	const contrib = new Float64Array(m.vertSize * 3);
	const mean = new Float64Array(m.vertSize);
	const gaussian = new Float64Array(m.vertSize);
	for (let v = 0; v < m.vertSize; v++) gaussian[v] = 2 * Math.PI;

	const angles = new Float64Array(3);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceAngles(m, f, angles);
		const v = [m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)];

		if (angles[0] < HALF_PI && angles[1] < HALF_PI && angles[2] < HALF_PI) {
			const e01 = squaredEdge(m, v[1], v[0]);
			const e12 = squaredEdge(m, v[2], v[1]);
			const e20 = squaredEdge(m, v[0], v[2]);
			const cot = (a: number) => 1 / Math.tan(a);
			area[v[0]] += (e20 * cot(angles[1]) + e01 * cot(angles[2])) / 8;
			area[v[1]] += (e01 * cot(angles[2]) + e12 * cot(angles[0])) / 8;
			area[v[2]] += (e12 * cot(angles[0]) + e20 * cot(angles[1])) / 8;
		} else {
			// The obtuse corner takes half the triangle and the other two a
			// quarter each — Meyer's fallback, because the Voronoi cell of an
			// obtuse triangle reaches outside it.
			const doubleArea = doubleAreaOf(m, f);
			const obtuse = angles[0] >= HALF_PI ? 0 : angles[1] >= HALF_PI ? 1 : 2;
			for (let k = 0; k < 3; k++) area[v[k]] += doubleArea / (k === obtuse ? 4 : 8);
		}
	}

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceAngles(m, f, angles);
		// A degenerate triangle has no well-defined cotangent.
		if (angles[0] === 0 || angles[1] === 0 || angles[2] === 0) continue;
		const v = [m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)];
		const e = [edgeVector(m, v[1], v[0]), edgeVector(m, v[2], v[1]), edgeVector(m, v[0], v[2])];
		const cot = [1 / Math.tan(angles[0]), 1 / Math.tan(angles[1]), 1 / Math.tan(angles[2])];
		for (let a = 0; a < 3; a++) {
			contrib[3 * v[0] + a] += (e[2][a] * cot[1] - e[0][a] * cot[2]) / 4;
			contrib[3 * v[1] + a] += (e[0][a] * cot[2] - e[1][a] * cot[0]) / 4;
			contrib[3 * v[2] + a] += (e[1][a] * cot[0] - e[2][a] * cot[1]) / 4;
		}
		for (let k = 0; k < 3; k++) gaussian[v[k]] -= angles[k];

		for (let k = 0; k < 3; k++) {
			if (!m.isBorderFF(f, k)) continue;
			// On a boundary the full turn is not available, so the deficit is
			// taken against the angle the boundary itself opens.
			const at = m.fv(f, k);
			const [e1, e2] = boundaryEdges(m, f, k);
			if (e1 !== null && e2 !== null) gaussian[at] -= Math.abs(angleBetween(e1, e2));
		}
	}

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || area[v] <= Number.EPSILON) {
			mean[v] = 0;
			gaussian[v] = 0;
			continue;
		}
		const dot =
			contrib[3 * v] * m.vertNormal[3 * v] +
			contrib[3 * v + 1] * m.vertNormal[3 * v + 1] +
			contrib[3 * v + 2] * m.vertNormal[3 * v + 2];
		const length = Math.hypot(contrib[3 * v], contrib[3 * v + 1], contrib[3 * v + 2]) / area[v];
		// The magnitude comes from the Laplacian; the normal decides the sign,
		// so a dome and a bowl do not read the same.
		mean[v] = (dot > 0 ? 1 : -1) * length;
		gaussian[v] /= area[v];
	}
	return { mean, gaussian };
}

/** How MeshLab's `Discrete Curvatures` enum is ordered. */
export const CurvatureType = {
	Mean: 0,
	Gaussian: 1,
	RMS: 2,
	ABS: 3,
} as const;

/**
 * One scalar per vertex, from the mean and Gaussian pair.
 *
 * `ABS` is `|H| + |K|` and `RMS` is `sqrt(4H² - 2K)`, both from Pulla, Razdan
 * and Farin. The RMS radicand can go negative on a saddle where the estimates
 * disagree, so it is floored at zero rather than left to produce NaN.
 */
export function discreteCurvature(m: CMeshO, type: number): Float64Array {
	const { mean, gaussian } = meanAndGaussian(m);
	const out = new Float64Array(m.vertSize);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const h = mean[v];
		const k = gaussian[v];
		switch (type) {
			case CurvatureType.Gaussian:
				out[v] = k;
				break;
			case CurvatureType.RMS:
				out[v] = Math.sqrt(Math.max(0, 4 * h * h - 2 * k));
				break;
			case CurvatureType.ABS:
				out[v] = Math.abs(h) + Math.abs(k);
				break;
			default:
				out[v] = h;
				break;
		}
	}
	return out;
}

/** The layout of the `vertCurvDir` channel: d1(xyz), d2(xyz), k1, k2. */
const CURV_STRIDE = 8;

/**
 * Taubin's principal directions, written into the `vertCurvDir` channel.
 *
 * The idea: every neighbour of a vertex gives a normal-section curvature in
 * its own direction, and the area-weighted sum of those directions' outer
 * products is a 3x3 matrix whose eigenvectors are the principal directions.
 * Taubin's contribution is that the matrix can be diagonalised in closed form
 * — a Householder reflection takes the normal onto an axis, and a single
 * Givens rotation finishes the 2x2 minor that is left.
 *
 * Requires the caller to have enabled `MM_VERTCURVDIR`.
 */
export function principalDirections(m: CMeshO): void {
	const curv = m.vertCurvDir;
	if (curv === null) {
		throw new Error("principalDirections needs the vertCurvDir channel (MM_VERTCURVDIR)");
	}
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	UpdateNormal.perVertexAngleWeighted(m);
	UpdateNormal.normalizePerVertex(m);

	const ring = buildOneRings(m);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const neighbours = ring[v];
		if (neighbours.length === 0) continue;

		const n: [number, number, number] = [
			m.vertNormal[3 * v],
			m.vertNormal[3 * v + 1],
			m.vertNormal[3 * v + 2],
		];

		// The area weights: each neighbour carries the mean of the two
		// triangles it borders, or just its own where the ring is open.
		let totalArea = 0;
		for (const nb of neighbours) totalArea += nb.doubleArea;
		if (totalArea <= 0) continue;

		// I - n n^T projects an edge onto the tangent plane.
		const project = (e: readonly number[]): [number, number, number] => {
			const along = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
			return [e[0] - along * n[0], e[1] - along * n[1], e[2] - along * n[2]];
		};

		const M = new Float64Array(9);
		for (let i = 0; i < neighbours.length; i++) {
			const nb = neighbours[i];
			const previous = neighbours[(i - 1 + neighbours.length) % neighbours.length];
			const weight = nb.onBorder
				? nb.doubleArea / totalArea
				: (0.5 * (nb.doubleArea + previous.doubleArea)) / totalArea;

			const edge: [number, number, number] = [
				m.vx(v) - m.vx(nb.vertex),
				m.vy(v) - m.vy(nb.vertex),
				m.vz(v) - m.vz(nb.vertex),
			];
			const squared = edge[0] ** 2 + edge[1] ** 2 + edge[2] ** 2;
			if (squared === 0) continue;
			// The normal-section curvature along this edge.
			const curvature = (2 * (n[0] * edge[0] + n[1] * edge[1] + n[2] * edge[2])) / squared;
			const t = project(edge);
			const len = Math.hypot(t[0], t[1], t[2]);
			if (len === 0) continue;
			for (let a = 0; a < 3; a++) t[a] /= len;
			const scale = weight * curvature;
			for (let a = 0; a < 3; a++) {
				for (let b = 0; b < 3; b++) M[3 * a + b] += scale * t[a] * t[b];
			}
		}

		// Householder: reflect e1 onto the normal, so the first column of Q is
		// the normal and the other two span the tangent plane.
		const e1: [number, number, number] = [1, 0, 0];
		const minus = (e1[0] - n[0]) ** 2 + n[1] ** 2 + n[2] ** 2;
		const plus = (e1[0] + n[0]) ** 2 + n[1] ** 2 + n[2] ** 2;
		const w: [number, number, number] =
			minus > plus ? [e1[0] - n[0], -n[1], -n[2]] : [e1[0] + n[0], n[1], n[2]];
		const wLen = Math.hypot(w[0], w[1], w[2]);
		if (wLen === 0) continue;
		for (let a = 0; a < 3; a++) w[a] /= wLen;

		// Q = I - 2ww^T, which is its own transpose and its own inverse.
		const Q = new Float64Array(9);
		for (let a = 0; a < 3; a++) {
			for (let b = 0; b < 3; b++) Q[3 * a + b] = (a === b ? 1 : 0) - 2 * w[a] * w[b];
		}
		const QtMQ = multiply3(transpose3(Q), multiply3(M, Q));
		const T1: [number, number, number] = [Q[1], Q[4], Q[7]];
		const T2: [number, number, number] = [Q[2], Q[5], Q[8]];

		const { c, s } = givens(QtMQ[4], QtMQ[8], QtMQ[7]);
		// S^T (2x2 minor) S, done by hand rather than through a matrix type.
		const m11 = QtMQ[4];
		const m12 = QtMQ[5];
		const m21 = QtMQ[7];
		const m22 = QtMQ[8];
		const a11 = c * (c * m11 - s * m21) - s * (c * m12 - s * m22);
		const a22 = s * (s * m11 + c * m21) + c * (s * m12 + c * m22);

		// Taubin's estimator: the tensor's trace is under-counted by the
		// weighting, and these coefficients undo that.
		const k1 = 3 * a11 - a22;
		const k2 = 3 * a22 - a11;
		const d1: [number, number, number] = [
			T1[0] * c - T2[0] * s,
			T1[1] * c - T2[1] * s,
			T1[2] * c - T2[2] * s,
		];
		const d2: [number, number, number] = [
			T1[0] * s + T2[0] * c,
			T1[1] * s + T2[1] * c,
			T1[2] * s + T2[2] * c,
		];

		const base = v * CURV_STRIDE;
		for (let a = 0; a < 3; a++) {
			curv[base + a] = d1[a];
			curv[base + 3 + a] = d2[a];
		}
		curv[base + 6] = k1;
		curv[base + 7] = k2;
	}
}

/** The smaller and larger principal curvature at `v`, as stored. */
export function principalCurvatures(m: CMeshO, v: number): { k1: number; k2: number } {
	const curv = m.vertCurvDir;
	if (curv === null) return { k1: 0, k2: 0 };
	return { k1: curv[v * CURV_STRIDE + 6], k2: curv[v * CURV_STRIDE + 7] };
}

/** How MeshLab's "Quality/Color Mapping" enum for the tensor is ordered. */
export const CurvatureMapping = {
	Mean: 0,
	Gaussian: 1,
	MinCurvature: 2,
	MaxCurvature: 3,
	ShapeIndex: 4,
	Curvedness: 5,
	None: 6,
} as const;

/**
 * Reduces the curvature tensor at `v` to the single number the mapping names.
 *
 * Shape index is Koenderink's: a number in -1..1 that says what *kind* of
 * shape the point is — cup, rut, saddle, ridge, cap — independently of how
 * strongly curved it is. Curvedness is the strength, independently of kind.
 */
export function curvatureToScalar(m: CMeshO, v: number, mapping: number): number {
	const { k1, k2 } = principalCurvatures(m, v);
	switch (mapping) {
		case CurvatureMapping.Gaussian:
			return k1 * k2;
		case CurvatureMapping.MinCurvature:
			return k1;
		case CurvatureMapping.MaxCurvature:
			return k2;
		case CurvatureMapping.ShapeIndex: {
			const hi = Math.max(k1, k2);
			const lo = Math.min(k1, k2);
			return (2 / Math.PI) * Math.atan2(hi + lo, hi - lo);
		}
		case CurvatureMapping.Curvedness:
			return Math.sqrt((k1 * k1 + k2 * k2) / 2);
		case CurvatureMapping.None:
			return 0;
		default:
			return (k1 + k2) / 2;
	}
}

interface RingNeighbour {
	readonly vertex: number;
	readonly doubleArea: number;
	readonly onBorder: boolean;
}

/**
 * The one-ring of every vertex, in the order the incident faces walk.
 *
 * Order matters: Taubin's weights average each neighbour's face with the one
 * before it, which only means anything if the neighbours come round the
 * vertex in sequence rather than in whatever order the face list happens to
 * hold them.
 */
function buildOneRings(m: CMeshO): RingNeighbour[][] {
	const incident: number[][] = Array.from({ length: m.vertSize }, () => []);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) incident[m.fv(f, k)].push(f);
	}

	const rings: RingNeighbour[][] = Array.from({ length: m.vertSize }, () => []);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v) || incident[v].length === 0) continue;
		// Each incident face contributes the edge (v -> next), so chaining
		// "next" to "previous" walks the ring.
		const nextOf = new Map<number, { to: number; face: number }>();
		for (const f of incident[v]) {
			for (let k = 0; k < 3; k++) {
				if (m.fv(f, k) !== v) continue;
				nextOf.set(m.fv(f, (k + 1) % 3), { to: m.fv(f, (k + 2) % 3), face: f });
			}
		}
		// Start at a boundary if there is one, so an open ring is walked from
		// its end rather than from the middle.
		let start = -1;
		for (const [from] of nextOf) {
			let isStart = true;
			for (const [, step] of nextOf) if (step.to === from) isStart = false;
			if (isStart) {
				start = from;
				break;
			}
		}
		const open = start >= 0;
		if (start < 0) start = [...nextOf.keys()][0];

		const out: RingNeighbour[] = [];
		const seen = new Set<number>();
		let at: number | undefined = start;
		while (at !== undefined && !seen.has(at)) {
			seen.add(at);
			const step = nextOf.get(at);
			if (step === undefined) break;
			out.push({ vertex: at, doubleArea: doubleAreaOf(m, step.face), onBorder: open });
			at = step.to;
		}
		// The last neighbour of an open ring is the far end of the final edge,
		// which no face lists as a "from".
		if (open && at !== undefined && !seen.has(at)) {
			const last = out[out.length - 1];
			out.push({
				vertex: at,
				doubleArea: last === undefined ? 0 : last.doubleArea,
				onBorder: true,
			});
		}
		rings[v] = out;
	}
	return rings;
}

/** Solves for the Givens rotation that kills the off-diagonal term. */
function givens(m11: number, m22: number, m21: number): { c: number; s: number } {
	const alpha = m11 - m22;
	const beta = m21;
	// Already diagonal: no rotation to make.
	if (beta === 0) return { c: 1, s: 0 };

	const delta = Math.sqrt(4 * alpha * alpha + 16 * beta * beta);
	const h = [(2 * alpha + delta) / (2 * beta), (2 * alpha - delta) / (2 * beta)];
	let best = { c: 1, s: 0 };
	let minError = Number.POSITIVE_INFINITY;
	for (const hi of h) {
		const d = Math.sqrt(hi * hi + 4);
		for (const t of [(hi + d) / 2, (hi - d) / 2]) {
			const squared = t * t;
			const denominator = 1 + squared;
			const s = (2 * t) / denominator;
			const c = (1 - squared) / denominator;
			// Two of the four roots rotate by the right amount but into the
			// wrong quadrant; upstream picks between them by how nearly the
			// off-diagonal vanishes and how well acos and asin agree.
			const approximation = c * s * alpha + (c * c - s * s) * beta;
			const similarity = Math.abs(Math.acos(c) / Math.asin(s));
			const error = Math.abs(1 - similarity) + Math.abs(approximation);
			if (error < minError) {
				minError = error;
				best = { c, s };
			}
		}
	}
	return best;
}

function multiply3(a: Float64Array, b: Float64Array): Float64Array {
	const out = new Float64Array(9);
	for (let i = 0; i < 3; i++) {
		for (let j = 0; j < 3; j++) {
			let sum = 0;
			for (let k = 0; k < 3; k++) sum += a[3 * i + k] * b[3 * k + j];
			out[3 * i + j] = sum;
		}
	}
	return out;
}

function transpose3(a: Float64Array): Float64Array {
	const out = new Float64Array(9);
	for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[3 * i + j] = a[3 * j + i];
	return out;
}

function faceAngles(m: CMeshO, f: number, out: Float64Array): void {
	const a = m.fv(f, 0);
	const b = m.fv(f, 1);
	const c = m.fv(f, 2);
	out[0] = Math.abs(angleBetween(edgeVector(m, b, a), edgeVector(m, c, a)));
	out[1] = Math.abs(angleBetween(edgeVector(m, a, b), edgeVector(m, c, b)));
	// The third follows, which also keeps the three summing to pi exactly.
	out[2] = Math.PI - (out[0] + out[1]);
}

const edgeVector = (m: CMeshO, to: number, from: number): [number, number, number] => [
	m.vx(to) - m.vx(from),
	m.vy(to) - m.vy(from),
	m.vz(to) - m.vz(from),
];

function angleBetween(u: readonly number[], v: readonly number[]): number {
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
	return Math.acos(cos < -1 ? -1 : cos > 1 ? 1 : cos);
}

const squaredEdge = (m: CMeshO, a: number, b: number): number =>
	(m.vx(a) - m.vx(b)) ** 2 + (m.vy(a) - m.vy(b)) ** 2 + (m.vz(a) - m.vz(b)) ** 2;

function doubleAreaOf(m: CMeshO, f: number): number {
	const a = m.fv(f, 0);
	const u = edgeVector(m, m.fv(f, 1), a);
	const v = edgeVector(m, m.fv(f, 2), a);
	return Math.hypot(
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0],
	);
}

/** The two boundary edges meeting at corner `k` of border face `f`. */
function boundaryEdges(m: CMeshO, f: number, k: number): [number[] | null, number[] | null] {
	const at = m.fv(f, k);
	const other = m.fv(f, (k + 1) % 3);
	const first = edgeVector(m, other, at);
	// Walk the boundary one step from `at` to find the edge on the other side.
	for (let g = 0; g < m.faceSize; g++) {
		if (m.isFaceD(g)) continue;
		for (let e = 0; e < 3; e++) {
			if (!m.isBorderFF(g, e)) continue;
			if (m.fv(g, (e + 1) % 3) !== at) continue;
			return [first, edgeVector(m, m.fv(g, e), at)];
		}
	}
	return [first, null];
}

export const UpdateCurvature = {
	meanAndGaussian,
	discreteCurvature,
	principalDirections,
	principalDirectionsFitting,
	principalCurvatures,
	curvatureToScalar,
	CurvatureType,
	CurvatureMapping,
} as const;

/**
 * Principal curvature by fitting a quadric patch to each one-ring.
 *
 * MeshLab's default, and the better estimator of the two: rotate the
 * neighbourhood into a frame whose z axis is the vertex normal, least-squares
 * fit `z = ax² + bxy + cy² + dx + ey`, and read the curvature off the surface
 * that fit describes. Taubin's method only ever sees the normal-section
 * curvature along each edge; this one sees the whole patch, so it is steadier
 * on an irregular ring.
 *
 * The linear terms `d` and `e` are fitted rather than assumed zero. They
 * would be zero if the normal were exact, and they are not — so keeping them
 * and carrying them into the first fundamental form is what stops a slightly
 * wrong normal from biasing the curvature.
 */
export function principalDirectionsFitting(m: CMeshO): void {
	const curv = m.vertCurvDir;
	if (curv === null) {
		throw new Error("principalDirectionsFitting needs the vertCurvDir channel (MM_VERTCURVDIR)");
	}
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	UpdateNormal.perVertexAngleWeighted(m);
	UpdateNormal.normalizePerVertex(m);

	const ring = buildOneRings(m);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const neighbours = ring[v];
		// Five unknowns need five samples; below that the fit is not determined
		// and the vertex keeps whatever it had.
		if (neighbours.length < 5) continue;

		const n: [number, number, number] = [
			m.vertNormal[3 * v],
			m.vertNormal[3 * v + 1],
			m.vertNormal[3 * v + 2],
		];
		const [u, w] = tangentFrame(n);

		// Normal equations for the 5-term fit, accumulated directly: the design
		// matrix is never formed, because a one-ring is small and A^T A is 5x5
		// however many neighbours there are.
		const ata = new Float64Array(25);
		const atb = new Float64Array(5);
		const row = new Float64Array(5);
		for (const nb of neighbours) {
			const d: [number, number, number] = [
				m.vx(nb.vertex) - m.vx(v),
				m.vy(nb.vertex) - m.vy(v),
				m.vz(nb.vertex) - m.vz(v),
			];
			const x = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
			const y = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
			// Measured *against* the normal, so a convex surface reads positive.
			// The natural sign convention for `z = f(x, y)` is the other way
			// round, and would have a sphere with outward normals come back at
			// curvature -1/R while Taubin's method reports +1/R.
			const z = -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]);
			row[0] = x * x;
			row[1] = x * y;
			row[2] = y * y;
			row[3] = x;
			row[4] = y;
			for (let i = 0; i < 5; i++) {
				atb[i] += row[i] * z;
				for (let j = 0; j < 5; j++) ata[5 * i + j] += row[i] * row[j];
			}
		}
		const fit = solveSymmetric(ata, atb, 5);
		if (fit === null) continue;
		const [a, b, c, dx, dy] = fit;

		// The two fundamental forms of z = f(x, y) at the origin.
		const scale = Math.sqrt(dx * dx + dy * dy + 1);
		const E = 1 + dx * dx;
		const F = dx * dy;
		const G = 1 + dy * dy;
		const L = (2 * a) / scale;
		const M = b / scale;
		const N = (2 * c) / scale;

		// Weingarten = first⁻¹ · second.
		const det = E * G - F * F;
		if (det === 0) continue;
		const w11 = (G * L - F * M) / det;
		const w12 = (G * M - F * N) / det;
		const w21 = (E * M - F * L) / det;
		const w22 = (E * N - F * M) / det;

		const { k1, k2, angle } = eigen2x2(w11, w12, w21, w22);
		const cos = Math.cos(angle);
		const sin = Math.sin(angle);
		const base = v * CURV_STRIDE;
		for (let axis = 0; axis < 3; axis++) {
			curv[base + axis] = u[axis] * cos + w[axis] * sin;
			curv[base + 3 + axis] = -u[axis] * sin + w[axis] * cos;
		}
		curv[base + 6] = k1;
		curv[base + 7] = k2;
	}
}

/** Two unit vectors spanning the plane perpendicular to `n`. */
function tangentFrame(n: readonly number[]): [[number, number, number], [number, number, number]] {
	// Start from whichever axis the normal leans on least, so the cross
	// product is well conditioned.
	const abs = [Math.abs(n[0]), Math.abs(n[1]), Math.abs(n[2])];
	const axis = abs.indexOf(Math.min(...abs));
	const seed = [axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0];
	const u: [number, number, number] = [
		seed[1] * n[2] - seed[2] * n[1],
		seed[2] * n[0] - seed[0] * n[2],
		seed[0] * n[1] - seed[1] * n[0],
	];
	const lu = Math.hypot(u[0], u[1], u[2]) || 1;
	for (let a = 0; a < 3; a++) u[a] /= lu;
	const w: [number, number, number] = [
		n[1] * u[2] - n[2] * u[1],
		n[2] * u[0] - n[0] * u[2],
		n[0] * u[1] - n[1] * u[0],
	];
	return [u, w];
}

/**
 * Eigenvalues and rotation of a 2x2 Weingarten matrix.
 *
 * It is symmetric in exact arithmetic but not quite in floating point, so the
 * off-diagonal used here is the average of the two.
 */
function eigen2x2(
	a: number,
	b: number,
	c: number,
	d: number,
): { k1: number; k2: number; angle: number } {
	const off = (b + c) / 2;
	const trace = a + d;
	const gap = a - d;
	const root = Math.sqrt(gap * gap + 4 * off * off);
	const k1 = (trace - root) / 2;
	const k2 = (trace + root) / 2;
	// Half-angle formula: the direction of the larger eigenvalue.
	const angle = off === 0 && gap === 0 ? 0 : 0.5 * Math.atan2(2 * off, gap);
	return { k1, k2, angle };
}

/** Cholesky-free Gaussian elimination with partial pivoting, for a small dense system. */
function solveSymmetric(a: Float64Array, b: Float64Array, n: number): number[] | null {
	const m = Float64Array.from(a);
	const rhs = Float64Array.from(b);
	for (let col = 0; col < n; col++) {
		let pivot = col;
		for (let r = col + 1; r < n; r++) {
			if (Math.abs(m[n * r + col]) > Math.abs(m[n * pivot + col])) pivot = r;
		}
		// A singular normal matrix means the ring is degenerate — all its
		// points on a line, say — and there is no patch to fit.
		if (Math.abs(m[n * pivot + col]) < 1e-12) return null;
		if (pivot !== col) {
			for (let k = 0; k < n; k++) {
				const t = m[n * col + k];
				m[n * col + k] = m[n * pivot + k];
				m[n * pivot + k] = t;
			}
			const t = rhs[col];
			rhs[col] = rhs[pivot];
			rhs[pivot] = t;
		}
		for (let r = col + 1; r < n; r++) {
			const factor = m[n * r + col] / m[n * col + col];
			if (factor === 0) continue;
			for (let k = col; k < n; k++) m[n * r + k] -= factor * m[n * col + k];
			rhs[r] -= factor * rhs[col];
		}
	}
	const out = new Array<number>(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let sum = rhs[r];
		for (let k = r + 1; k < n; k++) sum -= m[n * r + k] * out[k];
		out[r] = sum / m[n * r + r];
	}
	return out.every(Number.isFinite) ? out : null;
}
