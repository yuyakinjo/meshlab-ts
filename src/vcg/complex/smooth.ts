/**
 * `Smooth` — Laplacian and Taubin smoothing, mirroring `vcg::tri::Smooth`.
 *
 * All of these move each vertex toward an average of its neighbours. What
 * distinguishes them is how they avoid the obvious problem: plain Laplacian
 * smoothing shrinks a closed surface toward a point, a little more with every
 * iteration.
 *
 * - `vertexCoordLaplacian` shrinks. Fine for a few steps, not for many.
 * - `vertexCoordTaubin` alternates a positive and a slightly larger negative
 *   step, which acts as a low-pass filter that leaves the overall scale alone.
 * - `vertexCoordLaplacianHC` records where each vertex started and pulls it
 *   partway back, correcting the drift directly.
 */
import type { CMeshO } from "./cmesho.ts";
import { VertexFlag } from "./flags.ts";
import { vertexBorderFromNone } from "./update/flag.ts";
import { UpdateTopology } from "./update/topology.ts";

/**
 * Sum of neighbour positions and neighbour count, per vertex.
 *
 * VCGLib's `LaplacianInfo`, as two flat arrays rather than a struct per
 * vertex.
 */
interface LaplacianAccumulator {
	sum: Float64Array;
	count: Float64Array;
}

function newAccumulator(n: number): LaplacianAccumulator {
	return { sum: new Float64Array(3 * n), count: new Float64Array(n) };
}

/**
 * Accumulates each vertex's neighbours.
 *
 * With `cotangentWeight` the contributions are weighted by the cotangents of
 * the opposite angles, which makes the result depend on the surface rather
 * than on how it happens to be triangulated — a long thin triangle should not
 * pull harder just because it is long.
 */
function accumulate(m: CMeshO, acc: LaplacianAccumulator, cotangentWeight: boolean): void {
	acc.sum.fill(0);
	acc.count.fill(0);

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = m.faceVert[3 * f + e];
			const b = m.faceVert[3 * f + ((e + 1) % 3)];

			let w = 1;
			if (cotangentWeight) {
				// The angle at the vertex opposite this edge.
				const c = m.faceVert[3 * f + ((e + 2) % 3)];
				const ux = m.vx(a) - m.vx(c);
				const uy = m.vy(a) - m.vy(c);
				const uz = m.vz(a) - m.vz(c);
				const vx = m.vx(b) - m.vx(c);
				const vy = m.vy(b) - m.vy(c);
				const vz = m.vz(b) - m.vz(c);
				const dot = ux * vx + uy * vy + uz * vz;
				const crossX = uy * vz - uz * vy;
				const crossY = uz * vx - ux * vz;
				const crossZ = ux * vy - uy * vx;
				const crossLen = Math.hypot(crossX, crossY, crossZ);
				// cot = cos/sin. A degenerate corner has no angle to speak of,
				// so it contributes nothing rather than an infinity.
				if (crossLen < 1e-300) continue;
				w = dot / crossLen;
				// Obtuse corners give a negative cotangent, which can pull a
				// vertex the wrong way and blow the smoothing up. Clamping is
				// the standard remedy.
				if (!Number.isFinite(w)) continue;
				if (w < 0) w = 0;
			}

			acc.sum[3 * a] += m.vx(b) * w;
			acc.sum[3 * a + 1] += m.vy(b) * w;
			acc.sum[3 * a + 2] += m.vz(b) * w;
			acc.count[a] += w;

			acc.sum[3 * b] += m.vx(a) * w;
			acc.sum[3 * b + 1] += m.vy(a) * w;
			acc.sum[3 * b + 2] += m.vz(a) * w;
			acc.count[b] += w;
		}
	}
}

export interface SmoothOptions {
	readonly smoothSelected?: boolean;
	readonly cotangentWeight?: boolean;
	/**
	 * Leave boundary vertices where they are.
	 *
	 * On an open mesh, smoothing the boundary pulls it inward and visibly
	 * shrinks the outline, so a repair pipeline usually wants it pinned.
	 */
	readonly pinBoundary?: boolean;
}

/** One Laplacian step scaled by `delta`, into the mesh. */
function laplacianStep(
	m: CMeshO,
	acc: LaplacianAccumulator,
	delta: number,
	options: SmoothOptions,
	borderMask: Uint8Array | null,
): void {
	accumulate(m, acc, options.cotangentWeight ?? false);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		if (options.smoothSelected === true && !m.isVertS(v)) continue;
		if (borderMask !== null && borderMask[v] === 1) continue;
		const n = acc.count[v];
		if (n === 0) continue;
		const tx = acc.sum[3 * v] / n;
		const ty = acc.sum[3 * v + 1] / n;
		const tz = acc.sum[3 * v + 2] / n;
		m.setVert(
			v,
			m.vx(v) + delta * (tx - m.vx(v)),
			m.vy(v) + delta * (ty - m.vy(v)),
			m.vz(v) + delta * (tz - m.vz(v)),
		);
	}
}

/** Which vertices to hold fixed, or null when none. */
function borderMaskOf(m: CMeshO, options: SmoothOptions): Uint8Array | null {
	if (options.pinBoundary !== true) return null;
	vertexBorderFromNone(m);
	const mask = new Uint8Array(m.vertSize);
	for (let v = 0; v < m.vertSize; v++) {
		if ((m.vertFlags[v] & VertexFlag.BORDER) !== 0) mask[v] = 1;
	}
	return mask;
}

/**
 * Plain Laplacian smoothing: each vertex moves to the average of its
 * neighbours, `step` times.
 *
 * Shrinks a closed surface. That is inherent to the operation, not a defect —
 * see {@link vertexCoordTaubin} for the version that does not.
 */
export function vertexCoordLaplacian(m: CMeshO, step = 1, options: SmoothOptions = {}): void {
	if (m.vn === 0) return;
	const acc = newAccumulator(m.vertSize);
	const border = borderMaskOf(m, options);
	for (let i = 0; i < step; i++) laplacianStep(m, acc, 1, options, border);
	m.imark++;
}

/**
 * Taubin's λ|μ smoothing: a positive step followed by a slightly larger
 * negative one, repeated.
 *
 * The pair acts as a low-pass filter on the surface. Because the shrinking of
 * the λ step and the expansion of the μ step very nearly cancel, the volume
 * survives many more iterations than plain Laplacian would allow — which is
 * why the default step count is 10 rather than 3.
 */
export function vertexCoordTaubin(
	m: CMeshO,
	step = 10,
	lambda = 0.5,
	mu = -0.53,
	options: SmoothOptions = {},
): void {
	if (m.vn === 0) return;
	const acc = newAccumulator(m.vertSize);
	const border = borderMaskOf(m, options);
	for (let i = 0; i < step; i++) {
		laplacianStep(m, acc, lambda, options, border);
		laplacianStep(m, acc, mu, options, border);
	}
	m.imark++;
}

/**
 * Humphrey's Classes smoothing: Laplacian, then a correction pulling each
 * vertex back toward where it and its neighbours started.
 *
 * `alpha` weights the original position, `beta` the neighbours' displacement.
 * Corrects the shrinking explicitly rather than by cancellation.
 */
export function vertexCoordLaplacianHC(
	m: CMeshO,
	step = 1,
	alpha = 0,
	beta = 0.5,
	options: SmoothOptions = {},
): void {
	if (m.vn === 0) return;
	const acc = newAccumulator(m.vertSize);
	const border = borderMaskOf(m, options);
	const original = new Float64Array(m.vertCoord.subarray(0, 3 * m.vertSize));
	const displacement = new Float64Array(3 * m.vertSize);

	for (let i = 0; i < step; i++) {
		const before = new Float64Array(m.vertCoord.subarray(0, 3 * m.vertSize));
		laplacianStep(m, acc, 1, options, border);

		// How far each vertex moved, blended against how far it has drifted
		// from where it began.
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			for (let k = 0; k < 3; k++) {
				const o = 3 * v + k;
				displacement[o] = m.vertCoord[o] - (alpha * original[o] + (1 - alpha) * before[o]);
			}
		}

		// Average each vertex's neighbours' displacement, and back it out.
		accumulate(m, acc, false);
		const dAcc = newAccumulator(m.vertSize);
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) {
				const a = m.faceVert[3 * f + e];
				const b = m.faceVert[3 * f + ((e + 1) % 3)];
				for (let k = 0; k < 3; k++) {
					dAcc.sum[3 * a + k] += displacement[3 * b + k];
					dAcc.sum[3 * b + k] += displacement[3 * a + k];
				}
				dAcc.count[a]++;
				dAcc.count[b]++;
			}
		}

		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			if (options.smoothSelected === true && !m.isVertS(v)) continue;
			if (border !== null && border[v] === 1) continue;
			const n = dAcc.count[v];
			if (n === 0) continue;
			for (let k = 0; k < 3; k++) {
				const o = 3 * v + k;
				m.vertCoord[o] -= beta * displacement[o] + ((1 - beta) * dAcc.sum[o]) / n;
			}
		}
	}
	m.imark++;
}

/**
 * Scale-dependent Laplacian (Desbrun): each neighbour's pull is divided by its
 * edge length, so the step is in proportion to the local feature size rather
 * than to how densely the surface happens to be sampled.
 *
 * `delta` is the step size; it has units of length squared over time, so it
 * must be scaled to the mesh.
 */
export function vertexCoordScaleDependentLaplacian(
	m: CMeshO,
	step = 1,
	delta = 0.001,
	options: SmoothOptions = {},
): void {
	if (m.vn === 0) return;
	const border = borderMaskOf(m, options);
	const acc = newAccumulator(m.vertSize);

	for (let i = 0; i < step; i++) {
		acc.sum.fill(0);
		acc.count.fill(0);
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			for (let e = 0; e < 3; e++) {
				const a = m.faceVert[3 * f + e];
				const b = m.faceVert[3 * f + ((e + 1) % 3)];
				const len = Math.hypot(m.vx(b) - m.vx(a), m.vy(b) - m.vy(a), m.vz(b) - m.vz(a));
				if (len === 0) continue;
				const w = 1 / len;
				acc.sum[3 * a] += (m.vx(b) - m.vx(a)) * w;
				acc.sum[3 * a + 1] += (m.vy(b) - m.vy(a)) * w;
				acc.sum[3 * a + 2] += (m.vz(b) - m.vz(a)) * w;
				acc.count[a] += w;
				acc.sum[3 * b] += (m.vx(a) - m.vx(b)) * w;
				acc.sum[3 * b + 1] += (m.vy(a) - m.vy(b)) * w;
				acc.sum[3 * b + 2] += (m.vz(a) - m.vz(b)) * w;
				acc.count[b] += w;
			}
		}
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			if (options.smoothSelected === true && !m.isVertS(v)) continue;
			if (border !== null && border[v] === 1) continue;
			const n = acc.count[v];
			if (n === 0) continue;
			m.setVert(
				v,
				m.vx(v) + (delta * acc.sum[3 * v]) / n,
				m.vy(v) + (delta * acc.sum[3 * v + 1]) / n,
				m.vz(v) + (delta * acc.sum[3 * v + 2]) / n,
			);
		}
	}
	m.imark++;
}

/**
 * Laplacian smoothing of the per-vertex quality, over the same neighbour graph
 * the coordinate smoother uses.
 *
 * Written as a gather into a scratch buffer and copied back per step, so the
 * result does not depend on the order the vertices happen to be visited in.
 */
export function vertexQualityLaplacian(m: CMeshO, step = 1): void {
	if (m.vn === 0) return;
	const neighbours = vertexNeighbours(m);
	for (let i = 0; i < step; i++) {
		const next = Float64Array.from(m.vertQuality);
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v) || neighbours[v].length === 0) continue;
			let sum = 0;
			for (const w of neighbours[v]) sum += m.vertQuality[w];
			next[v] = sum / neighbours[v].length;
		}
		m.vertQuality.set(next);
	}
	m.imark++;
}

/** Laplacian smoothing of the per-vertex colour, channel by channel. */
export function vertexColorLaplacian(m: CMeshO, step = 1): void {
	if (m.vn === 0) return;
	const neighbours = vertexNeighbours(m);
	for (let i = 0; i < step; i++) {
		const next = Uint32Array.from(m.vertColor);
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v) || neighbours[v].length === 0) continue;
			let r = 0;
			let g = 0;
			let b = 0;
			let a = 0;
			for (const w of neighbours[v]) {
				const c = m.vertColor[w];
				r += c & 0xff;
				g += (c >>> 8) & 0xff;
				b += (c >>> 16) & 0xff;
				a += (c >>> 24) & 0xff;
			}
			const n = neighbours[v].length;
			next[v] = pack(r / n, g / n, b / n, a / n);
		}
		m.vertColor.set(next);
	}
	m.imark++;
}

/**
 * Laplacian smoothing of the face normals, averaging across shared edges.
 *
 * "FF" in upstream's name: neighbours are the (at most three) faces across
 * this face's edges, not the faces sharing a vertex. A border edge simply has
 * no neighbour to contribute.
 */
export function faceNormalLaplacianFF(m: CMeshO, step = 1): void {
	if (m.fn === 0) return;
	if (m.ffFace === null) UpdateTopology.faceFace(m);
	for (let i = 0; i < step; i++) {
		const next = Float64Array.from(m.faceNormal);
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			let x = m.faceNormal[3 * f];
			let y = m.faceNormal[3 * f + 1];
			let z = m.faceNormal[3 * f + 2];
			let count = 1;
			for (let e = 0; e < 3; e++) {
				if (m.isBorderFF(f, e)) continue;
				const g = m.ffp(f, e);
				if (g < 0 || m.isFaceD(g)) continue;
				x += m.faceNormal[3 * g];
				y += m.faceNormal[3 * g + 1];
				z += m.faceNormal[3 * g + 2];
				count++;
			}
			next[3 * f] = x / count;
			next[3 * f + 1] = y / count;
			next[3 * f + 2] = z / count;
		}
		m.faceNormal.set(next);
	}
	m.imark++;
}

function pack(r: number, g: number, b: number, a: number): number {
	const clamp = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
	return (clamp(r) | (clamp(g) << 8) | (clamp(b) << 16) | (clamp(a) << 24)) >>> 0;
}

/** The set of vertices sharing an edge with each vertex. */
function vertexNeighbours(m: CMeshO): number[][] {
	const sets: Array<Set<number>> = Array.from({ length: m.vertSize }, () => new Set<number>());
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = m.fv(f, k);
			const b = m.fv(f, (k + 1) % 3);
			sets[a].add(b);
			sets[b].add(a);
		}
	}
	return sets.map((s) => [...s]);
}

export const Smooth = {
	vertexCoordLaplacian,
	vertexCoordTaubin,
	vertexCoordLaplacianHC,
	vertexCoordScaleDependentLaplacian,
	vertexQualityLaplacian,
	vertexColorLaplacian,
	faceNormalLaplacianFF,
} as const;
