/**
 * `UpdateNormal` — per-face and per-vertex normals.
 *
 * Follows VCGLib's convention that an *unnormalised* face normal is the full
 * cross product, whose length is twice the triangle's area. That is not an
 * oversight: summing unnormalised face normals onto their vertices is what
 * makes `perVertex` area-weighted, which is the behaviour MeshLab's filters
 * expect. Use the `…Normalized` variants when unit length is wanted.
 */
import { safeAcos } from "../../math/base.ts";
import { addAt, normalizeAt, setAt } from "../../math/vec3.ts";
import type { CMeshO } from "../cmesho.ts";

/** Writes the unnormalised cross product of face `f`'s edges into `out`. */
export function faceNormalOf(m: CMeshO, f: number, out: Float64Array | number[]): void {
	const a = m.faceVert[3 * f];
	const b = m.faceVert[3 * f + 1];
	const c = m.faceVert[3 * f + 2];
	const ux = m.vx(b) - m.vx(a);
	const uy = m.vy(b) - m.vy(a);
	const uz = m.vz(b) - m.vz(a);
	const vx = m.vx(c) - m.vx(a);
	const vy = m.vy(c) - m.vy(a);
	const vz = m.vz(c) - m.vz(a);
	out[0] = uy * vz - uz * vy;
	out[1] = uz * vx - ux * vz;
	out[2] = ux * vy - uy * vx;
}

/** Twice the area of face `f`. */
export function faceDoubleArea(m: CMeshO, f: number): number {
	const n = scratch;
	faceNormalOf(m, f, n);
	return Math.hypot(n[0], n[1], n[2]);
}

const scratch = new Float64Array(3);

/** Per-face normals, unnormalised (length = 2 × area). */
export function perFace(m: CMeshO): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		setAt(m.faceNormal, f, scratch[0], scratch[1], scratch[2]);
	}
}

export function normalizePerFace(m: CMeshO): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		normalizeAt(m.faceNormal, f);
	}
}

export function perFaceNormalized(m: CMeshO): void {
	perFace(m);
	normalizePerFace(m);
}

export function perVertexClear(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		setAt(m.vertNormal, v, 0, 0, 0);
	}
}

/**
 * Per-vertex normals as the sum of the incident faces' unnormalised normals —
 * i.e. area-weighted. Not normalised.
 */
export function perVertex(m: CMeshO): void {
	perVertexClear(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		for (let k = 0; k < 3; k++) {
			addAt(m.vertNormal, m.faceVert[3 * f + k], scratch[0], scratch[1], scratch[2]);
		}
	}
}

/**
 * Per-vertex normals as the plain mean of the incident faces' *unit* normals.
 *
 * The one scheme with no weighting at all, which makes it the one that a
 * dense fan of slivers can skew: ten thin triangles on one side of a vertex
 * outvote one large one on the other.
 */
export function perVertexSimpleAverage(m: CMeshO): void {
	perVertexClear(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		const len = Math.hypot(scratch[0], scratch[1], scratch[2]);
		if (len === 0) continue;
		for (let k = 0; k < 3; k++) {
			addAt(
				m.vertNormal,
				m.faceVert[3 * f + k],
				scratch[0] / len,
				scratch[1] / len,
				scratch[2] / len,
			);
		}
	}
}

/**
 * Nelson Max's weighting: each face counts for `1 / (|e1|² · |e2|²)` times its
 * unnormalised normal, where the two edges are the ones meeting at the vertex.
 *
 * The point of it is exactness on a sphere — the weights are chosen so that a
 * polyhedron inscribed in a sphere reproduces the sphere's normal at every
 * vertex, whatever the triangulation.
 *
 * Weights for Computing Vertex Normals from Facet Normals, N. Max, JGT 1999.
 */
export function perVertexNelsonMaxWeighted(m: CMeshO): void {
	perVertexClear(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		for (let k = 0; k < 3; k++) {
			const v = m.faceVert[3 * f + k];
			const a = m.faceVert[3 * f + ((k + 1) % 3)];
			const b = m.faceVert[3 * f + ((k + 2) % 3)];
			const e1 = (m.vx(a) - m.vx(v)) ** 2 + (m.vy(a) - m.vy(v)) ** 2 + (m.vz(a) - m.vz(v)) ** 2;
			const e2 = (m.vx(b) - m.vx(v)) ** 2 + (m.vy(b) - m.vy(v)) ** 2 + (m.vz(b) - m.vz(v)) ** 2;
			const denominator = e1 * e2;
			if (denominator === 0) continue;
			addAt(
				m.vertNormal,
				v,
				scratch[0] / denominator,
				scratch[1] / denominator,
				scratch[2] / denominator,
			);
		}
	}
}

export function normalizePerVertex(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		normalizeAt(m.vertNormal, v);
	}
}

/**
 * Per-vertex normals weighted by the incident angle rather than by area.
 *
 * Less sensitive than area weighting to how a flat region happens to be
 * triangulated, which is why MeshLab offers it as an alternative.
 */
export function perVertexAngleWeighted(m: CMeshO): void {
	perVertexClear(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		const len = Math.hypot(scratch[0], scratch[1], scratch[2]);
		if (len === 0) continue;
		const nx = scratch[0] / len;
		const ny = scratch[1] / len;
		const nz = scratch[2] / len;
		for (let k = 0; k < 3; k++) {
			const v = m.faceVert[3 * f + k];
			const p = m.faceVert[3 * f + ((k + 1) % 3)];
			const q = m.faceVert[3 * f + ((k + 2) % 3)];
			let ax = m.vx(p) - m.vx(v);
			let ay = m.vy(p) - m.vy(v);
			let az = m.vz(p) - m.vz(v);
			let bx = m.vx(q) - m.vx(v);
			let by = m.vy(q) - m.vy(v);
			let bz = m.vz(q) - m.vz(v);
			const la = Math.hypot(ax, ay, az);
			const lb = Math.hypot(bx, by, bz);
			if (la === 0 || lb === 0) continue;
			ax /= la;
			ay /= la;
			az /= la;
			bx /= lb;
			by /= lb;
			bz /= lb;
			const angle = safeAcos(ax * bx + ay * by + az * bz);
			addAt(m.vertNormal, v, nx * angle, ny * angle, nz * angle);
		}
	}
}

/** Both normals recomputed and normalised — the usual one-liner. */
export function perVertexNormalizedPerFaceNormalized(m: CMeshO): void {
	perVertexPerFace(m);
	normalizePerVertex(m);
	normalizePerFace(m);
}

/** Both normals recomputed in one pass over the faces, neither normalised. */
export function perVertexPerFace(m: CMeshO): void {
	perVertexClear(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		faceNormalOf(m, f, scratch);
		setAt(m.faceNormal, f, scratch[0], scratch[1], scratch[2]);
		for (let k = 0; k < 3; k++) {
			addAt(m.vertNormal, m.faceVert[3 * f + k], scratch[0], scratch[1], scratch[2]);
		}
	}
}

export const UpdateNormal = {
	faceNormalOf,
	faceDoubleArea,
	perFace,
	perFaceNormalized,
	normalizePerFace,
	perVertex,
	perVertexClear,
	perVertexAngleWeighted,
	perVertexSimpleAverage,
	perVertexNelsonMaxWeighted,
	normalizePerVertex,
	perVertexPerFace,
	perVertexNormalizedPerFaceNormalized,
} as const;
