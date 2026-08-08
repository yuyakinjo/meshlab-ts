/**
 * `vcg/complex/algorithms/bitquad_creation.h` — turning a triangle mesh into a
 * quad or quad-dominant one.
 *
 * The representation is the one the rest of the library already uses: a quad is
 * two triangles joined by a *faux* edge, tagged with a per-face-edge flag and
 * otherwise an ordinary edge. Nothing about the mesh's storage changes; making
 * a quad mesh is a matter of deciding which edges to hide.
 *
 * Two strategies live here.
 *
 * - {@link makeDominant} hides edges and moves nothing. It pairs triangles
 *   greedily by how square the resulting quad would be, so it is fast and
 *   lossless but leaves behind whatever triangles could not be paired — an odd
 *   count guarantees at least one.
 * - {@link makePureByRefine} refines instead, and gets *every* face as a quad.
 *   The construction is worth knowing: split each triangle at its centroid into
 *   three, and every original edge becomes the diagonal of the quad formed by
 *   the two sub-triangles either side of it. So the mesh triples in triangle
 *   count and every original edge turns into a faux diagonal.
 */
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import { FaceFlag, fauxBit } from "./flags.ts";
import { faceFace } from "./update/topology.ts";

const isFaux = (m: CMeshO, f: number, e: number): boolean => (m.faceFlags[f] & fauxBit(e)) !== 0;

const setFaux = (m: CMeshO, f: number, e: number): void => {
	m.faceFlags[f] |= fauxBit(e);
};

const clearFaux = (m: CMeshO, f: number, e: number): void => {
	m.faceFlags[f] &= ~fauxBit(e);
};

const anyFaux = (m: CMeshO, f: number): boolean => (m.faceFlags[f] & FaceFlag.FAUX012) !== 0;

/** True when no face carries more than one faux edge. */
export function isTriQuadOnly(m: CMeshO): boolean {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		let count = 0;
		for (let e = 0; e < 3; e++) if (isFaux(m, f, e)) count++;
		if (count > 1) return false;
	}
	return true;
}

const point = (m: CMeshO, v: number): number[] => [m.vx(v), m.vy(v), m.vz(v)];

/** The cosine of the angle at `b` in the corner `a`–`b`–`c`. */
function cosAt(a: readonly number[], b: readonly number[], c: readonly number[]): number {
	const u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
	const v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
}

/**
 * How square the quad `a b c d` is, from 0 (degenerate) to 4 (a rectangle).
 *
 * One point per corner for being a right angle. Deliberately about the angles
 * only and not about the side lengths: a long thin rectangle still renders and
 * subdivides cleanly, whereas a quad with a reflex corner does not.
 */
export function quadQuality(
	a: readonly number[],
	b: readonly number[],
	c: readonly number[],
	d: readonly number[],
): number {
	return (
		1 -
		Math.abs(cosAt(a, b, c)) +
		(1 - Math.abs(cosAt(b, c, d))) +
		(1 - Math.abs(cosAt(c, d, a))) +
		(1 - Math.abs(cosAt(d, a, b)))
	);
}

/** The quality of the quad that hiding edge `e` of face `f` would produce. */
function qualityAcross(m: CMeshO, f: number, e: number): number {
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	return quadQuality(
		point(m, m.fv(f, e)),
		point(m, m.fv(g, (ge + 2) % 3)),
		point(m, m.fv(f, (e + 1) % 3)),
		point(m, m.fv(f, (e + 2) % 3)),
	);
}

/**
 * One greedy pass, pairing each unpaired face with its best neighbour.
 *
 * With `override`, a face may steal a neighbour that is already paired, but
 * only if it can offer a better score than the neighbour currently has — which
 * is what lets later passes improve on the first one's arbitrary order.
 * `faceQuality` records each face's current score so that comparison is
 * possible at all.
 */
function makeDominantPass(m: CMeshO, override: boolean): void {
	const quality = m.faceQuality;
	const score = (f: number) => (quality === null ? 0 : quality[f]);
	const setScore = (f: number, s: number) => {
		if (quality !== null) quality[f] = s;
	};

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (!override && anyFaux(m, f)) continue;

		let bestEdge = -1;
		let bestScore = score(f);
		for (let e = 0; e < 3; e++) {
			// A border has nothing across it to pair with.
			if (m.isBorderFF(f, e)) continue;
			const g = m.ffp(f, e);
			if (m.isFaceD(g)) continue;
			if (!override && anyFaux(m, g)) continue;
			const s = qualityAcross(m, f, e);
			// Do not steal a partner that is doing better where it is.
			if (override && s < score(g)) continue;
			if (s > bestScore) {
				bestScore = s;
				bestEdge = e;
			}
		}
		if (bestEdge < 0) continue;

		const g = m.ffp(f, bestEdge);
		if (override) {
			// Both faces have to be freed first; each break-up leaves the
			// abandoned partner unpaired and scoring zero.
			for (const face of [g, f]) {
				for (let e = 0; e < 3; e++) {
					if (!isFaux(m, face, e)) continue;
					clearFaux(m, face, e);
					const other = m.ffp(face, e);
					clearFaux(m, other, m.ffi(face, e));
					setScore(other, 0);
				}
			}
		}
		setFaux(m, f, bestEdge);
		setFaux(m, g, m.ffi(f, bestEdge));
		setScore(f, bestScore);
		setScore(g, bestScore);
	}
}

/**
 * Pairs triangles into quads without moving anything.
 *
 * `level` buys quality with passes: 0 is one greedy sweep and leaves the fewest
 * triangles; higher levels re-run with stealing allowed, which breaks up poor
 * pairings in favour of better ones and so tends to leave a few more triangles
 * behind but squarer quads.
 */
export function makeDominant(m: CMeshO, level: number): void {
	if (m.ffFace === null) faceFace(m);
	for (let f = 0; f < m.faceSize; f++) {
		m.faceFlags[f] &= ~FaceFlag.FAUX012;
		if (m.faceQuality !== null) m.faceQuality[f] = 0;
	}
	makeDominantPass(m, false);
	if (level > 0) makeDominantPass(m, true);
	if (level > 1) makeDominantPass(m, true);
	if (level > 0) makeDominantPass(m, false);
}

/**
 * Makes every face a quad by splitting each triangle at its centroid.
 *
 * Each original triangle becomes three, and each original *edge* becomes the
 * faux diagonal of the quad made from the two sub-triangles beside it. On a
 * closed mesh the result is pure quads. A border edge has only one sub-triangle
 * beside it and so cannot be paired — those stay triangles, which is the one
 * way the result falls short of pure and is reported by the return value.
 */
export function makePureByRefine(m: CMeshO): { quads: number; triangles: number } {
	if (m.ffFace === null) faceFace(m);
	// A face with two or more faux edges is a polygon larger than a quad, which
	// this construction has no answer for.
	if (!isTriQuadOnly(m)) {
		throw new Error("makePureByRefine needs a mesh of triangles and quads only");
	}

	const originalFaces: number[] = [];
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) originalFaces.push(f);
	if (originalFaces.length === 0) return { quads: 0, triangles: 0 };

	// Centroid per original face, and where the three sub-triangles landed.
	// `sub[f][e]` is the sub-triangle spanning edge `e` of face `f`.
	const centroid = new Int32Array(m.faceSize).fill(-1);
	const sub: Int32Array = new Int32Array(3 * m.faceSize).fill(-1);

	const firstVert = Allocator.addVertices(m, originalFaces.length);
	// Two extra faces per original: it becomes three.
	const firstFace = Allocator.addFaces(m, 2 * originalFaces.length);

	originalFaces.forEach((f, i) => {
		const c = firstVert + i;
		centroid[f] = c;
		const v = [0, 1, 2].map((k) => m.fv(f, k));
		m.setVert(
			c,
			(m.vx(v[0]) + m.vx(v[1]) + m.vx(v[2])) / 3,
			(m.vy(v[0]) + m.vy(v[1]) + m.vy(v[2])) / 3,
			(m.vz(v[0]) + m.vz(v[1]) + m.vz(v[2])) / 3,
		);
		// Reuse the original slot for the first sub-triangle and take two fresh
		// slots for the others, so no face index the caller held goes stale.
		const slots = [f, firstFace + 2 * i, firstFace + 2 * i + 1];
		for (let e = 0; e < 3; e++) {
			// The sub-triangle on edge e spans that edge and the centroid.
			m.setFace(slots[e], v[e], v[(e + 1) % 3], c);
			sub[3 * f + e] = slots[e];
			m.faceFlags[slots[e]] &= ~FaceFlag.FAUX012;
		}
	});

	// Now hide every original interior edge. Edge e of face f is edge 0 of its
	// sub-triangle, and the neighbour's matching sub-triangle meets it there.
	let quads = 0;
	for (const f of originalFaces) {
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(f, e)) continue;
			const g = m.ffp(f, e);
			if (centroid[g] < 0) continue;
			// Once per shared edge.
			if (g < f) continue;
			setFaux(m, sub[3 * f + e], 0);
			setFaux(m, sub[3 * g + m.ffi(f, e)], 0);
			quads++;
		}
	}

	// The adjacency described the old faces and is meaningless now.
	m.imark++;
	const triangles = 3 * originalFaces.length - 2 * quads;
	return { quads, triangles };
}

export const BitQuadCreation = {
	isTriQuadOnly,
	quadQuality,
	makeDominant,
	makePureByRefine,
} as const;
