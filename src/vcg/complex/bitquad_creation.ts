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

/** The index of a face's single hidden edge, or -1 when it has none. */
function fauxIndex(m: CMeshO, f: number): number {
	for (let e = 0; e < 3; e++) if (isFaux(m, f, e)) return e;
	return -1;
}

/**
 * Flips the hidden diagonal of the quad containing `f`, then re-hides whichever
 * edge the two halves now share.
 *
 * The quad keeps its four corners and its identity; only which diagonal splits
 * it changes. That is what lets the pairing search reshape the mesh without
 * adding or removing anything.
 */
function flipDiagonal(m: CMeshO, f: number): boolean {
	const faux = fauxIndex(m, f);
	if (faux < 0) return false;
	const g = m.ffp(f, faux);
	if (g === f || m.isFaceD(g)) return false;
	if (!flipDiagonalEdge(m, f, faux)) return false;

	faceFace(m);
	m.faceFlags[f] &= ~FaceFlag.FAUX012;
	m.faceFlags[g] &= ~FaceFlag.FAUX012;
	for (let e = 0; e < 3; e++) {
		if (m.ffp(f, e) === g) setFaux(m, f, e);
		if (m.ffp(g, e) === f) setFaux(m, g, e);
	}
	return true;
}

/** The bare index rewrite of an edge flip; the caller rebuilds adjacency. */
function flipDiagonalEdge(m: CMeshO, f: number, e: number): boolean {
	if (m.isBorderFF(f, e)) return false;
	const g = m.ffp(f, e);
	const ge = m.ffi(f, e);
	if (g === f) return false;
	const a = m.fv(f, e);
	const b = m.fv(f, (e + 1) % 3);
	const cf = m.fv(f, (e + 2) % 3);
	const cg = m.fv(g, (ge + 2) % 3);
	if (cf === cg) return false;
	// The new edge must not already exist, or the mesh becomes non-manifold.
	for (let h = 0; h < m.faceSize; h++) {
		if (m.isFaceD(h) || h === f || h === g) continue;
		for (let k = 0; k < 3; k++) {
			const p = m.fv(h, k);
			const q = m.fv(h, (k + 1) % 3);
			if ((p === cf && q === cg) || (p === cg && q === cf)) return false;
		}
	}
	m.setFace(f, cf, a, cg);
	m.setFace(g, cg, b, cf);
	return true;
}

/** Edge-distance from `from` to every reachable face, or -1. */
function faceDistances(m: CMeshO, from: number, maxDistance: number): Int32Array {
	const dist = new Int32Array(m.faceSize).fill(-1);
	dist[from] = 0;
	let frontier = [from];
	for (let d = 1; d <= maxDistance && frontier.length > 0; d++) {
		const next: number[] = [];
		for (const f of frontier) {
			for (let e = 0; e < 3; e++) {
				if (m.isBorderFF(f, e)) continue;
				const g = m.ffp(f, e);
				if (m.isFaceD(g) || dist[g] >= 0) continue;
				dist[g] = d;
				next.push(g);
			}
		}
		frontier = next;
	}
	return dist;
}

/**
 * Makes the mesh pure quads by flipping edges, adding nothing.
 *
 * Two unpaired triangles cannot always be joined directly — they may be far
 * apart, with quads in between. The trick is that a quad's diagonal can be
 * flipped, which moves the "unpaired" state one quad along: the near half of
 * the quad marries the lonely triangle and the far half becomes lonely in its
 * turn. Repeat and the two lonely triangles walk toward each other until they
 * meet.
 *
 * This is an augmenting-path matching in disguise, and it inherits the same
 * limit: an odd number of faces has no perfect matching whatever the
 * connectivity, so at least one triangle must always be left. Returns whether
 * it managed to pair everything.
 */
/**
 * Splits one border face so the face count becomes even.
 *
 * Pairing consumes two faces at a time, so an odd count can never be perfectly
 * matched. Splitting a border edge in two costs one extra face and fixes the
 * parity. A closed mesh with an odd face count cannot be helped this way and is
 * left alone — there is no border edge to split.
 */
export function makeTriEvenBySplit(m: CMeshO): boolean {
	if (m.fn % 2 === 0) return false;
	if (m.ffFace === null) faceFace(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			if (!m.isBorderFF(f, e)) continue;
			const a = m.fv(f, e);
			const b = m.fv(f, (e + 1) % 3);
			const c = m.fv(f, (e + 2) % 3);
			const mid = Allocator.addVertices(m, 1);
			m.setVert(mid, (m.vx(a) + m.vx(b)) / 2, (m.vy(a) + m.vy(b)) / 2, (m.vz(a) + m.vz(b)) / 2);
			const extra = Allocator.addFaces(m, 1);
			m.setFace(f, a, mid, c);
			m.setFace(extra, mid, b, c);
			m.faceFlags[extra] &= ~FaceFlag.FAUX012;
			faceFace(m);
			return true;
		}
	}
	return false;
}

export function makePureByFlip(m: CMeshO, maxDistance = 10000): boolean {
	if (m.ffFace === null) faceFace(m);
	const lonely = (): number => {
		for (let f = 0; f < m.faceSize; f++) {
			if (!m.isFaceD(f) && !anyFaux(m, f)) return f;
		}
		return -1;
	};

	// Bounded because each successful round removes two lonely triangles, and
	// a failed one gives up on that triangle for good.
	const abandoned = new Set<number>();
	for (let round = 0; round < m.faceSize + 1; round++) {
		let ta = -1;
		for (let f = 0; f < m.faceSize; f++) {
			if (!m.isFaceD(f) && !anyFaux(m, f) && !abandoned.has(f)) {
				ta = f;
				break;
			}
		}
		if (ta < 0) break;

		const dist = faceDistances(m, ta, maxDistance);
		let tb = -1;
		let best = Number.POSITIVE_INFINITY;
		for (let f = 0; f < m.faceSize; f++) {
			if (f === ta || m.isFaceD(f) || anyFaux(m, f) || dist[f] < 0) continue;
			if (dist[f] < best) {
				best = dist[f];
				tb = f;
			}
		}
		if (tb < 0) {
			abandoned.add(ta);
			continue;
		}
		if (!walkTogether(m, ta, tb, dist)) abandoned.add(ta);
	}
	return lonely() < 0;
}

/**
 * Walks `tb`'s loneliness along the quads toward `ta` until the two meet.
 *
 * Each step: find the neighbouring quad that is closer to `ta`, flip its
 * diagonal if the half facing `tb` is not the one that should marry it, pair
 * `tb` with that near half, and continue from the far half.
 */
function walkTogether(m: CMeshO, ta: number, tb: number, dist: Int32Array): boolean {
	let current = tb;
	for (let guard = 0; guard <= dist.length; guard++) {
		// Adjacent to another lonely triangle? Then simply marry it.
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(current, e)) continue;
			const g = m.ffp(current, e);
			if (m.isFaceD(g) || anyFaux(m, g) || g === current) continue;
			setFaux(m, current, e);
			setFaux(m, g, m.ffi(current, e));
			return true;
		}

		// Otherwise step through the quad that gets closest to `ta`.
		let bestEdge = -1;
		let bestDist = dist[current] < 0 ? Number.POSITIVE_INFINITY : dist[current];
		for (let e = 0; e < 3; e++) {
			if (m.isBorderFF(current, e)) continue;
			const g = m.ffp(current, e);
			if (m.isFaceD(g) || dist[g] < 0) continue;
			if (dist[g] < bestDist) {
				bestDist = dist[g];
				bestEdge = e;
			}
		}
		if (bestEdge < 0) return false;

		const near = m.ffp(current, bestEdge);
		const nearFaux = fauxIndex(m, near);
		if (nearFaux < 0) return false;
		// The quad's far half is the one across its hidden diagonal. If the
		// diagonal separates `current` from the wrong half, flip it first.
		let far = m.ffp(near, nearFaux);
		if (far === ta || dist[far] > dist[near]) {
			if (!flipDiagonal(m, near)) return false;
			const flipped = fauxIndex(m, near);
			if (flipped < 0) return false;
			far = m.ffp(near, flipped);
		}
		if (far === near || m.isFaceD(far)) return false;

		// Dissolve the quad and re-pair its near half with `current`.
		m.faceFlags[near] &= ~FaceFlag.FAUX012;
		m.faceFlags[far] &= ~FaceFlag.FAUX012;
		faceFace(m);
		let joined = -1;
		for (let e = 0; e < 3; e++) {
			if (m.ffp(current, e) === near) joined = e;
		}
		if (joined < 0) return false;
		setFaux(m, current, joined);
		setFaux(m, near, m.ffi(current, joined));
		current = far;
		if (current === ta) return true;
	}
	return false;
}

export const BitQuadCreation = {
	isTriQuadOnly,
	quadQuality,
	makeDominant,
	makePureByRefine,
	makePureByFlip,
	makeTriEvenBySplit,
} as const;
