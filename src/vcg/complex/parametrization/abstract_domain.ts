/**
 * The abstract domain of Pietroni, Tarini and Cignoni's isoparametrisation.
 *
 * The idea is to simplify a mesh down to a very coarse base domain while
 * *never losing track of where the original surface went*. Each face of the
 * domain carries a list of original vertices, each pinned by a barycentric
 * coordinate inside that face. Simplify far enough and you have a handful of
 * triangles that still address every point of the input — a parametrisation
 * whose domain is a mesh rather than a square, which is what lets it stay
 * almost isometric where a single flat chart could not.
 *
 * Everything here hangs on one operation: collapsing a domain edge and
 * carrying the pinned points across. That is {@link collapseWithParametrization},
 * and its six steps are
 *
 *   1. flatten the two vertices' combined star into the plane,
 *   2. record each pinned point's position in that flat patch,
 *   3. collapse the edge,
 *   4. flatten the new, smaller star,
 *   5. find each recorded point in the new patch, and
 *   6. write it back as a new (face, barycentric) pair.
 *
 * Steps 1 and 4 are why the parametrisation stays good: the flattening is a
 * genuine low-distortion map of that neighbourhood, so a point's position in
 * it means something. A collapse that cannot be carried across — because a
 * point lands outside the new patch — is refused rather than approximated,
 * which is what keeps the domain a faithful cover of the surface.
 *
 * This is the machinery under `filter_isoparametrization`; the filters
 * themselves are not wired up yet.
 */

import { Allocator } from "../allocator.ts";
import { CMeshO } from "../cmesho.ts";
import { buildVertexFaces, linkCondition, sharedFaces } from "../edge_ops.ts";
import { FaceFlag } from "../flags.ts";

/** One original vertex, pinned inside a domain face. */
export interface PinnedVertex {
	readonly vertex: number;
	/** Barycentric weights on the face's three corners; they sum to one. */
	readonly bary: readonly [number, number, number];
}

/**
 * A coarse mesh that still addresses every vertex of a finer one.
 *
 * `base` starts as a copy of the high-resolution mesh and shrinks; `hires`
 * never changes. The invariant that matters is that every live vertex of
 * `hires` appears in exactly one face's pin list, so the domain always covers
 * the whole surface.
 */
export class AbstractDomain {
	readonly base: CMeshO;
	readonly hires: CMeshO;
	/** Per base face, the original vertices it currently carries. */
	readonly pinned: Array<PinnedVertex[]>;

	private constructor(base: CMeshO, hires: CMeshO, pinned: Array<PinnedVertex[]>) {
		this.base = base;
		this.hires = hires;
		this.pinned = pinned;
	}

	/**
	 * Starts a domain off as the mesh itself.
	 *
	 * Each face begins carrying its own three corners, at the corners of the
	 * barycentric triangle. Every collapse from here shrinks the domain and
	 * moves pins around; nothing ever adds one.
	 */
	static from(hires: CMeshO): AbstractDomain {
		const base = copyMesh(hires);
		const pinned: Array<PinnedVertex[]> = Array.from({ length: base.faceSize }, () => []);
		// A vertex belongs to several faces; it is pinned in the first one that
		// claims it, since one pin per vertex is the invariant.
		const claimed = new Uint8Array(hires.vertSize);
		const unit: Array<[number, number, number]> = [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
		];
		for (let f = 0; f < base.faceSize; f++) {
			if (base.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = base.fv(f, k);
				if (claimed[v] === 1) continue;
				claimed[v] = 1;
				pinned[f].push({ vertex: v, bary: unit[k] });
			}
		}
		return new AbstractDomain(base, hires, pinned);
	}

	/** How many original vertices the domain still accounts for. */
	pinnedCount(): number {
		let n = 0;
		for (let f = 0; f < this.base.faceSize; f++) {
			if (!this.base.isFaceD(f)) n += this.pinned[f].length;
		}
		return n;
	}

	/** The 3D point a pin addresses, by interpolating its face's corners. */
	positionOf(face: number, bary: readonly number[]): [number, number, number] {
		const out: [number, number, number] = [0, 0, 0];
		for (let k = 0; k < 3; k++) {
			const v = this.base.fv(face, k);
			out[0] += this.base.vx(v) * bary[k];
			out[1] += this.base.vy(v) * bary[k];
			out[2] += this.base.vz(v) * bary[k];
		}
		return out;
	}
}

// ---- star flattening ------------------------------------------------------

/** A star flattened into the plane: its faces, and a UV for each of its vertices. */
export interface FlatStar {
	/** Faces of the base mesh making up the star, in no particular order. */
	readonly faces: number[];
	/** UV per base-mesh vertex index; only the star's vertices are filled. */
	readonly uv: Map<number, [number, number]>;
	/** The boundary loop, in the order it was laid onto the polygon. */
	readonly boundary: number[];
}

/**
 * Flattens the star of one or two vertices onto a regular polygon.
 *
 * `ParametrizeStarEquilateral` upstream. The boundary loop goes onto a regular
 * n-gon and the interior vertex — there is exactly one, or two before a
 * collapse — sits at the centroid. A regular polygon rather than the patch's
 * own outline because the star is about to be compared with a *different*
 * star, the one left after the collapse: both are flattened the same way, so
 * the two flat pictures are directly comparable.
 *
 * Returns null when the star is not a disk, which happens at a non-manifold
 * vertex and means the collapse must be refused.
 */
export function flattenStar(
	cm: CMeshO,
	centres: readonly number[],
	vertFaces: ReadonlyArray<Set<number>>,
	reuse?: ReadonlyMap<number, [number, number]>,
): FlatStar | null {
	const faces: number[] = [];
	const seen = new Set<number>();
	for (const c of centres) {
		for (const f of vertFaces[c] ?? []) {
			if (cm.isFaceD(f) || seen.has(f)) continue;
			seen.add(f);
			faces.push(f);
		}
	}
	if (faces.length === 0) return null;

	const loop = starBoundary(cm, faces, centres);
	if (loop === null) return null;

	const uv = new Map<number, [number, number]>();
	const n = loop.length;
	// Space the boundary by chord length rather than evenly. An equilateral
	// layout is simpler, but it distorts a star whose spokes differ in length,
	// and that distortion is exactly what smears a transferred point.
	const lengths: number[] = [];
	let perimeter = 0;
	for (let i = 0; i < n; i++) {
		const a = loop[i];
		const b = loop[(i + 1) % n];
		const d = Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)) || 1;
		lengths.push(d);
		perimeter += d;
	}
	let travelled = 0;
	for (let i = 0; i < n; i++) {
		// `reuse` carries the boundary placement of a *previous* flattening.
		// Without it the two pictures would be laid out from an arbitrary
		// starting vertex and the same UV would mean different places in
		// each, which is the difference between transferring a point and
		// scattering it.
		const already = reuse?.get(loop[i]);
		if (already !== undefined) {
			uv.set(loop[i], [already[0], already[1]]);
		} else {
			const angle = (2 * Math.PI * travelled) / perimeter;
			uv.set(loop[i], [Math.cos(angle), Math.sin(angle)]);
		}
		travelled += lengths[i];
	}

	// The interior — one centre after a collapse, two before — is relaxed to
	// the mean-value average of its neighbours, the same construction the
	// disk parametrisation uses. Placing it at the origin instead would be a
	// guess, and the whole point of the flattening is that positions in it
	// mean something.
	placeCentres(cm, faces, centres, uv);
	return { faces, uv, boundary: loop };
}

/**
 * Relaxes the star's interior vertices onto the mean value of their
 * neighbours, holding the boundary fixed.
 *
 * With one or two unknowns this converges in a handful of sweeps, so there is
 * no solver here — just the same fixed-point iteration the disk
 * parametrisation runs, stopped when it stops moving.
 */
function placeCentres(
	cm: CMeshO,
	faces: readonly number[],
	centres: readonly number[],
	uv: Map<number, [number, number]>,
): void {
	// Neighbours and mean-value weights, from the star's own 3D geometry.
	const rows = centres.map((c) => {
		const neighbours = new Map<number, number>();
		for (const f of faces) {
			let at = -1;
			for (let k = 0; k < 3; k++) if (cm.fv(f, k) === c) at = k;
			if (at < 0) continue;
			const a = cm.fv(f, (at + 1) % 3);
			const b = cm.fv(f, (at + 2) % 3);
			// Half the mean-value weight from each incident face: tan of the
			// half-angle at the centre, over the edge length.
			for (const [other, third] of [
				[a, b],
				[b, a],
			]) {
				const angle = angleAt(cm, c, other, third);
				const edge = distance3(cm, c, other);
				if (edge === 0) continue;
				neighbours.set(other, (neighbours.get(other) ?? 0) + Math.tan(angle / 2) / edge);
			}
		}
		return { centre: c, neighbours };
	});

	// Start each centre at the average of its boundary neighbours, then relax.
	for (const row of rows) {
		let u = 0;
		let v = 0;
		let count = 0;
		for (const n of row.neighbours.keys()) {
			const at = uv.get(n);
			if (at === undefined) continue;
			u += at[0];
			v += at[1];
			count++;
		}
		uv.set(row.centre, count === 0 ? [0, 0] : [u / count, v / count]);
	}
	for (let sweep = 0; sweep < 32; sweep++) {
		let worst = 0;
		for (const row of rows) {
			let u = 0;
			let v = 0;
			let total = 0;
			for (const [n, w] of row.neighbours) {
				const at = uv.get(n);
				if (at === undefined) continue;
				u += at[0] * w;
				v += at[1] * w;
				total += w;
			}
			if (total === 0) continue;
			const current = uv.get(row.centre) as [number, number];
			const next: [number, number] = [u / total, v / total];
			worst = Math.max(worst, Math.abs(next[0] - current[0]), Math.abs(next[1] - current[1]));
			uv.set(row.centre, next);
		}
		if (worst < 1e-12) break;
	}
}

function angleAt(cm: CMeshO, at: number, a: number, b: number): number {
	const u = [cm.vx(a) - cm.vx(at), cm.vy(a) - cm.vy(at), cm.vz(a) - cm.vz(at)];
	const v = [cm.vx(b) - cm.vx(at), cm.vy(b) - cm.vy(at), cm.vz(b) - cm.vz(at)];
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	const d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
	return Math.acos(Math.min(1, Math.max(-1, d)));
}

function distance3(cm: CMeshO, a: number, b: number): number {
	return Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
}

/**
 * The boundary loop of a star: the edges opposite the centre vertices.
 *
 * Returns null when those edges do not form one closed loop, which is the
 * test for "this star is a disk" and the reason a non-manifold vertex is
 * refused rather than mangled.
 */
function starBoundary(
	cm: CMeshO,
	faces: readonly number[],
	centres: readonly number[],
): number[] | null {
	const isCentre = new Set(centres);
	const next = new Map<number, number>();
	for (const f of faces) {
		for (let e = 0; e < 3; e++) {
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			if (isCentre.has(a) || isCentre.has(b)) continue;
			if (next.has(a)) return null; // two ways out: not a disk
			next.set(a, b);
		}
	}
	if (next.size < 3) return null;

	const start = next.keys().next().value as number;
	const loop = [start];
	let at = next.get(start) as number;
	while (at !== start) {
		loop.push(at);
		const step = next.get(at);
		if (step === undefined) return null;
		at = step;
		if (loop.length > next.size) return null;
	}
	return loop.length === next.size ? loop : null;
}

/** True when any face of the flattened star is turned inside out. */
function hasFold(cm: CMeshO, star: FlatStar): boolean {
	let positive = 0;
	let negative = 0;
	for (const f of star.faces) {
		const a = star.uv.get(cm.fv(f, 0));
		const b = star.uv.get(cm.fv(f, 1));
		const c = star.uv.get(cm.fv(f, 2));
		if (a === undefined || b === undefined || c === undefined) return true;
		const area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
		if (area > 0) positive++;
		else if (area < 0) negative++;
		else return true;
	}
	return positive > 0 && negative > 0;
}

/** Where a UV point falls in a flattened star. */
interface Located {
	readonly face: number;
	readonly bary: [number, number, number];
}

/**
 * Finds the face of a flat star containing a UV point.
 *
 * Exact containment first; if nothing claims the point — it can sit a hair
 * outside on the boundary after the flattening changes shape — the nearest
 * face by clamped barycentric distance wins. That fallback is what stops a
 * rounding error at the rim from failing an otherwise good collapse, and it
 * moves the point by less than a texel of the patch.
 */
function locate(cm: CMeshO, star: FlatStar, u: number, v: number): Located | null {
	let best: Located | null = null;
	let bestPenalty = Number.POSITIVE_INFINITY;

	for (const f of star.faces) {
		const p = [0, 1, 2].map((k) => star.uv.get(cm.fv(f, k)));
		if (p[0] === undefined || p[1] === undefined || p[2] === undefined) continue;
		const a = p[0] as [number, number];
		const b = p[1] as [number, number];
		const c = p[2] as [number, number];
		const area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
		if (area === 0) continue;

		const w0 = ((b[0] - u) * (c[1] - v) - (c[0] - u) * (b[1] - v)) / area;
		const w1 = ((c[0] - u) * (a[1] - v) - (a[0] - u) * (c[1] - v)) / area;
		const w2 = 1 - w0 - w1;
		const penalty = Math.max(0, -w0) + Math.max(0, -w1) + Math.max(0, -w2);
		if (penalty === 0) return { face: f, bary: [w0, w1, w2] };
		if (penalty < bestPenalty) {
			bestPenalty = penalty;
			best = { face: f, bary: clampBary([w0, w1, w2]) };
		}
	}
	// A point far outside every face of the star is not a rounding error.
	return bestPenalty <= 0.25 ? best : null;
}

function clampBary(b: readonly number[]): [number, number, number] {
	const clamped = b.map((x) => Math.max(0, x));
	const total = clamped[0] + clamped[1] + clamped[2];
	if (total === 0) return [1 / 3, 1 / 3, 1 / 3];
	return [clamped[0] / total, clamped[1] / total, clamped[2] / total];
}

/** The UV of a pin inside its face, by interpolating the face's corner UVs. */
function pinUV(
	cm: CMeshO,
	star: FlatStar,
	face: number,
	bary: readonly number[],
): [number, number] | null {
	let u = 0;
	let v = 0;
	for (let k = 0; k < 3; k++) {
		const corner = star.uv.get(cm.fv(face, k));
		if (corner === undefined) return null;
		u += corner[0] * bary[k];
		v += corner[1] * bary[k];
	}
	return [u, v];
}

// ---- the collapse ---------------------------------------------------------

export interface CollapseResult {
	readonly ok: boolean;
	/** Why it was refused, when it was. */
	readonly reason?: string;
	/** How many pins moved to a new face. */
	readonly moved?: number;
}

/**
 * Collapses `v0` into `v1`, carrying every pinned vertex across.
 *
 * Refused, leaving the domain untouched, when the link condition fails, when
 * either star is not a disk, or when any pin cannot be found in the new
 * patch. The last is the interesting one: it means the collapse would have
 * lost part of the surface, and losing part of the surface is exactly what
 * this whole construction exists to prevent.
 */
export function collapseWithParametrization(
	domain: AbstractDomain,
	v0: number,
	v1: number,
	vertFaces: Array<Set<number>>,
): CollapseResult {
	const cm = domain.base;
	if (v0 === v1) return { ok: false, reason: "an edge needs two distinct vertices" };
	if (cm.isVertD(v0) || cm.isVertD(v1)) return { ok: false, reason: "a vertex is already deleted" };
	const sharedFaceList = sharedFaces(cm, vertFaces, v0, v1);
	if (!linkCondition(cm, vertFaces, v0, v1, sharedFaceList)) {
		return {
			ok: false,
			reason: "the link condition fails: the collapse would change the topology",
		};
	}

	const before = flattenStar(cm, [v0, v1], vertFaces);
	if (before === null) return { ok: false, reason: "the pre-collapse star is not a disk" };

	// Step 2: every pin's position in the flat patch, recorded before the
	// mesh changes underneath it.
	const carried: Array<{ vertex: number; u: number; v: number }> = [];
	for (const f of before.faces) {
		for (const pin of domain.pinned[f]) {
			const at = pinUV(cm, before, f, pin.bary);
			if (at === null)
				return { ok: false, reason: "a pinned point has no place in the flat patch" };
			carried.push({ vertex: pin.vertex, u: at[0], v: at[1] });
		}
	}

	// Step 3. Snapshot enough to undo, since a refusal must leave no trace.
	const snapshot = before.faces.map((f) => ({ f, pins: domain.pinned[f], live: !cm.isFaceD(f) }));
	const geometry = { x: cm.vx(v1), y: cm.vy(v1), z: cm.vz(v1) };
	const faceVerts = before.faces.map((f) => [cm.fv(f, 0), cm.fv(f, 1), cm.fv(f, 2)]);
	const vertFlags0 = cm.vertFlags[v0];

	const midpoint: [number, number, number] = [
		(cm.vx(v0) + cm.vx(v1)) / 2,
		(cm.vy(v0) + cm.vy(v1)) / 2,
		(cm.vz(v0) + cm.vz(v1)) / 2,
	];
	doCollapse(cm, vertFaces, v0, v1, midpoint);

	// The combined star's boundary loop is exactly the collapsed star's, so
	// the second flattening reuses the first's placement and the two flat
	// pictures share a coordinate system.
	const after = flattenStar(cm, [v1], vertFaces, before.uv);
	if (after === null) {
		undo(cm, domain, snapshot, faceVerts, v0, v1, geometry, vertFlags0, vertFaces);
		return { ok: false, reason: "the post-collapse star is not a disk" };
	}
	// A folded triangle means the flat picture is not injective, so locating
	// a point in it would give an answer that is merely arithmetically valid.
	if (hasFold(cm, after)) {
		undo(cm, domain, snapshot, faceVerts, v0, v1, geometry, vertFlags0, vertFaces);
		return { ok: false, reason: "the collapsed star folds over itself when flattened" };
	}

	// Steps 5 and 6: re-pin everything into the new patch.
	const rehomed: Array<{ face: number; pin: PinnedVertex }> = [];
	for (const point of carried) {
		const found = locate(cm, after, point.u, point.v);
		if (found === null) {
			undo(cm, domain, snapshot, faceVerts, v0, v1, geometry, vertFlags0, vertFaces);
			return { ok: false, reason: `vertex ${point.vertex} falls outside the collapsed patch` };
		}
		rehomed.push({ face: found.face, pin: { vertex: point.vertex, bary: found.bary } });
	}

	for (const f of before.faces) domain.pinned[f] = [];
	for (const { face, pin } of rehomed) domain.pinned[face].push(pin);
	return { ok: true, moved: rehomed.length };
}

/**
 * Restores the mesh and the pin lists after a refused collapse.
 *
 * A collapse is only known to be bad after it has been performed, so the undo
 * is not optional bookkeeping — it is what makes "refuse and move on" a real
 * option rather than a corrupted domain.
 */
function undo(
	cm: CMeshO,
	domain: AbstractDomain,
	snapshot: ReadonlyArray<{ f: number; pins: PinnedVertex[]; live: boolean }>,
	faceVerts: ReadonlyArray<number[]>,
	v0: number,
	v1: number,
	geometry: { x: number; y: number; z: number },
	vertFlags0: number,
	vertFaces: Array<Set<number>>,
): void {
	cm.vertFlags[v0] = vertFlags0;
	cm.vn++;
	cm.setVert(v1, geometry.x, geometry.y, geometry.z);
	snapshot.forEach((entry, i) => {
		if (entry.live && cm.isFaceD(entry.f)) {
			cm.faceFlags[entry.f] &= ~FaceFlag.DELETED;
			cm.fn++;
		}
		const verts = faceVerts[i];
		cm.setFace(entry.f, verts[0], verts[1], verts[2]);
		domain.pinned[entry.f] = entry.pins;
	});
	// The adjacency was edited in place; rebuild the two stars from scratch.
	for (const set of [vertFaces[v0], vertFaces[v1]]) set?.clear();
	for (const entry of snapshot) {
		if (cm.isFaceD(entry.f)) continue;
		for (let k = 0; k < 3; k++) vertFaces[cm.fv(entry.f, k)]?.add(entry.f);
	}
}

/** The mesh half of a collapse: rewire the faces, delete the pair's two. */
function doCollapse(
	cm: CMeshO,
	vertFaces: Array<Set<number>>,
	v0: number,
	v1: number,
	position: readonly number[],
): void {
	const shared: number[] = [];
	for (const f of vertFaces[v0]) {
		if (!cm.isFaceD(f) && vertFaces[v1].has(f)) shared.push(f);
	}
	const touching = [...vertFaces[v0]];

	cm.setVert(v1, position[0], position[1], position[2]);
	for (const f of touching) {
		if (cm.isFaceD(f)) continue;
		if (shared.includes(f)) {
			Allocator.deleteFace(cm, f);
			for (let k = 0; k < 3; k++) vertFaces[cm.fv(f, k)]?.delete(f);
			continue;
		}
		const verts = [0, 1, 2].map((k) => (cm.fv(f, k) === v0 ? v1 : cm.fv(f, k)));
		cm.setFace(f, verts[0], verts[1], verts[2]);
		vertFaces[v1].add(f);
	}
	vertFaces[v0].clear();
	Allocator.deleteVertex(cm, v0);
}

/** A structural copy: geometry and faces, nothing optional. */
function copyMesh(src: CMeshO): CMeshO {
	const out = new CMeshO();
	const remap = new Int32Array(src.vertSize).fill(-1);
	let live = 0;
	for (let v = 0; v < src.vertSize; v++) if (!src.isVertD(v)) live++;
	if (live === 0) return out;

	const first = Allocator.addVertices(out, live);
	let at = first;
	for (let v = 0; v < src.vertSize; v++) {
		if (src.isVertD(v)) continue;
		remap[v] = at;
		out.setVert(at, src.vx(v), src.vy(v), src.vz(v));
		at++;
	}
	for (let f = 0; f < src.faceSize; f++) {
		if (src.isFaceD(f)) continue;
		Allocator.addFace(out, remap[src.fv(f, 0)], remap[src.fv(f, 1)], remap[src.fv(f, 2)]);
	}
	return out;
}

/** Builds the vertex-to-face index the collapse keeps up to date. */
export function domainVertexFaces(domain: AbstractDomain): Array<Set<number>> {
	return buildVertexFaces(domain.base);
}
