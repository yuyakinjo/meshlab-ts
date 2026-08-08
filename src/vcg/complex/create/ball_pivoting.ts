/**
 * `vcg/complex/algorithms/create/ball_pivoting.h` and `advancing_front.h` —
 * reconstructing a surface by rolling a ball over a point cloud.
 *
 * The idea is entirely geometric and needs no solver: place a ball of a chosen
 * radius so that it rests on three points and contains none, and those three
 * points are a triangle of the surface. Then pivot the ball around each edge
 * of that triangle until it lands on a fourth point, and that is the next
 * triangle. Repeat until nothing more can be reached.
 *
 * What makes this worth having alongside Screened Poisson is that it
 * *interpolates* — every output vertex is an input point, no more and no less.
 * Poisson fits an implicit function and re-samples it, which smooths noise but
 * also moves every point and invents new ones. For a scan you trust, ball
 * pivoting keeps exactly what was measured.
 *
 * The radius is the whole parameter: gaps wider than the ball stay holes, and
 * detail finer than the ball is bridged over.
 */

import { KdTree } from "../../space/index/kdtree.ts";
import { Allocator } from "../allocator.ts";
import type { CMeshO } from "../cmesho.ts";
import { VertexFlag } from "../flags.ts";
import { UpdateBounding } from "../update/bounding.ts";
import { faceBorderFromNone, vertexBorderFromFace } from "../update/flag.ts";

/** How many neighbours each pivot query asks the tree for. VCG's 16. */
const QUERY_K = 16;

/**
 * An edge of the advancing front: the boundary between what has been
 * reconstructed and what has not.
 *
 * `v0`–`v1` is the edge itself and `v2` the third vertex of the triangle it
 * came from, which is what fixes the orientation to pivot in. The `next` and
 * `previous` links thread the edges into closed loops — one loop per hole in
 * the front — and are independent of the order the edges sit in the work list.
 */
interface FrontEdge {
	v0: number;
	v1: number;
	v2: number;
	/** In the live front rather than the dead list. */
	active: boolean;
	next: FrontEdge;
	previous: FrontEdge;
	/** Work-list links, distinct from the loop links above. */
	listNext: FrontEdge | null;
	listPrev: FrontEdge | null;
	/** Which work list holds this edge, or null once erased. */
	owner: EdgeList | null;
}

/**
 * A doubly-linked work list.
 *
 * Upstream uses `std::list` and splices edges between the front and the dead
 * list, keeping iterators valid throughout. The loop links point at edges, not
 * at list positions, so the two structures have to be able to move
 * independently — which is why the edges carry their own list pointers rather
 * than living in an array.
 */
class EdgeList {
	head: FrontEdge | null = null;
	tail: FrontEdge | null = null;
	size = 0;

	pushBack(e: FrontEdge): void {
		e.owner = this;
		e.listPrev = this.tail;
		e.listNext = null;
		if (this.tail !== null) this.tail.listNext = e;
		else this.head = e;
		this.tail = e;
		this.size++;
	}

	pushFront(e: FrontEdge): void {
		e.owner = this;
		e.listNext = this.head;
		e.listPrev = null;
		if (this.head !== null) this.head.listPrev = e;
		else this.tail = e;
		this.head = e;
		this.size++;
	}

	remove(e: FrontEdge): void {
		if (e.owner !== this) return;
		if (e.listPrev !== null) e.listPrev.listNext = e.listNext;
		else this.head = e.listNext;
		if (e.listNext !== null) e.listNext.listPrev = e.listPrev;
		else this.tail = e.listPrev;
		e.listPrev = null;
		e.listNext = null;
		e.owner = null;
		this.size--;
	}

	*[Symbol.iterator](): Generator<FrontEdge> {
		for (let e = this.head; e !== null; e = e.listNext) yield e;
	}
}

export interface BallPivotingOptions {
	/** Ball radius in mesh units. Zero asks for the automatic guess. */
	readonly radius?: number;
	/** Minimum edge length as a fraction of the radius. VCG's 0.2. */
	readonly clustering?: number;
	/** Stop pivoting past this crease angle, in radians. */
	readonly creaseAngle?: number;
	readonly progress?: (percent: number) => void;
}

export interface BallPivotingResult {
	readonly addedFaces: number;
	readonly radius: number;
}

/**
 * Rolls a ball over `m`'s vertices, adding faces.
 *
 * Existing faces are kept and their boundary becomes the initial front, so
 * running the filter again with a larger radius extends the same surface
 * rather than starting over — which is how upstream expects it to be used.
 */
export function ballPivoting(m: CMeshO, options: BallPivotingOptions = {}): BallPivotingResult {
	Allocator.compactEveryVector(m);
	if (m.vn < 3) return { addedFaces: 0, radius: options.radius ?? 0 };

	UpdateBounding.box(m);
	let radius = options.radius ?? 0;
	if (!(radius > 0)) {
		// VCG's guess: spread the points evenly over the box and take the
		// spacing that implies.
		radius = Math.sqrt((m.bbox.diagonal * m.bbox.diagonal) / m.vn);
	}
	const minEdge = (options.clustering ?? 0.2) * radius;
	const maxEdge = 1.8 * radius;
	const maxAngle = Math.cos(options.creaseAngle ?? Math.PI / 2);
	const progress = options.progress ?? (() => {});

	const centroid = [0, 0, 0];
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		centroid[0] += m.vx(v);
		centroid[1] += m.vy(v);
		centroid[2] += m.vz(v);
	}
	for (let k = 0; k < 3; k++) centroid[k] /= m.vn;

	const tree = new KdTree(m.vertCoord, m.vertSize);
	// `used` is VCG's private user bit: this vertex is too close to one already
	// placed to be worth considering again.
	const used = new Uint8Array(m.vertSize);
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.VISITED;
	// A vertex already carried by a face is off limits as a seed.
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) mark(m.fv(f, k));
	}

	function mark(v: number): void {
		for (const w of tree.nearestToCoord(m.vx(v), m.vy(v), m.vz(v), QUERY_K)) {
			if (distance(m, v, w) < minEdge) used[w] = 1;
		}
		m.vertFlags[v] |= VertexFlag.VISITED;
	}

	/**
	 * The centre of the ball of `radius` resting on the three points, on the
	 * side its normal points to. Null when they are collinear or too far apart
	 * for a ball that size to touch all three.
	 */
	function findSphere(p0: number[], p1: number[], p2: number[]): number[] | null {
		// Rotate so the lexicographically smallest point comes first. Purely for
		// determinism: the arithmetic below is not symmetric in floating point,
		// and the same triangle reached from two directions must give the same
		// centre or the front will not close.
		let p: number[][];
		if (less(p0, p1) && less(p0, p2)) p = [p0, p1, p2];
		else if (less(p1, p0) && less(p1, p2)) p = [p1, p2, p0];
		else p = [p2, p0, p1];

		const q1 = sub(p[1], p[0]);
		const q2 = sub(p[2], p[0]);
		const up = cross(q1, q2);
		const upLen = norm(up);
		if (upLen < 0.001 * norm(q1) * norm(q2)) return null; // collinear
		for (let k = 0; k < 3; k++) up[k] /= upLen;

		const a11 = dot(q1, q1);
		const a12 = dot(q1, q2);
		const a22 = dot(q2, q2);
		const den = 4 * (a11 * a22 - a12 * a12);
		const l1 = (2 * (a11 * a22 - a22 * a12)) / den;
		const l2 = (2 * (a11 * a22 - a12 * a11)) / den;
		const centre = [0, 1, 2].map((k) => q1[k] * l1 + q2[k] * l2);
		const circumRadius = norm(centre);
		if (circumRadius > radius) return null; // the ball cannot reach all three
		const height = Math.sqrt(radius * radius - circumRadius * circumRadius);
		return [0, 1, 2].map((k) => centre[k] + p[0][k] + up[k] * height);
	}

	// ---- the advancing front ------------------------------------------------

	const front = new EdgeList();
	const deads = new EdgeList();
	const nb = new Int32Array(m.vertSize);

	function newEdge(v0: number, v1: number, v2: number): FrontEdge {
		const e = {
			v0,
			v1,
			v2,
			active: true,
			listNext: null,
			listPrev: null,
			owner: null,
		} as unknown as FrontEdge;
		e.next = e;
		e.previous = e;
		return e;
	}

	function addNewEdge(v0: number, v1: number, v2: number): FrontEdge {
		const e = newEdge(v0, v1, v2);
		front.pushBack(e);
		return e;
	}

	function erase(e: FrontEdge): void {
		(e.active ? front : deads).remove(e);
	}

	function killEdge(e: FrontEdge): void {
		if (!e.active) return;
		e.active = false;
		front.remove(e);
		deads.pushBack(e);
	}

	function detach(v: number): void {
		if (nb[v] > 0 && --nb[v] === 0) m.vertFlags[v] &= ~VertexFlag.BORDER;
	}

	/** Sews `a` to `b` when they close a two-edge loop. */
	function gluePair(a: FrontEdge, b: FrontEdge): boolean {
		if (a.v0 !== b.v1) return false;
		const previous = a.previous;
		const next = b.next;
		previous.next = next;
		next.previous = previous;
		detach(a.v1);
		detach(a.v0);
		erase(a);
		erase(b);
		return true;
	}

	const glue = (e: FrontEdge): boolean => gluePair(e.previous, e) || gluePair(e, e.next);

	/**
	 * Would adding the directed edge `v0 -> v1` keep the surface orientable and
	 * manifold?
	 *
	 * A face already using the same direction means the two would disagree
	 * about which side is out; two faces already using the reverse direction
	 * means a third would make the edge non-manifold.
	 */
	function checkEdge(v0: number, v1: number): boolean {
		let reverse = 0;
		for (let f = 0; f < m.faceSize; f++) {
			if (m.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const a = m.fv(f, k);
				const b = m.fv(f, (k + 1) % 3);
				if (a === v0 && b === v1) return false;
				if (a === v1 && b === v0) reverse++;
			}
			if (reverse >= 2) return false;
		}
		return true;
	}

	function addFaceTo(v0: number, v1: number, v2: number): void {
		const f = Allocator.addFaces(m, 1);
		m.setFace(f, v0, v1, v2);
	}

	/** The front edge, if any, that starts at `v`. */
	function touchingEdge(v: number): FrontEdge | null {
		for (const e of front) if (e.v0 === v) return e;
		for (const e of deads) if (e.v0 === v) return e;
		return null;
	}

	// ---- seeding -------------------------------------------------------------

	let lastSeed = -1;

	/** A first triangle for a fresh component, or null when none is left. */
	function seed(): [number, number, number] | null {
		while (++lastSeed < m.vertSize) {
			const s = lastSeed;
			if (m.isVertD(s) || used[s] === 1) continue;
			used[s] = 1;

			const targets: number[] = [];
			for (const w of tree.nearestToCoord(m.vx(s), m.vy(s), m.vz(s), QUERY_K)) {
				if (distance(m, s, w) > 2 * radius) continue;
				targets.push(w);
			}
			if (targets.length < 3) continue;
			// A neighbourhood that already touches the reconstruction is not a
			// fresh component; seeding there would build a second sheet over it.
			if (targets.some((w) => m.isVertV(w))) continue;

			const found = seedTriplet(targets);
			if (found === null) continue;
			for (const v of found) mark(v);
			return found;
		}
		return null;
	}

	function seedTriplet(targets: readonly number[]): [number, number, number] | null {
		const n = targets.length;
		for (let i = 0; i < n; i++) {
			const a = targets[i];
			const p0 = point(m, a);
			for (let k = i + 1; k < n; k++) {
				const b = targets[k];
				const p1 = point(m, b);
				const d2 = norm(sub(p1, p0));
				if (d2 < minEdge || d2 > maxEdge) continue;
				for (let j = k + 1; j < n; j++) {
					const c = targets[j];
					const p2 = point(m, c);
					const d1 = norm(sub(p2, p0));
					if (d1 < minEdge || d1 > maxEdge) continue;
					const d0 = norm(sub(p2, p1));
					if (d0 < minEdge || d0 > maxEdge) continue;

					const normal = cross(sub(p1, p0), sub(p2, p0));
					// Orient the seed outward from the centroid. Without this the
					// first triangle can face inward and the whole component comes
					// out inside-out.
					if (dot(normal, sub(p0, centroid)) < 0) continue;

					const centre = findSphere(p0, p1, p2);
					if (centre === null) continue;
					// The ball must be empty, or these three are not on the hull
					// of their own neighbourhood.
					if (targets.some((t) => norm(sub(centre, point(m, t))) < radius - minEdge)) continue;
					// And there must be no surface already on the other side.
					const back = 2 * (dot(sub(centre, p0), normal) / dot(normal, normal));
					const opposite = [0, 1, 2].map((k2) => centre[k2] + normal[k2] * back);
					if (targets.some((t) => m.isVertV(t) && norm(sub(opposite, point(m, t))) <= radius)) {
						continue;
					}
					return [a, b, c];
				}
			}
		}
		return null;
	}

	function seedFace(): boolean {
		const v = seed();
		if (v === null) return false;
		const made: FrontEdge[] = [];
		for (let i = 0; i < 3; i++) {
			m.vertFlags[v[i]] |= VertexFlag.BORDER;
			nb[v[i]]++;
			const e = newEdge(v[i], v[(i + 1) % 3], v[(i + 2) % 3]);
			front.pushFront(e);
			made.push(e);
		}
		// pushFront reverses the insertion order, so the loop runs the other way.
		for (let i = 0; i < 3; i++) {
			made[i].next = made[(i + 2) % 3];
			made[i].previous = made[(i + 1) % 3];
		}
		addFaceTo(v[0], v[1], v[2]);
		return true;
	}

	// ---- pivoting ------------------------------------------------------------

	/**
	 * Pivots the ball around `edge` and returns the vertex it lands on, or -1.
	 *
	 * The ball rotates about the edge on a torus of spheres that touch both its
	 * endpoints; the winner is whichever candidate the ball reaches first,
	 * i.e. the smallest rotation angle. That "first hit wins" is what makes the
	 * result a surface rather than an arbitrary triangulation.
	 */
	function place(edge: FrontEdge): { vertex: number; touch: FrontEdge | null } {
		const v0 = point(m, edge.v0);
		const v1 = point(m, edge.v1);
		const v2 = point(m, edge.v2);
		const normal = normalise(cross(sub(v1, v0), sub(v2, v0)));
		const middle = [0, 1, 2].map((k) => (v0[k] + v1[k]) / 2);

		const centre = findSphere(v0, v1, v2);
		if (centre === null) return { vertex: -1, touch: null };

		const startPivot = sub(centre, middle);
		const axis = sub(v1, v0);
		const axisLen = dot(axis, axis);
		if (axisLen > 4 * radius * radius) return { vertex: -1, touch: null };
		const axisUnit = normalise(axis);
		// The radius of the torus the ball's centre sweeps.
		const r = Math.sqrt(radius * radius - axisLen / 4);

		let candidate = -1;
		let minAngle = Math.PI;
		for (const w of tree.nearestToCoord(middle[0], middle[1], middle[2], QUERY_K)) {
			const p = point(m, w);
			if (norm(sub(middle, p)) > r + radius) continue;
			// A used vertex is only reconsidered when it is on the front: it is
			// otherwise interior and reusing it would fold the surface.
			if (used[w] === 1 && (m.vertFlags[w] & VertexFlag.BORDER) === 0) continue;
			if (w === edge.v0 || w === edge.v1 || w === edge.v2) continue;

			const next = findSphere(v0, p, v1);
			if (next === null) continue;
			const alpha = orientedAngle(startPivot, sub(next, middle), axisUnit);
			if (candidate === -1 || alpha < minAngle) {
				candidate = w;
				minAngle = alpha;
			}
		}
		// Close to a half turn means the ball went round the back of the edge and
		// would fold the surface onto itself.
		if (candidate === -1 || minAngle >= Math.PI - 0.1) return { vertex: -1, touch: null };

		const newNormal = normalise(cross(sub(point(m, candidate), v0), sub(v1, v0)));
		if (dot(normal, newNormal) < maxAngle || nb[candidate] >= 2) {
			return { vertex: -1, touch: null };
		}
		const touch = touchingEdge(candidate);
		mark(candidate);
		return { vertex: candidate, touch };
	}

	/**
	 * One step: pivot the head of the front and stitch the new triangle in.
	 *
	 * The four cases below are the whole bookkeeping of an advancing front —
	 * closing onto the previous edge, onto the next, onto some other loop
	 * (splitting or merging it), or onto virgin ground.
	 */
	function advance(): void {
		const current = front.head;
		if (current === null) return;
		const previous = current.previous;
		const next = current.next;
		const v0 = current.v0;
		const v1 = current.v1;

		const { vertex: v2, touch } = place(current);
		if (v2 === -1) {
			killEdge(current);
			return;
		}

		if (touch !== null) {
			if (v2 === previous.v0) {
				if (!checkEdge(v2, v1)) {
					killEdge(current);
					return;
				}
				detach(v0);
				const up = addNewEdge(v2, v1, v0);
				front.remove(up);
				front.pushFront(up);
				up.previous = previous.previous;
				up.next = current.next;
				previous.previous.next = up;
				next.previous = up;
				erase(previous);
				erase(current);
				glue(up);
			} else if (v2 === next.v1) {
				if (!checkEdge(v0, v2)) {
					killEdge(current);
					return;
				}
				detach(v1);
				const up = addNewEdge(v0, v2, v1);
				front.remove(up);
				front.pushFront(up);
				up.previous = current.previous;
				up.next = next.next;
				previous.next = up;
				next.next.previous = up;
				erase(next);
				erase(current);
				glue(up);
			} else {
				if (!checkEdge(v0, v2) || !checkEdge(v2, v1)) {
					killEdge(current);
					return;
				}
				const left = touch;
				const right = touch.previous;
				// Joining here would make a degenerate two-edge loop.
				if (v1 === right.v0 || v0 === left.v1) {
					killEdge(current);
					return;
				}
				nb[v2]++;
				const down = addNewEdge(v2, v1, v0);
				const up = addNewEdge(v0, v2, v1);
				right.next = down;
				down.previous = right;
				down.next = current.next;
				next.previous = down;
				left.previous = up;
				up.next = left;
				up.previous = current.previous;
				previous.next = up;
				erase(current);
			}
		} else {
			nb[v2]++;
			m.vertFlags[v2] |= VertexFlag.BORDER;
			const down = addNewEdge(v2, v1, v0);
			const up = addNewEdge(v0, v2, v1);
			down.previous = up;
			up.next = down;
			down.next = current.next;
			next.previous = down;
			up.previous = current.previous;
			previous.next = up;
			erase(current);
		}
		addFaceTo(v0, v2, v1);
	}

	// ---- drive ---------------------------------------------------------------

	faceBorderFromNone(m);
	vertexBorderFromFace(m);
	createLoops(m, front, nb, addNewEdge);

	const startingFn = m.fn;
	// A surface over n points has at most about 2n triangles; the cap turns a
	// pathological input into a partial result rather than a hang.
	const limit = 4 * m.vn + 16;
	let steps = 0;
	while (steps++ < limit) {
		if (front.size === 0 && !seedFace()) break;
		advance();
		if (steps % 256 === 0) progress(Math.min(99, (100 * (m.fn - startingFn)) / (2 * m.vn)));
	}
	m.imark++;
	return { addedFaces: m.fn - startingFn, radius };
}

/** Turns the existing faces' border edges into the initial front loops. */
function createLoops(
	m: CMeshO,
	front: EdgeList,
	nb: Int32Array,
	addNewEdge: (v0: number, v1: number, v2: number) => FrontEdge,
): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			if (!m.isFaceB(f, k)) continue;
			addNewEdge(m.fv(f, k), m.fv(f, (k + 1) % 3), m.fv(f, (k + 2) % 3));
			nb[m.fv(f, k)]++;
		}
	}
	// Chain each edge to the one continuing from its far endpoint. An edge with
	// no successor is left pointing at itself, which the pivot loop treats as a
	// degenerate loop and kills rather than following into nothing.
	const byStart = new Map<number, FrontEdge[]>();
	for (const e of front) {
		const list = byStart.get(e.v0);
		if (list === undefined) byStart.set(e.v0, [e]);
		else list.push(e);
	}
	const claimed = new Set<FrontEdge>();
	for (const e of front) {
		const candidates = byStart.get(e.v1);
		if (candidates === undefined) continue;
		const j = candidates.find((c) => c !== e && !claimed.has(c));
		if (j === undefined) continue;
		claimed.add(j);
		e.next = j;
		j.previous = e;
	}
}

// ---- small vector helpers ----------------------------------------------------

const point = (m: CMeshO, v: number): number[] => [m.vx(v), m.vy(v), m.vz(v)];
const distance = (m: CMeshO, a: number, b: number): number =>
	Math.hypot(m.vx(a) - m.vx(b), m.vy(a) - m.vy(b), m.vz(a) - m.vz(b));
const sub = (a: readonly number[], b: readonly number[]): number[] => [
	a[0] - b[0],
	a[1] - b[1],
	a[2] - b[2],
];
const dot = (a: readonly number[], b: readonly number[]): number =>
	a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]): number[] => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];
const norm = (a: readonly number[]): number => Math.hypot(a[0], a[1], a[2]);

function normalise(a: readonly number[]): number[] {
	const len = norm(a);
	return len === 0 ? [0, 0, 0] : [a[0] / len, a[1] / len, a[2] / len];
}

/** VCG's `Point3::operator<`: z first, then y, then x. */
function less(a: readonly number[], b: readonly number[]): boolean {
	if (a[2] !== b[2]) return a[2] < b[2];
	if (a[1] !== b[1]) return a[1] < b[1];
	return a[0] < b[0];
}

/** The angle from `p` to `q` about `axis`, always in 0..2π. */
function orientedAngle(
	p: readonly number[],
	q: readonly number[],
	axis: readonly number[],
): number {
	const pn = normalise(p);
	const qn = normalise(q);
	let angle = Math.acos(Math.min(1, Math.max(-1, dot(pn, qn))));
	if (dot(cross(pn, qn), axis) < 0) angle = -angle;
	return angle < 0 ? angle + 2 * Math.PI : angle;
}

export const BallPivoting = { ballPivoting } as const;
