/**
 * A bounding volume hierarchy over a mesh's triangles, for ray casting.
 *
 * Ambient occlusion, the shape diameter function and depth complexity all ask
 * the same question — what does this ray hit — hundreds of times per vertex,
 * so the query has to be sub-linear. A BVH built by binned surface-area
 * heuristic is the standard answer and needs no preprocessing of the mesh.
 *
 * Two queries are offered and the difference matters for performance:
 * {@link BVH.intersect} finds the *nearest* hit and must examine every node
 * the ray reaches, while {@link BVH.occluded} stops at the first one. An
 * occlusion test written in terms of `intersect` does several times the work
 * for an answer it throws away.
 */

import type { CMeshO } from "../../complex/cmesho.ts";

export interface RayHit {
	readonly face: number;
	/** Distance along the ray, in units of the direction's length. */
	readonly t: number;
	/** Barycentric weights on the face's three corners. */
	readonly bary: readonly [number, number, number];
	/** True when the ray struck the face from behind. */
	readonly backface: boolean;
}

/** Leaves hold at most this many triangles; deeper splitting stops paying. */
const LEAF_SIZE = 4;
/** How many candidate split planes the surface-area heuristic tries per axis. */
const BINS = 12;

export class BVH {
	private readonly faces: Int32Array;
	/** Six floats per node: the box, min then max. */
	private readonly bounds: Float64Array;
	private readonly left: Int32Array;
	private readonly start: Int32Array;
	private readonly count: Int32Array;
	private nodeCount = 0;

	private readonly corners: Float64Array;

	constructor(private readonly cm: CMeshO) {
		const live: number[] = [];
		for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f)) live.push(f);
		this.faces = Int32Array.from(live);

		// The corners are copied out once. Every intersection would otherwise
		// chase three indirections through faceVert into vertCoord, which
		// dominates the inner loop.
		this.corners = new Float64Array(live.length * 9);
		live.forEach((f, i) => {
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				this.corners[9 * i + 3 * k] = cm.vx(v);
				this.corners[9 * i + 3 * k + 1] = cm.vy(v);
				this.corners[9 * i + 3 * k + 2] = cm.vz(v);
			}
		});

		const capacity = Math.max(1, 2 * live.length + 1);
		this.bounds = new Float64Array(capacity * 6);
		this.left = new Int32Array(capacity).fill(-1);
		this.start = new Int32Array(capacity);
		this.count = new Int32Array(capacity);
		if (live.length > 0) this.build(0, live.length);
	}

	get faceCount(): number {
		return this.faces.length;
	}

	private build(from: number, to: number): number {
		const node = this.nodeCount++;
		this.start[node] = from;
		this.count[node] = to - from;
		const box = this.boundsOf(from, to);
		this.bounds.set(box, node * 6);

		if (to - from <= LEAF_SIZE) return node;

		// Split along the widest axis of the *centroids*, not of the box: a
		// few large triangles would otherwise drag every split towards them.
		const centroidBox = this.centroidBoundsOf(from, to);
		let axis = 0;
		for (let a = 1; a < 3; a++) {
			if (centroidBox[a + 3] - centroidBox[a] > centroidBox[axis + 3] - centroidBox[axis]) axis = a;
		}
		const span = centroidBox[axis + 3] - centroidBox[axis];
		if (span <= 0) {
			this.count[node] = to - from;
			return node;
		}

		const middle = this.partition(from, to, axis, centroidBox[axis], span);
		if (middle <= from || middle >= to) return node; // nothing separated

		this.count[node] = 0; // an internal node holds no triangles itself
		this.left[node] = this.build(from, middle);
		this.build(middle, to);
		return node;
	}

	/**
	 * Reorders `[from, to)` around the best of `BINS` candidate planes.
	 *
	 * The cost of a split is the surface-area heuristic: the chance a random
	 * ray enters each side, times how many triangles it would then test.
	 * Splitting at the median instead is simpler and noticeably worse on
	 * meshes whose triangles vary in size.
	 */
	private partition(from: number, to: number, axis: number, low: number, span: number): number {
		const counts = new Int32Array(BINS);
		const boxes = new Float64Array(BINS * 6);
		for (let b = 0; b < BINS; b++) resetBox(boxes, b);

		const binOf = (i: number) => {
			const c = this.centroid(i, axis);
			return Math.min(BINS - 1, Math.floor(((c - low) / span) * BINS));
		};
		for (let i = from; i < to; i++) {
			const b = binOf(i);
			counts[b]++;
			this.growBox(boxes, b, i);
		}

		let bestCost = Number.POSITIVE_INFINITY;
		let bestSplit = -1;
		const leftBox = new Float64Array(6);
		let leftCount = 0;
		resetBox(leftBox, 0);
		for (let b = 0; b < BINS - 1; b++) {
			mergeBox(leftBox, 0, boxes, b);
			leftCount += counts[b];
			if (leftCount === 0) continue;

			const rightBox = new Float64Array(6);
			resetBox(rightBox, 0);
			let rightCount = 0;
			for (let c = b + 1; c < BINS; c++) {
				mergeBox(rightBox, 0, boxes, c);
				rightCount += counts[c];
			}
			if (rightCount === 0) continue;
			const cost = surfaceArea(leftBox, 0) * leftCount + surfaceArea(rightBox, 0) * rightCount;
			if (cost < bestCost) {
				bestCost = cost;
				bestSplit = b;
			}
		}
		if (bestSplit < 0) return from;

		// Hoare partition on the chosen bin.
		let i = from;
		let j = to - 1;
		while (i <= j) {
			if (binOf(i) <= bestSplit) {
				i++;
			} else {
				this.swap(i, j);
				j--;
			}
		}
		return i;
	}

	private swap(a: number, b: number): void {
		const f = this.faces[a];
		this.faces[a] = this.faces[b];
		this.faces[b] = f;
		for (let k = 0; k < 9; k++) {
			const t = this.corners[9 * a + k];
			this.corners[9 * a + k] = this.corners[9 * b + k];
			this.corners[9 * b + k] = t;
		}
	}

	private centroid(i: number, axis: number): number {
		return (
			(this.corners[9 * i + axis] +
				this.corners[9 * i + 3 + axis] +
				this.corners[9 * i + 6 + axis]) /
			3
		);
	}

	private boundsOf(from: number, to: number): Float64Array {
		const box = new Float64Array(6);
		resetBox(box, 0);
		for (let i = from; i < to; i++) this.growBox(box, 0, i);
		return box;
	}

	private centroidBoundsOf(from: number, to: number): Float64Array {
		const box = new Float64Array(6);
		resetBox(box, 0);
		for (let i = from; i < to; i++) {
			for (let a = 0; a < 3; a++) {
				const c = this.centroid(i, a);
				if (c < box[a]) box[a] = c;
				if (c > box[a + 3]) box[a + 3] = c;
			}
		}
		return box;
	}

	private growBox(boxes: Float64Array, slot: number, i: number): void {
		for (let k = 0; k < 3; k++) {
			for (let a = 0; a < 3; a++) {
				const c = this.corners[9 * i + 3 * k + a];
				if (c < boxes[slot * 6 + a]) boxes[slot * 6 + a] = c;
				if (c > boxes[slot * 6 + a + 3]) boxes[slot * 6 + a + 3] = c;
			}
		}
	}

	/** The nearest hit along the ray, or null. */
	intersect(
		origin: readonly number[],
		direction: readonly number[],
		tMin = 1e-7,
		tMax = Number.POSITIVE_INFINITY,
	): RayHit | null {
		if (this.faces.length === 0) return null;
		const inverse = [1 / direction[0], 1 / direction[1], 1 / direction[2]];
		let best: RayHit | null = null;
		let bestT = tMax;

		const stack = [0];
		while (stack.length > 0) {
			const node = stack.pop() as number;
			if (!this.hitsBox(node, origin, inverse, tMin, bestT)) continue;
			if (this.left[node] < 0) {
				const from = this.start[node];
				const to = from + this.count[node];
				for (let i = from; i < to; i++) {
					const hit = this.triangle(i, origin, direction, tMin, bestT);
					if (hit !== null) {
						bestT = hit.t;
						best = hit;
					}
				}
				continue;
			}
			stack.push(this.left[node], this.left[node] + this.subtreeSize(this.left[node]));
		}
		return best;
	}

	/**
	 * Whether anything blocks the ray between `tMin` and `tMax`.
	 *
	 * Returns as soon as it finds one, which is what makes a shadow ray much
	 * cheaper than a nearest-hit query.
	 */
	occluded(
		origin: readonly number[],
		direction: readonly number[],
		tMin = 1e-7,
		tMax = Number.POSITIVE_INFINITY,
	): boolean {
		if (this.faces.length === 0) return false;
		const inverse = [1 / direction[0], 1 / direction[1], 1 / direction[2]];
		const stack = [0];
		while (stack.length > 0) {
			const node = stack.pop() as number;
			if (!this.hitsBox(node, origin, inverse, tMin, tMax)) continue;
			if (this.left[node] < 0) {
				const from = this.start[node];
				const to = from + this.count[node];
				for (let i = from; i < to; i++) {
					if (this.triangle(i, origin, direction, tMin, tMax) !== null) return true;
				}
				continue;
			}
			stack.push(this.left[node], this.left[node] + this.subtreeSize(this.left[node]));
		}
		return false;
	}

	/** Every hit along the ray, in no particular order — for depth complexity. */
	intersectAll(
		origin: readonly number[],
		direction: readonly number[],
		tMin = 1e-7,
		tMax = Number.POSITIVE_INFINITY,
	): RayHit[] {
		const out: RayHit[] = [];
		if (this.faces.length === 0) return out;
		const inverse = [1 / direction[0], 1 / direction[1], 1 / direction[2]];
		const stack = [0];
		while (stack.length > 0) {
			const node = stack.pop() as number;
			if (!this.hitsBox(node, origin, inverse, tMin, tMax)) continue;
			if (this.left[node] < 0) {
				const from = this.start[node];
				const to = from + this.count[node];
				for (let i = from; i < to; i++) {
					const hit = this.triangle(i, origin, direction, tMin, tMax);
					if (hit !== null) out.push(hit);
				}
				continue;
			}
			stack.push(this.left[node], this.left[node] + this.subtreeSize(this.left[node]));
		}
		return out;
	}

	/**
	 * The size of a subtree, so the right child can be found from the left.
	 *
	 * Nodes are laid out depth-first, so the right child starts immediately
	 * after the left subtree ends. Storing that index would be one more array
	 * to keep in step; computing it is a walk of the same tree.
	 */
	private subtreeSize(node: number): number {
		if (this.left[node] < 0) return 1;
		const l = this.left[node];
		const leftSize = this.subtreeSize(l);
		return 1 + leftSize + this.subtreeSize(l + leftSize);
	}

	/**
	 * The slab test, written so a NaN cannot cull a node.
	 *
	 * A ray with a zero direction component gives an infinite inverse, and if
	 * the origin sits exactly on that slab's plane the product is `0 *
	 * Infinity` — NaN. The branch-free `min`/`max` form propagates it and the
	 * node is silently dropped, which on an axis-aligned ray through the
	 * origin of a symmetric mesh means missing the mesh entirely. Comparing
	 * instead of taking minima means a NaN simply leaves that axis
	 * unconstrained, which is the conservative answer.
	 */
	private hitsBox(
		node: number,
		origin: readonly number[],
		inverse: readonly number[],
		tMin: number,
		tMax: number,
	): boolean {
		let near = tMin;
		let far = tMax;
		for (let a = 0; a < 3; a++) {
			let t1 = (this.bounds[node * 6 + a] - origin[a]) * inverse[a];
			let t2 = (this.bounds[node * 6 + a + 3] - origin[a]) * inverse[a];
			if (t1 > t2) {
				const t = t1;
				t1 = t2;
				t2 = t;
			}
			if (t1 > near) near = t1;
			if (t2 < far) far = t2;
			if (near > far) return false;
		}
		return true;
	}

	/** Möller–Trumbore, without the early bail so a back face still reports. */
	private triangle(
		i: number,
		origin: readonly number[],
		direction: readonly number[],
		tMin: number,
		tMax: number,
	): RayHit | null {
		const c = this.corners;
		const o = 9 * i;
		const e1 = [c[o + 3] - c[o], c[o + 4] - c[o + 1], c[o + 5] - c[o + 2]];
		const e2 = [c[o + 6] - c[o], c[o + 7] - c[o + 1], c[o + 8] - c[o + 2]];
		const p = [
			direction[1] * e2[2] - direction[2] * e2[1],
			direction[2] * e2[0] - direction[0] * e2[2],
			direction[0] * e2[1] - direction[1] * e2[0],
		];
		const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
		if (Math.abs(det) < 1e-14) return null; // parallel to the triangle
		const inverseDet = 1 / det;

		const t0 = [origin[0] - c[o], origin[1] - c[o + 1], origin[2] - c[o + 2]];
		const u = (t0[0] * p[0] + t0[1] * p[1] + t0[2] * p[2]) * inverseDet;
		if (u < 0 || u > 1) return null;

		const q = [
			t0[1] * e1[2] - t0[2] * e1[1],
			t0[2] * e1[0] - t0[0] * e1[2],
			t0[0] * e1[1] - t0[1] * e1[0],
		];
		const v = (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * inverseDet;
		if (v < 0 || u + v > 1) return null;

		const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inverseDet;
		if (t < tMin || t > tMax) return null;
		return { face: this.faces[i], t, bary: [1 - u - v, u, v], backface: det < 0 };
	}

	/** The mesh this hierarchy was built over. */
	get mesh(): CMeshO {
		return this.cm;
	}
}

function resetBox(boxes: Float64Array, slot: number): void {
	for (let a = 0; a < 3; a++) {
		boxes[slot * 6 + a] = Number.POSITIVE_INFINITY;
		boxes[slot * 6 + a + 3] = Number.NEGATIVE_INFINITY;
	}
}

function mergeBox(into: Float64Array, slot: number, from: Float64Array, other: number): void {
	for (let a = 0; a < 3; a++) {
		into[slot * 6 + a] = Math.min(into[slot * 6 + a], from[other * 6 + a]);
		into[slot * 6 + a + 3] = Math.max(into[slot * 6 + a + 3], from[other * 6 + a + 3]);
	}
}

function surfaceArea(boxes: Float64Array, slot: number): number {
	const dx = boxes[slot * 6 + 3] - boxes[slot * 6];
	const dy = boxes[slot * 6 + 4] - boxes[slot * 6 + 1];
	const dz = boxes[slot * 6 + 5] - boxes[slot * 6 + 2];
	if (dx < 0 || dy < 0 || dz < 0) return 0;
	return 2 * (dx * dy + dy * dz + dz * dx);
}

/**
 * Directions spread evenly over a hemisphere about `normal`, cosine
 * weighted.
 *
 * Cosine weighting is not decoration: ambient occlusion integrates the
 * visibility times the cosine of the angle to the normal, so sampling with
 * that density lets the estimate be a plain average instead of a weighted one
 * — the same number of rays gives a visibly less noisy answer.
 */
export function cosineHemisphere(
	normal: readonly number[],
	count: number,
	random: () => number,
): number[][] {
	const [tangent, bitangent] = frame(normal);
	const out: number[][] = [];
	for (let i = 0; i < count; i++) {
		const r = Math.sqrt(random());
		const phi = 2 * Math.PI * random();
		const x = r * Math.cos(phi);
		const y = r * Math.sin(phi);
		const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
		out.push([
			tangent[0] * x + bitangent[0] * y + normal[0] * z,
			tangent[1] * x + bitangent[1] * y + normal[1] * z,
			tangent[2] * x + bitangent[2] * y + normal[2] * z,
		]);
	}
	return out;
}

/** Directions inside a cone about an axis, uniformly in solid angle. */
export function coneDirections(
	axis: readonly number[],
	halfAngleRad: number,
	count: number,
	random: () => number,
): number[][] {
	const [tangent, bitangent] = frame(axis);
	const cosMax = Math.cos(halfAngleRad);
	const out: number[][] = [];
	for (let i = 0; i < count; i++) {
		// Uniform in cos(theta) rather than in theta: the latter crowds the
		// samples towards the cone's axis.
		const cosTheta = 1 - random() * (1 - cosMax);
		const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
		const phi = 2 * Math.PI * random();
		const x = sinTheta * Math.cos(phi);
		const y = sinTheta * Math.sin(phi);
		out.push([
			tangent[0] * x + bitangent[0] * y + axis[0] * cosTheta,
			tangent[1] * x + bitangent[1] * y + axis[1] * cosTheta,
			tangent[2] * x + bitangent[2] * y + axis[2] * cosTheta,
		]);
	}
	return out;
}

/** Two unit vectors perpendicular to `n` and to each other. */
function frame(n: readonly number[]): [number[], number[]] {
	// Picking the axis n is *least* aligned with keeps the cross product from
	// collapsing, which a fixed choice does whenever n happens to match it.
	const helper =
		Math.abs(n[0]) < Math.abs(n[1]) && Math.abs(n[0]) < Math.abs(n[2])
			? [1, 0, 0]
			: Math.abs(n[1]) < Math.abs(n[2])
				? [0, 1, 0]
				: [0, 0, 1];
	const t = [
		n[1] * helper[2] - n[2] * helper[1],
		n[2] * helper[0] - n[0] * helper[2],
		n[0] * helper[1] - n[1] * helper[0],
	];
	const length = Math.hypot(t[0], t[1], t[2]) || 1;
	const tangent = [t[0] / length, t[1] / length, t[2] / length];
	const bitangent = [
		n[1] * tangent[2] - n[2] * tangent[1],
		n[2] * tangent[0] - n[0] * tangent[2],
		n[0] * tangent[1] - n[1] * tangent[0],
	];
	return [tangent, bitangent];
}
