/**
 * `pointcloud_normal` — estimating a consistently oriented normal field for a
 * point set, mirroring `vcg/complex/algorithms/pointcloud_normal.h`.
 *
 * Three stages, and the third is the one that matters:
 *
 * 1. Fit a plane to each point's k nearest neighbours by PCA; the normal is the
 *    eigenvector of the smallest eigenvalue.
 * 2. Smooth the field, so noise in the individual fits does not survive.
 * 3. **Orient it.** A plane fit gives a normal only up to sign, and which sign
 *    it produces is arbitrary. Screened Poisson needs a globally consistent
 *    field — an inward-facing patch carves a dent into the reconstructed solid
 *    — so the sign is propagated across the neighbour graph in Hoppe's order,
 *    most-parallel edge first, seeded from a point where "outward" is known.
 *
 * Writes into `m.vertNormal` in place. Faces are ignored entirely: this is for
 * point sets, and works whether or not the mesh has any.
 */

import { KdTree } from "../space/index/kdtree.ts";
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";

type Vec3 = [number, number, number];

export interface NormalOptions {
	/** `k`: neighbours used for the plane fit. */
	neighbors: number;
	/** `smoothiter`: normal smoothing passes. */
	smoothIterations: number;
	/**
	 * `viewpos`: turn every normal toward this point, MeshLab's `flipflag`.
	 *
	 * When the scanner position is known this is both cheaper and more reliable
	 * than propagation. Without it, orientation falls back to the neighbour
	 * graph.
	 */
	viewpoint?: Vec3;
}

/** Smallest-eigenvalue eigenvector of a symmetric 3x3, by cyclic Jacobi rotation. */
function smallestEigenvector(covariance: Float64Array): Vec3 {
	const matrix = Float64Array.from(covariance);
	const basis = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
	for (let sweep = 0; sweep < 12; sweep++) {
		let offDiagonal = 0;
		for (const [row, column] of [
			[0, 1],
			[0, 2],
			[1, 2],
		])
			offDiagonal += Math.abs(matrix[row * 3 + column]);
		if (offDiagonal < 1e-18) break;
		for (const [row, column] of [
			[0, 1],
			[0, 2],
			[1, 2],
		] as const) {
			const value = matrix[row * 3 + column];
			if (Math.abs(value) < 1e-20) continue;
			const theta = (matrix[column * 3 + column] - matrix[row * 3 + row]) / (2 * value);
			const sign = theta >= 0 ? 1 : -1;
			const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
			const cosine = 1 / Math.sqrt(t * t + 1);
			const sine = t * cosine;
			for (let k = 0; k < 3; k++) {
				const first = matrix[row * 3 + k];
				const second = matrix[column * 3 + k];
				matrix[row * 3 + k] = cosine * first - sine * second;
				matrix[column * 3 + k] = sine * first + cosine * second;
			}
			for (let k = 0; k < 3; k++) {
				const first = matrix[k * 3 + row];
				const second = matrix[k * 3 + column];
				matrix[k * 3 + row] = cosine * first - sine * second;
				matrix[k * 3 + column] = sine * first + cosine * second;
				const firstBasis = basis[k * 3 + row];
				const secondBasis = basis[k * 3 + column];
				basis[k * 3 + row] = cosine * firstBasis - sine * secondBasis;
				basis[k * 3 + column] = sine * firstBasis + cosine * secondBasis;
			}
		}
	}
	let smallest = 0;
	for (let axis = 1; axis < 3; axis++)
		if (matrix[axis * 3 + axis] < matrix[smallest * 3 + smallest]) smallest = axis;
	const normal: Vec3 = [basis[smallest], basis[3 + smallest], basis[6 + smallest]];
	const size = Math.hypot(normal[0], normal[1], normal[2]);
	return size > 0 ? [normal[0] / size, normal[1] / size, normal[2] / size] : [0, 0, 1];
}

/** Minimal binary min-heap over (weight, payload) pairs. */
class Heap {
	private weights: Float64Array;
	private payload: Int32Array;
	private size = 0;

	constructor(capacity: number) {
		this.weights = new Float64Array(Math.max(1, capacity));
		this.payload = new Int32Array(Math.max(1, capacity));
	}

	get length(): number {
		return this.size;
	}

	push(weight: number, value: number): void {
		if (this.size === this.weights.length) {
			const grownWeights = new Float64Array(this.size * 2);
			const grownPayload = new Int32Array(this.size * 2);
			grownWeights.set(this.weights);
			grownPayload.set(this.payload);
			this.weights = grownWeights;
			this.payload = grownPayload;
		}
		let slot = this.size++;
		this.weights[slot] = weight;
		this.payload[slot] = value;
		while (slot > 0) {
			const parent = (slot - 1) >> 1;
			if (this.weights[parent] <= this.weights[slot]) break;
			this.swap(parent, slot);
			slot = parent;
		}
	}

	pop(): number {
		const top = this.payload[0];
		this.size--;
		if (this.size > 0) {
			this.weights[0] = this.weights[this.size];
			this.payload[0] = this.payload[this.size];
			let slot = 0;
			for (;;) {
				const left = slot * 2 + 1;
				const right = left + 1;
				let smallest = slot;
				if (left < this.size && this.weights[left] < this.weights[smallest]) smallest = left;
				if (right < this.size && this.weights[right] < this.weights[smallest]) smallest = right;
				if (smallest === slot) break;
				this.swap(smallest, slot);
				slot = smallest;
			}
		}
		return top;
	}

	private swap(a: number, b: number): void {
		const weight = this.weights[a];
		this.weights[a] = this.weights[b];
		this.weights[b] = weight;
		const value = this.payload[a];
		this.payload[a] = this.payload[b];
		this.payload[b] = value;
	}
}

/**
 * `compute_normal_for_point_clouds`: fit a plane to each point's neighbourhood,
 * smooth the field, then orient it.
 *
 * Without a viewpoint MeshLab leaves the per-point sign to the plane fit, which is
 * arbitrary. Screened Poisson needs a globally consistent field, so orientation is
 * propagated along the neighbour graph in Hoppe's order — most-parallel edge
 * first — seeded outward from an extreme point of each component.
 */
export function estimateNormals(m: CMeshO, options: NormalOptions): void {
	if (m.vn === 0) return;
	// Point indices below run 0..vn, which only lines up with the storage when
	// nothing is deleted. Compacting first is cheaper than threading a live-index
	// map through the kd-tree, the neighbour graph and the propagation.
	if (!m.isCompact) Allocator.compactEveryVector(m);
	const index = new KdTree(m.vertCoord, m.vertSize);
	const k = Math.max(3, Math.min(options.neighbors, m.vn));
	const graph = new Int32Array(m.vn * k);
	for (let point = 0; point < m.vn; point++) graph.set(index.nearest(point, k), point * k);

	const normals = new Float64Array(m.vn * 3);
	const covariance = new Float64Array(9);
	for (let point = 0; point < m.vn; point++) {
		const centre = [0, 0, 0];
		for (let slot = 0; slot < k; slot++) {
			const neighbour = graph[point * k + slot];
			for (let axis = 0; axis < 3; axis++) centre[axis] += m.vertCoord[neighbour * 3 + axis];
		}
		for (let axis = 0; axis < 3; axis++) centre[axis] /= k;
		covariance.fill(0);
		for (let slot = 0; slot < k; slot++) {
			const neighbour = graph[point * k + slot];
			const offset = [
				m.vertCoord[neighbour * 3] - centre[0],
				m.vertCoord[neighbour * 3 + 1] - centre[1],
				m.vertCoord[neighbour * 3 + 2] - centre[2],
			];
			for (let row = 0; row < 3; row++)
				for (let column = 0; column < 3; column++)
					covariance[row * 3 + column] += offset[row] * offset[column];
		}
		normals.set(smallestEigenvector(covariance), point * 3);
	}

	// Smoothing: average each normal with its neighbours, flipping any neighbour
	// that disagrees so an unoriented field still averages meaningfully.
	let current = normals;
	for (let pass = 0; pass < options.smoothIterations; pass++) {
		const next = new Float64Array(current.length);
		for (let point = 0; point < m.vn; point++) {
			const anchor = [current[point * 3], current[point * 3 + 1], current[point * 3 + 2]];
			const total = [0, 0, 0];
			for (let slot = 0; slot < k; slot++) {
				const neighbour = graph[point * k + slot];
				const sign =
					anchor[0] * current[neighbour * 3] +
						anchor[1] * current[neighbour * 3 + 1] +
						anchor[2] * current[neighbour * 3 + 2] <
					0
						? -1
						: 1;
				for (let axis = 0; axis < 3; axis++) total[axis] += sign * current[neighbour * 3 + axis];
			}
			const size = Math.hypot(total[0], total[1], total[2]);
			for (let axis = 0; axis < 3; axis++)
				next[point * 3 + axis] = size > 0 ? total[axis] / size : anchor[axis];
		}
		current = next;
	}

	if (options.viewpoint) {
		const [vx, vy, vz] = options.viewpoint;
		for (let point = 0; point < m.vn; point++) {
			const toCamera = [
				vx - m.vertCoord[point * 3],
				vy - m.vertCoord[point * 3 + 1],
				vz - m.vertCoord[point * 3 + 2],
			];
			const facing =
				current[point * 3] * toCamera[0] +
				current[point * 3 + 1] * toCamera[1] +
				current[point * 3 + 2] * toCamera[2];
			if (facing < 0) for (let axis = 0; axis < 3; axis++) current[point * 3 + axis] *= -1;
		}
	} else orientCoherently(m, current, graph, k);

	m.vertNormal.set(current.subarray(0, m.vn * 3));
	m.imark++;
}

/** Propagate a consistent sign along the neighbour graph, most-parallel edge first. */
function orientCoherently(m: CMeshO, normals: Float64Array, graph: Int32Array, k: number): void {
	const centre = [0, 0, 0];
	for (let point = 0; point < m.vn; point++)
		for (let axis = 0; axis < 3; axis++) centre[axis] += m.vertCoord[point * 3 + axis];
	for (let axis = 0; axis < 3; axis++) centre[axis] /= m.vn;

	const visited = new Uint8Array(m.vn);
	const flip = (point: number) => {
		for (let axis = 0; axis < 3; axis++) normals[point * 3 + axis] *= -1;
	};

	let searchFrom = 0;
	while (true) {
		// Seed each component at its point furthest from the centroid, whose normal
		// must face away from the bulk of the cloud.
		let seed = -1;
		let best = -Infinity;
		for (let point = searchFrom; point < m.vn; point++) {
			if (visited[point]) continue;
			if (seed < 0) searchFrom = point;
			const distance =
				(m.vertCoord[point * 3] - centre[0]) ** 2 +
				(m.vertCoord[point * 3 + 1] - centre[1]) ** 2 +
				(m.vertCoord[point * 3 + 2] - centre[2]) ** 2;
			if (distance > best) {
				best = distance;
				seed = point;
			}
		}
		if (seed < 0) return;

		const outward =
			normals[seed * 3] * (m.vertCoord[seed * 3] - centre[0]) +
			normals[seed * 3 + 1] * (m.vertCoord[seed * 3 + 1] - centre[1]) +
			normals[seed * 3 + 2] * (m.vertCoord[seed * 3 + 2] - centre[2]);
		if (outward < 0) flip(seed);
		visited[seed] = 1;

		const frontier = new Heap(k * 16);
		const pushEdges = (point: number) => {
			for (let slot = 0; slot < k; slot++) {
				const neighbour = graph[point * k + slot];
				if (visited[neighbour]) continue;
				const parallel =
					normals[point * 3] * normals[neighbour * 3] +
					normals[point * 3 + 1] * normals[neighbour * 3 + 1] +
					normals[point * 3 + 2] * normals[neighbour * 3 + 2];
				frontier.push(1 - Math.abs(parallel), point * k + slot);
			}
		};
		pushEdges(seed);
		while (frontier.length > 0) {
			const edge = frontier.pop();
			const from = Math.floor(edge / k);
			const to = graph[edge];
			if (visited[to]) continue;
			const parallel =
				normals[from * 3] * normals[to * 3] +
				normals[from * 3 + 1] * normals[to * 3 + 1] +
				normals[from * 3 + 2] * normals[to * 3 + 2];
			if (parallel < 0) flip(to);
			visited[to] = 1;
			pushEdges(to);
		}
	}
}

export const PointCloudNormal = { estimateNormals } as const;
