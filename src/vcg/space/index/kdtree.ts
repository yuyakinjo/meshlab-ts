/**
 * `KdTree` — a kd-tree over a flat coordinate array, mirroring
 * `vcg/space/index/kdtree/kdtree.h`.
 *
 * Takes `Float64Array` positions and a point count rather than a mesh, which
 * means it indexes `CMeshO.vertCoord` directly with no copy, and works just as
 * well on a bare point cloud that has no faces at all.
 *
 * Nodes live in flat typed arrays for the same reason the mesh does: a cloud of
 * a million points must not become a million objects.
 */

/** An axis-aligned box over a point set. */
export interface PointBounds {
	low: [number, number, number];
	high: [number, number, number];
	diagonal: number;
}

export function pointBounds(positions: Float64Array, count: number): PointBounds {
	if (count === 0) return { low: [0, 0, 0], high: [0, 0, 0], diagonal: 0 };
	const low: [number, number, number] = [
		Number.POSITIVE_INFINITY,
		Number.POSITIVE_INFINITY,
		Number.POSITIVE_INFINITY,
	];
	const high: [number, number, number] = [
		Number.NEGATIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
	];
	for (let point = 0; point < count; point++)
		for (let axis = 0; axis < 3; axis++) {
			const value = positions[point * 3 + axis];
			if (value < low[axis]) low[axis] = value;
			if (value > high[axis]) high[axis] = value;
		}
	return {
		low,
		high,
		diagonal: Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]),
	};
}

const LEAF_SIZE = 16;

/**
 * Kd-tree over a flat coordinate array, standing in for the VCG kd-tree the
 * MeshLab filter queries. Nodes live in flat typed arrays so a cloud of a million
 * points does not turn into a million objects.
 */
/** One hit from {@link KdTree.withinRadius}. */
export interface Neighbour {
	readonly index: number;
	readonly squaredDistance: number;
}

export class KdTree {
	private readonly axis: Int8Array;
	private readonly split: Float64Array;
	private readonly left: Int32Array;
	private readonly right: Int32Array;
	private readonly start: Int32Array;
	private readonly end: Int32Array;
	private readonly order: Int32Array;
	private nodeCount = 0;

	constructor(
		private readonly positions: Float64Array,
		private readonly count: number,
	) {
		this.order = new Int32Array(count);
		for (let index = 0; index < count; index++) this.order[index] = index;
		const capacity = Math.max(1, 2 * Math.ceil(count / LEAF_SIZE) + 1) * 2;
		this.axis = new Int8Array(capacity);
		this.split = new Float64Array(capacity);
		this.left = new Int32Array(capacity).fill(-1);
		this.right = new Int32Array(capacity).fill(-1);
		this.start = new Int32Array(capacity);
		this.end = new Int32Array(capacity);
		if (count > 0) this.build(0, count);
	}

	private coordinate(point: number, axis: number): number {
		return this.positions[point * 3 + axis];
	}

	/** Partition `order[from, to)` around the median of the widest axis. */
	private build(from: number, to: number): number {
		const node = this.nodeCount++;
		this.start[node] = from;
		this.end[node] = to;
		if (to - from <= LEAF_SIZE) {
			this.axis[node] = -1;
			return node;
		}
		const low = [Infinity, Infinity, Infinity];
		const high = [-Infinity, -Infinity, -Infinity];
		for (let slot = from; slot < to; slot++)
			for (let axis = 0; axis < 3; axis++) {
				const value = this.coordinate(this.order[slot], axis);
				if (value < low[axis]) low[axis] = value;
				if (value > high[axis]) high[axis] = value;
			}
		let axis = 0;
		for (let candidate = 1; candidate < 3; candidate++)
			if (high[candidate] - low[candidate] > high[axis] - low[axis]) axis = candidate;
		if (high[axis] - low[axis] <= 0) {
			// Every point coincides on this axis: a split would not make progress.
			this.axis[node] = -1;
			return node;
		}
		const middle = (from + to) >> 1;
		this.select(from, to, middle, axis);
		this.axis[node] = axis;
		this.split[node] = this.coordinate(this.order[middle], axis);
		this.left[node] = this.build(from, middle);
		this.right[node] = this.build(middle, to);
		return node;
	}

	/** Quickselect so `order[target]` holds the median without a full sort. */
	private select(from: number, to: number, target: number, axis: number): void {
		let low = from;
		let high = to - 1;
		while (low < high) {
			const pivot = this.coordinate(this.order[(low + high) >> 1], axis);
			let left = low;
			let right = high;
			while (left <= right) {
				while (this.coordinate(this.order[left], axis) < pivot) left++;
				while (this.coordinate(this.order[right], axis) > pivot) right--;
				if (left <= right) {
					const swap = this.order[left];
					this.order[left] = this.order[right];
					this.order[right] = swap;
					left++;
					right--;
				}
			}
			if (target <= right) high = right;
			else if (target >= left) low = left;
			else break;
		}
	}

	/**
	 * The index of the point nearest to an arbitrary coordinate, or -1 when the
	 * tree is empty.
	 *
	 * Distinct from {@link nearest}, which takes an index into the tree's own
	 * point set and so always finds that point first. This one is for querying
	 * from outside — comparing two clouds, for instance.
	 */
	/**
	 * Every point within `radius` of a coordinate, as index/squared-distance
	 * pairs in no particular order.
	 *
	 * The MLS surfaces need this rather than a k-nearest query: their weight
	 * function has compact support, so the neighbourhood is defined by a ball
	 * and the count varies from place to place.
	 */
	withinRadius(x: number, y: number, z: number, radius: number): Neighbour[] {
		const found: Neighbour[] = [];
		if (this.count === 0 || radius <= 0) return found;
		const limit = radius * radius;
		const stack: number[] = [0];
		while (stack.length > 0) {
			const node = stack.pop() as number;
			const axis = this.axis[node];
			if (axis < 0) {
				for (let slot = this.start[node]; slot < this.end[node]; slot++) {
					const point = this.order[slot];
					const dx = this.positions[3 * point] - x;
					const dy = this.positions[3 * point + 1] - y;
					const dz = this.positions[3 * point + 2] - z;
					const d2 = dx * dx + dy * dy + dz * dz;
					if (d2 <= limit) found.push({ index: point, squaredDistance: d2 });
				}
				continue;
			}
			const delta = [x, y, z][axis] - this.split[node];
			// Descend into the near side always, and into the far side only
			// when the splitting plane itself falls inside the ball.
			if (delta <= 0) {
				stack.push(this.left[node]);
				if (delta * delta <= limit) stack.push(this.right[node]);
			} else {
				stack.push(this.right[node]);
				if (delta * delta <= limit) stack.push(this.left[node]);
			}
		}
		return found;
	}

	nearestToPoint(x: number, y: number, z: number): number {
		if (this.count === 0) return -1;
		let best = -1;
		let bestDistance = Number.POSITIVE_INFINITY;

		const consider = (candidate: number) => {
			const dx = this.positions[candidate * 3] - x;
			const dy = this.positions[candidate * 3 + 1] - y;
			const dz = this.positions[candidate * 3 + 2] - z;
			const distance = dx * dx + dy * dy + dz * dz;
			if (distance < bestDistance) {
				bestDistance = distance;
				best = candidate;
			}
		};

		const visit = (node: number) => {
			if (this.axis[node] < 0) {
				for (let slot = this.start[node]; slot < this.end[node]; slot++) {
					consider(this.order[slot]);
				}
				return;
			}
			const axis = this.axis[node];
			const offset = (axis === 0 ? x : axis === 1 ? y : z) - this.split[node];
			const near = offset < 0 ? this.left[node] : this.right[node];
			const far = offset < 0 ? this.right[node] : this.left[node];
			visit(near);
			// Only cross the split when something on the far side could still beat
			// the best found so far.
			if (offset * offset < bestDistance) visit(far);
		};
		visit(0);
		return best;
	}

	/** The `k` points nearest to `point`, itself included, nearest first. */
	nearest(point: number, k: number): Int32Array {
		const wanted = Math.max(1, Math.min(k, this.count));
		const found = new Int32Array(wanted).fill(-1);
		const distances = new Float64Array(wanted).fill(Infinity);
		const target = [
			this.positions[point * 3],
			this.positions[point * 3 + 1],
			this.positions[point * 3 + 2],
		];
		let filled = 0;

		const consider = (candidate: number) => {
			const dx = this.positions[candidate * 3] - target[0];
			const dy = this.positions[candidate * 3 + 1] - target[1];
			const dz = this.positions[candidate * 3 + 2] - target[2];
			const distance = dx * dx + dy * dy + dz * dz;
			if (filled === wanted && distance >= distances[wanted - 1]) return;
			let slot = filled < wanted ? filled : wanted - 1;
			while (slot > 0 && distances[slot - 1] > distance) {
				distances[slot] = distances[slot - 1];
				found[slot] = found[slot - 1];
				slot--;
			}
			distances[slot] = distance;
			found[slot] = candidate;
			if (filled < wanted) filled++;
		};

		const visit = (node: number) => {
			if (this.axis[node] < 0) {
				for (let slot = this.start[node]; slot < this.end[node]; slot++) consider(this.order[slot]);
				return;
			}
			const axis = this.axis[node];
			const offset = target[axis] - this.split[node];
			const near = offset < 0 ? this.left[node] : this.right[node];
			const far = offset < 0 ? this.right[node] : this.left[node];
			visit(near);
			if (filled < wanted || offset * offset < distances[wanted - 1]) visit(far);
		};
		visit(0);
		return found;
	}
}
