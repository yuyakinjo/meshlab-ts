/**
 * A lazily-invalidated priority queue, mirroring `vcg::LocalOptimization`.
 *
 * Every local mesh modification changes the cost of its neighbours, and there
 * can be millions of them. Rather than find and update each affected entry —
 * which needs an index into the heap and makes every operation costlier —
 * VCGLib pushes a *new* entry with the new cost and lets the old one rot. When
 * a stale entry surfaces it is recognised and dropped.
 *
 * The recognition is the whole trick: each entry records the version of the
 * elements it touched, and any modification bumps those versions. An entry
 * whose recorded versions no longer match describes a mesh that no longer
 * exists.
 */

/** An operation waiting in the queue. */
export interface HeapEntry {
	/** Lower runs first. */
	readonly priority: number;
	/**
	 * Tie-break, so a run is reproducible.
	 *
	 * Floating-point priorities collide constantly on symmetric meshes, and
	 * without a deterministic second key the output would depend on heap
	 * implementation details and differ between runs.
	 */
	readonly tieBreak: number;
}

export class LazyPriorityQueue<T extends HeapEntry> {
	private readonly heap: T[] = [];

	get size(): number {
		return this.heap.length;
	}

	get isEmpty(): boolean {
		return this.heap.length === 0;
	}

	push(entry: T): void {
		this.heap.push(entry);
		this.siftUp(this.heap.length - 1);
	}

	/** The lowest-priority entry, without removing it. */
	peek(): T | undefined {
		return this.heap[0];
	}

	pop(): T | undefined {
		const n = this.heap.length;
		if (n === 0) return undefined;
		const top = this.heap[0];
		const last = this.heap.pop() as T;
		if (n > 1) {
			this.heap[0] = last;
			this.siftDown(0);
		}
		return top;
	}

	clear(): void {
		this.heap.length = 0;
	}

	private before(a: T, b: T): boolean {
		if (a.priority !== b.priority) return a.priority < b.priority;
		return a.tieBreak < b.tieBreak;
	}

	private siftUp(start: number): void {
		let i = start;
		const item = this.heap[i];
		while (i > 0) {
			const parent = (i - 1) >> 1;
			if (!this.before(item, this.heap[parent])) break;
			this.heap[i] = this.heap[parent];
			i = parent;
		}
		this.heap[i] = item;
	}

	private siftDown(start: number): void {
		const n = this.heap.length;
		let i = start;
		const item = this.heap[i];
		for (;;) {
			const left = 2 * i + 1;
			if (left >= n) break;
			const right = left + 1;
			const child = right < n && this.before(this.heap[right], this.heap[left]) ? right : left;
			if (!this.before(this.heap[child], item)) break;
			this.heap[i] = this.heap[child];
			i = child;
		}
		this.heap[i] = item;
	}
}

/** Why an optimisation run stopped. */
export type TerminationReason = "goalReached" | "exhausted" | "operationLimit" | "canceled";

export interface OptimizationResult {
	/** Modifications actually applied. */
	performed: number;
	/** Queue entries discarded as stale or infeasible. */
	discarded: number;
	reason: TerminationReason;
}
