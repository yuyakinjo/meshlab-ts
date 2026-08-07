/** An axis-aligned bounding box, mirroring `vcg::Box3`. */
export class Box3 {
	min: [number, number, number];
	max: [number, number, number];

	private constructor(min: [number, number, number], max: [number, number, number]) {
		this.min = min;
		this.max = max;
	}

	/**
	 * The empty box, encoded as VCG does it: min above max, so that the first
	 * `add()` sets both corners without a special case.
	 */
	static empty(): Box3 {
		return new Box3(
			[Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
			[Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
		);
	}

	static fromCorners(
		min: readonly [number, number, number],
		max: readonly [number, number, number],
	): Box3 {
		return new Box3([min[0], min[1], min[2]], [max[0], max[1], max[2]]);
	}

	get isEmpty(): boolean {
		return this.min[0] > this.max[0] || this.min[1] > this.max[1] || this.min[2] > this.max[2];
	}

	setEmpty(): void {
		this.min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
		this.max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
	}

	add(x: number, y: number, z: number): void {
		if (x < this.min[0]) this.min[0] = x;
		if (y < this.min[1]) this.min[1] = y;
		if (z < this.min[2]) this.min[2] = z;
		if (x > this.max[0]) this.max[0] = x;
		if (y > this.max[1]) this.max[1] = y;
		if (z > this.max[2]) this.max[2] = z;
	}

	addBox(other: Box3): void {
		if (other.isEmpty) return;
		this.add(other.min[0], other.min[1], other.min[2]);
		this.add(other.max[0], other.max[1], other.max[2]);
	}

	get dimX(): number {
		return this.max[0] - this.min[0];
	}
	get dimY(): number {
		return this.max[1] - this.min[1];
	}
	get dimZ(): number {
		return this.max[2] - this.min[2];
	}

	get center(): [number, number, number] {
		return [
			(this.min[0] + this.max[0]) / 2,
			(this.min[1] + this.max[1]) / 2,
			(this.min[2] + this.max[2]) / 2,
		];
	}

	get diagonal(): number {
		if (this.isEmpty) return 0;
		const dx = this.dimX;
		const dy = this.dimY;
		const dz = this.dimZ;
		return Math.sqrt(dx * dx + dy * dy + dz * dz);
	}

	/** Length of the longest side. */
	get maxDim(): number {
		return Math.max(this.dimX, this.dimY, this.dimZ);
	}

	contains(x: number, y: number, z: number): boolean {
		return (
			x >= this.min[0] &&
			x <= this.max[0] &&
			y >= this.min[1] &&
			y <= this.max[1] &&
			z >= this.min[2] &&
			z <= this.max[2]
		);
	}

	/** Grows the box by `delta` on every side. */
	offset(delta: number): void {
		if (this.isEmpty) return;
		for (let i = 0; i < 3; i++) {
			this.min[i] -= delta;
			this.max[i] += delta;
		}
	}

	clone(): Box3 {
		return new Box3([...this.min], [...this.max]);
	}
}
