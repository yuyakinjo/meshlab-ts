/**
 * Non-allocating 3-vector arithmetic over flat typed arrays.
 *
 * Every function takes a base array and an element index rather than an object,
 * so the mesh kernel can do vector math straight against `vertCoord` without
 * materialising a `Point3` per vertex. `Point3` exists for cold paths; nothing
 * in a hot loop should use it.
 */

export type Vec3Like = Float64Array | number[];

export function cross(
	ax: number,
	ay: number,
	az: number,
	bx: number,
	by: number,
	bz: number,
	out: Vec3Like,
): void {
	out[0] = ay * bz - az * by;
	out[1] = az * bx - ax * bz;
	out[2] = ax * by - ay * bx;
}

export function dot(
	ax: number,
	ay: number,
	az: number,
	bx: number,
	by: number,
	bz: number,
): number {
	return ax * bx + ay * by + az * bz;
}

export function norm(x: number, y: number, z: number): number {
	return Math.hypot(x, y, z);
}

export function squaredNorm(x: number, y: number, z: number): number {
	return x * x + y * y + z * z;
}

/**
 * Normalises the 3 values at `base + 3 * i` in place.
 *
 * Two robustness details, both of which matter on real scan data:
 *
 * - A zero-length vector is left alone rather than turned into NaN. Degenerate
 *   faces are common, and one NaN normal poisons every average it feeds.
 * - The components are divided by the largest of them before the length is
 *   taken. Without that pre-scaling a face normal whose components are
 *   subnormal — which a cross product of tiny coordinates produces — loses
 *   most of its precision in `hypot`, and the "unit" vector comes out with a
 *   length noticeably different from 1.
 */
export function normalizeAt(arr: Float64Array, i: number): void {
	const o = 3 * i;
	const x = arr[o];
	const y = arr[o + 1];
	const z = arr[o + 2];
	const scale = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
	if (scale === 0 || !Number.isFinite(scale)) return;
	const sx = x / scale;
	const sy = y / scale;
	const sz = z / scale;
	const len = Math.sqrt(sx * sx + sy * sy + sz * sz);
	if (len === 0) return;
	arr[o] = sx / len;
	arr[o + 1] = sy / len;
	arr[o + 2] = sz / len;
}

/** Adds (x, y, z) into the 3 values at `base + 3 * i`. */
export function addAt(arr: Float64Array, i: number, x: number, y: number, z: number): void {
	const o = 3 * i;
	arr[o] += x;
	arr[o + 1] += y;
	arr[o + 2] += z;
}

export function setAt(arr: Float64Array, i: number, x: number, y: number, z: number): void {
	const o = 3 * i;
	arr[o] = x;
	arr[o + 1] = y;
	arr[o + 2] = z;
}

/** A 3-vector value type, for readable code away from the hot loops. */
export class Point3 {
	constructor(
		public x = 0,
		public y = 0,
		public z = 0,
	) {}

	static from(arr: Float64Array, i: number): Point3 {
		return new Point3(arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]);
	}

	get norm(): number {
		return Math.hypot(this.x, this.y, this.z);
	}

	toArray(): [number, number, number] {
		return [this.x, this.y, this.z];
	}
}
