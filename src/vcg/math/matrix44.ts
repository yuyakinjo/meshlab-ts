/**
 * 4×4 transforms, mirroring `vcg::Matrix44`.
 *
 * Stored as 16 numbers in **row-major** order, so `m[4 * row + col]`, and
 * applied to a column vector on the right: `v' = M · v`. VCGLib uses the same
 * convention, which matters because the matrix is written into `.mlx` files
 * and read back.
 */

export type Matrix44 = Float64Array;

export function identity(): Matrix44 {
	const m = new Float64Array(16);
	m[0] = 1;
	m[5] = 1;
	m[10] = 1;
	m[15] = 1;
	return m;
}

export function fromArray(values: readonly number[]): Matrix44 {
	if (values.length !== 16) throw new Error(`a 4x4 matrix needs 16 values, got ${values.length}`);
	return new Float64Array(values);
}

export function scaling(sx: number, sy: number, sz: number): Matrix44 {
	const m = identity();
	m[0] = sx;
	m[5] = sy;
	m[10] = sz;
	return m;
}

export function translation(tx: number, ty: number, tz: number): Matrix44 {
	const m = identity();
	m[3] = tx;
	m[7] = ty;
	m[11] = tz;
	return m;
}

/** Rotation of `angleRad` about the unit axis `(x, y, z)`, by Rodrigues. */
export function rotation(angleRad: number, x: number, y: number, z: number): Matrix44 {
	const len = Math.hypot(x, y, z);
	if (len === 0) return identity();
	const ax = x / len;
	const ay = y / len;
	const az = z / len;
	const c = Math.cos(angleRad);
	const s = Math.sin(angleRad);
	const t = 1 - c;
	const m = identity();
	m[0] = t * ax * ax + c;
	m[1] = t * ax * ay - s * az;
	m[2] = t * ax * az + s * ay;
	m[4] = t * ax * ay + s * az;
	m[5] = t * ay * ay + c;
	m[6] = t * ay * az - s * ax;
	m[8] = t * ax * az - s * ay;
	m[9] = t * ay * az + s * ax;
	m[10] = t * az * az + c;
	return m;
}

/** `a · b`, applied right to left as usual. */
export function multiply(a: Matrix44, b: Matrix44): Matrix44 {
	const out = new Float64Array(16);
	for (let r = 0; r < 4; r++) {
		for (let c = 0; c < 4; c++) {
			let sum = 0;
			for (let k = 0; k < 4; k++) sum += a[4 * r + k] * b[4 * k + c];
			out[4 * r + c] = sum;
		}
	}
	return out;
}

/** Transforms a point, dividing through by w. */
export function transformPoint(
	m: Matrix44,
	x: number,
	y: number,
	z: number,
	out: Float64Array | number[],
): void {
	const px = m[0] * x + m[1] * y + m[2] * z + m[3];
	const py = m[4] * x + m[5] * y + m[6] * z + m[7];
	const pz = m[8] * x + m[9] * y + m[10] * z + m[11];
	const pw = m[12] * x + m[13] * y + m[14] * z + m[15];
	if (pw !== 0 && pw !== 1) {
		out[0] = px / pw;
		out[1] = py / pw;
		out[2] = pz / pw;
		return;
	}
	out[0] = px;
	out[1] = py;
	out[2] = pz;
}

export function isIdentity(m: Matrix44, epsilon = 0): boolean {
	const id = identity();
	for (let i = 0; i < 16; i++) if (Math.abs(m[i] - id[i]) > epsilon) return false;
	return true;
}

/** The determinant of the upper-left 3×3 block. */
export function determinant3(m: Matrix44): number {
	return (
		m[0] * (m[5] * m[10] - m[6] * m[9]) -
		m[1] * (m[4] * m[10] - m[6] * m[8]) +
		m[2] * (m[4] * m[9] - m[5] * m[8])
	);
}

/**
 * The inverse of a 4x4 matrix, or null when it is singular.
 *
 * Gauss-Jordan on the matrix beside an identity. General rather than
 * rigid-only, because a layer's transform may include a scale — and a caller
 * asking to invert one that is singular needs to be told, not handed NaNs.
 */
export function invert(m: Matrix44): Matrix44 | null {
	const a: number[][] = [];
	for (let r = 0; r < 4; r++) {
		a.push([...Array.from(m.subarray(4 * r, 4 * r + 4)), ...[0, 0, 0, 0]]);
		a[r][4 + r] = 1;
	}
	let scale = 1e-300;
	for (let k = 0; k < 16; k++) scale = Math.max(scale, Math.abs(m[k]));

	for (let col = 0; col < 4; col++) {
		let pivot = col;
		for (let r = col + 1; r < 4; r++) {
			if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
		}
		if (Math.abs(a[pivot][col]) < scale * 1e-14) return null;
		[a[col], a[pivot]] = [a[pivot], a[col]];
		const d = a[col][col];
		for (let k = 0; k < 8; k++) a[col][k] /= d;
		for (let r = 0; r < 4; r++) {
			if (r === col) continue;
			const factor = a[r][col];
			if (factor === 0) continue;
			for (let k = 0; k < 8; k++) a[r][k] -= factor * a[col][k];
		}
	}

	const out = new Float64Array(16) as Matrix44;
	for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[4 * r + c] = a[r][4 + c];
	return out;
}

export const Matrix44Ops = {
	invert,
	identity,
	fromArray,
	scaling,
	translation,
	rotation,
	multiply,
	transformPoint,
	isIdentity,
	determinant3,
} as const;
