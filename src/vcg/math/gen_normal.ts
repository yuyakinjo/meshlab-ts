/**
 * `vcg/math/gen_normal.h` — ways of scattering points evenly over the unit
 * sphere.
 *
 * MeshLab uses these both as directions to sample from and, through
 * `Points on a Sphere`, as point clouds in their own right. None of them is
 * random: each is a deterministic construction, and they differ in how even
 * the spacing comes out and in how exactly the requested count is honoured.
 */

export type SpherePoint = [number, number, number];

/**
 * The `i`th of `n` points of the Fibonacci spiral.
 *
 * The golden angle between consecutive points is what keeps them from lining
 * up into visible rows, and it gives the requested count exactly.
 */
export function fibonacciPoint(i: number, n: number): SpherePoint {
	const Phi = Math.sqrt(5) * 0.5 + 0.5;
	const phi = 2 * Math.PI * (i / Phi - Math.floor(i / Phi));
	const cosTheta = 1 - (2 * i + 1) / n;
	const sinTheta = Math.sqrt(Math.min(1, Math.max(0, 1 - cosTheta * cosTheta)));
	return [Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta];
}

export function fibonacci(n: number): SpherePoint[] {
	const out: SpherePoint[] = [];
	for (let i = 0; i < n; i++) out.push(fibonacciPoint(i, n));
	return out;
}

/**
 * Dave Rusin's disco ball: equally spaced rings from pole to pole, each ring
 * carrying as many points as fit at the same spacing.
 *
 * Visibly regular, unlike Fibonacci, and the count only approximates what was
 * asked for — the ring structure decides it.
 */
export function discoBall(pointNum: number): SpherePoint[] {
	let N = 1;
	for (; N < pointNum; N++) {
		const expected = 2 - (2 * N * Math.sin(Math.PI / N)) / (Math.cos(Math.PI / N) - 1);
		if (expected >= pointNum) break;
	}

	const verticalAngle = Math.PI / N;
	const out: SpherePoint[] = [[0, 0, 1]];
	for (let i = 1; i < N; i++) {
		const horizRadius = Math.sin(i * verticalAngle);
		const circleLength = 2 * Math.PI * horizRadius;
		const z = Math.cos(i * verticalAngle);
		const perCircle = Math.floor(circleLength / verticalAngle);
		const horizontalAngle = (2 * Math.PI) / perCircle;
		for (let j = 0; j < perCircle; j++) {
			out.push([
				Math.cos(j * horizontalAngle) * horizRadius,
				Math.sin(j * horizontalAngle) * horizRadius,
				z,
			]);
		}
	}
	out.push([0, 0, -1]);
	return out;
}

/**
 * One octant of a recursively subdivided octahedron, mirrored into the other
 * seven.
 *
 * The level is chosen so the total stays under the request, so asking for 500
 * points can hand back 258 — the construction only has certain sizes.
 */
export function recursiveOctahedron(pointNum: number): SpherePoint[] {
	let level = 10;
	while (4 ** level + 2 > pointNum) level--;
	if (level < 0) level = 0;

	const grid = octaLevel(level);
	// The mirroring writes the same point into several slots, so duplicates are
	// expected rather than a sign of trouble.
	const seen = new Set<string>();
	const out: SpherePoint[] = [];
	for (const p of grid) {
		const key = `${p[0]},${p[1]},${p[2]}`;
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(p);
	}
	return out;
}

/** The `(2^lev * 2 + 1)²` grid of the octahedron construction, one entry per slot. */
function octaLevel(lev: number): SpherePoint[] {
	const sz2 = 2 ** lev;
	const sz = sz2 * 2 + 1;
	const v: SpherePoint[] = Array.from({ length: sz * sz }, () => [0, 0, 0]);
	const at = (i: number, j: number) => v[i + sz2 + (j + sz2) * sz];
	const put = (i: number, j: number, p: readonly number[]) => {
		const slot = at(i, j);
		slot[0] = p[0];
		slot[1] = p[1];
		slot[2] = p[2];
	};

	if (lev === 0) {
		put(0, 0, [0, 0, 1]);
		put(1, 0, [1, 0, 0]);
		put(0, 1, [0, 1, 0]);
		return v;
	}

	const prev = octaLevel(lev - 1);
	const prevSz2 = 2 ** (lev - 1);
	const prevSz = prevSz2 * 2 + 1;
	const prevAt = (i: number, j: number) => prev[i + prevSz2 + (j + prevSz2) * prevSz];
	const mid = (a: readonly number[], b: readonly number[]): SpherePoint => [
		(a[0] + b[0]) / 2,
		(a[1] + b[1]) / 2,
		(a[2] + b[2]) / 2,
	];

	for (let i = 0; i <= sz2; i++) {
		for (let j = 0; j <= sz2 - i; j++) {
			if (i % 2 === 0 && j % 2 === 0) put(i, j, prevAt(i / 2, j / 2));
			else if (i % 2 !== 0 && j % 2 === 0) {
				put(i, j, mid(prevAt((i - 1) / 2, j / 2), prevAt((i + 1) / 2, j / 2)));
			} else if (i % 2 === 0 && j % 2 !== 0) {
				put(i, j, mid(prevAt(i / 2, (j - 1) / 2), prevAt(i / 2, (j + 1) / 2)));
			} else {
				put(i, j, mid(prevAt((i - 1) / 2, (j + 1) / 2), prevAt((i + 1) / 2, (j - 1) / 2)));
			}

			// The first octant is built; the other seven are its reflections.
			const p = at(i, j);
			put(sz2 - j, sz2 - i, [p[0], p[1], -p[2]]);
			put(-sz2 + j, sz2 - i, [-p[0], p[1], -p[2]]);
			put(sz2 - j, -sz2 + i, [p[0], -p[1], -p[2]]);
			put(-sz2 + j, -sz2 + i, [-p[0], -p[1], -p[2]]);
			put(-i, -j, [-p[0], -p[1], p[2]]);
			put(i, -j, [p[0], -p[1], p[2]]);
			put(-i, j, [-p[0], p[1], p[2]]);
		}
	}

	for (const p of v) {
		const len = Math.hypot(p[0], p[1], p[2]);
		if (len > 0) {
			p[0] /= len;
			p[1] /= len;
			p[2] /= len;
		}
	}
	return v;
}

export const GenNormal = {
	fibonacciPoint,
	fibonacci,
	discoBall,
	recursiveOctahedron,
} as const;
