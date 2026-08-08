/**
 * The symmetric 3x3 eigenproblem, and the two things built on it.
 *
 * Upstream reaches for Eigen's `SelfAdjointEigenSolver`. At 3x3 the cyclic
 * Jacobi rotation is a dozen lines and converges in a handful of sweeps, so
 * there is nothing to gain from a library — and a covariance or inertia matrix
 * is the only shape either caller ever passes.
 */

/** Eigenvalues ascending, with the matching unit eigenvectors as columns. */
export interface Eigen3 {
	/** Ascending. */
	readonly values: readonly number[];
	/** `vectors[i]` is the unit eigenvector for `values[i]`. */
	readonly vectors: ReadonlyArray<readonly number[]>;
}

/**
 * Cyclic Jacobi on a symmetric 3x3 matrix, given row-major.
 *
 * Each sweep zeroes the three off-diagonal entries in turn with a plane
 * rotation; the rotations accumulate into the eigenvector matrix. Symmetry is
 * assumed, not checked — the callers build the matrix from an outer product,
 * where it holds by construction.
 */
export function symmetricEigen3(matrix: readonly number[]): Eigen3 {
	const a = [
		[matrix[0], matrix[1], matrix[2]],
		[matrix[3], matrix[4], matrix[5]],
		[matrix[6], matrix[7], matrix[8]],
	];
	// Columns of `v` accumulate the eigenvectors.
	const v = [
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	];

	for (let sweep = 0; sweep < 64; sweep++) {
		const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
		if (off < 1e-18) break;
		for (const [p, q] of [
			[0, 1],
			[0, 2],
			[1, 2],
		] as const) {
			if (Math.abs(a[p][q]) < 1e-300) continue;
			// The rotation angle that annihilates a[p][q].
			const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
			const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
			const c = 1 / Math.sqrt(t * t + 1);
			const s = t * c;

			for (let k = 0; k < 3; k++) {
				const akp = a[k][p];
				const akq = a[k][q];
				a[k][p] = c * akp - s * akq;
				a[k][q] = s * akp + c * akq;
			}
			for (let k = 0; k < 3; k++) {
				const apk = a[p][k];
				const aqk = a[q][k];
				a[p][k] = c * apk - s * aqk;
				a[q][k] = s * apk + c * aqk;
			}
			for (let k = 0; k < 3; k++) {
				const vkp = v[k][p];
				const vkq = v[k][q];
				v[k][p] = c * vkp - s * vkq;
				v[k][q] = s * vkp + c * vkq;
			}
		}
	}

	const order = [0, 1, 2].sort((x, y) => a[x][x] - a[y][y]);
	return {
		values: order.map((i) => a[i][i]),
		vectors: order.map((i) => {
			const col = [v[0][i], v[1][i], v[2][i]];
			const len = Math.hypot(col[0], col[1], col[2]) || 1;
			return col.map((x) => x / len);
		}),
	};
}

/** The covariance of a point set about `centre`, row-major. */
export function covariance(
	points: ReadonlyArray<readonly number[]>,
	centre: readonly number[],
): number[] {
	const c = [0, 0, 0, 0, 0, 0, 0, 0, 0];
	for (const p of points) {
		const d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
		for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) c[3 * i + j] += d[i] * d[j];
	}
	if (points.length > 0) for (let k = 0; k < 9; k++) c[k] /= points.length;
	return c;
}

/** A plane through a point set: unit normal and offset, least-squares. */
export interface FittedPlane {
	readonly normal: readonly number[];
	/** A point `p` is on the plane when `dot(normal, p) === offset`. */
	readonly offset: number;
	readonly centre: readonly number[];
}

/**
 * The plane minimising the squared distance to `points`.
 *
 * The normal is the eigenvector of the covariance with the *smallest*
 * eigenvalue — the direction the points vary least in, which is exactly the
 * direction a plane through them should face.
 */
export function fitPlaneToPointSet(points: ReadonlyArray<readonly number[]>): FittedPlane | null {
	if (points.length < 3) return null;
	const centre = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
	const { vectors } = symmetricEigen3(covariance(points, centre));
	const normal = vectors[0];
	return {
		normal,
		offset: normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2],
		centre,
	};
}

export const Eigen3Ops = { symmetricEigen3, covariance, fitPlaneToPointSet } as const;
