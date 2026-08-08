/**
 * A sparse symmetric solver, in the smallest form that does the job.
 *
 * Several things in this library end up as "solve a sparse symmetric positive
 * definite system": the heat-method geodesic distance, least-squares
 * conformal maps, any Poisson problem on a mesh. Conjugate gradients needs
 * only a matrix-vector product, converges in far fewer iterations than
 * Gauss-Seidel on these systems, and fits in a page — which is why there is
 * no factorisation here and no linear-algebra dependency.
 *
 * The matrix is built by accumulating triples and is stored row-wise. It is
 * the caller's job to hand over something symmetric and positive definite;
 * {@link solveCG} reports the residual it reached so a caller can tell a slow
 * convergence from a wrong matrix.
 */

/** A sparse matrix under construction, then in use. */
export class SparseMatrix {
	readonly size: number;
	private readonly rows: Array<Map<number, number>>;

	constructor(size: number) {
		if (size < 0 || !Number.isInteger(size)) {
			throw new Error(`a matrix needs a non-negative integer size, got ${size}`);
		}
		this.size = size;
		this.rows = Array.from({ length: size }, () => new Map<number, number>());
	}

	/** Accumulates into an entry, as assembling from element contributions does. */
	add(row: number, col: number, value: number): void {
		if (value === 0) return;
		this.rows[row].set(col, (this.rows[row].get(col) ?? 0) + value);
	}

	get(row: number, col: number): number {
		return this.rows[row].get(col) ?? 0;
	}

	/** `out = A · x`. */
	multiply(x: Float64Array, out: Float64Array): void {
		for (let r = 0; r < this.size; r++) {
			let sum = 0;
			for (const [c, v] of this.rows[r]) sum += v * x[c];
			out[r] = sum;
		}
	}

	/**
	 * Replaces a row and column with the identity, moving its contribution to
	 * the right-hand side.
	 *
	 * This is how a Dirichlet condition is imposed without renumbering the
	 * unknowns: the pinned value stays in the solution vector where every
	 * caller expects it, and the matrix stays symmetric — which conjugate
	 * gradients requires and which simply zeroing the row would break.
	 */
	pin(index: number, value: number, rhs: Float64Array): void {
		for (const [c, v] of this.rows[index]) {
			if (c === index) continue;
			rhs[c] -= v * value;
			this.rows[c].delete(index);
		}
		this.rows[index].clear();
		this.rows[index].set(index, 1);
		rhs[index] = value;
	}

	/** The diagonal, for Jacobi preconditioning. */
	diagonal(): Float64Array {
		const out = new Float64Array(this.size);
		for (let r = 0; r < this.size; r++) out[r] = this.rows[r].get(r) ?? 0;
		return out;
	}
}

export interface SolveResult {
	readonly x: Float64Array;
	readonly iterations: number;
	/** The final residual norm, relative to the right-hand side's. */
	readonly residual: number;
	readonly converged: boolean;
}

/**
 * Jacobi-preconditioned conjugate gradients.
 *
 * The preconditioner is just the diagonal, which for a mesh Laplacian is
 * where most of the conditioning trouble lives — vertices of wildly different
 * valence or area. It costs one division per unknown per iteration and
 * typically halves the iteration count.
 */
export function solveCG(
	a: SparseMatrix,
	b: Float64Array,
	options: { iterations?: number; tolerance?: number; initial?: Float64Array } = {},
): SolveResult {
	const n = a.size;
	const maxIterations = options.iterations ?? Math.max(100, 4 * n);
	const tolerance = options.tolerance ?? 1e-10;

	const x = options.initial ? Float64Array.from(options.initial) : new Float64Array(n);
	const diagonal = a.diagonal();
	// A zero on the diagonal would divide by nothing; fall back to no
	// preconditioning for that row rather than producing NaN everywhere.
	for (let i = 0; i < n; i++) if (diagonal[i] === 0) diagonal[i] = 1;

	const r = new Float64Array(n);
	const z = new Float64Array(n);
	const p = new Float64Array(n);
	const ap = new Float64Array(n);

	a.multiply(x, ap);
	for (let i = 0; i < n; i++) r[i] = b[i] - ap[i];
	const bNorm = Math.sqrt(dot(b, b)) || 1;

	for (let i = 0; i < n; i++) z[i] = r[i] / diagonal[i];
	p.set(z);
	let rz = dot(r, z);

	let iteration = 0;
	for (; iteration < maxIterations; iteration++) {
		const residual = Math.sqrt(dot(r, r)) / bNorm;
		if (residual <= tolerance) {
			return { x, iterations: iteration, residual, converged: true };
		}
		a.multiply(p, ap);
		const pap = dot(p, ap);
		if (pap === 0) break; // the search direction died: not positive definite
		const alpha = rz / pap;
		for (let i = 0; i < n; i++) {
			x[i] += alpha * p[i];
			r[i] -= alpha * ap[i];
		}
		for (let i = 0; i < n; i++) z[i] = r[i] / diagonal[i];
		const rzNext = dot(r, z);
		const beta = rz === 0 ? 0 : rzNext / rz;
		for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
		rz = rzNext;
	}
	const residual = Math.sqrt(dot(r, r)) / bNorm;
	return { x, iterations: iteration, residual, converged: residual <= tolerance };
}

function dot(a: Float64Array, b: Float64Array): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return sum;
}
