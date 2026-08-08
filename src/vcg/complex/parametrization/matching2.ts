/**
 * Fitting a plane-to-plane transform between two matched point sets.
 *
 * When two texture charts are candidates for merging, their shared seam gives
 * two lists of UV coordinates for the same run of 3D vertices — one from each
 * chart's own parametrization. The question "how well would these two charts
 * fit together" is then "what single transform best takes one list onto the
 * other, and how much error is left over".
 *
 * Three answers, in increasing rigidity: affine (any linear map), similarity
 * (rotation, uniform scale, translation) and rigid (rotation and translation
 * only). The seam-removal driver uses the similarity fit to place one chart
 * against the other and the leftover error to decide whether the merge is
 * worth attempting at all.
 *
 * **Divergence from upstream, and it is a simplification rather than a
 * behaviour change.** MeshLab's `matching.cpp` reaches this through three
 * different Eigen paths — a full-pivot QR on a padded system, an eigenvalue
 * decomposition used to build `(MᵀM)^(-1/2)`, and a Jacobi SVD — each with its
 * own sign-correction afterwards. All three reduce to the same closed form in
 * two dimensions: the rotation is the polar factor of a 2×2 matrix, computed
 * once in {@link closestRotation2}. Same fit, one code path, and the
 * sign corrections upstream applies conditionally are unconditional here
 * because the 2×2 polar factor is always a proper rotation.
 */
import { closestRotation2, type Mat2, multiply2, solveSymmetric2 } from "../../math/mat2.ts";

/** A plane transform: `q ↦ M q + t`. */
export interface MatchingTransform {
	/** The linear part, row-major. */
	readonly m: Mat2;
	readonly tx: number;
	readonly ty: number;
}

export const IDENTITY_MATCHING: MatchingTransform = { m: [1, 0, 0, 1], tx: 0, ty: 0 };

/** A list of plane points, two numbers per point. */
export type Points2 = Float64Array;

export function applyMatching(t: MatchingTransform, x: number, y: number): [number, number] {
	return [t.m[0] * x + t.m[1] * y + t.tx, t.m[2] * x + t.m[3] * y + t.ty];
}

function centroid(p: Points2): [number, number] {
	const n = p.length / 2;
	let cx = 0;
	let cy = 0;
	for (let i = 0; i < n; i++) {
		cx += p[2 * i];
		cy += p[2 * i + 1];
	}
	return [cx / n, cy / n];
}

function checkPair(target: Points2, source: Points2): number {
	if (target.length !== source.length) {
		throw new Error(
			`matching needs two lists of the same length, got ${target.length / 2} and ${source.length / 2}`,
		);
	}
	const n = target.length / 2;
	if (n < 2) throw new Error(`matching needs at least two points, got ${n}`);
	return n;
}

/**
 * The affine map taking `source` onto `target` in the least-squares sense.
 *
 * The two rows of the linear part decouple — `x` and `y` are fitted
 * independently against the same 2×2 normal matrix — so this is one symmetric
 * solve reused twice. When the source points are collinear that matrix is
 * singular; {@link solveSymmetric2} then returns the minimum-norm answer
 * rather than an enormous one.
 */
export function matchAffine(target: Points2, source: Points2): MatchingTransform {
	const n = checkPair(target, source);
	const [ctx, cty] = centroid(target);
	const [csx, csy] = centroid(source);

	// The normal equations: N = Σ q qᵀ, with q the centred source points.
	let nxx = 0;
	let nxy = 0;
	let nyy = 0;
	let bxx = 0;
	let bxy = 0;
	let byx = 0;
	let byy = 0;
	for (let i = 0; i < n; i++) {
		const qx = source[2 * i] - csx;
		const qy = source[2 * i + 1] - csy;
		const px = target[2 * i] - ctx;
		const py = target[2 * i + 1] - cty;
		nxx += qx * qx;
		nxy += qx * qy;
		nyy += qy * qy;
		bxx += qx * px;
		bxy += qy * px;
		byx += qx * py;
		byy += qy * py;
	}
	const normal: Mat2 = [nxx, nxy, nxy, nyy];
	const [m00, m01] = solveSymmetric2(normal, bxx, bxy);
	const [m10, m11] = solveSymmetric2(normal, byx, byy);
	const m: Mat2 = [m00, m01, m10, m11];
	return {
		m,
		tx: ctx - (m[0] * csx + m[1] * csy),
		ty: cty - (m[2] * csx + m[3] * csy),
	};
}

/**
 * The best rotation, uniform scale and translation taking `source` onto
 * `target`.
 *
 * The scale is the ratio of the two point sets' spreads and the rotation is
 * the polar factor of their cross-covariance — the two are independent, which
 * is why this needs no iteration.
 */
export function matchSimilarity(target: Points2, source: Points2): MatchingTransform {
	const n = checkPair(target, source);
	const [ctx, cty] = centroid(target);
	const [csx, csy] = centroid(source);

	let targetSpread = 0;
	let sourceSpread = 0;
	let cxx = 0;
	let cxy = 0;
	let cyx = 0;
	let cyy = 0;
	for (let i = 0; i < n; i++) {
		const px = target[2 * i] - ctx;
		const py = target[2 * i + 1] - cty;
		const qx = source[2 * i] - csx;
		const qy = source[2 * i + 1] - csy;
		targetSpread += px * px + py * py;
		sourceSpread += qx * qx + qy * qy;
		cxx += px * qx;
		cxy += px * qy;
		cyx += py * qx;
		cyy += py * qy;
	}
	// A source that is a single repeated point has no spread to scale from.
	const scale = sourceSpread > 0 ? Math.sqrt(targetSpread / sourceSpread) : 1;
	const r = closestRotation2([cxx, cxy, cyx, cyy]);
	const m: Mat2 = [scale * r[0], scale * r[1], scale * r[2], scale * r[3]];
	return {
		m,
		tx: ctx - (m[0] * csx + m[1] * csy),
		ty: cty - (m[2] * csx + m[3] * csy),
	};
}

/**
 * The best rotation and translation taking `source` onto `target` — Kabsch,
 * in the plane.
 */
export function matchRigid(target: Points2, source: Points2): MatchingTransform {
	const n = checkPair(target, source);
	const [ctx, cty] = centroid(target);
	const [csx, csy] = centroid(source);

	let cxx = 0;
	let cxy = 0;
	let cyx = 0;
	let cyy = 0;
	for (let i = 0; i < n; i++) {
		const px = target[2 * i] - ctx;
		const py = target[2 * i + 1] - cty;
		const qx = source[2 * i] - csx;
		const qy = source[2 * i + 1] - csy;
		cxx += px * qx;
		cxy += px * qy;
		cyx += py * qx;
		cyy += py * qy;
	}
	const m = closestRotation2([cxx, cxy, cyx, cyy]);
	return {
		m,
		tx: ctx - (m[0] * csx + m[1] * csy),
		ty: cty - (m[2] * csx + m[3] * csy),
	};
}

/** The total distance left between `target` and the transformed `source`. */
export function matchingErrorTotal(t: MatchingTransform, target: Points2, source: Points2): number {
	const n = checkPair(target, source);
	let error = 0;
	for (let i = 0; i < n; i++) {
		const [x, y] = applyMatching(t, source[2 * i], source[2 * i + 1]);
		error += Math.hypot(target[2 * i] - x, target[2 * i + 1] - y);
	}
	return error;
}

/** The same, per point — what the merge threshold is expressed in. */
export function matchingError(t: MatchingTransform, target: Points2, source: Points2): number {
	return matchingErrorTotal(t, target, source) / (target.length / 2);
}

/** `a` after `b`: the transform applying `b` first. */
export function composeMatching(a: MatchingTransform, b: MatchingTransform): MatchingTransform {
	const m = multiply2(a.m, b.m);
	return {
		m,
		tx: a.m[0] * b.tx + a.m[1] * b.ty + a.tx,
		ty: a.m[2] * b.tx + a.m[3] * b.ty + a.ty,
	};
}

export const Matching2 = {
	matchAffine,
	matchSimilarity,
	matchRigid,
	matchingError,
	matchingErrorTotal,
	applyMatching,
	composeMatching,
} as const;
