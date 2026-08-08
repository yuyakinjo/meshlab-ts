/**
 * Making a mesh developable, after Stein, Grinspun and Crane (SIGGRAPH 2018).
 *
 * A surface is developable where it can be flattened into the plane without
 * stretching — a cylinder or a cone can be, a sphere cannot. The paper's
 * observation is that for a triangle mesh this is a condition on each vertex's
 * *star*: the surface is developable at a vertex exactly when the star's face
 * normals fall into two groups, each of which is internally flat. That is a
 * hinge (a crease) or a flat spot, and nothing else.
 *
 * So the energy at a vertex is: over every way of splitting the ordered star
 * into two contiguous runs, the smallest achievable "how much do the normals
 * within a run disagree". Driving that to zero everywhere turns the mesh into
 * developable patches joined along seam curves — which is what you want if the
 * thing is going to be cut out of sheet material and folded.
 *
 * Two divergences from upstream, both deliberate and both visible from here:
 *
 * 1. **The energy and the gradient come from different functionals.** MeshLab
 *    ships with `FILTERDEVELOPABILITY_AVOID_BRANCHING` defined, which makes the
 *    reported energy a *maximum* over pairs of normals, while the gradient it
 *    descends is always that of the *sum* over pairs. Reproduced as-is: it is
 *    the behaviour every MeshLab user gets, the sum is the paper's actual
 *    energy, and the max is only used to decide which partition wins and to
 *    report progress. {@link combinatorialEnergy} therefore is not the
 *    potential whose gradient {@link combinatorialEnergyGrad} returns, and the
 *    line search compares max-energies while stepping along a sum-gradient.
 * 2. **The remeshing pass reads a real corner.** Upstream computes the previous
 *    corner as `(i - 1) % 3`, which is `-1` in C++ when `i` is 0 — an
 *    out-of-bounds read of the angle array. There is no way to reproduce
 *    undefined behaviour faithfully, so the intended corner `(i + 2) % 3` is
 *    used. The surrounding loop's `i < 2` bound is upstream's and is kept: a
 *    face whose only sharp corner is the third one is left for a neighbour to
 *    deal with.
 */
import { MLException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";
import {
	buildVertexFaces,
	collapseEdge,
	edgePairOf,
	flipEdge,
	linkCondition,
	sharedFaces,
} from "./edge_ops.ts";
import { VertexFlag } from "./flags.ts";
import { Pos } from "./pos.ts";
import { box as boundingBox } from "./update/bounding.ts";
import { faceFace } from "./update/topology.ts";

/** The ordered ring of faces around one vertex. */
export type Star = number[];

/**
 * A split of one star into two contiguous runs of faces.
 *
 * `[begin, begin + size)` is the first run; the rest of the ring is the second.
 */
export interface StarPartitioning {
	begin: number;
	size: number;
}

/**
 * The faces around each vertex, in the order they meet each other.
 *
 * A closed star comes back as a full ring. A border star comes back as the
 * open fan, which is enough: border vertices carry no energy of their own.
 * Requires FF adjacency.
 */
export function orderedStars(m: CMeshO): Star[] {
	if (m.ffFace === null) faceFace(m);
	const stars: Star[] = Array.from({ length: m.vertSize }, () => []);
	const seen = new Uint8Array(m.vertSize);

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = m.fv(f, k);
			if (seen[v] === 1) continue;
			seen[v] = 1;
			stars[v] = orderedStarAt(m, f, k, v);
		}
	}
	return stars;
}

/**
 * Rotates around `v` from face `f`, returning the faces in ring order.
 *
 * On a border the rotation first walks backwards to the border edge and then
 * forwards, so the fan comes back in one consistent direction rather than
 * broken at whichever face happened to be visited first.
 */
function orderedStarAt(m: CMeshO, f: number, k: number, v: number): Star {
	const guard = m.faceSize + 3;

	// Rotate until the ring closes or a border stops it. The rotation is the
	// same direction the walk below uses, so an open fan is found at its far
	// end and then enumerated back through the seed to the near end — which is
	// the whole fan, in one consistent direction.
	const back = new Pos(m, f, k, v);
	let onBorder = false;
	for (let step = 0; step < guard; step++) {
		back.flipE();
		if (back.isBorder()) {
			onBorder = true;
			break;
		}
		// `flipF` is a no-op on a border, so the border test above must come
		// first: otherwise the rotation stalls on the last face of a fan and
		// the "back at the start" test below reads it as a closed ring.
		back.flipF();
		if (back.f === f) break;
	}

	const start = onBorder ? back : new Pos(m, f, k, v);
	const star: Star = [];
	const cur = start.clone();
	for (let step = 0; step < guard; step++) {
		star.push(cur.f);
		// `nextE` rotates across the *other* edge at v, so on an open fan the
		// last face's far edge is the border that ends the walk.
		cur.flipE();
		if (cur.isBorder()) break;
		cur.flipF();
		if (cur.f === start.f) break;
	}
	return star;
}

/** Per-face unit normals and areas, computed together from one cross product. */
export function normalsAndAreas(m: CMeshO): { normals: Float64Array; areas: Float64Array } {
	const normals = new Float64Array(3 * m.faceSize);
	const areas = new Float64Array(m.faceSize);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const [a, b, c] = [0, 1, 2].map((k) => m.fv(f, k));
		const e1 = [m.vx(b) - m.vx(a), m.vy(b) - m.vy(a), m.vz(b) - m.vz(a)];
		const e2 = [m.vx(c) - m.vx(a), m.vy(c) - m.vy(a), m.vz(c) - m.vz(a)];
		const n = [
			e1[1] * e2[2] - e1[2] * e2[1],
			e1[2] * e2[0] - e1[0] * e2[2],
			e1[0] * e2[1] - e1[1] * e2[0],
		];
		const norm = Math.hypot(n[0], n[1], n[2]);
		areas[f] = norm / 2;
		// A degenerate face has no direction to give; leaving its normal at zero
		// makes it contribute nothing rather than a NaN that poisons the star.
		if (norm > 0) for (let i = 0; i < 3; i++) normals[3 * f + i] = n[i] / norm;
	}
	return { normals, areas };
}

/**
 * How much the normals disagree within one run of a partitioned star.
 *
 * Upstream's shipping configuration takes the largest squared difference over
 * all pairs in the run; the paper's energy is their sum weighted by `1/size²`.
 * The max is what is reported and compared, so it is what this returns — see
 * the file header.
 */
function regionNormalDeviation(
	star: Star,
	part: StarPartitioning,
	second: boolean,
	normals: Float64Array,
): number {
	const begin = second ? part.begin + part.size : part.begin;
	const size = second ? star.length - part.size : part.size;
	let worst = 0;
	for (let i = begin; i < begin + size - 1; i++) {
		const fi = star[i % star.length];
		for (let j = i + 1; j < begin + size; j++) {
			const fj = star[j % star.length];
			let sq = 0;
			for (let c = 0; c < 3; c++) {
				const d = normals[3 * fi + c] - normals[3 * fj + c];
				sq += d * d;
			}
			if (sq > worst) worst = sq;
		}
	}
	return worst;
}

/**
 * The energy at one vertex, and the split that achieves it.
 *
 * Zero for a border vertex and for any star of three faces or fewer: three
 * normals can always be read as a hinge, so there is nothing to fix. Otherwise
 * every contiguous two-way split is tried and the cheapest wins.
 */
export function localCombinatorialEnergy(
	m: CMeshO,
	v: number,
	star: Star,
	normals: Float64Array,
	out?: StarPartitioning,
): number {
	if (star.length <= 3 || (m.vertFlags[v] & VertexFlag.BORDER) !== 0) return 0;

	let energy = -1;
	const part: StarPartitioning = { begin: 0, size: 0 };
	for (part.size = 2; part.size <= star.length - 2; part.size++) {
		// The run is not allowed to wrap past the end of the ring; upstream is
		// explicit about this, and it costs only the splits that a rotation of
		// the same ring would find anyway.
		for (part.begin = 0; part.begin < star.length - part.size; part.begin++) {
			const here = Math.max(
				regionNormalDeviation(star, part, false, normals),
				regionNormalDeviation(star, part, true, normals),
			);
			if (energy < 0 || here < energy) {
				energy = here;
				if (out !== undefined) {
					out.begin = part.begin;
					out.size = part.size;
				}
			}
		}
	}
	return energy < 0 ? 0 : energy;
}

/** The whole mesh's energy: the sum of every vertex's. */
export function combinatorialEnergy(m: CMeshO, stars: Star[], normals: Float64Array): number {
	let total = 0;
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		total += localCombinatorialEnergy(m, v, stars[v], normals);
	}
	return total;
}

/**
 * The energy, and its gradient with respect to every vertex position.
 *
 * The gradient is written into `grad` (3 per vertex, zeroed first) and is the
 * gradient of the *sum-over-pairs* energy, not of the maximum this function
 * returns — see the file header.
 */
export function combinatorialEnergyGrad(
	m: CMeshO,
	stars: Star[],
	normals: Float64Array,
	areas: Float64Array,
	grad: Float64Array,
): number {
	grad.fill(0);
	let total = 0;
	const part: StarPartitioning = { begin: 0, size: 0 };

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const star = stars[v];
		total += localCombinatorialEnergy(m, v, star, normals, part);
		if (star.length <= 3 || (m.vertFlags[v] & VertexFlag.BORDER) !== 0) continue;
		regionDeviationGrad(m, star, part, false, normals, areas, grad);
		regionDeviationGrad(m, star, part, true, normals, areas, grad);
	}
	return total;
}

/** Accumulates one run's contribution to the gradient. */
function regionDeviationGrad(
	m: CMeshO,
	star: Star,
	part: StarPartitioning,
	second: boolean,
	normals: Float64Array,
	areas: Float64Array,
	grad: Float64Array,
): void {
	const begin = second ? part.begin + part.size : part.begin;
	const size = second ? star.length - part.size : part.size;
	const weight = 2 / (size * size);

	for (let i = begin; i < begin + size - 1; i++) {
		const fa = star[i % star.length];
		for (let j = i + 1; j < begin + size; j++) {
			const fb = star[j % star.length];
			const diff = [0, 1, 2].map((c) => normals[3 * fa + c] - normals[3 * fb + c]);
			// The pair pushes its two faces' normals apart in opposite senses.
			accumulateFaceNormalGrad(m, fa, diff, weight, normals, areas, grad);
			accumulateFaceNormalGrad(m, fb, diff, -weight, normals, areas, grad);
		}
	}
}

/**
 * Adds `scale · (∂n_f/∂p)ᵀ · diff` to the gradient of each of `f`'s vertices.
 *
 * The derivative of a unit face normal with respect to one of its corners is
 * `(e × n) nᵀ / area`, where `e` is the edge opposite that corner (paper,
 * appendix B.1). Transposed and applied to `diff` this collapses to a vector
 * along `n`, so no 3×3 matrix is ever formed.
 */
function accumulateFaceNormalGrad(
	m: CMeshO,
	f: number,
	diff: readonly number[],
	scale: number,
	normals: Float64Array,
	areas: Float64Array,
	grad: Float64Array,
): void {
	const area = areas[f];
	if (area <= 0) return;
	const n = [normals[3 * f], normals[3 * f + 1], normals[3 * f + 2]];

	for (let k = 0; k < 3; k++) {
		const v = m.fv(f, k);
		const v1 = m.fv(f, (k + 1) % 3);
		const v2 = m.fv(f, (k + 2) % 3);
		const e = [m.vx(v2) - m.vx(v1), m.vy(v2) - m.vy(v1), m.vz(v2) - m.vz(v1)];
		const cross = [e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]];
		const dot = cross[0] * diff[0] + cross[1] * diff[1] + cross[2] * diff[2];
		const factor = (scale * dot) / area;
		for (let c = 0; c < 3; c++) grad[3 * v + c] += factor * n[c];
	}
}

// ------------------------------------------------------------- remeshing

/** The interior angle at corner `k` of face `f`, in radians. */
function wedgeAngle(m: CMeshO, f: number, k: number): number {
	const v = m.fv(f, k);
	const a = m.fv(f, (k + 1) % 3);
	const b = m.fv(f, (k + 2) % 3);
	const u = [m.vx(a) - m.vx(v), m.vy(a) - m.vy(v), m.vz(a) - m.vz(v)];
	const w = [m.vx(b) - m.vx(v), m.vy(b) - m.vy(v), m.vz(b) - m.vz(v)];
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lw = Math.hypot(w[0], w[1], w[2]);
	if (lu === 0 || lw === 0) return 0;
	const cos = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (lu * lw);
	return Math.acos(Math.max(-1, Math.min(1, cos)));
}

/**
 * Removes the sharpest slivers, by flipping or collapsing their short edges.
 *
 * The optimization is unstable near a small interior angle, because the face
 * normal there swings wildly for a tiny motion of the vertex — and the whole
 * energy is built out of face normals. Two small angles in one face mean a
 * needle that a flip fixes; one means a cap that a collapse fixes.
 *
 * Returns true when the mesh changed, in which case the caller must rebuild
 * everything derived from it.
 */
export function removeSmallAngles(
	m: CMeshO,
	doFlip: boolean,
	doCollapse: boolean,
	angleThreshold: number,
): boolean {
	if (m.ffFace === null) faceFace(m);
	const vertFaces = buildVertexFaces(m);
	let changed = false;

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const angles = [0, 1, 2].map((k) => wedgeAngle(m, f, k));

		let edgeToFlip = -1;
		let edgeToCollapse = -1;
		// Upstream stops at corner 1; a face sharp only at corner 2 is left to
		// whichever neighbour also sees that angle.
		for (let i = 0; i < 2; i++) {
			if (angles[i] >= angleThreshold) continue;
			if (angles[(i + 1) % 3] < angleThreshold) edgeToFlip = i;
			else if (angles[(i + 2) % 3] < angleThreshold) edgeToFlip = (i + 2) % 3;
			else edgeToCollapse = (i + 1) % 3;
			break;
		}

		if (doFlip && edgeToFlip >= 0) {
			const pair = edgePairOf(m, vertFaces, m.fv(f, edgeToFlip), m.fv(f, (edgeToFlip + 1) % 3));
			if (pair !== null && flipEdge(m, vertFaces, pair)) {
				changed = true;
				continue;
			}
		}
		if (doCollapse && edgeToCollapse >= 0) {
			const u = m.fv(f, edgeToCollapse);
			const v = m.fv(f, (edgeToCollapse + 1) % 3);
			const shared = sharedFaces(m, vertFaces, u, v);
			if (!linkCondition(m, vertFaces, u, v, shared)) continue;
			// Onto the midpoint: the two ends are close by construction, so
			// which one survives barely matters, and the midpoint does not
			// favour either side of the sliver.
			collapseEdge(
				m,
				vertFaces,
				u,
				v,
				(m.vx(u) + m.vx(v)) / 2,
				(m.vy(u) + m.vy(v)) / 2,
				(m.vz(u) + m.vz(v)) / 2,
			);
			changed = true;
		}
	}

	if (changed) {
		Allocator.compactFaceVector(m);
		Allocator.compactVertexVector(m);
		faceFace(m);
	}
	return changed;
}

// ------------------------------------------------------------ optimizers

export interface DevelopabilityOptions {
	/** 0 = fixed step, 1 = backtracking line search. Upstream's default is 1. */
	readonly method?: 0 | 1;
	readonly maxFunEvals?: number;
	/** Stops once the squared gradient norm falls to or below this. */
	readonly eps?: number;
	readonly stepSize?: number;
	readonly minStepSize?: number;
	/** How much the line search shrinks the step on each rejection. */
	readonly tau?: number;
	/** The Armijo constant. */
	readonly m1?: number;
	readonly edgeFlips?: boolean;
	readonly edgeCollapses?: boolean;
	/** Radians; corners sharper than this are remeshed away. */
	readonly angleThreshold?: number;
	readonly onProgress?: (fraction: number) => void;
}

export interface DevelopabilityResult {
	readonly functionEvaluations: number;
	readonly energy: number;
	readonly gradientSqNorm: number;
	readonly remeshingRounds: number;
	readonly converged: boolean;
}

/**
 * Runs the developability optimization in place.
 *
 * The mesh is moved to the origin and scaled to a unit bounding-box diagonal
 * for the duration, because the energy's step sizes are absolute lengths and
 * would otherwise mean something different for every model. It is put back
 * afterwards.
 */
export function makeDevelopable(m: CMeshO, opts: DevelopabilityOptions = {}): DevelopabilityResult {
	const method = opts.method ?? 1;
	const maxFunEvals = opts.maxFunEvals ?? 400;
	const eps = opts.eps ?? 1e-5;
	const initialStep = opts.stepSize ?? 0.01;
	const minStep = opts.minStepSize ?? 1e-10;
	const tau = opts.tau ?? 0.8;
	const m1 = opts.m1 ?? 1e-4;
	const angleThreshold = opts.angleThreshold ?? (18 * Math.PI) / 180;

	let remeshingRounds = 0;
	if (removeSmallAngles(m, opts.edgeFlips ?? true, opts.edgeCollapses ?? true, angleThreshold)) {
		remeshingRounds++;
	}

	boundingBox(m);
	const diag = m.bbox.diagonal;
	if (!(diag > 0)) throw new MLException("the mesh has no extent to optimize");
	const centre = m.bbox.center;
	scaleAbout(m, centre, 1 / diag);

	const state = new Optimizer(m, method, initialStep, minStep, tau, m1);
	let converged = false;

	while (true) {
		if (state.nFunEval >= maxFunEvals || state.gradSqNorm <= eps) {
			converged = state.gradSqNorm <= eps;
			break;
		}
		if (!state.step(maxFunEvals)) break;
		opts.onProgress?.(Math.min(1, state.nFunEval / maxFunEvals));

		if (removeSmallAngles(m, opts.edgeFlips ?? true, opts.edgeCollapses ?? true, angleThreshold)) {
			remeshingRounds++;
			state.reset();
		}
	}

	scaleAbout(m, [0, 0, 0], diag);
	translate(m, centre);
	boundingBox(m);

	return {
		functionEvaluations: state.nFunEval,
		energy: state.energy,
		gradientSqNorm: state.gradSqNorm,
		remeshingRounds,
		converged,
	};
}

function scaleAbout(m: CMeshO, centre: readonly number[], factor: number): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		m.setVert(
			v,
			(m.vx(v) - centre[0]) * factor,
			(m.vy(v) - centre[1]) * factor,
			(m.vz(v) - centre[2]) * factor,
		);
	}
}

function translate(m: CMeshO, by: readonly number[]): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		m.setVert(v, m.vx(v) + by[0], m.vy(v) + by[1], m.vz(v) + by[2]);
	}
}

/**
 * Gradient descent on the developability energy.
 *
 * The two methods upstream offers differ only in how far they step: a fixed
 * distance, or the longest one that satisfies Armijo's sufficient-decrease
 * condition. The line search is the default because a fixed step that works
 * for one mesh diverges on another.
 */
class Optimizer {
	stars: Star[] = [];
	normals: Float64Array = new Float64Array(0);
	areas: Float64Array = new Float64Array(0);
	grad: Float64Array = new Float64Array(0);
	saved: Float64Array = new Float64Array(0);
	energy = 0;
	gradSqNorm = 0;
	nFunEval = 0;
	stepSize: number;

	constructor(
		private readonly m: CMeshO,
		private readonly method: 0 | 1,
		private readonly initialStep: number,
		private readonly minStep: number,
		private readonly tau: number,
		private readonly m1: number,
	) {
		this.stepSize = initialStep;
		this.reset();
	}

	/** Rebuilds everything derived from the mesh. Called after any remeshing. */
	reset(): void {
		const m = this.m;
		if (m.ffFace === null) faceFace(m);
		vertexBorderFromFaceAdj(m);
		this.stars = orderedStars(m);
		this.grad = new Float64Array(3 * m.vertSize);
		this.saved = new Float64Array(3 * m.vertSize);
		this.refresh();
		this.energy = combinatorialEnergyGrad(m, this.stars, this.normals, this.areas, this.grad);
		this.updateGradSqNorm();
		this.savePositions();
	}

	private refresh(): void {
		const derived = normalsAndAreas(this.m);
		this.normals = derived.normals;
		this.areas = derived.areas;
	}

	private updateGradSqNorm(): void {
		let sum = 0;
		for (let v = 0; v < this.m.vertSize; v++) {
			if (this.m.isVertD(v)) continue;
			for (let c = 0; c < 3; c++) sum += this.grad[3 * v + c] ** 2;
		}
		this.gradSqNorm = sum;
	}

	private savePositions(): void {
		for (let v = 0; v < this.m.vertSize; v++) {
			this.saved[3 * v] = this.m.vx(v);
			this.saved[3 * v + 1] = this.m.vy(v);
			this.saved[3 * v + 2] = this.m.vz(v);
		}
	}

	/** Moves every vertex `-step · grad` from its saved position. */
	private moveBy(step: number): void {
		for (let v = 0; v < this.m.vertSize; v++) {
			if (this.m.isVertD(v)) continue;
			this.m.setVert(
				v,
				this.saved[3 * v] - this.grad[3 * v] * step,
				this.saved[3 * v + 1] - this.grad[3 * v + 1] * step,
				this.saved[3 * v + 2] - this.grad[3 * v + 2] * step,
			);
		}
	}

	step(maxFunEvals: number): boolean {
		return this.method === 0 ? this.fixedStep() : this.backtrackingStep(maxFunEvals);
	}

	private fixedStep(): boolean {
		this.moveBy(this.initialStep);
		this.savePositions();
		this.refresh();
		this.energy = combinatorialEnergyGrad(this.m, this.stars, this.normals, this.areas, this.grad);
		this.updateGradSqNorm();
		this.nFunEval++;
		return true;
	}

	private backtrackingStep(maxFunEvals: number): boolean {
		// Along the search direction `-grad`, the directional derivative is
		// `grad · (-grad)` — minus the squared norm, which is already known.
		const slope = -this.gradSqNorm;
		let step = this.initialStep;
		let trialEnergy = this.energy;

		for (; step > this.minStep; step *= this.tau) {
			this.moveBy(step);
			this.refresh();
			trialEnergy = combinatorialEnergy(this.m, this.stars, this.normals);
			this.nFunEval++;

			if (trialEnergy <= this.energy + this.m1 * step * slope) break;

			if (this.nFunEval >= maxFunEvals) {
				// Out of budget mid-search: the last trial position was not
				// accepted, so it must not be kept.
				this.moveBy(0);
				this.refresh();
				return false;
			}
		}

		this.savePositions();
		this.stepSize = step;
		this.energy = trialEnergy;
		combinatorialEnergyGrad(this.m, this.stars, this.normals, this.areas, this.grad);
		this.updateGradSqNorm();
		this.nFunEval++;
		return true;
	}
}

/** Marks every vertex on a boundary edge, clearing the flag elsewhere. */
export function vertexBorderFromFaceAdj(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.BORDER;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			if (!m.isBorderFF(f, k)) continue;
			m.vertFlags[m.fv(f, k)] |= VertexFlag.BORDER;
			m.vertFlags[m.fv(f, (k + 1) % 3)] |= VertexFlag.BORDER;
		}
	}
}

export const Developability = {
	orderedStars,
	normalsAndAreas,
	localCombinatorialEnergy,
	combinatorialEnergy,
	combinatorialEnergyGrad,
	removeSmallAngles,
	makeDevelopable,
} as const;
