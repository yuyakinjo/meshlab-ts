/**
 * Moving least squares surfaces: APSS and RIMLS.
 *
 * Both define an implicit surface as the zero set of a scalar field fitted
 * locally to an oriented point cloud, and both share the same machinery: a
 * per-point support radius, a compactly supported weight, and a neighbourhood
 * query that returns whichever points reach the query location.
 *
 * They differ in what they fit.
 *
 * - **APSS** (Guennebaud and Gross, Siggraph 2007) fits an *algebraic sphere*
 *   — the quadric `u₀ + u·x + u₄|x|²` constrained so its gradient matches the
 *   input normals. Fitting a sphere rather than a plane is what lets it stay
 *   stable where the surface curves sharply relative to the sample spacing.
 * - **RIMLS** (Öztireli, Guennebaud and Gross, Eurographics 2009) fits a plane
 *   through iteratively reweighted regression, down-weighting samples whose
 *   normal disagrees with the current gradient. That is what preserves a
 *   crease: the two sides of the edge stop averaging into each other.
 *
 * The weight is VCGLib's: `(1 - d²/r²)⁴` inside the support and zero outside,
 * with `r` the point's own radius times a filter scale. It is C² and compact,
 * which the derivative expressions below rely on.
 */

import { MLException } from "../../common/utilities/ml_exception.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { KdTree, type Neighbour } from "../../vcg/space/index/kdtree.ts";

export type Vec3 = [number, number, number];

export const MLS_DEFAULTS = {
	filterScale: 2,
	projectionAccuracy: 1e-4,
	maxProjectionIters: 15,
	sphericalParameter: 1,
	sigmaN: 0.75,
	maxRefittingIters: 3,
	minRefittingIters: 1,
	refittingThreshold: 1e-3,
	/** How many neighbours the density estimate averages over. */
	radiusNeighbours: 16,
} as const;

/**
 * A per-vertex support radius from the local point density.
 *
 * `2·sqrt(d²ₖ / k)` for the k-th nearest neighbour distance: the radius of a
 * disc that would hold k points at the local density, doubled so neighbouring
 * supports overlap. VCGLib's `computeVertexRadius`.
 */
export function estimateRadii(
	cm: CMeshO,
	neighbours: number = MLS_DEFAULTS.radiusNeighbours,
): Float64Array {
	const live: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) live.push(v);
	const out = new Float64Array(cm.vertSize);
	if (live.length === 0) return out;
	if (live.length === 1) {
		out[live[0]] = 1;
		return out;
	}

	const k = Math.min(neighbours, live.length - 1);
	const tree = new KdTree(cm.vertCoord, cm.vertSize);
	for (const v of live) {
		// `nearest` includes the query point itself, so ask for one more and
		// take the farthest of what comes back.
		const found = tree.nearest(v, k + 1);
		let worst = 0;
		for (const w of found) {
			const d2 =
				(cm.vx(w) - cm.vx(v)) ** 2 + (cm.vy(w) - cm.vy(v)) ** 2 + (cm.vz(w) - cm.vz(v)) ** 2;
			if (d2 > worst) worst = d2;
		}
		out[v] = 2 * Math.sqrt(worst / k);
	}
	return out;
}

/** The mean distance between neighbouring points, used to scale tolerances. */
export function averagePointSpacing(radii: Float64Array, cm: CMeshO): number {
	let sum = 0;
	let count = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v) || radii[v] === 0) continue;
		sum += radii[v];
		count++;
	}
	return count === 0 ? 1 : sum / count / 2;
}

interface Weighted {
	readonly index: number;
	readonly weight: number;
	/** d(weight)/d(distance²), used to build the weight's spatial gradient. */
	readonly derivative: number;
}

/** What the two surfaces share: the cloud, the radii and the weighting. */
export abstract class MlsSurface {
	protected readonly tree: KdTree;
	protected readonly radii: Float64Array;
	protected readonly maxRadius: number;
	readonly averageSpacing: number;

	filterScale: number = MLS_DEFAULTS.filterScale;
	projectionAccuracy: number = MLS_DEFAULTS.projectionAccuracy;
	maxProjectionIters: number = MLS_DEFAULTS.maxProjectionIters;

	constructor(
		protected readonly cm: CMeshO,
		radii?: Float64Array,
	) {
		if (cm.vn === 0) throw new MLException("an MLS surface needs at least one point");
		this.radii = radii ?? estimateRadii(cm);
		this.tree = new KdTree(cm.vertCoord, cm.vertSize);
		let maxRadius = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v) && this.radii[v] > maxRadius) maxRadius = this.radii[v];
		}
		this.maxRadius = maxRadius;
		this.averageSpacing = averagePointSpacing(this.radii, cm);
	}

	/**
	 * The points whose own support reaches `x`, with their weights.
	 *
	 * The kd-tree is asked for everything inside the *largest* support in the
	 * cloud and the per-point radius then rejects the rest. A ball tree would
	 * prune better on a cloud with wildly varying density, but it is a second
	 * index to build and keep correct, and the rejection here is a subtraction.
	 */
	protected neighbourhood(x: number, y: number, z: number): Weighted[] {
		const reach = this.maxRadius * this.filterScale;
		const found: Neighbour[] = this.tree.withinRadius(x, y, z, reach);
		const out: Weighted[] = [];
		for (const { index, squaredDistance } of found) {
			if (this.cm.isVertD(index)) continue;
			const support = this.radii[index] * this.filterScale;
			if (support <= 0) continue;
			const s = 1 / (support * support);
			const aux = 1 - squaredDistance * s;
			if (aux <= 0) continue;
			const w = aux * aux * aux * aux;
			out.push({ index, weight: w, derivative: -2 * s * (4 * aux * aux * aux) });
		}
		return out;
	}

	/** The scalar field whose zero set is the surface. */
	abstract potential(x: number, y: number, z: number): number | null;

	/** The field's gradient, which on the surface is the outward normal. */
	abstract gradient(x: number, y: number, z: number): Vec3 | null;

	/** The nearest point of the surface, and the normal there. */
	abstract project(x: number, y: number, z: number): { point: Vec3; normal: Vec3 } | null;

	/**
	 * Mean curvature from the field: the divergence of the normalised
	 * gradient, halved. Computed by central differences on the gradient, whose
	 * step is a fraction of the point spacing.
	 */
	meanCurvature(x: number, y: number, z: number): number | null {
		const h = this.averageSpacing * 0.05;
		const g = this.gradient(x, y, z);
		if (g === null) return null;
		const length = Math.hypot(g[0], g[1], g[2]);
		if (length === 0) return null;

		const hessian: number[][] = [
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0],
		];
		const at: Vec3 = [x, y, z];
		for (let axis = 0; axis < 3; axis++) {
			const plus = [...at] as Vec3;
			const minus = [...at] as Vec3;
			plus[axis] += h;
			minus[axis] -= h;
			const gp = this.gradient(plus[0], plus[1], plus[2]);
			const gm = this.gradient(minus[0], minus[1], minus[2]);
			if (gp === null || gm === null) return null;
			for (let row = 0; row < 3; row++) hessian[row][axis] = (gp[row] - gm[row]) / (2 * h);
		}

		// (|g|² tr(H) - gᵀHg) / (2|g|³)
		const trace = hessian[0][0] + hessian[1][1] + hessian[2][2];
		let quadratic = 0;
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++) quadratic += g[i] * hessian[i][j] * g[j];
		}
		return (length * length * trace - quadratic) / (2 * length ** 3);
	}
}

// ---- APSS -----------------------------------------------------------------

enum Fit {
	Sphere = 0,
	Plane = 1,
	Undetermined = 2,
}

/** The algebraic sphere fitted at one query point. */
interface Algebraic {
	status: Fit;
	uConstant: number;
	uLinear: Vec3;
	uQuad: number;
	centre: Vec3;
	radius: number;
}

export class Apss extends MlsSurface {
	sphericalParameter: number = MLS_DEFAULTS.sphericalParameter;

	/**
	 * Fits `u₀ + u·x + u₄|x|²` to the neighbourhood, in the algebraic
	 * normalisation where `|u|² - 4u₀u₄ = 1`.
	 *
	 * The closed form is Guennebaud and Gross's: `u₄` comes from the weighted
	 * covariance between the points and their normals, and the linear and
	 * constant parts follow. With `sphericalParameter` at zero `u₄` vanishes
	 * and the fit degenerates to the plane through the weighted mean.
	 */
	private fit(x: number, y: number, z: number): Algebraic | null {
		const samples = this.neighbourhood(x, y, z);
		if (samples.length === 0) return null;

		if (samples.length === 1) {
			const id = samples[0].index;
			const n: Vec3 = [
				this.cm.vertNormal[3 * id],
				this.cm.vertNormal[3 * id + 1],
				this.cm.vertNormal[3 * id + 2],
			];
			const p: Vec3 = [this.cm.vx(id), this.cm.vy(id), this.cm.vz(id)];
			return {
				status: Fit.Plane,
				uLinear: n,
				uConstant: -(p[0] * n[0] + p[1] * n[1] + p[2] * n[2]),
				uQuad: 0,
				centre: [0, 0, 0],
				radius: 0,
			};
		}

		const sumP: Vec3 = [0, 0, 0];
		const sumN: Vec3 = [0, 0, 0];
		let sumDotPN = 0;
		let sumDotPP = 0;
		let sumW = 0;
		for (const { index, weight } of samples) {
			const px = this.cm.vx(index);
			const py = this.cm.vy(index);
			const pz = this.cm.vz(index);
			const nx = this.cm.vertNormal[3 * index];
			const ny = this.cm.vertNormal[3 * index + 1];
			const nz = this.cm.vertNormal[3 * index + 2];
			sumP[0] += px * weight;
			sumP[1] += py * weight;
			sumP[2] += pz * weight;
			sumN[0] += nx * weight;
			sumN[1] += ny * weight;
			sumN[2] += nz * weight;
			sumDotPN += weight * (px * nx + py * ny + pz * nz);
			sumDotPP += weight * (px * px + py * py + pz * pz);
			sumW += weight;
		}
		if (sumW === 0) return null;

		const invW = 1 / sumW;
		const numerator = sumDotPN - invW * dot(sumP, sumN);
		const denominator = sumDotPP - invW * dot(sumP, sumP);
		// A degenerate spread means every sample sits at one place; fall back
		// to the plane rather than dividing by it.
		const uQuad = denominator === 0 ? 0 : this.sphericalParameter * 0.5 * (numerator / denominator);
		const uLinear: Vec3 = [
			(sumN[0] - sumP[0] * 2 * uQuad) * invW,
			(sumN[1] - sumP[1] * 2 * uQuad) * invW,
			(sumN[2] - sumP[2] * 2 * uQuad) * invW,
		];
		const uConstant = -invW * (dot(uLinear, sumP) + sumDotPP * uQuad);

		const fit: Algebraic = {
			status: Fit.Undetermined,
			uConstant,
			uLinear,
			uQuad,
			centre: [0, 0, 0],
			radius: 0,
		};
		if (Math.abs(uQuad) > 1e-7) {
			const b = 1 / uQuad;
			fit.status = Fit.Sphere;
			fit.centre = [uLinear[0] * -0.5 * b, uLinear[1] * -0.5 * b, uLinear[2] * -0.5 * b];
			fit.radius = Math.sqrt(Math.max(0, dot(fit.centre, fit.centre) - b * uConstant));
		} else if (uQuad === 0) {
			const length = Math.hypot(uLinear[0], uLinear[1], uLinear[2]);
			if (length === 0) return null;
			fit.status = Fit.Plane;
			fit.uLinear = [uLinear[0] / length, uLinear[1] / length, uLinear[2] / length];
			fit.uConstant = uConstant / length;
		} else {
			// Between a plane and a sphere: renormalise so the field's gradient
			// has unit length on the surface, which makes the potential a
			// signed distance to first order.
			const f = 1 / Math.sqrt(Math.abs(dot(uLinear, uLinear) - 4 * uConstant * uQuad));
			fit.uConstant = uConstant * f;
			fit.uLinear = [uLinear[0] * f, uLinear[1] * f, uLinear[2] * f];
			fit.uQuad = uQuad * f;
		}
		return fit;
	}

	potential(x: number, y: number, z: number): number | null {
		const fit = this.fit(x, y, z);
		if (fit === null) return null;
		return evaluate(fit, [x, y, z]);
	}

	gradient(x: number, y: number, z: number): Vec3 | null {
		const fit = this.fit(x, y, z);
		if (fit === null) return null;
		return algebraicGradient(fit, [x, y, z]);
	}

	/** The mean curvature of the fitted sphere: the reciprocal of its radius. */
	approxMeanCurvature(x: number, y: number, z: number): number | null {
		const fit = this.fit(x, y, z);
		if (fit === null) return null;
		if (fit.status !== Fit.Sphere || fit.radius === 0) return 0;
		return (fit.uQuad > 0 ? 1 : -1) / fit.radius;
	}

	project(x: number, y: number, z: number): { point: Vec3; normal: Vec3 } | null {
		let position: Vec3 = [x, y, z];
		let normal: Vec3 = [0, 0, 1];
		const epsilon = this.averageSpacing * this.projectionAccuracy;
		const epsilon2 = epsilon * epsilon;
		let iterations = 0;

		for (;;) {
			const fit = this.fit(position[0], position[1], position[2]);
			if (fit === null) return null;
			const previous = position;

			if (fit.status === Fit.Sphere) {
				// Straight onto the sphere, from its centre through the point.
				const d = sub(position, fit.centre);
				const length = Math.hypot(d[0], d[1], d[2]);
				if (length === 0) return null;
				const unit: Vec3 = [d[0] / length, d[1] / length, d[2] / length];
				position = [
					fit.centre[0] + unit[0] * fit.radius,
					fit.centre[1] + unit[1] * fit.radius,
					fit.centre[2] + unit[2] * fit.radius,
				];
				normal = normalise(algebraicGradient(fit, position));
			} else if (fit.status === Fit.Plane) {
				const distance = dot(fit.uLinear, position) + fit.uConstant;
				position = [
					position[0] - fit.uLinear[0] * distance,
					position[1] - fit.uLinear[1] * distance,
					position[2] - fit.uLinear[2] * distance,
				];
				normal = fit.uLinear;
			} else {
				// Newton along the gradient, three steps: the quadric is not a
				// sphere here, so there is no closed form to land on.
				let p = position;
				const dir = normalise(algebraicGradient(fit, p));
				for (let i = 0; i < 3; i++) {
					const g = algebraicGradient(fit, p);
					const gl = Math.hypot(g[0], g[1], g[2]);
					if (gl === 0) break;
					const step = -evaluate(fit, p) * Math.min(1 / gl, 1);
					p = [p[0] + dir[0] * step, p[1] + dir[1] * step, p[2] + dir[2] * step];
				}
				position = p;
				normal = normalise(algebraicGradient(fit, position));
			}

			const moved = sub(previous, position);
			if (dot(moved, moved) <= epsilon2) break;
			if (++iterations >= this.maxProjectionIters) break;
		}
		return { point: position, normal };
	}
}

function evaluate(fit: Algebraic, p: Vec3): number {
	if (fit.status === Fit.Sphere) {
		const d = sub(p, fit.centre);
		const value = Math.hypot(d[0], d[1], d[2]) - fit.radius;
		return fit.uQuad < 0 ? -value : value;
	}
	if (fit.status === Fit.Plane) return dot(p, fit.uLinear) + fit.uConstant;
	return fit.uConstant + dot(p, fit.uLinear) + fit.uQuad * dot(p, p);
}

function algebraicGradient(fit: Algebraic, p: Vec3): Vec3 {
	if (fit.status === Fit.Plane) return fit.uLinear;
	return [
		fit.uLinear[0] + 2 * fit.uQuad * p[0],
		fit.uLinear[1] + 2 * fit.uQuad * p[1],
		fit.uLinear[2] + 2 * fit.uQuad * p[2],
	];
}

// ---- RIMLS ----------------------------------------------------------------

export class Rimls extends MlsSurface {
	sigmaN: number = MLS_DEFAULTS.sigmaN;
	maxRefittingIters: number = MLS_DEFAULTS.maxRefittingIters;
	minRefittingIters: number = MLS_DEFAULTS.minRefittingIters;
	refittingThreshold: number = MLS_DEFAULTS.refittingThreshold;

	/**
	 * Iteratively reweighted implicit MLS.
	 *
	 * The first pass is plain IMLS — the weighted mean of each sample's signed
	 * distance to its own tangent plane. Every pass after that multiplies the
	 * spatial weight by `exp(-|n - ∇f|²/σ²)`, so a sample whose normal points
	 * away from the current gradient stops contributing. Across a crease the
	 * two faces therefore separate instead of blending, which is the whole
	 * point of the method.
	 */
	private evaluate(x: number, y: number, z: number): { potential: number; gradient: Vec3 } | null {
		const samples = this.neighbourhood(x, y, z);
		if (samples.length === 0) return null;

		const invSigma2 = 1 / (this.sigmaN * this.sigmaN);
		let gradient: Vec3 = [0, 0, 0];
		let previous: Vec3 = [0, 0, 0];
		let potential = 0;
		let iterations = 0;

		for (;;) {
			previous = gradient;
			const sumGradW: Vec3 = [0, 0, 0];
			const sumGradWf: Vec3 = [0, 0, 0];
			const sumN: Vec3 = [0, 0, 0];
			let sumW = 0;
			potential = 0;

			for (const { index, weight, derivative } of samples) {
				const diff: Vec3 = [x - this.cm.vx(index), y - this.cm.vy(index), z - this.cm.vz(index)];
				const n: Vec3 = [
					this.cm.vertNormal[3 * index],
					this.cm.vertNormal[3 * index + 1],
					this.cm.vertNormal[3 * index + 2],
				];
				const f = dot(diff, n);

				let refitting = 1;
				if (iterations > 0) {
					const dn = sub(n, previous);
					refitting = Math.exp(-dot(dn, dn) * invSigma2);
				}
				const w = weight * refitting;
				// The weight's spatial gradient is dw/d(d²) times d(d²)/dx.
				const gw: Vec3 = [
					diff[0] * derivative * refitting,
					diff[1] * derivative * refitting,
					diff[2] * derivative * refitting,
				];

				for (let k = 0; k < 3; k++) {
					sumGradW[k] += gw[k];
					sumGradWf[k] += gw[k] * f;
					sumN[k] += n[k] * w;
				}
				potential += w * f;
				sumW += w;
			}

			if (sumW === 0) return null;
			potential /= sumW;
			gradient = [
				(-sumGradW[0] * potential + sumGradWf[0] + sumN[0]) / sumW,
				(-sumGradW[1] * potential + sumGradWf[1] + sumN[1]) / sumW,
				(-sumGradW[2] * potential + sumGradWf[2] + sumN[2]) / sumW,
			];
			iterations++;

			if (iterations < this.minRefittingIters) continue;
			const change = sub(gradient, previous);
			if (dot(change, change) <= this.refittingThreshold) break;
			if (iterations >= this.maxRefittingIters) break;
		}
		return { potential, gradient };
	}

	potential(x: number, y: number, z: number): number | null {
		return this.evaluate(x, y, z)?.potential ?? null;
	}

	gradient(x: number, y: number, z: number): Vec3 | null {
		return this.evaluate(x, y, z)?.gradient ?? null;
	}

	project(x: number, y: number, z: number): { point: Vec3; normal: Vec3 } | null {
		let position: Vec3 = [x, y, z];
		let normal: Vec3 = [0, 0, 1];
		const epsilon = this.averageSpacing * this.projectionAccuracy;
		let iterations = 0;

		for (;;) {
			const state = this.evaluate(position[0], position[1], position[2]);
			if (state === null) return null;
			normal = normalise(state.gradient);
			// The potential is a signed distance to first order, so one step
			// along the normal is a Newton step towards the zero set.
			position = [
				position[0] - normal[0] * state.potential,
				position[1] - normal[1] * state.potential,
				position[2] - normal[2] * state.potential,
			];
			if (Math.abs(state.potential) <= epsilon) break;
			if (++iterations >= this.maxProjectionIters) break;
		}
		return { point: position, normal };
	}
}

// ---- small vector helpers -------------------------------------------------

function dot(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub(a: Vec3, b: Vec3): Vec3 {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function normalise(v: Vec3): Vec3 {
	const length = Math.hypot(v[0], v[1], v[2]);
	return length === 0 ? [0, 0, 1] : [v[0] / length, v[1] / length, v[2] / length];
}
