/**
 * `filter_cubization` — Liu and Jacobson's cubic stylization.
 *
 * The mesh is deformed so that its normals snap towards the coordinate axes,
 * while every vertex star keeps its original shape as nearly as it can. The
 * result reads as the same object rebuilt out of cubes: the silhouette and
 * the proportions survive, the surface becomes axis-aligned facets.
 *
 * It is as-rigid-as-possible with one extra term. ARAP alone finds, per
 * vertex, the rotation that best explains how its star moved, then moves the
 * vertices to fit those rotations, and repeats. Cubic stylization adds an L1
 * penalty on the *rotated normal*: the ℓ¹ norm is minimised by making
 * components exactly zero, so a normal is pushed onto an axis rather than
 * merely towards one. That is why the output has flats and creases instead of
 * a gently rounded approximation of them.
 *
 * The L1 term has no closed form, so the local step is solved by ADMM — a
 * few iterations of "rotate, shrink, update the multiplier" per vertex.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { RichBool, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import { FilterArity, type FilterArityValue } from "../../common/plugins/filter_arity.ts";
import { FilterClass, type FilterClassMask } from "../../common/plugins/filter_class.ts";
import {
	type ActionIDType,
	type FilterOutput,
	FilterPlugin,
	type PostConditionBox,
} from "../../common/plugins/interfaces/filter_plugin.ts";
import type { CallBackPos } from "../../common/utilities/callback.ts";
import { MLException } from "../../common/utilities/ml_exception.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { SparseMatrix, solveCG } from "../../vcg/math/sparse.ts";

export const FP = { FP_CUBIZATION: 0 } as const;

export class FilterCubization extends FilterPlugin {
	pluginName(): string {
		return "FilterCubization";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.FP_CUBIZATION) this.wrongActionCalled(id);
		return "Cubic stylization";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.FP_CUBIZATION) this.wrongActionCalled(id);
		return "apply_coord_cubic_stylization";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.FP_CUBIZATION) this.wrongActionCalled(id);
		return (
			"Turn a mesh into a cube's style maintaining its original shape. For all details about " +
			"cubic stylization see: Hsueh-Ti Derek Liu and Alec Jacobson, Cubic Stylization, in ACM " +
			"Transactions on Graphics, 2019"
		);
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Remeshing;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override initParameterList(_id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichFloat("lcubeness", 0.2, {
				description: "Cubeness",
				tooltip:
					"How strongly the normals are pulled onto the axes, against keeping the original " +
					"shape. Zero leaves the mesh alone; large values make it a cube.",
			}),
		);
		list.add(
			new RichInt("iterations", 20, {
				description: "Iterations",
				tooltip: "How many local/global rounds to run.",
			}),
		);
		list.add(
			new RichBool("applycol", false, {
				description: "Colorize",
				tooltip: "Colour each vertex by how far it moved.",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		if (id !== FP.FP_CUBIZATION) return this.wrongActionCalled(id);
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces to stylize");
		const cubeness = params.getFloat("lcubeness");
		if (cubeness < 0) throw new MLException(`the cubeness cannot be negative, got ${cubeness}`);
		const iterations = params.getInt("iterations");
		if (iterations < 1)
			throw new MLException(`the iteration count must be at least 1, got ${iterations}`);

		UpdateBounding.box(cm);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const before = Float64Array.from(cm.vertCoord);
		const moved = cubicStylize(cm, cubeness, iterations, cb);

		if (params.getBool("applycol")) {
			m.updateDataMask(MeshElement.MM_VERTQUALITY);
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				cm.vertQuality[v] = Math.hypot(
					cm.vx(v) - before[3 * v],
					cm.vy(v) - before[3 * v + 1],
					cm.vz(v) - before[3 * v + 2],
				);
			}
		}
		m.updateBoxAndNormals();
		post.mask = MeshElement.MM_VERTCOORD;
		doc.Log.log(`Cubic stylization over ${iterations} iterations, mean displacement ${moved}`);
		return { iterations, mean_displacement: moved };
	}
}

/** One vertex's star, as the edge vectors and their cotangent weights. */
interface Star {
	readonly neighbours: number[];
	readonly weights: number[];
	/** The rest-pose edge vectors, three per neighbour. */
	readonly rest: Float64Array;
	readonly normal: [number, number, number];
	readonly area: number;
}

function cubicStylize(cm: CMeshO, cubeness: number, iterations: number, cb: CallBackPos): number {
	const stars = buildStars(cm);
	// Only the vertices that actually have a star: an unreferenced one has no
	// edges to be rigid about, and would make the Laplacian singular.
	const live: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v) && stars[v] !== undefined) live.push(v);
	if (live.length === 0) return 0;

	// The global step's matrix — the cotangent Laplacian — never changes, so
	// it is assembled once and only the right-hand side is rebuilt.
	const laplacian = new SparseMatrix(cm.vertSize);
	for (const v of live) {
		const star = stars[v] as Star;
		for (let i = 0; i < star.neighbours.length; i++) {
			const n = star.neighbours[i];
			const w = star.weights[i];
			laplacian.add(v, v, w);
			laplacian.add(v, n, -w);
		}
	}
	// One vertex is pinned, or the system is singular: the energy is
	// invariant to translating the whole mesh.
	const anchor = live[0];
	const anchorPosition = [cm.vx(anchor), cm.vy(anchor), cm.vz(anchor)];

	const rotations = new Float64Array(cm.vertSize * 9);
	for (const v of live) {
		rotations[9 * v] = 1;
		rotations[9 * v + 4] = 1;
		rotations[9 * v + 8] = 1;
	}
	// The ADMM state, carried between iterations so it warms up rather than
	// restarting from nothing each round.
	const z = new Float64Array(cm.vertSize * 3);
	const u = new Float64Array(cm.vertSize * 3);
	const rho = new Float64Array(cm.vertSize).fill(1e-4);

	const before = Float64Array.from(cm.vertCoord);
	for (let round = 0; round < iterations; round++) {
		cb((100 * round) / iterations, "Cubic stylization");
		for (const v of live) localStep(cm, stars[v] as Star, v, cubeness, rotations, z, u, rho);
		globalStep(cm, stars, live, rotations, laplacian, anchor, anchorPosition);
	}
	void rho;

	let sum = 0;
	for (const v of live) {
		sum += Math.hypot(
			cm.vx(v) - before[3 * v],
			cm.vy(v) - before[3 * v + 1],
			cm.vz(v) - before[3 * v + 2],
		);
	}
	return sum / live.length;
}

/**
 * The local step: the rotation for one star, with the L1 term on its normal.
 *
 * Plain ARAP would take the rotation straight from the SVD of the covariance
 * between rest and current edges. The cubeness term adds `λ·a·|R·n|₁`, which
 * ADMM splits into a rotation fit and a soft-threshold — the shrinkage is
 * what actually zeroes a normal's components and so snaps it to an axis.
 */
function localStep(
	cm: CMeshO,
	star: Star,
	v: number,
	cubeness: number,
	rotations: Float64Array,
	z: Float64Array,
	u: Float64Array,
	rho: Float64Array,
): void {
	// The covariance between the rest edges and the current ones.
	const base = new Float64Array(9);
	for (let i = 0; i < star.neighbours.length; i++) {
		const n = star.neighbours[i];
		const w = star.weights[i];
		const e = [cm.vx(n) - cm.vx(v), cm.vy(n) - cm.vy(v), cm.vz(n) - cm.vz(v)];
		for (let a = 0; a < 3; a++) {
			for (let b = 0; b < 3; b++) base[3 * a + b] += w * star.rest[3 * i + a] * e[b];
		}
	}

	const lambda = cubeness * star.area;
	if (lambda === 0) {
		writeRotation(rotations, v, bestRotation(base));
		return;
	}

	const penalty = rho[v];
	for (let step = 0; step < 4; step++) {
		// Add the ADMM term: the normal has to explain (z - u) as well.
		const m = Float64Array.from(base);
		for (let a = 0; a < 3; a++) {
			for (let b = 0; b < 3; b++) {
				m[3 * a + b] += penalty * star.normal[a] * (z[3 * v + b] - u[3 * v + b]);
			}
		}
		const r = bestRotation(m);
		writeRotation(rotations, v, r);

		// Shrink: the proximal operator of the ℓ¹ norm, which is what sets a
		// component to exactly zero rather than merely reducing it.
		const rn = [
			r[0] * star.normal[0] + r[3] * star.normal[1] + r[6] * star.normal[2],
			r[1] * star.normal[0] + r[4] * star.normal[1] + r[7] * star.normal[2],
			r[2] * star.normal[0] + r[5] * star.normal[1] + r[8] * star.normal[2],
		];
		for (let a = 0; a < 3; a++) {
			const target = rn[a] + u[3 * v + a];
			const cut = lambda / penalty;
			z[3 * v + a] = Math.sign(target) * Math.max(0, Math.abs(target) - cut);
			u[3 * v + a] = target - z[3 * v + a];
		}
	}
}

/** The global step: move the vertices to fit the rotations just found. */
function globalStep(
	cm: CMeshO,
	stars: ReadonlyArray<Star | undefined>,
	live: readonly number[],
	rotations: Float64Array,
	laplacian: SparseMatrix,
	anchor: number,
	anchorPosition: readonly number[],
): void {
	for (let axis = 0; axis < 3; axis++) {
		const rhs = new Float64Array(cm.vertSize);
		for (const v of live) {
			const star = stars[v] as Star;
			let sum = 0;
			for (let i = 0; i < star.neighbours.length; i++) {
				const n = star.neighbours[i];
				const w = star.weights[i];
				const rest = [star.rest[3 * i], star.rest[3 * i + 1], star.rest[3 * i + 2]];
				// Half the sum of the two rotations, which is what makes the
				// system symmetric and the solve well posed.
				//
				// Negated: `rest` runs from v to n, while the Laplacian's row
				// is written in terms of (p_v - p_n). With the sign the other
				// way the solve converges to the mesh reflected through the
				// anchor, which on a unit sphere shows up as every vertex
				// moving by two diameters.
				const a = apply(rotations, v, rest);
				const b = apply(rotations, n, rest);
				sum -= (w / 2) * (a[axis] + b[axis]);
			}
			rhs[v] = sum;
		}
		// The matrix is reused across the three axes and the pinning has to be
		// undone each time, so it is copied rather than mutated in place.
		const system = copyWithPin(laplacian, cm.vertSize, anchor, anchorPosition[axis], rhs);
		const solved = solveCG(system, rhs, { tolerance: 1e-12, iterations: 4 * cm.vertSize });
		for (const v of live) {
			const p = [cm.vx(v), cm.vy(v), cm.vz(v)];
			p[axis] = solved.x[v];
			cm.setVert(v, p[0], p[1], p[2]);
		}
	}
}

function copyWithPin(
	source: SparseMatrix,
	size: number,
	anchor: number,
	value: number,
	rhs: Float64Array,
): SparseMatrix {
	const out = new SparseMatrix(size);
	for (let r = 0; r < size; r++) {
		for (let c = 0; c < size; c++) {
			const v = source.get(r, c);
			if (v !== 0) out.add(r, c, v);
		}
	}
	out.pin(anchor, value, rhs);
	return out;
}

function apply(rotations: Float64Array, v: number, e: readonly number[]): number[] {
	const r = rotations.subarray(9 * v, 9 * v + 9);
	return [
		r[0] * e[0] + r[1] * e[1] + r[2] * e[2],
		r[3] * e[0] + r[4] * e[1] + r[5] * e[2],
		r[6] * e[0] + r[7] * e[1] + r[8] * e[2],
	];
}

function writeRotation(rotations: Float64Array, v: number, r: Float64Array): void {
	rotations.set(r, 9 * v);
}

/**
 * The rotation nearest a 3x3 matrix, by polar decomposition.
 *
 * Iterating `R ← (R + R⁻ᵀ)/2` converges to the orthogonal factor and needs no
 * SVD; a handful of steps is exact to floating point for the well-conditioned
 * matrices a mesh produces. The determinant is checked at the end because the
 * nearest *orthogonal* matrix can be a reflection, and a reflected star turns
 * the surface inside out.
 */
function bestRotation(m: Float64Array): Float64Array {
	let r = Float64Array.from(m);
	for (let i = 0; i < 24; i++) {
		const inverseTranspose = invertTranspose(r);
		if (inverseTranspose === null) break;
		let change = 0;
		for (let k = 0; k < 9; k++) {
			const next = 0.5 * (r[k] + inverseTranspose[k]);
			change = Math.max(change, Math.abs(next - r[k]));
			r[k] = next;
		}
		if (change < 1e-14) break;
	}
	if (determinant(r) < 0) {
		// Flip the column with the least influence, which is the standard
		// repair and the one that moves the star least.
		for (let k = 2; k < 9; k += 3) r[k] = -r[k];
	}
	if (!Number.isFinite(r[0])) {
		r = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
	}
	return r;
}

function invertTranspose(m: Float64Array): Float64Array | null {
	const det = determinant(m);
	if (Math.abs(det) < 1e-18) return null;
	const c = new Float64Array(9);
	c[0] = m[4] * m[8] - m[5] * m[7];
	c[1] = m[5] * m[6] - m[3] * m[8];
	c[2] = m[3] * m[7] - m[4] * m[6];
	c[3] = m[2] * m[7] - m[1] * m[8];
	c[4] = m[0] * m[8] - m[2] * m[6];
	c[5] = m[1] * m[6] - m[0] * m[7];
	c[6] = m[1] * m[5] - m[2] * m[4];
	c[7] = m[2] * m[3] - m[0] * m[5];
	c[8] = m[0] * m[4] - m[1] * m[3];
	// The cofactor matrix over the determinant *is* the inverse transpose.
	for (let k = 0; k < 9; k++) c[k] /= det;
	return c;
}

function determinant(m: Float64Array): number {
	return (
		m[0] * (m[4] * m[8] - m[5] * m[7]) -
		m[1] * (m[3] * m[8] - m[5] * m[6]) +
		m[2] * (m[3] * m[7] - m[4] * m[6])
	);
}

function buildStars(cm: CMeshO): Array<Star | undefined> {
	const neighbours: Array<Map<number, number>> = Array.from(
		{ length: cm.vertSize },
		() => new Map<number, number>(),
	);
	const areas = new Float64Array(cm.vertSize);

	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const p = [0, 1, 2].map((k) => {
			const v = cm.fv(f, k);
			return [cm.vx(v), cm.vy(v), cm.vz(v)];
		});
		const area = triangleArea(p);
		for (let k = 0; k < 3; k++) areas[cm.fv(f, k)] += area / 3;
		for (let k = 0; k < 3; k++) {
			const a = cm.fv(f, (k + 1) % 3);
			const b = cm.fv(f, (k + 2) % 3);
			const w = 0.5 * cotangent(sub(p[(k + 1) % 3], p[k]), sub(p[(k + 2) % 3], p[k]));
			neighbours[a].set(b, (neighbours[a].get(b) ?? 0) + w);
			neighbours[b].set(a, (neighbours[b].get(a) ?? 0) + w);
		}
	}

	const stars: Array<Star | undefined> = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v) || neighbours[v].size === 0) {
			stars.push(undefined);
			continue;
		}
		const ns = [...neighbours[v].keys()];
		const rest = new Float64Array(ns.length * 3);
		ns.forEach((n, i) => {
			rest[3 * i] = cm.vx(n) - cm.vx(v);
			rest[3 * i + 1] = cm.vy(n) - cm.vy(v);
			rest[3 * i + 2] = cm.vz(n) - cm.vz(v);
		});
		stars.push({
			neighbours: ns,
			// A negative cotangent weight would make the Laplacian indefinite
			// and the solve meaningless; clamping keeps it usable on the
			// obtuse triangles every real mesh has.
			weights: ns.map((n) => Math.max(1e-8, neighbours[v].get(n) ?? 0)),
			rest,
			normal: [cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]],
			area: areas[v],
		});
	}
	return stars;
}

function sub(a: readonly number[], b: readonly number[]): number[] {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function triangleArea(p: readonly number[][]): number {
	const u = sub(p[1], p[0]);
	const w = sub(p[2], p[0]);
	return (
		Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2
	);
}

function cotangent(u: readonly number[], v: readonly number[]): number {
	const c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
	const length = Math.hypot(c[0], c[1], c[2]);
	if (length === 0) return 0;
	return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / length;
}
