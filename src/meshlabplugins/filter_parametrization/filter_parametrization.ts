/**
 * `filter_parametrization` — two ways of flattening a disk.
 *
 * Harmonic pins the boundary to a convex shape and solves for the interior;
 * it always produces a valid layout but the boundary's shape is imposed
 * rather than discovered. Least-squares conformal maps pins only two vertices
 * and lets the boundary find its own shape by minimising angle distortion —
 * which is usually much better, and which no longer guarantees an unfolded
 * result, so the filter reports how many faces came out folded.
 *
 * The harmonic machinery already lives in `vcg/complex/parametrization`;
 * LSCM is here because nothing else needs it.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { RichEnum } from "../../common/parameters/rich_parameter.ts";
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
import { foldedNum, meshAngleDistortion } from "../../vcg/complex/parametrization/distortion.ts";
import {
	boundaryLoop,
	parametrizeDisk,
	writeWedgeUV,
} from "../../vcg/complex/parametrization/harmonic.ts";
import { SparseMatrix, solveCG } from "../../vcg/math/sparse.ts";

export const FP = { FP_HARMONIC_PARAM: 0, FP_LEAST_SQUARES_PARAM: 1 } as const;

export class FilterParametrization extends FilterPlugin {
	pluginName(): string {
		return "FilterParametrization";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	filterName(id: ActionIDType): string {
		if (id === FP.FP_HARMONIC_PARAM) return "Harmonic Parametrization";
		if (id === FP.FP_LEAST_SQUARES_PARAM) return "Least Squares Conformal Maps Parametrization";
		return this.wrongActionCalled(id);
	}
	pythonFilterName(id: ActionIDType): string {
		if (id === FP.FP_HARMONIC_PARAM) return "compute_texcoord_parametrization_harmonic";
		if (id === FP.FP_LEAST_SQUARES_PARAM) {
			return "compute_texcoord_parametrization_least_squares_conformal_maps";
		}
		return this.wrongActionCalled(id);
	}
	filterInfo(id: ActionIDType): string {
		if (id === FP.FP_HARMONIC_PARAM) {
			return (
				"Computes a single patch, fixed boundary harmonic parametrization of a mesh. The filter " +
				"requires that the input mesh has a single fixed boundary."
			);
		}
		if (id === FP.FP_LEAST_SQUARES_PARAM) {
			return "Computes a least squares conformal maps parametrization of a mesh.";
		}
		return this.wrongActionCalled(id);
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Texture;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_WEDGTEXCOORD;
	}

	override initParameterList(id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		if (id === FP.FP_HARMONIC_PARAM) {
			list.add(
				new RichEnum("harm_function", 1, ["Uniform", "Mean value", "Cotangent"], {
					description: "Harmonic function",
					tooltip:
						"The weights used to place each interior vertex. Mean value weights can never " +
						"fold a triangle; cotangent weights preserve angles better but lose that guarantee " +
						"on obtuse triangles.",
				}),
			);
		}
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		_cb: CallBackPos,
	): FilterOutput {
		const m = doc.mm();
		const cm = m.cm;
		m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
		if (cm.fn === 0) throw new MLException("the mesh has no faces to parametrise");

		if (id === FP.FP_HARMONIC_PARAM) {
			const weights = (["uniform", "mean-value", "harmonic"] as const)[
				params.getEnum("harm_function")
			];
			const result = parametrizeDisk(cm, { weights, boundary: "circle" });
			writeWedgeUV(cm, result.uv);
			post.mask = MeshElement.MM_NONE;
			const folded = foldedNum(cm);
			doc.Log.log(
				`Harmonic parametrization with ${weights} weights in ${result.iterations} iterations` +
					(folded > 0 ? `; ${folded} faces came out folded` : ""),
			);
			return {
				iterations: result.iterations,
				folded_faces: folded,
				angle_distortion: meshAngleDistortion(cm),
			};
		}
		if (id !== FP.FP_LEAST_SQUARES_PARAM) return this.wrongActionCalled(id);

		const uv = leastSquaresConformal(cm);
		writeWedgeUV(cm, uv);
		post.mask = MeshElement.MM_NONE;
		const folded = foldedNum(cm);
		const distortion = meshAngleDistortion(cm);
		doc.Log.log(
			`LSCM parametrization, angle distortion ${distortion.toFixed(5)}` +
				(folded > 0 ? `; ${folded} faces came out folded` : ""),
		);
		return { folded_faces: folded, angle_distortion: distortion };
	}
}

/**
 * Least-squares conformal maps (Lévy, Petitjean, Ray and Maillot, 2002).
 *
 * Each triangle is laid out flat in its own local frame, which gives three
 * complex numbers `W₁, W₂, W₃`. The map is conformal on that triangle exactly
 * when `Σ Wⱼ zⱼ = 0` for the unknown planar positions `zⱼ = uⱼ + i vⱼ`. That
 * is one complex — two real — equation per triangle, and with far more
 * triangles than vertices the system is over-determined, so it is solved in
 * the least-squares sense: minimise the total conformal error.
 *
 * Two vertices have to be pinned or the whole thing collapses to a point; the
 * two ends of the longest boundary chord are chosen, which is what spreads
 * the result out rather than pinning two neighbours and getting a sliver. The
 * result is then normalised into 0..1, since a least-squares solution has no
 * particular scale.
 */
function leastSquaresConformal(cm: CMeshO): Float64Array {
	const loop = boundaryLoop(cm);
	const [pinA, pinB] = farthestPair(cm, loop);

	// Unknowns are laid out as [u₀…u_{n-1}, v₀…v_{n-1}].
	const n = cm.vertSize;
	const size = 2 * n;
	const normal = new SparseMatrix(size);
	const rhs = new Float64Array(size);

	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const local = flattenTriangle(cm, f);
		if (local === null) continue;
		const area2 = Math.abs(
			(local[1][0] - local[0][0]) * (local[2][1] - local[0][1]) -
				(local[2][0] - local[0][0]) * (local[1][1] - local[0][1]),
		);
		if (area2 === 0) continue;
		const scale = 1 / Math.sqrt(area2);

		// Wⱼ is the edge opposite vertex j, as a complex number.
		const w: Array<[number, number]> = [];
		for (let j = 0; j < 3; j++) {
			const a = local[(j + 2) % 3];
			const b = local[(j + 1) % 3];
			w.push([(a[0] - b[0]) * scale, (a[1] - b[1]) * scale]);
		}
		const vertex = [0, 1, 2].map((j) => cm.fv(f, j));

		// Real part: Σ (Wr·u - Wi·v) = 0. Imaginary: Σ (Wi·u + Wr·v) = 0.
		// Each becomes one row of A; the normal equations accumulate AᵀA.
		for (const row of [
			vertex.flatMap((v, j) => [
				{ col: v, coeff: w[j][0] },
				{ col: v + n, coeff: -w[j][1] },
			]),
			vertex.flatMap((v, j) => [
				{ col: v, coeff: w[j][1] },
				{ col: v + n, coeff: w[j][0] },
			]),
		]) {
			for (const p of row) {
				for (const q of row) normal.add(p.col, q.col, p.coeff * q.coeff);
			}
		}
	}

	// Pin two vertices at opposite corners of the unit square. Without them
	// the system is singular: any translation, rotation and scale of a
	// conformal map is still conformal.
	normal.pin(pinA, 0, rhs);
	normal.pin(pinA + n, 0, rhs);
	normal.pin(pinB, 1, rhs);
	normal.pin(pinB + n, 1, rhs);

	const solved = solveCG(normal, rhs, { tolerance: 1e-12, iterations: 20 * size });
	if (!solved.converged) {
		throw new MLException(
			`the LSCM system did not converge (residual ${solved.residual.toExponential(2)}); the mesh ` +
				"probably has degenerate triangles",
		);
	}

	// Normalise into 0..1 — a least-squares conformal map has no scale of
	// its own, and the two pinned vertices only fix one chord of it.
	const uv = new Float64Array(2 * n);
	let minU = Number.POSITIVE_INFINITY;
	let maxU = Number.NEGATIVE_INFINITY;
	let minV = Number.POSITIVE_INFINITY;
	let maxV = Number.NEGATIVE_INFINITY;
	for (let v = 0; v < n; v++) {
		if (cm.isVertD(v)) continue;
		minU = Math.min(minU, solved.x[v]);
		maxU = Math.max(maxU, solved.x[v]);
		minV = Math.min(minV, solved.x[v + n]);
		maxV = Math.max(maxV, solved.x[v + n]);
	}
	const spanU = maxU - minU || 1;
	const spanV = maxV - minV || 1;
	// One scale for both axes, so the map is not stretched after the fact —
	// which would undo exactly the conformality it was solved for.
	const span = Math.max(spanU, spanV);
	for (let v = 0; v < n; v++) {
		if (cm.isVertD(v)) continue;
		uv[2 * v] = (solved.x[v] - minU) / span;
		uv[2 * v + 1] = (solved.x[v + n] - minV) / span;
	}
	return uv;
}

/** A triangle laid flat in its own plane, as three 2D points. */
function flattenTriangle(cm: CMeshO, f: number): Array<[number, number]> | null {
	const p = [0, 1, 2].map((k) => {
		const v = cm.fv(f, k);
		return [cm.vx(v), cm.vy(v), cm.vz(v)];
	});
	const e1 = sub(p[1], p[0]);
	const length = Math.hypot(e1[0], e1[1], e1[2]);
	if (length === 0) return null;
	const x = e1.map((c) => c / length);
	const e2 = sub(p[2], p[0]);
	const normal = cross(e1, e2);
	const nl = Math.hypot(normal[0], normal[1], normal[2]);
	if (nl === 0) return null;
	const y = cross(
		normal.map((c) => c / nl),
		x,
	);
	return [
		[0, 0],
		[length, 0],
		[dot(e2, x), dot(e2, y)],
	];
}

/** The two boundary vertices farthest apart, as the pins. */
function farthestPair(cm: CMeshO, loop: readonly number[]): [number, number] {
	let best: [number, number] = [loop[0], loop[Math.floor(loop.length / 2)]];
	let bestDistance = -1;
	// The full pairwise scan is quadratic in the boundary length, which is
	// small; on a very long boundary it is still the cheapest part of LSCM.
	for (let i = 0; i < loop.length; i++) {
		for (let j = i + 1; j < loop.length; j++) {
			const a = loop[i];
			const b = loop[j];
			const d =
				(cm.vx(a) - cm.vx(b)) ** 2 + (cm.vy(a) - cm.vy(b)) ** 2 + (cm.vz(a) - cm.vz(b)) ** 2;
			if (d > bestDistance) {
				bestDistance = d;
				best = [a, b];
			}
		}
	}
	return best;
}

function sub(a: readonly number[], b: readonly number[]): number[] {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: readonly number[], b: readonly number[]): number[] {
	return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
