/**
 * `filter_sdfgpu` — three quantities measured by casting rays *into* the mesh.
 *
 * They share a single question: send a ray inwards from a point on the
 * surface, and what happens?
 *
 * - **Shape diameter function**: how far until it comes out the other side.
 *   Averaged over a cone, that is the local thickness of the object, which is
 *   what separates a limb from a torso independently of how either is posed.
 * - **Depth complexity**: how many surfaces a ray crosses. A watertight
 *   two-manifold solid gives an even number everywhere; an odd one is proof
 *   of a hole.
 * - **Volumetric obscurance**: what fraction of a ball around the point lies
 *   inside the object — ambient occlusion's near-field cousin, and the one
 *   that picks out creases rather than large-scale cavities.
 *
 * Upstream renders depth-peeled buffers on the GPU. This traces against a
 * BVH, which gives the same quantities without the peeling iteration count
 * and its tolerance — parameters that exist to work around the rasteriser.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { RichBool, RichEnum, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import { mulberry32 } from "../../vcg/math/noise.ts";
import { colorRamp } from "../../vcg/space/color4.ts";
import { BVH, coneDirections } from "../../vcg/space/index/bvh.ts";

export const FP = {
	FP_SDF_OBSCURANCE: 0,
	FP_SDF_SDF: 1,
	FP_SDF_DEPTH_COMPLEXITY: 2,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_SDF_OBSCURANCE]: {
		name: "Volumetric obscurance",
		pythonName: "compute_scalar_by_volumetric_obscurance_gpu",
		info: "Compute a volumetric obscurance value for each vertex, a measure of how enclosed it is.",
	},
	[FP.FP_SDF_SDF]: {
		name: "Shape Diameter Function",
		pythonName: "compute_scalar_by_shape_diameter_function_per_vertex_gpu",
		info:
			"Compute the shape diameter function, the local thickness of the object as seen from each " +
			"point of its surface.",
	},
	[FP.FP_SDF_DEPTH_COMPLEXITY]: {
		name: "Depth complexity",
		pythonName: "get_depth_complexity",
		info: "Compute how many surfaces a ray crosses, which for a solid must always be even.",
	},
};

export class FilterSDF extends FilterPlugin {
	pluginName(): string {
		return "FilterSDFGPU";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private spec(id: ActionIDType): FilterSpec {
		const s = SPECS[id];
		if (s === undefined) this.wrongActionCalled(id);
		return s;
	}

	filterName(id: ActionIDType): string {
		return this.spec(id).name;
	}
	pythonFilterName(id: ActionIDType): string {
		return this.spec(id).pythonName;
	}
	filterInfo(id: ActionIDType): string {
		return this.spec(id).info;
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Generic;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY;
	}

	override initParameterList(id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichEnum("onPrimitive", 0, ["On vertices", "On faces"], {
				description: "Metric applied on",
				tooltip: "Whether the value is computed at the vertices or at the face centres.",
			}),
		);
		list.add(
			new RichInt("numberRays", 128, {
				description: "Number of rays",
				tooltip: "How many rays are cast per element.",
			}),
		);
		list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
		if (id === FP.FP_SDF_SDF) {
			list.add(
				new RichFloat("coneAngle", 120, {
					description: "Cone amplitude",
					tooltip:
						"The full angle of the cone the rays are spread over, in degrees. A narrow cone " +
						"measures across the object, a wide one averages over the neighbourhood.",
				}),
			);
			list.add(
				new RichBool("removeFalse", true, {
					description: "Remove false intersections",
					tooltip:
						"Discard a ray whose exit face points the same way as the entry face; such a ray " +
						"grazed the surface rather than crossing the object.",
				}),
			);
			list.add(
				new RichBool("removeOutliers", false, {
					description: "Remove outliers",
					tooltip: "Discard the rays whose length is more than one standard deviation off.",
				}),
			);
		}
		if (id === FP.FP_SDF_OBSCURANCE) {
			list.add(
				new RichFloat("obscuranceExponent", 0.1, {
					description: "Obscurance exponent",
					tooltip:
						"How fast the obscurance falls off with distance. A larger value keeps only the " +
						"nearby geometry, which picks out creases rather than cavities.",
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
		cb: CallBackPos,
	): FilterOutput {
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces");
		const rays = params.getInt("numberRays");
		if (rays < 1) throw new MLException(`the ray count must be at least 1, got ${rays}`);

		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTCOLOR);
		UpdateBounding.box(cm);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const bvh = new BVH(cm);
		const diagonal = cm.bbox.diagonal || 1;
		const epsilon = diagonal * 1e-5;
		const random = mulberry32(params.getInt("randomSeed") >>> 0 || 1);
		const perFace = params.getEnum("onPrimitive") === 1;
		const targets = elementsOf(cm, perFace);

		const values = new Float64Array(targets.length);
		targets.forEach((target, i) => {
			if (i % 128 === 0) cb((100 * i) / targets.length, "Casting rays");
			// Inwards: the negated normal, so the cone points into the object.
			const inward = target.normal.map((c) => -c);
			const origin = [
				target.point[0] - target.normal[0] * epsilon,
				target.point[1] - target.normal[1] * epsilon,
				target.point[2] - target.normal[2] * epsilon,
			];

			if (id === FP.FP_SDF_DEPTH_COMPLEXITY) {
				let worst = 0;
				for (const d of coneDirections(inward, Math.PI / 2, rays, random)) {
					worst = Math.max(worst, bvh.intersectAll(origin, d, epsilon, diagonal * 4).length);
				}
				values[i] = worst;
				return;
			}

			if (id === FP.FP_SDF_OBSCURANCE) {
				const exponent = params.getFloat("obscuranceExponent");
				let sum = 0;
				let used = 0;
				for (const d of coneDirections(inward, Math.PI / 2, rays, random)) {
					used++;
					const hit = bvh.intersect(origin, d, epsilon, diagonal);
					// A ray that leaves without hitting anything sees open
					// space; one that hits close by is buried in a crease.
					sum += hit === null ? 1 : 1 - Math.exp(-exponent * (hit.t / diagonal) * 100);
				}
				values[i] = used === 0 ? 1 : sum / used;
				return;
			}

			const halfAngle = (params.getFloat("coneAngle") * Math.PI) / 180 / 2;
			const removeFalse = params.getBool("removeFalse");
			const lengths: number[] = [];
			const weights: number[] = [];
			for (const d of coneDirections(inward, halfAngle, rays, random)) {
				const hit = bvh.intersect(origin, d, epsilon, diagonal * 4);
				if (hit === null) continue;
				if (removeFalse && !hit.backface) continue;
				// Weighted by how close the ray is to the cone's axis, which is
				// what makes the SDF stable as the cone angle is widened.
				const weight = d[0] * inward[0] + d[1] * inward[1] + d[2] * inward[2];
				lengths.push(hit.t);
				weights.push(Math.max(0, weight));
			}
			values[i] = weightedMean(lengths, weights, params.getBool("removeOutliers"));
		});

		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (const v of values) {
			min = Math.min(min, v);
			max = Math.max(max, v);
		}
		if (perFace) m.updateDataMask(MeshElement.MM_FACEQUALITY | MeshElement.MM_FACECOLOR);

		targets.forEach((target, i) => {
			const colour = colorRamp(min, max === min ? min + 1 : max, values[i]);
			if (perFace) {
				const q = cm.faceQuality;
				if (q !== null) q[target.index] = values[i];
				const c = cm.faceColor;
				if (c !== null) c[target.index] = colour;
			} else {
				cm.vertQuality[target.index] = values[i];
				cm.vertColor[target.index] = colour;
			}
		});

		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`${this.spec(id).name} over ${targets.length} elements: ${min.toFixed(4)} to ${max.toFixed(4)}`,
		);
		return { min: min, max: max, elements: targets.length };
	}
}

interface Element {
	readonly point: number[];
	readonly normal: number[];
	readonly index: number;
}

function elementsOf(cm: CMeshO, perFace: boolean): Element[] {
	const out: Element[] = [];
	if (perFace) {
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			const p = [0, 0, 0];
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				p[0] += cm.vx(v) / 3;
				p[1] += cm.vy(v) / 3;
				p[2] += cm.vz(v) / 3;
			}
			out.push({
				point: p,
				normal: [cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]],
				index: f,
			});
		}
		return out;
	}
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		out.push({
			point: [cm.vx(v), cm.vy(v), cm.vz(v)],
			normal: [cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]],
			index: v,
		});
	}
	return out;
}

/** The weighted mean, optionally after dropping the far outliers. */
function weightedMean(
	values: readonly number[],
	weights: readonly number[],
	trim: boolean,
): number {
	if (values.length === 0) return 0;
	let sum = 0;
	let total = 0;
	for (let i = 0; i < values.length; i++) {
		sum += values[i] * weights[i];
		total += weights[i];
	}
	if (total === 0) return 0;
	const mean = sum / total;
	if (!trim) return mean;

	let variance = 0;
	for (let i = 0; i < values.length; i++) variance += weights[i] * (values[i] - mean) ** 2;
	const deviation = Math.sqrt(variance / total);
	// One standard deviation is upstream's cut. A ray that escaped through a
	// hole reads as enormously long, and one such ray would otherwise dominate.
	let trimmedSum = 0;
	let trimmedTotal = 0;
	for (let i = 0; i < values.length; i++) {
		if (Math.abs(values[i] - mean) > deviation) continue;
		trimmedSum += values[i] * weights[i];
		trimmedTotal += weights[i];
	}
	return trimmedTotal === 0 ? mean : trimmedSum / trimmedTotal;
}
