/**
 * `filter_ao` — ambient occlusion by ray casting.
 *
 * How much of the sky each vertex can see, written into the quality channel
 * and shown as grey. Upstream renders depth buffers on the GPU; there is no
 * GPU here, so this traces rays against a BVH. The result is the same
 * quantity and is if anything more accurate — a depth buffer quantises the
 * directions to whatever resolution it was rendered at — but it is slower,
 * which is why the ray count is a parameter.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import {
	RichBool,
	RichDirection,
	RichEnum,
	RichFloat,
	RichInt,
} from "../../common/parameters/rich_parameter.ts";
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
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { mulberry32 } from "../../vcg/math/noise.ts";
import { rgba } from "../../vcg/space/color4.ts";
import { BVH, coneDirections, cosineHemisphere } from "../../vcg/space/index/bvh.ts";

export const FP = { FP_VERT_AMBIENT_OCCLUSION: 0 } as const;

/** Where the light is assumed to come from. */
const MODE = { UNIFORM: 0, CONE: 1 } as const;

export class FilterAmbientOcclusion extends FilterPlugin {
	pluginName(): string {
		return "FilterAmbientOcclusion";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.FP_VERT_AMBIENT_OCCLUSION) this.wrongActionCalled(id);
		return "Ambient Occlusion";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.FP_VERT_AMBIENT_OCCLUSION) this.wrongActionCalled(id);
		return "compute_scalar_ambient_occlusion_gpu";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.FP_VERT_AMBIENT_OCCLUSION) this.wrongActionCalled(id);
		return (
			"Compute ambient occlusions values; it takes a number of well distributed view direction " +
			"and for point of the surface it computes how many time it is visible from these " +
			"directions. This value is saved into quality and automatically mapped into a gray shade."
		);
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.VertexColoring;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY;
	}

	override initParameterList(_id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichEnum("occMode", MODE.UNIFORM, ["per-Vertex", "per-Face (deprecated)"], {
				description: "Occlusion mode",
				tooltip: "Whether the occlusion is evaluated at the vertices or at the face centres.",
			}),
		);
		list.add(
			new RichFloat("dirBias", 0, {
				description: "Directional Bias",
				tooltip:
					"The balance between a uniform sky and one concentrated in the cone below: 0 is fully " +
					"uniform, 1 fully directional.",
			}),
		);
		list.add(
			new RichDirection("coneDir", [0, 1, 0], {
				description: "Cone axis",
				tooltip: "The direction the light comes from, when the bias is above zero.",
			}),
		);
		list.add(
			new RichFloat("coneAngle", 30, {
				description: "Cone angle",
				tooltip: "The half-angle of the light cone, in degrees.",
			}),
		);
		list.add(
			new RichInt("numberRays", 128, {
				description: "Number of rays",
				tooltip:
					"How many rays are cast per vertex. The estimate's noise falls with the square root " +
					"of this, so quadrupling it halves the noise.",
			}),
		);
		list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
		list.add(
			new RichBool("useGPU", false, {
				description: "Use GPU acceleration",
				tooltip: "Not available: this implementation traces rays on the CPU.",
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
		if (id !== FP.FP_VERT_AMBIENT_OCCLUSION) return this.wrongActionCalled(id);
		if (params.getBool("useGPU")) {
			throw new MLException(
				"there is no GPU path here; leave 'Use GPU acceleration' off and the rays are traced on " +
					"the CPU instead",
			);
		}
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces to occlude anything");
		const rays = params.getInt("numberRays");
		if (rays < 1) throw new MLException(`the ray count must be at least 1, got ${rays}`);

		const perFace = params.getEnum("occMode") === 1;
		const bias = Math.max(0, Math.min(1, params.getFloat("dirBias")));
		const coneAxis = normalise(params.getPoint3m("coneDir"));
		const coneAngle = (params.getFloat("coneAngle") * Math.PI) / 180;
		const random = mulberry32(params.getInt("randomSeed") >>> 0 || 1);

		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTCOLOR);
		UpdateBounding.box(cm);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const bvh = new BVH(cm);
		// Offset each ray's origin off the surface, or it hits the face it
		// started on. A fraction of the bounding diagonal is scale-free.
		const epsilon = (cm.bbox.diagonal || 1) * 1e-5;
		const reach = (cm.bbox.diagonal || 1) * 2;

		const targets: Array<{ point: number[]; normal: number[]; index: number }> = [];
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
				targets.push({
					point: p,
					normal: [cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]],
					index: f,
				});
			}
		} else {
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				targets.push({
					point: [cm.vx(v), cm.vy(v), cm.vz(v)],
					normal: [cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]],
					index: v,
				});
			}
		}

		const openness = new Float64Array(targets.length);
		targets.forEach((target, i) => {
			if (i % 256 === 0) cb((100 * i) / targets.length, "Casting occlusion rays");
			const coneCount = Math.round(rays * bias);
			const directions = [
				...cosineHemisphere(target.normal, rays - coneCount, random),
				...coneDirections(coneAxis, coneAngle, coneCount, random),
			];
			let open = 0;
			let used = 0;
			const origin = [
				target.point[0] + target.normal[0] * epsilon,
				target.point[1] + target.normal[1] * epsilon,
				target.point[2] + target.normal[2] * epsilon,
			];
			for (const d of directions) {
				// A cone direction can point into the surface; it contributes
				// nothing and must not be counted, or a downward-facing vertex
				// would read as brighter than it is.
				const facing = d[0] * target.normal[0] + d[1] * target.normal[1] + d[2] * target.normal[2];
				if (facing <= 0) continue;
				used++;
				if (!bvh.occluded(origin, d, epsilon, reach)) open++;
			}
			openness[i] = used === 0 ? 0 : open / used;
		});

		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (const o of openness) {
			min = Math.min(min, o);
			max = Math.max(max, o);
		}
		if (perFace) m.updateDataMask(MeshElement.MM_FACEQUALITY | MeshElement.MM_FACECOLOR);

		targets.forEach((target, i) => {
			const value = openness[i];
			const grey = Math.max(0, Math.min(255, Math.round(value * 255)));
			if (perFace) {
				const q = cm.faceQuality;
				if (q !== null) q[target.index] = value;
				const c = cm.faceColor;
				if (c !== null) c[target.index] = rgba(grey, grey, grey);
			} else {
				cm.vertQuality[target.index] = value;
				cm.vertColor[target.index] = rgba(grey, grey, grey);
			}
		});

		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`Ambient occlusion over ${targets.length} ${perFace ? "faces" : "vertices"} with ${rays} ` +
				`rays each: openness ${min.toFixed(3)} to ${max.toFixed(3)}`,
		);
		return { min_openness: min, max_openness: max, rays };
	}
}

function normalise(v: readonly number[]): number[] {
	const length = Math.hypot(v[0], v[1], v[2]);
	return length === 0 ? [0, 1, 0] : [v[0] / length, v[1] / length, v[2] / length];
}
