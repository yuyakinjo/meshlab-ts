/**
 * `filter_embree` — the ray-traced measures, again.
 *
 * Three of these five compute the same quantities as `filter_ao` and
 * `filter_sdfgpu`; upstream ships them twice because Embree replaces the
 * OpenGL path with a CPU ray tracer, and the two implementations reach the
 * same numbers by different routes. Here there was never an OpenGL path, so
 * they are genuinely the same computation exposed under the names a script
 * written against Embree would use — and the *tests* pin exactly that: the
 * Embree and GPU spellings must agree.
 *
 * The other two are new, and both are questions only a ray tracer can answer:
 * which way round a face really is, and whether it can be seen at all.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import {
	RichBool,
	RichDirection,
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { FaceFlag } from "../../vcg/complex/flags.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { mulberry32 } from "../../vcg/math/noise.ts";
import { colorRamp } from "../../vcg/space/color4.ts";
import { BVH, coneDirections, cosineHemisphere } from "../../vcg/space/index/bvh.ts";

export const FP = {
	FP_OCCLUSION: 0,
	FP_OBSCURANCE: 1,
	FP_SDF: 2,
	FP_REORIENT: 3,
	FP_SELECT_VISIBLE: 4,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_OCCLUSION]: {
		name: "Compute Ambient occlusion",
		pythonName: "compute_scalar_ambient_occlusion",
		info: "Compute the ambient occlusion of each face by ray casting, storing it in the face quality.",
	},
	[FP.FP_OBSCURANCE]: {
		name: "Compute Obscurance",
		pythonName: "compute_scalar_by_volumetric_obscurance",
		info:
			"Compute the obscurance of each face: like ambient occlusion, but a nearby occluder counts " +
			"for more than a distant one.",
	},
	[FP.FP_SDF]: {
		name: "Compute Shape-Diameter Function",
		pythonName: "compute_scalar_by_shape_diameter_function_per_vertex",
		info: "Compute the shape diameter function, the local thickness of the object at each face.",
	},
	[FP.FP_REORIENT]: {
		name: "Reorient face normals by geometry",
		pythonName: "meshing_re_orient_faces_by_geometry",
		info:
			"Flip the faces whose normals point into the object rather than out of it, judged by how " +
			"much of the surrounding space each side can see.",
	},
	[FP.FP_SELECT_VISIBLE]: {
		name: "Select Visible Faces",
		pythonName: "compute_selection_by_visibility_per_face",
		info: "Select the faces that can be seen from a given direction.",
	},
};

export class FilterEmbree extends FilterPlugin {
	pluginName(): string {
		return "FilterEmbree";
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
		return FilterClass.Quality;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override initParameterList(id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichInt("Rays", 64, {
				description: "Number of rays",
				tooltip: "How many rays are cast per face.",
			}),
		);
		list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
		switch (id) {
			case FP.FP_OBSCURANCE:
				list.add(
					new RichFloat("TAU", 0.1, {
						description: "Obscurance exponent",
						tooltip: "How fast a hit stops mattering with distance.",
					}),
				);
				break;
			case FP.FP_SDF:
				list.add(
					new RichFloat("cone_amplitude", 90, {
						description: "Cone amplitude",
						tooltip: "The full angle of the cone the rays are spread over, in degrees.",
					}),
				);
				break;
			case FP.FP_SELECT_VISIBLE:
				list.add(
					new RichDirection("dir", [1, 0, 0], {
						description: "Viewpoint direction",
						tooltip: "The direction the faces are looked at from.",
					}),
				);
				list.add(
					new RichBool("incrementalSelection", false, {
						description: "Incremental selection",
						tooltip: "Add to the current selection instead of replacing it.",
					}),
				);
				break;
			default:
				break;
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
		const rays = params.getInt("Rays");
		if (rays < 1) throw new MLException(`the ray count must be at least 1, got ${rays}`);

		UpdateBounding.box(cm);
		UpdateNormal.perFaceNormalized(cm);
		const bvh = new BVH(cm);
		const diagonal = cm.bbox.diagonal || 1;
		const epsilon = diagonal * 1e-5;
		const random = mulberry32(params.getInt("randomSeed") >>> 0 || 1);

		if (id === FP.FP_SELECT_VISIBLE) {
			const direction = normalise(params.getPoint3m("dir"));
			if (!params.getBool("incrementalSelection")) {
				for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] &= ~FaceFlag.SELECTED;
			}
			let selected = 0;
			for (let f = 0; f < cm.faceSize; f++) {
				if (cm.isFaceD(f)) continue;
				const n = [cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]];
				// Facing away from the viewer is invisible whatever lies in
				// between, and cheaper to reject than to trace.
				if (n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2] >= 0) continue;
				const centre = faceCentre(cm, f);
				const origin = [
					centre[0] + n[0] * epsilon,
					centre[1] + n[1] * epsilon,
					centre[2] + n[2] * epsilon,
				];
				const away = direction.map((c) => -c);
				if (bvh.occluded(origin, away, epsilon, diagonal * 4)) continue;
				cm.faceFlags[f] |= FaceFlag.SELECTED;
				selected++;
			}
			post.mask = MeshElement.MM_NONE;
			doc.Log.log(`Selected ${selected} faces visible from ${direction}`);
			return { selected_faces: selected };
		}

		if (id === FP.FP_REORIENT) {
			const flipped = reorient(cm, bvh, rays, diagonal, epsilon, random, cb);
			UpdateTopology.faceFace(cm);
			m.updateBoxAndNormals();
			post.mask = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;
			doc.Log.log(`Flipped ${flipped} faces to face outwards`);
			return { flipped_faces: flipped };
		}

		m.updateDataMask(MeshElement.MM_FACEQUALITY | MeshElement.MM_FACECOLOR);
		const values = new Float64Array(cm.faceSize);
		let seen = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			if (++seen % 256 === 0) cb((100 * seen) / cm.fn, "Casting rays");
			const n = [cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]];
			const centre = faceCentre(cm, f);
			const outward = [
				centre[0] + n[0] * epsilon,
				centre[1] + n[1] * epsilon,
				centre[2] + n[2] * epsilon,
			];
			const inward = [
				centre[0] - n[0] * epsilon,
				centre[1] - n[1] * epsilon,
				centre[2] - n[2] * epsilon,
			];

			if (id === FP.FP_OCCLUSION) {
				let open = 0;
				for (const d of cosineHemisphere(n, rays, random)) {
					if (!bvh.occluded(outward, d, epsilon, diagonal * 4)) open++;
				}
				values[f] = open / rays;
				continue;
			}
			if (id === FP.FP_OBSCURANCE) {
				const tau = params.getFloat("TAU");
				let sum = 0;
				for (const d of cosineHemisphere(n, rays, random)) {
					const hit = bvh.intersect(outward, d, epsilon, diagonal * 4);
					// An occluder far away barely darkens; one right here does.
					sum += hit === null ? 1 : 1 - Math.exp((-tau * hit.t * 100) / diagonal);
				}
				values[f] = sum / rays;
				continue;
			}

			const half = (params.getFloat("cone_amplitude") * Math.PI) / 180 / 2;
			const into = n.map((c) => -c);
			let sum = 0;
			let count = 0;
			for (const d of coneDirections(into, half, rays, random)) {
				const hit = bvh.intersect(inward, d, epsilon, diagonal * 4);
				if (hit === null || !hit.backface) continue;
				sum += hit.t;
				count++;
			}
			values[f] = count === 0 ? 0 : sum / count;
		}

		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			min = Math.min(min, values[f]);
			max = Math.max(max, values[f]);
		}
		const quality = cm.faceQuality;
		const colour = cm.faceColor;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			if (quality !== null) quality[f] = values[f];
			if (colour !== null) colour[f] = colorRamp(min, max === min ? min + 1 : max, values[f]);
		}
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(`${this.spec(id).name} over ${cm.fn} faces: ${min} to ${max}`);
		return { min, max, faces: cm.fn };
	}
}

/**
 * Flips the faces whose normals point inwards.
 *
 * Each face casts rays from both sides; the side that reaches open space more
 * often is the outside. This is a *geometric* test, unlike
 * `Re-Orient all faces coherently`, which only makes neighbours agree with
 * each other — a consistently-oriented mesh that is inside out passes that
 * one and fails this.
 */
function reorient(
	cm: CMeshO,
	bvh: BVH,
	rays: number,
	diagonal: number,
	epsilon: number,
	random: () => number,
	cb: CallBackPos,
): number {
	const flip: number[] = [];
	let seen = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		if (++seen % 256 === 0) cb((100 * seen) / cm.fn, "Testing orientation");
		const n = [cm.faceNormal[3 * f], cm.faceNormal[3 * f + 1], cm.faceNormal[3 * f + 2]];
		const centre = faceCentre(cm, f);
		const front = openness(bvh, centre, n, rays, diagonal, epsilon, random);
		const back = openness(
			bvh,
			centre,
			n.map((c) => -c),
			rays,
			diagonal,
			epsilon,
			random,
		);
		if (back > front) flip.push(f);
	}
	for (const f of flip) {
		cm.setFace(f, cm.fv(f, 0), cm.fv(f, 2), cm.fv(f, 1));
	}
	return flip.length;
}

function openness(
	bvh: BVH,
	centre: readonly number[],
	normal: readonly number[],
	rays: number,
	diagonal: number,
	epsilon: number,
	random: () => number,
): number {
	const origin = [
		centre[0] + normal[0] * epsilon,
		centre[1] + normal[1] * epsilon,
		centre[2] + normal[2] * epsilon,
	];
	let open = 0;
	for (const d of cosineHemisphere(normal, rays, random)) {
		if (!bvh.occluded(origin, d, epsilon, diagonal * 4)) open++;
	}
	return open / rays;
}

function faceCentre(cm: CMeshO, f: number): number[] {
	const p = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(f, k);
		p[0] += cm.vx(v) / 3;
		p[1] += cm.vy(v) / 3;
		p[2] += cm.vz(v) / 3;
	}
	return p;
}

function normalise(v: readonly number[]): number[] {
	const length = Math.hypot(v[0], v[1], v[2]);
	if (length === 0) throw new MLException("the direction has zero length");
	return [v[0] / length, v[1] / length, v[2] / length];
}
