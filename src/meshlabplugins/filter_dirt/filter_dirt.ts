/**
 * `filter_dirt` — where dust would settle, and where it would slide to.
 *
 * A particle is dropped on the surface and pushed along it by gravity. Where
 * the surface is steep it keeps sliding; where it is flat, or in a pocket, it
 * stops. Run enough particles and the places they stop are the places dirt
 * accumulates — an upward-facing ledge, the inside of a crease, the flat top
 * of anything.
 *
 * The two filters are the same simulation seen from two ends. `Dust
 * Accumulation` reports where the particles ended up as a point cloud;
 * `Points Cloud Movement` takes a cloud that is already there and moves it.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
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
import { Allocator } from "../../vcg/complex/allocator.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { mulberry32 } from "../../vcg/math/noise.ts";
import { colorRamp } from "../../vcg/space/color4.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = { FP_DIRT: 0, FP_CLOUD_MOVEMENT: 1 } as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_DIRT]: {
		name: "Dust Accumulation",
		pythonName: "generate_dust_accumulation_point_cloud",
		info: "Simulate dust accumulation over the mesh generating a cloud of points lying on the current mesh",
		filterClass: FilterClass.Sampling,
	},
	[FP.FP_CLOUD_MOVEMENT]: {
		name: "Points Cloud Movement",
		pythonName: "apply_coord_point_cloud_movement_over_mesh",
		info: "Simulate the movement of a point cloud over a mesh",
		filterClass: FilterClass.Remeshing,
	},
};

export class FilterDirt extends FilterPlugin {
	pluginName(): string {
		return "FilterDirt";
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
	override getClass(id: ActionIDType): FilterClassMask {
		return this.spec(id).filterClass;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichDirection("dust_dir", [0, 1, 0], {
				description: id === FP.FP_DIRT ? "Direction" : "Gravity direction",
				tooltip: "The direction dust falls from — usually straight down, negated.",
			}),
		);
		list.add(
			new RichInt("nparticles", id === FP.FP_DIRT ? 3 : 1, {
				description: id === FP.FP_DIRT ? "Particles per face" : "Steps",
				tooltip:
					id === FP.FP_DIRT
						? "How many particles are dropped onto each face."
						: "How many simulation steps to run.",
			}),
		);
		list.add(
			new RichFloat("slippiness", 1, {
				description: "Slippiness",
				tooltip:
					"How readily a particle slides. A larger value lets it travel further before it " +
					"settles, so dust gathers only in the deepest pockets.",
			}),
		);
		list.add(
			new RichFloat("adhesion", 0.2, {
				description: "Adhesion",
				tooltip:
					"How strongly the surface holds a particle. Above the slippiness nothing moves at " +
					"all, which is the sensible meaning of a sticky surface.",
			}),
		);
		if (id === FP.FP_DIRT) {
			list.add(
				new RichBool("colorize_mesh", false, {
					description: "Map to mesh color",
					tooltip: "Colour the mesh by how much dust settled on each face.",
				}),
			);
		}
		list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
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
		if (cm.fn === 0) throw new MLException("the mesh has no faces for dust to settle on");

		const gravity = normalise(params.getPoint3m("dust_dir"));
		const slippiness = params.getFloat("slippiness");
		const adhesion = params.getFloat("adhesion");
		if (slippiness < 0)
			throw new MLException(`the slippiness cannot be negative, got ${slippiness}`);
		const steps = params.getInt("nparticles");
		if (steps < 1) throw new MLException(`the count must be at least 1, got ${steps}`);

		UpdateBounding.box(cm);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const lookup = new SurfaceLookup(cm, cm.bbox.diagonal || 1);
		const stepSize = (cm.bbox.diagonal || 1) / 200;

		if (id === FP.FP_CLOUD_MOVEMENT) {
			// The current layer is the cloud; the surface is the layer below.
			const surface = doc.meshIterator().find((x) => x.id() !== m.id() && x.cm.fn > 0);
			if (surface === undefined) {
				throw new MLException(
					"there is no other layer with faces for the cloud to move over; add the surface as a " +
						"second layer",
				);
			}
			UpdateBounding.box(surface.cm);
			UpdateNormal.perVertexNormalizedPerFaceNormalized(surface.cm);
			const surfaceLookup = new SurfaceLookup(surface.cm, surface.cm.bbox.diagonal || 1);
			const surfaceStep = (surface.cm.bbox.diagonal || 1) / 200;

			let moved = 0;
			for (let v = 0; v < cm.vertSize; v++) {
				if (cm.isVertD(v)) continue;
				const start = [cm.vx(v), cm.vy(v), cm.vz(v)];
				const end = slide(surface.cm, surfaceLookup, start, gravity, {
					steps,
					slippiness,
					adhesion,
					stepSize: surfaceStep,
				});
				if (end === null) continue;
				cm.setVert(v, end[0], end[1], end[2]);
				moved++;
			}
			m.updateBoxAndNormals();
			post.mask = MeshElement.MM_VERTCOORD;
			doc.Log.log(`Moved ${moved} points over "${surface.label()}"`);
			return { moved_points: moved };
		}

		const random = mulberry32(params.getInt("randomSeed") >>> 0 || 1);
		const perFace = steps;
		const settled: number[] = [];
		const counts = new Int32Array(cm.faceSize);

		let seen = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			if (++seen % 512 === 0) cb((100 * seen) / cm.fn, "Dropping particles");
			for (let i = 0; i < perFace; i++) {
				const start = randomPointOn(cm, f, random);
				const end = slide(cm, lookup, start, gravity, {
					steps: 200,
					slippiness,
					adhesion,
					stepSize,
				});
				if (end === null) continue;
				settled.push(end[0], end[1], end[2]);
				const where = lookup.closest(end[0], end[1], end[2]);
				if (where !== null) counts[where.face]++;
			}
		}
		if (settled.length === 0) {
			throw new MLException(
				"no particle settled; the adhesion may exceed the slippiness everywhere",
			);
		}

		if (params.getBool("colorize_mesh")) {
			m.updateDataMask(MeshElement.MM_FACECOLOR | MeshElement.MM_FACEQUALITY);
			let max = 0;
			for (const c of counts) max = Math.max(max, c);
			const colour = cm.faceColor;
			const quality = cm.faceQuality;
			for (let f = 0; f < cm.faceSize; f++) {
				if (cm.isFaceD(f)) continue;
				if (quality !== null) quality[f] = counts[f];
				if (colour !== null) colour[f] = colorRamp(0, max || 1, counts[f]);
			}
		}

		const cloud = new CMeshO();
		const count = settled.length / 3;
		Allocator.addVertices(cloud, count);
		for (let i = 0; i < count; i++) {
			cloud.setVert(i, settled[3 * i], settled[3 * i + 1], settled[3 * i + 2]);
		}
		const target = doc.addNewMesh("", `${m.label()} dust`, true, cloud);
		target.updateBoxAndNormals();
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(`Settled ${count} dust particles onto "${m.label()}"`);
		return { new_mesh_id: target.id(), particles: count };
	}
}

interface SlideOptions {
	readonly steps: number;
	readonly slippiness: number;
	readonly adhesion: number;
	readonly stepSize: number;
}

/**
 * Slides a particle down the surface until it stops.
 *
 * Each step projects gravity onto the local tangent plane — the part of it
 * that can actually move the particle along the surface — and compares its
 * strength against the adhesion. Below that threshold the particle is stuck,
 * which is why a flat upward-facing face collects dust and a vertical one
 * does not: on the flat face the tangential component is zero.
 *
 * Reprojecting onto the surface after every step is what keeps the particle
 * on the mesh rather than tunnelling off a convex edge.
 */
function slide(
	cm: CMeshO,
	lookup: SurfaceLookup,
	start: readonly number[],
	gravity: readonly number[],
	options: SlideOptions,
): number[] | null {
	let position = [...start];
	for (let step = 0; step < options.steps; step++) {
		const hit = lookup.closest(position[0], position[1], position[2]);
		if (hit === null) return position;

		const normal = [0, 0, 0];
		const onSurface = [0, 0, 0];
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(hit.face, k);
			onSurface[0] += cm.vx(v) * hit.bary[k];
			onSurface[1] += cm.vy(v) * hit.bary[k];
			onSurface[2] += cm.vz(v) * hit.bary[k];
			for (let a = 0; a < 3; a++) normal[a] += cm.vertNormal[3 * v + a] * hit.bary[k];
		}
		const nl = Math.hypot(normal[0], normal[1], normal[2]) || 1;
		for (let a = 0; a < 3; a++) normal[a] /= nl;

		// Gravity minus its component along the normal: what is left is the
		// pull along the surface.
		const along = gravity[0] * normal[0] + gravity[1] * normal[1] + gravity[2] * normal[2];
		const tangent = [
			gravity[0] - along * normal[0],
			gravity[1] - along * normal[1],
			gravity[2] - along * normal[2],
		];
		const pull = Math.hypot(tangent[0], tangent[1], tangent[2]);
		if (pull * options.slippiness <= options.adhesion) return onSurface;

		const scale = (options.stepSize * options.slippiness) / pull;
		position = [
			onSurface[0] + tangent[0] * scale,
			onSurface[1] + tangent[1] * scale,
			onSurface[2] + tangent[2] * scale,
		];
	}
	// Ran out of steps still sliding: report where it got to, on the surface.
	const hit = lookup.closest(position[0], position[1], position[2]);
	if (hit === null) return position;
	const out = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(hit.face, k);
		out[0] += cm.vx(v) * hit.bary[k];
		out[1] += cm.vy(v) * hit.bary[k];
		out[2] += cm.vz(v) * hit.bary[k];
	}
	return out;
}

/** A uniformly distributed point inside a face. */
function randomPointOn(cm: CMeshO, f: number, random: () => number): number[] {
	// The square root makes the distribution uniform over the *area*; without
	// it the samples crowd towards the first corner.
	let u = random();
	let v = random();
	if (u + v > 1) {
		u = 1 - u;
		v = 1 - v;
	}
	const a = cm.fv(f, 0);
	const b = cm.fv(f, 1);
	const c = cm.fv(f, 2);
	return [
		cm.vx(a) + u * (cm.vx(b) - cm.vx(a)) + v * (cm.vx(c) - cm.vx(a)),
		cm.vy(a) + u * (cm.vy(b) - cm.vy(a)) + v * (cm.vy(c) - cm.vy(a)),
		cm.vz(a) + u * (cm.vz(b) - cm.vz(a)) + v * (cm.vz(c) - cm.vz(a)),
	];
}

function normalise(v: readonly number[]): number[] {
	const length = Math.hypot(v[0], v[1], v[2]);
	if (length === 0) throw new MLException("the gravity direction has zero length");
	return [v[0] / length, v[1] / length, v[2] / length];
}
