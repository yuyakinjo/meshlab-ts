/**
 * `filter_fractal` — procedural relief.
 *
 * Two of the three evaluate a fractal field over a surface: `Fractal Terrain`
 * makes a grid first, `Fractal Displacement` uses whatever mesh is there. The
 * third drops craters onto a mesh at positions given by a second layer.
 *
 * All three displace along the *normal*, not along an axis, so they work on
 * something other than a plane — and all three are deterministic in their
 * seed, so a run can be repeated.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichDynamicFloat,
	RichEnum,
	RichFloat,
	RichInt,
	RichMesh,
	RichPercentage,
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
import { FRACTAL_DEFAULTS, FractalField, mulberry32 } from "../../vcg/math/noise.ts";

export const FP = {
	CR_FRACTAL_TERRAIN: 0,
	FP_FRACTAL_MESH: 1,
	FP_CRATERS: 2,
} as const;

const ALGORITHMS = [
	"fBM (fractal Brownian Motion)",
	"Standard multifractal",
	"Heterogeneous multifractal",
	"Hybrid multifractal terrain",
	"Ridged multifractal terrain",
];

/** The radial profile of a crater. */
const RBF = { GAUSSIAN: 0, MULTIQUADRIC: 1, VARIANT: 2 } as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.CR_FRACTAL_TERRAIN]: {
		name: "Fractal Terrain",
		pythonName: "create_fractal_terrain",
		info: "Generates a fractal terrain perturbation with five different algorithms.",
		filterClass: FilterClass.MeshCreation,
		arity: FilterArity.NONE,
	},
	[FP.FP_FRACTAL_MESH]: {
		name: "Fractal Displacement",
		pythonName: "apply_coord_fractal_displacement",
		info: "Perturbs a mesh along its normals with a fractal field. Hint: search a good compromise between offset and height factor parameter.",
		filterClass: FilterClass.Smoothing,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_CRATERS]: {
		name: "Craters Generation",
		pythonName: "apply_coord_craters_from_point_cloud",
		info:
			"Generates craters onto a mesh using radial functions. There must be at least two layers " +
			"to apply this filter: the layer that contains the target mesh, which we assume is " +
			"sufficiently refined, and the layer that contains the samples which represent the " +
			"central points of craters.",
		filterClass: FilterClass.Smoothing,
		arity: FilterArity.VARIABLE,
	},
};

export class FilterFractal extends FilterPlugin {
	pluginName(): string {
		return "FilterFractal";
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
	filterArity(id: ActionIDType): FilterArityValue {
		return this.spec(id).arity;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
		}

		if (id === FP.FP_CRATERS) {
			const current = m?.id() ?? 0;
			list.add(
				new RichMesh("target_mesh", current, {
					description: "Target mesh",
					tooltip: "The mesh the craters are cut into; it has to be finely tessellated.",
				}),
			);
			list.add(
				new RichMesh("samples_mesh", current, {
					description: "Samples layer",
					tooltip: "The point set whose points are the centres of the craters.",
				}),
			);
			list.add(new RichInt("seed", 0, { description: "Seed", tooltip: "The random seed." }));
			list.add(
				new RichEnum(
					"rbf",
					RBF.MULTIQUADRIC,
					["Gaussian", "Multiquadric", "Variant multiquadric"],
					{
						description: "Radial function",
						tooltip: "The profile of the crater from its centre out to its rim.",
					},
				),
			);
			list.add(
				new RichDynamicFloat("min_radius", 0.1, 0, 1, {
					description: "Min crater radius",
					tooltip: "As a fraction of the mesh's bounding diagonal.",
				}),
			);
			list.add(
				new RichDynamicFloat("max_radius", 0.35, 0, 1, {
					description: "Max crater radius",
					tooltip: "As a fraction of the mesh's bounding diagonal.",
				}),
			);
			list.add(
				new RichDynamicFloat("min_depth", 0.05, 0, 1, {
					description: "Min crater depth",
					tooltip: "As a fraction of the mesh's bounding diagonal.",
				}),
			);
			list.add(
				new RichDynamicFloat("max_depth", 0.15, 0, 1, {
					description: "Max crater depth",
					tooltip: "As a fraction of the mesh's bounding diagonal.",
				}),
			);
			list.add(
				new RichDynamicFloat("elevation", 0.4, 0, 1, {
					description: "Rim elevation",
					tooltip: "How high the rim rises relative to the crater's depth.",
				}),
			);
			list.add(
				new RichBool("save_as_quality", false, {
					description: "Save as vertex quality",
					tooltip: "Write the displacement into the quality channel instead of moving the mesh.",
				}),
			);
			return list;
		}

		const terrain = id === FP.CR_FRACTAL_TERRAIN;
		if (terrain) {
			list.add(
				new RichInt("steps", 8, {
					description: "Subdivision steps",
					tooltip: "The grid is 2^steps cells on a side.",
				}),
			);
			list.add(
				new RichDynamicFloat("maxHeight", 0.2, 0, 1, {
					description: "Max height",
					tooltip: "The relief is scaled so its range is this tall.",
				}),
			);
		} else {
			list.add(
				new RichPercentage("maxHeight", 0.02 * diagonal, 0, diagonal, {
					description: "Max height",
					tooltip: "The relief is scaled so its range is this tall.",
				}),
			);
			list.add(
				new RichDynamicFloat("scale", 1, 0, 10, {
					description: "Scale factor",
					tooltip: "How large the features are relative to the mesh.",
				}),
			);
		}
		list.add(new RichFloat("seed", FRACTAL_DEFAULTS.seed, { description: "Seed" }));
		list.add(
			new RichEnum("algorithm", FRACTAL_DEFAULTS.algorithm, ALGORITHMS, {
				description: "Algorithm",
				tooltip: "The fractal function; they differ in whether the roughness varies with height.",
			}),
		);
		list.add(
			new RichDynamicFloat("octaves", FRACTAL_DEFAULTS.octaves, 1, 20, {
				description: "Octaves",
				tooltip: "How many scales of detail are summed.",
			}),
		);
		list.add(
			new RichFloat("lacunarity", FRACTAL_DEFAULTS.lacunarity, {
				description: "Lacunarity",
				tooltip: "The frequency step between octaves.",
			}),
		);
		list.add(
			new RichFloat("fractalIncrement", terrain ? 0.5 : 0.2, {
				description: "Fractal increment",
				tooltip: "How fast the octaves fade; larger means smoother.",
			}),
		);
		list.add(new RichFloat("offset", FRACTAL_DEFAULTS.offset, { description: "Offset" }));
		list.add(new RichFloat("gain", FRACTAL_DEFAULTS.gain, { description: "Gain" }));
		list.add(
			new RichBool("saveAsQuality", false, {
				description: "Save as vertex quality",
				tooltip: "Write the displacement into the quality channel instead of moving the mesh.",
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
		if (id === FP.FP_CRATERS) return craters(params, doc, post);

		const terrain = id === FP.CR_FRACTAL_TERRAIN;
		const field = new FractalField({
			algorithm: params.getEnum("algorithm"),
			octaves: params.getDynamicFloat("octaves"),
			lacunarity: params.getFloat("lacunarity"),
			fractalIncrement: params.getFloat("fractalIncrement"),
			offset: params.getFloat("offset"),
			gain: params.getFloat("gain"),
			seed: params.getFloat("seed"),
		});

		let m: MeshModel;
		let scale: number;
		let maxHeight: number;
		if (terrain) {
			const steps = params.getInt("steps");
			if (steps < 1 || steps > 12) {
				throw new MLException(`the subdivision steps must be within 1..12, got ${steps}`);
			}
			m = doc.addNewMesh("", "Fractal Terrain", true, unitGrid(2 ** steps));
			scale = 1;
			maxHeight = params.getDynamicFloat("maxHeight");
		} else {
			m = doc.mm();
			UpdateBounding.box(m.cm);
			scale = params.getDynamicFloat("scale") / (m.cm.bbox.diagonal || 1);
			maxHeight = params.getAbsPerc("maxHeight");
		}

		const cm = m.cm;
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const displacement = new Float64Array(cm.vertSize);
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		let seen = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			if (++seen % 4096 === 0) cb((100 * seen) / cm.vn, "Evaluating the fractal field");
			const value = field.at(cm.vx(v) * scale, cm.vy(v) * scale, cm.vz(v) * scale);
			displacement[v] = value;
			min = Math.min(min, value);
			max = Math.max(max, value);
		}
		// Normalising the *range* rather than the raw values is what makes
		// `maxHeight` mean the same thing across the five algorithms, whose
		// natural outputs differ by orders of magnitude.
		const span = max - min || 1;

		const asQuality = params.getBool("saveAsQuality");
		if (asQuality) m.updateDataMask(MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const height = ((displacement[v] - min) / span) * maxHeight;
			if (asQuality) {
				cm.vertQuality[v] = height;
			} else {
				cm.setVert(
					v,
					cm.vx(v) + cm.vertNormal[3 * v] * height,
					cm.vy(v) + cm.vertNormal[3 * v + 1] * height,
					cm.vz(v) + cm.vertNormal[3 * v + 2] * height,
				);
			}
		}
		m.updateBoxAndNormals();
		post.mask = asQuality ? MeshElement.MM_NONE : MeshElement.MM_VERTCOORD;
		doc.Log.log(
			`${terrain ? "Generated" : "Displaced"} ${cm.vn} vertices with ${ALGORITHMS[params.getEnum("algorithm")]}`,
		);
		return terrain
			? { new_mesh_id: m.id(), vertex_number: cm.vn, face_number: cm.fn }
			: { displaced_vertices: cm.vn, max_height: maxHeight };
	}
}

/**
 * Craters, cut where a second layer's points fall.
 *
 * Each sample becomes a crater of a random radius and depth. Within its
 * radius the surface is pushed in by the radial profile and lifted at the rim
 * — the lift is what makes it read as an impact rather than a dent.
 */
function craters(
	params: RichParameterList,
	doc: MeshDocument,
	post: PostConditionBox,
): FilterOutput {
	const target = doc.requireMesh(params.getMeshId("target_mesh"));
	const samples = doc.requireMesh(params.getMeshId("samples_mesh"));
	if (target.id() === samples.id()) {
		throw new MLException("the target and the samples must be two different layers");
	}
	if (samples.cm.vn === 0) throw new MLException("the samples layer has no points");

	UpdateBounding.box(target.cm);
	const diagonal = target.cm.bbox.diagonal || 1;
	const minRadius = params.getDynamicFloat("min_radius") * diagonal;
	const maxRadius = params.getDynamicFloat("max_radius") * diagonal;
	if (maxRadius < minRadius) {
		throw new MLException(`the crater radius range is inverted: ${minRadius} to ${maxRadius}`);
	}
	const minDepth = params.getDynamicFloat("min_depth") * diagonal;
	const maxDepth = params.getDynamicFloat("max_depth") * diagonal;
	const elevation = params.getDynamicFloat("elevation");
	const profile = params.getEnum("rbf");
	const random = mulberry32(params.getInt("seed") >>> 0 || 1);

	const cm = target.cm;
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const displacement = new Float64Array(cm.vertSize);

	let made = 0;
	for (let s = 0; s < samples.cm.vertSize; s++) {
		if (samples.cm.isVertD(s)) continue;
		const radius = minRadius + random() * (maxRadius - minRadius);
		const depth = minDepth + random() * (maxDepth - minDepth);
		if (radius <= 0) continue;
		made++;
		const cx = samples.cm.vx(s);
		const cy = samples.cm.vy(s);
		const cz = samples.cm.vz(s);

		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const d = Math.hypot(cm.vx(v) - cx, cm.vy(v) - cy, cm.vz(v) - cz);
			if (d >= radius) continue;
			const t = d / radius;
			// Negative in the bowl, positive at the rim, zero at the edge.
			displacement[v] += depth * (elevation * rim(t) - bowl(t, profile));
		}
	}

	const asQuality = params.getBool("save_as_quality");
	if (asQuality) target.updateDataMask(MeshElement.MM_VERTQUALITY);
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		if (asQuality) {
			cm.vertQuality[v] = displacement[v];
			continue;
		}
		cm.setVert(
			v,
			cm.vx(v) + cm.vertNormal[3 * v] * displacement[v],
			cm.vy(v) + cm.vertNormal[3 * v + 1] * displacement[v],
			cm.vz(v) + cm.vertNormal[3 * v + 2] * displacement[v],
		);
	}
	target.updateBoxAndNormals();
	post.mask = asQuality ? MeshElement.MM_NONE : MeshElement.MM_VERTCOORD;
	doc.Log.log(`Cut ${made} craters into "${target.label()}"`);
	return { craters: made };
}

/** The bowl: one at the centre, zero at the rim. */
function bowl(t: number, profile: number): number {
	switch (profile) {
		case RBF.GAUSSIAN:
			// Shifted so it reaches exactly zero at t = 1 rather than trailing.
			return (Math.exp(-4 * t * t) - Math.exp(-4)) / (1 - Math.exp(-4));
		case RBF.VARIANT:
			return 1 / (1 + 4 * t * t) - 1 / 5;
		default:
			return 1 - Math.sqrt(t);
	}
}

/** The rim: zero at the centre and at the edge, peaking near the edge. */
function rim(t: number): number {
	const x = Math.max(0, Math.min(1, t));
	return 4 * x * x * (1 - x);
}

/** A flat unit grid, `cells` on a side, for the terrain to be displaced from. */
function unitGrid(cells: number): CMeshO {
	const cm = new CMeshO();
	const side = cells + 1;
	Allocator.addVertices(cm, side * side);
	for (let j = 0; j < side; j++) {
		for (let i = 0; i < side; i++) {
			cm.setVert(j * side + i, i / cells - 0.5, j / cells - 0.5, 0);
		}
	}
	for (let j = 0; j < cells; j++) {
		for (let i = 0; i < cells; i++) {
			const a = j * side + i;
			Allocator.addFace(cm, a, a + 1, a + side + 1);
			Allocator.addFace(cm, a, a + side + 1, a + side);
		}
	}
	UpdateBounding.box(cm);
	return cm;
}
