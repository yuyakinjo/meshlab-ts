/**
 * `filter_sampling` — drawing point sets from a mesh, and comparing meshes.
 *
 * These filters do not edit the current mesh: they add a new layer holding the
 * samples, which is why several of them have MeshCreation-like behaviour
 * despite being classed as Sampling. `Hausdorff Distance` edits nothing at all
 * and reports numbers.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichEnum,
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
import { Rng, SurfaceSampling } from "../../vcg/complex/point_sampling.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";

export const FP = {
	FP_ELEMENT_SUBSAMPLING: 0,
	FP_MONTECARLO_SAMPLING: 1,
	FP_STRATIFIED_SAMPLING: 2,
	FP_CLUSTERED_SAMPLING: 3,
	FP_POISSONDISK_SAMPLING: 4,
	FP_POINTCLOUD_SIMPLIFICATION: 5,
	FP_HAUSDORFF_DISTANCE: 6,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_ELEMENT_SUBSAMPLING]: {
		name: "Mesh Element Sampling",
		pythonName: "generate_sampling_element",
		info:
			"Create a new layer populated with a point sampling of the current mesh; at most one " +
			"sample for each element of the mesh is created. Samples are taking in a uniform way, " +
			"one for each element (vertex/edge/face); all the elements have the same probabilty of " +
			"being choosen.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_MONTECARLO_SAMPLING]: {
		name: "Montecarlo Sampling",
		pythonName: "generate_sampling_montecarlo",
		info:
			"Create a new layer populated with a point sampling of the current mesh; samples are " +
			"generated in a randomly uniform way, or with a distribution biased by the per-vertex " +
			"quality values of the mesh.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_STRATIFIED_SAMPLING]: {
		name: "Stratified Triangle Sampling",
		pythonName: "generate_sampling_stratified_triangle",
		info:
			"Create a new layer populated with a point sampling of the current mesh; to generate " +
			"multiple samples inside a triangle each triangle is subdivided according to various " +
			"strategies and uniform point sampling is applied.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_CLUSTERED_SAMPLING]: {
		name: "Clustered Vertex Sampling",
		pythonName: "generate_sampling_clustered_vertex",
		info:
			"Create a new layer populated with a subsampling of the vertices of the current mesh; " +
			"the subsampling is driven by a simple one-per-grid-cell strategy.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_POISSONDISK_SAMPLING]: {
		name: "Poisson-disk Sampling",
		pythonName: "generate_sampling_poisson_disk",
		info:
			"Create a new layer populated with a point sampling of the current mesh; samples are " +
			"generated according to a Poisson-disk distribution, using the algorithm described in: " +
			"'Efficient and Flexible Sampling with Blue Noise Properties of Triangular Meshes', " +
			"Corsini, Cignoni, Scopigno, IEEE TVCG 2012.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.FIXED,
	},
	[FP.FP_POINTCLOUD_SIMPLIFICATION]: {
		name: "Point Cloud Simplification",
		pythonName: "generate_simplified_point_cloud",
		info:
			"Create a new layer populated with a simplified version of the current point cloud. The " +
			"simplification is performed by subsampling the original point cloud using a Poisson " +
			"Disk strategy.",
		filterClass: FilterClass.Sampling | FilterClass.PointSet,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_HAUSDORFF_DISTANCE]: {
		name: "Hausdorff Distance",
		pythonName: "get_hausdorff_distance",
		info: "Compute the Hausdorff Distance between two layers, sampling one of the two.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.FIXED,
	},
};

export class FilterSampling extends FilterPlugin {
	pluginName(): string {
		return "FilterSampling";
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

	/**
	 * These add a layer rather than editing the current mesh, so the current
	 * mesh needs no recomputation. `Hausdorff Distance` changes nothing at all.
	 */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diag = 1;
		let vertexCount = 1000;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diag = m.cm.bbox.diagonal || 1;
			vertexCount = Math.max(1, m.cm.vn);
		}

		switch (id) {
			case FP.FP_ELEMENT_SUBSAMPLING:
				list.add(
					new RichEnum("Sampling", 0, ["Vertex", "Edge", "Face"], {
						description: "Element to sample:",
						tooltip:
							"Choose what mesh element has to be used for the subsampling. At most one " +
							"sample will be added for each one.",
					}),
				);
				list.add(
					new RichInt("SampleNum", vertexCount, {
						description: "Number of samples",
						tooltip: "The desired number of elements that must be chosen.",
					}),
				);
				break;

			case FP.FP_MONTECARLO_SAMPLING:
			case FP.FP_STRATIFIED_SAMPLING:
				list.add(
					new RichInt("SampleNum", vertexCount, {
						description: "Number of samples",
						tooltip:
							"The desired number of samples. It can be smaller or larger than the mesh size.",
					}),
				);
				break;

			case FP.FP_CLUSTERED_SAMPLING:
				list.add(
					new RichPercentage("Threshold", diag * 0.01, 0, diag, {
						description: "Cell Size",
						tooltip:
							"The size of the cell of the clustering grid. Smaller the cell finer the " +
							"resulting mesh. For obtaining a very coarse mesh use larger values.",
					}),
				);
				list.add(
					new RichEnum("Sampling", 1, ["Average", "Closest to center"], {
						description: "Representative Strategy:",
						tooltip:
							"Choose the representative strategy for each cell: the average of the points, " +
							"or the one closest to the cell centre.",
					}),
				);
				break;

			case FP.FP_POISSONDISK_SAMPLING:
			case FP.FP_POINTCLOUD_SIMPLIFICATION:
				list.add(
					new RichInt("SampleNum", 1000, {
						description: "Number of samples",
						tooltip: "The desired number of samples. Used only if the radius is zero.",
					}),
				);
				list.add(
					new RichPercentage("Radius", 0, 0, diag, {
						description: "Explicit Radius",
						tooltip:
							"If not zero this parameter overrides the previous parameter to allow exact " +
							"radius specification.",
					}),
				);
				list.add(
					new RichInt("MontecarloRate", 20, {
						description: "MonterCarlo OverSampling",
						tooltip:
							"The over-sampling rate that is used to generate the initial Montecarlo samples " +
							"(e.g. if this parameter is 'K' means that we will generate K * <sample number> " +
							"samples).",
					}),
				);
				break;

			case FP.FP_HAUSDORFF_DISTANCE:
				list.add(
					new RichMesh("SampledMesh", 0, {
						description: "Sampled Mesh",
						tooltip:
							"The mesh whose surface is sampled. For each sample we search the closest point on the Target Mesh.",
					}),
				);
				list.add(
					new RichMesh("TargetMesh", 0, {
						description: "Target Mesh",
						tooltip: "The mesh that is sampled for the comparison.",
					}),
				);
				list.add(
					new RichInt("SampleNum", vertexCount, {
						description: "Number of samples",
						tooltip:
							"The desired number of samples. It can be smaller or larger than the mesh size.",
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
		_cb: CallBackPos,
	): FilterOutput {
		post.mask = MeshElement.MM_NONE;
		const source = doc.mm();

		switch (id) {
			case FP.FP_ELEMENT_SUBSAMPLING: {
				if (params.getEnum("Sampling") !== 0) {
					throw new MLException(
						"Mesh Element Sampling currently supports only the Vertex element; " +
							"Edge and Face sampling are not implemented yet.",
					);
				}
				const cloud = SurfaceSampling.vertexSampling(source.cm);
				return this.addLayer(doc, source, cloud, "Vertex Samples");
			}

			case FP.FP_MONTECARLO_SAMPLING: {
				const cloud = SurfaceSampling.montecarloSampling(
					source.cm,
					params.getInt("SampleNum"),
					new Rng(),
				);
				return this.addLayer(doc, source, cloud, "Montecarlo Samples");
			}

			case FP.FP_STRATIFIED_SAMPLING: {
				const cloud = SurfaceSampling.stratifiedSampling(
					source.cm,
					params.getInt("SampleNum"),
					new Rng(),
				);
				return this.addLayer(doc, source, cloud, "Stratified Samples");
			}

			case FP.FP_CLUSTERED_SAMPLING: {
				if (params.getEnum("Sampling") !== 1) {
					throw new MLException(
						'Clustered Vertex Sampling currently supports only the "Closest to center" ' +
							'strategy; "Average" is not implemented yet.',
					);
				}
				UpdateBounding.box(source.cm);
				const diag = source.cm.bbox.diagonal || 1;
				// The filter takes a cell size; the sampler takes it as a
				// percentage of the diagonal, which is how MeshLab's default is
				// expressed in the first place.
				const percent = (params.getAbsPerc("Threshold") / diag) * 100;
				const cloud = SurfaceSampling.clusteredVertexSampling(source.cm, percent);
				return this.addLayer(doc, source, cloud, "Cluster Samples");
			}

			case FP.FP_POISSONDISK_SAMPLING:
			case FP.FP_POINTCLOUD_SIMPLIFICATION: {
				UpdateBounding.box(source.cm);
				const diag = source.cm.bbox.diagonal || 1;
				const sampleNum = params.getInt("SampleNum");
				let radius = params.getAbsPerc("Radius");
				if (radius <= 0) {
					// MeshLab's rule of thumb for turning a sample count into a
					// radius: the disks tile roughly the surface area.
					const area = surfaceAreaOf(source);
					radius = area > 0 ? Math.sqrt(area / (sampleNum * Math.PI)) : diag / 100;
				}

				// A point cloud has no faces to scatter over, so its own points
				// are the candidates; a surface gets a dense Montecarlo pass
				// first, which is what MontecarloRate controls.
				const candidates =
					source.cm.fn === 0
						? SurfaceSampling.vertexSampling(source.cm)
						: SurfaceSampling.montecarloSampling(
								source.cm,
								Math.max(sampleNum, sampleNum * params.getInt("MontecarloRate")),
								new Rng(),
							);
				const cloud = SurfaceSampling.poissonDiskPruning(candidates, radius);
				return this.addLayer(doc, source, cloud, "Poisson-disk Samples", { radius });
			}

			case FP.FP_HAUSDORFF_DISTANCE: {
				const sampled = doc.requireMesh(params.getMeshId("SampledMesh"));
				const target = doc.requireMesh(params.getMeshId("TargetMesh"));
				// Point-to-point against a sampling of the target, so a coarse
				// target must be sampled densely or the distance is overstated.
				const targetCloud =
					target.cm.fn === 0
						? SurfaceSampling.vertexSampling(target.cm)
						: SurfaceSampling.montecarloSampling(target.cm, params.getInt("SampleNum"), new Rng());
				const sampledCloud =
					sampled.cm.fn === 0
						? SurfaceSampling.vertexSampling(sampled.cm)
						: SurfaceSampling.montecarloSampling(sampled.cm, params.getInt("SampleNum"), new Rng());
				const d = SurfaceSampling.hausdorffPointDistance(sampledCloud, targetCloud);
				UpdateBounding.box(sampled.cm);
				const diag = sampled.cm.bbox.diagonal || 1;
				doc.Log.log(`Hausdorff Distance computed: max ${d.max}  mean ${d.mean}  RMS ${d.rms}`);
				return {
					max: d.max,
					mean: d.mean,
					RMS: d.rms,
					diag_mesh_0: diag,
					max_over_diag: d.max / diag,
				};
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	/** Adds the sampled cloud as a new layer and reports what landed in it. */
	private addLayer(
		doc: MeshDocument,
		source: MeshModel,
		cloud: ReturnType<typeof SurfaceSampling.vertexSampling>,
		label: string,
		extra: FilterOutput = {},
	): FilterOutput {
		const target = doc.addNewMesh("", `${label} (${source.label()})`, true, cloud);
		target.updateBoxAndNormals();
		doc.Log.log(`Sampled ${cloud.vn} points into "${target.label()}"`);
		return { ...extra, sample_num: cloud.vn, new_mesh_id: target.id() };
	}
}

function surfaceAreaOf(m: MeshModel): number {
	const n = new Float64Array(3);
	let total = 0;
	for (let f = 0; f < m.cm.faceSize; f++) {
		if (m.cm.isFaceD(f)) continue;
		const a = m.cm.fv(f, 0);
		const b = m.cm.fv(f, 1);
		const c = m.cm.fv(f, 2);
		const ux = m.cm.vx(b) - m.cm.vx(a);
		const uy = m.cm.vy(b) - m.cm.vy(a);
		const uz = m.cm.vz(b) - m.cm.vz(a);
		const vx = m.cm.vx(c) - m.cm.vx(a);
		const vy = m.cm.vy(c) - m.cm.vy(a);
		const vz = m.cm.vz(c) - m.cm.vz(a);
		n[0] = uy * vz - uz * vy;
		n[1] = uz * vx - ux * vz;
		n[2] = ux * vy - uy * vx;
		total += Math.hypot(n[0], n[1], n[2]) / 2;
	}
	return total;
}
