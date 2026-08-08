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
	RichBool,
	RichDynamicFloat,
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
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import {
	distanceField,
	extractLevelSet,
	type Grid,
	gridFor,
	paddedBox,
} from "../../vcg/complex/create/resampler.ts";
import { VertexFlag } from "../../vcg/complex/flags.ts";
import { dijkstraGeodesic } from "../../vcg/complex/geodesic.ts";
import { Rng, SurfaceSampling } from "../../vcg/complex/point_sampling.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { alpha, blue, colorRamp, green, lerpColor, red, rgba } from "../../vcg/space/color4.ts";
import { isPng, readPng } from "../../vcg/space/image/png.ts";
import { KdTree } from "../../vcg/space/index/kdtree.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = {
	FP_ELEMENT_SUBSAMPLING: 0,
	FP_MONTECARLO_SAMPLING: 1,
	FP_STRATIFIED_SAMPLING: 2,
	FP_CLUSTERED_SAMPLING: 3,
	FP_POISSONDISK_SAMPLING: 4,
	FP_POINTCLOUD_SIMPLIFICATION: 5,
	FP_HAUSDORFF_DISTANCE: 6,
	FP_DISTANCE_REFERENCE: 7,
	FP_VERTEX_RESAMPLING: 8,
	FP_UNIFORM_MESH_RESAMPLING: 9,
	FP_VORONOI_COLORING: 10,
	FP_DISK_COLORING: 11,
	FP_REGULAR_RECURSIVE_SAMPLING: 12,
	FP_TEXEL_SAMPLING: 13,
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
	[FP.FP_DISTANCE_REFERENCE]: {
		name: "Distance from Reference Mesh",
		pythonName: "compute_scalar_by_distance_from_another_mesh_per_vertex",
		info:
			"Compute the signed/unsigned (per vertex) distance between a mesh/pointcloud and a " +
			"reference mesh/pointcloud. Distance is stored in vertex quality.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.FIXED,
	},
	[FP.FP_VERTEX_RESAMPLING]: {
		name: "Vertex Attribute Transfer",
		pythonName: "transfer_attributes_per_vertex",
		info:
			"Transfer the chosen per-vertex attributes from one mesh to another. Useful to transfer " +
			"attributes to different representations of a same object.<br>For each vertex of the " +
			"target mesh the closest point (not vertex!) on the source mesh is computed, and the " +
			"requested interpolated attributes from that source point are copied into the target " +
			"vertex.<br>The algorithm assumes that the two meshes are reasonably similar and well " +
			"aligned.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.FIXED,
	},
	[FP.FP_UNIFORM_MESH_RESAMPLING]: {
		name: "Uniform Mesh Resampling",
		pythonName: "generate_resampled_uniform_mesh",
		info:
			"Create a new mesh that is a resampled version of the current one.<br>The resampling is " +
			"done by building a uniform volumetric representation where each voxel contains the " +
			"signed distance from the original surface. The resampled surface is reconstructed using " +
			"the <b>marching cube</b> algorithm over this volume.",
		filterClass: FilterClass.Remeshing,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_VORONOI_COLORING]: {
		name: "Voronoi Vertex Coloring",
		pythonName: "compute_color_by_point_cloud_voronoi_projection",
		info:
			"Given a Mesh <b>M</b> and a Pointset <b>P</b>, The filter project each vertex of P over " +
			"M and color M according to the geodesic distance from these projected points. Projection " +
			"and coloring are done on a per vertex basis.",
		filterClass: FilterClass.Sampling | FilterClass.VertexColoring,
		arity: FilterArity.FIXED,
	},
	[FP.FP_DISK_COLORING]: {
		name: "Disk Vertex Coloring",
		pythonName: "compute_scalar_by_distance_from_point_cloud_per_vertex",
		info:
			"Given a Mesh <b>M</b> and a Pointset <b>P</b>, The filter project each vertex of P over " +
			"M and color M according to the Euclidean distance from these projected points. " +
			"Projection and coloring are done on a per vertex basis.",
		filterClass: FilterClass.Sampling | FilterClass.VertexColoring,
		arity: FilterArity.FIXED,
	},
	[FP.FP_REGULAR_RECURSIVE_SAMPLING]: {
		name: "Regular Recursive Sampling",
		pythonName: "generate_sampling_regular_recursive",
		info:
			"The bounding box is recursively partitioned in a octree style, center of bbox are " +
			"considered, when the center is nearer to the surface than a given threshold it is " +
			"projected on it. It works also for building offsetted samples.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_TEXEL_SAMPLING]: {
		name: "Texel Sampling",
		pythonName: "generate_sampling_texel",
		info:
			"Create a new layer with a point sampling of the current mesh, a sample for each texel of " +
			"the mesh is generated",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
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

			case FP.FP_DISTANCE_REFERENCE:
				list.add(
					new RichMesh("MeasureMesh", 0, {
						description: "Measured Mesh/PointCloud",
						tooltip:
							"The Mesh/Pointcloud that is measured, vertex by vertex, computing distance from " +
							"the REFERENCE mesh/pointcloud.",
					}),
				);
				list.add(
					new RichMesh("RefMesh", 0, {
						description: "Reference Mesh/PointCloud",
						tooltip: "The Mesh/Pointcloud that is used as a reference, to measure distance from.",
					}),
				);
				list.add(
					new RichBool("SignedDist", true, {
						description: "Compute Signed Distance",
						tooltip:
							"If TRUE, the distance is signed; if FALSE, it will compute the distance absolute value.",
					}),
				);
				list.add(
					new RichPercentage("MaxDist", diag, 0, diag, {
						description: "Max Distance [abs]",
						tooltip:
							"Search is interrupted when nothing is found within this distance range " +
							"[+maxDistance -maxDistance].",
					}),
				);
				break;

			case FP.FP_VERTEX_RESAMPLING:
				list.add(
					new RichMesh("SourceMesh", 0, {
						description: "Source Mesh",
						tooltip: "The mesh that contains the source data that we want to transfer.",
					}),
				);
				list.add(
					new RichMesh("TargetMesh", 0, {
						description: "Target Mesh",
						tooltip: "The mesh whose vertices will receive the data from the source.",
					}),
				);
				list.add(
					new RichBool("VertexSampling", false, {
						description: "Vertex Sampling",
						tooltip:
							"if enabled for each vertex of the target mesh we search the closest vertex in the " +
							"source mesh and directly copy the found attributes, otherwise we search for the " +
							"closest point on the source surface that usually falls inside a face and " +
							"attribute are therefore interpolated",
					}),
				);
				for (const [name, defval, label, what] of [
					["GeomTransfer", false, "Transfer Geometry", "position"],
					["NormalTransfer", false, "Transfer Normal", "normal"],
					["ColorTransfer", true, "Transfer Color", "color"],
					["QualityTransfer", false, "Transfer quality", "quality"],
				] as const) {
					list.add(
						new RichBool(name, defval, {
							description: label,
							tooltip:
								`if enabled, the ${what} of each vertex of the target mesh will come from the ` +
								"corresponding closest point on the source mesh",
						}),
					);
				}
				list.add(
					new RichBool("SelectionTransfer", false, {
						description: "Transfer Selection",
						tooltip:
							"if enabled, each vertex of the target mesh will be selected if the corresponding " +
							"closest point on the source mesh falls in a selected face",
					}),
				);
				list.add(
					new RichBool("QualityDistance", false, {
						description: "Store dist. as quality",
						tooltip:
							"if enabled, we store the distance of the transferred value as in the vertex quality",
					}),
				);
				list.add(
					new RichPercentage("UpperBound", diag / 50, 0, diag, {
						description: "Max Dist Search",
						tooltip:
							"Sample points for which we do not find anything within this distance are " +
							"rejected and not considered for recovering attributes.",
					}),
				);
				list.add(
					new RichBool("onSelected", false, {
						description: "Only on selection",
						tooltip: "If checked, only transfer to selected vertices on TARGET mesh",
					}),
				);
				break;

			case FP.FP_UNIFORM_MESH_RESAMPLING:
			case FP.FP_REGULAR_RECURSIVE_SAMPLING:
				list.add(
					new RichPercentage("CellSize", diag / 50, 0, diag, {
						description: "Precision",
						tooltip:
							"Size of the cell, the default is 1/50 of the box diag. Smaller cells give better " +
							"precision at a higher computational cost. Remember that halving the cell size " +
							"means that you build a volume 8 times larger.",
					}),
				);
				list.add(
					new RichPercentage("Offset", 0, -diag / 5, diag / 5, {
						description: "Offset",
						tooltip:
							"Offset of the created surface (i.e. distance of the created surface from the " +
							"original one).<br>If offset is zero, the created surface passes on the original " +
							"mesh itself. Values greater than zero mean an external surface, and lower than " +
							"zero mean an internal surface.",
					}),
				);
				if (id === FP.FP_REGULAR_RECURSIVE_SAMPLING) break;
				list.add(
					new RichBool("mergeCloseVert", false, {
						description: "Clean Vertices",
						tooltip:
							"If true the mesh generated by MC will be cleaned by unifying vertices that are " +
							"almost coincident",
					}),
				);
				list.add(
					new RichBool("discretize", false, {
						description: "Discretize",
						tooltip:
							"If true the position of the intersected edge of the marching cube grid is not " +
							"computed by linear interpolation, but it is placed in fixed middle position. As a " +
							"consequence the resampled object will look severely aliased by a stairstep " +
							"appearance.<br>Useful only for simulating the output of 3D printing devices.",
					}),
				);
				list.add(
					new RichBool("multisample", false, {
						description: "Multi-sample",
						tooltip:
							"If true the distance field is more accurately compute by multisampling the volume " +
							"(7 sample for each voxel). Much slower but less artifacts.",
					}),
				);
				list.add(
					new RichBool("absDist", false, {
						description: "Absolute Distance",
						tooltip:
							"If true a <b>not</b> signed distance field is computed. In this case you have to " +
							"choose a not zero Offset and a double surface is built around the original " +
							"surface, inside and outside.",
					}),
				);
				break;

			case FP.FP_VORONOI_COLORING:
			case FP.FP_DISK_COLORING:
				list.add(
					new RichMesh("ColoredMesh", 0, {
						description: "To be Colored Mesh",
						tooltip: "The mesh whose surface is colored.",
					}),
				);
				list.add(
					new RichMesh("VertexMesh", 0, {
						description: "Vertex Mesh",
						tooltip: "The mesh whose vertices are used as seed points for the coloring.",
					}),
				);
				if (id === FP.FP_VORONOI_COLORING) {
					list.add(
						new RichBool("backward", false, {
							description: "BackDistance",
							tooltip:
								"If true the mesh is colored according the distance from the frontier of the " +
								"voronoi diagram induced by the pointset.",
						}),
					);
					break;
				}
				list.add(
					new RichDynamicFloat("Radius", diag / 10, 0, diag / 2, {
						description: "Radius",
						tooltip: "the radius of the spherical influence of each sample",
					}),
				);
				list.add(
					new RichBool("SampleRadius", false, {
						description: "Use sample radius",
						tooltip: "Use the radius that is stored in each sample of the current mesh.",
					}),
				);
				list.add(
					new RichBool("ApproximateGeodetic", false, {
						description: "Approximate Geodetic",
						tooltip: "Use the approximate geodetic distance instead of the euclidean one.",
					}),
				);
				break;

			case FP.FP_TEXEL_SAMPLING:
				list.add(
					new RichInt("TextureW", 512, {
						description: "Texture Width",
						tooltip:
							"A sample for each texel is generated, so the desired texture size is need, only " +
							"samples for the texels falling inside some faces are generated.",
					}),
				);
				list.add(
					new RichInt("TextureH", 512, {
						description: "Texture Height",
						tooltip:
							"A sample for each texel is generated, so the desired texture size is need, only " +
							"samples for the texels falling inside some faces are generated.",
					}),
				);
				list.add(
					new RichBool("TextureSpace", false, {
						description: "UV Space Sampling",
						tooltip:
							"The generated texel samples have their UV coords as point positions. The resulting " +
							"point set is has a square domain, the texels/points, even if on a flat domain " +
							"retain the original vertex normal to help a better perception of the original " +
							"provenience.",
					}),
				);
				list.add(
					new RichBool("RecoverColor", m !== undefined && m.cm.textures.length > 0, {
						description: "RecoverColor",
						tooltip: "The generated point cloud has the current texture color",
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

			case FP.FP_DISTANCE_REFERENCE: {
				const measured = doc.requireMesh(params.getMeshId("MeasureMesh"));
				const reference = doc.requireMesh(params.getMeshId("RefMesh"));
				if (measured === reference) {
					throw new MLException("Cannot compute, it is the same mesh");
				}
				measured.updateDataMask(MeshElement.MM_VERTQUALITY);
				const useSigned = params.getBool("SignedDist");
				const maxDist = params.getAbsPerc("MaxDist");
				const lookup = referenceLookup(reference, maxDist);

				let min = Number.POSITIVE_INFINITY;
				let max = Number.NEGATIVE_INFINITY;
				let sum = 0;
				let sumSq = 0;
				let n = 0;
				const cm = measured.cm;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					const d = lookup(cm.vx(v), cm.vy(v), cm.vz(v), useSigned);
					cm.vertQuality[v] = d;
					const a = Math.abs(d);
					min = Math.min(min, a);
					max = Math.max(max, a);
					sum += a;
					sumSq += a * a;
					n++;
				}
				const mean = n === 0 ? 0 : sum / n;
				const rms = n === 0 ? 0 : Math.sqrt(sumSq / n);
				doc.Log.log(
					`Distance from Reference Mesh: sampled ${n} vertices on "${measured.label()}", ` +
						`searched closest on "${reference.label()}"`,
				);
				doc.Log.log(`     min: ${min}   max: ${max}   mean: ${mean}   RMS: ${rms}`);
				post.mask = MeshElement.MM_VERTQUALITY;
				return { min_distance: n === 0 ? 0 : min, max_distance: n === 0 ? 0 : max, mean, RMS: rms };
			}

			case FP.FP_VERTEX_RESAMPLING:
				return this.transferAttributes(params, doc, post);

			case FP.FP_UNIFORM_MESH_RESAMPLING:
			case FP.FP_REGULAR_RECURSIVE_SAMPLING: {
				if (source.cm.fn === 0) {
					throw new MLException(
						`${this.filterName(id)} requires a mesh with faces, it does not work on point clouds`,
					);
				}
				const cellSize = params.getAbsPerc("CellSize");
				if (!(cellSize > 0)) {
					throw new MLException(`The cell size must be positive, got ${cellSize}`);
				}
				const offset = params.getAbsPerc("Offset");
				// Upstream grows the box by a tenth of its diagonal plus the
				// offset, so an outward offset still has grid to live in.
				UpdateBounding.box(source.cm);
				const margin = (source.cm.bbox.diagonal || 1) / 10 + Math.abs(offset);
				const box = paddedBox(source.cm, margin);
				const grid = gridFor(box.min, box.max, cellSize);
				const LIMIT = 40_000_000;
				if (grid.total > LIMIT) {
					throw new MLException(
						`That cell size needs ${grid.total} samples, over the ${LIMIT} limit; use a larger one`,
					);
				}

				if (id === FP.FP_REGULAR_RECURSIVE_SAMPLING) {
					const cloud = recursiveOffsetSamples(source.cm, grid, offset, cellSize);
					return this.addLayer(doc, source, cloud, "Recursive Samples");
				}

				const absDist = params.getBool("absDist");
				if (absDist && offset === 0) {
					throw new MLException(
						"An unsigned distance field needs a non-zero offset: at offset zero the surface " +
							"is the field's minimum rather than a level set, and nothing is extracted.",
					);
				}
				const field = distanceField(source.cm, grid, {
					signed: !absDist,
					progress: (percent) => _cb(percent, "Resampling"),
				});
				const out = extractLevelSet(field, grid, offset, params.getBool("discretize"));
				if (params.getBool("mergeCloseVert")) {
					UpdateBounding.box(out);
					const threshold = (out.bbox.diagonal || 1) / 10000;
					const merged = Clean.mergeCloseVertex(out, threshold);
					doc.Log.log(`Merged ${merged} vertices closer than ${threshold}`);
				}
				doc.Log.log(
					`Resampled using a volume of ${grid.counts[0]} x ${grid.counts[1]} x ${grid.counts[2]}`,
				);
				const target = doc.addNewMesh("", "Offset mesh", true, out);
				target.updateBoxAndNormals();
				return { new_mesh_id: target.id(), vertex_number: out.vn, face_number: out.fn };
			}

			case FP.FP_VORONOI_COLORING:
			case FP.FP_DISK_COLORING: {
				const coloured = doc.requireMesh(params.getMeshId("ColoredMesh"));
				const seedsMesh = doc.requireMesh(params.getMeshId("VertexMesh"));
				if (coloured === seedsMesh) {
					throw new MLException("Cannot compute, it is the same mesh");
				}
				coloured.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
				post.mask = MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY;
				return id === FP.FP_VORONOI_COLORING
					? voronoiColouring(doc, coloured, seedsMesh, params.getBool("backward"))
					: diskColouring(doc, coloured, seedsMesh, params);
			}

			case FP.FP_TEXEL_SAMPLING: {
				const wt = source.cm.wedgeTexCoord;
				if (wt === null) {
					throw new MLException("Texel Sampling requires a mesh with per-wedge UV parametrization");
				}
				const cloud = texelSampling(
					source,
					params.getInt("TextureW"),
					params.getInt("TextureH"),
					params.getBool("TextureSpace"),
					params.getBool("RecoverColor"),
				);
				return this.addLayer(doc, source, cloud, "Texel samples");
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	/**
	 * `Vertex Attribute Transfer`: copy attributes from one mesh to another by
	 * closest point.
	 *
	 * The default is the closest *point on the surface*, not the closest vertex,
	 * and the difference matters: a coarse source mesh has few vertices to
	 * offer, so snapping to them quantises a smooth colour field into flat
	 * patches. Interpolating inside the face it lands in avoids that. The
	 * vertex mode is still there for when the source is a point cloud, or when
	 * exact source values rather than blends are wanted.
	 */
	private transferAttributes(
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
	): FilterOutput {
		const src = doc.requireMesh(params.getMeshId("SourceMesh"));
		const trg = doc.requireMesh(params.getMeshId("TargetMesh"));
		if (src === trg) throw new MLException("Cannot compute, it is the same mesh");

		const wantColor = params.getBool("ColorTransfer");
		const wantGeom = params.getBool("GeomTransfer");
		const wantNormal = params.getBool("NormalTransfer");
		const wantQuality = params.getBool("QualityTransfer");
		const wantSelection = params.getBool("SelectionTransfer");
		const storeDistance = params.getBool("QualityDistance");
		if (
			!wantColor &&
			!wantGeom &&
			!wantNormal &&
			!wantQuality &&
			!wantSelection &&
			!storeDistance
		) {
			throw new MLException("You have to choose at least one attribute to be sampled");
		}

		const onlySelected = params.getBool("onSelected");
		const upperBound = params.getAbsPerc("UpperBound");
		// A point cloud has no surface to find a closest point on, so it forces
		// the vertex mode whatever the parameter says.
		const byVertex = params.getBool("VertexSampling") || src.cm.fn === 0;

		let mask = 0;
		if (wantColor) mask |= MeshElement.MM_VERTCOLOR;
		if (wantQuality || storeDistance) mask |= MeshElement.MM_VERTQUALITY;
		if (mask !== 0) trg.updateDataMask(mask);

		const sc = src.cm;
		const tc = trg.cm;
		UpdateNormal.perVertexNormalizedPerFaceNormalized(sc);
		const tree = byVertex ? liveVertexTree(sc) : null;
		const lookup = byVertex
			? null
			: new SurfaceLookup(sc, upperBound > 0 ? upperBound : (sc.bbox.diagonal || 1) * 4);

		let transferred = 0;
		let rejected = 0;
		for (let v = 0; v < tc.vertSize; v++) {
			if (tc.isVertD(v)) continue;
			if (onlySelected && !tc.isVertS(v)) continue;
			const x = tc.vx(v);
			const y = tc.vy(v);
			const z = tc.vz(v);

			// Both branches produce the same four things: a point, a normal, a
			// colour and a quality, either read from one vertex or interpolated
			// across a face.
			let p: number[];
			let n: number[];
			let colour: number;
			let quality: number;
			let selected: boolean;
			if (tree !== null) {
				const w = tree.closest(x, y, z);
				if (w === null) {
					rejected++;
					continue;
				}
				p = [sc.vx(w), sc.vy(w), sc.vz(w)];
				n = [sc.vertNormal[3 * w], sc.vertNormal[3 * w + 1], sc.vertNormal[3 * w + 2]];
				colour = sc.vertColor[w];
				quality = sc.vertQuality[w];
				selected = sc.isVertS(w);
			} else {
				const hit = (lookup as SurfaceLookup).closest(x, y, z);
				if (hit === null) {
					rejected++;
					continue;
				}
				p = [0, 0, 0];
				n = [0, 0, 0];
				quality = 0;
				let r = 0;
				let g = 0;
				let b = 0;
				let a = 0;
				for (let k = 0; k < 3; k++) {
					const w = sc.fv(hit.face, k);
					const t = hit.bary[k];
					p[0] += sc.vx(w) * t;
					p[1] += sc.vy(w) * t;
					p[2] += sc.vz(w) * t;
					for (let c = 0; c < 3; c++) n[c] += sc.vertNormal[3 * w + c] * t;
					quality += sc.vertQuality[w] * t;
					const cc = sc.vertColor[w];
					r += red(cc) * t;
					g += green(cc) * t;
					b += blue(cc) * t;
					a += alpha(cc) * t;
				}
				colour = rgba(r, g, b, a);
				selected = sc.isFaceS(hit.face);
			}

			const distance = Math.hypot(p[0] - x, p[1] - y, p[2] - z);
			if (upperBound > 0 && distance > upperBound) {
				rejected++;
				continue;
			}

			if (wantColor) tc.vertColor[v] = colour;
			if (wantQuality) tc.vertQuality[v] = quality;
			if (storeDistance) tc.vertQuality[v] = distance;
			if (wantNormal) {
				const len = Math.hypot(n[0], n[1], n[2]) || 1;
				for (let c = 0; c < 3; c++) tc.vertNormal[3 * v + c] = n[c] / len;
			}
			if (wantSelection) {
				if (selected) tc.vertFlags[v] |= VertexFlag.SELECTED;
				else tc.vertFlags[v] &= ~VertexFlag.SELECTED;
			}
			// Geometry last: moving the vertex first would change the distance
			// every other attribute above was chosen by.
			if (wantGeom) tc.setVert(v, p[0], p[1], p[2]);
			transferred++;
		}

		if (wantGeom) trg.updateBoxAndNormals();
		post.mask = mask | (wantGeom ? MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE : 0);
		doc.Log.log(
			`Transferred attributes to ${transferred} vertices; ${rejected} were beyond the search distance`,
		);
		return { vertex_number: transferred, rejected };
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

/**
 * A k-d tree over the *live* vertices only, with the mapping back.
 *
 * `new KdTree(cm.vertCoord, cm.vertSize)` would index the deleted slots too,
 * and a deleted vertex keeps its old coordinates — so it can win a nearest
 * query and hand back an index nothing else in the mesh refers to. Copying
 * the survivors out costs one pass and removes the whole class of error.
 */
function liveVertexTree(cm: CMeshO): {
	closest: (x: number, y: number, z: number) => number | null;
} {
	const map: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) map.push(v);
	const coords = new Float64Array(3 * map.length);
	map.forEach((v, i) => {
		coords[3 * i] = cm.vx(v);
		coords[3 * i + 1] = cm.vy(v);
		coords[3 * i + 2] = cm.vz(v);
	});
	const tree = new KdTree(coords, map.length);
	return {
		closest: (x, y, z) => {
			if (map.length === 0) return null;
			const hit = tree.nearestToPoint(x, y, z);
			return hit < 0 ? null : map[hit];
		},
	};
}

/**
 * A closest-point query against `reference`, working for a mesh or a cloud.
 *
 * A point cloud has no surface to be inside of, so the signed variant falls
 * back to the unsigned distance there rather than inventing a sign from
 * whatever normals the points happen to carry.
 */
function referenceLookup(
	reference: MeshModel,
	maxDist: number,
): (x: number, y: number, z: number, signed: boolean) => number {
	const cm = reference.cm;
	const reach = maxDist > 0 ? maxDist : (cm.bbox.diagonal || 1) * 4;
	if (cm.fn === 0) {
		const tree = liveVertexTree(cm);
		return (x, y, z) => {
			const hit = tree.closest(x, y, z);
			return hit === null ? reach : Math.hypot(cm.vx(hit) - x, cm.vy(hit) - y, cm.vz(hit) - z);
		};
	}
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const lookup = new SurfaceLookup(cm, reach);
	return (x, y, z, signed) => {
		const hit = lookup.closest(x, y, z);
		if (hit === null) return reach;
		const p = [0, 0, 0];
		const n = [0, 0, 0];
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(hit.face, k);
			p[0] += cm.vx(v) * hit.bary[k];
			p[1] += cm.vy(v) * hit.bary[k];
			p[2] += cm.vz(v) * hit.bary[k];
			for (let a = 0; a < 3; a++) n[a] += cm.vertNormal[3 * v + a] * hit.bary[k];
		}
		const d = [x - p[0], y - p[1], z - p[2]];
		const dist = Math.hypot(d[0], d[1], d[2]);
		if (!signed) return dist;
		return d[0] * n[0] + d[1] * n[1] + d[2] * n[2] < 0 ? -dist : dist;
	};
}

/**
 * `Voronoi Vertex Coloring`: colour by geodesic distance to the nearest seed.
 *
 * Each seed point is snapped to the nearest vertex of the coloured mesh, then
 * a Dijkstra pass gives every vertex its distance along the surface. Geodesic
 * rather than Euclidean is the whole point — two sides of a thin wall are far
 * apart on the surface even though they are close in space, and only the
 * geodesic reading tells them apart.
 *
 * `backward` reports the distance to the *frontier* between regions instead,
 * which is what draws the Voronoi diagram rather than filling its cells.
 */
function voronoiColouring(
	doc: MeshDocument,
	coloured: MeshModel,
	seedsMesh: MeshModel,
	backward: boolean,
): FilterOutput {
	const cm = coloured.cm;
	Clean.removeUnreferencedVertex(cm);
	Allocator.compactEveryVector(cm);
	if (cm.vn === 0) throw new MLException("The mesh to be coloured has no vertices.");

	const seeds = new Set<number>();
	const tree = liveVertexTree(cm);
	const sm = seedsMesh.cm;
	for (let v = 0; v < sm.vertSize; v++) {
		if (sm.isVertD(v)) continue;
		const hit = tree.closest(sm.vx(v), sm.vy(v), sm.vz(v));
		if (hit !== null) seeds.add(hit);
	}
	if (seeds.size === 0) throw new MLException("The point set has no vertices to seed from.");
	const seedList = [...seeds];
	doc.Log.log(`Converted ${sm.vn} points into ${seedList.length} vertices`);

	// Nearest seed and its distance, one Dijkstra per seed. Doing them
	// separately is what makes the region known, not just the distance.
	const region = new Int32Array(cm.vertSize).fill(-1);
	const best = new Float64Array(cm.vertSize).fill(Number.POSITIVE_INFINITY);
	const second = new Float64Array(cm.vertSize).fill(Number.POSITIVE_INFINITY);
	seedList.forEach((s, i) => {
		const d = dijkstraGeodesic(cm, [s]);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			if (d[v] < best[v]) {
				second[v] = best[v];
				best[v] = d[v];
				region[v] = i;
			} else if (d[v] < second[v]) {
				second[v] = d[v];
			}
		}
	});

	// The frontier reading: how much closer this vertex is to its own seed
	// than to the next one. Zero exactly on the boundary between two regions.
	const value = (v: number) =>
		backward ? (second[v] === Number.POSITIVE_INFINITY ? 0 : second[v] - best[v]) : best[v];
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v) || !Number.isFinite(value(v))) continue;
		min = Math.min(min, value(v));
		max = Math.max(max, value(v));
	}
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const q = Number.isFinite(value(v)) ? value(v) : max;
		cm.vertQuality[v] = q;
		cm.vertColor[v] = colorRamp(min, max, q);
	}
	// The seeds themselves in red, so they are findable in the result.
	for (const s of seedList) cm.vertColor[s] = rgba(255, 0, 0);
	doc.Log.log(`Voronoi coloring over ${seedList.length} seeds, range [${min}, ${max}]`);
	return { seed_number: seedList.length, min_value: min, max_value: max };
}

/**
 * `Disk Vertex Coloring`: paint a disk of the given radius around each seed.
 *
 * Everything outside every disk stays light grey, which is what makes the
 * filter a coverage check — the grey is exactly what no sample reached.
 */
function diskColouring(
	doc: MeshDocument,
	coloured: MeshModel,
	seedsMesh: MeshModel,
	params: RichParameterList,
): FilterOutput {
	const cm = coloured.cm;
	const LIGHT_GREY = rgba(192, 192, 192);
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		cm.vertColor[v] = LIGHT_GREY;
		cm.vertQuality[v] = Number.MAX_VALUE;
	}

	const approximateGeodetic = params.getBool("ApproximateGeodetic");
	const useSampleRadius = params.getBool("SampleRadius");
	const fixedRadius = params.getDynamicFloat("Radius");
	if (approximateGeodetic) UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const tree = new KdTree(cm.vertCoord, cm.vertSize);
	const sm = seedsMesh.cm;
	if (useSampleRadius && sm.vertRadius === null) {
		throw new MLException(
			'"Use sample radius" needs a per-vertex radius on the point set, which it does not carry.',
		);
	}

	let painted = 0;
	for (let s = 0; s < sm.vertSize; s++) {
		if (sm.isVertD(s)) continue;
		const radius = useSampleRadius && sm.vertRadius !== null ? sm.vertRadius[s] : fixedRadius;
		if (!(radius > 0)) continue;
		const p = [sm.vx(s), sm.vy(s), sm.vz(s)];
		for (const near of tree.withinRadius(p[0], p[1], p[2], radius)) {
			const v = near.index;
			// The radius query indexes raw slots, so a deleted vertex can come
			// back; it keeps its coordinates but nothing refers to it.
			if (cm.isVertD(v)) continue;
			const dist = approximateGeodetic
				? approximateGeodesic(sm, s, cm, v)
				: Math.hypot(cm.vx(v) - p[0], cm.vy(v) - p[1], cm.vz(v) - p[2]);
			if (dist >= radius || cm.vertQuality[v] <= dist) continue;
			cm.vertQuality[v] = dist;
			cm.vertColor[v] = lerpColor(rgba(255, 255, 255), rgba(255, 0, 0), dist / radius);
			painted++;
		}
	}
	doc.Log.log(`Disk coloring touched ${painted} vertices`);
	return { vertex_number: painted };
}

/**
 * VCG's `ApproximateGeodesicDistance`: the chord corrected for how much the
 * two normals disagree.
 *
 * Cheaper than a real geodesic and good enough over a disk radius, which is
 * the only range this is ever asked for.
 */
function approximateGeodesic(am: CMeshO, a: number, bm: CMeshO, b: number): number {
	const d = [bm.vx(b) - am.vx(a), bm.vy(b) - am.vy(a), bm.vz(b) - am.vz(a)];
	const len = Math.hypot(d[0], d[1], d[2]);
	if (len === 0) return 0;
	const na = [am.vertNormal[3 * a], am.vertNormal[3 * a + 1], am.vertNormal[3 * a + 2]];
	const nb = [bm.vertNormal[3 * b], bm.vertNormal[3 * b + 1], bm.vertNormal[3 * b + 2]];
	const ca = (na[0] * d[0] + na[1] * d[1] + na[2] * d[2]) / len;
	const cb = (nb[0] * d[0] + nb[1] * d[1] + nb[2] * d[2]) / len;
	const delta = Math.abs(ca - cb);
	// The limit as the normals converge is the chord itself; asin/x → 1.
	return delta < 1e-12 ? len : (len * Math.asin(Math.min(1, delta))) / delta;
}

/**
 * `Regular Recursive Sampling`: the grid points near the offset surface,
 * projected onto it.
 *
 * Upstream subdivides an octree and only descends where the cell might touch
 * the surface. Sampling the same field on the flat grid the caller already
 * built lands on the same points and reuses the machinery the resampler
 * needs anyway; what it gives up is the early exit, so a very fine cell size
 * costs more here than upstream.
 */
function recursiveOffsetSamples(cm: CMeshO, grid: Grid, offset: number, cellSize: number): CMeshO {
	const field = distanceField(cm, grid);
	const points: number[] = [];
	const { counts, coord, index } = grid;
	for (let k = 0; k < counts[2]; k++) {
		for (let j = 0; j < counts[1]; j++) {
			for (let i = 0; i < counts[0]; i++) {
				const d = field[index(i, j, k)] - offset;
				// Within half a cell of the level set: any nearer and the
				// projection would be a long jump rather than a correction.
				if (Math.abs(d) > cellSize / 2) continue;
				points.push(coord(0, i), coord(1, j), coord(2, k));
			}
		}
	}

	const out = new CMeshO();
	if (points.length === 0) return out;
	const first = Allocator.addVertices(out, points.length / 3);
	// Project each survivor onto the offset surface along the field's gradient,
	// which for a distance field is the direction to the closest point.
	const reach = (cm.bbox.diagonal || 1) * 4;
	const lookup = new SurfaceLookup(cm, reach);
	for (let s = 0; s < points.length / 3; s++) {
		const p = [points[3 * s], points[3 * s + 1], points[3 * s + 2]];
		const hit = lookup.closest(p[0], p[1], p[2]);
		if (hit === null) {
			out.setVert(first + s, p[0], p[1], p[2]);
			continue;
		}
		const q = [0, 0, 0];
		for (let c = 0; c < 3; c++) {
			const v = cm.fv(hit.face, c);
			q[0] += cm.vx(v) * hit.bary[c];
			q[1] += cm.vy(v) * hit.bary[c];
			q[2] += cm.vz(v) * hit.bary[c];
		}
		if (offset === 0) {
			out.setVert(first + s, q[0], q[1], q[2]);
			continue;
		}
		const d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
		const len = Math.hypot(d[0], d[1], d[2]);
		if (len === 0) {
			out.setVert(first + s, q[0], q[1], q[2]);
			continue;
		}
		out.setVert(
			first + s,
			q[0] + (d[0] / len) * offset,
			q[1] + (d[1] / len) * offset,
			q[2] + (d[2] / len) * offset,
		);
	}
	return out;
}

/**
 * `Texel Sampling`: one sample per texel that any face covers.
 *
 * Rasterises each triangle in UV space and emits the surface point under every
 * texel centre it contains. That is what makes it the right sampling for
 * baking: the sample density follows the texture, so every texel that will be
 * written gets exactly one value and none is missed.
 */
function texelSampling(
	source: MeshModel,
	width: number,
	height: number,
	inTextureSpace: boolean,
	recoverColor: boolean,
): CMeshO {
	if (width <= 0 || height <= 0) {
		throw new MLException(`The texture size must be positive, got ${width}x${height}`);
	}
	const cm = source.cm;
	const wt = cm.wedgeTexCoord as Float64Array;
	const wti = cm.wedgeTexIndex;
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);

	const images = recoverColor
		? cm.textures.map((name) => {
				const bytes = source.textures.get(name);
				if (bytes === undefined || !isPng(bytes)) return null;
				return readPng(bytes);
			})
		: [];

	const coords: number[] = [];
	const normals: number[] = [];
	const colours: number[] = [];
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const u = [wt[6 * f], wt[6 * f + 2], wt[6 * f + 4]];
		const v = [wt[6 * f + 1], wt[6 * f + 3], wt[6 * f + 5]];
		const x0 = Math.max(0, Math.floor(Math.min(...u) * width));
		const x1 = Math.min(width - 1, Math.ceil(Math.max(...u) * width));
		const y0 = Math.max(0, Math.floor(Math.min(...v) * height));
		const y1 = Math.min(height - 1, Math.ceil(Math.max(...v) * height));
		const area = (u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]);
		// A face with no area in UV space covers no texel — it is either
		// degenerate or unparametrised, and either way has nothing to sample.
		if (area === 0) continue;

		for (let y = y0; y <= y1; y++) {
			for (let x = x0; x <= x1; x++) {
				const su = (x + 0.5) / width;
				const sv = (y + 0.5) / height;
				const b1 = ((su - u[0]) * (v[2] - v[0]) - (sv - v[0]) * (u[2] - u[0])) / area;
				const b2 = ((u[1] - u[0]) * (sv - v[0]) - (v[1] - v[0]) * (su - u[0])) / area;
				const b0 = 1 - b1 - b2;
				if (b0 < 0 || b1 < 0 || b2 < 0) continue;
				const bary = [b0, b1, b2];

				if (inTextureSpace) {
					// The samples become a flat sheet in UV space; the normals
					// still come from the surface, which is the only thing left
					// showing where each texel came from.
					coords.push(su, sv, 0);
				} else {
					const p = [0, 0, 0];
					for (let k = 0; k < 3; k++) {
						const w = cm.fv(f, k);
						p[0] += cm.vx(w) * bary[k];
						p[1] += cm.vy(w) * bary[k];
						p[2] += cm.vz(w) * bary[k];
					}
					coords.push(p[0], p[1], p[2]);
				}
				const n = [0, 0, 0];
				for (let k = 0; k < 3; k++) {
					const w = cm.fv(f, k);
					for (let a = 0; a < 3; a++) n[a] += cm.vertNormal[3 * w + a] * bary[k];
				}
				const len = Math.hypot(n[0], n[1], n[2]) || 1;
				normals.push(n[0] / len, n[1] / len, n[2] / len);

				const ti = wti === null ? 0 : wti[3 * f];
				const image = recoverColor && ti >= 0 && ti < images.length ? images[ti] : null;
				colours.push(
					image === null
						? rgba(255, 255, 255)
						: image.pixel(
								Math.min(image.width - 1, Math.floor(su * image.width)),
								Math.min(
									image.height - 1,
									Math.max(0, image.height - 1 - Math.floor(sv * image.height)),
								),
							),
				);
			}
		}
	}

	const out = new CMeshO();
	const count = colours.length;
	if (count === 0) return out;
	out.enableChannels(MeshElement.MM_VERTCOLOR);
	const first = Allocator.addVertices(out, count);
	for (let s = 0; s < count; s++) {
		out.setVert(first + s, coords[3 * s], coords[3 * s + 1], coords[3 * s + 2]);
		for (let a = 0; a < 3; a++) out.vertNormal[3 * (first + s) + a] = normals[3 * s + a];
		out.vertColor[first + s] = colours[s];
	}
	return out;
}
