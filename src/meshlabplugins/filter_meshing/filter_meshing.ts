/**
 * `filter_meshing` — orientation, hole closing, decimation and transforms.
 *
 * The rest of upstream's 37 filters (isotropic remeshing, subdivision, the
 * quad and texture families) arrive with the later tiers.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichDirection,
	RichEnum,
	RichFloat,
	RichInt,
	RichMatrix44,
	RichPercentage,
	RichPosition,
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
import { MLException, MLNotImplementedException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { CreaseCut } from "../../vcg/complex/crease_cut.ts";
import { UpdateCurvature as Curvature } from "../../vcg/complex/curvature.ts";
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import { Hole } from "../../vcg/complex/hole.ts";
import { Inertia } from "../../vcg/complex/inertia.ts";
import { IsotropicRemeshing } from "../../vcg/complex/isotropic_remeshing.ts";
import {
	defaultQuadricParameters,
	quadricSimplification,
} from "../../vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
import { estimateNormals } from "../../vcg/complex/pointcloud_normal.ts";
import { Refine } from "../../vcg/complex/refine.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdatePosition } from "../../vcg/complex/update/position.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { covariance, fitPlaneToPointSet, symmetricEigen3 } from "../../vcg/math/eigen3.ts";
import { Matrix44Ops } from "../../vcg/math/matrix44.ts";
import { colorRamp } from "../../vcg/space/color4.ts";

export const FP = {
	FP_REORIENT: 0,
	FP_INVERT_FACES: 1,
	FP_CLOSE_HOLES: 2,
	FP_QUADRIC_SIMPLIFICATION: 3,
	FP_SCALE: 4,
	FP_CENTER: 5,
	FP_ROTATE: 6,
	FP_FREEZE_TRANSFORM: 7,
	FP_RESET_TRANSFORM: 8,
	FP_NORMAL_EXTRAPOLATION: 9,
	FP_NORMAL_SMOOTH_POINTCLOUD: 10,
	FP_LOOP_SS: 11,
	FP_BUTTERFLY_SS: 12,
	FP_MIDPOINT: 13,
	FP_EXPLICIT_ISOTROPIC_REMESHING: 14,
	FP_CLUSTERING: 15,
	FP_COMPUTE_PRINC_CURV_DIR: 16,
	FP_INVERT_TRANSFORM: 17,
	FP_SET_TRANSFORM_MATRIX: 18,
	FP_SET_TRANSFORM_PARAMS: 19,
	FP_PRINCIPAL_AXIS: 20,
	FP_FLIP_AND_SWAP: 21,
	FP_ROTATE_FIT: 22,
	FP_FAUX_CREASE: 23,
	FP_MAKE_PURE_TRI: 24,
} as const;

const GEOMETRY_AND_TOPOLOGY = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly requirements: number;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_REORIENT]: {
		name: "Re-Orient all faces coherently",
		pythonName: "meshing_re_orient_faces_coherently",
		info: "Re-orient in a consistent way all the faces of the mesh.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_INVERT_FACES]: {
		name: "Invert Faces Orientation",
		pythonName: "meshing_invert_face_orientation",
		info:
			"Invert faces orientation, flipping the normals of the mesh. If requested, it tries to " +
			"guess the right orientation; mainly it decide to flip all the faces if the mesh is " +
			"'more' inside than outside.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_CLOSE_HOLES]: {
		name: "Close Holes",
		pythonName: "meshing_close_holes",
		info:
			"Close holes smaller than a given threshold. The hole is filled by ear cutting, " +
			"choosing at each step the triangle that is best shaped and folds least sharply away " +
			"from the surrounding surface.",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_QUADRIC_SIMPLIFICATION]: {
		name: "Simplification: Quadric Edge Collapse Decimation",
		pythonName: "meshing_decimation_quadric_edge_collapse",
		info:
			"Simplify a mesh using a Quadric based Edge Collapse Strategy; better than clustering " +
			"but slower.",
		filterClass: FilterClass.Remeshing,
		// Deliberately no adjacency: the decimator maintains its own incidence
		// through the collapses, and FF would be stale after the first one.
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_LOOP_SS]: {
		name: "Subdivision Surfaces: Loop",
		pythonName: "meshing_surface_subdivision_loop",
		info:
			"Apply Loop's Subdivision Surface algorithm. It is an approximant subdivision method and " +
			"it works for every triangle and has rules for extraordinary vertices.<br>",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_BUTTERFLY_SS]: {
		name: "Subdivision Surfaces: Butterfly Subdivision",
		pythonName: "meshing_surface_subdivision_butterfly",
		info:
			"Apply Butterfly Subdivision Surface algorithm. It is an interpolated method, defined on " +
			"arbitrary triangular meshes. The scheme is known to be C1 but not C2 on regular meshes<br>",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_MIDPOINT]: {
		name: "Subdivision Surfaces: Midpoint",
		pythonName: "meshing_surface_subdivision_midpoint",
		info:
			"Apply a plain subdivision scheme where every edge is split on its midpoint. Useful to " +
			"uniformly refine a mesh substituting each triangle with four smaller triangles.",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_EXPLICIT_ISOTROPIC_REMESHING]: {
		name: "Remeshing: Isotropic Explicit Remeshing",
		pythonName: "meshing_isotropic_explicit_remeshing",
		info:
			"Perform a explicit remeshing of a triangular mesh, by repeatedly applying edge flip, " +
			"collapse, relax and refine to improve aspect ratio (triangle quality) and topological " +
			"regularity.",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_CLUSTERING]: {
		name: "Simplification: Clustering Decimation",
		pythonName: "meshing_decimation_clustering",
		info:
			"Collapse vertices by creating a three dimensional grid enveloping the mesh and " +
			"discretizes them based on the cells of this grid",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_COMPUTE_PRINC_CURV_DIR]: {
		name: "Compute curvature principal directions",
		pythonName: "compute_curvature_principal_directions_per_vertex",
		info:
			"Compute the principal directions of curvature with several algorithms.<br>" +
			'<b>Taubin</b>: <i>"Estimating the tensor of curvature of a surface from a polyhedral ' +
			'approximation"</i>, Gabriel Taubin, ICCV 1995<br><b>Quadric Fitting</b>: fits a quadric ' +
			"patch to the one-ring of every vertex and reads the curvature off it.",
		filterClass: FilterClass.Normal | FilterClass.VertexColoring,
		requirements:
			MeshElement.MM_VERTCURVDIR |
			MeshElement.MM_VERTQUALITY |
			MeshElement.MM_VERTCOLOR |
			MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_SCALE]: {
		name: "Transform: Scale, Normalize",
		pythonName: "compute_matrix_from_scaling_or_normalization",
		info: "Generate a matrix transformation that scale the mesh. The mesh can be also automatically scaled to a unit side box.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_CENTER]: {
		name: "Transform: Translate, Center, set Origin",
		pythonName: "compute_matrix_from_translation",
		info: "Generate a matrix transformation that translate the mesh. The mesh can be translated around one of its bounding box centers or a given point.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_ROTATE]: {
		name: "Transform: Rotate",
		pythonName: "compute_matrix_from_rotation",
		info: "Generate a matrix transformation that rotates the mesh. The mesh can be rotated around one of the axis or a given axis and w.r.t. to the origin or the baricenter, or a given point.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_FREEZE_TRANSFORM]: {
		name: "Matrix: Freeze Current Matrix",
		pythonName: "apply_matrix_freeze",
		info: "Freeze the current transformation matrix into the coordinates of the vertices of the mesh (and set this matrix to the identity). In other words it applies in a definitive way the current matrix to the vertex coordinates.",
		filterClass: FilterClass.Layer | FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_RESET_TRANSFORM]: {
		name: "Matrix: Reset Current Matrix",
		pythonName: "set_matrix_identity",
		info: "Set the current transformation matrix to the Identity.",
		filterClass: FilterClass.Layer | FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_NORMAL_EXTRAPOLATION]: {
		name: "Compute normals for point sets",
		pythonName: "compute_normal_for_point_clouds",
		info:
			"Compute the normals of the vertices of a mesh without exploiting the triangle " +
			"connectivity, useful for dataset with no faces.",
		filterClass: FilterClass.Normal | FilterClass.PointSet,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_NORMAL_SMOOTH_POINTCLOUD]: {
		name: "Smooth normals on point sets",
		pythonName: "apply_normal_point_cloud_smoothing",
		info:
			"Smooth the normals of the vertices of a mesh without exploiting the triangle " +
			"connectivity, useful for dataset with no faces.",
		filterClass: FilterClass.Normal | FilterClass.PointSet,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_INVERT_TRANSFORM]: {
		name: "Matrix: Invert Current Matrix",
		pythonName: "apply_matrix_inverse",
		info:
			"Invert the current transformation matrix. The current transformation is reversed, " +
			"becoming its own inverse.",
		filterClass: FilterClass.Layer | FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_SET_TRANSFORM_MATRIX]: {
		name: "Matrix: Set/Copy Transformation",
		pythonName: "set_matrix",
		info: "Set the current transformation matrix by filling it, or copying from another layer.",
		filterClass: FilterClass.Layer | FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_SET_TRANSFORM_PARAMS]: {
		name: "Matrix: Set from translation/rotation/scale",
		pythonName: "compute_matrix_from_translation_rotation_scale",
		info: "Set the current transformation matrix starting from parameters: translation, rotation and scale.",
		filterClass: FilterClass.Layer | FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_PRINCIPAL_AXIS]: {
		name: "Transform: Align to Principal Axis",
		pythonName: "compute_matrix_by_principal_axis",
		info:
			"Generate a matrix transformation that aligns the mesh to its principal axes of inertia." +
			"<br>If the mesh has no faces, the vertex set is used instead.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_FLIP_AND_SWAP]: {
		name: "Transform: Flip and/or swap axis",
		pythonName: "apply_matrix_flip_or_swap_axis",
		info:
			"Generate a matrix transformation that flips each one of the axis or swaps a couple of " +
			"axis. The listed transformations are applied in that order. This kind of transformation " +
			"cannot be applied to set of Raster!",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_ROTATE_FIT]: {
		name: "Transform: Rotate to Fit to a plane",
		pythonName: "compute_matrix_by_fitting_to_plane",
		info:
			"Generate a matrix transformation that rotates the mesh. The mesh is rotated according " +
			"to the fitting plane of the selected vertices.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_FAUX_CREASE]: {
		name: "Select Crease Edges",
		pythonName: "compute_selection_crease_per_edge",
		info:
			"It select the crease edges of a mesh according to edge dihedral angle.<br>Angle between " +
			"face normal is considered signed according to convexity/concavity. Convex angles are " +
			"positive and concave are negative.",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_MAKE_PURE_TRI]: {
		name: "Turn into a Pure-Triangular mesh",
		pythonName: "meshing_poly_to_tri",
		info: "Convert into a tri-mesh by splitting any polygonal face.",
		filterClass: FilterClass.Remeshing | FilterClass.Polygonal,
		requirements: MeshElement.MM_NONE,
	},
};

export class FilterMeshing extends FilterPlugin {
	pluginName(): string {
		return "FilterMeshing";
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
	override getRequirements(id: ActionIDType): number {
		return this.spec(id).requirements;
	}
	override postCondition(_id: ActionIDType): number {
		return GEOMETRY_AND_TOPOLOGY;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		/** The two flags every matrix filter shares. */
		const transformOptions = () => {
			list.add(
				new RichBool("allLayers", false, {
					description: "Apply to all visible Layers",
					tooltip: "If selected the filter will be applied to all visible mesh layers",
				}),
			);
			list.add(
				new RichBool("Freeze", true, {
					description: "Freeze Matrix",
					tooltip:
						"The transformation is explicitly applied, and the vertex coordinates are actually changed",
				}),
			);
		};
		switch (id) {
			case FP.FP_INVERT_FACES:
				list.add(
					new RichBool("forceFlip", true, {
						description: "Force Flip",
						tooltip:
							"If selected, the normals will always be flipped; otherwise, the filter tries to " +
							"set them outside",
					}),
				);
				list.add(
					new RichBool("onlySelected", false, {
						description: "Flip only selected faces",
						tooltip: "If selected, only selected faces will be affected",
					}),
				);
				break;

			case FP.FP_CLOSE_HOLES: {
				let diag = 1;
				let anySelected = false;
				if (m !== undefined) {
					UpdateBounding.box(m.cm);
					diag = m.cm.bbox.diagonal || 1;
					for (let f = 0; f < m.cm.faceSize && !anySelected; f++) {
						if (!m.cm.isFaceD(f) && m.cm.isFaceS(f)) anySelected = true;
					}
				}
				list.add(
					new RichInt("MaxHoleSize", 30, {
						description: "Max size to be closed ",
						tooltip: "The size is expressed as number of edges composing the hole boundary",
					}),
				);
				list.add(
					new RichBool("Selected", anySelected, {
						description: "Close holes with selected faces",
						tooltip: "Only the holes with at least one of the boundary faces selected are closed",
					}),
				);
				list.add(
					new RichBool("NewFaceSelected", true, {
						description: "Select the newly created faces",
						tooltip:
							"After closing a hole the faces that have been created are left selected. Any " +
							"previous selection is lost. Useful for example for smoothing the newly created " +
							"holes.",
					}),
				);
				list.add(
					new RichBool("SelfIntersection", true, {
						description: "Prevent creation of selfIntersecting faces",
						tooltip:
							"When closing an holes it tries to prevent the creation of faces that intersect " +
							"faces adjacent to the boundary of the hole. It is an heuristic, non " +
							"intersetcting hole filling can be NP-complete.",
					}),
				);
				list.add(
					new RichBool("RefineHole", false, {
						description: "Refine Filled Hole",
						tooltip:
							"After closing the hole it will refine the newly created triangles to make the " +
							"surface more smooth and the triangulation more evenly spaced",
					}),
				);
				list.add(
					new RichPercentage("RefineHoleEdgeLen", diag * 0.03, 0, diag, {
						description: "Hole Refinement Edge Len",
						tooltip: "The target edge lenght of the triangulation inside the filled hole.",
					}),
				);
				break;
			}

			case FP.FP_LOOP_SS:
			case FP.FP_BUTTERFLY_SS:
			case FP.FP_MIDPOINT: {
				if (id === FP.FP_LOOP_SS) {
					list.add(
						new RichEnum("LoopWeight", 0, ["Loop", "Enhance regularity", "Enhance continuity"], {
							description: "Weighting scheme",
							tooltip:
								"Change the weights used. Allows one to optimize some behaviors over others.",
						}),
					);
				}
				list.add(
					new RichInt("Iterations", 3, {
						description: "Iterations",
						tooltip: "Number of time the model is subdivided.",
					}),
				);
				const diag = subdivisionDiagonal(m);
				list.add(
					new RichPercentage("Threshold", diag * 0.01, 0, diag, {
						description: "Edge Threshold",
						tooltip:
							"All the edges <b>longer</b> than this threshold will be refined.<br>Setting this " +
							"value to zero will force an uniform refinement.",
					}),
				);
				list.add(
					new RichBool("Selected", selectedFaceCount(m) > 0, {
						description: "Affect only selected faces",
						tooltip: "If selected the filter affect only the selected faces",
					}),
				);
				break;
			}

			case FP.FP_EXPLICIT_ISOTROPIC_REMESHING: {
				const diag = subdivisionDiagonal(m);
				list.add(
					new RichInt("Iterations", 3, {
						description: "Iterations",
						tooltip: "Number of iterations of the remeshing operations to repeat on the mesh.",
					}),
				);
				list.add(
					new RichBool("Adaptive", false, {
						description: "Adaptive remeshing",
						tooltip: "Toggles adaptive isotropic remeshing.",
					}),
				);
				list.add(
					new RichBool("SelectedOnly", false, {
						description: "Remesh only selected faces",
						tooltip:
							"If checked the remeshing operations will be applied only to the selected faces.",
					}),
				);
				list.add(
					new RichPercentage("TargetLen", diag * 0.01, 0, diag, {
						description: "Target Length",
						tooltip: "Sets the target length for the remeshed mesh edges.",
					}),
				);
				list.add(
					new RichFloat("FeatureDeg", 30, {
						description: "Crease Angle",
						tooltip:
							"Minimum angle between faces of the original to consider the shared edge as a " +
							"feature to be preserved.",
					}),
				);
				list.add(
					new RichBool("CheckSurfDist", true, {
						description: "Check Surface Distance",
						tooltip:
							"If toggled each local operation must deviate from original mesh by [Max. surface distance]",
					}),
				);
				list.add(
					new RichPercentage("MaxSurfDist", diag * 0.01, 0, diag, {
						description: "Max. Surface Distance",
						tooltip: "Maximal surface deviation allowed for each local operation",
					}),
				);
				for (const [name, description, tooltip] of [
					[
						"SplitFlag",
						"Refine Step",
						"If checked the remeshing operations will include a refine step.",
					],
					[
						"CollapseFlag",
						"Collapse Step",
						"If checked the remeshing operations will include a collapse step.",
					],
					[
						"SwapFlag",
						"Edge-Swap Step",
						"If checked the remeshing operations will include a edge-swap step, aimed at improving the vertex valence of the resulting mesh.",
					],
					[
						"SmoothFlag",
						"Smooth Step",
						"If checked the remeshing operations will include a smoothing step, aimed at relaxing the vertex positions in a Laplacian sense.",
					],
					[
						"ReprojectFlag",
						"Reproject Step",
						"If checked the remeshing operations will include a step to reproject the mesh vertices on the original surface.",
					],
				] as const) {
					list.add(new RichBool(name, true, { description, tooltip }));
				}
				break;
			}

			case FP.FP_CLUSTERING: {
				const diag = subdivisionDiagonal(m);
				list.add(
					new RichPercentage("Threshold", diag * 0.01, 0, diag, {
						description: "Cell Size",
						tooltip:
							"The size of the cell of the clustering grid. Smaller the cell finer the resulting " +
							"mesh. For obtaining a very coarse mesh use larger values.",
					}),
				);
				break;
			}

			case FP.FP_COMPUTE_PRINC_CURV_DIR: {
				const diag = subdivisionDiagonal(m);
				list.add(
					new RichEnum(
						"Method",
						3,
						[
							"Taubin approximation",
							"Principal Component Analysis",
							"Normal Cycles",
							"Quadric Fitting",
							"Scale Dependent Quadric Fitting",
						],
						{ description: "Method:", tooltip: "Choose a method" },
					),
				);
				list.add(
					new RichEnum(
						"CurvColorMethod",
						0,
						[
							"Mean Curvature",
							"Gaussian Curvature",
							"Min Curvature",
							"Max Curvature",
							"Shape Index",
							"CurvedNess",
							"None",
						],
						{
							description: "Quality/Color Mapping",
							tooltip:
								"Choose the curvature that is mapped into quality and visualized as per vertex color.",
						},
					),
				);
				list.add(
					new RichPercentage("Scale", diag * 0.1, 0, diag, {
						description: "Curvature Scale",
						tooltip:
							"This parameter is used only for scale dependent methods: 'Scale Dependent " +
							"Quadric Fitting' and 'PCA'.",
					}),
				);
				list.add(
					new RichBool("Autoclean", true, {
						description: "Remove Unreferenced Vertices",
						tooltip:
							"If selected, before starting the filter will remove any unreference vertex (for " +
							"which curvature values are not defined)",
					}),
				);
				break;
			}

			case FP.FP_QUADRIC_SIMPLIFICATION: {
				const faces = m === undefined ? 0 : m.cm.fn;
				let selectedFaces = 0;
				if (m !== undefined) {
					for (let f = 0; f < m.cm.faceSize; f++) {
						if (!m.cm.isFaceD(f) && m.cm.isFaceS(f)) selectedFaces++;
					}
				}
				const d = defaultQuadricParameters();
				list.add(
					new RichInt(
						"TargetFaceNum",
						Math.floor((selectedFaces > 0 ? selectedFaces : faces) / 2),
						{
							description: "Target number of faces",
							tooltip: "The desired final number of faces.",
						},
					),
				);
				list.add(
					new RichFloat("TargetPerc", 0, {
						description: "Percentage reduction (0..1)",
						tooltip:
							"If non zero, this parameter specifies the desired final size of the mesh as a " +
							"percentage of the initial size.",
					}),
				);
				list.add(
					new RichFloat("QualityThr", d.qualityThr, {
						description: "Quality threshold",
						tooltip:
							"Quality threshold for penalizing bad shaped faces. The value is in the range " +
							"[0..1]; 0 accepts any kind of face, 0.5 penalizes faces with quality < 0.5.",
					}),
				);
				list.add(
					new RichBool("PreserveBoundary", d.preserveBoundary, {
						description: "Preserve Boundary of the mesh",
						tooltip:
							"The simplification process tries to do not affect mesh boundaries during " +
							"simplification",
					}),
				);
				list.add(
					new RichFloat("BoundaryWeight", 1.0, {
						description: "Boundary Preserving Weight",
						tooltip:
							"The importance of the boundary during simplification. Default (1.0) means that " +
							"the boundary has the same importance of the rest.",
					}),
				);
				list.add(
					new RichBool("PreserveNormal", d.normalCheck, {
						description: "Preserve Normal",
						tooltip:
							"Try to avoid face flipping effects and try to preserve the original orientation " +
							"of the surface",
					}),
				);
				list.add(
					new RichBool("PreserveTopology", d.preserveTopology, {
						description: "Preserve Topology",
						tooltip:
							"Avoid all the collapses that should cause a topology change in the mesh (like " +
							"closing holes, squeezing handles, etc). If checked the genus of the mesh should " +
							"stay unchanged.",
					}),
				);
				list.add(
					new RichBool("OptimalPlacement", d.optimalPlacement, {
						description: "Optimal position of simplified vertices",
						tooltip:
							"Each collapsed vertex is placed in the position minimizing the quadric error. " +
							"If disabled edges are collapsed onto one of the two original vertices and the " +
							"final mesh is composed by a subset of the original vertices.",
					}),
				);
				list.add(
					new RichBool("PlanarQuadric", d.qualityQuadric, {
						description: "Planar Simplification",
						tooltip:
							"Add additional simplification constraints that improves the quality of the " +
							"simplification of the planar portion of the mesh.",
					}),
				);
				list.add(
					new RichFloat("PlanarWeight", d.qualityQuadricWeight, {
						description: "Planar Simp. Weight",
						tooltip:
							"How much we should try to preserve the triangles in the planar regions. If you " +
							"lower this value planar areas will be simplified more.",
					}),
				);
				list.add(
					new RichBool("QualityWeight", d.qualityWeight, {
						description: "Weighted Simplification",
						tooltip:
							"Use the Per-Vertex quality as a weighting factor for the simplification. The " +
							"weight is used as a error amplification value.",
					}),
				);
				list.add(
					new RichBool("AutoClean", true, {
						description: "Post-simplification cleaning",
						tooltip:
							"After the simplification an additional set of steps is performed to clean the " +
							"mesh (unreferenced vertices, bad faces, etc)",
					}),
				);
				list.add(
					new RichBool("Selected", selectedFaces > 0, {
						description: "Simplify only selected faces",
						tooltip: "The simplification is applied only to the selected set of faces.",
					}),
				);
				break;
			}

			case FP.FP_SCALE: {
				list.add(new RichFloat("axisX", 1, { description: "X Axis", tooltip: "Scaling" }));
				list.add(new RichFloat("axisY", 1, { description: "Y Axis", tooltip: "Scaling" }));
				list.add(new RichFloat("axisZ", 1, { description: "Z Axis", tooltip: "Scaling" }));
				list.add(
					new RichBool("uniformFlag", true, {
						description: "Uniform Scaling",
						tooltip:
							"If selected an uniform scaling (the same for all the three axis) is applied " +
							"(the X axis value is used)",
					}),
				);
				list.add(
					new RichEnum("scaleCenter", 0, ["origin", "barycenter", "custom point"], {
						description: "Center of scaling:",
						tooltip: "Choose a method",
					}),
				);
				list.add(
					new RichPosition("customCenter", [0, 0, 0], {
						description: "Custom center",
						tooltip: "This scaling center is used only if the 'custom point' option is chosen.",
					}),
				);
				list.add(
					new RichBool("unitFlag", false, {
						description: "Scale to Unit bbox",
						tooltip:
							"If selected, the object is scaled to a box whose sides are at most 1 unit length",
					}),
				);
				list.add(
					new RichBool("Freeze", true, {
						description: "Freeze Matrix",
						tooltip:
							"The transformation is explicitly applied, and the vertex coordinates are " +
							"actually changed",
					}),
				);
				break;
			}

			case FP.FP_CENTER: {
				list.add(
					new RichEnum(
						"traslMethod",
						0,
						["XYZ translation", "Center on Scene BBox", "Center on Layer BBox", "Set new Origin"],
						{ description: "Transformation:", tooltip: "Choose a method" },
					),
				);
				list.add(
					new RichFloat("axisX", 0, {
						description: "X Axis",
						tooltip: "Absolute translation amount along the X axis",
					}),
				);
				list.add(
					new RichFloat("axisY", 0, {
						description: "Y Axis",
						tooltip: "Absolute translation amount along the Y axis",
					}),
				);
				list.add(
					new RichFloat("axisZ", 0, {
						description: "Z Axis",
						tooltip: "Absolute translation amount along the Z axis",
					}),
				);
				list.add(
					new RichPosition("newOrigin", [0, 0, 0], {
						description: "New Origin:",
						tooltip: "when using [Set new Origin], this is the location of the new Origin.",
					}),
				);
				list.add(
					new RichBool("Freeze", true, {
						description: "Freeze Matrix",
						tooltip:
							"The transformation is explicitly applied, and the vertex coordinates are " +
							"actually changed",
					}),
				);
				break;
			}

			case FP.FP_ROTATE: {
				list.add(
					new RichEnum("rotAxis", 0, ["X axis", "Y axis", "Z axis", "custom axis"], {
						description: "Rotation on:",
						tooltip: "Choose a method",
					}),
				);
				list.add(
					new RichEnum("rotCenter", 0, ["origin", "barycenter", "custom point"], {
						description: "Center of rotation:",
						tooltip: "Choose a method",
					}),
				);
				list.add(
					new RichFloat("angle", 0, {
						description: "Rotation Angle",
						tooltip:
							"Angle of rotation (in degrees). If snapping is enabled this value is rounded according to the snap value",
					}),
				);
				list.add(
					new RichDirection("customAxis", [0, 0, 0], {
						description: "Custom axis",
						tooltip: "This rotation axis is used only if the 'custom axis' option is chosen.",
					}),
				);
				list.add(
					new RichPosition("customCenter", [0, 0, 0], {
						description: "Custom center",
						tooltip: "This rotation center is used only if the 'custom point' option is chosen.",
					}),
				);
				list.add(
					new RichBool("Freeze", true, {
						description: "Freeze Matrix",
						tooltip:
							"The transformation is explicitly applied, and the vertex coordinates are " +
							"actually changed",
					}),
				);
				break;
			}

			case FP.FP_NORMAL_EXTRAPOLATION:
				list.add(
					new RichInt("K", 10, {
						description: "Neighbour num",
						tooltip: "The number of neighbors used to estimate normals.",
					}),
				);
				list.add(
					new RichInt("smoothIter", 0, {
						description: "Smooth Iteration",
						tooltip:
							"The number of smoothing iteration done on the p used to estimate and propagate " +
							"normals.",
					}),
				);
				list.add(
					new RichBool("flipFlag", false, {
						description: "Flip normals w.r.t. viewpoint",
						tooltip:
							"If the 'viewpoint' (i.e. scanner position) is known, it can be used to " +
							"disambiguate normals orientation, so that all the normals will be oriented in " +
							"the same direction.",
					}),
				);
				list.add(
					new RichPosition("viewPos", [0, 0, 0], {
						description: "Viewpoint Pos.",
						tooltip: "The viewpoint position can be set by hand.",
					}),
				);
				break;

			case FP.FP_NORMAL_SMOOTH_POINTCLOUD:
				list.add(
					new RichInt("K", 10, {
						description: "Number of neigbors",
						tooltip: "The number of neighbors used to smooth normals.",
					}),
				);
				list.add(
					new RichBool("useDist", false, {
						description: "Weight using neighbour distance",
						tooltip: "If selected, the neighbour normals are waighted according to their distance.",
					}),
				);
				break;

			case FP.FP_INVERT_TRANSFORM:
				transformOptions();
				break;

			case FP.FP_SET_TRANSFORM_MATRIX:
				list.add(
					new RichMatrix44("TransformMatrix", Array.from(Matrix44Ops.identity()), {
						description: "Transform Matrix",
						tooltip: "The matrix to set as the layer's transformation.",
					}),
				);
				list.add(
					new RichBool("compose", false, {
						description: "Compose with current",
						tooltip:
							"If selected, the given matrix is multiplied onto the layer's current one rather " +
							"than replacing it.",
					}),
				);
				transformOptions();
				break;

			case FP.FP_SET_TRANSFORM_PARAMS: {
				for (const [axis, defval] of [
					["X", 0],
					["Y", 0],
					["Z", 0],
				] as const) {
					list.add(
						new RichFloat(`translation${axis}`, defval, {
							description: `Translation on ${axis}`,
							tooltip: `Translation factor on ${axis} axis`,
						}),
					);
				}
				for (const axis of ["X", "Y", "Z"] as const) {
					list.add(
						new RichFloat(`rotation${axis}`, 0, {
							description: `Rotation on ${axis}`,
							tooltip: `Rotation angle on ${axis} axis, in degrees`,
						}),
					);
				}
				for (const axis of ["X", "Y", "Z"] as const) {
					list.add(
						new RichFloat(`scale${axis}`, 1, {
							description: `Scale on ${axis}`,
							tooltip: `Scaling factor on ${axis} axis`,
						}),
					);
				}
				list.add(
					new RichBool("compose", false, {
						description: "Compose with current",
						tooltip:
							"If selected, the built matrix is multiplied onto the layer's current one rather " +
							"than replacing it.",
					}),
				);
				transformOptions();
				break;
			}

			case FP.FP_PRINCIPAL_AXIS:
				list.add(
					new RichBool("pointsFlag", true, {
						description: "Use vertex",
						tooltip:
							"If selected, only the vertices of the mesh are used to compute the Principal Axis. Mandatory for point clouds or for non water tight meshes",
					}),
				);
				transformOptions();
				break;

			case FP.FP_FLIP_AND_SWAP:
				for (const [name, label] of [
					["flipX", "Flip X axis"],
					["flipY", "Flip Y axis"],
					["flipZ", "Flip Z axis"],
				] as const) {
					list.add(
						new RichBool(name, false, {
							description: label,
							tooltip: `If selected the axis will be swapped (mesh mirrored along the ${name.slice(-1)} axis)`,
						}),
					);
				}
				for (const [name, label] of [
					["swapXY", "Swap X-Y axis"],
					["swapXZ", "Swap X-Z axis"],
					["swapYZ", "Swap Y-Z axis"],
				] as const) {
					list.add(
						new RichBool(name, false, {
							description: label,
							tooltip: `If selected the two axis will be swapped. All the swaps are performed in this order`,
						}),
					);
				}
				transformOptions();
				break;

			case FP.FP_ROTATE_FIT:
				list.add(
					new RichEnum("targetPlane", 0, ["XY plane", "YZ plane", "ZX plane"], {
						description: "Rotate to fit:",
						tooltip: "Choose the plane where the selection will fit",
					}),
				);
				list.add(
					new RichEnum("rotAxis", 0, ["any axis", "X axis", "Y axis", "Z axis"], {
						description: "Rotate on:",
						tooltip:
							"Choose on which axis do the rotation: 'any axis' guarantee the best fit of the " +
							"selection to the plane, only use X,Y or Z it if you want to preserve that axis.",
					}),
				);
				transformOptions();
				break;

			case FP.FP_FAUX_CREASE:
				list.add(
					new RichFloat("AngleDegNeg", -30, {
						description: "Concave Angle Thr. (deg)",
						tooltip: "Concave dihedral angle threshold",
					}),
				);
				list.add(
					new RichFloat("AngleDegPos", 30, {
						description: "Convex Angle Thr. (deg)",
						tooltip: "Convex dihedral angle threshold",
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
		const m = doc.mm();
		const cm = m.cm;
		post.mask = GEOMETRY_AND_TOPOLOGY;

		switch (id) {
			case FP.FP_REORIENT: {
				// Orientability is only a question once every edge has at most
				// two faces; upstream refuses rather than guessing, and so do we.
				if (Clean.countNonManifoldEdgeFF(cm) > 0) {
					throw new MLException(
						"Mesh has some not 2-manifold edges, orientability requires manifoldness. " +
							'Run "Repair non Manifold Edges" first.',
					);
				}
				const { isOriented, isOrientable } = Clean.orientCoherentlyMesh(cm);
				UpdateTopology.faceFace(cm);
				m.updateBoxAndNormals();
				doc.Log.log(
					isOrientable
						? isOriented
							? "Mesh was already coherently oriented"
							: "Mesh has been re-oriented coherently"
						: "Mesh is not orientable; it was left partially re-oriented",
				);
				return { isOriented, isOrientable };
			}

			case FP.FP_INVERT_FACES: {
				const onlySelected = params.getBool("onlySelected");
				let flipped: boolean;
				if (params.getBool("forceFlip")) {
					Clean.flipMesh(cm, onlySelected);
					flipped = true;
				} else {
					// "Guess" mode: only flip when the mesh encloses a negative
					// volume, i.e. it is inside out.
					flipped = Clean.flipNormalOutside(cm);
				}
				m.updateBoxAndNormals();
				doc.Log.log(flipped ? "Faces have been flipped" : "Faces were already facing outward");
				return { flipped };
			}

			case FP.FP_CLOSE_HOLES: {
				if (Clean.countNonManifoldEdgeFF(cm) > 0) {
					throw new MLException(
						"Mesh has some not 2-manifold edges, filter requires edge manifoldness. " +
							'Run "Repair non Manifold Edges" first.',
					);
				}
				// Refinement subdivides the cap so it is evenly spaced rather
				// than a fan of long thin triangles. It is not implemented yet,
				// and quietly ignoring the request would leave the caller
				// believing they got a refined patch. Refuse instead.
				if (params.getBool("RefineHole")) {
					throw new MLNotImplementedException(
						"Close Holes with RefineHole=true (the hole is closed, but refining the patch " +
							"is not implemented yet; pass RefineHole=false to close it unrefined)",
						this.pluginName(),
					);
				}

				// The ear scoring reads vertex normals to tell convex ears from
				// concave ones, so they have to be current before filling.
				m.updateBoxAndNormals();

				const { holeCount, newFaces, firstNewFace } = Hole.fillHoles(cm, {
					maxHoleSize: params.getInt("MaxHoleSize"),
					selected: params.getBool("Selected"),
					strategy: params.getBool("SelfIntersection") ? "selfIntersection" : "minimumWeight",
				});

				if (params.getBool("NewFaceSelected")) Hole.selectFacesFrom(cm, firstNewFace);
				Allocator.compactEveryVector(cm);
				UpdateTopology.faceFace(cm);
				m.updateBoxAndNormals();

				doc.Log.log(`Closed ${holeCount} holes and added ${newFaces} new faces`);
				// Upstream's output keys, so a caller reading them keeps working.
				return { closed_holes: holeCount, new_faces: newFaces };
			}

			case FP.FP_LOOP_SS:
			case FP.FP_BUTTERFLY_SS:
			case FP.FP_MIDPOINT: {
				if (id === FP.FP_LOOP_SS && params.getEnum("LoopWeight") !== 0) {
					throw new MLNotImplementedException(
						'Subdivision Surfaces: Loop currently supports only the "Loop" weighting scheme; ' +
							'"Enhance regularity" and "Enhance continuity" are not implemented yet.',
						"FilterMeshing",
					);
				}
				const iterations = params.getInt("Iterations");
				if (iterations < 1)
					throw new MLException(`Iterations must be at least 1, got ${iterations}`);
				const threshold = params.getAbsPerc("Threshold");
				// Zero means "every edge"; anything else refines only what is
				// longer than it, which is how a hand-tuned pass stays local.
				const predicate = threshold > 0 ? Refine.longerThan(threshold) : Refine.everyEdge;
				const options = { selectedOnly: params.getBool("Selected") };

				const before = { vn: cm.vn, fn: cm.fn };
				let done = 0;
				for (let i = 0; i < iterations; i++) {
					UpdateTopology.faceFace(cm);
					const changed =
						id === FP.FP_LOOP_SS
							? Refine.refineLoop(cm, predicate, options)
							: Refine.refineE(
									cm,
									id === FP.FP_BUTTERFLY_SS ? Refine.midPointButterfly : Refine.midPoint,
									predicate,
									options,
								);
					if (!changed) break;
					done++;
				}
				Allocator.compactEveryVector(cm);
				m.updateBoxAndNormals();
				doc.Log.log(
					`Subdivided ${done} time(s): ${before.vn} vertices and ${before.fn} faces became ${cm.vn} and ${cm.fn}`,
				);
				return { iterations_done: done, vertex_number: cm.vn, face_number: cm.fn };
			}

			case FP.FP_EXPLICIT_ISOTROPIC_REMESHING: {
				if (params.getBool("Adaptive")) {
					throw new MLNotImplementedException(
						"Adaptive isotropic remeshing is not implemented yet; the target length is " +
							"applied uniformly.",
						"FilterMeshing",
					);
				}
				UpdateBounding.box(cm);
				const diag = cm.bbox.diagonal || 1;
				const targetLen = params.getAbsPerc("TargetLen");
				if (!(targetLen > 0)) {
					throw new MLException(`Target Length must be greater than zero, got ${targetLen}`);
				}
				const before = { vn: cm.vn, fn: cm.fn };
				const stats = IsotropicRemeshing.isotropicRemeshing(cm, {
					iterations: params.getInt("Iterations"),
					targetLen,
					featureDeg: params.getFloat("FeatureDeg"),
					// Zero would refuse every operation, so read it as "no limit".
					maxSurfDist: params.getAbsPerc("MaxSurfDist") || diag,
					checkSurfDist: params.getBool("CheckSurfDist"),
					splitFlag: params.getBool("SplitFlag"),
					collapseFlag: params.getBool("CollapseFlag"),
					swapFlag: params.getBool("SwapFlag"),
					smoothFlag: params.getBool("SmoothFlag"),
					reprojectFlag: params.getBool("ReprojectFlag"),
					selectedOnly: params.getBool("SelectedOnly"),
				});
				m.updateBoxAndNormals();
				doc.Log.log(
					`Remeshed ${before.vn}/${before.fn} to ${cm.vn}/${cm.fn} ` +
						`(${stats.splits} splits, ${stats.collapses} collapses, ${stats.flips} flips)`,
				);
				return { vertex_number: cm.vn, face_number: cm.fn, ...stats };
			}

			case FP.FP_CLUSTERING: {
				const cellSize = params.getAbsPerc("Threshold");
				if (!(cellSize > 0)) {
					throw new MLException(`Cell Size must be greater than zero, got ${cellSize}`);
				}
				const before = { vn: cm.vn, fn: cm.fn };
				IsotropicRemeshing.clusteringDecimation(cm, cellSize);
				m.updateBoxAndNormals();
				doc.Log.log(
					`Clustered ${before.vn}/${before.fn} down to ${cm.vn}/${cm.fn} at cell size ${cellSize}`,
				);
				return { vertex_number: cm.vn, face_number: cm.fn };
			}

			case FP.FP_COMPUTE_PRINC_CURV_DIR: {
				const method = params.getInt("Method");
				if (method === 1 || method === 2 || method === 4) {
					throw new MLNotImplementedException(
						"Only 'Taubin approximation' and 'Quadric Fitting' are implemented; PCA, Normal " +
							"Cycles and the scale-dependent fit are not.",
						"FilterMeshing",
					);
				}
				if (params.getBool("Autoclean")) {
					// A vertex with no incident face has no neighbourhood to be
					// curved in, so upstream drops those before starting.
					const removed = Clean.removeUnreferencedVertex(cm);
					if (removed > 0) {
						Allocator.compactEveryVector(cm);
						doc.Log.log(`Removed ${removed} unreferenced vertices`);
					}
				}
				UpdateTopology.faceFace(cm);
				if (method === 0) Curvature.principalDirections(cm);
				else Curvature.principalDirectionsFitting(cm);

				const mapping = params.getEnum("CurvColorMethod");
				let min = Number.POSITIVE_INFINITY;
				let max = Number.NEGATIVE_INFINITY;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					const value = Curvature.curvatureToScalar(cm, v, mapping);
					cm.vertQuality[v] = value;
					min = Math.min(min, value);
					max = Math.max(max, value);
				}
				if (mapping !== Curvature.CurvatureMapping.None) {
					// Cropped at the 10th and 90th percentile, as upstream does,
					// so a few spikes do not flatten the ramp.
					const live: number[] = [];
					for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) live.push(cm.vertQuality[v]);
					live.sort((a, b) => a - b);
					const lo = live[Math.floor(0.1 * (live.length - 1))];
					const hi = live[Math.ceil(0.9 * (live.length - 1))];
					for (let v = 0; v < cm.vertSize; v++) {
						if (!cm.isVertD(v)) cm.vertColor[v] = colorRamp(lo, hi, cm.vertQuality[v]);
					}
				}
				doc.Log.log(`Curvature mapped into quality over [${min}, ${max}]`);
				return { min_value: min, max_value: max };
			}

			case FP.FP_QUADRIC_SIMPLIFICATION: {
				const perc = params.getFloat("TargetPerc");
				const targetFaceNum =
					perc !== 0 ? Math.round(cm.fn * perc) : params.getInt("TargetFaceNum");

				const before = cm.fn;
				const result = quadricSimplification(cm, {
					targetFaceNum,
					selected: params.getBool("Selected"),
					callback: _cb,
					params: {
						qualityThr: params.getFloat("QualityThr"),
						preserveBoundary: params.getBool("PreserveBoundary"),
						// Upstream multiplies its default by the user's factor
						// rather than replacing it.
						boundaryQuadricWeight:
							defaultQuadricParameters().boundaryQuadricWeight * params.getFloat("BoundaryWeight"),
						normalCheck: params.getBool("PreserveNormal"),
						preserveTopology: params.getBool("PreserveTopology"),
						optimalPlacement: params.getBool("OptimalPlacement"),
						qualityQuadric: params.getBool("PlanarQuadric"),
						qualityQuadricWeight: params.getFloat("PlanarWeight"),
						qualityWeight: params.getBool("QualityWeight"),
					},
				});

				if (params.getBool("AutoClean")) {
					const nullFaces = Clean.removeFaceOutOfRangeArea(cm, 0);
					if (nullFaces > 0) {
						doc.Log.log(`PostSimplification Cleaning: Removed ${nullFaces} null faces`);
					}
					const dupVerts = Clean.removeDuplicateVertex(cm);
					if (dupVerts > 0) {
						doc.Log.log(`PostSimplification Cleaning: Removed ${dupVerts} duplicated vertices`);
					}
					const unref = Clean.removeUnreferencedVertex(cm);
					if (unref > 0) {
						doc.Log.log(`PostSimplification Cleaning: Removed ${unref} unreferenced vertices`);
					}
				}

				Allocator.compactEveryVector(cm);
				m.clearDataMask(MeshElement.MM_FACEFACETOPO);
				m.updateBoxAndNormals();

				doc.Log.log(`Reduced from ${before} to ${cm.fn} faces`);
				// Falling short of the target is a normal outcome with
				// PreserveTopology — every surface has a coarsest
				// triangulation, and a torus cannot go below about 14 faces
				// without becoming a sphere. Say so, rather than leaving the
				// caller to wonder why they asked for 10 and got 18.
				if (cm.fn > targetFaceNum) {
					// Which of the three reasons it was matters a great deal to
					// whoever has to fix it, so name the likely one rather than
					// giving one vague message for all of them.
					let why: string;
					if (params.getBool("Selected") && result.performed === 0) {
						why =
							"only selected faces were eligible. Note that Close Holes leaves the faces it " +
							'creates selected, so a preceding "Select None" is usually what is wanted.';
					} else if (params.getBool("PreserveTopology")) {
						why = "no further collapse was possible without changing the topology of the mesh.";
					} else {
						why = "no further collapse was legal on this mesh.";
					}
					doc.Log.warning(`Could not reach the target of ${targetFaceNum} faces: ${why}`);
				}

				return {
					target_face_num: targetFaceNum,
					initial_faces: result.initialFaces,
					final_faces: cm.fn,
					collapses: result.performed,
					target_reached: cm.fn <= targetFaceNum,
				};
			}

			case FP.FP_SCALE:
			case FP.FP_CENTER:
			case FP.FP_ROTATE: {
				const matrix = this.buildTransform(id, params, m);
				// MeshLab keeps the transform on the layer and only bakes it
				// into the coordinates when asked. Freeze defaults to true,
				// because a headless caller almost always wants the geometry
				// actually moved.
				if (params.getBool("Freeze")) {
					UpdatePosition.applyMatrix(cm, matrix);
					cm.transformMatrix = Matrix44Ops.identity();
				} else {
					cm.transformMatrix = Matrix44Ops.multiply(matrix, cm.transformMatrix);
					post.mask = MeshElement.MM_TRANSFMATRIX;
				}
				m.updateBoxAndNormals();
				return { matrix: Array.from(matrix) };
			}

			case FP.FP_FREEZE_TRANSFORM: {
				UpdatePosition.applyMatrix(cm, cm.transformMatrix);
				cm.transformMatrix = Matrix44Ops.identity();
				m.updateBoxAndNormals();
				doc.Log.log("Transformation matrix frozen into the vertex coordinates");
				return {};
			}

			case FP.FP_RESET_TRANSFORM: {
				cm.transformMatrix = Matrix44Ops.identity();
				post.mask = MeshElement.MM_TRANSFMATRIX;
				doc.Log.log("Transformation matrix reset to the identity");
				return {};
			}

			case FP.FP_NORMAL_EXTRAPOLATION: {
				const flip = params.getBool("flipFlag");
				estimateNormals(cm, {
					neighbors: params.getInt("K"),
					smoothIterations: params.getInt("smoothIter"),
					// Without a viewpoint the sign of each plane fit is
					// arbitrary, so orientation falls back to propagating across
					// the neighbour graph.
					...(flip
						? { viewpoint: [...params.getPoint3m("viewPos")] as [number, number, number] }
						: {}),
				});
				post.mask = MeshElement.MM_VERTNORMAL;
				doc.Log.log(`Estimated normals for ${cm.vn} points`);
				return { vertices: cm.vn };
			}

			case FP.FP_NORMAL_SMOOTH_POINTCLOUD: {
				// Smoothing alone: re-run the estimator with zero neighbours for
				// the fit is not meaningful, so this simply runs the smoothing
				// passes over the field already present.
				estimateNormals(cm, {
					neighbors: params.getInt("K"),
					smoothIterations: 1,
				});
				post.mask = MeshElement.MM_VERTNORMAL;
				doc.Log.log(`Smoothed normals for ${cm.vn} points`);
				return { vertices: cm.vn };
			}

			case FP.FP_INVERT_TRANSFORM:
			case FP.FP_SET_TRANSFORM_MATRIX:
			case FP.FP_SET_TRANSFORM_PARAMS:
			case FP.FP_PRINCIPAL_AXIS:
			case FP.FP_FLIP_AND_SWAP:
			case FP.FP_ROTATE_FIT: {
				const built = this.buildMatrix(id, params, doc, m);
				const targets = params.getBool("allLayers") ? doc.visibleMeshes() : [m];
				const freeze = params.getBool("Freeze");
				// `compose` multiplies onto whatever the layer already carries;
				// otherwise the matrix replaces it outright. Inversion is the odd
				// one out: it has nothing to compose and acts on the layer's own
				// matrix rather than on a new one.
				const compose = params.hasParameter("compose") ? params.getBool("compose") : true;
				for (const layer of targets) {
					if (id === FP.FP_INVERT_TRANSFORM) {
						const inverse = Matrix44Ops.invert(layer.cm.transformMatrix);
						if (inverse === null) {
							throw new MLException(
								`The transformation matrix of "${layer.label()}" is singular and cannot be inverted.`,
							);
						}
						layer.cm.transformMatrix = inverse;
					} else if (compose) {
						layer.cm.transformMatrix = Matrix44Ops.multiply(built, layer.cm.transformMatrix);
					} else {
						layer.cm.transformMatrix = built;
					}
					if (freeze) {
						UpdatePosition.applyMatrix(layer.cm, layer.cm.transformMatrix);
						layer.cm.transformMatrix = Matrix44Ops.identity();
					}
					layer.updateBoxAndNormals();
				}
				post.mask = freeze ? GEOMETRY_AND_TOPOLOGY : MeshElement.MM_TRANSFMATRIX;
				return { ...this.lastMatrixOutput, matrix: Array.from(built) };
			}

			case FP.FP_FAUX_CREASE: {
				UpdateTopology.faceFace(cm);
				const negative = (params.getFloat("AngleDegNeg") * Math.PI) / 180;
				const positive = (params.getFloat("AngleDegPos") * Math.PI) / 180;
				CreaseCut.faceEdgeSelSignedCrease(cm, negative, positive);
				let selected = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					for (let e = 0; e < 3; e++) {
						if ((cm.faceFlags[f] & (FaceFlag.FACEEDGESEL0 << e)) !== 0) selected++;
					}
				}
				// Each interior crease is counted from both of its faces.
				doc.Log.log(`Selected ${selected} crease edge sides`);
				post.mask = MeshElement.MM_NONE;
				return { selected_edges: selected };
			}

			case FP.FP_MAKE_PURE_TRI: {
				// A "polygon" here is triangles joined by faux edges, so turning
				// the mesh pure-triangular is exactly forgetting which edges were
				// faux — no geometry changes at all.
				let cleared = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					if ((cm.faceFlags[f] & FaceFlag.FAUX012) !== 0) cleared++;
					cm.faceFlags[f] &= ~FaceFlag.FAUX012;
				}
				m.clearDataMask(MeshElement.MM_POLYGONAL);
				m.updateBoxAndNormals();
				doc.Log.log(`Cleared the faux flags of ${cleared} faces`);
				return { face_number: cm.fn };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	/** Builds the matrix a transform filter describes. */
	/**
	 * Extra numbers the last `buildMatrix` call wants to report — the fitting
	 * plane's normal and error, which only `Rotate to Fit` produces.
	 */
	private lastMatrixOutput: FilterOutput = {};

	/** The matrix each of the six matrix filters wants applied. */
	private buildMatrix(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		m: MeshModel,
	): Float64Array {
		this.lastMatrixOutput = {};
		switch (id) {
			// Inversion works on the layer's own matrix, so there is nothing to
			// build here; the caller handles it.
			case FP.FP_INVERT_TRANSFORM:
				return Matrix44Ops.identity();

			case FP.FP_SET_TRANSFORM_MATRIX:
				return Float64Array.from(params.getMatrix44("TransformMatrix"));

			case FP.FP_SET_TRANSFORM_PARAMS: {
				const t = Matrix44Ops.translation(
					params.getFloat("translationX"),
					params.getFloat("translationY"),
					params.getFloat("translationZ"),
				);
				// Euler angles in MeshLab's order: X, then Y, then Z, each about
				// the fixed world axis.
				let r = Matrix44Ops.identity();
				for (const [axis, unit] of [
					["rotationX", [1, 0, 0]],
					["rotationY", [0, 1, 0]],
					["rotationZ", [0, 0, 1]],
				] as const) {
					const deg = params.getFloat(axis);
					if (deg === 0) continue;
					r = Matrix44Ops.multiply(
						Matrix44Ops.rotation((deg * Math.PI) / 180, unit[0], unit[1], unit[2]),
						r,
					);
				}
				const s = Matrix44Ops.scaling(
					params.getFloat("scaleX"),
					params.getFloat("scaleY"),
					params.getFloat("scaleZ"),
				);
				return Matrix44Ops.multiply(Matrix44Ops.multiply(t, r), s);
			}

			case FP.FP_FLIP_AND_SWAP: {
				let tr = Matrix44Ops.identity();
				const apply = (next: Float64Array) => {
					tr = Matrix44Ops.multiply(tr, next);
				};
				for (const [name, axis] of [
					["flipX", 0],
					["flipY", 1],
					["flipZ", 2],
				] as const) {
					if (!params.getBool(name)) continue;
					const flip = Matrix44Ops.identity();
					flip[5 * axis] = -1;
					apply(flip);
				}
				for (const [name, a, b] of [
					["swapXY", 0, 1],
					["swapXZ", 0, 2],
					["swapYZ", 1, 2],
				] as const) {
					if (!params.getBool(name)) continue;
					const swap = Matrix44Ops.identity();
					swap[5 * a] = 0;
					swap[5 * b] = 0;
					swap[4 * a + b] = 1;
					swap[4 * b + a] = 1;
					apply(swap);
				}
				return tr;
			}

			case FP.FP_PRINCIPAL_AXIS:
				return principalAxisMatrix(m, params.getBool("pointsFlag"));

			case FP.FP_ROTATE_FIT: {
				const result = rotateToFitMatrix(
					m,
					params.getEnum("targetPlane"),
					params.getEnum("rotAxis"),
				);
				this.lastMatrixOutput = {
					fitting_plane_avg_error: result.error,
					fitting_plane_normal: Array.from(result.normal),
				};
				doc.Log.log(
					`Fitting plane normal is [${result.normal.join(", ")}], average error ${result.error}`,
				);
				return result.matrix;
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	private buildTransform(id: ActionIDType, params: RichParameterList, m: MeshModel): Float64Array {
		const cm = m.cm;
		UpdateBounding.box(cm);

		switch (id) {
			case FP.FP_SCALE: {
				let sx = params.getFloat("axisX");
				let sy = params.getFloat("axisY");
				let sz = params.getFloat("axisZ");
				if (params.getBool("uniformFlag")) {
					sy = sx;
					sz = sx;
				}
				if (params.getBool("unitFlag")) {
					// "at most 1 unit length" — the longest side sets the scale,
					// so the whole box fits inside the unit cube.
					const maxDim = cm.bbox.maxDim || 1;
					sx = 1 / maxDim;
					sy = sx;
					sz = sx;
				}
				const centre = this.centreOf(params.getEnum("scaleCenter"), params, cm);
				// Scale about the chosen centre: move it to the origin, scale,
				// move back.
				return Matrix44Ops.multiply(
					Matrix44Ops.translation(centre[0], centre[1], centre[2]),
					Matrix44Ops.multiply(
						Matrix44Ops.scaling(sx, sy, sz),
						Matrix44Ops.translation(-centre[0], -centre[1], -centre[2]),
					),
				);
			}

			case FP.FP_CENTER: {
				switch (params.getEnum("traslMethod")) {
					case 1:
					case 2: {
						const c = cm.bbox.center;
						return Matrix44Ops.translation(-c[0], -c[1], -c[2]);
					}
					case 3: {
						const o = params.getPoint3m("newOrigin");
						return Matrix44Ops.translation(-o[0], -o[1], -o[2]);
					}
					default:
						return Matrix44Ops.translation(
							params.getFloat("axisX"),
							params.getFloat("axisY"),
							params.getFloat("axisZ"),
						);
				}
			}

			case FP.FP_ROTATE: {
				const axes: ReadonlyArray<readonly [number, number, number]> = [
					[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1],
				];
				const which = params.getEnum("rotAxis");
				const axis = which < 3 ? axes[which] : params.getPoint3m("customAxis");
				const centre = this.centreOf(params.getEnum("rotCenter"), params, cm);
				const rot = Matrix44Ops.rotation(
					(params.getFloat("angle") * Math.PI) / 180,
					axis[0],
					axis[1],
					axis[2],
				);
				return Matrix44Ops.multiply(
					Matrix44Ops.translation(centre[0], centre[1], centre[2]),
					Matrix44Ops.multiply(rot, Matrix44Ops.translation(-centre[0], -centre[1], -centre[2])),
				);
			}

			default:
				return Matrix44Ops.identity();
		}
	}

	/** origin / barycentre / custom point, as the enum selects. */
	private centreOf(
		choice: number,
		params: RichParameterList,
		cm: CMeshO,
	): readonly [number, number, number] {
		if (choice === 1) return Inertia.vertexBarycenter(cm);
		if (choice === 2) return params.getPoint3m("customCenter");
		return [0, 0, 0];
	}
}

/** Bounding-box diagonal, or 1 when there is no mesh to measure yet. */
function subdivisionDiagonal(m: MeshModel | undefined): number {
	if (m === undefined) return 1;
	UpdateBounding.box(m.cm);
	return m.cm.bbox.diagonal || 1;
}

/** How many faces are selected, which is what decides the "Selected" default. */
function selectedFaceCount(m: MeshModel | undefined): number {
	if (m === undefined) return 0;
	let n = 0;
	for (let f = 0; f < m.cm.faceSize; f++) if (!m.cm.isFaceD(f) && m.cm.isFaceS(f)) n++;
	return n;
}

/**
 * A rotation taking the mesh's principal axes onto the world axes.
 *
 * The principal axes are the eigenvectors of the inertia tensor — of the mass
 * distribution when the mesh is a closed solid, or of the point positions when
 * `usePoints` is set. A point cloud or an open shell has no interior, so the
 * mass tensor is meaningless there and the point covariance is the only
 * sensible answer; upstream defaults to it for the same reason.
 *
 * The two modes order their axes *oppositely*, and it is worth knowing before
 * being surprised by it. Both take the eigenvalues ascending, but a covariance
 * eigenvalue grows with the spread along its axis while a moment of inertia
 * shrinks with it. So the covariance mode puts the longest axis last (on Z) and
 * the inertia mode puts it first (on X). This is upstream's behaviour in both
 * cases — it simply sorts whichever matrix it was given — and scripts depend on
 * it, so it is reproduced rather than reconciled.
 */
function principalAxisMatrix(m: MeshModel, usePoints: boolean): Float64Array {
	const cm = m.cm;
	const points: number[][] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v)) points.push([cm.vx(v), cm.vy(v), cm.vz(v)]);
	}
	if (points.length < 3) {
		throw new MLException("Aligning to the principal axes needs at least three points.");
	}

	let axes: ReadonlyArray<readonly number[]>;
	if (usePoints || cm.fn === 0) {
		const centre = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
		axes = symmetricEigen3(covariance(points, centre)).vectors;
	} else {
		axes = symmetricEigen3(Inertia.computeMassProperties(cm).inertiaTensor).vectors;
	}

	// The eigenvectors as rows: that takes world coordinates into the axis
	// frame, which is the direction "align to" means.
	const tr = Matrix44Ops.identity();
	for (let i = 0; i < 3; i++) {
		for (let j = 0; j < 3; j++) tr[4 * i + j] = axes[i][j];
	}
	// An eigenbasis is only defined up to sign, so it can come out
	// left-handed — which would mirror the mesh rather than rotate it.
	if (Matrix44Ops.determinant3(tr) < 0) {
		for (let j = 0; j < 3; j++) tr[8 + j] = -tr[8 + j];
	}
	return tr;
}

/**
 * A rotation bringing the selection's best-fit plane onto a world plane.
 *
 * With `rotAxis` set the rotation is constrained to that one axis, which
 * cannot generally align the plane exactly but preserves whatever the chosen
 * axis meant — a scan that is level but rotated wants exactly that.
 */
function rotateToFitMatrix(
	m: MeshModel,
	targetPlane: number,
	rotAxis: number,
): { matrix: Float64Array; normal: number[]; error: number } {
	const cm = m.cm;
	selectVerticesFromFacesIfNeeded(cm);
	const selected: number[][] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v) && cm.isVertS(v)) selected.push([cm.vx(v), cm.vy(v), cm.vz(v)]);
	}
	if (selected.length < 3) {
		throw new MLException(
			"Cannot compute rotation: at least three selected vertices are needed to fit a plane.",
		);
	}
	const plane = fitPlaneToPointSet(selected);
	if (plane === null) throw new MLException("Cannot compute rotation: the plane fit failed.");
	const normal = [...plane.normal];
	const error =
		selected.reduce(
			(s, p) => s + Math.abs(p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] - plane.offset),
			0,
		) / selected.length;

	const target = [
		[0, 0, 1],
		[1, 0, 0],
		[0, 1, 0],
	][targetPlane];

	let axis: number[];
	let angle: number;
	if (rotAxis === 0) {
		// `cross(target, normal)` with `+acos(target . normal)` is the rotation
		// carrying the *target* onto the normal. What is wanted is the reverse —
		// the plane brought onto the target — hence the negative angle.
		axis = cross3(target, normal);
		angle = -Math.acos(Math.min(1, Math.max(-1, dot3v(target, normal))));
	} else {
		// Constrained: project the normal into the plane perpendicular to the
		// chosen axis, and measure the angle there.
		const k = rotAxis - 1;
		axis = [0, 0, 0];
		axis[k] = -1;
		const projected = [...normal];
		projected[k] = 0;
		const len = Math.hypot(projected[0], projected[1], projected[2]);
		if (len === 0) {
			// The plane already faces along the constrained axis; nothing to do.
			return { matrix: Matrix44Ops.identity(), normal, error };
		}
		for (let i = 0; i < 3; i++) projected[i] /= len;
		angle = -Math.acos(Math.min(1, Math.max(-1, dot3v(target, projected))));
		const sign = dot3v(cross3(target, projected), axis);
		if (sign < 0) angle = -angle;
		else if (sign === 0) angle = 0;
	}

	const axisLen = Math.hypot(axis[0], axis[1], axis[2]);
	if (axisLen === 0 || angle === 0) return { matrix: Matrix44Ops.identity(), normal, error };
	for (let i = 0; i < 3; i++) axis[i] /= axisLen;

	// Rotate about the selection's own centre, not the origin, so the fit does
	// not fling the mesh across the scene.
	const c = plane.centre;
	const toOrigin = Matrix44Ops.translation(-c[0], -c[1], -c[2]);
	const back = Matrix44Ops.translation(c[0], c[1], c[2]);
	const rot = Matrix44Ops.rotation(angle, axis[0], axis[1], axis[2]);
	return {
		matrix: Matrix44Ops.multiply(Matrix44Ops.multiply(back, rot), toOrigin),
		normal,
		error,
	};
}

/** Promotes a face selection to its vertices when no vertex is selected. */
function selectVerticesFromFacesIfNeeded(cm: CMeshO): void {
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v) && cm.isVertS(v)) return;
	}
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] |= VertexFlag.SELECTED;
	}
}

const dot3v = (a: readonly number[], b: readonly number[]): number =>
	a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: readonly number[], b: readonly number[]): number[] => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];
