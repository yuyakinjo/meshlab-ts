/**
 * `filter_meshing` — orientation, hole closing and decimation.
 *
 * The rest of upstream's 37 filters (isotropic remeshing, subdivision, the
 * transform family) arrive with the later tiers; these four are what a
 * 3D-printing repair pipeline needs.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichFloat,
	RichInt,
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
import { Hole } from "../../vcg/complex/hole.ts";
import {
	defaultQuadricParameters,
	quadricSimplification,
} from "../../vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

export const FP = {
	FP_REORIENT: 0,
	FP_INVERT_FACES: 1,
	FP_CLOSE_HOLES: 2,
	FP_QUADRIC_SIMPLIFICATION: 3,
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
					doc.Log.warning(
						`Could not reach the target of ${targetFaceNum} faces: no further collapse was ` +
							(params.getBool("PreserveTopology")
								? "possible without changing the topology of the mesh."
								: "legal on this mesh."),
					);
				}

				return {
					target_face_num: targetFaceNum,
					initial_faces: result.initialFaces,
					final_faces: cm.fn,
					collapses: result.performed,
					target_reached: cm.fn <= targetFaceNum,
				};
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}
