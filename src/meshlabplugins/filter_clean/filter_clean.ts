/**
 * `filter_clean` — the mesh-hygiene filters.
 *
 * Parameter names, defaults and enum option labels are taken verbatim from
 * `src/meshlabplugins/filter_clean/cleanfilter.cpp`, because they are the API:
 * a PyMeshLab recipe passes `MinComponentSize=25`, and anything else here
 * would break it.
 */

import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichEnum,
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
import { ballPivoting } from "../../vcg/complex/create/ball_pivoting.ts";
import { snapVertexBorder } from "../../vcg/complex/snap_border.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTexture } from "../../vcg/complex/update/texture.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

/** Plugin-local action ids, mirroring the C++ enum. */
export const FP = {
	FP_REMOVE_ISOLATED_COMPLEXITY: 0,
	FP_REMOVE_ISOLATED_DIAMETER: 1,
	FP_REMOVE_TVERTEX: 2,
	FP_MERGE_CLOSE_VERTEX: 3,
	FP_REMOVE_DUPLICATE_FACE: 4,
	FP_REMOVE_FOLD_FACE: 5,
	FP_REPAIR_NON_MANIF_EDGE: 6,
	FP_REMOVE_NON_MANIF_VERT: 7,
	FP_REMOVE_UNREFERENCED_VERTEX: 8,
	FP_REMOVE_DUPLICATED_VERTEX: 9,
	FP_REMOVE_FACE_ZERO_AREA: 10,
	FP_BALL_PIVOTING: 11,
	FP_REMOVE_WRT_Q: 12,
	FP_SNAP_MISMATCHED_BORDER: 13,
	FP_MERGE_WEDGE_TEX: 14,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly requirements: number;
}

const GEOMETRY_AND_TOPOLOGY = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_REMOVE_ISOLATED_COMPLEXITY]: {
		name: "Remove Isolated pieces (wrt Face Num.)",
		pythonName: "meshing_remove_connected_component_by_face_number",
		info:
			"Delete isolated connected components composed by a limited number of triangles. " +
			"Useful to remove small floating pieces left over by a reconstruction.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_REMOVE_ISOLATED_DIAMETER]: {
		name: "Remove Isolated pieces (wrt Diameter)",
		pythonName: "meshing_remove_connected_component_by_diameter",
		info:
			"Delete isolated connected components whose diameter is smaller than the specified " +
			"constant.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_REMOVE_TVERTEX]: {
		name: "Remove T-Vertices",
		pythonName: "meshing_remove_t_vertices",
		info:
			"Delete t-vertices from the mesh by either flipping the opposite edge on the degenerate " +
			"face or collapsing the shortest edge. Notice that t-vertices are not a problem for the " +
			"topology but they can be a problem for the rendering and for some algorithms.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_MERGE_CLOSE_VERTEX]: {
		name: "Merge Close Vertices",
		pythonName: "meshing_merge_close_vertices",
		info:
			"Merge together all the vertices that are nearer than the specified threshold. Like a " +
			"'Remove Duplicated Vertices' but with a tolerance.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_REMOVE_DUPLICATE_FACE]: {
		name: "Remove Duplicate Faces",
		pythonName: "meshing_remove_duplicate_faces",
		info:
			"Delete all the duplicate faces. Two faces are considered equal if they are composed by " +
			"the same set of vertices, regardless of the order of the vertices.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_REMOVE_FOLD_FACE]: {
		name: "Remove Isolated Folded Faces by Edge Flip",
		pythonName: "meshing_remove_folded_faces",
		info:
			"Delete isolated folded faces, i.e. faces whose normal is almost opposite to the one of " +
			"an adjacent face, by flipping the shared edge.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_REPAIR_NON_MANIF_EDGE]: {
		name: "Repair non Manifold Edges",
		pythonName: "meshing_repair_non_manifold_edges",
		info:
			"For each non manifold edge it iteratively deletes the smallest area face until it " +
			"becomes 2-manifold, or it splits the vertices so that every sheet becomes a separate " +
			"component.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_REMOVE_NON_MANIF_VERT]: {
		name: "Repair non Manifold Vertices by splitting",
		pythonName: "meshing_repair_non_manifold_vertices",
		info:
			"Split non manifold vertices until it becomes 2-manifold. A vertex is non manifold when " +
			"the faces around it do not form a single fan.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_REMOVE_UNREFERENCED_VERTEX]: {
		name: "Remove Unreferenced Vertices",
		pythonName: "meshing_remove_unreferenced_vertices",
		info: "Check for every vertex on the mesh: if it is not referenced by any face, removes it.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_REMOVE_DUPLICATED_VERTEX]: {
		name: "Remove Duplicate Vertices",
		pythonName: "meshing_remove_duplicate_vertices",
		info:
			"Check for every vertex on the mesh: if there are two vertices with same coordinates " +
			"they are merged into a single one.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_REMOVE_FACE_ZERO_AREA]: {
		name: "Remove Zero Area Faces",
		pythonName: "meshing_remove_null_faces",
		info:
			"Remove null faces, i.e. faces with area equal to zero: the faces with two coincident " +
			"vertices and the faces with three collinear vertices.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_BALL_PIVOTING]: {
		name: "Surface Reconstruction: Ball Pivoting",
		pythonName: "generate_surface_reconstruction_ball_pivoting",
		info:
			"Given a point cloud with normals it reconstructs a surface using the <b>Ball Pivoting " +
			"Algorithm</b>. Starting with a seed triangle, the algorithm pivots a ball around an " +
			"edge until it touches another point, forming another triangle. The process continues " +
			"until all reachable edges have been tried.<br>" +
			"<b>The ball pivoting algorithm for surface reconstruction.</b><br>" +
			"F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, G. Taubin.<br>" +
			"IEEE TVCG 1999",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_VERTFACETOPO,
	},
	[FP.FP_REMOVE_WRT_Q]: {
		name: "Remove Vertices wrt Quality",
		pythonName: "meshing_remove_vertices_by_scalar",
		info:
			"Delete all the vertices with a quality lower than the given threshold, and all the " +
			"faces using them.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_VERTQUALITY,
	},
	[FP.FP_SNAP_MISMATCHED_BORDER]: {
		name: "Snap Mismatched Borders",
		pythonName: "meshing_snap_mismatched_borders",
		info:
			"Try to snap together adjacent borders that are slightly mismatched.<br>This situation " +
			"can happen on badly processed meshes that are the result of the fusion of many range " +
			"maps.",
		filterClass: FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_MERGE_WEDGE_TEX]: {
		name: "Merge Wedge Texture Coord",
		pythonName: "apply_texcoord_merge_per_wedge",
		info:
			"Merge together per-wedge texture coords that are very close. Used to correct real-valued " +
			"approximation errors of texture coordinates.",
		filterClass: FilterClass.Cleaning | FilterClass.Texture,
		requirements: MeshElement.MM_VERTFACETOPO | MeshElement.MM_WEDGTEXCOORD,
	},
};

/** The span of per-vertex quality, or 0..1 when there is nothing to measure. */
function qualityRangeOf(m: MeshModel | undefined): { min: number; max: number } {
	if (m === undefined) return { min: 0, max: 1 };
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (let v = 0; v < m.cm.vertSize; v++) {
		if (m.cm.isVertD(v)) continue;
		min = Math.min(min, m.cm.vertQuality[v]);
		max = Math.max(max, m.cm.vertQuality[v]);
	}
	return Number.isFinite(min) ? { min, max } : { min: 0, max: 1 };
}

export class FilterClean extends FilterPlugin {
	pluginName(): string {
		return "FilterClean";
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

	/**
	 * Every one of these can change both the geometry and the topology, which
	 * is what tells the framework to drop stale adjacency and recompute the
	 * box and normals afterwards.
	 */
	override postCondition(_id: ActionIDType): number {
		return GEOMETRY_AND_TOPOLOGY;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		// Several defaults are relative to the bounding box, so the box has to
		// be current before they are read. A degenerate or empty mesh has a
		// zero diagonal, which would make those defaults zero and the filters
		// silent no-ops; fall back to 1 so they at least behave sensibly.
		let diag = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diag = m.cm.bbox.diagonal || 1;
		}

		switch (id) {
			case FP.FP_REMOVE_ISOLATED_DIAMETER:
				list.add(
					new RichPercentage("MinComponentDiag", diag / 10, 0, diag, {
						description: "Enter max diameter of isolated pieces",
						tooltip:
							"Delete all the connected components (floating pieces) with a diameter smaller " +
							"than the specified one",
					}),
				);
				list.add(
					new RichBool("removeUnref", true, {
						description: "Remove unfreferenced vertices",
						tooltip:
							"if true, the unreferenced vertices remaining after the face deletion are removed.",
					}),
				);
				break;

			case FP.FP_REMOVE_ISOLATED_COMPLEXITY:
				list.add(
					new RichInt("MinComponentSize", 25, {
						description: "Enter minimum conn. comp size:",
						tooltip:
							"Delete all the connected components (floating pieces) composed by a number of " +
							"triangles smaller than the specified one",
					}),
				);
				list.add(
					new RichBool("removeUnref", true, {
						description: "Remove unfreferenced vertices",
						tooltip:
							"if true, the unreferenced vertices remaining after the face deletion are removed.",
					}),
				);
				break;

			case FP.FP_MERGE_CLOSE_VERTEX:
				list.add(
					new RichPercentage("Threshold", diag / 10000, 0, diag / 100, {
						description: "Merging distance",
						tooltip:
							"All the vertices that closer than this threshold are merged together. Use very " +
							"small values, default values is 1/10000 of bounding box diagonal. ",
					}),
				);
				break;

			case FP.FP_REMOVE_TVERTEX:
				list.add(
					new RichEnum("method", 0, ["Edge Collapse", "Edge Flip"], {
						description: "Method",
						tooltip: "Selects whether to remove t-vertices by edge collapse or edge flip.",
					}),
				);
				list.add(
					new RichFloat("Threshold", 40, {
						description: "Ratio",
						tooltip: "Detects faces where the base/height ratio is lower than this value",
					}),
				);
				list.add(
					new RichBool("Repeat", true, {
						description: "Iterate until convergence",
						tooltip: "Iterates the algorithm until it reaches convergence",
					}),
				);
				break;

			case FP.FP_REMOVE_NON_MANIF_VERT:
				list.add(
					new RichFloat("VertDispRatio", 0, {
						description: "Vertex Displacement Ratio",
						tooltip:
							"This parameter denote the ratio of displacement of a vertex. When a vertex is " +
							"split, it is moved towards the barycenter of the FF connected faces sharing it. " +
							"When it is zero the vertex is not displaced. Reasonable values are in the " +
							"[0 .. 0.1] range.",
					}),
				);
				break;

			case FP.FP_REPAIR_NON_MANIF_EDGE:
				list.add(
					new RichEnum("method", 0, ["Remove Faces", "Split Vertices"], {
						description: "Method",
						tooltip:
							"Selects whether to remove non manifold edges by removing faces or by splitting " +
							"vertices.",
					}),
				);
				break;

			case FP.FP_BALL_PIVOTING:
				list.add(
					new RichPercentage("BallRadius", 0, 0, diag, {
						description: "Pivoting Ball radius (0 autoguess)",
						tooltip:
							"The radius of the ball pivoting (rolling) over the set of points. Gaps that are " +
							"larger than the ball radius will not be filled; similarly the small pits that are " +
							"smaller than the ball radius will be filled.",
					}),
				);
				list.add(
					new RichFloat("Clustering", 20, {
						description: "Clustering radius (% of ball radius)",
						tooltip:
							"To avoid the creation of too small triangles, if a vertex is found too close to a " +
							"previous one, it is clustered/merged with it.",
					}),
				);
				list.add(
					new RichFloat("CreaseThr", 90, {
						description: "Angle Threshold (degrees)",
						tooltip:
							"If we encounter a crease angle that is too large we should stop the ball rolling",
					}),
				);
				list.add(
					new RichBool("DeleteFaces", false, {
						description: "Delete initial set of faces",
						tooltip:
							"if true all the initial faces of the mesh are deleted and the whole surface is " +
							"rebuilt from scratch. Otherwise the current faces are used as a starting point. " +
							"Useful if you run the algorithm multiple times with an increasing ball radius.",
					}),
				);
				break;

			case FP.FP_REMOVE_WRT_Q: {
				const range = qualityRangeOf(m);
				list.add(
					new RichPercentage("MaxQualityThr", range.min, range.min, range.max, {
						description: "Delete all vertices with quality under:",
					}),
				);
				break;
			}

			case FP.FP_SNAP_MISMATCHED_BORDER:
				list.add(
					new RichFloat("EdgeDistRatio", 1 / 100, {
						description: "Edge Distance Ratio",
						tooltip:
							"Collapse edge when the edge / distance ratio is greater than this value. Larger " +
							"values enforce that only vertices very close to the line are removed.",
					}),
				);
				list.add(
					new RichBool("UnifyVertices", true, {
						description: "UnifyVertices",
						tooltip: "if true the snap vertices are weld together.",
					}),
				);
				break;

			case FP.FP_MERGE_WEDGE_TEX:
				list.add(
					new RichFloat("MergeThr", 1 / 10000, {
						description: "Merging Threshold",
						tooltip:
							"All the per-wedge texture coords that are on the same vertex and are distant less " +
							"then the given threshold are merged together. It can be used to remove the fake " +
							"texture seams that arise from error. Distance is in texture space (the default, " +
							"1e-4, corresponds to one texel on a 10kx10k texture)",
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
		post.mask = GEOMETRY_AND_TOPOLOGY;

		switch (id) {
			case FP.FP_REMOVE_DUPLICATED_VERTEX: {
				const removed = Clean.removeDuplicateVertex(cm);
				doc.Log.log(`Removed ${removed} duplicated vertices`);
				return this.finish(m, { removedVertices: removed });
			}

			case FP.FP_REMOVE_UNREFERENCED_VERTEX: {
				const removed = Clean.removeUnreferencedVertex(cm);
				doc.Log.log(`Removed ${removed} unreferenced vertices`);
				return this.finish(m, { removedVertices: removed });
			}

			case FP.FP_REMOVE_FACE_ZERO_AREA: {
				const removed = Clean.removeZeroAreaFace(cm);
				doc.Log.log(`Removed ${removed} zero area faces`);
				return this.finish(m, { removedFaces: removed });
			}

			case FP.FP_REMOVE_DUPLICATE_FACE: {
				const removed = Clean.removeDuplicateFace(cm);
				doc.Log.log(`Removed ${removed} duplicate faces`);
				return this.finish(m, { removedFaces: removed });
			}

			case FP.FP_MERGE_CLOSE_VERTEX: {
				const merged = Clean.mergeCloseVertex(cm, params.getAbsPerc("Threshold"));
				doc.Log.log(`Successfully merged ${merged} vertices`);
				return this.finish(m, { mergedVertices: merged });
			}

			case FP.FP_REMOVE_ISOLATED_COMPLEXITY: {
				const { total, deleted } = Clean.removeSmallConnectedComponentsSize(
					cm,
					params.getInt("MinComponentSize"),
				);
				const removedVertices = params.getBool("removeUnref")
					? Clean.removeUnreferencedVertex(cm)
					: 0;
				doc.Log.log(`Removed ${deleted} connected components out of ${total}`);
				return this.finish(m, {
					totalComponents: total,
					deletedComponents: deleted,
					removedVertices,
				});
			}

			case FP.FP_REMOVE_ISOLATED_DIAMETER: {
				const { total, deleted } = Clean.removeSmallConnectedComponentsDiameter(
					cm,
					params.getAbsPerc("MinComponentDiag"),
				);
				const removedVertices = params.getBool("removeUnref")
					? Clean.removeUnreferencedVertex(cm)
					: 0;
				doc.Log.log(`Removed ${deleted} connected components out of ${total}`);
				return this.finish(m, {
					totalComponents: total,
					deletedComponents: deleted,
					removedVertices,
				});
			}

			case FP.FP_REPAIR_NON_MANIF_EDGE: {
				if (params.getEnum("method") === 0) {
					const removed = Clean.removeNonManifoldFace(cm);
					doc.Log.log(`Successfully removed ${removed} non-manifold faces`);
					return this.finish(m, { removedFaces: removed });
				}
				const components = Clean.splitManifoldComponents(cm);
				doc.Log.log(`Successfully split the mesh into ${components} edge manifold components`);
				return this.finish(m, { components });
			}

			case FP.FP_REMOVE_NON_MANIF_VERT: {
				const split = Clean.splitNonManifoldVertex(cm, params.getFloat("VertDispRatio"));
				doc.Log.log(`Split ${split} non-manifold vertices`);
				return this.finish(m, { splitVertices: split });
			}

			case FP.FP_REMOVE_TVERTEX: {
				const ratio = params.getFloat("Threshold");
				const repeat = params.getBool("Repeat");
				const total =
					params.getEnum("method") === 0
						? Clean.removeTVertexByCollapse(cm, ratio, repeat)
						: Clean.removeTVertexByFlip(cm, ratio, repeat);
				doc.Log.log(`Successfully removed ${total} t-vertices`);
				return this.finish(m, { removedTVertices: total });
			}

			case FP.FP_REMOVE_FOLD_FACE: {
				const total = Clean.removeFaceFoldByFlip(cm);
				doc.Log.log(`Successfully flipped ${total} folded faces`);
				return this.finish(m, { flippedFaces: total });
			}

			case FP.FP_BALL_PIVOTING: {
				if (params.getBool("DeleteFaces")) {
					for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f)) Allocator.deleteFace(cm, f);
					Allocator.compactEveryVector(cm);
				}
				const clustering = params.getFloat("Clustering") / 100;
				if (!(clustering > 0) || clustering >= 1) {
					throw new MLException(
						`The clustering radius must be a percentage strictly between 0 and 100, got ${params.getFloat("Clustering")}`,
					);
				}
				const result = ballPivoting(cm, {
					radius: params.getAbsPerc("BallRadius"),
					clustering,
					creaseAngle: (params.getFloat("CreaseThr") * Math.PI) / 180,
					progress: (percent) => cb(percent, "Ball pivoting"),
				});
				m.clearDataMask(MeshElement.MM_FACEFACETOPO);
				doc.Log.log(
					`Reconstructed surface with a ball of radius ${result.radius}: added ${result.addedFaces} faces`,
				);
				return this.finish(m, { added_faces: result.addedFaces, radius: result.radius });
			}

			case FP.FP_REMOVE_WRT_Q: {
				const threshold = params.getAbsPerc("MaxQualityThr");
				let deletedV = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v) || cm.vertQuality[v] >= threshold) continue;
					Allocator.deleteVertex(cm, v);
					deletedV++;
				}
				let deletedF = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					// A face is only as alive as its three vertices.
					if (!cm.isVertD(cm.fv(f, 0)) && !cm.isVertD(cm.fv(f, 1)) && !cm.isVertD(cm.fv(f, 2))) {
						continue;
					}
					Allocator.deleteFace(cm, f);
					deletedF++;
				}
				m.clearDataMask(MeshElement.MM_FACEFACETOPO);
				doc.Log.log(
					`Deleted ${deletedV} vertices and ${deletedF} faces with a quality lower than ${threshold}`,
				);
				return this.finish(m, { deleted_vertices: deletedV, deleted_faces: deletedF });
			}

			case FP.FP_SNAP_MISMATCHED_BORDER: {
				const split = snapVertexBorder(cm, params.getFloat("EdgeDistRatio"));
				let merged = 0;
				if (params.getBool("UnifyVertices")) {
					// The split alone only makes the two borders compatible. Welding
					// is what actually joins them, and it is a separate step because
					// it is the one that can go wrong.
					UpdateBounding.box(cm);
					merged = Clean.mergeCloseVertex(cm, (cm.bbox.diagonal || 1) / 100000);
				}
				m.clearDataMask(MeshElement.MM_FACEFACETOPO | MeshElement.MM_VERTFACETOPO);
				doc.Log.log(`Split ${split} faces to snap, and merged ${merged} vertices`);
				return this.finish(m, { split_faces: split, merged_vertices: merged });
			}

			case FP.FP_MERGE_WEDGE_TEX: {
				const threshold = params.getFloat("MergeThr");
				UpdateTopology.vertexFace(cm);
				const total = UpdateTexture.wedgeTexMergeClose(cm, threshold);
				doc.Log.log(`Merged ${total} wedge texture coords closer than ${threshold}`);
				return { merged: total };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	/**
	 * Reclaims the slots the filter marked deleted and refreshes the box.
	 *
	 * Compaction is deliberately here rather than inside each `Clean`
	 * function: leaving indices stable is what lets several of them run in
	 * sequence, so the filter is the right place to settle up.
	 */
	private finish(m: MeshModel, output: FilterOutput): FilterOutput {
		Allocator.compactEveryVector(m.cm);
		m.updateBoxAndNormals();
		return output;
	}
}
