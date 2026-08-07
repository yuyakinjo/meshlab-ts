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
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";

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
};

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
