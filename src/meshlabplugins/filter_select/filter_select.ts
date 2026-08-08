/**
 * `filter_select` — choosing a subset, and deleting it.
 *
 * MeshLab keeps two independent selections, one on vertices and one on faces,
 * and most filters that "work on the selection" mean the face one. The
 * transfer filters move between them, and the two `Delete` filters are what
 * turn a selection into an edit.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichColor,
	RichDynamicFloat,
	RichEnum,
	RichInt,
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
import { triQualityRadii } from "../../vcg/complex/edge_ops.ts";
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateFlags } from "../../vcg/complex/update/flag.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { blue, green, red, rgba } from "../../vcg/space/color4.ts";
import { KdTree } from "../../vcg/space/index/kdtree.ts";

export const FP = {
	FP_SELECT_ALL: 0,
	FP_SELECT_NONE: 1,
	FP_SELECT_INVERT: 2,
	FP_SELECT_BORDER: 3,
	FP_SELECT_FACE_FROM_VERT: 4,
	FP_SELECT_VERT_FROM_FACE: 5,
	FP_SELECT_DELETE_FACE: 6,
	FP_SELECT_DELETE_VERT: 7,
	FP_SELECT_DELETE_FACEVERT: 8,
	CP_SELECT_NON_MANIFOLD_FACE: 9,
	CP_SELECT_NON_MANIFOLD_VERTEX: 10,
	FP_SELECT_DILATE: 11,
	FP_SELECT_ERODE: 12,
	FP_SELECT_BY_VERT_QUALITY: 13,
	FP_SELECT_BY_FACE_QUALITY: 14,
	FP_SELECT_BY_COLOR: 15,
	FP_SELECTBYANGLE: 16,
	FP_SELECT_FACES_BY_EDGE: 17,
	FP_SELECT_CONNECTED: 18,
	FP_SELECT_UGLY: 19,
	FP_SELECT_OUTLIER: 20,
	FP_SELECT_DELETE_ALL_FACE: 21,
	CP_SELFINTERSECT_SELECT: 22,
	CP_SELECT_TEXBORDER: 23,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly requirements: number;
	/** Selecting does not change geometry; deleting does. */
	readonly destructive: boolean;
}

const C = FilterClass;

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_SELECT_ALL]: {
		name: "Select All",
		pythonName: "set_selection_all",
		info: "Select all the faces/vertices of the current mesh.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_NONE]: {
		name: "Select None",
		pythonName: "set_selection_none",
		info: "Clear the current set of selected faces/vertices.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_INVERT]: {
		name: "Invert Selection",
		pythonName: "apply_selection_inverse",
		info: "Inverts the current set of selected faces/vertices.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_BORDER]: {
		name: "Select Border",
		pythonName: "compute_selection_from_mesh_border",
		info: "Select vertices and faces on the boundary.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.FP_SELECT_FACE_FROM_VERT]: {
		name: "Select Faces from Vertices",
		pythonName: "compute_selection_transfer_vertex_to_face",
		info: "Select faces from selected vertices.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_VERT_FROM_FACE]: {
		name: "Select Vertices from Faces",
		pythonName: "compute_selection_transfer_face_to_vertex",
		info: "Select vertices from selected faces.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_DELETE_FACE]: {
		name: "Delete Selected Faces",
		pythonName: "meshing_remove_selected_faces",
		info: "Delete the current set of selected faces; the vertices that remain unreferenced are not deleted.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: true,
	},
	[FP.FP_SELECT_DELETE_VERT]: {
		name: "Delete Selected Vertices",
		pythonName: "meshing_remove_selected_vertices",
		info:
			"Delete the current set of selected vertices; faces that share one of the deleted " +
			"vertices are deleted too.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: true,
	},
	[FP.FP_SELECT_DELETE_FACEVERT]: {
		name: "Delete Selected Faces and Vertices",
		pythonName: "meshing_remove_selected_vertices_and_faces",
		info: "Delete the current set of selected faces and all the vertices surrounded by that faces.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: true,
	},
	[FP.CP_SELECT_NON_MANIFOLD_FACE]: {
		name: "Select non Manifold Edges",
		pythonName: "compute_selection_by_non_manifold_edges_per_face",
		info:
			"Select the faces and the vertices incident on non manifold edges (e.g. edges where " +
			"more than two faces are incident).",
		filterClass: C.Selection,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.CP_SELECT_NON_MANIFOLD_VERTEX]: {
		name: "Select non Manifold Vertices",
		pythonName: "compute_selection_by_non_manifold_per_vertex",
		info:
			"Select the vertices incident on non manifold edges (e.g. edges where more than two " +
			"faces are incident); note that this function select the vertices not the edges.",
		filterClass: C.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_DILATE]: {
		name: "Dilate Selection",
		pythonName: "apply_selection_dilatation",
		info: "Dilate (expand) the current set of selected faces.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.FP_SELECT_ERODE]: {
		name: "Erode Selection",
		pythonName: "apply_selection_erosion",
		info: "Erode (reduce) the current set of selected faces.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_BY_VERT_QUALITY]: {
		name: "Select by Vertex Quality",
		pythonName: "compute_selection_by_scalar_per_vertex",
		info: "Select all the faces/vertices within the specified vertex quality range.",
		filterClass: FilterClass.Selection | FilterClass.Quality,
		requirements: MeshElement.MM_VERTQUALITY,
		destructive: false,
	},
	[FP.FP_SELECT_BY_FACE_QUALITY]: {
		name: "Select by Face Quality",
		pythonName: "compute_selection_by_scalar_per_face",
		info: "Select all the faces/vertices with within the specified face quality range.",
		filterClass: FilterClass.Selection | FilterClass.Quality,
		requirements: MeshElement.MM_FACEQUALITY,
		destructive: false,
	},
	[FP.FP_SELECT_BY_COLOR]: {
		name: "Select Faces by Color",
		pythonName: "compute_selection_by_color_per_face",
		info: "Select part of the mesh based on its color.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_VERTCOLOR,
		destructive: false,
	},
	[FP.FP_SELECTBYANGLE]: {
		name: "Select Faces by view angle",
		pythonName: "compute_selection_by_angle_with_direction_per_face",
		info: "Select faces according to the angle between their normal and the view direction. It is used in range map processing to select and delete steep faces parallel to viewdirection.",
		filterClass: FilterClass.Selection | FilterClass.RangeMap,
		requirements: MeshElement.MM_FACENORMAL,
		destructive: false,
	},
	[FP.FP_SELECT_FACES_BY_EDGE]: {
		name: "Select Faces with edges longer than...",
		pythonName: "compute_selection_by_edge_length",
		info: "Select all triangles having an edge with length greater or equal than a given threshold.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_CONNECTED]: {
		name: "Select Connected Faces",
		pythonName: "apply_selection_by_same_connected_component",
		info: "Expand the current face selection so that it includes all the faces in the connected components where there is at least a selected face.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.FP_SELECT_UGLY]: {
		name: "Select 'problematic' faces",
		pythonName: "compute_selection_bad_faces",
		info: "Select faces with 'problems', like normal inverted w.r.t the surrounding areas, extremely elongated or folded.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.FP_SELECT_OUTLIER]: {
		name: "Select Outliers",
		pythonName: "compute_selection_point_cloud_outliers",
		info: "Select the vertex classified as outlier using Local Outlier Propabilty measure described in:<br> <b>'LoOP: Local Outlier Probabilities'</b> Kriegel et al.<br>CIKM 2009Unknown filter",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: false,
	},
	[FP.FP_SELECT_DELETE_ALL_FACE]: {
		name: "Delete ALL Faces",
		pythonName: "meshing_remove_all_faces",
		info: "Delete ALL faces, turning the mesh into a pointcloud. May be applied also to all visible layers.",
		filterClass: FilterClass.Selection,
		requirements: MeshElement.MM_NONE,
		destructive: true,
	},
	[FP.CP_SELFINTERSECT_SELECT]: {
		name: "Select Self Intersecting Faces",
		pythonName: "compute_selection_by_self_intersections_per_face",
		info: "Select only self intersecting faces.",
		filterClass: FilterClass.Selection | FilterClass.Cleaning,
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
	[FP.CP_SELECT_TEXBORDER]: {
		name: "Select Vertex Texture Seams",
		pythonName: "compute_selection_by_texture_seams_per_vertex",
		info:
			"Colorize only border edges of the texture atlas, in order to check the correctness of the " +
			"parametrization.",
		filterClass: FilterClass.Selection | FilterClass.Texture,
		// Deliberately not requiring MM_WEDGTEXCOORD: the framework would
		// allocate zeroed coordinates for a mesh that has none, every face would
		// then agree with its neighbour, and the filter would report "no seams"
		// for a mesh that has no parametrization at all. Asking for the channel
		// itself lets it say which of the two it found.
		requirements: MeshElement.MM_FACEFACETOPO,
		destructive: false,
	},
};

function countSelectedFaces(m: CMeshO): number {
	let n = 0;
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f) && m.isFaceS(f)) n++;
	return n;
}

function countSelectedVerts(m: CMeshO): number {
	let n = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v) && m.isVertS(v)) n++;
	return n;
}

export class FilterSelect extends FilterPlugin {
	pluginName(): string {
		return "FilterSelect";
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
	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const inclusive = (tooltip: string) =>
			list.add(new RichBool("Inclusive", true, { description: "Inclusive Sel.", tooltip }));

		switch (id) {
			case FP.FP_SELECT_BY_VERT_QUALITY:
			case FP.FP_SELECT_BY_FACE_QUALITY: {
				// The defaults span whatever the mesh currently holds, so the
				// filter opens selecting everything rather than nothing.
				const range = qualityRange(m, id === FP.FP_SELECT_BY_FACE_QUALITY);
				list.add(
					new RichDynamicFloat("minQ", range.min, range.min, range.max, {
						description: "Min Quality",
						tooltip: "Minimum value of the range.",
					}),
				);
				list.add(
					new RichDynamicFloat("maxQ", range.max, range.min, range.max, {
						description: "Max Quality",
						tooltip: "Maximum value of the range.",
					}),
				);
				inclusive(
					"If true only the faces with <b>all</b> the vertices within the specified range are " +
						"selected. Otherwise any face with at least one vertex within the range is selected.",
				);
				break;
			}

			case FP.FP_SELECT_BY_COLOR:
				list.add(
					new RichColor("Color", rgba(0, 0, 0), {
						description: "Color To Select",
						tooltip: "Color that you want to be selected.",
					}),
				);
				list.add(
					new RichEnum("ColorSpace", 0, ["HSV", "RGB"], {
						description: "Pick Color Space",
						tooltip: "The color space that the sliders will manipulate.",
					}),
				);
				inclusive(
					"If true only the faces with <b>all</b> the vertices within the specified range are " +
						"selected. Otherwise any face with at least one vertex within the range is selected.",
				);
				for (const [name, description] of [
					["PercentRH", "Variation from Red or Hue"],
					["PercentGS", "Variation from Green or Saturation"],
					["PercentBV", "Variation from Blue or Value"],
				] as const) {
					list.add(
						new RichDynamicFloat(name, 0.2, 0, 1, {
							description,
							tooltip: "A float between 0 and 1 that represents the percentage of variation.",
						}),
					);
				}
				break;

			case FP.FP_SELECTBYANGLE:
				list.add(
					new RichDynamicFloat("anglelimit", 75, 0, 180, {
						description: "angle threshold (deg)",
						tooltip: "faces with normal at higher angle w.r.t. the view direction are selected",
					}),
				);
				list.add(
					new RichBool("usecamera", false, {
						description: "Use ViewPoint from Mesh Camera",
						tooltip:
							"Uses the ViewPoint from a mesh camera, otherwise the following parameter is used",
					}),
				);
				list.add(
					new RichPosition("viewpoint", [0, 0, 0], {
						description: "ViewPoint",
						tooltip: "if UseCamera is true, this value is ignored",
					}),
				);
				break;

			case FP.FP_SELECT_FACES_BY_EDGE: {
				const diag = boundingDiagonal(m);
				list.add(
					new RichDynamicFloat("Threshold", diag * 0.01, 0, diag, {
						description: "Edge Threshold",
						tooltip:
							"All the faces with an edge <b>longer</b> than this threshold will be selected.",
					}),
				);
				break;
			}

			case FP.FP_SELECT_UGLY:
				list.add(
					new RichBool("useAR", true, {
						description: "Select by Aspect Ratio",
						tooltip: "if true, faces with aspect ratio below the limit will be selected",
					}),
				);
				list.add(
					new RichDynamicFloat("ARatio", 0.02, 0, 1, {
						description: "Aspect Ratio",
						tooltip:
							"Triangle face aspect ratio [1 (equilateral) - 0 (line)]: face will be selected if BELOW this threshold",
					}),
				);
				list.add(
					new RichBool("useNF", false, {
						description: "Select by Normal Angle",
						tooltip:
							"if true, adjacent faces with normals forming an angle above the limit will be selected",
					}),
				);
				list.add(
					new RichDynamicFloat("NFRatio", 60, 0, 180, {
						description: "Angle flip",
						tooltip:
							"angle between the adjacent faces: face will be selected if ABOVE this threshold",
					}),
				);
				list.add(
					new RichBool("select_folded_faces", false, {
						description: "Select folded faces",
						tooltip: "If true, folded faces created by the Marching Cube method will be selected.",
					}),
				);
				list.add(
					new RichDynamicFloat("folded_faces_angle_threshold", 160, 90, 180, {
						description: "Folded Faces Angle Threshold",
						tooltip:
							"Consider the adjacent faces of a face: if the angle between the normal of the " +
							"face and the normals of the adjacent faces is above this threshold, the face is " +
							"selected.",
					}),
				);
				break;

			case FP.FP_SELECT_OUTLIER:
				list.add(
					new RichDynamicFloat("PropThreshold", 0.8, 0, 1, {
						description: "Probability",
						tooltip:
							"Threshold to select the vertex. The vertex is selected if the LoOP value is " +
							"above the threshold.",
					}),
				);
				list.add(
					new RichInt("KNearest", 32, {
						description: "Number of neighbors",
						tooltip: "Number of neighbours used to compute the LoOP",
					}),
				);
				break;

			case FP.FP_SELECT_DELETE_ALL_FACE:
				list.add(
					new RichBool("allLayers", false, {
						description: "Apply to all visible Layers",
						tooltip: "If selected, the filter will be applied to all visible mesh layers.",
					}),
				);
				break;

			default:
				break;
		}
		return list;
	}

	/**
	 * A pure selection change touches only the flag bits, so it must not
	 * trigger a bounding-box or normal recompute — and, more importantly, must
	 * not invalidate the adjacency the caller may be mid-way through using.
	 */
	override postCondition(id: ActionIDType): number {
		return this.spec(id).destructive
			? MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE
			: MeshElement.MM_VERTFLAGSELECT | MeshElement.MM_FACEFLAGSELECT;
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
		post.mask = this.postCondition(id);

		switch (id) {
			case FP.FP_SELECT_ALL: {
				for (let v = 0; v < cm.vertSize; v++) {
					if (!cm.isVertD(v)) cm.vertFlags[v] |= VertexFlag.SELECTED;
				}
				for (let f = 0; f < cm.faceSize; f++) {
					if (!cm.isFaceD(f)) cm.faceFlags[f] |= FaceFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_NONE: {
				UpdateFlags.vertexClearS(cm);
				UpdateFlags.faceClearS(cm);
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_INVERT: {
				for (let v = 0; v < cm.vertSize; v++) {
					if (!cm.isVertD(v)) cm.vertFlags[v] ^= VertexFlag.SELECTED;
				}
				for (let f = 0; f < cm.faceSize; f++) {
					if (!cm.isFaceD(f)) cm.faceFlags[f] ^= FaceFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_BORDER: {
				UpdateFlags.faceBorderFromNone(cm);
				UpdateFlags.vertexBorderFromNone(cm);
				UpdateFlags.vertexClearS(cm);
				UpdateFlags.faceClearS(cm);
				for (let v = 0; v < cm.vertSize; v++) {
					if (!cm.isVertD(v) && cm.isVertB(v)) cm.vertFlags[v] |= VertexFlag.SELECTED;
				}
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					if ((cm.faceFlags[f] & FaceFlag.BORDER012) !== 0) {
						cm.faceFlags[f] |= FaceFlag.SELECTED;
					}
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_FACE_FROM_VERT: {
				// Strict: every corner must be selected, matching MeshLab.
				UpdateFlags.faceClearS(cm);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					let all = true;
					for (let k = 0; k < 3 && all; k++) if (!cm.isVertS(cm.fv(f, k))) all = false;
					if (all) cm.faceFlags[f] |= FaceFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_VERT_FROM_FACE: {
				UpdateFlags.vertexClearS(cm);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] |= VertexFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_DELETE_FACE: {
				let removed = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					Allocator.deleteFace(cm, f);
					removed++;
				}
				Allocator.compactEveryVector(cm);
				m.updateBoxAndNormals();
				doc.Log.log(`Deleted ${removed} faces`);
				return { removed_faces: removed };
			}

			case FP.FP_SELECT_DELETE_VERT: {
				// A face cannot outlive its vertices, so they go first.
				let removedFaces = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					let touches = false;
					for (let k = 0; k < 3 && !touches; k++) if (cm.isVertS(cm.fv(f, k))) touches = true;
					if (touches) {
						Allocator.deleteFace(cm, f);
						removedFaces++;
					}
				}
				let removedVerts = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v) || !cm.isVertS(v)) continue;
					Allocator.deleteVertex(cm, v);
					removedVerts++;
				}
				Allocator.compactEveryVector(cm);
				m.updateBoxAndNormals();
				doc.Log.log(`Deleted ${removedVerts} vertices and ${removedFaces} faces`);
				return { removed_vertices: removedVerts, removed_faces: removedFaces };
			}

			case FP.FP_SELECT_DELETE_FACEVERT: {
				let removedFaces = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					Allocator.deleteFace(cm, f);
					removedFaces++;
				}
				// Then whatever the deletion left stranded.
				const removedVerts = Clean.removeUnreferencedVertex(cm);
				Allocator.compactEveryVector(cm);
				m.updateBoxAndNormals();
				doc.Log.log(`Deleted ${removedFaces} faces and ${removedVerts} vertices`);
				return { removed_faces: removedFaces, removed_vertices: removedVerts };
			}

			case FP.CP_SELECT_NON_MANIFOLD_FACE: {
				UpdateFlags.faceClearS(cm);
				UpdateFlags.vertexClearS(cm);
				const count = Clean.countNonManifoldEdgeFF(cm, true);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] |= VertexFlag.SELECTED;
				}
				doc.Log.log(`Selected the faces on ${count} non-manifold edges`);
				return { ...this.report(doc, cm), non_manifold_edges: count };
			}

			case FP.CP_SELECT_NON_MANIFOLD_VERTEX: {
				UpdateFlags.vertexClearS(cm);
				// A vertex is non-manifold when its incident faces do not form
				// a single fan; the shared helper decides that, and here we
				// only need to mark the ones it finds.
				const incident: number[][] = Array.from({ length: cm.vertSize }, () => []);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					for (let k = 0; k < 3; k++) incident[cm.faceVert[3 * f + k]].push(f);
				}
				let count = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v) || incident[v].length < 2) continue;
					if (fanCount(cm, v, incident[v]) > 1) {
						cm.vertFlags[v] |= VertexFlag.SELECTED;
						count++;
					}
				}
				doc.Log.log(`Selected ${count} non-manifold vertices`);
				return { ...this.report(doc, cm), non_manifold_vertices: count };
			}

			case FP.FP_SELECT_DILATE:
				// Loose both ways: a vertex counts if any of its faces is
				// selected, and a face counts if any of its vertices is.
				vertexFromFaceLoose(cm);
				faceFromVertexLoose(cm);
				return this.report(doc, cm);

			case FP.FP_SELECT_ERODE:
				// Strict both ways, which is exactly the dual and so exactly
				// undoes a dilation of the interior.
				vertexFromFaceStrict(cm);
				faceFromVertexStrict(cm);
				return this.report(doc, cm);

			case FP.FP_SELECT_BY_VERT_QUALITY:
			case FP.FP_SELECT_BY_FACE_QUALITY: {
				const minQ = params.getDynamicFloat("minQ");
				const maxQ = params.getDynamicFloat("maxQ");
				const inclusive = params.getBool("Inclusive");
				if (id === FP.FP_SELECT_BY_FACE_QUALITY) {
					const q = cm.faceQuality;
					if (q === null) {
						throw new MLException("This filter needs per-face quality, which the mesh lacks.");
					}
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						if (q[f] >= minQ && q[f] <= maxQ) cm.faceFlags[f] |= FaceFlag.SELECTED;
						else cm.faceFlags[f] &= ~FaceFlag.SELECTED;
					}
					vertexFromFaceLoose(cm);
				} else {
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v)) continue;
						const q = cm.vertQuality[v];
						if (q >= minQ && q <= maxQ) cm.vertFlags[v] |= VertexFlag.SELECTED;
						else cm.vertFlags[v] &= ~VertexFlag.SELECTED;
					}
					if (inclusive) faceFromVertexStrict(cm);
					else faceFromVertexLoose(cm);
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_BY_COLOR: {
				const target = params.getColor("Color");
				const useHsv = params.getEnum("ColorSpace") === 0;
				const tolerance = [
					params.getDynamicFloat("PercentRH"),
					params.getDynamicFloat("PercentGS"),
					params.getDynamicFloat("PercentBV"),
				];
				// Everything is compared in 0..1, and in HSV the hue is
				// normalised into that range too rather than left in degrees.
				const wanted = useHsv
					? rgbToHsv(red(target) / 255, green(target) / 255, blue(target) / 255)
					: [red(target) / 255, green(target) / 255, blue(target) / 255];

				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					const c = cm.vertColor[v];
					const got = useHsv
						? rgbToHsv(red(c) / 255, green(c) / 255, blue(c) / 255)
						: [red(c) / 255, green(c) / 255, blue(c) / 255];
					let inside = true;
					for (let k = 0; k < 3; k++) {
						if (Math.abs(got[k] - wanted[k]) > tolerance[k]) inside = false;
					}
					if (inside) cm.vertFlags[v] |= VertexFlag.SELECTED;
					else cm.vertFlags[v] &= ~VertexFlag.SELECTED;
				}
				if (params.getBool("Inclusive")) faceFromVertexStrict(cm);
				else faceFromVertexLoose(cm);
				return this.report(doc, cm);
			}

			case FP.FP_SELECTBYANGLE: {
				if (params.getBool("usecamera")) {
					throw new MLNotImplementedException(
						"This mesh carries no camera, and cameras are not loaded yet; give a view point instead.",
						"FilterSelect",
					);
				}
				const eye = params.getPoint3m("viewpoint");
				const limit = Math.cos((params.getDynamicFloat("anglelimit") * Math.PI) / 180);
				UpdateNormal.perFaceNormalized(cm);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					const a = cm.fv(f, 0);
					const b = cm.fv(f, 1);
					const c = cm.fv(f, 2);
					const ray = [
						(cm.vx(a) + cm.vx(b) + cm.vx(c)) / 3 - eye[0],
						(cm.vy(a) + cm.vy(b) + cm.vy(c)) / 3 - eye[1],
						(cm.vz(a) + cm.vz(b) + cm.vz(c)) / 3 - eye[2],
					];
					const length = Math.hypot(ray[0], ray[1], ray[2]) || 1;
					const dot =
						(ray[0] * cm.faceNormal[3 * f] +
							ray[1] * cm.faceNormal[3 * f + 1] +
							ray[2] * cm.faceNormal[3 * f + 2]) /
						length;
					// A face turned away from the eye has a small dot product;
					// upstream adds to the selection rather than replacing it.
					if (dot < limit) cm.faceFlags[f] |= FaceFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_FACES_BY_EDGE: {
				const threshold = params.getDynamicFloat("Threshold");
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					let longest = 0;
					for (let k = 0; k < 3; k++) {
						const a = cm.fv(f, k);
						const b = cm.fv(f, (k + 1) % 3);
						longest = Math.max(
							longest,
							Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)),
						);
					}
					if (longest >= threshold) cm.faceFlags[f] |= FaceFlag.SELECTED;
					else cm.faceFlags[f] &= ~FaceFlag.SELECTED;
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_CONNECTED: {
				// Flood from every already-selected face over FF adjacency, so
				// a single selected triangle pulls in its whole component.
				UpdateTopology.faceFace(cm);
				const stack: number[] = [];
				for (let f = 0; f < cm.faceSize; f++) {
					if (!cm.isFaceD(f) && cm.isFaceS(f)) stack.push(f);
				}
				const seen = new Set<number>(stack);
				while (stack.length > 0) {
					const f = stack.pop() as number;
					cm.faceFlags[f] |= FaceFlag.SELECTED;
					for (let e = 0; e < 3; e++) {
						if (cm.isBorderFF(f, e)) continue;
						const g = cm.ffp(f, e);
						if (g < 0 || cm.isFaceD(g) || seen.has(g)) continue;
						seen.add(g);
						stack.push(g);
					}
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_UGLY: {
				for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] &= ~FaceFlag.SELECTED;
				UpdateNormal.perFaceNormalized(cm);

				if (params.getBool("useAR")) {
					const limit = params.getDynamicFloat("ARatio");
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						const a = cm.fv(f, 0);
						const b = cm.fv(f, 1);
						const c = cm.fv(f, 2);
						const quality = triQualityRadii(
							cm.vx(a),
							cm.vy(a),
							cm.vz(a),
							cm.vx(b),
							cm.vy(b),
							cm.vz(b),
							cm.vx(c),
							cm.vy(c),
							cm.vz(c),
						);
						if (quality < limit) cm.faceFlags[f] |= FaceFlag.SELECTED;
					}
				}

				if (params.getBool("useNF") || params.getBool("select_folded_faces")) {
					UpdateTopology.faceFace(cm);
				}
				if (params.getBool("useNF")) {
					const limit = params.getDynamicFloat("NFRatio");
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						let worst = 0;
						for (let e = 0; e < 3; e++) {
							if (cm.isBorderFF(f, e)) continue;
							const g = cm.ffp(f, e);
							if (g < 0 || cm.isFaceD(g)) continue;
							const dot =
								cm.faceNormal[3 * f] * cm.faceNormal[3 * g] +
								cm.faceNormal[3 * f + 1] * cm.faceNormal[3 * g + 1] +
								cm.faceNormal[3 * f + 2] * cm.faceNormal[3 * g + 2];
							const angle = (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
							worst = Math.max(worst, angle);
						}
						if (worst > limit) cm.faceFlags[f] |= FaceFlag.SELECTED;
					}
				}

				if (params.getBool("select_folded_faces")) {
					// A face that disagrees with the *average* of the faces
					// around its own vertices has folded back on itself, which
					// a pairwise edge test can miss on a thin fold.
					const limit = Math.cos(
						(params.getDynamicFloat("folded_faces_angle_threshold") * Math.PI) / 180,
					);
					const incident: number[][] = Array.from({ length: cm.vertSize }, () => []);
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						for (let k = 0; k < 3; k++) incident[cm.fv(f, k)].push(f);
					}
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						let x = 0;
						let y = 0;
						let z = 0;
						const ring = new Set<number>();
						for (let k = 0; k < 3; k++) for (const g of incident[cm.fv(f, k)]) ring.add(g);
						ring.delete(f);
						if (ring.size === 0) continue;
						for (const g of ring) {
							x += cm.faceNormal[3 * g];
							y += cm.faceNormal[3 * g + 1];
							z += cm.faceNormal[3 * g + 2];
						}
						const length = Math.hypot(x, y, z);
						if (length === 0) continue;
						const dot =
							(cm.faceNormal[3 * f] * x +
								cm.faceNormal[3 * f + 1] * y +
								cm.faceNormal[3 * f + 2] * z) /
							length;
						if (dot < limit) cm.faceFlags[f] |= FaceFlag.SELECTED;
					}
				}
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_OUTLIER: {
				const k = params.getInt("KNearest");
				if (k < 1) throw new MLException(`The neighbour count must be at least 1, got ${k}`);
				const selected = localOutlierProbability(cm, k, params.getDynamicFloat("PropThreshold"));
				doc.Log.log(`Selected ${selected} outlier vertices`);
				return this.report(doc, cm);
			}

			case FP.FP_SELECT_DELETE_ALL_FACE: {
				const targets = params.getBool("allLayers") ? doc.visibleMeshes() : [m];
				let deleted = 0;
				for (const target of targets) {
					for (let f = 0; f < target.cm.faceSize; f++) {
						if (target.cm.isFaceD(f)) continue;
						Allocator.deleteFace(target.cm, f);
						deleted++;
					}
					UpdateTopology.clearFaceFace(target.cm);
					target.updateBoxAndNormals();
				}
				doc.Log.log(`Deleted all ${deleted} faces, leaving a point cloud`);
				return { deleted_faces: deleted };
			}

			case FP.CP_SELFINTERSECT_SELECT: {
				const hits = Clean.selfIntersections(cm);
				for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] &= ~FaceFlag.SELECTED;
				for (const f of hits) cm.faceFlags[f] |= FaceFlag.SELECTED;
				doc.Log.log(`Selected ${hits.length} self-intersecting faces`);
				return { selected_faces: hits.length };
			}

			case FP.CP_SELECT_TEXBORDER: {
				if (cm.wedgeTexCoord === null) {
					throw new MLException(
						"Select Vertex Texture Seams needs per-wedge texture coordinates, which this mesh " +
							"does not carry.",
					);
				}
				// A seam is a border of the *texture* topology, so building that
				// adjacency turns the question into the one the border machinery
				// already answers.
				UpdateTopology.faceFaceFromTexCoord(cm);
				UpdateFlags.faceBorderFromFF(cm);
				UpdateFlags.vertexBorderFromFace(cm);
				let selected = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					if ((cm.vertFlags[v] & VertexFlag.BORDER) !== 0) {
						cm.vertFlags[v] |= VertexFlag.SELECTED;
						selected++;
					} else cm.vertFlags[v] &= ~VertexFlag.SELECTED;
				}
				// Put the ordinary topology back: leaving the mesh believing its
				// charts are its components would break every later filter.
				UpdateTopology.faceFace(cm);
				UpdateFlags.faceBorderFromFF(cm);
				UpdateFlags.vertexBorderFromFace(cm);
				doc.Log.log(`Selected ${selected} vertices on a texture seam`);
				return { selected_vertices: selected };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	private report(doc: MeshDocument, cm: CMeshO): FilterOutput {
		const verts = countSelectedVerts(cm);
		const faces = countSelectedFaces(cm);
		doc.Log.log(`Selected ${verts} vertices and ${faces} faces`);
		return { selected_vertices: verts, selected_faces: faces };
	}
}

/** How many separate fans the faces around `v` form. */
function fanCount(m: CMeshO, v: number, faces: readonly number[]): number {
	const index = new Map<number, number>(faces.map((f, i) => [f, i]));
	const parent = faces.map((_, i) => i);
	const find = (x: number): number => {
		let r = x;
		while (parent[r] !== r) r = parent[r];
		let cur = x;
		while (parent[cur] !== r) {
			const nxt = parent[cur];
			parent[cur] = r;
			cur = nxt;
		}
		return r;
	};
	const owner = new Map<number, number>();
	for (const f of faces) {
		for (let k = 0; k < 3; k++) {
			const a = m.faceVert[3 * f + k];
			const b = m.faceVert[3 * f + ((k + 1) % 3)];
			if (a !== v && b !== v) continue;
			const other = a === v ? b : a;
			const me = index.get(f) as number;
			const seen = owner.get(other);
			if (seen === undefined) owner.set(other, me);
			else {
				const ra = find(seen);
				const rb = find(me);
				if (ra !== rb) parent[ra] = rb;
			}
		}
	}
	return new Set(faces.map((_, i) => find(i))).size;
}

/** Whatever range the mesh's quality currently spans, for the slider defaults. */
function qualityRange(m: MeshModel | undefined, perFace: boolean): { min: number; max: number } {
	if (m === undefined) return { min: 0, max: 1 };
	const cm = m.cm;
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	if (perFace) {
		const q = cm.faceQuality;
		if (q === null) return { min: 0, max: 1 };
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			min = Math.min(min, q[f]);
			max = Math.max(max, q[f]);
		}
	} else {
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			min = Math.min(min, cm.vertQuality[v]);
			max = Math.max(max, cm.vertQuality[v]);
		}
	}
	return Number.isFinite(min) ? { min, max } : { min: 0, max: 1 };
}

function boundingDiagonal(m: MeshModel | undefined): number {
	if (m === undefined) return 1;
	UpdateBounding.box(m.cm);
	return m.cm.bbox.diagonal || 1;
}

/** RGB in 0..1 to HSV, with hue also normalised into 0..1. */
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
	const max = Math.max(r, g, b);
	const min = Math.min(r, g, b);
	const span = max - min;
	let h = 0;
	if (span !== 0) {
		if (max === r) h = ((g - b) / span) % 6;
		else if (max === g) h = (b - r) / span + 2;
		else h = (r - g) / span + 4;
		h /= 6;
		if (h < 0) h += 1;
	}
	return [h, max === 0 ? 0 : span / max, max];
}

/** Marks every vertex of a selected face — the "loose" direction. */
function vertexFromFaceLoose(cm: CMeshO): void {
	for (let v = 0; v < cm.vertSize; v++) cm.vertFlags[v] &= ~VertexFlag.SELECTED;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] |= VertexFlag.SELECTED;
	}
}

/** Marks a vertex only when *every* face using it is selected — the "strict" direction. */
function vertexFromFaceStrict(cm: CMeshO): void {
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v)) cm.vertFlags[v] |= VertexFlag.SELECTED;
	}
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f) || cm.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] &= ~VertexFlag.SELECTED;
	}
}

/** Marks a face when any of its vertices is selected. */
function faceFromVertexLoose(cm: CMeshO): number {
	let n = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		let any = false;
		for (let k = 0; k < 3; k++) if (cm.isVertS(cm.fv(f, k))) any = true;
		if (any) {
			cm.faceFlags[f] |= FaceFlag.SELECTED;
			n++;
		} else cm.faceFlags[f] &= ~FaceFlag.SELECTED;
	}
	return n;
}

/** Marks a face only when all three of its vertices are selected. */
function faceFromVertexStrict(cm: CMeshO): number {
	let n = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		let all = true;
		for (let k = 0; k < 3; k++) if (!cm.isVertS(cm.fv(f, k))) all = false;
		if (all) {
			cm.faceFlags[f] |= FaceFlag.SELECTED;
			n++;
		} else cm.faceFlags[f] &= ~FaceFlag.SELECTED;
	}
	return n;
}

/**
 * Local Outlier Probability, as Kriegel et al. define it.
 *
 * Each point gets a "probabilistic set distance" from its k neighbours — the
 * quadratic mean of those distances — and is then scored by how much larger
 * its own is than its neighbours' average. The score runs 0..1 and is
 * comparable between points, which a raw distance is not: a sparse region is
 * not automatically full of outliers.
 */
function localOutlierProbability(cm: CMeshO, k: number, threshold: number): number {
	const live: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) live.push(v);
	if (live.length <= k) return 0;

	const tree = new KdTree(cm.vertCoord, cm.vertSize);
	const neighbours = new Map<number, Int32Array>();
	const sigma = new Map<number, number>();
	for (const v of live) {
		// `nearest` includes the point itself, so ask for one more.
		const found = tree.nearest(v, k + 1);
		neighbours.set(v, found);
		let sum = 0;
		let count = 0;
		for (const w of found) {
			if (w === v) continue;
			sum += (cm.vx(w) - cm.vx(v)) ** 2 + (cm.vy(w) - cm.vy(v)) ** 2 + (cm.vz(w) - cm.vz(v)) ** 2;
			count++;
		}
		sigma.set(v, count === 0 ? 0 : Math.sqrt(sum / count));
	}

	const plof = new Map<number, number>();
	let squaredMean = 0;
	for (const v of live) {
		const found = neighbours.get(v) as Int32Array;
		let sum = 0;
		let count = 0;
		for (const w of found) {
			if (w === v) continue;
			sum += sigma.get(w) ?? 0;
			count++;
		}
		const expected = count === 0 ? 0 : sum / count;
		const value = expected === 0 ? 0 : (sigma.get(v) as number) / expected - 1;
		plof.set(v, value);
		squaredMean += value * value;
	}
	squaredMean = Math.sqrt(squaredMean / live.length);

	let selected = 0;
	for (const v of live) {
		// The error function turns the ratio into a probability; nLoOP is the
		// normalisation that makes 0.8 mean the same thing on any point set.
		const score =
			squaredMean === 0
				? 0
				: Math.max(0, erf((plof.get(v) as number) / (squaredMean * Math.SQRT2)));
		if (score > threshold) {
			cm.vertFlags[v] |= VertexFlag.SELECTED;
			selected++;
		} else cm.vertFlags[v] &= ~VertexFlag.SELECTED;
	}
	return selected;
}

/** Abramowitz and Stegun 7.1.26, good to about 1e-7 — plenty for a threshold test. */
function erf(x: number): number {
	const sign = x < 0 ? -1 : 1;
	const t = 1 / (1 + 0.3275911 * Math.abs(x));
	const y =
		1 -
		((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
			t *
			Math.exp(-x * x);
	return sign * y;
}
