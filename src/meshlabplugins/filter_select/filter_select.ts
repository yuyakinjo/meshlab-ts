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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import { UpdateFlags } from "../../vcg/complex/update/flag.ts";

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
	override initParameterList(): RichParameterList {
		return new RichParameterList();
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
		_params: RichParameterList,
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
