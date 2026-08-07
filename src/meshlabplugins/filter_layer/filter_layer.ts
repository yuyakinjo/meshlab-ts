/**
 * `filter_layer` — operations on the document rather than on a mesh.
 *
 * These are the only filters so far whose subject is the layer list: adding,
 * removing, renaming and merging. `Flatten Visible Layers` is the one with
 * VARIABLE arity, since it consumes as many meshes as the document holds.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichString } from "../../common/parameters/rich_parameter.ts";
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

export const FP = {
	FP_FLATTEN: 0,
	FP_DUPLICATE: 1,
	FP_DELETE_MESH: 2,
	FP_RENAME_MESH: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly arity: FilterArityValue;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_FLATTEN]: {
		name: "Flatten Visible Layers",
		pythonName: "generate_by_merging_visible_meshes",
		info:
			"Flatten all or only the visible layers into a single new mesh. Transformations are " +
			"preserved. Existing layers can be optionally deleted.",
		arity: FilterArity.VARIABLE,
	},
	[FP.FP_DUPLICATE]: {
		name: "Duplicate Current layer",
		pythonName: "generate_copy_of_current_mesh",
		info: "Create a new layer containing the same model as the current one.",
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_DELETE_MESH]: {
		name: "Delete Current Mesh",
		pythonName: "delete_current_mesh",
		info: "The current mesh layer is deleted.",
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_RENAME_MESH]: {
		name: "Rename Current Mesh",
		pythonName: "set_mesh_name",
		info: "Explicitly change the label shown for a given mesh.",
		arity: FilterArity.SINGLE_MESH,
	},
};

export class FilterLayer extends FilterPlugin {
	pluginName(): string {
		return "FilterLayer";
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
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Layer;
	}
	filterArity(id: ActionIDType): FilterArityValue {
		return this.spec(id).arity;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		switch (id) {
			case FP.FP_FLATTEN:
				list.add(
					new RichBool("MergeVisible", true, {
						description: "Merge Only Visible Layers",
						tooltip:
							"If true, flatten only the visible layers, otherwise, all the layers are used.",
					}),
				);
				list.add(
					new RichBool("DeleteLayer", true, {
						description: "Delete Layers",
						tooltip:
							"Delete all the merged layers. If all layers are visible only a single layer will remain after the invocation of this filter.",
					}),
				);
				list.add(
					new RichBool("MergeVertices", true, {
						description: "Merge duplicate vertices",
						tooltip: "Merge the vertices that are duplicated among different layers.",
					}),
				);
				break;

			case FP.FP_RENAME_MESH:
				list.add(
					new RichString("newName", m === undefined ? "" : m.label(), {
						description: "New Label",
						tooltip: "The new label for the mesh.",
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
		post.mask = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;

		switch (id) {
			case FP.FP_FLATTEN: {
				const onlyVisible = params.getBool("MergeVisible");
				const sources = doc.meshIterator().filter((layer) => !onlyVisible || layer.isVisible());
				if (sources.length === 0) throw new MLException("no layers to flatten");

				const merged = new CMeshO();
				for (const layer of sources) appendMesh(merged, layer.cm);

				if (params.getBool("MergeVertices")) {
					Clean.removeDuplicateVertex(merged);
					Allocator.compactEveryVector(merged);
				}

				const sourceIds = sources.map((s) => s.id());
				const target = doc.addNewMesh("", "Merged Mesh", true, merged);
				target.updateBoxAndNormals();

				if (params.getBool("DeleteLayer")) {
					for (const layerId of sourceIds) doc.delMesh(layerId);
					doc.setCurrentMesh(target.id());
				}

				doc.Log.log(
					`Merged ${sources.length} layers into "${target.label()}": ` +
						`${merged.vn} vertices, ${merged.fn} faces`,
				);
				return { merged_layers: sources.length, vertices: merged.vn, faces: merged.fn };
			}

			case FP.FP_DUPLICATE: {
				const source = doc.mm();
				const copy = new CMeshO();
				appendMesh(copy, source.cm);
				const target = doc.addNewMesh(source.fullName(), `copy of ${source.label()}`, true, copy);
				target.updateBoxAndNormals();
				doc.Log.log(`Duplicated "${source.label()}" as "${target.label()}"`);
				return { new_mesh_id: target.id() };
			}

			case FP.FP_DELETE_MESH: {
				const victim = doc.mm();
				const id_ = victim.id();
				doc.delMesh(id_);
				doc.Log.log(`Deleted layer "${victim.label()}"`);
				// Nothing left to run a postcondition against.
				post.mask = MeshElement.MM_NONE;
				return { deleted_mesh_id: id_ };
			}

			case FP.FP_RENAME_MESH: {
				const m = doc.mm();
				const previous = m.label();
				m.setLabel(params.getString("newName"));
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Renamed "${previous}" to "${m.label()}"`);
				return { name: m.label() };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}

/**
 * Appends `src`'s live geometry onto `dst`, offsetting the face indices.
 *
 * `vcg::tri::Append` in VCGLib. Only the always-present channels are carried
 * over; the optional ones would need the destination to have them enabled,
 * and silently dropping colour is better than silently inventing it.
 */
function appendMesh(dst: CMeshO, src: CMeshO): void {
	const remap = new Int32Array(src.vertSize).fill(-1);
	let live = 0;
	for (let v = 0; v < src.vertSize; v++) if (!src.isVertD(v)) live++;
	if (live === 0) return;

	const firstVert = Allocator.addVertices(dst, live);
	let next = firstVert;
	for (let v = 0; v < src.vertSize; v++) {
		if (src.isVertD(v)) continue;
		remap[v] = next;
		dst.setVert(next, src.vx(v), src.vy(v), src.vz(v));
		dst.vertQuality[next] = src.vertQuality[v];
		dst.vertColor[next] = src.vertColor[v];
		next++;
	}

	for (let f = 0; f < src.faceSize; f++) {
		if (src.isFaceD(f)) continue;
		Allocator.addFace(dst, remap[src.fv(f, 0)], remap[src.fv(f, 1)], remap[src.fv(f, 2)]);
	}
}

export { appendMesh };
