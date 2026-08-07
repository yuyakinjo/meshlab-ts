/**
 * `filter_layer` — operations on the document rather than on a mesh.
 *
 * These are the only filters so far whose subject is the layer list: adding,
 * removing, renaming and merging. `Flatten Visible Layers` is the one with
 * VARIABLE arity, since it consumes as many meshes as the document holds.
 */
import { readFileSync, writeFileSync } from "node:fs";
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { imageSizeOf } from "../../common/ml_document/raster_model.ts";
import {
	RichBool,
	RichEnum,
	RichFileOpen,
	RichString,
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
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import {
	applyShot,
	readAgisoftXml,
	readBundlerOut,
	writeAgisoftXml,
	writeBundlerOut,
} from "./cameras.ts";

export const FP = {
	FP_FLATTEN: 0,
	FP_DUPLICATE: 1,
	FP_DELETE_MESH: 2,
	FP_RENAME_MESH: 3,
	FP_DELETE_NON_VISIBLE_MESH: 4,
	FP_SPLITSELECTEDFACES: 5,
	FP_SPLITSELECTEDVERTICES: 6,
	FP_SPLITCONNECTED: 7,
	FP_DELETE_RASTER: 8,
	FP_DELETE_NON_SELECTED_RASTER: 9,
	FP_RENAME_RASTER: 10,
	FP_EXPORT_CAMERAS: 11,
	FP_IMPORT_CAMERAS: 12,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly arity: FilterArityValue;
	/** Defaults to {@link FilterClass.Layer}; the raster filters differ. */
	readonly filterClass?: FilterClassMask;
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
	[FP.FP_DELETE_NON_VISIBLE_MESH]: {
		name: "Delete all non visible Mesh Layers",
		pythonName: "delete_non_visible_meshes",
		info: "All the non visible mesh layers are deleted",
		arity: FilterArity.VARIABLE,
	},
	[FP.FP_SPLITSELECTEDFACES]: {
		name: "Move selected faces to another layer",
		pythonName: "generate_from_selected_faces",
		info:
			"Selected faces are moved (or duplicated) in a new layer. Warning! per-vertex and " +
			"per-face user defined attributes will not be transferred.",
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_SPLITSELECTEDVERTICES]: {
		name: "Move selected vertices to another layer",
		pythonName: "generate_from_selected_vertices",
		info:
			"Selected vertices are moved (or duplicated) in a new layer. Warning! per-vertex user " +
			"defined attributes will not be transferred.",
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_SPLITCONNECTED]: {
		name: "Split in Connected Components",
		pythonName: "generate_splitting_by_connected_components",
		info: "Split current Layer into many layers, one for each connected components",
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.FP_DELETE_RASTER]: {
		name: "Delete Current Raster",
		pythonName: "delete_current_raster",
		info: "The current raster layer is deleted",
		arity: FilterArity.NONE,
		filterClass: FilterClass.RasterLayer,
	},
	[FP.FP_DELETE_NON_SELECTED_RASTER]: {
		name: "Delete all Non Selected Rasters",
		pythonName: "delete_non_active_rasters",
		info: "All non selected raster layers are deleted",
		arity: FilterArity.NONE,
		filterClass: FilterClass.RasterLayer,
	},
	[FP.FP_RENAME_RASTER]: {
		name: "Rename Current Raster",
		pythonName: "set_raster_name",
		info: "Explicitly change the label shown for a given raster",
		arity: FilterArity.NONE,
		filterClass: FilterClass.RasterLayer,
	},
	[FP.FP_EXPORT_CAMERAS]: {
		name: "Export active rasters cameras to file",
		pythonName: "save_active_raster_cameras",
		info: "Export active cameras to file, in the .out or Agisoft .xml formats",
		arity: FilterArity.NONE,
		filterClass: FilterClass.RasterLayer,
	},
	[FP.FP_IMPORT_CAMERAS]: {
		name: "Import cameras for active rasters from file",
		pythonName: "load_active_raster_cameras",
		info: "Import cameras for active rasters from .out or Agisoft .xml formats",
		arity: FilterArity.NONE,
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
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
	override getClass(id: ActionIDType): FilterClassMask {
		return this.spec(id).filterClass ?? FilterClass.Layer;
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

			case FP.FP_SPLITSELECTEDFACES:
			case FP.FP_SPLITSELECTEDVERTICES:
				list.add(
					new RichBool("DeleteOriginal", true, {
						description: "Delete original selection",
						tooltip:
							"Deletes the original selected faces/vertices, thus splitting the mesh among layers.\n\n" +
							"if false, the selected faces/vertices are duplicated in the new layer.",
					}),
				);
				break;

			case FP.FP_SPLITCONNECTED:
				list.add(
					new RichBool("delete_source_mesh", false, {
						description: "Delete source mesh",
						tooltip:
							"Deletes the source mesh after all the connected component meshes are generated.",
					}),
				);
				break;

			case FP.FP_RENAME_RASTER:
				list.add(
					new RichString("newName", "", {
						description: "New Label",
						tooltip: "The new label for the raster.",
					}),
				);
				break;

			case FP.FP_EXPORT_CAMERAS:
				list.add(
					new RichEnum("ExportFile", 0, ["Bundler .out", "Agisoft xml"], {
						description: "Output format",
						tooltip:
							"Choose the output format, The filter enables to export the cameras to both " +
							"Bundler and Agisoft Photoscan.",
					}),
				);
				list.add(
					new RichString("newName", "cameras", {
						description: "Export file name (the right extension will be added at the end)",
						tooltip:
							"Name of the output file, it will be saved in the same folder as the project file.",
					}),
				);
				break;

			case FP.FP_IMPORT_CAMERAS:
				list.add(
					new RichFileOpen("ImportFile", "", ["out", "xml"], {
						description: "Import file",
						tooltip: "The Bundler .out or Agisoft .xml file to read the cameras from.",
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

			case FP.FP_DELETE_NON_VISIBLE_MESH: {
				const victims = doc.meshIterator().filter((layer) => !layer.isVisible());
				for (const layer of victims) doc.delMesh(layer.id());
				doc.Log.log(`Deleted ${victims.length} non visible layers`);
				post.mask = MeshElement.MM_NONE;
				return { deleted_layers: victims.length };
			}

			case FP.FP_SPLITSELECTEDFACES:
			case FP.FP_SPLITSELECTEDVERTICES: {
				const source = doc.mm();
				const cm = source.cm;
				const byFace = id === FP.FP_SPLITSELECTEDFACES;

				if (byFace) {
					// A face cannot travel without its vertices, so the loose
					// closure runs first and defines what gets copied.
					vertexFromFaceLoose(cm);
				}
				const faces = countSelectedFaces(cm);
				const verts = countSelectedVerts(cm);
				if (verts === 0) throw new MLException("nothing is selected, so there is nothing to move");

				const subset = new CMeshO();
				appendMesh(subset, cm, true);
				const target = doc.addNewMesh(
					"",
					byFace ? "SelectedFacesSubset" : "SelectedVerticesSubset",
					true,
					subset,
				);
				clearSelection(subset);
				target.updateBoxAndNormals();

				if (params.getBool("DeleteOriginal")) {
					if (byFace) {
						// Strict this time: a vertex on the seam is still used
						// by a face that stays, so only the interior goes.
						clearVertexSelection(cm);
						vertexFromFaceStrict(cm);
					} else {
						// Any face touching a departing vertex has to go too.
						faceFromVertexLoose(cm);
					}
					let deletedFaces = 0;
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
						Allocator.deleteFace(cm, f);
						deletedFaces++;
					}
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v) || !cm.isVertS(v)) continue;
						Allocator.deleteVertex(cm, v);
					}
					clearSelection(cm);
					source.clearDataMask(MeshElement.MM_FACEFACETOPO);
					source.updateBoxAndNormals();
					doc.Log.log(
						`Moved ${byFace ? `${faces} faces and ` : ""}${verts} vertices out of ` +
							`"${source.label()}", deleting ${deletedFaces} faces`,
					);
				} else {
					doc.Log.log(
						`Copied ${byFace ? `${faces} faces and ` : ""}${verts} vertices from "${source.label()}"`,
					);
				}

				return { new_mesh_id: target.id(), vertices: subset.vn, faces: subset.fn };
			}

			case FP.FP_SPLITCONNECTED: {
				const source = doc.mm();
				const cm = source.cm;
				source.updateDataMask(MeshElement.MM_FACEFACETOPO);
				const components = Clean.connectedComponents(cm);
				doc.Log.log(`Found ${components.length} Connected Components`);

				const ids: number[] = [];
				components.forEach(([, seed], i) => {
					clearSelection(cm);
					cm.faceFlags[seed] |= FaceFlag.SELECTED;
					faceConnectedFF(cm);
					vertexFromFaceLoose(cm);

					const part = new CMeshO();
					appendMesh(part, cm, true);
					const target = doc.addNewMesh("", `CC ${i}`, true, part);
					clearSelection(part);
					target.updateBoxAndNormals();
					ids.push(target.id());
				});
				clearSelection(cm);

				if (params.getBool("delete_source_mesh")) doc.delMesh(source.id());
				post.mask = MeshElement.MM_NONE;
				return { connected_components: components.length, new_mesh_ids: ids.join(" ") };
			}

			case FP.FP_DELETE_RASTER: {
				const raster = doc.rm();
				if (raster === null) throw new MLException("the document has no current raster");
				doc.delRaster(raster.id());
				doc.Log.log(`Deleted raster "${raster.label()}"`);
				post.mask = MeshElement.MM_NONE;
				return { deleted_raster_id: raster.id() };
			}

			case FP.FP_DELETE_NON_SELECTED_RASTER: {
				// Visibility is what "selected" means for a raster layer.
				const victims = doc.rasterIterator().filter((r) => !r.isVisible());
				for (const raster of victims) doc.delRaster(raster.id());
				doc.Log.log(`Deleted ${victims.length} non selected rasters`);
				post.mask = MeshElement.MM_NONE;
				return { deleted_rasters: victims.length };
			}

			case FP.FP_RENAME_RASTER: {
				const raster = doc.rm();
				if (raster === null) {
					throw new MLException("Error: Call to Rename Current Raster with no valid raster.");
				}
				const previous = raster.label();
				raster.setLabel(params.getString("newName"));
				doc.Log.log(`Renamed raster "${previous}" to "${raster.label()}"`);
				post.mask = MeshElement.MM_NONE;
				return { name: raster.label() };
			}

			case FP.FP_EXPORT_CAMERAS: {
				const active = doc.visibleRasters();
				if (active.length === 0) throw new MLException("no active raster to export a camera for");
				const agisoft = params.getEnum("ExportFile") === 1;
				const path = `${params.getString("newName")}.${agisoft ? "xml" : "out"}`;
				const text = agisoft
					? writeAgisoftXml(active.map((r) => ({ label: r.label(), shot: r.shot })))
					: writeBundlerOut(active.map((r) => r.shot));
				writeFileSync(path, text, "utf8");
				doc.Log.log(`Exported ${active.length} cameras to ${path}`);
				post.mask = MeshElement.MM_NONE;
				return { file_name: path, cameras: active.length };
			}

			case FP.FP_IMPORT_CAMERAS: {
				const path = params.getString("ImportFile");
				if (path === "") throw new MLException("No file to open");
				const active = doc.rasterIterator().filter((r) => r.isVisible());
				const text = readFileSync(path, "utf8");
				const extension = (path.split(".").pop() ?? "").toLowerCase();

				if (extension === "out") {
					const shots = readBundlerOut(text);
					if (shots.length !== active.length) {
						throw new MLException(
							"Wait! The number of active rasters and the number of cams in the Bundler file " +
								`is not the same! (${active.length} rasters, ${shots.length} cameras)`,
						);
					}
					active.forEach((raster, i) => {
						const shot = shots[i];
						// Bundler stores no image size, so it comes from the
						// image itself — the same recovery MeshLab performs.
						const size = readImageSize(raster.currentPlane?.fullPathFileName);
						if (size !== null) {
							shot.Intrinsics.ViewportPx = size;
							shot.Intrinsics.centreOnViewport();
						}
						applyShot(raster, shot);
					});
					doc.Log.log(`Imported ${shots.length} cameras from ${path}`);
					post.mask = MeshElement.MM_NONE;
					return { cameras: shots.length };
				}

				if (extension === "xml") {
					const { cameras, warnings } = readAgisoftXml(text);
					for (const warning of warnings) doc.Log.log(warning);
					let matched = 0;
					for (const camera of cameras) {
						// Agisoft names its cameras after the image files, so
						// that name is the only link back to a raster.
						const raster = active.find(
							(r) => r.currentPlane?.shortName() === camera.label || r.label() === camera.label,
						);
						if (raster === undefined) continue;
						applyShot(raster, camera.shot);
						matched++;
					}
					if (matched === 0) {
						throw new MLException(
							`none of the ${cameras.length} cameras in ${path} matches an active raster by name`,
						);
					}
					doc.Log.log(`Imported ${matched} cameras from ${path}`);
					post.mask = MeshElement.MM_NONE;
					return { cameras: matched };
				}

				throw new MLException(`Unknown file type "${extension}"`);
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}

/** The pixel size of a raster's image, or null when it cannot be read. */
function readImageSize(path: string | undefined): [number, number] | null {
	if (path === undefined || path === "") return null;
	try {
		return imageSizeOf(readFileSync(path));
	} catch {
		// A missing image is not a reason to refuse the whole camera file.
		return null;
	}
}

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

function clearVertexSelection(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.SELECTED;
}

function clearSelection(m: CMeshO): void {
	clearVertexSelection(m);
	for (let f = 0; f < m.faceSize; f++) m.faceFlags[f] &= ~FaceFlag.SELECTED;
}

function vertexFromFaceLoose(m: CMeshO): void {
	clearVertexSelection(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || !m.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) m.vertFlags[m.fv(f, k)] |= VertexFlag.SELECTED;
	}
}

function vertexFromFaceStrict(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (!m.isVertD(v)) m.vertFlags[v] |= VertexFlag.SELECTED;
	}
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f) || m.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) m.vertFlags[m.fv(f, k)] &= ~VertexFlag.SELECTED;
	}
}

function faceFromVertexLoose(m: CMeshO): void {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		let any = false;
		for (let k = 0; k < 3; k++) if (m.isVertS(m.fv(f, k))) any = true;
		if (any) m.faceFlags[f] |= FaceFlag.SELECTED;
		else m.faceFlags[f] &= ~FaceFlag.SELECTED;
	}
}

/** Grows the face selection across FF adjacency until it stops changing. */
function faceConnectedFF(m: CMeshO): void {
	UpdateTopology.faceFace(m);
	const stack: number[] = [];
	for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f) && m.isFaceS(f)) stack.push(f);
	const seen = new Set<number>(stack);
	while (stack.length > 0) {
		const f = stack.pop() as number;
		m.faceFlags[f] |= FaceFlag.SELECTED;
		for (let e = 0; e < 3; e++) {
			// Walk the whole ring, so a non-manifold edge joins every face on it.
			let nf = m.ffp(f, e);
			let ne = m.ffi(f, e);
			while (nf !== f || ne !== e) {
				if (!m.isFaceD(nf) && !seen.has(nf)) {
					seen.add(nf);
					stack.push(nf);
				}
				const tf = m.ffp(nf, ne);
				const te = m.ffi(nf, ne);
				nf = tf;
				ne = te;
			}
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
function appendMesh(dst: CMeshO, src: CMeshO, selectedOnly = false): void {
	const wanted = (v: number) => !src.isVertD(v) && (!selectedOnly || src.isVertS(v));
	const remap = new Int32Array(src.vertSize).fill(-1);
	let live = 0;
	for (let v = 0; v < src.vertSize; v++) if (wanted(v)) live++;
	if (live === 0) return;

	const firstVert = Allocator.addVertices(dst, live);
	let next = firstVert;
	for (let v = 0; v < src.vertSize; v++) {
		if (!wanted(v)) continue;
		remap[v] = next;
		dst.setVert(next, src.vx(v), src.vy(v), src.vz(v));
		dst.vertQuality[next] = src.vertQuality[v];
		dst.vertColor[next] = src.vertColor[v];
		next++;
	}

	for (let f = 0; f < src.faceSize; f++) {
		if (src.isFaceD(f)) continue;
		if (selectedOnly && !src.isFaceS(f)) continue;
		// A selected face whose vertices did not all come along would index
		// nothing; upstream relies on the caller closing the selection first,
		// and this is the check that turns a mistake into a dropped face
		// rather than a corrupt mesh.
		const a = remap[src.fv(f, 0)];
		const b = remap[src.fv(f, 1)];
		const c = remap[src.fv(f, 2)];
		if (a < 0 || b < 0 || c < 0) continue;
		Allocator.addFace(dst, a, b, c);
	}
}

export { appendMesh };
