/**
 * `io_base` — the core I/O plugin.
 *
 * Upstream also reads OBJ, OFF, PTX, VMI and FBX and writes WRL and DXF;
 * those arrive with the later tiers. PLY and STL come first because they are
 * what a 3D-printing pipeline actually moves around.
 */
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool } from "../../common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import {
	type ExportCapability,
	IOPlugin,
	type OpenMaskBox,
} from "../../common/plugins/interfaces/io_plugin.ts";
import type { CallBackPos } from "../../common/utilities/callback.ts";
import { FileFormat } from "../../common/utilities/file_format.ts";
import { readPly, writePly } from "./ply.ts";
import { readStl, writeStl } from "./stl.ts";

const PLY = new FileFormat("Stanford Polygon File Format", "PLY");
const STL = new FileFormat("STL (Stereolithography)", "STL");

export class BaseMeshIOPlugin extends IOPlugin {
	pluginName(): string {
		return "io_base";
	}

	importFormats(): readonly FileFormat[] {
		return [PLY, STL];
	}

	exportFormats(): readonly FileFormat[] {
		return [PLY, STL];
	}

	open(
		format: string,
		fileName: string,
		data: Uint8Array,
		m: MeshModel,
		mask: OpenMaskBox,
		_params: RichParameterList,
		_cb: CallBackPos,
	): void {
		switch (format.toUpperCase()) {
			case "PLY":
				mask.mask = readPly(m.cm, data, fileName).mask;
				return;
			case "STL":
				readStl(m.cm, data, fileName);
				// STL carries nothing but geometry: no colour, no quality, and
				// a per-facet normal we recompute rather than trust.
				mask.mask = MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT;
				return;
			default:
				this.wrongOpenFormat(format);
		}
	}

	save(
		format: string,
		_fileName: string,
		m: MeshModel,
		mask: number,
		params: RichParameterList,
		_cb: CallBackPos,
	): Uint8Array {
		const binary = params.hasParameter("Binary") ? params.getBool("Binary") : true;
		switch (format.toUpperCase()) {
			case "PLY":
				return writePly(m.cm, {
					binary,
					saveNormals: (mask & MeshElement.MM_VERTNORMAL) !== 0,
					saveColors: (mask & MeshElement.MM_VERTCOLOR) !== 0,
					saveQuality: (mask & MeshElement.MM_VERTQUALITY) !== 0,
				});
			case "STL":
				return writeStl(m.cm, { binary });
			default:
				return this.wrongSaveFormat(format);
		}
	}

	/**
	 * What each format can carry, and what it writes by default.
	 *
	 * PLY's defaults are geometry only. Writing normals and colours whenever
	 * the mesh happens to have them would silently change a file's contents
	 * from one run to the next as earlier filters enable channels; a caller
	 * who wants them asks.
	 */
	exportMaskCapability(format: string): ExportCapability {
		switch (format.toUpperCase()) {
			case "PLY":
				return {
					capability:
						MeshElement.MM_VERTCOORD |
						MeshElement.MM_VERTNORMAL |
						MeshElement.MM_VERTCOLOR |
						MeshElement.MM_VERTQUALITY |
						MeshElement.MM_FACEVERT,
					defaultBits: MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT,
				};
			case "STL":
				return {
					capability: MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT,
					defaultBits: MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT,
				};
			default:
				return this.wrongSaveFormat(format);
		}
	}

	override initSaveParameter(format: string, _m: MeshModel): RichParameterList {
		const list = new RichParameterList();
		const upper = format.toUpperCase();
		if (upper === "PLY" || upper === "STL") {
			list.add(
				new RichBool("Binary", true, {
					description: "Binary encoding",
					tooltip: "Write the file in binary rather than ASCII. Binary is smaller and exact.",
				}),
			);
		}
		return list;
	}
}
