/**
 * `IOPlugin` — reading and writing mesh files.
 *
 * Mirrors `src/common/plugins/interfaces/io_plugin.h`. As with filters,
 * failure is thrown ({@link MLIOException}) rather than signalled by a return
 * value, and the `int& mask` out-parameter becomes a small box.
 */
import type { MeshModel } from "../../ml_document/mesh_model.ts";
import { RichParameterList } from "../../parameters/rich_parameter_list.ts";
import type { CallBackPos } from "../../utilities/callback.ts";
import type { FileFormat } from "../../utilities/file_format.ts";
import { MLIOException } from "../../utilities/ml_exception.ts";
import { MeshLabPlugin } from "../meshlab_plugin.ts";

/** What an importer reports it actually found, as `MM_*` bits. */
export interface OpenMaskBox {
	mask: number;
}

/** What a format is able to store, and what it stores unless told otherwise. */
export interface ExportCapability {
	capability: number;
	defaultBits: number;
}

export abstract class IOPlugin extends MeshLabPlugin {
	readonly pluginType = "io" as const;

	abstract importFormats(): readonly FileFormat[];
	abstract exportFormats(): readonly FileFormat[];

	/**
	 * Reads `data` into `m`, reporting through `mask` which attributes the
	 * file actually carried.
	 *
	 * Takes bytes rather than a path so the same code serves files, buffers
	 * and streams; the path is passed alongside only for error messages.
	 */
	abstract open(
		format: string,
		fileName: string,
		data: Uint8Array,
		m: MeshModel,
		mask: OpenMaskBox,
		params: RichParameterList,
		cb: CallBackPos,
	): void;

	/** Serialises `m`, returning the bytes to write. */
	abstract save(
		format: string,
		fileName: string,
		m: MeshModel,
		mask: number,
		params: RichParameterList,
		cb: CallBackPos,
	): Uint8Array;

	abstract exportMaskCapability(format: string): ExportCapability;

	/** Parameters resolved before the file is parsed. */
	initPreOpenParameter(_format: string): RichParameterList {
		return new RichParameterList();
	}

	/** Parameters for writing, such as ascii versus binary. */
	initSaveParameter(_format: string, _m: MeshModel): RichParameterList {
		return new RichParameterList();
	}

	/** How many meshes `data` contains. Formats that only ever hold one say 1. */
	numberMeshesContainedInFile(
		_format: string,
		_fileName: string,
		_data: Uint8Array,
		_preParams: RichParameterList,
	): number {
		return 1;
	}

	protected wrongOpenFormat(format: string): never {
		throw new MLIOException(`${this.pluginName()} cannot read the format "${format}"`);
	}

	protected wrongSaveFormat(format: string): never {
		throw new MLIOException(`${this.pluginName()} cannot write the format "${format}"`);
	}
}
