/**
 * `MeshLabKernel` — the front door.
 *
 * ```ts
 * const k = MeshLabKernel.default();
 * const doc = new MeshDocument();
 * k.loadMesh(doc, "broken.stl");
 * k.applyFilter(doc, "Close Holes", { maxholesize: 100 });
 * k.saveMesh(doc, "fixed.stl");
 * ```
 *
 * `MeshDocument` stays a data container, as in C++; invocation lives here so
 * that a document can be built and inspected without the plugin registry.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createStubPlugins } from "../meshlabplugins/_stub/stub_plugins.ts";
import { FilterAmbientOcclusion } from "../meshlabplugins/filter_ao/filter_ao.ts";
import { FilterCamera } from "../meshlabplugins/filter_camera/filter_camera.ts";
import { FilterClean } from "../meshlabplugins/filter_clean/filter_clean.ts";
import { FilterColorProjection } from "../meshlabplugins/filter_color_projection/filter_color_projection.ts";
import { FilterColorProc } from "../meshlabplugins/filter_colorproc/filter_colorproc.ts";
import { FilterCreate } from "../meshlabplugins/filter_create/filter_create.ts";
import { FilterCreateIso } from "../meshlabplugins/filter_createiso/filter_createiso.ts";
import { FilterCubization } from "../meshlabplugins/filter_cubization/filter_cubization.ts";
import { FilterDirt } from "../meshlabplugins/filter_dirt/filter_dirt.ts";
import { FilterEmbree } from "../meshlabplugins/filter_embree/filter_embree.ts";
import { FilterFractal } from "../meshlabplugins/filter_fractal/filter_fractal.ts";
import { FilterFunc } from "../meshlabplugins/filter_func/filter_func.ts";
import { FilterGeodesic } from "../meshlabplugins/filter_geodesic/filter_geodesic.ts";
import { FilterICP } from "../meshlabplugins/filter_icp/filter_icp.ts";
import { FilterIsoParametrization } from "../meshlabplugins/filter_isoparametrization/filter_isoparametrization.ts";
import { FilterLayer } from "../meshlabplugins/filter_layer/filter_layer.ts";
import { FilterMeasure } from "../meshlabplugins/filter_measure/filter_measure.ts";
import { FilterMeshAlphaWrap } from "../meshlabplugins/filter_mesh_alpha_wrap/filter_mesh_alpha_wrap.ts";
import { FilterMeshBoolean } from "../meshlabplugins/filter_mesh_booleans/filter_mesh_booleans.ts";
import { FilterMeshing } from "../meshlabplugins/filter_meshing/filter_meshing.ts";
import { FilterMLS } from "../meshlabplugins/filter_mls/filter_mls.ts";
import { FilterParametrization } from "../meshlabplugins/filter_parametrization/filter_parametrization.ts";
import { FilterQuality } from "../meshlabplugins/filter_quality/filter_quality.ts";
import { FilterSample } from "../meshlabplugins/filter_sample/filter_sample.ts";
import { FilterSampling } from "../meshlabplugins/filter_sampling/filter_sampling.ts";
import { FilterScreenedPoisson } from "../meshlabplugins/filter_screened_poisson/filter_screened_poisson.ts";
import { FilterSDF } from "../meshlabplugins/filter_sdfgpu/filter_sdfgpu.ts";
import { FilterSelect } from "../meshlabplugins/filter_select/filter_select.ts";
import { FilterTexture } from "../meshlabplugins/filter_texture/filter_texture.ts";
import { FilterTriOptimize } from "../meshlabplugins/filter_trioptimize/filter_trioptimize.ts";
import { FilterUnsharp } from "../meshlabplugins/filter_unsharp/filter_unsharp.ts";
import { FilterVoronoi } from "../meshlabplugins/filter_voronoi/filter_voronoi.ts";
import { BaseMeshIOPlugin } from "../meshlabplugins/io_base/io_base.ts";
import type { MeshDocument } from "./ml_document/mesh_document.ts";
import type { MeshModel } from "./ml_document/mesh_model.ts";
import type { RichParameterList } from "./parameters/rich_parameter_list.ts";
import { type ExecuteOptions, executeFilter } from "./plugins/filter_executor.ts";
import type { FilterOutput, FilterPlugin } from "./plugins/interfaces/filter_plugin.ts";
import type { IOPlugin, OpenMaskBox } from "./plugins/interfaces/io_plugin.ts";
import { type FilterAction, PluginManager } from "./plugins/plugin_manager.ts";
import { type CallBackPos, noCallback } from "./utilities/callback.ts";
import { extensionOf } from "./utilities/file_format.ts";
import { MLIOException } from "./utilities/ml_exception.ts";

/** The filter plugins that have real implementations. */
const IMPLEMENTED_FILTER_PLUGINS: ReadonlyArray<() => FilterPlugin> = [
	() => new FilterClean(),
	() => new FilterMeshing(),
	() => new FilterMeasure(),
	() => new FilterSelect(),
	() => new FilterUnsharp(),
	() => new FilterLayer(),
	() => new FilterMLS(),
	() => new FilterTexture(),
	() => new FilterIsoParametrization(),
	() => new FilterGeodesic(),
	() => new FilterFractal(),
	() => new FilterCreateIso(),
	() => new FilterVoronoi(),
	() => new FilterAmbientOcclusion(),
	() => new FilterSDF(),
	() => new FilterICP(),
	() => new FilterCamera(),
	() => new FilterDirt(),
	() => new FilterColorProjection(),
	() => new FilterMeshAlphaWrap(),
	() => new FilterCubization(),
	() => new FilterEmbree(),
	() => new FilterMeshBoolean(),
	() => new FilterTriOptimize(),
	() => new FilterParametrization(),
	() => new FilterQuality(),
	() => new FilterSample(),
	() => new FilterColorProc(),
	() => new FilterCreate(),
	() => new FilterFunc(),
	() => new FilterSampling(),
	() => new FilterScreenedPoisson(),
];

/** The I/O plugins that have real implementations. */
const IO_PLUGINS: ReadonlyArray<() => IOPlugin> = [() => new BaseMeshIOPlugin()];

export class MeshLabKernel {
	readonly pluginManager: PluginManager;

	constructor(pluginManager: PluginManager) {
		this.pluginManager = pluginManager;
	}

	/** A kernel with every built-in plugin registered, stubs included. */
	static default(): MeshLabKernel {
		const pm = new PluginManager();

		const implemented = IMPLEMENTED_FILTER_PLUGINS.map((make) => make());
		for (const plugin of implemented) pm.registerFilterPlugin(plugin);

		// Whatever the real plugins did not claim stays registered as a stub,
		// so filterList() is always the complete MeshLab catalogue.
		const claimed = new Set<string>();
		for (const plugin of implemented) {
			for (const id of plugin.actions()) claimed.add(plugin.filterName(id));
		}
		for (const stub of createStubPlugins(claimed)) pm.registerFilterPlugin(stub);

		for (const make of IO_PLUGINS) pm.registerIOPlugin(make());
		return new MeshLabKernel(pm);
	}

	// ---- filters -------------------------------------------------------------

	filterList(): readonly FilterAction[] {
		return this.pluginManager.filterList();
	}

	filterAction(name: string): FilterAction {
		return this.pluginManager.requireFilterAction(name);
	}

	filterInfo(name: string): string {
		return this.filterAction(name).info;
	}

	/** The parameters and defaults for `name`, given the document's current mesh. */
	initParameterList(name: string, doc?: MeshDocument): RichParameterList {
		const action = this.filterAction(name);
		const mm = doc !== undefined && doc.meshNumber() > 0 ? doc.mm() : undefined;
		return action.plugin.initParameterList(action.id, mm);
	}

	/**
	 * Runs a filter.
	 *
	 * `params` is merged onto the defaults; an unknown key raises
	 * `InvalidParameterException` rather than being ignored, so a typo cannot
	 * quietly run the filter with a default value.
	 */
	applyFilter(
		doc: MeshDocument,
		name: string,
		params: Readonly<Record<string, unknown>> = {},
		options: ExecuteOptions = {},
	): FilterOutput {
		const action = this.filterAction(name);
		const list = action.plugin.initParameterList(
			action.id,
			doc.meshNumber() > 0 ? doc.mm() : undefined,
		);
		list.applyPlain(params);
		return executeFilter(doc, action, list, options);
	}

	// ---- I/O -----------------------------------------------------------------

	/** Reads `path` into a new layer and makes it current. */
	loadMesh(
		doc: MeshDocument,
		path: string,
		params: Readonly<Record<string, unknown>> = {},
	): MeshModel {
		const hit = this.pluginManager.inputMeshPlugin(path);
		if (hit === undefined) {
			const { input } = this.pluginManager.supportedExtensions();
			throw new MLIOException(
				`no plugin can read ".${extensionOf(path)}"; supported: ${input.join(", ")}`,
				path,
			);
		}
		let data: Uint8Array;
		try {
			data = new Uint8Array(readFileSync(path));
		} catch (cause) {
			throw new MLIOException(`could not read the file (${(cause as Error).message})`, path);
		}
		return this.openMeshData(doc, path, data, hit.plugin, hit.format, params);
	}

	/**
	 * Reads bytes already in memory, for callers that are not on a filesystem.
	 *
	 * The plugin and format are worked out from `name` when they are not given,
	 * so a caller holding bytes and a filename need not look them up.
	 */
	openMeshData(
		doc: MeshDocument,
		name: string,
		data: Uint8Array,
		plugin?: IOPlugin,
		format?: string,
		params: Readonly<Record<string, unknown>> = {},
	): MeshModel {
		if (plugin === undefined || format === undefined) {
			const hit = this.pluginManager.inputMeshPlugin(name);
			if (hit === undefined) {
				const { input } = this.pluginManager.supportedExtensions();
				throw new MLIOException(
					`no plugin can read ".${extensionOf(name)}"; supported: ${input.join(", ")}`,
					name,
				);
			}
			plugin = hit.plugin;
			format = hit.format;
		}
		const m = doc.addNewMesh(name, "", true);
		const list = plugin.initPreOpenParameter(format);
		list.applyPlain(params);
		const mask: OpenMaskBox = { mask: 0 };
		plugin.open(format, name, data, m, mask, list, noCallback);
		m.updateDataMask(mask.mask);
		m.updateBoxAndNormals();
		m.setMeshModified(false);
		return m;
	}

	/** Writes a layer to `path`, choosing the format from its extension. */
	saveMesh(
		doc: MeshDocument,
		path: string,
		mesh?: MeshModel,
		params: Readonly<Record<string, unknown>> = {},
		cb: CallBackPos = noCallback,
		mask?: number,
	): void {
		const bytes = this.serializeMesh(doc, path, mesh, params, cb, mask);
		try {
			writeFileSync(path, bytes);
		} catch (cause) {
			throw new MLIOException(`could not write the file (${(cause as Error).message})`, path);
		}
	}

	/**
	 * As {@link saveMesh}, but hands back the bytes instead of writing them.
	 *
	 * `mask` names the per-element channels to write, as MeshLab's `MM_*` bits.
	 * The default is the format's own — geometry only — because writing
	 * whatever the mesh happens to carry would make a file's contents depend on
	 * which filters ran earlier. A point cloud is the case that matters: it is
	 * useless downstream without its normals, so saving one means asking for
	 * `MM_VERTNORMAL` explicitly.
	 */
	serializeMesh(
		doc: MeshDocument,
		path: string,
		mesh?: MeshModel,
		params: Readonly<Record<string, unknown>> = {},
		cb: CallBackPos = noCallback,
		mask?: number,
	): Uint8Array {
		const hit = this.pluginManager.outputMeshPlugin(path);
		if (hit === undefined) {
			const { output } = this.pluginManager.supportedExtensions();
			throw new MLIOException(
				`no plugin can write ".${extensionOf(path)}"; supported: ${output.join(", ")}`,
				path,
			);
		}
		const m = mesh ?? doc.mm();
		const list = hit.plugin.initSaveParameter(hit.format, m);
		list.applyPlain(params);
		const { capability, defaultBits } = hit.plugin.exportMaskCapability(hit.format);
		if (mask !== undefined && (mask & ~capability) !== 0) {
			throw new MLIOException(
				`${hit.format} cannot store every requested channel (unsupported bits: ${mask & ~capability})`,
				path,
			);
		}
		return hit.plugin.save(hit.format, path, m, mask ?? defaultBits, list, cb);
	}
}
