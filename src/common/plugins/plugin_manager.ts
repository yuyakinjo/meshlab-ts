/**
 * `PluginManager` — the registry.
 *
 * The one invariant it exists to enforce is the same as C++
 * `checkFilterPlugin`: **filter names are globally unique**. That is what lets
 * a caller say `applyFilter(doc, "Close Holes")` without naming a plugin, and
 * a collision is a load-time error rather than a silent shadowing.
 */

import { extensionOf, type FileFormat } from "../utilities/file_format.ts";
import { MLException } from "../utilities/ml_exception.ts";
import type { FilterArityValue } from "./filter_arity.ts";
import type { FilterClassMask } from "./filter_class.ts";
import type { FilterPlugin } from "./interfaces/filter_plugin.ts";
import type { IOPlugin } from "./interfaces/io_plugin.ts";
import type { ActionIDType } from "./meshlab_plugin.ts";

/**
 * One filter, resolved. This is what `QAction*` carries in C++, minus the GUI.
 */
export interface FilterAction {
	readonly plugin: FilterPlugin;
	readonly id: ActionIDType;
	readonly name: string;
	readonly pythonName: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
	readonly info: string;
	/** False while the filter is registry-only and throws when applied. */
	readonly implemented: boolean;
}

export class PluginManager {
	private readonly filterPlugins: FilterPlugin[] = [];
	private readonly ioPluginList: IOPlugin[] = [];
	private readonly byName = new Map<string, FilterAction>();
	private readonly byPythonName = new Map<string, FilterAction>();

	/**
	 * Registers every filter a plugin provides.
	 *
	 * `implemented` is not something a plugin declares separately — it is read
	 * from the plugin, so a stub cannot accidentally advertise itself as real.
	 */
	registerFilterPlugin(plugin: FilterPlugin): void {
		const implemented = !isStub(plugin);
		for (const id of plugin.actions()) {
			const name = plugin.filterName(id);
			const existing = this.byName.get(name);
			if (existing !== undefined) {
				throw new MLException(
					`duplicate filter name "${name}": ${plugin.pluginName()} and ` +
						`${existing.plugin.pluginName()} both provide it`,
				);
			}
			const pythonName = plugin.pythonFilterName(id);
			const pyClash = this.byPythonName.get(pythonName);
			if (pyClash !== undefined) {
				throw new MLException(
					`duplicate PyMeshLab name "${pythonName}": "${name}" and "${pyClash.name}"`,
				);
			}
			const action: FilterAction = {
				plugin,
				id,
				name,
				pythonName,
				filterClass: plugin.getClass(id),
				arity: plugin.filterArity(id),
				info: plugin.filterInfo(id),
				implemented,
			};
			this.byName.set(name, action);
			this.byPythonName.set(pythonName, action);
		}
		this.filterPlugins.push(plugin);
	}

	registerIOPlugin(plugin: IOPlugin): void {
		this.ioPluginList.push(plugin);
	}

	/**
	 * Looks a filter up by its display name or its PyMeshLab name.
	 *
	 * Accepting both is deliberate: a recipe transcribed from PyMeshLab uses
	 * `meshing_close_holes`, one transcribed from the MeshLab GUI uses
	 * "Close Holes", and neither caller should have to know about the other.
	 */
	filterAction(name: string): FilterAction | undefined {
		return this.byName.get(name) ?? this.byPythonName.get(name);
	}

	requireFilterAction(name: string): FilterAction {
		const action = this.filterAction(name);
		if (action === undefined) {
			throw new MLException(`no filter named "${name}"${this.suggest(name)}`);
		}
		return action;
	}

	/** Every registered filter, sorted by display name. */
	filterList(): readonly FilterAction[] {
		return [...this.byName.values()].sort((a, b) => a.name.localeCompare(b.name));
	}

	filterCount(): number {
		return this.byName.size;
	}

	ioPlugins(): readonly IOPlugin[] {
		return this.ioPluginList;
	}

	/** The plugin that can read `path`, chosen by its extension. */
	inputMeshPlugin(path: string): { plugin: IOPlugin; format: string } | undefined {
		return this.findIO(path, (p) => p.importFormats());
	}

	/** The plugin that can write `path`, chosen by its extension. */
	outputMeshPlugin(path: string): { plugin: IOPlugin; format: string } | undefined {
		return this.findIO(path, (p) => p.exportFormats());
	}

	private findIO(
		path: string,
		formatsOf: (p: IOPlugin) => readonly FileFormat[],
	): { plugin: IOPlugin; format: string } | undefined {
		const ext = extensionOf(path);
		if (ext === "") return undefined;
		for (const plugin of this.ioPluginList) {
			for (const format of formatsOf(plugin)) {
				if (format.matches(ext)) return { plugin, format: ext };
			}
		}
		return undefined;
	}

	/** Extensions that can be read / written, for help text. */
	supportedExtensions(): { input: string[]; output: string[] } {
		const gather = (formatsOf: (p: IOPlugin) => readonly FileFormat[]) => {
			const out = new Set<string>();
			for (const p of this.ioPluginList) {
				for (const f of formatsOf(p)) for (const e of f.extensions) out.add(e);
			}
			return [...out].sort();
		};
		return {
			input: gather((p) => p.importFormats()),
			output: gather((p) => p.exportFormats()),
		};
	}

	/** "Did you mean" for a mistyped filter name. */
	private suggest(name: string): string {
		const needle = name.toLowerCase();
		const close = this.filterList()
			.filter(
				(a) =>
					a.name.toLowerCase().includes(needle) ||
					a.pythonName.includes(needle.replaceAll(" ", "_")),
			)
			.slice(0, 5)
			.map((a) => `"${a.name}"`);
		return close.length === 0 ? "" : `; did you mean ${close.join(", ")}?`;
	}
}

/**
 * Whether a plugin is registry-only.
 *
 * Stubs mark themselves with a property rather than being detected by
 * behaviour, so the check stays a fact rather than a guess.
 */
function isStub(plugin: FilterPlugin): boolean {
	return (plugin as unknown as { readonly isStub?: boolean }).isStub === true;
}
