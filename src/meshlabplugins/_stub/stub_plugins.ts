/**
 * Registry-only plugins for the filters that are not implemented yet.
 *
 * Every MeshLab filter is registered from the moment the library loads, so
 * `filterList()` is complete, `meshlab-ts list` shows the real catalogue, and
 * a caller can discover a name before anyone has written the code behind it.
 * Applying one throws {@link MLNotImplementedException} — the failure mode
 * that matters is a filter that silently does nothing and reports success, and
 * this design makes that impossible.
 *
 * As each filter gets a real implementation, its row moves out of the
 * generated table and into a real plugin; `scripts/check-registry.ts` verifies
 * that the two together still cover exactly the upstream set, with no name
 * appearing twice.
 */

import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import { type FilterArityValue, filterArityFromString } from "../../common/plugins/filter_arity.ts";
import { type FilterClassMask, filterClassFromString } from "../../common/plugins/filter_class.ts";
import {
	type ActionIDType,
	type FilterOutput,
	FilterPlugin,
	type PostConditionBox,
} from "../../common/plugins/interfaces/filter_plugin.ts";
import type { CallBackPos } from "../../common/utilities/callback.ts";
import { MLNotImplementedException } from "../../common/utilities/ml_exception.ts";
import { FILTER_TABLE, type FilterTableRow } from "./filter_table.ts";

interface StubEntry {
	readonly row: FilterTableRow;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
}

class StubFilterPlugin extends FilterPlugin {
	/** Read by PluginManager to mark these actions as not implemented. */
	readonly isStub = true;

	private readonly name: string;
	private readonly entries: StubEntry[];

	constructor(pluginDir: string, pluginName: string, rows: readonly FilterTableRow[]) {
		super();
		this.name = pluginName === "" ? pluginDir : pluginName;
		this.entries = rows.map((row) => ({
			row,
			filterClass: filterClassFromString(row.filterClass),
			arity: filterArityFromString(row.arity),
		}));
	}

	pluginName(): string {
		return this.name;
	}

	actions(): readonly ActionIDType[] {
		return this.entries.map((_, i) => i);
	}

	private entry(id: ActionIDType): StubEntry {
		const e = this.entries[id];
		if (e === undefined) this.wrongActionCalled(id);
		return e;
	}

	filterName(id: ActionIDType): string {
		return this.entry(id).row.filterName;
	}

	pythonFilterName(id: ActionIDType): string {
		return this.entry(id).row.pythonName;
	}

	filterInfo(id: ActionIDType): string {
		const { row } = this.entry(id);
		return row.info === "" ? `${row.filterName} (no description in the MeshLab source)` : row.info;
	}

	override getClass(id: ActionIDType): FilterClassMask {
		return this.entry(id).filterClass;
	}

	filterArity(id: ActionIDType): FilterArityValue {
		return this.entry(id).arity;
	}

	/**
	 * A stub touches nothing, so it declares no requirements and no
	 * preconditions — it must fail because it is unimplemented, not because
	 * the mesh happened to be missing an attribute.
	 */
	override getRequirements(): number {
		return MeshElement.MM_NONE;
	}

	override getPreConditions(): number {
		return MeshElement.MM_NONE;
	}

	override postCondition(): number {
		return MeshElement.MM_NONE;
	}

	applyFilter(
		id: ActionIDType,
		_params: RichParameterList,
		_doc: MeshDocument,
		_post: PostConditionBox,
		_cb: CallBackPos,
	): FilterOutput {
		throw new MLNotImplementedException(this.filterName(id), this.pluginName());
	}
}

/**
 * Builds one stub plugin per upstream plugin directory, skipping any filter
 * name a real implementation has already claimed.
 *
 * The skip is by *name* rather than by plugin, so a partially implemented
 * plugin works: the filters that exist come from the real code and the rest
 * stay registered as stubs.
 */
export function createStubPlugins(implementedNames: ReadonlySet<string>): StubFilterPlugin[] {
	const byDir = new Map<string, FilterTableRow[]>();
	for (const row of FILTER_TABLE) {
		if (implementedNames.has(row.filterName)) continue;
		const hit = byDir.get(row.pluginDir);
		if (hit === undefined) byDir.set(row.pluginDir, [row]);
		else hit.push(row);
	}
	// The plugin keeps its real upstream name; MLNotImplementedException
	// already says the filter is unimplemented, and repeating it in the
	// plugin name only makes the message read twice.
	return [...byDir].map(([dir, rows]) => new StubFilterPlugin(dir, rows[0].pluginName, rows));
}

export { StubFilterPlugin };
