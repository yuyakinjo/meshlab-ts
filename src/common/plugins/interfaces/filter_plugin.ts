/**
 * `FilterPlugin` — the interface every filter implements.
 *
 * Mirrors `src/common/plugins/interfaces/filter_plugin.h`, with the Qt bits
 * translated:
 *
 * - `QAction*` is gone. A plugin lists its `ActionIDType`s and the registry
 *   builds the lookup, so no GUI concept leaks into a headless library.
 * - `std::map<std::string, QVariant>` is {@link FilterOutput}.
 * - Failure is thrown, never returned. `applyFilter` either produces its
 *   output map or raises an `MLException`.
 * - The `unsigned&` out-parameter for the postcondition mask is a small box,
 *   which keeps the signature the same shape as the C++ one.
 */
import type { MeshDocument } from "../../ml_document/mesh_document.ts";
import { MeshElement } from "../../ml_document/mesh_element.ts";
import type { MeshModel } from "../../ml_document/mesh_model.ts";
import { RichParameterList } from "../../parameters/rich_parameter_list.ts";
import type { CallBackPos } from "../../utilities/callback.ts";
import { MLException } from "../../utilities/ml_exception.ts";
import type { FilterArityValue } from "../filter_arity.ts";
import { FilterArity } from "../filter_arity.ts";
import type { FilterClassMask } from "../filter_class.ts";
import { FilterClass } from "../filter_class.ts";
import { type ActionIDType, MeshLabPlugin } from "../meshlab_plugin.ts";

/** A value a filter can report back. Stands in for `QVariant`. */
export type OutputValue =
	| number
	| string
	| boolean
	| readonly number[]
	| Readonly<Record<string, number>>;

/** `std::map<std::string, QVariant>`: what a filter tells the caller. */
export type FilterOutput = Record<string, OutputValue>;

/**
 * The postcondition mask, passed by reference in C++.
 *
 * A filter narrows `mask` to describe what it actually touched, so the
 * framework can recompute exactly that much afterwards.
 */
export interface PostConditionBox {
	mask: number;
}

export abstract class FilterPlugin extends MeshLabPlugin {
	readonly pluginType = "filter" as const;

	/** The action ids this plugin provides, in registration order. */
	abstract actions(): readonly ActionIDType[];

	/** The exact display name, which is also the key callers pass. */
	abstract filterName(id: ActionIDType): string;

	/** One-paragraph description, shown in the GUI and by `meshlab-ts info`. */
	abstract filterInfo(id: ActionIDType): string;

	abstract filterArity(id: ActionIDType): FilterArityValue;

	/**
	 * The snake_case name PyMeshLab exposes.
	 *
	 * The base implementation is not `computePythonName(filterName(id))`: 281
	 * of MeshLab's 285 filters override this with something quite different
	 * ("Close Holes" is `meshing_close_holes`), so a plugin that does not
	 * supply the real name would be advertising one nobody can use. Concrete
	 * plugins are expected to override.
	 */
	abstract pythonFilterName(id: ActionIDType): string;

	getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Generic;
	}

	/** Attributes the filter needs; the framework allocates and computes them. */
	getRequirements(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
	}

	/** Attributes the mesh must already have; missing ones abort the filter. */
	getPreConditions(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
	}

	/** Attributes the filter may invalidate. Conservatively everything. */
	postCondition(_id: ActionIDType): number {
		return MeshElement.MM_ALL;
	}

	/** The parameters, with their defaults, for this filter and this mesh. */
	initParameterList(_id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		return new RichParameterList();
	}

	abstract applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		postCondition: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput;

	/** Raised when a plugin is handed an action id it does not own. */
	protected wrongActionCalled(id: ActionIDType): never {
		throw new MLException(`${this.pluginName()} was called with unknown action id ${id}`);
	}
}

export type { ActionIDType };
export { FilterArity, FilterClass };
