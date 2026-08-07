/**
 * `FilterExecutor` — everything that happens around `applyFilter`.
 *
 * In MeshLab this lives in the GUI (`MainWindow`/`FilterThread`), which means
 * a headless caller silently loses precondition checking, requirement
 * satisfaction and postcondition recomputation. Centralising it here is the
 * one deliberate divergence from the C++ structure, and it is what makes a
 * filter's declared masks a tested contract rather than a GUI hint.
 *
 * `{ enforce: false }` opts out, for benchmarking or for reproducing an
 * upstream quirk. If a difference from real MeshLab ever shows up, MeshLab
 * wins and this code changes.
 */

import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import type { MeshDocument } from "../ml_document/mesh_document.ts";
import { MeshElement, maskAnd, maskIntersects, maskWithout } from "../ml_document/mesh_element.ts";
import type { RichParameterList } from "../parameters/rich_parameter_list.ts";
import { type CallBackPos, noCallback } from "../utilities/callback.ts";
import { MissingPreconditionException, MLException } from "../utilities/ml_exception.ts";
import { FilterArity } from "./filter_arity.ts";
import type { FilterOutput, PostConditionBox } from "./interfaces/filter_plugin.ts";
import type { FilterAction } from "./plugin_manager.ts";

export interface ExecuteOptions {
	readonly cb?: CallBackPos;
	/** Set false to skip precondition, requirement and postcondition handling. */
	readonly enforce?: boolean;
}

/** Rejects a filter whose arity the document cannot satisfy. */
export function checkArity(action: FilterAction, doc: MeshDocument): void {
	switch (action.arity) {
		case FilterArity.NONE:
			// Creation filters need no input layer.
			return;
		case FilterArity.SINGLE_MESH:
		case FilterArity.FIXED:
		case FilterArity.VARIABLE:
			if (doc.meshNumber() === 0) {
				throw new MLException(
					`filter "${action.name}" needs at least one mesh, but the document is empty`,
				);
			}
			return;
		default:
			return;
	}
}

export function executeFilter(
	doc: MeshDocument,
	action: FilterAction,
	params: RichParameterList,
	options: ExecuteOptions = {},
): FilterOutput {
	const cb = options.cb ?? noCallback;
	const enforce = options.enforce ?? true;
	const { plugin, id } = action;

	if (enforce) {
		checkArity(action, doc);

		const pre = plugin.getPreConditions(id);
		if (pre !== MeshElement.MM_NONE && doc.meshNumber() > 0) {
			const missing = maskWithout(pre, doc.mm().currentCapability());
			if (missing !== 0) throw new MissingPreconditionException(action.name, missing);
		}

		const req = plugin.getRequirements(id);
		if (req !== MeshElement.MM_NONE && doc.meshNumber() > 0) {
			doc.mm().updateDataMask(req);
		}
	}

	const post: PostConditionBox = { mask: plugin.postCondition(id) };
	const out = plugin.applyFilter(id, params, doc, post, cb);

	if (enforce) {
		if (doc.meshNumber() > 0) applyPostCondition(doc, post.mask);
		doc.filterHistory.push({ filterName: action.name, params: params.toPlain() });
	}
	return out;
}

/**
 * Brings derived data back in line with what the filter changed.
 *
 * Adjacency is dropped rather than rebuilt: recomputing it eagerly would cost
 * every filter that does not need it, and the next `updateDataMask` rebuilds
 * it on demand. Bounding box and normals *are* recomputed, because callers
 * read them straight off the mesh without asking.
 */
export function applyPostCondition(doc: MeshDocument, mask: number): void {
	const m = doc.mm();
	const geometryChanged = maskIntersects(
		mask,
		MeshElement.MM_VERTCOORD | MeshElement.MM_TRANSFMATRIX,
	);
	const topologyChanged = maskIntersects(
		mask,
		MeshElement.MM_FACEVERT | MeshElement.MM_VERTNUMBER | MeshElement.MM_FACENUMBER,
	);

	if (topologyChanged) {
		const stale = maskAnd(
			m.cm.currentDataMask,
			MeshElement.MM_FACEFACETOPO | MeshElement.MM_VERTFACETOPO,
		);
		if (stale !== 0) m.clearDataMask(stale);
	}

	if (geometryChanged || topologyChanged) {
		UpdateBounding.box(m.cm);
		if (m.cm.fn > 0) UpdateNormal.perVertexNormalizedPerFaceNormalized(m.cm);
	}

	m.setMeshModified(true);
}
