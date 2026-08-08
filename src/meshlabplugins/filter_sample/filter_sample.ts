/**
 * `filter_sample` — MeshLab's example plugin, one filter.
 *
 * It exists upstream to show plugin authors the shape of the thing, and it is
 * here for the same reason: it is the smallest complete filter in the library.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichInt, RichPercentage } from "../../common/parameters/rich_parameter.ts";
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
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";

export const FP = { FP_MOVE_VERTEX: 0 } as const;

export class FilterSample extends FilterPlugin {
	pluginName(): string {
		return "FilterSample";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.FP_MOVE_VERTEX) this.wrongActionCalled(id);
		return "Random Vertex Displacement";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.FP_MOVE_VERTEX) this.wrongActionCalled(id);
		return "apply_coord_random_displacement";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.FP_MOVE_VERTEX) this.wrongActionCalled(id);
		return "Move the vertices of the mesh of a random quantity.";
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Smoothing;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override initParameterList(_id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
		}
		list.add(
			new RichPercentage("Displacement", diagonal / 100, 0, diagonal / 2, {
				description: "Max Displacement",
				tooltip: "The vertex are displaced of a vector whose norm is bounded by this value.",
			}),
		);
		list.add(
			new RichInt("RandomSeed", 0, {
				description: "Random Seed",
				tooltip:
					"The seed of the random generator. The same seed gives the same displacement, which " +
					"is what makes a run reproducible.",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		_cb: CallBackPos,
	): FilterOutput {
		if (id !== FP.FP_MOVE_VERTEX) return this.wrongActionCalled(id);
		const m = doc.mm();
		const cm = m.cm;
		const max = params.getAbsPerc("Displacement");
		// A named generator rather than Math.random: an unreproducible filter
		// makes every downstream hash useless, which is the same reason
		// filter_func refuses to implement muParser's `rnd`.
		const random = mulberry32(params.getInt("RandomSeed") >>> 0);

		let moved = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			// A uniform direction on the sphere, then a uniform length: a
			// per-axis random offset would bias towards the cube's corners.
			const z = 2 * random() - 1;
			const theta = 2 * Math.PI * random();
			const r = Math.sqrt(Math.max(0, 1 - z * z));
			const length = max * random();
			cm.setVert(
				v,
				cm.vx(v) + r * Math.cos(theta) * length,
				cm.vy(v) + r * Math.sin(theta) * length,
				cm.vz(v) + z * length,
			);
			moved++;
		}
		m.updateBoxAndNormals();
		post.mask = MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL;
		doc.Log.log(`Displaced ${moved} vertices by up to ${max}`);
		return { moved_vertices: moved };
	}
}

/** Mulberry32: small, fast, and good enough for jitter. */
function mulberry32(seed: number): () => number {
	let a = seed;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
