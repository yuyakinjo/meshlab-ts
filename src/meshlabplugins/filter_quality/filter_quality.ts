/**
 * `filter_quality` — mapping the quality channel through a transfer function.
 *
 * The colour ramp in `filter_colorproc` is fixed; this one is adjustable. The
 * caller sets the range that maps onto the ramp and where the middle of it
 * falls, which is what turns a channel whose interesting values sit in a
 * narrow band into a picture with contrast in that band.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichFloat } from "../../common/parameters/rich_parameter.ts";
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { blue, colorRamp, green, red, rgba } from "../../vcg/space/color4.ts";

export const FP = { FP_QUALITY_MAPPER: 0 } as const;

export class FilterQuality extends FilterPlugin {
	pluginName(): string {
		return "FilterQuality";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.FP_QUALITY_MAPPER) this.wrongActionCalled(id);
		return "Quality Mapper applier";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.FP_QUALITY_MAPPER) this.wrongActionCalled(id);
		return "compute_color_from_scalar_using_transfer_function_per_vertex";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.FP_QUALITY_MAPPER) this.wrongActionCalled(id);
		return (
			"The filter maps quality levels into colors using a colorband built from a transfer " +
			"function and colorizes the mesh vertices. The minimum, medium and maximum quality " +
			"values can be set by user to obtain a custom quality range for mapping"
		);
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Quality;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override getPreConditions(_id: ActionIDType): number {
		return MeshElement.MM_VERTQUALITY;
	}
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTCOLOR;
	}

	override initParameterList(_id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const { min, max } = qualityRange(m?.cm);
		list.add(
			new RichFloat("minQualityVal", min, {
				description: "Minimum mesh quality",
				tooltip: "Quality at or below this maps to the start of the colour band.",
			}),
		);
		list.add(
			new RichFloat("maxQualityVal", max, {
				description: "Maximum mesh quality",
				tooltip: "Quality at or above this maps to the end of the colour band.",
			}),
		);
		list.add(
			new RichFloat("midHandlePos", 50, {
				description: "Middle position (%)",
				tooltip:
					"Where the middle of the colour band falls within the range, as a percentage. Moving " +
					"it stretches one half of the band and compresses the other.",
			}),
		);
		list.add(
			new RichFloat("brightness", 1, {
				description: "Brightness",
				tooltip: "Scales the resulting colour; 1 leaves it alone.",
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
		if (id !== FP.FP_QUALITY_MAPPER) return this.wrongActionCalled(id);
		const m = doc.mm();
		const cm = m.cm;
		m.updateDataMask(MeshElement.MM_VERTCOLOR);

		const min = params.getFloat("minQualityVal");
		const max = params.getFloat("maxQualityVal");
		if (max <= min) {
			throw new MLException(`the quality range is empty or inverted: ${min} to ${max}`);
		}
		const midPercent = params.getFloat("midHandlePos");
		if (midPercent <= 0 || midPercent >= 100) {
			throw new MLException(
				`the middle position must be strictly within 0..100, got ${midPercent}`,
			);
		}
		const brightness = params.getFloat("brightness");

		const mid = midPercent / 100;
		let coloured = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const t = clamp01((cm.vertQuality[v] - min) / (max - min));
			// The handle bends the parameter rather than the colours: below it
			// the first half of the band is stretched over 0..mid, above it the
			// second half over mid..1. At 50% this is the identity.
			const bent = t <= mid ? (0.5 * t) / mid : 0.5 + (0.5 * (t - mid)) / (1 - mid);
			cm.vertColor[v] = scale(colorRamp(0, 1, bent), brightness);
			coloured++;
		}
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(`Mapped ${coloured} vertices over quality ${min}..${max}`);
		return { colored: coloured, min, max };
	}
}

function qualityRange(cm: CMeshO | undefined): { min: number; max: number } {
	if (cm === undefined) return { min: 0, max: 1 };
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		min = Math.min(min, cm.vertQuality[v]);
		max = Math.max(max, cm.vertQuality[v]);
	}
	if (!Number.isFinite(min) || min === max) return { min: 0, max: 1 };
	return { min, max };
}

function clamp01(x: number): number {
	return Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0;
}

function scale(colour: number, factor: number): number {
	if (factor === 1) return colour;
	const to = (x: number) => Math.max(0, Math.min(255, Math.round(x * factor)));
	return rgba(to(red(colour)), to(green(colour)), to(blue(colour)), (colour >>> 24) & 0xff);
}
