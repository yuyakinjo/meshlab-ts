/**
 * `filter_unsharp` — the smoothing family.
 *
 * Upstream's plugin also holds the unsharp-mask and normal filters; the four
 * position smoothers here are the ones a repair pipeline uses, typically on
 * just the faces a hole fill created.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { Smooth } from "../../vcg/complex/smooth.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";

export const FP = {
	FP_LAPLACIAN_SMOOTH: 0,
	FP_TAUBIN_SMOOTH: 1,
	FP_HC_LAPLACIAN_SMOOTH: 2,
	FP_SD_LAPLACIAN_SMOOTH: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_LAPLACIAN_SMOOTH]: {
		name: "Laplacian Smooth",
		pythonName: "apply_coord_laplacian_smoothing",
		info:
			"Laplacian smooth: for each vertex it calculates the average position with nearest " +
			"vertex.",
	},
	[FP.FP_TAUBIN_SMOOTH]: {
		name: "Taubin Smooth",
		pythonName: "apply_coord_taubin_smoothing",
		info:
			"The $lambda-mu$ taubin smoothing, it make two steps of smoothing, forth and back, for " +
			"each iteration.",
	},
	[FP.FP_HC_LAPLACIAN_SMOOTH]: {
		name: "HC Laplacian Smooth",
		pythonName: "apply_coord_hc_laplacian_smoothing",
		info: "HC Laplacian Smoothing: a Laplacian smoothing that tries to reduce the shrinkage.",
	},
	[FP.FP_SD_LAPLACIAN_SMOOTH]: {
		name: "ScaleDependent Laplacian Smooth",
		pythonName: "apply_coord_laplacian_smoothing_scale_dependent",
		info:
			"Scale Dependent Laplacian Smoothing, extended version of Laplacian Smoothing based on " +
			"the Fujiwara extended umbrella operator.",
	},
};

export class FilterUnsharp extends FilterPlugin {
	pluginName(): string {
		return "FilterUnsharp";
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
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Smoothing;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	/** Smoothing moves vertices but never changes which faces exist. */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL | MeshElement.MM_FACENORMAL;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const selected = m !== undefined && hasSelectedFaces(m.cm);

		switch (id) {
			case FP.FP_LAPLACIAN_SMOOTH:
				list.add(
					new RichInt("stepSmoothNum", 3, {
						description: "Smoothing steps",
						tooltip: "The number of times that the whole algorithm is iterated.",
					}),
				);
				list.add(
					new RichBool("Boundary", true, {
						description: "1D Boundary Smoothing",
						tooltip:
							"Smooth boundary edges only by themselves. This can reduce the shrinking on the " +
							"border but can have strange effects on very small boundaries.",
					}),
				);
				list.add(
					new RichBool("cotangentWeight", true, {
						description: "Cotangent weighting",
						tooltip:
							"Use cotangent weighting scheme for the averaging of the position. Otherwise the " +
							"simpler umbrella scheme (1 if the edge is present) is used.",
					}),
				);
				list.add(
					new RichBool("Selected", selected, {
						description: "Affect only selection",
						tooltip: "If checked the filter is performed only on the selected area",
					}),
				);
				break;

			case FP.FP_TAUBIN_SMOOTH:
				list.add(
					new RichFloat("lambda", 0.5, {
						description: "Lambda",
						tooltip: "The lambda parameter of the Taubin Smoothing algorithm",
					}),
				);
				list.add(
					new RichFloat("mu", -0.53, {
						description: "mu",
						tooltip: "The mu parameter of the Taubin Smoothing algorithm",
					}),
				);
				list.add(
					new RichInt("stepSmoothNum", 10, {
						description: "Smoothing steps",
						tooltip:
							"The number of times that the taubin smoothing is iterated. Usually it requires " +
							"a larger number of iteration than the classical laplacian",
					}),
				);
				list.add(
					new RichBool("Selected", selected, {
						description: "Affect only selected faces",
						tooltip: "If checked the filter is performed only on the selected faces",
					}),
				);
				break;

			case FP.FP_HC_LAPLACIAN_SMOOTH:
				list.add(
					new RichBool("Selected", selected, {
						description: "Affect only selection",
						tooltip: "If checked the filter is performed only on the selected area",
					}),
				);
				break;

			case FP.FP_SD_LAPLACIAN_SMOOTH: {
				// The step has units of length, so it is scaled to the mesh.
				let delta = 0.001;
				if (m !== undefined) {
					UpdateBounding.box(m.cm);
					delta = (m.cm.bbox.diagonal || 1) * 0.01;
				}
				list.add(
					new RichInt("stepSmoothNum", 3, {
						description: "Smoothing steps",
						tooltip: "The number of times that the whole algorithm is iterated.",
					}),
				);
				list.add(
					new RichFloat("delta", delta, {
						description: "delta",
						tooltip: "The scale-dependent step size.",
					}),
				);
				list.add(
					new RichBool("Selected", selected, {
						description: "Affect only selected faces",
						tooltip: "If checked the filter is performed only on the selected faces",
					}),
				);
				break;
			}

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
		const m = doc.mm();
		const cm = m.cm;
		post.mask = this.postCondition(id);

		const smoothSelected = params.hasParameter("Selected") && params.getBool("Selected");
		if (smoothSelected) selectVerticesFromFaces(cm);

		switch (id) {
			case FP.FP_LAPLACIAN_SMOOTH: {
				const steps = params.getInt("stepSmoothNum");
				Smooth.vertexCoordLaplacian(cm, steps, {
					smoothSelected,
					cotangentWeight: params.getBool("cotangentWeight"),
					// "1D Boundary Smoothing" off means hold the boundary put.
					pinBoundary: !params.getBool("Boundary"),
				});
				doc.Log.log(`Smoothed with ${steps} Laplacian steps`);
				break;
			}

			case FP.FP_TAUBIN_SMOOTH: {
				const steps = params.getInt("stepSmoothNum");
				Smooth.vertexCoordTaubin(cm, steps, params.getFloat("lambda"), params.getFloat("mu"), {
					smoothSelected,
				});
				doc.Log.log(`Smoothed with ${steps} Taubin steps`);
				break;
			}

			case FP.FP_HC_LAPLACIAN_SMOOTH:
				Smooth.vertexCoordLaplacianHC(cm, 1, 0, 0.5, { smoothSelected });
				doc.Log.log("Smoothed with HC Laplacian");
				break;

			case FP.FP_SD_LAPLACIAN_SMOOTH: {
				const steps = params.getInt("stepSmoothNum");
				Smooth.vertexCoordScaleDependentLaplacian(cm, steps, params.getFloat("delta"), {
					smoothSelected,
				});
				doc.Log.log(`Smoothed with ${steps} scale-dependent Laplacian steps`);
				break;
			}

			default:
				return this.wrongActionCalled(id);
		}

		m.updateBoxAndNormals();
		return {};
	}
}

function hasSelectedFaces(cm: CMeshO): boolean {
	for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f) && cm.isFaceS(f)) return true;
	return false;
}

/**
 * Marks the vertices of the selected faces.
 *
 * Smoothing works on vertices, but the selection a user makes is on faces, so
 * the two have to be reconciled before "affect only the selection" can mean
 * anything.
 */
function selectVerticesFromFaces(cm: CMeshO): void {
	for (let v = 0; v < cm.vertSize; v++) cm.vertFlags[v] &= ~0x0020;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) cm.vertFlags[cm.fv(f, k)] |= 0x0020;
	}
}
