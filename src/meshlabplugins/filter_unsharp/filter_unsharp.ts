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
import {
	RichBool,
	RichDynamicFloat,
	RichEnum,
	RichFloat,
	RichInt,
	RichMesh,
} from "../../common/parameters/rich_parameter.ts";
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
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { Smooth } from "../../vcg/complex/smooth.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { rgba } from "../../vcg/space/color4.ts";

export const FP = {
	FP_LAPLACIAN_SMOOTH: 0,
	FP_TAUBIN_SMOOTH: 1,
	FP_HC_LAPLACIAN_SMOOTH: 2,
	FP_SD_LAPLACIAN_SMOOTH: 3,
	FP_RECOMPUTE_FACE_NORMAL: 4,
	FP_RECOMPUTE_VERTEX_NORMAL: 5,
	FP_FACE_NORMAL_NORMALIZE: 6,
	FP_VERTEX_NORMAL_NORMALIZE: 7,
	FP_FACE_NORMAL_SMOOTHING: 8,
	FP_VERTEX_QUALITY_SMOOTHING: 9,
	FP_UNSHARP_GEOMETRY: 10,
	FP_UNSHARP_NORMAL: 11,
	FP_UNSHARP_VERTEX_COLOR: 12,
	FP_UNSHARP_QUALITY: 13,
	FP_LINEAR_MORPH: 14,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	/** Defaults to Smoothing, which is what the four coordinate smoothers are. */
	readonly filterClass?: FilterClassMask;
	readonly arity?: FilterArityValue;
	readonly requirements?: number;
	/** Defaults to the geometry a coordinate smoother touches. */
	readonly postCondition?: number;
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
	[FP.FP_RECOMPUTE_FACE_NORMAL]: {
		name: "Re-Compute Face Normals",
		pythonName: "compute_normal_per_face",
		info: "Recompute face normals as the normal of the plane of the face.<br>",
		filterClass: FilterClass.Normal,
		postCondition: MeshElement.MM_FACENORMAL,
	},
	[FP.FP_RECOMPUTE_VERTEX_NORMAL]: {
		name: "Re-Compute Vertex Normals",
		pythonName: "compute_normal_per_vertex",
		info:
			"Recompute vertex normals according to four different schemes:<br>" +
			"1) simple average of normals of the incident faces <br>" +
			"2) area weighted average of normals of the incident faces <br>" +
			"3) angle weighted sum of normals of the incident faces according to " +
			"<i>Computing Vertex Normals from Polygonal Facets</i> Grit Thurmer and Charles A. " +
			"Wuthrich, JGT 1998<br> 4) weighted sum of normals of the incident faces, as defined " +
			"by <i>Weights for Computing Vertex Normals from Facet Normals</i>, N.Max, JGT 1999",
		filterClass: FilterClass.Normal,
		postCondition: MeshElement.MM_VERTNORMAL,
	},
	[FP.FP_FACE_NORMAL_NORMALIZE]: {
		name: "Normalize Face Normals",
		pythonName: "apply_normal_normalization_per_face",
		info: "Normalize Face Normal Lengths to unit vectors.",
		filterClass: FilterClass.Normal,
		postCondition: MeshElement.MM_FACENORMAL,
	},
	[FP.FP_VERTEX_NORMAL_NORMALIZE]: {
		name: "Normalize Vertex Normals",
		pythonName: "apply_normal_normalization_per_vertex",
		info: "Normalize Vertex Normal Lengths to unit vectors.",
		filterClass: FilterClass.Normal,
		postCondition: MeshElement.MM_VERTNORMAL,
	},
	[FP.FP_FACE_NORMAL_SMOOTHING]: {
		name: "Smooth Face Normals",
		pythonName: "apply_normal_smoothing_per_face",
		info: "Laplacian smooth of the face normals, without touching the position of the vertices.",
		filterClass: FilterClass.Smoothing,
		requirements: MeshElement.MM_FACEFACETOPO,
		postCondition: MeshElement.MM_FACENORMAL,
	},
	[FP.FP_VERTEX_QUALITY_SMOOTHING]: {
		name: "Smooth Vertex Quality",
		pythonName: "apply_scalar_smoothing_per_vertex",
		info: "Laplacian smooth of the quality values.",
		filterClass: FilterClass.Smoothing,
		requirements: MeshElement.MM_VERTQUALITY,
		postCondition: MeshElement.MM_VERTQUALITY,
	},
	[FP.FP_UNSHARP_GEOMETRY]: {
		name: "UnSharp Mask Geometry",
		pythonName: "apply_coord_unsharp_mask",
		info: "Apply Unsharp filter to geometric shape, putting in more evidence ridges and valleys variations.<br>",
		filterClass: FilterClass.Smoothing,
		postCondition: MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL | MeshElement.MM_FACENORMAL,
	},
	[FP.FP_UNSHARP_NORMAL]: {
		name: "UnSharp Mask Normals",
		pythonName: "apply_normal_unsharp_mask_per_vertex",
		info:
			"Unsharp mask filtering of the normals per face, putting in more evidence normal " +
			"variations.<br>",
		filterClass: FilterClass.Smoothing,
		requirements: MeshElement.MM_FACEFACETOPO,
		postCondition: MeshElement.MM_FACENORMAL,
	},
	[FP.FP_UNSHARP_VERTEX_COLOR]: {
		name: "UnSharp Mask Color",
		pythonName: "apply_color_unsharp_mask_per_vertex",
		info: "Apply Unsharp filter to the vertex color, putting in more evidence color variations.<br>",
		filterClass: FilterClass.VertexColoring | FilterClass.Smoothing,
		requirements: MeshElement.MM_VERTCOLOR,
		postCondition: MeshElement.MM_VERTCOLOR,
	},
	[FP.FP_UNSHARP_QUALITY]: {
		name: "UnSharp Mask Quality",
		pythonName: "apply_scalar_unsharp_mask_per_vertex",
		info: "Apply Unsharp filter to values of quality, putting in more evidence variations.<br>",
		filterClass: FilterClass.Smoothing,
		requirements: MeshElement.MM_VERTQUALITY,
		postCondition: MeshElement.MM_VERTQUALITY,
	},
	[FP.FP_LINEAR_MORPH]: {
		name: "Vertex Linear Morphing",
		pythonName: "compute_coord_linear_morphing",
		info:
			"Morph deformation of current mesh towards a target mesh with the same number of " +
			"vertices. The filter assumes that the two meshes have also the same vertex ordering.",
		filterClass: FilterClass.Smoothing,
		arity: FilterArity.FIXED,
		postCondition: MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL | MeshElement.MM_FACENORMAL,
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
	override getClass(id: ActionIDType): FilterClassMask {
		return this.spec(id).filterClass ?? FilterClass.Smoothing;
	}
	filterArity(id: ActionIDType): FilterArityValue {
		return this.spec(id).arity ?? FilterArity.SINGLE_MESH;
	}
	override getRequirements(id: ActionIDType): number {
		return this.spec(id).requirements ?? MeshElement.MM_NONE;
	}

	/** Smoothing moves vertices but never changes which faces exist. */
	override postCondition(id: ActionIDType): number {
		return (
			this.spec(id).postCondition ??
			MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL | MeshElement.MM_FACENORMAL
		);
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

			case FP.FP_RECOMPUTE_VERTEX_NORMAL:
				list.add(
					new RichEnum(
						"weightMode",
						0,
						["Simple Average", "By Area", "By Angle", "As defined by N. Max"],
						{ description: "Weighting Mode:" },
					),
				);
				break;

			case FP.FP_UNSHARP_NORMAL:
			case FP.FP_UNSHARP_GEOMETRY:
			case FP.FP_UNSHARP_VERTEX_COLOR:
			case FP.FP_UNSHARP_QUALITY:
				// Only the normal mask offers this; the other three have
				// nothing to recompute.
				if (id === FP.FP_UNSHARP_NORMAL) {
					list.add(
						new RichBool("recalc", false, {
							description: "Recompute Normals",
							tooltip: "Recompute normals from scratch before the unsharp masking",
						}),
					);
				}
				list.add(
					new RichFloat("weight", 0.3, {
						description: "Unsharp Weight",
						tooltip:
							"the unsharp weight <i>alpha<sub>us</sub></i> in the formula <br>" +
							"<i>result = weight * original + weight<sub>orig</sub> * (original - smoothed)</i><br>",
					}),
				);
				list.add(
					new RichFloat("weightOrig", 1, {
						description: "Original Weight",
						tooltip: "How much the original signal is used to compute the unsharp result<br>",
					}),
				);
				list.add(
					new RichInt("iterations", 5, {
						description: "Smooth Iterations",
						tooltip:
							"number of iterations of the augmented laplacian smoothing that is used to " +
							"build the low pass filtered version",
					}),
				);
				break;

			case FP.FP_LINEAR_MORPH:
				list.add(
					new RichMesh("TargetMesh", 0, {
						description: "Target Mesh",
						tooltip: "The mesh that is the morph target.",
					}),
				);
				list.add(
					new RichDynamicFloat("PercentMorph", 0, -150, 250, {
						description: "% Morph",
						tooltip:
							"The percent you want to morph towards (or away from) the target. <br>" +
							"0 means current mesh <br>100 means targe mesh <br>" +
							"<0 and >100 linearly extrapolate between the two mesh <br>",
					}),
				);
				break;

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

			case FP.FP_RECOMPUTE_FACE_NORMAL:
				UpdateNormal.perFace(cm);
				return { face_number: cm.fn };

			case FP.FP_RECOMPUTE_VERTEX_NORMAL: {
				// Upstream normalises the face normals first, so the four
				// schemes differ only in their weights and not in the lengths
				// they happen to be summing.
				UpdateNormal.normalizePerFace(cm);
				const mode = params.getEnum("weightMode");
				if (mode === 2) UpdateNormal.perVertexAngleWeighted(cm);
				else if (mode === 1) UpdateNormal.perVertexPerFace(cm);
				else if (mode === 3) UpdateNormal.perVertexNelsonMaxWeighted(cm);
				else UpdateNormal.perVertexSimpleAverage(cm);
				UpdateNormal.normalizePerVertex(cm);
				return { vertex_number: cm.vn };
			}

			case FP.FP_FACE_NORMAL_NORMALIZE:
				UpdateNormal.normalizePerFace(cm);
				return { face_number: cm.fn };

			case FP.FP_VERTEX_NORMAL_NORMALIZE:
				UpdateNormal.normalizePerVertex(cm);
				return { vertex_number: cm.vn };

			case FP.FP_FACE_NORMAL_SMOOTHING:
				UpdateTopology.faceFace(cm);
				Smooth.faceNormalLaplacianFF(cm);
				return { face_number: cm.fn };

			case FP.FP_VERTEX_QUALITY_SMOOTHING:
				Smooth.vertexQualityLaplacian(cm);
				return { vertex_number: cm.vn };

			case FP.FP_UNSHARP_GEOMETRY: {
				const { alpha, alphaOrig, iterations } = unsharpWeights(params);
				Allocator.compactVertexVector(cm);
				const original = Float64Array.from(cm.vertCoord.subarray(0, cm.vn * 3));
				Smooth.vertexCoordLaplacian(cm, iterations);
				for (let v = 0; v < cm.vn; v++) {
					// The mask: keep `alphaOrig` of the signal and add back
					// `alpha` of whatever the low pass threw away.
					cm.setVert(
						v,
						unsharp(original[3 * v], cm.vx(v), alpha, alphaOrig),
						unsharp(original[3 * v + 1], cm.vy(v), alpha, alphaOrig),
						unsharp(original[3 * v + 2], cm.vz(v), alpha, alphaOrig),
					);
				}
				break;
			}

			case FP.FP_UNSHARP_NORMAL: {
				const { alpha, alphaOrig, iterations } = unsharpWeights(params);
				Allocator.compactFaceVector(cm);
				if (params.getBool("recalc")) UpdateNormal.perFace(cm);
				UpdateTopology.faceFace(cm);
				const original = Float64Array.from(cm.faceNormal.subarray(0, cm.fn * 3));
				Smooth.faceNormalLaplacianFF(cm, iterations);
				for (let i = 0; i < cm.fn * 3; i++) {
					cm.faceNormal[i] = unsharp(original[i], cm.faceNormal[i], alpha, alphaOrig);
				}
				// The postcondition would otherwise recompute what was just
				// written, so this one reports the normals as already current.
				post.mask = MeshElement.MM_NONE;
				return { face_number: cm.fn };
			}

			case FP.FP_UNSHARP_VERTEX_COLOR: {
				const { alpha, alphaOrig, iterations } = unsharpWeights(params);
				Allocator.compactVertexVector(cm);
				const original = Uint32Array.from(cm.vertColor.subarray(0, cm.vn));
				Smooth.vertexColorLaplacian(cm, iterations);
				for (let v = 0; v < cm.vn; v++) {
					const o = original[v];
					const c = cm.vertColor[v];
					cm.vertColor[v] = rgba(
						unsharp(o & 0xff, c & 0xff, alpha, alphaOrig),
						unsharp((o >>> 8) & 0xff, (c >>> 8) & 0xff, alpha, alphaOrig),
						unsharp((o >>> 16) & 0xff, (c >>> 16) & 0xff, alpha, alphaOrig),
						(o >>> 24) & 0xff,
					);
				}
				return { vertex_number: cm.vn };
			}

			case FP.FP_UNSHARP_QUALITY: {
				const { alpha, alphaOrig, iterations } = unsharpWeights(params);
				Allocator.compactVertexVector(cm);
				const original = Float64Array.from(cm.vertQuality.subarray(0, cm.vn));
				Smooth.vertexQualityLaplacian(cm, iterations);
				for (let v = 0; v < cm.vn; v++) {
					cm.vertQuality[v] = unsharp(original[v], cm.vertQuality[v], alpha, alphaOrig);
				}
				return { vertex_number: cm.vn };
			}

			case FP.FP_LINEAR_MORPH: {
				const target = doc.requireMesh(params.getMeshId("TargetMesh")).cm;
				if (cm.vn !== target.vn) {
					throw new MLException(
						`Number of vertices is not the same (${cm.vn} against ${target.vn}), so you can't ` +
							"morph between these two meshes.",
					);
				}
				Allocator.compactEveryVector(cm);
				Allocator.compactEveryVector(target);
				// Beyond 0..100 this extrapolates rather than clamping, which is
				// upstream's behaviour and the reason the slider runs to 250.
				const t = params.getDynamicFloat("PercentMorph") / 100;
				for (let v = 0; v < cm.vn; v++) {
					cm.setVert(
						v,
						cm.vx(v) + (target.vx(v) - cm.vx(v)) * t,
						cm.vy(v) + (target.vy(v) - cm.vy(v)) * t,
						cm.vz(v) + (target.vz(v) - cm.vz(v)) * t,
					);
				}
				break;
			}

			default:
				return this.wrongActionCalled(id);
		}

		m.updateBoxAndNormals();
		return {};
	}
}

/** `result = original * weightOrig + (original - smoothed) * weight`. */
const unsharp = (original: number, smoothed: number, alpha: number, alphaOrig: number): number =>
	original * alphaOrig + (original - smoothed) * alpha;

function unsharpWeights(params: RichParameterList): {
	alpha: number;
	alphaOrig: number;
	iterations: number;
} {
	const iterations = params.getInt("iterations");
	if (iterations < 0)
		throw new MLException(`Smooth Iterations cannot be negative, got ${iterations}`);
	return { alpha: params.getFloat("weight"), alphaOrig: params.getFloat("weightOrig"), iterations };
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
