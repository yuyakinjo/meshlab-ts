/**
 * `filter_colorproc` — the colour and quality channels.
 *
 * Two families that share a plugin because they share a purpose: adjust the
 * colours a mesh already carries (the photo-editing operations), and turn a
 * per-element scalar into a colour so it can be looked at (the quality
 * mappings). The transfer filters move either channel between vertices and
 * faces.
 *
 * Everything here is per-element and order-independent, with one exception —
 * the two Laplacian smoothers, which read neighbours and so are written as
 * gather passes into a scratch buffer rather than in place.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichColor,
	RichDynamicFloat,
	RichEnum,
	RichFloat,
	RichInt,
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
import { MLException, MLNotImplementedException } from "../../common/utilities/ml_exception.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { triQuality } from "../../vcg/complex/edge_ops.ts";
import { Rng } from "../../vcg/complex/point_sampling.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import {
	ALL_CHANNELS,
	alpha,
	BLUE_CHANNEL,
	brightnessContrast,
	colorRamp,
	desaturate,
	fromHsv,
	GREEN_CHANNEL,
	invert,
	lerpColor,
	levels,
	lightness,
	RED_CHANNEL,
	rgba,
	whiteBalance,
} from "../../vcg/space/color4.ts";

export const CP = {
	CP_FILLING: 0,
	CP_THRESHOLDING: 1,
	CP_CONTR_BRIGHT: 2,
	CP_INVERT: 3,
	CP_LEVELS: 4,
	CP_COLOURISATION: 5,
	CP_DESATURATION: 6,
	CP_WHITE_BAL: 7,
	CP_COLOR_NOISE: 8,
	CP_MAP_VQUALITY_INTO_COLOR: 9,
	CP_MAP_FQUALITY_INTO_COLOR: 10,
	CP_CLAMP_QUALITY: 11,
	CP_VERTEX_TO_FACE: 12,
	CP_FACE_TO_VERTEX: 13,
	CP_VERTEX_TO_FACE_QUALITY: 15,
	CP_FACE_TO_VERTEX_QUALITY: 16,
	CP_RANDOM_FACE: 17,
	CP_RANDOM_CONNECTED_COMPONENT: 18,
	CP_VERTEX_SMOOTH: 19,
	CP_FACE_SMOOTH: 20,
	CP_TRIANGLE_QUALITY: 21,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly requirements: number;
	readonly postCondition: number;
}

const V_COLOR = MeshElement.MM_VERTCOLOR;
const F_COLOR = MeshElement.MM_FACECOLOR;
const V_QUALITY = MeshElement.MM_VERTQUALITY;
const F_QUALITY = MeshElement.MM_FACEQUALITY;

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[CP.CP_FILLING]: {
		name: "Vertex Color Filling",
		pythonName: "set_color_per_vertex",
		info: "Fills the color of the vertices of the mesh with a color chosen by the user.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_THRESHOLDING]: {
		name: "Vertex Color Thresholding",
		pythonName: "apply_color_thresholding_per_vertex",
		info: "Colors the vertices of the mesh using two colors according to a lightness border (threshold).",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_CONTR_BRIGHT]: {
		name: "Vertex Color Brightness Contrast Gamma",
		pythonName: "apply_color_brightness_contrast_gamma_per_vertex",
		info: "Change color values applying brightness, contrast and gamma correction to the mesh.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_INVERT]: {
		name: "Vertex Color Invert",
		pythonName: "apply_color_inverse_per_vertex",
		info: "Inverts the colors of the vertices of the mesh.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_LEVELS]: {
		name: "Vertex Color Levels Adjustment",
		pythonName: "apply_color_level_adjustment_per_vertex",
		info: "The filter allows adjustment of color levels. It is a custom implementation of the Levels tool of Photoshop and Gimp.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_COLOURISATION]: {
		name: "Vertex Color Colourisation",
		pythonName: "apply_color_intensity_colourisation_per_vertex",
		info: "Allows the application of a color to the mesh. In spite of the Fill filter, this filter works on Hue Saturation Lightness parameters.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_DESATURATION]: {
		name: "Vertex Color Desaturation",
		pythonName: "apply_color_desaturation_per_vertex",
		info: "The filter desaturates the colors of the mesh. This provides a simple way to convert a mesh in gray tones.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_WHITE_BAL]: {
		name: "Vertex Color White Balance",
		pythonName: "apply_color_white_balance_per_vertex",
		info: "Applies a color cast removal operation to unbalanced photos.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_COLOR_NOISE]: {
		name: "Color noise",
		pythonName: "apply_color_noising_per_vertex",
		info: "Adds to the color the requested amount of bits of noise. Bits of noise are added independently for each RGB channel.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_MAP_VQUALITY_INTO_COLOR]: {
		name: "Colorize by vertex Quality",
		pythonName: "compute_color_from_scalar_per_vertex",
		info: "Color vertices depending on their quality field (manually equalized).",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR | V_QUALITY,
		postCondition: V_COLOR,
	},
	[CP.CP_MAP_FQUALITY_INTO_COLOR]: {
		name: "Colorize by face Quality",
		pythonName: "compute_color_from_scalar_per_face",
		info: "Color faces depending on their quality field (manually equalized).",
		filterClass: FilterClass.FaceColoring,
		requirements: F_COLOR | F_QUALITY,
		postCondition: F_COLOR,
	},
	[CP.CP_CLAMP_QUALITY]: {
		name: "Clamp Vertex Quality",
		pythonName: "apply_scalar_clamping_per_vertex",
		info: "Clamps vertex quality values to a given range according to specific values or to percentiles",
		filterClass: FilterClass.Quality,
		requirements: V_QUALITY,
		postCondition: V_QUALITY,
	},
	[CP.CP_VERTEX_TO_FACE]: {
		name: "Transfer Color: Vertex to Face",
		pythonName: "compute_color_transfer_vertex_to_face",
		info: "Face color is computed as average of vertex color",
		filterClass: FilterClass.FaceColoring,
		requirements: V_COLOR | F_COLOR,
		postCondition: F_COLOR,
	},
	[CP.CP_FACE_TO_VERTEX]: {
		name: "Transfer Color: Face to Vertex",
		pythonName: "compute_color_transfer_face_to_vertex",
		info: "Vertex color is computed as an average of the surrounding faces",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR | F_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_VERTEX_TO_FACE_QUALITY]: {
		name: "Transfer Quality: Vertex to Face",
		pythonName: "compute_scalar_transfer_vertex_to_face",
		info: "Face quality is computed as average of vertex quality.",
		filterClass: FilterClass.Quality,
		requirements: V_QUALITY | F_QUALITY,
		postCondition: F_QUALITY,
	},
	[CP.CP_FACE_TO_VERTEX_QUALITY]: {
		name: "Transfer Quality: Face to Vertexerror!",
		pythonName: "compute_scalar_transfer_face_to_vertex",
		info: "Vertex quality is computed as an average of the surrounding faces.",
		filterClass: FilterClass.Quality,
		requirements: V_QUALITY | F_QUALITY,
		postCondition: V_QUALITY,
	},
	[CP.CP_RANDOM_FACE]: {
		name: "Random Face Color",
		pythonName: "compute_color_random_per_face",
		info: "Colorize Faces randomly. If internal edges are present they are used. Useful for quads.",
		filterClass: FilterClass.FaceColoring,
		requirements: F_COLOR,
		postCondition: F_COLOR,
	},
	[CP.CP_RANDOM_CONNECTED_COMPONENT]: {
		name: "Random Component Color",
		pythonName: "compute_color_by_conntected_component_per_face",
		info: "Colorize each connected component randomly.",
		filterClass: FilterClass.FaceColoring,
		requirements: F_COLOR | MeshElement.MM_FACEFACETOPO,
		postCondition: F_COLOR,
	},
	[CP.CP_VERTEX_SMOOTH]: {
		name: "Smooth: Laplacian Vertex Color",
		pythonName: "apply_color_laplacian_smoothing_per_vertex",
		info: "Laplacian smooth of the color values of the mesh, the color of each vertex is averaged with the color of the adjacent vertices.",
		filterClass: FilterClass.VertexColoring,
		requirements: V_COLOR,
		postCondition: V_COLOR,
	},
	[CP.CP_FACE_SMOOTH]: {
		name: "Smooth: Laplacian Face Color",
		pythonName: "apply_color_laplacian_smoothing_per_face",
		info: "Laplacian smooth of the color values of the faces of the mesh, the color of each face is averaged with the color of the adjacent faces.",
		filterClass: FilterClass.FaceColoring,
		requirements: F_COLOR | MeshElement.MM_FACEFACETOPO,
		postCondition: F_COLOR,
	},
	[CP.CP_TRIANGLE_QUALITY]: {
		name: "Per Face Quality according to Triangle shape and aspect ratio",
		pythonName: "compute_scalar_by_aspect_ratio_per_face",
		info: "Compute a quality and colorize faces depending on triangle shape:<ol><li>area/max side of triangle<li>ratio inradius/circumradius (radii of incircle and circumcircle)<li>Mean ratio of triangle = area/(a*a + b*b + c*c) where a,b,c are the sides</ol>",
		filterClass: FilterClass.FaceColoring | FilterClass.Quality,
		requirements: F_QUALITY,
		postCondition: F_QUALITY,
	},
};

export class FilterColorProc extends FilterPlugin {
	pluginName(): string {
		return "FilterColorProc";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(CP);
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
		return this.spec(id).filterClass;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override getRequirements(id: ActionIDType): number {
		return this.spec(id).requirements;
	}
	override postCondition(id: ActionIDType): number {
		return this.spec(id).postCondition;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const onSelected = () =>
			list.add(
				new RichBool("onSelected", false, {
					description: "Only on selection",
					tooltip: "If checked, only affects selected vertices",
				}),
			);

		switch (id) {
			case CP.CP_FILLING:
				list.add(
					new RichColor("color1", rgba(255, 255, 255), {
						description: "Color:",
						tooltip: "Sets the color to apply to vertices.",
					}),
				);
				onSelected();
				break;

			case CP.CP_THRESHOLDING:
				list.add(
					new RichColor("color1", rgba(0, 0, 0), {
						description: "Color 1:",
						tooltip: "Sets the color to apply below the threshold.",
					}),
				);
				list.add(
					new RichColor("color2", rgba(255, 255, 255), {
						description: "Color 2:",
						tooltip: "Sets the color to apply above the threshold.",
					}),
				);
				list.add(
					new RichDynamicFloat("threshold", 128, 0, 255, {
						description: "Threshold:",
						tooltip:
							"Vertices with color above the lightness threshold becomes Color 2, the others Color 1.",
					}),
				);
				onSelected();
				break;

			case CP.CP_CONTR_BRIGHT:
				list.add(
					new RichDynamicFloat("brightness", 0, -255, 255, {
						description: "Brightness:",
						tooltip:
							"Sets the amount of brightness that will be added/subtracted to the colors." +
							"<br>Brightness = 255  ->  all white;<br>Brightness = -255  ->  all black;",
					}),
				);
				list.add(
					new RichDynamicFloat("contrast", 0, -255, 255, {
						description: "Contrast factor:",
						tooltip: "Sets the amount of contrast of the mesh.",
					}),
				);
				list.add(
					new RichDynamicFloat("gamma", 1, 0.1, 5, {
						description: "Gamma:",
						tooltip: "Sets the values of the exponent gamma.",
					}),
				);
				onSelected();
				break;

			case CP.CP_INVERT:
				onSelected();
				break;

			case CP.CP_LEVELS:
				list.add(new RichDynamicFloat("gamma", 1, 0.1, 5, { description: "Gamma:" }));
				list.add(new RichDynamicFloat("in_min", 0, 0, 255, { description: "Min input level:" }));
				list.add(new RichDynamicFloat("in_max", 255, 0, 255, { description: "Max input level:" }));
				list.add(new RichDynamicFloat("out_min", 0, 0, 255, { description: "Min output level:" }));
				list.add(
					new RichDynamicFloat("out_max", 255, 0, 255, { description: "Max output level:" }),
				);
				list.add(new RichBool("rCh", true, { description: "Red Channel:" }));
				list.add(new RichBool("gCh", true, { description: "Green Channel:" }));
				list.add(new RichBool("bCh", true, { description: "Blue Channel:" }));
				onSelected();
				list.add(
					new RichBool("apply_to_all", false, {
						description: "All visible layers",
						tooltip: "if true, apply to all visible layers",
					}),
				);
				break;

			case CP.CP_COLOURISATION:
				list.add(
					new RichDynamicFloat("hue", 0, 0, 360, {
						description: "Hue:",
						tooltip: "Changes the hue of the mesh.",
					}),
				);
				list.add(
					new RichDynamicFloat("saturation", 100, 0, 100, {
						description: "Saturation:",
						tooltip: "Changes the saturation of the mesh.",
					}),
				);
				list.add(
					new RichDynamicFloat("luminance", 100, 0, 100, {
						description: "Luminance:",
						tooltip: "Changes the luminance of the mesh.",
					}),
				);
				list.add(
					new RichDynamicFloat("intensity", 50, 0, 100, {
						description: "Blending:",
						tooltip: "Sets the blending factor used in adding the new color to the existing one.",
					}),
				);
				onSelected();
				break;

			case CP.CP_DESATURATION:
				list.add(
					new RichEnum("method", 0, ["Lightness", "Luminosity", "Average"], {
						description: "Desaturation method:",
						tooltip:
							"Lightness is computed as (Max(r,g,b)+Min(r,g,b))/2<br>Luminosity is computed as " +
							"0.212*r + 0.715*g + 0.072*b<br>Average is computed as (r+g+b)/3",
					}),
				);
				onSelected();
				break;

			case CP.CP_WHITE_BAL:
				list.add(
					new RichColor("color", rgba(255, 255, 255), {
						description: "Unbalanced white: ",
						tooltip: "The color that is supposed to be white.",
					}),
				);
				onSelected();
				break;

			case CP.CP_COLOR_NOISE:
				list.add(
					new RichInt("noiseBits", 1, {
						description: "Noise bits:",
						tooltip:
							"Bits of noise added to each RGB channel. Example: 3 noise bits adds three random " +
							"offsets in the [-4,+4] interval to each RGB channels.",
					}),
				);
				onSelected();
				break;

			case CP.CP_MAP_VQUALITY_INTO_COLOR:
			case CP.CP_MAP_FQUALITY_INTO_COLOR:
			case CP.CP_CLAMP_QUALITY: {
				const range = qualityRange(m, id === CP.CP_MAP_FQUALITY_INTO_COLOR);
				list.add(
					new RichFloat("minVal", range.min, {
						description: "Min",
						tooltip: "The value that will be mapped with the lower end of the scale (red)",
					}),
				);
				list.add(
					new RichFloat("maxVal", range.max, {
						description: "Max",
						tooltip: "The value that will be mapped with the upper end of the scale (blue)",
					}),
				);
				list.add(
					new RichDynamicFloat("perc", 0, 0, 100, {
						description: "Percentile Crop [0..100]",
						tooltip:
							"If not zero this value will be used for a percentile cropping of the quality " +
							"values.<br> If this parameter is set to a value <i>P</i> then the two values " +
							"<i>V_min,V_max</i> for which <i>P</i>% of the vertices have a quality <b>lower or " +
							"greater</b> than <i>V_min,V_max</i> are used as min/max values for clamping.<br><br> " +
							"The automated percentile cropping is very useful for automatically discarding outliers.",
					}),
				);
				list.add(
					new RichBool("zeroSym", false, {
						description: "Zero Symmetric",
						tooltip:
							"If true the min max range will be enlarged to be symmetric (so that green is always Zero)",
					}),
				);
				break;
			}

			case CP.CP_VERTEX_SMOOTH:
			case CP.CP_FACE_SMOOTH:
				list.add(
					new RichInt("iteration", 1, {
						description: "Iteration",
						tooltip: "the number of iteration of the smoothing algorithm",
					}),
				);
				break;

			case CP.CP_TRIANGLE_QUALITY:
				list.add(
					new RichEnum(
						"Metric",
						0,
						[
							"area/max side",
							"inradius/circumradius",
							"Mean ratio",
							"Area",
							"Texture Angle Distortion",
							"Texture Area Distortion",
							"Polygonal planarity (max)",
							"Polygonal planarity (relative)",
						],
						{
							description: "Metric:",
							tooltip: "Choose a metric to compute triangle quality.",
						},
					),
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
		post.mask = this.spec(id).postCondition;
		const only = params.hasParameter("onSelected") && params.getBool("onSelected");

		switch (id) {
			case CP.CP_FILLING:
				return mapVertices(cm, only, () => params.getColor("color1"));

			case CP.CP_THRESHOLDING: {
				const below = params.getColor("color1");
				const above = params.getColor("color2");
				const threshold = params.getDynamicFloat("threshold");
				return mapVertices(cm, only, (c) => (lightness(c) <= threshold ? below : above));
			}

			case CP.CP_CONTR_BRIGHT: {
				// MeshLab's sliders run -255..255 and 0.1..5; the arithmetic
				// underneath wants -1..1 for the first two and a plain exponent
				// for gamma, applied through the levels transform.
				const brightness = params.getDynamicFloat("brightness") / 255;
				const contrast = params.getDynamicFloat("contrast") / 255;
				const gamma = params.getDynamicFloat("gamma");
				return mapVertices(cm, only, (c) =>
					levels(brightnessContrast(c, brightness, contrast), gamma, 0, 1, 0, 1, ALL_CHANNELS),
				);
			}

			case CP.CP_INVERT:
				return mapVertices(cm, only, invert);

			case CP.CP_LEVELS: {
				if (params.getBool("apply_to_all")) {
					throw new MLNotImplementedException(
						'Vertex Color Levels Adjustment does not support "All visible layers" yet; ' +
							"apply it to one layer at a time.",
						"FilterColorProc",
					);
				}
				const mask =
					(params.getBool("rCh") ? RED_CHANNEL : 0) |
					(params.getBool("gCh") ? GREEN_CHANNEL : 0) |
					(params.getBool("bCh") ? BLUE_CHANNEL : 0);
				const gamma = params.getDynamicFloat("gamma");
				// The sliders are in 0..255; the transform works in 0..1.
				const inMin = params.getDynamicFloat("in_min") / 255;
				const inMax = params.getDynamicFloat("in_max") / 255;
				const outMin = params.getDynamicFloat("out_min") / 255;
				const outMax = params.getDynamicFloat("out_max") / 255;
				return mapVertices(cm, only, (c) => levels(c, gamma, inMin, inMax, outMin, outMax, mask));
			}

			case CP.CP_COLOURISATION: {
				const target = fromHsv(
					params.getDynamicFloat("hue"),
					params.getDynamicFloat("saturation") / 100,
					params.getDynamicFloat("luminance") / 100,
				);
				const intensity = params.getDynamicFloat("intensity") / 100;
				return mapVertices(cm, only, (c) => lerpColor(c, target, intensity));
			}

			case CP.CP_DESATURATION: {
				const method = params.getEnum("method");
				return mapVertices(cm, only, (c) => desaturate(c, method));
			}

			case CP.CP_WHITE_BAL: {
				const unbalanced = params.getColor("color");
				return mapVertices(cm, only, (c) => whiteBalance(c, unbalanced));
			}

			case CP.CP_COLOR_NOISE: {
				const bits = params.getInt("noiseBits");
				if (bits < 0) throw new MLException(`Noise bits cannot be negative, got ${bits}`);
				const span = 2 ** bits;
				const rng = new Rng();
				// Each channel gets its own draw, which is what makes the result
				// coloured noise rather than a brightness wobble.
				const jitter = () => Math.round((rng.next() * 2 - 1) * span);
				return mapVertices(cm, only, (c) =>
					rgba(redOf(c) + jitter(), greenOf(c) + jitter(), blueOf(c) + jitter(), alpha(c)),
				);
			}

			case CP.CP_MAP_VQUALITY_INTO_COLOR:
			case CP.CP_MAP_FQUALITY_INTO_COLOR:
			case CP.CP_CLAMP_QUALITY: {
				const perFace = id === CP.CP_MAP_FQUALITY_INTO_COLOR;
				const values = liveQuality(cm, perFace);
				if (values.length === 0) throw new MLException("The mesh has no elements to work on.");
				let { min, max } = resolveRange(
					values,
					params.getFloat("minVal"),
					params.getFloat("maxVal"),
					params.getDynamicFloat("perc"),
				);
				if (params.getBool("zeroSym")) {
					// Widen to whichever side is further from zero, so the middle
					// of the ramp is exactly zero and the sign is readable.
					const reach = Math.max(Math.abs(min), Math.abs(max));
					min = -reach;
					max = reach;
				}

				if (id === CP.CP_CLAMP_QUALITY) {
					let clamped = 0;
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v)) continue;
						const q = cm.vertQuality[v];
						const next = q < min ? min : q > max ? max : q;
						if (next !== q) clamped++;
						cm.vertQuality[v] = next;
					}
					doc.Log.log(`Clamped ${clamped} vertex quality values into [${min}, ${max}]`);
					return { min_value: min, max_value: max, clamped };
				}

				if (perFace) {
					m.updateDataMask(F_COLOR);
					const colors = requireFaceColor(cm);
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						colors[f] = colorRamp(min, max, faceQualityOf(cm, f));
					}
				} else {
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v)) continue;
						cm.vertColor[v] = colorRamp(min, max, cm.vertQuality[v]);
					}
				}
				doc.Log.log(`Quality mapped onto the ramp over [${min}, ${max}]`);
				return { min_value: min, max_value: max };
			}

			case CP.CP_VERTEX_TO_FACE: {
				m.updateDataMask(F_COLOR);
				const colors = requireFaceColor(cm);
				let n = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					colors[f] = averageColors([
						cm.vertColor[cm.fv(f, 0)],
						cm.vertColor[cm.fv(f, 1)],
						cm.vertColor[cm.fv(f, 2)],
					]);
					n++;
				}
				return { face_number: n };
			}

			case CP.CP_FACE_TO_VERTEX: {
				const colors = requireFaceColor(cm);
				return gatherFromFaces(cm, (indices) => averageColors(indices.map((f) => colors[f])));
			}

			case CP.CP_VERTEX_TO_FACE_QUALITY: {
				m.updateDataMask(F_QUALITY);
				const quality = requireFaceQuality(cm);
				let n = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					quality[f] =
						(cm.vertQuality[cm.fv(f, 0)] +
							cm.vertQuality[cm.fv(f, 1)] +
							cm.vertQuality[cm.fv(f, 2)]) /
						3;
					n++;
				}
				return { face_number: n };
			}

			case CP.CP_FACE_TO_VERTEX_QUALITY: {
				const quality = requireFaceQuality(cm);
				const incident = incidentFaces(cm);
				let n = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v) || incident[v].length === 0) continue;
					let sum = 0;
					for (const f of incident[v]) sum += quality[f];
					cm.vertQuality[v] = sum / incident[v].length;
					n++;
				}
				return { vertex_number: n };
			}

			case CP.CP_RANDOM_FACE: {
				m.updateDataMask(F_COLOR);
				const colors = requireFaceColor(cm);
				const rng = new Rng();
				let n = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					colors[f] = scatterColor(rng.next());
					n++;
				}
				return { face_number: n };
			}

			case CP.CP_RANDOM_CONNECTED_COMPONENT: {
				m.updateDataMask(F_COLOR);
				UpdateTopology.faceFace(cm);
				const colors = requireFaceColor(cm);
				const components = Clean.connectedComponents(cm);
				const rng = new Rng();
				for (const [, seed] of components) {
					const colour = scatterColor(rng.next());
					// Flood from the seed face over FF adjacency: every face of the
					// component takes the same colour, which is the whole point.
					const stack = [seed];
					const seen = new Set<number>([seed]);
					while (stack.length > 0) {
						const f = stack.pop() as number;
						colors[f] = colour;
						for (let e = 0; e < 3; e++) {
							if (cm.isBorderFF(f, e)) continue;
							const g = cm.ffp(f, e);
							if (g < 0 || cm.isFaceD(g) || seen.has(g)) continue;
							seen.add(g);
							stack.push(g);
						}
					}
				}
				doc.Log.log(`Coloured ${components.length} connected components`);
				return { component_number: components.length };
			}

			case CP.CP_VERTEX_SMOOTH: {
				const iterations = params.getInt("iteration");
				for (let i = 0; i < iterations; i++) {
					const next = Uint32Array.from(cm.vertColor);
					const neighbours = vertexNeighbours(cm);
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v) || neighbours[v].length === 0) continue;
						next[v] = averageColors([
							cm.vertColor[v],
							...neighbours[v].map((w) => cm.vertColor[w]),
						]);
					}
					cm.vertColor.set(next);
				}
				return { iterations };
			}

			case CP.CP_FACE_SMOOTH: {
				UpdateTopology.faceFace(cm);
				const colors = requireFaceColor(cm);
				const iterations = params.getInt("iteration");
				for (let i = 0; i < iterations; i++) {
					const next = Uint32Array.from(colors);
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						const group = [colors[f]];
						for (let e = 0; e < 3; e++) {
							if (cm.isBorderFF(f, e)) continue;
							const g = cm.ffp(f, e);
							if (g >= 0 && !cm.isFaceD(g)) group.push(colors[g]);
						}
						next[f] = averageColors(group);
					}
					colors.set(next);
				}
				return { iterations };
			}

			case CP.CP_TRIANGLE_QUALITY: {
				const metric = params.getEnum("Metric");
				if (metric > 2) {
					throw new MLNotImplementedException(
						`Triangle quality metric "${metric}" needs texture coordinates or polygonal faces, ` +
							"neither of which is supported yet; use one of the first three.",
						"FilterColorProc",
					);
				}
				m.updateDataMask(F_QUALITY);
				const quality = requireFaceQuality(cm);
				let n = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					quality[f] = shapeMetric(cm, f, metric);
					n++;
				}
				return { face_number: n };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}

const redOf = (c: number): number => c & 0xff;
const greenOf = (c: number): number => (c >>> 8) & 0xff;
const blueOf = (c: number): number => (c >>> 16) & 0xff;

/** Applies a colour function to every live vertex, or only the selected ones. */
function mapVertices(cm: CMeshO, onlySelected: boolean, fn: (c: number) => number): FilterOutput {
	let n = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		if (onlySelected && !cm.isVertS(v)) continue;
		cm.vertColor[v] = fn(cm.vertColor[v]);
		n++;
	}
	return { vertex_number: n };
}

/** Averages a vertex channel from the faces around it. */
function gatherFromFaces(cm: CMeshO, fn: (faces: number[]) => number): FilterOutput {
	const incident = incidentFaces(cm);
	let n = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v) || incident[v].length === 0) continue;
		cm.vertColor[v] = fn(incident[v]);
		n++;
	}
	return { vertex_number: n };
}

function incidentFaces(cm: CMeshO): number[][] {
	const out: number[][] = Array.from({ length: cm.vertSize }, () => []);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) out[cm.fv(f, k)].push(f);
	}
	return out;
}

function vertexNeighbours(cm: CMeshO): number[][] {
	const sets: Array<Set<number>> = Array.from({ length: cm.vertSize }, () => new Set<number>());
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = cm.fv(f, k);
			const b = cm.fv(f, (k + 1) % 3);
			sets[a].add(b);
			sets[b].add(a);
		}
	}
	return sets.map((s) => [...s]);
}

/** Channel-wise mean of a group of packed colours. */
function averageColors(colors: readonly number[]): number {
	if (colors.length === 0) return rgba(255, 255, 255);
	let r = 0;
	let g = 0;
	let b = 0;
	let a = 0;
	for (const c of colors) {
		r += redOf(c);
		g += greenOf(c);
		b += blueOf(c);
		a += alpha(c);
	}
	return rgba(r / colors.length, g / colors.length, b / colors.length, a / colors.length);
}

/**
 * A saturated colour from a number in 0..1.
 *
 * Full saturation and value on purpose: these are labels, meant to be told
 * apart at a glance, not to look like anything.
 */
function scatterColor(t: number): number {
	return fromHsv(t * 360, 0.7 + 0.3 * ((t * 7) % 1), 0.7 + 0.3 * ((t * 13) % 1));
}

function requireFaceColor(cm: CMeshO): Uint32Array {
	if (cm.faceColor === null) {
		throw new MLException("This filter needs per-face colour, which the mesh does not carry.");
	}
	return cm.faceColor;
}

function requireFaceQuality(cm: CMeshO): Float64Array {
	if (cm.faceQuality === null) {
		throw new MLException("This filter needs per-face quality, which the mesh does not carry.");
	}
	return cm.faceQuality;
}

const faceQualityOf = (cm: CMeshO, f: number): number =>
	cm.faceQuality === null ? 0 : cm.faceQuality[f];

/** The live per-vertex or per-face quality values. */
function liveQuality(cm: CMeshO, perFace: boolean): number[] {
	const out: number[] = [];
	if (perFace) {
		const q = cm.faceQuality;
		if (q === null) return out;
		for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f)) out.push(q[f]);
	} else {
		for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) out.push(cm.vertQuality[v]);
	}
	return out;
}

/**
 * The range to map onto the ramp.
 *
 * A non-zero percentile overrides the explicit min and max: it trims that
 * fraction off each end, which is how a single wild outlier stops flattening
 * the whole colouring into one band.
 */
function resolveRange(
	values: readonly number[],
	minVal: number,
	maxVal: number,
	percentile: number,
): { min: number; max: number } {
	if (percentile <= 0) return { min: minVal, max: maxVal };
	const sorted = [...values].sort((a, b) => a - b);
	const fraction = Math.min(49.9, percentile) / 100;
	const lo = Math.floor(fraction * (sorted.length - 1));
	const hi = Math.ceil((1 - fraction) * (sorted.length - 1));
	return { min: sorted[lo], max: sorted[hi] };
}

/** The first three of MeshLab's triangle-shape metrics. */
function shapeMetric(cm: CMeshO, f: number, metric: number): number {
	const a = cm.fv(f, 0);
	const b = cm.fv(f, 1);
	const c = cm.fv(f, 2);
	const side = (p: number, q: number) =>
		Math.hypot(cm.vx(p) - cm.vx(q), cm.vy(p) - cm.vy(q), cm.vz(p) - cm.vz(q));
	const la = side(b, c);
	const lb = side(c, a);
	const lc = side(a, b);
	const s = (la + lb + lc) / 2;
	const area = Math.sqrt(Math.max(0, s * (s - la) * (s - lb) * (s - lc)));

	switch (metric) {
		case 0:
			// area / longest side, the same measure the collapse guard uses.
			return triQuality(
				cm.vx(a),
				cm.vy(a),
				cm.vz(a),
				cm.vx(b),
				cm.vy(b),
				cm.vz(b),
				cm.vx(c),
				cm.vy(c),
				cm.vz(c),
			);
		case 1: {
			// inradius / circumradius, normalised so an equilateral triangle
			// scores exactly 1 rather than 1/2.
			if (area === 0 || s === 0) return 0;
			const inradius = area / s;
			const circumradius = (la * lb * lc) / (4 * area);
			return (2 * inradius) / circumradius;
		}
		default: {
			// Mean ratio: 4·sqrt(3)·area / (a² + b² + c²), also 1 when equilateral.
			const sum = la * la + lb * lb + lc * lc;
			return sum === 0 ? 0 : (4 * Math.sqrt(3) * area) / sum;
		}
	}
}

/** Whatever quality range the mesh currently spans, for the parameter defaults. */
function qualityRange(m: MeshModel | undefined, perFace: boolean): { min: number; max: number } {
	if (m === undefined) return { min: 0, max: 1 };
	const values = liveQuality(m.cm, perFace);
	if (values.length === 0) return { min: 0, max: 1 };
	return { min: Math.min(...values), max: Math.max(...values) };
}
