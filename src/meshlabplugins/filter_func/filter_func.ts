/**
 * `filter_func` — the filters that take an expression from the user and run it
 * once per element.
 *
 * MeshLab hands muParser a table of per-element variables and evaluates the
 * user's formula over every vertex or face. What makes these worth having is
 * that they turn the whole mesh into something scriptable without a plugin:
 * "select every vertex below the midpoint", "colour by height", "push the
 * surface along its normal by its own quality" are all one expression each.
 *
 * The variable sets are upstream's, name for name, because every published
 * recipe is written against them — see {@link VERTEX_VARIABLES} and
 * {@link FACE_VARIABLES}. So is one oddity worth flagging: `Per Vertex Color
 * Function` names its three channel parameters `x`, `y` and `z` rather than
 * `r`, `g` and `b`, while `Per Face Color Function` uses `r`, `g`, `b`.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichFloat,
	RichInt,
	RichString,
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
import { Clean } from "../../vcg/complex/clean.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import type { Pos } from "../../vcg/complex/pos.ts";
import { refineE } from "../../vcg/complex/refine.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { compileExpression } from "../../vcg/math/expression.ts";
import { alpha, blue, colorRamp, green, rgba } from "../../vcg/space/color4.ts";

export const FF = {
	FF_VERT_SELECTION: 0,
	FF_FACE_SELECTION: 1,
	FF_GEOM_FUNC: 2,
	FF_VERT_NORMAL: 3,
	FF_FACE_NORMAL: 4,
	FF_VERT_COLOR: 5,
	FF_FACE_COLOR: 6,
	FF_VERT_QUALITY: 7,
	FF_FACE_QUALITY: 8,
	FF_GRID: 9,
	FF_ISOSURFACE: 10,
	FF_REFINE: 11,
} as const;

/** The globals every expression sees, describing the mesh's bounding box. */
const GLOBAL_VARIABLES = [
	"xmin",
	"ymin",
	"zmin",
	"xmax",
	"ymax",
	"zmax",
	"bbdiag",
	"xdim",
	"ydim",
	"zdim",
	"xmid",
	"ymid",
	"zmid",
] as const;

/** Upstream's per-vertex variable set, in upstream's order. */
export const VERTEX_VARIABLES: readonly string[] = [
	"x",
	"y",
	"z",
	"nx",
	"ny",
	"nz",
	"r",
	"g",
	"b",
	"a",
	"q",
	"vi",
	"vtu",
	"vtv",
	"ti",
	"vsel",
	...GLOBAL_VARIABLES,
];

/** Upstream's per-face variable set, in upstream's order. */
export const FACE_VARIABLES: readonly string[] = [
	"x0",
	"y0",
	"z0",
	"x1",
	"y1",
	"z1",
	"x2",
	"y2",
	"z2",
	"nx0",
	"ny0",
	"nz0",
	"nx1",
	"ny1",
	"nz1",
	"nx2",
	"ny2",
	"nz2",
	"r0",
	"g0",
	"b0",
	"a0",
	"r1",
	"g1",
	"b1",
	"a1",
	"r2",
	"g2",
	"b2",
	"a2",
	"q0",
	"q1",
	"q2",
	"fr",
	"fg",
	"fb",
	"fa",
	"fnx",
	"fny",
	"fnz",
	"fq",
	"fi",
	"vi0",
	"vi1",
	"vi2",
	"wtu0",
	"wtv0",
	"wtu1",
	"wtv1",
	"wtu2",
	"wtv2",
	"ti",
	"vsel0",
	"vsel1",
	"vsel2",
	"fsel",
	...GLOBAL_VARIABLES,
];

/** The two endpoints of an edge, which is all `Refine User-Defined` sees. */
export const EDGE_VARIABLES: readonly string[] = [
	"x0",
	"y0",
	"z0",
	"x1",
	"y1",
	"z1",
	"nx0",
	"ny0",
	"nz0",
	"nx1",
	"ny1",
	"nz1",
	"r0",
	"g0",
	"b0",
	"r1",
	"g1",
	"b1",
	"q0",
	"q1",
];

/** The three coordinates an implicit surface is evaluated at. */
const ISO_VARIABLES: readonly string[] = ["x", "y", "z"];

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
	readonly requirements: number;
	readonly postCondition: number;
}

const GEOMETRY = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FF.FF_VERT_SELECTION]: {
		name: "Conditional Vertex Selection",
		pythonName: "compute_selection_by_condition_per_vertex",
		info:
			"Boolean function using muparser lib to perform vertex selection over current mesh.<br>" +
			"It's possible to use parenthesis <b>()</b>, and boolean operator <b>and</b>, <b>or</b>, " +
			"<b>&lt;</b>, <b>&gt;</b>, <b>=</b><br>" +
			"It's possible to use the following per-vertex variables in the expression:<br>" +
			"<b>x,y,z</b> (position), <b>nx,ny,nz</b> (normal), <b>r,g,b,a</b> (color), <b>q</b> " +
			"(quality), <b>rad</b>, <b>vi</b> (vertex index), <b>vtu,vtv</b> (texture coords), " +
			"<b>ti</b> (texture index), <b>vsel</b> (is the vertex selected? 1 yes, 0 no) and all " +
			"custom <i>vertex attributes</i> already defined by user.<br>",
		filterClass: FilterClass.Selection,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_VERTFLAGSELECT,
	},
	[FF.FF_FACE_SELECTION]: {
		name: "Conditional Face Selection",
		pythonName: "compute_selection_by_condition_per_face",
		info:
			"Boolean function using muparser lib to perform faces selection over current mesh.<br>" +
			"It's possible to use parenthesis <b>()</b>, and boolean operator <b>and</b>, <b>or</b>, " +
			"<b>&lt;</b>, <b>&gt;</b>, <b>=</b><br>",
		filterClass: FilterClass.Selection,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_FACEFLAGSELECT,
	},
	[FF.FF_GEOM_FUNC]: {
		name: "Per Vertex Geometric Function",
		pythonName: "compute_coord_by_function",
		info:
			"Geometric function using muparser lib to generate new Coord<br>You can change x,y,z for " +
			"every vertex according to the function specified.<br>",
		filterClass: FilterClass.Smoothing,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_NONE,
		postCondition: GEOMETRY,
	},
	[FF.FF_VERT_NORMAL]: {
		name: "Per Vertex Normal Function",
		pythonName: "compute_normal_by_function_per_vertex",
		info: "Normal function using muparser to generate new Normal for every vertex<br>",
		filterClass: FilterClass.Normal,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_VERTNORMAL,
	},
	[FF.FF_FACE_NORMAL]: {
		name: "Per Face Normal Function",
		pythonName: "compute_normal_by_function_per_face",
		info: "Normal function using muparser to generate new Normal for every face<br>",
		filterClass: FilterClass.Normal,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_FACENORMAL,
	},
	[FF.FF_VERT_COLOR]: {
		name: "Per Vertex Color Function",
		pythonName: "compute_color_by_function_per_vertex",
		info:
			"Color function using muparser lib to generate new RGBA color for every vertex<br>" +
			"Insert three function subexpression for R,G,B channels and one for Alpha channel<br>",
		filterClass: FilterClass.VertexColoring,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_VERTCOLOR,
		postCondition: MeshElement.MM_VERTCOLOR,
	},
	[FF.FF_FACE_COLOR]: {
		name: "Per Face Color Function",
		pythonName: "compute_color_by_function_per_face",
		info:
			"Color function using muparser lib to generate new RGBA color for every face<br>" +
			"Insert three function subexpression for R,G,B channels and one for Alpha channel<br>",
		filterClass: FilterClass.FaceColoring,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_FACECOLOR,
		postCondition: MeshElement.MM_FACECOLOR,
	},
	[FF.FF_VERT_QUALITY]: {
		name: "Per Vertex Quality Function",
		pythonName: "compute_scalar_by_function_per_vertex",
		info: "Quality function using muparser to generate new Quality for every vertex<br>",
		filterClass: FilterClass.VertexColoring | FilterClass.Quality,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_VERTQUALITY,
		postCondition: MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTCOLOR,
	},
	[FF.FF_FACE_QUALITY]: {
		name: "Per Face Quality Function",
		pythonName: "compute_scalar_by_function_per_face",
		info: "Quality function using muparser to generate new Quality for every face<br>",
		filterClass: FilterClass.FaceColoring | FilterClass.Quality,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_FACEQUALITY,
		postCondition: MeshElement.MM_FACEQUALITY | MeshElement.MM_FACECOLOR,
	},
	[FF.FF_GRID]: {
		name: "Grid Generator",
		pythonName: "create_grid",
		info:
			"Generate a new 2D Grid mesh with number of vertices on X and Y axis specified by user " +
			"with absolute length/height.<br>It's possible to center Grid on origin.",
		filterClass: FilterClass.MeshCreation,
		arity: FilterArity.NONE,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_NONE,
	},
	[FF.FF_ISOSURFACE]: {
		name: "Implicit Surface",
		pythonName: "create_implicit_surface",
		info: "Generate a new mesh that corresponds to the 0 valued isosurface defined by the scalar field generated by the given expression",
		filterClass: FilterClass.MeshCreation,
		arity: FilterArity.NONE,
		requirements: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_NONE,
	},
	[FF.FF_REFINE]: {
		name: "Refine User-Defined",
		pythonName: "meshing_refine_by_function",
		info:
			"Refine current mesh with user defined parameters.<br>Specify a boolean function which " +
			"takes variables describing an edge and returns true if the edge should be subdivided, " +
			"and a function describing where the new vertex goes.",
		filterClass: FilterClass.Remeshing,
		arity: FilterArity.SINGLE_MESH,
		requirements: MeshElement.MM_FACEFACETOPO,
		postCondition: GEOMETRY,
	},
};

export class FilterFunc extends FilterPlugin {
	pluginName(): string {
		return "FilterFunc";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FF);
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
	filterArity(id: ActionIDType): FilterArityValue {
		return this.spec(id).arity;
	}
	override getRequirements(id: ActionIDType): number {
		return this.spec(id).requirements;
	}
	override postCondition(id: ActionIDType): number {
		return this.spec(id).postCondition;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const onSelected = () =>
			list.add(
				new RichBool("onselected", false, {
					description: "only on selection",
					tooltip: "If checked, only affects selected vertices",
				}),
			);
		const expr = (name: string, defval: string, description: string, tooltip = "") =>
			list.add(new RichString(name, defval, { description, tooltip }));

		switch (id) {
			case FF.FF_VERT_SELECTION:
				expr(
					"condSelect",
					"(q < 0)",
					"boolean function",
					"type a boolean function that will be evaluated in order to select a subset of vertices",
				);
				break;

			case FF.FF_FACE_SELECTION:
				expr(
					"condSelect",
					"(fi == 0)",
					"boolean function",
					"type a boolean function that will be evaluated in order to select a subset of faces",
				);
				break;

			case FF.FF_GEOM_FUNC:
				expr("x", "x", "func x = ", "insert function to generate new coord for x");
				expr("y", "y", "func y = ", "insert function to generate new coord for y");
				expr("z", "sin(x+y)", "func z = ", "insert function to generate new coord for z");
				onSelected();
				break;

			case FF.FF_VERT_NORMAL:
				expr("x", "-nx", "func nx = ", "insert function to generate new x for the normal");
				expr("y", "-ny", "func ny = ", "insert function to generate new y for the normal");
				expr("z", "-nz", "func nz = ", "insert function to generate new z for the normal");
				onSelected();
				break;

			case FF.FF_FACE_NORMAL:
				expr("x", "-fnx", "func nx = ", "insert function to generate new x for the normal");
				expr("y", "-fny", "func ny = ", "insert function to generate new y for the normal");
				expr("z", "-fnz", "func nz = ", "insert function to generate new z for the normal");
				onSelected();
				break;

			case FF.FF_VERT_COLOR:
				// Upstream names these x/y/z rather than r/g/b; kept as-is so an
				// existing script keeps working.
				expr("x", "255", "func r = ", "function to generate Red component");
				expr("y", "255", "func g = ", "function to generate Green component");
				expr("z", "0", "func b = ", "function to generate Blue component");
				expr("a", "255", "func alpha = ", "function to generate Alpha component");
				onSelected();
				break;

			case FF.FF_FACE_COLOR:
				expr("r", "255", "func r = ", "function to generate Red component");
				expr("g", "0", "func g = ", "function to generate Green component");
				expr("b", "255", "func b = ", "function to generate Blue component");
				expr("a", "255", "func alpha = ", "function to generate Alpha component");
				onSelected();
				break;

			case FF.FF_VERT_QUALITY:
			case FF.FF_FACE_QUALITY:
				expr(
					"q",
					id === FF.FF_VERT_QUALITY ? "vi" : "x0+y0+z0",
					"func q = ",
					"function to generate new Quality for every vertex",
				);
				list.add(
					new RichBool("normalize", false, {
						description: "normalize",
						tooltip: "if checked normalize all quality values in range [0..1]",
					}),
				);
				list.add(
					new RichBool("map", false, {
						description: "map into color",
						tooltip: "if checked map quality generated values into per-vertex color",
					}),
				);
				onSelected();
				break;

			case FF.FF_GRID:
				list.add(
					new RichInt("numVertX", 10, {
						description: "num vertices on x",
						tooltip: "number of vertices on x. it must be positive",
					}),
				);
				list.add(
					new RichInt("numVertY", 10, {
						description: "num vertices on y",
						tooltip: "number of vertices on y. it must be positive",
					}),
				);
				list.add(
					new RichFloat("absScaleX", 0.3, {
						description: "x scale",
						tooltip: "absolute scale on x (float)",
					}),
				);
				list.add(
					new RichFloat("absScaleY", 0.3, {
						description: "y scale",
						tooltip: "absolute scale on y (float)",
					}),
				);
				list.add(
					new RichBool("center", false, {
						description: "centered on origin",
						tooltip: "center grid generated by filter on origin",
					}),
				);
				break;

			case FF.FF_ISOSURFACE:
				list.add(
					new RichFloat("voxelSize", 0.05, {
						description: "Size of the voxel",
						tooltip: "Size of the voxel",
					}),
				);
				for (const [name, defval] of [
					["minX", -1],
					["minY", -1],
					["minZ", -1],
					["maxX", 1],
					["maxY", 1],
					["maxZ", 1],
				] as const) {
					list.add(
						new RichFloat(name, defval, {
							description: name,
							tooltip: `${name} of the bounding box`,
						}),
					);
				}
				expr("expr", "x*x+y*y+z*z-0.5", "Function =", "the function that defines the surface");
				break;

			case FF.FF_REFINE:
				expr(
					"condSelect",
					"(q0 >= 0 && q1 >= 0)",
					"boolean function",
					"type a boolean function that will be evaluated on every edge",
				);
				expr("x", "(x0+x1)/2", "x =", "function to generate x coord of the new vertex");
				expr("y", "(y0+y1)/2", "y =", "function to generate y coord of the new vertex");
				expr("z", "(z0+z1)/2", "z =", "function to generate z coord of the new vertex");
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
		post.mask = this.spec(id).postCondition;

		if (id === FF.FF_GRID) return this.makeGrid(params, doc);
		if (id === FF.FF_ISOSURFACE) return this.makeIsoSurface(params, doc);

		const m = doc.mm();
		const cm = m.cm;
		const onlySelected = params.hasParameter("onselected") && params.getBool("onselected");

		switch (id) {
			case FF.FF_VERT_SELECTION: {
				const condition = compileExpression(params.getString("condSelect"), VERTEX_VARIABLES);
				let selected = 0;
				forEachVertex(cm, false, (v, values) => {
					if (condition.evaluate(values) !== 0) {
						cm.vertFlags[v] |= VertexFlag.SELECTED;
						selected++;
					} else cm.vertFlags[v] &= ~VertexFlag.SELECTED;
				});
				doc.Log.log(`Selected ${selected} vertices`);
				return { selected };
			}

			case FF.FF_FACE_SELECTION: {
				const condition = compileExpression(params.getString("condSelect"), FACE_VARIABLES);
				let selected = 0;
				forEachFace(cm, false, (f, values) => {
					if (condition.evaluate(values) !== 0) {
						cm.faceFlags[f] |= FaceFlag.SELECTED;
						selected++;
					} else cm.faceFlags[f] &= ~FaceFlag.SELECTED;
				});
				doc.Log.log(`Selected ${selected} faces`);
				return { selected };
			}

			case FF.FF_GEOM_FUNC: {
				const [fx, fy, fz] = ["x", "y", "z"].map((p) =>
					compileExpression(params.getString(p), VERTEX_VARIABLES),
				);
				// Every coordinate is computed from the *old* position, so the
				// three expressions cannot see each other's output.
				let n = 0;
				forEachVertex(cm, onlySelected, (v, values) => {
					cm.setVert(v, fx.evaluate(values), fy.evaluate(values), fz.evaluate(values));
					n++;
				});
				m.updateBoxAndNormals();
				return { vertex_number: n };
			}

			case FF.FF_VERT_NORMAL: {
				const [fx, fy, fz] = ["x", "y", "z"].map((p) =>
					compileExpression(params.getString(p), VERTEX_VARIABLES),
				);
				let n = 0;
				forEachVertex(cm, onlySelected, (v, values) => {
					cm.vertNormal[3 * v] = fx.evaluate(values);
					cm.vertNormal[3 * v + 1] = fy.evaluate(values);
					cm.vertNormal[3 * v + 2] = fz.evaluate(values);
					n++;
				});
				return { vertex_number: n };
			}

			case FF.FF_FACE_NORMAL: {
				const [fx, fy, fz] = ["x", "y", "z"].map((p) =>
					compileExpression(params.getString(p), FACE_VARIABLES),
				);
				let n = 0;
				forEachFace(cm, onlySelected, (f, values) => {
					cm.faceNormal[3 * f] = fx.evaluate(values);
					cm.faceNormal[3 * f + 1] = fy.evaluate(values);
					cm.faceNormal[3 * f + 2] = fz.evaluate(values);
					n++;
				});
				return { face_number: n };
			}

			case FF.FF_VERT_COLOR: {
				const [fr, fg, fb, fa] = ["x", "y", "z", "a"].map((p) =>
					compileExpression(params.getString(p), VERTEX_VARIABLES),
				);
				let n = 0;
				forEachVertex(cm, onlySelected, (v, values) => {
					cm.vertColor[v] = rgba(
						fr.evaluate(values),
						fg.evaluate(values),
						fb.evaluate(values),
						fa.evaluate(values),
					);
					n++;
				});
				return { vertex_number: n };
			}

			case FF.FF_FACE_COLOR: {
				const colors = requireFaceColor(cm);
				const [fr, fg, fb, fa] = ["r", "g", "b", "a"].map((p) =>
					compileExpression(params.getString(p), FACE_VARIABLES),
				);
				let n = 0;
				forEachFace(cm, onlySelected, (f, values) => {
					colors[f] = rgba(
						fr.evaluate(values),
						fg.evaluate(values),
						fb.evaluate(values),
						fa.evaluate(values),
					);
					n++;
				});
				return { face_number: n };
			}

			case FF.FF_VERT_QUALITY:
			case FF.FF_FACE_QUALITY: {
				const perFace = id === FF.FF_FACE_QUALITY;
				const fq = compileExpression(
					params.getString("q"),
					perFace ? FACE_VARIABLES : VERTEX_VARIABLES,
				);
				const written: number[] = [];
				if (perFace) {
					const quality = requireFaceQuality(cm);
					forEachFace(cm, onlySelected, (f, values) => {
						quality[f] = fq.evaluate(values);
						written.push(f);
					});
				} else {
					forEachVertex(cm, onlySelected, (v, values) => {
						cm.vertQuality[v] = fq.evaluate(values);
						written.push(v);
					});
				}

				const read = (i: number) =>
					perFace ? (cm.faceQuality as Float64Array)[i] : cm.vertQuality[i];
				const write = (i: number, value: number) => {
					if (perFace) (cm.faceQuality as Float64Array)[i] = value;
					else cm.vertQuality[i] = value;
				};
				let min = Number.POSITIVE_INFINITY;
				let max = Number.NEGATIVE_INFINITY;
				for (const i of written) {
					min = Math.min(min, read(i));
					max = Math.max(max, read(i));
				}
				if (params.getBool("normalize") && written.length > 0 && max > min) {
					for (const i of written) write(i, (read(i) - min) / (max - min));
					min = 0;
					max = 1;
				}
				if (params.getBool("map") && written.length > 0) {
					mapQualityToColor(cm, perFace, written, min, max);
				}
				doc.Log.log(`Quality now spans [${min}, ${max}] over ${written.length} elements`);
				return { min_value: min, max_value: max };
			}

			case FF.FF_REFINE: {
				const condition = compileExpression(params.getString("condSelect"), EDGE_VARIABLES);
				const [fx, fy, fz] = ["x", "y", "z"].map((p) =>
					compileExpression(params.getString(p), EDGE_VARIABLES),
				);
				const values = new Float64Array(EDGE_VARIABLES.length);
				const before = { vn: cm.vn, fn: cm.fn };
				UpdateTopology.faceFace(cm);
				refineE(
					cm,
					(mesh, pos) => {
						fillEdge(mesh, pos, values);
						return [fx.evaluate(values), fy.evaluate(values), fz.evaluate(values)];
					},
					(mesh, pos) => {
						fillEdge(mesh, pos, values);
						return condition.evaluate(values) !== 0;
					},
				);
				Allocator.compactEveryVector(cm);
				m.updateBoxAndNormals();
				doc.Log.log(`Refined ${before.vn}/${before.fn} into ${cm.vn} vertices and ${cm.fn} faces`);
				return { vertex_number: cm.vn, face_number: cm.fn };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	/** `Grid Generator`: a flat `numVertX` x `numVertY` lattice in the XY plane. */
	private makeGrid(params: RichParameterList, doc: MeshDocument): FilterOutput {
		const nx = params.getInt("numVertX");
		const ny = params.getInt("numVertY");
		if (nx <= 0 || ny <= 0) {
			throw new MLException(
				`The grid needs a positive vertex count on both axes, got ${nx} x ${ny}`,
			);
		}
		const sx = params.getFloat("absScaleX");
		const sy = params.getFloat("absScaleY");
		const centered = params.getBool("center");
		// Centring shifts by half the *whole* extent, which is (n-1) steps.
		const ox = centered ? (-(nx - 1) * sx) / 2 : 0;
		const oy = centered ? (-(ny - 1) * sy) / 2 : 0;

		const cm = new CMeshO();
		Allocator.addVertices(cm, nx * ny);
		for (let i = 0; i < nx; i++) {
			for (let j = 0; j < ny; j++) cm.setVert(i * ny + j, i * sx + ox, j * sy + oy, 0);
		}
		const quads = (nx - 1) * (ny - 1);
		if (quads > 0) {
			Allocator.addFaces(cm, quads * 2);
			let f = 0;
			for (let i = 0; i + 1 < nx; i++) {
				for (let j = 0; j + 1 < ny; j++) {
					const a = i * ny + j;
					const b = a + 1;
					const c = (i + 1) * ny + j;
					const d = c + 1;
					cm.setFace(f++, a, c, b);
					cm.setFace(f++, b, c, d);
				}
			}
		}
		const m = doc.addNewMesh("", "Grid", true, cm);
		m.updateBoxAndNormals();
		return { new_mesh_id: m.id(), vertex_number: cm.vn, face_number: cm.fn };
	}

	/**
	 * `Implicit Surface`: the zero level set of the user's scalar field.
	 *
	 * Marching tetrahedra rather than marching cubes — every cube is split on
	 * the same diagonal, so neighbouring cells agree on their shared faces and
	 * the surface comes out watertight without a 256-case table.
	 */
	private makeIsoSurface(params: RichParameterList, doc: MeshDocument): FilterOutput {
		const voxel = params.getFloat("voxelSize");
		if (!(voxel > 0)) throw new MLException(`The voxel size must be positive, got ${voxel}`);
		const low = [params.getFloat("minX"), params.getFloat("minY"), params.getFloat("minZ")];
		const high = [params.getFloat("maxX"), params.getFloat("maxY"), params.getFloat("maxZ")];
		for (let a = 0; a < 3; a++) {
			if (!(high[a] > low[a])) {
				throw new MLException(`The bounding box is empty along axis ${a}: ${low[a]} to ${high[a]}`);
			}
		}
		const counts = [0, 1, 2].map((a) => Math.max(2, Math.floor((high[a] - low[a]) / voxel) + 1));
		const total = counts[0] * counts[1] * counts[2];
		const LIMIT = 40_000_000;
		if (total > LIMIT) {
			throw new MLException(
				`That voxel size needs ${total} samples, over the ${LIMIT} limit; use a larger one`,
			);
		}

		const field = compileExpression(params.getString("expr"), ISO_VARIABLES);
		const at = new Float64Array(3);
		const values = new Float64Array(total);
		const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;
		const coord = (a: number, i: number) => low[a] + (i * (high[a] - low[a])) / (counts[a] - 1);
		for (let k = 0; k < counts[2]; k++) {
			at[2] = coord(2, k);
			for (let j = 0; j < counts[1]; j++) {
				at[1] = coord(1, j);
				for (let i = 0; i < counts[0]; i++) {
					at[0] = coord(0, i);
					values[index(i, j, k)] = field.evaluate(at);
				}
			}
		}

		const cm = marchingTetrahedra(values, counts, (a, i) => coord(a, i), index);
		const m = doc.addNewMesh("", "Implicit Surface", true, cm);
		m.updateBoxAndNormals();
		doc.Log.log(`Sampled ${total} points; the level set has ${cm.vn} vertices and ${cm.fn} faces`);
		return { new_mesh_id: m.id(), vertex_number: cm.vn, face_number: cm.fn };
	}
}

/** The 6 tetrahedra of a cube, all sharing the 0-7 diagonal so faces stay matched. */
const TETRAHEDRA: ReadonlyArray<readonly [number, number, number, number]> = [
	[0, 1, 3, 7],
	[0, 3, 2, 7],
	[0, 2, 6, 7],
	[0, 6, 4, 7],
	[0, 4, 5, 7],
	[0, 5, 1, 7],
];

/** Corner offsets of a cell, indexed the way {@link TETRAHEDRA} expects. */
const CORNERS: ReadonlyArray<readonly [number, number, number]> = [
	[0, 0, 0],
	[1, 0, 0],
	[0, 1, 0],
	[1, 1, 0],
	[0, 0, 1],
	[1, 0, 1],
	[0, 1, 1],
	[1, 1, 1],
];

function marchingTetrahedra(
	values: Float64Array,
	counts: readonly number[],
	coord: (axis: number, i: number) => number,
	index: (i: number, j: number, k: number) => number,
): CMeshO {
	const px: number[] = [];
	const py: number[] = [];
	const pz: number[] = [];
	const faces: number[] = [];
	const onEdge = new Map<number, number>();
	const total = values.length;

	const cut = (a: number, b: number, ai: readonly number[], bi: readonly number[]): number => {
		const key = a < b ? a * total + b : b * total + a;
		const seen = onEdge.get(key);
		if (seen !== undefined) return seen;
		const va = values[a];
		const vb = values[b];
		const gap = vb - va;
		const t = gap === 0 ? 0.5 : -va / gap;
		const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
		const p = [0, 1, 2].map((axis) => {
			const from = coord(axis, ai[axis]);
			return from + (coord(axis, bi[axis]) - from) * clamped;
		});
		const made = px.length;
		px.push(p[0]);
		py.push(p[1]);
		pz.push(p[2]);
		onEdge.set(key, made);
		return made;
	};

	const corner = new Int32Array(8);
	const cornerAt: Array<readonly number[]> = new Array(8);
	for (let k = 0; k + 1 < counts[2]; k++) {
		for (let j = 0; j + 1 < counts[1]; j++) {
			for (let i = 0; i + 1 < counts[0]; i++) {
				for (let c = 0; c < 8; c++) {
					const o = CORNERS[c];
					cornerAt[c] = [i + o[0], j + o[1], k + o[2]];
					corner[c] = index(i + o[0], j + o[1], k + o[2]);
				}
				for (const tet of TETRAHEDRA) {
					const inside: number[] = [];
					const outside: number[] = [];
					for (const c of tet) (values[corner[c]] < 0 ? inside : outside).push(c);
					if (inside.length === 0 || inside.length === 4) continue;
					const at = (c: number) => cornerAt[c];
					const edge = (a: number, b: number) => cut(corner[a], corner[b], at(a), at(b));
					if (inside.length === 1) {
						faces.push(
							edge(inside[0], outside[0]),
							edge(inside[0], outside[1]),
							edge(inside[0], outside[2]),
						);
					} else if (outside.length === 1) {
						faces.push(
							edge(outside[0], inside[0]),
							edge(outside[0], inside[1]),
							edge(outside[0], inside[2]),
						);
					} else {
						const quad = [
							edge(inside[0], outside[0]),
							edge(inside[0], outside[1]),
							edge(inside[1], outside[1]),
							edge(inside[1], outside[0]),
						];
						faces.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
					}
				}
			}
		}
	}

	const out = new CMeshO();
	if (px.length === 0) return out;
	const firstVert = Allocator.addVertices(out, px.length);
	for (let v = 0; v < px.length; v++) out.setVert(firstVert + v, px[v], py[v], pz[v]);
	const faceCount = faces.length / 3;
	if (faceCount > 0) {
		const firstFace = Allocator.addFaces(out, faceCount);
		for (let f = 0; f < faceCount; f++) {
			out.setFace(firstFace + f, faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]);
		}
	}
	Clean.removeDegenerateFace(out);
	Clean.removeUnreferencedVertex(out);
	Allocator.compactEveryVector(out);
	// Marching tetrahedra winds each triangle from whichever corners fell
	// inside, so the sheet is only consistent after a propagation pass.
	UpdateTopology.faceFace(out);
	Clean.orientCoherentlyMesh(out);
	Clean.flipNormalOutside(out);
	UpdateTopology.faceFace(out);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(out);
	return out;
}

/** Fills the bounding-box globals, which every element of a pass shares. */
function fillGlobals(cm: CMeshO, values: Float64Array, first: number): void {
	UpdateBounding.box(cm);
	const { min, max } = cm.bbox;
	const dim = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
	const all = [
		min[0],
		min[1],
		min[2],
		max[0],
		max[1],
		max[2],
		Math.hypot(dim[0], dim[1], dim[2]),
		dim[0],
		dim[1],
		dim[2],
		(min[0] + max[0]) / 2,
		(min[1] + max[1]) / 2,
		(min[2] + max[2]) / 2,
	];
	for (let i = 0; i < all.length; i++) values[first + i] = all[i];
}

/** Runs `fn` over every live vertex with its variable buffer filled. */
function forEachVertex(
	cm: CMeshO,
	onlySelected: boolean,
	fn: (v: number, values: Float64Array) => void,
): void {
	const values = new Float64Array(VERTEX_VARIABLES.length);
	fillGlobals(cm, values, VERTEX_VARIABLES.indexOf("xmin"));
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		if (onlySelected && !cm.isVertS(v)) continue;
		const c = cm.vertColor[v];
		values[0] = cm.vx(v);
		values[1] = cm.vy(v);
		values[2] = cm.vz(v);
		values[3] = cm.vertNormal[3 * v];
		values[4] = cm.vertNormal[3 * v + 1];
		values[5] = cm.vertNormal[3 * v + 2];
		values[6] = c & 0xff;
		values[7] = green(c);
		values[8] = blue(c);
		values[9] = alpha(c);
		values[10] = cm.vertQuality[v];
		values[11] = v;
		// Texture coordinates and index are not carried yet; upstream leaves
		// them at zero for a mesh without them, and so does this.
		values[12] = 0;
		values[13] = 0;
		values[14] = 0;
		values[15] = cm.isVertS(v) ? 1 : 0;
		fn(v, values);
	}
}

/** Runs `fn` over every live face with its variable buffer filled. */
function forEachFace(
	cm: CMeshO,
	onlySelected: boolean,
	fn: (f: number, values: Float64Array) => void,
): void {
	const values = new Float64Array(FACE_VARIABLES.length);
	fillGlobals(cm, values, FACE_VARIABLES.indexOf("xmin"));
	const scratch = new Float64Array(3);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		if (onlySelected && !cm.isFaceS(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(f, k);
			const c = cm.vertColor[v];
			values[3 * k] = cm.vx(v);
			values[3 * k + 1] = cm.vy(v);
			values[3 * k + 2] = cm.vz(v);
			values[9 + 3 * k] = cm.vertNormal[3 * v];
			values[10 + 3 * k] = cm.vertNormal[3 * v + 1];
			values[11 + 3 * k] = cm.vertNormal[3 * v + 2];
			values[18 + 4 * k] = c & 0xff;
			values[19 + 4 * k] = green(c);
			values[20 + 4 * k] = blue(c);
			values[21 + 4 * k] = alpha(c);
			values[30 + k] = cm.vertQuality[v];
			values[42 + k] = v;
			values[45 + 2 * k] = 0;
			values[46 + 2 * k] = 0;
			values[53 + k] = cm.isVertS(v) ? 1 : 0;
		}
		const fc = cm.faceColor === null ? 0xffffffff : cm.faceColor[f];
		values[33] = fc & 0xff;
		values[34] = green(fc);
		values[35] = blue(fc);
		values[36] = alpha(fc);
		UpdateNormal.faceNormalOf(cm, f, scratch);
		const len = Math.hypot(scratch[0], scratch[1], scratch[2]) || 1;
		values[37] = scratch[0] / len;
		values[38] = scratch[1] / len;
		values[39] = scratch[2] / len;
		values[40] = cm.faceQuality === null ? 0 : cm.faceQuality[f];
		values[41] = f;
		values[51] = 0;
		values[56] = cm.isFaceS(f) ? 1 : 0;
		fn(f, values);
	}
}

/** Fills the edge variable buffer `Refine User-Defined` evaluates against. */
function fillEdge(cm: CMeshO, pos: Pos, values: Float64Array): void {
	const ends = [pos.v, pos.vFlip];
	for (let k = 0; k < 2; k++) {
		const v = ends[k];
		const c = cm.vertColor[v];
		values[3 * k] = cm.vx(v);
		values[3 * k + 1] = cm.vy(v);
		values[3 * k + 2] = cm.vz(v);
		values[6 + 3 * k] = cm.vertNormal[3 * v];
		values[7 + 3 * k] = cm.vertNormal[3 * v + 1];
		values[8 + 3 * k] = cm.vertNormal[3 * v + 2];
		values[12 + 3 * k] = c & 0xff;
		values[13 + 3 * k] = green(c);
		values[14 + 3 * k] = blue(c);
		values[18 + k] = cm.vertQuality[v];
	}
}

/** Paints the quality range onto the colour channel, as the `map` flag asks. */
function mapQualityToColor(
	cm: CMeshO,
	perFace: boolean,
	written: readonly number[],
	min: number,
	max: number,
): void {
	// The same ramp the colorproc filters use, so a quality map looks the same
	// however it was produced.
	if (perFace) {
		const colors = cm.faceColor;
		if (colors === null) return;
		const quality = cm.faceQuality as Float64Array;
		for (const f of written) colors[f] = colorRamp(min, max, quality[f]);
	} else {
		for (const v of written) cm.vertColor[v] = colorRamp(min, max, cm.vertQuality[v]);
	}
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
