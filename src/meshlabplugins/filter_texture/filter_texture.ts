/**
 * `filter_texture` — parametrisation and texture baking.
 *
 * Three groups of filters live here. The UV conversions move coordinates
 * between the per-vertex and per-wedge channels, which differ in exactly one
 * way that matters: a wedge can carry a seam, a vertex cannot, so going from
 * wedges to vertices has to split the vertices that disagree. The
 * parametrisations lay out a mesh in texture space. The transfers bake an
 * attribute into a texture, or read one back out of it.
 *
 * `Parametrization: Voronoi Atlas` is not here: it needs geodesic Voronoi
 * partitioning and harmonic mapping per region, neither of which exists yet,
 * so it stays registered as unimplemented rather than approximated.
 */
import { readFileSync } from "node:fs";
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichEnum,
	RichFileOpen,
	RichFloat,
	RichInt,
	RichMesh,
	RichPercentage,
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { blue, green, red, rgba } from "../../vcg/space/color4.ts";
import { Image } from "../../vcg/space/image/image.ts";
import { isPng, readPng, writePng } from "../../vcg/space/image/png.ts";
import { type Hit, SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";
import { dummyTexture, faceUV, pullPushFill, rasteriseFace, setFaceUV } from "./rastering.ts";

export const FP = {
	FP_UV_WEDGE_TO_VERTEX: 0,
	FP_UV_VERTEX_TO_WEDGE: 1,
	FP_BASIC_TRIANGLE_MAPPING: 2,
	FP_PLANAR_MAPPING: 3,
	FP_SET_TEXTURE: 4,
	FP_COLOR_TO_TEXTURE: 5,
	FP_TRANSFER_TO_TEXTURE: 6,
	FP_TEX_TO_VCOLOR_TRANSFER: 7,
} as const;

/** What `Transfer: Vertex Attributes to Texture` can bake. */
const ATTRIBUTE = { COLOR: 0, NORMAL: 1, QUALITY: 2, TEXTURE: 3 } as const;

/** Unmapped texels, before the pull-push fill replaces them. */
const BACKGROUND = rgba(0, 0, 0, 0);

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly preConditions: number;
	readonly postCondition: number;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_UV_WEDGE_TO_VERTEX]: {
		name: "Convert PerWedge UV into PerVertex UV",
		pythonName: "compute_texcoord_transfer_wedge_to_vertex",
		info:
			"Converts per Wedge Texture Coordinates to per Vertex Texture Coordinates splitting " +
			"vertices with not coherent Wedge coordinates.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_WEDGTEXCOORD,
		postCondition: MeshElement.MM_VERTTEXCOORD,
	},
	[FP.FP_UV_VERTEX_TO_WEDGE]: {
		name: "Convert PerVertex UV into PerWedge UV",
		pythonName: "compute_texcoord_transfer_vertex_to_wedge",
		info:
			"Converts per Vertex Texture Coordinates to per Wedge Texture Coordinates. It does not " +
			"merge superfluous vertices...",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_VERTTEXCOORD,
		postCondition: MeshElement.MM_WEDGTEXCOORD,
	},
	[FP.FP_BASIC_TRIANGLE_MAPPING]: {
		name: "Parametrization: Trivial Per-Triangle",
		pythonName: "compute_texcoord_parametrization_triangle_trivial_per_wedge",
		info:
			"Builds a trivial triangle-by-triangle parametrization. Two methods are provided, the " +
			"first maps all triangles into equal sized triangles, while the second one adapt the size " +
			"of the triangles in texture space to their original size.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_FACENUMBER,
		postCondition: MeshElement.MM_WEDGTEXCOORD,
	},
	[FP.FP_PLANAR_MAPPING]: {
		name: "Parametrization: Flat Plane",
		pythonName: "compute_texcoord_parametrization_flat_plane_per_wedge",
		info: "Builds a trivial flat-plane parametrization.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_FACENUMBER,
		postCondition: MeshElement.MM_WEDGTEXCOORD,
	},
	[FP.FP_SET_TEXTURE]: {
		name: "Set Texture",
		pythonName: "set_texture_per_mesh",
		info:
			"Set a texture associated with current mesh parametrization. If the texture provided " +
			"exists, then it will be simply associated to the current mesh; else the filter will fail " +
			"with no further actions. If specified it can create and associate a dummy texture with a " +
			"specified grid or checkboard pattern.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_WEDGTEXCOORD,
		postCondition: MeshElement.MM_NONE,
	},
	[FP.FP_COLOR_TO_TEXTURE]: {
		name: "Transfer: Vertex Color to Texture",
		pythonName: "compute_texmap_from_color",
		info: "Fills the specified texture using per-vertex color data of the mesh.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_VERTCOLOR | MeshElement.MM_WEDGTEXCOORD,
		postCondition: MeshElement.MM_NONE,
	},
	[FP.FP_TRANSFER_TO_TEXTURE]: {
		name: "Transfer: Vertex Attributes to Texture (1 or 2 meshes)",
		pythonName: "transfer_attributes_to_texture_per_vertex",
		info:
			"Transfer texture color, vertex color or normal from one mesh the texture of another " +
			"mesh. This may be useful to restore detail lost in simplification, or resample a texture " +
			"in a different parametrization.",
		filterClass: FilterClass.Texture,
		preConditions: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_NONE,
	},
	[FP.FP_TEX_TO_VCOLOR_TRANSFER]: {
		name: "Transfer: Texture to Vertex Color (1 or 2 meshes)",
		pythonName: "transfer_texture_to_color_per_vertex",
		info: "Generates Vertex Color values picking color from a texture (same mesh or another mesh).",
		filterClass: FilterClass.VertexColoring | FilterClass.Texture,
		preConditions: MeshElement.MM_NONE,
		postCondition: MeshElement.MM_VERTCOLOR,
	},
};

export class FilterTexture extends FilterPlugin {
	pluginName(): string {
		return "FilterTexture";
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
		return this.spec(id).filterClass;
	}
	filterArity(id: ActionIDType): FilterArityValue {
		return id === FP.FP_TRANSFER_TO_TEXTURE || id === FP.FP_TEX_TO_VCOLOR_TRANSFER
			? FilterArity.FIXED
			: FilterArity.SINGLE_MESH;
	}
	override getPreConditions(id: ActionIDType): number {
		return this.spec(id).preConditions;
	}
	override postCondition(id: ActionIDType): number {
		return this.spec(id).postCondition;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const currentId = m?.id() ?? 0;
		const diagonal = boundingDiagonal(m);

		switch (id) {
			case FP.FP_PLANAR_MAPPING:
				list.add(
					new RichEnum("projectionPlane", 0, ["XY", "XZ", "YZ"], {
						description: "Projection plane",
						tooltip: "Choose the projection plane",
					}),
				);
				list.add(
					new RichBool("aspectRatio", false, {
						description: "Preserve Ratio",
						tooltip:
							"If checked the resulting parametrization will preserve the original aspect ratio " +
							"of the model otherwise it will fill up the whole 0..1 uv space",
					}),
				);
				list.add(
					new RichFloat("sideGutter", 0, {
						description: "Side Gutter",
						tooltip:
							"Leave an empty space around the parametrization area of the specified size (in " +
							"texture space); accepted range [0.0 - 0.5].",
					}),
				);
				break;

			case FP.FP_BASIC_TRIANGLE_MAPPING:
				list.add(
					new RichInt("sidedim", 0, {
						description: "Quads per line",
						tooltip:
							"Indicates how many triangles have to be put on each line (every quad contains " +
							"two triangles). Leave 0 for automatic calculation",
					}),
				);
				list.add(
					new RichInt("textdim", 1024, {
						description: "Texture Dimension (px)",
						tooltip: "Gives an indication on how big the texture is",
					}),
				);
				list.add(
					new RichInt("border", 2, {
						description: "Inter-Triangle border (px)",
						tooltip:
							"Specifies how many pixels to be left between triangles in parametrization domain",
					}),
				);
				list.add(
					new RichEnum("method", 1, ["Basic", "Space-optimizing"], {
						description: "Method",
						tooltip:
							"Choose space optimizing to map smaller faces into smaller triangles in " +
							"parametrizazion domain",
					}),
				);
				break;

			case FP.FP_SET_TEXTURE:
				list.add(
					new RichFileOpen("textName", "", ["png"], {
						description: "Texture file",
						tooltip: "Sets the given input image as unique texture of the mesh.",
					}),
				);
				list.add(
					new RichBool("use_dummy_texture", false, {
						description: "Use dummy texture",
						tooltip:
							"If checked, the filter will set a dummy texture instead of loading an image. The " +
							"'Texture File' parameter will be ignored.",
					}),
				);
				list.add(
					new RichInt("dummy_img_size", 512, {
						description: "Dummy size",
						tooltip: "Size in pixel of the square dummy texture.",
					}),
				);
				list.add(
					new RichInt("dummy_check_size", 8, {
						description: "Check size",
						tooltip: "Size in pixel of the checkerboard of the dummy texture.",
					}),
				);
				list.add(
					new RichEnum("dummy_type", 0, ["Checkboard", "Grid"], {
						description: "Dummy Texture Type",
						tooltip:
							"Choose what kind of dummy texture you want, a grid with lines or a checkboard",
					}),
				);
				break;

			case FP.FP_COLOR_TO_TEXTURE:
				list.add(
					new RichString("textName", "", {
						description: "Texture name",
						tooltip: "The name of the texture to be created",
					}),
				);
				addTextureSizeParameters(list);
				break;

			case FP.FP_TRANSFER_TO_TEXTURE:
				list.add(
					new RichMesh("sourceMesh", currentId, {
						description: "Source Mesh",
						tooltip: "The mesh that contains the source data that we want to transfer",
					}),
				);
				list.add(
					new RichMesh("targetMesh", currentId, {
						description: "Target Mesh",
						tooltip: "The mesh whose texture will be filled according to source mesh data",
					}),
				);
				list.add(
					new RichEnum(
						"AttributeEnum",
						0,
						["Vertex Color", "Vertex Normal", "Vertex Quality", "Texture Color"],
						{
							description: "Color Data Source",
							tooltip:
								"Choose what attribute has to be transferred onto the target texture. You can " +
								"choose between Per vertex attributes (color,normal,quality) or to transfer " +
								"color information from source mesh texture",
						},
					),
				);
				list.add(
					new RichPercentage("upperBound", diagonal / 50, 0, diagonal, {
						description: "Max Dist Search",
						tooltip:
							"Sample points for which we do not find anything within this distance are rejected and left unchanged",
					}),
				);
				list.add(
					new RichString("textName", "", {
						description: "Texture file",
						tooltip: "The texture file to be created",
					}),
				);
				addTextureSizeParameters(list);
				break;

			case FP.FP_TEX_TO_VCOLOR_TRANSFER:
				list.add(
					new RichMesh("sourceMesh", currentId, {
						description: "Source Mesh",
						tooltip: "The mesh with associated texture that we want to sample from",
					}),
				);
				list.add(
					new RichMesh("targetMesh", currentId, {
						description: "Target Mesh",
						tooltip: "The mesh whose vertex color will be filled according to source mesh texture",
					}),
				);
				list.add(
					new RichPercentage("upperBound", diagonal / 50, 0, diagonal, {
						description: "Max Dist Search",
						tooltip:
							"Sample points for which we do not find anything within this distance are rejected and left unchanged",
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
		cb: CallBackPos,
	): FilterOutput {
		switch (id) {
			case FP.FP_UV_VERTEX_TO_WEDGE: {
				const m = doc.mm();
				m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
				const cm = m.cm;
				const vt = cm.vertTexCoord;
				if (vt === null) throw new MLException("the mesh has no per-vertex texture coordinates");
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					setFaceUV(
						cm,
						f,
						[0, 1, 2].map((k) => {
							const v = cm.fv(f, k);
							return [vt[2 * v], vt[2 * v + 1]] as [number, number];
						}),
					);
				}
				doc.Log.log(`Copied texture coordinates onto ${cm.fn} faces' wedges`);
				return { faces: cm.fn };
			}

			case FP.FP_UV_WEDGE_TO_VERTEX: {
				const m = doc.mm();
				m.updateDataMask(MeshElement.MM_VERTTEXCOORD);
				const before = m.cm.vn;
				const split = splitVerticesBySeam(m.cm);
				if (split > 0) {
					// The topology is stale the moment a vertex is duplicated.
					m.clearDataMask(MeshElement.MM_FACEFACETOPO);
					m.clearDataMask(MeshElement.MM_VERTFACETOPO);
				}
				doc.Log.log(`Split ${split} vertices along texture seams: ${before} became ${m.cm.vn}`);
				return { split_vertices: split, vertices: m.cm.vn };
			}

			case FP.FP_PLANAR_MAPPING: {
				const m = doc.mm();
				m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
				const plane = params.getEnum("projectionPlane");
				const gutter = params.getFloat("sideGutter");
				if (gutter < 0 || gutter > 0.5) {
					throw new MLException(`the side gutter must be within 0..0.5, got ${gutter}`);
				}
				planarMapping(m.cm, plane, params.getBool("aspectRatio"), gutter);
				doc.Log.log(`Projected ${m.cm.fn} faces onto the ${["XY", "XZ", "YZ"][plane]} plane`);
				return { faces: m.cm.fn };
			}

			case FP.FP_BASIC_TRIANGLE_MAPPING: {
				const m = doc.mm();
				m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
				const textDim = params.getInt("textdim");
				const pxBorder = params.getInt("border");
				let sideDim = params.getInt("sidedim");
				if (textDim <= 0) throw new MLException("Texture Dimension has an incorrect value");
				if (pxBorder < 0) throw new MLException("Inter-Triangle border has an incorrect value");
				if (sideDim < 0) throw new MLException("Quads per line border has an incorrect value");

				const optimal = Math.ceil(Math.sqrt(m.cm.fn / 2));
				if (sideDim === 0) sideDim = optimal;
				else if (optimal > sideDim) {
					throw new MLException(
						"Quads per line aren't enough to obtain a correct parametrization. Try setting at " +
							`least ${optimal}`,
					);
				}
				const border = pxBorder / textDim;
				if (border * (1 + Math.SQRT2) + 2 / textDim > 1 / sideDim) {
					throw new MLException("Inter-Triangle border is too much");
				}
				const spaceOptimising = params.getEnum("method") === 1;
				trivialMapping(m.cm, sideDim, border, spaceOptimising, cb);

				const cathetus = (1 / sideDim - border - border / Math.SQRT2) * textDim;
				doc.Log.log(`Triangles' catheti are ${cathetus.toFixed(2)} px long`);
				return { faces: m.cm.fn, side_dim: sideDim };
			}

			case FP.FP_SET_TEXTURE: {
				const m = doc.mm();
				if (params.getBool("use_dummy_texture")) {
					const size = params.getInt("dummy_img_size");
					const check = params.getInt("dummy_check_size");
					if (size <= 0)
						throw new MLException(`the dummy texture size must be positive, got ${size}`);
					const image = dummyTexture(size, check, params.getEnum("dummy_type") === 1);
					const name = "dummy.png";
					setSingleTexture(m, name, writePng(image));
					post.mask = MeshElement.MM_NONE;
					doc.Log.log(`Attached a ${size}x${size} dummy texture`);
					return { texture: name };
				}

				const path = params.getString("textName");
				if (path === "") throw new MLException("Texture file not specified");
				const bytes = readFileSync(path);
				if (!isPng(bytes)) {
					throw new MLException(`only PNG textures can be read so far, and "${path}" is not one`);
				}
				const image = readPng(bytes);
				const name = path.split(/[\\/]/).pop() as string;
				setSingleTexture(m, name, bytes);
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Attached texture "${name}" (${image.width}x${image.height})`);
				return { texture: name, width: image.width, height: image.height };
			}

			case FP.FP_COLOR_TO_TEXTURE: {
				const m = doc.mm();
				const cm = m.cm;
				const image = newTexture(params);
				let painted = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					if (f % 256 === 0) cb((100 * f) / cm.faceSize, "Rasterising vertex colours");
					const colours = [0, 1, 2].map((k) => cm.vertColor[cm.fv(f, k)]);
					rasteriseFace(cm, f, image.width, image.height, ({ bary, x, y }) => {
						image.setPixel(x, y, mixColours(colours, bary));
						painted++;
					});
				}
				if (painted === 0) {
					throw new MLException(
						"nothing was rasterised: the mesh's texture coordinates cover no texel",
					);
				}
				if (params.getBool("pullpush")) pullPushFill(image, BACKGROUND);

				const name = textureName(params, "color");
				setSingleTexture(m, name, writePng(image));
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Painted ${painted} texels into "${name}"`);
				return { texture: name, texels: painted };
			}

			case FP.FP_TRANSFER_TO_TEXTURE: {
				const source = doc.requireMesh(params.getMeshId("sourceMesh"));
				const target = doc.requireMesh(params.getMeshId("targetMesh"));
				const attribute = params.getEnum("AttributeEnum");
				const upperBound = params.getAbsPerc("upperBound");
				const image = newTexture(params);
				const lookup = new SurfaceLookup(source.cm, upperBound);
				const sourceTexture = attribute === ATTRIBUTE.TEXTURE ? loadTexture(source) : null;
				if (attribute === ATTRIBUTE.TEXTURE && sourceTexture === null) {
					throw new MLException(`layer "${source.label()}" has no texture to transfer`);
				}
				if (attribute === ATTRIBUTE.NORMAL)
					UpdateNormal.perVertexNormalizedPerFaceNormalized(source.cm);

				const cm = target.cm;
				let painted = 0;
				let missed = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f)) continue;
					if (f % 128 === 0) cb((100 * f) / cm.faceSize, "Baking attributes");
					rasteriseFace(cm, f, image.width, image.height, ({ bary, x, y }) => {
						// The texel's position on the target surface, then the
						// nearest place on the source that carries the data.
						const p = interpolatePosition(cm, f, bary);
						const hit = lookup.closest(p[0], p[1], p[2]);
						if (hit === null) {
							missed++;
							return;
						}
						const colour = sampleAttribute(source.cm, hit, attribute, sourceTexture);
						if (colour === null) {
							missed++;
							return;
						}
						image.setPixel(x, y, colour);
						painted++;
					});
				}
				if (painted === 0) {
					throw new MLException(
						"nothing was transferred: no texel found the source mesh within the search distance",
					);
				}
				if (params.getBool("pullpush")) pullPushFill(image, BACKGROUND);

				const name = textureName(params, "transferred");
				setSingleTexture(target, name, writePng(image));
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(
					`Baked ${painted} texels into "${name}"` +
						(missed > 0 ? `; ${missed} found nothing within ${upperBound}` : ""),
				);
				return { texture: name, texels: painted, missed };
			}

			case FP.FP_TEX_TO_VCOLOR_TRANSFER: {
				const source = doc.requireMesh(params.getMeshId("sourceMesh"));
				const target = doc.requireMesh(params.getMeshId("targetMesh"));
				const upperBound = params.getAbsPerc("upperBound");
				const texture = loadTexture(source);
				if (texture === null) throw new MLException(`layer "${source.label()}" has no texture`);
				target.updateDataMask(MeshElement.MM_VERTCOLOR);

				const lookup = new SurfaceLookup(source.cm, upperBound);
				const cm = target.cm;
				let coloured = 0;
				let missed = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					const hit = lookup.closest(cm.vx(v), cm.vy(v), cm.vz(v));
					if (hit === null) {
						missed++;
						continue;
					}
					const uv = interpolateUV(source.cm, hit);
					if (uv === null) {
						missed++;
						continue;
					}
					cm.vertColor[v] = texture.sample(uv[0], uv[1]);
					coloured++;
				}
				if (coloured === 0) {
					throw new MLException("no vertex found the source surface within the search distance");
				}
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(
					`Coloured ${coloured} vertices from "${source.label()}"` +
						(missed > 0 ? `; ${missed} found nothing within ${upperBound}` : ""),
				);
				return { colored: coloured, missed };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}

/** The mesh's bounding diagonal, which the search-distance defaults scale by. */
function boundingDiagonal(m: MeshModel | undefined): number {
	if (m === undefined) return 1;
	UpdateBounding.box(m.cm);
	return m.cm.bbox.diagonal || 1;
}

function addTextureSizeParameters(list: RichParameterList): void {
	list.add(
		new RichInt("textW", 1024, { description: "Texture width (px)", tooltip: "The texture width" }),
	);
	list.add(
		new RichInt("textH", 1024, {
			description: "Texture height (px)",
			tooltip: "The texture height",
		}),
	);
	list.add(
		new RichBool("overwrite", false, {
			description: "Overwrite texture",
			tooltip:
				"if current mesh has a texture will be overwritten (with provided texture dimension)",
		}),
	);
	list.add(
		new RichBool("pullpush", true, {
			description: "Fill texture",
			tooltip:
				"if enabled the unmapped texture space is colored using a pull push filling algorithm, " +
				"if false is set to black",
		}),
	);
}

function newTexture(params: RichParameterList): Image {
	const w = params.getInt("textW");
	const h = params.getInt("textH");
	if (w <= 0 || h <= 0) throw new MLException(`the texture size must be positive, got ${w}x${h}`);
	return new Image(w, h, BACKGROUND);
}

function textureName(params: RichParameterList, fallback: string): string {
	const given = params.getString("textName").trim();
	if (given === "") return `${fallback}.png`;
	return given.toLowerCase().endsWith(".png") ? given : `${given}.png`;
}

/**
 * Replaces the mesh's texture list with a single entry.
 *
 * `cm.textures` is the list every face's wedge indexes into, so it and the
 * byte store have to move together — a name in one and not the other is a
 * texture reference that resolves to nothing.
 */
function setSingleTexture(m: MeshModel, name: string, bytes: Uint8Array): void {
	m.textures.clear();
	m.textures.set(name, bytes);
	m.cm.textures = [name];
}

function loadTexture(m: MeshModel): Image | null {
	const name = m.cm.textures[0];
	if (name === undefined) return null;
	const bytes = m.textures.get(name);
	if (bytes === undefined) return null;
	if (!isPng(bytes)) throw new MLException(`texture "${name}" is not a PNG, which is all we read`);
	return readPng(bytes);
}

function mixColours(colours: readonly number[], bary: readonly number[]): number {
	let r = 0;
	let g = 0;
	let b = 0;
	let a = 0;
	for (let k = 0; k < 3; k++) {
		r += red(colours[k]) * bary[k];
		g += green(colours[k]) * bary[k];
		b += blue(colours[k]) * bary[k];
		a += ((colours[k] >>> 24) & 0xff) * bary[k];
	}
	return rgba(clampByte(r), clampByte(g), clampByte(b), clampByte(a));
}

function clampByte(x: number): number {
	return Math.max(0, Math.min(255, Math.round(x)));
}

// ---- parametrisations -----------------------------------------------------

const PLANE_AXES: ReadonlyArray<readonly [number, number]> = [
	[0, 1], // XY
	[2, 0], // XZ
	[1, 2], // YZ
];

/**
 * Projects the mesh onto a coordinate plane and normalises into 0..1.
 *
 * With `aspectRatio` the two axes share the larger extent, so a long thin
 * model stays long and thin in texture space instead of being stretched to
 * fill the square.
 */
function planarMapping(cm: CMeshO, plane: number, aspectRatio: boolean, gutter: number): void {
	const [uAxis, vAxis] = PLANE_AXES[plane];
	const coord = (v: number, axis: number) => cm.vertCoord[3 * v + axis];

	let minU = Number.POSITIVE_INFINITY;
	let maxU = Number.NEGATIVE_INFINITY;
	let minV = Number.POSITIVE_INFINITY;
	let maxV = Number.NEGATIVE_INFINITY;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const uv: [number, number][] = [];
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(f, k);
			const u = coord(v, uAxis);
			const w = coord(v, vAxis);
			uv.push([u, w]);
			minU = Math.min(minU, u);
			maxU = Math.max(maxU, u);
			minV = Math.min(minV, w);
			maxV = Math.max(maxV, w);
		}
		setFaceUV(cm, f, uv);
	}
	if (!Number.isFinite(minU)) return;

	let wideU = maxU - minU;
	let wideV = maxV - minV;
	if (gutter > 0) {
		const delta = Math.min(wideU, wideV) * Math.min(gutter, 0.5);
		minU -= delta;
		minV -= delta;
		wideU += 2 * delta;
		wideV += 2 * delta;
	}
	if (aspectRatio) {
		wideU = Math.max(wideU, wideV);
		wideV = wideU;
	}
	// A flat model projected onto its own plane has no extent on one axis;
	// dividing would give NaN texture coordinates rather than a flat layout.
	if (wideU === 0) wideU = 1;
	if (wideV === 0) wideV = 1;

	const wt = cm.wedgeTexCoord as Float64Array;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			wt[6 * f + 2 * k] = (wt[6 * f + 2 * k] - minU) / wideU;
			wt[6 * f + 2 * k + 1] = (wt[6 * f + 2 * k + 1] - minV) / wideV;
		}
	}
}

/**
 * One triangle per half-square of a regular grid.
 *
 * Each cell of a `sideDim` x `sideDim` grid holds two right triangles, one
 * filling each half of the square. Every face gets an identical slot, which
 * wastes texture space on small faces and starves large ones — that is what
 * the space-optimising variant is for, sorting faces by area first so the
 * larger ones are laid out before the grid fills up.
 *
 * The border is inset asymmetrically: half of it on the two axis-aligned
 * sides, and half plus `border/√2` on the hypotenuse, because a diagonal edge
 * needs a wider margin to keep the same perpendicular gap.
 */
function trivialMapping(
	cm: CMeshO,
	sideDim: number,
	border: number,
	spaceOptimising: boolean,
	cb: CallBackPos,
): void {
	const faces: number[] = [];
	for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f)) faces.push(f);
	if (spaceOptimising) {
		// Largest first, so the faces that need the room get the slots that
		// are laid out before the grid runs out.
		faces.sort((a, b) => doubleArea(cm, b) - doubleArea(cm, a));
	}

	const bordersq2 = border / Math.SQRT2;
	const half = border / 2;
	const step = 1 / sideDim;

	faces.forEach((f, at) => {
		if (at % 256 === 0) cb((100 * at) / faces.length, "Generating parametrization...");
		const cell = Math.floor(at / 2);
		const row = Math.floor(cell / sideDim);
		const col = cell % sideDim;
		const odd = at % 2 === 0;

		// v runs downwards through the rows so the first face lands top-left.
		const left = col * step;
		const right = left + step;
		const top = 1 - row * step;
		const bottom = top - step;

		const longest = longestEdge(cm, f);
		const uv: [number, number][] = [
			[0, 0],
			[0, 0],
			[0, 0],
		];
		if (odd) {
			const bl: [number, number] = [left + half, bottom + half + bordersq2];
			const tr: [number, number] = [right - (half + bordersq2), top - half];
			uv[longest] = bl;
			uv[(longest + 1) % 3] = tr;
			uv[(longest + 2) % 3] = [bl[0], tr[1]];
		} else {
			const bl: [number, number] = [left + (half + bordersq2), bottom + half];
			const tr: [number, number] = [right - half, top - (half + bordersq2)];
			uv[longest] = tr;
			uv[(longest + 1) % 3] = bl;
			uv[(longest + 2) % 3] = [tr[0], bl[1]];
		}
		setFaceUV(cm, f, uv);
	});
}

/** The index of the edge opposite the longest one, as upstream orients by. */
function longestEdge(cm: CMeshO, f: number): number {
	let best = 0;
	let bestLength = -1;
	for (let k = 0; k < 3; k++) {
		const a = cm.fv(f, k);
		const b = cm.fv(f, (k + 1) % 3);
		const length =
			(cm.vx(a) - cm.vx(b)) ** 2 + (cm.vy(a) - cm.vy(b)) ** 2 + (cm.vz(a) - cm.vz(b)) ** 2;
		if (length > bestLength) {
			bestLength = length;
			best = k;
		}
	}
	return best;
}

function doubleArea(cm: CMeshO, f: number): number {
	const a = cm.fv(f, 0);
	const b = cm.fv(f, 1);
	const c = cm.fv(f, 2);
	const ux = cm.vx(b) - cm.vx(a);
	const uy = cm.vy(b) - cm.vy(a);
	const uz = cm.vz(b) - cm.vz(a);
	const vx = cm.vx(c) - cm.vx(a);
	const vy = cm.vy(c) - cm.vy(a);
	const vz = cm.vz(c) - cm.vz(a);
	return Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
}

/**
 * Splits every vertex whose incident wedges disagree about their UV.
 *
 * A per-vertex coordinate cannot represent a seam, so the seam has to become
 * an actual cut in the mesh. Each distinct (vertex, UV) pair gets its own
 * vertex; the geometry is unchanged and the surface stays where it was, but
 * it is no longer connected across the seam.
 */
function splitVerticesBySeam(cm: CMeshO): number {
	const wt = cm.wedgeTexCoord;
	if (wt === null) throw new MLException("the mesh has no per-wedge texture coordinates");
	if (cm.vertTexCoord === null) {
		throw new MLException("the per-vertex texture coordinates were not allocated");
	}
	// Re-read on every write: adding a vertex can reallocate the channel, and
	// a captured reference would keep writing into the old buffer.
	const setVertexUV = (v: number, u: number, w: number) => {
		const vt = cm.vertTexCoord as Float64Array;
		vt[2 * v] = u;
		vt[2 * v + 1] = w;
	};

	// The first UV seen at a vertex keeps the original; any other UV there
	// needs a copy of the vertex to hang on.
	const assigned = new Map<number, string>();
	const copies = new Map<string, number>();
	let split = 0;

	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(f, k);
			const u = wt[6 * f + 2 * k];
			const w = wt[6 * f + 2 * k + 1];
			const key = `${u},${w}`;
			const already = assigned.get(v);
			if (already === undefined) {
				assigned.set(v, key);
				setVertexUV(v, u, w);
				continue;
			}
			if (already === key) continue;

			const copyKey = `${v}|${key}`;
			let copy = copies.get(copyKey);
			if (copy === undefined) {
				copy = Allocator.addVertices(cm, 1);
				cm.setVert(copy, cm.vx(v), cm.vy(v), cm.vz(v));
				cm.vertQuality[copy] = cm.vertQuality[v];
				cm.vertColor[copy] = cm.vertColor[v];
				for (let a = 0; a < 3; a++) cm.vertNormal[3 * copy + a] = cm.vertNormal[3 * v + a];
				setVertexUV(copy, u, w);
				copies.set(copyKey, copy);
				split++;
			}
			cm.setFace(
				f,
				k === 0 ? copy : cm.fv(f, 0),
				k === 1 ? copy : cm.fv(f, 1),
				k === 2 ? copy : cm.fv(f, 2),
			);
		}
	}
	return split;
}

// ---- transfers ------------------------------------------------------------

function interpolatePosition(
	cm: CMeshO,
	f: number,
	bary: readonly number[],
): [number, number, number] {
	const out: [number, number, number] = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(f, k);
		out[0] += cm.vx(v) * bary[k];
		out[1] += cm.vy(v) * bary[k];
		out[2] += cm.vz(v) * bary[k];
	}
	return out;
}

function interpolateUV(cm: CMeshO, hit: Hit): [number, number] | null {
	if (cm.wedgeTexCoord === null) return null;
	const uv = faceUV(cm, hit.face);
	return [
		uv[0][0] * hit.bary[0] + uv[1][0] * hit.bary[1] + uv[2][0] * hit.bary[2],
		uv[0][1] * hit.bary[0] + uv[1][1] * hit.bary[1] + uv[2][1] * hit.bary[2],
	];
}

function sampleAttribute(
	cm: CMeshO,
	hit: Hit,
	attribute: number,
	texture: Image | null,
): number | null {
	switch (attribute) {
		case ATTRIBUTE.COLOR:
			return mixColours(
				[0, 1, 2].map((k) => cm.vertColor[cm.fv(hit.face, k)]),
				hit.bary,
			);
		case ATTRIBUTE.NORMAL: {
			// The usual normal-map encoding: -1..1 mapped onto 0..255. The
			// normals themselves are computed once by the caller, not here:
			// this runs per texel.
			const n = [0, 0, 0];
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(hit.face, k);
				for (let a = 0; a < 3; a++) n[a] += cm.vertNormal[3 * v + a] * hit.bary[k];
			}
			const length = Math.hypot(n[0], n[1], n[2]) || 1;
			return rgba(
				clampByte(((n[0] / length + 1) / 2) * 255),
				clampByte(((n[1] / length + 1) / 2) * 255),
				clampByte(((n[2] / length + 1) / 2) * 255),
			);
		}
		case ATTRIBUTE.QUALITY: {
			let q = 0;
			for (let k = 0; k < 3; k++) q += cm.vertQuality[cm.fv(hit.face, k)] * hit.bary[k];
			const grey = clampByte(q * 255);
			return rgba(grey, grey, grey);
		}
		default: {
			if (texture === null) return null;
			const uv = interpolateUV(cm, hit);
			return uv === null ? null : texture.sample(uv[0], uv[1]);
		}
	}
}
