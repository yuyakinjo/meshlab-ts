/**
 * `filter_color_projection` — painting a mesh with the photographs registered
 * against it.
 *
 * Every raster layer is an image plus the {@link Shot} it was taken with. To
 * colour a vertex, project it through that camera into the image and read the
 * pixel. Doing only that gives a mesh painted through its own back: a camera
 * sees the front of an object and the projection happily paints the far side
 * too. The depth test is what stops it, and it is the reason this needs a ray
 * caster rather than only a projection.
 *
 * Several cameras usually see the same vertex, so the contributions are
 * blended by weight. The weights say how much each camera should be trusted
 * there: straight on beats grazing, near beats far, the middle of the frame
 * beats the edge.
 */

import { readFileSync } from "node:fs";
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { RasterModel } from "../../common/ml_document/raster_model.ts";
import {
	RichBool,
	RichColor,
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
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import type { Shot } from "../../vcg/math/shot.ts";
import { blue, green, red, rgba } from "../../vcg/space/color4.ts";
import { Image } from "../../vcg/space/image/image.ts";
import { isPng, readPng, writePng } from "../../vcg/space/image/png.ts";
import { BVH } from "../../vcg/space/index/bvh.ts";
import { pullPushFill, rasteriseFace } from "../filter_texture/rastering.ts";

export const FP = {
	FP_SINGLEIMAGEPROJ: 0,
	FP_MULTIIMAGETRIVIALPROJ: 1,
	FP_MULTIIMAGETRIVIALPROJTEXTURE: 2,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_SINGLEIMAGEPROJ]: {
		name: "Project current raster color to current mesh",
		pythonName: "compute_color_from_current_raster_projection",
		info: "Color information from the current raster is perspective-projected on the current mesh",
		filterClass: FilterClass.VertexColoring | FilterClass.Camera,
	},
	[FP.FP_MULTIIMAGETRIVIALPROJ]: {
		name: "Project active rasters color to current mesh",
		pythonName: "compute_color_from_active_rasters_projection",
		info:
			"Color information from all the active rasters is perspective-projected on the current " +
			"mesh using basic weighting",
		filterClass: FilterClass.VertexColoring | FilterClass.Camera,
	},
	[FP.FP_MULTIIMAGETRIVIALPROJTEXTURE]: {
		name: "Project active rasters color to current mesh, filling the texture",
		pythonName: "compute_color_and_texture_from_active_rasters_projection",
		info:
			"Color information from all the active rasters is perspective-projected on the current " +
			"mesh, filling the texture, using basic weighting",
		filterClass: FilterClass.Texture | FilterClass.Camera,
	},
};

export class FilterColorProjection extends FilterPlugin {
	pluginName(): string {
		return "FilterColorProjection";
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
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override initParameterList(id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichBool("usedepth", true, {
				description: "use depth for projection",
				tooltip:
					"Reject a vertex the camera cannot actually see. Without this the projection paints " +
					"the far side of the object as well as the near one.",
			}),
		);
		list.add(
			new RichFloat("deptheta", 0.5, {
				description: "depth threshold",
				tooltip: "How much nearer something has to be, in scene units, to count as occluding.",
			}),
		);
		list.add(
			new RichBool("onselection", false, {
				description: "Only on selection",
				tooltip: "Colour only the selected vertices.",
			}),
		);
		if (id === FP.FP_SINGLEIMAGEPROJ) {
			list.add(
				new RichColor("blankColor", rgba(0, 0, 0, 255), {
					description: "Blank Color",
					tooltip:
						"The colour given to a vertex no camera sees. Fully transparent leaves the old " +
						"colour in place.",
				}),
			);
			return list;
		}
		list.add(
			new RichBool("useangle", true, {
				description: "use angle weight",
				tooltip: "Weight each camera's contribution by how squarely it sees the surface.",
			}),
		);
		list.add(
			new RichBool("usedistance", true, {
				description: "use distance weight",
				tooltip: "Weight each camera's contribution by how close it is.",
			}),
		);
		list.add(
			new RichBool("useborders", true, {
				description: "use image borders weight",
				tooltip:
					"Fade a contribution out towards the edge of its frame, where lens distortion and " +
					"registration error are worst.",
			}),
		);
		list.add(
			new RichBool("usealpha", false, {
				description: "use image alpha weight",
				tooltip: "Multiply by the image's own alpha channel, so a masked photo can be excluded.",
			}),
		);
		if (id === FP.FP_MULTIIMAGETRIVIALPROJTEXTURE) {
			list.add(
				new RichString("textName", "", {
					description: "Texture file",
					tooltip: "The texture to be created.",
				}),
			);
			list.add(new RichInt("textW", 1024, { description: "Texture width (px)" }));
			list.add(new RichInt("textH", 1024, { description: "Texture height (px)" }));
			list.add(
				new RichBool("pullpush", true, {
					description: "Fill texture",
					tooltip: "Spread the painted colour into the unmapped texels.",
				}),
			);
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
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces to project onto");

		const rasters =
			id === FP.FP_SINGLEIMAGEPROJ ? [requireCurrentRaster(doc)] : [...doc.visibleRasters()];
		if (rasters.length === 0) throw new MLException("there is no active raster to project from");

		const cameras = rasters.map((r) => ({ shot: r.shot, image: imageOf(r), label: r.label() }));
		UpdateBounding.box(cm);
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
		const bvh = params.getBool("usedepth") ? new BVH(cm) : null;
		const epsilon = params.getFloat("deptheta");
		const onSelection = params.getBool("onselection");

		const weights = {
			angle: id !== FP.FP_SINGLEIMAGEPROJ && params.getBool("useangle"),
			distance: id !== FP.FP_SINGLEIMAGEPROJ && params.getBool("usedistance"),
			borders: id !== FP.FP_SINGLEIMAGEPROJ && params.getBool("useborders"),
			alpha: id !== FP.FP_SINGLEIMAGEPROJ && params.getBool("usealpha"),
		};

		if (id === FP.FP_MULTIIMAGETRIVIALPROJTEXTURE) {
			return projectToTexture(params, doc, cameras, bvh, epsilon, weights, cb);
		}

		m.updateDataMask(MeshElement.MM_VERTCOLOR);
		const blank = id === FP.FP_SINGLEIMAGEPROJ ? params.getColor("blankColor") : rgba(0, 0, 0, 255);
		let painted = 0;
		let missed = 0;

		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			if (onSelection && !cm.isVertS(v)) continue;
			if (v % 512 === 0) cb((100 * v) / cm.vertSize, "Projecting colour");

			const p = [cm.vx(v), cm.vy(v), cm.vz(v)];
			const n = [cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]];
			const blended = blend(cameras, p, n, bvh, epsilon, weights);
			if (blended === null) {
				missed++;
				// A fully transparent blank colour means "leave it alone",
				// which is the only way to top up a partial colouring.
				if (((blank >>> 24) & 0xff) !== 0) cm.vertColor[v] = blank;
				continue;
			}
			cm.vertColor[v] = blended;
			painted++;
		}
		if (painted === 0) {
			throw new MLException(
				"no vertex was seen by any camera; check that the rasters are registered against this mesh",
			);
		}
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`Projected ${cameras.length} raster${cameras.length === 1 ? "" : "s"} onto ${painted} ` +
				`vertices; ${missed} were not seen`,
		);
		return { colored: painted, not_seen: missed, rasters: cameras.length };
	}
}

interface Camera {
	readonly shot: Shot;
	readonly image: Image;
	readonly label: string;
}

interface Weights {
	readonly angle: boolean;
	readonly distance: boolean;
	readonly borders: boolean;
	readonly alpha: boolean;
}

function requireCurrentRaster(doc: MeshDocument): RasterModel {
	const r = doc.rm();
	if (r === null) throw new MLException("the document has no current raster");
	return r;
}

/**
 * The raster's photograph, decoded.
 *
 * A `RasterPlane` stores only a path — nothing in this library decodes pixels
 * until something needs them, which is here. Only PNG is read, and a missing
 * or unreadable file is named rather than skipped: a projection silently
 * missing one of its cameras looks like a registration problem.
 */
function imageOf(raster: RasterModel): Image {
	const plane = raster.currentPlane;
	if (plane === null) {
		throw new MLException(`raster "${raster.label()}" has no image plane to project`);
	}
	let bytes: Uint8Array;
	try {
		bytes = readFileSync(plane.fullPathFileName);
	} catch {
		throw new MLException(`cannot read "${plane.fullPathFileName}" for raster "${raster.label()}"`);
	}
	if (!isPng(bytes)) {
		throw new MLException(`"${plane.fullPathFileName}" is not a PNG, which is all we decode`);
	}
	return readPng(bytes);
}

/** Where a world point lands in a camera's image, and how far in front it is. */
function project(
	shot: Shot,
	image: Image,
	p: readonly number[],
): { x: number; y: number; depth: number } | null {
	const eye = shot.GetViewPoint();
	const rot = shot.Extrinsics.rot;
	const d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
	// Into the camera's own frame: the rows of `rot` are its axes.
	const cam = [
		rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2],
		rot[4] * d[0] + rot[5] * d[1] + rot[6] * d[2],
		rot[8] * d[0] + rot[9] * d[1] + rot[10] * d[2],
	];
	if (cam[2] <= 0) return null; // behind the camera

	const focalX = shot.Intrinsics.FocalMm / (shot.Intrinsics.PixelSizeMm[0] || 1);
	const focalY = shot.Intrinsics.FocalMm / (shot.Intrinsics.PixelSizeMm[1] || 1);
	const x = shot.Intrinsics.CenterPx[0] + (cam[0] / cam[2]) * focalX;
	// The camera's y points up and the image's rows go down, so the vertical
	// coordinate is measured from the bottom.
	const y = shot.Intrinsics.CenterPx[1] - (cam[1] / cam[2]) * focalY;
	if (x < 0 || y < 0 || x > image.width - 1 || y > image.height - 1) return null;
	return { x, y, depth: cam[2] };
}

/** All the cameras that can see a point, blended by weight. */
function blend(
	cameras: readonly Camera[],
	p: readonly number[],
	normal: readonly number[],
	bvh: BVH | null,
	epsilon: number,
	weights: Weights,
): number | null {
	let r = 0;
	let g = 0;
	let b = 0;
	let total = 0;

	for (const camera of cameras) {
		const at = project(camera.shot, camera.image, p);
		if (at === null) continue;
		const eye = camera.shot.GetViewPoint();
		const toEye = [eye[0] - p[0], eye[1] - p[1], eye[2] - p[2]];
		const distance = Math.hypot(toEye[0], toEye[1], toEye[2]);
		if (distance === 0) continue;
		const cos = (toEye[0] * normal[0] + toEye[1] * normal[1] + toEye[2] * normal[2]) / distance;
		// A surface turned away from the camera is not seen by it, whatever
		// the depth test says about the geometry in between.
		if (cos <= 0) continue;

		if (bvh !== null) {
			// Anything between the point and the eye means this camera sees
			// something else here. The offset keeps the point's own faces from
			// counting as their own occluder.
			const direction = toEye.map((c) => c / distance);
			if (bvh.occluded(p, direction, epsilon, distance - epsilon)) continue;
		}

		let weight = 1;
		if (weights.angle) weight *= cos;
		if (weights.distance) weight *= 1 / (distance * distance);
		if (weights.borders) {
			// Fades linearly to zero at the frame's edge, where lens
			// distortion and registration error are worst.
			const u = at.x / (camera.image.width - 1);
			const v = at.y / (camera.image.height - 1);
			weight *= Math.min(u, 1 - u, v, 1 - v) * 2;
		}
		if (weight <= 0) continue;

		const colour = camera.image.sample(
			at.x / (camera.image.width - 1),
			1 - at.y / (camera.image.height - 1),
		);
		if (weights.alpha) weight *= ((colour >>> 24) & 0xff) / 255;
		if (weight <= 0) continue;

		r += red(colour) * weight;
		g += green(colour) * weight;
		b += blue(colour) * weight;
		total += weight;
	}
	if (total === 0) return null;
	return rgba(
		Math.max(0, Math.min(255, Math.round(r / total))),
		Math.max(0, Math.min(255, Math.round(g / total))),
		Math.max(0, Math.min(255, Math.round(b / total))),
	);
}

/** The same projection, baked into a texture instead of into the vertices. */
function projectToTexture(
	params: RichParameterList,
	doc: MeshDocument,
	cameras: readonly Camera[],
	bvh: BVH | null,
	epsilon: number,
	weights: Weights,
	cb: CallBackPos,
): FilterOutput {
	const m = doc.mm();
	const cm = m.cm;
	if (cm.wedgeTexCoord === null) {
		throw new MLException(
			'the mesh has no texture coordinates; run a parametrisation such as "Parametrization: ' +
				'Trivial Per-Triangle" first',
		);
	}
	const width = params.getInt("textW");
	const height = params.getInt("textH");
	if (width <= 0 || height <= 0) {
		throw new MLException(`the texture size must be positive, got ${width}x${height}`);
	}
	const background = rgba(0, 0, 0, 0);
	const texture = new Image(width, height, background);

	let painted = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		if (f % 128 === 0) cb((100 * f) / cm.faceSize, "Baking projected colour");
		const corners = [0, 1, 2].map((k) => {
			const v = cm.fv(f, k);
			return {
				p: [cm.vx(v), cm.vy(v), cm.vz(v)],
				n: [cm.vertNormal[3 * v], cm.vertNormal[3 * v + 1], cm.vertNormal[3 * v + 2]],
			};
		});
		rasteriseFace(cm, f, width, height, ({ bary, x, y }) => {
			const p = [0, 0, 0];
			const n = [0, 0, 0];
			for (let k = 0; k < 3; k++) {
				for (let a = 0; a < 3; a++) {
					p[a] += corners[k].p[a] * bary[k];
					n[a] += corners[k].n[a] * bary[k];
				}
			}
			const colour = blend(cameras, p, n, bvh, epsilon, weights);
			if (colour === null) return;
			texture.setPixel(x, y, colour);
			painted++;
		});
	}
	if (painted === 0) {
		throw new MLException("no texel was seen by any camera");
	}
	if (params.getBool("pullpush")) pullPushFill(texture, background);

	const given = params.getString("textName").trim();
	const name = given === "" ? "projected.png" : given.endsWith(".png") ? given : `${given}.png`;
	m.textures.clear();
	m.textures.set(name, writePng(texture));
	m.cm.textures = [name];
	doc.Log.log(`Painted ${painted} texels into "${name}" from ${cameras.length} rasters`);
	return { texture: name, texels: painted, rasters: cameras.length };
}
