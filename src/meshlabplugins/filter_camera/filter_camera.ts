/**
 * `filter_camera` — setting and moving the cameras a document carries.
 *
 * Both a mesh layer and a raster layer have a {@link Shot}: where it was seen
 * from. Five of these filters move those cameras about, one assigns one
 * outright, and two use a camera to say something about the mesh — which way
 * its normals should face, and how well each vertex is seen.
 *
 * The subtlety throughout is that {@link Shot}'s translation is the *view
 * point*, not the translation of a world-to-camera matrix. Moving a camera is
 * therefore a transform of that point plus a rotation of the axes, and never
 * a matrix multiply on the pose as stored. Getting that backwards produces
 * cameras that end up somewhere plausible and pointing the wrong way.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import type { RasterModel } from "../../common/ml_document/raster_model.ts";
import {
	RichBool,
	RichDirection,
	RichDynamicFloat,
	RichEnum,
	RichFloat,
	RichMatrix44,
	RichPosition,
	RichShot,
} from "../../common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import type { ShotValue } from "../../common/parameters/value.ts";
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
import { identity, type Matrix44 } from "../../vcg/math/matrix44.ts";
import type { Shot } from "../../vcg/math/shot.ts";
import { colorRamp } from "../../vcg/space/color4.ts";

export const FP = {
	FP_SET_MESH_CAMERA: 0,
	FP_SET_RASTER_CAMERA: 1,
	FP_QUALITY_FROM_CAMERA: 2,
	FP_CAMERA_ROTATE: 3,
	FP_CAMERA_SCALE: 4,
	FP_CAMERA_TRANSLATE: 5,
	FP_CAMERA_TRANSFORM: 6,
	FP_ORIENT_NORMALS_WITH_CAMERAS: 7,
} as const;

/** Which cameras a transform applies to. */
const CAMERA = { RASTER: 0, MESH: 1 } as const;

const SHOT_TYPES = ["Raster Camera", "Mesh Camera"];

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_SET_MESH_CAMERA]: {
		name: "Set Mesh Camera",
		pythonName: "set_camera_per_mesh",
		info: "This filter allows one to set a shot for the current mesh",
		filterClass: FilterClass.Layer | FilterClass.Camera,
	},
	[FP.FP_SET_RASTER_CAMERA]: {
		name: "Set Raster Camera",
		pythonName: "set_camera_per_raster",
		info: "This filter allows one to set a shot for the current raster",
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_QUALITY_FROM_CAMERA]: {
		name: "Vertex Quality from Camera",
		pythonName: "compute_scalar_from_camera_per_vertex",
		info: "Compute vertex quality using the camera definition, according to viewing angle or distance",
		filterClass: FilterClass.Quality | FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_CAMERA_ROTATE]: {
		name: "Transform: Rotate Camera or set of cameras",
		pythonName: "apply_cameras_rotation",
		info:
			"Rotate the camera, or all the cameras of the project. The selected raster is the " +
			"reference if viewpoint rotation is selected.",
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_CAMERA_SCALE]: {
		name: "Transform: Scale Camera or set of cameras",
		pythonName: "apply_cameras_scaling",
		info:
			"Scale the camera, or all the cameras of the project. The selected raster is the " +
			"reference if viewpoint scaling is selected.",
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_CAMERA_TRANSLATE]: {
		name: "Transform: Translate Camera or set of cameras",
		pythonName: "apply_cameras_translation",
		info: "Translate the camera, or all the cameras of the project.",
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_CAMERA_TRANSFORM]: {
		name: "Transform the camera extrinsics, or all the cameras of the project",
		pythonName: "apply_cameras_extrinsics_transformation",
		info: "Transform the camera extrinsics, or all the cameras of the project.",
		filterClass: FilterClass.RasterLayer | FilterClass.Camera,
	},
	[FP.FP_ORIENT_NORMALS_WITH_CAMERAS]: {
		name: "Re-Orient vertex normals using cameras",
		pythonName: "compute_normal_from_cameras_per_vertex",
		info:
			"Reorient vertex normals using cameras: a normal that points away from every camera that " +
			"can see its vertex is flipped.",
		filterClass: FilterClass.Normal | FilterClass.Camera,
	},
};

export class FilterCamera extends FilterPlugin {
	pluginName(): string {
		return "FilterCamera";
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
		const defaultShot: ShotValue = {
			rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
			translation: [0, 0, 0],
			focalMm: 1,
			pixelSizeMm: [1, 1],
			centerPx: [0, 0],
			viewportPx: [0, 0],
		};

		switch (id) {
			case FP.FP_SET_MESH_CAMERA:
			case FP.FP_SET_RASTER_CAMERA:
				list.add(
					new RichShot("Shot", defaultShot, {
						description: "New shot",
						tooltip: "The camera to assign to the current layer.",
					}),
				);
				break;

			case FP.FP_QUALITY_FROM_CAMERA:
				list.add(
					new RichBool("Depth", true, {
						description: "Depth",
						tooltip: "Use the distance from the camera as a factor.",
					}),
				);
				list.add(
					new RichBool("Facing", false, {
						description: "ViewAngle",
						tooltip: "Use the cosine of the viewing angle as a factor.",
					}),
				);
				list.add(
					new RichBool("Clip", false, {
						description: "Clipping",
						tooltip: "Set the quality to zero outside the camera's viewport.",
					}),
				);
				list.add(
					new RichBool("normalize", false, {
						description: "normalize",
						tooltip: "Rescale the resulting quality into 0..1.",
					}),
				);
				list.add(
					new RichBool("map", false, {
						description: "map into color",
						tooltip: "Also write the quality into the per-vertex colour.",
					}),
				);
				break;

			case FP.FP_CAMERA_ROTATE:
				addCameraChoice(list);
				list.add(
					new RichEnum("rotAxis", 0, ["X axis", "Y axis", "Z axis", "custom axis"], {
						description: "Rotation on",
						tooltip: "Choose the axis to rotate about.",
					}),
				);
				list.add(
					new RichEnum("rotCenter", 0, ["origin", "camera viewpoint", "custom point"], {
						description: "Rotation Center",
						tooltip: "Choose the point to rotate around.",
					}),
				);
				list.add(
					new RichDynamicFloat("angle", 0, -360, 360, {
						description: "Rotation Angle",
						tooltip: "The angle in degrees.",
					}),
				);
				list.add(new RichDirection("customAxis", [0, 0, 0], { description: "Custom axis" }));
				list.add(new RichPosition("customCenter", [0, 0, 0], { description: "Custom center" }));
				break;

			case FP.FP_CAMERA_SCALE:
				addCameraChoice(list);
				list.add(
					new RichEnum("scaleCenter", 0, ["origin", "camera viewpoint", "custom point"], {
						description: "Center of scaling",
					}),
				);
				list.add(new RichPosition("customCenter", [0, 0, 0], { description: "Custom center" }));
				list.add(new RichFloat("scale", 1, { description: "Scale factor" }));
				break;

			case FP.FP_CAMERA_TRANSLATE:
				addCameraChoice(list);
				list.add(new RichDynamicFloat("axisX", 0, -1000, 1000, { description: "X Axis" }));
				list.add(new RichDynamicFloat("axisY", 0, -1000, 1000, { description: "Y Axis" }));
				list.add(new RichDynamicFloat("axisZ", 0, -1000, 1000, { description: "Z Axis" }));
				list.add(
					new RichBool("centerFlag", false, {
						description: "translate camera to the origin",
						tooltip: "Move the camera to the origin instead of by the offset above.",
					}),
				);
				break;

			default:
				if (id === FP.FP_CAMERA_TRANSFORM) {
					list.add(
						new RichMatrix44("TransformMatrix", [...identity()], {
							description: "Transformation matrix",
						}),
					);
					addCameraChoice(list);
					list.add(
						new RichEnum(
							"behaviour",
							0,
							[
								"The matrix is the transformation to apply to the extrinsics",
								"The matrix represent the new extrinsics",
							],
							{ description: "Matrix semantic", tooltip: "What the matrix is used for." },
						),
					);
				}
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
		post.mask = MeshElement.MM_NONE;

		switch (id) {
			case FP.FP_SET_MESH_CAMERA: {
				const m = doc.mm();
				applyShotValue(m.shot, params.getShotf("Shot"));
				doc.Log.log(`Set the camera of "${m.label()}"`);
				return { mesh_id: m.id() };
			}

			case FP.FP_SET_RASTER_CAMERA: {
				const r = doc.rm();
				if (r === null) throw new MLException("the document has no current raster");
				applyShotValue(r.shot, params.getShotf("Shot"));
				doc.Log.log(`Set the camera of raster "${r.label()}"`);
				return { raster_id: r.id() };
			}

			case FP.FP_QUALITY_FROM_CAMERA:
				return qualityFromCamera(params, doc);

			case FP.FP_ORIENT_NORMALS_WITH_CAMERAS:
				return orientNormals(doc);

			default:
				return transformCameras(id, params, doc);
		}
	}
}

function addCameraChoice(list: RichParameterList): void {
	list.add(
		new RichEnum("camera", CAMERA.RASTER, SHOT_TYPES, {
			description: "Camera type",
			tooltip: "Whether the raster layers' cameras or the mesh layers' are moved.",
		}),
	);
	list.add(
		new RichBool("toallRaster", false, {
			description: "Apply to all active Raster layers",
			tooltip: "Taken into account only when 'Raster Camera' is selected.",
		}),
	);
	list.add(
		new RichBool("toall", false, {
			description: "Apply to all active Raster and visible Mesh layers",
			tooltip: "Apply the same transform to every layer, mesh and raster alike.",
		}),
	);
}

/** Copies a parameter's shot into a live one. */
function applyShotValue(shot: Shot, value: ShotValue): void {
	const m = identity();
	for (let r = 0; r < 3; r++) {
		for (let c = 0; c < 3; c++) m[4 * r + c] = value.rotation[3 * r + c];
	}
	shot.Extrinsics.SetRot(m);
	shot.Extrinsics.SetTra(value.translation);
	shot.Intrinsics.FocalMm = value.focalMm;
	shot.Intrinsics.PixelSizeMm = [...value.pixelSizeMm];
	shot.Intrinsics.CenterPx = [...value.centerPx];
	shot.Intrinsics.ViewportPx = [...value.viewportPx];
}

/** Every camera the caller asked to move. */
function chosenShots(params: RichParameterList, doc: MeshDocument): Shot[] {
	const all = params.getBool("toall");
	const allRasters = params.getBool("toallRaster");
	const meshCamera = params.getEnum("camera") === CAMERA.MESH;

	if (all) {
		return [
			...doc.visibleMeshes().map((m: MeshModel) => m.shot),
			...doc.visibleRasters().map((r: RasterModel) => r.shot),
		];
	}
	if (meshCamera) return [doc.mm().shot];
	if (allRasters) return doc.visibleRasters().map((r: RasterModel) => r.shot);

	const raster = doc.rm();
	if (raster === null) {
		throw new MLException(
			"the document has no current raster; choose 'Mesh Camera' to move the mesh's own camera",
		);
	}
	return [raster.shot];
}

function transformCameras(
	id: ActionIDType,
	params: RichParameterList,
	doc: MeshDocument,
): FilterOutput {
	const shots = chosenShots(params, doc);
	if (shots.length === 0) throw new MLException("there is no camera to transform");

	for (const shot of shots) {
		if (id === FP.FP_CAMERA_TRANSLATE) {
			const offset: [number, number, number] = params.getBool("centerFlag")
				? [-shot.Extrinsics.tra[0], -shot.Extrinsics.tra[1], -shot.Extrinsics.tra[2]]
				: [
						params.getDynamicFloat("axisX"),
						params.getDynamicFloat("axisY"),
						params.getDynamicFloat("axisZ"),
					];
			// Only the view point moves: a translation does not turn a camera.
			shot.SetViewPoint([
				shot.Extrinsics.tra[0] + offset[0],
				shot.Extrinsics.tra[1] + offset[1],
				shot.Extrinsics.tra[2] + offset[2],
			]);
			continue;
		}

		if (id === FP.FP_CAMERA_SCALE) {
			const scale = params.getFloat("scale");
			const centre = scaleCentre(params, shot);
			shot.SetViewPoint([
				centre[0] + (shot.Extrinsics.tra[0] - centre[0]) * scale,
				centre[1] + (shot.Extrinsics.tra[1] - centre[1]) * scale,
				centre[2] + (shot.Extrinsics.tra[2] - centre[2]) * scale,
			]);
			// The focal length scales with the scene so the field of view is
			// unchanged; scaling the position alone would zoom the camera in.
			shot.Intrinsics.FocalMm *= scale;
			continue;
		}

		const rotation =
			id === FP.FP_CAMERA_ROTATE
				? rotationMatrix(params)
				: matrixFrom(params.getMatrix44("TransformMatrix"));
		const centre = id === FP.FP_CAMERA_ROTATE ? rotationCentre(params, shot) : ([0, 0, 0] as const);
		const replace = id === FP.FP_CAMERA_TRANSFORM && params.getEnum("behaviour") === 1;

		if (replace) {
			// The matrix *is* the new pose: its rotation is the extrinsics and
			// its translation column the view point.
			const m = matrixFrom(params.getMatrix44("TransformMatrix"));
			shot.Extrinsics.SetRot(rotationPart(m));
			shot.SetViewPoint([m[3], m[7], m[11]]);
			continue;
		}

		// The view point is moved by the transform about the centre, and the
		// camera's axes are turned by its rotation.
		const p = shot.Extrinsics.tra;
		const relative = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
		const moved = applyMatrix(rotation, relative);
		shot.SetViewPoint([
			centre[0] + moved[0] + rotation[3],
			centre[1] + moved[1] + rotation[7],
			centre[2] + moved[2] + rotation[11],
		]);
		// `rot` maps world to camera, so a world rotation R composes on its
		// right: the new pose is `rot · Rᵀ`. Multiplying on the left would
		// turn the camera in the wrong direction and by the wrong amount.
		shot.Extrinsics.SetRot(multiply3(shot.Extrinsics.rot, transpose3(rotation)));
	}

	doc.Log.log(`Transformed ${shots.length} camera${shots.length === 1 ? "" : "s"}`);
	return { cameras: shots.length };
}

function scaleCentre(params: RichParameterList, shot: Shot): readonly number[] {
	switch (params.getEnum("scaleCenter")) {
		case 1:
			return shot.Extrinsics.tra;
		case 2:
			return params.getPoint3m("customCenter");
		default:
			return [0, 0, 0];
	}
}

function rotationCentre(params: RichParameterList, shot: Shot): readonly number[] {
	switch (params.getEnum("rotCenter")) {
		case 1:
			return shot.Extrinsics.tra;
		case 2:
			return params.getPoint3m("customCenter");
		default:
			return [0, 0, 0];
	}
}

function rotationMatrix(params: RichParameterList): Matrix44 {
	const which = params.getEnum("rotAxis");
	const custom = params.getPoint3m("customAxis");
	const axis =
		which === 0 ? [1, 0, 0] : which === 1 ? [0, 1, 0] : which === 2 ? [0, 0, 1] : [...custom];
	const length = Math.hypot(axis[0], axis[1], axis[2]);
	if (length === 0) throw new MLException("the custom rotation axis has zero length");
	const u = axis.map((c) => c / length);
	const angle = (params.getDynamicFloat("angle") * Math.PI) / 180;
	const c = Math.cos(angle);
	const s = Math.sin(angle);
	const t = 1 - c;

	const m = identity();
	m[0] = t * u[0] * u[0] + c;
	m[1] = t * u[0] * u[1] - s * u[2];
	m[2] = t * u[0] * u[2] + s * u[1];
	m[4] = t * u[0] * u[1] + s * u[2];
	m[5] = t * u[1] * u[1] + c;
	m[6] = t * u[1] * u[2] - s * u[0];
	m[8] = t * u[0] * u[2] - s * u[1];
	m[9] = t * u[1] * u[2] + s * u[0];
	m[10] = t * u[2] * u[2] + c;
	return m;
}

function matrixFrom(values: readonly number[]): Matrix44 {
	const m = identity();
	for (let i = 0; i < 16 && i < values.length; i++) m[i] = values[i];
	return m;
}

function rotationPart(m: Matrix44): Matrix44 {
	const out = identity();
	for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[4 * r + c] = m[4 * r + c];
	return out;
}

function applyMatrix(m: Matrix44, v: readonly number[]): number[] {
	return [
		m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
		m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
		m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
	];
}

function transpose3(m: Matrix44): Matrix44 {
	const out = identity();
	for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[4 * r + c] = m[4 * c + r];
	return out;
}

function multiply3(a: Matrix44, b: Matrix44): Matrix44 {
	const out = identity();
	for (let r = 0; r < 3; r++) {
		for (let c = 0; c < 3; c++) {
			let sum = 0;
			for (let k = 0; k < 3; k++) sum += a[4 * r + k] * b[4 * k + c];
			out[4 * r + c] = sum;
		}
	}
	return out;
}

/**
 * Quality from how well each vertex is seen.
 *
 * Depth and facing are multiplied when both are asked for, which is what
 * makes the result a rough "how useful is this camera here" — a vertex has to
 * be both close and facing the lens to score well.
 */
function qualityFromCamera(params: RichParameterList, doc: MeshDocument): FilterOutput {
	const m = doc.mm();
	const cm = m.cm;
	const raster = doc.rm();
	const shot = raster !== null ? raster.shot : m.shot;
	if (raster === null && isIdentityShot(shot)) {
		throw new MLException(
			'no camera to measure from: set one with "Set Mesh Camera", or add a raster layer',
		);
	}
	m.updateDataMask(MeshElement.MM_VERTQUALITY);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);

	const useDepth = params.getBool("Depth");
	const useFacing = params.getBool("Facing");
	const clip = params.getBool("Clip");
	const eye = shot.GetViewPoint();
	const forward = shot.GetViewDir();

	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const ray = [cm.vx(v) - eye[0], cm.vy(v) - eye[1], cm.vz(v) - eye[2]];
		const distance = Math.hypot(ray[0], ray[1], ray[2]);
		let value = 1;
		if (useDepth) value *= distance;
		if (useFacing && distance > 0) {
			// The cosine between the surface normal and the direction back to
			// the camera: one when facing it head on, zero edge on.
			const cos =
				-(
					ray[0] * cm.vertNormal[3 * v] +
					ray[1] * cm.vertNormal[3 * v + 1] +
					ray[2] * cm.vertNormal[3 * v + 2]
				) / distance;
			value *= Math.max(0, cos);
		}
		if (clip) {
			// Behind the camera counts as unseen whatever the other factors say.
			const along = ray[0] * forward[0] + ray[1] * forward[1] + ray[2] * forward[2];
			if (along <= 0) value = 0;
		}
		cm.vertQuality[v] = value;
		min = Math.min(min, value);
		max = Math.max(max, value);
	}

	if (params.getBool("normalize") && max > min) {
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) cm.vertQuality[v] = (cm.vertQuality[v] - min) / (max - min);
		}
	}
	if (params.getBool("map")) {
		m.updateDataMask(MeshElement.MM_VERTCOLOR);
		const lo = params.getBool("normalize") ? 0 : min;
		const hi = params.getBool("normalize") ? 1 : max;
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) {
				cm.vertColor[v] = colorRamp(lo, hi === lo ? lo + 1 : hi, cm.vertQuality[v]);
			}
		}
	}
	doc.Log.log(`Vertex quality from the camera: ${min} to ${max}`);
	return { min, max };
}

/**
 * Flips each normal that points away from every camera that can see it.
 *
 * Upstream needs the `correspondences` attribute a Bundler project carries,
 * which says which cameras actually saw which vertex. Without it the sensible
 * reading is "every visible camera", and a vertex facing away from all of
 * them has its normal reversed.
 */
function orientNormals(doc: MeshDocument): FilterOutput {
	const m = doc.mm();
	const cm = m.cm;
	const shots: Shot[] = doc.visibleRasters().map((r: RasterModel) => r.shot);
	if (shots.length === 0 && !isIdentityShot(m.shot)) shots.push(m.shot);
	if (shots.length === 0) {
		throw new MLException(
			"there is no camera to orient against: add a raster layer, or set the mesh's own camera",
		);
	}

	m.updateDataMask(MeshElement.MM_VERTNORMAL);
	UpdateBounding.box(cm);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);

	let flipped = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		let best = Number.NEGATIVE_INFINITY;
		for (const shot of shots) {
			const eye = shot.GetViewPoint();
			const toEye = [eye[0] - cm.vx(v), eye[1] - cm.vy(v), eye[2] - cm.vz(v)];
			const length = Math.hypot(toEye[0], toEye[1], toEye[2]);
			if (length === 0) continue;
			const cos =
				(toEye[0] * cm.vertNormal[3 * v] +
					toEye[1] * cm.vertNormal[3 * v + 1] +
					toEye[2] * cm.vertNormal[3 * v + 2]) /
				length;
			best = Math.max(best, cos);
		}
		if (best < 0) {
			for (let k = 0; k < 3; k++) cm.vertNormal[3 * v + k] = -cm.vertNormal[3 * v + k];
			flipped++;
		}
	}
	doc.Log.log(`Flipped ${flipped} normals to face the ${shots.length} camera(s)`);
	return { flipped_normals: flipped, cameras: shots.length };
}

/** True when nothing has set this camera. */
function isIdentityShot(shot: Shot): boolean {
	const t = shot.Extrinsics.tra;
	if (t[0] !== 0 || t[1] !== 0 || t[2] !== 0) return false;
	const r = shot.Extrinsics.rot;
	for (let i = 0; i < 3; i++) {
		for (let j = 0; j < 3; j++) {
			if (r[4 * i + j] !== (i === j ? 1 : 0)) return false;
		}
	}
	return true;
}
