/**
 * `filter_mls` — moving least squares surfaces over a point set.
 *
 * The maths lives in {@link mls_surface.ts}; this is the MeshLab surface over
 * it: projection, iso-surface extraction, curvature colouring, the radius
 * estimate the others depend on, and the small-component selection that ships
 * in the same plugin for historical reasons rather than mathematical ones.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
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
import { Clean } from "../../vcg/complex/clean.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { marchingTetrahedra } from "../../vcg/complex/create/marching.ts";
import { FaceFlag, VertexFlag } from "../../vcg/complex/flags.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { colorRamp } from "../../vcg/space/color4.ts";
import { Apss, estimateRadii, MLS_DEFAULTS, type MlsSurface, Rimls } from "./mls_surface.ts";

export const FP = {
	FP_APSS_PROJECTION: 0,
	FP_RIMLS_PROJECTION: 1,
	FP_APSS_MCUBE: 2,
	FP_RIMLS_MCUBE: 3,
	FP_APSS_COLORIZE: 4,
	FP_RIMLS_COLORIZE: 5,
	FP_RADIUS_FROM_DENSITY: 6,
	FP_SELECT_SMALL_COMPONENTS: 7,
} as const;

/** Upstream's curvature enum, shared by both colourise filters. */
const CT = { MEAN: 0, GAUSS: 1, K1: 2, K2: 3, APSS: 4 } as const;

const PROJ_INFO =
	"Project a mesh (or a point set) onto the MLS surface defined by itself or another point set.";
const MCUBE_INFO =
	"Extract the iso-surface (as a mesh) of a MLS surface defined by the current point set (or " +
	"mesh) using the marching cubes algorithm. The coarse extraction is followed by an accurate " +
	"projection step onto the MLS, and an extra zero removal procedure.";
const COLORIZE_INFO =
	"Colorize the vertices of a mesh or point set using the curvature of the underlying surface.";
const APSS_INFO =
	" This is the algebraic point set surfaces (APSS) variant which is based on the local fitting " +
	"of algebraic spheres. It requires points equipped with oriented normals. For all the details " +
	"about APSS see: Guennebaud and Gross, 'Algebraic Point Set Surfaces', Siggraph 2007, and " +
	"Guennebaud et al., 'Dynamic Sampling and Rendering of APSS', Eurographics 2008";
const RIMLS_INFO =
	" This is the Robust Implicit MLS (RIMLS) variant which is an extension of Implicit MLS " +
	"preserving sharp features using non linear regression. For more details see: Oztireli, " +
	"Guennebaud and Gross, 'Feature Preserving Point Set Surfaces based on Non-Linear Kernel " +
	"Regression' Eurographics 2009.";

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_APSS_PROJECTION]: {
		name: "MLS projection (APSS)",
		pythonName: "compute_mls_projection_apss",
		info: PROJ_INFO + APSS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.Smoothing,
	},
	[FP.FP_RIMLS_PROJECTION]: {
		name: "MLS projection (RIMLS)",
		pythonName: "compute_mls_projection_rimls",
		info: PROJ_INFO + RIMLS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.Smoothing,
	},
	[FP.FP_APSS_MCUBE]: {
		name: "Marching Cubes (APSS)",
		pythonName: "generate_marching_cubes_apss",
		info: MCUBE_INFO + APSS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.Remeshing,
	},
	[FP.FP_RIMLS_MCUBE]: {
		name: "Marching Cubes (RIMLS)",
		pythonName: "generate_marching_cubes_rimls",
		info: MCUBE_INFO + RIMLS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.Remeshing,
	},
	[FP.FP_APSS_COLORIZE]: {
		name: "Colorize curvature (APSS)",
		pythonName: "compute_curvature_and_color_apss_per_vertex",
		info: COLORIZE_INFO + APSS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.VertexColoring,
	},
	[FP.FP_RIMLS_COLORIZE]: {
		name: "Colorize curvature (RIMLS)",
		pythonName: "compute_curvature_and_color_rimls_per_vertex",
		info: COLORIZE_INFO + RIMLS_INFO,
		filterClass: FilterClass.PointSet | FilterClass.VertexColoring,
	},
	[FP.FP_RADIUS_FROM_DENSITY]: {
		name: "Estimate radius from density",
		pythonName: "compute_custom_radius_scalar_attribute_per_vertex",
		info:
			"Estimate the local point spacing (aka radius) around each vertex using a basic estimate " +
			"of the local density.",
		filterClass: FilterClass.PointSet,
	},
	[FP.FP_SELECT_SMALL_COMPONENTS]: {
		name: "Select small disconnected component",
		pythonName: "compute_selection_by_small_disconnected_components_per_face",
		info: "Select the small disconnected components of a mesh.",
		filterClass: FilterClass.Selection,
	},
};

export class FilterMLS extends FilterPlugin {
	pluginName(): string {
		return "FilterMLS";
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
		// Projection reads a control mesh and a proxy mesh, which may differ.
		return id === FP.FP_APSS_PROJECTION || id === FP.FP_RIMLS_PROJECTION
			? FilterArity.FIXED
			: FilterArity.SINGLE_MESH;
	}

	override getRequirements(id: ActionIDType): number {
		return id === FP.FP_SELECT_SMALL_COMPONENTS
			? MeshElement.MM_FACEFACETOPO
			: MeshElement.MM_VERTNORMAL;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const currentId = m?.id() ?? 0;

		switch (id) {
			case FP.FP_APSS_PROJECTION:
			case FP.FP_RIMLS_PROJECTION:
				list.add(
					new RichMesh("ControlMesh", currentId, {
						description: "Point set",
						tooltip: "The point set (or mesh) which defines the MLS surface.",
					}),
				);
				list.add(
					new RichMesh("ProxyMesh", currentId, {
						description: "Proxy Mesh",
						tooltip: "The mesh that will be projected/resampled onto the MLS surface.",
					}),
				);
				list.add(
					new RichBool("SelectionOnly", false, {
						description: "Selection only",
						tooltip: "If checked, only selected vertices will be projected.",
					}),
				);
				addMlsParameters(list);
				if (id === FP.FP_APSS_PROJECTION) addApssParameters(list);
				else addRimlsParameters(list);
				break;

			case FP.FP_APSS_MCUBE:
			case FP.FP_RIMLS_MCUBE:
				addMlsParameters(list);
				if (id === FP.FP_APSS_MCUBE) addApssParameters(list);
				else addRimlsParameters(list);
				list.add(
					new RichInt("Resolution", 200, {
						description: "Grid Resolution",
						tooltip:
							"The resolution of the grid on which we run the marching cubes. This marching " +
							"cube is memory friendly, so you can safely set large values up to 1000 or even more.",
					}),
				);
				break;

			case FP.FP_APSS_COLORIZE:
			case FP.FP_RIMLS_COLORIZE: {
				addMlsParameters(list);
				const apss = id === FP.FP_APSS_COLORIZE;
				if (apss) addApssParameters(list);
				else addRimlsParameters(list);
				list.add(
					new RichBool("SelectionOnly", false, {
						description: "Selection only",
						tooltip: "If checked, only selected vertices will be colorized.",
					}),
				);
				// APSS offers a fifth type: the curvature of the fitted sphere,
				// which is cheaper and steadier than differentiating the field.
				const types = ["Mean", "Gauss", "K1", "K2"];
				if (apss) types.push("Approx. Mean");
				list.add(
					new RichEnum("CurvatureType", CT.MEAN, types, {
						description: "Curvature type",
						tooltip: "The type of the curvature to plot.",
					}),
				);
				break;
			}

			case FP.FP_SELECT_SMALL_COMPONENTS:
				list.add(
					new RichFloat("NbFaceRatio", 0.1, {
						description: "Small component ratio",
						tooltip:
							"This ratio (between 0 and 1) defines the meaning of small as the threshold ratio " +
							"between the number of faces of the largest component and the other ones. A larger " +
							"value will select more components.",
					}),
				);
				list.add(
					new RichBool("NonClosedOnly", false, {
						description: "Select only non closed components",
						tooltip: "If checked, only components with a boundary are considered.",
					}),
				);
				break;

			case FP.FP_RADIUS_FROM_DENSITY:
				list.add(
					new RichInt("NbNeighbors", MLS_DEFAULTS.radiusNeighbours, {
						description: "Number of neighbors",
						tooltip:
							"Number of neighbors used to estimate the local density. Larger values lead to " +
							"smoother variations.",
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
			case FP.FP_RADIUS_FROM_DENSITY: {
				const m = doc.mm();
				const neighbours = params.getInt("NbNeighbors");
				if (neighbours < 1) {
					throw new MLException(`the neighbour count must be at least 1, got ${neighbours}`);
				}
				m.updateDataMask(MeshElement.MM_VERTQUALITY);
				const radii = estimateRadii(m.cm, neighbours);
				// The radius goes into the quality channel, which is where
				// MeshLab's own "radius" per-vertex attribute is surfaced to
				// every filter that has no idea what a radius is.
				let sum = 0;
				let count = 0;
				for (let v = 0; v < m.cm.vertSize; v++) {
					if (m.cm.isVertD(v)) continue;
					m.cm.vertQuality[v] = radii[v];
					sum += radii[v];
					count++;
				}
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(
					`Estimated radii from ${neighbours} neighbours; mean ${count === 0 ? 0 : sum / count}`,
				);
				return { mean_radius: count === 0 ? 0 : sum / count };
			}

			case FP.FP_SELECT_SMALL_COMPONENTS: {
				const m = doc.mm();
				const cm = m.cm;
				const ratio = params.getFloat("NbFaceRatio");
				const nonClosedOnly = params.getBool("NonClosedOnly");
				m.updateDataMask(MeshElement.MM_FACEFACETOPO);

				const components = smallComponents(cm, nonClosedOnly);
				let largest = 0;
				let covered = 0;
				for (const c of components) {
					covered += c.length;
					largest = Math.max(largest, c.length);
				}
				// Faces left out — the closed ones, when nonClosedOnly is set —
				// count towards the yardstick, so asking only about open shells
				// does not make every one of them look large.
				const remaining = cm.fn - covered;
				const threshold = Math.max(largest, remaining) * ratio;

				for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] &= ~FaceFlag.SELECTED;
				let selected = 0;
				for (const c of components) {
					if (c.length >= threshold) continue;
					for (const f of c) cm.faceFlags[f] |= FaceFlag.SELECTED;
					selected += c.length;
				}
				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Selected ${selected} faces in small components`);
				return { selected_faces: selected, components: components.length };
			}

			case FP.FP_APSS_PROJECTION:
			case FP.FP_RIMLS_PROJECTION: {
				const control = doc.requireMesh(params.getMeshId("ControlMesh"));
				const proxy = doc.requireMesh(params.getMeshId("ProxyMesh"));
				const selectionOnly = params.getBool("SelectionOnly");
				const surface = this.buildSurface(id === FP.FP_APSS_PROJECTION, control, params);

				const cm = proxy.cm;
				let moved = 0;
				let failed = 0;
				const live = cm.vn;
				let seen = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					seen++;
					if (seen % 512 === 0) cb((100 * seen) / live, "Projecting");
					if (selectionOnly && !cm.isVertS(v)) continue;
					const hit = surface.project(cm.vx(v), cm.vy(v), cm.vz(v));
					if (hit === null) {
						failed++;
						continue;
					}
					cm.setVert(v, hit.point[0], hit.point[1], hit.point[2]);
					cm.vertNormal[3 * v] = hit.normal[0];
					cm.vertNormal[3 * v + 1] = hit.normal[1];
					cm.vertNormal[3 * v + 2] = hit.normal[2];
					moved++;
				}
				proxy.updateBoxAndNormals();
				// The normals were set from the field, so recomputing them from
				// the (possibly absent) faces would throw that away.
				post.mask = MeshElement.MM_VERTCOORD;
				doc.Log.log(
					`Projected ${moved} vertices onto the MLS surface` +
						(failed > 0 ? `; ${failed} were out of range and left alone` : ""),
				);
				return { projected: moved, out_of_range: failed };
			}

			case FP.FP_APSS_MCUBE:
			case FP.FP_RIMLS_MCUBE: {
				const source = doc.mm();
				const resolution = params.getInt("Resolution");
				if (resolution < 2) {
					throw new MLException(`the grid resolution must be at least 2, got ${resolution}`);
				}
				const surface = this.buildSurface(id === FP.FP_APSS_MCUBE, source, params);
				const cm = marchIsoSurface(surface, source.cm, resolution, cb);
				const target = doc.addNewMesh("", `${source.label()} mc`, true, cm);
				target.updateBoxAndNormals();
				doc.Log.log(`Extracted an iso-surface with ${cm.vn} vertices and ${cm.fn} faces`);
				return { new_mesh_id: target.id(), vertex_number: cm.vn, face_number: cm.fn };
			}

			case FP.FP_APSS_COLORIZE:
			case FP.FP_RIMLS_COLORIZE: {
				const m = doc.mm();
				const cm = m.cm;
				const apss = id === FP.FP_APSS_COLORIZE;
				const type = params.getEnum("CurvatureType");
				const selectionOnly = params.getBool("SelectionOnly");
				const surface = this.buildSurface(apss, m, params);
				m.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);

				if (type === CT.GAUSS || type === CT.K1 || type === CT.K2) {
					// Those need the two principal curvatures, which come from
					// the shape operator rather than from the field's mean
					// curvature. Not derived here yet, and guessing would be
					// worse than saying so.
					throw new MLException(
						"Only the mean curvatures are available so far; Gauss, K1 and K2 need the " +
							"principal curvatures of the MLS shape operator, which is not implemented.",
					);
				}

				const values: number[] = [];
				const targets: number[] = [];
				let seen = 0;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					if (selectionOnly && !cm.isVertS(v)) continue;
					if (++seen % 256 === 0) cb((100 * seen) / cm.vn, "Computing curvature");
					const value =
						type === CT.APSS
							? (surface as Apss).approxMeanCurvature(cm.vx(v), cm.vy(v), cm.vz(v))
							: surface.meanCurvature(cm.vx(v), cm.vy(v), cm.vz(v));
					if (value === null) continue;
					targets.push(v);
					values.push(value);
				}
				if (targets.length === 0) throw new MLException("no vertex is within the MLS domain");

				// The ramp spans the 5th to 95th percentile: a handful of
				// outliers would otherwise flatten the whole mesh to one colour.
				const sorted = [...values].sort((a, b) => a - b);
				const low = sorted[Math.floor(0.05 * (sorted.length - 1))];
				const high = sorted[Math.ceil(0.95 * (sorted.length - 1))];
				targets.forEach((v, i) => {
					cm.vertQuality[v] = values[i];
					cm.vertColor[v] = colorRamp(low, high, values[i]);
				});

				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Colorized ${targets.length} vertices by curvature, ramp ${low} to ${high}`);
				return { min: sorted[0], max: sorted[sorted.length - 1], colorized: targets.length };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	private buildSurface(apss: boolean, m: MeshModel, params: RichParameterList): MlsSurface {
		if (m.cm.vn === 0) throw new MLException(`layer "${m.label()}" has no points`);
		UpdateBounding.box(m.cm);
		const surface = apss ? new Apss(m.cm) : new Rimls(m.cm);
		surface.filterScale = params.getFloat("FilterScale");
		surface.projectionAccuracy = params.getFloat("ProjectionAccuracy");
		surface.maxProjectionIters = params.getInt("MaxProjectionIters");
		if (surface instanceof Apss) {
			surface.sphericalParameter = params.getFloat("SphericalParameter");
		} else if (surface instanceof Rimls) {
			surface.sigmaN = params.getFloat("SigmaN");
			surface.maxRefittingIters = params.getInt("MaxRefittingIters");
		}
		return surface;
	}
}

function addMlsParameters(list: RichParameterList): void {
	list.add(
		new RichFloat("FilterScale", MLS_DEFAULTS.filterScale, {
			description: "MLS - Filter scale",
			tooltip:
				"Scale of the spatial low pass filter. It is relative to the radius (local point " +
				"spacing) of the vertices.",
		}),
	);
	list.add(
		new RichFloat("ProjectionAccuracy", MLS_DEFAULTS.projectionAccuracy, {
			description: "Projection - Accuracy (adv)",
			tooltip:
				"Threshold value used to stop the projections. This value is scaled by the mean point " +
				"spacing to get the actual threshold.",
		}),
	);
	list.add(
		new RichInt("MaxProjectionIters", MLS_DEFAULTS.maxProjectionIters, {
			description: "Projection - Max iterations (adv)",
			tooltip: "Max number of iterations for the projection.",
		}),
	);
}

function addApssParameters(list: RichParameterList): void {
	list.add(
		new RichFloat("SphericalParameter", MLS_DEFAULTS.sphericalParameter, {
			description: "MLS - Spherical parameter",
			tooltip:
				"Control the curvature of the fitted spheres: 0 is equivalent to a pure plane fit, 1 to " +
				"a pure spherical fit, values between 0 and 1 gives intermediate results, while other " +
				"real values might give interesting results, but take care with extreme settings!",
		}),
	);
}

function addRimlsParameters(list: RichParameterList): void {
	list.add(
		new RichFloat("SigmaN", MLS_DEFAULTS.sigmaN, {
			description: "MLS - Sharpness",
			tooltip:
				"Width of the filter used by the normal refitting weight. This weight function is a " +
				"Gaussian on the distance between two unit vectors: the current gradient and the input " +
				"normal. Therefore, typical value range between 0.5 (sharp) to 2 (smooth).",
		}),
	);
	list.add(
		new RichInt("MaxRefittingIters", MLS_DEFAULTS.maxRefittingIters, {
			description: "MLS - Max fitting iterations",
			tooltip: "Max number of fitting iterations. (0 or 1 is equivalent to the standard IMLS)",
		}),
	);
}

/**
 * The connected components of the face graph, optionally restricted to the
 * ones with a boundary.
 *
 * A component is skipped entirely when `nonClosedOnly` is set and its seed has
 * no border edge — which is the same as saying the whole component is closed,
 * since the flood below reaches every face of it.
 */
function smallComponents(cm: CMeshO, nonClosedOnly: boolean): number[][] {
	const visited = new Uint8Array(cm.faceSize);
	const out: number[][] = [];
	for (let seed = 0; seed < cm.faceSize; seed++) {
		if (cm.isFaceD(seed) || visited[seed] === 1) continue;
		if (nonClosedOnly && !hasBorder(cm, seed)) continue;

		const component: number[] = [];
		const stack = [seed];
		visited[seed] = 1;
		while (stack.length > 0) {
			const f = stack.pop() as number;
			component.push(f);
			for (let e = 0; e < 3; e++) {
				if (cm.isBorderFF(f, e)) continue;
				const g = cm.ffp(f, e);
				if (g < 0 || cm.isFaceD(g) || visited[g] === 1) continue;
				visited[g] = 1;
				stack.push(g);
			}
		}
		out.push(component);
	}
	return out;
}

function hasBorder(cm: CMeshO, f: number): boolean {
	for (let e = 0; e < 3; e++) if (cm.isBorderFF(f, e)) return true;
	return false;
}

/**
 * Marching tetrahedra over the MLS field, followed by a projection pass.
 *
 * The grid interpolation puts each vertex on the straight line between two
 * samples of the field, which is only first-order accurate. Projecting each
 * one onto the surface afterwards recovers the detail the grid misses, and
 * costs one MLS evaluation per vertex rather than per grid node.
 *
 * Nodes outside every point's support have no field value at all. They are
 * filled with a positive number larger than the cell size, which reads as
 * "outside" and keeps the iso-surface from being dragged into empty space.
 */
function marchIsoSurface(
	surface: MlsSurface,
	source: CMeshO,
	resolution: number,
	cb: CallBackPos,
): CMeshO {
	UpdateBounding.box(source);
	const box = source.bbox;
	const pad = surface.averageSpacing * 2;
	const min = [box.min[0] - pad, box.min[1] - pad, box.min[2] - pad];
	const max = [box.max[0] + pad, box.max[1] + pad, box.max[2] + pad];
	const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
	if (span <= 0) throw new MLException("the point set has no extent to march over");

	const step = span / resolution;
	const counts = [0, 1, 2].map((a) => Math.max(2, Math.ceil((max[a] - min[a]) / step) + 1));
	const total = counts[0] * counts[1] * counts[2];
	const values = new Float64Array(total);
	const outside = step * 4;

	const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;
	const coord = (axis: number, i: number) => min[axis] + i * step;

	for (let k = 0; k < counts[2]; k++) {
		cb((100 * k) / counts[2], "Sampling the MLS field");
		for (let j = 0; j < counts[1]; j++) {
			for (let i = 0; i < counts[0]; i++) {
				const value = surface.potential(coord(0, i), coord(1, j), coord(2, k));
				values[index(i, j, k)] = value === null ? outside : value;
			}
		}
	}

	const cm = marchingTetrahedra(values, counts, coord, index);

	// The grid puts each vertex on a straight line between two samples, which
	// is first-order accurate at best. Projecting recovers the rest.
	const stranded = new Uint8Array(cm.vertSize);
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const hit = surface.project(cm.vx(v), cm.vy(v), cm.vz(v));
		if (hit === null) {
			stranded[v] = 1;
			continue;
		}
		cm.setVert(v, hit.point[0], hit.point[1], hit.point[2]);
	}

	// Upstream's "extra zero removal": a vertex the projection could not place
	// is sitting wherever the grid left it, which is not on the surface. Its
	// faces are guesses, so they go rather than being reported as geometry.
	let dropped = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		if (stranded[cm.fv(f, 0)] === 0 && stranded[cm.fv(f, 1)] === 0 && stranded[cm.fv(f, 2)] === 0) {
			continue;
		}
		Allocator.deleteFace(cm, f);
		dropped++;
	}
	if (dropped > 0) Clean.removeUnreferencedVertex(cm);

	UpdateTopology.clearFaceFace(cm);
	for (let v = 0; v < cm.vertSize; v++) cm.vertFlags[v] &= ~VertexFlag.SELECTED;
	return cm;
}
