/**
 * `filter_measure` — reporting what a mesh is.
 *
 * These filters change nothing; their whole product is the output map. The
 * keys are upstream's exactly, because a caller reads them by name.
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
import { MLException } from "../../common/utilities/ml_exception.ts";
import {
	countBitLargePolygons,
	countBitPolygons,
	countBitQuads,
	countBitTris,
	hasConsistentPerFaceFauxFlag,
	isBitTriQuadOnly,
	isFFAdjacencyConsistent,
} from "../../vcg/complex/bit_quad.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { fauxBit } from "../../vcg/complex/flags.ts";
import { Inertia } from "../../vcg/complex/inertia.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { Distribution, Histogram } from "../../vcg/math/histogram.ts";

export const FP = {
	COMPUTE_TOPOLOGICAL_MEASURES: 0,
	COMPUTE_GEOMETRIC_MEASURES: 1,
	PER_VERTEX_QUALITY_STAT: 2,
	PER_FACE_QUALITY_STAT: 3,
	COMPUTE_TOPOLOGICAL_MEASURES_QUAD_MESHES: 4,
	COMPUTE_AREA_PERIMETER_SELECTION: 5,
	PER_VERTEX_QUALITY_HISTOGRAM: 6,
	PER_FACE_QUALITY_HISTOGRAM: 7,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.COMPUTE_TOPOLOGICAL_MEASURES]: {
		name: "Compute Topological Measures",
		pythonName: "get_topological_measures",
		info: "Compute a set of topological measures over a mesh.",
	},
	[FP.COMPUTE_GEOMETRIC_MEASURES]: {
		name: "Compute Geometric Measures",
		pythonName: "get_geometric_measures",
		info:
			"Compute a set of geometric measures of a mesh/pointcloud. Bounding box extents and " +
			"diagonal, principal axis, thin shell barycenter (mesh only), vertex barycenter and " +
			"quality-weighted barycenter (pointcloud only), surface area (mesh only), volume (closed " +
			"mesh) and Inertia tensor Matrix (closed mesh).",
	},
	[FP.PER_VERTEX_QUALITY_STAT]: {
		name: "Per Vertex Quality Stat",
		pythonName: "get_scalar_statistics_per_vertex",
		info: "Compute some aggregate statistics over the per vertex quality, like Min, Max, Average.",
	},
	[FP.PER_FACE_QUALITY_STAT]: {
		name: "Per Face Quality Stat",
		pythonName: "get_scalar_statistics_per_face",
		info: "Compute some aggregate statistics over the per face quality, like Min, Max, Average.",
	},
	[FP.COMPUTE_TOPOLOGICAL_MEASURES_QUAD_MESHES]: {
		name: "Compute Topological Measures for Quad Meshes",
		pythonName: "get_topological_measures_from_quad_mesh",
		info: "Compute a set of topological measures over a quad mesh.",
	},
	[FP.COMPUTE_AREA_PERIMETER_SELECTION]: {
		name: "Compute Area/Perimeter of selection",
		pythonName: "get_area_and_perimeter_of_selection",
		info: "Compute area and perimeter of the FACE selection.",
	},
	[FP.PER_VERTEX_QUALITY_HISTOGRAM]: {
		name: "Per Vertex Quality Histogram",
		pythonName: "get_scalar_histogram_per_vertex",
		info:
			"Compute an histogram of the values of the per-vertex quality. It can be useful to " +
			"evaluate the distribution of the quality value over the surface. It can be discrete " +
			"(e.g. based on vertex count or area weighted).",
	},
	[FP.PER_FACE_QUALITY_HISTOGRAM]: {
		name: "Per Face Quality Histogram",
		pythonName: "get_scalar_histogram_per_face",
		info: "Compute an histogram of the values of the per-face quality.",
	},
};

export class FilterMeasure extends FilterPlugin {
	pluginName(): string {
		return "FilterMeasure";
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
		return FilterClass.Measure;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override getPreConditions(id: ActionIDType): number {
		switch (id) {
			case FP.PER_VERTEX_QUALITY_STAT:
			case FP.PER_VERTEX_QUALITY_HISTOGRAM:
				return MeshElement.MM_VERTQUALITY;
			case FP.PER_FACE_QUALITY_STAT:
			case FP.PER_FACE_QUALITY_HISTOGRAM:
				return MeshElement.MM_FACEQUALITY;
			default:
				return MeshElement.MM_NONE;
		}
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		if (id !== FP.PER_VERTEX_QUALITY_HISTOGRAM && id !== FP.PER_FACE_QUALITY_HISTOGRAM) {
			return list;
		}
		// The default range is whatever the mesh actually spans, so the
		// histogram is useful without touching a single parameter.
		const perFace = id === FP.PER_FACE_QUALITY_HISTOGRAM;
		const { min, max } = qualityRange(m?.cm, perFace);
		const what = perFace ? "faces" : "vertices";
		list.add(
			new RichFloat("HistMin", min, {
				description: "Hist Min",
				tooltip: "The lower bound of the histogram; anything below lands in an underflow bin.",
			}),
		);
		list.add(
			new RichFloat("HistMax", max, {
				description: "Hist Max",
				tooltip: "The upper bound of the histogram; anything above lands in an overflow bin.",
			}),
		);
		list.add(
			new RichBool("areaWeighted", false, {
				description: "Area Weighted",
				tooltip:
					`If false, the histogram will report the number of ${what} with quality values ` +
					"falling in each bin of the histogram. If true each bin of the histogram will " +
					"report the approximate area of the mesh with that range of values." +
					(perFace
						? ""
						: " Area is computed by assigning to each vertex one third of the area of all " +
							"the incident triangles."),
			}),
		);
		list.add(
			new RichInt("binNum", 20, {
				description: "Bin number",
				tooltip:
					"The number of bins of the histogram. E.g. the number of intervals in which the " +
					"min..max range is subdivided into.",
			}),
		);
		return list;
	}

	/** Measuring reads; it never writes. */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
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
		post.mask = MeshElement.MM_NONE;

		switch (id) {
			case FP.COMPUTE_TOPOLOGICAL_MEASURES: {
				UpdateTopology.faceFace(cm);
				const counts = Clean.countEdgeNum(cm);
				const unref = Clean.countUnreferencedVertex(cm);
				const components = Clean.countConnectedComponents(cm);
				const nonManifEdges = counts.nonManifold;
				const nonManifVerts = Clean.countNonManifoldVertexFF(cm);
				const twoManifold = nonManifEdges === 0 && nonManifVerts === 0;

				const out: FilterOutput = {
					vertices_number: cm.vn,
					edges_number: counts.total,
					faces_number: cm.fn,
					unreferenced_vertices: unref,
					boundary_edges: counts.boundary,
					connected_components_number: components,
					is_mesh_two_manifold: twoManifold,
					non_two_manifold_edges: nonManifEdges,
					non_two_manifold_vertices: nonManifVerts,
				};

				// Holes and genus need a two-manifold surface to mean
				// anything. Upstream reports -1 rather than a wrong number,
				// and a caller checking is_mesh_two_manifold first will not be
				// misled either way.
				if (twoManifold) {
					const holes = Clean.countHoles(cm);
					out.number_holes = holes;
					out.genus = Clean.meshGenus(cm.vn, counts.total, cm.fn, holes, components);
				} else {
					out.number_holes = -1;
					out.genus = -1;
				}

				doc.Log.log(
					`V: ${cm.vn} E: ${counts.total} F: ${cm.fn} — ` +
						`${components} connected component(s), ` +
						(twoManifold
							? `${out.number_holes} hole(s), genus ${out.genus}`
							: `not two-manifold (${nonManifEdges} bad edges, ${nonManifVerts} bad vertices)`),
				);
				return out;
			}

			case FP.COMPUTE_GEOMETRIC_MEASURES: {
				UpdateBounding.box(cm);
				const props = Inertia.computeMassProperties(cm);
				const barycenter = Inertia.vertexBarycenter(cm);
				const edges = Inertia.edgeLengthStats(cm);
				const watertight = Clean.isWaterTight(cm);

				const out: FilterOutput = {
					bbox_min: [...cm.bbox.min],
					bbox_max: [...cm.bbox.max],
					bbox_diagonal: cm.bbox.diagonal,
					barycenter: [...barycenter],
					total_edge_length: edges.total,
					avg_edge_length: edges.average,
				};

				if (cm.fn > 0) {
					out.surface_area = props.area;
					out.shell_barycenter = [...props.shellBarycenter];
				}

				// Volume, centre of mass and the inertia tensor are integrals
				// over an interior, so they only exist once there is one. An
				// open surface has no inside, and reporting a number computed
				// from its boundary anyway would be worse than reporting none.
				if (watertight && cm.fn > 0) {
					out.mesh_volume = props.volume;
					out.center_of_mass = [...props.centerOfMass];
					out.inertia_tensor = [...props.inertiaTensor];
				}

				doc.Log.log(
					`Mesh Bounding Box Size ${cm.bbox.dimX} ${cm.bbox.dimY} ${cm.bbox.dimZ}\n` +
						`Mesh Bounding Box Diag ${cm.bbox.diagonal}\n` +
						(cm.fn > 0 ? `Mesh Surface Area is ${props.area}\n` : "") +
						(watertight && cm.fn > 0
							? `Mesh Volume is ${props.volume}`
							: "Mesh is not watertight: no volume"),
				);
				return out;
			}

			case FP.PER_VERTEX_QUALITY_STAT: {
				const values: number[] = [];
				for (let v = 0; v < cm.vertSize; v++) {
					if (!cm.isVertD(v)) values.push(cm.vertQuality[v]);
				}
				return this.stats(doc, values, "vertex");
			}

			case FP.PER_FACE_QUALITY_STAT: {
				const values: number[] = [];
				if (cm.faceQuality !== null) {
					for (let f = 0; f < cm.faceSize; f++) {
						if (!cm.isFaceD(f)) values.push(cm.faceQuality[f]);
					}
				}
				return this.stats(doc, values, "face");
			}

			case FP.COMPUTE_TOPOLOGICAL_MEASURES_QUAD_MESHES: {
				m.updateDataMask(MeshElement.MM_FACEFACETOPO);
				if (!isFFAdjacencyConsistent(cm)) {
					throw new MLException("Error: mesh has a not consistent FF adjacency");
				}
				if (!hasConsistentPerFaceFauxFlag(cm)) {
					throw new MLException("QuadMesh problem: mesh has a not consistent FauxEdge tagging");
				}

				const tris = countBitTris(cm);
				const polys = countBitPolygons(cm);
				const largePolys = countBitLargePolygons(cm);
				// A polygon with interior ("faux") vertices makes the quad count
				// meaningless, so it is reported as zero.
				//
				// This is the one place where we do NOT copy upstream. MeshLab
				// writes `if (nLargePolys > 0) nQuads = 0;`, but
				// CountBitLargePolygons returns the polygon count corrected for
				// faux vertices — for a clean quad mesh it equals
				// CountBitPolygons, which is positive whenever the mesh has any
				// faces at all. Taken literally, MeshLab therefore reports zero
				// quads for every quad mesh. The condition that matches the
				// comment beside it, and the one used here, is that the
				// correction is non-zero: some vertex really is interior.
				const quads = largePolys > polys ? 0 : countBitQuads(cm);

				doc.Log.log(
					`Mesh has ${tris} triangles, ${quads} quads, ${polys} polygons, ` +
						`${largePolys} large polygons (with internal faux vertices)`,
				);
				if (!isBitTriQuadOnly(cm)) {
					throw new MLException("QuadMesh problem: the mesh is not TriQuadOnly");
				}

				const angles = new Distribution();
				const ratios = new Distribution();
				const visited = new Uint8Array(cm.faceSize);
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || visited[f] === 1) continue;
					visited[f] = 1;
					const quad = quadCorners(cm, f);
					if (quad === null) {
						throw new MLException("QuadMesh problem: current mesh doesn't contain quads.");
					}
					// Both halves of the quad reach the same corners, so the
					// other one must not contribute a second time.
					const partner = partnerFace(cm, f);
					if (partner >= 0) visited[partner] = 1;

					for (let i = 0; i < 4; i++) {
						angles.Add(Math.abs(90 - cornerAngle(quad, i)));
					}
					const lengths: number[] = [];
					for (let i = 0; i < 4; i++) {
						lengths.push(distance(quad[i], quad[(i + 1) % 4]));
					}
					lengths.sort((a, b) => a - b);
					ratios.Add(lengths[3] === 0 ? 0 : lengths[0] / lengths[3]);
				}

				doc.Log.log(
					`Right Angle Discrepancy Avg ${angles.Avg()} Min ${angles.Min()} Max ${angles.Max()}\n` +
						`Quad Ratio Avg ${ratios.Avg()} Min ${ratios.Min()} Max ${ratios.Max()}`,
				);
				return {
					triangles_number: tris,
					quads_number: quads,
					polys_number: polys,
					large_polys_number: largePolys,
					right_angle_discrepancy_avg: angles.Avg(),
					right_angle_discrepancy_min: angles.Min(),
					right_angle_discrepancy_max: angles.Max(),
					right_angle_discrepancy_stddev: angles.StandardDeviation(),
					"right_angle_discrepancy_perc0.05": angles.Percentile(0.05),
					right_angle_discrepancy_perc95: angles.Percentile(0.95),
					quad_ratio_avg: ratios.Avg(),
					quad_ratio_min: ratios.Min(),
					quad_ratio_max: ratios.Max(),
				};
			}

			case FP.COMPUTE_AREA_PERIMETER_SELECTION: {
				let selected = 0;
				let area = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					selected++;
					area += triangleArea(cm, f);
				}
				if (selected === 0) throw new MLException("Cannot apply: there is no face selection");

				// The perimeter is every edge of the selection whose other side
				// is not selected — a border of the mesh counts, since a face
				// is never its own neighbour's neighbour there.
				m.updateDataMask(MeshElement.MM_FACEFACETOPO);
				let borderEdges = 0;
				let perimeter = 0;
				for (let f = 0; f < cm.faceSize; f++) {
					if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
					for (let e = 0; e < 3; e++) {
						const adj = cm.ffp(f, e);
						if (adj !== f && cm.isFaceS(adj)) continue;
						borderEdges++;
						const a = cm.fv(f, e);
						const b = cm.fv(f, (e + 1) % 3);
						perimeter += Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
					}
				}

				doc.Log.log(
					`Selection is ${selected} triangles, area ${area}, ` +
						`border ${borderEdges} edges, perimeter ${perimeter}`,
				);
				return {
					seleced_triangles_number: selected,
					selected_surface_area: area,
					border_edge_number: borderEdges,
					perimeter,
				};
			}

			case FP.PER_VERTEX_QUALITY_HISTOGRAM:
			case FP.PER_FACE_QUALITY_HISTOGRAM: {
				const perFace = id === FP.PER_FACE_QUALITY_HISTOGRAM;
				const binNum = params.getInt("binNum");
				if (binNum < 1) throw new MLException(`the bin count must be at least 1, got ${binNum}`);
				const areaWeighted = params.getBool("areaWeighted");

				const histogram = new Histogram();
				histogram.SetRange(params.getFloat("HistMin"), params.getFloat("HistMax"), binNum);

				if (perFace) {
					const q = cm.faceQuality;
					if (q === null) throw new MLException("This filter needs per-face quality.");
					for (let f = 0; f < cm.faceSize; f++) {
						if (cm.isFaceD(f)) continue;
						histogram.Add(q[f], areaWeighted ? triangleArea(cm, f) : 1);
					}
				} else {
					const weights = areaWeighted ? perVertexArea(cm) : null;
					for (let v = 0; v < cm.vertSize; v++) {
						if (cm.isVertD(v)) continue;
						histogram.Add(cm.vertQuality[v], weights === null ? 1 : weights[v]);
					}
				}

				// Two extra bins on the ends, carrying everything outside the
				// range; their outer bounds are infinite rather than clamped so
				// that "outside" is visible instead of merged into the edge.
				const binMin: number[] = [];
				const binMax: number[] = [];
				const counts: number[] = [];
				for (let i = 0; i <= binNum + 1; i++) {
					binMin.push(histogram.BinLowerBound(i));
					binMax.push(histogram.BinUpperBound(i));
					counts.push(histogram.BinCountInd(i));
				}
				doc.Log.log(
					`Histogram of per-${perFace ? "face" : "vertex"} quality over ${binNum} bins` +
						(areaWeighted ? ", area weighted" : ""),
				);
				return { hist_bin_min: binMin, hist_bin_max: binMax, hist_count: counts };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}

	private stats(doc: MeshDocument, values: readonly number[], what: string): FilterOutput {
		if (values.length === 0) {
			doc.Log.log(`No per-${what} quality to report`);
			return { min: 0, max: 0, avg: 0, med: 0, stddev: 0, variance: 0 };
		}
		const d = new Distribution();
		for (const x of values) d.Add(x);
		doc.Log.log(
			`Per-${what} quality: min ${d.Min()} max ${d.Max()} avg ${d.Avg()} med ${d.Percentile(0.5)}`,
		);
		return {
			min: d.Min(),
			max: d.Max(),
			avg: d.Avg(),
			med: d.Percentile(0.5),
			stddev: d.StandardDeviation(),
			variance: d.Variance(),
		};
	}
}

type Corner = readonly [number, number, number];

/** Whatever range the mesh's quality spans, for the histogram defaults. */
function qualityRange(cm: CMeshO | undefined, perFace: boolean): { min: number; max: number } {
	if (cm === undefined) return { min: 0, max: 1 };
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	if (perFace) {
		const q = cm.faceQuality;
		if (q === null) return { min: 0, max: 1 };
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			min = Math.min(min, q[f]);
			max = Math.max(max, q[f]);
		}
	} else {
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			min = Math.min(min, cm.vertQuality[v]);
			max = Math.max(max, cm.vertQuality[v]);
		}
	}
	return Number.isFinite(min) ? { min, max } : { min: 0, max: 1 };
}

function triangleArea(cm: CMeshO, f: number): number {
	const a = cm.fv(f, 0);
	const b = cm.fv(f, 1);
	const c = cm.fv(f, 2);
	const ux = cm.vx(b) - cm.vx(a);
	const uy = cm.vy(b) - cm.vy(a);
	const uz = cm.vz(b) - cm.vz(a);
	const vx = cm.vx(c) - cm.vx(a);
	const vy = cm.vy(c) - cm.vy(a);
	const vz = cm.vz(c) - cm.vz(a);
	return Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
}

/**
 * A third of each incident triangle's area, per vertex.
 *
 * The barycentric split. It is not the Voronoi area a curvature estimator
 * would want, but it is what MeshLab weights a histogram by, and the two
 * agree in total: both sum to the surface area.
 */
function perVertexArea(cm: CMeshO): Float64Array {
	const out = new Float64Array(cm.vertSize);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const third = triangleArea(cm, f) / 3;
		for (let k = 0; k < 3; k++) out[cm.fv(f, k)] += third;
	}
	return out;
}

/**
 * The four corners of the quad that `f` is half of, in order around it, or
 * null when `f` is a plain triangle.
 *
 * The corner opposite the faux edge comes first, then the far face's opposite
 * corner, then the two shared ones — which walks the quad's boundary rather
 * than crossing it.
 */
function quadCorners(cm: CMeshO, f: number): readonly Corner[] | null {
	for (let i = 0; i < 3; i++) {
		const only =
			(cm.faceFlags[f] & fauxBit(i)) !== 0 &&
			(cm.faceFlags[f] & fauxBit((i + 1) % 3)) === 0 &&
			(cm.faceFlags[f] & fauxBit((i + 2) % 3)) === 0;
		if (!only) continue;
		const other = cm.ffp(f, i);
		const otherEdge = cm.ffi(f, i);
		if (other === f) return null;
		return [
			point(cm, cm.fv(f, i)),
			point(cm, cm.fv(other, (otherEdge + 2) % 3)),
			point(cm, cm.fv(f, (i + 1) % 3)),
			point(cm, cm.fv(f, (i + 2) % 3)),
		];
	}
	return null;
}

/** The face on the other side of `f`'s faux edge, or -1. */
function partnerFace(cm: CMeshO, f: number): number {
	for (let i = 0; i < 3; i++) {
		if ((cm.faceFlags[f] & fauxBit(i)) !== 0) return cm.ffp(f, i);
	}
	return -1;
}

function point(cm: CMeshO, v: number): Corner {
	return [cm.vx(v), cm.vy(v), cm.vz(v)];
}

function distance(a: Corner, b: Corner): number {
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** The interior angle at corner `i` of a quad, in degrees. */
function cornerAngle(quad: readonly Corner[], i: number): number {
	const at = quad[(i + 1) % 4];
	const u = sub(quad[i], at);
	const v = sub(quad[(i + 2) % 4], at);
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	const dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
	return (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
}

function sub(a: Corner, b: Corner): Corner {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
