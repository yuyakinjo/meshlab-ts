/**
 * `filter_measure` — reporting what a mesh is.
 *
 * These filters change nothing; their whole product is the output map. The
 * keys are upstream's exactly, because a caller reads them by name.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
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
import { Clean } from "../../vcg/complex/clean.ts";
import { Inertia } from "../../vcg/complex/inertia.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

export const FP = {
	COMPUTE_TOPOLOGICAL_MEASURES: 0,
	COMPUTE_GEOMETRIC_MEASURES: 1,
	PER_VERTEX_QUALITY_STAT: 2,
	PER_FACE_QUALITY_STAT: 3,
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
	override initParameterList(): RichParameterList {
		return new RichParameterList();
	}

	/** Measuring reads; it never writes. */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
	}

	applyFilter(
		id: ActionIDType,
		_params: RichParameterList,
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

			default:
				return this.wrongActionCalled(id);
		}
	}

	private stats(doc: MeshDocument, values: readonly number[], what: string): FilterOutput {
		if (values.length === 0) {
			doc.Log.log(`No per-${what} quality to report`);
			return { min: 0, max: 0, mean: 0, variance: 0, stddev: 0 };
		}
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		let sum = 0;
		for (const x of values) {
			if (x < min) min = x;
			if (x > max) max = x;
			sum += x;
		}
		const mean = sum / values.length;
		let sq = 0;
		for (const x of values) sq += (x - mean) * (x - mean);
		const variance = sq / values.length;
		doc.Log.log(`Per-${what} quality: min ${min} max ${max} mean ${mean}`);
		return { min, max, mean, variance, stddev: Math.sqrt(variance) };
	}
}
