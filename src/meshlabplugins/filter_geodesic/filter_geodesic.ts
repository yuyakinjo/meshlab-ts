/**
 * `filter_geodesic` — distance along the surface, written into the quality
 * channel and shown as colour.
 *
 * Three of the four use Dijkstra over the edges; the fourth uses the heat
 * method, which is what its name promises. They differ only in where the
 * sources come from: a point, the selection, or the border.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichFloat, RichPercentage, RichPosition } from "../../common/parameters/rich_parameter.ts";
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { borderVertices, dijkstraGeodesic, heatGeodesic } from "../../vcg/complex/geodesic.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { colorRamp } from "../../vcg/space/color4.ts";

export const FP = {
	FP_QUALITY_POINT_GEODESIC: 0,
	FP_QUALITY_SELECTED_GEODESIC: 1,
	FP_QUALITY_SELECTED_GEODESIC_HEAT: 2,
	FP_QUALITY_BORDER_GEODESIC: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_QUALITY_POINT_GEODESIC]: {
		name: "Colorize by geodesic distance from a given point",
		pythonName: "compute_scalar_by_geodesic_distance_from_given_point_per_vertex",
		info:
			"Store in the quality field the geodesic distance from a given point on the mesh surface " +
			"and color the mesh accordingly.",
	},
	[FP.FP_QUALITY_SELECTED_GEODESIC]: {
		name: "Colorize by geodesic distance from the selected points",
		pythonName: "compute_scalar_by_geodesic_distance_from_selection_per_vertex",
		info:
			"Store in the quality field the geodesic distance from the selected points on the mesh " +
			"surface and color the mesh accordingly.",
	},
	[FP.FP_QUALITY_SELECTED_GEODESIC_HEAT]: {
		name: "Colorize by approximated geodesic distance from the selected points",
		pythonName: "compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex",
		info:
			"Store in the quality field the approximated geodesic distance, computed via heat method " +
			"(Crane et al.), from the selected points on the mesh surface and color the mesh " +
			"accordingly. As this implementation does not use intrinsic triangulation it is very " +
			"sensitive to triangulation.",
	},
	[FP.FP_QUALITY_BORDER_GEODESIC]: {
		name: "Colorize by border distance",
		pythonName: "compute_scalar_by_border_distance_per_vertex",
		info: "Store in the quality field the geodesic distance from borders and color the mesh accordingly.",
	},
};

export class FilterGeodesic extends FilterPlugin {
	pluginName(): string {
		return "FilterGeodesic";
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
		return FilterClass.VertexColoring | FilterClass.Quality;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		let min: [number, number, number] = [0, 0, 0];
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
			min = [m.cm.bbox.min[0], m.cm.bbox.min[1], m.cm.bbox.min[2]];
		}

		if (id === FP.FP_QUALITY_POINT_GEODESIC) {
			list.add(
				new RichPosition("startPoint", min, {
					description: "Starting point",
					tooltip: "The starting point from which geodesic distance has to be computed.",
				}),
			);
			list.add(
				new RichPercentage("maxDistance", diagonal, 0, diagonal, {
					description: "Max Distance",
					tooltip:
						"If not zero it indicates a cut off value to be used during geodesic distance " +
						"computation.",
				}),
			);
		} else if (id === FP.FP_QUALITY_SELECTED_GEODESIC) {
			list.add(
				new RichPercentage("maxDistance", diagonal, 0, diagonal, {
					description: "Max Distance",
					tooltip:
						"If not zero it indicates a cut off value to be used during geodesic distance " +
						"computation.",
				}),
			);
		} else if (id === FP.FP_QUALITY_SELECTED_GEODESIC_HEAT) {
			list.add(
				new RichFloat("m", 1, {
					description: "Heat diffusion time",
					tooltip:
						"A larger value diffuses the heat further before the direction is read off, which " +
						"smooths the result. It is a multiple of the mean edge length squared.",
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
		_cb: CallBackPos,
	): FilterOutput {
		const m = doc.mm();
		const cm = m.cm;
		m.updateDataMask(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTCOLOR);
		if (cm.vn === 0) throw new MLException("the mesh has no vertices");

		const sources = this.sourcesFor(id, params, cm);
		const distance =
			id === FP.FP_QUALITY_SELECTED_GEODESIC_HEAT
				? heatGeodesic(cm, sources, { m: params.getFloat("m") })
				: dijkstraGeodesic(cm, sources);
		if (distance === null) {
			throw new MLException(
				"the heat method did not converge on this mesh; its triangulation is too degenerate. " +
					'Use "Colorize by geodesic distance from the selected points" instead.',
			);
		}

		const cutOff =
			id === FP.FP_QUALITY_POINT_GEODESIC || id === FP.FP_QUALITY_SELECTED_GEODESIC
				? params.getAbsPerc("maxDistance")
				: 0;

		let reached = 0;
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			let d = distance[v];
			// An unreached vertex — a separate component, or one beyond the cut
			// off — gets the cut off rather than infinity, so the colour ramp
			// below is not swamped by a single unreachable island.
			if (!Number.isFinite(d) || (cutOff > 0 && d > cutOff)) {
				d = cutOff > 0 ? cutOff : 0;
			} else {
				reached++;
			}
			cm.vertQuality[v] = d;
			max = Math.max(max, d);
		}
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) cm.vertColor[v] = colorRamp(0, max || 1, cm.vertQuality[v]);
		}

		post.mask = MeshElement.MM_NONE;
		doc.Log.log(
			`Geodesic distance from ${sources.length} source${sources.length === 1 ? "" : "s"}: ` +
				`max ${max}, ${reached} of ${cm.vn} vertices reached`,
		);
		return { sources: sources.length, max_distance: max, reached_vertices: reached };
	}

	private sourcesFor(id: ActionIDType, params: RichParameterList, cm: CMeshO): number[] {
		switch (id) {
			case FP.FP_QUALITY_POINT_GEODESIC: {
				const p = params.getPoint3m("startPoint");
				// Upstream snaps to the nearest vertex; the point is a place on
				// the surface, not necessarily one of its vertices.
				let best = -1;
				let bestDistance = Number.POSITIVE_INFINITY;
				for (let v = 0; v < cm.vertSize; v++) {
					if (cm.isVertD(v)) continue;
					const d = Math.hypot(cm.vx(v) - p[0], cm.vy(v) - p[1], cm.vz(v) - p[2]);
					if (d < bestDistance) {
						bestDistance = d;
						best = v;
					}
				}
				if (best < 0) throw new MLException("the mesh has no vertices to start from");
				return [best];
			}
			case FP.FP_QUALITY_BORDER_GEODESIC: {
				const border = borderVertices(cm);
				if (border.length === 0) {
					throw new MLException("the mesh has no border, so there is no distance to it");
				}
				return border;
			}
			default: {
				const selected: number[] = [];
				for (let v = 0; v < cm.vertSize; v++) {
					if (!cm.isVertD(v) && cm.isVertS(v)) selected.push(v);
				}
				if (selected.length === 0) {
					throw new MLException("no vertex is selected, so there is nothing to measure from");
				}
				return selected;
			}
		}
	}
}
