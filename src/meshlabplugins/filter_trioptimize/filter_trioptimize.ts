/**
 * `filter_trioptimize` — improving a triangulation without moving the surface.
 *
 * Two of the three flip edges. A flip changes nothing about where the surface
 * is; it only changes which pairs of vertices are joined, so it is the
 * cheapest way to improve a mesh that is already the right shape. The two
 * differ in what they optimise for: planar flipping wants well-shaped
 * triangles, curvature flipping wants the triangulation to follow the
 * surface's own curvature.
 *
 * The third moves vertices, but only as far as the original surface allows —
 * a Laplacian smooth that refuses to round off the shape.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichEnum, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import { buildVertexFaces, edgePairOf, flipEdge, triQuality } from "../../vcg/complex/edge_ops.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

export const FP = {
	FP_PLANAR_EDGE_FLIP: 0,
	FP_CURVATURE_EDGE_FLIP: 1,
	FP_NEAR_LAPLACIAN_SMOOTH: 2,
} as const;

/** The shape measures upstream offers for planar flipping. */
const PLANAR = { QUALITY: 0, DELAUNAY: 1, DEGREE: 2 } as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_PLANAR_EDGE_FLIP]: {
		name: "Planar flipping optimization",
		pythonName: "meshing_edge_flip_by_planar_optimization",
		info: "Mesh optimization by edge flipping, to improve local triangle quality",
		filterClass: FilterClass.Remeshing,
	},
	[FP.FP_CURVATURE_EDGE_FLIP]: {
		name: "Curvature flipping optimization",
		pythonName: "meshing_edge_flip_by_curvature_optimization",
		info: "Mesh optimization by edge flipping, to improve local mesh curvature",
		filterClass: FilterClass.Remeshing,
	},
	[FP.FP_NEAR_LAPLACIAN_SMOOTH]: {
		name: "Laplacian Smooth (surface preserving)",
		pythonName: "apply_coord_laplacian_smoothing_surface_preserving",
		info:
			"Laplacian smooth with limited surface modification: move each vertex in the average " +
			"position of neighbors vertices, only if the new position still (almost) lies on original " +
			"surface",
		filterClass: FilterClass.Smoothing,
	},
};

export class FilterTriOptimize extends FilterPlugin {
	pluginName(): string {
		return "FilterTriOptimize";
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
	override getRequirements(_id: ActionIDType): number {
		return MeshElement.MM_FACEFACETOPO;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		const hasSelection = m !== undefined && countSelected(m.cm) > 0;

		switch (id) {
			case FP.FP_PLANAR_EDGE_FLIP:
				list.add(
					new RichBool("selection", hasSelection, {
						description: "Update selection",
						tooltip: "Restrict the flipping to the current selection.",
					}),
				);
				list.add(
					new RichFloat("pthreshold", 1, {
						description: "Planar threshold (deg)",
						tooltip:
							"The angle between two faces, above which the edge between them is treated as a " +
							"feature and left alone.",
					}),
				);
				list.add(
					new RichEnum(
						"planartype",
						PLANAR.QUALITY,
						["area/max side", "inradius/circumradius", "mean ratio"],
						{
							description: "Planar metric",
							tooltip: "Choose the shape measure the flip tries to improve.",
						},
					),
				);
				break;

			case FP.FP_CURVATURE_EDGE_FLIP:
				list.add(
					new RichBool("selection", hasSelection, {
						description: "Update selection",
						tooltip: "Restrict the flipping to the current selection.",
					}),
				);
				list.add(
					new RichFloat("pthreshold", 1, {
						description: "Curvature threshold",
						tooltip: "Minimum improvement in the total curvature for a flip to be worth doing.",
					}),
				);
				list.add(
					new RichEnum("curvtype", 0, ["mean", "norm squared", "absolute"], {
						description: "Curvature metric",
						tooltip: "Choose the curvature the flip tries to reduce.",
					}),
				);
				break;

			default:
				list.add(
					new RichBool("selection", false, {
						description: "Update selection",
						tooltip: "Restrict the smoothing to the current selection.",
					}),
				);
				list.add(
					new RichFloat("AngleDeg", 0.5, {
						description: "Max Normal Dev (deg)",
						tooltip:
							"How far a vertex's normal may turn as a result of the move. A vertex whose move " +
							"would turn it further is left where it is, which is what keeps the surface.",
					}),
				);
				list.add(
					new RichInt("iterations", 1, {
						description: "Iterations",
						tooltip: "How many smoothing passes to run.",
					}),
				);
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
		m.updateDataMask(MeshElement.MM_FACEFACETOPO);
		const onlySelected = params.getBool("selection");

		if (id === FP.FP_NEAR_LAPLACIAN_SMOOTH) {
			const iterations = params.getInt("iterations");
			if (iterations < 1)
				throw new MLException(`the iteration count must be at least 1, got ${iterations}`);
			const limit = Math.cos((params.getFloat("AngleDeg") * Math.PI) / 180);
			const moved = surfacePreservingSmooth(cm, iterations, limit, onlySelected);
			m.updateBoxAndNormals();
			doc.Log.log(`Moved ${moved} vertices, leaving the rest where the surface required`);
			return { moved_vertices: moved };
		}

		const curvature = id === FP.FP_CURVATURE_EDGE_FLIP;
		const threshold = params.getFloat("pthreshold");
		const metric = curvature ? params.getEnum("curvtype") : params.getEnum("planartype");
		const flipped = optimizeByFlipping(cm, {
			curvature,
			threshold,
			metric,
			onlySelected,
		});

		UpdateTopology.faceFace(cm);
		m.updateBoxAndNormals();
		post.mask = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;
		doc.Log.log(`Flipped ${flipped} edges`);
		return { flipped_edges: flipped };
	}
}

interface FlipOptions {
	readonly curvature: boolean;
	readonly threshold: number;
	readonly metric: number;
	readonly onlySelected: boolean;
}

/**
 * Repeatedly flips whichever edge improves the mesh most, until none does.
 *
 * A flip is only legal on an interior edge whose two faces form a convex
 * quadrilateral — otherwise the flipped triangles overlap. That check is why
 * this cannot simply be a sort by improvement: flipping one edge changes
 * whether its neighbours are legal.
 */
function optimizeByFlipping(cm: CMeshO, options: FlipOptions): number {
	UpdateTopology.faceFace(cm);
	UpdateNormal.perFaceNormalized(cm);
	const vertFaces = buildVertexFaces(cm);
	let flipped = 0;

	// A bounded number of sweeps: each sweep flips everything that helps, and
	// the mesh settles quickly. Without the bound a pair of edges that keep
	// improving each other would spin forever.
	for (let sweep = 0; sweep < 20; sweep++) {
		let did = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			if (options.onlySelected && !cm.isFaceS(f)) continue;
			for (let e = 0; e < 3; e++) {
				if (cm.isBorderFF(f, e)) continue;
				const other = cm.ffp(f, e);
				if (other < 0 || cm.isFaceD(other) || other === f) continue;
				if (options.onlySelected && !cm.isFaceS(other)) continue;

				const gain = flipGain(cm, f, e, other, options);
				if (gain <= 0) continue;
				const pair = edgePairOf(cm, vertFaces, cm.fv(f, e), cm.fv(f, (e + 1) % 3));
				if (pair === null) continue;
				if (flipEdge(cm, vertFaces, pair)) {
					did++;
					UpdateTopology.faceFace(cm);
					UpdateNormal.perFaceNormalized(cm);
					break;
				}
			}
		}
		flipped += did;
		if (did === 0) break;
	}
	return flipped;
}

/** How much better the mesh gets from this flip; zero or less means leave it. */
function flipGain(cm: CMeshO, f: number, e: number, other: number, options: FlipOptions): number {
	const a = cm.fv(f, e);
	const b = cm.fv(f, (e + 1) % 3);
	const c = cm.fv(f, (e + 2) % 3);
	const oppositeEdge = cm.ffi(f, e);
	const d = cm.fv(other, (oppositeEdge + 2) % 3);
	if (c === d) return 0;

	if (options.curvature) {
		// Flipping should reduce the surface's discrete curvature: the angle
		// between the two faces. A flat pair has nothing to gain.
		const before = dihedral(cm, f, other);
		const after = dihedralOf(cm, [c, d, a], [d, c, b]);
		return before - after - options.threshold * 0.0001;
	}

	// Planar: a feature edge is left alone whatever the shape gain.
	const featureLimit = Math.cos((options.threshold * Math.PI) / 180);
	if (Math.cos(dihedral(cm, f, other)) < featureLimit) return 0;

	const before = Math.min(
		quality(cm, a, b, c, options.metric),
		quality(cm, b, a, d, options.metric),
	);
	const after = Math.min(
		quality(cm, c, d, a, options.metric),
		quality(cm, d, c, b, options.metric),
	);
	return after - before;
}

function quality(cm: CMeshO, a: number, b: number, c: number, metric: number): number {
	const p = [a, b, c].map((v) => [cm.vx(v), cm.vy(v), cm.vz(v)]);
	if (metric === PLANAR.DEGREE) {
		// Mean ratio: the equilateral-normalised ratio of area to the sum of
		// the squared sides. Zero for a degenerate triangle, one for an
		// equilateral one, like the other two.
		const l = distance2(p[0], p[1]) + distance2(p[1], p[2]) + distance2(p[2], p[0]);
		if (l === 0) return 0;
		const area = triangleArea(p);
		return (4 * Math.sqrt(3) * area) / l;
	}
	// Both remaining metrics are already in edge_ops.
	return triQuality(
		p[0][0],
		p[0][1],
		p[0][2],
		p[1][0],
		p[1][1],
		p[1][2],
		p[2][0],
		p[2][1],
		p[2][2],
	);
}

function dihedral(cm: CMeshO, f: number, g: number): number {
	const dot =
		cm.faceNormal[3 * f] * cm.faceNormal[3 * g] +
		cm.faceNormal[3 * f + 1] * cm.faceNormal[3 * g + 1] +
		cm.faceNormal[3 * f + 2] * cm.faceNormal[3 * g + 2];
	return Math.acos(Math.min(1, Math.max(-1, dot)));
}

/** The dihedral angle two hypothetical triangles would make. */
function dihedralOf(cm: CMeshO, t1: readonly number[], t2: readonly number[]): number {
	const n1 = normalOf(cm, t1);
	const n2 = normalOf(cm, t2);
	const dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
	return Math.acos(Math.min(1, Math.max(-1, dot)));
}

function normalOf(cm: CMeshO, t: readonly number[]): number[] {
	const p = t.map((v) => [cm.vx(v), cm.vy(v), cm.vz(v)]);
	const u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
	const w = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
	const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
	const length = Math.hypot(n[0], n[1], n[2]) || 1;
	return [n[0] / length, n[1] / length, n[2] / length];
}

/**
 * Laplacian smoothing that refuses to change the shape.
 *
 * A vertex moves to the average of its neighbours only if its normal barely
 * turns as a result. On a flat region every vertex moves and the mesh becomes
 * regular; on a crease none of them do, so the crease survives — which is the
 * difference between this and the plain Laplacian smooth in filter_unsharp.
 */
function surfacePreservingSmooth(
	cm: CMeshO,
	iterations: number,
	normalLimit: number,
	onlySelected: boolean,
): number {
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const rings: Array<Set<number>> = Array.from({ length: cm.vertSize }, () => new Set<number>());
	// The faces around each vertex, gathered once. Rebuilding the normal from
	// a full face sweep per vertex would make this quadratic.
	const incident: Array<number[]> = Array.from({ length: cm.vertSize }, () => []);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			rings[cm.fv(f, k)].add(cm.fv(f, (k + 1) % 3));
			rings[cm.fv(f, (k + 1) % 3)].add(cm.fv(f, k));
			incident[cm.fv(f, k)].push(f);
		}
	}

	let moved = 0;
	for (let pass = 0; pass < iterations; pass++) {
		const before = Float64Array.from(cm.vertNormal);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			if (onlySelected && !cm.isVertS(v)) continue;
			const ring = rings[v];
			if (ring.size === 0) continue;

			let x = 0;
			let y = 0;
			let z = 0;
			for (const n of ring) {
				x += cm.vx(n);
				y += cm.vy(n);
				z += cm.vz(n);
			}
			const target = [x / ring.size, y / ring.size, z / ring.size];
			const old = [cm.vx(v), cm.vy(v), cm.vz(v)];
			cm.setVert(v, target[0], target[1], target[2]);

			// Recompute this vertex's normal from its own faces and compare.
			const after = vertexNormal(cm, v, incident);
			const dot =
				after[0] * before[3 * v] + after[1] * before[3 * v + 1] + after[2] * before[3 * v + 2];
			if (dot < normalLimit) {
				cm.setVert(v, old[0], old[1], old[2]);
			} else {
				moved++;
			}
		}
		UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	}
	return moved;
}

/** The mean of the normals of the faces around a vertex. */
function vertexNormal(cm: CMeshO, v: number, incident: ReadonlyArray<number[]>): number[] {
	const n = [0, 0, 0];
	for (const f of incident[v]) {
		if (cm.isFaceD(f)) continue;
		const face = normalOf(cm, [cm.fv(f, 0), cm.fv(f, 1), cm.fv(f, 2)]);
		for (let k = 0; k < 3; k++) n[k] += face[k];
	}
	const length = Math.hypot(n[0], n[1], n[2]) || 1;
	return [n[0] / length, n[1] / length, n[2] / length];
}

function countSelected(cm: CMeshO): number {
	let n = 0;
	for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f) && cm.isFaceS(f)) n++;
	return n;
}

function distance2(a: readonly number[], b: readonly number[]): number {
	return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

function triangleArea(p: readonly number[][]): number {
	const u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
	const w = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
	return (
		Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2
	);
}
