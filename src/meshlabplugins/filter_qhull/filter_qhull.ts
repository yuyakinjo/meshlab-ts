/**
 * `filter_qhull` — the four filters MeshLab builds on Qhull.
 *
 * Upstream links Qhull and asks it for a convex hull or a Delaunay
 * triangulation; here those two come from `vcg/space/convex_hull.ts` and
 * `vcg/space/delaunay3.ts`, and everything above them is the same algorithm.
 *
 * All four are really one idea seen from different angles. The Delaunay
 * tetrahedralization and the Voronoi diagram are duals, so a tetrahedron's
 * circumcentre is a Voronoi vertex and its circumradius is the size of the
 * empty ball sitting there. Alpha shapes threshold that radius; Voronoi
 * filtering picks out the two extreme Voronoi vertices per sample; and the
 * visible-points operator is a convex hull after a change of coordinates.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichDirection,
	RichDynamicFloat,
	RichEnum,
	RichPercentage,
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
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { VertexFlag } from "../../vcg/complex/flags.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { convexHull } from "../../vcg/space/convex_hull.ts";
import {
	delaunay3,
	type Tetrahedron,
	tetraFaces,
	triangleCircumradius,
} from "../../vcg/space/delaunay3.ts";

export const FP = {
	FP_QHULL_CONVEX_HULL: 0,
	FP_QHULL_VORONOI_FILTERING: 1,
	FP_QHULL_ALPHA_COMPLEX_AND_SHAPE: 2,
	FP_QHULL_VISIBLE_POINTS: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_QHULL_CONVEX_HULL]: {
		name: "Convex Hull",
		pythonName: "generate_convex_hull",
		info:
			"Calculate the <b>convex hull</b> with Qhull library " +
			"(http://www.qhull.org/html/qconvex.htm).<br><br> The convex hull of a set of points is " +
			"the boundary of the minimal convex set containing the given non-empty finite set of points.",
		filterClass: FilterClass.Remeshing,
	},
	[FP.FP_QHULL_VORONOI_FILTERING]: {
		name: "Voronoi Filtering",
		pythonName: "generate_voronoi_filtering",
		info:
			"Compute a <b>Voronoi filtering</b> (Amenta and Bern 1998) with Qhull library " +
			"(http://www.qhull.org/). <br><br>The algorithm calculates a triangulation of the input " +
			"point cloud without requiring vertex normals.It uses a subset of the Voronoi vertices to " +
			"remove triangles from the Delaunay triangulation. <br>After computing the Voronoi " +
			"diagram, foreach sample point it chooses the two farthest opposite Voronoi vertices." +
			"Then computes a Delaunay triangulation of the sample points and the selected Voronoi " +
			"vertices, and keep only those triangles in witch all three vertices are sample points.",
		filterClass: FilterClass.Remeshing,
	},
	[FP.FP_QHULL_ALPHA_COMPLEX_AND_SHAPE]: {
		name: "Alpha Complex/Shape",
		pythonName: "generate_alpha_shape",
		info:
			"Calculate the <b>Alpha Shape</b> of the mesh(Edelsbrunner and P.Mucke 1994) with Qhull " +
			"library (http://www.qhull.org/). <br><br>From a given finite point set in the space it " +
			"computes 'the shape' of the set.The Alpha Shape is the boundary of the alpha complex, " +
			"that is a subcomplex of the Delaunay triangulation of the given point set.<br>For a " +
			"given value of 'alpha', the alpha complex includes all the simplices in the Delaunay " +
			"triangulation which have an empty circumsphere with radius equal or smaller than " +
			"'alpha'.<br>The filter inserts the minimum value of alpha (the circumradius of the " +
			"triangle) in attribute Quality foreach face.",
		filterClass: FilterClass.Remeshing,
	},
	[FP.FP_QHULL_VISIBLE_POINTS]: {
		name: "Select Convex Hull Visible Points",
		pythonName: "compute_selection_of_visible_convex_hull_per_vertex",
		info:
			"Select the <b>visible points</b> in the convex hull of a point cloud, as viewed from a " +
			"given viewpoint.<br>It uses the Qhull library (http://www.qhull.org/ <br><br>The " +
			"algorithm used (Katz, Tal and Basri 2007) determines visibility without reconstructing a " +
			"surface or estimating normals.A point is considered visible if its transformed point " +
			"lies on the convex hull of a transformed points cloud from the original mesh points.",
		filterClass: FilterClass.Selection | FilterClass.PointSet,
	},
};

export class FilterQhull extends FilterPlugin {
	pluginName(): string {
		return "FilterQhull";
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

	override postCondition(id: ActionIDType): number {
		// Only the visible-points filter edits the current layer; the other
		// three add one and leave the input untouched.
		return id === FP.FP_QHULL_VISIBLE_POINTS ? MeshElement.MM_VERTFLAGSELECT : MeshElement.MM_NONE;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diag = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diag = m.cm.bbox.diagonal || 1;
		}

		switch (id) {
			case FP.FP_QHULL_VORONOI_FILTERING:
				list.add(
					new RichDynamicFloat("threshold", 10, 0, 2000, {
						description: "Pole Discard Thr",
						tooltip:
							"Threshold used to discard the Voronoi vertices too far from the origin. We discard " +
							"vertices are further than this factor times the bbox diagonal <br>Growing values of " +
							"this value will add more Voronoi vertices for a better tightier surface " +
							"reconstruction.",
					}),
				);
				break;

			case FP.FP_QHULL_ALPHA_COMPLEX_AND_SHAPE:
				list.add(
					new RichPercentage("alpha", diag / 100, 0, diag, {
						description: "Alpha value",
						tooltip: "Compute the alpha value as percentage of the diagonal of the bbox",
					}),
				);
				list.add(
					new RichEnum("Filtering", 0, ["Alpha Complex", "Alpha Shape"], {
						description: "Get:",
						tooltip: "Select the output. The Alpha Shape is the boundary of the Alpha Complex",
					}),
				);
				break;

			case FP.FP_QHULL_VISIBLE_POINTS:
				list.add(
					new RichDynamicFloat("radiusThreshold", 0, 0, 7, {
						description: "radius threshold ",
						tooltip:
							"Bounds the radius of the sphere used to select visible points. It is used to adjust " +
							"the radius of the sphere (calculated as distance between the center and the " +
							"farthest point from it) according to the following equation: <br>radius = radius * " +
							"pow(10,threshold); <br>As the radius increases more points are marked as visible. " +
							"Use a big threshold for dense point clouds, a small one for sparse clouds.",
					}),
				);
				list.add(
					new RichBool("usecamera", false, {
						description: "Use ViewPoint from Mesh Camera",
						tooltip:
							"Uses the ViewPoint from the camera associated to the current mesh\\n if there is no " +
							"camera, an error occurs",
					}),
				);
				list.add(
					new RichDirection("viewpoint", [0, 0, 0], {
						description: "ViewPoint",
						tooltip: "if UseCamera is true, this value is ignored",
					}),
				);
				list.add(
					new RichBool("convex_hullFP", false, {
						description: "Show Partial Convex Hull of flipped points",
						tooltip: "Show Partial Convex Hull of the transformed point cloud",
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
		_cb: CallBackPos,
	): FilterOutput {
		post.mask = this.postCondition(id);
		const m = doc.mm();
		const cm = m.cm;
		const { coords, index } = livePoints(cm);
		if (index.length < 4) {
			throw new MLException(
				`${this.filterName(id)} needs at least four points, and this mesh has ${index.length}.`,
			);
		}

		switch (id) {
			case FP.FP_QHULL_CONVEX_HULL: {
				const hull = convexHull(coords, index.length);
				if (hull === null) {
					throw new MLException(
						"The points are degenerate — all on one line or plane — so they have no convex hull.",
					);
				}
				const out = meshFromFaces(
					coords,
					hull.faces.map((f) => f.v),
				);
				const target = doc.addNewMesh("", `${m.label()} convex hull`, true, out);
				target.updateBoxAndNormals();
				doc.Log.log(`Convex hull: ${out.vn} vertices, ${out.fn} faces`);
				return { new_mesh_id: target.id(), vertex_number: out.vn, face_number: out.fn };
			}

			case FP.FP_QHULL_ALPHA_COMPLEX_AND_SHAPE:
				return this.alphaShape(params, doc, m, coords, index.length);

			case FP.FP_QHULL_VORONOI_FILTERING:
				return this.voronoiFiltering(params, doc, m, coords, index.length);

			case FP.FP_QHULL_VISIBLE_POINTS:
				return this.visiblePoints(params, doc, m, coords, index);

			default:
				return this.wrongActionCalled(id);
		}
	}

	/**
	 * `Alpha Complex/Shape`: the Delaunay simplices small enough to be filled
	 * by a ball of radius alpha.
	 *
	 * The complex is every such triangle; the shape is only those on its
	 * boundary — the faces belonging to exactly one accepted tetrahedron. As
	 * alpha grows the shape sweeps from the point cloud itself up to the convex
	 * hull, which is the property that makes it a shape descriptor rather than
	 * just another reconstruction.
	 */
	private alphaShape(
		params: RichParameterList,
		doc: MeshDocument,
		m: MeshModel,
		coords: Float64Array,
		count: number,
	): FilterOutput {
		const alpha = params.getAbsPerc("alpha");
		if (!(alpha > 0)) throw new MLException(`The alpha value must be positive, got ${alpha}`);
		const wantShape = params.getEnum("Filtering") === 1;

		const tetra = delaunay3(coords, count);
		if (tetra.length === 0) {
			throw new MLException(
				"The points are degenerate — all coplanar — so they have no Delaunay tetrahedralization.",
			);
		}
		const accepted = tetra.filter((t) => t.radius <= alpha);

		// How many accepted tetrahedra each triangle belongs to. One means the
		// triangle is on the complex's boundary, two means it is inside it.
		const shared = new Map<string, { face: [number, number, number]; count: number }>();
		for (const t of accepted) {
			for (const f of tetraFaces(t)) {
				const key = `${f[0]}_${f[1]}_${f[2]}`;
				const seen = shared.get(key);
				if (seen === undefined) shared.set(key, { face: f, count: 1 });
				else seen.count++;
			}
		}

		const at = (i: number): number[] => [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
		const faces: Array<[number, number, number]> = [];
		const alphas: number[] = [];
		for (const { face, count: uses } of shared.values()) {
			if (wantShape && uses !== 1) continue;
			faces.push(face);
			alphas.push(triangleCircumradius(at(face[0]), at(face[1]), at(face[2])));
		}
		// A triangle that no accepted tetrahedron uses can still be in the
		// complex: a sliver on the hull whose own circumcircle is small even
		// though every tetrahedron through it is large.
		if (!wantShape) {
			for (const t of tetra) {
				if (t.radius <= alpha) continue;
				for (const f of tetraFaces(t)) {
					const key = `${f[0]}_${f[1]}_${f[2]}`;
					if (shared.has(key)) continue;
					const r = triangleCircumradius(at(f[0]), at(f[1]), at(f[2]));
					if (r > alpha) continue;
					shared.set(key, { face: f, count: 0 });
					faces.push(f);
					alphas.push(r);
				}
			}
		}

		const out = meshFromFaces(coords, faces);
		out.enableChannels(MeshElement.MM_FACEQUALITY);
		const quality = out.faceQuality as Float64Array;
		// Upstream stores each face's own alpha, so the result can be
		// re-thresholded by colouring rather than by recomputing.
		alphas.forEach((a, i) => {
			quality[i] = a;
		});
		const label = wantShape ? "alpha shape" : "alpha complex";
		const target = doc.addNewMesh("", `${m.label()} ${label}`, true, out);
		target.updateBoxAndNormals();
		doc.Log.log(
			`${label} at alpha ${alpha}: ${accepted.length} of ${tetra.length} tetrahedra, ${out.fn} faces`,
		);
		return {
			new_mesh_id: target.id(),
			vertex_number: out.vn,
			face_number: out.fn,
			tetrahedra: accepted.length,
		};
	}

	/**
	 * `Voronoi Filtering`: Amenta and Bern's reconstruction from poles.
	 *
	 * For each sample the two *poles* are the Voronoi vertices of its cell
	 * furthest away on either side of the surface. They approximate the medial
	 * axis, so a second Delaunay of the samples plus their poles has its
	 * sample-only triangles hugging the surface — the poles crowd out every
	 * tetrahedron that would have bridged across the interior. No normals
	 * needed, which is the whole point.
	 */
	private voronoiFiltering(
		params: RichParameterList,
		doc: MeshDocument,
		m: MeshModel,
		coords: Float64Array,
		count: number,
	): FilterOutput {
		const tetra = delaunay3(coords, count);
		if (tetra.length === 0) {
			throw new MLException(
				"The points are degenerate — all coplanar — so they have no Delaunay tetrahedralization.",
			);
		}
		UpdateBounding.box(m.cm);
		const centre = [0, 1, 2].map((k) => (m.cm.bbox.min[k] + m.cm.bbox.max[k]) / 2);
		const reach = params.getDynamicFloat("threshold") * (m.cm.bbox.diagonal || 1);

		// Every Voronoi vertex of each sample's cell.
		const cells: Tetrahedron[][] = Array.from({ length: count }, () => []);
		for (const t of tetra) for (const v of t.v) cells[v].push(t);

		const poles: number[] = [];
		for (let i = 0; i < count; i++) {
			const p = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
			const region = cells[i];
			if (region.length === 0) continue;

			let first: Tetrahedron | null = null;
			let firstDist = 0;
			for (const t of region) {
				const d = Math.hypot(t.centre[0] - p[0], t.centre[1] - p[1], t.centre[2] - p[2]);
				if (d > firstDist) {
					firstDist = d;
					first = t;
				}
			}
			if (first === null) continue;
			const up = [0, 1, 2].map((k) => first.centre[k] - p[k]);

			// The second pole is the furthest vertex on the *other* side, which
			// is what makes the pair straddle the surface rather than both
			// running off inwards.
			let second: Tetrahedron | null = null;
			let secondDist = 0;
			for (const t of region) {
				if (t === first) continue;
				const d = [0, 1, 2].map((k) => t.centre[k] - p[k]);
				if (d[0] * up[0] + d[1] * up[1] + d[2] * up[2] >= 0) continue;
				const len = Math.hypot(d[0], d[1], d[2]);
				if (len > secondDist) {
					secondDist = len;
					second = t;
				}
			}

			for (const pole of [first, second]) {
				if (pole === null) continue;
				// A pole far outside the box is a Voronoi vertex "at infinity"
				// belonging to a cell open to the outside; keeping it would drag
				// the second triangulation out with it.
				const d = Math.hypot(
					pole.centre[0] - centre[0],
					pole.centre[1] - centre[1],
					pole.centre[2] - centre[2],
				);
				if (d > reach) continue;
				poles.push(pole.centre[0], pole.centre[1], pole.centre[2]);
			}
		}

		const combined = new Float64Array(3 * count + poles.length);
		combined.set(coords.subarray(0, 3 * count));
		combined.set(poles, 3 * count);
		const second = delaunay3(combined, count + poles.length / 3);

		// Keep only the triangles all of whose corners are original samples.
		const kept = new Map<string, [number, number, number]>();
		for (const t of second) {
			for (const f of tetraFaces(t)) {
				if (f[2] >= count) continue;
				kept.set(`${f[0]}_${f[1]}_${f[2]}`, f);
			}
		}

		const out = meshFromFaces(coords, [...kept.values()]);
		const target = doc.addNewMesh("", `${m.label()} voronoi filtered`, true, out);
		target.updateBoxAndNormals();
		doc.Log.log(
			`Voronoi filtering kept ${poles.length / 3} poles and produced ${out.fn} faces from ${count} samples`,
		);
		return {
			new_mesh_id: target.id(),
			vertex_number: out.vn,
			face_number: out.fn,
			poles: poles.length / 3,
		};
	}

	/**
	 * `Select Convex Hull Visible Points`: Katz, Tal and Basri's hidden point
	 * removal.
	 *
	 * Reflect every point through a sphere about the viewpoint — near points go
	 * far, far points come near — and take the convex hull of the result
	 * together with the viewpoint itself. A point survives exactly when it was
	 * visible. No surface and no normals are needed, which is why it works on a
	 * raw scan.
	 */
	private visiblePoints(
		params: RichParameterList,
		doc: MeshDocument,
		m: MeshModel,
		coords: Float64Array,
		index: readonly number[],
	): FilterOutput {
		const cm = m.cm;
		const viewpoint = params.getBool("usecamera")
			? cameraViewpoint(m)
			: [...params.getPoint3m("viewpoint")];

		const count = index.length;
		const relative = new Float64Array(3 * count);
		const dist = new Float64Array(count);
		let radius = 0;
		for (let i = 0; i < count; i++) {
			for (let k = 0; k < 3; k++) relative[3 * i + k] = coords[3 * i + k] - viewpoint[k];
			dist[i] = Math.hypot(relative[3 * i], relative[3 * i + 1], relative[3 * i + 2]);
			radius = Math.max(radius, dist[i]);
		}
		if (radius === 0) {
			throw new MLException("Every point coincides with the viewpoint, so none can be seen.");
		}
		radius *= 10 ** params.getDynamicFloat("radiusThreshold");

		// The spherical flip, plus the viewpoint as the last point so that the
		// hull is anchored on the near side.
		const flipped = new Float64Array(3 * (count + 1));
		for (let i = 0; i < count; i++) {
			// A point exactly at the viewpoint has no direction to flip along.
			const k = dist[i] === 0 ? 0 : (2 * (radius - dist[i])) / dist[i];
			for (let a = 0; a < 3; a++) {
				flipped[3 * i + a] = relative[3 * i + a] * (1 + k);
			}
		}
		// The viewpoint is the origin of this frame.
		flipped[3 * count] = 0;
		flipped[3 * count + 1] = 0;
		flipped[3 * count + 2] = 0;

		const hull = convexHull(flipped, count + 1);
		if (hull === null) {
			throw new MLException(
				"The flipped points are degenerate, so no visibility hull could be built.",
			);
		}

		let selected = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v)) cm.vertFlags[v] &= ~VertexFlag.SELECTED;
		}
		for (const i of hull.vertices) {
			// The viewpoint itself is on the hull by construction and is not a
			// point of the cloud.
			if (i >= count) continue;
			cm.vertFlags[index[i]] |= VertexFlag.SELECTED;
			selected++;
		}

		if (params.getBool("convex_hullFP")) {
			const out = meshFromFaces(
				flipped,
				hull.faces.map((f) => f.v),
			);
			const target = doc.addNewMesh("", `${m.label()} flipped hull`, true, out);
			target.updateBoxAndNormals();
		}
		doc.Log.log(`Selected ${selected} visible points out of ${count}`);
		return { selected_vertices: selected };
	}
}

/** The live vertex coordinates, and which vertex each one came from. */
function livePoints(cm: CMeshO): { coords: Float64Array; index: number[] } {
	const index: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) index.push(v);
	const coords = new Float64Array(3 * index.length);
	index.forEach((v, i) => {
		coords[3 * i] = cm.vx(v);
		coords[3 * i + 1] = cm.vy(v);
		coords[3 * i + 2] = cm.vz(v);
	});
	return { coords, index };
}

/**
 * A mesh holding only the points the given faces use, renumbered.
 *
 * Dropping the unused points matters: a convex hull of a dense cloud would
 * otherwise carry every interior point along as an unreferenced vertex.
 */
function meshFromFaces(
	coords: Float64Array,
	faces: ReadonlyArray<readonly [number, number, number]>,
): CMeshO {
	const cm = new CMeshO();
	const remap = new Map<number, number>();
	for (const f of faces) {
		for (const v of f) if (!remap.has(v)) remap.set(v, remap.size);
	}
	if (remap.size === 0) return cm;

	const first = Allocator.addVertices(cm, remap.size);
	for (const [from, to] of remap) {
		cm.setVert(first + to, coords[3 * from], coords[3 * from + 1], coords[3 * from + 2]);
	}
	const firstFace = Allocator.addFaces(cm, faces.length);
	faces.forEach((f, i) => {
		cm.setFace(
			firstFace + i,
			first + (remap.get(f[0]) as number),
			first + (remap.get(f[1]) as number),
			first + (remap.get(f[2]) as number),
		);
	});
	return cm;
}

function cameraViewpoint(m: MeshModel): number[] {
	if (!m.hasDataMask(MeshElement.MM_CAMERA)) {
		throw new MLException(
			'"Use ViewPoint from Mesh Camera" was asked for, but this mesh has no camera.',
		);
	}
	return [...m.shot.Extrinsics.tra];
}
