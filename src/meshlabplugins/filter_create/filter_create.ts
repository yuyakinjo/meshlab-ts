/**
 * `filter_create` — the primitives, and the plane fitted to a selection.
 *
 * These take no input mesh (arity NONE): each adds a new layer and makes it
 * current. `Fit a plane to selection` is the exception in spirit if not in
 * arity — it reads the current layer's selection, and fails without one.
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
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { Platonic } from "../../vcg/complex/create/platonic.ts";
import { VertexFlag } from "../../vcg/complex/flags.ts";
import { Rng, SurfaceSampling } from "../../vcg/complex/point_sampling.ts";
import { GenNormal, type SpherePoint } from "../../vcg/math/gen_normal.ts";

export const CR = {
	CR_BOX: 0,
	CR_ANNULUS: 1,
	CR_SPHERE: 2,
	CR_SPHERE_CAP: 3,
	CR_RANDOM_SPHERE: 4,
	CR_ICOSAHEDRON: 5,
	CR_DODECAHEDRON: 6,
	CR_DODECAHEDRON_SYM: 7,
	CR_OCTAHEDRON: 8,
	CR_TETRAHEDRON: 9,
	CR_CONE: 10,
	CR_TORUS: 11,
	CR_FITPLANE: 12,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[CR.CR_BOX]: {
		name: "Box/Cube",
		pythonName: "create_cube",
		info: "Create a Box, Cube, Hexahedron. You can specify the side length.",
	},
	[CR.CR_ANNULUS]: {
		name: "Annulus",
		pythonName: "create_annulus",
		info: "Create an Annulus e.g. a flat region bounded by two concentric circles, or a holed disk.",
	},
	[CR.CR_SPHERE]: {
		name: "Sphere",
		pythonName: "create_sphere",
		info: "Create a Sphere, whose topology is obtained as regular subdivision of an icosahedron.",
	},
	[CR.CR_SPHERE_CAP]: {
		name: "Sphere Cap",
		pythonName: "create_sphere_cap",
		info: "Create a Sphere Cap, or spherical dome, subtended by a cone of given angle",
	},
	[CR.CR_RANDOM_SPHERE]: {
		name: "Points on a Sphere",
		pythonName: "create_sphere_points",
		info: "Create a spherical point cloud, it can be random or regularly distributed.",
	},
	[CR.CR_ICOSAHEDRON]: {
		name: "Icosahedron",
		pythonName: "create_icosahedron",
		info: "Create an Icosahedron",
	},
	[CR.CR_DODECAHEDRON]: {
		name: "Dodecahedron",
		pythonName: "create_dodecahedron",
		info: "Create a Dodecahedron",
	},
	[CR.CR_DODECAHEDRON_SYM]: {
		name: "Dodecahedron (symmetric)",
		pythonName: "create_dodecahedron_sym",
		info:
			"Create a Dodecahedron, but triangulated with an additional vertex in the middle of each " +
			"face to preserve symmetry.",
	},
	[CR.CR_OCTAHEDRON]: {
		name: "Octahedron",
		pythonName: "create_octahedron",
		info: "Create an Octahedron",
	},
	[CR.CR_TETRAHEDRON]: {
		name: "Tetrahedron",
		pythonName: "create_tetrahedron",
		info: "Create a Tetrahedron",
	},
	[CR.CR_CONE]: { name: "Cone", pythonName: "create_cone", info: "Create a Cone" },
	[CR.CR_TORUS]: { name: "Torus", pythonName: "create_torus", info: "Create a Torus" },
	[CR.CR_FITPLANE]: {
		name: "Fit a plane to selection",
		pythonName: "generate_plane_fitting_to_selection",
		info: "Create a quad on the plane fitting the selection",
	},
};

export class FilterCreate extends FilterPlugin {
	pluginName(): string {
		return "FilterCreate";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(CR);
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
		return FilterClass.MeshCreation;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.NONE;
	}

	/** A layer appears; nothing that already existed changes. */
	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_NONE;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		switch (id) {
			case CR.CR_SPHERE:
				list.add(
					new RichFloat("radius", 1, { description: "Radius", tooltip: "Radius of the sphere" }),
				);
				list.add(
					new RichInt("subdiv", 3, {
						description: "Subdiv. Level",
						tooltip:
							"Number of the recursive subdivision of the surface. Default is 3 (a sphere " +
							"approximation with 1280 faces).<br>Admitted values are in the range 0 (an " +
							"icosahedron) to 8 (a 1.3 MegaTris approximation of a sphere)",
					}),
				);
				break;

			case CR.CR_SPHERE_CAP:
				list.add(
					new RichFloat("angle", 60, {
						description: "Angle",
						tooltip: "Angle of the cone subtending the cap. It must be < 180",
					}),
				);
				list.add(
					new RichInt("subdiv", 3, {
						description: "Subdiv. Level",
						tooltip: "Number of the recursive subdivision of the surface.",
					}),
				);
				break;

			case CR.CR_ANNULUS:
				list.add(
					new RichFloat("internalRadius", 0.5, {
						description: "Internal Radius",
						tooltip: "Internal Radius of the annulus",
					}),
				);
				list.add(
					new RichFloat("externalRadius", 1, {
						description: "External Radius",
						tooltip: "Externale Radius of the annulus",
					}),
				);
				list.add(
					new RichInt("sides", 32, {
						description: "Sides",
						tooltip: "Number of the sides of the poligonal approximation of the annulus ",
					}),
				);
				break;

			case CR.CR_RANDOM_SPHERE:
				list.add(
					new RichInt("pointNum", 100, {
						description: "Point Num",
						tooltip: "Number of points (approximate).",
					}),
				);
				list.add(
					new RichEnum(
						"sphereGenTech",
						3,
						["Montecarlo", "Poisson Sampling", "DiscoBall", "Octahedron", "Fibonacci"],
						{
							description: "Generation Technique:",
							tooltip:
								"Generation Technique:<b>Montecarlo</b>: The points are randomly generated with " +
								"an uniform distribution.<br><b>Poisson Disk</b>: The points are to follow a " +
								"poisson disk distribution.<br><b>Disco Ball</b> Dave Rusin's disco ball " +
								"algorithm for the regular placement of points on a sphere is used. <br>" +
								"<b>Recursive Octahedron</b> Points are generated on the vertex of a recursively " +
								"subdivided octahedron <br><b>Fibonacci</b> . ",
						},
					),
				);
				break;

			case CR.CR_BOX:
				list.add(
					new RichFloat("size", 1, {
						description: "Scale factor",
						tooltip: "Scales the new mesh",
					}),
				);
				break;

			case CR.CR_CONE:
				list.add(
					new RichFloat("r0", 1, {
						description: "Radius 1",
						tooltip: "Radius of the bottom circumference",
					}),
				);
				list.add(
					new RichFloat("r1", 2, {
						description: "Radius 2",
						tooltip: "Radius of the top circumference",
					}),
				);
				list.add(new RichFloat("h", 3, { description: "Height", tooltip: "Height of the Cone" }));
				list.add(
					new RichInt("subdiv", 36, {
						description: "Side",
						tooltip: "Number of sides of the polygonal approximation of the cone",
					}),
				);
				break;

			case CR.CR_TORUS:
				list.add(
					new RichFloat("hRadius", 3, {
						description: "Horizontal Radius",
						tooltip: "Radius of the whole horizontal ring of the torus",
					}),
				);
				list.add(
					new RichFloat("vRadius", 1, {
						description: "Vertical Radius",
						tooltip: "Radius of the vertical section of the ring",
					}),
				);
				list.add(
					new RichInt("hSubdiv", 24, {
						description: "Horizontal Subdivision",
						tooltip: "Subdivision step of the ring",
					}),
				);
				list.add(
					new RichInt("vSubdiv", 12, {
						description: "Vertical Subdivision",
						tooltip: "Number of sides of the polygonal approximation of the torus section",
					}),
				);
				break;

			case CR.CR_FITPLANE:
				list.add(
					new RichFloat("extent", 1, {
						description: "Extent (with respect to selection)",
						tooltip:
							"How large is the plane, with respect to the size of the selection: 1.0 means as " +
							"large as the selection, 1.1 means 10% larger then the selection",
					}),
				);
				list.add(
					new RichInt("subdiv", 3, {
						description: "Plane XY subivisions",
						tooltip: "Subdivision steps of plane borders",
					}),
				);
				list.add(
					new RichBool("hasuv", false, {
						description: "UV parametrized",
						tooltip: "The created plane has an UV parametrization",
					}),
				);
				list.add(
					new RichEnum(
						"orientation",
						0,
						["quasi-Straight Fit", "Best Fit", "XZ Parallel", "YZ Parallel", "XY Parallel"],
						{
							description: "Plane orientation",
							tooltip:
								"Orientation:<b>quasi-Straight Fit</b>: The fitting plane will be oriented (as much " +
								"as possible) straight with the axeses.<br><b>Best Fit</b>: The fitting plane will " +
								"be oriented and sized trying to best fit to the selected area.<br><b>-- Parallel</b>: " +
								"The fitting plane will be oriented with a side parallel with the chosen plane. " +
								"BEWARE: the plane may fail its purpose if not enough vertexes are selected.",
						},
					),
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
		post.mask = MeshElement.MM_NONE;

		switch (id) {
			case CR.CR_TETRAHEDRON:
				return this.addLayer(doc, id, Platonic.tetrahedron());
			case CR.CR_OCTAHEDRON:
				return this.addLayer(doc, id, Platonic.octahedron());
			case CR.CR_ICOSAHEDRON:
				return this.addLayer(doc, id, Platonic.icosahedron());
			case CR.CR_DODECAHEDRON:
				return this.addLayer(doc, id, Platonic.dodecahedron());
			case CR.CR_DODECAHEDRON_SYM:
				return this.addLayer(doc, id, Platonic.dodecahedronSym());

			case CR.CR_BOX: {
				const h = params.getFloat("size") / 2;
				return this.addLayer(doc, id, Platonic.box([-h, -h, -h], [h, h, h]));
			}

			case CR.CR_SPHERE: {
				const radius = params.getFloat("radius");
				const m = Platonic.sphere(params.getInt("subdiv"));
				for (let v = 0; v < m.vertSize; v++) {
					m.setVert(v, m.vx(v) * radius, m.vy(v) * radius, m.vz(v) * radius);
				}
				return this.addLayer(doc, id, m);
			}

			case CR.CR_SPHERE_CAP: {
				const angle = params.getFloat("angle");
				if (!(angle > 0 && angle < 180)) {
					throw new MLException(
						`Sphere Cap needs an angle strictly between 0 and 180, got ${angle}`,
					);
				}
				return this.addLayer(
					doc,
					id,
					Platonic.sphericalCap((angle * Math.PI) / 180, params.getInt("subdiv")),
				);
			}

			case CR.CR_ANNULUS: {
				// The argument order is VCGLib's, and MeshLab passes its own
				// internalRadius into the outer slot; kept as-is so the vertex
				// order matches.
				const m = Platonic.annulus(
					params.getFloat("internalRadius"),
					params.getFloat("externalRadius"),
					params.getInt("sides"),
				);
				return this.addLayer(doc, id, m);
			}

			case CR.CR_CONE:
				return this.addLayer(
					doc,
					id,
					Platonic.cone(
						params.getFloat("r0"),
						params.getFloat("r1"),
						params.getFloat("h"),
						params.getInt("subdiv"),
					),
				);

			case CR.CR_TORUS:
				return this.addLayer(
					doc,
					id,
					Platonic.torus(
						params.getFloat("hRadius"),
						params.getFloat("vRadius"),
						params.getInt("hSubdiv"),
						params.getInt("vSubdiv"),
					),
				);

			case CR.CR_RANDOM_SPHERE:
				return this.addLayer(
					doc,
					id,
					Platonic.pointCloudFrom(
						spherePoints(params.getInt("pointNum"), params.getEnum("sphereGenTech")),
					),
				);

			case CR.CR_FITPLANE:
				return this.addLayer(doc, id, fitPlane(doc, params));

			default:
				return this.wrongActionCalled(id);
		}
	}

	private addLayer(doc: MeshDocument, id: ActionIDType, cm: CMeshO): FilterOutput {
		const m = doc.addNewMesh("", this.filterName(id), true, cm);
		m.updateBoxAndNormals();
		return { new_mesh_id: m.id(), vertex_number: cm.vn, face_number: cm.fn };
	}
}

/** The five ways `Points on a Sphere` can lay its points out. */
function spherePoints(pointNum: number, technique: number): SpherePoint[] {
	if (pointNum <= 0) return [];
	switch (technique) {
		case 0:
			return montecarloSphere(pointNum);
		case 1:
			return poissonSphere(pointNum);
		case 2:
			return GenNormal.discoBall(pointNum);
		case 3:
			return GenNormal.recursiveOctahedron(pointNum);
		case 4:
			return GenNormal.fibonacci(pointNum);
		default:
			throw new MLException(`Unknown sphere generation technique ${technique}`);
	}
}

/** Uniform on the sphere: z uniform in [-1,1] gives equal area per band. */
function montecarloSphere(pointNum: number, rng = new Rng()): SpherePoint[] {
	const out: SpherePoint[] = [];
	for (let i = 0; i < pointNum; i++) {
		const z = 2 * rng.next() - 1;
		const phi = 2 * Math.PI * rng.next();
		const r = Math.sqrt(Math.max(0, 1 - z * z));
		out.push([r * Math.cos(phi), r * Math.sin(phi), z]);
	}
	return out;
}

/** Poisson-disk pruning of a heavily oversampled sphere, as MeshLab does it. */
function poissonSphere(pointNum: number): SpherePoint[] {
	let oversampling = 100;
	if (pointNum <= 100) oversampling = 1000;
	if (pointNum >= 10000) oversampling = 50;
	if (pointNum >= 100000) oversampling = 20;

	const candidates = Platonic.pointCloudFrom(montecarloSphere(pointNum * oversampling));
	const sphereArea = 4 * Math.PI;
	const radius = 2 * Math.sqrt(sphereArea / (pointNum * 2) / Math.PI);
	const pruned = SurfaceSampling.poissonDiskPruning(candidates, radius);

	const out: SpherePoint[] = [];
	for (let v = 0; v < pruned.vertSize; v++) {
		if (!pruned.isVertD(v)) out.push([pruned.vx(v), pruned.vy(v), pruned.vz(v)]);
	}
	return out;
}

/**
 * A quad on the plane that best fits the current layer's selected vertices.
 *
 * The plane itself is the least-squares one, from the smallest eigenvector of
 * the covariance; `orientation` only decides how the quad is turned within
 * that plane, and how big it is.
 */
function fitPlane(doc: MeshDocument, params: RichParameterList): CMeshO {
	const current = doc.currentMeshOrNull();
	if (current === null) throw new MLException("No mesh layer selected");
	const cm = current.cm;

	const picked: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) {
		if (!cm.isVertD(v) && (cm.vertFlags[v] & VertexFlag.SELECTED) !== 0) picked.push(v);
	}
	// Faces carry their vertices into the fit, which is how MeshLab treats a
	// face selection here.
	if (picked.length === 0) {
		const seen = new Set<number>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f) || !cm.isFaceS(f)) continue;
			for (let k = 0; k < 3; k++) seen.add(cm.fv(f, k));
		}
		picked.push(...seen);
	}
	if (picked.length === 0) throw new MLException("No selection");
	if (picked.length < 3) {
		throw new MLException("A plane needs at least 3 selected vertices to fit to");
	}

	let cx = 0;
	let cy = 0;
	let cz = 0;
	for (const v of picked) {
		cx += cm.vx(v);
		cy += cm.vy(v);
		cz += cm.vz(v);
	}
	cx /= picked.length;
	cy /= picked.length;
	cz /= picked.length;

	const cov = new Float64Array(9);
	for (const v of picked) {
		const d = [cm.vx(v) - cx, cm.vy(v) - cy, cm.vz(v) - cz];
		for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[3 * a + b] += d[a] * d[b];
	}
	const normal = smallestEigenvector(cov);

	// Two in-plane axes. Which one is "first" is what `orientation` picks.
	const orientation = params.getEnum("orientation");
	const hint: number[] =
		orientation === 2
			? [0, 1, 0] // XZ parallel
			: orientation === 3
				? [1, 0, 0] // YZ parallel
				: orientation === 4
					? [0, 0, 1] // XY parallel
					: leastAlignedAxis(normal);
	let u = cross(hint, normal);
	if (Math.hypot(u[0], u[1], u[2]) < 1e-12) u = cross(leastAlignedAxis(normal), normal);
	u = unit(u);
	const w = unit(cross(normal, u));

	// Size the quad to cover the selection along those two axes.
	let uLow = Number.POSITIVE_INFINITY;
	let uHigh = Number.NEGATIVE_INFINITY;
	let wLow = Number.POSITIVE_INFINITY;
	let wHigh = Number.NEGATIVE_INFINITY;
	for (const v of picked) {
		const d = [cm.vx(v) - cx, cm.vy(v) - cy, cm.vz(v) - cz];
		const du = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
		const dw = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
		uLow = Math.min(uLow, du);
		uHigh = Math.max(uHigh, du);
		wLow = Math.min(wLow, dw);
		wHigh = Math.max(wHigh, dw);
	}
	const extent = params.getFloat("extent");
	const uMid = (uLow + uHigh) / 2;
	const wMid = (wLow + wHigh) / 2;
	const uHalf = ((uHigh - uLow) / 2) * extent;
	const wHalf = ((wHigh - wLow) / 2) * extent;

	const steps = Math.max(1, params.getInt("subdiv"));
	const out = Platonic.pointCloudFrom([]);
	const index = (i: number, j: number) => i * (steps + 1) + j;
	Allocator.addVertices(out, (steps + 1) * (steps + 1));
	for (let i = 0; i <= steps; i++) {
		const du = uMid + uHalf * ((2 * i) / steps - 1);
		for (let j = 0; j <= steps; j++) {
			const dw = wMid + wHalf * ((2 * j) / steps - 1);
			out.setVert(
				index(i, j),
				cx + u[0] * du + w[0] * dw,
				cy + u[1] * du + w[1] * dw,
				cz + u[2] * du + w[2] * dw,
			);
		}
	}
	for (let i = 0; i < steps; i++) {
		for (let j = 0; j < steps; j++) {
			Allocator.addFace(out, index(i, j), index(i + 1, j), index(i + 1, j + 1));
			Allocator.addFace(out, index(i, j), index(i + 1, j + 1), index(i, j + 1));
		}
	}
	return out;
}

/** The axis the normal leans on least, so the cross product stays well conditioned. */
function leastAlignedAxis(n: readonly number[]): number[] {
	const abs = [Math.abs(n[0]), Math.abs(n[1]), Math.abs(n[2])];
	const axis = abs.indexOf(Math.min(...abs));
	return [axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0];
}

const cross = (a: readonly number[], b: readonly number[]): number[] => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0],
];

function unit(v: readonly number[]): number[] {
	const len = Math.hypot(v[0], v[1], v[2]);
	return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 1];
}

/** Cyclic Jacobi on a 3x3 symmetric matrix; returns the least-variance direction. */
function smallestEigenvector(a: Float64Array): number[] {
	const m = Float64Array.from(a);
	const v = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
	for (let sweep = 0; sweep < 24; sweep++) {
		let off = 0;
		for (let p = 0; p < 3; p++) {
			for (let q = p + 1; q < 3; q++) off += m[3 * p + q] * m[3 * p + q];
		}
		if (off < 1e-30) break;
		for (let p = 0; p < 2; p++) {
			for (let q = p + 1; q < 3; q++) {
				const apq = m[3 * p + q];
				if (Math.abs(apq) < 1e-300) continue;
				const theta = (m[3 * q + q] - m[3 * p + p]) / (2 * apq);
				const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
				const c = 1 / Math.sqrt(t * t + 1);
				const s = t * c;
				for (let k = 0; k < 3; k++) {
					const mkp = m[3 * k + p];
					const mkq = m[3 * k + q];
					m[3 * k + p] = c * mkp - s * mkq;
					m[3 * k + q] = s * mkp + c * mkq;
				}
				for (let k = 0; k < 3; k++) {
					const mpk = m[3 * p + k];
					const mqk = m[3 * q + k];
					m[3 * p + k] = c * mpk - s * mqk;
					m[3 * q + k] = s * mpk + c * mqk;
				}
				for (let k = 0; k < 3; k++) {
					const vkp = v[3 * k + p];
					const vkq = v[3 * k + q];
					v[3 * k + p] = c * vkp - s * vkq;
					v[3 * k + q] = s * vkp + c * vkq;
				}
			}
		}
	}
	let best = 0;
	for (let i = 1; i < 3; i++) if (m[3 * i + i] < m[3 * best + best]) best = i;
	return unit([v[best], v[3 + best], v[6 + best]]);
}
