/**
 * `filter_voronoi` — Voronoi partitions of a surface and of the volume it
 * encloses.
 *
 * A geodesic Voronoi partition assigns every vertex to its nearest seed
 * *along the surface*, which is a different and much better partition than
 * nearest-in-space on anything that folds back on itself. Lloyd relaxation
 * then moves each seed to the centre of its own region and repeats, which
 * spreads the seeds evenly however uneven the triangulation is.
 *
 * The two scaffolding filters use the same partition to build something
 * solid: `Voronoi Scaffolding` thickens the boundaries between regions into a
 * lattice, `Create Solid Wireframe` thickens the mesh's own edges. Both go
 * through an implicit field and marching tetrahedra rather than by
 * constructing geometry directly, which is what keeps the joints watertight
 * where several struts meet.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichBool,
	RichEnum,
	RichFloat,
	RichInt,
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
import { marchingTetrahedra } from "../../vcg/complex/create/marching.ts";
import { VertexFlag } from "../../vcg/complex/flags.ts";
import { dijkstraGeodesic } from "../../vcg/complex/geodesic.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { mulberry32 } from "../../vcg/math/noise.ts";
import { colorRamp, rgba } from "../../vcg/space/color4.ts";
import { KdTree } from "../../vcg/space/index/kdtree.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export const FP = {
	VORONOI_SAMPLING: 0,
	VOLUME_SAMPLING: 1,
	VORONOI_SCAFFOLDING: 2,
	BUILD_SHELL: 3,
} as const;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly arity: FilterArityValue;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.VORONOI_SAMPLING]: {
		name: "Voronoi Sampling",
		pythonName: "generate_sampling_voronoi",
		info:
			"Compute a point sampling over a mesh and perform a Lloyd relaxation. The filter selects " +
			"the vertices of the starting mesh that correspond to the sampled points.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.VOLUME_SAMPLING]: {
		name: "Volumetric Sampling",
		pythonName: "generate_sampling_volumetric",
		info: "Compute a volumetric sampling over a watertight mesh.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.VARIABLE,
	},
	[FP.VORONOI_SCAFFOLDING]: {
		name: "Voronoi Scaffolding",
		pythonName: "generate_voronoi_scaffolding",
		info: "Compute a volumetric sampling over a watertight mesh, and build a Voronoi scaffolding from it.",
		filterClass: FilterClass.Sampling,
		arity: FilterArity.SINGLE_MESH,
	},
	[FP.BUILD_SHELL]: {
		name: "Create Solid Wireframe",
		pythonName: "generate_solid_wireframe",
		info: "Build a solid by thickening the mesh's own edges and vertices into struts and balls.",
		filterClass: FilterClass.Remeshing,
		arity: FilterArity.VARIABLE,
	},
};

export class FilterVoronoi extends FilterPlugin {
	pluginName(): string {
		return "FilterVoronoi";
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
		return this.spec(id).arity;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		let diagonal = 1;
		if (m !== undefined) {
			UpdateBounding.box(m.cm);
			diagonal = m.cm.bbox.diagonal || 1;
		}

		switch (id) {
			case FP.VORONOI_SAMPLING:
				list.add(
					new RichInt("iterNum", 10, {
						description: "Iteration",
						tooltip: "Lloyd relaxation steps.",
					}),
				);
				list.add(
					new RichInt("sampleNum", 10, { description: "Sample Num.", tooltip: "How many seeds." }),
				);
				list.add(
					new RichEnum(
						"colorStrategy",
						1,
						["None", "Seed Distance", "Border Distance", "Region Area"],
						{
							description: "Color Strategy",
							tooltip: "What the mesh is coloured by after the relaxation.",
						},
					),
				);
				list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
				break;

			case FP.VOLUME_SAMPLING:
				list.add(
					new RichInt("sampleVolNum", 200000, {
						description: "Volume Sample Num.",
						tooltip: "How many candidate points are tested inside the volume.",
					}),
				);
				list.add(
					new RichBool("poissonFiltering", true, {
						description: "Poisson Filtering",
						tooltip: "Thin the samples so none is closer than the radius below.",
					}),
				);
				list.add(
					new RichPercentage("poissonRadius", diagonal / 100, 0, diagonal, {
						description: "Poisson Radius",
						tooltip: "The minimum distance between two kept samples.",
					}),
				);
				list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
				break;

			case FP.VORONOI_SCAFFOLDING:
				list.add(new RichInt("sampleVolNum", 20000, { description: "Volume Sample Num." }));
				list.add(new RichInt("voxelRes", 50, { description: "Volume Side Resolution" }));
				list.add(
					new RichFloat("isoThr", 1, {
						description: "Width of the entity (in voxel)",
						tooltip: "How thick the struts are, in grid cells.",
					}),
				);
				list.add(new RichInt("relaxStep", 5, { description: "Lloyd Relax Step" }));
				list.add(new RichInt("randomSeed", 0, { description: "Random seed" }));
				break;

			default:
				list.add(new RichInt("voxelRes", 64, { description: "Volume Side Resolution" }));
				list.add(
					new RichPercentage("edgeCylRadius", diagonal / 100, 0, diagonal / 4, {
						description: "Edge -> Cyl. radius",
						tooltip: "The radius of the strut replacing each edge.",
					}),
				);
				list.add(
					new RichPercentage("vertSphRadius", diagonal / 100, 0, diagonal / 4, {
						description: "Vertex -> Sph. radius",
						tooltip: "The radius of the ball at each vertex. Zero leaves the joints bare.",
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
		cb: CallBackPos,
	): FilterOutput {
		const m = doc.mm();
		const cm = m.cm;
		if (cm.fn === 0) throw new MLException("the mesh has no faces");
		UpdateBounding.box(cm);

		switch (id) {
			case FP.VORONOI_SAMPLING: {
				const count = params.getInt("sampleNum");
				if (count < 1) throw new MLException(`the sample count must be at least 1, got ${count}`);
				if (count > cm.vn) {
					throw new MLException(`asked for ${count} seeds but the mesh has only ${cm.vn} vertices`);
				}
				const iterations = params.getInt("iterNum");
				const seeds = lloydRelax(cm, count, iterations, params.getInt("randomSeed"), cb);

				m.updateDataMask(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
				const { region, distance } = partition(cm, seeds);
				for (let v = 0; v < cm.vertSize; v++) cm.vertFlags[v] &= ~VertexFlag.SELECTED;
				for (const s of seeds) cm.vertFlags[s] |= VertexFlag.SELECTED;
				colourByStrategy(cm, params.getEnum("colorStrategy"), region, distance, seeds.length);

				post.mask = MeshElement.MM_NONE;
				doc.Log.log(`Relaxed ${seeds.length} seeds over ${iterations} Lloyd iterations`);
				return { seeds: seeds.length, iterations };
			}

			case FP.VOLUME_SAMPLING: {
				const samples = volumeSamples(
					cm,
					params.getInt("sampleVolNum"),
					params.getInt("randomSeed"),
					cb,
				);
				const kept = params.getBool("poissonFiltering")
					? poissonThin(samples, params.getAbsPerc("poissonRadius"))
					: samples;
				if (kept.length === 0) {
					throw new MLException(
						"no sample landed inside the mesh; it may not be watertight, or the radius may be " +
							"larger than the mesh",
					);
				}
				const cloud = cloudFrom(kept);
				const target = doc.addNewMesh("", `${m.label()} volume samples`, true, cloud);
				target.updateBoxAndNormals();
				doc.Log.log(`Kept ${kept.length} of ${samples.length / 3} interior samples`);
				return { new_mesh_id: target.id(), samples: kept.length };
			}

			case FP.VORONOI_SCAFFOLDING: {
				const resolution = params.getInt("voxelRes");
				if (resolution < 4)
					throw new MLException(`the voxel resolution must be at least 4, got ${resolution}`);
				const samples = volumeSamples(
					cm,
					params.getInt("sampleVolNum"),
					params.getInt("randomSeed"),
					cb,
				);
				if (samples.length === 0) {
					throw new MLException("no sample landed inside the mesh; it may not be watertight");
				}
				const seeds = relaxVolume(samples, params.getInt("relaxStep"));
				const scaffold = scaffoldFrom(cm, seeds, resolution, params.getFloat("isoThr"), cb);
				const target = doc.addNewMesh("", `${m.label()} scaffolding`, true, scaffold);
				target.updateBoxAndNormals();
				doc.Log.log(
					`Built a scaffolding of ${scaffold.fn} faces from ${seeds.length / 3} Voronoi seeds`,
				);
				return { new_mesh_id: target.id(), seeds: seeds.length / 3, face_number: scaffold.fn };
			}

			default: {
				const resolution = params.getInt("voxelRes");
				if (resolution < 4)
					throw new MLException(`the voxel resolution must be at least 4, got ${resolution}`);
				const shell = solidWireframe(
					cm,
					resolution,
					params.getAbsPerc("edgeCylRadius"),
					params.getAbsPerc("vertSphRadius"),
					cb,
				);
				if (shell.fn === 0) {
					throw new MLException(
						"the wireframe came out empty; the radii are probably too small for the resolution",
					);
				}
				const target = doc.addNewMesh("", `${m.label()} wireframe`, true, shell);
				target.updateBoxAndNormals();
				doc.Log.log(`Built a solid wireframe of ${shell.vn} vertices and ${shell.fn} faces`);
				return { new_mesh_id: target.id(), vertex_number: shell.vn, face_number: shell.fn };
			}
		}
	}
}

// ---- the surface partition ------------------------------------------------

/** Each vertex's nearest seed along the surface, and how far. */
function partition(
	cm: CMeshO,
	seeds: readonly number[],
): { region: Int32Array; distance: Float64Array } {
	const region = new Int32Array(cm.vertSize).fill(-1);
	const distance = new Float64Array(cm.vertSize).fill(Number.POSITIVE_INFINITY);
	// One Dijkstra per seed. A single multi-source pass would be faster, but
	// it only reports the distance, and the partition needs to know *which*
	// seed won.
	seeds.forEach((s, i) => {
		const d = dijkstraGeodesic(cm, [s]);
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v) || d[v] >= distance[v]) continue;
			distance[v] = d[v];
			region[v] = i;
		}
	});
	return { region, distance };
}

/**
 * Lloyd relaxation on the surface.
 *
 * Each round partitions the mesh by geodesic distance and moves every seed to
 * the vertex of its own region that is most central — the one whose greatest
 * distance to the rest of the region is smallest. Using the geodesic centre
 * rather than the centroid of the positions is what keeps a seed on the
 * surface instead of drifting inside it.
 */
function lloydRelax(
	cm: CMeshO,
	count: number,
	iterations: number,
	randomSeed: number,
	cb: CallBackPos,
): number[] {
	const live: number[] = [];
	for (let v = 0; v < cm.vertSize; v++) if (!cm.isVertD(v)) live.push(v);
	const random = mulberry32(randomSeed >>> 0 || 1);

	// Farthest-point initialisation: a random start, then repeatedly the
	// vertex farthest from everything chosen so far. Random seeds would need
	// many more relaxation rounds to spread out.
	const seeds = [live[Math.floor(random() * live.length)]];
	const spread = dijkstraGeodesic(cm, seeds);
	while (seeds.length < count) {
		let best = -1;
		let bestDistance = -1;
		for (const v of live) {
			if (Number.isFinite(spread[v]) && spread[v] > bestDistance) {
				bestDistance = spread[v];
				best = v;
			}
		}
		if (best < 0) break;
		seeds.push(best);
		const fresh = dijkstraGeodesic(cm, [best]);
		for (const v of live) spread[v] = Math.min(spread[v], fresh[v]);
	}

	for (let round = 0; round < iterations; round++) {
		cb((100 * round) / Math.max(1, iterations), "Relaxing the Voronoi seeds");
		const { region } = partition(cm, seeds);
		const members: number[][] = seeds.map(() => []);
		for (const v of live) if (region[v] >= 0) members[region[v]].push(v);

		let moved = 0;
		for (let i = 0; i < seeds.length; i++) {
			const centre = regionCentre(cm, members[i]);
			if (centre >= 0 && centre !== seeds[i]) {
				seeds[i] = centre;
				moved++;
			}
		}
		if (moved === 0) break;
	}
	return seeds;
}

/**
 * Lloyd's step, restricted to the mesh's vertices.
 *
 * The centroid of a region's positions is generally not on the surface, so it
 * cannot be a seed; the member nearest it can, and is. Trying to be cleverer
 * — walking to the geodesic midpoint of the region's longest path — was worse
 * in practice: it moves a seed off centre whenever the region is not roughly
 * symmetric about that path, and the covering radius grew instead of
 * shrinking.
 */
function regionCentre(cm: CMeshO, members: readonly number[]): number {
	if (members.length === 0) return -1;
	let cx = 0;
	let cy = 0;
	let cz = 0;
	for (const v of members) {
		cx += cm.vx(v);
		cy += cm.vy(v);
		cz += cm.vz(v);
	}
	cx /= members.length;
	cy /= members.length;
	cz /= members.length;

	let best = members[0];
	let bestDistance = Number.POSITIVE_INFINITY;
	for (const v of members) {
		const d = Math.hypot(cm.vx(v) - cx, cm.vy(v) - cy, cm.vz(v) - cz);
		if (d < bestDistance) {
			bestDistance = d;
			best = v;
		}
	}
	return best;
}

function colourByStrategy(
	cm: CMeshO,
	strategy: number,
	region: Int32Array,
	distance: Float64Array,
	regions: number,
): void {
	if (strategy === 0) return;
	if (strategy === 1) {
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			if (!cm.isVertD(v) && Number.isFinite(distance[v])) max = Math.max(max, distance[v]);
		}
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			cm.vertQuality[v] = Number.isFinite(distance[v]) ? distance[v] : max;
			cm.vertColor[v] = colorRamp(0, max || 1, cm.vertQuality[v]);
		}
		return;
	}
	// The remaining strategies all want one colour per region; a hue spun
	// round the wheel keeps neighbouring regions distinguishable.
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		const i = region[v];
		cm.vertQuality[v] = i;
		cm.vertColor[v] = i < 0 ? rgba(128, 128, 128) : regionColour(i, regions);
	}
}

function regionColour(index: number, regions: number): number {
	// The golden ratio step keeps successive indices far apart in hue.
	const hue = ((index * 0.618033988749895) % 1) * 6;
	const sector = Math.floor(hue) % 6;
	const f = hue - Math.floor(hue);
	const q = Math.round(255 * (1 - f));
	const t = Math.round(255 * f);
	void regions;
	switch (sector) {
		case 0:
			return rgba(255, t, 0);
		case 1:
			return rgba(q, 255, 0);
		case 2:
			return rgba(0, 255, t);
		case 3:
			return rgba(0, q, 255);
		case 4:
			return rgba(t, 0, 255);
		default:
			return rgba(255, 0, q);
	}
}

// ---- the volume -----------------------------------------------------------

/**
 * Points inside a watertight mesh, on a jittered grid.
 *
 * Inside-ness is decided by the sign of the offset from the nearest surface
 * point against the *interpolated vertex normal* there. That is the
 * angle-weighted pseudonormal test, and it is the one that stays right when
 * the closest point lands on an edge or a corner — a plain face normal is
 * ambiguous exactly there.
 */
function volumeSamples(cm: CMeshO, count: number, randomSeed: number, cb: CallBackPos): number[] {
	UpdateBounding.box(cm);
	const box = cm.bbox;
	const lookup = new SurfaceLookup(cm, box.diagonal || 1);
	const random = mulberry32(randomSeed >>> 0 || 1);
	const normals = vertexNormals(cm);

	const side = Math.max(2, Math.ceil(Math.cbrt(Math.max(1, count))));
	const out: number[] = [];
	for (let k = 0; k < side; k++) {
		if (k % 8 === 0) cb((100 * k) / side, "Sampling the volume");
		for (let j = 0; j < side; j++) {
			for (let i = 0; i < side; i++) {
				// Jittered so the samples are not a lattice, which would alias
				// against any regular structure in the mesh.
				const x = box.min[0] + ((i + random()) / side) * (box.max[0] - box.min[0]);
				const y = box.min[1] + ((j + random()) / side) * (box.max[1] - box.min[1]);
				const z = box.min[2] + ((k + random()) / side) * (box.max[2] - box.min[2]);
				if (inside(cm, lookup, normals, x, y, z)) out.push(x, y, z);
			}
		}
	}
	return out;
}

function inside(
	cm: CMeshO,
	lookup: SurfaceLookup,
	normals: Float64Array,
	x: number,
	y: number,
	z: number,
): boolean {
	const hit = lookup.closest(x, y, z);
	if (hit === null) return false;
	const point = [0, 0, 0];
	const normal = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		const v = cm.fv(hit.face, k);
		point[0] += cm.vx(v) * hit.bary[k];
		point[1] += cm.vy(v) * hit.bary[k];
		point[2] += cm.vz(v) * hit.bary[k];
		for (let a = 0; a < 3; a++) normal[a] += normals[3 * v + a] * hit.bary[k];
	}
	const dx = x - point[0];
	const dy = y - point[1];
	const dz = z - point[2];
	return dx * normal[0] + dy * normal[1] + dz * normal[2] < 0;
}

function vertexNormals(cm: CMeshO): Float64Array {
	const out = new Float64Array(cm.vertSize * 3);
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const a = cm.fv(f, 0);
		const b = cm.fv(f, 1);
		const c = cm.fv(f, 2);
		const u = [cm.vx(b) - cm.vx(a), cm.vy(b) - cm.vy(a), cm.vz(b) - cm.vz(a)];
		const w = [cm.vx(c) - cm.vx(a), cm.vy(c) - cm.vy(a), cm.vz(c) - cm.vz(a)];
		// Not normalised per face: the cross product's length is twice the
		// area, which is the weighting the pseudonormal wants.
		const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
		for (const v of [a, b, c]) for (let k = 0; k < 3; k++) out[3 * v + k] += n[k];
	}
	for (let v = 0; v < cm.vertSize; v++) {
		const length = Math.hypot(out[3 * v], out[3 * v + 1], out[3 * v + 2]) || 1;
		for (let k = 0; k < 3; k++) out[3 * v + k] /= length;
	}
	return out;
}

/** Keeps a subset no two of which are closer than the radius. */
function poissonThin(samples: readonly number[], radius: number): number[] {
	if (radius <= 0) return [...samples];
	const kept: number[] = [];
	const cell = radius;
	const grid = new Map<string, number[]>();
	const key = (x: number, y: number, z: number) =>
		`${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;

	for (let i = 0; i < samples.length; i += 3) {
		const x = samples[i];
		const y = samples[i + 1];
		const z = samples[i + 2];
		// Only the 27 neighbouring cells can hold anything within the radius.
		let clash = false;
		for (let dx = -1; dx <= 1 && !clash; dx++) {
			for (let dy = -1; dy <= 1 && !clash; dy++) {
				for (let dz = -1; dz <= 1 && !clash; dz++) {
					const bucket = grid.get(key(x + dx * cell, y + dy * cell, z + dz * cell));
					if (bucket === undefined) continue;
					for (const j of bucket) {
						if (Math.hypot(kept[j] - x, kept[j + 1] - y, kept[j + 2] - z) < radius) {
							clash = true;
							break;
						}
					}
				}
			}
		}
		if (clash) continue;
		const at = kept.length;
		kept.push(x, y, z);
		const k = key(x, y, z);
		const bucket = grid.get(k);
		if (bucket === undefined) grid.set(k, [at]);
		else bucket.push(at);
	}
	return kept;
}

/** Lloyd relaxation in space: each seed moves to the centroid of its cell. */
function relaxVolume(samples: readonly number[], steps: number): number[] {
	const seeds = [...samples];
	const count = seeds.length / 3;
	if (count < 2) return seeds;
	for (let round = 0; round < steps; round++) {
		const coords = Float64Array.from(seeds);
		const tree = new KdTree(coords, count);
		const sums = new Float64Array(count * 3);
		const counts = new Int32Array(count);
		for (let i = 0; i < samples.length; i += 3) {
			const nearest = tree.nearestToPoint(samples[i], samples[i + 1], samples[i + 2]);
			if (nearest < 0) continue;
			sums[3 * nearest] += samples[i];
			sums[3 * nearest + 1] += samples[i + 1];
			sums[3 * nearest + 2] += samples[i + 2];
			counts[nearest]++;
		}
		for (let i = 0; i < count; i++) {
			if (counts[i] === 0) continue;
			for (let k = 0; k < 3; k++) seeds[3 * i + k] = sums[3 * i + k] / counts[i];
		}
	}
	return seeds;
}

/**
 * A lattice along the boundaries between Voronoi cells.
 *
 * The field is the difference between the distances to the nearest and the
 * second-nearest seed: it is zero exactly on a cell boundary and grows into
 * the cells, so thresholding it gives struts of a controlled thickness that
 * meet correctly wherever three or more cells do. Intersecting with the
 * mesh's interior keeps the lattice inside the object.
 */
function scaffoldFrom(
	cm: CMeshO,
	seeds: readonly number[],
	resolution: number,
	widthInVoxels: number,
	cb: CallBackPos,
): CMeshO {
	UpdateBounding.box(cm);
	const box = cm.bbox;
	const lookup = new SurfaceLookup(cm, box.diagonal || 1);
	const normals = vertexNormals(cm);
	const count = seeds.length / 3;
	if (count < 2) throw new MLException("a scaffolding needs at least two seeds");
	const tree = new KdTree(Float64Array.from(seeds), count);

	const size = [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
	const step = Math.max(...size) / resolution;
	const counts = size.map((s) => Math.max(2, Math.ceil(s / step) + 3));
	const min = box.min.map((c) => c - step);
	const coord = (axis: number, i: number) => min[axis] + i * step;
	const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;
	const values = new Float64Array(counts[0] * counts[1] * counts[2]);
	const width = widthInVoxels * step;

	for (let k = 0; k < counts[2]; k++) {
		cb((100 * k) / counts[2], "Sampling the scaffolding field");
		for (let j = 0; j < counts[1]; j++) {
			for (let i = 0; i < counts[0]; i++) {
				const x = coord(0, i);
				const y = coord(1, j);
				const z = coord(2, k);
				let value = width;
				if (inside(cm, lookup, normals, x, y, z)) {
					const near = tree.nearest(tree.nearestToPoint(x, y, z), 2);
					const d: number[] = [];
					for (const s of near) {
						d.push(Math.hypot(seeds[3 * s] - x, seeds[3 * s + 1] - y, seeds[3 * s + 2] - z));
					}
					d.sort((a, b) => a - b);
					value = (d[1] ?? d[0]) - d[0] - width;
				}
				values[index(i, j, k)] = value;
			}
		}
	}
	return marchingTetrahedra(values, counts, coord, index);
}

/**
 * A solid built from the mesh's own edges and vertices.
 *
 * The field is the distance to the nearest strut minus its radius, so
 * thresholding it at zero gives a surface that already includes the joins.
 * Constructing cylinders and spheres directly would be faster, and would
 * leave a self-intersecting mess at every vertex where several meet.
 */
function solidWireframe(
	cm: CMeshO,
	resolution: number,
	edgeRadius: number,
	vertexRadius: number,
	cb: CallBackPos,
): CMeshO {
	UpdateBounding.box(cm);
	const box = cm.bbox;
	const edges: Array<[number, number]> = [];
	const seen = new Set<string>();
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			const key = a < b ? `${a},${b}` : `${b},${a}`;
			if (seen.has(key)) continue;
			seen.add(key);
			edges.push([a, b]);
		}
	}
	const radius = Math.max(edgeRadius, vertexRadius);
	if (radius <= 0) throw new MLException("both radii are zero, so there is nothing to build");

	const size = [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
	const step = Math.max(...size) / resolution;
	const pad = radius + 2 * step;
	const counts = size.map((s) => Math.max(2, Math.ceil((s + 2 * pad) / step) + 1));
	const min = box.min.map((c) => c - pad);
	const coord = (axis: number, i: number) => min[axis] + i * step;
	const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;
	const values = new Float64Array(counts[0] * counts[1] * counts[2]).fill(radius * 4);

	// Only the cells near a strut are visited: filling the whole grid from
	// every edge would be the product of two large numbers.
	const mark = (
		visit: (i: number, j: number, k: number, x: number, y: number, z: number) => void,
		lo: number[],
		hi: number[],
	) => {
		const from = lo.map((c, a) => Math.max(0, Math.floor((c - min[a]) / step)));
		const to = hi.map((c, a) => Math.min(counts[a] - 1, Math.ceil((c - min[a]) / step)));
		for (let k = from[2]; k <= to[2]; k++) {
			for (let j = from[1]; j <= to[1]; j++) {
				for (let i = from[0]; i <= to[0]; i++) {
					visit(i, j, k, coord(0, i), coord(1, j), coord(2, k));
				}
			}
		}
	};

	edges.forEach(([a, b], n) => {
		if (n % 256 === 0) cb((100 * n) / edges.length, "Thickening the edges");
		if (edgeRadius <= 0) return;
		const p = [cm.vx(a), cm.vy(a), cm.vz(a)];
		const q = [cm.vx(b), cm.vy(b), cm.vz(b)];
		mark(
			(i, j, k, x, y, z) => {
				const d = pointSegment([x, y, z], p, q) - edgeRadius;
				const at = index(i, j, k);
				if (d < values[at]) values[at] = d;
			},
			[
				Math.min(p[0], q[0]) - edgeRadius,
				Math.min(p[1], q[1]) - edgeRadius,
				Math.min(p[2], q[2]) - edgeRadius,
			],
			[
				Math.max(p[0], q[0]) + edgeRadius,
				Math.max(p[1], q[1]) + edgeRadius,
				Math.max(p[2], q[2]) + edgeRadius,
			],
		);
	});

	if (vertexRadius > 0) {
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const p = [cm.vx(v), cm.vy(v), cm.vz(v)];
			mark(
				(i, j, k, x, y, z) => {
					const d = Math.hypot(x - p[0], y - p[1], z - p[2]) - vertexRadius;
					const at = index(i, j, k);
					if (d < values[at]) values[at] = d;
				},
				p.map((c) => c - vertexRadius),
				p.map((c) => c + vertexRadius),
			);
		}
	}
	return marchingTetrahedra(values, counts, coord, index);
}

function pointSegment(p: readonly number[], a: readonly number[], b: readonly number[]): number {
	const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
	const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
	const denominator = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
	const t =
		denominator === 0
			? 0
			: Math.max(0, Math.min(1, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / denominator));
	return Math.hypot(ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]);
}

function cloudFrom(coords: readonly number[]): CMeshO {
	const cm = new CMeshO();
	const count = coords.length / 3;
	if (count === 0) return cm;
	Allocator.addVertices(cm, count);
	for (let i = 0; i < count; i++) {
		cm.setVert(i, coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
	}
	return cm;
}
