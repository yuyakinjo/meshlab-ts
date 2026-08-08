/**
 * Screened Poisson surface reconstruction, standing in for the PoissonRecon
 * library MeshLab vendors under `filter_screened_poisson/Src`.
 *
 * The structure follows Kazhdan's method — splat the oriented samples into a
 * vector field, solve for the scalar field whose gradient matches it while
 * screening the solution toward zero at the samples, then extract the level
 * set. Two deliberate departures from the C++ original, neither of which
 * changes the parameter contract:
 *
 *  - Trilinear (degree 1) basis functions with a 7-point Laplacian, rather
 *    than the degree-2 B-spline finite elements PoissonRecon assembles. The
 *    solve is first-order accurate instead of second.
 *  - Cascadic multigrid with Gauss-Seidel relaxation, rather than
 *    PoissonRecon's conjugate-gradient solver. `cgDepth` buys extra sweeps on
 *    the coarse levels it names instead of switching solver.
 *
 * The output is therefore geometrically equivalent to MeshLab's, not
 * bit-identical. Everything the reconstruction feeds — hole closing, the
 * manifold checks, decimation — is verified on its own terms.
 *
 * Sample density lands in the output's per-vertex quality, which is where
 * MeshLab puts it too (`pm->updateDataMask(MM_VERTQUALITY)`).
 */

import { MLException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

export interface PoissonOptions {
	/** Finest octree depth: the grid runs to 2^depth cells per axis. */
	depth: number;
	/** Levels at or below this are kept complete rather than sample-driven. */
	fullDepth: number;
	/** Levels at or below this get extra relaxation sweeps. */
	cgDepth: number;
	/** Reconstruction cube size relative to the sample bounding box. */
	scale: number;
	/** Noise smoothing: widens the splat kernel, as more samples per node does. */
	samplesPerNode: number;
	/** Screening strength pulling the level set onto the samples. */
	pointWeight: number;
	/** Gauss-Seidel sweeps per level. */
	iters: number;
	/** Weight each sample by its quality. */
	confidence: boolean;
	/** Drop samples carrying no usable position or orientation. */
	preClean: boolean;
}

export const POISSON_DEFAULTS: PoissonOptions = {
	depth: 8,
	fullDepth: 5,
	cgDepth: 0,
	scale: 1.1,
	samplesPerNode: 1.5,
	pointWeight: 4,
	iters: 8,
	confidence: false,
	preClean: false,
};

/** Neighbour slots per node: -x, +x, -y, +y, -z, +z. */
const NEIGHBOURS = 6;

interface Level {
	resolution: number;
	count: number;
	gridX: Int32Array;
	gridY: Int32Array;
	gridZ: Int32Array;
	neighbour: Int32Array;
	solution: Float64Array;
	rhs: Float64Array;
	weight: Float64Array;
	screen: Float64Array;
}

interface Samples {
	count: number;
	/** Positions in the unit reconstruction cube. */
	positions: Float64Array;
	normals: Float64Array;
	confidence: Float64Array;
	/**
	 * Local sample spacing, in unit-cube units, filled by
	 * {@link estimateLocalSpacing}.
	 *
	 * This is what lets the grid be sized by the density of the region that has
	 * the most detail to offer, rather than by the average over the whole cloud.
	 * See {@link adaptiveDepth}.
	 */
	spacing: Float64Array;
}

/**
 * MeshLab's `HasGoodNormal` check, worded as it words it: without a normal at
 * every sample the vector field has nothing to point at, and one NaN poisons
 * the entire solve.
 */
const BAD_NORMAL_MESSAGE =
	"Filter requires correct per vertex normals.<br>" +
	"E.g. it is necessary that your <b>ALL</b> the input vertices have a proper, not-null normal.<br> " +
	"Try enabling the <i>pre-clean<i> option and retry.";

/**
 * Gather the live vertices of every source into the unit cube with `scale`
 * worth of padding, matching how PoissonRecon frames the reconstruction domain.
 */
function toUnitCube(
	sources: readonly CMeshO[],
	options: PoissonOptions,
): {
	samples: Samples;
	centre: [number, number, number];
	span: number;
	depth: number;
	/** The depth the mean density alone would have chosen, for calibration. */
	uniformDepth: number;
} {
	const kept: Array<{ m: CMeshO; v: number }> = [];
	let sawAny = false;
	for (const m of sources) {
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			sawAny = true;
			const nx = m.vertNormal[3 * v];
			const ny = m.vertNormal[3 * v + 1];
			const nz = m.vertNormal[3 * v + 2];
			const usable =
				Number.isFinite(m.vx(v)) &&
				Number.isFinite(m.vy(v)) &&
				Number.isFinite(m.vz(v)) &&
				Number.isFinite(nx) &&
				Number.isFinite(ny) &&
				Number.isFinite(nz) &&
				Math.hypot(nx, ny, nz) > 0;
			if (usable) kept.push({ m, v });
			else if (!options.preClean) throw new MLException(BAD_NORMAL_MESSAGE);
		}
	}
	if (!sawAny)
		throw new MLException("Screened Poisson reconstruction needs a non-empty point set.");
	if (kept.length === 0) throw new MLException(BAD_NORMAL_MESSAGE);

	let lowX = Number.POSITIVE_INFINITY;
	let lowY = Number.POSITIVE_INFINITY;
	let lowZ = Number.POSITIVE_INFINITY;
	let highX = Number.NEGATIVE_INFINITY;
	let highY = Number.NEGATIVE_INFINITY;
	let highZ = Number.NEGATIVE_INFINITY;
	for (const { m, v } of kept) {
		lowX = Math.min(lowX, m.vx(v));
		lowY = Math.min(lowY, m.vy(v));
		lowZ = Math.min(lowZ, m.vz(v));
		highX = Math.max(highX, m.vx(v));
		highY = Math.max(highY, m.vy(v));
		highZ = Math.max(highZ, m.vz(v));
	}
	const extent = Math.max(highX - lowX, highY - lowY, highZ - lowZ);
	if (!(extent > 0)) {
		throw new MLException("The point set has no extent, so there is no volume to reconstruct.");
	}
	const centre: [number, number, number] = [
		(lowX + highX) / 2,
		(lowY + highY) / 2,
		(lowZ + highZ) / 2,
	];

	// The depth the samples can support decides the grid resolution, and the
	// resolution decides how much padding `scale` actually buys — so both have
	// to be settled before the positions can be placed in the cube.
	//
	// The depth is chosen in two steps: the mean-density answer first, then the
	// dense-region answer measured against it. The spacing is estimated in world
	// units, before the cube mapping, so it does not depend on the depth it is
	// about to help choose.
	const uniformDepth = effectiveDepth(kept.length, options);
	const worldPositions = new Float64Array(kept.length * 3);
	for (let slot = 0; slot < kept.length; slot++) {
		const { m, v } = kept[slot];
		worldPositions[slot * 3] = m.vx(v);
		worldPositions[slot * 3 + 1] = m.vy(v);
		worldPositions[slot * 3 + 2] = m.vz(v);
	}
	const worldOrigin = centre.map((value) => value - extent / 2);
	const worldSpacing = estimateLocalSpacing(worldPositions, kept.length, worldOrigin, extent);

	const samples: Samples = {
		count: kept.length,
		positions: new Float64Array(kept.length * 3),
		normals: new Float64Array(kept.length * 3),
		confidence: new Float64Array(kept.length),
		spacing: new Float64Array(kept.length),
	};
	const depth = adaptiveDepth({ ...samples, spacing: worldSpacing }, uniformDepth, extent, options);
	const span = extent * paddedScale(options.scale, 2 ** depth);
	// Spacing follows the positions into the cube, so both are in cube units.
	for (let slot = 0; slot < kept.length; slot++) samples.spacing[slot] = worldSpacing[slot] / span;
	for (let slot = 0; slot < kept.length; slot++) {
		const { m, v } = kept[slot];
		const p = [m.vx(v), m.vy(v), m.vz(v)];
		for (let axis = 0; axis < 3; axis++) {
			samples.positions[slot * 3 + axis] = (p[axis] - centre[axis]) / span + 0.5;
			samples.normals[slot * 3 + axis] = m.vertNormal[3 * v + axis];
		}
		const size = Math.hypot(
			samples.normals[slot * 3],
			samples.normals[slot * 3 + 1],
			samples.normals[slot * 3 + 2],
		);
		for (let axis = 0; axis < 3; axis++) samples.normals[slot * 3 + axis] /= size;
		samples.confidence[slot] = options.confidence ? Math.max(0, m.vertQuality[v]) : 1;
	}
	return { samples, centre, span, depth, uniformDepth };
}

/**
 * Nodes of clearance the grid needs between the samples and the wall of the
 * reconstruction cube: the skirt `finestKeys` lays down, its dilations, and a
 * little room for the level set to close.
 */
const WALL_CLEARANCE = 6;

/** Ceiling on the widening below, so a coarse grid does not shrink the subject away. */
const MAX_SCALE = 2;

/**
 * Widen the cube when `scale` would not leave {@link WALL_CLEARANCE} nodes of
 * clearance at this resolution.
 *
 * MeshLab's default of 1.1 is 5% of the extent on each side, which is generous
 * for a sphere and not nearly enough for a shape that fills its own bounding
 * box: reconstruct a cube at resolution 64 and 5% is under three nodes, so the
 * node set runs into the wall of the cube and the surface gets clipped there
 * into a boundary rather than closing.
 *
 * The clamp at {@link MAX_SCALE} matters as much as the widening. Below about
 * resolution 24 the clearance cannot be met at any sane scale, and letting the
 * formula run free there makes a coarser depth produce a *smaller* surface
 * than the depth below it.
 */
function paddedScale(scale: number, resolution: number): number {
	const usable = Math.max(1, scale);
	// padding in nodes = resolution * (scale - 1) / (2 * scale)
	const wanted = 1 - (2 * WALL_CLEARANCE) / resolution;
	const needed = wanted > 0 ? 1 / wanted : Number.POSITIVE_INFINITY;
	// The clamp applies to the widening this function adds, never to the scale
	// the caller asked for.
	return Math.max(usable, Math.min(MAX_SCALE, needed));
}

const keyOf = (resolution: number, x: number, y: number, z: number) =>
	x + (resolution + 1) * (y + (resolution + 1) * z);

/** Nodes the samples actually reach at the finest level, plus a stencil margin. */
function finestKeys(samples: Samples, resolution: number): Set<number> {
	const keys = new Set<number>();
	for (let slot = 0; slot < samples.count; slot++) {
		const base = [0, 0, 0];
		for (let axis = 0; axis < 3; axis++) {
			base[axis] = Math.min(
				resolution - 1,
				Math.max(0, Math.floor(samples.positions[slot * 3 + axis] * resolution)),
			);
		}
		// A one-cell skirt leaves the Laplacian stencil and marching cells room.
		for (let dz = -1; dz <= 2; dz++) {
			const z = base[2] + dz;
			if (z < 0 || z > resolution) continue;
			for (let dy = -1; dy <= 2; dy++) {
				const y = base[1] + dy;
				if (y < 0 || y > resolution) continue;
				for (let dx = -1; dx <= 2; dx++) {
					const x = base[0] + dx;
					if (x < 0 || x > resolution) continue;
					keys.add(keyOf(resolution, x, y, z));
				}
			}
		}
	}
	// Marching skips any cell missing a corner, so wherever the level set
	// wanders outside this set it gets clipped into a boundary instead of
	// closing. Three dilations put the edge of the set far enough out that it
	// does not: at one, a torus came back at genus 19 rather than 1, and at
	// two it still had a 44-edge hole on the outer equator.
	let grown = keys;
	for (let pass = 0; pass < NODE_SET_DILATIONS; pass++) grown = dilate(grown, resolution);
	return grown;
}

/** Dilation passes applied to the sample-driven node set. See {@link finestKeys}. */
const NODE_SET_DILATIONS = 3;

function dilate(keys: Set<number>, resolution: number): Set<number> {
	const grown = new Set(keys);
	const stride = resolution + 1;
	for (const key of keys) {
		const x = key % stride;
		const y = Math.floor(key / stride) % stride;
		const z = Math.floor(key / (stride * stride));
		for (let dz = -1; dz <= 1; dz++) {
			const nz = z + dz;
			if (nz < 0 || nz > resolution) continue;
			for (let dy = -1; dy <= 1; dy++) {
				const ny = y + dy;
				if (ny < 0 || ny > resolution) continue;
				for (let dx = -1; dx <= 1; dx++) {
					const nx = x + dx;
					if (nx < 0 || nx > resolution) continue;
					grown.add(keyOf(resolution, nx, ny, nz));
				}
			}
		}
	}
	return grown;
}

function materialize(resolution: number, keys: Set<number>): Level {
	const count = keys.size;
	const level: Level = {
		resolution,
		count,
		gridX: new Int32Array(count),
		gridY: new Int32Array(count),
		gridZ: new Int32Array(count),
		neighbour: new Int32Array(count * NEIGHBOURS).fill(-1),
		solution: new Float64Array(count),
		rhs: new Float64Array(count),
		weight: new Float64Array(count),
		screen: new Float64Array(count),
	};
	const stride = resolution + 1;
	// Sorting keys makes slot order, and therefore Gauss-Seidel, deterministic.
	const ordered = Int32Array.from(keys).sort();
	const lookup = new Map<number, number>();
	for (let slot = 0; slot < count; slot++) {
		const key = ordered[slot];
		lookup.set(key, slot);
		level.gridX[slot] = key % stride;
		level.gridY[slot] = Math.floor(key / stride) % stride;
		level.gridZ[slot] = Math.floor(key / (stride * stride));
	}
	const OFFSETS: ReadonlyArray<readonly [number, number, number]> = [
		[-1, 0, 0],
		[1, 0, 0],
		[0, -1, 0],
		[0, 1, 0],
		[0, 0, -1],
		[0, 0, 1],
	];
	for (let slot = 0; slot < count; slot++) {
		const x = level.gridX[slot];
		const y = level.gridY[slot];
		const z = level.gridZ[slot];
		for (let direction = 0; direction < NEIGHBOURS; direction++) {
			const [dx, dy, dz] = OFFSETS[direction];
			const nx = x + dx;
			const ny = y + dy;
			const nz = z + dz;
			if (nx < 0 || ny < 0 || nz < 0 || nx > resolution || ny > resolution || nz > resolution) {
				continue;
			}
			const found = lookup.get(keyOf(resolution, nx, ny, nz));
			if (found !== undefined) level.neighbour[slot * NEIGHBOURS + direction] = found;
		}
	}
	return level;
}

/** Guard against densifying a level that would not fit in memory. */
const DENSE_LIMIT = 2_100_000;

function buildHierarchy(samples: Samples, options: PoissonOptions): Level[] {
	const perLevel: Set<number>[] = [finestKeys(samples, 2 ** options.depth)];
	for (let depth = options.depth - 1; depth >= 0; depth--) {
		const resolution = 2 ** depth;
		const coarse = new Set<number>();
		const finer = perLevel[0];
		const fineStride = 2 * resolution + 1;
		for (const key of finer) {
			const x = key % fineStride;
			const y = Math.floor(key / fineStride) % fineStride;
			const z = Math.floor(key / (fineStride * fineStride));
			coarse.add(keyOf(resolution, x >> 1, y >> 1, z >> 1));
		}
		if (depth <= options.fullDepth && (resolution + 1) ** 3 <= DENSE_LIMIT) {
			for (let z = 0; z <= resolution; z++) {
				for (let y = 0; y <= resolution; y++) {
					for (let x = 0; x <= resolution; x++) coarse.add(keyOf(resolution, x, y, z));
				}
			}
		}
		perLevel.unshift(coarse);
	}
	const levels: Level[] = [];
	for (let depth = 0; depth <= options.depth; depth++) {
		levels.push(materialize(2 ** depth, perLevel[depth]));
	}
	return levels;
}

/** Separable tent kernel, one cell wide by default and wider for more samples per node. */
function kernelRadius(samplesPerNode: number): number {
	return Math.min(3, Math.max(1, Math.round(Math.cbrt(Math.max(1, samplesPerNode)))));
}

function lookupOf(level: Level): Map<number, number> {
	const lookup = new Map<number, number>();
	for (let slot = 0; slot < level.count; slot++) {
		lookup.set(
			keyOf(level.resolution, level.gridX[slot], level.gridY[slot], level.gridZ[slot]),
			slot,
		);
	}
	return lookup;
}

/**
 * Splat the oriented samples into a vector field, then set the right-hand side
 * to its divergence and record the sample weight that screening and density need.
 */
function accumulate(level: Level, samples: Samples, options: PoissonOptions): void {
	const { resolution } = level;
	const radius = kernelRadius(options.samplesPerNode);
	const fieldX = new Float64Array(level.count);
	const fieldY = new Float64Array(level.count);
	const fieldZ = new Float64Array(level.count);
	const lookup = lookupOf(level);

	const touched: number[] = [];
	const weights: number[] = [];
	for (let sample = 0; sample < samples.count; sample++) {
		const grid = [
			samples.positions[sample * 3] * resolution,
			samples.positions[sample * 3 + 1] * resolution,
			samples.positions[sample * 3 + 2] * resolution,
		];
		const base = grid.map((value) => Math.floor(value));
		touched.length = 0;
		weights.length = 0;
		let total = 0;
		for (let dz = -radius + 1; dz <= radius; dz++) {
			const z = base[2] + dz;
			if (z < 0 || z > resolution) continue;
			const wz = Math.max(0, 1 - Math.abs(grid[2] - z) / radius);
			if (wz <= 0) continue;
			for (let dy = -radius + 1; dy <= radius; dy++) {
				const y = base[1] + dy;
				if (y < 0 || y > resolution) continue;
				const wy = Math.max(0, 1 - Math.abs(grid[1] - y) / radius);
				if (wy <= 0) continue;
				for (let dx = -radius + 1; dx <= radius; dx++) {
					const x = base[0] + dx;
					if (x < 0 || x > resolution) continue;
					const wx = Math.max(0, 1 - Math.abs(grid[0] - x) / radius);
					if (wx <= 0) continue;
					const slot = lookup.get(keyOf(resolution, x, y, z));
					if (slot === undefined) continue;
					const weight = wx * wy * wz;
					touched.push(slot);
					weights.push(weight);
					total += weight;
				}
			}
		}
		if (total <= 0) continue;
		const strength = samples.confidence[sample] / total;
		for (let entry = 0; entry < touched.length; entry++) {
			const slot = touched[entry];
			const weight = weights[entry] * strength;
			fieldX[slot] += samples.normals[sample * 3] * weight;
			fieldY[slot] += samples.normals[sample * 3 + 1] * weight;
			fieldZ[slot] += samples.normals[sample * 3 + 2] * weight;
			level.weight[slot] += weight;
		}
	}

	// Divergence by central differences, falling back to one-sided at the margin.
	const spacing = 1 / resolution;
	const fields = [fieldX, fieldY, fieldZ];
	for (let slot = 0; slot < level.count; slot++) {
		let divergence = 0;
		for (let axis = 0; axis < 3; axis++) {
			const before = level.neighbour[slot * NEIGHBOURS + axis * 2];
			const after = level.neighbour[slot * NEIGHBOURS + axis * 2 + 1];
			const field = fields[axis];
			if (before >= 0 && after >= 0) divergence += (field[after] - field[before]) / (2 * spacing);
			else if (after >= 0) divergence += (field[after] - field[slot]) / spacing;
			else if (before >= 0) divergence += (field[slot] - field[before]) / spacing;
		}
		level.rhs[slot] = divergence;
	}

	// Screening is expressed relative to the mean sample weight so pointWeight
	// stays comparable to the Laplacian diagonal at any resolution or density.
	let occupied = 0;
	let totalWeight = 0;
	for (let slot = 0; slot < level.count; slot++) {
		if (level.weight[slot] > 0) {
			occupied++;
			totalWeight += level.weight[slot];
		}
	}
	const mean = occupied > 0 ? totalWeight / occupied : 1;
	for (let slot = 0; slot < level.count; slot++) {
		level.screen[slot] = mean > 0 ? (options.pointWeight * level.weight[slot]) / mean : 0;
	}
}

/** Gauss-Seidel sweeps over the screened Poisson system. */
function relax(level: Level, sweeps: number): void {
	const spacing = 1 / level.resolution;
	const squared = spacing * spacing;
	for (let sweep = 0; sweep < sweeps; sweep++) {
		for (let slot = 0; slot < level.count; slot++) {
			let sum = 0;
			let degree = 0;
			for (let direction = 0; direction < NEIGHBOURS; direction++) {
				const neighbour = level.neighbour[slot * NEIGHBOURS + direction];
				if (neighbour < 0) continue;
				sum += level.solution[neighbour];
				degree++;
			}
			if (degree === 0) continue;
			level.solution[slot] = (sum - squared * level.rhs[slot]) / (degree + level.screen[slot]);
		}
	}
}

/** Trilinear sample of a level's solution at a continuous grid coordinate. */
function sampleLevel(level: Level, lookup: Map<number, number>, grid: number[]): number {
	const { resolution } = level;
	const base = grid.map((value) => Math.floor(value));
	const frac = grid.map((value, axis) => value - base[axis]);
	let total = 0;
	for (let dz = 0; dz < 2; dz++) {
		for (let dy = 0; dy < 2; dy++) {
			for (let dx = 0; dx < 2; dx++) {
				const x = base[0] + dx;
				const y = base[1] + dy;
				const z = base[2] + dz;
				if (x < 0 || y < 0 || z < 0 || x > resolution || y > resolution || z > resolution) {
					continue;
				}
				const slot = lookup.get(keyOf(resolution, x, y, z));
				if (slot === undefined) continue;
				const weight =
					(dx ? frac[0] : 1 - frac[0]) *
					(dy ? frac[1] : 1 - frac[1]) *
					(dz ? frac[2] : 1 - frac[2]);
				total += weight * level.solution[slot];
			}
		}
	}
	return total;
}

/** Seed a level from its parent so each solve starts near the answer. */
function prolongate(coarse: Level, fine: Level, coarseLookup: Map<number, number>): void {
	for (let slot = 0; slot < fine.count; slot++) {
		fine.solution[slot] = sampleLevel(coarse, coarseLookup, [
			fine.gridX[slot] / 2,
			fine.gridY[slot] / 2,
			fine.gridZ[slot] / 2,
		]);
	}
}

/** The 6 tetrahedra of a cube, all sharing the 0-7 diagonal so faces stay matched. */
const TETRAHEDRA: ReadonlyArray<readonly [number, number, number, number]> = [
	[0, 1, 3, 7],
	[0, 3, 2, 7],
	[0, 2, 6, 7],
	[0, 6, 4, 7],
	[0, 4, 5, 7],
	[0, 5, 1, 7],
];

/** How far past the splatted samples the level set is still trusted, in nodes. */
const EXTRACTION_MARGIN = 4;

/**
 * The nodes the level set is allowed to cross: those the samples splatted onto,
 * grown by {@link EXTRACTION_MARGIN}.
 *
 * Away from every sample the right-hand side is zero and the solution is merely
 * harmonic, so wherever it happens to cross the iso value it produces surface
 * out of nothing — stray shards in mid-air, and a sheet running into the wall
 * of the reconstruction cube where it then gets clipped into a boundary. Both
 * disappear once extraction is confined to the band the samples support.
 */
function supportedNodes(level: Level): Uint8Array {
	const supported = new Uint8Array(level.count);
	let frontier: number[] = [];
	for (let slot = 0; slot < level.count; slot++) {
		if (level.weight[slot] <= 0) continue;
		supported[slot] = 1;
		frontier.push(slot);
	}
	for (let step = 0; step < EXTRACTION_MARGIN && frontier.length > 0; step++) {
		const next: number[] = [];
		for (const slot of frontier) {
			for (let direction = 0; direction < NEIGHBOURS; direction++) {
				const neighbour = level.neighbour[slot * NEIGHBOURS + direction];
				if (neighbour < 0 || supported[neighbour]) continue;
				supported[neighbour] = 1;
				next.push(neighbour);
			}
		}
		frontier = next;
	}
	return supported;
}

/**
 * Marching tetrahedra over the finest level. Every cube uses the same diagonal,
 * so neighbouring cells agree on their shared faces and the level set comes out
 * watertight without a 256-case table.
 */
function extractSurface(
	level: Level,
	isoValue: number,
	centre: readonly [number, number, number],
	span: number,
): CMeshO {
	const { resolution } = level;
	const supported = supportedNodes(level);
	const px: number[] = [];
	const py: number[] = [];
	const pz: number[] = [];
	const density: number[] = [];
	const faces: number[] = [];
	const onEdge = new Map<number, number>();

	const cut = (a: number, b: number): number => {
		const key = a < b ? a * level.count + b : b * level.count + a;
		const existing = onEdge.get(key);
		if (existing !== undefined) return existing;
		const va = level.solution[a];
		const vb = level.solution[b];
		const gap = vb - va;
		const t = gap === 0 ? 0.5 : (isoValue - va) / gap;
		const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
		const gridA = [level.gridX[a], level.gridY[a], level.gridZ[a]];
		const gridB = [level.gridX[b], level.gridY[b], level.gridZ[b]];
		const point = [0, 0, 0];
		for (let axis = 0; axis < 3; axis++) {
			const unit = (gridA[axis] + (gridB[axis] - gridA[axis]) * clamped) / resolution;
			point[axis] = (unit - 0.5) * span + centre[axis];
		}
		const index = px.length;
		px.push(point[0]);
		py.push(point[1]);
		pz.push(point[2]);
		density.push(level.weight[a] + (level.weight[b] - level.weight[a]) * clamped);
		onEdge.set(key, index);
		return index;
	};

	const corner = new Int32Array(8);
	for (let slot = 0; slot < level.count; slot++) {
		// Walk +x, +y then +z through the neighbour table to reach all 8 corners.
		corner[0] = slot;
		corner[1] = level.neighbour[slot * NEIGHBOURS + 1];
		corner[2] = level.neighbour[slot * NEIGHBOURS + 3];
		if (corner[1] < 0 || corner[2] < 0) continue;
		corner[3] = level.neighbour[corner[1] * NEIGHBOURS + 3];
		corner[4] = level.neighbour[corner[0] * NEIGHBOURS + 5];
		corner[5] = level.neighbour[corner[1] * NEIGHBOURS + 5];
		corner[6] = level.neighbour[corner[2] * NEIGHBOURS + 5];
		if (corner[3] < 0 || corner[4] < 0 || corner[5] < 0 || corner[6] < 0) continue;
		corner[7] = level.neighbour[corner[3] * NEIGHBOURS + 5];
		if (corner[7] < 0) continue;
		let inBand = true;
		for (let k = 0; k < 8 && inBand; k++) inBand = supported[corner[k]] === 1;
		if (!inBand) continue;

		for (const tet of TETRAHEDRA) {
			const nodes = [corner[tet[0]], corner[tet[1]], corner[tet[2]], corner[tet[3]]];
			const inside: number[] = [];
			const outside: number[] = [];
			for (const node of nodes) (level.solution[node] < isoValue ? inside : outside).push(node);
			if (inside.length === 0 || inside.length === 4) continue;
			if (inside.length === 1) {
				faces.push(
					cut(inside[0], outside[0]),
					cut(inside[0], outside[1]),
					cut(inside[0], outside[2]),
				);
			} else if (outside.length === 1) {
				faces.push(
					cut(outside[0], inside[0]),
					cut(outside[0], inside[1]),
					cut(outside[0], inside[2]),
				);
			} else {
				const quad = [
					cut(inside[0], outside[0]),
					cut(inside[0], outside[1]),
					cut(inside[1], outside[1]),
					cut(inside[1], outside[0]),
				];
				faces.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
			}
		}
	}

	const out = new CMeshO();
	if (px.length === 0) return out;
	const firstVert = Allocator.addVertices(out, px.length);
	for (let i = 0; i < px.length; i++) {
		out.setVert(firstVert + i, px[i], py[i], pz[i]);
		// Density rides in the quality channel, which is where MeshLab keeps it.
		out.vertQuality[firstVert + i] = density[i];
	}
	const faceCount = faces.length / 3;
	if (faceCount > 0) {
		const firstFace = Allocator.addFaces(out, faceCount);
		for (let f = 0; f < faceCount; f++) {
			out.setFace(firstFace + f, faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]);
		}
	}
	return out;
}

/**
 * Local sample spacing, in the same units as `positions`.
 *
 * The samples lie on a *surface*, so a box of side `h` holding `k` of them
 * covers a patch of area about `h²` — the spacing is `h / sqrt(k)`, not
 * `h / cbrt(k)`. Using the volumetric exponent here underestimates the spacing
 * badly and the splats come out too narrow to overlap.
 *
 * The grid is sized so an occupied cell holds a handful of samples: for a
 * surface the occupied cell count grows as `R²`, so `R = sqrt(count / 8)`
 * averages eight per cell, enough for the count to mean something and cheap
 * enough to do in one pass.
 */
function estimateLocalSpacing(
	positions: Float64Array,
	count: number,
	origin: readonly number[],
	extent: number,
): Float64Array {
	const spacing = new Float64Array(count);
	if (count === 0) return spacing;
	if (count < 8 || !(extent > 0)) return spacing.fill(extent > 0 ? extent : 1);

	const resolution = Math.max(2, Math.min(256, Math.round(Math.sqrt(count / 8))));
	const cell = extent / resolution;
	const cellOf = (slot: number, axis: number): number =>
		Math.max(
			0,
			Math.min(resolution - 1, Math.floor((positions[slot * 3 + axis] - origin[axis]) / cell)),
		);
	const key = (x: number, y: number, z: number): number => x + resolution * (y + resolution * z);

	const buckets = new Map<number, number>();
	const cells = new Int32Array(count * 3);
	for (let slot = 0; slot < count; slot++) {
		for (let axis = 0; axis < 3; axis++) cells[slot * 3 + axis] = cellOf(slot, axis);
		const id = key(cells[slot * 3], cells[slot * 3 + 1], cells[slot * 3 + 2]);
		buckets.set(id, (buckets.get(id) ?? 0) + 1);
	}

	// A 3x3x3 block around each sample, so a sample near a cell edge is not
	// counted as sparse just for being near an edge.
	const side = 3 * cell;
	for (let slot = 0; slot < count; slot++) {
		let neighbours = 0;
		for (let dz = -1; dz <= 1; dz++) {
			const z = cells[slot * 3 + 2] + dz;
			if (z < 0 || z >= resolution) continue;
			for (let dy = -1; dy <= 1; dy++) {
				const y = cells[slot * 3 + 1] + dy;
				if (y < 0 || y >= resolution) continue;
				for (let dx = -1; dx <= 1; dx++) {
					const x = cells[slot * 3] + dx;
					if (x < 0 || x >= resolution) continue;
					neighbours += buckets.get(key(x, y, z)) ?? 0;
				}
			}
		}
		spacing[slot] = side / Math.sqrt(Math.max(1, neighbours));
	}
	return spacing;
}

/** The `fraction` quantile of a copy of `values`, without sorting in place. */
function percentileOf(values: Float64Array, fraction: number): number {
	if (values.length === 0) return 0;
	const sorted = Float64Array.from(values).sort();
	const index = Math.max(
		0,
		Math.min(sorted.length - 1, Math.round(fraction * (sorted.length - 1))),
	);
	return sorted[index];
}

/** How dense the densest part of the cloud is: a low quantile of spacing. */
const DENSE_QUANTILE = 0.1;

/**
 * The depth the *densest* region supports, rather than the one the average
 * does.
 *
 * {@link effectiveDepth} sizes the grid from the mean sample density, which is
 * the right answer only when the density is uniform. A cloud with a well-sampled
 * front and a thin back gets a grid the back can live with and the front cannot
 * use — and the front is usually the part that matters. Upstream's octree does
 * not have this problem because it subdivides locally.
 *
 * The scale factor is calibrated against {@link effectiveDepth} on the same
 * cloud rather than derived from a shape constant: whatever cell size the mean
 * density would have produced, the same *ratio* to the spacing is applied to
 * the dense quantile. So a uniform cloud comes out at exactly the depth it
 * always did, and the depth can only ever increase.
 */
function adaptiveDepth(
	samples: Samples,
	uniformDepth: number,
	extent: number,
	options: PoissonOptions,
): number {
	if (samples.count === 0 || !(extent > 0)) return uniformDepth;
	// Anchored on the root-mean-square spacing, not the median. Each sample
	// stands for an area of about `spacing²`, so the mean-density formula the
	// uniform depth came from corresponds to the RMS — and a median taken over
	// samples sits inside whichever population is *more numerous*, which is the
	// dense one, making a skewed cloud look uniform.
	let sumSquares = 0;
	for (const value of samples.spacing) sumSquares += value * value;
	const rms = Math.sqrt(sumSquares / samples.count);
	const dense = percentileOf(samples.spacing, DENSE_QUANTILE);
	if (!(rms > 0) || !(dense > 0)) return uniformDepth;

	const cellAtUniformDepth = extent / 2 ** uniformDepth;
	const ratio = cellAtUniformDepth / rms;
	const targetCell = ratio * dense;
	if (!(targetCell > 0)) return uniformDepth;

	const wanted = Math.round(Math.log2(extent / targetCell));
	// Never coarser than the uniform answer, and never past what the caller asked.
	return Math.max(uniformDepth, Math.min(options.depth, wanted));
}

/**
 * How deep the grid can usefully go for this many samples.
 *
 * MeshLab words it as "since the reconstructor adapts the octree to the
 * sampling density, the specified reconstruction depth is only an upper
 * bound" — PoissonRecon stops subdividing a node once it holds fewer than
 * `samplesPerNode` samples. This grid is uniform rather than adaptive, so the
 * same rule has to be applied globally: a resolution-R grid meets a surface in
 * roughly R² cells, so R is capped at sqrt(count / samplesPerNode).
 *
 * Skipping this is not a cosmetic loss. Run 4000 samples at the default depth
 * of 8 and the splats no longer touch each other, so the level set comes out
 * as several hundred disconnected shards instead of one closed surface.
 */
function effectiveDepth(count: number, options: PoissonOptions): number {
	const perNode = Math.max(1, options.samplesPerNode);
	const resolution = Math.sqrt(count / perNode);
	if (!(resolution >= 2)) return 1;
	return Math.max(1, Math.min(options.depth, Math.floor(Math.log2(resolution))));
}

/**
 * Reconstruct a watertight surface from one or more oriented point sets.
 *
 * Returns a fresh mesh whose per-vertex quality holds the sample density —
 * the number {@link trimByDensity} thresholds on.
 */
export function reconstructScreenedPoisson(
	sources: readonly CMeshO[],
	options: Partial<PoissonOptions> = {},
): CMeshO {
	const settings: PoissonOptions = { ...POISSON_DEFAULTS, ...options };
	if (!Number.isInteger(settings.depth) || settings.depth < 1 || settings.depth > 12) {
		throw new MLException(
			`Reconstruction depth must be an integer in 1..12, got ${settings.depth}`,
		);
	}

	const { samples, centre, span, depth } = toUnitCube(sources, settings);
	settings.depth = depth;
	settings.fullDepth = Math.max(0, Math.min(settings.fullDepth, depth));
	const levels = buildHierarchy(samples, settings);

	// Cascadic multigrid: solve the coarsest level, then carry each solution
	// down as the starting point for the next.
	for (let depth = 0; depth < levels.length; depth++) {
		const level = levels[depth];
		accumulate(level, samples, settings);
		if (depth > 0) prolongate(levels[depth - 1], level, lookupOf(levels[depth - 1]));
		relax(level, depth <= settings.cgDepth ? settings.iters * 4 : settings.iters);
	}

	// The level set sits at the average of the solution over the samples, which
	// is what screening drives toward zero.
	const finest = levels[levels.length - 1];
	const finestLookup = lookupOf(finest);
	let isoTotal = 0;
	let isoWeight = 0;
	for (let sample = 0; sample < samples.count; sample++) {
		const value = sampleLevel(finest, finestLookup, [
			samples.positions[sample * 3] * finest.resolution,
			samples.positions[sample * 3 + 1] * finest.resolution,
			samples.positions[sample * 3 + 2] * finest.resolution,
		]);
		isoTotal += value * samples.confidence[sample];
		isoWeight += samples.confidence[sample];
	}

	const mesh = extractSurface(finest, isoWeight > 0 ? isoTotal / isoWeight : 0, centre, span);
	if (mesh.fn === 0) return mesh;

	Clean.removeDegenerateFace(mesh);
	Clean.removeUnreferencedVertex(mesh);
	Allocator.compactEveryVector(mesh);
	// Marching tetrahedra winds each triangle from whichever corners fell
	// inside, so the sheet is consistent only after a propagation pass; the
	// signed volume then decides which way "outside" is.
	UpdateTopology.faceFace(mesh);
	Clean.orientCoherentlyMesh(mesh);
	Clean.flipNormalOutside(mesh);
	// Both of those rewound faces, which leaves every FF edge index pointing at
	// the wrong corner. Rebuild rather than hand back adjacency that lies.
	UpdateTopology.faceFace(mesh);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(mesh);
	UpdateBounding.box(mesh);
	return mesh;
}

/** numpy.quantile's default: linear interpolation between order statistics. */
export function quantile(values: ArrayLike<number>, fraction: number): number {
	if (values.length === 0) return Number.NaN;
	const sorted = Float64Array.from(values).sort();
	const position = fraction * (sorted.length - 1);
	const low = Math.floor(position);
	const high = Math.ceil(position);
	if (low === high) return sorted[low];
	return sorted[low] + (position - low) * (sorted[high] - sorted[low]);
}

/**
 * Drop the sparsest `percent` of the reconstruction, and every face that used
 * one of those vertices.
 *
 * Screened Poisson always closes the volume, so a scan that did not see the
 * back of an object still gets a back — invented from nothing, and marked as
 * such by a low density. This is Open3D's
 * `remove_vertices_by_mask(densities < quantile(densities, percent/100))`;
 * MeshLab leaves the same job to a quality-based selection.
 */
export function trimByDensity(m: CMeshO, percent: number): number {
	if (percent <= 0 || m.vn === 0) return 0;
	const live: number[] = [];
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) live.push(v);
	const threshold = quantile(
		live.map((v) => m.vertQuality[v]),
		percent / 100,
	);

	let dropped = 0;
	const doomed = new Uint8Array(m.vertSize);
	for (const v of live) {
		if (m.vertQuality[v] >= threshold) continue;
		doomed[v] = 1;
		dropped++;
	}
	if (dropped === 0) return 0;

	// Deleting faces opens the surface up, so any adjacency built over it stops
	// being true the moment the first one goes.
	UpdateTopology.clearFaceFace(m);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		if (doomed[m.fv(f, 0)] || doomed[m.fv(f, 1)] || doomed[m.fv(f, 2)]) {
			Allocator.deleteFace(m, f);
		}
	}
	for (let v = 0; v < m.vertSize; v++) if (doomed[v] && !m.isVertD(v)) Allocator.deleteVertex(m, v);
	return dropped;
}
