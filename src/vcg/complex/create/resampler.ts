/**
 * `vcg/complex/algorithms/create/resampler.h` — rebuilding a surface by
 * sampling its distance field on a grid and extracting a level set.
 *
 * Two callers wanted the same machinery for different reasons: the booleans
 * need a field they can combine with `min`/`max`, and `Uniform Mesh
 * Resampling` needs one it can offset. What they share is everything hard —
 * signing the distance correctly, picking a grid, running the extraction — so
 * it lives here rather than being written twice slightly differently.
 */
import { SurfaceLookup } from "../../space/index/surface_lookup.ts";
import type { CMeshO } from "../cmesho.ts";
import { UpdateBounding } from "../update/bounding.ts";
import { UpdateNormal } from "../update/normal.ts";
import { marchingTetrahedra } from "./marching.ts";

/** A grid over a box: how many samples along each axis, and where each sits. */
export interface Grid {
	readonly counts: readonly number[];
	readonly coord: (axis: number, i: number) => number;
	readonly index: (i: number, j: number, k: number) => number;
	readonly total: number;
}

/**
 * VCG's `BestDim`: the sample counts that make the cells nearest to cubes of
 * side `voxelSize`, with at least two samples on every axis.
 *
 * Two is the floor because a single sample along an axis gives the extraction
 * no cell at all to march through, and the result would silently be empty.
 */
export function gridFor(min: readonly number[], max: readonly number[], voxelSize: number): Grid {
	const counts = [0, 1, 2].map((a) => Math.max(2, Math.round((max[a] - min[a]) / voxelSize) + 1));
	const total = counts[0] * counts[1] * counts[2];
	return {
		counts,
		total,
		coord: (axis, i) => min[axis] + (i * (max[axis] - min[axis])) / (counts[axis] - 1),
		index: (i, j, k) => (k * counts[1] + j) * counts[0] + i,
	};
}

/** The mesh's box grown by `margin` on every side. */
export function paddedBox(cm: CMeshO, margin: number): { min: number[]; max: number[] } {
	UpdateBounding.box(cm);
	// An empty mesh has an inverted box; degenerating to a point around the
	// origin keeps `gridFor` producing a grid rather than NaNs.
	const min = cm.bbox.isEmpty ? [0, 0, 0] : cm.bbox.min;
	const max = cm.bbox.isEmpty ? [0, 0, 0] : cm.bbox.max;
	return {
		min: [0, 1, 2].map((a) => min[a] - margin),
		max: [0, 1, 2].map((a) => max[a] + margin),
	};
}

/**
 * A signed distance field over `grid`: negative inside, positive outside.
 *
 * The sign comes from the *interpolated vertex normal* at the closest surface
 * point — the angle-weighted pseudonormal, which is the one that stays right
 * when the closest point lands on an edge or a corner. A plain face normal is
 * ambiguous exactly there, and both callers meet surfaces along edges.
 *
 * Pass `signed: false` for an unsigned field, which is what an offset built
 * around a sheet rather than a solid needs.
 */
export function distanceField(
	cm: CMeshO,
	grid: Grid,
	options: { signed?: boolean; progress?: (percent: number) => void } = {},
): Float64Array {
	const useSign = options.signed ?? true;
	const progress = options.progress ?? (() => {});
	UpdateBounding.box(cm);
	UpdateNormal.perVertexNormalizedPerFaceNormalized(cm);
	const reach = (cm.bbox.diagonal || 1) * 4;
	const lookup = new SurfaceLookup(cm, reach);
	const { counts, coord, index } = grid;
	const out = new Float64Array(grid.total);

	for (let k = 0; k < counts[2]; k++) {
		progress((100 * k) / counts[2]);
		for (let j = 0; j < counts[1]; j++) {
			for (let i = 0; i < counts[0]; i++) {
				const x = coord(0, i);
				const y = coord(1, j);
				const z = coord(2, k);
				const hit = lookup.closest(x, y, z);
				if (hit === null) {
					// Beyond the search radius is outside, and far enough that
					// the exact value cannot matter to the extraction.
					out[index(i, j, k)] = reach;
					continue;
				}
				const point = [0, 0, 0];
				const normal = [0, 0, 0];
				for (let c = 0; c < 3; c++) {
					const v = cm.fv(hit.face, c);
					point[0] += cm.vx(v) * hit.bary[c];
					point[1] += cm.vy(v) * hit.bary[c];
					point[2] += cm.vz(v) * hit.bary[c];
					for (let axis = 0; axis < 3; axis++) {
						normal[axis] += cm.vertNormal[3 * v + axis] * hit.bary[c];
					}
				}
				const d = [x - point[0], y - point[1], z - point[2]];
				const dist = Math.hypot(d[0], d[1], d[2]);
				if (!useSign) {
					out[index(i, j, k)] = dist;
					continue;
				}
				const side = d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2];
				out[index(i, j, k)] = side < 0 ? -dist : dist;
			}
		}
	}
	return out;
}

/**
 * Extracts the `offset` level set of `field`.
 *
 * `discretize` puts every crossing at the midpoint of its edge rather than
 * interpolating, which is exactly the stair-stepped surface a layer-based
 * printer would produce — the reason MeshLab offers it at all.
 */
export function extractLevelSet(
	field: Float64Array,
	grid: Grid,
	offset = 0,
	discretize = false,
): CMeshO {
	const shifted = offset === 0 ? field : field.map((v) => v - offset);
	return marchingTetrahedra(shifted, grid.counts, grid.coord, grid.index, { discretize });
}
