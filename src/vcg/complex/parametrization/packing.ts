/**
 * Packing charts into a texture atlas.
 *
 * After defragmentation the charts are wherever the merges left them, which is
 * overlapping and unbounded. Packing decides where each one actually goes: a
 * rotation, a translation, and the size of the image they all fit in.
 *
 * The method is the same as upstream's — rasterise each chart, then place it
 * where it rests lowest against the charts already down, trying a few rotations
 * each. What differs is what gets rasterised. VCGLib's
 * `RasterizedOutline2Packer` works from an extracted outline polygon; here the
 * chart's *triangles* are rasterised directly. It is the same occupancy, and it
 * removes a step that can fail: a chart with an interior hole, or one whose
 * boundary the merges left non-simple, has no single outline polygon, and the
 * failure mode of asking for one is a chart packed by the wrong shape.
 *
 * The other divergence is search effort. Upstream tries every permutation of
 * chart order when there are fewer than 50 charts, and 16 rotations. Here the
 * order is by descending area — the standard heuristic, and the one that makes
 * the result deterministic — with four rotations. The atlas comes out somewhat
 * larger; {@link PackResult.occupancy} reports how much, so a caller can say so
 * rather than leaving it to be discovered from a file size.
 */
import { MLException } from "../../../common/utilities/ml_exception.ts";

export interface PackOptions {
	/** The grid the packing is computed on. Larger is tighter and slower. */
	readonly resolution?: number;
	/** Empty cells kept between charts, so filtering cannot bleed across. */
	readonly gutter?: number;
	/** How many quarter-turns to try per chart. 1 means no rotation. */
	readonly rotations?: 1 | 2 | 4;
}

/** Where one chart ended up. */
export interface ChartPlacement {
	readonly chart: number;
	/** Quarter-turns applied before translating. */
	readonly rotation: 0 | 1 | 2 | 3;
	/** Scale from UV units to atlas units. */
	readonly scale: number;
	/** Translation in the final [0, 1] atlas, applied after rotation and scale. */
	readonly offsetU: number;
	readonly offsetV: number;
}

export interface PackResult {
	readonly placements: ChartPlacement[];
	/** The atlas's aspect, as the grid it was packed on. */
	readonly width: number;
	readonly height: number;
	/** Occupied cells over total cells — how much of the image is used. */
	readonly occupancy: number;
	/** Charts that could not be placed at all. */
	readonly failed: number[];
}

/** The UV triangles of one chart, ready to rasterise. */
export interface ChartGeometry {
	readonly id: number;
	/** Triangles, six numbers each: u0, v0, u1, v1, u2, v2. */
	readonly triangles: Float64Array;
}

/** A chart rasterised at a given rotation, as a per-column height profile. */
interface Mask {
	readonly width: number;
	readonly height: number;
	/** Lowest occupied row per column, or -1 for an empty column. */
	readonly bottom: Int32Array;
	/** Highest occupied row per column, or -1. */
	readonly top: Int32Array;
	readonly cells: number;
	/** Where the chart's UV origin sits in the mask, in cells. */
	readonly originU: number;
	readonly originV: number;
}

/** Rotates a UV point by `quarters` quarter-turns about the origin. */
function rotate(u: number, v: number, quarters: number): [number, number] {
	switch (quarters & 3) {
		case 1:
			return [-v, u];
		case 2:
			return [-u, -v];
		case 3:
			return [v, -u];
		default:
			return [u, v];
	}
}

/**
 * Rasterises a chart's triangles into a column profile.
 *
 * Only the lowest and highest occupied cell of each column are kept, which is
 * all the placement below needs and turns a bitmap into two small arrays. A
 * chart with a concave underside therefore packs as though it were convex from
 * below — the same approximation VCGLib's packer makes.
 */
function rasteriseChart(
	geometry: ChartGeometry,
	scale: number,
	quarters: number,
	gutter: number,
): Mask {
	let minU = Number.POSITIVE_INFINITY;
	let minV = Number.POSITIVE_INFINITY;
	let maxU = Number.NEGATIVE_INFINITY;
	let maxV = Number.NEGATIVE_INFINITY;
	const points: Array<[number, number]> = [];
	for (let i = 0; i < geometry.triangles.length; i += 2) {
		const [u, v] = rotate(geometry.triangles[i], geometry.triangles[i + 1], quarters);
		points.push([u, v]);
		if (u < minU) minU = u;
		if (v < minV) minV = v;
		if (u > maxU) maxU = u;
		if (v > maxV) maxV = v;
	}

	const width = Math.max(1, Math.ceil((maxU - minU) * scale) + 2 * gutter + 1);
	const height = Math.max(1, Math.ceil((maxV - minV) * scale) + 2 * gutter + 1);
	const bottom = new Int32Array(width).fill(-1);
	const top = new Int32Array(width).fill(-1);
	const occupied = new Uint8Array(width * height);

	const toCell = (u: number, v: number): [number, number] => [
		(u - minU) * scale + gutter,
		(v - minV) * scale + gutter,
	];

	let cells = 0;
	for (let t = 0; t < points.length; t += 3) {
		const p = [points[t], points[t + 1], points[t + 2]].map(([u, v]) => toCell(u, v));
		const area =
			(p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
		if (area === 0) continue;
		const x0 = Math.max(0, Math.floor(Math.min(p[0][0], p[1][0], p[2][0])) - gutter);
		const x1 = Math.min(width - 1, Math.ceil(Math.max(p[0][0], p[1][0], p[2][0])) + gutter);
		const y0 = Math.max(0, Math.floor(Math.min(p[0][1], p[1][1], p[2][1])) - gutter);
		const y1 = Math.min(height - 1, Math.ceil(Math.max(p[0][1], p[1][1], p[2][1])) + gutter);

		for (let y = y0; y <= y1; y++) {
			for (let x = x0; x <= x1; x++) {
				if (occupied[y * width + x] === 1) continue;
				// A cell counts as covered when its centre is inside the triangle
				// grown by the gutter — which is what keeps neighbouring charts
				// from touching, without needing a separate dilation pass.
				if (!nearTriangle(x + 0.5, y + 0.5, p, gutter)) continue;
				occupied[y * width + x] = 1;
				cells++;
				if (bottom[x] < 0 || y < bottom[x]) bottom[x] = y;
				if (y > top[x]) top[x] = y;
			}
		}
	}

	return {
		width,
		height,
		bottom,
		top,
		cells,
		originU: minU * scale - gutter,
		originV: minV * scale - gutter,
	};
}

/** True when the point is inside the triangle, or within `margin` cells of it. */
function nearTriangle(
	x: number,
	y: number,
	p: ReadonlyArray<readonly number[]>,
	margin: number,
): boolean {
	let allNonNegative = true;
	let allNonPositive = true;
	for (let i = 0; i < 3; i++) {
		const a = p[i];
		const b = p[(i + 1) % 3];
		const side = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
		if (side < 0) allNonNegative = false;
		if (side > 0) allNonPositive = false;
	}
	if (allNonNegative || allNonPositive) return true;
	if (margin <= 0) return false;
	for (let i = 0; i < 3; i++) {
		if (distanceToSegment(x, y, p[i], p[(i + 1) % 3]) <= margin) return true;
	}
	return false;
}

function distanceToSegment(
	x: number,
	y: number,
	a: readonly number[],
	b: readonly number[],
): number {
	const dx = b[0] - a[0];
	const dy = b[1] - a[1];
	const lengthSq = dx * dx + dy * dy;
	if (lengthSq === 0) return Math.hypot(x - a[0], y - a[1]);
	let t = ((x - a[0]) * dx + (y - a[1]) * dy) / lengthSq;
	t = Math.max(0, Math.min(1, t));
	return Math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy));
}

/**
 * Places every chart, growing the atlas until they all fit.
 *
 * The scale is chosen so the charts' total area fills the grid, then relaxed
 * until the packing succeeds — a first attempt that assumes perfect packing
 * always fails, and each retry gives the charts 20% more room.
 */
export function packCharts(
	charts: readonly ChartGeometry[],
	options: PackOptions = {},
): PackResult {
	if (charts.length === 0) throw new MLException("nothing to pack");
	const resolution = options.resolution ?? 512;
	const gutter = options.gutter ?? 2;
	const rotations = options.rotations ?? 4;

	// Area in UV units, to pick a starting scale.
	let totalArea = 0;
	for (const chart of charts) {
		for (let t = 0; t < chart.triangles.length; t += 6) {
			const [u0, v0, u1, v1, u2, v2] = [0, 1, 2, 3, 4, 5].map((i) => chart.triangles[t + i]);
			totalArea += Math.abs((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)) / 2;
		}
	}
	if (!(totalArea > 0)) throw new MLException("the charts have no area to pack");

	// Largest first: the standard heuristic, and it makes the result depend on
	// the charts rather than on the order they happen to arrive in.
	const order = charts
		.map((chart, index) => ({ index, area: chartArea(chart) }))
		.sort((a, b) => b.area - a.area || a.index - b.index)
		.map((entry) => entry.index);

	let scale = (resolution * 0.9) / Math.sqrt(totalArea);
	for (let attempt = 0; attempt < 12; attempt++) {
		const packed = tryPack(charts, order, resolution, scale, gutter, rotations);
		if (packed !== null) return packed;
		scale /= 1.2;
	}

	// Everything failed: report it rather than returning a silently bad atlas.
	return {
		placements: [],
		width: resolution,
		height: resolution,
		occupancy: 0,
		failed: charts.map((chart) => chart.id),
	};
}

function chartArea(chart: ChartGeometry): number {
	let area = 0;
	for (let t = 0; t < chart.triangles.length; t += 6) {
		const [u0, v0, u1, v1, u2, v2] = [0, 1, 2, 3, 4, 5].map((i) => chart.triangles[t + i]);
		area += Math.abs((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)) / 2;
	}
	return area;
}

/** One packing attempt at a fixed scale. Null when a chart did not fit. */
function tryPack(
	charts: readonly ChartGeometry[],
	order: readonly number[],
	resolution: number,
	scale: number,
	gutter: number,
	rotations: number,
): PackResult | null {
	const horizon = new Int32Array(resolution);
	const placements: ChartPlacement[] = [];
	let used = 0;
	let skyline = 0;

	for (const index of order) {
		const chart = charts[index];
		let best: { cost: number; x: number; y: number; quarters: number; mask: Mask } | null = null;

		for (let q = 0; q < rotations; q++) {
			const quarters = (q * (4 / rotations)) | 0;
			const mask = rasteriseChart(chart, scale, quarters, gutter);
			if (mask.width > resolution) continue;

			for (let x = 0; x + mask.width <= resolution; x++) {
				// Rest the chart on the horizon: the lowest offset at which no
				// column would overlap what is already placed.
				let y = 0;
				for (let i = 0; i < mask.width; i++) {
					if (mask.bottom[i] < 0) continue;
					const needed = horizon[x + i] - mask.bottom[i];
					if (needed > y) y = needed;
				}
				let reach = 0;
				for (let i = 0; i < mask.width; i++) {
					if (mask.top[i] < 0) continue;
					const columnTop = y + mask.top[i] + 1;
					if (columnTop > reach) reach = columnTop;
				}
				// The cost upstream calls LowestHorizon: keep the packing flat.
				const cost = Math.max(reach, skyline) * resolution + y;
				if (best === null || cost < best.cost) best = { cost, x, y, quarters, mask };
			}
		}

		if (best === null) return null;

		const { x, y, quarters, mask } = best;
		for (let i = 0; i < mask.width; i++) {
			if (mask.top[i] < 0) continue;
			const columnTop = y + mask.top[i] + 1;
			if (columnTop > horizon[x + i]) horizon[x + i] = columnTop;
			if (columnTop > skyline) skyline = columnTop;
		}
		if (skyline > resolution) return null;
		used += mask.cells;

		// The chart's UV origin lands at (x, y) minus where the mask put it.
		placements.push({
			chart: chart.id,
			rotation: quarters as 0 | 1 | 2 | 3,
			scale,
			offsetU: x - mask.originU,
			offsetV: y - mask.originV,
		});
	}

	const height = Math.max(1, skyline);
	return {
		placements,
		width: resolution,
		height,
		occupancy: used / (resolution * height),
		failed: [],
	};
}

/**
 * Where a chart's UV goes after packing, in the final `[0, 1]` atlas.
 *
 * The atlas is square in the output image even when the packing was not, so
 * both axes divide by the same number and a chart cannot come out stretched.
 */
export function applyPlacement(
	placement: ChartPlacement,
	result: PackResult,
	u: number,
	v: number,
): [number, number] {
	const [ru, rv] = rotate(u, v, placement.rotation);
	const side = Math.max(result.width, result.height);
	return [
		(ru * placement.scale + placement.offsetU) / side,
		(rv * placement.scale + placement.offsetV) / side,
	];
}

export const Packing = { packCharts, applyPlacement } as const;
