/**
 * Segment intersection in the plane, and finding all of it at once.
 *
 * The atlas work asks one question repeatedly: after moving a chart, does its
 * boundary cross anything — itself, or the boundary that was already there? A
 * chart that overlaps another in UV space is worse than a fragmented atlas,
 * because two surface patches then paint the same texels.
 *
 * The all-pairs sweep is done through a uniform grid rather than a sweep line:
 * each segment is registered in the cells its bounding box covers, and only
 * segments sharing a cell are tested. That is upstream's approach and it suits
 * the input, where the segments are mesh edges of broadly similar length.
 *
 * **One repair.** Upstream reports a pair once per grid cell the two segments
 * share, so a long pair of crossing edges is reported two or three times. The
 * callers count intersections to decide whether a merge is acceptable, so the
 * duplicates are not harmless. Pairs are deduplicated here.
 */

/** A box in the plane. */
export interface Box2 {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

export function emptyBox2(): Box2 {
	return {
		minX: Number.POSITIVE_INFINITY,
		minY: Number.POSITIVE_INFINITY,
		maxX: Number.NEGATIVE_INFINITY,
		maxY: Number.NEGATIVE_INFINITY,
	};
}

export function addToBox2(box: Box2, x: number, y: number): void {
	if (x < box.minX) box.minX = x;
	if (y < box.minY) box.minY = y;
	if (x > box.maxX) box.maxX = x;
	if (y > box.maxY) box.maxY = y;
}

export function isEmptyBox2(box: Box2): boolean {
	return box.minX > box.maxX || box.minY > box.maxY;
}

/** A segment, as its two endpoints. */
export interface Segment2 {
	readonly x0: number;
	readonly y0: number;
	readonly x1: number;
	readonly y1: number;
}

const cross = (ax: number, ay: number, bx: number, by: number): number => ax * by - ay * bx;

/**
 * Where two segments cross, or null.
 *
 * Proper crossings only: two segments that merely touch at an endpoint, or lie
 * along each other, are not reported. That is what the callers want — mesh
 * edges share endpoints constantly and it means nothing, whereas a genuine
 * crossing means the parametrization has folded.
 */
export function segmentIntersection(a: Segment2, b: Segment2): [number, number] | null {
	const rx = a.x1 - a.x0;
	const ry = a.y1 - a.y0;
	const sx = b.x1 - b.x0;
	const sy = b.y1 - b.y0;
	const denom = cross(rx, ry, sx, sy);
	if (denom === 0) return null; // parallel or collinear

	const qpx = b.x0 - a.x0;
	const qpy = b.y0 - a.y0;
	const t = cross(qpx, qpy, sx, sy) / denom;
	const u = cross(qpx, qpy, rx, ry) / denom;
	if (t < 0 || t > 1 || u < 0 || u > 1) return null;
	return [a.x0 + t * rx, a.y0 + t * ry];
}

/** True when the segment meets the box, including being wholly inside it. */
export function segmentBoxIntersection(s: Segment2, box: Box2): boolean {
	// Trivially outside on one side: the cheap rejection that does most of the
	// work, since a segment covers few of the cells its bounding box does.
	if (Math.max(s.x0, s.x1) < box.minX || Math.min(s.x0, s.x1) > box.maxX) return false;
	if (Math.max(s.y0, s.y1) < box.minY || Math.min(s.y0, s.y1) > box.maxY) return false;
	// Inside, or crossing: with the bounding boxes overlapping, the only case
	// left to exclude is a diagonal segment passing a corner. Both endpoints on
	// the same side of the box's diagonal-facing edges settles it.
	const inside = (x: number, y: number): boolean =>
		x >= box.minX && x <= box.maxX && y >= box.minY && y <= box.maxY;
	if (inside(s.x0, s.y0) || inside(s.x1, s.y1)) return true;
	const corners: Array<[number, number]> = [
		[box.minX, box.minY],
		[box.maxX, box.minY],
		[box.maxX, box.maxY],
		[box.minX, box.maxY],
	];
	for (let i = 0; i < 4; i++) {
		const [cx0, cy0] = corners[i];
		const [cx1, cy1] = corners[(i + 1) % 4];
		if (segmentIntersection(s, { x0: cx0, y0: cy0, x1: cx1, y1: cy1 }) !== null) return true;
	}
	return false;
}

/** Whether two segments share an endpoint, exactly. */
function shareEndpoint(a: Segment2, b: Segment2): boolean {
	return (
		(a.x0 === b.x0 && a.y0 === b.y0) ||
		(a.x0 === b.x1 && a.y0 === b.y1) ||
		(a.x1 === b.x0 && a.y1 === b.y0) ||
		(a.x1 === b.x1 && a.y1 === b.y1)
	);
}

/**
 * A uniform grid over a set of segments, sized so that cells hold a few
 * segments each.
 */
class SegmentGrid {
	private readonly cells = new Map<number, number[]>();
	private readonly box: Box2;
	private readonly nx: number;
	private readonly ny: number;
	private readonly cellW: number;
	private readonly cellH: number;

	constructor(box: Box2, count: number) {
		this.box = box;
		const width = Math.max(box.maxX - box.minX, Number.MIN_VALUE);
		const height = Math.max(box.maxY - box.minY, Number.MIN_VALUE);
		// Aim at roughly one segment per cell while keeping the aspect ratio of
		// the data, so a long thin chart does not get one enormous row of cells.
		const target = Math.max(1, Math.ceil(Math.sqrt(Math.max(1, count))));
		const ratio = width / height;
		this.nx = Math.max(1, Math.min(1024, Math.round(target * Math.sqrt(ratio)) || 1));
		this.ny = Math.max(1, Math.min(1024, Math.round(target / Math.sqrt(ratio)) || 1));
		this.cellW = width / this.nx;
		this.cellH = height / this.ny;
	}

	private key(ix: number, iy: number): number {
		return iy * (this.nx + 1) + ix;
	}

	private cellBox(ix: number, iy: number): Box2 {
		return {
			minX: this.box.minX + ix * this.cellW,
			minY: this.box.minY + iy * this.cellH,
			maxX: this.box.minX + (ix + 1) * this.cellW,
			maxY: this.box.minY + (iy + 1) * this.cellH,
		};
	}

	insert(s: Segment2, id: number): void {
		const clampX = (v: number): number => Math.max(0, Math.min(this.nx - 1, v));
		const clampY = (v: number): number => Math.max(0, Math.min(this.ny - 1, v));
		const ix0 = clampX(Math.floor((Math.min(s.x0, s.x1) - this.box.minX) / this.cellW));
		const ix1 = clampX(Math.floor((Math.max(s.x0, s.x1) - this.box.minX) / this.cellW));
		const iy0 = clampY(Math.floor((Math.min(s.y0, s.y1) - this.box.minY) / this.cellH));
		const iy1 = clampY(Math.floor((Math.max(s.y0, s.y1) - this.box.minY) / this.cellH));
		for (let ix = ix0; ix <= ix1; ix++) {
			for (let iy = iy0; iy <= iy1; iy++) {
				if (!segmentBoxIntersection(s, this.cellBox(ix, iy))) continue;
				const k = this.key(ix, iy);
				const list = this.cells.get(k);
				if (list === undefined) this.cells.set(k, [id]);
				else list.push(id);
			}
		}
	}

	buckets(): Iterable<number[]> {
		return this.cells.values();
	}
}

function boxOf(segments: readonly Segment2[]): Box2 {
	const box = emptyBox2();
	for (const s of segments) {
		addToBox2(box, s.x0, s.y0);
		addToBox2(box, s.x1, s.y1);
	}
	return box;
}

/** A crossing, as the indices of the two segments that make it. */
export type IntersectionPair = readonly [number, number];

/**
 * Every pair of segments in the list that properly cross.
 *
 * Pairs are reported once, with the lower index first.
 */
export function selfIntersections(segments: readonly Segment2[]): IntersectionPair[] {
	if (segments.length < 2) return [];
	const grid = new SegmentGrid(boxOf(segments), segments.length);
	segments.forEach((s, i) => {
		grid.insert(s, i);
	});

	const seen = new Set<number>();
	const out: IntersectionPair[] = [];
	for (const bucket of grid.buckets()) {
		for (let j = 0; j < bucket.length; j++) {
			for (let k = j + 1; k < bucket.length; k++) {
				const a = Math.min(bucket[j], bucket[k]);
				const b = Math.max(bucket[j], bucket[k]);
				const key = a * segments.length + b;
				if (seen.has(key)) continue;
				seen.add(key);
				if (shareEndpoint(segments[a], segments[b])) continue;
				if (segmentIntersection(segments[a], segments[b]) === null) continue;
				out.push([a, b]);
			}
		}
	}
	return out;
}

/**
 * Every crossing between one list of segments and another.
 *
 * Pairs come back as `[index into first, index into second]`. Crossings within
 * either list are not reported — that is {@link selfIntersections}' job.
 */
export function crossIntersections(
	first: readonly Segment2[],
	second: readonly Segment2[],
): IntersectionPair[] {
	if (first.length === 0 || second.length === 0) return [];
	const box = boxOf(first);
	const other = boxOf(second);
	addToBox2(box, other.minX, other.minY);
	addToBox2(box, other.maxX, other.maxY);

	const grid = new SegmentGrid(box, first.length + second.length);
	first.forEach((s, i) => {
		grid.insert(s, i);
	});
	second.forEach((s, i) => {
		grid.insert(s, first.length + i);
	});

	const seen = new Set<number>();
	const out: IntersectionPair[] = [];
	for (const bucket of grid.buckets()) {
		for (let j = 0; j < bucket.length; j++) {
			for (let k = j + 1; k < bucket.length; k++) {
				const lo = Math.min(bucket[j], bucket[k]);
				const hi = Math.max(bucket[j], bucket[k]);
				// One from each list, or it is not a cross intersection.
				if (lo >= first.length || hi < first.length) continue;
				const b = hi - first.length;
				const key = lo * second.length + b;
				if (seen.has(key)) continue;
				seen.add(key);
				if (shareEndpoint(first[lo], second[b])) continue;
				if (segmentIntersection(first[lo], second[b]) === null) continue;
				out.push([lo, b]);
			}
		}
	}
	return out;
}

export const Intersection2 = {
	segmentIntersection,
	segmentBoxIntersection,
	selfIntersections,
	crossIntersections,
} as const;
