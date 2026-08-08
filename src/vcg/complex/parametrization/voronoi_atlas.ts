/**
 * `vcg/complex/algorithms/parametrization/voronoi_atlas.h` — cutting a surface
 * into charts and parametrising each one.
 *
 * A closed surface has no parametrisation at all: you cannot flatten a sphere.
 * So it must first be cut into pieces that *are* flattenable, and the pieces
 * laid out side by side in texture space. That is what an atlas is, and the
 * quality of one is judged by two things in tension — how much each chart is
 * distorted, and how much of the texture is wasted between them.
 *
 * The cutting here is a geodesic Voronoi partition: scatter seeds over the
 * surface, give every face to its nearest seed measured *along the surface*,
 * and the regions come out roughly round and roughly equal, which is exactly
 * what parametrises well. Each region is then flattened by harmonic mapping
 * and the results packed into a grid.
 */

import { MeshElement } from "../../../common/ml_document/mesh_element.ts";
import { Allocator } from "../allocator.ts";
import { connectedComponents } from "../clean.ts";
import { CMeshO } from "../cmesho.ts";
import { dijkstraGeodesic } from "../geodesic.ts";
import { faceFace } from "../update/topology.ts";
import { parametrizeDisk } from "./harmonic.ts";

export interface VoronoiAtlasResult {
	/** How many charts the surface was cut into. */
	readonly regions: number;
	/** Regions that were not disks and fell back to a per-triangle layout. */
	readonly failed: number;
	/** How many faces those fallback regions covered. */
	readonly failedFaces: number;
}

/**
 * Cuts `m` into `regionCount` geodesic Voronoi charts and parametrises each.
 *
 * Writes per-wedge UVs. A region that is not a disk — one that wrapped around
 * and met itself, or came out in two pieces — is skipped rather than forced,
 * because a forced flattening of a non-disk is UVs that look plausible and are
 * wrong. The count of those is reported so the caller can raise the region
 * count and try again, which is the fix.
 */
export function voronoiAtlas(m: CMeshO, regionCount: number): VoronoiAtlasResult {
	if (m.fn === 0) return { regions: 0, failed: 0, failedFaces: 0 };
	if (m.ffFace === null) faceFace(m);
	m.enableChannels(MeshElement.MM_WEDGTEXCOORD);
	const wt = m.wedgeTexCoord as Float64Array;

	const live: number[] = [];
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) live.push(v);
	const wanted = Math.max(1, Math.min(regionCount, live.length));

	// Farthest-point seeding: start anywhere, then repeatedly add the vertex
	// geodesically furthest from every seed so far.
	//
	// Spreading seeds evenly through the *vertex list* instead is much cheaper
	// and much worse: the list order has nothing to do with the surface, so two
	// seeds can land next to each other and leave a region elsewhere large
	// enough to wrap around and stop being a disk. That failure got worse with
	// more regions rather than better, which is the opposite of what a user
	// raising the count expects.
	const region = new Int32Array(m.vertSize).fill(-1);
	const best = new Float64Array(m.vertSize).fill(Number.POSITIVE_INFINITY);
	const seeds: number[] = [];
	for (let i = 0; i < wanted; i++) {
		let next = live[0];
		if (i > 0) {
			let far = -1;
			for (const v of live) {
				if (best[v] > far && Number.isFinite(best[v])) {
					far = best[v];
					next = v;
				}
			}
			// Everything reachable is already covered; more seeds would be
			// arbitrary rather than better.
			if (far <= 0) break;
		}
		seeds.push(next);
		const d = dijkstraGeodesic(m, [next]);
		for (const v of live) {
			if (d[v] < best[v]) {
				best[v] = d[v];
				region[v] = i;
			}
		}
	}

	// A face belongs to whichever region owns most of its corners; ties go to
	// the lowest index, which only has to be consistent, not principled.
	const faceRegion = new Int32Array(m.faceSize).fill(-1);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const counts = new Map<number, number>();
		for (let k = 0; k < 3; k++) {
			const r = region[m.fv(f, k)];
			if (r >= 0) counts.set(r, (counts.get(r) ?? 0) + 1);
		}
		let bestRegion = -1;
		let bestCount = 0;
		for (const [r, c] of [...counts].sort((a, b) => a[0] - b[0])) {
			if (c > bestCount) {
				bestCount = c;
				bestRegion = r;
			}
		}
		faceRegion[f] = bestRegion;
	}

	// Parametrise each region on its own, as a mesh in its own right.
	const charts: Array<{ faces: number[]; uv: Float64Array; local: Map<number, number> }> = [];
	// Regions that were not disks. They still need *some* parametrisation, or
	// their faces would keep whatever was in the channel — which for a freshly
	// enabled one is zero, and a whole region collapsed onto a single texel
	// looks far more like working UVs than it is.
	const fallbacks: number[][] = [];
	let failedFaces = 0;
	for (let r = 0; r < seeds.length; r++) {
		const faces: number[] = [];
		for (let f = 0; f < m.faceSize; f++) if (faceRegion[f] === r) faces.push(f);
		if (faces.length === 0) continue;

		const local = new Map<number, number>();
		const patch = new CMeshO();
		for (const f of faces) {
			for (let k = 0; k < 3; k++) {
				const v = m.fv(f, k);
				if (local.has(v)) continue;
				const nv = Allocator.addVertices(patch, 1);
				patch.setVert(nv, m.vx(v), m.vy(v), m.vz(v));
				local.set(v, nv);
			}
		}
		const first = Allocator.addFaces(patch, faces.length);
		faces.forEach((f, i) => {
			patch.setFace(
				first + i,
				local.get(m.fv(f, 0)) as number,
				local.get(m.fv(f, 1)) as number,
				local.get(m.fv(f, 2)) as number,
			);
		});
		faceFace(patch);

		// A chart has to be one piece and a disk. Both are cheap to check and
		// the flattening is meaningless without them.
		if (connectedComponents(patch).length !== 1) {
			fallbacks.push(faces);
			failedFaces += faces.length;
			continue;
		}
		try {
			const flat = parametrizeDisk(patch);
			charts.push({ faces, uv: flat.uv, local });
		} catch {
			fallbacks.push(faces);
			failedFaces += faces.length;
		}
	}

	// Pack the charts into a square grid of equal cells. Wasteful compared with
	// a real bin packer, but every chart is already normalised to the unit
	// square by the flattening, so a grid needs no measurement and cannot
	// overlap.
	const cells = charts.length + fallbacks.length;
	const columns = Math.max(1, Math.ceil(Math.sqrt(cells)));
	const cell = 1 / columns;
	// A sliver of padding so two charts never share a texel.
	const inset = cell * 0.02;
	charts.forEach((chart, i) => {
		const ox = (i % columns) * cell + inset;
		const oy = Math.floor(i / columns) * cell + inset;
		const scale = cell - 2 * inset;
		for (const f of chart.faces) {
			for (let k = 0; k < 3; k++) {
				const nv = chart.local.get(m.fv(f, k)) as number;
				wt[6 * f + 2 * k] = ox + chart.uv[2 * nv] * scale;
				wt[6 * f + 2 * k + 1] = oy + chart.uv[2 * nv + 1] * scale;
			}
		}
	});

	// The fallback layout: each non-disk region gets a cell of its own and its
	// faces are laid out side by side as separate unit triangles. Every texel
	// is then addressed by exactly one face, which is all a bake needs — what
	// is lost is the continuity a real chart would have given.
	fallbacks.forEach((faces, i) => {
		const slot = charts.length + i;
		const ox = (slot % columns) * cell + inset;
		const oy = Math.floor(slot / columns) * cell + inset;
		const scale = cell - 2 * inset;
		const across = Math.max(1, Math.ceil(Math.sqrt(faces.length)));
		const step = scale / across;
		faces.forEach((f, j) => {
			const cx = ox + (j % across) * step;
			const cy = oy + Math.floor(j / across) * step;
			const corners: Array<[number, number]> = [
				[cx, cy],
				[cx + step * 0.95, cy],
				[cx, cy + step * 0.95],
			];
			for (let k = 0; k < 3; k++) {
				wt[6 * f + 2 * k] = corners[k][0];
				wt[6 * f + 2 * k + 1] = corners[k][1];
			}
		});
	});

	m.imark++;
	return { regions: charts.length, failed: fallbacks.length, failedFaces };
}

export const VoronoiAtlas = { voronoiAtlas } as const;
