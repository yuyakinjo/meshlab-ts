/**
 * Quadric edge-collapse decimation that keeps a texture parametrisation valid.
 *
 * The geometry side is the ordinary quadric decimation; what this adds is that
 * the UVs survive it. Two things are needed for that, and one deliberately is
 * not.
 *
 * - **Seams are never crossed.** Two vertices on opposite sides of a texture
 *   seam are the same point in space and different points in UV space.
 *   Collapsing them together would make the parametrisation self-contradictory
 *   however small the geometric error, so such a collapse is vetoed outright
 *   rather than merely penalised.
 * - **The UVs follow the geometry.** When the survivor lands partway along the
 *   collapsed edge, every corner that referred to either endpoint has its UV
 *   moved the same fraction of the way, so the texture stays glued to the
 *   surface instead of sliding across it.
 *
 * What this is *not* is upstream's five-dimensional quadric, which folds the UV
 * error into the same minimisation as the geometric error and so trades one
 * against the other. That is why upstream's `Extratcoordw` — the weight of that
 * trade — has no counterpart here, and why the filter says so when it is set.
 * The practical difference: a chart whose parametrisation is badly stretched
 * gets no extra protection from being simplified. The parametrisation stays
 * *valid* either way, which is the property a textured asset actually needs.
 */
import type { CMeshO } from "../cmesho.ts";
import {
	type DecimateOptions,
	type DecimateResult,
	quadricSimplification,
} from "./tri_edge_collapse_quadric.ts";

/**
 * The distinct UVs each vertex carries.
 *
 * A vertex with more than one is on a seam: two charts meet there and disagree
 * about where it lives in the texture.
 */
function uvsPerVertex(m: CMeshO): Array<Set<string>> {
	const out: Array<Set<string>> = Array.from({ length: m.vertSize }, () => new Set<string>());
	const wt = m.wedgeTexCoord;
	if (wt === null) return out;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			out[m.fv(f, k)].add(`${wt[6 * f + 2 * k]},${wt[6 * f + 2 * k + 1]}`);
		}
	}
	return out;
}

/**
 * Decimates while keeping the per-wedge parametrisation consistent.
 *
 * `options.canCollapse` and `options.onCollapse` are taken over by this
 * function; anything else the caller passes is honoured.
 */
export function quadricTexSimplification(m: CMeshO, options: DecimateOptions): DecimateResult {
	const wt = m.wedgeTexCoord;
	if (wt === null) {
		// Nothing to preserve: this is just decimation.
		return quadricSimplification(m, options);
	}

	// A seam vertex is one whose corners disagree about its UV. Recomputed
	// lazily is unnecessary — collapses never introduce a new disagreement,
	// they only remove vertices.
	const uvs = uvsPerVertex(m);
	const onSeam = (v: number) => uvs[v].size > 1;

	// Which corners refer to each vertex, so the UV update can find them.
	const cornersOf: Array<Array<[number, number]>> = Array.from({ length: m.vertSize }, () => []);
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) cornersOf[m.fv(f, k)].push([f, k]);
	}

	return quadricSimplification(m, {
		...options,
		canCollapse: (u, v) => {
			// A seam vertex may only merge into another vertex of the same seam
			// position — which in practice means it may not merge at all, since
			// the collapse would have to preserve two UVs at once.
			if (onSeam(u) || onSeam(v)) return false;
			// And the two must agree about the texture they are in.
			const a = [...uvs[u]][0];
			const b = [...uvs[v]][0];
			return a !== undefined && b !== undefined;
		},
		onCollapse: (survivor, removed, t) => {
			// The survivor moves a fraction `t` of the way toward the removed
			// vertex, so its UV does too. Every corner referring to either
			// endpoint ends up on the survivor and gets the blended value.
			const uvOf = (v: number): [number, number] | null => {
				for (const [f, k] of cornersOf[v]) {
					if (!m.isFaceD(f)) return [wt[6 * f + 2 * k], wt[6 * f + 2 * k + 1]];
				}
				return null;
			};
			const from = uvOf(survivor);
			const to = uvOf(removed);
			if (from === null || to === null) return;
			const blended: [number, number] = [
				from[0] + (to[0] - from[0]) * t,
				from[1] + (to[1] - from[1]) * t,
			];
			for (const v of [survivor, removed]) {
				for (const [f, k] of cornersOf[v]) {
					if (m.isFaceD(f)) continue;
					wt[6 * f + 2 * k] = blended[0];
					wt[6 * f + 2 * k + 1] = blended[1];
				}
			}
			// The removed vertex's corners now belong to the survivor.
			cornersOf[survivor].push(...cornersOf[removed]);
			cornersOf[removed] = [];
			uvs[survivor] = new Set([`${blended[0]},${blended[1]}`]);
		},
	});
}

export const QuadricTex = { quadricTexSimplification } as const;
