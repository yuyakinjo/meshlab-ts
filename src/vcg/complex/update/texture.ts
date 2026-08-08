/**
 * `vcg/complex/algorithms/update/texture.h` — the per-wedge UV channel.
 */
import type { CMeshO } from "../cmesho.ts";
import { forEachVFCorner, vertexFace } from "./topology.ts";

/**
 * Snaps together the UVs at a vertex that are nearly but not exactly equal.
 *
 * Every wedge meeting a vertex is compared with the ones already seen there;
 * anything closer than `threshold` in texture space is set to the value it
 * matched. The point is to remove seams that exist only because a UV was
 * written twice with different rounding — a real seam is far wider than a
 * texel, so a threshold of a texel or two separates the two cases cleanly.
 *
 * Works per vertex rather than globally, which is what keeps two genuinely
 * distinct charts that happen to pass close by in UV space from being welded.
 */
export function wedgeTexMergeClose(m: CMeshO, threshold = 1 / 65536): number {
	const wt = m.wedgeTexCoord;
	if (wt === null) return 0;
	if (m.vfHeadFace === null) vertexFace(m);
	let merged = 0;

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const cluster: Array<[number, number]> = [];
		forEachVFCorner(m, v, (f, k) => {
			const u = wt[6 * f + 2 * k];
			const w = wt[6 * f + 2 * k + 1];
			let matched = false;
			for (const [cu, cw] of cluster) {
				if (cu === u && cw === w) {
					// Already exactly this value: nothing to merge, and it must
					// not start a new cluster either.
					matched = true;
					break;
				}
				if (Math.hypot(cu - u, cw - w) < threshold) {
					wt[6 * f + 2 * k] = cu;
					wt[6 * f + 2 * k + 1] = cw;
					merged++;
					matched = true;
					break;
				}
			}
			if (!matched) cluster.push([u, w]);
		});
	}
	return merged;
}

export const UpdateTexture = { wedgeTexMergeClose } as const;
