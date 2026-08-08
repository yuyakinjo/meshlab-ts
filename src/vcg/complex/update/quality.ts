/**
 * `vcg/complex/algorithms/update/quality.h` — the per-vertex scalar channel.
 */

import type { CMeshO } from "../cmesho.ts";
import { VertexFlag } from "../flags.ts";
import { forEachVFCorner } from "./topology.ts";

/** The vertices sharing an edge with `v`. Needs VF adjacency. */
export function vvStarVF(m: CMeshO, v: number, out: number[]): void {
	out.length = 0;
	forEachVFCorner(m, v, (f, k) => {
		for (const j of [(k + 1) % 3, (k + 2) % 3]) {
			const w = m.fv(f, j);
			if (w !== v && !out.includes(w)) out.push(w);
		}
	});
}

/**
 * Limits how fast quality may change across an edge.
 *
 * After this runs, `|q(a) − q(b)| ≤ |a − b| / gradientThr` for every edge — a
 * quality field that a distance-like reading can be taken from, however spiky
 * it started. The correction only ever *lowers* a value, so the result stays
 * under the input everywhere; upstream calls that conservative, and it is what
 * makes the fixed point unique rather than dependent on visiting order.
 *
 * The traversal is a work stack seeded at vertex 0 and re-primed at every
 * component, since a stack started in one component can never reach another.
 * Upstream seeds only the first vertex and so silently leaves every other
 * component untouched — a mesh with islands comes back unsaturated. Rather
 * than reproduce that, the loop below restarts at each unvisited vertex.
 */
export function vertexSaturate(m: CMeshO, gradientThr = 1.0): void {
	for (let v = 0; v < m.vertSize; v++) m.vertFlags[v] &= ~VertexFlag.VISITED;

	const star: number[] = [];
	const stack: number[] = [];
	for (let seed = 0; seed < m.vertSize; seed++) {
		if (m.isVertD(seed) || m.isVertV(seed)) continue;
		stack.push(seed);
		while (stack.length > 0) {
			const vc = stack.pop() as number;
			m.vertFlags[vc] |= VertexFlag.VISITED;
			vvStarVF(m, vc, star);
			for (const vi of star) {
				const qi = m.vertQuality[vi];
				const dx = m.vx(vi) - m.vx(vc);
				const dy = m.vy(vi) - m.vy(vc);
				const dz = m.vz(vi) - m.vz(vc);
				const distGeom = Math.hypot(dx, dy, dz) / gradientThr;
				if (distGeom < Math.abs(qi - m.vertQuality[vc])) {
					if (m.vertQuality[vc] > qi) {
						// The centre is the high one: lower it and re-examine it,
						// because its other neighbours were compared against the
						// old value.
						const delta = Math.min(0.00001, distGeom / 2);
						m.vertQuality[vc] = qi + distGeom - delta;
						stack.push(vc);
						break;
					}
					// The neighbour is the high one. It gets lowered when it is
					// the centre in its turn, which the un-visit below guarantees
					// will happen.
					m.vertFlags[vi] &= ~VertexFlag.VISITED;
				}
				if (!m.isVertV(vi)) {
					stack.push(vi);
					m.vertFlags[vi] |= VertexFlag.VISITED;
				}
			}
		}
	}
}

export const UpdateQuality = {
	vertexSaturate,
	vvStarVF,
} as const;
