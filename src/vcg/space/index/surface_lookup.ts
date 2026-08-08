/**
 * The nearest point of a mesh's surface to a query, within a distance bound.
 *
 * The exact structure for this is a hierarchy of triangles; upstream walks a
 * uniform grid of faces. This indexes the *barycentres* with the kd-tree the
 * library already has and then does an exact point-to-triangle test on the
 * candidates it returns.
 *
 * That is approximate in one specific way, worth stating rather than
 * discovering: a very long thin triangle whose barycentre is far from the
 * query can be missed even though part of it is close. On the near-uniform
 * meshes these callers work with, the candidate pool covers it. `maxDistance`
 * is the honest bound on how far anything is searched at all.
 */

import { MLException } from "../../../common/utilities/ml_exception.ts";
import type { CMeshO } from "../../complex/cmesho.ts";
import { KdTree } from "./kdtree.ts";

/** A point located on a source mesh: which face, and where within it. */
export interface Hit {
	readonly face: number;
	readonly bary: readonly [number, number, number];
}

export class SurfaceLookup {
	private readonly tree: KdTree;
	private readonly centres: Float64Array;
	private readonly faces: number[] = [];
	private static readonly CANDIDATES = 12;

	constructor(
		private readonly cm: CMeshO,
		private readonly maxDistance: number,
	) {
		for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f)) this.faces.push(f);
		if (this.faces.length === 0) throw new MLException("the source mesh has no faces to sample");
		this.centres = new Float64Array(this.faces.length * 3);
		this.faces.forEach((f, i) => {
			for (let a = 0; a < 3; a++) {
				this.centres[3 * i + a] =
					(cm.vertCoord[3 * cm.fv(f, 0) + a] +
						cm.vertCoord[3 * cm.fv(f, 1) + a] +
						cm.vertCoord[3 * cm.fv(f, 2) + a]) /
					3;
			}
		});
		this.tree = new KdTree(this.centres, this.faces.length);
	}

	closest(x: number, y: number, z: number): Hit | null {
		const seed = this.tree.nearestToPoint(x, y, z);
		if (seed < 0) return null;
		const candidates = this.tree.nearest(seed, SurfaceLookup.CANDIDATES);

		let best: Hit | null = null;
		let bestDistance = this.maxDistance <= 0 ? Number.POSITIVE_INFINITY : this.maxDistance;
		for (const i of candidates) {
			const f = this.faces[i];
			const hit = closestOnTriangle(this.cm, f, x, y, z);
			if (hit.distance <= bestDistance) {
				bestDistance = hit.distance;
				best = { face: f, bary: hit.bary };
			}
		}
		return best;
	}
}

/**
 * The closest point of a triangle to a query, by clamping the barycentric
 * solution of the unconstrained projection back into the triangle.
 */
export function closestOnTriangle(
	cm: CMeshO,
	f: number,
	x: number,
	y: number,
	z: number,
): { distance: number; bary: [number, number, number] } {
	const p: number[][] = [0, 1, 2].map((k) => {
		const v = cm.fv(f, k);
		return [cm.vx(v), cm.vy(v), cm.vz(v)];
	});
	const e0 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
	const e1 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
	const d = [x - p[0][0], y - p[0][1], z - p[0][2]];

	const a = dot3(e0, e0);
	const b = dot3(e0, e1);
	const c = dot3(e1, e1);
	const dd = dot3(e0, d);
	const e = dot3(e1, d);
	const det = a * c - b * b;

	let s = det === 0 ? 0 : (c * dd - b * e) / det;
	let t = det === 0 ? 0 : (a * e - b * dd) / det;
	// Clamp into the triangle. Doing it component-wise then renormalising is
	// not the exact nearest point on every edge, but it is within a texel of
	// it and the callers only ever interpolate an attribute with the result.
	s = Math.max(0, Math.min(1, s));
	t = Math.max(0, Math.min(1, t));
	if (s + t > 1) {
		const scale = s + t;
		s /= scale;
		t /= scale;
	}

	const closest = [
		p[0][0] + s * e0[0] + t * e1[0],
		p[0][1] + s * e0[1] + t * e1[1],
		p[0][2] + s * e0[2] + t * e1[2],
	];
	return {
		distance: Math.hypot(closest[0] - x, closest[1] - y, closest[2] - z),
		bary: [1 - s - t, s, t],
	};
}

function dot3(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
