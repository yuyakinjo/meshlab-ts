/** `UpdateBounding` — recomputing the mesh bounding box. */
import { Box3 } from "../../space/box3.ts";
import type { CMeshO } from "../cmesho.ts";

/** Recomputes `m.bbox` from the live vertices. */
export function box(m: CMeshO): void {
	const b = Box3.empty();
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		b.add(m.vx(v), m.vy(v), m.vz(v));
	}
	m.bbox = b;
}

export const UpdateBounding = { box } as const;
