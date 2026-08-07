/** `UpdatePosition` — applying a transform to a mesh's coordinates. */
import { determinant3, type Matrix44, transformPoint } from "../../math/matrix44.ts";
import { flipMesh } from "../clean.ts";
import type { CMeshO } from "../cmesho.ts";
import { UpdateBounding } from "./bounding.ts";
import { perVertexNormalizedPerFaceNormalized } from "./normal.ts";

/**
 * Applies `matrix` to every live vertex.
 *
 * A transform whose 3×3 block has negative determinant is a reflection, which
 * turns the surface inside out: the geometry mirrors but the winding does not,
 * so an outward-facing solid becomes inward-facing. The windings are reversed
 * to compensate, which is what keeps `Transform: Flip and/or swap axis` from
 * silently inverting a printable model.
 */
export function applyMatrix(m: CMeshO, matrix: Matrix44, updateNormals = true): void {
	const p = new Float64Array(3);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		transformPoint(matrix, m.vx(v), m.vy(v), m.vz(v), p);
		m.setVert(v, p[0], p[1], p[2]);
	}
	if (determinant3(matrix) < 0) flipMesh(m);
	UpdateBounding.box(m);
	if (updateNormals && m.fn > 0) perVertexNormalizedPerFaceNormalized(m);
	m.imark++;
}

export const UpdatePosition = { applyMatrix } as const;
