/**
 * Per-simplex bit flags.
 *
 * Bit values are taken from VCGLib's `vcg/simplex/vertex/base.h` and
 * `vcg/simplex/face/base.h` so that meshes serialised with the flags intact
 * (VMI, and MeshLab's own project files) round-trip with the same numbers.
 */

export const VertexFlag = {
	DELETED: 0x0001,
	NOTREAD: 0x0002,
	NOTWRITE: 0x0004,
	MODIFIED: 0x0008,
	VISITED: 0x0010,
	SELECTED: 0x0020,
	BORDER: 0x0100,
	/** First bit an algorithm may claim for itself. */
	USER0: 0x0200,
} as const;

export const FaceFlag = {
	DELETED: 0x00000001,
	NOTREAD: 0x00000002,
	NOTWRITE: 0x00000004,
	VISITED: 0x00000010,
	SELECTED: 0x00000020,

	// Per-edge border bits. The invariant BORDER<i> === BORDER0 << i is relied
	// upon by borderBit() below and by UpdateFlags.
	BORDER0: 0x00000040,
	BORDER1: 0x00000080,
	BORDER2: 0x00000100,
	BORDER012: 0x000001c0,

	// Dominant-normal-axis cache, used by point-to-face distance.
	NORMX: 0x00000200,
	NORMY: 0x00000400,
	NORMZ: 0x00000800,

	FACEEDGESEL0: 0x00008000,
	FACEEDGESEL1: 0x00010000,
	FACEEDGESEL2: 0x00020000,
	FACEEDGESEL012: 0x00038000,

	// "Faux" edges are the diagonals introduced when a polygon is stored as a
	// fan of triangles; a quad-dominant mesh is a triangle mesh whose internal
	// edges carry this bit.
	FAUX0: 0x00040000,
	FAUX1: 0x00080000,
	FAUX2: 0x00100000,
	FAUX012: 0x001c0000,

	USER0: 0x00200000,
} as const;

/** Border bit for edge `e` (0..2) of a face. */
export function borderBit(e: number): number {
	return FaceFlag.BORDER0 << e;
}

/** Faux bit for edge `e` (0..2) of a face. */
export function fauxBit(e: number): number {
	return FaceFlag.FAUX0 << e;
}

/** Per-edge selection bit for edge `e` (0..2) of a face. */
export function faceEdgeSelBit(e: number): number {
	return FaceFlag.FACEEDGESEL0 << e;
}
