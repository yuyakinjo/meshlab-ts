/**
 * The `MeshModel::MeshElement` bitmask — MeshLab's contract for "which optional
 * mesh attributes are currently live".
 *
 * Values are copied from `src/common/ml_document/mesh_model.h` and must not
 * drift: filters declare their `getRequirements()` / `getPreConditions()` /
 * `postCondition()` in terms of these bits, and MeshLab project files persist
 * them.
 *
 * Note the two top-bit entries exceed the signed 32-bit range. JavaScript's
 * bitwise operators work on int32, so `MM_ALL & x` yields a negative number
 * even though the bit test is correct. Use {@link maskHas} and {@link maskOf}
 * rather than raw operators when the *value* of a mask matters.
 */
export const MeshElement = {
	MM_NONE: 0x00000000,

	MM_VERTCOORD: 0x00000001,
	MM_VERTNORMAL: 0x00000002,
	MM_VERTFLAG: 0x00000004,
	MM_VERTCOLOR: 0x00000008,
	MM_VERTQUALITY: 0x00000010,
	MM_VERTMARK: 0x00000020,
	MM_VERTFACETOPO: 0x00000040,
	MM_VERTCURV: 0x00000080,
	MM_VERTCURVDIR: 0x00000100,
	MM_VERTRADIUS: 0x00000200,
	MM_VERTTEXCOORD: 0x00000400,
	MM_VERTNUMBER: 0x00000800,

	MM_FACEVERT: 0x00001000,
	MM_FACENORMAL: 0x00002000,
	MM_FACEFLAG: 0x00004000,
	MM_FACECOLOR: 0x00008000,
	MM_FACEQUALITY: 0x00010000,
	MM_FACEMARK: 0x00020000,
	MM_FACEFACETOPO: 0x00040000,
	MM_FACENUMBER: 0x00080000,
	MM_FACECURVDIR: 0x00100000,

	MM_WEDGTEXCOORD: 0x00200000,
	MM_WEDGNORMAL: 0x00400000,
	MM_WEDGCOLOR: 0x00800000,

	MM_VERTFLAGSELECT: 0x01000000,
	MM_FACEFLAGSELECT: 0x02000000,

	// 0x04000000 is unused upstream.

	MM_CAMERA: 0x08000000,
	MM_TRANSFMATRIX: 0x10000000,
	MM_COLOR: 0x20000000,
	MM_POLYGONAL: 0x40000000,
	MM_UNKNOWN: 0x80000000,

	MM_GEOMETRY_AND_TOPOLOGY_CHANGE: 0x431e7be7,
	MM_ALL: 0xffffffff,
} as const;

export type MeshElementMask = number;

/** Bitwise AND that stays in the unsigned 32-bit domain. */
export function maskAnd(a: number, b: number): number {
	return (a & b) >>> 0;
}

/** Bitwise OR that stays in the unsigned 32-bit domain. */
export function maskOr(a: number, b: number): number {
	return (a | b) >>> 0;
}

/** `a` with every bit of `b` cleared. */
export function maskWithout(a: number, b: number): number {
	return (a & ~b) >>> 0;
}

/** True when every bit of `bits` is set in `mask`. */
export function maskHas(mask: number, bits: number): boolean {
	return maskAnd(mask, bits) === bits >>> 0;
}

/** True when `mask` and `bits` share at least one bit. */
export function maskIntersects(mask: number, bits: number): boolean {
	return maskAnd(mask, bits) !== 0;
}

/** OR of every argument, normalised to unsigned. */
export function maskOf(...bits: number[]): number {
	let out = 0;
	for (const b of bits) out = maskOr(out, b);
	return out;
}

const MASK_NAMES: ReadonlyArray<readonly [string, number]> = Object.entries(MeshElement).filter(
	([name, value]) =>
		value !== 0 &&
		name !== "MM_ALL" &&
		name !== "MM_GEOMETRY_AND_TOPOLOGY_CHANGE" &&
		// only single-bit entries
		((value >>> 0) & ((value >>> 0) - 1)) === 0,
);

/**
 * Renders a mask as `MM_VERTCOORD|MM_FACEVERT`, for error messages. Filters
 * that fail a precondition report exactly which attributes were missing, so
 * this string ends up in front of users.
 */
export function maskToString(mask: number): string {
	const m = mask >>> 0;
	if (m === 0) return "MM_NONE";
	if (m === 0xffffffff) return "MM_ALL";
	const parts = MASK_NAMES.filter(([, bit]) => maskAnd(m, bit) !== 0).map(([name]) => name);
	return parts.length > 0 ? parts.join("|") : `0x${m.toString(16)}`;
}
