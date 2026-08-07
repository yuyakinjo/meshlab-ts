/**
 * The boxed values a `RichParameter` can hold — MeshLab's `Value` hierarchy as
 * a discriminated union.
 */

/** A camera shot. Only the fields MeshLab actually serialises. */
export interface ShotValue {
	/** 3×3 rotation, row-major. */
	rotation: readonly number[];
	translation: readonly [number, number, number];
	focalMm: number;
	pixelSizeMm: readonly [number, number];
	centerPx: readonly [number, number];
	viewportPx: readonly [number, number];
}

export type Value =
	| { readonly kind: "bool"; readonly value: boolean }
	| { readonly kind: "int"; readonly value: number }
	| { readonly kind: "float"; readonly value: number }
	| { readonly kind: "string"; readonly value: string }
	/** Packed 0xAABBGGRR, matching `vcg::Color4b`. */
	| { readonly kind: "color"; readonly value: number }
	| { readonly kind: "point3"; readonly value: readonly [number, number, number] }
	/** 16 numbers, column-major as in `vcg::Matrix44`. */
	| { readonly kind: "matrix44"; readonly value: readonly number[] }
	| { readonly kind: "shot"; readonly value: ShotValue };

export type ValueKind = Value["kind"];

export const boolValue = (value: boolean): Value => ({ kind: "bool", value });
export const intValue = (value: number): Value => ({ kind: "int", value });
export const floatValue = (value: number): Value => ({ kind: "float", value });
export const stringValue = (value: string): Value => ({ kind: "string", value });
export const colorValue = (value: number): Value => ({ kind: "color", value: value >>> 0 });
export const point3Value = (value: readonly [number, number, number]): Value => ({
	kind: "point3",
	value: [value[0], value[1], value[2]],
});
export const matrix44Value = (value: readonly number[]): Value => ({
	kind: "matrix44",
	value: [...value],
});
export const shotValue = (value: ShotValue): Value => ({ kind: "shot", value });

export const IDENTITY_MATRIX44: readonly number[] = [
	1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

export function valuesEqual(a: Value, b: Value): boolean {
	if (a.kind !== b.kind) return false;
	switch (a.kind) {
		case "point3":
		case "matrix44": {
			const bv = (b as typeof a).value;
			return a.value.length === bv.length && a.value.every((x, i) => x === bv[i]);
		}
		case "shot":
			return JSON.stringify(a.value) === JSON.stringify((b as typeof a).value);
		default:
			return a.value === (b as { value: unknown }).value;
	}
}

/** A plain JSON-safe rendering, for `.mlx` and for reporting. */
export function valueToPlain(v: Value): boolean | number | string | number[] | ShotValue {
	switch (v.kind) {
		case "point3":
		case "matrix44":
			return [...v.value];
		case "shot":
			return v.value;
		default:
			return v.value;
	}
}
