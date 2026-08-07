/**
 * `FilterPlugin::FilterArity` — how many meshes a filter consumes.
 *
 * `NONE` covers the creation filters, which take no input at all;
 * `SINGLE_MESH` is the overwhelming majority; `VARIABLE` covers filters that
 * work over every layer, such as Flatten Visible Layers.
 */
export const FilterArity = {
	NONE: 0,
	SINGLE_MESH: 1,
	FIXED: 2,
	VARIABLE: 3,
	UNKNOWN_ARITY: 4,
} as const;

export type FilterArityName = keyof typeof FilterArity;
export type FilterArityValue = (typeof FilterArity)[FilterArityName];

export function filterArityFromString(text: string): FilterArityValue {
	const v = FilterArity[text.trim() as FilterArityName];
	if (v === undefined) throw new Error(`unknown FilterArity "${text}"`);
	return v;
}

export function filterArityToString(value: FilterArityValue): FilterArityName {
	const hit = (Object.entries(FilterArity) as ReadonlyArray<[FilterArityName, number]>).find(
		([, v]) => v === value,
	);
	if (hit === undefined) throw new Error(`unknown FilterArity value ${value}`);
	return hit[0];
}
