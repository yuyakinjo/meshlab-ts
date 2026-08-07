/**
 * `FilterPlugin::FilterClass` — the category bit flags a filter reports from
 * `getClass()`. OR-able: "Merge Wedge Texture Coord" is Cleaning|Texture.
 */
export const FilterClass = {
	Generic: 0x00000,
	Selection: 0x00001,
	Cleaning: 0x00002,
	Remeshing: 0x00004,
	FaceColoring: 0x00008,
	VertexColoring: 0x00010,
	MeshColoring: 0x00020,
	MeshCreation: 0x00040,
	Smoothing: 0x00080,
	Quality: 0x00100,
	Layer: 0x00200,
	RasterLayer: 0x00400,
	Normal: 0x00800,
	Sampling: 0x01000,
	Texture: 0x02000,
	RangeMap: 0x04000,
	PointSet: 0x08000,
	Measure: 0x10000,
	Polygonal: 0x20000,
	Camera: 0x40000,
	Other: 0x80000,
} as const;

export type FilterClassName = keyof typeof FilterClass;
export type FilterClassMask = number;

const ENTRIES = Object.entries(FilterClass) as ReadonlyArray<[FilterClassName, number]>;

/** Parses `"Cleaning"` or `"Cleaning|Texture"` into a mask. */
export function filterClassFromString(text: string): FilterClassMask {
	let mask = 0;
	for (const part of text.split("|")) {
		const name = part.trim() as FilterClassName;
		const bit = FilterClass[name];
		if (bit === undefined) throw new Error(`unknown FilterClass "${part}"`);
		mask |= bit;
	}
	return mask;
}

/** Renders a mask as `Cleaning|Texture`, or `Generic` when empty. */
export function filterClassToString(mask: FilterClassMask): string {
	if (mask === 0) return "Generic";
	const parts = ENTRIES.filter(([, bit]) => bit !== 0 && (mask & bit) !== 0).map(([name]) => name);
	return parts.length > 0 ? parts.join("|") : "Generic";
}
