/**
 * The channel-descriptor table: the single place that knows how {@link CMeshO}
 * lays out its data.
 *
 * VCGLib expresses optional attributes as "Ocf" components — storage that only
 * exists after `EnableXxx()` is called, driven by MeshLab's `MM_*` datamask.
 * Here each attribute is one typed array plus one row in this table, and that
 * row drives allocation, growth, compaction and copying. No other module
 * branches per attribute.
 *
 * Layering note: this file imports the `MM_*` mask from `common/ml_document`,
 * which is otherwise the layer above. The datamask *is* the contract between
 * the two layers, and duplicating the bit values here would be worse than the
 * import.
 */
import {
	MeshElement,
	maskAnd,
	maskOr,
	maskWithout,
} from "../../common/ml_document/mesh_element.ts";
import type { CMeshO } from "./cmesho.ts";

export type NumArray = Float64Array | Int32Array | Uint32Array | Uint8Array;
export type NumArrayCtor =
	| Float64ArrayConstructor
	| Int32ArrayConstructor
	| Uint32ArrayConstructor
	| Uint8ArrayConstructor;

export type VertChannelKey =
	| "vertCoord"
	| "vertFlags"
	| "vertNormal"
	| "vertQuality"
	| "vertColor"
	| "vertTexCoord"
	| "vertRadius"
	| "vertMark"
	| "vertCurvDir"
	| "vfHeadFace"
	| "vfHeadIndex";

export type FaceChannelKey =
	| "faceVert"
	| "faceFlags"
	| "faceNormal"
	| "faceQuality"
	| "faceColor"
	| "faceMark"
	| "faceCurvDir"
	| "wedgeTexCoord"
	| "wedgeTexIndex"
	| "wedgeNormal"
	| "wedgeColor"
	| "ffFace"
	| "ffEdge"
	| "vfNextFace"
	| "vfNextIndex";

export type EdgeChannelKey = "edgeVert" | "edgeFlags";

export type ChannelKey = VertChannelKey | FaceChannelKey | EdgeChannelKey;
export type ChannelDomain = "vert" | "face" | "edge";

/**
 * A run-time named attribute — VCG's `PerVertexAttributeHandle`.
 *
 * Always `Float64Array`: upstream templates the handle on any type, but the
 * only two MeshLab's own filters ever ask for are a scalar and a `Point3`, and
 * both are floating point. `arity` tells them apart.
 */
export interface CustomAttribute {
	readonly name: string;
	readonly domain: ChannelDomain;
	readonly arity: number;
	data: Float64Array;
}

export interface ChannelDesc {
	readonly key: ChannelKey;
	readonly domain: ChannelDomain;
	/** Elements of the typed array per mesh element. */
	readonly arity: number;
	readonly ctor: NumArrayCtor;
	/** `MM_*` bit that governs this channel; 0 for always-present channels. */
	readonly mask: number;
	/** Value written into freshly grown slots. Defaults to 0. */
	readonly fill?: number;
	/** Optional channels are null until enabled; mandatory ones always exist. */
	readonly optional: boolean;
}

const MM = MeshElement;

export const CHANNELS: readonly ChannelDesc[] = [
	// ---- vertex, always present (CVertexO's non-Ocf components) ------------
	{
		key: "vertCoord",
		domain: "vert",
		arity: 3,
		ctor: Float64Array,
		mask: MM.MM_VERTCOORD,
		optional: false,
	},
	{
		key: "vertFlags",
		domain: "vert",
		arity: 1,
		ctor: Uint32Array,
		mask: MM.MM_VERTFLAG,
		optional: false,
	},
	{
		key: "vertNormal",
		domain: "vert",
		arity: 3,
		ctor: Float64Array,
		mask: MM.MM_VERTNORMAL,
		optional: false,
	},
	{
		key: "vertQuality",
		domain: "vert",
		arity: 1,
		ctor: Float64Array,
		mask: MM.MM_VERTQUALITY,
		optional: false,
	},
	{
		key: "vertColor",
		domain: "vert",
		arity: 1,
		ctor: Uint32Array,
		mask: MM.MM_VERTCOLOR,
		fill: 0xffffffff,
		optional: false,
	},

	// ---- vertex, optional (Ocf) -------------------------------------------
	{
		key: "vertTexCoord",
		domain: "vert",
		arity: 2,
		ctor: Float64Array,
		mask: MM.MM_VERTTEXCOORD,
		optional: true,
	},
	{
		key: "vertRadius",
		domain: "vert",
		arity: 1,
		ctor: Float64Array,
		mask: MM.MM_VERTRADIUS,
		optional: true,
	},
	{
		key: "vertMark",
		domain: "vert",
		arity: 1,
		ctor: Int32Array,
		mask: MM.MM_VERTMARK,
		optional: true,
	},
	/** principal directions d1(xyz), d2(xyz) then curvatures k1, k2 */
	{
		key: "vertCurvDir",
		domain: "vert",
		arity: 8,
		ctor: Float64Array,
		mask: MM.MM_VERTCURVDIR,
		optional: true,
	},

	// ---- face, always present ---------------------------------------------
	{
		key: "faceVert",
		domain: "face",
		arity: 3,
		ctor: Int32Array,
		mask: MM.MM_FACEVERT,
		fill: -1,
		optional: false,
	},
	{
		key: "faceFlags",
		domain: "face",
		arity: 1,
		ctor: Uint32Array,
		mask: MM.MM_FACEFLAG,
		optional: false,
	},
	{
		key: "faceNormal",
		domain: "face",
		arity: 3,
		ctor: Float64Array,
		mask: MM.MM_FACENORMAL,
		optional: false,
	},

	// ---- face, optional (Ocf) ---------------------------------------------
	{
		key: "faceQuality",
		domain: "face",
		arity: 1,
		ctor: Float64Array,
		mask: MM.MM_FACEQUALITY,
		optional: true,
	},
	{
		key: "faceColor",
		domain: "face",
		arity: 1,
		ctor: Uint32Array,
		mask: MM.MM_FACECOLOR,
		fill: 0xffffffff,
		optional: true,
	},
	{
		key: "faceMark",
		domain: "face",
		arity: 1,
		ctor: Int32Array,
		mask: MM.MM_FACEMARK,
		optional: true,
	},
	{
		key: "faceCurvDir",
		domain: "face",
		arity: 8,
		ctor: Float64Array,
		mask: MM.MM_FACECURVDIR,
		optional: true,
	},
	{
		key: "wedgeTexCoord",
		domain: "face",
		arity: 6,
		ctor: Float64Array,
		mask: MM.MM_WEDGTEXCOORD,
		optional: true,
	},
	// VCG stores the texture index inside the tex-coord object (`WT(i).N()`), so
	// it lives and dies with the coordinates and shares their datamask bit.
	{
		key: "wedgeTexIndex",
		domain: "face",
		arity: 3,
		ctor: Int32Array,
		mask: MM.MM_WEDGTEXCOORD,
		optional: true,
	},
	{
		key: "wedgeNormal",
		domain: "face",
		arity: 9,
		ctor: Float64Array,
		mask: MM.MM_WEDGNORMAL,
		optional: true,
	},
	{
		key: "wedgeColor",
		domain: "face",
		arity: 3,
		ctor: Uint32Array,
		mask: MM.MM_WEDGCOLOR,
		fill: 0xffffffff,
		optional: true,
	},

	// ---- adjacency ---------------------------------------------------------
	// FF: for corner k of face f, the opposite face and which of its edges we
	// arrive on. A border is encoded VCG-style as a self-reference:
	// ffFace[3f+k] === f && ffEdge[3f+k] === k.
	{
		key: "ffFace",
		domain: "face",
		arity: 3,
		ctor: Int32Array,
		mask: MM.MM_FACEFACETOPO,
		fill: -1,
		optional: true,
	},
	{
		key: "ffEdge",
		domain: "face",
		arity: 3,
		ctor: Uint8Array,
		mask: MM.MM_FACEFACETOPO,
		optional: true,
	},
	// ---- edge, always present ------------------------------------------------
	// An edge mesh carries only these two; a polyline has no normals, no
	// quality and no adjacency to speak of.
	//
	// Mask 0, because MeshLab's `MM_*` enum has no edge bits at all — edges are
	// not something a filter declares a requirement on, they are simply there.
	// The table already treats 0 as "always present, governed by nothing".
	{
		key: "edgeVert",
		domain: "edge",
		arity: 2,
		ctor: Int32Array,
		mask: 0,
		fill: -1,
		optional: false,
	},
	{
		key: "edgeFlags",
		domain: "edge",
		arity: 1,
		ctor: Uint32Array,
		mask: 0,
		optional: false,
	},
	// VF: an intrusive singly-linked list threading every face corner that
	// touches a vertex, exactly VCG's VFp/VFi chain. -1 terminates.
	{
		key: "vfHeadFace",
		domain: "vert",
		arity: 1,
		ctor: Int32Array,
		mask: MM.MM_VERTFACETOPO,
		fill: -1,
		optional: true,
	},
	{
		key: "vfHeadIndex",
		domain: "vert",
		arity: 1,
		ctor: Uint8Array,
		mask: MM.MM_VERTFACETOPO,
		optional: true,
	},
	{
		key: "vfNextFace",
		domain: "face",
		arity: 3,
		ctor: Int32Array,
		mask: MM.MM_VERTFACETOPO,
		fill: -1,
		optional: true,
	},
	{
		key: "vfNextIndex",
		domain: "face",
		arity: 3,
		ctor: Uint8Array,
		mask: MM.MM_VERTFACETOPO,
		optional: true,
	},
];

/** Mask of the channels that always exist on a freshly constructed CMeshO. */
export const MANDATORY_MASK: number = CHANNELS.filter((c) => !c.optional).reduce(
	(acc, c) => maskOr(acc, c.mask),
	0,
);

const CHANNELS_BY_KEY = new Map<ChannelKey, ChannelDesc>(CHANNELS.map((c) => [c.key, c]));

export function channelDesc(key: ChannelKey): ChannelDesc {
	const d = CHANNELS_BY_KEY.get(key);
	if (d === undefined) throw new Error(`unknown channel "${key}"`);
	return d;
}

// CMeshO declares each channel with its exact array type so that hot code sees
// `Float64Array`, not a union. These two helpers are the only place that treats
// the channels generically.
type ChannelStore = { [K in ChannelKey]: NumArray | null };

export function getChannel(m: CMeshO, key: ChannelKey): NumArray | null {
	return (m as unknown as ChannelStore)[key];
}

export function setChannel(m: CMeshO, key: ChannelKey, value: NumArray | null): void {
	(m as unknown as ChannelStore)[key] = value;
}

function newArray(desc: ChannelDesc, count: number): NumArray {
	const arr = new desc.ctor(count * desc.arity);
	if (desc.fill !== undefined && desc.fill !== 0) arr.fill(desc.fill);
	return arr;
}

/**
 * Allocates storage for every channel selected by `mask` that is not allocated
 * yet, and records the bits in `m.currentDataMask`.
 *
 * This is storage only. Enabling `MM_FACEFACETOPO` here gives you zeroed
 * adjacency arrays, not valid adjacency — `UpdateTopology.FaceFace` computes
 * that, and `MeshModel.updateDataMask` is the entry point that does both, which
 * is how MeshLab behaves.
 */
export function enableChannels(m: CMeshO, mask: number): void {
	if (mask === 0) return;
	for (const desc of CHANNELS) {
		if (!desc.optional) continue;
		if (maskAnd(mask, desc.mask) === 0) continue;
		if (getChannel(m, desc.key) !== null) continue;
		const cap = desc.domain === "vert" ? m.vertCap : m.faceCap;
		setChannel(m, desc.key, newArray(desc, cap));
	}
	m.currentDataMask = maskOr(m.currentDataMask, maskAnd(mask, ALL_CHANNEL_MASK));
}

/** Releases the storage of every optional channel selected by `mask`. */
export function disableChannels(m: CMeshO, mask: number): void {
	if (mask === 0) return;
	for (const desc of CHANNELS) {
		if (!desc.optional) continue;
		if (maskAnd(mask, desc.mask) === 0) continue;
		setChannel(m, desc.key, null);
	}
	m.currentDataMask = maskWithout(m.currentDataMask, maskAnd(mask, OPTIONAL_CHANNEL_MASK));
}

export const OPTIONAL_CHANNEL_MASK: number = CHANNELS.filter((c) => c.optional).reduce(
	(acc, c) => maskOr(acc, c.mask),
	0,
);

const ALL_CHANNEL_MASK: number = maskOr(MANDATORY_MASK, OPTIONAL_CHANNEL_MASK);

/** True when every channel governed by `mask` is currently allocated. */
export function hasChannels(m: CMeshO, mask: number): boolean {
	for (const desc of CHANNELS) {
		if (maskAnd(mask, desc.mask) === 0) continue;
		if (getChannel(m, desc.key) === null) return false;
	}
	return true;
}

/**
 * Reallocates every channel of `domain` to hold `newCap` elements, preserving
 * the first `liveCount` elements' data.
 *
 * Callers must not hold a reference to any channel array across this call —
 * the arrays are replaced, not resized in place.
 */
export function growDomain(
	m: CMeshO,
	domain: ChannelDomain,
	newCap: number,
	liveCount: number,
): void {
	for (const desc of CHANNELS) {
		if (desc.domain !== domain) continue;
		const old = getChannel(m, desc.key);
		if (old === null) continue;
		const next = newArray(desc, newCap);
		next.set(old.subarray(0, liveCount * desc.arity) as never);
		setChannel(m, desc.key, next);
	}
	for (const attr of m.customAttrs) {
		if (attr.domain !== domain) continue;
		const next = new Float64Array(newCap * attr.arity);
		next.set(attr.data.subarray(0, liveCount * attr.arity));
		attr.data = next;
	}
	if (domain === "vert") m.vertCap = newCap;
	else if (domain === "face") m.faceCap = newCap;
	else m.edgeCap = newCap;
}

/**
 * Resets slots `[from, to)` of every channel of `domain` to their fill values.
 *
 * Capacity outlives the elements that occupied it: after a compaction or a
 * `clear()` the slots beyond the new size still hold the old bytes, including
 * a set DELETED flag. Allocating into that space without resetting it would
 * hand out a vertex that is already deleted — which is exactly the bug the
 * allocator property test found.
 */
export function resetDomainRange(m: CMeshO, domain: ChannelDomain, from: number, to: number): void {
	if (to <= from) return;
	for (const desc of CHANNELS) {
		if (desc.domain !== domain) continue;
		const arr = getChannel(m, desc.key);
		if (arr === null) continue;
		arr.fill(desc.fill ?? 0, from * desc.arity, to * desc.arity);
	}
	for (const attr of m.customAttrs) {
		if (attr.domain !== domain) continue;
		attr.data.fill(0, from * attr.arity, to * attr.arity);
	}
}

/**
 * Compacts every channel of `domain` in place according to `remap`
 * (old index → new index, or -1 for deleted).
 *
 * Safe as a single forward pass because compaction never moves an element to a
 * higher index.
 */
export function compactDomain(m: CMeshO, domain: ChannelDomain, remap: Int32Array): void {
	const move = (arr: NumArray, arity: number): void => {
		for (let oldIdx = 0; oldIdx < remap.length; oldIdx++) {
			const newIdx = remap[oldIdx];
			if (newIdx < 0 || newIdx === oldIdx) continue;
			const src = oldIdx * arity;
			const dst = newIdx * arity;
			for (let k = 0; k < arity; k++) arr[dst + k] = arr[src + k];
		}
	};
	for (const desc of CHANNELS) {
		if (desc.domain !== domain) continue;
		const arr = getChannel(m, desc.key);
		if (arr === null) continue;
		move(arr, desc.arity);
	}
	for (const attr of m.customAttrs) {
		if (attr.domain === domain) move(attr.data, attr.arity);
	}
}
