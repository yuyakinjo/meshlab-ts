/**
 * `vcg/complex/algorithms/attribute_seam.h` — splitting vertices so that a
 * per-wedge attribute becomes a per-vertex one.
 *
 * A texture seam, a hard crease, a colour discontinuity: all of them are places
 * where two faces meeting at a vertex disagree about what that vertex looks
 * like. The mesh stores the disagreement per *corner*, but most file formats
 * and every renderer want it per *vertex* — so the vertex has to become as many
 * vertices as there are distinct answers around it.
 *
 * The mesh's shape never changes. Only its vertex count and which corner refers
 * to which vertex.
 */
import { Allocator } from "./allocator.ts";
import type { CMeshO } from "./cmesho.ts";

/** Which per-corner attributes count as distinguishing. */
export interface SeamMask {
	readonly normal?: "vertex" | "wedge" | "face";
	readonly color?: "vertex" | "wedge" | "face";
	readonly texcoord?: "vertex" | "wedge";
}

/**
 * Splits every vertex whose corners disagree about the chosen attributes.
 *
 * Returns how many vertices were added. The attributes are also *written back*
 * to the per-vertex channels, since that is the point: after this runs, the
 * per-vertex normal, colour and UV are the ones the corners actually meant.
 */
export function splitVertexBySeam(m: CMeshO, mask: SeamMask): number {
	const parts: Array<(f: number, k: number) => number[]> = [];

	const wt = m.wedgeTexCoord;
	const wn = m.wedgeNormal;
	const wc = m.wedgeColor;
	const fc = m.faceColor;

	if (mask.normal === "vertex") {
		parts.push((f, k) => {
			const v = m.fv(f, k);
			return [m.vertNormal[3 * v], m.vertNormal[3 * v + 1], m.vertNormal[3 * v + 2]];
		});
	} else if (mask.normal === "wedge" && wn !== null) {
		parts.push((f, k) => [wn[9 * f + 3 * k], wn[9 * f + 3 * k + 1], wn[9 * f + 3 * k + 2]]);
	} else if (mask.normal === "face") {
		parts.push((f) => [m.faceNormal[3 * f], m.faceNormal[3 * f + 1], m.faceNormal[3 * f + 2]]);
	}

	if (mask.color === "vertex") {
		parts.push((f, k) => [m.vertColor[m.fv(f, k)]]);
	} else if (mask.color === "wedge" && wc !== null) {
		parts.push((f, k) => [wc[3 * f + k]]);
	} else if (mask.color === "face" && fc !== null) {
		parts.push((f) => [fc[f]]);
	}

	if (mask.texcoord === "vertex" && m.vertTexCoord !== null) {
		const vt = m.vertTexCoord;
		parts.push((f, k) => [vt[2 * m.fv(f, k)], vt[2 * m.fv(f, k) + 1]]);
	} else if (mask.texcoord === "wedge" && wt !== null) {
		parts.push((f, k) => [wt[6 * f + 2 * k], wt[6 * f + 2 * k + 1]]);
	}

	if (parts.length === 0) return 0;

	// Each distinct (vertex, attribute signature) becomes one output vertex.
	// The first signature seen at a vertex keeps the original slot; the rest
	// get copies.
	interface Slot {
		readonly source: number;
		index: number;
	}
	const slots = new Map<string, Slot>();
	const claimed = new Uint8Array(m.vertSize);
	const rewrite: Array<{ f: number; k: number; key: string }> = [];

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = m.fv(f, k);
			const key = `${v}|${parts.flatMap((part) => part(f, k)).join(",")}`;
			if (!slots.has(key)) {
				// -1 means "needs a fresh vertex"; filled in once the total is known.
				slots.set(key, { source: v, index: claimed[v] === 0 ? v : -1 });
				claimed[v] = 1;
			}
			rewrite.push({ f, k, key });
		}
	}

	const fresh = [...slots.values()].filter((slot) => slot.index < 0);
	if (fresh.length > 0) {
		const first = Allocator.addVertices(m, fresh.length);
		fresh.forEach((slot, i) => {
			slot.index = first + i;
			copyVertex(m, slot.source, slot.index);
		});
		for (const { f, k, key } of rewrite) {
			m.faceVert[3 * f + k] = (slots.get(key) as Slot).index;
		}
	}
	writeBack(m, mask, rewrite, slots);
	m.imark++;
	return fresh.length;
}

/**
 * Pushes the corner attributes onto the vertices that now represent them.
 *
 * Without this the split would be pointless: the corners would refer to
 * distinct vertices that all still carry the same per-vertex values.
 */
function writeBack(
	m: CMeshO,
	mask: SeamMask,
	rewrite: ReadonlyArray<{ f: number; k: number; key: string }>,
	slots: ReadonlyMap<string, { index: number }>,
): void {
	const wt = m.wedgeTexCoord;
	const wn = m.wedgeNormal;
	const wc = m.wedgeColor;
	const fc = m.faceColor;
	for (const { f, k, key } of rewrite) {
		const v = (slots.get(key) as { index: number }).index;
		if (mask.normal === "wedge" && wn !== null) {
			for (let a = 0; a < 3; a++) m.vertNormal[3 * v + a] = wn[9 * f + 3 * k + a];
		} else if (mask.normal === "face") {
			for (let a = 0; a < 3; a++) m.vertNormal[3 * v + a] = m.faceNormal[3 * f + a];
		}
		if (mask.color === "wedge" && wc !== null) m.vertColor[v] = wc[3 * f + k];
		else if (mask.color === "face" && fc !== null) m.vertColor[v] = fc[f];
		if (mask.texcoord === "wedge" && wt !== null && m.vertTexCoord !== null) {
			m.vertTexCoord[2 * v] = wt[6 * f + 2 * k];
			m.vertTexCoord[2 * v + 1] = wt[6 * f + 2 * k + 1];
		}
	}
}

function copyVertex(m: CMeshO, from: number, to: number): void {
	m.setVert(to, m.vx(from), m.vy(from), m.vz(from));
	for (let k = 0; k < 3; k++) m.vertNormal[3 * to + k] = m.vertNormal[3 * from + k];
	m.vertQuality[to] = m.vertQuality[from];
	m.vertColor[to] = m.vertColor[from];
	if (m.vertTexCoord !== null) {
		m.vertTexCoord[2 * to] = m.vertTexCoord[2 * from];
		m.vertTexCoord[2 * to + 1] = m.vertTexCoord[2 * from + 1];
	}
	if (m.vertRadius !== null) m.vertRadius[to] = m.vertRadius[from];
	for (const attr of m.customAttrs) {
		if (attr.domain !== "vert") continue;
		for (let k = 0; k < attr.arity; k++) {
			attr.data[attr.arity * to + k] = attr.data[attr.arity * from + k];
		}
	}
}

export const AttributeSeam = { splitVertexBySeam } as const;
