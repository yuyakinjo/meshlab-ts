/**
 * `CMeshO` — the mesh type every filter operates on, mirroring MeshLab's
 * `vcg::tri::TriMesh<vertex::vector_ocf<CVertexO>, face::vector_ocf<CFaceO>>`.
 *
 * Two things about the representation are worth knowing before reading any
 * algorithm in this repo:
 *
 * 1. **Structure of arrays, addressed by index.** A vertex is a number, not an
 *    object. `m.vertCoord[3 * v + 1]` is its y. C++ VCGLib hands out raw
 *    pointers into a reallocating vector and therefore needs `PointerUpdater`
 *    to fix them up after every insertion; indices make that machinery
 *    unnecessary, and let adjacency live in `Int32Array`s.
 *
 * 2. **Deletion is lazy.** `Allocator.deleteVertex` sets a DELETED flag and
 *    decrements `vn`; the slot stays put so that every other index keeps
 *    meaning what it meant. Consequently `vertSize !== vn` in general, and the
 *    canonical loop is
 *
 *    ```ts
 *    for (let f = 0; f < m.faceSize; f++) {
 *        if (m.isFaceD(f)) continue;
 *        ...
 *    }
 *    ```
 *
 *    Reclaiming the slots is an explicit `Allocator.compact*` call that returns
 *    an old→new remap.
 */
import { MeshElement, maskAnd, maskOr } from "../../common/ml_document/mesh_element.ts";
import { Box3 } from "../space/box3.ts";
import {
	CHANNELS,
	type ChannelDomain,
	type CustomAttribute,
	disableChannels,
	enableChannels,
	hasChannels,
	MANDATORY_MASK,
	setChannel,
} from "./components.ts";
import { borderBit, FaceFlag, VertexFlag } from "./flags.ts";

/** A colour packed as 0xAABBGGRR, matching `vcg::Color4b`'s byte order. */
export type Color4b = number;

export const WHITE: Color4b = 0xffffffff;

export class CMeshO {
	/** Allocated vertex slots, live or deleted. VCG's `vert.size()`. */
	vertSize = 0;
	/** Allocated face slots, live or deleted. VCG's `face.size()`. */
	faceSize = 0;
	/** Live (non-deleted) vertex count. VCG's `vn`. */
	vn = 0;
	/** Live (non-deleted) face count. VCG's `fn`. */
	fn = 0;

	/** Capacity of the vertex-domain channel arrays, in elements. */
	vertCap = 0;
	/** Capacity of the face-domain channel arrays, in elements. */
	faceCap = 0;

	// ---- vertex channels, always present -----------------------------------
	vertCoord: Float64Array = new Float64Array(0);
	vertFlags: Uint32Array = new Uint32Array(0);
	vertNormal: Float64Array = new Float64Array(0);
	vertQuality: Float64Array = new Float64Array(0);
	vertColor: Uint32Array = new Uint32Array(0);

	// ---- vertex channels, optional (VCG "Ocf") ------------------------------
	vertTexCoord: Float64Array | null = null;
	vertRadius: Float64Array | null = null;
	vertMark: Int32Array | null = null;
	vertCurvDir: Float64Array | null = null;

	// ---- face channels, always present --------------------------------------
	faceVert: Int32Array = new Int32Array(0);
	faceFlags: Uint32Array = new Uint32Array(0);
	faceNormal: Float64Array = new Float64Array(0);

	// ---- face channels, optional --------------------------------------------
	faceQuality: Float64Array | null = null;
	faceColor: Uint32Array | null = null;
	faceMark: Int32Array | null = null;
	faceCurvDir: Float64Array | null = null;
	wedgeTexCoord: Float64Array | null = null;
	/** Which entry of {@link textures} each corner's UV addresses. */
	wedgeTexIndex: Int32Array | null = null;
	wedgeNormal: Float64Array | null = null;
	wedgeColor: Uint32Array | null = null;

	// ---- adjacency -----------------------------------------------------------
	ffFace: Int32Array | null = null;
	ffEdge: Uint8Array | null = null;
	vfHeadFace: Int32Array | null = null;
	vfHeadIndex: Uint8Array | null = null;
	vfNextFace: Int32Array | null = null;
	vfNextIndex: Uint8Array | null = null;

	// ---- mesh-level state -----------------------------------------------------
	bbox: Box3 = Box3.empty();
	/** VCG's `Tr`: the mesh's own transform, not applied to the coordinates. */
	transformMatrix: Float64Array = identity4();
	textures: string[] = [];
	/**
	 * VCG's `m.C()`: one colour for the whole mesh, distinct from the per-vertex
	 * and per-face channels. A viewer falls back to it when neither is present.
	 */
	color: Color4b = WHITE;
	/**
	 * Attributes a filter added at run time, VCG's `PerVertexAttributeHandle`
	 * and `PerFaceAttributeHandle`.
	 *
	 * A list rather than rows in {@link CHANNELS} because the set is not known
	 * until a script runs. They still grow, reset and compact with everything
	 * else: `components.ts` walks this list alongside the table.
	 */
	customAttrs: CustomAttribute[] = [];
	/** Incremented on every structural mutation; caches key off this. */
	imark = 0;
	/** Which channels are currently live. Mirrors `MeshModel.dataMask()`. */
	currentDataMask: number = MANDATORY_MASK;

	constructor() {
		// Mandatory channels start out as real (zero-length) arrays so no hot
		// path ever has to null-check them.
		for (const desc of CHANNELS) {
			if (desc.optional) continue;
			setChannel(this, desc.key, new desc.ctor(0));
		}
	}

	// ---- vertex accessors ------------------------------------------------------
	vx(v: number): number {
		return this.vertCoord[3 * v];
	}
	vy(v: number): number {
		return this.vertCoord[3 * v + 1];
	}
	vz(v: number): number {
		return this.vertCoord[3 * v + 2];
	}

	setVert(v: number, x: number, y: number, z: number): void {
		const i = 3 * v;
		this.vertCoord[i] = x;
		this.vertCoord[i + 1] = y;
		this.vertCoord[i + 2] = z;
	}

	getVert(v: number, out: Float64Array | number[]): void {
		const i = 3 * v;
		out[0] = this.vertCoord[i];
		out[1] = this.vertCoord[i + 1];
		out[2] = this.vertCoord[i + 2];
	}

	isVertD(v: number): boolean {
		return (this.vertFlags[v] & VertexFlag.DELETED) !== 0;
	}
	isVertS(v: number): boolean {
		return (this.vertFlags[v] & VertexFlag.SELECTED) !== 0;
	}
	isVertB(v: number): boolean {
		return (this.vertFlags[v] & VertexFlag.BORDER) !== 0;
	}
	isVertV(v: number): boolean {
		return (this.vertFlags[v] & VertexFlag.VISITED) !== 0;
	}

	// ---- face accessors --------------------------------------------------------
	/** Vertex index at corner `k` (0..2) of face `f`. */
	fv(f: number, k: number): number {
		return this.faceVert[3 * f + k];
	}

	setFace(f: number, v0: number, v1: number, v2: number): void {
		const i = 3 * f;
		this.faceVert[i] = v0;
		this.faceVert[i + 1] = v1;
		this.faceVert[i + 2] = v2;
	}

	isFaceD(f: number): boolean {
		return (this.faceFlags[f] & FaceFlag.DELETED) !== 0;
	}
	isFaceS(f: number): boolean {
		return (this.faceFlags[f] & FaceFlag.SELECTED) !== 0;
	}
	isFaceV(f: number): boolean {
		return (this.faceFlags[f] & FaceFlag.VISITED) !== 0;
	}
	/** True when edge `e` of face `f` carries the border bit. */
	isFaceB(f: number, e: number): boolean {
		return (this.faceFlags[f] & borderBit(e)) !== 0;
	}

	// ---- adjacency accessors -----------------------------------------------------
	/** Face adjacent across edge `e` of face `f`. Equals `f` itself on a border. */
	ffp(f: number, e: number): number {
		return this.ffFace![3 * f + e];
	}
	/** Which edge of `ffp(f, e)` we arrive on. */
	ffi(f: number, e: number): number {
		return this.ffEdge![3 * f + e];
	}
	/**
	 * True when edge `e` of face `f` has no neighbour, using VCG's
	 * self-reference encoding. This reads the FF adjacency, unlike
	 * {@link isFaceB} which reads a cached flag bit.
	 *
	 * Both the face *and* the edge index must match. VCGLib tests only the
	 * face, which is fine while no face shares an edge with itself — but a
	 * degenerate face such as (v, v, v) links its own three corners into one
	 * ring, and the face-only test would call every one of them a border.
	 */
	isBorderFF(f: number, e: number): boolean {
		return this.ffFace![3 * f + e] === f && this.ffEdge![3 * f + e] === e;
	}

	// ---- datamask ------------------------------------------------------------------
	hasDataMask(mask: number): boolean {
		return maskAnd(this.currentDataMask, mask) === mask >>> 0 && hasChannels(this, mask);
	}

	/** Allocates storage for `mask`. Does not compute adjacency; see MeshModel. */
	enableChannels(mask: number): void {
		enableChannels(this, mask);
	}

	disableChannels(mask: number): void {
		disableChannels(this, mask);
	}

	// ---- custom attributes ----------------------------------------------------------
	/** The attribute of that name and domain, or undefined. */
	customAttribute(name: string, domain: ChannelDomain): CustomAttribute | undefined {
		return this.customAttrs.find((a) => a.name === name && a.domain === domain);
	}

	/**
	 * Adds an attribute, or returns the existing one when it already matches.
	 *
	 * Re-adding under a different arity throws rather than silently reshaping
	 * the storage: a scalar and a point attribute of the same name are two
	 * different things, and every expression already compiled against the old
	 * one would start reading garbage.
	 */
	addCustomAttribute(name: string, domain: ChannelDomain, arity: number): CustomAttribute {
		const existing = this.customAttribute(name, domain);
		if (existing !== undefined) {
			if (existing.arity !== arity) {
				throw new Error(
					`the ${domain} attribute "${name}" already exists with ${existing.arity} components, not ${arity}`,
				);
			}
			return existing;
		}
		const cap = domain === "vert" ? this.vertCap : this.faceCap;
		const attr: CustomAttribute = { name, domain, arity, data: new Float64Array(cap * arity) };
		this.customAttrs.push(attr);
		return attr;
	}

	/** Drops the attribute if present; returns whether there was one. */
	deleteCustomAttribute(name: string, domain: ChannelDomain): boolean {
		const i = this.customAttrs.findIndex((a) => a.name === name && a.domain === domain);
		if (i < 0) return false;
		this.customAttrs.splice(i, 1);
		return true;
	}

	// ---- whole-mesh operations -------------------------------------------------------
	/** Drops all geometry but keeps which optional channels are enabled. */
	clear(): void {
		const mask = this.currentDataMask;
		this.vertSize = 0;
		this.faceSize = 0;
		this.vn = 0;
		this.fn = 0;
		this.vertCap = 0;
		this.faceCap = 0;
		for (const desc of CHANNELS) {
			if (desc.optional && maskAnd(mask, desc.mask) === 0) continue;
			setChannel(this, desc.key, new desc.ctor(0));
		}
		// The attributes survive the way an enabled channel does — the mesh
		// keeps its shape, it just has no elements left.
		for (const attr of this.customAttrs) attr.data = new Float64Array(0);
		this.bbox.setEmpty();
		this.imark = 0;
	}

	/** True when no slot is deleted, i.e. indices are dense. */
	get isCompact(): boolean {
		return this.vertSize === this.vn && this.faceSize === this.fn;
	}
}

function identity4(): Float64Array {
	const m = new Float64Array(16);
	m[0] = 1;
	m[5] = 1;
	m[10] = 1;
	m[15] = 1;
	return m;
}

/** The datamask a mesh reports right after construction. */
export const DEFAULT_DATA_MASK: number = maskOr(
	MANDATORY_MASK,
	maskOr(MeshElement.MM_VERTNUMBER, MeshElement.MM_FACENUMBER),
);
