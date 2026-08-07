/**
 * `Allocator` — adding, deleting and compacting mesh elements.
 *
 * Mirrors `vcg::tri::Allocator<MeshType>` minus the `PointerUpdater` family,
 * which exists in C++ only to repair raw pointers after a `std::vector`
 * reallocates. Indices survive reallocation, so the compaction functions here
 * simply *return* the old→new remap and let the caller translate whatever it
 * was holding.
 *
 * Two rules callers must follow:
 *
 * - **Never hoist a channel array across an add.** `addVertices` may replace
 *   `m.vertCoord` with a larger array; a hoisted `const c = m.vertCoord` would
 *   then write into the discarded one.
 * - **Deletion does not repair adjacency.** As in VCGLib, deleting a face
 *   leaves its neighbours pointing at it. Compaction therefore drops stale
 *   adjacency rather than preserving a lie; re-request it via
 *   `MeshModel.updateDataMask` (or `UpdateTopology`) when you next need it.
 */
import { MeshElement, maskAnd, maskOr } from "../../common/ml_document/mesh_element.ts";
import { MLInternalException } from "../../common/utilities/ml_exception.ts";
import type { CMeshO } from "./cmesho.ts";
import { compactDomain, disableChannels, growDomain, resetDomainRange } from "./components.ts";
import { FaceFlag, VertexFlag } from "./flags.ts";

const MIN_CAPACITY = 16;
const GROWTH_FACTOR = 1.5;

function nextCapacity(current: number, needed: number): number {
	let cap = Math.max(current, MIN_CAPACITY);
	while (cap < needed) cap = Math.ceil(cap * GROWTH_FACTOR);
	return cap;
}

const ADJACENCY_MASK = maskOr(MeshElement.MM_FACEFACETOPO, MeshElement.MM_VERTFACETOPO);

/**
 * Appends `n` vertices and returns the index of the first one.
 *
 * New vertices are zeroed, unselected and undeleted; their colour is white and
 * their VF chain empty, per the channel table's fill values.
 */
export function addVertices(m: CMeshO, n: number): number {
	if (n < 0) throw new MLInternalException(`addVertices(${n}): negative count`);
	const first = m.vertSize;
	if (n === 0) return first;

	const needed = first + n;
	if (needed > m.vertCap) growDomain(m, "vert", nextCapacity(m.vertCap, needed), m.vertSize);
	// The slots may be recycled capacity left behind by a compaction, still
	// holding the previous occupant's flags. Start them clean.
	resetDomainRange(m, "vert", first, needed);

	m.vertSize = needed;
	m.vn += n;
	m.imark++;
	return first;
}

/** Appends `n` faces and returns the index of the first one. */
export function addFaces(m: CMeshO, n: number): number {
	if (n < 0) throw new MLInternalException(`addFaces(${n}): negative count`);
	const first = m.faceSize;
	if (n === 0) return first;

	const needed = first + n;
	if (needed > m.faceCap) growDomain(m, "face", nextCapacity(m.faceCap, needed), m.faceSize);
	resetDomainRange(m, "face", first, needed);

	m.faceSize = needed;
	m.fn += n;
	m.imark++;
	return first;
}

/** Appends one vertex at `(x, y, z)` and returns its index. */
export function addVertex(m: CMeshO, x: number, y: number, z: number): number {
	const v = addVertices(m, 1);
	m.setVert(v, x, y, z);
	return v;
}

/** Appends one triangle on the given vertex indices and returns its index. */
export function addFace(m: CMeshO, v0: number, v1: number, v2: number): number {
	const f = addFaces(m, 1);
	m.setFace(f, v0, v1, v2);
	return f;
}

/**
 * Appends a whole mesh's worth of geometry at once.
 *
 * `coords` is xyz-interleaved and `faces` is a flat list of vertex indices
 * *relative to this batch*; they are offset by the first new vertex index.
 */
export function addMeshData(
	m: CMeshO,
	coords: ArrayLike<number>,
	faces: ArrayLike<number>,
): { firstVert: number; firstFace: number } {
	if (coords.length % 3 !== 0) {
		throw new MLInternalException(`coords length ${coords.length} is not a multiple of 3`);
	}
	if (faces.length % 3 !== 0) {
		throw new MLInternalException(`faces length ${faces.length} is not a multiple of 3`);
	}
	const nv = coords.length / 3;
	const nf = faces.length / 3;
	const firstVert = addVertices(m, nv);
	const firstFace = addFaces(m, nf);

	const vc = m.vertCoord;
	for (let i = 0; i < coords.length; i++) vc[firstVert * 3 + i] = coords[i];

	const fvArr = m.faceVert;
	for (let i = 0; i < faces.length; i++) fvArr[firstFace * 3 + i] = firstVert + faces[i];

	return { firstVert, firstFace };
}

/** Marks a vertex deleted. Throws if it already was — as VCG's assert does. */
export function deleteVertex(m: CMeshO, v: number): void {
	if (v < 0 || v >= m.vertSize)
		throw new MLInternalException(`deleteVertex: index ${v} out of range`);
	if ((m.vertFlags[v] & VertexFlag.DELETED) !== 0) {
		throw new MLInternalException(`deleteVertex: vertex ${v} is already deleted`);
	}
	m.vertFlags[v] |= VertexFlag.DELETED;
	m.vn--;
	m.imark++;
}

/** Marks a face deleted. Does not detach it from its neighbours. */
export function deleteFace(m: CMeshO, f: number): void {
	if (f < 0 || f >= m.faceSize)
		throw new MLInternalException(`deleteFace: index ${f} out of range`);
	if ((m.faceFlags[f] & FaceFlag.DELETED) !== 0) {
		throw new MLInternalException(`deleteFace: face ${f} is already deleted`);
	}
	m.faceFlags[f] |= FaceFlag.DELETED;
	m.fn--;
	m.imark++;
}

/** Builds the old→new index map for a domain: sequential for live, -1 for deleted. */
function buildRemap(size: number, isDeleted: (i: number) => boolean): Int32Array {
	const remap = new Int32Array(size);
	let next = 0;
	for (let i = 0; i < size; i++) {
		remap[i] = isDeleted(i) ? -1 : next++;
	}
	return remap;
}

function isIdentity(remap: Int32Array): boolean {
	for (let i = 0; i < remap.length; i++) if (remap[i] !== i) return false;
	return true;
}

/**
 * Adjacency describes relationships between elements that may have just been
 * deleted, so any real compaction invalidates it. Dropping the channels is
 * honest: the next `updateDataMask` rebuilds them from scratch.
 */
function dropAdjacency(m: CMeshO): void {
	if (maskAnd(m.currentDataMask, ADJACENCY_MASK) === 0) return;
	disableChannels(m, ADJACENCY_MASK);
}

/**
 * Removes deleted vertices and returns the old→new remap (-1 where deleted).
 *
 * Rewrites `faceVert` through the remap. Throws if a live face still references
 * a deleted vertex, which would mean the caller deleted a vertex without
 * deleting the faces around it.
 */
export function compactVertexVector(m: CMeshO): Int32Array {
	const remap = buildRemap(m.vertSize, (v) => m.isVertD(v));
	if (isIdentity(remap)) return remap;

	dropAdjacency(m);
	compactDomain(m, "vert", remap);
	m.vertSize = m.vn;

	const fvArr = m.faceVert;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const old = fvArr[3 * f + k];
			const next = remap[old];
			if (next < 0) {
				throw new MLInternalException(
					`compactVertexVector: live face ${f} references deleted vertex ${old}`,
				);
			}
			fvArr[3 * f + k] = next;
		}
	}

	m.imark++;
	return remap;
}

/** Removes deleted faces and returns the old→new remap (-1 where deleted). */
export function compactFaceVector(m: CMeshO): Int32Array {
	const remap = buildRemap(m.faceSize, (f) => m.isFaceD(f));
	if (isIdentity(remap)) return remap;

	dropAdjacency(m);
	compactDomain(m, "face", remap);
	m.faceSize = m.fn;
	m.imark++;
	return remap;
}

/**
 * Compacts both domains. Faces first, so that the vertex pass only has to walk
 * the surviving faces.
 */
export function compactEveryVector(m: CMeshO): {
	vertRemap: Int32Array;
	faceRemap: Int32Array;
} {
	const faceRemap = compactFaceVector(m);
	const vertRemap = compactVertexVector(m);
	return { vertRemap, faceRemap };
}

/**
 * Clears the DELETED bit from every slot and recomputes `vn`/`fn`.
 *
 * Only meaningful directly after a deletion the caller wants to take back;
 * mostly useful in tests.
 */
export function undeleteEverything(m: CMeshO): void {
	let vn = 0;
	for (let v = 0; v < m.vertSize; v++) {
		m.vertFlags[v] &= ~VertexFlag.DELETED;
		vn++;
	}
	let fn = 0;
	for (let f = 0; f < m.faceSize; f++) {
		m.faceFlags[f] &= ~FaceFlag.DELETED;
		fn++;
	}
	m.vn = vn;
	m.fn = fn;
	m.imark++;
}

export const Allocator = {
	addVertices,
	addFaces,
	addVertex,
	addFace,
	addMeshData,
	deleteVertex,
	deleteFace,
	compactVertexVector,
	compactFaceVector,
	compactEveryVector,
	undeleteEverything,
} as const;
