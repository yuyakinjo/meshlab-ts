import { describe, expect, test } from "bun:test";
import { MeshElement } from "../../../src/common/ml_document/mesh_element.ts";
import { MLInternalException } from "../../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import { FaceFlag, VertexFlag } from "../../../src/vcg/complex/flags.ts";
import { assertAllocatorConsistent, geometryDigest, liveVerts } from "../../helpers/invariants.ts";
import { cube, tetrahedron } from "../../helpers/mesh_builders.ts";

describe("Allocator: adding", () => {
	test("a fresh mesh is empty and consistent", () => {
		const m = new CMeshO();
		expect(m.vn).toBe(0);
		expect(m.fn).toBe(0);
		expect(m.vertSize).toBe(0);
		expect(m.isCompact).toBe(true);
		assertAllocatorConsistent(m);
	});

	test("addVertex returns sequential indices and stores coordinates", () => {
		const m = new CMeshO();
		expect(Allocator.addVertex(m, 1, 2, 3)).toBe(0);
		expect(Allocator.addVertex(m, 4, 5, 6)).toBe(1);
		expect(m.vn).toBe(2);
		expect([m.vx(1), m.vy(1), m.vz(1)]).toEqual([4, 5, 6]);
		assertAllocatorConsistent(m);
	});

	test("new vertices start white, unflagged and with an empty VF chain", () => {
		const m = new CMeshO();
		m.enableChannels(MeshElement.MM_VERTFACETOPO);
		const v = Allocator.addVertex(m, 0, 0, 0);
		expect(m.vertColor[v]).toBe(0xffffffff);
		expect(m.vertFlags[v]).toBe(0);
		expect(m.vfHeadFace![v]).toBe(-1);
	});

	test("adding zero elements is a no-op that still returns the next index", () => {
		const m = new CMeshO();
		Allocator.addVertices(m, 3);
		expect(Allocator.addVertices(m, 0)).toBe(3);
		expect(m.vn).toBe(3);
	});

	test("a negative count is rejected", () => {
		const m = new CMeshO();
		expect(() => Allocator.addVertices(m, -1)).toThrow(MLInternalException);
		expect(() => Allocator.addFaces(m, -1)).toThrow(MLInternalException);
	});

	test("growth preserves the data already written", () => {
		const m = new CMeshO();
		const n = 500;
		for (let i = 0; i < n; i++) Allocator.addVertex(m, i, i * 2, i * 3);
		expect(m.vn).toBe(n);
		expect(m.vertCap).toBeGreaterThanOrEqual(n);
		for (let i = 0; i < n; i++) {
			expect([m.vx(i), m.vy(i), m.vz(i)]).toEqual([i, i * 2, i * 3]);
		}
		assertAllocatorConsistent(m);
	});

	test("growth preserves optional channels too", () => {
		const m = new CMeshO();
		m.enableChannels(MeshElement.MM_FACEQUALITY);
		const a = Allocator.addVertex(m, 0, 0, 0);
		const b = Allocator.addVertex(m, 1, 0, 0);
		const c = Allocator.addVertex(m, 0, 1, 0);
		for (let i = 0; i < 300; i++) {
			const f = Allocator.addFace(m, a, b, c);
			m.faceQuality![f] = i * 0.5;
		}
		for (let i = 0; i < 300; i++) expect(m.faceQuality![i]).toBe(i * 0.5);
		assertAllocatorConsistent(m);
	});

	test("addMeshData offsets face indices into the existing mesh", () => {
		const m = new CMeshO();
		Allocator.addMeshData(m, [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]);
		const { firstVert, firstFace } = Allocator.addMeshData(
			m,
			[5, 0, 0, 6, 0, 0, 5, 1, 0],
			[0, 1, 2],
		);
		expect(firstVert).toBe(3);
		expect(firstFace).toBe(1);
		expect([m.fv(1, 0), m.fv(1, 1), m.fv(1, 2)]).toEqual([3, 4, 5]);
		assertAllocatorConsistent(m);
	});

	test("addMeshData rejects arrays that are not triples", () => {
		const m = new CMeshO();
		expect(() => Allocator.addMeshData(m, [0, 0], [])).toThrow(MLInternalException);
		expect(() => Allocator.addMeshData(m, [], [0, 1])).toThrow(MLInternalException);
	});
});

describe("Allocator: lazy deletion", () => {
	test("deleting sets the flag and decrements the live count but keeps the slot", () => {
		const { mesh } = cube();
		Allocator.deleteFace(mesh, 3);
		expect(mesh.fn).toBe(11);
		expect(mesh.faceSize).toBe(12);
		expect(mesh.isFaceD(3)).toBe(true);
		expect(mesh.isCompact).toBe(false);
		// Untouched faces keep their indices, which is the whole point.
		expect(mesh.fv(4, 0)).toBe(5);
	});

	test("deleting twice is an error rather than a silent no-op", () => {
		const { mesh } = cube();
		Allocator.deleteFace(mesh, 0);
		expect(() => Allocator.deleteFace(mesh, 0)).toThrow(MLInternalException);
		Allocator.deleteVertex(mesh, 0);
		expect(() => Allocator.deleteVertex(mesh, 0)).toThrow(MLInternalException);
	});

	test("out-of-range deletion is an error", () => {
		const { mesh } = cube();
		expect(() => Allocator.deleteFace(mesh, 99)).toThrow(MLInternalException);
		expect(() => Allocator.deleteVertex(mesh, -1)).toThrow(MLInternalException);
	});

	test("deletion does not disturb other flags on the same element", () => {
		const { mesh } = cube();
		mesh.faceFlags[2] |= FaceFlag.SELECTED;
		Allocator.deleteFace(mesh, 2);
		expect(mesh.isFaceS(2)).toBe(true);
		mesh.vertFlags[1] |= VertexFlag.BORDER;
		Allocator.deleteVertex(mesh, 1);
		expect(mesh.isVertB(1)).toBe(true);
	});
});

describe("Allocator: compaction", () => {
	test("compacting an untouched mesh is the identity and changes nothing", () => {
		const { mesh } = cube();
		const before = geometryDigest(mesh);
		const remap = Allocator.compactEveryVector(mesh);
		expect(Array.from(remap.faceRemap)).toEqual([...Array(12).keys()]);
		expect(Array.from(remap.vertRemap)).toEqual([...Array(8).keys()]);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("compaction removes the deleted slots and returns the remap", () => {
		const { mesh } = cube();
		Allocator.deleteFace(mesh, 1);
		Allocator.deleteFace(mesh, 5);
		const remap = Allocator.compactFaceVector(mesh);
		expect(mesh.faceSize).toBe(10);
		expect(mesh.fn).toBe(10);
		expect(remap[1]).toBe(-1);
		expect(remap[5]).toBe(-1);
		expect(remap[0]).toBe(0);
		expect(remap[2]).toBe(1);
		expect(remap[6]).toBe(4);
		assertAllocatorConsistent(mesh);
	});

	test("compaction preserves the surviving geometry exactly", () => {
		const { mesh } = cube();
		const survivor = geometryDigest(cubeMinusFaces([1, 5]));
		Allocator.deleteFace(mesh, 1);
		Allocator.deleteFace(mesh, 5);
		Allocator.compactEveryVector(mesh);
		expect(geometryDigest(mesh)).toBe(survivor);
	});

	test("vertex compaction rewrites face references through the remap", () => {
		const { mesh } = tetrahedron();
		// Drop the faces touching vertex 0, then the vertex itself.
		for (const f of [2, 1, 0]) Allocator.deleteFace(mesh, f);
		Allocator.deleteVertex(mesh, 0);
		Allocator.compactEveryVector(mesh);
		expect(mesh.vn).toBe(3);
		expect(mesh.fn).toBe(1);
		// The one surviving face was (1,3,2); after removing vertex 0 those
		// become (0,2,1).
		expect([mesh.fv(0, 0), mesh.fv(0, 1), mesh.fv(0, 2)]).toEqual([0, 2, 1]);
		assertAllocatorConsistent(mesh);
	});

	test("compacting a vertex still used by a live face is an error", () => {
		const { mesh } = cube();
		Allocator.deleteVertex(mesh, 0);
		expect(() => Allocator.compactVertexVector(mesh)).toThrow(MLInternalException);
	});

	test("compaction preserves per-vertex attribute values, not just positions", () => {
		const { mesh } = cube();
		for (let v = 0; v < mesh.vn; v++) mesh.vertQuality[v] = v * 10;
		for (const f of [0, 1, 4, 5, 8, 9, 10, 11, 2]) Allocator.deleteFace(mesh, f);
		// Faces 3 (1,3,2), 6 (0,4,7) and 7 (0,7,3) remain; vertices 5 and 6 are free.
		for (const v of [5, 6]) Allocator.deleteVertex(mesh, v);
		Allocator.compactEveryVector(mesh);
		const qualities = liveVerts(mesh).map((v) => mesh.vertQuality[v]);
		expect(qualities).toEqual([0, 10, 20, 30, 40, 70]);
		assertAllocatorConsistent(mesh);
	});

	test("compaction drops stale adjacency rather than preserving a lie", () => {
		const { mesh } = cube();
		mesh.enableChannels(MeshElement.MM_FACEFACETOPO);
		expect(mesh.ffFace).not.toBeNull();
		Allocator.deleteFace(mesh, 0);
		Allocator.compactFaceVector(mesh);
		expect(mesh.ffFace).toBeNull();
		expect(mesh.hasDataMask(MeshElement.MM_FACEFACETOPO)).toBe(false);
	});

	test("capacity recycled by a compaction hands out clean slots", () => {
		// Regression: compaction lowers vertSize but leaves the freed slots
		// holding the old occupant's flags. Allocating into that space without
		// resetting it produced a vertex that was already marked deleted, so
		// vn and the deleted-flag count disagreed from birth.
		const m = new CMeshO();
		const v = Allocator.addVertex(m, 1, 2, 3);
		Allocator.deleteVertex(m, v);
		Allocator.compactEveryVector(m);
		expect(m.vertSize).toBe(0);
		expect(m.vertCap).toBeGreaterThan(0); // the capacity is still there

		const reused = Allocator.addVertex(m, 9, 9, 9);
		expect(reused).toBe(0);
		expect(m.isVertD(reused)).toBe(false);
		expect(m.vn).toBe(1);
		expect(m.vertColor[reused]).toBe(0xffffffff);
		assertAllocatorConsistent(m);
	});

	test("faces allocated into recycled capacity are clean too", () => {
		const m = new CMeshO();
		const a = Allocator.addVertex(m, 0, 0, 0);
		const b = Allocator.addVertex(m, 1, 0, 0);
		const c = Allocator.addVertex(m, 0, 1, 0);
		const f = Allocator.addFace(m, a, b, c);
		Allocator.deleteFace(m, f);
		Allocator.compactFaceVector(m);
		const reused = Allocator.addFace(m, a, b, c);
		expect(m.isFaceD(reused)).toBe(false);
		expect(m.fn).toBe(1);
		assertAllocatorConsistent(m);
	});

	test("a no-op compaction keeps adjacency", () => {
		const { mesh } = cube();
		mesh.enableChannels(MeshElement.MM_FACEFACETOPO);
		Allocator.compactEveryVector(mesh);
		expect(mesh.ffFace).not.toBeNull();
	});
});

describe("CMeshO: channels", () => {
	test("optional channels are absent until enabled", () => {
		const m = new CMeshO();
		expect(m.faceQuality).toBeNull();
		expect(m.hasDataMask(MeshElement.MM_FACEQUALITY)).toBe(false);
		m.enableChannels(MeshElement.MM_FACEQUALITY);
		expect(m.faceQuality).not.toBeNull();
		expect(m.hasDataMask(MeshElement.MM_FACEQUALITY)).toBe(true);
	});

	test("enabling an already-enabled channel keeps its contents", () => {
		const m = new CMeshO();
		Allocator.addFace(m, 0, 0, 0);
		m.enableChannels(MeshElement.MM_FACEQUALITY);
		m.faceQuality![0] = 7;
		m.enableChannels(MeshElement.MM_FACEQUALITY);
		expect(m.faceQuality![0]).toBe(7);
	});

	test("disabling releases the storage and clears the mask bit", () => {
		const m = new CMeshO();
		m.enableChannels(MeshElement.MM_FACECOLOR);
		m.disableChannels(MeshElement.MM_FACECOLOR);
		expect(m.faceColor).toBeNull();
		expect(m.hasDataMask(MeshElement.MM_FACECOLOR)).toBe(false);
	});

	test("mandatory channels are always present", () => {
		const m = new CMeshO();
		expect(m.hasDataMask(MeshElement.MM_VERTCOORD)).toBe(true);
		expect(m.hasDataMask(MeshElement.MM_FACEVERT)).toBe(true);
		expect(m.hasDataMask(MeshElement.MM_VERTQUALITY)).toBe(true);
	});

	test("clear() empties the mesh but keeps the enabled channels", () => {
		const { mesh } = cube();
		mesh.enableChannels(MeshElement.MM_FACEQUALITY);
		mesh.clear();
		expect(mesh.vn).toBe(0);
		expect(mesh.fn).toBe(0);
		expect(mesh.vertSize).toBe(0);
		expect(mesh.faceQuality).not.toBeNull();
		assertAllocatorConsistent(mesh);
	});
});

/** A cube with the given faces never added, as an independent expected value. */
function cubeMinusFaces(dropped: readonly number[]): CMeshO {
	const src = cube().mesh;
	const m = new CMeshO();
	for (let v = 0; v < src.vn; v++) Allocator.addVertex(m, src.vx(v), src.vy(v), src.vz(v));
	for (let f = 0; f < src.fn; f++) {
		if (dropped.includes(f)) continue;
		Allocator.addFace(m, src.fv(f, 0), src.fv(f, 1), src.fv(f, 2));
	}
	return m;
}
