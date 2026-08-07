/**
 * Topology and the analysis half of Clean, checked against the builders'
 * mathematically derived facts and against the naive implementations in
 * `invariants.ts`.
 */
import { describe, expect, test } from "bun:test";
import { MeshElement } from "../../../src/common/ml_document/mesh_element.ts";
import { MLInternalException } from "../../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../../src/vcg/complex/allocator.ts";
import { Clean } from "../../../src/vcg/complex/clean.ts";
import { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import { UpdateFlags } from "../../../src/vcg/complex/update/flag.ts";
import { UpdateTopology } from "../../../src/vcg/complex/update/topology.ts";
import { assertAllocatorConsistent, assertFFConsistent } from "../../helpers/invariants.ts";
import {
	ALL_BUILDERS,
	type BuiltMesh,
	bowtieVertex,
	buildMesh,
	cube,
	cubeSoup,
	cubeWithFlippedFaces,
	gridPlane,
	mobiusStrip,
	nonManifoldEdgeFan,
	singleTriangle,
	sphereWithHoles,
	tetrahedron,
	torus,
	WELL_FORMED_BUILDERS,
} from "../../helpers/mesh_builders.ts";

/** Builds a mesh with both adjacency relations ready. */
function withTopology(built: BuiltMesh): CMeshO {
	UpdateTopology.faceFace(built.mesh);
	UpdateTopology.vertexFace(built.mesh);
	return built.mesh;
}

describe("UpdateTopology.faceFace", () => {
	for (const build of ALL_BUILDERS) {
		const built = build();
		test(`${built.name}: FF forms valid rings`, () => {
			UpdateTopology.faceFace(built.mesh);
			assertFFConsistent(built.mesh, built.name);
			assertAllocatorConsistent(built.mesh, built.name);
		});
	}

	test("allocates the channel when it is missing", () => {
		const { mesh } = cube();
		expect(mesh.ffFace).toBeNull();
		UpdateTopology.faceFace(mesh);
		expect(mesh.ffFace).not.toBeNull();
		expect(mesh.hasDataMask(MeshElement.MM_FACEFACETOPO)).toBe(true);
	});

	test("is idempotent", () => {
		const { mesh } = torus();
		UpdateTopology.faceFace(mesh);
		const first = Array.from(mesh.ffFace as Int32Array);
		UpdateTopology.faceFace(mesh);
		expect(Array.from(mesh.ffFace as Int32Array)).toEqual(first);
	});

	test("a closed mesh has no border edges", () => {
		const mesh = withTopology(cube());
		for (let f = 0; f < mesh.fn; f++) {
			for (let e = 0; e < 3; e++) expect(mesh.isBorderFF(f, e)).toBe(false);
		}
	});

	test("every edge of a lone triangle is a border", () => {
		const mesh = withTopology(singleTriangle());
		for (let e = 0; e < 3; e++) {
			expect(mesh.isBorderFF(0, e)).toBe(true);
			expect(mesh.ffi(0, e)).toBe(e);
		}
	});

	test("rings are as long as the number of faces on the edge", () => {
		// Three blades share edge (0,1), so that edge's ring has three members
		// while every other edge is a border.
		const mesh = withTopology(nonManifoldEdgeFan(1));
		const sizes: number[] = [];
		for (let f = 0; f < mesh.fn; f++) {
			for (let e = 0; e < 3; e++) sizes.push(UpdateTopology.faceRingSize(mesh, f, e));
		}
		expect(sizes.filter((s) => s === 3)).toHaveLength(3);
		expect(sizes.filter((s) => s === 1)).toHaveLength(6);
	});

	test("isManifoldEdge distinguishes the three ring sizes", () => {
		const closed = withTopology(cube());
		expect(UpdateTopology.isManifoldEdge(closed, 0, 0)).toBe(true);
		const open = withTopology(singleTriangle());
		expect(UpdateTopology.isManifoldEdge(open, 0, 0)).toBe(true);
		const fan = withTopology(nonManifoldEdgeFan(1));
		expect(UpdateTopology.isManifoldEdge(fan, 0, 0)).toBe(false);
	});

	test("a face that shares an edge with itself is not mistaken for a border", () => {
		// Regression: VCGLib tests only `FFp(e) === f` for a border, which
		// holds while no face shares an edge with itself. The degenerate face
		// (v, v, v) links its own three corners into one ring, so the
		// face-only test called all three edges borders. Both the face and the
		// edge index have to match.
		const m = buildMesh([0, 0, 0], [0, 0, 0]);
		UpdateTopology.faceFace(m);
		for (let e = 0; e < 3; e++) {
			expect(m.ffp(0, e)).toBe(0); // same face...
			expect(m.ffi(0, e)).not.toBe(e); // ...but a different edge
			expect(m.isBorderFF(0, e)).toBe(false);
		}
		expect(UpdateTopology.faceRingSize(m, 0, 0)).toBe(3);
		assertFFConsistent(m, "self-edged face");
	});

	test("clearFaceFace makes every edge a border", () => {
		const mesh = withTopology(cube());
		UpdateTopology.clearFaceFace(mesh);
		for (let f = 0; f < mesh.fn; f++) {
			for (let e = 0; e < 3; e++) expect(mesh.isBorderFF(f, e)).toBe(true);
		}
	});
});

describe("UpdateTopology.vertexFace", () => {
	for (const build of ALL_BUILDERS) {
		const built = build();
		test(`${built.name}: VF chains cover every corner exactly once`, () => {
			UpdateTopology.vertexFace(built.mesh);
			// assertAllocatorConsistent runs assertVFConsistent when VF is present.
			assertAllocatorConsistent(built.mesh, built.name);
		});
	}

	test("each cube vertex belongs to the faces that reference it", () => {
		const mesh = withTopology(cube());
		for (let v = 0; v < mesh.vn; v++) {
			const viaVF: number[] = [];
			UpdateTopology.forEachVFCorner(mesh, v, (f) => viaVF.push(f));
			const viaScan: number[] = [];
			for (let f = 0; f < mesh.fn; f++) {
				for (let k = 0; k < 3; k++) if (mesh.fv(f, k) === v) viaScan.push(f);
			}
			expect(viaVF.sort()).toEqual(viaScan.sort());
		}
	});

	test("an unreferenced vertex has an empty chain", () => {
		const m = new CMeshO();
		const v = Allocator.addVertex(m, 0, 0, 0);
		UpdateTopology.vertexFace(m);
		const seen: number[] = [];
		UpdateTopology.forEachVFCorner(m, v, (f) => seen.push(f));
		expect(seen).toEqual([]);
	});
});

describe("Clean.countEdgeNum", () => {
	for (const build of ALL_BUILDERS) {
		const built = build();
		test(`${built.name}: totals match the declared facts`, () => {
			const counts = Clean.countEdgeNum(built.mesh);
			expect(counts.total, `${built.name}.total`).toBe(built.expected.en);
			expect(counts.nonManifold, `${built.name}.nonManifold`).toBe(built.expected.nonManifoldEdges);
			expect(Clean.isWaterTight(built.mesh), `${built.name}.watertight`).toBe(
				built.expected.watertight,
			);
		});
	}

	test("a triangle soup has three boundary edges per triangle", () => {
		const { mesh } = cubeSoup();
		const counts = Clean.countEdgeNum(mesh);
		expect(counts.total).toBe(36);
		expect(counts.boundary).toBe(36);
		expect(counts.nonManifold).toBe(0);
	});
});

describe("Clean: components, holes and genus", () => {
	for (const build of WELL_FORMED_BUILDERS) {
		const built = build();
		test(`${built.name}: components and holes match`, () => {
			const mesh = withTopology(built);
			expect(Clean.countConnectedComponents(mesh), `${built.name}.components`).toBe(
				built.expected.components,
			);
			if (built.expected.boundaryLoops !== undefined) {
				expect(Clean.countHoles(mesh), `${built.name}.boundaryLoops`).toBe(
					built.expected.boundaryLoops,
				);
			}
		});
	}

	test("genus is 0 for a sphere and 1 for a torus", () => {
		for (const [built, wantGenus] of [
			[cube(), 0],
			[tetrahedron(), 0],
			[torus(), 1],
		] as const) {
			const mesh = withTopology(built);
			const { total } = Clean.countEdgeNum(mesh);
			const genus = Clean.meshGenus(
				mesh.vn,
				total,
				mesh.fn,
				Clean.countHoles(mesh),
				Clean.countConnectedComponents(mesh),
			);
			expect(genus, `${built.name}.genus`).toBe(wantGenus);
		}
	});

	test("a disk has genus 0 despite its boundary", () => {
		const built = gridPlane();
		const mesh = withTopology(built);
		const { total } = Clean.countEdgeNum(mesh);
		expect(Clean.meshGenus(mesh.vn, total, mesh.fn, Clean.countHoles(mesh), 1)).toBe(0);
	});

	test("a punctured sphere still has genus 0", () => {
		const built = sphereWithHoles(5);
		const mesh = withTopology(built);
		const { total } = Clean.countEdgeNum(mesh);
		expect(Clean.countHoles(mesh)).toBe(5);
		expect(Clean.meshGenus(mesh.vn, total, mesh.fn, 5, 1)).toBe(0);
	});

	test("disconnected components are counted separately, not merged", () => {
		const mesh = withTopology(bowtieVertex());
		// Sharing only a vertex is not adjacency, so this is two components.
		expect(Clean.countConnectedComponents(mesh)).toBe(2);
	});

	test("countHoles refuses to run without FF adjacency", () => {
		const { mesh } = cube();
		expect(() => Clean.countHoles(mesh)).toThrow(MLInternalException);
		expect(() => Clean.connectedComponents(mesh)).toThrow(MLInternalException);
	});
});

describe("Clean: orientation", () => {
	test("well-built closed meshes are already coherent", () => {
		for (const build of [cube, tetrahedron, torus, gridPlane]) {
			const mesh = withTopology(build());
			expect(Clean.isCoherentlyOrientedMesh(mesh)).toBe(true);
		}
	});

	test("flipping some faces breaks coherence", () => {
		const mesh = withTopology(cubeWithFlippedFaces([0, 5, 9]));
		expect(Clean.isCoherentlyOrientedMesh(mesh)).toBe(false);
	});

	test("orientCoherentlyMesh repairs an orientable mesh", () => {
		const built = cubeWithFlippedFaces([0, 5, 9]);
		const mesh = withTopology(built);
		const result = Clean.orientCoherentlyMesh(mesh);
		expect(result.isOriented).toBe(false); // it was not, before
		expect(result.isOrientable).toBe(true);
		expect(Clean.isCoherentlyOrientedMesh(mesh)).toBe(true);
		assertFFConsistent(mesh, "reoriented cube");
	});

	test("reorienting an already-coherent mesh reports so and changes nothing", () => {
		const built = cube();
		const mesh = withTopology(built);
		const before = Array.from(mesh.faceVert.subarray(0, mesh.fn * 3));
		const result = Clean.orientCoherentlyMesh(mesh);
		expect(result.isOriented).toBe(true);
		expect(result.isOrientable).toBe(true);
		expect(Array.from(mesh.faceVert.subarray(0, mesh.fn * 3))).toEqual(before);
	});

	test("the Möbius strip is reported as not orientable", () => {
		const mesh = withTopology(mobiusStrip());
		const result = Clean.orientCoherentlyMesh(mesh);
		expect(result.isOrientable).toBe(false);
	});

	test("flipMesh reverses every winding and is its own inverse", () => {
		const { mesh } = cube();
		const before = Array.from(mesh.faceVert.subarray(0, mesh.fn * 3));
		Clean.flipMesh(mesh);
		expect(Array.from(mesh.faceVert.subarray(0, mesh.fn * 3))).not.toEqual(before);
		Clean.flipMesh(mesh);
		expect(Array.from(mesh.faceVert.subarray(0, mesh.fn * 3))).toEqual(before);
	});
});

describe("Clean.countNonManifoldVertexFF", () => {
	test("a bowtie has exactly one non-manifold vertex", () => {
		expect(Clean.countNonManifoldVertexFF(bowtieVertex().mesh)).toBe(1);
	});

	test("well-formed meshes have none", () => {
		for (const build of [cube, tetrahedron, torus, gridPlane, mobiusStrip]) {
			const built = build();
			expect(Clean.countNonManifoldVertexFF(built.mesh), built.name).toBe(0);
		}
	});
});

describe("UpdateFlags", () => {
	test("border bits from FF and from scratch agree", () => {
		for (const build of [gridPlane, cube, () => sphereWithHoles(5), singleTriangle]) {
			const built = build();
			const mesh = withTopology(built);
			UpdateFlags.faceBorderFromFF(mesh);
			const fromFF: boolean[] = [];
			for (let f = 0; f < mesh.fn; f++) for (let e = 0; e < 3; e++) fromFF.push(mesh.isFaceB(f, e));

			UpdateFlags.faceBorderFromNone(mesh);
			const fromNone: boolean[] = [];
			for (let f = 0; f < mesh.fn; f++)
				for (let e = 0; e < 3; e++) fromNone.push(mesh.isFaceB(f, e));

			expect(fromNone, built.name).toEqual(fromFF);
		}
	});

	test("clearing flags never resurrects a deleted element", () => {
		const { mesh } = cube();
		Allocator.deleteFace(mesh, 0);
		Allocator.deleteVertex(mesh, 6);
		UpdateFlags.clear(mesh);
		expect(mesh.isFaceD(0)).toBe(true);
		expect(mesh.isVertD(6)).toBe(true);
		expect(mesh.fn).toBe(11);
		expect(mesh.vn).toBe(7);
	});

	test("vertex border bits agree between the two derivations", () => {
		const built = gridPlane();
		const mesh = withTopology(built);
		UpdateFlags.vertexBorderFromNone(mesh);
		const fromNone = Array.from({ length: mesh.vn }, (_, v) => mesh.isVertB(v));
		UpdateFlags.faceBorderFromFF(mesh);
		UpdateFlags.vertexBorderFromFace(mesh);
		const fromFace = Array.from({ length: mesh.vn }, (_, v) => mesh.isVertB(v));
		expect(fromFace).toEqual(fromNone);
		// A 4x3 grid's boundary is its outer ring: 20 - 6 interior = 14.
		expect(fromNone.filter(Boolean)).toHaveLength(14);
	});
});
