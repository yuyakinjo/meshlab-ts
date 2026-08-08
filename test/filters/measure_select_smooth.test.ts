/**
 * filter_measure, filter_select and filter_unsharp.
 *
 * The measure tests are the sharpest in the suite: every value has a closed
 * form, so they check the number rather than a range.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	geometryDigest,
	signedVolume,
	surfaceArea,
} from "../helpers/invariants.ts";
import {
	type BuiltMesh,
	bowtieVertex,
	cube,
	cubeWithHoles,
	gridPlane,
	icosahedron,
	nonManifoldEdgeFan,
	octahedron,
	sphereIcosa,
	tetrahedron,
	torus,
	unreferencedVerts,
} from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function apply(
	built: BuiltMesh | CMeshO,
	name: string,
	params: Record<string, unknown> = {},
): { mesh: CMeshO; doc: MeshDocument; out: Record<string, unknown> } {
	const cm = "mesh" in built ? built.mesh : built;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("test", "test", true, cm);
	m.updateBoxAndNormals();
	const out = kernel.applyFilter(doc, name, params);
	return { mesh: m.cm, doc, out };
}

describe("Compute Topological Measures", () => {
	const NAME = "Compute Topological Measures";

	test("reports the counts and genus of well-known solids", () => {
		for (const [built, chi, genus] of [
			[tetrahedron(), 2, 0],
			[cube(), 2, 0],
			[octahedron(), 2, 0],
			[icosahedron(), 2, 0],
			[torus(2, 0.6, 12, 8), 0, 1],
		] as const) {
			const { out } = apply(built, NAME);
			const v = out.vertices_number as number;
			const e = out.edges_number as number;
			const f = out.faces_number as number;
			expect(v - e + f, `${built.name} chi`).toBe(chi);
			expect(out.genus, `${built.name} genus`).toBe(genus);
			expect(out.number_holes, built.name).toBe(0);
			expect(out.boundary_edges, built.name).toBe(0);
			expect(out.connected_components_number, built.name).toBe(1);
			expect(out.is_mesh_two_manifold, built.name).toBe(true);
		}
	});

	test("counts boundary edges and holes on an open mesh", () => {
		const { out } = apply(gridPlane(4, 3), NAME);
		expect(out.number_holes).toBe(1);
		expect(out.boundary_edges).toBe(14);
		expect(out.genus).toBe(0);
	});

	test("a punctured cube reports its holes", () => {
		const { out } = apply(cubeWithHoles(2), NAME);
		expect(out.number_holes).toBe(2);
		expect(out.genus).toBe(0);
	});

	test("counts unreferenced vertices without removing them", () => {
		const { mesh, out } = apply(unreferencedVerts(5), NAME);
		expect(out.unreferenced_vertices).toBe(5);
		expect(mesh.vn).toBe(13); // measuring must not modify
	});

	test("reports -1 for genus rather than a wrong number", () => {
		// Genus needs a two-manifold surface. Upstream returns -1 instead of a
		// value derived from a formula that does not apply.
		const { out } = apply(nonManifoldEdgeFan(1), NAME);
		expect(out.is_mesh_two_manifold).toBe(false);
		expect(out.non_two_manifold_edges).toBe(1);
		expect(out.genus).toBe(-1);
		expect(out.number_holes).toBe(-1);
	});

	test("finds a bowtie's non-manifold vertex", () => {
		const { out } = apply(bowtieVertex(), NAME);
		expect(out.non_two_manifold_vertices).toBe(1);
		expect(out.is_mesh_two_manifold).toBe(false);
		expect(out.connected_components_number).toBe(2);
	});

	test("the mesh is untouched", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh } = apply(cube(), NAME);
		expect(geometryDigest(mesh)).toBe(before);
	});
});

describe("Compute Geometric Measures", () => {
	const NAME = "Compute Geometric Measures";

	test("area and volume match the closed forms exactly", () => {
		for (const built of [tetrahedron(), cube(1), cube(2.5), octahedron(), icosahedron()]) {
			const { out } = apply(built, NAME);
			expect(out.surface_area as number, `${built.name} area`).toBeCloseTo(
				built.expected.area as number,
				9,
			);
			expect(out.mesh_volume as number, `${built.name} volume`).toBeCloseTo(
				built.expected.volume as number,
				9,
			);
		}
	});

	test("the inertia tensor of a cube matches s^5/6", () => {
		for (const s of [1, 2, 3]) {
			const { out } = apply(cube(s), NAME);
			const t = out.inertia_tensor as number[];
			const want = s ** 5 / 6;
			// A cube about its centre: diagonal s^5/6, products of inertia zero.
			expect(t[0], `cube(${s}) Ixx`).toBeCloseTo(want, 9);
			expect(t[4], `cube(${s}) Iyy`).toBeCloseTo(want, 9);
			expect(t[8], `cube(${s}) Izz`).toBeCloseTo(want, 9);
			for (const off of [t[1], t[2], t[3], t[5], t[6], t[7]]) {
				expect(off, `cube(${s}) off-diagonal`).toBeCloseTo(0, 9);
			}
		}
	});

	test("a sphere converges to 2/5 M r^2", () => {
		const { out } = apply(sphereIcosa(5), NAME);
		expect(out.mesh_volume as number).toBeCloseTo((4 * Math.PI) / 3, 2);
		// I = 2/5 M r² with M = 4π/3 and r = 1.
		expect((out.inertia_tensor as number[])[0]).toBeCloseTo((8 * Math.PI) / 15, 2);
	});

	test("the centre of mass of a centred solid is the origin", () => {
		for (const built of [cube(2), sphereIcosa(3), octahedron()]) {
			const c = apply(built, NAME).out.center_of_mass as number[];
			for (const x of c) expect(x, built.name).toBeCloseTo(0, 9);
		}
	});

	test("volume is omitted for an open surface, where it has no meaning", () => {
		// A grid has no interior; reporting a number from its boundary would
		// be worse than reporting nothing.
		const { out } = apply(gridPlane(), NAME);
		expect(out.mesh_volume).toBeUndefined();
		expect(out.center_of_mass).toBeUndefined();
		expect(out.inertia_tensor).toBeUndefined();
		// Area still applies.
		expect(out.surface_area as number).toBeCloseTo(1, 9);
	});

	test("the bounding box and diagonal are right", () => {
		const { out } = apply(cube(4), NAME);
		expect(out.bbox_min).toEqual([-2, -2, -2]);
		expect(out.bbox_max).toEqual([2, 2, 2]);
		expect(out.bbox_diagonal as number).toBeCloseTo(Math.sqrt(48), 9);
	});

	test("edge length statistics are right for a cube", () => {
		// A cube of side 1 split into triangles: 12 edges of length 1 and 6
		// face diagonals of length sqrt(2).
		const { out } = apply(cube(1), NAME);
		expect(out.total_edge_length as number).toBeCloseTo(12 + 6 * Math.SQRT2, 9);
		expect(out.avg_edge_length as number).toBeCloseTo((12 + 6 * Math.SQRT2) / 18, 9);
	});

	test("flipping the mesh flips the sign of the volume", () => {
		const flipped = cube(2).mesh;
		Clean.flipMesh(flipped);
		expect(apply(flipped, NAME).out.mesh_volume as number).toBeCloseTo(-8, 9);
	});
});

describe("filter_select", () => {
	test("all eleven are registered as Selection", () => {
		for (const name of [
			"Select All",
			"Select None",
			"Invert Selection",
			"Select Border",
			"Select Faces from Vertices",
			"Select Vertices from Faces",
			"Delete Selected Faces",
			"Delete Selected Vertices",
			"Delete Selected Faces and Vertices",
			"Select non Manifold Edges",
			"Select non Manifold Vertices",
		]) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toBe("Selection");
		}
	});

	test("Select All then Select None round-trips", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		let out = kernel.applyFilter(doc, "Select All");
		expect(out.selected_faces).toBe(12);
		expect(out.selected_vertices).toBe(8);
		out = kernel.applyFilter(doc, "Select None");
		expect(out.selected_faces).toBe(0);
		expect(out.selected_vertices).toBe(0);
	});

	test("Invert Selection is its own inverse", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		m.cm.faceFlags[0] |= 0x20;
		kernel.applyFilter(doc, "Invert Selection");
		expect(m.cm.isFaceS(0)).toBe(false);
		kernel.applyFilter(doc, "Invert Selection");
		expect(m.cm.isFaceS(0)).toBe(true);
	});

	test("Select Border finds an open mesh's edge and nothing on a closed one", () => {
		const open = apply(gridPlane(4, 3), "Select Border");
		expect(open.out.selected_vertices).toBe(14);
		const closed = apply(cube(), "Select Border");
		expect(closed.out.selected_vertices).toBe(0);
		expect(closed.out.selected_faces).toBe(0);
	});

	test("Select non Manifold Edges finds the fan, and a clean mesh has none", () => {
		const bad = apply(nonManifoldEdgeFan(1), "Select non Manifold Edges");
		expect(bad.out.non_manifold_edges).toBe(1);
		expect(bad.out.selected_faces).toBe(3);
		const good = apply(cube(), "Select non Manifold Edges");
		expect(good.out.non_manifold_edges).toBe(0);
	});

	test("Select non Manifold Vertices finds a bowtie apex", () => {
		const { out } = apply(bowtieVertex(), "Select non Manifold Vertices");
		expect(out.non_manifold_vertices).toBe(1);
		expect(out.selected_vertices).toBe(1);
	});

	test("selection transfers both ways", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		m.cm.faceFlags[0] |= 0x20;
		kernel.applyFilter(doc, "Select Vertices from Faces");
		// Face 0 is (4,5,6), so exactly three vertices.
		expect([4, 5, 6].every((v) => m.cm.isVertS(v))).toBe(true);
		const back = kernel.applyFilter(doc, "Select Faces from Vertices");
		// Strict transfer: only faces with all three corners selected.
		expect(back.selected_faces).toBe(1);
	});

	test("Delete Selected Faces leaves the vertices behind", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		m.cm.faceFlags[0] |= 0x20;
		m.cm.faceFlags[3] |= 0x20;
		const out = kernel.applyFilter(doc, "Delete Selected Faces");
		expect(out.removed_faces).toBe(2);
		expect(m.cm.fn).toBe(10);
		expect(m.cm.vn).toBe(8); // deliberately not cleaned up
		assertAllocatorConsistent(m.cm);
	});

	test("Delete Selected Faces and Vertices cleans up after itself", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		// Both faces of the +z side, which frees no vertex (all are shared).
		for (const f of [0, 1]) m.cm.faceFlags[f] |= 0x20;
		const out = kernel.applyFilter(doc, "Delete Selected Faces and Vertices");
		expect(out.removed_faces).toBe(2);
		expect(m.cm.fn).toBe(10);
		assertAllocatorConsistent(m.cm);
	});

	test("Delete Selected Vertices takes the faces around them", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		m.cm.vertFlags[0] |= 0x20;
		const out = kernel.applyFilter(doc, "Delete Selected Vertices");
		expect(out.removed_vertices).toBe(1);
		expect(out.removed_faces as number).toBeGreaterThan(0);
		expect(m.cm.vn).toBe(7);
		assertAllocatorConsistent(m.cm);
	});

	test("a pure selection change does not invalidate adjacency", () => {
		// Selecting touches only flag bits, so a caller mid-traversal should
		// not lose the FF rings it is walking.
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube().mesh);
		m.updateBoxAndNormals();
		m.updateDataMask(0x00040000); // MM_FACEFACETOPO
		kernel.applyFilter(doc, "Select All");
		expect(m.cm.ffFace).not.toBeNull();
	});
});

describe("filter_unsharp: smoothing", () => {
	const ALL = [
		"Laplacian Smooth",
		"Taubin Smooth",
		"HC Laplacian Smooth",
		"ScaleDependent Laplacian Smooth",
	] as const;

	test("all four are registered as Smoothing", () => {
		for (const name of ALL) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toBe("Smoothing");
		}
	});

	test("upstream's defaults", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, sphereIcosa(2).mesh);
		m.updateBoxAndNormals();
		const lap = kernel.initParameterList("Laplacian Smooth", doc);
		expect(lap.getInt("stepSmoothNum")).toBe(3);
		expect(lap.getBool("Boundary")).toBe(true);
		expect(lap.getBool("cotangentWeight")).toBe(true);
		const taubin = kernel.initParameterList("Taubin Smooth", doc);
		expect(taubin.getFloat("lambda")).toBe(0.5);
		expect(taubin.getFloat("mu")).toBe(-0.53);
		expect(taubin.getInt("stepSmoothNum")).toBe(10);
	});

	test("none of them changes the topology", () => {
		for (const name of ALL) {
			const built = sphereIcosa(2);
			const before = computeFacts(built.mesh);
			const { mesh } = apply(built, name);
			const after = computeFacts(mesh);
			expect(after.vn, name).toBe(before.vn);
			expect(after.fn, name).toBe(before.fn);
			expect(after.genus, name).toBe(0);
			expect(after.watertight, name).toBe(true);
			assertAllocatorConsistent(mesh, name);
		}
	});

	test("Laplacian smoothing shrinks a closed surface", () => {
		// Inherent to the operation, and the reason Taubin exists.
		const original = sphereIcosa(3).mesh;
		const { mesh } = apply(sphereIcosa(3), "Laplacian Smooth", {
			stepSmoothNum: 10,
			cotangentWeight: false,
		});
		expect(signedVolume(mesh)).toBeLessThan(signedVolume(original));
	});

	test("Taubin smoothing preserves volume far better than Laplacian", () => {
		const original = signedVolume(sphereIcosa(3).mesh);
		const laplacian = signedVolume(
			apply(sphereIcosa(3), "Laplacian Smooth", { stepSmoothNum: 10, cotangentWeight: false }).mesh,
		);
		const taubin = signedVolume(apply(sphereIcosa(3), "Taubin Smooth", { stepSmoothNum: 10 }).mesh);
		const lapLoss = Math.abs(original - laplacian);
		const taubinLoss = Math.abs(original - taubin);
		// The whole point of the lambda/mu pair.
		expect(taubinLoss).toBeLessThan(lapLoss);
	});

	test("smoothing a sphere keeps it near the sphere", () => {
		for (const name of ALL) {
			const { mesh } = apply(sphereIcosa(3), name);
			for (let v = 0; v < mesh.vn; v++) {
				const r = Math.hypot(mesh.vx(v), mesh.vy(v), mesh.vz(v));
				expect(r, `${name} radius`).toBeGreaterThan(0.8);
				expect(r, `${name} radius`).toBeLessThan(1.2);
			}
		}
	});

	test("Boundary switches between 1D outline smoothing and full smoothing", () => {
		// Neither setting pins the outline — that was this test's old premise,
		// and MeshLab does not do it. Boundary=true smooths the outline along
		// its own curve, which rounds the square's corners off; Boundary=false
		// clears the border flags and the outline is averaged with the
		// interior, shrinking the whole sheet. The expected numbers are what
		// real PyMeshLab (2025.7.post1) measures on this same grid, and the
		// tolerance is its float32 storage.
		const along = apply(gridPlane(6, 6), "Laplacian Smooth", {
			stepSmoothNum: 20,
			Boundary: true,
			cotangentWeight: false,
		});
		expect(along.mesh.bbox.dimX).toBeCloseTo(0.811923, 5);
		expect(along.mesh.bbox.dimY).toBeCloseTo(0.811923, 5);

		const across = apply(gridPlane(6, 6), "Laplacian Smooth", {
			stepSmoothNum: 20,
			Boundary: false,
			cotangentWeight: false,
		});
		expect(across.mesh.bbox.dimX).toBeCloseTo(0.343293, 5);
		expect(across.mesh.bbox.dimY).toBeCloseTo(0.343293, 5);
	});

	test("smoothing never produces NaN", () => {
		for (const name of ALL) {
			const { mesh } = apply(sphereIcosa(2), name, {});
			for (let v = 0; v < mesh.vn; v++) {
				expect(Number.isFinite(mesh.vx(v)), name).toBe(true);
				expect(Number.isFinite(mesh.vy(v)), name).toBe(true);
				expect(Number.isFinite(mesh.vz(v)), name).toBe(true);
			}
		}
	});

	test("zero steps is a no-op", () => {
		const before = geometryDigest(sphereIcosa(2).mesh);
		const { mesh } = apply(sphereIcosa(2), "Laplacian Smooth", { stepSmoothNum: 0 });
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("smoothing a degenerate mesh does not crash", () => {
		for (const name of ALL) {
			const doc = new MeshDocument();
			doc.addNewMesh("t", "t");
			expect(() => kernel.applyFilter(doc, name), name).not.toThrow();
		}
	});

	test("area shrinks under smoothing but stays positive", () => {
		for (const name of ALL) {
			const { mesh } = apply(sphereIcosa(3), name);
			const a = surfaceArea(mesh);
			expect(a, name).toBeGreaterThan(0);
			expect(a, name).toBeLessThan(surfaceArea(sphereIcosa(3).mesh) * 1.05);
		}
	});
});
