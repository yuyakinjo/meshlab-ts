/**
 * filter_clean, against the per-filter checklist.
 *
 * The recurring shape: build a mesh whose defect is known by construction,
 * apply the filter, and assert the defect is gone *and* that nothing else
 * changed — cleaning must not quietly alter the surface it is repairing.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../../src/common/ml_document/mesh_document.ts";
import { filterClassToString } from "../../../src/common/plugins/filter_class.ts";
import { InvalidParameterException } from "../../../src/common/utilities/ml_exception.ts";
import { Clean } from "../../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import { UpdateTopology } from "../../../src/vcg/complex/update/topology.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	countNonManifoldEdges,
	geometryDigest,
	signedVolume,
	surfaceArea,
} from "../../helpers/invariants.ts";
import {
	type BuiltMesh,
	bowtieVertex,
	buildMesh,
	cube,
	cubePlusIslands,
	cubeSoup,
	emptyMesh,
	nonManifoldEdgeFan,
	singleTriangle,
	sphereIcosa,
	torus,
	unreferencedVerts,
} from "../../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** Runs a filter on a standalone mesh and hands back the document's mesh. */
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

/** Every filter this plugin provides, for the sweeps that apply to all. */
const ALL_CLEAN_FILTERS = [
	"Remove Duplicate Vertices",
	"Remove Duplicate Faces",
	"Remove Unreferenced Vertices",
	"Remove Zero Area Faces",
	"Merge Close Vertices",
	"Remove Isolated pieces (wrt Face Num.)",
	"Remove Isolated pieces (wrt Diameter)",
	"Repair non Manifold Edges",
	"Repair non Manifold Vertices by splitting",
	"Remove T-Vertices",
	"Remove Isolated Folded Faces by Edge Flip",
] as const;

describe("filter_clean: registry", () => {
	test("all eleven filters are implemented and classed as Cleaning", () => {
		for (const name of ALL_CLEAN_FILTERS) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toBe("Cleaning");
			expect(a.plugin.pluginName()).toBe("FilterClean");
		}
	});

	test("PyMeshLab names match upstream", () => {
		for (const [display, python] of [
			["Remove Duplicate Vertices", "meshing_remove_duplicate_vertices"],
			["Remove Zero Area Faces", "meshing_remove_null_faces"],
			["Merge Close Vertices", "meshing_merge_close_vertices"],
			["Repair non Manifold Edges", "meshing_repair_non_manifold_edges"],
			[
				"Remove Isolated pieces (wrt Face Num.)",
				"meshing_remove_connected_component_by_face_number",
			],
		] as const) {
			expect(kernel.filterAction(display).pythonName, display).toBe(python);
			// Both spellings must reach the same filter.
			expect(kernel.filterAction(python).name).toBe(display);
		}
	});

	test("parameter names and defaults match upstream", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube(10).mesh);
		m.updateBoxAndNormals();

		const size = kernel.initParameterList("Remove Isolated pieces (wrt Face Num.)", doc);
		expect(size.getInt("MinComponentSize")).toBe(25);
		expect(size.getBool("removeUnref")).toBe(true);

		const tvert = kernel.initParameterList("Remove T-Vertices", doc);
		expect(tvert.getEnum("method")).toBe(0);
		expect(tvert.getFloat("Threshold")).toBe(40);
		expect(tvert.getBool("Repeat")).toBe(true);

		const nonManif = kernel.initParameterList("Repair non Manifold Vertices by splitting", doc);
		expect(nonManif.getFloat("VertDispRatio")).toBe(0);

		// Bounding-box relative defaults: a cube of side 10 has diagonal 10√3.
		const diag = Math.sqrt(300);
		const merge = kernel.initParameterList("Merge Close Vertices", doc);
		expect(merge.getAbsPerc("Threshold")).toBeCloseTo(diag / 10000, 9);
		const dia = kernel.initParameterList("Remove Isolated pieces (wrt Diameter)", doc);
		expect(dia.getAbsPerc("MinComponentDiag")).toBeCloseTo(diag / 10, 9);
	});

	test("an unknown parameter is rejected", () => {
		expect(() =>
			apply(cube(), "Remove Isolated pieces (wrt Face Num.)", { MinComponentSizze: 3 }),
		).toThrow(InvalidParameterException);
	});
});

describe("Remove Duplicate Vertices", () => {
	const NAME = "Remove Duplicate Vertices";

	test("welds an STL-style triangle soup back into a solid", () => {
		const before = cube(2);
		const soup = cubeSoup(2);
		expect(soup.mesh.vn).toBe(36);

		const { mesh, out } = apply(soup, NAME);
		expect(out.removedVertices).toBe(28); // 36 -> 8
		expect(mesh.vn).toBe(8);
		expect(mesh.fn).toBe(12);
		// The surface is unchanged; only the sharing is.
		expect(geometryDigest(mesh)).toBe(geometryDigest(before.mesh));
		expect(Clean.isWaterTight(mesh)).toBe(true);
		assertAllocatorConsistent(mesh);
	});

	test("a welded mesh is left alone", () => {
		const built = cube();
		const before = geometryDigest(built.mesh);
		const { mesh, out } = apply(built, NAME);
		expect(out.removedVertices).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("is idempotent", () => {
		const { mesh } = apply(cubeSoup(), NAME);
		const once = geometryDigest(mesh);
		const { mesh: twice } = apply(mesh, NAME);
		expect(geometryDigest(twice)).toBe(once);
	});

	test("faces that collapse as a result are removed", () => {
		// Two coincident vertices make the second face degenerate once welded.
		const m = buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 1, 2, 1, 3, 2]);
		const { mesh } = apply(m, NAME);
		expect(mesh.vn).toBe(3);
		expect(mesh.fn).toBe(1);
		assertAllocatorConsistent(mesh);
	});

	test("survives an empty mesh and a single triangle", () => {
		expect(apply(emptyMesh(), NAME).mesh.vn).toBe(0);
		expect(apply(singleTriangle(), NAME).mesh.fn).toBe(1);
	});
});

describe("Remove Unreferenced Vertices", () => {
	const NAME = "Remove Unreferenced Vertices";

	test("removes the strays and restores the Euler characteristic", () => {
		const built = unreferencedVerts(5);
		expect(built.mesh.vn).toBe(13);
		const { mesh, out } = apply(built, NAME);
		expect(out.removedVertices).toBe(5);
		expect(mesh.vn).toBe(8);
		// Stray vertices were what pushed chi away from 2.
		expect(computeFacts(mesh).chi).toBe(2);
		expect(computeFacts(mesh).genus).toBe(0);
	});

	test("touches nothing when every vertex is used", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), NAME);
		expect(out.removedVertices).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});
});

describe("Remove Zero Area Faces", () => {
	const NAME = "Remove Zero Area Faces";

	test("removes collinear and repeated-vertex faces", () => {
		// One good triangle, one collinear, one with a repeated vertex.
		const m = buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0], [0, 1, 2, 0, 1, 3, 0, 1, 1]);
		const { mesh, out } = apply(m, NAME);
		expect(out.removedFaces).toBe(2);
		expect(mesh.fn).toBe(1);
		expect(surfaceArea(mesh)).toBeCloseTo(0.5, 12);
	});

	test("leaves a healthy mesh untouched", () => {
		const { mesh, out } = apply(sphereIcosa(2), NAME);
		expect(out.removedFaces).toBe(0);
		expect(mesh.fn).toBe(320);
	});
});

describe("Remove Duplicate Faces", () => {
	const NAME = "Remove Duplicate Faces";

	test("removes repeats regardless of winding order", () => {
		const m = buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2, 0, 1, 2, 2, 1, 0]);
		const { mesh, out } = apply(m, NAME);
		expect(out.removedFaces).toBe(2);
		expect(mesh.fn).toBe(1);
	});

	test("leaves distinct faces alone", () => {
		const { out } = apply(cube(), NAME);
		expect(out.removedFaces).toBe(0);
	});
});

describe("Merge Close Vertices", () => {
	const NAME = "Merge Close Vertices";

	test("welds vertices within the threshold and leaves the rest", () => {
		// Two triangles whose shared edge is split by a hair.
		const eps = 1e-6;
		const m = buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0, eps, 0, 0, 1, 1, 0], [0, 1, 2, 3, 4, 1]);
		const { mesh, out } = apply(m, NAME, { Threshold: 1e-3 });
		expect(out.mergedVertices as number).toBeGreaterThan(0);
		expect(mesh.vn).toBe(4);
		assertAllocatorConsistent(mesh);
	});

	test("a threshold of zero is a no-op", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), NAME, { Threshold: 0 });
		expect(out.mergedVertices).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("a threshold below every gap changes nothing", () => {
		const { mesh } = apply(sphereIcosa(2), NAME, { Threshold: 1e-9 });
		expect(mesh.vn).toBe(162);
	});
});

describe("Remove Isolated pieces", () => {
	test("by face count: the islands go, the body stays", () => {
		const built = cubePlusIslands(3);
		expect(computeFacts(built.mesh).components).toBe(4);
		const { mesh, out } = apply(built, "Remove Isolated pieces (wrt Face Num.)", {
			MinComponentSize: 5,
		});
		expect(out.totalComponents).toBe(4);
		expect(out.deletedComponents).toBe(3);
		expect(mesh.fn).toBe(12);
		expect(mesh.vn).toBe(8); // removeUnref cleaned up after the deletions
		expect(computeFacts(mesh).genus).toBe(0);
		assertAllocatorConsistent(mesh);
	});

	test("by face count: a threshold below every component keeps everything", () => {
		const { mesh, out } = apply(cubePlusIslands(3), "Remove Isolated pieces (wrt Face Num.)", {
			MinComponentSize: 1,
		});
		expect(out.deletedComponents).toBe(0);
		expect(mesh.fn).toBe(24);
	});

	test("by diameter: size in space, not in triangle count", () => {
		const built = cubePlusIslands(3, 0.05);
		const { mesh, out } = apply(built, "Remove Isolated pieces (wrt Diameter)", {
			MinComponentDiag: 0.5,
		});
		expect(out.deletedComponents).toBe(3);
		expect(mesh.fn).toBe(12);
	});

	test("removeUnref=false leaves the orphaned vertices behind", () => {
		const { mesh, out } = apply(cubePlusIslands(3), "Remove Isolated pieces (wrt Face Num.)", {
			MinComponentSize: 5,
			removeUnref: false,
		});
		expect(out.removedVertices).toBe(0);
		expect(mesh.vn).toBe(20); // 8 + 3 islands x 4, all still present
	});
});

describe("Repair non Manifold Edges", () => {
	const NAME = "Repair non Manifold Edges";

	test("removing faces makes the edge manifold, smallest face first", () => {
		const built = nonManifoldEdgeFan(1); // three blades on one edge
		expect(countNonManifoldEdges(built.mesh)).toBe(1);
		const { mesh, out } = apply(built, NAME, { method: "Remove Faces" });
		expect(out.removedFaces).toBe(1);
		expect(mesh.fn).toBe(2);
		expect(countNonManifoldEdges(mesh)).toBe(0);
		assertAllocatorConsistent(mesh);
	});

	test("a wider fan loses exactly enough blades", () => {
		const built = nonManifoldEdgeFan(3); // five blades
		const { mesh } = apply(built, NAME, { method: 0 });
		expect(countNonManifoldEdges(mesh)).toBe(0);
		expect(mesh.fn).toBe(2);
	});

	test("splitting vertices keeps every face instead", () => {
		const built = nonManifoldEdgeFan(1);
		const faces = built.mesh.fn;
		const { mesh, out } = apply(built, NAME, { method: "Split Vertices" });
		expect(mesh.fn).toBe(faces); // nothing deleted
		expect(countNonManifoldEdges(mesh)).toBe(0);
		expect(out.components as number).toBeGreaterThan(1);
		assertAllocatorConsistent(mesh);
	});

	test("a manifold mesh is untouched by either method", () => {
		for (const method of [0, 1]) {
			const before = geometryDigest(cube().mesh);
			const { mesh } = apply(cube(), NAME, { method });
			expect(geometryDigest(mesh), `method ${method}`).toBe(before);
		}
	});
});

describe("Repair non Manifold Vertices by splitting", () => {
	const NAME = "Repair non Manifold Vertices by splitting";

	test("a bowtie apex is split into one vertex per fan", () => {
		const built = bowtieVertex();
		expect(Clean.countNonManifoldVertexFF(built.mesh)).toBe(1);
		const { mesh, out } = apply(built, NAME);
		expect(out.splitVertices).toBe(1);
		expect(mesh.vn).toBe(6); // the apex became two
		expect(mesh.fn).toBe(2); // no face was harmed
		expect(Clean.countNonManifoldVertexFF(mesh)).toBe(0);
		expect(surfaceArea(mesh)).toBeCloseTo(surfaceArea(built.mesh), 9);
		assertAllocatorConsistent(mesh);
	});

	test("VertDispRatio pulls the copies apart", () => {
		const { mesh } = apply(bowtieVertex(), NAME, { VertDispRatio: 0.1 });
		// With a displacement the two copies no longer coincide.
		const apexes = [0, 1, 2, 3, 4, 5]
			.filter((v) => v < mesh.vn)
			.map((v) => `${mesh.vx(v)},${mesh.vy(v)},${mesh.vz(v)}`);
		expect(new Set(apexes).size).toBe(mesh.vn);
	});

	test("a clean mesh is untouched", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), NAME);
		expect(out.splitVertices).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});
});

describe("Remove T-Vertices", () => {
	const NAME = "Remove T-Vertices";

	/** A quad split so that one triangle is a needle-thin sliver. */
	function sliverStrip(): CMeshO {
		return buildMesh([0, 0, 0, 10, 0, 0, 10, 0.01, 0, 0, 5, 0], [0, 1, 2, 0, 2, 3]);
	}

	test("collapse removes the sliver", () => {
		const before = sliverStrip();
		const { mesh, out } = apply(sliverStrip(), NAME, { method: "Edge Collapse", Threshold: 40 });
		expect(out.removedTVertices as number).toBeGreaterThan(0);
		expect(mesh.fn).toBeLessThan(before.fn);
		assertAllocatorConsistent(mesh);
	});

	test("flip keeps the face count", () => {
		const { mesh, out } = apply(sliverStrip(), NAME, { method: "Edge Flip", Threshold: 40 });
		expect(out.removedTVertices as number).toBeGreaterThan(0);
		// Flipping rewires rather than removes.
		expect(mesh.fn).toBe(2);
		assertAllocatorConsistent(mesh);
	});

	test("a well-shaped mesh is untouched by either method", () => {
		for (const method of ["Edge Collapse", "Edge Flip"]) {
			const before = geometryDigest(sphereIcosa(1).mesh);
			const { mesh, out } = apply(sphereIcosa(1), NAME, { method, Threshold: 40 });
			expect(out.removedTVertices, method).toBe(0);
			expect(geometryDigest(mesh), method).toBe(before);
		}
	});
});

describe("Remove Isolated Folded Faces by Edge Flip", () => {
	const NAME = "Remove Isolated Folded Faces by Edge Flip";

	test("a flat mesh has nothing to flip", () => {
		const before = geometryDigest(sphereIcosa(1).mesh);
		const { mesh, out } = apply(sphereIcosa(1), NAME);
		expect(out.flippedFaces).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});
});

describe("filter_clean: general properties", () => {
	test("no filter increases non-manifoldness", () => {
		for (const name of ALL_CLEAN_FILTERS) {
			for (const build of [cubeSoup, bowtieVertex, () => nonManifoldEdgeFan(1), () => cube()]) {
				const built = build();
				const before = countNonManifoldEdges(built.mesh);
				const { mesh } = apply(built, name);
				expect(countNonManifoldEdges(mesh), `${name} on ${built.name}`).toBeLessThanOrEqual(before);
			}
		}
	});

	test("no filter crashes on degenerate input", () => {
		for (const name of ALL_CLEAN_FILTERS) {
			for (const build of [emptyMesh, singleTriangle]) {
				const built = build();
				const { mesh } = apply(built, name);
				assertAllocatorConsistent(mesh, `${name} on ${built.name}`);
			}
		}
	});

	test("every filter leaves a healthy closed mesh healthy", () => {
		// Meshes comfortably larger than MinComponentSize's default of 25, so
		// that the isolated-piece filters have nothing to legitimately remove
		// — see the test below for what happens when they do.
		for (const name of ALL_CLEAN_FILTERS) {
			for (const build of [() => sphereIcosa(2), () => torus(2, 0.6, 12, 8)]) {
				const built = build();
				const wantVolume = signedVolume(built.mesh);
				const { mesh } = apply(built, name);
				const facts = computeFacts(mesh);
				expect(facts.watertight, `${name} on ${built.name}`).toBe(true);
				expect(facts.nonManifoldEdges, `${name} on ${built.name}`).toBe(0);
				expect(signedVolume(mesh), `${name} on ${built.name}`).toBeCloseTo(wantVolume, 6);
			}
		}
	});

	test("a whole mesh under MinComponentSize is itself an isolated piece", () => {
		// Surprising but correct, and worth pinning: the default threshold is
		// 25 faces, so running this filter on a 12-face cube deletes the cube.
		// MeshLab behaves the same way — the filter has no concept of a "main"
		// component, only of components smaller than the threshold.
		const { mesh, out } = apply(cube(), "Remove Isolated pieces (wrt Face Num.)");
		expect(out.totalComponents).toBe(1);
		expect(out.deletedComponents).toBe(1);
		expect(mesh.fn).toBe(0);

		// With a threshold below the component's size it survives intact.
		const kept = apply(cube(), "Remove Isolated pieces (wrt Face Num.)", {
			MinComponentSize: 12,
		});
		expect(kept.out.deletedComponents).toBe(0);
		expect(kept.mesh.fn).toBe(12);
	});

	test("the document records what ran, and the mesh is marked modified", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cubeSoup().mesh);
		m.updateBoxAndNormals();
		kernel.applyFilter(doc, "Remove Duplicate Vertices");
		expect(doc.filterHistory).toHaveLength(1);
		expect(doc.filterHistory[0].filterName).toBe("Remove Duplicate Vertices");
		expect(m.meshModified()).toBe(true);
		expect(doc.Log.messages()[0]).toContain("Removed 28 duplicated vertices");
	});

	test("stale adjacency is dropped after a topology change", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cubeSoup().mesh);
		m.updateBoxAndNormals();
		UpdateTopology.faceFace(m.cm);
		expect(m.cm.ffFace).not.toBeNull();
		kernel.applyFilter(doc, "Remove Duplicate Vertices");
		// The welding invalidated it, so it must not still be lying around.
		expect(m.cm.ffFace).toBeNull();
	});

	test("the bounding box is current afterwards", () => {
		const { mesh } = apply(cubePlusIslands(3), "Remove Isolated pieces (wrt Face Num.)", {
			MinComponentSize: 5,
		});
		// Only the unit cube survives, so the box shrinks back to it.
		expect(mesh.bbox.max[0]).toBeCloseTo(0.5, 9);
	});
});
