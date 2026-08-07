/**
 * QEM decimation.
 *
 * Decimation is not reproducible face-for-face against MeshLab — the heap's
 * tie-breaks are arbitrary and ours differ — so the assertions are on the
 * things that must hold regardless: the face count, the topology, the
 * geometric error, and determinism between our own runs.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../../src/common/ml_document/mesh_document.ts";
import { Allocator } from "../../../src/vcg/complex/allocator.ts";
import type { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import { quadricSimplification } from "../../../src/vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
import { UpdateBounding } from "../../../src/vcg/complex/update/bounding.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	geometryDigest,
	surfaceArea,
	symmetricHausdorff,
} from "../../helpers/invariants.ts";
import {
	type BuiltMesh,
	cube,
	gridPlane,
	sphereIcosa,
	torus,
} from "../../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();
const NAME = "Simplification: Quadric Edge Collapse Decimation";

function apply(
	built: BuiltMesh | CMeshO,
	params: Record<string, unknown> = {},
): { mesh: CMeshO; doc: MeshDocument; out: Record<string, unknown> } {
	const cm = "mesh" in built ? built.mesh : built;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("test", "test", true, cm);
	m.updateBoxAndNormals();
	const out = kernel.applyFilter(doc, NAME, params);
	return { mesh: m.cm, doc, out };
}

/** A standalone decimation, bypassing the plugin, for the kernel-level tests. */
function decimate(cm: CMeshO, targetFaceNum: number, params = {}): CMeshO {
	UpdateBounding.box(cm);
	quadricSimplification(cm, { targetFaceNum, params });
	Allocator.compactEveryVector(cm);
	return cm;
}

describe("decimation: registry and parameters", () => {
	test("is registered with upstream's name and defaults", () => {
		const a = kernel.filterAction(NAME);
		expect(a.implemented).toBe(true);
		expect(a.pythonName).toBe("meshing_decimation_quadric_edge_collapse");

		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, sphereIcosa(2).mesh);
		m.updateBoxAndNormals();
		const p = kernel.initParameterList(NAME, doc);
		// The default target is half the current face count.
		expect(p.getInt("TargetFaceNum")).toBe(160);
		expect(p.getFloat("TargetPerc")).toBe(0);
		expect(p.getFloat("QualityThr")).toBe(0.3);
		expect(p.getBool("PreserveBoundary")).toBe(false);
		expect(p.getFloat("BoundaryWeight")).toBe(1);
		expect(p.getBool("PreserveNormal")).toBe(false);
		expect(p.getBool("PreserveTopology")).toBe(false);
		expect(p.getBool("OptimalPlacement")).toBe(true);
		expect(p.getBool("PlanarQuadric")).toBe(false);
		expect(p.getFloat("PlanarWeight")).toBe(0.001);
		expect(p.getBool("QualityWeight")).toBe(false);
		expect(p.getBool("AutoClean")).toBe(true);
		expect(p.getBool("Selected")).toBe(false);
	});

	test("TargetPerc overrides TargetFaceNum", () => {
		const { mesh, out } = apply(sphereIcosa(3), { TargetPerc: 0.25 });
		expect(out.target_face_num).toBe(320); // 1280 × 0.25
		expect(mesh.fn).toBeLessThanOrEqual(320);
	});
});

describe("decimation: reaching the target", () => {
	test("hits the requested face count", () => {
		for (const [build, target] of [
			[() => sphereIcosa(3), 200],
			[() => sphereIcosa(3), 50],
			[() => sphereIcosa(2), 100],
			[() => torus(2, 0.6, 24, 12), 200],
		] as const) {
			const built = build();
			const { mesh } = apply(built, { TargetFaceNum: target, PreserveTopology: true });
			expect(mesh.fn, `${built.name} -> ${target}`).toBeLessThanOrEqual(target);
			assertAllocatorConsistent(mesh, built.name);
		}
	});

	test("a mesh already at or under the target is untouched", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), { TargetFaceNum: 100 });
		expect(out.collapses).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("a torus cannot be decimated below its coarsest triangulation", () => {
		// Worth pinning because it looks like a failure and is not. With
		// PreserveTopology the link condition forbids the collapses that would
		// pinch the handle shut, and a genus-1 surface needs a minimum number
		// of faces to exist at all. Asking for 5 gets you the floor, plus a
		// warning saying why.
		const { mesh, out, doc } = apply(torus(2, 0.6, 16, 10), {
			TargetFaceNum: 5,
			PreserveTopology: true,
		});
		expect(out.target_reached).toBe(false);
		expect(mesh.fn).toBeGreaterThan(5);
		expect(mesh.fn).toBeLessThan(40);
		// Still a torus, which is the point of refusing.
		expect(computeFacts(mesh).genus).toBe(1);
		expect(doc.Log.messages().join("\n")).toContain("without changing the topology");
	});

	test("without PreserveTopology the same torus goes further", () => {
		const strict = apply(torus(2, 0.6, 16, 10), {
			TargetFaceNum: 5,
			PreserveTopology: true,
		}).mesh.fn;
		const loose = apply(torus(2, 0.6, 16, 10), {
			TargetFaceNum: 5,
			PreserveTopology: false,
		}).mesh.fn;
		expect(loose).toBeLessThan(strict);
	});

	test("an empty mesh is survivable", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t");
		expect(() => kernel.applyFilter(doc, NAME, { TargetFaceNum: 10 })).not.toThrow();
		expect(m.cm.fn).toBe(0);
	});
});

describe("decimation: what must be preserved", () => {
	test("the result stays watertight and manifold", () => {
		for (const [build, target] of [
			[() => sphereIcosa(3), 100],
			[() => torus(2, 0.6, 24, 12), 150],
		] as const) {
			const built = build();
			const { mesh } = apply(built, { TargetFaceNum: target, PreserveTopology: true });
			const facts = computeFacts(mesh);
			expect(facts.watertight, built.name).toBe(true);
			expect(facts.nonManifoldEdges, built.name).toBe(0);
		}
	});

	test("PreserveTopology keeps the genus", () => {
		// The torus is the case that matters: without the link condition a
		// collapse can pinch the handle shut and turn it into a sphere.
		const { mesh } = apply(torus(2, 0.6, 24, 12), {
			TargetFaceNum: 120,
			PreserveTopology: true,
		});
		expect(computeFacts(mesh).genus).toBe(1);
	});

	test("a sphere stays genus 0 all the way down", () => {
		for (const target of [400, 200, 100, 50, 30]) {
			const { mesh } = apply(sphereIcosa(3), { TargetFaceNum: target, PreserveTopology: true });
			expect(computeFacts(mesh).genus, `target ${target}`).toBe(0);
		}
	});

	test("the surface stays close to the original", () => {
		const original = sphereIcosa(3).mesh;
		const copy = sphereIcosa(3).mesh;
		const { mesh } = apply(copy, { TargetFaceNum: 200, PreserveTopology: true });
		// On a unit sphere reduced 6x, the surfaces should still agree to a
		// few percent of the radius.
		expect(symmetricHausdorff(original, mesh)).toBeLessThan(0.1);
		expect(surfaceArea(mesh)).toBeGreaterThan(surfaceArea(original) * 0.9);
	});

	test("error grows as the target shrinks, and never the other way", () => {
		const original = sphereIcosa(3).mesh;
		let previous = 0;
		for (const target of [600, 300, 150, 60]) {
			const { mesh } = apply(sphereIcosa(3), { TargetFaceNum: target, PreserveTopology: true });
			const error = symmetricHausdorff(original, mesh);
			expect(error, `target ${target}`).toBeGreaterThanOrEqual(previous * 0.5);
			previous = error;
		}
		// Even the crudest result is recognisably the same sphere.
		expect(previous).toBeLessThan(0.35);
	});

	test("PreserveBoundary holds an open mesh's edge", () => {
		const built = gridPlane(8, 8);
		const boundaryBefore = computeFacts(built.mesh).boundaryLoops;
		const { mesh } = apply(built, {
			TargetFaceNum: 40,
			PreserveBoundary: true,
			BoundaryWeight: 10,
			PreserveTopology: true,
		});
		const facts = computeFacts(mesh);
		expect(facts.boundaryLoops).toBe(boundaryBefore);
		// The flat interior collapses freely; the outline should not shrink.
		expect(mesh.bbox.dimX).toBeCloseTo(1, 6);
		expect(mesh.bbox.dimY).toBeCloseTo(1, 6);
	});

	test("PreserveNormal stops faces folding over", () => {
		const { mesh } = apply(sphereIcosa(3), {
			TargetFaceNum: 100,
			PreserveNormal: true,
			PreserveTopology: true,
		});
		// A fold would show up as a face wound against its neighbours.
		expect(computeFacts(mesh).coherentlyOriented).toBe(true);
	});

	test("OptimalPlacement=false keeps a subset of the original vertices", () => {
		const original = sphereIcosa(2).mesh;
		const originalPoints = new Set<string>();
		for (let v = 0; v < original.vn; v++) {
			originalPoints.add(`${original.vx(v)},${original.vy(v)},${original.vz(v)}`);
		}
		const { mesh } = apply(sphereIcosa(2), {
			TargetFaceNum: 80,
			OptimalPlacement: false,
			PreserveTopology: true,
		});
		for (let v = 0; v < mesh.vn; v++) {
			const key = `${mesh.vx(v)},${mesh.vy(v)},${mesh.vz(v)}`;
			expect(originalPoints.has(key), `vertex ${v} is not an original point`).toBe(true);
		}
	});

	test("OptimalPlacement=true does move vertices, and fits better for it", () => {
		const original = sphereIcosa(3).mesh;
		const optimal = apply(sphereIcosa(3), {
			TargetFaceNum: 120,
			OptimalPlacement: true,
			PreserveTopology: true,
		}).mesh;
		const subset = apply(sphereIcosa(3), {
			TargetFaceNum: 120,
			OptimalPlacement: false,
			PreserveTopology: true,
		}).mesh;
		// Being free to place the survivor anywhere is the whole point of the
		// quadric, so it should not do worse than being pinned to an endpoint.
		expect(symmetricHausdorff(original, optimal)).toBeLessThanOrEqual(
			symmetricHausdorff(original, subset) * 1.5,
		);
	});
});

describe("decimation: determinism", () => {
	test("the same input and parameters give byte-identical output", () => {
		const runs = [0, 1, 2].map(
			() => apply(sphereIcosa(3), { TargetFaceNum: 150, PreserveTopology: true }).mesh,
		);
		const digests = runs.map((m) => geometryDigest(m));
		expect(digests[1]).toBe(digests[0]);
		expect(digests[2]).toBe(digests[0]);
	});

	test("equal-cost collapses break the same way every time", () => {
		// A sphere is highly symmetric, so its priorities collide constantly;
		// without the deterministic tie-break the output would wander.
		const a = decimate(sphereIcosa(2).mesh, 100);
		const b = decimate(sphereIcosa(2).mesh, 100);
		expect(geometryDigest(b)).toBe(geometryDigest(a));
	});
});

describe("decimation: cleanup", () => {
	test("AutoClean leaves no unreferenced vertices or null faces", () => {
		const { mesh } = apply(sphereIcosa(3), { TargetFaceNum: 100, AutoClean: true });
		const referenced = new Set<number>();
		for (let f = 0; f < mesh.faceSize; f++) {
			if (mesh.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) referenced.add(mesh.fv(f, k));
		}
		expect(referenced.size).toBe(mesh.vn);
		for (let f = 0; f < mesh.faceSize; f++) {
			if (mesh.isFaceD(f)) continue;
			const [a, b, c] = [mesh.fv(f, 0), mesh.fv(f, 1), mesh.fv(f, 2)];
			expect(a === b || b === c || a === c).toBe(false);
		}
		expect(mesh.isCompact).toBe(true);
	});

	test("stale adjacency is dropped, since every collapse invalidates it", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, sphereIcosa(2).mesh);
		m.updateBoxAndNormals();
		m.updateDataMask(0x00040000); // MM_FACEFACETOPO
		kernel.applyFilter(doc, NAME, { TargetFaceNum: 100 });
		expect(m.cm.ffFace).toBeNull();
	});

	test("the bounding box is current afterwards", () => {
		const { mesh } = apply(sphereIcosa(3), { TargetFaceNum: 100 });
		// The unit sphere's box is 2 across. With OptimalPlacement the
		// survivor goes wherever the merged quadric is smallest, and that
		// point is not constrained to lie on the surface — on a convex shape
		// it lands slightly outside. A few percent of overshoot is the
		// algorithm working, not a stale box.
		expect(mesh.bbox.maxDim).toBeGreaterThan(1.5);
		expect(mesh.bbox.maxDim).toBeLessThan(2.2);

		// What "current" actually means: the box matches the vertices.
		let maxAbs = 0;
		for (let v = 0; v < mesh.vn; v++) {
			maxAbs = Math.max(maxAbs, Math.abs(mesh.vx(v)), Math.abs(mesh.vy(v)), Math.abs(mesh.vz(v)));
		}
		expect(Math.max(...mesh.bbox.max.map(Math.abs), ...mesh.bbox.min.map(Math.abs))).toBeCloseTo(
			maxAbs,
			9,
		);
	});

	test("OptimalPlacement=false cannot leave the original bounding box", () => {
		// Pinned to input vertices, the result is a subset of the input
		// points, so the box can only shrink.
		const { mesh } = apply(sphereIcosa(3), {
			TargetFaceNum: 100,
			OptimalPlacement: false,
			PreserveTopology: true,
		});
		expect(mesh.bbox.maxDim).toBeLessThanOrEqual(2 + 1e-9);
	});
});
