import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../../src/common/ml_document/mesh_document.ts";
import { filterClassToString } from "../../../src/common/plugins/filter_class.ts";
import { MLException } from "../../../src/common/utilities/ml_exception.ts";
import { Clean } from "../../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../../src/vcg/complex/cmesho.ts";
import { Hole } from "../../../src/vcg/complex/hole.ts";
import { UpdateTopology } from "../../../src/vcg/complex/update/topology.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	geometryDigest,
	signedVolume,
	surfaceArea,
} from "../../helpers/invariants.ts";
import {
	type BuiltMesh,
	buildMesh,
	cube,
	cubeWithFlippedFaces,
	cubeWithHoles,
	gridPlane,
	mobiusStrip,
	nonManifoldEdgeFan,
	sphereIcosa,
	sphereWithHoles,
	tetrahedron,
	torus,
} from "../../helpers/mesh_builders.ts";

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

describe("filter_meshing: registry", () => {
	test("the three filters are implemented with the right class", () => {
		for (const [name, klass] of [
			["Re-Orient all faces coherently", "Normal"],
			["Invert Faces Orientation", "Normal"],
			["Close Holes", "Remeshing"],
		] as const) {
			const a = kernel.filterAction(name);
			expect(a.implemented, name).toBe(true);
			expect(filterClassToString(a.filterClass), name).toBe(klass);
			expect(a.plugin.pluginName()).toBe("FilterMeshing");
		}
	});

	test("PyMeshLab names match upstream", () => {
		expect(kernel.filterAction("Close Holes").pythonName).toBe("meshing_close_holes");
		expect(kernel.filterAction("meshing_re_orient_faces_coherently").name).toBe(
			"Re-Orient all faces coherently",
		);
	});

	test("Close Holes parameter names and defaults match upstream", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, cube(10).mesh);
		m.updateBoxAndNormals();
		const p = kernel.initParameterList("Close Holes", doc);
		expect(p.getInt("MaxHoleSize")).toBe(30);
		expect(p.getBool("Selected")).toBe(false);
		expect(p.getBool("NewFaceSelected")).toBe(true);
		expect(p.getBool("SelfIntersection")).toBe(true);
		expect(p.getBool("RefineHole")).toBe(false);
		expect(p.getAbsPerc("RefineHoleEdgeLen")).toBeCloseTo(Math.sqrt(300) * 0.03, 9);

		const inv = kernel.initParameterList("Invert Faces Orientation", doc);
		expect(inv.getBool("forceFlip")).toBe(true);
		expect(inv.getBool("onlySelected")).toBe(false);
	});
});

describe("Re-Orient all faces coherently", () => {
	const NAME = "Re-Orient all faces coherently";

	test("repairs a mesh with some faces flipped", () => {
		const built = cubeWithFlippedFaces([0, 5, 9]);
		const wantArea = surfaceArea(built.mesh);
		const { mesh, out } = apply(built, NAME);
		expect(out.isOrientable).toBe(true);
		expect(out.isOriented).toBe(false); // it was not, coming in
		UpdateTopology.faceFace(mesh);
		expect(Clean.isCoherentlyOrientedMesh(mesh)).toBe(true);
		// Reorienting rewinds faces; it must not move the surface.
		expect(surfaceArea(mesh)).toBeCloseTo(wantArea, 9);
		expect(computeFacts(mesh).watertight).toBe(true);
		assertAllocatorConsistent(mesh);
	});

	test("reports an already-coherent mesh and changes nothing", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), NAME);
		expect(out.isOriented).toBe(true);
		expect(out.isOrientable).toBe(true);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("reports the Möbius strip as not orientable", () => {
		const { out } = apply(mobiusStrip(), NAME);
		expect(out.isOrientable).toBe(false);
	});

	test("refuses a non-manifold mesh instead of guessing", () => {
		expect(() => apply(nonManifoldEdgeFan(1), NAME)).toThrow(MLException);
		try {
			apply(nonManifoldEdgeFan(1), NAME);
		} catch (err) {
			expect((err as Error).message).toContain("Repair non Manifold Edges");
		}
	});
});

describe("Invert Faces Orientation", () => {
	const NAME = "Invert Faces Orientation";

	test("forceFlip reverses the winding and the sign of the volume", () => {
		const want = signedVolume(cube().mesh);
		const { mesh, out } = apply(cube(), NAME, { forceFlip: true });
		expect(out.flipped).toBe(true);
		expect(signedVolume(mesh)).toBeCloseTo(-want, 9);
	});

	test("flipping twice is the identity", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh } = apply(cube(), NAME, { forceFlip: true });
		const { mesh: back } = apply(mesh, NAME, { forceFlip: true });
		expect(geometryDigest(back)).toBe(before);
	});

	test("guess mode leaves an outward mesh alone", () => {
		const want = signedVolume(cube().mesh);
		const { mesh, out } = apply(cube(), NAME, { forceFlip: false });
		expect(out.flipped).toBe(false);
		expect(signedVolume(mesh)).toBeCloseTo(want, 9);
	});

	test("guess mode turns an inside-out mesh the right way round", () => {
		const inverted = cube().mesh;
		Clean.flipMesh(inverted);
		expect(signedVolume(inverted)).toBeLessThan(0);
		const { mesh, out } = apply(inverted, NAME, { forceFlip: false });
		expect(out.flipped).toBe(true);
		expect(signedVolume(mesh)).toBeGreaterThan(0);
	});

	test("guess mode declines on an open surface, where inside is undefined", () => {
		const { out } = apply(gridPlane(), NAME, { forceFlip: false });
		expect(out.flipped).toBe(false);
	});
});

describe("Close Holes", () => {
	const NAME = "Close Holes";

	test("a punctured cube becomes watertight and genus 0", () => {
		for (const k of [1, 2]) {
			const built = cubeWithHoles(k);
			expect(computeFacts(built.mesh).boundaryLoops).toBe(k);
			const { mesh, out } = apply(built, NAME);
			expect(out.closed_holes, `k=${k}`).toBe(k);
			expect(out.new_faces, `k=${k}`).toBe(k);
			const facts = computeFacts(mesh);
			expect(facts.watertight, `k=${k}`).toBe(true);
			expect(facts.chi, `k=${k}`).toBe(2);
			expect(facts.genus, `k=${k}`).toBe(0);
			expect(facts.coherentlyOriented, `k=${k}`).toBe(true);
			assertAllocatorConsistent(mesh);
		}
	});

	test("a punctured sphere closes all five holes", () => {
		const built = sphereWithHoles(5);
		const { mesh, out } = apply(built, NAME);
		expect(out.closed_holes).toBe(5);
		const facts = computeFacts(mesh);
		expect(facts.watertight).toBe(true);
		expect(facts.genus).toBe(0);
		expect(facts.nonManifoldEdges).toBe(0);
	});

	test("filling keeps the winding coherent with the surrounding surface", () => {
		const { mesh } = apply(sphereWithHoles(5), NAME);
		UpdateTopology.faceFace(mesh);
		expect(Clean.isCoherentlyOrientedMesh(mesh)).toBe(true);
		// A coherently wound closed surface encloses a positive volume when it
		// was outward-facing to begin with.
		expect(signedVolume(mesh)).toBeGreaterThan(0);
	});

	test("a large boundary is capped by a fan of triangles", () => {
		// A 3x2 grid has a 10-edge boundary, which needs 8 triangles.
		const built = gridPlane(3, 2);
		const { mesh, out } = apply(built, NAME, { MaxHoleSize: 100 });
		expect(out.closed_holes).toBe(1);
		expect(out.new_faces).toBe(8);
		expect(computeFacts(mesh).watertight).toBe(true);
	});

	test("MaxHoleSize skips the holes that are too big", () => {
		const built = gridPlane(3, 2); // a 10-edge boundary
		const { mesh, out } = apply(built, NAME, { MaxHoleSize: 4 });
		expect(out.closed_holes).toBe(0);
		expect(out.new_faces).toBe(0);
		expect(computeFacts(mesh).watertight).toBe(false);
	});

	test("a closed mesh has nothing to do", () => {
		const before = geometryDigest(cube().mesh);
		const { mesh, out } = apply(cube(), NAME);
		expect(out.closed_holes).toBe(0);
		expect(geometryDigest(mesh)).toBe(before);
	});

	test("is idempotent", () => {
		const { mesh } = apply(sphereWithHoles(5), NAME);
		const once = geometryDigest(mesh);
		const { mesh: twice, out } = apply(mesh, NAME);
		expect(out.closed_holes).toBe(0);
		expect(geometryDigest(twice)).toBe(once);
	});

	test("NewFaceSelected leaves exactly the new faces selected", () => {
		const built = cubeWithHoles(2);
		const originalFaces = built.mesh.fn;
		const { mesh } = apply(built, NAME, { NewFaceSelected: true });
		let selected = 0;
		for (let f = 0; f < mesh.faceSize; f++) if (!mesh.isFaceD(f) && mesh.isFaceS(f)) selected++;
		expect(selected).toBe(mesh.fn - originalFaces);
		expect(selected).toBe(2);
	});

	test("NewFaceSelected=false leaves the selection alone", () => {
		const { mesh } = apply(cubeWithHoles(2), NAME, { NewFaceSelected: false });
		let selected = 0;
		for (let f = 0; f < mesh.faceSize; f++) if (!mesh.isFaceD(f) && mesh.isFaceS(f)) selected++;
		expect(selected).toBe(0);
	});

	test("both ear strategies produce a watertight result", () => {
		for (const selfIntersection of [true, false]) {
			const { mesh } = apply(sphereWithHoles(5), NAME, { SelfIntersection: selfIntersection });
			expect(computeFacts(mesh).watertight, `SelfIntersection=${selfIntersection}`).toBe(true);
		}
	});

	test("refuses a non-manifold mesh instead of producing nonsense", () => {
		expect(() => apply(nonManifoldEdgeFan(1), NAME)).toThrow(MLException);
	});

	test("survives a mesh with no faces at all", () => {
		const { mesh, out } = apply(buildMesh([0, 0, 0], []), NAME);
		expect(out.closed_holes).toBe(0);
		assertAllocatorConsistent(mesh);
	});
});

describe("Hole.getInfo", () => {
	test("finds the loops and their sizes", () => {
		for (const [built, want] of [
			[cubeWithHoles(1), [3]],
			[cubeWithHoles(2), [3, 3]],
			[gridPlane(3, 2), [10]],
			[cube(), []],
			[torus(2, 0.6, 8, 6), []],
		] as const) {
			UpdateTopology.faceFace(built.mesh);
			const sizes = Hole.getInfo(built.mesh)
				.map((h) => h.size)
				.sort((a, b) => a - b);
			expect(sizes, built.name).toEqual([...want].sort((a, b) => a - b));
		}
	});

	test("each loop's vertices are distinct and consecutive", () => {
		const built = gridPlane(3, 2);
		UpdateTopology.faceFace(built.mesh);
		const [loop] = Hole.getInfo(built.mesh);
		expect(new Set(loop.vertices).size).toBe(loop.vertices.length);
		// Consecutive boundary vertices must actually be joined by an edge.
		const m = built.mesh;
		for (let i = 0; i < loop.vertices.length; i++) {
			const a = loop.vertices[i];
			const b = loop.vertices[(i + 1) % loop.vertices.length];
			let found = false;
			for (let f = 0; f < m.faceSize && !found; f++) {
				if (m.isFaceD(f)) continue;
				for (let k = 0; k < 3; k++) {
					const p = m.fv(f, k);
					const q = m.fv(f, (k + 1) % 3);
					if ((p === a && q === b) || (p === b && q === a)) found = true;
				}
			}
			expect(found, `${a}->${b} should be an edge`).toBe(true);
		}
	});

	test("selected-only skips loops with no selected face", () => {
		const built = cubeWithHoles(2);
		UpdateTopology.faceFace(built.mesh);
		expect(Hole.getInfo(built.mesh, true)).toHaveLength(0);
		built.mesh.faceFlags[0] |= 0x20; // SELECTED
		expect(Hole.getInfo(built.mesh, true).length).toBeGreaterThan(0);
	});
});

describe("orientation and hole closing together", () => {
	test("a flipped, punctured sphere comes out solid", () => {
		// The order a repair pipeline uses: orient, then close.
		const built = sphereWithHoles(5);
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, built.mesh);
		m.updateBoxAndNormals();
		// Scramble the winding of a few faces first.
		for (const f of [0, 7, 40, 100]) {
			if (f >= m.cm.fn) continue;
			const [a, b, c] = [m.cm.fv(f, 0), m.cm.fv(f, 1), m.cm.fv(f, 2)];
			m.cm.setFace(f, a, c, b);
		}

		kernel.applyFilter(doc, "Re-Orient all faces coherently");
		kernel.applyFilter(doc, "Close Holes");

		const facts = computeFacts(m.cm);
		expect(facts.watertight).toBe(true);
		expect(facts.coherentlyOriented).toBe(true);
		expect(facts.genus).toBe(0);
		expect(Math.abs(signedVolume(m.cm))).toBeGreaterThan(0);
		assertAllocatorConsistent(m.cm);
	});

	test("closing preserves the area of the original surface", () => {
		const built = sphereWithHoles(5);
		const before = surfaceArea(built.mesh);
		const { mesh } = apply(built, "Close Holes");
		// The caps add a little area, and nothing is taken away.
		expect(surfaceArea(mesh)).toBeGreaterThan(before);
		expect(surfaceArea(mesh)).toBeLessThan(before * 1.2);
	});

	test("the whole thing works on a closed solid too", () => {
		for (const build of [() => cube(), tetrahedron, () => sphereIcosa(2)]) {
			const built = build();
			const want = signedVolume(built.mesh);
			const doc = new MeshDocument();
			const m = doc.addNewMesh("t", "t", true, built.mesh);
			m.updateBoxAndNormals();
			kernel.applyFilter(doc, "Re-Orient all faces coherently");
			kernel.applyFilter(doc, "Close Holes");
			expect(signedVolume(m.cm), built.name).toBeCloseTo(want, 9);
		}
	});
});
