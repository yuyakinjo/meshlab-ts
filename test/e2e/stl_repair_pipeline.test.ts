/**
 * The Tier 1 acceptance test: a thoroughly broken STL becomes a printable
 * solid in one script.
 *
 * Every assertion is on an invariant rather than on recorded output, so a
 * change in how the repair reaches the answer does not fail the test but a
 * change in whether it reaches it does.
 */
import { describe, expect, test } from "bun:test";
import { unlinkSync } from "node:fs";
import { FilterScript } from "../../src/common/filterscript.ts";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { readStl, writeStl } from "../../src/meshlabplugins/io_base/stl.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { insideOutCube, thoroughlyBrokenSphere } from "../helpers/broken_meshes.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	signedVolume,
	surfaceArea,
	symmetricHausdorff,
} from "../helpers/invariants.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/**
 * The repair pipeline.
 *
 * The order is not arbitrary and the comments say why, because getting it
 * wrong produces a mesh that looks repaired and is not.
 */
function repairScript(): FilterScript {
	return (
		new FilterScript()
			// STL is an unwelded soup, so nothing else can see the surface as a
			// surface until the vertices are shared. This must come first.
			.add("Remove Duplicate Vertices")
			// Degenerate and duplicate faces confuse adjacency, so clear them
			// before anything reads the topology.
			.add("Remove Zero Area Faces")
			.add("Remove Duplicate Faces")
			// Now the topology is meaningful: drop the floating specks.
			.add("Remove Isolated pieces (wrt Diameter)", { MinComponentDiag: 0.5 })
			.add("Remove Unreferenced Vertices")
			// Orientability needs edge-manifoldness, so repair that first, and
			// only then agree on a direction.
			.add("Repair non Manifold Edges", { method: "Remove Faces" })
			.add("Re-Orient all faces coherently")
			// A hole's boundary is only well defined on a manifold surface, which
			// it now is.
			.add("Close Holes", { MaxHoleSize: 100 })
			// Finally, make sure the solid faces outward rather than inward.
			.add("Invert Faces Orientation", { forceFlip: false })
			// Close Holes leaves the faces it created selected, and several
			// filters default "operate on the selection" to "true if anything is
			// selected". Leaving that behind would silently confine whatever runs
			// next to a handful of cap triangles.
			.add("Select None")
	);
}

/** Round-trips through a real binary STL, as a caller's file would. */
function throughStl(mesh: CMeshO): CMeshO {
	const back = new CMeshO();
	readStl(back, writeStl(mesh, { binary: true }));
	return back;
}

describe("STL repair pipeline", () => {
	test("a thoroughly broken sphere becomes a printable solid", () => {
		const broken = thoroughlyBrokenSphere(3);
		const input = throughStl(broken.mesh);

		// Confirm the input really is as bad as advertised, or the test would
		// pass by repairing nothing.
		const before = computeFacts(input);
		expect(before.watertight, "input should not be watertight").toBe(false);
		expect(input.vn, "input should be an unwelded soup").toBe(input.fn * 3);

		const doc = new MeshDocument();
		const m = doc.addNewMesh("broken.stl", "broken", true, input);
		m.updateBoxAndNormals();

		repairScript().run(kernel, doc);

		const after = computeFacts(m.cm);
		expect(after.watertight, "repaired mesh must be watertight").toBe(true);
		expect(after.nonManifoldEdges, "no non-manifold edges").toBe(0);
		expect(after.coherentlyOriented, "windings must agree").toBe(true);
		expect(after.components, "one solid, not several").toBe(1);
		expect(after.genus, "a sphere is genus 0").toBe(0);
		expect(after.boundaryLoops, "no holes left").toBe(0);
		expect(signedVolume(m.cm), "must enclose a positive volume").toBeGreaterThan(0);
		assertAllocatorConsistent(m.cm, "repaired");
	});

	test("the repaired sphere is still the sphere it started as", () => {
		const broken = thoroughlyBrokenSphere(3);
		const doc = new MeshDocument();
		const m = doc.addNewMesh("b", "b", true, throughStl(broken.mesh));
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);

		// Repairing must not reshape the model. Compared against the icosphere
		// it started as, not against the ideal sphere:
		// sphereIcosa(3) inscribes the unit sphere and undershoots both by
		// about half a percent, which is the mesh being a mesh rather than the
		// repair losing anything.
		const intact = sphereIcosa(3).mesh;
		expect(signedVolume(m.cm)).toBeCloseTo(signedVolume(intact), 3);
		expect(surfaceArea(m.cm)).toBeCloseTo(surfaceArea(intact), 3);
		expect(symmetricHausdorff(m.cm, sphereIcosa(3).mesh)).toBeLessThan(0.05);
		// The island specks and the sliver are gone, so the face count is back
		// near the intact surface's.
		expect(m.cm.fn).toBeGreaterThan(broken.intactFaces * 0.9);
		expect(m.cm.fn).toBeLessThan(broken.intactFaces * 1.1);
	});

	test("an inside-out cube is turned the right way round", () => {
		const broken = insideOutCube(2);
		expect(signedVolume(broken.mesh)).toBeLessThan(0);

		const doc = new MeshDocument();
		const m = doc.addNewMesh("c", "c", true, throughStl(broken.mesh));
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);

		expect(signedVolume(m.cm)).toBeCloseTo(8, 4);
		expect(computeFacts(m.cm).watertight).toBe(true);
	});

	test("running the pipeline on an already-good mesh changes nothing material", () => {
		const good = sphereIcosa(2);
		const wantVolume = signedVolume(good.mesh);
		const doc = new MeshDocument();
		const m = doc.addNewMesh("g", "g", true, good.mesh);
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);

		expect(m.cm.fn).toBe(320);
		expect(signedVolume(m.cm)).toBeCloseTo(wantVolume, 9);
		expect(computeFacts(m.cm).genus).toBe(0);
	});

	test("the pipeline is idempotent", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("b", "b", true, throughStl(thoroughlyBrokenSphere(2).mesh));
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);
		const once = { fn: m.cm.fn, vn: m.cm.vn, volume: signedVolume(m.cm) };

		repairScript().run(kernel, doc);
		expect(m.cm.fn).toBe(once.fn);
		expect(m.cm.vn).toBe(once.vn);
		expect(signedVolume(m.cm)).toBeCloseTo(once.volume, 9);
	});

	test("Close Holes leaves a selection that would confine the next filter", () => {
		// Faithful to MeshLab and a genuine hazard: NewFaceSelected defaults
		// to true, and the decimator's Selected default is "true if anything
		// is selected". Without an intervening Select None, decimation only
		// sees the cap triangles and does nothing — while reporting success.
		const doc = new MeshDocument();
		const m = doc.addNewMesh("b", "b", true, throughStl(thoroughlyBrokenSphere(3).mesh));
		m.updateBoxAndNormals();
		// Everything except the trailing Select None.
		const withoutClear = new FilterScript(repairScript().steps.slice(0, -1));
		withoutClear.run(kernel, doc);

		let selected = 0;
		for (let f = 0; f < m.cm.faceSize; f++) if (m.cm.isFaceS(f)) selected++;
		expect(selected).toBeGreaterThan(0);

		const before = m.cm.fn;
		const out = kernel.applyFilter(doc, "Simplification: Quadric Edge Collapse Decimation", {
			TargetFaceNum: 200,
			PreserveTopology: true,
		});
		expect(m.cm.fn).toBe(before); // nothing happened
		expect(out.target_reached).toBe(false);
		// And it says so, naming the selection rather than blaming topology.
		expect(doc.Log.all().some((e) => e.message.includes("Select None"))).toBe(true);
	});

	test("repair then decimate still gives a printable solid", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("b", "b", true, throughStl(thoroughlyBrokenSphere(3).mesh));
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);
		kernel.applyFilter(doc, "Simplification: Quadric Edge Collapse Decimation", {
			TargetFaceNum: 200,
			PreserveTopology: true,
		});

		const facts = computeFacts(m.cm);
		expect(m.cm.fn).toBeLessThanOrEqual(200);
		expect(facts.watertight).toBe(true);
		expect(facts.genus).toBe(0);
		expect(signedVolume(m.cm)).toBeGreaterThan(0);
	});

	test("the repaired mesh survives a round trip back to STL", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("b", "b", true, throughStl(thoroughlyBrokenSphere(2).mesh));
		m.updateBoxAndNormals();
		repairScript().run(kernel, doc);

		// Written out and read back, it is a soup again — that is what STL is.
		// What matters is that welding it recovers exactly the same solid.
		const reloaded = throughStl(m.cm);
		const doc2 = new MeshDocument();
		const m2 = doc2.addNewMesh("r", "r", true, reloaded);
		m2.updateBoxAndNormals();
		kernel.applyFilter(doc2, "Remove Duplicate Vertices");

		expect(m2.cm.fn).toBe(m.cm.fn);
		expect(computeFacts(m2.cm).watertight).toBe(true);
		expect(signedVolume(m2.cm)).toBeCloseTo(signedVolume(m.cm), 4);
	});
});

describe("FilterScript", () => {
	test("round-trips through .mlx", () => {
		const script = repairScript();
		const reparsed = FilterScript.fromMLX(script.toMLX());
		expect(reparsed.steps.map((s) => s.filterName)).toEqual(script.steps.map((s) => s.filterName));
		expect(reparsed.steps[3].params.MinComponentDiag).toBe(0.5);
		expect(reparsed.steps[7].params.MaxHoleSize).toBe(100);
		expect(reparsed.steps[8].params.forceFlip).toBe(false);
		// Regression: a self-closing <filter/> used to be read as an opening
		// tag, swallowing every filter up to the next </filter>.
		expect(reparsed.length).toBe(script.length);
	});

	test("self-closing filters do not swallow the ones after them", () => {
		const mlx =
			"<FilterScript>\n" +
			' <filter name="Select None"/>\n' +
			' <filter name="Select All"/>\n' +
			' <filter name="Close Holes">\n  <Param name="MaxHoleSize" type="RichInt" value="7"/>\n </filter>\n' +
			' <filter name="Invert Selection"/>\n' +
			"</FilterScript>\n";
		const script = FilterScript.fromMLX(mlx);
		expect(script.steps.map((s) => s.filterName)).toEqual([
			"Select None",
			"Select All",
			"Close Holes",
			"Invert Selection",
		]);
		expect(script.steps[2].params.MaxHoleSize).toBe(7);
	});

	test("round-trips through JSON", () => {
		const script = repairScript();
		const reparsed = FilterScript.fromJSON(script.toJSON());
		expect(reparsed.steps).toEqual(script.steps);
	});

	test("honours the declared parameter type rather than guessing", () => {
		// A string that looks like a number must stay a string, or a filename
		// like "123" would arrive as an integer.
		const mlx =
			"<!DOCTYPE FilterScript>\n<FilterScript>\n" +
			' <filter name="Rename Current Mesh">\n' +
			'  <Param name="newName" type="RichString" value="123"/>\n' +
			" </filter>\n</FilterScript>\n";
		const script = FilterScript.fromMLX(mlx);
		expect(script.steps[0].params.newName).toBe("123");
	});

	test("reads a position written as x/y/z attributes, as MeshLab writes them", () => {
		const mlx =
			'<FilterScript>\n <filter name="Transform: Rotate">\n' +
			'  <Param name="customAxis" type="RichDirection" x="1" y="2" z="3" value="0"/>\n' +
			" </filter>\n</FilterScript>\n";
		expect(FilterScript.fromMLX(mlx).steps[0].params.customAxis).toEqual([1, 2, 3]);
	});

	test("a self-closing filter with no parameters parses", () => {
		const mlx = '<FilterScript>\n <filter name="Select All"/>\n</FilterScript>\n';
		const script = FilterScript.fromMLX(mlx);
		expect(script.length).toBe(1);
		expect(script.steps[0].params).toEqual({});
	});

	test("a failing step names itself and stops the run", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("t", "t", true, sphereIcosa(1).mesh);
		m.updateBoxAndNormals();
		const script = new FilterScript()
			.add("Remove Duplicate Vertices")
			.add("Close Holes", { nonsenseParam: 1 })
			.add("Select All");
		try {
			script.run(kernel, doc);
			throw new Error("should have thrown");
		} catch (err) {
			expect((err as Error).message).toContain('step 2 of 3, "Close Holes"');
			expect((err as Error).message).toContain("unknown parameter");
		}
		// The third step must not have run.
		expect(doc.filterHistory.map((h) => h.filterName)).toEqual(["Remove Duplicate Vertices"]);
	});

	test("a malformed script is rejected", () => {
		expect(() => FilterScript.fromMLX("not xml at all")).toThrow(/is this an .mlx script/);
		expect(() => FilterScript.fromJSON("{]")).toThrow(/not valid JSON/);
		expect(() => FilterScript.fromJSON('{"nope": 1}')).toThrow(/expected an array/);
	});
});

describe("the CLI script command", () => {
	test("repairs a file end to end", () => {
		const stl = "/tmp/meshlab-ts-e2e-broken.stl";
		const mlx = "/tmp/meshlab-ts-e2e-repair.mlx";
		const out = "/tmp/meshlab-ts-e2e-fixed.ply";
		try {
			Bun.write(stl, writeStl(thoroughlyBrokenSphere(2).mesh, { binary: true }));
			Bun.write(mlx, repairScript().toMLX());

			const proc = Bun.spawnSync(["bun", "run", "bin/meshlab-ts", "script", mlx, stl, "-o", out], {
				cwd: new URL("../..", import.meta.url).pathname,
			});
			const stderr = new TextDecoder().decode(proc.stderr);
			expect(proc.exitCode, stderr).toBe(0);
			expect(stderr).toContain("Close Holes");

			// Reload what it wrote and check it really is a solid.
			const doc = new MeshDocument();
			const m = kernel.loadMesh(doc, out);
			const facts = computeFacts(m.cm);
			expect(facts.watertight).toBe(true);
			expect(facts.genus).toBe(0);
			expect(signedVolume(m.cm)).toBeGreaterThan(0);
		} finally {
			for (const p of [stl, mlx, out]) {
				try {
					unlinkSync(p);
				} catch {
					// already gone
				}
			}
		}
	});
});
