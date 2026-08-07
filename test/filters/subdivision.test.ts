/**
 * Edge-split refinement and the three subdivision schemes built on it.
 *
 * The schemes differ in exactly one respect that is easy to state and easy to
 * check: midpoint moves nothing, butterfly interpolates so the original
 * vertices stay put, and Loop approximates so they move. Each has a matching
 * signature in the volume — unchanged, growing toward the sphere, shrinking
 * away from it — and all three must leave the mesh as manifold as they found it.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MLNotImplementedException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { FaceFlag } from "../../src/vcg/complex/flags.ts";
import { Rng } from "../../src/vcg/complex/point_sampling.ts";
import { Refine } from "../../src/vcg/complex/refine.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { assertAllocatorConsistent, computeFacts, signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();

const SCHEMES = [
	["Subdivision Surfaces: Midpoint", "meshing_surface_subdivision_midpoint"],
	["Subdivision Surfaces: Loop", "meshing_surface_subdivision_loop"],
	["Subdivision Surfaces: Butterfly Subdivision", "meshing_surface_subdivision_butterfly"],
] as const;

/** Runs a subdivision filter on a fresh mesh and returns the result. */
function subdivided(name: string, mesh: CMeshO, params: Record<string, unknown> = {}): CMeshO {
	const doc = new MeshDocument();
	doc.addNewMesh("", "m", true, mesh);
	kernel.applyFilter(doc, name, params);
	return doc.mm().cm;
}

/** A torus with its vertices nudged, so no two candidate diagonals tie. */
function jitteredTorus(amount: number, seed = 12345): CMeshO {
	const m = Platonic.torus(3, 1, 16, 8);
	const rng = new Rng(seed);
	for (let v = 0; v < m.vertSize; v++) {
		m.setVert(
			v,
			m.vx(v) + (rng.next() - 0.5) * amount,
			m.vy(v) + (rng.next() - 0.5) * amount,
			m.vz(v) + (rng.next() - 0.5) * amount,
		);
	}
	return m;
}

function hasDegenerateFace(m: CMeshO): boolean {
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		if (a === b || b === c || a === c) return true;
	}
	return false;
}

describe("refineE", () => {
	test("splitting every edge turns one triangle into four", () => {
		for (const start of [Platonic.tetrahedron(), Platonic.octahedron(), Platonic.icosahedron()]) {
			const before = { vn: start.vn, fn: start.fn };
			UpdateTopology.faceFace(start);
			expect(Refine.refineE(start, Refine.midPoint, Refine.everyEdge)).toBe(true);
			Allocator.compactEveryVector(start);
			expect(start.fn).toBe(before.fn * 4);
			// Euler: a closed surface gains one vertex per edge, and a closed
			// triangle mesh has 3F/2 edges.
			expect(start.vn).toBe(before.vn + (before.fn * 3) / 2);
			assertAllocatorConsistent(start);
		}
	});

	test("midpoint refinement is a pure retriangulation", () => {
		// It adds no shape: the solid it describes has exactly the volume it had.
		const m = Platonic.icosahedron();
		const before = signedVolume(m);
		UpdateTopology.faceFace(m);
		Refine.refineE(m, Refine.midPoint, Refine.everyEdge);
		Allocator.compactEveryVector(m);
		expect(signedVolume(m)).toBeCloseTo(before, 9);
	});

	test("reports false, and changes nothing, when no edge qualifies", () => {
		const m = Platonic.icosahedron();
		const before = { vn: m.vn, fn: m.fn };
		UpdateTopology.faceFace(m);
		// Every edge of this icosahedron is 2; nothing is longer than 100.
		expect(Refine.refineE(m, Refine.midPoint, Refine.longerThan(100))).toBe(false);
		expect(m.vn).toBe(before.vn);
		expect(m.fn).toBe(before.fn);
	});

	test("a partial refinement stays manifold and closed", () => {
		// The case with real teeth: faces with one or two split edges, where the
		// triangle has to be cut up without leaving a T-vertex behind.
		for (const threshold of [0.4, 0.7, 0.9, 1.1, 1.5]) {
			const m = jitteredTorus(0.4);
			UpdateTopology.faceFace(m);
			Refine.refineE(m, Refine.midPoint, Refine.longerThan(threshold));
			Allocator.compactEveryVector(m);
			const label = `threshold ${threshold}`;
			// A degenerate face here means the two-split-edge case chose its
			// diagonal by reading a vertex it had already overwritten.
			expect(hasDegenerateFace(m), label).toBe(false);
			const facts = computeFacts(m);
			expect(facts.nonManifoldEdges, label).toBe(0);
			expect(facts.watertight, label).toBe(true);
			expect(facts.genus, label).toBe(1);
			assertAllocatorConsistent(m, label);
		}
	});

	test("a lower threshold splits more", () => {
		let previous = Number.POSITIVE_INFINITY;
		for (const threshold of [2, 1.5, 1.1, 0.9, 0.4]) {
			const m = jitteredTorus(0.4);
			UpdateTopology.faceFace(m);
			Refine.refineE(m, Refine.midPoint, Refine.longerThan(threshold));
			expect(m.fn, `threshold ${threshold}`).toBeGreaterThanOrEqual(
				previous === Number.POSITIVE_INFINITY ? 0 : previous,
			);
			previous = m.fn;
		}
	});

	test("an open mesh keeps its boundary", () => {
		const m = Platonic.sphericalCap(Math.PI / 3, 2);
		UpdateTopology.faceFace(m);
		const before = computeFacts(m);
		Refine.refineE(m, Refine.midPoint, Refine.everyEdge);
		Allocator.compactEveryVector(m);
		const after = computeFacts(m);
		expect(after.watertight).toBe(false);
		expect(after.boundaryLoops).toBe(before.boundaryLoops);
		expect(after.nonManifoldEdges).toBe(0);
		// One rim edge becomes two, and no interior edge becomes a rim edge.
		expect(countBorderEdges(m)).toBe(countBorderEdges(Platonic.sphericalCap(Math.PI / 3, 2)) * 2);
	});

	test("selectedOnly leaves the rest of the mesh alone, and still matches up", () => {
		const m = Platonic.torus(3, 1, 16, 8);
		UpdateTopology.faceFace(m);
		let selected = 0;
		for (let f = 0; f < m.faceSize; f++) {
			// A contiguous band, so the selection has a real border to handle.
			if (f % 32 < 12) {
				m.faceFlags[f] |= FaceFlag.SELECTED;
				selected++;
			}
		}
		const before = m.fn;
		Refine.refineE(m, Refine.midPoint, Refine.everyEdge, { selectedOnly: true });
		Allocator.compactEveryVector(m);
		expect(m.fn).toBeGreaterThan(before);
		expect(m.fn).toBeLessThan(before * 4);
		expect(selected).toBeGreaterThan(0);
		// The whole point of the restriction: no crack where the selection ends.
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(true);
		expect(facts.nonManifoldEdges).toBe(0);
		expect(hasDegenerateFace(m)).toBe(false);
	});
});

describe("the subdivision schemes", () => {
	test("all three quadruple the face count and stay closed", () => {
		for (const [name] of SCHEMES) {
			const m = subdivided(name, Platonic.sphere(1), { Iterations: 1, Threshold: 0 });
			expect(m.vn, name).toBe(162);
			expect(m.fn, name).toBe(320);
			const facts = computeFacts(m);
			expect(facts.watertight, name).toBe(true);
			expect(facts.nonManifoldEdges, name).toBe(0);
			expect(facts.genus, name).toBe(0);
			expect(facts.coherentlyOriented, name).toBe(true);
			assertAllocatorConsistent(m, name);
		}
	});

	test("midpoint keeps the shape; the other two change it", () => {
		const base = signedVolume(Platonic.sphere(1));
		const volumeOf = (name: string) =>
			signedVolume(subdivided(name, Platonic.sphere(1), { Iterations: 3, Threshold: 0 }));

		expect(volumeOf("Subdivision Surfaces: Midpoint")).toBeCloseTo(base, 9);
		// Butterfly interpolates, so the surface bulges out toward the sphere
		// the vertices sit on — but never past it.
		const butterfly = volumeOf("Subdivision Surfaces: Butterfly Subdivision");
		expect(butterfly).toBeGreaterThan(base);
		expect(butterfly).toBeLessThan((4 / 3) * Math.PI);
		// Loop approximates, so it pulls the surface inside its own cage.
		expect(volumeOf("Subdivision Surfaces: Loop")).toBeLessThan(base);
	});

	test("butterfly leaves the original vertices exactly where they were", () => {
		// That is what "interpolating" means, and it is the property that
		// distinguishes it from Loop.
		const before = Platonic.sphere(1);
		const original = new Set<string>();
		for (let v = 0; v < before.vn; v++) {
			original.add(`${before.vx(v)},${before.vy(v)},${before.vz(v)}`);
		}
		const after = subdivided("Subdivision Surfaces: Butterfly Subdivision", Platonic.sphere(1), {
			Iterations: 2,
			Threshold: 0,
		});
		let kept = 0;
		for (let v = 0; v < after.vn; v++) {
			if (original.has(`${after.vx(v)},${after.vy(v)},${after.vz(v)}`)) kept++;
		}
		expect(kept).toBe(original.size);
	});

	test("Loop moves every original vertex", () => {
		const before = Platonic.sphere(1);
		const original = new Set<string>();
		for (let v = 0; v < before.vn; v++) {
			original.add(`${before.vx(v)},${before.vy(v)},${before.vz(v)}`);
		}
		const after = subdivided("Subdivision Surfaces: Loop", Platonic.sphere(1), {
			Iterations: 1,
			Threshold: 0,
		});
		let kept = 0;
		for (let v = 0; v < after.vn; v++) {
			if (original.has(`${after.vx(v)},${after.vy(v)},${after.vz(v)}`)) kept++;
		}
		expect(kept).toBe(0);
	});

	test("Loop converges rather than shrinking away", () => {
		// It shrinks, but toward a limit surface — the loss per step has to fall
		// off, or repeated subdivision would collapse the mesh.
		let previous = signedVolume(Platonic.sphere(1));
		const drops: number[] = [];
		for (const iterations of [1, 2, 3, 4]) {
			const v = signedVolume(
				subdivided("Subdivision Surfaces: Loop", Platonic.sphere(1), {
					Iterations: iterations,
					Threshold: 0,
				}),
			);
			drops.push(previous - v);
			previous = v;
		}
		for (let i = 1; i < drops.length; i++) expect(drops[i]).toBeLessThan(drops[i - 1]);
		expect(previous).toBeGreaterThan(0);
	});

	test("Loop's beta is the classical weight", () => {
		// 3/16 at valence 3, and the Warren/Loop formula above it. At valence 6
		// — the regular case — it is exactly 1/16.
		expect(Refine.loopBeta(3)).toBeCloseTo(3 / 16, 12);
		expect(Refine.loopBeta(6)).toBeCloseTo(1 / 16, 12);
		for (let k = 3; k <= 12; k++) {
			// The centre keeps a positive share of itself at every valence.
			expect(1 - k * Refine.loopBeta(k), `k=${k}`).toBeGreaterThan(0);
		}
	});

	test("the edge threshold refines only what is longer than it", () => {
		for (const [name] of SCHEMES) {
			// Every edge of a subdiv-1 sphere is well under 0.8.
			const untouched = subdivided(name, Platonic.sphere(1), { Iterations: 2, Threshold: 0.8 });
			expect(untouched.vn, name).toBe(42);
			expect(untouched.fn, name).toBe(80);
			const refined = subdivided(name, Platonic.sphere(1), { Iterations: 2, Threshold: 0 });
			expect(refined.fn, name).toBe(1280);
		}
	});

	test("iterating twice matches subdividing twice", () => {
		for (const [name] of SCHEMES) {
			const once = subdivided(name, Platonic.sphere(0), { Iterations: 1, Threshold: 0 });
			const twiceInOneGo = subdivided(name, Platonic.sphere(0), { Iterations: 2, Threshold: 0 });
			const twiceInTwoGoes = subdivided(name, once, { Iterations: 1, Threshold: 0 });
			expect(twiceInTwoGoes.vn, name).toBe(twiceInOneGo.vn);
			expect(twiceInTwoGoes.fn, name).toBe(twiceInOneGo.fn);
			expect(signedVolume(twiceInTwoGoes), name).toBeCloseTo(signedVolume(twiceInOneGo), 9);
		}
	});

	test("an open mesh keeps its rim under every scheme", () => {
		for (const [name] of SCHEMES) {
			const m = subdivided(name, Platonic.sphericalCap(Math.PI / 3, 2), {
				Iterations: 1,
				Threshold: 0,
			});
			const facts = computeFacts(m);
			expect(facts.boundaryLoops, name).toBe(1);
			expect(facts.components, name).toBe(1);
			expect(facts.nonManifoldEdges, name).toBe(0);
		}
	});

	test("Loop's boundary rule keeps the rim on its own curve", () => {
		// The interior rule would drag the boundary inwards; the 3/4-1/8-1/8
		// rule along the boundary curve is what stops it.
		const before = Platonic.sphericalCap(Math.PI / 3, 2);
		let beforeRim = 0;
		for (let v = 0; v < before.vn; v++)
			beforeRim = Math.max(beforeRim, Math.hypot(before.vx(v), before.vy(v)));
		const after = subdivided("Subdivision Surfaces: Loop", Platonic.sphericalCap(Math.PI / 3, 2), {
			Iterations: 2,
			Threshold: 0,
		});
		let afterRim = 0;
		for (let v = 0; v < after.vn; v++)
			afterRim = Math.max(afterRim, Math.hypot(after.vx(v), after.vy(v)));
		// It may tighten a little as the curve smooths, but not collapse.
		expect(afterRim).toBeGreaterThan(beforeRim * 0.95);
		expect(afterRim).toBeLessThanOrEqual(beforeRim + 1e-9);
	});

	test("are registered as MeshLab registers them", () => {
		for (const [name, pythonName] of SCHEMES) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(action.plugin.pluginName(), name).toBe("FilterMeshing");
		}
	});

	test("carry MeshLab's parameter defaults", () => {
		for (const [name] of SCHEMES) {
			const list = kernel.initParameterList(name);
			expect(list.getParameterByName("Iterations").defaultValue.value, name).toBe(3);
			expect(list.hasParameter("Threshold"), name).toBe(true);
			expect(list.getParameterByName("Selected").defaultValue.value, name).toBe(false);
		}
		// Only Loop offers a weighting scheme.
		expect(kernel.initParameterList("Subdivision Surfaces: Loop").hasParameter("LoopWeight")).toBe(
			true,
		);
		expect(
			kernel.initParameterList("Subdivision Surfaces: Midpoint").hasParameter("LoopWeight"),
		).toBe(false);
	});

	test("Loop refuses a weighting scheme it does not implement", () => {
		for (const weight of [1, 2]) {
			const doc = new MeshDocument();
			doc.addNewMesh("", "m", true, Platonic.sphere(1));
			expect(() =>
				kernel.applyFilter(doc, "Subdivision Surfaces: Loop", { LoopWeight: weight }),
			).toThrow(MLNotImplementedException);
		}
	});

	test("the quad schemes are still explicitly unimplemented", () => {
		// They produce quads, which this CMeshO does not carry; better to say so
		// than to hand back a triangulation nobody asked for.
		for (const name of [
			"Subdivision Surfaces: Catmull-Clark",
			"Subdivision Surfaces: Doo Sabin",
			"Subdivision Surfaces: LS3 Loop",
		]) {
			const doc = new MeshDocument();
			doc.addNewMesh("", "m", true, Platonic.sphere(1));
			expect(() => kernel.applyFilter(doc, name), name).toThrow(MLNotImplementedException);
		}
	});
});

function countBorderEdges(m: CMeshO): number {
	UpdateTopology.faceFace(m);
	let n = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) if (m.isBorderFF(f, e)) n++;
	}
	return n;
}
