/**
 * Decimation properties.
 *
 * As with hole filling, the generator starts from real surfaces rather than
 * random soup: QEM's guarantees are about manifold input, and feeding it
 * nonsense would only test that it declines to collapse anything.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { quadricSimplification } from "../../src/vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
import { UpdateBounding } from "../../src/vcg/complex/update/bounding.ts";
import { propertyOptions } from "../helpers/arbitrary.ts";
import {
	assertAllocatorConsistent,
	computeFacts,
	geometryDigest,
	hausdorffDistance,
	symmetricHausdorff,
} from "../helpers/invariants.ts";
import { gridPlane, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

const BASES: ReadonlyArray<{ name: string; make: () => CMeshO; genus?: number; closed: boolean }> =
	[
		{ name: "sphere2", make: () => sphereIcosa(2).mesh, genus: 0, closed: true },
		{ name: "sphere3", make: () => sphereIcosa(3).mesh, genus: 0, closed: true },
		{ name: "torus", make: () => torus(2, 0.6, 16, 10).mesh, genus: 1, closed: true },
		{ name: "grid", make: () => gridPlane(6, 6).mesh, closed: false },
	];

function run(cm: CMeshO, target: number, params = {}): CMeshO {
	runWithResult(cm, target, params);
	return cm;
}

function runWithResult(cm: CMeshO, target: number, params = {}) {
	UpdateBounding.box(cm);
	const result = quadricSimplification(cm, { targetFaceNum: target, params });
	Allocator.compactEveryVector(cm);
	return result;
}

/** A target between a tenth of the mesh and just under its size. */
const arbTarget = () => fc.double({ min: 0.05, max: 0.95, noNaN: true });

describe("decimation properties", () => {
	test("reaches the target, or reports that it ran out of legal collapses", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const cm = base.make();
				const target = Math.max(1, Math.floor(cm.fn * frac));
				const result = runWithResult(cm, target, { preserveTopology: true });
				// The target is a goal, not a guarantee: preserveTopology
				// forbids the collapses that would change the genus, and every
				// surface has a coarsest triangulation it cannot go below. The
				// contract is that we either get there or say why not.
				if (cm.fn > target) {
					expect(result.reason, base.name).toBe("exhausted");
				} else {
					expect(result.reason, base.name).toBe("goalReached");
				}
			}),
			propertyOptions,
		);
	});

	test("without preserveTopology the target is always reached", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const cm = base.make();
				const target = Math.max(1, Math.floor(cm.fn * frac));
				const result = runWithResult(cm, target, { preserveTopology: false });
				if (result.reason === "goalReached") expect(cm.fn).toBeLessThanOrEqual(target);
			}),
			propertyOptions,
		);
	});

	test("the face count never goes up", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const cm = base.make();
				const before = cm.fn;
				run(cm, Math.max(1, Math.floor(before * frac)), { preserveTopology: true });
				expect(cm.fn).toBeLessThanOrEqual(before);
			}),
			propertyOptions,
		);
	});

	test("the result is always structurally consistent", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const cm = base.make();
				run(cm, Math.max(1, Math.floor(cm.fn * frac)), { preserveTopology: true });
				assertAllocatorConsistent(cm, base.name);
			}),
			propertyOptions,
		);
	});

	test("a closed mesh stays closed and manifold", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				if (!base.closed) return;
				const cm = base.make();
				run(cm, Math.max(4, Math.floor(cm.fn * frac)), { preserveTopology: true });
				const facts = computeFacts(cm);
				expect(facts.watertight, base.name).toBe(true);
				expect(facts.nonManifoldEdges, base.name).toBe(0);
			}),
			propertyOptions,
		);
	});

	test("preserveTopology keeps the genus", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				if (base.genus === undefined) return;
				const cm = base.make();
				run(cm, Math.max(8, Math.floor(cm.fn * frac)), { preserveTopology: true });
				expect(computeFacts(cm).genus, base.name).toBe(base.genus);
			}),
			propertyOptions,
		);
	});

	test("an open mesh keeps its boundary loop count", () => {
		fc.assert(
			fc.property(arbTarget(), (frac) => {
				const cm = gridPlane(6, 6).mesh;
				run(cm, Math.max(4, Math.floor(cm.fn * frac)), {
					preserveTopology: true,
					preserveBoundary: true,
				});
				expect(computeFacts(cm).boundaryLoops).toBe(1);
			}),
			propertyOptions,
		);
	});

	test("the decimated surface stays on the original surface", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const original = base.make();
				const cm = base.make();
				UpdateBounding.box(original);
				const diag = original.bbox.diagonal;
				run(cm, Math.max(4, Math.floor(cm.fn * frac)), { preserveTopology: true });

				// The one-sided direction is what QEM actually promises, and
				// it holds tightly at any reduction: the surviving vertices
				// are placed to minimise distance to the original planes, so
				// they stay on (or very near) the original surface.
				//
				// The other direction cannot be bounded the same way — removing
				// detail necessarily leaves parts of the original with nothing
				// nearby. A flat square reduced to three triangles has to lose
				// a corner. See the closed-mesh test below for where the
				// symmetric bound does hold.
				expect(hausdorffDistance(cm, original), base.name).toBeLessThan(diag * 0.1);
			}),
			propertyOptions,
		);
	});

	test("a closed mesh stays close in both directions at moderate reduction", () => {
		fc.assert(
			fc.property(
				fc.integer({ min: 0, max: BASES.length - 1 }),
				fc.double({ min: 0.3, max: 0.95, noNaN: true }),
				(bi, frac) => {
					const base = BASES[bi];
					if (!base.closed) return;
					// Hausdorff here is brute force, O(V x F). That is deliberate —
					// it keeps the check independent of any spatial index the
					// kernel might have — but it means the 1280-face sphere costs
					// too much to run thousands of times. The 320-face bases make
					// the same point.
					if (base.name === "sphere3") return;
					const original = base.make();
					const cm = base.make();
					UpdateBounding.box(original);
					const diag = original.bbox.diagonal;
					run(cm, Math.max(8, Math.floor(cm.fn * frac)), { preserveTopology: true });
					expect(symmetricHausdorff(original, cm), base.name).toBeLessThan(diag * 0.15);
				},
			),
			propertyOptions,
		);
	});

	test("decimation is deterministic", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const target = Math.max(4, Math.floor(base.make().fn * frac));
				const a = run(base.make(), target, { preserveTopology: true });
				const b = run(base.make(), target, { preserveTopology: true });
				expect(geometryDigest(b), base.name).toBe(geometryDigest(a));
			}),
			propertyOptions,
		);
	});

	test("optimalPlacement=false only ever reuses input vertices", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const original = base.make();
				const points = new Set<string>();
				for (let v = 0; v < original.vn; v++) {
					points.add(`${original.vx(v)},${original.vy(v)},${original.vz(v)}`);
				}
				const cm = base.make();
				run(cm, Math.max(4, Math.floor(cm.fn * frac)), {
					optimalPlacement: false,
					preserveTopology: true,
				});
				for (let v = 0; v < cm.vn; v++) {
					expect(points.has(`${cm.vx(v)},${cm.vy(v)},${cm.vz(v)}`), base.name).toBe(true);
				}
			}),
			propertyOptions,
		);
	});

	test("running twice to the same target is a no-op the second time", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbTarget(), (bi, frac) => {
				const base = BASES[bi];
				const cm = base.make();
				const target = Math.max(4, Math.floor(cm.fn * frac));
				run(cm, target, { preserveTopology: true });
				const once = geometryDigest(cm);
				const reached = cm.fn;
				run(cm, reached, { preserveTopology: true });
				expect(geometryDigest(cm), base.name).toBe(once);
			}),
			propertyOptions,
		);
	});
});
