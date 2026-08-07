/**
 * A performance floor for decimation.
 *
 * Not a benchmark — the numbers are deliberately loose, chosen so the test
 * fails only on an algorithmic regression (an accidental O(n²), a lost lazy
 * invalidation) rather than on a slow machine. Measured at roughly 0.6 s on an
 * M-series laptop; the budget is ten times that.
 *
 * For reference, the same code takes about 16 s to reduce a 1.31 M-face sphere
 * to 10 000 faces, which is the scale the plan set as the target.
 */
import { describe, expect, test } from "bun:test";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { quadricSimplification } from "../../src/vcg/complex/local_optimization/tri_edge_collapse_quadric.ts";
import { UpdateBounding } from "../../src/vcg/complex/update/bounding.ts";
import { computeFacts } from "../helpers/invariants.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

describe("decimation performance", () => {
	test("reduces 80k faces by 20x within the budget", () => {
		const cm = sphereIcosa(6).mesh; // 81 920 faces
		expect(cm.fn).toBe(81920);
		UpdateBounding.box(cm);

		const start = performance.now();
		const result = quadricSimplification(cm, {
			targetFaceNum: 4096,
			params: { preserveTopology: true },
		});
		const elapsed = performance.now() - start;
		Allocator.compactEveryVector(cm);

		expect(result.reason).toBe("goalReached");
		expect(cm.fn).toBeLessThanOrEqual(4096);
		expect(computeFacts(cm).genus).toBe(0);
		expect(elapsed, `took ${elapsed.toFixed(0)}ms`).toBeLessThan(6000);
	});

	test("the queue discards stale entries rather than growing without bound", () => {
		// Lazy invalidation means a modification pushes new entries instead of
		// updating old ones, so discards outnumber collapses by a wide margin
		// — that is the design working. What would signal a real problem is
		// the ratio exploding, which is what this pins.
		const cm = sphereIcosa(5).mesh;
		UpdateBounding.box(cm);
		const result = quadricSimplification(cm, {
			targetFaceNum: 1024,
			params: { preserveTopology: true },
		});
		expect(result.performed).toBeGreaterThan(0);
		expect(result.discarded / result.performed).toBeLessThan(60);
	});
});
