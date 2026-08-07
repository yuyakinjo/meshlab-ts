/**
 * The builders are the oracle for every other test, so they are checked first
 * and on their own terms: each one must match the facts it declares, computed
 * by the independent naive implementations in `invariants.ts`.
 */
import { describe, expect, test } from "bun:test";
import {
	assertAllocatorConsistent,
	assertFacts,
	isCoherentlyOriented,
	isOrientable,
	signedVolume,
	surfaceArea,
} from "../../helpers/invariants.ts";
import {
	ALL_BUILDERS,
	CLOSED_POLYHEDRA,
	cubeWithFlippedFaces,
	mobiusStrip,
	sphereIcosa,
	torus,
} from "../../helpers/mesh_builders.ts";

describe("mesh builders", () => {
	for (const build of ALL_BUILDERS) {
		const built = build();
		test(`${built.name} matches its declared facts`, () => {
			assertFacts(built.mesh, built.expected, built.name);
			assertAllocatorConsistent(built.mesh, built.name);
		});
	}

	test("closed polyhedra are wound outward", () => {
		for (const build of CLOSED_POLYHEDRA) {
			const { mesh, name, expected } = build();
			expect(signedVolume(mesh), `${name} should have positive signed volume`).toBeGreaterThan(0);
			expect(signedVolume(mesh)).toBeCloseTo(expected.volume as number, 9);
		}
	});

	test("the Möbius strip is not orientable", () => {
		const { mesh } = mobiusStrip();
		expect(isOrientable(mesh)).toBe(false);
	});

	test("flipping faces breaks coherence but not orientability", () => {
		const { mesh } = cubeWithFlippedFaces([0, 5, 9]);
		expect(isCoherentlyOriented(mesh)).toBe(false);
		expect(isOrientable(mesh)).toBe(true);
	});

	test("the icosphere converges to the unit sphere", () => {
		// Each subdivision should roughly quarter the area deficit.
		const deficits = [1, 2, 3].map((n) => 4 * Math.PI - surfaceArea(sphereIcosa(n).mesh));
		for (const d of deficits) expect(d).toBeGreaterThan(0);
		for (let i = 1; i < deficits.length; i++) {
			expect(deficits[i]).toBeLessThan(deficits[i - 1] / 3);
		}
		expect(surfaceArea(sphereIcosa(5).mesh)).toBeCloseTo(4 * Math.PI, 2);
	});

	test("the icosphere lies on the unit sphere", () => {
		const { mesh } = sphereIcosa(2);
		for (let v = 0; v < mesh.vn; v++) {
			expect(Math.hypot(mesh.vx(v), mesh.vy(v), mesh.vz(v))).toBeCloseTo(1, 12);
		}
	});

	test("the torus approaches its analytic area and volume", () => {
		const R = 2;
		const r = 0.6;
		const { mesh } = torus(R, r, 200, 100);
		expect(surfaceArea(mesh)).toBeCloseTo(4 * Math.PI * Math.PI * R * r, 1);
		expect(signedVolume(mesh)).toBeCloseTo(2 * Math.PI * Math.PI * R * r * r, 1);
	});
});
