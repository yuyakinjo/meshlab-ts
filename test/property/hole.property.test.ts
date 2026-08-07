/**
 * Hole filling over randomly punctured surfaces.
 *
 * Random triangle soup is the wrong input here — a hole is only well defined
 * on an edge-manifold surface. So the generator starts from a real closed
 * surface and knocks faces out of it, which is exactly the situation the
 * filter exists for and lets the properties be sharp: the result must be
 * watertight again, with the same genus it started with.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Hole } from "../../src/vcg/complex/hole.ts";
import { UpdateNormal } from "../../src/vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { propertyOptions } from "../helpers/arbitrary.ts";
import { assertAllocatorConsistent, computeFacts, surfaceArea } from "../helpers/invariants.ts";
import { cube, sphereIcosa, torus } from "../helpers/mesh_builders.ts";

/** The closed surfaces we puncture, with the genus each one should keep. */
const BASES: ReadonlyArray<{ name: string; make: () => CMeshO; genus: number }> = [
	{ name: "cube", make: () => cube(2).mesh, genus: 0 },
	{ name: "sphere1", make: () => sphereIcosa(1).mesh, genus: 0 },
	{ name: "sphere2", make: () => sphereIcosa(2).mesh, genus: 0 },
	{ name: "torus", make: () => torus(2, 0.6, 12, 8).mesh, genus: 1 },
];

/**
 * Deletes the faces at the given fractions of the way through the mesh.
 *
 * Returns null when the punctures merged into something whose boundary is no
 * longer a set of simple loops — two holes sharing a vertex is a pinch point,
 * not two holes, and filling it is a different problem.
 */
function puncture(m: CMeshO, picks: readonly number[]): CMeshO | null {
	const victims = new Set<number>();
	for (const p of picks) victims.add(Math.min(m.fn - 1, Math.floor(p * m.fn)));
	for (const f of victims) Allocator.deleteFace(m, f);
	Allocator.compactEveryVector(m);
	UpdateTopology.faceFace(m);

	const facts = computeFacts(m);
	if (facts.boundaryLoops === undefined) return null; // pinch point
	if (facts.nonManifoldEdges > 0) return null;
	return m;
}

const arbPicks = () =>
	fc.array(fc.double({ min: 0, max: 0.999, noNaN: true }), { minLength: 1, maxLength: 6 });

describe("hole filling properties", () => {
	test("a punctured closed surface becomes watertight again", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const base = BASES[bi];
				const m = puncture(base.make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);

				Hole.fillHoles(m, { maxHoleSize: 1000 });
				Allocator.compactEveryVector(m);

				const facts = computeFacts(m);
				expect(facts.watertight, base.name).toBe(true);
				expect(facts.nonManifoldEdges, base.name).toBe(0);
				assertAllocatorConsistent(m, base.name);
			}),
			propertyOptions,
		);
	});

	test("filling restores the genus the surface started with", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const base = BASES[bi];
				const m = puncture(base.make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				Hole.fillHoles(m, { maxHoleSize: 1000 });
				Allocator.compactEveryVector(m);
				// Capping a hole must not add a handle or remove one.
				expect(computeFacts(m).genus, base.name).toBe(base.genus);
			}),
			propertyOptions,
		);
	});

	test("filling reports exactly the loops it found", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				const loops = Hole.getInfo(m).length;
				const { holeCount } = Hole.fillHoles(m, { maxHoleSize: 1000 });
				expect(holeCount).toBe(loops);
			}),
			propertyOptions,
		);
	});

	test("a cap of an n-edge loop is exactly n-2 triangles", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				const expected = Hole.getInfo(m).reduce((acc, h) => acc + h.size - 2, 0);
				const { newFaces } = Hole.fillHoles(m, { maxHoleSize: 1000 });
				expect(newFaces).toBe(expected);
			}),
			propertyOptions,
		);
	});

	test("filling only adds area, never removes any", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				const before = surfaceArea(m);
				Hole.fillHoles(m, { maxHoleSize: 1000 });
				expect(surfaceArea(m)).toBeGreaterThanOrEqual(before - 1e-9);
			}),
			propertyOptions,
		);
	});

	test("filling adds no vertices — it only connects existing ones", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				const before = m.vn;
				Hole.fillHoles(m, { maxHoleSize: 1000 });
				// Ear cutting is a triangulation of the boundary, not a
				// refinement; RefineHole is the separate step that adds points.
				expect(m.vn).toBe(before);
			}),
			propertyOptions,
		);
	});

	test("filling is idempotent", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				Hole.fillHoles(m, { maxHoleSize: 1000 });
				Allocator.compactEveryVector(m);
				UpdateTopology.faceFace(m);
				const second = Hole.fillHoles(m, { maxHoleSize: 1000 });
				expect(second.holeCount).toBe(0);
				expect(second.newFaces).toBe(0);
			}),
			propertyOptions,
		);
	});

	test("maxHoleSize is respected exactly", () => {
		fc.assert(
			fc.property(
				fc.integer({ min: 0, max: BASES.length - 1 }),
				arbPicks(),
				fc.integer({ min: 0, max: 12 }),
				(bi, picks, limit) => {
					const m = puncture(BASES[bi].make(), picks);
					if (m === null) return;
					UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
					const eligible = Hole.getInfo(m).filter((h) => h.size <= limit && h.size >= 3);
					const { holeCount } = Hole.fillHoles(m, { maxHoleSize: limit });
					expect(holeCount).toBe(eligible.length);
				},
			),
			propertyOptions,
		);
	});

	test("all three ear strategies close every hole", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				for (const strategy of ["trivial", "minimumWeight", "selfIntersection"] as const) {
					const m = puncture(BASES[bi].make(), picks);
					if (m === null) return;
					UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
					Hole.fillHoles(m, { maxHoleSize: 1000, strategy });
					Allocator.compactEveryVector(m);
					expect(computeFacts(m).watertight, strategy).toBe(true);
				}
			}),
			propertyOptions,
		);
	});

	test("the filled surface stays coherently oriented", () => {
		fc.assert(
			fc.property(fc.integer({ min: 0, max: BASES.length - 1 }), arbPicks(), (bi, picks) => {
				const m = puncture(BASES[bi].make(), picks);
				if (m === null) return;
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				Hole.fillHoles(m, { maxHoleSize: 1000 });
				Allocator.compactEveryVector(m);
				UpdateTopology.faceFace(m);
				// The caps must agree with the surface they close, or the
				// result is watertight but inside out in patches.
				expect(Clean.isCoherentlyOrientedMesh(m)).toBe(true);
			}),
			propertyOptions,
		);
	});
});
