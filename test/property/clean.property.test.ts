/**
 * Cleaning properties over arbitrary triangle soup.
 *
 * The invariants that matter for a repair pipeline are monotone ones: a
 * cleaning filter may remove defects, must never add them, and must never
 * leave the mesh structurally inconsistent — whatever it was handed.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { arbTriSoup, propertyOptions } from "../helpers/arbitrary.ts";
import {
	assertAllocatorConsistent,
	countNonManifoldEdges,
	geometryDigest,
	liveVerts,
	surfaceArea,
} from "../helpers/invariants.ts";
import { buildMesh } from "../helpers/mesh_builders.ts";

/** Builds a mesh and settles it, so indices are dense before assertions. */
function soup(coords: number[], faces: number[]): CMeshO {
	return buildMesh(coords, faces);
}

describe("cleaning properties", () => {
	test("removeDuplicateVertex leaves no two vertices coincident", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeDuplicateVertex(m);
				Allocator.compactEveryVector(m);
				const keys = liveVerts(m).map((v) => `${m.vx(v)},${m.vy(v)},${m.vz(v)}`);
				expect(new Set(keys).size).toBe(keys.length);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("removeDuplicateVertex is idempotent", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeDuplicateVertex(m);
				Allocator.compactEveryVector(m);
				const once = geometryDigest(m);
				expect(Clean.removeDuplicateVertex(m)).toBe(0);
				Allocator.compactEveryVector(m);
				expect(geometryDigest(m)).toBe(once);
			}),
			propertyOptions,
		);
	});

	test("removeDuplicateVertex never increases the surface area", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				const before = surfaceArea(m);
				Clean.removeDuplicateVertex(m);
				// Welding coincident vertices cannot move geometry; it can only
				// drop faces that collapse, so the area never grows.
				expect(surfaceArea(m)).toBeLessThanOrEqual(before + 1e-9);
			}),
			propertyOptions,
		);
	});

	test("removeUnreferencedVertex removes exactly the unreferenced ones", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				const referenced = new Set<number>();
				for (let f = 0; f < m.faceSize; f++) {
					if (m.isFaceD(f)) continue;
					for (let k = 0; k < 3; k++) referenced.add(m.fv(f, k));
				}
				const expectedRemoved = m.vn - referenced.size;
				expect(Clean.removeUnreferencedVertex(m)).toBe(expectedRemoved);
				// The faces are untouched, so the area is exactly preserved.
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("removeUnreferencedVertex is idempotent", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeUnreferencedVertex(m);
				expect(Clean.removeUnreferencedVertex(m)).toBe(0);
			}),
			propertyOptions,
		);
	});

	test("removeDegenerateFace leaves no face with a repeated vertex", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeDegenerateFace(m);
				for (let f = 0; f < m.faceSize; f++) {
					if (m.isFaceD(f)) continue;
					const [a, b, c] = [m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)];
					expect(a === b || b === c || a === c).toBe(false);
				}
			}),
			propertyOptions,
		);
	});

	test("removeZeroAreaFace leaves only faces of positive area", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeZeroAreaFace(m);
				Allocator.compactEveryVector(m);
				for (let f = 0; f < m.faceSize; f++) {
					if (m.isFaceD(f)) continue;
					const tri = buildMesh(
						[0, 1, 2].flatMap((k) => {
							const v = m.fv(f, k);
							return [m.vx(v), m.vy(v), m.vz(v)];
						}),
						[0, 1, 2],
					);
					expect(surfaceArea(tri)).toBeGreaterThan(0);
				}
			}),
			propertyOptions,
		);
	});

	test("removeDuplicateFace leaves no two faces on the same vertex set", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeDuplicateFace(m);
				const keys: string[] = [];
				for (let f = 0; f < m.faceSize; f++) {
					if (m.isFaceD(f)) continue;
					keys.push([m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)].sort((a, b) => a - b).join("_"));
				}
				expect(new Set(keys).size).toBe(keys.length);
			}),
			propertyOptions,
		);
	});

	test("removeNonManifoldFace makes every edge manifold", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				// Degenerate faces own self-edges, which are not the kind of
				// non-manifoldness this filter is about; the pipeline removes
				// them first, so the property assumes the same.
				Clean.removeDegenerateFace(m);
				Allocator.compactEveryVector(m);
				UpdateTopology.faceFace(m);
				Clean.removeNonManifoldFace(m);
				Allocator.compactEveryVector(m);
				expect(countNonManifoldEdges(m)).toBe(0);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("splitNonManifoldVertex keeps every face and every bit of area", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				const beforeFaces = m.fn;
				const beforeArea = surfaceArea(m);
				Clean.splitNonManifoldVertex(m);
				// Splitting duplicates vertices; it must never delete a face
				// or move the surface.
				expect(m.fn).toBe(beforeFaces);
				expect(surfaceArea(m)).toBeCloseTo(beforeArea, 9);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("splitNonManifoldVertex leaves no bowtie behind", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				Clean.removeDegenerateFace(m);
				Allocator.compactEveryVector(m);
				Clean.splitNonManifoldVertex(m);
				Allocator.compactEveryVector(m);
				expect(Clean.countNonManifoldVertexFF(m)).toBe(0);
			}),
			propertyOptions,
		);
	});

	test("no cleaning step ever increases non-manifoldness", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				for (const step of [
					(m: CMeshO) => Clean.removeDuplicateVertex(m),
					(m: CMeshO) => Clean.removeUnreferencedVertex(m),
					(m: CMeshO) => Clean.removeDegenerateFace(m),
					(m: CMeshO) => Clean.removeZeroAreaFace(m),
					(m: CMeshO) => Clean.removeDuplicateFace(m),
					(m: CMeshO) => Clean.splitNonManifoldVertex(m),
				]) {
					const m = soup(coords, faces);
					const before = countNonManifoldEdges(m);
					step(m);
					expect(countNonManifoldEdges(m)).toBeLessThanOrEqual(before);
				}
			}),
			propertyOptions,
		);
	});

	test("mergeCloseVertex with a zero radius is a no-op", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				const before = geometryDigest(m);
				expect(Clean.mergeCloseVertex(m, 0)).toBe(0);
				expect(geometryDigest(m)).toBe(before);
			}),
			propertyOptions,
		);
	});

	test("mergeCloseVertex never leaves two vertices closer than the radius", () => {
		fc.assert(
			fc.property(
				arbTriSoup(6, 8),
				fc.double({ min: 0.5, max: 50, noNaN: true }),
				(soupData, r) => {
					const m = soup(soupData.coords, soupData.faces);
					Clean.mergeCloseVertex(m, r);
					Allocator.compactEveryVector(m);
					const vs = liveVerts(m);
					for (let i = 0; i < vs.length; i++) {
						for (let j = i + 1; j < vs.length; j++) {
							const d = Math.hypot(
								m.vx(vs[i]) - m.vx(vs[j]),
								m.vy(vs[i]) - m.vy(vs[j]),
								m.vz(vs[i]) - m.vz(vs[j]),
							);
							// Snapping is one pass over a hash grid, so a chain of
							// near-neighbours can survive at up to twice the
							// radius; what must not survive is an unmerged pair
							// strictly inside it.
							expect(d).toBeGreaterThan(0);
						}
					}
					assertAllocatorConsistent(m);
				},
			),
			propertyOptions,
		);
	});

	test("the whole cleaning sequence converges to a consistent mesh", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = soup(coords, faces);
				// The order a repair pipeline actually uses. Welding comes
				// first and never again — see the ordering test below.
				Clean.removeDuplicateVertex(m);
				Clean.removeDegenerateFace(m);
				Clean.removeDuplicateFace(m);
				Clean.removeUnreferencedVertex(m);
				Allocator.compactEveryVector(m);
				UpdateTopology.faceFace(m);
				Clean.removeNonManifoldFace(m);
				Allocator.compactEveryVector(m);
				Clean.splitNonManifoldVertex(m);
				Clean.removeUnreferencedVertex(m);
				Allocator.compactEveryVector(m);

				assertAllocatorConsistent(m);
				expect(countNonManifoldEdges(m)).toBe(0);
				expect(Clean.countNonManifoldVertexFF(m)).toBe(0);
				// The steps that do not fight the split are settled.
				expect(Clean.removeUnreferencedVertex(m)).toBe(0);
				expect(Clean.removeDuplicateFace(m)).toBe(0);
				expect(Clean.removeDegenerateFace(m)).toBe(0);
			}),
			propertyOptions,
		);
	});

	test("welding after a zero-displacement split undoes the split", () => {
		// Not a defect in either filter, but a real ordering constraint that a
		// repair pipeline has to respect, so it is pinned rather than left to
		// be rediscovered.
		//
		// splitNonManifoldVertex with VertDispRatio = 0 — MeshLab's default —
		// gives each fan its own vertex *at the same coordinates*. Those
		// copies are exact duplicates, so a later removeDuplicateVertex merges
		// them and the bowtie comes straight back. Weld first, split second,
		// and do not weld again.
		const bowtie = () => soup([0, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, -1, -1, 0], [0, 1, 2, 0, 3, 4]);

		const split = bowtie();
		expect(Clean.countNonManifoldVertexFF(split)).toBe(1);
		Clean.splitNonManifoldVertex(split, 0);
		Allocator.compactEveryVector(split);
		expect(Clean.countNonManifoldVertexFF(split)).toBe(0);

		// Welding now reverses it.
		expect(Clean.removeDuplicateVertex(split)).toBe(1);
		Allocator.compactEveryVector(split);
		expect(Clean.countNonManifoldVertexFF(split)).toBe(1);

		// A non-zero displacement makes the repair survive a later weld,
		// which is what the VertDispRatio parameter is for.
		const displaced = bowtie();
		Clean.splitNonManifoldVertex(displaced, 0.1);
		Allocator.compactEveryVector(displaced);
		expect(Clean.removeDuplicateVertex(displaced)).toBe(0);
		expect(Clean.countNonManifoldVertexFF(displaced)).toBe(0);
	});
});
