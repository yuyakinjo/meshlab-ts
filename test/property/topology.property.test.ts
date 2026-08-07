/**
 * Cross-validation of the adjacency builders against the naive edge-hash
 * implementations in `invariants.ts`, over arbitrary triangle soup.
 *
 * This is where the real confidence comes from: the kernel sorts edges and
 * threads intrusive rings, the oracle builds a `Map<string, number[]>`, and the
 * two must agree on every random input including the non-manifold, degenerate
 * and disconnected ones.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { Clean } from "../../src/vcg/complex/clean.ts";
import { UpdateFlags } from "../../src/vcg/complex/update/flag.ts";
import { UpdateNormal } from "../../src/vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { arbTriSoup, propertyOptions } from "../helpers/arbitrary.ts";
import {
	assertAllocatorConsistent,
	assertFFConsistent,
	countBoundaryEdges,
	countComponents,
	countEdges,
	countNonManifoldEdges,
	hasDegenerateFaces,
	isCoherentlyOriented,
	isOrientable,
	isWatertight,
	surfaceArea,
} from "../helpers/invariants.ts";
import { buildMesh } from "../helpers/mesh_builders.ts";

describe("topology properties", () => {
	test("FF rings are valid for any triangle soup", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				assertFFConsistent(m);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("VF chains cover every corner for any triangle soup", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.vertexFace(m);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("building adjacency repeatedly, in any order, converges", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				UpdateTopology.vertexFace(m);
				const ff = Array.from(m.ffFace as Int32Array);
				const vf = Array.from(m.vfHeadFace as Int32Array);
				UpdateTopology.faceFace(m);
				UpdateTopology.vertexFace(m);
				expect(Array.from(m.ffFace as Int32Array)).toEqual(ff);
				expect(Array.from(m.vfHeadFace as Int32Array)).toEqual(vf);
			}),
			propertyOptions,
		);
	});

	test("countEdgeNum agrees with the naive edge hash", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				const counts = Clean.countEdgeNum(m);
				expect(counts.total).toBe(countEdges(m));
				expect(counts.boundary).toBe(countBoundaryEdges(m));
				expect(counts.nonManifold).toBe(countNonManifoldEdges(m));
				expect(Clean.isWaterTight(m)).toBe(isWatertight(m));
			}),
			propertyOptions,
		);
	});

	test("connected components agree with naive union-find", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				const comps = Clean.connectedComponents(m);
				expect(comps.length).toBe(countComponents(m));
				// The reported sizes must add up to every live face.
				expect(comps.reduce((a, [n]) => a + n, 0)).toBe(m.fn);
			}),
			propertyOptions,
		);
	});

	test("FF-derived border bits agree with sort-derived ones", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				UpdateFlags.faceBorderFromFF(m);
				const fromFF: boolean[] = [];
				for (let f = 0; f < m.faceSize; f++)
					for (let e = 0; e < 3; e++) fromFF.push(m.isFaceB(f, e));
				UpdateFlags.faceBorderFromNone(m);
				const fromNone: boolean[] = [];
				for (let f = 0; f < m.faceSize; f++)
					for (let e = 0; e < 3; e++) fromNone.push(m.isFaceB(f, e));
				expect(fromNone).toEqual(fromFF);
			}),
			propertyOptions,
		);
	});

	test("coherence agrees with the naive check where it is defined", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				// The naive check inspects every edge; the kernel skips
				// non-manifold ones, so compare only when there are none. A
				// self-edge from a degenerate face has no direction at all, so
				// those meshes are out of scope for this comparison too.
				if (countNonManifoldEdges(m) > 0 || hasDegenerateFaces(m)) return;
				expect(Clean.isCoherentlyOrientedMesh(m)).toBe(isCoherentlyOriented(m));
			}),
			propertyOptions,
		);
	});

	test("orientCoherentlyMesh either succeeds fully or reports non-orientable", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				if (countNonManifoldEdges(m) > 0 || hasDegenerateFaces(m)) return;
				const wasOrientable = isOrientable(m);
				const result = Clean.orientCoherentlyMesh(m);
				expect(result.isOrientable).toBe(wasOrientable);
				if (result.isOrientable) {
					// Success must be total, not partial.
					expect(Clean.isCoherentlyOrientedMesh(m)).toBe(true);
				}
				// Reorienting never changes the surface itself.
				assertFFConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("reorienting preserves area and every topological count", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateTopology.faceFace(m);
				if (countNonManifoldEdges(m) > 0) return;
				const before = {
					area: surfaceArea(m),
					edges: countEdges(m),
					boundary: countBoundaryEdges(m),
					components: countComponents(m),
				};
				Clean.orientCoherentlyMesh(m);
				expect(surfaceArea(m)).toBeCloseTo(before.area, 9);
				expect(countEdges(m)).toBe(before.edges);
				expect(countBoundaryEdges(m)).toBe(before.boundary);
				expect(countComponents(m)).toBe(before.components);
			}),
			propertyOptions,
		);
	});

	test("flipMesh is an involution and preserves area", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				const before = Array.from(m.faceVert);
				const area = surfaceArea(m);
				Clean.flipMesh(m);
				expect(surfaceArea(m)).toBeCloseTo(area, 9);
				Clean.flipMesh(m);
				expect(Array.from(m.faceVert)).toEqual(before);
			}),
			propertyOptions,
		);
	});
});

describe("normal properties", () => {
	test("face normals are perpendicular to both edges", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateNormal.perFace(m);
				const n = new Float64Array(3);
				for (let f = 0; f < m.fn; f++) {
					UpdateNormal.faceNormalOf(m, f, n);
					const scale = Math.max(1, Math.hypot(n[0], n[1], n[2]));
					const a = m.fv(f, 0);
					for (const b of [m.fv(f, 1), m.fv(f, 2)]) {
						const ex = m.vx(b) - m.vx(a);
						const ey = m.vy(b) - m.vy(a);
						const ez = m.vz(b) - m.vz(a);
						const edgeLen = Math.max(1, Math.hypot(ex, ey, ez));
						const d = (n[0] * ex + n[1] * ey + n[2] * ez) / (scale * edgeLen);
						expect(Math.abs(d)).toBeLessThan(1e-9);
					}
				}
			}),
			propertyOptions,
		);
	});

	test("the unnormalised face normal has length twice the area", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				let total = 0;
				for (let f = 0; f < m.fn; f++) total += UpdateNormal.faceDoubleArea(m, f) / 2;
				expect(total).toBeCloseTo(surfaceArea(m), 6);
			}),
			propertyOptions,
		);
	});

	test("normalised normals are unit length, or exactly zero for a degenerate face", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateNormal.perFaceNormalized(m);
				for (let f = 0; f < m.fn; f++) {
					const len = Math.hypot(
						m.faceNormal[3 * f],
						m.faceNormal[3 * f + 1],
						m.faceNormal[3 * f + 2],
					);
					// A zero-area face keeps a zero normal rather than becoming
					// NaN, so that averages downstream stay finite.
					expect(len === 0 || Math.abs(len - 1) < 1e-9).toBe(true);
				}
			}),
			propertyOptions,
		);
	});

	test("no normal is ever NaN", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				UpdateNormal.perVertexNormalizedPerFaceNormalized(m);
				for (const arr of [m.faceNormal, m.vertNormal]) {
					for (const x of arr) expect(Number.isNaN(x)).toBe(false);
				}
				UpdateNormal.perVertexAngleWeighted(m);
				for (const x of m.vertNormal) expect(Number.isNaN(x)).toBe(false);
			}),
			propertyOptions,
		);
	});
});
