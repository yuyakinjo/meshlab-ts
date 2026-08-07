/**
 * The allocator is the one component every other component sits on, so it gets
 * the heaviest fuzzing: arbitrary interleavings of add, delete and compact,
 * with the full structural invariant checked after every single step.
 */
import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import {
	type AllocatorOp,
	arbAllocatorOps,
	arbTriSoup,
	propertyOptions,
} from "../helpers/arbitrary.ts";
import { assertAllocatorConsistent, liveVerts } from "../helpers/invariants.ts";
import { buildMesh } from "../helpers/mesh_builders.ts";

function liveList(size: number, deleted: (i: number) => boolean): number[] {
	const out: number[] = [];
	for (let i = 0; i < size; i++) if (!deleted(i)) out.push(i);
	return out;
}

/**
 * Runs one random workload.
 *
 * Faces are always given three live vertices, and a vertex is only deleted
 * once nothing references it — otherwise the workload would be asking the
 * allocator to build a mesh that is invalid by construction, and every
 * compaction would legitimately throw.
 */
function runOps(m: CMeshO, ops: readonly AllocatorOp[], check: (m: CMeshO) => void): void {
	for (const op of ops) {
		switch (op.kind) {
			case "addVerts": {
				const first = Allocator.addVertices(m, op.n);
				for (let i = 0; i < op.n; i++) m.setVert(first + i, first + i, 0, 0);
				break;
			}
			case "addFaces": {
				const verts = liveList(m.vertSize, (v) => m.isVertD(v));
				if (verts.length < 3) break;
				for (let i = 0; i < op.n; i++) {
					const a = verts[(i * 3) % verts.length];
					const b = verts[(i * 3 + 1) % verts.length];
					const c = verts[(i * 3 + 2) % verts.length];
					if (a === b || b === c || a === c) continue;
					Allocator.addFace(m, a, b, c);
				}
				break;
			}
			case "deleteVert": {
				const referenced = new Set<number>();
				for (let f = 0; f < m.faceSize; f++) {
					if (m.isFaceD(f)) continue;
					for (let k = 0; k < 3; k++) referenced.add(m.fv(f, k));
				}
				const free = liveList(m.vertSize, (v) => m.isVertD(v)).filter((v) => !referenced.has(v));
				if (free.length === 0) break;
				Allocator.deleteVertex(m, free[Math.floor(op.pick * free.length)]);
				break;
			}
			case "deleteFace": {
				const faces = liveList(m.faceSize, (f) => m.isFaceD(f));
				if (faces.length === 0) break;
				Allocator.deleteFace(m, faces[Math.floor(op.pick * faces.length)]);
				break;
			}
			case "compactFaces":
				Allocator.compactFaceVector(m);
				break;
			case "compactVerts":
				Allocator.compactVertexVector(m);
				break;
			case "compactAll":
				Allocator.compactEveryVector(m);
				break;
		}
		check(m);
	}
}

describe("Allocator properties", () => {
	test("every structural invariant survives any add/delete/compact sequence", () => {
		fc.assert(
			fc.property(arbAllocatorOps(), (ops) => {
				const m = new CMeshO();
				runOps(m, ops, (mesh) => assertAllocatorConsistent(mesh));
			}),
			propertyOptions,
		);
	});

	test("optional channels stay the right length through the same workload", () => {
		fc.assert(
			fc.property(arbAllocatorOps(), (ops) => {
				const m = new CMeshO();
				m.enableChannels(
					MeshElement.MM_FACEQUALITY |
						MeshElement.MM_FACECOLOR |
						MeshElement.MM_VERTTEXCOORD |
						MeshElement.MM_VERTRADIUS,
				);
				runOps(m, ops, (mesh) => assertAllocatorConsistent(mesh));
			}),
			propertyOptions,
		);
	});

	test("compaction preserves the multiset of live vertex positions", () => {
		fc.assert(
			fc.property(arbAllocatorOps(), (ops) => {
				const m = new CMeshO();
				runOps(m, ops, () => {});
				const before = liveVerts(m)
					.map((v) => `${m.vx(v)},${m.vy(v)},${m.vz(v)}`)
					.sort();
				Allocator.compactEveryVector(m);
				const after = liveVerts(m)
					.map((v) => `${m.vx(v)},${m.vy(v)},${m.vz(v)}`)
					.sort();
				expect(after).toEqual(before);
			}),
			propertyOptions,
		);
	});

	test("compaction preserves the set of triangles, as positions", () => {
		fc.assert(
			fc.property(arbAllocatorOps(), (ops) => {
				const m = new CMeshO();
				runOps(m, ops, () => {});
				const tri = (mesh: CMeshO, f: number) =>
					[0, 1, 2]
						.map((k) => mesh.fv(f, k))
						.map((v) => `${mesh.vx(v)},${mesh.vy(v)},${mesh.vz(v)}`)
						.join("|");
				const before: string[] = [];
				for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) before.push(tri(m, f));
				Allocator.compactEveryVector(m);
				const after: string[] = [];
				for (let f = 0; f < m.faceSize; f++) if (!m.isFaceD(f)) after.push(tri(m, f));
				expect(after.sort()).toEqual(before.sort());
			}),
			propertyOptions,
		);
	});

	test("compaction is idempotent and leaves the mesh compact", () => {
		fc.assert(
			fc.property(arbAllocatorOps(), (ops) => {
				const m = new CMeshO();
				runOps(m, ops, () => {});
				Allocator.compactEveryVector(m);
				expect(m.isCompact).toBe(true);
				const remap = Allocator.compactEveryVector(m);
				// A second pass has nothing to do, so both remaps are identity.
				for (let i = 0; i < remap.vertRemap.length; i++) expect(remap.vertRemap[i]).toBe(i);
				for (let i = 0; i < remap.faceRemap.length; i++) expect(remap.faceRemap[i]).toBe(i);
			}),
			propertyOptions,
		);
	});

	test("arbitrary triangle soup builds into a consistent mesh", () => {
		fc.assert(
			fc.property(arbTriSoup(), ({ coords, faces }) => {
				const m = buildMesh(coords, faces);
				expect(m.vn).toBe(coords.length / 3);
				expect(m.fn).toBe(faces.length / 3);
				assertAllocatorConsistent(m);
			}),
			propertyOptions,
		);
	});

	test("growing never loses a value already written", () => {
		fc.assert(
			fc.property(fc.array(fc.integer({ min: 1, max: 20 }), { maxLength: 30 }), (batches) => {
				const m = new CMeshO();
				let written = 0;
				for (const n of batches) {
					const first = Allocator.addVertices(m, n);
					for (let i = 0; i < n; i++) {
						m.setVert(first + i, written, written * 2, written * 3);
						written++;
					}
					// Re-read everything: a botched growth would have dropped
					// the earlier batches.
					for (let v = 0; v < m.vertSize; v++) {
						expect(m.vx(v)).toBe(v);
						expect(m.vy(v)).toBe(v * 2);
					}
				}
			}),
			propertyOptions,
		);
	});
});
