#!/usr/bin/env bun
/**
 * Writes the fixture meshes the golden tests feed to both implementations.
 *
 * The differential tests load *the same file* into PyMeshLab and into this
 * library, so the file is the single source of truth for the input geometry —
 * whatever either loader does with it, both see the identical bytes. PLY with
 * double coordinates keeps the builders' values exact.
 *
 * Deterministic by construction (the builders use no randomness), so running
 * this again reproduces the same files. It only needs running when a new
 * fixture shape is wanted.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { writePly } from "../src/meshlabplugins/io_base/ply.ts";
import { Allocator } from "../src/vcg/complex/allocator.ts";
import type { CMeshO } from "../src/vcg/complex/cmesho.ts";
import {
	cube,
	gridPlane,
	nonManifoldEdgeFan,
	sphereIcosa,
	torus,
} from "../test/helpers/mesh_builders.ts";

const meshes = join(import.meta.dir, "..", "test", "fixtures", "meshes");

/** Appends `src` onto `dst`, offset by `d` and scaled by `s`. */
function appendAt(dst: CMeshO, src: CMeshO, d: readonly number[], s: number): void {
	const base = Allocator.addVertices(dst, src.vn);
	const remap = new Int32Array(src.vertSize).fill(-1);
	let next = base;
	for (let v = 0; v < src.vertSize; v++) {
		if (src.isVertD(v)) continue;
		remap[v] = next;
		dst.setVert(next, src.vx(v) * s + d[0], src.vy(v) * s + d[1], src.vz(v) * s + d[2]);
		next++;
	}
	for (let f = 0; f < src.faceSize; f++) {
		if (src.isFaceD(f)) continue;
		Allocator.addFace(dst, remap[src.fv(f, 0)], remap[src.fv(f, 1)], remap[src.fv(f, 2)]);
	}
}

function save(name: string, mesh: CMeshO): void {
	const path = join(meshes, name);
	writeFileSync(path, writePly(mesh, { binary: true }));
	console.log(`wrote ${name}: ${mesh.vn} vertices, ${mesh.fn} faces`);
}

// A closed genus-1 surface: the shape that catches genus/topology mistakes.
save("torus.ply", torus(2, 0.6, 24, 12).mesh);

// A finer sphere for the smoothing and decimation cases, where a tetrahedron
// would be all boundary effects and no interior.
save("sphere3.ply", sphereIcosa(3).mesh);

// A main body plus two small islands, for the component-removal cases: the
// sphere has 320 faces, each island 12, so a threshold between the two is
// unambiguous.
{
	const island = cube().mesh;
	const combined = sphereIcosa(2).mesh;
	appendAt(combined, island, [3, 0, 0], 0.3);
	appendAt(combined, island, [0, 3, 0], 0.3);
	save("islands.ply", combined);
}

// A fan of extra faces on one edge: the non-manifold repair case.
save("fan.ply", nonManifoldEdgeFan(2).mesh);

// An open sheet, for the cases where boundary handling is the behaviour under
// test — smoothing borders, boundary-preserving decimation.
save("plane.ply", gridPlane(6, 6).mesh);
