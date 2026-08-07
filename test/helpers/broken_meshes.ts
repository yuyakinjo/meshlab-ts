/**
 * Meshes with several defects at once, for the end-to-end pipeline tests.
 *
 * The single-defect builders in `mesh_builders.ts` are for testing one filter
 * at a time. These are what a real scan or a bad export actually looks like:
 * everything wrong simultaneously, so the pipeline has to handle the
 * interactions rather than each problem in isolation.
 */
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { sphereIcosa } from "./mesh_builders.ts";

export interface BrokenMesh {
	mesh: CMeshO;
	name: string;
	/** What is wrong with it, for failure messages. */
	defects: string[];
	/** Faces the intact surface would have, for sanity-checking the repair. */
	intactFaces: number;
}

/**
 * The full catalogue of defects on one sphere:
 *
 * - unwelded, as every STL is
 * - several holes
 * - some faces wound backwards
 * - a few tiny disconnected islands
 * - a zero-area sliver
 * - duplicated faces
 * - stray unreferenced vertices
 *
 * Deliberately built in that order so the defects overlap — the flipped faces
 * sit next to the holes, and the duplicates are of faces that also got
 * flipped.
 */
export function thoroughlyBrokenSphere(subdiv = 3): BrokenMesh {
	const base = sphereIcosa(subdiv);
	const intactFaces = base.mesh.fn;
	const src = base.mesh;

	// Punch holes, well separated so each is its own loop.
	const holeFaces = new Set<number>();
	const usedVerts = new Set<number>();
	const stride = Math.max(1, Math.floor(src.fn / 6));
	for (let i = 0; holeFaces.size < 4 && i < src.fn; i++) {
		const f = (i * stride) % src.fn;
		const vs = [src.fv(f, 0), src.fv(f, 1), src.fv(f, 2)];
		if (vs.some((v) => usedVerts.has(v))) continue;
		for (const v of vs) usedVerts.add(v);
		holeFaces.add(f);
	}

	// Rebuild as an unwelded soup, dropping the hole faces, flipping some
	// windings and duplicating a couple of triangles along the way.
	const coords: number[] = [];
	const faces: number[] = [];
	const flipped = new Set([2, 11, 40, 97, 150]);
	const duplicated = new Set([5, 40]);

	const emit = (f: number, flip: boolean) => {
		const order = flip ? [0, 2, 1] : [0, 1, 2];
		const base3 = coords.length / 3;
		for (const k of order) {
			const v = src.fv(f, k);
			coords.push(src.vx(v), src.vy(v), src.vz(v));
		}
		faces.push(base3, base3 + 1, base3 + 2);
	};

	for (let f = 0; f < src.fn; f++) {
		if (holeFaces.has(f)) continue;
		emit(f, flipped.has(f));
		if (duplicated.has(f)) emit(f, flipped.has(f));
	}

	// A zero-area sliver: three collinear points.
	const sliverBase = coords.length / 3;
	coords.push(2, 0, 0, 2.5, 0, 0, 3, 0, 0);
	faces.push(sliverBase, sliverBase + 1, sliverBase + 2);

	// Small disconnected islands, far enough away to be separate components.
	for (let i = 0; i < 3; i++) {
		const b = coords.length / 3;
		const dx = 5 + i;
		coords.push(dx, 0, 0, dx + 0.02, 0, 0, dx, 0.02, 0, dx, 0, 0.02);
		faces.push(b, b + 1, b + 2, b, b + 2, b + 3, b, b + 3, b + 1, b + 1, b + 3, b + 2);
	}

	const mesh = new CMeshO();
	Allocator.addMeshData(mesh, coords, faces);

	// Stray vertices nothing references.
	for (let i = 0; i < 7; i++) Allocator.addVertex(mesh, -10 - i, 0, 0);

	return {
		mesh,
		name: `thoroughlyBrokenSphere(${subdiv})`,
		defects: [
			"unwelded triangle soup",
			`${holeFaces.size} holes`,
			`${flipped.size} flipped faces`,
			"3 disconnected islands",
			"1 zero-area sliver",
			`${duplicated.size} duplicated faces`,
			"7 unreferenced vertices",
		],
		intactFaces,
	};
}

/**
 * A cube whose faces are all wound inward, as an inside-out export.
 *
 * Nothing else is wrong with it, which is the point: the pipeline should
 * notice the negative volume and turn it the right way round without changing
 * anything else.
 */
export function insideOutCube(size = 2): BrokenMesh {
	const coords: number[] = [];
	const faces: number[] = [];
	const h = size / 2;
	const corners = [
		[-h, -h, -h],
		[h, -h, -h],
		[h, h, -h],
		[-h, h, -h],
		[-h, -h, h],
		[h, -h, h],
		[h, h, h],
		[-h, h, h],
	];
	for (const c of corners) coords.push(c[0], c[1], c[2]);
	// Every winding reversed relative to the outward cube.
	const outward = [
		4, 5, 6, 4, 6, 7, 1, 0, 3, 1, 3, 2, 5, 1, 2, 5, 2, 6, 0, 4, 7, 0, 7, 3, 7, 6, 2, 7, 2, 3, 0, 1,
		5, 0, 5, 4,
	];
	for (let i = 0; i < outward.length; i += 3) {
		faces.push(outward[i], outward[i + 2], outward[i + 1]);
	}
	const mesh = new CMeshO();
	Allocator.addMeshData(mesh, coords, faces);
	return {
		mesh,
		name: `insideOutCube(${size})`,
		defects: ["every face wound inward"],
		intactFaces: 12,
	};
}
