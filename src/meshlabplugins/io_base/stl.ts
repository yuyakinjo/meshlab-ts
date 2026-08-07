/**
 * STL, binary and ASCII.
 *
 * STL stores an unwelded triangle soup: every triangle carries its own three
 * vertex positions and nothing is shared. Reading one therefore produces
 * `3 × triangles` vertices, and turning it into a usable mesh means running
 * `Remove Duplicate Vertices` afterwards. That is not a shortcoming of this
 * reader — it is what the format is, and it is why `cubeSoup()` exists as a
 * test fixture.
 */

import { MLIOException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";

const BINARY_HEADER_BYTES = 80;
const BINARY_TRIANGLE_BYTES = 50;

/**
 * Decides whether `data` is binary STL.
 *
 * The leading "solid" token is not reliable — plenty of binary writers put a
 * product name there that happens to start with those letters. The size check
 * is: a binary file is exactly 84 + 50n bytes for the n it declares, and that
 * is what actually distinguishes the two.
 */
export function isBinaryStl(data: Uint8Array): boolean {
	if (data.length < BINARY_HEADER_BYTES + 4) return false;
	const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
	const declared = view.getUint32(BINARY_HEADER_BYTES, true);
	const expected = BINARY_HEADER_BYTES + 4 + declared * BINARY_TRIANGLE_BYTES;
	if (data.length === expected) return true;
	// Not a clean size match: fall back to the token, since a truncated or
	// padded binary file is still more likely binary than ASCII.
	const head = new TextDecoder().decode(data.subarray(0, 5)).toLowerCase();
	return head !== "solid";
}

export function readStl(mesh: CMeshO, data: Uint8Array, fileName = ""): void {
	if (isBinaryStl(data)) readBinaryStl(mesh, data, fileName);
	else readAsciiStl(mesh, data, fileName);
}

function readBinaryStl(mesh: CMeshO, data: Uint8Array, fileName: string): void {
	const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
	const count = view.getUint32(BINARY_HEADER_BYTES, true);
	const needed = BINARY_HEADER_BYTES + 4 + count * BINARY_TRIANGLE_BYTES;
	if (data.length < needed) {
		throw new MLIOException(
			`binary STL declares ${count} triangles, which needs ${needed} bytes, but the file is ${data.length}`,
			fileName,
		);
	}

	const firstVert = Allocator.addVertices(mesh, count * 3);
	const firstFace = Allocator.addFaces(mesh, count);
	let off = BINARY_HEADER_BYTES + 4;
	for (let t = 0; t < count; t++) {
		// Bytes 0..11 are the stored facet normal, which we ignore: it is
		// frequently wrong or zero, and recomputing from the winding is both
		// cheap and trustworthy.
		off += 12;
		for (let k = 0; k < 3; k++) {
			const v = firstVert + t * 3 + k;
			mesh.setVert(
				v,
				view.getFloat32(off, true),
				view.getFloat32(off + 4, true),
				view.getFloat32(off + 8, true),
			);
			off += 12;
		}
		mesh.setFace(firstFace + t, firstVert + t * 3, firstVert + t * 3 + 1, firstVert + t * 3 + 2);
		off += 2; // attribute byte count
	}
}

function readAsciiStl(mesh: CMeshO, data: Uint8Array, fileName: string): void {
	const text = new TextDecoder().decode(data);
	const coords: number[] = [];
	const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
	let hit = re.exec(text);
	while (hit !== null) {
		for (let k = 1; k <= 3; k++) {
			const x = Number.parseFloat(hit[k]);
			if (Number.isNaN(x)) {
				throw new MLIOException(`ASCII STL has a non-numeric coordinate "${hit[k]}"`, fileName);
			}
			coords.push(x);
		}
		hit = re.exec(text);
	}
	if (coords.length % 9 !== 0) {
		throw new MLIOException(
			`ASCII STL has ${coords.length / 3} vertices, which is not a whole number of triangles`,
			fileName,
		);
	}
	const faces = Array.from({ length: coords.length / 3 }, (_, i) => i);
	Allocator.addMeshData(mesh, coords, faces);
}

export interface StlSaveOptions {
	readonly binary?: boolean;
	/** Text placed in the 80-byte binary header. Truncated to fit. */
	readonly header?: string;
	readonly solidName?: string;
}

export function writeStl(mesh: CMeshO, options: StlSaveOptions = {}): Uint8Array {
	return options.binary === false ? writeAsciiStl(mesh, options) : writeBinaryStl(mesh, options);
}

function writeBinaryStl(mesh: CMeshO, options: StlSaveOptions): Uint8Array {
	const out = new Uint8Array(BINARY_HEADER_BYTES + 4 + mesh.fn * BINARY_TRIANGLE_BYTES);
	const view = new DataView(out.buffer);

	const header = options.header ?? "Binary STL written by meshlab-ts";
	out.set(new TextEncoder().encode(header).subarray(0, BINARY_HEADER_BYTES), 0);
	view.setUint32(BINARY_HEADER_BYTES, mesh.fn, true);

	const normal = new Float64Array(3);
	let off = BINARY_HEADER_BYTES + 4;
	for (let f = 0; f < mesh.faceSize; f++) {
		if (mesh.isFaceD(f)) continue;
		UpdateNormal.faceNormalOf(mesh, f, normal);
		const len = Math.hypot(normal[0], normal[1], normal[2]);
		for (let i = 0; i < 3; i++) {
			view.setFloat32(off + i * 4, len === 0 ? 0 : normal[i] / len, true);
		}
		off += 12;
		for (let k = 0; k < 3; k++) {
			const v = mesh.fv(f, k);
			view.setFloat32(off, mesh.vx(v), true);
			view.setFloat32(off + 4, mesh.vy(v), true);
			view.setFloat32(off + 8, mesh.vz(v), true);
			off += 12;
		}
		view.setUint16(off, 0, true);
		off += 2;
	}
	return out;
}

function writeAsciiStl(mesh: CMeshO, options: StlSaveOptions): Uint8Array {
	const name = options.solidName ?? "meshlab-ts";
	const parts: string[] = [`solid ${name}\n`];
	const normal = new Float64Array(3);
	const num = (x: number) => (Object.is(x, -0) ? "0" : String(x));

	for (let f = 0; f < mesh.faceSize; f++) {
		if (mesh.isFaceD(f)) continue;
		UpdateNormal.faceNormalOf(mesh, f, normal);
		const len = Math.hypot(normal[0], normal[1], normal[2]);
		const n = len === 0 ? [0, 0, 0] : [normal[0] / len, normal[1] / len, normal[2] / len];
		parts.push(`  facet normal ${n.map(num).join(" ")}\n    outer loop\n`);
		for (let k = 0; k < 3; k++) {
			const v = mesh.fv(f, k);
			parts.push(`      vertex ${num(mesh.vx(v))} ${num(mesh.vy(v))} ${num(mesh.vz(v))}\n`);
		}
		parts.push("    endloop\n  endfacet\n");
	}
	parts.push(`endsolid ${name}\n`);
	return new TextEncoder().encode(parts.join(""));
}
