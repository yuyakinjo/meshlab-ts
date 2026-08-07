/**
 * Geomview OFF.
 *
 * The simplest of the three formats here: a magic word, three counts, then the
 * vertices and then the faces. Two things still need care:
 *
 *  - The counts line may be on the same line as the magic word (`OFF 8 12 0`)
 *    or the line after it. Both are in the wild.
 *  - The edge count is written but universally wrong or zero, so it is read
 *    and discarded rather than trusted.
 *
 * The `C` prefix (`COFF`) means each vertex carries a colour after its
 * position, and `N` (`NOFF`) a normal before it. Both are handled; the binary
 * and 4-dimensional variants are not, and say so.
 */

import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { MLIOException, MLNotImplementedException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { rgba } from "../../vcg/space/color4.ts";

export interface OffReadResult {
	readonly mask: number;
}

export function readOff(m: CMeshO, data: Uint8Array, fileName = "<memory>"): OffReadResult {
	const text = new TextDecoder().decode(data);
	// Comments run to end of line; everything else is whitespace-separated.
	const tokens = text
		.split(/\r?\n/)
		.map((line) => {
			const hash = line.indexOf("#");
			return hash >= 0 ? line.slice(0, hash) : line;
		})
		.join("\n")
		.split(/\s+/)
		.filter((t) => t.length > 0);

	let at = 0;
	const magic = tokens[at++];
	if (magic === undefined) throw new MLIOException("the file is empty", fileName);
	if (!/^(ST)?(C)?(N)?(4)?(n)?OFF$/.test(magic)) {
		throw new MLIOException(`expected an OFF header, found "${magic}"`, fileName);
	}
	if (magic.includes("4") || magic.startsWith("n")) {
		throw new MLNotImplementedException(
			`"${magic}" is a higher-dimensional OFF variant, which is not supported`,
			"io_base",
		);
	}
	const hasColor = magic.includes("C");
	const hasNormal = magic.includes("N");
	const hasTexture = magic.startsWith("ST");

	const next = (what: string): number => {
		const token = tokens[at++];
		if (token === undefined) throw new MLIOException(`the file ends before its ${what}`, fileName);
		const value = Number(token);
		if (!Number.isFinite(value)) {
			throw new MLIOException(`expected a number for ${what}, found "${token}"`, fileName);
		}
		return value;
	};

	const vertCount = next("vertex count");
	const faceCount = next("face count");
	// The edge count is written by convention and wrong by convention.
	next("edge count");
	if (vertCount < 0 || faceCount < 0) {
		throw new MLIOException(
			`negative counts (${vertCount} vertices, ${faceCount} faces)`,
			fileName,
		);
	}

	m.clear();
	if (vertCount > 0) {
		const first = Allocator.addVertices(m, vertCount);
		for (let v = 0; v < vertCount; v++) {
			const x = next("a vertex");
			const y = next("a vertex");
			const z = next("a vertex");
			m.setVert(first + v, x, y, z);
			if (hasNormal) {
				m.vertNormal[3 * (first + v)] = next("a normal");
				m.vertNormal[3 * (first + v) + 1] = next("a normal");
				m.vertNormal[3 * (first + v) + 2] = next("a normal");
			}
			if (hasColor) {
				// Colours are 0..1 floats or 0..255 integers, told apart by
				// whether anything exceeds 1 — the format does not say which.
				const r = next("a colour");
				const g = next("a colour");
				const b = next("a colour");
				const a = next("a colour");
				const scale = r > 1 || g > 1 || b > 1 ? 1 : 255;
				m.vertColor[first + v] = rgba(r * scale, g * scale, b * scale, a > 1 ? a : a * 255);
			}
			if (hasTexture) {
				next("a texture coordinate");
				next("a texture coordinate");
			}
		}
	}

	const faces: number[][] = [];
	for (let f = 0; f < faceCount; f++) {
		const corners = next("a face's corner count");
		if (corners < 3) {
			throw new MLIOException(`face ${f} has only ${corners} corners`, fileName);
		}
		const indices: number[] = [];
		for (let k = 0; k < corners; k++) {
			const index = next("a face index");
			if (index < 0 || index >= vertCount) {
				throw new MLIOException(
					`face ${f} refers to vertex ${index}, which does not exist`,
					fileName,
				);
			}
			indices.push(index);
		}
		// A face may be followed by its own colour; skip whatever is left on
		// the line rather than mistaking it for the next face's corner count.
		// (Detected by peeking: a colour is 3 or 4 numbers, a corner count is
		// an integer 3 or more, so the only safe rule is to stop at the line
		// end — which the token split has already thrown away. Faces with
		// per-face colour are therefore not supported, and would be caught by
		// the index check above.)
		for (let k = 1; k + 1 < indices.length; k++) {
			faces.push([indices[0], indices[k], indices[k + 1]]);
		}
	}
	if (faces.length > 0) {
		const first = Allocator.addFaces(m, faces.length);
		for (let f = 0; f < faces.length; f++) {
			m.setFace(first + f, faces[f][0], faces[f][1], faces[f][2]);
		}
	}

	let mask = MeshElement.MM_VERTCOORD;
	if (faces.length > 0) mask |= MeshElement.MM_FACEVERT;
	if (hasNormal) mask |= MeshElement.MM_VERTNORMAL;
	if (hasColor) mask |= MeshElement.MM_VERTCOLOR;
	return { mask };
}

export interface OffSaveOptions {
	readonly saveColors?: boolean;
	readonly saveNormals?: boolean;
}

export function writeOff(m: CMeshO, options: OffSaveOptions = {}): Uint8Array {
	const withColors = options.saveColors ?? false;
	const withNormals = options.saveNormals ?? false;

	const remap = new Int32Array(m.vertSize).fill(-1);
	let n = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) remap[v] = n++;

	// The prefix has to match the per-vertex payload exactly, or a reader will
	// mis-split every line after the first.
	const magic = `${withColors ? "C" : ""}${withNormals ? "N" : ""}OFF`;
	const out: string[] = [magic, `${n} ${m.fn} 0`];

	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const fields = [m.vx(v), m.vy(v), m.vz(v)];
		if (withNormals) {
			fields.push(m.vertNormal[3 * v], m.vertNormal[3 * v + 1], m.vertNormal[3 * v + 2]);
		}
		if (withColors) {
			const c = m.vertColor[v];
			fields.push(c & 0xff, (c >>> 8) & 0xff, (c >>> 16) & 0xff, (c >>> 24) & 0xff);
		}
		out.push(fields.join(" "));
	}
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		out.push(`3 ${remap[m.fv(f, 0)]} ${remap[m.fv(f, 1)]} ${remap[m.fv(f, 2)]}`);
	}
	out.push("");
	return new TextEncoder().encode(out.join("\n"));
}
