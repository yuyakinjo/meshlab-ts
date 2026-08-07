/**
 * Wavefront OBJ.
 *
 * A plain text format with one element per line and no header, which makes it
 * forgiving to read and awkward to read *correctly*. The three things that
 * catch a naive parser, all handled here:
 *
 *  - Indices are 1-based, and may be **negative**, counting back from the end
 *    of whatever has been declared so far. `-1` is the most recent vertex.
 *  - A face vertex is `v`, `v/vt`, `v//vn` or `v/vt/vn`. The doubled slash is
 *    not a typo — it is how a face says it has normals but no texture
 *    coordinates.
 *  - Faces may have any number of corners. A quad or an n-gon is fanned into
 *    triangles, because `CMeshO` holds triangles.
 *
 * Materials (`mtllib`/`usemtl`) are parsed far enough to be ignored without
 * confusing the geometry; there is no texture support to attach them to yet.
 */

import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { MLIOException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { rgba } from "../../vcg/space/color4.ts";

export interface ObjReadResult {
	/** The `MM_*` channels the file actually carried. */
	readonly mask: number;
}

/**
 * Resolves an OBJ index against the count declared so far.
 *
 * Positive indices are 1-based; negative ones count back from the end, which
 * is what lets a file be concatenated onto another without renumbering.
 */
function resolveIndex(raw: number, count: number, fileName: string, line: number): number {
	const index = raw > 0 ? raw - 1 : count + raw;
	if (!Number.isInteger(raw) || raw === 0 || index < 0 || index >= count) {
		throw new MLIOException(
			`line ${line}: index ${raw} is out of range (${count} declared)`,
			fileName,
		);
	}
	return index;
}

export function readObj(m: CMeshO, data: Uint8Array, fileName = "<memory>"): ObjReadResult {
	const text = new TextDecoder().decode(data);
	const positions: number[] = [];
	const normals: number[] = [];
	const colors: number[] = [];
	let sawColor = false;
	let sawNormal = false;
	const faces: number[][] = [];
	const faceNormals: number[][] = [];

	const lines = text.split(/\r?\n/);
	for (let i = 0; i < lines.length; i++) {
		// A trailing backslash continues onto the next line.
		let line = lines[i];
		while (line.endsWith("\\") && i + 1 < lines.length) {
			line = `${line.slice(0, -1)} ${lines[++i]}`;
		}
		const hash = line.indexOf("#");
		if (hash >= 0) line = line.slice(0, hash);
		const parts = line
			.trim()
			.split(/\s+/)
			.filter((p) => p.length > 0);
		if (parts.length === 0) continue;

		switch (parts[0]) {
			case "v": {
				if (parts.length < 4) {
					throw new MLIOException(`line ${i + 1}: a vertex needs three coordinates`, fileName);
				}
				positions.push(
					num(parts[1], fileName, i + 1),
					num(parts[2], fileName, i + 1),
					num(parts[3], fileName, i + 1),
				);
				// The extended form carries r g b after the position, in 0..1.
				if (parts.length >= 7) {
					sawColor = true;
					colors.push(
						num(parts[4], fileName, i + 1),
						num(parts[5], fileName, i + 1),
						num(parts[6], fileName, i + 1),
					);
				} else {
					colors.push(1, 1, 1);
				}
				break;
			}
			case "vn":
				if (parts.length < 4) {
					throw new MLIOException(`line ${i + 1}: a normal needs three components`, fileName);
				}
				sawNormal = true;
				normals.push(
					num(parts[1], fileName, i + 1),
					num(parts[2], fileName, i + 1),
					num(parts[3], fileName, i + 1),
				);
				break;
			case "f": {
				if (parts.length < 4) {
					throw new MLIOException(`line ${i + 1}: a face needs at least three corners`, fileName);
				}
				const corners: number[] = [];
				const cornerNormals: number[] = [];
				for (let k = 1; k < parts.length; k++) {
					// "v", "v/vt", "v//vn" or "v/vt/vn" — the empty middle field
					// of the third form is meaningful, so do not filter it out.
					const fields = parts[k].split("/");
					corners.push(
						resolveIndex(num(fields[0], fileName, i + 1), positions.length / 3, fileName, i + 1),
					);
					const vn = fields.length >= 3 && fields[2] !== "" ? fields[2] : null;
					cornerNormals.push(
						vn === null
							? -1
							: resolveIndex(num(vn, fileName, i + 1), normals.length / 3, fileName, i + 1),
					);
				}
				// Fan the polygon from its first corner. Fine for the convex
				// faces OBJ almost always holds, and the alternative needs a
				// full triangulator.
				for (let k = 1; k + 1 < corners.length; k++) {
					faces.push([corners[0], corners[k], corners[k + 1]]);
					faceNormals.push([cornerNormals[0], cornerNormals[k], cornerNormals[k + 1]]);
				}
				break;
			}
			default:
				// vt, vp, g, o, s, usemtl, mtllib and anything else: skipped.
				break;
		}
	}

	m.clear();
	const vertCount = positions.length / 3;
	if (vertCount > 0) {
		const first = Allocator.addVertices(m, vertCount);
		for (let v = 0; v < vertCount; v++) {
			m.setVert(first + v, positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
			if (sawColor) {
				m.vertColor[first + v] = rgba(
					colors[3 * v] * 255,
					colors[3 * v + 1] * 255,
					colors[3 * v + 2] * 255,
				);
			}
		}
	}
	if (faces.length > 0) {
		const first = Allocator.addFaces(m, faces.length);
		for (let f = 0; f < faces.length; f++) {
			m.setFace(first + f, faces[f][0], faces[f][1], faces[f][2]);
		}
	}

	// OBJ normals are per face-corner, not per vertex. Averaging the corners
	// that landed on each vertex is the only per-vertex reading of them, and
	// it is what every importer does.
	if (sawNormal && vertCount > 0) {
		const counts = new Int32Array(vertCount);
		for (let f = 0; f < faces.length; f++) {
			for (let k = 0; k < 3; k++) {
				const n = faceNormals[f][k];
				if (n < 0) continue;
				const v = faces[f][k];
				for (let a = 0; a < 3; a++) m.vertNormal[3 * v + a] += normals[3 * n + a];
				counts[v]++;
			}
		}
		for (let v = 0; v < vertCount; v++) {
			if (counts[v] === 0) continue;
			const len = Math.hypot(m.vertNormal[3 * v], m.vertNormal[3 * v + 1], m.vertNormal[3 * v + 2]);
			if (len > 0) for (let a = 0; a < 3; a++) m.vertNormal[3 * v + a] /= len;
		}
	}

	let mask = MeshElement.MM_VERTCOORD;
	if (faces.length > 0) mask |= MeshElement.MM_FACEVERT;
	if (sawNormal) mask |= MeshElement.MM_VERTNORMAL;
	if (sawColor) mask |= MeshElement.MM_VERTCOLOR;
	return { mask };
}

function num(token: string, fileName: string, line: number): number {
	const value = Number(token);
	if (!Number.isFinite(value)) {
		throw new MLIOException(`line ${line}: "${token}" is not a number`, fileName);
	}
	return value;
}

export interface ObjSaveOptions {
	readonly saveNormals?: boolean;
	readonly saveColors?: boolean;
}

export function writeObj(m: CMeshO, options: ObjSaveOptions = {}): Uint8Array {
	const withNormals = options.saveNormals ?? false;
	const withColors = options.saveColors ?? false;

	// Deleted slots are not written, so the file's numbering is the compacted
	// one and every face index has to go through this table.
	const remap = new Int32Array(m.vertSize).fill(-1);
	let n = 0;
	for (let v = 0; v < m.vertSize; v++) if (!m.isVertD(v)) remap[v] = n++;

	const out: string[] = ["# written by meshlab-ts"];
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const row = `v ${m.vx(v)} ${m.vy(v)} ${m.vz(v)}`;
		out.push(
			withColors
				? `${row} ${(m.vertColor[v] & 0xff) / 255} ${((m.vertColor[v] >>> 8) & 0xff) / 255} ${((m.vertColor[v] >>> 16) & 0xff) / 255}`
				: row,
		);
	}
	if (withNormals) {
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v)) continue;
			out.push(`vn ${m.vertNormal[3 * v]} ${m.vertNormal[3 * v + 1]} ${m.vertNormal[3 * v + 2]}`);
		}
	}
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		// Back to 1-based, and `v//vn` when there are normals but no texture
		// coordinates to sit between them.
		const corner = (k: number) => {
			const i = remap[m.fv(f, k)] + 1;
			return withNormals ? `${i}//${i}` : `${i}`;
		};
		out.push(`f ${corner(0)} ${corner(1)} ${corner(2)}`);
	}
	out.push("");
	return new TextEncoder().encode(out.join("\n"));
}
