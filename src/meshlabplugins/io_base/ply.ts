/**
 * PLY, in all three encodings: ASCII, binary little-endian and binary
 * big-endian.
 *
 * PLY is self-describing — the header lists each element's properties in
 * order, and a reader must honour that order and those types rather than
 * assume a layout. Properties we do not model are skipped by width, which is
 * what lets a file with extra per-vertex fields still load.
 */
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { MLIOException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import type { CMeshO } from "../../vcg/complex/cmesho.ts";

export type PlyFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

export type PlyScalar =
	| "char"
	| "uchar"
	| "short"
	| "ushort"
	| "int"
	| "uint"
	| "float"
	| "double"
	| "int8"
	| "uint8"
	| "int16"
	| "uint16"
	| "int32"
	| "uint32"
	| "float32"
	| "float64";

const SCALAR_BYTES: Readonly<Record<PlyScalar, number>> = {
	char: 1,
	uchar: 1,
	int8: 1,
	uint8: 1,
	short: 2,
	ushort: 2,
	int16: 2,
	uint16: 2,
	int: 4,
	uint: 4,
	int32: 4,
	uint32: 4,
	float: 4,
	float32: 4,
	double: 8,
	float64: 8,
};

export interface PlyProperty {
	readonly name: string;
	readonly type: PlyScalar;
	/** Present on list properties, e.g. `property list uchar int vertex_indices`. */
	readonly countType?: PlyScalar;
}

export interface PlyElement {
	readonly name: string;
	readonly count: number;
	readonly properties: PlyProperty[];
}

export interface PlyHeader {
	readonly format: PlyFormat;
	readonly version: string;
	readonly elements: PlyElement[];
	/** Byte offset where the data begins. */
	readonly dataStart: number;
	readonly comments: string[];
}

function scalarBytes(t: string, fileName: string): number {
	const n = SCALAR_BYTES[t as PlyScalar];
	if (n === undefined) throw new MLIOException(`unknown PLY scalar type "${t}"`, fileName);
	return n;
}

export function parsePlyHeader(data: Uint8Array, fileName = ""): PlyHeader {
	// The header is ASCII, so decode a bounded prefix rather than the whole
	// file — a 200 MB binary PLY should not be turned into a string.
	const probe = new TextDecoder("utf-8", { fatal: false }).decode(
		data.subarray(0, Math.min(data.length, 1 << 16)),
	);
	const endToken = /end_header\r?\n/.exec(probe);
	if (endToken === null) {
		throw new MLIOException("no end_header found in the first 64 KiB", fileName);
	}
	const headerText = probe.slice(0, endToken.index);
	// Byte offset, not character offset: the header is ASCII so they coincide,
	// but computing it from the encoded length keeps that assumption explicit.
	const dataStart = new TextEncoder().encode(
		probe.slice(0, endToken.index + endToken[0].length),
	).length;

	const lines = headerText
		.split(/\r?\n/)
		.map((l) => l.trim())
		.filter((l) => l !== "");
	if (lines[0] !== "ply") throw new MLIOException("the file does not start with 'ply'", fileName);

	let format: PlyFormat | null = null;
	let version = "1.0";
	const elements: PlyElement[] = [];
	const comments: string[] = [];

	for (const line of lines.slice(1)) {
		const parts = line.split(/\s+/);
		switch (parts[0]) {
			case "format": {
				const f = parts[1];
				if (f !== "ascii" && f !== "binary_little_endian" && f !== "binary_big_endian") {
					throw new MLIOException(`unknown PLY format "${f}"`, fileName);
				}
				format = f;
				version = parts[2] ?? "1.0";
				break;
			}
			case "comment":
			case "obj_info":
				comments.push(parts.slice(1).join(" "));
				break;
			case "element":
				elements.push({ name: parts[1], count: Number.parseInt(parts[2], 10), properties: [] });
				break;
			case "property": {
				const el = elements[elements.length - 1];
				if (el === undefined) {
					throw new MLIOException("a property appears before any element", fileName);
				}
				if (parts[1] === "list") {
					scalarBytes(parts[2], fileName);
					scalarBytes(parts[3], fileName);
					el.properties.push({
						name: parts[4],
						type: parts[3] as PlyScalar,
						countType: parts[2] as PlyScalar,
					});
				} else {
					scalarBytes(parts[1], fileName);
					el.properties.push({ name: parts[2], type: parts[1] as PlyScalar });
				}
				break;
			}
			default:
				// Unknown header directives are ignored, as PLY readers must be.
				break;
		}
	}
	if (format === null) throw new MLIOException("the PLY header has no format line", fileName);
	return { format, version, elements, dataStart, comments };
}

/** Sequential reader over the data section, ASCII or binary. */
class PlyDataReader {
	private readonly view: DataView;
	private readonly little: boolean;
	private readonly ascii: string[] | null;
	private offset: number;
	private token = 0;

	constructor(
		private readonly data: Uint8Array,
		header: PlyHeader,
		private readonly fileName: string,
	) {
		this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
		this.little = header.format !== "binary_big_endian";
		this.offset = header.dataStart;
		this.ascii =
			header.format === "ascii"
				? new TextDecoder()
						.decode(data.subarray(header.dataStart))
						.split(/\s+/)
						.filter((s) => s !== "")
				: null;
	}

	readScalar(type: PlyScalar): number {
		if (this.ascii !== null) {
			const raw = this.ascii[this.token++];
			if (raw === undefined)
				throw new MLIOException("the PLY data section ended early", this.fileName);
			const x = Number(raw);
			if (Number.isNaN(x)) {
				throw new MLIOException(`non-numeric PLY value "${raw}"`, this.fileName);
			}
			return x;
		}
		const o = this.offset;
		const size = SCALAR_BYTES[type];
		if (o + size > this.data.byteLength) {
			throw new MLIOException("the PLY data section ended early", this.fileName);
		}
		this.offset += size;
		switch (type) {
			case "char":
			case "int8":
				return this.view.getInt8(o);
			case "uchar":
			case "uint8":
				return this.view.getUint8(o);
			case "short":
			case "int16":
				return this.view.getInt16(o, this.little);
			case "ushort":
			case "uint16":
				return this.view.getUint16(o, this.little);
			case "int":
			case "int32":
				return this.view.getInt32(o, this.little);
			case "uint":
			case "uint32":
				return this.view.getUint32(o, this.little);
			case "float":
			case "float32":
				return this.view.getFloat32(o, this.little);
			case "double":
			case "float64":
				return this.view.getFloat64(o, this.little);
			default:
				throw new MLIOException(`unknown PLY scalar type "${type}"`, this.fileName);
		}
	}

	skipScalar(type: PlyScalar): void {
		this.readScalar(type);
	}
}

export interface PlyReadResult {
	/** `MM_*` bits for the attributes the file actually carried. */
	mask: number;
}

export function readPly(mesh: CMeshO, data: Uint8Array, fileName = ""): PlyReadResult {
	const header = parsePlyHeader(data, fileName);
	const reader = new PlyDataReader(data, header, fileName);

	const vertexEl = header.elements.find((e) => e.name === "vertex");
	if (vertexEl === undefined) {
		throw new MLIOException("the PLY file has no vertex element", fileName);
	}
	const faceEl = header.elements.find((e) => e.name === "face");

	let mask = MeshElement.MM_VERTCOORD;
	const has = (n: string) => vertexEl.properties.some((p) => p.name === n);
	const hasNormal = has("nx") && has("ny") && has("nz");
	const hasColor = has("red") && has("green") && has("blue");
	const hasQuality = has("quality");
	if (hasNormal) mask |= MeshElement.MM_VERTNORMAL;
	if (hasColor) mask |= MeshElement.MM_VERTCOLOR;
	if (hasQuality) mask |= MeshElement.MM_VERTQUALITY;

	const firstVert = Allocator.addVertices(mesh, vertexEl.count);

	// Elements before "vertex" have to be consumed, not skipped over, because
	// their widths are what put us at the right offset.
	for (const el of header.elements) {
		if (el === vertexEl) break;
		consumeElement(reader, el);
	}

	for (let i = 0; i < vertexEl.count; i++) {
		const v = firstVert + i;
		let x = 0,
			y = 0,
			z = 0;
		let nx = 0,
			ny = 0,
			nz = 0;
		let r = 255,
			g = 255,
			b = 255,
			a = 255;
		for (const p of vertexEl.properties) {
			if (p.countType !== undefined) {
				const n = reader.readScalar(p.countType);
				for (let k = 0; k < n; k++) reader.skipScalar(p.type);
				continue;
			}
			const value = reader.readScalar(p.type);
			switch (p.name) {
				case "x":
					x = value;
					break;
				case "y":
					y = value;
					break;
				case "z":
					z = value;
					break;
				case "nx":
					nx = value;
					break;
				case "ny":
					ny = value;
					break;
				case "nz":
					nz = value;
					break;
				case "red":
				case "r":
					r = value;
					break;
				case "green":
				case "g":
					g = value;
					break;
				case "blue":
				case "b":
					b = value;
					break;
				case "alpha":
					a = value;
					break;
				case "quality":
					mesh.vertQuality[v] = value;
					break;
				default:
					break; // an attribute we do not model
			}
		}
		mesh.setVert(v, x, y, z);
		if (hasNormal) {
			mesh.vertNormal[3 * v] = nx;
			mesh.vertNormal[3 * v + 1] = ny;
			mesh.vertNormal[3 * v + 2] = nz;
		}
		if (hasColor) mesh.vertColor[v] = ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
	}

	if (faceEl !== undefined) {
		// Anything between vertex and face still has to be consumed.
		let reached = false;
		for (const el of header.elements) {
			if (el === vertexEl) {
				reached = true;
				continue;
			}
			if (!reached || el === faceEl) continue;
			if (header.elements.indexOf(el) > header.elements.indexOf(faceEl)) break;
			consumeElement(reader, el);
		}

		mask |= MeshElement.MM_FACEVERT;
		for (let i = 0; i < faceEl.count; i++) {
			for (const p of faceEl.properties) {
				if (p.countType === undefined) {
					reader.skipScalar(p.type);
					continue;
				}
				const n = reader.readScalar(p.countType);
				const idx: number[] = [];
				for (let k = 0; k < n; k++) idx.push(reader.readScalar(p.type));
				if (p.name !== "vertex_indices" && p.name !== "vertex_index") continue;
				// Fan-triangulate: PLY happily stores quads and n-gons, and
				// CMeshO is a triangle mesh.
				for (let k = 2; k < idx.length; k++) {
					Allocator.addFace(mesh, firstVert + idx[0], firstVert + idx[k - 1], firstVert + idx[k]);
				}
			}
		}
	}

	return { mask };
}

function consumeElement(reader: PlyDataReader, el: PlyElement): void {
	for (let i = 0; i < el.count; i++) {
		for (const p of el.properties) {
			if (p.countType === undefined) {
				reader.skipScalar(p.type);
				continue;
			}
			const n = reader.readScalar(p.countType);
			for (let k = 0; k < n; k++) reader.skipScalar(p.type);
		}
	}
}

export interface PlySaveOptions {
	readonly binary?: boolean;
	readonly saveNormals?: boolean;
	readonly saveColors?: boolean;
	readonly saveQuality?: boolean;
}

export function writePly(mesh: CMeshO, options: PlySaveOptions = {}): Uint8Array {
	const binary = options.binary ?? true;
	const withNormals = options.saveNormals ?? false;
	const withColors = options.saveColors ?? false;
	const withQuality = options.saveQuality ?? false;

	// Deleted slots are not written, so the file's indices are the compacted
	// ones and every face has to be remapped through this table.
	const remap = new Int32Array(mesh.vertSize).fill(-1);
	let n = 0;
	for (let v = 0; v < mesh.vertSize; v++) if (!mesh.isVertD(v)) remap[v] = n++;

	const header: string[] = [
		"ply",
		`format ${binary ? "binary_little_endian" : "ascii"} 1.0`,
		"comment written by meshlab-ts",
		`element vertex ${n}`,
		"property double x",
		"property double y",
		"property double z",
	];
	if (withNormals) header.push("property double nx", "property double ny", "property double nz");
	if (withColors) {
		header.push(
			"property uchar red",
			"property uchar green",
			"property uchar blue",
			"property uchar alpha",
		);
	}
	if (withQuality) header.push("property double quality");
	header.push(
		`element face ${mesh.fn}`,
		"property list uchar int vertex_indices",
		"end_header",
		"",
	);

	const headerBytes = new TextEncoder().encode(header.join("\n"));

	if (!binary) {
		const lines: string[] = [];
		for (let v = 0; v < mesh.vertSize; v++) {
			if (mesh.isVertD(v)) continue;
			const row = [mesh.vx(v), mesh.vy(v), mesh.vz(v)];
			if (withNormals) {
				row.push(mesh.vertNormal[3 * v], mesh.vertNormal[3 * v + 1], mesh.vertNormal[3 * v + 2]);
			}
			if (withColors) {
				const c = mesh.vertColor[v];
				row.push(c & 0xff, (c >>> 8) & 0xff, (c >>> 16) & 0xff, (c >>> 24) & 0xff);
			}
			if (withQuality) row.push(mesh.vertQuality[v]);
			lines.push(row.join(" "));
		}
		for (let f = 0; f < mesh.faceSize; f++) {
			if (mesh.isFaceD(f)) continue;
			lines.push(`3 ${remap[mesh.fv(f, 0)]} ${remap[mesh.fv(f, 1)]} ${remap[mesh.fv(f, 2)]}`);
		}
		return concat(headerBytes, new TextEncoder().encode(`${lines.join("\n")}\n`));
	}

	const vertexBytes = 24 + (withNormals ? 24 : 0) + (withColors ? 4 : 0) + (withQuality ? 8 : 0);
	const body = new Uint8Array(n * vertexBytes + mesh.fn * 13);
	const view = new DataView(body.buffer);
	let off = 0;
	for (let v = 0; v < mesh.vertSize; v++) {
		if (mesh.isVertD(v)) continue;
		view.setFloat64(off, mesh.vx(v), true);
		view.setFloat64(off + 8, mesh.vy(v), true);
		view.setFloat64(off + 16, mesh.vz(v), true);
		off += 24;
		if (withNormals) {
			for (let k = 0; k < 3; k++) view.setFloat64(off + k * 8, mesh.vertNormal[3 * v + k], true);
			off += 24;
		}
		if (withColors) {
			const c = mesh.vertColor[v];
			view.setUint8(off, c & 0xff);
			view.setUint8(off + 1, (c >>> 8) & 0xff);
			view.setUint8(off + 2, (c >>> 16) & 0xff);
			view.setUint8(off + 3, (c >>> 24) & 0xff);
			off += 4;
		}
		if (withQuality) {
			view.setFloat64(off, mesh.vertQuality[v], true);
			off += 8;
		}
	}
	for (let f = 0; f < mesh.faceSize; f++) {
		if (mesh.isFaceD(f)) continue;
		view.setUint8(off, 3);
		for (let k = 0; k < 3; k++) view.setInt32(off + 1 + k * 4, remap[mesh.fv(f, k)], true);
		off += 13;
	}
	return concat(headerBytes, body);
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
	const out = new Uint8Array(a.length + b.length);
	out.set(a, 0);
	out.set(b, a.length);
	return out;
}
