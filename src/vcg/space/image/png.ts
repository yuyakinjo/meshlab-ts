/**
 * PNG, read and written.
 *
 * The texture filters have to produce a file somebody can open and read one
 * somebody produced, and PNG is the only format in MeshLab's list that is both
 * lossless and simple enough to implement honestly. `node:zlib` supplies the
 * DEFLATE half, which is the part that would otherwise be prohibitive; what is
 * left is the chunk framing and the per-scanline filters.
 *
 * What is supported on read: 8-bit greyscale, RGB, palette, greyscale+alpha
 * and RGBA, non-interlaced. Everything comes back as RGBA. 16-bit samples are
 * truncated to 8 with a note in the error path rather than silently, and Adam7
 * interlacing is refused — it is rare for a texture, and a half-right
 * de-interlacer would produce a plausible-looking wrong image.
 *
 * Writing always emits 8-bit RGBA, non-interlaced, with the Paeth filter on
 * every row. Trying every filter per row the way libpng does would compress a
 * few percent better for noticeably more code.
 */

import { deflateSync, inflateSync } from "node:zlib";
import { Image } from "./image.ts";

const MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export function isPng(bytes: Uint8Array): boolean {
	if (bytes.length < MAGIC.length) return false;
	for (let i = 0; i < MAGIC.length; i++) if (bytes[i] !== MAGIC[i]) return false;
	return true;
}

// ---- writing --------------------------------------------------------------

export function writePng(image: Image): Uint8Array {
	const raw = new Uint8Array(image.height * (image.width * 4 + 1));
	const stride = image.width * 4;
	for (let y = 0; y < image.height; y++) {
		const to = y * (stride + 1);
		raw[to] = 4; // Paeth
		for (let x = 0; x < stride; x++) {
			const a = x >= 4 ? image.data[y * stride + x - 4] : 0;
			const b = y > 0 ? image.data[(y - 1) * stride + x] : 0;
			const c = x >= 4 && y > 0 ? image.data[(y - 1) * stride + x - 4] : 0;
			raw[to + 1 + x] = (image.data[y * stride + x] - paeth(a, b, c)) & 0xff;
		}
	}

	const ihdr = new Uint8Array(13);
	const view = new DataView(ihdr.buffer);
	view.setUint32(0, image.width);
	view.setUint32(4, image.height);
	ihdr[8] = 8; // bit depth
	ihdr[9] = 6; // colour type: RGBA
	// 10..12 stay zero: deflate, adaptive filtering, no interlace.

	const chunks = [
		new Uint8Array(MAGIC),
		chunk("IHDR", ihdr),
		chunk("IDAT", new Uint8Array(deflateSync(raw))),
		chunk("IEND", new Uint8Array(0)),
	];
	const total = chunks.reduce((n, c) => n + c.length, 0);
	const out = new Uint8Array(total);
	let at = 0;
	for (const c of chunks) {
		out.set(c, at);
		at += c.length;
	}
	return out;
}

function chunk(type: string, payload: Uint8Array): Uint8Array {
	const out = new Uint8Array(12 + payload.length);
	const view = new DataView(out.buffer);
	view.setUint32(0, payload.length);
	for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
	out.set(payload, 8);
	view.setUint32(8 + payload.length, crc32(out.subarray(4, 8 + payload.length)));
	return out;
}

// ---- reading --------------------------------------------------------------

export function readPng(bytes: Uint8Array): Image {
	if (!isPng(bytes)) throw new Error("not a PNG file");
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

	let width = 0;
	let height = 0;
	let depth = 0;
	let colourType = 0;
	let palette: Uint8Array | null = null;
	let transparency: Uint8Array | null = null;
	const idat: Uint8Array[] = [];

	let at = 8;
	while (at + 8 <= bytes.length) {
		const length = view.getUint32(at);
		const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
		const body = bytes.subarray(at + 8, at + 8 + length);
		if (type === "IHDR") {
			width = view.getUint32(at + 8);
			height = view.getUint32(at + 12);
			depth = body[8];
			colourType = body[9];
			if (body[12] !== 0) throw new Error("interlaced PNGs are not supported");
			if (depth !== 8) {
				throw new Error(`only 8-bit PNGs are supported, this one is ${depth}-bit`);
			}
		} else if (type === "PLTE") {
			palette = body.slice();
		} else if (type === "tRNS") {
			transparency = body.slice();
		} else if (type === "IDAT") {
			idat.push(body);
		} else if (type === "IEND") {
			break;
		}
		at += 12 + length;
	}

	if (width === 0 || height === 0) throw new Error("the PNG has no IHDR");
	if (idat.length === 0) throw new Error("the PNG has no image data");

	const channels = CHANNELS[colourType];
	if (channels === undefined) throw new Error(`unsupported PNG colour type ${colourType}`);

	const joined = new Uint8Array(idat.reduce((n, c) => n + c.length, 0));
	let offset = 0;
	for (const c of idat) {
		joined.set(c, offset);
		offset += c.length;
	}
	const raw = new Uint8Array(inflateSync(joined));

	const stride = width * channels;
	if (raw.length < height * (stride + 1)) {
		throw new Error("the PNG's image data is shorter than its header claims");
	}
	const flat = unfilter(raw, width, height, channels);

	const image = new Image(width, height);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const from = y * stride + x * channels;
			const to = (y * width + x) * 4;
			expand(flat, from, colourType, palette, transparency, image.data, to);
		}
	}
	return image;
}

const CHANNELS: Readonly<Record<number, number>> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function expand(
	src: Uint8Array,
	from: number,
	colourType: number,
	palette: Uint8Array | null,
	transparency: Uint8Array | null,
	dst: Uint8Array,
	to: number,
): void {
	switch (colourType) {
		case 0: // greyscale
			dst[to] = src[from];
			dst[to + 1] = src[from];
			dst[to + 2] = src[from];
			dst[to + 3] = 255;
			return;
		case 2: // RGB
			dst[to] = src[from];
			dst[to + 1] = src[from + 1];
			dst[to + 2] = src[from + 2];
			dst[to + 3] = 255;
			return;
		case 3: {
			// palette; tRNS, when present, is one alpha byte per entry
			if (palette === null) throw new Error("a palette PNG with no PLTE chunk");
			const i = src[from] * 3;
			dst[to] = palette[i];
			dst[to + 1] = palette[i + 1];
			dst[to + 2] = palette[i + 2];
			dst[to + 3] =
				transparency !== null && src[from] < transparency.length ? transparency[src[from]] : 255;
			return;
		}
		case 4: // greyscale + alpha
			dst[to] = src[from];
			dst[to + 1] = src[from];
			dst[to + 2] = src[from];
			dst[to + 3] = src[from + 1];
			return;
		default: // 6, RGBA
			dst[to] = src[from];
			dst[to + 1] = src[from + 1];
			dst[to + 2] = src[from + 2];
			dst[to + 3] = src[from + 3];
	}
}

/**
 * Undoes the per-scanline filters, in place into a fresh buffer.
 *
 * Each row picks its own filter and predicts from the row above and the pixel
 * to the left, so the rows must be undone in order — the reason a PNG cannot
 * be decoded a row at a time out of sequence.
 */
function unfilter(raw: Uint8Array, width: number, height: number, channels: number): Uint8Array {
	const stride = width * channels;
	const out = new Uint8Array(height * stride);
	for (let y = 0; y < height; y++) {
		const type = raw[y * (stride + 1)];
		const from = y * (stride + 1) + 1;
		const to = y * stride;
		for (let x = 0; x < stride; x++) {
			const value = raw[from + x];
			const a = x >= channels ? out[to + x - channels] : 0;
			const b = y > 0 ? out[to - stride + x] : 0;
			const c = x >= channels && y > 0 ? out[to - stride + x - channels] : 0;
			let predicted = 0;
			switch (type) {
				case 0:
					predicted = 0;
					break;
				case 1:
					predicted = a;
					break;
				case 2:
					predicted = b;
					break;
				case 3:
					predicted = (a + b) >> 1;
					break;
				case 4:
					predicted = paeth(a, b, c);
					break;
				default:
					throw new Error(`unknown PNG row filter ${type} on row ${y}`);
			}
			out[to + x] = (value + predicted) & 0xff;
		}
	}
	return out;
}

/** The Paeth predictor: whichever of left, above and above-left is closest. */
function paeth(a: number, b: number, c: number): number {
	const p = a + b - c;
	const pa = Math.abs(p - a);
	const pb = Math.abs(p - b);
	const pc = Math.abs(p - c);
	if (pa <= pb && pa <= pc) return a;
	return pb <= pc ? b : c;
}

// ---- CRC ------------------------------------------------------------------

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();

function crc32(bytes: Uint8Array): number {
	let c = 0xffffffff;
	for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}
