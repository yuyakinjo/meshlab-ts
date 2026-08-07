/**
 * A plain 8-bit RGBA raster.
 *
 * MeshLab leans on `QImage` for its textures. There is no Qt here, and none of
 * what the texture filters need goes beyond reading and writing pixels, so
 * this is a width, a height and a byte array — deliberately not an image
 * library. Colours are the same packed `Color4b` the mesh uses, so a vertex
 * colour can be written into a texture without conversion.
 */

import type { Color4b } from "../../complex/cmesho.ts";
import { blue, green, red, rgba } from "../color4.ts";

export class Image {
	readonly width: number;
	readonly height: number;
	/** Row-major RGBA, four bytes per pixel. */
	readonly data: Uint8Array;

	constructor(width: number, height: number, fill?: Color4b) {
		if (width <= 0 || height <= 0 || !Number.isInteger(width) || !Number.isInteger(height)) {
			throw new Error(`an image needs positive integer dimensions, got ${width}x${height}`);
		}
		this.width = width;
		this.height = height;
		this.data = new Uint8Array(width * height * 4);
		if (fill !== undefined) this.fill(fill);
	}

	static fromRgba(width: number, height: number, data: Uint8Array): Image {
		const image = new Image(width, height);
		if (data.length !== image.data.length) {
			throw new Error(
				`expected ${image.data.length} bytes for ${width}x${height}, got ${data.length}`,
			);
		}
		image.data.set(data);
		return image;
	}

	private at(x: number, y: number): number {
		return (y * this.width + x) * 4;
	}

	inside(x: number, y: number): boolean {
		return x >= 0 && y >= 0 && x < this.width && y < this.height;
	}

	pixel(x: number, y: number): Color4b {
		const i = this.at(x, y);
		return rgba(this.data[i], this.data[i + 1], this.data[i + 2], this.data[i + 3]);
	}

	setPixel(x: number, y: number, colour: Color4b): void {
		const i = this.at(x, y);
		this.data[i] = red(colour);
		this.data[i + 1] = green(colour);
		this.data[i + 2] = blue(colour);
		this.data[i + 3] = (colour >>> 24) & 0xff;
	}

	fill(colour: Color4b): void {
		for (let y = 0; y < this.height; y++) {
			for (let x = 0; x < this.width; x++) this.setPixel(x, y, colour);
		}
	}

	clone(): Image {
		return Image.fromRgba(this.width, this.height, this.data);
	}

	/**
	 * Bilinear sample at a texture coordinate, with u to the right and v
	 * *upwards*.
	 *
	 * The flip matters: OBJ, PLY and MeshLab all put v = 0 at the bottom of
	 * the image while a raster's row 0 is the top. Sampling without it gives a
	 * texture that is subtly upside down, which reads as a bad parametrisation
	 * rather than as a bug in the sampler.
	 */
	sample(u: number, v: number): Color4b {
		const x = wrap(u) * (this.width - 1);
		const y = (1 - wrap(v)) * (this.height - 1);
		const x0 = Math.floor(x);
		const y0 = Math.floor(y);
		const x1 = Math.min(this.width - 1, x0 + 1);
		const y1 = Math.min(this.height - 1, y0 + 1);
		const fx = x - x0;
		const fy = y - y0;

		const channel = (offset: number) => {
			const a =
				this.data[this.at(x0, y0) + offset] * (1 - fx) + this.data[this.at(x1, y0) + offset] * fx;
			const b =
				this.data[this.at(x0, y1) + offset] * (1 - fx) + this.data[this.at(x1, y1) + offset] * fx;
			return Math.round(a * (1 - fy) + b * fy);
		};
		return rgba(channel(0), channel(1), channel(2), channel(3));
	}
}

/**
 * Texture coordinates repeat outside 0..1, as they do everywhere else in
 * graphics — but 1 itself stays 1.
 *
 * Plain `t - floor(t)` sends 1 to 0, which for a bake that covers exactly the
 * unit square means the far edge samples the near one. That is a one-texel
 * seam down two sides of every texture, and it looks like a parametrisation
 * problem rather than a sampling one.
 */
function wrap(t: number): number {
	if (!Number.isFinite(t)) return 0;
	if (t >= 0 && t <= 1) return t;
	const f = t - Math.floor(t);
	return f < 0 ? f + 1 : f;
}
