/**
 * Rasterising a mesh into its own texture, and filling in what is left.
 *
 * A parametrised mesh maps each face to a triangle in texture space. Baking an
 * attribute into a texture means walking that triangle's pixels, recovering
 * the barycentric coordinates, and asking the mesh what it looks like there.
 *
 * Two details decide whether the result is usable.
 *
 * The **v flip**: texture coordinates put v = 0 at the bottom, an image puts
 * row 0 at the top. Everything in this file converts once, at the boundary.
 *
 * The **gutter**: a triangle covers a pixel only when the pixel's centre is
 * inside it, so the pixels straddling a UV seam stay empty. Sampling the
 * texture with any filtering at all then bleeds the background in along every
 * seam. {@link pullPushFill} is upstream's answer — build a mip pyramid of
 * what *is* filled, then push it back down into the holes, so the empty space
 * carries a smooth continuation of the nearest real colour instead of black.
 */

import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { blue, green, red, rgba } from "../../vcg/space/color4.ts";
import { Image } from "../../vcg/space/image/image.ts";

/** Where a pixel falls on the mesh. */
export interface Sample {
	readonly face: number;
	/** Barycentric weights on the face's three corners, summing to one. */
	readonly bary: readonly [number, number, number];
	readonly x: number;
	readonly y: number;
}

/** The wedge texture coordinates of one face, as (u, v) pairs. */
export function faceUV(cm: CMeshO, f: number): [number, number][] {
	const wt = cm.wedgeTexCoord;
	if (wt === null) throw new Error("the mesh has no per-wedge texture coordinates");
	return [
		[wt[6 * f], wt[6 * f + 1]],
		[wt[6 * f + 2], wt[6 * f + 3]],
		[wt[6 * f + 4], wt[6 * f + 5]],
	];
}

export function setFaceUV(cm: CMeshO, f: number, uv: readonly (readonly [number, number])[]): void {
	const wt = cm.wedgeTexCoord;
	if (wt === null) throw new Error("the mesh has no per-wedge texture coordinates");
	for (let k = 0; k < 3; k++) {
		wt[6 * f + 2 * k] = uv[k][0];
		wt[6 * f + 2 * k + 1] = uv[k][1];
	}
}

/**
 * Visits every pixel whose centre lies inside the face's UV triangle.
 *
 * A degenerate triangle — zero area in texture space, which a collapsed
 * parametrisation produces — is skipped rather than divided by.
 */
export function rasteriseFace(
	cm: CMeshO,
	f: number,
	width: number,
	height: number,
	visit: (sample: Sample) => void,
): void {
	const uv = faceUV(cm, f);
	// To pixel centres: u across, v up, so the row index is flipped.
	const px = uv.map(([u, v]) => [u * width - 0.5, (1 - v) * height - 0.5] as const);

	const area =
		(px[1][0] - px[0][0]) * (px[2][1] - px[0][1]) - (px[2][0] - px[0][0]) * (px[1][1] - px[0][1]);
	if (area === 0) return;

	const minX = Math.max(0, Math.floor(Math.min(px[0][0], px[1][0], px[2][0])));
	const maxX = Math.min(width - 1, Math.ceil(Math.max(px[0][0], px[1][0], px[2][0])));
	const minY = Math.max(0, Math.floor(Math.min(px[0][1], px[1][1], px[2][1])));
	const maxY = Math.min(height - 1, Math.ceil(Math.max(px[0][1], px[1][1], px[2][1])));

	for (let y = minY; y <= maxY; y++) {
		for (let x = minX; x <= maxX; x++) {
			const w0 = ((px[1][0] - x) * (px[2][1] - y) - (px[2][0] - x) * (px[1][1] - y)) / area;
			const w1 = ((px[2][0] - x) * (px[0][1] - y) - (px[0][0] - x) * (px[2][1] - y)) / area;
			const w2 = 1 - w0 - w1;
			// A small negative tolerance keeps the pixels exactly on a shared
			// edge from being dropped by both of the faces that meet there.
			if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
			visit({ face: f, bary: [w0, w1, w2], x, y });
		}
	}
}

/**
 * Pull-push hole filling.
 *
 * The "pull" half halves the image repeatedly, averaging only the pixels that
 * carry real data, so each level is a coarser but more complete picture. The
 * "push" half walks back down, and wherever the finer level is still empty it
 * takes a weighted blend of the coarser level — 144 for the pixel above it and
 * 48/48/16 for its neighbours, which is a smooth reconstruction rather than a
 * blocky upscale. The weights are upstream's.
 */
export function pullPushFill(image: Image, background: number): void {
	if (image.width < 2 || image.height < 2) return;
	const half = new Image(Math.floor(image.width / 2), Math.floor(image.height / 2), background);
	pullPushMip(image, half, background);
	pullPushFill(half, background);
	pullPushPush(image, half, background);
}

/** The "pull": average the four children that carry data. */
function pullPushMip(fine: Image, coarse: Image, background: number): void {
	for (let y = 0; y < coarse.height; y++) {
		for (let x = 0; x < coarse.width; x++) {
			const children = [
				fine.pixel(2 * x, 2 * y),
				fine.pixel(2 * x + 1, 2 * y),
				fine.pixel(2 * x, 2 * y + 1),
				fine.pixel(2 * x + 1, 2 * y + 1),
			];
			const weights = children.map((c) => (c === background ? 0 : 255));
			if (weights.some((w) => w > 0)) coarse.setPixel(x, y, blend(children, weights, background));
		}
	}
}

/** The "push": fill each remaining hole from the level above. */
function pullPushPush(fine: Image, coarse: Image, background: number): void {
	const get = (x: number, y: number) =>
		x >= 0 && y >= 0 && x < coarse.width && y < coarse.height ? coarse.pixel(x, y) : background;

	for (let y = 0; y < coarse.height; y++) {
		for (let x = 0; x < coarse.width; x++) {
			// Each of the four children leans towards its own corner of the
			// coarse neighbourhood, which is what keeps the fill from looking
			// like a doubled pixel.
			const corners: Array<[number, number, number, number]> = [
				[0, 0, -1, -1],
				[1, 0, 1, -1],
				[0, 1, -1, 1],
				[1, 1, 1, 1],
			];
			for (const [dx, dy, sx, sy] of corners) {
				const fx = 2 * x + dx;
				const fy = 2 * y + dy;
				if (!fine.inside(fx, fy) || fine.pixel(fx, fy) !== background) continue;
				const neighbours = [get(x, y), get(x + sx, y), get(x, y + sy), get(x + sx, y + sy)];
				const weights = [144, 48, 48, 16].map((w, i) => (neighbours[i] === background ? 0 : w));
				if (weights.some((w) => w > 0)) {
					fine.setPixel(fx, fy, blend(neighbours, weights, background));
				}
			}
		}
	}
}

function blend(colours: readonly number[], weights: readonly number[], fallback: number): number {
	let total = 0;
	const sums = [0, 0, 0, 0];
	for (let i = 0; i < colours.length; i++) {
		const w = weights[i];
		if (w === 0) continue;
		total += w;
		sums[0] += red(colours[i]) * w;
		sums[1] += green(colours[i]) * w;
		sums[2] += blue(colours[i]) * w;
		sums[3] += ((colours[i] >>> 24) & 0xff) * w;
	}
	if (total === 0) return fallback;
	return rgba(
		Math.round(sums[0] / total),
		Math.round(sums[1] / total),
		Math.round(sums[2] / total),
		Math.round(sums[3] / total),
	);
}

/**
 * A checkerboard or grid, for `Set Texture`'s dummy pattern.
 *
 * The point of it is to make a parametrisation visible: a checker shows how
 * the texels are stretched, a grid shows where the seams fall.
 */
export function dummyTexture(size: number, checkSize: number, grid: boolean): Image {
	const image = new Image(size, size, rgba(255, 255, 255, 255));
	const step = Math.max(1, checkSize);
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			if (grid) {
				if (x % step === 0 || y % step === 0) image.setPixel(x, y, rgba(0, 0, 0, 255));
			} else if ((Math.floor(x / step) + Math.floor(y / step)) % 2 === 0) {
				image.setPixel(x, y, rgba(0, 0, 0, 255));
			}
		}
	}
	return image;
}
