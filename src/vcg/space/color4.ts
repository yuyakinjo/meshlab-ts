/**
 * `vcg/space/color4.h` — colours, and the per-channel arithmetic the colour
 * filters are built from.
 *
 * A colour is one packed number in `0xAABBGGRR` order, the same byte order
 * `vcg::Color4b` uses, so a colour read from a PLY needs no rearranging.
 * Everything here works on that packing directly rather than boxing each
 * colour into an object: these run once per vertex over meshes with millions
 * of them.
 */

import type { Color4b } from "../complex/cmesho.ts";

export const RED_CHANNEL = 1;
export const GREEN_CHANNEL = 2;
export const BLUE_CHANNEL = 4;
export const ALL_CHANNELS = RED_CHANNEL | GREEN_CHANNEL | BLUE_CHANNEL;

export const red = (c: Color4b): number => c & 0xff;
export const green = (c: Color4b): number => (c >>> 8) & 0xff;
export const blue = (c: Color4b): number => (c >>> 16) & 0xff;
export const alpha = (c: Color4b): number => (c >>> 24) & 0xff;

/** Packs four 0..255 channels, clamping each. */
export function rgba(r: number, g: number, b: number, a = 255): Color4b {
	return ((clamp255(r) | (clamp255(g) << 8) | (clamp255(b) << 16) | (clamp255(a) << 24)) >>>
		0) as Color4b;
}

const clamp255 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v) | 0);

export const BLACK = rgba(0, 0, 0);
export const WHITE_COLOR = rgba(255, 255, 255);

/** Linear blend, `t` from 0 (all `a`) to 1 (all `b`). Alpha rides along. */
export function lerpColor(a: Color4b, b: Color4b, t: number): Color4b {
	const k = t < 0 ? 0 : t > 1 ? 1 : t;
	return rgba(
		red(a) + (red(b) - red(a)) * k,
		green(a) + (green(b) - green(a)) * k,
		blue(a) + (blue(b) - blue(a)) * k,
		alpha(a) + (alpha(b) - alpha(a)) * k,
	);
}

/** `(max + min) / 2` — the HSL sense of "how light is this". */
export const lightness = (c: Color4b): number =>
	(Math.max(red(c), green(c), blue(c)) + Math.min(red(c), green(c), blue(c))) / 2;

/** The perceptual weighting: green counts for most of what the eye sees. */
export const luminosity = (c: Color4b): number =>
	0.2126 * red(c) + 0.7152 * green(c) + 0.0722 * blue(c);

export const averageLightness = (c: Color4b): number => (red(c) + green(c) + blue(c)) / 3;

/** How MeshLab's "Desaturation method" enum is ordered. */
export const DesaturationMethod = {
	Lightness: 0,
	Luminosity: 1,
	Average: 2,
} as const;

export function desaturate(c: Color4b, method: number): Color4b {
	const v =
		method === DesaturationMethod.Average
			? averageLightness(c)
			: method === DesaturationMethod.Luminosity
				? luminosity(c)
				: lightness(c);
	return rgba(v, v, v, 255);
}

export function invert(c: Color4b): Color4b {
	return rgba(255 - red(c), 255 - green(c), 255 - blue(c), alpha(c));
}

/**
 * Brightness then contrast, both in the -1..1 range MeshLab normalises to.
 *
 * The contrast curve is `tan((contrast + 1) · π/4)`, which is 1 at contrast 0
 * and runs to infinity at 1 — so full contrast is a hard threshold rather
 * than merely a steep ramp.
 */
export function brightnessContrast(c: Color4b, brightness: number, contrast: number): Color4b {
	const apply = (channel: number): number => {
		let v = channel / 255;
		v = brightness < 0 ? v * (1 + brightness) : v + (1 - v) * brightness;
		v = (v - 0.5) * Math.tan(((contrast + 1) * Math.PI) / 4) + 0.5;
		return 255 * v;
	};
	return rgba(apply(red(c)), apply(green(c)), apply(blue(c)), 255);
}

/** The levels tool: crop the input range, apply gamma, rescale to the output range. */
export function levels(
	c: Color4b,
	gamma: number,
	inMin: number,
	inMax: number,
	outMin: number,
	outMax: number,
	channelMask: number,
): Color4b {
	const apply = (channel: number): number => {
		let v = channel / 255;
		// The denominator is floored at one 8-bit step, so an inverted or
		// collapsed input range cannot divide by zero.
		v = clamp01(v - inMin) / clampRange(inMax - inMin);
		v = v ** (1 / gamma);
		return (v * (outMax - outMin) + outMin) * 255;
	};
	return rgba(
		channelMask & RED_CHANNEL ? apply(red(c)) : red(c),
		channelMask & GREEN_CHANNEL ? apply(green(c)) : green(c),
		channelMask & BLUE_CHANNEL ? apply(blue(c)) : blue(c),
		255,
	);
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const clampRange = (v: number): number => (v < 1 / 255 ? 1 / 255 : v > 1 ? 1 : v);

/**
 * Rescales each channel so that `unbalancedWhite` would come out white.
 *
 * The point of a grey-card correction: name the colour the camera recorded
 * for something you know to be white, and everything else follows.
 */
export function whiteBalance(c: Color4b, unbalancedWhite: Color4b): Color4b {
	// A zero channel would mean "this white had no red at all", which cannot be
	// corrected for; treating it as 1 keeps the arithmetic finite.
	const r = Math.max(1, red(unbalancedWhite));
	const g = Math.max(1, green(unbalancedWhite));
	const b = Math.max(1, blue(unbalancedWhite));
	return rgba((red(c) * 255) / r, (green(c) * 255) / g, (blue(c) * 255) / b, 255);
}

/** HSV to a packed colour; `h` in degrees, `s` and `v` in 0..1. */
export function fromHsv(h: number, s: number, v: number): Color4b {
	const hue = ((h % 360) + 360) % 360;
	const c = v * s;
	const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
	const m = v - c;
	const sector = Math.floor(hue / 60) % 6;
	const table: ReadonlyArray<readonly [number, number, number]> = [
		[c, x, 0],
		[x, c, 0],
		[0, c, x],
		[0, x, c],
		[x, 0, c],
		[c, 0, x],
	];
	const [r, g, b] = table[sector];
	return rgba((r + m) * 255, (g + m) * 255, (b + m) * 255, 255);
}

/**
 * VCGLib's quality ramp: red at the bottom, through yellow, green and cyan, to
 * blue at the top.
 *
 * Four equal steps rather than a hue sweep, which is why the green band is so
 * wide — it is the convention every MeshLab quality visualisation uses, so
 * matching it matters more than the ramp being perceptually even.
 */
export function colorRamp(minf: number, maxf: number, value: number): Color4b {
	if (minf > maxf) return colorRamp(maxf, minf, maxf + (minf - value));
	const RED = rgba(255, 0, 0);
	const YELLOW = rgba(255, 255, 0);
	const GREEN = rgba(0, 255, 0);
	const CYAN = rgba(0, 255, 255);
	const BLUE = rgba(0, 0, 255);

	const step = (maxf - minf) / 4;
	if (value < minf) return RED;
	if (step === 0) return value > minf ? BLUE : RED;
	let v = value - minf;
	for (const [from, to] of [
		[RED, YELLOW],
		[YELLOW, GREEN],
		[GREEN, CYAN],
		[CYAN, BLUE],
	] as const) {
		if (v < step) return lerpColor(from, to, v / step);
		v -= step;
	}
	return BLUE;
}

/**
 * VCG's `Color4::Scatter`: the `value`-th of `range` colours, ordered so that
 * successive values land far apart on the hue circle.
 *
 * The loop is a bit-reversal — it walks `value` down a binary subdivision of
 * `[0, range)` and reassembles the hue from the branches taken. Consecutive
 * values therefore differ in the *high* bit of the result, which is what makes
 * neighbouring layers or components easy to tell apart, where a plain
 * `value / range` sweep would give two nearly identical colours.
 */
export function scatter(range: number, value: number, sat = 0.3, val = 0.9): Color4b {
	let b = 0;
	let m = range;
	let v = value;
	for (let k = 1; k < range; k <<= 1) {
		if (v * 2 >= m) {
			b += k;
			v -= (m + 1) >> 1;
			m >>= 1;
		} else {
			m = (m + 1) >> 1;
		}
	}
	return fromHsv((360 * b) / Math.max(1, range), sat, val);
}

/**
 * Histogram equalisation over a set of colours.
 *
 * Returns a per-channel lookup table: build it from whichever vertices are in
 * scope, then push every colour through {@link equalizeColor}. Splitting it in
 * two is what lets the "only on selection" flag mean *both* that the histogram
 * is built from the selection and that only the selection is rewritten.
 *
 * With no channel selected the caller is asking to equalise lightness, and the
 * fourth table covers that case.
 */
export interface EqualizeTables {
	readonly cdf: readonly Int32Array[];
}

/** Index of the lightness table in {@link EqualizeTables.cdf}. */
export const LIGHTNESS_TABLE = 3;

export function buildEqualizeTables(colors: Iterable<Color4b>): EqualizeTables {
	const cdf = [new Int32Array(256), new Int32Array(256), new Int32Array(256), new Int32Array(256)];
	for (const c of colors) {
		cdf[0][red(c)]++;
		cdf[1][green(c)]++;
		cdf[2][blue(c)]++;
		cdf[LIGHTNESS_TABLE][Math.min(255, Math.round(lightness(c)))]++;
	}
	for (const table of cdf) {
		for (let i = 1; i < 256; i++) table[i] += table[i - 1];
	}
	return { cdf };
}

function equalizeValue(table: Int32Array, v: number): number {
	// The span can be zero when every colour in scope shares one value; there is
	// nothing to stretch then, so the channel passes through unchanged.
	const span = table[255] - table[0];
	return span === 0 ? v : ((table[v] - table[0]) / span) * 255;
}

export function equalizeColor(c: Color4b, tables: EqualizeTables, channelMask: number): Color4b {
	const { cdf } = tables;
	if (channelMask === 0) {
		const l = Math.min(255, Math.round(lightness(c)));
		const v = equalizeValue(cdf[LIGHTNESS_TABLE], l);
		return rgba(v, v, v, 255);
	}
	return rgba(
		channelMask & RED_CHANNEL ? equalizeValue(cdf[0], red(c)) : red(c),
		channelMask & GREEN_CHANNEL ? equalizeValue(cdf[1], green(c)) : green(c),
		channelMask & BLUE_CHANNEL ? equalizeValue(cdf[2], blue(c)) : blue(c),
		255,
	);
}

export const Color4 = {
	red,
	green,
	blue,
	alpha,
	rgba,
	lerpColor,
	lightness,
	luminosity,
	averageLightness,
	desaturate,
	invert,
	brightnessContrast,
	levels,
	whiteBalance,
	fromHsv,
	colorRamp,
	scatter,
	buildEqualizeTables,
	equalizeColor,
} as const;
