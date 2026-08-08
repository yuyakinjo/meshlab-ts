/**
 * Procedural noise, and the five fractal functions built on it.
 *
 * The base is Perlin's improved gradient noise: smooth, band-limited, and
 * repeatable from a seed. The fractal functions above it are Musgrave's, from
 * *Texturing and Modeling: A Procedural Approach*, and they are what MeshLab's
 * fractal filters offer. They differ in one idea:
 *
 * - **fBm** sums octaves at decreasing amplitude. Statistically the same
 *   everywhere — good for rolling ground, wrong for mountains.
 * - **Standard multifractal** *multiplies* the octaves together, so a region
 *   that is already low stays smooth and a region that is high gets rougher.
 *   That is what real terrain does.
 * - **Heterogeneous** and **hybrid multifractal** weight each octave by the
 *   value accumulated so far, in two different ways; both give plains and
 *   crags in the same field.
 * - **Ridged multifractal** folds each octave through `offset - |n|` before
 *   squaring it, which turns the zero crossings into sharp ridges.
 *
 * Everything is deterministic in the seed. A filter whose output changed
 * between runs would make every downstream hash useless, which is the same
 * reason `filter_func` refuses muParser's `rnd`.
 */

/** The fractal functions, in MeshLab's own order. */
export const FractalAlgorithm = {
	FBM: 0,
	STANDARD_MF: 1,
	HETERO_MF: 2,
	HYBRID_MF: 3,
	RIDGED_MF: 4,
} as const;

export interface FractalOptions {
	readonly algorithm: number;
	readonly octaves: number;
	readonly lacunarity: number;
	/** The fractal increment H: larger means the octaves fade faster. */
	readonly fractalIncrement: number;
	readonly offset: number;
	readonly gain: number;
	readonly seed: number;
}

export const FRACTAL_DEFAULTS: FractalOptions = {
	algorithm: FractalAlgorithm.RIDGED_MF,
	octaves: 8,
	lacunarity: 4,
	fractalIncrement: 0.5,
	offset: 0.9,
	gain: 2.5,
	seed: 2,
};

/** Perlin gradient noise over a permutation table built from a seed. */
export class PerlinNoise {
	private readonly permutation: Uint8Array;

	constructor(seed: number) {
		// A seeded Fisher-Yates shuffle of 0..255, doubled so the lookups in
		// `at` never need a modulo.
		const table = new Uint8Array(256);
		for (let i = 0; i < 256; i++) table[i] = i;
		const random = mulberry32(seed >>> 0 || 1);
		for (let i = 255; i > 0; i--) {
			const j = Math.floor(random() * (i + 1));
			const t = table[i];
			table[i] = table[j];
			table[j] = t;
		}
		this.permutation = new Uint8Array(512);
		for (let i = 0; i < 512; i++) this.permutation[i] = table[i & 255];
	}

	/** Noise at a point, in roughly -1..1. */
	at(x: number, y: number, z: number): number {
		const xi = Math.floor(x) & 255;
		const yi = Math.floor(y) & 255;
		const zi = Math.floor(z) & 255;
		const xf = x - Math.floor(x);
		const yf = y - Math.floor(y);
		const zf = z - Math.floor(z);
		const u = fade(xf);
		const v = fade(yf);
		const w = fade(zf);
		const p = this.permutation;

		const a = p[xi] + yi;
		const aa = p[a] + zi;
		const ab = p[a + 1] + zi;
		const b = p[xi + 1] + yi;
		const ba = p[b] + zi;
		const bb = p[b + 1] + zi;

		return lerp(
			w,
			lerp(
				v,
				lerp(u, grad(p[aa], xf, yf, zf), grad(p[ba], xf - 1, yf, zf)),
				lerp(u, grad(p[ab], xf, yf - 1, zf), grad(p[bb], xf - 1, yf - 1, zf)),
			),
			lerp(
				v,
				lerp(u, grad(p[aa + 1], xf, yf, zf - 1), grad(p[ba + 1], xf - 1, yf, zf - 1)),
				lerp(u, grad(p[ab + 1], xf, yf - 1, zf - 1), grad(p[bb + 1], xf - 1, yf - 1, zf - 1)),
			),
		);
	}
}

/**
 * One of Musgrave's fractal functions, evaluated at a point.
 *
 * The exponent table — how much each octave contributes — depends only on the
 * options, so a caller evaluating over a whole mesh should build the
 * {@link FractalField} once rather than calling this in a loop.
 */
export class FractalField {
	private readonly noise: PerlinNoise;
	private readonly exponents: Float64Array;
	private readonly options: FractalOptions;

	constructor(options: FractalOptions) {
		if (options.octaves < 1) {
			throw new Error(`a fractal needs at least one octave, got ${options.octaves}`);
		}
		if (options.lacunarity <= 0) {
			throw new Error(`the lacunarity must be positive, got ${options.lacunarity}`);
		}
		this.options = options;
		this.noise = new PerlinNoise(options.seed);
		const count = Math.ceil(options.octaves) + 1;
		this.exponents = new Float64Array(count);
		let frequency = 1;
		for (let i = 0; i < count; i++) {
			this.exponents[i] = frequency ** -options.fractalIncrement;
			frequency *= options.lacunarity;
		}
	}

	at(x: number, y: number, z: number): number {
		switch (this.options.algorithm) {
			case FractalAlgorithm.STANDARD_MF:
				return this.standardMultifractal(x, y, z);
			case FractalAlgorithm.HETERO_MF:
				return this.heteroMultifractal(x, y, z);
			case FractalAlgorithm.HYBRID_MF:
				return this.hybridMultifractal(x, y, z);
			case FractalAlgorithm.RIDGED_MF:
				return this.ridgedMultifractal(x, y, z);
			default:
				return this.fBm(x, y, z);
		}
	}

	private fBm(x: number, y: number, z: number): number {
		let value = 0;
		let px = x;
		let py = y;
		let pz = z;
		const whole = Math.floor(this.options.octaves);
		for (let i = 0; i < whole; i++) {
			value += this.noise.at(px, py, pz) * this.exponents[i];
			px *= this.options.lacunarity;
			py *= this.options.lacunarity;
			pz *= this.options.lacunarity;
		}
		// The fractional part of the octave count fades the last one in, so
		// sweeping the octaves is continuous rather than stepped.
		const rest = this.options.octaves - whole;
		if (rest > 0) value += rest * this.noise.at(px, py, pz) * this.exponents[whole];
		return value;
	}

	private standardMultifractal(x: number, y: number, z: number): number {
		let value = 1;
		let px = x;
		let py = y;
		let pz = z;
		const whole = Math.floor(this.options.octaves);
		for (let i = 0; i < whole; i++) {
			value *= this.options.offset * this.exponents[i] * this.noise.at(px, py, pz);
			px *= this.options.lacunarity;
			py *= this.options.lacunarity;
			pz *= this.options.lacunarity;
		}
		const rest = this.options.octaves - whole;
		if (rest > 0) {
			value *= rest * this.noise.at(px, py, pz) * this.exponents[whole];
		}
		return value;
	}

	private heteroMultifractal(x: number, y: number, z: number): number {
		let px = x;
		let py = y;
		let pz = z;
		let value = this.options.offset + this.noise.at(px, py, pz);
		px *= this.options.lacunarity;
		py *= this.options.lacunarity;
		pz *= this.options.lacunarity;

		const whole = Math.floor(this.options.octaves);
		for (let i = 1; i < whole; i++) {
			// Each octave is scaled by what has accumulated, so a low region
			// stays smooth and a high one keeps gaining detail.
			const increment = (this.options.offset + this.noise.at(px, py, pz)) * this.exponents[i];
			value += increment * value;
			px *= this.options.lacunarity;
			py *= this.options.lacunarity;
			pz *= this.options.lacunarity;
		}
		const rest = this.options.octaves - whole;
		if (rest > 0) {
			const increment = (this.options.offset + this.noise.at(px, py, pz)) * this.exponents[whole];
			value += rest * increment * value;
		}
		return value;
	}

	private hybridMultifractal(x: number, y: number, z: number): number {
		let px = x;
		let py = y;
		let pz = z;
		let value = (this.options.offset + this.noise.at(px, py, pz)) * this.exponents[0];
		let weight = value;
		px *= this.options.lacunarity;
		py *= this.options.lacunarity;
		pz *= this.options.lacunarity;

		const whole = Math.floor(this.options.octaves);
		for (let i = 1; i < whole; i++) {
			if (weight > 1) weight = 1;
			const signal = (this.options.offset + this.noise.at(px, py, pz)) * this.exponents[i];
			value += weight * signal;
			weight *= signal;
			px *= this.options.lacunarity;
			py *= this.options.lacunarity;
			pz *= this.options.lacunarity;
		}
		const rest = this.options.octaves - whole;
		if (rest > 0) {
			const signal = (this.options.offset + this.noise.at(px, py, pz)) * this.exponents[whole];
			value += rest * weight * signal;
		}
		return value;
	}

	private ridgedMultifractal(x: number, y: number, z: number): number {
		let px = x;
		let py = y;
		let pz = z;
		// `offset - |n|` turns the noise's zero crossings — smooth lines —
		// into creases, and squaring sharpens them into ridges.
		let signal = ridge(this.noise.at(px, py, pz), this.options.offset);
		let value = signal * this.exponents[0];
		let weight = 1;
		px *= this.options.lacunarity;
		py *= this.options.lacunarity;
		pz *= this.options.lacunarity;

		const whole = Math.floor(this.options.octaves);
		for (let i = 1; i < whole; i++) {
			weight = Math.max(0, Math.min(1, signal * this.options.gain));
			signal = ridge(this.noise.at(px, py, pz), this.options.offset) * weight;
			value += signal * this.exponents[i];
			px *= this.options.lacunarity;
			py *= this.options.lacunarity;
			pz *= this.options.lacunarity;
		}
		const rest = this.options.octaves - whole;
		if (rest > 0) {
			weight = Math.max(0, Math.min(1, signal * this.options.gain));
			signal = ridge(this.noise.at(px, py, pz), this.options.offset) * weight;
			value += rest * signal * this.exponents[whole];
		}
		return value;
	}
}

function ridge(n: number, offset: number): number {
	const s = offset - Math.abs(n);
	return s * s;
}

function fade(t: number): number {
	return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(t: number, a: number, b: number): number {
	return a + t * (b - a);
}

function grad(hash: number, x: number, y: number, z: number): number {
	// Perlin's improved gradients: the twelve edge midpoints of a cube.
	const h = hash & 15;
	const u = h < 8 ? x : y;
	const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
	return (h & 1 ? -u : u) + (h & 2 ? -v : v);
}

/** Mulberry32: small, fast, and reproducible from a seed. */
export function mulberry32(seed: number): () => number {
	let a = seed;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
