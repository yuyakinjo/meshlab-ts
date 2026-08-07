/**
 * How badly a parametrisation distorts the surface.
 *
 * A map from a curved surface to the plane cannot preserve both angles and
 * areas unless the surface is developable, so every parametrisation trades
 * one against the other and every measure here reports one side of that
 * trade. `vcg::tri::Distortion`.
 *
 * All of them read the *per-wedge* texture coordinates, which is where every
 * parametrisation in this library writes. A face with no UV assigned reads as
 * a degenerate triangle and is reported as such rather than skipped, because
 * a parametrisation with holes in it is a fact the caller wants to know.
 */

import type { CMeshO } from "../cmesho.ts";

/** The 3D area of a face. */
export function area3D(cm: CMeshO, f: number): number {
	const p = corners3D(cm, f);
	const u = sub(p[1], p[0]);
	const v = sub(p[2], p[0]);
	return (
		Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]) / 2
	);
}

/** The signed area of a face in texture space; negative means folded over. */
export function signedAreaUV(cm: CMeshO, f: number): number {
	const p = cornersUV(cm, f);
	return (
		((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])) / 2
	);
}

export function areaUV(cm: CMeshO, f: number): number {
	return Math.abs(signedAreaUV(cm, f));
}

/**
 * True when the face is flipped in texture space.
 *
 * A single folded triangle makes the parametrisation non-injective, so this
 * is the first thing to check before trusting any of the numbers below.
 */
export function isFolded(cm: CMeshO, f: number): boolean {
	return signedAreaUV(cm, f) < 0;
}

/** How many faces are folded — zero for a valid parametrisation. */
export function foldedNum(cm: CMeshO): number {
	let n = 0;
	for (let f = 0; f < cm.faceSize; f++) if (!cm.isFaceD(f) && isFolded(cm, f)) n++;
	return n;
}

/**
 * True when no face is folded *and* they all agree on which way is up.
 *
 * A parametrisation whose faces are all negative is a mirror image, which is
 * perfectly usable; one with a mixture is not.
 */
export function globallyUnfolded(cm: CMeshO): boolean {
	let positive = 0;
	let negative = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const area = signedAreaUV(cm, f);
		if (area > 0) positive++;
		else if (area < 0) negative++;
	}
	return positive === 0 || negative === 0;
}

export function edgeLength3D(cm: CMeshO, f: number, e: number): number {
	const p = corners3D(cm, f);
	const a = p[e];
	const b = p[(e + 1) % 3];
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function edgeLengthUV(cm: CMeshO, f: number, e: number): number {
	const p = cornersUV(cm, f);
	const a = p[e];
	const b = p[(e + 1) % 3];
	return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** The interior angle at corner `e`, in radians. */
export function angleRad3D(cm: CMeshO, f: number, e: number): number {
	const p = corners3D(cm, f);
	return angleBetween(sub(p[(e + 1) % 3], p[e]), sub(p[(e + 2) % 3], p[e]));
}

export function angleRadUV(cm: CMeshO, f: number, e: number): number {
	const p = cornersUV(cm, f).map((q) => [q[0], q[1], 0]);
	return angleBetween(sub(p[(e + 1) % 3], p[e]), sub(p[(e + 2) % 3], p[e]));
}

/**
 * The total angle a face's corners are off by, in radians.
 *
 * Zero for a conformal map. Since the three angles of both triangles sum to
 * π, this is a genuine measure of shape change and not of scale.
 */
export function angleDistortion(cm: CMeshO, f: number): number {
	let sum = 0;
	for (let e = 0; e < 3; e++) sum += Math.abs(angleRad3D(cm, f, e) - angleRadUV(cm, f, e));
	return sum;
}

/**
 * The factor that makes the parametrisation's total area match the surface's.
 *
 * Every area comparison needs it: a map can be perfectly area-preserving up
 * to a global scale, and without normalising, that scale would swamp the
 * distortion it is meant to expose.
 */
export function meshScalingFactor(cm: CMeshO): { area3D: number; areaUV: number; ratio: number } {
	let total3D = 0;
	let totalUV = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		total3D += area3D(cm, f);
		totalUV += areaUV(cm, f);
	}
	return { area3D: total3D, areaUV: totalUV, ratio: totalUV === 0 ? 1 : total3D / totalUV };
}

/**
 * How much a face's area changed, as a signed fraction.
 *
 * Positive means the face grew in texture space relative to its share of the
 * surface, negative that it shrank. Reported relative to whichever of the two
 * is larger, so a face that doubled and a face that halved come back with the
 * same magnitude rather than 1 and 0.5.
 */
export function areaDistortion(cm: CMeshO, f: number, ratio: number): number {
	const a3 = area3D(cm, f);
	const a2 = areaUV(cm, f) * ratio;
	if (a3 === 0 && a2 === 0) return 0;
	if (a3 === 0 || a2 === 0) return 1;
	return a2 > a3 ? a2 / a3 - 1 : -(a3 / a2 - 1);
}

/** The same idea for one edge's length. */
export function edgeDistortion(cm: CMeshO, f: number, e: number, ratio: number): number {
	const l3 = edgeLength3D(cm, f, e);
	const l2 = edgeLengthUV(cm, f, e) * ratio;
	if (l3 === 0 && l2 === 0) return 0;
	if (l3 === 0 || l2 === 0) return 1;
	return l2 > l3 ? l2 / l3 - 1 : -(l3 / l2 - 1);
}

/**
 * Sander's L2 stretch, squared, for one face.
 *
 * The map's Jacobian has two singular values — how much the surface is
 * stretched along each principal direction. L2 stretch is their quadratic
 * mean, so it penalises stretching in *either* direction, unlike an area
 * measure which a map that stretches one way and squashes the other would
 * pass with full marks.
 */
export function l2StretchEnergySquared(cm: CMeshO, f: number, areaScale: number): number {
	const q = corners3D(cm, f);
	// `areaScale` is an *area* ratio, so the UV coordinates take its square
	// root. Scaling the area alone would leave the Jacobian inconsistent and
	// make a plain uniform shrink read as a distortion, which it is not.
	const scale = Math.sqrt(areaScale);
	const p = cornersUV(cm, f).map((c) => [c[0] * scale, c[1] * scale]);
	const a = areaUV(cm, f) * areaScale;
	if (a === 0) return Number.POSITIVE_INFINITY;

	// The Jacobian's columns, from Sander et al.'s closed form.
	const s = [0, 0, 0];
	const t = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		s[k] =
			(q[0][k] * (p[1][1] - p[2][1]) +
				q[1][k] * (p[2][1] - p[0][1]) +
				q[2][k] * (p[0][1] - p[1][1])) /
			(2 * a);
		t[k] =
			(q[0][k] * (p[2][0] - p[1][0]) +
				q[1][k] * (p[0][0] - p[2][0]) +
				q[2][k] * (p[1][0] - p[0][0])) /
			(2 * a);
	}
	return (dot(s, s) + dot(t, t)) / 2;
}

/** The worse of the two singular values: the largest local stretch. */
export function lInfStretchEnergy(cm: CMeshO, f: number, areaScale: number): number {
	const q = corners3D(cm, f);
	const scale = Math.sqrt(areaScale);
	const p = cornersUV(cm, f).map((c) => [c[0] * scale, c[1] * scale]);
	const a = areaUV(cm, f) * areaScale;
	if (a === 0) return Number.POSITIVE_INFINITY;

	const s = [0, 0, 0];
	const t = [0, 0, 0];
	for (let k = 0; k < 3; k++) {
		s[k] =
			(q[0][k] * (p[1][1] - p[2][1]) +
				q[1][k] * (p[2][1] - p[0][1]) +
				q[2][k] * (p[0][1] - p[1][1])) /
			(2 * a);
		t[k] =
			(q[0][k] * (p[2][0] - p[1][0]) +
				q[1][k] * (p[0][0] - p[2][0]) +
				q[2][k] * (p[1][0] - p[0][0])) /
			(2 * a);
	}
	const A = dot(s, s);
	const B = dot(s, t);
	const C = dot(t, t);
	// The larger eigenvalue of [[A,B],[B,C]].
	const discriminant = Math.sqrt(Math.max(0, (A - C) * (A - C) + 4 * B * B));
	return Math.sqrt(Math.max(0, (A + C + discriminant) / 2));
}

/** The area-weighted mean of a per-face measure over the whole mesh. */
export function meshMean(cm: CMeshO, measure: (f: number) => number): number {
	let sum = 0;
	let weight = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const a = area3D(cm, f);
		sum += measure(f) * a;
		weight += a;
	}
	return weight === 0 ? 0 : sum / weight;
}

/** The mesh-wide angle distortion, area weighted. */
export function meshAngleDistortion(cm: CMeshO): number {
	return meshMean(cm, (f) => angleDistortion(cm, f));
}

/** The mesh-wide L2 stretch, as Sander defines it: area weighted, then rooted. */
export function meshL2Stretch(cm: CMeshO): number {
	const { ratio } = meshScalingFactor(cm);
	return Math.sqrt(meshMean(cm, (f) => l2StretchEnergySquared(cm, f, ratio)));
}

// ---- small helpers --------------------------------------------------------

function corners3D(cm: CMeshO, f: number): number[][] {
	return [0, 1, 2].map((k) => {
		const v = cm.fv(f, k);
		return [cm.vx(v), cm.vy(v), cm.vz(v)];
	});
}

function cornersUV(cm: CMeshO, f: number): number[][] {
	const wt = cm.wedgeTexCoord;
	if (wt === null) throw new Error("the mesh has no per-wedge texture coordinates");
	return [0, 1, 2].map((k) => [wt[6 * f + 2 * k], wt[6 * f + 2 * k + 1]]);
}

function sub(a: readonly number[], b: readonly number[]): number[] {
	return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: readonly number[], b: readonly number[]): number {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function angleBetween(u: readonly number[], v: readonly number[]): number {
	const lu = Math.hypot(u[0], u[1], u[2]);
	const lv = Math.hypot(v[0], v[1], v[2]);
	if (lu === 0 || lv === 0) return 0;
	return Math.acos(Math.min(1, Math.max(-1, dot(u, v) / (lu * lv))));
}
