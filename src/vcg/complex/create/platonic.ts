/**
 * `vcg/complex/algorithms/create/platonic.h` — the primitive builders behind
 * MeshLab's `filter_create`.
 *
 * These are transcriptions rather than reinventions: the vertex order, the
 * face order and the winding all follow the C++, so a mesh created here has
 * the same indices as the same mesh created by MeshLab. That matters more than
 * it looks — a great deal of downstream test material is "create a sphere,
 * apply a filter, compare", and it only holds if the sphere is the same sphere.
 */

import { Allocator } from "../allocator.ts";
import { Clean } from "../clean.ts";
import { CMeshO } from "../cmesho.ts";
import { VertexFlag } from "../flags.ts";
import { Smooth } from "../smooth.ts";
import { UpdateTopology } from "../update/topology.ts";

/** Builds a mesh from a flat coordinate list and a flat vertex-index list. */
function build(coords: readonly number[], faces: readonly number[]): CMeshO {
	const m = new CMeshO();
	const vertCount = coords.length / 3;
	if (vertCount > 0) {
		const first = Allocator.addVertices(m, vertCount);
		for (let v = 0; v < vertCount; v++) {
			m.setVert(first + v, coords[3 * v], coords[3 * v + 1], coords[3 * v + 2]);
		}
	}
	const faceCount = faces.length / 3;
	if (faceCount > 0) {
		const first = Allocator.addFaces(m, faceCount);
		for (let f = 0; f < faceCount; f++) {
			m.setFace(first + f, faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]);
		}
	}
	return m;
}

export function tetrahedron(): CMeshO {
	// biome-ignore format: one vertex, then one face, per group
	return build(
		[1, 1, 1,   -1, 1, -1,   -1, -1, 1,   1, -1, -1],
		[0, 1, 2,   0, 2, 3,   0, 3, 1,   3, 2, 1],
	);
}

export function octahedron(): CMeshO {
	// biome-ignore format: one vertex, then one face, per group
	return build(
		[1, 0, 0,   0, 1, 0,   0, 0, 1,   -1, 0, 0,   0, -1, 0,   0, 0, -1],
		[
			0, 1, 2,   0, 2, 4,   0, 4, 5,   0, 5, 1,
			3, 1, 5,   3, 5, 4,   3, 4, 2,   3, 2, 1,
		],
	);
}

export function icosahedron(): CMeshO {
	const L = (Math.sqrt(5) + 1) / 2;
	// biome-ignore format: laid out one vertex, then one face, per line
	return build(
		[
			0, L, 1,   0, L, -1,   0, -L, 1,   0, -L, -1,
			L, 1, 0,   L, -1, 0,   -L, 1, 0,   -L, -1, 0,
			1, 0, L,   -1, 0, L,   1, 0, -L,   -1, 0, -L,
		],
		[
			1, 0, 4,    0, 1, 6,    2, 3, 5,    3, 2, 7,
			4, 5, 10,   5, 4, 8,    6, 7, 9,    7, 6, 11,
			8, 9, 2,    9, 8, 0,    10, 11, 1,  11, 10, 3,
			0, 8, 4,    0, 6, 9,    1, 4, 10,   1, 11, 6,
			2, 5, 8,    2, 9, 7,    3, 10, 5,   3, 7, 11,
		],
	);
}

/** The 12 pentagons, as vertex indices into {@link DODECAHEDRON_COORDS}. */
const DODECAHEDRON_PENTAGONS: ReadonlyArray<readonly number[]> = [
	[0, 8, 10, 2, 16],
	[0, 16, 17, 1, 12],
	[0, 12, 14, 4, 8],
	[5, 14, 12, 1, 9],
	[5, 19, 18, 4, 14],
	[5, 9, 11, 7, 19],
	[3, 11, 9, 1, 17],
	[3, 13, 15, 7, 11],
	[3, 17, 16, 2, 13],
	[6, 18, 19, 7, 15],
	[6, 15, 13, 2, 10],
	[6, 10, 8, 4, 18],
];

function dodecahedronCoords(): number[] {
	const phi = (1 + Math.sqrt(5)) / 2;
	const a = 1 / Math.sqrt(3);
	const b = a / phi;
	const c = a * phi;
	// biome-ignore format: one vertex per column group, as the three families of coordinates
	return [
		a, a, a,     a, a, -a,    a, -a, a,    a, -a, -a,
		-a, a, a,    -a, a, -a,   -a, -a, a,   -a, -a, -a,
		0, b, c,     0, b, -c,    0, -b, c,    0, -b, -c,
		b, c, 0,     b, -c, 0,    -b, c, 0,    -b, -c, 0,
		c, 0, b,     c, 0, -b,    -c, 0, b,    -c, 0, -b,
	];
}

/**
 * A dodecahedron, each pentagon fanned into 3 triangles.
 *
 * The fan does not preserve the solid's symmetry; {@link dodecahedronSym}
 * is the version that does, at the cost of an extra vertex per face.
 */
export function dodecahedron(): CMeshO {
	const faces: number[] = [];
	for (const p of DODECAHEDRON_PENTAGONS) {
		faces.push(p[0], p[1], p[2], p[0], p[2], p[3], p[0], p[3], p[4]);
	}
	return build(dodecahedronCoords(), faces);
}

/**
 * A dodecahedron star-triangulated from a vertex at the centre of each
 * pentagon, so the triangulation has the symmetry the solid does.
 */
export function dodecahedronSym(): CMeshO {
	const coords = dodecahedronCoords();
	const m = build(coords, []);
	const centre: number[] = [];
	for (const p of DODECAHEDRON_PENTAGONS) {
		let sx = 0;
		let sy = 0;
		let sz = 0;
		for (const v of p) {
			sx += coords[3 * v];
			sy += coords[3 * v + 1];
			sz += coords[3 * v + 2];
		}
		centre.push(Allocator.addVertex(m, sx / 5, sy / 5, sz / 5));
	}
	for (const [i, p] of DODECAHEDRON_PENTAGONS.entries()) {
		for (let j = 0; j < 5; j++) Allocator.addFace(m, centre[i], p[j], p[(j + 1) % 5]);
	}
	return m;
}

/** An axis-aligned box spanning the given corners. */
export function box(
	low: readonly [number, number, number] = [-0.5, -0.5, -0.5],
	high: readonly [number, number, number] = [0.5, 0.5, 0.5],
): CMeshO {
	const c = (x: number, y: number, z: number) => [
		x ? high[0] : low[0],
		y ? high[1] : low[1],
		z ? high[2] : low[2],
	];
	// Vertex order is the C++ one: x fastest, then y, then z.
	const coords: number[] = [];
	for (let z = 0; z < 2; z++) {
		for (let y = 0; y < 2; y++) {
			for (let x = 0; x < 2; x++) coords.push(...c(x, y, z));
		}
	}
	// biome-ignore format: two triangles per line, one line per face of the box
	return build(coords, [
		2, 1, 0,   1, 2, 3,
		4, 2, 0,   2, 4, 6,
		1, 4, 0,   4, 1, 5,
		6, 5, 7,   5, 6, 4,
		3, 6, 7,   6, 3, 2,
		5, 3, 7,   3, 5, 1,
	]);
}

/** Splits every triangle into 4 on its edge midpoints, welding shared edges. */
function midpointRefine(m: CMeshO): CMeshO {
	const out = new CMeshO();
	const midpoint = new Map<number, number>();
	const copy = new Int32Array(m.vertSize).fill(-1);

	const carry = (v: number): number => {
		if (copy[v] < 0) copy[v] = Allocator.addVertex(out, m.vx(v), m.vy(v), m.vz(v));
		return copy[v];
	};
	const between = (a: number, b: number): number => {
		const key = a < b ? a * m.vertSize + b : b * m.vertSize + a;
		const seen = midpoint.get(key);
		if (seen !== undefined) return seen;
		const made = Allocator.addVertex(
			out,
			(m.vx(a) + m.vx(b)) / 2,
			(m.vy(a) + m.vy(b)) / 2,
			(m.vz(a) + m.vz(b)) / 2,
		);
		midpoint.set(key, made);
		return made;
	};

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const v = [m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)].map(carry);
		const e = [
			between(m.fv(f, 0), m.fv(f, 1)),
			between(m.fv(f, 1), m.fv(f, 2)),
			between(m.fv(f, 2), m.fv(f, 0)),
		];
		Allocator.addFace(out, v[0], e[0], e[2]);
		Allocator.addFace(out, v[1], e[1], e[0]);
		Allocator.addFace(out, v[2], e[2], e[1]);
		Allocator.addFace(out, e[0], e[1], e[2]);
	}
	return out;
}

/**
 * A geodesic sphere of unit radius: an icosahedron refined `subdiv` times,
 * every vertex pushed back out to the sphere after each pass.
 */
export function sphere(subdiv = 3): CMeshO {
	let m = icosahedron();
	normalizeVertices(m);
	for (let i = 0; i < subdiv; i++) {
		m = midpointRefine(m);
		normalizeVertices(m);
	}
	return m;
}

function normalizeVertices(m: CMeshO): void {
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		const len = Math.hypot(m.vx(v), m.vy(v), m.vz(v));
		if (len > 0) m.setVert(v, m.vx(v) / len, m.vy(v) / len, m.vz(v) / len);
	}
}

/**
 * A spherical dome subtended by a cone of `angleRad`.
 *
 * Built from a hexagon rather than from a piece of a sphere: refine it, push
 * the rim onto the unit circle, relax the interior, and only then bend the
 * disc up into the cap. Doing it this way keeps the triangles even, which
 * projecting a flat triangulation would not.
 */
export function sphericalCap(angleRad: number, subdiv = 3): CMeshO {
	let m = new CMeshO();
	Allocator.addVertex(m, 0, 0, 0);
	for (let i = 0; i < 6; i++) {
		const a = (i * 60 * Math.PI) / 180;
		Allocator.addVertex(m, Math.cos(a), Math.sin(a), 0);
	}
	for (let i = 0; i < 6; i++) Allocator.addFace(m, 0, 1 + i, 1 + ((i + 1) % 6));

	for (let i = 0; i < subdiv; i++) {
		m = midpointRefine(m);
		UpdateTopology.faceFace(m);
		// Only the rim is pushed out; the interior is then relaxed to follow,
		// which is what keeps the triangles from bunching up at the edge.
		for (let v = 0; v < m.vertSize; v++) {
			if (m.isVertD(v) || (m.vertFlags[v] & VertexFlag.BORDER) === 0) continue;
			const len = Math.hypot(m.vx(v), m.vy(v), m.vz(v));
			if (len > 0) m.setVert(v, m.vx(v) / len, m.vy(v) / len, m.vz(v) / len);
		}
		Smooth.vertexCoordLaplacian(m, 10, { pinBoundary: true });
	}

	const halfAngle = angleRad / 2;
	const width = Math.sin(halfAngle);
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		m.setVert(v, m.vx(v) * width, m.vy(v) * width, m.vz(v) * width);
	}
	Allocator.compactEveryVector(m);
	// Lift each point of the flat disc onto the sphere it belongs to.
	for (let v = 0; v < m.vertSize; v++) {
		const r = Math.hypot(m.vx(v), m.vy(v), m.vz(v));
		m.setVert(v, m.vx(v), m.vy(v), Math.cos(Math.asin(Math.min(1, r))) - Math.cos(halfAngle));
	}
	return m;
}

/**
 * A cone or frustum about the Y axis, from radius `r1` at `-h/2` to `r2` at
 * `+h/2`. Either radius may be zero, which closes that end to a point; equal
 * radii give a cylinder.
 */
export function cone(r1: number, r2: number, h: number, subdiv = 36): CMeshO {
	const m = new CMeshO();
	Allocator.addVertex(m, 0, -h / 2, 0);
	Allocator.addVertex(m, 0, h / 2, 0);
	const b1 = 2;
	let b2 = 2;
	if (r1 !== 0) {
		for (let i = 0; i < subdiv; i++) {
			const a = (i * 2 * Math.PI) / subdiv;
			Allocator.addVertex(m, r1 * Math.cos(a), -h / 2, r1 * Math.sin(a));
		}
		b2 += subdiv;
	}
	if (r2 !== 0) {
		for (let i = 0; i < subdiv; i++) {
			const a = (i * 2 * Math.PI) / subdiv;
			Allocator.addVertex(m, r2 * Math.cos(a), h / 2, r2 * Math.sin(a));
		}
	}

	const wrap = (base: number, i: number) => base + ((i + 1) % subdiv);
	if (r1 !== 0) {
		for (let i = 0; i < subdiv; i++) Allocator.addFace(m, 0, b1 + i, wrap(b1, i));
	}
	if (r2 !== 0) {
		for (let i = 0; i < subdiv; i++) Allocator.addFace(m, 1, wrap(b2, i), b2 + i);
	}
	if (r1 === 0) {
		for (let i = 0; i < subdiv; i++) Allocator.addFace(m, 0, b2 + i, wrap(b2, i));
	}
	if (r2 === 0) {
		for (let i = 0; i < subdiv; i++) Allocator.addFace(m, 1, wrap(b1, i), b1 + i);
	}
	if (r1 !== 0 && r2 !== 0) {
		for (let i = 0; i < subdiv; i++) {
			Allocator.addFace(m, b1 + i, b2 + i, wrap(b2, i));
			Allocator.addFace(m, b1 + i, wrap(b2, i), wrap(b1, i));
		}
	}
	return m;
}

/** Lays a triangulated grid over `w * h` vertices already in the mesh. */
function faceGrid(m: CMeshO, w: number, h: number): void {
	for (let i = 0; i < h - 1; i++) {
		for (let j = 0; j < w - 1; j++) {
			Allocator.addFace(m, (i + 1) * w + j + 1, i * w + j + 1, i * w + j);
			Allocator.addFace(m, i * w + j, (i + 1) * w + j, (i + 1) * w + j + 1);
		}
	}
}

/** A torus about the Z axis: a tube of radius `vRadius` swept at `hRadius`. */
export function torus(hRadius: number, vRadius: number, hDiv = 24, vDiv = 12): CMeshO {
	const m = new CMeshO();
	const stepV = (2 * Math.PI) / vDiv;
	const stepH = (2 * Math.PI) / hDiv;
	Allocator.addVertices(m, (vDiv + 1) * (hDiv + 1));
	for (let i = 0; i < hDiv + 1; i++) {
		const a = (i % hDiv) * stepH;
		const ca = Math.cos(a);
		const sa = Math.sin(a);
		for (let j = 0; j < vDiv + 1; j++) {
			const x = vRadius * Math.cos((j % vDiv) * stepV) + hRadius;
			const z = vRadius * Math.sin((j % vDiv) * stepV);
			// The C++ builds this with a rotation matrix about Z; with y = 0 the
			// product is just this.
			m.setVert(i * (vDiv + 1) + j, x * ca, x * sa, z);
		}
	}
	faceGrid(m, vDiv + 1, hDiv + 1);
	// The seam row and column duplicate the first; welding closes the torus.
	Clean.removeDuplicateVertex(m);
	Allocator.compactEveryVector(m);
	return m;
}

/**
 * A flat ring in the XY plane between two concentric circles.
 *
 * The argument order is VCGLib's — outer radius first — and MeshLab calls it
 * with its own `internalRadius` first, so the two circles come out swapped
 * relative to the parameter names. That is upstream behaviour, and the ring it
 * describes is the same one either way.
 */
export function annulus(externalRadius: number, internalRadius: number, slices: number): CMeshO {
	const m = new CMeshO();
	Allocator.addVertices(m, slices * 2);
	for (let j = 0; j < slices; j++) {
		const a = ((2 * Math.PI) / slices) * j;
		const x = Math.cos(a);
		const y = Math.sin(a);
		m.setVert(2 * j, x * internalRadius, y * internalRadius, 0);
		m.setVert(2 * j + 1, x * externalRadius, y * externalRadius, 0);
	}
	const n = slices * 2;
	for (let j = 0; j < slices; j++) {
		Allocator.addFace(m, (j * 2 + 0) % n, ((j + 1) * 2 + 1) % n, (j * 2 + 1) % n);
		Allocator.addFace(m, ((j + 1) * 2 + 0) % n, ((j + 1) * 2 + 1) % n, (j * 2 + 0) % n);
	}
	return m;
}

/** A vertex-only mesh holding the given points, each with itself as its normal. */
export function pointCloudFrom(points: ReadonlyArray<readonly [number, number, number]>): CMeshO {
	const m = new CMeshO();
	if (points.length === 0) return m;
	const first = Allocator.addVertices(m, points.length);
	for (let i = 0; i < points.length; i++) {
		const [x, y, z] = points[i];
		m.setVert(first + i, x, y, z);
		// MeshLab's `AddVertex(cm, p, p)` — on the unit sphere the position is
		// the normal, which is what makes these clouds reconstructable.
		m.vertNormal[3 * (first + i)] = x;
		m.vertNormal[3 * (first + i) + 1] = y;
		m.vertNormal[3 * (first + i) + 2] = z;
	}
	return m;
}

export const Platonic = {
	tetrahedron,
	octahedron,
	icosahedron,
	dodecahedron,
	dodecahedronSym,
	box,
	sphere,
	sphericalCap,
	cone,
	torus,
	annulus,
	pointCloudFrom,
} as const;
