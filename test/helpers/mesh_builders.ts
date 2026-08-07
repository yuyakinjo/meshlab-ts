/**
 * Analytic mesh builders — the foundation of the whole test suite.
 *
 * Every builder returns the mesh together with the facts we know about it from
 * mathematics rather than from running our own code: vertex/edge/face counts,
 * Euler characteristic, genus, orientability, and closed-form area and volume
 * where one exists. Algorithms are then checked against those facts instead of
 * against golden output, which means a bug in an algorithm cannot quietly
 * become the expected result.
 *
 * The builders are themselves oracle-tested in `test/unit/helpers/mesh_builders.test.ts`
 * before anything else relies on them.
 */
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";

export interface MeshFacts {
	/** Live vertex count. */
	vn: number;
	/** Live face count. */
	fn: number;
	/** Distinct undirected edges. */
	en: number;
	/** Euler characteristic V - E + F. */
	chi: number;
	/** Connected components (by face adjacency). */
	components: number;
	/** Every edge shared by exactly two faces. */
	watertight: boolean;
	/** Edges shared by more than two faces. */
	nonManifoldEdges: number;
	/**
	 * Boundary loops; 0 for a closed surface. Omitted when the boundary does
	 * not decompose into loops, which happens once an edge is non-manifold.
	 */
	boundaryLoops?: number;
	/**
	 * An orientation of the whole surface exists. Omitted for meshes with
	 * non-manifold edges, where orientability is not the question being asked.
	 */
	orientable?: boolean;
	/** The faces as built are already coherently oriented. */
	coherentlyOriented?: boolean;
	/**
	 * Genus, for a closed orientable surface. `undefined` when the surface has
	 * a boundary or is non-orientable, where the notion needs qualification.
	 */
	genus?: number;
	/** Exact surface area, when a closed form exists for this discrete mesh. */
	area?: number;
	/** Exact enclosed volume, when a closed form exists for this discrete mesh. */
	volume?: number;
}

export interface BuiltMesh {
	mesh: CMeshO;
	expected: MeshFacts;
	/** Human-readable name, used in test failure messages. */
	name: string;
}

/** Assembles a mesh from xyz-interleaved coordinates and a flat index list. */
export function buildMesh(coords: readonly number[], faces: readonly number[]): CMeshO {
	const m = new CMeshO();
	Allocator.addMeshData(m, coords, faces);
	return m;
}

const PHI = (1 + Math.sqrt(5)) / 2;

// ---------------------------------------------------------------------------
// Closed, orientable, genus 0
// ---------------------------------------------------------------------------

/**
 * A regular tetrahedron with edge length 2√2, inscribed in the cube [-1,1]³.
 *
 * Volume a³/(6√2) = 8/3, area √3·a² = 8√3.
 */
export function tetrahedron(): BuiltMesh {
	const coords = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1];
	const faces = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
	return {
		name: "tetrahedron",
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 4,
			fn: 4,
			en: 6,
			chi: 2,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
			area: 8 * Math.sqrt(3),
			volume: 8 / 3,
		},
	};
}

/** An axis-aligned cube of side `s` centred on the origin, split into 12 triangles. */
export function cube(s = 1): BuiltMesh {
	const h = s / 2;
	const coords = [
		-h,
		-h,
		-h, // 0
		h,
		-h,
		-h, // 1
		h,
		h,
		-h, // 2
		-h,
		h,
		-h, // 3
		-h,
		-h,
		h, // 4
		h,
		-h,
		h, // 5
		h,
		h,
		h, // 6
		-h,
		h,
		h, // 7
	];
	// Each face wound counter-clockwise seen from outside.
	const faces = [
		4,
		5,
		6,
		4,
		6,
		7, // +z
		1,
		0,
		3,
		1,
		3,
		2, // -z
		5,
		1,
		2,
		5,
		2,
		6, // +x
		0,
		4,
		7,
		0,
		7,
		3, // -x
		7,
		6,
		2,
		7,
		2,
		3, // +y
		0,
		1,
		5,
		0,
		5,
		4, // -y
	];
	return {
		name: `cube(${s})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 8,
			fn: 12,
			en: 18,
			chi: 2,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
			area: 6 * s * s,
			volume: s * s * s,
		},
	};
}

/** A regular octahedron with unit circumradius (edge length √2). */
export function octahedron(): BuiltMesh {
	const coords = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1];
	const faces = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5];
	const a = Math.SQRT2;
	return {
		name: "octahedron",
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 6,
			fn: 8,
			en: 12,
			chi: 2,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
			area: 2 * Math.sqrt(3) * a * a,
			volume: (Math.SQRT2 / 3) * a * a * a,
		},
	};
}

/** A regular icosahedron with edge length 2. */
export function icosahedron(): BuiltMesh {
	const coords = [
		-1,
		PHI,
		0,
		1,
		PHI,
		0,
		-1,
		-PHI,
		0,
		1,
		-PHI,
		0,
		0,
		-1,
		PHI,
		0,
		1,
		PHI,
		0,
		-1,
		-PHI,
		0,
		1,
		-PHI,
		PHI,
		0,
		-1,
		PHI,
		0,
		1,
		-PHI,
		0,
		-1,
		-PHI,
		0,
		1,
	];
	const faces = [
		0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1,
		8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
	];
	const a = 2;
	return {
		name: "icosahedron",
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 12,
			fn: 20,
			en: 30,
			chi: 2,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
			area: 5 * Math.sqrt(3) * a * a,
			volume: (5 / 12) * (3 + Math.sqrt(5)) * a * a * a,
		},
	};
}

/**
 * An icosphere: `subdiv` rounds of 1→4 triangle subdivision applied to an
 * icosahedron, with every vertex projected onto the unit sphere.
 *
 * Area and volume converge to 4π and 4π/3 but do not equal them, so neither is
 * declared; use it for convergence tests instead.
 */
export function sphereIcosa(subdiv = 2): BuiltMesh {
	const base = icosahedron();
	let coords: number[] = [];
	for (let v = 0; v < base.mesh.vn; v++) {
		coords.push(base.mesh.vx(v), base.mesh.vy(v), base.mesh.vz(v));
	}
	let faces = Array.from(base.mesh.faceVert.subarray(0, base.mesh.fn * 3));

	for (let round = 0; round < subdiv; round++) {
		const midpoints = new Map<number, number>();
		const nextFaces: number[] = [];
		const midpoint = (a: number, b: number): number => {
			const key = a < b ? a * 1e7 + b : b * 1e7 + a;
			const hit = midpoints.get(key);
			if (hit !== undefined) return hit;
			const x = (coords[3 * a] + coords[3 * b]) / 2;
			const y = (coords[3 * a + 1] + coords[3 * b + 1]) / 2;
			const z = (coords[3 * a + 2] + coords[3 * b + 2]) / 2;
			const idx = coords.length / 3;
			coords.push(x, y, z);
			midpoints.set(key, idx);
			return idx;
		};
		for (let f = 0; f < faces.length; f += 3) {
			const [a, b, c] = [faces[f], faces[f + 1], faces[f + 2]];
			const ab = midpoint(a, b);
			const bc = midpoint(b, c);
			const ca = midpoint(c, a);
			nextFaces.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
		}
		faces = nextFaces;
	}

	// Project onto the unit sphere.
	const projected: number[] = [];
	for (let i = 0; i < coords.length; i += 3) {
		const len = Math.hypot(coords[i], coords[i + 1], coords[i + 2]);
		projected.push(coords[i] / len, coords[i + 1] / len, coords[i + 2] / len);
	}
	coords = projected;

	const vn = coords.length / 3;
	const fn = faces.length / 3;
	// A closed triangulated sphere: 2E = 3F.
	const en = (3 * fn) / 2;
	return {
		name: `sphereIcosa(${subdiv})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn,
			fn,
			en,
			chi: 2,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
		},
	};
}

// ---------------------------------------------------------------------------
// Closed, orientable, genus 1
// ---------------------------------------------------------------------------

/**
 * A torus of major radius `R` and minor radius `r`, as an `nu` × `nv` grid.
 *
 * χ = 0, genus 1. The discrete area and volume approach 4π²Rr and 2π²Rr² but
 * are not equal to them, so they are not declared.
 */
export function torus(R = 2, r = 0.6, nu = 24, nv = 12): BuiltMesh {
	const coords: number[] = [];
	for (let i = 0; i < nu; i++) {
		const u = (2 * Math.PI * i) / nu;
		for (let j = 0; j < nv; j++) {
			const v = (2 * Math.PI * j) / nv;
			const rad = R + r * Math.cos(v);
			coords.push(rad * Math.cos(u), rad * Math.sin(u), r * Math.sin(v));
		}
	}
	const idx = (i: number, j: number) => (((i % nu) + nu) % nu) * nv + (((j % nv) + nv) % nv);
	const faces: number[] = [];
	for (let i = 0; i < nu; i++) {
		for (let j = 0; j < nv; j++) {
			const a = idx(i, j);
			const b = idx(i + 1, j);
			const c = idx(i + 1, j + 1);
			const d = idx(i, j + 1);
			faces.push(a, b, c, a, c, d);
		}
	}
	const vn = nu * nv;
	const fn = 2 * nu * nv;
	return {
		name: `torus(${R},${r},${nu},${nv})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn,
			fn,
			en: 3 * nu * nv,
			chi: 0,
			components: 1,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 1,
		},
	};
}

// ---------------------------------------------------------------------------
// With boundary
// ---------------------------------------------------------------------------

/** A flat `nu` × `nv` quad grid in the z = 0 plane, split into triangles. */
export function gridPlane(nu = 4, nv = 3): BuiltMesh {
	const coords: number[] = [];
	for (let i = 0; i <= nu; i++) {
		for (let j = 0; j <= nv; j++) coords.push(i / nu, j / nv, 0);
	}
	const idx = (i: number, j: number) => i * (nv + 1) + j;
	const faces: number[] = [];
	for (let i = 0; i < nu; i++) {
		for (let j = 0; j < nv; j++) {
			faces.push(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1));
			faces.push(idx(i, j), idx(i + 1, j + 1), idx(i, j + 1));
		}
	}
	return {
		name: `gridPlane(${nu},${nv})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: (nu + 1) * (nv + 1),
			fn: 2 * nu * nv,
			en: 3 * nu * nv + nu + nv,
			chi: 1,
			components: 1,
			boundaryLoops: 1,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			area: 1,
		},
	};
}

/**
 * A Möbius strip with `n` segments — the canonical non-orientable surface.
 *
 * χ = 0 with one boundary loop. `Clean.OrientCoherentlyMesh` must report this
 * as not orientable rather than silently producing something.
 */
export function mobiusStrip(n = 16, R = 2, halfWidth = 0.5): BuiltMesh {
	const coords: number[] = [];
	for (let i = 0; i < n; i++) {
		const u = (2 * Math.PI * i) / n;
		for (const w of [-1, 1]) {
			const rad = R + w * halfWidth * Math.cos(u / 2);
			coords.push(rad * Math.cos(u), rad * Math.sin(u), w * halfWidth * Math.sin(u / 2));
		}
	}
	const faces: number[] = [];
	for (let i = 0; i < n; i++) {
		const a0 = 2 * i;
		const a1 = 2 * i + 1;
		const last = i === n - 1;
		// The half twist: the final quad joins back to column 0 with its two
		// vertices swapped, which is exactly what makes the strip one-sided.
		const b0 = last ? 1 : 2 * (i + 1);
		const b1 = last ? 0 : 2 * (i + 1) + 1;
		faces.push(a0, b0, b1, a0, b1, a1);
	}
	return {
		name: `mobiusStrip(${n})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 2 * n,
			fn: 2 * n,
			en: 4 * n,
			chi: 0,
			components: 1,
			boundaryLoops: 1,
			watertight: false,
			orientable: false,
			coherentlyOriented: false,
			nonManifoldEdges: 0,
		},
	};
}

// ---------------------------------------------------------------------------
// Deliberately broken meshes
// ---------------------------------------------------------------------------

/**
 * A cube as an unwelded triangle soup: 36 vertices, every triangle with its
 * own three. This is what a binary STL of a cube looks like, and the input
 * `Remove Duplicate Vertices` exists for.
 */
export function cubeSoup(s = 1): BuiltMesh {
	const src = cube(s);
	const coords: number[] = [];
	const faces: number[] = [];
	for (let f = 0; f < src.mesh.fn; f++) {
		for (let k = 0; k < 3; k++) {
			const v = src.mesh.fv(f, k);
			faces.push(coords.length / 3);
			coords.push(src.mesh.vx(v), src.mesh.vy(v), src.mesh.vz(v));
		}
	}
	return {
		name: `cubeSoup(${s})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 36,
			fn: 12,
			// Every edge is duplicated per incident face, so no two triangles
			// share an edge: 12 faces × 3 edges, all distinct.
			en: 36,
			chi: 36 - 36 + 12,
			components: 12,
			boundaryLoops: 12,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			area: 6 * s * s,
		},
	};
}

/** A cube with the winding of the faces at `faceIndices` reversed. */
export function cubeWithFlippedFaces(faceIndices: readonly number[] = [0, 5, 9]): BuiltMesh {
	const built = cube(1);
	const m = built.mesh;
	for (const f of faceIndices) {
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		m.setFace(f, a, c, b);
	}
	// Signed volume and genus stop being meaningful once the winding is
	// inconsistent, so they are dropped rather than restated. The mesh is
	// still watertight and still orientable: flipping a winding changes
	// neither which vertex pairs are edges nor whether a consistent
	// orientation exists.
	const { volume: _volume, genus: _genus, ...rest } = built.expected;
	const expected: MeshFacts = {
		...rest,
		coherentlyOriented: false,
		watertight: true,
		orientable: true,
	};
	return {
		name: `cubeWithFlippedFaces([${faceIndices.join(",")}])`,
		mesh: m,
		expected,
	};
}

/**
 * A cube with `k` faces removed, leaving `k` triangular holes.
 *
 * Capped at two: a cube has eight vertices, so at most two of its triangles
 * can be pairwise vertex-disjoint, and holes that touch at a vertex would not
 * be two loops. Use {@link sphereWithHoles} when more are needed.
 */
export function cubeWithHoles(k = 2): BuiltMesh {
	if (k < 1 || k > 2) throw new Error(`cubeWithHoles: k must be 1 or 2, got ${k}`);
	const built = cube(1);
	const m = built.mesh;
	// Faces 0 = {4,5,6} on +z and 3 = {1,3,2} on -z share no vertex.
	const victims = [0, 3].slice(0, k);
	for (const f of victims) Allocator.deleteFace(m, f);
	Allocator.compactFaceVector(m);
	return {
		name: `cubeWithHoles(${k})`,
		mesh: m,
		expected: {
			vn: 8,
			fn: 12 - k,
			en: 18,
			chi: 8 - 18 + (12 - k),
			components: 1,
			boundaryLoops: k,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
		},
	};
}

/**
 * An icosphere with `k` pairwise vertex-disjoint faces removed, leaving `k`
 * triangular holes.
 *
 * The disjointness is verified here rather than assumed, so the declared
 * `boundaryLoops: k` is a fact and not a hope.
 */
export function sphereWithHoles(k = 5, subdiv = 2): BuiltMesh {
	const built = sphereIcosa(subdiv);
	const m = built.mesh;
	const used = new Set<number>();
	const victims: number[] = [];
	// Stride through the faces so the holes end up spread over the sphere.
	const stride = Math.max(1, Math.floor(m.fn / (k + 1)));
	for (let i = 0; victims.length < k && i < m.fn; i++) {
		const f = (i * stride) % m.fn;
		if (m.isFaceD(f)) continue;
		const vs = [m.fv(f, 0), m.fv(f, 1), m.fv(f, 2)];
		if (vs.some((v) => used.has(v))) continue;
		for (const v of vs) used.add(v);
		victims.push(f);
	}
	if (victims.length < k) {
		throw new Error(`sphereWithHoles: could not find ${k} disjoint faces at subdiv ${subdiv}`);
	}
	for (const f of victims) Allocator.deleteFace(m, f);
	Allocator.compactFaceVector(m);

	const fn = built.expected.fn - k;
	return {
		name: `sphereWithHoles(${k},${subdiv})`,
		mesh: m,
		expected: {
			vn: built.expected.vn,
			fn,
			// Removing a face removes no edges: all three survive on the
			// neighbours that remain.
			en: built.expected.en,
			chi: built.expected.vn - built.expected.en + fn,
			components: 1,
			boundaryLoops: k,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
		},
	};
}

/** `k + 2` triangles all sharing one edge — a non-manifold edge fan. */
export function nonManifoldEdgeFan(k = 1): BuiltMesh {
	const coords = [0, 0, 0, 1, 0, 0];
	const faces: number[] = [];
	const blades = k + 2;
	for (let i = 0; i < blades; i++) {
		const angle = (Math.PI * i) / blades;
		coords.push(0.5, Math.cos(angle), Math.sin(angle));
		faces.push(0, 1, 2 + i);
	}
	return {
		name: `nonManifoldEdgeFan(${k})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 2 + blades,
			fn: blades,
			en: 1 + 2 * blades,
			chi: 2 + blades - (1 + 2 * blades) + blades,
			components: 1,
			watertight: false,
			nonManifoldEdges: 1,
			// boundaryLoops / orientable / coherentlyOriented are deliberately
			// not declared: at a non-manifold edge the boundary does not close
			// into loops and orientability is not well posed.
		},
	};
}

/** Two triangles meeting at a single vertex and nowhere else. */
export function bowtieVertex(): BuiltMesh {
	const coords = [
		0,
		0,
		0, // shared apex
		1,
		0,
		0,
		1,
		1,
		0,
		-1,
		0,
		0,
		-1,
		-1,
		0,
	];
	const faces = [0, 1, 2, 0, 3, 4];
	return {
		name: "bowtieVertex",
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 5,
			fn: 2,
			en: 6,
			chi: 1,
			// One component by face adjacency requires a shared edge; these
			// two triangles share only a vertex.
			components: 2,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			// boundaryLoops is not declared: the apex has four incident
			// boundary edges, so the boundary is a figure eight rather than
			// two loops. That pinch point is the defect this mesh exists to
			// exercise.
		},
	};
}

/** A cube plus `n` small disconnected tetrahedra, for component filtering. */
export function cubePlusIslands(n = 3, islandScale = 0.05): BuiltMesh {
	const coords: number[] = [];
	const faces: number[] = [];

	const push = (b: BuiltMesh, dx: number, dy: number, dz: number, scale: number) => {
		const base = coords.length / 3;
		for (let v = 0; v < b.mesh.vn; v++) {
			coords.push(b.mesh.vx(v) * scale + dx, b.mesh.vy(v) * scale + dy, b.mesh.vz(v) * scale + dz);
		}
		for (let f = 0; f < b.mesh.fn; f++) {
			faces.push(base + b.mesh.fv(f, 0), base + b.mesh.fv(f, 1), base + b.mesh.fv(f, 2));
		}
	};

	push(cube(1), 0, 0, 0, 1);
	for (let i = 0; i < n; i++) push(tetrahedron(), 3 + i * 0.5, 0, 0, islandScale);

	return {
		name: `cubePlusIslands(${n})`,
		mesh: buildMesh(coords, faces),
		expected: {
			vn: 8 + 4 * n,
			fn: 12 + 4 * n,
			en: 18 + 6 * n,
			chi: 2 * (1 + n),
			components: 1 + n,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			genus: 0,
		},
	};
}

/** A cube with `k` extra vertices that no face references. */
export function unreferencedVerts(k = 5): BuiltMesh {
	const built = cube(1);
	const m = built.mesh;
	for (let i = 0; i < k; i++) Allocator.addVertex(m, 10 + i, 0, 0);
	// Stray vertices push the Euler characteristic away from 2, so the genus
	// formula stops describing the surface. That is exactly the defect
	// `Remove Unreferenced Vertices` repairs, so genus is dropped here.
	const { genus: _genus, ...rest } = built.expected;
	const expected: MeshFacts = { ...rest, vn: 8 + k, chi: 8 + k - 18 + 12 };
	return { name: `unreferencedVerts(${k})`, mesh: m, expected };
}

/** An empty mesh — the degenerate input every filter must survive. */
export function emptyMesh(): BuiltMesh {
	return {
		name: "emptyMesh",
		mesh: new CMeshO(),
		expected: {
			vn: 0,
			fn: 0,
			en: 0,
			chi: 0,
			components: 0,
			boundaryLoops: 0,
			watertight: true,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
		},
	};
}

/** A single triangle: the smallest non-empty mesh. */
export function singleTriangle(): BuiltMesh {
	return {
		name: "singleTriangle",
		mesh: buildMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]),
		expected: {
			vn: 3,
			fn: 1,
			en: 3,
			chi: 1,
			components: 1,
			boundaryLoops: 1,
			watertight: false,
			orientable: true,
			coherentlyOriented: true,
			nonManifoldEdges: 0,
			area: 0.5,
		},
	};
}

/** Closed, orientable meshes for which area and volume are known exactly. */
export const CLOSED_POLYHEDRA: ReadonlyArray<() => BuiltMesh> = [
	tetrahedron,
	() => cube(1),
	() => cube(2.5),
	octahedron,
	icosahedron,
];

/** Every well-formed builder, for sweeps that should hold across the board. */
export const WELL_FORMED_BUILDERS: ReadonlyArray<() => BuiltMesh> = [
	...CLOSED_POLYHEDRA,
	() => sphereIcosa(1),
	() => sphereIcosa(2),
	() => torus(),
	() => gridPlane(),
	() => mobiusStrip(),
	singleTriangle,
	emptyMesh,
];

/** Meshes with a defect some filter is supposed to fix. */
export const BROKEN_BUILDERS: ReadonlyArray<() => BuiltMesh> = [
	() => cubeSoup(),
	() => cubeWithFlippedFaces(),
	() => cubeWithHoles(1),
	() => cubeWithHoles(2),
	() => sphereWithHoles(5),
	() => nonManifoldEdgeFan(1),
	() => nonManifoldEdgeFan(3),
	bowtieVertex,
	() => cubePlusIslands(3),
	() => unreferencedVerts(5),
];

export const ALL_BUILDERS: ReadonlyArray<() => BuiltMesh> = [
	...WELL_FORMED_BUILDERS,
	...BROKEN_BUILDERS,
];
