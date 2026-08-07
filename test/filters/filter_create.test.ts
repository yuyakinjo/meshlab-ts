/**
 * `filter_create`, and the point-cloud side of PLY.
 *
 * Every primitive here has a volume, a genus and a vertex count that follow
 * from the solid it names, so the tests check those rather than a recorded
 * output. The counts are the C++ ones on purpose: a sphere built here should
 * be indexed the same as a sphere built by MeshLab, or nothing downstream can
 * be compared between the two.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { FilterArity } from "../../src/common/plugins/filter_arity.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { VertexFlag } from "../../src/vcg/complex/flags.ts";
import { GenNormal } from "../../src/vcg/math/gen_normal.ts";
import { assertAllocatorConsistent, computeFacts, signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();

/** Applies a create filter and hands back the layer it made. */
function created(name: string, params: Record<string, unknown> = {}): CMeshO {
	const doc = new MeshDocument();
	const out = kernel.applyFilter(doc, name, params);
	return doc.requireMesh(out.new_mesh_id as number).cm;
}

describe("the platonic solids", () => {
	// Volume of each solid at the size VCGLib builds it, from the closed form.
	const CASES: Array<[string, number, number, number]> = [
		// name, vn, fn, volume
		["Tetrahedron", 4, 4, 8 / 3],
		["Octahedron", 6, 8, 4 / 3],
		// Edge 2, so V = (5/12)(3+sqrt5) a^3 with a = 2.
		["Icosahedron", 12, 20, (5 / 12) * (3 + Math.sqrt(5)) * 8],
		// Circumradius 1, so V follows from the edge length that implies.
		["Dodecahedron", 20, 36, dodecahedronVolume()],
		["Dodecahedron (symmetric)", 32, 60, dodecahedronVolume()],
	];

	for (const [name, vn, fn, volume] of CASES) {
		test(`${name} is closed, genus 0, and the right size`, () => {
			const m = created(name);
			expect(m.vn).toBe(vn);
			expect(m.fn).toBe(fn);
			const facts = computeFacts(m);
			expect(facts.watertight).toBe(true);
			expect(facts.components).toBe(1);
			expect(facts.genus).toBe(0);
			expect(facts.coherentlyOriented).toBe(true);
			expect(signedVolume(m)).toBeCloseTo(volume, 9);
			assertAllocatorConsistent(m);
		});
	}

	test("both dodecahedra describe the same solid", () => {
		// The symmetric one only differs in how the pentagons are cut up, so it
		// must enclose exactly the same volume.
		expect(signedVolume(created("Dodecahedron (symmetric)"))).toBeCloseTo(
			signedVolume(created("Dodecahedron")),
			9,
		);
	});
});

describe("Box/Cube", () => {
	test("has the requested side, centred on the origin", () => {
		for (const size of [1, 2, 0.5]) {
			const m = created("Box/Cube", { size });
			expect(m.vn).toBe(8);
			expect(m.fn).toBe(12);
			expect(signedVolume(m)).toBeCloseTo(size ** 3, 9);
			for (let v = 0; v < m.vn; v++) {
				expect(Math.abs(m.vx(v))).toBeCloseTo(size / 2, 12);
				expect(Math.abs(m.vy(v))).toBeCloseTo(size / 2, 12);
				expect(Math.abs(m.vz(v))).toBeCloseTo(size / 2, 12);
			}
			expect(computeFacts(m).genus).toBe(0);
		}
	});
});

describe("Sphere", () => {
	test("subdivides an icosahedron, four faces for one", () => {
		// The counts MeshLab produces, which is what makes a subdiv-3 sphere a
		// usable fixture on both sides.
		const EXPECTED: Array<[number, number, number]> = [
			[0, 12, 20],
			[1, 42, 80],
			[2, 162, 320],
			[3, 642, 1280],
			[4, 2562, 5120],
		];
		for (const [subdiv, vn, fn] of EXPECTED) {
			const m = created("Sphere", { subdiv });
			expect(m.vn, `subdiv ${subdiv}`).toBe(vn);
			expect(m.fn, `subdiv ${subdiv}`).toBe(fn);
		}
	});

	test("every vertex sits at the requested radius", () => {
		for (const radius of [1, 5, 0.25]) {
			const m = created("Sphere", { radius, subdiv: 2 });
			for (let v = 0; v < m.vn; v++) {
				expect(Math.hypot(m.vx(v), m.vy(v), m.vz(v))).toBeCloseTo(radius, 9);
			}
		}
	});

	test("converges on the volume of a ball as it refines", () => {
		let previous = 0;
		for (const subdiv of [1, 2, 3, 4]) {
			const v = signedVolume(created("Sphere", { subdiv }));
			// Every refinement pushes new vertices out onto the sphere, so the
			// inscribed polyhedron only ever grows — toward 4/3 pi, never past.
			expect(v, `subdiv ${subdiv}`).toBeGreaterThan(previous);
			expect(v, `subdiv ${subdiv}`).toBeLessThan((4 / 3) * Math.PI);
			previous = v;
		}
		expect(previous).toBeCloseTo((4 / 3) * Math.PI, 1);
	});

	test("is closed and manifold at every level", () => {
		for (const subdiv of [0, 1, 3]) {
			const facts = computeFacts(created("Sphere", { subdiv }));
			expect(facts.watertight, `subdiv ${subdiv}`).toBe(true);
			expect(facts.genus, `subdiv ${subdiv}`).toBe(0);
			expect(facts.nonManifoldEdges, `subdiv ${subdiv}`).toBe(0);
		}
	});
});

describe("Cone", () => {
	test("a frustum holds the volume the formula says", () => {
		// pi h / 3 * (r0^2 + r0 r1 + r1^2), approached from below by the
		// polygonal approximation.
		const m = created("Cone", { r0: 1, r1: 2, h: 3, subdiv: 256 });
		const exact = ((Math.PI * 3) / 3) * (1 + 2 + 4);
		expect(signedVolume(m)).toBeGreaterThan(exact * 0.999);
		expect(signedVolume(m)).toBeLessThan(exact);
		expect(computeFacts(m).watertight).toBe(true);
	});

	test("a zero radius closes that end to a point", () => {
		const tip = created("Cone", { r0: 0, r1: 2, h: 3, subdiv: 256 });
		// One apex instead of a ring, so two fewer rings of vertices.
		expect(tip.vn).toBe(256 + 2);
		expect(signedVolume(tip)).toBeCloseTo((Math.PI * 4 * 3) / 3, 2);
		expect(computeFacts(tip).watertight).toBe(true);

		const other = created("Cone", { r0: 1, r1: 0, h: 3, subdiv: 256 });
		expect(signedVolume(other)).toBeCloseTo((Math.PI * 1 * 3) / 3, 2);
		expect(computeFacts(other).watertight).toBe(true);
	});

	test("equal radii give a cylinder", () => {
		const m = created("Cone", { r0: 1, r1: 1, h: 3, subdiv: 256 });
		expect(signedVolume(m)).toBeCloseTo(Math.PI * 3, 2);
		expect(computeFacts(m).genus).toBe(0);
	});

	test("is coherently oriented, whichever end is closed", () => {
		for (const params of [
			{ r0: 1, r1: 2 },
			{ r0: 0, r1: 2 },
			{ r0: 1, r1: 0 },
		]) {
			const facts = computeFacts(created("Cone", { ...params, h: 3, subdiv: 24 }));
			expect(facts.coherentlyOriented, JSON.stringify(params)).toBe(true);
			expect(facts.watertight, JSON.stringify(params)).toBe(true);
		}
	});
});

describe("Torus", () => {
	test("is genus 1, and holds 2 pi^2 R r^2", () => {
		const m = created("Torus", { hRadius: 3, vRadius: 1, hSubdiv: 256, vSubdiv: 128 });
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(1);
		expect(signedVolume(m)).toBeCloseTo(2 * Math.PI ** 2 * 3 * 1, 1);
	});

	test("the seam is welded shut, not left doubled", () => {
		// The grid is built with a duplicate row and column so the wrap is easy
		// to write; if the weld were skipped the count would be (h+1)(v+1).
		const m = created("Torus", { hSubdiv: 24, vSubdiv: 12 });
		expect(m.vn).toBe(24 * 12);
		expect(m.fn).toBe(24 * 12 * 2);
		expect(computeFacts(m).nonManifoldEdges).toBe(0);
	});
});

describe("Annulus and Sphere Cap", () => {
	test("an annulus is a flat open ring", () => {
		const m = created("Annulus", { internalRadius: 0.5, externalRadius: 1, sides: 32 });
		expect(m.vn).toBe(64);
		expect(m.fn).toBe(64);
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(false);
		// Two rims, inner and outer.
		expect(facts.boundaryLoops).toBe(2);
		for (let v = 0; v < m.vn; v++) {
			expect(m.vz(v)).toBe(0);
			const r = Math.hypot(m.vx(v), m.vy(v));
			expect(r > 0.49 && r < 1.01, `radius ${r}`).toBe(true);
		}
	});

	test("a sphere cap is a disc bent onto a sphere", () => {
		const m = created("Sphere Cap", { angle: 60, subdiv: 3 });
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(false);
		expect(facts.boundaryLoops).toBe(1);
		expect(facts.components).toBe(1);
		// Every point of the cap lies on the unit sphere centred one
		// cos(halfAngle) below the origin.
		const centre = -Math.cos((30 * Math.PI) / 180);
		for (let v = 0; v < m.vn; v++) {
			expect(Math.hypot(m.vx(v), m.vy(v), m.vz(v) - centre)).toBeCloseTo(1, 6);
		}
	});

	test("a wider cap covers more of the sphere", () => {
		let previous = 0;
		for (const angle of [30, 60, 120, 170]) {
			const m = created("Sphere Cap", { angle, subdiv: 2 });
			let rim = 0;
			for (let v = 0; v < m.vn; v++) rim = Math.max(rim, Math.hypot(m.vx(v), m.vy(v)));
			expect(rim, `angle ${angle}`).toBeGreaterThan(previous);
			expect(rim, `angle ${angle}`).toBeCloseTo(Math.sin((angle / 2) * (Math.PI / 180)), 6);
			previous = rim;
		}
	});

	test("rejects a cap angle that does not subtend one", () => {
		for (const angle of [0, -10, 180, 200]) {
			expect(() => created("Sphere Cap", { angle })).toThrow(MLException);
		}
	});
});

describe("Points on a Sphere", () => {
	const TECHNIQUES = [
		[0, "Montecarlo"],
		[1, "Poisson Sampling"],
		[2, "DiscoBall"],
		[3, "Octahedron"],
		[4, "Fibonacci"],
	] as const;

	for (const [tech, label] of TECHNIQUES) {
		test(`${label} puts every point on the unit sphere, with its normal`, () => {
			const m = created("Points on a Sphere", { pointNum: 300, sphereGenTech: tech });
			expect(m.fn).toBe(0);
			expect(m.vn).toBeGreaterThan(0);
			for (let v = 0; v < m.vn; v++) {
				expect(Math.hypot(m.vx(v), m.vy(v), m.vz(v))).toBeCloseTo(1, 9);
				// The position doubles as the normal, which is what makes these
				// clouds reconstructable straight away.
				expect(m.vertNormal[3 * v]).toBe(m.vx(v));
				expect(m.vertNormal[3 * v + 1]).toBe(m.vy(v));
				expect(m.vertNormal[3 * v + 2]).toBe(m.vz(v));
			}
		});
	}

	test("Fibonacci and Montecarlo give the exact count; the lattices approximate it", () => {
		expect(created("Points on a Sphere", { pointNum: 137, sphereGenTech: 4 }).vn).toBe(137);
		expect(created("Points on a Sphere", { pointNum: 137, sphereGenTech: 0 }).vn).toBe(137);
		// A subdivided octahedron only comes in certain sizes, 4^lev * 4 + 2, and
		// upstream picks the level by comparing 4^lev + 2 against the request —
		// so it overshoots by about a factor of four. Faithfully reproduced.
		expect(created("Points on a Sphere", { pointNum: 137, sphereGenTech: 3 }).vn).toBe(258);
	});

	test("more points asked for is never fewer points given", () => {
		for (const tech of [2, 3, 4]) {
			let previous = 0;
			for (const pointNum of [50, 200, 1000, 4000]) {
				const n = created("Points on a Sphere", { pointNum, sphereGenTech: tech }).vn;
				expect(n, `tech ${tech} at ${pointNum}`).toBeGreaterThanOrEqual(previous);
				previous = n;
			}
		}
	});

	test("the lattices are deterministic", () => {
		for (const tech of [2, 3, 4]) {
			const a = created("Points on a Sphere", { pointNum: 500, sphereGenTech: tech });
			const b = created("Points on a Sphere", { pointNum: 500, sphereGenTech: tech });
			expect(Array.from(a.vertCoord.subarray(0, 3 * a.vn))).toEqual(
				Array.from(b.vertCoord.subarray(0, 3 * b.vn)),
			);
		}
	});

	test("the disco ball spaces its points more uniformly than Fibonacci", () => {
		// Which is the reason to keep both. The disco ball lays points along
		// rings at a fixed spacing, so nearest-neighbour distance barely varies;
		// the Fibonacci spiral trades that for having no visible rings at all.
		expect(spreadOf(GenNormal.discoBall(500))).toBeLessThan(spreadOf(GenNormal.fibonacci(500)));
		// Both are still far more even than chance would give.
		expect(spreadOf(GenNormal.fibonacci(500))).toBeLessThan(0.1);
	});

	test("zero points is empty, not an error", () => {
		expect(created("Points on a Sphere", { pointNum: 0 }).vn).toBe(0);
	});
});

describe("Fit a plane to selection", () => {
	/** A document holding a plane of points, tilted, with all of them selected. */
	function tiltedPlane(): MeshDocument {
		const doc = new MeshDocument();
		const cloud = Platonic.pointCloudFrom([]);
		const pts: Array<[number, number, number]> = [];
		for (let i = 0; i <= 8; i++) {
			for (let j = 0; j <= 8; j++) {
				const x = i / 4 - 1;
				const y = j / 4 - 1;
				// z = x/2 + y/3, so the true normal is (-1/2, -1/3, 1) normalized.
				pts.push([x, y, x / 2 + y / 3]);
			}
		}
		const m = doc.addNewMesh("", "cloud", true, Platonic.pointCloudFrom(pts));
		for (let v = 0; v < m.cm.vertSize; v++) m.cm.vertFlags[v] |= VertexFlag.SELECTED;
		void cloud;
		return doc;
	}

	test("finds the plane the points lie on", () => {
		const doc = tiltedPlane();
		const out = kernel.applyFilter(doc, "Fit a plane to selection", {});
		const quad = doc.requireMesh(out.new_mesh_id as number).cm;
		expect(quad.fn).toBeGreaterThan(0);
		// Every vertex of the quad must satisfy z = x/2 + y/3 as well.
		for (let v = 0; v < quad.vn; v++) {
			expect(quad.vz(v)).toBeCloseTo(quad.vx(v) / 2 + quad.vy(v) / 3, 9);
		}
	});

	test("subdiv sets how finely the quad is divided", () => {
		for (const subdiv of [1, 2, 4]) {
			const doc = tiltedPlane();
			const out = kernel.applyFilter(doc, "Fit a plane to selection", { subdiv });
			const quad = doc.requireMesh(out.new_mesh_id as number).cm;
			expect(quad.vn, `subdiv ${subdiv}`).toBe((subdiv + 1) ** 2);
			expect(quad.fn, `subdiv ${subdiv}`).toBe(subdiv * subdiv * 2);
		}
	});

	test("extent scales the quad about the selection", () => {
		const sizes = [1, 2].map((extent) => {
			const doc = tiltedPlane();
			const out = kernel.applyFilter(doc, "Fit a plane to selection", { extent });
			const quad = doc.requireMesh(out.new_mesh_id as number).cm;
			let low = Number.POSITIVE_INFINITY;
			let high = Number.NEGATIVE_INFINITY;
			for (let v = 0; v < quad.vn; v++) {
				low = Math.min(low, quad.vx(v));
				high = Math.max(high, quad.vx(v));
			}
			return high - low;
		});
		expect(sizes[1]).toBeCloseTo(sizes[0] * 2, 6);
	});

	test("says so when there is no selection to fit to", () => {
		const doc = new MeshDocument();
		doc.addNewMesh(
			"",
			"cloud",
			true,
			Platonic.pointCloudFrom([
				[0, 0, 0],
				[1, 0, 0],
				[0, 1, 0],
			]),
		);
		expect(() => kernel.applyFilter(doc, "Fit a plane to selection", {})).toThrow(MLException);
	});
});

describe("the filters", () => {
	test("are all registered as MeshLab registers them", () => {
		const NAMES: Array<[string, string]> = [
			["Box/Cube", "create_cube"],
			["Annulus", "create_annulus"],
			["Sphere", "create_sphere"],
			["Sphere Cap", "create_sphere_cap"],
			["Points on a Sphere", "create_sphere_points"],
			["Icosahedron", "create_icosahedron"],
			["Dodecahedron", "create_dodecahedron"],
			["Dodecahedron (symmetric)", "create_dodecahedron_sym"],
			["Octahedron", "create_octahedron"],
			["Tetrahedron", "create_tetrahedron"],
			["Cone", "create_cone"],
			["Torus", "create_torus"],
			["Fit a plane to selection", "generate_plane_fitting_to_selection"],
		];
		for (const [name, pythonName] of NAMES) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(filterClassToString(action.filterClass), name).toBe("MeshCreation");
			// They read no mesh, which is what lets the CLI run them with no input.
			expect(action.arity, name).toBe(FilterArity.NONE);
		}
	});

	test("carry MeshLab's parameter defaults", () => {
		const cone = kernel.initParameterList("Cone");
		expect(cone.getParameterByName("r0").defaultValue.value).toBe(1);
		expect(cone.getParameterByName("r1").defaultValue.value).toBe(2);
		expect(cone.getParameterByName("h").defaultValue.value).toBe(3);
		expect(cone.getParameterByName("subdiv").defaultValue.value).toBe(36);

		const torus = kernel.initParameterList("Torus");
		expect(torus.getParameterByName("hRadius").defaultValue.value).toBe(3);
		expect(torus.getParameterByName("vRadius").defaultValue.value).toBe(1);
		expect(torus.getParameterByName("hSubdiv").defaultValue.value).toBe(24);
		expect(torus.getParameterByName("vSubdiv").defaultValue.value).toBe(12);

		expect(kernel.initParameterList("Sphere").getParameterByName("subdiv").defaultValue.value).toBe(
			3,
		);
		// The default technique is the recursive octahedron, not Montecarlo.
		expect(
			kernel.initParameterList("Points on a Sphere").getParameterByName("sphereGenTech")
				.defaultValue.value,
		).toBe(3);
	});

	test("make the new layer current, so a pipeline can carry on from it", () => {
		const doc = new MeshDocument();
		const out = kernel.applyFilter(doc, "Torus", {});
		expect(doc.meshNumber()).toBe(1);
		expect(doc.mm().id()).toBe(out.new_mesh_id as number);
		expect(doc.mm().label()).toBe("Torus");
	});
});

describe("point clouds through PLY", () => {
	test("normals survive a round trip when they are asked for", () => {
		const doc = new MeshDocument();
		const cloud = created("Points on a Sphere", { pointNum: 200, sphereGenTech: 4 });
		doc.addNewMesh("", "cloud", true, cloud);

		const mask = MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL;
		const bytes = kernel.serializeMesh(doc, "cloud.ply", undefined, {}, undefined, mask);

		const back = new MeshDocument();
		const read = kernel.openMeshData(back, "cloud.ply", bytes);
		expect(read.cm.vn).toBe(cloud.vn);
		expect(read.cm.fn).toBe(0);
		for (let v = 0; v < cloud.vn; v++) {
			for (let a = 0; a < 3; a++) {
				expect(read.cm.vertCoord[3 * v + a]).toBe(cloud.vertCoord[3 * v + a]);
				expect(read.cm.vertNormal[3 * v + a]).toBe(cloud.vertNormal[3 * v + a]);
			}
		}
		expect(read.hasDataMask(MeshElement.MM_VERTNORMAL)).toBe(true);
	});

	test("without the mask the normals are left out, not silently guessed", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "cloud", true, created("Points on a Sphere", { pointNum: 50 }));
		const bytes = kernel.serializeMesh(doc, "cloud.ply");
		expect(new TextDecoder().decode(bytes.subarray(0, 200))).not.toContain("property double nx");
	});

	test("quality rides along too", () => {
		const doc = new MeshDocument();
		const cloud = created("Points on a Sphere", { pointNum: 60, sphereGenTech: 4 });
		for (let v = 0; v < cloud.vn; v++) cloud.vertQuality[v] = v * 0.5;
		doc.addNewMesh("", "cloud", true, cloud);

		const bytes = kernel.serializeMesh(
			doc,
			"cloud.ply",
			undefined,
			{},
			undefined,
			MeshElement.MM_VERTCOORD | MeshElement.MM_VERTQUALITY,
		);
		const back = new MeshDocument();
		const read = kernel.openMeshData(back, "cloud.ply", bytes);
		for (let v = 0; v < cloud.vn; v++) expect(read.cm.vertQuality[v]).toBe(v * 0.5);
	});

	test("ascii and binary round trip a cloud identically", () => {
		const doc = new MeshDocument();
		const cloud = created("Points on a Sphere", { pointNum: 80, sphereGenTech: 2 });
		doc.addNewMesh("", "cloud", true, cloud);
		const mask = MeshElement.MM_VERTCOORD | MeshElement.MM_VERTNORMAL;

		const read = (binary: boolean) => {
			const bytes = kernel.serializeMesh(
				doc,
				"c.ply",
				undefined,
				{ Binary: binary },
				undefined,
				mask,
			);
			const back = new MeshDocument();
			return kernel.openMeshData(back, "c.ply", bytes).cm;
		};
		const a = read(false);
		const b = read(true);
		expect(a.vn).toBe(b.vn);
		for (let v = 0; v < a.vn; v++) {
			for (let k = 0; k < 3; k++) {
				expect(a.vertCoord[3 * v + k]).toBeCloseTo(b.vertCoord[3 * v + k], 12);
			}
		}
	});

	test("STL refuses a channel it cannot store", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "s", true, created("Sphere", { subdiv: 1 }));
		expect(() =>
			kernel.serializeMesh(
				doc,
				"s.stl",
				undefined,
				{},
				undefined,
				MeshElement.MM_VERTCOORD | MeshElement.MM_FACEVERT | MeshElement.MM_VERTNORMAL,
			),
		).toThrow();
	});
});

/** Volume of a regular dodecahedron of circumradius 1. */
function dodecahedronVolume(): number {
	// Edge of a dodecahedron whose circumradius is 1: a = 2 / (sqrt3 * phi).
	const a = 4 / (Math.sqrt(3) * (1 + Math.sqrt(5)));
	return ((15 + 7 * Math.sqrt(5)) / 4) * a ** 3;
}

/** Coefficient of variation of the nearest-neighbour distance: lower is more even. */
function spreadOf(points: ReadonlyArray<readonly [number, number, number]>): number {
	const nearest: number[] = [];
	for (let i = 0; i < points.length; i++) {
		let best = Number.POSITIVE_INFINITY;
		for (let j = 0; j < points.length; j++) {
			if (i === j) continue;
			const d =
				(points[i][0] - points[j][0]) ** 2 +
				(points[i][1] - points[j][1]) ** 2 +
				(points[i][2] - points[j][2]) ** 2;
			if (d < best) best = d;
		}
		nearest.push(Math.sqrt(best));
	}
	const mean = nearest.reduce((a, b) => a + b, 0) / nearest.length;
	const variance = nearest.reduce((a, b) => a + (b - mean) ** 2, 0) / nearest.length;
	return Math.sqrt(variance) / mean;
}
