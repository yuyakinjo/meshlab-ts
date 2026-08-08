/**
 * Procedural noise, the noisy isosurface, and the Voronoi filters.
 *
 * Noise is checked by its defining properties rather than by recorded values:
 * it is deterministic in the seed, band-limited (neighbouring samples are
 * close), and zero at the integer lattice, which is what "gradient noise"
 * means. The Voronoi filters are checked by what their partition guarantees —
 * every vertex belongs to its nearest seed, and relaxation makes the regions
 * more even, not less.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { dijkstraGeodesic } from "../../src/vcg/complex/geodesic.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { FractalAlgorithm, FractalField, PerlinNoise } from "../../src/vcg/math/noise.ts";
import { assertAllocatorConsistent } from "../helpers/invariants.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, label = "test") {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", label, true, cm);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

describe("Perlin noise", () => {
	test("is deterministic in its seed", () => {
		const a = new PerlinNoise(7);
		const b = new PerlinNoise(7);
		const c = new PerlinNoise(8);
		expect(a.at(0.3, 0.7, 1.1)).toBe(b.at(0.3, 0.7, 1.1));
		expect(a.at(0.3, 0.7, 1.1)).not.toBe(c.at(0.3, 0.7, 1.1));
	});

	test("vanishes on the integer lattice, as gradient noise does", () => {
		const noise = new PerlinNoise(3);
		for (const p of [
			[0, 0, 0],
			[1, 2, 3],
			[-4, 5, -6],
		]) {
			expect(Math.abs(noise.at(p[0], p[1], p[2]))).toBeLessThan(1e-12);
		}
	});

	test("stays inside its range and is not constant", () => {
		const noise = new PerlinNoise(11);
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < 2000; i++) {
			const v = noise.at(i * 0.137, i * 0.239, i * 0.313);
			min = Math.min(min, v);
			max = Math.max(max, v);
		}
		expect(min).toBeGreaterThan(-1.001);
		expect(max).toBeLessThan(1.001);
		expect(max - min).toBeGreaterThan(0.5);
	});

	test("is band-limited: nearby samples are close", () => {
		const noise = new PerlinNoise(5);
		let worst = 0;
		for (let i = 0; i < 500; i++) {
			const x = i * 0.031;
			worst = Math.max(worst, Math.abs(noise.at(x, 0.5, 0.5) - noise.at(x + 0.001, 0.5, 0.5)));
		}
		// A tiny step in space cannot give a big step in value; that is what
		// separates gradient noise from a hash.
		expect(worst).toBeLessThan(0.05);
	});
});

describe("fractal fields", () => {
	test("all five algorithms produce a varying field", () => {
		for (const algorithm of Object.values(FractalAlgorithm)) {
			const field = new FractalField({
				algorithm,
				octaves: 6,
				lacunarity: 2,
				fractalIncrement: 0.8,
				offset: 0.8,
				gain: 2,
				seed: 4,
			});
			let min = Number.POSITIVE_INFINITY;
			let max = Number.NEGATIVE_INFINITY;
			for (let i = 0; i < 400; i++) {
				const v = field.at(i * 0.07, i * 0.11, 0.5);
				expect(Number.isFinite(v), `algorithm ${algorithm}`).toBe(true);
				min = Math.min(min, v);
				max = Math.max(max, v);
			}
			expect(max - min, `algorithm ${algorithm} varies`).toBeGreaterThan(0);
		}
	});

	test("more octaves add detail without changing the broad shape", () => {
		const build = (octaves: number) =>
			new FractalField({
				algorithm: FractalAlgorithm.FBM,
				octaves,
				lacunarity: 2,
				fractalIncrement: 1,
				offset: 0,
				gain: 1,
				seed: 9,
			});
		const coarse = build(2);
		const fine = build(8);
		// The extra octaves are small, so the two agree to within their own
		// amplitude — a large gap would mean the exponent table is wrong.
		let worst = 0;
		for (let i = 0; i < 200; i++) {
			const p = i * 0.05;
			worst = Math.max(worst, Math.abs(coarse.at(p, p, p) - fine.at(p, p, p)));
		}
		expect(worst).toBeLessThan(0.5);
	});

	test("a zero or negative octave count is refused", () => {
		expect(
			() =>
				new FractalField({
					algorithm: 0,
					octaves: 0,
					lacunarity: 2,
					fractalIncrement: 1,
					offset: 0,
					gain: 1,
					seed: 1,
				}),
		).toThrow(/at least one octave/);
	});
});

describe("Fractal Terrain", () => {
	const NAME = "Fractal Terrain";

	test("makes a grid whose relief is bounded by the requested height", () => {
		const doc = new MeshDocument();
		const out = kernel.applyFilter(doc, NAME, {
			steps: 5,
			maxHeight: 0.2,
			seed: 2,
			algorithm: 4,
			octaves: 6,
			lacunarity: 4,
			fractalIncrement: 0.5,
			offset: 0.9,
			gain: 2.5,
			saveAsQuality: false,
		});
		expect(out.face_number).toBe(2 * 32 * 32);

		const cm = doc.mm().cm;
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			min = Math.min(min, cm.vz(v));
			max = Math.max(max, cm.vz(v));
		}
		// The relief is normalised to span exactly the requested height.
		expect(max - min).toBeCloseTo(0.2, 6);
		assertAllocatorConsistent(cm);
	});

	test("the same seed gives the same terrain", () => {
		const build = (seed: number) => {
			const doc = new MeshDocument();
			kernel.applyFilter(doc, NAME, {
				steps: 4,
				maxHeight: 0.3,
				seed,
				algorithm: 0,
				octaves: 5,
				lacunarity: 2,
				fractalIncrement: 1,
				offset: 0.5,
				gain: 2,
				saveAsQuality: false,
			});
			return Float64Array.from(doc.mm().cm.vertCoord);
		};
		expect([...build(3)]).toEqual([...build(3)]);
		expect([...build(3)]).not.toEqual([...build(4)]);
	});

	test("saveAsQuality leaves the geometry flat", () => {
		const doc = new MeshDocument();
		kernel.applyFilter(doc, NAME, {
			steps: 4,
			maxHeight: 0.5,
			seed: 1,
			algorithm: 0,
			octaves: 4,
			lacunarity: 2,
			fractalIncrement: 1,
			offset: 0.5,
			gain: 2,
			saveAsQuality: true,
		});
		const cm = doc.mm().cm;
		for (let v = 0; v < cm.vertSize; v++) expect(cm.vz(v)).toBe(0);
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) max = Math.max(max, cm.vertQuality[v]);
		expect(max).toBeCloseTo(0.5, 6);
	});

	test("an absurd subdivision count is refused before allocating it", () => {
		const doc = new MeshDocument();
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				steps: 20,
				maxHeight: 0.2,
				seed: 1,
				algorithm: 0,
				octaves: 4,
				lacunarity: 2,
				fractalIncrement: 1,
				offset: 0.5,
				gain: 2,
				saveAsQuality: false,
			}),
		).toThrow(/within 1\.\.12/);
	});
});

describe("Fractal Displacement", () => {
	test("displaces a sphere along its normals, keeping it a sphere-ish shell", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const before = cm.fn;
		kernel.applyFilter(doc, "Fractal Displacement", {
			maxHeight: 0.1,
			scale: 1,
			seed: 2,
			algorithm: 0,
			octaves: 5,
			lacunarity: 2,
			fractalIncrement: 1,
			offset: 0.5,
			gain: 2,
			saveAsQuality: false,
		});
		// A displacement along normals changes no connectivity at all.
		expect(cm.fn).toBe(before);
		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			const r = Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
			min = Math.min(min, r);
			max = Math.max(max, r);
		}
		expect(min).toBeGreaterThan(0.85);
		expect(max).toBeLessThan(1.15);
	});
});

describe("Craters Generation", () => {
	const NAME = "Craters Generation";

	test("cuts craters where the samples are and leaves the rest alone", () => {
		const doc = new MeshDocument();
		const target = doc.addNewMesh("", "target", true, gridPlane(40, 40).mesh);
		target.updateBoxAndNormals();
		const samples = new CMeshO();
		Allocator.addVertices(samples, 1);
		samples.setVert(0, 0.5, 0.5, 0);
		const sampleLayer = doc.addNewMesh("", "samples", false, samples);
		sampleLayer.updateBoxAndNormals();

		const before = Float64Array.from(target.cm.vertCoord);
		const out = kernel.applyFilter(doc, NAME, {
			target_mesh: target.id(),
			samples_mesh: sampleLayer.id(),
			seed: 1,
			rbf: 1,
			min_radius: 0.2,
			max_radius: 0.2,
			min_depth: 0.1,
			max_depth: 0.1,
			elevation: 0.4,
			save_as_quality: false,
		});
		expect(out.craters).toBe(1);

		const cm = target.cm;
		// The centre moved; a far corner did not.
		let centre = 0;
		let corner = 0;
		for (let v = 0; v < cm.vertSize; v++) {
			const d = Math.hypot(cm.vx(v) - 0.5, cm.vy(v) - 0.5);
			if (d < Math.hypot(cm.vx(centre) - 0.5, cm.vy(centre) - 0.5)) centre = v;
			if (d > Math.hypot(cm.vx(corner) - 0.5, cm.vy(corner) - 0.5)) corner = v;
		}
		expect(Math.abs(cm.vz(centre) - before[3 * centre + 2])).toBeGreaterThan(0);
		expect(cm.vz(corner)).toBe(before[3 * corner + 2]);
	});

	test("the same layer for target and samples is refused", () => {
		const { doc, m } = scene(gridPlane(6, 6).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				target_mesh: m.id(),
				samples_mesh: m.id(),
				seed: 1,
				rbf: 1,
				min_radius: 0.1,
				max_radius: 0.2,
				min_depth: 0.05,
				max_depth: 0.1,
				elevation: 0.4,
				save_as_quality: false,
			}),
		).toThrow(/two different layers/);
	});
});

describe("Noisy Isosurface", () => {
	const NAME = "Noisy Isosurface";

	test("extracts a closed surface around the origin", () => {
		const doc = new MeshDocument();
		const out = kernel.applyFilter(doc, NAME, { Resolution: 24, Seed: 1, NoiseScale: 0.2 });
		expect(out.face_number as number).toBeGreaterThan(100);

		const cm = doc.mm().cm;
		UpdateTopology.faceFace(cm);
		expect(Clean.countEdgeNum(cm).boundary).toBe(0);
		// Roughly a unit sphere perturbed by a fifth of its radius.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			const r = Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
			expect(r).toBeGreaterThan(0.5);
			expect(r).toBeLessThan(1.6);
		}
	});

	test("zero noise gives a plain sphere", () => {
		const doc = new MeshDocument();
		kernel.applyFilter(doc, NAME, { Resolution: 24, Seed: 1, NoiseScale: 0 });
		const cm = doc.mm().cm;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect(Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v))).toBeCloseTo(1, 1);
		}
	});

	test("too coarse a grid is refused", () => {
		const doc = new MeshDocument();
		expect(() =>
			kernel.applyFilter(doc, NAME, { Resolution: 2, Seed: 1, NoiseScale: 0.2 }),
		).toThrow(/at least 4/);
	});
});

describe("Voronoi Sampling", () => {
	const NAME = "Voronoi Sampling";

	test("selects exactly the requested number of seeds", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			iterNum: 3,
			sampleNum: 12,
			colorStrategy: 1,
			randomSeed: 1,
		});
		expect(out.seeds).toBe(12);
		let selected = 0;
		for (let v = 0; v < cm.vertSize; v++) if (cm.isVertS(v)) selected++;
		expect(selected).toBe(12);
	});

	test("relaxation spreads the seeds out", () => {
		const spread = (iterations: number) => {
			const { doc, cm } = scene(sphereIcosa(3).mesh);
			kernel.applyFilter(doc, NAME, {
				iterNum: iterations,
				sampleNum: 10,
				colorStrategy: 0,
				randomSeed: 5,
			});
			const seeds: number[] = [];
			for (let v = 0; v < cm.vertSize; v++) if (cm.isVertS(v)) seeds.push(v);
			// The largest distance from any vertex to its nearest seed: the
			// covering radius, which relaxation is supposed to shrink.
			const d = dijkstraGeodesic(cm, seeds);
			let worst = 0;
			for (let v = 0; v < cm.vertSize; v++)
				if (Number.isFinite(d[v])) worst = Math.max(worst, d[v]);
			return worst;
		};
		// Farthest-point initialisation is already good, so the bar is that
		// relaxation does not undo it. An earlier centre heuristic did exactly
		// that, which is what this test is here to catch.
		expect(spread(5)).toBeLessThanOrEqual(spread(0) * 1.05);
	});

	test("asking for more seeds than vertices is refused", () => {
		const { doc, cm } = scene(sphereIcosa(1).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				iterNum: 1,
				sampleNum: cm.vn + 1,
				colorStrategy: 0,
				randomSeed: 1,
			}),
		).toThrow(/only \d+ vertices/);
	});
});

describe("Volumetric Sampling", () => {
	const NAME = "Volumetric Sampling";

	test("every sample lands inside the mesh", () => {
		const { doc } = scene(cube(2).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			sampleVolNum: 4000,
			poissonFiltering: false,
			poissonRadius: 0,
			randomSeed: 1,
		});
		expect(out.samples as number).toBeGreaterThan(100);

		const cloud = doc.mm().cm;
		for (let v = 0; v < cloud.vertSize; v++) {
			// The builder's cube spans -1..1 on each axis.
			expect(Math.abs(cloud.vx(v))).toBeLessThan(1.0001);
			expect(Math.abs(cloud.vy(v))).toBeLessThan(1.0001);
			expect(Math.abs(cloud.vz(v))).toBeLessThan(1.0001);
		}
	});

	test("Poisson filtering enforces the minimum spacing", () => {
		const { doc } = scene(cube(2).mesh);
		const radius = 0.4;
		kernel.applyFilter(doc, NAME, {
			sampleVolNum: 8000,
			poissonFiltering: true,
			poissonRadius: radius,
			randomSeed: 2,
		});
		const cloud = doc.mm().cm;
		expect(cloud.vn).toBeGreaterThan(2);
		for (let a = 0; a < cloud.vertSize; a++) {
			for (let b = a + 1; b < cloud.vertSize; b++) {
				const d = Math.hypot(
					cloud.vx(a) - cloud.vx(b),
					cloud.vy(a) - cloud.vy(b),
					cloud.vz(a) - cloud.vz(b),
				);
				expect(d).toBeGreaterThanOrEqual(radius - 1e-9);
			}
		}
	});

	test("filtering keeps fewer samples than not filtering", () => {
		const count = (filter: boolean) => {
			const { doc } = scene(cube(2).mesh);
			return kernel.applyFilter(doc, NAME, {
				sampleVolNum: 4000,
				poissonFiltering: filter,
				poissonRadius: 0.3,
				randomSeed: 3,
			}).samples as number;
		};
		expect(count(true)).toBeLessThan(count(false));
	});
});

describe("Voronoi Scaffolding", () => {
	test("builds a closed lattice inside the mesh", () => {
		const { doc } = scene(cube(2).mesh);
		const out = kernel.applyFilter(doc, "Voronoi Scaffolding", {
			sampleVolNum: 2000,
			voxelRes: 24,
			isoThr: 1.5,
			relaxStep: 2,
			randomSeed: 1,
		});
		expect(out.face_number as number).toBeGreaterThan(50);
		const cm = doc.mm().cm;
		assertAllocatorConsistent(cm);
		// The struts must stay inside the cube they were carved from.
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			expect(Math.abs(cm.vx(v))).toBeLessThan(1.3);
		}
	});

	test("too coarse a voxel grid is refused", () => {
		const { doc } = scene(cube(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Voronoi Scaffolding", {
				sampleVolNum: 500,
				voxelRes: 2,
				isoThr: 1,
				relaxStep: 1,
				randomSeed: 1,
			}),
		).toThrow(/at least 4/);
	});
});

describe("Create Solid Wireframe", () => {
	const NAME = "Create Solid Wireframe";

	test("thickens the edges into a closed solid", () => {
		const { doc } = scene(cube(2).mesh);
		const out = kernel.applyFilter(doc, NAME, {
			voxelRes: 40,
			edgeCylRadius: 0.15,
			vertSphRadius: 0.2,
		});
		expect(out.face_number as number).toBeGreaterThan(100);

		const cm = doc.mm().cm;
		UpdateTopology.faceFace(cm);
		// A solid: no boundary edges anywhere.
		expect(Clean.countEdgeNum(cm).boundary).toBe(0);
		assertAllocatorConsistent(cm);
	});

	test("a larger strut radius gives a bigger solid", () => {
		const volume = (radius: number) => {
			const { doc } = scene(cube(2).mesh);
			kernel.applyFilter(doc, NAME, {
				voxelRes: 32,
				edgeCylRadius: radius,
				vertSphRadius: radius,
			});
			return Math.abs(Clean.signedVolume(doc.mm().cm));
		};
		expect(volume(0.2)).toBeGreaterThan(volume(0.1));
	});

	test("both radii at zero is refused rather than producing nothing", () => {
		const { doc } = scene(cube(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, NAME, { voxelRes: 24, edgeCylRadius: 0, vertSphRadius: 0 }),
		).toThrow(/both radii are zero/);
	});
});

describe("registry", () => {
	test("all eight are registered under their own plugins", () => {
		const expected: Array<[string, string]> = [
			["Fractal Terrain", "FilterFractal"],
			["Fractal Displacement", "FilterFractal"],
			["Craters Generation", "FilterFractal"],
			["Noisy Isosurface", "FilterCreateIso"],
			["Voronoi Sampling", "FilterVoronoi"],
			["Volumetric Sampling", "FilterVoronoi"],
			["Voronoi Scaffolding", "FilterVoronoi"],
			["Create Solid Wireframe", "FilterVoronoi"],
		];
		for (const [name, plugin] of expected) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe(plugin);
		}
	});
});
