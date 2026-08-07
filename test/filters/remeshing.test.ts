/**
 * Isotropic explicit remeshing and clustering decimation.
 *
 * Remeshing has no single right answer, so what is checked is what it
 * promises: the topology is exactly the topology it was given, the solid keeps
 * roughly its volume, the edge lengths cluster around the target, and running
 * it longer converges rather than running away. The last one is not a nicety —
 * without the guard that a collapse may not create an over-long edge, split
 * and collapse cycle against each other and a cylinder loses 80% of its volume
 * in a single pass.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MLException, MLNotImplementedException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { triQuality } from "../../src/vcg/complex/edge_ops.ts";
import {
	clusteringDecimation,
	isotropicRemeshing,
	REMESH_DEFAULTS,
	type RemeshOptions,
} from "../../src/vcg/complex/isotropic_remeshing.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { assertAllocatorConsistent, computeFacts, signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();
const REMESH = "Remeshing: Isotropic Explicit Remeshing";
const CLUSTER = "Simplification: Clustering Decimation";

function remesh(mesh: CMeshO, over: Partial<RemeshOptions> = {}): CMeshO {
	isotropicRemeshing(mesh, {
		...REMESH_DEFAULTS,
		targetLen: 0.25,
		maxSurfDist: 0.05,
		...over,
	});
	return mesh;
}

/** Every unique edge length in the mesh. */
function edgeLengths(m: CMeshO): number[] {
	UpdateTopology.faceFace(m);
	const out: number[] = [];
	const seen = new Set<number>();
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const a = m.fv(f, k);
			const b = m.fv(f, (k + 1) % 3);
			const key = a < b ? a * m.vertSize + b : b * m.vertSize + a;
			if (seen.has(key)) continue;
			seen.add(key);
			out.push(Math.hypot(m.vx(a) - m.vx(b), m.vy(a) - m.vy(b), m.vz(a) - m.vz(b)));
		}
	}
	return out;
}

/** Standard deviation over the mean: how uneven the edge lengths are. */
function spread(lengths: readonly number[]): number {
	const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
	const variance = lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / lengths.length;
	return Math.sqrt(variance) / mean;
}

function worstQuality(m: CMeshO): number {
	let worst = Number.POSITIVE_INFINITY;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const a = m.fv(f, 0);
		const b = m.fv(f, 1);
		const c = m.fv(f, 2);
		worst = Math.min(
			worst,
			triQuality(m.vx(a), m.vy(a), m.vz(a), m.vx(b), m.vy(b), m.vz(b), m.vx(c), m.vy(c), m.vz(c)),
		);
	}
	return worst;
}

/** A cylinder: short edges around, very long ones along, and two sharp rims. */
const cylinder = () => Platonic.cone(1, 1, 3, 36);
const CYLINDER_VOLUME = signedVolume(cylinder());

describe("isotropic remeshing", () => {
	test("keeps the topology it was given", () => {
		for (const [label, mesh, genus] of [
			["sphere", Platonic.sphere(3), 0],
			["torus", Platonic.torus(3, 1, 24, 12), 1],
			["cylinder", cylinder(), 0],
		] as const) {
			const out = remesh(mesh, { iterations: 4 });
			const facts = computeFacts(out);
			expect(facts.watertight, label).toBe(true);
			expect(facts.components, label).toBe(1);
			expect(facts.nonManifoldEdges, label).toBe(0);
			expect(facts.genus, label).toBe(genus);
			expect(facts.coherentlyOriented, label).toBe(true);
			assertAllocatorConsistent(out, label);
		}
	});

	test("leaves no degenerate triangle behind", () => {
		// Two distinct edges can bisect to exactly the same point — the two
		// diagonals of a parallelogram on a cylinder wall do — and once a flip
		// puts both midpoints in one face, that face has zero area. The weld at
		// the end of the pass is what removes them.
		for (const iterations of [1, 3, 5, 6]) {
			const out = remesh(cylinder(), { iterations });
			expect(worstQuality(out), `iterations ${iterations}`).toBeGreaterThan(0);
		}
	});

	test("holds the volume", () => {
		// The surface may wobble by the allowed distance, but a remesher that
		// loses a fifth of the solid has folded it, not resampled it.
		const out = remesh(cylinder(), { iterations: 5 });
		expect(signedVolume(out)).toBeGreaterThan(CYLINDER_VOLUME * 0.97);
		expect(signedVolume(out)).toBeLessThan(CYLINDER_VOLUME * 1.03);
	});

	test("converges instead of running away", () => {
		// Split and collapse must not undo each other. They did until the
		// collapse learned to refuse anything that would leave an over-long
		// edge: ten passes turned 74 vertices into 52,000.
		const counts = [4, 6, 8, 10].map((iterations) => remesh(cylinder(), { iterations }).vn);
		for (let i = 1; i < counts.length; i++) {
			// Growth from one setting to the next has to be modest, not fourfold.
			expect(counts[i], `at ${[4, 6, 8, 10][i]} iterations`).toBeLessThan(counts[i - 1] * 1.5);
		}
	});

	test("evens out an anisotropic mesh", () => {
		const before = spread(edgeLengths(cylinder()));
		const after = spread(edgeLengths(remesh(cylinder(), { iterations: 6, targetLen: 0.2 })));
		// The cylinder starts with edges from 0.17 to 3.01; the rims stay pinned
		// as creases, so the spread improves substantially without vanishing.
		expect(before).toBeGreaterThan(0.8);
		expect(after).toBeLessThan(before * 0.8);
	});

	test("brings edge lengths to the target on a smooth surface", () => {
		const lengths = edgeLengths(remesh(Platonic.sphere(3), { iterations: 5, targetLen: 0.3 }));
		const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
		expect(mean).toBeGreaterThan(0.3 * 0.8);
		expect(mean).toBeLessThan(0.3 * 1.34);
		// And every edge inside the band the algorithm is defined by.
		expect(Math.min(...lengths)).toBeGreaterThan(0.3 * 0.5);
		expect(Math.max(...lengths)).toBeLessThan(0.3 * 1.5);
		expect(spread(lengths)).toBeLessThan(0.2);
	});

	test("a smaller target gives a denser mesh", () => {
		let previous = 0;
		for (const targetLen of [0.5, 0.35, 0.25, 0.15]) {
			const out = remesh(Platonic.sphere(3), { iterations: 4, targetLen });
			expect(out.vn, `target ${targetLen}`).toBeGreaterThan(previous);
			previous = out.vn;
		}
	});

	test("a denser remesh is closer to the sphere it came from", () => {
		const ideal = signedVolume(Platonic.sphere(3));
		let previous = 0;
		for (const targetLen of [0.5, 0.35, 0.25, 0.15]) {
			const v = signedVolume(remesh(Platonic.sphere(3), { iterations: 4, targetLen }));
			expect(v, `target ${targetLen}`).toBeGreaterThan(previous);
			expect(v, `target ${targetLen}`).toBeLessThanOrEqual(ideal * 1.001);
			previous = v;
		}
	});

	test("each step can be turned off on its own", () => {
		for (const off of [
			"splitFlag",
			"collapseFlag",
			"swapFlag",
			"smoothFlag",
			"reprojectFlag",
		] as const) {
			const out = remesh(Platonic.sphere(3), { iterations: 2, [off]: false });
			const facts = computeFacts(out);
			expect(facts.watertight, off).toBe(true);
			expect(facts.genus, off).toBe(0);
			expect(facts.nonManifoldEdges, off).toBe(0);
		}
	});

	test("with every step off it changes nothing", () => {
		const out = remesh(Platonic.sphere(2), {
			splitFlag: false,
			collapseFlag: false,
			swapFlag: false,
			smoothFlag: false,
		});
		expect(out.vn).toBe(162);
		expect(out.fn).toBe(320);
	});

	test("a target the mesh already meets is close to a no-op", () => {
		// A subdiv-3 sphere has edges around 0.15 already.
		const out = remesh(Platonic.sphere(3), { iterations: 3, targetLen: 0.15 });
		expect(out.vn).toBeGreaterThan(600);
		expect(out.vn).toBeLessThan(700);
	});

	test("rejects a target length of zero", () => {
		expect(() => remesh(Platonic.sphere(2), { targetLen: 0 })).toThrow();
		expect(() => remesh(Platonic.sphere(2), { targetLen: -1 })).toThrow();
	});

	test("an open mesh keeps its boundary", () => {
		const out = remesh(Platonic.sphericalCap(Math.PI / 2, 3), { iterations: 3, targetLen: 0.15 });
		const facts = computeFacts(out);
		expect(facts.boundaryLoops).toBe(1);
		expect(facts.components).toBe(1);
		expect(facts.nonManifoldEdges).toBe(0);
	});
});

describe("clustering decimation", () => {
	test("a bigger cell gives a coarser mesh", () => {
		let previous = Number.POSITIVE_INFINITY;
		for (const cell of [0.05, 0.1, 0.3, 0.6]) {
			const m = Platonic.sphere(3);
			clusteringDecimation(m, cell);
			expect(m.vn, `cell ${cell}`).toBeLessThanOrEqual(previous);
			previous = m.vn;
		}
	});

	test("keeps the surface closed and the right genus", () => {
		for (const [label, mesh, genus] of [
			["sphere", Platonic.sphere(4), 0],
			["torus", Platonic.torus(3, 1, 40, 20), 1],
		] as const) {
			clusteringDecimation(mesh, 0.35);
			const facts = computeFacts(mesh);
			expect(facts.nonManifoldEdges, label).toBe(0);
			expect(facts.components, label).toBe(1);
			expect(facts.watertight, label).toBe(true);
			expect(facts.genus, label).toBe(genus);
			assertAllocatorConsistent(mesh, label);
		}
	});

	test("a cell smaller than every edge changes nothing", () => {
		const m = Platonic.sphere(3);
		clusteringDecimation(m, 0.01);
		expect(m.vn).toBe(642);
		expect(m.fn).toBe(1280);
	});

	test("every kept vertex is the average of the points in its cell", () => {
		// Not a subset of the input, unlike Poisson-disk sampling — the cell
		// representative is a new point, which is what lets it be so cheap.
		const m = Platonic.sphere(3);
		const original = new Set<string>();
		for (let v = 0; v < m.vn; v++) original.add(`${m.vx(v)},${m.vy(v)},${m.vz(v)}`);
		clusteringDecimation(m, 0.4);
		let survivors = 0;
		for (let v = 0; v < m.vn; v++) {
			if (original.has(`${m.vx(v)},${m.vy(v)},${m.vz(v)}`)) survivors++;
		}
		expect(survivors).toBeLessThan(m.vn);
	});

	test("shrinks the solid, as averaging a convex surface must", () => {
		const before = signedVolume(Platonic.sphere(3));
		const m = Platonic.sphere(3);
		clusteringDecimation(m, 0.4);
		expect(signedVolume(m)).toBeLessThan(before);
		expect(signedVolume(m)).toBeGreaterThan(before * 0.8);
	});

	test("thins a point cloud too, with no faces to speak of", () => {
		const cloud = Platonic.pointCloudFrom(
			Array.from({ length: 500 }, (_, i) => {
				const a = (i * 2.399963) % (2 * Math.PI);
				const z = 1 - (2 * i + 1) / 500;
				const r = Math.sqrt(Math.max(0, 1 - z * z));
				return [r * Math.cos(a), r * Math.sin(a), z] as [number, number, number];
			}),
		);
		clusteringDecimation(cloud, 0.3);
		expect(cloud.fn).toBe(0);
		expect(cloud.vn).toBeGreaterThan(0);
		expect(cloud.vn).toBeLessThan(500);
	});

	test("rejects a cell size of zero", () => {
		expect(() => clusteringDecimation(Platonic.sphere(2), 0)).toThrow();
		expect(() => clusteringDecimation(Platonic.sphere(2), -1)).toThrow();
	});
});

describe("the filters", () => {
	test("are registered as MeshLab registers them", () => {
		for (const [name, pythonName] of [
			[REMESH, "meshing_isotropic_explicit_remeshing"],
			[CLUSTER, "meshing_decimation_clustering"],
		] as const) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(action.plugin.pluginName(), name).toBe("FilterMeshing");
		}
	});

	test("carry MeshLab's parameter defaults", () => {
		const list = kernel.initParameterList(REMESH);
		expect(list.getParameterByName("Iterations").defaultValue.value).toBe(3);
		expect(list.getParameterByName("FeatureDeg").defaultValue.value).toBe(30);
		expect(list.getParameterByName("Adaptive").defaultValue.value).toBe(false);
		for (const flag of ["SplitFlag", "CollapseFlag", "SwapFlag", "SmoothFlag", "ReprojectFlag"]) {
			expect(list.getParameterByName(flag).defaultValue.value, flag).toBe(true);
		}
		expect(kernel.initParameterList(CLUSTER).hasParameter("Threshold")).toBe(true);
	});

	test("remeshing runs end to end and reports what it did", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(3));
		const out = kernel.applyFilter(doc, REMESH, { Iterations: 3, TargetLen: 0.3 });
		expect(out.vertex_number).toBe(doc.mm().cm.vn);
		expect(out.face_number).toBe(doc.mm().cm.fn);
		expect(out.collapses as number).toBeGreaterThan(0);
		expect(computeFacts(doc.mm().cm).watertight).toBe(true);
	});

	test("clustering runs end to end", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(4));
		const out = kernel.applyFilter(doc, CLUSTER, { Threshold: 0.3 });
		expect(out.vertex_number).toBe(doc.mm().cm.vn);
		expect(doc.mm().cm.vn).toBeLessThan(2562);
		expect(computeFacts(doc.mm().cm).watertight).toBe(true);
	});

	test("refuse a degenerate setting rather than doing something arbitrary", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(2));
		expect(() => kernel.applyFilter(doc, REMESH, { TargetLen: 0 })).toThrow(MLException);
		expect(() => kernel.applyFilter(doc, CLUSTER, { Threshold: 0 })).toThrow(MLException);
	});

	test("adaptive remeshing says it is not implemented", () => {
		// The target length is applied uniformly; pretending to adapt it to
		// curvature would give a mesh the user did not ask for.
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, Platonic.sphere(2));
		expect(() => kernel.applyFilter(doc, REMESH, { Adaptive: true })).toThrow(
			MLNotImplementedException,
		);
	});
});
