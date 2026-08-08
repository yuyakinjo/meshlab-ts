/**
 * Stage C of the texture-defragmentation port: the greedy driver.
 *
 * This is the stage with no exact oracle — which merges a heuristic accepts is
 * not something to assert. Two kinds of test instead.
 *
 * The first is one case where the answer *is* exact: a grid split into two
 * charts by a pure translation has to merge back into the grid, and the
 * resulting outline has to be the grid's own perimeter, to the digit. Anything
 * short of a correct alignment, weld and relaxation fails that.
 *
 * The second is the monotone properties, which hold whatever the heuristic
 * decides: charts never increase, faces never move between meshes, the atlas
 * outline never grows, coordinates stay finite, and a rejected move leaves
 * nothing behind. Those are what can actually be promised about this stage.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import {
	type AtlasMesh,
	buildAtlasMesh,
	computeChartGraph,
} from "../../src/vcg/complex/parametrization/chart_graph.ts";
import {
	type DefragParameters,
	defragmentAtlas,
} from "../../src/vcg/complex/parametrization/defragment.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** An `n × m` grid whose right part's UVs are put through `place`. */
function splitGrid(
	n: number,
	m: number,
	cut: number,
	place: (x: number, y: number) => [number, number],
): CMeshO {
	const cm = new CMeshO();
	const index = (i: number, j: number): number => j * (n + 1) + i;
	Allocator.addVertices(cm, (n + 1) * (m + 1));
	for (let j = 0; j <= m; j++) {
		for (let i = 0; i <= n; i++) cm.setVert(index(i, j), i, j, 0);
	}
	Allocator.addFaces(cm, 2 * n * m);
	let f = 0;
	for (let j = 0; j < m; j++) {
		for (let i = 0; i < n; i++) {
			cm.setFace(f++, index(i, j), index(i + 1, j), index(i + 1, j + 1));
			cm.setFace(f++, index(i, j), index(i + 1, j + 1), index(i, j + 1));
		}
	}
	const doc = new MeshDocument();
	doc.addNewMesh("", "m", true, cm).updateDataMask(MeshElement.MM_WEDGTEXCOORD);
	const wt = cm.wedgeTexCoord as Float64Array;
	for (let g = 0; g < cm.faceSize; g++) {
		const centre = [0, 1, 2].reduce((s, q) => s + cm.vx(cm.fv(g, q)), 0) / 3;
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(g, k);
			const [u, w] = centre > cut ? place(cm.vx(v), cm.vy(v)) : [cm.vx(v), cm.vy(v)];
			wt[6 * g + 2 * k] = u;
			wt[6 * g + 2 * k + 1] = w;
		}
	}
	return cm;
}

/** A sphere cut into one chart per triangle: the most fragmented atlas there is. */
function shatteredSphere(subdiv = 2): AtlasMesh {
	const cm = sphereIcosa(subdiv).mesh;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
	m.updateBoxAndNormals();
	kernel.applyFilter(doc, "Parametrization: Trivial Per-Triangle", {});
	return buildAtlasMesh(cm);
}

const run = (am: AtlasMesh, params: DefragParameters = {}) =>
	defragmentAtlas(am, { boundaryTolerance: 0, ...params });

describe("the greedy driver", () => {
	test("puts a grid split by a translation back together, exactly", () => {
		// The one case with an exact answer. A 5x4 grid cut down the middle has
		// an outline of 2(5+4) + 2·4 = 26; merged, it is the grid's own 18.
		const am = buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [x + 30, y]));
		const result = run(am);
		expect(result.chartsBefore).toBe(2);
		expect(result.chartsAfter).toBe(1);
		expect(result.merges).toBe(1);
		expect(result.borderUVBefore).toBeCloseTo(26, 9);
		expect(result.borderUVAfter).toBeCloseTo(18, 9);
		expect(result.seamEdgesRemoved).toBe(4);
	});

	test("and puts it back in the right place, not just the right shape", () => {
		// After the merge every face should hold the UV triangle it started
		// with, up to the translation that separated the two halves.
		const am = buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [x + 30, y]));
		const result = run(am);
		for (let f = 0; f < am.faceCount; f++) {
			for (let k = 0; k < 3; k++) {
				const v = am.faces[3 * f + k];
				// The atlas vertex's own 3D position is its expected UV, because
				// the grid is its own parametrization.
				expect(result.uv[2 * v] % 30).toBeCloseTo(am.positions[3 * v], 6);
				expect(result.uv[2 * v + 1]).toBeCloseTo(am.positions[3 * v + 1], 6);
			}
		}
	});

	test("leaves an atlas that is already one chart alone", () => {
		const am = buildAtlasMesh(splitGrid(4, 3, 2, (x, y) => [x, y]));
		const result = run(am);
		expect(result.chartsBefore).toBe(1);
		expect(result.merges).toBe(0);
		expect(result.borderUVAfter).toBe(result.borderUVBefore);
		expect([...result.uv]).toEqual([...am.uv]);
	});

	test("merges the most fragmented atlas there is, and shrinks its outline", () => {
		const am = shatteredSphere();
		const result = run(am);
		expect(result.chartsBefore).toBe(am.faceCount);
		expect(result.chartsAfter).toBeLessThan(result.chartsBefore / 4);
		expect(result.borderUVAfter).toBeLessThan(result.borderUVBefore);
		expect(result.merges).toBeGreaterThan(0);
	});
});

describe("what holds however the heuristic decides", () => {
	const scenarios: Array<[string, () => AtlasMesh]> = [
		["translated halves", () => buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [x + 30, y]))],
		["rotated half", () => buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [-y + 40, x]))],
		["scaled half", () => buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [2 * x + 40, 2 * y]))],
		["shattered sphere", () => shatteredSphere()],
	];

	for (const [name, build] of scenarios) {
		test(`${name}: charts never increase and no face is lost`, () => {
			const am = build();
			const result = run(am);
			expect(result.chartsAfter).toBeLessThanOrEqual(result.chartsBefore);
			expect(result.chartOf.length).toBe(am.faceCount);
			expect(new Set(result.chartOf).size).toBe(result.chartsAfter);
			for (const id of result.chartOf) expect(id).toBeGreaterThanOrEqual(0);
		});

		test(`${name}: the atlas outline never grows`, () => {
			const result = run(build());
			expect(result.borderUVAfter).toBeLessThanOrEqual(result.borderUVBefore + 1e-9);
		});

		test(`${name}: every coordinate stays finite and no face is degenerate`, () => {
			const am = build();
			const result = run(am);
			for (const value of result.uv) expect(Number.isFinite(value)).toBe(true);
			let degenerate = 0;
			for (let f = 0; f < am.faceCount; f++) {
				const t = [0, 1, 2].map((k) => {
					const v = am.faces[3 * f + k];
					return [result.uv[2 * v], result.uv[2 * v + 1]];
				});
				const area =
					Math.abs(
						(t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]),
					) / 2;
				if (area < 1e-12) degenerate++;
			}
			expect(degenerate).toBe(0);
		});

		test(`${name}: the same input gives the same output`, () => {
			// No randomness and no iteration-order dependence: a heuristic that
			// wandered would make every downstream test flaky.
			const first = run(build());
			const second = run(build());
			expect(second.merges).toBe(first.merges);
			expect([...second.uv]).toEqual([...first.uv]);
			expect([...second.chartOf]).toEqual([...first.chartOf]);
		});
	}

	test("a merge only ever joins charts that shared a seam", () => {
		const am = shatteredSphere();
		const before = computeChartGraph(am);
		const result = run(am);
		// Faces that ended in one chart must be connected through the original
		// chart adjacency — a merge cannot join two charts that never touched.
		const members = new Map<number, number[]>();
		for (let f = 0; f < am.faceCount; f++) {
			const id = result.chartOf[f];
			const list = members.get(id);
			if (list === undefined) members.set(id, [f]);
			else list.push(f);
		}
		for (const faces of members.values()) {
			const originals = new Set(faces.map((f) => before.chartOf[f]));
			// Walk the original adjacency and check the set is connected in it.
			const start = [...originals][0];
			const seen = new Set([start]);
			const stack = [start];
			while (stack.length > 0) {
				const id = stack.pop() as number;
				for (const next of (before.charts.get(id) as { adjacent: Set<number> }).adjacent) {
					if (!originals.has(next) || seen.has(next)) continue;
					seen.add(next);
					stack.push(next);
				}
			}
			expect(seen.size).toBe(originals.size);
		}
	});
});

describe("the controls", () => {
	test("a move limit stops it where it says", () => {
		const am = shatteredSphere();
		const limited = run(am, { maxMoves: 5 });
		expect(limited.stopped).toBe("move-limit");
		expect(limited.iterations).toBe(5);
		expect(limited.merges).toBeLessThanOrEqual(5);
		expect(limited.chartsAfter).toBeGreaterThan(run(am).chartsAfter);
	});

	test("an outline target stops it once reached", () => {
		const am = shatteredSphere();
		const result = run(am, { uvBorderLengthReduction: 0.8 });
		expect(result.stopped).toBe("border-target");
		expect(result.borderUVAfter).toBeLessThanOrEqual(0.8 * result.borderUVBefore);
	});

	test("a boundary tolerance above 1 refuses every seam", () => {
		// The tolerance is a fraction of a chart's outline, so nothing can
		// exceed it and no move should even be attempted.
		const am = buildAtlasMesh(splitGrid(5, 4, 2, (x, y) => [x + 30, y]));
		const result = defragmentAtlas(am, { boundaryTolerance: 10 });
		expect(result.merges).toBe(0);
		expect(result.rejected["unfeasible-boundary"]).toBeGreaterThan(0);
		expect([...result.uv]).toEqual([...am.uv]);
	});

	test("a distortion tolerance of zero allows only the free merges", () => {
		// Not zero merges: two triangles that already agree about the seam cost
		// exactly nothing to join, and refusing those would be wrong. What the
		// tolerance must do is refuse everything else.
		const am = shatteredSphere();
		const strict = run(am, { distortionTolerance: 0 });
		expect(strict.rejected["distortion-local"]).toBeGreaterThan(0);
		expect(strict.merges).toBeLessThan(run(am).merges / 4);
	});

	test("rejections are counted, and every rejection has a reason", () => {
		const result = run(shatteredSphere(), { distortionTolerance: 0.05 });
		const total = Object.values(result.rejected).reduce((a, b) => a + b, 0);
		expect(total).toBeGreaterThan(0);
		expect(result.iterations).toBeGreaterThanOrEqual(result.merges);
	});

	test("a rejected move leaves nothing behind", () => {
		// The reason moves are computed on a copy rather than unwound: with every
		// move refused, the coordinates must be bit-identical to the input, not
		// merely close to it.
		const am = shatteredSphere();
		const result = defragmentAtlas(am, { boundaryTolerance: 10 });
		expect(result.merges).toBe(0);
		expect([...result.uv]).toEqual([...am.uv]);
		expect([...result.chartOf]).toEqual([...computeChartGraph(am).chartOf]);
		expect([...result.vertexRep]).toEqual([...am.faces.map((_, i) => i).slice(0, am.vertexCount)]);
	});
});
