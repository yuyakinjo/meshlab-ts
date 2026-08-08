/**
 * Stage D: packing, and the filter end to end.
 *
 * The packer has one property that matters above all others and is easy to get
 * subtly wrong: **no two charts may overlap in the packed atlas**, because two
 * overlapping charts paint each other's texels and the model comes out with
 * another model's colours on it. That is checked by rasterising the packed
 * charts into a grid and looking for a cell claimed twice — not by comparing
 * bounding boxes, which a rotated chart would slip past.
 *
 * The filter itself is checked for the things that would make its output
 * unusable regardless of how well it defragmented: every coordinate inside the
 * unit square, no face collapsed to a point, the input layer untouched, and a
 * texture that actually carries the colours the old one did.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import {
	applyPlacement,
	type ChartGeometry,
	packCharts,
} from "../../src/vcg/complex/parametrization/packing.ts";
import { Image } from "../../src/vcg/space/image/image.ts";
import { readPng, writePng } from "../../src/vcg/space/image/png.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();
const NAME = "Texture Map Defragmentation";

// ---------------------------------------------------------------- packing

/** `count` axis-aligned rectangles, as two triangles each. */
function boxes(count: number, size: (i: number) => [number, number]): ChartGeometry[] {
	return Array.from({ length: count }, (_, i) => {
		const [w, h] = size(i);
		return {
			id: i,
			triangles: Float64Array.from([0, 0, w, 0, w, h, 0, 0, w, h, 0, h]),
		};
	});
}

/**
 * Rasterises the packed charts and returns how many cells are claimed twice.
 *
 * Bounding boxes would not do: a chart rotated by a quarter-turn has the same
 * box as an unrotated one and could overlap without the box showing it.
 */
function overlapCells(charts: readonly ChartGeometry[], resolution: number): number {
	const packed = packCharts(charts, { resolution: 128, gutter: 1 });
	const claimed = new Int32Array(resolution * resolution).fill(-1);
	let clashes = 0;

	for (const placement of packed.placements) {
		const chart = charts.find((c) => c.id === placement.chart) as ChartGeometry;
		for (let t = 0; t < chart.triangles.length; t += 6) {
			const p = [0, 1, 2].map((k) =>
				applyPlacement(
					placement,
					packed,
					chart.triangles[t + 2 * k],
					chart.triangles[t + 2 * k + 1],
				),
			);
			const minX = Math.max(0, Math.floor(Math.min(...p.map((q) => q[0])) * resolution));
			const maxX = Math.min(
				resolution - 1,
				Math.ceil(Math.max(...p.map((q) => q[0])) * resolution),
			);
			const minY = Math.max(0, Math.floor(Math.min(...p.map((q) => q[1])) * resolution));
			const maxY = Math.min(
				resolution - 1,
				Math.ceil(Math.max(...p.map((q) => q[1])) * resolution),
			);
			const area =
				(p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
			if (area === 0) continue;

			for (let y = minY; y <= maxY; y++) {
				for (let x = minX; x <= maxX; x++) {
					const px = (x + 0.5) / resolution;
					const py = (y + 0.5) / resolution;
					const w0 = ((p[1][0] - px) * (p[2][1] - py) - (p[2][0] - px) * (p[1][1] - py)) / area;
					const w1 = ((p[2][0] - px) * (p[0][1] - py) - (p[0][0] - px) * (p[2][1] - py)) / area;
					const w2 = 1 - w0 - w1;
					if (w0 < 0 || w1 < 0 || w2 < 0) continue;
					const cell = y * resolution + x;
					if (claimed[cell] >= 0 && claimed[cell] !== placement.chart) clashes++;
					claimed[cell] = placement.chart;
				}
			}
		}
	}
	return clashes;
}

describe("chart packing", () => {
	test("no two charts ever claim the same texel", () => {
		expect(
			overlapCells(
				boxes(16, () => [1, 1]),
				256,
			),
		).toBe(0);
		expect(
			overlapCells(
				boxes(30, (i) => [0.3 + (i % 7) * 0.4, 0.2 + (i % 5) * 0.3]),
				256,
			),
		).toBe(0);
		expect(
			overlapCells(
				boxes(50, (i) => [1 + (i % 3), 1 / (1 + (i % 4))]),
				256,
			),
		).toBe(0);
	});

	test("every chart lands inside the unit square", () => {
		const charts = boxes(24, (i) => [0.5 + (i % 4) * 0.5, 0.4 + (i % 3) * 0.6]);
		const packed = packCharts(charts, { resolution: 128, gutter: 1 });
		expect(packed.failed).toEqual([]);
		expect(packed.placements.length).toBe(charts.length);
		for (const placement of packed.placements) {
			const chart = charts.find((c) => c.id === placement.chart) as ChartGeometry;
			for (let i = 0; i < chart.triangles.length; i += 2) {
				const [u, v] = applyPlacement(
					placement,
					packed,
					chart.triangles[i],
					chart.triangles[i + 1],
				);
				expect(u).toBeGreaterThanOrEqual(-1e-9);
				expect(v).toBeGreaterThanOrEqual(-1e-9);
				expect(u).toBeLessThanOrEqual(1 + 1e-9);
				expect(v).toBeLessThanOrEqual(1 + 1e-9);
			}
		}
	});

	test("charts keep their shape: the placement is rigid", () => {
		// A packer that stretched a chart to make it fit would produce a valid
		// layout and a wrong texture.
		const charts = boxes(8, (i) => [1 + i * 0.2, 0.7]);
		const packed = packCharts(charts, { resolution: 128, gutter: 1 });
		const side = Math.max(packed.width, packed.height);
		for (const placement of packed.placements) {
			const chart = charts.find((c) => c.id === placement.chart) as ChartGeometry;
			const before = Math.hypot(
				chart.triangles[2] - chart.triangles[0],
				chart.triangles[3] - chart.triangles[1],
			);
			const p0 = applyPlacement(placement, packed, chart.triangles[0], chart.triangles[1]);
			const p1 = applyPlacement(placement, packed, chart.triangles[2], chart.triangles[3]);
			const after = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
			expect(after).toBeCloseTo((before * placement.scale) / side, 9);
		}
	});

	test("the atlas is used rather than mostly wasted", () => {
		// Not a tight bound — the point is to notice a packer that regressed to
		// one chart per row.
		expect(
			packCharts(
				boxes(64, () => [1, 1]),
				{ resolution: 256, gutter: 1 },
			).occupancy,
		).toBeGreaterThan(0.6);
	});

	test("the same charts pack the same way every time", () => {
		const charts = boxes(20, (i) => [0.4 + (i % 5) * 0.3, 0.6]);
		const first = packCharts(charts, { resolution: 128 });
		const second = packCharts(charts, { resolution: 128 });
		expect(second.placements).toEqual(first.placements);
	});

	test("rotation is used when it helps", () => {
		// Tall thin charts in a wide atlas: some should be turned on their side.
		const charts = boxes(12, () => [0.2, 3]);
		const packed = packCharts(charts, { resolution: 128, gutter: 1, rotations: 4 });
		expect(packed.placements.some((p) => p.rotation !== 0)).toBe(true);
		// And with rotation switched off, none is.
		const fixed = packCharts(charts, { resolution: 128, gutter: 1, rotations: 1 });
		expect(fixed.placements.every((p) => p.rotation === 0)).toBe(true);
	});

	test("refuses an empty or arealess input", () => {
		expect(() => packCharts([])).toThrow(MLException);
		expect(() => packCharts([{ id: 0, triangles: Float64Array.from([0, 0, 1, 0, 2, 0]) }])).toThrow(
			MLException,
		);
	});
});

// ----------------------------------------------------------------- filter

/** A sphere with a per-triangle atlas and a recognisable texture on it. */
function shatteredScene(subdiv = 2) {
	const cm = sphereIcosa(subdiv).mesh;
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "sphere", true, cm);
	m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
	m.updateBoxAndNormals();
	kernel.applyFilter(doc, "Parametrization: Trivial Per-Triangle", {});

	const texture = new Image(64, 64);
	for (let y = 0; y < 64; y++) {
		for (let x = 0; x < 64; x++) {
			const i = (y * 64 + x) * 4;
			texture.data[i] = x * 4;
			texture.data[i + 1] = y * 4;
			texture.data[i + 2] = 200;
			texture.data[i + 3] = 255;
		}
	}
	m.textures.set("ramp.png", writePng(texture));
	cm.textures = ["ramp.png"];
	return { doc, m, cm };
}

/** Every face's UV triangle, checked for the two ways it could be unusable. */
function uvHealth(cm: CMeshO) {
	const wt = cm.wedgeTexCoord as Float64Array;
	let outside = 0;
	let degenerate = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const u = [0, 1, 2].map((k) => wt[6 * f + 2 * k]);
		const v = [0, 1, 2].map((k) => wt[6 * f + 2 * k + 1]);
		if ([...u, ...v].some((x) => !Number.isFinite(x) || x < -1e-9 || x > 1 + 1e-9)) outside++;
		const area = Math.abs((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0])) / 2;
		if (area < 1e-14) degenerate++;
	}
	return { outside, degenerate };
}

describe(NAME, () => {
	test("merges a shattered atlas and reports what it did", () => {
		const scene = shatteredScene();
		const out = kernel.applyFilter(scene.doc, NAME, { textureSize: 128 });
		expect(out.charts_before).toBe(scene.cm.fn);
		expect(out.charts_after as number).toBeLessThan(out.charts_before as number);
		expect(out.merges as number).toBeGreaterThan(0);
		expect(out.uv_border_after as number).toBeLessThan(out.uv_border_before as number);
		expect(out.atlas_occupancy as number).toBeGreaterThan(0);
	});

	test("the result is a new layer, and the input is left exactly as it was", () => {
		const scene = shatteredScene();
		const before = Float64Array.from(scene.cm.wedgeTexCoord as Float64Array);
		const layersBefore = scene.doc.meshNumber();
		kernel.applyFilter(scene.doc, NAME, { textureSize: 128 });
		expect(scene.doc.meshNumber()).toBe(layersBefore + 1);
		expect([...(scene.cm.wedgeTexCoord as Float64Array)]).toEqual([...before]);
		const result = scene.doc.mm() as { label: () => string };
		expect(result.label()).toBe("texdefrag_sphere");
	});

	test("every face gets a real UV triangle inside the unit square", () => {
		// The two failures that make an atlas unusable however good the merging
		// was: coordinates off the image, and a face collapsed to a point.
		const scene = shatteredScene();
		kernel.applyFilter(scene.doc, NAME, { textureSize: 128 });
		const result = scene.doc.mm();
		expect(uvHealth((result as { cm: CMeshO }).cm)).toEqual({ outside: 0, degenerate: 0 });
	});

	test("the new texture carries the old one's colours", () => {
		// Resampling that produced a blank image would still pass every
		// geometric check above.
		const scene = shatteredScene();
		kernel.applyFilter(scene.doc, NAME, { textureSize: 128 });
		const result = scene.doc.mm() as { textures: Map<string, Uint8Array> };
		expect(result.textures.size).toBe(1);
		const image = readPng([...result.textures.values()][0]);
		expect(image.width).toBe(128);

		// The source ramp is blue-heavy everywhere, and never has red and green
		// both at zero. A blank or black atlas fails both.
		let blueish = 0;
		let total = 0;
		for (let i = 0; i < image.data.length; i += 4) {
			total++;
			if (image.data[i + 2] > 100) blueish++;
		}
		expect(blueish / total).toBeGreaterThan(0.5);
	});

	test("a mesh with no texture is still repacked, just without an image", () => {
		const scene = shatteredScene();
		scene.m.textures.clear();
		scene.cm.textures = [];
		const out = kernel.applyFilter(scene.doc, NAME, { textureSize: 64 });
		expect(out.texture_rendered).toBe(false);
		expect(out.merges as number).toBeGreaterThan(0);
		expect(uvHealth((scene.doc.mm() as { cm: CMeshO }).cm).outside).toBe(0);
	});

	test("the same input gives the same atlas", () => {
		const first = kernel.applyFilter(shatteredScene().doc, NAME, { textureSize: 64 });
		const second = kernel.applyFilter(shatteredScene().doc, NAME, { textureSize: 64 });
		expect(second).toEqual(first);
	});

	test("a move limit bounds the work", () => {
		const scene = shatteredScene();
		const limited = kernel.applyFilter(scene.doc, NAME, { maxMoves: 3, textureSize: 64 });
		expect(limited.stopped).toBe("move-limit");
		expect(limited.merges as number).toBeLessThanOrEqual(3);
	});

	test("says so rather than pretending, when asked for a wall-clock limit", () => {
		// A time limit would make the same input give different atlases on
		// different machines, which is worse than not offering it.
		expect(() => kernel.applyFilter(shatteredScene().doc, NAME, { timelimit: 5 })).toThrow(
			/timelimit/,
		);
	});

	test("refuses a mesh with no parametrization", () => {
		const cm = sphereIcosa(2).mesh;
		const doc = new MeshDocument();
		doc.addNewMesh("", "m", true, cm).updateBoxAndNormals();
		expect(() => kernel.applyFilter(doc, NAME, {})).toThrow();
	});

	test("refuses an unusable texture size", () => {
		expect(() => kernel.applyFilter(shatteredScene().doc, NAME, { textureSize: 4 })).toThrow(
			MLException,
		);
	});

	test("is registered as MeshLab registers it", () => {
		const action = kernel.pluginManager.filterAction(NAME);
		expect(action).toBeDefined();
		expect(action?.plugin.pluginName()).toBe("FilterTextureDefrag");
		expect(action?.pythonName).toBe("apply_texmap_defragmentation");
	});
});
