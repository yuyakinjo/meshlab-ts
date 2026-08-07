/**
 * The second half of filter_measure: the selection's area and perimeter, the
 * quad-mesh measures, and the quality histograms.
 *
 * Every expected number here has a closed form. A unit-square grid has an area
 * equal to its cell count and a perimeter equal to its boundary length; a quad
 * built from right angles has zero angle discrepancy and a side ratio of one.
 * Where a value is only approachable — a percentile, an area-weighted bin —
 * the test states the identity it must satisfy rather than the number.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import {
	countBitLargePolygons,
	countBitPolygons,
	countBitQuads,
	countBitTris,
	hasConsistentPerFaceFauxFlag,
	isBitTriQuadOnly,
} from "../../src/vcg/complex/bit_quad.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { FaceFlag } from "../../src/vcg/complex/flags.ts";
import { Distribution, Histogram } from "../../src/vcg/math/histogram.ts";
import { buildMesh, cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function scene(cm: CMeshO, channels = 0) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "test", true, cm);
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/**
 * A grid of unit squares, each stored as two triangles sharing a faux
 * diagonal — a bit-quad mesh in the VCGLib sense.
 */
function quadGrid(nu: number, nv: number): CMeshO {
	const coords: number[] = [];
	for (let j = 0; j <= nv; j++) {
		for (let i = 0; i <= nu; i++) coords.push(i, j, 0);
	}
	const faces: number[] = [];
	const at = (i: number, j: number) => j * (nu + 1) + i;
	for (let j = 0; j < nv; j++) {
		for (let i = 0; i < nu; i++) {
			const a = at(i, j);
			const b = at(i + 1, j);
			const c = at(i + 1, j + 1);
			const d = at(i, j + 1);
			faces.push(a, b, c, a, c, d);
		}
	}
	const m = buildMesh(coords, faces);
	// The shared diagonal is a→c: edge 2 of the first triangle (c,a) and
	// edge 0 of the second (a,c).
	for (let f = 0; f < m.faceSize; f += 2) {
		m.faceFlags[f] |= FaceFlag.FAUX2;
		m.faceFlags[f + 1] |= FaceFlag.FAUX0;
	}
	return m;
}

describe("Compute Area/Perimeter of selection", () => {
	const NAME = "Compute Area/Perimeter of selection";

	test("a whole unit-square grid gives its area and its boundary", () => {
		const { doc, cm } = scene(quadGrid(3, 2));
		for (let f = 0; f < cm.faceSize; f++) cm.faceFlags[f] |= FaceFlag.SELECTED;

		const out = kernel.applyFilter(doc, NAME);
		expect(out.seleced_triangles_number).toBe(12);
		expect(out.selected_surface_area as number).toBeCloseTo(6, 10);
		// The 3x2 rectangle's boundary is 2*(3+2) unit edges.
		expect(out.border_edge_number).toBe(10);
		expect(out.perimeter as number).toBeCloseTo(10, 10);
	});

	test("one cell out of the middle has a perimeter of four unit edges", () => {
		const { doc, cm } = scene(quadGrid(3, 3));
		// The two triangles of the quad at column 1, row 1.
		const first = 2 * (1 * 3 + 1);
		cm.faceFlags[first] |= FaceFlag.SELECTED;
		cm.faceFlags[first + 1] |= FaceFlag.SELECTED;

		const out = kernel.applyFilter(doc, NAME);
		expect(out.selected_surface_area as number).toBeCloseTo(1, 10);
		expect(out.border_edge_number).toBe(4);
		expect(out.perimeter as number).toBeCloseTo(4, 10);
		// The shared diagonal is interior to the selection and must not count.
	});

	test("a single triangle's perimeter is all three of its edges", () => {
		const { doc, cm } = scene(quadGrid(2, 2));
		cm.faceFlags[0] |= FaceFlag.SELECTED;
		const out = kernel.applyFilter(doc, NAME);
		expect(out.selected_surface_area as number).toBeCloseTo(0.5, 10);
		expect(out.border_edge_number).toBe(3);
		expect(out.perimeter as number).toBeCloseTo(2 + Math.SQRT2, 10);
	});

	test("no selection is an error, not a zero", () => {
		const { doc } = scene(cube().mesh);
		expect(() => kernel.applyFilter(doc, NAME)).toThrow(/no face selection/);
	});
});

describe("bit-quad counting", () => {
	test("a quad grid counts as quads, not as triangles", () => {
		const m = quadGrid(4, 3);
		expect(countBitQuads(m)).toBe(12);
		expect(countBitTris(m)).toBe(0);
		expect(countBitPolygons(m)).toBe(12);
		expect(countBitLargePolygons(m)).toBe(12);
		expect(isBitTriQuadOnly(m)).toBe(true);
	});

	test("a mesh with no faux edges is all triangles", () => {
		const m = gridPlane(4, 4).mesh;
		expect(countBitQuads(m)).toBe(0);
		expect(countBitTris(m)).toBe(m.fn);
		expect(countBitPolygons(m)).toBe(m.fn);
	});

	test("faux bits must be reciprocated", () => {
		const { m, cm } = scene(quadGrid(2, 2));
		m.updateDataMask(MeshElement.MM_FACEFACETOPO);
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(true);

		// Clear one side of a shared diagonal and the tagging is broken.
		cm.faceFlags[0] &= ~FaceFlag.FAUX2;
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(false);
	});

	test("a faux border edge is inconsistent", () => {
		const { m, cm } = scene(quadGrid(1, 1));
		m.updateDataMask(MeshElement.MM_FACEFACETOPO);
		// Edge 0 of face 0 is on the boundary of the grid.
		cm.faceFlags[0] |= FaceFlag.FAUX0;
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(false);
	});
});

describe("Compute Topological Measures for Quad Meshes", () => {
	const NAME = "Compute Topological Measures for Quad Meshes";

	test("a grid of squares is perfectly rectangular", () => {
		const { doc } = scene(quadGrid(4, 3));
		const out = kernel.applyFilter(doc, NAME);

		expect(out.quads_number).toBe(12);
		expect(out.triangles_number).toBe(0);
		expect(out.polys_number).toBe(12);
		expect(out.large_polys_number).toBe(12);
		// Right angles everywhere, and all four sides the same length.
		expect(out.right_angle_discrepancy_max as number).toBeCloseTo(0, 10);
		expect(out.right_angle_discrepancy_avg as number).toBeCloseTo(0, 10);
		expect(out.right_angle_discrepancy_stddev as number).toBeCloseTo(0, 10);
		expect(out.quad_ratio_min as number).toBeCloseTo(1, 10);
		expect(out.quad_ratio_max as number).toBeCloseTo(1, 10);
	});

	test("stretching the cells changes the ratio but not the angles", () => {
		const cm = quadGrid(3, 3);
		for (let v = 0; v < cm.vertSize; v++) cm.vertCoord[3 * v] *= 4; // x only
		const { doc } = scene(cm);
		const out = kernel.applyFilter(doc, NAME);

		expect(out.right_angle_discrepancy_max as number).toBeCloseTo(0, 10);
		expect(out.quad_ratio_avg as number).toBeCloseTo(0.25, 10);
	});

	test("shearing a cell shows up as an angle discrepancy", () => {
		const cm = quadGrid(3, 3);
		// Slide every row sideways in proportion to its height: a shear, so
		// the sides stay straight but the corners stop being right angles.
		for (let v = 0; v < cm.vertSize; v++) cm.vertCoord[3 * v] += 0.5 * cm.vertCoord[3 * v + 1];
		const { doc } = scene(cm);
		const out = kernel.applyFilter(doc, NAME);

		// atan(0.5) is about 26.57 degrees away from square, both ways.
		expect(out.right_angle_discrepancy_avg as number).toBeCloseTo(
			(Math.atan(0.5) * 180) / Math.PI,
			6,
		);
	});

	test("a plain triangle mesh has no quads to measure", () => {
		const { doc } = scene(gridPlane(3, 3).mesh);
		expect(() => kernel.applyFilter(doc, NAME)).toThrow(/doesn't contain quads/);
	});

	test("inconsistent faux tagging is refused before anything is counted", () => {
		const cm = quadGrid(2, 2);
		cm.faceFlags[0] &= ~FaceFlag.FAUX2;
		const { doc } = scene(cm);
		expect(() => kernel.applyFilter(doc, NAME)).toThrow(/FauxEdge tagging/);
	});
});

describe("Per Vertex Quality Histogram", () => {
	const NAME = "Per Vertex Quality Histogram";

	test("every vertex lands in exactly one bin", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vz(v);

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: -1,
			HistMax: 1,
			binNum: 10,
			areaWeighted: false,
		});
		const counts = out.hist_count as number[];
		// Ten bins plus the two overflow ones.
		expect(counts).toHaveLength(12);
		expect(counts.reduce((a, b) => a + b, 0)).toBe(cm.vn);
	});

	test("values outside the range go to the overflow bins, not the nearest one", () => {
		const { doc, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v < 2 ? -100 : 100;

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: 0,
			HistMax: 1,
			binNum: 4,
			areaWeighted: false,
		});
		const counts = out.hist_count as number[];
		expect(counts[0]).toBe(2);
		expect(counts[5]).toBe(cm.vn - 2);
		expect(counts.slice(1, 5)).toEqual([0, 0, 0, 0]);
	});

	test("the outer bounds are infinite, and the inner ones tile the range", () => {
		const { doc, cm } = scene(gridPlane(2, 2).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 0.5;

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: 0,
			HistMax: 2,
			binNum: 4,
			areaWeighted: false,
		});
		const lower = out.hist_bin_min as number[];
		const upper = out.hist_bin_max as number[];
		expect(lower[0]).toBe(Number.NEGATIVE_INFINITY);
		expect(upper[5]).toBe(Number.POSITIVE_INFINITY);
		expect(lower.slice(1, 5)).toEqual([0, 0.5, 1, 1.5]);
		expect(upper.slice(1, 5)).toEqual([0.5, 1, 1.5, 2]);
	});

	test("area weighting makes the bins sum to the surface area", () => {
		const { doc, cm } = scene(sphereIcosa(3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vz(v);
		const area = kernel.applyFilter(doc, "Compute Geometric Measures").surface_area as number;

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: -1,
			HistMax: 1,
			binNum: 8,
			areaWeighted: true,
		});
		const total = (out.hist_count as number[]).reduce((a, b) => a + b, 0);
		expect(total).toBeCloseTo(area, 8);
	});

	test("the default range covers the mesh's own quality span", () => {
		const { m, cm } = scene(gridPlane(3, 3).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = v;
		const action = kernel.pluginManager.filterAction(NAME);
		const params = action?.plugin.initParameterList(action.id, m);

		expect(params?.getFloat("HistMin")).toBe(0);
		expect(params?.getFloat("HistMax")).toBe(cm.vn - 1);
	});

	test("the filter declares that it needs per-vertex quality", () => {
		const action = kernel.pluginManager.filterAction(NAME);
		const needed = action?.plugin.getPreConditions(action.id) ?? 0;
		expect(needed & MeshElement.MM_VERTQUALITY).toBe(MeshElement.MM_VERTQUALITY);
	});

	test("a bin count below one is refused", () => {
		const { doc, cm } = scene(gridPlane(2, 2).mesh, MeshElement.MM_VERTQUALITY);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = 1;
		expect(() =>
			kernel.applyFilter(doc, NAME, {
				HistMin: 0,
				HistMax: 2,
				binNum: 0,
				areaWeighted: false,
			}),
		).toThrow(/at least 1/);
	});
});

describe("Per Face Quality Histogram", () => {
	const NAME = "Per Face Quality Histogram";

	test("every face lands in exactly one bin", () => {
		const { doc, m, cm } = scene(sphereIcosa(2).mesh, MeshElement.MM_FACEQUALITY);
		const q = m.cm.faceQuality as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) q[f] = f % 7;

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: 0,
			HistMax: 7,
			binNum: 7,
			areaWeighted: false,
		});
		const counts = out.hist_count as number[];
		expect(counts).toHaveLength(9);
		expect(counts.reduce((a, b) => a + b, 0)).toBe(cm.fn);
		expect(counts[0]).toBe(0);
		expect(counts[8]).toBe(0);
	});

	test("area weighting sums to the surface area", () => {
		const { doc, m, cm } = scene(sphereIcosa(2).mesh, MeshElement.MM_FACEQUALITY);
		const q = m.cm.faceQuality as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) q[f] = f % 5;
		const area = kernel.applyFilter(doc, "Compute Geometric Measures").surface_area as number;

		const out = kernel.applyFilter(doc, NAME, {
			HistMin: 0,
			HistMax: 5,
			binNum: 5,
			areaWeighted: true,
		});
		const total = (out.hist_count as number[]).reduce((a, b) => a + b, 0);
		expect(total).toBeCloseTo(area, 8);
	});
});

describe("Distribution and Histogram", () => {
	test("moments match their closed forms on 1..n", () => {
		const d = new Distribution();
		for (let i = 1; i <= 100; i++) d.Add(i);
		expect(d.Cnt()).toBe(100);
		expect(d.Sum()).toBe(5050);
		expect(d.Avg()).toBe(50.5);
		expect(d.Min()).toBe(1);
		expect(d.Max()).toBe(100);
		// The population variance of 1..n is (n²-1)/12.
		expect(d.Variance()).toBeCloseTo((100 * 100 - 1) / 12, 6);
		expect(d.StandardDeviation()).toBeCloseTo(Math.sqrt((100 * 100 - 1) / 12), 6);
	});

	test("percentiles are exact, not binned", () => {
		const d = new Distribution();
		for (let i = 1; i <= 100; i++) d.Add(i);
		expect(d.Percentile(0)).toBe(1);
		expect(d.Percentile(0.5)).toBe(51);
		expect(d.Percentile(0.99)).toBe(100);
		expect(d.Percentile(1)).toBe(100);
	});

	test("an empty distribution answers zero rather than a NaN", () => {
		const d = new Distribution();
		expect(d.Avg()).toBe(0);
		expect(d.Min()).toBe(0);
		expect(d.Max()).toBe(0);
		expect(d.Variance()).toBe(0);
		expect(d.Percentile(0.5)).toBe(0);
	});

	test("a value exactly on a bin boundary goes to the upper bin", () => {
		const h = new Histogram();
		h.SetRange(0, 4, 4);
		h.Add(1);
		expect(h.BinCountInd(1)).toBe(0);
		expect(h.BinCountInd(2)).toBe(1);
	});

	test("the maximum itself overflows, which is what makes the bins half-open", () => {
		const h = new Histogram();
		h.SetRange(0, 1, 2);
		h.Add(1);
		expect(h.BinCountInd(3)).toBe(1);
	});

	test("a histogram with no bins is refused", () => {
		expect(() => new Histogram().SetRange(0, 1, 0)).toThrow(/at least one bin/);
	});
});

describe("registry", () => {
	test("all four are implemented and in the Measure class", () => {
		for (const name of [
			"Compute Area/Perimeter of selection",
			"Compute Topological Measures for Quad Meshes",
			"Per Vertex Quality Histogram",
			"Per Face Quality Histogram",
		]) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			expect(action?.plugin.pluginName(), name).toBe("FilterMeasure");
		}
	});
});
