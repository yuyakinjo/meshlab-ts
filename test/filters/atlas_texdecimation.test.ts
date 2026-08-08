/**
 * The Voronoi atlas and texture-aware decimation.
 *
 * Both are about a parametrisation staying *valid* rather than staying
 * beautiful, and the tests reflect that. A valid atlas covers every face with a
 * non-degenerate UV triangle inside the unit square; a valid decimation leaves
 * the UVs finite, in range, and never merges two vertices that disagreed about
 * where they were in the texture.
 *
 * Two of the sharper checks are about what would look right and be wrong. A
 * chart that failed to flatten and kept whatever was in the channel would give
 * every one of its faces the same zeroed UV — plausible-looking numbers that
 * collapse a whole region onto a single texel. And a collapse across a seam
 * would give a small geometric error and a parametrisation that contradicts
 * itself. Both are tested for directly.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO, channels: number = MeshElement.MM_NONE) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	if (channels !== MeshElement.MM_NONE) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** Every face's UV triangle, as area and bounding range. */
function uvStats(cm: CMeshO) {
	const wt = cm.wedgeTexCoord as Float64Array;
	let degenerate = 0;
	let outside = 0;
	let nonFinite = 0;
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		const u = [0, 1, 2].map((k) => wt[6 * f + 2 * k]);
		const v = [0, 1, 2].map((k) => wt[6 * f + 2 * k + 1]);
		if ([...u, ...v].some((x) => !Number.isFinite(x))) {
			nonFinite++;
			continue;
		}
		if ([...u, ...v].some((x) => x < -1e-9 || x > 1 + 1e-9)) outside++;
		const area = Math.abs((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0])) / 2;
		if (area < 1e-12) degenerate++;
	}
	return { degenerate, outside, nonFinite };
}

// ------------------------------------------------------------ Voronoi atlas

describe("Parametrization: Voronoi Atlas", () => {
	test("covers every face with a real UV triangle inside the unit square", () => {
		for (const regionNum of [4, 12, 30]) {
			const { doc, cm } = docWith(sphereIcosa(3).mesh);
			kernel.applyFilter(doc, "Parametrization: Voronoi Atlas", { regionNum });
			const stats = uvStats(cm);
			// The one that matters: a failed chart left with zeroed UVs would
			// show up here as a whole region of degenerate triangles.
			expect(stats.degenerate, `regionNum ${regionNum}`).toBe(0);
			expect(stats.nonFinite, `regionNum ${regionNum}`).toBe(0);
			expect(stats.outside, `regionNum ${regionNum}`).toBe(0);
		}
	});

	test("the charts do not overlap each other", () => {
		// Each chart gets its own grid cell, so two faces from different charts
		// can never claim the same texel. Checking the cell each face falls in
		// is enough: a face straddling two cells would mean the packing leaked.
		const { doc, cm } = docWith(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Parametrization: Voronoi Atlas", { regionNum: 12 });
		const cells = (out.region_number as number) + (out.failed_regions as number);
		const columns = Math.ceil(Math.sqrt(cells));
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			const cellsHit = new Set<string>();
			for (let k = 0; k < 3; k++) {
				const cx = Math.min(columns - 1, Math.floor(wt[6 * f + 2 * k] * columns));
				const cy = Math.min(columns - 1, Math.floor(wt[6 * f + 2 * k + 1] * columns));
				cellsHit.add(`${cx},${cy}`);
			}
			expect(cellsHit.size).toBe(1);
		}
	});

	test("more regions asked for means more charts", () => {
		const charts = (regionNum: number) => {
			const { doc } = docWith(sphereIcosa(3).mesh);
			return kernel.applyFilter(doc, "Parametrization: Voronoi Atlas", { regionNum })
				.region_number as number;
		};
		expect(charts(30)).toBeGreaterThan(charts(6));
	});

	test("reports the faces that fell back to a per-triangle layout", () => {
		// Not every Voronoi region is a disk, and the filter has to say how much
		// of the mesh it could not give a continuous chart.
		const { doc } = docWith(sphereIcosa(3).mesh);
		const out = kernel.applyFilter(doc, "Parametrization: Voronoi Atlas", { regionNum: 8 });
		expect(typeof out.failed_regions).toBe("number");
		expect(typeof out.failed_faces).toBe("number");
		if ((out.failed_regions as number) > 0) {
			expect(out.failed_faces as number).toBeGreaterThan(0);
		}
	});

	test("refuses a non-manifold mesh", () => {
		const cm = new CMeshO();
		Allocator.addVertices(cm, 5);
		cm.setVert(0, 0, 0, 0);
		cm.setVert(1, 1, 0, 0);
		cm.setVert(2, 0, 1, 0);
		cm.setVert(3, 0, 0, 1);
		cm.setVert(4, 0, -1, 0);
		Allocator.addFaces(cm, 3);
		cm.setFace(0, 0, 1, 2);
		cm.setFace(1, 0, 1, 3);
		cm.setFace(2, 0, 1, 4);
		const { doc } = docWith(cm);
		expect(() => kernel.applyFilter(doc, "Parametrization: Voronoi Atlas")).toThrow(MLException);
	});

	test("says so rather than pretending, when asked for overlapping charts", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Parametrization: Voronoi Atlas", { overlapFlag: true }),
		).toThrow();
	});
});

// -------------------------------------------------- textured decimation

describe("Simplification: Quadric Edge Collapse Decimation (with texture)", () => {
	/** A sphere with a spherical parametrisation, seam included. */
	function texturedSphere(subdiv = 4) {
		const scene = docWith(sphereIcosa(subdiv).mesh, MeshElement.MM_WEDGTEXCOORD);
		const wt = scene.cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < scene.cm.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				const v = scene.cm.fv(f, k);
				wt[6 * f + 2 * k] = Math.atan2(scene.cm.vz(v), scene.cm.vx(v)) / (2 * Math.PI) + 0.5;
				wt[6 * f + 2 * k + 1] = Math.acos(Math.max(-1, Math.min(1, scene.cm.vy(v)))) / Math.PI;
			}
		}
		return scene;
	}

	test("reaches the target and leaves the UVs finite and in range", () => {
		const { doc, cm } = texturedSphere();
		const before = cm.fn;
		const out = kernel.applyFilter(
			doc,
			"Simplification: Quadric Edge Collapse Decimation (with texture)",
			{ TargetPerc: 0.3 },
		);
		expect(out.target_reached).toBe(true);
		expect(cm.fn).toBeLessThan(before);
		const stats = uvStats(cm);
		expect(stats.nonFinite).toBe(0);
		expect(stats.outside).toBe(0);
	});

	test("keeps the mesh closed", () => {
		const { doc, cm } = texturedSphere(3);
		kernel.applyFilter(doc, "Simplification: Quadric Edge Collapse Decimation (with texture)", {
			TargetPerc: 0.4,
		});
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(true);
	});

	test("never merges two vertices that disagreed about their UV", () => {
		// The invariant that distinguishes this from plain decimation: after it
		// runs, no vertex carries two different UVs. A collapse across a seam
		// would produce exactly that.
		const { doc, cm } = texturedSphere(3);
		kernel.applyFilter(doc, "Simplification: Quadric Edge Collapse Decimation (with texture)", {
			TargetPerc: 0.4,
		});
		const wt = cm.wedgeTexCoord as Float64Array;
		const seen = new Map<number, Set<string>>();
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			for (let k = 0; k < 3; k++) {
				const v = cm.fv(f, k);
				if (!seen.has(v)) seen.set(v, new Set());
				(seen.get(v) as Set<string>).add(
					`${wt[6 * f + 2 * k].toFixed(9)},${wt[6 * f + 2 * k + 1].toFixed(9)}`,
				);
			}
		}
		// Every vertex has exactly one UV: the seam vertices were never merged
		// away, and every merged pair agreed.
		for (const [v, uvs] of seen) expect(uvs.size, `vertex ${v}`).toBe(1);
	});

	test("refuses a mesh with no parametrization", () => {
		const { doc } = docWith(sphereIcosa(2).mesh);
		expect(() =>
			kernel.applyFilter(doc, "Simplification: Quadric Edge Collapse Decimation (with texture)", {
				TargetPerc: 0.5,
			}),
		).toThrow(MLException);
	});

	test("decimates less than the plain filter, because it will not cross seams", () => {
		// The cost of the guarantee, stated rather than hidden: a seam is a wall
		// the collapse cannot pass through, so the same target is harder to hit.
		const textured = texturedSphere(3);
		const plain = texturedSphere(3);
		kernel.applyFilter(
			textured.doc,
			"Simplification: Quadric Edge Collapse Decimation (with texture)",
			{ TargetPerc: 0.1 },
		);
		kernel.applyFilter(plain.doc, "Simplification: Quadric Edge Collapse Decimation", {
			TargetPerc: 0.1,
		});
		expect(textured.cm.fn).toBeGreaterThanOrEqual(plain.cm.fn);
	});
});
