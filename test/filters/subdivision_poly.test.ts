/**
 * Catmull–Clark and Doo–Sabin, and the polygon bridge underneath them.
 *
 * Both schemes have exact combinatorics, which is what makes them testable
 * without a reference implementation:
 *
 * - Catmull–Clark makes **one quad per corner**. A mesh of `F` faces with `C`
 *   corners in total gives `C` quads, and `V + E + F` vertices — every original
 *   vertex moves, every edge contributes one, every face contributes one. So a
 *   cube read as six quads gives 24 quads over 8 + 12 + 6 = 26 vertices, and a
 *   triangle mesh gives three quads per triangle. Pure quads from anything is
 *   the scheme's whole selling point.
 * - Doo–Sabin makes **one face per original face, edge and vertex**. A cube
 *   gives 6 + 12 + 8 = 26 faces over 6 × 4 = 24 vertices, one per corner.
 *
 * Both must keep a closed mesh closed, and the shrinking is one-directional:
 * these are approximating schemes, so the surface pulls inside the original and
 * never outside it.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import {
	countBitPolygons,
	countBitQuads,
	countBitTris,
	hasConsistentPerFaceFauxFlag,
} from "../../src/vcg/complex/bit_quad.ts";
import { Clean } from "../../src/vcg/complex/clean.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { PolygonSupport } from "../../src/vcg/complex/polygon_support.ts";
import { UpdateTopology } from "../../src/vcg/complex/update/topology.ts";
import { cube, gridPlane, sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

function docWith(cm: CMeshO, channels: number = MeshElement.MM_NONE) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	if (channels !== MeshElement.MM_NONE) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** A cube whose twelve triangles have been paired into its six real quads. */
function quadCube() {
	const scene = docWith(cube(1).mesh, MeshElement.MM_FACEQUALITY);
	kernel.applyFilter(scene.doc, "Turn into Quad-Dominant mesh", { level: 1 });
	return scene;
}

function meanRadius(cm: CMeshO): number {
	let sum = 0;
	let n = 0;
	for (let v = 0; v < cm.vertSize; v++) {
		if (cm.isVertD(v)) continue;
		sum += Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v));
		n++;
	}
	return sum / n;
}

// -------------------------------------------------------- the polygon bridge

describe("the polygon bridge", () => {
	test("reads a quad mesh back as quads", () => {
		const { cm } = quadCube();
		UpdateTopology.faceFace(cm);
		const polygons = PolygonSupport.extractPolygons(cm);
		expect(polygons.length).toBe(6);
		for (const ring of polygons) expect(ring.length).toBe(4);
	});

	test("reads a plain triangle mesh back as triangles", () => {
		const cm = sphereIcosa(1).mesh;
		UpdateTopology.faceFace(cm);
		const polygons = PolygonSupport.extractPolygons(cm);
		expect(polygons.length).toBe(cm.fn);
		for (const ring of polygons) expect(ring.length).toBe(3);
	});

	test("the rings are ordered, not just sets", () => {
		// Consecutive entries must be joined by a real edge of the mesh; a set
		// would pass a length check and fail this.
		const { cm } = quadCube();
		UpdateTopology.faceFace(cm);
		for (const ring of PolygonSupport.extractPolygons(cm)) {
			for (let i = 0; i < ring.length; i++) {
				const a = ring[i];
				const b = ring[(i + 1) % ring.length];
				const d = Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b));
				// A cube of side 1: adjacent corners are 1 apart, diagonals more.
				expect(d).toBeCloseTo(1, 9);
			}
		}
	});

	test("a round trip through the bridge preserves the polygons", () => {
		const { cm } = quadCube();
		UpdateTopology.faceFace(cm);
		const polygons = PolygonSupport.extractPolygons(cm);
		const positions: number[][] = [];
		for (let v = 0; v < cm.vertSize; v++) positions.push([cm.vx(v), cm.vy(v), cm.vz(v)]);
		const rebuilt = PolygonSupport.meshFromPolygons(positions, polygons);
		UpdateTopology.faceFace(rebuilt);
		expect(countBitPolygons(rebuilt)).toBe(6);
		expect(countBitQuads(rebuilt)).toBe(6);
		expect(hasConsistentPerFaceFauxFlag(rebuilt)).toBe(true);
	});

	test("a pentagon fans into three triangles with two hidden edges", () => {
		const positions = [
			[0, 0, 0],
			[1, 0, 0],
			[1.5, 1, 0],
			[0.5, 1.7, 0],
			[-0.5, 1, 0],
		];
		const mesh = PolygonSupport.meshFromPolygons(positions, [[0, 1, 2, 3, 4]]);
		UpdateTopology.faceFace(mesh);
		expect(mesh.fn).toBe(3);
		expect(countBitPolygons(mesh)).toBe(1);
		expect(hasConsistentPerFaceFauxFlag(mesh)).toBe(true);
	});
});

// ------------------------------------------------------------ Catmull-Clark

describe("Subdivision Surfaces: Catmull-Clark", () => {
	test("a cube gives one quad per corner over V + E + F vertices", () => {
		const { doc, cm } = quadCube();
		const out = kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", {
			Iterations: 1,
		});
		// 6 faces x 4 corners = 24 quads; 8 + 12 + 6 = 26 vertices.
		expect(out.vertex_number).toBe(26);
		UpdateTopology.faceFace(cm);
		expect(countBitQuads(cm)).toBe(24);
		expect(countBitTris(cm)).toBe(0);
	});

	test("a triangle mesh comes back as pure quads", () => {
		// The scheme's whole point: whatever goes in, quads come out.
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 });
		UpdateTopology.faceFace(cm);
		// Three quads per original triangle.
		expect(countBitQuads(cm)).toBe(3 * before.fn);
		expect(countBitTris(cm)).toBe(0);
		// V + E + F, with E = 3V - 6 on a closed triangle mesh.
		expect(cm.vn).toBe(before.vn + (3 * before.vn - 6) + before.fn);
	});

	test("iterating twice is the same as applying it twice", () => {
		const once = quadCube();
		kernel.applyFilter(once.doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 });
		kernel.applyFilter(once.doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 });
		const twice = quadCube();
		kernel.applyFilter(twice.doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 2 });
		expect(twice.cm.vn).toBe(once.cm.vn);
		expect(twice.cm.fn).toBe(once.cm.fn);
	});

	test("stays closed and shrinks toward the surface, never outward", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = meanRadius(cm);
		kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 });
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(true);
		// Approximating, not interpolating: every new point is an average of
		// points on or inside the surface.
		expect(meanRadius(cm)).toBeLessThan(before);
		expect(meanRadius(cm)).toBeGreaterThan(before * 0.9);
	});

	test("the faux tagging stays consistent", () => {
		const { doc, cm } = quadCube();
		kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 2 });
		UpdateTopology.faceFace(cm);
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(true);
	});

	test("an open sheet keeps its boundary a curve of its own", () => {
		// The boundary rule exists so the interior does not drag the border in.
		// A flat square sheet must stay exactly square.
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 });
		let lo = Number.POSITIVE_INFINITY;
		let hi = Number.NEGATIVE_INFINITY;
		for (let v = 0; v < cm.vertSize; v++) {
			if (cm.isVertD(v)) continue;
			lo = Math.min(lo, cm.vx(v));
			hi = Math.max(hi, cm.vx(v));
			// And it stays flat.
			expect(cm.vz(v)).toBeCloseTo(0, 9);
		}
		expect(lo).toBeCloseTo(0, 9);
		expect(hi).toBeCloseTo(1, 9);
	});

	test("refuses inconsistent faux tagging", () => {
		const cm = sphereIcosa(1).mesh;
		UpdateTopology.faceFace(cm);
		// One face claims a hidden edge its neighbour does not.
		cm.faceFlags[0] |= 0x00040000;
		const { doc } = docWith(cm);
		expect(() =>
			kernel.applyFilter(doc, "Subdivision Surfaces: Catmull-Clark", { Iterations: 1 }),
		).toThrow(MLException);
	});
});

// ---------------------------------------------------------------- Doo-Sabin

describe("Subdivision Surfaces: Doo Sabin", () => {
	test("a cube gives one face per original face, edge and vertex", () => {
		const { doc, cm } = quadCube();
		const out = kernel.applyFilter(doc, "Subdivision Surfaces: Doo Sabin");
		// 6 + 12 + 8 = 26 faces, over 6 x 4 = 24 vertices — one per corner.
		expect(out.vertex_number).toBe(24);
		UpdateTopology.faceFace(cm);
		expect(countBitPolygons(cm)).toBe(26);
		// The 6 face copies and 12 edge quads are quads; the 8 corners are
		// triangles, since a cube vertex is 3-valent.
		expect(countBitQuads(cm)).toBe(18);
		expect(countBitTris(cm)).toBe(8);
	});

	test("stays closed", () => {
		const { doc, cm } = quadCube();
		kernel.applyFilter(doc, "Subdivision Surfaces: Doo Sabin");
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(true);
		expect(hasConsistentPerFaceFauxFlag(cm)).toBe(true);
	});

	test("works on a triangle mesh too", () => {
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = { vn: cm.vn, fn: cm.fn };
		kernel.applyFilter(doc, "Subdivision Surfaces: Doo Sabin");
		UpdateTopology.faceFace(cm);
		// One new vertex per corner of the original.
		expect(cm.vn).toBe(3 * before.fn);
		// One face per original face, edge and vertex.
		expect(countBitPolygons(cm)).toBe(before.fn + (3 * before.vn - 6) + before.vn);
		expect(Clean.isWaterTight(cm)).toBe(true);
	});

	test("shrinks each face toward its own centroid", () => {
		// Every new vertex is a weighted average within one face, so the result
		// sits strictly inside the original's convex hull.
		const { doc, cm } = docWith(sphereIcosa(2).mesh);
		const before = meanRadius(cm);
		kernel.applyFilter(doc, "Subdivision Surfaces: Doo Sabin");
		expect(meanRadius(cm)).toBeLessThan(before);
	});

	test("leaves a bordered mesh's boundary alone rather than closing it", () => {
		// A border has no second face to build the connecting quad from, and
		// inventing one would fill a hole the mesh genuinely has.
		const { doc, cm } = docWith(gridPlane(4, 4).mesh);
		kernel.applyFilter(doc, "Subdivision Surfaces: Doo Sabin");
		UpdateTopology.faceFace(cm);
		expect(Clean.isWaterTight(cm)).toBe(false);
		expect(Clean.countHoles(cm)).toBeGreaterThan(0);
	});
});

// -------------------------------------------------------------- registration

describe("registration", () => {
	test("only the two heaviest filter_meshing entries are still pending", () => {
		const pending = kernel
			.filterList()
			.filter((f) => f.plugin.pluginName() === "FilterMeshing" && !f.implemented)
			.map((f) => f.name)
			.sort();
		expect(pending).toEqual([
			"Simplification: Quadric Edge Collapse Decimation (with texture)",
			"Tri to Quad by smart triangle pairing",
		]);
	});
});
