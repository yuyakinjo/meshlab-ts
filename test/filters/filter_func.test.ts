/**
 * `filter_func` — the expression-driven filters.
 *
 * The evaluator is tested on its own; what these check is the wiring: that
 * each variable carries the value its name promises, that the right channel
 * gets written, and that the two mesh generators produce the surface their
 * expression describes.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { filterClassToString } from "../../src/common/plugins/filter_class.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import {
	EDGE_VARIABLES,
	FACE_VARIABLES,
	VERTEX_VARIABLES,
} from "../../src/meshlabplugins/filter_func/filter_func.ts";
import type { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { Platonic } from "../../src/vcg/complex/create/platonic.ts";
import { blue, green, red, rgba } from "../../src/vcg/space/color4.ts";
import { assertAllocatorConsistent, computeFacts, signedVolume } from "../helpers/invariants.ts";

const kernel = MeshLabKernel.default();

function scene(channels = 0, subdiv = 2) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, Platonic.sphere(subdiv));
	if (channels !== 0) m.updateDataMask(channels);
	m.updateBoxAndNormals();
	return { doc, m, cm: m.cm };
}

/** Applies a creation filter and hands back the layer it made. */
function created(name: string, params: Record<string, unknown>): CMeshO {
	const doc = new MeshDocument();
	const out = kernel.applyFilter(doc, name, params);
	return doc.requireMesh(out.new_mesh_id as number).cm;
}

describe("the variable sets", () => {
	test("are upstream's, name for name", () => {
		// Every published MeshLab recipe is written against these, so a rename
		// or an omission silently breaks scripts rather than failing loudly.
		expect(VERTEX_VARIABLES).toEqual([
			"x",
			"y",
			"z",
			"nx",
			"ny",
			"nz",
			"r",
			"g",
			"b",
			"a",
			"q",
			"vi",
			"vtu",
			"vtv",
			"ti",
			"vsel",
			"xmin",
			"ymin",
			"zmin",
			"xmax",
			"ymax",
			"zmax",
			"bbdiag",
			"xdim",
			"ydim",
			"zdim",
			"xmid",
			"ymid",
			"zmid",
		]);
		expect(FACE_VARIABLES).toHaveLength(69);
		expect(FACE_VARIABLES.slice(0, 9)).toEqual([
			"x0",
			"y0",
			"z0",
			"x1",
			"y1",
			"z1",
			"x2",
			"y2",
			"z2",
		]);
		expect(FACE_VARIABLES).toContain("fsel");
		expect(FACE_VARIABLES).toContain("fnz");
		expect(EDGE_VARIABLES).toHaveLength(20);
	});

	test("every per-vertex variable is readable, and none throws", () => {
		const { doc } = scene(MeshElement.MM_VERTCOLOR | MeshElement.MM_VERTQUALITY);
		for (const name of VERTEX_VARIABLES) {
			expect(
				() =>
					kernel.applyFilter(doc, "Conditional Vertex Selection", {
						condSelect: `${name} == ${name}`,
					}),
				name,
			).not.toThrow();
		}
	});

	test("every per-face variable is readable", () => {
		const { doc } = scene(MeshElement.MM_FACECOLOR | MeshElement.MM_FACEQUALITY);
		for (const name of FACE_VARIABLES) {
			expect(
				() =>
					kernel.applyFilter(doc, "Conditional Face Selection", {
						condSelect: `${name} == ${name}`,
					}),
				name,
			).not.toThrow();
		}
	});

	test("the bounding-box globals describe the mesh", () => {
		// A unit sphere: the box runs -1..1 on every axis.
		const { doc, cm } = scene();
		const count = (condition: string) =>
			kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: condition })
				.selected as number;
		expect(count("xmin == -1 && xmax == 1")).toBe(cm.vn);
		expect(count("xmid == 0 && ymid == 0 && zmid == 0")).toBe(cm.vn);
		expect(count("xdim == 2 && ydim == 2 && zdim == 2")).toBe(cm.vn);
		expect(count("bbdiag > 3.46 && bbdiag < 3.47")).toBe(cm.vn);
	});

	test("vi is the vertex index, fi the face index", () => {
		const { doc } = scene();
		expect(
			kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "vi < 10" }).selected,
		).toBe(10);
		expect(
			kernel.applyFilter(doc, "Conditional Face Selection", { condSelect: "fi < 7" }).selected,
		).toBe(7);
	});
});

describe("conditional selection", () => {
	test("selects exactly what the condition names", () => {
		const { doc, cm } = scene();
		const out = kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "z > 0" });
		let expected = 0;
		for (let v = 0; v < cm.vn; v++) if (cm.vz(v) > 0) expected++;
		expect(out.selected).toBe(expected);
		for (let v = 0; v < cm.vn; v++) expect(cm.isVertS(v), `v${v}`).toBe(cm.vz(v) > 0);
	});

	test("replaces the previous selection rather than adding to it", () => {
		// A selection filter that only ever grew the set would make a sequence
		// of conditions useless.
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "1" });
		expect(
			kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "z > 0.9" })
				.selected as number,
		).toBeLessThan(cm.vn);
	});

	test("vsel reads the selection back", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "z > 0" });
		const again = kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "vsel" });
		let expected = 0;
		for (let v = 0; v < cm.vn; v++) if (cm.vz(v) > 0) expected++;
		expect(again.selected).toBe(expected);
	});

	test("a bad expression is reported, not ignored", () => {
		const { doc } = scene();
		expect(() =>
			kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "z >" }),
		).toThrow(MLException);
		expect(() =>
			kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "fi == 0" }),
		).toThrow(MLException);
	});
});

describe("the per-element function filters", () => {
	test("the geometric function moves every vertex", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Per Vertex Geometric Function", { x: "x*2", y: "y", z: "z" });
		expect(cm.bbox.min[0]).toBeCloseTo(-2, 9);
		expect(cm.bbox.max[0]).toBeCloseTo(2, 9);
		expect(cm.bbox.max[1]).toBeCloseTo(1, 9);
	});

	test("all three coordinates are read from the old position", () => {
		// A swap has to actually swap; computing x then reading it back for y
		// would give y = the new x.
		const { doc, cm } = scene(0, 1);
		const before = Array.from({ length: cm.vn }, (_, v) => [cm.vx(v), cm.vy(v), cm.vz(v)]);
		kernel.applyFilter(doc, "Per Vertex Geometric Function", { x: "y", y: "x", z: "z" });
		for (let v = 0; v < cm.vn; v++) {
			expect(cm.vx(v), `v${v}`).toBe(before[v][1]);
			expect(cm.vy(v), `v${v}`).toBe(before[v][0]);
		}
	});

	test("only-on-selection leaves the rest where it was", () => {
		const { doc, cm } = scene();
		kernel.applyFilter(doc, "Conditional Vertex Selection", { condSelect: "z > 0" });
		kernel.applyFilter(doc, "Per Vertex Geometric Function", {
			x: "x",
			y: "y",
			z: "z + 10",
			onselected: true,
		});
		for (let v = 0; v < cm.vn; v++) {
			// The moved ones are the ones that were above the equator.
			expect(cm.vz(v) > 5, `v${v}`).toBe(cm.isVertS(v));
		}
	});

	test("the normal functions write the normal channel", () => {
		const { doc, cm } = scene();
		const before = Array.from({ length: cm.vn }, (_, v) => cm.vertNormal[3 * v]);
		kernel.applyFilter(doc, "Per Vertex Normal Function", { x: "-nx", y: "-ny", z: "-nz" });
		for (let v = 0; v < cm.vn; v++) expect(cm.vertNormal[3 * v], `v${v}`).toBe(-before[v]);

		kernel.applyFilter(doc, "Per Face Normal Function", { x: "1", y: "0", z: "0" });
		for (let f = 0; f < cm.fn; f++) expect(cm.faceNormal[3 * f], `f${f}`).toBe(1);
	});

	test("the colour functions write each channel from its own expression", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		// Upstream names the vertex channels x, y, z — not r, g, b.
		kernel.applyFilter(doc, "Per Vertex Color Function", {
			x: "255*(z>0)",
			y: "0",
			z: "128",
			a: "255",
		});
		for (let v = 0; v < cm.vn; v++) {
			expect(red(cm.vertColor[v]), `v${v}`).toBe(cm.vz(v) > 0 ? 255 : 0);
			expect(green(cm.vertColor[v]), `v${v}`).toBe(0);
			expect(blue(cm.vertColor[v]), `v${v}`).toBe(128);
		}
	});

	test("the face colour function names its channels r, g and b", () => {
		const { doc, cm } = scene(MeshElement.MM_FACECOLOR);
		kernel.applyFilter(doc, "Per Face Color Function", { r: "10", g: "20", b: "30", a: "255" });
		for (let f = 0; f < cm.fn; f++) expect(red(cm.faceColor?.[f] ?? 0), `f${f}`).toBe(10);
		expect(green(cm.faceColor?.[0] ?? 0)).toBe(20);
		expect(blue(cm.faceColor?.[0] ?? 0)).toBe(30);
	});

	test("out-of-range channels clamp rather than wrapping", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Vertex Color Function", {
			x: "1000",
			y: "-1000",
			z: "0",
			a: "255",
		});
		expect(red(cm.vertColor[0])).toBe(255);
		expect(green(cm.vertColor[0])).toBe(0);
	});

	test("the quality functions report the range they produced", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY);
		const out = kernel.applyFilter(doc, "Per Vertex Quality Function", { q: "z" });
		expect(out.min_value).toBeCloseTo(-1, 9);
		expect(out.max_value).toBeCloseTo(1, 9);
		for (let v = 0; v < cm.vn; v++) expect(cm.vertQuality[v], `v${v}`).toBe(cm.vz(v));
	});

	test("normalize rescales into 0..1", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY);
		const out = kernel.applyFilter(doc, "Per Vertex Quality Function", {
			q: "z",
			normalize: true,
		});
		expect(out.min_value).toBe(0);
		expect(out.max_value).toBe(1);
		for (let v = 0; v < cm.vn; v++) {
			expect(cm.vertQuality[v]).toBeGreaterThanOrEqual(0);
			expect(cm.vertQuality[v]).toBeLessThanOrEqual(1);
		}
	});

	test("map paints the ramp, low red and high blue", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY | MeshElement.MM_VERTCOLOR);
		kernel.applyFilter(doc, "Per Vertex Quality Function", { q: "z", map: true });
		let lowest = 0;
		let highest = 0;
		for (let v = 0; v < cm.vn; v++) {
			if (cm.vz(v) < cm.vz(lowest)) lowest = v;
			if (cm.vz(v) > cm.vz(highest)) highest = v;
		}
		expect(cm.vertColor[lowest]).toBe(rgba(255, 0, 0));
		expect(cm.vertColor[highest]).toBe(rgba(0, 0, 255));
	});

	test("the face quality function sees its three corners", () => {
		const { doc, cm } = scene(MeshElement.MM_FACEQUALITY);
		kernel.applyFilter(doc, "Per Face Quality Function", { q: "(z0+z1+z2)/3" });
		for (let f = 0; f < cm.fn; f++) {
			const want = (cm.vz(cm.fv(f, 0)) + cm.vz(cm.fv(f, 1)) + cm.vz(cm.fv(f, 2))) / 3;
			expect(cm.faceQuality?.[f], `f${f}`).toBeCloseTo(want, 12);
		}
	});
});

describe("Grid Generator", () => {
	test("lays out the requested lattice", () => {
		const m = created("Grid Generator", {
			numVertX: 4,
			numVertY: 3,
			absScaleX: 1,
			absScaleY: 1,
		});
		expect(m.vn).toBe(12);
		// (4-1) x (3-1) quads, two triangles each.
		expect(m.fn).toBe(12);
		expect(m.bbox.min).toEqual([0, 0, 0]);
		expect(m.bbox.max).toEqual([3, 2, 0]);
		assertAllocatorConsistent(m);
	});

	test("centring puts the middle on the origin", () => {
		const m = created("Grid Generator", {
			numVertX: 4,
			numVertY: 3,
			absScaleX: 1,
			absScaleY: 1,
			center: true,
		});
		expect(m.bbox.min).toEqual([-1.5, -1, 0]);
		expect(m.bbox.max).toEqual([1.5, 1, 0]);
	});

	test("is a single manifold sheet with one boundary", () => {
		const facts = computeFacts(created("Grid Generator", { numVertX: 6, numVertY: 5 }));
		expect(facts.components).toBe(1);
		expect(facts.nonManifoldEdges).toBe(0);
		expect(facts.boundaryLoops).toBe(1);
		expect(facts.watertight).toBe(false);
	});

	test("refuses a lattice with no extent", () => {
		for (const params of [{ numVertX: 0 }, { numVertY: -3 }]) {
			expect(() => created("Grid Generator", params), JSON.stringify(params)).toThrow(MLException);
		}
	});

	test("a single row is still valid, just faceless", () => {
		const m = created("Grid Generator", { numVertX: 1, numVertY: 5 });
		expect(m.vn).toBe(5);
		expect(m.fn).toBe(0);
	});
});

describe("Implicit Surface", () => {
	test("extracts the sphere its expression describes", () => {
		// x^2 + y^2 + z^2 = 0.5, so radius sqrt(0.5).
		const m = created("Implicit Surface", { voxelSize: 0.05, expr: "x*x+y*y+z*z-0.5" });
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(0);
		expect(facts.nonManifoldEdges).toBe(0);

		const radius = Math.sqrt(0.5);
		for (let v = 0; v < m.vn; v++) {
			expect(Math.hypot(m.vx(v), m.vy(v), m.vz(v)), `v${v}`).toBeCloseTo(radius, 1);
		}
		expect(signedVolume(m)).toBeCloseTo((4 / 3) * Math.PI * radius ** 3, 1);
	});

	test("a finer voxel gets closer to the ideal volume", () => {
		const ideal = (4 / 3) * Math.PI * 0.5 ** 1.5;
		let previous = 0;
		for (const voxelSize of [0.2, 0.1, 0.05]) {
			const error = Math.abs(
				signedVolume(created("Implicit Surface", { voxelSize, expr: "x*x+y*y+z*z-0.5" })) - ideal,
			);
			if (previous !== 0) expect(error, `voxel ${voxelSize}`).toBeLessThan(previous);
			previous = error;
		}
	});

	test("a field with a hole in it comes out genus 1", () => {
		// A torus of major radius 0.5 and minor 0.2, as an implicit field.
		const m = created("Implicit Surface", {
			voxelSize: 0.03,
			minX: -1,
			maxX: 1,
			minY: -1,
			maxY: 1,
			minZ: -0.5,
			maxZ: 0.5,
			expr: "(sqrt(x*x+y*y)-0.5)^2 + z*z - 0.04",
		});
		const facts = computeFacts(m);
		expect(facts.watertight).toBe(true);
		expect(facts.components).toBe(1);
		expect(facts.genus).toBe(1);
	});

	test("an empty field gives an empty mesh rather than throwing", () => {
		// Nothing crosses zero, so there is no level set to extract.
		const m = created("Implicit Surface", { voxelSize: 0.2, expr: "1" });
		expect(m.vn).toBe(0);
		expect(m.fn).toBe(0);
	});

	test("refuses a voxel size or a box that cannot work", () => {
		expect(() => created("Implicit Surface", { voxelSize: 0 })).toThrow(MLException);
		expect(() => created("Implicit Surface", { voxelSize: -1 })).toThrow(MLException);
		expect(() => created("Implicit Surface", { minX: 1, maxX: -1 })).toThrow(MLException);
		// And a voxel size so fine it would need billions of samples.
		expect(() => created("Implicit Surface", { voxelSize: 1e-4 })).toThrow(/limit/);
	});

	test("only x, y and z are in scope", () => {
		expect(() => created("Implicit Surface", { expr: "q" })).toThrow(MLException);
	});
});

describe("Refine User-Defined", () => {
	test("splits only the edges the condition names", () => {
		const { doc, cm } = scene(MeshElement.MM_VERTQUALITY, 1);
		for (let v = 0; v < cm.vertSize; v++) cm.vertQuality[v] = cm.vz(v);
		const before = { vn: cm.vn, fn: cm.fn };
		const out = kernel.applyFilter(doc, "Refine User-Defined", {});
		// Some but not all: only edges with both endpoints above the equator.
		expect(out.face_number as number).toBeGreaterThan(before.fn);
		expect(out.face_number as number).toBeLessThan(before.fn * 4);
		const facts = computeFacts(cm);
		expect(facts.watertight).toBe(true);
		expect(facts.nonManifoldEdges).toBe(0);
		expect(facts.genus).toBe(0);
	});

	test("splitting every edge quadruples the faces", () => {
		const { doc, cm } = scene(0, 1);
		kernel.applyFilter(doc, "Refine User-Defined", { condSelect: "1" });
		expect(cm.fn).toBe(320);
		expect(cm.vn).toBe(162);
	});

	test("the new vertex goes where the expression says", () => {
		// Pushed out to the unit sphere rather than left at the midpoint, which
		// is the standard trick for refining a sphere.
		const { doc, cm } = scene(0, 1);
		kernel.applyFilter(doc, "Refine User-Defined", {
			condSelect: "1",
			x: "(x0+x1)/2 / sqrt(((x0+x1)/2)^2 + ((y0+y1)/2)^2 + ((z0+z1)/2)^2)",
			y: "(y0+y1)/2 / sqrt(((x0+x1)/2)^2 + ((y0+y1)/2)^2 + ((z0+z1)/2)^2)",
			z: "(z0+z1)/2 / sqrt(((x0+x1)/2)^2 + ((y0+y1)/2)^2 + ((z0+z1)/2)^2)",
		});
		for (let v = 0; v < cm.vn; v++) {
			expect(Math.hypot(cm.vx(v), cm.vy(v), cm.vz(v)), `v${v}`).toBeCloseTo(1, 9);
		}
	});

	test("a condition nothing satisfies leaves the mesh alone", () => {
		const { doc, cm } = scene(0, 1);
		kernel.applyFilter(doc, "Refine User-Defined", { condSelect: "0" });
		expect(cm.vn).toBe(42);
		expect(cm.fn).toBe(80);
	});

	test("only the edge variables are in scope", () => {
		const { doc } = scene(0, 1);
		// `x2` belongs to a face, not an edge.
		expect(() => kernel.applyFilter(doc, "Refine User-Defined", { condSelect: "x2 > 0" })).toThrow(
			MLException,
		);
	});
});

describe("registration", () => {
	test("matches the upstream catalogue", () => {
		for (const [name, pythonName, cls] of [
			["Conditional Vertex Selection", "compute_selection_by_condition_per_vertex", "Selection"],
			["Conditional Face Selection", "compute_selection_by_condition_per_face", "Selection"],
			["Per Vertex Geometric Function", "compute_coord_by_function", "Smoothing"],
			["Per Vertex Normal Function", "compute_normal_by_function_per_vertex", "Normal"],
			["Per Face Normal Function", "compute_normal_by_function_per_face", "Normal"],
			["Per Vertex Color Function", "compute_color_by_function_per_vertex", "VertexColoring"],
			["Per Face Color Function", "compute_color_by_function_per_face", "FaceColoring"],
			["Grid Generator", "create_grid", "MeshCreation"],
			["Implicit Surface", "create_implicit_surface", "MeshCreation"],
			["Refine User-Defined", "meshing_refine_by_function", "Remeshing"],
		] as const) {
			const action = kernel.pluginManager.filterAction(name);
			expect(action, name).toBeDefined();
			if (!action) continue;
			expect(action.pythonName, name).toBe(pythonName);
			expect(filterClassToString(action.filterClass), name).toBe(cls);
			expect(action.plugin.pluginName(), name).toBe("FilterFunc");
		}
	});

	test("carries MeshLab's parameter defaults", () => {
		const geom = kernel.initParameterList("Per Vertex Geometric Function");
		expect(geom.getParameterByName("z").defaultValue.value).toBe("sin(x+y)");
		expect(
			kernel.initParameterList("Conditional Vertex Selection").getParameterByName("condSelect")
				.defaultValue.value,
		).toBe("(q < 0)");
		expect(
			kernel.initParameterList("Conditional Face Selection").getParameterByName("condSelect")
				.defaultValue.value,
		).toBe("(fi == 0)");
		expect(
			kernel.initParameterList("Per Vertex Quality Function").getParameterByName("q").defaultValue
				.value,
		).toBe("vi");
		expect(
			kernel.initParameterList("Per Face Quality Function").getParameterByName("q").defaultValue
				.value,
		).toBe("x0+y0+z0");
		expect(
			kernel.initParameterList("Implicit Surface").getParameterByName("expr").defaultValue.value,
		).toBe("x*x+y*y+z*z-0.5");
		const grid = kernel.initParameterList("Grid Generator");
		expect(grid.getParameterByName("numVertX").defaultValue.value).toBe(10);
		expect(grid.getParameterByName("absScaleX").defaultValue.value).toBeCloseTo(0.3, 6);
	});

	test("every filter in the plugin is implemented", () => {
		const pending = kernel
			.filterList()
			.filter((f) => f.plugin.pluginName() === "FilterFunc" && !f.implemented)
			.map((f) => f.name);
		expect(pending).toEqual([]);
	});
});
