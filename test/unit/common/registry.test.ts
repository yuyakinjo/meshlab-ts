/**
 * The registry is the API-compatibility contract: if a name is wrong or
 * missing here, a recipe written against MeshLab or PyMeshLab fails even
 * though the filter behind it works.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../../src/common/ml_document/mesh_document.ts";
import { FilterArity } from "../../../src/common/plugins/filter_arity.ts";
import { filterClassToString } from "../../../src/common/plugins/filter_class.ts";
import { type FilterAction, PluginManager } from "../../../src/common/plugins/plugin_manager.ts";
import {
	MLException,
	MLNotImplementedException,
} from "../../../src/common/utilities/ml_exception.ts";
import { FILTER_TABLE } from "../../../src/meshlabplugins/_stub/filter_table.ts";
import { createStubPlugins } from "../../../src/meshlabplugins/_stub/stub_plugins.ts";

const kernel = MeshLabKernel.default();

describe("the filter registry", () => {
	test("registers every filter in the upstream table", () => {
		expect(kernel.filterList().length).toBe(FILTER_TABLE.length);
		const registered = new Set(kernel.filterList().map((f) => f.name));
		for (const row of FILTER_TABLE) {
			expect(registered.has(row.filterName), `missing "${row.filterName}"`).toBe(true);
		}
	});

	test("covers the plugins and counts the C++ sources declare", () => {
		// Independently counted from the MeshLab sources; a drift here means
		// the extractor stopped seeing part of a plugin.
		//
		// The counts are of *live* filters. filter_unsharp declares 22 cases
		// but one is commented out, and filter_voronoi declares 6 with two
		// commented out, so the numbers below are 21 and 4 — reading
		// commented-out code is exactly the bug the extractor's comment
		// stripping exists to prevent.
		expect(FILTER_TABLE.length).toBe(282);
		expect(new Set(FILTER_TABLE.map((r) => r.pluginDir)).size).toBe(43);
		const perPlugin = new Map<string, number>();
		for (const r of FILTER_TABLE) perPlugin.set(r.pluginDir, (perPlugin.get(r.pluginDir) ?? 0) + 1);
		for (const [dir, n] of [
			["filter_meshing", 37],
			["filter_colorproc", 28],
			["filter_select", 24],
			["filter_unsharp", 21],
			["filter_voronoi", 4],
			["filter_func", 18],
			["filter_clean", 15],
			["filter_sampling", 14],
			["filter_create", 13],
			["filter_layer", 13],
			["filter_measure", 8],
			["filter_mesh_booleans", 4],
			["filter_screened_poisson", 1],
		] as const) {
			expect(perPlugin.get(dir), dir).toBe(n);
		}
	});

	test("filter names are unique", () => {
		const seen = new Set<string>();
		for (const f of kernel.filterList()) {
			expect(seen.has(f.name), `duplicate "${f.name}"`).toBe(false);
			seen.add(f.name);
		}
	});

	test("PyMeshLab names are unique", () => {
		const seen = new Set<string>();
		for (const f of kernel.filterList()) {
			expect(seen.has(f.pythonName), `duplicate "${f.pythonName}"`).toBe(false);
			seen.add(f.pythonName);
		}
	});

	test("PyMeshLab names look like Python identifiers", () => {
		for (const f of kernel.filterList()) {
			expect(f.pythonName, f.name).toMatch(/^[a-z_][a-z0-9_]*$/);
		}
	});

	test("filters resolve by display name and by PyMeshLab name alike", () => {
		for (const [display, python] of [
			["Close Holes", "meshing_close_holes"],
			["Remove Duplicate Vertices", "meshing_remove_duplicate_vertices"],
			[
				"Surface Reconstruction: Screened Poisson",
				"generate_surface_reconstruction_screened_poisson",
			],
			["Box/Cube", "create_cube"],
			["Compute Geometric Measures", "get_geometric_measures"],
		] as const) {
			const byDisplay = kernel.filterAction(display);
			const byPython = kernel.filterAction(python);
			expect(byPython.name, python).toBe(byDisplay.name);
			expect(byDisplay.pythonName).toBe(python);
		}
	});

	test("a handful of well-known filters have the right class and arity", () => {
		const expectations = [
			["Close Holes", "Remeshing", FilterArity.SINGLE_MESH],
			["Remove Duplicate Vertices", "Cleaning", FilterArity.SINGLE_MESH],
			["Select None", "Selection", FilterArity.SINGLE_MESH],
			["Compute Geometric Measures", "Measure", FilterArity.SINGLE_MESH],
			// A creation filter takes no input mesh at all.
			["Box/Cube", "MeshCreation", FilterArity.NONE],
		] as const;
		for (const [name, klass, arity] of expectations) {
			const a = kernel.filterAction(name);
			expect(filterClassToString(a.filterClass), name).toBe(klass);
			expect(a.arity, name).toBe(arity);
		}
	});

	test("composite classes survive the round trip", () => {
		// "Merge Wedge Texture Coord" is Cleaning|Texture upstream.
		const a = kernel.filterAction("Merge Wedge Texture Coord");
		expect(filterClassToString(a.filterClass)).toBe("Cleaning|Texture");
	});

	test("an unknown filter name is an error, with suggestions", () => {
		expect(() => kernel.filterAction("Close Hole")).toThrow(MLException);
		try {
			kernel.filterAction("duplicate vert");
		} catch (err) {
			expect((err as Error).message).toContain("did you mean");
		}
	});

	test("registering the same filter name twice is rejected", () => {
		const pm = new PluginManager();
		const [a, b] = createStubPlugins(new Set());
		pm.registerFilterPlugin(a);
		expect(() => pm.registerFilterPlugin(a)).toThrow(/duplicate filter name/);
		// A different plugin is fine.
		expect(() => pm.registerFilterPlugin(b)).not.toThrow();
	});

	test("every filter has a non-empty description", () => {
		for (const f of kernel.filterList()) {
			expect(f.info.length, f.name).toBeGreaterThan(0);
		}
	});
});

describe("unimplemented filters", () => {
	test("the implemented set is exactly what the real plugins declare", () => {
		// Derived from the plugins rather than hardcoded, so the test does not
		// need editing every time a filter lands — but it still catches the
		// failure that matters: a filter that stops being implemented, or a
		// rename that leaves the old name sitting on a stub.
		// The plugins are discovered from the registry rather than listed, so
		// this needs no edit when one is added.
		const plugins = new Set(
			kernel
				.filterList()
				.filter((f) => f.implemented)
				.map((f) => f.plugin),
		);
		const declared = new Set<string>();
		for (const plugin of plugins) {
			for (const id of plugin.actions()) declared.add(plugin.filterName(id));
		}
		const registered = new Set(
			kernel
				.filterList()
				.filter((f) => f.implemented)
				.map((f) => f.name),
		);
		expect([...registered].sort()).toEqual([...declared].sort());
	});

	/** A filter that is still registry-only, whichever tier we are on. */
	function someUnimplemented(): FilterAction {
		const hit = kernel.filterList().find((f) => !f.implemented);
		if (hit === undefined) throw new Error("every filter is implemented — update this test");
		return hit;
	}

	test("throw MLNotImplementedException rather than doing nothing", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("m", "m");
		// Named filters, so a regression points at something specific rather
		// than at whatever happens to be first. These are far enough down the
		// tiers that they will not land soon; the assertion below fails loudly
		// with instructions if one does.
		for (const name of ["Mesh Boolean: Union", "Parametrization: Voronoi Atlas"]) {
			const action = kernel.filterAction(name);
			expect(action.implemented, `${name} is now implemented — pick another here`).toBe(false);
			expect(() => kernel.applyFilter(doc, name), name).toThrow(MLNotImplementedException);
		}
	});

	test("the exception names the filter and its plugin", () => {
		const doc = new MeshDocument();
		doc.addNewMesh("m", "m");
		const action = someUnimplemented();
		try {
			kernel.applyFilter(doc, action.name);
			throw new Error("should have thrown");
		} catch (err) {
			expect(err).toBeInstanceOf(MLNotImplementedException);
			const e = err as MLNotImplementedException;
			expect(e.filterName).toBe(action.name);
			expect(e.pluginName).toBe(action.plugin.pluginName());
			expect(e.message).toContain("not implemented");
		}
	});

	test("a stub never mutates the document", () => {
		const doc = new MeshDocument();
		const m = doc.addNewMesh("m", "m");
		expect(() => kernel.applyFilter(doc, someUnimplemented().name)).toThrow();
		expect(m.meshModified()).toBe(false);
		expect(doc.filterHistory).toHaveLength(0);
	});
});
