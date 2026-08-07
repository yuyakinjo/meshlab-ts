#!/usr/bin/env bun
/**
 * Verifies that the live registry still matches the upstream filter table.
 *
 * Runs in CI. It is the guard against the quiet failure mode of a staged
 * migration: a filter that gets implemented but registered under a slightly
 * different name, so the old name disappears and every recipe using it breaks
 * while the tests for the new implementation pass.
 */
import { MeshLabKernel } from "../src/common/meshlab_kernel.ts";
import { filterClassToString } from "../src/common/plugins/filter_class.ts";
import { FILTER_TABLE } from "../src/meshlabplugins/_stub/filter_table.ts";

const kernel = MeshLabKernel.default();
const registered = kernel.filterList();
const problems: string[] = [];

const byName = new Map(registered.map((f) => [f.name, f]));
for (const row of FILTER_TABLE) {
	const hit = byName.get(row.filterName);
	if (hit === undefined) {
		problems.push(`missing from the registry: "${row.filterName}" (${row.pluginDir})`);
		continue;
	}
	if (hit.pythonName !== row.pythonName) {
		problems.push(
			`"${row.filterName}": PyMeshLab name is "${hit.pythonName}", upstream says "${row.pythonName}"`,
		);
	}
	if (filterClassToString(hit.filterClass) !== row.filterClass) {
		problems.push(
			`"${row.filterName}": class is ${filterClassToString(hit.filterClass)}, upstream says ${row.filterClass}`,
		);
	}
}

const upstreamNames = new Set(FILTER_TABLE.map((r) => r.filterName));
for (const f of registered) {
	if (!upstreamNames.has(f.name)) {
		problems.push(`registered but not in MeshLab: "${f.name}" (${f.plugin.pluginName()})`);
	}
}

const seenNames = new Set<string>();
const seenPython = new Set<string>();
for (const f of registered) {
	if (seenNames.has(f.name)) problems.push(`duplicate filter name: "${f.name}"`);
	if (seenPython.has(f.pythonName)) problems.push(`duplicate PyMeshLab name: "${f.pythonName}"`);
	seenNames.add(f.name);
	seenPython.add(f.pythonName);
}

const implemented = registered.filter((f) => f.implemented);
console.log(
	`registry: ${registered.length} filters (${implemented.length} implemented, ` +
		`${registered.length - implemented.length} pending), upstream table has ${FILTER_TABLE.length}`,
);

if (problems.length > 0) {
	console.error(`\n${problems.length} problem(s):`);
	for (const p of problems) console.error(`  - ${p}`);
	process.exit(1);
}
console.log("registry matches the upstream filter table");
