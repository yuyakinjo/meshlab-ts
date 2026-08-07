#!/usr/bin/env bun
import { MeshLabKernel } from "../common/meshlab_kernel.ts";
/**
 * `meshlab-ts` — the command-line front end.
 *
 * Mirrors what `meshlabserver` is for: discovering filters and running them
 * without a GUI.
 */
import { MeshDocument } from "../common/ml_document/mesh_document.ts";
import { filterArityToString } from "../common/plugins/filter_arity.ts";
import { filterClassToString } from "../common/plugins/filter_class.ts";
import { MLException } from "../common/utilities/ml_exception.ts";

const USAGE = `meshlab-ts — a TypeScript port of MeshLab

Usage:
  meshlab-ts list [--class <name>] [--implemented|--todo] [--json] [pattern]
  meshlab-ts info <filter>
  meshlab-ts apply <filter> <input> -o <output> [--param key=value ...]
  meshlab-ts formats

Filters can be named either the way MeshLab shows them ("Close Holes") or the
way PyMeshLab does (meshing_close_holes).

Examples:
  meshlab-ts list --class Cleaning
  meshlab-ts info "Close Holes"
  meshlab-ts apply "Remove Duplicate Vertices" in.stl -o out.stl
`;

function fail(message: string): never {
	console.error(`meshlab-ts: ${message}`);
	process.exit(1);
}

/** Parses `key=value`, reading the value as JSON when it looks like JSON. */
function parseParam(text: string): [string, unknown] {
	const eq = text.indexOf("=");
	if (eq < 0) fail(`--param expects key=value, got "${text}"`);
	const key = text.slice(0, eq);
	const raw = text.slice(eq + 1);
	if (raw === "true") return [key, true];
	if (raw === "false") return [key, false];
	if (raw !== "" && !Number.isNaN(Number(raw))) return [key, Number(raw)];
	if (raw.startsWith("[") || raw.startsWith("{")) {
		try {
			return [key, JSON.parse(raw)];
		} catch {
			return [key, raw];
		}
	}
	return [key, raw];
}

function cmdList(args: string[]): void {
	const kernel = MeshLabKernel.default();
	const json = args.includes("--json");
	const onlyImplemented = args.includes("--implemented");
	const onlyTodo = args.includes("--todo");
	const classIdx = args.indexOf("--class");
	const wantClass = classIdx >= 0 ? args[classIdx + 1]?.toLowerCase() : undefined;
	const pattern = args.find((a, i) => !a.startsWith("--") && i !== classIdx + 1)?.toLowerCase();

	let filters = kernel.filterList();
	if (onlyImplemented) filters = filters.filter((f) => f.implemented);
	if (onlyTodo) filters = filters.filter((f) => !f.implemented);
	if (wantClass !== undefined) {
		filters = filters.filter((f) =>
			filterClassToString(f.filterClass).toLowerCase().includes(wantClass),
		);
	}
	if (pattern !== undefined) {
		filters = filters.filter(
			(f) => f.name.toLowerCase().includes(pattern) || f.pythonName.includes(pattern),
		);
	}

	if (json) {
		console.log(
			JSON.stringify(
				filters.map((f) => ({
					name: f.name,
					pythonName: f.pythonName,
					class: filterClassToString(f.filterClass),
					arity: filterArityToString(f.arity),
					implemented: f.implemented,
				})),
				null,
				2,
			),
		);
		return;
	}

	const width = Math.max(0, ...filters.map((f) => f.name.length));
	for (const f of filters) {
		const status = f.implemented ? "  " : "· ";
		console.log(
			`${status}${f.name.padEnd(width)}  ${filterClassToString(f.filterClass).padEnd(16)}  ${f.pythonName}`,
		);
	}
	const done = filters.filter((f) => f.implemented).length;
	console.log(
		`\n${filters.length} filter(s): ${done} implemented, ${filters.length - done} not yet ` +
			`(marked ·). Total registered: ${kernel.filterList().length}.`,
	);
}

function cmdInfo(args: string[]): void {
	const name = args[0];
	if (name === undefined) fail("info needs a filter name");
	const kernel = MeshLabKernel.default();
	const action = kernel.filterAction(name);

	console.log(action.name);
	console.log("=".repeat(action.name.length));
	console.log(`plugin:      ${action.plugin.pluginName()}`);
	console.log(`pymeshlab:   ${action.pythonName}`);
	console.log(`class:       ${filterClassToString(action.filterClass)}`);
	console.log(`arity:       ${filterArityToString(action.arity)}`);
	console.log(`implemented: ${action.implemented ? "yes" : "not yet"}`);
	if (action.info !== "") console.log(`\n${action.info}`);

	const params = action.plugin.initParameterList(action.id, undefined);
	if (params.size === 0) {
		console.log("\nno parameters");
		return;
	}
	console.log("\nparameters:");
	for (const p of params) {
		console.log(`  ${p.name} (${p.stringType()}) = ${JSON.stringify(p.value.value)}`);
		if (p.toolTip !== "") console.log(`      ${p.toolTip}`);
	}
}

function cmdApply(args: string[]): void {
	const name = args[0];
	const input = args[1];
	if (name === undefined || input === undefined)
		fail("apply needs a filter name and an input file");

	const outIdx = args.findIndex((a) => a === "-o" || a === "--output");
	const output = outIdx >= 0 ? args[outIdx + 1] : undefined;
	if (output === undefined) fail("apply needs -o <output>");

	const params: Record<string, unknown> = {};
	for (let i = 0; i < args.length; i++) {
		if (args[i] !== "--param" && args[i] !== "-p") continue;
		const spec = args[i + 1];
		if (spec === undefined) fail("--param needs key=value");
		const [k, v] = parseParam(spec);
		params[k] = v;
	}

	const kernel = MeshLabKernel.default();
	const doc = new MeshDocument();
	kernel.loadMesh(doc, input);
	console.error(`loaded ${input}: ${doc.mm().cm.vn} vertices, ${doc.mm().cm.fn} faces`);

	const out = kernel.applyFilter(doc, name, params);
	for (const [k, v] of Object.entries(out)) console.error(`  ${k}: ${JSON.stringify(v)}`);

	kernel.saveMesh(doc, output);
	console.error(`wrote ${output}: ${doc.mm().cm.vn} vertices, ${doc.mm().cm.fn} faces`);
}

function cmdFormats(): void {
	const { input, output } = MeshLabKernel.default().pluginManager.supportedExtensions();
	console.log(`read:  ${input.join(", ")}`);
	console.log(`write: ${output.join(", ")}`);
}

function main(argv: string[]): void {
	const [command, ...rest] = argv;
	switch (command) {
		case "list":
			cmdList(rest);
			return;
		case "info":
			cmdInfo(rest);
			return;
		case "apply":
			cmdApply(rest);
			return;
		case "formats":
			cmdFormats();
			return;
		case "-h":
		case "--help":
		case undefined:
			console.log(USAGE);
			return;
		default:
			fail(`unknown command "${command}"\n\n${USAGE}`);
	}
}

try {
	main(process.argv.slice(2));
} catch (err) {
	// An MLException is an expected outcome — an unimplemented filter, a bad
	// parameter, an unreadable file — so report it plainly. Anything else is a
	// bug here and keeps its stack.
	if (err instanceof MLException) {
		console.error(`meshlab-ts: ${err.name}: ${err.message}`);
		process.exit(1);
	}
	throw err;
}
