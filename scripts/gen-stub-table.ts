#!/usr/bin/env bun
/**
 * Extracts the complete filter table from the MeshLab C++ sources in
 * `.reference/`.
 *
 * Transcribing ~280 filter names across 43 plugins by hand would guarantee
 * typos, and a mistyped name is a silent API-compatibility break: a caller's
 * recipe would fail with "no such filter" for a filter we do implement. So the
 * table is generated, and the generated diff is what gets reviewed.
 *
 * The extraction is deliberately simple-minded — it reads the `filterName`,
 * `filterInfo`, `getClass` and `filterArity` switch bodies with regexes rather
 * than parsing C++. Anything it cannot read it reports rather than guesses;
 * run with `--report` to see the gaps.
 *
 *   bun run scripts/gen-stub-table.ts [--report]
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { computePythonName } from "../src/common/utilities/python_name.ts";

const REFERENCE = join(import.meta.dir, "..", ".reference", "meshlab", "src", "meshlabplugins");
const OUTPUT = join(import.meta.dir, "..", "src", "meshlabplugins", "_stub", "filter_table.ts");

if (!existsSync(REFERENCE)) {
	console.error(`missing ${REFERENCE}\nRun: bun run reference:clone`);
	process.exit(1);
}

interface FilterRow {
	pluginDir: string;
	pluginName: string;
	actionId: string;
	filterName: string;
	pythonName: string;
	filterClass: string;
	arity: string;
	info: string;
}

/**
 * Removes C and C++ comments, leaving string and character literals intact.
 *
 * Not cosmetic. `filter_voronoi` has two `case` arms commented out, and
 * reading them made the extractor believe the PyMeshLab name of "Cross Field
 * Creation" was the string `"Cross Field Creation"` — a name no caller could
 * ever type. Naively deleting from `//` to end of line would be just as wrong
 * in the other direction, because filter descriptions contain URLs.
 */
function stripComments(src: string): string {
	let out = "";
	let i = 0;
	while (i < src.length) {
		const c = src[i];
		const next = src[i + 1];
		if (c === '"' || c === "'") {
			// Copy the literal verbatim, honouring backslash escapes.
			const quote = c;
			out += c;
			i++;
			while (i < src.length) {
				if (src[i] === "\\") {
					out += src.slice(i, i + 2);
					i += 2;
					continue;
				}
				out += src[i];
				if (src[i] === quote) {
					i++;
					break;
				}
				i++;
			}
			continue;
		}
		if (c === "/" && next === "/") {
			while (i < src.length && src[i] !== "\n") i++;
			continue;
		}
		if (c === "/" && next === "*") {
			i += 2;
			while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
			i += 2;
			// Keep a space so `a/*x*/b` does not become `ab`.
			out += " ";
			continue;
		}
		out += c;
		i++;
	}
	return out;
}

/** Every .cpp/.h under a plugin directory, concatenated and de-commented. */
function pluginSources(dir: string): string {
	const out: string[] = [];
	const walk = (d: string) => {
		for (const entry of readdirSync(d)) {
			const p = join(d, entry);
			if (statSync(p).isDirectory()) walk(p);
			else if (entry.endsWith(".cpp") || entry.endsWith(".h")) out.push(readFileSync(p, "utf8"));
		}
	};
	walk(dir);
	return stripComments(out.join("\n"));
}

/** Body of a member function whose signature matches `sigPattern`. */
function functionBody(src: string, sigPattern: RegExp): string | null {
	const match = sigPattern.exec(src);
	if (match === null) return null;
	let i = src.indexOf("{", match.index);
	if (i < 0) return null;
	let depth = 0;
	const start = i;
	for (; i < src.length; i++) {
		if (src[i] === "{") depth++;
		else if (src[i] === "}" && --depth === 0) return src.slice(start + 1, i);
	}
	return null;
}

/** Joins adjacent C string literals: `"a" "b"` -> `ab`. */
function joinLiterals(text: string): string {
	const parts = [...text.matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]);
	return parts
		.join("")
		.replaceAll("\\n", "\n")
		.replaceAll("\\t", "\t")
		.replaceAll('\\"', '"')
		.replaceAll("\\\\", "\\")
		.trim();
}

/**
 * Splits a function body into one chunk per action id it dispatches on.
 *
 * Covers both dispatch styles upstream uses: a `switch` with `case FP_X:`
 * labels, and an if-chain of `filter == FP_X`. Action ids are always
 * SCREAMING_SNAKE, which is what keeps the `==` form from matching unrelated
 * comparisons.
 */
function dispatchChunks(body: string): Array<{ id: string; text: string }> {
	const anchors = [
		...body.matchAll(/case\s+([A-Za-z_][A-Za-z0-9_:]*)\s*:|==\s*([A-Za-z_][A-Za-z0-9_:]*)/g),
	];
	const out: Array<{ id: string; text: string }> = [];
	for (let i = 0; i < anchors.length; i++) {
		const raw = anchors[i][1] ?? anchors[i][2];
		const id = (raw.split("::").pop() as string).trim();
		if (!/^[A-Z][A-Z0-9_]*$/.test(id)) continue;
		const from = (anchors[i].index as number) + anchors[i][0].length;
		const to = i + 1 < anchors.length ? (anchors[i + 1].index as number) : body.length;
		out.push({ id, text: body.slice(from, to) });
	}
	return out;
}

/**
 * `id -> string literal` pairs, e.g. from `filterName` or `filterInfo`.
 *
 * A chunk with no literal falls through to the next one that has any, which is
 * how `case A: case B: return "x";` gives both ids the same value.
 */
function idStrings(body: string | null): Map<string, string> {
	const out = new Map<string, string>();
	if (body === null) return out;
	let pending: string[] = [];
	for (const { id, text } of dispatchChunks(body)) {
		pending.push(id);
		if (!text.includes('"')) continue;
		const value = joinLiterals(text);
		if (value === "") continue;
		for (const p of pending) out.set(p, value);
		pending = [];
	}
	return out;
}

/**
 * `case FP_X: ... return <enum member>;` pairs.
 *
 * Handles the two shapes upstream actually uses beyond the simple one:
 *
 * - **Fall-through.** `case A: case B: case C: return Cleaning;` gives all
 *   three ids the same value; only the last arm carries the return, so bare
 *   arms accumulate until one does.
 * - **Composite values.** `FilterClass(Cleaning + Texture)` is an OR of two
 *   members, so every known member named in the arm is collected, not just
 *   the first.
 */
function caseEnumMembers(body: string | null, members: ReadonlySet<string>): Map<string, string> {
	const out = new Map<string, string>();
	if (body === null) return out;
	let pending: string[] = [];
	for (const { id, text } of dispatchChunks(body)) {
		pending.push(id);
		const found = enumMembersIn(text, members);
		if (found === null) continue; // falls through to the next arm
		for (const p of pending) out.set(p, found);
		pending = [];
	}
	return out;
}

/**
 * The enum members named in a `return ...;` statement, joined with `|`.
 *
 * Sorted into declaration (bit) order rather than the order they happen to
 * appear in the C++ expression, so that `Camera + Normal` and `Normal + Camera`
 * produce the same string. They are the same mask, and a table that recorded
 * the source order would make `check-registry` report dozens of differences
 * that are not differences.
 */
function enumMembersIn(text: string, members: ReadonlySet<string>): string | null {
	const ret = /return\b([^;]*);/.exec(text);
	if (ret === null) return null;
	const order = [...members];
	const found = [
		...new Set(
			[...ret[1].matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)]
				.map((m) => m[0])
				.filter((w) => members.has(w)),
		),
	].sort((a, b) => order.indexOf(a) - order.indexOf(b));
	return found.length === 0 ? null : found.join("|");
}

/** Value the function returns unconditionally, if it has no switch. */
function constantReturn(body: string | null, members: ReadonlySet<string>): string | null {
	if (body === null || /case\s+[A-Za-z_]/.test(body)) return null;
	return enumMembersIn(body, members);
}

const CLASS_MEMBERS: ReadonlySet<string> = new Set([
	"Generic",
	"Selection",
	"Cleaning",
	"Remeshing",
	"FaceColoring",
	"VertexColoring",
	"MeshColoring",
	"MeshCreation",
	"Smoothing",
	"Quality",
	"Layer",
	"RasterLayer",
	"Normal",
	"Sampling",
	"Texture",
	"RangeMap",
	"PointSet",
	"Measure",
	"Polygonal",
	"Camera",
	"Other",
]);

const ARITY_MEMBERS: ReadonlySet<string> = new Set([
	"NONE",
	"SINGLE_MESH",
	"FIXED",
	"VARIABLE",
	"UNKNOWN_ARITY",
]);

const rows: FilterRow[] = [];
const problems: string[] = [];

const pluginDirs = readdirSync(REFERENCE)
	.filter((d) => d.startsWith("filter_"))
	.filter((d) => statSync(join(REFERENCE, d)).isDirectory())
	.sort();

for (const dir of pluginDirs) {
	const src = pluginSources(join(REFERENCE, dir));

	const names = idStrings(functionBody(src, /::\s*filterName\s*\(\s*ActionIDType[^)]*\)[^{;]*/));
	if (names.size === 0) {
		problems.push(`${dir}: no filterName dispatch found`);
		continue;
	}

	// Most plugins inherit pythonFilterName, which is computePythonName of the
	// display name. A handful override it — filter_screened_poisson calls its
	// filter `generate_surface_reconstruction_screened_poisson`, not the
	// derived `surface_reconstruction_screened_poisson`. Missing an override
	// means a PyMeshLab recipe would name a filter we claim not to have, so
	// the override wins wherever it exists.
	const pythonOverrides = idStrings(
		functionBody(src, /::\s*pythonFilterName\s*\(\s*ActionIDType[^)]*\)[^{;]*/),
	);

	const infos = idStrings(functionBody(src, /::\s*filterInfo\s*\(\s*ActionIDType[^)]*\)[^{;]*/));
	// `::name(` catches out-of-line definitions in the .cpp; the bare form
	// catches the plugins that define these inline in the header.
	const classBody =
		functionBody(src, /::\s*getClass\s*\([^)]*\)[^{;]*/) ??
		functionBody(src, /\bgetClass\s*\(\s*const\s+QAction[^)]*\)[^{;]*/);
	const classes = caseEnumMembers(classBody, CLASS_MEMBERS);
	const classDefault = constantReturn(classBody, CLASS_MEMBERS);
	const arityBody =
		functionBody(src, /::\s*filterArity\s*\([^)]*\)[^{;]*/) ??
		functionBody(src, /\bfilterArity\s*\(\s*const\s+QAction[^)]*\)[^{;]*/);
	const arities = caseEnumMembers(arityBody, ARITY_MEMBERS);
	const arityDefault = constantReturn(arityBody, ARITY_MEMBERS);

	const pluginNameBody = functionBody(src, /::\s*pluginName\s*\([^)]*\)[^{;]*/);
	const pluginName = pluginNameBody === null ? dir : joinLiterals(pluginNameBody) || dir;

	for (const [actionId, filterName] of names) {
		const filterClass = classes.get(actionId) ?? classDefault ?? "Generic";
		const arity = arities.get(actionId) ?? arityDefault ?? "SINGLE_MESH";
		if (!classes.has(actionId) && classDefault === null) {
			problems.push(`${dir}/${actionId}: no getClass arm, defaulted to Generic`);
		}
		if (!arities.has(actionId) && arityDefault === null) {
			problems.push(`${dir}/${actionId}: no filterArity arm, defaulted to SINGLE_MESH`);
		}
		rows.push({
			pluginDir: dir,
			pluginName,
			actionId,
			filterName,
			pythonName: pythonOverrides.get(actionId) ?? computePythonName(filterName),
			filterClass,
			arity,
			info: infos.get(actionId) ?? "",
		});
	}
}

// A duplicate filter name is exactly what PluginManager::checkFilterPlugin
// rejects at load time; catching it here means the generated table can never
// produce a registry that refuses to build.
const byName = new Map<string, FilterRow[]>();
for (const r of rows) {
	const hit = byName.get(r.filterName);
	if (hit === undefined) byName.set(r.filterName, [r]);
	else hit.push(r);
}
for (const [name, dupes] of byName) {
	if (dupes.length > 1) {
		problems.push(`duplicate filter name "${name}": ${dupes.map((d) => d.pluginDir).join(", ")}`);
	}
}

rows.sort((a, b) =>
	a.pluginDir !== b.pluginDir
		? a.pluginDir.localeCompare(b.pluginDir)
		: a.filterName.localeCompare(b.filterName),
);

const esc = (s: string) => JSON.stringify(s);
const body = rows
	.map(
		(r) =>
			`\t{\n` +
			`\t\tpluginDir: ${esc(r.pluginDir)},\n` +
			`\t\tpluginName: ${esc(r.pluginName)},\n` +
			`\t\tactionId: ${esc(r.actionId)},\n` +
			`\t\tfilterName: ${esc(r.filterName)},\n` +
			`\t\tpythonName: ${esc(r.pythonName)},\n` +
			`\t\tfilterClass: ${esc(r.filterClass)},\n` +
			`\t\tarity: ${esc(r.arity)},\n` +
			`\t\tinfo: ${esc(r.info)},\n` +
			`\t},`,
	)
	.join("\n");

const out = `// GENERATED by scripts/gen-stub-table.ts from .reference/meshlab — do not edit.
//
// The complete list of MeshLab filters: ${rows.length} filters across
// ${pluginDirs.length} plugins. Every one is registered from the moment the
// library loads, so filterList() is complete and a caller can discover names
// even for filters that are not implemented yet — those throw
// MLNotImplementedException rather than silently doing nothing.
//
// Regenerate with: bun run stub:gen

/** One row of the upstream filter table, as read from the C++ sources. */
export interface FilterTableRow {
	/** Directory under src/meshlabplugins, e.g. "filter_clean". */
	readonly pluginDir: string;
	/** The plugin's own reported name. */
	readonly pluginName: string;
	/** The plugin-local action enum constant, e.g. "FP_CLOSE_HOLES". */
	readonly actionId: string;
	/** The exact display name callers pass to applyFilter. */
	readonly filterName: string;
	/** The snake_case name PyMeshLab exposes. */
	readonly pythonName: string;
	/** FilterClass enum member name. */
	readonly filterClass: string;
	/** FilterArity enum member name. */
	readonly arity: string;
	/** The filter's description; empty when upstream builds it dynamically. */
	readonly info: string;
}

export const FILTER_TABLE: readonly FilterTableRow[] = [
${body}
];
`;

await Bun.write(OUTPUT, out);
console.log(`wrote ${rows.length} filters from ${pluginDirs.length} plugins to ${OUTPUT}`);

if (problems.length > 0) {
	console.log(`\n${problems.length} thing(s) the extractor could not read cleanly:`);
	const show = process.argv.includes("--report") ? problems : problems.slice(0, 15);
	for (const p of show) console.log(`  - ${p}`);
	if (show.length < problems.length) {
		console.log(`  ... and ${problems.length - show.length} more (pass --report)`);
	}
}
