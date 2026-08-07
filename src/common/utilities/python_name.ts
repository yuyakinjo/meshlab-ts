/**
 * `pymeshlab::computePythonName` — the default derivation of a PyMeshLab
 * filter name from a MeshLab display name.
 *
 * Ported from `src/common/python/python_utils.cpp`.
 *
 * **This is only the fallback.** `FilterPlugin::pythonFilterName` is virtual,
 * and 281 of MeshLab's 285 filters override it, so the derived name is usually
 * *not* the name PyMeshLab exposes: "Close Holes" is `meshing_close_holes`,
 * not `close_holes`, and "Box/Cube" is `create_cube`, not `box_cube`. The real
 * names are extracted from the C++ sources into `filter_table.ts`; this
 * function covers only the handful that do not override.
 *
 * Treating the derivation as authoritative would silently give 281 filters the
 * wrong PyMeshLab name, which is why the table is generated rather than
 * computed.
 */

/**
 * Python's reserved words, from `python_utils.h`.
 *
 * Reproduced with upstream's typo intact — the entry `"def\tfrom"` is a
 * missing comma in the C++ source, which means the real words `def` and `from`
 * are not actually in the list and so never get their trailing underscore. We
 * match the behaviour rather than the intent, because PyMeshLab's actual
 * attribute names are what a caller has to type.
 */
const PYTHON_KEYWORDS: ReadonlySet<string> = new Set([
	"False",
	"await",
	"else",
	"import",
	"pass",
	"None",
	"break",
	"except",
	"in",
	"raise",
	"True",
	"class",
	"finally",
	"is",
	"return",
	"and",
	"continue",
	"for",
	"lambda",
	"try",
	"as",
	"def\tfrom",
	"nonlocal",
	"while",
	"assert",
	"del",
	"global",
	"not",
	"with",
	"async",
	"elif",
	"if",
	"or",
	"yield",
]);

/** Characters upstream strips outright, from the regex `[().,'":+]+`. */
const STRIPPED = /[().,'":+]+/g;

export function computePythonName(name: string): string {
	let pythonName = name.toLowerCase();
	pythonName = pythonName.replaceAll(" ", "_");
	pythonName = pythonName.replaceAll("/", "_");
	pythonName = pythonName.replaceAll("-", "_");
	pythonName = pythonName.replace(STRIPPED, "");
	if (PYTHON_KEYWORDS.has(pythonName)) pythonName += "_";
	return pythonName;
}
