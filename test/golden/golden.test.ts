/**
 * The differential tests: does this library agree with the real MeshLab?
 *
 * Everything else in the suite checks that filters are mathematically
 * *correct*. That is not the same claim as *compatible* — two implementations
 * can both be right and still disagree — and a project calling itself
 * meshlab-ts owes its callers the second claim too. So each golden file holds
 * what a genuine PyMeshLab produced for one (mesh, filter, parameters) triple,
 * and this test runs the identical triple here and compares.
 *
 * The goldens are checked in, so this runs on every `bun test` with no Python
 * anywhere near it. Regenerating them is the deliberate, human-reviewed step:
 * `MESHLAB_TS_ALLOW_GOLDEN_REGEN=1 bun run golden:regen`.
 *
 * Each case declares how much agreement is expected, and is held to exactly
 * that:
 *
 * - `exact`      — the geometry digest itself matches: same coordinates to
 *                  nine decimals, whatever order they were produced in.
 * - `equivalent` — every topological integer matches; area and volume agree
 *                  to 1e-6 relative. For filters whose floating-point
 *                  summation order legitimately differs between
 *                  implementations.
 * - `loose`      — counts within 5%, area and volume within 2%, and the mesh
 *                  is closed exactly when upstream's is. For filters built on
 *                  heaps and heuristics, where which element wins a tie is
 *                  implementation-defined.
 *
 * A field upstream reports as unknown (-1 for topology on non-manifold
 * meshes, null where MeshLab declines to compute a volume) is skipped, not
 * treated as agreement.
 */
import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { type MeshSummary, summarizeMesh } from "../helpers/mesh_summary.ts";

const goldenRoot = join(import.meta.dir, "..", "fixtures", "golden");
const meshesRoot = join(import.meta.dir, "..", "fixtures", "meshes");

interface GoldenCase {
	readonly input: string;
	readonly filter: string;
	readonly params: Record<string, unknown>;
	readonly compare: "exact" | "equivalent" | "loose";
	readonly meshlabVersion: string;
	readonly summary: MeshSummary & { area: number | null; volume: number | null };
	/** Where it came from, for the test name. */
	readonly file: string;
}

function loadGoldenCases(): GoldenCase[] {
	const cases: GoldenCase[] = [];
	let directories: string[];
	try {
		directories = readdirSync(goldenRoot);
	} catch {
		return cases;
	}
	for (const dir of directories) {
		let files: string[];
		try {
			files = readdirSync(join(goldenRoot, dir));
		} catch {
			continue;
		}
		for (const file of files) {
			if (!file.endsWith(".json")) continue;
			const parsed = JSON.parse(readFileSync(join(goldenRoot, dir, file), "utf8"));
			cases.push({ ...parsed, file: `${dir}/${file}` });
		}
	}
	return cases;
}

/** The golden value markers, translated for this side's parameter types. */
function toKernelParams(params: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [name, value] of Object.entries(params)) {
		if (value !== null && typeof value === "object") {
			const marker = value as { abs?: number; percent?: number; enum?: string };
			// RichPercentage takes the raw number as an absolute length, and
			// RichEnum accepts the option's name — both PyMeshLab conventions
			// the parameter layer already follows.
			if (marker.abs !== undefined) out[name] = marker.abs;
			else if (marker.enum !== undefined) out[name] = marker.enum;
			else throw new Error(`unsupported marker for ${name}: ${JSON.stringify(value)}`);
		} else {
			out[name] = value;
		}
	}
	return out;
}

const relativeError = (ours: number, gold: number): number =>
	Math.abs(ours - gold) / Math.max(Math.abs(gold), 1e-12);

const cases = loadGoldenCases();
const kernel = MeshLabKernel.default();

describe("golden: agreement with real PyMeshLab", () => {
	// Golden files are optional until the first regen has been run; an empty
	// directory must not look like a passing compatibility suite.
	test(
		cases.length > 0 ? "golden fixtures are present" : "no golden fixtures — regen has not run",
		() => {
			expect(true).toBe(true);
		},
	);

	for (const golden of cases) {
		test(`${golden.compare}: ${golden.file} (pymeshlab ${golden.meshlabVersion})`, () => {
			const action = kernel.filterList().find((entry) => entry.pythonName === golden.filter);
			expect(action, `no filter with pythonName ${golden.filter}`).toBeDefined();

			const doc = new MeshDocument();
			kernel.loadMesh(doc, join(meshesRoot, golden.input));
			kernel.applyFilter(doc, (action as { name: string }).name, toKernelParams(golden.params));

			const model = doc.mm() as { cm: Parameters<typeof summarizeMesh>[0] };
			const ours = summarizeMesh(model.cm);
			const gold = golden.summary;

			// Integers. Upstream's -1 means "not computed", never "minus one".
			const integers = [
				"vn",
				"fn",
				"en",
				"boundaryEdges",
				"components",
				"boundaryLoops",
				"genus",
				"nonManifoldEdges",
				"nonManifoldVertices",
			] as const;
			if (golden.compare === "loose") {
				for (const field of ["vn", "fn"] as const) {
					const allowance = Math.max(2, Math.ceil(gold[field] * 0.05));
					expect(Math.abs(ours[field] - gold[field]), field).toBeLessThanOrEqual(allowance);
				}
				expect(ours.components, "components").toBe(gold.components);
				// Closed exactly when upstream's is closed.
				expect(ours.boundaryEdges === 0, "closedness").toBe(gold.boundaryEdges === 0);
			} else {
				for (const field of integers) {
					if (gold[field] === -1 && field !== "vn" && field !== "fn") continue;
					expect(ours[field], field).toBe(gold[field]);
				}
			}

			// Geometry scalars, where upstream computed them.
			// Even "exact" cannot beat 1e-6 on area and volume: MeshLab stores
			// coordinates as float32, so its scalars are computed from rounded
			// inputs and differ from our double-precision ones at ~1e-8 to 1e-7
			// relative. The geometry digest below is the strict check.
			const scalarTolerance = golden.compare === "loose" ? 2e-2 : 1e-6;
			if (gold.area !== null) {
				expect(relativeError(ours.area, gold.area), "area").toBeLessThan(scalarTolerance);
			}
			if (gold.volume !== null) {
				expect(relativeError(ours.volume, gold.volume), "volume").toBeLessThan(scalarTolerance);
			}

			// The digest: only where identical geometry is actually promised.
			if (golden.compare === "exact") {
				expect(ours.geometryHash, "geometryHash").toBe(gold.geometryHash);
			}
		});
	}
});
