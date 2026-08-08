#!/usr/bin/env bun
/**
 * Regenerates the golden fixtures from a *real* PyMeshLab, in Docker.
 *
 * Opt-in, and never run by CI. The whole reason this project exists is that
 * keeping a working Python environment around is a burden, so the test suite
 * proper is built on analytic meshes, mathematical invariants and
 * property-based checks — none of which need Python. This script is the
 * separate, occasional question: "does upstream actually agree?"
 *
 *   MESHLAB_TS_ALLOW_GOLDEN_REGEN=1 bun run golden:regen
 *
 * Review the resulting diff by hand. A change here means either we fixed
 * something or we broke something, and only a person can tell which.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";

if (process.env.MESHLAB_TS_ALLOW_GOLDEN_REGEN !== "1") {
	console.error(
		"refusing to run.\n\n" +
			"This talks to Docker and a real pymeshlab, and overwrites checked-in\n" +
			"golden fixtures. It must never run in CI. To run it deliberately:\n\n" +
			"  MESHLAB_TS_ALLOW_GOLDEN_REGEN=1 bun run golden:regen\n",
	);
	process.exit(1);
}

const root = join(import.meta.dir, "..");
const fixtures = join(root, "test", "fixtures");
const script = join(fixtures, "_regen", "regen.py");

if (!existsSync(script)) {
	console.error(`missing ${script}`);
	process.exit(1);
}

// Pinned: golden values are only comparable when they come from one known
// MeshLab version, and the version is recorded in every golden file. 2025.7
// matches the source in `.reference/` that the port was written against.
const PYMESHLAB_VERSION = "2025.7.post1";

// A local interpreter with pymeshlab installed skips Docker entirely:
//
//   python3.12 -m venv /tmp/pml && /tmp/pml/bin/pip install pymeshlab==2025.7.post1
//   MESHLAB_TS_ALLOW_GOLDEN_REGEN=1 PYMESHLAB_PYTHON=/tmp/pml/bin/python bun run golden:regen
//
// pymeshlab ships wheels for CPython 3.10-3.13 only.
const localPython = process.env.PYMESHLAB_PYTHON;
if (localPython !== undefined) {
	console.log(`running regen.py with ${localPython} against ${fixtures}`);
	const proc = Bun.spawnSync([localPython, script], {
		stdout: "inherit",
		stderr: "inherit",
		env: { ...process.env, FIXTURES_DIR: fixtures, PYMESHLAB_VERSION },
	});
	process.exit(proc.exitCode ?? 1);
}

const docker = Bun.spawnSync(["docker", "--version"], { stdout: "pipe", stderr: "pipe" });
if (docker.exitCode !== 0) {
	console.error(
		"docker is not available; either install it, or point PYMESHLAB_PYTHON at a\n" +
			"CPython 3.10-3.13 with pymeshlab installed (see the comment in this script).",
	);
	process.exit(1);
}

console.log(`running pymeshlab ${PYMESHLAB_VERSION} in Docker against ${fixtures}`);
const proc = Bun.spawnSync(
	[
		"docker",
		"run",
		"--rm",
		"-v",
		`${fixtures}:/fixtures`,
		"-e",
		`PYMESHLAB_VERSION=${PYMESHLAB_VERSION}`,
		"python:3.12-slim",
		"sh",
		"-c",
		`pip install --quiet pymeshlab==${PYMESHLAB_VERSION} && python /fixtures/_regen/regen.py`,
	],
	{ stdout: "inherit", stderr: "inherit" },
);

if (proc.exitCode !== 0) {
	console.error(`\npymeshlab run failed (exit ${proc.exitCode})`);
	process.exit(proc.exitCode ?? 1);
}

console.log("\ngolden fixtures regenerated — review `git diff test/fixtures/golden` by hand.");
