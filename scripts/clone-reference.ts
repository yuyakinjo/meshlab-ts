#!/usr/bin/env bun
/**
 * Clones MeshLab and VCGLib into .reference/ as a read-only specification.
 *
 * Nothing in src/ imports from .reference/ — it exists so that filter names,
 * parameter defaults and algorithm behaviour can be checked against upstream
 * while implementing. The directory is gitignored.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";

const REPOS = [
	{
		dir: "meshlab",
		url: "https://github.com/cnr-isti-vclab/meshlab.git",
		branch: "main",
		blobless: true,
	},
	{
		dir: "vcglib",
		url: "https://github.com/cnr-isti-vclab/vcglib.git",
		branch: "devel",
		blobless: false,
	},
] as const;

const referenceDir = join(import.meta.dir, "..", ".reference");

for (const repo of REPOS) {
	const target = join(referenceDir, repo.dir);
	if (existsSync(target)) {
		console.log(`${repo.dir}: already present, skipping`);
		continue;
	}
	const args = ["clone", "--depth", "1", "--branch", repo.branch];
	if (repo.blobless) args.push("--filter=blob:none");
	args.push(repo.url, target);

	console.log(`${repo.dir}: cloning ${repo.url} (${repo.branch})`);
	const proc = Bun.spawnSync(["git", ...args], { stdout: "inherit", stderr: "inherit" });
	if (proc.exitCode !== 0) {
		throw new Error(`git clone failed for ${repo.dir} (exit ${proc.exitCode})`);
	}
}

console.log("reference sources ready under .reference/");
