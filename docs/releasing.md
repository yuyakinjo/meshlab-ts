# Releasing to npm

Publishing runs through **OIDC trusted publishing**: the `Publish` workflow
authenticates to npm with a short-lived token GitHub mints for that one run, so
no npm token is stored anywhere — not in repository secrets, not on a laptop —
and npm attests build provenance automatically. What follows is the one-time
bootstrap and the routine that publishing becomes afterwards.

## One-time bootstrap (manual, deliberately)

The trusted-publisher setting lives in the package's settings on npmjs.com,
which means it can only be created once the package exists. So the very first
publish is done by a person:

```bash
npm login                 # the npmjs.com account that will own the package
npm whoami                # confirm
bun run typecheck && bun run lint && bun test && bun run registry:check
npm pack --dry-run        # read the file list; this is what the world gets
npm publish               # unscoped packages are public by default
```

npm may require a recent CLI and 2FA for a manual publish; `npm install -g
npm@latest` first if it complains.

Then, on npmjs.com → package `meshlab-ts` → **Settings → Trusted Publisher**:

| field             | value                 |
| ----------------- | --------------------- |
| Publisher         | GitHub Actions        |
| Organization/user | `yuyakinjo`           |
| Repository        | `meshlab-ts`          |
| Workflow filename | `publish.yml`         |
| Environment       | (leave empty)         |

From that point the workflow can publish and nothing else can be configured to.
Optionally, in the same settings, restrict publishing access to
**Require two-factor authentication and disallow tokens** — trusted publishing
still works, and stolen tokens become useless by construction.

## Versioning: calendar, like MeshLab's own

Versions are **CalVer, `YYYY.M.PATCH`** — the first release of August 2026 is
`2026.8.0`, a fix on top of it `2026.8.1`, the next month's release `2026.9.0`.
This matches the upstream culture (MeshLab releases as `2023.12`, PyMeshLab as
`2025.7.post3`), and it is honest about what a version of a port can promise:
the compatibility contract lives in the README and the golden suite, not in a
major-version number.

Two consequences worth knowing:

- npm requires versions to *parse* as semver, which forbids leading zeros:
  `2026.8.0`, never `2026.08.0`.
- Semver range operators read the year as a major version: `^2026.8.0` accepts
  anything in 2026, `~2026.8.0` only `2026.8.x`. Since the year says when a
  release happened rather than whether it breaks anything, consumers should
  pin exact versions (a lockfile does this anyway) or use `~`.

## Every release after that

1. Set `version` in `package.json` to today's `YYYY.M.PATCH`, commit, push,
   wait for CI.
2. Tag and release — the tag **must** be `v` + the exact version, and the
   workflow refuses to publish when they disagree:

   ```bash
   git tag v2026.8.1
   git push origin v2026.8.1
   gh release create v2026.8.1 --generate-notes
   ```

3. The `Publish` workflow re-runs the full gate (typecheck, lint, the whole
   test suite including the golden comparisons, `registry:check`) and then
   `npm publish`es with provenance. A release tag is not a reason to skip the
   checks; it is the reason to run them.

## What ships

`files` pins the tarball to `dist/`, `src/`, `bin/`, `.agents/`, `LICENSE`
and `README.md`. `dist/` is built by `prepack` (`tsc -p tsconfig.build.json`),
which npm runs automatically on every publish and pack — a stale build cannot
ship, because there is no way to pack without rebuilding. Node consumers get
`dist/`; Bun consumers resolve the `"bun"` export condition to `src/` and use
the TypeScript directly. `npm pack --dry-run` is the audit; run it whenever
`files` or the tree layout changes.

`.reference/` (the 430 MB MeshLab/VCGLib clones), `test/` and the golden
fixtures never ship.
