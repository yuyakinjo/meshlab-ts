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

## Every release after that

1. Bump `version` in `package.json` (semver; `0.x` until the golden suite is
   broad enough to promise more), commit, push, wait for CI.
2. Tag and release — the tag **must** be `v` + the exact version, and the
   workflow refuses to publish when they disagree:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   gh release create v0.1.1 --generate-notes
   ```

3. The `Publish` workflow re-runs the full gate (typecheck, lint, the whole
   test suite including the golden comparisons, `registry:check`) and then
   `npm publish`es with provenance. A release tag is not a reason to skip the
   checks; it is the reason to run them.

## What ships

`files` in package.json pins the tarball to `src/`, `bin/`, `LICENSE` and
`README.md` — TypeScript source, no build step, which is honest about the
runtime: the package declares `engines.bun` and is consumed as TS by Bun.
`npm pack --dry-run` is the audit; run it whenever `files` or the tree layout
changes.

`.reference/` (the 430 MB MeshLab/VCGLib clones), `test/` and the golden
fixtures never ship.
