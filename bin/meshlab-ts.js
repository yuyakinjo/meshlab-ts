#!/usr/bin/env node
// The published CLI entry: plain ESM against the built dist/, so it runs on
// Node and Bun alike. The extensionless `bin/meshlab-ts` next to it is the
// repo-development entry (bun, straight from src/) and works without a build.
import "../dist/cli/meshlabserver.js";
