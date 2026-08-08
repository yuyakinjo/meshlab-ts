/**
 * Stage B of the texture-defragmentation port: the structure.
 *
 * Chart decomposition, the seam network and the shell have no numerical
 * oracle, but they have exact combinatorics, and that is what is checked:
 * chart faces sum to the mesh's, every seam edge belongs to exactly one seam,
 * consecutive edges of a seam share a node, a shell is its chart plus its
 * filler.
 *
 * The sharpest test here is not combinatorial though. A grid split into two
 * charts by translating one half's UVs by a known amount must produce a seam
 * whose two sides, fed to the rigid matching fit from Stage A, recover exactly
 * that translation. That checks the seam pairing — which side of the cut each
 * coordinate came from — and nothing else would: a pairing that was reversed
 * or off by one still yields a plausible-looking seam.
 */
import { describe, expect, test } from "bun:test";
import { MeshLabKernel } from "../../src/common/meshlab_kernel.ts";
import { MeshDocument } from "../../src/common/ml_document/mesh_document.ts";
import { MeshElement } from "../../src/common/ml_document/mesh_element.ts";
import { MLException } from "../../src/common/utilities/ml_exception.ts";
import { Allocator } from "../../src/vcg/complex/allocator.ts";
import { CMeshO } from "../../src/vcg/complex/cmesho.ts";
import { getInfo } from "../../src/vcg/complex/hole.ts";
import { arapEnergy } from "../../src/vcg/complex/parametrization/arap2d.ts";
import {
	type AtlasMesh,
	buildAtlasMesh,
	computeChartGraph,
	isBorder,
	isSeamEdge,
	measureChart,
} from "../../src/vcg/complex/parametrization/chart_graph.ts";
import { matchingError, matchRigid } from "../../src/vcg/complex/parametrization/matching2.ts";
import {
	buildSeamMesh,
	clusterEndpoints,
	clusterSeamsByChartPair,
	extractSeamUV,
	generateSeams,
	seamLength3D,
} from "../../src/vcg/complex/parametrization/seams.ts";
import { boundaryPins, buildShell } from "../../src/vcg/complex/parametrization/shell.ts";
import { faceFace } from "../../src/vcg/complex/update/topology.ts";
import { sphereIcosa } from "../helpers/mesh_builders.ts";

const kernel = MeshLabKernel.default();

/** Gives a mesh a wedge-UV channel and returns it inside a document. */
function withUV(cm: CMeshO) {
	const doc = new MeshDocument();
	const m = doc.addNewMesh("", "m", true, cm);
	m.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
	m.updateBoxAndNormals();
	return { doc, m, cm };
}

/** An `n × m` grid of quads in the z = 0 plane. */
function grid(n: number, m: number): CMeshO {
	const cm = new CMeshO();
	const index = (i: number, j: number): number => j * (n + 1) + i;
	Allocator.addVertices(cm, (n + 1) * (m + 1));
	for (let j = 0; j <= m; j++) {
		for (let i = 0; i <= n; i++) cm.setVert(index(i, j), i, j, 0);
	}
	Allocator.addFaces(cm, 2 * n * m);
	let f = 0;
	for (let j = 0; j < m; j++) {
		for (let i = 0; i < n; i++) {
			cm.setFace(f++, index(i, j), index(i + 1, j), index(i + 1, j + 1));
			cm.setFace(f++, index(i, j), index(i + 1, j + 1), index(i, j + 1));
		}
	}
	return cm;
}

/** A grid whose UVs are its own coordinates, with one half shifted away. */
function twoChartGrid(n: number, m: number, cut: number, offset: number) {
	const scene = withUV(grid(n, m));
	const wt = scene.cm.wedgeTexCoord as Float64Array;
	for (let f = 0; f < scene.cm.faceSize; f++) {
		const centre = [0, 1, 2].reduce((s, q) => s + scene.cm.vx(scene.cm.fv(f, q)), 0) / 3;
		const right = centre > cut;
		for (let k = 0; k < 3; k++) {
			const v = scene.cm.fv(f, k);
			wt[6 * f + 2 * k] = scene.cm.vx(v) + (right ? offset : 0);
			wt[6 * f + 2 * k + 1] = scene.cm.vy(v);
		}
	}
	return scene;
}

/** A closed tube, unwrapped into one chart cut along a single line. */
function tube(nu: number, nv: number) {
	const cm = new CMeshO();
	const index = (i: number, j: number): number => j * nu + (i % nu);
	Allocator.addVertices(cm, nu * (nv + 1));
	for (let j = 0; j <= nv; j++) {
		for (let i = 0; i < nu; i++) {
			const angle = (2 * Math.PI * i) / nu;
			cm.setVert(index(i, j), Math.cos(angle), j * 0.5, Math.sin(angle));
		}
	}
	Allocator.addFaces(cm, 2 * nu * nv);
	let f = 0;
	for (let j = 0; j < nv; j++) {
		for (let i = 0; i < nu; i++) {
			cm.setFace(f++, index(i, j), index(i + 1, j), index(i + 1, j + 1));
			cm.setFace(f++, index(i, j), index(i + 1, j + 1), index(i, j + 1));
		}
	}
	const scene = withUV(cm);
	const wt = cm.wedgeTexCoord as Float64Array;
	for (let g = 0; g < cm.faceSize; g++) {
		const columns = [0, 1, 2].map((q) => cm.fv(g, q) % nu);
		const wraps = Math.max(...columns) - Math.min(...columns) > nu / 2;
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(g, k);
			const i = v % nu;
			wt[6 * g + 2 * k] = (wraps && i === 0 ? nu : i) / nu;
			wt[6 * g + 2 * k + 1] = Math.floor(v / nu) / nv;
		}
	}
	return scene;
}

/** Everything downstream of the cut, in one call. */
function decompose(cm: CMeshO) {
	const am = buildAtlasMesh(cm);
	const graph = computeChartGraph(am);
	const sm = buildSeamMesh(am, graph);
	const seams = generateSeams(sm, graph);
	const clusters = clusterSeamsByChartPair(sm, seams, graph);
	return { am, graph, sm, seams, clusters };
}

function countSeamHalfEdges(am: AtlasMesh): number {
	let count = 0;
	for (let f = 0; f < am.faceCount; f++) {
		for (let k = 0; k < 3; k++) if (isSeamEdge(am, f, k)) count++;
	}
	return count;
}

// ------------------------------------------------------------ atlas mesh

describe("the atlas mesh", () => {
	test("keeps the face numbering and only ever adds vertices", () => {
		// The whole reason this is not a CMeshO: face i here must be face i
		// there, so writing results back is the identity.
		const scene = twoChartGrid(4, 3, 2, 10);
		const am = buildAtlasMesh(scene.cm);
		expect(am.faceCount).toBe(scene.cm.fn);
		expect(am.vertexCount).toBeGreaterThanOrEqual(scene.cm.vn);
		for (let f = 0; f < am.faceCount; f++) {
			for (let k = 0; k < 3; k++) {
				expect(am.sourceVertex[am.faces[3 * f + k]]).toBe(scene.cm.fv(f, k));
			}
		}
	});

	test("splits a vertex exactly where the wedges disagree", () => {
		const scene = twoChartGrid(4, 3, 2, 10);
		const am = buildAtlasMesh(scene.cm);
		// The four vertices along the cut are each used by both halves, so each
		// becomes two. Nothing else moves.
		expect(am.vertexCount).toBe(scene.cm.vn + 4);
	});

	test("both adjacencies are symmetric", () => {
		const scene = twoChartGrid(4, 3, 2, 10);
		const am = buildAtlasMesh(scene.cm);
		for (const [face, edge] of [
			[am.ffFace, am.ffEdge],
			[am.ff3DFace, am.ff3DEdge],
		] as const) {
			for (let f = 0; f < am.faceCount; f++) {
				for (let k = 0; k < 3; k++) {
					const g = face[3 * f + k];
					const h = edge[3 * f + k];
					expect(face[3 * g + h], `face ${f} edge ${k}`).toBe(f);
					expect(edge[3 * g + h], `face ${f} edge ${k}`).toBe(k);
				}
			}
		}
	});

	test("cutting can only create borders, never remove them", () => {
		// A 3D border is a border in UV too; the reverse is what a seam is.
		const scene = twoChartGrid(4, 3, 2, 10);
		const am = buildAtlasMesh(scene.cm);
		for (let f = 0; f < am.faceCount; f++) {
			for (let k = 0; k < 3; k++) {
				if (isBorder(am.ff3DFace, f, k)) expect(isBorder(am.ffFace, f, k)).toBe(true);
			}
		}
	});

	test("a mesh with no texture coordinates is refused", () => {
		expect(() => buildAtlasMesh(grid(2, 2))).toThrow(MLException);
	});
});

// ----------------------------------------------------------- chart graph

describe("the chart graph", () => {
	test("every face belongs to exactly one chart", () => {
		for (const scene of [twoChartGrid(4, 3, 2, 10), tube(8, 4)]) {
			const { am, graph } = decompose(scene.cm);
			let total = 0;
			for (const chart of graph.charts.values()) {
				total += chart.faces.length;
				for (const f of chart.faces) expect(graph.chartOf[f]).toBe(chart.id);
			}
			expect(total).toBe(am.faceCount);
		}
	});

	test("each chart is connected in UV", () => {
		const { am, graph } = decompose(twoChartGrid(6, 4, 3, 20).cm);
		for (const chart of graph.charts.values()) {
			const seen = new Set([chart.faces[0]]);
			const stack = [chart.faces[0]];
			while (stack.length > 0) {
				const f = stack.pop() as number;
				for (let k = 0; k < 3; k++) {
					const g = am.ffFace[3 * f + k];
					if (g === f || seen.has(g)) continue;
					seen.add(g);
					stack.push(g);
				}
			}
			expect(seen.size).toBe(chart.faces.length);
		}
	});

	test("a per-triangle parametrization gives one chart per face", () => {
		// The extreme atlas: every triangle its own island, so the chart count
		// and the face count must agree exactly.
		const scene = withUV(sphereIcosa(2).mesh);
		kernel.applyFilter(scene.doc, "Parametrization: Trivial Per-Triangle", {});
		const { am, graph, sm } = decompose(scene.cm);
		expect(graph.charts.size).toBe(am.faceCount);
		// Every edge of a closed mesh is then a seam, and each is seen twice.
		expect(countSeamHalfEdges(am)).toBe(3 * am.faceCount);
		expect(sm.edges.length).toBe((3 * am.faceCount) / 2);
	});

	test("a single-chart parametrization has no seams at all", () => {
		const scene = twoChartGrid(4, 3, 2, 0); // offset 0: nothing is cut
		const { am, graph, sm, seams } = decompose(scene.cm);
		expect(graph.charts.size).toBe(1);
		expect(countSeamHalfEdges(am)).toBe(0);
		expect(sm.edges.length).toBe(0);
		expect(seams).toEqual([]);
	});

	test("charts that share a seam are recorded as adjacent", () => {
		const { graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		expect(graph.charts.size).toBe(2);
		const [a, b] = [...graph.charts.values()];
		expect(a.adjacent).toEqual(new Set([b.id]));
		expect(b.adjacent).toEqual(new Set([a.id]));
	});

	test("the measures are the ones the driver will rank on", () => {
		const scene = twoChartGrid(4, 3, 2, 10);
		const { am, graph } = decompose(scene.cm);
		let area3D = 0;
		let areaUV = 0;
		for (const chart of graph.charts.values()) {
			const measures = measureChart(am, chart);
			area3D += measures.area3D;
			areaUV += measures.areaUV;
			expect(measures.flipped).toBe(false);
			expect(measures.borderUV).toBeGreaterThan(0);
		}
		// The grid is its own parametrization, so the two areas agree exactly,
		// and both equal the grid's.
		expect(area3D).toBeCloseTo(12, 9);
		expect(areaUV).toBeCloseTo(12, 9);
	});

	test("a mirrored chart is reported as flipped", () => {
		const scene = twoChartGrid(4, 3, 2, 10);
		const wt = scene.cm.wedgeTexCoord as Float64Array;
		for (let i = 0; i < wt.length; i += 2) wt[i] = -wt[i];
		const { am, graph } = decompose(scene.cm);
		for (const chart of graph.charts.values()) {
			expect(measureChart(am, chart).flipped).toBe(true);
		}
	});
});

// ------------------------------------------------------------ seam network

describe("the seam network", () => {
	test("every seam edge belongs to exactly one seam", () => {
		for (const scene of [twoChartGrid(6, 4, 3, 20), tube(8, 4)]) {
			const { sm, seams } = decompose(scene.cm);
			const used = seams.flatMap((s) => s.edges);
			expect(used.length).toBe(sm.edges.length);
			expect(new Set(used).size).toBe(sm.edges.length);
		}
	});

	test("consecutive edges of a seam share a node", () => {
		// What "sorted" has to mean: shortening a seam from an end is only
		// meaningful if the list is a path.
		const { sm, seams } = decompose(tube(10, 5).cm);
		for (const seam of seams) {
			for (let i = 0; i + 1 < seam.edges.length; i++) {
				const a = sm.edges[seam.edges[i]];
				const b = sm.edges[seam.edges[i + 1]];
				const shared = [a.v0, a.v1].filter((v) => v === b.v0 || v === b.v1);
				expect(shared.length, `step ${i}`).toBeGreaterThan(0);
			}
		}
	});

	test("a straight cut is one seam with two endpoints", () => {
		const { sm, seams, clusters } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		expect(seams.length).toBe(1);
		expect(seams[0].edges.length).toBe(3); // one per row of the grid
		expect(seams[0].endpoints.length).toBe(2);
		expect(clusters.length).toBe(1);
		expect(clusterEndpoints(clusters[0]).length).toBe(2);
		expect(seamLength3D(sm, seams[0])).toBeCloseTo(3, 9);
	});

	test("a cut inside one chart is not clustered with cross-chart seams", () => {
		// The tube is a single chart cut along one line: the seam separates the
		// chart from itself, which is a different move from joining two charts.
		const { graph, clusters } = decompose(tube(8, 4).cm);
		expect(graph.charts.size).toBe(1);
		expect(clusters.length).toBe(1);
		expect(clusters[0].charts).toEqual([0, 0]);
	});

	test("seams between the same pair of charts are clustered together", () => {
		// A tube cut at two places around its circumference: two charts, meeting
		// along two disconnected runs. Merging them deals with both at once, so
		// this must be one cluster of two seams, not two moves.
		const nu = 8;
		const scene = tube(nu, 3);
		const wt = scene.cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < scene.cm.faceSize; f++) {
			const columns = [0, 1, 2].map((q) => scene.cm.fv(f, q) % nu);
			// The half a face sits in decides which chart, and the two cuts fall
			// at columns 0 and nu / 2.
			const half = Math.min(...columns) < nu / 2 && Math.max(...columns) < nu / 2 ? 0 : 1;
			for (let k = 0; k < 3; k++) {
				const v = scene.cm.fv(f, k);
				wt[6 * f + 2 * k] = (v % nu) / nu + half * 20;
				wt[6 * f + 2 * k + 1] = Math.floor(v / nu) / 3;
			}
		}
		const { graph, seams, clusters } = decompose(scene.cm);
		expect(graph.charts.size).toBe(2);
		expect(seams.length).toBe(2);
		expect(clusters.length).toBe(1);
		expect(clusters[0].seams.length).toBe(2);
	});

	test("the two sides of a seam recover the transform that separated them", () => {
		// The test that pins down the pairing. A reversed or off-by-one pairing
		// still produces a seam of the right length and the right endpoints.
		for (const offset of [10, -3.5, 100]) {
			const scene = twoChartGrid(5, 4, 2, offset);
			const { am, graph, sm, clusters } = decompose(scene.cm);
			const cluster = clusters[0];
			const { a, b } = extractSeamUV(am, sm, cluster, graph, cluster.charts[0]);
			expect(a.length / 2).toBe(5); // one per row boundary of the grid
			const found = matchRigid(a, b);
			expect(matchingError(found, a, b)).toBeLessThan(1e-12);
			// A pure translation: the linear part is the identity, up to the
			// signed zeros an atan2-based rotation can produce.
			for (const [i, want] of [1, 0, 0, 1].entries()) {
				expect(found.m[i]).toBeCloseTo(want, 12);
			}
			expect(found.tx).toBeCloseTo(-offset, 9);
			expect(found.ty).toBeCloseTo(0, 9);
		}
	});

	test("the seam graph joins edges the atlas cut apart", () => {
		// Nodes are surface points, not cut vertices — that is what makes a run
		// of seam edges connected at all.
		const { am, sm } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		expect(sm.positions.length / 3).toBe(4);
		expect(sm.edges.length).toBe(3);
		for (const edge of sm.edges) {
			expect(isSeamEdge(am, edge.fa, edge.ea)).toBe(true);
			expect(isSeamEdge(am, edge.fb, edge.eb)).toBe(true);
		}
	});
});

// ------------------------------------------------------------------ shell

describe("the shell", () => {
	test("is its chart, and maps every face back to where it came from", () => {
		const { am, graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		for (const chart of graph.charts.values()) {
			const shell = buildShell(am, chart);
			expect(shell.mesh.fn).toBe(chart.faces.length);
			expect([...shell.sourceFace].sort((x, y) => x - y)).toEqual(
				[...chart.faces].sort((x, y) => x - y),
			);
		}
	});

	test("at full scale its target shapes are the parametrization it came from", () => {
		// downscale = 1 must mean "change nothing", and the energy says so
		// exactly: the target is the current UV triangle, so distortion is zero.
		const { am, graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		for (const chart of graph.charts.values()) {
			const shell = buildShell(am, chart, { downscale: 1 });
			expect(arapEnergy(shell.faces, shell.target, shell.uv).energy).toBeLessThan(1e-18);
		}
	});

	test("downscaling shrinks the target area by the square of the factor", () => {
		// The grid is isometrically parametrized, so both singular values are 1
		// and the cap applies to both — the area scales as the factor squared.
		const { am, graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		const chart = [...graph.charts.values()][0];
		const areaOf = (downscale: number): number => {
			const shell = buildShell(am, chart, { downscale });
			let area = 0;
			for (let f = 0; f < shell.faces.length / 3; f++) {
				area +=
					Math.abs(
						shell.target[4 * f] * shell.target[4 * f + 3] -
							shell.target[4 * f + 1] * shell.target[4 * f + 2],
					) / 2;
			}
			return area;
		};
		expect(areaOf(0.5)).toBeCloseTo(areaOf(1) * 0.25, 9);
	});

	test("closing holes turns an annulus into a disk", () => {
		// The tube's chart is welded back into a ring by the shell, so it has two
		// boundaries. Capping hole size just below the longest loop would fill
		// neither, since both are the same length — the case worth a test.
		const { am, graph } = decompose(tube(8, 4).cm);
		const chart = [...graph.charts.values()][0];

		const open = buildShell(am, chart, { closeHoles: false });
		faceFace(open.mesh);
		expect(getInfo(open.mesh).length).toBe(2);
		expect(open.holeFillingFaces).toBe(0);

		const closed = buildShell(am, chart, { closeHoles: true });
		expect(closed.holeFillingFaces).toBeGreaterThan(0);
		expect(getInfo(closed.mesh).length).toBe(1);
		expect(closed.mesh.fn).toBe(chart.faces.length + closed.holeFillingFaces);
		// Filler faces have no source, and only filler faces have no source.
		for (let f = 0; f < closed.mesh.faceSize; f++) {
			expect(closed.sourceFace[f] < 0).toBe(f >= chart.faces.length);
		}
	});

	test("the pins it offers are exactly its boundary", () => {
		const { am, graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		const chart = [...graph.charts.values()][0];
		const shell = buildShell(am, chart);
		const pins = boundaryPins(shell);

		const onBorder = new Set<number>();
		for (let f = 0; f < shell.mesh.faceSize; f++) {
			for (let k = 0; k < 3; k++) {
				if (!shell.mesh.isBorderFF(f, k)) continue;
				onBorder.add(shell.mesh.fv(f, k));
				onBorder.add(shell.mesh.fv(f, (k + 1) % 3));
			}
		}
		expect(new Set(pins.keys())).toEqual(onBorder);
		for (const [v, position] of pins) {
			expect(position[0]).toBe(shell.uv[2 * v]);
			expect(position[1]).toBe(shell.uv[2 * v + 1]);
		}
	});

	test("a chart in one piece is reported as one piece", () => {
		const { am, graph } = decompose(twoChartGrid(4, 3, 2, 10).cm);
		for (const chart of graph.charts.values()) {
			expect(buildShell(am, chart).singleComponent).toBe(true);
		}
	});

	test("an empty face list is refused", () => {
		const { am } = decompose(twoChartGrid(2, 2, 1, 5).cm);
		expect(() => buildShell(am, { id: 0, faces: [], adjacent: new Set() })).toThrow(MLException);
	});
});

// ------------------------------------------- a real, fragmented atlas

describe("on an atlas produced by another filter", () => {
	/** A sphere run through the Voronoi atlas: many charts, many seams. */
	function voronoiScene() {
		const scene = withUV(sphereIcosa(3).mesh);
		kernel.applyFilter(scene.doc, "Parametrization: Voronoi Atlas", { regionNum: 10 });
		return scene;
	}

	test("decomposes without losing a face or a seam edge", () => {
		const { am, graph, sm, seams, clusters } = decompose(voronoiScene().cm);
		let total = 0;
		for (const chart of graph.charts.values()) total += chart.faces.length;
		expect(total).toBe(am.faceCount);
		expect(countSeamHalfEdges(am)).toBe(2 * sm.edges.length);
		expect(seams.flatMap((s) => s.edges).length).toBe(sm.edges.length);
		expect(clusters.length).toBeGreaterThan(0);
		expect(clusters.length).toBeLessThanOrEqual(seams.length);
	});

	test("every cluster's two sides pair up, one coordinate each", () => {
		const { am, graph, sm, clusters } = decompose(voronoiScene().cm);
		let checked = 0;
		for (const cluster of clusters) {
			const { a, b } = extractSeamUV(am, sm, cluster, graph, cluster.charts[0]);
			expect(a.length).toBe(b.length);
			expect(a.length).toBeGreaterThan(0);
			checked++;
		}
		expect(checked).toBeGreaterThan(10);
	});

	test("every chart builds a shell whose faces are its own", () => {
		const { am, graph } = decompose(voronoiScene().cm);
		for (const chart of graph.charts.values()) {
			const shell = buildShell(am, chart, { closeHoles: true });
			expect(shell.mesh.fn).toBe(chart.faces.length + shell.holeFillingFaces);
			for (let f = 0; f < chart.faces.length; f++) {
				expect(shell.sourceFace[f]).toBe(chart.faces[f]);
			}
		}
	});
});
