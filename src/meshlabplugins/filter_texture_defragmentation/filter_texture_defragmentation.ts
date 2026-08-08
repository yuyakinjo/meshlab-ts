/**
 * `filter_texture_defragmentation` — merging a fragmented atlas back together.
 *
 * A photo-reconstructed model arrives with its texture cut into hundreds of
 * small charts. Every cut costs a visible seam and wasted image, and the filter
 * undoes as many of them as it can without introducing distortion or overlap:
 * merge, repack, and resample the texture through the new parametrization.
 *
 * The algorithm is Maggiordomo, Cignoni and Tarini (Eurographics 2021), and it
 * lives in `vcg/complex/parametrization/` — chart graph, seams, shell, ARAP,
 * the greedy driver and the packer. This file is the parameter list, the
 * layer handling and the texture resampling.
 *
 * **Two divergences worth knowing before reading the output.**
 *
 * 1. **The new texture is rendered in software, not through OpenGL.** Upstream
 *    makes a GL context current and rasterises the resampled atlas with a
 *    shader. There is no GL here, and none is needed: the same rasteriser the
 *    `Transfer: … to Texture` filters use does it, with the same bilinear
 *    sampling and the same pull-push gutter fill.
 * 2. **`timelimit` is not implemented and is refused rather than ignored.** It
 *    stops the optimization by wall clock, which would make the output depend
 *    on the machine it ran on — the same result would not be reproducible from
 *    the same input. `maxMoves` is offered instead, which bounds the work
 *    deterministically. Asking for a time limit throws.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichDynamicFloat, RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
import { RichParameterList } from "../../common/parameters/rich_parameter_list.ts";
import { FilterArity, type FilterArityValue } from "../../common/plugins/filter_arity.ts";
import { FilterClass, type FilterClassMask } from "../../common/plugins/filter_class.ts";
import {
	type ActionIDType,
	type FilterOutput,
	FilterPlugin,
	type PostConditionBox,
} from "../../common/plugins/interfaces/filter_plugin.ts";
import type { CallBackPos } from "../../common/utilities/callback.ts";
import { MLException, MLNotImplementedException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import { countNonManifoldEdgeFF } from "../../vcg/complex/clean.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { enableChannels } from "../../vcg/complex/components.ts";
import { buildAtlasMesh } from "../../vcg/complex/parametrization/chart_graph.ts";
import { defragmentAtlas } from "../../vcg/complex/parametrization/defragment.ts";
import {
	applyPlacement,
	type ChartGeometry,
	packCharts,
} from "../../vcg/complex/parametrization/packing.ts";
import { faceFace } from "../../vcg/complex/update/topology.ts";
import { Image } from "../../vcg/space/image/image.ts";
import { isPng, readPng, writePng } from "../../vcg/space/image/png.ts";
import { pullPushFill, rasteriseFace } from "../filter_texture/rastering.ts";

export const FP = {
	FP_TEXTURE_DEFRAG: 0,
} as const;

const NAME = "Texture Map Defragmentation";
const PYTHON_NAME = "apply_texmap_defragmentation";
const INFO =
	"Reduces the texture fragmentation by merging atlas charts.  The used algorithm is: <br><b>" +
	"Texture Defragmentation for Photo-Reconstructed 3D Models</b><br>  <i>Andrea Maggiordomo, " +
	"Paolo Cignoni and Marco Tarini</i> <br> Eurographics 2021";

export class FilterTextureDefrag extends FilterPlugin {
	pluginName(): string {
		return "FilterTextureDefrag";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private check(id: ActionIDType): void {
		if (id !== FP.FP_TEXTURE_DEFRAG) this.wrongActionCalled(id);
	}

	filterName(id: ActionIDType): string {
		this.check(id);
		return NAME;
	}
	pythonFilterName(id: ActionIDType): string {
		this.check(id);
		return PYTHON_NAME;
	}
	filterInfo(id: ActionIDType): string {
		this.check(id);
		return INFO;
	}
	override getClass(id: ActionIDType): FilterClassMask {
		this.check(id);
		return FilterClass.Texture;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}

	override getPreConditions(_id: ActionIDType): number {
		return MeshElement.MM_WEDGTEXCOORD;
	}

	override getRequirements(_id: ActionIDType): number {
		return MeshElement.MM_FACEFACETOPO;
	}

	override postCondition(_id: ActionIDType): number {
		return MeshElement.MM_WEDGTEXCOORD | MeshElement.MM_ALL;
	}

	override initParameterList(id: ActionIDType, _m: MeshModel | undefined): RichParameterList {
		this.check(id);
		const list = new RichParameterList();
		list.add(
			new RichFloat("matchingThreshold", 2.0, {
				description: "Matching Error Threshold",
				tooltip:
					"Threshold on the seam alignment error. Using a higher threshold can reduce the " +
					"fragmentation but increase runtime and distortion.",
			}),
		);
		list.add(
			new RichFloat("boundaryTolerance", 0.2, {
				description: "Seam to chart-boundary-length tolerance",
				tooltip:
					"Cutoff on the minimum fractional seam length. Seams with lower fractional length " +
					"(relative to the chart perimeter) are not merged to keep the chart borders compact.",
			}),
		);
		list.add(
			new RichFloat("distortionTolerance", 0.5, {
				description: "Local ARAP distortion tolerance",
				tooltip:
					"Local UV-optimization distortion tolerance when merging a seam. If the local energy " +
					"is higher than this value, the operation is reverted.",
			}),
		);
		list.add(
			new RichFloat("globalDistortionTolerance", 0.025, {
				description: "Global ARAP distortion tolerance",
				tooltip:
					"Global ARAP distortion tolerance when merging a seam. If the global atlas energy is " +
					"higher than this value, the operation is reverted.",
			}),
		);
		list.add(
			new RichDynamicFloat("uvReductionLimit", 0.0, 0.0, 100.0, {
				description: "UV Length Target (percentage)",
				tooltip:
					"Target UV length as percentage of the input length. The algorithm halts if the " +
					"target UV length has been reached, or if no further seams can be merged.",
			}),
		);
		list.add(
			new RichFloat("offsetFactor", 5.0, {
				description: "Local expansion coefficient",
				tooltip:
					"Coefficient used to control the extension of the UV-optimization area. Larger values " +
					"can increase the efficacy of the defragmentation, but increase the cost of the " +
					"geometric optimization and the algorithm runtime.",
			}),
		);
		list.add(
			new RichFloat("timelimit", 0.0, {
				description: "Time limit (seconds)",
				tooltip:
					"Time limit for the defragmentation process (zero means unlimited). Not implemented " +
					"here: a wall-clock limit makes the result depend on the machine it ran on. Use " +
					"maxMoves for a deterministic bound.",
			}),
		);
		list.add(
			new RichInt("maxMoves", 0, {
				description: "Maximum merge attempts",
				tooltip:
					"Stop after this many attempted merges, accepted or not. Zero means no limit. This " +
					"is the deterministic replacement for the time limit.",
				advanced: true,
			}),
		);
		list.add(
			new RichInt("textureSize", 1024, {
				description: "Output texture size",
				tooltip: "The side, in pixels, of the texture the repacked atlas is rendered into.",
				advanced: true,
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		_post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		this.check(id);
		const source = doc.mm();
		if (source === undefined) throw new MLException("no current mesh");
		if (source.cm.wedgeTexCoord === null) {
			throw new MLException("the mesh has no per-wedge texture coordinates to defragment");
		}
		if (params.getFloat("timelimit") !== 0) {
			throw new MLNotImplementedException(
				"timelimit is not implemented: a wall-clock limit would make the result depend on the " +
					"machine it ran on, so the same input would not give the same atlas twice. Use " +
					"maxMoves for a deterministic bound.",
				"FilterTextureDefrag",
			);
		}
		const textureSize = params.getInt("textureSize");
		if (textureSize < 8)
			throw new MLException(`textureSize must be at least 8, got ${textureSize}`);

		cb(0, "Initializing layer...");
		// Upstream leaves the input alone and puts the result in a new layer.
		const target = doc.addNewMesh("", `texdefrag_${source.label()}`, true, cloneMesh(source.cm));
		target.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
		for (const [name, bytes] of source.textures) target.textures.set(name, bytes);
		target.cm.textures = [...source.cm.textures];
		target.updateBoxAndNormals();

		const cm = target.cm;
		faceFace(cm);
		if (countNonManifoldEdgeFF(cm) > 0) {
			this.log(
				doc,
				"Texture Defragmentation: mesh has non-manifold edges, seam topology may be unreliable",
			);
		}

		cb(10, "Reading the atlas...");
		const am = buildAtlasMesh(cm);

		cb(20, "Defragmenting atlas...");
		const defrag = defragmentAtlas(am, {
			matchingThreshold: params.getFloat("matchingThreshold"),
			boundaryTolerance: params.getFloat("boundaryTolerance"),
			distortionTolerance: params.getFloat("distortionTolerance"),
			globalDistortionThreshold: params.getFloat("globalDistortionTolerance"),
			uvBorderLengthReduction: params.getDynamicFloat("uvReductionLimit") / 100,
			// Upstream's offsetFactor grows the optimization area by a distance;
			// here the area is a ring count, so the factor is mapped onto it.
			optimizationRings: Math.max(1, Math.round(params.getFloat("offsetFactor") / 2)),
			maxMoves: params.getInt("maxMoves"),
		});

		cb(70, "Packing atlas...");
		const charts = chartGeometries(am, defrag.chartOf, defrag.uv);
		const packed = packCharts(charts, { resolution: 512, gutter: 2 });
		if (packed.failed.length > 0) {
			throw new MLException(
				`packing failed: ${packed.failed.length} of ${charts.length} charts could not be placed`,
			);
		}

		// The old coordinates have to survive until the texture is resampled.
		const oldUV = Float64Array.from(cm.wedgeTexCoord as Float64Array);
		const placementOf = new Map(packed.placements.map((p) => [p.chart, p]));
		const wt = cm.wedgeTexCoord as Float64Array;
		for (let f = 0; f < am.faceCount; f++) {
			const placement = placementOf.get(defrag.chartOf[f]);
			if (placement === undefined) continue;
			for (let k = 0; k < 3; k++) {
				const v = am.faces[3 * f + k];
				const [u, w] = applyPlacement(placement, packed, defrag.uv[2 * v], defrag.uv[2 * v + 1]);
				wt[6 * f + 2 * k] = u;
				wt[6 * f + 2 * k + 1] = w;
			}
		}

		cb(85, "Rendering texture...");
		const rendered = renderAtlas(cm, oldUV, source, textureSize);
		target.textures.clear();
		if (rendered !== null) {
			const name = `${target.label()}_optimized_texture_0.png`;
			target.textures.set(name, writePng(rendered));
			target.cm.textures = [name];
		} else {
			target.cm.textures = [];
		}

		target.updateBoxAndNormals();
		this.log(
			doc,
			`texture defragmentation: ${defrag.chartsBefore} -> ${defrag.chartsAfter} charts, ` +
				`${defrag.merges} merges, UV border ${defrag.borderUVBefore.toFixed(4)} -> ` +
				`${defrag.borderUVAfter.toFixed(4)}, atlas occupancy ${(packed.occupancy * 100).toFixed(1)}%`,
		);

		return {
			charts_before: defrag.chartsBefore,
			charts_after: defrag.chartsAfter,
			merges: defrag.merges,
			seam_edges_removed: defrag.seamEdgesRemoved,
			uv_border_before: defrag.borderUVBefore,
			uv_border_after: defrag.borderUVAfter,
			atlas_occupancy: packed.occupancy,
			stopped: defrag.stopped,
			texture_rendered: rendered !== null,
		};
	}

	private log(doc: MeshDocument, message: string): void {
		doc.Log.log(message);
	}
}

/**
 * A copy of a mesh, wedge coordinates included.
 *
 * `filter_layer`'s `appendMesh` is the general one, and it deliberately drops
 * the optional channels — but the whole point here is the wedge coordinates,
 * so they are carried over explicitly.
 */
function cloneMesh(source: CMeshO): CMeshO {
	const copy = new CMeshO();
	const live: number[] = [];
	const remap = new Int32Array(source.vertSize).fill(-1);
	for (let v = 0; v < source.vertSize; v++) {
		if (!source.isVertD(v)) live.push(v);
	}
	Allocator.addVertices(copy, live.length);
	live.forEach((v, index) => {
		remap[v] = index;
		copy.setVert(index, source.vx(v), source.vy(v), source.vz(v));
	});

	const faces: number[] = [];
	for (let f = 0; f < source.faceSize; f++) if (!source.isFaceD(f)) faces.push(f);
	Allocator.addFaces(copy, faces.length);
	faces.forEach((f, index) => {
		copy.setFace(index, remap[source.fv(f, 0)], remap[source.fv(f, 1)], remap[source.fv(f, 2)]);
	});

	const sourceWT = source.wedgeTexCoord;
	if (sourceWT !== null) {
		enableChannels(copy, MeshElement.MM_WEDGTEXCOORD);
		const wt = copy.wedgeTexCoord as Float64Array;
		faces.forEach((f, index) => {
			for (let k = 0; k < 6; k++) wt[6 * index + k] = sourceWT[6 * f + k];
		});
	}
	copy.textures = [...source.textures];
	return copy;
}

/** Each chart's UV triangles, for the packer. */
function chartGeometries(
	am: { faceCount: number; faces: Int32Array },
	chartOf: Int32Array,
	uv: Float64Array,
): ChartGeometry[] {
	const byChart = new Map<number, number[]>();
	for (let f = 0; f < am.faceCount; f++) {
		const list = byChart.get(chartOf[f]);
		if (list === undefined) byChart.set(chartOf[f], [f]);
		else list.push(f);
	}
	// Sorted so the packer sees the charts in a stable order whatever the map's
	// insertion order happened to be.
	return [...byChart.entries()]
		.sort((a, b) => a[0] - b[0])
		.map(([id, faces]) => {
			const triangles = new Float64Array(6 * faces.length);
			faces.forEach((f, index) => {
				for (let k = 0; k < 3; k++) {
					const v = am.faces[3 * f + k];
					triangles[6 * index + 2 * k] = uv[2 * v];
					triangles[6 * index + 2 * k + 1] = uv[2 * v + 1];
				}
			});
			return { id, triangles };
		});
}

/**
 * Resamples the source textures through the new parametrization.
 *
 * For every pixel of the new atlas, the face that covers it gives barycentric
 * weights; those weights on the *old* coordinates say where to read from the
 * old image. Pixels no face covers are filled by pull-push afterwards, so
 * bilinear filtering at a chart's edge picks up its own colour rather than the
 * background.
 *
 * Returns null when the mesh has no texture to resample — the atlas is still
 * repacked, there is simply no image to go with it.
 */
function renderAtlas(
	cm: CMeshO,
	oldUV: Float64Array,
	source: MeshModel,
	size: number,
): Image | null {
	const images: Image[] = [];
	for (const name of source.cm.textures) {
		const bytes = source.textures.get(name);
		if (bytes === undefined || !isPng(bytes)) continue;
		images.push(readPng(bytes));
	}
	if (images.length === 0) return null;

	const out = new Image(size, size);
	const covered = new Uint8Array(size * size);

	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		rasteriseFace(cm, f, size, size, (sample) => {
			// The same weights on the old coordinates: where this texel used to be.
			let u = 0;
			let v = 0;
			for (let k = 0; k < 3; k++) {
				u += sample.bary[k] * oldUV[6 * f + 2 * k];
				v += sample.bary[k] * oldUV[6 * f + 2 * k + 1];
			}
			const colour = sampleBilinear(images[0], u, v);
			const index = (sample.y * size + sample.x) * 4;
			out.data.set(colour, index);
			covered[sample.y * size + sample.x] = 1;
		});
	}

	// Pull-push wants the uncovered pixels distinguishable; alpha zero marks them.
	for (let i = 0; i < covered.length; i++) if (covered[i] === 0) out.data[4 * i + 3] = 0;
	pullPushFill(out, 0);
	for (let i = 0; i < covered.length; i++) out.data[4 * i + 3] = 255;
	return out;
}

/** Bilinear sample of an image at a UV coordinate, clamped at the edges. */
function sampleBilinear(image: Image, u: number, v: number): [number, number, number, number] {
	const x = Math.min(image.width - 1, Math.max(0, u * image.width - 0.5));
	// v runs up, rows run down.
	const y = Math.min(image.height - 1, Math.max(0, (1 - v) * image.height - 0.5));
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const x1 = Math.min(image.width - 1, x0 + 1);
	const y1 = Math.min(image.height - 1, y0 + 1);
	const fx = x - x0;
	const fy = y - y0;

	const out: [number, number, number, number] = [0, 0, 0, 255];
	for (let c = 0; c < 3; c++) {
		const p00 = image.data[(y0 * image.width + x0) * 4 + c];
		const p10 = image.data[(y0 * image.width + x1) * 4 + c];
		const p01 = image.data[(y1 * image.width + x0) * 4 + c];
		const p11 = image.data[(y1 * image.width + x1) * 4 + c];
		out[c] = Math.round(
			p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + p01 * (1 - fx) * fy + p11 * fx * fy,
		);
	}
	return out;
}
