/**
 * `filter_createiso` — an isosurface of a noisy implicit function.
 *
 * Upstream's one filter, and MeshLab's smallest creation plugin: sample a
 * scalar field over a grid and extract its zero set. The field here is a
 * sphere perturbed by fractal noise, which is what makes the result
 * interesting rather than a sphere.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { RichFloat, RichInt } from "../../common/parameters/rich_parameter.ts";
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
import { MLException } from "../../common/utilities/ml_exception.ts";
import { marchingTetrahedra } from "../../vcg/complex/create/marching.ts";
import { FractalAlgorithm, FractalField } from "../../vcg/math/noise.ts";

export const FP = { FP_CREATEISO: 0 } as const;

export class FilterCreateIso extends FilterPlugin {
	pluginName(): string {
		return "FilterCreateIso";
	}
	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}
	filterName(id: ActionIDType): string {
		if (id !== FP.FP_CREATEISO) this.wrongActionCalled(id);
		return "Noisy Isosurface";
	}
	pythonFilterName(id: ActionIDType): string {
		if (id !== FP.FP_CREATEISO) this.wrongActionCalled(id);
		return "create_noisy_isosurface";
	}
	filterInfo(id: ActionIDType): string {
		if (id !== FP.FP_CREATEISO) this.wrongActionCalled(id);
		return "Create a isosurface perturbed by a noisy isosurface.";
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.MeshCreation;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.NONE;
	}

	override initParameterList(_id: ActionIDType): RichParameterList {
		const list = new RichParameterList();
		list.add(
			new RichInt("Resolution", 64, {
				description: "Grid Resolution",
				tooltip: "The resolution of the grid the isosurface is extracted from.",
			}),
		);
		list.add(
			new RichFloat("Seed", 1, {
				description: "Seed",
				tooltip: "The random seed. The same seed always gives the same surface.",
			}),
		);
		list.add(
			new RichFloat("NoiseScale", 0.35, {
				description: "Noise amount",
				tooltip: "How far the noise pushes the sphere in and out; 0 gives a plain sphere.",
			}),
		);
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		if (id !== FP.FP_CREATEISO) return this.wrongActionCalled(id);
		const resolution = params.getInt("Resolution");
		if (resolution < 4) {
			throw new MLException(`the grid resolution must be at least 4, got ${resolution}`);
		}
		const amount = params.getFloat("NoiseScale");
		const field = new FractalField({
			algorithm: FractalAlgorithm.FBM,
			octaves: 5,
			lacunarity: 2,
			fractalIncrement: 1,
			offset: 0,
			gain: 1,
			seed: params.getFloat("Seed"),
		});

		// The grid spans -1.5..1.5 so the perturbed unit sphere stays inside
		// it: a surface touching the wall would be cut open by the extraction.
		const counts = [resolution + 1, resolution + 1, resolution + 1];
		const step = 3 / resolution;
		const coord = (_axis: number, i: number) => -1.5 + i * step;
		const index = (i: number, j: number, k: number) => (k * counts[1] + j) * counts[0] + i;
		const values = new Float64Array(counts[0] * counts[1] * counts[2]);

		for (let k = 0; k < counts[2]; k++) {
			cb((100 * k) / counts[2], "Sampling the implicit function");
			for (let j = 0; j < counts[1]; j++) {
				for (let i = 0; i < counts[0]; i++) {
					const x = coord(0, i);
					const y = coord(1, j);
					const z = coord(2, k);
					const radius = Math.hypot(x, y, z);
					values[index(i, j, k)] = radius - 1 - amount * field.at(x * 2, y * 2, z * 2);
				}
			}
		}

		const cm = marchingTetrahedra(values, counts, coord, index);
		if (cm.fn === 0) {
			throw new MLException(
				"the field has no zero crossing; the noise amount may have swallowed the surface",
			);
		}
		const m = doc.addNewMesh("", "Noisy Isosurface", true, cm);
		m.updateBoxAndNormals();
		post.mask = MeshElement.MM_NONE;
		doc.Log.log(`Extracted an isosurface of ${cm.vn} vertices and ${cm.fn} faces`);
		return { new_mesh_id: m.id(), vertex_number: cm.vn, face_number: cm.fn };
	}
}
