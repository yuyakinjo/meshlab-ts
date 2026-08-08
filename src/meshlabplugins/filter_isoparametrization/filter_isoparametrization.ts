/**
 * `filter_isoparametrization` — Pietroni, Tarini and Cignoni's almost
 * isometric parametrisation over an abstract domain.
 *
 * The four filters form a pipeline. `Main` builds the domain and stores it on
 * the mesh; the other three consume it. Nothing else in the library produces
 * one, so each of them checks for it and says so rather than silently doing
 * nothing.
 *
 * The domain lives in a `WeakMap` keyed by the mesh, which is this library's
 * stand-in for MeshLab's per-mesh attribute: it disappears when the mesh does
 * and it does not serialise, which matches upstream's behaviour of losing the
 * parametrisation when the layer is deleted.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import {
	RichDynamicFloat,
	RichEnum,
	RichInt,
	RichMesh,
} from "../../common/parameters/rich_parameter.ts";
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
import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import {
	meshAngleDistortion,
	meshL2Stretch,
} from "../../vcg/complex/parametrization/distortion.ts";
import { IsoParametrization } from "./iso_parametrization.ts";

export const FP = {
	ISOP_PARAM: 0,
	ISOP_REMESHING: 1,
	ISOP_DIAMPARAM: 2,
	ISOP_TRANSFER: 3,
} as const;

/** The per-mesh attribute, in the only form a garbage-collected host allows. */
const DOMAINS = new WeakMap<CMeshO, IsoParametrization>();

const BIB =
	" For more details see: N. Pietroni, M. Tarini and P. Cignoni, 'Almost isometric mesh " +
	"parameterization through abstract domains', IEEE Transaction of Visualization and Computer " +
	"Graphics, 2010";

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.ISOP_PARAM]: {
		name: "Iso Parametrization: Main",
		pythonName: "compute_iso_parametrization",
		info:
			"The filter builds the abstract domain mesh representing the Isoparameterization of a " +
			"watertight two-manifold triangular mesh. This abstract mesh can be used to uniformly " +
			"remesh the input mesh, or to build a atlased texture parametrization. In short this " +
			"filter build a very coarse almost regular triangulation such that original mesh can be " +
			"reprojected from this abstract mesh with minimal distortion." +
			BIB,
	},
	[FP.ISOP_REMESHING]: {
		name: "Iso Parametrization Remeshing",
		pythonName: "generate_iso_parametrization_remeshing",
		info:
			"Uniform Remeshing based on Isoparameterization, each triangle of the domain is " +
			"recursively subdivided." +
			BIB,
	},
	[FP.ISOP_DIAMPARAM]: {
		name: "Iso Parametrization Build Atlased Mesh",
		pythonName: "generate_iso_parametrization_atlased_mesh",
		info:
			"The filter build a new mesh with a standard atlased per wedge texture. The atlas is " +
			"simply done by exploiting the low distortion, coarse, regular, mesh of the abstract " +
			"domain." +
			BIB,
	},
	[FP.ISOP_TRANSFER]: {
		name: "Iso Parametrization transfer between meshes",
		pythonName: "transfer_iso_parametrization_between_meshes",
		info:
			"Transfer the Isoparametrization between two meshes, the two meshes must be reasonably " +
			"similar and well aligned. It is useful to transfer back an isoparam onto the original " +
			"mesh after having computed it on a dummy, clean watertight model." +
			BIB,
	},
};

export class FilterIsoParametrization extends FilterPlugin {
	pluginName(): string {
		return "FilterIsoParametrization";
	}

	actions(): readonly ActionIDType[] {
		return Object.values(FP);
	}

	private spec(id: ActionIDType): FilterSpec {
		const s = SPECS[id];
		if (s === undefined) this.wrongActionCalled(id);
		return s;
	}

	filterName(id: ActionIDType): string {
		return this.spec(id).name;
	}
	pythonFilterName(id: ActionIDType): string {
		return this.spec(id).pythonName;
	}
	filterInfo(id: ActionIDType): string {
		return this.spec(id).info;
	}
	override getClass(_id: ActionIDType): FilterClassMask {
		return FilterClass.Remeshing;
	}
	filterArity(id: ActionIDType): FilterArityValue {
		return id === FP.ISOP_TRANSFER ? FilterArity.FIXED : FilterArity.SINGLE_MESH;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		switch (id) {
			case FP.ISOP_PARAM:
				list.add(
					new RichInt("targetAbstractMinFaceNum", 150, {
						description: "AM Min Size",
						tooltip:
							"This number and the following one indicate the range face number of the abstract " +
							"mesh that is used for the parametrization process. The algorithm will choose the " +
							"best abstract mesh with the number of triangles within the specified interval. If " +
							"the mesh has a very simple structure this range can be very low and strict; for a " +
							"roughly spherical object if you can specify a range of [8,8] faces you get an " +
							"octahedral abstract mesh, e.g. a geometry image. Large numbers (greater than 400) " +
							"are usually not of practical use.",
					}),
				);
				list.add(
					new RichInt("targetAbstractMaxFaceNum", 200, {
						description: "AM Max Size",
						tooltip: "An interval of 50 should be fine.",
					}),
				);
				// The optimisation criteria upstream picks between are all
				// stopping rules for a search over intermediate domains that we
				// do not keep; the enum is registered so a script's parameters
				// still load, and the value is reported back untouched.
				list.add(
					new RichEnum("stopCriteria", 1, ["Best Heuristic", "Area + Angle", "Regularity", "L2"], {
						description: "Optimization Criteria",
						tooltip: "Choose a metric to stop the parametrization within the interval.",
					}),
				);
				break;

			case FP.ISOP_REMESHING:
				list.add(
					new RichInt("SamplingRate", 10, {
						description: "Sampling Rate",
						tooltip: "This specify the sampling rate for remeshing. Must be greater than 2",
					}),
				);
				break;

			case FP.ISOP_DIAMPARAM:
				list.add(
					new RichDynamicFloat("BorderSize", 0.1, 0.01, 0.5, {
						description: "BorderSize ratio",
						tooltip:
							"This parameter controls the amount of space that must be left between each " +
							"diamond when building the atlas. The unit of the threshold is in percentage of " +
							"the size of the diamond; the bigger the threshold the less triangles are split, " +
							"but the more UV space is used.",
					}),
				);
				break;

			case FP.ISOP_TRANSFER: {
				const current = m?.id() ?? 0;
				list.add(
					new RichMesh("sourceMesh", current, {
						description: "Source Mesh",
						tooltip: "The mesh already having an Isoparameterization",
					}),
				);
				list.add(
					new RichMesh("targetMesh", current, {
						description: "Target Mesh",
						tooltip: "The mesh to be Isoparameterized",
					}),
				);
				break;
			}

			default:
				break;
		}
		return list;
	}

	applyFilter(
		id: ActionIDType,
		params: RichParameterList,
		doc: MeshDocument,
		post: PostConditionBox,
		cb: CallBackPos,
	): FilterOutput {
		switch (id) {
			case FP.ISOP_PARAM: {
				const m = doc.mm();
				const min = params.getInt("targetAbstractMinFaceNum");
				const max = params.getInt("targetAbstractMaxFaceNum");
				const iso = IsoParametrization.build(m.cm, {
					targetMinFaces: min,
					targetMaxFaces: max,
					onProgress: (fraction) => cb(100 * fraction, "Building the abstract domain"),
				});
				DOMAINS.set(m.cm, iso);

				// Reaching the target is not guaranteed: the collapses that
				// would get there may all be refused, and a refused collapse is
				// the mechanism that keeps the domain valid. Say so plainly
				// rather than reporting success at the wrong size.
				const reached = iso.faceCount <= max;
				doc.Log.log(
					`Built an abstract domain of ${iso.faceCount} faces from ${m.cm.fn}` +
						(reached ? "" : `; it could not be simplified below ${iso.faceCount}`),
				);
				post.mask = MeshElement.MM_NONE;
				return {
					abstract_mesh_faces: iso.faceCount,
					reached_target: reached,
					stop_criteria: params.getEnum("stopCriteria"),
				};
			}

			case FP.ISOP_REMESHING: {
				const m = doc.mm();
				const iso = requireDomain(m);
				const rate = params.getInt("SamplingRate");
				if (rate <= 2)
					throw new MLException(`the sampling rate must be greater than 2, got ${rate}`);

				const remeshed = iso.remesh(rate);
				const target = doc.addNewMesh("", `${m.label()} remeshed`, true, remeshed);
				target.updateBoxAndNormals();
				doc.Log.log(`Remeshed to ${remeshed.vn} vertices and ${remeshed.fn} faces at rate ${rate}`);
				return { new_mesh_id: target.id(), vertex_number: remeshed.vn, face_number: remeshed.fn };
			}

			case FP.ISOP_DIAMPARAM: {
				const m = doc.mm();
				const iso = requireDomain(m);
				const { cm, straddling } = iso.atlasUV(params.getDynamicFloat("BorderSize"));

				const target = doc.addNewMesh("", `${m.label()} atlased`, true, cm);
				target.updateDataMask(MeshElement.MM_WEDGTEXCOORD);
				target.updateBoxAndNormals();
				// The mesh copy already carries the wedge coordinates, so the
				// datamask call above must not clear them.
				const angle = meshAngleDistortion(cm);
				const stretch = meshL2Stretch(cm);
				doc.Log.log(
					`Atlased ${cm.fn} faces over ${iso.faceCount} domain slots; ` +
						`${straddling} faces straddled a slot boundary. ` +
						`Angle distortion ${angle.toFixed(4)}, L2 stretch ${stretch.toFixed(4)}`,
				);
				return {
					new_mesh_id: target.id(),
					straddling_faces: straddling,
					angle_distortion: angle,
					l2_stretch: stretch,
				};
			}

			case FP.ISOP_TRANSFER: {
				const source = doc.requireMesh(params.getMeshId("sourceMesh"));
				const target = doc.requireMesh(params.getMeshId("targetMesh"));
				const iso = requireDomain(source);
				const moved = iso.transferTo(target.cm);
				DOMAINS.set(target.cm, moved);
				doc.Log.log(
					`Transferred an isoparametrization of ${moved.faceCount} domain faces from ` +
						`"${source.label()}" to "${target.label()}"`,
				);
				post.mask = MeshElement.MM_NONE;
				return { abstract_mesh_faces: moved.faceCount };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}

function requireDomain(m: MeshModel): IsoParametrization {
	const iso = DOMAINS.get(m.cm);
	if (iso === undefined) {
		throw new MLException(
			`layer "${m.label()}" has no isoparametrization; run "Iso Parametrization: Main" on it first`,
		);
	}
	return iso;
}

/** Test and CLI access to the stored domain. */
export function isoParametrizationOf(cm: CMeshO): IsoParametrization | undefined {
	return DOMAINS.get(cm);
}
