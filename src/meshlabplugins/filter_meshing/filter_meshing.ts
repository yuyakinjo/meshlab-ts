/**
 * `filter_meshing` — orientation and hole closing.
 *
 * The rest of upstream's 37 filters (decimation, remeshing, subdivision, the
 * transform family) arrive with the later tiers; these three are what a repair
 * pipeline needs.
 */
import type { MeshDocument } from "../../common/ml_document/mesh_document.ts";
import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import type { MeshModel } from "../../common/ml_document/mesh_model.ts";
import { RichBool, RichInt, RichPercentage } from "../../common/parameters/rich_parameter.ts";
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
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import { Hole } from "../../vcg/complex/hole.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";

export const FP = {
	FP_REORIENT: 0,
	FP_INVERT_FACES: 1,
	FP_CLOSE_HOLES: 2,
} as const;

const GEOMETRY_AND_TOPOLOGY = MeshElement.MM_GEOMETRY_AND_TOPOLOGY_CHANGE;

interface FilterSpec {
	readonly name: string;
	readonly pythonName: string;
	readonly info: string;
	readonly filterClass: FilterClassMask;
	readonly requirements: number;
}

const SPECS: Readonly<Record<number, FilterSpec>> = {
	[FP.FP_REORIENT]: {
		name: "Re-Orient all faces coherently",
		pythonName: "meshing_re_orient_faces_coherently",
		info: "Re-orient in a consistent way all the faces of the mesh.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
	[FP.FP_INVERT_FACES]: {
		name: "Invert Faces Orientation",
		pythonName: "meshing_invert_face_orientation",
		info:
			"Invert faces orientation, flipping the normals of the mesh. If requested, it tries to " +
			"guess the right orientation; mainly it decide to flip all the faces if the mesh is " +
			"'more' inside than outside.",
		filterClass: FilterClass.Normal,
		requirements: MeshElement.MM_NONE,
	},
	[FP.FP_CLOSE_HOLES]: {
		name: "Close Holes",
		pythonName: "meshing_close_holes",
		info:
			"Close holes smaller than a given threshold. The hole is filled by ear cutting, " +
			"choosing at each step the triangle that is best shaped and folds least sharply away " +
			"from the surrounding surface.",
		filterClass: FilterClass.Remeshing,
		requirements: MeshElement.MM_FACEFACETOPO,
	},
};

export class FilterMeshing extends FilterPlugin {
	pluginName(): string {
		return "FilterMeshing";
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
	override getClass(id: ActionIDType): FilterClassMask {
		return this.spec(id).filterClass;
	}
	filterArity(_id: ActionIDType): FilterArityValue {
		return FilterArity.SINGLE_MESH;
	}
	override getRequirements(id: ActionIDType): number {
		return this.spec(id).requirements;
	}
	override postCondition(_id: ActionIDType): number {
		return GEOMETRY_AND_TOPOLOGY;
	}

	override initParameterList(id: ActionIDType, m: MeshModel | undefined): RichParameterList {
		const list = new RichParameterList();
		switch (id) {
			case FP.FP_INVERT_FACES:
				list.add(
					new RichBool("forceFlip", true, {
						description: "Force Flip",
						tooltip:
							"If selected, the normals will always be flipped; otherwise, the filter tries to " +
							"set them outside",
					}),
				);
				list.add(
					new RichBool("onlySelected", false, {
						description: "Flip only selected faces",
						tooltip: "If selected, only selected faces will be affected",
					}),
				);
				break;

			case FP.FP_CLOSE_HOLES: {
				let diag = 1;
				let anySelected = false;
				if (m !== undefined) {
					UpdateBounding.box(m.cm);
					diag = m.cm.bbox.diagonal || 1;
					for (let f = 0; f < m.cm.faceSize && !anySelected; f++) {
						if (!m.cm.isFaceD(f) && m.cm.isFaceS(f)) anySelected = true;
					}
				}
				list.add(
					new RichInt("MaxHoleSize", 30, {
						description: "Max size to be closed ",
						tooltip: "The size is expressed as number of edges composing the hole boundary",
					}),
				);
				list.add(
					new RichBool("Selected", anySelected, {
						description: "Close holes with selected faces",
						tooltip: "Only the holes with at least one of the boundary faces selected are closed",
					}),
				);
				list.add(
					new RichBool("NewFaceSelected", true, {
						description: "Select the newly created faces",
						tooltip:
							"After closing a hole the faces that have been created are left selected. Any " +
							"previous selection is lost. Useful for example for smoothing the newly created " +
							"holes.",
					}),
				);
				list.add(
					new RichBool("SelfIntersection", true, {
						description: "Prevent creation of selfIntersecting faces",
						tooltip:
							"When closing an holes it tries to prevent the creation of faces that intersect " +
							"faces adjacent to the boundary of the hole. It is an heuristic, non " +
							"intersetcting hole filling can be NP-complete.",
					}),
				);
				list.add(
					new RichBool("RefineHole", false, {
						description: "Refine Filled Hole",
						tooltip:
							"After closing the hole it will refine the newly created triangles to make the " +
							"surface more smooth and the triangulation more evenly spaced",
					}),
				);
				list.add(
					new RichPercentage("RefineHoleEdgeLen", diag * 0.03, 0, diag, {
						description: "Hole Refinement Edge Len",
						tooltip: "The target edge lenght of the triangulation inside the filled hole.",
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
		_cb: CallBackPos,
	): FilterOutput {
		const m = doc.mm();
		const cm = m.cm;
		post.mask = GEOMETRY_AND_TOPOLOGY;

		switch (id) {
			case FP.FP_REORIENT: {
				// Orientability is only a question once every edge has at most
				// two faces; upstream refuses rather than guessing, and so do we.
				if (Clean.countNonManifoldEdgeFF(cm) > 0) {
					throw new MLException(
						"Mesh has some not 2-manifold edges, orientability requires manifoldness. " +
							'Run "Repair non Manifold Edges" first.',
					);
				}
				const { isOriented, isOrientable } = Clean.orientCoherentlyMesh(cm);
				UpdateTopology.faceFace(cm);
				m.updateBoxAndNormals();
				doc.Log.log(
					isOrientable
						? isOriented
							? "Mesh was already coherently oriented"
							: "Mesh has been re-oriented coherently"
						: "Mesh is not orientable; it was left partially re-oriented",
				);
				return { isOriented, isOrientable };
			}

			case FP.FP_INVERT_FACES: {
				const onlySelected = params.getBool("onlySelected");
				let flipped: boolean;
				if (params.getBool("forceFlip")) {
					Clean.flipMesh(cm, onlySelected);
					flipped = true;
				} else {
					// "Guess" mode: only flip when the mesh encloses a negative
					// volume, i.e. it is inside out.
					flipped = Clean.flipNormalOutside(cm);
				}
				m.updateBoxAndNormals();
				doc.Log.log(flipped ? "Faces have been flipped" : "Faces were already facing outward");
				return { flipped };
			}

			case FP.FP_CLOSE_HOLES: {
				if (Clean.countNonManifoldEdgeFF(cm) > 0) {
					throw new MLException(
						"Mesh has some not 2-manifold edges, filter requires edge manifoldness. " +
							'Run "Repair non Manifold Edges" first.',
					);
				}
				// The ear scoring reads vertex normals to tell convex ears from
				// concave ones, so they have to be current before filling.
				m.updateBoxAndNormals();

				const { holeCount, newFaces, firstNewFace } = Hole.fillHoles(cm, {
					maxHoleSize: params.getInt("MaxHoleSize"),
					selected: params.getBool("Selected"),
					strategy: params.getBool("SelfIntersection") ? "selfIntersection" : "minimumWeight",
				});

				if (params.getBool("NewFaceSelected")) Hole.selectFacesFrom(cm, firstNewFace);
				Allocator.compactEveryVector(cm);
				UpdateTopology.faceFace(cm);
				m.updateBoxAndNormals();

				doc.Log.log(`Closed ${holeCount} holes and added ${newFaces} new faces`);
				// Upstream's output keys, so a caller reading them keeps working.
				return { closed_holes: holeCount, new_faces: newFaces };
			}

			default:
				return this.wrongActionCalled(id);
		}
	}
}
