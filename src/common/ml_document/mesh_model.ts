/**
 * `MeshModel` — one layer of a document: a `CMeshO` plus its identity, its
 * datamask and its textures.
 */
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { UpdateNormal } from "../../vcg/complex/update/normal.ts";
import { UpdateTopology } from "../../vcg/complex/update/topology.ts";
import { MeshElement, maskAnd, maskIntersects, maskWithout } from "./mesh_element.ts";

export class MeshModel {
	/** The mesh itself. Public, as in MeshLab. */
	readonly cm: CMeshO;

	private readonly _id: number;
	private _label: string;
	private _fullPathFileName: string;
	private _visible = true;
	private _modified = false;
	/** Index within a multi-mesh file; -1 when it did not come from one. */
	idInFile = -1;
	readonly textures = new Map<string, Uint8Array>();

	constructor(id: number, fullFileName = "", label = "", mesh?: CMeshO) {
		this._id = id;
		this._fullPathFileName = fullFileName;
		this._label = label;
		this.cm = mesh ?? new CMeshO();
	}

	id(): number {
		return this._id;
	}

	/** The display name, falling back to the file's base name. */
	label(): string {
		return this._label !== "" ? this._label : this.shortName();
	}

	setLabel(label: string): void {
		this._label = label;
	}

	fullName(): string {
		return this._fullPathFileName;
	}

	setFileName(name: string): void {
		this._fullPathFileName = name;
	}

	shortName(): string {
		return this._fullPathFileName.split(/[\\/]/).pop() ?? "";
	}

	pathName(): string {
		const parts = this._fullPathFileName.split(/[\\/]/);
		parts.pop();
		return parts.join("/");
	}

	suffixName(): string {
		const short = this.shortName();
		const dot = short.lastIndexOf(".");
		return dot < 0 ? "" : short.slice(dot + 1);
	}

	isVisible(): boolean {
		return this._visible;
	}

	setVisible(visible = true): void {
		this._visible = visible;
	}

	meshModified(): boolean {
		return this._modified;
	}

	setMeshModified(modified = true): void {
		this._modified = modified;
	}

	// ---- datamask ----------------------------------------------------------

	dataMask(): number {
		return this.cm.currentDataMask;
	}

	hasDataMask(mask: number): boolean {
		return this.cm.hasDataMask(mask);
	}

	/**
	 * Makes `neededDataMask` available, allocating storage and — for the
	 * adjacency bits — computing the relation.
	 *
	 * This is the distinction that matters: enabling `MM_FACEFACETOPO` on the
	 * mesh alone gives zeroed arrays, whereas MeshLab's `updateDataMask`
	 * promises *valid* adjacency. Filters declare `getRequirements()` and rely
	 * on that promise, so it is honoured here exactly as the C++ does.
	 */
	updateDataMask(neededDataMask: number): void {
		if (neededDataMask === MeshElement.MM_NONE) return;
		if (maskIntersects(neededDataMask, MeshElement.MM_FACEFACETOPO)) {
			UpdateTopology.faceFace(this.cm);
		}
		if (maskIntersects(neededDataMask, MeshElement.MM_VERTFACETOPO)) {
			UpdateTopology.vertexFace(this.cm);
		}
		this.cm.enableChannels(neededDataMask);
	}

	clearDataMask(unneededDataMask: number): void {
		this.cm.disableChannels(unneededDataMask);
	}

	hasPerVertexColor(): boolean {
		return this.hasDataMask(MeshElement.MM_VERTCOLOR);
	}
	hasPerVertexQuality(): boolean {
		return this.hasDataMask(MeshElement.MM_VERTQUALITY);
	}
	hasPerVertexTexCoord(): boolean {
		return this.hasDataMask(MeshElement.MM_VERTTEXCOORD);
	}
	hasPerFaceColor(): boolean {
		return this.hasDataMask(MeshElement.MM_FACECOLOR);
	}
	hasPerFaceQuality(): boolean {
		return this.hasDataMask(MeshElement.MM_FACEQUALITY);
	}
	hasPerFaceWedgeTexCoords(): boolean {
		return this.hasDataMask(MeshElement.MM_WEDGTEXCOORD);
	}

	/**
	 * The attributes this mesh can currently satisfy a precondition with.
	 *
	 * Distinct from `dataMask()` because the vertex- and face-count bits are
	 * conditions on the geometry, not on which channels are allocated: a
	 * filter that requires `MM_FACENUMBER` needs the mesh to *have* faces.
	 */
	currentCapability(): number {
		let mask = this.cm.currentDataMask;
		if (this.cm.vn > 0) mask |= MeshElement.MM_VERTNUMBER;
		if (this.cm.fn > 0) mask |= MeshElement.MM_FACENUMBER;
		return mask >>> 0;
	}

	// ---- derived data ------------------------------------------------------

	/** Recomputes the bounding box and both sets of normals. */
	updateBoxAndNormals(): void {
		UpdateBounding.box(this.cm);
		if (this.cm.fn > 0) {
			UpdateNormal.perVertexNormalizedPerFaceNormalized(this.cm);
		}
	}

	clear(): void {
		this.cm.clear();
		this.textures.clear();
		this._modified = false;
	}

	/** Enables the channels an importer reported it had read. */
	enable(openingFileMask: number): void {
		this.updateDataMask(maskWithout(openingFileMask, 0));
	}

	/** True when every bit of `mask` is currently live. */
	isCapableOf(mask: number): boolean {
		return maskAnd(this.currentCapability(), mask) === mask >>> 0;
	}
}
