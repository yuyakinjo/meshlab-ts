/**
 * `MeshDocument` — the set of mesh layers a filter operates on, plus the
 * document-level log and filter history.
 *
 * Kept a plain data container, as in C++. Running filters is
 * `MeshLabKernel`'s job, so a document can be built, inspected and serialised
 * without dragging the whole plugin registry along.
 */

import type { CMeshO } from "../../vcg/complex/cmesho.ts";
import { Box3 } from "../../vcg/space/box3.ts";
import { Log } from "../utilities/log.ts";
import { MLException } from "../utilities/ml_exception.ts";
import { MeshModel } from "./mesh_model.ts";
import { RasterModel } from "./raster_model.ts";

/** One entry of the applied-filter history, replayable as a script. */
export interface FilterScriptStep {
	readonly filterName: string;
	readonly params: Record<string, unknown>;
}

export class MeshDocument {
	private meshList: MeshModel[] = [];
	private meshIdCounter = 0;
	private currentId = -1;
	private rasterList: RasterModel[] = [];
	private rasterIdCounter = 0;
	private currentRasterId = -1;

	readonly Log = new Log();
	readonly filterHistory: FilterScriptStep[] = [];
	docLabel = "";
	private _fileName = "";
	private _busy = false;

	// ---- layers ------------------------------------------------------------

	meshNumber(): number {
		return this.meshList.length;
	}

	/** The current mesh. Throws when the document is empty. */
	mm(): MeshModel {
		const m = this.currentMeshOrNull();
		if (m === null) throw new MLException("the document has no current mesh");
		return m;
	}

	currentMeshOrNull(): MeshModel | null {
		if (this.currentId < 0) return null;
		return this.meshList.find((m) => m.id() === this.currentId) ?? null;
	}

	getMesh(id: number): MeshModel | undefined {
		return this.meshList.find((m) => m.id() === id);
	}

	/** Like {@link getMesh} but reports the bad id instead of returning undefined. */
	requireMesh(id: number): MeshModel {
		const m = this.getMesh(id);
		if (m === undefined) {
			const known = this.meshList.map((x) => x.id()).join(", ");
			throw new MLException(`no mesh with id ${id}${known === "" ? "" : ` (have ${known})`}`);
		}
		return m;
	}

	meshIterator(): readonly MeshModel[] {
		return this.meshList;
	}

	visibleMeshes(): readonly MeshModel[] {
		return this.meshList.filter((m) => m.isVisible());
	}

	setCurrentMesh(id: number): void {
		this.requireMesh(id);
		this.currentId = id;
	}

	addNewMesh(fullPath = "", label = "", setAsCurrent = true, mesh?: CMeshO): MeshModel {
		const m = new MeshModel(this.meshIdCounter++, fullPath, label, mesh);
		this.meshList.push(m);
		if (setAsCurrent || this.currentId < 0) this.currentId = m.id();
		return m;
	}

	/** Returns the layer already loaded from `fullPath`, or adds a new one. */
	addOrGetMesh(fullPath: string, label = "", setAsCurrent = true): MeshModel {
		const hit = this.meshList.find((m) => m.fullName() === fullPath);
		if (hit !== undefined) {
			if (setAsCurrent) this.currentId = hit.id();
			return hit;
		}
		return this.addNewMesh(fullPath, label, setAsCurrent);
	}

	delMesh(id: number): boolean {
		const i = this.meshList.findIndex((m) => m.id() === id);
		if (i < 0) return false;
		this.meshList.splice(i, 1);
		if (this.currentId === id) {
			this.currentId = this.meshList.length > 0 ? this.meshList[0].id() : -1;
		}
		return true;
	}

	// ---- rasters ------------------------------------------------------------

	rasterNumber(): number {
		return this.rasterList.length;
	}

	/**
	 * The current raster, or null. Unlike {@link mm} this does not throw:
	 * a document with no rasters is the normal case, and every raster filter
	 * upstream tests for null rather than assuming one exists.
	 */
	rm(): RasterModel | null {
		if (this.currentRasterId < 0) return null;
		return this.rasterList.find((r) => r.id() === this.currentRasterId) ?? null;
	}

	getRaster(id: number): RasterModel | undefined {
		return this.rasterList.find((r) => r.id() === id);
	}

	rasterIterator(): readonly RasterModel[] {
		return this.rasterList;
	}

	visibleRasters(): readonly RasterModel[] {
		return this.rasterList.filter((r) => r.isVisible());
	}

	setCurrentRaster(id: number): void {
		if (this.getRaster(id) === undefined) {
			const known = this.rasterList.map((r) => r.id()).join(", ");
			throw new MLException(`no raster with id ${id}${known === "" ? "" : ` (have ${known})`}`);
		}
		this.currentRasterId = id;
	}

	addNewRaster(label = "", setAsCurrent = true): RasterModel {
		const r = new RasterModel(this.rasterIdCounter++, label);
		this.rasterList.push(r);
		if (setAsCurrent || this.currentRasterId < 0) this.currentRasterId = r.id();
		return r;
	}

	delRaster(id: number): boolean {
		const i = this.rasterList.findIndex((r) => r.id() === id);
		if (i < 0) return false;
		this.rasterList.splice(i, 1);
		if (this.currentRasterId === id) {
			this.currentRasterId = this.rasterList.length > 0 ? this.rasterList[0].id() : -1;
		}
		return true;
	}

	clear(): void {
		this.meshList = [];
		this.meshIdCounter = 0;
		this.currentId = -1;
		this.rasterList = [];
		this.rasterIdCounter = 0;
		this.currentRasterId = -1;
		this.filterHistory.length = 0;
		this.Log.clear();
	}

	// ---- aggregates ---------------------------------------------------------

	/** Total live vertices across every layer. */
	vn(): number {
		return this.meshList.reduce((a, m) => a + m.cm.vn, 0);
	}

	/** Total live faces across every layer. */
	fn(): number {
		return this.meshList.reduce((a, m) => a + m.cm.fn, 0);
	}

	bbox(): Box3 {
		const b = Box3.empty();
		for (const m of this.meshList) b.addBox(m.cm.bbox);
		return b;
	}

	hasBeenModified(): boolean {
		return this.meshList.some((m) => m.meshModified());
	}

	// ---- document metadata ---------------------------------------------------

	pathName(): string {
		const parts = this._fileName.split(/[\\/]/);
		parts.pop();
		return parts.join("/");
	}

	fileName(): string {
		return this._fileName;
	}

	setFileName(name: string): void {
		this._fileName = name;
	}

	isBusy(): boolean {
		return this._busy;
	}

	setBusy(busy: boolean): void {
		this._busy = busy;
	}
}
