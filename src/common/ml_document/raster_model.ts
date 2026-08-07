/**
 * `RasterModel` — a photograph registered against the meshes in a document.
 *
 * A raster is an image plus the {@link Shot} that says where the camera was
 * when it was taken. MeshLab uses them to project colour onto geometry and to
 * drive image-based alignment; here they exist so that the camera filters have
 * something to read and write, and so a document round-trips through a project
 * file without losing its cameras.
 *
 * The image data itself is not decoded. Nothing in this library looks at
 * pixels yet, and pulling in a JPEG decoder to store bytes nobody reads would
 * be a poor trade. What is parsed is the size — see {@link imageSizeOf} — since
 * the viewport in pixels is a genuine camera parameter and the Bundler format
 * expects the reader to recover it from the image.
 */

import { Shot } from "../../vcg/math/shot.ts";

/**
 * One image belonging to a raster. MeshLab allows several — a colour plane, a
 * depth plane, a mask — of which one is current.
 */
export class RasterPlane {
	readonly fullPathFileName: string;
	readonly semantic: string;

	constructor(fullPathFileName: string, semantic = "RGBA") {
		this.fullPathFileName = fullPathFileName;
		this.semantic = semantic;
	}

	shortName(): string {
		return this.fullPathFileName.split(/[\\/]/).pop() ?? "";
	}
}

export class RasterModel {
	readonly shot = new Shot();
	readonly planeList: RasterPlane[] = [];
	currentPlane: RasterPlane | null = null;

	private readonly _id: number;
	private _label: string;
	private _visible = true;

	constructor(id: number, label = "") {
		this._id = id;
		this._label = label;
	}

	id(): number {
		return this._id;
	}

	/** The display name, falling back to the current plane's file name. */
	label(): string {
		if (this._label !== "") return this._label;
		return this.currentPlane?.shortName() ?? "";
	}

	setLabel(label: string): void {
		this._label = label;
	}

	/**
	 * Visibility doubles as "selected" for the raster filters, exactly as it
	 * does upstream: `delete_non_active_rasters` deletes the invisible ones.
	 */
	isVisible(): boolean {
		return this._visible;
	}

	setVisible(visible: boolean): void {
		this._visible = visible;
	}

	addPlane(plane: RasterPlane, setAsCurrent = true): RasterPlane {
		this.planeList.push(plane);
		if (setAsCurrent || this.currentPlane === null) this.currentPlane = plane;
		return plane;
	}
}

/**
 * The pixel size of a PNG or JPEG, read from its header alone.
 *
 * Returns null when the file is missing or in a format this does not know,
 * which callers treat as "leave the viewport as it is" rather than as an
 * error: a camera file is still worth importing when the images are not
 * beside it.
 */
export function imageSizeOf(bytes: Uint8Array): [number, number] | null {
	return pngSize(bytes) ?? jpegSize(bytes) ?? null;
}

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function pngSize(b: Uint8Array): [number, number] | null {
	if (b.length < 24) return null;
	for (let i = 0; i < PNG_MAGIC.length; i++) if (b[i] !== PNG_MAGIC[i]) return null;
	// IHDR is required to be the first chunk, and its width and height are the
	// first two big-endian 32-bit fields of its payload.
	const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
	return [view.getUint32(16), view.getUint32(20)];
}

function jpegSize(b: Uint8Array): [number, number] | null {
	if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
	const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
	let i = 2;
	while (i + 3 < b.length) {
		if (b[i] !== 0xff) {
			i++; // fill byte or padding; markers are allowed to be preceded by them
			continue;
		}
		const marker = b[i + 1];
		if (marker === 0xff) {
			i++;
			continue;
		}
		// Standalone markers carry no length field.
		if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
			i += 2;
			continue;
		}
		const length = view.getUint16(i + 2);
		// Any SOFn. The 0xc0..0xcf block also holds three markers that are not
		// frame headers at all — DHT, the reserved JPG, and DAC — and reading
		// a size out of those would give nonsense.
		const isFrame =
			marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
		if (isFrame) {
			if (i + 9 > b.length) return null;
			return [view.getUint16(i + 7), view.getUint16(i + 5)];
		}
		if (length < 2) return null;
		i += 2 + length;
	}
	return null;
}
