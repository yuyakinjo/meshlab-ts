/**
 * Camera interchange: Bundler `.out` and Agisoft Photoscan `.xml`.
 *
 * Both formats store a world-to-camera pose, while {@link Shot} stores the
 * view point in world coordinates, so both directions go through the same
 * conversion. For Bundler:
 *
 *     file rotation R   = Extrinsics.rot                  (world -> camera)
 *     file translation  = -Rᵀ_as_read · viewpoint
 *
 * and the readers invert it. The transposes below look redundant written out,
 * but they are what upstream does and getting one of them backwards produces
 * cameras that are plausibly placed and silently mirrored, which is much worse
 * than an obvious failure.
 *
 * Neither reader applies lens distortion. MeshLab warns and drops non-zero
 * coefficients, asking for undistorted images instead; this does the same,
 * through the `warnings` array on the result rather than a log side effect.
 */

import type { RasterModel } from "../../common/ml_document/raster_model.ts";
import { MLException } from "../../common/utilities/ml_exception.ts";
import { identity, type Matrix44 } from "../../vcg/math/matrix44.ts";
import { Shot } from "../../vcg/math/shot.ts";

/** Multiplies the transpose of the 3x3 block of `m` by a vector. */
function mulTranspose3(m: Matrix44, v: readonly number[]): [number, number, number] {
	return [
		m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
		m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
		m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
	];
}

/** Multiplies the 3x3 block of `m` by a vector. */
function mul3(m: Matrix44, v: readonly number[]): [number, number, number] {
	return [
		m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
		m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
		m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
	];
}

// ---- Bundler .out ---------------------------------------------------------

const BUNDLE_HEADER = "# Bundle file v0.3";

export function writeBundlerOut(shots: readonly Shot[]): string {
	const out: string[] = [BUNDLE_HEADER, `${shots.length} 0`];
	for (const shot of shots) {
		out.push(`${fmt(shot.Intrinsics.focalPxX())} 0 0`);
		const rot = shot.Extrinsics.rot;
		// The stored translation is R applied to the view point, negated.
		const pos = mul3(rot, shot.Extrinsics.tra);
		// One line per row of R. Upstream arrives here by transposing and then
		// reading columns, which is the same thing written twice.
		for (let r = 0; r < 3; r++) {
			out.push(`${fmt(rot[4 * r])} ${fmt(rot[4 * r + 1])} ${fmt(rot[4 * r + 2])}`);
		}
		out.push(`${fmt(-pos[0])} ${fmt(-pos[1])} ${fmt(-pos[2])}`);
	}
	out.push("0 0 0");
	return `${out.join("\n")}\n`;
}

/**
 * Reads a Bundle file, returning one shot per camera.
 *
 * Viewport and principal point are left at zero: Bundler stores neither, and
 * the caller is expected to fill them from the image, as MeshLab does.
 */
export function readBundlerOut(text: string): Shot[] {
	const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
	if (lines.length === 0 || !lines[0].startsWith(BUNDLE_HEADER)) {
		throw new MLException(`not a Bundler file: expected a "${BUNDLE_HEADER}" header`);
	}
	const counts = numbers(lines[1]);
	if (counts.length < 1) throw new MLException("the Bundler header has no camera count");
	const cameraCount = counts[0];

	const shots: Shot[] = [];
	let at = 2;
	for (let i = 0; i < cameraCount; i++) {
		// No length pre-check: `expect` below names the line that is missing,
		// which is more use than a count that ran out somewhere.
		const [focal] = expect(lines[at++], 3, `camera ${i} intrinsics`);
		const rot = identity();
		for (let r = 0; r < 3; r++) {
			const row = expect(lines[at++], 3, `camera ${i} rotation row ${r}`);
			rot[4 * r] = row[0];
			rot[4 * r + 1] = row[1];
			rot[4 * r + 2] = row[2];
		}
		const t = expect(lines[at++], 3, `camera ${i} translation`);

		const shot = new Shot();
		shot.Extrinsics.SetRot(rot);
		const pos = mulTranspose3(rot, t);
		shot.Extrinsics.SetTra([-pos[0], -pos[1], -pos[2]]);
		// Bundler's focal is in pixels, so a unit pitch makes the millimetre
		// focal numerically equal to it. Any other pitch would be invented.
		shot.Intrinsics.FocalMm = focal;
		shot.Intrinsics.PixelSizeMm = [1, 1];
		shots.push(shot);
	}
	return shots;
}

// ---- Agisoft .xml ---------------------------------------------------------

export interface AgisoftCamera {
	readonly label: string;
	readonly shot: Shot;
}

export function writeAgisoftXml(cameras: readonly AgisoftCamera[]): string {
	const out: string[] = ['<?xml version="1.0" encoding="UTF-8"?>', '<document version="1.2.0">'];
	out.push("  <chunk>");
	out.push("    <sensors>");
	cameras.forEach(({ shot }, i) => {
		const int = shot.Intrinsics;
		// Upstream rescales an implausibly long focal so that Photoscan reads
		// it as millimetres rather than as pixels; the ratio is preserved, so
		// the projection is unchanged.
		const big = int.FocalMm > 1000;
		const focal = big ? int.FocalMm / 500 : int.FocalMm;
		const pixelX = big ? int.PixelSizeMm[0] / 500 : int.PixelSizeMm[0];
		const pixelY = big ? int.PixelSizeMm[1] / 500 : int.PixelSizeMm[1];
		out.push(`      <sensor id="${i}" label="unknown${i}" type="frame">`);
		out.push(`        <resolution width="${int.ViewportPx[0]}" height="${int.ViewportPx[1]}"/>`);
		out.push(`        <property name="pixel_width" value="${fmt(pixelX)}"/>`);
		out.push(`        <property name="pixel_height" value="${fmt(pixelY)}"/>`);
		out.push(`        <property name="focal_length" value="${fmt(focal)}"/>`);
		out.push('        <property name="fixed" value="false"/>');
		out.push('        <calibration type="frame" class="adjusted">');
		out.push(`          <resolution width="${int.ViewportPx[0]}" height="${int.ViewportPx[1]}"/>`);
		out.push(`          <fx>${fmt(int.focalPxX())}</fx>`);
		out.push(`          <fy>${fmt(int.focalPxY())}</fy>`);
		out.push(`          <cx>${fmt(int.CenterPx[0])}</cx>`);
		out.push(`          <cy>${fmt(int.CenterPx[1])}</cy>`);
		out.push("          <k1>0</k1>");
		out.push("          <k2>0</k2>");
		out.push("          <p1>0</p1>");
		out.push("          <p2>0</p2>");
		out.push("        </calibration>");
		out.push("      </sensor>");
	});
	out.push("    </sensors>");
	out.push("    <cameras>");
	cameras.forEach(({ label, shot }, i) => {
		const m = shot.Extrinsics.rot;
		const p = shot.Extrinsics.tra;
		// Photoscan's y and z axes point the other way, so rows 1 and 2 of the
		// rotation are negated on the way out and again on the way back in.
		const t = [
			m[0],
			-m[4],
			-m[8],
			p[0],
			m[1],
			-m[5],
			-m[9],
			p[1],
			m[2],
			-m[6],
			-m[10],
			p[2],
			0,
			0,
			0,
			1,
		];
		out.push(
			`      <camera id="${i}" label="${escapeXml(label)}" sensor_id="${i}" enabled="true">`,
		);
		out.push(`        <transform>${t.map(fmt).join(" ")}</transform>`);
		out.push("      </camera>");
	});
	out.push("    </cameras>");
	out.push("  </chunk>");
	out.push("</document>");
	return `${out.join("\n")}\n`;
}

export interface AgisoftImport {
	readonly cameras: readonly AgisoftCamera[];
	readonly warnings: readonly string[];
}

/**
 * Reads an Agisoft chunk.
 *
 * The parsing is deliberately shallow — a regex sweep over the elements that
 * matter rather than a DOM. A camera file is a flat list of sensors and
 * cameras with no mixed content and no namespaces, and a general XML parser
 * would be a much larger dependency than the format warrants.
 */
export function readAgisoftXml(text: string): AgisoftImport {
	const warnings: string[] = [];
	const sensors = new Map<number, Shot>();

	for (const block of matchAll(text, /<sensor\b([^>]*)>([\s\S]*?)<\/sensor>/g)) {
		const id = Number(attr(block[1], "id") ?? sensors.size);
		const shot = new Shot();
		const calibration = /<calibration\b[^>]*>([\s\S]*?)<\/calibration>/.exec(block[2]);
		const body = calibration === null ? block[2] : calibration[1];

		const resolution = /<resolution\b([^>]*)\/?>/.exec(body);
		if (resolution !== null) {
			const w = Number(attr(resolution[1], "width") ?? 0);
			const h = Number(attr(resolution[1], "height") ?? 0);
			shot.Intrinsics.ViewportPx = [w, h];
			shot.Intrinsics.centreOnViewport();
		}
		const fx = tagNumber(body, "fx");
		if (fx !== null) {
			// A focal above 100 cannot be millimetres for any real sensor, so
			// it is pixels; upstream converts with a 0.01 mm pitch.
			if (fx > 100) {
				shot.Intrinsics.FocalMm = fx / 100;
				shot.Intrinsics.PixelSizeMm = [0.01, 0.01];
			} else {
				shot.Intrinsics.FocalMm = fx;
				shot.Intrinsics.PixelSizeMm = [1, 1];
			}
		}
		const cx = tagNumber(body, "cx");
		const cy = tagNumber(body, "cy");
		if (cx !== null && cy !== null) shot.Intrinsics.CenterPx = [cx, cy];
		for (const k of ["k1", "k2", "p1", "p2"]) {
			const value = tagNumber(body, k);
			if (value !== null && value !== 0) {
				warnings.push(
					`sensor ${id} has a non-zero ${k}; distortion is not imported, so undistort the images first`,
				);
			}
		}
		sensors.set(id, shot);
	}

	if (sensors.size === 0) throw new MLException("the Agisoft file declares no sensors");

	const cameras: AgisoftCamera[] = [];
	for (const block of matchAll(text, /<camera\b([^>]*)>([\s\S]*?)<\/camera>/g)) {
		const label = unescapeXml(attr(block[1], "label") ?? "");
		const sensorId = Number(attr(block[1], "sensor_id") ?? 0);
		const transform = /<transform>([\s\S]*?)<\/transform>/.exec(block[2]);
		if (transform === null) continue;
		const v = numbers(transform[1]);
		if (v.length < 12) {
			throw new MLException(`camera "${label}" has a transform with only ${v.length} values`);
		}

		const source = sensors.get(sensorId);
		if (source === undefined) {
			throw new MLException(`camera "${label}" refers to sensor ${sensorId}, which is not defined`);
		}
		const shot = source.clone();
		const m = identity();
		m[0] = v[0];
		m[4] = -v[1];
		m[8] = -v[2];
		m[1] = v[4];
		m[5] = -v[5];
		m[9] = -v[6];
		m[2] = v[8];
		m[6] = -v[9];
		m[10] = -v[10];
		shot.Extrinsics.SetRot(m);
		shot.Extrinsics.SetTra([v[3], v[7], v[11]]);
		cameras.push({ label, shot });
	}

	if (cameras.length === 0) throw new MLException("the Agisoft file declares no cameras");
	return { cameras, warnings };
}

// ---- shared ---------------------------------------------------------------

/**
 * Copies a shot onto a raster without replacing the raster's viewport when
 * the incoming one is unknown, which is the Bundler case.
 */
export function applyShot(raster: RasterModel, shot: Shot): void {
	const dst = raster.shot;
	const keepViewport = shot.Intrinsics.ViewportPx[0] === 0 && shot.Intrinsics.ViewportPx[1] === 0;
	const viewport = keepViewport ? dst.Intrinsics.ViewportPx : shot.Intrinsics.ViewportPx;
	const centre = keepViewport ? dst.Intrinsics.CenterPx : shot.Intrinsics.CenterPx;
	dst.Intrinsics.FocalMm = shot.Intrinsics.FocalMm;
	dst.Intrinsics.PixelSizeMm = [...shot.Intrinsics.PixelSizeMm];
	dst.Intrinsics.ViewportPx = [...viewport];
	dst.Intrinsics.CenterPx = [...centre];
	dst.Intrinsics.k = [0, 0, 0, 0];
	dst.Extrinsics.SetRot(shot.Extrinsics.rot);
	dst.Extrinsics.SetTra(shot.Extrinsics.tra);
}

function fmt(x: number): string {
	// Enough digits that a double survives the round trip, without the long
	// tails that a plain toString gives for values like 0.1 + 0.2.
	return Number.isInteger(x) ? String(x) : String(Number(x.toPrecision(9)));
}

function numbers(line: string): number[] {
	const found = line.trim().match(/-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g);
	return found === null ? [] : found.map(Number);
}

function expect(line: string, count: number, what: string): number[] {
	const v = numbers(line ?? "");
	if (v.length < count) {
		throw new MLException(`${what}: expected ${count} numbers, got "${(line ?? "").trim()}"`);
	}
	return v;
}

function attr(attributes: string, name: string): string | null {
	const m = new RegExp(`\\b${name}\\s*=\\s*"([^"]*)"`).exec(attributes);
	return m === null ? null : m[1];
}

function tagNumber(body: string, tag: string): number | null {
	const m = new RegExp(`<${tag}>([^<]*)</${tag}>`).exec(body);
	if (m === null) return null;
	const value = Number(m[1].trim());
	return Number.isFinite(value) ? value : null;
}

function matchAll(text: string, re: RegExp): RegExpExecArray[] {
	const out: RegExpExecArray[] = [];
	let m = re.exec(text);
	while (m !== null) {
		out.push(m);
		m = re.exec(text);
	}
	return out;
}

function escapeXml(s: string): string {
	return s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

function unescapeXml(s: string): string {
	return s
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&amp;/g, "&");
}
