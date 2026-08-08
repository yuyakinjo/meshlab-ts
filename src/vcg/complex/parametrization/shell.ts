/**
 * The shell: the local patch a candidate merge is tried on.
 *
 * Relaxing a seam means moving UVs, and moving UVs means solving an ARAP
 * problem — but not over the whole mesh. The shell is a small mesh carrying
 * just the faces involved, with three things attached to each face:
 *
 * - the **target shape** it should be relaxed toward,
 * - the **3D shape** it has on the surface,
 * - which face of the source mesh it came from, or `-1` if it is filler.
 *
 * The target shape is where the interesting decision is. Relaxing toward the
 * face's *3D* shape would undo the parametrization's deliberate scaling; toward
 * its *current UV* shape would preserve whatever distortion is already there.
 * Upstream interpolates: it takes the singular values of the current
 * parametrization and shrinks only the larger one, capping how many texels a
 * face may claim in its longest direction while leaving a face that is already
 * near-isometric alone. That is reproduced here.
 *
 * A shell must be a disk for ARAP to behave, so every boundary but the longest
 * is filled in. The filler faces are marked, given target shapes of their own,
 * and thrown away afterwards — they exist to stop the patch from having holes
 * that let it fold through itself.
 */
import { MLException } from "../../../common/utilities/ml_exception.ts";
import {
	closestRotation2,
	invert2,
	type Mat2,
	multiply2,
	rotation2,
	svd2,
} from "../../math/mat2.ts";
import { Allocator } from "../allocator.ts";
import { CMeshO } from "../cmesho.ts";
import { FaceFlag } from "../flags.ts";
import { fillHoles, getInfo } from "../hole.ts";
import { faceFace } from "../update/topology.ts";
import { localIsometry, type TargetShapes } from "./arap2d.ts";
import type { AtlasMesh, Chart } from "./chart_graph.ts";

export interface Shell {
	/** A mesh in its own numbering; positions start as the 3D surface's. */
	readonly mesh: CMeshO;
	/** The faces, flattened, for handing straight to `arap2D`. */
	readonly faces: Int32Array;
	readonly vertexCount: number;
	/** Four numbers per face, in `arap2D`'s format. */
	readonly target: TargetShapes;
	/** UV per shell vertex, the thing ARAP moves. */
	readonly uv: Float64Array;
	/** The atlas face each shell face came from, or -1 for hole filling. */
	readonly sourceFace: Int32Array;
	/** The 3D shape of each face, kept so the shell can be put back. */
	readonly shape3D: Float64Array;
	/** True when the chart was a single connected piece to begin with. */
	readonly singleComponent: boolean;
	readonly holeFillingFaces: number;
}

/**
 * The target shape of one face: its current UV triangle with the longer
 * direction shrunk by `downscale`.
 *
 * With `downscale` at 1 this is the current parametrization exactly, so a
 * relaxation with it changes nothing — which is the right meaning for "do not
 * shrink". Below 1 the larger singular value is scaled and the smaller left
 * alone unless it would then exceed it, so a face already close to isometric
 * keeps its shape and a badly stretched one loses only its stretch.
 */
function targetShapeOf(
	p: readonly number[][],
	uv: readonly number[][],
	downscale: number,
): [number, number, number, number] {
	const u10 = [uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]];
	const u20 = [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]];
	const areaUV = Math.abs(u10[0] * u20[1] - u10[1] * u20[0]) / 2;

	if (areaUV === 0) {
		// Nothing to decompose: a zero-area UV triangle has no singular values
		// worth the name, so it is simply scaled.
		return [u10[0] * downscale, u10[1] * downscale, u20[0] * downscale, u20[1] * downscale];
	}

	const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
	const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
	const [x10x, x10y, x20x, x20y] = localIsometry(e1, e2);

	// The map from the flattened 3D triangle to the UV one.
	const x: Mat2 = [x10x, x20x, x10y, x20y];
	const u: Mat2 = [u10[0], u20[0], u10[1], u20[1]];
	const inverse = invert2(x);
	if (inverse === null) return [u10[0], u10[1], u20[0], u20[1]];
	const a = multiply2(u, inverse);

	const { u: phi, v: theta, s0, s1 } = svd2(a);
	const newS0 = s0 * downscale;
	// The sign of the second singular value carries the orientation, and
	// dropping it here would silently un-mirror a flipped chart.
	const magnitude = Math.min(newS0, Math.abs(s1));
	const newS1 = Math.sign(s1 || 1) * magnitude;

	// Rebuild the map with the new singular values and push the 3D triangle
	// through it.
	const uRot = rotation2(phi);
	const vRot = rotation2(theta);
	const generator = multiply2(multiply2(uRot, [newS0, 0, 0, newS1]), [
		vRot[0],
		vRot[2],
		vRot[1],
		vRot[3],
	]);
	return [
		generator[0] * x10x + generator[1] * x10y,
		generator[2] * x10x + generator[3] * x10y,
		generator[0] * x20x + generator[1] * x20y,
		generator[2] * x20x + generator[3] * x20y,
	];
}

export interface ShellOptions {
	/** How much to shrink each face's longest direction. 1 leaves it alone. */
	readonly downscale?: number;
	/** Fill every boundary but the longest, so the shell is a disk. */
	readonly closeHoles?: boolean;
	/** Holes with more than this many edges are left open. */
	readonly maxHoleSize?: number;
}

/**
 * Builds the shell of one chart.
 *
 * The mesh's positions are the surface's, and the UV array starts at the
 * chart's current parametrization — so the shell as returned is a faithful
 * copy, and any change to it is something a caller asked for.
 */
export function buildShell(am: AtlasMesh, chart: Chart, options: ShellOptions = {}): Shell {
	return buildShellFromFaces(am, chart.faces, options);
}

/** The same, over any set of atlas faces — a merge spans two charts. */
export function buildShellFromFaces(
	am: AtlasMesh,
	atlasFaces: readonly number[],
	options: ShellOptions = {},
): Shell {
	if (atlasFaces.length === 0) throw new MLException("a shell needs at least one face");
	const downscale = options.downscale ?? 1;

	// The shell is welded in 3D: the chart's own seams are internal to it and
	// keeping them cut would leave the patch as several pieces.
	const mesh = new CMeshO();
	const slotOf = new Map<string, number>();
	const positions: number[] = [];
	const uvs: number[] = [];
	const corner: number[] = [];

	for (const f of atlasFaces) {
		for (let k = 0; k < 3; k++) {
			const v = am.faces[3 * f + k];
			const id = `${am.positions[3 * v]},${am.positions[3 * v + 1]},${am.positions[3 * v + 2]}`;
			let slot = slotOf.get(id);
			if (slot === undefined) {
				slot = positions.length / 3;
				slotOf.set(id, slot);
				positions.push(am.positions[3 * v], am.positions[3 * v + 1], am.positions[3 * v + 2]);
				uvs.push(am.uv[2 * v], am.uv[2 * v + 1]);
			}
			corner.push(slot);
		}
	}

	Allocator.addVertices(mesh, positions.length / 3);
	for (let v = 0; v < positions.length / 3; v++) {
		mesh.setVert(v, positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
	}
	Allocator.addFaces(mesh, atlasFaces.length);
	for (let f = 0; f < atlasFaces.length; f++) {
		mesh.setFace(f, corner[3 * f], corner[3 * f + 1], corner[3 * f + 2]);
	}
	faceFace(mesh);
	const singleComponent = countComponents(mesh) === 1;

	const originalFaceCount = atlasFaces.length;
	let holeFillingFaces = 0;
	if (options.closeHoles ?? false) {
		holeFillingFaces = closeAllButLongest(mesh, options.maxHoleSize ?? 1000);
	}

	// Target shapes, 3D shapes and provenance, over the final face list.
	const faceCount = mesh.faceSize;
	const faces = new Int32Array(3 * faceCount);
	const target = new Float64Array(4 * faceCount);
	const shape3D = new Float64Array(9 * faceCount);
	const sourceFace = new Int32Array(faceCount).fill(-1);
	const uv = Float64Array.from(uvs.concat(new Array(2 * (mesh.vertSize - uvs.length / 2)).fill(0)));

	// A filler face has no UV of its own; its corners take the flattened 3D
	// shape instead, scaled so it sits alongside the real faces rather than
	// dominating or vanishing from the system.
	const scale = fillerScale(atlasFaces, am, downscale, originalFaceCount);

	for (let f = 0; f < faceCount; f++) {
		const v = [0, 1, 2].map((k) => mesh.fv(f, k));
		for (let k = 0; k < 3; k++) faces[3 * f + k] = v[k];
		const p = v.map((i) => [mesh.vx(i), mesh.vy(i), mesh.vz(i)]);
		for (let k = 0; k < 3; k++) shape3D.set(p[k], 9 * f + 3 * k);

		if (f < originalFaceCount) {
			sourceFace[f] = atlasFaces[f];
			const t = v.map((i) => [uv[2 * i], uv[2 * i + 1]]);
			target.set(targetShapeOf(p, t, downscale), 4 * f);
		} else {
			const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
			const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
			const flat = localIsometry(e1, e2);
			target.set([flat[0] * scale, flat[1] * scale, flat[2] * scale, flat[3] * scale], 4 * f);
		}
	}

	// The filler vertices need somewhere to start; their own 3D positions,
	// flattened by dropping z, is as good a guess as any and ARAP moves them.
	for (let v = uvs.length / 2; v < mesh.vertSize; v++) {
		uv[2 * v] = mesh.vx(v) * scale;
		uv[2 * v + 1] = mesh.vy(v) * scale;
	}

	return {
		mesh,
		faces,
		vertexCount: mesh.vertSize,
		target,
		uv,
		sourceFace,
		shape3D,
		singleComponent,
		holeFillingFaces,
	};
}

/** How much to shrink a filler face so it matches the real ones in size. */
function fillerScale(
	atlasFaces: readonly number[],
	am: AtlasMesh,
	downscale: number,
	originalFaceCount: number,
): number {
	let targetArea = 0;
	let surfaceArea = 0;
	for (let f = 0; f < originalFaceCount && f < atlasFaces.length; f++) {
		const a = atlasFaces[f];
		const v = [0, 1, 2].map((k) => am.faces[3 * a + k]);
		const t = v.map((i) => [am.uv[2 * i], am.uv[2 * i + 1]]);
		targetArea +=
			(Math.abs(
				(t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]),
			) /
				2) *
			downscale *
			downscale;
		const p = v.map((i) => [am.positions[3 * i], am.positions[3 * i + 1], am.positions[3 * i + 2]]);
		const e1 = [0, 1, 2].map((i) => p[1][i] - p[0][i]);
		const e2 = [0, 1, 2].map((i) => p[2][i] - p[0][i]);
		surfaceArea +=
			Math.hypot(
				e1[1] * e2[2] - e1[2] * e2[1],
				e1[2] * e2[0] - e1[0] * e2[2],
				e1[0] * e2[1] - e1[1] * e2[0],
			) / 2;
	}
	if (!(surfaceArea > 0) || !(targetArea > 0)) return 1;
	const scale = Math.sqrt(targetArea / surfaceArea);
	return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

/** Connected components of a mesh, over FF adjacency. */
function countComponents(m: CMeshO): number {
	const seen = new Int32Array(m.faceSize).fill(-1);
	let count = 0;
	for (let start = 0; start < m.faceSize; start++) {
		if (m.isFaceD(start) || seen[start] >= 0) continue;
		const stack = [start];
		seen[start] = count;
		while (stack.length > 0) {
			const f = stack.pop() as number;
			for (let k = 0; k < 3; k++) {
				const g = m.ffp(f, k);
				if (g === f || m.isFaceD(g) || seen[g] >= 0) continue;
				seen[g] = count;
				stack.push(g);
			}
		}
		count++;
	}
	return count;
}

/**
 * Fills every boundary of the shell but the longest, leaving a disk.
 *
 * The longest one is the patch's outer edge and must stay open; the rest are
 * genuine holes that would let ARAP fold the patch through itself.
 */
function closeAllButLongest(mesh: CMeshO, maxHoleSize: number): number {
	const holes = getInfo(mesh);
	if (holes.length <= 1) return 0;

	let longest = 0;
	for (let i = 1; i < holes.length; i++) {
		if (holes[i].size > holes[longest].size) longest = i;
	}

	// Selected explicitly rather than by a size cap. Capping just below the
	// longest loop is the obvious trick and it is wrong: an annulus has two
	// boundaries of equal length, so the cap excludes both and nothing gets
	// filled — silently leaving the shell as a non-disk.
	for (let f = 0; f < mesh.faceSize; f++) mesh.faceFlags[f] &= ~FaceFlag.SELECTED;
	for (let i = 0; i < holes.length; i++) {
		if (i === longest) continue;
		for (const [f] of holes[i].borders) mesh.faceFlags[f] |= FaceFlag.SELECTED;
	}

	const before = mesh.fn;
	fillHoles(mesh, { maxHoleSize, strategy: "minimumWeight", selected: true });
	for (let f = 0; f < mesh.faceSize; f++) mesh.faceFlags[f] &= ~FaceFlag.SELECTED;
	faceFace(mesh);
	return mesh.fn - before;
}

/** Moves the shell's vertex positions onto its UVs, in the z = 0 plane. */
export function syncShellWithUV(shell: Shell): void {
	for (let v = 0; v < shell.vertexCount; v++) {
		shell.mesh.setVert(v, shell.uv[2 * v], shell.uv[2 * v + 1], 0);
	}
}

/** Puts the shell's vertex positions back on the surface. */
export function syncShellWith3D(shell: Shell): void {
	for (let f = 0; f < shell.mesh.faceSize; f++) {
		if (shell.mesh.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			const v = shell.mesh.fv(f, k);
			shell.mesh.setVert(
				v,
				shell.shape3D[9 * f + 3 * k],
				shell.shape3D[9 * f + 3 * k + 1],
				shell.shape3D[9 * f + 3 * k + 2],
			);
		}
	}
}

/**
 * The shell's boundary vertices, at their current UVs — what a merge pins.
 *
 * Everything outside the patch stays where it is, so the patch's own rim is
 * fixed and only its inside is free to move.
 */
export function boundaryPins(shell: Shell): Map<number, readonly [number, number]> {
	const pins = new Map<number, readonly [number, number]>();
	for (let f = 0; f < shell.mesh.faceSize; f++) {
		if (shell.mesh.isFaceD(f)) continue;
		for (let k = 0; k < 3; k++) {
			if (!shell.mesh.isBorderFF(f, k)) continue;
			for (const v of [shell.mesh.fv(f, k), shell.mesh.fv(f, (k + 1) % 3)]) {
				pins.set(v, [shell.uv[2 * v], shell.uv[2 * v + 1]]);
			}
		}
	}
	return pins;
}

/** The rotation carrying a shell face's target shape onto its current UVs. */
export function faceRotation(shell: Shell, f: number): Mat2 {
	const v = [0, 1, 2].map((k) => shell.faces[3 * f + k]);
	const x: Mat2 = [
		shell.target[4 * f],
		shell.target[4 * f + 2],
		shell.target[4 * f + 1],
		shell.target[4 * f + 3],
	];
	const u: Mat2 = [
		shell.uv[2 * v[1]] - shell.uv[2 * v[0]],
		shell.uv[2 * v[2]] - shell.uv[2 * v[0]],
		shell.uv[2 * v[1] + 1] - shell.uv[2 * v[0] + 1],
		shell.uv[2 * v[2] + 1] - shell.uv[2 * v[0] + 1],
	];
	const inverse = invert2(x);
	if (inverse === null) return [1, 0, 0, 1];
	return closestRotation2(multiply2(u, inverse));
}

export const ShellOps = {
	buildShell,
	buildShellFromFaces,
	syncShellWithUV,
	syncShellWith3D,
	boundaryPins,
	faceRotation,
} as const;
