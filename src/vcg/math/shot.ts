/**
 * `vcg::Shot` — a camera: where it is, where it points, and how it projects.
 *
 * A shot is split the way photogrammetry splits it. `Intrinsics` is the
 * camera itself — focal length, sensor pitch, image size — and does not
 * change when the camera moves. `Extrinsics` is the pose, and in VCG's
 * convention `tra` is the view point in world coordinates rather than the
 * usual `-R·t` translation of a world-to-camera matrix. That choice is worth
 * knowing about, because every file format below stores the other one, and
 * the conversions in `cameras.ts` exist entirely to bridge the two.
 *
 * Lens distortion is carried but never applied: MeshLab itself refuses to
 * import non-zero distortion coefficients and asks for undistorted images
 * instead, so storing them and silently ignoring them would be worse than
 * not storing them at all.
 */

import { identity, type Matrix44 } from "./matrix44.ts";

/** The intrinsic parameters — VCG's `vcg::Camera`. */
export class Camera {
	/** Focal length in millimetres. */
	FocalMm = 1;
	/** Sensor pitch in millimetres, horizontal then vertical. */
	PixelSizeMm: [number, number] = [1, 1];
	/** Image size in pixels. */
	ViewportPx: [number, number] = [0, 0];
	/** Principal point in pixels, normally half the viewport. */
	CenterPx: [number, number] = [0, 0];
	/** Radial and tangential distortion, kept but never applied. */
	k: [number, number, number, number] = [0, 0, 0, 0];

	/** Focal length in pixels along x — what most formats actually store. */
	focalPxX(): number {
		return this.FocalMm / this.PixelSizeMm[0];
	}

	focalPxY(): number {
		return this.FocalMm / this.PixelSizeMm[1];
	}

	/** Centre the principal point on the current viewport. */
	centreOnViewport(): void {
		this.CenterPx = [this.ViewportPx[0] / 2, this.ViewportPx[1] / 2];
	}

	clone(): Camera {
		const c = new Camera();
		c.FocalMm = this.FocalMm;
		c.PixelSizeMm = [...this.PixelSizeMm];
		c.ViewportPx = [...this.ViewportPx];
		c.CenterPx = [...this.CenterPx];
		c.k = [...this.k];
		return c;
	}
}

/**
 * The extrinsic parameters — VCG's `vcg::Similarity` restricted to a rigid
 * motion.
 *
 * `rot` maps world directions into camera directions; its rows are the
 * camera's own axes expressed in world coordinates.
 */
export class Extrinsics {
	rot: Matrix44 = identity();
	tra: [number, number, number] = [0, 0, 0];

	Rot(): Matrix44 {
		return this.rot;
	}

	SetRot(m: Matrix44): void {
		this.rot = m.slice() as Matrix44;
	}

	Tra(): readonly [number, number, number] {
		return this.tra;
	}

	SetTra(t: readonly [number, number, number]): void {
		this.tra = [t[0], t[1], t[2]];
	}

	clone(): Extrinsics {
		const e = new Extrinsics();
		e.rot = this.rot.slice() as Matrix44;
		e.tra = [...this.tra];
		return e;
	}
}

export class Shot {
	readonly Intrinsics = new Camera();
	readonly Extrinsics = new Extrinsics();

	/** The i-th camera axis in world coordinates: row i of the rotation. */
	Axis(i: number): [number, number, number] {
		const m = this.Extrinsics.rot;
		return [m[4 * i], m[4 * i + 1], m[4 * i + 2]];
	}

	/** The direction the camera looks along. */
	GetViewDir(): [number, number, number] {
		return this.Axis(2);
	}

	GetViewPoint(): readonly [number, number, number] {
		return this.Extrinsics.tra;
	}

	SetViewPoint(p: readonly [number, number, number]): void {
		this.Extrinsics.SetTra(p);
	}

	/** Vertical field of view in degrees, from the focal length. */
	GetFovFromFocal(): number {
		const viewportYMm = this.Intrinsics.PixelSizeMm[1] * this.Intrinsics.ViewportPx[1];
		return 2 * ((Math.atan(viewportYMm / (2 * this.Intrinsics.FocalMm)) * 180) / Math.PI);
	}

	clone(): Shot {
		const s = new Shot();
		Object.assign(s.Intrinsics, this.Intrinsics.clone());
		Object.assign(s.Extrinsics, this.Extrinsics.clone());
		return s;
	}
}
