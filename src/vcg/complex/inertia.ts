/**
 * `Inertia` — volume, centre of mass and inertia tensor of the solid a closed
 * mesh bounds, mirroring `vcg::tri::Inertia`.
 *
 * Everything here is the divergence theorem: a volume integral over the solid
 * becomes a surface integral over its triangles, so quantities that sound like
 * they need the interior are computed from the boundary alone. All of them
 * are exact for a closed, coherently oriented mesh — no sampling, no
 * approximation — and meaningless for an open one, which is why the filter
 * checks watertightness before reporting them.
 */
import type { CMeshO } from "./cmesho.ts";

export interface MassProperties {
	/** Signed volume; positive when the winding faces outward. */
	volume: number;
	/** Centre of mass of the enclosed solid, assuming uniform density. */
	centerOfMass: [number, number, number];
	/** Inertia tensor about the centre of mass, row-major 3×3, unit density. */
	inertiaTensor: number[];
	/** Barycentre of the *surface*, weighted by triangle area. */
	shellBarycenter: [number, number, number];
	/** Total surface area. */
	area: number;
}

/**
 * Computes the mass properties of the solid bounded by `m`.
 *
 * Uses the divergence-theorem formulation over tetrahedra from the origin to
 * each triangle. The signed contributions of tetrahedra outside the solid
 * cancel exactly, so the origin's position does not matter.
 */
export function computeMassProperties(m: CMeshO): MassProperties {
	let volume = 0;
	let cx = 0;
	let cy = 0;
	let cz = 0;
	let area = 0;
	let shellX = 0;
	let shellY = 0;
	let shellZ = 0;

	// Second moments, accumulated about the origin and shifted at the end.
	let ixx = 0;
	let iyy = 0;
	let izz = 0;
	let ixy = 0;
	let iyz = 0;
	let izx = 0;

	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		const ia = m.fv(f, 0);
		const ib = m.fv(f, 1);
		const ic = m.fv(f, 2);
		const ax = m.vx(ia);
		const ay = m.vy(ia);
		const az = m.vz(ia);
		const bx = m.vx(ib);
		const by = m.vy(ib);
		const bz = m.vz(ib);
		const gx = m.vx(ic);
		const gy = m.vy(ic);
		const gz = m.vz(ic);

		// Six times the signed volume of the tetrahedron (origin, a, b, c).
		const det = ax * (by * gz - bz * gy) - ay * (bx * gz - bz * gx) + az * (bx * gy - by * gx);
		const tetVolume = det / 6;
		volume += tetVolume;

		// The tetrahedron's centroid is the average of its four corners, one
		// of which is the origin.
		cx += tetVolume * ((ax + bx + gx) / 4);
		cy += tetVolume * ((ay + by + gy) / 4);
		cz += tetVolume * ((az + bz + gz) / 4);

		// Surface quantities.
		const ux = bx - ax;
		const uy = by - ay;
		const uz = bz - az;
		const vx = gx - ax;
		const vy = gy - ay;
		const vz = gz - az;
		const nx = uy * vz - uz * vy;
		const ny = uz * vx - ux * vz;
		const nz = ux * vy - uy * vx;
		const triArea = Math.hypot(nx, ny, nz) / 2;
		area += triArea;
		shellX += triArea * ((ax + bx + gx) / 3);
		shellY += triArea * ((ay + by + gy) / 3);
		shellZ += triArea * ((az + bz + gz) / 3);

		// Second moments of the tetrahedron about the origin. For a
		// tetrahedron with one vertex at the origin the integral of x² is
		// V/10 · Σᵢ Σⱼ≥ᵢ xᵢxⱼ over the three other corners, and the products
		// of inertia follow the analogous symmetric form.
		const sxx = ax * ax + bx * bx + gx * gx + ax * bx + bx * gx + gx * ax;
		const syy = ay * ay + by * by + gy * gy + ay * by + by * gy + gy * ay;
		const szz = az * az + bz * bz + gz * gz + az * bz + bz * gz + gz * az;
		ixx += tetVolume * sxx;
		iyy += tetVolume * syy;
		izz += tetVolume * szz;

		const sxy =
			2 * (ax * ay + bx * by + gx * gy) + ax * by + bx * ay + bx * gy + gx * by + gx * ay + ax * gy;
		const syz =
			2 * (ay * az + by * bz + gy * gz) + ay * bz + by * az + by * gz + gy * bz + gy * az + ay * gz;
		const szx =
			2 * (az * ax + bz * bx + gz * gx) + az * bx + bz * ax + bz * gx + gz * bx + gz * ax + az * gx;
		ixy += tetVolume * sxy;
		iyz += tetVolume * syz;
		izx += tetVolume * szx;
	}

	const com: [number, number, number] =
		volume === 0 ? [0, 0, 0] : [cx / volume, cy / volume, cz / volume];

	// Scale the raw moments, then shift from the origin to the centre of mass
	// with the parallel-axis theorem.
	ixx /= 10;
	iyy /= 10;
	izz /= 10;
	ixy /= 20;
	iyz /= 20;
	izx /= 20;

	const [mx, my, mz] = com;
	const jxx = iyy + izz - volume * (my * my + mz * mz);
	const jyy = izz + ixx - volume * (mz * mz + mx * mx);
	const jzz = ixx + iyy - volume * (mx * mx + my * my);
	const jxy = -(ixy - volume * mx * my);
	const jyz = -(iyz - volume * my * mz);
	const jzx = -(izx - volume * mz * mx);

	return {
		volume,
		centerOfMass: com,
		inertiaTensor: [jxx, jxy, jzx, jxy, jyy, jyz, jzx, jyz, jzz],
		shellBarycenter: area === 0 ? [0, 0, 0] : [shellX / area, shellY / area, shellZ / area],
		area,
	};
}

/** The barycentre of the vertices, unweighted. */
export function vertexBarycenter(m: CMeshO): [number, number, number] {
	let x = 0;
	let y = 0;
	let z = 0;
	let n = 0;
	for (let v = 0; v < m.vertSize; v++) {
		if (m.isVertD(v)) continue;
		x += m.vx(v);
		y += m.vy(v);
		z += m.vz(v);
		n++;
	}
	return n === 0 ? [0, 0, 0] : [x / n, y / n, z / n];
}

/** Statistics over the mesh's distinct edges. */
export function edgeLengthStats(m: CMeshO): { total: number; average: number; count: number } {
	const seen = new Set<string>();
	let total = 0;
	for (let f = 0; f < m.faceSize; f++) {
		if (m.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = m.faceVert[3 * f + e];
			const b = m.faceVert[3 * f + ((e + 1) % 3)];
			const key = a < b ? `${a}_${b}` : `${b}_${a}`;
			if (seen.has(key)) continue;
			seen.add(key);
			total += Math.hypot(m.vx(b) - m.vx(a), m.vy(b) - m.vy(a), m.vz(b) - m.vz(a));
		}
	}
	return { total, average: seen.size === 0 ? 0 : total / seen.size, count: seen.size };
}

export const Inertia = {
	computeMassProperties,
	vertexBarycenter,
	edgeLengthStats,
} as const;
