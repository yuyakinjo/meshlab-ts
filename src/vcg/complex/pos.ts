/**
 * `Pos` — VCGLib's half-edge cursor, as a small mutable class over indices.
 *
 * A Pos is a (face, edge, vertex) triple where the vertex is one of the two
 * endpoints of the edge, and the edge is one of the three edges of the face.
 * The three flips each change exactly one component and are involutions, which
 * is what makes mesh traversal expressible as short sequences of them.
 *
 * This is the one place in the kernel where a small object per traversal is
 * the right call: a border walk needs mutable state, and a Pos is three
 * numbers that live for the length of one loop.
 */
import { MLInternalException } from "../../common/utilities/ml_exception.ts";
import type { CMeshO } from "./cmesho.ts";
import { isManifoldEdge } from "./update/topology.ts";

const next3 = (z: number): number => (z + 1) % 3;
const prev3 = (z: number): number => (z + 2) % 3;

export class Pos {
	constructor(
		readonly mesh: CMeshO,
		/** Face index. */
		public f: number,
		/** Edge index within the face, 0..2. */
		public z: number,
		/** Vertex index; must be an endpoint of edge `z`. */
		public v: number,
	) {}

	/** A Pos on edge `z` of face `f`, starting at that edge's first vertex. */
	static onEdge(mesh: CMeshO, f: number, z: number): Pos {
		return new Pos(mesh, f, z, mesh.fv(f, z));
	}

	clone(): Pos {
		return new Pos(this.mesh, this.f, this.z, this.v);
	}

	equals(other: Pos): boolean {
		return this.f === other.f && this.z === other.z && this.v === other.v;
	}

	/** The other endpoint of the current edge. */
	get vFlip(): number {
		const m = this.mesh;
		return m.fv(this.f, next3(this.z)) === this.v
			? m.fv(this.f, this.z)
			: m.fv(this.f, next3(this.z));
	}

	/** True when the current edge has no neighbouring face. */
	isBorder(): boolean {
		return this.mesh.isBorderFF(this.f, this.z);
	}

	/** True when the current edge is shared by at most two faces. */
	isManifold(): boolean {
		return isManifoldEdge(this.mesh, this.f, this.z);
	}

	/** Same edge and vertex, the face on the other side. A no-op on a border. */
	flipF(): void {
		const m = this.mesh;
		const nf = m.ffp(this.f, this.z);
		const nz = m.ffi(this.f, this.z);
		this.f = nf;
		this.z = nz;
	}

	/** Same face and vertex, the other edge of the face incident on that vertex. */
	flipE(): void {
		const m = this.mesh;
		this.z = m.fv(this.f, next3(this.z)) === this.v ? next3(this.z) : prev3(this.z);
	}

	/** Same face and edge, the other endpoint. */
	flipV(): void {
		this.v = this.vFlip;
	}

	/** Rotates to the next face around the current vertex. */
	nextE(): void {
		this.flipE();
		this.flipF();
	}

	/**
	 * Advances one step along a boundary loop.
	 *
	 * Rotates around the current vertex until the next border edge is found,
	 * then steps over it. Requires the current edge to be a border.
	 */
	nextB(): void {
		if (!this.isBorder()) {
			throw new MLInternalException(`Pos.nextB called on a non-border edge (${this.f}, ${this.z})`);
		}
		let guard = 0;
		do {
			this.nextE();
			if (++guard > this.mesh.faceSize * 3 + 3) {
				throw new MLInternalException(
					`Pos.nextB did not find a border rotating around vertex ${this.v}`,
				);
			}
		} while (!this.isBorder());
		this.flipV();
	}
}

/**
 * Walks the boundary loop containing `start`, calling `fn` at each step.
 *
 * `start` must sit on a border edge. The loop terminates when the walk returns
 * to where it began.
 */
export function forEachBorderStep(start: Pos, fn: (p: Pos) => void): void {
	const cur = start.clone();
	do {
		fn(cur);
		cur.nextB();
	} while (!cur.equals(start));
}
