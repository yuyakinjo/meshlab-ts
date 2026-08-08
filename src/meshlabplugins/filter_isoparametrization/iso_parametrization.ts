/**
 * The isoparametrisation itself: a coarse domain plus the maps to and from
 * the surface it parametrises.
 *
 * {@link AbstractDomain} does the hard half — simplifying while keeping every
 * original vertex pinned. What is left is to use it: sample the domain
 * uniformly and get points on the surface (remeshing), or take a surface
 * point and get a coordinate in the domain (atlasing, transfer).
 *
 * **The inverse map is a projection, not an exact inversion.** Upstream keeps
 * the fine triangulation restricted to each domain face and inverts the
 * barycentric map through it exactly; that machinery is most of
 * `iso_parametrization.h`'s two thousand lines. Here a domain sample is
 * evaluated on the coarse domain and then projected onto the nearest point of
 * the original surface. The result lands on the surface and is uniform in
 * domain space, which is what the remeshing wants, but it is not
 * bit-comparable with MeshLab and it inherits {@link SurfaceLookup}'s blind
 * spot for long thin triangles. The trade is deliberate and is the reason
 * this file is short.
 */

import { MeshElement } from "../../common/ml_document/mesh_element.ts";
import { MLException } from "../../common/utilities/ml_exception.ts";
import { Allocator } from "../../vcg/complex/allocator.ts";
import { Clean } from "../../vcg/complex/clean.ts";
import { CMeshO } from "../../vcg/complex/cmesho.ts";
import { enableChannels } from "../../vcg/complex/components.ts";
import {
	AbstractDomain,
	collapseWithParametrization,
	domainVertexFaces,
	type PinnedVertex,
} from "../../vcg/complex/parametrization/abstract_domain.ts";
import { UpdateBounding } from "../../vcg/complex/update/bounding.ts";
import { SurfaceLookup } from "../../vcg/space/index/surface_lookup.ts";

export interface BuildOptions {
	/** Stop once the domain has at most this many faces. */
	readonly targetMaxFaces: number;
	/** Never go below this many, even if collapses are still available. */
	readonly targetMinFaces: number;
	readonly onProgress?: (fraction: number) => void;
}

/** A location on the surface, expressed in the domain. */
export interface DomainPoint {
	readonly face: number;
	readonly bary: readonly [number, number, number];
}

export class IsoParametrization {
	readonly domain: AbstractDomain;
	private readonly lookup: SurfaceLookup;
	/** Where each original vertex sits in the domain. */
	private readonly byVertex = new Map<number, DomainPoint>();

	constructor(domain: AbstractDomain) {
		this.domain = domain;
		UpdateBounding.box(domain.hires);
		this.lookup = new SurfaceLookup(domain.hires, domain.hires.bbox.diagonal || 1);
		for (let f = 0; f < domain.base.faceSize; f++) {
			if (domain.base.isFaceD(f)) continue;
			for (const pin of domain.pinned[f]) {
				this.byVertex.set(pin.vertex, { face: f, bary: pin.bary });
			}
		}
	}

	/**
	 * Simplifies a mesh into a domain of roughly the requested size.
	 *
	 * Shortest edge first, and every collapse the domain refuses is simply
	 * skipped — the refusals are what keep the parametrisation valid, so a
	 * run that stops above the target has hit the mesh's real limit rather
	 * than failed.
	 */
	static build(hires: CMeshO, options: BuildOptions): IsoParametrization {
		const { targetMaxFaces, targetMinFaces } = options;
		if (targetMinFaces < 4) {
			throw new MLException(`the domain needs at least 4 faces, was asked for ${targetMinFaces}`);
		}
		if (targetMaxFaces < targetMinFaces) {
			throw new MLException(`the face range is inverted: ${targetMinFaces} to ${targetMaxFaces}`);
		}
		if (hires.fn === 0) throw new MLException("the mesh has no faces to parametrise");

		const domain = AbstractDomain.from(hires);
		const cm = domain.base;
		const vertFaces = domainVertexFaces(domain);
		const startFaces = cm.fn;

		// Refused edges are remembered so the sweep does not retry them every
		// round; a collapse elsewhere can make one viable again, so the set is
		// cleared whenever anything succeeds.
		let refused = new Set<string>();
		while (cm.fn > targetMaxFaces) {
			const candidates = edgesByLength(cm, refused);
			if (candidates.length === 0) break;

			let progressed = false;
			for (const { a, b, key } of candidates) {
				if (cm.fn <= targetMinFaces) break;
				if (cm.isVertD(a) || cm.isVertD(b)) continue;
				if (collapseWithParametrization(domain, a, b, vertFaces).ok) {
					progressed = true;
					refused = new Set();
					options.onProgress?.(
						Math.min(1, (startFaces - cm.fn) / Math.max(1, startFaces - targetMaxFaces)),
					);
					break;
				}
				refused.add(key);
			}
			if (!progressed) break;
		}
		return new IsoParametrization(domain);
	}

	/** The domain face count, which is what the filter reports. */
	get faceCount(): number {
		return this.domain.base.fn;
	}

	/** Where an original vertex lives in the domain. */
	locate(vertex: number): DomainPoint | undefined {
		return this.byVertex.get(vertex);
	}

	/**
	 * A domain coordinate mapped back onto the original surface.
	 *
	 * Evaluated on the coarse domain, then projected. See the note at the top
	 * of the file for what that costs relative to an exact inversion.
	 */
	toSurface(face: number, bary: readonly number[]): [number, number, number] {
		const coarse = this.domain.positionOf(face, bary);
		const hit = this.lookup.closest(coarse[0], coarse[1], coarse[2]);
		if (hit === null) return coarse;
		const cm = this.domain.hires;
		const out: [number, number, number] = [0, 0, 0];
		for (let k = 0; k < 3; k++) {
			const v = cm.fv(hit.face, k);
			out[0] += cm.vx(v) * hit.bary[k];
			out[1] += cm.vy(v) * hit.bary[k];
			out[2] += cm.vz(v) * hit.bary[k];
		}
		return out;
	}

	/**
	 * A uniform remesh: every domain face subdivided into `rate²` triangles,
	 * with each sample projected onto the surface.
	 *
	 * Because every domain face is subdivided the same way, and because the
	 * domain's own edges are shared, samples on a shared edge coincide exactly
	 * — so welding at the end joins the patches rather than papering over a
	 * gap. That regularity is the point of the whole construction: the output
	 * is near-uniform however uneven the input was.
	 */
	remesh(rate: number): CMeshO {
		if (rate < 2) throw new MLException(`the sampling rate must be at least 2, got ${rate}`);
		const out = new CMeshO();
		const cm = this.domain.base;

		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			// A triangular lattice of (rate+1)(rate+2)/2 barycentric samples.
			const index: number[][] = [];
			for (let i = 0; i <= rate; i++) {
				index.push([]);
				for (let j = 0; j <= rate - i; j++) {
					const bary = [(rate - i - j) / rate, j / rate, i / rate];
					const p = this.toSurface(f, bary);
					const v = Allocator.addVertices(out, 1);
					out.setVert(v, p[0], p[1], p[2]);
					index[i].push(v);
				}
			}
			for (let i = 0; i < rate; i++) {
				for (let j = 0; j < rate - i; j++) {
					Allocator.addFace(out, index[i][j], index[i][j + 1], index[i + 1][j]);
					if (j < rate - i - 1) {
						Allocator.addFace(out, index[i][j + 1], index[i + 1][j + 1], index[i + 1][j]);
					}
				}
			}
		}

		// The per-face lattices meet exactly on the domain's shared edges.
		Clean.removeDuplicateVertex(out);
		Allocator.compactEveryVector(out);
		return out;
	}

	/**
	 * A texture atlas laid out one domain face per slot.
	 *
	 * Each domain triangle becomes a right triangle in a grid of texture
	 * space, and each original vertex takes the barycentric position of its
	 * pin inside its face's slot. Because the domain is coarse and almost
	 * regular, this wastes far less texture space than a per-triangle atlas of
	 * the original — which is the practical reason the isoparametrisation is
	 * worth computing at all.
	 *
	 * An original face whose three vertices are pinned in *different* domain
	 * faces straddles a slot boundary. Upstream cuts such faces; here they are
	 * assigned to the domain face that holds most of them and the strays are
	 * clamped into the slot, which keeps the mesh whole at the cost of a
	 * little distortion right at the seams. `border` is the gap left between
	 * slots so filtered sampling does not bleed across them.
	 */
	atlasUV(border = 0.1): { cm: CMeshO; straddling: number } {
		const cm = copyForAtlas(this.domain.hires);
		const wt = cm.wedgeTexCoord as Float64Array;
		const domainFaces: number[] = [];
		for (let f = 0; f < this.domain.base.faceSize; f++) {
			if (!this.domain.base.isFaceD(f)) domainFaces.push(f);
		}
		const slotOf = new Map<number, number>();
		domainFaces.forEach((f, i) => {
			slotOf.set(f, i);
		});

		const side = Math.max(1, Math.ceil(Math.sqrt(domainFaces.length)));
		const step = 1 / side;
		const inset = Math.max(0, Math.min(0.49, border)) * step;

		let straddling = 0;
		for (let f = 0; f < cm.faceSize; f++) {
			if (cm.isFaceD(f)) continue;
			const pins = [0, 1, 2].map((k) => this.byVertex.get(cm.fv(f, k)));
			if (pins.some((p) => p === undefined)) continue;
			const homes = pins.map((p) => (p as DomainPoint).face);
			const home = majority(homes);
			if (homes.some((h) => h !== home)) straddling++;

			const slot = slotOf.get(home) ?? 0;
			const col = slot % side;
			const row = Math.floor(slot / side);
			for (let k = 0; k < 3; k++) {
				const pin = pins[k] as DomainPoint;
				// A vertex pinned elsewhere is re-expressed in the home face by
				// clamping, which is the seam distortion the doc comment warns
				// about; a vertex already at home keeps its exact coordinate.
				const bary = pin.face === home ? pin.bary : clampBary(pin.bary);
				const [u, v] = slotPoint(col, row, step, inset, bary);
				wt[6 * f + 2 * k] = u;
				wt[6 * f + 2 * k + 1] = v;
			}
		}
		return { cm, straddling };
	}

	/**
	 * Copies this parametrisation onto another, similar mesh.
	 *
	 * Every target vertex inherits the domain coordinate of the closest point
	 * of the source surface. It is only meaningful when the two meshes are
	 * genuinely aligned versions of the same object, which is exactly what the
	 * filter's description says.
	 */
	transferTo(target: CMeshO): IsoParametrization {
		const source = this.domain.hires;
		// The coarse domain is kept as it is; only the pin lists are rebuilt,
		// from the target's vertices instead of the source's.
		const pinned: Array<PinnedVertex[]> = Array.from(
			{ length: this.domain.base.faceSize },
			() => [],
		);
		let missed = 0;
		for (let v = 0; v < target.vertSize; v++) {
			if (target.isVertD(v)) continue;
			const hit = this.lookup.closest(target.vx(v), target.vy(v), target.vz(v));
			if (hit === null) {
				missed++;
				continue;
			}
			// The domain point of the closest source face, blended from its
			// three vertices' own pins. They may sit in different domain
			// faces; the nearest one wins, since a blend across a domain edge
			// is not a domain coordinate.
			const candidates = [0, 1, 2]
				.map((k) => ({ weight: hit.bary[k], point: this.byVertex.get(source.fv(hit.face, k)) }))
				.filter((c) => c.point !== undefined);
			if (candidates.length === 0) {
				missed++;
				continue;
			}
			const best = candidates.reduce((a, b) => (a.weight >= b.weight ? a : b));
			const point = best.point as DomainPoint;
			pinned[point.face].push({ vertex: v, bary: point.bary });
		}
		if (missed > 0) {
			throw new MLException(
				`${missed} target vertices found no point on the source surface; the two meshes are ` +
					"not close enough for a transfer",
			);
		}

		return new IsoParametrization(AbstractDomain.adopt(this.domain.base, target, pinned));
	}
}

function slotPoint(
	col: number,
	row: number,
	step: number,
	inset: number,
	bary: readonly number[],
): [number, number] {
	// The slot's right triangle, inset by the border.
	const x0 = col * step + inset;
	const y0 = row * step + inset;
	const size = step - 2 * inset;
	const corners: Array<[number, number]> = [
		[x0, y0],
		[x0 + size, y0],
		[x0, y0 + size],
	];
	let u = 0;
	let v = 0;
	for (let k = 0; k < 3; k++) {
		u += corners[k][0] * bary[k];
		v += corners[k][1] * bary[k];
	}
	return [u, v];
}

function majority(values: readonly number[]): number {
	const counts = new Map<number, number>();
	for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
	let best = values[0];
	let bestCount = 0;
	for (const [v, n] of counts) {
		if (n > bestCount) {
			bestCount = n;
			best = v;
		}
	}
	return best;
}

function clampBary(b: readonly number[]): [number, number, number] {
	const clamped = b.map((x) => Math.max(0, Math.min(1, x)));
	const total = clamped[0] + clamped[1] + clamped[2];
	if (total === 0) return [1 / 3, 1 / 3, 1 / 3];
	return [clamped[0] / total, clamped[1] / total, clamped[2] / total];
}

function edgesByLength(
	cm: CMeshO,
	skip: ReadonlySet<string>,
): Array<{ a: number; b: number; key: string }> {
	const out: Array<{ a: number; b: number; key: string; length: number }> = [];
	const seen = new Set<string>();
	for (let f = 0; f < cm.faceSize; f++) {
		if (cm.isFaceD(f)) continue;
		for (let e = 0; e < 3; e++) {
			const a = cm.fv(f, e);
			const b = cm.fv(f, (e + 1) % 3);
			const key = a < b ? `${a},${b}` : `${b},${a}`;
			if (seen.has(key) || skip.has(key)) continue;
			seen.add(key);
			out.push({
				a,
				b,
				key,
				length: Math.hypot(cm.vx(a) - cm.vx(b), cm.vy(a) - cm.vy(b), cm.vz(a) - cm.vz(b)),
			});
		}
	}
	out.sort((x, y) => x.length - y.length);
	return out;
}

/** A copy of the mesh with the wedge channel ready for the atlas. */
function copyForAtlas(src: CMeshO): CMeshO {
	const out = new CMeshO();
	const remap = new Int32Array(src.vertSize).fill(-1);
	let live = 0;
	for (let v = 0; v < src.vertSize; v++) if (!src.isVertD(v)) live++;
	if (live > 0) {
		const first = Allocator.addVertices(out, live);
		let at = first;
		for (let v = 0; v < src.vertSize; v++) {
			if (src.isVertD(v)) continue;
			remap[v] = at;
			out.setVert(at, src.vx(v), src.vy(v), src.vz(v));
			at++;
		}
	}
	for (let f = 0; f < src.faceSize; f++) {
		if (src.isFaceD(f)) continue;
		Allocator.addFace(out, remap[src.fv(f, 0)], remap[src.fv(f, 1)], remap[src.fv(f, 2)]);
	}
	enableChannels(out, MeshElement.MM_WEDGTEXCOORD);
	return out;
}
