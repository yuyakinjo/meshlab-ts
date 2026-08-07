/**
 * `vcg::Distribution` and `vcg::Histogram`.
 *
 * A distribution accumulates values and answers questions about them; a
 * histogram bins them into a fixed range. VCGLib's `Distribution` answers
 * percentile queries out of an internal 10000-bin histogram, which is
 * approximate. This one keeps the values and sorts on demand, so a percentile
 * is exact. That is a deliberate divergence: the approximation exists in C++
 * to bound memory over meshes that stream past, and nothing here streams.
 */

export class Distribution {
	private values: number[] = [];
	private sorted: number[] | null = null;
	private _sum = 0;
	private _sumSquared = 0;
	private _min = Number.POSITIVE_INFINITY;
	private _max = Number.NEGATIVE_INFINITY;

	Add(value: number, weight = 1): void {
		// A weight is a multiplicity, which is how VCG uses it for area
		// weighting; fractional weights work for the moments but a percentile
		// of a fractionally weighted sample is not well defined, so the
		// percentile path below uses the values alone.
		this.values.push(value);
		this.sorted = null;
		this._sum += value * weight;
		this._sumSquared += value * value * weight;
		if (value < this._min) this._min = value;
		if (value > this._max) this._max = value;
		this._weight += weight;
	}

	private _weight = 0;

	Cnt(): number {
		return this.values.length;
	}

	Sum(): number {
		return this._sum;
	}

	Min(): number {
		return this.values.length === 0 ? 0 : this._min;
	}

	Max(): number {
		return this.values.length === 0 ? 0 : this._max;
	}

	Avg(): number {
		return this._weight === 0 ? 0 : this._sum / this._weight;
	}

	Variance(): number {
		if (this._weight === 0) return 0;
		const avg = this.Avg();
		return Math.max(0, this._sumSquared / this._weight - avg * avg);
	}

	StandardDeviation(): number {
		return Math.sqrt(this.Variance());
	}

	/** The value below which the given fraction of the sample falls. */
	Percentile(fraction: number): number {
		if (this.values.length === 0) return 0;
		if (this.sorted === null) this.sorted = [...this.values].sort((a, b) => a - b);
		const at = Math.min(
			this.sorted.length - 1,
			Math.max(0, Math.floor(fraction * this.sorted.length)),
		);
		return this.sorted[at];
	}
}

/**
 * A fixed-range histogram with two overflow bins.
 *
 * Bin 0 collects everything below the range and bin `binNum + 1` everything
 * above it, which is why the reported bins are `binNum + 2` and why the first
 * and last carry infinite bounds. Dropping out-of-range values instead would
 * make a histogram whose counts silently fail to add up to the sample size.
 */
export class Histogram {
	private minRange = 0;
	private maxRange = 1;
	private binNum = 0;
	private counts: Float64Array = new Float64Array(0);

	SetRange(minRange: number, maxRange: number, binNum: number): void {
		if (binNum < 1) throw new Error(`a histogram needs at least one bin, got ${binNum}`);
		this.minRange = minRange;
		this.maxRange = maxRange;
		this.binNum = binNum;
		this.counts = new Float64Array(binNum + 2);
	}

	Add(value: number, weight = 1): void {
		this.counts[this.binIndex(value)] += weight;
	}

	/** 0 for the underflow bin, `binNum + 1` for the overflow one. */
	binIndex(value: number): number {
		if (value < this.minRange) return 0;
		if (value >= this.maxRange) return this.binNum + 1;
		const span = this.maxRange - this.minRange;
		// A zero-width range would put every finite value in the overflow bin
		// above, so span is positive by the time we get here.
		const at = Math.floor(((value - this.minRange) / span) * this.binNum);
		return Math.min(this.binNum, at) + 1;
	}

	BinCountInd(i: number): number {
		return this.counts[i];
	}

	BinLowerBound(i: number): number {
		if (i === 0) return Number.NEGATIVE_INFINITY;
		const span = this.maxRange - this.minRange;
		return this.minRange + (span * (i - 1)) / this.binNum;
	}

	BinUpperBound(i: number): number {
		if (i === this.binNum + 1) return Number.POSITIVE_INFINITY;
		const span = this.maxRange - this.minRange;
		return this.minRange + (span * i) / this.binNum;
	}

	bins(): number {
		return this.binNum;
	}
}
