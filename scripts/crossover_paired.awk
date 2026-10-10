# Summarizes BenchmarkFrequencyDispatch rounds by pairing paths within each
# round, which cancels round-to-round runner speed drift (macOS arm64 runners
# drift 5-25% between rounds; paired ratios roughly halve that spread).
# Input: concatenated round outputs (raw.txt). For each n/m/w/a case prints the
# median over rounds of dense/hessenberg and of auto/min(dense, hessenberg),
# each with its half-range relative to the median.
# Usage: awk -f scripts/crossover_paired.awk raw.txt
function median(a, k,    i, j, t, b) {
	for (i = 1; i <= k; i++) b[i] = a[i]
	for (i = 2; i <= k; i++) {
		t = b[i]
		for (j = i - 1; j >= 1 && b[j] > t; j--) b[j + 1] = b[j]
		b[j + 1] = t
	}
	if (k % 2) return b[(k + 1) / 2]
	return (b[k / 2] + b[k / 2 + 1]) / 2
}
function halfrange(a, k, med,    i, lo, hi) {
	lo = hi = a[1]
	for (i = 2; i <= k; i++) {
		if (a[i] < lo) lo = a[i]
		if (a[i] > hi) hi = a[i]
	}
	return (hi - lo) / 2 / med
}
/^BenchmarkFrequencyDispatch\// {
	name = $1
	sub(/-[0-9]+$/, "", name)
	path = name
	sub(/.*\/path=/, "", path)
	c = name
	sub(/^BenchmarkFrequencyDispatch\//, "", c)
	sub(/\/path=.*/, "", c)
	r = ++seen[c, path]
	t[c, path, r] = $3
	if (!(c in known)) {
		known[c] = 1
		order[++ncase] = c
	}
}
END {
	printf "%-28s %8s %7s %8s %7s %6s\n", "case", "d/h", "±", "auto/min", "±", "rounds"
	for (i = 1; i <= ncase; i++) {
		c = order[i]
		k = seen[c, "dense"]
		if (seen[c, "hessenberg"] < k) k = seen[c, "hessenberg"]
		if (seen[c, "auto"] < k) k = seen[c, "auto"]
		if (k == 0) continue
		for (r = 1; r <= k; r++) {
			d = t[c, "dense", r]
			h = t[c, "hessenberg", r]
			ratio[r] = d / h
			loss[r] = t[c, "auto", r] / (d < h ? d : h)
		}
		mr = median(ratio, k)
		ml = median(loss, k)
		printf "%-28s %8.3f %6.1f%% %8.3f %6.1f%% %6d\n", c, mr, 100 * halfrange(ratio, k, mr), ml, 100 * halfrange(loss, k, ml), k
	}
}
