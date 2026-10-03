# Conversion performance

Apple M1 Pro, darwin/arm64, Go1.27.1. Existing benchmark workloads, CPU1,
200ms/sample, six baseline/change pairs alternating AB/BA. The same compiled
benchmark harness was used on baseline commit
`1bb229fadd67c7de20898bba66b2c5d09693c430` and the implementation checkout.
Other agents paused CPU-heavy work. Functional and browser tests started after
the final benchmark window.

| Benchmark | Baseline | Changed | benchstat | Bytes change | Allocations |
| --- | ---: | ---: | --- | ---: | --- |
| C2D ZOH, 20 states | 20.96us | 21.12us | No significant difference | 0% | 28 unchanged |
| D2C ZOH, 2 states | 3.028us | 3.072us | +1.44%, p=.002 | -2.89% | 46 unchanged |
| D2C ZOH, 20 states | 58.81us | 59.00us | No significant difference | -4.15% | 50 unchanged |
| D2C ZOH, 50 states | 410.5us | 409.6us | No significant difference | -4.40% | 52 unchanged |
| D2C Tustin, 20 states | 13.76us | 13.74us | No significant difference | 0% | 32 unchanged |

A small two-state ZOH reverse latency regression accompanies stricter spectral
conditioning checks; these prevent lost Jordan terms and retain a fast common path. The
robust inverse-scaling/squaring fallback is required for defective/integrator
models. The changes reduce copied matrix memory. These results do not establish
a general speedup or zero regression.

Separately measured scoped matched conversion, two states: median7.596us,
45,156B/114alloc versus baseline13.415us,87,743B/158alloc (five200ms samples).
The corrected method also changes artificial-zero/gain semantics; interpret
this as comparative runtime of the two implementations, not identical outputs.

New least-squares fitting has no baseline API. Agent measurements: source2/fit1
2.761ms/.627MB/318alloc; source2/fit2 5.935ms/1.266MB/375alloc;
source2/fit4 12.949ms/2.059MB/388alloc; source8/fit8 19.118ms/2.984MB/~981alloc.
SVD-buffer reuse and scalar polynomial evaluation reduced a profiled refined
fit2 from2.226MB/3574alloc to1.266MB/375alloc. Fitting objective quality was
checked independently against analytic response and an offline NumPy oracle.
Exact MATLAB coefficients are unverified.

Exact decomposable ZOH delays retain shared rational states. Fractional FOH and
nondecomposable path delays may replicate states per input/output path; their
state-count and runtime growth are explicit limitations. No speedup is claimed
for these newly supported cases. Common no-delay paths remain separate from
those fallbacks.

Negative-pole ZOH compression, measured separately against the earlier full
state-doubling helper in an otherwise identical implementation: n2/negative1
111.09us to98.62us; n4/negative2 250.3us to225.7us, six interleaved samples,
p=.002. Returned orders fall from4 to3 and8 to6. Memory falls9.12–10.49%.
These rare-path measurements do not compare against the released baseline,
which rejects these cases. See controlsys `docs/benchmarks/conversion-negative-zoh`.

Tustin+Thiran external delay (`BenchmarkDiscretizeWithOpts_Thiran`, fixed to
Tustin because ZOH+Thiran is now rejected per MATLAB; the old benchmark timed
that error path): 2.52us to 4.09us, 3.46KiB to 8.19KiB, 74 to 160 allocs,
eight interleaved samples, p<.001. This accompanies the MATLAB maximum-order
split and LFT delay modeling; outputs differ from the baseline by design.
Generic `Series`/`Copy` composition dominated. No-delay
paths: Bilinear +0.99%, C2D ZOH +1.28% (p<.05), D2C ZOH memory -2.8 to -4.1%,
allocations unchanged. See `docs/benchmarks/conversion-thiran`.

Composing the Thiran bank directly into the discretized matrices instead of
through `BlkDiag`/`Series` (v1.11.0 baseline, twelve interleaved samples,
p<.001): `DiscretizeWithOpts_Thiran` 4.09us to 2.84us (-31%), 8.19KiB to
4.73KiB, 160 to 100 allocs; `IODelayThiran` 4.13us to 2.98us (-28%), 169 to 109
allocs. Responses, state counts, and delay metadata match generic `Series`
(`TestConversionSeriesMatchesGenericSeries`). No-delay, ZOH, D2C, and standalone
`ThiranDelay` benchmarks are unchanged (p>.05). Nondecomposable Thiran path
delays, previously rejected, cost 5.4-6.0us for n=4/2x2 and 14-15us for n=8/4x4
(`DiscretizeWithOpts_PathThiran`). See `docs/benchmarks/conversion-thiran-bank`.
