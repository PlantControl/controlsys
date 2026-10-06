# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `HEAD` | `.` |
| Commit | `b4357dac6be0d5fd106877f394f7b63841dc1636` | `b4357dac6be0d5fd106877f394f7b63841dc1636` |
| Uncommitted changes | false | true (diff sha256 `5defd9e001bdf69849832f95de32f3b40a5d0f152c79c7fcad1c875879449428`) |
| Backend | goos=darwin goarch=arm64 backend=gonum.Implementation/pure-go | goos=darwin goarch=arm64 backend=gonum.Implementation/pure-go |

- Go: go version go1.27.1 darwin/arm64; GOAMD64= GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: darwin/arm64, Apple M1 Pro, 8 CPUs, GOMAXPROCS env ""
- Samples: 10 interleaved rounds per revision; flags `-test.run=^$ -test.bench=^Benchmark(FrequencySweepKernels|Bode|D2C_ZOH_N20|DiscretizeZOH)$ -test.benchtime=100ms -test.benchmem -test.count=1`

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencySweepKernels/N4_M2_W8/Hessenberg-8 | sec/op | 3.949e-06 | 4.646e-06 | +17.66% | p=0.000 n=10 |
| FrequencySweepKernels/N4_M2_W8/Hessenberg-8 | B/op | 2532 | 3556 | +40.44% | p=0.000 n=10 |
| FrequencySweepKernels/N4_M2_W100/Hessenberg-8 | B/op | 8548 | 9572 | +11.98% | p=0.000 n=10 |
| FrequencySweepKernels/N8_M1_W100/Hessenberg-8 | B/op | 5960 | 8520 | +42.95% | p=0.000 n=10 |
| FrequencySweepKernels/N8_M2_W100/Hessenberg-8 | B/op | 1.146e+04 | 1.454e+04 | +26.80% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M1_W100/Hessenberg-8 | B/op | 1e+04 | 1.563e+04 | +56.32% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M2_W100/Hessenberg-8 | B/op | 1.934e+04 | 2.651e+04 | +37.06% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M4_W100/Hessenberg-8 | B/op | 3.842e+04 | 4.661e+04 | +21.32% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M1_W100/Hessenberg-8 | B/op | 1.868e+04 | 3.097e+04 | +65.78% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M2_W100/Hessenberg-8 | B/op | 2.482e+04 | 3.762e+04 | +51.56% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M4_W100/Hessenberg-8 | B/op | 4.838e+04 | 6.476e+04 | +33.87% | p=0.000 n=10 |
| FrequencySweepKernels/N30_M1_W100/Hessenberg-8 | B/op | 4.576e+04 | 7.853e+04 | +71.61% | p=0.000 n=10 |
| FrequencySweepKernels/N30_M5_W100/Hessenberg-8 | B/op | 9.466e+04 | 1.336e+05 | +41.11% | p=0.000 n=10 |
| FrequencySweepKernels/N48_M1_W100/Hessenberg-8 | B/op | 1.136e+05 | 1.956e+05 | +72.08% | p=0.000 n=10 |
| FrequencySweepKernels/N48_M2_W100/Hessenberg-8 | B/op | 1.184e+05 | 2.003e+05 | +69.20% | p=0.000 n=10 |
| FrequencySweepKernels/N64_M1_W100/Hessenberg-8 | B/op | 1.917e+05 | 3.556e+05 | +85.45% | p=0.000 n=10 |
| FrequencySweepKernels/N64_M4_W100/Hessenberg-8 | B/op | 2.254e+05 | 3.893e+05 | +72.69% | p=0.000 n=10 |
| FrequencySweepKernels/N100_M2_W100/Hessenberg-8 | B/op | 4.431e+05 | 8.035e+05 | +81.35% | p=0.000 n=10 |
| FrequencySweepKernels/N4_M2_W8/Hessenberg-8 | allocs/op | 7 | 11 | +57.14% | p=0.000 n=10 |
| FrequencySweepKernels/N4_M2_W100/Hessenberg-8 | allocs/op | 7 | 11 | +57.14% | p=0.000 n=10 |
| FrequencySweepKernels/N8_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N8_M2_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M2_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M4_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M2_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M4_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N30_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N30_M5_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N48_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N48_M2_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N64_M1_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N64_M4_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |
| FrequencySweepKernels/N100_M2_W100/Hessenberg-8 | allocs/op | 8 | 12 | +50.00% | p=0.000 n=10 |

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencySweepKernels/N4_M2_W100/Hessenberg-8 | sec/op | 3.544e-05 | 3.675e-05 | +3.67% | p=0.000 n=10 |
| FrequencySweepKernels/N8_M1_W100/Hessenberg-8 | sec/op | 6.948e-05 | 7.102e-05 | +2.22% | p=0.029 n=10 |
| FrequencySweepKernels/N8_M2_W100/Hessenberg-8 | sec/op | 0.0001135 | 0.0001155 | +1.74% | p=0.005 n=10 |
| FrequencySweepKernels/N12_M1_W100/Hessenberg-8 | sec/op | 0.0001263 | 0.0001306 | +3.45% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M2_W100/Hessenberg-8 | sec/op | 0.0002123 | 0.0002197 | +3.48% | p=0.000 n=10 |
| FrequencySweepKernels/N12_M4_W100/Hessenberg-8 | sec/op | 0.000373 | 0.0003872 | +3.80% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M1_W100/Hessenberg-8 | sec/op | 0.0002226 | 0.0002315 | +4.02% | p=0.000 n=10 |
| FrequencySweepKernels/N18_M2_W100/Hessenberg-8 | sec/op | 0.0003805 | 0.0003953 | +3.88% | p=0.001 n=10 |
| FrequencySweepKernels/N18_M4_W100/Hessenberg-8 | sec/op | 0.0006717 | 0.0006936 | +3.26% | p=0.000 n=10 |
| FrequencySweepKernels/N30_M1_W100/Hessenberg-8 | sec/op | 0.0004551 | 0.0004677 | +2.76% | p=0.001 n=10 |
| FrequencySweepKernels/N30_M5_W100/Hessenberg-8 | sec/op | 0.001738 | 0.001779 | +2.40% | p=0.023 n=10 |
| FrequencySweepKernels/N48_M1_W100/Hessenberg-8 | sec/op | 0.001079 | 0.001103 | +2.21% | p=0.005 n=10 |
| FrequencySweepKernels/N48_M2_W100/Hessenberg-8 | sec/op | 0.001837 | 0.001875 | +2.09% | p=0.007 n=10 |
| FrequencySweepKernels/N64_M1_W100/Hessenberg-8 | sec/op | 0.001968 | 0.002005 | +1.85% | p=0.015 n=10 |

## Inconclusive

62 case metrics show no significant difference. This is not evidence of equivalence.

## Missing or incompatible cases

Compared on both revisions: 37 cases.


Full table: benchstat.txt. Raw samples: baseline.txt, candidate.txt, raw/. Metadata: metadata.json.
