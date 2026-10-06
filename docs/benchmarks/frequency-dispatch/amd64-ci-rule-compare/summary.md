# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `6e61daf2fbe2fc130b0c05f4c2b8a8f639d8e965` | `523aa232b7462ccb17df83b5442f03427327fc80` |
| Commit | `6e61daf2fbe2fc130b0c05f4c2b8a8f639d8e965` | `523aa232b7462ccb17df83b5442f03427327fc80` |
| Backend | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm |

- Go: go version go1.27.1 linux/amd64; GOAMD64=v1 GOARM64= CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: linux/amd64, AMD EPYC 9V45 96-Core Processor, 4 CPUs, GOMAXPROCS env ""
- Runner: GitHub Actions 1000002649 ubuntu24 20260927.320.1
- Samples: 6 interleaved rounds per revision; flags `-test.run=^$ -test.bench=. -test.benchtime=100ms -test.benchmem -test.count=1`

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-4 | sec/op | 4.245e-05 | 7.926e-05 | +86.71% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-4 | sec/op | 3.6e-05 | 4.047e-05 | +12.41% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=200/a=hessenberg/path=hessenberg-4 | sec/op | 0.0006917 | 0.0007349 | +6.25% | p=0.015 n=6 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-4 | sec/op | 0.0001195 | 0.0001916 | +60.33% | p=0.002 n=6 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-4 | sec/op | 0.0001002 | 0.0001193 | +19.04% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-4 | sec/op | 0.0004863 | 0.0006814 | +40.11% | p=0.002 n=6 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-4 | sec/op | 0.0004214 | 0.0005309 | +25.99% | p=0.002 n=6 |
| Dare_N100_M5-4 | sec/op | 0.02082 | 0.02207 | +6.00% | p=0.004 n=6 |
| Estim_N100_M5_P5-4 | sec/op | 4.84e-05 | 5.278e-05 | +9.04% | p=0.015 n=6 |
| Reg_N10_M2_P3-4 | sec/op | 1.612e-06 | 1.734e-06 | +7.60% | p=0.002 n=6 |
| ZPKFreqResponse_SISO_100-4 | sec/op | 1.722e-06 | 1.811e-06 | +5.14% | p=0.041 n=6 |
| MatLog_N20-4 | sec/op | 3.5e-05 | 3.675e-05 | +5.00% | p=0.041 n=6 |
| Ssbal_N50-4 | sec/op | 1.174e-05 | 1.266e-05 | +7.85% | p=0.026 n=6 |
| Sminreal_N10-4 | sec/op | 4.665e-07 | 4.973e-07 | +6.61% | p=0.015 n=6 |

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| DiscretizeZOH-4 | sec/op | 1.485e-05 | 1.545e-05 | +4.04% | p=0.009 n=6 |
| Feedback-4 | sec/op | 4.328e-06 | 4.438e-06 | +2.55% | p=0.026 n=6 |
| FeedbackLFT-4 | sec/op | 1.116e-05 | 1.152e-05 | +3.25% | p=0.026 n=6 |
| FrequencySweepKernels/N12_M1_W100/Dense-4 | sec/op | 0.0001132 | 0.0001165 | +2.96% | p=0.041 n=6 |
| FrequencySweepKernels/N100_M2_W100/Hessenberg-4 | sec/op | 0.005756 | 0.005985 | +3.99% | p=0.009 n=6 |
| FrequencyDispatch/n=8/m=1/w=20/a=full/path=dense-4 | sec/op | 9.876e-06 | 1.001e-05 | +1.37% | p=0.041 n=6 |
| FrequencyDispatch/n=16/m=1/w=3/a=full/path=auto-4 | sec/op | 7.782e-06 | 8.029e-06 | +3.18% | p=0.013 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | sec/op | 1.54e-05 | 1.364e-05 | -11.37% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=200/a=full/path=hessenberg-4 | sec/op | 0.0007053 | 0.0007223 | +2.41% | p=0.041 n=6 |
| FrequencyDispatch/n=32/m=1/w=200/a=hessenberg/path=dense-4 | sec/op | 0.000924 | 0.0009675 | +4.71% | p=0.015 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=dense-4 | sec/op | 3.101e-05 | 3.214e-05 | +3.66% | p=0.026 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | sec/op | 3.464e-05 | 3.17e-05 | -8.47% | p=0.002 n=6 |
| FrequencyDispatch/n=48/m=2/w=200/a=full/path=auto-4 | sec/op | 0.002665 | 0.002768 | +3.89% | p=0.026 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | sec/op | 0.0001163 | 0.0001074 | -7.60% | p=0.009 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=hessenberg/path=hessenberg-4 | sec/op | 0.003571 | 0.003692 | +3.38% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=hessenberg/path=auto-4 | sec/op | 0.001862 | 0.001936 | +4.00% | p=0.009 n=6 |
| BlkDiag-4 | sec/op | 1.576e-06 | 1.619e-06 | +2.70% | p=0.039 n=6 |
| DescriptorToExplicit_N10-4 | sec/op | 2.786e-06 | 2.9e-06 | +4.09% | p=0.041 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | B/op | 2.134e+04 | 1.319e+04 | -38.19% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-4 | B/op | 2.175e+04 | 1.36e+04 | -37.48% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-4 | B/op | 4.958e+04 | 2.9e+04 | -41.50% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | B/op | 4.146e+04 | 2.37e+04 | -42.83% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-4 | B/op | 4.269e+04 | 2.493e+04 | -41.60% | p=0.002 n=6 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-4 | B/op | 1.122e+05 | 6.294e+04 | -43.91% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | B/op | 1.107e+05 | 6.14e+04 | -44.52% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-4 | B/op | 1.154e+05 | 6.614e+04 | -42.69% | p=0.002 n=6 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-4 | B/op | 2.976e+05 | 1.746e+05 | -41.33% | p=0.002 n=6 |
| DiscretizeWithOpts_PathThiran/n=8/io=4/state-4 | B/op | 4.068e+04 | 4.068e+04 | -0.01% | p=0.022 n=6 |
| Canon_Modal_N50-4 | B/op | 2.568e+05 | 2.563e+05 | -0.21% | p=0.041 n=6 |
| PhysicalAssembly_8Components-4 | B/op | 5.858e+04 | 5.859e+04 | +0.01% | p=0.048 n=6 |
| ModifiedFOHForwardN20-4 | B/op | 3.759e+04 | 3.761e+04 | +0.03% | p=0.011 n=6 |
| DiskMargin_PIDExactDelay/io/DiskMarginSkew-4 | B/op | 7.01e+04 | 7.011e+04 | +0.02% | p=0.015 n=6 |
| DiskMargin_PIDExactDelay/internal/DiskMarginSkew-4 | B/op | 7.192e+04 | 7.194e+04 | +0.03% | p=0.017 n=6 |
| TransferFunction/pade18-4 | B/op | 7.765e+04 | 7.771e+04 | +0.07% | p=0.041 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |

## Inconclusive

1308 case metrics show no significant difference. This is not evidence of equivalence.

## Missing or incompatible cases

Compared on both revisions: 455 cases.

Baseline only (removed or renamed) (27):

- BenchmarkFrequencyDispatch/n=12/m=1/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=2/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=4/w=3/a=full/path=hessenberg-4

Candidate only (new or renamed, no baseline) (288):

- BenchmarkFrequencyDispatch/n=12/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=2/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=16/m=1/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=1/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=20/m=4/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=1/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=24/m=2/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=28/m=2/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=32/m=1/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=36/m=2/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=40/m=4/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=44/m=4/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=48/m=2/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=60/m=4/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=8/m=1/w=50/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=20/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=20/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=20/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=3/a=hessenberg/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=3/a=hessenberg/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=3/a=hessenberg/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=80/m=4/w=50/a=full/path=hessenberg-4


Full table: benchstat.txt. Raw samples: baseline.txt, candidate.txt, raw/. Metadata: metadata.json.
