# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `0fb4bf93b2e67be7856b4906f5cd206dd52f478d` | `ef0c522ce10f8ae04f0a22e62bf2fe28dbeffb17` |
| Commit | `0fb4bf93b2e67be7856b4906f5cd206dd52f478d` | `ef0c522ce10f8ae04f0a22e62bf2fe28dbeffb17` |
| Backend | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm |

- Go: go version go1.27.1 linux/amd64; GOAMD64=v1 GOARM64= CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: linux/amd64, AMD EPYC 9V74 80-Core Processor, 4 CPUs, GOMAXPROCS env ""
- Runner: GitHub Actions 1000002721 ubuntu24 20260927.320.1
- Samples: 6 interleaved rounds per revision; flags `-test.run=^$ -test.bench=. -test.benchtime=100ms -test.benchmem -test.count=1`

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| LFTExtract/n=50-4 | sec/op | 3.719e-06 | 3.944e-06 | +6.05% | p=0.026 n=6 |
| FRDSeries_SISO_200-4 | sec/op | 6.585e-06 | 7.192e-06 | +9.23% | p=0.004 n=6 |
| FRDSeries_SISO_2000-4 | sec/op | 6.936e-05 | 7.528e-05 | +8.53% | p=0.002 n=6 |
| FRDSeries_MIMO_200-4 | sec/op | 3.034e-05 | 3.556e-05 | +17.20% | p=0.015 n=6 |
| FRDParallel_MIMO_2000-4 | sec/op | 0.0002789 | 0.0002981 | +6.88% | p=0.004 n=6 |
| FRDFeedback_MIMO_10000-4 | sec/op | 0.003804 | 0.004025 | +5.81% | p=0.004 n=6 |
| Sminreal_N50-4 | sec/op | 5.161e-06 | 5.428e-06 | +5.18% | p=0.026 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | B/op | 8416 | 1.464e+04 | +73.95% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | B/op | 1.293e+04 | 1.915e+04 | +48.14% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | B/op | 1.859e+04 | 3.161e+04 | +70.01% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | B/op | 3.251e+04 | 4.553e+04 | +40.03% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | B/op | 5.584e+04 | 9.282e+04 | +66.22% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | B/op | 1.094e+05 | 1.464e+05 | +33.79% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| DenseNorm-4 | sec/op | 7.742e-07 | 7.972e-07 | +2.96% | p=0.002 n=6 |
| FrequencySweepKernels/N4_M2_W100/Dense-4 | sec/op | 2.029e-05 | 2.044e-05 | +0.74% | p=0.026 n=6 |
| FrequencySweepKernels/N8_M1_W100/Hessenberg-4 | sec/op | 6.299e-05 | 6.329e-05 | +0.48% | p=0.026 n=6 |
| FrequencySweepKernels/N12_M2_W100/Hessenberg-4 | sec/op | 0.000194 | 0.0001948 | +0.40% | p=0.041 n=6 |
| FrequencySweepKernels/N12_M4_W100/Hessenberg-4 | sec/op | 0.0003647 | 0.0003675 | +0.77% | p=0.002 n=6 |
| FrequencyDispatch/n=8/m=1/w=20/a=full/path=hessenberg-4 | sec/op | 1.528e-05 | 1.543e-05 | +0.96% | p=0.041 n=6 |
| FrequencyDispatch/n=8/m=1/w=200/a=full/path=dense-4 | sec/op | 0.0001212 | 0.0001216 | +0.34% | p=0.026 n=6 |
| FrequencyDispatch/n=12/m=1/w=3/a=full/path=dense-4 | sec/op | 5.526e-06 | 5.665e-06 | +2.51% | p=0.015 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=dense-4 | sec/op | 0.0002972 | 0.000296 | -0.40% | p=0.026 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | sec/op | 6.115e-05 | 4.157e-05 | -32.01% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | sec/op | 0.0005853 | 0.0003258 | -44.34% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | sec/op | 2.068e-05 | 1.846e-05 | -10.72% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=200/a=full/path=dense-4 | sec/op | 0.001022 | 0.001015 | -0.64% | p=0.009 n=6 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=dense-4 | sec/op | 5.66e-05 | 5.718e-05 | +1.03% | p=0.004 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=hessenberg/path=dense-4 | sec/op | 0.0002983 | 0.0002994 | +0.37% | p=0.004 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | sec/op | 0.0001822 | 0.0001313 | -27.94% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | sec/op | 0.001777 | 0.001139 | -35.89% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | sec/op | 4.792e-05 | 4.476e-05 | -6.59% | p=0.009 n=6 |
| FrequencyDispatch/n=40/m=4/w=3/a=full/path=auto-4 | sec/op | 0.0001218 | 0.0001229 | +0.93% | p=0.041 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | sec/op | 0.0007687 | 0.0005853 | -23.87% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | sec/op | 0.007474 | 0.005233 | -29.99% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | sec/op | 0.0001617 | 0.0001557 | -3.71% | p=0.002 n=6 |
| FrequencyDispatch/n=80/m=4/w=200/a=hessenberg/path=dense-4 | sec/op | 0.009205 | 0.00923 | +0.28% | p=0.041 n=6 |
| SimulateNoDelay-4 | sec/op | 8.928e-05 | 8.645e-05 | -3.16% | p=0.009 n=6 |
| ThiranDelay-4 | sec/op | 3.856e-07 | 4.002e-07 | +3.79% | p=0.041 n=6 |
| DecomposeIODelay-4 | sec/op | 2.528e-07 | 2.464e-07 | -2.53% | p=0.002 n=6 |
| SetDelayModel-4 | sec/op | 6.486e-07 | 6.616e-07 | +2.00% | p=0.026 n=6 |
| ZeroDelayApprox-4 | sec/op | 2.289e-06 | 2.337e-06 | +2.10% | p=0.045 n=6 |
| Simulate_DCMotor-4 | sec/op | 8.249e-05 | 7.147e-05 | -13.36% | p=0.002 n=6 |
| Simulate_MassSpringDamper-4 | sec/op | 4.493e-05 | 4.368e-05 | -2.77% | p=0.002 n=6 |
| DiscretizeAndSimulate_B747Longitudinal-4 | sec/op | 2.047e-05 | 1.831e-05 | -10.57% | p=0.002 n=6 |
| Simulate_Large-4 | sec/op | 0.0004919 | 0.0004882 | -0.76% | p=0.028 n=6 |
| FeedbackAndSimulate-4 | sec/op | 1.945e-05 | 1.758e-05 | -9.60% | p=0.002 n=6 |
| Bode_LargeMIMO-4 | sec/op | 0.001102 | 0.001106 | +0.44% | p=0.026 n=6 |
| Dare_N100_M5-4 | sec/op | 0.02868 | 0.02889 | +0.73% | p=0.015 n=6 |
| Reg_N50_M5_P5-4 | sec/op | 2.382e-05 | 2.423e-05 | +1.70% | p=0.041 n=6 |
| ZPKEval_SISO-4 | sec/op | 5.848e-08 | 5.945e-08 | +1.66% | p=0.013 n=6 |
| SSToZPK_SISO-4 | sec/op | 2.288e-05 | 2.342e-05 | +2.35% | p=0.041 n=6 |
| BlkDiag-4 | sec/op | 2.378e-06 | 2.481e-06 | +4.35% | p=0.041 n=6 |
| LFT-4 | sec/op | 6.392e-06 | 6.616e-06 | +3.51% | p=0.015 n=6 |
| LFT_Large-4 | sec/op | 1.307e-05 | 1.343e-05 | +2.82% | p=0.015 n=6 |
| EvalFr_N4-4 | sec/op | 8.241e-07 | 7.954e-07 | -3.49% | p=0.026 n=6 |
| SystemFRD_Large_2000-4 | sec/op | 0.02519 | 0.02538 | +0.77% | p=0.026 n=6 |
| FRDSigma_MIMO_200-4 | sec/op | 0.0002893 | 0.0002881 | -0.43% | p=0.009 n=6 |
| FRDSigma_MIMO_10000-4 | sec/op | 0.01451 | 0.01444 | -0.47% | p=0.041 n=6 |
| FRDMargin_200-4 | sec/op | 1.071e-05 | 1.107e-05 | +3.38% | p=0.026 n=6 |
| FRDMargin_10000-4 | sec/op | 0.0005232 | 0.0005474 | +4.63% | p=0.015 n=6 |
| FRDSeries_MIMO_10000-4 | sec/op | 0.001693 | 0.001759 | +3.87% | p=0.041 n=6 |
| FRDFeedback_SISO_2000-4 | sec/op | 0.0001122 | 0.0001139 | +1.58% | p=0.015 n=6 |
| FRDSelectFrequencies_MIMO_2000-4 | sec/op | 0.0001656 | 0.0001717 | +3.67% | p=0.041 n=6 |
| Step_SISO_N10-4 | sec/op | 0.0001264 | 0.0001211 | -4.18% | p=0.002 n=6 |
| Step_SISO_N50-4 | sec/op | 0.002132 | 0.002067 | -3.04% | p=0.002 n=6 |
| Impulse_SISO_N10-4 | sec/op | 0.0001253 | 0.000115 | -8.23% | p=0.002 n=6 |
| Lsim_SISO_N2-4 | sec/op | 5.001e-05 | 4.436e-05 | -11.30% | p=0.002 n=6 |
| Lsim_SISO_N10-4 | sec/op | 6.139e-05 | 5.704e-05 | -7.09% | p=0.002 n=6 |
| Lsim_SISO_N50-4 | sec/op | 0.0003098 | 0.0003011 | -2.79% | p=0.009 n=6 |
| Lsim_MIMO_N10-4 | sec/op | 8.225e-05 | 7.785e-05 | -5.34% | p=0.002 n=6 |
| Lsim_SISO_1e4-4 | sec/op | 0.0009192 | 0.0008371 | -8.94% | p=0.002 n=6 |
| Lsim_MIMO_1e4-4 | sec/op | 0.001446 | 0.001288 | -10.93% | p=0.002 n=6 |
| Lsim_Discrete_SISO-4 | sec/op | 8.818e-05 | 8.233e-05 | -6.64% | p=0.002 n=6 |
| Canon_Modal_N100-4 | sec/op | 0.008213 | 0.008311 | +1.18% | p=0.041 n=6 |
| Ssbal_N50-4 | sec/op | 1.877e-05 | 1.917e-05 | +2.12% | p=0.041 n=6 |
| Prescale_N50-4 | sec/op | 5.642e-05 | 5.572e-05 | -1.24% | p=0.041 n=6 |
| Covar_SISO_N2-4 | sec/op | 2.896e-06 | 2.805e-06 | -3.14% | p=0.002 n=6 |
| ConversionExternalDelay/residual-path-4 | sec/op | 1.033e-05 | 1.05e-05 | +1.72% | p=0.026 n=6 |
| HinfNorm_InternalDelay/mimo-4 | sec/op | 0.002328 | 0.002357 | +1.22% | p=0.009 n=6 |
| DiskMargin_SISO-4 | sec/op | 2.299e-05 | 2.342e-05 | +1.85% | p=0.041 n=6 |
| DiskMargin_PIDExactDelay/internal/DiskMarginSkew-4 | sec/op | 0.0007753 | 0.0007824 | +0.91% | p=0.015 n=6 |
| DiscretizeZOH-4 | B/op | 2.216e+04 | 2.217e+04 | +0.04% | p=0.026 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | B/op | 2.134e+04 | 1.319e+04 | -38.19% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | B/op | 4.146e+04 | 2.37e+04 | -42.83% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | B/op | 1.107e+05 | 6.14e+04 | -44.52% | p=0.002 n=6 |
| HSV-4 | B/op | 2.933e+04 | 2.934e+04 | +0.03% | p=0.050 n=6 |
| HinfNorm_MixedSensitivityLoop/n=64-4 | B/op | 9.737e+05 | 9.779e+05 | +0.43% | p=0.050 n=6 |
| SSToZPK_SISO-4 | B/op | 4.862e+04 | 4.863e+04 | +0.03% | p=0.026 n=6 |
| Nyquist-4 | B/op | 4.059e+04 | 4.06e+04 | +0.02% | p=0.037 n=6 |
| ModelArrayStep_MIMO_16-4 | B/op | 1.759e+06 | 1.758e+06 | -0.05% | p=0.041 n=6 |
| Impulse_SISO_N10-4 | B/op | 5.823e+04 | 5.827e+04 | +0.07% | p=0.026 n=6 |
| Lsim_SISO_N10-4 | B/op | 1.804e+04 | 1.805e+04 | +0.02% | p=0.011 n=6 |
| Loopsens_MIMO_N10-4 | B/op | 4.282e+04 | 4.282e+04 | +0.01% | p=0.002 n=6 |
| Canon_Modal_N100-4 | B/op | 1.312e+06 | 1.331e+06 | +1.43% | p=0.026 n=6 |
| Passivity_SISO-4 | B/op | 2.433e+04 | 2.433e+04 | +0.01% | p=0.006 n=6 |
| TransferFunction/pade18-4 | B/op | 7.768e+04 | 7.766e+04 | -0.03% | p=0.037 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| ModelArrayStep_MIMO_16-4 | allocs/op | 857 | 854 | -0.35% | p=0.024 n=6 |

## Inconclusive

1286 case metrics show no significant difference. This is not evidence of equivalence.

## Missing or incompatible cases

Compared on both revisions: 464 cases.

Baseline only (removed or renamed) (18):

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

Candidate only (new or renamed, no baseline) (351):

- BenchmarkFrequencyDispatch/n=12/m=1/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=12/m=1/w=50/a=full/path=hessenberg-4
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
- BenchmarkFrequencyDispatch/n=18/m=2/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=18/m=2/w=50/a=full/path=hessenberg-4
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
- BenchmarkFrequencyDispatch/n=30/m=4/w=1/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=1/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=1/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=10/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=10/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=10/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=2/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=2/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=2/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=20/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=20/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=20/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=200/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=200/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=200/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=3/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=3/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=3/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=5/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=5/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=5/a=full/path=hessenberg-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=50/a=full/path=auto-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=50/a=full/path=dense-4
- BenchmarkFrequencyDispatch/n=30/m=4/w=50/a=full/path=hessenberg-4
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
