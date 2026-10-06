# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `0fb4bf93b2e67be7856b4906f5cd206dd52f478d` | `8e56d2c7d09c6460f7b8469624aea4908644a5fa` |
| Commit | `0fb4bf93b2e67be7856b4906f5cd206dd52f478d` | `8e56d2c7d09c6460f7b8469624aea4908644a5fa` |
| Backend | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm | goos=linux goarch=amd64 backend=gonum.Implementation/amd64-asm |

- Go: go version go1.27.1 linux/amd64; GOAMD64=v1 GOARM64= CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: linux/amd64, AMD EPYC 7763 64-Core Processor, 4 CPUs, GOMAXPROCS env ""
- Runner: GitHub Actions 1000002750 ubuntu24 20260927.320.1
- Samples: 6 interleaved rounds per revision; flags `-test.run=^$ -test.bench=. -test.benchtime=100ms -test.benchmem -test.count=1`

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencySweepKernels/N4_M2_W100/Dense-4 | sec/op | 2.562e-05 | 2.815e-05 | +9.87% | p=0.002 n=6 |
| FreqResponse_B747Lateral-4 | sec/op | 5.285e-05 | 5.791e-05 | +9.58% | p=0.004 n=6 |
| Estim_N10_M2_P3-4 | sec/op | 3.465e-06 | 3.707e-06 | +6.98% | p=0.002 n=6 |
| BlkDiag-4 | sec/op | 2.789e-06 | 2.945e-06 | +5.61% | p=0.002 n=6 |
| LFTExtract/n=50-4 | sec/op | 4.747e-06 | 5.107e-06 | +7.60% | p=0.015 n=6 |
| SystemFRD_SISO_2000-4 | sec/op | 0.0001861 | 0.0001998 | +7.33% | p=0.002 n=6 |
| FRDSigma_SISO_200-4 | sec/op | 3.41e-06 | 3.937e-06 | +15.46% | p=0.002 n=6 |
| FRDSigma_SISO_2000-4 | sec/op | 3.318e-05 | 4.536e-05 | +36.70% | p=0.002 n=6 |
| FRDMargin_200-4 | sec/op | 1.601e-05 | 1.721e-05 | +7.50% | p=0.002 n=6 |
| FRDMargin_10000-4 | sec/op | 0.0007789 | 0.0008237 | +5.76% | p=0.041 n=6 |
| FRDSeries_SISO_200-4 | sec/op | 8.393e-06 | 9.558e-06 | +13.88% | p=0.002 n=6 |
| FRDSeries_SISO_2000-4 | sec/op | 8.892e-05 | 0.0001028 | +15.58% | p=0.002 n=6 |
| FRDSeries_MIMO_200-4 | sec/op | 3.898e-05 | 4.095e-05 | +5.05% | p=0.015 n=6 |
| FRDParallel_SISO_2000-4 | sec/op | 8.577e-05 | 9.575e-05 | +11.64% | p=0.002 n=6 |
| FRDParallel_MIMO_2000-4 | sec/op | 0.0003718 | 0.0003907 | +5.09% | p=0.041 n=6 |
| FRDFeedback_SISO_200-4 | sec/op | 1.475e-05 | 1.594e-05 | +8.06% | p=0.015 n=6 |
| FRDFeedback_SISO_2000-4 | sec/op | 0.0001508 | 0.0001604 | +6.32% | p=0.041 n=6 |
| FRDFeedback_MIMO8x8_2000-4 | sec/op | 0.008048 | 0.008555 | +6.30% | p=0.002 n=6 |
| Passivity_SISO-4 | sec/op | 1.841e-05 | 1.958e-05 | +6.38% | p=0.009 n=6 |
| FrequencyDispatch/n=12/m=1/w=20/a=full/path=auto-4 | B/op | 5280 | 8816 | +66.97% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=auto-4 | B/op | 9792 | 1.333e+04 | +36.11% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | B/op | 8416 | 1.464e+04 | +73.95% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | B/op | 1.293e+04 | 1.915e+04 | +48.14% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | B/op | 1.859e+04 | 3.161e+04 | +70.01% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | B/op | 3.251e+04 | 4.553e+04 | +40.03% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | B/op | 5.584e+04 | 9.282e+04 | +66.22% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | B/op | 1.094e+05 | 1.464e+05 | +33.79% | p=0.002 n=6 |
| MarginOrder/Margin/n=10-4 | B/op | 2.623e+04 | 2.874e+04 | +9.58% | p=0.002 n=6 |
| MarginOrder/AllMargin/n=10-4 | B/op | 2.618e+04 | 2.87e+04 | +9.59% | p=0.002 n=6 |
| MarginOrder/Bandwidth/n=10-4 | B/op | 1.744e+04 | 1.995e+04 | +14.42% | p=0.002 n=6 |
| MarginOrder/Pidtune/n=10-4 | B/op | 4.749e+04 | 5.252e+04 | +10.59% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | allocs/op | 7 | 10 | +42.86% | p=0.002 n=6 |
| MarginOrder/Margin/n=10-4 | allocs/op | 27 | 30 | +11.11% | p=0.002 n=6 |
| MarginOrder/AllMargin/n=10-4 | allocs/op | 26 | 29 | +11.54% | p=0.002 n=6 |
| MarginOrder/Bandwidth/n=10-4 | allocs/op | 34 | 37 | +8.82% | p=0.002 n=6 |
| MarginOrder/Pidtune/n=10-4 | allocs/op | 69 | 75 | +8.70% | p=0.002 n=6 |

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| SimulateWithDelay_SISO-4 | sec/op | 3.444e-05 | 3.54e-05 | +2.79% | p=0.002 n=6 |
| SimulateWithDelay_MIMO-4 | sec/op | 0.0001636 | 0.0001652 | +0.98% | p=0.041 n=6 |
| FrequencySweepKernels/N4_M2_W8/Hessenberg-4 | sec/op | 6.342e-06 | 6.515e-06 | +2.73% | p=0.041 n=6 |
| FrequencySweepKernels/N4_M2_W100/Hessenberg-4 | sec/op | 5.396e-05 | 5.578e-05 | +3.36% | p=0.002 n=6 |
| FrequencySweepKernels/N8_M1_W100/Dense-4 | sec/op | 7.482e-05 | 7.595e-05 | +1.51% | p=0.041 n=6 |
| FrequencySweepKernels/N8_M1_W100/Hessenberg-4 | sec/op | 7.715e-05 | 7.823e-05 | +1.40% | p=0.002 n=6 |
| FrequencySweepKernels/N8_M2_W100/Dense-4 | sec/op | 8.91e-05 | 8.996e-05 | +0.97% | p=0.004 n=6 |
| FrequencySweepKernels/N8_M2_W100/Hessenberg-4 | sec/op | 0.0001326 | 0.0001349 | +1.73% | p=0.002 n=6 |
| FrequencySweepKernels/N12_M1_W100/Dense-4 | sec/op | 0.0001838 | 0.0001854 | +0.83% | p=0.004 n=6 |
| FrequencySweepKernels/N12_M1_W100/Hessenberg-4 | sec/op | 0.0001369 | 0.0001391 | +1.66% | p=0.002 n=6 |
| FrequencySweepKernels/N12_M2_W100/Dense-4 | sec/op | 0.0002157 | 0.000218 | +1.02% | p=0.002 n=6 |
| FrequencySweepKernels/N12_M2_W100/Hessenberg-4 | sec/op | 0.0002486 | 0.0002504 | +0.71% | p=0.026 n=6 |
| FrequencySweepKernels/N12_M4_W100/Hessenberg-4 | sec/op | 0.0004588 | 0.0004612 | +0.52% | p=0.015 n=6 |
| FrequencySweepKernels/N18_M1_W100/Dense-4 | sec/op | 0.0004829 | 0.0004851 | +0.46% | p=0.041 n=6 |
| FrequencySweepKernels/N18_M1_W100/Hessenberg-4 | sec/op | 0.0002532 | 0.0002548 | +0.64% | p=0.002 n=6 |
| FrequencySweepKernels/N18_M2_W100/Hessenberg-4 | sec/op | 0.0004548 | 0.0004578 | +0.65% | p=0.041 n=6 |
| FrequencySweepKernels/N30_M5_W100/Dense-4 | sec/op | 0.002418 | 0.002433 | +0.63% | p=0.015 n=6 |
| FrequencyDispatch/n=8/m=1/w=20/a=full/path=auto-4 | sec/op | 1.599e-05 | 1.628e-05 | +1.84% | p=0.004 n=6 |
| FrequencyDispatch/n=8/m=1/w=200/a=full/path=dense-4 | sec/op | 0.000149 | 0.0001519 | +1.98% | p=0.002 n=6 |
| FrequencyDispatch/n=8/m=1/w=200/a=full/path=hessenberg-4 | sec/op | 0.0001507 | 0.0001531 | +1.62% | p=0.002 n=6 |
| FrequencyDispatch/n=8/m=1/w=200/a=full/path=auto-4 | sec/op | 0.00015 | 0.0001519 | +1.26% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=20/a=full/path=auto-4 | sec/op | 3.867e-05 | 3.451e-05 | -10.76% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=dense-4 | sec/op | 0.000366 | 0.0003681 | +0.55% | p=0.015 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=hessenberg-4 | sec/op | 0.0002638 | 0.0002667 | +1.07% | p=0.002 n=6 |
| FrequencyDispatch/n=12/m=1/w=200/a=full/path=auto-4 | sec/op | 0.0003688 | 0.0002688 | -27.12% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=dense-4 | sec/op | 7.457e-05 | 7.566e-05 | +1.47% | p=0.004 n=6 |
| FrequencyDispatch/n=16/m=1/w=20/a=full/path=auto-4 | sec/op | 7.522e-05 | 5.313e-05 | -29.36% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=hessenberg-4 | sec/op | 0.0004107 | 0.0004129 | +0.51% | p=0.026 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=full/path=auto-4 | sec/op | 0.0007248 | 0.000413 | -43.01% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | sec/op | 2.615e-05 | 2.29e-05 | -12.41% | p=0.002 n=6 |
| FrequencyDispatch/n=20/m=1/w=200/a=full/path=auto-4 | sec/op | 0.0005824 | 0.0005868 | +0.75% | p=0.009 n=6 |
| FrequencyDispatch/n=32/m=1/w=200/a=full/path=hessenberg-4 | sec/op | 0.001292 | 0.001301 | +0.69% | p=0.002 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=hessenberg/path=dense-4 | sec/op | 0.0003606 | 0.0003629 | +0.64% | p=0.041 n=6 |
| FrequencyDispatch/n=16/m=1/w=200/a=hessenberg/path=auto-4 | sec/op | 0.0003626 | 0.0003651 | +0.69% | p=0.002 n=6 |
| FrequencyDispatch/n=32/m=1/w=200/a=hessenberg/path=hessenberg-4 | sec/op | 0.001263 | 0.001273 | +0.83% | p=0.009 n=6 |
| FrequencyDispatch/n=12/m=2/w=200/a=full/path=hessenberg-4 | sec/op | 0.0004778 | 0.0004813 | +0.75% | p=0.041 n=6 |
| FrequencyDispatch/n=24/m=2/w=20/a=full/path=auto-4 | sec/op | 0.0002252 | 0.000166 | -26.30% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=dense-4 | sec/op | 0.002171 | 0.002191 | +0.92% | p=0.026 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=auto-4 | sec/op | 0.00218 | 0.001423 | -34.72% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | sec/op | 6.097e-05 | 5.469e-05 | -10.30% | p=0.002 n=6 |
| FrequencyDispatch/n=24/m=2/w=200/a=hessenberg/path=dense-4 | sec/op | 0.0008765 | 0.00089 | +1.53% | p=0.004 n=6 |
| FrequencyDispatch/n=48/m=2/w=200/a=hessenberg/path=auto-4 | sec/op | 0.003382 | 0.003393 | +0.34% | p=0.004 n=6 |
| FrequencyDispatch/n=20/m=4/w=200/a=full/path=dense-4 | sec/op | 0.00172 | 0.001731 | +0.62% | p=0.026 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=dense-4 | sec/op | 0.0009419 | 0.0009506 | +0.92% | p=0.009 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=hessenberg-4 | sec/op | 0.0007296 | 0.0007333 | +0.52% | p=0.026 n=6 |
| FrequencyDispatch/n=40/m=4/w=20/a=full/path=auto-4 | sec/op | 0.0009496 | 0.0007381 | -22.27% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=full/path=auto-4 | sec/op | 0.009233 | 0.00661 | -28.40% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | sec/op | 0.0002057 | 0.0001954 | -4.98% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=200/a=full/path=dense-4 | sec/op | 0.01184 | 0.01192 | +0.64% | p=0.002 n=6 |
| FrequencyDispatch/n=40/m=4/w=200/a=hessenberg/path=hessenberg-4 | sec/op | 0.006601 | 0.006643 | +0.63% | p=0.026 n=6 |
| Bode-4 | sec/op | 0.0003036 | 0.0003073 | +1.21% | p=0.009 n=6 |
| Zeros_SISO-4 | sec/op | 6.517e-06 | 6.659e-06 | +2.17% | p=0.026 n=6 |
| Zeros_MIMO-4 | sec/op | 1.977e-05 | 2.031e-05 | +2.74% | p=0.015 n=6 |
| Zeros_NonSquare-4 | sec/op | 1.608e-05 | 1.639e-05 | +1.93% | p=0.009 n=6 |
| ControllabilityStaircase-4 | sec/op | 7.142e-05 | 7.271e-05 | +1.81% | p=0.015 n=6 |
| SimulateNoDelay-4 | sec/op | 8.817e-05 | 9.027e-05 | +2.38% | p=0.015 n=6 |
| ThiranDelay-4 | sec/op | 5.464e-07 | 5.659e-07 | +3.56% | p=0.002 n=6 |
| DecomposeIODelay-4 | sec/op | 3.706e-07 | 3.375e-07 | -8.94% | p=0.002 n=6 |
| PullDelaysToLFT-4 | sec/op | 2.191e-06 | 2.243e-06 | +2.40% | p=0.024 n=6 |
| GetDelayModel-4 | sec/op | 2.345e-06 | 2.428e-06 | +3.54% | p=0.009 n=6 |
| AbsorbInternalDelay-4 | sec/op | 2.604e-06 | 2.705e-06 | +3.90% | p=0.015 n=6 |
| AbsorbInternalDelayContinuous-4 | sec/op | 1.017e-05 | 1.054e-05 | +3.60% | p=0.015 n=6 |
| DiscretizeWithOpts_PathThiran/n=4/io=2/delay-4 | sec/op | 1.266e-05 | 1.293e-05 | +2.15% | p=0.009 n=6 |
| FreqResponseWithDelay-4 | sec/op | 4.771e-05 | 4.921e-05 | +3.14% | p=0.002 n=6 |
| IsStrictlyUpperTriangular-4 | sec/op | 1.675e-07 | 1.634e-07 | -2.42% | p=0.002 n=6 |
| Simulate_DCMotor-4 | sec/op | 7.183e-05 | 7.285e-05 | +1.43% | p=0.026 n=6 |
| DiscretizeAndSimulate_B747Longitudinal-4 | sec/op | 1.717e-05 | 1.746e-05 | +1.70% | p=0.026 n=6 |
| Simulate_Large-4 | sec/op | 0.0005603 | 0.0005665 | +1.11% | p=0.026 n=6 |
| Gram-4 | sec/op | 4.361e-05 | 4.419e-05 | +1.32% | p=0.002 n=6 |
| Ctrb-4 | sec/op | 3.035e-06 | 3.094e-06 | +1.93% | p=0.026 n=6 |
| HinfNorm-4 | sec/op | 0.0002347 | 0.0002408 | +2.58% | p=0.026 n=6 |
| HinfNorm_MixedSensitivityLoop/n=64-4 | sec/op | 0.01635 | 0.01649 | +0.86% | p=0.004 n=6 |
| Place_N10_M2-4 | sec/op | 6.197e-05 | 6.29e-05 | +1.50% | p=0.009 n=6 |
| Place_Random_N10_M3-4 | sec/op | 7.466e-05 | 7.604e-05 | +1.84% | p=0.015 n=6 |
| Place_Random_N30_M4-4 | sec/op | 0.0007288 | 0.0007432 | +1.98% | p=0.009 n=6 |
| Acker_N10-4 | sec/op | 4.141e-05 | 4.178e-05 | +0.89% | p=0.041 n=6 |
| Kalman_N100_M5_P5-4 | sec/op | 0.03976 | 0.04003 | +0.67% | p=0.009 n=6 |
| Reg_N10_M2_P3-4 | sec/op | 3.057e-06 | 3.174e-06 | +3.83% | p=0.002 n=6 |
| Reg_N50_M5_P5-4 | sec/op | 3.129e-05 | 3.2e-05 | +2.26% | p=0.004 n=6 |
| SSToZPK_SISO-4 | sec/op | 3.123e-05 | 3.231e-05 | +3.45% | p=0.041 n=6 |
| Sigma_MIMO-4 | sec/op | 0.0001818 | 0.0001878 | +3.27% | p=0.002 n=6 |
| FreqRespEst_MIMO-4 | sec/op | 0.0005207 | 0.0005285 | +1.49% | p=0.026 n=6 |
| Connect-4 | sec/op | 8.948e-06 | 9.094e-06 | +1.64% | p=0.041 n=6 |
| LFT-4 | sec/op | 8.436e-06 | 8.521e-06 | +1.00% | p=0.028 n=6 |
| LFT_Large-4 | sec/op | 1.663e-05 | 1.743e-05 | +4.83% | p=0.004 n=6 |
| SystemFRD_SISO_10000-4 | sec/op | 0.0009997 | 0.001036 | +3.59% | p=0.026 n=6 |
| FRDSigma_Large_2000-4 | sec/op | 0.006715 | 0.006765 | +0.75% | p=0.015 n=6 |
| FRDMargin_2000-4 | sec/op | 0.0001569 | 0.0001613 | +2.84% | p=0.041 n=6 |
| FRDMargin_NegGM_2000-4 | sec/op | 0.0001589 | 0.0001638 | +3.09% | p=0.041 n=6 |
| FRDSigma_MIMO5x5_2000-4 | sec/op | 0.009412 | 0.009461 | +0.53% | p=0.002 n=6 |
| Systune_SISO-4 | sec/op | 0.0001446 | 0.0001497 | +3.54% | p=0.015 n=6 |
| PrewarpMIMO/prewarp-4 | sec/op | 3.063e-06 | 2.996e-06 | -2.19% | p=0.024 n=6 |
| HinfNorm_InternalDelay/mimo-4 | sec/op | 0.003123 | 0.003085 | -1.21% | p=0.002 n=6 |
| MarginOrder/Margin/n=10-4 | sec/op | 0.0002233 | 0.0002279 | +2.05% | p=0.002 n=6 |
| MarginOrder/AllMargin/n=10-4 | sec/op | 0.0002224 | 0.0002271 | +2.13% | p=0.002 n=6 |
| MarginOrder/Pidtune/n=10-4 | sec/op | 0.0003915 | 0.000399 | +1.90% | p=0.002 n=6 |
| DiskMargin_PIDExactDelay/internal/Margin-4 | sec/op | 0.001111 | 0.001135 | +2.16% | p=0.004 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | B/op | 2.134e+04 | 1.319e+04 | -38.19% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | B/op | 4.146e+04 | 2.37e+04 | -42.83% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | B/op | 1.107e+05 | 6.14e+04 | -44.52% | p=0.002 n=6 |
| LFT_Large-4 | B/op | 3.579e+04 | 3.579e+04 | +0.00% | p=0.011 n=6 |
| D2C_ZOH_N5-4 | B/op | 6530 | 6537 | +0.10% | p=0.004 n=6 |
| D2C_Tustin_N20-4 | B/op | 1.518e+04 | 1.519e+04 | +0.04% | p=0.045 n=6 |
| MatLog_N20-4 | B/op | 6.604e+04 | 6.639e+04 | +0.54% | p=0.004 n=6 |
| Step_SISO_N10-4 | B/op | 5.511e+04 | 5.513e+04 | +0.04% | p=0.004 n=6 |
| Impulse_SISO_N2-4 | B/op | 1.854e+04 | 1.854e+04 | +0.02% | p=0.039 n=6 |
| Loopsens_SISO_N10-4 | B/op | 3.313e+04 | 3.314e+04 | +0.01% | p=0.022 n=6 |
| FractionalFeedbackHold/foh/mixed-output-4 | B/op | 2.253e+05 | 2.253e+05 | +0.01% | p=0.026 n=6 |
| HinfNorm_InternalDelay/mimo-4 | B/op | 1.205e+05 | 1.205e+05 | -0.01% | p=0.048 n=6 |
| TransferFunction/pade18-4 | B/op | 7.762e+04 | 7.769e+04 | +0.09% | p=0.015 n=6 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-4 | allocs/op | 10 | 7 | -30.00% | p=0.002 n=6 |

## Inconclusive

1236 case metrics show no significant difference. This is not evidence of equivalence.

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
