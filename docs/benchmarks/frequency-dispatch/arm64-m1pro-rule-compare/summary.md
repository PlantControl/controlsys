# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `9170085` | `4409607` |
| Commit | `9170085eb71166dab5d537009d9f8ec917a4b788` | `440960763eafc4bebddd321678ca1e7467f82f36` |
| Backend | goos=darwin goarch=arm64 backend=gonum.Implementation/pure-go | goos=darwin goarch=arm64 backend=gonum.Implementation/pure-go |

- Go: go version go1.27.1 darwin/arm64; GOAMD64= GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: darwin/arm64, Apple M1 Pro, 8 CPUs, GOMAXPROCS env ""
- Samples: 10 interleaved rounds per revision; flags `-test.run=^$ -test.bench=^(BenchmarkFrequencyDispatch|BenchmarkFreqResponse|BenchmarkFreqResponse_ShortSweep|BenchmarkFreqResponse_B747Lateral|BenchmarkBode|BenchmarkBode_LargeMIMO|BenchmarkSigma_SISO|BenchmarkSigma_MIMO|BenchmarkNichols)$ -test.benchtime=100ms -test.benchmem -test.count=1`

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

None.

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-8 | sec/op | 2.672e-05 | 1.153e-05 | -56.86% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=5/a=full/path=auto-8 | sec/op | 3.18e-05 | 1.806e-05 | -43.22% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=10/a=full/path=auto-8 | sec/op | 4.49e-05 | 3.417e-05 | -23.90% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-8 | sec/op | 6.994e-05 | 6.514e-05 | -6.86% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=200/a=full/path=hessenberg-8 | sec/op | 0.0005087 | 0.000516 | +1.45% | p=0.009 n=10 |
| FrequencyDispatch/n=20/m=1/w=200/a=full/path=auto-8 | sec/op | 0.0005105 | 0.0005192 | +1.70% | p=0.001 n=10 |
| FrequencyDispatch/n=24/m=1/w=3/a=full/path=auto-8 | sec/op | 3.878e-05 | 1.789e-05 | -53.88% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=5/a=full/path=auto-8 | sec/op | 4.446e-05 | 2.778e-05 | -37.52% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=10/a=full/path=auto-8 | sec/op | 5.89e-05 | 5.238e-05 | -11.08% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=200/a=full/path=hessenberg-8 | sec/op | 0.0006035 | 0.0006148 | +1.88% | p=0.043 n=10 |
| FrequencyDispatch/n=24/m=1/w=200/a=full/path=auto-8 | sec/op | 0.0006042 | 0.0006152 | +1.83% | p=0.015 n=10 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=dense-8 | sec/op | 3.633e-05 | 3.575e-05 | -1.60% | p=0.029 n=10 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-8 | sec/op | 7.496e-05 | 3.674e-05 | -50.99% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=5/a=full/path=auto-8 | sec/op | 8.404e-05 | 5.773e-05 | -31.31% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=10/a=full/path=auto-8 | sec/op | 0.0001059 | 0.0001108 | +4.59% | p=0.000 n=10 |
| FrequencyDispatch/n=12/m=2/w=50/a=full/path=hessenberg-8 | sec/op | 0.000109 | 0.0001102 | +1.15% | p=0.035 n=10 |
| FrequencyDispatch/n=12/m=2/w=200/a=full/path=hessenberg-8 | sec/op | 0.000415 | 0.0004208 | +1.40% | p=0.011 n=10 |
| FrequencyDispatch/n=24/m=2/w=50/a=full/path=hessenberg-8 | sec/op | 0.0002821 | 0.0002854 | +1.17% | p=0.035 n=10 |
| FrequencyDispatch/n=24/m=2/w=200/a=full/path=hessenberg-8 | sec/op | 0.00103 | 0.001051 | +2.05% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-8 | sec/op | 6.515e-05 | 2.807e-05 | -56.91% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=5/a=full/path=hessenberg-8 | sec/op | 7.791e-05 | 7.938e-05 | +1.89% | p=0.043 n=10 |
| FrequencyDispatch/n=28/m=2/w=5/a=full/path=auto-8 | sec/op | 7.843e-05 | 4.45e-05 | -43.26% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=10/a=full/path=auto-8 | sec/op | 0.0001104 | 8.424e-05 | -23.72% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-8 | sec/op | 0.0001736 | 0.000165 | -4.95% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=50/a=full/path=hessenberg-8 | sec/op | 0.0003665 | 0.0003745 | +2.18% | p=0.002 n=10 |
| FrequencyDispatch/n=28/m=2/w=50/a=full/path=auto-8 | sec/op | 0.0003659 | 0.0003759 | +2.71% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=200/a=full/path=hessenberg-8 | sec/op | 0.001327 | 0.001356 | +2.13% | p=0.002 n=10 |
| FrequencyDispatch/n=28/m=2/w=200/a=full/path=auto-8 | sec/op | 0.001325 | 0.001347 | +1.65% | p=0.007 n=10 |
| FrequencyDispatch/n=36/m=2/w=3/a=full/path=auto-8 | sec/op | 0.0001164 | 5.367e-05 | -53.89% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=5/a=full/path=auto-8 | sec/op | 0.0001366 | 8.535e-05 | -37.51% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=10/a=full/path=auto-8 | sec/op | 0.0001853 | 0.0001624 | -12.36% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=50/a=full/path=hessenberg-8 | sec/op | 0.000576 | 0.0005805 | +0.79% | p=0.043 n=10 |
| FrequencyDispatch/n=36/m=2/w=50/a=full/path=auto-8 | sec/op | 0.0005753 | 0.0005856 | +1.79% | p=0.011 n=10 |
| FrequencyDispatch/n=36/m=2/w=200/a=full/path=auto-8 | sec/op | 0.002043 | 0.002082 | +1.91% | p=0.019 n=10 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-8 | sec/op | 0.0002397 | 0.0001115 | -53.46% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=5/a=full/path=auto-8 | sec/op | 0.0002721 | 0.0001787 | -34.33% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=2/w=3/a=hessenberg/path=hessenberg-8 | sec/op | 2.483e-05 | 2.538e-05 | +2.25% | p=0.043 n=10 |
| FrequencyDispatch/n=24/m=2/w=20/a=hessenberg/path=hessenberg-8 | sec/op | 0.0001094 | 0.0001129 | +3.15% | p=0.019 n=10 |
| FrequencyDispatch/n=24/m=2/w=200/a=hessenberg/path=hessenberg-8 | sec/op | 0.00101 | 0.001026 | +1.62% | p=0.009 n=10 |
| FrequencyDispatch/n=48/m=2/w=20/a=hessenberg/path=auto-8 | sec/op | 0.0001555 | 0.0001572 | +1.14% | p=0.035 n=10 |
| FrequencyDispatch/n=48/m=2/w=200/a=hessenberg/path=dense-8 | sec/op | 0.001429 | 0.00144 | +0.79% | p=0.043 n=10 |
| FrequencyDispatch/n=20/m=4/w=1/a=full/path=hessenberg-8 | sec/op | 2.929e-05 | 3.007e-05 | +2.66% | p=0.023 n=10 |
| FrequencyDispatch/n=20/m=4/w=2/a=full/path=hessenberg-8 | sec/op | 3.708e-05 | 3.79e-05 | +2.22% | p=0.019 n=10 |
| FrequencyDispatch/n=20/m=4/w=3/a=full/path=hessenberg-8 | sec/op | 4.506e-05 | 4.544e-05 | +0.83% | p=0.029 n=10 |
| FrequencyDispatch/n=20/m=4/w=10/a=full/path=hessenberg-8 | sec/op | 9.875e-05 | 0.0001015 | +2.83% | p=0.003 n=10 |
| FrequencyDispatch/n=20/m=4/w=20/a=full/path=hessenberg-8 | sec/op | 0.0001767 | 0.0001812 | +2.57% | p=0.001 n=10 |
| FrequencyDispatch/n=20/m=4/w=50/a=full/path=hessenberg-8 | sec/op | 0.00041 | 0.000421 | +2.67% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=4/w=200/a=full/path=hessenberg-8 | sec/op | 0.001559 | 0.001608 | +3.15% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-8 | sec/op | 0.0002377 | 9.881e-05 | -58.43% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=5/a=full/path=auto-8 | sec/op | 0.0002916 | 0.000161 | -44.80% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=10/a=full/path=auto-8 | sec/op | 0.0004288 | 0.000314 | -26.77% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-8 | sec/op | 0.0006965 | 0.0006178 | -11.30% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=50/a=full/path=auto-8 | sec/op | 0.00151 | 0.001534 | +1.61% | p=0.003 n=10 |
| FrequencyDispatch/n=60/m=4/w=3/a=full/path=auto-8 | sec/op | 0.0005163 | 0.000222 | -57.00% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=5/a=full/path=auto-8 | sec/op | 0.0006149 | 0.0003625 | -41.05% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=10/a=full/path=hessenberg-8 | sec/op | 0.0008566 | 0.000868 | +1.33% | p=0.043 n=10 |
| FrequencyDispatch/n=60/m=4/w=10/a=full/path=auto-8 | sec/op | 0.0008548 | 0.0007123 | -16.67% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-8 | sec/op | 0.001112 | 0.0004827 | -56.58% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=5/a=full/path=auto-8 | sec/op | 0.001293 | 0.0007877 | -39.10% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=10/a=full/path=auto-8 | sec/op | 0.001748 | 0.001564 | -10.50% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=20/a=full/path=hessenberg-8 | sec/op | 0.002656 | 0.00264 | -0.59% | p=0.029 n=10 |
| FrequencyDispatch/n=80/m=4/w=50/a=full/path=auto-8 | sec/op | 0.005378 | 0.00533 | -0.89% | p=0.023 n=10 |
| FrequencyDispatch/n=80/m=4/w=20/a=hessenberg/path=dense-8 | sec/op | 0.0005661 | 0.0005739 | +1.38% | p=0.023 n=10 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-8 | B/op | 2.134e+04 | 1.319e+04 | -38.19% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=5/a=full/path=auto-8 | B/op | 2.14e+04 | 1.325e+04 | -38.09% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=10/a=full/path=auto-8 | B/op | 2.151e+04 | 1.336e+04 | -37.90% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-8 | B/op | 2.175e+04 | 1.36e+04 | -37.48% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=3/a=full/path=auto-8 | B/op | 2.845e+04 | 1.658e+04 | -41.70% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=5/a=full/path=auto-8 | B/op | 2.85e+04 | 1.664e+04 | -41.62% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=10/a=full/path=auto-8 | B/op | 2.862e+04 | 1.675e+04 | -41.46% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-8 | B/op | 4.958e+04 | 2.9e+04 | -41.50% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=5/a=full/path=auto-8 | B/op | 4.963e+04 | 2.906e+04 | -41.46% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=10/a=full/path=auto-8 | B/op | 4.974e+04 | 2.917e+04 | -41.36% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-8 | B/op | 4.146e+04 | 2.37e+04 | -42.83% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=5/a=full/path=auto-8 | B/op | 4.162e+04 | 2.386e+04 | -42.68% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=10/a=full/path=auto-8 | B/op | 4.197e+04 | 2.421e+04 | -42.32% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-8 | B/op | 4.269e+04 | 2.493e+04 | -41.60% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=3/a=full/path=auto-8 | B/op | 6.554e+04 | 3.817e+04 | -41.77% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=5/a=full/path=auto-8 | B/op | 6.57e+04 | 3.832e+04 | -41.67% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=10/a=full/path=auto-8 | B/op | 6.605e+04 | 3.867e+04 | -41.45% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-8 | B/op | 1.122e+05 | 6.294e+04 | -43.91% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=5/a=full/path=auto-8 | B/op | 1.124e+05 | 6.309e+04 | -43.85% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=10/a=full/path=auto-8 | B/op | 1.127e+05 | 6.344e+04 | -43.71% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-8 | B/op | 1.107e+05 | 6.14e+04 | -44.52% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=5/a=full/path=auto-8 | B/op | 1.112e+05 | 6.194e+04 | -44.30% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=10/a=full/path=auto-8 | B/op | 1.126e+05 | 6.338e+04 | -43.74% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-8 | B/op | 1.154e+05 | 6.614e+04 | -42.69% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=50/a=full/path=auto-8 | B/op | 1.239e+05 | 7.459e+04 | -39.78% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=3/a=full/path=auto-8 | B/op | 1.744e+05 | 1.006e+05 | -42.31% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=5/a=full/path=auto-8 | B/op | 1.75e+05 | 1.012e+05 | -42.18% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=10/a=full/path=auto-8 | B/op | 1.764e+05 | 1.026e+05 | -41.83% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-8 | B/op | 2.976e+05 | 1.746e+05 | -41.33% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=5/a=full/path=auto-8 | B/op | 2.982e+05 | 1.752e+05 | -41.26% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=10/a=full/path=auto-8 | B/op | 2.996e+05 | 1.766e+05 | -41.06% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=20/m=1/w=20/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=24/m=1/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=32/m=1/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=28/m=2/w=20/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=36/m=2/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=48/m=2/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=20/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=44/m=4/w=50/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=60/m=4/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=3/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=5/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |
| FrequencyDispatch/n=80/m=4/w=10/a=full/path=auto-8 | allocs/op | 10 | 7 | -30.00% | p=0.000 n=10 |

## Inconclusive

1141 case metrics show no significant difference. This is not evidence of equivalence.

## Missing or incompatible cases

Compared on both revisions: 422 cases.


Full table: benchstat.txt. Raw samples: baseline.txt, candidate.txt, raw/. Metadata: metadata.json.
