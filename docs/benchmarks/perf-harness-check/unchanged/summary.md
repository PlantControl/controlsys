# Benchmark comparison

Status: complete. Advisory evidence, not a pass/fail performance gate.

| | Baseline | Candidate |
| --- | --- | --- |
| Ref | `origin/main` | `HEAD` |
| Commit | `c837ec31fe6a5e71130b87a52b2f7237484bd8f6` | `715423c2934740a723b4f8a36f08191e69afac86` |
| Backend | unreported | goos=darwin goarch=arm64 backend=gonum.Implementation/pure-go |

- Go: go version go1.27.1 darwin/arm64; GOAMD64= GOARM64=v8.0 CGO_ENABLED=1 GOFLAGS="" GOEXPERIMENT=""
- Host: darwin/arm64, Apple M1 Pro, 8 CPUs, GOMAXPROCS env ""
- Samples: 10 interleaved rounds per revision; flags `-test.run=^$ -test.bench=^Benchmark(FrequencySweepKernels|Bode|D2C_ZOH_N20|DiscretizeZOH)$ -test.benchtime=100ms -test.benchmem -test.count=1`
- **Warning:** baseline and candidate report different backends.

## Review items

Significant (benchstat, alpha 0.05) time or B/op increases of at least 5%, and any allocs/op increase.

None.

## Other significant changes

Improvements and smaller regressions; report both.

| Benchmark | Unit | Baseline | Candidate | Change | p |
| --- | --- | ---: | ---: | ---: | --- |
| FrequencySweepKernels/N64_M1_W100/Hessenberg-8 | B/op | 1.917e+05 | 1.917e+05 | -0.00% | p=0.033 n=10 |

## Inconclusive

110 case metrics show no significant difference. This is not evidence of equivalence.

## Missing or incompatible cases

Compared on both revisions: 37 cases.


Full table: benchstat.txt. Raw samples: baseline.txt, candidate.txt, raw/. Metadata: metadata.json.
