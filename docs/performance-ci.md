# Performance and native-architecture CI

## Correctness on native architectures

`.github/workflows/ci.yml` runs the `test` and `bench` jobs on `ubuntu-latest`
(linux/amd64) and `macos-latest` (darwin/arm64). Before testing, each `test`
job prints `go env`, `uname` and the CPU model, then asserts:

- `go env GOARCH` and `uname -m` match the matrix entry (macOS also rejects
  Rosetta translation through `sysctl.proc_translated`);
- `TestNativeBackend` (`backend_test.go`) reports the expected Gonum backend:
  `gonum.Implementation/amd64-asm` on amd64, `gonum.Implementation/pure-go` on
  arm64. The suffix mirrors the build constraint of Gonum's `internal/asm/f64`
  kernels; this Gonum fork has no arm64 assembly.

Cross-compilation or emulation only shows compatibility, not native behavior or
speed.

## Comparative performance

`.github/workflows/perf.yml` compares an explicit baseline with a candidate on
one `ubuntu-latest` runner. It is advisory: it fails only when the comparison
itself is invalid, never because a benchmark got slower.

Triggers:

- Manual `workflow_dispatch`, with optional base, candidate, benchmark regex,
  rounds, benchtime and the crossover job.
- Pull requests labelled `performance` (comparison and crossover). Other pull
  requests skip it; a full comparison takes 30–45 minutes.

The default baseline is the merge base of the candidate and the PR base (or
`origin/main`); the candidate is the PR head, not the merge commit. Pull
requests use 6 rounds of 100ms; manual runs default to 10.

### Harness

`scripts/benchcmp` (`go run ./scripts/benchcmp`) does the comparison and runs
the same way locally:

1. Resolves both revisions to commits. `-candidate .` builds the working tree
   and records the diff and its SHA-256.
2. Exports each commit with `git archive` and builds it with
   `GOWORK=off go test -c`. Both binaries exist before any timing.
3. Runs the binaries alternately, baseline first on odd rounds and candidate
   first on even rounds, with `-test.benchmem -test.count=1`.
4. Writes `baseline.txt`, `candidate.txt`, `raw/`, `benchstat.txt`,
   `benchstat.csv`, `summary.md` and `metadata.json` to a new directory.

`metadata.json` records both refs and commits, dirty state, Go version, `go env`
(GOARCH, GOAMD64/GOARM64, CGO, GOFLAGS, GOEXPERIMENT), OS, CPU model, CPU count,
`GOMAXPROCS`/`GODEBUG`/`GOGC`, runner image, flags, rounds, binary SHA-256 and
build info, each binary's backend, the benchstat version and the run order.

The comparison fails, writes no benchstat output and gets a `FAILED` summary
when a build fails, a benchmark process exits non-zero, prints `--- FAIL` or
`panic:`, lacks the `PASS` trailer, reports a malformed or non-finite metric,
selects no benchmarks, or changes its case set between rounds. Cases present on
only one revision and cases with different metric units are listed explicitly;
they are not compared.

Local use (benchstat from `golang.org/x/perf/cmd/benchstat`, pinned in CI to
`v0.0.0-20260312031701-16a31bc5fbd0`):

```sh
go run ./scripts/benchcmp -base origin/main -candidate HEAD \
  -bench '^BenchmarkFrequencySweepKernels$' -rounds 10 -benchtime 100ms \
  -out /tmp/perf-compare
```

Keep the output directory outside the repository when using `-candidate .`.
Pause builds, tests and other CPU-heavy work while sampling.

### Review criteria

`summary.md` lists:

- **Review items**: benchstat-significant (alpha 0.05) increases of at least 5%
  in sec/op or B/op, and any significant allocs/op increase.
- **Other significant changes**: improvements and smaller regressions.
  Report both sides of a tradeoff.
- **Inconclusive** count: no significant difference is not evidence of
  equivalence.
- Missing, renamed, skipped and incompatible cases.

A reviewer should rerun review items with a focused `-bench`, more rounds or a
longer `-benchtime` before accepting or rejecting them, and explain any accepted
regression in the PR. With hundreds of cases, some significant results are
false positives; one-off small effects need confirmation. A CI comparison is
not a substitute for measuring on target hardware.

### Hardware limitations

GitHub-hosted runners are shared VMs: no CPU affinity, frequency or turbo
control, noisy neighbours, and the CPU model can differ between runs (it is
recorded). Only comparisons within a single job are meaningful; never compare
timings across runs or runners. `ubuntu-latest` provides 4 vCPUs on x86-64,
`macos-latest` 3 Apple M-series cores. No paid or self-hosted runners are used.
A full-suite comparison takes roughly 45-75s per pass per revision on an
Apple M1 Pro; expect several times longer on runners.

## Frequency dispatch crossover

`BenchmarkFrequencyDispatch` times the dense and Hessenberg sweep kernels and
the public `FreqResponse` (`path=auto`) around `useDenseSweep`'s scale
`c = 8m+8` (n in c/2..2c) on 1- to 200-point grids, with coupled and already
upper-Hessenberg A. The rule keeps dense while
`(s0 + s1/n)/nw + kappa·c/n + rho0 >= 1`, with constants per Gonum kernel
backend (`backend_asm.go`, `backend_noasm.go`).
`scripts/crossover.sh OUT [ROUNDS] [BENCHTIME]` builds once, runs the rounds,
and writes `environment.txt`, `raw.txt` and `benchstat -col /path` tables. The
perf workflow's `crossover` job runs it on `ubuntu-latest` (amd64) and
`macos-latest` (arm64);
`docs/benchmarks/frequency-dispatch/evalrule.py` scores a rule against its
`raw.txt`.

Evidence (retained in
[benchmarks/frequency-dispatch](benchmarks/frequency-dispatch/README.md)):

- linux/amd64 (amd64-asm, EPYC 7763): CI runs 37404887559, 37408022253,
  37409545812. Benchstat intervals are about ±1%.
- darwin/arm64 (pure-go): local Apple M1 Pro (±1%) and `macos-latest`
  (Apple M1 Virtual, 3 cores). The `macos-latest` crossover is too noisy for
  single-run decisions (median interval ±17-38%); rerun the job or pool
  attempts before acting on it.

## Harness verification

Local runs on Apple M1 Pro, retained in
[benchmarks/perf-harness-check](benchmarks/perf-harness-check):

- `unchanged/`: `origin/main` vs this branch (identical library code, 37
  cases, 10 rounds): no review items; one B/op difference below 0.01%.
- `slowdown/`: a temporary change repeating the balanced realization four
  times in `newHessenbergSweep` (`candidate.diff`, removed afterwards). Every
  Hessenberg case shows +4 allocs/op and higher B/op; N4_M2_W8 time +17.7%;
  other Hessenberg time increases of 1.7-4.0% appear as smaller significant
  regressions. Dense cases are unaffected.
- `scripts/benchcmp` tests cover failed, panicking and empty runs, case-set
  differences, missing cases, run order and refusal to overwrite output.
