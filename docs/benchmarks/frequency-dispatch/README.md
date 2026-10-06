# Frequency dispatch crossover

`BenchmarkFrequencyDispatch`, collected with `scripts/crossover.sh OUT 10 100ms`.
`path=dense` and `path=hessenberg` force the sweep kernel (including its setup);
`path=auto` is the public `FreqResponse`. Models are fully coupled
(`benchDenseSys`, square m=p) unless `a=hessenberg`, where entries below the
subdiagonal are zeroed. Benchstat compares each path with dense.

## darwin/arm64, Apple M1 Pro (`arm64-m1pro/`)

Commit `b4357dac6be0d5fd106877f394f7b63841dc1636`, Go 1.27.1,
`gonum.Implementation/pure-go`, GOMAXPROCS 8, 10 rounds x 100ms per case.
Normal desktop host without affinity control; other agents were running (load
average 3-6), which explains the wide intervals on a few rows.

Long grids (w=200), around n = 8m+8:

| m | n | dense | hessenberg | auto |
| --- | --- | ---: | ---: | --- |
| 1 | 12 | 189us | +32% | dense |
| 1 | 16 | 368us | ~ (p=0.089) | dense |
| 1 | 20 | 634us | -20% | hessenberg |
| 2 | 20 | 699us | +26% | dense |
| 2 | 24 | 1.097ms | -5.5% | dense |
| 2 | 28 | 1.613ms | -17% | hessenberg |
| 4 | 36 | 3.705ms | +4.7% | dense |
| 4 | 40 | 4.822ms | ~ (p=0.089) | dense |
| 4 | 44 | 6.171ms | -9.2% | hessenberg |

- The n = 8m+8 crossover fits long grids on this host; at m=2 the threshold
  is one step conservative (Hessenberg 5.5% faster at n=24).
- Already upper-Hessenberg A: dense is 1.6-3.4x faster at every size, and
  auto stays dense.
- `auto` medians are within about 3% of the kernel it selects, so validation
  and dispatch overhead is small at these sizes.
- Short grids: with n above the threshold, auto picks Hessenberg although
  dense is 2.1-2.4x faster at w=3 (n=20/m=1 to 80/4) and 5.6-14% faster at w=20
  just above the crossover. Hessenberg wins at w=20 for larger n (32/1 -30%,
  48/2 -25%, 80/4 -13%). The rule only sends `nw <= 2` to dense.

No threshold was changed in that run; the follow-up below makes the rule
grid-length aware.

## Grid-length-aware rule (darwin/arm64, Apple M1 Pro; superseded)

PR #325. Fitted on arm64 only; on amd64-asm it loses up to 2.1x (see
[Backend-aware rule](#backend-aware-rule)). Kept for its arm64 evidence.

### Crossover data (`arm64-m1pro-grid/`)

Commit `9170085eb71166dab5d537009d9f8ec917a4b788` (bench grid only, old rule),
Go 1.27.1, pure-go backend, GOMAXPROCS 8, `scripts/crossover.sh OUT 10 100ms`.
Load average 2.7-4.0 during the run (other agents active), not recorded by the
script. n in {c/2, c, c+4, 3c/2, 2c} with c = 8m+8, w in {1, 2, 3, 5, 10, 20,
50, 200}, plus upper-Hessenberg A at n in {c, 2c}, w in {3, 20, 200}.

Median Hessenberg/dense time ratio (below 1: Hessenberg faster):

| m | n | w=1 | 2 | 3 | 5 | 10 | 20 | 50 | 200 | rule switches at w |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 4.00 | 3.41 | 3.06 | 2.59 | 2.22 | 2.01 | 1.85 | 1.76 | never |
| 1 | 16 | 4.47 | 3.22 | 2.55 | 2.00 | 1.52 | 1.27 | 1.10 | 1.03 | never |
| 1 | 20 | 4.07 | 2.88 | 2.32 | 1.74 | 1.31 | 1.05 | 0.89 | 0.81 | 30 |
| 1 | 24 | 4.37 | 2.83 | 2.15 | 1.61 | 1.12 | 0.86 | 0.69 | 0.61 | 18 |
| 1 | 32 | 4.33 | 2.70 | 2.06 | 1.46 | 0.96 | 0.69 | 0.53 | 0.45 | 12 |
| 2 | 12 | 4.69 | 3.68 | 3.25 | 2.86 | 2.43 | 2.21 | 2.07 | 2.00 | never |
| 2 | 24 | 4.51 | 3.04 | 2.43 | 1.89 | 1.42 | 1.18 | 1.03 | 0.95 | never |
| 2 | 28 | 4.42 | 2.93 | 2.33 | 1.77 | 1.31 | 1.06 | 0.90 | 0.83 | 42 |
| 2 | 36 | 4.48 | 2.84 | 2.18 | 1.61 | 1.14 | 0.89 | 0.73 | 0.65 | 18 |
| 2 | 48 | 4.75 | 2.88 | 2.18 | 1.53 | 1.02 | 0.75 | 0.59 | 0.52 | 12 |
| 4 | 20 | 4.67 | 3.51 | 2.95 | 2.53 | 2.16 | 1.98 | 1.85 | 1.78 | never |
| 4 | 40 | 4.78 | 3.06 | 2.47 | 1.86 | 1.44 | 1.18 | 1.05 | 0.97 | never |
| 4 | 44 | 4.84 | 3.07 | 2.39 | 1.82 | 1.37 | 1.13 | 0.97 | 0.90 | 66 |
| 4 | 60 | 5.13 | 3.07 | 2.32 | 1.71 | 1.20 | 0.95 | 0.80 | 0.72 | 18 |
| 4 | 80 | 5.47 | 3.16 | 2.34 | 1.65 | 1.13 | 0.86 | 0.69 | 0.62 | 12 |

Already upper-Hessenberg A: Hessenberg is 1.5-3.4x slower than dense at every
w measured; A stays dense.

### Cost model and rule

Fitting `ratio(w) = sigma/w + rho` between w=10 and w=200 gives a Hessenberg
setup of sigma = 4.9-5.4 dense points for every (n, m) above c, and a per-point
ratio rho of roughly c/n (rho·n is 14-16 for m=1, 22-24 for m=2, 38-48 for
m=4). The sweep wins once w·(1 - rho) > sigma, so with rho = c/n:

    dense  iff  w·(n - c) < 6n   (c = 8m+8), or A is upper Hessenberg

This keeps n <= c dense for any grid, switches at 12 points for n = 2c, and
needs longer grids just above c, where the per-point saving is small. The
setup constant 6 (slightly above the fit) was chosen over 4, 5, 7, 8 and
alternative state thresholds (9m+8, 10m+6, 10m+8, 12m+4) by the worst-case
loss against the faster forced kernel on the cases above: the worst is 5%
(n=32/m=1/w=10, picks dense at ratio 0.96; n=24/m=2/w=200, the unchanged
state threshold). The old rule (`w <= 2 || n <= c`) loses up to 2.4x on
short grids. The implementation compares `w <= (6n-1)/(n-c)` so callers that
pass `math.MaxInt` (margins, H-infinity with delays) cannot overflow.

### Old vs new rule (`arm64-m1pro-rule-compare/`)

`scripts/benchcmp -base 9170085 -candidate 4409607`, 10 interleaved rounds,
100ms, `BenchmarkFrequencyDispatch` plus `FreqResponse`, `FreqResponse_ShortSweep`,
`FreqResponse_B747Lateral`, `Bode`, `Bode_LargeMIMO`, `Sigma_SISO`,
`Sigma_MIMO`, `Nichols`. No review items (no significant increase of 5% or
more, no allocs/op increase).

- `path=auto` on short grids above c: -51% to -58% at w=3, -31% to -45% at
  w=5, -11% to -27% at w=10, -5% to -11% at w=20 (n=20/1, 28/2, 44/4).
  B/op -38% to -45% and allocs 10 to 7 where the path changed.
- Regressions where the rule picks dense although Hessenberg is slightly
  faster: n=32/m=1/w=10 +4.6%, n=44/m=4/w=50 +1.6%.
- Forced `path=dense` and `path=hessenberg` (identical code) shift by up to
  +3% with p < 0.05 in several rows (mostly Hessenberg, candidate slower).
  Treat changes of this size, including +1.7-2.7% on unchanged `auto` long-grid
  rows, as the noise floor of this loaded host.
- End-to-end callers: no significant change (all `~`); they use 100-200
  points or small/banded models that were already dense.

### amd64 (`amd64-ci-crossover/`, `amd64-ci-rule-compare/`)

`Performance comparison` run 37404887559 (AMD EPYC 7763 crossover; EPYC 9V45
comparison, amd64-asm). Hessenberg is far cheaper per point than on arm64
(rho about 0.6c/n, setup sigma 1.5-2.5 dense points), so it already wins at
n = c for w >= 10. Against the old rule, the grid-length rule regresses
`path=auto` by +87% (n=20/m=1/w=20), +60% (28/2/20), +40% (44/4/20) and
+12-26% at w=3.

## Backend-aware rule

### Rule

`useDenseSweep` keeps per-point GEPP while

    (s0 + s1/n)/nw + kappa·c/n + rho0 >= 1     (c = 8m+8)

or A is already upper Hessenberg. The left side is the modelled
Hessenberg/dense time ratio: setup s0 + s1/n dense points, then
kappa·c/n + rho0 per point. The constants are chosen at build time with the
build constraint of Gonum's `internal/asm/f64` kernels (`backend_asm.go`,
`backend_noasm.go`), in twentieths so the test is an integer comparison
`nw <= (20·s0·n + 20·s1) / ((20 - 20·rho0)·n - 20·kappa·c)` that cannot
overflow for `nw = math.MaxInt`:

| backend | s0 | s1 | kappa | rho0 | always dense | Hessenberg at n = c | at n = 2c |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| pure-go (arm64, `noasm`) | 6 | 0 | 0.9 | 0 | n <= 0.9c | > 60 points | > 10 points |
| amd64-asm | 1 | 16 | 0.5 | 0.15 | n <= 10c/17 | > 4-5 points | > 2 points |

pure-go keeps the #325 form with kappa 0.9 instead of 1. amd64 needs the
extra terms: its fitted setup falls from about 3 dense points at n=12/m=1 to
1.1 at n=80/m=4, and its rho·n/c rises with n (0.5 to 0.85), so a two-term
rule fitted there loses 38% at n=3c/4, w=200. Constants come from a least
squares fit of the ratio model (w >= 3, ratio 0.5-2) on
`amd64-ci-crossover/` and CI run 37408022253, rounded.

`evalrule.py` scores rules offline: worst loss of the chosen kernel against
the faster forced kernel ("vs best") and against the old rule `w <= 2 or
n <= c` ("vs main"), over coupled A:

    python3 evalrule.py --rule 1 16 0.5 0.15 --rule 6 0 0.9 0 RAW...
    uv run --with numpy python evalrule.py --fit RAW...   # refit

### Verification

Bench grid adds n = 3c/4 (n in c/2, 3c/4, c, c+4, 3c/2, 2c). Worst case of
the shipped rule (`(n, m, w)`):

| data | host | vs best | vs main |
| --- | --- | ---: | ---: |
| `amd64-ci-crossover/` (fit) | EPYC 7763 | +1.7% (80,4,2) | +0.6% (24,2,5) |
| `ci-crossover-37408022253/amd64` (fit) | EPYC 7763 | +3.0% (80,4,2) | +2.1% (30,4,10) |
| `ci-crossover-37409545812/amd64` (holdout) | EPYC 7763 | +3.2% (80,4,2) | +1.1% (24,2,5) |
| `arm64-m1pro-grid/` (fit) | M1 Pro | +4.6% (32,1,10) | +4.6% (32,1,10) |
| `arm64-m1pro-backend/` (holdout) | M1 Pro | +3.7% (32,1,10) | +3.7% (32,1,10) |
| `ci-crossover-37408022253/arm64` | M1 (Virtual) | +5.5% (16,1,200) | +5.5% (16,1,200) |
| `ci-crossover-37409545812/arm64` | M1 (Virtual) | +19.9% (48,2,10) | +19.9% (48,2,10) |
| `ci-crossover-37409545812/arm64-attempt2` | M1 (Virtual) | +2.4% (16,1,200) | +2.4% (16,1,200) |
| all three macos-latest runs pooled | M1 (Virtual) | +2.5% (16,1,200) | +2.5% (16,1,200) |

The rule switches near ratio 1, so these are cases where the two kernels are
within a few percent; the old rule loses up to 75% (amd64) and 2.4x (arm64)
on the same data. The `macos-latest` runner is noisy (median benchstat
interval ±17-38%, ±1% for the amd64 runner and the local M1 Pro); the 19.9%
row comes from the noisiest attempt (±38%, dense times 30-40% above the
other attempts) and does not repeat in the rerun or the pooled samples.
n = c with m=1 at w=200 is a tie on pure-go (ratio 1.00-1.06) and either
choice of kappa moves the worst case between (16,1,200) and (24,2,200).

`arm64-m1pro-backend/`: commit `8e56d2c`, Go 1.27.1, pure-go, GOMAXPROCS 8,
`scripts/crossover.sh OUT 10 100ms`, load average 1.3-2.8.

`ci-crossover-37409545812/compare-summary.md` (`scripts/benchcmp`, merge base
vs `8e56d2c`, EPYC 7763, 6 rounds): `path=auto` improves by up to 43% (n=16/m=1/w=200)
and shows no time regression; where the path moves to Hessenberg B/op and
allocs/op rise (7 to 10 allocs), as for any Hessenberg sweep. Unrelated rows
flagged at 5-37% (FRD arithmetic, `FreqResponse_B747Lateral`,
forced-dense kernels) do not reach `useDenseSweep` or keep the same path,
and vary run to run on the shared runner.

Already upper-Hessenberg A stays dense on both backends. On amd64 the sweep
is 6% faster for n=32/m=1/w=200 (ratio 0.94), 1.0-2.1x slower elsewhere; not
changed.
