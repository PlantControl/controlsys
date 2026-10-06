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

## Grid-length-aware rule (darwin/arm64, Apple M1 Pro)

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

### amd64

Not measured: no amd64 host is available locally, and the `crossover` job of
the `Performance comparison` workflow has not been run for this rule. The
change only moves cases from Hessenberg to dense, and on arm64 the moved cases
are at most 4.6% slower (dense 2x faster at w=3). If amd64's assembly
kernels shift the per-point ratio towards Hessenberg, losses would be largest
just above c at w of 20-60 where the arm64 margin is only 3-13%; the far
short-grid region (w <= 5) has a 1.5-2.5x arm64 margin. Rerun the
`crossover` job and retain `crossover-amd64-*` here to confirm or tune the
constant.

## linux/amd64

Pending for both the state threshold and the grid-length rule: run the
`Performance comparison` workflow (`crossover` job, `ubuntu-latest`) and
retain its `crossover-amd64-*` artifact here. A
GitHub-hosted runner is a shared VM; record its CPU model from
`environment.txt`.
