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

No threshold was changed: the short-grid effect is costly only in relative
terms (tens to hundreds of microseconds), and amd64 evidence is missing.
Follow-up: ergo task W4RUT4.

## linux/amd64

Pending: run the `Performance comparison` workflow (`crossover` job,
`ubuntu-latest`) and retain its `crossover-amd64-*` artifact here. A
GitHub-hosted runner is a shared VM; record its CPU model from
`environment.txt`.
