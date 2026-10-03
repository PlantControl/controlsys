# Thiran bank composition

Baseline: v1.11.0 (df86d39). Candidate: direct bank composition plus
nondecomposable path-delay support. Go 1.27.1, darwin/arm64, Apple M1 Pro,
default GOMAXPROCS. Twelve interleaved runs of prebuilt test binaries:

    ./base.test -test.run '^$' -test.bench '^Benchmark(Discretize|ThiranDelay|D2C|D2D|Undiscretize)' -test.benchmem -test.benchtime 200ms
    ./new.test  (same flags)

`benchstat.txt` holds the comparison. `DiscretizeWithOpts_PathThiran` has no
baseline because v1.11.0 rejects nondecomposable fractional path delays.
