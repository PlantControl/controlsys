# Negative-pole ZOH reverse conversion

Apple M1 Pro, darwin/arm64, Go 1.27.1, GOMAXPROCS=1. Six interleaved runs,
200ms per case; order alternated each repetition. Identical saved working-tree
snapshots except `d2cZOHRealExtension`: baseline retains the previous full
2n-state extension; candidate selects a branch-safe rotation and compresses
only negative alias states with SVD and verified projector residuals.

| Fixture | States before/after | Time before/after | Bytes before/after | Allocations before/after |
| --- | --- | --- | --- | --- |
| n=2, one negative mode | 4 / 3 | 111.09 / 98.62 us | 90.45 / 80.97 KiB | 893 / 823 |
| n=4, two negative modes | 8 / 6 | 250.3 / 225.7 us | 275.9 / 250.8 KiB | 1066 / 998 |

Both time differences have p=0.002. Results include the changed logarithm
rotation and subspace compression together; they do not isolate SVD cost.
No added work occurs on the ordinary positive-pole fast path. These are rare
branch measurements; common conversion workloads have separate validation.

Command per repetition:

```
GOMAXPROCS=1 ./variant.test -test.run='^$' \
  -test.bench='^BenchmarkNegativeZOHCompressedExtension$' \
  -test.benchmem -test.benchtime=200ms -test.count=1
```

Independent persistent tests compare sampled outputs and mapped nonzero states
with discrete recurrence versus RK4 integration under held inputs. Cases cover
nonsymmetric dense coordinates, positive and negative defective poles, existing
conjugate pairs near the negative branch cut, state count, immutable sources,
large sample times, and explicit unresolved-cut/overflow failures.
