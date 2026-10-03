# Tustin+Thiran external delay: main 1bb229f vs working tree

M1 Pro, darwin/arm64, Go go1.27.1, GOMAXPROCS 8, 8 interleaved runs x 200ms.
Bench fixed to Tustin (ZOH+Thiran now rejected per MATLAB; old bench timed that error path).
Main: order-3 filter over full 3.5 samples (5 states). New: MATLAB split N=min(ceil(D),3), filter 2.5 + 1 integer sample, filter memory as internal delays (2 states + LFT).
Cost is generic Series/Copy LFT composition (~47% allocs); tracked separately.

```
goos: darwin
goarch: arm64
pkg: plantcontrol.org/v1/controlsys
cpu: Apple M1 Pro
                                   │ old2.txt │ new2.txt │
                                   │                                                        sec/op                                                         │                                             sec/op                                              vs base               │
DiscretizeWithOpts_Thiran-8                                                                                                                    2.519µ ± 2%                                                                                      4.086µ ± 1%  +62.24% (p=0.000 n=8)
DiscretizeWithOpts_IODelayThiran-8                                                                                                             2.607µ ± 1%                                                                                      4.158µ ± 2%  +59.44% (p=0.000 n=8)
geomean                                                                                                                                        2.563µ                                                                                           4.122µ       +60.84%

                                   │ old2.txt │ new2.txt │
                                   │                                                         B/op                                                          │                                             B/op                                               vs base                │
DiscretizeWithOpts_Thiran-8                                                                                                                   3.460Ki ± 0%                                                                                    8.188Ki ± 0%  +136.66% (p=0.000 n=8)
DiscretizeWithOpts_IODelayThiran-8                                                                                                            3.640Ki ± 0%                                                                                    8.368Ki ± 0%  +129.92% (p=0.000 n=8)
geomean                                                                                                                                       3.549Ki                                                                                         8.278Ki       +133.27%

                                   │ old2.txt │ new2.txt │
                                   │                                                       allocs/op                                                       │                                           allocs/op                                            vs base                │
DiscretizeWithOpts_Thiran-8                                                                                                                     74.00 ± 0%                                                                                     160.00 ± 0%  +116.22% (p=0.000 n=8)
DiscretizeWithOpts_IODelayThiran-8                                                                                                              83.00 ± 0%                                                                                     169.00 ± 0%  +103.61% (p=0.000 n=8)
geomean                                                                                                                                         78.37                                                                                           164.4       +109.82%
```

## No-delay conversion paths

```
goos: darwin
goarch: arm64
pkg: plantcontrol.org/v1/controlsys
cpu: Apple M1 Pro
                                         │ old.txt │ new.txt │
                                         │                                                        sec/op                                                        │                                            sec/op                                              vs base               │
BilinearDiscretize-8                                                                                                                                10.33µ ± 0%                                                                                     10.43µ ± 1%   +0.99% (p=0.038 n=8)
DiscretizeZOH-8                                                                                                                                     21.17µ ± 0%                                                                                     21.45µ ± 1%   +1.28% (p=0.014 n=8)
DiscretizeAndSimulate_B747Longitudinal-8                                                                                                            13.11µ ± 0%                                                                                     13.07µ ± 1%        ~ (p=0.645 n=8)
D2C_ZOH_N2-8                                                                                                                                        2.853µ ± 1%                                                                                     2.865µ ± 2%        ~ (p=0.068 n=8)
D2C_ZOH_N5-8                                                                                                                                        6.353µ ± 0%                                                                                     6.394µ ± 1%   +0.65% (p=0.038 n=8)
D2C_ZOH_N20-8                                                                                                                                       60.41µ ± 2%                                                                                     60.91µ ± 1%        ~ (p=0.442 n=8)
D2C_ZOH_N50-8                                                                                                                                       411.6µ ± 0%                                                                                     414.2µ ± 1%        ~ (p=0.083 n=8)
D2C_Tustin_N20-8                                                                                                                                    13.69µ ± 1%                                                                                     13.74µ ± 1%        ~ (p=0.130 n=8)
geomean                                                                                                                                             13.36µ                                                                                          6.808µ       -49.04%

                                         │ old.txt │ new.txt │
                                         │                                                         B/op                                                         │                                            B/op                                              vs base                 │
BilinearDiscretize-8                                                                                                                               14.59Ki ± 0%                                                                                  14.58Ki ± 0%        ~ (p=0.505 n=8)
DiscretizeZOH-8                                                                                                                                    21.66Ki ± 0%                                                                                  21.66Ki ± 0%        ~ (p=0.818 n=8)
DiscretizeAndSimulate_B747Longitudinal-8                                                                                                           6.578Ki ± 0%                                                                                  6.578Ki ± 0%        ~ (p=1.000 n=8) ¹
D2C_ZOH_N2-8                                                                                                                                       2.170Ki ± 0%                                                                                  2.109Ki ± 0%   -2.81% (p=0.000 n=8)
D2C_ZOH_N5-8                                                                                                                                       6.622Ki ± 0%                                                                                  6.389Ki ± 0%   -3.52% (p=0.000 n=8)
D2C_ZOH_N20-8                                                                                                                                      77.32Ki ± 0%                                                                                  74.14Ki ± 0%   -4.12% (p=0.000 n=8)
D2C_ZOH_N50-8                                                                                                                                      462.6Ki ± 0%                                                                                  446.0Ki ± 0%   -3.59% (p=0.000 n=8)
D2C_Tustin_N20-8                                                                                                                                   14.84Ki ± 0%                                                                                  14.84Ki ± 0%        ~ (p=0.803 n=8)
geomean                                                                                                                                            13.59Ki                                                                                       6.631Ki       -51.20%
¹ all samples are equal

                                         │ old.txt │ new.txt │
                                         │                                                      allocs/op                                                       │                                          allocs/op                                           vs base                 │
BilinearDiscretize-8                                                                                                                                 32.00 ± 0%                                                                                    32.00 ± 0%        ~ (p=1.000 n=8) ¹
DiscretizeZOH-8                                                                                                                                      28.00 ± 0%                                                                                    28.00 ± 0%        ~ (p=1.000 n=8) ¹
DiscretizeAndSimulate_B747Longitudinal-8                                                                                                             11.00 ± 0%                                                                                    11.00 ± 0%        ~ (p=1.000 n=8) ¹
D2C_ZOH_N2-8                                                                                                                                         46.00 ± 0%                                                                                    46.00 ± 0%        ~ (p=1.000 n=8) ¹
D2C_ZOH_N5-8                                                                                                                                         48.00 ± 0%                                                                                    48.00 ± 0%        ~ (p=1.000 n=8) ¹
D2C_ZOH_N20-8                                                                                                                                        50.00 ± 0%                                                                                    50.00 ± 0%        ~ (p=1.000 n=8) ¹
D2C_ZOH_N50-8                                                                                                                                        53.00 ± 0%                                                                                    53.00 ± 0%        ~ (p=1.000 n=8) ¹
D2C_Tustin_N20-8                                                                                                                                     32.00 ± 0%                                                                                    32.00 ± 0%        ~ (p=1.000 n=8) ¹
geomean                                                                                                                                              40.10                                                                                         19.36       -51.74%
¹ all samples are equal
```
