#!/usr/bin/env python3
"""Score dense/Hessenberg dispatch rules on BenchmarkFrequencyDispatch output.

Usage: evalrule.py [--fit] [--rule S0 S1 KAPPA RHO0] RAW...

RAW is raw.txt or raw.txt.gz from scripts/crossover.sh. For each file, prints
the worst loss of each rule against the faster forced kernel and against the
main rule (w <= 2 or n <= c), over coupled A (a=full; already-Hessenberg A is
always dense) for the cost rule used by useDenseSweep,
"dense iff (S0 + S1/n)/w + KAPPA*c/n + RHO0 >= 1", c = 8m+8. --fit
least-squares fits that model to the Hessenberg/dense median ratios (w >= 3,
ratio in [0.5, 2]) of all RAW files together; needs numpy.
"""

import argparse
import collections
import gzip
import re
import statistics

LINE = re.compile(r"BenchmarkFrequencyDispatch/n=(\d+)/m=(\d+)/w=(\d+)/a=(\w+)/path=(\w+)\S*\s+\d+\s+([\d.]+) ns/op")


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    samples = collections.defaultdict(list)
    with opener(path, "rt") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                n, mm, w, a, p, t = m.groups()
                samples[(int(n), int(mm), int(w), a, p)].append(float(t))
    cases = {}
    for (n, m, w, a, p), v in samples.items():
        cases.setdefault((n, m, w, a), {})[p] = statistics.median(v)
    return {k: v for k, v in cases.items() if k[3] == "full"}


def main_rule(n, m, w):
    return w <= 2 or n <= 8 * m + 8


def pr325_rule(n, m, w):
    c = 8 * m + 8
    return n <= c or w <= (6 * n - 1) // (n - c)


def cost_rule(s0, s1, kappa, rho0):
    return lambda n, m, w: w * (n * (1 - rho0) - kappa * (8 * m + 8)) <= s0 * n + s1


def fit(datasets):
    import numpy as np

    X, y = [], []
    for cases in datasets:
        for (n, m, w, _), t in cases.items():
            r = t["hessenberg"] / t["dense"]
            if w >= 3 and 0.5 <= r <= 2:
                X.append([1 / w, 1 / (n * w), (8 * m + 8) / n, 1])
                y.append(r)
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return [float(c) for c in coef]


def score(cases, rule):
    worst_best, worst_main = (0.0, ()), (0.0, ())
    for (n, m, w, _), t in cases.items():
        pick = t["dense"] if rule(n, m, w) else t["hessenberg"]
        ref = t["dense"] if main_rule(n, m, w) else t["hessenberg"]
        worst_best = max(worst_best, (pick / min(t["dense"], t["hessenberg"]) - 1, (n, m, w)))
        worst_main = max(worst_main, (pick / ref - 1, (n, m, w)))
    return worst_best, worst_main


def fmt(s):
    (b, kb), (m, km) = s
    return "vs best %+.1f%% %s, vs main %+.1f%% %s" % (100 * b, kb, 100 * m, km)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--rule", nargs=4, type=float, action="append", default=[], metavar=("S0", "S1", "KAPPA", "RHO0"))
    ap.add_argument("raw", nargs="+")
    args = ap.parse_args()
    datasets = [load(path) for path in args.raw]
    rules = [tuple(r) for r in args.rule]
    if args.fit:
        coef = fit(datasets)
        print("fit S0 S1 KAPPA RHO0 =", " ".join("%.3f" % c for c in coef))
        rules.append(tuple(coef))
    for path, cases in zip(args.raw, datasets):
        print("==", path, len(cases), "cases")
        print("  main ", fmt(score(cases, main_rule)))
        print("  pr325", fmt(score(cases, pr325_rule)))
        for r in rules:
            print("  cost", "(%g, %g, %g, %g)" % r, fmt(score(cases, cost_rule(*r))))


if __name__ == "__main__":
    main()
