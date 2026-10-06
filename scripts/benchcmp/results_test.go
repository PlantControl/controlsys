package main

import (
	"slices"
	"strings"
	"testing"
)

const okOutput = `goos: darwin
goarch: arm64
pkg: plantcontrol.org/v2/controlsys/v2
BenchmarkA-8   	  1000	      1200 ns/op	     64 B/op	       2 allocs/op
BenchmarkB/n=4-8   	  500	      2400 ns/op	    128 B/op	       3 allocs/op
--- SKIP: BenchmarkC
    bench_test.go:10: needs data
PASS
`

func TestParseResults(t *testing.T) {
	s, err := parseResults(okOutput)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.units["BenchmarkB/n=4-8"]; got != "B/op allocs/op ns/op" {
		t.Fatalf("units = %q", got)
	}
	if len(s.units) != 2 || !slices.Equal(s.skipped, []string{"BenchmarkC"}) {
		t.Fatalf("parsed %v skipped %v", s.units, s.skipped)
	}
}

func TestParseResultsRejectsUnsuccessfulRuns(t *testing.T) {
	tests := map[string]string{
		"fail":       strings.Replace(okOutput, "PASS\n", "--- FAIL: BenchmarkD\n    x_test.go:3: boom\nFAIL\n", 1),
		"subfail":    strings.Replace(okOutput, "PASS\n", "    --- FAIL: BenchmarkD/n=2\nPASS\n", 1),
		"panic":      okOutput + "panic: runtime error\n",
		"noPass":     strings.Replace(okOutput, "PASS\n", "", 1),
		"empty":      "PASS\n",
		"zeroN":      "BenchmarkA-8 0 1 ns/op\nPASS\n",
		"nonFinite":  "BenchmarkA-8 1 NaN ns/op\nPASS\n",
		"oddFields":  "BenchmarkA-8 1 1 ns/op 3\nPASS\n",
		"duplicated": "BenchmarkA-8 1 1 ns/op\nBenchmarkA-8 1 1 ns/op\nPASS\n",
	}
	for name, out := range tests {
		if _, err := parseResults(out); err == nil {
			t.Errorf("%s: parseResults succeeded", name)
		}
	}
}

func TestCompareSetsReportsMissingAndIncompatible(t *testing.T) {
	base := caseSet{units: map[string]string{"BenchmarkA": "ns/op", "BenchmarkOld": "ns/op", "BenchmarkU": "ns/op"}}
	cand := caseSet{units: map[string]string{"BenchmarkA": "ns/op", "BenchmarkNew": "ns/op", "BenchmarkU": "B/op ns/op"}, skipped: []string{"BenchmarkS"}}
	r := compareSets(base, cand)
	if r.Common != 1 || !slices.Equal(r.BaselineOnly, []string{"BenchmarkOld"}) || !slices.Equal(r.CandidateOnly, []string{"BenchmarkNew"}) ||
		len(r.IncompatibleUnit) != 1 || !slices.Equal(r.Skipped[candLabel], []string{"BenchmarkS"}) {
		t.Fatalf("report = %+v", r)
	}
	if base.equal(cand) || !base.equal(base) {
		t.Fatal("caseSet.equal")
	}
}

const statCSV = `,baseline,,candidate,,,
,sec/op,CI,sec/op,CI,vs base,P
Fast-8,2e-05,2%,1e-05,1%,-50.00%,p=0.000 n=10
Slow-8,1e-05,2%,1.1e-05,1%,+10.00%,p=0.000 n=10
Small-8,1e-05,2%,1.02e-05,1%,+2.00%,p=0.001 n=10
Same-8,1e-05,2%,1e-05,1%,~,p=0.630 n=10
New-8,,,5e-06,1%
geomean,1e-05,,1e-05,,-1.00%,

,baseline,,candidate,,,
,allocs/op,CI,allocs/op,CI,vs base,P
Slow-8,2,0%,3,0%,+50.00%,p=0.000 n=10
Same-8,2,0%,2,0%,~,p=1.000 n=10
`

func TestParseBenchstatCSVAndReview(t *testing.T) {
	cs, err := parseBenchstatCSV(statCSV)
	if err != nil {
		t.Fatal(err)
	}
	if len(cs) != 6 {
		t.Fatalf("parsed %d changes: %+v", len(cs), cs)
	}
	if c := cs[1]; c.Name != "Slow-8" || c.Unit != "sec/op" || c.Delta != 10 || !c.Significant {
		t.Fatalf("Slow-8 = %+v", c)
	}
	if cs[3].Significant {
		t.Fatal("~ parsed as significant")
	}
	var got []string
	for _, c := range reviewItems(cs, 5) {
		got = append(got, c.Name+" "+c.Unit)
	}
	if want := []string{"Slow-8 sec/op", "Slow-8 allocs/op"}; !slices.Equal(got, want) {
		t.Fatalf("review = %v, want %v", got, want)
	}
}

func TestSummaryListsTradeoffs(t *testing.T) {
	cs, _ := parseBenchstatCSV(statCSV)
	md := &metadata{
		Revisions:   map[string]revision{baseLabel: {Ref: "origin/main", Commit: "aaa"}, candLabel: {Ref: "HEAD", Commit: "bbb"}},
		Binaries:    map[string]*binary{baseLabel: {Backend: "x"}, candLabel: {Backend: "y"}},
		Cases:       &caseReport{Common: 5, CandidateOnly: []string{"BenchmarkNew-8"}},
		Regressions: reviewItems(cs, 5),
		Host:        map[string]string{},
		GoEnv:       map[string]string{},
		RuntimeEnv:  map[string]string{},
	}
	s := summary(md, cs, 5)
	for _, want := range []string{"`aaa`", "`bbb`", "| Slow-8 | sec/op", "| Fast-8 | sec/op", "| Small-8 | sec/op", "2 case metrics show no significant difference", "BenchmarkNew-8", "different backends"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary missing %q:\n%s", want, s)
		}
	}
	md.Binaries[baseLabel].Backend = unreported
	if s := summary(md, cs, 5); strings.Contains(s, "different backends") {
		t.Error("warned about an unreported backend")
	}
}
