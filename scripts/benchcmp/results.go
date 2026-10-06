package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// caseSet maps each completed benchmark case to its sorted metric units.
type caseSet struct {
	units   map[string]string
	skipped []string
}

func (s caseSet) equal(o caseSet) bool {
	return maps.Equal(s.units, o.units) && slices.Equal(s.skipped, o.skipped)
}

var resultLine = regexp.MustCompile(`^(Benchmark\S+)\s+(\d+)\s+(.+)$`)

// parseResults validates one benchmark process's output. Any failure, panic,
// malformed or non-finite result, missing PASS trailer or empty selection is
// an error.
func parseResults(out string) (caseSet, error) {
	s := caseSet{units: map[string]string{}}
	passed := false
	sc := bufio.NewScanner(strings.NewReader(out))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		line := sc.Text()
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "--- FAIL"), strings.HasPrefix(line, "panic:"),
			line == "FAIL", strings.HasPrefix(line, "FAIL\t"):
			return caseSet{}, fmt.Errorf("benchmark failed: %s", trimmed)
		case strings.HasPrefix(trimmed, "--- SKIP: "):
			s.skipped = append(s.skipped, strings.Fields(strings.TrimPrefix(trimmed, "--- SKIP: "))[0])
		case line == "PASS":
			passed = true
		}
		m := resultLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		fields := strings.Fields(m[3])
		if n, _ := strconv.Atoi(m[2]); n < 1 || len(fields)%2 != 0 {
			return caseSet{}, fmt.Errorf("malformed result: %s", line)
		}
		var units []string
		for i := 0; i < len(fields); i += 2 {
			v, err := strconv.ParseFloat(fields[i], 64)
			if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				return caseSet{}, fmt.Errorf("invalid metric %q: %s", fields[i], line)
			}
			units = append(units, fields[i+1])
		}
		if _, dup := s.units[m[1]]; dup {
			return caseSet{}, fmt.Errorf("duplicate case %s", m[1])
		}
		slices.Sort(units)
		s.units[m[1]] = strings.Join(units, " ")
	}
	if err := sc.Err(); err != nil {
		return caseSet{}, err
	}
	if !passed {
		return caseSet{}, fmt.Errorf("missing PASS trailer")
	}
	if len(s.units) == 0 {
		return caseSet{}, fmt.Errorf("no benchmark results; check -bench")
	}
	slices.Sort(s.skipped)
	return s, nil
}

type caseReport struct {
	Common           int                 `json:"common"`
	BaselineOnly     []string            `json:"baseline_only,omitempty"`
	CandidateOnly    []string            `json:"candidate_only,omitempty"`
	IncompatibleUnit []string            `json:"incompatible_units,omitempty"`
	Skipped          map[string][]string `json:"skipped,omitempty"`
}

func compareSets(base, cand caseSet) caseReport {
	var r caseReport
	for name, u := range base.units {
		cu, ok := cand.units[name]
		switch {
		case !ok:
			r.BaselineOnly = append(r.BaselineOnly, name)
		case cu != u:
			r.IncompatibleUnit = append(r.IncompatibleUnit, fmt.Sprintf("%s (baseline %s; candidate %s)", name, u, cu))
		default:
			r.Common++
		}
	}
	for name := range cand.units {
		if _, ok := base.units[name]; !ok {
			r.CandidateOnly = append(r.CandidateOnly, name)
		}
	}
	for label, s := range map[string]caseSet{baseLabel: base, candLabel: cand} {
		if len(s.skipped) > 0 {
			if r.Skipped == nil {
				r.Skipped = map[string][]string{}
			}
			r.Skipped[label] = s.skipped
		}
	}
	slices.Sort(r.BaselineOnly)
	slices.Sort(r.CandidateOnly)
	slices.Sort(r.IncompatibleUnit)
	return r
}

// change is one benchmark metric present in both columns of benchstat -format
// csv output.
type change struct {
	Name        string  `json:"name"`
	Unit        string  `json:"unit"`
	Baseline    float64 `json:"baseline"`
	Candidate   float64 `json:"candidate"`
	Delta       float64 `json:"delta_percent"`
	Significant bool    `json:"significant"`
	P           string  `json:"p"`
}

func parseBenchstatCSV(text string) ([]change, error) {
	var out []change
	unit := ""
	for line := range strings.Lines(text) {
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			unit = ""
			continue
		}
		rec, err := csv.NewReader(strings.NewReader(line)).Read()
		if err != nil {
			return nil, fmt.Errorf("%q: %w", line, err)
		}
		if len(rec) >= 7 && rec[0] == "" && rec[5] == "vs base" {
			unit = rec[1]
			continue
		}
		if unit == "" || len(rec) < 7 || rec[0] == "geomean" || rec[0] == "" {
			continue
		}
		base, err1 := strconv.ParseFloat(rec[1], 64)
		cand, err2 := strconv.ParseFloat(rec[3], 64)
		if err1 != nil || err2 != nil {
			continue
		}
		c := change{Name: rec[0], Unit: unit, Baseline: base, Candidate: cand, P: rec[6]}
		if d, ok := strings.CutSuffix(rec[5], "%"); ok {
			if c.Delta, err = strconv.ParseFloat(d, 64); err != nil {
				return nil, fmt.Errorf("delta %q: %w", rec[5], err)
			}
			c.Significant = true
		}
		out = append(out, c)
	}
	return out, nil
}

// reviewItems returns significant regressions that need human review: time
// or bytes up by at least threshold percent, or any allocation increase.
func reviewItems(changes []change, threshold float64) []change {
	var out []change
	for _, c := range changes {
		if !c.Significant || c.Delta <= 0 {
			continue
		}
		switch c.Unit {
		case "sec/op", "B/op":
			if c.Delta >= threshold {
				out = append(out, c)
			}
		case "allocs/op":
			out = append(out, c)
		}
	}
	return out
}

func summary(md *metadata, changes []change, threshold float64) string {
	var b strings.Builder
	b.WriteString("# Benchmark comparison\n\nStatus: complete. Advisory evidence, not a pass/fail performance gate.\n\n")
	header(&b, md)
	fmt.Fprintf(&b, "\n## Review items\n\nSignificant (benchstat, alpha 0.05) time or B/op increases of at least %g%%, and any allocs/op increase.\n\n", threshold)
	changeTable(&b, md.Regressions)
	var other []change
	inconclusive := 0
	for _, c := range changes {
		switch {
		case !c.Significant:
			inconclusive++
		case !slices.Contains(md.Regressions, c):
			other = append(other, c)
		}
	}
	b.WriteString("\n## Other significant changes\n\nImprovements and smaller regressions; report both.\n\n")
	changeTable(&b, other)
	fmt.Fprintf(&b, "\n## Inconclusive\n\n%d case metrics show no significant difference. This is not evidence of equivalence.\n", inconclusive)
	b.WriteString("\n## Missing or incompatible cases\n\n")
	r := md.Cases
	fmt.Fprintf(&b, "Compared on both revisions: %d cases.\n\n", r.Common)
	list(&b, "Baseline only (removed or renamed)", r.BaselineOnly)
	list(&b, "Candidate only (new or renamed, no baseline)", r.CandidateOnly)
	list(&b, "Incompatible metric units", r.IncompatibleUnit)
	for _, label := range []string{baseLabel, candLabel} {
		list(&b, "Skipped on "+label, r.Skipped[label])
	}
	b.WriteString("\nFull table: benchstat.txt. Raw samples: baseline.txt, candidate.txt, raw/. Metadata: metadata.json.\n")
	return b.String()
}

func failedSummary(md *metadata) string {
	var b strings.Builder
	b.WriteString("# Benchmark comparison\n\nStatus: **FAILED**. No comparison was produced.\n\n")
	fmt.Fprintf(&b, "Error: `%s`\n\n", md.Error)
	header(&b, md)
	return b.String()
}

func header(b *strings.Builder, md *metadata) {
	b.WriteString("| | Baseline | Candidate |\n| --- | --- | --- |\n")
	base, cand := md.Revisions[baseLabel], md.Revisions[candLabel]
	fmt.Fprintf(b, "| Ref | `%s` | `%s` |\n| Commit | `%s` | `%s` |\n", base.Ref, cand.Ref, base.Commit, cand.Commit)
	if base.Dirty || cand.Dirty {
		fmt.Fprintf(b, "| Uncommitted changes | %t | %t (diff sha256 `%s`) |\n", base.Dirty, cand.Dirty, cand.DiffSHA256)
	}
	bb, cb := md.Binaries[baseLabel], md.Binaries[candLabel]
	if bb != nil && cb != nil {
		fmt.Fprintf(b, "| Backend | %s | %s |\n", bb.Backend, cb.Backend)
	}
	fmt.Fprintf(b, "\n- Go: %s; GOAMD64=%s GOARM64=%s CGO_ENABLED=%s GOFLAGS=%q GOEXPERIMENT=%q\n",
		md.GoVersion, md.GoEnv["GOAMD64"], md.GoEnv["GOARM64"], md.GoEnv["CGO_ENABLED"], md.GoEnv["GOFLAGS"], md.GoEnv["GOEXPERIMENT"])
	fmt.Fprintf(b, "- Host: %s/%s, %s, %s CPUs, GOMAXPROCS env %q\n", md.Host["os"], md.Host["arch"], md.Host["cpu"], md.Host["num_cpu"], md.RuntimeEnv["GOMAXPROCS"])
	if img := md.Host["ImageOS"]; img != "" {
		fmt.Fprintf(b, "- Runner: %s %s %s\n", md.Host["RUNNER_NAME"], img, md.Host["ImageVersion"])
	}
	fmt.Fprintf(b, "- Samples: %d interleaved rounds per revision; flags `%s`\n", md.Rounds, strings.Join(md.Flags, " "))
	if bb != nil && cb != nil && bb.Backend != cb.Backend {
		b.WriteString("- **Warning:** baseline and candidate report different backends.\n")
	}
}

func changeTable(b *strings.Builder, cs []change) {
	if len(cs) == 0 {
		b.WriteString("None.\n")
		return
	}
	b.WriteString("| Benchmark | Unit | Baseline | Candidate | Change | p |\n| --- | --- | ---: | ---: | ---: | --- |\n")
	for _, c := range cs {
		fmt.Fprintf(b, "| %s | %s | %.4g | %.4g | %+.2f%% | %s |\n", c.Name, c.Unit, c.Baseline, c.Candidate, c.Delta, c.P)
	}
}

func list(b *strings.Builder, title string, items []string) {
	if len(items) == 0 {
		return
	}
	fmt.Fprintf(b, "%s (%d):\n\n", title, len(items))
	for _, it := range items {
		fmt.Fprintf(b, "- %s\n", it)
	}
	b.WriteString("\n")
}
