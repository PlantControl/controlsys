package main

import (
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strings"
)

// Tier names a level of checking.
type Tier string

const (
	TierDocs    Tier = "docs"
	TierTests   Tier = "tests"
	TierLibrary Tier = "library"
	TierFull    Tier = "full"
)

const (
	// MaxFraction is the share of tests above which a full run is simpler.
	MaxFraction = 0.5
	// MaxRegexLen stays well under Linux's 128 KiB limit on one argument.
	MaxRegexLen = 64 << 10
	noneRegex   = "^$"
)

// Plan is the selector's output.
type Plan struct {
	Tier       Tier     `json:"tier"`
	Reasons    []string `json:"reasons,omitempty"`
	Run        string   `json:"run"`
	Bench      string   `json:"bench"`
	Tests      []string `json:"tests,omitempty"`
	Benches    []string `json:"benches,omitempty"`
	TotalTests int      `json:"total_tests"`
	Changed    []string `json:"changed,omitempty"`
}

func fullPlan(reasons ...string) Plan {
	sort.Strings(reasons)
	return Plan{Tier: TierFull, Reasons: reasons, Run: "", Bench: "."}
}

// Options configures Select.
type Options struct {
	Configs   []Config
	BaseCache string
	HeadCache string
}

// Select compares two checkouts. It never fails: errors give a full plan.
func Select(baseDir, headDir string, opt Options) (p Plan) {
	defer func() {
		if r := recover(); r != nil {
			p = fullPlan(fmt.Sprintf("selector panic: %v", r))
		}
	}()
	if opt.Configs == nil {
		opt.Configs = DefaultConfigs
	}
	base, err := readTree(baseDir)
	if err != nil {
		return fullPlan("base: " + err.Error())
	}
	head, err := readTree(headDir)
	if err != nil {
		return fullPlan("head: " + err.Error())
	}
	fc := compareFiles(base, head)
	dc := compareDecls(base, head)
	reasons := append(append([]string(nil), fc.reasons...), dc.reasons...)
	if len(reasons) > 0 {
		return fullPlan(reasons...)
	}
	if !fc.any {
		return finishPlan(TierDocs, nil, nil, head)
	}
	if hasTestMain(base) || hasTestMain(head) {
		return fullPlan("TestMain present")
	}
	if !fc.goDiff {
		return finishPlan(TierDocs, nil, nil, head)
	}

	var graphs []*Graph
	for _, cfg := range opt.Configs {
		bg, err := LoadGraph(baseDir, cfg, cacheName(opt.BaseCache, cfg))
		if err != nil {
			return fullPlan("base graph " + cfg.Name + ": " + err.Error())
		}
		hg, err := LoadGraph(headDir, cfg, cacheName(opt.HeadCache, cfg))
		if err != nil {
			return fullPlan("head graph " + cfg.Name + ": " + err.Error())
		}
		graphs = append(graphs, bg, hg)
		if u := append(append([]string(nil), bg.Uncertain...), hg.Uncertain...); len(u) > 0 {
			return fullPlan(u...)
		}
	}

	declFile := func(k string) *decl {
		if d := head.decls[k]; d != nil {
			return d
		}
		return base.decls[k]
	}
	libKeys, testKeys, rootKeys := dc.lib, dc.test, dc.roots
	for i := 0; i+1 < len(graphs); i += 2 {
		for _, ck := range changedConsts(graphs[i], graphs[i+1]) {
			for _, g := range graphs[i : i+2] {
				if g.ConstOutside[ck] {
					return fullPlan("constant used outside a function: " + ck)
				}
				for _, u := range g.ConstUsers[ck] {
					d := declFile(u)
					switch {
					case d == nil:
						return fullPlan("constant user without declaration: " + u)
					case d.root != "":
						rootKeys = append(rootKeys, u)
					case d.isTest:
						testKeys = append(testKeys, u)
					default:
						libKeys = append(libKeys, u)
					}
				}
			}
		}
	}
	libKeys, testKeys, rootKeys = uniq(libKeys), uniq(testKeys), uniq(rootKeys)

	for _, g := range graphs {
		for _, w := range g.TestGlobalWriters {
			for _, set := range [][]string{testKeys, rootKeys} {
				for _, k := range set {
					if k == w {
						return fullPlan("changed test writes a package variable: " + k)
					}
				}
			}
		}
	}

	selected := map[string]Root{}
	collect := func(k string, sharedCheck bool) string {
		known := false
		for _, g := range graphs {
			roots, ini, ok := g.RootsReaching(k)
			if !ok {
				continue
			}
			known = true
			if ini {
				return "package initialization reaches " + k
			}
			for _, r := range roots {
				if sharedCheck {
					if d := declFile(k); d != nil && r.File != d.file {
						return "shared test helper: " + k + " (used by " + r.File + ")"
					}
				}
				selected[r.Name] = r
			}
		}
		if !known {
			return "changed function missing from call graph: " + k
		}
		return ""
	}
	for _, k := range libKeys {
		if why := collect(k, false); why != "" {
			return fullPlan(why)
		}
	}
	for _, k := range testKeys {
		if why := collect(k, true); why != "" {
			return fullPlan(why)
		}
	}
	for _, k := range rootKeys {
		if why := collect(k, false); why != "" {
			return fullPlan(why)
		}
	}
	tier := TierDocs
	if len(testKeys)+len(rootKeys) > 0 {
		tier = TierTests
	}
	if len(libKeys) > 0 {
		tier = TierLibrary
	}
	changed := uniq(append(append(append([]string(nil), libKeys...), testKeys...), rootKeys...))
	return finishPlan(tier, selected, changed, head)
}

func hasTestMain(t *tree) bool {
	return t.decls["TestMain"] != nil || t.decls["_test:TestMain"] != nil
}

func cacheName(prefix string, cfg Config) string {
	if prefix == "" {
		return ""
	}
	return prefix + "-" + cfg.Name + ".gob.gz"
}

func changedConsts(a, b *Graph) []string {
	var out []string
	for k, v := range a.Consts {
		if b.Consts[k] != v {
			out = append(out, k)
		}
	}
	for k := range b.Consts {
		if _, ok := a.Consts[k]; !ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func uniq(in []string) []string {
	sort.Strings(in)
	out := in[:0]
	for i, s := range in {
		if i == 0 || s != in[i-1] {
			out = append(out, s)
		}
	}
	return out
}

// finishPlan adds source readers, drops names head no longer has, applies caps.
func finishPlan(tier Tier, selected map[string]Root, changed []string, head *tree) Plan {
	exists := map[string]Root{}
	maps.Copy(exists, head.roots)
	total := 0
	for _, r := range exists {
		if r.Kind == KindTest {
			total++
		}
	}
	var tests, benches []string
	add := func(n string) {
		r, ok := exists[n]
		if !ok {
			return
		}
		if r.Kind == KindBench {
			benches = append(benches, n)
		} else {
			tests = append(tests, n)
		}
	}
	for n := range selected {
		add(n)
	}
	for n := range head.readers {
		if _, dup := selected[n]; !dup {
			add(n)
		}
	}
	sort.Strings(tests)
	sort.Strings(benches)
	if total > 0 && float64(len(tests)) > MaxFraction*float64(total) {
		return withTotal(fullPlan(fmt.Sprintf("selected %d of %d tests exceeds cap", len(tests), total)), total)
	}
	run, err := buildRegex(tests)
	if err != nil {
		return withTotal(fullPlan(err.Error()), total)
	}
	bench, err := buildRegex(benches)
	if err != nil {
		return withTotal(fullPlan(err.Error()), total)
	}
	return Plan{Tier: tier, Run: run, Bench: bench, Tests: tests, Benches: benches, TotalTests: total, Changed: changed}
}

func withTotal(p Plan, total int) Plan {
	p.TotalTests = total
	return p
}

// buildRegex anchors names for go test; a bare alternation would also match
// longer names sharing a prefix.
func buildRegex(names []string) (string, error) {
	if len(names) == 0 {
		return noneRegex, nil
	}
	s := "^(?:" + strings.Join(names, "|") + ")$"
	if len(s) > MaxRegexLen {
		return "", fmt.Errorf("regexp of %d bytes exceeds cap", len(s))
	}
	re, err := regexp.Compile(s)
	if err != nil {
		return "", fmt.Errorf("regexp: %w", err)
	}
	for _, n := range names {
		if !re.MatchString(n) || re.MatchString(n+"X") || re.MatchString("X"+n) {
			return "", fmt.Errorf("regexp does not isolate %s", n)
		}
		if strings.ContainsAny(n, "/#") {
			return "", fmt.Errorf("root name %q would split in go test", n)
		}
	}
	return s, nil
}

// Commands returns the checks for a plan.
func (p Plan) Commands() []string {
	lint := []string{`test -z "$(gofmt -l .)"`, "go vet ./..."}
	q := func(s string) string { return "'" + s + "'" }
	switch p.Tier {
	case TierDocs:
		c := append(lint, "go build ./...")
		if p.Run != noneRegex {
			c = append(c, "go test -run "+q(p.Run)+" .")
		}
		return c
	case TierTests:
		c := append(lint, "go test -run "+q(p.Run)+" .")
		if p.Bench != noneRegex {
			c = append(c, "go test -run '^$' -bench "+q(p.Bench)+" -benchtime=1x .")
		}
		return c
	case TierLibrary:
		c := append(lint,
			"go test -run "+q(p.Run)+" .",
			"GOARCH=amd64 go test -run "+q(p.Run)+" .",
			"go test -race -run "+q(p.Run)+" .")
		if p.Bench != noneRegex {
			c = append(c, "go test -run '^$' -bench "+q(p.Bench)+" -benchtime=1x -benchmem .")
		}
		return c
	}
	return append(lint,
		"go fix ./... && git diff --exit-code",
		"go test -count=1 -timeout 0 ./...",
		"GOARCH=amd64 go test -count=1 -timeout 0 ./...",
		"go test -race -count=1 -timeout 0 ./...",
		"go test -run '^$' -bench . -benchtime=1x -benchmem ./...",
		"GOARCH=amd64 go test -run '^$' -bench . -benchtime=1x -benchmem ./...",
		"go -C scripts/selecttests vet ./...",
		"go -C scripts/selecttests test ./...")
}
