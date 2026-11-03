package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"testing"
)

const fxLib = `package fx

import "reflect"

var _ = reflect.TypeOf

type Shape interface{ Area() float64 }

type Square struct{ S float64 }

func (s Square) Area() float64 { return s.S * s.S }

type Circle struct{ R float64 }

func (c Circle) Area() float64 { return 3 * c.R * c.R }

func Total(xs []Shape) float64 {
	t := 0.0
	for _, x := range xs {
		t += x.Area()
	}
	return t
}

func Apply(f func(int) int, x int) int { return f(x) }

func Inc(x int) int { return x + 1 }

func UseInc(x int) int { return Apply(Inc, x) }

func Double(x int) int { return x * 2 }

func Twice(x int) int { return Double(x) }

func Gen[T any](x T) T { return genHelper(x) }

func genHelper[T any](x T) T { return x }

const Limit = 10

func UnderLimit(x int) bool { return x < Limit }

func Closure() func() int { return func() int { return inner() } }

func inner() int { return 1 }

var counter int

func Bump() { counter++ }

func Read() int { return counter }

func Deleted() int { return 1 }

func CallsDeleted() int { return Deleted() }

func Seed() int { return 7 }

var seeded = Seed()

func UseSeeded() int { return seeded }

// Comment is documented.
func Comment() int { return 1 }

func Everywhere() int { return 1 }
`

const fxTest = `package fx

import (
	"fmt"
	"testing"
)

func TestTotal(t *testing.T) { Total([]Shape{Square{1}}) }
func TestUseInc(t *testing.T) { UseInc(1) }
func TestTwice(t *testing.T) { Twice(1) }
func TestGen(t *testing.T) { Gen(1) }
func TestLimit(t *testing.T) { UnderLimit(1) }
func TestClosure(t *testing.T) { Closure()() }
func TestRead(t *testing.T) { Read() }
func TestBump(t *testing.T) { Bump() }
func TestCallsDeleted(t *testing.T) { CallsDeleted() }
func TestSub(t *testing.T) { t.Run("x", func(t *testing.T) { Inner2() }) }
func TestComment(t *testing.T) { Comment() }
func BenchmarkTwice(b *testing.B) { Twice(1) }
func ExampleTwice() {
	fmt.Println(Twice(1))
	// Output: 2
}
func FuzzTwice(f *testing.F) { f.Fuzz(func(t *testing.T, x int) { Twice(x) }) }
func Inner2() {}
`

const fxHelpers = `package fx

import "testing"

func sharedHelper() int { return Everywhere2() }
func localHelper() int { return 1 }
func TestA(t *testing.T) { sharedHelper(); localHelper() }
func Everywhere2() int { return 1 }
`

const fxOther = `package fx

import "testing"

func TestB(t *testing.T) { sharedHelper() }
`

func fxFiller() string {
	var b strings.Builder
	b.WriteString("package fx\n\nimport \"testing\"\n\n")
	for i := range 60 {
		fmt.Fprintf(&b, "func TestFill%d(t *testing.T) { }\n", i)
	}
	return b.String()
}

func baseFiles() map[string]string {
	return map[string]string{
		"go.mod":          "module example.org/fx\n\ngo 1.27\n",
		"README.md":       "# fx\n",
		"lib.go":          fxLib,
		"lib_test.go":     fxTest,
		"helper_test.go":  fxHelpers,
		"other_test.go":   fxOther,
		"filler_test.go":  fxFiller(),
		"testdata/d.json": "{}\n",
	}
}

func writeFiles(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func hostOnly() []Config { return []Config{{Name: runtime.GOARCH, GOARCH: runtime.GOARCH}} }

type edit func(map[string]string)

func sub(file, old, new string) edit {
	return func(f map[string]string) {
		if !strings.Contains(f[file], old) {
			panic("fixture lacks " + old)
		}
		f[file] = strings.Replace(f[file], old, new, 1)
	}
}

func put(file, content string) edit { return func(f map[string]string) { f[file] = content } }
func add(file, content string) edit { return func(f map[string]string) { f[file] += content } }

func selectEdits(t *testing.T, cfgs []Config, baseEdits []edit, headEdits []edit) Plan {
	t.Helper()
	b, h := baseFiles(), baseFiles()
	for _, e := range baseEdits {
		e(b)
		e(h)
	}
	for _, e := range headEdits {
		e(h)
	}
	return Select(writeFiles(t, b), writeFiles(t, h), Options{Configs: cfgs})
}

func TestTiers(t *testing.T) {
	for _, tc := range []struct {
		name    string
		edits   []edit
		tier    Tier
		tests   []string
		benches []string
	}{
		{name: "identical", tier: TierDocs},
		{name: "markdown", edits: []edit{sub("README.md", "fx", "gx")}, tier: TierDocs},
		{name: "library comment", edits: []edit{sub("lib.go", "// Comment is documented.", "// Comment is reworded.")}, tier: TierDocs},
		{name: "library gofmt", edits: []edit{sub("lib.go", "func Comment() int { return 1 }", "func Comment() int {\n\treturn 1\n}")}, tier: TierDocs},
		{name: "test comment", edits: []edit{add("other_test.go", "\n// note\n")}, tier: TierDocs},
		{name: "test body", edits: []edit{sub("filler_test.go", "TestFill3(t *testing.T) { }", "TestFill3(t *testing.T) { t.Log(1) }")}, tier: TierTests, tests: []string{"TestFill3"}},
		{name: "new test", edits: []edit{add("other_test.go", "func TestNew(t *testing.T) { Twice(2) }\n")}, tier: TierTests, tests: []string{"TestNew"}},
		{name: "deleted test", edits: []edit{sub("filler_test.go", "func TestFill3(t *testing.T) { }\n", "")}, tier: TierTests},
		{name: "changed benchmark", edits: []edit{sub("lib_test.go", "BenchmarkTwice(b *testing.B) { Twice(1) }", "BenchmarkTwice(b *testing.B) { Twice(2) }")}, tier: TierTests, benches: []string{"BenchmarkTwice"}},
		{name: "example output", edits: []edit{sub("lib_test.go", "// Output: 2", "// Output: 3")}, tier: TierTests, tests: []string{"ExampleTwice"}},
		{name: "local helper", edits: []edit{sub("helper_test.go", "localHelper() int { return 1 }", "localHelper() int { return 2 }")}, tier: TierTests, tests: []string{"TestA"}},
		{name: "subtest body", edits: []edit{sub("lib_test.go", `t.Run("x", func(t *testing.T) { Inner2() })`, `t.Run("x", func(t *testing.T) { Inner2(); Inner2() })`)}, tier: TierTests, tests: []string{"TestSub"}},
		{name: "static call", edits: []edit{sub("lib.go", "x * 2", "x * 3")}, tier: TierLibrary, tests: []string{"ExampleTwice", "FuzzTwice", "TestTwice"}, benches: []string{"BenchmarkTwice"}},
		{name: "interface dispatch cha", edits: []edit{sub("lib.go", "3 * c.R", "4 * c.R")}, tier: TierLibrary, tests: []string{"TestTotal"}},
		{name: "function value", edits: []edit{sub("lib.go", "x + 1", "x + 2")}, tier: TierLibrary, tests: []string{"TestUseInc"}},
		{name: "closure callee", edits: []edit{sub("lib.go", "func inner() int { return 1 }", "func inner() int { return 2 }")}, tier: TierLibrary, tests: []string{"TestClosure"}},
		{name: "closure body", edits: []edit{sub("lib.go", "func() int { return inner() }", "func() int { return inner() + 1 }")}, tier: TierLibrary, tests: []string{"TestClosure"}},
		{name: "generic instance", edits: []edit{sub("lib.go", "func genHelper[T any](x T) T { return x }", "func genHelper[T any](x T) T { y := x; return y }")}, tier: TierLibrary, tests: []string{"TestGen"}},
		{name: "constant by referencing function", edits: []edit{sub("lib.go", "const Limit = 10", "const Limit = 11")}, tier: TierLibrary, tests: []string{"TestLimit"}},
		{name: "global writer reaches reader", edits: []edit{sub("lib.go", "counter++", "counter += 2")}, tier: TierLibrary, tests: []string{"TestBump", "TestRead"}},
		{name: "rename across graphs", edits: []edit{
			sub("lib.go", "func Deleted() int { return 1 }", "func Renamed() int { return 1 }"),
			sub("lib.go", "return Deleted()", "return Renamed()"),
		}, tier: TierLibrary, tests: []string{"TestCallsDeleted"}},
		{name: "new type additive", edits: []edit{add("lib.go", "type Hex struct{}\n\nfunc (Hex) Area() float64 { return 1 }\n")}, tier: TierLibrary, tests: []string{"TestTotal"}},
		{name: "unexported method added", edits: []edit{add("lib.go", "func (Circle) scale() float64 { return 1 }\n")}, tier: TierLibrary},
		{name: "unexported method signature", edits: []edit{
			add("lib.go", "func (c Circle) half() float64 { return c.R / 2 }\n\nfunc UseHalf() float64 { return Circle{}.half() }\n"),
		}, tier: TierLibrary},
		{name: "unreached function", edits: []edit{add("lib.go", "func Dead() int { return 1 }\n")}, tier: TierLibrary},
		{name: "source reader always selected", edits: []edit{
			put("architecture_x_test.go", "package fx\n\nimport \"testing\"\n\nfunc TestArchX(t *testing.T) {}\n"),
		}, tier: TierTests, tests: []string{"TestArchX"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := selectEdits(t, hostOnly(), nil, tc.edits)
			if p.Tier != tc.tier {
				t.Fatalf("tier = %s %v, want %s", p.Tier, p.Reasons, tc.tier)
			}
			if !slices.Equal(p.Tests, tc.tests) {
				t.Errorf("tests = %v, want %v", p.Tests, tc.tests)
			}
			if !slices.Equal(p.Benches, tc.benches) {
				t.Errorf("benches = %v, want %v", p.Benches, tc.benches)
			}
		})
	}
}

func TestFailOpen(t *testing.T) {
	for _, tc := range []struct {
		name   string
		base   []edit
		edits  []edit
		reason string
	}{
		{name: "go.mod", edits: []edit{add("go.mod", "\n// x\n")}, reason: "go.mod"},
		{name: "go.sum", edits: []edit{put("go.sum", "\n")}, reason: "go.sum"},
		{name: "testdata", edits: []edit{sub("testdata/d.json", "{}", "[]")}, reason: "testdata"},
		{name: "workflow", edits: []edit{put(".github/workflows/ci.yml", "x")}, reason: ".github"},
		{name: "selector itself", edits: []edit{put("scripts/selecttests/main.go", "package main")}, reason: "scripts"},
		{name: "subpackage", edits: []edit{put("sub/a.go", "package sub")}, reason: "sub/a.go"},
		{name: "asm", edits: []edit{put("k_amd64.s", "")}, reason: "k_amd64.s"},
		{name: "arch suffix", edits: []edit{put("k_amd64.go", "package fx\n")}, reason: "arch-specific"},
		{name: "arch test suffix", edits: []edit{put("k_arm64_test.go", "package fx\n")}, reason: "arch-specific"},
		{name: "build tag", edits: []edit{put("k.go", "//go:build race\n\npackage fx\n")}, reason: "build-constrained"},
		{name: "build tag comment only", base: []edit{put("k.go", "//go:build race\n\npackage fx\n")}, edits: []edit{add("k.go", "// x\n")}, reason: "build-constrained"},
		{name: "generated", edits: []edit{put("g.go", "// Code generated by x. DO NOT EDIT.\n\npackage fx\n")}, reason: "generated"},
		{name: "directive", edits: []edit{sub("lib.go", "func Comment()", "//go:noinline\nfunc Comment()")}, reason: "directive"},
		{name: "blank import", edits: []edit{sub("lib.go", `import "reflect"`, "import (\n\t\"reflect\"\n\t_ \"embed\"\n)")}, reason: "blank or dot"},
		{name: "var changed", edits: []edit{sub("lib.go", "var counter int", "var counter = 1")}, reason: "package variable"},
		{name: "var added", edits: []edit{add("lib.go", "var extra = 1\n")}, reason: "package variable"},
		{name: "test var added", edits: []edit{add("other_test.go", "var extra = 1\n")}, reason: "package variable"},
		{name: "init added", edits: []edit{add("lib.go", "func init() {}\n")}, reason: "init"},
		{name: "initializer callee", edits: []edit{sub("lib.go", "func Seed() int { return 7 }", "func Seed() int { return 8 }")}, reason: "initialization reaches"},
		{name: "testmain added", edits: []edit{add("other_test.go", "func TestMain(m *testing.M) { m.Run() }\n")}, reason: "TestMain"},
		{name: "testmain present", base: []edit{add("other_test.go", "func TestMain(m *testing.M) { m.Run() }\n")}, edits: []edit{sub("lib.go", "x * 2", "x * 3")}, reason: "TestMain"},
		{name: "type changed", edits: []edit{sub("lib.go", "type Circle struct{ R float64 }", "type Circle struct {\n\tR float64\n\tZ int\n}")}, reason: "type declaration"},
		{name: "type removed", base: []edit{add("lib.go", "type Gone struct{}\n")}, edits: []edit{sub("lib.go", "type Gone struct{}\n", "")}, reason: "type declaration"},
		{name: "method added", edits: []edit{add("lib.go", "func (Circle) Name() string { return \"c\" }\n")}, reason: "method set"},
		{name: "method signature changed", edits: []edit{
			sub("lib.go", "func (c Circle) Area() float64 { return 3 * c.R * c.R }", "func (c *Circle) Area() float64 { return 3 * c.R * c.R }"),
		}, reason: "method signature"},
		{name: "method result changed", base: []edit{add("lib.go", "func (Circle) Name() string { return \"c\" }\n")}, edits: []edit{sub("lib.go", "func (Circle) Name() string { return \"c\" }", "func (Circle) Name() int { return 1 }")}, reason: "method signature"},
		{name: "func value in package state", edits: []edit{add("lib.go", "var hook func() int\n\nfunc SetHook() { hook = inner }\n")}, reason: "package variable"},
		{name: "unexported method named by local interface", base: []edit{add("lib.go", "type scaler interface{ scale() float64 }\n")}, edits: []edit{add("lib.go", "func (Circle) scale() float64 { return 1 }\n")}, reason: "method set"},
		{name: "interface value invoked on", base: []edit{add("lib.go", "type setter interface{ Set(func() int) }\n\nvar sink setter\n\nfunc Hook() { sink.Set(inner) }\n")}, edits: []edit{sub("lib.go", "x * 2", "x * 3")}, reason: "function value stored"},
		{name: "method removed", base: []edit{add("lib.go", "func (Circle) Name() string { return \"c\" }\n")}, edits: []edit{sub("lib.go", "func (Circle) Name() string { return \"c\" }\n", "")}, reason: "method set"},
		{name: "const outside function", base: []edit{add("lib.go", "type arr [Limit]int\n")}, edits: []edit{sub("lib.go", "const Limit = 10", "const Limit = 11")}, reason: "outside a function"},
		{name: "shared helper", edits: []edit{sub("helper_test.go", "sharedHelper() int { return Everywhere2() }", "sharedHelper() int { return 2 }")}, reason: "shared test helper"},
		{name: "reflect call", edits: []edit{add("lib.go", "func R() { reflect.ValueOf(Comment).Call(nil) }\nfunc _x() { _ = 1 }\n")}, reason: "reflect.Call"},
		{name: "linkname", edits: []edit{add("lib.go", "func L() {}\n"), put("l.go", "package fx\n\nimport _ \"unsafe\"\n\n//go:linkname L2 runtime.foo\nfunc L2()\n")}, reason: ""},
		{name: "syntax error", edits: []edit{add("lib.go", "func {\n")}, reason: "head"},
		{name: "type error", edits: []edit{sub("lib.go", "x * 2", "x * nope")}, reason: "head graph"},
		{name: "base type error", base: nil, edits: []edit{sub("lib.go", "x * 2", "x * 3")}, reason: ""},
		{name: "fraction cap", edits: func() []edit {
			var es []edit
			es = append(es, sub("lib.go", "func Comment() int { return 1 }", "func Comment() int { return 2 }"))
			var b strings.Builder
			b.WriteString("package fx\n\nimport \"testing\"\n\n")
			for i := range 200 {
				fmt.Fprintf(&b, "func TestC%d(t *testing.T) { Comment() }\n", i)
			}
			es = append(es, put("more_test.go", b.String()))
			return es
		}(), reason: "exceeds cap"},
	} {
		if tc.name == "base type error" {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			p := selectEdits(t, hostOnly(), tc.base, tc.edits)
			if p.Tier != TierFull {
				t.Fatalf("tier = %s, tests %v, want full", p.Tier, p.Tests)
			}
			if !strings.Contains(strings.Join(p.Reasons, ";"), tc.reason) {
				t.Errorf("reasons = %v, want %q", p.Reasons, tc.reason)
			}
		})
	}
}

func TestFailOpenBaseUnloadable(t *testing.T) {
	b := baseFiles()
	b["lib.go"] = strings.Replace(b["lib.go"], "x * 2", "x * nope", 1)
	h := baseFiles()
	p := Select(writeFiles(t, b), writeFiles(t, h), Options{Configs: hostOnly()})
	if p.Tier != TierFull || !strings.Contains(strings.Join(p.Reasons, ";"), "base graph") {
		t.Fatalf("plan = %+v", p)
	}
}

func TestFailOpenMissingTrees(t *testing.T) {
	if p := Select("/nonexistent/a", "/nonexistent/b", Options{Configs: hostOnly()}); p.Tier != TierFull {
		t.Fatalf("plan = %+v", p)
	}
	if p := run("/nonexistent", "HEAD", "HEAD", ""); p.Tier != TierFull {
		t.Fatalf("plan = %+v", p)
	}
}

func TestDeletedFunctionUsesBaseGraph(t *testing.T) {
	b := baseFiles()
	h := baseFiles()
	h["lib.go"] = strings.Replace(h["lib.go"], "func Deleted() int { return 1 }\n", "", 1)
	h["lib.go"] = strings.Replace(h["lib.go"], "return Deleted()", "return 1", 1)
	dirB, dirH := writeFiles(t, b), writeFiles(t, h)
	g, err := buildGraph(dirH, hostOnly()[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, _, ok := g.RootsReaching("Deleted"); ok {
		t.Fatal("head graph should not know Deleted")
	}
	p := Select(dirB, dirH, Options{Configs: hostOnly()})
	if p.Tier != TierLibrary || !slices.Contains(p.Tests, "TestCallsDeleted") {
		t.Fatalf("plan = %+v", p)
	}
}

func TestPerConfigGraphUnion(t *testing.T) {
	constrained := []edit{
		put("arch_amd64_test.go", "package fx\n\nimport \"testing\"\n\nfunc TestOnlyAmd64(t *testing.T) { Double(1) }\n"),
		put("arch_arm64_test.go", "package fx\n\nimport \"testing\"\n\nfunc TestOnlyArm64(t *testing.T) { Double(1) }\n"),
		put("race_test.go", "//go:build race\n\npackage fx\n\nimport \"testing\"\n\nfunc TestOnlyRace(t *testing.T) { Double(1) }\n"),
		put("norace_test.go", "//go:build !race\n\npackage fx\n\nimport \"testing\"\n\nfunc TestOnlyNorace(t *testing.T) { Double(1) }\n"),
		put("combo_amd64_test.go", "//go:build !race\n\npackage fx\n\nimport \"testing\"\n\nfunc TestAmd64Norace(t *testing.T) { Double(1) }\n"),
		put("combo_arm64_test.go", "//go:build race\n\npackage fx\n\nimport \"testing\"\n\nfunc TestArm64Race(t *testing.T) { Double(1) }\n"),
	}
	p := selectEdits(t, DefaultConfigs, constrained, []edit{sub("lib.go", "x * 2", "x * 3")})
	if p.Tier != TierLibrary {
		t.Fatalf("plan = %+v", p)
	}
	for _, want := range []string{"TestOnlyAmd64", "TestOnlyArm64", "TestOnlyRace", "TestOnlyNorace", "TestAmd64Norace", "TestArm64Race", "TestTwice"} {
		if !slices.Contains(p.Tests, want) {
			t.Errorf("missing %s from %v", want, p.Tests)
		}
	}
}

func TestSourceReaders(t *testing.T) {
	for _, tc := range []struct {
		name, file, src string
	}{
		{"architecture name", "architecture_a_test.go", `package fx
import "testing"
func TestReader(t *testing.T) {}
`},
		{"go parser", "r_test.go", `package fx
import ("testing"; "go/parser")
var _ = parser.ParseFile
func TestReader(t *testing.T) {}
`},
		{"go file literal", "r_test.go", `package fx
import ("testing"; "os")
func TestReader(t *testing.T) { os.ReadFile("lib.go") }
`},
		{"readme literal", "r_test.go", `package fx
import ("testing"; "os")
func TestReader(t *testing.T) { os.ReadFile("README.md") }
`},
		{"directory walk", "r_test.go", `package fx
import ("testing"; "os")
func TestReader(t *testing.T) { os.ReadDir(".") }
`},
		{"runtime caller", "r_test.go", `package fx
import ("testing"; "runtime")
func TestReader(t *testing.T) { runtime.Caller(0) }
`},
		{"exec", "r_test.go", `package fx
import ("testing"; "os/exec")
func TestReader(t *testing.T) { exec.Command("go") }
`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := []edit{put(tc.file, tc.src)}
			for _, head := range [][]edit{
				{sub("README.md", "fx", "gx")},
				{sub("lib.go", "// Comment is documented.", "// Reworded.")},
				{sub("lib.go", "x * 2", "x * 3")},
				{sub("filler_test.go", "TestFill3(t *testing.T) { }", "TestFill3(t *testing.T) { t.Log(1) }")},
			} {
				p := selectEdits(t, hostOnly(), base, head)
				if !slices.Contains(p.Tests, "TestReader") {
					t.Errorf("TestReader not selected: %+v", p)
				}
			}
		})
	}
	p := selectEdits(t, hostOnly(), []edit{put("r_test.go", `package fx
import ("testing"; "os")
func TestPlain(t *testing.T) { os.ReadFile("testdata/d.json") }
`)}, []edit{sub("README.md", "fx", "gx")})
	if slices.Contains(p.Tests, "TestPlain") {
		t.Errorf("testdata reader treated as source reader")
	}
}

func TestBuildRegex(t *testing.T) {
	re, err := buildRegex([]string{"TestA", "TestAB"})
	if err != nil {
		t.Fatal(err)
	}
	if want := "^(?:TestA|TestAB)$"; re != want {
		t.Fatalf("regex = %s", re)
	}
	if none, _ := buildRegex(nil); none != "^$" {
		t.Fatalf("empty regex = %s", none)
	}
	r := regexp.MustCompile(re)
	for name, want := range map[string]bool{"TestA": true, "TestAB": true, "TestABC": false, "XTestA": false} {
		if r.MatchString(name) != want {
			t.Errorf("%s: want %v", name, want)
		}
	}
	long := make([]string, 0, 8000)
	for i := range 8000 {
		long = append(long, fmt.Sprintf("TestLongName%04d", i))
	}
	if _, err := buildRegex(long); err == nil {
		t.Fatal("over-long regexp accepted")
	}
	if _, err := buildRegex([]string{"TestA/b"}); err == nil {
		t.Fatal("name splitting under go test accepted")
	}
}

func TestLongRegexFailsOpen(t *testing.T) {
	pad := strings.Repeat("P", 200)
	tree := &tree{roots: map[string]Root{}}
	sel := map[string]Root{}
	for i := range 400 {
		n := fmt.Sprintf("Test%s%d", pad, i)
		tree.roots[n] = Root{Name: n, Kind: KindTest}
		sel[n] = tree.roots[n]
	}
	for i := range 2000 {
		n := fmt.Sprintf("TestQ%d", i)
		tree.roots[n] = Root{Name: n, Kind: KindTest}
	}
	p := finishPlan(TierLibrary, sel, nil, tree)
	if p.Tier != TierFull || !strings.Contains(strings.Join(p.Reasons, ";"), "exceeds cap") {
		t.Fatalf("plan tier %s reasons %v", p.Tier, p.Reasons)
	}
}

func TestCommandsPerTier(t *testing.T) {
	has := func(cs []string, sub string) bool {
		return slices.ContainsFunc(cs, func(c string) bool { return strings.Contains(c, sub) })
	}
	docs := Plan{Tier: TierDocs, Run: noneRegex, Bench: noneRegex}.Commands()
	if !has(docs, "go build") || !has(docs, "go vet") || !has(docs, "gofmt") || has(docs, "go test") {
		t.Errorf("docs = %v", docs)
	}
	tests := Plan{Tier: TierTests, Run: "^(?:TestA)$", Bench: noneRegex}.Commands()
	if !has(tests, "-run '^(?:TestA)$'") || has(tests, "-race") || has(tests, "GOARCH") {
		t.Errorf("tests = %v", tests)
	}
	lib := Plan{Tier: TierLibrary, Run: "^(?:TestA)$", Bench: "^(?:BenchmarkA)$"}.Commands()
	for _, want := range []string{"GOARCH=amd64 go test -run", "go test -race -run", "-bench '^(?:BenchmarkA)$'"} {
		if !has(lib, want) {
			t.Errorf("library lacks %q: %v", want, lib)
		}
	}
	full := fullPlan("x").Commands()
	for _, want := range []string{"go test -race -count=1", "GOARCH=amd64 go test -count=1", "-bench .", "selecttests test"} {
		if !has(full, want) {
			t.Errorf("full lacks %q: %v", want, full)
		}
	}
}

func TestGraphCacheRoundTrip(t *testing.T) {
	dir := writeFiles(t, baseFiles())
	cache := filepath.Join(t.TempDir(), "g.gob.gz")
	g, err := LoadGraph(dir, hostOnly()[0], cache)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadGraph("/nonexistent", hostOnly()[0], cache)
	if err != nil {
		t.Fatal(err)
	}
	a, _, _ := g.RootsReaching("Double")
	b, _, _ := got.RootsReaching("Double")
	if !slices.Equal(a, b) || len(a) == 0 {
		t.Fatalf("cached reach %v, fresh %v", b, a)
	}
}

func TestRootClassification(t *testing.T) {
	src := `package fx
func TestA(t *T) {}
func Testable() {}
func Test() {}
func Test1() {}
func TestMain(m *M) {}
func BenchmarkA(b *B) {}
func Benchmarks() {}
func FuzzA(f *F) {}
func ExampleA() {}
func Examples() {}
func (r R) TestM() {}
`
	f, err := parser.ParseFile(token.NewFileSet(), "x.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]RootKind{}
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok {
			if k, ok := classifyRoot(fd); ok {
				got[fd.Name.Name] = k
			}
		}
	}
	want := map[string]RootKind{"TestA": KindTest, "Test": KindTest, "Test1": KindTest, "BenchmarkA": KindBench, "FuzzA": KindTest, "ExampleA": KindTest}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestFuncStateAfterInitIsUncertain(t *testing.T) {
	h := baseFiles()
	h["lib.go"] += "var hook func() int\n\nfunc SetHook() { hook = inner }\n\nfunc CallHook() int { return hook() }\n"
	g, err := buildGraph(writeFiles(t, h), hostOnly()[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(g.Uncertain, ";"), "hook") {
		t.Fatalf("uncertain = %v", g.Uncertain)
	}
	m := baseFiles()
	m["lib.go"] += "var syncFuncs sync.Map\n\nfunc PutSync() { syncFuncs.Store(1, inner) }\n"
	m["lib.go"] = strings.Replace(m["lib.go"], `import "reflect"`, "import (\n\t\"reflect\"\n\t\"sync\"\n)", 1)
	m["lib.go"] += "var anyState = map[string]any{}\n\nfunc PutFunc() { anyState[\"f\"] = inner }\n"
	g, err = buildGraph(writeFiles(t, m), hostOnly()[0])
	if err != nil {
		t.Fatal(err)
	}
	if u := strings.Join(g.Uncertain, ";"); !strings.Contains(u, "anyState") || !strings.Contains(u, "syncFuncs") {
		t.Fatalf("func boxed into interface state not uncertain: %v", g.Uncertain)
	}
	for _, src := range []string{
		"type box struct{ f func() int }\n\nvar boxes = []box{{}}\n\nfunc setBox(b *box) { b.f = inner }\n\nfunc Fill() { setBox(&boxes[0]) }\n",
		"var fns = make([]func() int, 1)\n\nfunc Fill() { copy(fns, []func() int{inner}) }\n",
	} {
		f := baseFiles()
		f["lib.go"] += src
		g, err := buildGraph(writeFiles(t, f), hostOnly()[0])
		if err != nil {
			t.Fatal(err)
		}
		if len(g.Uncertain) == 0 {
			t.Errorf("not uncertain: %s", src)
		}
	}
	b := baseFiles()
	b["lib.go"] += "var table = []func() int{inner}\n\nfunc Size() int { return len(table) }\n\nvar counts = map[string]any{}\n\nfunc Count() { counts[\"n\"] = new(int) }\n\nvar syncCounts sync.Map\n\nfunc SyncCount() { syncCounts.Store(\"n\", new(int)) }\n"
	b["lib.go"] = strings.Replace(b["lib.go"], `import "reflect"`, "import (\n\t\"reflect\"\n\t\"sync\"\n)", 1)
	g, err = buildGraph(writeFiles(t, b), hostOnly()[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Uncertain) != 0 {
		t.Fatalf("init-only func table uncertain: %v", g.Uncertain)
	}
}

func TestWriterThroughLoadedReference(t *testing.T) {
	base := []edit{
		add("lib.go", "var cache = map[int]int{}\n\nfunc Fill() { cache[1] = 2 }\n\nfunc Look() int { return cache[1] }\n\nvar rows = [][]int{{1}}\n\nfunc Poke() { rows[0][0] = 2 }\n\nfunc Peek() int { return rows[0][0] }\n"),
		add("lib_test.go", "func TestFill(t *testing.T) { Fill() }\nfunc TestLook(t *testing.T) { Look() }\nfunc TestPoke(t *testing.T) { Poke() }\nfunc TestPeek(t *testing.T) { Peek() }\n"),
	}
	for _, tc := range []struct {
		edit edit
		want []string
	}{
		{sub("lib.go", "cache[1] = 2", "cache[1] = 3"), []string{"TestFill", "TestLook"}},
		{sub("lib.go", "rows[0][0] = 2", "rows[0][0] = 3"), []string{"TestPeek", "TestPoke"}},
	} {
		p := selectEdits(t, hostOnly(), base, []edit{tc.edit})
		if p.Tier != TierLibrary || !slices.Equal(p.Tests, tc.want) {
			t.Errorf("plan %s %v %v, want %v", p.Tier, p.Tests, p.Reasons, tc.want)
		}
	}
}

func TestChangedTestWriterSelectsReaders(t *testing.T) {
	base := []edit{
		add("other_test.go", "var table = [][]int{{1}}\n\nfunc TestW(t *testing.T) { counter = 1; mutate(table[0]) }\nfunc TestTableReader(t *testing.T) { _ = table[0][0] }\n"),
		add("lib.go", "func mutate(x []int) { x[0] = 2 }\n"),
	}
	p := selectEdits(t, hostOnly(), base, []edit{sub("other_test.go", "counter = 1", "counter = 2")})
	if p.Tier != TierTests {
		t.Fatalf("plan %+v", p)
	}
	for _, want := range []string{"TestW", "TestRead", "TestBump", "TestTableReader"} {
		if !slices.Contains(p.Tests, want) {
			t.Errorf("missing %s: %v", want, p.Tests)
		}
	}
}
