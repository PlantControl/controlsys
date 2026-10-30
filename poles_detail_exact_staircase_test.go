package controlsys

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/big"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// exactStairRecord runs finiteCountBracket on the row-major n×n pencil
// and returns its result, the SHA-256 digest of the per-level records of
// both runs (every big value in exact binary 'p' form, in the record
// format of the 76a7a1e verifier) and a float64 summary for logs.
func exactStairRecord(n int, a, e []float64) (exactFiniteCount, string, string) {
	var rec [2]strings.Builder
	var dump [2]strings.Builder
	txt := func(x *big.Float) string {
		if x == nil {
			return "-"
		}
		return x.Text('p', 0)
	}
	f := func(x *big.Float) float64 {
		if x == nil {
			return math.NaN()
		}
		v, _ := x.Float64()
		return v
	}
	trace := func(lo bool, lv *exactStairLevel) {
		run := 0
		if lo {
			run = 1
		}
		b := &rec[run]
		fmt.Fprintf(b, "%d %d %d %d %d %s %s %s %s", lv.k, lv.r, lv.zero, lv.undecided, lv.nonzero, txt(lv.rho), txt(lv.sigmaA), txt(lv.gap), txt(lv.sinZ))
		sv := make([]float64, len(lv.sv))
		for i, s := range lv.sv {
			b.WriteString(" " + txt(s))
			sv[i] = f(s)
		}
		b.WriteString("\n")
		fmt.Fprintf(&dump[run], "\n  %s k=%d r=%d zones=%d/%d/%d rho=%.3g sigmaA=%.3g gap=%.3g sinZ=%.3g sv=%.4g",
			[]string{"hi", "lo"}[run], lv.k, lv.r, lv.zero, lv.undecided, lv.nonzero, f(lv.rho), f(lv.sigmaA), f(lv.gap), f(lv.sinZ), sv)
	}
	c := finiteCountBracket(mat.NewDense(n, n, a), mat.NewDense(n, n, e), trace)
	decided := c.lo == c.hi && c.reason == ""
	var b strings.Builder
	for _, r := range rec {
		b.WriteString(r.String())
		b.WriteString("|\n")
	}
	fmt.Fprintf(&b, "%d %d %d %v %q", c.hi, c.lo, c.hi, decided, c.reason)
	summary := fmt.Sprintf("Lo=%d Hi=%d %s%s%s", c.lo, c.hi, c.reason, dump[0].String(), dump[1].String())
	return c, fmt.Sprintf("%x", sha256.Sum256([]byte(b.String()))), summary
}

// TestPolesDetailExactStaircase_MatchesParkedVerifier: the library exact
// staircase reproduces, record for record, the 320-bit verifier parked at
// commit 76a7a1e (ELT76F) on its 18 stiff-chain literals. The digests are
// those of that verifier, equal on arm64, amd64 and -race; big.Float is
// correctly rounded, so they hold on every platform.
func TestPolesDetailExactStaircase_MatchesParkedVerifier(t *testing.T) {
	want := map[string]string{
		"two:5093:0": "ef1679cd6c05d87b41a9ab05bb7270edc3500eb3e282a0420567624a6853ab90",
		"cpx:1297:2": "f5968f8506356ca5f65977b86222a3cff596ee916e634c0a8dd4f1fc6b74fe38",
	}
	var all strings.Builder
	for _, l := range poleCountCases() {
		c, d, s := exactStairRecord(l.n, l.a, l.e)
		all.WriteString(d)
		if w, ok := want[l.name]; ok && d != w {
			t.Errorf("%s: digest %s, want %s\n%s", l.name, d, w, s)
		}
		if c.lo > l.want || c.hi < l.want {
			t.Errorf("%s: bracket [%d, %d] misses the structural count %d\n%s", l.name, c.lo, c.hi, l.want, s)
		}
	}
	const wantAll = "3a023662f51be79a581aaea427fb130ceaa016bc235889c1e78e377dc0632ce2"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(all.String()))); got != wantAll {
		t.Errorf("combined level-record digest %s, want %s", got, wantAll)
	}
}
