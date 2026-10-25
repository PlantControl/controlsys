package controlsys

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"sort"
	"sync"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Fourth frozen RootLocus regression sweep (track rlocus, wave 20 review
// 5): the independent reviewer's generator and exact oracle, seed 20261022,
// 2500 cases (frozen, rvSweep4). Families: diagonal, Jordan, fast/slow, near pole-zero
// cancellation, D sign, discrete with input/output delay, integrators,
// near-axis complex zeros, high relative degree; each identity, transposed,
// diagonally scaled, Givens-rotated or triangular-similar. Breakaway is
// judged by the exact big.Rat TF of the float realization (rvOracleFor's
// A, numerator cut to the intended degree; RootLocus semantics since wave
// 24, J7OYOY), the other components by that of the intended realization
// (I). A breakaway verdict a 1-ulp change of A, B, C changes (Prec) is
// logged, not enforced, as are the components judged by I of a case whose
// float TF is not the intended one (Differs), a centroid base was not
// judged on (its asymptote count was wrong) and the rlrvLimits verdicts.
// testdata/rlocus_sweep_rv_base.json holds base rlswBaseRev verdicts per
// arch; regenerate only with base code: RLOCUS_SWEEPRV_WRITE=<path>.

const (
	rlrvSeed  = 20261022
	rlrvCases = 2500
)

const (
	rlrvBk = 1 << iota
	rlrvAsym
	rlrvCen
	rlrvDep
	rlrvArr
)

func rvRun(c rvCase) rvOut {
	o := rvOut{ID: c.ID}
	defer func() {
		if r := recover(); r != nil {
			o.Err = fmt.Sprintf("PANIC %v", r)
		}
	}()
	n := c.N
	sys, err := New(mat.NewDense(n, n, c.A), mat.NewDense(n, 1, c.B), mat.NewDense(1, n, c.C), mat.NewDense(1, 1, []float64{c.D}), c.Dt)
	if err != nil {
		o.Err = "new: " + err.Error()
		return o
	}
	if c.InDelay > 0 {
		if err := sys.SetInputDelay([]float64{float64(c.InDelay)}); err != nil {
			o.Err = "delay: " + err.Error()
			return o
		}
	}
	if c.OutDelay > 0 {
		if err := sys.SetOutputDelay([]float64{float64(c.OutDelay)}); err != nil {
			o.Err = "delay: " + err.Error()
			return o
		}
	}
	abs := sys
	if sys.HasDelay() {
		abs, _ = sys.AbsorbDelay()
	}
	if p, err := abs.Poles(); err == nil {
		o.Poles = rvPairs(p)
	}
	if z, err := abs.Zeros(); err == nil {
		o.Zeros = rvPairs(z)
	}
	res, err := RootLocus(sys, c.Gains)
	if err != nil {
		o.Err = err.Error()
		return o
	}
	o.Bk = rvPairs(res.Breakaway)
	o.Asym = res.AsymptoteAngles
	o.Cen, o.CenOK = res.AsymptoteCentroid()
	o.Dep, o.Arr = res.DepartureAngles, res.ArrivalAngles
	return o
}

func rvPairs(z []complex128) [][2]float64 {
	o := make([][2]float64, len(z))
	for i, v := range z {
		o[i] = [2]float64{real(v), imag(v)}
	}
	return o
}

func rvBits(s rvScore) int {
	if s.Err != "" {
		return 0
	}
	b := 0
	for i, ok := range []bool{s.Bk, s.As, s.Ce, s.Dp, s.Ar} {
		if ok {
			b |= 1 << i
		}
	}
	return b
}

// rvOracleCache is rvOra in JSON form (the oracle is frozen with the
// cases: testdata/rlocus_sweep_rv_oracle.json.gz, rewritten by
// RLOCUS_SWEEPRV_ORACLE=<path>) less the intended poles and zeros, which
// are float roots of the exact TF and so are recomputed on each arch.
type rvOracleCache struct {
	I, A    rvOracle
	Differs bool
	Prec    bool
	PrecWhy string
	Phase0  float64
	DegN    int
}

// rvSweep is one frozen sweep of rvCase cases: cases, oracle cache and base
// verdicts live in testdata; the env variables regenerate them (cases only
// when gen is set).
type rvSweep struct {
	name                  string
	count                 int
	gen                   func() []rvCase
	cases, oracle, golden string
	envCases, envOracle   string
	envWrite              string
	limits                map[int]string
}

var rvSweep4 = rvSweep{
	name: "RV", count: rlrvCases, gen: func() []rvCase { return rvGenerate(rlrvSeed, rlrvCases) },
	cases: "testdata/rlocus_sweep_rv_cases.json.gz", oracle: "testdata/rlocus_sweep_rv_oracle.json.gz",
	golden: "testdata/rlocus_sweep_rv_base.json", envCases: "RLOCUS_SWEEPRV_CASES",
	envOracle: "RLOCUS_SWEEPRV_ORACLE", envWrite: "RLOCUS_SWEEPRV_WRITE", limits: rlrvLimits,
}

// frozenCases returns the sweep cases, generated once on arm64 (the
// Givens angles go through math.Cos/Sin, whose last bits differ by arch)
// and frozen in sw.cases.
func (sw rvSweep) frozenCases(t *testing.T) []rvCase {
	t.Helper()
	if path := os.Getenv(sw.envCases); path != "" && sw.gen != nil {
		cs := sw.gen()
		rvWriteGzipJSON(t, path, cs)
		return cs
	}
	f, err := os.Open(sw.cases)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var cs []rvCase
	if err := json.NewDecoder(zr).Decode(&cs); err != nil {
		t.Fatal(err)
	}
	if len(cs) != sw.count {
		t.Fatalf("frozen cases: %d, want %d", len(cs), sw.count)
	}
	return cs
}

func rvWriteGzipJSON(t *testing.T, path string, v any) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := gzip.NewWriter(f)
	if err := json.NewEncoder(zw).Encode(v); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func (sw rvSweep) cachedOracles(t *testing.T, cs []rvCase) []rvOra {
	t.Helper()
	if path := os.Getenv(sw.envOracle); path != "" {
		oras := rvOracles(cs)
		cache := make([]rvOracleCache, len(oras))
		for i, o := range oras {
			cache[i] = rvOracleCache{I: o.I, A: o.A, Differs: o.Differs, Prec: o.Prec, PrecWhy: o.PrecWhy, Phase0: o.Phase0, DegN: o.DegN}
		}
		rvWriteGzipJSON(t, path, cache)
		return oras
	}
	f, err := os.Open(sw.oracle)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var cache []rvOracleCache
	if err := json.NewDecoder(zr).Decode(&cache); err != nil {
		t.Fatal(err)
	}
	if len(cache) != len(cs) {
		t.Fatalf("oracle cache has %d cases, sweep %d", len(cache), len(cs))
	}
	oras := make([]rvOra, len(cache))
	for i, c := range cache {
		ni, di := rvExactTF(cs[i].A0, cs[i].B0, cs[i].C0, cs[i].D, cs[i].N, cs[i].InDelay+cs[i].OutDelay)
		oras[i] = rvOra{I: c.I, A: c.A, Differs: c.Differs, Prec: c.Prec, PrecWhy: c.PrecWhy, IPoles: rvNumRoots(rvPFloat(di)), IZeros: rvNumRoots(rvPFloat(ni)), Phase0: c.Phase0, DegN: c.DegN}
	}
	return oras
}

func rvOracles(cs []rvCase) []rvOra {
	oras := make([]rvOra, len(cs))
	var wg sync.WaitGroup
	ch := make(chan int)
	for range runtime.NumCPU() {
		wg.Go(func() {
			for i := range ch {
				oras[i] = rvOracleFor(cs[i])
			}
		})
	}
	for i := range cs {
		ch <- i
	}
	close(ch)
	wg.Wait()
	return oras
}

// rlrvLimits are base-correct breakaway verdicts logged, not enforced
// (none since the exact breakaway path; the w17 #58 cancellation class
// 1328, 1517, 1622, 1794 is resolved exactly).
var rlrvLimits = map[int]string{}

type rlrvGolden struct {
	Base string            `json:"base"`
	OK   map[string]string `json:"ok"`
}

func TestRootLocusSweepRVNoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2500-case sweep")
	}
	skipSlowUnderRace(t)
	rvSweep4.check(t, rlswBaseRev)
}

// check fails when a component that base got right is now wrong; cases
// are named by ID and limits are keyed by ID.
func (sw rvSweep) check(t *testing.T, baseRev string) {
	t.Helper()
	cs := sw.frozenCases(t)
	oras := sw.cachedOracles(t, cs)
	bits := make([]int, len(cs))
	outs := make([]rvOut, len(cs))
	for i, c := range cs {
		outs[i] = rvRun(c)
		bits[i] = rvBits(rvScoreCase(c, oras[i], outs[i]))
	}
	enc := func(v []int) string {
		b := make([]byte, 0, 2*len(v))
		for _, x := range v {
			b = fmt.Appendf(b, "%02x", x)
		}
		return string(b)
	}
	if path := os.Getenv(sw.envWrite); path != "" {
		g := rlrvGolden{Base: baseRev, OK: map[string]string{}}
		if data, err := os.ReadFile(path); err == nil {
			if err := json.Unmarshal(data, &g); err != nil {
				t.Fatal(err)
			}
		}
		g.OK[runtime.GOARCH] = enc(bits)
		data, err := json.Marshal(g)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	data, err := os.ReadFile(sw.golden)
	if err != nil {
		t.Fatal(err)
	}
	var g rlrvGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	s := g.OK[runtime.GOARCH]
	if len(s) != 2*len(cs) {
		t.Fatalf("golden has %d cases for %s, sweep %d", len(s)/2, runtime.GOARCH, len(cs))
	}
	names := []string{"Breakaway", "Asymptotes", "Centroid", "DepartureAngles", "ArrivalAngles"}
	var fixed, regressed [5]int
	for i, c := range cs {
		var base int
		fmt.Sscanf(s[2*i:2*i+2], "%02x", &base)
		for k := range 5 {
			m := 1 << k
			switch {
			case base&m != 0 && bits[i]&m == 0 && m == rlrvBk && oras[i].Prec:
				t.Logf("case %d %s/%s: Breakaway differs from base at a precision limit (%s): %v", c.ID, c.Fam, c.Xf, oras[i].PrecWhy, outs[i].Bk)
			case base&m != 0 && bits[i]&m == 0 && m != rlrvBk && oras[i].Differs:
				t.Logf("case %d %s/%s: %s differs from base on a float TF that is not the intended one", c.ID, c.Fam, c.Xf, names[k])
			case base&m != 0 && bits[i]&m == 0 && m == rlrvCen && base&rlrvAsym == 0:
				t.Logf("case %d %s/%s: Centroid not judged at base (asymptote count wrong there), now %v vs %v", c.ID, c.Fam, c.Xf, outs[i].Cen, oras[i].I.Centroid)
			case base&m != 0 && bits[i]&m == 0 && m == rlrvBk && sw.limits[c.ID] != "":
				t.Logf("case %d %s/%s: Breakaway differs from base at a known limit: %s", c.ID, c.Fam, c.Xf, sw.limits[c.ID])
			case base&m != 0 && bits[i]&m == 0:
				regressed[k]++
				if regressed[k] <= 5 {
					t.Errorf("case %d %s/%s: %s regressed: Breakaway %v want %v (rejected %v) err %q", c.ID, c.Fam, c.Xf, names[k], outs[i].Bk, oras[i].A.Pts, oras[i].A.Rej, outs[i].Err)
				}
			case base&m == 0 && bits[i]&m != 0:
				fixed[k]++
			}
		}
	}
	t.Logf("fixed vs base: %v; regressed: %v", fixed, regressed)
}

type rvCase struct {
	ID       int
	Fam, Xf  string
	N        int
	A0, B0   []float64
	C0       []float64
	A, B, C  []float64
	D        float64
	Dt       float64
	InDelay  int
	OutDelay int
	Gains    []float64
}

type rvGen struct{ r *rand.Rand }

func (g *rvGen) dy(bits, emin, emax int) float64 {
	m := 1 + g.r.IntN(1<<bits-1)
	e := emin + g.r.IntN(emax-emin+1)
	return math.Ldexp(float64(m), e)
}
func (g *rvGen) sgn(pneg float64) float64 {
	if g.r.Float64() < pneg {
		return -1
	}
	return 1
}
func (g *rvGen) vec(n, bits, emin, emax int, pzero float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		if g.r.Float64() < pzero {
			continue
		}
		v[i] = g.sgn(0.5) * g.dy(bits, emin, emax)
	}
	nz := false
	for _, x := range v {
		nz = nz || x != 0
	}
	if !nz {
		v[g.r.IntN(n)] = 1
	}
	return v
}

func (g *rvGen) dval(p float64) float64 {
	if g.r.Float64() >= p {
		return 0
	}
	ds := []float64{0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8}
	return g.sgn(0.5) * ds[g.r.IntN(len(ds))]
}

type rvRoot struct{ re, im float64 } // im>0 means pair

func rvPolyRoots(rs []rvRoot) rvP {
	p := rvP{rvRi(1)}
	for _, x := range rs {
		if x.im == 0 {
			p = rvPMul(p, rvP{rvRat(-x.re), rvRi(1)})
		} else {
			re, im := rvRat(x.re), rvRat(x.im)
			c0 := new(big.Rat).Mul(re, re)
			c0.Add(c0, new(big.Rat).Mul(im, im))
			c1 := new(big.Rat).Mul(re, rvRi(-2))
			p = rvPMul(p, rvP{c0, c1, rvRi(1)})
		}
	}
	return p
}
func rvNroots(rs []rvRoot) int {
	n := 0
	for _, x := range rs {
		if x.im == 0 {
			n++
		} else {
			n += 2
		}
	}
	return n
}

func rvToF(p rvP, n int) ([]float64, bool) {
	out := make([]float64, n)
	for i := range n {
		if i < len(p) {
			f, ex := p[i].Float64()
			if !ex {
				return nil, false
			}
			out[i] = f
		}
	}
	return out, true
}

// rvCompanion realization of k*num/den + d: den monic deg n, num deg<n.
func rvCompanion(den, num rvP, n int) (a, b, c []float64, ok bool) {
	df, ok1 := rvToF(den, n+1)
	nf, ok2 := rvToF(num, n)
	if !ok1 || !ok2 {
		return nil, nil, nil, false
	}
	a = make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -df[j]
	}
	b = make([]float64, n)
	b[n-1] = 1
	c = nf
	for _, x := range append(append([]float64{}, a...), c...) {
		if math.IsInf(x, 0) || math.Abs(x) > 1e300 {
			return nil, nil, nil, false
		}
	}
	return a, b, c, true
}

// blockdiag of Jordan blocks.
type rvBlk struct {
	re, im float64
	size   int // real: jordan size; complex: 2
}

func rvJordanA(bs []rvBlk) ([]float64, int) {
	n := 0
	for _, b := range bs {
		n += b.size
	}
	a := make([]float64, n*n)
	o := 0
	for _, b := range bs {
		if b.im != 0 {
			a[o*n+o], a[o*n+o+1] = b.re, b.im
			a[(o+1)*n+o], a[(o+1)*n+o+1] = -b.im, b.re
		} else {
			for i := range b.size {
				a[(o+i)*n+o+i] = b.re
				if i+1 < b.size {
					a[(o+i)*n+o+i+1] = 1
				}
			}
		}
		o += b.size
	}
	return a, n
}

func (g *rvGen) realPole(bits, emin, emax int, pneg float64) float64 {
	return g.sgn(pneg) * g.dy(bits, emin, emax)
}

func (g *rvGen) poleSet(n, bits, emin, emax int, pneg, pcplx float64) []rvRoot {
	var rs []rvRoot
	for rvNroots(rs) < n {
		if n-rvNroots(rs) >= 2 && g.r.Float64() < pcplx {
			rs = append(rs, rvRoot{g.realPole(bits, emin, emax, pneg), g.dy(bits, emin, emax)})
		} else {
			rs = append(rs, rvRoot{g.realPole(bits, emin, emax, pneg), 0})
		}
	}
	return rs
}

func (g *rvGen) famDiag(c *rvCase) bool {
	n := 2 + g.r.IntN(5)
	a := make([]float64, n*n)
	seen := map[float64]bool{}
	for i := range n {
		p := g.realPole(6, -4, 0, 0.8)
		if seen[p] {
			return false
		}
		seen[p] = true
		a[i*n+i] = p
	}
	c.N, c.A0, c.B0, c.C0 = n, a, g.vec(n, 5, -3, 0, 0), g.vec(n, 5, -3, 0, 0)
	c.D = g.dval(0.25)
	return true
}

func (g *rvGen) jordanBlocks(nmax, bits, emin, emax int) []rvBlk {
	var bs []rvBlk
	n := 0
	target := 3 + g.r.IntN(nmax-2)
	for n < target {
		u := g.r.Float64()
		switch {
		case u < 0.35 && target-n >= 2:
			bs = append(bs, rvBlk{g.realPole(bits, emin, emax, 0.8), g.dy(bits, emin, emax), 2})
			n += 2
		case u < 0.65:
			s := min(2+g.r.IntN(2), target-n)
			bs = append(bs, rvBlk{g.realPole(bits, emin, emax, 0.8), 0, s})
			n += s
		default:
			bs = append(bs, rvBlk{g.realPole(bits, emin, emax, 0.8), 0, 1})
			n++
		}
	}
	return bs
}

func (g *rvGen) famJordan(c *rvCase) bool {
	a, n := rvJordanA(g.jordanBlocks(7, 5, -3, 1))
	c.N, c.A0, c.B0, c.C0 = n, a, g.vec(n, 4, -2, 0, 0.15), g.vec(n, 4, -2, 0, 0.15)
	c.D = g.dval(0.25)
	return true
}

func (g *rvGen) famFastSlow(c *rvCase) bool {
	if g.r.IntN(2) == 0 {
		n := 3 + g.r.IntN(4)
		a := make([]float64, n*n)
		b, cc := make([]float64, n), make([]float64, n)
		for i := range n {
			e := -16 + g.r.IntN(27)
			a[i*n+i] = g.sgn(0.85) * g.dy(3, e, e)
			b[i] = g.sgn(0.5) * g.dy(4, -3+e/2, e/2)
			cc[i] = g.sgn(0.5) * g.dy(4, -3+e/2, e/2)
		}
		for i := range n {
			for j := range i {
				if a[i*n+i] == a[j*n+j] {
					return false
				}
			}
		}
		c.N, c.A0, c.B0, c.C0 = n, a, b, cc
	} else {
		var bs []rvBlk
		n := 0
		target := 3 + g.r.IntN(5)
		for n < target {
			e := -16 + g.r.IntN(27)
			if target-n >= 2 && g.r.Float64() < 0.4 {
				bs = append(bs, rvBlk{g.sgn(0.85) * g.dy(3, e, e), g.dy(3, e-2, e+1), 2})
				n += 2
			} else {
				s := 1
				if g.r.Float64() < 0.3 && target-n >= 2 {
					s = 2
				}
				bs = append(bs, rvBlk{g.sgn(0.85) * g.dy(3, e, e), 0, s})
				n += s
			}
		}
		a, n := rvJordanA(bs)
		c.N, c.A0, c.B0, c.C0 = n, a, g.vec(n, 4, -4, 4, 0.1), g.vec(n, 4, -4, 4, 0.1)
	}
	c.D = g.dval(0.2)
	return true
}

func (g *rvGen) famNearCancel(c *rvCase) bool {
	if g.r.IntN(2) == 0 {
		if !g.famDiag(c) {
			return false
		}
		k := 8 + g.r.IntN(29)
		i := g.r.IntN(c.N)
		c.C0[i] = g.sgn(0.5) * math.Ldexp(1, -k)
		return true
	}
	n := 3 + g.r.IntN(4)
	poles := g.poleSet(n, 5, -3, 1, 0.8, 0.35)
	var zs []rvRoot
	m := g.r.IntN(n)
	for rvNroots(zs) < m {
		p := poles[g.r.IntN(len(poles))]
		k := 6 + g.r.IntN(15)
		if p.im != 0 && m-rvNroots(zs) >= 2 {
			zs = append(zs, rvRoot{p.re * (1 + g.sgn(0.5)*math.Ldexp(1, -k)), p.im * (1 + g.sgn(0.5)*math.Ldexp(1, -k))})
		} else if p.im == 0 {
			zs = append(zs, rvRoot{p.re * (1 + g.sgn(0.5)*math.Ldexp(1, -k)), 0})
		} else {
			zs = append(zs, rvRoot{g.realPole(4, -2, 1, 0.6), 0})
		}
	}
	return g.companionCase(c, poles, zs, n)
}

func (g *rvGen) companionCase(c *rvCase, poles, zs []rvRoot, n int) bool {
	den := rvPolyRoots(poles)
	num := rvPScale(rvPolyRoots(zs), rvRat(g.sgn(0.4)*g.dy(4, -2, 2)))
	a, b, cc, ok := rvCompanion(den, num, n)
	if !ok {
		return false
	}
	c.N, c.A0, c.B0, c.C0 = n, a, b, cc
	if c.D == 0 {
		c.D = g.dval(0.25)
	}
	return true
}

func (g *rvGen) famDSign(c *rvCase) bool {
	c.D = g.sgn(0.5) * []float64{0.125, 0.25, 0.5, 1, 1.5, 2, 4, 8}[g.r.IntN(8)]
	switch g.r.IntN(3) {
	case 0:
		d := c.D
		ok := g.famDiag(c)
		c.D = d
		return ok
	case 1:
		d := c.D
		g.famJordan(c)
		c.D = d
		return true
	}
	n := 3 + g.r.IntN(4)
	rel := min(1+g.r.IntN(3), n)
	poles := g.poleSet(n, 5, -3, 1, 0.75, 0.35)
	zs := g.poleSet(n-rel, 5, -3, 1, 0.6, 0.3)
	if rvNroots(zs) > n-rel {
		zs = zs[:len(zs)-1]
	}
	return g.companionCase(c, poles, zs, n)
}

func (g *rvGen) famDisc(c *rvCase) bool {
	c.Dt = []float64{1, 0.1, 0.5}[g.r.IntN(3)]
	n := 2 + g.r.IntN(4)
	var poles []rvRoot
	for rvNroots(poles) < n {
		u := g.r.Float64()
		switch {
		case u < 0.15:
			poles = append(poles, rvRoot{1, 0})
		case u < 0.45 && n-rvNroots(poles) >= 2:
			poles = append(poles, rvRoot{g.sgn(0.3) * g.dy(5, -6, 0), g.dy(5, -6, -1)})
		default:
			poles = append(poles, rvRoot{g.sgn(0.3) * float64(1+g.r.IntN(80)) / 64, 0})
		}
	}
	var ok bool
	switch g.r.IntN(3) {
	case 0:
		var bs []rvBlk
		for _, p := range poles {
			if p.im != 0 {
				bs = append(bs, rvBlk{p.re, p.im, 2})
			} else {
				bs = append(bs, rvBlk{p.re, 0, 1})
			}
		}
		for i := 0; i+1 < len(bs); i++ {
			if bs[i].im == 0 && bs[i+1].im == 0 && bs[i].re == bs[i+1].re {
				bs[i].size, bs = 2, append(bs[:i+1], bs[i+2:]...)
			}
		}
		a, nn := rvJordanA(bs)
		c.N, c.A0, c.B0, c.C0 = nn, a, g.vec(nn, 4, -2, 0, 0.1), g.vec(nn, 4, -2, 0, 0.1)
		c.D = g.dval(0.25)
		ok = true
	default:
		m := g.r.IntN(n)
		zs := g.poleSet(m, 5, -6, 0, 0.4, 0.3)
		if rvNroots(zs) > m {
			zs = zs[:len(zs)-1]
		}
		ok = g.companionCase(c, poles, zs, n)
	}
	L := 1 + g.r.IntN(4)
	if g.r.IntN(2) == 0 {
		c.InDelay = L
	} else {
		c.OutDelay = L
	}
	return ok
}

func (g *rvGen) famInteg(c *rvCase) bool {
	k := 1 + g.r.IntN(3)
	if g.r.IntN(2) == 0 {
		bs := []rvBlk{{0, 0, k}}
		rest := g.jordanBlocks(5, 5, -3, 1)
		if len(rest) > 2 {
			rest = rest[:2]
		}
		bs = append(bs, rest...)
		a, n := rvJordanA(bs)
		c.N, c.A0, c.B0, c.C0 = n, a, g.vec(n, 4, -2, 0, 0.1), g.vec(n, 4, -2, 0, 0.1)
		c.D = g.dval(0.2)
		return true
	}
	other := g.poleSet(1+g.r.IntN(4), 5, -3, 1, 0.85, 0.3)
	poles := other
	for range k {
		poles = append(poles, rvRoot{0, 0})
	}
	n := rvNroots(poles)
	m := g.r.IntN(n)
	zs := g.poleSet(m, 5, -4, 1, 0.8, 0.3)
	if rvNroots(zs) > m {
		zs = zs[:len(zs)-1]
	}
	return g.companionCase(c, poles, zs, n)
}

func (g *rvGen) famCplxZero(c *rvCase) bool {
	n := 3 + g.r.IntN(5)
	poles := g.poleSet(n, 5, -3, 1, 0.8, 0.35)
	var zs []rvRoot
	npair := 1 + g.r.IntN(max(1, (n-1)/2))
	for range npair {
		if rvNroots(zs)+2 > n-1 {
			break
		}
		a := g.realPole(5, -3, 1, 0.75)
		k := 2 + g.r.IntN(17)
		zs = append(zs, rvRoot{a, math.Abs(a) * math.Ldexp(1, -k)})
	}
	if rvNroots(zs) < n-1 && g.r.IntN(2) == 0 {
		zs = append(zs, rvRoot{g.realPole(4, -2, 1, 0.7), 0})
	}
	return g.companionCase(c, poles, zs, n)
}

func (g *rvGen) famHighRel(c *rvCase) bool {
	n := 5 + g.r.IntN(4)
	poles := g.poleSet(n, 4, -2, 1, 0.75, 0.45)
	var zs []rvRoot
	if g.r.IntN(2) == 0 {
		zs = []rvRoot{{g.realPole(4, -2, 1, 0.7), 0}}
	}
	if g.r.Float64() < 0.1 {
		c.D = g.dval(1)
	} else {
		c.D = 0
	}
	d := c.D
	ok := g.companionCase(c, poles, zs, n)
	c.D = d
	return ok
}

// transforms
func rvMatmul(a, b []float64, n int) []float64 {
	r := make([]float64, n*n)
	for i := range n {
		for k := range n {
			if a[i*n+k] == 0 {
				continue
			}
			for j := range n {
				r[i*n+j] += a[i*n+k] * b[k*n+j]
			}
		}
	}
	return r
}

func (g *rvGen) transform(c *rvCase) {
	n := c.N
	A, B, C := append([]float64{}, c.A0...), append([]float64{}, c.B0...), append([]float64{}, c.C0...)
	u := g.r.Float64()
	switch {
	case u < 0.35:
		c.Xf = "none"
	case u < 0.5:
		c.Xf = "transpose"
		At := make([]float64, n*n)
		for i := range n {
			for j := range n {
				At[i*n+j] = A[j*n+i]
			}
		}
		A, B, C = At, C, B
	case u < 0.7:
		c.Xf = "scale"
		s := make([]float64, n)
		for i := range s {
			s[i] = math.Ldexp(1, g.r.IntN(25)-12)
		}
		for i := range n {
			for j := range n {
				A[i*n+j] = A[i*n+j] * s[i] / s[j]
			}
			B[i] *= s[i]
			C[i] /= s[i]
		}
	case u < 0.9:
		c.Xf = "givens"
		for range n + 1 {
			i := g.r.IntN(n)
			j := g.r.IntN(n)
			if i == j {
				continue
			}
			th := g.r.Float64() * 2 * math.Pi
			cs, sn := math.Cos(th), math.Sin(th)
			// A <- G A G^T, B <- G B, C <- C G^T
			for k := range n {
				x, y := A[i*n+k], A[j*n+k]
				A[i*n+k], A[j*n+k] = cs*x-sn*y, sn*x+cs*y
			}
			for k := range n {
				x, y := A[k*n+i], A[k*n+j]
				A[k*n+i], A[k*n+j] = cs*x-sn*y, sn*x+cs*y
			}
			B[i], B[j] = cs*B[i]-sn*B[j], sn*B[i]+cs*B[j]
			C[i], C[j] = cs*C[i]-sn*C[j], sn*C[i]+cs*C[j]
		}
	default:
		c.Xf = "tri"
		vals := []float64{0.5, -0.5, 1, -1, 2, -0.25, 0.75, -1.5}
		T := make([]float64, n*n)
		for i := range n {
			T[i*n+i] = 1
			for j := range i {
				if g.r.IntN(2) == 0 {
					T[i*n+j] = vals[g.r.IntN(len(vals))]
				}
			}
		}
		// inverse of unit lower triangular by forward substitution
		Ti := make([]float64, n*n)
		for col := range n {
			for i := range n {
				s := 0.0
				if i == col {
					s = 1
				}
				for k := range i {
					s -= T[i*n+k] * Ti[k*n+col]
				}
				Ti[i*n+col] = s
			}
		}
		A = rvMatmul(rvMatmul(T, A, n), Ti, n)
		nb := make([]float64, n)
		nc := make([]float64, n)
		for i := range n {
			for k := range n {
				nb[i] += T[i*n+k] * B[k]
				nc[i] += C[k] * Ti[k*n+i]
			}
		}
		B, C = nb, nc
	}
	c.A, c.B, c.C = A, B, C
}

func (g *rvGen) gains(d float64) []float64 {
	if g.r.Float64() < 0.1 {
		return nil
	}
	gs := []float64{0}
	for range 5 {
		gs = append(gs, math.Ldexp(1+g.r.Float64(), g.r.IntN(13)-6))
	}
	gs = append(gs, 0.3, 7.7)
	out := gs[:0]
	for _, k := range gs {
		if 1+k*d != 0 {
			out = append(out, k)
		}
	}
	return out
}

var rvFams = []struct {
	name string
	f    func(*rvGen, *rvCase) bool
	w    int
}{
	{"diag", (*rvGen).famDiag, 8},
	{"jordan", (*rvGen).famJordan, 12},
	{"fastslow", (*rvGen).famFastSlow, 12},
	{"nearcancel", (*rvGen).famNearCancel, 12},
	{"dsign", (*rvGen).famDSign, 12},
	{"disc", (*rvGen).famDisc, 14},
	{"integ", (*rvGen).famInteg, 10},
	{"cplxzero", (*rvGen).famCplxZero, 10},
	{"highrel", (*rvGen).famHighRel, 10},
}

func rvGenerate(seed uint64, count int) []rvCase {
	g := &rvGen{rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
	tw := 0
	for _, f := range rvFams {
		tw += f.w
	}
	var out []rvCase
	for len(out) < count {
		x := g.r.IntN(tw)
		fi := 0
		for x >= rvFams[fi].w {
			x -= rvFams[fi].w
			fi++
		}
		c := rvCase{ID: len(out), Fam: rvFams[fi].name}
		if !rvFams[fi].f(g, &c) {
			continue
		}
		if c.N > 9 {
			continue
		}
		g.transform(&c)
		c.Gains = g.gains(c.D)
		out = append(out, c)
	}
	return out
}

// rvP is an exact polynomial, ascending coefficients.
type rvP []*big.Rat

func rvRat(f float64) *big.Rat { r := new(big.Rat); r.SetFloat64(f); return r }
func rvRi(i int64) *big.Rat    { return big.NewRat(i, 1) }

func (p rvP) trim() rvP {
	n := len(p)
	for n > 0 && p[n-1].Sign() == 0 {
		n--
	}
	return p[:n]
}
func (p rvP) deg() int { return len(p.trim()) - 1 }

func rvPAdd(a, b rvP) rvP {
	n := max(len(a), len(b))
	r := make(rvP, n)
	for i := range n {
		r[i] = new(big.Rat)
		if i < len(a) {
			r[i].Add(r[i], a[i])
		}
		if i < len(b) {
			r[i].Add(r[i], b[i])
		}
	}
	return r.trim()
}
func rvPNeg(a rvP) rvP {
	r := make(rvP, len(a))
	for i := range a {
		r[i] = new(big.Rat).Neg(a[i])
	}
	return r
}
func rvPSub(a, b rvP) rvP { return rvPAdd(a, rvPNeg(b)) }
func rvPMul(a, b rvP) rvP {
	a, b = a.trim(), b.trim()
	if len(a) == 0 || len(b) == 0 {
		return rvP{}
	}
	r := make(rvP, len(a)+len(b)-1)
	for i := range r {
		r[i] = new(big.Rat)
	}
	t := new(big.Rat)
	for i := range a {
		for j := range b {
			t.Mul(a[i], b[j])
			r[i+j].Add(r[i+j], t)
		}
	}
	return r.trim()
}
func rvPScale(a rvP, s *big.Rat) rvP {
	r := make(rvP, len(a))
	for i := range a {
		r[i] = new(big.Rat).Mul(a[i], s)
	}
	return r.trim()
}
func rvPDeriv(a rvP) rvP {
	a = a.trim()
	if len(a) <= 1 {
		return rvP{}
	}
	r := make(rvP, len(a)-1)
	for i := 1; i < len(a); i++ {
		r[i-1] = new(big.Rat).Mul(a[i], rvRi(int64(i)))
	}
	return r.trim()
}
func rvPDivMod(a, b rvP) (rvP, rvP) {
	a, b = a.trim(), b.trim()
	if len(a) < len(b) {
		return rvP{}, a
	}
	rem := make(rvP, len(a))
	for i := range a {
		rem[i] = new(big.Rat).Set(a[i])
	}
	q := make(rvP, len(a)-len(b)+1)
	for i := range q {
		q[i] = new(big.Rat)
	}
	lb := b[len(b)-1]
	t := new(big.Rat)
	for i := len(a) - 1; i >= len(b)-1; i-- {
		if rem[i].Sign() == 0 {
			continue
		}
		c := new(big.Rat).Quo(rem[i], lb)
		q[i-len(b)+1] = c
		for j := range b {
			t.Mul(c, b[j])
			rem[i-len(b)+1+j].Sub(rem[i-len(b)+1+j], t)
		}
	}
	return q.trim(), rem.trim()
}
func rvPMonic(a rvP) rvP {
	a = a.trim()
	if len(a) == 0 {
		return a
	}
	inv := new(big.Rat).Inv(a[len(a)-1])
	return rvPScale(a, inv)
}
func rvPGcd(a, b rvP) rvP {
	a, b = rvPMonic(a), rvPMonic(b)
	for len(b) > 0 {
		_, r := rvPDivMod(a, b)
		a, b = b, rvPMonic(r)
	}
	return rvPMonic(a)
}
func rvPEval(a rvP, x *big.Rat) *big.Rat {
	r := new(big.Rat)
	for _, v := range slices.Backward(a) {
		r.Mul(r, x)
		r.Add(r, v)
	}
	return r
}
func rvPFloat(a rvP) []float64 {
	r := make([]float64, len(a))
	for i := range a {
		r[i], _ = a[i].Float64()
	}
	return r
}

// rvYun returns square-free factors f[i] of multiplicity i+1.
func rvYun(f rvP) []rvP {
	f = rvPMonic(f)
	if f.deg() <= 0 {
		return nil
	}
	var out []rvP
	fp := rvPDeriv(f)
	a := rvPGcd(f, fp)
	b, _ := rvPDivMod(f, a)
	c, _ := rvPDivMod(fp, a)
	d := rvPSub(c, rvPDeriv(b))
	for b.deg() > 0 {
		a = rvPGcd(b, d)
		out = append(out, a)
		b, _ = rvPDivMod(b, a)
		c, _ = rvPDivMod(d, a)
		d = rvPSub(c, rvPDeriv(b))
	}
	return out
}

// rvPInt returns p times a positive rational with coprime integer
// coefficients: same roots and signs.
func rvPInt(p rvP) rvP {
	p = p.trim()
	l, g := big.NewInt(1), new(big.Int)
	for _, c := range p {
		d := c.Denom()
		g.GCD(nil, nil, l, d)
		l.Mul(l, new(big.Int).Quo(d, g))
	}
	out := make(rvP, len(p))
	g.SetInt64(0)
	for i, c := range p {
		v := new(big.Int).Mul(c.Num(), new(big.Int).Quo(l, c.Denom()))
		out[i] = new(big.Rat).SetInt(v)
		g.GCD(nil, nil, g, new(big.Int).Abs(v))
	}
	if g.Sign() > 0 {
		for _, c := range out {
			c.SetInt(new(big.Int).Quo(c.Num(), g))
		}
	}
	return out
}

// rvSign returns the sign of p(x) for p with integer coefficients
// (rvPInt), by homogeneous Horner in big.Int.
func rvSign(p rvP, x *big.Rat) int {
	a, b := x.Num(), x.Denom()
	acc, bp, t := new(big.Int), big.NewInt(1), new(big.Int)
	for _, v := range slices.Backward(p) {
		acc.Mul(acc, a)
		acc.Add(acc, t.Mul(v.Num(), bp))
		bp.Mul(bp, b)
	}
	return acc.Sign()
}

func rvSturmSeq(f rvP) []rvP {
	s := []rvP{rvPInt(f), rvPInt(rvPDeriv(f))}
	for s[len(s)-1].deg() > 0 {
		_, r := rvPDivMod(s[len(s)-2], s[len(s)-1])
		if len(r) == 0 {
			break
		}
		s = append(s, rvPInt(rvPNeg(r)))
	}
	return s
}
func rvSturmV(s []rvP, x *big.Rat) int {
	v, prev := 0, 0
	for _, p := range s {
		sg := rvSign(p, x)
		if sg == 0 {
			continue
		}
		if prev != 0 && sg != prev {
			v++
		}
		prev = sg
	}
	return v
}

// roots in (lo, hi] of square-free f
func rvSturmCount(s []rvP, lo, hi *big.Rat) int { return rvSturmV(s, lo) - rvSturmV(s, hi) }

type rvIval struct {
	lo, hi *big.Rat
	exact  bool
}

func rvCauchy(f rvP) *big.Rat {
	f = f.trim()
	n := len(f) - 1
	m := new(big.Rat)
	for i := range n {
		t := new(big.Rat).Quo(f[i], f[n])
		t.Abs(t)
		if t.Cmp(m) > 0 {
			m = t
		}
	}
	return m.Add(m, rvRi(1))
}

func rvRealRoots(f rvP) []rvIval {
	f = f.trim()
	if f.deg() <= 0 {
		return nil
	}
	f = rvPInt(f)
	s := rvSturmSeq(f)
	B := rvCauchy(f)
	var out []rvIval
	var rec func(lo, hi *big.Rat, cnt, depth int)
	rec = func(lo, hi *big.Rat, cnt, depth int) {
		if cnt == 0 {
			return
		}
		if cnt == 1 || depth > 3000 {
			out = append(out, rvRefine(f, lo, hi))
			return
		}
		mid := new(big.Rat).Add(lo, hi)
		mid.Quo(mid, rvRi(2))
		c1 := rvSturmCount(s, lo, mid)
		rec(lo, mid, c1, depth+1)
		rec(mid, hi, cnt-c1, depth+1)
	}
	lo := new(big.Rat).Neg(B)
	rec(lo, B, rvSturmCount(s, lo, B), 0)
	return out
}

// rvRefine: single rvRoot in (lo, hi].
func rvRefine(f rvP, lo, hi *big.Rat) rvIval {
	lo, hi = new(big.Rat).Set(lo), new(big.Rat).Set(hi)
	if rvSign(f, hi) == 0 {
		return rvIval{hi, hi, true}
	}
	sh := rvSign(f, hi)
	two := rvRi(2)
	for it := range 600 {
		w := new(big.Rat).Sub(hi, lo)
		m := new(big.Rat).Abs(lo)
		if a := new(big.Rat).Abs(hi); a.Cmp(m) > 0 {
			m = a
		}
		lim := new(big.Rat).Mul(m, big.NewRat(1, 1<<62))
		lim.Mul(lim, big.NewRat(1, 1<<40))
		if w.Cmp(lim) <= 0 && it > 0 {
			break
		}
		mid := new(big.Rat).Add(lo, hi)
		mid.Quo(mid, two)
		sm := rvSign(f, mid)
		if sm == 0 {
			return rvIval{mid, mid, true}
		}
		if sm == sh {
			hi = mid
		} else {
			lo = mid
		}
	}
	return rvIval{lo, hi, false}
}

func rvMidF(iv rvIval) float64 {
	m := new(big.Rat).Add(iv.lo, iv.hi)
	m.Quo(m, rvRi(2))
	f, _ := m.Float64()
	return f
}
func rvMidR(iv rvIval) *big.Rat {
	m := new(big.Rat).Add(iv.lo, iv.hi)
	return m.Quo(m, rvRi(2))
}

// rvHasRootIn reports whether g has a rvRoot in iv (closed).
func rvHasRootIn(g rvP, iv rvIval) bool {
	g = g.trim()
	if g.deg() <= 0 {
		return false
	}
	if iv.exact {
		return rvPEval(g, iv.lo).Sign() == 0
	}
	if rvPEval(g, iv.lo).Sign() == 0 || rvPEval(g, iv.hi).Sign() == 0 {
		return true
	}
	s := rvSturmSeq(rvPMonic(g))
	return rvSturmCount(s, iv.lo, iv.hi) > 0
}

// numeric roots (Aberth) of float coefficients ascending.
func rvNumRoots(c []float64) []complex128 {
	n := len(c)
	for n > 0 && c[n-1] == 0 {
		n--
	}
	c = c[:n]
	deg := n - 1
	if deg < 1 {
		return nil
	}
	lead := c[deg]
	a := make([]complex128, deg+1)
	for i := range a {
		a[i] = complex(c[i]/lead, 0)
	}
	R := 0.0
	for i := range deg {
		R = math.Max(R, math.Pow(cmplx.Abs(a[i]), 1/float64(deg-i)))
	}
	if R == 0 {
		R = 1
	}
	z := make([]complex128, deg)
	for k := range z {
		z[k] = cmplx.Rect(R, 2*math.Pi*float64(k)/float64(deg)+0.4)
	}
	ev := func(x complex128) (complex128, complex128) {
		p, dp := complex(0, 0), complex(0, 0)
		for i := deg; i >= 0; i-- {
			dp = dp*x + p
			p = p*x + a[i]
		}
		return p, dp
	}
	for range 500 {
		maxc := 0.0
		for k := range z {
			p, dp := ev(z[k])
			if p == 0 {
				continue
			}
			r := p / dp
			s := complex(0, 0)
			for j := range z {
				if j != k {
					s += 1 / (z[k] - z[j])
				}
			}
			w := r / (1 - r*s)
			z[k] -= w
			maxc = math.Max(maxc, cmplx.Abs(w)/math.Max(cmplx.Abs(z[k]), 1e-300))
		}
		if maxc < 1e-17 {
			break
		}
	}
	return z
}

// rvExactTF of float realization (n states, SISO), plus z^L delay factor.
func rvExactTF(a, b, c []float64, d float64, n, L int) (num, den rvP) {
	A := make([]*big.Rat, n*n)
	for i := range A {
		A[i] = rvRat(a[i])
	}
	M := make([]*big.Rat, n*n)
	for i := range M {
		M[i] = new(big.Rat)
	}
	for i := range n {
		M[i*n+i].SetInt64(1)
	}
	den = make(rvP, n+1)
	num = make(rvP, n+1)
	for i := range den {
		den[i] = new(big.Rat)
		num[i] = new(big.Rat)
	}
	den[n].SetInt64(1)
	B := make([]*big.Rat, n)
	C := make([]*big.Rat, n)
	for i := range n {
		B[i], C[i] = rvRat(b[i]), rvRat(c[i])
	}
	t := new(big.Rat)
	for k := 1; k <= n; k++ {
		// num coefficient s^{n-k} = C M_k B
		acc := new(big.Rat)
		for i := range n {
			if C[i].Sign() == 0 {
				continue
			}
			row := new(big.Rat)
			for j := range n {
				if B[j].Sign() == 0 || M[i*n+j].Sign() == 0 {
					continue
				}
				t.Mul(M[i*n+j], B[j])
				row.Add(row, t)
			}
			t.Mul(C[i], row)
			acc.Add(acc, t)
		}
		num[n-k] = acc
		AM := make([]*big.Rat, n*n)
		for i := range n {
			for j := range n {
				s := new(big.Rat)
				for l := range n {
					if A[i*n+l].Sign() == 0 || M[l*n+j].Sign() == 0 {
						continue
					}
					t.Mul(A[i*n+l], M[l*n+j])
					s.Add(s, t)
				}
				AM[i*n+j] = s
			}
		}
		tr := new(big.Rat)
		for i := range n {
			tr.Add(tr, AM[i*n+i])
		}
		ck := new(big.Rat).Quo(tr, rvRi(int64(k)))
		ck.Neg(ck)
		den[n-k] = ck
		for i := range n {
			AM[i*n+i].Add(AM[i*n+i], ck)
		}
		M = AM
	}
	num = rvPAdd(num, rvPScale(den, rvRat(d)))
	if L > 0 {
		zl := make(rvP, L+1)
		for i := range zl {
			zl[i] = new(big.Rat)
		}
		zl[L].SetInt64(1)
		den = rvPMul(den, zl)
	}
	return num.trim(), den.trim()
}

// rvSPDegree returns the degree of the exact numerator of C(sI−A)⁻¹B.
func rvSPDegree(a, b, c []float64, n, L int) int {
	num, _ := rvExactTF(a, b, c, 0, n, L)
	return num.deg()
}

// rvTrimmedTF is rvExactTF with the numerator of C(sI−A)⁻¹B cut to degree
// spDeg, the intended one: the float realization's higher coefficients
// are rounding noise, which RootLocus trims by the relative degree.
func rvTrimmedTF(a, b, c []float64, d float64, n, L, spDeg int) (num, den rvP) {
	sp, den := rvExactTF(a, b, c, 0, n, L)
	if len(sp) > spDeg+1 {
		sp = sp[:max(spDeg+1, 0)]
	}
	return rvPAdd(sp, rvPScale(den[L:], rvRat(d))).trim(), den
}

type rvOPt struct {
	X     float64
	Kind  string // pole, zero, stat
	Mult  int    // multiplicity in G (pole/zero) or of G-K (stat q-mult+1)
	Exact bool
}

type rvOracle struct {
	Cancel   bool
	Const    bool
	Pts      []rvOPt // accepted (K>0 or multiple pole/zero)
	Rej      []rvOPt // K<0 real stationary points
	NA       int
	Sign     int // sign(lead N / lead D)
	Centroid float64
	R        float64
}

func rvFujiwara(p rvP) float64 {
	f := rvPFloat(p.trim())
	n := len(f) - 1
	if n < 1 {
		return 0
	}
	R := 0.0
	for i := range n {
		R = math.Max(R, math.Pow(math.Abs(f[i]/f[n]), 1/float64(n-i)))
	}
	return R
}

func rvBreakOracle(N, Dn rvP) rvOracle {
	N, Dn = N.trim(), Dn.trim()
	var o rvOracle
	o.R = math.Max(rvFujiwara(N), rvFujiwara(Dn))
	if o.R == 0 {
		o.R = 1
	}
	if len(N) == 0 {
		o.Const = true
		return o
	}
	o.NA = Dn.deg() - N.deg()
	sg := N[len(N)-1].Sign() * Dn[len(Dn)-1].Sign()
	o.Sign = sg
	if o.NA > 0 {
		sp := new(big.Rat).Quo(Dn[len(Dn)-2], Dn[len(Dn)-1])
		sp.Neg(sp)
		sz := new(big.Rat)
		if len(N) >= 2 {
			sz.Quo(N[len(N)-2], N[len(N)-1])
			sz.Neg(sz)
		}
		c := new(big.Rat).Sub(sp, sz)
		c.Quo(c, rvRi(int64(o.NA)))
		o.Centroid, _ = c.Float64()
	}
	if g := rvPGcd(N, Dn); g.deg() > 0 {
		o.Cancel = true
		return o
	}
	q := rvPSub(rvPMul(rvPDeriv(N), Dn), rvPMul(N, rvPDeriv(Dn)))
	if q.deg() < 0 {
		o.Const = true
		return o
	}
	if q.deg() == 0 {
		return o
	}
	for i, f := range rvYun(q) {
		qm := i + 1
		for _, iv := range rvRealRoots(f) {
			pt := rvOPt{X: rvMidF(iv), Mult: qm + 1, Exact: iv.exact}
			switch {
			case rvHasRootIn(rvPGcd(f, Dn), iv):
				pt.Kind = "pole"
				o.Pts = append(o.Pts, pt)
			case rvHasRootIn(rvPGcd(f, N), iv):
				pt.Kind = "zero"
				o.Pts = append(o.Pts, pt)
			default:
				pt.Kind = "stat"
				x := rvMidR(iv)
				s := -rvPEval(Dn, x).Sign() * rvPEval(N, x).Sign()
				if s > 0 {
					o.Pts = append(o.Pts, pt)
				} else {
					o.Rej = append(o.Rej, pt)
				}
			}
		}
	}
	return o
}

func (o rvOracle) tol(p rvOPt) float64 {
	ax := math.Abs(p.X)
	t := 1e-5*ax + 1e-9*o.R
	if p.Mult >= 2 {
		t += 1e2 * math.Pow(2.2e-16, 1/float64(p.Mult)) * math.Max(ax, o.R)
	}
	return t
}

// match: reported (real parts, imag must be small) vs oracle points.
// returns ok, and description of missing/extra.
func (o rvOracle) match(rep []complex128) (ok bool, missing []rvOPt, extra []complex128) {
	used := make([]bool, len(rep))
	for _, p := range o.Pts {
		best, bd := -1, math.Inf(1)
		for j, r := range rep {
			if used[j] {
				continue
			}
			d := math.Hypot(real(r)-p.X, imag(r))
			if d < bd {
				bd, best = d, j
			}
		}
		if best >= 0 && bd <= o.tol(p) {
			used[best] = true
		} else {
			missing = append(missing, p)
		}
	}
	for j, r := range rep {
		if !used[j] {
			extra = append(extra, r)
		}
	}
	return len(missing) == 0 && len(extra) == 0, missing, extra
}

// rvSamePts: compare two oracles' accepted sets (positions within tol of a).
func rvSamePts(a, b rvOracle, far float64) bool {
	var bp []complex128
	for _, p := range b.Pts {
		if math.Abs(p.X) <= far {
			bp = append(bp, complex(p.X, 0))
		}
	}
	var aa rvOracle = a
	aa.Pts = nil
	for _, p := range a.Pts {
		if math.Abs(p.X) <= far {
			aa.Pts = append(aa.Pts, p)
		}
	}
	ok, _, _ := aa.match(bp)
	return ok
}

type rvOut struct {
	ID     int
	Err    string
	Bk     [][2]float64
	Asym   []float64
	Cen    float64
	CenOK  bool
	Dep    []float64
	Arr    []float64
	Poles  [][2]float64
	Zeros  [][2]float64
	GHash  string
	BHash  string
	NGains int
}

func rvCx(v [][2]float64) []complex128 {
	o := make([]complex128, len(v))
	for i, x := range v {
		o[i] = complex(x[0], x[1])
	}
	return o
}

type rvOra struct {
	I, A rvOracle
	// Differs marks a float TF whose break set is not the intended one;
	// the components judged by the intended TF are then not enforced.
	Differs bool
	// Prec marks a break set of the float TF that a 1-ulp change of A,
	// B, C changes.
	Prec    bool
	PrecWhy string
	IPoles  []complex128
	IZeros  []complex128
	Phase0  float64
	DegN    int
}

func rvPerturb(r *rand.Rand, v []float64) []float64 {
	o := append([]float64{}, v...)
	for i, x := range o {
		if x == 0 {
			continue
		}
		if r.IntN(2) == 0 {
			o[i] = math.Nextafter(x, math.Inf(1))
		} else {
			o[i] = math.Nextafter(x, math.Inf(-1))
		}
	}
	return o
}

func rvOracleFor(c rvCase) rvOra {
	L := c.InDelay + c.OutDelay
	var o rvOra
	Ni, Di := rvExactTF(c.A0, c.B0, c.C0, c.D, c.N, L)
	o.I = rvBreakOracle(Ni, Di)
	spDeg := rvSPDegree(c.A0, c.B0, c.C0, c.N, L)
	Na, Da := rvTrimmedTF(c.A, c.B, c.C, c.D, c.N, L, spDeg)
	o.A = rvBreakOracle(Na, Da)
	far := 1e6 * o.I.R
	if !o.A.Cancel && !o.A.Const {
		if !o.I.Cancel && !o.I.Const && (!rvSamePts(o.I, o.A, far) || !rvSamePts(o.A, o.I, far)) {
			o.Differs = true
		}
		r := rand.New(rand.NewPCG(uint64(c.ID), 77))
		for k := 0; k < 2 && !o.Prec; k++ {
			Np, Dp := rvTrimmedTF(rvPerturb(r, c.A), rvPerturb(r, c.B), rvPerturb(r, c.C), c.D, c.N, L, spDeg)
			op := rvBreakOracle(Np, Dp)
			if op.Cancel || !rvSamePts(o.A, op, far) || !rvSamePts(op, o.A, far) {
				o.Prec, o.PrecWhy = true, "ulp"
			}
		}
	}
	o.IPoles = rvNumRoots(rvPFloat(Di))
	o.IZeros = rvNumRoots(rvPFloat(Ni))
	o.DegN = Ni.deg()
	o.Phase0 = math.Pi
	if o.I.Sign < 0 {
		o.Phase0 = 0
	}
	return o
}

func rvAngDiff(a, b float64) float64 {
	return math.Abs(math.Remainder(a-b, 2*math.Pi))
}

func rvNearest(x complex128, set []complex128) (int, float64) {
	bi, bd := -1, math.Inf(1)
	for i, y := range set {
		if d := cmplx.Abs(x - y); d < bd {
			bi, bd = i, d
		}
	}
	return bi, bd
}

func rvSeparated(x complex128, idx int, poles, zeros []complex128, isPole bool, R float64) bool {
	th := 1e-2 * math.Max(cmplx.Abs(x), 1e-3*R)
	for j, p := range poles {
		if isPole && j == idx {
			continue
		}
		if cmplx.Abs(x-p) < th {
			return false
		}
	}
	for j, z := range zeros {
		if !isPole && j == idx {
			continue
		}
		if cmplx.Abs(x-z) < th {
			return false
		}
	}
	return true
}

type rvScore struct {
	Err                string
	Bk, As, Ce, Dp, Ar bool
	Miss               []rvOPt
	Extra              []complex128
	Why                []string
}

func rvScoreCase(c rvCase, o rvOra, r rvOut) rvScore {
	var s rvScore
	if r.Err != "" {
		s.Err = r.Err
		return s
	}
	if o.A.Cancel {
		s.Bk = true
	} else {
		s.Bk, s.Miss, s.Extra = o.A.match(rvCx(r.Bk))
		if !s.Bk {
			// RootLocus cuts the numerator by a float Markov test, which may
			// keep rounding-level coefficients the intended TF has as zero.
			L := c.InDelay + c.OutDelay
			for deg := rvSPDegree(c.A0, c.B0, c.C0, c.N, L) + 1; deg < c.N && !s.Bk; deg++ {
				if a := rvBreakOracle(rvTrimmedTF(c.A, c.B, c.C, c.D, c.N, L, deg)); !a.Cancel {
					s.Bk, _, _ = a.match(rvCx(r.Bk))
				}
			}
		}
	}
	na := o.I.NA
	s.As, s.Ce = true, true
	if na <= 0 {
		s.As = len(r.Asym) == 0
	} else if len(r.Asym) != na {
		s.As = false
		s.Why = append(s.Why, fmt.Sprintf("nasym %d want %d", len(r.Asym), na))
	} else {
		for k, a := range r.Asym {
			if rvAngDiff(a, (2*float64(k)*math.Pi+o.Phase0)/float64(na)) > 1e-9 {
				s.As = false
				s.Why = append(s.Why, fmt.Sprintf("asym[%d]=%g", k, a))
				break
			}
		}
		if !r.CenOK || math.Abs(r.Cen-o.I.Centroid) > 1e-6*(o.I.R+math.Abs(o.I.Centroid)) {
			s.Ce = false
			s.Why = append(s.Why, fmt.Sprintf("cen %g want %g", r.Cen, o.I.Centroid))
		}
	}
	R := o.I.R
	ip, iz := o.IPoles, o.IZeros
	s.Dp = true
	rp := rvCx(r.Poles)
	if len(r.Dep) != len(rp) {
		s.Dp = false
		s.Why = append(s.Why, "ndep")
	} else {
		for i, p := range rp {
			j, d := rvNearest(p, ip)
			if j < 0 || d > 1e-6*math.Max(cmplx.Abs(ip[j]), R) || !rvSeparated(ip[j], j, ip, iz, true, R) {
				continue
			}
			want := o.Phase0
			for k, q := range ip {
				if k != j {
					want -= cmplx.Phase(ip[j] - q)
				}
			}
			for _, z := range iz {
				want += cmplx.Phase(ip[j] - z)
			}
			if rvAngDiff(r.Dep[i], want) > 1e-4 {
				s.Dp = false
				s.Why = append(s.Why, fmt.Sprintf("dep[%d] %g want %g at %v", i, r.Dep[i], want, ip[j]))
				break
			}
		}
	}
	s.Ar = true
	rz := rvCx(r.Zeros)
	if len(r.Arr) != o.DegN {
		s.Ar = false
		s.Why = append(s.Why, fmt.Sprintf("narr %d want %d", len(r.Arr), o.DegN))
	} else if len(rz) >= len(r.Arr) {
		ex := len(rz) - len(r.Arr)
		idx := make([]int, len(rz))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return cmplx.Abs(rz[idx[a]]) > cmplx.Abs(rz[idx[b]]) })
		drop := map[int]bool{}
		for _, i := range idx[:ex] {
			drop[i] = true
		}
		var kept []complex128
		for i, z := range rz {
			if !drop[i] {
				kept = append(kept, z)
			}
		}
		for i, z := range kept {
			j, d := rvNearest(z, iz)
			if j < 0 || d > 1e-6*math.Max(cmplx.Abs(iz[j]), R) || !rvSeparated(iz[j], j, ip, iz, false, R) {
				continue
			}
			want := o.Phase0
			for k, q := range iz {
				if k != j {
					want -= cmplx.Phase(iz[j] - q)
				}
			}
			for _, p := range ip {
				want += cmplx.Phase(iz[j] - p)
			}
			if rvAngDiff(r.Arr[i], want) > 1e-4 {
				s.Ar = false
				s.Why = append(s.Why, fmt.Sprintf("arr[%d] %g want %g", i, r.Arr[i], want))
				break
			}
		}
	}
	return s
}

func (s rvScore) ok() bool { return s.Err == "" && s.Bk && s.As && s.Ce && s.Dp && s.Ar }
