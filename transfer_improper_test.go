package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func ratPolyDiv(num, den []float64) (q, r []*big.Rat) {
	rem := make([]*big.Rat, len(num))
	for i, v := range num {
		rem[i] = new(big.Rat).SetFloat64(v)
	}
	d := len(den) - 1
	lead := new(big.Rat).SetFloat64(den[0])
	for k := 0; k+d < len(num); k++ {
		c := new(big.Rat).Quo(rem[k], lead)
		q = append(q, c)
		for l := 1; l <= d; l++ {
			t := new(big.Rat).Mul(c, new(big.Rat).SetFloat64(den[l]))
			rem[k+l].Sub(rem[k+l], t)
		}
	}
	return q, rem[len(q):]
}

func ratHorner(p []float64, s *big.Rat) *big.Rat {
	acc := new(big.Rat)
	for _, c := range p {
		acc.Mul(acc, s)
		acc.Add(acc, new(big.Rat).SetFloat64(c))
	}
	return acc
}

func ratF(r *big.Rat) float64 { f, _ := r.Float64(); return f }

func hornerC(p []float64, s complex128) complex128 {
	var acc complex128
	for _, c := range p {
		acc = acc*s + complex(c, 0)
	}
	return acc
}

func TestTFStateSpaceImproperMatchesMATLAB(t *testing.T) {
	num, den := []float64{7, 5, 3}, []float64{2, 1}
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := res.Sys
	if n, m, p := sys.Dims(); n != 3 || m != 1 || p != 1 || sys.E == nil || res.MinimalOrder != 3 {
		t.Fatalf("dims %d,%d,%d E=%v order=%d", n, m, p, sys.E, res.MinimalOrder)
	}
	if proper, err := sys.IsProper(); err != nil || proper {
		t.Fatalf("IsProper = %v, %v", proper, err)
	}
	q, r := ratPolyDiv(num, den)
	if got, want := sys.D.At(0, 0), ratF(q[1]); got != want {
		t.Fatalf("D = %v, want %v", got, want)
	}
	if got, want := sys.B.At(2, 0), -ratF(q[0]); got != want {
		t.Fatalf("B chain = %v, want %v", got, want)
	}
	if got, want := sys.B.At(0, 0)*sys.C.At(0, 0), ratF(new(big.Rat).Quo(r[0], new(big.Rat).SetFloat64(den[0]))); math.Abs(got-want) > 4e-16*math.Abs(want) {
		t.Fatalf("remainder residue = %v, want %v", got, want)
	}
	poles, err := sys.Poles()
	if err != nil || len(poles) != 1 || cmplx.Abs(poles[0]+0.5) > 1e-14 {
		t.Fatalf("poles = %v, %v", poles, err)
	}
	omega := []float64{0.01, 0.4, 3, 70, 1e4}
	h, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		s := complex(0, w)
		want := hornerC(num, s) / hornerC(den, s)
		if got := h.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12*cmplx.Abs(want) {
			t.Fatalf("H(j%g) = %v, want %v", w, got, want)
		}
	}
}

func TestTFStateSpaceImproperMIMOAndMetadata(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{
			{{1, -2, 0, 3}, {4, 1}, {0, 0, 2, 5}},
			{{-3, 0, 1}, {2}, {1, 1, 1, 1}},
			{{1}, {0.5, -1}, {2, 0, 0}},
		},
		Den:        [][]float64{{2, 3, -1}, {1}, {1, 4}},
		Delay:      [][]float64{{0, 0.1, 0}, {0.2, 0, 0}, {0, 0, 0.3}},
		InputName:  []string{"u1", "u2", "u3"},
		OutputName: []string{"y1", "y2", "y3"},
	}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := res.Sys
	if n, _, _ := sys.Dims(); n != 3+2+4+2 || res.MinimalOrder != n || sys.E == nil {
		t.Fatalf("n = %d order = %d E = %v", n, res.MinimalOrder, sys.E)
	}
	if sys.Delay == nil || sys.Delay.At(2, 2) != 0.3 || sys.InputName[1] != "u2" || sys.OutputName[2] != "y3" {
		t.Fatalf("metadata lost: delay=%v in=%v out=%v", sys.Delay, sys.InputName, sys.OutputName)
	}
	rational := *sys
	rational.Delay = nil
	for _, s := range []complex128{0.3i, 1.7 + 0.2i, 25i, -0.7 + 3i} {
		got, err := rational.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range tf.Num {
			for j := range tf.Num[i] {
				want := hornerC(tf.Num[i][j], s) / hornerC(tf.Den[i], s)
				if d := cmplx.Abs(got[i][j] - want); d > 1e-12*math.Max(1, cmplx.Abs(want)) {
					t.Fatalf("H[%d][%d](%v) = %v, want %v", i, j, s, got[i][j], want)
				}
			}
		}
	}
}

func TestTFStateSpaceImproperDiscrete(t *testing.T) {
	num, den, dt := []float64{1, 0, -0.3}, []float64{1, -0.5}, 0.1
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if res.Sys.E == nil || res.Sys.Dt != dt {
		t.Fatalf("E=%v Dt=%v", res.Sys.E, res.Sys.Dt)
	}
	for _, z := range []complex128{0.2 + 0.9i, -1.3, 2 + 1i} {
		got, err := res.Sys.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		want := hornerC(num, z) / hornerC(den, z)
		if cmplx.Abs(got[0][0]-want) > 1e-13*cmplx.Abs(want) {
			t.Fatalf("H(%v) = %v, want %v", z, got[0][0], want)
		}
	}
}

func TestTFStateSpaceLeadingZeroNumeratorStaysExplicit(t *testing.T) {
	padded := &TransferFunc{Num: [][][]float64{{{0, 0, 2, 1}}}, Den: [][]float64{{1, 3}}}
	trimmed := &TransferFunc{Num: [][][]float64{{{2, 1}}}, Den: [][]float64{{1, 3}}}
	a, err := padded.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	b, err := trimmed.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if a.Sys.E != nil || !mat.Equal(a.Sys.A, b.Sys.A) || !mat.Equal(a.Sys.B, b.Sys.B) || !mat.Equal(a.Sys.C, b.Sys.C) || !mat.Equal(a.Sys.D, b.Sys.D) {
		t.Fatalf("padded realization differs: E=%v", a.Sys.E)
	}
}

// Seeded improper sweep against exact big.Rat evaluation of num/den at
// rational real points away from the denominator roots.
func TestTFStateSpaceImproperSweepExactOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(20, 5))
	coef := func() float64 { return float64(rng.IntN(19) - 9) }
	poly := func(deg int) []float64 {
		c := make([]float64, deg+1)
		for k := range c {
			c[k] = coef()
		}
		for c[0] == 0 {
			c[0] = coef()
		}
		return c
	}
	points := []*big.Rat{big.NewRat(3, 10), big.NewRat(-17, 7), big.NewRat(41, 9), big.NewRat(-1, 3), big.NewRat(11, 2)}
	for trial := range 300 {
		p, m := 1+rng.IntN(3), 1+rng.IntN(3)
		tf := &TransferFunc{Num: make([][][]float64, p), Den: make([][]float64, p)}
		for i := range p {
			tf.Den[i] = poly(rng.IntN(4))
			tf.Num[i] = make([][]float64, m)
			for j := range m {
				tf.Num[i][j] = poly(rng.IntN(len(tf.Den[i]) + 3))
			}
		}
		if tf.isProper() {
			tf.Num[0][0] = poly(len(tf.Den[0]))
		}
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		for _, s := range points {
			sf := ratF(s)
			got, err := res.Sys.EvalFr(complex(sf, 0))
			if err != nil {
				t.Fatalf("trial %d: %v", trial, err)
			}
			for i := range p {
				dv := ratHorner(tf.Den[i], s)
				scale := 0.0
				for k, c := range tf.Den[i] {
					scale += math.Abs(c) * math.Pow(math.Abs(sf), float64(len(tf.Den[i])-1-k))
				}
				if math.Abs(ratF(dv)) < 0.05*scale {
					continue
				}
				for j := range m {
					want := ratF(new(big.Rat).Quo(ratHorner(tf.Num[i][j], s), dv))
					if d := cmplx.Abs(got[i][j] - complex(want, 0)); d > 1e-10*math.Max(1, math.Abs(want)) {
						t.Fatalf("trial %d H[%d][%d](%v) = %v, want %v (num %v den %v)", trial, i, j, sf, got[i][j], want, tf.Num[i][j], tf.Den[i])
					}
				}
			}
		}
	}
}

func TestZPKStateSpaceImproperIsDescriptor(t *testing.T) {
	z, err := NewZPK([]complex128{-1, -3}, []complex128{-2}, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := z.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	s := complex(0.5, 1.5)
	got, err := res.Sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	want := 4 * (s + 1) * (s + 3) / (s + 2)
	if res.Sys.E == nil || cmplx.Abs(got[0][0]-want) > 1e-13*cmplx.Abs(want) {
		t.Fatalf("E=%v H(%v) = %v, want %v", res.Sys.E, s, got[0][0], want)
	}
}

// requireImproperDescriptorTF checks an improper descriptor's TransferFunction
// (MATLAB tf(dss) keeps the polynomial part): some numerator degree exceeds
// its row denominator's, and the tf matches the pencil C(sE−A)⁻¹B + D.
func requireImproperDescriptorTF(t *testing.T, label string, sys *System, res *TransferFuncResult, err error, tol float64) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: TransferFunction: %v", label, err)
	}
	if res.TF.isProper() {
		t.Fatalf("%s: tf %v / %v is proper", label, res.TF.Num, res.TF.Den)
	}
	for _, s := range []complex128{1i, complex(0.5, 3), 20i, 1e3i} {
		got, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want := pencilResponse(t, sys, s)
		for i := range want {
			for j := range want[i] {
				if d := cmplx.Abs(got[i][j] - want[i][j]); !(d <= tol*math.Max(1, cmplx.Abs(want[i][j]))) {
					t.Fatalf("%s: G[%d][%d](%v) = %v, pencil %v", label, i, j, s, got[i][j], want[i][j])
				}
			}
		}
	}
}

// SC36CE: MATLAB tf(ss(tf([7 5 3],[2 1]))) is (3.5s^2+2.5s+1.5)/(s+0.5) and
// zpk of it keeps both zeros; base rejected the descriptor realization.
func TestTransferFunctionImproperDescriptorMatchesMATLAB(t *testing.T) {
	cases := []struct {
		num, den, wantNum, wantDen []float64
		dt                         float64
	}{
		{[]float64{7, 5, 3}, []float64{2, 1}, []float64{3.5, 2.5, 1.5}, []float64{1, 0.5}, 0},
		{[]float64{1, 2}, []float64{1}, []float64{1, 2}, []float64{1}, 0},
		{[]float64{2, 0, 0, 1}, []float64{4, -2}, []float64{0.5, 0, 0, 0.25}, []float64{1, -0.5}, 0},
		{[]float64{1, 0, -0.3}, []float64{1, -0.5}, []float64{1, 0, -0.3}, []float64{1, -0.5}, 0.1},
	}
	for _, c := range cases {
		ss, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := ss.Sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%v/%v: %v", c.num, c.den, err)
		}
		got := res.TF
		if len(got.Num[0][0]) != len(c.wantNum) || len(got.Den[0]) != len(c.wantDen) || got.Dt != c.dt {
			t.Fatalf("%v/%v: tf %v/%v dt %v, want %v/%v", c.num, c.den, got.Num[0][0], got.Den[0], got.Dt, c.wantNum, c.wantDen)
		}
		for k, w := range c.wantNum {
			if math.Abs(got.Num[0][0][k]-w) > 1e-14*math.Max(1, math.Abs(w)) {
				t.Fatalf("%v/%v: num %v, want %v", c.num, c.den, got.Num[0][0], c.wantNum)
			}
		}
		for k, w := range c.wantDen {
			if math.Abs(got.Den[0][k]-w) > 1e-14*math.Max(1, math.Abs(w)) {
				t.Fatalf("%v/%v: den %v, want %v", c.num, c.den, got.Den[0], c.wantDen)
			}
		}
		z, err := ss.Sys.ZPKModel(nil)
		if err != nil {
			t.Fatalf("%v/%v: ZPKModel: %v", c.num, c.den, err)
		}
		if nz, np := len(z.ZPK.Zeros[0][0]), len(z.ZPK.Poles[0][0]); nz != len(c.num)-1 || np != len(c.den)-1 {
			t.Fatalf("%v/%v: zpk zeros %v poles %v", c.num, c.den, z.ZPK.Zeros[0][0], z.ZPK.Poles[0][0])
		}
		if g, w := z.ZPK.Gain[0][0], c.num[0]/c.den[0]; math.Abs(g-w) > 1e-14*math.Abs(w) {
			t.Fatalf("%v/%v: zpk gain %v, want %v", c.num, c.den, g, w)
		}
	}
	pd := idealPDDescriptor(t, 2, 0.5)
	res, err := pd.TransferFunction(nil)
	requireImproperDescriptorTF(t, "ideal PD", pd, res, err, 1e-14)
	if n := res.TF.Num[0][0]; len(n) != 2 || n[0] != 0.5 || n[1] != 2 || len(res.TF.Den[0]) != 1 || res.TF.Den[0][0] != 1 {
		t.Fatalf("ideal PD tf %v/%v, want (0.5s+2)/1", n, res.TF.Den[0])
	}
}

// SC36CE: tf -> ss -> tf on a seeded improper SISO/MIMO sweep, continuous
// and discrete. Each channel keeps its relative degree and evaluates to the
// exact big.Rat num/den.
func TestTransferFunctionImproperRoundTripExactOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(36, 0xce))
	coef := func() float64 { return float64(rng.IntN(19) - 9) }
	poly := func(deg int) []float64 {
		c := make([]float64, deg+1)
		for k := range c {
			c[k] = coef()
		}
		for c[0] == 0 {
			c[0] = coef()
		}
		return c
	}
	points := []*big.Rat{big.NewRat(3, 10), big.NewRat(-17, 7), big.NewRat(41, 9), big.NewRat(-1, 3), big.NewRat(11, 2)}
	for trial := range 400 {
		p, m := 1+rng.IntN(3), 1+rng.IntN(3)
		tf := &TransferFunc{Num: make([][][]float64, p), Den: make([][]float64, p)}
		if trial%4 == 3 {
			tf.Dt = 0.1
		}
		for i := range p {
			tf.Den[i] = poly(rng.IntN(4))
			tf.Num[i] = make([][]float64, m)
			for j := range m {
				tf.Num[i][j] = poly(rng.IntN(len(tf.Den[i]) + 3))
			}
		}
		if tf.isProper() {
			tf.Num[0][0] = poly(len(tf.Den[0]))
		}
		ss, err := tf.StateSpace()
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		res, err := ss.Sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("trial %d: TransferFunction: %v (tf %v / %v)", trial, err, tf.Num, tf.Den)
		}
		got := res.TF
		if got.Dt != tf.Dt {
			t.Fatalf("trial %d: Dt %v", trial, got.Dt)
		}
		for i := range p {
			for j := range m {
				if rel, want := len(got.Num[i][j])-len(got.Den[i]), len(tf.Num[i][j])-len(tf.Den[i]); rel != want && (want > 0 || rel > 0) {
					t.Fatalf("trial %d: H[%d][%d] = %v / %v, relative degree %d, want %d (in %v / %v)", trial, i, j, got.Num[i][j], got.Den[i], rel, want, tf.Num[i][j], tf.Den[i])
				}
			}
		}
		for _, s := range points {
			sf := ratF(s)
			for i := range p {
				dv := ratHorner(tf.Den[i], s)
				scale := 0.0
				for k, c := range tf.Den[i] {
					scale += math.Abs(c) * math.Pow(math.Abs(sf), float64(len(tf.Den[i])-1-k))
				}
				if math.Abs(ratF(dv)) < 0.05*scale {
					continue
				}
				gd := hornerC(got.Den[i], complex(sf, 0))
				for j := range m {
					want := ratF(new(big.Rat).Quo(ratHorner(tf.Num[i][j], s), dv))
					g := hornerC(got.Num[i][j], complex(sf, 0)) / gd
					if d := cmplx.Abs(g - complex(want, 0)); d > 1e-9*math.Max(1, math.Abs(want)) {
						t.Fatalf("trial %d H[%d][%d](%v) = %v, want %v (num %v den %v; got %v / %v)", trial, i, j, sf, g, want, tf.Num[i][j], tf.Den[i], got.Num[i][j], got.Den[i])
					}
				}
			}
		}
	}
}
