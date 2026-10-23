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

func TestTFStateSpaceImproperMATLABRepro(t *testing.T) {
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
