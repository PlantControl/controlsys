package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func assertSweepMatchesPointwise(t *testing.T, label string, sys *System, omega []float64, relTol float64) {
	t.Helper()
	got, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	want, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponsePointwise: %v", label, err)
	}
	for k := range want.Data {
		g, w := got.Data[k], want.Data[k]
		if cmplx.IsInf(w) || cmplx.IsNaN(w) {
			if g != w && !(cmplx.IsInf(g) && cmplx.IsInf(w)) {
				t.Fatalf("%s: entry %d = %v, want %v", label, k, g, w)
			}
			continue
		}
		if d := cmplx.Abs(g - w); d > relTol*cmplx.Abs(w) {
			t.Fatalf("%s: entry %d (w=%g) rel diff %g", label, k, omega[k/(want.P*want.M)], d/cmplx.Abs(w))
		}
	}
}

func stiffChain(t *testing.T, n int, lo, hi, dt float64) *System {
	t.Helper()
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, 2, nil)
	C := mat.NewDense(2, n, nil)
	for i := range n {
		A.Set(i, i, -lo*math.Pow(hi/lo, float64(i)/float64(n-1)))
		if i+1 < n {
			A.Set(i, i+1, 0.3*float64(i%3+1))
		}
		B.Set(i, 0, 1)
		B.Set(i, 1, math.Sin(float64(i)))
		C.Set(0, i, math.Cos(float64(i)))
		C.Set(1, i, 1/float64(i+1))
	}
	sys, err := New(A, B, C, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dt == 0 {
		return sys
	}
	d, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// Regression for ELN5IG: the long-sweep transfer-function shortcut lost
// 5e-7 on this 18-state Padé cascade and O(1) on stiff chains.
func TestFreqResponseSweepPadeCascade(t *testing.T) {
	sys := absorbScopePlant(t, 0, false, false)
	absorbScopeCases[0].apply(t, sys, 0.125)
	pade, err := replaceContinuousDelays(sys, 5)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := pade.Dims(); n != 18 {
		t.Fatalf("n = %d, want 18", n)
	}
	assertSweepMatchesPointwise(t, "absorbScopeOmega", pade, absorbScopeOmega, 1e-12)
	assertSweepMatchesPointwise(t, "logspace", pade, logspace(-2, 3, 200), 1e-12)
}

func TestFreqResponseSweepStiffModels(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		lo, hi, dt float64
	}{
		{"n10", 10, 1e-2, 1e2, 0},
		{"n20", 20, 1e-3, 1e3, 0},
		{"n30", 30, 1e-3, 1e3, 0},
		{"n40", 40, 1e-2, 1e2, 0},
		{"discrete_n20", 20, 1e-3, 1e2, 0.01},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := stiffChain(t, tc.n, tc.lo, tc.hi, tc.dt)
			omega := logspace(-4, 4, 120)
			if tc.dt > 0 {
				omega = logspace(-4, math.Log10(math.Pi/tc.dt), 120)
			}
			assertSweepMatchesPointwise(t, tc.name, sys, omega, 1e-12)
		})
	}
}

func TestFreqResponseSweepResonanceAndEdges(t *testing.T) {
	zeta, wn := 1e-4, 7.0
	reson, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	omega := logspace(0, 1, 101)
	omega = append(omega, wn, wn*(1+zeta), wn*(1-zeta))
	assertSweepMatchesPointwise(t, "resonance", reson, omega, 1e-12)

	first, err := New(mat.NewDense(1, 1, []float64{-3}), mat.NewDense(1, 2, []float64{1, -2}),
		mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 2, []float64{0.5, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "n1", first, logspace(-2, 2, 50), 1e-12)

	static, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "n0", static, logspace(-2, 2, 10), 0)

	integ, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, -2, 1, 0, 0, -5}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "pole at w=0", integ, append([]float64{0}, logspace(-2, 2, 20)...), 1e-12)
}

func TestFreqResponseRejectsInvalidSystem(t *testing.T) {
	cases := map[string]func(*System){
		"Delay":       func(s *System) { s.Delay = mat.NewDense(3, 3, nil) },
		"InputDelay":  func(s *System) { s.InputDelay = []float64{1} },
		"OutputDelay": func(s *System) { s.OutputDelay = []float64{1, 2, 3} },
		"E":           func(s *System) { s.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1}) },
		"LFT.B2":      func(s *System) { s.LFT.B2 = mat.NewDense(2, 2, nil) },
		"C":           func(s *System) { s.C = mat.NewDense(2, 4, nil) },
	}
	for name, corrupt := range cases {
		sys := absorbScopePlant(t, 0, true, false)
		corrupt(sys)
		if _, err := sys.FreqResponse(absorbScopeOmega); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: FreqResponse err = %v", name, err)
		}
		if _, err := sys.FreqResponsePointwise(absorbScopeOmega); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: FreqResponsePointwise err = %v", name, err)
		}
		if _, err := sys.EvalFr(1i); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: EvalFr err = %v", name, err)
		}
	}
	static, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	static.LFT = &LFTDelay{Tau: []float64{0.5}, D12: mat.NewDense(1, 1, nil), D21: mat.NewDense(1, 1, nil), D22: mat.NewDense(1, 1, nil)}
	if _, err := static.DCGain(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("static LFT without B2: DCGain err = %v", err)
	}
	var nilSys *System
	if _, err := nilSys.FreqResponse(absorbScopeOmega); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("nil: FreqResponse err = %v", err)
	}
}
