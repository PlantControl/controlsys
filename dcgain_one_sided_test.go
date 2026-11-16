package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dcGainPadded embeds an n0-state model at states shift… of an n-state one,
// the other states stable, decoupled and unreached.
func dcGainPadded(t *testing.T, n0, n, shift, m, p int, a, b, c []float64, dt float64) *System {
	t.Helper()
	A, B, C := make([]float64, n*n), make([]float64, n*m), make([]float64, p*n)
	for k := range n {
		A[k*n+k] = -1 - 0.1*float64(k)
		if dt > 0 {
			A[k*n+k] = 0.5 / float64(k+1)
		}
	}
	for i := range n0 {
		for j := range n0 {
			A[(i+shift)*n+j+shift] = a[i*n0+j]
		}
		copy(B[(i+shift)*m:(i+shift+1)*m], b[i*m:(i+1)*m])
		for o := range p {
			C[o*n+i+shift] = c[o*n0+i]
		}
	}
	sys, err := NewFromSlices(n, m, p, A, B, C, make([]float64, p*m), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func requireExactLimitNil(t *testing.T, name string, sys *System) *mat.Dense {
	t.Helper()
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("%s: DCGain: %v", name, err)
	}
	x0 := 0.0
	if sys.IsDiscrete() {
		x0 = 1
	}
	want, _ := exactDCGainLimitChecked(sys.A, sys.B, sys.C, sys.D, x0)
	gr, wr := g.RawMatrix().Data, want.RawMatrix().Data
	for k, w := range wr {
		if v := gr[k]; math.IsInf(w, 0) || w == 0 || math.IsInf(v, 0) {
			if v != w {
				t.Errorf("%s: entry %d = %v, exact limit %v", name, k, v, w)
			}
		} else if !(math.Abs(v-w) <= 1e-9*math.Abs(w)) {
			t.Errorf("%s: entry %d = %.17g, exact limit %.17g", name, k, v, w)
		}
	}
	return g
}

// An integrator coupled one way into a regular block (zero row: input-only
// integrator, zero column: no state reads it) has the 1/σ coefficient of a
// certified regular DC solve, so its Inf sign is proven at any order
// (ONYZO7, approved one-sided certificate); before it, the float Laurent
// reading sent the model to the exact solve at n ≤ 16 and returned
// ErrAccuracyUnverified above.
func TestDCGainOneSidedIntegratorCertifiedPastExactCap(t *testing.T) {
	cases := []struct {
		name    string
		dt      float64
		a, b, c []float64
	}{
		{"zero row, ct", 0,
			[]float64{-2667.425062386912, -1054.0540016254233, 0, 0},
			[]float64{-0.8458515296071365, 0.39592128934968418}, []float64{-0.62072159849106279, 0.54739684811667466}},
		{"zero row, dt, integrator first", 0.1,
			[]float64{1, 0, 0, 0.00095885762387342831, 0.99904114237612629, 5.2376664672140425e-17, 0.00096557154822828734, 2.0026851311837888, 0.97048040712303796},
			[]float64{0.031324398473193897, 1.5020220287690116e-05, 1.0103109940484665e-05}, []float64{0, 0, 1}},
		{"zero column, negative sign", 0,
			[]float64{-3, 1, 0, 2, -5, 0, 0.5, -1, 0},
			[]float64{1, -2, 0.25}, []float64{0, 0, -3}},
	}
	for _, c := range cases {
		n0 := len(c.b)
		for _, n := range []int{n0, n0 + 18} {
			sys := dcGainPadded(t, n0, n, n-n0, 1, 1, c.a, c.b, c.c, c.dt)
			g := requireExactLimitNil(t, c.name, sys)
			if v := g.At(0, 0); !math.IsInf(v, 0) {
				t.Errorf("%s n=%d: DCGain = %v, want an infinity", c.name, n, v)
			}
		}
	}
}

// A MIMO model with a zero-row integrator: the input that misses it (B_kj
// = 0) has the complementary block's finite gain, certified, beside the
// other input's proven infinity.
func TestDCGainOneSidedIntegratorFiniteEntryBesideInf(t *testing.T) {
	a := []float64{-4, 1, 3, 0.5, -2, -1, 0, 0, 0}
	b := []float64{1, 0.5, -1, 2, 2, 0}
	c := []float64{1, -1, 0.5, 0, 2, -1}
	for _, n := range []int{3, 21} {
		sys := dcGainPadded(t, 3, n, 0, 2, 2, a, b, c, 0)
		g := requireExactLimitNil(t, "MIMO", sys)
		if !math.IsInf(g.At(0, 0), 0) || math.IsInf(g.At(0, 1), 0) {
			t.Errorf("n=%d: DCGain = %v, want Inf in column 0 and finite in column 1", n, mat.Formatted(g))
		}
	}
}

// A bracket within its own rounding (a near-cancelling residue) or two
// integrator terms of opposite signs prove nothing by the certificate: past
// the exact check DCGain is the exact limit with nil only by another proof,
// otherwise ErrAccuracyUnverified.
func TestDCGainOneSidedIntegratorUnprovenPastExactCap(t *testing.T) {
	cases := []struct {
		name    string
		n0, m   int
		a, b, c []float64
		dt      float64
	}{
		{"cancelling bracket, zero column, dt", 3, 1,
			[]float64{0, -0.82303606596215528, 0, 1, 1.8150869958131495, 0, 0, 1, 1},
			[]float64{0.96812587861227395, -1.0116132737360233, 5.4707524664615272}, []float64{0, 0, 1}, 0.1},
		{"opposite-sign integrators", 3, 1,
			[]float64{-2, 1, 1, 0, 0, 0, 0, 0, 0},
			[]float64{1, 1, 1}, []float64{1, 1, -1}, 0},
	}
	for _, c := range cases {
		sys := dcGainPadded(t, c.n0, c.n0+18, 0, c.m, 1, c.a, c.b, c.c, c.dt)
		x0 := 0.0
		if c.dt > 0 {
			x0 = 1
		}
		if g, ok := sys.dcGainOneSidedCertified(x0); ok {
			t.Errorf("%s: certified %v", c.name, mat.Formatted(g))
		}
		if _, err := sys.DCGain(); err == nil {
			requireExactLimitNil(t, c.name, sys)
		} else if !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: DCGain error %v, want ErrAccuracyUnverified", c.name, err)
		}
	}
}
