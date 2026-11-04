package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var integratorExactMethods = []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse, C2DMethodTustin}

func integratorExactSystem(t *testing.T, n, m int, a, b []float64) *System {
	t.Helper()
	c := make([]float64, n)
	c[n-1] = 1
	sys, err := NewFromSlices(n, m, 1, a, b, c, make([]float64, m), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func integratorExactRow(t *testing.T, label string, ad *mat.Dense, i int, value float64) {
	t.Helper()
	_, n := ad.Dims()
	for j := range n {
		want := 0.0
		if j == i {
			want = value
		}
		if got := ad.At(i, j); got != want || math.Signbit(got) {
			t.Errorf("%s: [%d][%d] = %.17g, want %g", label, i, j, got, want)
		}
	}
}

func integratorExactColumn(t *testing.T, label string, ad *mat.Dense, j int) {
	t.Helper()
	n, _ := ad.Dims()
	for i := range n {
		want := 0.0
		if i == j {
			want = 1
		}
		if got := ad.At(i, j); got != want {
			t.Errorf("%s: Ad[%d][%d] = %.17g, want %g", label, i, j, got, want)
		}
	}
}

// For ZOH, FOH and impulse an input-driven integrator x_i' = b_i·u also has
// Bd row i = dt·b_i exactly.
func integratorExactInputRow(t *testing.T, label string, method C2DMethod, sys, d *System, i int, dt float64) {
	t.Helper()
	if method == C2DMethodTustin {
		return
	}
	_, m := sys.B.Dims()
	for k := range m {
		if got, want := d.B.At(i, k), dt*sys.B.At(i, k); got != want {
			t.Errorf("%s: Bd[%d][%d] = %.17g, want %.17g", label, i, k, got, want)
		}
	}
}

// RXOK37: 1/(s³+10s²+21s) and the integrator, lag, resonance pipeline of the
// 2TUAD6 sweep, whose C2D moved the integrator to 0.99999999999999989
// (ZOH) and 0.99999999999999978 (Tustin).
func TestC2DKeepsZeroRowIntegratorExact(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
		a, b []float64
		dt   float64
	}{
		{"cubic", 3, []float64{0, 0, -0, 1, 0, -21, 0, 1, -10}, []float64{1, 0, 0}, 0.1},
		{"lag and resonance", 4, []float64{
			-0, 0, 0, 0,
			0.14226469775794898, -0.14226469775794898, 0, 0,
			0, 5140.224542382912, 0, -5140.224542382912,
			0, 0, 1, -43.01721556839596,
		}, []float64{1, 0, 0, 0}, 0.04387426072060871},
	} {
		sys := integratorExactSystem(t, tc.n, 1, tc.a, tc.b)
		for _, method := range integratorExactMethods {
			d, err := sys.C2D(tc.dt, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s %s: %v", tc.name, method, err)
			}
			label := tc.name + " " + string(method)
			integratorExactRow(t, label, d.A, 0, 1)
			integratorExactInputRow(t, label, method, sys, d, 0, tc.dt)
		}
	}
}

// A non-symmetric A with a zero row inside: the row stays e_1 for every
// method and step, and the block of the other states is the exponential of
// that block alone, which A is block triangular around.
func TestC2DZeroRowOfNonSymmetricA(t *testing.T) {
	a := []float64{
		-3, 7, 0.5, 2,
		0, -0, 0, 0,
		11, -5, -2, 13,
		-1, 4, -17, 0.25,
	}
	sys := integratorExactSystem(t, 4, 2, a, []float64{1, -2, 3, 0.5, 0, 4, -1, 2})
	rest := []int{0, 2, 3}
	for _, dt := range []float64{1e-4, 0.01, 0.1, 1, 7} {
		block := expmScaled(submatrix(sys.A, rest), dt)
		for _, method := range integratorExactMethods {
			d, err := sys.C2D(dt, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("dt=%g %s: %v", dt, method, err)
			}
			label := string(method)
			integratorExactRow(t, label, d.A, 1, 1)
			integratorExactInputRow(t, label, method, sys, d, 1, dt)
			if method == C2DMethodTustin {
				continue
			}
			for p, i := range rest {
				for q, j := range rest {
					if got, want := d.A.At(i, j), block.At(p, q); got != want {
						t.Errorf("dt=%g %s: Ad[%d][%d] = %.17g, block exponential %.17g", dt, method, i, j, got, want)
					}
				}
			}
		}
	}
}

// 4POFJL fam5/36: a zero column (state 2 feeds nothing) stays e_2, and the
// integrator's DCGain stays -Inf as the continuous one.
func TestC2DKeepsZeroColumnIntegratorExact(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{-83.145847519739149, 74.900022086183782, 0, -64.716449937881947, -42.981654534948696, 0, 53.76368994194894, 28.937849530829752, 0},
		[]float64{-0.50184029378928918, 0.94882088399006226, -0.26283498653712134},
		[]float64{0.8854764002366402, -0.57708201044231111, -0.12648187463586336},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, method := range integratorExactMethods {
		d, err := sys.C2D(2.2800951917671597, C2DOptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		integratorExactColumn(t, string(method), d.A, 2)
		if method == C2DMethodImpulse {
			continue
		}
		g, err := d.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		if got := g.At(0, 0); !math.IsInf(got, -1) {
			t.Errorf("%s: DCGain = %.17g, want -Inf", method, got)
		}
	}
}

// 1/s²·a/(s+a): the second integrator is a zero row of the block left after
// the first, so the double pole stays at z = 1 and DCGain keeps the sign of
// the continuous +Inf (rounding put it at -Inf for ZOH, FOH and Tustin).
func TestC2DDoubleIntegratorDCGainSign(t *testing.T) {
	sys := integratorExactSystem(t, 3, 1, []float64{0, -0, 0, 1, -0, 0, 0, 6.144509238034576, -6.144509238034576}, []float64{1, 0, 0})
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		d, err := sys.C2D(1.6062886258144187, C2DOptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		integratorExactRow(t, string(method), d.A, 0, 1)
		if got := d.A.At(1, 1); got != 1 {
			t.Errorf("%s: Ad[1][1] = %.17g, want 1", method, got)
		}
		if got := d.A.At(1, 2); got != 0 {
			t.Errorf("%s: Ad[1][2] = %.17g, want 0", method, got)
		}
		g, err := d.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		if got := g.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("%s: DCGain = %.17g, want +Inf", method, got)
		}
	}
}

// A=[0 0; 1 -1], dt=25: the lag pole e^-25 is computed alone, so D2C, whose
// logarithm amplifies its absolute rounding by e^25, recovers A; with the
// pole from the exponential of all of A it was off by 2.7e-8. D2C keeps the
// integrator row exactly zero and B's row as Bd/dt.
func TestC2DIntegratorWithFastLagRoundTrip(t *testing.T) {
	sys := integratorExactSystem(t, 2, 1, []float64{0, 0, 1, -1}, []float64{1, 0})
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		d, err := sys.C2D(25, C2DOptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		integratorExactRow(t, string(method), d.A, 0, 1)
		if method != C2DMethodTustin {
			if got, want := d.A.At(1, 1), math.Exp(-25); math.Abs(got-want) > 2e-16*want {
				t.Errorf("%s: Ad[1][1] = %.17g, want e^-25 = %.17g", method, got, want)
			}
		}
		c, err := d.D2C(D2COptions{Method: method})
		if err != nil {
			t.Fatalf("%s D2C: %v", method, err)
		}
		integratorExactRow(t, string(method)+" D2C", c.A, 0, 0)
		if got := c.B.At(0, 0); got != 1 {
			t.Errorf("%s D2C: B[0] = %.17g, want 1", method, got)
		}
		for i := range 2 {
			for j := range 2 {
				if diff := math.Abs(c.A.At(i, j) - sys.A.At(i, j)); diff > 1e-14 {
					t.Errorf("%s D2C: A[%d][%d] = %.17g, want %g", method, i, j, c.A.At(i, j), sys.A.At(i, j))
				}
			}
		}
	}
}

// Without inputs and on the stiff modal path the zero row is exact too.
func TestC2DZeroRowWithoutInputsAndWhenStiff(t *testing.T) {
	free, err := NewFromSlices(3, 0, 1, []float64{0, 0, 0, 1, 0, -21, 0, 1, -10}, nil, []float64{0, 0, 1}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	stiff := integratorExactSystem(t, 3, 1, []float64{-1e10, 3e9, 2, 0, -1, 5, 0, 0, 0}, []float64{1, 1, 1})
	for _, tc := range []struct {
		name string
		sys  *System
		row  int
	}{{"no inputs", free, 0}, {"stiff", stiff, 2}} {
		for _, method := range integratorExactMethods {
			d, err := tc.sys.C2D(0.1, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s %s: %v", tc.name, method, err)
			}
			label := tc.name + " " + string(method)
			integratorExactRow(t, label, d.A, tc.row, 1)
			integratorExactInputRow(t, label, method, tc.sys, d, tc.row, 0.1)
		}
	}
}

// A nonzero row, however small, is not an integrator and keeps its
// exponential.
func TestC2DSmallNonzeroRowNotIntegrator(t *testing.T) {
	sys := integratorExactSystem(t, 2, 1, []float64{1e-3, 0, 1, -2}, []float64{1, 0})
	d, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := d.A.At(0, 0), math.Exp(1e-4); got == 1 || math.Abs(got-want) > 1e-15 {
		t.Errorf("Ad[0][0] = %.17g, want %.17g", got, want)
	}
}
