package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// Regression for XO2CWA: Dgebal graded the delay-absorbed shift chain of
// x⁺ = a·x + b·u[t-k], y = c·x + d·u[t-k] toward the tiny diagonal a, so
// TransferFunction returned junk coefficients (1e44 at k = 3, a = 3.7e-44)
// and lost ~1e-7 at a = 1e-3, k = 10. Oracle: the closed form
// G = (d·z + c·b − d·a)/(z^k·(z − a)), also for the integrator-chain
// continuous twin near |s| = 1 (s^-k in coefficient form loses ~eps·|s|^±k).
func TestTransferFunctionShiftChain(t *testing.T) {
	const b, c, d = -0.9667981925794609, -0.8568887533689471, 57.7675665396686
	for _, a := range []float64{3.72e-44, 1e-3, 0.5} {
		for _, k := range []int{1, 2, 3, 5, 10, 20} {
			for _, dt := range []float64{1, 0} {
				if dt == 0 && k > 5 {
					continue
				}
				t.Run(fmt.Sprintf("a=%g/k=%d/dt=%g", a, k, dt), func(t *testing.T) {
					sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{b}, []float64{c}, []float64{d}, 1)
					if err != nil {
						t.Fatal(err)
					}
					if err := sys.SetInputDelay([]float64{float64(k)}); err != nil {
						t.Fatal(err)
					}
					m, err := sys.AbsorbDelay(AbsorbAll)
					if err != nil {
						t.Fatal(err)
					}
					if dt == 0 {
						if m, err = New(m.A, m.B, m.C, m.D, 0); err != nil {
							t.Fatal(err)
						}
					}
					res, err := m.TransferFunction(nil)
					if err != nil {
						t.Fatal(err)
					}
					if k <= 5 && res.MinimalOrder != k+1 {
						t.Errorf("MinimalOrder = %d, want %d", res.MinimalOrder, k+1)
					}
					worst := 0.0
					for i := range 20 {
						var z complex128
						if dt == 0 {
							z = complex(0, math.Pow(10, -0.5+float64(i)/19))
						} else {
							z = cmplx.Rect(1, 0.05+3*float64(i)/19)
						}
						want := (complex(d, 0)*z + complex(c*b-d*a, 0)) / (cmplx.Pow(z, complex(float64(k), 0)) * (z - complex(a, 0)))
						got := mustEval(t, res.TF, z)[0][0]
						worst = max(worst, cmplx.Abs(got-want)/cmplx.Abs(want))
					}
					if worst > 1e-12 {
						t.Errorf("TF eval rel err %g > 1e-12; num %v den %v", worst, res.TF.Num[0][0], res.TF.Den[0])
					}
					roots, err := Poly(res.TF.Den[0]).Roots()
					if err != nil {
						t.Fatal(err)
					}
					for _, r := range roots {
						if dt != 0 && cmplx.Abs(r) >= 1 {
							t.Errorf("unstable pole %v", r)
						}
					}
				})
			}
		}
	}
}
