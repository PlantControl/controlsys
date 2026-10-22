package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// padeUnitResponse evaluates the order-n Padé approximant of e^{-x} at x
// from the closed-form coefficients c_k = (2n-k)!n!/((2n)!k!(n-k)!).
func padeUnitResponse(n int, x complex128) complex128 {
	var num, den complex128
	c, xk := 1.0, complex(1, 0)
	for k := 0; k <= n; k++ {
		if k > 0 {
			c *= float64(n-k+1) / float64(k*(2*n-k+1))
			xk *= x
		}
		term := complex(c, 0) * xk
		den += term
		if k%2 == 1 {
			num -= term
		} else {
			num += term
		}
	}
	return num / den
}

// gradedPadeCompanion realizes the Padé approximant of e^{-tau·s} from its
// unscaled coefficients, whose companion realization spans tau^n decades.
func gradedPadeCompanion(t *testing.T, tau float64, n int) *System {
	t.Helper()
	den := make([]float64, n+1)
	num := make([]float64, n+1)
	c := 1.0
	for k := 0; k <= n; k++ {
		if k > 0 {
			c *= float64(n-k+1) / float64(k*(2*n-k+1))
		}
		den[n-k] = c * math.Pow(tau, float64(k))
		num[n-k] = den[n-k]
		if k%2 == 1 {
			num[n-k] = -den[n-k]
		}
	}
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatalf("tau=%g n=%d: StateSpace: %v", tau, n, err)
	}
	return res.Sys
}

func checkGradedResponse(t *testing.T, label string, sys *System, tau float64, n int) {
	t.Helper()
	ws := []float64{1e-3, 0.1, 1, 3, 10, 1e3}
	grid := make([]float64, 0, 120)
	for len(grid) < cap(grid) {
		for _, w := range ws {
			grid = append(grid, w/tau)
		}
	}
	fr, err := sys.FreqResponse(grid)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	for k, w := range ws {
		want := padeUnitResponse(n, complex(0, w))
		g, err := sys.EvalFr(complex(0, w/tau))
		if err != nil {
			t.Errorf("%s w·tau=%g: EvalFr: %v", label, w, err)
			continue
		}
		for name, got := range map[string]complex128{"EvalFr": g[0][0], "FreqResponse": fr.At(k, 0, 0)} {
			if e := cmplx.Abs(got-want) / cmplx.Abs(want); !(e < 1e-9) {
				t.Errorf("%s w·tau=%g: %s = %v, want %v (rel err %g)", label, w, name, got, want, e)
			}
		}
	}
}

func TestFrequencyResponseGradedCompanion(t *testing.T) {
	for _, c := range []struct {
		tau float64
		n   int
	}{{1e24, 10}, {1e27, 10}, {1e31, 6}, {1e40, 5}} {
		checkGradedResponse(t, "companion", gradedPadeCompanion(t, c.tau, c.n), c.tau, c.n)
	}
}

func TestFrequencyResponseExtremeTimeScale(t *testing.T) {
	for _, c := range []struct {
		tau float64
		n   int
	}{{1.7e35, 10}, {1.7e80, 5}, {1.7e53, 7}, {1.7e161, 3}, {1.7e162, 3}, {6.8e162, 9}, {1.7e212, 2}, {1.7e-250, 4}} {
		sys, err := PadeDelay(c.tau, c.n)
		if err != nil {
			t.Fatalf("PadeDelay(%g, %d): %v", c.tau, c.n, err)
		}
		checkGradedResponse(t, "PadeDelay", sys, c.tau, c.n)
	}
}

func TestBalanceKeepsEntriesNormal(t *testing.T) {
	sys, err := PadeDelay(1.7e212, 2)
	if err != nil {
		t.Fatal(err)
	}
	br := newBalancedRealization(sys, 2, 1, 1)
	raw := sys.A.RawMatrix().Data
	for i, v := range br.a {
		if v == 0 {
			continue
		}
		if math.Abs(v) < 0x1p-1022 {
			t.Errorf("balanced a[%d] = %g is subnormal", i, v)
		}
		got, _ := math.Frexp(v)
		want, _ := math.Frexp(raw[i])
		if got != want {
			t.Errorf("balanced a[%d] = %g is not a power-of-two scaling of %g", i, v, raw[i])
		}
	}
}

// 2^k·(A, B) at 2^k·s has exactly the response of (A, B) at s, so the unit
// model is the oracle next to each real pole, where the double solve of the
// scaled one overflows or underflows to NaN.
func TestEvalFrExtremeScaleNearPole(t *testing.T) {
	a := []float64{-1, 2, 0.5, 0, -3, 1, 0.3, 0.2, -2}
	b := []float64{1, 0.5, -1}
	build := func(sc float64) *System {
		as, bs := make([]float64, len(a)), make([]float64, len(b))
		for i, v := range a {
			as[i] = sc * v
		}
		for i, v := range b {
			bs[i] = sc * v
		}
		sys, err := New(mat.NewDense(3, 3, as), mat.NewDense(3, 1, bs),
			mat.NewDense(1, 3, []float64{1, 1, 0.5}), mat.NewDense(1, 1, []float64{0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	unit := build(1)
	var eig mat.Eigen
	if !eig.Factorize(unit.A, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	for _, k := range []int{990, -996} {
		sc := math.Ldexp(1, k)
		sys := build(sc)
		n, m, p := sys.Dims()
		for _, l := range eig.Values(nil) {
			if imag(l) != 0 {
				continue
			}
			s := complex(real(l)*(1-1e-10), 0)
			ref, err := unit.EvalFr(s)
			if err != nil {
				t.Fatal(err)
			}
			want := ref[0][0]
			check := func(name string, got complex128, err error) {
				t.Helper()
				if err != nil {
					t.Errorf("2^%d λ=%g %s: %v", k, real(l), name, err)
				} else if e := cmplx.Abs(got-want) / cmplx.Abs(want); !(e < 1e-5) {
					t.Errorf("2^%d λ=%g %s = %v, want %v (rel err %g)", k, real(l), name, got, want, e)
				}
			}
			ss := s * complex(sc, 0)
			var g00 complex128
			g, err := sys.EvalFr(ss)
			if err == nil {
				g00 = g[0][0]
			}
			check("EvalFr", g00, err)
			solvers := map[string]frequencyPointSolver{
				"dense":      newBalancedDense(sys, n, m, p),
				"hessenberg": newHessenbergSweep(sys, n, m, p),
			}
			for name, solver := range solvers {
				dst := make([]complex128, 1)
				err := evalWithPoleLimit(solver.evalInto, sys, pointAt(ss), dst)
				check(name, dst[0], err)
			}
		}
	}
}
