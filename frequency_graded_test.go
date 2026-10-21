package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
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
