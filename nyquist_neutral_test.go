package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// neutralChainRoot refines a root of χ(s) = den + num·e^{-sτ} from the
// asymptotic guess s_m = (ln|d| + j·(2m+1)π)/τ (d > 0) by Newton.
func neutralChainRoot(q quasiLoop, m int) (complex128, bool) {
	d := q.num[0] / q.den[0]
	s := complex(math.Log(math.Abs(d))/q.tau, float64(2*m+1)*math.Pi/q.tau)
	if d < 0 {
		s = complex(math.Log(-d)/q.tau, float64(2*m)*math.Pi/q.tau)
	}
	dnum, dden := polyDerivCoeffs(q.num), polyDerivCoeffs(q.den)
	for range 60 {
		e := cmplx.Exp(-s * complex(q.tau, 0))
		f := polyEval(q.den, s) + polyEval(q.num, s)*e
		df := polyEval(dden, s) + (polyEval(dnum, s)-complex(q.tau, 0)*polyEval(q.num, s))*e
		step := f / df
		s -= step
		if cmplx.Abs(step) <= 1e-13*cmplx.Abs(s) {
			return s, true
		}
	}
	return s, false
}

func polyDerivCoeffs(p []float64) []float64 {
	n := len(p) - 1
	if n <= 0 {
		return []float64{0}
	}
	d := make([]float64, n)
	for i := range n {
		d[i] = p[i] * float64(n-i)
	}
	return d
}

func TestNyquistNeutralDelayLoopUnsupported(t *testing.T) {
	k, a1, a0 := 1.8749, 2.6223, 0.25411
	cases := []struct{ d, tau float64 }{
		{1.49706, 0.71947},
		{1.269, 2.713},
		{1.427, 2.950},
		{1.0144, 2.225},
		{-1.2, 1},
		{1, 1},
	}
	for _, c := range cases {
		q := quasiLoop{num: []float64{c.d, c.d*a1 + 0.3*k, c.d*a0 + k}, den: []float64{1, a1, a0}, tau: c.tau}
		if math.Abs(c.d) > 1 {
			for _, m := range []int{1000, 10000, 100000} {
				s, ok := neutralChainRoot(q, m)
				if !ok || !(real(s) > 0) {
					t.Fatalf("d=%g tau=%g: chain root %d = %v (converged %v), want Re > 0", c.d, c.tau, m, s, ok)
				}
			}
		}
		sys := q.system(t)
		lft, err := sys.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range []*System{sys, lft} {
			if res, err := m.Nyquist(nil, 0); !errors.Is(err, ErrDelayUnsupported) {
				n := -1
				if res != nil {
					n = res.RHPZerosCL
				}
				t.Errorf("d=%g tau=%g lft=%v: RHPZerosCL %d, err %v; want ErrDelayUnsupported", c.d, c.tau, m.LFT != nil, n, err)
			}
		}
	}
}

func TestNyquistSubunitFeedthroughDelayLoopCounts(t *testing.T) {
	k, a1, a0 := 1.8749, 2.6223, 0.25411
	d := 0.3
	q := quasiLoop{num: []float64{d, d*a1 + 0.3*k, d*a0 + k}, den: []float64{1, a1, a0}, tau: 1}
	res, err := q.system(t).Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, ok := sweepOracle(q)
	if !ok {
		t.Fatal("oracle undecided")
	}
	if res.RHPZerosCL != want {
		t.Errorf("RHPZerosCL = %d, want %d", res.RHPZerosCL, want)
	}
}
