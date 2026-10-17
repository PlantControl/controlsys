package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidExactFRD samples num/den at n log-spaced points on [lo, hi].
func pidExactFRD(t *testing.T, num, den []float64, lo, hi float64, n int) *FRD {
	t.Helper()
	omega := make([]float64, n)
	values := make([][][]complex128, n)
	for k := range omega {
		omega[k] = math.Exp(math.Log(lo) + float64(k)*math.Log(hi/lo)/float64(n-1))
		s := complex(0, omega[k])
		values[k] = [][]complex128{{pidPolyAt(num, s) / pidPolyAt(den, s)}}
	}
	frd, err := NewFRD(values, omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	return frd
}

// pidClosedLoopMaxRe returns the largest real part among the roots of
// den·Cd + num·Cn for continuous C = Cn/Cd, from companion eigenvalues.
func pidClosedLoopMaxRe(num, den []float64, c PID2) float64 {
	cn, cd := []float64{c.Kd, c.Kp}, []float64{1}
	switch {
	case c.Ki != 0 && c.Tf != 0:
		cn = []float64{c.Kp*c.Tf + c.Kd, c.Kp + c.Ki*c.Tf, c.Ki}
		cd = []float64{c.Tf, 1, 0}
	case c.Ki != 0:
		cn, cd = []float64{c.Kd, c.Kp, c.Ki}, []float64{1, 0}
	case c.Tf != 0:
		cn, cd = []float64{c.Kp*c.Tf + c.Kd, c.Kp}, []float64{c.Tf, 1}
	}
	a, b := polyMulTest(den, cd), polyMulTest(num, cn)
	f := make([]float64, max(len(a), len(b)))
	for i, v := range a {
		f[len(f)-len(a)+i] += v
	}
	for i, v := range b {
		f[len(f)-len(b)+i] += v
	}
	for len(f) > 1 && f[0] == 0 {
		f = f[1:]
	}
	n := len(f) - 1
	comp := mat.NewDense(n, n, nil)
	for j := range n {
		comp.Set(0, j, -f[j+1]/f[0])
	}
	for i := 1; i < n; i++ {
		comp.Set(i, i-1, 1)
	}
	var e mat.Eigen
	if !e.Factorize(comp, mat.EigenNone) {
		return math.Inf(1)
	}
	worst := math.Inf(-1)
	for _, z := range e.Values(nil) {
		worst = math.Max(worst, real(z))
	}
	return worst
}

// Sampled data carries no stability certificate, so a crossing whose
// wrapped margin is at or below −floor cannot count the far way round from
// −1. Counting it did: on this plant, with G(0) < 0, PID and PIDF designs
// with Ki > 0 crossed 0 dB at low frequency with ∠L in (0°, 123°] and
// closed the loop unstable (root at Re +1.205 for PID). Oracle: roots of
// the closed-loop characteristic polynomial.
func TestTunePIDFRDUncertifiedCrossingKeepsSign(t *testing.T) {
	num := []float64{1.475767261587179, -2.761797130009405}
	den := []float64{1, 2.8234906807454356, 6.053557929541098, 12.422570998833214}
	frd := pidExactFRD(t, num, den, .07531, 73.01, 600)
	for _, family := range []PidtuneType{PidtunePID, PidtunePIDF} {
		r, err := TunePIDFRD(context.Background(), frd, family, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", family, err)
			continue
		}
		if re := pidClosedLoopMaxRe(num, den, *r.Controller); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g", family, *r.Controller, re)
		}
	}
}

// A free P design on sampled data keeps the grid's crossings: the refined
// scan adds crossings near a lightly damped mode, one 129.8° from −1 and one
// with ∠L ≈ +21°, which the plant's own margin less 3° as floor, or the
// wrapped sign, rejected, though the designs are robust. Oracles: roots of
// the closed-loop characteristic polynomial and a dense |S| sweep.
func TestTunePIDFRDFreePKeepsRobustDesign(t *testing.T) {
	cases := []struct {
		num, den []float64
		lo, hi   float64
	}{
		{[]float64{0.7939620977398079, 0.13791298408186412}, []float64{1, 0.010682696869646655, 0.10582524497554917}, .01084, 9.759},
		{[]float64{0.3159448471911536, 0.5458028650195357}, []float64{1, 0.030629362065009907, 7.396247920990802}, .09065, 81.59},
	}
	for _, c := range cases {
		r, err := TunePIDFRD(context.Background(), pidExactFRD(t, c.num, c.den, c.lo, c.hi, 600), PidtuneP, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%v/%v: %v", c.num, c.den, err)
			continue
		}
		k := *r.Controller
		if re := pidClosedLoopMaxRe(c.num, c.den, k); re >= 0 {
			t.Errorf("%v/%v: %+v closed-loop root at Re %g", c.num, c.den, k, re)
		}
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		if peak, at := pidDenseSensitivity(k, g, c.lo, c.hi); peak > 1.2 {
			t.Errorf("%v/%v: %+v |S| %g at ω=%g", c.num, c.den, k, peak, at)
		}
		for i, m := range r.Evidence.PhaseMargins {
			w := r.Evidence.GainCrossovers[i]
			if want := 180 + cmplx.Phase(g(w)*complex(k.Kp, 0))*180/math.Pi; math.Abs(math.Remainder(m-want, 360)) > .5 || m < 57 {
				t.Errorf("%v/%v: evidence margin %g at %g, want %g", c.num, c.den, m, w, want)
			}
		}
	}
}
