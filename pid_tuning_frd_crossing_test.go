package controlsys

import (
	"context"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
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

// TunePIDFRD scans for gain crossings on the measured samples too, since its
// log-linear interpolant can cross 0 dB between the 241-point crossover
// grid's points. A lightly damped mode near 2 rad/s, sampled 600 times on
// [1e-3, 1e3], put |L| over 1 between grid points: PID at wc 0.1623
// reported only wc and 63° while samples straddled |L| = 1 near 1.95 and
// 2.0 (ζ = 1e-2, a graze peaking at |L| 1.0000025, ∠L 24°) or four times
// over 1.93–2.06 (ζ = 1e-3, ζ = 0, PID and PIDF), with ∠L near +24° there,
// a wrapped margin near −156°. Oracles: sign changes of |L| − 1 over
// adjacent samples of the documented interpolant, and closed-loop roots of
// the exact plant.
func TestTunePIDFRDCrossingsBetweenGridPoints(t *testing.T) {
	zeta2 := [2][]float64{{6.335797216731407, 24.391227797024047, 54.72113901759225, 74.15735471976932, 17.515790582716292}, {1, 6.680688332652979, 7.2460737902166565, 26.873377152592504, 11.955834263494538, 1.258925411794167}}
	zeta3 := [2][]float64{{6.335797216731407, 24.176553599151454, 54.076453749896416, 73.97279314928015, 17.511839053168945}, {1, 6.6447736109835365, 7.007571914003141, 26.76563298758415, 11.944477031293877, 1.2589254117941655}}
	zeta0 := [2][]float64{{6.335797216731407, 24.152681801253724, 54.004695153133355, 73.95222898037906, 17.51139395145642}, {1, 6.640783086353598, 6.981071705534972, 26.75366141369434, 11.943215116604915, 1.258925411794167}}
	for i, c := range []struct {
		plant  [2][]float64
		family PidtuneType
	}{{zeta2, PidtunePID}, {zeta3, PidtunePID}, {zeta3, PidtunePIDF}, {zeta0, PidtunePID}} {
		num, den := c.plant[0], c.plant[1]
		name := fmt.Sprintf("case %d %s", i, c.family)
		frd := pidExactFRD(t, num, den, 1e-3, 1e3, 600)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{CrossoverFrequency: 0.1623})
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		k, ev := *r.Controller, r.Evidence
		if re := pidClosedLoopMaxRe(num, den, k); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g", name, k, re)
		}
		g, ctrl := fzPidInterp(frd), fzPidCtrlOf2(&k)
		excess := func(w float64) float64 {
			v, _ := g(w)
			return cmplx.Abs(v*ctrl.resp(w)) - 1
		}
		for j := 1; j < len(frd.Omega); j++ {
			lo, hi := frd.Omega[j-1], frd.Omega[j]
			if excess(lo)*excess(hi) >= 0 {
				continue
			}
			if !slices.ContainsFunc(ev.GainCrossovers, func(w float64) bool { return w >= lo*(1-1e-12) && w <= hi*(1+1e-12) }) {
				t.Errorf("%s: |L| crosses 1 in [%.9g, %.9g], reported %v", name, lo, hi, ev.GainCrossovers)
			}
		}
		fzPidCheckMissed(t, name, g, g, frd.Omega, "", ctrl, ev)
		for j, m := range ev.PhaseMargins {
			if ev.AchievedPhaseMargin > m+1e-9 {
				t.Errorf("%s: achieved margin %g above %g at %g", name, ev.AchievedPhaseMargin, m, ev.GainCrossovers[j])
			}
		}
	}
}

// A free one-gain design on sampled data keeps base's design, held to the
// floor at the grid's crossings, and reports every crossing the samples
// resolve, the achieved margin their least. Base reported wc alone; holding
// the sampled crossings to the floor of the plant's own margin at wc less
// 3° instead rejected robust designs: P on two resonant plants (|S| 1.04,
// 1.08), P at wc 0.119 (|S| 1.33) and 7.58 (|S| 1.0), and I at wc 0.401
// (|S| 2.47) and 0.0624 (|S| 1.60), whose extra crossings sit at wrapped
// margins down to −172° on −1's far side. The I design at wc 0.0262 is
// unstable (root at Re +0.012) and is now returned with its −80° crossing
// as the achieved margin, not the 90.2° at wc. Oracles: closed-loop roots
// and a dense |S| sweep of the exact plant, and sign changes of |L| − 1
// and wrapped phases of the documented interpolant.
func TestTunePIDFRDFreeOneGainReportsSampledCrossings(t *testing.T) {
	for i, c := range []struct {
		num, den []float64
		lo, hi   float64
		n        int
		family   PidtuneType
		wc       float64
		maxS     float64
		stable   bool
	}{
		{[]float64{0.7939620977398079, 0.13791298408186412}, []float64{1, 0.010682696869646655, 0.10582524497554917}, .01084, 9.759, 600, PidtuneP, 0, 1.1, true},
		{[]float64{0.3159448471911536, 0.5458028650195357}, []float64{1, 0.030629362065009907, 7.396247920990802}, .09065, 81.59, 600, PidtuneP, 0, 1.15, true},
		{[]float64{1.5903543306381103}, []float64{1, 0.11318929931127387, 0.020361548600023264}, 1e-3, 1e3, 300, PidtuneP, 0.119026, 1.4, true},
		{[]float64{-0.0019483147102757746, -0.00258257825573115, -0.016970841926992237}, []float64{1, 0.9091916613710589, 5.752828019275366, 4.8374031348092545, 2.0260560018589606}, 1e-3, 1e3, 600, PidtuneI, 0.400602, 2.6, true},
		{[]float64{0.05717012722456311, 0.04140408783392416}, []float64{1, 0.42065340344555174, 49.67277713267589}, 1e-3, 1e3, 100, PidtuneP, 7.5756, 1.1, true},
		{[]float64{0.1004901079990616, 0.3496524855798061}, []float64{1, 0.4227702150952549, 0.3232980277536846, 0.12154095272876622, 0.010295197850730308}, 1e-3, 1e3, 2000, PidtuneI, 0.062406911627341584, 1.7, true},
		{[]float64{0.017257967204262814, 0.1192876907101629}, []float64{1, 0.002049203028195259, 0.2955636770772375}, 1e-3, 1e3, 200, PidtuneI, 0.026179536491585137, 0, false},
	} {
		frd := pidExactFRD(t, c.num, c.den, c.lo, c.hi, c.n)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{CrossoverFrequency: c.wc})
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}
		k, ev := *r.Controller, r.Evidence
		exact := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		re := pidClosedLoopMaxRe(c.num, c.den, k)
		if c.stable {
			if re >= 0 {
				t.Errorf("case %d: %+v closed-loop root at Re %g", i, k, re)
			}
			if peak, at := pidDenseSensitivity(k, exact, c.lo, c.hi); peak > c.maxS {
				t.Errorf("case %d: %+v |S| %g at ω=%g", i, k, peak, at)
			}
		} else if re <= 0 || ev.AchievedPhaseMargin >= 0 {
			t.Errorf("case %d: %+v root at Re %g, achieved margin %g, want an unstable loop reported below 0°", i, k, re, ev.AchievedPhaseMargin)
		}
		g, ctrl := fzPidInterp(frd), fzPidCtrlOf2(&k)
		crossings := 0
		for j := 1; j < len(frd.Omega); j++ {
			lo, hi := frd.Omega[j-1], frd.Omega[j]
			a, _ := g(lo)
			b, _ := g(hi)
			if (cmplx.Abs(a*ctrl.resp(lo))-1)*(cmplx.Abs(b*ctrl.resp(hi))-1) >= 0 {
				continue
			}
			crossings++
			if !slices.ContainsFunc(ev.GainCrossovers, func(w float64) bool { return w >= lo*(1-1e-12) && w <= hi*(1+1e-12) }) {
				t.Errorf("case %d: |L| crosses 1 in [%.9g, %.9g], reported %v", i, lo, hi, ev.GainCrossovers)
			}
		}
		if crossings < 2 {
			t.Errorf("case %d: %d sampled crossings, want the extra ones this case pins", i, crossings)
		}
		least := math.Inf(1)
		for j, m := range ev.PhaseMargins {
			w := ev.GainCrossovers[j]
			l, _ := g(w)
			if want := fzPidWrap(180 + cmplx.Phase(l*ctrl.resp(w))*180/math.Pi); math.Abs(math.Remainder(m-want, 360)) > .5 {
				t.Errorf("case %d: margin %g at %g, interpolant gives %g", i, m, w, want)
			}
			least = math.Min(least, m)
		}
		if math.Abs(ev.AchievedPhaseMargin-least) > 1e-9 {
			t.Errorf("case %d: achieved margin %g, least crossing margin %g", i, ev.AchievedPhaseMargin, least)
		}
	}
}
