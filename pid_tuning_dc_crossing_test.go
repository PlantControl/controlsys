package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// TL2SR6: an integral-free loop with |L(0)| = |Kp·G(0)| < 1 that is over
// 0 dB at wc/100 crosses 0 dB an odd number of times below the band, where
// the scan never looked: PD PM 30 on the first plant returned a 20.7°
// crossing at 2.2 rad/s (floor 27°), PD PM 70 on the second (two RHP zeros)
// a 24° one at 0.399, PDF PM 60 on the third a 40.8° one at 0.21; evidence
// listed only wc. Oracle: a dense sweep of L(jω) = C(jω)·N/D(jω) from closed
// forms over [1e-6·wc, 1e4·wc] for every crossing, its distance from −1 and
// the |S| peak, and Routh on D·(Tf·s+1) + N·((Kd+Kp·Tf)s + Kp).
func TestTunePIDDCCrossingBelowBand(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		fam      PidtuneType
		pm       float64
	}{
		{[]float64{96.46474652911039, 16.51842066238342, -0.8586508679087753, 0.008930275177641552}, []float64{1, 180.81855672115316, 11253.923157468167, 269494.11653469934, 1.7758776576580715e+06, 196191.35369955824}, PidtunePD, 30},
		{[]float64{0.018827899758323052, -0.001834309166705823, 2.720462874759158e-05}, []float64{1, 40.76435691098496, 410.20832612883845, 864.4337789067902, 500.81429732734296}, PidtunePD, 70},
		{[]float64{-1.1205453792335838, 0.1994486499039301, -0.004470014410911086}, []float64{1, 27.00547628595657, 243.38383311322823, 735.163520890191, 28.34977869181296}, PidtunePDF, 60},
	} {
		r, err := TunePID(context.Background(), makePlant(t, c.num, c.den), c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Fatalf("%v %s: %v", c.num, c.fam, err)
		}
		k := r.Controller
		wc := r.Evidence.AchievedCrossover
		eval := func(p []float64, s complex128) complex128 {
			var v complex128
			for _, x := range p {
				v = v*s + complex(x, 0)
			}
			return v
		}
		loop := func(w float64) complex128 {
			s := complex(0, w)
			return eval(c.num, s) / eval(c.den, s) * (complex(k.Kp, 0) + complex(k.Kd, 0)*s/(complex(k.Tf, 0)*s+1))
		}
		const n = 200000
		peak, crossings := 0., 0
		prev := cmplx.Abs(loop(1e-6*wc)) - 1
		for i := 1; i <= n; i++ {
			w := 1e-6 * wc * math.Pow(1e10, float64(i)/n)
			l := loop(w)
			peak = math.Max(peak, 1/cmplx.Abs(1+l))
			d := cmplx.Abs(l) - 1
			if math.Signbit(d) != math.Signbit(prev) {
				crossings++
				if m := 180 - math.Abs(cmplx.Phase(l)*180/math.Pi); m < c.pm-3-.1 {
					t.Errorf("%v %s: 0 dB crossing at %.4g rad/s %.3g° from −1, floor %g", c.num, c.fam, w, m, c.pm-3)
				}
				found := false
				for _, x := range r.Evidence.GainCrossovers {
					found = found || math.Abs(math.Log(x/w)) < .01
				}
				if !found {
					t.Errorf("%v %s: crossing at %.4g rad/s not in evidence %v", c.num, c.fam, w, r.Evidence.GainCrossovers)
				}
			}
			prev = d
		}
		if crossings != len(r.Evidence.GainCrossovers) {
			t.Errorf("%v %s: %d crossings, evidence %v", c.num, c.fam, crossings, r.Evidence.GainCrossovers)
		}
		if peak > 4 {
			t.Errorf("%v %s: |S| peak %.3g", c.num, c.fam, peak)
		}
		ch := polyMul(c.den, []float64{k.Tf, 1})
		nc := polyMul(c.num, []float64{k.Kd + k.Kp*k.Tf, k.Kp})
		for i := range nc {
			ch[len(ch)-len(nc)+i] += nc[i]
		}
		for len(ch) > 1 && ch[0] == 0 {
			ch = ch[1:]
		}
		if !routhStable(ch) {
			t.Errorf("%v %s: closed loop %v unstable", c.num, c.fam, ch)
		}
	}
}

// The DC walk extends the zero grid below lo only on a sign mismatch of
// |L| − 1 between ω = 0 and lo, to the first point matching ω = 0.
func TestPIDTuningDCCrossingFloor(t *testing.T) {
	g := func(w float64) complex128 { return 1 / complex(1, w) }
	p := pidTuningPlant{at: g, dcEdge: 1}
	step := math.Log(10) / 60
	for _, c := range []struct {
		c    PID2
		ok   bool
		want float64
	}{
		{PID2{Kp: .5, Kd: 2, B: 1, C: 1}, true, .5},
		{PID2{Kp: 2, Kd: 2, B: 1, C: 1}, false, 0},
		{PID2{Kp: .5, Ki: 1, Kd: 2, B: 1, C: 1}, false, 0},
		{PID2{Kp: .5, Kd: .01, B: 1, C: 1}, false, 0},
	} {
		w, ok := pidTuningDCCrossingFloor(p, c.c, 10, step, map[float64]complex128{})
		if ok != c.ok || ok && (w > c.want || w < c.want*math.Exp(-step)) {
			t.Errorf("%+v: %g %v, want %v just under %g", c.c, w, ok, c.ok, c.want)
		}
	}
}
