package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// WALFFO: the zero grid stopped a tenth below the slowest controller zero,
// so a 0 dB crossing set by a slow plant zero, pole pair or the integral
// asymptote |Ki·G(0)|/ω just under the band went unchecked and unreported:
// PIDF PM 60 on the first plant returned a 53.7° crossing at 1.62 rad/s
// (floor 57), PIDF on the second a 24.7° one at 0.94, PD PM 70 on the
// delayed third a 64.6° one at 0.128 (floor 67), PIDF on the open-loop
// unstable fourth a 14.4° one at 2.45 with |S| 3.9997; PIDF PM 70 on the
// fifth omitted its 110° and 72° crossings at 0.0108 and 0.486 from the
// evidence, which must list every crossing above 5e-5·wc. Oracle: a dense sweep of L(jω) = C(jω)·N/D(jω)·e^{−jωτ} from the
// coefficients over [1e-6·wc, 1e4·wc] for every crossing, its distance from
// −1 and the |S| peak; Routh on D·s·(Tf·s+1) + N·((Kd+Kp·Tf)s² + (Kp+Ki·Tf)s
// + Ki), or, delayed on a stable plant, the winding of 1 + L.
func TestTunePIDCrossingBelowBand(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		num, den []float64
		fam      PidtuneType
		pm, tau  float64
		mayFail  bool
	}{
		{[]float64{2504.444153211668, 8000.8360921350695, 5652.654115776044, 220.57179315706313, 113.05533127906948}, []float64{1, 14.351525646425516, 994.9380339175436, 27560.042875296203, 721693.8026117089, 8273883.872548497, 9947177.84989779}, PidtunePIDF, 60, 0, false},
		{[]float64{0.11298704946902867, 33.733128940004924, 4.763112162344632, 0.2686315693287133, 0.007777432027729306, 0.00010166155924553476}, []float64{1, 234.40258682863535, 27594.51272256439, 772826.886465239, 73303085.60566911, 103234796.97975771, 36587250.37243428}, PidtunePIDF, 0, 0, false},
		{[]float64{56.67042627442348, 0.3935356569641866, 0.8876682529886802}, []float64{1, 1.9280998028198946, 0.08336661973608298, 0.06715013757420492}, PidtunePD, 70, 0.0262797, false},
		{[]float64{169.3086264199369, 117.1376256569317, 14.060955638217882, 0.26988589144247266}, []float64{1, -0.39027089636060064, 104492.84937990247, -69370.87902312992, 6.930706329573018e+06}, PidtunePIDF, 0, 0, true},
		{[]float64{2.941320948811059, 4.605230336947651, 0.7032887081942705, 0.021970969668926563}, []float64{1, -89.76653114586391, 184.0303979291221, -4680.089550275783, 7069.266971795929}, PidtunePIDF, 70, 0, false},
	} {
		plant := makePlant(t, c.num, c.den)
		if c.tau > 0 {
			if err := plant.SetInputDelay([]float64{c.tau}); err != nil {
				t.Fatal(err)
			}
		}
		r, err := TunePID(context.Background(), plant, c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			if !c.mayFail || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%v %s: %v", c.num, c.fam, err)
			}
			continue
		}
		pm := c.pm
		if pm == 0 {
			pm = 60
		}
		k := r.Controller
		wc := r.Evidence.AchievedCrossover
		band := r.Evidence.FrequencyBand
		eval := func(p []float64, s complex128) complex128 {
			var v complex128
			for _, x := range p {
				v = v*s + complex(x, 0)
			}
			return v
		}
		loop := func(w float64) complex128 {
			s := complex(0, w)
			ctrl := complex(k.Kp, 0) + complex(k.Kd, 0)*s/(complex(k.Tf, 0)*s+1)
			if k.Ki != 0 {
				ctrl += complex(k.Ki, 0) / s
			}
			return eval(c.num, s) / eval(c.den, s) * ctrl * cmplx.Exp(-s*complex(c.tau, 0))
		}
		const n = 200000
		peak := 0.
		prev := cmplx.Abs(loop(1e-6*wc)) - 1
		for i := 1; i <= n; i++ {
			w := 1e-6 * wc * math.Pow(1e10, float64(i)/n)
			l := loop(w)
			peak = math.Max(peak, 1/cmplx.Abs(1+l))
			d := cmplx.Abs(l) - 1
			if math.Signbit(d) != math.Signbit(prev) {
				if m := 180 - math.Abs(cmplx.Phase(l)*180/math.Pi); m < pm-3-.1 {
					t.Errorf("%v %s: 0 dB crossing at %.4g rad/s %.3g° from −1, floor %g", c.num, c.fam, w, m, pm-3)
				}
				found := w < 5e-5*wc || w > band[1]
				for _, x := range r.Evidence.GainCrossovers {
					found = found || math.Abs(math.Log(x/w)) < .01
				}
				if !found {
					t.Errorf("%v %s: crossing at %.4g rad/s not in evidence %v", c.num, c.fam, w, r.Evidence.GainCrossovers)
				}
			}
			prev = d
		}
		if peak > 4 {
			t.Errorf("%v %s: |S| peak %.4g", c.num, c.fam, peak)
		}
		if c.tau == 0 {
			ch := polyMul(c.den, []float64{k.Tf, 1, 0})
			nc := polyMul(c.num, []float64{k.Kd + k.Kp*k.Tf, k.Kp + k.Ki*k.Tf, k.Ki})
			for i := range nc {
				ch[len(ch)-len(nc)+i] += nc[i]
			}
			for len(ch) > 1 && ch[0] == 0 {
				ch = ch[1:]
			}
			if !routhStable(ch) {
				t.Errorf("%v %s: closed loop %v unstable", c.num, c.fam, ch)
			}
			continue
		}
		// Stable plant, integral-free ideal PD: |L| → |Kd·N₀| < 1, so past
		// wEnd 1 + L stays in a disk about 1 off the origin and winds no
		// more; a closed-loop RHP root would turn arg(1 + L) by −π over ω ≥ 0.
		if !routhStable(c.den) || k.Ki != 0 || k.Tf != 0 {
			t.Fatalf("%v %s: oracle needs a stable plant and ideal PD, got %+v", c.num, c.fam, k)
		}
		const wEnd, m = 1e5, 2000000
		if r := math.Abs(k.Kd * c.num[0]); r > .9 || cmplx.Abs(loop(wEnd)) > .9 {
			t.Fatalf("%v %s: tail |L| %.3g, %.3g not under 1", c.num, c.fam, r, cmplx.Abs(loop(wEnd)))
		}
		arg, pv := 0., 1+loop(1e-7)
		for i := 1; i <= m; i++ {
			v := 1 + loop(1e-7*math.Pow(wEnd/1e-7, float64(i)/m))
			arg += cmplx.Phase(v / pv)
			pv = v
		}
		if math.Abs(arg) > math.Pi/2 {
			t.Errorf("%v %s: arg(1+L) turns %.3g rad, closed loop unstable", c.num, c.fam, arg)
		}
	}
}
