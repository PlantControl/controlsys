package controlsys

import (
	"context"
	"math"
	"testing"
)

// Zeros at 0.0124 and 0.375 rad/s, both in the right half-plane, put the
// unwrapped phase of G near −540° at wc, so pidTuningReachable, judging PD
// bands unwrapped, skipped every PM±3° and floor band and TunePID failed.
// A PD loop has |L(0)| = Kp·G(0) ≈ 0.01 here: it crosses −180° inside the
// unit circle and is stable at the wrapped margin, as a ladder design
// C = e^{j(−180°+pm)}/G(jwc) at wc0/2 confirms (min PM 45, |S| 1.55).
// Oracles: dense margins and |S|, and Routh-Hurwitz on den + num·(Kp+Kd·s).
func TestTunePIDNonIntegralJudgedWrapped(t *testing.T) {
	num := []float64{0.25735368672434666, -0.09959483477301487, 0.0011398317458989433}
	den := []float64{1, 96.41485396485868, 3538.780004777385, 53732.01273661257, 13120.99411576446}
	plant := makePlant(t, num, den)
	for _, pm := range []float64{0, 45} {
		r, err := TunePID(context.Background(), plant, PidtunePD, PIDTuningOptions{PhaseMargin: pm})
		if err != nil {
			t.Errorf("PM %v: %v", pm, err)
			continue
		}
		k := r.Controller
		if k.Kp <= 0 || k.Kd <= 0 {
			t.Errorf("PM %v: %+v, want positive gains", pm, k)
		}
		want := pm
		if want == 0 {
			want = 60
		}
		L := func(w float64) complex128 {
			s := complex(0, w)
			return (complex(k.Kp, 0) + complex(k.Kd, 0)*s) * pidPolyAt(num, s) / pidPolyAt(den, s)
		}
		minPM, peakS := pidDenseLoop(L, 1e-5, 1e7)
		if minPM < want-3-1 || peakS > 4 {
			t.Errorf("PM %v: %+v dense min PM %.3g, |S| %.3g; want PM ≥ %v−3°, |S| ≤ 4", pm, k, minPM, peakS, want)
		}
		a := make([]float64, len(den))
		copy(a, den)
		for i, c := range num {
			a[i+1] += k.Kd * c
			a[i+2] += k.Kp * c
		}
		if !pidRouthHurwitz4(a) {
			t.Errorf("PM %v: %+v closed loop %v not Hurwitz", pm, k, a)
		}
	}
}

// pidRouthHurwitz4 reports whether a0·s⁴+a1·s³+a2·s²+a3·s+a4, a0 > 0, has
// all roots in the open left half-plane.
func pidRouthHurwitz4(a []float64) bool {
	for _, c := range a {
		if c <= 0 {
			return false
		}
	}
	d2 := a[1]*a[2] - a[0]*a[3]
	return d2 > 0 && d2*a[3]-a[1]*a[1]*a[4] > 0 && !math.IsNaN(d2)
}
