package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// pidExactSensitivityPeak returns max |1/(1+P·C)| of the exact continuous
// model on a dense log grid over [lo, hi] and at ω → ∞.
func pidExactSensitivityPeak(num, den []float64, c PID2, lo, hi float64) float64 {
	peak := 0.
	for k := 0; k <= 20000; k++ {
		w := lo * math.Pow(hi/lo, float64(k)/20000)
		s := complex(0, w)
		peak = math.Max(peak, 1/cmplx.Abs(1+pidPolyAt(num, s)/pidPolyAt(den, s)*pidTuningFeedback(c, w)))
	}
	if c.Kd != 0 && c.Tf == 0 && len(den) == len(num)+1 {
		peak = math.Max(peak, 1/math.Abs(1+c.Kd*num[0]/den[0]))
	}
	return peak
}

func pidSensitivityWarned(r *PIDTuningResult) bool {
	return slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "Sampled sensitivity peak") })
}

// PD designs parked where Kd·CB is just past −1 are stable but have |S| of
// 47 and 139. Under the |S| bound the first finds a design within it on a
// floor band; the second's PD bands have none at the default crossover or
// any lowered or raised one, and as the true peak is at least the sampled
// one the call fails (JW7LKX) rather than keep it with a warning; at an
// explicit crossover it is kept with one. Oracles: closed-loop roots and
// dense exact |S|.
func TestTunePIDFRDBoundsFragileSensitivity(t *testing.T) {
	for i, c := range []struct {
		num, den []float64
		n        int
		pm, wc   float64
		bounded  bool
	}{
		{[]float64{17.66332276886301}, []float64{1, 0.4166065677430895}, 600, 45, 0, true},
		{[]float64{0.06711826191147859, -0.009292143346332136}, []float64{1, 0.580742570603219, 0.08335532261816535}, 100, 60, 0, false},
		{[]float64{0.06711826191147859, -0.009292143346332136}, []float64{1, 0.580742570603219, 0.08335532261816535}, 100, 60, 1, false},
	} {
		frd := pidExactFRD(t, c.num, c.den, 1e-3, 1e3, c.n)
		r, err := TunePIDFRD(context.Background(), frd, PidtunePD, PIDTuningOptions{PhaseMargin: c.pm, CrossoverFrequency: c.wc})
		if !c.bounded && c.wc == 0 {
			te, ok := errors.AsType[*PIDTuningError](err)
			if !ok || !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "the true peak is at least the sampled one") }) {
				t.Errorf("case %d: want unattainable over the |S| bound, got %v", i, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}
		k := *r.Controller
		if re := pidClosedLoopMaxRe(c.num, c.den, k); re >= 0 {
			t.Errorf("case %d: %+v closed-loop root at Re %g", i, k, re)
		}
		s, warned := pidExactSensitivityPeak(c.num, c.den, k, 1e-3, 1e3), pidSensitivityWarned(r)
		if c.bounded && (s > 4*1.05 || warned) {
			t.Errorf("case %d: %+v |S| %g, warned %v", i, k, s, warned)
		} else if !c.bounded && !warned {
			t.Errorf("case %d: %+v |S| %g without warning", i, k, s)
		}
		if r.Evidence.AchievedPhaseMargin < c.pm-3-1e-6 {
			t.Errorf("case %d: achieved margin %g", i, r.Evidence.AchievedPhaseMargin)
		}
	}
}

// Against the unbounded search on random stable plants, the |S| bound never
// turns a design into an error, never loses closed-loop stability and only
// replaces a design over the bound by one within it. Oracles: closed-loop
// roots and dense exact |S|.
func TestTunePIDFRDSensitivityBoundKeepsDesigns(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(16, 0xffc3d6))
	n := 300
	if testing.Short() {
		n = 60
	}
	replaced := 0
	for i := range n {
		num, den := pidRandomStableTest(rng, false, .01, 100)
		family := []PidtuneType{PidtunePD, PidtunePI, PidtunePID, PidtunePDF, PidtunePIDF}[rng.IntN(5)]
		o := PIDTuningOptions{PhaseMargin: []float64{0, 30, 45, 60}[rng.IntN(4)]}
		frd := pidExactFRD(t, num, den, 1e-3, 1e3, []int{100, 300}[rng.IntN(2)])
		name := fmt.Sprintf("case %d %s %v/%v pm %g", i, family, num, den, o.PhaseMargin)
		q := o
		q.CrossoverFrequency = 1
		p := pidTuningFRDPlant(frd, nil)
		ref, refErr := tunePID(context.Background(), "ref", p, family, q)
		r, err := TunePIDFRD(context.Background(), frd, family, o)
		if refErr != nil {
			continue
		}
		if err != nil {
			t.Errorf("%s: reference design %+v became %v", name, *ref.Controller, err)
			continue
		}
		k, kr := *r.Controller, *ref.Controller
		if pidClosedLoopMaxRe(num, den, kr) < 0 && pidClosedLoopMaxRe(num, den, k) >= 0 {
			t.Errorf("%s: stable %+v became unstable %+v", name, kr, k)
		}
		if k == kr {
			continue
		}
		replaced++
		// Sparse samples can miss a resonant |S| peak, so the exact peak need
		// only fall; the sampled one meets the bound.
		p.maxSensitivity = pidTuningMaxSensitivity
		sampled := pidTuningSensitivityPeak(p, k)
		if s, sr := pidExactSensitivityPeak(num, den, k, 1e-3, 1e3), pidExactSensitivityPeak(num, den, kr, 1e-3, 1e3); sampled > 4 || s > sr {
			t.Errorf("%s: %+v |S| %g (sampled %g) replaced %+v |S| %g", name, k, s, sampled, kr, sr)
		}
	}
	t.Logf("%d of %d designs replaced", replaced, n)
}
