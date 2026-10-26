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
	t.Parallel()
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

// Certified designs whose |S| peaks where the sampled scan reads none:
// between two samples across a lightly damped mode whose two fits
// disagree (J42E7A: 4.75 and 8.86 against sampled 3.07 and 2.56), on the
// DC-to-lo line below the band (CXWHXX: 11.5), at z = −1 above discrete
// data stopping at 0.955π (4.38) and at ω → ∞ of an ideal PD on a
// relative-degree-one plant (4.24). Each design is returned unchanged,
// stable, now without certification, the ideal PD's also as the strict
// count certifies no ideal derivative's end above the band (QHMGQL).
// Ideal PDs parked just under the bound at ω → ∞ (strict sweep seeds 519,
// 565, 833) keep their unsampled peak under it, as the Markov end is
// extrapolated, not widened by its spread, though that end leaves them
// uncertified. Oracles: companion roots, exact Schur–Cohn and dense exact
// |S|.
func TestTunePIDFRDUnsampledSensitivityUncertified(t *testing.T) {
	for _, c := range []pidFRDStrictCase{
		{"between samples PIDF", []float64{5.256412315376749e-08, 4.7338299227851e-09}, []float64{1, 0.00696924768219797, 0.0005706342067653564, 1.49497704056467e-06, 5.978781921603224e-08}, 0, 0.00018504889257226238, 1.0694725898516853, 67, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: 2.052565140571584, Ki: 0.019062621341399227, Kd: 245.4031504143213, Tf: 0.07109858305474219}},
		{"sparse between samples PIDF", []float64{-0.0009501753302565275}, []float64{1, 0.27163067136624075, 0.0611715345300015, 0.009142841844187882, 0.00013121342125506342}, 0, 0.0017225746868072026, 8.035444271007805, 29, PidtunePIDF, PIDTuningOptions{PhaseMargin: 70}, PID2{Kp: -0.6838885802382229, Ki: -0.012940973039722694, Kd: -1.6868808667511357, Tf: 0.01202861076201854}},
		{"below band PID", []float64{8956.585594900349, -136138.83030121162, 203360.74062388795}, []float64{1, 65.324624930553185, 138.64933209380754, 3008.0268873768041, 1038.3228838918781}, 0, 0.090582667614022838, 4655.7551987022716, 218, PidtunePID, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -0.004648420111217961, Ki: 1.6093841187230203e-05, Kd: -0.00010947744200795812}},
		{"short of Nyquist P", []float64{3.2276235464735703}, []float64{1, -0.1353352832366127}, 1, 0.1, 3, 235, PidtuneP, PIDTuningOptions{DFormula: BackwardEuler}, PID2{Kp: 0.27146407295759734, DFormula: BackwardEuler, Dt: 1}},
		{"ideal PD at infinity", []float64{29.738740460879715}, []float64{1, 58.620180397923804}, 0, 14.294148998804204, 428.82446996412614, 212, PidtunePD, PIDTuningOptions{PhaseMargin: 30}, PID2{Kp: 2.3083139862711555, Kd: -0.02568974752121812}},
	} {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Ki, c.k.Ki}, {k.Kd, c.k.Kd}, {k.Tf, c.k.Tf}} {
			if math.Abs(g[0]-g[1]) > 1e-3*math.Abs(g[1]) {
				t.Errorf("%s: %+v, base gave %+v", c.name, k, c.k)
				break
			}
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", c.name, k)
		}
		ideal := k.Dt == 0 && k.Tf == 0 && k.Kd != 0 && pidHasWarning(r, "loop's end above the band is established")
		if peak := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo); !(peak > 4.2) || !ideal && !pidHasWarning(r, "sensitivity peak is not certified") {
			t.Errorf("%s: dense |S| %g, warnings %q", c.name, peak, r.Evidence.Warnings)
		}
	}
	for _, seed := range []uint64{519, 565, 833} {
		c := pidFRDStrictSweepCase(rand.New(rand.NewPCG(7729, seed)))
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Errorf("seed %d: %v", seed, err)
			continue
		}
		k := *r.Controller
		strict, _ := pidTuningFRDStrictPlant(frd, c.o.UnstablePoles)
		peak, unsampled := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo), pidTuningFRDUnsampledPeak(strict, k)
		if !pidFRDStrictStable(c.num, c.den, k) || !(peak <= 4) || unsampled > pidTuningMaxSensitivity || !pidHasWarning(r, "loop's end above the band is established") {
			t.Errorf("seed %d: %+v dense |S| %g, unsampled %g, warnings %q", seed, k, peak, unsampled, r.Evidence.Warnings)
		}
	}
}
