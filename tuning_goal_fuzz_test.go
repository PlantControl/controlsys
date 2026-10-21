package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzTuningGoalEvaluate (task O5FYRC): Gain, Tracking, Sensitivity and
// Rejection goals evaluated on a stable delay-free model with a known focus
// band. The returned Value must equal the scaled σmax of the exact rational
// response at the reported peak_frequency (inside the band), and be at least
// the scaled exact σmax at every point of the band's log grid, up to the
// ω-rounding sensitivity. The Poles goal value is checked against the
// intended eigenvalues. Pass must be Value ≤ 1.

func FuzzTuningGoalEvaluate(f *testing.F) {
	for _, s := range fzTGSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		sys.InputName, sys.OutputName = fzTGNames("u", s.M), fzTGNames("y", s.P)
		if !s.Decidable() {
			fzSkip(t, "tuninggoal: stability undecidable")
		}
		fzTGPoles(t, s, sys)
		if s.Unstable() > 0 || s.N == 0 && s.Vals[7] == 0 {
			return
		}
		lo, hi := 1e-3, 1e3
		for _, p := range s.S {
			if a := cmplx.Abs(p); a > 0 {
				lo, hi = math.Min(lo, a/100), math.Max(hi, a*100)
			}
		}
		top := hi
		if s.Discrete {
			top = math.Min(hi, math.Pi/s.Dt)
		}
		c := math.Pow(10, fzClamp(s.Vals[12], -3, 3))
		rt := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
		type goalCase struct {
			name  string
			goal  func() (TuningGoal, error)
			scale func(w float64) float64
			track bool
		}
		cases := []goalCase{
			{"Gain", func() (TuningGoal, error) { return NewGainGoal("u", "y", c) }, func(float64) float64 { return 1 / c }, false},
			{"Sensitivity", func() (TuningGoal, error) { return NewSensitivityGoal("x", fzTGStatic(t, c, s.Dt)) }, func(float64) float64 { return 1 / c }, false},
			{"Rejection", func() (TuningGoal, error) { return NewRejectionGoal("x", fzTGStatic(t, c, s.Dt)) }, func(float64) float64 { return c }, false},
			{"Tracking", func() (TuningGoal, error) { return NewTrackingGoal("u", "y", rt, 0, 0) }, func(w float64) float64 {
				wc, x := 2/rt, complex(0, w)
				return cmplx.Abs((x + complex(wc, 0)) / (x + complex(wc*0.001, 0)))
			}, true},
		}
		cache := map[float64][]complex128{}
		for _, gc := range cases {
			g, err := gc.goal()
			fzErr(t, s, "New"+gc.name+"Goal", err)
			g, err = g.WithFocus(lo, hi)
			fzErr(t, s, gc.name+".WithFocus", err)
			res, err := g.Evaluate(sys)
			if gc.track && s.M != s.P {
				if !errors.Is(err, ErrDimensionMismatch) {
					t.Fatalf("Tracking %dx%d: err %v, want ErrDimensionMismatch\n%s", s.P, s.M, err, s)
				}
				continue
			}
			if lo >= top {
				if !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("%s: focus above Nyquist: err %v, want ErrInvalidArgument\n%s", gc.name, err, s)
				}
				continue
			}
			fzErr(t, s, gc.name+".Evaluate", err)
			fzTGCheckPeak(t, s, cache, gc.name, res, lo, top, gc.scale, gc.track)
		}
	})
}

func fzTGNames(prefix string, k int) []string {
	if k == 1 {
		return []string{prefix}
	}
	out := make([]string, k)
	for i := range out {
		out[i] = fmt.Sprintf("%s(%d)", prefix, i+1)
	}
	return out
}

func fzTGStatic(t *testing.T, c, dt float64) *System {
	t.Helper()
	g, err := NewFromSlices(0, 1, 1, nil, nil, nil, []float64{c}, dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v", err)
	}
	return g
}

// fzTGSigma returns scale·σmax(H − I?) of the exact response at ω and the
// relative error allowed for it: CondT² backward error plus the ω-rounding
// sensitivity ω·Σ 1/|x − λ| (·Dt for discrete).
func fzTGSigma(t *testing.T, s *fzSpec, cache map[float64][]complex128, w float64, track bool) (float64, float64) {
	t.Helper()
	x := complex(0, w)
	if s.Discrete {
		x = cmplx.Exp(complex(0, w*s.Dt))
	}
	h0, ok := cache[w]
	if !ok {
		if h0, ok = fzExactResponse(s.N, s.M, s.P, s.A, s.B, s.C, s.D, x); !ok {
			fzSkip(t, "tuninggoal: grid point on a pole")
		}
		cache[w] = h0
	}
	h := slices.Clone(h0)
	if track {
		for i := range s.P {
			h[i*s.M+i]--
		}
	}
	sv := fzSingularValues(h, s.P, s.M)
	if sv == nil {
		fzSkip(t, "tuninggoal: oracle SVD failed")
	}
	sens := 0.0
	for _, e := range s.Eig {
		sens += 1 / math.Max(cmplx.Abs(x-e), 1e-300)
	}
	if s.Discrete {
		sens *= s.Dt
	}
	return sv[0], 1e-8*math.Pow(s.CondT, 2) + 1e-13*w*sens
}

func fzTGCheckPeak(t *testing.T, s *fzSpec, cache map[float64][]complex128, name string, res TuningGoalResult, lo, hi float64, scale func(float64) float64, track bool) {
	t.Helper()
	if res.Pass != (res.Value <= 1) || res.Limit != 1 || res.Violation != math.Max(0, res.Value-1) {
		t.Fatalf("%s: inconsistent result %+v\n%s", name, res, s)
	}
	at, ok := res.Diagnostics["peak_frequency"]
	if !ok || !(at >= lo*(1-1e-12) && at <= hi*(1+1e-12)) {
		t.Fatalf("%s: peak_frequency %g outside focus [%g, %g]\n%s", name, at, lo, hi, s)
	}
	sv, tol := fzTGSigma(t, s, cache, at, track)
	if tol > 1e-3 {
		fzSkip(t, "tuninggoal: ill-conditioned peak")
	}
	if want := sv * scale(at); math.Abs(res.Value-want) > tol*want+1e-300 {
		t.Fatalf("%s: Value %.17g at ω=%g, exact %.17g (tol %g)\n%s", name, res.Value, at, want, tol, s)
	}
	a, b := math.Log10(lo), math.Log10(hi)
	for i := range tuningGoalGridPoints {
		w := math.Pow(10, a+(b-a)*float64(i)/float64(tuningGoalGridPoints-1))
		if i == 0 {
			w = lo
		} else if i == tuningGoalGridPoints-1 {
			w = hi
		}
		sv, tol := fzTGSigma(t, s, cache, w, track)
		if want := sv * scale(w); res.Value < want*(1-tol) {
			t.Fatalf("%s: Value %.17g below exact %.17g at grid ω=%g (tol %g)\n%s", name, res.Value, want, w, tol, s)
		}
	}
}

// fzTGPoles checks the Poles goal value from the intended poles
// s = λ (continuous) or log(λ)/Dt (discrete), bounding the error of each
// pole by δ = EigErr·κ(balancing) mapped through the log.
func fzTGPoles(t *testing.T, s *fzSpec, sys *System) {
	t.Helper()
	mindecay := math.Pow(10, fzClamp(s.Vals[12], -3, 3))
	mindamp := fzClamp(math.Abs(s.Vals[13])/10, 0, 1)
	maxfreq := math.Inf(1)
	if s.Vals[14] < 4.5 {
		maxfreq = math.Pow(10, fzClamp(s.Vals[14], -2, 4.5))
	}
	g, err := NewPolesGoal("", mindecay, mindamp, maxfreq)
	fzErr(t, s, "NewPolesGoal", err)
	res, err := g.Evaluate(sys)
	fzErr(t, s, "Poles.Evaluate", err)
	if s.N == 0 {
		if res.Value != 0 {
			t.Fatalf("Poles goal on static gain: Value %g, want 0\n%s", res.Value, s)
		}
		return
	}
	if s.Unstable() > 0 {
		if !math.IsInf(res.Value, 1) || res.Pass {
			t.Fatalf("Poles goal on unstable model: %+v, want Value +Inf\n%s", res, s)
		}
		return
	}
	delta := s.EigErr() * fzBalanceCond(s.A, s.N)
	decay, damp, freq, rel := math.Inf(1), math.Inf(1), 0.0, 0.0
	for _, e := range s.Eig {
		v, d := e, delta
		if s.Discrete {
			if cmplx.Abs(e) <= 2*delta {
				fzSkip(t, "tuninggoal: pole near z = 0")
			}
			v = cmplx.Log(e) / complex(s.Dt, 0)
			d = delta / ((cmplx.Abs(e) - delta) * s.Dt)
			if math.Abs(imag(e)) <= 2*delta && real(e) < 0 {
				fzSkip(t, "tuninggoal: pole near the log branch cut")
			}
		}
		mag := cmplx.Abs(v)
		decay = math.Min(decay, -real(v))
		damp = math.Min(damp, -real(v)/mag)
		freq = math.Max(freq, mag)
		rel = math.Max(rel, d/math.Min(-real(v), mag))
	}
	if rel > 1e-4 {
		fzSkip(t, "tuninggoal: pole error large against decay")
	}
	want := math.Max(mindecay/decay, mindamp/damp)
	if !math.IsInf(maxfreq, 1) {
		want = math.Max(want, freq/maxfreq)
	}
	if math.Abs(res.Value-want) > 4*rel*want+1e-300 {
		t.Fatalf("Poles goal Value %.17g, intended %.17g (rel tol %g) decay=%g damp=%g freq=%g diag=%v\n%s", res.Value, want, 4*rel, decay, damp, freq, res.Diagnostics, s)
	}
}

func fzTGSeeds() []fzSeed {
	none := fzSeed{In: -1, Out: 0, IO: 5, Tau: fzNoDelay, Shear: -10}
	mk := func(f func(*fzSeed)) fzSeed {
		x := none
		f(&x)
		return x
	}
	return []fzSeed{
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain = 2, [3]float64{0}, [3]float64{2.5}, 6 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.DScal = 2, [3]float64{1}, [3]float64{3.5}, 6, 0.3 }),
		mk(func(x *fzSeed) {
			x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.DScal = 3, 2, 2, [3]float64{0, 0.5}, [3]float64{2, 1}, 6, 0.5
		}),
		mk(func(x *fzSeed) {
			x.N, x.M, x.P, x.Mag, x.Damp, x.Gain, x.Shear, x.Rot = 4, 3, 2, [3]float64{-0.5, 0.7}, [3]float64{1.5, 2.8}, 5, -1, 0
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.Mag, x.Damp, x.Gain, x.DScal = true, -1, 2, [3]float64{0.5}, [3]float64{2.2}, 6, 0.1
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.M, x.P, x.Mag, x.Damp, x.Gain = true, -2, 3, 2, 2, [3]float64{0, 1}, [3]float64{1, 2.5}, 6
		}),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain = 2, [3]float64{0}, [3]float64{-2.5}, 6 }),
		mk(func(x *fzSeed) { x.N, x.DScal = 0, 2 }),
	}
}
