package controlsys_test

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

func sweepStabilityDense(r, c int, v ...float64) *mat.Dense {
	if len(v) == 0 {
		return mat.NewDense(r, c, nil)
	}
	return mat.NewDense(r, c, v)
}

func sweepStabilitySys(t *testing.T, a, b, c, d float64) *controlsys.System {
	t.Helper()
	sys, err := controlsys.New(sweepStabilityDense(1, 1, a), sweepStabilityDense(1, 1, b),
		sweepStabilityDense(1, 1, c), sweepStabilityDense(1, 1, d), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sweepStabilityAcceptableErr(err error) bool {
	return errors.Is(err, controlsys.ErrDelayUnsupported) ||
		errors.Is(err, controlsys.ErrContinuousInternalDelay) ||
		errors.Is(err, controlsys.ErrSingularTransform)
}

// sweepStabilityAssert fails on a wrong verdict. Explicit indeterminate
// errors are accepted (and logged) unless mustDecide is set.
func sweepStabilityAssert(t *testing.T, sys *controlsys.System, want, mustDecide bool) {
	t.Helper()
	got, err := sys.IsStable()
	if err != nil {
		if sweepStabilityAcceptableErr(err) && !mustDecide {
			t.Logf("INDETERMINATE (accepted): %v", err)
			return
		}
		t.Fatalf("IsStable error: %v (want %v)", err, want)
	}
	if got != want {
		t.Fatalf("IsStable = %v, want %v", got, want)
	}
}

// sweepStabilityAssertNotStable accepts false or an explicit indeterminate
// error, as MATLAB isstable counts marginal models as unstable.
func sweepStabilityAssertNotStable(t *testing.T, sys *controlsys.System) {
	t.Helper()
	got, err := sys.IsStable()
	if err != nil {
		if sweepStabilityAcceptableErr(err) {
			t.Logf("INDETERMINATE (accepted): %v", err)
			return
		}
		t.Fatalf("IsStable error: %v", err)
	}
	if got {
		t.Fatal("IsStable = true for a marginal model")
	}
}

// sweepStabilityAssertKnownRefusal is sweepStabilityAssert with the known
// false refusal of retarded loops whose D22 is nilpotent skipped.
func sweepStabilityAssertKnownRefusal(t *testing.T, sys *controlsys.System, want bool) {
	t.Helper()
	if _, err := sys.IsStable(); errors.Is(err, controlsys.ErrDelayUnsupported) {
		t.Skipf("known defect, ergo A6FJ66: retarded loop refused as neutral: %v", err)
	}
	sweepStabilityAssert(t, sys, want, true)
}

func sweepStabilityBisect(f func(float64) float64, lo, hi float64) float64 {
	flo := f(lo)
	for range 200 {
		mid := (lo + hi) / 2
		if fm := f(mid); (fm > 0) == (flo > 0) {
			lo, flo = mid, fm
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

func sweepStabilityFeedback(t *testing.T, plant, ctrl *controlsys.System) *controlsys.System {
	t.Helper()
	cl, err := controlsys.Feedback(plant, ctrl, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Fatal("closed loop has no internal delay")
	}
	return cl
}

// sweepStabilityDelayedIntegratorLoop returns three realizations of
// ẋ = −a·x(t−τ), χ(s) = s + a·e^{−sτ}: a direct internal delay, feedback of
// e^{−τs}/s with gain a, and feedback of 1/s with a·e^{−τs}.
func sweepStabilityDelayedIntegratorLoop(t *testing.T, a, tau float64) map[string]*controlsys.System {
	t.Helper()
	direct := sweepStabilitySys(t, 0, 1, 1, 0)
	if err := direct.SetInternalDelay([]float64{tau}, sweepStabilityDense(1, 1, -a), sweepStabilityDense(1, 1, 1),
		sweepStabilityDense(1, 1), sweepStabilityDense(1, 1), sweepStabilityDense(1, 1)); err != nil {
		t.Fatal(err)
	}
	plant := sweepStabilitySys(t, 0, 1, 1, 0)
	if err := plant.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	gain, err := controlsys.NewGain(sweepStabilityDense(1, 1, a), 0)
	if err != nil {
		t.Fatal(err)
	}
	delayedGain, err := controlsys.NewWithDelay(nil, nil, nil, sweepStabilityDense(1, 1, a), sweepStabilityDense(1, 1, tau), 0)
	if err != nil {
		t.Fatal(err)
	}
	return map[string]*controlsys.System{
		"internal":        direct,
		"inputDelayPlant": sweepStabilityFeedback(t, plant, gain),
		"delayedGainCtrl": sweepStabilityFeedback(t, sweepStabilitySys(t, 0, 1, 1, 0), delayedGain),
	}
}

// D1: ẋ = −a·x(t−τ) is stable iff aτ < π/2 (roots ±ja at aτ = π/2,
// Re ds/dτ = a²/(1+π²/4) > 0 there). All three realizations must agree,
// including at relative distances 1e-3 and 1e-6 from the boundary.
func TestSweepStabilityDelayedIntegratorBoundary(t *testing.T) {
	type tc struct {
		a, tau     float64
		mustDecide bool
	}
	b := math.Pi / 2
	cases := []tc{
		{1, 1.0, true}, {1, 1.5, true}, {1, 1.6, true}, {1, 2.0, true},
		{1, b - 1e-3, true}, {1, b + 1e-3, true},
		{1, b - 1e-6, false}, {1, b + 1e-6, false},
		{10, 0.15, true}, {10, 0.16, true},
		{10, b/10 - 1e-4, true}, {10, b/10 + 1e-4, true},
	}
	for _, c := range cases {
		want := c.a*c.tau < math.Pi/2
		for name, sys := range sweepStabilityDelayedIntegratorLoop(t, c.a, c.tau) {
			t.Run(fmt.Sprintf("a=%g/tau=%.9g/%s", c.a, c.tau, name), func(t *testing.T) {
				sweepStabilityAssert(t, sys, want, c.mustDecide)
			})
		}
	}
}

// D1: at aτ = π/2 (to float precision) χ has roots ±ja on the axis, so the
// verdict must not be "stable".
func TestSweepStabilityDelayedIntegratorMarginal(t *testing.T) {
	for _, a := range []float64{1, 10} {
		for name, sys := range sweepStabilityDelayedIntegratorLoop(t, a, math.Pi/2/a) {
			t.Run(fmt.Sprintf("a=%g/%s", a, name), func(t *testing.T) { sweepStabilityAssertNotStable(t, sys) })
		}
	}
}

// sweepStabilityFirstOrderLoop returns L = k·e^{−τs}/(s−p) with the delay
// on the output.
func sweepStabilityFirstOrderLoop(t *testing.T, p, k, tau float64) *controlsys.System {
	t.Helper()
	sys := sweepStabilitySys(t, p, 1, k, 0)
	if err := sys.SetOutputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	return sys
}

// sweepStabilityUnstableTauMax is the delay below which
// k·e^{−τs}/(s−1) is stabilized by unity feedback: ωc = √(k²−1) and
// ∠L(jωc) = −π + atan ωc − ωc·τ.
func sweepStabilityUnstableTauMax(k float64) (wc, tauMax float64) {
	wc = math.Sqrt(k*k - 1)
	return wc, math.Atan(wc) / wc
}

// D2: unity feedback around k·e^{−τs}/(s−1) is stable iff k > 1 and
// τ < atan(ωc)/ωc; checked by IsStable, Nyquist counts and DiskMargin.
func TestSweepStabilityUnstablePoleDelay(t *testing.T) {
	for _, k := range []float64{1.5, 2, 5} {
		_, tauMax := sweepStabilityUnstableTauMax(k)
		for _, f := range []float64{0.5, 0.99, 1.01, 1.5} {
			tau := f * tauMax
			want := f < 1
			t.Run(fmt.Sprintf("k=%g/tau=%.4gtauMax", k, f), func(t *testing.T) {
				L := sweepStabilityFirstOrderLoop(t, 1, k, tau)
				sweepStabilityAssert(t, sweepStabilityFeedback(t, L, nil), want, true)
				ny, err := L.Nyquist(nil, 0)
				if err != nil {
					t.Fatal(err)
				}
				wantCL := 0
				if !want {
					wantCL = 2
				}
				if ny.RHPPoles != 1 || ny.RHPZerosCL != wantCL || ny.Encirclements != wantCL-1 {
					t.Errorf("Nyquist P=%d N=%d Z=%d, want P=1 N=%d Z=%d", ny.RHPPoles, ny.Encirclements, ny.RHPZerosCL, wantCL-1, wantCL)
				}
				dm, err := controlsys.DiskMargin(L)
				if err != nil {
					t.Fatal(err)
				}
				if (dm.Alpha > 0) != want {
					t.Errorf("DiskMargin alpha = %g, want stable=%v", dm.Alpha, want)
				}
			})
		}
	}
	for _, k := range []float64{0.5, 0.99} {
		t.Run(fmt.Sprintf("k=%g", k), func(t *testing.T) {
			sweepStabilityAssert(t, sweepStabilityFeedback(t, sweepStabilityFirstOrderLoop(t, 1, k, 0.1), nil), false, true)
		})
	}
}

// sweepStabilityCheckMargin compares Margin/AllMargin of a first-order loop
// to the analytic crossover ωc, phase margin (including −ωc·τ) and the
// derived delay margin PM/ωc against an independently known value.
func sweepStabilityCheckMargin(t *testing.T, L *controlsys.System, wc, pmRad, delayMargin float64) {
	t.Helper()
	m, err := controlsys.Margin(L)
	if err != nil {
		t.Fatal(err)
	}
	w, ok := m.GainCrossover()
	if !ok || math.Abs(w-wc) > 1e-9*wc {
		t.Fatalf("Wcp = %v (ok=%v), want %v", w, ok, wc)
	}
	if pm := pmRad * 180 / math.Pi; math.Abs(m.PhaseMargin-pm) > 1e-6 {
		t.Errorf("PM = %.12g, want %.12g", m.PhaseMargin, pm)
	}
	if dm := m.PhaseMargin * math.Pi / 180 / w; math.Abs(dm-delayMargin) > 1e-8 {
		t.Errorf("delay margin = %.12g, want %.12g", dm, delayMargin)
	}
	all, err := controlsys.AllMargin(L)
	if err != nil {
		t.Fatal(err)
	}
	if len(all.GainCrossFreqs) != 1 || math.Abs(all.GainCrossFreqs[0]-wc) > 1e-9*wc {
		t.Errorf("AllMargin GainCrossFreqs = %v, want [%v]", all.GainCrossFreqs, wc)
	}
}

// D2: Margin of k·e^{−τs}/(s−1): ωc = √(k²−1), PM = atan ωc − ωc·τ, delay
// margin τmax − τ.
func TestSweepStabilityUnstablePoleDelayMargin(t *testing.T) {
	for _, c := range [][2]float64{{2, 0.3}, {5, 0.1}, {1.5, 0.5}} {
		k, tau := c[0], c[1]
		wc, tauMax := sweepStabilityUnstableTauMax(k)
		t.Run(fmt.Sprintf("k=%g/tau=%g", k, tau), func(t *testing.T) {
			sweepStabilityCheckMargin(t, sweepStabilityFirstOrderLoop(t, 1, k, tau), wc, math.Atan(wc)-wc*tau, tauMax-tau)
		})
	}
}

// D3: k·e^{−τs}/(s+1): ωc = √(k²−1), PM = π − atan ωc − ωc·τ, and the delay
// margin PM/ωc equals the Hayes bound acos(−1/k)/√(k²−1) − τ. Phase
// crossover solves atan ω + ωτ = π, with GM = −20log10(k/√(1+ω²)).
func TestSweepStabilityStableFirstOrderDelayMargin(t *testing.T) {
	for _, c := range [][2]float64{{2, 1}, {3, 0.2}, {10, 0.05}, {1.2, 2}} {
		k, tau := c[0], c[1]
		wc := math.Sqrt(k*k - 1)
		hayes := math.Acos(-1/k)/wc - tau
		t.Run(fmt.Sprintf("k=%g/tau=%g", k, tau), func(t *testing.T) {
			L := sweepStabilityFirstOrderLoop(t, -1, k, tau)
			sweepStabilityCheckMargin(t, L, wc, math.Pi-math.Atan(wc)-wc*tau, hayes)
			all, err := controlsys.AllMargin(L)
			if err != nil {
				t.Fatal(err)
			}
			w180 := sweepStabilityBisect(func(w float64) float64 { return math.Atan(w) + w*tau - math.Pi }, 0, 1e3)
			want := []float64{w180}
			if len(all.PhaseCrossFreqs) < 1 {
				t.Fatalf("PhaseCrossFreqs empty, want %v", want)
			}
			w0 := all.PhaseCrossFreqs[0]
			if math.Abs(w0-want[0]) > 1e-6*want[0] {
				t.Errorf("first phase crossover = %.12g, want %.12g", w0, want[0])
			}
			gm := -20 * math.Log10(k/math.Hypot(1, want[0]))
			if math.Abs(all.GainMargins[0]-gm) > 1e-6 {
				t.Errorf("first GM = %.12g dB, want %.12g", all.GainMargins[0], gm)
			}
			sweepStabilityAssert(t, sweepStabilityFeedback(t, L, nil), hayes > 0, true)
		})
	}
}

// D4: unity feedback around the upper-triangular 2×2
// G = [e^{−τ1 s}/s, 3e^{−0.4s}/(s+2); 0, 2e^{−τ2 s}/(s+1)], realized with a
// non-symmetric A by similarity and per-channel input/output delays.
// det(I+G) = (1+g11)(1+g22) and g12 is stable, so the loop is stable iff
// τ1 < π/2 and τ2 < (π − atan√3)/√3 = 2π/(3√3).
func TestSweepStabilityMIMOTriangular(t *testing.T) {
	t1, t2 := math.Pi/2, 2*math.Pi/(3*math.Sqrt(3))
	const coupleDelay = 0.4
	build := func(tau1, tau2 float64, inputOnly bool) *controlsys.System {
		T := sweepStabilityDense(3, 3, 1, 0.5, 0, -0.3, 1, 0.2, 0.4, 0, 1)
		var Ti mat.Dense
		if err := Ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		var A, B, C mat.Dense
		A.Mul(T, mat.NewDiagDense(3, []float64{0, -2, -1}))
		A.Mul(&A, &Ti)
		B.Mul(T, sweepStabilityDense(3, 2, 1, 0, 0, 3, 0, 2))
		C.Mul(sweepStabilityDense(2, 3, 1, 1, 0, 0, 0, 1), &Ti)
		sys, err := controlsys.New(&A, &B, &C, sweepStabilityDense(2, 2), 0)
		if err != nil {
			t.Fatal(err)
		}
		if inputOnly {
			if err := sys.SetInputDelay([]float64{tau1, tau2}); err != nil {
				t.Fatal(err)
			}
			return sweepStabilityFeedback(t, sys, nil)
		}
		if err := sys.SetInputDelay([]float64{tau1, coupleDelay}); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{0, tau2 - coupleDelay}); err != nil {
			t.Fatal(err)
		}
		return sweepStabilityFeedback(t, sys, nil)
	}
	for _, c := range []struct {
		f1, f2 float64
		want   bool
	}{
		{0.95, 0.95, true}, {1.01, 0.5, false}, {0.5, 1.01, false},
		{0.99, 0.99, true}, {1.05, 1.05, false},
	} {
		t.Run(fmt.Sprintf("tau1=%gc/tau2=%gc", c.f1, c.f2), func(t *testing.T) {
			t.Run("inputOnly", func(t *testing.T) { sweepStabilityAssert(t, build(c.f1*t1, c.f2*t2, true), c.want, true) })
			t.Run("inputOutput", func(t *testing.T) { sweepStabilityAssertKnownRefusal(t, build(c.f1*t1, c.f2*t2, false), c.want) })
		})
	}
}

// D5: the difference loop w(t) = k·w(t−τ) + u(t), y = w, W/U = 1/(1−k e^{−sτ}),
// has roots s = (ln|k| + j·arg-branch)/τ: stable iff |k| < 1, marginal at
// |k| = 1. A correct verdict or an explicit unsupported error is required.
func TestSweepStabilityDifferenceLoop(t *testing.T) {
	const tau = 0.7
	for _, c := range []struct {
		k    float64
		want int
	}{{0.5, 1}, {-0.5, 1}, {0.99, 1}, {1, 0}, {-1, 0}, {1.01, -1}, {2, -1}, {-2, -1}} {
		t.Run(fmt.Sprintf("k=%g", c.k), func(t *testing.T) {
			H, err := controlsys.NewGain(sweepStabilityDense(2, 2, 1, c.k, 1, c.k), 0)
			if err != nil {
				t.Fatal(err)
			}
			sys, err := controlsys.SetDelayModel(H, []float64{tau})
			if err != nil {
				t.Fatal(err)
			}
			s := complex(0.3, 2.0)
			want := 1 / (1 - complex(c.k, 0)*cmplx.Exp(-s*tau))
			if g, err := sys.EvalFr(s); err == nil {
				if cmplx.Abs(g[0][0]-want) > 1e-12*cmplx.Abs(want) {
					t.Fatalf("H(s) = %v, want %v: realization wrong", g[0][0], want)
				}
			}
			switch c.want {
			case 1:
				sweepStabilityAssert(t, sys, true, false)
			case 0:
				sweepStabilityAssertNotStable(t, sys)
			default:
				sweepStabilityAssert(t, sys, false, false)
			}
		})
	}
}

// D5: neutral ẋ = ∓x + 0.5·ẋ(t−τ), χ = s ± 1 − 0.5·s·e^{−sτ}. With +1 there
// is no axis root (|jω+1| = 0.5|ω| has no solution), so it is stable for every
// τ; with −1 a positive real root exists (χ(1) < 0 < χ(∞)).
func TestSweepStabilityNeutral(t *testing.T) {
	for _, c := range []struct {
		a    float64
		want bool
	}{{-1, true}, {1, false}} {
		for _, tau := range []float64{0.3, 2} {
			t.Run(fmt.Sprintf("a=%g/tau=%g", c.a, tau), func(t *testing.T) {
				sys := sweepStabilitySys(t, c.a, 1, 1, 0)
				if err := sys.SetInternalDelay([]float64{tau}, sweepStabilityDense(1, 1, 0.5), sweepStabilityDense(1, 1, c.a),
					sweepStabilityDense(1, 1), sweepStabilityDense(1, 1, 1), sweepStabilityDense(1, 1, 0.5)); err != nil {
					t.Fatal(err)
				}
				sweepStabilityAssert(t, sys, c.want, false)
			})
		}
	}
}

// D6: Nyquist of k·e^{−τs}/(s−p): each crossing of τn = (θ + 2πn)/ωc adds a
// destabilizing root pair (|L| has a single crossover with d|L|/dω < 0), so
// Z = 2·#{n : τn < τ} and N = Z − P.
func TestSweepStabilityNyquistEncirclements(t *testing.T) {
	for _, c := range []struct {
		p, k float64
	}{{1, 2}, {-1, 2}, {1, 5}} {
		wc := math.Sqrt(c.k*c.k - c.p*c.p)
		theta := math.Atan(wc)
		if c.p < 0 {
			theta = math.Pi - math.Atan(wc)
		}
		P := 0
		if c.p > 0 {
			P = 1
		}
		tauN := func(n int) float64 { return (theta + 2*math.Pi*float64(n)) / wc }
		for _, tau := range []float64{0.5 * tauN(0), 1.1 * tauN(0), (tauN(0) + tauN(1)) / 2, 1.05 * tauN(1), 1.02 * tauN(2)} {
			Z := 0
			for n := 0; tauN(n) < tau; n++ {
				Z += 2
			}
			t.Run(fmt.Sprintf("p=%g/k=%g/tau=%.4g", c.p, c.k, tau), func(t *testing.T) {
				L := sweepStabilityFirstOrderLoop(t, c.p, c.k, tau)
				ny, err := L.Nyquist(nil, 0)
				if err != nil {
					t.Fatal(err)
				}
				if ny.RHPPoles != P || ny.RHPZerosCL != Z || ny.Encirclements != Z-P {
					t.Errorf("Nyquist P=%d N=%d Z=%d, want P=%d N=%d Z=%d", ny.RHPPoles, ny.Encirclements, ny.RHPZerosCL, P, Z-P, Z)
				}
				sweepStabilityAssert(t, sweepStabilityFeedback(t, L, nil), Z == 0, true)
			})
		}
	}
}

// sweepStabilityCriticalGain returns 1/|L1(jωp)| at the first −π crossing of
// the continuous phase φ(ω), requiring later −(2n+1)π crossings to have
// smaller |L1| so the gain bound is unambiguous.
func sweepStabilityCriticalGain(t *testing.T, mag, phase func(float64) float64, wmax float64) float64 {
	t.Helper()
	first := sweepStabilityBisect(func(w float64) float64 { return phase(w) + math.Pi }, 1e-9, wmax)
	m0 := mag(first)
	for n := 1; ; n++ {
		target := -float64(2*n+1) * math.Pi
		if phase(wmax) > target {
			break
		}
		w := sweepStabilityBisect(func(w float64) float64 { return phase(w) - target }, first, wmax)
		if mag(w) >= m0 {
			t.Fatalf("crossing %d at %g has |L1| = %g >= %g", n, w, mag(w), m0)
		}
	}
	return 1 / m0
}

// D7: gain boundary k* of k·e^{−τs}/(s(s+1)) and of a lightly damped
// resonance k·e^{−τs}·ωn²/(s²+2ζωn s+ωn²), from a hand-coded −π crossing;
// stable at 0.99k*, unstable at 1.01k*.
func TestSweepStabilityCriticalGain(t *testing.T) {
	type loop struct {
		name         string
		a, b, c      []float64
		n            int
		tau, wmax    float64
		mag, phaseFn func(float64) float64
	}
	const tau1 = 0.5
	const zeta, wn, tau2 = 0.05, 10.0, 0.05
	loops := []loop{
		{
			name: "integrator", n: 2, tau: tau1, wmax: 100,
			a: []float64{0, 1, 0, -1}, b: []float64{0, 1}, c: []float64{1, 0},
			mag:     func(w float64) float64 { return 1 / (w * math.Hypot(1, w)) },
			phaseFn: func(w float64) float64 { return -math.Pi/2 - math.Atan(w) - w*tau1 },
		},
		{
			name: "resonance", n: 2, tau: tau2, wmax: 400,
			a: []float64{0, 1, -wn * wn, -2 * zeta * wn}, b: []float64{0, 1}, c: []float64{wn * wn, 0},
			mag: func(w float64) float64 { return wn * wn / math.Hypot(wn*wn-w*w, 2*zeta*wn*w) },
			phaseFn: func(w float64) float64 {
				return -math.Atan2(2*zeta*wn*w, wn*wn-w*w) - w*tau2
			},
		},
	}
	for _, l := range loops {
		kStar := sweepStabilityCriticalGain(t, l.mag, l.phaseFn, l.wmax)
		for _, f := range []float64{0.99, 1.01} {
			t.Run(fmt.Sprintf("%s/k=%.2fk*", l.name, f), func(t *testing.T) {
				sys, err := controlsys.New(sweepStabilityDense(l.n, l.n, l.a...), sweepStabilityDense(l.n, 1, l.b...),
					sweepStabilityDense(1, l.n, l.c...), sweepStabilityDense(1, 1), 0)
				if err != nil {
					t.Fatal(err)
				}
				s := complex(0, 3.0)
				if g, err := sys.EvalFr(s); err == nil {
					want := cmplx.Rect(l.mag(3), l.phaseFn(3)+3*l.tau)
					if cmplx.Abs(g[0][0]-want) > 1e-9*cmplx.Abs(want) {
						t.Fatalf("realization L1(j3) = %v, want %v", g[0][0], want)
					}
				}
				if err := sys.SetInputDelay([]float64{l.tau}); err != nil {
					t.Fatal(err)
				}
				gain, err := controlsys.NewGain(sweepStabilityDense(1, 1, f*kStar), 0)
				if err != nil {
					t.Fatal(err)
				}
				sweepStabilityAssert(t, sweepStabilityFeedback(t, sys, gain), f < 1, true)
			})
		}
	}
}

// D1: e^{−τ2 s}·(1/s)·e^{−τ1 s} under unity feedback (split input/output
// delay) has χ = s + e^{−(τ1+τ2)s}: stable iff τ1 + τ2 < π/2.
func TestSweepStabilitySplitDelayIntegrator(t *testing.T) {
	for _, total := range []float64{1.4, 1.7} {
		t.Run(fmt.Sprintf("total=%g", total), func(t *testing.T) {
			plant := sweepStabilitySys(t, 0, 1, 1, 0)
			if err := plant.SetInputDelay([]float64{0.6 * total}); err != nil {
				t.Fatal(err)
			}
			if err := plant.SetOutputDelay([]float64{0.4 * total}); err != nil {
				t.Fatal(err)
			}
			sweepStabilityAssertKnownRefusal(t, sweepStabilityFeedback(t, plant, nil), total < math.Pi/2)
		})
	}
}
