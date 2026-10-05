package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func delayedLoop(t *testing.T, a, b, c, d []float64, n int, tau float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, 1, 1, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	return sys
}

// bisectRoot finds the root of f on [lo, hi] given a sign change.
func bisectRoot(f func(float64) float64, lo, hi float64) float64 {
	flo := f(lo)
	for range 200 {
		mid := (lo + hi) / 2
		if fm := f(mid); (fm < 0) == (flo < 0) {
			lo, flo = mid, fm
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// oraclePeakS returns sup |1/(1+L(jω))| by a dense grid and golden section.
func oraclePeakS(l func(w float64) complex128, wmax float64) (float64, float64) {
	return oraclePeakShifted(l, wmax, 0)
}

// oracleDiskAlpha returns 1/sup |1/(1+L(jω)) − 1/2|, the balanced disk
// margin, given the high-frequency limit |L(j∞)| = tail of a delayed loop.
func oracleDiskAlpha(l func(w float64) complex128, wmax, tail float64) float64 {
	peak, _ := oraclePeakShifted(l, wmax, -0.5)
	return 1 / math.Max(peak, (1+tail)/(2*(1-tail)))
}

// oraclePeakShifted returns sup |1/(1+L(jω)) + c| by a dense grid and
// ternary search.
func oraclePeakShifted(l func(w float64) complex128, wmax, c float64) (float64, float64) {
	f := func(w float64) float64 { return cmplx.Abs(1/(1+l(w)) + complex(c, 0)) }
	const n = 400000
	best, bw := 0.0, 0.0
	for i := range n + 1 {
		w := wmax * float64(i) / n
		if v := f(w); v > best {
			best, bw = v, w
		}
	}
	a, b := math.Max(bw-wmax/n, 0), bw+wmax/n
	for range 200 {
		m1, m2 := a+(b-a)/3, b-(b-a)/3
		if f(m1) < f(m2) {
			a = m1
		} else {
			b = m2
		}
	}
	w := (a + b) / 2
	return f(w), w
}

func TestDiskMarginDelayedFirstOrderBoundary(t *testing.T) {
	const a, tau = 1.0, 0.5
	wc := bisectRoot(func(w float64) float64 { return w*tau + math.Atan(w/a) - math.Pi }, 1e-9, math.Pi/tau)
	kc := math.Hypot(a, wc)
	for _, tc := range []struct {
		k      float64
		stable bool
	}{{0.97 * kc, true}, {1.03 * kc, false}, {0.2 * kc, true}, {-0.5, true}, {-1.2, false}} {
		sys := delayedLoop(t, []float64{-a}, []float64{1}, []float64{tc.k}, []float64{0}, 1, tau)
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatalf("k=%g: %v", tc.k, err)
		}
		if (dm.Alpha > 0) != tc.stable {
			t.Fatalf("k=%g (kc=%g): alpha=%g, want stable=%v", tc.k, kc, dm.Alpha, tc.stable)
		}
		if !tc.stable {
			if dm.GainMargin != [2]float64{1, 1} || dm.PhaseMargin != 0 {
				t.Fatalf("k=%g: unstable result %+v", tc.k, dm)
			}
			continue
		}
		L := func(w float64) complex128 {
			s := complex(0, w)
			return complex(tc.k, 0) * cmplx.Exp(-s*complex(tau, 0)) / (s + complex(a, 0))
		}
		ms, wp := oraclePeakS(L, 200)
		alpha := oracleDiskAlpha(L, 200, 0)
		if math.Abs(dm.PeakSensitivity-ms) > 1e-7*ms || math.Abs(dm.Alpha-alpha) > 1e-7*alpha {
			t.Fatalf("k=%g: Ms=%.12g alpha=%.12g, oracle Ms=%.12g alpha=%.12g", tc.k, dm.PeakSensitivity, dm.Alpha, ms, alpha)
		}
		if ms > 1+1e-6 && math.Abs(dmPeakFreq(dm)-wp) > 1e-4*max(1, wp) {
			t.Fatalf("k=%g: wPeak=%g, oracle %g", tc.k, dmPeakFreq(dm), wp)
		}
	}
}

func TestDiskMarginDelayedIntegrator(t *testing.T) {
	const tau = 0.4
	for _, tc := range []struct {
		k      float64
		stable bool
	}{{0.95 * math.Pi / 2 / tau, true}, {1.05 * math.Pi / 2 / tau, false}, {0.5, true}} {
		sys := delayedLoop(t, []float64{0}, []float64{1}, []float64{tc.k}, []float64{0}, 1, tau)
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatalf("k=%g: %v", tc.k, err)
		}
		if (dm.Alpha > 0) != tc.stable {
			t.Fatalf("k=%g: alpha=%g, want stable=%v", tc.k, dm.Alpha, tc.stable)
		}
		if tc.stable {
			L := func(w float64) complex128 {
				s := complex(0, w)
				return complex(tc.k, 0) * cmplx.Exp(-s*complex(tau, 0)) / s
			}
			Lp := func(w float64) complex128 { return L(math.Max(w, 1e-9)) }
			ms, _ := oraclePeakS(Lp, 100)
			if math.Abs(dm.PeakSensitivity-ms) > 1e-7*ms {
				t.Fatalf("k=%g: Ms=%.12g, oracle %.12g", tc.k, dm.PeakSensitivity, ms)
			}
			if alpha := oracleDiskAlpha(Lp, 100, 0); math.Abs(dm.Alpha-alpha) > 1e-7*alpha {
				t.Fatalf("k=%g: alpha=%.12g, oracle %.12g", tc.k, dm.Alpha, alpha)
			}
		}
	}
}

// L = k e^{-τs}/(s-a): one open-loop RHP pole, stable for a < k < kc where
// the phase crossover solves ωτ = atan(ω/a).
func TestDiskMarginDelayedUnstableOpenLoop(t *testing.T) {
	const a, tau = 1.0, 0.3
	wc := bisectRoot(func(w float64) float64 { return w*tau - math.Atan(w/a) }, 1e-6, math.Pi/2/tau)
	kc := math.Hypot(a, wc)
	for _, tc := range []struct {
		k      float64
		stable bool
	}{{0.9 * a, false}, {(a + kc) / 2, true}, {1.05 * kc, false}} {
		sys := delayedLoop(t, []float64{a}, []float64{1}, []float64{tc.k}, []float64{0}, 1, tau)
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatalf("k=%g: %v", tc.k, err)
		}
		if (dm.Alpha > 0) != tc.stable {
			t.Fatalf("k=%g (a=%g kc=%g): alpha=%g, want stable=%v", tc.k, a, kc, dm.Alpha, tc.stable)
		}
	}
}

// Non-symmetric second-order realization of k(s+3)/((s+1)(s+2)) e^{-τs}.
func TestDiskMarginDelayedSecondOrderNonSymmetric(t *testing.T) {
	const tau = 0.7
	G := func(w float64) complex128 {
		s := complex(0, w)
		return (s + 3) / ((s + 1) * (s + 2)) * cmplx.Exp(-s*complex(tau, 0))
	}
	wc := bisectRoot(func(w float64) float64 { return imag(G(w)) }, 0.5, 3)
	kc := -1 / real(G(wc))
	for _, tc := range []struct {
		k      float64
		stable bool
	}{{0.98 * kc, true}, {1.02 * kc, false}} {
		sys := delayedLoop(t, []float64{0, 1, -2, -3}, []float64{0, 1}, []float64{3 * tc.k, tc.k}, []float64{0}, 2, tau)
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatal(err)
		}
		if (dm.Alpha > 0) != tc.stable {
			t.Fatalf("k=%g (kc=%g): alpha=%g, want stable=%v", tc.k, kc, dm.Alpha, tc.stable)
		}
		if tc.stable {
			L := func(w float64) complex128 { return complex(tc.k, 0) * G(w) }
			ms, _ := oraclePeakS(L, 100)
			if math.Abs(dm.PeakSensitivity-ms) > 1e-7*ms {
				t.Fatalf("Ms=%.12g, oracle %.12g", dm.PeakSensitivity, ms)
			}
			if alpha := oracleDiskAlpha(L, 100, 0); math.Abs(dm.Alpha-alpha) > 1e-7*alpha {
				t.Fatalf("alpha=%.12g, oracle %.12g", dm.Alpha, alpha)
			}
			for _, sigma := range []float64{-1.5, 1, 2.5} {
				got, err := DiskMarginSkew(sys, sigma)
				if err != nil {
					t.Fatal(err)
				}
				peak, wp := oraclePeakShifted(L, 100, (sigma-1)/2)
				peak = math.Max(peak, math.Abs((sigma+1)/2))
				if math.Abs(got.Alpha-1/peak) > 1e-7/peak || (peak > math.Abs((sigma+1)/2)+1e-6 && math.Abs(dmFreq(got)-wp) > 1e-4*max(1, wp)) {
					t.Fatalf("sigma=%g: alpha=%.12g at %g, oracle %.12g at %g", sigma, got.Alpha, dmFreq(got), 1/peak, wp)
				}
			}
		}
	}
}

// L = (0.5s+0.4)/(s+1) e^{-0.2s}: |L| rises monotonically to |D| = 0.5, so
// sup|S| = 1/(1-0.5) is the unattained high-frequency limit.
func TestDiskMarginDelayedFeedthroughLimit(t *testing.T) {
	sys := delayedLoop(t, []float64{-1}, []float64{1}, []float64{-0.1}, []float64{0.5}, 1, 0.2)
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dm.PeakSensitivity-2) > 1e-12 || !math.IsInf(dmPeakFreq(dm), 1) {
		t.Fatalf("Ms=%g at %g, want 2 at +Inf", dm.PeakSensitivity, dmPeakFreq(dm))
	}
	if math.Abs(dm.Alpha-2.0/3) > 1e-12 || !math.IsInf(dmFreq(dm), 1) {
		t.Fatalf("alpha=%g at %g, want 2/3 = 1/sup|(1-L)/(2(1+L))| at +Inf", dm.Alpha, dmFreq(dm))
	}
}

// L = (0.5 + 0.1/(s+1)) e^{-0.2s}: the finite-frequency peak exceeds the
// high-frequency limit 2.
func TestDiskMarginDelayedFeedthroughPeak(t *testing.T) {
	sys := delayedLoop(t, []float64{-1}, []float64{1}, []float64{0.1}, []float64{0.5}, 1, 0.2)
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		return (0.5 + 0.1/(s+1)) * cmplx.Exp(-0.2*s)
	}
	ms, _ := oraclePeakS(L, 2000)
	if math.Abs(dm.PeakSensitivity-ms) > 1e-9*ms || ms <= 2 {
		t.Fatalf("Ms=%.12g, oracle %.12g", dm.PeakSensitivity, ms)
	}
	if alpha := oracleDiskAlpha(L, 2000, 0.5); math.Abs(dm.Alpha-alpha) > 1e-9*alpha || 1/alpha <= 1.5 {
		t.Fatalf("alpha=%.12g, oracle %.12g", dm.Alpha, alpha)
	}
}

func TestDiskMarginDelayedRepresentationsAgree(t *testing.T) {
	const k, a, tau = 2.0, 1.5, 0.25
	base := delayedLoop(t, []float64{-a}, []float64{1}, []float64{k}, []float64{0}, 1, tau)
	out, err := NewFromSlices(1, 1, 1, []float64{-a}, []float64{1}, []float64{k}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := out.SetOutputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	io, err := NewFromSlices(1, 1, 1, []float64{-a}, []float64{1}, []float64{k}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	io.Delay = mat.NewDense(1, 1, []float64{tau})
	lft, err := base.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	if !lft.HasInternalDelay() {
		t.Fatal("expected internal delay")
	}
	want, err := DiskMargin(base)
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"output": out, "iodelay": io, "lft": lft} {
		got, err := DiskMargin(sys)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if math.Abs(got.Alpha-want.Alpha) > 1e-9 || math.Abs(dmPeakFreq(got)-dmPeakFreq(want)) > 1e-6 {
			t.Fatalf("%s: %+v, want %+v", name, got, want)
		}
	}
}

func TestDiskMarginDelayedUnsupported(t *testing.T) {
	neutral := delayedLoop(t, []float64{-1}, []float64{1}, []float64{1}, []float64{1}, 1, 0.1)
	if _, err := DiskMargin(neutral); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("neutral: err = %v", err)
	}
	cyclic, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	cyclic.LFT = &LFTDelay{
		Tau: []float64{0.1},
		B2:  mat.NewDense(1, 1, []float64{1}), C2: mat.NewDense(1, 1, []float64{1}),
		D12: mat.NewDense(1, 1, []float64{1}), D21: mat.NewDense(1, 1, []float64{0}),
		D22: mat.NewDense(1, 1, []float64{0}),
	}
	if _, err := DiskMargin(cyclic); !errors.Is(err, ErrContinuousInternalDelay) {
		t.Fatalf("cyclic: err = %v", err)
	}
}

// An uncontrollable integrator is a closed-loop pole at s=0 that the Nyquist
// plot of L cannot see; χ = det(sI-A)(1+L) catches it.
func TestDiskMarginDelayedHiddenAxisMode(t *testing.T) {
	sys := delayedLoop(t, []float64{0, 0, 0, -1}, []float64{0, 1}, []float64{1, 1}, []float64{0}, 2, 0.1)
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if dm.Alpha != 0 {
		t.Fatalf("alpha=%g, want 0 for hidden marginal mode", dm.Alpha)
	}
}

// L = k e^{-τs}/(s²+1): open-loop poles on the axis at ±j. The characteristic
// s²+1+k e^{-τs} crosses the axis only at k = 0 (ω = 1) and k = -1 (s = 0)
// for |k| < 1-(π/τ)², so -1 < k < 0 is the stable range.
func TestDiskMarginDelayedOscillatorAxisPoles(t *testing.T) {
	for _, tc := range []struct {
		k      float64
		stable bool
	}{{-0.5, true}, {-0.95, true}, {0.5, false}, {-1.5, false}} {
		sys := delayedLoop(t, []float64{0, 1, -1, 0}, []float64{0, 1}, []float64{tc.k, 0}, []float64{0}, 2, 0.1)
		dm, err := DiskMargin(sys)
		if err != nil {
			t.Fatalf("k=%g: %v", tc.k, err)
		}
		if (dm.Alpha > 0) != tc.stable {
			t.Fatalf("k=%g: alpha=%g, want stable=%v", tc.k, dm.Alpha, tc.stable)
		}
	}
}

func TestDelayLoopNyquistBudgetIsErrorNotUnstable(t *testing.T) {
	const k = 0.95
	l := &delayLoop{
		at: func(w float64) complex128 {
			s := complex(0, w)
			return complex(k, 0) * cmplx.Exp(-50*s) / (s + 1)
		},
		tail:      func(w float64) float64 { return k / math.Max(w-1, 1e-300) },
		tailLimit: 0,
		scales:    []float64{1},
	}
	stable, err := l.stableClosedLoop()
	if err != nil || !stable {
		t.Fatalf("default budget: stable=%v err=%v, want stable", stable, err)
	}
	l.maxPoints = 3
	stable, err = l.stableClosedLoop()
	if !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("exhausted budget: stable=%v err=%v, want ErrDelayUnsupported", stable, err)
	}
}

func TestDelayLoopEvaluationFailureIsError(t *testing.T) {
	sys := delayedLoop(t, []float64{-1, 2, 0, -3}, []float64{0, 1}, []float64{1, 0.5}, []float64{0.2}, 2, 0.5)
	for _, peaks := range []bool{false, true} {
		l, err := delayLoopFromSystem(sys, "delay loop")
		if err != nil {
			t.Fatal(err)
		}
		at := l.at
		l.at = func(w float64) complex128 {
			if w > 1 {
				return l.eval.fail(w, ErrSingularTransform)
			}
			return at(w)
		}
		if peaks {
			_, _, err = l.sensitivityPeaks(0, -0.5)
		} else {
			_, err = l.stableClosedLoop()
		}
		if !errors.Is(err, ErrSingularTransform) {
			t.Errorf("peaks=%v: err = %v, want ErrSingularTransform", peaks, err)
		}
	}
}
