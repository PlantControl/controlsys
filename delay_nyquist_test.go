package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
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

func polyEval(p []float64, s complex128) complex128 {
	var v complex128
	for _, c := range p {
		v = v*s + complex(c, 0)
	}
	return v
}

func polyMul(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			out[i+j] += x * y
		}
	}
	return out
}

func polyAdd(a, b []float64) []float64 {
	if len(a) < len(b) {
		a, b = b, a
	}
	out := slices.Clone(a)
	for i, y := range b {
		out[len(a)-len(b)+i] += y
	}
	return out
}

func polyScale(a []float64, k float64) []float64 {
	out := slices.Clone(a)
	for i := range out {
		out[i] *= k
	}
	return out
}

// quasiLoop is L(s) = num(s)/den(s)·e^{-τs}, deg num <= deg den, with
// characteristic quasi-polynomial χ(s) = den(s) + num(s)e^{-τs}.
type quasiLoop struct {
	name     string
	num, den []float64
	tau      float64
	box      float64 // |L(s)| < 1 on Re s >= 0, |s| >= box
	wmax     float64 // sensitivity oracle sweep end
	analytic int     // known verdict: 0 none, 1 stable, 2 unstable
}

func (q quasiLoop) L(w float64) complex128 {
	s := complex(0, w)
	return polyEval(q.num, s) / polyEval(q.den, s) * cmplx.Exp(-s*complex(q.tau, 0))
}

func (q quasiLoop) chi(s complex128) complex128 {
	return polyEval(q.den, s) + polyEval(q.num, s)*cmplx.Exp(-s*complex(q.tau, 0))
}

// system returns a controllable-canonical realization with input delay τ.
func (q quasiLoop) system(t *testing.T) *System {
	t.Helper()
	lead := q.den[0]
	den := polyScale(q.den, 1/lead)
	num := polyScale(q.num, 1/lead)
	n := len(den) - 1
	num = append(make([]float64, len(den)-len(num)), num...)
	d := num[0]
	a := make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	b := make([]float64, n)
	b[n-1] = 1
	c := make([]float64, n)
	for j := range n {
		a[(n-1)*n+j] = -den[n-j]
		c[j] = num[n-j] - d*den[n-j]
	}
	return delayedLoop(t, a, b, c, []float64{d}, n, q.tau)
}

// oracleRHPRoots counts the roots of χ in the open right half-plane by the
// argument principle on the rectangle [0, box]×[-box, box], sampled densely
// and bisected wherever arg χ moves by more than 0.2 between samples.
func oracleRHPRoots(t *testing.T, chi func(complex128) complex128, box float64) int {
	t.Helper()
	corners := []complex128{complex(0, -box), complex(box, -box), complex(box, box), complex(0, box), complex(0, -box)}
	total := 0.0
	var walk func(a, b complex128, fa, fb complex128, depth int)
	walk = func(a, b complex128, fa, fb complex128, depth int) {
		d := cmplx.Phase(fb / fa)
		if math.Abs(d) <= 0.2 || depth > 50 {
			if depth > 50 {
				t.Fatalf("oracle contour passes through a root near %v", a)
			}
			total += d
			return
		}
		m := (a + b) / 2
		fm := chi(m)
		walk(a, m, fa, fm, depth+1)
		walk(m, b, fm, fb, depth+1)
	}
	const n = 20000
	for e := range 4 {
		a, b := corners[e], corners[e+1]
		prev, fprev := a, chi(a)
		for i := 1; i <= n; i++ {
			z := a + (b-a)*complex(float64(i)/n, 0)
			fz := chi(z)
			walk(prev, z, fprev, fz, 0)
			prev, fprev = z, fz
		}
	}
	turns := total / (2 * math.Pi)
	if math.Abs(turns-math.Round(turns)) > 1e-6 {
		t.Fatalf("oracle winding %g not an integer", turns)
	}
	return int(math.Round(turns))
}

func pidDelayQuasiLoop(name string, k float64) quasiLoop {
	const kp, ki, kd, tf = 0.8, 0.3, 0.2, 0.05
	num := polyScale([]float64{kp*tf + kd, kp + ki*tf, ki}, k)
	den := polyMul([]float64{tf, 1, 0}, polyMul([]float64{1, 2, 1}, []float64{1, 1}))
	return quasiLoop{name: name, num: num, den: den, tau: 0.5, box: 60, wmax: 60}
}

// resonantQuasiLoop adds a lightly damped mode at 40 rad/s with |L| ≈ peak;
// tau = (π/2 + 2π)/40 points the resonance circle at -1.
func resonantQuasiLoop(name string, peak, tau float64) quasiLoop {
	const k0, wr, zeta = 0.5, 40.0, 0.01
	kr := 2 * zeta * peak
	res := []float64{1, 2 * zeta * wr, wr * wr}
	num := polyAdd(polyScale(res, k0), polyScale([]float64{wr * wr, wr * wr}, kr))
	return quasiLoop{name: name, num: num, den: polyMul([]float64{1, 1}, res), tau: tau, box: 200, wmax: 200}
}

func delayStabilityCases() []quasiLoop {
	var cases []quasiLoop
	const a, tau = 1.0, 0.5
	wc := bisectRoot(func(w float64) float64 { return w*tau + math.Atan(w/a) - math.Pi }, 1e-9, math.Pi/tau)
	kc := math.Hypot(a, wc)
	for _, b := range []float64{-1.2, -0.5, 0.5 * kc, 0.97 * kc, 1.03 * kc, 3 * kc, 9 * kc} {
		v := 2
		if -a < b && b < kc {
			v = 1
		}
		cases = append(cases, quasiLoop{name: fmt.Sprintf("hayes/a=1/b=%.4g", b), num: []float64{b}, den: []float64{1, a}, tau: tau, box: 4 * math.Abs(b), wmax: 100, analytic: v})
	}
	for _, b := range []float64{0.3, 0.8, 1.2, 2, 6} {
		cases = append(cases, quasiLoop{name: fmt.Sprintf("hayes/a=-0.5/b=%.4g", b), num: []float64{b}, den: []float64{1, -0.5}, tau: 0.3, box: 4 * b, wmax: 100})
	}
	for _, k := range []float64{0.5, 1, 2, 4, 8, 20} {
		cases = append(cases, pidDelayQuasiLoop(fmt.Sprintf("pid/k=%g", k), k))
	}
	aligned := (math.Pi/2 + 2*math.Pi) / 40
	for _, p := range []float64{0.8, 0.97, 1.03, 1.5, 4} {
		cases = append(cases, resonantQuasiLoop(fmt.Sprintf("resonance/peak=%g", p), p, aligned))
	}
	cases = append(cases, resonantQuasiLoop("resonance/misaligned", 4, 0.3))
	cases = append(cases,
		quasiLoop{name: "neutral/0.9", num: []float64{0.9, 0.95}, den: []float64{1, 1}, tau: 1, box: 20, wmax: 300},
		quasiLoop{name: "neutral/-0.9", num: []float64{-0.9, -0.95}, den: []float64{1, 1}, tau: 1, box: 20, wmax: 300},
		quasiLoop{name: "neutral/0.36", num: []float64{0.36, 1.02}, den: []float64{1, 2}, tau: 1, box: 20, wmax: 300},
		quasiLoop{name: "neutral/0.36-unstable", num: []float64{0.36, 3.72}, den: []float64{1, 2}, tau: 1, box: 20, wmax: 300},
		quasiLoop{name: "neutral/0.3-near1", num: []float64{0.3, 0.6, 0.95 * 41}, den: []float64{1, 2, 41}, tau: 0.2, box: 60, wmax: 300},
		quasiLoop{name: "axis/integrator-oscillator", num: []float64{0.05, 0.02}, den: []float64{1, 0, 1, 0}, tau: 0.1, box: 10, wmax: 50},
		quasiLoop{name: "axis/integrator-two-oscillators", num: []float64{-0.3, -0.15}, den: polyMul([]float64{1, 0, 1, 0}, []float64{1, 0, 9}), tau: 0.1, box: 10, wmax: 50},
		quasiLoop{name: "axis/oscillator-stable", num: []float64{-0.5}, den: []float64{1, 0, 1}, tau: 0.1, box: 10, wmax: 50},
		quasiLoop{name: "axis/oscillator-unstable", num: []float64{0.5}, den: []float64{1, 0, 1}, tau: 0.1, box: 10, wmax: 50},
		quasiLoop{name: "axis/double-integrator", num: []float64{0.2, 0.1}, den: []float64{1, 0, 0}, tau: 0.2, box: 10, wmax: 50},
		quasiLoop{name: "axis/double-integrator-unstable", num: []float64{2, 1}, den: []float64{1, 0, 0}, tau: 0.8, box: 10, wmax: 50},
	)
	return cases
}

// TestDelayLoopEncirclementsMatchRootOracle pins the closed-loop RHP root
// count rhp − W of the delay Nyquist test against an independent count of the
// roots of χ(s) = den(s) + num(s)e^{-τs}, and the sensitivity peaks against a
// dense sweep, for loops with |L| near 1 in the tail, high-frequency
// resonances and several imaginary-axis poles.
func TestDelayLoopEncirclementsMatchRootOracle(t *testing.T) {
	for _, q := range delayStabilityCases() {
		t.Run(q.name, func(t *testing.T) {
			want := oracleRHPRoots(t, q.chi, max(q.box, 1))
			if q.analytic != 0 && (want == 0) != (q.analytic == 1) {
				t.Fatalf("oracle roots %d contradict the analytic verdict", want)
			}
			sys := q.system(t)
			for _, frac := range []float64{0.5, 1e-3} {
				l, err := delayLoopFromSystem(sys, "delay loop")
				if err != nil {
					t.Fatal(err)
				}
				res, err := l.nyquist(frac)
				if err != nil {
					t.Fatalf("frac=%g: %v", frac, err)
				}
				if res.roots != want || res.stable != (want == 0) {
					t.Fatalf("frac=%g: roots=%d stable=%v, oracle %d", frac, res.roots, res.stable, want)
				}
			}
			for _, lin := range []float64{0.1, 0.3, 0.38} {
				l, err := delayLoopFromSystem(sys, "delay loop")
				if err != nil {
					t.Fatal(err)
				}
				res, err := l.nyquistLinearTo(1e-3, lin)
				if err != nil {
					t.Fatalf("lin=%g: %v", lin, err)
				}
				if res.roots != want {
					t.Fatalf("lin=%g: roots=%d, oracle %d", lin, res.roots, want)
				}
			}
			l, err := delayLoopFromSystem(sys, "delay loop")
			if err != nil {
				t.Fatal(err)
			}
			shifts := []float64{0, -0.5, 0.75, -1.75}
			stable, peaks, err := l.sensitivityPeaks(shifts...)
			if err != nil {
				t.Fatal(err)
			}
			if stable != (want == 0) {
				t.Fatalf("sensitivityPeaks stable=%v, oracle roots %d", stable, want)
			}
			if !stable {
				return
			}
			d := math.Abs(q.num[0] / q.den[0])
			if len(q.num) < len(q.den) {
				d = 0
			}
			Lp := func(w float64) complex128 { return q.L(math.Max(w, 1e-9)) }
			for i, c := range shifts {
				peak, _ := oraclePeakShifted(Lp, q.wmax, c)
				peak = math.Max(peak, diskBound(d, c))
				if math.Abs(peaks[i].peak-peak) > 1e-7*peak {
					t.Errorf("shift %g: peak=%.12g at %g, oracle %.12g", c, peaks[i].peak, peaks[i].w, peak)
				}
			}
		})
	}
}

// The PID loop of BenchmarkDiskMargin_PIDExactDelay needs R = 25600 for the
// 0.1% sensitivity tail, ~33k points at linear spacing π/(8τ); the tail bound
// certifies |L| <= 0.3 far earlier, so the linear grid stops there.
func TestDelayLoopSensitivityTailStopsLinearGrid(t *testing.T) {
	q := pidDelayQuasiLoop("pid", 1)
	l, err := delayLoopFromSystem(q.system(t), "delay loop")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	at := l.at
	l.at = func(w float64) complex128 { n++; return at(w) }
	stable, peaks, err := l.sensitivityPeaks(0, -0.5)
	if err != nil || !stable {
		t.Fatalf("stable=%v err=%v", stable, err)
	}
	if n > 3000 {
		t.Fatalf("%d evaluations, want <= 3000", n)
	}
	for i, c := range []float64{0, -0.5} {
		peak, _ := oraclePeakShifted(func(w float64) complex128 { return q.L(math.Max(w, 1e-9)) }, q.wmax, c)
		if math.Abs(peaks[i].peak-peak) > 1e-7*peak {
			t.Errorf("shift %g: peak=%.12g, oracle %.12g", c, peaks[i].peak, peak)
		}
	}
}
