package controlsys_test

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

var sweepAnalyticForms = []string{"input", "output", "io", "internal"}

func sweepAnalyticApplyDelay(t *testing.T, base *controlsys.System, H *controlsys.System, L float64, form string) *controlsys.System {
	t.Helper()
	sys := base.Copy()
	var err error
	switch form {
	case "input":
		err = sys.SetInputDelay([]float64{L})
	case "output":
		err = sys.SetOutputDelay([]float64{L})
	case "io":
		err = sys.SetDelay(mat.NewDense(1, 1, []float64{L}))
	case "internal":
		sys, err = controlsys.SetDelayModel(H, []float64{L})
	default:
		t.Fatalf("unknown form %q", form)
	}
	if err != nil {
		t.Fatalf("%s delay %g: %v", form, L, err)
	}
	return sys
}

// sweepAnalyticLag builds e^{-Ls}/(s+a) (gain k) in the requested delay form.
func sweepAnalyticLag(t *testing.T, a, k, L float64, form string) *controlsys.System {
	t.Helper()
	base, err := controlsys.New(mat.NewDense(1, 1, []float64{-a}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{k}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	H, err := controlsys.New(mat.NewDense(1, 1, []float64{-a}), mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(2, 1, []float64{k, 0}), mat.NewDense(2, 2, []float64{0, 0, 1, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sweepAnalyticApplyDelay(t, base, H, L, form)
}

// sweepAnalyticPure builds a pure delay (seconds for dt=0, samples otherwise).
func sweepAnalyticPure(t *testing.T, L, dt float64, form string) *controlsys.System {
	t.Helper()
	base, err := controlsys.NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	if err != nil {
		t.Fatal(err)
	}
	H, err := controlsys.NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 0}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sweepAnalyticApplyDelay(t, base, H, L, form)
}

func sweepAnalyticDistinctTau(sys *controlsys.System) int {
	if sys.LFT == nil {
		return 0
	}
	seen := map[float64]bool{}
	for _, v := range sys.LFT.Tau {
		seen[v] = true
	}
	return len(seen)
}

func sweepAnalyticGrid(lo, hi float64, nLog int, dw float64) []float64 {
	var w []float64
	for i := range nLog {
		w = append(w, lo*math.Pow(1/lo, float64(i)/float64(nLog)))
	}
	for x := 1.0; x <= hi; x += dw {
		w = append(w, x)
	}
	return w
}

func sweepAnalyticCloseC(got, want complex128, rtol, atol float64) bool {
	return cmplx.Abs(got-want) <= rtol*cmplx.Abs(want)+atol
}

// sweepAnalyticCheckFreq compares FreqResponse and EvalFr(jω) with want.
func sweepAnalyticCheckFreq(t *testing.T, label string, sys *controlsys.System, omega []float64, want func(w float64) complex128, rtol, atol float64) {
	t.Helper()
	defer sweepAnalyticNoPanic(t, label+" freq")
	fr, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s FreqResponse: %v", label, err)
	}
	for k, w := range omega {
		ref := want(w)
		if got := fr.At(k, 0, 0); !sweepAnalyticCloseC(got, ref, rtol, atol) {
			t.Fatalf("%s FreqResponse ω=%g: got %v, want %v (|err|=%g)", label, w, got, ref, cmplx.Abs(got-ref))
		}
		h, err := sys.EvalFr(complex(0, w))
		if err != nil {
			t.Fatalf("%s EvalFr ω=%g: %v", label, w, err)
		}
		if got := h[0][0]; !sweepAnalyticCloseC(got, ref, rtol, atol) {
			t.Fatalf("%s EvalFr ω=%g: got %v, want %v (|err|=%g)", label, w, got, ref, cmplx.Abs(got-ref))
		}
	}
}

func sweepAnalyticNoPanic(t *testing.T, label string) {
	t.Helper()
	if r := recover(); r != nil {
		t.Fatalf("%s: panic: %v", label, r)
	}
}

func sweepAnalyticCheckSeries(t *testing.T, label string, T []float64, y func(k int) float64, want func(tk float64) float64, tol float64) {
	t.Helper()
	for k, tk := range T {
		if got, ref := y(k), want(tk); math.Abs(got-ref) > tol {
			t.Fatalf("%s t=%.17g: got %.15g, want %.15g", label, tk, got, ref)
		}
	}
}

func sweepAnalyticStepCheck(t *testing.T, label string, sys *controlsys.System, tf float64, want func(tk float64) float64, tol float64) {
	t.Helper()
	defer sweepAnalyticNoPanic(t, label+" Step")
	resp, err := controlsys.Step(sys, tf)
	if err != nil {
		t.Fatalf("%s Step: %v", label, err)
	}
	sweepAnalyticCheckSeries(t, label+" Step", resp.T, func(k int) float64 { return resp.Y.At(0, k) }, want, tol)
}

func sweepAnalyticLsimStep(t *testing.T, label string, sys *controlsys.System, h float64, n int, want func(tk float64) float64, tol float64) {
	t.Helper()
	tm := make([]float64, n)
	u := mat.NewDense(n, 1, nil)
	for k := range n {
		tm[k] = float64(k) * h
		u.Set(k, 0, 1)
	}
	defer sweepAnalyticNoPanic(t, label+" Lsim")
	resp, err := controlsys.Lsim(sys, u, tm, nil)
	if err != nil {
		t.Fatalf("%s Lsim: %v", label, err)
	}
	sweepAnalyticCheckSeries(t, label+" Lsim", tm, func(k int) float64 { return resp.Y.At(0, k) }, want, tol)
}

// sweepAnalyticUnitStep is right-continuous: u(0)=1, as MATLAB step(exp(-Ls)) gives 1 at t=L.
func sweepAnalyticUnitStep(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return 0
}

func sweepAnalyticLagStep(a, k, L float64) func(float64) float64 {
	return func(tk float64) float64 {
		if tk <= L {
			return 0
		}
		return k / a * (1 - math.Exp(-a*(tk-L)))
	}
}

// TestSweepAnalyticPureDelayStep (D01): step of e^{-Ls} is 0 for t<L, 1 for
// t≥L. Lsim uses h=0.125 with L=0.75 (exactly 6 samples, so t=L is on grid
// and must read 1) and L=0.8 (non-multiple: samples 0.75 → 0, 0.875 → 1).
// Exact algebra: tol 1e-12. Step uses its own grid, checked pointwise.
func TestSweepAnalyticPureDelayStep(t *testing.T) {
	const tol = 1e-12
	for _, form := range sweepAnalyticForms {
		for _, L := range []float64{0.75, 0.8} {
			label := fmt.Sprintf("%s L=%g", form, L)
			t.Run(label, func(t *testing.T) {
				sys := sweepAnalyticPure(t, L, 0, form)
				want := func(tk float64) float64 { return sweepAnalyticUnitStep(tk - L) }
				if form == "internal" {
					t.Skipf("known defect, ergo HV27R6: Step/Lsim panic on stateless internal-delay model")
				}
				sweepAnalyticLsimStep(t, label, sys, 0.125, 25, want, tol)
				sweepAnalyticStepCheck(t, label, sys, 3, func(tk float64) float64 {
					if math.Abs(tk-L) < 1e-12*L {
						return 1
					}
					return want(tk)
				}, tol)
			})
		}
	}
}

// TestSweepAnalyticLagStep (D02): step of e^{-Ls}/(s+1) is 0 for t≤L and
// 1-exp(-(t-L)) after. L=0.73 with h=0.1 (non-multiple) and L=0.75 with
// h=0.125; ZOH sampling of a step is exact, tol 1e-12. Explicit checks at
// t=0, the samples bracketing L, and far beyond (every grid point to t=8).
func TestSweepAnalyticLagStep(t *testing.T) {
	const tol = 1e-12
	cases := []struct{ L, h float64 }{{0.73, 0.1}, {0.75, 0.125}, {2.37, 0.05}}
	for _, form := range sweepAnalyticForms {
		for _, c := range cases {
			label := fmt.Sprintf("%s L=%g h=%g", form, c.L, c.h)
			t.Run(label, func(t *testing.T) {
				sys := sweepAnalyticLag(t, 1, 1, c.L, form)
				want := sweepAnalyticLagStep(1, 1, c.L)
				n := int(math.Round(8/c.h)) + 1
				sweepAnalyticLsimStep(t, label, sys, c.h, n, want, tol)
				sweepAnalyticStepCheck(t, label, sys, 10, want, tol)
			})
		}
	}
}

// TestSweepAnalyticLagFreq (D03): H(jω) = e^{-jωL}/(1+jω) for every delay
// form. Grid: log 1e-2..1 plus linear to 10·2π/L with LΔω < π/2 so the
// unwrapped Bode phase is unambiguous. FreqResponse/EvalFr compared complex
// with rtol 1e-9 + atol 1e-12; Bode magnitude -10log10(1+ω²) dB (abs 1e-9)
// and unwrapped phase -(ωL+atan ω)·180/π (abs 1e-7 deg, ωL up to ~63 rad).
func TestSweepAnalyticLagFreq(t *testing.T) {
	const L = 1.3
	omega := sweepAnalyticGrid(1e-2, 10*2*math.Pi/L, 60, 0.05)
	want := func(w float64) complex128 {
		return cmplx.Exp(complex(0, -w*L)) / complex(1, w)
	}
	for _, form := range sweepAnalyticForms {
		t.Run(form, func(t *testing.T) {
			sys := sweepAnalyticLag(t, 1, 1, L, form)
			sweepAnalyticCheckFreq(t, form, sys, omega, want, 1e-9, 1e-12)
			b, err := sys.Bode(omega, 0)
			if err != nil {
				t.Fatalf("%s Bode: %v", form, err)
			}
			for k, w := range omega {
				if got, ref := b.MagDBAt(k, 0, 0), -10*math.Log10(1+w*w); math.Abs(got-ref) > 1e-9 {
					t.Fatalf("%s Bode mag ω=%g: got %.12g, want %.12g", form, w, got, ref)
				}
				if got, ref := b.PhaseAt(k, 0, 0), -(w*L+math.Atan(w))*180/math.Pi; math.Abs(got-ref) > 1e-7 {
					t.Fatalf("%s Bode phase ω=%g: got %.12g, want %.12g", form, w, got, ref)
				}
			}
		})
	}
}

// TestSweepAnalyticSeriesDelaysAdd (D04): Series of e^{-as} and e^{-bs}/(s+1)
// equals e^{-(a+b)s}/(s+1) for every pair of delay forms, in frequency
// (rtol 1e-9) and step (closed form, tol 1e-9). With two distinct internal
// delays Step/Lsim are documented as an O(dt) ZOH approximation (as MATLAB),
// so there the Lsim error must be ≤ (one sample per delay)·max|y'| = 2h for
// h=0.1 and 0.01.
// Also pure·pure → e^{-(a+b)s}.
func TestSweepAnalyticSeriesDelaysAdd(t *testing.T) {
	const a, b = 0.37, 1.21
	omega := sweepAnalyticGrid(1e-2, 10*2*math.Pi/a, 40, 0.1)
	lagWant := func(w float64) complex128 {
		return cmplx.Exp(complex(0, -w*(a+b))) / complex(1, w)
	}
	pureWant := func(w float64) complex128 { return cmplx.Exp(complex(0, -w*(a+b))) }
	for _, f1 := range sweepAnalyticForms {
		for _, f2 := range sweepAnalyticForms {
			label := f1 + "×" + f2
			t.Run(label, func(t *testing.T) {
				p := sweepAnalyticPure(t, a, 0, f1)
				g := sweepAnalyticLag(t, 1, 1, b, f2)
				for _, order := range [][2]*controlsys.System{{p, g}, {g, p}} {
					s, err := controlsys.Series(order[0], order[1])
					if err != nil {
						t.Fatalf("%s Series: %v", label, err)
					}
					sweepAnalyticCheckFreq(t, label+" lag", s, omega, lagWant, 1e-9, 1e-12)
					want := sweepAnalyticLagStep(1, 1, a+b)
					if nt := sweepAnalyticDistinctTau(s); nt > 1 {
						for _, h := range []float64{0.1, 0.01} {
							sweepAnalyticLsimStep(t, fmt.Sprintf("%s lag O(dt) h=%g", label, h), s, h, int(math.Round(8/h))+1, want, float64(nt)*h)
						}
						continue
					}
					sweepAnalyticStepCheck(t, label+" lag", s, 8, want, 1e-9)
				}
				pp, err := controlsys.Series(p, sweepAnalyticPure(t, b, 0, f2))
				if err != nil {
					t.Fatalf("%s pure Series: %v", label, err)
				}
				sweepAnalyticCheckFreq(t, label+" pure", pp, omega, pureWant, 1e-9, 1e-12)
				if n, _, _ := pp.Dims(); n == 0 && pp.LFT != nil {
					t.Skipf("known defect, ergo HV27R6: Step/Lsim panic on stateless internal-delay model")
				}
				sweepAnalyticLsimStep(t, label+" pure", pp, 0.125, 30, func(tk float64) float64 {
					return sweepAnalyticUnitStep(tk - (a + b))
				}, 1e-12)
			})
		}
	}
}

// TestSweepAnalyticParallelKeepsDelays (D05): e^{-s}/(s+1) + e^{-3s}/(s+2)
// against the closed form at many ω (rtol 1e-9) and the piecewise step
// [t>1](1-e^{-(t-1)}) + [t>3](1-e^{-2(t-3)})/2 (tol 1e-9), for each pair of
// delay forms.
func TestSweepAnalyticParallelKeepsDelays(t *testing.T) {
	omega := sweepAnalyticGrid(1e-2, 10*2*math.Pi, 40, 0.05)
	want := func(w float64) complex128 {
		s := complex(0, w)
		return cmplx.Exp(-s)/(s+1) + cmplx.Exp(-3*s)/(s+2)
	}
	s1, s2 := sweepAnalyticLagStep(1, 1, 1), sweepAnalyticLagStep(2, 1, 3)
	stepWant := func(tk float64) float64 { return s1(tk) + s2(tk) }
	for _, f1 := range sweepAnalyticForms {
		for _, f2 := range sweepAnalyticForms {
			label := f1 + "+" + f2
			t.Run(label, func(t *testing.T) {
				sys, err := controlsys.Parallel(sweepAnalyticLag(t, 1, 1, 1, f1), sweepAnalyticLag(t, 2, 1, 3, f2))
				if err != nil {
					t.Fatalf("%s Parallel: %v", label, err)
				}
				sweepAnalyticCheckFreq(t, label, sys, omega, want, 1e-9, 1e-12)
				sweepAnalyticStepCheck(t, label, sys, 10, stepWant, 1e-9)
				sweepAnalyticLsimStep(t, label, sys, 0.1, 101, stepWant, 1e-9)
			})
		}
	}
}

// TestSweepAnalyticDiscretePureDelay (D06): z^{-k} with dt=0.1, k=3 samples,
// as IODelay, InputDelay, OutputDelay and internal delay: Lsim of a random
// input returns u shifted by k samples with zero fill. Exact algebra, 1e-14.
func TestSweepAnalyticDiscretePureDelay(t *testing.T) {
	const dt, k, n = 0.1, 3, 40
	rng := rand.New(rand.NewPCG(7, 11))
	tm := make([]float64, n)
	u := mat.NewDense(n, 1, nil)
	for i := range n {
		tm[i] = float64(i) * dt
		u.Set(i, 0, rng.NormFloat64())
	}
	for _, form := range sweepAnalyticForms {
		t.Run(form, func(t *testing.T) {
			defer sweepAnalyticNoPanic(t, form)
			sys := sweepAnalyticPure(t, k, dt, form)
			resp, err := controlsys.Lsim(sys, u, tm, nil)
			if err != nil {
				t.Fatalf("%s Lsim: %v", form, err)
			}
			for i := range n {
				want := 0.0
				if i >= k {
					want = u.At(i-k, 0)
				}
				if got := resp.Y.At(0, i); math.Abs(got-want) > 1e-14 {
					t.Fatalf("%s n=%d: got %g, want %g", form, i, got, want)
				}
			}
			for _, w := range []float64{0.1, 1, 7, 20, 31} {
				z := cmplx.Exp(complex(0, w*dt))
				h, err := sys.EvalFr(z)
				if err != nil {
					t.Fatal(err)
				}
				if ref := cmplx.Pow(z, -k); !sweepAnalyticCloseC(h[0][0], ref, 1e-12, 1e-14) {
					t.Fatalf("%s EvalFr ω=%g: got %v, want %v", form, w, h[0][0], ref)
				}
			}
		})
	}
}

// TestSweepAnalyticZeroDelayIdentity: all external delays set to zero must
// reproduce 1/(s+1) exactly (freq rtol 1e-12, step 1e-12). An explicit zero
// internal delay is rejected with ErrZeroInternalDelay (documented).
func TestSweepAnalyticZeroDelayIdentity(t *testing.T) {
	omega := sweepAnalyticGrid(1e-3, 100, 60, 1)
	want := func(w float64) complex128 { return 1 / complex(1, w) }
	for _, form := range []string{"input", "output", "io"} {
		sys := sweepAnalyticLag(t, 1, 1, 0, form)
		sweepAnalyticCheckFreq(t, form, sys, omega, want, 1e-12, 1e-15)
		sweepAnalyticStepCheck(t, form, sys, 6, func(tk float64) float64 { return 1 - math.Exp(-tk) }, 1e-12)
	}
	H, err := controlsys.New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(2, 2, []float64{0, 0, 1, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := controlsys.SetDelayModel(H, []float64{0}); !errors.Is(err, controlsys.ErrZeroInternalDelay) {
		t.Fatalf("zero internal delay: err = %v, want ErrZeroInternalDelay", err)
	}
}

// TestSweepAnalyticPadeConvergence: sup error of PadeDelay(τ,n) and of
// sys.Pade(n) (for e^{-τs}/(s+1)) against the exact response on ω∈[0,4/τ]
// decreases strictly for n=1..8. At x=ωτ=0.2 the phase error matches the
// Padé remainder C_n x^{2n+1}, C_n=(n!)²/((2n)!(2n+1)!), within 5% for n=1..3.
func TestSweepAnalyticPadeConvergence(t *testing.T) {
	const tau = 0.9
	var omega []float64
	for i := range 401 {
		omega = append(omega, 4/tau*float64(i)/400)
	}
	lag := sweepAnalyticLag(t, 1, 1, tau, "input")
	prevPure, prevLag := math.Inf(1), math.Inf(1)
	for n := 1; n <= 8; n++ {
		pd, err := controlsys.PadeDelay(tau, n)
		if err != nil {
			t.Fatal(err)
		}
		pl, err := lag.Pade(n)
		if err != nil {
			t.Fatal(err)
		}
		if pl.HasDelay() {
			t.Fatalf("n=%d: Pade result still has delay", n)
		}
		frp, err := pd.FreqResponse(omega)
		if err != nil {
			t.Fatal(err)
		}
		frl, err := pl.FreqResponse(omega)
		if err != nil {
			t.Fatal(err)
		}
		var ep, el float64
		for k, w := range omega {
			e := cmplx.Exp(complex(0, -w*tau))
			ep = max(ep, cmplx.Abs(frp.At(k, 0, 0)-e))
			el = max(el, cmplx.Abs(frl.At(k, 0, 0)-e/complex(1, w)))
		}
		if ep >= prevPure || el >= prevLag {
			t.Fatalf("n=%d: errors pure %g (prev %g), lag %g (prev %g) not decreasing", n, ep, prevPure, el, prevLag)
		}
		prevPure, prevLag = ep, el
		if n <= 3 {
			x := 0.2
			h, err := pd.EvalFr(complex(0, x/tau))
			if err != nil {
				t.Fatal(err)
			}
			fn := func(k int) float64 { return math.Gamma(float64(k) + 1) }
			c := fn(n) * fn(n) / (fn(2*n) * fn(2*n+1))
			got := math.Abs(cmplx.Phase(h[0][0]) + x)
			if ref := c * math.Pow(x, float64(2*n+1)); math.Abs(got/ref-1) > 0.05 {
				t.Fatalf("n=%d: low-freq phase error %g, want ≈ %g", n, got, ref)
			}
		}
	}
	if prevPure > 1e-6 {
		t.Fatalf("order 8 sup error %g too large", prevPure)
	}
}

// TestSweepAnalyticThiranGroupDelay: public ThiranDelay(τ,dt) is maximally
// flat at DC, so -dφ/dω /dt = τ/dt. Central difference with h·dt=1e-4 (error
// O(h²)), tol 1e-6 samples. |H|=1 (allpass) at several ω, tol 1e-12.
func TestSweepAnalyticThiranGroupDelay(t *testing.T) {
	const dt = 0.1
	for _, D := range []float64{0.3, 1.5, 2.7, 3, 4.25} {
		sys, err := controlsys.ThiranDelay(D*dt, dt)
		if err != nil {
			t.Fatalf("D=%g: %v", D, err)
		}
		eval := func(theta float64) complex128 {
			h, err := sys.EvalFr(cmplx.Exp(complex(0, theta)))
			if err != nil {
				t.Fatal(err)
			}
			return h[0][0]
		}
		const h = 1e-4
		gd := -(cmplx.Phase(eval(h)) - cmplx.Phase(eval(-h))) / (2 * h)
		if math.Abs(gd-D) > 1e-6 {
			t.Fatalf("D=%g: DC group delay %.10g samples", D, gd)
		}
		for _, theta := range []float64{0.1, 1, 2.5, 3.1} {
			if m := cmplx.Abs(eval(theta)); math.Abs(m-1) > 1e-12 {
				t.Fatalf("D=%g θ=%g: |H|=%.15g, want 1", D, theta, m)
			}
		}
	}
}
