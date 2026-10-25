package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func assertSweepMatchesPointwise(t *testing.T, label string, sys *System, omega []float64, relTol float64) {
	t.Helper()
	got, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	want, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponsePointwise: %v", label, err)
	}
	for k := range want.Data {
		g, w := got.Data[k], want.Data[k]
		if cmplx.IsInf(w) || cmplx.IsNaN(w) {
			if g != w && !(cmplx.IsInf(g) && cmplx.IsInf(w)) {
				t.Fatalf("%s: entry %d = %v, want %v", label, k, g, w)
			}
			continue
		}
		if d := cmplx.Abs(g - w); d > relTol*cmplx.Abs(w) {
			t.Fatalf("%s: entry %d (w=%g) rel diff %g", label, k, omega[k/(want.P*want.M)], d/cmplx.Abs(w))
		}
	}
}

// assertHessenbergMatchesPointwise holds the Hessenberg kernel itself to the
// pointwise path, for models the sweep tier would hand to dense GEPP.
func assertHessenbergMatchesPointwise(t *testing.T, label string, sys *System, omega []float64, relTol float64) {
	t.Helper()
	want, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponsePointwise: %v", label, err)
	}
	n, m, p := sys.Dims()
	hs := newHessenbergSweep(sys, n, m, p)
	td := newTimeDomain(sys.Dt)
	got := make([]complex128, p*m)
	for k, w := range omega {
		if err := hs.evalInto(td.frequencyPoint(w), got); err != nil {
			t.Fatalf("%s: w=%g: %v", label, w, err)
		}
		for i, g := range got {
			wv := want.Data[k*p*m+i]
			if d := cmplx.Abs(g - wv); d > relTol*cmplx.Abs(wv) {
				t.Fatalf("%s: w=%g entry %d rel diff %g", label, w, i, d/cmplx.Abs(wv))
			}
		}
	}
}

func stiffChain(t *testing.T, n int, lo, hi, dt float64) *System {
	t.Helper()
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, 2, nil)
	C := mat.NewDense(2, n, nil)
	for i := range n {
		A.Set(i, i, -lo*math.Pow(hi/lo, float64(i)/float64(n-1)))
		if i+1 < n {
			A.Set(i, i+1, 0.3*float64(i%3+1))
		}
		B.Set(i, 0, 1)
		B.Set(i, 1, math.Sin(float64(i)))
		C.Set(0, i, math.Cos(float64(i)))
		C.Set(1, i, 1/float64(i+1))
	}
	sys, err := New(A, B, C, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if dt == 0 {
		return sys
	}
	d, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// Regression for ELN5IG: the long-sweep transfer-function shortcut lost
// 5e-7 on this 18-state Padé cascade and O(1) on stiff chains.
func TestFreqResponseSweepPadeCascade(t *testing.T) {
	sys := absorbScopePlant(t, 0, false, false)
	absorbScopeCases[0].apply(t, sys, 0.125)
	pade, err := sys.Pade(5)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := pade.Dims(); n != 18 {
		t.Fatalf("n = %d, want 18", n)
	}
	assertSweepMatchesPointwise(t, "absorbScopeOmega", pade, absorbScopeOmega, 1e-12)
	assertSweepMatchesPointwise(t, "logspace", pade, logspace(-2, 3, 200), 1e-12)
}

func TestFreqResponseSweepStiffModels(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		lo, hi, dt float64
	}{
		{"n10", 10, 1e-2, 1e2, 0},
		{"n20", 20, 1e-3, 1e3, 0},
		{"n30", 30, 1e-3, 1e3, 0},
		{"n40", 40, 1e-2, 1e2, 0},
		{"discrete_n20", 20, 1e-3, 1e2, 0.01},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := stiffChain(t, tc.n, tc.lo, tc.hi, tc.dt)
			omega := logspace(-4, 4, 120)
			if tc.dt > 0 {
				omega = logspace(-4, math.Log10(math.Pi/tc.dt), 120)
			}
			assertSweepMatchesPointwise(t, tc.name, sys, omega, 1e-12)
			assertHessenbergMatchesPointwise(t, tc.name, sys, omega, 1e-12)
		})
	}
}

func TestFreqResponseSweepResonanceAndEdges(t *testing.T) {
	zeta, wn := 1e-4, 7.0
	reson, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	omega := logspace(0, 1, 101)
	omega = append(omega, wn, wn*(1+zeta), wn*(1-zeta))
	assertSweepMatchesPointwise(t, "resonance", reson, omega, 1e-12)

	first, err := New(mat.NewDense(1, 1, []float64{-3}), mat.NewDense(1, 2, []float64{1, -2}),
		mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 2, []float64{0.5, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "n1", first, logspace(-2, 2, 50), 1e-12)

	static, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "n0", static, logspace(-2, 2, 10), 0)

	integ, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, -2, 1, 0, 0, -5}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertSweepMatchesPointwise(t, "pole at w=0", integ, append([]float64{0}, logspace(-2, 2, 20)...), 1e-12)
}

func TestFreqResponseRejectsInvalidSystem(t *testing.T) {
	cases := map[string]func(*System){
		"Delay":       func(s *System) { s.Delay = mat.NewDense(3, 3, nil) },
		"InputDelay":  func(s *System) { s.InputDelay = []float64{1} },
		"OutputDelay": func(s *System) { s.OutputDelay = []float64{1, 2, 3} },
		"E":           func(s *System) { s.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1}) },
		"LFT.B2":      func(s *System) { s.LFT.B2 = mat.NewDense(2, 2, nil) },
		"C":           func(s *System) { s.C = mat.NewDense(2, 4, nil) },
	}
	for name, corrupt := range cases {
		sys := absorbScopePlant(t, 0, true, false)
		corrupt(sys)
		if _, err := sys.FreqResponse(absorbScopeOmega); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: FreqResponse err = %v", name, err)
		}
		if _, err := sys.FreqResponsePointwise(absorbScopeOmega); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: FreqResponsePointwise err = %v", name, err)
		}
		if _, err := sys.EvalFr(1i); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: EvalFr err = %v", name, err)
		}
	}
	static, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	static.LFT = &LFTDelay{Tau: []float64{0.5}, B2: mat.NewDense(1, 1, nil), D12: mat.NewDense(1, 1, nil), D21: mat.NewDense(1, 1, nil), D22: mat.NewDense(1, 1, nil)}
	if _, err := static.DCGain(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("static LFT with 1x1 B2: DCGain err = %v", err)
	}
	var nilSys *System
	if _, err := nilSys.FreqResponse(absorbScopeOmega); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil: FreqResponse err = %v", err)
	}
}

// A descriptor mode at 1-10⁻⁶ (or -10⁻⁶) leaves zÊ-Â entries far smaller
// than zÊ near the pole. Forming them as round(round(zÊ)-Â) loses
// ε|zÊ|/|zÊ-Â| ≈ 10⁻¹⁰ of the dominant response; a fused multiply-add keeps
// one rounding.
func TestBalancedDenseDescriptorNearPole(t *testing.T) {
	for _, dt := range []float64{0, 0.05} {
		slow := 1e-6
		pole := -slow
		if dt > 0 {
			pole = 1 - slow
		}
		a0 := mat.NewDense(3, 3, []float64{
			pole, 0, 0,
			0.3, 0.5, -0.2,
			-0.7, 0.1, 0.4,
		})
		if dt == 0 {
			a0.Set(1, 1, -0.5)
			a0.Set(2, 2, -0.4)
		}
		E := mat.NewDense(3, 3, []float64{
			0.7, 0, 0,
			0.11, 1.3, -0.23,
			0.05, 0.17, 0.9,
		})
		b0 := mat.NewDense(3, 2, []float64{1, -0.4, 0.6, 0.8, -0.3, 1.1})
		var A, B mat.Dense
		A.Mul(E, a0)
		B.Mul(E, b0)
		C := mat.NewDense(2, 3, []float64{1.2, -0.5, 0.3, 0.4, 0.9, -1})
		D := mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3})
		sys, err := NewDescriptor(&A, &B, C, D, E, dt)
		if err != nil {
			t.Fatal(err)
		}
		td := newTimeDomain(dt)
		bd := newBalancedDense(sys, 3, 2, 2)
		got := make([]complex128, 4)
		for _, w := range []float64{1e-6, 2e-5, 1e-2} {
			if err := bd.evalInto(td.frequencyPoint(w), got); err != nil {
				t.Fatal(err)
			}
			want := frExactResponse(sys, frExactPoint(w, dt))
			norm, diff := 0.0, 0.0
			for i, v := range want {
				norm = max(norm, cmplx.Abs(v))
				diff = max(diff, cmplx.Abs(got[i]-v))
			}
			if e := diff / norm; e > 1e-12 {
				t.Errorf("dt=%g ω=%g: relative error %g", dt, w, e)
			}
		}
	}
}

// lightlyDampedContinuousSys is a 3×2 continuous model with D ≠ 0, a pole
// pair at −ζω ± jω (ζ = 1e-9) for each omega, a stable real pole, the first
// state feeding the last so A is not upper Hessenberg, and, for extra > 0,
// extra well-damped pairs.
func lightlyDampedContinuousSys(t *testing.T, rng *rand.Rand, omegas []float64, extra int) *System {
	t.Helper()
	n := 2*(len(omegas)+extra) + 1
	m, p := 2, 3
	A := mat.NewDense(n, n, nil)
	for k := range len(omegas) + extra {
		w, sigma := 0.0, 0.0
		if k < len(omegas) {
			w = omegas[k]
			sigma = -1e-9 * w
		} else {
			w = 10 * rng.Float64()
			sigma = -0.3 * w
		}
		A.Set(2*k, 2*k, sigma)
		A.Set(2*k, 2*k+1, w*1.5)
		A.Set(2*k+1, 2*k, -w/1.5)
		A.Set(2*k+1, 2*k+1, sigma)
	}
	A.Set(n-1, n-1, -2)
	A.Set(n-1, 0, 0.7)
	for i := range n - 1 {
		for j := i + 2; j < n-1; j++ {
			A.Set(i, j, 0.3*rng.NormFloat64())
		}
	}
	B := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	D := mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			C.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, 0.5+rng.Float64())
		}
	}
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// The continuous analog of TestFreqResponseDiscreteLightlyDampedPeaks: at
// a pole pair with ζ = 1e-9 the pencil jωI − A is exact, but GEPP pivots
// the coupling row into the resonant rows and the Hessenberg reduction mixes
// them, which leaves about ε/ζ relative error unless refined against the
// exact pencil. Both kernels and both entry points must match a 200-bit
// evaluation.
func TestFreqResponseContinuousLightlyDampedPeaks(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	omegas := []float64{0.02, 1.3, 40}
	frob := func(g []complex128) float64 {
		v := 0.0
		for _, x := range g {
			v += real(x)*real(x) + imag(x)*imag(x)
		}
		return math.Sqrt(v)
	}
	var omega []float64
	for _, w0 := range omegas {
		for k := -4; k <= 4; k++ {
			omega = append(omega, w0*(1+float64(k)*0x1p-52))
		}
	}
	for _, extra := range []int{0, 12} {
		sys := lightlyDampedContinuousSys(t, rng, omegas, extra)
		n, _, _ := sys.Dims()
		resps := map[string]*FreqResponseMatrix{}
		var err error
		if resps["FreqResponse"], err = sys.FreqResponse(omega); err != nil {
			t.Fatal(err)
		}
		if resps["FreqResponsePointwise"], err = sys.FreqResponsePointwise(omega); err != nil {
			t.Fatal(err)
		}
		for k, w := range omega {
			want := frob(frExactResponse(sys, frExactPoint(w, 0)))
			for name, resp := range resps {
				got := frob(resp.Data[k*6 : (k+1)*6])
				if e := math.Abs(got-want) / want; !(e <= 1e-13) {
					t.Errorf("n=%d %s ω=%v: |G| = %.17g, exact %.17g, rel err %.3g", n, name, w, got, want, e)
				}
			}
		}
	}
}

// A small pivot that GEPP resolves exactly, as at an integrator's pole-limit
// sample point, must not be refined: refine's first-correction check
// leaves x bit-identical.
func TestBalancedDenseRefineLeavesAccurateSolution(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, 0, -2, -3}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	bd := newBalancedDense(sys, 3, 1, 1)
	got := make([]complex128, 1)
	if err := bd.evalInto(pointAt(1e-5), got); err != nil {
		t.Fatal(err)
	}
	x := append([]complex128(nil), bd.x...)
	bd.refine()
	for i := range x {
		if bd.x[i] != x[i] {
			t.Fatalf("refine changed x[%d] from %v to %v", i, x[i], bd.x[i])
		}
	}
	want := frExactResponse(sys, frBigC(1e-5))[0]
	if e := cmplx.Abs(got[0]-want) / cmplx.Abs(want); e > 1e-15 {
		t.Errorf("G(1e-5) = %v, exact %v, rel err %.3g", got[0], want, e)
	}
}

// nearDoublePoleCases (PZMG3F) are companion realizations with a near-double
// pole at z = 1 and a frequency where GEPP's solution was off.
var nearDoublePoleCases = []struct {
	num, den []float64
	dt, w    float64
}{
	{[]float64{-1.726944008929345, -1.5131820042093045}, []float64{1, -2.4409333068225934, 1.5714018141979766, 0.1799962921160343, -0.3104647994843685}, 0.1, 4.913970576589073e-05},
	{[]float64{1.5841368901875659, -1.4026199636681365}, []float64{1, -2.348244102065107, 1.6964882041386697, -0.3482441020680517}, 1, 3.1339155517751782e-06},
}

func nearDoublePoleSystem(tb testing.TB, i int) (*System, []float64) {
	tb.Helper()
	c := nearDoublePoleCases[i]
	r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
	if err != nil {
		tb.Fatal(err)
	}
	omega := []float64{c.w}
	for k := -200; k <= 200; k++ {
		omega = append(omega, c.w*(1+float64(k)*1e-14))
	}
	return r.Sys, omega
}

func BenchmarkFreqResponseNearDoublePole(b *testing.B) {
	for i := range nearDoublePoleCases {
		sys, omega := nearDoublePoleSystem(b, i)
		b.Run(fmt.Sprintf("case%d", i), func(b *testing.B) {
			for b.Loop() {
				if _, err := sys.FreqResponse(omega); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// PZMG3F: next to a near-double pole on the unit circle a working-precision
// residual of GEPP's solution can round to zero, and refine skipped
// refinement with x off by 3e-4 relative (wrong Im sign) at isolated
// frequencies. Oracle: the stored realization solved in big.Float.
func TestBalancedDenseNearDoublePoleRefines(t *testing.T) {
	for i, c := range nearDoublePoleCases {
		sys, omega := nearDoublePoleSystem(t, i)
		for _, pointwise := range []bool{false, true} {
			eval := sys.FreqResponse
			if pointwise {
				eval = sys.FreqResponsePointwise
			}
			fr, err := eval(omega)
			if err != nil {
				t.Fatalf("case %d: %v", i, err)
			}
			for k, w := range omega {
				want := frExactResponse(sys, frExactPoint(w, c.dt))[0]
				if e := cmplx.Abs(fr.Data[k]-want) / cmplx.Abs(want); e > 1e-10 {
					t.Errorf("case %d pointwise %v: G(%.17g) = %v, exact %v, rel err %.3g", i, pointwise, w, fr.Data[k], want, e)
				}
			}
		}
	}
}

// reversedCompanion returns A with first column −den[1:] and a unit
// superdiagonal, the reversed observable companion of den.
func reversedCompanion(den []float64) []float64 {
	n := len(den) - 1
	a := make([]float64, n*n)
	for i := range n {
		a[i*n] = -den[i+1]
		if i+1 < n {
			a[i*n+i+1] = 1
		}
	}
	return a
}

// LEAZRP: below the poles of a graded reversed companion C = e₁ reads only
// x₀ ≈ 1e-20 beside x₁ ≈ 2, so GEPP's normwise error swamped the response
// (6.9e4 relative at ω = 0.0176) although den and num resolve it to ~13ε,
// and refinement stopped on its normwise test. Every entry point must match
// the exact rational response.
func TestFrequencyResponseGradedReversedCompanionBelowPoles(t *testing.T) {
	den := []float64{1, 754, 255356, 5.1226392e+07, 6.755561344e+09, 6.140269664e+11, 3.9174454525952e+13, 1.749820945778688e+15, 5.344039379091456e+16, 1.0575501408690831e+18, 1.2115752641409319e+19, 6.01702493877751e+19, 0}
	b0 := []float64{0.05078125, 0, -2.6702880859375e-05, -8.821487426757812e-06, -524288, -0.0072021484375, 0.21875, 1.7642974853515625e-05, 3.375, 2.288818359375e-05, 1.1444091796875e-05, -15.25}
	c0 := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	n := len(den) - 1
	a, b, c := companionShape(den, b0, c0, 1)
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	num := ratObservableNum(den, b0, c0)
	ws := []float64{1e-5, 1e-4, 1.76e-3, 1.76e-2, 0.08, 1.38, 108}
	sweep, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	pw, err := sys.FreqResponsePointwise(ws)
	if err != nil {
		t.Fatal(err)
	}
	hs := newHessenbergSweep(sys, n, 1, 1)
	for k, w := range ws {
		ex := ratRatioJ(num, den, w)
		g, err := sys.EvalFr(complex(0, w))
		if err != nil {
			t.Fatal(err)
		}
		h := make([]complex128, 1)
		if err := hs.evalInto(pointAt(complex(0, w)), h); err != nil {
			t.Fatal(err)
		}
		for _, got := range []struct {
			name string
			v    complex128
		}{{"EvalFr", g[0][0]}, {"FreqResponse", sweep.At(k, 0, 0)}, {"FreqResponsePointwise", pw.At(k, 0, 0)}, {"hessenbergSweep", h[0]}} {
			if e := cmplx.Abs(got.v-ex) / cmplx.Abs(ex); e > 1e-12 {
				t.Errorf("%s at w=%g off exact by %.3g", got.name, w, e)
			}
		}
	}
}

// stalledCompanion25 is a 25-state graded reversed companion, C = e₁, on
// which refinement stalls below its poles.
func stalledCompanion25(t *testing.T) (*System, int) {
	t.Helper()
	den := []float64{1, 1960.54296875, 1324778.0375671387, 453017973.58403015, 90672566995.1926, 11432861553837.39, 949716471541430.8, 5.3628407407340744e+16, 2.109198001835061e+18, 5.8960716105277686e+19, 1.1918233492782816e+21, 1.767078704390106e+22, 1.9434022535784184e+23, 1.5980782540760173e+24, 9.868327675548808e+24, 4.5762879802036216e+25, 1.5853874470748784e+26, 4.054058535754468e+26, 7.490006007724049e+26, 9.649212545235104e+26, 8.173161151038438e+26, 4.1141976943341275e+26, 1.0271973116920471e+26, 1.0051652236654178e+25, 3.0487172137841685e+23, 2.8278471496166277e+21}
	b := []float64{-6.866455078125e-05, 0, 272, 0, 0, 0.017578125, 0, -9, 0, -5.7220458984375e-05, 0, -0.125, 0, -0.00390625, 0, 0, 0, -2, -256, 0, 0, 0, 0, 0, 0}
	n := len(den) - 1
	c := make([]float64, n)
	c[0] = 1
	sys, err := NewFromSlices(n, 1, 1, reversedCompanion(den), b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys, n
}

// Where refinement stalls (a 25-state graded companion, ε·cond ≳ 1), the
// sens bound of the trim gate must cover the reference's own error, so an
// inaccurate reference never counts as resolved; it was the data
// conditioning alone, 2.6e-14 relative beside an error of 27 at the first
// point.
func TestSensBoundCoversStalledSolve(t *testing.T) {
	sys, n := stalledCompanion25(t)
	bd := newBalancedDense(sys, n, 1, 1)
	sw := newSensWork(n, 1, 1)
	for _, w := range []float64{2.309734638458412e-05, 2.21300266697768e-05, 1.8709073417783588e-05, 0.000101680294768261, 0.0009890718650590099, 0.09303031026002107} {
		s := complex(0, w)
		g := make([]complex128, 1)
		if err := bd.evalInto(pointAt(s), g); err != nil {
			t.Fatalf("w=%g: %v", w, err)
		}
		bnd := sw.bound(bd, s)
		ex := frExactResponse(sys, frBigC(s))[0]
		e, scale := cmplx.Abs(g[0]-ex), cmplx.Abs(ex)
		if e > max(10*bnd[0], 1e-9*scale) {
			t.Errorf("w=%g: reference off exact by %.3g relative, sens bound %.3g", w, e/scale, bnd[0]/scale)
		}
	}
}

// CKTRTW: on stalledCompanion25 the Hessenberg sweep accepted refinement
// whose output correction passed only at the resolution floor, x₀ being
// ε-small beside the largest state, and FreqResponse was 2.2e10, 3.6e-5 and
// 6.4e-7 relative off at the first three points where EvalFr is exact. A
// stalled point must take the dense solve.
func TestHessenbergSweepStallTakesDense(t *testing.T) {
	sys, n := stalledCompanion25(t)
	ws := []float64{2.309734638458412e-05, 0.007690823083080494, 0.00931887964638536}
	for k := 3; k < 40; k++ {
		ws = append(ws, 1e-3*float64(k))
	}
	if (frequencyEvaluator{sys: sys, n: n, m: 1, p: 1}).useDenseSweep(len(ws)) {
		t.Logf("FreqResponse takes the dense solver for %d points here", len(ws))
	}
	sweep, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	hs := newHessenbergSweep(sys, n, 1, 1)
	h := make([]complex128, 1)
	for k, w := range ws {
		ex := frExactResponse(sys, frBigC(complex(0, w)))[0]
		if err := hs.evalInto(pointAt(complex(0, w)), h); err != nil {
			t.Fatal(err)
		}
		for name, v := range map[string]complex128{"FreqResponse": sweep.At(k, 0, 0), "hessenbergSweep": h[0]} {
			if e := cmplx.Abs(v-ex) / cmplx.Abs(ex); e > 1e-10 {
				t.Errorf("%s at w=%.17g off exact by %.3g", name, w, e)
			}
		}
	}
}

// CKTRTW (#23): on amd64 the sweep was 2.6e-4 and 8.3e-5 relative off on
// this transposed graded companion (unit superdiagonal, last row −den
// reversed, B = e₁₄) where EvalFr is 4e-16 and the perturbation condition
// is 4: the last correction reached the output through ĈQ, whose reduction
// error swamps what this graded Ĉ cancels.
func TestHessenbergSweepTransposedCompanionMatchesExact(t *testing.T) {
	const n = 14
	last := []float64{-7.073762343067559e25, -1.558731796419265e26, -1.3175471662496587e26, -8.677710890794021e25, -2.645822728268286e25, -5.414438168429777e24, -7.512558301352139e23, -6.406381165983646e21, -3.0820683117072744e19, -5.839490363739341e16, -5.7657809436672e13, -3.750166528e10, -1.7804908e7, -5655.60498046875}
	a := make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	copy(a[(n-1)*n:], last)
	b := make([]float64, n)
	b[n-1] = 1
	c := []float64{0, -15, 4, 3, 0, 0, 0, 1, 0.9375, 11, 0, -0.5, 3.25, -13}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	ws := []float64{0.030679185069885945, 0.009773307733924387}
	for k := 2; k < 40; k++ {
		ws = append(ws, 1e-3*float64(k))
	}
	sweep, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	hs := newHessenbergSweep(sys, n, 1, 1)
	h := make([]complex128, 1)
	for k, w := range ws {
		ex := frExactResponse(sys, frBigC(complex(0, w)))[0]
		if err := hs.evalInto(pointAt(complex(0, w)), h); err != nil {
			t.Fatal(err)
		}
		for name, v := range map[string]complex128{"FreqResponse": sweep.At(k, 0, 0), "hessenbergSweep": h[0]} {
			if e := cmplx.Abs(v-ex) / cmplx.Abs(ex); e > 1e-12 {
				t.Errorf("%s at w=%.17g off exact by %.3g", name, w, e)
			}
		}
	}
}

// GEPP on a reversed graded companion below its poles left x₁ exactly 0,
// the only state C reads, and outputCancels skipped the exact zero, the
// extreme of cancellation: every entry point returned 0 beside an exact
// 4.06e-26 that den and num determine to a few ε.
func TestFrequencyResponseReversedCompanionExactZeroState(t *testing.T) {
	den := []float64{1, 38836.14453125, 7.6326432e8, 1.3158335905792e13, 1.289109780782121e17, 1.0746583724948676e21, 4.3269613883408094e24, 1.1031075037031345e28, 9.569098187117596e29}
	b := []float64{0, -0.75, -4, -7.5, 0, -2, -0.875, 1}
	c := []float64{0, 1, 0, 0, 0, 0, 0, 0}
	n := len(den) - 1
	sys, err := NewFromSlices(n, 1, 1, reversedCompanion(den), b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	num := ratObservableNum(den, []float64{1, -0.875, -2, 0, -7.5, -4, -0.75, 0}, []float64{0, 0, 0, 0, 0, 0, 1, 0})
	ws := []float64{1.7302345030553948e-4, 1e-5, 1e-3}
	sweep, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	pw, err := sys.FreqResponsePointwise(ws)
	if err != nil {
		t.Fatal(err)
	}
	hs := newHessenbergSweep(sys, n, 1, 1)
	for k, w := range ws {
		ex := frExactResponse(sys, frBigC(complex(0, w)))[0]
		if e := cmplx.Abs(ratRatioJ(num, den, w)-ex) / cmplx.Abs(ex); e > 1e-13 {
			t.Fatalf("w=%g: oracles disagree by %.3g", w, e)
		}
		g, err := sys.EvalFr(complex(0, w))
		if err != nil {
			t.Fatal(err)
		}
		h := make([]complex128, 1)
		if err := hs.evalInto(pointAt(complex(0, w)), h); err != nil {
			t.Fatal(err)
		}
		for _, got := range []struct {
			name string
			v    complex128
		}{{"EvalFr", g[0][0]}, {"FreqResponse", sweep.At(k, 0, 0)}, {"FreqResponsePointwise", pw.At(k, 0, 0)}, {"hessenbergSweep", h[0]}} {
			if e := cmplx.Abs(got.v-ex) / cmplx.Abs(ex); e > 1e-12 {
				t.Errorf("%s at w=%g: %v off exact %v by %.3g", got.name, w, got.v, ex, e)
			}
		}
	}
}

// A block-diagonal MIMO model leaves each output's states exactly zero
// for the other input. Lifting the w > 0 guard of outputCancels must not
// refine such structural zeros or mark them stalled.
func TestBalancedDenseStructuralZeroNotStalled(t *testing.T) {
	a := []float64{
		-1, 2, 0, 0,
		-3, -4, 0, 0,
		0, 0, -5, 1,
		0, 0, -2, -6,
	}
	b := []float64{1, 0, 0.5, 0, 0, 1, 0, 2}
	c := []float64{1, 1, 0, 0, 0, 0, 1, -1}
	sys, err := NewFromSlices(4, 2, 2, a, b, c, make([]float64, 4), 0)
	if err != nil {
		t.Fatal(err)
	}
	bd := newBalancedDense(sys, 4, 2, 2)
	g := make([]complex128, 4)
	for _, w := range []float64{0.01, 1, 100} {
		if err := bd.evalInto(pointAt(complex(0, w)), g); err != nil {
			t.Fatal(err)
		}
		if g[1] != 0 || g[2] != 0 || bd.stalled {
			t.Errorf("w=%g: off-diagonal %v %v, stalled %v", w, g[1], g[2], bd.stalled)
		}
		if bd.outputCancels() {
			t.Errorf("w=%g: structural zero flagged as cancelling", w)
		}
		dd := make([]complex128, 4)
		if !ddResponseInto(sys, pointAt(complex(0, w)), dd) {
			t.Fatalf("w=%g: double-double solve failed", w)
		}
		ex := frExactResponse(sys, frBigC(complex(0, w)))
		for i, v := range dd {
			if cmplx.Abs(v-ex[i]) > 1e-13*cmplx.Abs(ex[i]) {
				t.Errorf("w=%g: double-double entry %d %v, exact %v", w, i, v, ex[i])
			}
		}
	}
}

// Below the poles of a graded tf2ss companion refine stalls and the dense
// solve was off by up to 0.5 relative (4e-2 at ω = 0.00139) although the
// double-double solve is exact there; a stalled point whose error estimate
// exceeds refineCondition·ε of an entry must take the double-double
// response at every entry point.
func TestFrequencyResponseStalledRefineTakesDD(t *testing.T) {
	r := []float64{3033.55, 8.635e6, 1.3387e10, 1.1713e13, 6.3856e15, 6.1226e17, 2.768e19, 7.2368e20, 1.1678e22, 1.5195e23, 1.0695e24, 6.2747e24, 1.6832e25, 3.404e25, 3.5338e25, 2.4398e25}
	n := len(r)
	a := make([]float64, n*n)
	for j, v := range r {
		a[j] = -v
	}
	for i := 1; i < n; i++ {
		a[i*n+i-1] = 1
	}
	b := make([]float64, n)
	b[14] = 1
	c := []float64{1, .25, 1.875, -.109375, -.125, 3, -2, 0, -.015625, -8, -1.5, 0, -.75, 0, 0, 0}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	ws := []float64{1e-4, 0.0013941527, 0.01, 0.30392}
	sweep, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	pw, err := sys.FreqResponsePointwise(ws)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range ws {
		s := complex(0, w)
		ex := frExactResponse(sys, frBigC(s))[0]
		dd := make([]complex128, 1)
		if !ddResponseInto(sys, pointAt(s), dd) {
			t.Fatalf("w=%g: double-double solve failed", w)
		}
		if e := cmplx.Abs(dd[0]-ex) / cmplx.Abs(ex); e > 1e-13 {
			t.Fatalf("w=%g: oracles disagree by %.3g", w, e)
		}
		g, err := sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for _, got := range []struct {
			name string
			v    complex128
		}{{"EvalFr", g[0][0]}, {"FreqResponse", sweep.At(k, 0, 0)}, {"FreqResponsePointwise", pw.At(k, 0, 0)}} {
			if e := cmplx.Abs(got.v-ex) / cmplx.Abs(ex); e > 1e-12 {
				t.Errorf("%s at w=%g: %v off exact %v by %.3g", got.name, w, got.v, ex, e)
			}
		}
	}
}

// stalledSweepRoots draws n poles graded over four decades, real or
// complex pairs.
func stalledSweepRoots(r *rand.Rand, n int) []complex128 {
	var roots []complex128
	for deg := 0; deg < n; {
		mag := math.Ldexp(float64(16+r.IntN(48))/32, r.IntN(14))
		if r.IntN(3) == 0 && deg+2 <= n {
			roots = append(roots, complex(-mag*float64(1+r.IntN(8))/8, mag))
			deg += 2
			continue
		}
		roots = append(roots, complex(-mag, 0))
		deg++
	}
	return roots
}

// Seeded graded companions (observable and tf2ss shapes, B a unit vector,
// dense C): EvalFr against the exact response must resolve every point at
// which the dense solve stalls, and never be worse than the dense solve.
func TestEvalFrStalledRefineSweep(t *testing.T) {
	skipSlowUnderRace(t)
	r := rand.New(rand.NewPCG(24, 16))
	stalls := 0
	for trial := range 300 {
		n := 6 + r.IntN(13)
		den := ratPolyFromRoots(stalledSweepRoots(r, n))
		b, c := make([]float64, n), make([]float64, n)
		b[r.IntN(n)] = 1
		for k := range n {
			if r.IntN(3) > 0 {
				c[k] = float64(r.IntN(33)-16) / 8
			}
		}
		shape := []int{0, 3}[r.IntN(2)]
		a, bt, ct := companionShape(den, b, c, shape)
		sys, err := NewFromSlices(n, 1, 1, a, bt, ct, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		bd := newBalancedDense(sys, n, 1, 1)
		for _, w := range []float64{1e-4, 1.3e-3, 0.0088, 0.107, 0.3, 3, 100} {
			s := complex(0, w)
			ex := frExactResponse(sys, frBigC(s))[0]
			g := make([]complex128, 1)
			if ex == 0 || bd.evalInto(pointAt(s), g) != nil {
				continue
			}
			ev, err := sys.EvalFr(s)
			if err != nil {
				t.Errorf("trial %d w=%g: %v", trial, w, err)
				continue
			}
			eDense := cmplx.Abs(g[0]-ex) / cmplx.Abs(ex)
			e := cmplx.Abs(ev[0][0]-ex) / cmplx.Abs(ex)
			if bd.stalled {
				stalls++
				if e > 1e-12 {
					t.Errorf("trial %d n %d shape %d w=%g: stalled point off exact by %.3g", trial, n, shape, w, e)
				}
			}
			if e > max(2*eDense, 1e-11) {
				t.Errorf("trial %d n %d shape %d w=%g: EvalFr off exact by %.3g, dense solve %.3g", trial, n, shape, w, e, eDense)
			}
		}
	}
	if stalls == 0 {
		t.Error("no stalled point in the sweep")
	}
}
