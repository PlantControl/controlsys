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

// hugeDiagonalModel is the upper bidiagonal A with diagonal -k, -2k, …
// and superdiagonal k, B = bScale·1 with B₀ = 0 where coupled, and
// C = 1ᵀ/bScale.
func hugeDiagonalModel(t *testing.T, n int, k, bScale float64, coupled bool) *System {
	t.Helper()
	a := make([]float64, n*n)
	b := make([]float64, n)
	c := make([]float64, n)
	for i := range n {
		a[i*n+i] = -float64(i+1) * k
		if i+1 < n {
			a[i*n+i+1] = k
		}
		b[i], c[i] = bScale, 1/bScale
	}
	if coupled {
		b[0] = 0
	}
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, 1, b), mat.NewDense(1, n, c), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// ECC62L: with the diagonal skipped, balancing a 1e300 model is finite but
// leaves D⁻¹x below the normal range (x̂₀ ≈ 1.7e-400 at 1i), so the
// response takes the verified double-double solve of the unbalanced
// realization. Coupled has B̂₀ = 0: x̂₀ is driven only through â₀₁ and
// underflows the same way. Mid scales do not trigger it and stay on the
// double path.
func TestEvalFr_HugeDiagonalBalanceNoOverflow(t *testing.T) {
	for _, coupled := range []bool{false, true} {
		for _, n := range []int{2, 3} {
			for _, k := range []float64{1e3, 1e100, 1e300} {
				t.Run(fmt.Sprintf("coupled=%v/n=%d/k=%g", coupled, n, k), func(t *testing.T) {
					sys := hugeDiagonalModel(t, n, k, 1, coupled)
					omega := []float64{1, k}
					fr, err := sys.FreqResponse(omega)
					if err != nil {
						t.Fatalf("FreqResponse: %v", err)
					}
					for i, w := range omega {
						s := complex(0, w)
						want := bigSISOResponse(sys, s)
						got := map[string]complex128{"FreqResponse": fr.At(i, 0, 0)}
						if g, err := sys.EvalFr(s); err != nil {
							t.Errorf("EvalFr(%gi): %v", w, err)
						} else {
							got["EvalFr"] = g[0][0]
						}
						dst := make([]complex128, 1)
						if err := newBalancedDense(sys, n, 1, 1).evalInto(pointAt(s), dst); err != nil {
							t.Errorf("dense(%gi): %v", w, err)
						} else {
							got["dense"] = dst[0]
						}
						if err := newHessenbergSweep(sys, n, 1, 1).evalInto(pointAt(s), dst); err != nil {
							t.Errorf("hessenberg(%gi): %v", w, err)
						} else {
							got["hessenberg"] = dst[0]
						}
						for name, g := range got {
							if rel := cmplx.Abs(g-want) / cmplx.Abs(want); !(rel <= 1e-12) {
								t.Errorf("%s(%gi) = %v, want %v (rel %.3g)", name, w, g, want, rel)
							}
						}
					}
				})
			}
		}
	}
}

// ECC62L: where the unbalanced states underflow too, the fallback cannot
// verify and the point fails instead of returning the lost-state value.
// DZSEVY: scaling each B column by its own power of two instead of σ keeps
// B/σ ≈ 3.7e-331 in range, so the point now resolves; it must match the
// oracle or fail.
func TestEvalFr_StateUnderflowVerifiedOrFails(t *testing.T) {
	sys := hugeDiagonalModel(t, 2, 1e300, 1e-30, false)
	want := bigSISOResponse(sys, 1i)
	g, err := sys.EvalFr(1i)
	if err != nil {
		if !errors.Is(err, ErrOverflow) {
			t.Errorf("EvalFr(1i): %v, want ErrOverflow", err)
		}
	} else if rel := cmplx.Abs(g[0][0]-want) / cmplx.Abs(want); !(rel <= 1e-12) {
		t.Errorf("EvalFr(1i) = %v, want %v (rel %.3g)", g[0][0], want, rel)
	}
}

// ECC62L: balancing scaled the invariant diagonal by 1/f then f, so a
// diagonal above ~5.5e157 overflowed at f = 2⁻⁵⁰⁰. The diagonal of D⁻¹AD and D⁻¹ED is invariant and must stay
// bit-identical through balancing.
func TestBalancedRealization_DiagonalInvariant(t *testing.T) {
	const k = 1e300
	tests := []struct {
		name string
		n    int
		a, e []float64
		b, c []float64
	}{
		{"explicit huge", 2, []float64{-k, k, 0, -2 * k}, nil, []float64{1, 1}, []float64{1, 1}},
		{"descriptor huge", 2, []float64{-1, 1, 0, -2}, []float64{k, k, 0, 2 * k}, []float64{1, 1}, []float64{1, 1}},
		{"graded mid", 3, []float64{-1e3, 1e3, 1e-2, 3, -2e3, 1e6, 1e-6, 7, -3e3}, []float64{1, 1e-8, 0, 2, 3, 0, 1e9, 0, 5}, []float64{1, 1e-9, 1e9}, []float64{1e9, 1, 1e-9}},
	}
	for _, tc := range tests {
		br := balancedRealization{n: tc.n, m: 1, p: 1, a: append([]float64(nil), tc.a...), b: append([]float64(nil), tc.b...), c: append([]float64(nil), tc.c...)}
		if tc.e != nil {
			br.e = append([]float64(nil), tc.e...)
		}
		br.balance(nil)
		if !br.finite() {
			t.Errorf("%s: non-finite balanced realization a=%v e=%v", tc.name, br.a, br.e)
		}
		for i := range tc.n {
			d := i*tc.n + i
			if br.a[d] != tc.a[d] || tc.e != nil && br.e[d] != tc.e[d] {
				t.Errorf("%s: diagonal %d changed: a=%v e=%v", tc.name, i, br.a, br.e)
			}
		}
	}
}

// ECC62L: an exactly zero state at a transmission zero satisfies its pencil
// rows and is not a lost state: s/(s²+1) at s = 0 stays 0 with nil error.
func TestEvalFr_TransmissionZeroStateNotUnderflow(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{0, -1, 1, 0}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.EvalFr(0)
	if err != nil || g[0][0] != 0 {
		t.Errorf("EvalFr(0) = %v, %v; want 0, nil", g, err)
	}
}

// ECC62L: states no input reaches stay exactly zero and are masked out of
// the underflow screen; a reachable chain is not.
func TestBalancedRealization_UndrivenMask(t *testing.T) {
	br := balancedRealization{n: 3, m: 2, p: 1,
		a: []float64{-1, 2, 0, 0, -2, 0, 0, 0, -3},
		b: []float64{0, 0, 1, 0, 0, 0},
		c: []float64{1, 1, 1}}
	if got, want := br.undrivenMask(br.m, 1), uint64(1<<1|1<<3|1<<4|1<<5); got != want {
		t.Errorf("row-major mask = %b, want %b", got, want)
	}
	if got, want := br.undrivenMask(1, br.n), uint64(1<<2|1<<3|1<<4|1<<5); got != want {
		t.Errorf("column-major mask = %b, want %b", got, want)
	}
}

// ECC62L: the fallback's dd GEPP fills the exactly zero, undriven state 0
// with roundoff once pivoting mixes its row; verification must accept that
// unread noise rather than demand a componentwise residual of that row.
func TestEvalFr_UnderflowFallbackZeroStateFill(t *testing.T) {
	sys, err := New(
		mat.NewDense(4, 4, []float64{
			-1.214442610429162e+290, 0, 0, 0,
			0, -1.986329444554168e+290, -9.779371965707476e+289, 0,
			1.8549166562589078e+290, -8.319566527649257e+289, -1.8874556859501172e+290, 0,
			2.0837908894259076e+290, -3.4030527121515706e+290, 0, -1.2573724467060058e+290}),
		mat.NewDense(4, 1, []float64{0, 0.5871774780013506, -0.4878070560341442, -1.834676564228113}),
		mat.NewDense(1, 4, []float64{0.79772142433102, -0.2428722708167741, -0.11053151151669649, 0.2568107334791625}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{1, 1e287, 1e290, 1e291} {
		s := complex(0, w)
		want := bigSISOResponse(sys, s)
		g, err := sys.EvalFr(s)
		if err != nil {
			t.Errorf("EvalFr(%gi): %v", w, err)
			continue
		}
		if rel := cmplx.Abs(g[0][0]-want) / cmplx.Abs(want); !(rel <= 1e-12) {
			t.Errorf("EvalFr(%gi) = %v, want %v (rel %.3g)", w, g[0][0], want, rel)
		}
	}
}

// DZSEVY: the double-double fallback solves the balanced realization scaled
// by σ ≈ ‖A‖, so on hugeDiagonalModel at 1e300 B̂₀/σ ≈ 1e-100/1e300 flushes
// to zero and the state the output reads most is lost at every s; it
// returned 5e-301 for 2e-300 at 1i. A lost state takes the verified
// unbalanced solve, which resolves bScale = 1 and must fail for 1e-30.
func TestDDResponse_BalancedStateUnderflow(t *testing.T) {
	const k = 1e300
	for _, bScale := range []float64{1, 1e-30} {
		for _, coupled := range []bool{false, true} {
			for _, n := range []int{2, 3} {
				sys := hugeDiagonalModel(t, n, k, bScale, coupled)
				for _, s := range []complex128{1i, -k * (1 + 1e-15), -2 * k * (1 + 4e-16)} {
					want := bigSISOResponse(sys, s)
					dd := make([]complex128, 1)
					ok := ddResponseInto(sys, pointAt(s), dd)
					rel := cmplx.Abs(dd[0]-want) / cmplx.Abs(want)
					if ok && !(rel <= 1e-12) || !ok && bScale == 1 {
						t.Errorf("bScale=%g coupled=%v n=%d s=%v: ddResponseInto = %v, %v; want %v (rel %.3g)", bScale, coupled, n, s, dd[0], ok, want, rel)
					}
				}
			}
		}
	}
}

// DZSEVY: next to a pole the double solve is singular and EvalFr takes the
// double-double response, which returned the lost-state 1e-300 for
// -2.24e-285 at s = -1e300·(1+1e-15). The point is within working precision
// of the pole, so the pole response +Inf is the documented answer; a finite
// value must match the oracle.
func TestEvalFr_NearPoleBalancedStateUnderflow(t *testing.T) {
	for _, bScale := range []float64{1, 1e-30} {
		sys := hugeDiagonalModel(t, 2, 1e300, bScale, false)
		s := complex(-1e300*(1+1e-15), 0)
		want := bigSISOResponse(sys, s)
		g, err := sys.EvalFr(s)
		if err != nil {
			if !errors.Is(err, ErrOverflow) {
				t.Errorf("bScale=%g: EvalFr(%v): %v, want ErrOverflow", bScale, s, err)
			}
			continue
		}
		if v := g[0][0]; !cmplx.IsInf(v) && !(cmplx.Abs(v-want) <= 1e-12*cmplx.Abs(want)) {
			t.Errorf("bScale=%g: EvalFr(%v) = %v, want %v or +Inf", bScale, s, v, want)
		}
	}
}

// DZSEVY: a pair at ±j·1e250 behind a 1e300 state. Every |ω| ≤ 1e285 is
// within normwise working precision of the pair, the double solve is
// singular, and the double-double fallback's complex division x·ȳ/|y|²
// underflows (B/σ ≈ 7e-301 times a 7e-51 pivot), losing every state; the
// response was 0 with nil error. It must match the oracle, fail with
// ErrOverflow, or be the pole response +Inf: every such ω is within
// 10ε·‖A‖ of the pair, the pole rule (XKU3SI).
func TestFreqResponse_DDFallbackLostState(t *testing.T) {
	const k, w = 1e300, 1e250
	for _, damp := range []float64{1e-15, 1e-10} {
		a := -w * damp
		sys, err := New(mat.NewDense(3, 3, []float64{-k, k, 0, 0, a, w, 0, -w, a}), mat.NewDense(3, 1, []float64{1, 1, 1}), mat.NewDense(1, 3, []float64{1, 1, 1}), mat.NewDense(1, 1, []float64{0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		omega := []float64{w, w * (1 + 1e-15)}
		fr, frErr := sys.FreqResponse(omega)
		for i, om := range omega {
			s := complex(0, om)
			want := bigSISOResponse(sys, s)
			got := map[string]complex128{}
			errs := map[string]error{"FreqResponse": frErr}
			if frErr == nil {
				got["FreqResponse"] = fr.At(i, 0, 0)
			}
			g, eErr := sys.EvalFr(s)
			errs["EvalFr"] = eErr
			if eErr == nil {
				got["EvalFr"] = g[0][0]
			}
			for name, e := range errs {
				if e != nil && !errors.Is(e, ErrOverflow) {
					t.Errorf("damp=%g %s(%g): %v, want ErrOverflow", damp, name, om, e)
				}
			}
			for name, v := range got {
				if !cmplx.IsInf(v) && !(cmplx.Abs(v-want) <= 1e-12*cmplx.Abs(want)) {
					t.Errorf("damp=%g %s(%g) = %v, want %v", damp, name, om, v, want)
				}
			}
		}
	}
}

// DZSEVY: x·ȳ/|y|² underflowed for 7.5e-301/7.5e-51 and overflowed to NaN
// for 1e200/1e200; scaling y to unit size keeps the quotient, and the result
// is bit-identical to the unscaled form in the normal range.
func TestDDComplexDiv_ScaledRange(t *testing.T) {
	for _, tc := range []struct{ x, y, want complex128 }{
		{7.5e-301, 7.5e-51, 1e-250},
		{7.5e-301, 7.5e-51i, -1e-250i},
		{1e200, 1e200, 1},
		{1e300 + 1e300i, 1e299 - 1e299i, 10i},
		{3e-200, 1e-200 + 1e-200i, 1.5 - 1.5i},
	} {
		got := ddComplex{dd{real(tc.x), 0}, dd{imag(tc.x), 0}}.div(ddComplex{dd{real(tc.y), 0}, dd{imag(tc.y), 0}}).complex128()
		if !(cmplx.Abs(got-tc.want) <= 1e-15*cmplx.Abs(tc.want)) {
			t.Errorf("%v / %v = %v, want %v", tc.x, tc.y, got, tc.want)
		}
	}
	r := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		v := func() dd {
			hi := r.NormFloat64() * math.Pow(2, float64(r.IntN(200)-100))
			return dd{hi, hi * 0x1p-60 * r.NormFloat64()}
		}
		x, y := ddComplex{v(), v()}, ddComplex{v(), v()}
		n := y.re.mul(y.re).add(y.im.mul(y.im))
		tt := x.mul(ddComplex{y.re, y.im.neg()})
		want := ddComplex{tt.re.div(n), tt.im.div(n)}
		if got := x.div(y); got != want {
			t.Fatalf("%v / %v = %v, want unscaled %v", x, y, got, want)
		}
	}
}

// DZSEVY: a pair at ±j·4.6e283 behind a 1.7e303 state. The verified
// unbalanced fallback returned DCGain/EvalFr(0) 15% off (base 40%): its
// normwise check passed a residual 265× ‖B‖ since ‖M‖‖x‖ ≈ 5e19. The value
// must match the oracle or fail.
func TestEvalFr_GradedPairVerifiedFallback(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1.7450552415535818e+303, 1.1997738833608034e+303, 0,
			0, -1.9245718341395303e+273, 4.5664431807527864e+283,
			0, -4.5664431807527864e+283, -1.9245718341395303e+273}),
		mat.NewDense(3, 1, []float64{1.5963082914223263, 1, -0.9441679141493659}),
		mat.NewDense(1, 3, []float64{1, 1, 0.6417677485106921}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	want := bigSISOResponse(sys, 0)
	got := map[string]complex128{}
	if g, err := sys.EvalFr(0); err == nil {
		got["EvalFr"] = g[0][0]
	} else if !errors.Is(err, ErrOverflow) && !errors.Is(err, ErrSingularTransform) {
		t.Errorf("EvalFr(0): %v", err)
	}
	if g, err := sys.DCGain(); err == nil {
		got["DCGain"] = complex(g.At(0, 0), 0)
	}
	for name, v := range got {
		if !(cmplx.Abs(v-want) <= 1e-12*cmplx.Abs(want)) {
			t.Errorf("%s = %v, want %v", name, v, want)
		}
	}
}

// DZSEVY: the componentwise verifier passes the rounded exact solution of a
// graded pencil and rejects one perturbed by 1e-12 in a state whose row
// terms are 1e-30 of the largest, which a normwise test cannot see.
func TestDDScaled_ComponentwiseVerifier(t *testing.T) {
	br := balancedRealization{n: 2, m: 1, p: 1,
		a: []float64{-1, 0, 1e-30, -1e-30},
		b: []float64{1, 0}, c: []float64{1, 1}}
	sc := ddScaled{br: &br, pr: 0, pi: 0, qr: -1, sigma: 1, bExp: []int{1}}
	sc.x = []complex128{0.5, 0.5}
	if !sc.componentwiseStable() {
		t.Errorf("exact solution rejected")
	}
	sc.x[1] = 0.5 * (1 + 1e-12)
	if sc.componentwiseStable() {
		t.Errorf("perturbed state accepted")
	}
}
