package controlsys

import (
	"math"
	"math/cmplx"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// E2AY73: the transfer-limit fallback took the Inf/finite decision and its
// sign from transfer-function coefficients that are pure roundoff for
// coupled exact integrators. Oracle: the exact rational limit of
// C(x0·I − A)⁻¹B + D on the float model; the sign is free when further
// intended poles lie within roundoff of x0, as in FuzzDCGain. 0xab already
// passes on base 2e4a07d and is kept as a guard; the others fail there.
func TestDCGain_CoupledExactIntegratorsExactLimit(t *testing.T) {
	cases := []struct {
		kind uint8
		v    [fzNVals]float64
	}{
		{0xab, [fzNVals]float64{-40, 10, 0.4, 1.5, 42, 0, 0, 2, -75, 2, 0, -204, -0.15625, -1, 20, -17.5}},
		{0xd4, [fzNVals]float64{84.6, 1, -44.5, 28, -78, 34, -87, 65.5, 0, -90, 0.37, 0, -1.1111111111111112, -10, -0.4, 23}},
		{0xa2, [fzNVals]float64{-92.33333333333333, 36, 33.5, 3.4, 0, 0, -38, 0.5, 130, 81, 0, 32.5, -100, -10, -5, 6}},
		{0x61, [fzNVals]float64{-40, 1, 0.2, 1.5, 0, 0, 0, 0.5, 59, 54, 0, -552, -10, -10, -5, -10}},
	}
	for _, tc := range cases {
		s := fzDecode(t, tc.kind, tc.v)
		g, err := s.System(t, fzSysOpts{}).DCGain()
		if err != nil {
			t.Fatalf("%#x: %v", tc.kind, err)
		}
		x0f := 0.0
		if s.Discrete {
			x0f = 1
		}
		x0 := fzRat(x0f)
		cp := fzCharPolyExact(s.A, s.N)
		sh := fzTaylorShift(cp, x0)
		ord := 0
		for k := s.N; k >= 0 && sh[k].Sign() == 0; k-- {
			ord++
		}
		band := math.Max(4*s.EigErr(), 1e-10*math.Max(1, s.NormJ))
		near := 0
		for _, e := range s.Eig {
			if cmplx.Abs(e-complex(x0f, 0)) <= band {
				near++
			}
		}
		for i := range s.P {
			for j := range s.M {
				bj := make([]float64, s.N)
				for k := range s.N {
					bj[k] = s.B[k*s.M+j]
				}
				num := fzPencilDetRat(s.N, 1, s.A, bj, s.C[i*s.N:(i+1)*s.N], []float64{s.D[i*s.M+j]})
				want := fzLimitAt(num, cp, x0)
				if !math.IsInf(want, 0) {
					t.Fatalf("%#x: oracle [%d,%d] = %v finite; case no longer exercises an integrator", tc.kind, i, j, want)
				}
				got := g.At(i, j)
				if near > ord && math.IsInf(got, 0) {
					continue
				}
				if got != want {
					t.Errorf("%#x: DCGain[%d,%d] = %v, exact limit %v", tc.kind, i, j, got, want)
				}
			}
		}
	}
}

// E2AY73: a tiny third pole dropped both exact z = 1 poles from the
// transfer function, so DCGain was finite. Exact residue at z = 1 is
// c1b1 + c2b2 + (c1+c2)·a·b3/(1−ε) ≈ 1.576 ≠ 0, so the gain is +∞.
func TestDCGain_TinyPoleKeepsDoubleZEqualsOne(t *testing.T) {
	a, c := -3.1622776601683795, []float64{-0.9291407314223547, 0.6502534890410231, 0.5087003937751763}
	b := []float64{-0.38211320287195655, 0.7882242424655154, 0.8038113259779769}
	for _, e := range []float64{0, 1e-12, 1e-20, 1e-40, 1e-69} {
		sys, err := NewFromSlices(3, 1, 1, []float64{1, 0, a, 0, 1, a, 0, 0, e}, b, c, []float64{-0.23623091926062198}, 100)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		if res := c[0]*b[0] + c[1]*b[1] + (c[0]+c[1])*a*b[2]/(1-e); res <= 0 {
			t.Fatalf("eps=%g: residue %v, want > 0", e, res)
		}
		if got := g.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("eps=%g: DCGain = %v, want +Inf", e, got)
		}
	}
}

// An exact integrator cancelled by a zero that is not structural (both
// states reachable and observable): w0 = (b0 + w1)/s with w1 = b1/(s+2)
// and b0 = −b1/2 gives w0 = −0.5/(s+2), so G(0) = D + (−0.5·c0 + c1)/2.
func TestDCGain_NonStructuralIntegratorCancellation(t *testing.T) {
	for _, dt := range []float64{0, 0.5} {
		a := []float64{0, 1, 0, -2}
		if dt > 0 {
			a = []float64{1, 1, 0, -1}
		}
		sys, err := NewFromSlices(2, 1, 1, a, []float64{-0.5, 1}, []float64{1, 3}, []float64{0.25}, dt)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		if got, want := g.At(0, 0), 0.25+(-0.5+3)/2; math.Abs(got-want) > 1e-14 {
			t.Errorf("dt=%v: DCGain = %v, want %v", dt, got, want)
		}
	}
}

// E2AY73 (near-singular solve): det(I − A) = 3e-16 is not zero, so by
// MATLAB dcgain on the float model the gain is finite. Exact rational
// value on that model is 0.50825; cond(I − A) ≈ 4e15 leaves only the
// leading digit, and 2.5538 (the intended cancelled model) is not it.
func TestDCGain_NearSingularSolveStaysFinite(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{0.2772706061286758, -0.29385449949935655, 0.6983526834182192, 1.2839431743057481}),
		mat.NewDense(2, 1, []float64{-2.436691822351444, 2.3545054168716004}),
		mat.NewDense(1, 2, []float64{-0.05023449497436656, 0.28984069544431224}),
		mat.NewDense(1, 1, []float64{0.7195914212138912}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if got := g.At(0, 0); math.IsInf(got, 0) || math.Abs(got-0.5082518651448561) > 0.25 {
		t.Errorf("DCGain = %v, want finite near 0.508", got)
	}
}

// OTMOFO: B and C entries just above the old 2⁻²⁰·‖A‖ scaling threshold
// and a 1e-4 coupling put the transfer numerator (~1e-15) under the limit's
// absolute coefficient floor, so an integrator in a 2×2 exactly singular
// block (no 1×1 exact pole for the Laurent path) read as 0. Oracle: exact
// rational limit on the float model.
func TestDCGain_SmallNumeratorBelowAbsoluteFloor(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    []float64
		c    []float64
		dt   float64
	}{
		{"ct", []float64{-1, 1, 0, 1, -1, 0, 0, 1e-4, -1}, []float64{0, 0, 4e-6}, 0},
		{"ct-neg", []float64{-1, 1, 0, 1, -1, 0, 0, 1e-4, -1}, []float64{0, 0, -4e-6}, 0},
		{"dt", []float64{0.5, 0.5, 0, 0.5, 0.5, 0, 0, 1e-4, 0.5}, []float64{0, 0, 4e-6}, 1},
		{"mimo", []float64{-1, 1, 0, 1, -1, 0, 0, 1e-4, -1}, []float64{3e-6, 0, 4e-6}, 0},
		{"mimo-dt", []float64{0.5, 0.5, 0, 0.5, 0.5, 0, 0, 1e-4, 0.5}, []float64{3e-6, 0, 4e-6}, 1},
	} {
		const k = 4e-6
		b := []float64{k, k, 0, -k, 0, 0}
		d := []float64{0.5, 0}
		m := 2
		if !strings.HasPrefix(tc.name, "mimo") {
			b, d, m = []float64{k, 0, 0}, []float64{0}, 1
		}
		sys, err := NewFromSlices(3, m, 1, tc.a, b, tc.c, d, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		x0 := fzRat(0)
		if tc.dt > 0 {
			x0 = fzRat(1)
		}
		cp := fzCharPolyExact(tc.a, 3)
		for j := range m {
			bj := []float64{b[j], b[m+j], b[2*m+j]}
			want := fzLimitAt(fzPencilDetRat(3, 1, tc.a, bj, tc.c, []float64{d[j]}), cp, x0)
			got := g.At(0, j)
			if j == 0 && !math.IsInf(want, 0) {
				t.Fatalf("%s: oracle [0,0] = %v finite; case no longer exercises an integrator", tc.name, want)
			}
			if math.IsInf(want, 0) {
				if got != want {
					t.Errorf("%s [0,%d]: DCGain = %v, want %v", tc.name, j, got, want)
				}
			} else if math.Abs(got-want) > 1e-8*math.Abs(want) {
				t.Errorf("%s [0,%d]: DCGain = %v, want %v", tc.name, j, got, want)
			}
		}
	}
}

// fzSISOExactLimit is the exact rational limit at s = 0 of C(sI−A)⁻¹B on
// the float model.
func fzSISOExactLimit(n int, a, b, c []float64) float64 {
	num := fzPencilDetRat(n, 1, a, b, c, []float64{0})
	return fzLimitAt(num, fzCharPolyExact(a, n), fzRat(0))
}

// E2AY73: deflating the {1,2,3} block's ±6.4e-8i pair to exact 0 made a
// triple pole whose order −3 coefficient flipped the sign of the exact
// integrator at state 0. The block is idle at order −1, so the expansion on
// the model as given settles it without deflation.
func TestDCGain_LaurentPrefersUndeflatedModel(t *testing.T) {
	a := []float64{0, 0, 0, 0, 0, 0, 0, -12.696489978696855, 0, 1, 0, 0, 0, 0, -1.2788825566727037e-15, 4}
	b := []float64{0.0007045326307198843, -0.0001380169790660376, -0.0005520679162641504, -0.00041405093719811285}
	c := []float64{1.4342515182324627e-10, -2.8685030364649254e-10, -8.941142520710081e-12, 0}
	want := fzSISOExactLimit(4, a, b, c)
	if !math.IsInf(want, 1) {
		t.Fatalf("oracle %v, want +Inf", want)
	}
	sys, err := NewFromSlices(4, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if got := g.At(0, 0); got != want {
		t.Errorf("DCGain = %v, exact limit %v", got, want)
	}
}

// OTMOFO: G = γ²·1e-4/(s(s+2γ)(s+γ)) has numerator 1e-4·γ², trimmed by the
// transfer limit's absolute floor 100·ε once γ ≤ 1e-6, so the gain was 0.
// The integrator is not a 1×1 block, so Laurent declines.
func TestDCGain_SmallNormAIntegrator(t *testing.T) {
	for _, gm := range []float64{1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12} {
		a := []float64{-gm, gm, 0, gm, -gm, 0, 0, gm * 1e-4, -gm}
		b, c := []float64{1, 0, 0}, []float64{0, 0, 1}
		want := fzSISOExactLimit(3, a, b, c)
		if !math.IsInf(want, 1) {
			t.Fatalf("γ=%g: oracle %v, want +Inf", gm, want)
		}
		sys, err := NewFromSlices(3, 1, 1, a, b, c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		if got := g.At(0, 0); got != want {
			t.Errorf("γ=%g: DCGain = %v, exact limit %v", gm, got, want)
		}
	}
}

// E2AY73 follow-ups, integer models with an exact integrator beside a
// singular 2×2 or 3×3 block, where the deflated Laurent gave the wrong Inf
// sign (base 2e4a07d was right):
//   - nilpotent: Schur roundoff (1e-16) in a coupling that cancels exactly
//     was weighed against itself and decided an order −3 coefficient;
//   - bump: Dgees returned the nilpotent block as a 2×2 standardized bump,
//     and zeroing its strictly lower part dropped the entry 8;
//   - rowsum: an exactly singular 3×3 block (zero row sums) passed as
//     regular on its roundoff LU pivot, hiding a pole at 0;
//   - cancel: the integrator's input x1 − x2 cancels exactly, so G(0) = 0;
//     |entries| as magnitudes let transform roundoff decide +Inf.
func TestDCGain_LaurentSingularBlocks(t *testing.T) {
	cases := []struct {
		name    string
		n       int
		a, b, c []float64
	}{
		{"nilpotent", 4, []float64{4, 0, 0, -4, -1, 0, 0, 1, 0, -4, -1, 5, 4, 0, 0, -4}, []float64{0, -1, 1, -1}, []float64{-2, 0, 1, 2}},
		{"bump", 3, []float64{-4, 0, 4, 1, 0, -1, -4, 0, 4}, []float64{1, -2, -2}, []float64{1, -2, 0}},
		{"rowsum", 4, []float64{4, -2, 0, -2, 0, 3, 0, -3, 0, 0, 0, 0, 4, 2, 0, -6}, []float64{1, -1, 1, -2}, []float64{-1, -1, 1, 0}},
		{"cancel", 3, []float64{0, 1, -1, 0, -3, 3, 0, -1, 1}, []float64{1, 1, 1}, []float64{2, 0, -2}},
	}
	for _, tc := range cases {
		for _, sc := range []float64{1, 0x1p-14} {
			a := make([]float64, len(tc.a))
			for i, v := range tc.a {
				a[i] = sc * v
			}
			want := fzSISOExactLimit(tc.n, a, tc.b, tc.c)
			if math.IsInf(want, 0) == (tc.name == "cancel") {
				t.Fatalf("%s: oracle %v", tc.name, want)
			}
			sys, err := NewFromSlices(tc.n, 1, 1, a, tc.b, tc.c, []float64{0}, 0)
			if err != nil {
				t.Fatal(err)
			}
			g, err := sys.DCGain()
			if err != nil {
				t.Fatal(err)
			}
			if got := g.At(0, 0); got != want && !(math.Abs(got-want) <= 1e-9*math.Max(math.Abs(want), 1)) {
				t.Errorf("%s ×%g: DCGain = %v, exact limit %v", tc.name, sc, got, want)
			}
		}
	}
}

func TestExactDetIsZero(t *testing.T) {
	cases := []struct {
		a    []float64
		want bool
	}{
		{[]float64{4, -2, -2, 0, 3, -3, 4, 2, -6}, true},
		{[]float64{0, 0, 12.696489978696855, -1, 0, 0, 0, 1.2788825566727037e-15, -4}, false},
		{[]float64{0, 1, 1, 0}, false},
		{[]float64{0, 0, 0, 0}, true},
		{[]float64{1e-300, 1e300, 1e-300, 1e300}, true},
		{[]float64{1e-300, 1e300, 1e-300, math.Nextafter(1e300, 0)}, false},
	}
	for _, tc := range cases {
		r := int(math.Sqrt(float64(len(tc.a))))
		if got := exactDetIsZero(blas64.General{Rows: r, Cols: r, Stride: r, Data: tc.a}); got != tc.want {
			t.Errorf("%v: exactDetIsZero = %v, want %v", tc.a, got, tc.want)
		}
	}
}
