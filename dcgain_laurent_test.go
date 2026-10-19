package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

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
