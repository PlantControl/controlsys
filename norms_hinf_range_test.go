package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hinfRangeSys returns G(s) = (2s+3)/((s+1)(s+2))·2^(a+b), with B scaled by
// 2^a and C by 2^b. |G(jω)|² = (4ω²+9)/((1+ω²)(4+ω²)) falls from 9/4 at
// ω = 0, so ‖G‖∞ = 1.5·2^(a+b) at ω = 0.
func hinfRangeSys(t *testing.T, a, b int) *System {
	t.Helper()
	sb, sc := math.Ldexp(1, a), math.Ldexp(1, b)
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 2 * sb}),
		mat.NewDense(1, 2, []float64{sc, sc}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// YUXSI7: B and C near 2^520 returned 2^-133 with a nil error.
func TestHinfNormScaledGainsWithinRange(t *testing.T) {
	for _, tt := range []struct {
		name string
		a, b int
	}{
		{"large B and C", 500, 500},
		{"large B tiny C", 900, -880},
		{"tiny B and C", -300, -300},
		{"subnormal gain", -520, -520},
	} {
		t.Run(tt.name, func(t *testing.T) {
			sys := hinfRangeSys(t, tt.a, tt.b)
			want := math.Ldexp(1.5, tt.a+tt.b)
			got, w, err := HinfNorm(sys)
			tol := 1e-9 * want
			if tt.a+tt.b < -1022 {
				tol = math.Max(tol, math.Ldexp(1, -1074+2))
			}
			if err != nil || math.Abs(got-want) > tol || w != 0 {
				t.Errorf("HinfNorm = %g, %g, %v; want %g, 0, nil", got, w, err, want)
			}
			viaNorm, err := Norm(sys, math.Inf(1))
			if err != nil || viaNorm != got {
				t.Errorf("Norm(sys, Inf) = %g, %v; want %g, nil", viaNorm, err, got)
			}
		})
	}
}

func TestHinfNormOutOfRangeIsErrOverflow(t *testing.T) {
	for _, e := range []int{520, 580} {
		sys := hinfRangeSys(t, e, e)
		if got, w, err := HinfNorm(sys); !errors.Is(err, ErrOverflow) {
			t.Errorf("2^%d: HinfNorm = %g, %g, %v; want ErrOverflow", e, got, w, err)
		}
		if got, err := Norm(sys, math.Inf(1)); !errors.Is(err, ErrOverflow) {
			t.Errorf("2^%d: Norm(sys, Inf) = %g, %v; want ErrOverflow", e, got, err)
		}
	}
	static, err := NewGain(mat.NewDense(2, 2, []float64{1e308, 1e308, 1e308, 1e308}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, w, err := HinfNorm(static); !errors.Is(err, ErrOverflow) {
		t.Errorf("static: HinfNorm = %g, %g, %v; want ErrOverflow", got, w, err)
	}
}

// A zero B or C, or C(sI−A)⁻¹B ≡ 0, leaves the static gain D; the bisection
// used to return its last level, 1e-10·2⁻¹⁰⁰.
func TestHinfNormZeroInputOrOutputMatrix(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2})
	for _, tt := range []struct {
		name string
		B, C *mat.Dense
		D    float64
	}{
		{"zero B", mat.NewDense(2, 1, nil), mat.NewDense(1, 2, []float64{1, 1}), 0},
		{"zero C", mat.NewDense(2, 1, []float64{0, 2}), mat.NewDense(1, 2, nil), -3},
		{"zero transfer", mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{0, 1}), 0},
	} {
		sys, err := New(A, tt.B, tt.C, mat.NewDense(1, 1, []float64{tt.D}), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, w, err := HinfNorm(sys)
		if err != nil || got != math.Abs(tt.D) || w != 0 {
			t.Errorf("%s: HinfNorm = %g, %g, %v; want %g, 0, nil", tt.name, got, w, err, math.Abs(tt.D))
		}
	}
}

// ‖2^b·C·G·2^a·B + 2^(a+b)·D‖∞ = 2^(a+b)·‖G‖∞ for continuous, discrete and
// descriptor models with D ≠ 0; the base norm is computed in working range.
func TestHinfNormHomogeneousUnderIOScaling(t *testing.T) {
	A := []float64{-1.2, 0.7, 0.3, -0.4, -0.9, 1.1, 0.2, -0.6, -1.5}
	Ad := []float64{0.5, 0.3, -0.1, -0.2, 0.4, 0.25, 0.1, -0.3, 0.6}
	E := []float64{2, 0.3, -0.1, 0.2, 1.5, 0.4, -0.3, 0.1, 1.8}
	B := []float64{1, -0.5, 0.3, 2, -1, 0.7}
	C := []float64{0.4, -1, 0.6, 1.3, 0.2, -0.8}
	D := []float64{0.3, -0.2, 0.1, 0.25}
	build := func(t *testing.T, kind string, a, b int) *System {
		t.Helper()
		bm := mat.NewDense(3, 2, nil)
		cm := mat.NewDense(2, 3, nil)
		dm := mat.NewDense(2, 2, nil)
		for i, x := range B {
			bm.Set(i/2, i%2, math.Ldexp(x, a))
		}
		for i, x := range C {
			cm.Set(i/3, i%3, math.Ldexp(x, b))
		}
		for i, x := range D {
			dm.Set(i/2, i%2, math.Ldexp(x, a+b))
		}
		var sys *System
		var err error
		switch kind {
		case "continuous":
			sys, err = New(mat.NewDense(3, 3, A), bm, cm, dm, 0)
		case "discrete":
			sys, err = New(mat.NewDense(3, 3, Ad), bm, cm, dm, 0.1)
		case "descriptor":
			sys, err = NewDescriptor(mat.NewDense(3, 3, A), bm, cm, dm, mat.NewDense(3, 3, E), 0)
		}
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	for _, kind := range []string{"continuous", "discrete", "descriptor"} {
		base, baseW, err := HinfNorm(build(t, kind, 0, 0))
		if err != nil || !(base > 0) {
			t.Fatalf("%s: base HinfNorm = %g, %v", kind, base, err)
		}
		for _, s := range [][2]int{{500, 450}, {540, 470}, {-480, -500}, {-540, -500}, {700, -690}, {-30, 450}} {
			got, w, err := HinfNorm(build(t, kind, s[0], s[1]))
			want := math.Ldexp(base, s[0]+s[1])
			if err != nil || math.Abs(got-want) > 1e-9*want {
				t.Errorf("%s 2^%d,2^%d: HinfNorm = %g, %v; want %g", kind, s[0], s[1], got, err, want)
			}
			if math.Abs(w-baseW) > 1e-4*baseW {
				t.Errorf("%s 2^%d,2^%d: omega = %g, want %g", kind, s[0], s[1], w, baseW)
			}
		}
	}
}

// The non-minimal fallback of HSV kept B·C·2^-2e unscaled even when it
// underflowed, so B near 1e-160 gave σ = 0.
func TestHSVNonMinimalTinyInputScale(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 0.4, 0.2, 0.3, -2, 0.5, 0, 0, -3})
	C := mat.NewDense(1, 3, []float64{1, -0.5, 0.7})
	build := func(s float64) *System {
		sys, err := New(A, mat.NewDense(3, 1, []float64{s, -2 * s, 0}), C, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	base, err := HSV(build(1))
	if err != nil {
		t.Fatal(err)
	}
	const s = 1e-160
	got, err := HSV(build(s))
	if err != nil {
		t.Fatal(err)
	}
	for i := range base {
		if math.Abs(got[i]-base[i]*s) > 1e-9*base[0]*s {
			t.Errorf("σ%d = %g, want %g", i, got[i], base[i]*s)
		}
	}
	if !(got[0] > 0) {
		t.Errorf("σ1 = %g, want > 0", got[0])
	}
}

// C(sI−A)⁻¹B ≡ 0 with B, C near 2^600 and D = 1: the scaled search must keep
// D representable, as the peak is D alone.
func TestHinfNormScaledZeroTransferKeepsFeedthrough(t *testing.T) {
	big := math.Ldexp(1, 600)
	for _, tt := range []struct {
		a  []float64
		dt float64
	}{
		{[]float64{-1, 0.5, 0, -2}, 0},
		{[]float64{0.5, 0.2, 0, -0.3}, 0.1},
	} {
		sys, err := New(
			mat.NewDense(2, 2, tt.a),
			mat.NewDense(2, 1, []float64{big, 0}),
			mat.NewDense(1, 2, []float64{0, big}),
			mat.NewDense(1, 1, []float64{1}), tt.dt)
		if err != nil {
			t.Fatal(err)
		}
		if got, w, err := HinfNorm(sys); err != nil || math.Abs(got-1) > 1e-9 {
			t.Errorf("dt=%g: HinfNorm = %g, %g, %v; want 1, nil", tt.dt, got, w, err)
		}
	}
}
