package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var (
	lyapOscA  = []float64{-.1, 5, -5, -.1}
	lyapTriA  = []float64{-1, 2, 0, -3}
	lyapSlowA = []float64{-1e-10, 1, 0, -2}
)

func scaledIdentity2(s float64) *mat.Dense {
	return mat.NewDense(2, 2, []float64{s, 0, 0, s})
}

func TestLyapNonFiniteResultReturnsErrOverflow(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    []float64
		q    *mat.Dense
	}{
		{"oscillatory NaN", lyapOscA, scaledIdentity2(1e308)},
		{"oscillatory max float", lyapOscA, scaledIdentity2(math.MaxFloat64)},
		{"triangular dense Q", lyapTriA, mat.NewDense(2, 2, []float64{1e308, 1e308, 1e308, 1e308})},
		{"slow pole", lyapSlowA, scaledIdentity2(1e300)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, ws := range []*LyapunovOpts{nil, {Workspace: NewLyapunovWorkspace(2)}} {
				X, err := Lyap(mat.NewDense(2, 2, tc.a), tc.q, ws)
				if !errors.Is(err, ErrOverflow) {
					t.Fatalf("X=%v err=%v, want ErrOverflow", X, err)
				}
				if X != nil {
					t.Errorf("X=%v with error, want nil", X)
				}
			}
		})
	}
}

func TestDLyapNonFiniteResultReturnsErrOverflow(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1 - 1e-12, 1, 0, .5})
	X, err := DLyap(A, scaledIdentity2(1e300), nil)
	if !errors.Is(err, ErrOverflow) {
		t.Fatalf("X=%v err=%v, want ErrOverflow", X, err)
	}
	if X != nil {
		t.Errorf("X=%v with error, want nil", X)
	}
}

func TestLyapNonFiniteInputKeepsErrInvalidArgument(t *testing.T) {
	A := mat.NewDense(2, 2, lyapOscA)
	for name, solve := range map[string]func(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error){"Lyap": Lyap, "DLyap": DLyap} {
		_, err := solve(A, scaledIdentity2(math.Inf(1)), nil)
		if !errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrOverflow) {
			t.Errorf("%s: err=%v, want ErrInvalidArgument only", name, err)
		}
	}
}

func TestLyapLargeFiniteResultSurvivesOverflowGuard(t *testing.T) {
	for _, tc := range []struct {
		name     string
		discrete bool
		a        []float64
		scale    float64
	}{
		{"Lyap oscillatory", false, lyapOscA, 1e307},
		{"Lyap triangular", false, lyapTriA, 1e308},
		{"DLyap oscillatory", true, []float64{.9, .3, -.3, .9}, 1e307},
		{"DLyap triangular", true, []float64{.5, .2, 0, -.3}, 1e308},
	} {
		t.Run(tc.name, func(t *testing.T) {
			A := mat.NewDense(2, 2, tc.a)
			solve, residual := Lyap, lyapResidual
			if tc.discrete {
				solve, residual = DLyap, dlyapResidual
			}
			X, err := solve(A, scaledIdentity2(tc.scale), nil)
			if err != nil {
				t.Fatal(err)
			}
			var xs mat.Dense
			xs.Scale(1/tc.scale, X)
			if r := residual(A, &xs, scaledIdentity2(1)); r > 1e-12 {
				t.Errorf("relative residual %g", r)
			}
		})
	}
}

func TestGramNonFiniteResultReturnsErrOverflow(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, lyapOscA), scaledIdentity2(1e154), mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, typ := range []GramType{GramControllability, GramControllabilityFactor} {
		W, err := Gram(sys, typ)
		if !errors.Is(err, ErrOverflow) {
			t.Fatalf("type %d: W=%v err=%v, want ErrOverflow", typ, W, err)
		}
		if W != nil {
			t.Errorf("type %d: W=%v with error, want nil", typ, W)
		}
	}
	for name, f := range map[string]func() error{
		"HSV":     func() error { _, err := HSV(sys); return err },
		"Balreal": func() error { _, err := Balreal(sys); return err },
	} {
		if err := f(); !errors.Is(err, ErrOverflow) {
			t.Errorf("%s: err=%v, want ErrOverflow", name, err)
		}
	}
}

func TestCovarFeedthroughEntryStaysInfiniteOverOverflowedLyapunovSolution(t *testing.T) {
	for _, tau := range []float64{0, 1} {
		sys, err := New(mat.NewDense(2, 2, lyapOscA), scaledIdentity2(1), mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 2, []float64{1, 0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if tau != 0 {
			if err := sys.SetInputDelay([]float64{0, tau}); err != nil {
				t.Fatal(err)
			}
		}
		P, err := Covar(sys, scaledIdentity2(1e308))
		if err != nil {
			t.Fatalf("tau=%g: %v", tau, err)
		}
		if got := P.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("tau=%g: P=%g, want +Inf", tau, got)
		}
	}
}
