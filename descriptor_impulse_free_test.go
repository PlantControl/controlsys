package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func idealPDDescriptor(t *testing.T, kp, kd float64) *System {
	t.Helper()
	c, err := NewDescriptor(eyeDense(2), mat.NewDense(2, 1, []float64{0, -1}), mat.NewDense(1, 2, []float64{kd, 0}), mat.NewDense(1, 1, []float64{kp}), mat.NewDense(2, 2, []float64{0, 1, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func transposeDescriptor(t *testing.T, sys *System) *System {
	t.Helper()
	var a, b, c, d, e mat.Dense
	a.CloneFrom(sys.A.T())
	b.CloneFrom(sys.C.T())
	c.CloneFrom(sys.B.T())
	d.CloneFrom(sys.D.T())
	e.CloneFrom(sys.E.T())
	out, err := NewDescriptor(&a, &b, &c, &d, &e, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// RMGVA3: Feedback(Series(ideal PD descriptor, 1/(s(s−1)))) is proper, but
// its impulsive mode is unobservable, so C∞ is rounding noise; the
// impulse-free test scaled by that noise rejected it. Oracle: the explicit
// companion form of (Kd·s + Kp)/(s² + (Kd−1)·s + Kp), MATLAB
// step(feedback(pid(Kp,0,Kd)*G,1)).
func TestDescriptorProperClosedLoopIdealPD(t *testing.T) {
	const kp, kd = 1.9019237886466818, 3.0980762113533165
	g, err := New(mat.NewDense(2, 2, []float64{0, 1, 0, 1}), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	loop, err := Series(idealPDDescriptor(t, kp, kd), g)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(loop, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.E == nil {
		t.Fatal("closed loop lost E")
	}
	oracle, err := New(mat.NewDense(2, 2, []float64{0, 1, -kp, 1 - kd}), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{kp, kd}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Step(oracle, 3)
	if err != nil {
		t.Fatal(err)
	}
	wantTF := func(s complex128) complex128 {
		return (complex(kd, 0)*s + complex(kp, 0)) / (s*s + complex(kd-1, 0)*s + complex(kp, 0))
	}
	for name, sys := range map[string]*System{"cl": cl, "transpose": transposeDescriptor(t, cl)} {
		if proper, err := sys.IsProper(); err != nil || !proper {
			t.Fatalf("%s: IsProper = %v, %v, want true", name, proper, err)
		}
		got, err := Step(sys, 3)
		if err != nil {
			t.Fatalf("%s: Step: %v", name, err)
		}
		if len(got.T) != len(want.T) {
			t.Fatalf("%s: %d samples, oracle %d", name, len(got.T), len(want.T))
		}
		for k := range got.T {
			if d := math.Abs(got.Y.At(0, k) - want.Y.At(0, k)); d > 1e-10 {
				t.Fatalf("%s: y(%g) = %v, oracle %v", name, got.T[k], got.Y.At(0, k), want.Y.At(0, k))
			}
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", name, err)
		}
		for _, w := range []float64{0, 0.3, 1, 7} {
			s := complex(0, w)
			got := polyEvalC(res.TF.Num[0][0], s) / polyEvalC(res.TF.Den[0], s)
			fr, err := sys.EvalFr(s)
			if err != nil {
				t.Fatal(err)
			}
			if d := cmplx.Abs(got - wantTF(s)); d > 1e-12*cmplx.Abs(wantTF(s)) {
				t.Errorf("%s: tf(%gj) = %v, oracle %v", name, w, got, wantTF(s))
			}
			if d := cmplx.Abs(fr[0][0] - wantTF(s)); d > 1e-12*cmplx.Abs(wantTF(s)) {
				t.Errorf("%s: EvalFr(%gj) = %v, oracle %v", name, w, fr[0][0], wantTF(s))
			}
		}
		if len(res.TF.Den[0]) != 3 {
			t.Errorf("%s: den %v, want degree 2", name, res.TF.Den[0])
		}
	}
}

// RMGVA3 guard: an observable derivative stays improper, also for a small
// Kd in a loop with a biproper plant (s+2)/(s+1).
func TestDescriptorObservableDerivativeStaysImproper(t *testing.T) {
	g, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, kd := range []float64{1, 1e-5} {
		pd := idealPDDescriptor(t, 2, kd)
		loop, err := Series(pd, g)
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"pd": pd, "loop": loop, "transpose": transposeDescriptor(t, loop)} {
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Errorf("Kd %g %s: Step err = %v, want ErrImproperModel", kd, name, err)
			}
			if _, err := sys.TransferFunction(nil); !errors.Is(err, ErrImproperModel) || !errors.Is(err, ErrDescriptorUnsupported) {
				t.Errorf("Kd %g %s: TransferFunction err = %v, want ErrImproperModel", kd, name, err)
			}
		}
	}
}

func smallDerivativeDescriptor(t *testing.T, c *mat.Dense) *System {
	t.Helper()
	p, _ := c.Dims()
	sys, err := NewDescriptor(mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 1, 0, 0, 0, 1}), mat.NewDense(3, 1, []float64{1, 0, -1}), c, mat.NewDense(p, 1, nil), mat.NewDense(3, 3, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Review of RMGVA3: y = c1/(s+1)·u + kd·s·u with kd ≤ 1e-8·c1 passed the
// 1e-8-relative impulse test, so Step and TransferFunction dropped kd·s
// (TF(1e6j) = −0.001i, EvalFr +0.999i) while IsProper reported false. Base
// and MATLAB step reject them. The test is now rounding level and IsProper
// uses it, so all three agree; a 1e-15 s-term is rounding noise.
func TestDescriptorSmallDerivativeStaysImproper(t *testing.T) {
	cases := []struct {
		name   string
		c      *mat.Dense
		proper bool
	}{
		{"c1=1e3 kd=1e-6", mat.NewDense(1, 3, []float64{1e3, 1e-6, 0}), false},
		{"c1=1e2 kd=1e-6", mat.NewDense(1, 3, []float64{1e2, 1e-6, 0}), false},
		{"c1=1 kd=1e-9", mat.NewDense(1, 3, []float64{1, 1e-9, 0}), false},
		{"c1=1e4 kd=1e-5", mat.NewDense(1, 3, []float64{1e4, 1e-5, 0}), false},
		{"c1=1 kd=1e-11", mat.NewDense(1, 3, []float64{1, 1e-11, 0}), false},
		{"two outputs", mat.NewDense(2, 3, []float64{1e4, 0, 0, 0, 1e-5, 0}), false},
		{"c1=1 kd=1e-15", mat.NewDense(1, 3, []float64{1, 1e-15, 0}), true},
	}
	for _, tc := range cases {
		sys := smallDerivativeDescriptor(t, tc.c)
		for name, s := range map[string]*System{"sys": sys, "transpose": transposeDescriptor(t, sys)} {
			label := tc.name + " " + name
			proper, err := s.IsProper()
			if err != nil {
				t.Fatalf("%s: IsProper: %v", label, err)
			}
			if proper != tc.proper {
				t.Errorf("%s: IsProper = %v, want %v", label, proper, tc.proper)
			}
			_, stepErr := Step(s, 1)
			res, tfErr := s.TransferFunction(nil)
			if (stepErr == nil) != proper || (tfErr == nil) != proper {
				t.Errorf("%s: IsProper %v but Step err %v, TransferFunction err %v", label, proper, stepErr, tfErr)
			}
			if !tc.proper {
				if !errors.Is(stepErr, ErrImproperModel) {
					t.Errorf("%s: Step err = %v, want ErrImproperModel", label, stepErr)
				}
				if !errors.Is(tfErr, ErrImproperModel) || !errors.Is(tfErr, ErrDescriptorUnsupported) {
					t.Errorf("%s: TransferFunction err = %v, want ErrImproperModel", label, tfErr)
				}
				continue
			}
			if tfErr != nil {
				continue
			}
			for _, w := range []float64{0, 1, 1e3} {
				z := complex(0, w)
				got := polyEvalC(res.TF.Num[0][0], z) / polyEvalC(res.TF.Den[0], z)
				want := 1 / (z + 1)
				if cmplx.Abs(got-want) > 1e-12 {
					t.Errorf("%s: tf(%gj) = %v, want %v", label, w, got, want)
				}
			}
		}
	}
}
