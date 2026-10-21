package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// normDescriptorTwin returns the descriptor model (E·A, E·B, C, D, E), with
// E·B2 for internal delays, of an explicit model: same transfer function,
// same poles.
func normDescriptorTwin(sys *System, E *mat.Dense) *System {
	ds := sys.Copy()
	ds.E = mat.DenseCopyOf(E)
	ds.A.Mul(E, sys.A)
	ds.B.Mul(E, sys.B)
	if sys.LFT != nil {
		ds.LFT.B2.Mul(E, sys.LFT.B2)
	}
	return ds
}

func mustNew(t *testing.T, A, B, C, D *mat.Dense, dt float64) *System {
	t.Helper()
	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// normsClose compares norms; omega is compared to 1e-5 since a peak located
// to 1e-10 in gain fixes its frequency only to about √1e-10.
func normsClose(got, want float64) bool {
	if math.IsInf(want, 1) {
		return math.IsInf(got, 1)
	}
	return math.Abs(got-want) <= 1e-8*math.Max(1, math.Abs(want))
}

func TestNormsDescriptorNonsingularE(t *testing.T) {
	E := mat.NewDense(3, 3, []float64{
		2, 1, 0.5,
		-1, 3, 1,
		0.5, -2, 4,
	})
	cases := []struct {
		name string
		sys  func(t *testing.T) *System
	}{
		{"continuous", func(t *testing.T) *System {
			return mustNew(t, mat.NewDense(3, 3, []float64{
				-1, 2, 0,
				-3, -0.5, 1,
				0.2, 0, -2,
			}), mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, 0, -1}),
				mat.NewDense(2, 3, []float64{1, -1, 0.3, 0, 2, 1}),
				mat.NewDense(2, 2, nil), 0)
		}},
		{"continuous-D", func(t *testing.T) *System {
			return mustNew(t, mat.NewDense(3, 3, []float64{
				-0.1, 5, 0,
				-5, -0.1, 1,
				0, 0, -3,
			}), mat.NewDense(3, 1, []float64{0, 1, 1}),
				mat.NewDense(1, 3, []float64{1, 0, 2}),
				mat.NewDense(1, 1, []float64{0.4}), 0)
		}},
		{"discrete", func(t *testing.T) *System {
			return mustNew(t, mat.NewDense(3, 3, []float64{
				0.5, 0.3, 0,
				-0.2, 0.4, 0.1,
				0, 0.6, -0.3,
			}), mat.NewDense(3, 1, []float64{1, 0, 1}),
				mat.NewDense(1, 3, []float64{0, 1, -1}),
				mat.NewDense(1, 1, []float64{0.2}), 0.1)
		}},
		{"unstable", func(t *testing.T) *System {
			return mustNew(t, mat.NewDense(3, 3, []float64{
				0.5, 1, 0,
				-1, -1, 0,
				0, 1, -2,
			}), mat.NewDense(3, 1, []float64{1, 0, 1}),
				mat.NewDense(1, 3, []float64{1, 1, 0}),
				mat.NewDense(1, 1, nil), 0)
		}},
		{"internal-delay", func(t *testing.T) *System {
			sys := mustNew(t, mat.NewDense(3, 3, []float64{
				-1, 2, 0,
				-3, -0.5, 1,
				0.2, 0, -2,
			}), mat.NewDense(3, 1, []float64{1, 0.5, 0}),
				mat.NewDense(1, 3, []float64{1, -1, 0.3}),
				mat.NewDense(1, 1, nil), 0)
			k, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0)
			if err := sys.SetOutputDelay([]float64{0.4}); err != nil {
				t.Fatal(err)
			}
			cl, err := Feedback(sys, k, -1)
			if err != nil {
				t.Fatal(err)
			}
			if !cl.HasInternalDelay() {
				t.Fatal("want internal delay")
			}
			return cl
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			ds := normDescriptorTwin(sys, E)
			if !ds.IsDescriptor() {
				t.Fatal("twin is not a descriptor model")
			}

			wantInf, wantW, err := HinfNorm(sys)
			if err != nil {
				t.Fatal(err)
			}
			gotInf, gotW, err := HinfNorm(ds)
			if err != nil {
				t.Fatalf("HinfNorm(descriptor): %v", err)
			}
			wOK := math.IsInf(wantW, 1) && math.IsInf(gotW, 1) ||
				math.Abs(gotW-wantW) <= 1e-5*math.Max(1, wantW)
			if !normsClose(gotInf, wantInf) || !wOK {
				t.Errorf("HinfNorm = %v at %v, explicit %v at %v", gotInf, gotW, wantInf, wantW)
			}
			wantL, err := Norm(sys, math.Inf(1))
			if err != nil {
				t.Fatal(err)
			}
			gotL, err := Norm(ds, math.Inf(1))
			if err != nil || !normsClose(gotL, wantL) {
				t.Errorf("Norm(Inf) = %v, %v; explicit %v", gotL, err, wantL)
			}
			if sys.HasInternalDelay() {
				return
			}
			want2, wantErr := H2Norm(sys)
			got2, err := H2Norm(ds)
			if !errors.Is(err, wantErr) || err == nil && !normsClose(got2, want2) {
				t.Errorf("H2Norm = %v, %v; explicit %v, %v", got2, err, want2, wantErr)
			}
			wantN2, _ := Norm(sys, 2)
			gotN2, err := Norm(ds, 2)
			if err != nil || !normsClose(gotN2, wantN2) {
				t.Errorf("Norm(2) = %v, %v; explicit %v", gotN2, err, wantN2)
			}
		})
	}
}

// TestNormsDescriptorAnalytic: 2ẋ = −2x + 2u, y = x is G(s) = 1/(s+1),
// with H2 norm 1/√2 and peak gain 1 at ω = 0.
func TestNormsDescriptorAnalytic(t *testing.T) {
	ds, err := NewDescriptor(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := H2Norm(ds)
	if err != nil || math.Abs(h2-1/math.Sqrt2) > 1e-12 {
		t.Errorf("H2Norm = %v, %v, want 1/√2", h2, err)
	}
	hinf, w, err := HinfNorm(ds)
	if err != nil || math.Abs(hinf-1) > 1e-9 || w != 0 {
		t.Errorf("HinfNorm = %v at %v, %v, want 1 at 0", hinf, w, err)
	}
}

func TestNormsDescriptorSingularOrIllConditionedE(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 1, 0.5, -2})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	limit := normDescriptorCondLimit
	for _, tc := range []struct {
		name string
		e    []float64
		ok   bool
	}{
		{"singular", []float64{1, 0, 0, 0}, false},
		{"rank-one", []float64{1, 2, 2, 4}, false},
		{"over-limit", []float64{1, 0, 0, 1 / (4 * limit)}, false},
		{"under-limit", []float64{1, 0, 0, 4 / limit}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			E := mat.NewDense(2, 2, tc.e)
			ds, err := NewDescriptor(A, B, C, D, E, 0)
			if err != nil {
				t.Fatal(err)
			}
			_, h2err := H2Norm(ds)
			_, _, hinfErr := HinfNorm(ds)
			_, linfErr := Norm(ds, math.Inf(1))
			for _, err := range []error{h2err, hinfErr, linfErr} {
				if tc.ok && err != nil {
					t.Errorf("err = %v, want nil", err)
				}
				if !tc.ok && (!errors.Is(err, ErrDescriptorUnsupported) || !errors.Is(err, ErrDescriptorSingular)) {
					t.Errorf("err = %v, want ErrDescriptorUnsupported and ErrDescriptorSingular", err)
				}
			}
		})
	}
}
