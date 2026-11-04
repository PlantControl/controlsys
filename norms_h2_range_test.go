package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// UAZL45: tr(B'XB) overflowed to +Inf with a nil error although the norm is
// representable. A = σ·R(θ), so ‖C·e^{At}‖ and ‖C·Aᵏ‖ decay as scalars and
// tr X = ‖C‖²/(2·0.1) = 25 (continuous) or ‖C‖²/(1−0.9) = 50 (discrete).
func TestH2NormScaledGainsWithinRange(t *testing.T) {
	ac := []float64{-0.1, 5, -5, -0.1}
	ad := []float64{0.9, 0.3, -0.3, 0.9}
	tests := []struct {
		name  string
		a     []float64
		dt    float64
		scale float64
		want  float64
	}{
		{"continuous large B", ac, 0, 1e154, 5e154},
		{"discrete large B", ad, 1, 1e154, math.Sqrt(50) * 1e154},
		{"continuous tiny B", ac, 0, 1e-200, 5e-200},
		{"discrete tiny B", ad, 1, 1e-200, math.Sqrt(50) * 1e-200},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sys, err := New(
				mat.NewDense(2, 2, tt.a),
				mat.NewDense(2, 2, []float64{tt.scale, 0, 0, tt.scale}),
				mat.NewDense(1, 2, []float64{1, 2}),
				mat.NewDense(1, 2, nil), tt.dt)
			if err != nil {
				t.Fatal(err)
			}
			got, err := H2Norm(sys)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(got-tt.want) > 1e-10*tt.want {
				t.Errorf("H2Norm = %.17g, want %.17g", got, tt.want)
			}
			viaNorm, err := Norm(sys, NormH2)
			if err != nil || viaNorm != got {
				t.Errorf("Norm(sys, 2) = %g, %v; want %g, nil", viaNorm, err, got)
			}
		})
	}
}

func TestH2NormDiscreteLargeFeedthroughWithinRange(t *testing.T) {
	static, err := NewGain(mat.NewDense(2, 2, []float64{1e200, 0, 0, 1e200}), 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := H2Norm(static)
	if want := math.Sqrt2 * 1e200; err != nil || math.Abs(got-want) > 1e-14*want {
		t.Errorf("static H2Norm = %g, %v; want %g, nil", got, err, want)
	}

	dyn, err := New(
		mat.NewDense(2, 2, []float64{0.9, 0.3, -0.3, 0.9}),
		mat.NewDense(2, 2, []float64{1e154, 0, 0, 1e154}),
		mat.NewDense(1, 2, []float64{1, 2}),
		mat.NewDense(1, 2, []float64{3e154, 4e154}), 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err = H2Norm(dyn)
	if want := math.Sqrt(75) * 1e154; err != nil || math.Abs(got-want) > 1e-10*want {
		t.Errorf("dynamic H2Norm = %g, %v; want %g, nil", got, err, want)
	}
}

// UAZL45: a norm beyond float64 range is ErrOverflow, never +Inf with nil.
func TestH2NormOutOfRangeIsErrOverflow(t *testing.T) {
	big := mat.NewDense(2, 2, []float64{1e308, 0, 0, 1e308})
	tests := []struct {
		name string
		sys  func() (*System, error)
	}{
		{"continuous", func() (*System, error) {
			return New(mat.NewDense(2, 2, []float64{-0.1, 5, -5, -0.1}), big,
				mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 2, nil), 0)
		}},
		{"discrete", func() (*System, error) {
			return New(mat.NewDense(2, 2, []float64{0.9, 0.3, -0.3, 0.9}), big,
				mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 2, nil), 1)
		}},
		{"discrete static", func() (*System, error) {
			return NewGain(mat.NewDense(2, 2, []float64{1e308, 1e308, 1e308, 1e308}), 1)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sys, err := tt.sys()
			if err != nil {
				t.Fatal(err)
			}
			got, err := H2Norm(sys)
			if !errors.Is(err, ErrOverflow) {
				t.Errorf("H2Norm = %g, %v; want ErrOverflow", got, err)
			}
		})
	}
}
