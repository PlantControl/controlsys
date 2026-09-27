package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestNyquist_StableFirstOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if r.Encirclements != 0 {
		t.Errorf("Encirclements = %d, want 0", r.Encirclements)
	}
	if r.RHPZerosCL != 0 {
		t.Errorf("RHPZerosCL = %d, want 0", r.RHPZerosCL)
	}

	for k := range r.Contour {
		if real(r.Contour[k]) < -1 {
			t.Errorf("contour crosses -1 at index %d: %v", k, r.Contour[k])
			break
		}
	}
}

func TestNyquist_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}

	hasLargeValue := false
	for _, v := range r.Contour {
		if cmplx.Abs(v) > 100 {
			hasLargeValue = true
			break
		}
	}
	if !hasLargeValue {
		t.Error("expected large magnitude values near indentation arc")
	}
}

func TestNyquist_UnstableOL_StableCL(t *testing.T) {
	// G(s) = 2(s+1)/((s-1)(s+2)) = 2(s+1)/(s^2+s-2)
	// Poles at s=1 (RHP) and s=-2
	// With unity feedback: closed-loop should be stable
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, 1, 2, -1}, // companion form for s^2+s-2
		[]float64{0, 1},
		[]float64{4, 2}, // 2(s+1) = 2s+2, so C*[x1,x2]' with B gives 2s+2 in numerator
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 1000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 1 {
		t.Errorf("RHPPoles = %d, want 1", r.RHPPoles)
	}
	if r.Encirclements != -1 {
		t.Errorf("Encirclements = %d, want -1 (one CCW)", r.Encirclements)
	}
	if r.RHPZerosCL != 0 {
		t.Errorf("RHPZerosCL = %d, want 0", r.RHPZerosCL)
	}
}

func TestNyquist_StableOL_UnstableCL(t *testing.T) {
	// G(s) = K / ((s+1)(s+2)(s+3)), K=180
	// Poles all LHP, but high gain causes instability
	// Critical gain K_c = 60, with K=180: 2 CW encirclements
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6, // char poly: s^3+6s^2+11s+6
		},
		[]float64{0, 0, 1},
		[]float64{180, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 2000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if r.Encirclements != 2 {
		t.Errorf("Encirclements = %d, want 2", r.Encirclements)
	}
	if r.RHPZerosCL != 2 {
		t.Errorf("RHPZerosCL = %d, want 2", r.RHPZerosCL)
	}
}

func TestNyquist_Discrete(t *testing.T) {
	// G(z) = 0.5/(z-0.9), stable discrete system
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if r.Encirclements != 0 {
		t.Errorf("Encirclements = %d, want 0", r.Encirclements)
	}
}

func TestNyquist_PureGain(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist([]float64{1.0, 10.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.Encirclements != 0 {
		t.Errorf("Encirclements = %d, want 0", r.Encirclements)
	}

	for k, v := range r.Contour {
		if math.Abs(real(v)-0.5) > 1e-6 || math.Abs(imag(v)) > 1e-6 {
			t.Errorf("contour[%d] = %v, want 0.5+0i", k, v)
			break
		}
	}
}

func TestNyquist_WithDelay(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	err = sys.SetInputDelay([]float64{0.5})
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.Encirclements != 0 {
		t.Errorf("Encirclements = %d, want 0", r.Encirclements)
	}

	if len(r.Contour) == 0 {
		t.Fatal("empty contour")
	}
}

func TestNyquist_MIMO_Error(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = sys.Nyquist(nil, 0)
	if err == nil {
		t.Fatal("expected error for MIMO system")
	}
}

// python-control: pole at origin - tf([3],[1,2,2,1,0])
// Uses indentation contour around origin
func TestNyquist_PoleAtOrigin(t *testing.T) {
	// s^4+2s^3+2s^2+s = s(s^3+2s^2+2s+1) = s(s+1)(s^2+s+1)
	sys, err := NewFromSlices(4, 1, 1,
		[]float64{
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			0, -1, -2, -2,
		},
		[]float64{0, 0, 0, 1},
		[]float64{3, 0, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 2000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if len(r.Contour) == 0 {
		t.Fatal("empty contour")
	}
}

// python-control FBS: L(s) = 1/(s*(s+1)^2)
// Pole at origin, open-loop stable except for integrator
func TestNyquist_IntegratorWithSecondOrder(t *testing.T) {
	// s(s+1)^2 = s^3 + 2s^2 + s
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			0, -1, -2,
		},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 2000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
}

// python-control FBS: 3*(s+6)^2/(s*(s+1)^2)
// Non-minimum phase with pole at origin
func TestNyquist_FBS_Figure10_10(t *testing.T) {
	// Numerator: 3*(s+6)^2 = 3*(s^2+12s+36) = 3s^2+36s+108
	// Denominator: s*(s+1)^2 = s^3+2s^2+s
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			0, -1, -2,
		},
		[]float64{0, 0, 1},
		[]float64{108, 36, 3},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 2000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if len(r.Contour) == 0 {
		t.Fatal("empty contour")
	}
}

// Two CW encirclements from high gain (K > Kc)
func TestNyquist_HighGain_TwoEncirclements(t *testing.T) {
	// G = 200/((s+1)(s+2)(s+3)) = 200/(s^3+6s^2+11s+6)
	// Kc = 60, so K=200 should give 2 CW encirclements
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{0, 0, 1},
		[]float64{200, 0, 0},
		[]float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 2000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if r.Encirclements != 2 {
		t.Errorf("Encirclements = %d, want 2", r.Encirclements)
	}
	if r.RHPZerosCL != 2 {
		t.Errorf("RHPZerosCL = %d, want 2", r.RHPZerosCL)
	}
}

// Discrete system with encirclement
func TestNyquist_Discrete_Unstable(t *testing.T) {
	// High gain discrete system should have encirclements
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.99}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{0}), 0.01)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nyquist(nil, 1000)
	if err != nil {
		t.Fatal(err)
	}

	if r.RHPPoles != 0 {
		t.Errorf("RHPPoles = %d, want 0", r.RHPPoles)
	}
	if len(r.Contour) == 0 {
		t.Fatal("empty contour")
	}
}

func TestWindingNumber(t *testing.T) {
	tests := []struct {
		name    string
		contour []complex128
		point   complex128
		want    int
	}{
		{
			name:  "CW circle around origin",
			point: 0,
			want:  -1,
			contour: func() []complex128 {
				n := 100
				c := make([]complex128, n)
				for i := range n {
					theta := -2 * math.Pi * float64(i) / float64(n)
					c[i] = cmplx.Exp(complex(0, theta))
				}
				return c
			}(),
		},
		{
			name:  "CCW circle around origin",
			point: 0,
			want:  1,
			contour: func() []complex128 {
				n := 100
				c := make([]complex128, n)
				for i := range n {
					theta := 2 * math.Pi * float64(i) / float64(n)
					c[i] = cmplx.Exp(complex(0, theta))
				}
				return c
			}(),
		},
		{
			name:  "circle not enclosing point",
			point: complex(5, 0),
			want:  0,
			contour: func() []complex128 {
				n := 100
				c := make([]complex128, n)
				for i := range n {
					theta := 2 * math.Pi * float64(i) / float64(n)
					c[i] = cmplx.Exp(complex(0, theta))
				}
				return c
			}(),
		},
		{
			name:  "double CCW wrap",
			point: 0,
			want:  2,
			contour: func() []complex128 {
				n := 200
				c := make([]complex128, n)
				for i := range n {
					theta := 4 * math.Pi * float64(i) / float64(n)
					c[i] = cmplx.Exp(complex(0, theta))
				}
				return c
			}(),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := windingNumber(tc.contour, tc.point)
			if got != tc.want {
				t.Errorf("windingNumber = %d, want %d", got, tc.want)
			}
		})
	}
}
