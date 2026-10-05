package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
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

	if r.RHPPoles != 0 || r.Encirclements != 0 {
		t.Errorf("P=%d N=%d, want 0 0", r.RHPPoles, r.Encirclements)
	}
	checkNyquistGrid(t, r, 0, func(w float64) complex128 { return 1 / complex(0, w) })
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

func TestNyquist_DiscretePoleAtOrigin(t *testing.T) {
	// L = 0.5/z; L = 1/z would put the closed-loop pole at z = -1 on the contour.
	d, _ := New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	ny, err := d.Nyquist(nil, 50)
	if err != nil {
		t.Fatal(err)
	}
	w := ny.Omega
	if len(w) < 10 {
		t.Fatalf("len(omega) = %d", len(w))
	}
	for k, v := range w {
		if math.IsNaN(v) || math.IsInf(v, 0) || (k > 0 && v <= w[k-1]) {
			t.Fatalf("omega[%d] = %g not finite increasing", k, v)
		}
		z := cmplx.Exp(complex(0, v*0.1))
		if got, want := ny.Contour[k], 0.5/z; cmplx.Abs(got-want) > 1e-12 {
			t.Fatalf("H(%g) = %v, want %v", v, got, want)
		}
	}
	if math.Abs(w[len(w)-1]-math.Pi/0.1) > 1e-12 {
		t.Errorf("last omega = %g, want π/dt", w[len(w)-1])
	}
}

// nyquistTestTF realizes num/den (descending powers, len(num) <= len(den)) in
// controllable companion form, which has a non-symmetric A.
func nyquistTestTF(t *testing.T, num, den []float64, dt float64) *System {
	t.Helper()
	n := len(den) - 1
	d := make([]float64, n+1)
	for i := range den {
		d[i] = den[i] / den[0]
	}
	nm := make([]float64, n+1)
	copy(nm[n+1-len(num):], num)
	for i := range nm {
		nm[i] /= den[0]
	}
	a := make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -d[n-j]
	}
	b := make([]float64, n)
	b[n-1] = 1
	c := make([]float64, n)
	for j := range n {
		c[j] = nm[n-j] - nm[0]*d[n-j]
	}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{nm[0]}, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func polyEvalC(p []float64, x complex128) complex128 {
	var v complex128
	for _, c := range p {
		v = v*x + complex(c, 0)
	}
	return v
}

// checkNyquistGrid asserts a strictly increasing, finite, in-range grid with
// Contour[k] = h(Omega[k]) and the mirrored negative branch.
func checkNyquistGrid(t *testing.T, r *NyquistResult, dt float64, h func(float64) complex128) {
	t.Helper()
	if len(r.Omega) < 10 || len(r.Contour) != len(r.Omega) || len(r.ContourN) != len(r.Omega) {
		t.Fatalf("len Omega/Contour/ContourN = %d/%d/%d", len(r.Omega), len(r.Contour), len(r.ContourN))
	}
	for k, w := range r.Omega {
		if math.IsNaN(w) || math.IsInf(w, 0) || w <= 0 || (k > 0 && w <= r.Omega[k-1]) {
			t.Fatalf("Omega[%d] = %g not finite, positive, strictly increasing (prev %g)", k, w, r.Omega[max(k-1, 0)])
		}
		if dt > 0 && w > math.Pi/dt*(1+1e-12) {
			t.Fatalf("Omega[%d] = %g above Nyquist frequency %g", k, w, math.Pi/dt)
		}
		want := h(w)
		if got := r.Contour[k]; cmplx.Abs(got-want) > 1e-9*math.Max(1, cmplx.Abs(want)) {
			t.Fatalf("Contour[%d] at w=%g = %v, want %v", k, w, got, want)
		}
		if got := r.ContourN[len(r.Omega)-1-k]; got != cmplx.Conj(r.Contour[k]) {
			t.Fatalf("ContourN mirror mismatch at %d", k)
		}
	}
}

func nyquistCLUnstable(t *testing.T, sys *System) int {
	t.Helper()
	k, err := NewGain(mat.NewDense(1, 1, []float64{1}), sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(sys, k, -1)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := cl.Poles()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, p := range poles {
		if (sys.IsContinuous() && real(p) > 1e-9) || (sys.IsDiscrete() && cmplx.Abs(p) > 1+1e-9) {
			n++
		}
	}
	return n
}

// Loops with poles on the stability boundary (s=0, s=±j, z=1, z=-1, z=±j),
// unstable open loops, D != 0 and delays. N, P, Z are hand-derived
// (Routh/Jury or delay margin) and Z is cross-checked against the closed-loop
// poles where the loop is rational.
func TestNyquist_KnownLoops(t *testing.T) {
	tests := []struct {
		name     string
		num, den []float64
		dt       float64
		delay    float64
		n, p     int
	}{
		{"1/(s+1)", []float64{1}, []float64{1, 1}, 0, 0, 0, 0},
		{"1/s", []float64{1}, []float64{1, 0}, 0, 0, 0, 0},
		{"1/(s(s+1)^2)", []float64{1}, []float64{1, 2, 1, 0}, 0, 0, 0, 0},
		{"3/(s(s+1)^2)", []float64{3}, []float64{1, 2, 1, 0}, 0, 0, 2, 0},
		{"(s+1)/s^2", []float64{1, 1}, []float64{1, 0, 0}, 0, 0, 0, 0},
		{"(s+0.2)/(s^2(s+1))", []float64{1, 0.2}, []float64{1, 1, 0, 0}, 0, 0, 0, 0},
		{"s/(s^2+1)", []float64{1, 0}, []float64{1, 0, 1}, 0, 0, 0, 0},
		{"-s/(s^2+1)", []float64{-1, 0}, []float64{1, 0, 1}, 0, 0, 2, 0},
		{"2/(s-1)", []float64{2}, []float64{1, -1}, 0, 0, -1, 1},
		{"(2s+1)/(s-0.5)", []float64{2, 1}, []float64{1, -0.5}, 0, 0, -1, 1},
		{"180/((s+1)(s+2)(s+3))", []float64{180}, []float64{1, 6, 11, 6}, 0, 0, 2, 0},
		{"2e^-s/(s+1)", []float64{2}, []float64{1, 1}, 0, 1, 0, 0},
		{"3e^-s/(s+1)", []float64{3}, []float64{1, 1}, 0, 1, 2, 0},
		{"e^-s/s", []float64{1}, []float64{1, 0}, 0, 1, 0, 0},
		{"2e^-s/s", []float64{2}, []float64{1, 0}, 0, 1, 2, 0},
		{"0.5/(z-1)", []float64{0.5}, []float64{1, -1}, 0.1, 0, 0, 0},
		{"3/(z-1)", []float64{3}, []float64{1, -1}, 0.1, 0, 1, 0},
		{"0.2/((z-1)(z-0.5))", []float64{0.2}, []float64{1, -1.5, 0.5}, 0.1, 0, 0, 0},
		{"1/((z-1)(z-0.5))", []float64{1}, []float64{1, -1.5, 0.5}, 0.1, 0, 2, 0},
		{"0.1/(z-1)^2", []float64{0.1}, []float64{1, -2, 1}, 0.1, 0, 2, 0},
		{"-0.5/(z+1)", []float64{-0.5}, []float64{1, 1}, 0.1, 0, 0, 0},
		{"0.5/(z+1)", []float64{0.5}, []float64{1, 1}, 0.1, 0, 1, 0},
		{"-0.5/(z^2+1)", []float64{-0.5}, []float64{1, 0, 1}, 1, 0, 0, 0},
		{"0.5/(z^2+1)", []float64{0.5}, []float64{1, 0, 1}, 1, 0, 2, 0},
		{"2/(z-2)", []float64{2}, []float64{1, -2}, 0.1, 0, -1, 1},
		{"-10/(s(s+1000))", []float64{-10}, []float64{1, 1000, 0}, 0, 0, 1, 0},
		{"1/((s-0.005)(s+1000))", []float64{1}, []float64{1, 999.995, -5}, 0, 0, 0, 1},
		{"0.5/(z-0.999)", []float64{0.5}, []float64{1, -0.999}, 0.001, 0, 0, 0},
		{"2.5/(z-0.999)", []float64{2.5}, []float64{1, -0.999}, 0.001, 0, 1, 0},
		{"-0.01/((z-1)(z-0.01))", []float64{-0.01}, []float64{1, -1.01, 0.01}, 0.1, 0, 1, 0},
		{"(2z+1)/(z-1.5)", []float64{2, 1}, []float64{1, -1.5}, 0.1, 0, -1, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := nyquistTestTF(t, tc.num, tc.den, tc.dt)
			if tc.delay > 0 {
				if err := sys.SetInputDelay([]float64{tc.delay}); err != nil {
					t.Fatal(err)
				}
			}
			r, err := sys.Nyquist(nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			if r.Encirclements != tc.n || r.RHPPoles != tc.p || r.RHPZerosCL != tc.n+tc.p {
				t.Errorf("N=%d P=%d Z=%d, want N=%d P=%d Z=%d",
					r.Encirclements, r.RHPPoles, r.RHPZerosCL, tc.n, tc.p, tc.n+tc.p)
			}
			if tc.delay == 0 {
				if z := nyquistCLUnstable(t, sys); z != tc.n+tc.p {
					t.Errorf("closed-loop unstable poles = %d, table Z = %d", z, tc.n+tc.p)
				}
			}
			checkNyquistGrid(t, r, tc.dt, func(w float64) complex128 {
				x := complex(0, w)
				if tc.dt > 0 {
					x = cmplx.Exp(complex(0, w*tc.dt))
				}
				return polyEvalC(tc.num, x) / polyEvalC(tc.den, x) * cmplx.Exp(complex(0, -w*tc.delay))
			})
		})
	}
}

func TestNyquist_DiscreteIntegratorGrid(t *testing.T) {
	const dt = 0.1
	sys := nyquistTestTF(t, []float64{0.5}, []float64{1, -1}, dt)
	r, err := sys.Nyquist(nil, 100)
	if err != nil {
		t.Fatal(err)
	}
	checkNyquistGrid(t, r, dt, func(w float64) complex128 {
		return 0.5 / (cmplx.Exp(complex(0, w*dt)) - 1)
	})
	if math.Abs(r.Omega[len(r.Omega)-1]-math.Pi/dt) > 1e-12 {
		t.Errorf("last omega = %g, want π/dt", r.Omega[len(r.Omega)-1])
	}
}

func TestNyquist_DiscreteDelayMatchesAbsorbed(t *testing.T) {
	for _, k := range []float64{0.1, 0.5} {
		delayed := nyquistTestTF(t, []float64{k}, []float64{1, -1}, 0.1)
		if err := delayed.SetInputDelay([]float64{2}); err != nil {
			t.Fatal(err)
		}
		absorbed := nyquistTestTF(t, []float64{k}, []float64{1, -1, 0, 0}, 0.1)
		rd, err := delayed.Nyquist(nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ra, err := absorbed.Nyquist(nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		z := nyquistCLUnstable(t, absorbed)
		if rd.RHPZerosCL != z || ra.RHPZerosCL != z {
			t.Errorf("k=%g: Z delayed=%d absorbed=%d, closed-loop unstable poles=%d", k, rd.RHPZerosCL, ra.RHPZerosCL, z)
		}
	}
}

func TestNyquist_SimilarityTransformedDoubleIntegrator(t *testing.T) {
	// L = (s+1)/s^2 under x = T z with non-orthogonal T; closed loop s^2+s+1.
	base := nyquistTestTF(t, []float64{1, 1}, []float64{1, 0, 0}, 0)
	T := mat.NewDense(2, 2, []float64{2, 1, 0.5, 3})
	var Ti mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	var a, tmp, b, c mat.Dense
	tmp.Mul(&Ti, base.A)
	a.Mul(&tmp, T)
	b.Mul(&Ti, base.B)
	c.Mul(base.C, T)
	sys, err := New(&a, &b, &c, mat.DenseCopyOf(base.D), 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.RHPZerosCL != 0 {
		t.Errorf("N=%d P=%d Z=%d, want Z=0", r.Encirclements, r.RHPPoles, r.RHPZerosCL)
	}
}

func nyquistPolyMul(a, b []float64) []float64 {
	r := make([]float64, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			r[i+j] += a[i] * b[j]
		}
	}
	return r
}

// Random loops (integrators, z=1 poles, lightly damped pairs, stiff and unstable
// poles, D != 0) against the closed-loop pole count; loops with a
// closed-loop pole within 1e-3 of the stability boundary are skipped.
func TestNyquist_RandomLoopsMatchClosedLoop(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	checked := 0
	for trial := range 1000 {
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		den := []float64{1}
		nInt := rng.IntN(3)
		for i := range 1 + rng.IntN(4) {
			r := rng.NormFloat64() * 2 * math.Pow(10, float64(rng.IntN(5)-2))
			if dt > 0 {
				r = rng.Float64()*2.4 - 1.2
			}
			if i < nInt {
				r = 0
				if dt > 0 {
					r = 1
				}
			}
			den = nyquistPolyMul(den, []float64{1, -r})
		}
		if rng.IntN(3) == 0 {
			w := rng.Float64() * 3
			if dt > 0 {
				rr := 0.7 + rng.Float64()*0.4
				den = nyquistPolyMul(den, []float64{1, -2 * rr * math.Cos(w*0.3), rr * rr})
			} else {
				den = nyquistPolyMul(den, []float64{1, 2 * rng.Float64() * 0.3 * w, w * w})
			}
		}
		num := []float64{rng.NormFloat64() * 3}
		for range rng.IntN(len(den)) {
			num = nyquistPolyMul(num, []float64{1, rng.NormFloat64() * 2})
		}
		sys := nyquistTestTF(t, num, den, dt)
		k, err := NewGain(mat.NewDense(1, 1, []float64{1}), dt)
		if err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(sys, k, -1)
		if err != nil {
			continue
		}
		clPoles, err := cl.Poles()
		if err != nil {
			t.Fatal(err)
		}
		z, marginal := 0, false
		for _, p := range clPoles {
			m := real(p)
			if dt > 0 {
				m = cmplx.Abs(p) - 1
			}
			marginal = marginal || math.Abs(m) < 1e-3
			if m > 0 {
				z++
			}
		}
		if marginal {
			continue
		}
		checked++
		r, err := sys.Nyquist(nil, 0)
		if err != nil {
			t.Fatalf("num=%v den=%v dt=%g: %v", num, den, dt, err)
		}
		if r.RHPZerosCL != z {
			t.Errorf("num=%v den=%v dt=%g: N=%d P=%d Z=%d, closed-loop unstable poles %d",
				num, den, dt, r.Encirclements, r.RHPPoles, r.RHPZerosCL, z)
		}
	}
	if checked < 400 {
		t.Fatalf("only %d loops checked", checked)
	}
}

func TestNyquistClosedLoopPoleOnBoundaryErrors(t *testing.T) {
	// L = 8/(s+1)^3: 1+L = 0 at s = ±j√3, a marginally stable closed loop.
	cont, err := New(
		mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}),
		mat.NewDense(3, 1, []float64{0, 0, 8}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	// L = 1.5/(z-0.5): 1+L = 0 at z = -1.
	disc, err := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1.5}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"continuous": cont, "discrete": disc} {
		_, err := sys.Nyquist(nil, 0)
		if !errors.Is(err, ErrSingularTransform) || !strings.HasPrefix(err.Error(), "Nyquist: ") {
			t.Errorf("%s: err = %v, want Nyquist: ... ErrSingularTransform", name, err)
		}
	}

	for _, k := range []float64{7.9, 8.1} {
		near := cont.Copy()
		near.B.Set(2, 0, k)
		res, err := near.Nyquist(nil, 0)
		if err != nil {
			t.Fatalf("k=%g: %v", k, err)
		}
		want := 0
		if k > 8 {
			want = 2
		}
		if res.RHPZerosCL != want {
			t.Errorf("k=%g: RHPZerosCL = %d, want %d", k, res.RHPZerosCL, want)
		}
	}
}

func TestNyquistValidatesArguments(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	disc, err := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		sys    *System
		omega  []float64
		points int
	}{
		{"empty", sys, []float64{}, 0},
		{"negative", sys, []float64{-1, 1}, 0},
		{"NaN", sys, []float64{1, math.NaN()}, 0},
		{"Inf", sys, []float64{math.Inf(1)}, 0},
		{"above Nyquist", disc, []float64{1, 40}, 0},
		{"nPoints", sys, nil, -3},
		{"nil", nil, nil, 0},
	}
	for _, c := range cases {
		_, err := c.sys.Nyquist(c.omega, c.points)
		if !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "Nyquist: ") {
			t.Errorf("%s: err = %v, want Nyquist: ... ErrInvalidArgument", c.name, err)
		}
	}
	if _, err := disc.Nyquist([]float64{0, math.Pi / 0.1}, 0); err != nil {
		t.Errorf("omega up to π/dt: %v", err)
	}
}
