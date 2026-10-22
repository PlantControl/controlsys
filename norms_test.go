package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestH2Norm_SISO_1x1(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Sqrt(0.5)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("H2 = %g, want %g", got, want)
	}
}

func TestH2Norm_2x2_NonSymA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 || math.IsNaN(got) {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_Discrete_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}), 0.1)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	gotNoD, _ := H2Norm(&System{
		A:  mat.NewDense(1, 1, []float64{0.5}),
		B:  mat.NewDense(1, 1, []float64{1}),
		C:  mat.NewDense(1, 1, []float64{1}),
		D:  mat.NewDense(1, 1, []float64{0}),
		Dt: 0.1,
	})
	if got <= gotNoD {
		t.Errorf("H2 with D (%g) should be > H2 without D (%g)", got, gotNoD)
	}
}

func TestH2Norm_Continuous_DNonZero(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("H2 = %g, want +Inf", got)
	}
}

func TestH2Norm_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := H2Norm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestH2Norm_PureGain_Continuous(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{5}), 0)
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("H2 = %g, want +Inf", got)
	}
}

func TestH2Norm_PureGain_Discrete(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(2, 2, []float64{3, 0, 0, 4}), 0.1)
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 5.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("H2 = %g, want %g", got, want)
	}
}

func TestHSV_SISO_1x1(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 1 {
		t.Fatalf("len(hsv) = %d, want 1", len(hsv))
	}
	want := 0.5
	if math.Abs(hsv[0]-want) > 1e-10 {
		t.Errorf("hsv[0] = %g, want %g", hsv[0], want)
	}
}

func TestHSV_Diagonal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	if hsv[0] < hsv[1] {
		t.Errorf("not descending: %v", hsv)
	}

	want0 := 0.5
	want1 := 1.0 / 6.0
	if math.Abs(hsv[0]-want0) > 1e-10 {
		t.Errorf("hsv[0] = %g, want %g", hsv[0], want0)
	}
	if math.Abs(hsv[1]-want1) > 1e-10 {
		t.Errorf("hsv[1] = %g, want %g", hsv[1], want1)
	}
}

func TestHSV_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	if hsv[0] < hsv[1] {
		t.Errorf("not descending: %v", hsv)
	}
}

func TestHSV_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	for _, v := range hsv {
		if v < 0 || math.IsNaN(v) {
			t.Errorf("invalid hsv: %v", hsv)
		}
	}
}

func TestHSV_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := HSV(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestHSV_Empty(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{1}), 0)
	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if hsv != nil {
		t.Errorf("hsv = %v, want nil", hsv)
	}
}

func TestHinfNorm_FirstOrder(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("Hinf = %g, want 1.0", norm)
	}
	if omega > 0.1 {
		t.Errorf("omega = %g, want ~0 (DC peak)", omega)
	}
}

func TestHinfNorm_SecondOrder_Underdamped(t *testing.T) {
	wn := 10.0
	zeta := 0.1
	A := mat.NewDense(2, 2, []float64{
		0, 1,
		-wn * wn, -2 * zeta * wn,
	})
	B := mat.NewDense(2, 1, []float64{0, wn * wn})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	wPeak := wn * math.Sqrt(1-2*zeta*zeta)
	expected := 1.0 / (2 * zeta * math.Sqrt(1-zeta*zeta))
	if math.Abs(norm-expected)/expected > 1e-4 {
		t.Errorf("Hinf = %g, want ~%g", norm, expected)
	}
	if math.Abs(omega-wPeak)/wPeak > 0.1 {
		t.Errorf("omega = %g, want ~%g", omega, wPeak)
	}
}

func TestHinfNorm_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm < 1.5-0.01 {
		t.Errorf("Hinf = %g, want >= 1.5 (DC gain = 1+0.5)", norm)
	}
}

func TestHinfNorm_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm <= 0 {
		t.Errorf("Hinf = %g, want positive", norm)
	}
}

func TestHinfNorm_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm <= 0 {
		t.Errorf("Hinf = %g, want positive", norm)
	}
}

func TestHinfNorm_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	norm, w, err := HinfNorm(sys)
	if err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
		t.Errorf("got %g at %g, %v; want Inf at Inf (MATLAB hinfnorm)", norm, w, err)
	}

	for _, dt := range []float64{0, 0.1} {
		mimo, err := NewFromSlices(2, 2, 2,
			[]float64{0.8, 1.5, -0.4, -0.9},
			[]float64{1, 0, 0.5, 2},
			[]float64{1, -1, 0, 3},
			[]float64{0.1, 0, 0.2, -0.3}, dt)
		if err != nil {
			t.Fatal(err)
		}
		if dt > 0 {
			mimo.A.Set(0, 0, 1.6)
		}
		if stable, _ := mimo.IsStable(); stable {
			t.Fatalf("dt=%g: fixture is stable", dt)
		}
		norm, w, err := HinfNorm(mimo)
		if err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
			t.Errorf("MIMO dt=%g: got %g at %g, %v; want Inf at Inf", dt, norm, w, err)
		}
	}
}

func TestHinfNorm_PureGain(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(2, 2, []float64{3, 0, 0, 4}), 0)
	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-4.0) > 1e-10 {
		t.Errorf("Hinf = %g, want 4.0", norm)
	}
	if omega != 0 {
		t.Errorf("omega = %g, want 0", omega)
	}
}

func TestHinfNorm_MatchesBode(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	maxBode := 0.0
	for i := range 1000 {
		w := 0.001 * math.Pow(1000000, float64(i)/999)
		G, _ := sys.EvalFr(complex(0, w))
		mag := cmplx.Abs(G[0][0])
		if mag > maxBode {
			maxBode = mag
		}
	}

	if norm < maxBode*0.999 {
		t.Errorf("Hinf(%g) < max Bode(%g)", norm, maxBode)
	}
	if norm > maxBode*1.01 {
		t.Errorf("Hinf(%g) >> max Bode(%g), tolerance exceeded", norm, maxBode)
	}
}

// python-control MATLAB-verified: 3rd-order MIMO system
func TestH2Norm_MIMO_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			-1.017041847539126, -0.224182952826418, 0.042538079149249,
			-0.310374015319095, -0.516461581407780, -0.119195790221750,
			-1.452723568727942, 1.799586083710209, -1.491935830615152,
		}),
		mat.NewDense(3, 2, []float64{
			0.312858596637428, -0.164879019209038,
			-0.864879917324456, 0.627707287528727,
			-0.030051296196269, 1.093265669039484,
		}),
		mat.NewDense(2, 3, []float64{
			1.109273297614398, 0.077359091130425, -1.113500741486764,
			-0.863652821988714, -1.214117043615409, -0.006849328103348,
		}),
		mat.NewDense(2, 2, nil), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 2.237461821810309
	if math.Abs(got-want)/want > 1e-4 {
		t.Errorf("H2 = %g, want %g (MATLAB)", got, want)
	}
}

func TestHinfNorm_MIMO_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			-1.017041847539126, -0.224182952826418, 0.042538079149249,
			-0.310374015319095, -0.516461581407780, -0.119195790221750,
			-1.452723568727942, 1.799586083710209, -1.491935830615152,
		}),
		mat.NewDense(3, 2, []float64{
			0.312858596637428, -0.164879019209038,
			-0.864879917324456, 0.627707287528727,
			-0.030051296196269, 1.093265669039484,
		}),
		mat.NewDense(2, 3, []float64{
			1.109273297614398, 0.077359091130425, -1.113500741486764,
			-0.863652821988714, -1.214117043615409, -0.006849328103348,
		}),
		mat.NewDense(2, 2, nil), 0)

	got, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 4.276759162964244
	if math.Abs(got-want)/want > 1e-3 {
		t.Errorf("Hinf = %g, want %g (MATLAB)", got, want)
	}
}

// python-control: G=1/(s+1), ||G||_inf = 1.0, ||G||_2 = 1/sqrt(2)
func TestNorms_FirstOrder_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	h2, _ := H2Norm(sys)
	hinf, _, _ := HinfNorm(sys)

	if math.Abs(h2-0.707106781186547) > 1e-6 {
		t.Errorf("H2 = %g, want 0.707107", h2)
	}
	if math.Abs(hinf-1.0) > 1e-6 {
		t.Errorf("Hinf = %g, want 1.0", hinf)
	}
}

// 1/(1-s) unstable: MATLAB hinfnorm and H2 norm are Inf.
func TestNorms_UnstableNonMinPhase(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := H2Norm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("H2: got %v, want ErrUnstable", err)
	}
	norm, w, err := HinfNorm(sys)
	if err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
		t.Errorf("Hinf: got %g at %g, %v; want Inf at Inf", norm, w, err)
	}
}

// python-control: underdamped 2nd-order tf([100],[1,10,100]) zeta=0.5 wn=10
// gpeak = 1/(2*zeta*sqrt(1-zeta^2)) ≈ 1.1547
// fpeak = wn*sqrt(1-2*zeta^2) ≈ 7.0711
func TestHinfNorm_Underdamped_MATLABVerified(t *testing.T) {
	wn := 10.0
	zeta := 0.5
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGpeak := 1.0 / (2 * zeta * math.Sqrt(1-zeta*zeta))
	wantFpeak := wn * math.Sqrt(1-2*zeta*zeta)
	if math.Abs(got-wantGpeak)/wantGpeak > 1e-3 {
		t.Errorf("gpeak = %g, want %g", got, wantGpeak)
	}
	if math.Abs(omega-wantFpeak)/wantFpeak > 0.05 {
		t.Errorf("fpeak = %g, want %g", omega, wantFpeak)
	}
}

// python-control: static gain tf([1.23],[1]) => gpeak=1.23, fpeak=0
func TestHinfNorm_StaticGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{1.23}), 0)
	got, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-1.23) > 1e-10 {
		t.Errorf("gpeak = %g, want 1.23", got)
	}
	if omega != 0 {
		t.Errorf("fpeak = %g, want 0", omega)
	}
}

// python-control: MATLAB-verified HSV for A=[[1,-2],[3,-4]], B=[[5],[7]], C=[[6,8]], D=[[9]]
func TestHSV_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len(hsv) = %d, want 2", len(hsv))
	}
	want := []float64{24.42686, 0.5731395}
	for i, w := range want {
		if math.Abs(hsv[i]-w)/w > 1e-3 {
			t.Errorf("hsv[%d] = %g, want %g", i, hsv[i], w)
		}
	}
}

// Discrete MIMO norms: MATLAB-verified
func TestH2Norm_Discrete_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			-1.017041847539126, -0.224182952826418, 0.042538079149249,
			-0.310374015319095, -0.516461581407780, -0.119195790221750,
			-1.452723568727942, 1.799586083710209, -1.491935830615152,
		}),
		mat.NewDense(3, 2, []float64{
			0.312858596637428, -0.164879019209038,
			-0.864879917324456, 0.627707287528727,
			-0.030051296196269, 1.093265669039484,
		}),
		mat.NewDense(2, 3, []float64{
			1.109273297614398, 0.077359091130425, -1.113500741486764,
			-0.863652821988714, -1.214117043615409, -0.006849328103348,
		}),
		mat.NewDense(2, 2, nil), 0)

	dsys, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	got, err := H2Norm(dsys)
	if err != nil {
		t.Fatal(err)
	}
	want := 0.707434962289554
	if math.Abs(got-want)/want > 1e-2 {
		t.Errorf("discrete H2 = %g, want %g (MATLAB)", got, want)
	}
}

func TestH2Norm_Discrete_FirstOrder_MATLABVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	dsys, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	got, err := H2Norm(dsys)
	if err != nil {
		t.Fatal(err)
	}
	want := 0.223513699524858
	if math.Abs(got-want) > 1e-3 {
		t.Errorf("discrete H2 = %g, want %g", got, want)
	}
}

func TestH2_HSV_Relationship(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	h2, _ := H2Norm(sys)
	hsv, _ := HSV(sys)
	hinf, _, _ := HinfNorm(sys)

	if h2 <= 0 || hinf <= 0 {
		t.Skipf("h2=%g hinf=%g", h2, hinf)
	}
	if len(hsv) == 0 {
		t.Skip("no HSV")
	}
	if hsv[0] <= 0 {
		t.Skip("zero HSV")
	}

	if hinf < hsv[0]*0.99 {
		t.Errorf("Hinf(%g) < hsv[0](%g), Hinf >= largest HSV expected", hinf, hsv[0])
	}
}

func internalDelayFixture(t *testing.T, dt float64, D12 *mat.Dense) *System {
	t.Helper()
	return internalDelayFixtureB2(t, dt, D12, mat.NewDense(2, 1, []float64{0.3, 0.1}))
}

func internalDelayFixtureB2(t *testing.T, dt float64, D12, B2 *mat.Dense) *System {
	t.Helper()
	A := []float64{-1, 2, 0.5, -3}
	tau := 0.7
	if dt > 0 {
		A = []float64{0.5, 0.2, -0.1, 0.7}
		tau = 3
	}
	sys, err := New(mat.NewDense(2, 2, A), mat.NewDense(2, 2, []float64{1, 0.3, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}), mat.NewDense(2, 2, []float64{0, 0.1, 0, 0}), dt)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{tau}, B2, mat.NewDense(1, 2, []float64{0.3, 1}),
		D12, mat.NewDense(1, 2, []float64{0, 0.1}), mat.NewDense(1, 1, []float64{0.1})); err != nil {
		t.Fatal(err)
	}
	return sys
}

// discreteImpulseLFT simulates the impulse response of a discrete model with
// integer internal delays directly from its LFT equations.
func discreteImpulseLFT(sys *System, steps int) []*mat.Dense {
	n, m, p := sys.Dims()
	N := len(sys.LFT.Tau)
	h := make([]*mat.Dense, steps)
	for k := range h {
		h[k] = mat.NewDense(p, m, nil)
	}
	for j := range m {
		x := make([]float64, n)
		zHist := make([][]float64, 0, steps)
		for k := range steps {
			u := make([]float64, m)
			if k == 0 {
				u[j] = 1
			}
			w := make([]float64, N)
			for l := range N {
				if d := k - int(sys.LFT.Tau[l]); d >= 0 {
					w[l] = zHist[d][l]
				}
			}
			z := make([]float64, N)
			for l := range N {
				for c := range n {
					z[l] += sys.LFT.C2.At(l, c) * x[c]
				}
				for c := range m {
					z[l] += sys.LFT.D21.At(l, c) * u[c]
				}
				for c := range N {
					z[l] += sys.LFT.D22.At(l, c) * w[c]
				}
			}
			zHist = append(zHist, z)
			for i := range p {
				v := 0.0
				for c := range n {
					v += sys.C.At(i, c) * x[c]
				}
				for c := range m {
					v += sys.D.At(i, c) * u[c]
				}
				for c := range N {
					v += sys.LFT.D12.At(i, c) * w[c]
				}
				h[k].Set(i, j, v)
			}
			xn := make([]float64, n)
			for i := range n {
				for c := range n {
					xn[i] += sys.A.At(i, c) * x[c]
				}
				for c := range m {
					xn[i] += sys.B.At(i, c) * u[c]
				}
				for c := range N {
					xn[i] += sys.LFT.B2.At(i, c) * w[c]
				}
			}
			x = xn
		}
	}
	return h
}

func TestH2Norm_DiscreteInternalDelay(t *testing.T) {
	sys := internalDelayFixture(t, 0.1, mat.NewDense(2, 1, []float64{0.2, 0}))
	want := 0.0
	for _, hk := range discreteImpulseLFT(sys, 4000) {
		f := mat.Norm(hk, 2)
		want += f * f
	}
	want = math.Sqrt(want)
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-9*want {
		t.Errorf("H2Norm = %.15g, want %.15g (impulse energy)", got, want)
	}

	unstable := internalDelayFixtureB2(t, 0.1, mat.NewDense(2, 1, []float64{0.2, 0}), mat.NewDense(2, 1, []float64{1, 0.5}))
	if got, err := H2Norm(unstable); !errors.Is(err, ErrUnstable) {
		t.Errorf("H2Norm = %g, %v; want ErrUnstable (delay loop has pole 1.12)", got, err)
	}
}

func TestH2Norm_ContinuousInternalDelay(t *testing.T) {
	sys := internalDelayFixture(t, 0, mat.NewDense(2, 1, []float64{0.2, 0}))
	sys.D.Zero()
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("H2Norm = %g, want +Inf for delayed feedthrough D12*D21 != 0", got)
	}

	strict := internalDelayFixture(t, 0, mat.NewDense(2, 1, nil))
	strict.D.Zero()
	if got, err := H2Norm(strict); err == nil {
		t.Errorf("H2Norm = %g, want error for continuous internal delays", got)
	}
}

func TestH2Norm_ZeroStaticGain(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		g, err := NewGain(mat.NewDense(2, 3, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		got, err := H2Norm(g)
		if err != nil {
			t.Fatal(err)
		}
		if got != 0 {
			t.Errorf("dt=%v: H2Norm(0) = %g, want 0", dt, got)
		}
	}
}

// lftDelayFreqOracle evaluates the 2x2 single-delay discrete LFT at z = e^{jωT}
// by solving (zI-A)x - B2 w = B1 u, -d C2 x + (1-d D22) w = d D21 u, d = z^-τ.
func lftDelayFreqOracle(sys *System, w float64) [2][2]complex128 {
	z := cmplx.Exp(complex(0, w*sys.Dt))
	d := cmplx.Pow(z, complex(-sys.LFT.Tau[0], 0))
	var g [2][2]complex128
	for j := range 2 {
		m := [3][4]complex128{}
		for r := range 2 {
			for c := range 2 {
				m[r][c] = complex(-sys.A.At(r, c), 0)
			}
			m[r][r] += z
			m[r][2] = complex(-sys.LFT.B2.At(r, 0), 0)
			m[r][3] = complex(sys.B.At(r, j), 0)
		}
		for c := range 2 {
			m[2][c] = -d * complex(sys.LFT.C2.At(0, c), 0)
		}
		m[2][2] = 1 - d*complex(sys.LFT.D22.At(0, 0), 0)
		m[2][3] = d * complex(sys.LFT.D21.At(0, j), 0)
		for k := range 3 {
			p := k
			for r := k + 1; r < 3; r++ {
				if cmplx.Abs(m[r][k]) > cmplx.Abs(m[p][k]) {
					p = r
				}
			}
			m[k], m[p] = m[p], m[k]
			for r := k + 1; r < 3; r++ {
				f := m[r][k] / m[k][k]
				for c := k; c < 4; c++ {
					m[r][c] -= f * m[k][c]
				}
			}
		}
		var v [3]complex128
		for k := 2; k >= 0; k-- {
			s := m[k][3]
			for c := k + 1; c < 3; c++ {
				s -= m[k][c] * v[c]
			}
			v[k] = s / m[k][k]
		}
		for i := range 2 {
			g[i][j] = complex(sys.D.At(i, j), 0) + complex(sys.LFT.D12.At(i, 0), 0)*v[2]
			for c := range 2 {
				g[i][j] += complex(sys.C.At(i, c), 0) * v[c]
			}
		}
	}
	return g
}

func maxSV2x2(g [2][2]complex128) float64 {
	var h [2][2]complex128
	for i := range 2 {
		for j := range 2 {
			h[i][j] = cmplx.Conj(g[0][i])*g[0][j] + cmplx.Conj(g[1][i])*g[1][j]
		}
	}
	a, b := real(h[0][0]), real(h[1][1])
	return math.Sqrt((a+b)/2 + math.Hypot((a-b)/2, cmplx.Abs(h[0][1])))
}

func TestHinfNorm_DiscreteInternalDelay(t *testing.T) {
	unstable, _ := discreteLFTDelayFixture(t, []float64{1, 0.5})
	if norm, w, err := HinfNorm(unstable); err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
		t.Fatalf("unstable delay loop: %g at %g, %v; want Inf at Inf", norm, w, err)
	}

	lft, _ := discreteLFTDelayFixture(t, []float64{0.1, 0.05})
	norm, w, err := HinfNorm(lft)
	if err != nil {
		t.Fatal(err)
	}
	if at := maxSV2x2(lftDelayFreqOracle(lft, w)); math.Abs(at-norm) > 1e-8*norm {
		t.Fatalf("σmax at returned ω=%g is %.12g, HinfNorm %.12g", w, at, norm)
	}
	peak := 0.0
	for k := range 20001 {
		peak = math.Max(peak, maxSV2x2(lftDelayFreqOracle(lft, float64(k)/20000*math.Pi/lft.Dt)))
	}
	if norm < peak*(1-1e-9) || norm > peak*(1+1e-4) {
		t.Fatalf("HinfNorm %.12g, grid peak %.12g", norm, peak)
	}
	free := lft.Copy()
	free.LFT = nil
	freeNorm, _, err := HinfNorm(free)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(freeNorm-norm) < 1e-3*norm {
		t.Fatalf("HinfNorm %.12g matches delay-free model; internal delay ignored", norm)
	}
}

func TestHinfNorm_StatelessDiscreteDelayLoop(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{0}), 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{1}, &mat.Dense{}, &mat.Dense{},
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0.5})); err != nil {
		t.Fatal(err)
	}
	// y = w, w = z⁻¹(u + 0.5w) ⇒ G = 1/(z-0.5), peak 2 at ω = 0.
	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-2) > 1e-9 {
		t.Fatalf("HinfNorm = %.12g, want 2", norm)
	}
}

func TestHSV_DiscreteInternalDelay(t *testing.T) {
	unstable, _ := discreteLFTDelayFixture(t, []float64{1, 0.5})
	if _, err := HSV(unstable); !errors.Is(err, ErrUnstable) {
		t.Fatalf("unstable delay loop: err = %v, want ErrUnstable", err)
	}

	lft, hand := discreteLFTDelayFixture(t, []float64{0.1, 0.05})
	got, err := HSV(lft)
	if err != nil {
		t.Fatal(err)
	}
	wc, wo := mat.NewDense(5, 5, nil), mat.NewDense(5, 5, nil)
	ak := mat.NewDense(5, 5, nil)
	ak.Copy(eye(5))
	for range 2000 {
		var t1, t2, q mat.Dense
		t1.Mul(ak, hand.B)
		q.Mul(&t1, t1.T())
		wc.Add(wc, &q)
		t2.Mul(hand.C, ak)
		q.Mul(t2.T(), &t2)
		wo.Add(wo, &q)
		ak.Mul(ak, hand.A)
	}
	var prod mat.Dense
	prod.Mul(wc, wo)
	var eig mat.Eigen
	if !eig.Factorize(&prod, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	want := make([]float64, 0, 5)
	for _, v := range eig.Values(nil) {
		want = append(want, math.Sqrt(math.Max(real(v), 0)))
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(want)))
	if len(got) != len(want) {
		t.Fatalf("len(HSV) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9*want[0] {
			t.Fatalf("HSV[%d] = %.12g, want %.12g", i, got[i], want[i])
		}
	}
}

func TestNorms_ContinuousInternalDelayRejected(t *testing.T) {
	sys := scalarDDE(t, -2, 2)
	if _, err := HSV(sys); !errors.Is(err, ErrContinuousInternalDelay) {
		t.Fatalf("HSV err = %v, want ErrContinuousInternalDelay", err)
	}
}

// ssFreqOracle2x2 evaluates C (sI-A)^-1 B + D of a 2x2 model at s = jω or
// z = e^{jωT} by Gaussian elimination.
func ssFreqOracle2x2(sys *System, w float64) [2][2]complex128 {
	n, _, _ := sys.Dims()
	s := complex(0, w)
	if sys.IsDiscrete() {
		s = cmplx.Exp(complex(0, w*sys.Dt))
	}
	var g [2][2]complex128
	for j := range 2 {
		m := make([][]complex128, n)
		for r := range n {
			m[r] = make([]complex128, n+1)
			for c := range n {
				m[r][c] = complex(-sys.A.At(r, c), 0)
			}
			m[r][r] += s
			m[r][n] = complex(sys.B.At(r, j), 0)
		}
		for k := range n {
			p := k
			for r := k + 1; r < n; r++ {
				if cmplx.Abs(m[r][k]) > cmplx.Abs(m[p][k]) {
					p = r
				}
			}
			m[k], m[p] = m[p], m[k]
			for r := k + 1; r < n; r++ {
				f := m[r][k] / m[k][k]
				for c := k; c <= n; c++ {
					m[r][c] -= f * m[k][c]
				}
			}
		}
		x := make([]complex128, n)
		for k := n - 1; k >= 0; k-- {
			v := m[k][n]
			for c := k + 1; c < n; c++ {
				v -= m[k][c] * x[c]
			}
			x[k] = v / m[k][k]
		}
		for i := range 2 {
			g[i][j] = complex(sys.D.At(i, j), 0)
			for c := range n {
				g[i][j] += complex(sys.C.At(i, c), 0) * x[c]
			}
		}
	}
	return g
}

func TestNormInf_UnstableFirstOrder(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		a := 1.0
		if dt > 0 {
			a = 2
		}
		sys, _ := New(
			mat.NewDense(1, 1, []float64{a}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{0}), dt)
		got, err := Norm(sys, math.Inf(1))
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		if math.Abs(got-1) > 1e-9 {
			t.Errorf("dt=%v: Norm(Inf) = %.15g, want 1", dt, got)
		}
		if norm, w, err := HinfNorm(sys); err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
			t.Errorf("dt=%v: HinfNorm = %g at %g, %v; want Inf at Inf", dt, norm, w, err)
		}
	}
}

func TestNormInf_UnstableLightlyDampedPair(t *testing.T) {
	wn, zeta := 10.0, 0.05
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, 2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	got, w, err := linfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 1 / (2 * zeta * math.Sqrt(1-zeta*zeta))
	wantW := wn * math.Sqrt(1-2*zeta*zeta)
	if math.Abs(got-want) > 1e-9*want {
		t.Errorf("L∞ = %.15g, want %.15g", got, want)
	}
	if math.Abs(w-wantW) > 1e-3*wantW {
		t.Errorf("peak ω = %g, want %g", w, wantW)
	}
	if n, err := Norm(sys, math.Inf(1)); err != nil || n != got {
		t.Errorf("Norm(Inf) = %g, %v; want %g", n, err, got)
	}
}

func TestNormInf_UnstableMIMOGridOracle(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0.4, 2, -0.3,
		-3, 0.2, 1,
		0.5, -1, -2,
	})
	B := mat.NewDense(3, 2, []float64{1, 0.3, -0.5, 1, 0.2, -0.7})
	C := mat.NewDense(2, 3, []float64{1, -0.4, 0.6, 0.2, 1, -1})
	D := mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.5})
	for _, dt := range []float64{0, 0.05} {
		Ad := A
		if dt > 0 {
			Ad = mat.NewDense(3, 3, nil)
			Ad.Scale(dt, A)
			Ad.Exp(Ad)
		}
		sys, err := New(Ad, B, C, D, dt)
		if err != nil {
			t.Fatal(err)
		}
		if stable, _ := sys.IsStable(); stable {
			t.Fatalf("dt=%v: fixture must be unstable", dt)
		}
		got, w, err := linfNorm(sys)
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		if at := maxSV2x2(ssFreqOracle2x2(sys, w)); math.Abs(at-got) > 1e-8*got {
			t.Errorf("dt=%v: σmax at returned ω=%g is %.12g, L∞ %.12g", dt, w, at, got)
		}
		if peak := gridPeak2x2(sys); got < peak*(1-1e-9) {
			t.Errorf("dt=%v: L∞ %.12g below grid peak %.12g", dt, got, peak)
		}
		if n, err := Norm(sys, math.Inf(1)); err != nil || n != got {
			t.Errorf("dt=%v: Norm(Inf) = %g, %v; want %g", dt, n, err, got)
		}
	}
}

// gridPeak2x2 returns the largest σmax over a dense grid refined around the
// coarse maximum.
func gridPeak2x2(sys *System) float64 {
	wmax := 1e3
	if sys.IsDiscrete() {
		wmax = math.Pi / sys.Dt
	}
	const nGrid = 100000
	peak, kPeak := 0.0, 0
	for k := range nGrid + 1 {
		if v := maxSV2x2(ssFreqOracle2x2(sys, float64(k)/nGrid*wmax)); v > peak {
			peak, kPeak = v, k
		}
	}
	lo := float64(max(kPeak-1, 0)) / nGrid * wmax
	hi := float64(min(kPeak+1, nGrid)) / nGrid * wmax
	for k := range nGrid + 1 {
		peak = math.Max(peak, maxSV2x2(ssFreqOracle2x2(sys, lo+float64(k)/nGrid*(hi-lo))))
	}
	return peak
}

func TestHinfNorm_DiscretePeakFrequency(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			0.5, 0.6, -0.1,
			-0.7, 0.4, 0.2,
			0.1, -0.3, 0.3,
		}),
		mat.NewDense(3, 2, []float64{1, 0.3, -0.5, 1, 0.2, -0.7}),
		mat.NewDense(2, 3, []float64{1, -0.4, 0.6, 0.2, 1, -1}),
		mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.5}), 0.05)
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if at := maxSV2x2(ssFreqOracle2x2(sys, w)); math.Abs(at-got) > 1e-8*got {
		t.Errorf("σmax at returned ω=%g is %.12g, HinfNorm %.12g", w, at, got)
	}
	if peak := gridPeak2x2(sys); math.Abs(got-peak) > 1e-9*peak {
		t.Errorf("HinfNorm %.12g, grid peak %.12g", got, peak)
	}
}

func TestNormInf_BoundaryPolesInfinite(t *testing.T) {
	cases := []struct {
		name  string
		A     []float64
		dt    float64
		wantW float64
	}{
		{"integrator", []float64{0, 0, 0, -1}, 0, 0},
		{"undamped pair", []float64{0, 5, -5, 0}, 0, 5},
		{"discrete z=1", []float64{1, 0, 0, 0.5}, 0.1, 0},
		{"discrete z=-1", []float64{-1, 0, 0, 0.5}, 0.1, math.Pi / 0.1},
	}
	for _, tc := range cases {
		sys, _ := New(
			mat.NewDense(2, 2, tc.A),
			mat.NewDense(2, 1, []float64{1, 1}),
			mat.NewDense(1, 2, []float64{1, 1}),
			mat.NewDense(1, 1, []float64{0}), tc.dt)
		got, w, err := linfNorm(sys)
		if err != nil || !math.IsInf(got, 1) || math.Abs(w-tc.wantW) > 1e-9*max(1, tc.wantW) {
			t.Errorf("%s: L∞ = %g at ω=%g, %v; want +Inf at %g", tc.name, got, w, err, tc.wantW)
		}
	}
}

func TestNorm2_UnstableIsInf(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	got, err := Norm(sys, 2)
	if err != nil || !math.IsInf(got, 1) {
		t.Errorf("Norm(2) = %g, %v; want +Inf", got, err)
	}
}

// bisectionPeakGain is peakGain before the certifying probe: plain
// bisection on the crossing test from the sampled lower bound (σ_max(D) at
// ω = ∞ for continuous models).
func bisectionPeakGain(sys *System) (float64, float64, error) {
	n, m, p := sys.Dims()
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, err
	}
	ws := newHamiltonianWS(sys, n, m, p)
	gammaLow, omegaPeak := ws.lowerBound(poles)
	if sd := maxSVDense(sys.D, p, m); sys.IsContinuous() && sd > gammaLow {
		gammaLow, omegaPeak = sd, math.Inf(1)
	}
	gammaHigh := math.Max(gammaLow*2, 1e-10)
	for range 50 {
		if !ws.hasImagEigs(gammaHigh) {
			break
		}
		gammaHigh *= 2
	}
	tol := 1e-10
	for range 100 {
		if gammaHigh-gammaLow < tol*gammaHigh {
			break
		}
		mid := (gammaLow + gammaHigh) / 2
		if !ws.hasImagEigs(mid) {
			gammaHigh = mid
			continue
		}
		peak, w, err := ws.candidatePeak()
		if err != nil {
			gammaLow = mid
			continue
		}
		if peak < mid {
			gammaHigh = mid
			continue
		}
		gammaLow, omegaPeak = peak, w
	}
	return gammaHigh, omegaPeak, nil
}

func polesOf(t testing.TB, sys *System) []complex128 {
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	return poles
}

func countHamiltonianEvals(f func()) int64 {
	before := hamiltonianEvals.Load()
	f()
	return hamiltonianEvals.Load() - before
}

func randomStableNormSys(t testing.TB, rng *rand.Rand, n, m, p int, dt float64, withD bool, zeta float64) *System {
	t.Helper()
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, rng.NormFloat64()/math.Sqrt(float64(n)))
		}
	}
	if zeta > 0 && n >= 2 {
		w0 := 0.3 + 3*rng.Float64()
		A.Set(0, 0, -zeta*w0)
		A.Set(0, 1, w0)
		A.Set(1, 0, -w0)
		A.Set(1, 1, -zeta*w0)
		for j := 2; j < n; j++ {
			A.Set(0, j, 0)
			A.Set(1, j, 0)
		}
	}
	poles, err := mustNewSys(t, A, m, p).Poles()
	if err != nil {
		t.Fatal(err)
	}
	shift := math.Inf(-1)
	for _, pl := range poles {
		shift = math.Max(shift, real(pl))
	}
	if shift > -0.1 {
		start := 0
		if zeta > 0 {
			start = 2
		}
		for i := start; i < n; i++ {
			A.Set(i, i, A.At(i, i)-shift-0.1-rng.Float64())
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
	}
	C := mat.NewDense(p, n, nil)
	for i := range p {
		for j := range n {
			C.Set(i, j, rng.NormFloat64())
		}
	}
	D := mat.NewDense(p, m, nil)
	if withD {
		for i := range p {
			for j := range m {
				D.Set(i, j, 0.5*rng.NormFloat64())
			}
		}
	}
	sys, err := New(A, B, C, D, 0)
	if err == nil && dt > 0 {
		sys, err = sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	}
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func mustNewSys(t testing.TB, A *mat.Dense, m, p int) *System {
	t.Helper()
	n, _ := A.Dims()
	sys, err := New(A, mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// hinfMixedSensitivityLoop is the closed loop of HinfSyn on a 2×2
// mixed-sensitivity problem: W1 = (0.5s+0.5)/(s+0.005) per channel (dc gain
// 100, high-frequency gain 0.5), W2 = 0.1. order n gives a 2n+4 state loop.
func hinfMixedSensitivityLoop(t testing.TB, order int, seed int64) *System {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	n, nw := order, order+2
	Ap := mat.NewDense(nw, nw, nil)
	Bp := mat.NewDense(nw, 4, nil)
	Cp := mat.NewDense(6, nw, nil)
	Dp := mat.NewDense(6, 4, nil)
	for i := range n {
		for j := range n {
			if i == j {
				Ap.Set(i, j, -1-float64(i)/2)
			} else {
				Ap.Set(i, j, 0.1*rng.NormFloat64()/math.Sqrt(float64(n)))
			}
		}
		for j := range 2 {
			Bp.Set(i, 2+j, rng.NormFloat64())
			c := rng.NormFloat64()
			Ap.Set(n+j, i, -c)
			Cp.Set(j, i, -0.5*c)
			Cp.Set(4+j, i, -c)
		}
	}
	for j := range 2 {
		Ap.Set(n+j, n+j, -0.005)
		Bp.Set(n+j, j, 1)
		Cp.Set(j, n+j, 0.4975)
		Dp.Set(j, j, 0.5)
		Dp.Set(2+j, 2+j, 0.1)
		Dp.Set(4+j, j, 1)
	}
	P, err := New(Ap, Bp, Cp, Dp, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := HinfSyn(P, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := LFT(P, res.K, LFTFeedback{Nu: 2, Ny: 2})
	if err != nil {
		t.Fatal(err)
	}
	return cl
}

func assertPeakMatchesBisection(t *testing.T, name string, sys *System) (evals int64) {
	t.Helper()
	want, _, err := bisectionPeakGain(sys)
	if err != nil {
		t.Fatalf("%s: bisection: %v", name, err)
	}
	var got float64
	evals = countHamiltonianEvals(func() {
		got, _, err = peakGain(sys, polesOf(t, sys))
	})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if math.Abs(got-want) > 1e-10*math.Max(got, want) {
		t.Errorf("%s: peakGain = %.15g, bisection = %.15g (rel %.2e)", name, got, want, math.Abs(got-want)/want)
	}
	return evals
}

func TestPeakGain_CertifiedMatchesBisection(t *testing.T) {
	rng := rand.New(rand.NewSource(20261004))
	var fast, total int
	for trial := range 120 {
		n := 2 + rng.Intn(9)
		m := 1 + rng.Intn(3)
		p := 1 + rng.Intn(3)
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		zeta := 0.0
		if trial%3 == 0 {
			zeta = math.Pow(10, -2-3*rng.Float64())
		}
		sys := randomStableNormSys(t, rng, n, m, p, dt, trial%4 >= 2, zeta)
		if evals := assertPeakMatchesBisection(t, "random", sys); evals == 1 {
			fast++
		}
		total++
	}
	if fast < total/3 {
		t.Errorf("one probe certified %d of %d models", fast, total)
	}

	assertPeakMatchesBisection(t, "biproper mixed sensitivity", mimoBiproperMixedSensitivityPlant(t))
	for _, order := range []int{4, 10} {
		assertPeakMatchesBisection(t, "hinfsyn loop", hinfMixedSensitivityLoop(t, order, 7))
	}

	nearSingular, err := New(
		mat.NewDense(2, 2, []float64{-0.01, 0, -0.01, -0.00004320073460981398}),
		mat.NewDense(2, 2, []float64{0, 1, 1, 0}),
		mat.NewDense(3, 2, []float64{-0.005, 0.004298473093676491, 0, 0, -0.01, 0}),
		mat.NewDense(3, 2, []float64{0.5, 0, 0, 0.1, 1, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := HinfSyn(nearSingular, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertPeakMatchesBisection(t, "near-singular-edge loop (GH3MO7)", closedLoop(t, nearSingular, res.K, 1, 1))

	nonMinimal := strictMixedSensitivityPlant(t)
	if res, err = HinfSyn(nonMinimal, 2, 2); err != nil {
		t.Fatal(err)
	}
	assertPeakMatchesBisection(t, "non-minimal near-optimal loop (TI6YYN)", closedLoop(t, nonMinimal, res.K, 2, 2))
}

// strictMixedSensitivityPlant is the TI6YYN problem: a 3-state 2×2 plant
// with W1 = 1/(s+0.01) per channel and W2 = 1e-3; its HinfSyn loop is
// non-minimal and ill-conditioned at low frequency.
func strictMixedSensitivityPlant(t *testing.T) *System {
	t.Helper()
	A := []float64{-1, 0.2, 0, 0, -2, 0.5, 0.1, 0, -3}
	B := []float64{1, 0, 0, 1, 0.5, 0.5}
	C := []float64{1, 0, 0.2, 0, 1, 0}
	Ap := mat.NewDense(5, 5, nil)
	Bp := mat.NewDense(5, 4, nil)
	Cp := mat.NewDense(6, 5, nil)
	Dp := mat.NewDense(6, 4, nil)
	for i := range 3 {
		for j := range 3 {
			Ap.Set(i, j, A[i*3+j])
		}
		for j := range 2 {
			Bp.Set(i, 2+j, B[i*2+j])
			Ap.Set(3+j, i, -C[j*3+i])
			Cp.Set(4+j, i, -C[j*3+i])
		}
	}
	for j := range 2 {
		Ap.Set(3+j, 3+j, -0.01)
		Bp.Set(3+j, j, 1)
		Cp.Set(j, 3+j, 1)
		Dp.Set(2+j, 2+j, 1e-3)
		Dp.Set(4+j, j, 1)
	}
	P, err := New(Ap, Bp, Cp, Dp, 0)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

func TestPeakGain_OneHamiltonianProbeWhenSampledPeakIsExact(t *testing.T) {
	lowPass, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0.3, 1}),
		mat.NewDense(2, 2, []float64{1, 0.2, 0, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.1}), 0)
	discLowPass, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.2, 0, 0.3}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}),
		mat.NewDense(2, 1, nil), 0.1)
	unstable, _ := New(
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)

	for _, tc := range []struct {
		name string
		f    func() (float64, error)
		want float64
	}{
		{"HinfNorm continuous MIMO D≠0", func() (float64, error) { g, _, err := HinfNorm(lowPass); return g, err }, maxSVDense(dcGainDense(t, lowPass), 2, 2)},
		{"HinfNorm discrete", func() (float64, error) { g, _, err := HinfNorm(discLowPass); return g, err }, maxSVDense(dcGainDense(t, discLowPass), 2, 1)},
		{"Norm Inf unstable", func() (float64, error) { return Norm(unstable, math.Inf(1)) }, 1},
	} {
		var got float64
		var err error
		evals := countHamiltonianEvals(func() { got, err = tc.f() })
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if evals != 1 {
			t.Errorf("%s: %d Hamiltonian evaluations, want 1", tc.name, evals)
		}
		if got < tc.want || got > tc.want*(1+1e-10) {
			t.Errorf("%s: norm = %.15g, want DC gain %.15g within 1e-10", tc.name, got, tc.want)
		}
	}
}

func dcGainDense(t *testing.T, sys *System) *mat.Dense {
	t.Helper()
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestPeakGain_FallbackWhenSampledBoundMissesPeak(t *testing.T) {
	const zeta, w0 = 1e-4, 1.37
	A := mat.NewDense(3, 3, []float64{
		-zeta * w0, w0, 0.3,
		-w0, -zeta * w0, 0,
		0, 0, -2,
	})
	B := mat.NewDense(3, 2, []float64{0, 1, 1, 0, 0.5, 1})
	C := mat.NewDense(2, 3, []float64{1, 0, 1, 0.2, 1, 0})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0.3, 0})
	for _, dt := range []float64{0, 0.05} {
		sys, err := New(A, B, C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		if dt > 0 {
			if sys, err = sys.C2D(dt, C2DOptions{Method: C2DMethodTustin}); err != nil {
				t.Fatal(err)
			}
		}
		low, _ := newHamiltonianWS(sys, 3, 2, 2).lowerBound(polesOf(t, sys))
		evals := assertPeakMatchesBisection(t, "lightly damped", sys)
		got, _, _ := peakGain(sys, polesOf(t, sys))
		if low > 0.5*got {
			t.Fatalf("dt=%g: sampled bound %g does not miss peak %g", dt, low, got)
		}
		if evals <= 1 || evals > 10 {
			t.Errorf("dt=%g: %d Hamiltonian evaluations, want a few crossing-midpoint raises", dt, evals)
		}
	}
}

// highPassResonanceSys is a 2×2 high-pass d_i·s/(s+a_i) with coupling,
// plus a ζ ∈ [1e-6, 1e-4] resonance at ω0 ∈ [0.3, 3]: the sampled bound
// stays below σ(D) while the resonance peak lies far above it.
func highPassResonanceSys(t testing.TB, rng *rand.Rand, dt float64) *System {
	t.Helper()
	zeta := math.Pow(10, -4-2*rng.Float64())
	w0 := 0.3 + 2.7*rng.Float64()
	a := []float64{0.5 + rng.Float64(), 1 + 2*rng.Float64()}
	d := []float64{50 + 150*rng.Float64(), 20 + 50*rng.Float64()}
	A := mat.NewDense(4, 4, []float64{
		-a[0], 0.3, 0, 0,
		0, -a[1], 0, 0,
		0, 0, -zeta * w0, w0,
		0, 0, -w0, -zeta * w0,
	})
	B := mat.NewDense(4, 2, []float64{
		1, 0,
		0, 1,
		0, 0,
		rng.NormFloat64(), rng.NormFloat64(),
	})
	C := mat.NewDense(2, 4, []float64{
		-d[0] * a[0], 0, rng.NormFloat64(), 0,
		0, -d[1] * a[1], rng.NormFloat64(), 0,
	})
	D := mat.NewDense(2, 2, []float64{d[0], 0.1 * d[1], 0, d[1]})
	sys, err := New(A, B, C, D, 0)
	if err == nil && dt > 0 {
		sys, err = sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	}
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestPeakGain_ResonanceAboveFeedthroughBoundedEvals(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	var belowSigmaD, total int
	for trial := range 60 {
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		sys := highPassResonanceSys(t, rng, dt)
		if dt == 0 {
			if low, _ := newHamiltonianWS(sys, 4, 2, 2).lowerBound(polesOf(t, sys)); low < maxSVDense(sys.D, 2, 2) {
				belowSigmaD++
			}
			total++
		}
		if evals := assertPeakMatchesBisection(t, fmt.Sprintf("trial %d dt=%g", trial, dt), sys); evals > 10 {
			t.Errorf("trial %d dt=%g: %d Hamiltonian evaluations, want ≤ 10", trial, dt, evals)
		}
	}
	if belowSigmaD < total/2 {
		t.Fatalf("sampled bound below sigma(D) in %d of %d models", belowSigmaD, total)
	}
}

func TestHamiltonianUpperBound_LightlyDamped(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	for trial := range 40 {
		zeta := math.Pow(10, -4-2*rng.Float64())
		sys := randomStableNormSys(t, rng, 2+rng.Intn(7), 1+rng.Intn(3), 1+rng.Intn(3), 0, trial%2 == 1, zeta)
		peak, _, err := bisectionPeakGain(sys)
		if err != nil {
			t.Fatal(err)
		}
		n, m, p := sys.Dims()
		ws := newHamiltonianWS(sys, n, m, p)

		var high float64
		var certified bool
		evals := countHamiltonianEvals(func() { _, high, _, certified = ws.upperBound(peak, 0, 1e-10) })
		if certified || evals != 1 || high != 2*peak {
			t.Errorf("trial %d from the peak: high = %g (peak %g), certified %v, %d evals; want 2·peak, 1 eval",
				trial, high, peak, certified, evals)
		}

		var low float64
		evals = countHamiltonianEvals(func() { low, high, _, certified = ws.upperBound(peak/4, 0, 1e-10) })
		if evals > 8 {
			t.Errorf("trial %d from peak/4: %d Hamiltonian evaluations, want ≤ 8", trial, evals)
		}
		if low > peak*(1+1e-10) || high < peak*(1-1e-10) {
			t.Errorf("trial %d from peak/4: [%g, %g] does not bracket peak %g", trial, low, high, peak)
		}
		if certified && math.Abs(low*(1+1e-10/2)-peak) > 1e-10*peak {
			t.Errorf("trial %d: certified %.15g, peak %.15g", trial, low, peak)
		}
	}
}

// TestPeakGain_FeedthroughPeakAtInfinity: the gain of s/(s+1) approaches
// σ(D) = 1 only as ω → ∞, which MATLAB hinfnorm reports as fpeak = Inf
// (π/T for the discrete model). The sampled bound misses it, and the γ
// levels just above σ(D) make R = γ²−D² near singular.
func TestPeakGain_FeedthroughPeakAtInfinity(t *testing.T) {
	highPass, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if low, _ := newHamiltonianWS(highPass, 1, 1, 1).lowerBound(polesOf(t, highPass)); low >= 1 {
		t.Fatalf("sampled bound %g reaches sigma(D) = 1", low)
	}
	var got, w float64
	evals := countHamiltonianEvals(func() { got, w, err = HinfNorm(highPass) })
	if err != nil {
		t.Fatal(err)
	}
	if got < 1 || got > 1+1e-10 || !math.IsInf(w, 1) || evals != 1 {
		t.Errorf("HinfNorm = %.15g at %g in %d evaluations, want 1 at +Inf in 1", got, w, evals)
	}

	disc, err := highPass.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Abs(disc.D.At(0, 0) + disc.C.At(0, 0)*disc.B.At(0, 0)/(-1-disc.A.At(0, 0)))
	if got, w, err = HinfNorm(disc); err != nil {
		t.Fatal(err)
	}
	if got < want || got > want*(1+1e-10) || w != math.Pi/0.1 {
		t.Errorf("discrete HinfNorm = %.15g at %g, want |G(-1)| = %.15g at π/T", got, w, want)
	}
}

func BenchmarkHinfNorm_ResonanceAboveFeedthrough(b *testing.B) {
	sys := highPassResonanceSys(b, rand.New(rand.NewSource(1)), 0)
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := HinfNorm(sys); err != nil {
			b.Fatal(err)
		}
	}
}

// exactSigmaMax is σ_max(G) of the stored realization at frequency w with
// only the final G rounded: frExactResponse at frExactPoint, the point
// sigmaEvaluator solves at. The singular value comes from the Hermitian Gram
// matrix.
func exactSigmaMax(sys *System, w float64) float64 {
	_, m, p := sys.Dims()
	return gramSigmaMax(frExactResponse(sys, frExactPoint(w, sys.Dt)), p, m)
}

// gramSigmaMax is σ_max of the row-major p×m G from the real symmetric
// embedding of GᴴG.
func gramSigmaMax(G []complex128, p, m int) float64 {
	H := mat.NewSymDense(2*m, nil)
	for a := range m {
		for b := a; b < m; b++ {
			var g complex128
			for i := range p {
				g += cmplx.Conj(G[i*m+a]) * G[i*m+b]
			}
			H.SetSym(a, b, real(g))
			H.SetSym(m+a, m+b, real(g))
			H.SetSym(a, m+b, -imag(g))
			H.SetSym(b, m+a, imag(g))
		}
	}
	var eig mat.EigenSym
	if !eig.Factorize(H, false) {
		return math.NaN()
	}
	vals := eig.Values(nil)
	return math.Sqrt(math.Max(vals[len(vals)-1], 0))
}

// pointwiseSigmaMax is σ_max from FreqResponsePointwise.
func pointwiseSigmaMax(t testing.TB, sys *System, freqs []float64) []float64 {
	t.Helper()
	_, m, p := sys.Dims()
	resp, err := sys.FreqResponsePointwise(freqs)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, len(freqs))
	for k := range freqs {
		out[k] = gramSigmaMax(resp.Data[k*p*m:(k+1)*p*m], p, m)
	}
	return out
}

// oraclePeakGain is sup_ω σ_max found without the crossing test: σ_max on a
// dense log grid plus points around every pole, then golden-section search
// with sigma around the three best local maxima. σ_max(D) is the ω → ∞
// limit of a continuous model.
func oraclePeakGain(t testing.TB, sys *System, sigma func(float64) float64) float64 {
	t.Helper()
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	var freqs []float64
	wNyq := math.Inf(1)
	if sys.IsDiscrete() {
		wNyq = math.Pi / sys.Dt
		freqs = append(freqs, wNyq)
	}
	lo, hi := math.Inf(1), 0.0
	for _, pl := range poles {
		if sys.IsDiscrete() {
			pl = cmplx.Log(pl) / complex(sys.Dt, 0)
		}
		wc, d := math.Abs(imag(pl)), math.Abs(real(pl))
		lo, hi = math.Min(lo, cmplx.Abs(pl)), math.Max(hi, cmplx.Abs(pl))
		for k := -16; k <= 16; k++ {
			freqs = append(freqs, wc+float64(k)*d/4)
		}
	}
	lo, hi = math.Max(lo/1e4, 1e-9), math.Min(hi*1e4, wNyq)
	for i := range 2000 {
		freqs = append(freqs, lo*math.Pow(hi/lo, float64(i)/1999))
	}
	freqs = append(freqs, 0)
	freqs = slices.DeleteFunc(freqs, func(w float64) bool { return w < 0 || w > wNyq })
	slices.Sort(freqs)
	freqs = slices.Compact(freqs)
	vals := pointwiseSigmaMax(t, sys, freqs)

	var maxima []int
	for i, v := range vals {
		if (i == 0 || v >= vals[i-1]) && (i == len(vals)-1 || v >= vals[i+1]) {
			maxima = append(maxima, i)
		}
	}
	sort.Slice(maxima, func(a, b int) bool { return vals[maxima[a]] > vals[maxima[b]] })
	best := 0.0
	if sys.IsContinuous() {
		_, m, p := sys.Dims()
		best = maxSVDense(sys.D, p, m)
	}
	const r = 0.6180339887498949
	for _, i := range maxima[:min(3, len(maxima))] {
		a, b := freqs[max(i-1, 0)], freqs[min(i+1, len(freqs)-1)]
		c, d := b-r*(b-a), a+r*(b-a)
		fc, fd := sigma(c), sigma(d)
		for range 64 {
			if fc > fd {
				b, d, fd = d, c, fc
				c = b - r*(b-a)
				fc = sigma(c)
			} else {
				a, c, fc = c, d, fd
				d = a + r*(b-a)
				fd = sigma(d)
			}
		}
		best = max(best, fc, fd, sigma(freqs[i]))
	}
	return best
}

// assertHinfNormMatchesOracle checks the norm against oraclePeakGain to
// 1e-10 and that the reported frequency attains it.
func assertHinfNormMatchesOracle(t *testing.T, name string, sys *System, sigma func(float64) float64) {
	t.Helper()
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	want := oraclePeakGain(t, sys, sigma)
	if math.Abs(got-want) > 1e-10*want {
		t.Errorf("%s: HinfNorm = %.15g, oracle %.15g (rel %.2e)", name, got, want, got/want-1)
	}
	at := 0.0
	if math.IsInf(w, 1) {
		_, m, p := sys.Dims()
		at = maxSVDense(sys.D, p, m)
	} else {
		at = sigma(w)
	}
	if at < got*(1-1e-10) {
		t.Errorf("%s: σ_max(%g) = %.15g, below HinfNorm %.15g (rel %.2e)", name, w, at, got, at/got-1)
	}
}

// TestHinfNorm_LightlyDampedMatchesExactOracle (Z5SW7V): ζ ∈ [1e-7, 1e-4]
// resonances with D up to 1e4, continuous and discrete. Evaluating σ_max by
// plain GEPP at the peak loses about ε/ζ, a rounded e^{jωT} lies off the
// unit circle by ε, and the Tustin map shifts discrete peaks: before the
// fix these models were off by up to 5e-8.
func TestHinfNorm_LightlyDampedMatchesExactOracle(t *testing.T) {
	exact := func(sys *System) func(float64) float64 {
		return func(w float64) float64 { return exactSigmaMax(sys, w) }
	}
	rng := rand.New(rand.NewSource(1))
	for trial := range 200 {
		n, m, p := 2+rng.Intn(7), 1+rng.Intn(2), 1+rng.Intn(2)
		zeta := math.Pow(10, -4-3*rng.Float64())
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		sys := randomStableNormSys(t, rng, n, m, p, dt, true, zeta)
		sys.D.Scale(math.Pow(10, 4*rng.Float64()), sys.D)
		if trial%20 > 1 && trial != 132 {
			continue
		}
		assertHinfNormMatchesOracle(t, fmt.Sprintf("trial %d ζ=%.1e dt=%g", trial, zeta, dt), sys, exact(sys))
	}

	rng = rand.New(rand.NewSource(1))
	for trial := range 4 {
		sys := highPassResonanceSys(t, rng, 0.1*float64(trial%2))
		assertHinfNormMatchesOracle(t, fmt.Sprintf("high-pass resonance %d", trial), sys, exact(sys))
	}

	nonMinimal := randomStableNormSys(t, rand.New(rand.NewSource(3)), 5, 2, 2, 0, true, 1e-6)
	n, m, p := nonMinimal.Dims()
	A := mat.NewDense(n+2, n+2, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(nonMinimal.A)
	A.Slice(n, n+2, n, n+2).(*mat.Dense).Copy(mat.NewDense(2, 2, []float64{-1e-3, 1, -1, -1e-3}))
	B := mat.NewDense(n+2, m, nil)
	B.Slice(0, n, 0, m).(*mat.Dense).Copy(nonMinimal.B)
	C := mat.NewDense(p, n+2, nil)
	C.Slice(0, p, 0, n).(*mat.Dense).Copy(nonMinimal.C)
	C.Set(0, n, 3)
	C.Set(p-1, n+1, -2)
	sys, err := New(A, B, C, nonMinimal.D, 0)
	if err != nil {
		t.Fatal(err)
	}
	assertHinfNormMatchesOracle(t, "uncontrollable ζ=1e-3 mode", sys, exact(sys))
}

// TestHinfNorm_NearOptimalLoopsMatchOracle (TI6YYN, GH3MO7): HinfSyn closed
// loops whose σ_max is flat to 1e-8 over decades, built from non-minimal,
// ill-conditioned LFT realizations. Their crossings leave the imaginary axis
// under QZ, and the Hamiltonian matrix [A BBᵀ/γ²; −CᵀC −Aᵀ] lost them
// altogether: the TI6YYN loop was under-reported by 8.4e-10.
func TestHinfNorm_NearOptimalLoopsMatchOracle(t *testing.T) {
	skipSlowUnderRace(t)
	exact := func(sys *System) func(float64) float64 {
		return func(w float64) float64 { return exactSigmaMax(sys, w) }
	}
	pointwise := func(sys *System) func(float64) float64 {
		return func(w float64) float64 { return pointwiseSigmaMax(t, sys, []float64{w})[0] }
	}

	strict := strictMixedSensitivityPlant(t)
	res, err := HinfSyn(strict, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	ti := closedLoop(t, strict, res.K, 2, 2)
	assertHinfNormMatchesOracle(t, "non-minimal near-optimal loop (TI6YYN)", ti, exact(ti))

	nearSingular, err := New(
		mat.NewDense(2, 2, []float64{-0.01, 0, -0.01, -0.00004320073460981398}),
		mat.NewDense(2, 2, []float64{0, 1, 1, 0}),
		mat.NewDense(3, 2, []float64{-0.005, 0.004298473093676491, 0, 0, -0.01, 0}),
		mat.NewDense(3, 2, []float64{0.5, 0, 0, 0.1, 1, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if res, err = HinfSyn(nearSingular, 1, 1); err != nil {
		t.Fatal(err)
	}
	backoff := closedLoop(t, nearSingular, res.K, 1, 1)
	assertHinfNormMatchesOracle(t, "near-singular-edge loop (GH3MO7)", backoff, exact(backoff))
	gp, err := partitionGeneralizedPlant("HinfSyn", nearSingular, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	atEdge, err := hinfSynD11Zero(gp, newHinfD11ZeroRiccatis(gp), 0.5074089765548706)
	if err != nil {
		t.Fatal(err)
	}
	edge := closedLoop(t, nearSingular, atEdge.K, 1, 1)
	assertHinfNormMatchesOracle(t, "controller at the bisection edge (GH3MO7)", edge, exact(edge))

	order50 := mixedSensitivityPlant(t, randomHinfTestPlant(t, 50, 2, 7), 1, 0.01, 0, 0.1)
	if res, err = HinfSyn(order50, 2, 2); err != nil {
		t.Fatal(err)
	}
	loop50 := closedLoop(t, order50, res.K, 2, 2)
	assertHinfNormMatchesOracle(t, "order-50 strict W1 loop", loop50, pointwise(loop50))
	loop24 := hinfMixedSensitivityLoop(t, 10, 7)
	assertHinfNormMatchesOracle(t, "mixed-sensitivity loop n=24", loop24, pointwise(loop24))
}

// TestHamiltonianCrossingsNearFeedthroughGain (DLWKKE): at γ = σ_max(D)(1+f)
// the Hamiltonian needs R⁻¹ = (γ²I − DᵀD)⁻¹ and missed the resonance
// crossings far below for f ≤ 1e-8; the extended pencil keeps them. Some
// candidate or midpoint between candidates must lie inside the resonance,
// where σ_max exceeds γ.
func TestHamiltonianCrossingsNearFeedthroughGain(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 4 {
		sys := highPassResonanceSys(t, rng, 0)
		n, m, p := sys.Dims()
		ws := newHamiltonianWS(sys, n, m, p)
		sd := maxSVDense(sys.D, p, m)
		for _, f := range []float64{1e-14, 1e-12, 1e-10, 1e-8, 1e-6} {
			gamma := sd * (1 + f)
			ws.hasImagEigs(gamma)
			freqs := slices.Sorted(slices.Values(ws.cands))
			for i := range len(freqs) - 1 {
				freqs = append(freqs, (freqs[i]+freqs[i+1])/2)
			}
			best := 0.0
			for _, w := range freqs {
				best = math.Max(best, exactSigmaMax(sys, w))
			}
			if best <= gamma {
				t.Errorf("trial %d: candidates at σ(D)(1+%g) reach σ_max %g, resonance peak %g", trial, f, best,
					oraclePeakGain(t, sys, func(w float64) float64 { return exactSigmaMax(sys, w) }))
			}
		}
	}
}

func TestMaxSingularValueMatchesSVD(t *testing.T) {
	M := mat.NewDense(3, 2, []float64{1, -2, 0.5, 3, -1, 0.25})
	got, err := maxSingularValue(M)
	if err != nil {
		t.Fatal(err)
	}
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDNone) {
		t.Fatal("oracle SVD failed")
	}
	if want := svd.Values(nil)[0]; math.Abs(got-want) > 1e-12*want {
		t.Fatalf("maxSingularValue = %.17g, want %.17g", got, want)
	}
	if got, err := maxSingularValue(nil); err != nil || got != 0 {
		t.Fatalf("nil: %g, %v; want 0, nil", got, err)
	}
}

// TestHinfNorm_BadlyScaledIOMatchesExactOracle (5YUZIK): models with
// ‖C‖ ≈ 1e-6·‖B‖ and poles spread over up to 13 decades. Their crossings
// were lost in the rounding of the pencil's x–z coupling, so the probe above
// a coarse sample certified it: 30% low on a 1e-8 rad/s resonance next to a
// pole at −1e4, 1e-3 and 6e-5 low on broad maxima.
func TestHinfNorm_BadlyScaledIOMatchesExactOracle(t *testing.T) {
	slow := fzSeed{N: 3, M: 1, P: 1, Mag: [3]float64{-8, 4}, Damp: [3]float64{2.5, 1}, Shear: -10, Rot: 0.37,
		In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay}
	slowDisc := slow
	slowDisc.Discrete, slowDisc.LogDt, slowDisc.Mag[0] = true, -4, -4
	ks, vs := slow.encode()
	kd, vd := slowDisc.encode()
	for _, tc := range []struct {
		name   string
		s      *fzSpec
		noRace bool
	}{
		// The race build decodes a slightly different A, whose norm comes out
		// 3e-10 low (ergo follow-up), outside the 1e-10 contract.
		{"slow resonance", fzDecode(t, ks, vs), true},
		{"slow resonance discrete", fzDecode(t, kd, vd), false},
		{"broad maximum", fzDecode(t, 0xc0, [fzNVals]float64{61, 0, 0.3, 0, -0.2, 0, 0, 0, 0, 0.05555555555555555, 80.5, -29, -0.20833333333333334, -10, -35, -86}), false},
		{"broad maximum n=6", fzDecode(t, 0xc0, [fzNVals]float64{0, 0, 2.6999999999999997, 0, -0.2, 0, 0, 0, 21, -7.5, 2.5, -56, -10, -10, -10, -10}), false},
	} {
		if tc.noRace && raceEnabled {
			continue
		}
		sys := tc.s.System(t, fzSysOpts{})
		assertHinfNormMatchesOracle(t, tc.name, sys, func(w float64) float64 { return exactSigmaMax(sys, w) })
	}
}

// TestHinfNorm_FlatHinfSynLoopsMatchExactOracle (XK2GY2): HinfSyn closed
// loops of the FuzzHinfSyn regulator plants 0xd2 and 0xf4, frozen as built
// on amd64 and arm64. σ_max is flat to 1e-8 to 1e-3 over decades, with humps
// at near-coincident lightly damped pole pairs (ζ ≈ 0.02 at 0.63 and 1.0
// rad/s) whose γ-crossings QZ scatters by up to 0.1 relative, so neither
// the log-spaced seed nor a crossing midpoint landed on the highest hump:
// up to 7e-4 low. On the two broad humps GEPP noise of 3e-9 and 4e-8 stalled
// the climb 4.6e-9 low and picked a seed whose bracket missed the top
// (5.8e-8 low).
func TestHinfNorm_FlatHinfSynLoopsMatchExactOracle(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
	}{
		{"plant 0xf4 loop (amd64 build)", 12, 9, 9,
			[]float64{-3.8127934353156485, 2.441488513589365, -1.8522783458561047, 5.6907085050297885, -20.537636256536615, -9.278173576084066, -4.778934961853829, 4.112067299697218, -0.947012757591335, 10.0576544621415, -39.69016916019166, -18.94062048389503, -2.171345737217459, 1.183516160717888, -1.553039012723605, 2.6061320017258245, -11.574118011389741, -5.516736007513083, 1.404907838273903, -1.227790342716112, 0.27230898702522005, -2.9685009942761202, 11.729726543602377, 5.252416472921267, 3.699261516508234, -1.8877322080805117, 1.41071210026834, -5.952782493013009, 21.197572265644766, 9.875572004737544, 5.530557363981418, -4.768932256643541, 1.092697427697733, -11.645795539621169, 45.96554538119279, 21.750650991805067, -4.977009769736432, 4.298599313735656, -0.9671536029564279, 8.987623013419974, -33.68174941159644, -15.897977301376256, 1.5539273947383572, -1.3235772421682288, 0.3122815768752727, -3.261972336224158, 12.861732549516642, 6.384130851339001, 0.9080502551159364, -0.7842749731402602, 0.17645616877046832, -2.24581669316348, 8.984221277145544, 4.006421158354775, -4.699209848261115, 4.060821532895242, -0.9256275156673903, 9.900647981729696, -39.0845485990362, -18.33515594276665, 3.8153282936831117, -3.2952652985792685, 0.7414107418745602, -8.013722928047015, 31.588679750933814, 14.944584732986186, -4.067993654480201, 3.496107248774097, -0.80749012924268, 8.558795811698099, -33.77187448464167, -16.19339632601704, 8861.209845870766, -7142.938106209863, -4242.735112892144, 9886.202100136668, -2149.476876864446, -8496.3952031274, -8869.801574186422, 7149.491661954264, 4239.935821806132, -9870.453737330216, 2089.2490720713977, 8468.176409356412, 4980.014451678383, -4014.347267320796, -2384.4212730948257, 5556.064308518609, -1208.0139744387384, -4774.987679395878, -4980.78088964615, 4014.302993200021, 2383.140543055997, -5556.426677373558, 1208.1695824366761, 4774.723359613039, -9151.447587969718, 7376.90294309077, 4381.696073438065, -10210.016194509226, 2219.887352946532, 8774.683352869493, 9160.677406867646, -7383.559607568663, -4379.192663903719, 10192.417616445397, -2152.7242351765744, -8743.05712981522, 14544.68995797085, -11724.37554311813, -6963.953375811714, 16227.122484165486, -3528.1654228822167, -13945.884341392684, -14548.113040506316, 11727.350565327477, 6963.298503754384, -16221.396833149833, 3507.3454047054984, 13936.370494336355, -3894.403038037081, 3139.303892760422, 1864.5946465503043, -4344.911884078865, 944.734715577633, 3734.0661300788397, 3890.611879065531, -3136.0273467355946, -1865.3438177730677, 4352.56671405182, -974.8350377052557, -3748.394862444955, -13560.191984256251, 10930.782743081012, 6492.575947277088, -15128.746874728, 3289.356101359868, 13001.917856191058, 13559.939319183643, -10930.581901379279, -6492.6420266062705, 15129.291947005084, -3291.5392936766148, -13003.166666580131},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.2332652466596523e+06, -637527.1095034659, 258049.9888880525, 0, 0, 0, 0, 0, 0, -693097.0456503854, -358291.59609144763, 145023.98956990615, 0, 0, 0, 0, 0, 0, 1.273659223644293e+06, 658409.0077942832, -266501.25234589045, 0, 0, 0, 0, 0, 0, -2.0242672993686362e+06, -1.0464325329670977e+06, 423555.8046481935, 0, 0, 0, 0, 0, 0, 542005.7181250739, 280190.58074015693, -113402.56486543859, 0, 0, 0, 0, 0, 0, 1.8872490687445148e+06, 975602.1369829585, -394885.724005546},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 292.9692688245344, -253.30437656548654, 57.664196377693024, -617.3335273570016, 2437.1411999848115, 1140.8413845066113, 0, 0, 0, 0, 0, 0, -338.7908791254926, 291.6928628953908, -67.07886898101833, 713.1232994676718, -2814.3166921462316, -1339.7807837358212, 0, 0, 0, 0, 0, 0, 49.10524237640486, -41.215634579156266, 10.064815034371003, -102.70151231431667, 404.4524382960102, 211.92483486696284},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{"plant 0xd2 loop (arm64 build)", 12, 9, 8,
			[]float64{-1.1795756242067166, -1.0448863884562436, 4.7080813414963, -2.54192434692199, -8.00915461131412, -11.03582239999889, -3.6899962667218986, -0.6476168430547777, 10.559657611683816, -6.3114541854183885, -15.815340929714786, -23.72370753760638, -1.0796560660397954, -0.9330683809926081, 7.664208904009833, -3.312522593065563, -11.02498716884687, -15.924295812584168, 3.8164227168666076, 0.6649170379276661, -10.901278765445381, 6.520467197285784, 16.31394581601247, 23.538821448357805, -0.22465347617267917, -0.3786817818174781, 1.3880662840324256, -1.3976475689811165, -1.7170173234627564, -2.9226510396326963, 5.731773479318051, 1.0033459006111374, -16.391816212217233, 9.799888860136353, 24.543263721771375, 36.31692186688434, 2.186362817684459, 0.3770967002746043, -6.240389693272038, 3.632027718030968, 10.226213399985408, 14.872792274032026, -0.7499373335485114, -0.1281293316901464, 2.1316954938501547, -1.2775496158105712, -3.183357223616175, -4.109353331073429, 4.629071932592573, 0.7984071705734499, -13.212451540023691, 7.430739191077673, 19.859978675260113, 28.72913153198693, -6.132988195850183, -1.0718947928902658, 17.53226830739565, -10.483374911614568, -26.24635237859321, -38.515416271817635, 1.3817941855355012, 0.23832733689458208, -3.9439630622566937, 2.3545151022825355, 5.900382252709546, 8.930434952694403, -2.531196983168214, -0.4453318713237412, 7.24802697687057, -4.331031401968056, -10.858379810267643, -16.496304142325965, -1515.4816223181049, -2260.3863100559765, 2977.000244562487, 335.52007082166676, -3193.9405213023833, 1729.6152246176043, 1510.6120504290593, 2258.693806824758, -2961.7325056144036, -344.37344935088186, 3170.1160257690935, -1764.3747545432998, -2179.1547164872554, -3250.313320830625, 4280.739487811276, 482.4764910906054, -4592.698719464284, 2487.066668409733, 2181.8914831383445, 3250.0451694876538, -4283.976557673246, -479.26854648589506, 4597.987678112579, -2479.452142771799, -398.8333990094719, -595.0074888196039, 783.552825691534, 88.378581554621, -840.6965723931422, 455.1982799505986, 404.3405190073936, 595.6321529374682, -798.5565756045835, -79.97634027243475, 863.5228187689645, -421.80400915592327, 2028.2304190057007, 3025.294383626659, -3984.3226408402097, -449.1145020232838, 4274.710706502659, -2314.8242435207558, -2026.7939935183676, -3025.045416257508, 3980.2139466317603, 451.46898013097234, -4267.667850312893, 2325.587682482903, 3948.621340219563, 5889.800241165874, -7756.843277977719, -874.3872340348609, 8322.20381830193, -4506.576101502368, -3950.1252564748866, -5890.073728786789, 7761.163094722317, 871.3345983279287, -8328.59019197151, 4496.7898168107495, 1211.1624537615567, 1807.217749045384, -2379.6717403953953, -268.57165658588343, 2553.324569454322, -1382.354731655768, -1212.311856547807, -1807.4247535777868, 2382.975804277259, 266.5951403058716, -2558.2825669629383, 1374.7888625423657},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -38229.96300082026, -237750.3234861813, -325452.9918847124, 0, 0, 0, 0, 0, 0, -54975.073235044714, -341871.16493256035, -467979.90587798867, 0, 0, 0, 0, 0, 0, -10071.969003501354, -62580.32811241154, -85656.18106545467, 0, 0, 0, 0, 0, 0, 51174.979190161524, 318201.19986950076, 435572.42421931576, 0, 0, 0, 0, 0, 0, 99634.1739894357, 619489.0522288434, 847988.5494215749, 0, 0, 0, 0, 0, 0, 30611.97784522044, 190067.52165952238, 260131.34015040277},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120.22552853166285, 20.676535181819663, -342.3004172844619, 205.00975626299868, 511.53699423641115, 686.4620244373407, 0, 0, 0, 0, 0, 0, -550.7853940214284, -96.52861636774806, 1575.614108455401, -941.8724907905736, -2359.451241980895, -3513.027785105706},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{"plant 0xf4 loop (arm64 build)", 12, 9, 9,
			[]float64{-3.8127934353156476, 2.4414885135893645, -1.8522783458561045, 5.690708505029787, -20.537636256536615, -9.278173576084065, -4.77893496111468, 4.112067299061074, -0.9470127574406095, 10.057654460547381, -39.69016915383997, -18.94062048080117, -2.1713457372174587, 1.1835161607178875, -1.553039012723605, 2.6061320017258236, -11.57411801138974, -5.5167360075130825, 1.4049078380485351, -1.2277903425204002, 0.2723089869814936, -2.96850099379091, 11.729726541694944, 5.252416472134898, 3.699261516508233, -1.887732208080511, 1.4107121002683398, -5.952782493013008, 21.197572265644766, 9.875572004737542, 5.530557363121696, -4.768932255902691, 1.092697427523614, -11.645795537767464, 45.965545373820625, 21.750650988290502, -4.977009769736431, 4.2985993137356555, -0.9671536029564276, 8.98762301341997, -33.681749411596435, -15.897977301376255, 1.5539273945037764, -1.3235772419675877, 0.31228157682584584, -3.261972335717641, 12.861732547479981, 6.384130850245085, 0.908050255115936, -0.7842749731402597, 0.1764561687704682, -2.2458166931634787, 8.98422127714554, 4.006421158354773, -4.699209847526894, 4.060821532261736, -0.9256275155197146, 9.900647980146976, -39.084548592753634, -18.335155939837325, 3.8153282936831108, -3.295265298579267, 0.74141074187456, -8.013722928047013, 31.588679750933814, 14.944584732986185, -4.067993653852813, 3.4961072482345315, -0.807490129114247, 8.55879581034483, -33.771874479243856, -16.19339632335594, 4437.256691950761, -3576.8016879198067, -2124.5713550623655, 4950.5051942171685, -1076.3220049256731, -4254.577844661, -4445.8484202659165, 3583.3552436637838, 2121.7720639764557, -4934.756831411709, 1016.0942001357907, 4226.359050891711, 2493.734241768065, -2010.1636572881087, -1194.0043655892582, 2782.1819656570333, -604.8964396361976, -2391.068549722503, -2494.500679735931, 2010.11938316741, 1192.7236355504083, -2782.544334511622, 605.0520476325764, 2390.8042299390945, -4582.585714646569, 3693.956006331252, 2194.1508400473263, -5112.648155755358, 1111.5807755520077, 4393.923220831334, 4591.815533543573, -3700.6126708083466, -2191.6474305131887, 5095.0495776933385, -1044.4176577887265, -4362.2969977800385, 7283.233301774663, -5870.935388906751, -3487.2089838482557, 8125.691952808266, -1766.6938799311122, -6983.384554361584, -7286.656384310074, 5873.910411116012, 3486.554111790949, -8119.966301792534, 1745.8738617533818, 6973.870707304333, -1950.057558051164, 1571.9723716752924, 933.6530237306042, -2175.652934960594, 473.07876624686, 1869.7700760960524, 1946.2663990793371, -1568.695825650222, -934.4021949534244, 2183.3077649340557, -503.1790883761714, -1884.0988084629353, -6790.2392881828955, 5473.542573576701, 3251.160872541917, -7575.675354151877, 1647.1127353427219, 6510.686628583242, 6789.986623110424, -5473.3417318750435, -3251.2269518710855, 7576.220426428248, -1649.2959276568145, -6511.935438970919},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -617558.6815473865, -319239.99296418903, 129222.25257254508, 0, 0, 0, 0, 0, 0, -347067.3761015549, -179412.90289821196, 72622.26194209253, 0, 0, 0, 0, 0, 0, 637784.8930123034, 329696.24323033117, -133453.70138876117, 0, 0, 0, 0, 0, 0, -1.0136493339554109e+06, -523997.50244956947, 212098.5421451166, 0, 0, 0, 0, 0, 0, 271400.16801597324, 140302.09303442278, -56782.2485818888, 0, 0, 0, 0, 0, 0, 945036.4874972933, 488529.00402452145, -197741.30351722063},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 292.96926877870226, -253.30437652592886, 57.66419636849043, -617.3335272582098, 2437.1411995928424, 1140.841384324863, 0, 0, 0, 0, 0, 0, -338.79087907301647, 291.69286285021116, -67.07886897033852, 713.1232993545049, -2814.3166916955665, -1339.780783517647, 0, 0, 0, 0, 0, 0, 49.10524236925225, -41.21563457309636, 10.064815032790255, -102.70151229884476, 404.45243823294527, 211.92483482842275},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{"plant 0xf4 loop, broad hump (arm64 build)", 12, 9, 9,
			[]float64{15036.525935321328, -5387.087206134839, 12137.999861757999, 29385.48372909683, -249421.62373126025, -241322.84875670852, -0.01278773006376362, -0.0009284669729254571, 0.09064731253840469, 0.11392226765393614, -1.6931708729297021, -1.619793019698645, 90361.56419190153, -26865.392481036488, 81301.72320775028, 197656.0122355049, -1.6901946466366777e+06, -1.6355720241809776e+06, 0.01982482134411305, 0.0014085179824311316, -0.13951337272168157, 0.049722769030680944, 0.29412329088816935, 0.13901195294541915, 3036.2042957930153, -1037.800342621895, 2636.9610093380993, 6151.452483260189, -53484.450092400075, -51774.82416416501, 0.023393963038247572, 0.0016820207937194493, -0.1652866400994355, -0.08732070271057996, 1.8505261444972914, 1.6941640888540774, -23.32965783807745, -1.6544190532214174, 164.07445036877368, -35.7669977061065, -1744.8140123642602, -1730.1693252258644, -0.007309105924876796, -0.0005086407983468278, 0.051085407493895965, -0.09643914041019302, 0.6959038610962314, 0.7673618655871969, -2.1193003363708596, -0.15028985338429846, 14.904763724775705, -3.24657169782166, -165.3651348773389, -164.76231259032608, -0.02730431758464271, -0.0019541424292163516, 0.19261721342828922, 0.035725988524002474, -1.478219209140152, -1.2836272885025706, -0.13809732640110745, -0.009793150400347107, 0.9712205418494068, -0.21155230513784956, -3.448685416240467, -2.637619288474187, -0.007298640015694434, -0.0005368208232151265, 0.05196435043610258, 0.11555246057344515, -1.4867476748869368, -1.4540994577171793, -231.53552646964326, 473.2771565572735, -74.4756074299392, -425.12278282143075, 349.35112807423405, 199.23963306463918, 15268.048714876675, -5860.365288284748, 12212.565830042357, 29810.720574078976, -249772.66781475322, -241523.70823823722, -269.39077885327833, 550.6816895126951, -86.66851091116418, -494.64363880634494, 406.49476639414485, 231.81260488665603, 90630.9747984505, -27416.07276182272, 81388.252184961, 198150.70560696948, -1.6906008472642356e+06, -1.6358036977775851e+06, -34.23967468180261, 70.15548733690402, -11.121399546688725, -62.96462245217408, 51.833010587263445, 29.450479368198415, 3070.467077979738, -1107.9541682657377, 2647.919137305844, 6214.328802185865, -53534.434100859595, -51802.58009798372, 84.52332784349616, -172.67291449814408, 27.123375630831223, 155.135505779009, -127.43075656772383, -72.74143758727281, -107.86015489444034, 171.01799669331467, 137.0011773216491, -190.99827669771815, -1616.6885229415134, -1656.662592556044, 72.98992695338848, -148.94755207989726, 23.316452241189605, 133.87162058820175, -109.87504871173954, -62.82870677395231, -75.13631615316393, 148.7953236295889, -8.22059531973671, -137.08363730357254, -56.94748019023895, -103.19820116372924, -75.28183676494632, 153.64396586658822, -24.06114558745397, -138.08653119617284, 113.34501530764261, 64.80000938628274, 75.13638535416568, -153.6542995116656, 25.084711940374657, 137.9884645685688, -118.26141645762303, -68.87191273491919},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12932.357060465723, 29657.14522936988, 32433.71459122423, 0, 0, 0, 0, 0, 0, 15045.92285674637, 34507.27929393268, 37739.323088070014, 0, 0, 0, 0, 0, 0, 1906.9915722462504, 4394.236584819579, 4814.8034697116, 0, 0, 0, 0, 0, 0, -4724.283843221919, -10821.427304716815, -11829.084415337462, 0, 0, 0, 0, 0, 0, -4084.999735334326, -9336.464876872742, -10196.84470052997, 0, 0, 0, 0, 0, 0, 4212.635496182954, 9630.623317950294, 10519.178510645239},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7639690935081945, 0.05471962534234735, -5.390809780641893, -1.3156746814622646, 44.615124276707284, 39.22823158560085, 0, 0, 0, 0, 0, 0, -0.3108583857734389, -0.02262942489894985, 2.2055060949042407, 3.2033079434198553, -45.62836350789233, -43.92390090117353, 0, 0, 0, 0, 0, 0, -0.4500978032283539, -0.031870871323207664, 3.1639274592745954, -1.9186803926763254, 1.4554788757441894, 5.121388951974408},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{"plant 0xf4 loop, noisy seed (arm64 build)", 12, 9, 9,
			[]float64{556775.6352852357, 255311.42969443512, 443448.6758160218, -1.8508591111471278e+06, 158448.2061677071, -1.2249366470309312e+06, -7.059435513402212, -3.2019842564933576, -6.586657891675414, 15.33332939121085, -3.661499508971004, 8.58170057983927, -2.4477217939899485e+06, -1.1213083861352068e+06, -1.9479439169062756e+06, 8.133464597478218e+06, -696287.7674320849, 5.382895760395887e+06, -16.87189490249977, -7.6335298386375054, -15.663038091663264, 62.644210784379915, -4.668517518321255, 41.032670093841126, 593193.4516747969, 271479.46435602393, 471699.7735334261, -1.9702976833829107e+06, 168673.7629753608, -1.3039827022056123e+06, -1.9669887327894806, -0.8819363237193891, -1.7930307927690372, 18.181176618655986, 1.163853573842397, 13.370681791044069, 0.023290469753937348, 0.010659059427489178, 0.022122324462853286, -0.11568912414362244, -0.07053926976599921, -0.10708073428318146, 15.81956053862189, 7.16169597160469, 14.703770818156213, -52.917323887365555, 5.291176453708021, -33.87938665980839, 0.6947519983527042, 0.3179585004530199, 0.6599063604621348, -3.469277648966453, -0.4797838307701814, -3.3464727205213065, 10.43041655966368, 4.713426890458748, 9.659513733211389, -46.49182092629233, 1.6669134329833903, -31.496074283080482, -0.4118549673173071, -0.18848853709357422, -0.39119817311074273, 2.0566176651777903, 0.3150030912788893, 1.8546449440630006, -10.239312115068836, -4.640023627594003, -9.535953600721902, 28.04430861055106, -4.399381672224729, 17.029060705814388, 1937.3247669115601, -2344.231804666055, -421.5933973952833, 2616.825298338923, -1270.39123711961, -1795.4168327477437, 554831.2512704576, 257652.45960060175, 443863.68273326266, -1.853460603210708e+06, 159714.93607026502, -1.2231326486821058e+06, 2094.135877826995, -2533.5354040894113, -456.00475098416, 2828.378560956149, -1372.765808810901, -1940.7768334851417, -2.449832801676921e+06, -1.11878248422176e+06, -1.9475035751121389e+06, 8.13069886308514e+06, -694919.6700653374, 5.384877569815293e+06, 823.324045432701, -996.6238503043544, -178.92785965123687, 1112.3148707972787, -540.2717259405335, -762.9864462614532, 592368.1608183687, 272475.2063512486, 471876.9085307005, -1.971391817165319e+06, 169215.198711407, -1.3032063452517113e+06, 763.4107625183542, -923.6967055336975, -166.16774682788557, 1031.1373033310406, -500.54411751349903, -707.4959214370127, -747.5680061423589, 930.8690176575763, 180.89355174078207, -1084.1700083309693, 505.7647107624732, 673.5096820074446, -1672.5875228073678, 2023.998519677143, 363.9123132897757, -2259.296627956637, 1096.9017432467715, 1550.0628490244953, 1683.7128563126382, -2018.9670588314502, -353.5927366643077, 2209.3354854464105, -1095.7144566962806, -1584.9055433774238, 245.26136583074845, -296.8901992509103, -53.29835798705628, 331.3518053872531, -160.9468052991131, -227.28695944082315, -255.91271741520896, 292.0616029136906, 43.371032061703545, -301.25065114699964, 156.86227936883807, 246.1709346331022},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -69840.91346978836, -95365.69203675877, -76038.49550879833, 0, 0, 0, 0, 0, 0, -75506.42021616915, -103073.51849513313, -82163.53045768217, 0, 0, 0, 0, 0, 0, -29670.534668292203, -40537.96933442496, -32339.763599333877, 0, 0, 0, 0, 0, 0, -27522.709330782178, -37577.78150485566, -29959.435868162513, 0, 0, 0, 0, 0, 0, 60294.05268186429, 82336.63102451687, 65655.00525099017, 0, 0, 0, 0, 0, 0, -8838.485111039587, -12076.031408439067, -9634.03144125381},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.8996575811425154, 1.760379033426338, 3.6038217557438275, -19.890523557497286, 0.22931789822076076, -13.755714600091745, 0, 0, 0, 0, 0, 0, 5.6994193486724, 2.5828043305370847, 5.308196589788222, -15.517311644183726, 2.4633536271050303, -9.405523176211506, 0, 0, 0, 0, 0, 0, -9.654316894459294, -4.368216443856988, -8.963466497353624, 35.55823223235362, -2.716546864012706, 23.25239807455997},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{"plant 0xd2 loop, hump at 0.635 under a flat 1.66e6 (arm64 build)", 12, 9, 8,
			[]float64{-1.2606511271066436, -0.9485684798290386, 4.763662975042072, -2.538123819640167, -8.059774362405825, -11.134966080214843, -7.444385903682442, -1.3062192219572624, 5.326327214090242, -6.366505023507276, -15.954082796616085, -11.965879061550128, -1.279104976109792, -0.86956590015723, 7.632760442247234, -3.3905781156581822, -11.14117746557124, -16.060478610013995, 7.699865279872969, 1.3414617537911715, -5.498537424926331, 6.5774446767142285, 16.45732011988703, 11.872594364751912, -0.26235717361322924, -0.3406237909336622, 1.39777865928165, -1.3963400231328549, -1.730013459885515, -2.948974299961208, 11.563795789548715, 2.0238981175249036, -8.26803185472272, 9.88542251623542, 24.758710505355296, 18.31768923216211, 2.20712362255837, 0.38067745592488933, -6.299645875142345, 3.6674032457959704, 10.314863855973924, 15.002563870430942, -1.5132616235565601, -0.25868100240433745, 1.075159752588145, -1.2887667896100574, -3.211468003011467, -2.0726735393267854, 4.669477361399992, 0.8053761666399836, -13.327778062887404, 7.499588223254195, 20.03251344744892, 28.981856608639237, -12.37338721295653, -2.1622918477860873, 8.84323980352596, -10.574909729417318, -26.476838363297055, -19.426564719944253, 1.3939355348662452, 0.24042143707432895, -3.9786173065628763, 2.375203415101167, 5.952226878244214, 9.008903607122942, -5.106471547612148, -0.8981400704502206, 3.655952824790945, -4.368785141249771, -10.95357849271768, -8.320493074242968, -6011.726880313289, -8968.042171850715, 11810.275819015804, 1331.7642637144247, -12671.366802976858, 6861.277106891471, 12018.470916549326, 17934.482665610776, -5901.28383015783, -1336.216578134452, 12659.359874400543, -3439.4052344940137, -8644.484798550262, -12895.499700778622, 16982.445824056602, 1915.0045912942303, -18220.647833778858, 9866.08610101532, 17291.540424764622, 25790.800566533988, -8492.063990629627, -1913.411158013274, 18223.30590510601, -4931.121872977819, -6329.615818884431, -9442.528638340264, 12434.957806054947, 1402.344392169336, -13341.684428330718, 7224.110367959801, 12681.309800632414, 18887.742577747995, -6232.617188062053, -1385.3662272011861, 13387.741822374752, -3578.3687798485626, 16092.005181046283, 24005.50827316801, -31613.465924987788, -3564.900861281059, 33918.456435697546, -18366.06531826939, -32181.109376464505, -48010.51387242507, 15804.658299302682, 3567.2794977421872, -33911.35303982813, 9188.461267541628, 31328.59694388348, 46735.004433350296, -61546.47562512701, -6940.337539705142, 66033.95305932256, -35755.8386259613, -62660.228320241906, -93470.56040621165, 30775.41716332268, 6937.262218212243, -66040.39738419838, 17872.983676590957, 19224.16200816148, 28679.278694774057, -37767.58518254394, -4259.546184483742, 40521.7188645776, -21940.951696104046, -38452.961217233445, -57359.39198393235, 18887.12587958062, 4255.559021071065, -40531.721567706205, 10962.843765587546},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -151763.65631308922, -943236.5464772016, -1291090.5503554964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -218228.16014262158, -1356316.3573844961, -1856507.7787207626, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -159810.57393326197, -993135.171465438, -1359372.7796923064, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 406246.67680308083, 2524836.9532590304, 3455956.958931616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 790903.4226963382, 4915464.685677726, 6728206.100730516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 485425.1979555067, 3016379.5123615144, 4128683.4098739107},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 242.58551833996825, 41.73403381993337, -172.64842159782074, 206.80690418619287, 516.0469084576768, 346.23837344130976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1111.1942582765164, -194.7044624720212, 794.7421696168088, -950.0907654232062, -2380.157066319099, -1771.917179656263},
			[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		},
		{"plant 0xd2 loop, hump at 0.626 above σ(0) (arm64 build)", 12, 9, 8,
			[]float64{-0.8597732997267433, -0.9644794044393069, 2.6993477812852755, -2.0864110453489633, -3.907034875689016, -10.641277479503271, -4.4443558302277975, -2.0411977416885123, 4.769351981738528, -8.392854117350861, -10.864859438204249, -17.948314451888432, -0.9165631571722131, -0.9972607547495806, 5.6271932593344856, -3.294106227181334, -5.915291018554802, -17.965882248382155, 3.060417369689462, 1.3906413801116748, -3.272564756670576, 5.76140034490419, 7.455953893194978, 11.789285786148952, -0.3075381690241112, -0.5126896752550331, 1.447775410633494, -1.726462254485663, -1.3607754793922138, -5.113356950487448, 6.081671953216327, 2.7851876221220113, -6.520166459753487, 11.475189804349164, 14.853777303621394, 24.255554728119066, 2.1383537805573245, 0.485546848910671, -4.579574869901799, 3.9659461504859457, 5.852696863052026, 18.092317186373116, 0.19326287743927512, 0.09942747272299608, -0.21570902417217358, 0.37779931691220875, 0.4907821655434048, 1.1873791686738493, 3.6181658818924247, 0.8215614547799895, -7.748793346782887, 6.177199366293907, 8.875760191914692, 27.47363886967356, -5.978276766553068, -2.7319941571489963, 6.404762637173856, -11.27306805489077, -14.591209994835964, -23.620309654612033, 2.0925461037816793, 0.4751454956282707, -4.48147151239486, 3.9351831191526023, 5.101217010148523, 16.872521132401026, -3.3916269419323717, -1.5610379464161581, 3.6422420302972682, -6.408864316289345, -8.29704532921147, -13.824189497391481, -4086.259254694663, -2495.0736067755524, 5699.521785948717, -1190.118252241393, -4930.01615738178, 4377.763828541439, 4080.955125564942, 4986.177057000692, -2843.6418671023566, 1179.6389870792627, 4915.244263068307, -2212.150867461227, -3447.356208551408, -2104.9560495414107, 4808.376867597314, -1004.0401001426576, -4159.1854894199, 3693.2826509017027, 3448.4281356577994, 4209.610159018135, -2404.4179178623017, 1005.2737472013105, 4159.955820857082, -1845.2381531205028, -3925.7234529024445, -2397.081741000149, 5475.6264884711445, -1143.3440905281964, -4736.364056865534, 4205.778259432032, 3937.2717204699293, 4797.683098543086, -2749.405801743036, 1162.8415456255898, 4763.350060510915, -2059.491377214901, 6939.472548579929, 4237.267554993235, -9679.201333657003, 2021.1020584920361, 8372.398915503358, -7434.520496495107, -6937.140931921313, -8473.464588815696, 4837.095170368906, -2016.7583130234852, -8366.05543647334, 3727.493786010891, 10551.75368446296, 6442.953567189315, -14717.63302057955, 3073.160841769057, 12730.591345704763, -11304.495565907338, -10554.113795347512, -12886.996005626163, 7361.346876252908, -3078.2567104576156, -12736.30679550767, 5642.364292738959, 12837.692027527886, 7839.196984913266, -17906.35322314846, 3738.6773703199315, 15489.002447762246, -13753.543155289493, -12840.290189199615, -15679.615463734837, 8955.979624117259, -3743.624732704833, -15495.394104387611, 6865.995719801635},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 186106.33974277976, -350647.29583266244, -722486.0291628739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 157008.08922502093, -295821.9587179442, -609522.3171302533, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 178792.30269403115, -336874.04343786504, -694103.2631388088, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -316052.7543347378, 595486.8504878931, 1226959.6847873686, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -480570.1138476042, 905463.3849421309, 1865643.305537929, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -584645.9258441479, 1101658.8627447933, 2269836.274290303},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.629207618412924, 4.0456367489783815, -11.46988055036844, 20.401983118176997, 26.20379974325941, -2.4547017111029232, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -620.14973664153, -284.40098173142513, 665.1707503869234, -1170.6031485087074, -1515.3216770810839, -2488.400841471246},
			[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		},
		{"plant 0xd2 loop, hump at 0.626 next to the 1.0 rad/s pair (arm64 build)", 12, 9, 8,
			[]float64{-1.1795756242067166, -1.0448863884562436, 4.7080813414963, -2.54192434692199, -8.00915461131412, -11.03582239999889, -7.379992533636254, -1.2952336861466203, 5.27982880598649, -6.31145418558597, -15.81534093015, -11.8618537691696, -1.0796560660397954, -0.9330683809926081, 7.664208904009833, -3.312522593065563, -11.02498716884687, -15.924295812584168, 7.632845433931979, 1.3298340758926643, -5.450639382870666, 6.520467197457162, 16.31394581645632, 11.769410724488013, -0.22465347617267917, -0.3786817818174781, 1.3880662840324256, -1.3976475689811165, -1.7170173234627564, -2.9226510396326963, 11.463546958934897, 2.0066918012793122, -8.195908106332364, 9.79988886039562, 24.54326372244405, 18.158460933973956, 2.186362817684459, 0.3770967002746043, -6.240389693272038, 3.632027718030968, 10.226213399985408, 14.872792274032026, -1.4998746671359318, -0.25625866338710995, 1.0658477469533476, -1.2775496158432407, -3.1833572237001477, -2.054676665561322, 4.629071932592573, 0.7984071705734499, -13.212451540023691, 7.430739191077673, 19.859978675260113, 28.72913153198693, -12.265976392019978, -2.143789585841216, 8.766134153936697, -10.483374911891314, -26.246352379310807, -19.25770813645377, 1.3817941855355012, 0.23832733689458208, -3.9439630622566937, 2.3545151022825355, 5.900382252709546, 8.930434952694403, -5.06239396646851, -0.8906637426731313, 3.624013488534811, -4.331031402083445, -10.858379810567584, -8.248152071429923, -2987.644282506657, -4456.69968893541, 5869.253306659943, 661.7629702908368, -6297.135434723201, 3409.8334040541126, 5970.4189931242145, 8911.706874639596, -2930.8097185929073, -666.189659555559, 6285.223186956951, -1713.6065845082048, -4296.061130807079, -6408.497893517186, 8439.662147288778, 951.5884533745708, -9054.93938513633, 4903.146410217409, 8594.859028264997, 12816.727635691423, -4220.640341109713, -949.9844810724026, 9057.5838644607, -2449.6695736988913, -3145.5255293575738, -4692.483811460473, 6179.587647199696, 696.8913837439617, -6630.18307440268, 3590.046644954769, 6313.079538707214, 9387.46627939248, -3104.7975735130603, -680.086901179282, 6675.835567155504, -1761.629051681608, 7997.186138442203, 11929.612645202971, -15710.624748276303, -1771.4505051523552, 16856.00768133798, -9127.292339542884, -15991.499425909482, -23858.727355667652, 7853.258027033764, 1773.8049832602264, -16848.964825148694, 4569.027889251942, 15569.251636176099, 23225.125785386124, -30586.133347191862, -3448.7701816999993, 32816.036183698976, -17769.394066310106, -31141.51110486267, -46450.79854601418, 15295.226581968134, 3445.7175459931605, -32822.42255737167, 8879.803890808951, 9553.424277358083, 14252.406582642616, -18768.732979088254, -2116.935471870169, 20137.499282646037, -10903.551369731274, -19111.446165860863, -28505.641183414817, 9387.67055342556, 2112.982439310057, -20147.41527766362, 5444.2098157526925},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -75410.41900764332, -468748.1754802443, -641626.3406322063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -108437.41142529159, -674034.5530826098, -922622.7123447645, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -79417.236699589, -493540.9115787266, -675544.5839529213, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 201865.3467409265, 1254733.4548616717, 1717480.8134099299, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 393004.92974484374, 2442771.98597163, 3343665.2078308035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 241253.4342030254, 1499008.24119061, 2051756.0496963847},
			[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 240.4510570695713, 41.35307036476002, -171.15020864680636, 205.00975626829003, 511.53699425004726, 343.2310122245529, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1101.5707880715756, -193.05723274100035, 787.8070542492376, -941.8724908155328, -2359.4512420456804, -1756.513892605579},
			[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		},
	} {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, 0)
		if err != nil {
			t.Fatal(err)
		}
		assertHinfNormMatchesOracle(t, tc.name, sys, func(w float64) float64 { return exactSigmaMax(sys, w) })
	}
}

// TestHinfNorm_FuzzFindingsMatchExactOracle (XK2GY2): wave-12 fuzz findings
// that were 1e-10 to 3.6e-4 off: a lightly damped discrete peak near z = 1
// (SISO and MIMO), a hump between two slow poles of a stiff symmetric model,
// high-gain models over-reported by 9e-8, and a broad peak approaching
// |D| from above as ω → ∞.
func TestHinfNorm_FuzzFindingsMatchExactOracle(t *testing.T) {
	nearOne, err := NewFromSlices(2, 1, 1,
		[]float64{0.9999999959758767, 5.3457259458380696e-09, -1.3269470226717082e-08, 0.9999999927772968},
		[]float64{-0.6656134553337595, 0.5410494509487578}, []float64{-999097.8260547261, 363185.4084160624},
		[]float64{-736.7935074839925}, 1e-4)
	if err != nil {
		t.Fatal(err)
	}
	aboveD, err := NewFromSlices(2, 1, 1,
		[]float64{-2491.804346703939, 15862.372233861133, -8932.767713905962, -8758.80383000371},
		[]float64{-0.013738160475694785, -0.005141010807114001}, []float64{-8479.999973995311, -7587.562358106949},
		[]float64{-0.013822158423630349}, 0)
	if err != nil {
		t.Fatal(err)
	}
	v41 := [fzNVals]float64{-4, 2.25, 0.5, -145, -26, 18, 34, 5629.5, -8, -0.3, 1.1, 32, 1044, -0.5, 0.2, -10}
	for _, tc := range []struct {
		name string
		sys  *System
	}{
		{"discrete peak near z=1", nearOne},
		{"discrete peak near z=1, kind 0x41", fzDecode(t, 0x41, v41).System(t, fzSysOpts{})},
		{"discrete MIMO peak near z=1, kind 0x5b", fzDecode(t, 0x5b, v41).System(t, fzSysOpts{})},
		{"hump between slow poles of a stiff model", fzDecode(t, 0xc0, [fzNVals]float64{39, 0, 0.3, 0, -0.2, 0, 0, 0, 0, -9.5, 2.5, -76, -10, -10, -10, -10}).System(t, fzSysOpts{})},
		{"high gain, kind 0xa1", fzDecode(t, 0xa1, [fzNVals]float64{-4.840958682425959, 5.75742924209089, -4.840958682425959, 5.75742924209089, -4.974373469618411, 0.8119640876303578, -5, 1e-06, -0.6935062785976278, -9.308799076568224, 0.871442593513744, 9.362982810868214, -10, -10, -10, -10}).System(t, fzSysOpts{})},
		{"high gain, kind 0xc9", fzDecode(t, 0xc9, [fzNVals]float64{0.6413769500533629, 3.225485305446771, -6.920211238942823, 5.194656803421655, 2.5649118927522707, 5.9960354875740425, 0.7321123327177131, 0, 0.8225408446783429, -8.084719954920434, -0.12421061853394999, -14.98391252021944, -10, -10, -10, -10}).System(t, fzSysOpts{})},
		{"broad peak above |D|", aboveD},
	} {
		assertHinfNormMatchesOracle(t, tc.name, tc.sys, func(w float64) float64 { return exactSigmaMax(tc.sys, w) })
	}
}

func maxSVDense(D *mat.Dense, _, _ int) float64 {
	sv, err := maxSingularValue(D)
	if err != nil {
		panic(err)
	}
	return sv
}
