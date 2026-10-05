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
	if _, _, err := HinfNorm(sys); !errors.Is(err, ErrContinuousInternalDelay) {
		t.Fatalf("HinfNorm err = %v, want ErrContinuousInternalDelay", err)
	}
	if _, err := HSV(sys); !errors.Is(err, ErrContinuousInternalDelay) {
		t.Fatalf("HSV err = %v, want ErrContinuousInternalDelay", err)
	}
	if _, err := Norm(sys, math.Inf(1)); !errors.Is(err, ErrContinuousInternalDelay) {
		t.Fatalf("Norm(Inf) err = %v, want ErrContinuousInternalDelay", err)
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
	atEdge, err := hinfSynD11Zero(gp, 0.5074089765548706)
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

func maxSVDense(D *mat.Dense, _, _ int) float64 {
	sv, err := maxSingularValue(D)
	if err != nil {
		panic(err)
	}
	return sv
}
