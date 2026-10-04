package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
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

	dsys, err := sys.DiscretizeZOH(0.1)
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

	dsys, err := sys.DiscretizeZOH(0.1)
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
