package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
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

	_, _, err := HinfNorm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
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

// python-control: 1/(1-s) unstable => Hinf = 1.0, H2 = inf
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
	_, _, err = HinfNorm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("Hinf: got %v, want ErrUnstable", err)
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
