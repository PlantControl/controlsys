package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestDCGain_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g.At(0, 0)-1.0) > 1e-12 {
		t.Errorf("got %f, want 1.0", g.At(0, 0))
	}
}

func TestDCGain_SISO_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g.At(0, 0)-2.0) > 1e-12 {
		t.Errorf("got %f, want 2.0", g.At(0, 0))
	}
}

func TestDCGain_MIMO_NonSymmetricA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	// G(0) = -C*inv(A)*B = -inv(A) since C=I, B=I
	// A = [-2 1; 0 -3], inv(A) = [-1/2 -1/6; 0 -1/3]
	// G(0) = [1/2 1/6; 0 1/3]
	want := mat.NewDense(2, 2, []float64{0.5, 1.0 / 6, 0, 1.0 / 3})
	if !matEqual(g, want, 1e-12) {
		t.Errorf("got\n%v\nwant\n%v", mat.Formatted(g), mat.Formatted(want))
	}
}

func TestDCGain_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}), 0)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if !matEqual(g, want, 1e-15) {
		t.Errorf("DCGain of pure gain != D")
	}
}

func TestDCGain_Integrator_Infinite(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Fatalf("DCGain = %v, want +Inf", g.At(0, 0))
	}
}

func TestDCGain_NegativeIntegrator_NegativeInfinite(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), -1) {
		t.Fatalf("DCGain = %v, want -Inf", g.At(0, 0))
	}
}

func TestDCGain_DiscretePoleAtOne_Infinite(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Fatalf("DCGain = %v, want +Inf", g.At(0, 0))
	}
}

func TestDCGain_MIMOIntegratorChannelwise(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{
			0, 0,
			0, -2,
		}),
		mat.NewDense(2, 2, []float64{
			1, 0,
			0, 1,
		}),
		mat.NewDense(2, 2, []float64{
			1, 0,
			0, 1,
		}),
		mat.NewDense(2, 2, nil),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Fatalf("DCGain[0,0] = %v, want +Inf", g.At(0, 0))
	}
	if g.At(0, 1) != 0 || g.At(1, 0) != 0 {
		t.Fatalf("off-diagonal DCGain = [%v %v], want zeros", g.At(0, 1), g.At(1, 0))
	}
	if math.Abs(g.At(1, 1)-0.5) > 1e-12 {
		t.Fatalf("DCGain[1,1] = %v, want 0.5", g.At(1, 1))
	}
}

func TestDCGain_MIMOIntegratorHiddenFromOutput(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(2, 1, nil),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Fatalf("DCGain[0,0] = %v, want +Inf", g.At(0, 0))
	}
	if g.At(1, 0) != 0 {
		t.Fatalf("DCGain[1,0] = %v, want 0", g.At(1, 0))
	}
}

func TestDCGain_DecoupledIntegratorResiduesCancel(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{
			0, 0,
			0, 0,
		}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, -1}),
		mat.NewDense(1, 1, nil),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(g.At(0, 0), 0) || math.IsNaN(g.At(0, 0)) {
		t.Fatalf("DCGain = %v, want finite cancellation", g.At(0, 0))
	}
	if math.Abs(g.At(0, 0)) > 1e-12 {
		t.Fatalf("DCGain = %v, want 0", g.At(0, 0))
	}
}

func TestDCGain_CoupledSingularBlockFallsBackToTransferLimit(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{
			0, 1,
			0, 0,
		}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, nil),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(g.At(0, 0), 1) {
		t.Fatalf("DCGain = %v, want +Inf", g.At(0, 0))
	}
}

func TestDamp_ContinuousUnderdamped(t *testing.T) {
	wn := 10.0
	zeta := 0.3
	// Poles: -zeta*wn ± j*wn*sqrt(1-zeta^2)
	sigma := -zeta * wn
	wd := wn * math.Sqrt(1-zeta*zeta)
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -(wn * wn), 2 * sigma}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 2 {
		t.Fatalf("got %d poles, want 2", len(info))
	}

	for _, d := range info {
		if math.Abs(d.Wn-wn) > 1e-10 {
			t.Errorf("Wn = %f, want %f", d.Wn, wn)
		}
		if math.Abs(d.Zeta-zeta) > 1e-10 {
			t.Errorf("Zeta = %f, want %f", d.Zeta, zeta)
		}
		wantTau := 1.0 / (zeta * wn)
		if math.Abs(d.Tau-wantTau) > 1e-10 {
			t.Errorf("Tau = %f, want %f", d.Tau, wantTau)
		}
	}

	_ = wd
}

func TestDamp_Discrete(t *testing.T) {
	dt := 0.01
	wnCont := 10.0
	zetaCont := 0.5

	sigma := -zetaCont * wnCont
	wd := wnCont * math.Sqrt(1-zetaCont*zetaCont)
	p := cmplx.Exp(complex(sigma, wd) * complex(dt, 0))

	sys, _ := New(
		mat.NewDense(2, 2, []float64{
			real(p + cmplx.Conj(p)), -cmplx.Abs(p) * cmplx.Abs(p),
			1, 0,
		}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		dt,
	)

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}

	for _, d := range info {
		if math.Abs(d.Wn-wnCont) > 0.5 {
			t.Errorf("Wn = %f, want ~%f", d.Wn, wnCont)
		}
		if math.Abs(d.Zeta-zetaCont) > 0.05 {
			t.Errorf("Zeta = %f, want ~%f", d.Zeta, zetaCont)
		}
	}
}

func TestDamp_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 0 {
		t.Errorf("expected empty DampInfo for pure gain, got %d", len(info))
	}
}

func TestStep_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	resp, err := Step(sys, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := range steps {
		tk := resp.T[k]
		want := 1 - math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.01 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestStep_Discrete_Integrator(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)

	resp, err := Step(sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	for k := range 5 {
		got := resp.Y.At(0, k)
		want := float64(k)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("k=%d: got %f, want %f", k, got, want)
		}
	}
}

func TestStep_MIMO_NonSymmetricA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	resp, err := Step(sys, 3.0)
	if err != nil {
		t.Fatal(err)
	}

	rows, steps := resp.Y.Dims()
	if rows != 4 {
		t.Fatalf("Y rows = %d, want 4 (p*m=2*2)", rows)
	}

	dcGain, _ := sys.DCGain()
	lastK := steps - 1
	for j := range 2 {
		for i := range 2 {
			got := resp.Y.At(j*2+i, lastK)
			want := dcGain.At(i, j)
			if math.Abs(got-want) > 0.05 {
				t.Errorf("steady-state Y[%d,%d]=%f, want DC gain %f", i, j, got, want)
			}
		}
	}
}

func TestStep_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)
	resp, err := Step(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := range steps {
		if math.Abs(resp.Y.At(0, k)-3.0) > 1e-12 {
			t.Errorf("k=%d: got %f, want 3.0", k, resp.Y.At(0, k))
		}
	}
}

func TestImpulse_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	resp, err := Impulse(sys, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := 1; k < steps; k++ {
		tk := resp.T[k]
		want := math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.05 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestImpulse_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)

	resp, err := Impulse(sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	// y[0] = C*B*1 = 1*1*1 = 1 (B*impulse at k=0, then propagate through C)
	// Actually: y[0] = C*x0 + D*u[0] = 0 + 0 = 0... wait
	// x0 = nil → zero. y[0] = D*u[0] = 0. x[1] = A*0 + B*1 = 1
	// y[1] = C*x[1] + D*u[1] = 1 + 0 = 1
	// y[2] = C*(A*1 + B*0) = 0.5
	if math.Abs(resp.Y.At(0, 0)-0) > 1e-12 {
		t.Errorf("y[0] = %f, want 0 (D=0, x0=0)", resp.Y.At(0, 0))
	}
	if math.Abs(resp.Y.At(0, 1)-1) > 1e-12 {
		t.Errorf("y[1] = %f, want 1", resp.Y.At(0, 1))
	}
	if math.Abs(resp.Y.At(0, 2)-0.5) > 1e-12 {
		t.Errorf("y[2] = %f, want 0.5", resp.Y.At(0, 2))
	}
}

func TestInitial_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	x0 := mat.NewVecDense(1, []float64{1})
	resp, err := Initial(sys, x0, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := range steps {
		tk := resp.T[k]
		want := math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.01 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestInitial_NilX0_Error(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_, err := Initial(sys, nil, 1.0)
	if err == nil {
		t.Fatal("expected error for nil x0")
	}
}

// python-control: A=[[1,-2],[3,-4]], B=[[5],[7]], C=[[6,8]], D=[[9]]
// step response at t = linspace(0,1,10)
func TestStep_PythonControlVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)

	resp, err := Step(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	// python-control verified: y at t=linspace(0,1,10)
	want := []float64{9.0, 17.6457, 24.7072, 30.4855, 35.2234, 39.1165, 42.3227, 44.9694, 47.1599, 48.9776}

	_, steps := resp.Y.Dims()
	if steps < len(want) {
		t.Fatalf("only %d steps, want at least %d", steps, len(want))
	}

	for i, w := range want {
		ti := resp.T[0] + float64(i)*(1.0/float64(len(want)-1))
		k := 0
		for j := 1; j < steps; j++ {
			if math.Abs(resp.T[j]-ti) < math.Abs(resp.T[k]-ti) {
				k = j
			}
		}
		got := resp.Y.At(0, k)
		tol := 1.0
		if i == 0 {
			tol = 0.01 // D=9, so y(0)=9 exactly
		}
		if math.Abs(got-w) > tol {
			t.Errorf("t=%.3f: got %f, want %f (±%.1f)", resp.T[k], got, w, tol)
		}
	}
}

// python-control: A=[[1,-2],[3,-4]], B=[[5],[7]], C=[[6,8]], D=[[0]]
// C*B = 6*5 + 8*7 = 86, so continuous impulse should start near 86 and decay
func TestImpulse_PythonControlVerified(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{0}), 0)

	resp, err := Impulse(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	if steps < 10 {
		t.Fatal("too few steps")
	}

	foundNear86 := false
	for k := 0; k < min(10, steps); k++ {
		if math.Abs(resp.Y.At(0, k)-86.0) < 5.0 {
			foundNear86 = true
			break
		}
	}
	if !foundNear86 {
		t.Errorf("early impulse response should be near C*B=86")
	}

	yLast := resp.Y.At(0, steps-1)
	if math.Abs(yLast-14.89) > 3.0 {
		t.Errorf("final value = %f, want near 14.89", yLast)
	}
}

// python-control: discrete tf([1],[1,1,0.25], dt=True)
// step response: [0, 0, 1, 0, 0.75]
func TestStep_Discrete_PythonControl(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{1}}},
		Den:   [][]float64{{1, 1, 0.25}},
		Dt:    1.0,
		Delay: nil,
	}
	ssr, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(ssr.Sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{0, 0, 1, 0, 0.75}
	_, steps := resp.Y.Dims()
	if steps < len(want) {
		t.Fatalf("steps = %d, want >= %d", steps, len(want))
	}
	for k, w := range want {
		got := resp.Y.At(0, k)
		if math.Abs(got-w) > 0.05 {
			t.Errorf("y[%d] = %f, want %f", k, got, w)
		}
	}
}

// python-control: initial response siso_dss1
// A=[[-1,-0.25],[1,0]], B=[[1],[0]], C=[[0,1]], D=[[0]], dt=True
// X0=[0.5,1.0], expected: [1.0, 0.5, -0.75, 0.625, -0.4375]
func TestInitial_Discrete_PythonControl(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, -0.25, 1, 0}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}), 1.0)

	x0 := mat.NewVecDense(2, []float64{0.5, 1.0})
	resp, err := Initial(sys, x0, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{1.0, 0.5, -0.75, 0.625, -0.4375}
	_, steps := resp.Y.Dims()
	if steps < len(want) {
		t.Fatalf("steps = %d, want >= %d", steps, len(want))
	}
	for k, w := range want {
		got := resp.Y.At(0, k)
		if math.Abs(got-w) > 1e-10 {
			t.Errorf("y[%d] = %f, want %f", k, got, w)
		}
	}
}

// Impulse with feedthrough D!=0
func TestImpulse_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}), 0)

	resp, err := Impulse(sys, 3.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	if steps < 10 {
		t.Fatal("too few steps")
	}
	yLast := resp.Y.At(0, steps-1)
	if math.Abs(yLast) > 0.1 {
		t.Errorf("final value = %f, want ~0 (decay)", yLast)
	}
}

// Step for discrete system with feedthrough
func TestStep_Discrete_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), 1.0)

	resp, err := Step(sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	// y[0] = C*x0 + D*1 = 0 + 1 = 1
	// y[1] = C*(A*0+B*1) + D*1 = 1 + 1 = 2
	// y[2] = C*(A*1+B*1) + D*1 = 1.5 + 1 = 2.5
	if math.Abs(resp.Y.At(0, 0)-1.0) > 1e-12 {
		t.Errorf("y[0] = %f, want 1", resp.Y.At(0, 0))
	}
	if math.Abs(resp.Y.At(0, 1)-2.0) > 1e-12 {
		t.Errorf("y[1] = %f, want 2", resp.Y.At(0, 1))
	}
}

// Damp for real poles
func TestDamp_RealPoles(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -5}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 2 {
		t.Fatalf("got %d poles, want 2", len(info))
	}
	for _, d := range info {
		if d.Zeta != 1.0 {
			t.Errorf("real pole zeta = %f, want 1.0", d.Zeta)
		}
	}
}

func TestStep_PythonControl_D0(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{0}), 0)

	resp, err := Step(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{0., 8.6457, 15.7072, 21.4855, 26.2234, 30.1165, 33.3227, 35.9694, 38.1599, 39.9776}

	_, steps := resp.Y.Dims()
	if steps < len(want) {
		t.Fatalf("only %d steps, want at least %d", steps, len(want))
	}

	for i, w := range want {
		ti := resp.T[0] + float64(i)*(1.0/float64(len(want)-1))
		k := 0
		for j := 1; j < steps; j++ {
			if math.Abs(resp.T[j]-ti) < math.Abs(resp.T[k]-ti) {
				k = j
			}
		}
		got := resp.Y.At(0, k)
		tol := 1.0
		if i == 0 {
			tol = 1e-12
		}
		if math.Abs(got-w) > tol {
			t.Errorf("t=%.3f: got %f, want %f (±%.1f)", resp.T[k], got, w, tol)
		}
	}
}

func TestImpulse_PythonControl_D0(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{0}), 0)

	resp, err := Impulse(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344, 26.1668, 21.6292, 17.9245, 14.8945}

	_, steps := resp.Y.Dims()
	if steps < 10 {
		t.Fatal("too few steps")
	}

	foundNear86 := false
	for k := 0; k < min(10, steps); k++ {
		if math.Abs(resp.Y.At(0, k)-86.0) < 5.0 {
			foundNear86 = true
			break
		}
	}
	if !foundNear86 {
		t.Errorf("first non-zero sample should be near C*B=86")
	}

	lastK := steps - 1
	if math.Abs(resp.Y.At(0, lastK)-want[len(want)-1]) > 2.0 {
		t.Errorf("final value got %f, want near %f", resp.Y.At(0, lastK), want[len(want)-1])
	}
}

func TestInitial_PythonControl_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)

	x0 := mat.NewVecDense(2, []float64{0.5, 1.0})
	resp, err := Initial(sys, x0, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092, 1.1508, 0.5833, 0.1645, -0.1391}

	_, steps := resp.Y.Dims()
	if steps < len(want) {
		t.Fatalf("only %d steps, want at least %d", steps, len(want))
	}

	for i, w := range want {
		ti := resp.T[0] + float64(i)*(1.0/float64(len(want)-1))
		k := 0
		for j := 1; j < steps; j++ {
			if math.Abs(resp.T[j]-ti) < math.Abs(resp.T[k]-ti) {
				k = j
			}
		}
		got := resp.Y.At(0, k)
		tol := 1.0
		if i == 0 {
			tol = 0.01
		}
		if math.Abs(got-w) > tol {
			t.Errorf("t=%.3f: got %f, want %f (±%.2f)", resp.T[k], got, w, tol)
		}
	}
}

func TestStep_AutoTFinal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	resp, err := Step(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) < 50 {
		t.Errorf("auto tFinal produced only %d points, expected more", len(resp.T))
	}
	last := resp.Y.At(0, len(resp.T)-1)
	if math.Abs(last-1.0) > 0.01 {
		t.Errorf("final value %f not near steady-state 1.0", last)
	}
}

func trMIMO(t *testing.T, dt float64) *System {
	t.Helper()
	a := []float64{-1, 2, 0.5, -3}
	if dt > 0 {
		a = []float64{0.5, 0.2, -0.1, 0.7}
	}
	sys, err := New(
		mat.NewDense(2, 2, a),
		mat.NewDense(2, 2, []float64{1, 0.3, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}),
		mat.NewDense(2, 2, []float64{0.2, 0.1, 0, -0.3}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func trExp(a *mat.Dense, t float64) *mat.Dense {
	var scaled, e mat.Dense
	scaled.Scale(t, a)
	e.Exp(&scaled)
	return &e
}

// trImpulse is C·e^{A(t−τ)}·B for t ≥ τ, else 0.
func trImpulse(a, b, c *mat.Dense, i, j int, t, tau float64) float64 {
	if t < tau {
		return 0
	}
	var cb, h mat.Dense
	cb.Mul(c, trExp(a, t-tau))
	h.Mul(&cb, b)
	return h.At(i, j)
}

func trIODelay(sys *System, i, j int) float64 {
	tau := 0.0
	if sys.Delay != nil {
		tau += sys.Delay.At(i, j)
	}
	if sys.InputDelay != nil {
		tau += sys.InputDelay[j]
	}
	if sys.OutputDelay != nil {
		tau += sys.OutputDelay[i]
	}
	return tau
}

func TestTimeResponseGridIncludesFinalSample(t *testing.T) {
	disc := trMIMO(t, 0.1)
	resp, err := Step(disc, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) != 4 || math.Abs(resp.T[3]-0.3) > 1e-12 {
		t.Fatalf("discrete T = %v, want 4 samples ending at 0.3", resp.T)
	}

	cont := trMIMO(t, 0)
	for _, tFinal := range []float64{0.3, 0.7, 2.3, 4.6} {
		resp, err := Step(cont, tFinal)
		if err != nil {
			t.Fatal(err)
		}
		if last := resp.T[len(resp.T)-1]; last != tFinal {
			t.Errorf("tFinal=%g: last sample %g", tFinal, last)
		}
	}
}

func TestTimeResponseAutoDtFollowsUserFinalTime(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	resp, err := Step(sys, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) < 101 {
		t.Fatalf("Step(0.5) has %d samples; dt must resolve the requested horizon", len(resp.T))
	}
	for k, tk := range resp.T {
		if want := 1 - math.Exp(-tk); math.Abs(resp.Y.At(0, k)-want) > 1e-12 {
			t.Fatalf("y(%g) = %g, want %g", tk, resp.Y.At(0, k), want)
		}
	}
}

func TestTimeResponseRejectsNonFiniteFinalTime(t *testing.T) {
	x0 := mat.NewVecDense(2, []float64{1, -1})
	for _, dt := range []float64{0, 0.1} {
		sys := trMIMO(t, dt)
		for _, tFinal := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			if _, err := Step(sys, tFinal); err == nil {
				t.Errorf("dt=%g Step(%g): want error", dt, tFinal)
			}
			if _, err := Impulse(sys, tFinal); err == nil {
				t.Errorf("dt=%g Impulse(%g): want error", dt, tFinal)
			}
			if _, err := Initial(sys, x0, tFinal); err == nil {
				t.Errorf("dt=%g Initial(%g): want error", dt, tFinal)
			}
		}
	}
}

func TestTimeResponseAutoHorizonCoversSettling(t *testing.T) {
	disc, _ := New(mat.NewDense(2, 2, []float64{1.6, -0.81, 1, 0}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{0.1, 0.11}), mat.NewDense(1, 1, nil), 0.1)
	zeta, wn := 0.05, 2.0
	cont, _ := New(mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}), mat.NewDense(2, 1, []float64{0, wn * wn}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	delayed, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	delayed.InputDelay = []float64{20}

	for name, sys := range map[string]*System{"discrete": disc, "lightly damped": cont, "long delay": delayed} {
		resp, err := Step(sys, 0)
		if err != nil {
			t.Fatal(err)
		}
		gain, _ := sys.DCGain()
		last := resp.Y.At(0, len(resp.T)-1)
		if math.Abs(last-gain.At(0, 0)) > 0.02*math.Abs(gain.At(0, 0)) {
			t.Errorf("%s: auto horizon ends at t=%g with y=%g, DC gain %g", name, resp.T[len(resp.T)-1], last, gain.At(0, 0))
		}
	}
}

func TestTimeResponseAutoHorizonDeadbeat(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	resp, err := Step(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) < 2 {
		t.Fatalf("deadbeat auto Step has %d samples", len(resp.T))
	}
	for k := range resp.T {
		want := 1.0
		if k == 0 {
			want = 0
		}
		if resp.Y.At(0, k) != want {
			t.Fatalf("y[%d] = %g, want %g", k, resp.Y.At(0, k), want)
		}
	}
}

func TestTimeResponseAutoHorizonStiffBounded(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{-1e-3, 1, 0, -1e6})
	b := mat.NewDense(2, 1, []float64{0, 1})
	c := mat.NewDense(1, 2, []float64{1, 0})
	sys, _ := New(a, b, c, mat.NewDense(1, 1, nil), 0)
	resp, err := Step(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) > autoMaxSamples {
		t.Fatalf("stiff auto Step has %d samples, cap %d", len(resp.T), autoMaxSamples)
	}
	var aInv mat.Dense
	if err := aInv.Inverse(a); err != nil {
		t.Fatal(err)
	}
	for _, k := range []int{1, len(resp.T) / 2, len(resp.T) - 1} {
		var e, m, y mat.Dense
		e.Sub(trExp(a, resp.T[k]), eye(2))
		m.Mul(&aInv, &e)
		y.Mul(c, &m)
		var want mat.Dense
		want.Mul(&y, b)
		if math.Abs(resp.Y.At(0, k)-want.At(0, 0)) > 1e-7*math.Abs(want.At(0, 0)) {
			t.Errorf("y(%g) = %g, want %g", resp.T[k], resp.Y.At(0, k), want.At(0, 0))
		}
	}
}

func TestImpulseContinuousIsExactlySampled(t *testing.T) {
	plain := trMIMO(t, 0)
	delayed := trMIMO(t, 0)
	delayed.InputDelay = []float64{0.137, 0}
	delayed.OutputDelay = []float64{0, 0.291}
	iod := trMIMO(t, 0)
	iod.Delay = mat.NewDense(2, 2, []float64{0.413, 0, 0, 0})

	for name, sys := range map[string]*System{"plain": plain, "input+output delay": delayed, "iodelay": iod} {
		resp, err := Impulse(sys, 2)
		if err != nil {
			t.Fatal(err)
		}
		worst := 0.0
		for k, tk := range resp.T {
			for j := range 2 {
				for i := range 2 {
					want := trImpulse(sys.A, sys.B, sys.C, i, j, tk, trIODelay(sys, i, j))
					worst = math.Max(worst, math.Abs(resp.Y.At(j*2+i, k)-want))
				}
			}
		}
		if worst > 1e-9 {
			t.Errorf("%s: worst impulse error %g", name, worst)
		}
	}

	resp, _ := Impulse(plain, 1)
	var cb mat.Dense
	cb.Mul(plain.C, plain.B)
	for j := range 2 {
		for i := range 2 {
			if math.Abs(resp.Y.At(j*2+i, 0)-cb.At(i, j)) > 1e-14 {
				t.Errorf("y_%d%d(0) = %g, want C·B = %g", i, j, resp.Y.At(j*2+i, 0), cb.At(i, j))
			}
		}
	}
}

func TestImpulseContinuousDescriptor(t *testing.T) {
	e := mat.NewDense(2, 2, []float64{2, 0.5, -0.3, 1.5})
	explicit := trMIMO(t, 0)
	var a, b mat.Dense
	a.Mul(e, explicit.A)
	b.Mul(e, explicit.B)
	desc, err := NewDescriptor(&a, &b, explicit.C, explicit.D, e, 0)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Impulse(desc, 2)
	if err != nil {
		t.Fatal(err)
	}
	for k, tk := range resp.T {
		for j := range 2 {
			for i := range 2 {
				want := trImpulse(explicit.A, explicit.B, explicit.C, i, j, tk, 0)
				if math.Abs(resp.Y.At(j*2+i, k)-want) > 1e-9 {
					t.Fatalf("h_%d%d(%g) = %g, want %g", i, j, tk, resp.Y.At(j*2+i, k), want)
				}
			}
		}
	}
}

func TestImpulseContinuousInternalDelayOutputPath(t *testing.T) {
	sys := trMIMO(t, 0)
	grid, err := Impulse(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	tau := 25 * grid.T[1]
	d12 := mat.NewDense(2, 1, []float64{0.7, -0.4})
	c2 := mat.NewDense(1, 2, []float64{0.3, 1})
	if err := sys.SetInternalDelay([]float64{tau}, mat.NewDense(2, 1, nil), c2, d12, mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	resp, err := Impulse(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	var d12c2 mat.Dense
	d12c2.Mul(d12, c2)
	if resp.T[1] != grid.T[1] {
		t.Fatalf("grid changed: dt %g vs %g", resp.T[1], grid.T[1])
	}
	for k, tk := range resp.T {
		for j := range 2 {
			for i := range 2 {
				want := trImpulse(sys.A, sys.B, sys.C, i, j, tk, 0) + trImpulse(sys.A, sys.B, &d12c2, i, j, tk, tau-1e-12)
				if math.Abs(resp.Y.At(j*2+i, k)-want) > 1e-9 {
					t.Fatalf("h_%d%d(%g) = %g, want %g", i, j, tk, resp.Y.At(j*2+i, k), want)
				}
			}
		}
	}
}

func TestDamp_DiscretePoleAtOrigin(t *testing.T) {
	sys, _ := New(mat.NewDense(2, 2, []float64{0, 0.3, 0, -0.5}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, nil), 0.1)
	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, d := range info {
		if d.Pole != 0 {
			continue
		}
		found = true
		if !math.IsInf(d.Wn, 1) || d.Zeta != 1 || d.Tau != 0 {
			t.Errorf("z=0: Wn=%g Zeta=%g Tau=%g, want +Inf 1 0", d.Wn, d.Zeta, d.Tau)
		}
	}
	if !found {
		t.Fatal("pole at origin not reported")
	}
}

func TestDCGain_DescriptorSingularAFallsBackToLimit(t *testing.T) {
	e := mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, -1, 0, 0.4, 1.5})
	b := mat.NewDense(3, 1, []float64{1, 0, 0})
	c := mat.NewDense(1, 3, []float64{0, 1, 0})
	for _, tc := range []struct {
		dt float64
		a  []float64
	}{
		{0, []float64{0, 0, 0, 0, -1, 0.5, 0, 0.3, -2}},
		{0.1, []float64{1, 0, 0, 0, 0.5, 0.1, 0, 0.2, -0.3}},
	} {
		sys, err := NewDescriptor(mat.NewDense(3, 3, tc.a), b, c, mat.NewDense(1, 1, nil), e, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		gain, err := sys.DCGain()
		if err != nil {
			t.Fatalf("dt=%g: %v", tc.dt, err)
		}
		point := complex(0, 1e-9)
		if tc.dt > 0 {
			point = cmplx.Exp(point)
		}
		near, err := sys.EvalFr(point)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(gain.At(0, 0)-real(near[0][0])) > 1e-6 {
			t.Errorf("dt=%g: DCGain %g, G near DC %v", tc.dt, gain.At(0, 0), near[0][0])
		}
	}
}

func TestDCGain_InternalDelayWithIntegrator(t *testing.T) {
	sys, _ := New(mat.NewDense(2, 2, []float64{0, 0, 0, -2}), mat.NewDense(2, 1, []float64{1, 1}), eye(2), mat.NewDense(2, 1, nil), 0)
	if err := sys.SetInternalDelay([]float64{0.5}, mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{0, 1}), mat.NewDense(2, 1, nil), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	gain, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(gain.At(0, 0), 1) || math.Abs(gain.At(1, 0)-1) > 1e-12 {
		t.Fatalf("DCGain = %v, want [+Inf; 1] for [1/s; 1/(s+2-e^{-s/2})]", mat.Formatted(gain))
	}
}

func trFreeOracle(sys *System, x0 *mat.VecDense, i int, t float64) float64 {
	if sys.OutputDelay != nil {
		t -= sys.OutputDelay[i]
	}
	if t < 0 {
		return 0
	}
	var x mat.VecDense
	x.MulVec(trExp(sys.A, t), x0)
	return mat.Dot(sys.C.RowView(i), &x)
}

func TestInitialContinuousDelaysExact(t *testing.T) {
	x0 := mat.NewVecDense(2, []float64{1, -0.5})
	decomposable := mat.NewDense(2, 2, []float64{0.1, 0.35, 0.4, 0.65})
	for _, tc := range []struct {
		name    string
		in, out []float64
		iod     *mat.Dense
		share   []float64
	}{
		{"fractional input", []float64{0.013, 0.2}, nil, nil, nil},
		{"output", nil, []float64{0, 0.5}, nil, nil},
		{"fractional output", nil, []float64{0.013, 0.5}, nil, nil},
		{"decomposable iodelay", nil, nil, decomposable, []float64{0, 0.3}},
		{"mixed", []float64{0.1, 0.2}, []float64{0, 0.3}, decomposable, []float64{0, 0.3}},
	} {
		sys := trMIMO(t, 0)
		sys.InputDelay, sys.OutputDelay, sys.Delay = tc.in, tc.out, tc.iod
		resp, err := Initial(sys, x0, 2)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for k, tk := range resp.T {
			for i := range 2 {
				ti := tk
				if tc.share != nil {
					ti -= tc.share[i]
				}
				if want := trFreeOracle(sys, x0, i, ti); math.Abs(resp.Y.At(i, k)-want) > 1e-9 {
					t.Fatalf("%s: y_%d(%g) = %g, want %g", tc.name, i, tk, resp.Y.At(i, k), want)
				}
			}
		}
	}

	sys := trMIMO(t, 0)
	sys.InputDelay, sys.OutputDelay = []float64{0.1, 0.2}, []float64{0, 0.3}
	sys.Delay = mat.NewDense(2, 2, []float64{0.4, 0, 0, 0})
	if _, err := Initial(sys, x0, 2); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("nondecomposable iodelay: err = %v, want ErrDelayUnsupported", err)
	}
	if _, err := Lsim(sys, mat.NewDense(3, 2, nil), []float64{0, 0.1, 0.2}, x0); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("nondecomposable iodelay Lsim: err = %v, want ErrDelayUnsupported", err)
	}
}

func TestLsimContinuousX0WithFractionalOutputDelay(t *testing.T) {
	sys := trMIMO(t, 0)
	sys.OutputDelay = []float64{0.013, 0.5}
	sys.Delay = mat.NewDense(2, 2, []float64{0.1, 0.35, 0.4, 0.65})
	share := []float64{0, 0.3}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})
	steps := 61
	tv := make([]float64, steps)
	u := mat.NewDense(steps, 2, nil)
	for k := range tv {
		tv[k] = 0.03 * float64(k)
		u.Set(k, 1, 1)
	}
	resp, err := Lsim(sys, u, tv, x0)
	if err != nil {
		t.Fatal(err)
	}
	var aInv mat.Dense
	if err := aInv.Inverse(sys.A); err != nil {
		t.Fatal(err)
	}
	for k, tk := range tv {
		for i := range 2 {
			want := trFreeOracle(sys, x0, i, tk-share[i])
			if s := tk - trIODelay(sys, i, 1); s >= -1e-12 {
				var e, m, y mat.Dense
				e.Sub(trExp(sys.A, math.Max(s, 0)), eye(2))
				m.Mul(&aInv, &e)
				y.Mul(sys.C, &m)
				var g mat.Dense
				g.Mul(&y, sys.B)
				want += g.At(i, 1) + sys.D.At(i, 1)
			}
			if math.Abs(resp.Y.At(i, k)-want) > 1e-9 {
				t.Fatalf("y_%d(%g) = %g, want %g", i, tk, resp.Y.At(i, k), want)
			}
		}
	}
}

// trDDE integrates x' = Ax + Bu + B2·w, w(t) = z(t−τ), z = C2x + D21u + D22w
// with zero history by RK4 on a grid h (τ = lag·h) using cubic Hermite dense
// output for delayed values; global error O(h⁴).
type trDDE struct {
	sys      *System
	h        float64
	lag      int
	u        func(i int) []float64
	x, z     [][]float64
	fp, fm   [][]float64
	n, m, q  int
	uZero    []float64
	zeroHist []float64
}

func newTrDDE(sys *System, h float64, lag, steps int, x0 []float64, u func(i int) []float64) *trDDE {
	n, m, _ := sys.Dims()
	q := len(sys.LFT.Tau)
	d := &trDDE{sys: sys, h: h, lag: lag, u: u, n: n, m: m, q: q, uZero: make([]float64, m), zeroHist: make([]float64, q)}
	if d.u == nil {
		d.u = func(int) []float64 { return d.uZero }
	}
	d.x = make([][]float64, steps+1)
	d.z = make([][]float64, steps+1)
	d.fp = make([][]float64, steps+1)
	d.fm = make([][]float64, steps+1)
	d.x[0] = append([]float64(nil), x0...)
	d.z[0] = d.zOf(d.x[0], d.u(0), d.zeroHist)
	for i := range steps {
		wp, wmid, wm := d.zRight(i-lag), d.zAt(i-lag, 0.5), d.zAt(i-lag, 1)
		ui := d.u(i)
		k1 := d.f(d.x[i], ui, wp)
		k2 := d.f(trAxpy(d.x[i], h/2, k1), ui, wmid)
		k3 := d.f(trAxpy(d.x[i], h/2, k2), ui, wmid)
		k4 := d.f(trAxpy(d.x[i], h, k3), ui, wm)
		next := make([]float64, n)
		for s := range n {
			next[s] = d.x[i][s] + h/6*(k1[s]+2*k2[s]+2*k3[s]+k4[s])
		}
		d.x[i+1] = next
		d.fp[i] = k1
		d.fm[i+1] = d.f(next, ui, wm)
		d.z[i+1] = d.zOf(next, d.u(i+1), d.zRight(i+1-lag))
	}
	return d
}

func trAxpy(x []float64, a float64, y []float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] + a*y[i]
	}
	return out
}

func trMulVec(a *mat.Dense, x []float64) []float64 {
	r, _ := a.Dims()
	out := make([]float64, r)
	for i := range r {
		for j, v := range x {
			out[i] += a.At(i, j) * v
		}
	}
	return out
}

func trAdd(vs ...[]float64) []float64 {
	out := make([]float64, len(vs[0]))
	for _, v := range vs {
		for i := range v {
			out[i] += v[i]
		}
	}
	return out
}

func (d *trDDE) f(x, u, w []float64) []float64 {
	return trAdd(trMulVec(d.sys.A, x), trMulVec(d.sys.B, u), trMulVec(d.sys.LFT.B2, w))
}

func (d *trDDE) zOf(x, u, w []float64) []float64 {
	return trAdd(trMulVec(d.sys.LFT.C2, x), trMulVec(d.sys.LFT.D21, u), trMulVec(d.sys.LFT.D22, w))
}

func (d *trDDE) zRight(i int) []float64 {
	if i < 0 {
		return d.zeroHist
	}
	return d.z[i]
}

func (d *trDDE) zAt(i int, theta float64) []float64 {
	if i < 0 {
		return d.zeroHist
	}
	if theta == 0 {
		return d.z[i]
	}
	h00 := 2*theta*theta*theta - 3*theta*theta + 1
	h10 := theta*theta*theta - 2*theta*theta + theta
	h01 := -2*theta*theta*theta + 3*theta*theta
	h11 := theta*theta*theta - theta*theta
	x := make([]float64, d.n)
	for s := range d.n {
		x[s] = h00*d.x[i][s] + h10*d.h*d.fp[i][s] + h01*d.x[i+1][s] + h11*d.h*d.fm[i+1][s]
	}
	return d.zOf(x, d.u(i), d.zAt(i-d.lag, theta))
}

func (d *trDDE) y(i int) []float64 {
	if i < 0 {
		_, _, p := d.sys.Dims()
		return make([]float64, p)
	}
	return trAdd(trMulVec(d.sys.C, d.x[i]), trMulVec(d.sys.D, d.u(i)), trMulVec(d.sys.LFT.D12, d.zRight(i-d.lag)))
}

// trDelayFeedback is trMIMO with two internal delays of equal length τ fed
// back through B2 and D22, and fed by u through D21.
func trDelayFeedback(t *testing.T, tau float64, d21 bool) *System {
	t.Helper()
	sys := trMIMO(t, 0)
	D21 := mat.NewDense(2, 2, []float64{1, 0, 0.2, -0.6})
	if !d21 {
		D21 = mat.NewDense(2, 2, nil)
	}
	if err := sys.SetInternalDelay([]float64{tau, tau},
		mat.NewDense(2, 2, []float64{0.4, -0.2, 0.1, 0.3}),
		mat.NewDense(2, 2, []float64{0.3, 1, -0.5, 0.2}),
		mat.NewDense(2, 2, []float64{0.7, 0.1, -0.4, 0.5}),
		D21,
		mat.NewDense(2, 2, []float64{0.3, 0.1, 0, -0.2}),
	); err != nil {
		t.Fatal(err)
	}
	return sys
}

const trSub = 100

// trChainGrid returns the auto step of Step(·, 2) for trMIMO and a delay
// τ = 23.37·dt that is a whole number of oracle substeps but not of dt.
func trChainGrid(t *testing.T) (dt, tau float64, lag int) {
	t.Helper()
	zero, err := trDelayFeedback(t, 1, true).ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Step(zero, 2)
	if err != nil {
		t.Fatal(err)
	}
	dt = resp.T[1]
	lag = 2337
	return dt, float64(lag) * dt / trSub, lag
}

func TestStepContinuousInternalDelayExact(t *testing.T) {
	dt, tau, lag := trChainGrid(t)
	sys := trDelayFeedback(t, tau, true)
	resp, err := Step(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	if resp.T[1] != dt {
		t.Fatalf("dt = %g, want %g", resp.T[1], dt)
	}
	steps := len(resp.T)
	for in := range 2 {
		u := make([]float64, 2)
		u[in] = 1
		dde := newTrDDE(sys, dt/trSub, lag, (steps-1)*trSub, []float64{0, 0}, func(int) []float64 { return u })
		for k := range steps {
			want := dde.y(k * trSub)
			for out := range 2 {
				if got := resp.Y.At(in*2+out, k); math.Abs(got-want[out]) > 1e-9 {
					t.Fatalf("u%d→y%d t=%g: got %.12g, want %.12g", in, out, resp.T[k], got, want[out])
				}
			}
		}
	}
}

func TestInitialContinuousInternalDelayFeedbackExact(t *testing.T) {
	dt, tau, lag := trChainGrid(t)
	sys := trDelayFeedback(t, tau, true)
	x0 := []float64{1, -0.5}
	resp, err := Initial(sys, mat.NewVecDense(2, x0), 2)
	if err != nil {
		t.Fatal(err)
	}
	steps := len(resp.T)
	dde := newTrDDE(sys, dt/trSub, lag, (steps-1)*trSub, x0, nil)
	for k := range steps {
		want := dde.y(k * trSub)
		for out := range 2 {
			if got := resp.Y.At(out, k); math.Abs(got-want[out]) > 1e-9 {
				t.Fatalf("y%d t=%g: got %.12g, want %.12g", out, resp.T[k], got, want[out])
			}
		}
	}
}

func TestImpulseContinuousInternalDelayFeedbackExact(t *testing.T) {
	dt, tau, lag := trChainGrid(t)
	sys := trDelayFeedback(t, tau, false)
	resp, err := Impulse(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	steps := len(resp.T)
	for in := range 2 {
		x0 := []float64{sys.B.At(0, in), sys.B.At(1, in)}
		dde := newTrDDE(sys, dt/trSub, lag, (steps-1)*trSub, x0, nil)
		for k := range steps {
			want := trMulVec(sys.C, dde.x[k*trSub])
			want = trAdd(want, trMulVec(sys.LFT.D12, dde.zRight(k*trSub-lag)))
			for out := range 2 {
				if got := resp.Y.At(in*2+out, k); math.Abs(got-want[out]) > 1e-9 {
					t.Fatalf("u%d→y%d t=%g: got %.12g, want %.12g", in, out, resp.T[k], got, want[out])
				}
			}
		}
	}
}

func TestLsimContinuousInternalDelayExact(t *testing.T) {
	dt := 0.0125
	lag := 2337
	tau := float64(lag) * dt / trSub
	sys := trDelayFeedback(t, tau, true)
	inLag, outLag := 731, 350
	sys.InputDelay = []float64{0, float64(inLag) * dt / trSub}
	sys.OutputDelay = []float64{float64(outLag) * dt / trSub, 0}
	steps := 161
	tm := make([]float64, steps)
	u := mat.NewDense(steps, 2, nil)
	for k := range steps {
		tm[k] = float64(k) * dt
		u.Set(k, 0, math.Sin(3*tm[k]))
		u.Set(k, 1, 1-0.5*math.Floor(tm[k]))
	}
	x0 := []float64{1, -0.5}
	resp, err := Lsim(sys, u, tm, mat.NewVecDense(2, x0))
	if err != nil {
		t.Fatal(err)
	}
	inputAt := func(i int) []float64 {
		v := make([]float64, 2)
		for c, l := range []int{0, inLag} {
			if j := i - l; j >= 0 {
				v[c] = u.At(min(j/trSub, steps-1), c)
			}
		}
		return v
	}
	base := sys.Copy()
	base.InputDelay, base.OutputDelay = nil, nil
	dde := newTrDDE(base, dt/trSub, lag, (steps-1)*trSub, x0, inputAt)
	for k := range steps {
		for out, l := range []int{outLag, 0} {
			want := 0.0
			if i := k*trSub - l; i >= 0 {
				want = dde.y(i)[out]
			}
			if got := resp.Y.At(out, k); math.Abs(got-want) > 1e-9 {
				t.Fatalf("y%d t=%g: got %.12g, want %.12g", out, tm[k], got, want)
			}
		}
	}
}

func trFreeDelayModel(t *testing.T) *System {
	t.Helper()
	sys := trMIMO(t, 0)
	if err := sys.SetInternalDelay([]float64{0.5013}, mat.NewDense(2, 1, nil), mat.NewDense(1, 2, []float64{0.3, 1}), mat.NewDense(2, 1, []float64{0.7, -0.4}), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

// trFreeDelayOracle is C·e^{At}·x0 + D12·C2·e^{A(t−τ)}·x0 (t ≥ τ) for B2 = 0.
func trFreeDelayOracle(sys *System, x0 *mat.VecDense, out int, tm float64) float64 {
	var x mat.VecDense
	x.MulVec(trExp(sys.A, tm), x0)
	y := mat.Dot(sys.C.RowView(out), &x)
	if tau := sys.LFT.Tau[0]; tm >= tau {
		x.MulVec(trExp(sys.A, tm-tau), x0)
		y += sys.LFT.D12.At(out, 0) * mat.Dot(sys.LFT.C2.RowView(0), &x)
	}
	return y
}

func TestInitialContinuousFractionalInternalDelayExact(t *testing.T) {
	sys := trFreeDelayModel(t)
	x0 := mat.NewVecDense(2, []float64{1, -0.5})
	resp, err := Initial(sys, x0, 2)
	if err != nil {
		t.Fatal(err)
	}
	for k, tm := range resp.T {
		for out := range 2 {
			if got, want := resp.Y.At(out, k), trFreeDelayOracle(sys, x0, out, tm); math.Abs(got-want) > 1e-9 {
				t.Fatalf("y%d t=%g: got %.12g, want %.12g", out, tm, got, want)
			}
		}
	}

	steps := 147
	tm := make([]float64, steps)
	for k := range tm {
		tm[k] = float64(k) * 0.0137
	}
	lsim, err := Lsim(sys, mat.NewDense(steps, 2, nil), tm, x0)
	if err != nil {
		t.Fatal(err)
	}
	for k := range tm {
		for out := range 2 {
			if got, want := lsim.Y.At(out, k), trFreeDelayOracle(sys, x0, out, tm[k]); math.Abs(got-want) > 1e-9 {
				t.Fatalf("Lsim y%d t=%g: got %.12g, want %.12g", out, tm[k], got, want)
			}
		}
	}
}

func TestImpulseContinuousInternalDelayFedByInputErrors(t *testing.T) {
	sys := trMIMO(t, 0)
	if err := sys.SetInternalDelay([]float64{0.5013}, mat.NewDense(2, 1, nil), mat.NewDense(1, 2, []float64{0.3, 1}), mat.NewDense(2, 1, []float64{0.7, -0.4}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := Impulse(sys, 2); !errors.Is(err, ErrInternalDelayImpulse) {
		t.Fatalf("err = %v, want ErrInternalDelayImpulse", err)
	}
}

func TestTimeResponseSingularDescriptorRejectsInitialState(t *testing.T) {
	sys := trMIMO(t, 0)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 0})
	x0 := mat.NewVecDense(2, []float64{1, 1})
	tGrid := makeTimeVector(11, 0.1)
	u := mat.NewDense(11, 2, nil)
	for name, run := range map[string]func() error{
		"Initial": func() error { _, err := Initial(sys, x0, 2); return err },
		"Lsim":    func() error { _, err := Lsim(sys, u, tGrid, x0); return err },
	} {
		err := run()
		if !errors.Is(err, ErrDescriptorInitialState) || !errors.Is(err, ErrDescriptorSingular) {
			t.Errorf("%s err = %v, want ErrDescriptorInitialState", name, err)
		}
	}
	resp, err := Initial(sys, mat.NewVecDense(2, nil), 2)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Norm(resp.Y, 1) != 0 {
		t.Fatalf("zero-state Initial = %v, want zeros", mat.Formatted(resp.Y))
	}
}

func TestTimeResponseSingularDescriptorMatchesHandReduction(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, delays := range []bool{false, true} {
			desc, oracle := index1Descriptor(t, dt, delays)
			name := fmt.Sprintf("dt=%g delays=%v", dt, delays)

			got, err := Step(desc, 2)
			if err != nil {
				t.Fatalf("%s Step: %v", name, err)
			}
			want, err := Step(oracle, 2)
			if err != nil {
				t.Fatal(err)
			}
			compareTimeResponses(t, got, want, 1e-10)

			got, err = Impulse(desc, 2)
			if dt == 0 && delays {
				if !errors.Is(err, ErrInternalDelayImpulse) {
					t.Fatalf("%s Impulse err = %v, want ErrInternalDelayImpulse", name, err)
				}
			} else {
				if err != nil {
					t.Fatalf("%s Impulse: %v", name, err)
				}
				want, err := Impulse(oracle, 2)
				if err != nil {
					t.Fatal(err)
				}
				compareTimeResponses(t, got, want, 1e-10)
			}

			step := 0.05
			if dt > 0 {
				step = dt
			}
			tGrid := makeTimeVector(31, step)
			u := mat.NewDense(31, 2, nil)
			for k := range 31 {
				u.Set(k, 0, math.Sin(0.7*float64(k)))
				u.Set(k, 1, 1-0.03*float64(k))
			}
			got, err = Lsim(desc, u, tGrid, mat.NewVecDense(3, nil))
			if err != nil {
				t.Fatalf("%s Lsim: %v", name, err)
			}
			want, err = Lsim(oracle, u, tGrid, nil)
			if err != nil {
				t.Fatal(err)
			}
			compareTimeResponses(t, got, want, 1e-10)
		}
	}
}

func TestTimeResponseIndex2Descriptor(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		desc, oracle := index2Descriptor(t, dt, false)
		for name, run := range map[string]func(*System, float64) (*TimeResponse, error){"Step": Step, "Impulse": Impulse} {
			got, err := run(desc, 2)
			if err != nil {
				t.Fatalf("dt=%g %s: %v", dt, name, err)
			}
			want, err := run(oracle, 2)
			if err != nil {
				t.Fatal(err)
			}
			compareTimeResponses(t, got, want, 1e-9)
		}

		improper, _ := index2Descriptor(t, dt, true)
		for name, run := range map[string]func() error{
			"Step":    func() error { _, err := Step(improper, 2); return err },
			"Impulse": func() error { _, err := Impulse(improper, 2); return err },
			"Lsim": func() error {
				_, err := Lsim(improper, mat.NewDense(3, 2, nil), makeTimeVector(3, 0.1), nil)
				return err
			},
		} {
			if err := run(); !errors.Is(err, ErrImproperModel) {
				t.Errorf("dt=%g %s err = %v, want ErrImproperModel", dt, name, err)
			}
		}
	}
}

func TestStepContinuousInternalDelayDescriptorMatchesExplicit(t *testing.T) {
	_, tau, _ := trChainGrid(t)
	sys := trDelayFeedback(t, tau, true)
	desc := sys.Copy()
	E := mat.NewDense(2, 2, []float64{2, 0.5, -0.3, 1.5})
	var ea, eb, eb2 mat.Dense
	ea.Mul(E, sys.A)
	eb.Mul(E, sys.B)
	eb2.Mul(E, sys.LFT.B2)
	desc.E, desc.A, desc.B, desc.LFT.B2 = E, &ea, &eb, &eb2
	got, err := Step(desc, 2)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Step(sys, 2)
	if err != nil {
		t.Fatal(err)
	}
	compareTimeResponses(t, got, want, 1e-10)
}

func wantErr(t *testing.T, tag string, err, sentinel error, prefix string) {
	t.Helper()
	if !errors.Is(err, sentinel) || !strings.HasPrefix(err.Error(), prefix+": ") {
		t.Errorf("%s: err = %v, want %q prefix wrapping %v", tag, err, prefix, sentinel)
	}
}

func TestResponsesRejectModelsWithoutInputsOrOutputs(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		auto := autonomousFixture(t, dt)
		noOut, err := New(auto.A, mat.NewDense(2, 1, []float64{1, -1}), nil, nil, dt)
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"m=0": auto, "p=0": noOut} {
			tag := fmt.Sprintf("dt=%g %s", dt, name)
			_, err := sys.DCGain()
			wantErr(t, tag+" DCGain", err, ErrDimensionMismatch, "DCGain")
			_, err = Step(sys, 1)
			wantErr(t, tag+" Step", err, ErrDimensionMismatch, "Step")
			_, err = Impulse(sys, 1)
			wantErr(t, tag+" Impulse", err, ErrDimensionMismatch, "Impulse")
		}
		_, err = Initial(noOut, mat.NewVecDense(2, []float64{1, 2}), 1)
		wantErr(t, fmt.Sprintf("dt=%g Initial p=0", dt), err, ErrDimensionMismatch, "Initial")
		tg := []float64{0, 0.1, 0.2}
		_, err = Lsim(noOut, mat.NewDense(3, 1, nil), tg, nil)
		wantErr(t, fmt.Sprintf("dt=%g Lsim p=0", dt), err, ErrDimensionMismatch, "Lsim")

		resp, err := Initial(auto, mat.NewVecDense(2, []float64{1, 2}), 1)
		if err != nil || resp.Y == nil || resp.Y.IsEmpty() {
			t.Fatalf("dt=%g Initial m=0 = %v, %v; want samples", dt, resp, err)
		}
	}
}

func TestResponsesRejectInvalidFinalTime(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0.5}),
		mat.NewDense(1, 1, []float64{0.2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	x0 := mat.NewVecDense(2, []float64{1, -1})
	for _, tf := range []float64{-5, math.NaN(), math.Inf(1)} {
		_, err := Step(sys, tf)
		wantErr(t, fmt.Sprintf("Step(%g)", tf), err, ErrInvalidArgument, "Step")
		_, err = Impulse(sys, tf)
		wantErr(t, fmt.Sprintf("Impulse(%g)", tf), err, ErrInvalidArgument, "Impulse")
		_, err = Initial(sys, x0, tf)
		wantErr(t, fmt.Sprintf("Initial(%g)", tf), err, ErrInvalidArgument, "Initial")
	}
	resp, err := Step(sys, 0)
	if err != nil || len(resp.T) < 2 {
		t.Fatalf("Step(0) auto = %v, %v", resp, err)
	}
}

func TestResponsesRejectNilSystem(t *testing.T) {
	var sys *System
	_, err := Step(sys, 1)
	wantErr(t, "Step", err, ErrInvalidArgument, "Step")
	_, err = Impulse(sys, 1)
	wantErr(t, "Impulse", err, ErrInvalidArgument, "Impulse")
	_, err = Initial(sys, mat.NewVecDense(1, nil), 1)
	wantErr(t, "Initial", err, ErrInvalidArgument, "Initial")
	_, err = Lsim(sys, mat.NewDense(2, 1, nil), []float64{0, 1}, nil)
	wantErr(t, "Lsim", err, ErrInvalidArgument, "Lsim")
	_, err = sys.DCGain()
	wantErr(t, "DCGain", err, ErrInvalidArgument, "DCGain")
	_, err = Damp(sys)
	wantErr(t, "Damp", err, ErrInvalidArgument, "Damp")
	_, err = sys.Pade(2)
	wantErr(t, "Pade", err, ErrInvalidArgument, "Pade")
}

func TestLsimTimeGridErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.6, 0.3, -0.2, 0.4})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0.5})
	dsys, err := New(A, B, C, mat.NewDense(1, 1, []float64{0.1}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	u := mat.NewDense(3, 1, []float64{1, 1, 1})
	_, err = Lsim(dsys, u, []float64{0, 0.2, 0.4}, nil)
	wantErr(t, "Dt mismatch", err, ErrInvalidArgument, "Lsim")
	_, err = Lsim(dsys, u, []float64{0, 0.1, 0.3}, nil)
	wantErr(t, "non-uniform", err, ErrInvalidArgument, "Lsim")
	_, err = Lsim(dsys, u, []float64{0.2, 0.1, 0}, nil)
	wantErr(t, "decreasing", err, ErrInvalidArgument, "Lsim")
	_, err = Lsim(dsys, u, []float64{0, math.NaN(), 0.2}, nil)
	wantErr(t, "NaN", err, ErrInvalidArgument, "Lsim")
	_, err = Lsim(dsys, nil, []float64{0, 0.1, 0.2}, nil)
	wantErr(t, "nil u", err, ErrInvalidArgument, "Lsim")
	_, err = Lsim(dsys, mat.NewDense(2, 1, nil), []float64{0, 0.1, 0.2}, nil)
	wantErr(t, "u size", err, ErrDimensionMismatch, "Lsim")
}

func TestDampMatchesMATLABDefinitions(t *testing.T) {
	cont, err := New(mat.NewDense(3, 3, []float64{0, 0, 0, 0, 2, 0.3, 0, 0, -1}), mat.NewDense(3, 1, []float64{1, 1, 1}), mat.NewDense(1, 3, []float64{1, 1, 1}), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err := Damp(cont)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range info {
		p := real(d.Pole)
		var zeta, tau float64
		switch p {
		case 0:
			zeta, tau = -1, math.Inf(1)
		case 2:
			zeta, tau = -1, -0.5
		case -1:
			zeta, tau = 1, 1
		default:
			t.Fatalf("unexpected pole %v", d.Pole)
		}
		if d.Wn != math.Abs(p) || d.Zeta != zeta || d.Tau != tau {
			t.Errorf("pole %v: Wn=%g Zeta=%g Tau=%g, want %g %g %g", d.Pole, d.Wn, d.Zeta, d.Tau, math.Abs(p), zeta, tau)
		}
	}

	osc, err := New(mat.NewDense(2, 2, []float64{0, 3, -3, 0}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 0}), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err = Damp(osc)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range info {
		if math.Abs(d.Wn-3) > 1e-12 || d.Zeta != 0 || !math.IsInf(d.Tau, 1) {
			t.Errorf("±3j: %+v, want Wn 3 Zeta 0 Tau +Inf", d)
		}
	}

	dt := 0.5
	disc, err := New(mat.NewDense(2, 2, []float64{1, 0.2, 0, 1.5}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), nil, dt)
	if err != nil {
		t.Fatal(err)
	}
	info, err = Damp(disc)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range info {
		switch real(d.Pole) {
		case 1:
			if d.Wn != 0 || d.Zeta != -1 || !math.IsInf(d.Tau, 1) {
				t.Errorf("z=1: %+v, want Wn 0 Zeta -1 Tau +Inf", d)
			}
		case 1.5:
			s := math.Log(1.5) / dt
			if math.Abs(d.Wn-s) > 1e-12 || d.Zeta != -1 || math.Abs(d.Tau+1/s) > 1e-12 {
				t.Errorf("z=1.5: %+v, want Wn %g Zeta -1 Tau %g", d, s, -1/s)
			}
		}
	}

	gain, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	info, err = Damp(gain)
	if err != nil || info == nil || len(info) != 0 {
		t.Errorf("Damp(gain) = %v, %v; want empty non-nil", info, err)
	}
}
