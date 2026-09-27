package controlsys

import (
	"math"
	"math/cmplx"
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
	ssr, err := tf.StateSpace(nil)
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
