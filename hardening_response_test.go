package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStep_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 2.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := range steps {
		tk := resp.T[k]
		got := resp.Y.At(0, k)
		if math.Abs(got-tk) > 0.05 {
			t.Errorf("t=%.3f: got %f, want %f (ramp)", tk, got, tk)
		}
	}

	k1 := 0
	for j := 1; j < steps; j++ {
		if math.Abs(resp.T[j]-1.0) < math.Abs(resp.T[k1]-1.0) {
			k1 = j
		}
	}
	if math.Abs(resp.Y.At(0, k1)-1.0) > 0.05 {
		t.Errorf("y(1.0) = %f, want ~1.0", resp.Y.At(0, k1))
	}

	k2 := 0
	for j := 1; j < steps; j++ {
		if math.Abs(resp.T[j]-2.0) < math.Abs(resp.T[k2]-2.0) {
			k2 = j
		}
	}
	if math.Abs(resp.Y.At(0, k2)-2.0) > 0.05 {
		t.Errorf("y(2.0) = %f, want ~2.0", resp.Y.At(0, k2))
	}
}

func TestStep_DCGain_Convergence(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 20.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	lastVal := resp.Y.At(0, steps-1)
	if math.Abs(lastVal-3.0) > 0.03 {
		t.Errorf("final value = %f, want ~3.0 (DC gain)", lastVal)
	}
}

func TestDCGain_WithFeedthrough(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g.At(0, 0)-2.5) > 1e-12 {
		t.Errorf("DC gain = %f, want 2.5", g.At(0, 0))
	}
}

func TestDCGain_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}

	want := mat.NewDense(2, 2, []float64{1, 0, 0, 0.5})
	if !matEqual(g, want, 1e-12) {
		t.Errorf("DC gain =\n%v\nwant\n%v", mat.Formatted(g), mat.Formatted(want))
	}
}

func TestDamp_ContinuousSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -4, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 2 {
		t.Fatalf("got %d poles, want 2", len(info))
	}

	for _, d := range info {
		if math.Abs(d.Wn-2.0) > 1e-10 {
			t.Errorf("Wn = %f, want 2.0", d.Wn)
		}
		if math.Abs(d.Zeta-0.5) > 1e-10 {
			t.Errorf("Zeta = %f, want 0.5", d.Zeta)
		}
	}
}

func TestDamp_DiscreteNegativePole(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 1 {
		t.Fatalf("got %d poles, want 1", len(info))
	}

	d := info[0]
	if math.IsNaN(d.Wn) || math.IsNaN(d.Zeta) {
		t.Fatalf("NaN in Damp result: Wn=%f, Zeta=%f", d.Wn, d.Zeta)
	}

	sc := cmplx.Log(complex(-0.5, 0)) / complex(0.1, 0)
	wantWn := cmplx.Abs(sc)
	if math.Abs(d.Wn-wantWn) > 1e-10 {
		t.Errorf("Wn = %f, want %f", d.Wn, wantWn)
	}
	if d.Wn > 0 {
		wantZeta := -real(sc) / wantWn
		if math.Abs(d.Zeta-wantZeta) > 1e-10 {
			t.Errorf("Zeta = %f, want %f", d.Zeta, wantZeta)
		}
	}
}

func TestDiscretize_ZOH_PreservesDCGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	dcCont, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	wantDC := dcCont.At(0, 0)
	if math.Abs(wantDC-1.5) > 1e-12 {
		t.Fatalf("continuous DC gain = %f, want 1.5", wantDC)
	}

	dsys, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	dcDisc, err := dsys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dcDisc.At(0, 0)-wantDC) > 1e-6 {
		t.Errorf("discrete ZOH DC gain = %f, want %f", dcDisc.At(0, 0), wantDC)
	}
}

func TestDiscretize_Tustin_PreservesDCGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	dcCont, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	wantDC := dcCont.At(0, 0)

	dsys, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}

	dcDisc, err := dsys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dcDisc.At(0, 0)-wantDC) > 1e-6 {
		t.Errorf("discrete Tustin DC gain = %f, want %f", dcDisc.At(0, 0), wantDC)
	}
}

func TestStep_DiscreteTime(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 10.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	lastVal := resp.Y.At(0, steps-1)

	dcGain, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	wantDC := dcGain.At(0, 0)
	if math.Abs(lastVal-wantDC) > 0.05 {
		t.Errorf("final value = %f, want ~%f (DC gain)", lastVal, wantDC)
	}
}

func TestStep_NonMinimumPhase(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, -1, 1, 0}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{-1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 10.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()

	foundUndershoot := false
	for k := range steps {
		if resp.Y.At(0, k) < -0.01 {
			foundUndershoot = true
			break
		}
	}
	if !foundUndershoot {
		t.Error("non-minimum phase system should exhibit initial undershoot (y < 0)")
	}

	lastVal := resp.Y.At(0, steps-1)
	if math.Abs(lastVal-1.0) > 0.05 {
		t.Errorf("final value = %f, want ~1.0 (DC gain)", lastVal)
	}
}
