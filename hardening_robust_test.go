package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestDiskMargin_FifthOrder(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
		0, -100, -200.18, -100.36, -2.18,
	})
	B := mat.NewDense(5, 1, []float64{0, 0, 0, 0, 1})
	C := mat.NewDense(1, 5, []float64{93.75, 50, 6.25, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.Alpha <= 0 {
		t.Errorf("Alpha = %v, want > 0", dm.Alpha)
	}
	if dm.PeakSensitivity <= 1 {
		t.Errorf("PeakSensitivity = %v, want > 1", dm.PeakSensitivity)
	}
	if dm.GainMargin[0] >= 1 || dm.GainMargin[1] <= 1 {
		t.Errorf("GainMargin = %v, want [<1, >1]", dm.GainMargin)
	}
	if dm.PhaseMargin <= 0 {
		t.Errorf("PhaseMargin = %v, want > 0", dm.PhaseMargin)
	}

	poles, _ := sys.Poles()
	if len(poles) != 5 {
		t.Errorf("got %d poles, want 5", len(poles))
	}
	hasIntegrator := false
	for _, p := range poles {
		if cmplx.Abs(p) < 0.01 {
			hasIntegrator = true
		}
	}
	if !hasIntegrator {
		t.Error("expected an integrator pole near s=0")
	}
}

func TestMargin_Type2System(t *testing.T) {
	G1 := buildSS(t, 1, 1, []float64{-10}, []float64{1}, []float64{10}, []float64{0})
	integrator := buildSS(t, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0})

	plant, err := Series(integrator, G1)
	if err != nil {
		t.Fatal(err)
	}

	Kp := 16.5
	Ki := 16.5 * 5.0
	piCtrl := buildSS(t, 1, 1, []float64{0}, []float64{1}, []float64{Ki}, []float64{Kp})

	L2, err := Series(plant, piCtrl)
	if err != nil {
		t.Fatal(err)
	}

	unity, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	cl2, err := Feedback(L2, unity, -1)
	if err != nil {
		t.Fatal(err)
	}
	stable2, err := cl2.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable2 {
		t.Fatal("type 2 closed-loop should be stable")
	}

	mr2, err := Margin(L2)
	if err != nil {
		t.Fatal(err)
	}
	if mr2.PhaseMargin <= 0 || math.IsInf(mr2.PhaseMargin, 0) {
		t.Errorf("Type2 PM = %v, want finite > 0", mr2.PhaseMargin)
	}
	t.Logf("Type2: GM=%.1f dB, PM=%.1f deg", mr2.GainMargin, mr2.PhaseMargin)
}

func TestMargin_Type2VsType1(t *testing.T) {
	G1 := buildSS(t, 1, 1, []float64{-10}, []float64{1}, []float64{10}, []float64{0})
	integrator := buildSS(t, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0})

	plant, err := Series(integrator, G1)
	if err != nil {
		t.Fatal(err)
	}

	Kp := 16.5
	Ki := 16.5 * 5.0
	piCtrl := buildSS(t, 1, 1, []float64{0}, []float64{1}, []float64{Ki}, []float64{Kp})

	L2, err := Series(plant, piCtrl)
	if err != nil {
		t.Fatal(err)
	}

	unity, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	cl2, err := Feedback(L2, unity, -1)
	if err != nil {
		t.Fatal(err)
	}
	stable2, err := cl2.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable2 {
		t.Fatal("type 2 closed-loop should be stable")
	}

	mr2, err := Margin(L2)
	if err != nil {
		t.Fatal(err)
	}
	if mr2.PhaseMargin <= 0 || math.IsInf(mr2.PhaseMargin, 0) {
		t.Errorf("Type2 PM = %v, want finite > 0", mr2.PhaseMargin)
	}

	pCtrl, _ := NewGain(mat.NewDense(1, 1, []float64{Kp}), 0)
	L1, err := Series(plant, pCtrl)
	if err != nil {
		t.Fatal(err)
	}
	mr1, err := Margin(L1)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Type1: PM=%.1f deg; Type2: PM=%.1f deg", mr1.PhaseMargin, mr2.PhaseMargin)

	if !math.IsInf(mr1.PhaseMargin, 0) && !math.IsInf(mr2.PhaseMargin, 0) {
		if mr2.PhaseMargin >= mr1.PhaseMargin {
			t.Errorf("Type2 PM=%.1f should be < Type1 PM=%.1f (more aggressive)", mr2.PhaseMargin, mr1.PhaseMargin)
		}
	}
}

func buildSS(t *testing.T, n, m int, a, b, c, d []float64) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(n, n, a),
		mat.NewDense(n, m, b),
		mat.NewDense(m, n, c),
		mat.NewDense(m, m, d),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestDamp_SecondOrder_SpringMassDamper(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -0.16, -0.24})
	B := mat.NewDense(2, 1, []float64{0, 0.004})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
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

	wnExpect := 0.4
	zetaExpect := 0.3

	for i, d := range info {
		if math.Abs(d.Wn-wnExpect) > 0.01 {
			t.Errorf("pole[%d] Wn=%.4f, want ~%.4f", i, d.Wn, wnExpect)
		}
		if math.Abs(d.Zeta-zetaExpect) > 0.01 {
			t.Errorf("pole[%d] Zeta=%.4f, want ~%.4f", i, d.Zeta, zetaExpect)
		}
	}
}

func TestCtrb_3State_PhysicalParams(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		1, -1, 1,
		1, -0.16, -0.24,
		1, 1, 1,
	})
	B := mat.NewDense(3, 1, []float64{0, 0.004, 1})
	C := mat.NewDense(1, 3, []float64{1, 1, 0})

	cm, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	r, c := cm.Dims()
	if r != 3 || c != 3 {
		t.Fatalf("Ctrb dims = (%d,%d), want (3,3)", r, c)
	}

	ctrbRank := svdRank(t, cm)
	if ctrbRank != 3 {
		t.Errorf("Ctrb rank = %d, want 3 (fully controllable)", ctrbRank)
	}

	om, err := Obsv(A, C)
	if err != nil {
		t.Fatal(err)
	}
	r2, c2 := om.Dims()
	if r2 != 3 || c2 != 3 {
		t.Fatalf("Obsv dims = (%d,%d), want (3,3)", r2, c2)
	}

	obsvRank := svdRank(t, om)
	if obsvRank != 3 {
		t.Errorf("Obsv rank = %d, want 3 (fully observable)", obsvRank)
	}
}

func svdRank(t *testing.T, m *mat.Dense) int {
	t.Helper()
	var s mat.SVD
	if !s.Factorize(m, mat.SVDNone) {
		t.Fatal("SVD factorization failed")
	}
	sv := s.Values(nil)
	rank := 0
	for _, v := range sv {
		if v > 1e-10 {
			rank++
		}
	}
	return rank
}

func TestSeries_TF_SS_Roundtrip(t *testing.T) {
	sys1, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -4, -1}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	sys2, err := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-4.5}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	ser, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := ser.Dims()
	if n != 3 {
		t.Errorf("Series states = %d, want 3", n)
	}

	dc, err := ser.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	dcVal := dc.At(0, 0)
	if math.Abs(dcVal-0.025) > 0.001 {
		t.Errorf("DC gain = %v, want 0.025", dcVal)
	}

	poles, err := ser.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 3 {
		t.Fatalf("got %d poles, want 3", len(poles))
	}

	wantPoles := []complex128{
		complex(-0.5, math.Sqrt(15.0)/2.0),
		complex(-0.5, -math.Sqrt(15.0)/2.0),
		complex(-5, 0),
	}
	for _, wp := range wantPoles {
		found := false
		for _, gp := range poles {
			if cmplx.Abs(gp-wp) < 0.01 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole near %v not found in %v", wp, poles)
		}
	}
}

func TestParallel_SS_Verify(t *testing.T) {
	sys1, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -4, -1}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	sys2, err := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-4.5}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	par, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := par.Dims()
	if n != 3 {
		t.Errorf("Parallel states = %d, want 3", n)
	}

	dc, err := par.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	dcVal := dc.At(0, 0)
	if math.Abs(dcVal-0.35) > 0.01 {
		t.Errorf("DC gain = %v, want 0.35", dcVal)
	}
}

func TestBandwidth_FirstOrder_Robust(t *testing.T) {
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

	bw3, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(bw3-1.0) > 0.05 {
		t.Errorf("3dB bandwidth = %v, want ~1.0", bw3)
	}

	bw6, err := Bandwidth(sys, -6)
	if err != nil {
		t.Fatal(err)
	}
	if bw6 <= bw3 {
		t.Errorf("6dB bandwidth=%v should exceed 3dB bandwidth=%v", bw6, bw3)
	}
}

func TestBandwidth_SecondOrder_Butterworth(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -100, -14.14}),
		mat.NewDense(2, 1, []float64{0, 100}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(bw-10.0)/10.0 > 0.2 {
		t.Errorf("Bandwidth = %v, want ~10.0 (within 20%%)", bw)
	}
}
