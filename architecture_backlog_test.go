package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func architectureTestSystem(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{0, 1, -2, -3},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 1, 1},
		[]float64{0.1, 0.2, -0.1, 0.3},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestEquivalentDelayFormsAgreeThroughPublicOperations(t *testing.T) {
	base := architectureTestSystem(t, 0)

	split := base.Copy()
	if err := split.SetInputDelay([]float64{0.2, 0.4}); err != nil {
		t.Fatal(err)
	}
	if err := split.SetOutputDelay([]float64{0.1, 0.1}); err != nil {
		t.Fatal(err)
	}
	if err := split.SetDelay(mat.NewDense(2, 2, []float64{
		0.0, 0.1,
		0.2, 0.0,
	})); err != nil {
		t.Fatal(err)
	}

	total := mustTotalDelay(t, split)
	collapsed := base.Copy()
	if err := collapsed.SetDelay(total); err != nil {
		t.Fatal(err)
	}

	assertDenseApprox(t, mustTotalDelay(t, split), mustTotalDelay(t, collapsed), 0)

	omega := []float64{0.2, 1.7, 6.0}
	assertFreqResponseApprox(t, split, collapsed, omega, 1e-10)

	right, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	seriesSplit, err := Series(split, right)
	if err != nil {
		t.Fatal(err)
	}
	seriesCollapsed, err := Series(collapsed, right)
	if err != nil {
		t.Fatal(err)
	}
	assertFreqResponseApprox(t, seriesSplit, seriesCollapsed, omega, 1e-10)
}

func TestFrequencyResponseEntryPointsAgreeForDelayedDiscreteMIMO(t *testing.T) {
	sys := architectureTestSystem(t, 0.1)
	if err := sys.SetInputDelay([]float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{0, 2}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, []float64{0, 1, 2, 0})); err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.2, 1.1, 4.0}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	frdResp := frd.FreqResponse()

	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*sys.Dt))
		eval, err := sys.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < resp.P; i++ {
			for j := 0; j < resp.M; j++ {
				assertComplexApprox(t, eval[i][j], resp.At(k, i, j), 1e-10)
				assertComplexApprox(t, frd.At(k, i, j), resp.At(k, i, j), 1e-10)
				assertComplexApprox(t, frdResp.At(k, i, j), resp.At(k, i, j), 1e-10)
			}
		}
	}
}

func TestTransferFunctionPreservesRowRealizationBehavior(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{
			0, 1,
			1, 0,
			1, 1,
		},
		[]float64{
			1, 0, 1,
			0, 1, 1,
		},
		[]float64{
			0.2, -0.1,
			0.3, 0.4,
		},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 0.4})); err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"u1", "u2"}
	sys.OutputName = []string{"y1", "y2"}

	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder <= 0 {
		t.Fatalf("MinimalOrder = %d, want positive", res.MinimalOrder)
	}
	if len(res.RowDegrees) != 2 {
		t.Fatalf("len(RowDegrees) = %d, want 2", len(res.RowDegrees))
	}
	for i, degree := range res.RowDegrees {
		if degree <= 0 {
			t.Fatalf("RowDegrees[%d] = %d, want positive", i, degree)
		}
	}
	if !stringSlicesEqual(res.TF.InputName, sys.InputName) {
		t.Fatalf("InputName = %v, want %v", res.TF.InputName, sys.InputName)
	}
	if !stringSlicesEqual(res.TF.OutputName, sys.OutputName) {
		t.Fatalf("OutputName = %v, want %v", res.TF.OutputName, sys.OutputName)
	}
	for i := range 2 {
		for j := range 2 {
			if res.TF.Delay[i][j] != sys.Delay.At(i, j) {
				t.Fatalf("Delay[%d][%d] = %v, want %v", i, j, res.TF.Delay[i][j], sys.Delay.At(i, j))
			}
		}
	}

	omega := []float64{0.3, 1.4, 5.0}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		tfEval := mustEval(t, res.TF, complex(0, w))
		for i := range 2 {
			for j := range 2 {
				assertComplexApprox(t, tfEval[i][j], resp.At(k, i, j), 1e-8)
			}
		}
	}
}

func TestDirectFeedthroughLoopBehaviorAcrossPublicInterconnections(t *testing.T) {
	t.Run("singular", func(t *testing.T) {
		plant, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		controller, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Feedback(plant, controller, 1); !errors.Is(err, ErrSingularTransform) {
			t.Fatalf("Feedback err = %v, want ErrSingularTransform", err)
		}

		lftPlant, err := NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := LFT(lftPlant, controller); !errors.Is(err, ErrAlgebraicLoop) {
			t.Fatalf("LFT err = %v, want ErrAlgebraicLoop", err)
		}
	})

	t.Run("nonsingular", func(t *testing.T) {
		plant, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
		if err != nil {
			t.Fatal(err)
		}
		controller, err := NewGain(mat.NewDense(1, 1, []float64{0.25}), 0)
		if err != nil {
			t.Fatal(err)
		}
		closed, err := Feedback(plant, controller, -1)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := closed.D.At(0, 0), 4.0/3.0; math.Abs(got-want) > 1e-12 {
			t.Fatalf("Feedback D = %v, want %v", got, want)
		}

		lftPlant, err := NewGain(mat.NewDense(2, 2, []float64{2, 2, 1, 0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		lftClosed, err := LFT(lftPlant, controller)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := lftClosed.D.At(0, 0), 2.5; math.Abs(got-want) > 1e-12 {
			t.Fatalf("LFT D = %v, want %v", got, want)
		}
	})
}

func TestStabilityBoundaryClassificationAcrossPublicConsumers(t *testing.T) {
	continuous, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := continuous.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if stable {
		t.Fatal("continuous boundary pole reported stable")
	}
	sep, err := Stabsep(continuous)
	if err != nil {
		t.Fatal(err)
	}
	ns, _, _ := sep.Stable.Dims()
	nu, _, _ := sep.Unstable.Dims()
	if ns != 0 || nu != 1 {
		t.Fatalf("continuous Stabsep orders = stable %d unstable %d, want 0 and 1", ns, nu)
	}
	nyq, err := continuous.Nyquist([]float64{0.1, 1, 10}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if nyq.RHPPoles != 0 {
		t.Fatalf("continuous boundary pole counted as RHP: got %d", nyq.RHPPoles)
	}

	discrete, err := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, nil),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	stable, err = discrete.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if stable {
		t.Fatal("discrete unit-circle pole reported stable")
	}
	sep, err = Stabsep(discrete)
	if err != nil {
		t.Fatal(err)
	}
	ns, _, _ = sep.Stable.Dims()
	nu, _, _ = sep.Unstable.Dims()
	if ns != 0 || nu != 1 {
		t.Fatalf("discrete Stabsep orders = stable %d unstable %d, want 0 and 1", ns, nu)
	}
}

func assertFreqResponseApprox(t *testing.T, a, b *System, omega []float64, tol float64) {
	t.Helper()
	ra, err := a.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	rb, err := b.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	if ra.NFreq != rb.NFreq || ra.P != rb.P || ra.M != rb.M {
		t.Fatalf("response dims = (%d,%d,%d), want (%d,%d,%d)", ra.NFreq, ra.P, ra.M, rb.NFreq, rb.P, rb.M)
	}
	for k := 0; k < ra.NFreq; k++ {
		for i := 0; i < ra.P; i++ {
			for j := 0; j < ra.M; j++ {
				assertComplexApprox(t, ra.At(k, i, j), rb.At(k, i, j), tol)
			}
		}
	}
}

func assertDenseApprox(t *testing.T, got, want *mat.Dense, tol float64) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("dims = %dx%d, want %dx%d", gr, gc, wr, wc)
	}
	for i := range gr {
		for j := range gc {
			if math.Abs(got.At(i, j)-want.At(i, j)) > tol {
				t.Fatalf("(%d,%d) = %v, want %v", i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

func assertComplexApprox(t *testing.T, got, want complex128, tol float64) {
	t.Helper()
	if cmplx.Abs(got-want) > tol {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
