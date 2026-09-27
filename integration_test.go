package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSSToTFToSSRoundtrip(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			-1, 0, 0,
			0, -2, 0,
			0, 0, -3,
		},
		[]float64{
			1, 0,
			0, 1,
			1, 1,
		},
		[]float64{
			1, 0, 1,
			0, 1, 1,
		},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	ssRes, err := tfRes.TF.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	tfRes2, err := ssRes.Sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := make([]complex128, 100)
	for i := range freqs {
		w := 0.01 + float64(i)*0.2
		freqs[i] = complex(0, w)
	}

	for _, s := range freqs {
		m1 := tfRes.TF.Eval(s)
		m2 := tfRes2.TF.Eval(s)
		for i := range 2 {
			for j := range 2 {
				if cmplx.Abs(m1[i][j]-m2[i][j]) > 1e-6 {
					t.Errorf("at s=%v [%d][%d]: first=%v roundtrip=%v", s, i, j, m1[i][j], m2[i][j])
				}
			}
		}
	}
}

func TestDiscretizeAndSimulate(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 1,
		[]float64{-1},
		[]float64{1},
		[]float64{1},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.01
	dsys, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	nSteps := 500
	u := mat.NewDense(1, nSteps, nil)
	for i := range nSteps {
		u.Set(0, i, 1.0)
	}

	resp, err := dsys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for i := range nSteps {
		tVal := float64(i) * dt
		analytical := 1.0 - math.Exp(-tVal)
		got := resp.Y.At(0, i)
		if math.Abs(got-analytical) > 0.02 {
			t.Errorf("t=%.2f: got=%g want≈%g", tVal, got, analytical)
		}
	}
}

func TestReduceThenTransferFunction(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0, 1, 0, 0,
		-2, -3, 0, 0,
		0, 0, -5, 0,
		0, 0, 0, -7,
	})
	B := mat.NewDense(4, 1, []float64{0, 1, 0, 1})
	C := mat.NewDense(1, 4, []float64{1, 0, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	directTF, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}
	reducedTF, err := reduced.Sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{0.1i, 1i, 5i, 10i, complex(0.5, 1)}
	for _, s := range freqs {
		v1 := directTF.TF.Eval(s)[0][0]
		v2 := reducedTF.TF.Eval(s)[0][0]
		if cmplx.Abs(v1-v2) > 1e-6 {
			t.Errorf("at s=%v: direct=%v reduced=%v", s, v1, v2)
		}
	}
}

func TestDiscretizeTustinRoundtrip(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := sys.Discretize(0.1)
	if err != nil {
		t.Fatal(err)
	}
	csys, err := dsys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{0.1i, 1i, 5i}
	for _, s := range freqs {
		v1 := evalSS(sys, s)
		v2 := evalSS(csys, s)
		if cmplx.Abs(v1-v2) > 1e-6 {
			t.Errorf("at s=%v: orig=%v roundtrip=%v", s, v1, v2)
		}
	}
}

func TestMIMOTransferFunctionFreqResponse(t *testing.T) {
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

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := make([]complex128, 50)
	for i := range freqs {
		w := 0.01 + float64(i)*0.5
		freqs[i] = complex(0, w)
	}

	for _, s := range freqs {
		tfMat := tfRes.TF.Eval(s)
		for i := range 2 {
			for j := range 2 {
				ssVal := evalSSij(sys, s, i, j)
				if cmplx.Abs(tfMat[i][j]-ssVal) > 1e-8 {
					t.Errorf("at s=%v [%d][%d]: TF=%v SS=%v", s, i, j, tfMat[i][j], ssVal)
				}
			}
		}
	}
}
