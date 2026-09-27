package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStaircaseControllable(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 3 {
		t.Errorf("expected ncont=3, got %d", res.NCont)
	}
	checkEigenvaluesPreserved(t, A, res.A, 1e-10)
}

func TestStaircaseUncontrollable(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		0, 4, 5,
		0, 0, 6,
	})
	B := mat.NewDense(3, 1, nil)
	C := mat.NewDense(1, 3, []float64{1, 0, 0})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 0 {
		t.Errorf("expected ncont=0 for zero B, got %d", res.NCont)
	}
}

func TestStaircasePartiallyControllable(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		-2, -3, 0,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{0, 1, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 2 {
		t.Errorf("expected ncont=2, got %d", res.NCont)
	}
	checkEigenvaluesPreserved(t, A, res.A, 1e-10)
}

func TestStaircaseSingleInput(t *testing.T) {
	// Use well-separated eigenvalues for accurate comparison.
	// Companion form for (s+1)(s+2)(s+3)(s+4) = s^4+10s^3+35s^2+50s+24
	A := mat.NewDense(4, 4, []float64{
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		-24, -50, -35, -10,
	})
	B := mat.NewDense(4, 1, []float64{0, 0, 0, 1})
	C := mat.NewDense(1, 4, []float64{1, 0, 0, 0})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 4 {
		t.Errorf("expected ncont=4 for single-input controllable, got %d", res.NCont)
	}
	for i, bs := range res.BlockSizes {
		if bs != 1 {
			t.Errorf("block %d: expected size 1, got %d", i, bs)
		}
	}
	if len(res.BlockSizes) != 4 {
		t.Errorf("expected 4 blocks for single-input 4-state, got %d", len(res.BlockSizes))
	}
	checkEigenvaluesPreserved(t, A, res.A, 1e-10)
}

func TestStaircaseMultiInput(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 2, 0,
		0, 0, 3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	C := mat.NewDense(1, 3, []float64{1, 1, 1})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 3 {
		t.Errorf("expected ncont=3, got %d", res.NCont)
	}
	checkEigenvaluesPreserved(t, A, res.A, 1e-10)
}

func TestStaircaseZeroDimension(t *testing.T) {
	A := &mat.Dense{}
	B := &mat.Dense{}
	C := &mat.Dense{}

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 0 {
		t.Errorf("expected ncont=0 for n=0, got %d", res.NCont)
	}
}

func TestStaircaseZeroInputs(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := &mat.Dense{}
	C := mat.NewDense(1, 2, []float64{1, 0})

	res := ControllabilityStaircase(A, B, C, 0)
	if res.NCont != 0 {
		t.Errorf("expected ncont=0 for m=0, got %d", res.NCont)
	}
}

func TestStaircaseOrthogonalSimilarity(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})

	res := ControllabilityStaircase(A, B, C, 0)

	_, bCols := res.B.Dims()
	for i := res.NCont; i < 3; i++ {
		for j := range bCols {
			if v := math.Abs(res.B.At(i, j)); v > 1e-10 {
				t.Errorf("B[%d,%d] = %g, expected ~0 in uncontrollable rows", i, j, res.B.At(i, j))
			}
		}
	}
}

func checkEigenvaluesPreserved(t *testing.T, origA, newA *mat.Dense, tol float64) {
	t.Helper()
	n, _ := origA.Dims()
	if n == 0 {
		return
	}

	var eig1, eig2 mat.Eigen
	ok1 := eig1.Factorize(origA, mat.EigenNone)
	ok2 := eig2.Factorize(newA, mat.EigenNone)
	if !ok1 || !ok2 {
		t.Error("eigenvalue decomposition failed")
		return
	}
	ev1 := eig1.Values(nil)
	ev2 := eig2.Values(nil)

	sortComplex(ev1)
	sortComplex(ev2)

	for i := range ev1 {
		if cmplx.Abs(ev1[i]-ev2[i]) > tol {
			t.Errorf("eigenvalue %d: original=%v, transformed=%v", i, ev1[i], ev2[i])
		}
	}
}

func sortComplex(vals []complex128) {
	sort.Slice(vals, func(i, j int) bool {
		if real(vals[i]) != real(vals[j]) {
			return real(vals[i]) < real(vals[j])
		}
		return imag(vals[i]) < imag(vals[j])
	})
}
