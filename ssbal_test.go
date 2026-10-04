package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSsbal_PoorlyConditioned(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1000, 1, 0, 0.001}),
		mat.NewDense(2, 1, []float64{1000, 0.001}),
		mat.NewDense(1, 2, []float64{1, 1000}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)

	for _, w := range []float64{0.01, 0.1, 1, 10} {
		g1, _ := sys.EvalFr(complex(0, w))
		g2, _ := res.Sys.EvalFr(complex(0, w))
		diff := math.Abs(real(g1[0][0]) - real(g2[0][0]))
		relDiff := diff
		if math.Abs(real(g1[0][0])) > 1e-10 {
			relDiff = diff / math.Abs(real(g1[0][0]))
		}
		if relDiff > 1e-6 {
			t.Errorf("w=%g: freq response mismatch rel=%g", w, relDiff)
		}
	}
}

func TestSsbal_AlreadyBalanced(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		d := res.T.At(i, i)
		if math.Abs(d-1) > 1 {
			t.Errorf("T[%d,%d] = %g, expected near 1 for already balanced system", i, i, d)
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestSsbal_Empty(t *testing.T) {
	sys, err := New(nil, nil, nil, mat.NewDense(1, 1, []float64{3}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	dc, _ := res.Sys.DCGain()
	if dc.At(0, 0) != 3 {
		t.Errorf("dcgain = %g, want 3", dc.At(0, 0))
	}
}

func TestSsbal_TransformIsDiagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 100, 0,
			0.01, -2, 50,
			0, 0.02, -3,
		}),
		mat.NewDense(3, 1, []float64{100, 1, 0.01}),
		mat.NewDense(1, 3, []float64{0.01, 1, 100}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		for j := range n {
			if i != j && res.T.At(i, j) != 0 {
				t.Errorf("T[%d,%d] = %g, want 0 (T should be diagonal)", i, j, res.T.At(i, j))
			}
		}
		if res.T.At(i, i) <= 0 {
			t.Errorf("T[%d,%d] = %g, want positive", i, i, res.T.At(i, i))
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)
}

// TB01ID/MATLAB ssbal: T is diagonal and [T·A/T, T·B; C/T, 0] has row and
// column norms (diagonal excluded) within the Osborne stopping band.
func TestSsbal_BalancesRowColumnNorms(t *testing.T) {
	A0 := mat.NewDense(4, 4, []float64{
		-1.2, 0.7, -0.3, 0.5,
		0.4, -2.1, 0.9, -0.2,
		-0.6, 0.3, -0.8, 1.1,
		0.2, -0.5, 0.4, -1.7,
	})
	B0 := mat.NewDense(4, 2, []float64{1, -0.5, 0.3, 2, -1.4, 0.6, 0.8, 0.1})
	C0 := mat.NewDense(2, 4, []float64{0.7, -1.1, 0.4, 0.9, -0.3, 0.5, 1.6, -0.8})
	D := mat.NewDense(2, 2, []float64{0.2, 0, -0.1, 0.4})
	scale := []float64{1e6, 1e-3, 1e2, 1e-5}
	A, B, C := mat.DenseCopyOf(A0), mat.DenseCopyOf(B0), mat.DenseCopyOf(C0)
	for i, s := range scale {
		for j := range 4 {
			A.Set(i, j, A.At(i, j)*s)
			A.Set(j, i, A.At(j, i)/s)
		}
		for j := range 2 {
			B.Set(i, j, B.At(i, j)*s)
			C.Set(j, i, C.At(j, i)/s)
		}
	}
	for _, dt := range []float64{0, 0.1} {
		sys, err := New(A, B, C, D, dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Ssbal(sys)
		if err != nil {
			t.Fatal(err)
		}
		n, m, p := 4, 2, 2
		for i := range n {
			ti := res.T.At(i, i)
			if fr, _ := math.Frexp(ti); fr != 0.5 {
				t.Errorf("dt=%g: T[%d,%d]=%g not a power of two", dt, i, i, ti)
			}
			for j := range n {
				if want := ti * A.At(i, j) / res.T.At(j, j); res.Sys.A.At(i, j) != want {
					t.Errorf("dt=%g: Ab[%d,%d]=%g, want T·A/T = %g", dt, i, j, res.Sys.A.At(i, j), want)
				}
			}
			for j := range m {
				if want := ti * B.At(i, j); res.Sys.B.At(i, j) != want {
					t.Errorf("dt=%g: Bb[%d,%d]=%g, want T·B = %g", dt, i, j, res.Sys.B.At(i, j), want)
				}
			}
			for j := range p {
				if want := C.At(j, i) / ti; res.Sys.C.At(j, i) != want {
					t.Errorf("dt=%g: Cb[%d,%d]=%g, want C/T = %g", dt, j, i, res.Sys.C.At(j, i), want)
				}
			}
		}
		for i := range n {
			var row, col float64
			for j := range n {
				if j != i {
					row += math.Abs(res.Sys.A.At(i, j))
					col += math.Abs(res.Sys.A.At(j, i))
				}
			}
			for j := range m {
				row += math.Abs(res.Sys.B.At(i, j))
			}
			for j := range p {
				col += math.Abs(res.Sys.C.At(j, i))
			}
			if r := max(row/col, col/row); r > 2.5 {
				t.Errorf("dt=%g state %d: row norm %g, col norm %g (ratio %g)", dt, i, row, col, r)
			}
		}
		for _, w := range []float64{0.01, 0.3, 2, 9} {
			s := complex(0, w)
			if dt > 0 {
				s = cmplx.Exp(complex(0, w*dt))
			}
			g0, _ := sys.EvalFr(s)
			g1, _ := res.Sys.EvalFr(s)
			for i := range p {
				for j := range m {
					if d := cmplx.Abs(g0[i][j] - g1[i][j]); d > 1e-12*max(1, cmplx.Abs(g0[i][j])) {
						t.Errorf("dt=%g ω=%g G[%d,%d] differs by %g", dt, w, i, j, d)
					}
				}
			}
		}
	}
}
