package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/mat"
)

// riccatiResidualBig is the Riccati residual of X in 320-bit arithmetic:
//
//	continuous: AᵀX + XA + Q − (XB + S)R⁻¹(BᵀX + Sᵀ)
//	discrete:   AᵀXA − X + Q − (AᵀXB + S)(R + BᵀXB)⁻¹(BᵀXA + Sᵀ)
func riccatiResidualBig(continuous bool, A, B, Q, R, S, X *mat.Dense) *mat.Dense {
	a, b, q, r, s, x := hbFrom(A), hbFrom(B), hbFrom(Q), hbFrom(R), hbFrom(S), hbFrom(X)
	var res, m, w hbMat
	if continuous {
		res = hbAdd(hbAdd(hbMul(a.t(), x), hbMul(x, a), 1), q, 1)
		m, w = hbMul(b.t(), x), r
	} else {
		xa := hbMul(x, a)
		res = hbAdd(hbAdd(hbMul(a.t(), xa), x, -1), q, 1)
		m, w = hbMul(b.t(), xa), hbAdd(r, hbMul(hbMul(b.t(), x), b), 1)
	}
	m = hbAdd(m, s.t(), 1)
	res = hbAdd(res, hbMul(m.t(), hbSolve(w, m)), -1)
	o := mat.NewDense(res.r, res.c, nil)
	for i := range res.r {
		for j := range res.c {
			v, _ := res.at(i, j).Float64()
			o.Set(i, j, v)
		}
	}
	return o
}

// TestRiccatiResidual_DoubleDouble: riccatiResidual forms the Riccati
// residual of the Newton step of verifyRiccati to the rounding of the
// residual itself, not of its terms. X is the stabilizing solution moved
// by 1e-9 and K its gain moved by 1e-7, so the residual is about 1e-9 of
// terms that a float64 sum rounds at 1e-16; the K-dependent cost residual
// is corrected to the K-free Riccati residual. Before this a float64
// Newton step from a rounded Q_K = Q + KᵀRK measured 8.6e-9 for a Dare X
// with error 5.4e-7 (TestDare_NewtonVerified "w15 #16").
func TestRiccatiResidual_DoubleDouble(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.9, 2.4, -0.3, -1.1, 1.3, 0.8, 0.2, -0.7, 1.6})
	B := mat.NewDense(3, 2, []float64{0.5, -1.2, 1, 0.3, -0.4, 0.9})
	Q := mat.NewDense(3, 3, []float64{1e-6, 2e-7, 0, 2e-7, 3e-6, -1e-7, 0, -1e-7, 2e-6})
	R := mat.NewDense(2, 2, []float64{4e5, 3.999e5, 3.999e5, 4e5})
	S := mat.NewDense(3, 2, []float64{0.01, -0.02, 0.03, 0.01, -0.01, 0.02})
	for _, continuous := range []bool{false, true} {
		t.Run(fmt.Sprintf("continuous=%v", continuous), func(t *testing.T) {
			solve := Dare
			if continuous {
				solve = Care
			}
			sol, err := solve(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: true})
			if err != nil {
				t.Fatal(err)
			}
			n, m := 3, 2
			X := mat.DenseCopyOf(sol.X)
			for i := range n {
				for j := range n {
					X.Set(i, j, X.At(i, j)*(1+1e-9*math.Cos(float64(i+j))))
				}
			}
			problem, err := newRiccatiProblem(A, B, Q, R, &RiccatiOpts{S: S})
			if err != nil {
				t.Fatal(err)
			}
			copy(problem.ws.rChol, R.RawMatrix().Data)
			if !impl.Dpotrf(blas.Upper, m, problem.ws.rChol, m) {
				t.Fatal("R not positive definite")
			}
			K, err := problem.gain(continuous, X.RawMatrix().Data)
			if err != nil {
				t.Fatal(err)
			}
			K = mat.DenseCopyOf(K)
			for i := range m {
				for j := range n {
					K.Set(i, j, K.At(i, j)*(1+1e-7*math.Sin(float64(3*i+j+1))))
				}
			}
			aclT, res := make([]float64, n*n), make([]float64, n*n)
			problem.riccatiResidual(continuous, K.RawMatrix().Data, X.RawMatrix().Data, aclT, res)
			want := riccatiResidualBig(continuous, A, B, Q, R, S, X)
			if e := relErr(mat.NewDense(n, n, res), want); !(e <= 1e-12) {
				t.Errorf("residual rel err %g (‖Ric‖ %g)", e, mat.Norm(want, 2))
			}
		})
	}
}
