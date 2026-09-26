package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type H2SynResult struct {
	K       *System
	X       *mat.Dense
	Y       *mat.Dense
	CLPoles []complex128
}

// H2Syn computes the H2-optimal output-feedback controller for the
// continuous generalized plant P whose last nmeas outputs are measurements
// and last ncont inputs are controls. D11 must be zero. A nonzero D22 is
// handled by a loop shift: K is designed for D22 = 0 and returned as
// K0 (I + D22 K0)^-1, giving the same closed loop and H2 norm.
func H2Syn(P *System, nmeas, ncont int) (*H2SynResult, error) {
	gp, err := partitionGeneralizedPlant(P, nmeas, ncont)
	if err != nil {
		return nil, err
	}
	n := gp.n
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D11, D12, D21 := gp.D11, gp.D12, gp.D21

	tol := 1e-10
	d11Raw := D11.RawMatrix()
	for i := range gp.p1 {
		for j := range gp.m1 {
			if math.Abs(d11Raw.Data[i*d11Raw.Stride+j]) > tol {
				return nil, ErrNoFiniteH2Norm
			}
		}
	}

	if err := gp.validateControllerChannels(); err != nil {
		return nil, err
	}

	// State-feedback CARE: A'X + XA - (XB2+S1)*R1^{-1}*(B2'X+S1') + Q1 = 0
	Q1 := mulDense(mat.DenseCopyOf(C1.T()), C1)
	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	S1 := mulDense(mat.DenseCopyOf(C1.T()), D12)

	resX, err := Care(A, B2, Q1, R1, &RiccatiOpts{S: S1})
	if err != nil {
		return nil, err
	}
	X := resX.X

	// F = -K_care (state feedback gain u = F*x)
	F := mat.NewDense(gp.m2, n, nil)
	F.Scale(-1, resX.K)

	// Filter CARE (dual): Care(A', C2', B1*B1', D21*D21', S=B1*D21')
	Q2 := mulDense(B1, mat.DenseCopyOf(B1.T()))
	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	resY, err := Care(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C2.T()), Q2, R2, &RiccatiOpts{S: S2})
	if err != nil {
		return nil, err
	}
	Y := resY.X

	// L = K_dual' (observer gain, n×p2) — matches Lqe convention
	L := mat.DenseCopyOf(resY.K.T())

	var Ak, Bk, Ck *mat.Dense

	// Ak = A + B2*F - L*C2
	BF := mulDense(B2, F)
	LC := mulDense(L, C2)
	Ak = mat.NewDense(n, n, nil)
	Ak.Add(A, BF)
	Ak.Sub(Ak, LC)

	Bk = denseCopy(L)
	Ck = denseCopy(F)

	K, err := gp.newController(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	clPoles, err := gp.closedLoopPoles(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	return &H2SynResult{K: K, X: X, Y: Y, CLPoles: clPoles}, nil
}
