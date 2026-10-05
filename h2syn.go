package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// H2SynResult is an H2-optimal controller, as MATLAB h2syn returns K and
// info: X and Y are the state-feedback and filter Riccati solutions and
// CLPoles the closed-loop poles.
type H2SynResult struct {
	K       *System
	X       *mat.Dense
	Y       *mat.Dense
	CLPoles []complex128
}

// H2Syn computes the H2-optimal output-feedback controller for the
// generalized plant P whose last nmeas outputs are measurements and last
// ncont inputs are controls, as MATLAB h2syn(P,nmeas,ncont); see
// https://www.mathworks.com/help/robust/ref/dynamicsystem.h2syn.html.
// D12 must have full column rank and D21 full row rank; otherwise H2Syn
// returns ErrInvalidPartition (MATLAB regularizes the plant instead). A
// nonzero D22 is handled by a loop shift: K is designed for D22 = 0 and
// returned as K0 (I + D22 K0)^-1, giving the same closed loop and H2 norm.
//
// A continuous P needs D11 = 0, since otherwise no strictly proper K gives a
// finite H2 norm; K is then strictly proper. A discrete P (Ts > 0) may have
// any D11 and gets the discrete-Riccati (current-estimator) controller,
// which has feedthrough, with X and Y the stabilizing solutions of the
// control and filter DAREs.
func H2Syn(P *System, nmeas, ncont int) (*H2SynResult, error) {
	gp, err := partitionGeneralizedPlant("H2Syn", P, nmeas, ncont)
	if err != nil {
		return nil, err
	}
	if gp.dt == 0 {
		d11Raw := gp.D11.RawMatrix()
		for i := range gp.p1 {
			for j := range gp.m1 {
				if v := d11Raw.Data[i*d11Raw.Stride+j]; math.Abs(v) > 1e-10 {
					return nil, fmt.Errorf("%s: D11[%d,%d] = %g: %w", gp.op, i, j, v, ErrNoFiniteH2Norm)
				}
			}
		}
	}
	if err := gp.requireRegularFeedthrough("D12", "D21"); err != nil {
		return nil, err
	}
	if err := gp.validateControllerChannels(); err != nil {
		return nil, err
	}
	if gp.dt > 0 {
		return h2SynDiscrete(gp)
	}
	n := gp.n
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21

	// State-feedback CARE: A'X + XA - (XB2+S1)*R1^{-1}*(B2'X+S1') + Q1 = 0
	Q1 := mulDense(mat.DenseCopyOf(C1.T()), C1)
	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	S1 := mulDense(mat.DenseCopyOf(C1.T()), D12)

	resX, err := Care(A, B2, Q1, R1, &RiccatiOpts{S: S1})
	if err != nil {
		return nil, fmt.Errorf("%s: state-feedback %w", gp.op, err)
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
		return nil, fmt.Errorf("%s: filter %w", gp.op, err)
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

// h2SynDiscrete returns the discrete H2-optimal controller u = F·x̂(k|k) +
// F0·ŵ(k|k) built on the one-step predictor x̂ (Chen & Francis, Optimal
// Sampled-Data Control Systems, §6.4): with X, F from the control DARE,
// F0 = −(R1 + B2ᵀXB2)⁻¹(B2ᵀXB1 + D12ᵀD11), Y, M from the filter DARE,
// R̃ = C2YC2ᵀ + D21D21ᵀ and L0 = (FYC2ᵀ + F0D21ᵀ)R̃⁻¹, K0 is
// (A + B2F − MC2 − B2L0C2, M + B2L0, F − L0C2, L0).
func h2SynDiscrete(gp *generalizedPlantPartition) (*H2SynResult, error) {
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D11, D12, D21 := gp.D11, gp.D12, gp.D21
	D12t, C2t := mat.DenseCopyOf(D12.T()), mat.DenseCopyOf(C2.T())

	R1 := mulDense(D12t, D12)
	resX, err := Dare(A, B2, mulDense(mat.DenseCopyOf(C1.T()), C1), R1, &RiccatiOpts{S: mulDense(mat.DenseCopyOf(C1.T()), D12)})
	if err != nil {
		return nil, fmt.Errorf("%s: state-feedback %w", gp.op, err)
	}
	X := resX.X
	F := mat.DenseCopyOf(resX.K)
	F.Scale(-1, F)

	B2tX := mulDense(mat.DenseCopyOf(B2.T()), X)
	Rx := mulDense(B2tX, B2)
	Rx.Add(Rx, R1)
	F0rhs := mulDense(B2tX, B1)
	F0rhs.Add(F0rhs, mulDense(D12t, D11))
	Rxinv, err := invertSmall(Rx, gp.m2)
	if err != nil {
		return nil, fmt.Errorf("%s: R1 + B2ᵀXB2: %w", gp.op, err)
	}
	F0 := mulDense(Rxinv, F0rhs)
	F0.Scale(-1, F0)

	B1D21t := mulDense(B1, mat.DenseCopyOf(D21.T()))
	resY, err := Dare(mat.DenseCopyOf(A.T()), C2t, mulDense(B1, mat.DenseCopyOf(B1.T())), mulDense(D21, mat.DenseCopyOf(D21.T())), &RiccatiOpts{S: B1D21t})
	if err != nil {
		return nil, fmt.Errorf("%s: filter %w", gp.op, err)
	}
	Y := resY.X
	M := mat.DenseCopyOf(resY.K.T())

	Rt := mulDense(mulDense(C2, Y), C2t)
	Rt.Add(Rt, mulDense(D21, mat.DenseCopyOf(D21.T())))
	Rtinv, err := invertSmall(Rt, gp.p2)
	if err != nil {
		return nil, fmt.Errorf("%s: C2YC2ᵀ + D21D21ᵀ: %w", gp.op, err)
	}
	L0 := mulDense(mulDense(F, Y), C2t)
	L0.Add(L0, mulDense(F0, mat.DenseCopyOf(D21.T())))
	L0 = mulDense(L0, Rtinv)

	B2L0 := mulDense(B2, L0)
	Ak := mulDense(B2, F)
	Ak.Add(A, Ak)
	Ak.Sub(Ak, mulDense(M, C2))
	Ak.Sub(Ak, mulDense(B2L0, C2))
	Bk := mat.DenseCopyOf(M)
	Bk.Add(Bk, B2L0)
	Ck := mulDense(L0, C2)
	Ck.Sub(F, Ck)

	K, err := gp.newController(Ak, Bk, Ck, L0)
	if err != nil {
		return nil, err
	}
	clPoles, err := gp.closedLoopPoles(Ak, Bk, Ck, L0)
	if err != nil {
		return nil, err
	}
	return &H2SynResult{K: K, X: X, Y: Y, CLPoles: clPoles}, nil
}
