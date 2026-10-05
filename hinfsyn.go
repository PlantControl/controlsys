package controlsys

import (
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type HinfSynResult struct {
	K *System
	// GammaOpt is the gamma K is built for: ||T_zw||inf < GammaOpt, within
	// hinfControllerBackoff (relative) of the smallest achievable gamma.
	GammaOpt float64
	X        *mat.Dense
	Y        *mat.Dense
	CLPoles  []complex128
}

// HinfSyn computes a suboptimal H-infinity output-feedback controller for the
// continuous generalized plant P whose last nmeas outputs are measurements
// and last ncont inputs are controls, bisecting to the smallest achievable
// gamma. A nonzero D11 uses the Glover-Doyle general formulas, whose central
// controller may have feedthrough. A nonzero D22 is handled by a loop shift:
// K is designed for D22 = 0 and returned as K0 (I + D22 K0)^-1, giving the
// same closed loop and gamma.
func HinfSyn(P *System, nmeas, ncont int) (*HinfSynResult, error) {
	gp, err := partitionGeneralizedPlant(P, nmeas, ncont)
	if err != nil {
		return nil, err
	}
	if err := gp.validateControllerChannels(); err != nil {
		return nil, err
	}
	if !allZeroDense(gp.D11) {
		return hinfSynGeneral(gp)
	}
	gamma, err := hinfControllerGamma(0, func(g float64) bool { return hinfFeasible(gp, g) })
	if err != nil {
		return nil, err
	}
	return hinfSynD11Zero(gp, gamma)
}

// hinfGammaFloor ends bisection when the optimum is zero, where the
// relative stopping rule never triggers.
const hinfGammaFloor = 1e-12

// hinfControllerBackoff is the relative gamma margin above the bisection
// edge. At the edge X or Y grows without bound, so the central controller is
// ill-conditioned and overshoots gamma; 1e-4 restores a genuine margin.
const hinfControllerBackoff = 1e-4

// hinfControllerGamma returns the gamma to build the central controller at.
func hinfControllerGamma(gammaLB float64, feasible func(float64) bool) (float64, error) {
	gamma, err := hinfBisect(gammaLB, feasible)
	if err != nil {
		return 0, err
	}
	return gamma * (1 + hinfControllerBackoff), nil
}

// hinfBisect returns the smallest gamma above gammaLB, to relative 1e-6 or
// below hinfGammaFloor, that feasible accepts.
func hinfBisect(gammaLB float64, feasible func(float64) bool) (float64, error) {
	gammaUB := gammaLB*2 + 1
	for !feasible(gammaUB) {
		gammaUB *= 2
		if gammaUB > 1e12 {
			return 0, ErrGammaNotAchievable
		}
	}
	for gammaUB-gammaLB > 1e-6*gammaUB && gammaUB > hinfGammaFloor {
		mid := (gammaLB + gammaUB) / 2
		if feasible(mid) {
			gammaUB = mid
		} else {
			gammaLB = mid
		}
	}
	return gammaUB, nil
}

func hinfSynD11Zero(gp *generalizedPlantPartition, gamma float64) (*HinfSynResult, error) {
	n := gp.n
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	X, Y, err := hinfSolveRiccatis(gp, gamma)
	if err != nil {
		return nil, err
	}

	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	R1inv, err := invertSmall(R1, gp.m2)
	if err != nil {
		return nil, err
	}
	S1 := mulDense(mat.DenseCopyOf(D12.T()), C1)

	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	R2inv, err := invertSmall(R2, gp.p2)
	if err != nil {
		return nil, err
	}
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	// F = -R1inv * (B2'*X + S1)
	B2tX := mulDense(mat.DenseCopyOf(B2.T()), X)
	B2tX.Add(B2tX, S1)
	F := mulDense(R1inv, B2tX)
	F.Scale(-1, F)

	// L = -(Y*C2' + S2) * R2inv
	YC2t := mulDense(Y, mat.DenseCopyOf(C2.T()))
	YC2t.Add(YC2t, S2)
	L := mulDense(YC2t, R2inv)
	L.Scale(-1, L)

	ginv2 := 1.0 / (gamma * gamma)

	YX := mulDense(Y, X)
	nn := n
	eye := mat.NewDense(nn, nn, nil)
	for i := range nn {
		eye.Set(i, i, 1)
	}
	ZpArg := mat.NewDense(nn, nn, nil)
	ZpArg.Scale(ginv2, YX)
	ZpArg.Sub(eye, ZpArg)

	var lu mat.LU
	lu.Factorize(ZpArg)
	Zp := mat.NewDense(nn, nn, nil)
	if err := lu.SolveTo(Zp, false, eye); err != nil {
		return nil, ErrGammaNotAchievable
	}

	// Ak = A + ginv2*B1*B1'*X + B2*F + Zp*L*(C2 + ginv2*D21*B1'*X); the D21 term feeds the worst-case disturbance into the observer.
	Ak := denseCopy(A)

	tmp1 := mulDense(B1, mat.DenseCopyOf(B1.T()))
	tmp1.Mul(tmp1, X)
	tmp1.Scale(ginv2, tmp1)
	Ak.Add(Ak, tmp1)

	Ak.Add(Ak, mulDense(B2, F))

	ZpL := mulDense(Zp, L)
	C2w := C2
	if !allZeroDense(S2) {
		C2w = mulDense(mat.DenseCopyOf(S2.T()), X)
		C2w.Scale(ginv2, C2w)
		C2w.Add(C2w, C2)
	}
	Ak.Add(Ak, mulDense(ZpL, C2w))

	Bk := mulDense(Zp, L)
	Bk.Scale(-1, Bk)
	Ck := denseCopy(F)

	K, err := gp.newController(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	clPoles, err := gp.closedLoopPoles(Ak, Bk, Ck, nil)
	if err != nil {
		return nil, err
	}

	return &HinfSynResult{K: K, GammaOpt: gamma, X: X, Y: Y, CLPoles: clPoles}, nil
}

func hinfFeasible(gp *generalizedPlantPartition, gamma float64) bool {
	_, _, err := hinfSolveRiccatis(gp, gamma)
	return err == nil
}

func hinfSolveRiccatis(gp *generalizedPlantPartition, gamma float64) (*mat.Dense, *mat.Dense, error) {
	A := gp.A
	B1, B2 := gp.B1, gp.B2
	C1, C2 := gp.C1, gp.C2
	D12, D21 := gp.D12, gp.D21
	n := gp.n
	ginv2 := 1.0 / (gamma * gamma)

	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	R1inv, err := invertSmall(R1, gp.m2)
	if err != nil {
		return nil, nil, ErrInvalidPartition
	}
	S1 := mulDense(mat.DenseCopyOf(D12.T()), C1)

	B1B1t := mulDense(B1, mat.DenseCopyOf(B1.T()))
	C1tC1 := mulDense(mat.DenseCopyOf(C1.T()), C1)

	// X-Riccati: Ahat = A - B2*R1inv*S1
	B2R1inv := mulDense(B2, R1inv)
	Ahat := mat.NewDense(n, n, nil)
	Ahat.Sub(A, mulDense(B2R1inv, S1))

	// Qhat = C1'C1 - S1'*R1inv*S1
	Qhat := mat.NewDense(n, n, nil)
	Qhat.Sub(C1tC1, mulDense(mat.DenseCopyOf(S1.T()), mulDense(R1inv, S1)))

	// Gx = ginv2*B1*B1' - B2*R1inv*B2'
	Gx := mat.NewDense(n, n, nil)
	Gx.Scale(ginv2, B1B1t)
	Gx.Sub(Gx, mulDense(B2R1inv, mat.DenseCopyOf(B2.T())))

	Hx := mat.NewDense(2*n, 2*n, nil)
	setBlock(Hx, 0, 0, Ahat)
	setBlock(Hx, 0, n, Gx)
	negQhat := mat.NewDense(n, n, nil)
	negQhat.Scale(-1, Qhat)
	setBlock(Hx, n, 0, negQhat)
	negAhatT := mat.NewDense(n, n, nil)
	negAhatT.Scale(-1, mat.DenseCopyOf(Ahat.T()))
	setBlock(Hx, n, n, negAhatT)

	X, err := solveHamiltonianRiccati(Hx, n)
	if err != nil {
		return nil, nil, err
	}

	// Y-Riccati
	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	R2inv, err := invertSmall(R2, gp.p2)
	if err != nil {
		return nil, nil, ErrInvalidPartition
	}
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	// Atilde = A - S2*R2inv*C2
	S2R2inv := mulDense(S2, R2inv)
	Atilde := mat.NewDense(n, n, nil)
	Atilde.Sub(A, mulDense(S2R2inv, C2))

	// Qy = B1*B1' - S2*R2inv*S2'
	Qy := mat.NewDense(n, n, nil)
	Qy.Sub(B1B1t, mulDense(S2R2inv, mat.DenseCopyOf(S2.T())))

	// Gy = ginv2*C1'C1 - C2'*R2inv*C2
	Gy := mat.NewDense(n, n, nil)
	Gy.Scale(ginv2, C1tC1)
	Gy.Sub(Gy, mulDense(mat.DenseCopyOf(C2.T()), mulDense(R2inv, C2)))

	Hy := mat.NewDense(2*n, 2*n, nil)
	setBlock(Hy, 0, 0, mat.DenseCopyOf(Atilde.T()))
	setBlock(Hy, 0, n, Gy)
	negQy := mat.NewDense(n, n, nil)
	negQy.Scale(-1, Qy)
	setBlock(Hy, n, 0, negQy)
	negAtilde := mat.NewDense(n, n, nil)
	negAtilde.Scale(-1, Atilde)
	setBlock(Hy, n, n, negAtilde)

	Y, err := solveHamiltonianRiccati(Hy, n)
	if err != nil {
		return nil, nil, err
	}

	XY := mulDense(X, Y)
	var eig mat.Eigen
	ok := eig.Factorize(XY, mat.EigenNone)
	if !ok {
		return nil, nil, ErrGammaNotAchievable
	}
	vals := eig.Values(nil)
	g2 := gamma * gamma
	for _, v := range vals {
		if math.Abs(real(v)) >= g2 {
			return nil, nil, ErrGammaNotAchievable
		}
	}

	return X, Y, nil
}

// solveHamiltonianRiccati returns the stabilizing solution of the Riccati
// equation with Hamiltonian H. H is rejected when an eigenvalue is on the
// imaginary axis to within rounding, where the stable/unstable split is
// arbitrary and yields spurious solutions.
func solveHamiltonianRiccati(H *mat.Dense, n int) (*mat.Dense, error) {
	nn := 2 * n
	hRaw := H.RawMatrix()
	hData := make([]float64, nn*nn)
	copyStrided(hData, nn, hRaw.Data, hRaw.Stride, nn, nn)

	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)

	selctg := func(wr, wi float64) bool { return wr < 0 }

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}
	if sdim != n {
		return nil, ErrNoStabilizing
	}
	if hasImaginaryAxisEigenvalue(hData, nn, wr, wi) {
		return nil, ErrNoStabilizing
	}

	u11 := make([]float64, n*n)
	u21 := make([]float64, n*n)
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)

	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	// X = U21 * U11^{-1}  =>  solve U11' * X' = U21'  =>  transpose approach
	xData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)
	symmetrize(xData, n, n)

	// Check positive semi-definiteness via eigenvalues
	X := mat.NewDense(n, n, xData)
	var eig mat.Eigen
	eigOk := eig.Factorize(X, mat.EigenNone)
	if !eigOk {
		return nil, ErrNoStabilizing
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if real(v) < -1e-8 {
			return nil, ErrNoStabilizing
		}
	}

	return X, nil
}

// An eigenvalue with |Re| <= hamiltonianAxisCandidate*|lambda| is on the
// imaginary axis when |Re| is also within hamiltonianAxisErr times its
// perturbation bound eps*||T||_F/s, s being its reciprocal condition number.
// Rounding splits an axis pair into a near-Jordan block with tiny s, while
// lightly damped normal modes keep s near 1 and slow modes of a stiff H are
// not candidates. A mode both lightly damped and ill-conditioned is treated
// conservatively.
const (
	hamiltonianAxisCandidate = 1e-4
	hamiltonianAxisErr       = 10
)

// hasImaginaryAxisEigenvalue reports whether the real Schur form t (n x n,
// eigenvalues wr + i wi) has an eigenvalue on the imaginary axis to within
// rounding.
func hasImaginaryAxisEigenvalue(t []float64, n int, wr, wi []float64) bool {
	const c2 = hamiltonianAxisCandidate * hamiltonianAxisCandidate
	var tNorm float64
	var tc, work []float64
	var selected []bool
	var iwork [1]int
	for i := range n {
		re := math.Abs(wr[i])
		if wi[i] < 0 || re*re > c2*(wr[i]*wr[i]+wi[i]*wi[i]) {
			continue
		}
		if tc == nil {
			for r := range n {
				for _, v := range t[r*n+r-min(r, 1) : (r+1)*n] {
					tNorm += v * v
				}
			}
			tNorm = math.Sqrt(tNorm)
			tc = make([]float64, n*n)
			selected = make([]bool, n)
			work = make([]float64, max(1, 2*n))
		}
		copy(tc, t)
		wrc, wic := slices.Clone(wr), slices.Clone(wi)
		clear(selected)
		selected[i] = true
		_, s, _, ok := impl.Dtrsen(1, false, selected, n, tc, n, nil, 1, wrc, wic, work, len(work), iwork[:], 1)
		if !ok || re <= hamiltonianAxisErr*eps()*tNorm/s {
			return true
		}
	}
	return false
}
