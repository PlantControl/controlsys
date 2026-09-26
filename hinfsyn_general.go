package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// hinfGeneralPlant is the D11 != 0 H-infinity problem scaled so that
// D12 = [0; I] and D21 = [0 I] (Zhou, Doyle & Glover, Robust and Optimal
// Control, ch. 17): z~ = Th1' z, w = Th2 w~, u = R12^-1 u~, y~ = R21^-1 y.
// The Riccati solutions X and Y are invariant under this scaling.
type hinfGeneralPlant struct {
	gp             *generalizedPlantPartition
	B, B2          *mat.Dense
	C, C2          *mat.Dense
	D11            *mat.Dense
	D1dtC1, BT, CT *mat.Dense
	D1dtD1d        *mat.Dense
	B1Dd1t, Dd1Dd1 *mat.Dense
	C1tC1, B1B1t   *mat.Dense
	R12inv, R21inv *mat.Dense
	gammaLB        float64
}

// hinfAxisTol rejects Hamiltonians with imaginary-axis eigenvalues that
// rounding would otherwise split into a spurious stable/unstable pair.
const hinfAxisTol = 1e-9

func newHinfGeneralPlant(gp *generalizedPlantPartition) (*hinfGeneralPlant, error) {
	n, m1, m2, p1, p2 := gp.n, gp.m1, gp.m2, gp.p1, gp.p2
	if p1 < m2 || m1 < p2 {
		return nil, ErrInvalidPartition
	}

	var svd12 mat.SVD
	if !svd12.Factorize(gp.D12, mat.SVDFull) {
		return nil, ErrInvalidPartition
	}
	s12 := svd12.Values(nil)
	if !fullRankValues(s12, max(p1, m2)) {
		return nil, ErrInvalidPartition
	}
	var U12, V12 mat.Dense
	svd12.UTo(&U12)
	svd12.VTo(&V12)
	Th1 := rotateColumns(&U12, m2)
	R12inv := mat.NewDense(m2, m2, nil)
	for i := range m2 {
		for j := range m2 {
			R12inv.Set(i, j, V12.At(i, j)/s12[j])
		}
	}

	var svd21 mat.SVD
	if !svd21.Factorize(gp.D21, mat.SVDFull) {
		return nil, ErrInvalidPartition
	}
	s21 := svd21.Values(nil)
	if !fullRankValues(s21, max(p2, m1)) {
		return nil, ErrInvalidPartition
	}
	var U21, V21 mat.Dense
	svd21.UTo(&U21)
	svd21.VTo(&V21)
	Th2 := rotateColumns(&V21, p2)
	R21inv := mat.NewDense(p2, p2, nil)
	for i := range p2 {
		for j := range p2 {
			R21inv.Set(i, j, U21.At(j, i)/s21[i])
		}
	}

	B1 := mulDense(gp.B1, Th2)
	B2 := mulDense(gp.B2, R12inv)
	C1 := mulDense(mat.DenseCopyOf(Th1.T()), gp.C1)
	C2 := mulDense(R21inv, gp.C2)
	D11 := mulDense(mulDense(mat.DenseCopyOf(Th1.T()), gp.D11), Th2)

	m, p := m1+m2, p1+p2
	B := mat.NewDense(n, m, nil)
	setBlock(B, 0, 0, B1)
	setBlock(B, 0, m1, B2)
	C := mat.NewDense(p, n, nil)
	setBlock(C, 0, 0, C1)
	setBlock(C, p1, 0, C2)
	D1d := mat.NewDense(p1, m, nil)
	setBlock(D1d, 0, 0, D11)
	for i := range m2 {
		D1d.Set(p1-m2+i, m1+i, 1)
	}
	Dd1 := mat.NewDense(p, m1, nil)
	setBlock(Dd1, 0, 0, D11)
	for i := range p2 {
		Dd1.Set(p1+i, m1-p2+i, 1)
	}

	var gammaLB float64
	if p1 > m2 {
		gammaLB = maxSVD(extractBlock(D11, 0, 0, p1-m2, m1))
	}
	if m1 > p2 {
		gammaLB = math.Max(gammaLB, maxSVD(extractBlock(D11, 0, 0, p1, m1-p2)))
	}

	D1dT := mat.DenseCopyOf(D1d.T())
	return &hinfGeneralPlant{
		gp:      gp,
		B:       B,
		B2:      B2,
		C:       C,
		C2:      C2,
		D11:     D11,
		D1dtC1:  mulDense(D1dT, C1),
		BT:      mat.DenseCopyOf(B.T()),
		CT:      mat.DenseCopyOf(C.T()),
		D1dtD1d: mulDense(D1dT, D1d),
		B1Dd1t:  mulDense(B1, mat.DenseCopyOf(Dd1.T())),
		Dd1Dd1:  mulDense(Dd1, mat.DenseCopyOf(Dd1.T())),
		C1tC1:   mulDense(mat.DenseCopyOf(C1.T()), C1),
		B1B1t:   mulDense(B1, mat.DenseCopyOf(B1.T())),
		R12inv:  R12inv,
		R21inv:  R21inv,
		gammaLB: gammaLB,
	}, nil
}

func fullRankValues(s []float64, dim int) bool {
	return len(s) > 0 && s[len(s)-1] > s[0]*float64(dim)*eps()
}

// rotateColumns returns M with its first k columns moved to the end.
func rotateColumns(M *mat.Dense, k int) *mat.Dense {
	r, c := M.Dims()
	out := mat.NewDense(r, c, nil)
	for j := range c {
		src := (j + k) % c
		for i := range r {
			out.Set(i, j, M.At(i, src))
		}
	}
	return out
}

// shiftedInverse returns (G - gamma^2 diag(I_k, 0))^-1.
func shiftedInverse(G *mat.Dense, k int, gamma float64) (*mat.Dense, error) {
	R := denseCopy(G)
	g2 := gamma * gamma
	for i := range k {
		R.Set(i, i, R.At(i, i)-g2)
	}
	n, _ := R.Dims()
	return invertSmall(R, n)
}

// riccatis solves the general H-infinity Riccati equations at gamma and
// returns X, Y together with R^-1 and R~^-1.
func (hp *hinfGeneralPlant) riccatis(gamma float64) (X, Y, Rinv, Rtinv *mat.Dense, err error) {
	gp := hp.gp
	n := gp.n
	if !(gamma > hp.gammaLB) {
		return nil, nil, nil, nil, ErrGammaNotAchievable
	}

	Rinv, err = shiftedInverse(hp.D1dtD1d, gp.m1, gamma)
	if err != nil {
		return nil, nil, nil, nil, ErrGammaNotAchievable
	}
	BRinv := mulDense(hp.B, Rinv)
	Ax := mulDense(BRinv, hp.D1dtC1)
	Ax.Sub(gp.A, Ax)
	Gx := mulDense(BRinv, hp.BT)
	Gx.Scale(-1, Gx)
	Qx := mulDense(mat.DenseCopyOf(hp.D1dtC1.T()), mulDense(Rinv, hp.D1dtC1))
	Qx.Sub(hp.C1tC1, Qx)
	X, err = solveHamiltonianRiccati(hamiltonian(Ax, Gx, Qx, n), n, hinfAxisTol)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	Rtinv, err = shiftedInverse(hp.Dd1Dd1, gp.p1, gamma)
	if err != nil {
		return nil, nil, nil, nil, ErrGammaNotAchievable
	}
	B1Dd1tRtinv := mulDense(hp.B1Dd1t, Rtinv)
	Ay := mulDense(B1Dd1tRtinv, hp.C)
	Ay.Sub(gp.A, Ay)
	Gy := mulDense(hp.CT, mulDense(Rtinv, hp.C))
	Gy.Scale(-1, Gy)
	Qy := mulDense(B1Dd1tRtinv, mat.DenseCopyOf(hp.B1Dd1t.T()))
	Qy.Sub(hp.B1B1t, Qy)
	Y, err = solveHamiltonianRiccati(hamiltonian(mat.DenseCopyOf(Ay.T()), Gy, Qy, n), n, hinfAxisTol)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	var eig mat.Eigen
	if !eig.Factorize(mulDense(X, Y), mat.EigenNone) {
		return nil, nil, nil, nil, ErrGammaNotAchievable
	}
	g2 := gamma * gamma
	for _, v := range eig.Values(nil) {
		if math.Abs(real(v)) >= g2 {
			return nil, nil, nil, nil, ErrGammaNotAchievable
		}
	}
	return X, Y, Rinv, Rtinv, nil
}

// hamiltonian returns [F G; -Q -F'].
func hamiltonian(F, G, Q *mat.Dense, n int) *mat.Dense {
	H := mat.NewDense(2*n, 2*n, nil)
	setBlock(H, 0, 0, F)
	setBlock(H, 0, n, G)
	negQ := mat.NewDense(n, n, nil)
	negQ.Scale(-1, Q)
	setBlock(H, n, 0, negQ)
	negFt := mat.NewDense(n, n, nil)
	negFt.Scale(-1, F.T())
	setBlock(H, n, n, negFt)
	return H
}

// centralD11 returns the central controller feedthrough
// -D1121 D1111' (gamma^2 I - D1111 D1111')^-1 D1112 - D1122.
func (hp *hinfGeneralPlant) centralD11(gamma float64) (*mat.Dense, error) {
	gp := hp.gp
	r, c := gp.p1-gp.m2, gp.m1-gp.p2
	Dk := extractBlock(hp.D11, r, c, gp.m2, gp.p2)
	if r > 0 && c > 0 {
		D1111 := extractBlock(hp.D11, 0, 0, r, c)
		D1112 := extractBlock(hp.D11, 0, c, r, gp.p2)
		D1121 := extractBlock(hp.D11, r, 0, gp.m2, c)
		W := mulDense(D1111, mat.DenseCopyOf(D1111.T()))
		W.Scale(-1, W)
		for i := range r {
			W.Set(i, i, W.At(i, i)+gamma*gamma)
		}
		var lu mat.LU
		lu.Factorize(W)
		WinvD1112 := mat.NewDense(r, gp.p2, nil)
		if err := lu.SolveTo(WinvD1112, false, D1112); err != nil {
			return nil, ErrGammaNotAchievable
		}
		Dk.Add(Dk, mulDense(mulDense(D1121, mat.DenseCopyOf(D1111.T())), WinvD1112))
	}
	Dk.Scale(-1, Dk)
	return Dk, nil
}

func hinfSynGeneral(gp *generalizedPlantPartition) (*HinfSynResult, error) {
	hp, err := newHinfGeneralPlant(gp)
	if err != nil {
		return nil, err
	}
	gamma, err := hinfBisect(hp.gammaLB, func(g float64) bool {
		_, _, _, _, err := hp.riccatis(g)
		return err == nil
	})
	if err != nil {
		return nil, err
	}
	X, Y, Rinv, Rtinv, err := hp.riccatis(gamma)
	if err != nil {
		return nil, err
	}

	n, m1, p1, m2, p2 := gp.n, gp.m1, gp.p1, gp.m2, gp.p2
	F := mulDense(hp.BT, X)
	F.Add(F, hp.D1dtC1)
	F = mulDense(Rinv, F)
	F.Scale(-1, F)
	L := mulDense(Y, hp.CT)
	L.Add(L, hp.B1Dd1t)
	L = mulDense(L, Rtinv)
	L.Scale(-1, L)

	Dhat, err := hp.centralD11(gamma)
	if err != nil {
		return nil, err
	}

	Zarg := mulDense(Y, X)
	Zarg.Scale(-1/(gamma*gamma), Zarg)
	for i := range n {
		Zarg.Set(i, i, Zarg.At(i, i)+1)
	}
	var lu mat.LU
	lu.Factorize(Zarg)

	C2F := extractBlock(F, m1-p2, 0, p2, n)
	C2F.Add(C2F, hp.C2)
	BL := extractBlock(L, 0, p1-m2, n, m2)
	BL.Add(BL, hp.B2)
	BL = mulDense(BL, Dhat)
	BL.Sub(BL, extractBlock(L, 0, p1, n, p2))
	Bhat := mat.NewDense(n, p2, nil)
	if err := lu.SolveTo(Bhat, false, BL); err != nil {
		return nil, ErrGammaNotAchievable
	}

	Chat := mulDense(Dhat, C2F)
	Chat.Sub(extractBlock(F, m1, 0, m2, n), Chat)
	Ak := mulDense(hp.B, F)
	Ak.Add(gp.A, Ak)
	Ak.Sub(Ak, mulDense(Bhat, C2F))

	Bk := mulDense(Bhat, hp.R21inv)
	Ck := mulDense(hp.R12inv, Chat)
	Dk := mulDense(mulDense(hp.R12inv, Dhat), hp.R21inv)

	K, err := gp.newController(Ak, Bk, Ck, Dk)
	if err != nil {
		return nil, err
	}
	clPoles, err := gp.closedLoopPoles(Ak, Bk, Ck, Dk)
	if err != nil {
		return nil, err
	}
	return &HinfSynResult{K: K, GammaOpt: gamma, X: X, Y: Y, CLPoles: clPoles}, nil
}
