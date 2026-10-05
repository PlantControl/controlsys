package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
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

func newHinfGeneralPlant(gp *generalizedPlantPartition) (*hinfGeneralPlant, error) {
	n, m1, m2, p1, p2 := gp.n, gp.m1, gp.m2, gp.p1, gp.p2
	if p1 < m2 || m1 < p2 {
		return nil, fmt.Errorf("D12 or D21 rank deficient: %w", ErrInvalidPartition)
	}

	var svd12 mat.SVD
	if !svd12.Factorize(gp.D12, mat.SVDFull) {
		return nil, fmt.Errorf("D12 or D21 rank deficient: %w", ErrInvalidPartition)
	}
	s12 := svd12.Values(nil)
	if !fullRankValues(s12, max(p1, m2)) {
		return nil, fmt.Errorf("D12 or D21 rank deficient: %w", ErrInvalidPartition)
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
		return nil, fmt.Errorf("D12 or D21 rank deficient: %w", ErrInvalidPartition)
	}
	s21 := svd21.Values(nil)
	if !fullRankValues(s21, max(p2, m1)) {
		return nil, fmt.Errorf("D12 or D21 rank deficient: %w", ErrInvalidPartition)
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
		sv, err := maxSingularValue(extractBlock(D11, 0, 0, p1-m2, m1))
		if err != nil {
			return nil, err
		}
		gammaLB = sv
	}
	if m1 > p2 {
		sv, err := maxSingularValue(extractBlock(D11, 0, 0, p1, m1-p2))
		if err != nil {
			return nil, err
		}
		gammaLB = math.Max(gammaLB, sv)
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
		return nil, nil, nil, nil, fmt.Errorf("γ = %g not achievable: %w", gamma, ErrGammaNotAchievable)
	}

	Rinv, err = shiftedInverse(hp.D1dtD1d, gp.m1, gamma)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("γ = %g not achievable: %w", gamma, ErrGammaNotAchievable)
	}
	BRinv := mulDense(hp.B, Rinv)
	Ax := mulDense(BRinv, hp.D1dtC1)
	Ax.Sub(gp.A, Ax)
	Gx := mulDense(BRinv, hp.BT)
	Gx.Scale(-1, Gx)
	Qx := mulDense(mat.DenseCopyOf(hp.D1dtC1.T()), mulDense(Rinv, hp.D1dtC1))
	Qx.Sub(hp.C1tC1, Qx)
	X, err = solveHamiltonianRiccati(hamiltonian(Ax, Gx, Qx, n), n)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	Rtinv, err = shiftedInverse(hp.Dd1Dd1, gp.p1, gamma)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("γ = %g not achievable: %w", gamma, ErrGammaNotAchievable)
	}
	B1Dd1tRtinv := mulDense(hp.B1Dd1t, Rtinv)
	Ay := mulDense(B1Dd1tRtinv, hp.C)
	Ay.Sub(gp.A, Ay)
	Gy := mulDense(hp.CT, mulDense(Rtinv, hp.C))
	Gy.Scale(-1, Gy)
	Qy := mulDense(B1Dd1tRtinv, mat.DenseCopyOf(hp.B1Dd1t.T()))
	Qy.Sub(hp.B1B1t, Qy)
	Y, err = solveHamiltonianRiccati(hamiltonian(mat.DenseCopyOf(Ay.T()), Gy, Qy, n), n)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	var eig mat.Eigen
	if !eig.Factorize(mulDense(X, Y), mat.EigenNone) {
		return nil, nil, nil, nil, fmt.Errorf("γ = %g not achievable: %w", gamma, ErrGammaNotAchievable)
	}
	g2 := gamma * gamma
	for _, v := range eig.Values(nil) {
		if math.Abs(real(v)) >= g2 {
			return nil, nil, nil, nil, fmt.Errorf("γ = %g not achievable: %w", gamma, ErrGammaNotAchievable)
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
			return nil, fmt.Errorf("central D11 singular at γ = %g: %w", gamma, ErrGammaNotAchievable)
		}
		Dk.Add(Dk, mulDense(mulDense(D1121, mat.DenseCopyOf(D1111.T())), WinvD1112))
	}
	Dk.Scale(-1, Dk)
	return Dk, nil
}

// hinfYoula is the parametrization K = F_l(M∞, Q) of all controllers with
// ‖T_zw‖∞ < γ for the scaled plant, Q ∈ RH∞ with ‖Q‖∞ < γ (Zhou, Doyle &
// Glover, Robust and Optimal Control, Thm 17.13). Its fields are M∞ less the
// zero (2,2) block; Q = 0 is the central controller.
type hinfYoula struct {
	hp                  *hinfGeneralPlant
	gamma               float64
	A, B1, B2, C1, C2   *mat.Dense
	D11, D12hat, D21hat *mat.Dense
}

func hinfSynGeneral(gp *generalizedPlantPartition) (*HinfSynResult, error) {
	hp, err := newHinfGeneralPlant(gp)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}
	gamma, err := hinfControllerGamma(hp.gammaLB, func(g float64) bool {
		_, _, _, _, err := hp.riccatis(g)
		return err == nil
	})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}
	X, Y, Rinv, Rtinv, err := hp.riccatis(gamma)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}
	M, err := hp.youla(gamma, X, Y, Rinv, Rtinv)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", gp.op, err)
	}
	Ak, Bk, Ck, Dk := M.controller(nil)
	if loopShiftIllPosed(gp.D22, Dk) {
		Dq, err := M.wellPosedQ(Dk)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", gp.op, err)
		}
		Ak, Bk, Ck, Dk = M.controller(Dq)
	}

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

// youla returns M∞ at gamma from the Riccati solutions X and Y.
func (hp *hinfGeneralPlant) youla(gamma float64, X, Y, Rinv, Rtinv *mat.Dense) (*hinfYoula, error) {
	gp := hp.gp
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
	D12hat, D21hat, err := hp.youlaFeedthrough(gamma)
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
	B2L := extractBlock(L, 0, p1-m2, n, m2)
	B2L.Add(B2L, hp.B2)
	BL := mulDense(B2L, Dhat)
	BL.Sub(BL, extractBlock(L, 0, p1, n, p2))
	B1 := mat.NewDense(n, p2, nil)
	B2 := mat.NewDense(n, m2, nil)
	if err := lu.SolveTo(B1, false, BL); err != nil {
		return nil, fmt.Errorf("controller state matrix singular at γ = %g: %w", gamma, ErrGammaNotAchievable)
	}
	if err := lu.SolveTo(B2, false, mulDense(B2L, D12hat)); err != nil {
		return nil, fmt.Errorf("controller state matrix singular at γ = %g: %w", gamma, ErrGammaNotAchievable)
	}

	C1 := mulDense(Dhat, C2F)
	C1.Sub(extractBlock(F, m1, 0, m2, n), C1)
	A := mulDense(hp.B, F)
	A.Add(gp.A, A)
	A.Sub(A, mulDense(B1, C2F))
	C2 := mulDense(D21hat, C2F)
	C2.Scale(-1, C2)
	return &hinfYoula{hp: hp, gamma: gamma, A: A, B1: B1, B2: B2, C1: C1, C2: C2, D11: Dhat, D12hat: D12hat, D21hat: D21hat}, nil
}

// youlaFeedthrough returns D̂12 and D̂21 with
// D̂12 D̂12ᵀ = I − D1121 (γ²I − D1111ᵀD1111)⁻¹ D1121ᵀ and
// D̂21ᵀ D̂21 = I − D1112ᵀ (γ²I − D1111 D1111ᵀ)⁻¹ D1112.
func (hp *hinfGeneralPlant) youlaFeedthrough(gamma float64) (D12hat, D21hat *mat.Dense, err error) {
	gp := hp.gp
	r, c := gp.p1-gp.m2, gp.m1-gp.p2
	// gram returns I − Dᵀ (γ²I − EᵀE)⁻¹ D for D (k×dim) and E (rows×k).
	gram := func(D, E func() *mat.Dense, rows, k, dim int) (*mat.Dense, error) {
		G := eyeDense(dim)
		if k == 0 {
			return G, nil
		}
		W := mat.NewDense(k, k, nil)
		if rows > 0 {
			e := E()
			W.Mul(e.T(), e)
			W.Scale(-1, W)
		}
		for i := range k {
			W.Set(i, i, W.At(i, i)+gamma*gamma)
		}
		WinvD := mat.NewDense(k, dim, nil)
		var lu mat.LU
		lu.Factorize(W)
		d := D()
		if err := lu.SolveTo(WinvD, false, d); err != nil {
			return nil, fmt.Errorf("Youla feedthrough singular at γ = %g: %w", gamma, ErrGammaNotAchievable)
		}
		G.Sub(G, mulDense(mat.DenseCopyOf(d.T()), WinvD))
		return G, nil
	}
	cholUpper := func(G *mat.Dense, dim int) (*mat.Dense, error) {
		var ch mat.Cholesky
		if !ch.Factorize(mat.NewSymDense(dim, G.RawMatrix().Data)) {
			return nil, fmt.Errorf("Youla feedthrough not positive definite at γ = %g: %w", gamma, ErrGammaNotAchievable)
		}
		var U mat.TriDense
		ch.UTo(&U)
		return mat.DenseCopyOf(&U), nil
	}
	D1111 := func() *mat.Dense { return extractBlock(hp.D11, 0, 0, r, c) }
	G12, err := gram(
		func() *mat.Dense { return mat.DenseCopyOf(extractBlock(hp.D11, r, 0, gp.m2, c).T()) },
		D1111, r, c, gp.m2)
	if err != nil {
		return nil, nil, err
	}
	U12, err := cholUpper(G12, gp.m2)
	if err != nil {
		return nil, nil, err
	}
	G21, err := gram(
		func() *mat.Dense { return extractBlock(hp.D11, 0, c, r, gp.p2) },
		func() *mat.Dense { return mat.DenseCopyOf(D1111().T()) }, c, r, gp.p2)
	if err != nil {
		return nil, nil, err
	}
	D21hat, err = cholUpper(G21, gp.p2)
	if err != nil {
		return nil, nil, err
	}
	return mat.DenseCopyOf(U12.T()), D21hat, nil
}

// controller returns F_l(M∞, Q) for the constant Q (nil for the central
// controller), unscaled to the original plant coordinates.
func (M *hinfYoula) controller(Q *mat.Dense) (Ak, Bk, Ck, Dk *mat.Dense) {
	A, B1, C1, D11 := M.A, M.B1, M.C1, M.D11
	if Q != nil {
		B2Q := mulDense(M.B2, Q)
		D12Q := mulDense(M.D12hat, Q)
		A = mulDense(B2Q, M.C2)
		A.Add(M.A, A)
		B1 = mulDense(B2Q, M.D21hat)
		B1.Add(M.B1, B1)
		C1 = mulDense(D12Q, M.C2)
		C1.Add(M.C1, C1)
		D11 = mulDense(D12Q, M.D21hat)
		D11.Add(M.D11, D11)
	}
	hp := M.hp
	return A, mulDense(B1, hp.R21inv), mulDense(hp.R12inv, C1), mulDense(mulDense(hp.R12inv, D11), hp.R21inv)
}

// hinfYoulaFraction is ‖Q‖/γ for the constant Q that makes an ill-posed
// central loop shift well posed.
const hinfYoulaFraction = 0.5

// wellPosedQ returns a constant Q with ‖Q‖ = hinfYoulaFraction·γ for which
// I + D22·Dk(Q) is invertible, Dk0 = Dk(0) being the central feedthrough.
// With D22 = U1 S V1ᵀ (rank r) and Dk(Q) = Dk0 + T1 Q T2,
// det(I + D22 Dk(Q)) = det(H0 + A Q B) for H0 = I + S V1ᵀ Dk0 U1,
// A = S V1ᵀ T1 (full row rank) and B = T2 U1 (full column rank). Q = A⁺ X B⁺
// with X = c·Uh Vhᵀ from the SVD H0 = Uh Σ Vhᵀ gives H0 + A Q B =
// Uh (Σ + c I) Vhᵀ, whose smallest singular value is at least
// c = hinfYoulaFraction·γ·σmin(A)·σmin(B), while ‖Q‖ ≤ c/(σmin(A)·σmin(B)).
func (M *hinfYoula) wellPosedQ(Dk0 *mat.Dense) (*mat.Dense, error) {
	gp := M.hp.gp
	var svd mat.SVD
	if !svd.Factorize(gp.D22, mat.SVDThin) {
		return nil, fmt.Errorf("SVD of D22 failed: %w", ErrSchurFailed)
	}
	s := svd.Values(nil)
	r := 0
	for r < len(s) && s[r] > s[0]*float64(max(gp.p2, gp.m2))*eps() {
		r++
	}
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)
	SV1t := mat.DenseCopyOf(V.Slice(0, gp.m2, 0, r).T())
	for i := range r {
		for j := range gp.m2 {
			SV1t.Set(i, j, s[i]*SV1t.At(i, j))
		}
	}
	U1 := mat.DenseCopyOf(U.Slice(0, gp.p2, 0, r))
	H0 := mulDense(mulDense(SV1t, Dk0), U1)
	for i := range r {
		H0.Set(i, i, H0.At(i, i)+1)
	}
	Am := mulDense(SV1t, mulDense(M.hp.R12inv, M.D12hat))
	Bm := mulDense(mulDense(M.D21hat, M.hp.R21inv), U1)

	Ainv, sa, err := thinPseudoInverse(Am)
	if err != nil {
		return nil, err
	}
	Binv, sb, err := thinPseudoInverse(Bm)
	if err != nil {
		return nil, err
	}
	var hsvd mat.SVD
	if !hsvd.Factorize(H0, mat.SVDThin) {
		return nil, fmt.Errorf("SVD of the loop-shift matrix failed: %w", ErrSchurFailed)
	}
	var Uh, Vh mat.Dense
	hsvd.UTo(&Uh)
	hsvd.VTo(&Vh)
	X := mulDense(&Uh, mat.DenseCopyOf(Vh.T()))
	X.Scale(hinfYoulaFraction*M.gamma*sa*sb, X)
	return mulDense(mulDense(Ainv, X), Binv), nil
}

// thinPseudoInverse returns the pseudo-inverse of the full-rank M and its
// smallest singular value.
func thinPseudoInverse(M *mat.Dense) (*mat.Dense, float64, error) {
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDThin) {
		return nil, 0, fmt.Errorf("SVD failed: %w", ErrSchurFailed)
	}
	s := svd.Values(nil)
	k := len(s)
	if !fullRankValues(s, k) {
		return nil, 0, fmt.Errorf("Youla loop-shift factor rank deficient: %w", ErrAlgebraicLoop)
	}
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)
	r, c := V.Dims()
	VS := mat.NewDense(r, c, nil)
	for i := range r {
		for j := range c {
			VS.Set(i, j, V.At(i, j)/s[j])
		}
	}
	return mulDense(VS, mat.DenseCopyOf(U.T())), s[k-1], nil
}
