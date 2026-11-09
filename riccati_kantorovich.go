package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// newtonUnproven reports whether the Newton–Kantorovich test fails for
// the Dare solution X with gain K (E of opts, if any), so that a Newton
// step of X within newtonTol does not bound its error. In the explicit
// model a = E⁻¹A, b = E⁻¹B, Y = EᵀXE, the residual of Y + Δ is exactly
// F(Y) + L(Δ) − A_KᵀΔb(W + bᵀΔb)⁻¹bᵀΔA_K, W = R + BᵀXB and L the Stein
// operator Δ − A_KᵀΔA_K, with ‖L⁻¹‖ = ‖P‖, P = A_KᵀPA_K + I. The
// Newton step η = ‖L⁻¹F(Y)‖ then bounds the error by 2η only for
// h = ‖P‖·‖A_K‖²·‖bW⁻¹bᵀ‖·η ≤ 1/2, η at least the rounding eps·‖Y‖ of
// Y. A weakly controllable large mode broke it by 1e20: Y and K were
// within 1e-9 of their Hewer iterate in exact arithmetic, the float64
// step within newtonTol, and both 0.17 and 0.2 off the 320-bit solution.
// It also reports true when the test cannot be formed.
func newtonUnproven(A, B, R *mat.Dense, opts *RiccatiOpts, X, K *mat.Dense, step float64) bool {
	h, ok := newtonKantorovich(A, B, R, opts, X, K, step)
	return !ok || !(h <= 0.5)
}

// newtonKantorovich returns h of newtonUnproven.
func newtonKantorovich(A, B, R *mat.Dense, opts *RiccatiOpts, X, K *mat.Dense, step float64) (float64, bool) {
	n, m := B.Dims()
	if n == 0 || m == 0 {
		return 0, true
	}
	var E *mat.Dense
	if opts != nil {
		E = opts.E
	}
	aRaw, bRaw, kRaw := A.RawMatrix(), B.RawMatrix(), K.RawMatrix()
	hi, lo := make([]float64, n*n), make([]float64, n*n)
	copyStrided(hi, n, aRaw.Data, aRaw.Stride, n, n)
	ddMulAcc(n, n, m, -1, bRaw.Data, bRaw.Stride, 1, kRaw.Data, kRaw.Stride, 1, hi, lo, n)
	for i := range hi {
		hi[i] += lo[i]
	}
	acl := mat.NewDense(n, n, hi)
	b := B
	Y := X
	if E != nil {
		var lu mat.LU
		lu.Factorize(E)
		var f, g mat.Dense
		if lu.SolveTo(&f, false, acl) != nil || lu.SolveTo(&g, false, B) != nil {
			return 0, false
		}
		acl, b = &f, &g
		var xe, ete mat.Dense
		xe.Mul(X, E)
		ete.Mul(E.T(), &xe)
		Y = &ete
	}
	var bx, w mat.Dense
	bx.Mul(B.T(), X)
	w.Mul(&bx, B)
	w.Add(&w, R)
	var wb mat.Dense
	if err := wb.Solve(&w, b.T()); err != nil {
		return 0, false
	}
	var bwb mat.Dense
	bwb.Mul(b, &wb)
	P, err := dlyap(mat.DenseCopyOf(acl.T()), eyeDense(n), nil)
	if err != nil {
		return 0, false
	}
	an := mat.Norm(acl, 2)
	h := mat.Norm(P, 2) * an * an * mat.Norm(&bwb, 2) * mat.Norm(Y, 2) * max(step, eps())
	return h, !math.IsNaN(h)
}

// unprovenReduced returns res, the K of reduced inputs gainOfX accepts as
// the gain of its X, unless newtonUnproven: then it is the converged
// Hewer solution (hewerSolution, K, X and EᵀXE replaced), else ok is
// false with miss d in units of gainFloor (Accuracy 1). On weakly
// controllable large modes such K were 0.0175 to 10.4 off the 320-bit
// gain at Accuracy 0, the Hewer sequence from their X moving K by 1e-10
// at its first step and by 0.03 to 0.95 at its second. X is not held to
// res.X (hewerGainConverged): there 180 of 40000 rows with X 1.2e-8 to
// 2.3e-7 off the 320-bit X, within their eps-sensitivity, would have lost
// Accuracy 0.
func unprovenReduced(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, d gainDist) (*RiccatiResult, float64, bool) {
	if !newtonUnproven(A, B, R, opts, res.X, res.K, res.step) {
		return res, 0, true
	}
	if h, ok := hewerSolution(A, B, Q, R, opts, red, res); ok {
		return h, 0, true
	}
	return res, d.units(gainFloor), false
}
