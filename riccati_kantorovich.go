package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// newtonUnproven reports whether the Newton–Kantorovich test fails for
// the Dare solution X with gain K (E of opts, if any; red the reduced
// inputs, W formed in their inputs w), so that a Newton step of X within
// newtonTol does not bound its error. In the explicit model a = E⁻¹A,
// b = E⁻¹B, Y = EᵀXE, the residual of Y + Δ is exactly
// F(Y) + L(Δ) − A_KᵀMA_K, M = Δb(W + bᵀΔb)⁻¹bᵀΔ, W = R + BᵀXB and L the
// Stein operator Δ − A_KᵀΔA_K, whose inverse maps A_KᵀMA_K to at most
// ‖M‖·‖P − I‖, P = A_KᵀPA_K + I. The Newton step η = ‖L⁻¹F(Y)‖ then
// bounds the error by 2η only for h = ‖P − I‖·‖bW⁻¹bᵀ‖·η ≤ 1/2, η at
// least the rounding eps·‖Y‖ of Y. A weakly controllable large mode broke
// it by 1e5: Y and K were within 1e-9 of their Hewer iterate in exact
// arithmetic, the float64 step within newtonTol, and both 0.17 and 0.2
// off the 320-bit solution. Where W is not positive definite in float64
// (cheap control) the test is not formed and reports false: X keeps the
// certificate it had.
func newtonUnproven(A, B *mat.Dense, opts *RiccatiOpts, red *reducedInputs, X, K *mat.Dense, step float64) bool {
	h, ok := newtonKantorovich(A, B, opts, red, X, K, step)
	return ok && !(h <= 0.5)
}

// newtonKantorovich returns h of newtonUnproven.
func newtonKantorovich(A, B *mat.Dense, opts *RiccatiOpts, red *reducedInputs, X, K *mat.Dense, step float64) (float64, bool) {
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
	bw, rw := red.bw, red.rw
	var xb, wd mat.Dense
	xb.Mul(X, bw)
	wd.Mul(bw.T(), &xb)
	wd.Add(&wd, rw)
	w := mat.NewSymDense(m, nil)
	for i := range m {
		for j := i; j < m; j++ {
			w.SetSym(i, j, 0.5*(wd.At(i, j)+wd.At(j, i)))
		}
	}
	var wc mat.Cholesky
	if !wc.Factorize(w) {
		return 0, false
	}
	b, Y := bw, X
	if E != nil {
		var lu mat.LU
		lu.Factorize(E)
		var f, g mat.Dense
		if lu.SolveTo(&f, false, acl) != nil || lu.SolveTo(&g, false, bw) != nil {
			return 0, false
		}
		acl, b = &f, &g
		var xe, ete mat.Dense
		xe.Mul(X, E)
		ete.Mul(E.T(), &xe)
		Y = &ete
	}
	var wb, bwb mat.Dense
	if wc.SolveTo(&wb, b.T()) != nil {
		return 0, false
	}
	bwb.Mul(b, &wb)
	P, err := dlyap(mat.DenseCopyOf(acl.T()), eyeDense(n), nil)
	if err != nil {
		return 0, false
	}
	for i := range n {
		P.Set(i, i, P.At(i, i)-1)
	}
	h := mat.Norm(P, 2) * mat.Norm(&bwb, 2) * mat.Norm(Y, 2) * max(step, eps())
	return h, !math.IsNaN(h)
}

// unprovenReduced returns res, the K of reduced inputs gainOfX accepts as
// the gain of its X, unless newtonUnproven: then it is the converged
// Hewer solution (hewerSolution: K replaced, X too when res.X is beyond
// newtonTol of it), else ok is false with miss d in units of gainFloor
// (Accuracy 1). The K so demoted were 0.0175 to 10.4 off the 320-bit
// gain at Accuracy 0, the Hewer sequence from their X moving K by 1e-10
// at its first step and by 0.03 to 0.95 at its second; the 320-bit gain,
// rounded, destabilized A − BK, so no float64 K near it is certifiable.
// Two float64 refinements of such an X and K do not show it: on one
// platform they converged within 3.6e-9 and 3e-8 of it.
func unprovenReduced(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, d gainDist) (*RiccatiResult, float64, bool) {
	if !newtonUnproven(A, B, opts, red, res.X, res.K, res.step) {
		return res, 0, true
	}
	if h, ok := hewerSolution(A, B, Q, R, opts, red, res); ok {
		return h, 0, true
	}
	return res, d.units(gainFloor), false
}
