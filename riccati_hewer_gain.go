package controlsys

import "plantcontrol.org/v1/gonum/mat"

// hewerGainSubstituted returns res with K the gain of X taken two Hewer
// steps on in extended precision (extendedDare's model, E unrounded),
// starting from the gain gainMissIn measures K against, and Eig that of
// A − BK, once the second step moves K by at most newtonTol, A − BK is
// shown stable in that precision and in double-double (closedLoopStable),
// and gainMissIn accepts the new K (Y6BJEX am.4). X is kept. With reduced
// inputs under cheap control both the float64 pencil K and the
// double-double gain of the rounded X were off the 320-bit gain by up to
// 6e-6 and 4.7e-6, either one the worse depending on the rounding of X,
// within the gain's move under X∘(1 ± eps); the gain of the extended
// Hewer iterate was within 1e-10 of it.
func hewerGainSubstituted(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, bool) {
	k0, cond, ok := gainIn(A, B, R, opts, red)(res.X)
	if !ok {
		return nil, false
	}
	n, _ := A.Dims()
	var E *mat.Dense
	if opts != nil {
		E = opts.E
	}
	model := E
	if E != nil {
		var lu mat.LU
		lu.Factorize(E)
		if luNearSingular(&lu) {
			return nil, false
		}
		cond = max(cond, lu.Cond())
	} else {
		model = eyeDense(n)
	}
	prec := extendedPrec(cond)
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: crossTerm(opts), E: model}
	sol, ok := d.model(prec, nil)
	if !ok {
		return nil, false
	}
	k, prev := xFrom(k0, prec), xMat{}
	for range 2 {
		y, ok := sol.hewer(k)
		if !ok {
			return nil, false
		}
		k1, ok := d.gain(sol, y)
		if !ok {
			return nil, false
		}
		prev, k = k, k1
	}
	if !(xRel(k, prev) <= newtonTol) {
		return nil, false
	}
	K := k.dense()
	if !stableLoop(d.closedLoop(sol, K)) {
		return nil, false
	}
	eig := make([]complex128, len(res.Eig))
	if !closedLoopStable(A, B, K, E, eig) {
		return nil, false
	}
	c := *res
	c.K, c.Eig = K, eig
	if !(gainMissIn(A, B, R, opts, red, &c).units(gainTol) <= 1) {
		return nil, false
	}
	return &c, true
}
