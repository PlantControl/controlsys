package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// hewerGainPrecMin is the least precision of hewerGainSubstituted, which
// carries log2 of the gain's condition plus 96 bits: at 106 bits the
// Hewer gain stalled 8e-9 off the 320-bit one, at 160 it matched it.
const hewerGainPrecMin = 128

// hewerGainStepsMax is the most Hewer steps hewerGainMiss takes:
// from a gain of X 3.6e-4 off, the second step still moved K by 1.6e-7.
const hewerGainStepsMax = 4

// hewerGainSubstituted returns res with K the gain of X taken two Hewer
// steps on in extended precision (hewerGain) and Eig that of A − BK, once
// gainMissIn accepts the new K (Y6BJEX am.4). X is kept. With reduced
// inputs under cheap control both the float64 pencil K and the
// double-double gain of the rounded X were off the 320-bit gain by up to
// 6e-6 and 4.7e-6, either one the worse depending on the rounding of X,
// within the gain's move under X∘(1 ± eps); the gain of the extended
// Hewer iterate was within 1e-10 of it.
func hewerGainSubstituted(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, bool) {
	K, eig, ok := hewerGain(A, B, Q, R, opts, red, res, 2)
	if !ok {
		return nil, false
	}
	c := *res
	c.K, c.Eig = K, eig
	if !(gainMissIn(A, B, R, opts, red, &c).units(gainTol) <= 1) {
		return nil, false
	}
	return &c, true
}

// hewerGainMiss returns the relative distance of the K of res from the
// gain hewerGain forms from its X, with a descriptor E; ok is false
// without one or when hewerGain fails. There the certified float64 X
// wanders within cond(E)-amplified rounding, 1e-11 to 1e-9 off over
// successive Newton steps against an eps band of 3e-15, and a
// cheap-control gain amplifies that by 1e5 to 1e6: the gain of X was
// 1.2e-4 to 1.6e-3 off the 320-bit gain (band 1.3e-9), the Hewer gain
// within 1e-12 of it (V4OJPR).
func hewerGainMiss(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (float64, bool) {
	if opts == nil || opts.E == nil {
		return 0, false
	}
	K, _, ok := hewerGain(A, B, Q, R, opts, red, res, hewerGainStepsMax)
	if !ok {
		return 0, false
	}
	var diff mat.Dense
	diff.Sub(K, res.K)
	return mat.Norm(&diff, 1) / mat.Norm(K, 1), true
}

// hewerGain returns the gain of the X of res taken Hewer steps on in
// extended precision (extendedDare's model, E unrounded), starting from
// the gain gainMissIn measures K against, and the eigenvalues of A − BK,
// once a step from the second up to the steps-th moves K by at most
// newtonTol and A − BK is shown stable in that precision and in
// double-double (closedLoopStable).
func hewerGain(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, steps int) (*mat.Dense, []complex128, bool) {
	k0, cond, ok := gainIn(A, B, R, opts, red)(res.X)
	if !ok {
		return nil, nil, false
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
			return nil, nil, false
		}
		cond = max(cond, lu.Cond())
	} else {
		model = eyeDense(n)
	}
	prec := uint(min(extendedPrecMax, max(hewerGainPrecMin, math.Log2(cond)+96)))
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: crossTerm(opts), E: model}
	sol, ok := d.model(prec, nil)
	if !ok {
		return nil, nil, false
	}
	k, prev := xFrom(k0, prec), xMat{}
	for i := range steps {
		y, ok := sol.hewer(k)
		if !ok {
			return nil, nil, false
		}
		k1, ok := d.gain(sol, y)
		if !ok {
			return nil, nil, false
		}
		prev, k = k, k1
		if i >= 1 && xRel(k, prev) <= newtonTol {
			break
		}
	}
	if !(xRel(k, prev) <= newtonTol) {
		return nil, nil, false
	}
	K := k.dense()
	if !stableLoop(d.closedLoop(sol, K)) {
		return nil, nil, false
	}
	eig := make([]complex128, len(res.Eig))
	if !closedLoopStable(A, B, K, E, eig) {
		return nil, nil, false
	}
	return K, eig, true
}
