package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// hewerGainPrecMin is the least precision of hewerGainSubstituted, which
// carries log2 of the gain's condition plus 96 bits: at 106 bits the
// Hewer gain stalled 8e-9 off the 320-bit one, at 160 it matched it.
const hewerGainPrecMin = 128

// hewerGainStepsMax is the most Hewer steps hewerGainContradicts takes:
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
	h, ok := hewerGain(A, B, Q, R, opts, red, res, 2, 0)
	if !ok {
		return nil, false
	}
	c := *res
	c.K, c.Eig = h.K, h.eig
	if !(gainMissIn(A, B, R, opts, red, &c).units(gainTol) <= 1) {
		return nil, false
	}
	return &c, true
}

// hewerGainContradicts returns the relative distance miss of the K of res
// from the gain hewerGain forms from its X, with a descriptor E, and
// whether miss exceeds both floor and 4√N times the largest move of that
// gain under two componentwise eps perturbations of the N entries of the
// data (perturbedMove), a lower estimate of K's band that √N lifts to an
// upper one. ok is false without E or when hewerGain fails. There the
// certified float64 X wanders within cond(E)-amplified rounding, 1e-11
// to 1e-9 off over successive Newton steps against an eps band of 3e-15,
// and a cheap-control gain amplifies that by 1e5 to 1e6: the gain of X
// was 1.2e-4 to 1.6e-3 off the 320-bit gain (band 1.3e-9), the Hewer gain
// within 1e-12 of it (V4OJPR). Where K's band is wide the gain of X is
// as good as any: K 1.4e-6 off the Hewer gain had a band of 9.8e-6. The
// precision also carries the condition of the gain in the given inputs:
// with duplicated inputs that was 1.9e33 against under 10 in the reduced
// ones, and at 128 bits the Hewer gain converged 4.9e-6 off the 320-bit
// one.
func hewerGainContradicts(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, floor float64) (miss float64, contradicts, ok bool) {
	if opts == nil || opts.E == nil {
		return 0, false, false
	}
	_, cond, ok := ddGain(A, B, R, crossTerm(opts), res.X)
	if !ok {
		cond = math.Inf(1)
	}
	h, ok := hewerGain(A, B, Q, R, opts, red, res, hewerGainStepsMax, cond)
	if !ok {
		return 0, false, false
	}
	var diff mat.Dense
	diff.Sub(h.K, res.K)
	miss = mat.Norm(&diff, 1) / mat.Norm(h.K, 1)
	if !(miss > floor) {
		return miss, false, true
	}
	x, ok := h.sol.xOf(h.sol.y)
	if !ok {
		return miss, false, true
	}
	kw, ok := h.d.noiseGain(h.sol, h.sol.y)
	if !ok {
		return miss, false, true
	}
	move := 0.0
	for seed := range uint64(2) {
		mv, ok := h.d.perturbedMove(h.sol, x, kw, h.prec, seed)
		if !ok {
			return miss, false, true
		}
		move = max(move, mv.k)
	}
	n, m := B.Dims()
	entries := 2*n*n + 2*n*m + m*m
	if opts.S != nil {
		entries += n * m
	}
	return miss, miss > 4*math.Sqrt(float64(entries))*move, true
}

// hewerGainSol is the gain K of hewerGain, the eigenvalues eig of A − BK,
// and the model sol (y and k of its last step) of d in prec bits.
type hewerGainSol struct {
	K    *mat.Dense
	eig  []complex128
	d    extendedDare
	sol  extendedSol
	prec uint
}

// hewerGain returns the gain of the X of res taken Hewer steps on in
// extended precision (extendedDare's model, E unrounded), starting from
// the gain gainMissIn measures K against, and the eigenvalues of A − BK,
// once a step from the second up to the steps-th moves K by at most
// newtonTol and A − BK is shown stable in that precision and in
// double-double (closedLoopStable). The precision carries log2 of the
// largest of the gain's condition, that of E and cond0, plus 96 bits.
func hewerGain(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, steps int, cond0 float64) (hewerGainSol, bool) {
	var h hewerGainSol
	k0, cond, ok := gainIn(A, B, R, opts, red)(res.X)
	if !ok {
		return h, false
	}
	cond = max(cond, cond0)
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
			return h, false
		}
		cond = max(cond, lu.Cond())
	} else {
		model = eyeDense(n)
	}
	h.prec = uint(min(extendedPrecMax, max(hewerGainPrecMin, math.Log2(cond)+96)))
	h.d = extendedDare{A: A, B: B, Q: Q, R: R, S: crossTerm(opts), E: model}
	if h.sol, ok = h.d.model(h.prec, nil); !ok {
		return h, false
	}
	k, prev := xFrom(k0, h.prec), xMat{}
	for i := range steps {
		y, ok := h.sol.hewer(k)
		if !ok {
			return h, false
		}
		k1, ok := h.d.gain(h.sol, y)
		if !ok {
			return h, false
		}
		prev, k = k, k1
		h.sol.y, h.sol.k = y, k1
		if i >= 1 && xRel(k, prev) <= newtonTol {
			break
		}
	}
	if !(xRel(k, prev) <= newtonTol) {
		return h, false
	}
	h.K = k.dense()
	if !stableLoop(h.d.closedLoop(h.sol, h.K)) {
		return h, false
	}
	h.eig = make([]complex128, len(res.Eig))
	if !closedLoopStable(A, B, h.K, E, h.eig) {
		return h, false
	}
	return h, true
}
