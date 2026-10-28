package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// hewerGainPrecMin is the least precision of hewerGainSubstituted, which
// carries log2 of the gain's condition plus 96 bits: at 106 bits the
// Hewer gain stalled 8e-9 off the 320-bit one, at 160 it matched it.
const hewerGainPrecMin = 128

// hewerGainStepsMax is the most Hewer steps hewerGainRef takes:
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

// hewerGainRef is the gain hewerGain forms, with reduced inputs, from the
// X of the result res gainOfX was given, against which contradicts
// measures the K of each result gainOfX would return. There the certified
// float64 X wanders within cond(E)-amplified rounding, 1e-11 to 1e-9 off
// over successive Newton steps against an eps band of 3e-15, and a
// cheap-control gain amplifies that by 1e5 to 1e6: the gain of X was
// 1.2e-4 to 1.6e-3 off the 320-bit gain (band 1.3e-9), the Hewer gain
// from either X within 1e-12 of it. The precision also carries
// the condition of the gain in the given inputs: with duplicated inputs
// that was 1.9e33 against under 10 in the reduced ones, and at 128 bits
// the Hewer gain converged 4.9e-6 off the 320-bit one.
type hewerGainRef struct {
	A, B, Q, R *mat.Dense
	opts       *RiccatiOpts
	red        *reducedInputs
	res        *RiccatiResult
	h          hewerGainSol
	tried      bool
	have       bool
	band       float64
}

// newHewerGainRef returns the reference of the Dare whose result res
// enters gainOfX, never contradicting without reduced inputs.
func newHewerGainRef(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) *hewerGainRef {
	return &hewerGainRef{A: A, B: B, Q: Q, R: R, opts: opts, red: red, res: res, band: -1}
}

// contradicts reports whether the reference contradicts the K of r: its
// relative distance from the reference gain exceeds gainFloor and 4√N
// times the largest move of that gain under two componentwise eps
// perturbations of the N entries of E, A, B, Q, R and S (perturbedMove),
// a lower estimate of K's band that √N lifts to an upper one: where that
// band is wide the gain of X is as good as any (K 1.4e-6 off the Hewer
// gain had a band of 9.8e-6). K is checked only when the gain of X may be
// off by gainFloor: when its move across a Newton step of X (move) or
// under X∘(1 ± eps) (tol/8, gainMissOf) exceeds gainFloor, or when res
// entered at Accuracy 1, which an X refined to Accuracy 0 may leave only
// with a K the reference confirms (Y6BJEX am.7). Where neither moved by
// gainFloor (by at most 1e-15, K band 3e5 to 2e11) the reference was
// itself 2e-9 to 7e-3 off and the check doubled the time of the Dare.
// A reference that fails to converge, or whose band cannot be formed,
// contradicts (fails closed): under -race the gain of an X refined from
// Accuracy 1 was 0.44 off the 320-bit gain (band 2.1e-5) while hewerGain
// failed.
func (g *hewerGainRef) contradicts(r *RiccatiResult, move, tol float64) bool {
	if g.red == nil || g.res.Accuracy == 0 && move <= gainFloor && tol <= 8*gainFloor {
		return false
	}
	if !g.tried {
		g.tried = true
		_, cond, ok := ddGain(g.A, g.B, g.R, crossTerm(g.opts), g.res.X)
		if !ok {
			cond = math.Inf(1)
		}
		g.h, g.have = hewerGain(g.A, g.B, g.Q, g.R, g.opts, g.red, g.res, hewerGainStepsMax, cond)
	}
	if !g.have {
		return true
	}
	var diff mat.Dense
	diff.Sub(g.h.K, r.K)
	miss := mat.Norm(&diff, 1) / mat.Norm(g.h.K, 1)
	if !(miss > gainFloor) {
		return false
	}
	if g.band < 0 {
		g.band = math.NaN()
		h := g.h
		x, ok := h.sol.xOf(h.sol.y)
		if !ok {
			return true
		}
		kw, ok := h.d.noiseGain(h.sol, h.sol.y)
		if !ok {
			return true
		}
		move := 0.0
		for seed := range uint64(2) {
			mv, ok := h.d.perturbedMove(h.sol, x, kw, h.prec, seed)
			if !ok {
				return true
			}
			move = max(move, mv.k)
		}
		n, m := g.B.Dims()
		entries := 3*n*n + n*m + m*m
		if g.opts != nil && g.opts.S != nil {
			entries += n * m
		}
		g.band = 4 * math.Sqrt(float64(entries)) * move
	}
	return !(miss <= g.band)
}

// substituted returns res with K the reference gain and Eig that of
// A − BK, X kept, or nil without a reference. gainOfX returns it for
// Accuracy 1 when the reference contradicts every K it would return.
func (g *hewerGainRef) substituted() *RiccatiResult {
	if !g.have {
		return nil
	}
	c := *g.res
	c.K, c.Eig = g.h.K, g.h.eig
	return &c
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
