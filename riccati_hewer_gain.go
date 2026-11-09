package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// hewerGainPrecMin is the least precision of hewerGainSubstituted, which
// carries log2 of the gain's condition plus 96 bits: at 106 bits the
// Hewer gain stalled 8e-9 off the 320-bit one, at 160 it matched it.
const hewerGainPrecMin = 128

// hewerGainSubstituted returns res with K the gain of X taken two Hewer
// steps on in extended precision (extendedDare's model, E unrounded),
// starting from the gain gainMissIn measures K against, and Eig that of
// A − BK, once the second step moves K by at most newtonTol, A − BK is
// shown stable in that precision and in double-double (closedLoopStable),
// and X is within newtonTol of the iterate the new K is the gain of. X is
// kept. With reduced inputs under cheap control both the float64 pencil K
// and the double-double gain of the rounded X were off the 320-bit gain by
// up to 6e-6 and 4.7e-6, either one the worse depending on the rounding of
// X, within the gain's move under X∘(1 ± eps); the gain of the extended
// Hewer iterate was within 1e-10 of it. So K is held to the gain of its
// next iterate in this precision, not to the ddGain of the rounded X: with
// X 9.7e-11 off the 320-bit X, that ddGain was 1.2e-4 from a K within 2e-17
// of the 320-bit gain, at a tolerance of 1.7e-9.
func hewerGainSubstituted(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, bool) {
	return hewerGain(A, B, Q, R, opts, red, res, false, false)
}

// hewerGainConverged is hewerGainSubstituted with the Hewer sequence run on
// should its second step move K beyond newtonTol: K is taken at the first
// step from the second on that moves it by at most newtonTol, failing on a
// move from the third step on that does not shrink (Hewer is monotone from
// its first iterate; the first starts from the gain of X, which is not one)
// or after ceil(log2 prec) steps, enough for quadratic convergence from one
// correct bit to the whole precision. The two-step check lagged the
// sequence by a step: on cheap-control reduced inputs its moves were
// 1.7e-5, 8.1e-8, 2.7e-12, then 3e-21 (rv48 846), the third iterate
// matching the 320-bit gain where the K of X was 1.35e-5 off it (2.76e-3
// on rv49v 4080, Q(1 - 171 eps)).
func hewerGainConverged(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, bool) {
	return hewerGain(A, B, Q, R, opts, red, res, true, false)
}

// hewerSolution is hewerGainConverged run until Y converges with K and
// then one Hewer step on, X and EᵀXE (xe) replaced by that iterate's,
// rounded once, when res.X is beyond newtonTol of it, rather than
// failing: res.X need not be within newtonTol of the solution where
// newtonUnproven voids its certificate. Under cheap control K converged
// within newtonTol while Y was still 2.8e-4 off.
func hewerSolution(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, bool) {
	return hewerGain(A, B, Q, R, opts, red, res, true, true)
}

func hewerGain(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult, converge, takeX bool) (*RiccatiResult, bool) {
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
	prec := uint(min(extendedPrecMax, max(hewerGainPrecMin, math.Log2(cond)+96)))
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: crossTerm(opts), E: model}
	sol, ok := d.model(prec, nil)
	if !ok {
		return nil, false
	}
	steps := 2
	if converge {
		steps = int(math.Ceil(math.Log2(float64(prec))))
	}
	k, y, last := xFrom(k0, prec), xMat{}, math.Inf(1)
	for i := 0; ; i++ {
		if i == steps {
			return nil, false
		}
		yPrev := y
		if y, ok = sol.hewer(k); !ok {
			return nil, false
		}
		k1, ok := d.gain(sol, y)
		if !ok {
			return nil, false
		}
		prev := k
		k = k1
		if i == 0 {
			continue
		}
		move := xRel(k, prev)
		if move <= newtonTol {
			if !takeX || xRel(y, yPrev) <= newtonTol {
				break
			}
			last = move
			continue
		}
		if !converge || i >= 2 && !(move < last) {
			return nil, false
		}
		last = move
	}
	if takeX {
		if y, ok = sol.hewer(k); !ok {
			return nil, false
		}
		if k, ok = d.gain(sol, y); !ok {
			return nil, false
		}
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
	x, ok := sol.xOf(y)
	if !ok {
		return nil, false
	}
	if takeX && !(xRel(x, xFrom(res.X, prec)) <= newtonTol) {
		c.X, c.step, c.x1 = x.dense(), 0, nil
		if E != nil {
			c.xe = y.dense()
			symmetrize(c.xe.RawMatrix().Data, n, n)
		}
		symmetrize(c.X.RawMatrix().Data, n, n)
		c.xeStep, c.xeX, c.xeLoose = 0, 0, false
		return &c, true
	}
	if !(xRel(x, xFrom(res.X, prec)) <= newtonTol) {
		return nil, false
	}
	return &c, true
}
