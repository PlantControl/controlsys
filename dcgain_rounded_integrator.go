package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// dcGainByRoundedIntegrator resolves a continuous explicit model, n ≤
// exactDetMaxOrder, whose A has one real eigenvalue λ the data do not tell
// from 0: det(sI − A), in exact arithmetic, changes sign on [−c, c] for c
// the first-order bound ε·|y|ᵀ|A||x|/|yᴴx| on its move under componentwise
// rounding of A (poleEigen.movement), and every other computed pole lies
// beyond c plus its own move under that rounding and the eigensolver's
// normwise backward error η = nε‖A‖_F. The computed λ can be far off the
// float model's own: the eigensolver's error is of order η, not ε|A|. As
// MATLAB dcgain takes A singular to working precision for an integrator,
// and EvalFr(0) such a pole for a pole at DC, each entry is then ±∞ by the
// sign of its residue R = (Cx)(yᴴB)/(yᴴx) for right and left eigenvectors
// x, y; the exact limit of the float data, R/(−λ) plus the rest, rests on
// the rounding that put λ off 0. ok is false unless every |R_ij| exceeds
// its first-order bound under the same perturbations of A and
// componentwise rounding of B and C: an entry the integrator may not
// reach, or eigenvectors too ill-conditioned to tell, is left to the
// transfer limit.
func (sys *System) dcGainByRoundedIntegrator() (*mat.Dense, bool) {
	if !sys.IsContinuous() {
		return nil, false
	}
	n, m, p := sys.Dims()
	if n > exactDetMaxOrder {
		return nil, false
	}
	pe, ok := newPoleEigen(sys.A)
	if !ok {
		return nil, false
	}
	e := eps()
	en := float64(n) * e
	eta := en * denseNorm(sys.A)
	vecNorm := func(v *mat.CDense, j int) float64 {
		s := 0.0
		for i := range n {
			s += cmplx.Abs(v.At(i, j)) * cmplx.Abs(v.At(i, j))
		}
		return math.Sqrt(s)
	}
	k := -1
	for j, lam := range pe.poles {
		if imag(lam) == 0 && (k < 0 || cmplx.Abs(lam) < cmplx.Abs(pe.poles[k])) {
			k = j
		}
	}
	if k < 0 {
		return nil, false
	}
	c := pe.movement(sys.A, k)
	for j, lam := range pe.poles {
		if j == k {
			continue
		}
		_, yx := pe.coupling(sys.A, j, j)
		if !(cmplx.Abs(lam) > c+pe.movement(sys.A, j)+eta*vecNorm(&pe.x, j)*vecNorm(&pe.y, j)/cmplx.Abs(yx)) {
			return nil, false
		}
	}
	if !(c > 0) || exactCharPolySign(sys.A, -c)*exactCharPolySign(sys.A, c) > 0 {
		return nil, false
	}
	cRaw, bRaw := sys.C.RawMatrix(), sys.B.RawMatrix()
	cx := func(i, j int) (v complex128, mag float64) {
		for l, c := range cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+n] {
			v += complex(c, 0) * pe.x.At(l, j)
			mag += math.Abs(c) * cmplx.Abs(pe.x.At(l, j))
		}
		return v, mag
	}
	yb := func(j, col int) (v complex128, mag float64) {
		for l := range n {
			b := bRaw.Data[l*bRaw.Stride+col]
			v += cmplx.Conj(pe.y.At(l, j)) * complex(b, 0)
			mag += cmplx.Abs(pe.y.At(l, j)) * math.Abs(b)
		}
		return v, mag
	}
	_, sk := pe.coupling(sys.A, k, k)
	xk, yk := vecNorm(&pe.x, k), vecNorm(&pe.y, k)
	cxk, dcx := make([]float64, p), make([]float64, p)
	for i := range p {
		v, mag := cx(i, k)
		cxk[i], dcx[i] = real(v), en*mag
	}
	ybk, dyb := make([]float64, m), make([]float64, m)
	for l := range m {
		v, mag := yb(k, l)
		ybk[l], dyb[l] = real(v), en*mag
	}
	for j, lam := range pe.poles {
		if j == k {
			continue
		}
		_, sj := pe.coupling(sys.A, j, j)
		d := cmplx.Abs(pe.poles[k]-lam) * cmplx.Abs(sj)
		if !(d > 0) {
			return nil, false
		}
		xj, yj := vecNorm(&pe.x, j), vecNorm(&pe.y, j)
		yaxJK, _ := pe.coupling(sys.A, j, k)
		yaxKJ, _ := pe.coupling(sys.A, k, j)
		toX := (e*yaxJK + eta*yj*xk) / d
		toY := (e*yaxKJ + eta*yk*xj) / d
		for i := range p {
			v, _ := cx(i, j)
			dcx[i] += cmplx.Abs(v) * toX
		}
		for l := range m {
			v, _ := yb(j, l)
			dyb[l] += cmplx.Abs(v) * toY
		}
	}
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for l := range m {
			r := cxk[i] * ybk[l]
			bound := dcx[i]*math.Abs(ybk[l]) + math.Abs(cxk[i])*dyb[l] + dcx[i]*dyb[l]
			if !(math.Abs(r) > bound) {
				return nil, false
			}
			gain.Set(i, l, math.Inf(signInt(r/real(sk))))
		}
	}
	return gain, true
}

// exactCharPolySign returns the sign of det(s·I − A) in exact arithmetic.
func exactCharPolySign(A *mat.Dense, s float64) int {
	raw := A.RawMatrix()
	n := raw.Rows
	M := make([][]*big.Rat, n)
	for i := range n {
		M[i] = make([]*big.Rat, n)
		for j := range n {
			M[i][j] = new(big.Rat).SetFloat64(-raw.Data[i*raw.Stride+j])
		}
		M[i][i].Add(M[i][i], new(big.Rat).SetFloat64(s))
	}
	return ratDet(M).Sign()
}
