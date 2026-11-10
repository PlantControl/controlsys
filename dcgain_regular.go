package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// dcRegularSolve is the regular DC solve of P = x0·I − A, P·X = B, with an
// a posteriori forward-error certificate (HUUZRO).
type dcRegularSolve struct {
	// g is D + C·X from the refined X, rounded once from double-double.
	g *mat.Dense
	// cond is the LU condition estimate of the double P.
	cond float64
	// solveErr is the LU solve's failure (an ill-conditioned or exactly
	// singular double P).
	solveErr error
	// overflow reports a non-finite X or gain.
	overflow bool
	// certified reports that every entry of g lies within
	// ε·(|D_ij| + Σ_l |C_il||X_lj|) + ½ulp of the exact gain.
	certified bool
	// steps counts the refinement steps taken; stalled reports that the
	// residual stopped halving, provable whether β ≥ 1 or a term below the
	// double-double range made the bound unavailable.
	steps            int
	stalled, unbound bool
}

// dcUnit is the unit roundoff u = 2⁻⁵³; dcGamma(k) is γ_k = k·u/(1 − k·u).
const dcUnit = 0x1p-53

func dcGamma(k int) float64 { return float64(k) * dcUnit / (1 - float64(k)*dcUnit) }

// dcProductFloor is the smallest nonzero product the certificate admits:
// below MinNormal/u² the low word of a double-double product, or the error
// of a double one, leaves the normal range and the u-based bounds fail.
const dcProductFloor = 0x1p-1022 / (dcUnit * dcUnit)

// solveDCRegular solves P·X = B with the double LU of P = x0·I − A and
// certifies G = D + C·X entry by entry. P⁻¹ is bounded rigorously from an
// approximate inverse Rinv and the Neumann series:
//
//	β = (‖I − fl(P·Rinv)‖∞ + (γ_n‖P‖∞ + u·max|P_ii|·[x0 ≠ 0])‖Rinv‖∞)(1 + γ_{n+4}),
//	ν = ‖Rinv‖∞/(1 − β)·(1 + γ_{n+4}) ≥ ‖P⁻¹‖∞ for β < 1,
//
// the u·max|P_ii| term covering the rounded diagonal x0 − a_ii. The
// residual R = B − P·X is formed in double-double with P exact, so
//
//	‖X_j − X*_j‖∞ ≤ η_j = ν·max_i(|R_ij| + (3n+7)u²·(|B_ij| + Σ_k|P_ik||X_kj|)),
//
// and G_ij is certified when ‖C_i‖₁·η_j ≤ τ_ij·(1 − 4(n+1)u),
// τ_ij = ε·(|D_ij| + Σ_l|C_il||X_lj|): the last factor absorbs the
// double-double sum forming G_ij, so the once-rounded entry is within
// τ_ij + ½ulp of the exact gain. A factor (1 + γ_{3n+8}) on the left
// covers the double rounding of the bound and τ themselves. Uncertified, X
// is refined in double-double on the same LU until every column certifies
// or one stops halving its residual bound (LAPACK dgerfs's test).
//
// β ≥ 1, a non-finite bound, or a nonzero product below dcProductFloor
// leaves the row unbound; with refine unset it then returns at once.
func solveDCRegular(A, B, C, D *mat.Dense, x0 float64, refine bool) dcRegularSolve {
	n, m := B.Dims()
	p, _ := C.Dims()
	ar, br, cr := A.RawMatrix(), B.RawMatrix(), C.RawMatrix()
	pd := make([]float64, n*n)
	plo := make([]float64, n)
	pdiag := 0.0
	for i := range n {
		for j := range n {
			pd[i*n+j] = -ar.Data[i*ar.Stride+j]
		}
		if x0 != 0 {
			pd[i*n+i], plo[i] = twoSum(x0, -ar.Data[i*ar.Stride+i])
			pdiag = max(pdiag, math.Abs(pd[i*n+i]))
		}
	}
	P := mat.NewDense(n, n, pd)
	var lu mat.LU
	lu.Factorize(P)
	res := dcRegularSolve{cond: lu.Cond()}
	var X mat.Dense
	if err := lu.SolveTo(&X, false, B); err != nil {
		res.solveErr = err
		return res
	}
	if !allFiniteDense(&X) {
		res.overflow = true
		return res
	}

	nu, bound := dcInverseBound(&lu, P, pd, n, pdiag)
	res.unbound = !bound
	if !bound && !refine {
		return res
	}

	xr := X.RawMatrix()
	x := make([]dd, n*m)
	for i := range n {
		for j := range m {
			x[i*m+j] = dd{xr.Data[i*xr.Stride+j], 0}
		}
	}
	var dr blas64.General
	if D != nil {
		dr = D.RawMatrix()
	}
	cnorm := make([]float64, p)
	for i := range p {
		for _, v := range cr.Data[i*cr.Stride : i*cr.Stride+n] {
			cnorm[i] += math.Abs(v)
		}
	}
	rho := make([]float64, 2*m)
	rhoPrev := rho[m:]
	rho = rho[:m]
	colOK := make([]bool, m)
	rhs := mat.NewDense(n, m, nil)
	rr := rhs.RawMatrix()
	var delta mat.Dense
	resGamma := float64(3*n+7) * dcUnit * dcUnit
	tauFactor := 1 - 4*float64(n+1)*dcUnit
	eFactor := 1 + dcGamma(3*n+8)
	for step := 0; ; step++ {
		for j := range m {
			rho[j] = 0
		}
		finite := true
		for i := range n {
			prow := pd[i*n : i*n+n]
			for j := range m {
				bij := br.Data[i*br.Stride+j]
				s := dd{bij, 0}
				mag := math.Abs(bij)
				for k, v := range prow {
					if v == 0 {
						continue
					}
					pk := dd{v, 0}
					if k == i {
						pk.lo = plo[i]
					}
					xk := x[k*m+j]
					s = s.add(pk.mul(xk).neg())
					t := math.Abs(pk.hi) * (math.Abs(xk.hi) + math.Abs(xk.lo))
					if t != 0 && t < dcProductFloor {
						bound = false
					}
					mag += t
				}
				rr.Data[i*rr.Stride+j] = s.hi + s.lo
				v := math.Abs(s.hi) + math.Abs(s.lo) + resGamma*mag
				if math.IsNaN(v) || math.IsInf(v, 0) {
					finite = false
				}
				rho[j] = max(rho[j], v)
			}
		}
		if !finite {
			res.unbound = true
			break
		}
		certified := bound
		for j := range m {
			colOK[j] = bound
		}
		for i := 0; bound && i < p; i++ {
			crow := cr.Data[i*cr.Stride : i*cr.Stride+n]
			for j := range m {
				t := 0.0
				if D != nil {
					t = math.Abs(dr.Data[i*dr.Stride+j])
				}
				for l, c := range crow {
					xl := x[l*m+j]
					v := math.Abs(c) * (math.Abs(xl.hi) + math.Abs(xl.lo))
					if v != 0 && v < dcProductFloor {
						bound = false
					}
					t += v
				}
				tau := 0x1p-52 * t * tauFactor
				e := cnorm[i] * nu * rho[j] * eFactor
				ok := e <= tau && !math.IsInf(tau, 0) && (tau == 0 || tau >= 0x1p-1022)
				if e == 0 && cnorm[i] != 0 && rho[j] != 0 {
					ok = false
				}
				if !ok {
					certified, colOK[j] = false, false
				}
			}
		}
		if !bound {
			res.unbound = true
			if !refine {
				return res
			}
			certified = false
		}
		if certified {
			res.certified = true
			break
		}
		if step > 0 {
			stalled := false
			for j := range m {
				if !colOK[j] && rho[j] >= rhoPrev[j]/2 {
					stalled = true
				}
			}
			if stalled {
				res.stalled = true
				break
			}
		}
		copy(rhoPrev, rho)
		if lu.SolveTo(&delta, false, rhs) != nil {
			break
		}
		drr := delta.RawMatrix()
		for i := range n {
			for j := range m {
				x[i*m+j] = x[i*m+j].add(dd{drr.Data[i*drr.Stride+j], 0})
			}
		}
		res.steps++
	}
	if !res.certified && !refine {
		return res
	}
	g := mat.NewDense(p, m, nil)
	gr := g.RawMatrix()
	for i := range p {
		crow := cr.Data[i*cr.Stride : i*cr.Stride+n]
		for j := range m {
			var s dd
			if D != nil {
				s.hi = dr.Data[i*dr.Stride+j]
			}
			for l, c := range crow {
				s = s.add(dd{c, 0}.mul(x[l*m+j]))
			}
			gr.Data[i*gr.Stride+j] = s.hi + s.lo
		}
	}
	if !allFiniteDense(g) {
		res.overflow, res.certified = true, false
		return res
	}
	res.g = g
	return res
}

// dcInverseBound returns ν ≥ ‖P⁻¹‖∞ for the exact P whose rounded form pd
// the LU factored, from Rinv = LU⁻¹·I and F = fl(P·Rinv); ok is false where
// β ≥ 1, a bound is not finite, or a product of P and Rinv may lie below
// dcProductFloor.
func dcInverseBound(lu *mat.LU, P *mat.Dense, pd []float64, n int, pdiag float64) (nu float64, ok bool) {
	eye := mat.NewDense(n, n, nil)
	er := eye.RawMatrix()
	for i := range n {
		er.Data[i*er.Stride+i] = 1
	}
	var Rinv mat.Dense
	if lu.SolveTo(&Rinv, false, eye) != nil {
		return 0, false
	}
	ir := Rinv.RawMatrix()
	pmin, pnorm := math.Inf(1), 0.0
	for i := range n {
		s := 0.0
		for _, v := range pd[i*n : i*n+n] {
			a := math.Abs(v)
			s += a
			if a != 0 {
				pmin = min(pmin, a)
			}
		}
		pnorm = max(pnorm, s)
	}
	rmin, rnorm := math.Inf(1), 0.0
	for i := range n {
		s := 0.0
		for _, v := range ir.Data[i*ir.Stride : i*ir.Stride+n] {
			a := math.Abs(v)
			s += a
			if a != 0 {
				rmin = min(rmin, a)
			}
		}
		rnorm = max(rnorm, s)
	}
	if pmin*rmin < dcProductFloor {
		return 0, false
	}
	eye.Mul(P, &Rinv)
	resid := 0.0
	for i := range n {
		s := 0.0
		for j, v := range er.Data[i*er.Stride : i*er.Stride+n] {
			if i == j {
				v = 1 - v
			}
			s += math.Abs(v)
		}
		resid = max(resid, s)
	}
	g := 1 + dcGamma(n+4)
	beta := (resid + (dcGamma(n)*pnorm+dcUnit*pdiag)*rnorm) * g
	if !(beta < 1) {
		return 0, false
	}
	nu = rnorm / (1 - beta) * g
	if math.IsInf(nu, 0) || math.IsNaN(nu) {
		return 0, false
	}
	return nu, true
}
