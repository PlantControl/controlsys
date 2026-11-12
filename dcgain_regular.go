package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack/lapack64"
	"plantcontrol.org/v1/gonum/mat"
)

// dcRegularSolve is the regular DC solve of P = x0·E − A, P·X = B, with an
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
	// ε·(|D_ij| + Σ_l |C_il||X_lj|) + ½ulp of the exact gain, and that an
	// entry of 0 is exact: its error bound is 0.
	certified bool
	// steps counts the refinement steps taken; stalled reports that the
	// residual stopped halving; unbound that β ≥ 1, a non-finite bound or a
	// product below dcProductFloor left no bound.
	steps            int
	stalled, unbound bool
}

// dcUnit is the unit roundoff u = 2⁻⁵³; dcGamma(k) is γ_k = k·u/(1 − k·u).
const dcUnit = 0x1p-53

// dcUnderflow is the smallest subnormal 2⁻¹⁰⁷⁴: a product rounded in the
// subnormal range errs by up to ½·dcUnderflow absolutely, which no factor
// 1 + γ_k covers.
const dcUnderflow = 0x1p-1074

func dcGamma(k int) float64 { return float64(k) * dcUnit / (1 - float64(k)*dcUnit) }

// dcProductFloor is the smallest nonzero product the certificate admits:
// below MinNormal/u² the low word of a double-double product, or the error
// of a double one, leaves the normal range and the u-based bounds fail.
const dcProductFloor = 0x1p-1022 / (dcUnit * dcUnit)

// solveDCRegular solves P·X = B with the double LU of P = x0·E − A, E nil
// for the identity, and certifies G = D + C·X entry by entry. x0·E must be
// exact (x0 = 1 for a descriptor); P is rounded entry by entry, its low
// words plo carried into the residual (OECG2I), and u·max|P_ii| below is
// u·‖fl(P)‖∞ for E ≠ I. P⁻¹ is bounded rigorously from an
// approximate inverse Rinv and the Neumann series:
//
//	β = (‖I − fl(P·Rinv)‖∞ + (γ_n‖P‖∞ + u·max|P_ii|·[x0 ≠ 0])‖Rinv‖∞)(1 + γ_{n+4}),
//	ν = ‖Rinv‖∞/(1 − β)·(1 + γ_{n+4}) ≥ ‖P⁻¹‖∞ for β < 1,
//
// the u·max|P_ii| term covering the rounded diagonal x0 − a_ii. The
// residual R = B − P·X, P exact and X double-double, is a compensated dot
// product: s collects B and the products' high words through twoSum, which
// with the FMA product errors e is exact, and c sums in double the twoSum
// errors q, the e and the low-word products t; only c's 3n roundings and
// t's own err, so with S_ij = Σ_k(|q_k| + |e_k| + |t_k|),
//
//	‖X_j − X*_j‖∞ ≤ η_j = ν·max_i(|fl(s_ij + c_ij)|·(1 + γ₁) + γ_{6n+5}·S_ij),
//
// γ_{6n+5} covering c's recursive sum (γ_{3n}), t's rounding (γ₂), the
// dropped low·low product (u), and the double rounding of S itself.
// G_ij = D_ij + Σ_l C_il·X_lj is the same compensated sum, s' + c' with
// error terms summing to S'_ij, and is certified when
//
//	(‖C_i‖₁·η_j + γ_{6n+5}·S'_ij)·(1 + γ_{3n+10}) ≤ τ_ij = ε·(|D_ij| + Σ_l|C_il||X_lj|),
//
// so the once-rounded fl(s' + c') is within τ_ij + ½ulp of the exact gain.
// A rounded 0 is certified only with a bound of 0: within a nonzero bound it
// has no correct digit, and consumers read 0 as a zero gain (UP7H7O), so
// the refinement goes on until the sum resolves from 0 or stalls.
// The last factor covers the double rounding of the bound and τ
// themselves, and τ's own error ε·‖C_i‖₁·η_j against τ at the exact X
// (2u). An entry whose bound exceeds ε·(|D_ij| + ‖C_i‖₁·max_l|X_lj|), above
// τ_ij, skips the sum. After a refinement step X₁ = X₀ ⊕ δ, δ solving
// P·δ = r = fl(R₀) on the LU, the next residual is first bounded without
// forming it (Higham, Accuracy and Stability of Numerical Algorithms,
// Thm 9.4: the solve's backward error is at most γ_{3n}|L̂||Û|; the
// double-double add errs by at most 4u²|X₁|):
//
//	η_j = ν·(‖R₀_j − r_j‖∞ + γ_{3n}‖|L̂||Û||δ_j|‖∞ + ‖(P − fl(P))δ_j‖∞ + 4u²‖P‖∞‖X₁_j‖∞)·(1 + γ_{2n+6}),
//
// ‖R₀ − r‖ ≤ u|r| + γ_{6n+5}·S from the residual above, the last factor
// covering the bound's own rounding; only where that fails to certify is
// R₁ formed. Uncertified, X is refined in double-double on the same LU until
// every column certifies or one stops halving its residual bound (LAPACK
// dgerfs's test).
//
// The factorization, its condition estimate and the solve are mat.LU's
// (Getrf, Gecon in the ∞-norm, Getrs), on raw storage the certificate
// reuses.
//
// The roundings of ρ_j (two products) and of ν·ρ_j may fall in the
// subnormal range, each erring by up to ½·dcUnderflow absolutely, so a
// column with a nonzero residual term takes
//
//	η_j = ν·ρ_j + ν·dcUnderflow + dcUnderflow,
//
// ν·dcUnderflow covering ρ_j's own two roundings scaled by ν, the last
// dcUnderflow those of ν·ρ_j and ν·dcUnderflow. An addition that rounds
// either away lost at most ½ulp(η), within the bound's own rounding; in
// the subnormal range the additions are exact. Without them an X
// that underflowed to 0 leaves ρ_j = |B_j| ≠ 0 while ν·ρ_j rounds to 0,
// and the zero gain passes with τ = 0. A one-step η whose sum of terms
// lies below dcProductFloor, or which lies below MinNormal itself, forms
// the residual instead: above MinNormal/u² the at most n² + n + 6
// subnormal product errors of the sum stay below u² of it, within
// 1 + γ_{2n+6}.
//
// β ≥ 1, a non-finite bound, or a nonzero product below dcProductFloor
// leaves the row unbound; with refine unset it then returns at once.
func solveDCRegular(E, A, B, C, D *mat.Dense, x0 float64, refine bool) dcRegularSolve {
	n, m := B.Dims()
	p, _ := C.Dims()
	ar, br, cr := A.RawMatrix(), B.RawMatrix(), C.RawMatrix()
	nn, nm := n*n, n*m
	full := E != nil && x0 != 0
	nlo := n
	if full {
		nlo = nn
	}
	slab := make([]float64, 4*nn+2*nm+5*n+nlo+p+5*m)
	take := func(k int) []float64 {
		v := slab[:k:k]
		slab = slab[k:]
		return v
	}
	pd, luf, rinv, f := take(nn), take(nn), take(nn), take(nn)
	xd, rhs := take(nm), take(nm)
	plo, work := take(nlo), take(4*n)
	cnorm, rho, rhoPrev, xmax := take(p), take(m), take(m), take(m)
	eta, rerr, w := take(m), take(m), take(n)
	ints := make([]int, 2*n)
	ipiv, iwork := ints[:n:n], ints[n:]

	pround := 0.0
	for i := range n {
		for j := range n {
			pd[i*n+j] = -ar.Data[i*ar.Stride+j]
		}
		switch {
		case full:
			er := E.RawMatrix()
			rs := 0.0
			for j := range n {
				pd[i*n+j], plo[i*n+j] = twoSum(x0*er.Data[i*er.Stride+j], -ar.Data[i*ar.Stride+j])
				rs += math.Abs(pd[i*n+j])
			}
			pround = max(pround, rs)
		case x0 != 0:
			pd[i*n+i], plo[i] = twoSum(x0, -ar.Data[i*ar.Stride+i])
			pround = max(pround, math.Abs(pd[i*n+i]))
		}
	}
	copy(luf, pd)
	P := blas64.General{Rows: n, Cols: n, Stride: n, Data: pd}
	LU := blas64.General{Rows: n, Cols: n, Stride: n, Data: luf}
	anorm := lapack64.Lange(mat.CondNorm, LU, work)
	ok := lapack64.Getrf(LU, ipiv)
	res := dcRegularSolve{cond: 1 / lapack64.Gecon(mat.CondNorm, LU, anorm, work, iwork)}
	if !ok {
		res.solveErr = mat.Condition(math.Inf(1))
		return res
	}
	if res.cond > mat.ConditionTolerance {
		res.solveErr = mat.Condition(res.cond)
		return res
	}
	X := blas64.General{Rows: n, Cols: m, Stride: m, Data: xd}
	for i := range n {
		copy(xd[i*m:i*m+m], br.Data[i*br.Stride:i*br.Stride+m])
	}
	lapack64.Getrs(blas.NoTrans, LU, X, ipiv)
	for _, v := range xd {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			res.overflow = true
			return res
		}
	}

	nu, pnorm, bound := dcInverseBound(P, LU, ipiv, rinv, f, work, pround)
	res.unbound = !bound
	if !bound && !refine {
		return res
	}

	x := make([]dd, nm)
	for k, v := range xd {
		x[k] = dd{v, 0}
	}
	var dr blas64.General
	if D != nil {
		dr = D.RawMatrix()
	}
	for i := range p {
		for _, v := range cr.Data[i*cr.Stride : i*cr.Stride+n] {
			cnorm[i] += math.Abs(v)
		}
	}
	flags := make([]bool, 2*m)
	colOK, live := flags[:m], flags[m:]
	R := blas64.General{Rows: n, Cols: m, Stride: m, Data: rhs}
	g := mat.NewDense(p, m, nil)
	gr := g.RawMatrix()
	resGamma := dcGamma(6*n + 5)
	rFactor := 1 + dcGamma(1)
	eFactor := 1 + dcGamma(3*n+10)
	tauFactor := 1 + dcGamma(2*n+4)
	lnGamma, onestepFactor := dcGamma(3*n), 1+dcGamma(2*n+6)
	plomax := 0.0
	for i := range n {
		if !full {
			plomax = max(plomax, math.Abs(plo[i]))
			continue
		}
		rs := 0.0
		for _, v := range plo[i*n : i*n+n] {
			rs += math.Abs(v)
		}
		plomax = max(plomax, rs)
	}
	pexact := (pnorm + plomax) * (1 + dcGamma(1))
	onestep := false
	for step := 0; ; step++ {
		finite := true
		for j := 0; !onestep && j < m; j++ {
			rho[j], rerr[j], live[j] = 0, 0, false
		}
		for i := 0; !onestep && i < n; i++ {
			prow := pd[i*n : i*n+n]
			for j := range m {
				s, c, cabs := br.Data[i*br.Stride+j], 0.0, 0.0
				for k, v := range prow {
					if v == 0 {
						continue
					}
					xk := x[k*m+j]
					pr := float64(-v * xk.hi)
					if pr != 0 && math.Abs(pr) < dcProductFloor {
						bound = false
					}
					e := math.FMA(-v, xk.hi, -pr)
					var q float64
					s, q = twoSum(s, pr)
					t := -v * xk.lo
					if full {
						t -= plo[i*n+k] * xk.hi
					} else if k == i {
						t -= plo[i] * xk.hi
					}
					c += q
					c += e
					c += t
					cabs += math.Abs(q) + math.Abs(e) + math.Abs(t)
				}
				r := s + c
				rhs[i*m+j] = r
				v := math.Abs(r)*rFactor + resGamma*cabs
				if math.IsNaN(v) || math.IsInf(v, 0) {
					finite = false
				}
				rho[j] = max(rho[j], v)
				live[j] = live[j] || r != 0 || cabs != 0
				rerr[j] = max(rerr[j], dcUnit*math.Abs(r)+resGamma*cabs)
			}
		}
		for j := 0; !onestep && j < m; j++ {
			eta[j] = nu * rho[j]
			if live[j] {
				eta[j] += nu*dcUnderflow + dcUnderflow
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
		for j := range m {
			xmax[j] = 0
			for l := range n {
				xmax[j] = max(xmax[j], math.Abs(x[l*m+j].hi)+math.Abs(x[l*m+j].lo))
			}
		}
		for i := 0; bound && i < p; i++ {
			crow := cr.Data[i*cr.Stride : i*cr.Stride+n]
			for j := range m {
				d := 0.0
				if D != nil {
					d = math.Abs(dr.Data[i*dr.Stride+j])
				}
				if cnorm[i]*eta[j] > 0x1p-52*(d+cnorm[i]*xmax[j])*tauFactor {
					certified, colOK[j] = false, false
					continue
				}
				gs, gc, gabs, t := 0.0, 0.0, 0.0, 0.0
				if D != nil {
					gs = dr.Data[i*dr.Stride+j]
					t = math.Abs(gs)
				}
				for l, cv := range crow {
					xl := x[l*m+j]
					pr := float64(cv * xl.hi)
					if pr != 0 && math.Abs(pr) < dcProductFloor {
						bound = false
					}
					e := math.FMA(cv, xl.hi, -pr)
					var q float64
					gs, q = twoSum(gs, pr)
					tl := cv * xl.lo
					gc += q
					gc += e
					gc += tl
					gabs += math.Abs(q) + math.Abs(e) + math.Abs(tl)
					t += math.Abs(pr) + math.Abs(tl)
				}
				gr.Data[i*gr.Stride+j] = gs + gc
				tau := 0x1p-52 * t
				e := (cnorm[i]*eta[j] + resGamma*gabs) * eFactor
				ok := e <= tau && !math.IsInf(tau, 0) && (tau == 0 || tau >= 0x1p-1022)
				if e == 0 && cnorm[i] != 0 && eta[j] != 0 {
					ok = false
				}
				if gr.Data[i*gr.Stride+j] == 0 && e > 0 {
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
		if onestep {
			onestep = false
			continue
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
		lapack64.Getrs(blas.NoTrans, LU, R, ipiv)
		for k, v := range rhs {
			x[k] = x[k].add(dd{v, 0})
		}
		res.steps++
		onestep = bound
		for j := 0; onestep && j < m; j++ {
			lud, plod, xm := 0.0, 0.0, 0.0
			for i := n - 1; i >= 0; i-- {
				t := 0.0
				for k := i; k < n; k++ {
					t += math.Abs(luf[i*n+k]) * math.Abs(rhs[k*m+j])
				}
				w[i] = t
			}
			for i := range n {
				t := w[i]
				for k := range i {
					t += math.Abs(luf[i*n+k]) * w[k]
				}
				lud = max(lud, t)
				if full {
					t = 0
					for k, v := range plo[i*n : i*n+n] {
						t += math.Abs(v * rhs[k*m+j])
					}
					plod = max(plod, t)
				} else {
					plod = max(plod, math.Abs(plo[i]*rhs[i*m+j]))
				}
				xm = max(xm, math.Abs(x[i*m+j].hi)+math.Abs(x[i*m+j].lo))
			}
			r := rerr[j] + lnGamma*lud + plod + pexact*4*dcUnit*dcUnit*xm
			eta[j] = nu * r * onestepFactor
			onestep = !math.IsNaN(eta[j]) && !math.IsInf(eta[j], 0) && r >= dcProductFloor && eta[j] >= 0x1p-1022
		}
	}
	if !res.certified && !refine {
		return res
	}
	for i := 0; !res.certified && i < p; i++ {
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

// dcInverseBound returns ν ≥ ‖P⁻¹‖∞ for the exact P whose rounded form P
// the factors LU (with ipiv) are of, from Rinv = LU⁻¹ (in rinv) and
// F = fl(P·Rinv) (in f); ok is false where β ≥ 1, a bound is not finite, or
// a product of P and Rinv may lie below dcProductFloor. u·pround bounds
// ‖P − fl(P)‖∞.
func dcInverseBound(P, LU blas64.General, ipiv []int, rinv, f, work []float64, pround float64) (nu, pnorm float64, ok bool) {
	n := P.Rows
	copy(rinv, LU.Data)
	Rinv := blas64.General{Rows: n, Cols: n, Stride: n, Data: rinv}
	if !lapack64.Getri(Rinv, ipiv, work, len(work)) {
		return 0, 0, false
	}
	pmin := math.Inf(1)
	rmin, rnorm := math.Inf(1), 0.0
	for i := range n {
		ps, rs := 0.0, 0.0
		for j := range n {
			a, b := math.Abs(P.Data[i*n+j]), math.Abs(rinv[i*n+j])
			ps += a
			rs += b
			if a != 0 {
				pmin = min(pmin, a)
			}
			if b != 0 {
				rmin = min(rmin, b)
			}
		}
		pnorm, rnorm = max(pnorm, ps), max(rnorm, rs)
	}
	if pmin*rmin < dcProductFloor {
		return 0, 0, false
	}
	resid := 0.0
	for i := range n {
		fi := f[i*n : i*n+n]
		clear(fi)
		for k, a := range P.Data[i*n : i*n+n] {
			if a == 0 {
				continue
			}
			for j, b := range rinv[k*n : k*n+n] {
				fi[j] += a * b
			}
		}
		s := 0.0
		for j, v := range fi {
			if i == j {
				v = 1 - v
			}
			s += math.Abs(v)
		}
		resid = max(resid, s)
	}
	g := 1 + dcGamma(n+4)
	beta := (resid + (dcGamma(n)*pnorm+dcUnit*pround)*rnorm) * g
	if !(beta < 1) {
		return 0, 0, false
	}
	nu = rnorm / (1 - beta) * g
	if math.IsInf(nu, 0) || math.IsNaN(nu) {
		return 0, 0, false
	}
	return nu, pnorm, true
}
