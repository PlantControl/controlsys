package controlsys

import (
	"math"
	"math/bits"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// differenceBounds bounds ‖(I − D22·Z)⁻¹‖ for the difference operator
// I − D22·Δ(s) of a neutral delay loop, +Inf where nothing is certified.
// polydisk holds over every diagonal Z with |Zkk| <= 1, which certifies the
// strong stability sup_θ ρ(D22·diag(e^{jθ})) < 1 of Hale and Henry; rhp holds
// over Z = Δ(s), Re s >= 0, for the given delays; torus holds over |Zkk| = 1,
// which contains Δ(jω).
type differenceBounds struct{ polydisk, rhp, torus float64 }

// difference returns the bounds of e, computed once. The polydisk bound is
// the smallest of 1/(1 − ‖D22‖), ‖(I − |D22|)⁻¹‖ (Neumann series) and
// κ(S)/(1 − ‖S⁻¹D22S‖) for a positive diagonal S, which commutes with Z, so
// that (I − D22Z)⁻¹ = S(I − S⁻¹D22S·Z)⁻¹S⁻¹. Without it, equal delays make
// Z = zI, |z| <= 1, and any similarity S does, taken from the Stein equation
// P − D22ᵀPD22 = I, solvable when ρ(D22) < 1; commensurate delays
// τ_i = k_i·h reduce to that case through liftD22. On the torus an
// invertible D22 gives (I − D22Θ)⁻¹ = −(I − Θ⁻¹W)⁻¹Θ⁻¹W with W = D22⁻¹,
// bounded by κ(S)γ/(1 − γ) when γ = ‖S⁻¹WS‖ < 1, the anti-stable
// difference operator. When all of these fail for nd >= 4, where diagonal
// scaling is not exact, torusCellBound may still certify the polydisk.
func (e *delayLFT) difference() *differenceBounds {
	if e.diff != nil {
		return e.diff
	}
	inf := math.Inf(1)
	b := &differenceBounds{inf, inf, inf}
	e.diff = b
	d22 := e.sys.LFT.D22
	if d22 == nil || d22.IsEmpty() || e.d22 == 0 {
		b.polydisk, b.rhp, b.torus = 1, 1, 1
		return b
	}
	p := e.r0
	if e.d22 < 1 {
		p = min(p, 1/(1-e.d22))
	}
	if math.IsInf(p, 1) {
		p = diagonalScalingBound(d22, 0, false)
	}
	b.polydisk, b.rhp, b.torus = p, p, p
	if !math.IsInf(p, 1) {
		return b
	}
	if mult := e.multiples(); mult != nil {
		b.rhp = steinBound(liftD22(d22, mult))
	}
	var w, res mat.Dense
	if err := w.Inverse(d22); err == nil {
		res.Mul(d22, &w)
		if delta := residualNorm(&res, mat.Norm(d22, 2)*mat.Norm(&w, 2)); delta < 1 {
			b.torus = diagonalScalingBound(&w, mat.Norm(&w, 2)*delta/(1-delta), true)
		}
	}
	if e.nd >= 4 && e.nd <= maxCellDelays && math.IsInf(b.rhp, 1) && math.IsInf(b.torus, 1) {
		if c := torusCellBound(d22); !math.IsInf(c, 1) {
			b.polydisk, b.rhp, b.torus = c, c, c
		}
	}
	return b
}

const (
	// maxCellDelays caps nd for torusCellBound, whose box count grows
	// exponentially in nd; below 4 diagonal scaling is already exact.
	maxCellDelays = 8
	// maxCells caps the boxes torusCellBound examines.
	maxCells = 1 << 15
	// cellGamma is the scaled perturbation norm below which a box is
	// accepted, so that its resolvent bound stays within 1/(1 − cellGamma)
	// of κ(S)‖X‖.
	cellGamma = 0.75
)

// torusCellBound bounds ‖(I − D22Z)⁻¹‖ over the unit polydisk by branch and
// bound on the torus of phases θ, for the nd >= 4 where diagonal scaling
// (the μ upper bound) is not exact. Over a box |θ_k − c_k| <= δ_k,
// e^{jθ_k} = C_k + s_k·u_k, |u_k| <= 1, with C_k = cos δ_k·e^{jc_k},
// s_k = sin δ_k for δ_k <= π/2 (else C_k = 0, s_k = 1). With X ≈ B⁻¹,
// B = I − D22·C, N = X·D22·diag(s) and R = I − XB, X(I − D22Θ) =
// I − (R + N·U), so γ = κ(S)‖R‖ + ‖S⁻¹NS‖ < 1 for a positive diagonal S
// (Osborne-balanced on |N|) gives ‖(I − D22Θ)⁻¹‖ <= κ(S)‖X‖/(1 − γ).
// Boxes, dyadic in each phase so that centres carry one rounding, are
// inflated by it; those failing γ <= cellGamma are halved in the phase of
// the largest scaled column of N. D22 real makes the bound at Θ̄ that at Θ, so θ_1 ∈ [0, π]
// suffices. I − D22Θ invertible on the connected torus and ρ(D22) < 1
// (steinBound) keep every eigenvalue of D22Θ inside the unit circle, since
// a crossing e^{jα} would make I − D22·e^{−jα}Θ singular; so ρ(D22Z) < 1
// on the polydisk, and ‖(I − D22Z)⁻¹‖, subharmonic in each z_k, peaks on
// the torus. +Inf when the budget runs out or a box centre has
// ρ(D22Θ) >= 1.
func torusCellBound(d22 *mat.Dense) float64 {
	inf := math.Inf(1)
	n, _ := d22.Dims()
	if math.IsInf(steinBound(d22), 1) {
		return inf
	}
	m := 2 * n
	const eps = 0x1p-52
	d := d22.RawMatrix()
	dfrob := mat.Norm(d22, 2)
	be := mat.NewDense(m, m, nil)
	ne := mat.NewDense(m, m, nil)
	var xe, res mat.Dense
	bRaw, nRaw := be.RawMatrix().Data, ne.RawMatrix().Data
	cr, ci, s := make([]float64, n), make([]float64, n), make([]float64, n)
	x, ex, abs := make([]float64, n), make([]float64, n), make([]float64, n*n)
	bc, br := make([]float64, n), make([]float64, n)
	type box struct {
		idx []uint64
		lvl []int
	}
	stack := []box{{make([]uint64, n), make([]int, n)}}
	worst := 0.0
	for count := 0; len(stack) > 0; count++ {
		if count >= maxCells {
			return inf
		}
		bx := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for k := range n {
			span := 2 * math.Pi
			if k == 0 {
				span = math.Pi
			}
			if bx.lvl[k] > 40 {
				return inf
			}
			br[k] = math.Ldexp(span, -bx.lvl[k]-1)
			bc[k] = float64(2*bx.idx[k]+1) * br[k]
		}
		split, wide := -1, 0.0
		for k, r := range br {
			if r > math.Pi/2 && r > wide {
				split, wide = k, r
			}
		}
		if split < 0 {
			for k := range n {
				dk := br[k] + 8*eps*(math.Abs(bc[k])+math.Pi)
				sc, cc := math.Sincos(bc[k])
				if dk <= math.Pi/2 {
					sd, cd := math.Sincos(dk)
					cr[k], ci[k], s[k] = cd*cc, cd*sc, sd*(1+2*eps)+8*eps
				} else {
					cr[k], ci[k], s[k] = 0, 0, 1
				}
			}
			for i := range n {
				for j := range n {
					v := d.Data[i*d.Stride+j]
					re, im := -v*cr[j], -v*ci[j]
					if i == j {
						re++
					}
					bRaw[i*m+j], bRaw[(i+n)*m+j+n] = re, re
					bRaw[i*m+j+n], bRaw[(i+n)*m+j] = -im, im
				}
			}
			var g, kappa, xn float64
			ok := xe.Inverse(be) == nil
			if ok {
				res.Mul(&xe, be)
				xRaw := xe.RawMatrix().Data
				xfrob := mat.Norm(&xe, 2)
				delta := residualNorm(&res, xfrob*mat.Norm(be, 2))
				for i := range m {
					for j := range n {
						var p, q float64
						for k := range n {
							p += xRaw[i*m+k] * d.Data[k*d.Stride+j]
							q += xRaw[i*m+k+n] * d.Data[k*d.Stride+j]
						}
						nRaw[i*m+j], nRaw[i*m+j+n] = p*s[j], q*s[j]
					}
				}
				for i := range n {
					for j := range n {
						re, im := nRaw[i*m+j], nRaw[(i+n)*m+j]
						abs[i*n+j] = re*re + im*im
					}
				}
				osborne(abs, x, ex, n, 10)
				for i := range m {
					for j := range m {
						nRaw[i*m+j] *= ex[j%n] / ex[i%n]
					}
				}
				kappa = slices.Max(ex) / slices.Min(ex)
				xn = spectralNorm(&xe)
				rounding := 4 * float64(m) * eps * kappa * xfrob * (dfrob + 1) * (1 + slices.Max(s))
				g = (spectralNorm(ne) + kappa*delta + rounding) * (1 + similarityGap)
				ok = !math.IsInf(g, 0) && !math.IsNaN(g)
			}
			if ok && g <= cellGamma {
				worst = max(worst, kappa*xn/(1-g))
				continue
			}
			best := -1.0
			for j := range n {
				var c float64
				for i := range m {
					c += nRaw[i*m+j] * nRaw[i*m+j]
				}
				if !ok {
					c = br[j]
				}
				if c > best {
					split, best = j, c
				}
			}
			if slices.Max(br) <= math.Pi/16 && centreRadius(d22, bc) >= 1 {
				return inf
			}
		}
		for half := range uint64(2) {
			idx, lvl := slices.Clone(bx.idx), slices.Clone(bx.lvl)
			idx[split], lvl[split] = 2*idx[split]+half, lvl[split]+1
			stack = append(stack, box{idx, lvl})
		}
	}
	return worst
}

// osborne balances the n×n matrix of squared magnitudes a2, returning in x
// the log-scales that minimize the off-diagonal Frobenius norm of
// diag(e^{−x})·|A|·diag(e^x), and e^x in ex.
func osborne(a2, x, ex []float64, n, sweeps int) {
	for i := range n {
		x[i], ex[i] = 0, 1
	}
	for range sweeps {
		for i := range n {
			r, c := 0.0, 0.0
			for j := range n {
				if j != i {
					w := ex[j] / ex[i]
					r += a2[i*n+j] * w * w
					c += a2[j*n+i] / (w * w)
				}
			}
			if r > 0 && c > 0 {
				x[i] = math.Max(-30, math.Min(30, x[i]+0.25*math.Log(r/c)))
				ex[i] = math.Exp(x[i])
			}
		}
	}
}

// centreRadius returns the computed ρ(D22·diag(e^{jθ})), NaN on failure.
func centreRadius(d22 *mat.Dense, theta []float64) float64 {
	n, _ := d22.Dims()
	e := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			sn, cs := math.Sincos(theta[j])
			v := d22.At(i, j)
			e.Set(i, j, v*cs)
			e.Set(i+n, j+n, v*cs)
			e.Set(i, j+n, -v*sn)
			e.Set(i+n, j, v*sn)
		}
	}
	var eig mat.Eigen
	if !eig.Factorize(e, mat.EigenNone) {
		return math.NaN()
	}
	r := 0.0
	for _, v := range eig.Values(nil) {
		r = max(r, cmplx.Abs(v))
	}
	return r
}

// opNorm returns the largest singular value of m, +Inf when the SVD fails.
func opNorm(m mat.Matrix) float64 {
	sv, err := maxSingularValue(mat.DenseCopyOf(m))
	if err != nil {
		return math.Inf(1)
	}
	return sv
}

// residualNorm returns the Frobenius norm of I − r, an upper bound on its
// spectral norm, raised by the rounding of the product r of factors whose
// norms multiply to scale.
func residualNorm(r *mat.Dense, scale float64) float64 {
	n, _ := r.Dims()
	d := mat.DenseCopyOf(r)
	for i := range n {
		d.Set(i, i, d.At(i, i)-1)
	}
	return mat.Norm(d, 2) + 4*float64(n)*0x1p-52*scale
}

func equalDelays(tau []float64) bool {
	for _, t := range tau {
		if t != tau[0] {
			return false
		}
	}
	return true
}

// maxLiftedOrder caps Σk_i, the order of liftD22.
const maxLiftedOrder = 64

// commensurateTol is the relative gap |τ_i·l − τ_min·k_i| up to which
// commensurateMultiples takes τ_i/τ_min = k_i/l: a few ulps, covering
// fl(k·h) and decimal literals such as 0.1 and 0.3, while ratios p/q with
// q <= maxLiftedOrder stay at least 1/maxLiftedOrder² apart.
const commensurateTol = 8 * 0x1p-52

// commensurateMultiples returns integers k_i with τ_i = k_i·h for a common
// h > 0, up to floating-point rounding (commensurateTol), and
// Σk_i <= maxLiftedOrder; all ones for equal delays, nil otherwise. As in
// MATLAB, which evaluates the frequency response at the stored delays and
// never resolves the gap, delays that are integer multiples up to rounding
// are commensurate: the lifted model, its stability and its limit at
// ω = ∞ are those of the exact multiples, whose phases ω·τ_i differ from
// the stored ones by ω·|τ_i − k_i·h| <= ω·τ_i·commensurateTol, about 2e-9
// rad at ω·τ_i = 1e6, beyond the grids scanned. A ratio further from k_i/l
// is rejected: stability that is not strong is not robust to the delays.
func commensurateMultiples(tau []float64) []int {
	if len(tau) == 0 {
		return nil
	}
	if equalDelays(tau) {
		k := make([]int, len(tau))
		for i := range k {
			k[i] = 1
		}
		return k
	}
	tmin, tmax := slices.Min(tau), slices.Max(tau)
	if !(tmin >= 0x1p-900) || !(tmax <= 0x1p900) || tmax/tmin > maxLiftedOrder {
		return nil
	}
	for l := 1; l <= maxLiftedOrder; l++ {
		sum := 0
		for _, t := range tau {
			k := math.Round(t / tmin * float64(l))
			if !nearProductsEqual(t, float64(l), tmin, k) {
				sum = -1
				break
			}
			sum += int(k)
		}
		if sum > maxLiftedOrder {
			return nil
		}
		if sum > 0 {
			k := make([]int, len(tau))
			for i, t := range tau {
				k[i] = int(math.Round(t / tmin * float64(l)))
			}
			return k
		}
	}
	return nil
}

// commensurateClasses partitions the delays into classes of commensurate
// ones, τ_i = k_i·h_c up to rounding (see commensurateMultiples) for class
// c = grp[i], greedily in index order: τ_i joins the first class whose delays it extends to a set that
// commensurateMultiples accepts, else opens a new one. k holds the multiples
// within each class; dim is the number of classes. All delays form one class
// exactly when commensurateMultiples accepts the whole set.
func commensurateClasses(tau []float64) (grp, k []int, dim int) {
	nd := len(tau)
	grp, k = make([]int, nd), make([]int, nd)
	if mult := commensurateMultiples(tau); mult != nil {
		copy(k, mult)
		return grp, k, min(nd, 1)
	}
	buf := make([]float64, 0, nd)
	members := func(c, n int) []float64 {
		buf = buf[:0]
		for j, g := range grp[:n] {
			if g == c {
				buf = append(buf, tau[j])
			}
		}
		return buf
	}
	for i, t := range tau {
		grp[i] = dim
		for c := range dim {
			if commensurateMultiples(append(members(c, i), t)) != nil {
				grp[i] = c
				break
			}
		}
		if grp[i] == dim {
			dim++
		}
	}
	for c := range dim {
		mult := commensurateMultiples(members(c, nd))
		for j, i := 0, 0; i < nd; i++ {
			if grp[i] == c {
				k[i] = mult[j]
				j++
			}
		}
	}
	return grp, k, dim
}

// nearProductsEqual reports |a·b − c·d| <= commensurateTol·|c·d| for
// nonnegative normal products, a·b − fl(c·d) taken exactly by fma.
func nearProductsEqual(a, b, c, d float64) bool {
	q := c * d
	return math.Abs(math.FMA(a, b, -q)) <= commensurateTol*q
}

// liftD22 returns the companion lift M of order Σk_i with
// det(I − zM) = det(I − D22·diag(z^{k_i})), whose resolvent (I − zM)⁻¹ has
// (I − D22·diag(z^{k_i}))⁻¹ as its leading nd×nd block: state i holds x_i and
// the k_i − 1 delayed copies z^m·x_i, m < k_i, after it.
func liftD22(d22 *mat.Dense, k []int) *mat.Dense {
	nd := len(k)
	n, last := 0, make([]int, nd)
	for _, ki := range k {
		n += ki
	}
	m := mat.NewDense(n, n, nil)
	row := nd
	for j, kj := range k {
		last[j] = j
		for range kj - 1 {
			m.Set(row, last[j], 1)
			last[j] = row
			row++
		}
	}
	for i := range nd {
		for j := range nd {
			m.Set(i, last[j], d22.At(i, j))
		}
	}
	return m
}

// similarityGap is the relative margin on a computed norm γ before γ < 1 is
// trusted.
const similarityGap = 1e-10

// resolventFromSimilarity returns κ/(1 − γ), or κγ/(1 − γ) for the inverse
// form, when γ < 1.
func resolventFromSimilarity(kappa, gamma float64, inverse bool) float64 {
	gamma *= 1 + similarityGap
	if !(gamma < 1) || math.IsInf(kappa, 0) || math.IsNaN(kappa) {
		return math.Inf(1)
	}
	if inverse {
		return kappa * gamma / (1 - gamma)
	}
	return kappa / (1 - gamma)
}

// diagonalScalingBound minimizes ‖S⁻¹MS‖ over positive diagonal S =
// diag(e^x), a convex function of x, by Osborne balancing and coordinate
// golden-section descent, and returns the resulting resolvent bound for a
// matrix within err of m: ‖S⁻¹(M + E)S‖ <= ‖S⁻¹MS‖ + κ(S)‖E‖. A computed
// inverse W̃ of D22 has ‖W − W̃‖ <= ‖W̃‖δ/(1 − δ), δ = ‖I − D22W̃‖, since
// W = W̃(D22W̃)⁻¹.
func diagonalScalingBound(m *mat.Dense, err float64, inverse bool) float64 {
	n, _ := m.Dims()
	src := mat.DenseCopyOf(m).RawMatrix().Data
	t := mat.NewDense(n, n, nil)
	raw := t.RawMatrix().Data
	x := make([]float64, n)
	norm := func() float64 {
		for i := range n {
			for j := range n {
				raw[i*n+j] = src[i*n+j] * math.Exp(x[j]-x[i])
			}
		}
		return opNorm(t)
	}
	for range 20 {
		for i := range n {
			r, c := 0.0, 0.0
			for j := range n {
				if j != i {
					r += math.Pow(src[i*n+j]*math.Exp(x[j]-x[i]), 2)
					c += math.Pow(src[j*n+i]*math.Exp(x[i]-x[j]), 2)
				}
			}
			if r > 0 && c > 0 {
				x[i] = math.Max(-30, math.Min(30, x[i]+0.25*math.Log(r/c)))
			}
		}
	}
	best := norm()
	for range 30 {
		prev := best
		for i := range n {
			xi := x[i]
			at := func(v float64) float64 {
				x[i] = v
				return -norm()
			}
			v, g := goldenMax(max(xi-4, -30), min(xi+4, 30), xi, -best, at)
			x[i] = v
			best = -g
		}
		if prev-best <= 1e-9*prev {
			break
		}
	}
	best = norm()
	kappa := math.Exp(slices.Max(x) - slices.Min(x))
	return resolventFromSimilarity(kappa, best+kappa*err, inverse)
}

// steinBound returns κ(S)/(1 − ‖S⁻¹MS‖) with S = L⁻ᵀ from the Cholesky
// factor P = LLᵀ of P = Σ (Mᵀ)ᵏMᵏ, which satisfies MᵀPM = P − I, so that
// ‖LᵀML⁻ᵀ‖ < 1. Any invertible S gives a valid bound, so P need not be
// exact. With X the computed L⁻¹ and δ = ‖I − LᵀXᵀ‖ < 1, L⁻ᵀ =
// Xᵀ(LᵀXᵀ)⁻¹ gives ‖LᵀML⁻ᵀ‖ <= ‖LᵀMXᵀ‖/(1 − δ) and ‖L⁻ᵀ‖ <= ‖X‖/(1 − δ);
// the computed product norm is raised by its rounding.
func steinBound(m *mat.Dense) float64 {
	n, _ := m.Dims()
	p := mat.NewDense(n, n, nil)
	for i := range n {
		p.Set(i, i, 1)
	}
	ak := mat.DenseCopyOf(m)
	var tmp, next mat.Dense
	for range 64 {
		tmp.Mul(ak.T(), p)
		next.Mul(&tmp, ak)
		p.Add(p, &next)
		next.Mul(ak, ak)
		ak.Copy(&next)
		s := mat.Norm(ak, 2)
		if math.IsInf(s, 0) || math.IsNaN(s) || !(mat.Norm(p, 2) < 1e16) {
			return math.Inf(1)
		}
		if s < 1e-18 {
			break
		}
	}
	sym := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, (p.At(i, j)+p.At(j, i))/2)
		}
	}
	var ch mat.Cholesky
	if !ch.Factorize(sym) {
		return math.Inf(1)
	}
	var l mat.TriDense
	ch.LTo(&l)
	var linv mat.Dense
	if err := linv.Inverse(&l); err != nil {
		return math.Inf(1)
	}
	var t, res mat.Dense
	res.Mul(l.T(), linv.T())
	delta := residualNorm(&res, mat.Norm(&l, 2)*mat.Norm(&linv, 2))
	if !(delta < 0.5) {
		return math.Inf(1)
	}
	t.Product(l.T(), m, linv.T())
	kappa := opNorm(&l) * opNorm(&linv) / (1 - delta)
	rounding := 8 * float64(n) * 0x1p-52 * kappa * mat.Norm(m, 2)
	return resolventFromSimilarity(kappa, (opNorm(&t)+rounding)/(1-delta), false)
}

// expPoly is det(I − D22·Δ(s)) = Σ_S c_S·e^{−s·τ_S} over the subsets S of
// the delays, with c_S = (−1)^|S|·det(D22[S,S]) and τ_S = Σ_{k∈S} τ_k.
type expPoly struct{ c, tau []float64 }

func newExpPoly(d22 *mat.Dense, tau []float64) expPoly {
	nd := len(tau)
	p := expPoly{c: make([]float64, 0, 1<<nd), tau: make([]float64, 0, 1<<nd)}
	idx := make([]int, 0, nd)
	for mask := range 1 << nd {
		idx = idx[:0]
		ts := 0.0
		for k := range nd {
			if mask&(1<<k) != 0 {
				idx = append(idx, k)
				ts += tau[k]
			}
		}
		c := 1.0
		if len(idx) > 0 {
			sub := mat.NewDense(len(idx), len(idx), nil)
			for i, r := range idx {
				for j, q := range idx {
					sub.Set(i, j, d22.At(r, q))
				}
			}
			c = mat.Det(sub)
			if bits.OnesCount(uint(mask))%2 == 1 {
				c = -c
			}
		}
		if c != 0 {
			p.c, p.tau = append(p.c, c), append(p.tau, ts)
		}
	}
	return p
}

// at returns f(s), f'(s) and Σ|c_S|·e^{−Re s·τ_S}, the scale of their
// rounding error.
func (p expPoly) at(s complex128) (f, df complex128, scale float64) {
	for i, c := range p.c {
		v := complex(c, 0) * cmplx.Exp(-s*complex(p.tau[i], 0))
		f += v
		df -= v * complex(p.tau[i], 0)
		scale += math.Abs(c) * math.Exp(-real(s)*p.tau[i]) * (1 + p.tau[i])
	}
	return f, df, scale
}

// certifiedRoot reports whether f has a zero in the disk |s − s0| <= ρ,
// ρ <= Re s0/2, by Rouché against f'(s0)(s − s0): on the circle
// |f − f'(s0)(s − s0)| <= |f(s0)| + Kρ²/2 < |f'(s0)|ρ, where
// K = Σ|c_S|τ_S²e^{−Re s0·τ_S/2} bounds the second derivative on the disk.
func (p expPoly) certifiedRoot(s0 complex128) bool {
	sigma := real(s0)
	if !(sigma > 0) || cmplx.IsInf(s0) || cmplx.IsNaN(s0) {
		return false
	}
	f, df, scale := p.at(s0)
	k := 0.0
	for i, c := range p.c {
		k += math.Abs(c) * p.tau[i] * p.tau[i] * math.Exp(-sigma/2*p.tau[i])
	}
	a := cmplx.Abs(f) + 1e-12*scale
	b := cmplx.Abs(df) - 1e-12*scale
	if !(b > 0) || !(k > 0) {
		return false
	}
	r := min(sigma/2, b/k)
	return a+k*r*r/2 < b*r
}

// rhpRoot reports whether det(I − D22·Δ(s)) has a certified zero in
// Re s > 0: a sign change on the positive real axis, where f is real and
// f(+∞) = 1, or a Newton limit from seeds (ln μ + 2πjm)/τ for the
// eigenvalues and diagonal entries μ of D22 with |μ| > 1 and each delay τ,
// certified by certifiedRoot.
func (p expPoly) rhpRoot(d22 *mat.Dense, tau []float64) bool {
	tmin, sum := math.Inf(1), 0.0
	for i, c := range p.c {
		if p.tau[i] > 0 {
			tmin = min(tmin, p.tau[i])
			sum += math.Abs(c)
		}
	}
	if math.IsInf(tmin, 1) {
		return false
	}
	if sigmaMax := math.Log(2*sum) / tmin; sigmaMax > 0 {
		const k = 256
		for i := 1; i <= k; i++ {
			f, _, scale := p.at(complex(sigmaMax*float64(i)/k, 0))
			if real(f) < -1e-12*scale {
				return true
			}
		}
	}
	var mus []complex128
	var eig mat.Eigen
	if eig.Factorize(d22, mat.EigenNone) {
		mus = eig.Values(nil)
	}
	n, _ := d22.Dims()
	for i := range n {
		mus = append(mus, complex(d22.At(i, i), 0))
	}
	for _, mu := range mus {
		if !(cmplx.Abs(mu) > 1) {
			continue
		}
		for _, t := range slices.Compact(slices.Sorted(slices.Values(tau))) {
			if !(t > 0) {
				continue
			}
			for m := range 4 {
				s := (cmplx.Log(mu) + complex(0, 2*math.Pi*float64(m))) / complex(t, 0)
				if p.certifiedRoot(p.newton(s)) {
					return true
				}
			}
		}
	}
	return false
}

func (p expPoly) newton(s complex128) complex128 {
	for range 60 {
		f, df, _ := p.at(s)
		if df == 0 {
			return s
		}
		step := f / df
		s -= step
		if cmplx.Abs(step) <= 1e-14*max(1, cmplx.Abs(s)) {
			break
		}
	}
	return s
}
