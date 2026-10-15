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
// P − D22ᵀPD22 = I, solvable when ρ(D22) < 1. On the torus an invertible D22
// gives (I − D22Θ)⁻¹ = −(I − Θ⁻¹W)⁻¹Θ⁻¹W with W = D22⁻¹, bounded by
// κ(S)γ/(1 − γ) when γ = ‖S⁻¹WS‖ < 1, the anti-stable difference operator.
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
	if equalDelays(e.tau) {
		b.rhp = steinBound(d22)
	}
	var w, res mat.Dense
	if err := w.Inverse(d22); err == nil {
		res.Mul(d22, &w)
		if delta := residualNorm(&res, mat.Norm(d22, 2)*mat.Norm(&w, 2)); delta < 1 {
			b.torus = diagonalScalingBound(&w, mat.Norm(&w, 2)*delta/(1-delta), true)
		}
	}
	return b
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
