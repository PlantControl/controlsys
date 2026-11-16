package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
)

// FreqResponse's within-terms contract (FKYKOG) for an explicit model
// without internal delays: a nil error certifies each entry ĥ within
// δ/(1−η) of the stored model's exact h = D + c·Mb at the exact point λ,
// M = (λI − A)⁻¹, where δ = u(|c||M||A||M||b| + |c||M||b| + |D|),
// η = u·ρ(|M||A|) < 1 and u = n·ε. Terms are invariant under the
// balancing's exact diagonal scaling, so termsCert runs on (Â, B̂, Ĉ).
//
// termsCert proves |ĥ − h| ≤ δ a posteriori for a double solution x̂ of
// column j, with θ = u/2, X = |c||M||A||M||b| + |c||M||b| (δ = uX + u|D|)
// and L = |c||x̂|, all moduli complex moduli:
//
//  1. Rows: the residual r = b − (λI − A)x̂, compensated against the exact
//     pencil, has |rᵢ| ≤ θsᵢ, s = |A||x̂| + |b|, in every row
//     (Oettli–Prager with A and b perturbed, λ exact).
//  2. e = |x̂ − x| = |Mr| ≤ θ|M|s, and |x̂| ≤ |M||b| + e, so
//     |c|e ≤ θX + θ|c||M||A|e ≤ θX + T₂, T₂ = θ²|c|N|A|Ns, where
//     N = M(Û)⁻¹M(L̂)⁻¹P, the comparison matrices of the LU factors (|·|₁
//     off the diagonal, 1/|·⁻¹|₁ on it), bounds |M|.
//  3. The output ĥ differs from c·x̂ + D by o, measured in compensated
//     arithmetic, so |ĥ − h| ≤ o + θX + T₂; and X ≥ |c||x| ≥ L − |c|e
//     gives X ≥ (L − T₂)/(1 + θ).
//  4. Accept where o + T₂ ≤ (u − θ)·(L − T₂)/(1 + θ) + u|D|: then
//     |ĥ − h| ≤ uX + u|D| = δ ≤ δ/(1−η) for every η < 1.
//
// The first-order term θX cancels against uX exactly, with no Neumann
// series and so no condition on η; |M| enters only the second-order T₂,
// through N, which bounds the inverse of the factored pencil, so T₂ is
// exact to first order in the factorization's backward error. θ = u/2
// leaves the other half of u to the output, formed with one rounding where
// refinement re-forms it (compensatedOutput); at n = 1 that rounding alone
// takes ε/2 of the ε budget.
type termsCert struct {
	n, m, p int
	absA    []float64
	xc, r   []complex128
	xm, s   []float64
	t, t2   []float64
}

func newTermsCert(br *balancedRealization) *termsCert {
	n := br.n
	f := make([]float64, n*n+4*n)
	c := make([]complex128, 2*n)
	tc := &termsCert{
		n: n, m: br.m, p: br.p,
		absA: f[: n*n : n*n],
		xc:   c[:n:n], r: c[n:],
		xm: f[n*n : n*n+n : n*n+n], s: f[n*n+n : n*n+2*n : n*n+2*n],
		t: f[n*n+2*n : n*n+3*n : n*n+3*n], t2: f[n*n+3*n:],
	}
	for i, v := range br.a {
		tc.absA[i] = math.Abs(v)
	}
	return tc
}

// load sets tc.xc and tc.xm to column j of x (entry k at x[k*ks+j*js])
// and its moduli.
func (tc *termsCert) load(x []complex128, ks, js, j int) {
	for k := range tc.n {
		v := x[k*ks+j*js]
		tc.xc[k] = v
		tc.xm[k] = termsAbs(v)
	}
}

// termsAbs is |z|, by the plain formula where its squares stay normal.
func termsAbs(z complex128) float64 {
	re, im := math.Abs(real(z)), math.Abs(imag(z))
	if mx := max(re, im); mx > 0x1p-480 && mx < 0x1p480 {
		return math.Sqrt(re*re + im*im)
	}
	return math.Hypot(re, im)
}

// rows reports whether column j, loaded, passes the row test of step 1 at
// pt, leaving s in tc.s.
func (tc *termsCert) rows(br *balancedRealization, pt frequencyPoint, j int) bool {
	n, m := tc.n, tc.m
	theta := float64(n) / 2 * eps()
	kappa := complex(1, 0)
	if !pt.isPlain() {
		kappa = pt.scale()
	}
	pencilResidual(tc.r, tc.xc, 1, br.a, nil, n, br.b[j:], m, n, pt.p, pt.q, kappa)
	ik := crecip(kappa)
	lam := cabs1(pt.value())
	slack := 2 * float64(n+2) * eps() * eps()
	for i := range n {
		s := math.Abs(br.b[i*m+j])
		for k, a := range tc.absA[i*n : (i+1)*n] {
			s += a * tc.xm[k]
		}
		tc.s[i] = s
		r := tc.r[i]
		if kappa != 1 {
			r *= ik
		}
		rb := termsAbs(r)*(1+4*eps()) + slack*(s+lam*tc.xm[i]+cabs1(tc.xc[i]))
		if !(rb <= theta*s) {
			return false
		}
	}
	return true
}

// tail sets tc.t2 to N|Â|Ns of step 2, applying N in place with nsolve.
func (tc *termsCert) tail(nsolve func([]float64)) {
	n := tc.n
	copy(tc.t, tc.s)
	nsolve(tc.t)
	for i := range n {
		var v float64
		for k, a := range tc.absA[i*n : (i+1)*n] {
			v += a * tc.t[k]
		}
		tc.t2[i] = v
	}
	nsolve(tc.t2)
}

// output reports whether each entry of column j of dst passes steps 3–4,
// column j of x̂ loaded and its tail formed.
func (tc *termsCert) output(br *balancedRealization, dst []complex128, j int) bool {
	n, m := tc.n, tc.m
	u := float64(n) * eps()
	theta := u / 2
	grow := 1 + 8*float64(n)*eps()
	for i := range tc.p {
		cr := br.c[i*n : (i+1)*n]
		h := dst[i*m+j]
		var re, im compensatedSum
		var l, ct, mag float64
		for k, c := range cr {
			re.addProd(c, real(tc.xc[k]))
			im.addProd(c, imag(tc.xc[k]))
			ac := math.Abs(c)
			l += ac * tc.xm[k]
			ct += ac * tc.t2[k]
			mag += ac * cabs1(tc.xc[k])
		}
		d := 0.0
		if br.d != nil {
			d = br.d[i*br.dStride+j]
			re.add(d)
		}
		re.add(-real(h))
		im.add(-imag(h))
		o := termsAbs(complex(re.value(), im.value()))*(1+2*eps()) + 2*float64(n+2)*eps()*eps()*(mag+math.Abs(d)+cabs1(h))
		t2 := theta * theta * ct * grow
		if !(o <= math.MaxFloat64 && t2 <= math.MaxFloat64) {
			return false
		}
		lo := max(l*(1-2*float64(n)*eps())-t2, 0) / (1 + theta)
		if !(o+t2 <= ((u-theta)*lo+u*math.Abs(d))*(1-4*eps())) {
			return false
		}
	}
	return true
}

// certify runs steps 1–4 on x (entry (k, j) at x[k*ks+j*js]) and dst.
// rowsOK reports step 1 alone, which refinement can still fix.
func (tc *termsCert) certify(br *balancedRealization, pt frequencyPoint, x []complex128, ks, js int, dst []complex128, nsolve func([]float64)) (rowsOK, ok bool) {
	rowsOK = true
	for j := range tc.m {
		tc.load(x, ks, js, j)
		if !tc.rows(br, pt, j) {
			return false, false
		}
		tc.tail(nsolve)
		if !tc.output(br, dst, j) {
			ok = false
			return rowsOK, false
		}
	}
	return true, true
}

// compensatedOutput sets dst = Ĉx + D, each entry rounded once.
func compensatedOutput(br *balancedRealization, x []complex128, ks, js int, dst []complex128) {
	n, m := br.n, br.m
	for i := range br.p {
		cr := br.c[i*n : (i+1)*n]
		for j := range m {
			var re, im compensatedSum
			for k, c := range cr {
				v := x[k*ks+j*js]
				re.addProd(c, real(v))
				im.addProd(c, imag(v))
			}
			if br.d != nil {
				re.add(br.d[i*br.dStride+j])
			}
			dst[i*m+j] = complex(re.value(), im.value())
		}
	}
}

// termsSolver is a delay-free explicit solver whose last evalInto value
// certifyTerms checks against the within-terms contract, refining it where
// only the row test fails.
type termsSolver interface {
	frequencyPointSolver
	certifyTerms(pt frequencyPoint, dst []complex128) bool
}

// errTermsUnverified is FreqResponse's failure where neither the double
// solve, the verified double-double solve nor the exact solve (n ≤
// exactDetMaxOrder) certifies the response within the rounding of the
// stored data's terms.
var errTermsUnverified = fmt.Errorf("response not certified within the rounding of the model's terms: %w", ErrAccuracyUnverified)

// evalWithinTerms is FreqResponse's point evaluation of an explicit model
// without internal delays (see termsCert): the solver's value where
// certifyTerms passes, else the verified double-double response
// (ddResponseVerified), else for n ≤ exactDetMaxOrder the exact response
// (exactResponseAt), whose entries are infinite only where their exact
// value lies beyond float64's range, and errTermsUnverified otherwise. The
// point at infinity keeps evalWithPoleLimit. A point where pI + qA
// is exactly singular is a pole of the stored model and takes the pole
// limit (evalWithPoleLimit). Off a pole of the stored model no entry is
// ever infinite: a pencil singular only within rounding is not a pole.
func evalWithinTerms(solver termsSolver, sys *System, pt frequencyPoint, dst []complex128) error {
	if cmplx.IsInf(pt.value()) {
		return evalWithPoleLimit(solver.evalInto, sys, pt, dst)
	}
	err := solver.evalInto(pt, dst)
	if err == nil && solver.certifyTerms(pt, dst) {
		return nil
	}
	if errors.Is(err, errNonFinitePencil) || errors.Is(err, errStateUnderflow) {
		return err
	}
	if ddResponseVerified(sys, pt, dst) {
		return nil
	}
	n, _, _ := sys.Dims()
	if n > exactDetMaxOrder {
		return errTermsUnverified
	}
	ps, ok := cachedPoleSchur(sys)
	if !ok {
		return errTermsUnverified
	}
	g, _, ok := exactResponseAt(ps, pt)
	if !ok {
		return evalWithPoleLimit(solver.evalInto, sys, pt, dst)
	}
	copy(dst, g)
	return nil
}
