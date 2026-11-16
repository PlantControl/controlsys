package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sync/atomic"

	"plantcontrol.org/v1/gonum/blas/blas64"
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
//  1. Rows: the residual r = b − (λI − A)x̂ against the exact pencil has
//     |rᵢ| ≤ θ'sᵢ, s = |A||x̂| + |b|, in every row, for some θ' ≤ θ
//     (Oettli–Prager with A and b perturbed, λ exact); the steps below
//     hold with θ' in place of θ, and θ' is the largest proven |rᵢ|/sᵢ. A row passes first on its
//     working-precision residual plus that residual's rounding bound
//     (nnzᵢ+5)(ε/2)Sᵢ, Sᵢ the sum of its terms' magnitudes; the rows that
//     fail it are formed again in compensated arithmetic. With θ = (n/2)ε
//     only rows with nnzᵢ + 5 ≤ n can pass that tier (Sᵢ ≥ sᵢ), so denser
//     rows skip it.
//  2. e = |x̂ − x| = |Mr| ≤ θ|M|s, and |x̂| ≤ |M||b| + e, so
//     |c|e ≤ θX + θ|c||M||A|e ≤ θX + T₂, T₂ = θ²|c||M||A||M|s.
//  3. |M| is bounded through the computed factors F = P⁻¹L̂Û of the
//     rounded pencil: F = (λI − A) + ΔF with |ΔF| ≤ γ·P⁻¹|L̂||Û| (γ =
//     (2n+4)ε covers complex GEPP and the rounded diagonal λ − aᵢᵢ), and
//     N = M(Û)⁻¹M(L̂)⁻¹P, the comparison matrices of the factors (|·|₁ off
//     the diagonal, 1/|·⁻¹|₁ on it), bounds |F⁻¹|. M = (I − F⁻¹ΔF)⁻¹F⁻¹,
//     so for y ≥ 0 and v = Ny, |M|y ≤ Σₖ (N|ΔF|)ᵏ v. Where the q check
//     N(|ΔF|v) ≤ q·v holds with q < 1, that sum is at most v/(1−q).
//     Applied to y = s and then to y = |A|·Ns/(1−q₁), T₂ ≤
//     θ²|c|v₂/(1−q₂) with v₂ = N|A|Ns/(1−q₁). q ≥ 1 fails the point. The
//     Hessenberg sweep adds the reduction's normwise backward error,
//     γᵣ‖Â‖_F, to |ΔF| as a rank-one all-ones term (comparisonBounds).
//  4. The output ĥ differs from c·x̂ + D by o, measured in compensated
//     arithmetic, so |ĥ − h| ≤ o + θX + T₂; and X ≥ |c||x| ≥ L − |c|e
//     gives X ≥ (L − T₂)/(1 + θ).
//  5. Accept where o + T₂ ≤ (u − θ)·(L − T₂)/(1 + θ) + u|D|: then
//     |ĥ − h| ≤ uX + u|D| = δ ≤ δ/(1−η) for every η < 1.
//
// The first-order term θX cancels against uX exactly, with no Neumann
// series in |M||A| and so no condition on η; |M| enters only the
// second-order T₂. θ = u/2 leaves the other half of u to the output,
// formed with one rounding where refinement re-forms it
// (compensatedOutput); at n = 1 that rounding alone takes ε/2 of the ε
// budget.
type termsCert struct {
	n, m, p int
	absA    []float64
	nnz     []int
	xc      []complex128
	xm, s   []float64
	t, t2   []float64
	w       []float64
	// undriven marks, at k*m+j, a state input j cannot reach (undrivenStates):
	// exactly zero in x, where the solve leaves only roundoff fill.
	undriven []bool
}

// termsEps is eps(), 2⁻⁵², as a constant: the certificate's inner loops
// read it per row.
const termsEps = 0x1p-52

// comparisonBounds bounds the inverse of a factored pencil (step 3):
// nsolve overwrites the nonnegative y with Ny, and factorError sets dst
// to an upper bound on |ΔF|y.
type comparisonBounds interface {
	nsolve(y []float64)
	factorError(dst, y []float64)
}

func newTermsCert(br *balancedRealization) *termsCert {
	n := br.n
	f := make([]float64, n*n+5*n)
	tc := &termsCert{
		n: n, m: br.m, p: br.p,
		absA: f[: n*n : n*n],
		nnz:  make([]int, n),
		xc:   make([]complex128, n),
		xm:   f[n*n : n*n+n : n*n+n], s: f[n*n+n : n*n+2*n : n*n+2*n],
		t: f[n*n+2*n : n*n+3*n : n*n+3*n], t2: f[n*n+3*n : n*n+4*n : n*n+4*n], w: f[n*n+4*n:],
	}
	for i, v := range br.a {
		tc.absA[i] = math.Abs(v)
		if v != 0 {
			tc.nnz[i/n]++
		}
	}
	tc.undriven = br.undrivenStates()
	return tc
}

// load sets tc.xc and tc.xm to column j of x (entry k at x[k*ks+j*js])
// and its moduli, with the states input j cannot reach set to their exact
// 0: their rows then hold exactly, and the output check measures what
// their fill added to the value.
func (tc *termsCert) load(x []complex128, ks, js, j int) {
	for k := range tc.n {
		v := x[k*ks+j*js]
		if tc.undriven[k*tc.m+j] {
			v = 0
		}
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
// pt, leaving s in tc.s, and returns the smallest θ' ≤ θ the bounds on
// |rᵢ|/sᵢ prove, with cheap reporting whether a row passed on its
// working-precision residual alone. compensated skips that tier.
func (tc *termsCert) rows(br *balancedRealization, pt frequencyPoint, j int, compensated bool) (ok bool, thetaR float64, cheap bool) {
	n, m := tc.n, tc.m
	theta := float64(n) / 2 * termsEps
	kappa := complex(1, 0)
	if !pt.isPlain() {
		kappa = pt.scale()
	}
	ka, km, pa, qa := cabs1(kappa), termsAbs(kappa), cabs1(pt.p), cabs1(pt.q)
	lam := cabs1(pt.value())
	slack := 2 * float64(n+2) * termsEps * termsEps
	for i := range n {
		bi := br.b[i*m+j]
		s := math.Abs(bi)
		var sum float64
		var ar, ai float64
		row := br.a[i*n : (i+1)*n]
		for k, a := range tc.absA[i*n : (i+1)*n] {
			if a == 0 {
				continue
			}
			s += a * tc.xm[k]
			x := tc.xc[k]
			ar += row[k] * real(x)
			ai += row[k] * imag(x)
			sum += a * cabs1(x)
		}
		tc.s[i] = s
		xi := tc.xc[i]
		if !compensated && tc.nnz[i]+5 <= n {
			r := kappa*complex(bi, 0) - pt.p*xi - pt.q*complex(ar, ai)
			bound := float64(tc.nnz[i]+5) * termsEps / 2 * (ka*math.Abs(bi) + pa*cabs1(xi) + qa*sum)
			if rb := (termsAbs(r) + bound) * (1 + 2*termsEps) / km; rb <= theta*s {
				if rb > 0 {
					thetaR = max(thetaR, rb/s)
				}
				cheap = true
				continue
			}
		}
		rc := pencilResidualRow(tc.xc, row, i, bi, pt.p, pt.q, kappa)
		if kappa != 1 {
			rc *= crecip(kappa)
		}
		rb := termsAbs(rc)*(1+4*termsEps)*(1+2*termsEps) + slack*(s+lam*tc.xm[i]+cabs1(xi))
		if !(rb <= theta*s) {
			return false, 0, cheap
		}
		if rb > 0 {
			thetaR = max(thetaR, rb/s)
		}
	}
	return true, thetaR, cheap
}

// pencilResidualRow is row i of pencilResidual for an explicit model:
// κbᵢ − p·xᵢ − q·(Ax)ᵢ in compensated arithmetic, row the row of A.
func pencilResidualRow(x []complex128, row []float64, i int, b float64, p, q, kappa complex128) complex128 {
	pr, pi, qr, qi := real(p), imag(p), real(q), imag(q)
	var axr, axi compensatedSum
	for k, av := range row {
		if av == 0 {
			continue
		}
		axr.addProd(av, real(x[k]))
		axi.addProd(av, imag(x[k]))
	}
	var re, im compensatedSum
	if kappa == 1 {
		re.add(b)
	} else {
		re.addProd(real(kappa), b)
		im.addProd(imag(kappa), b)
	}
	xr, xi := real(x[i]), imag(x[i])
	re.addProd(-pr, xr)
	re.addProd(pi, xi)
	im.addProd(-pr, xi)
	im.addProd(-pi, xr)
	re.addScaled(-qr, axr)
	re.addScaled(qi, axi)
	im.addScaled(-qr, axi)
	im.addScaled(-qi, axr)
	return complex(re.value(), im.value())
}

// mBound sets y to an upper bound on |M|y (step 3) and reports false where
// the q check fails.
func (tc *termsCert) mBound(y []float64, cb comparisonBounds) bool {
	cb.nsolve(y)
	cb.factorError(tc.w, y)
	cb.nsolve(tc.w)
	q := 0.0
	for i, v := range y {
		if tc.w[i] == 0 {
			continue
		}
		if !(v > 0) {
			return false
		}
		q = max(q, tc.w[i]/v)
	}
	q *= 1 + 8*float64(tc.n)*termsEps
	if !(q < 1) {
		return false
	}
	f := 1 / (1 - q) * (1 + 2*termsEps)
	for i := range y {
		y[i] *= f
	}
	return true
}

// tail sets tc.t2 to an upper bound on |M||Â||M|s of step 2 and reports
// false where a q check fails.
func (tc *termsCert) tail(cb comparisonBounds) bool {
	n := tc.n
	copy(tc.t, tc.s)
	if !tc.mBound(tc.t, cb) {
		return false
	}
	for i := range n {
		var v float64
		for k, a := range tc.absA[i*n : (i+1)*n] {
			v += a * tc.t[k]
		}
		tc.t2[i] = v
	}
	return tc.mBound(tc.t2, cb)
}

// output reports whether each entry of column j of dst passes steps 4–5,
// column j of x̂ loaded and its tail formed.
func (tc *termsCert) output(br *balancedRealization, dst []complex128, j int, theta float64) bool {
	n, m := tc.n, tc.m
	u := float64(n) * termsEps
	grow := 1 + 8*float64(n)*termsEps
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
		o := termsAbs(complex(re.value(), im.value()))*(1+2*termsEps) + 2*float64(n+2)*termsEps*termsEps*(mag+math.Abs(d)+cabs1(h))
		t2 := theta * theta * ct * grow
		if !(o <= math.MaxFloat64 && t2 <= math.MaxFloat64) {
			return false
		}
		lo := max(l*(1-2*float64(n)*termsEps)-t2, 0) / (1 + theta)
		if !(o+t2 <= ((u-theta)*lo+u*math.Abs(d))*(1-4*termsEps)) {
			return false
		}
	}
	return true
}

// certify runs steps 1–5 on x (entry (k, j) at x[k*ks+j*js]) and dst.
// rowsOK reports step 1 alone, which refinement can still fix.
func (tc *termsCert) certify(br *balancedRealization, pt frequencyPoint, x []complex128, ks, js int, dst []complex128, cb comparisonBounds) (rowsOK, ok bool) {
	for j := range tc.m {
		tc.load(x, ks, js, j)
		rowsOK, thetaR, cheap := tc.rows(br, pt, j, false)
		if !rowsOK {
			return false, false
		}
		if !tc.tail(cb) {
			return true, false
		}
		if tc.output(br, dst, j, thetaR) {
			continue
		}
		if !cheap {
			return true, false
		}
		if _, thetaR, _ = tc.rows(br, pt, j, true); !tc.output(br, dst, j, thetaR) {
			return true, false
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
// certifyTerms passes, else the verified double-double response where it
// passes termsCert too (ddResponseCertified), else for n ≤
// exactDetMaxOrder the exact response (exactResponseAt), whose entries are
// infinite only where their exact value lies beyond float64's range. A
// point where pI + qA is exactly singular, for n > exactDetMaxOrder decided
// on the diagonal blocks of the block-triangular form (blockSingularAt), is
// a pole of the stored model and takes the pole limit (evalWithPoleLimit).
// Otherwise dst holds the best unverified estimate (the solver's value, else
// the double-double response, else the pole limit, else NaN) and the error
// is errTermsUnverified. Off a pole of the stored model no entry is ever
// infinite: a pencil singular only within rounding is not a pole. The point
// at infinity keeps evalWithPoleLimit.
func evalWithinTerms(solver termsSolver, sys *System, pt frequencyPoint, dst []complex128) error {
	if cmplx.IsInf(pt.value()) {
		return evalWithPoleLimit(solver.evalInto, sys, pt, dst)
	}
	err := solver.evalInto(pt, dst)
	if err == nil && solver.certifyTerms(pt, dst) {
		termsPassCount.Add(1)
		return nil
	}
	if errors.Is(err, errNonFinitePencil) || errors.Is(err, errStateUnderflow) {
		return err
	}
	var estimate []complex128
	if err == nil {
		estimate = append(estimate, dst...)
	}
	if ddResponseCertified(sys, pt, dst) {
		termsDDCount.Add(1)
		return nil
	}
	n, _, _ := sys.Dims()
	singular, decided := false, false
	if n <= exactDetMaxOrder {
		g, _, ok := exactResponseOf(sys, pt)
		if ok {
			termsExactCount.Add(1)
			copy(dst, g)
			return nil
		}
		singular, decided = true, true
	} else {
		singular, decided = blockSingularAt(sys, pt), true
	}
	poleLimit := func() error {
		return evalWithPoleLimit(func(at frequencyPoint, d []complex128) error {
			if at == pt {
				return errSingularPencil
			}
			return solver.evalInto(at, d)
		}, sys, pt, dst)
	}
	if decided && singular {
		termsPoleCount.Add(1)
		return poleLimit()
	}
	termsUnverifiedCount.Add(1)
	switch {
	case estimate != nil:
		copy(dst, estimate)
	case ddResponse(sys, pt, dst) == nil:
	case poleLimit() == nil:
	default:
		for i := range dst {
			dst[i] = complex(math.NaN(), math.NaN())
		}
	}
	return errTermsUnverified
}

// exactResponseOf is exactResponseAt of the explicit sys, memoized where
// its poleSchurData builds and solved unmemoized where that fails.
func exactResponseOf(sys *System, pt frequencyPoint) (g []complex128, overflow, ok bool) {
	if ps, ok := cachedPoleSchur(sys); ok {
		return exactResponseAt(ps, pt)
	}
	n, m, p := sys.Dims()
	ps := &poleSchurData{n: n, m: m, p: p, a: make([]float64, n*n), b: make([]float64, n*m), c: make([]float64, p*n), d: make([]float64, p*m)}
	copyStrided(ps.a, n, sys.A.RawMatrix().Data, sys.A.RawMatrix().Stride, n, n)
	copyStrided(ps.b, m, sys.B.RawMatrix().Data, sys.B.RawMatrix().Stride, n, m)
	copyStrided(ps.c, n, sys.C.RawMatrix().Data, sys.C.RawMatrix().Stride, p, n)
	copyStrided(ps.d, m, sys.D.RawMatrix().Data, sys.D.RawMatrix().Stride, p, m)
	return solveExactAt(ps, pt)
}

// blockSingularAt reports whether pI + qA of the explicit sys is exactly
// singular at pt on some diagonal block of order up to exactDetMaxOrder
// of its block-triangular form (tarjanBlocks of A's pattern): det(pI + qA)
// is the product of the blocks' determinants.
func blockSingularAt(sys *System, pt frequencyPoint) bool {
	n, _, _ := sys.Dims()
	pat := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	raw := sys.A.RawMatrix()
	for i := range n {
		for j := range n {
			if raw.Data[i*raw.Stride+j] != 0 {
				pat.Data[i*n+j] = 1
			}
		}
	}
	for _, block := range tarjanBlocks(pat, n) {
		k := len(block)
		if k > exactDetMaxOrder {
			continue
		}
		a := make([]float64, k*k)
		for i, bi := range block {
			for j, bj := range block {
				a[i*k+j] = raw.Data[bi*raw.Stride+bj]
			}
		}
		b := make([]float64, k)
		b[0] = 1
		ps := &poleSchurData{n: k, m: 1, p: 1, a: a, b: b, c: slices.Clone(b), d: []float64{0}}
		if _, _, ok := solveExactAt(ps, pt); !ok {
			return true
		}
	}
	return false
}

// termsPassCount, termsDDCount, termsExactCount, termsPoleCount and
// termsUnverifiedCount count how evalWithinTerms settled each point: the
// certified double solve (refined where termsRefinedCount also counts it),
// the verified double-double response, the exact response, the pole limit
// at an exactly singular pencil, or errTermsUnverified; for the census.
var termsPassCount, termsRefinedCount, termsDDCount, termsExactCount, termsPoleCount, termsUnverifiedCount atomic.Int64
