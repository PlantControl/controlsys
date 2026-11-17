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
	xc      []complex128 // n×m column blocks: column j at [j*n:(j+1)*n]
	xm, s   []float64    // likewise
	sAll    []float64
	t, t2   []float64
	w, yb   []float64
	// undriven marks, at k*m+j, a state input j cannot reach (undrivenStates):
	// exactly zero in x, where the solve leaves only roundoff fill.
	undriven []bool
}

// termsEps is eps(), 2⁻⁵², as a constant: the certificate's inner loops
// read it per row.
const termsEps = 0x1p-52

// termsU is the contract's u = 4(n+5)·eps() = 4(n+5)·2⁻⁵² of an n-state
// model; θ = u/2.
func termsU(n int) float64 { return 4 * float64(n+5) * termsEps }

// comparisonBounds bounds the inverse of a factored pencil (step 3):
// nsolve overwrites the nonnegative y with Ny, and factorError sets dst
// to an upper bound on |ΔF|y.
type comparisonBounds interface {
	nsolve(y []float64)
	factorError(dst, y []float64)
}

func newTermsCert(br *balancedRealization) *termsCert {
	n, m := br.n, br.m
	f := make([]float64, n*n+2*n*m+5*n)
	take := func(k int) []float64 {
		out := f[:k:k]
		f = f[k:]
		return out
	}
	tc := &termsCert{n: n, m: m, p: br.p, nnz: make([]int, n), xc: make([]complex128, n*m)}
	tc.absA, tc.xm, tc.s = take(n*n), take(n*m), take(n*m)
	tc.sAll, tc.t, tc.t2, tc.w = take(n), take(n), take(n), take(n)
	for i, v := range br.a {
		tc.absA[i] = math.Abs(v)
		if v != 0 {
			tc.nnz[i/n]++
		}
	}
	tc.undriven = br.undrivenStates()
	return tc
}

// load sets column j of tc.xc and tc.xm to column j of x (entry k at
// x[k*ks+j*js]) and its moduli, with the states input j cannot reach set to
// their exact 0: their rows then hold exactly, and the output check
// measures what their fill added to the value.
func (tc *termsCert) load(x []complex128, ks, js, j int) {
	n := tc.n
	for k := range n {
		v := x[k*ks+j*js]
		if tc.undriven[k*tc.m+j] {
			v = 0
		}
		tc.xc[j*n+k] = v
		tc.xm[j*n+k] = termsAbs(v)
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
// pt, leaving s in column j of tc.s, and returns the θ' ≤ θ the bounds on
// |rᵢ|/sᵢ prove, with cheap reporting whether a row passed on its
// working-precision residual alone. compensated skips that tier.
func (tc *termsCert) rows(br *balancedRealization, pt frequencyPoint, j int, compensated bool) (ok bool, thetaR float64, cheap bool) {
	n, m := tc.n, tc.m
	theta := termsU(n) / 2
	xc, xm, sv := tc.xc[j*n:(j+1)*n], tc.xm[j*n:(j+1)*n], tc.s[j*n:(j+1)*n]
	kappa := complex(1, 0)
	if !pt.isPlain() {
		kappa = pt.scale()
	}
	ka, km, pa, qa := cabs1(kappa), termsAbs(kappa), cabs1(pt.p), cabs1(pt.q)
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
			s += a * xm[k]
			x := xc[k]
			ar += row[k] * real(x)
			ai += row[k] * imag(x)
			sum += a * cabs1(x)
		}
		sv[i] = s
		xi := xc[i]
		if !compensated {
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
		rc, e := pencilResidualRow(xc, row, i, bi, pt.p, pt.q, kappa)
		if kappa != 1 {
			rc *= crecip(kappa)
		}
		rb := termsAbs(rc)*(1+4*termsEps)*(1+2*termsEps) + e/km*(1+4*termsEps)
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
// κbᵢ − p·xᵢ − q·(Ax)ᵢ in compensated arithmetic, row the row of A, and e
// bounding the distance of the exact residual from the sums' unrounded
// s + c, before value() rounds them: the inner sums of (Ax)ᵢ hold their
// exact value within dot2Err(nnz)·Σ|aₖ||xₖ| (real and imaginary parts
// together) and enter both outer sums scaled by q; each outer sum has at
// most 7 terms, q times both words of an inner sum among them, and adds
// dot2Err(7) times their moduli, and every rounding into the subnormal
// range, where TwoProduct's FMA residual and the bound's relative terms
// fail, at most 2⁻¹⁰⁷⁴ more.
func pencilResidualRow(x []complex128, row []float64, i int, b float64, p, q, kappa complex128) (r complex128, e float64) {
	pr, pi, qr, qi := real(p), imag(p), real(q), imag(q)
	var axr, axi compensatedSum
	var inner float64
	nnz := 0
	for k, av := range row {
		if av == 0 {
			continue
		}
		axr.addProd(av, real(x[k]))
		axi.addProd(av, imag(x[k]))
		inner += math.Abs(av) * cabs1(x[k])
		nnz++
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
	qa := cabs1(q)
	outer := cabs1(kappa)*math.Abs(b) + cabs1(p)*cabs1(x[i]) +
		qa*(math.Abs(axr.s)+math.Abs(axr.c)+math.Abs(axi.s)+math.Abs(axi.c))
	e = (dot2Err(7)*outer+qa*dot2Err(nnz)*inner)*(1+float64(nnz+16)*termsEps) +
		(float64(4*nnz)*max(qa, 1)+64)*0x1p-1074
	return complex(re.value(), im.value()), e
}

// dot2Err is γₖ², γₖ = kε/(1 − kε) with ε = termsEps = 2⁻⁵², twice the
// unit roundoff u: a compensatedSum of k add or addProd calls represents
// its exact sum Στᵢ, s + c unrounded, within dot2Err(k)·Σ|τᵢ| (Ogita, Rump
// and Oishi's Dot2 bound, quadratic in k). Of the exact Στᵢ = s +
// Σ(TwoSum and TwoProduct errors), c sums at most 2k error terms with
// 2k − 1 roundings, and those terms total at most u·k(1 + γₖ(u))·Σ|τᵢ|, so
// c's rounding is at most γ₂ₖ₋₁(u)·u·k(1 + γₖ(u))·Σ|τᵢ| ≤ γ₂ₖ(u)²/2·Σ|τᵢ|,
// and γ₂ₖ(u) = γₖ(ε), without underflow, which callers bound apart. It is
// +Inf where kε ≥ 1.
func dot2Err(k int) float64 {
	ke := float64(k) * termsEps
	if !(ke < 1) {
		return math.Inf(1)
	}
	g := ke / (1 - ke) * (1 + 2*termsEps)
	return g * g * (1 + 2*termsEps)
}

// mUpper overwrites the nonnegative y with an upper bound on |M|y and
// reports false where it has none.
type mUpper func(y []float64) bool

// comparisonUpper is step 3's mUpper from the factors' comparison
// matrices: v = Ny, then the q check N(|ΔF|v) ≤ q·v, q < 1, gives
// |M|y ≤ v/(1−q).
func (tc *termsCert) comparisonUpper(cb comparisonBounds) mUpper {
	return func(y []float64) bool {
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
}

// minUpper is the entrywise smaller of the bounds of a and b on the same
// |M|y, either alone where the other has none: the comparison matrices can
// bound one output's tail far tighter than the inverse tier while the
// latter alone bounds another's X from below (sweepInt scaled/0.1/false/56
// at ω = 1e-10: T₂ 2e-19 against 4e-6 on output 1).
func (tc *termsCert) minUpper(a, b mUpper) mUpper {
	return func(y []float64) bool {
		if tc.yb == nil {
			tc.yb = make([]float64, tc.n)
		}
		yb := tc.yb
		copy(yb, y)
		okA := a(y)
		if !b(yb) {
			return okA
		}
		for i, v := range yb {
			if !okA || v < y[i] {
				y[i] = v
			}
		}
		return true
	}
}

// tail sets tc.t2 to an upper bound on |M||Â||M|sAll of step 2, shared by
// every column, and reports false where up has no bound.
func (tc *termsCert) tail(up mUpper) bool {
	n := tc.n
	copy(tc.t, tc.sAll)
	if !up(tc.t) {
		return false
	}
	for i := range n {
		var v float64
		for k, a := range tc.absA[i*n : (i+1)*n] {
			v += a * tc.t[k]
		}
		tc.t2[i] = v
	}
	return up(tc.t2)
}

// output reports whether each entry of column j of dst passes steps 4–5,
// column j of x̂ loaded and the tail formed, with ib's lower bound on X
// where ib is not nil.
func (tc *termsCert) output(br *balancedRealization, dst []complex128, j int, theta float64, ib *inverseBound) bool {
	n, m := tc.n, tc.m
	u := termsU(n)
	grow := 1 + 8*float64(n)*termsEps
	xc, xm := tc.xc[j*n:(j+1)*n], tc.xm[j*n:(j+1)*n]
	for i := range tc.p {
		cr := br.c[i*n : (i+1)*n]
		h := dst[i*m+j]
		var re, im compensatedSum
		var l, ct, mag float64
		for k, c := range cr {
			re.addProd(c, real(xc[k]))
			im.addProd(c, imag(xc[k]))
			ac := math.Abs(c)
			l += ac * xm[k]
			ct += ac * tc.t2[k]
			mag += ac * cabs1(xc[k])
		}
		d := 0.0
		if br.d != nil {
			d = br.d[i*br.dStride+j]
			re.add(d)
		}
		re.add(-real(h))
		im.add(-imag(h))
		o := termsAbs(complex(re.value(), im.value()))*(1+2*termsEps) + dot2Err(n+2)*(mag+math.Abs(d)+cabs1(h))*(1+float64(n+4)*termsEps) + float64(4*n+16)*0x1p-1074
		t2 := theta * theta * ct * grow
		if !(o <= math.MaxFloat64 && t2 <= math.MaxFloat64) {
			return false
		}
		lo := max(l*(1-2*float64(n)*termsEps)-t2, 0) / (1 + theta)
		if ib != nil {
			lo = max(lo, ib.xLow(br, i, j))
		}
		if !(o+t2 <= ((u-theta)*lo+u*math.Abs(d))*(1-4*termsEps)) {
			return false
		}
	}
	return true
}

// certify runs steps 1–5 on x (entry (k, j) at x[k*ks+j*js]) and dst, with
// |M| bounded by the comparison matrices of cb, or by ib where ib is not
// nil. The tail is formed once per point on s summed over the columns, and
// θ' is the largest over the columns. rowsOK reports step 1 alone, which
// refinement can still fix.
func (tc *termsCert) certify(br *balancedRealization, pt frequencyPoint, x []complex128, ks, js int, dst []complex128, cb comparisonBounds, ib *inverseBound) (rowsOK, ok bool) {
	n := tc.n
	clear(tc.sAll)
	thetaR, cheap := 0.0, false
	for j := range tc.m {
		tc.load(x, ks, js, j)
		ok, th, ch := tc.rows(br, pt, j, false)
		if !ok {
			return false, false
		}
		thetaR, cheap = max(thetaR, th), cheap || ch
		for i, v := range tc.s[j*n : (j+1)*n] {
			tc.sAll[i] += v
		}
	}
	up := tc.comparisonUpper(cb)
	if ib != nil {
		up = tc.minUpper(up, ib.upper)
	}
	if !tc.tail(up) {
		return true, false
	}
	pass := func(th float64) bool {
		for j := range tc.m {
			if !tc.output(br, dst, j, th, ib) {
				return false
			}
		}
		return true
	}
	if pass(thetaR) {
		return true, true
	}
	if !cheap {
		return true, false
	}
	thetaR = 0
	for j := range tc.m {
		_, th, _ := tc.rows(br, pt, j, true)
		thetaR = max(thetaR, th)
	}
	return true, pass(thetaR)
}

// inverseBound is the approximate-inverse tier of step 3 at a point the
// comparison matrices do not certify. Y ≈ (λI − Â)⁻¹ comes from the LU
// factors; R̄ ≥ |I − (λI − Â)Y| bounds the right residual against the
// stored pencil P = λI − Â, its rounding included. GEPP keeps the right
// residual near ε|L̂||Û||Y| while the left one, I − YP, can exceed 1 beside
// a pole cluster (hunt-true/496: 12 against 1.6e-12), and M = P⁻¹ =
// Y(PY)⁻¹ = Y(I − R)⁻¹ needs only the right one. Where R̄v ≤ qv for some
// v > 0 with q < 1/2, |M| ≤ |Y|(I − R̄)⁻¹ (upper) and, as M − Y =
// Y(I − R)⁻¹R, |M| ≥ |Y| − |Y|R̄(I − R̄)⁻¹ (Mlow, clamped at 0); a Neumann
// tail Σₖ≥₁ R̄ᵏz is at most maxₗ(zₗ/vₗ)·q/(1−q)·v. X ≥
// |c|Mlow|Â|Mlow|b| + |c|Mlow|b| (xLow), as every factor is nonnegative.
// v = 𝟙 is the plain row-sum test; where it fails, v = 𝟙 + 4R̄v iterated
// approximates the Perron vector of R̄ (any v > 0 is sound, the weights
// only sharpen q): R̄'s rounding part γ|Â||Y| makes its row sums reach 6
// at ρ(R̄) = 8e-13 where Â's rows differ in scale (hunt-false/5123).
type inverseBound struct {
	n    int
	q    float64
	v    []float64 // > 0, R̄v ≤ qv
	rbar []float64 // n×n
	yUp  []float64 // n×n, ≥ |Y|
	mLow []float64 // n×n, ≤ |M|
	absA []float64
	w, z []float64
	r1   []float64
}

// newInverseBound builds the tier at pt for br, with solve overwriting the
// row-major n×n identity with Y; nil where q ≥ 1/2 or a bound is not
// finite.
func newInverseBound(br *balancedRealization, absA []float64, pt frequencyPoint, solve func(y []complex128)) *inverseBound {
	n := br.n
	y := make([]complex128, n*n)
	for i := range n {
		y[i*n+i] = 1
	}
	solve(y)
	lam := pt.value()
	la := cabs1(lam)
	ib := &inverseBound{n: n, rbar: make([]float64, n*n), yUp: make([]float64, n*n), mLow: make([]float64, n*n), absA: absA,
		w: make([]float64, n), z: make([]float64, n), r1: make([]float64, n)}
	for i, v := range y {
		ib.yUp[i] = cabs1(v)
		if !(ib.yUp[i] <= math.MaxFloat64) {
			return nil
		}
	}
	gamma := float64(n+8) * termsEps
	re, im, ya := make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range n {
		clear(re)
		clear(im)
		clear(ya)
		for k, a := range br.a[i*n : (i+1)*n] {
			if a == 0 {
				continue
			}
			aa := absA[i*n+k]
			for j, v := range y[k*n : (k+1)*n] {
				re[j] += a * real(v)
				im[j] += a * imag(v)
			}
			for j, v := range ib.yUp[k*n : (k+1)*n] {
				ya[j] += aa * v
			}
		}
		for j := range n {
			r := complex(re[j], im[j]) - lam*y[i*n+j]
			e := gamma * (la*ib.yUp[i*n+j] + ya[j])
			if i == j {
				r += 1
				e += gamma
			}
			ib.rbar[i*n+j] = cabs1(r)*(1+2*termsEps) + e
		}
	}
	if !ib.weigh() {
		ib.compensatedResidual(br, pt, y)
		if !ib.weigh() {
			return nil
		}
	}
	f := ib.q / (1 - ib.q) * (1 + 4*termsEps)
	cmax := make([]float64, n)
	for k := range n {
		for j, r := range ib.rbar[k*n : (k+1)*n] {
			cmax[j] = max(cmax[j], r/ib.v[k])
		}
	}
	grow := 1 + 4*float64(n)*termsEps
	e := make([]float64, n)
	for i := range n {
		clear(e)
		var tail float64
		for k, a := range ib.yUp[i*n : (i+1)*n] {
			if a == 0 {
				continue
			}
			for j, r := range ib.rbar[k*n : (k+1)*n] {
				e[j] += a * r
			}
			tail += a * ib.v[k]
		}
		for j := range n {
			ej := (e[j] + tail*cmax[j]*f) * grow
			ib.mLow[i*n+j] = max(termsAbs(y[i*n+j])*(1-2*termsEps)-ej, 0)
		}
	}
	return ib
}

// weigh sets ib.v and ib.q < 1/2 with R̄v ≤ qv, reporting false where the
// iteration finds none, at once where minᵢ (R̄v)ᵢ/vᵢ ≤ ρ(R̄) shows there is
// none.
func (ib *inverseBound) weigh() bool {
	n := ib.n
	if ib.v == nil {
		ib.v = make([]float64, n)
	}
	for i := range ib.v {
		ib.v[i] = 1
	}
	rv := ib.z
	for it := 0; ; it++ {
		ib.q = 0
		lo := math.Inf(1)
		for i := range n {
			var r float64
			for k, a := range ib.rbar[i*n : (i+1)*n] {
				r += a * ib.v[k]
			}
			rv[i] = r
			ib.q = max(ib.q, r/ib.v[i]*(1+float64(n+2)*termsEps))
			lo = min(lo, r/ib.v[i])
		}
		if ib.q < 0.5 {
			return true
		}
		if lo >= 0.5 || it == n {
			return false
		}
		for i, r := range rv {
			ib.v[i] = 1 + 4*r
		}
		if !(slices.Max(ib.v) <= math.MaxFloat64) {
			return false
		}
	}
}

// compensatedResidual sets R̄ again from I − (λI − Â)Y formed against the
// exact pencil in compensated arithmetic, as termsCert.rows forms a row it
// cannot pass in working precision: the working-precision rounding bound
// γ(|λ||Y| + |Â||Y|) alone holds ρ(R̄) near η·(n+8)/(4(n+5)) + ε|λ||Y|, above
// 1/2 as η nears 1 (sweepInt decoupled/0.1/false/3 at ω = 3.2e-13:
// ρ(R̄) = 0.53 against ‖I − PY‖ = 6e-15).
func (ib *inverseBound) compensatedResidual(br *balancedRealization, pt frequencyPoint, y []complex128) {
	n := ib.n
	kappa := complex(1, 0)
	if !pt.isPlain() {
		kappa = pt.scale()
	}
	km := termsAbs(kappa)
	col := make([]complex128, n)
	for j := range n {
		for k := range n {
			col[k] = y[k*n+j]
		}
		for i := range n {
			b := 0.0
			if i == j {
				b = 1
			}
			rc, e := pencilResidualRow(col, br.a[i*n:(i+1)*n], i, b, pt.p, pt.q, kappa)
			if kappa != 1 {
				rc *= crecip(kappa)
			}
			ib.rbar[i*n+j] = termsAbs(rc)*(1+4*termsEps)*(1+2*termsEps) + e/km*(1+4*termsEps)
		}
	}
}

// upper is inverseBound's mUpper:|Y|(I − R̄)⁻¹y ≤ |Y|(y + z + maxₗ(zₗ/vₗ)·
// q/(1−q)·v), z = R̄y.
func (ib *inverseBound) upper(y []float64) bool {
	n := ib.n
	mx := 0.0
	for i := range n {
		var v float64
		for k, r := range ib.rbar[i*n : (i+1)*n] {
			v += r * y[k]
		}
		ib.z[i] = v
		mx = max(mx, v/ib.v[i])
	}
	f := mx * ib.q / (1 - ib.q) * (1 + 4*termsEps)
	for i := range n {
		ib.w[i] = y[i] + ib.z[i] + f*ib.v[i]
	}
	grow := 1 + 4*float64(n)*termsEps
	for i := range n {
		var v float64
		for k, a := range ib.yUp[i*n : (i+1)*n] {
			v += a * ib.w[k]
		}
		y[i] = v * grow
		if !(y[i] <= math.MaxFloat64) {
			return false
		}
	}
	return true
}

// xLow is the lower bound |cᵢ|Mlow|Â|Mlow|bⱼ| + |cᵢ|Mlow|bⱼ| on entry
// (i, j)'s X, rounded down.
func (ib *inverseBound) xLow(br *balancedRealization, i, j int) float64 {
	n, m := ib.n, br.m
	r1, w, z := ib.r1, ib.w, ib.z
	clear(r1)
	for k, c := range br.c[i*n : (i+1)*n] {
		if c == 0 {
			continue
		}
		ac := math.Abs(c)
		for l, v := range ib.mLow[k*n : (k+1)*n] {
			r1[l] += ac * v
		}
	}
	clear(w)
	for k, v := range r1 {
		if v == 0 {
			continue
		}
		for l, a := range ib.absA[k*n : (k+1)*n] {
			w[l] += v * a
		}
	}
	clear(z)
	for k, v := range w {
		if v == 0 {
			continue
		}
		for l, a := range ib.mLow[k*n : (k+1)*n] {
			z[l] += v * a
		}
	}
	var x float64
	for k := range n {
		b := math.Abs(br.b[k*m+j])
		x += (z[k] + r1[k]) * b
	}
	return x * (1 - 8*float64(n)*termsEps)
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
// is the product of the blocks' determinants. A block whose exact integer
// system is nonsingular modulo a prime (exactNonsingularModP) is
// nonsingular; only the others take the fraction-free solve.
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
		if ip, _, _, _, _ := exactPencilSystem(ps, pt); exactNonsingularModP(ip, 2*k, 2*k+1) {
			continue
		}
		if _, _, ok := solveExactAt(ps, pt); !ok {
			return true
		}
	}
	return false
}

// termsPassCount, termsDDCount, termsExactCount, termsPoleCount and
// termsUnverifiedCount count how evalWithinTerms settled each point: the
// certified double solve (refined where termsRefinedCount also counts it,
// through the approximate-inverse tier where termsInverseCount does),
// the verified double-double response, the exact response, the pole limit
// at an exactly singular pencil, or errTermsUnverified; for the census.
var termsPassCount, termsRefinedCount, termsInverseCount, termsDDCount, termsExactCount, termsPoleCount, termsUnverifiedCount atomic.Int64
