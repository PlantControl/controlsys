package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/mat"
)

// errMatchedUnverified comes with the matched model when its DC gain, or
// the root set it maps, is not verified against the DCGain of the caller's
// model; C2D and D2C return it as ErrAccuracyUnverified.
var errMatchedUnverified = errors.New("matched model not verified")

func matchedUnverified(format string, args ...any) error {
	return fmt.Errorf("matched: "+format+": %w", append(args, errMatchedUnverified)...)
}

// carriesModel reports an error that comes with the model it describes.
func carriesModel(err error) bool {
	return errors.Is(err, errPencilFitUncertified) || errors.Is(err, errMatchedUnverified) || errors.Is(err, errHoldUnverified)
}

// matchedReference is the DC gain g0 of the caller's delay-free SISO model,
// at s = 0 or z = 1, with its terms T0 = |D| + Σ|C||X| and the descriptor
// bar u·(‖C‖‖M‖·(‖A‖ + |x0|‖E‖)·‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + |D|), M = (x0·E − A)⁻¹,
// u = 4(n+5)ε, Frobenius norms (E = I when nil). err is set where DCGain
// fails, is unverified, or is 0 or non-finite.
type matchedReference struct {
	g0, terms, bar float64
	err            error
	// zero reports a transfer function exactly zero on the stored data
	// (matchedExactZero): the matched model is then the zero gain, nil.
	zero bool
	// barLow ≤ bar takes ‖X‖/‖B‖ for ‖M‖; where it does not decide, bar is
	// formed from M = (x0·E − A)⁻¹ by the LU factors lu, ipiv (fullBar).
	barLow      float64
	lu          []float64
	ipiv        []int
	barNorms    [5]float64
	barX0, barU float64
}

// barOf is the bar for ‖M‖ = normM.
func (r matchedReference) barOf(normM float64) float64 {
	normA, normB, normC, normE, d := r.barNorms[0], r.barNorms[1], r.barNorms[2], r.barNorms[3], r.barNorms[4]
	cm := normC * normM
	return r.barU * (cm*(normA+r.barX0*normE)*normM*normB + cm*normB + d)
}

// fullBar is the bar with ‖M‖ formed from the LU factors.
func (r matchedReference) fullBar() float64 {
	n := len(r.ipiv)
	inv := make([]float64, n*n)
	for i := range n {
		inv[i*n+i] = 1
	}
	impl.Dgetrs(blas.NoTrans, n, n, r.lu, n, r.ipiv, inv, n)
	s := 0.0
	for _, v := range inv {
		s += v * v
	}
	return r.barOf(math.Sqrt(s))
}

func newMatchedReference(ref *System) matchedReference {
	free := ioDelayFree(ref)
	g, err := free.DCGain()
	return matchedReferenceAt(free, g, err)
}

// matchedReferenceAt is newMatchedReference of the delay-free free whose
// DCGain returned g, err.
func matchedReferenceAt(free *System, g *mat.Dense, err error) matchedReference {
	if (err != nil || g.At(0, 0) == 0) && matchedExactZero(free) {
		return matchedReference{zero: true}
	}
	if err != nil {
		return matchedReference{err: matchedUnverified("DCGain of the model: %v", err)}
	}
	g0 := g.At(0, 0)
	if g0 == 0 || math.IsInf(g0, 0) || math.IsNaN(g0) {
		return matchedReference{err: matchedUnverified("DCGain of the model is %v", g0)}
	}
	return newDCBar(free).matched(g0)
}

// ioDelayFree returns sys without its Delay, InputDelay and OutputDelay.
func ioDelayFree(sys *System) *System {
	if !sys.HasDelay() {
		return sys
	}
	c := *sys
	c.Delay, c.InputDelay, c.OutputDelay = nil, nil, nil
	return &c
}

// dcBar is, for a delay-free model and each entry (i, j) of its gain at
// s = 0 or z = 1, the terms T0_ij = |D_ij| + Σ_l |C_il||X_lj|, X = M·B,
// and the descriptor bar u·(‖C_i‖‖M‖·(‖A‖ + |x0|‖E‖)·‖M‖‖B_j‖ +
// ‖C_i‖‖M‖‖B_j‖ + |D_ij|), M = (x0·E − A)⁻¹, u = 4(n+5)ε, Frobenius norms
// (E = I when nil), row C_i and column B_j. reason is set where no bar
// exists (an exactly singular pencil, a bar not finite).
type dcBar struct {
	m      int
	terms  []float64
	barLow []float64
	// bar is the bar itself where lu is nil; otherwise barLow ≤ bar takes
	// ‖X_j‖/‖B_j‖ for ‖M‖ and fullBar forms ‖M‖ from the LU factors.
	bar   []float64
	lu    []float64
	ipiv  []int
	normM float64
	// factor reports lu holding x0·E − A not yet factored.
	factor       bool
	normA, normE float64
	normC, normB []float64
	absD         []float64
	x0, u        float64
	reason       string
}

func newDCBar(free *System) dcBar { return newDCBarSolved(free, nil) }

// newDCBarSolved is newDCBar from X = M·B already solved (DCGain's
// certified solve) where solved is not nil; x0·E − A is then factored only
// where fullBar needs it.
func newDCBarSolved(free *System, solved []dd) dcBar {
	n, m, p := free.Dims()
	b := dcBar{m: m, u: 4 * float64(n+5) * 0x1p-52, terms: make([]float64, p*m), absD: make([]float64, p*m)}
	for i := range p {
		for j := range m {
			b.absD[i*m+j] = math.Abs(free.D.At(i, j))
		}
	}
	if n == 0 {
		copy(b.terms, b.absD)
		b.bar = make([]float64, p*m)
		for k, d := range b.absD {
			b.bar[k] = b.u * d
		}
		return b
	}
	if free.IsDiscrete() {
		b.x0 = 1
	}
	b.normE = math.Sqrt(float64(n))
	buf := make([]float64, n*n+n*m)
	lu, x := buf[:n*n], buf[n*n:]
	ar := free.A.RawMatrix()
	for i := range n {
		for j := range n {
			lu[i*n+j] = -ar.Data[i*ar.Stride+j]
		}
	}
	if free.E != nil {
		b.normE = mat.Norm(free.E, 2)
		if b.x0 != 0 {
			er := free.E.RawMatrix()
			for i := range n {
				for j := range n {
					lu[i*n+j] += er.Data[i*er.Stride+j]
				}
			}
		}
	} else if b.x0 != 0 {
		for i := range n {
			lu[i*n+i] += b.x0
		}
	}
	b.normA = mat.Norm(free.A, 2)
	b.normC, b.normB = make([]float64, p), make([]float64, m)
	for i := range p {
		b.normC[i] = mat.Norm(free.C.Slice(i, i+1, 0, n), 2)
	}
	for j := range m {
		b.normB[j] = mat.Norm(free.B.Slice(0, n, j, j+1), 2)
	}
	ipiv := make([]int, n)
	if solved != nil {
		for k := range x {
			x[k] = solved[k].hi
		}
		b.lu, b.ipiv, b.factor = lu, ipiv, true
	} else {
		if !impl.Dgetrf(n, n, lu, n, ipiv) {
			return b.singular(free)
		}
		br := free.B.RawMatrix()
		for i := range n {
			copy(x[i*m:i*m+m], br.Data[i*br.Stride:i*br.Stride+m])
		}
		impl.Dgetrs(blas.NoTrans, n, m, lu, n, ipiv, x, max(m, 1))
		b.lu, b.ipiv = lu, ipiv
	}
	b.barLow = make([]float64, p*m)
	for j := range m {
		normX := 0.0
		for l := range n {
			normX += x[l*m+j] * x[l*m+j]
		}
		for i := range p {
			k := i*m + j
			terms := b.absD[k]
			for l := range n {
				terms += math.Abs(free.C.At(i, l) * x[l*m+j])
			}
			b.terms[k] = terms
			b.barLow[k] = b.absD[k] * b.u
			if b.normB[j] > 0 {
				b.barLow[k] = b.barOf(k, math.Sqrt(normX)/b.normB[j])
			}
			if math.IsNaN(b.barLow[k]) || math.IsInf(b.barLow[k], 0) || math.IsNaN(terms) || math.IsInf(terms, 0) {
				b.reason = "DC bar not finite"
			}
		}
	}
	return b
}

// barOf is the bar of entry k for ‖M‖ = normM.
func (b dcBar) barOf(k int, normM float64) float64 {
	cm := b.normC[k/b.m] * normM
	return b.u * (cm*(b.normA+b.x0*b.normE)*normM*b.normB[k%b.m] + cm*b.normB[k%b.m] + b.absD[k])
}

// fullBar is the bar of entry k with ‖M‖ formed from the LU factors,
// computed once.
func (b *dcBar) fullBar(k int) float64 {
	if b.factor {
		b.factor = false
		if !impl.Dgetrf(len(b.ipiv), len(b.ipiv), b.lu, len(b.ipiv), b.ipiv) {
			b.normM = math.Inf(1)
		}
	}
	if b.normM == 0 {
		n := len(b.ipiv)
		inv := make([]float64, n*n)
		for i := range n {
			inv[i*n+i] = 1
		}
		impl.Dgetrs(blas.NoTrans, n, n, b.lu, n, b.ipiv, inv, n)
		s := 0.0
		for _, v := range inv {
			s += v * v
		}
		b.normM = math.Sqrt(s)
	}
	return b.barOf(k, b.normM)
}

// within reports need ≤ bar + allow for entry k.
func (b *dcBar) within(k int, need, allow float64) bool {
	if b.lu == nil {
		return need <= b.bar[k]+allow
	}
	return need <= b.barLow[k]+allow || need <= b.fullBar(k)+allow
}

// matched is the matchedReference of the SISO gain g0 with this bar.
func (b dcBar) matched(g0 float64) matchedReference {
	if b.reason != "" {
		return matchedReference{err: matchedUnverified("%s", b.reason)}
	}
	r := matchedReference{g0: g0, terms: b.terms[0]}
	if b.normC == nil {
		r.bar = b.bar[0]
		return r
	}
	r.barX0, r.barU = b.x0, b.u
	r.barNorms = [5]float64{b.normA, b.normB[0], b.normC[0], b.normE, b.absD[0]}
	if b.lu == nil {
		r.bar = b.bar[0]
		return r
	}
	r.barLow, r.lu, r.ipiv = b.barLow[0], b.lu, b.ipiv
	return r
}

// rootsAgree is the ZPK within-terms inequality at DC for the DC product v
// of a root set with its own gain: |v − g0| ≤ √ε·max(|g0|, |v|, T0).
func (r matchedReference) rootsAgree(v complex128) bool {
	tol := math.Sqrt(0x1p-52) * math.Max(math.Max(math.Abs(r.g0), cmplx.Abs(v)), r.terms)
	return cmplx.Abs(v-complex(r.g0, 0)) <= tol
}

// dcAgrees reports |h − g0| + err(h) + ε·T0 ≤ bar for the DC gain h of the
// result, exact for its stored entries up to h.err; ok false is undecided.
func (r matchedReference) dcAgrees(h matchedDC, ok bool) bool {
	if r.zero {
		return true
	}
	if !ok || math.IsNaN(h.err) || math.IsInf(h.err, 0) {
		return false
	}
	diff := mddAdd(h, mddFloat(-r.g0))
	need := mddAbs(diff) + h.err + 0x1p-52*r.terms
	if r.lu == nil {
		return need <= r.bar
	}
	return need <= r.barLow || need <= r.fullBar()
}

// matchedGain returns the gain that matches g0 for the discrete or
// continuous roots of the result, whose monic DC product is ratio, and
// the root gate's verdict on the DC product v of the root set mapped.
// Without a reference (r.err) it returns base, the root-product gain.
func (r matchedReference) matchedGain(v, ratio complex128, base float64) (float64, error) {
	if r.zero {
		return 0, nil
	}
	if r.err != nil {
		return base, r.err
	}
	q := real(ratio)
	if q == 0 || math.IsInf(q, 0) || math.IsNaN(q) {
		return base, matchedUnverified("DC product of the mapped roots is %v", q)
	}
	gain := r.g0 / q
	if !r.rootsAgree(v) {
		return gain, matchedUnverified("DC product %v of the roots is not within terms of DCGain %v", real(v), r.g0)
	}
	return gain, nil
}

func matchedDiscreteProduct(k float64, zeros, poles []complex128, artificial int) complex128 {
	v := complex(k, 0)
	for _, z := range zeros {
		v *= 1 - z
	}
	for range artificial {
		v *= 2
	}
	for _, p := range poles {
		v /= 1 - p
	}
	return v
}

func matchedContinuousProduct(k float64, zeros, poles []complex128) complex128 {
	v := complex(k, 0)
	for _, z := range zeros {
		v *= -z
	}
	for _, p := range poles {
		v /= -p
	}
	return v
}

// matchedPolish refines each root of the polynomial (descending powers)
// by Newton steps on the polynomial itself until its residual is within
// the a priori bound 4√2·n·u·Σ|aᵢ||x|^{n−i}, u = 2⁻⁵³, of Horner's rule in
// complex arithmetic, a step no longer lowers |p|, p′ vanishes, or 60
// steps. A real root stays real and a conjugate pair moves together.
func matchedPolish(coef []float64, roots []complex128) []complex128 {
	for len(coef) > 1 && coef[0] == 0 {
		coef = coef[1:]
	}
	out := slices.Clone(roots)
	done := make([]bool, len(out))
	for i, r := range roots {
		if done[i] || imag(r) < 0 {
			continue
		}
		x := r
		px, dpx, mu := matchedHorner(coef, x)
		for range 60 {
			if cmplx.Abs(px) <= mu || dpx == 0 {
				break
			}
			xn := x - px/dpx
			if imag(r) == 0 {
				xn = complex(real(xn), 0)
			}
			pn, dn, mun := matchedHorner(coef, xn)
			if !(cmplx.Abs(pn) < cmplx.Abs(px)) {
				break
			}
			x, px, dpx, mu = xn, pn, dn, mun
		}
		out[i] = x
		done[i] = true
		if imag(r) > 0 {
			best := -1
			for k, q := range roots {
				if !done[k] && imag(q) < 0 && (best < 0 || cmplx.Abs(q-cmplx.Conj(r)) < cmplx.Abs(roots[best]-cmplx.Conj(r))) {
					best = k
				}
			}
			if best >= 0 {
				out[best] = cmplx.Conj(x)
				done[best] = true
			}
		}
	}
	return out
}

func matchedHorner(coef []float64, x complex128) (p, dp complex128, bound float64) {
	ax := cmplx.Abs(x)
	s := 0.0
	for _, c := range coef {
		dp = dp*x + p
		p = p*x + complex(c, 0)
		s = s*ax + math.Abs(c)
	}
	return p, dp, 4 * math.Sqrt2 * float64(len(coef)-1) * 0x1p-53 * s
}

const matchedDCPrec = 192

// matchedDC is a DC gain hi + lo in double-double arithmetic with an
// absolute bound err on its rounding, first order in u = 2⁻¹⁰⁰ per
// operation, against the exact value for the stored entries.
type matchedDC struct {
	hi, lo, err float64
}

const matchedDDUnit = 0x1p-100

func mddTwoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	return s, (a - (s - bb)) + (b - bb)
}

func mddAdd(x, y matchedDC) matchedDC {
	s, e := mddTwoSum(x.hi, y.hi)
	e += x.lo + y.lo
	hi := s + e
	return matchedDC{hi: hi, lo: e - (hi - s)}
}

func mddMul(x, y matchedDC) matchedDC {
	p := x.hi * y.hi
	e := math.FMA(x.hi, y.hi, -p) + x.hi*y.lo + x.lo*y.hi
	hi := p + e
	return matchedDC{hi: hi, lo: e - (hi - p)}
}

func mddDiv(x, y matchedDC) matchedDC {
	q1 := x.hi / y.hi
	r := mddAdd(x, mddMul(matchedDC{hi: -q1}, y))
	q2 := r.hi / y.hi
	r = mddAdd(r, mddMul(matchedDC{hi: -q2}, y))
	q3 := r.hi / y.hi
	return mddAdd(mddAdd(matchedDC{hi: q1}, matchedDC{hi: q2}), matchedDC{hi: q3})
}

func mddFloat(v float64) matchedDC { return matchedDC{hi: v} }

func mddSub(a, b float64) matchedDC {
	s, e := mddTwoSum(a, -b)
	return matchedDC{hi: s, lo: e}
}

func mddAbs(x matchedDC) float64 { return math.Abs(x.hi + x.lo) }

// matchedSectionDC is the section's gain d + c·(x0·I − A)⁻¹·b at x0, from
// its stored entries, with its rounding bound; ok is false where an entry
// is not finite, x0·I − A is singular, or its determinant is not resolved.
func matchedSectionDC(a, b, c []float64, d, x0 float64) (matchedDC, bool) {
	if !finiteSlice(a) || !finiteSlice(b) || !finiteSlice(c) || !finiteSlice([]float64{d}) {
		return matchedDC{}, false
	}
	u := matchedDDUnit
	if len(b) == 1 {
		den := mddSub(x0, a[0])
		if den.hi == 0 {
			return matchedDC{}, false
		}
		q := mddDiv(mddMul(mddFloat(c[0]), mddFloat(b[0])), den)
		s := mddAdd(q, mddFloat(d))
		s.err = u * (2*mddAbs(q) + math.Abs(d))
		return s, true
	}
	m00, m11 := mddSub(x0, a[0]), mddSub(x0, a[3])
	m01, m10 := mddFloat(-a[1]), mddFloat(-a[2])
	p1, p2 := mddMul(m00, m11), mddMul(m01, m10)
	det := mddAdd(p1, matchedDC{hi: -p2.hi, lo: -p2.lo})
	eDet := 2 * u * (mddAbs(p1) + mddAbs(p2))
	if !(eDet < mddAbs(det)/2) {
		return matchedDC{}, false
	}
	t1, t2 := mddMul(m11, mddFloat(b[0])), mddMul(m01, mddFloat(b[1]))
	x0v := mddAdd(t1, matchedDC{hi: -t2.hi, lo: -t2.lo})
	e0 := 2 * u * (mddAbs(t1) + mddAbs(t2))
	t3, t4 := mddMul(m00, mddFloat(b[1])), mddMul(m10, mddFloat(b[0]))
	x1v := mddAdd(t3, matchedDC{hi: -t4.hi, lo: -t4.lo})
	e1 := 2 * u * (mddAbs(t3) + mddAbs(t4))
	n0, n1 := mddMul(mddFloat(c[0]), x0v), mddMul(mddFloat(c[1]), x1v)
	num := mddAdd(n0, n1)
	eNum := math.Abs(c[0])*e0 + math.Abs(c[1])*e1 + 2*u*(mddAbs(n0)+mddAbs(n1))
	q := mddDiv(num, det)
	eq := (mddAbs(q)*(eDet/mddAbs(det)+u) + eNum/mddAbs(det)) / (1 - eDet/mddAbs(det))
	s := mddAdd(q, mddFloat(d))
	s.err = eq + u*(mddAbs(q)+math.Abs(d))
	return s, true
}

// matchedDCTimes is the product of the DC gains x and y with its bound.
func matchedDCTimes(x, y matchedDC) matchedDC {
	p := mddMul(x, y)
	p.err = mddAbs(x)*y.err + mddAbs(y)*x.err + x.err*y.err + matchedDDUnit*mddAbs(p)
	return p
}

// matchedExactZero reports a transfer function that is exactly zero on the
// stored data: D = 0 and no states, B = 0 or C = 0, or for an explicit
// model C·Aᵏ·B = 0 for every k (exactZeroTransfer). Anything undecided is
// not zero.
func matchedExactZero(sys *System) bool {
	if !zeroDense(sys.D) {
		return false
	}
	n, _, _ := sys.Dims()
	if n == 0 || zeroDense(sys.B) || zeroDense(sys.C) {
		return true
	}
	return sys.E == nil && exactZeroTransfer(sys.A, sys.B, sys.C) == transferZero
}

// matchedSingularLU is the matchedReference of dcBar.singular for the
// SISO sys of gain g0.
func matchedSingularLU(sys *System, g0, x0, u, normE, d float64) matchedReference {
	b := dcBar{m: 1, u: u, x0: x0, normE: normE, normA: mat.Norm(sys.A, 2),
		normC: []float64{mat.Norm(sys.C, 2)}, normB: []float64{mat.Norm(sys.B, 2)},
		absD: []float64{d}, terms: make([]float64, 1)}
	return b.singular(sys).matched(g0)
}

// singular is newDCBar where the LU of x0·E − A meets an exactly zero
// pivot. An exactly singular pencil (exactShiftedDetIsZero, explicit or
// continuous, order up to exactDetMaxOrder) has no bar. Otherwise the
// pencil is only ill-conditioned, and X and ‖M‖ come from its singular
// values.
func (b dcBar) singular(sys *System) dcBar {
	const singular = "DC pencil not invertible"
	b.reason = singular
	n, m, p := sys.Dims()
	if n > exactDetMaxOrder || (b.x0 != 0 && sys.E != nil) {
		return b
	}
	if exactShiftedDetIsZero(sys.A.RawMatrix(), b.x0) {
		return b
	}
	k := mat.NewDense(n, n, nil)
	k.Scale(-1, sys.A)
	if sys.E != nil {
		if b.x0 != 0 {
			k.Add(k, sys.E)
		}
	} else {
		for i := range n {
			k.Set(i, i, k.At(i, i)+b.x0)
		}
	}
	var svd mat.SVD
	if !svd.Factorize(k, mat.SVDThin) {
		return b
	}
	sv := svd.Values(nil)
	var uu, vv mat.Dense
	svd.UTo(&uu)
	svd.VTo(&vv)
	sumInv := 0.0
	for _, v := range sv {
		if v == 0 {
			return b
		}
		sumInv += 1 / (v * v)
	}
	var utb mat.Dense
	utb.Mul(uu.T(), sys.B)
	for i, v := range sv {
		for j := range m {
			utb.Set(i, j, utb.At(i, j)/v)
		}
	}
	var x mat.Dense
	x.Mul(&vv, &utb)
	b.reason = ""
	b.bar = make([]float64, p*m)
	for i := range p {
		for j := range m {
			e := i*m + j
			terms := b.absD[e]
			for l := range n {
				terms += math.Abs(sys.C.At(i, l) * x.At(l, j))
			}
			b.terms[e] = terms
			b.bar[e] = b.barOf(e, math.Sqrt(sumInv))
			if math.IsNaN(b.bar[e]) || math.IsInf(b.bar[e], 0) || math.IsNaN(terms) || math.IsInf(terms, 0) {
				b.reason = singular
			}
		}
	}
	return b
}
