package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// zerosRankFloorK is the K of the rank floor at data rounding (GVD7AS
// ruling 5 addendum, user decision): det G (an entry of G, at rank 0) is
// within the data's rounding of zero where it is at most K·ε·T_data,
// T_data the first-order image of ε-rounding every stored entry.
const zerosRankFloorK = 8

// zerosCertMaxRank bounds the k of the k×k determinants the certificate
// expands by permutations (k! terms); beyond it a model is left open.
const zerosCertMaxRank = 4

// zerosCertVerdict is the outcome of zerosCertify.
type zerosCertVerdict int8

const (
	zerosCertOpen zerosCertVerdict = iota
	zerosCertVerified
	zerosCertRefuted
)

// Evaluation levels of a point: double, double-double, exact rational.
const (
	zerosLevelDouble int8 = iota
	zerosLevelDD
	zerosLevelExact
)

// Point states after judging.
const (
	zerosPtPending int8 = iota
	zerosPtVerified
	zerosPtConcededPole
	zerosPtConcededDrop
	zerosPtConcededUndet
	zerosPtOpen
	zerosPtConcededWindow
)

// zerosCertPoint is det G at one axis point with its bounds: d the
// computed det Ĝ (k×k response, D included) with |d − det G| ≤ dd, T_detG =
// Σ|adj G_ji|·T_ij in [tLo, tHi], T_ij = Σ_l |C_il||X_lj| + |D_ij| the
// entry's terms, and χ̂_A = chiM·2^chiE = det(sI − A) with relative bound
// relChi. valid is false where the level's bounds do not hold (η ≥ 1, a
// non-finite value) or T_detG has no positive lower bound (rank G ≤ k − 2
// to the precision reached); pole marks an exactly zero pivot, s an
// exact eigenvalue to the level's arithmetic; zero an exactly vanishing
// response and terms (exact level).
type zerosCertPoint struct {
	s           complex128
	level       int8
	valid, pole bool
	zero        bool
	d           complex128
	dd          float64
	tLo, tHi    float64
	chiM        complex128
	chiE        int
	relChi      float64
	state       int8
	g           []complex128
	ge, gl, gh  []float64
	eok         bool
	wantExact   bool
	drop, late  bool
}

// zerosCertEntries stores the k×k response G, its entry bounds e and the
// terms bounds tl ≤ T ≤ th in pt; an all-zero computed T sends the point
// to the exact rung, where only an exact zero response is decided.
func zerosCertEntries(pt *zerosCertPoint, G []complex128, e, tl, th []float64) {
	pt.g, pt.ge, pt.gl, pt.gh = slices.Clone(G), slices.Clone(e), slices.Clone(tl), slices.Clone(th)
	pt.eok = true
	allZero := true
	for i := range th {
		allZero = allZero && th[i] == 0
	}
	if allZero && pt.level < zerosLevelExact {
		pt.eok, pt.wantExact = false, true
	}
}

// zerosCertResult is the verdict of zerosCertify with the evaluated points
// and, for a nonzero candidate, the exhibited gain c = cm·2^ce of
// Z = c·Π(s − ẑ)/χ_A.
type zerosCertResult struct {
	verdict zerosCertVerdict
	pts     []zerosCertPoint
	nStruct int
	cm      float64
	ce      int
	hasC    bool
	capped  bool
}

// zerosCertMode selects what zerosCertify judges: the candidate zeros
// against det G (zerosModeCand), det G ≡ 0, normal rank below k
// (zerosModeDet0), or G ≡ 0 entrywise, normal rank 0 (zerosModeEntry0).
type zerosCertMode int8

const (
	zerosModeCand zerosCertMode = iota
	zerosModeDet0
	zerosModeEntry0
)

// zerosCertify judges the candidate zeros of the explicit square model sys
// against det G, G its k×k response: a point verifies when |det Ĝ − Z| +
// δ ≤ max(√ε·max(|det G|, |Z|), K·ε·T_data), Z = c·Π(s − ẑ)/χ_A and T_data
// the first-order image of ε-rounding the stored data (zerosRankFloorK),
// at the fixed probe set: the axis points beside every ẑ
// (eigenAxisProbes), the static point(s) and zpkFillProbes with 2(n+1)
// points. c is exhibited: any real c inside the intersection, over the
// points, of the sets each point's verification bounds admit; at a point
// where G and T_data vanish exactly only c = 0 is admitted, unless a ẑ
// lies there. A probe within √ε|ẑ| of a nonzero ẑ is not judged. Mode
// zerosModeDet0 judges det G ≡ 0 to the data's rounding (rank below k:
// every point within K·ε·T_data, one beyond refutes), and zerosModeEntry0
// every entry the same way (rank 0). A point a level leaves undecided is
// judged again in double-double (zpkDDFactor) and exactly
// (exactSolveAt); still undecided, or excluding the candidate, it leaves
// the row open; only an exact eigenvalue of A is conceded. A candidate
// verifies with n + 1 conjugate-weighted points both verified and
// determined (det G beyond K·ε·T_data); across a gap of the nonzero zeros
// each side needs its verified points (frequencyAxisCutAcrossGap). A
// rank-0 or rank-deficient candidate with an exactly vanishing response
// at n + 1 such points not at a pole is verified outright. The candidate
// is refuted when no real c satisfies the refutation bounds at the valid
// points. All of it is charged to the work budget b.
func zerosCertify(sys *System, zeros []complex128, mode zerosCertMode) zerosCertResult {
	return zerosCertifyB(sys, zeros, mode, newZerosBudget())
}

// zerosCertifyB is zerosCertify charging b; when b runs out the result is
// open with capped set.
func zerosCertifyB(sys *System, zeros []complex128, mode zerosCertMode, b *zerosBudget) zerosCertResult {
	n, m, p := sys.Dims()
	res := zerosCertResult{}
	det := mode != zerosModeEntry0
	if m != p || n == 0 || det && m > zerosCertMaxRank {
		return res
	}
	discrete := sys.Dt != 0
	pts, nStruct := zerosCertProbes(sys, zeros, discrete)
	if mode != zerosModeCand {
		zeros = nil
	}
	res.nStruct = nStruct
	n3 := int64(n) * int64(n) * int64(n)
	if !b.spend(int64(len(pts)) * (n3 + 1000)) {
		res.capped = true
		return res
	}
	cp := zerosCertDouble(sys, pts, det)
	check, checkOK := newZPKRowCheck(&System{A: sys.A, B: mat.NewDense(n, 1, nil), C: mat.NewDense(1, n, nil), D: mat.NewDense(1, 1, nil), Dt: sys.Dt})
	tol := math.Sqrt(eps())
	for q, s := range pts {
		cp[q].state = zerosPtPending
		if mode == zerosModeCand && slices.ContainsFunc(zeros, func(z complex128) bool { return z != 0 && cmplx.Abs(s-z) <= tol*cmplx.Abs(z) }) {
			cp[q].state = zerosPtConcededWindow
			continue
		}
		if checkOK {
			if drop, ok := check.dropped(s); ok && drop {
				cp[q].drop, cp[q].late = true, true
			}
		}
	}
	br := newBalancedRealization(sys, n, m, p)
	var ddw zpkDDWork
	escalate := func(q int, open bool) bool {
		pt := &cp[q]
		late := pt.late
		switch pt.level {
		case zerosLevelDouble:
			if !b.spend(5*n3 + 3000) {
				res.capped = true
				return false
			}
			*pt = zerosCertDD(sys, &br, &ddw, pt.s, det)
		case zerosLevelDD:
			if !b.spend(zerosExactPointCost) {
				res.capped = true
				return false
			}
			*pt = zerosCertExact(sys, pt.s, det, b.spend)
			if b.used > b.cap {
				res.capped = true
			}
		default:
			return false
		}
		pt.state, pt.late = zerosPtPending, late
		return true
	}
	usable := func(pt zerosCertPoint) bool {
		if det {
			return pt.valid
		}
		return pt.eok || pt.zero
	}
	var scale zerosCertScaleInfo
	var c float64
	cOK := false
	run := func() bool {
		for round := 0; round < 4; round++ {
			changed := false
			for q := range cp {
				if cp[q].state == zerosPtPending && !usable(cp[q]) && !cp[q].pole {
					changed = escalate(q, false) || changed
				}
			}
			scale = zerosCertScale(cp, zeros, mode)
			ver, ref := zerosCertSets(cp, zeros, mode, scale)
			if !ref.nonempty() {
				return false
			}
			c, cOK = ver.pick()
			if mode != zerosModeCand {
				cOK = true
			}
			for q := range cp {
				if cp[q].state != zerosPtPending || !usable(cp[q]) || cp[q].pole {
					continue
				}
				if !cOK || !zerosCertPointVerifies(cp[q], zeros, mode, c, scale) {
					changed = escalate(q, true) || changed
				}
			}
			if !changed {
				break
			}
		}
		return true
	}
	if !run() {
		res.verdict, res.pts = zerosCertRefuted, cp
		return res
	}
	allOK := cOK && !res.capped
	cut, sided := 0.0, false
	needSlow, needFast, slowScale := 0, 0, 0.0
	if !discrete && len(zeros) >= 2 {
		mask := make([]bool, len(zeros))
		for i, z := range zeros {
			mask[i] = z != 0
		}
		cut, sided = topGap(zeros, mask)
	}
	if sided {
		needSlow, needFast = 1, 1
		for _, z := range zeros {
			if cmplx.Abs(z) >= cut {
				needFast++
			} else {
				needSlow++
				slowScale = math.Max(slowScale, cmplx.Abs(z))
			}
		}
	}
	counted, slow, fast, zeroW := 0, 0, 0, 0
	for q := range cp {
		pt := &cp[q]
		switch {
		case pt.state == zerosPtConcededWindow:
			continue
		case pt.pole:
			pt.state = zerosPtConcededPole
			continue
		case !usable(*pt):
			pt.state = zerosPtOpen
			allOK = false
			continue
		case cOK && zerosCertPointVerifies(*pt, zeros, mode, c, scale):
			pt.state = zerosPtVerified
		default:
			pt.state = zerosPtOpen
			allOK = false
			continue
		}
		w := probeWeight(pts[q])
		if pt.zero {
			zeroW += w
		}
		if mode != zerosModeCand || zerosCertDetermined(*pt) {
			counted += w
		}
		if sided {
			switch a := cmplx.Abs(pts[q]); {
			case a >= cut:
				fast += w
			case a <= slowScale:
				slow += w
			}
		}
	}
	if counted < n+1 || slow < needSlow || fast < needFast {
		allOK = false
	}
	res.pts = cp
	if mode == zerosModeCand && cOK {
		res.cm, res.ce, res.hasC = c, scale.ec, true
	}
	if allOK || mode != zerosModeCand && zeroW >= n+1 && !res.capped {
		res.verdict = zerosCertVerified
	}
	return res
}

// zerosCertDetermined reports whether the point determines det G as
// nonzero beyond the data's rounding: |det Ĝ| − δ > K·ε·T_data. Only
// such points count toward a candidate's n + 1 points; elsewhere det G is
// within the data's rounding of zero and any candidate fits.
func zerosCertDetermined(pt zerosCertPoint) bool {
	if !pt.valid || pt.zero || pt.pole {
		return false
	}
	ad := cmplx.Abs(pt.d)
	return ad-pt.dd > zerosRankFloorK*eps()*pt.tHi
}

// zerosBudget is the deterministic work budget of one Zeros certification
// (Tier 1 and Tier 2 together), in units of about 4 ns on arm64: a point
// in double costs n³ + 1000, in double-double 5n³ + 3000, exactly
// zerosExactPointCost plus ratSubCost for every rational update of its
// eliminations (exactSolveAtSpend) and response and zerosRatDetCost for
// its determinant and cofactors; Tier 2 charges its pencil
// determinant N⁴·(N·w)², w the 64-bit words zerosExactWords gives the
// stored entries, root iterations 7200d² and the exact zero count
// 4n⁵w² + 50000 per compression. Work stops where a charge would exceed
// the cap, and the certification is then not verified.
type zerosBudget struct{ used, cap int64 }

// zerosWorkCap is the budget of one certification, about 0.6 s.
const zerosWorkCap = 140_000_000

// zerosExactPointCost is the fixed charge of one exact point.
const zerosExactPointCost = 100_000

func newZerosBudget() *zerosBudget { return &zerosBudget{cap: zerosWorkCap} }

func (b *zerosBudget) spend(u int64) bool {
	b.used += u
	return b.used <= b.cap
}

// zerosExactWords returns 1 + the binary exponent span of the nonzero
// entries of A, B, C and D in 64-bit words, the size of the exact
// rationals an exact point forms to first order.
func zerosExactWords(sys *System) int64 {
	lo, hi := math.MaxInt, math.MinInt
	for _, a := range []*mat.Dense{sys.A, sys.B, sys.C, sys.D} {
		r, c := a.Dims()
		for i := range r {
			for j := range c {
				if v := a.At(i, j); v != 0 {
					_, e := math.Frexp(v)
					lo, hi = min(lo, e), max(hi, e)
				}
			}
		}
	}
	if lo > hi {
		return 1
	}
	return 1 + int64(hi-lo+53)/64
}

// zerosCertProbes returns the fixed probe set of zerosCertify, the
// structured points first, and their count.
func zerosCertProbes(sys *System, zeros []complex128, discrete bool) ([]complex128, int) {
	var out []complex128
	add := func(ps ...complex128) {
		for _, p := range ps {
			if !slices.Contains(out, p) {
				out = append(out, p)
			}
		}
	}
	add(eigenAxisProbes(zeros, discrete)...)
	if discrete {
		add(1, -1)
	} else {
		add(0)
	}
	ns := len(out)
	n, _, _ := sys.Dims()
	add(zpkFillProbes(zerosCertBandAnchors(sys, discrete), zeros, discrete, 2*(n+1))...)
	return out, ns
}

// zerosCertBandAnchors returns the eigenvalues that anchor the fill band:
// one not resolved from the origin (discrete: from 1) by its rounding bound
// β is taken as the origin, so a nilpotent A's rounding-level eigenvalues
// do not shrink the band to where no point is determined (GVD7AS ruling 6).
func zerosCertBandAnchors(sys *System, discrete bool) []complex128 {
	ev, beta, ok := eigenRoundingBounds(sys.A)
	if !ok {
		evals, _ := sys.Poles()
		return evals
	}
	ref := complex(0, 0)
	if discrete {
		ref = 1
	}
	out := slices.Clone(ev)
	for i, l := range out {
		if !(beta[i] < cmplx.Abs(l-ref)) {
			out[i] = ref
		}
	}
	return out
}

// zerosCertPerm expands det a of the k×k a by permutations, with the
// permanent of |a| and of |a| + e.
func zerosCertPerm(a []complex128, e []float64, k int) (det complex128, perm, permE float64) {
	idx := make([]int, k)
	for i := range idx {
		idx[i] = i
	}
	var rec func(i int, sign float64)
	rec = func(i int, sign float64) {
		if i == k {
			v := complex(sign, 0)
			pa, pe := 1.0, 1.0
			for r := range k {
				x := a[r*k+idx[r]]
				v *= x
				pa *= cmplx.Abs(x)
				pe *= cmplx.Abs(x) + e[r*k+idx[r]]
			}
			det += v
			perm += pa
			permE += pe
			return
		}
		for j := i; j < k; j++ {
			idx[i], idx[j] = idx[j], idx[i]
			s := sign
			if i != j {
				s = -s
			}
			rec(i+1, s)
			idx[i], idx[j] = idx[j], idx[i]
		}
	}
	rec(0, 1)
	return det, perm, permE
}

func zerosCertMinor(a []complex128, e []float64, k, r, c int) ([]complex128, []float64) {
	ma, me := make([]complex128, 0, (k-1)*(k-1)), make([]float64, 0, (k-1)*(k-1))
	for i := range k {
		if i == r {
			continue
		}
		for j := range k {
			if j != c {
				ma = append(ma, a[i*k+j])
				me = append(me, e[i*k+j])
			}
		}
	}
	return ma, me
}

// zerosCertDet returns det a with a bound on its error for entry errors
// e: Σ|adj_ji|e_ij, the rest of the permanent expansion of |a| + e beyond
// first order, and γ_{2k+2}·perm(|a| + e) for the expansion's rounding.
func zerosCertDet(a []complex128, e []float64, k int) (complex128, float64) {
	det, perm, permE := zerosCertPerm(a, e, k)
	if k == 1 {
		return det, e[0] + gammaN(1)*cmplx.Abs(det)
	}
	zero := make([]float64, (k-1)*(k-1))
	first, pfirst := 0.0, 0.0
	for i := range k {
		for j := range k {
			ma, _ := zerosCertMinor(a, e, k, i, j)
			c, pa, _ := zerosCertPerm(ma, zero, k-1)
			first += cmplx.Abs(c) * e[i*k+j]
			pfirst += pa * e[i*k+j]
		}
	}
	return det, first + math.Max(0, permE-perm-pfirst) + gammaN(2*k+2)*permE
}

// zerosCertDetTerms fills d, dd, tLo, tHi of pt from the k×k response G
// with entry bounds e and terms bounds tl ≤ T ≤ th.
func zerosCertDetTerms(pt *zerosCertPoint, G []complex128, e, tl, th []float64, k int) {
	pt.d, pt.dd = zerosCertDet(G, e, k)
	if k == 1 {
		pt.tLo, pt.tHi = tl[0], th[0]
	} else {
		pt.tLo, pt.tHi = 0, 0
		for i := range k {
			for j := range k {
				ma, me := zerosCertMinor(G, e, k, i, j)
				c, dc := zerosCertDet(ma, me, k-1)
				pt.tLo += math.Max(0, cmplx.Abs(c)-dc) * tl[i*k+j]
				pt.tHi += (cmplx.Abs(c) + dc) * th[i*k+j]
			}
		}
	}
	pt.valid = pt.tLo > 0 && !math.IsInf(pt.tHi, 0) && !math.IsNaN(pt.dd) && !cmplx.IsNaN(pt.d)
}

// zerosCertDouble evaluates the points in double: Ĝ from boundedResponse
// (ver bounds), χ̂_A the pivot product of zpkPointFactors' GEPP factors
// with relative bound γ·Σ|W|(|L_f||U_f|)ᵀ/(1 − η) + γ_n. A point with
// γ·S_A above √ε·T_detG/|det Ĝ| is not determined in double.
func zerosCertDouble(sys *System, pts []complex128, det bool) []zerosCertPoint {
	n, k, p := sys.Dims()
	out := make([]zerosCertPoint, len(pts))
	resp := newBoundedResponse(sys, pts)
	defer resp.release()
	br := newBalancedRealization(sys, n, k, p)
	var pf zpkPointFactors
	G := make([]complex128, k*k)
	E, tl, th := make([]float64, k*k), make([]float64, k*k), make([]float64, k*k)
	tol := math.Sqrt(eps())
	for q, s := range pts {
		pt := &out[q]
		pt.s, pt.level = s, zerosLevelDouble
		if !resp.ok[q] || !pf.factor(&br, s) || !(pf.eta < 1) {
			continue
		}
		f := 1 / (1 - pf.eta)
		dT := 2*f*pf.eta + gammaN(2*n+6)
		o := q * k * k
		for i := range k {
			for j := range k {
				dv := sys.D.At(i, j)
				gv := resp.g[o+i*k+j] + complex(dv, 0)
				G[i*k+j] = gv
				E[i*k+j] = f*resp.ver[o+i*k+j] + gammaN(1)*cmplx.Abs(gv)
				t := resp.data[o+i*k+j]/eps() + math.Abs(dv)
				tl[i*k+j], th[i*k+j] = t*(1-dT), t*(1+dT)
			}
		}
		zerosCertEntries(pt, G, E, tl, th)
		if !det {
			continue
		}
		zerosCertDetTerms(pt, G, E, tl, th, k)
		pt.valid = pt.valid && pt.eok
		cm, ce := complex(1, 0), 0
		for i := range n {
			if pf.piv[i] != i {
				cm = -cm
			}
			cm *= pf.x[i*n+i]
			cm, ce = zerosCertNorm(cm, ce)
		}
		sa := 0.0
		for i := range n {
			for l := range n {
				sa += cabs(pf.w[i*n+l]) * pf.absLU[l*n+i]
			}
		}
		pt.chiM, pt.chiE = cm, ce
		pt.relChi = zpkGEPPConstant(n, eps())*sa*f + gammaN(n)
		if ad := cmplx.Abs(pt.d); pt.valid && ad > 0 && pt.relChi > 0.5*math.Max(tol, zerosRankFloorK*eps()*pt.tLo/ad) {
			pt.valid = false
		}
		if cm == 0 || math.IsNaN(pt.relChi) || math.IsInf(pt.relChi, 0) {
			pt.valid = false
		}
	}
	return out
}

func zerosCertNorm(v complex128, e int) (complex128, int) {
	mx := math.Max(math.Abs(real(v)), math.Abs(imag(v)))
	if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
		return v, e
	}
	_, k := math.Frexp(mx)
	return complex(math.Ldexp(real(v), -k), math.Ldexp(imag(v), -k)), e + k
}

// zerosCertDD evaluates s in complex double-double (zpkDDSolve and
// zpkDDBound, the entries as zpkDDEntryBounds forms them but kept in
// double-double), χ̂_A the dd pivot product with relative bound
// γ(u)·Σ_il |M̂⁻¹_il|(Pᵀ|L_f||U_f|)_li/(1 − η) + γ_{2n}(u) + ε.
func zerosCertDD(sys *System, br *balancedRealization, w *zpkDDWork, s complex128, det bool) zerosCertPoint {
	n, k, _ := sys.Dims()
	pt := zerosCertPoint{s: s, level: zerosLevelDD}
	zero, ok := zpkDDSolve(br, s, w)
	if zero {
		pt.pole = true
		return pt
	}
	if !ok || !zpkDDBound(n, k, w) {
		return pt
	}
	nInv, nLU := 0.0, 0.0
	for i := range n {
		ri, rl := 0.0, 0.0
		for j := range n {
			ri += w.inv[i*n+j].abs()
			rl += w.absLU[i*n+j]
		}
		nInv, nLU = math.Max(nInv, ri), math.Max(nLU, rl)
	}
	eta := zpkGEPPConstant(n, ddOpErr) * nInv * nLU
	G := make([]ddComplex, k*k)
	E, tl, th := make([]float64, k*k), make([]float64, k*k), make([]float64, k*k)
	au := make([]float64, n*k)
	for l := range n {
		for j := range k {
			v := math.Abs(br.b[l*k+j])
			for r := range n {
				v += math.Abs(br.a[l*n+r]) * w.x[r*k+j].abs()
			}
			au[l*k+j] = v
		}
	}
	dTd := 2*eta/(1-eta) + gammaN(2*n+6)
	cy := make([]float64, n)
	for i := range k {
		c := br.c[i*n : (i+1)*n]
		for l := range n {
			var v ddComplex
			for q := range n {
				if c[q] != 0 {
					v = v.add(ddComplex{re: dd{c[q], 0}}.mul(w.inv[q*n+l]))
				}
			}
			cy[l] = v.abs()
		}
		for j := range k {
			e, ok := zpkDDEntryBounds(c, w, n, k, j)
			if !ok {
				return pt
			}
			td := 0.0
			for l := range n {
				td += cy[l] * au[l*k+j]
			}
			var gs ddComplex
			cu, du := 0.0, 0.0
			for l := range n {
				gs = gs.add(ddComplex{re: dd{c[l], 0}}.mul(w.x[l*k+j]))
				cu += math.Abs(c[l]) * w.x[l*k+j].abs()
				du += math.Abs(c[l]) * w.du[l*k+j]
			}
			dv := sys.D.At(i, j)
			gv := gs.add(ddComplex{re: dd{dv, 0}})
			G[i*k+j] = gv
			E[i*k+j] = du + gammaDD(n+3)*cu + gammaDD(1)*gv.abs()
			tl[i*k+j] = (e.lo+math.Abs(dv))*(1-gammaN(1)) + td*(1-dTd)
			th[i*k+j] = (e.hi+math.Abs(dv))*(1+gammaN(1)) + td*(1+dTd)
		}
	}
	Gr, Er := make([]complex128, k*k), make([]float64, k*k)
	for i := range G {
		Gr[i] = G[i].complex128()
		Er[i] = E[i] + 2*eps()*cmplx.Abs(Gr[i]) + 2*math.SmallestNonzeroFloat64
	}
	zerosCertEntries(&pt, Gr, Er, tl, th)
	if !det {
		return pt
	}
	zerosCertDetTermsDD(&pt, G, E, tl, th, k)
	pt.valid = pt.valid && pt.eok
	rowAt := make([]int, n)
	for i := range rowAt {
		rowAt[i] = i
	}
	for kk := range n {
		rowAt[kk], rowAt[w.perm[kk]] = rowAt[w.perm[kk]], rowAt[kk]
	}
	prod := ddComplex{re: dd{1, 0}}
	ce := 0
	for kk := range n {
		if w.perm[kk] != kk {
			prod = ddComplex{prod.re.neg(), prod.im.neg()}
		}
		prod = prod.mul(w.a[kk*n+kk])
		mx := math.Max(math.Abs(prod.re.hi), math.Abs(prod.im.hi))
		if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
			pt.valid = false
			return pt
		}
		_, e := math.Frexp(mx)
		prod = ddComplex{prod.re.ldexp(-e), prod.im.ldexp(-e)}
		ce += e
	}
	sa := 0.0
	for i := range n {
		for r := range n {
			sa += w.inv[i*n+rowAt[r]].abs() * w.absLU[r*n+i]
		}
	}
	pt.chiM, pt.chiE = prod.complex128(), ce
	pt.relChi = zpkGEPPConstant(n, ddOpErr)*sa/(1-eta) + gammaDD(2*n) + eps()
	if math.IsNaN(pt.relChi) || math.IsInf(pt.relChi, 0) {
		pt.valid = false
	}
	return pt
}

// zerosCertExact evaluates s exactly in complex rationals (exactSolveAt),
// det G and adj G by permutations of the exact entries, rounded once.
func zerosCertExact(sys *System, s complex128, det bool, spend func(int64) bool) zerosCertPoint {
	n, k, _ := sys.Dims()
	pt := zerosCertPoint{s: s, level: zerosLevelExact}
	x, chi, zero, ok := exactSolveAtSpend(sys.A, sys.B, s, spend)
	if zero {
		pt.pole = true
		return pt
	}
	if !ok {
		return pt
	}
	cyT, _, _, ok := exactSolveAtSpend(mat.DenseCopyOf(sys.A.T()), mat.DenseCopyOf(sys.C.T()), s, spend)
	if !ok {
		return pt
	}
	G := make([]ratC, k*k)
	T := make([]float64, k*k)
	aRaw, bRaw, cRaw, dRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix(), sys.D.RawMatrix()
	xf, cyf := make([]float64, n*k), make([]float64, n*k)
	for i := range xf {
		xf[i], cyf[i] = cabs(x[i].complex128()), cabs(cyT[i].complex128())
	}
	xa := make([]float64, n*k)
	for l := range n {
		for j := range k {
			v := math.Abs(bRaw.Data[l*bRaw.Stride+j])
			for r := range n {
				v += math.Abs(aRaw.Data[l*aRaw.Stride+r]) * xf[r*k+j]
			}
			xa[l*k+j] = v
		}
	}
	allZero := true
	for i := range k {
		for j := range k {
			dv := dRaw.Data[i*dRaw.Stride+j]
			g := newRatC(dv, 0)
			t := math.Abs(dv)
			for l := range n {
				t += cyf[l*k+i] * xa[l*k+j]
				c := cRaw.Data[i*cRaw.Stride+l]
				if c == 0 || x[l*k+j].isZero() {
					continue
				}
				if !spend(ratSubCost(1, ratCWords(x[l*k+j]), ratCWords(g))) {
					return zerosCertPoint{s: s, level: zerosLevelExact}
				}
				v := x[l*k+j].mulReal(c)
				g.re.Add(g.re, v.re)
				g.im.Add(g.im, v.im)
				t += math.Abs(c) * xf[l*k+j]
			}
			G[i*k+j], T[i*k+j] = g, t
			allZero = allZero && g.isZero() && t == 0
		}
	}
	gt := gammaN(2*n + 8)
	Gr, Er, tl, th := make([]complex128, k*k), make([]float64, k*k), make([]float64, k*k), make([]float64, k*k)
	for i := range G {
		Gr[i] = G[i].complex128()
		Er[i] = 2*eps()*cmplx.Abs(Gr[i]) + 2*math.SmallestNonzeroFloat64
		if G[i].isZero() {
			Er[i] = 0
		}
		tl[i], th[i] = T[i]*(1-gt), T[i]*(1+gt)
	}
	zerosCertEntries(&pt, Gr, Er, tl, th)
	if allZero {
		pt.zero, pt.valid = true, true
		return pt
	}
	if !det {
		return pt
	}
	var wg int64
	for _, g := range G {
		wg = max(wg, ratCWords(g))
	}
	if cost := zerosRatDetCost(k, wg); k > 1 && !spend(cost+int64(k*k)*zerosRatDetCost(k-1, wg)) {
		return zerosCertPoint{s: s, level: zerosLevelExact}
	}
	dt := zerosCertRatDet(G, k)
	pt.d = dt.complex128()
	pt.dd = 2*eps()*cmplx.Abs(pt.d) + 2*math.SmallestNonzeroFloat64
	if k == 1 {
		pt.tLo, pt.tHi = T[0]*(1-gt), T[0]*(1+gt)
	} else {
		for i := range k {
			for j := range k {
				ma := make([]ratC, 0, (k-1)*(k-1))
				for r := range k {
					if r == i {
						continue
					}
					for c := range k {
						if c != j {
							ma = append(ma, G[r*k+c])
						}
					}
				}
				a := cmplx.Abs(zerosCertRatDet(ma, k-1).complex128())
				pt.tLo += a * (1 - 3*eps()) * T[i*k+j] * (1 - gt)
				pt.tHi += a * (1 + 3*eps()) * T[i*k+j] * (1 + gt)
			}
		}
	}
	l := bigLog2Abs(new(big.Float).SetRat(chi.re), new(big.Float).SetRat(chi.im))
	if math.IsInf(l, 0) || math.IsNaN(l) {
		return pt
	}
	e := int(math.Ceil(l))
	re, _ := new(big.Float).SetMantExp(new(big.Float).SetPrec(128).SetRat(chi.re), -e).Float64()
	im, _ := new(big.Float).SetMantExp(new(big.Float).SetPrec(128).SetRat(chi.im), -e).Float64()
	pt.chiM, pt.chiE, pt.relChi = complex(re, im), e, 2*eps()
	pt.valid = pt.tLo > 0 && !cmplx.IsNaN(pt.d) && !cmplx.IsInf(pt.d)
	return pt
}

// zerosRatDetCost is the work charged for zerosCertRatDet of a k×k
// matrix of entries up to w words: k! products of growing size.
func zerosRatDetCost(k int, w int64) int64 {
	f := int64(1)
	for i := 2; i <= k; i++ {
		f *= int64(i)
	}
	kk := int64(k)
	return f * (kk*kk*kk*w*w/6 + 40*kk)
}

func zerosCertRatDet(a []ratC, k int) ratC {
	if k == 0 {
		return newRatC(1, 0)
	}
	idx := make([]int, k)
	for i := range idx {
		idx[i] = i
	}
	det := newRatC(0, 0)
	var rec func(i int, neg bool)
	rec = func(i int, neg bool) {
		if i == k {
			v := newRatC(1, 0)
			for r := range k {
				v = v.mul(a[r*k+idx[r]])
			}
			if neg {
				det.re.Sub(det.re, v.re)
				det.im.Sub(det.im, v.im)
			} else {
				det.re.Add(det.re, v.re)
				det.im.Add(det.im, v.im)
			}
			return
		}
		for j := i; j < k; j++ {
			idx[i], idx[j] = idx[j], idx[i]
			rec(i+1, neg != (i != j))
			idx[i], idx[j] = idx[j], idx[i]
		}
	}
	rec(0, false)
	return det
}

// zerosCertScaleInfo carries the common exponent ec of the candidate gain
// c = c̃·2^ec; per point the inequality is judged on d, T and q =
// Π(s − ẑ)/χ_A·2^ec scaled by 2^−E, E the exponent of max(|d|, T_hi).
type zerosCertScaleInfo struct {
	ec int
	ok bool
}

func zerosCertScale(cp []zerosCertPoint, zeros []complex128, mode zerosCertMode) zerosCertScaleInfo {
	if mode != zerosModeCand {
		return zerosCertScaleInfo{ok: true}
	}
	best, bestRel := -1, math.Inf(1)
	for q, pt := range cp {
		if pt.state != zerosPtPending || !pt.valid || pt.pole || pt.zero || pt.d == 0 || pt.chiM == 0 {
			continue
		}
		if pm, _ := zerosCertProd(zeros, pt.s); pm == 0 {
			continue
		}
		if rel := pt.dd / cmplx.Abs(pt.d); rel < bestRel {
			best, bestRel = q, rel
		}
	}
	if best < 0 {
		return zerosCertScaleInfo{}
	}
	pt := cp[best]
	pm, pe := zerosCertProd(zeros, pt.s)
	_, ed := math.Frexp(cmplx.Abs(pt.d))
	_, er := math.Frexp(cmplx.Abs(pm / pt.chiM))
	return zerosCertScaleInfo{ec: ed - (pe - pt.chiE + er), ok: true}
}

// zerosCertProd returns Π(s − z) as mantissa·2^exp.
func zerosCertProd(zeros []complex128, s complex128) (complex128, int) {
	v, e := complex(1, 0), 0
	for _, z := range zeros {
		v, e = zerosCertNorm(v*(s-z), e)
		if v == 0 {
			return 0, 0
		}
	}
	return v, e
}

// zerosCertScaled returns the point's d, δd, T bounds and q with its
// relative bound, all scaled by 2^−E.
func zerosCertScaled(pt zerosCertPoint, zeros []complex128, sc zerosCertScaleInfo) (d complex128, dd, tLo, tHi float64, q complex128, dq float64) {
	_, E := math.Frexp(math.Max(cmplx.Abs(pt.d), pt.tHi))
	d = ldexpC(pt.d, -E)
	dd, tLo, tHi = math.Ldexp(pt.dd, -E), math.Ldexp(pt.tLo, -E), math.Ldexp(pt.tHi, -E)
	pm, pe := zerosCertProd(zeros, pt.s)
	q = ldexpC(pm/pt.chiM, pe-pt.chiE+sc.ec-E)
	dq = pt.relChi + gammaN(4*len(zeros)+4)
	return
}

func ldexpC(v complex128, e int) complex128 {
	return complex(math.Ldexp(real(v), e), math.Ldexp(imag(v), e))
}

// zerosCertEntriesVerdict reports whether every entry of the point's G
// verifies G_ij ≡ 0 within its terms, and whether some entry refutes it.
func zerosCertEntriesVerdict(pt zerosCertPoint) (ver, ref bool) {
	k := zerosRankFloorK * eps()
	ver = true
	for i, g := range pt.g {
		a, e := cmplx.Abs(g), pt.ge[i]
		if !(a+e <= k*pt.gl[i]) {
			ver = false
		}
		if a-e > k*pt.gh[i] {
			ref = true
		}
	}
	return ver, ref
}

// zerosCertZeroAt reports whether a candidate zero lies exactly at s.
func zerosCertZeroAt(zeros []complex128, s complex128) bool {
	return slices.Contains(zeros, s)
}

// zerosCertPointVerifies reports whether the point verifies the candidate
// c̃ (mode Det0: det G ≡ 0; Entry0: G ≡ 0) with its bounds. At an exactly
// vanishing response a nonzero candidate verifies only with a zero there
// or c = 0.
func zerosCertPointVerifies(pt zerosCertPoint, zeros []complex128, mode zerosCertMode, c float64, sc zerosCertScaleInfo) bool {
	tol := math.Sqrt(eps())
	if pt.zero {
		return mode != zerosModeCand || c == 0 || zerosCertZeroAt(zeros, pt.s)
	}
	switch mode {
	case zerosModeEntry0:
		v, _ := zerosCertEntriesVerdict(pt)
		return v
	case zerosModeDet0:
		ad := cmplx.Abs(pt.d)
		return ad+pt.dd <= zerosRankFloorK*eps()*pt.tLo
	}
	d, dd, tLo, _, q, dq := zerosCertScaled(pt, zeros, sc)
	z := complex(c, 0) * q
	az := cmplx.Abs(z)
	ad := cmplx.Abs(d)
	return cmplx.Abs(d-z)+dd+az*dq <= math.Max(tol*math.Max(ad, az), zerosRankFloorK*eps()*tLo)
}

// zerosCertIntervals is a union of disjoint closed intervals of c̃.
type zerosCertIntervals []zerosCertIv

type zerosCertIv struct{ lo, hi float64 }

func (s zerosCertIntervals) nonempty() bool { return len(s) > 0 }

// pick returns a point inside the widest interval: its midpoint, or a
// finite point of a half-line.
func (s zerosCertIntervals) pick() (float64, bool) {
	best, bw := -1, -1.0
	for i, v := range s {
		if w := v.hi - v.lo; w > bw || math.IsInf(w, 1) {
			best, bw = i, w
		}
	}
	if best < 0 {
		return 0, false
	}
	v := s[best]
	switch {
	case math.IsInf(v.lo, -1) && math.IsInf(v.hi, 1):
		return 0, true
	case math.IsInf(v.lo, -1):
		return v.hi - math.Max(1, math.Abs(v.hi)), true
	case math.IsInf(v.hi, 1):
		return v.lo + math.Max(1, math.Abs(v.lo)), true
	}
	return v.lo + (v.hi-v.lo)/2, true
}

func (s zerosCertIntervals) intersect(t zerosCertIntervals) zerosCertIntervals {
	var out zerosCertIntervals
	for _, a := range s {
		for _, b := range t {
			if l, h := math.Max(a.lo, b.lo), math.Min(a.hi, b.hi); l <= h {
				out = append(out, zerosCertIv{l, h})
			}
		}
	}
	return out.merge()
}

func (s zerosCertIntervals) merge() zerosCertIntervals {
	slices.SortFunc(s, func(a, b zerosCertIv) int { return cmp.Compare(a.lo, b.lo) })
	out := s[:0]
	for _, v := range s {
		if n := len(out); n > 0 && v.lo <= out[n-1].hi {
			out[n-1].hi = math.Max(out[n-1].hi, v.hi)
			continue
		}
		out = append(out, v)
	}
	return out
}

var zerosCertAll = zerosCertIntervals{{math.Inf(-1), math.Inf(1)}}

// zerosCertBall returns the real c with |d − c·q| ≤ α + β·|c|; loose
// selects a superset where the quadratic degenerates, else a subset.
func zerosCertBall(d, q complex128, alpha, beta float64, loose bool) zerosCertIntervals {
	a2 := real(q)*real(q) + imag(q)*imag(q)
	w := cmplx.Conj(d) * q
	bh, im := real(w), imag(w)
	dd := real(d)*real(d) + imag(d)*imag(d)
	var out zerosCertIntervals
	if a2 == 0 {
		ad := math.Sqrt(dd)
		switch {
		case beta == 0:
			if ad <= alpha {
				out = append(out, zerosCertIv{math.Inf(-1), math.Inf(1)})
			}
		case beta > 0:
			r := math.Max(0, (ad-alpha)/beta)
			out = append(out, zerosCertIv{math.Inf(-1), -r}, zerosCertIv{r, math.Inf(1)})
		default:
			if r := (alpha - ad) / -beta; r >= 0 {
				out = append(out, zerosCertIv{-r, r})
			}
		}
		return out.merge()
	}
	for _, sg := range [2]float64{1, -1} {
		half := zerosCertIv{0, math.Inf(1)}
		if sg < 0 {
			half = zerosCertIv{math.Inf(-1), 0}
		}
		lin := half
		switch {
		case beta > 0 && alpha < 0:
			r := -alpha / beta
			if sg > 0 {
				lin.lo = math.Max(lin.lo, r)
			} else {
				lin.hi = math.Min(lin.hi, -r)
			}
		case beta < 0:
			if alpha < 0 {
				continue
			}
			r := alpha / -beta
			if sg > 0 {
				lin.hi = math.Min(lin.hi, r)
			} else {
				lin.lo = math.Max(lin.lo, -r)
			}
		case beta == 0 && alpha < 0:
			continue
		}
		if lin.lo > lin.hi {
			continue
		}
		A := a2 - beta*beta
		B := bh + sg*alpha*beta
		if !(A > 0) {
			if loose {
				out = append(out, lin)
			}
			continue
		}
		disc := -im*im + a2*alpha*alpha + beta*beta*dd + 2*sg*bh*alpha*beta
		if disc < 0 || math.IsNaN(disc) {
			continue
		}
		sq := math.Sqrt(disc)
		lo, hi := (B-sq)/A, (B+sq)/A
		lo, hi = math.Max(lo, lin.lo), math.Min(hi, lin.hi)
		if lo <= hi {
			out = append(out, zerosCertIv{lo, hi})
		}
	}
	return out.merge()
}

// zerosCertSets returns the intersections, over the usable unconceded
// points, of the c̃ each point's verification bounds admit (a subset of
// the exact set) and of those its refutation bounds admit (a superset).
// For the modes without c the sets are all of ℝ or empty by the point's
// own verdict. At an exactly vanishing response the exact set is {0}
// unless a candidate zero lies there.
func zerosCertSets(cp []zerosCertPoint, zeros []complex128, mode zerosCertMode, sc zerosCertScaleInfo) (ver, ref zerosCertIntervals) {
	tol := math.Sqrt(eps())
	kf := zerosRankFloorK * eps()
	ver, ref = zerosCertAll, zerosCertAll
	if mode == zerosModeCand && !sc.ok {
		ver = nil
	}
	origin := zerosCertIntervals{{0, 0}}
	for _, pt := range cp {
		if pt.state != zerosPtPending || pt.pole {
			continue
		}
		if pt.zero {
			if mode == zerosModeCand && !zerosCertZeroAt(zeros, pt.s) {
				ver, ref = ver.intersect(origin), ref.intersect(origin)
			}
			continue
		}
		switch mode {
		case zerosModeEntry0:
			if !pt.eok {
				continue
			}
			v, r := zerosCertEntriesVerdict(pt)
			if !v {
				ver = nil
			}
			if r {
				ref = nil
			}
			continue
		case zerosModeDet0:
			if !pt.valid {
				continue
			}
			ad := cmplx.Abs(pt.d)
			if !(ad+pt.dd <= kf*pt.tLo) {
				ver = nil
			}
			if ad-pt.dd > kf*pt.tHi {
				ref = nil
			}
			continue
		}
		if !pt.valid || !sc.ok {
			continue
		}
		d, dd, tLo, tHi, q, dq := zerosCertScaled(pt, zeros, sc)
		ad, aq := cmplx.Abs(d), cmplx.Abs(q)
		v := zerosCertBall(d, q, math.Max(tol*ad, kf*tLo)-dd, -aq*dq, false)
		v = append(v, zerosCertBall(d, q, -dd, aq*(tol-dq), false)...)
		ver = ver.intersect(v.merge())
		r := zerosCertBall(d, q, math.Max(tol*(ad+dd), kf*tHi)+dd, aq*dq, true)
		r = append(r, zerosCertBall(d, q, dd, aq*(tol*(1+dq)+dq), true)...)
		ref = ref.intersect(r.merge())
	}
	return ver, ref
}

// zerosCertPermDD is zerosCertPerm in complex double-double.
func zerosCertPermDD(a []ddComplex, e []float64, k int) (det ddComplex, perm, permE float64) {
	idx := make([]int, k)
	for i := range idx {
		idx[i] = i
	}
	var rec func(i int, neg bool)
	rec = func(i int, neg bool) {
		if i == k {
			v := ddComplex{re: dd{1, 0}}
			pa, pe := 1.0, 1.0
			for r := range k {
				x := a[r*k+idx[r]]
				v = v.mul(x)
				pa *= x.abs()
				pe *= x.abs() + e[r*k+idx[r]]
			}
			if neg {
				det = det.sub(v)
			} else {
				det = det.add(v)
			}
			perm += pa
			permE += pe
			return
		}
		for j := i; j < k; j++ {
			idx[i], idx[j] = idx[j], idx[i]
			rec(i+1, neg != (i != j))
			idx[i], idx[j] = idx[j], idx[i]
		}
	}
	rec(0, false)
	return det, perm, permE
}

// zerosCertDetDD is zerosCertDet in complex double-double, its rounding
// term at the double-double unit; the determinant is returned rounded to
// double with that rounding added to the bound.
func zerosCertDetDD(a []ddComplex, e []float64, k int) (complex128, float64) {
	det, perm, permE := zerosCertPermDD(a, e, k)
	d := det.complex128()
	round := 2*eps()*cmplx.Abs(d) + 2*math.SmallestNonzeroFloat64
	if k == 1 {
		return d, e[0] + round
	}
	zero := make([]float64, (k-1)*(k-1))
	first, pfirst := 0.0, 0.0
	for i := range k {
		for j := range k {
			ma, _ := zerosCertMinorDD(a, e, k, i, j)
			c, pa, _ := zerosCertPermDD(ma, zero, k-1)
			first += c.abs() * e[i*k+j]
			pfirst += pa * e[i*k+j]
		}
	}
	return d, first + math.Max(0, permE-perm-pfirst) + gammaDD(2*k+2)*permE + round
}

func zerosCertMinorDD(a []ddComplex, e []float64, k, r, c int) ([]ddComplex, []float64) {
	ma, me := make([]ddComplex, 0, (k-1)*(k-1)), make([]float64, 0, (k-1)*(k-1))
	for i := range k {
		if i == r {
			continue
		}
		for j := range k {
			if j != c {
				ma = append(ma, a[i*k+j])
				me = append(me, e[i*k+j])
			}
		}
	}
	return ma, me
}

// zerosCertDetTermsDD is zerosCertDetTerms for double-double entries.
func zerosCertDetTermsDD(pt *zerosCertPoint, G []ddComplex, e, tl, th []float64, k int) {
	pt.d, pt.dd = zerosCertDetDD(G, e, k)
	if k == 1 {
		pt.tLo, pt.tHi = tl[0], th[0]
	} else {
		pt.tLo, pt.tHi = 0, 0
		for i := range k {
			for j := range k {
				ma, me := zerosCertMinorDD(G, e, k, i, j)
				c, dc := zerosCertDetDD(ma, me, k-1)
				pt.tLo += math.Max(0, cmplx.Abs(c)-dc) * tl[i*k+j]
				pt.tHi += (cmplx.Abs(c) + dc) * th[i*k+j]
			}
		}
	}
	pt.valid = pt.tLo > 0 && !math.IsInf(pt.tHi, 0) && !math.IsNaN(pt.dd) && !cmplx.IsNaN(pt.d)
}

// zerosVerdict judges the zeros and normal rank of the explicit square
// sys within terms: at full rank m the candidate det form (zerosCertify);
// rank 0 entrywise (every G_ij ≡ 0 within its terms), its zeros not
// judged (a departure, follow-up 6UHLZO) except where G ≡ 0 exactly,
// whose zero count must match the exact one (zerosExactZeroCount);
// otherwise (i) det G ≡ 0 (rank < m, skipped above zerosCertMaxRank,
// where (ii) and (iii) suffice), (ii) the rows of [C D] and columns of
// [B; D] of the balanced model dependent on rank of them to rounding
// (dependentRowBasis, rank ≤ k), and (iii) the candidate on the k×k
// sub-data of those rows and columns (rank ≥ k and the zeros). It is
// refuted when any part is, verified when every part is, else open,
// including a dynamic dependence (ii) does not show.
func zerosVerdict(sys *System, zeros []complex128, rank int) zerosCertVerdict {
	return zerosVerdictB(sys, zeros, rank, newZerosBudget())
}

func zerosVerdictB(sys *System, zeros []complex128, rank int, b *zerosBudget) zerosCertVerdict {
	n, m, p := sys.Dims()
	if m != p || n == 0 {
		return zerosCertOpen
	}
	switch {
	case rank == m:
		return zerosCertifyB(sys, zeros, zerosModeCand, b).verdict
	case rank == 0:
		r := zerosCertifyB(sys, zeros, zerosModeEntry0, b)
		if r.verdict != zerosCertVerified {
			return r.verdict
		}
		exactZero := true
		for _, pt := range r.pts {
			if pt.state == zerosPtVerified && !pt.zero {
				exactZero = false
			}
		}
		if exactZero {
			if cnt, ok := zerosExactZeroCount(sys, b); !ok || cnt != len(zeros) {
				return zerosCertOpen
			}
		}
		return zerosCertVerified
	case rank > zerosCertMaxRank:
		return zerosCertOpen
	}
	v0 := zerosCertVerified
	if m <= zerosCertMaxRank {
		v0 = zerosCertifyB(sys, zeros, zerosModeDet0, b).verdict
		if v0 == zerosCertRefuted {
			return v0
		}
	}
	sub, ok := zerosCertSub(sys, rank)
	if !ok {
		return zerosCertOpen
	}
	vs := zerosCertifyB(sub, zeros, zerosModeCand, b).verdict
	switch {
	case vs == zerosCertRefuted:
		return zerosCertRefuted
	case vs == zerosCertVerified && v0 == zerosCertVerified:
		return zerosCertVerified
	}
	return zerosCertOpen
}

// zerosExactZeroCount returns the number of finite zeros, with
// multiplicity, of the system pencil P(s) = [sI − A, −B; −C, −D] of normal
// rank n (G ≡ 0): the degree of the gcd of det(Q₁P(s)Q₂) over three fixed
// integer compressions Q₁ (n×(n+p)), Q₂ ((n+m)×n), exact on the stored
// data scaled to integers (integerPencil, exactDetPoly); compressions
// whose det vanishes identically are skipped. false when the pencil is
// outside exactPencilWithinBudget or b runs out.
func zerosExactZeroCount(sys *System, b *zerosBudget) (int, bool) {
	n, m, p := sys.Dims()
	R, Cc := n+p, n+m
	L, M := make([]float64, R*Cc), make([]float64, R*Cc)
	for i := range n {
		L[i*Cc+i] = 1
		for j := range n {
			M[i*Cc+j] = sys.A.At(i, j)
		}
		for j := range m {
			M[i*Cc+n+j] = sys.B.At(i, j)
		}
	}
	for i := range p {
		for j := range n {
			M[(n+i)*Cc+j] = sys.C.At(i, j)
		}
		for j := range m {
			M[(n+i)*Cc+n+j] = sys.D.At(i, j)
		}
	}
	li, mi, ok := integerPencil(L, M)
	if !ok {
		return 0, false
	}
	w := zerosExactWords(sys)
	seed := int64(12345)
	next := func() int64 {
		seed = (seed*1103515245 + 12345) % 2147483648
		return seed%7 - 3
	}
	var g []*big.Rat
	for range 3 {
		if !b.spend(4*int64(n)*int64(n)*int64(n)*int64(n)*int64(n)*w*w + 50_000) {
			return 0, false
		}
		q1, q2 := make([]int64, n*R), make([]int64, Cc*n)
		for i := range q1 {
			q1[i] = next()
		}
		for i := range q2 {
			q2[i] = next()
		}
		lc, mc := zerosCompress(li, q1, q2, n, R, Cc), zerosCompress(mi, q1, q2, n, R, Cc)
		if !exactPencilWithinBudget(lc, mc, n) {
			return 0, false
		}
		d := exactDetPoly(lc, mc, n, n)
		for len(d) > 0 && d[len(d)-1].Sign() == 0 {
			d = d[:len(d)-1]
		}
		if len(d) == 0 {
			continue
		}
		if g == nil {
			g = d
		} else {
			g = zerosRatPolyGcd(g, d)
		}
	}
	if g == nil {
		return 0, false
	}
	return len(g) - 1, true
}

// zerosCompress returns Q₁·X·Q₂ for the R×C integer X.
func zerosCompress(x []*big.Int, q1, q2 []int64, n, R, C int) []*big.Int {
	t := make([]*big.Int, n*C)
	var u big.Int
	for i := range n {
		for j := range C {
			v := new(big.Int)
			for r := range R {
				if q1[i*R+r] != 0 && x[r*C+j].Sign() != 0 {
					v.Add(v, u.Mul(big.NewInt(q1[i*R+r]), x[r*C+j]))
				}
			}
			t[i*C+j] = v
		}
	}
	out := make([]*big.Int, n*n)
	for i := range n {
		for j := range n {
			v := new(big.Int)
			for c := range C {
				if q2[c*n+j] != 0 && t[i*C+c].Sign() != 0 {
					v.Add(v, u.Mul(t[i*C+c], big.NewInt(q2[c*n+j])))
				}
			}
			out[i*n+j] = v
		}
	}
	return out
}

// zerosRatPolyGcd returns the monic gcd of the ascending a and b, both of
// nonzero leading coefficient.
func zerosRatPolyGcd(a, b []*big.Rat) []*big.Rat {
	monic := func(p []*big.Rat) []*big.Rat {
		lead := new(big.Rat).Set(p[len(p)-1])
		out := make([]*big.Rat, len(p))
		for i, c := range p {
			out[i] = new(big.Rat).Quo(c, lead)
		}
		return out
	}
	a = monic(a)
	for {
		for len(b) > 0 && b[len(b)-1].Sign() == 0 {
			b = b[:len(b)-1]
		}
		if len(b) == 0 {
			return a
		}
		b = monic(b)
		r := make([]*big.Rat, len(a))
		for i, c := range a {
			r[i] = new(big.Rat).Set(c)
		}
		var t big.Rat
		for len(r) >= len(b) {
			f := new(big.Rat).Set(r[len(r)-1])
			off := len(r) - len(b)
			for i, c := range b {
				r[off+i].Sub(r[off+i], t.Mul(f, c))
			}
			r = r[:len(r)-1]
		}
		a, b = b, r
	}
}

// zerosCertSub returns the k×k sub-data of sys on the rows of [C D] and the
// columns of [B; D] of its balanced model that span the others to rounding
// (dependentRowBasis, as squareDownZeros); false when there are none.
func zerosCertSub(sys *System, k int) (*System, bool) {
	n, m, p := sys.Dims()
	zs := zerosPrescale(sys)
	rows := dependentRowBasis(zs.C, zs.D, n, m, p, k)
	cols := dependentRowBasis(mat.DenseCopyOf(zs.B.T()), mat.DenseCopyOf(zs.D.T()), n, p, m, k)
	if rows == nil || cols == nil {
		return nil, false
	}
	B, C, D := mat.NewDense(n, k, nil), mat.NewDense(k, n, nil), mat.NewDense(k, k, nil)
	for a, i := range rows {
		C.SetRow(a, sys.C.RawRowView(i))
		for b, j := range cols {
			D.Set(a, b, sys.D.At(i, j))
		}
	}
	for b, j := range cols {
		for i := range n {
			B.Set(i, b, sys.B.At(i, j))
		}
	}
	return &System{A: sys.A, B: B, C: C, D: D, Dt: sys.Dt}, true
}

// zerosTier2 proposes zeros from the exact det of the system pencil of the
// stored data and returns the first the det form certifies: on sys at
// full rank m (also when (i) refutes a rank below it), else on the k×k
// sub-data of zerosCertSub. The pencil P(s) = [A − sI B; C D] is taken
// exactly within exactPencilWithinBudget (pencilDetBounded); the
// candidates are the roots (aberthRoots) of det P truncated to degree d,
// d climbing from its degree to the data (dataDegree, the
// coefficient-rounding convention) to its exact degree. All of it is
// charged to the certification's work budget.
func zerosTier2(sys *System, rank int, rankRefuted bool) ([]complex128, int, bool) {
	return zerosTier2B(sys, rank, rankRefuted, newZerosBudget())
}

func zerosTier2B(sys *System, rank int, rankRefuted bool, b *zerosBudget) ([]complex128, int, bool) {
	n, m, p := sys.Dims()
	if m != p || n == 0 {
		return nil, 0, false
	}
	model, k := sys, m
	if rank < m && !rankRefuted {
		if rank == 0 {
			return nil, 0, false
		}
		sub, ok := zerosCertSub(sys, rank)
		if !ok {
			return nil, 0, false
		}
		model, k = sub, rank
	}
	if k > zerosCertMaxRank {
		return nil, 0, false
	}
	N := n + k
	L, M := make([]float64, N*N), make([]float64, N*N)
	for i := range n {
		L[i*N+i] = 1
		for j := range n {
			M[i*N+j] = model.A.At(i, j)
		}
		for j := range k {
			M[i*N+n+j] = model.B.At(i, j)
		}
	}
	for i := range k {
		for j := range n {
			M[(n+i)*N+j] = model.C.At(i, j)
		}
		for j := range k {
			M[(n+i)*N+n+j] = model.D.At(i, j)
		}
	}
	Nw := int64(N) * zerosExactWords(model)
	N4 := int64(N) * int64(N) * int64(N) * int64(N)
	if !b.spend(N4 * Nw * Nw) {
		return nil, 0, false
	}
	bp, ok := pencilDetBounded(L, M, N)
	if !ok {
		return nil, 0, false
	}
	exact := -1
	for d := len(bp.coef) - 1; d >= 0; d-- {
		if bp.coef[d].Sign() != 0 {
			exact = d
			break
		}
	}
	if exact < 0 {
		return nil, 0, false
	}
	for d := max(bp.dataDegree(), 0); d <= exact; d++ {
		if bp.coef[d].Sign() == 0 {
			continue
		}
		if !b.spend(int64(d) * int64(d) * 7200) {
			return nil, 0, false
		}
		zeros, ok := zerosPolyRoots(bp.coef[:d+1])
		if !ok {
			continue
		}
		if zerosCertifyB(model, zeros, zerosModeCand, b).verdict == zerosCertVerified {
			sortZeros(zeros)
			return zeros, k, true
		}
	}
	return nil, 0, false
}

// zerosPolyRoots returns the roots of the ascending rational polynomial p
// of nonzero leading coefficient, those at the origin exactly.
func zerosPolyRoots(p []*big.Rat) ([]complex128, bool) {
	lo := 0
	for lo < len(p)-1 && p[lo].Sign() == 0 {
		lo++
	}
	var out []complex128
	if lo < len(p)-1 {
		roots, ok := aberthRoots(p[lo:])
		if !ok {
			return nil, false
		}
		for _, r := range roots {
			z := r.complex()
			if cmplx.IsNaN(z) || cmplx.IsInf(z) {
				return nil, false
			}
			if math.Abs(imag(z)) < math.Abs(real(z))*eps()*100 {
				z = complex(real(z), 0)
			}
			out = append(out, z)
		}
	}
	for range lo {
		out = append(out, 0)
	}
	return out, true
}

// certifyZeros returns res, the zeros and normal rank mimoZeros computed for
// the explicit sys, when zerosVerdict verifies them, else Tier 2's
// (zerosTier2) when it certifies, else res with ErrAccuracyUnverified.
// Non-square models keep res unchecked.
func certifyZeros(sys *System, res *ZerosResult) (*ZerosResult, error) {
	out, err := certifyZerosRun(sys, res)
	if certifyZerosHook != nil {
		certifyZerosHook(sys, res, out, err)
	}
	return out, err
}

// certifyZerosHook is a test-only seam, nil outside tests: it sees every
// certification with the computed and the returned result.
var certifyZerosHook func(sys *System, base, out *ZerosResult, err error)

func certifyZerosRun(sys *System, res *ZerosResult) (*ZerosResult, error) {
	n, m, p := sys.Dims()
	if m != p || n == 0 {
		return res, nil
	}
	b := newZerosBudget()
	v := zerosVerdictB(sys, res.Zeros, res.Rank, b)
	if v == zerosCertVerified {
		return res, nil
	}
	rankRefuted := false
	if res.Rank < m && m <= zerosCertMaxRank {
		rankRefuted = zerosCertifyB(sys, nil, zerosModeDet0, b).verdict == zerosCertRefuted
	}
	if z, k, ok := zerosTier2B(sys, res.Rank, rankRefuted, b); ok {
		return &ZerosResult{Zeros: z, Rank: k}, nil
	}
	return res, fmt.Errorf("zeros not certified within the data's rounding terms: %w", ErrAccuracyUnverified)
}
