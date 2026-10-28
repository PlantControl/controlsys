package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/rand/v2"

	"plantcontrol.org/v1/gonum/mat"
)

// xMat is a dense row-major matrix of big.Float at one precision.
type xMat struct {
	r, c int
	prec uint
	d    []big.Float
}

func newXMat(r, c int, prec uint) xMat {
	m := xMat{r: r, c: c, prec: prec, d: make([]big.Float, r*c)}
	for i := range m.d {
		m.d[i].SetPrec(prec)
	}
	return m
}

func xFrom(a *mat.Dense, prec uint) xMat {
	if a == nil {
		return xMat{prec: prec}
	}
	raw := a.RawMatrix()
	m := newXMat(raw.Rows, raw.Cols, prec)
	for i := range raw.Rows {
		for j := range raw.Cols {
			m.d[i*m.c+j].SetFloat64(raw.Data[i*raw.Stride+j])
		}
	}
	return m
}

func xEye(n int, prec uint) xMat {
	m := newXMat(n, n, prec)
	for i := range n {
		m.d[i*n+i].SetInt64(1)
	}
	return m
}

func (m xMat) at(i, j int) *big.Float { return &m.d[i*m.c+j] }

func (m xMat) t() xMat {
	o := newXMat(m.c, m.r, m.prec)
	for i := range m.r {
		for j := range m.c {
			o.at(j, i).Set(m.at(i, j))
		}
	}
	return o
}

func xMul(a, b xMat) xMat {
	o := newXMat(a.r, b.c, a.prec)
	p := new(big.Float).SetPrec(a.prec)
	for i := range a.r {
		for k := range a.c {
			aik := a.at(i, k)
			if aik.Sign() == 0 {
				continue
			}
			for j := range b.c {
				o.at(i, j).Add(o.at(i, j), p.Mul(aik, b.at(k, j)))
			}
		}
	}
	return o
}

// xAdd returns a + sign·b, sign ±1.
func xAdd(a, b xMat, sign int) xMat {
	o := newXMat(a.r, a.c, a.prec)
	for i := range o.d {
		if sign < 0 {
			o.d[i].Sub(&a.d[i], &b.d[i])
		} else {
			o.d[i].Add(&a.d[i], &b.d[i])
		}
	}
	return o
}

func (m xMat) clone() xMat {
	o := newXMat(m.r, m.c, m.prec)
	for i := range o.d {
		o.d[i].Set(&m.d[i])
	}
	return o
}

// xSolve solves a·x = b by Gaussian elimination with partial pivoting; ok
// is false on a zero pivot.
func xSolve(a, b xMat) (xMat, bool) {
	n := a.r
	a, b = a.clone(), b.clone()
	f, p := new(big.Float).SetPrec(a.prec), new(big.Float).SetPrec(a.prec)
	u, v := new(big.Float).SetPrec(a.prec), new(big.Float).SetPrec(a.prec)
	for k := range n {
		piv := k
		v.Abs(a.at(k, k))
		for i := k + 1; i < n; i++ {
			if u.Abs(a.at(i, k)).Cmp(v) > 0 {
				piv = i
				v.Set(u)
			}
		}
		if a.at(piv, k).Sign() == 0 {
			return xMat{}, false
		}
		if piv != k {
			for j := range n {
				a.d[k*n+j], a.d[piv*n+j] = a.d[piv*n+j], a.d[k*n+j]
			}
			for j := range b.c {
				b.d[k*b.c+j], b.d[piv*b.c+j] = b.d[piv*b.c+j], b.d[k*b.c+j]
			}
		}
		for i := k + 1; i < n; i++ {
			if a.at(i, k).Sign() == 0 {
				continue
			}
			f.Quo(a.at(i, k), a.at(k, k))
			for j := k + 1; j < n; j++ {
				a.at(i, j).Sub(a.at(i, j), p.Mul(f, a.at(k, j)))
			}
			for j := range b.c {
				b.at(i, j).Sub(b.at(i, j), p.Mul(f, b.at(k, j)))
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		for j := range b.c {
			for k := i + 1; k < n; k++ {
				b.at(i, j).Sub(b.at(i, j), p.Mul(a.at(i, k), b.at(k, j)))
			}
			b.at(i, j).Quo(b.at(i, j), a.at(i, i))
		}
	}
	return b, true
}

func (m xMat) dense() *mat.Dense {
	d := make([]float64, m.r*m.c)
	for i := range d {
		d[i], _ = m.d[i].Float64()
	}
	return mat.NewDense(m.r, m.c, d)
}

// maxAbs is max |m_ij| rounded to float64 (+Inf beyond its range).
func (m xMat) maxAbs() float64 {
	v := 0.0
	for i := range m.d {
		f, _ := m.d[i].Float64()
		v = max(v, math.Abs(f))
	}
	return v
}

func (m xMat) symmetrized() xMat {
	o := m.clone()
	h := new(big.Float).SetPrec(m.prec)
	for i := range m.r {
		for j := i + 1; j < m.c; j++ {
			h.Add(m.at(i, j), m.at(j, i))
			h.SetMantExp(h, -1)
			o.at(i, j).Set(h)
			o.at(j, i).Set(h)
		}
	}
	return o
}

// Precision range of the extended-precision explicit Dare of
// nearSingularDare.
const (
	extendedPrecMin = 192
	extendedPrecMax = 320
)

// Iteration caps of the extended-precision solves, variables only so that
// tests can hit them: SDA doublings of nearSingularDare (ρ(A − BK) up to
// 1 − 2⁻⁴⁰ converges; no retry at higher precision once hit), squarings of
// stableLoop and Smith doublings of smithStein. A solve that hits its cap
// fails, and its caller keeps the float64 result.
var (
	extendedDoublings = 48
	stableSquarings   = 48
	steinDoublings    = 48
)

// nearSingularDare solves a Dare with a near-singular E that Dare's
// float64 pencil left with no stabilizing solution (err) on its explicit
// model Ae = E⁻¹A, Be = E⁻¹B, formed with every later step in big.Float
// arithmetic, bit-identical on every platform. EᵀXE = Y is the solution
// of that model's Dare, and Y and K are well posed while X ~ 1/σ_min(E)²
// is not: at cond(E)·eps ≈ 0.1 the float64 QZ subspace left Y off by 6.5%
// and the float64 K from the exact Y was off by 360%.
//
// It runs only when E (and Kalman's plant E) is nonsingular to both
// MATLAB rank and LU condition (nearSingularEGate rejects anything
// else) with cond(E)·eps above newtonTol, the regime where the float64
// fold E⁻¹A alone leaves Accuracy 0's tolerance; a well-conditioned E
// keeps the pencil's error. The precision is 4·log₂cond(E) + 64 bits in
// [192, 320]. Y is the limit of SDA doubling, accepted only when its
// residual in the given equation (with S) is within 2^(−prec/2) of
// ‖Y‖ + ‖Q‖, once more at 64 bits more on a miss; K is formed from it in
// the same precision and rounded, and the closed loop of the rounded K
// is shown stable in extended precision: ‖(Ae − BeK)^(2^j)‖_∞ ≤ ½ for
// some j. Eig is the float64 eigenvalues of that extended-precision
// Ae − BeK rounded once, the exact eig(A − BK, E): a float64 A − BK
// errs by eps·|B||K|, which 1/σ_min(E) carries to O(1) in the QZ of
// (A − BK, E) (|λ| 1.27 for an exact 0.58). Any failure, an iteration
// cap hit or a failed perturbed solve included, keeps err, so no value it
// returned before changes.
//
// Y and K carry Accuracy 0 (Dlqr, Lqg: xeCertified) when they move by at
// most newtonTol under each of two componentwise eps perturbations of A,
// B, E, Q, R and S (one Hewer step from Y in extended precision); X =
// E⁻ᵀYE⁻¹, formed in extended precision, also needs its first-order
// componentwise bound ‖|E⁻ᵀ||δE|ᵀ|X| + |X||δE||E⁻¹|‖/‖X‖, δE = eps·|E|,
// and its move under the same perturbations within newtonTol, else Dare
// returns it at Accuracy 1.
func nearSingularDare(A, B, Q, R *mat.Dense, opts *RiccatiOpts, err error) (*RiccatiResult, bool) {
	if opts == nil || isIdentityDescriptor(opts.E) ||
		!errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
		return nil, false
	}
	n, m := B.Dims()
	if n == 0 || m == 0 {
		return nil, false
	}
	cond := 0.0
	for _, E := range []*mat.Dense{opts.E, opts.plantE} {
		if E == nil {
			continue
		}
		var lu mat.LU
		lu.Factorize(E)
		if luNearSingular(&lu) || eRankDeficient(E, &lu) {
			return nil, false
		}
		cond = max(cond, lu.Cond())
	}
	if !(cond*eps() > newtonTol) {
		return nil, false
	}
	prec := extendedPrec(cond)
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: opts.S, E: opts.E}
	sol, capped, ok := d.solve(prec)
	if !ok && !capped && prec < extendedPrecMax {
		prec = min(extendedPrecMax, prec+64)
		sol, _, ok = d.solve(prec)
	}
	if !ok {
		return nil, false
	}
	K := sol.k.dense()
	acl := d.closedLoop(sol, K)
	if !stableLoop(acl) {
		return nil, false
	}
	eig := loopEig(acl)
	x, ok := sol.xOf(sol.y)
	if !ok {
		return nil, false
	}
	X, Y := x.dense(), sol.y.dense()
	symmetrize(X.RawMatrix().Data, n, n)
	symmetrize(Y.RawMatrix().Data, n, n)

	kw, ok := d.noiseGain(sol, sol.y)
	if !ok {
		return nil, false
	}
	ykOK, xOK, kwOK := true, d.xBound(sol, x), true
	for seed := range uint64(2) {
		mv, ok := d.perturbedMove(sol, x, kw, prec, seed)
		if !ok {
			return nil, false
		}
		ykOK = ykOK && mv.y <= newtonTol && mv.k <= newtonTol
		xOK = xOK && mv.x <= newtonTol
		kwOK = kwOK && mv.kw <= newtonTol
	}
	res := &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: placedRcnd(X.RawMatrix().Data, n), Accuracy: 1, xe: Y, kw: kw.dense(), kwCertified: kwOK}
	if ykOK {
		res.xeCertified = true
		if xOK {
			res.Accuracy = 0
		}
	}
	return res, true
}

// extendedPrec is the precision of nearSingularDare for cond(E).
func extendedPrec(cond float64) uint {
	return uint(min(extendedPrecMax, max(extendedPrecMin, 4*math.Log2(cond)+64)))
}

// descriptorNoiseGain returns Lqg's noise gain (R + BᵀXB)⁻¹BᵀX of the
// descriptor Dare (A, B, Q, R, S, E) whose stabilizing gain is K, formed
// in extended precision (nearSingularDare's model, precision from
// cond(E)) from Y = EᵀXE of two Hewer steps from K: where BᵀXB of the
// float64 X cancels, its Kw was off by 3e-7 at cond(E) 8e9. certified
// reports that Kw moves by at most newtonTol under two componentwise eps
// perturbations of the data; ok is false unless the closed loop of K is
// shown stable, K moves by at most newtonTol in the first step (K is the
// stabilizing gain to tolerance) and every step, the perturbed ones and
// their iteration caps included, succeeds.
func descriptorNoiseGain(A, B, Q, R, S, E, K *mat.Dense) (kw *mat.Dense, certified, ok bool) {
	var lu mat.LU
	lu.Factorize(E)
	if luNearSingular(&lu) {
		return nil, false, false
	}
	prec := extendedPrec(lu.Cond())
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: S, E: E}
	sol, ok := d.model(prec, nil)
	if !ok || !stableLoop(d.closedLoop(sol, K)) {
		return nil, false, false
	}
	k := xFrom(K, prec)
	for i := range 2 {
		y, ok := sol.hewer(k)
		if !ok {
			return nil, false, false
		}
		k1, ok := d.gain(sol, y)
		if !ok {
			return nil, false, false
		}
		if i == 0 && !(xRel(k1, k) <= newtonTol) {
			return nil, false, false
		}
		sol.y, sol.k, k = y, k1, k1
	}
	x, ok := sol.xOf(sol.y)
	if !ok {
		return nil, false, false
	}
	g, ok := d.noiseGain(sol, sol.y)
	if !ok {
		return nil, false, false
	}
	certified = true
	for seed := range uint64(2) {
		mv, ok := d.perturbedMove(sol, x, g, prec, seed)
		if !ok {
			return nil, false, false
		}
		certified = certified && mv.kw <= newtonTol
	}
	return g.dense(), certified, true
}

type extendedDare struct {
	A, B, Q, R, S, E *mat.Dense
}

// extendedSol is the converged explicit model and its solution: e, ae, be
// and the data, y = EᵀXE and the unrounded gain k.
type extendedSol struct {
	e, ae, be, q, r, s, y, k xMat
}

func (d extendedDare) model(prec uint, perturb func(xMat) xMat) (sol extendedSol, ok bool) {
	id := func(m xMat) xMat { return m }
	if perturb == nil {
		perturb = id
	}
	sol.e = perturb(xFrom(d.E, prec))
	if sol.ae, ok = xSolve(sol.e, perturb(xFrom(d.A, prec))); !ok {
		return sol, false
	}
	if sol.be, ok = xSolve(sol.e, perturb(xFrom(d.B, prec))); !ok {
		return sol, false
	}
	sol.q, sol.r = perturb(xFrom(d.Q, prec)).symmetrized(), perturb(xFrom(d.R, prec)).symmetrized()
	if d.S != nil {
		sol.s = perturb(xFrom(d.S, prec))
	}
	return sol, true
}

// solve runs at most extendedDoublings SDA doublings; capped reports that
// it did not converge within them, so a retry at higher precision cannot.
func (d extendedDare) solve(prec uint) (sol extendedSol, capped, ok bool) {
	sol, ok = d.model(prec, nil)
	if !ok {
		return sol, false, false
	}
	n := sol.ae.r
	a, h := sol.ae, sol.q
	if sol.s.r > 0 {
		rs, ok := xSolve(sol.r, sol.s.t())
		if !ok {
			return sol, false, false
		}
		a, h = xAdd(a, xMul(sol.be, rs), -1), xAdd(h, xMul(sol.s, rs), -1)
	}
	rb, ok := xSolve(sol.r, sol.be.t())
	if !ok {
		return sol, false, false
	}
	g := xMul(sol.be, rb)
	eye := xEye(n, prec)
	tol := math.Ldexp(1, -int(prec/2))
	conv := false
	for range extendedDoublings {
		w := xAdd(eye, xMul(g, h), 1)
		wa, ok1 := xSolve(w, a)
		wg, ok2 := xSolve(w, g)
		if !ok1 || !ok2 {
			return sol, false, false
		}
		hn := xAdd(h, xMul(xMul(a.t(), h), wa), 1)
		g = xAdd(g, xMul(xMul(a, wg), a.t()), 1).symmetrized()
		a = xMul(a, wa).flushTiny()
		dh, hm := xAdd(hn, h, -1).maxAbs(), hn.maxAbs()
		h = hn.symmetrized()
		if !(hm < math.Ldexp(1, int(prec/2))) {
			return sol, false, false
		}
		if dh <= tol*hm {
			conv = true
			break
		}
	}
	if !conv {
		return sol, true, false
	}
	sol.y = h
	if sol.k, ok = d.gain(sol, h); !ok {
		return sol, false, false
	}
	res := d.residual(sol, h, sol.k)
	return sol, false, res <= tol*(h.maxAbs()+sol.q.maxAbs())
}

// gain is (R + BeᵀYBe)⁻¹(BeᵀYAe + Sᵀ).
func (d extendedDare) gain(sol extendedSol, y xMat) (xMat, bool) {
	btx := xMul(sol.be.t(), y)
	rh := xMul(btx, sol.ae)
	if sol.s.r > 0 {
		rh = xAdd(rh, sol.s.t(), 1)
	}
	return xSolve(xAdd(sol.r, xMul(btx, sol.be), 1), rh)
}

// noiseGain is (R + BᵀXB)⁻¹BᵀX = (R + BeᵀYBe)⁻¹BeᵀYE⁻¹, the gain of a
// process noise w in Ex[k+1] = Ax[k] + Bu[k] + w[k]: formed from X in
// float64 BᵀXB cancels, ‖X‖ ~ 1/σ_min(E)².
func (d extendedDare) noiseGain(sol extendedSol, y xMat) (xMat, bool) {
	ei, ok := xSolve(sol.e, xEye(sol.e.r, sol.e.prec))
	if !ok {
		return xMat{}, false
	}
	bty := xMul(sol.be.t(), y)
	return xSolve(xAdd(sol.r, xMul(bty, sol.be), 1), xMul(bty, ei))
}

// residual is max |AeᵀYAe − Y − (AeᵀYBe + S)K + Q|.
func (d extendedDare) residual(sol extendedSol, y, k xMat) float64 {
	aty := xMul(sol.ae.t(), y)
	c := xMul(aty, sol.be)
	if sol.s.r > 0 {
		c = xAdd(c, sol.s, 1)
	}
	r := xAdd(xAdd(xMul(aty, sol.ae), y, -1), xMul(c, k), -1)
	return xAdd(r, sol.q, 1).maxAbs()
}

func (d extendedDare) closedLoop(sol extendedSol, K *mat.Dense) xMat {
	return xAdd(sol.ae, xMul(sol.be, xFrom(K, sol.ae.prec)), -1)
}

// stableLoop reports ‖m^(2^j)‖_∞ ≤ ½ for some j ≤ stableSquarings, a
// proof that ρ(m) < 1.
func stableLoop(m xMat) bool {
	limit := math.Ldexp(1, int(m.prec/4))
	for range stableSquarings {
		v := m.infNorm()
		if v <= 0.5 {
			return true
		}
		if !(v < limit) {
			return false
		}
		m = xMul(m, m).flushTiny()
	}
	return false
}

// flushTiny zeroes, in place, the entries below 2^(−2·prec) times the
// largest: a big.Float sum shifts its mantissa by the full exponent gap,
// and repeated squaring with modes near 1 next to fast ones grows that gap
// to ~2³⁰ bits (GB-scale allocation), far below the precision carried.
func (m xMat) flushTiny() xMat {
	top, seen := 0, false
	for i := range m.d {
		if m.d[i].Sign() != 0 {
			e := m.d[i].MantExp(nil)
			if !seen || e > top {
				top, seen = e, true
			}
		}
	}
	lim := top - 2*int(m.prec)
	for i := range m.d {
		if m.d[i].Sign() != 0 && m.d[i].MantExp(nil) < lim {
			m.d[i].SetInt64(0)
		}
	}
	return m
}

func (m xMat) infNorm() float64 {
	v := 0.0
	for i := range m.r {
		s := new(big.Float).SetPrec(m.prec)
		for j := range m.c {
			s.Add(s, new(big.Float).Abs(m.at(i, j)))
		}
		f, _ := s.Float64()
		v = max(v, f)
	}
	return v
}

// xBound reports the first-order componentwise bound of X's move under
// δE = eps·|E| within newtonTol.
func (d extendedDare) xBound(sol extendedSol, x xMat) bool {
	n := sol.e.r
	ei, ok := xSolve(sol.e, xEye(n, sol.e.prec))
	if !ok {
		return false
	}
	absEi, absX := absDense(ei.dense()), absDense(x.dense())
	absE := absDense(d.E)
	var t, b1, b2 mat.Dense
	t.Mul(absEi.T(), absE.T())
	b1.Mul(&t, absX)
	t.Mul(absX, absE)
	b2.Mul(&t, absEi)
	b1.Add(&b1, &b2)
	xn := denseNorm(absX)
	return xn > 0 && denseNorm(&b1)*eps() <= newtonTol*xn
}

// extendedMove is the relative move of Y, K, X and the noise gain.
type extendedMove struct{ y, k, x, kw float64 }

// perturbedMove returns the relative moves of Y, K, X and the noise gain
// kw under a componentwise eps perturbation (seeded signs) of the data:
// one Hewer step from Y, the Stein equation solved by Smith doubling.
func (d extendedDare) perturbedMove(sol extendedSol, x, kw xMat, prec uint, seed uint64) (mv extendedMove, ok bool) {
	rng := rand.New(rand.NewPCG(seed+1, 0x0a2d))
	ep := new(big.Float).SetPrec(prec).SetFloat64(eps())
	perturb := func(m xMat) xMat {
		o := m.clone()
		f := new(big.Float).SetPrec(prec)
		for i := range o.d {
			f.Mul(&o.d[i], ep)
			if rng.IntN(2) == 0 {
				o.d[i].Sub(&o.d[i], f)
			} else {
				o.d[i].Add(&o.d[i], f)
			}
		}
		return o
	}
	p, ok := d.model(prec, perturb)
	if !ok {
		return mv, false
	}
	k, ok := d.gain(p, sol.y)
	if !ok {
		return mv, false
	}
	y, ok := p.hewer(k)
	if !ok {
		return mv, false
	}
	k1, ok := d.gain(p, y)
	if !ok {
		return mv, false
	}
	xp, ok := p.xOf(y)
	if !ok {
		return mv, false
	}
	kwp, ok := d.noiseGain(p, y)
	if !ok {
		return mv, false
	}
	return extendedMove{y: xRel(y, sol.y), k: xRel(k1, sol.k), x: xRel(xp, x), kw: xRel(kwp, kw)}, true
}

// hewer is the cost Y of the gain k: Y = (Ae − Bek)ᵀY(Ae − Bek) + Q +
// kᵀRk − Sk − kᵀSᵀ, by Smith doubling.
func (sol extendedSol) hewer(k xMat) (xMat, bool) {
	acl := xAdd(sol.ae, xMul(sol.be, k), -1)
	w := xAdd(sol.q, xMul(k.t(), xMul(sol.r, k)), 1)
	if sol.s.r > 0 {
		sk := xMul(sol.s, k)
		w = xAdd(xAdd(w, sk, -1), sk.t(), -1)
	}
	return smithStein(acl, w.symmetrized())
}

// xOf is X = E⁻ᵀYE⁻¹, symmetrized.
func (sol extendedSol) xOf(y xMat) (xMat, bool) {
	z, ok := xSolve(sol.e.t(), y)
	if !ok {
		return xMat{}, false
	}
	xt, ok := xSolve(sol.e.t(), z.t())
	if !ok {
		return xMat{}, false
	}
	return xt.t().symmetrized(), true
}

// xRel is the relative difference of a from b in the 1- and 2-norm,
// whichever is larger, rounded to float64.
func xRel(a, b xMat) float64 {
	v, w := xAdd(a, b, -1).dense(), b.dense()
	return max(mat.Norm(v, 1)/mat.Norm(w, 1), mat.Norm(v, 2)/mat.Norm(w, 2))
}

// smithStein solves Y = FᵀYF + W for ρ(F) < 1 by Smith doubling.
func smithStein(f, w xMat) (xMat, bool) {
	y := w
	tol := math.Ldexp(1, -int(w.prec/2))
	for range steinDoublings {
		inc := xMul(xMul(f.t(), y), f)
		y = xAdd(y, inc, 1)
		if !(inc.maxAbs() > tol*y.maxAbs()) {
			return y.symmetrized(), true
		}
		f = xMul(f, f).flushTiny()
		if !(f.maxAbs() < math.Ldexp(1, int(w.prec/4))) {
			return y, false
		}
	}
	return y, false
}

// loopEig is the float64 eigenvalues of the closed loop Ae − BeK formed
// in extended precision and rounded once; nil if they do not converge.
func loopEig(acl xMat) []complex128 {
	var ed mat.Eigen
	if !ed.Factorize(acl.dense(), mat.EigenNone) {
		return nil
	}
	return ed.Values(nil)
}
