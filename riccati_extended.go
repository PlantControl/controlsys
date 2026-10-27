package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"

	"plantcontrol.org/v1/gonum/lapack"
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

// Limits of the extended-precision explicit Dare of nearSingularDare:
// doublings (ρ(A − BK) up to 1 − 2⁻⁴⁰ converges) and the precision range.
const (
	extendedDoublings = 48
	extendedPrecMin   = 192
	extendedPrecMax   = 320
)

// nearSingularDare solves a Dare with a near-singular E that Dare's
// float64 pencil left with no stabilizing solution (err) on its explicit
// model Ae = E⁻¹A, Be = E⁻¹B, formed with every later step in big.Float
// arithmetic, bit-identical on every platform. EᵀXE = Y is the solution
// of that model's Dare, and Y and K are well posed while X ~ 1/σ_min(E)²
// is not: at cond(E)·eps ≈ 0.1 the float64 QZ subspace left Y off by 6.5%
// and the float64 K from the exact Y was off by 360% (O2YDYA).
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
// some j. Eig is the float64 QZ of (A − BK, E), MATLAB pole(dss). Any
// failure keeps err, so no value it returned before changes.
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
	prec := uint(min(extendedPrecMax, max(extendedPrecMin, 4*math.Log2(cond)+64)))
	d := extendedDare{A: A, B: B, Q: Q, R: R, S: opts.S, E: opts.E}
	sol, ok := d.solve(prec)
	if !ok && prec < extendedPrecMax {
		prec = min(extendedPrecMax, prec+64)
		sol, ok = d.solve(prec)
	}
	if !ok {
		return nil, false
	}
	K := sol.k.dense()
	if !d.stable(sol, K) {
		return nil, false
	}
	z, ok := xSolve(sol.e.t(), sol.y)
	if !ok {
		return nil, false
	}
	xt, ok := xSolve(sol.e.t(), z.t())
	if !ok {
		return nil, false
	}
	x := xt.t().symmetrized()
	X, Y := x.dense(), sol.y.dense()
	symmetrize(X.RawMatrix().Data, n, n)
	symmetrize(Y.RawMatrix().Data, n, n)

	ykOK, xOK := true, d.xBound(sol, x)
	for seed := range uint64(2) {
		dy, dk, dx, ok := d.perturbedMove(sol, x, prec, seed)
		ykOK = ykOK && ok && dy <= newtonTol && dk <= newtonTol
		xOK = xOK && ok && dx <= newtonTol
	}
	res := &RiccatiResult{X: X, K: K, Eig: d.eig(K), Rcnd: placedRcnd(X.RawMatrix().Data, n), Accuracy: 1, xe: Y}
	if ykOK {
		res.xeCertified = true
		if xOK {
			res.Accuracy = 0
		}
	}
	return res, true
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

func (d extendedDare) solve(prec uint) (extendedSol, bool) {
	sol, ok := d.model(prec, nil)
	if !ok {
		return sol, false
	}
	n := sol.ae.r
	a, h := sol.ae, sol.q
	if sol.s.r > 0 {
		rs, ok := xSolve(sol.r, sol.s.t())
		if !ok {
			return sol, false
		}
		a, h = xAdd(a, xMul(sol.be, rs), -1), xAdd(h, xMul(sol.s, rs), -1)
	}
	rb, ok := xSolve(sol.r, sol.be.t())
	if !ok {
		return sol, false
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
			return sol, false
		}
		hn := xAdd(h, xMul(xMul(a.t(), h), wa), 1)
		g = xAdd(g, xMul(xMul(a, wg), a.t()), 1).symmetrized()
		a = xMul(a, wa)
		dh, hm := xAdd(hn, h, -1).maxAbs(), hn.maxAbs()
		h = hn.symmetrized()
		if !(hm < math.Ldexp(1, int(prec/2))) {
			return sol, false
		}
		if dh <= tol*hm {
			conv = true
			break
		}
	}
	if !conv {
		return sol, false
	}
	sol.y = h
	if sol.k, ok = d.gain(sol, h); !ok {
		return sol, false
	}
	res := d.residual(sol, h, sol.k)
	return sol, res <= tol*(h.maxAbs()+sol.q.maxAbs())
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

// stable reports ‖(Ae − BeK)^(2^j)‖_∞ ≤ ½ for some j ≤ extendedDoublings,
// a proof that ρ(Ae − BeK) < 1.
func (d extendedDare) stable(sol extendedSol, K *mat.Dense) bool {
	m := d.closedLoop(sol, K)
	limit := math.Ldexp(1, int(m.prec/4))
	for range extendedDoublings {
		v := m.infNorm()
		if v <= 0.5 {
			return true
		}
		if !(v < limit) {
			return false
		}
		m = xMul(m, m)
	}
	return false
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

// perturbedMove returns the relative moves of Y, K and X under a
// componentwise eps perturbation (seeded signs) of the data: one Hewer
// step from Y, the Stein equation solved by Smith doubling.
func (d extendedDare) perturbedMove(sol extendedSol, x xMat, prec uint, seed uint64) (dy, dk, dx float64, ok bool) {
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
		return 0, 0, 0, false
	}
	k, ok := d.gain(p, sol.y)
	if !ok {
		return 0, 0, 0, false
	}
	acl := xAdd(p.ae, xMul(p.be, k), -1)
	rk := xMul(p.r, k)
	w := xAdd(p.q, xMul(k.t(), rk), 1)
	if p.s.r > 0 {
		sk := xMul(p.s, k)
		w = xAdd(xAdd(w, sk, -1), sk.t(), -1)
	}
	y, ok := smithStein(acl, w.symmetrized())
	if !ok {
		return 0, 0, 0, false
	}
	k1, ok := d.gain(p, y)
	if !ok {
		return 0, 0, 0, false
	}
	z, ok1 := xSolve(p.e.t(), y)
	if !ok1 {
		return 0, 0, 0, false
	}
	xt, ok2 := xSolve(p.e.t(), z.t())
	if !ok2 {
		return 0, 0, 0, false
	}
	rel := func(a, b xMat) float64 {
		v := xAdd(a, b, -1).dense()
		w := b.dense()
		return max(mat.Norm(v, 1)/mat.Norm(w, 1), mat.Norm(v, 2)/mat.Norm(w, 2))
	}
	return rel(y, sol.y), rel(k1, sol.k), rel(xt.t(), x), true
}

// smithStein solves Y = FᵀYF + W for ρ(F) < 1 by Smith doubling.
func smithStein(f, w xMat) (xMat, bool) {
	y := w
	tol := math.Ldexp(1, -int(w.prec/2))
	for range extendedDoublings {
		inc := xMul(xMul(f.t(), y), f)
		y = xAdd(y, inc, 1)
		if !(inc.maxAbs() > tol*y.maxAbs()) {
			return y.symmetrized(), true
		}
		f = xMul(f, f)
		if !(f.maxAbs() < math.Ldexp(1, int(w.prec/4))) {
			return y, false
		}
	}
	return y, false
}

// eig is the float64 QZ of (A − BK, E).
func (d extendedDare) eig(K *mat.Dense) []complex128 {
	n, _ := d.A.Dims()
	var acl mat.Dense
	acl.Mul(d.B, K)
	acl.Sub(d.A, &acl)
	a := make([]float64, n*n)
	e := make([]float64, n*n)
	aRaw, eRaw := acl.RawMatrix(), d.E.RawMatrix()
	copyStrided(a, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(e, n, eRaw.Data, eRaw.Stride, n, n)
	wr, wi, beta := make([]float64, n), make([]float64, n), make([]float64, n)
	work := make([]float64, max(1, 8*n))
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n, a, n, e, n, wr, wi, beta, nil, 1, nil, 1, work, len(work)) {
		return nil
	}
	eig := make([]complex128, n)
	for k := range n {
		eig[k] = complex(wr[k]/beta[k], wi[k]/beta[k])
		if beta[k] == 0 {
			eig[k] = cmplx.Inf()
		}
	}
	return eig
}
