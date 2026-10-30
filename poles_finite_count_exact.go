package controlsys

import (
	"fmt"
	"math"
	"math/big"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// exactCountPrec is the working precision in bits of the exact staircase,
// that of the 320-bit pencil oracle (pencilOraclePrec,
// descriptor_poles_fold_loss_test.go:29).
const exactCountPrec = 320

// exactJacobiStop and exactJacobiSweeps are the stop rule of the one-sided
// Jacobi SVD in the working arithmetic, not data tolerances: rows p, q count
// as orthogonal once γ² ≤ 2⁻⁶⁰⁰·α·β (a cosine of at most 2⁻³⁰⁰), within at
// most 100 cyclic sweeps. Both are those of the exact staircase of commit
// 76a7a1e.
var exactJacobiStop = new(big.Float).SetMantExp(big.NewFloat(1), -600)

const exactJacobiSweeps = 100

// exactFiniteCount is the finite-pole bracket of a pencil (A, E) from the
// exact staircase: lo ≤ hi. reason, when not empty, names the first event
// that kept a level from being certified (an unconverged Jacobi sweep, a
// nullity rise d_j > d_{j−1}, or an A row block with σ_min ≤ τ_A).
type exactFiniteCount struct {
	lo, hi int
	reason string
}

// exactStairLevel records one staircase level: the k×k E block's singular
// values sv, the zone radius rho (τ_E at level 0), the counts in the zones
// s ≤ τ_E, τ_E < s ≤ ρ, s > ρ, the rank r kept and, for a deflating level,
// the smallest singular value sigmaA of the A row block, the E gap
// σ_r − σ_{r+1} and the rotation bound sinZ (nil where not reached).
type exactStairLevel struct {
	k, r                     int
	zero, undecided, nonzero int
	rho                      *big.Float
	sv                       []*big.Float
	sigmaA, gap, sinZ        *big.Float
}

// exactStair is the workspace of the exact staircase of an n×n pencil:
// every big.Float is allocated once at exactCountPrec and reused by both
// runs. trace, when set, receives each level record once the level is done;
// the record aliases workspace buffers and is valid only during the call.
type exactStair struct {
	n                        int
	a0, e0, a, e             [][]*big.Float
	q, comp                  [][]*big.Float
	basis, zc, rows          [][]*big.Float
	norms, sv, sa            []*big.Float
	cand, best, row          []*big.Float
	idx                      []int
	s                        [14]*big.Float
	one, two                 *big.Float
	normA, normE, tauA, tauE *big.Float
	rho, sumSin, thQ, sinZ   *big.Float
	gap, den                 *big.Float
	trace                    func(lo bool, lv *exactStairLevel)
}

func newExactStair(A, E *mat.Dense) *exactStair {
	n, _ := A.Dims()
	w := &exactStair{n: n}
	pool := make([]big.Float, 6*n*n+6*n+len(w.s)+12)
	next := func() *big.Float {
		x := &pool[0]
		pool = pool[1:]
		return x.SetPrec(exactCountPrec)
	}
	mat := func(rows int) [][]*big.Float {
		m := make([][]*big.Float, rows)
		for i := range m {
			m[i] = make([]*big.Float, n)
			for j := range m[i] {
				m[i][j] = next()
			}
		}
		return m
	}
	vec := func() []*big.Float {
		v := make([]*big.Float, n)
		for i := range v {
			v[i] = next()
		}
		return v
	}
	w.a0, w.e0, w.a, w.e, w.q, w.comp = mat(n), mat(n), mat(n), mat(n), mat(n), mat(n)
	w.norms, w.sv, w.sa, w.cand, w.best, w.row = vec(), vec(), vec(), vec(), vec(), vec()
	for i := range w.s {
		w.s[i] = next()
	}
	w.one, w.two = next().SetInt64(1), next().SetInt64(2)
	w.normA, w.normE, w.tauA, w.tauE = next(), next(), next(), next()
	w.rho, w.sumSin, w.thQ, w.sinZ = next(), next(), next(), next()
	w.gap, w.den = next(), next()
	w.basis = make([][]*big.Float, 0, n)
	w.zc = make([][]*big.Float, 0, n)
	w.rows = make([][]*big.Float, n)
	w.idx = make([]int, n)
	ar, er := A.RawMatrix(), E.RawMatrix()
	for i := range n {
		for j := range n {
			w.a0[i][j].SetFloat64(ar.Data[i*ar.Stride+j])
			w.e0[i][j].SetFloat64(er.Data[i*er.Stride+j])
		}
	}
	return w
}

// dot sets z to Σ x_i·y_i, accumulated in index order.
func (w *exactStair) dot(z *big.Float, x, y []*big.Float) *big.Float {
	t := w.s[0]
	z.SetInt64(0)
	for i := range x {
		z.Add(z, t.Mul(x[i], y[i]))
	}
	return z
}

// rotate applies the plane rotation (c, s) to rows p, q of m, first k
// columns: (a, b) ← (c·a − s·b, s·a + c·b).
func (w *exactStair) rotate(m [][]*big.Float, k, p, q int, c, s *big.Float) {
	t1, t2 := w.s[1], w.s[2]
	for j := range k {
		a, b := m[p][j], m[q][j]
		na, nb := w.cand[j], w.best[j]
		na.Sub(t1.Mul(c, a), t2.Mul(s, b))
		nb.Add(t1.Mul(s, a), t2.Mul(c, b))
		m[p][j], w.cand[j] = na, a
		m[q][j], w.best[j] = nb, b
	}
}

// jacobiRows orthogonalizes the first k columns of the rows of x by
// one-sided Jacobi in fixed cyclic order, applying every rotation to the
// same rows of also (when not nil), then sorts the rows by norm, descending
// (stable). It stores the row norms, the singular values of x[:, :k], in
// norms[:len(x)] and reports whether the sweeps converged.
func (w *exactStair) jacobiRows(x [][]*big.Float, k int, also [][]*big.Float, norms []*big.Float) bool {
	m := len(x)
	alpha, beta, gamma := w.s[3], w.s[4], w.s[5]
	g2, ab, zeta, den, t, u, c, sn := w.s[6], w.s[7], w.s[8], w.s[9], w.s[10], w.s[11], w.s[12], w.s[13]
	converged := false
	for sweep := 0; sweep < exactJacobiSweeps && !converged; sweep++ {
		converged = true
		for p := range m {
			for q := p + 1; q < m; q++ {
				w.dot(alpha, x[p][:k], x[p][:k])
				w.dot(beta, x[q][:k], x[q][:k])
				w.dot(gamma, x[p][:k], x[q][:k])
				if gamma.Sign() == 0 {
					continue
				}
				g2.Mul(gamma, gamma)
				ab.Mul(alpha, beta)
				if g2.Cmp(ab.Mul(ab, exactJacobiStop)) <= 0 {
					continue
				}
				converged = false
				zeta.Sub(beta, alpha)
				zeta.Quo(zeta, t.Mul(w.two, gamma))
				u.Add(w.one, t.Mul(zeta, zeta))
				den.Sqrt(u)
				den.Add(den, t.Abs(zeta))
				t.Quo(w.one, den)
				if zeta.Sign() < 0 {
					t.Neg(t)
				}
				u.Add(w.one, u.Mul(t, t))
				c.Quo(w.one, den.Sqrt(u))
				sn.Mul(c, t)
				w.rotate(x, k, p, q, c, sn)
				if also != nil {
					w.rotate(also, k, p, q, c, sn)
				}
			}
		}
	}
	for i := range m {
		w.norms[i].Sqrt(w.dot(w.s[3], x[i][:k], x[i][:k]))
	}
	idx := w.idx[:m]
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(i, j int) int { return w.norms[j].Cmp(w.norms[i]) })
	w.permute(x, idx)
	if also != nil {
		w.permute(also, idx)
	}
	for i, j := range idx {
		norms[i].Set(w.norms[j])
	}
	return converged
}

func (w *exactStair) permute(rows [][]*big.Float, idx []int) {
	tmp := w.rows[:len(idx)]
	copy(tmp, rows)
	for i, j := range idx {
		rows[i] = tmp[j]
	}
}

// sigma1 sets z to σ₁ of the n×n matrix m, using w.a as scratch.
func (w *exactStair) sigma1(z *big.Float, m [][]*big.Float) bool {
	for i := range w.n {
		for j := range w.n {
			w.a[i][j].Set(m[i][j])
		}
	}
	ok := w.jacobiRows(w.a, w.n, nil, w.sv)
	z.Set(w.sv[0])
	return ok
}

// setTau sets z to n·eps(σ) with eps the float64 spacing at the correctly
// rounded σ: the MATLAB rank floor of descriptorRank with full set
// (eigdecomp.go:1238-1242).
func (w *exactStair) setTau(z, sigma *big.Float) {
	s, _ := sigma.Float64()
	z.SetFloat64(float64(w.n) * (math.Nextafter(s, math.Inf(1)) - s))
}

// minOne sets x to min(x, 1).
func (w *exactStair) minOne(x *big.Float) {
	if x.Cmp(w.one) > 0 {
		x.Set(w.one)
	}
}

// rowComplement sets w.zc to an orthonormal basis of the orthogonal
// complement of the d orthonormal rows of w.q in R^k, followed by those
// rows: greedy, each step taking the unit vector e_j with the largest
// residual (lowest j on ties), projected out twice.
func (w *exactStair) rowComplement(d, k int) {
	w.basis = append(w.basis[:0], w.q[:d]...)
	w.zc = w.zc[:0]
	dd, t, bestN := w.s[3], w.s[4], w.s[5]
	project := func(v []*big.Float) {
		for range 2 {
			for _, b := range w.basis {
				w.dot(dd, b[:k], v)
				for i := range k {
					v[i].Sub(v[i], t.Mul(dd, b[i]))
				}
			}
		}
	}
	for len(w.basis) < k {
		cand, best := w.cand[:k], w.best[:k]
		found := false
		for j := range k {
			for i := range k {
				cand[i].SetInt64(0)
			}
			cand[j].SetInt64(1)
			project(cand)
			if nn := w.dot(w.s[6], cand, cand); !found || nn.Cmp(bestN) > 0 {
				bestN.Set(nn)
				cand, best = best, cand
				found = true
			}
		}
		project(best)
		out := w.comp[len(w.zc)][:k]
		nrm := w.s[7]
		nrm.Sqrt(w.dot(w.s[8], best, best))
		for i := range k {
			out[i].Quo(best[i], nrm)
		}
		w.basis = append(w.basis, out)
		w.zc = append(w.zc, out)
	}
	w.zc = append(w.zc, w.q[:d]...)
}

// mulZ replaces the first k columns of rows 0..k-1 of m by m·Zᵀ with Z the
// rows of w.zc.
func (w *exactStair) mulZ(m [][]*big.Float, k int) {
	row := w.row[:k]
	for i := range k {
		for j := range k {
			w.dot(row[j], m[i][:k], w.zc[j][:k])
		}
		for j := range k {
			m[i][j], row[j] = row[j], m[i][j]
		}
	}
}

// staircase runs the exact staircase on copies of (A, E). With lo false a
// level zeroes the singular values ≤ τ_E; with lo true those ≤ ρ_j. It
// returns the finite count (a bound in the direction of lo when a level
// cannot be certified) and the reason, if any.
func (w *exactStair) staircase(lo bool) (int, string) {
	n := w.n
	for i := range n {
		for j := range n {
			w.a[i][j].Set(w.a0[i][j])
			w.e[i][j].Set(w.e0[i][j])
		}
	}
	a, e := w.a, w.e
	reason := ""
	w.sumSin.SetInt64(0)
	prevD := n + 1
	k := n
	var lv exactStairLevel
	emit := func() {
		if w.trace != nil {
			w.trace(lo, &lv)
		}
	}
	for level := 0; ; level++ {
		if k == 0 {
			return 0, reason
		}
		sv := w.sv[:k]
		if !w.jacobiRows(e[:k], k, a[:k], sv) {
			reason = fmt.Sprintf("Jacobi unconverged at level %d", level)
		}
		rho := w.rho
		rho.Mul(w.two, rho.Mul(w.normE, w.sumSin))
		rho.Add(rho, w.tauE)
		thr := w.tauE
		if lo {
			thr = rho
		}
		lv = exactStairLevel{k: k, rho: rho, sv: sv}
		for _, s := range sv {
			switch {
			case s.Cmp(w.tauE) <= 0:
				lv.zero++
			case s.Cmp(rho) <= 0:
				lv.undecided++
			default:
				lv.nonzero++
			}
			if s.Cmp(thr) > 0 {
				lv.r++
			}
		}
		r := lv.r
		d := k - r
		if d == 0 {
			emit()
			return k, reason
		}
		if d > prevD && reason == "" {
			reason = fmt.Sprintf("nullity rises at level %d (%d > %d)", level, d, prevD)
		}
		prevD = d
		for i := r; i < k; i++ {
			for j := range k {
				e[i][j].SetInt64(0)
			}
		}
		q := w.q[:d]
		for i := range d {
			for j := range k {
				q[i][j].Set(a[r+i][j])
			}
		}
		sa := w.sa[:d]
		w.jacobiRows(q, k, nil, sa)
		lv.sigmaA = sa[d-1]
		if r > 0 {
			lv.gap = w.gap.Sub(sv[r-1], sv[r])
		}
		if lv.sigmaA.Cmp(w.tauA) <= 0 {
			emit()
			if reason == "" {
				reason = fmt.Sprintf("A row block σ_min ≤ τ_A at level %d", level)
			}
			if lo {
				return 0, reason
			}
			return k, reason
		}
		if r == 0 {
			emit()
			return 0, reason
		}
		gap := lv.gap
		for i := range d {
			for j := range k {
				q[i][j].Quo(q[i][j], sa[i])
			}
		}
		w.rowComplement(d, k)
		w.mulZ(a, k)
		w.mulZ(e, k)
		for i := r; i < k; i++ {
			for j := range r {
				a[i][j].SetInt64(0)
			}
		}
		thQ := w.thQ.SetInt64(1)
		if gap.Cmp(w.tauE) > 0 {
			thQ.Quo(w.tauE, gap)
			w.minOne(thQ)
		}
		sinZ := w.sinZ
		sinZ.Add(thQ, w.sumSin)
		sinZ.Mul(sinZ, w.normA)
		sinZ.Add(sinZ, w.tauA)
		sinZ.Quo(sinZ, w.den.Sub(lv.sigmaA, w.tauA))
		w.minOne(sinZ)
		lv.sinZ = sinZ
		emit()
		w.sumSin.Add(w.sumSin, sinZ)
		k = r
	}
}

// finiteCountBracket returns the finite-pole bracket of the n×n pencil
// (A, E) from the Van Dooren staircase run exactly (exactCountPrec bits,
// correctly rounded, so identical on every platform) on the stored float64
// data, every tolerance input computed from that data.
//
// The upper count hi zeroes at every level only the singular values
// s ≤ τ_E = n·eps(σ₁(E)), the MATLAB rank floor of descriptorRank
// (eigdecomp.go:1238-1242; level 0 is MATLAB rank(E), r₀). The lower count
// lo zeroes every s ≤ ρ_j, with ρ_0 = τ_E and ρ_j = τ_E + 2·‖E‖·Σ_{i<j} sin θ_i
// a Weyl bound on how far a perturbation of size τ_E on E and
// τ_A = n·eps(σ₁(A)) on A moves a level-j singular value through the
// rotations of the levels before it: sin θ_i bounds the rotation Z of level
// i by Davis–Kahan on the E split (τ_E over the gap σ_r − σ_{r+1}) and
// Wedin on the A row block (τ_A over its σ_min). The factor 2 is a derived
// norm constant, not a tuned one. The E-split term takes τ_E as the size
// of the deflated part, so it under-covers a level that deflates values
// far above τ_E, and lo can then exceed the count of a nearby pencil. A
// level whose A row block has σ_min ≤ τ_A stops both runs in the safe
// direction (hi keeps it finite, lo counts no finite pole).
func finiteCountBracket(A, E *mat.Dense, trace func(lo bool, lv *exactStairLevel)) exactFiniteCount {
	if n, _ := A.Dims(); n == 0 {
		return exactFiniteCount{}
	}
	w := newExactStair(A, E)
	w.trace = trace
	okA := w.sigma1(w.normA, w.a0)
	okE := w.sigma1(w.normE, w.e0)
	w.setTau(w.tauA, w.normA)
	w.setTau(w.tauE, w.normE)
	var c exactFiniteCount
	var rh, rl string
	c.hi, rh = w.staircase(false)
	c.lo, rl = w.staircase(true)
	switch {
	case !okA || !okE:
		c.reason = "Jacobi unconverged on σ₁"
	case rh != "":
		c.reason = "hi: " + rh
	case rl != "":
		c.reason = "lo: " + rl
	}
	return c
}
