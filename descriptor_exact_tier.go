package controlsys

import (
	"fmt"
	"math/big"
	"slices"
	"sync"

	"plantcontrol.org/v1/gonum/mat"
)

// boundedPoly is a polynomial in s, ascending, exact in the units of the
// stored data, with per coefficient its first-order data-rounding bound:
// the largest change of that coefficient, to first order, when every
// stored entry it is formed from moves by ε relative.
type boundedPoly struct {
	coef  []*big.Rat
	bound []float64
}

// zeroToData reports whether coefficient k is zero to the data: at most its
// own first-order data-rounding bound. Rounding of the stored entries
// alone can then make it vanish, so the data do not determine it as
// nonzero (the coefficient-rounding convention).
func (bp boundedPoly) zeroToData(k int) bool {
	if k >= len(bp.coef) {
		return true
	}
	c, _ := new(big.Rat).Abs(bp.coef[k]).Float64()
	return c <= bp.bound[k]
}

// dataDegree is the degree of bp to the data: the largest k whose
// coefficient is not zeroToData, −1 when every one is.
func (bp boundedPoly) dataDegree() int {
	for k := range slices.Backward(bp.coef) {
		if !bp.zeroToData(k) {
			return k
		}
	}
	return -1
}

// pencilDetBounded returns det(sL − M) for the N×N row-major float pencil
// as a boundedPoly of N + 1 coefficients, the bound of coefficient k being
// ε·Σ_rc (|cof_rc,k−1|·|L_rc| + |cof_rc,k|·|M_rc|), cof_rc(s) the cofactor
// of entry (r, c) of sL − M: ∂det/∂(sL − M)_rc = cof_rc. Exact in integer
// arithmetic on the power-of-two scaled pencil (integerPencil, exactDetPoly);
// false when the pencil is outside exactPencilWithinBudget.
func pencilDetBounded(L, M []float64, N int) (boundedPoly, bool) {
	li, mi, ok := integerPencil(L, M)
	if !ok || !exactPencilWithinBudget(li, mi, N) {
		return boundedPoly{}, false
	}
	unit := pencilUnit(L, M, li, mi, N)
	det := exactDetPoly(li, mi, N, N)
	bp := boundedPoly{coef: make([]*big.Rat, N+1), bound: make([]float64, N+1)}
	for k := range bp.coef {
		bp.coef[k] = new(big.Rat)
		if k < len(det) {
			bp.coef[k].Mul(det[k], unit)
		}
	}
	bound := make([]*big.Rat, N+1)
	for k := range bound {
		bound[k] = new(big.Rat)
	}
	if minors, ok := pencilMinorPolysIf(li, mi, N); ok {
		var t, a, el, em big.Int
		acc := make([]*big.Int, N+1)
		for k := range acc {
			acc[k] = new(big.Int)
		}
		for r := range N {
			for c := range N {
				if li[r*N+c].Sign() == 0 && mi[r*N+c].Sign() == 0 {
					continue
				}
				el.Abs(li[r*N+c])
				em.Abs(mi[r*N+c])
				for k, cf := range minors[r*N+c] {
					a.Abs(cf)
					acc[k+1].Add(acc[k+1], t.Mul(&a, &el))
					acc[k].Add(acc[k], t.Mul(&a, &em))
				}
			}
		}
		for k := range acc {
			bound[k].SetInt(acc[k])
		}
	} else if N > 1 {
		ls, ms := make([]*big.Int, (N-1)*(N-1)), make([]*big.Int, (N-1)*(N-1))
		var t big.Rat
		el, em := new(big.Rat), new(big.Rat)
		for r := range N {
			for c := range N {
				if li[r*N+c].Sign() == 0 && mi[r*N+c].Sign() == 0 {
					continue
				}
				q := 0
				for i := range N {
					if i == r {
						continue
					}
					for j := range N {
						if j != c {
							ls[q], ms[q] = li[i*N+j], mi[i*N+j]
							q++
						}
					}
				}
				el.SetInt(new(big.Int).Abs(li[r*N+c]))
				em.SetInt(new(big.Int).Abs(mi[r*N+c]))
				for k, cf := range exactDetPoly(ls, ms, N-1, N-1) {
					a := new(big.Rat).Abs(cf)
					bound[k+1].Add(bound[k+1], t.Mul(a, el))
					bound[k].Add(bound[k], t.Mul(a, em))
				}
			}
		}
	} else if N == 1 {
		bound[1].SetInt(new(big.Int).Abs(li[0]))
		bound[0].SetInt(new(big.Int).Abs(mi[0]))
	}
	for k, b := range bound {
		f, _ := b.Mul(b, unit).Float64()
		bp.bound[k] = eps() * f
	}
	return bp, true
}

// pencilMinorPolys returns, per entry (r, c) of the N×N integer pencil sL −
// M (N > 1), the ascending coefficients of ±its minor, the adjugate's entry
// (c, r), exactly, as integers: the adjugate det(P)·P⁻¹ at N integer points s where P =
// sL − M is nonsingular, found among s = 0, 1, …, 2N, by fraction-free
// Gauss–Jordan elimination of [P | I] (Bareiss's exact divisions; the right
// block ends as ±adj P, the sign of the row swaps), and Newton
// interpolation of each entry, of degree at most N − 1, in integers: the
// divided differences of an integer polynomial at integer points are
// integers, so every division is exact. False when fewer than N such points exist (det(sL − M) ≡ 0).
func pencilMinorPolys(L, M []*big.Int, N int) ([][]*big.Int, bool) {
	if N < 2 {
		return nil, false
	}
	var xs []int64
	var vals [][]*big.Int
	aug := make([][]*big.Int, N)
	for i := range aug {
		aug[i] = make([]*big.Int, 2*N)
		for j := range aug[i] {
			aug[i][j] = new(big.Int)
		}
	}
	var u, w big.Int
	prev := new(big.Int)
	for x := int64(0); x <= int64(2*N) && len(xs) < N; x++ {
		sx := big.NewInt(x)
		for i := range N {
			for j := range N {
				aug[i][j].Sub(u.Mul(sx, L[i*N+j]), M[i*N+j])
				aug[i][N+j].SetInt64(0)
			}
			aug[i][N+i].SetInt64(1)
		}
		prev.SetInt64(1)
		singular := false
		for k := range N {
			piv := -1
			for r := k; r < N; r++ {
				if aug[r][k].Sign() != 0 {
					piv = r
					break
				}
			}
			if piv < 0 {
				singular = true
				break
			}
			aug[k], aug[piv] = aug[piv], aug[k]
			pk := aug[k][k]
			for r := range N {
				if r == k {
					continue
				}
				ark := new(big.Int).Set(aug[r][k])
				for j := range 2 * N {
					if j == k {
						continue
					}
					u.Mul(pk, aug[r][j])
					w.Mul(ark, aug[k][j])
					aug[r][j].Quo(u.Sub(&u, &w), prev)
				}
				aug[r][k].SetInt64(0)
			}
			prev.Set(pk)
		}
		if singular {
			continue
		}
		v := make([]*big.Int, N*N)
		for r := range N {
			for c := range N {
				v[r*N+c] = new(big.Int).Set(aug[c][N+r])
			}
		}
		xs = append(xs, x)
		vals = append(vals, v)
	}
	if len(xs) < N {
		return nil, false
	}
	out := make([][]*big.Int, N*N)
	var d, q big.Int
	for e := range out {
		y := make([]*big.Int, N)
		for k := range N {
			y[k] = new(big.Int).Set(vals[k][e])
		}
		for j := 1; j < N; j++ {
			for k := N - 1; k >= j; k-- {
				y[k].Sub(y[k], y[k-1])
				d.SetInt64(xs[k] - xs[k-j])
				y[k].Quo(y[k], &d)
			}
		}
		poly := make([]*big.Int, N)
		for i := range poly {
			poly[i] = new(big.Int)
		}
		poly[0].Set(y[N-1])
		deg := 0
		for k := N - 2; k >= 0; k-- {
			d.SetInt64(xs[k])
			for i := deg; i >= 0; i-- {
				poly[i+1].Add(poly[i+1], poly[i])
				poly[i].Neg(q.Mul(&d, poly[i]))
			}
			poly[0].Add(poly[0], y[k])
			deg++
		}
		out[e] = poly
	}
	return out, true
}

func pencilMinorPolysIf(L, M []*big.Int, N int) ([][]*big.Int, bool) {
	if !pencilMinorsByAdjugate {
		return nil, false
	}
	return pencilMinorPolys(L, M, N)
}

// pencilMinorsByAdjugate is a test seam: false forces the minors of
// pencilDetBounded through one exactDetPoly per entry.
var pencilMinorsByAdjugate = true

// pencilUnit is the factor 2^(N·lo) that takes det of the integer pencil
// (li, mi) of integerPencil back to the units of the float pencil (L, M).
func pencilUnit(L, M []float64, li, mi []*big.Int, N int) *big.Rat {
	unit := big.NewRat(1, 1)
	for _, pair := range [...]struct {
		f []float64
		z []*big.Int
	}{{L, li}, {M, mi}} {
		for i, x := range pair.f {
			if x == 0 {
				continue
			}
			step := new(big.Rat).SetFloat64(x)
			step.Quo(step, new(big.Rat).SetInt(pair.z[i]))
			for range N {
				unit.Mul(unit, step)
			}
			return unit
		}
	}
	return unit
}

// exactPencilMaxStates and exactPencilBitBudget cap the exact pencil
// tier's cost: a descriptor of order n gives (n + 1)×(n + 1) bordered
// pencils, each with N² cofactor determinants, so the tier takes n at most
// exactPencilMaxStates and N·(s + 64) at most exactPencilBitBudget, s the
// largest scaled entry's bit length, which bounds the integer size of the
// determinants. Beyond either the float path decides alone.
const (
	exactPencilMaxStates = 11
	exactPencilBitBudget = 4800
)

// exactPencilWithinBudget applies exactWithinBudget-style gates to an N×N
// integer pencil: N at most exactPencilMaxStates + 1 and N·(s+64) at most
// exactPencilBitBudget, s the largest entry's bit length.
func exactPencilWithinBudget(li, mi []*big.Int, N int) bool {
	if N > exactPencilMaxStates+1 {
		return false
	}
	s := 0
	for _, v := range [][]*big.Int{li, mi} {
		for _, x := range v {
			s = max(s, x.BitLen())
		}
	}
	return N*(s+64) <= exactPencilBitBudget
}

// exactPencilData holds, for a descriptor model (no internal delays), den =
// det(sE − A) and the numerators num_ij = det([sE − A, −b_j; c_i, d_ij]) =
// den·G_ij exactly on the stored data, with their first-order
// data-rounding bounds (pencilDetBounded).
type exactPencilData struct {
	n, m, p int
	den     boundedPoly
	num     [][]boundedPoly
}

// newExactPencilData computes exactPencilData for sys; false when a pencil
// is outside exactPencilWithinBudget.
func newExactPencilData(sys *System) (*exactPencilData, bool) {
	n, m, p := sys.Dims()
	if n == 0 || m == 0 || p == 0 || sys.E == nil || sys.internalDelayCount() > 0 {
		return nil, false
	}
	if exactPencilShared {
		if x, ok, done := newExactPencilDataShared(sys); done {
			return x, ok
		}
	}
	den, ok := pencilDetBounded(denseData(sys.E), denseData(sys.A), n)
	if !ok {
		return nil, false
	}
	x := &exactPencilData{n: n, m: m, p: p, den: den, num: make([][]boundedPoly, p)}
	N := n + 1
	L, M := make([]float64, N*N), make([]float64, N*N)
	for i := range p {
		x.num[i] = make([]boundedPoly, m)
		for j := range m {
			for r := range n {
				for c := range n {
					L[r*N+c] = sys.E.At(r, c)
					M[r*N+c] = sys.A.At(r, c)
				}
				M[r*N+n] = sys.B.At(r, j)
				M[n*N+r] = -sys.C.At(i, r)
			}
			M[n*N+n] = -sys.D.At(i, j)
			if x.num[i][j], ok = pencilDetBounded(L, M, N); !ok {
				return nil, false
			}
		}
	}
	return x, true
}

// exactPencilShared is a test seam: false computes every pencil of
// newExactPencilData on its own (pencilDetBounded).
var exactPencilShared = true

// newExactPencilDataShared is newExactPencilData computing every
// determinant and minor from one exact inverse of K = sE − A per point:
// with Q = d·K⁻¹ and d = ±det K from fraction-free Gauss–Jordan elimination
// of [K | I], the bordered pencil P = [K, u; v, w] (u = −b_j, v = c_i, w =
// d_ij) has σ = d·w − v·Q·u = ±det P and σ·P⁻¹ = [(σQ + Qu·vQ)/d, −Qu; −vQ,
// d] (the division exact), so its minors are the moduli of those entries.
// Evaluated in integers at n + 1 points where K is nonsingular, on the
// data scaled to integers by one common power of two, and interpolated
// (Newton, in integers), the determinants and minors are the same
// polynomials, up to that scaling, as pencilDetBounded's, and every
// coefficient and bound is the same rational: bound_k = ε·Σ|minor||entry|
// at that scaling times its unit. done is false where the common scaling
// or the points are not available, and the caller computes each pencil on
// its own; ok false as pencilDetBounded's for a pencil outside its gates.
func newExactPencilDataShared(sys *System) (x *exactPencilData, ok, done bool) {
	n, m, p := sys.Dims()
	N := n + 1
	e, a, b, c, d := denseData(sys.E), denseData(sys.A), denseData(sys.B), denseData(sys.C), denseData(sys.D)
	if li, mi, ok := integerPencil(e, a); !ok || !exactPencilWithinBudget(li, mi, n) {
		return nil, false, true
	}
	L, M := make([]float64, N*N), make([]float64, N*N)
	for i := range p {
		for j := range m {
			for r := range n {
				for cc := range n {
					L[r*N+cc] = e[r*n+cc]
					M[r*N+cc] = a[r*n+cc]
				}
				M[r*N+n] = b[r*m+j]
				M[n*N+r] = -c[i*n+r]
			}
			M[n*N+n] = -d[i*m+j]
			if li, mi, ok := integerPencil(L, M); !ok || !exactPencilWithinBudget(li, mi, N) {
				return nil, false, true
			}
		}
	}
	rest := slices.Concat(a, b, c, d)
	ei, ri, ok := integerPencil(e, rest)
	if !ok {
		return nil, false, false
	}
	step := new(big.Rat)
	found := false
	for k, v := range slices.Concat(e, rest) {
		if v != 0 {
			z := ei
			idx := k
			if k >= len(e) {
				z, idx = ri, k-len(e)
			}
			step.SetFloat64(v)
			step.Quo(step, new(big.Rat).SetInt(z[idx]))
			found = true
			break
		}
	}
	if !found {
		return nil, false, false
	}
	ai := ri[:n*n]
	bi := ri[n*n : n*n+n*m]
	ci := ri[n*n+n*m : n*n+n*m+p*n]
	di := ri[n*n+n*m+p*n:]
	var xs []int64
	var dets [][]*big.Int
	var qs [][]*big.Int
	type border struct{ sigma, adj []*big.Int }
	var nums [][]border
	aug := make([][]*big.Int, n)
	for i := range aug {
		aug[i] = make([]*big.Int, 2*n)
		for j := range aug[i] {
			aug[i][j] = new(big.Int)
		}
	}
	var u, w big.Int
	prev := new(big.Int)
	for xv := int64(0); xv <= int64(2*N+1) && len(xs) < N; xv++ {
		sx := big.NewInt(xv)
		for i := range n {
			for j := range n {
				aug[i][j].Sub(u.Mul(sx, ei[i*n+j]), ai[i*n+j])
				aug[i][n+j].SetInt64(0)
			}
			aug[i][n+i].SetInt64(1)
		}
		prev.SetInt64(1)
		swaps := 0
		singular := false
		for k := range n {
			piv := -1
			for r := k; r < n; r++ {
				if aug[r][k].Sign() != 0 {
					piv = r
					break
				}
			}
			if piv < 0 {
				singular = true
				break
			}
			if piv != k {
				aug[k], aug[piv] = aug[piv], aug[k]
				swaps++
			}
			pk := aug[k][k]
			for r := range n {
				if r == k {
					continue
				}
				ark := new(big.Int).Set(aug[r][k])
				for j := range 2 * n {
					if j == k {
						continue
					}
					u.Mul(pk, aug[r][j])
					w.Mul(ark, aug[k][j])
					aug[r][j].Quo(u.Sub(&u, &w), prev)
				}
				aug[r][k].SetInt64(0)
			}
			prev.Set(pk)
		}
		if singular {
			continue
		}
		dd := new(big.Int).Set(aug[0][0])
		Q := make([]*big.Int, n*n)
		for r := range n {
			for cc := range n {
				Q[r*n+cc] = new(big.Int).Set(aug[r][n+cc])
			}
		}
		sgn := int64(1)
		if swaps%2 == 1 {
			sgn = -1
		}
		detK := new(big.Int).Mul(dd, big.NewInt(sgn))
		pt := make([]border, p*m)
		qu, vq := make([]*big.Int, n), make([]*big.Int, n)
		for i := range n {
			qu[i], vq[i] = new(big.Int), new(big.Int)
		}
		var t big.Int
		for i := range p {
			for j := range m {
				for r := range n {
					qu[r].SetInt64(0)
					vq[r].SetInt64(0)
				}
				for r := range n {
					for l := range n {
						qu[r].Sub(qu[r], t.Mul(Q[r*n+l], bi[l*m+j]))
						vq[r].Add(vq[r], t.Mul(ci[i*n+l], Q[l*n+r]))
					}
				}
				sigma := new(big.Int).Mul(dd, di[i*m+j])
				for l := range n {
					sigma.Sub(sigma, t.Mul(ci[i*n+l], qu[l]))
				}
				adj := make([]*big.Int, N*N)
				for r := range n {
					for cc := range n {
						z := new(big.Int).Mul(sigma, Q[r*n+cc])
						z.Add(z, t.Mul(qu[r], vq[cc]))
						adj[r*N+cc] = z.Quo(z, dd)
					}
					adj[r*N+n] = new(big.Int).Neg(qu[r])
					adj[n*N+r] = new(big.Int).Neg(vq[r])
				}
				adj[n*N+n] = new(big.Int).Set(dd)
				pt[i*m+j] = border{sigma: []*big.Int{new(big.Int).Mul(sigma, big.NewInt(sgn))}, adj: adj}
			}
		}
		xs = append(xs, xv)
		dets = append(dets, []*big.Int{detK})
		qs = append(qs, Q)
		nums = append(nums, pt)
	}
	if len(xs) < N {
		return nil, false, false
	}
	interp := func(y []*big.Int, pts int) []*big.Int {
		y = y[:pts]
		var dv, q big.Int
		for j := 1; j < pts; j++ {
			for k := pts - 1; k >= j; k-- {
				y[k].Sub(y[k], y[k-1])
				dv.SetInt64(xs[k] - xs[k-j])
				y[k].Quo(y[k], &dv)
			}
		}
		poly := make([]*big.Int, pts)
		for i := range poly {
			poly[i] = new(big.Int)
		}
		poly[0].Set(y[pts-1])
		deg := 0
		for k := pts - 2; k >= 0; k-- {
			dv.SetInt64(xs[k])
			for i := deg; i >= 0; i-- {
				poly[i+1].Add(poly[i+1], poly[i])
				poly[i].Neg(q.Mul(&dv, poly[i]))
			}
			poly[0].Add(poly[0], y[k])
			deg++
		}
		return poly
	}
	series := func(get func(k int) *big.Int, pts int) []*big.Int {
		y := make([]*big.Int, pts)
		for k := range pts {
			y[k] = new(big.Int).Set(get(k))
		}
		return interp(y, pts)
	}
	unit := func(size int) *big.Rat {
		u := big.NewRat(1, 1)
		for range size {
			u.Mul(u, step)
		}
		return u
	}
	bounded := func(det []*big.Int, minor func(r, cc int) []*big.Int, lent, ment func(r, cc int) *big.Int, size int) boundedPoly {
		un := unit(size)
		bp := boundedPoly{coef: make([]*big.Rat, size+1), bound: make([]float64, size+1)}
		for k := range bp.coef {
			bp.coef[k] = new(big.Rat)
			if k < len(det) {
				bp.coef[k].Mul(new(big.Rat).SetInt(det[k]), un)
			}
		}
		acc := make([]*big.Int, size+1)
		for k := range acc {
			acc[k] = new(big.Int)
		}
		var t, av, el, em big.Int
		for r := range size {
			for cc := range size {
				le, me := lent(r, cc), ment(r, cc)
				if le.Sign() == 0 && me.Sign() == 0 {
					continue
				}
				el.Abs(le)
				em.Abs(me)
				for k, cf := range minor(r, cc) {
					av.Abs(cf)
					acc[k+1].Add(acc[k+1], t.Mul(&av, &el))
					acc[k].Add(acc[k], t.Mul(&av, &em))
				}
			}
		}
		for k, v := range acc {
			f, _ := new(big.Rat).Mul(new(big.Rat).SetInt(v), un).Float64()
			bp.bound[k] = eps() * f
		}
		return bp
	}
	zero := new(big.Int)
	x = &exactPencilData{n: n, m: m, p: p, num: make([][]boundedPoly, p)}
	denDet := series(func(k int) *big.Int { return dets[k][0] }, N)
	denMinors := make([][]*big.Int, n*n)
	if n > 1 {
		for r := range n {
			for cc := range n {
				denMinors[r*n+cc] = series(func(k int) *big.Int { return qs[k][cc*n+r] }, n)
			}
		}
	}
	if n == 1 {
		x.den = bounded(denDet, func(r, cc int) []*big.Int { return []*big.Int{big.NewInt(1)} },
			func(r, cc int) *big.Int { return ei[0] }, func(r, cc int) *big.Int { return ai[0] }, 1)
	} else {
		x.den = bounded(denDet, func(r, cc int) []*big.Int { return denMinors[r*n+cc] },
			func(r, cc int) *big.Int { return ei[r*n+cc] }, func(r, cc int) *big.Int { return ai[r*n+cc] }, n)
	}
	for i := range p {
		x.num[i] = make([]boundedPoly, m)
		for j := range m {
			e2 := i*m + j
			det := series(func(k int) *big.Int { return nums[k][e2].sigma[0] }, N)
			minors := make([][]*big.Int, N*N)
			for r := range N {
				for cc := range N {
					minors[r*N+cc] = series(func(k int) *big.Int { return nums[k][e2].adj[cc*N+r] }, N)
				}
			}
			lent := func(r, cc int) *big.Int {
				if r < n && cc < n {
					return ei[r*n+cc]
				}
				return zero
			}
			ment := func(r, cc int) *big.Int {
				switch {
				case r < n && cc < n:
					return ai[r*n+cc]
				case r < n:
					return bi[r*m+j]
				case cc < n:
					return ci[i*n+cc]
				}
				return di[i*m+j]
			}
			x.num[i][j] = bounded(det, func(r, cc int) []*big.Int { return minors[r*N+cc] }, lent, ment, N)
		}
	}
	return x, true, true
}

// exactPencilTF is a trimmed model of exactPencilData: den kept to degree
// nf and num_ij to deg[i][j], the coefficients above dropped, all divided
// by den's coefficient of s^nf (den monic), ascending; numFull and denFull
// keep every coefficient in the same units. nf is the finite pole count
// and polyDeg, the most by which a deg[i][j] exceeds nf, the degree of the
// polynomial part of num_ij/den (0 for a proper model).
type exactPencilTF struct {
	nf, polyDeg      int
	deg              [][]int
	num, numFull     [][][]*big.Rat
	den, denFull     []*big.Rat
	denBound         []float64
	numLead, denLead []float64
}

// trimmed returns the model of x with den kept to degree nf and num_ij to
// deg[i][j] (−1: zero); false when den's coefficient of s^nf is zero.
func (x *exactPencilData) trimmed(nf int, deg [][]int) (exactPencilTF, bool) {
	lead := x.den.coef[nf]
	if lead.Sign() == 0 {
		return exactPencilTF{}, false
	}
	t := exactPencilTF{nf: nf, deg: deg}
	t.den, t.denFull = scaledCoefs(x.den, nf, lead)
	la, _ := new(big.Rat).Abs(lead).Float64()
	t.denBound = make([]float64, len(x.den.bound))
	for k, b := range x.den.bound {
		t.denBound[k] = b / la
	}
	t.num, t.numFull = make([][][]*big.Rat, x.p), make([][][]*big.Rat, x.p)
	for i := range x.p {
		t.num[i], t.numFull[i] = make([][]*big.Rat, x.m), make([][]*big.Rat, x.m)
		for j := range x.m {
			t.num[i][j], t.numFull[i][j] = scaledCoefs(x.num[i][j], deg[i][j], lead)
			t.polyDeg = max(t.polyDeg, deg[i][j]-nf)
		}
	}
	return t, true
}

// scaledCoefs returns the coefficients of bp up to degree deg and all of
// them, each divided by lead.
func scaledCoefs(bp boundedPoly, deg int, lead *big.Rat) (trim, full []*big.Rat) {
	full = make([]*big.Rat, len(bp.coef))
	for k, c := range bp.coef {
		full[k] = new(big.Rat).Quo(c, lead)
	}
	return full[:max(deg+1, 0)], full
}

// transferFunc returns t as a transfer function of sys's I/O, num_ij/den
// rounded to float64, each row over den.
func (t exactPencilTF) transferFunc(sys *System) *TransferFunc {
	_, m, p := sys.Dims()
	desc := func(c []*big.Rat) []float64 {
		out := make([]float64, max(len(c), 1))
		for k, v := range c {
			out[len(c)-1-k], _ = v.Float64()
		}
		return out
	}
	tf := &TransferFunc{Num: make([][][]float64, p), Den: make([][]float64, p), Dt: sys.Dt,
		InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}
	for i := range p {
		tf.Den[i] = desc(t.den)
		tf.Num[i] = make([][]float64, m)
		for j := range m {
			tf.Num[i][j] = desc(t.num[i][j])
		}
	}
	return tf
}

// chainVerdict is the exact tier's verdict on a descriptor model beside an
// index ≥ 2 chain, shared by TransferFunction, IsProper and Step: tf its
// transfer function to the data, nf its finite pole count, polyDeg the
// degree of its polynomial part and certified whether descriptorTFJudge
// verifies tf. nf0 and deg0 are the convention's starting degrees, climbs
// the numerator degree raises the search took and ceiling whether it ended
// refuted with no degree left to raise.
type chainVerdict struct {
	tf        *TransferFunc
	nf        int
	polyDeg   int
	certified bool
	climbs    int
	ceiling   bool
	// convDeg holds the convention's numerator degrees (dataDegree), and
	// why the last candidate is not certified (descriptorTFJudgeWhy);
	// diagnostics only.
	convDeg [][]int
	why     string
}

// exactChainVerdict returns the verdict of the exact tier, false when it
// does not apply: an explicit model, internal delays, no inputs or outputs,
// E of full rank, a singular or index-1 pencil, a pencil outside
// exactPencilWithinBudget, or a convention count of finite poles above the
// numerical rank of E (stiffChainERank): deg det(sE − A) ≤ rank E, so such
// a count takes rounded chain eigenvalues for finite poles. There the float
// path decides alone. The value is the exact num_ij/den of the stored data
// trimmed, rounded to float64. The coefficient-rounding convention
// (boundedPoly.dataDegree) gives den's degree nf, a bound of the search;
// every numerator starts at degree nf, the least-degree (proper) model, and
// a refutation of entries by descriptorTFJudge raises each refuted
// numerator's degree by one, up to n, until tf is certified, is only short
// of verified points, or a refuted entry has no degree left to raise. The
// last model tried is the value. The convention's numerator degrees are
// kept as a diagnostic only: a polynomial term the data cannot tell from
// the tail of a pole beyond the determined band counts as absent.
func (sys *System) exactChainVerdict() (chainVerdict, bool) {
	return sys.exactChainVerdictCached(newPencilCache())
}

// exactChainVerdictCached is exactChainVerdict with c's per-point work
// (pencilCache), shared across the candidates it judges.
func (sys *System) exactChainVerdictCached(c *pencilCache) (chainVerdict, bool) {
	if !sys.exactChainApplies() {
		return chainVerdict{}, false
	}
	n, m, p := sys.Dims()
	x, ok := newExactPencilData(sys)
	if !ok {
		return chainVerdict{}, false
	}
	nf := x.den.dataDegree()
	if nf < 0 || nf > stiffChainERank(sys.E) {
		return chainVerdict{}, false
	}
	deg := make([][]int, p)
	for i := range p {
		deg[i] = make([]int, m)
		for j := range m {
			deg[i][j] = x.num[i][j].dataDegree()
		}
	}
	conv := deg
	deg = make([][]int, p)
	for i := range p {
		deg[i] = make([]int, m)
		for j := range m {
			deg[i][j] = nf
		}
	}
	climbs := 0
	for {
		t, ok := x.trimmed(nf, deg)
		if !ok {
			return chainVerdict{}, false
		}
		tf := t.transferFunc(sys)
		cert, refuted, why := descriptorTFJudgeWhy(sys, tf, c)
		v := chainVerdict{tf: tf, nf: nf, polyDeg: t.polyDeg, certified: cert, climbs: climbs, convDeg: conv,
			why: fmt.Sprintf("nf %d, numerator degrees %v: %s", nf, deg, why)}
		if cert || len(refuted) == 0 {
			return v, true
		}
		for _, e := range refuted {
			i, j := e/m, e%m
			if deg[i][j] >= n {
				v.ceiling = true
				v.why += "; ceiling"
				return v, true
			}
			deg[i][j]++
		}
		climbs++
	}
}

// exactChainApplies reports whether the exact tier can apply to sys: a
// descriptor without internal delays, with inputs and outputs, E rank
// deficient and the pencil regular and not index 1. It is cheap; the
// budget gates come after it.
func (sys *System) exactChainApplies() bool {
	if !exactChainTier || !sys.IsDescriptor() || sys.internalDelayCount() > 0 {
		return false
	}
	n, m, p := sys.Dims()
	if n == 0 || m == 0 || p == 0 {
		return false
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if !eRankDeficient(sys.E, &lu) {
		return false
	}
	if _, ok := regularDescriptorShift(sys.A, sys.E, n); !ok {
		return false
	}
	_, index1 := pencilIndexOne(sys.A, sys.E, n, mat.Norm(sys.A, 2))
	return !index1
}

// exactChainTier is a test seam: false leaves every model to the float
// path, as before the exact tier.
var exactChainTier = true

// descriptorRegime is which model TransferFunction, IsProper and Step
// share for a descriptor: regimeFloat, the float path's (the Weierstrass
// reduction or E⁻¹), whenever its transfer function certifies
// (descriptorTFVerified) or the exact tier does not apply; regimeExact,
// the exact tier's certified least-degree model (exactChainVerdict) when
// the float path's does not certify; regimeNone, the float path's
// uncertified value or error, when neither certifies. In regimeNone v
// holds the exact tier's uncertified model, if any, which the APIs return
// with ErrAccuracyUnverified only where the float path has no value.
type descriptorRegime struct {
	kind      int
	v         chainVerdict
	hasV      bool
	mainRes   *TransferFuncResult
	mainErr   error
	certified bool
}

const (
	regimeFloat = iota
	regimeExact
	regimeNone
)

// chainRegime returns sys's descriptorRegime, memoized per model (the key
// holds every stored entry bit for bit, tfVerdictKey), so the three APIs
// read one decision. The float path is judged without TransferFuncOpts;
// the returned results are the caller's own copies.
func (sys *System) chainRegime() descriptorRegime {
	if !exactChainTier || !sys.IsDescriptor() || sys.internalDelayCount() > 0 {
		return descriptorRegime{kind: regimeFloat}
	}
	key := tfVerdictKey(sys, &TransferFunc{})
	chainRegimes.Lock()
	r, hit := chainRegimes.m[key]
	chainRegimes.Unlock()
	if !hit {
		r = descriptorRegime{kind: regimeFloat}
		if sys.exactChainApplies() {
			r = sys.decideRegime()
		}
		chainRegimes.Lock()
		if chainRegimes.m == nil || len(chainRegimes.m) >= tfVerdictMemoSize {
			chainRegimes.m = make(map[string]descriptorRegime)
		}
		chainRegimes.m[key] = r
		chainRegimes.Unlock()
	}
	if r.hasV {
		r.v.tf = r.v.tf.Copy()
	}
	if r.mainRes != nil {
		c := *r.mainRes
		c.TF = c.TF.Copy()
		c.RowDegrees = slices.Clone(c.RowDegrees)
		r.mainRes = &c
	}
	return r
}

func (sys *System) decideRegime() descriptorRegime {
	c := newPencilCache()
	res, certified, err := sys.transferRationalCertified(nil, c)
	r := descriptorRegime{kind: regimeFloat, mainRes: res, mainErr: err, certified: certified}
	if err == nil && certified {
		return r
	}
	r.v, r.hasV = sys.exactChainVerdictCached(c)
	if r.hasV && r.v.certified {
		r.kind = regimeExact
	} else {
		r.kind = regimeNone
	}
	return r
}

var chainRegimes struct {
	sync.Mutex
	m map[string]descriptorRegime
}

// result returns v as a TransferFunction result: MinimalOrder and every
// row degree nf.
func (v chainVerdict) result() *TransferFuncResult {
	rows := make([]int, len(v.tf.Den))
	for i := range rows {
		rows[i] = v.nf
	}
	return &TransferFuncResult{TF: v.tf, MinimalOrder: v.nf, RowDegrees: rows}
}
