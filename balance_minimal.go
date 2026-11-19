package controlsys

import (
	"math"
	"math/big"
	"math/bits"
	"sync"

	"plantcontrol.org/v1/gonum/mat"
)

// minimality is the exact verdict on the stored data of a model.
type minimality int

const (
	minimalExact minimality = iota
	notMinimalExact
	minimalityUndetermined
)

// minimalityBudget bounds the exact test: certPrimes primes for the
// invariant-subspace certificate, then work modular operations, about
// primes·n³·m, for the Hadamard-bound proof.
type minimalityBudget struct {
	certPrimes int
	work       float64
}

var defaultMinimalityBudget = minimalityBudget{certPrimes: 24, work: 1 << 28}

// exactMinimality decides whether the stored (A, B, C) is minimal, that is
// whether [B AB … Aⁿ⁻¹B] and [Cᵀ AᵀCᵀ … ] have exact rank n over ℚ. Each
// float64 is a dyadic rational, so after a power-of-two scaling per matrix,
// which scales whole Krylov columns, the Krylov matrix is an integer matrix K.
// Full rank modulo any prime proves full rank over ℚ. A rank deficiency is
// proved by a rational W of full row rank with W·B = 0 and W·A = H·W,
// reconstructed from its images modulo primes and verified exactly, so that
// W·AᵏB = HᵏW·B = 0; failing that, by rank below n modulo primes whose
// product exceeds twice the Hadamard bound on the n×n minors of K, which
// proves every such minor zero.
func exactMinimality(A, B, C *mat.Dense) minimality {
	a := newDyadicImage(A, false)
	ctrb := exactKrylovFullRank(a, newDyadicImage(B, false), defaultMinimalityBudget)
	if ctrb == minimalExact {
		return exactKrylovFullRank(newDyadicImage(A, true), newDyadicImage(C, true), defaultMinimalityBudget)
	}
	return ctrb
}

// dyadicImage holds a matrix M as M = 2^e0·Z with Z integer: entry k of Z is
// ±mant[k]·2^shift[k].
type dyadicImage struct {
	rows, cols int
	mant       []uint64
	neg        []bool
	shift      []int
	bits       float64
}

func newDyadicImage(M *mat.Dense, transpose bool) dyadicImage {
	r, c := M.Dims()
	if transpose {
		r, c = c, r
	}
	raw := M.RawMatrix()
	d := dyadicImage{rows: r, cols: c, mant: make([]uint64, r*c), neg: make([]bool, r*c), shift: make([]int, r*c)}
	exps := make([]int, r*c)
	e0 := math.MaxInt
	for i := range r {
		for j := range c {
			src := i*raw.Stride + j
			if transpose {
				src = j*raw.Stride + i
			}
			x := raw.Data[src]
			k := i*c + j
			if x == 0 {
				continue
			}
			fr, e := math.Frexp(math.Abs(x))
			mant := uint64(math.Ldexp(fr, 53))
			tz := bits.TrailingZeros64(mant)
			d.mant[k] = mant >> tz
			d.neg[k] = x < 0
			exps[k] = e - 53 + tz
			e0 = min(e0, exps[k])
		}
	}
	for k, m := range d.mant {
		if m != 0 {
			d.shift[k] = exps[k] - e0
			d.bits = math.Max(d.bits, float64(d.shift[k]+bits.Len64(m)))
		}
	}
	return d
}

func (d dyadicImage) modP(p uint64, dst []uint64) {
	for k, m := range d.mant {
		if m == 0 {
			dst[k] = 0
			continue
		}
		v := m % p * powModP(2, uint64(d.shift[k]), p) % p
		if d.neg[k] && v != 0 {
			v = p - v
		}
		dst[k] = v
	}
}

func powModP(b, e, p uint64) uint64 {
	r := uint64(1)
	b %= p
	for ; e > 0; e >>= 1 {
		if e&1 != 0 {
			r = r * b % p
		}
		b = b * b % p
	}
	return r
}

// exactKrylovFullRank decides whether [B AB … Aⁿ⁻¹B] of the integer images
// a (n×n) and b (n×m) has rank n over ℚ.
func exactKrylovFullRank(a, b dyadicImage, budget minimalityBudget) minimality {
	n, m := b.rows, b.cols
	ws := newKrylovModP(n, m)
	if ws.rank(a, b, modPPrime) == n {
		return minimalExact
	}
	if v, ok := ws.invariantCertificate(a, b, budget.certPrimes); ok {
		return v
	}
	la := math.Log2(float64(n)) + a.bits
	lb := 0.5*math.Log2(float64(n)) + b.bits
	hadamard := 0.0
	for k, cols := n-1, 0; k >= 0 && cols < n; k-- {
		take := min(m, n-cols)
		hadamard += float64(take) * (float64(k)*la + lb)
		cols += take
	}
	need := hadamard + 2 - math.Log2(float64(modPPrime))
	perPrime := float64(n) * float64(n) * float64(n) * float64(m)
	if perPrime*need/31 > budget.work {
		return minimalityUndetermined
	}
	for i := 0; need > 0; i++ {
		p := krylovPrime(i)
		if ws.rank(a, b, p) == n {
			return minimalExact
		}
		need -= math.Log2(float64(p))
	}
	return notMinimalExact
}

type krylovModP struct {
	n, m    int
	a, cur  []uint64
	next, v []uint64
	basis   []uint64
	pivots  []int
}

func newKrylovModP(n, m int) *krylovModP {
	return &krylovModP{
		n: n, m: m,
		a: make([]uint64, n*n), cur: make([]uint64, n*m), next: make([]uint64, n*m),
		v: make([]uint64, n), basis: make([]uint64, n*n), pivots: make([]int, 0, n),
	}
}

// rank returns the rank modulo the prime p < 2³² of the Krylov matrix of
// a and b, stopping once a block adds no new direction (the span is then
// A-invariant).
func (w *krylovModP) rank(a, b dyadicImage, p uint64) int {
	n, m := w.n, w.m
	a.modP(p, w.a)
	b.modP(p, w.cur)
	w.pivots = w.pivots[:0]
	for range n {
		grew := false
		for j := range m {
			for i := range n {
				w.v[i] = w.cur[i*m+j]
			}
			if w.insert(p) {
				grew = true
				if len(w.pivots) == n {
					return n
				}
			}
		}
		if !grew {
			break
		}
		for i := range n {
			for j := range m {
				var hi, lo uint64
				for k := range n {
					var c uint64
					lo, c = bits.Add64(lo, w.a[i*n+k]*w.cur[k*m+j], 0)
					hi += c
				}
				w.next[i*m+j] = bits.Rem64(hi, lo, p)
			}
		}
		w.cur, w.next = w.next, w.cur
	}
	return len(w.pivots)
}

// insert reduces w.v against the stored basis and appends it, normalized,
// when a nonzero remainder is left.
func (w *krylovModP) insert(p uint64) bool {
	n := w.n
	v := w.v
	for r, piv := range w.pivots {
		c := v[piv]
		if c == 0 {
			continue
		}
		row := w.basis[r*n : r*n+n]
		for i := range n {
			if row[i] != 0 {
				v[i] = (v[i] + (p-c)*row[i]) % p
			}
		}
	}
	piv := -1
	for i, x := range v {
		if x != 0 {
			piv = i
			break
		}
	}
	if piv < 0 {
		return false
	}
	inv := powModP(v[piv], p-2, p)
	r := len(w.pivots)
	row := w.basis[r*n : r*n+n]
	for i := range n {
		row[i] = v[i] * inv % p
	}
	w.pivots = append(w.pivots, piv)
	return true
}

var krylovPrimeCache struct {
	sync.Mutex
	primes []uint64
}

// krylovPrime returns the i-th prime below modPPrime in decreasing order.
// ProbablyPrime is exact below 2⁶⁴.
func krylovPrime(i int) uint64 {
	c := &krylovPrimeCache
	c.Lock()
	defer c.Unlock()
	x := uint64(modPPrime)
	if len(c.primes) > 0 {
		x = c.primes[len(c.primes)-1]
	}
	for len(c.primes) <= i {
		x -= 2
		if new(big.Int).SetUint64(x).ProbablyPrime(0) {
			c.primes = append(c.primes, x)
		}
	}
	return c.primes[i]
}

// invariantCertificate looks for an exact left uncontrollable subspace W of
// the integer images a and b: the reduced row echelon basis, with identity
// on its free columns, of the left null space of the Krylov matrix modulo
// primes, combined by the Chinese remainder theorem, rationally
// reconstructed and verified in exact arithmetic. ok is false when no
// verified W turns up within primes primes.
func (w *krylovModP) invariantCertificate(a, b dyadicImage, primes int) (verdict minimality, ok bool) {
	n := w.n
	var (
		free    []int
		residue []*big.Int
		modulus *big.Int
		rank    = -1
	)
	for i := -1; i < primes; i++ {
		p := uint64(modPPrime)
		if i >= 0 {
			p = krylovPrime(i)
		}
		r := w.rank(a, b, p)
		if r == n {
			return minimalExact, true
		}
		if r < rank {
			continue
		}
		f, img := w.leftNullImage(p)
		if r > rank || !slicesEqual(f, free) {
			rank, free = r, f
			modulus = new(big.Int).SetUint64(p)
			residue = make([]*big.Int, len(img))
			for k, v := range img {
				residue[k] = new(big.Int).SetUint64(v)
			}
		} else {
			crtCombine(residue, modulus, img, p)
		}
		W, ok := reconstructRows(residue, modulus, len(free), n)
		if ok && verifyLeftInvariant(W, free, a, b) {
			return notMinimalExact, true
		}
	}
	return 0, false
}

func slicesEqual(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// leftNullImage returns the free columns and the d×n reduced basis W, with
// W[:, free] = I, of {w : wᵀv = 0 for every stored basis vector v} mod p.
func (w *krylovModP) leftNullImage(p uint64) (free []int, img []uint64) {
	n, r := w.n, len(w.pivots)
	R := append([]uint64(nil), w.basis[:r*n]...)
	for k, piv := range w.pivots {
		for j := range r {
			if j == k {
				continue
			}
			c := R[j*n+piv]
			if c == 0 {
				continue
			}
			for i := range n {
				R[j*n+i] = (R[j*n+i] + (p-c)*R[k*n+i]) % p
			}
		}
	}
	isPivot := make([]bool, n)
	for _, piv := range w.pivots {
		isPivot[piv] = true
	}
	for j := range n {
		if !isPivot[j] {
			free = append(free, j)
		}
	}
	img = make([]uint64, len(free)*n)
	for t, f := range free {
		img[t*n+f] = 1
		for k, piv := range w.pivots {
			if c := R[k*n+f]; c != 0 {
				img[t*n+piv] = p - c
			}
		}
	}
	return free, img
}

// crtCombine replaces residue (mod modulus) by the values congruent to it
// and to img modulo p, and modulus by modulus·p.
func crtCombine(residue []*big.Int, modulus *big.Int, img []uint64, p uint64) {
	pb := new(big.Int).SetUint64(p)
	inv := new(big.Int).ModInverse(new(big.Int).Mod(modulus, pb), pb)
	t := new(big.Int)
	for k, x := range residue {
		t.SetUint64(img[k])
		t.Sub(t, x)
		t.Mul(t, inv)
		t.Mod(t, pb)
		t.Mul(t, modulus)
		x.Add(x, t)
	}
	modulus.Mul(modulus, pb)
}

// reconstructRows maps each residue to the fraction a/b with |a|, b below
// √(modulus/2) congruent to it, by the extended Euclidean algorithm; ok is
// false when some residue has none.
func reconstructRows(residue []*big.Int, modulus *big.Int, d, n int) ([]*big.Rat, bool) {
	bound := new(big.Int).Sqrt(new(big.Int).Rsh(modulus, 1))
	out := make([]*big.Rat, d*n)
	for k, x := range residue {
		r0, r1 := new(big.Int).Set(modulus), new(big.Int).Set(x)
		t0, t1 := new(big.Int), big.NewInt(1)
		q, tmp := new(big.Int), new(big.Int)
		for r1.CmpAbs(bound) > 0 {
			q.Quo(r0, r1)
			tmp.Mul(q, r1)
			r0.Sub(r0, tmp)
			r0, r1 = r1, r0
			tmp.Mul(q, t1)
			t0.Sub(t0, tmp)
			t0, t1 = t1, t0
		}
		if t1.Sign() == 0 || t1.CmpAbs(bound) > 0 {
			return nil, false
		}
		out[k] = new(big.Rat).SetFrac(r1, t1)
	}
	return out, true
}

// verifyLeftInvariant reports whether W (d×n, identity on free) satisfies
// W·B = 0 and W·A = H·W exactly for the integer images A = a and B = b,
// with H = (W·A)[:, free].
func verifyLeftInvariant(W []*big.Rat, free []int, a, b dyadicImage) bool {
	n, m, d := b.rows, b.cols, len(free)
	A, B := a.ints(), b.ints()
	t := new(big.Rat)
	for i := range d {
		for j := range m {
			s := new(big.Rat)
			for k := range n {
				if W[i*n+k].Sign() != 0 && B[k*m+j].Sign() != 0 {
					s.Add(s, t.Mul(W[i*n+k], B[k*m+j]))
				}
			}
			if s.Sign() != 0 {
				return false
			}
		}
	}
	WA := make([]*big.Rat, d*n)
	for i := range d {
		for j := range n {
			s := new(big.Rat)
			for k := range n {
				if W[i*n+k].Sign() != 0 && A[k*n+j].Sign() != 0 {
					s.Add(s, t.Mul(W[i*n+k], A[k*n+j]))
				}
			}
			WA[i*n+j] = s
		}
	}
	for i := range d {
		for j := range n {
			s := new(big.Rat)
			for k, f := range free {
				if WA[i*n+f].Sign() != 0 && W[k*n+j].Sign() != 0 {
					s.Add(s, t.Mul(WA[i*n+f], W[k*n+j]))
				}
			}
			if s.Cmp(WA[i*n+j]) != 0 {
				return false
			}
		}
	}
	return true
}

// ints returns the integer entries ±mant·2^shift as rationals.
func (d dyadicImage) ints() []*big.Rat {
	out := make([]*big.Rat, len(d.mant))
	for k, m := range d.mant {
		v := new(big.Int).SetUint64(m)
		v.Lsh(v, uint(d.shift[k]))
		if d.neg[k] {
			v.Neg(v)
		}
		out[k] = new(big.Rat).SetInt(v)
	}
	return out
}
