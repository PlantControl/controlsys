package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

var (
	errAxisEigenvalue       = fmt.Errorf("Hamiltonian pencil has eigenvalues on the imaginary axis: %w", ErrNoStabilizing)
	errUnitCircleEigenvalue = fmt.Errorf("symplectic pencil has eigenvalues on the unit circle: %w", ErrNoStabilizing)
)

// exactPencilMaxDim and exactPencilMaxBits cap the order 2n+m of the
// extended pencil and the binary exponent span of its entries that
// boundaryEigenvalueProved takes to exact arithmetic; above either it
// proves nothing. exactPencilCandidate is the relative distance from the
// boundary within which a float64 QZ eigenvalue of the pencil sends it
// there: it only saves the exact work, never certifies.
const (
	exactPencilMaxDim    = 40
	exactPencilMaxBits   = 600
	exactPencilCandidate = 1e-4
)

// certifyBoundary is the exit of Care, Dare, Lqr, Dlqr, Lqe and Kalman:
// certify, then, for an error that is ErrNotConverged, the boundaryProof of
// the Care (continuous) or Dare problem (A, B, Q, R, opts) op solved (for
// Lqe and Kalman the dual one) in its place when it proves one.
func certifyBoundary(op string, unsolvable error, continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult, err error) (*RiccatiResult, error) {
	res, err = certify(op, unsolvable, res, err)
	if errors.Is(err, ErrNotConverged) {
		if perr := boundaryProof(op, continuous, A, B, Q, R, opts); perr != nil {
			return nil, perr
		}
	}
	return res, err
}

// boundaryProof is the error op reports for a solve of Care (continuous)
// or Dare that ended in ErrNotConverged when boundaryEigenvalueProved holds.
func boundaryProof(op string, continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts) error {
	if !boundaryEigenvalueProved(continuous, A, B, Q, R, opts) {
		return nil
	}
	if continuous {
		return fmt.Errorf("%s: %w", op, errAxisEigenvalue)
	}
	return fmt.Errorf("%s: %w", op, errUnitCircleEigenvalue)
}

// boundaryEigenvalueProved reports whether the extended pencil of the
// stored data, s·L − M with (continuous)
//
//	L = diag(E, Eᵀ, 0),  M = [A 0 B; −Q −Aᵀ −S; Sᵀ Bᵀ R]
//
// or (discrete) L = [E 0 0; 0 Aᵀ 0; 0 −Bᵀ 0], M = [A 0 B; −Q Eᵀ −S; Sᵀ 0 R],
// has a finite eigenvalue exactly on the imaginary axis (unit circle). A
// stabilizing X splits the finite spectrum into the closed-loop
// eigenvalues and their mirrors −λ̄ (1/λ̄), none on the boundary, so such an
// eigenvalue proves that no stabilizing solution exists (MATLAB icare and
// idare: "no stabilizing solution"). The proof is exact: det(sL − M) is
// interpolated in integers from the power-of-two scaled entries, checked
// even (palindromic: its reversal a multiple of itself), and, through
// z = (1+s)/(1−s) in the discrete case after an exact test of z = −1,
// written q(s²); a root of q at 0 or, by its Sturm sequence, on (−∞, 0) is
// an eigenvalue on the axis. Q, R and E must be exactly symmetric where the
// equation requires it, det(sL − M) must not vanish identically and, for
// Care, have degree 2n; else, or above the caps, nothing is proved.
func boundaryEigenvalueProved(continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts) bool {
	n, m := B.Dims()
	N := 2*n + m
	if N > exactPencilMaxDim || !exactlySymmetric(Q) || !exactlySymmetric(R) {
		return false
	}
	L, M := extendedPencil(continuous, A, B, Q, R, opts)
	if !boundaryCandidate(continuous, L, M, N) {
		return false
	}
	li, mi, ok := integerPencil(L, M)
	if !ok {
		return false
	}
	p := exactDetPoly(li, mi, N, 2*n)
	if len(p) == 0 || continuous && len(p) != 2*n+1 {
		return false
	}
	if continuous {
		return axisRootProved(p)
	}
	return unitCircleRootProved(p, 2*n)
}

func exactlySymmetric(a *mat.Dense) bool {
	raw := a.RawMatrix()
	for i := range raw.Rows {
		for j := range i {
			if raw.Data[i*raw.Stride+j] != raw.Data[j*raw.Stride+i] {
				return false
			}
		}
	}
	return true
}

// extendedPencil returns L and M of boundaryEigenvalueProved, row-major
// (2n+m)².
func extendedPencil(continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts) (L, M []float64) {
	n, m := B.Dims()
	N := 2*n + m
	L, M = make([]float64, N*N), make([]float64, N*N)
	var E, S *mat.Dense
	if opts != nil {
		E, S = opts.E, opts.S
	}
	e := func(i, j int) float64 {
		if E == nil {
			if i == j {
				return 1
			}
			return 0
		}
		return E.At(i, j)
	}
	for i := range n {
		for j := range n {
			a, q := A.At(i, j), Q.At(i, j)
			L[i*N+j] = e(i, j)
			M[i*N+j] = a
			M[(n+i)*N+j] = -q
			if continuous {
				L[(n+i)*N+n+j] = e(j, i)
				M[(n+i)*N+n+j] = -A.At(j, i)
			} else {
				L[(n+i)*N+n+j] = A.At(j, i)
				M[(n+i)*N+n+j] = e(j, i)
			}
		}
		for j := range m {
			b := B.At(i, j)
			M[i*N+2*n+j] = b
			var s float64
			if S != nil {
				s = S.At(i, j)
			}
			M[(n+i)*N+2*n+j] = -s
			M[(2*n+j)*N+i] = s
			if continuous {
				M[(2*n+j)*N+n+i] = b
			} else {
				L[(2*n+j)*N+n+i] = -b
			}
		}
	}
	for i := range m {
		for j := range m {
			M[(2*n+i)*N+2*n+j] = R.At(i, j)
		}
	}
	return L, M
}

// boundaryCandidate reports whether the float64 QZ eigenvalues α/β of
// (M, L) include one within exactPencilCandidate of the boundary, relative
// to |α| + ν|β| (ν = ‖M‖_F/‖L‖_F) for the axis and to |α| + |β| for the
// circle, or whether QZ fails.
func boundaryCandidate(continuous bool, L, M []float64, N int) bool {
	a, b := append([]float64(nil), M...), append([]float64(nil), L...)
	ar, ai, be := make([]float64, N), make([]float64, N), make([]float64, N)
	var q [1]float64
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, N, a, N, b, N, ar, ai, be, nil, 1, nil, 1, q[:], -1)
	work := make([]float64, int(q[0]))
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, N, a, N, b, N, ar, ai, be, nil, 1, nil, 1, work, len(work)) {
		return true
	}
	var nm, nl float64
	for i := range M {
		nm, nl = math.Hypot(nm, M[i]), math.Hypot(nl, L[i])
	}
	nu := nm / nl
	for i := range N {
		alpha, beta := math.Hypot(ar[i], ai[i]), math.Abs(be[i])
		if continuous {
			if math.Abs(ar[i]) <= exactPencilCandidate*(alpha+nu*beta) {
				return true
			}
		} else if math.Abs(alpha-beta) <= exactPencilCandidate*(alpha+beta) {
			return true
		}
	}
	return false
}

// integerPencil returns L and M scaled by one power of two to integers, or
// false when their nonzero entries span more than exactPencilMaxBits
// binary orders.
func integerPencil(L, M []float64) (li, mi []*big.Int, ok bool) {
	lo, hi := math.MaxInt, math.MinInt
	for _, v := range [][]float64{L, M} {
		for _, x := range v {
			if x == 0 {
				continue
			}
			_, e := math.Frexp(x)
			lo, hi = min(lo, e-53), max(hi, e)
		}
	}
	if lo == math.MaxInt {
		lo, hi = 0, 0
	}
	if hi-lo > exactPencilMaxBits {
		return nil, nil, false
	}
	conv := func(v []float64) []*big.Int {
		out := make([]*big.Int, len(v))
		for i, x := range v {
			f, e := math.Frexp(x)
			z := big.NewInt(int64(f * (1 << 53)))
			out[i] = z.Lsh(z, uint(e-53-lo))
		}
		return out
	}
	return conv(L), conv(M), true
}

// exactDetPoly returns det(sL − M), ascending and trimmed of exactly zero
// leading coefficients, for an integer N×N pencil whose determinant has
// degree at most d, by Bareiss elimination at s = 0..d and Newton
// interpolation.
func exactDetPoly(L, M []*big.Int, N, d int) []*big.Rat {
	y := make([]*big.Rat, d+1)
	P := make([]*big.Int, N*N)
	for i := range P {
		P[i] = new(big.Int)
	}
	var t big.Int
	for k := range y {
		s := big.NewInt(int64(k))
		for i := range P {
			P[i].Sub(t.Mul(s, L[i]), M[i])
		}
		y[k] = new(big.Rat).SetInt(bareissDet(P, N))
	}
	for j := 1; j <= d; j++ {
		for k := d; k >= j; k-- {
			y[k].Sub(y[k], y[k-1])
			y[k].Quo(y[k], big.NewRat(int64(j), 1))
		}
	}
	poly := []*big.Rat{new(big.Rat).Set(y[d])}
	for k := d - 1; k >= 0; k-- {
		next := make([]*big.Rat, len(poly)+1)
		next[0] = new(big.Rat)
		for i := range poly {
			next[i+1] = new(big.Rat).Set(poly[i])
		}
		var u big.Rat
		for i := range poly {
			next[i].Sub(next[i], u.Mul(big.NewRat(int64(k), 1), poly[i]))
		}
		next[0].Add(next[0], y[k])
		poly = next
	}
	for len(poly) > 0 && poly[len(poly)-1].Sign() == 0 {
		poly = poly[:len(poly)-1]
	}
	return poly
}

// axisRootProved reports whether the ascending polynomial p, which must be
// even, has a root on the imaginary axis: p = q(s²) with q(0) = 0 or a
// root of q on (−∞, 0).
func axisRootProved(p []*big.Rat) bool {
	q := make([]*big.Rat, 0, len(p)/2+1)
	for k, c := range p {
		if k%2 == 1 {
			if c.Sign() != 0 {
				return false
			}
			continue
		}
		q = append(q, c)
	}
	return q[0].Sign() == 0 || hasNegativeRoot(q)
}

// unitCircleRootProved reports whether the ascending polynomial p of
// formal degree d, which must be palindromic up to a constant, has a root
// on the unit circle: at z = −1 exactly, or else, through
// P(s) = (1−s)^d·p((1+s)/(1−s)), on the imaginary axis.
func unitCircleRootProved(p []*big.Rat, d int) bool {
	c := make([]*big.Rat, d+1)
	for k := range c {
		c[k] = new(big.Rat)
		if k < len(p) {
			c[k].Set(p[k])
		}
	}
	k0 := len(p) - 1
	kappa := new(big.Rat).Quo(c[d-k0], c[k0])
	var t big.Rat
	for k := range c {
		if t.Mul(kappa, c[k]).Cmp(c[d-k]) != 0 {
			return false
		}
	}
	at := new(big.Rat)
	for k := d; k >= 0; k-- {
		at.Neg(at).Add(at, c[k])
	}
	if at.Sign() == 0 {
		return true
	}
	P := make([]*big.Rat, d+1)
	for i := range P {
		P[i] = new(big.Rat)
	}
	for k, ck := range c {
		if ck.Sign() == 0 {
			continue
		}
		f := polyMulInt(binomialRow(k, 1), binomialRow(d-k, -1))
		for i, v := range f {
			P[i].Add(P[i], t.Mul(ck, new(big.Rat).SetInt(v)))
		}
	}
	if P[d].Sign() == 0 {
		return false
	}
	return axisRootProved(P)
}

// binomialRow returns the ascending coefficients of (1 + sign·s)^k.
func binomialRow(k int, sign int64) []*big.Int {
	row := make([]*big.Int, k+1)
	c := big.NewInt(1)
	for i := range row {
		row[i] = new(big.Int).Set(c)
		if sign < 0 && i%2 == 1 {
			row[i].Neg(row[i])
		}
		c.Mul(c, big.NewInt(int64(k-i)))
		c.Quo(c, big.NewInt(int64(i+1)))
	}
	return row
}

func polyMulInt(a, b []*big.Int) []*big.Int {
	out := make([]*big.Int, len(a)+len(b)-1)
	for i := range out {
		out[i] = new(big.Int)
	}
	var t big.Int
	for i := range a {
		for j := range b {
			out[i+j].Add(out[i+j], t.Mul(a[i], b[j]))
		}
	}
	return out
}

// hasNegativeRoot reports whether the ascending polynomial q with
// q(0) ≠ 0 has a real root on (−∞, 0): at once when q changes sign between
// −∞ and 0, else by the count of its Sturm sequence, kept in integers as a
// primitive pseudo-remainder sequence (each remainder scaled by a positive
// factor, which leaves the sign changes unchanged).
func hasNegativeRoot(q []*big.Rat) bool {
	f := integerDescending(q)
	deg := len(f) - 1
	if deg <= 0 {
		return false
	}
	atNegInf := func(p []*big.Int) int {
		if (len(p)-1)%2 == 1 {
			return -p[0].Sign()
		}
		return p[0].Sign()
	}
	atZero := func(p []*big.Int) int { return p[len(p)-1].Sign() }
	if atNegInf(f) != atZero(f) {
		return true
	}
	df := make([]*big.Int, deg)
	for i := range deg {
		df[i] = new(big.Int).Mul(f[i], big.NewInt(int64(deg-i)))
	}
	seq := [][]*big.Int{f, primitive(df)}
	for {
		r := sturmRem(seq[len(seq)-2], seq[len(seq)-1])
		if len(r) == 0 {
			break
		}
		seq = append(seq, r)
	}
	changes := func(sign func([]*big.Int) int) int {
		k, last := 0, 0
		for _, p := range seq {
			s := sign(p)
			if s == 0 {
				continue
			}
			if last != 0 && s != last {
				k++
			}
			last = s
		}
		return k
	}
	return changes(atNegInf) > changes(atZero)
}

// integerDescending returns the ascending rational q as a descending
// integer polynomial with the same roots and signs, trimmed of exactly
// zero leading coefficients.
func integerDescending(q []*big.Rat) []*big.Int {
	lcm := big.NewInt(1)
	var g big.Int
	for _, c := range q {
		d := c.Denom()
		g.GCD(nil, nil, lcm, d)
		lcm.Mul(lcm, new(big.Int).Quo(d, &g))
	}
	out := make([]*big.Int, 0, len(q))
	for _, c := range slices.Backward(q) {
		v := new(big.Int).Mul(c.Num(), new(big.Int).Quo(lcm, c.Denom()))
		if len(out) == 0 && v.Sign() == 0 {
			continue
		}
		out = append(out, v)
	}
	return primitive(out)
}

// primitive divides the integer polynomial p by the positive gcd of its
// coefficients, in place.
func primitive(p []*big.Int) []*big.Int {
	g := new(big.Int)
	for _, c := range p {
		g.GCD(nil, nil, g, new(big.Int).Abs(c))
	}
	if g.Sign() > 0 && g.Cmp(big.NewInt(1)) != 0 {
		for _, c := range p {
			c.Quo(c, g)
		}
	}
	return p
}

// sturmRem returns a positive multiple of −(a mod b) for descending integer
// a and b, primitive and trimmed of exactly zero leading coefficients.
func sturmRem(a, b []*big.Int) []*big.Int {
	r := make([]*big.Int, len(a))
	for i, x := range a {
		r[i] = new(big.Int).Set(x)
	}
	lb := b[0]
	var t big.Int
	for len(r) >= len(b) {
		f := new(big.Int).Set(r[0])
		for i := range r {
			r[i].Mul(r[i], lb)
		}
		for i := range b {
			r[i].Sub(r[i], t.Mul(f, b[i]))
		}
		r = r[1:]
		if lb.Sign() < 0 {
			for _, x := range r {
				x.Neg(x)
			}
		}
	}
	for len(r) > 0 && r[0].Sign() == 0 {
		r = r[1:]
	}
	for _, x := range r {
		x.Neg(x)
	}
	return primitive(r)
}
