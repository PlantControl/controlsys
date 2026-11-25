package controlsys

import (
	"fmt"
	"math"
	"math/big"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

const (
	// hsvTau is the relative accuracy of σ₁ that a nil Hankel singular value
	// result promises.
	hsvTau = 1e-5
	// hsvTier1K scales the float64 rounding n·ε·T² of the Lyapunov gramians
	// in the plain-path gate γₙ.
	hsvTier1K = 16
	// hsvDDK scales the double-double rounding n·2⁻¹⁰⁰·T² of the refined
	// gramians in the resolution floor.
	hsvDDK = 16
	// exactZeroMaxN caps the state dimension of the exact test of a zero
	// transfer function.
	exactZeroMaxN = 32
)

// hsvPlainGate returns γₙ = min(10⁻¹², τ²/(16·n)), the bound on ε·T²/σ̃₁²
// under which Hankel singular values from float64 gramians with both
// Cholesky factors are accurate to the contract.
func hsvPlainGate(n int) float64 {
	return math.Min(hsvDDGate, hsvTau*hsvTau/(hsvTier1K*float64(n)))
}

// hsvPlainResolved reports whether ε·(t/s1)² ≤ γₙ; t = √(‖Wc‖_F·‖Wo‖_F)
// and s1 = σ̃₁ in the same coordinates. A zero or non-finite ratio fails.
func hsvPlainResolved(n int, t, s1 float64) bool {
	r := t / s1
	return s1 > 0 && eps()*r*r <= hsvPlainGate(n)
}

// hsvDDResolved reports whether 16·n·2⁻¹⁰⁰·(t/s1)² ≤ 2τ, that is σ̃₁ from
// double-double gramians is resolved to τ. A zero or non-finite ratio fails.
func hsvDDResolved(n int, t, s1 float64) bool {
	r := t / s1
	return s1 > 0 && hsvDDK*float64(n)*0x1p-100*r*r <= 2*hsvTau
}

// balancedFactors holds the Hankel singular values of (A, B·2⁻ᵉᴮ, C·2⁻ᵉᶜ)
// from refined gramians and their factors Lc, Lo, with the SVD
// Loᵀ·Lc = U·diag(s)·Vᵀ when vectors were requested. t is √(‖Wc‖_F·‖Wo‖_F)
// of the refined gramians in the same coordinates.
type balancedFactors struct {
	n      int
	lc, lo []float64
	s      []float64
	u, vt  []float64
	eB, eC int
	t      float64
}

// resolved reports whether σ̃₁ is resolved to τ against the double-double
// rounding of the gramians.
func (f *balancedFactors) resolved() bool { return hsvDDResolved(f.n, f.t, f.s[0]) }

// newBalancedFactors computes Balreal's Hankel singular values of the
// standard state-space sys without a minimality test: refined gramians,
// their semidefinite factors and the SVD of Loᵀ·Lc, recomputed from
// double-double Cholesky factors when ε·‖Wc‖_F·‖Wo‖_F/σ̃₁² exceeds gate.
func newBalancedFactors(op string, sys *System, gate float64, vectors bool) (*balancedFactors, error) {
	n, m, p := sys.Dims()
	g := newGramianOperands(sys)
	A, At, fb, fc, eB, eC := g.A, g.At, g.fb, g.fc, g.eB, g.eC
	discrete := !sys.IsContinuous()
	gc, err := balrealGramian(A, fb, m, discrete, "controllability", nil)
	if err != nil {
		return nil, err
	}
	gob, err := balrealGramian(At, fc, p, discrete, "observability", &gc.schur)
	if err != nil {
		return nil, err
	}
	lc, err := semidefiniteFactor(gc.x, n)
	if err != nil {
		return nil, fmt.Errorf("%s: controllability gramian factor: %w", op, err)
	}
	lo, err := semidefiniteFactor(gob.x, n)
	if err != nil {
		return nil, fmt.Errorf("%s: observability gramian factor: %w", op, err)
	}

	f := &balancedFactors{n: n, lc: lc, lo: lo, s: make([]float64, n), eB: eB, eC: eC}
	mData := make([]float64, n*n)
	job, ldv := lapack.SVDNone, 1
	if vectors {
		job, ldv = lapack.SVDAll, n
		f.u, f.vt = make([]float64, n*n), make([]float64, n*n)
	}
	svd := func() error {
		wq := make([]float64, 1)
		impl.Dgesvd(job, job, n, n, mData, n, f.s, f.u, ldv, f.vt, ldv, wq, -1)
		work := make([]float64, int(wq[0]))
		if !impl.Dgesvd(job, job, n, n, mData, n, f.s, f.u, ldv, f.vt, ldv, work, len(work)) {
			return fmt.Errorf("%s: SVD of gramian factors did not converge: %w", op, ErrSchurFailed)
		}
		return nil
	}
	gemmTN(lo, lc, mData, n)
	if err := svd(); err != nil {
		return nil, err
	}
	nc, no := scaledNorm2(gc.x), scaledNorm2(gob.x)
	if f.s[0] > 0 && eps()*nc*no/(f.s[0]*f.s[0]) > gate {
		ddc, ok, err := gc.ddCholesky()
		if err != nil {
			return nil, fmt.Errorf("%s: controllability gramian: %w", op, err)
		}
		ddo, ok2, err := gob.ddCholesky()
		if err != nil {
			return nil, fmt.Errorf("%s: observability gramian: %w", op, err)
		}
		if !ok || !ok2 {
			return nil, fmt.Errorf("%s: gramians too unbalanced for accurate Hankel singular values: %w", op, ErrSingularEquation)
		}
		ddProductTN(ddo, ddc, mData, n)
		for i := range lc {
			lc[i] = ddc[i].hi + ddc[i].lo
			lo[i] = ddo[i].hi + ddo[i].lo
		}
		if err := svd(); err != nil {
			return nil, err
		}
	}
	f.t = math.Sqrt(nc) * math.Sqrt(no)
	return f, nil
}

// gramianOperands holds the operands of the refined gramians of a standard
// state-space model: A and Aᵀ (row stride n) and the right-hand factors
// fb = B·2⁻ᵉᴮ (n×m) and fc = Cᵀ·2⁻ᵉᶜ (n×p), scaled as pow2Scaled.
type gramianOperands struct {
	A, At  *mat.Dense
	fb, fc []float64
	eB, eC int
}

func newGramianOperands(sys *System) gramianOperands {
	return newGramianOperandsOf(sys.A, sys.B, sys.C)
}

// newGramianOperandsOf is newGramianOperands for the n×n A, n×m B and p×n C.
func newGramianOperandsOf(Am, Bm, Cm *mat.Dense) gramianOperands {
	n, m := Bm.Dims()
	p, _ := Cm.Dims()
	aRaw := Am.RawMatrix()
	bScaled, eB := pow2Scaled(Bm)
	cScaled, eC := pow2Scaled(Cm)
	bsRaw := bScaled.RawMatrix()
	csRaw := cScaled.RawMatrix()

	ws := make([]float64, 2*n*n+n*m+p*n)
	aData, at, fb, fc := ws[:n*n], ws[n*n:2*n*n], ws[2*n*n:2*n*n+n*m], ws[2*n*n+n*m:]
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}
	copyStrided(fb, m, bsRaw.Data, bsRaw.Stride, n, m)
	for i := range n {
		for j := range p {
			fc[i*p+j] = csRaw.Data[j*csRaw.Stride+i]
		}
	}
	return gramianOperands{A: mat.NewDense(n, n, aData), At: mat.NewDense(n, n, at), fb: fb, fc: fc, eB: eB, eC: eC}
}

// h2DDResolved reports whether h² ≥ 16·n·2⁻¹⁰⁰·terms/(2τ), h resolved to τ
// against the double-double rounding of the observability gramian terms.
// A zero or non-finite ratio fails.
func h2DDResolved(n int, h, terms float64) bool {
	r := math.Sqrt(terms) / h
	return h > 0 && hsvDDK*float64(n)*0x1p-100*r*r <= 2*hsvTau
}

// h2PlainResolved reports whether h² ≥ 16·n·ε·terms/(2τ), the float64
// rounding of the Lyapunov gramian terms within τ of h. A zero or
// non-finite ratio fails.
func h2PlainResolved(n int, h, terms float64) bool {
	r := math.Sqrt(terms) / h
	return h > 0 && hsvTier1K*float64(n)*eps()*r*r <= 2*hsvTau
}

// refinedH2 returns, in the coordinates B·2⁻ᵉᴮ, C·2⁻ᶜ with e = eB+eC,
// h = ‖Lᵀ·B‖_F accumulated in double-double from the double-double Cholesky
// factor L of the refined observability gramian Wo, and terms = ‖B‖_F²·‖Wo‖_F.
func refinedH2(sys *System) (h, terms float64, e int, err error) {
	n, m, p := sys.Dims()
	g := newGramianOperands(sys)
	l, xn, err := refinedFactor(g.At, g.fc, p, !sys.IsContinuous(), "observability")
	if err != nil {
		return 0, 0, 0, err
	}
	var sum dd
	for j := range m {
		for k := range n {
			var t dd
			for i := k; i < n; i++ {
				t = t.add(l[i*n+k].mul(dd{g.fb[i*m+j], 0}))
			}
			sum = sum.add(t.mul(t))
		}
	}
	bn := scaledNorm2(g.fb)
	return math.Sqrt(sum.hi + sum.lo), bn * bn * xn, g.eB + g.eC, nil
}

// refinedFactor returns the double-double lower Cholesky factor L (n×n,
// row-major) of the refined gramian of A and the n×k factor f, and ‖X‖_F of
// that gramian; a stalled refinement returns ErrSingularEquation.
func refinedFactor(A *mat.Dense, f []float64, k int, discrete bool, name string) ([]dd, float64, error) {
	g, err := balrealGramian(A, f, k, discrete, name, nil)
	if err != nil {
		return nil, 0, err
	}
	l, ok, err := g.ddCholesky()
	if err != nil {
		return nil, 0, err
	}
	if !ok {
		return nil, 0, fmt.Errorf("%s gramian refinement stalled: %w", name, ErrSingularEquation)
	}
	return l, scaledNorm2(g.x), nil
}

// exactZeroVerdict is the exact verdict on C·Aᵏ·B = 0 for k = 0..n−1.
type exactZeroVerdict int

const (
	transferNonzero exactZeroVerdict = iota
	transferZero
	transferUndetermined
)

// exactZeroTransfer decides on the stored floats whether the strictly
// proper part C·(sI − A)⁻¹·B of the transfer function is exactly zero,
// that is C·Aᵏ·B = 0 for k = 0..n−1. Each matrix is 2ᵉ times an integer
// matrix, so the test runs in integers: a nonzero residue modulo a prime
// proves a nonzero, otherwise the products are formed exactly. Models with
// more than exactZeroMaxN states are undetermined.
func exactZeroTransfer(A, B, C *mat.Dense) exactZeroVerdict {
	n, _ := B.Dims()
	return exactZeroTransferSteps(A, B, C, n, n)
}

// exactZeroTransferSteps is exactZeroTransfer testing C·Aᵏ·B = 0 for
// k = 0..steps−1, with the state-count cap applied to order instead of the
// dimension of A.
func exactZeroTransferSteps(A, B, C *mat.Dense, order, steps int) exactZeroVerdict {
	n, m := B.Dims()
	p, _ := C.Dims()
	if n == 0 || m == 0 || p == 0 || zeroDense(B) || zeroDense(C) {
		return transferZero
	}
	if order > exactZeroMaxN {
		return transferUndetermined
	}
	trans := m < p
	a, b, c := newDyadicImage(A, trans), newDyadicImage(B, trans), newDyadicImage(C, trans)
	if trans {
		b, c = c, b
	}
	if markovNonzeroModP(a, b, c, modPPrime, steps) {
		return transferNonzero
	}
	return markovExactSteps(a, b, c, steps)
}

func zeroDense(M *mat.Dense) bool {
	raw := M.RawMatrix()
	for i := range raw.Rows {
		for _, x := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			if x != 0 {
				return false
			}
		}
	}
	return true
}

// markovNonzeroModP reports whether some entry of c·aᵏ·b, k < n, is nonzero
// modulo the prime q, which proves it nonzero over ℚ.
func markovNonzeroModP(a, b, c dyadicImage, q uint64, steps int) bool {
	n, m, p := a.rows, b.cols, c.rows
	top := max(a.maxShift, b.maxShift, c.maxShift)
	pow2 := make([]uint64, top+1)
	pow2[0] = 1
	for i := 1; i <= top; i++ {
		pow2[i] = pow2[i-1] * 2 % q
	}
	buf := make([]uint64, n*n+n*m+2*p*n)
	am, bm, w, nw := buf[:n*n], buf[n*n:n*n+n*m], buf[n*n+n*m:n*n+n*m+p*n], buf[n*n+n*m+p*n:]
	a.modP(q, pow2, am)
	b.modP(q, pow2, bm)
	c.modP(q, pow2, w)
	for k := range steps {
		for i := range p {
			for j := range m {
				s := uint64(0)
				for l := range n {
					s = addModP(s, w[i*n+l]*bm[l*m+j]%q, q)
				}
				if s != 0 {
					return true
				}
			}
		}
		if k == steps-1 {
			break
		}
		for i := range p {
			for j := range n {
				s := uint64(0)
				for l := range n {
					s = addModP(s, w[i*n+l]*am[l*n+j]%q, q)
				}
				nw[i*n+j] = s
			}
		}
		w, nw = nw, w
	}
	return false
}

// addModP returns (s + x) mod q for s, x < q < 2⁶³.
func addModP(s, x, q uint64) uint64 {
	s += x
	if s >= q {
		s -= q
	}
	return s
}

// markovExact forms c·aᵏ·b, k < n, in integers. Each row of c·aᵏ is
// divided by the largest power of two common to its entries, which keeps
// every product's zero pattern.
func markovExact(a, b, c dyadicImage) exactZeroVerdict {
	return markovExactSteps(a, b, c, a.rows)
}

// markovExactSteps is markovExact for k < steps.
func markovExactSteps(a, b, c dyadicImage, steps int) exactZeroVerdict {
	n, m, p := a.rows, b.cols, c.rows
	toBig := func(d dyadicImage) []*big.Int {
		out := make([]*big.Int, len(d.mant))
		for k, v := range d.mant {
			x := new(big.Int).SetUint64(v)
			x.Lsh(x, uint(d.shift[k]))
			if d.neg[k] {
				x.Neg(x)
			}
			out[k] = x
		}
		return out
	}
	am, bm := toBig(a), toBig(b)
	w := toBig(c)
	nw := make([]*big.Int, p*n)
	for i := range nw {
		nw[i] = new(big.Int)
	}
	s, t := new(big.Int), new(big.Int)
	for k := range steps {
		live := false
		for i := range p {
			row := w[i*n : i*n+n]
			tz := -1
			for _, x := range row {
				if x.Sign() != 0 {
					if z := int(x.TrailingZeroBits()); tz < 0 || z < tz {
						tz = z
					}
				}
			}
			if tz < 0 {
				continue
			}
			live = true
			if tz > 0 {
				for _, x := range row {
					x.Rsh(x, uint(tz))
				}
			}
			for j := range m {
				s.SetInt64(0)
				for l := range n {
					s.Add(s, t.Mul(row[l], bm[l*m+j]))
				}
				if s.Sign() != 0 {
					return transferNonzero
				}
			}
		}
		if !live {
			return transferZero
		}
		if k == steps-1 {
			break
		}
		for i := range p {
			for j := range n {
				x := nw[i*n+j].SetInt64(0)
				for l := range n {
					x.Add(x, t.Mul(w[i*n+l], am[l*n+j]))
				}
			}
		}
		w, nw = nw, w
	}
	return transferZero
}
