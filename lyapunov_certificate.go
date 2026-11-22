package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

const (
	// lyapunovTau is the relative accuracy a nil Lyap/DLyap result is
	// certified to: the last refinement correction, or 8·n·μ·ε·s for the
	// comparison bound μ, at most this fraction of ‖X‖.
	lyapunovTau = 1e-10
	// lyapunovRefineSteps caps the double-double refinement corrections.
	lyapunovRefineSteps = 8
)

// lyapunovBandK is the band factor K = 4(n+5): an operator whose estimated
// sep(L) is within K·ε·s of zero is singular
// within the rounding of the stored A (ε·s is lyapunovUnit).
func lyapunovBandK(n int) float64 { return 4 * float64(n+5) }

// lyapunovUnit returns ε·s, s = 2‖A‖_F (continuous) or ‖A‖_F² + 1
// (discrete), the size of the Lyapunov (Stein) operator ‖L‖ is measured
// against, multiplied by ε first so that it overflows only with ‖L‖.
func lyapunovUnit(A *mat.Dense, discrete bool) float64 {
	raw := A.RawMatrix()
	f := lyapFrobenius(raw.Data, raw.Rows, raw.Cols, raw.Stride)
	if discrete {
		return eps()*f*f + eps()
	}
	return 2 * eps() * f
}

// lyapFrobenius returns the Frobenius norm of the rows×cols matrix in data (row
// stride ld): a plain sum of squares, rescaled only when that leaves the
// safe range. Inf or NaN when an entry is.
func lyapFrobenius(data []float64, rows, cols, ld int) float64 {
	sum := 0.0
	for i := range rows {
		for _, x := range data[i*ld : i*ld+cols] {
			sum += x * x
		}
	}
	if sum > 0x1p-900 && sum < 0x1p900 {
		return math.Sqrt(sum)
	}
	if math.IsNaN(sum) {
		return sum
	}
	return denseNormScaled(blas64.General{Rows: rows, Cols: cols, Stride: ld, Data: data})
}

// certifyLyapunov certifies, refines in place or rejects the X (in w, n×n
// of stride n) that s.solve computed from Q. It returns nil when X is the
// unique solution of the stored equation to relative accuracy lyapunovTau;
// ErrSingularEquation when the operator is within lyapunovBandK·ε·s of
// singular by any of three lower bounds on ‖L⁻¹‖ (the growth ‖X‖/‖Q‖ of
// the solve, that of each refinement correction, and the Dlacn2 estimate on
// the Schur form); and ErrAccuracyUnverified, with w the refined iterate of
// smallest correction, when refinement stalls above lyapunovTau.
//
// The tiers are, in order: the growth witness; the rigorous comparison
// bound μ ≥ ‖L_T⁻¹‖, which certifies X unrefined when 8·n·μ·ε·s ≤ τ and
// makes the estimator unnecessary when n·μ·ε·s ≤ 1/(2K); the Dlacn2 band
// test; refinement with residuals of the stored A and Q in double-double.
// A non-finite X is left to the caller's overflow check.
//
// With scale d (the diagonal of D, nil for I), A, Q and X are those of the
// balanced equation Ã = D⁻¹·A·D, Q̃ = D⁻¹·Q·D⁻¹, X̃ = D⁻¹·X·D⁻¹, and the
// accuracy is judged in the original norm of X = D·X̃·D: the comparison
// bound by γ = max(d)²·‖X̃‖_F/‖X‖_F, refinement by ‖D·d̃·D‖_F/‖X‖_F. The
// residuals of the balanced data are those of the stored data scaled
// exactly; the growth witnesses stay in balanced coordinates.
func certifyLyapunov(s *schurLyapunov, A, Q *mat.Dense, w, d []float64) error {
	n := s.n
	xn := lyapFrobenius(w, n, n, n)
	if !(xn <= math.MaxFloat64) {
		return nil
	}
	u := lyapunovUnit(A, s.discrete)
	k := lyapunovBandK(n)
	band := 1 / (k * u)

	singular := lyapunovSingular(k)
	qRaw := Q.RawMatrix()
	qn := lyapFrobenius(qRaw.Data, n, n, qRaw.Stride)
	growth := 0.0
	if qn > 0 {
		growth = xn / qn
	}
	if !(growth < band) {
		return singular
	}
	nf := float64(n)
	gamma := 1.0
	if d != nil && xn > 0 {
		dmax := 0.0
		for _, v := range d {
			dmax = math.Max(dmax, v)
		}
		gamma = dmax * dmax * xn / weightedNorm(w, n, d)
	}
	mu, diag := lyapunovComparisonBound(s.t, n, s.discrete, s.ws)
	if 8*nf*mu*u*gamma <= lyapunovTau {
		return nil
	}
	// The two-sided bound √(μ·μ₁) ≥ ‖L_T⁻¹‖₂, μ₁ the comparison bound of the
	// adjoint (‖·‖₁); formed first from order 24, where it is cheap against
	// the residual, and only when needed below it.
	mu1 := -1.0
	twoSided := func() bool {
		if mu1 < 0 {
			mu1 = lyapunovAdjointBound(s.t, n, s.discrete)
		}
		return 128*math.Sqrt(mu*mu1)*u*gamma <= lyapunovTau
	}
	if n >= 24 && twoSided() {
		return nil
	}
	skip := nf*mu*u <= 1/(2*k)
	if !skip {
		// Entering the last tier: μ₁ is formed here, and the two-sided bound
		// may still certify X or put the operator outside the band.
		if twoSided() {
			return nil
		}
		skip = math.Sqrt(mu*mu1)*u*2*k <= 1
	}
	if skip {
		var r []float64
		if d == nil {
			// Residual-only exit on the stored form: the error of X is at
			// most bound·‖r‖_F, r the double-double residual.
			r = make([]float64, n*n)
			lyapunovResidual(s, A, Q, w, r)
			rn := scaledNorm2(r)
			if nf*mu*rn <= lyapunovTau/2*xn {
				return nil
			}
			// Below order 24 μ₁ is formed only when the two-sided bounds
			// can still certify: n·μ·‖r‖ within 64 of the residual bar.
			if mu1 >= 0 || nf*mu*rn <= 64*(lyapunovTau/2)*xn {
				if twoSided() || math.Sqrt(mu*mu1)*rn <= lyapunovTau/2*xn {
					return nil
				}
			}
		} else if twoSided() {
			return nil
		}
		return refineLyapunov(s, A, Q, w, d, band, singular, r)
	}
	// The last tier: μ certifies neither X nor the distance to singular.
	// Witnesses can only prove Singular (Dlacn2, a lower bound in exact
	// arithmetic that can underestimate by orders, is a cheap exit); a nil
	// needs the exact 1-norm of the computed inverse, formed only up to
	// order lyapunovExactCap.
	if !(diag < band) {
		return singular
	}
	o := newLyapunovOperator(s)
	if !(math.Max(growth, o.inverseNormEstimate()) < band) {
		return singular
	}
	if n > lyapunovExactCap {
		err := refineLyapunov(s, A, Q, w, d, band, singular, nil)
		if err == nil {
			err = lyapunovUncapped(n)
		}
		return err
	}
	if e, ok := o.inverseNorm1(band); !ok || !(e < band) {
		return singular
	}
	return refineLyapunov(s, A, Q, w, d, band, singular, nil)
}

// lyapunovExactCap is the largest order whose last-tier certificate forms
// the exact 1-norm of the computed inverse (n(n+1)/2 triangular solves);
// above it that tier returns ErrAccuracyUnverified, never nil.
const lyapunovExactCap = 16

// lyapunovUncapped is the last-tier error above lyapunovExactCap.
type lyapunovUncapped int

func (n lyapunovUncapped) Error() string {
	return fmt.Sprintf("order %d above %d: distance to singular not certified: %v", int(n), lyapunovExactCap, ErrAccuracyUnverified)
}

func (n lyapunovUncapped) Unwrap() error { return ErrAccuracyUnverified }

// lyapunovResidual writes into r the double-double residual of the stored
// equation at w (refineLyapunov's first-step residual).
func lyapunovResidual(s *schurLyapunov, A, Q *mat.Dense, w, r []float64) {
	var ax []dd
	if s.discrete {
		ax = make([]dd, s.n*s.n)
	}
	qRaw := Q.RawMatrix()
	lyapResidualDot2(A, w, nil, 0, qRaw.Data, qRaw.Stride, s.discrete, ax, r)
}

// lyapunovAdjointBound returns μ₁, the comparison bound on J·Tᵀ·J (J the
// reversal), the quasi-triangular form of the adjoint operator: μ₁ ≥ ‖L_T⁻¹‖₁.
func lyapunovAdjointBound(t []float64, n int, discrete bool) float64 {
	tt := make([]float64, n*n)
	for i := range n {
		for j := range n {
			tt[i*n+j] = t[(n-1-j)*n+(n-1-i)]
		}
	}
	mu, _ := lyapunovComparisonBound(tt, n, discrete, nil)
	return mu
}

// lyapunovEigU returns the smallest eigenvalue-pair distance |λᵢ+λⱼ|
// (discrete |λᵢλⱼ−1|) of the quasi-triangular t over unit.
func lyapunovEigU(t []float64, n int, discrete bool, unit float64) float64 {
	ev := make([]complex128, 0, n)
	for i := 0; i < n; {
		if i+1 < n && t[(i+1)*n+i] != 0 {
			a, b, c, d := t[i*n+i], t[i*n+i+1], t[(i+1)*n+i], t[(i+1)*n+i+1]
			tr, det := a+d, a*d-b*c
			disc := tr*tr/4 - det
			if disc < 0 {
				im := math.Sqrt(-disc)
				ev = append(ev, complex(tr/2, im), complex(tr/2, -im))
			} else {
				sq := math.Sqrt(disc)
				ev = append(ev, complex(tr/2+sq, 0), complex(tr/2-sq, 0))
			}
			i += 2
		} else {
			ev = append(ev, complex(t[i*n+i], 0))
			i++
		}
	}
	m := math.Inf(1)
	for i := range ev {
		for j := i; j < len(ev); j++ {
			v := ev[i] + ev[j]
			if discrete {
				v = ev[i]*ev[j] - 1
			}
			if a := math.Hypot(real(v), imag(v)); a < m {
				m = a
			}
		}
	}
	return m / unit
}

// weightedNorm returns ‖D·V·D‖_F of the n×n v, D = diag(d) (d nil for I).
func weightedNorm(v []float64, n int, d []float64) float64 {
	if d == nil {
		return scaledNorm2(v[:n*n])
	}
	f := 0.0
	for i := range n {
		for j := range n {
			f = math.Hypot(f, d[i]*v[i*n+j]*d[j])
		}
	}
	return f
}

// lyapunovSingular is the band error, formed lazily so that certified
// solves do not allocate it.
type lyapunovSingular float64

func (k lyapunovSingular) Error() string {
	return fmt.Sprintf("operator within %g·ε·‖L‖ of singular: %v", float64(k), ErrSingularEquation)
}

func (k lyapunovSingular) Unwrap() error { return ErrSingularEquation }

// refineLyapunov refines X in w by corrections solved on the Schur form from
// double-double residuals of the stored A and Q.
// A correction growth ‖d‖/‖r‖ of band or more returns singular.
// r0, when not nil, is the residual of w already formed for the first step.
func refineLyapunov(s *schurLyapunov, A, Q *mat.Dense, w, scale []float64, band float64, singular error, r0 []float64) error {
	n := s.n
	nn := n * n
	r, d := r0, make([]float64, nn)
	if r == nil {
		r = make([]float64, nn)
	}
	var ax []dd
	if s.discrete {
		ax = make([]dd, nn)
	}
	var best []float64
	bestCorr := math.Inf(1)
	prev := math.Inf(1)
	qRaw := Q.RawMatrix()
	for step := range lyapunovRefineSteps {
		if step > 0 || r0 == nil {
			lyapResidualDot2(A, w, nil, 0, qRaw.Data, qRaw.Stride, s.discrete, ax, r)
		}
		if err := s.solve(r, n, d); err != nil {
			return singular
		}
		dn := scaledNorm2(d)
		if dn == 0 {
			return nil
		}
		if !(dn/scaledNorm2(r) < band) {
			return singular
		}
		corr := dn / scaledNorm2(w[:nn])
		if scale != nil {
			corr = weightedNorm(d, n, scale) / weightedNorm(w, n, scale)
		}
		if corr <= lyapunovTau {
			for i, v := range d {
				w[i] += v
			}
			return nil
		}
		if !(corr < prev/2) {
			if best != nil && bestCorr < corr {
				copy(w[:nn], best)
			}
			return fmt.Errorf("refinement stalled at relative correction %.2g: %w", math.Min(corr, bestCorr), ErrAccuracyUnverified)
		}
		if corr < bestCorr {
			if best == nil {
				best = make([]float64, nn)
			}
			copy(best, w[:nn])
			bestCorr = corr
		}
		prev = corr
		for i, v := range d {
			w[i] += v
		}
	}
	copy(w[:nn], best)
	return fmt.Errorf("refinement stopped at relative correction %.2g: %w", bestCorr, ErrAccuracyUnverified)
}

// lyapunovComparisonBound returns μ ≥ ‖L_T⁻¹‖ in the max-entry norm, L_T the
// operator X ↦ T·X + X·Tᵀ (or T·X·Tᵀ − X) of the quasi-triangular t: the
// solution, with every right-hand block bounded by 1, of the block
// comparison system whose diagonal entries are the ∞-norm inverse of each
// diagonal block-pair operator and whose couplings are the ∞-norms of the
// off-diagonal blocks of t (Higham, Accuracy and Stability of Numerical
// Algorithms, §8.3). +Inf when a diagonal block-pair operator is singular.
// diag is the largest of those inverse norms, a lower bound on ‖L_T⁻¹‖∞.
func lyapunovComparisonBound(t []float64, n int, discrete bool, ws *LyapunovWorkspace) (mu, diag float64) {
	const small = 6
	var startBuf [16]int
	var smallBuf [4 * small * small]float64
	start := startBuf[:0]
	if n > len(startBuf) {
		start = make([]int, 0, n)
	}
	for i := 0; i < n; {
		start = append(start, i)
		if i+1 < n && t[(i+1)*n+i] != 0 {
			i += 2
		} else {
			i++
		}
	}
	nb := len(start)
	size := func(k int) int {
		if k+1 < nb {
			return start[k+1] - start[k]
		}
		return n - start[k]
	}
	var buf []float64
	if nb <= small {
		buf = smallBuf[:4*nb*nb]
	} else {
		buf = reuseSlice(&ws, 4*nb*nb, func(w *LyapunovWorkspace) *[]float64 { return &w.cmp })
	}
	nrm, y, z, dinv := buf[:nb*nb], buf[nb*nb:2*nb*nb], buf[2*nb*nb:3*nb*nb], buf[3*nb*nb:]
	// The block-pair operators of (k,l) and (l,k) are similar under
	// X ↦ Xᵀ, a permutation, so their inverse ∞-norms agree.
	for k := range nb {
		for l := k; l < nb; l++ {
			v := blockPairInverseNorm(t, n, start[k], size(k), start[l], size(l), discrete)
			dinv[k*nb+l], dinv[l*nb+k] = v, v
			if !(v <= diag) {
				diag = v
			}
		}
	}
	for k := range nb {
		k0, k1 := start[k], start[k]+size(k)
		for m := k; m < nb; m++ {
			m0, m1 := start[m], start[m]+size(m)
			v := 0.0
			for i := k0; i < k1; i++ {
				row := 0.0
				for _, x := range t[i*n+m0 : i*n+m1] {
					row += math.Abs(x)
				}
				if row > v {
					v = row
				}
			}
			nrm[k*nb+m] = v
		}
	}
	// y is symmetric (the (k,l) and (l,k) equations coincide under the
	// transpose similarity), so only k ≤ l is solved, with row l of y
	// standing for column l and contiguous dot products.
	for l := nb - 1; l >= 0; l-- {
		yl := y[l*nb : l*nb+nb]
		nl := nrm[l*nb : l*nb+nb]
		if discrete {
			zl := z[l*nb : l*nb+nb]
			for k := range nb {
				zl[k] = dotUnrolled(nl[l+1:], y[k*nb+l+1:k*nb+nb])
			}
			for m := l + 1; m < nb; m++ {
				zl[m] += nl[l] * yl[m]
			}
			for k := l; k >= 0; k-- {
				sum := 1 + dotUnrolled(nrm[k*nb+k+1:k*nb+nb], zl[k+1:]) + nrm[k*nb+k]*zl[k]
				v := dinv[k*nb+l] * sum
				if !(v <= math.MaxFloat64) {
					return math.Inf(1), diag
				}
				y[k*nb+l], y[l*nb+k] = v, v
				zl[k] += nl[l] * v
				if v > mu {
					mu = v
				}
			}
			continue
		}
		for k := l; k >= 0; k-- {
			sum := 1 + dotUnrolled(nrm[k*nb+k+1:k*nb+nb], yl[k+1:]) + dotUnrolled(nl[l+1:], y[k*nb+l+1:k*nb+nb])
			v := dinv[k*nb+l] * sum
			if !(v <= math.MaxFloat64) {
				return math.Inf(1), diag
			}
			y[k*nb+l], y[l*nb+k] = v, v
			if v > mu {
				mu = v
			}
		}
	}
	return mu, diag
}

// blockPairInverseNorm returns ‖M⁻¹‖∞ for M the matrix of X ↦ T_kk·X + X·T_llᵀ
// (or T_kk·X·T_llᵀ − X) on the da×db block X, +Inf when M is singular.
func blockPairInverseNorm(t []float64, n, ka, da, la, db int, discrete bool) float64 {
	if da == 1 && db == 1 {
		a, b := t[ka*n+ka], t[la*n+la]
		den := a + b
		if discrete {
			den = a*b - 1
		}
		return 1 / math.Abs(den)
	}
	N := da * db
	if N == 2 {
		var m [4]float64
		blockPairMatrix(m[:], t, n, ka, da, la, db, discrete)
		det := m[0]*m[3] - m[1]*m[2]
		if det == 0 {
			return math.Inf(1)
		}
		return math.Max(math.Abs(m[3])+math.Abs(m[1]), math.Abs(m[2])+math.Abs(m[0])) / math.Abs(det)
	}
	var m [16]float64
	blockPairMatrix(m[:], t, n, ka, da, la, db, discrete)
	return inverseNormInf4(&m)
}

// inverseNormInf4 returns ‖M⁻¹‖∞ of the row-major 4×4 m (overwritten) by
// Gauss–Jordan elimination with partial pivoting, +Inf when singular.
func inverseNormInf4(m *[16]float64) float64 {
	const N = 4
	var inv [16]float64
	for i := range N {
		inv[i*N+i] = 1
	}
	for c := range N {
		piv, best := c, math.Abs(m[c*N+c])
		for r := c + 1; r < N; r++ {
			if v := math.Abs(m[r*N+c]); v > best {
				piv, best = r, v
			}
		}
		if best == 0 {
			return math.Inf(1)
		}
		if piv != c {
			for j := range N {
				m[c*N+j], m[piv*N+j] = m[piv*N+j], m[c*N+j]
				inv[c*N+j], inv[piv*N+j] = inv[piv*N+j], inv[c*N+j]
			}
		}
		d := 1 / m[c*N+c]
		for j := range N {
			m[c*N+j] *= d
			inv[c*N+j] *= d
		}
		for r := range N {
			if r == c || m[r*N+c] == 0 {
				continue
			}
			f := m[r*N+c]
			for j := range N {
				m[r*N+j] -= f * m[c*N+j]
				inv[r*N+j] -= f * inv[c*N+j]
			}
		}
	}
	norm := 0.0
	for i := range N {
		s := 0.0
		for j := range N {
			s += math.Abs(inv[i*N+j])
		}
		norm = math.Max(norm, s)
	}
	return norm
}

// blockPairMatrix writes into m the (da·db)² row-major matrix of
// X ↦ T_kk·X + X·T_llᵀ (or T_kk·X·T_llᵀ − X) on the da×db block X.
func blockPairMatrix(m []float64, t []float64, n, ka, da, la, db int, discrete bool) {
	if da == 2 && db == 2 {
		blockPairMatrix22((*[16]float64)(m[:16]), t, n, ka, la, discrete)
		return
	}
	N := da * db
	for i := range da {
		for j := range db {
			row := i*db + j
			for p := range da {
				for q := range db {
					a, b := t[(ka+i)*n+ka+p], t[(la+j)*n+la+q]
					var v float64
					if discrete {
						v = a * b
						if row == p*db+q {
							v--
						}
					} else {
						if j == q {
							v += a
						}
						if i == p {
							v += b
						}
					}
					m[row*N+p*db+q] = v
				}
			}
		}
	}
}

// blockPairMatrix22 is blockPairMatrix for two 2×2 blocks, written out
// (the same operations, so the same values).
func blockPairMatrix22(m *[16]float64, t []float64, n, ka, la int, discrete bool) {
	a := [4]float64{t[ka*n+ka], t[ka*n+ka+1], t[(ka+1)*n+ka], t[(ka+1)*n+ka+1]}
	b := [4]float64{t[la*n+la], t[la*n+la+1], t[(la+1)*n+la], t[(la+1)*n+la+1]}
	if discrete {
		for i := range 2 {
			for j := range 2 {
				for p := range 2 {
					for q := range 2 {
						v := a[i*2+p] * b[j*2+q]
						if i == p && j == q {
							v--
						}
						m[(i*2+j)*4+p*2+q] = v
					}
				}
			}
		}
		return
	}
	*m = [16]float64{
		a[0] + b[0], b[1], a[1], 0,
		b[2], a[0] + b[3], 0, a[1],
		a[2], 0, a[3] + b[0], b[1],
		0, a[2], b[2], a[3] + b[3],
	}
}

// lyapunovOperator is L_T restricted to symmetric matrices in scaled-packed
// coordinates (off-diagonal entries ×√2, so that the packed 2-inner product
// is the Frobenius one and the adjoint is the solve with Tᵀ).
type lyapunovOperator struct {
	s     schurLyapunov
	tt    []float64
	c     []float64
	n, nn int
}

func newLyapunovOperator(s *schurLyapunov) *lyapunovOperator {
	n := s.n
	o := &lyapunovOperator{s: *s, n: n, nn: n * (n + 1) / 2, c: make([]float64, n*n)}
	if s.discrete {
		o.tt = make([]float64, n*n)
		for i := range n {
			for j := range n {
				o.tt[i*n+j] = s.t[(n-1-j)*n+(n-1-i)]
			}
		}
	}
	return o
}

// unpack writes the symmetric matrix of packed x into c, index-reversed
// (for the J·Tᵀ·J form of the adjoint) with flip.
func (o *lyapunovOperator) unpack(x []float64, flip bool) {
	n := o.n
	k := 0
	for i := range n {
		for j := i; j < n; j++ {
			v := x[k]
			if i != j {
				v *= math.Sqrt2 / 2
			}
			a, b := i, j
			if flip {
				a, b = n-1-i, n-1-j
			}
			o.c[a*n+b], o.c[b*n+a] = v, v
			k++
		}
	}
}

func (o *lyapunovOperator) pack(x []float64, flip bool) {
	n := o.n
	k := 0
	for i := range n {
		for j := i; j < n; j++ {
			a, b := i, j
			if flip {
				a, b = n-1-i, n-1-j
			}
			if a > b {
				a, b = b, a
			}
			v := o.c[a*n+b]
			if i != j {
				v *= math.Sqrt2
			}
			x[k] = v
			k++
		}
	}
}

// apply overwrites packed x with L_T⁻¹·x, or L_T⁻ᵀ·x with adj; false when a
// solve reports a singular block or a zero scale.
func (o *lyapunovOperator) apply(x []float64, adj bool) bool {
	n := o.n
	var scale float64
	if o.s.discrete {
		t := o.s.t
		if adj {
			t = o.tt
		}
		o.unpack(x, adj)
		var err error
		if scale, err = solveDiscreteSchurLyap(n, t, n, o.c, n, o.s.ws); err != nil {
			return false
		}
		o.pack(x, adj)
	} else {
		o.unpack(x, false)
		ta, tb := blas.NoTrans, blas.Trans
		if adj {
			ta, tb = blas.Trans, blas.NoTrans
		}
		var ok bool
		if scale, ok = impl.Dtrsyl(ta, tb, 1, n, n, o.s.t, n, o.s.t, n, o.c, n); !ok {
			return false
		}
		for i := range n {
			for j := i + 1; j < n; j++ {
				o.c[i*n+j] = 0.5 * (o.c[i*n+j] + o.c[j*n+i])
			}
		}
		o.pack(x, false)
	}
	if scale == 0 {
		return false
	}
	if scale != 1 {
		for i := range x {
			x[i] /= scale
		}
	}
	return true
}

// inverseNormEstimate returns the Dlacn2 estimate of ‖L_T⁻¹‖₁, a lower bound
// in exact arithmetic, or +Inf when a solve fails or leaves float64 range.
func (o *lyapunovOperator) inverseNormEstimate() float64 {
	N := o.nn
	v, x := make([]float64, N), make([]float64, N)
	isgn := make([]int, N)
	var isave [3]int
	est, kase := 0.0, 0
	for {
		est, kase = impl.Dlacn2(N, v, x, isgn, est, kase, &isave)
		if kase == 0 {
			return est
		}
		if !o.apply(x, kase == 2) || !finiteSlice(x) {
			return math.Inf(1)
		}
	}
}

// unverifiedLyapunov splits a certified Lyapunov solve's error into the hard
// error a consumer returns with a nil value and the ErrAccuracyUnverified
// it returns alongside the value it forms from the uncertified X.
func unverifiedLyapunov(err error) (hard, unv error) {
	if errors.Is(err, ErrAccuracyUnverified) {
		return nil, err
	}
	return err, nil
}

// balanceLyapunov balances the n×n a in place by the classic Parlett–Reinsch
// power-of-two diagonal similarity a ← D⁻¹·a·D (off-diagonal row and column
// 1-norms within a factor of two, iterated to convergence; EISPACK BALANC)
// and returns diag(D), or nil when D is a multiple of I.
func balanceLyapunov(a []float64, n int) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = 1
	}
	for range 200 {
		done := true
		for i := range n {
			c, r := 0.0, 0.0
			for j := range n {
				if j != i {
					c += math.Abs(a[j*n+i])
					r += math.Abs(a[i*n+j])
				}
			}
			if c == 0 || r == 0 || math.IsInf(c+r, 0) {
				continue
			}
			f, sum := 1.0, c+r
			for c < r/2 && f < 0x1p400 {
				f *= 2
				c *= 4
			}
			for c >= r*2 && f > 0x1p-400 {
				f /= 2
				c /= 4
			}
			if (c+r)/f >= 0.95*sum {
				continue
			}
			if nd := d[i] * f; nd > 0x1p900 || nd < 0x1p-900 {
				continue
			}
			d[i] *= f
			for j := range n {
				a[i*n+j] /= f
				a[j*n+i] *= f
			}
			done = false
		}
		if done {
			break
		}
	}
	for _, v := range d[1:] {
		if v != d[0] {
			return d
		}
	}
	return nil
}

// exactlyScaled reports whether v, a power-of-two multiple of the nonzero
// or zero x, kept every bit: finite, and normal unless x is zero.
func exactlyScaled(x, v float64) bool {
	if x == 0 {
		return true
	}
	return !math.IsInf(v, 0) && math.Abs(v) >= 0x1p-1022
}

// balancedLyapunov is the certified solve of the balanced equation,
// Ã = D⁻¹·A·D and Q̃ = D⁻¹·Q·D⁻¹ for D = balanceLyapunov(A), returning
// X = D·X̃·D. ok is false when D is a multiple of I or a scaling of A, Q or
// X̃ overflows or underflows (the caller keeps its raw verdict), and unless
// balancing shrinks ‖A‖_F by 4 or more or, after a certificate verdict
// (t the raw Schur form, nil after the solver's own diagonal test), the
// eigenvalue pairs of t are more than K units ε·s̃ of the balanced matrix
// from singular.
func balancedLyapunov(A, Q *mat.Dense, discrete bool, t []float64) (X *mat.Dense, err error, ok bool) {
	n, _ := A.Dims()
	aRaw, qRaw := A.RawMatrix(), Q.RawMatrix()
	a := make([]float64, n*n)
	copyStrided(a, n, aRaw.Data, aRaw.Stride, n, n)
	d := balanceLyapunov(a, n)
	if d == nil {
		return nil, nil, false
	}
	Ab := mat.NewDense(n, n, a)
	shrink := lyapFrobenius(aRaw.Data, n, n, aRaw.Stride) / lyapFrobenius(a, n, n, n)
	if !(shrink >= 4) && (t == nil || !(lyapunovEigU(t, n, discrete, lyapunovUnit(Ab, discrete)) > lyapunovBandK(n))) {
		return nil, nil, false
	}
	q := make([]float64, n*n)
	for i := range n {
		for j := range n {
			x := aRaw.Data[i*aRaw.Stride+j]
			if !exactlyScaled(x, a[i*n+j]) {
				return nil, nil, false
			}
			x = qRaw.Data[i*qRaw.Stride+j]
			q[i*n+j] = x / d[i] / d[j]
			if !exactlyScaled(x, q[i*n+j]) {
				return nil, nil, false
			}
		}
	}
	Qb := mat.NewDense(n, n, q)
	s, err := newSchurLyapunov(Ab, n, discrete, nil)
	if err != nil {
		return nil, nil, false
	}
	w := make([]float64, n*n)
	if err := s.solve(q, n, w); err != nil {
		return nil, err, true
	}
	if err = certifyLyapunov(&s, Ab, Qb, w, d); err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		return nil, err, true
	}
	x := make([]float64, n*n)
	for i := range n {
		for j := range n {
			x[i*n+j] = d[i] * w[i*n+j] * d[j]
			if !exactlyScaled(w[i*n+j], x[i*n+j]) {
				return nil, nil, false
			}
		}
	}
	return mat.NewDense(n, n, x), err, true
}

// inverseNorm1 returns the exact 1-norm of the computed L_T⁻¹ in
// scaled-packed coordinates, the largest column sum over all n(n+1)/2 unit
// right-hand sides, stopping at the first column sum that reaches stop; ok
// is false when a solve fails or leaves float64 range. A unit right-hand
// side at (p,q), p ≤ q, has its solution in the leading principal block
// through q's diagonal block, so only that block is solved.
func (o *lyapunovOperator) inverseNorm1(stop float64) (norm float64, ok bool) {
	n, t := o.n, o.s.t
	c := o.c
	for q := range n {
		m := q + 1
		if m < n && t[m*n+q] != 0 {
			m++
		}
		for p := 0; p <= q; p++ {
			clear(c[:m*m])
			v := 1.0
			if p != q {
				v = math.Sqrt2 / 2
			}
			c[p*m+q], c[q*m+p] = v, v
			var scale float64
			if o.s.discrete {
				var err error
				if scale, err = solveDiscreteSchurLyap(m, t, n, c, m, o.s.ws); err != nil {
					return norm, false
				}
			} else {
				var solved bool
				if scale, solved = impl.Dtrsyl(blas.NoTrans, blas.Trans, 1, m, m, t, n, t, n, c, m); !solved {
					return norm, false
				}
			}
			if scale == 0 {
				return norm, false
			}
			sum := 0.0
			for i := range m {
				sum += math.Abs(c[i*m+i])
				for j := i + 1; j < m; j++ {
					sum += math.Sqrt2 / 2 * math.Abs(c[i*m+j]+c[j*m+i])
				}
			}
			sum /= scale
			if !(sum <= math.MaxFloat64) {
				return norm, false
			}
			if sum > norm {
				norm = sum
			}
			if !(norm < stop) {
				return norm, true
			}
		}
	}
	return norm, true
}

// dotUnrolled returns Σ a[i]·b[i] over len(a) ≤ len(b), four partial sums.
func dotUnrolled(a, b []float64) float64 {
	var s0, s1, s2, s3 float64
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
