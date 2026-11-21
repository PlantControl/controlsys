package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
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
// sep(L) is within K·ε·s of zero, s the scale of lyapunovScale, is singular
// within the rounding of the stored A.
func lyapunovBandK(n int) float64 { return 4 * float64(n+5) }

// lyapunovScale returns s = 2‖A‖_F (continuous) or ‖A‖_F² + 1 (discrete),
// the size of the Lyapunov (Stein) operator ‖L‖ is measured against.
func lyapunovScale(A *mat.Dense, discrete bool) float64 {
	raw := A.RawMatrix()
	f := 0.0
	for i := range raw.Rows {
		f = math.Hypot(f, scaledNorm2(raw.Data[i*raw.Stride:i*raw.Stride+raw.Cols]))
	}
	if discrete {
		return f*f + 1
	}
	return 2 * f
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
func certifyLyapunov(s *schurLyapunov, A, Q *mat.Dense, w []float64) error {
	n := s.n
	if !finiteSlice(w[:n*n]) {
		return nil
	}
	u := eps() * lyapunovScale(A, s.discrete)
	k := lyapunovBandK(n)
	banded := func(e float64) bool { return !(e*k*u < 1) }

	singular := fmt.Errorf("operator within %g·ε·‖L‖ of singular: %w", k, ErrSingularEquation)
	qn := denseNormScaled(Q.RawMatrix())
	xn := scaledNorm2(w[:n*n])
	growth := 0.0
	if qn > 0 {
		growth = xn / qn
	}
	if banded(growth) {
		return singular
	}
	nf := float64(n)
	mu := lyapunovComparisonBound(s.t, n, s.discrete, s.ws)
	if 8*nf*mu*u <= lyapunovTau {
		return nil
	}
	if !(nf*mu*u <= 1/(2*k)) {
		if banded(math.Max(growth, newLyapunovOperator(s).inverseNormEstimate())) {
			return singular
		}
	}
	return refineLyapunov(s, A, Q, w, banded, singular)
}

// refineLyapunov refines X in w by corrections solved on the Schur form from
// double-double residuals of the stored A and Q.
func refineLyapunov(s *schurLyapunov, A, Q *mat.Dense, w []float64, banded func(float64) bool, singular error) error {
	n := s.n
	nn := n * n
	r, d := make([]float64, nn), make([]float64, nn)
	var ax []dd
	if s.discrete {
		ax = make([]dd, nn)
	}
	var best []float64
	bestCorr := math.Inf(1)
	prev := math.Inf(1)
	qRaw := Q.RawMatrix()
	for range lyapunovRefineSteps {
		lyapResidualDot2(A, w, nil, 0, qRaw.Data, qRaw.Stride, s.discrete, ax, r)
		if err := s.solve(r, n, d); err != nil {
			return singular
		}
		dn := scaledNorm2(d)
		if dn == 0 {
			return nil
		}
		if banded(dn / scaledNorm2(r)) {
			return singular
		}
		corr := dn / scaledNorm2(w[:nn])
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

// lyapunovBlocks returns the starts and sizes of the diagonal blocks of the
// quasi-triangular t.
func lyapunovBlocks(t []float64, n int, ws *LyapunovWorkspace) (start, size []int) {
	start = reuseIntSlice(ws, n, func(w *LyapunovWorkspace) *[]int { return &w.blockStart })
	size = reuseIntSlice(ws, n, func(w *LyapunovWorkspace) *[]int { return &w.blocks })
	for i := 0; i < n; {
		start = append(start, i)
		if i+1 < n && t[(i+1)*n+i] != 0 {
			size = append(size, 2)
			i += 2
		} else {
			size = append(size, 1)
			i++
		}
	}
	return start, size
}

// lyapunovComparisonBound returns μ ≥ ‖L_T⁻¹‖ in the max-entry norm, L_T the
// operator X ↦ T·X + X·Tᵀ (or T·X·Tᵀ − X) of the quasi-triangular t: the
// solution, with every right-hand block bounded by 1, of the block
// comparison system whose diagonal entries are the ∞-norm inverse of each
// diagonal block-pair operator and whose couplings are the ∞-norms of the
// off-diagonal blocks of t (Higham, Accuracy and Stability of Numerical
// Algorithms, §8.3). +Inf when a diagonal block-pair operator is singular.
func lyapunovComparisonBound(t []float64, n int, discrete bool, ws *LyapunovWorkspace) float64 {
	start, size := lyapunovBlocks(t, n, ws)
	nb := len(start)
	buf := reuseSlice(&ws, 3*nb*nb, func(w *LyapunovWorkspace) *[]float64 { return &w.cmp })
	nrm, y, z := buf[:nb*nb], buf[nb*nb:2*nb*nb], buf[2*nb*nb:]
	for k := range nb {
		for m := k; m < nb; m++ {
			v := 0.0
			for i := start[k]; i < start[k]+size[k]; i++ {
				row := 0.0
				for j := start[m]; j < start[m]+size[m]; j++ {
					row += math.Abs(t[i*n+j])
				}
				v = math.Max(v, row)
			}
			nrm[k*nb+m] = v
		}
	}
	mu := 0.0
	for l := nb - 1; l >= 0; l-- {
		if discrete {
			for k := range nb {
				zp := 0.0
				for p := l + 1; p < nb; p++ {
					zp += nrm[l*nb+p] * y[k*nb+p]
				}
				z[k*nb+l] = zp
			}
		}
		for k := nb - 1; k >= 0; k-- {
			sum := 1.0
			if discrete {
				for m := k + 1; m < nb; m++ {
					sum += nrm[k*nb+m] * z[m*nb+l]
				}
				sum += nrm[k*nb+k] * z[k*nb+l]
			} else {
				for m := k + 1; m < nb; m++ {
					sum += nrm[k*nb+m] * y[m*nb+l]
				}
				for m := l + 1; m < nb; m++ {
					sum += nrm[l*nb+m] * y[k*nb+m]
				}
			}
			v := blockPairInverseNorm(t, n, start[k], size[k], start[l], size[l], discrete) * sum
			if math.IsInf(v, 0) || math.IsNaN(v) {
				return math.Inf(1)
			}
			y[k*nb+l] = v
			if discrete {
				z[k*nb+l] += nrm[l*nb+l] * v
			}
			mu = math.Max(mu, v)
		}
	}
	return mu
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
	var m, inv [16]float64
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

// lyapunovOperator is L_T restricted to symmetric matrices in scaled-packed
// coordinates (off-diagonal entries ×√2, so that the packed 2-inner product
// is the Frobenius one and the adjoint is the solve with Tᵀ).
type lyapunovOperator struct {
	s     *schurLyapunov
	tt    []float64
	c     []float64
	n, nn int
}

func newLyapunovOperator(s *schurLyapunov) *lyapunovOperator {
	n := s.n
	o := &lyapunovOperator{s: s, n: n, nn: n * (n + 1) / 2, c: make([]float64, n*n)}
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
