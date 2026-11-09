package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// hinfRiccatiData is an H∞ Riccati equation in the input form of MATLAB
// icare with an indefinite R,
//
//	AᵀX + XA − (XB + S)R⁻¹(BᵀX + Sᵀ) + Q = 0,
//
// R = diag(−γ²I, ·) in the D11 = 0 case and Dᵀ D − γ² diag(I, 0) in
// general. Its Hamiltonian [A − BR⁻¹Sᵀ, −BR⁻¹Bᵀ; −(Q − SR⁻¹Sᵀ), ·] forms
// R⁻¹, which under cheap control or a nearly noise-free measurement
// (D12ᵀD12 or D21D21ᵀ ≈ 1e-16) is ≈ 1e16 in a few directions only: no
// symplectic scaling balances it, the slow eigenvalues and the stable
// subspace lose every digit, and the γ test fails at every γ.
type hinfRiccatiData struct {
	A, B, Q, S, R *mat.Dense
}

// pencilSolve returns the stabilizing positive semidefinite solution from
// the stable deflating subspace of the extended pencil
//
//	[A 0 B; −Q −Aᵀ −S; Sᵀ Bᵀ R] − λ diag(I, I, 0),
//
// as MATLAB icare, which never inverts R; the input columns are first
// compressed by a QR factorization. The basis [U11; U21] is accepted only
// when U11 is well conditioned and U21·U11⁻¹ is symmetric to within
// hinfPencilSymTol, the mark of a Lagrangian, hence accurate, subspace;
// any failure rejects γ with ErrGammaNotAchievable or a numerical sentinel,
// as the Hamiltonian solve does.
func (d hinfRiccatiData) pencilSolve() (*mat.Dense, error) {
	d = d.inputScaled()
	n, k := d.B.Dims()
	nn := 2 * n
	rows := nn + k
	hl := make([]float64, rows*nn)
	jl := make([]float64, rows*nn)
	in := make([]float64, rows*k)
	a, b, q, r := d.A.RawMatrix(), d.B.RawMatrix(), d.Q.RawMatrix(), d.R.RawMatrix()
	s := d.S
	for i := range n {
		for j := range n {
			hl[i*nn+j] = a.Data[i*a.Stride+j]
			hl[(n+i)*nn+j] = -q.Data[i*q.Stride+j]
			hl[(n+i)*nn+n+j] = -a.Data[j*a.Stride+i]
		}
		jl[i*nn+i] = 1
		jl[(n+i)*nn+n+i] = 1
		for j := range k {
			bij := b.Data[i*b.Stride+j]
			in[i*k+j] = bij
			hl[(nn+j)*nn+n+i] = bij
			if s != nil {
				sij := s.At(i, j)
				in[(n+i)*k+j] = -sij
				hl[(nn+j)*nn+i] = sij
			}
		}
	}
	for i := range k {
		copy(in[(nn+i)*k:(nn+i+1)*k], r.Data[i*r.Stride:i*r.Stride+k])
	}
	var work []float64
	if k > 0 {
		tau := make([]float64, k)
		var qrQuery, applyQuery [1]float64
		impl.Dgeqrf(rows, k, nil, k, nil, qrQuery[:], -1)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, k, nil, k, nil, nil, nn, applyQuery[:], -1)
		work = make([]float64, max(int(qrQuery[0]), int(applyQuery[0])))
		impl.Dgeqrf(rows, k, in, k, tau, work, len(work))
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, k, in, k, tau, hl, nn, work, len(work))
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, k, in, k, tau, jl, nn, work, len(work))
	}
	h0, j0 := hl[k*nn:], jl[k*nn:]
	h, j := make([]float64, nn*nn), make([]float64, nn*nn)
	copy(h, h0)
	copy(j, j0)
	ar, ai, beta := make([]float64, nn), make([]float64, nn), make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)
	stable := func(ar, _, b float64) bool { return ar*b < 0 }
	sdim, ok := stableDeflatingQZ(nn, h, j, h0, j0, stable, ar, ai, beta, vs, &work, bwork)
	if !ok {
		return nil, fmt.Errorf("Riccati pencil QZ did not converge: %w", ErrSchurFailed)
	}
	if sdim != n {
		return nil, fmt.Errorf("Riccati pencil has %d stable eigenvalues, not %d: %w", sdim, n, errNoCandidate)
	}
	u11, u21 := make([]float64, n*n), make([]float64, n*n)
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)
	ipiv := make([]int, n)
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, make([]float64, n))
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, fmt.Errorf("Riccati pencil %w", errSingularBasis)
	}
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, make([]float64, 4*n), make([]int, n))
	if !(rcnd > hinfPencilRcond) {
		return nil, fmt.Errorf("Riccati pencil basis rcond %g: %w", rcnd, errSingularBasis)
	}
	x := make([]float64, n*n)
	for i := range n {
		for j := range n {
			x[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, x, n)
	var asym, xmax float64
	for i := range n {
		for j := range n {
			asym = math.Max(asym, math.Abs(x[i*n+j]-x[j*n+i]))
			xmax = math.Max(xmax, math.Abs(x[i*n+j]))
		}
	}
	if !(asym <= hinfPencilSymTol*math.Max(1, xmax)) {
		return nil, fmt.Errorf("Riccati pencil solution asymmetric (%g of %g): %w", asym, xmax, errNoCandidate)
	}
	symmetrize(x, n, n)
	X := mat.NewDense(n, n, x)
	var eig mat.EigenSym
	if !eig.Factorize(mat.NewSymDense(n, x), false) {
		return nil, fmt.Errorf("eigenvalues of the pencil Riccati solution: %w", ErrSchurFailed)
	}
	vals := eig.Values(nil)
	floor := hamiltonianPSDTol * math.Max(1, math.Max(math.Abs(vals[0]), math.Abs(vals[n-1])))
	if vals[0] < -floor {
		return nil, fmt.Errorf("pencil Riccati solution not positive semidefinite: %w", ErrGammaNotAchievable)
	}
	return X, nil
}

// hinfPencilRcond and hinfPencilSymTol bound the reciprocal condition of
// U11 and the relative asymmetry of U21·U11⁻¹ pencilSolve accepts.
const (
	hinfPencilRcond  = 1e-12
	hinfPencilSymTol = 1e-6
)

// atGamma returns d with −γ² added to the first k diagonal entries of R.
func (d hinfRiccatiData) atGamma(gamma float64, k int) hinfRiccatiData {
	R := denseCopy(d.R)
	for i := range k {
		R.Set(i, i, R.At(i, i)-gamma*gamma)
	}
	d.R = R
	return d
}

// newHinfRiccatiData returns the data with B = [B1 B2], S = [0 S2] and
// R = diag(0, R2), the γ² term left to atGamma.
func newHinfRiccatiData(A, B1, B2, Q, S2, R2 *mat.Dense) hinfRiccatiData {
	n, k1 := B1.Dims()
	_, k2 := B2.Dims()
	B := mat.NewDense(n, k1+k2, nil)
	setBlock(B, 0, 0, B1)
	setBlock(B, 0, k1, B2)
	S := mat.NewDense(n, k1+k2, nil)
	setBlock(S, 0, k1, S2)
	R := mat.NewDense(k1+k2, k1+k2, nil)
	setBlock(R, k1, k1, R2)
	return hinfRiccatiData{A: A, B: B, Q: Q, S: S, R: R}
}

// inputScaled returns d for the inputs ũ = diag(t)⁻¹u, t_j = |R_jj|^(−1/2):
// B·diag(t), S·diag(t) and diag(t)·R·diag(t). The solution X is unchanged,
// and a cheap-control block R_jj ≈ 1e-16, which the QR compression would
// round to a singular R (an infinite eigenvalue pair in place of the fast
// modes ±1/√R_jj), becomes ±1.
func (d hinfRiccatiData) inputScaled() hinfRiccatiData {
	_, k := d.B.Dims()
	t := make([]float64, k)
	for j := range t {
		t[j] = 1
		if v := math.Abs(d.R.At(j, j)); v > 0 {
			t[j] = 1 / math.Sqrt(v)
		}
	}
	cols := func(M *mat.Dense) *mat.Dense {
		out := denseCopy(M)
		raw := out.RawMatrix()
		for i := range raw.Rows {
			for j := range k {
				raw.Data[i*raw.Stride+j] *= t[j]
			}
		}
		return out
	}
	d.B, d.S = cols(d.B), cols(d.S)
	R := cols(d.R)
	raw := R.RawMatrix()
	for i := range k {
		for j := range k {
			raw.Data[i*raw.Stride+j] *= t[i]
		}
	}
	d.R = R
	return d
}

// solveWith returns the Hamiltonian solution, or, when that fails and
// pencil is set, the pencilSolve solution of d at gamma (k the size of
// the γ² block of R).
func (w *hamiltonianRiccatiWork) solveWith(d hinfRiccatiData, gamma float64, k int, pencil bool) (*mat.Dense, error) {
	X, err := w.solve()
	if err == nil || !pencil {
		return X, err
	}
	if X, perr := d.atGamma(gamma, k).pencilSolve(); perr == nil {
		return X, nil
	}
	return nil, err
}
