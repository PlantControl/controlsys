package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// LyapunovWorkspace pre-allocates buffers for repeated Lyap/DLyap calls.
// The returned *mat.Dense shares backing storage with the workspace;
// callers who keep results across calls must copy them.
type LyapunovWorkspace struct {
	aData      []float64
	wr, wi     []float64
	vs         []float64
	work       []float64
	tmp, w     []float64
	blocks     []int
	blockStart []int
	buf        []float64
	cmp        []float64
}

// LyapunovOpts configures Lyap and DLyap; a nil *LyapunovOpts allocates
// fresh storage.
type LyapunovOpts struct {
	// Workspace supplies reusable scratch storage. Results may share its storage
	// until the next call that reuses the same workspace.
	Workspace *LyapunovWorkspace
}

func reuseSlice(ws **LyapunovWorkspace, n int, field func(*LyapunovWorkspace) *[]float64) []float64 {
	if *ws != nil {
		p := field(*ws)
		if len(*p) >= n {
			return (*p)[:n]
		}
		*p = make([]float64, n)
		return *p
	}
	return make([]float64, n)
}

func reuseIntSlice(ws *LyapunovWorkspace, n int, field func(*LyapunovWorkspace) *[]int) []int {
	if ws != nil {
		p := field(ws)
		if cap(*p) < n {
			*p = make([]int, 0, n)
		} else {
			*p = (*p)[:0]
		}
		return *p
	}
	return make([]int, 0, n)
}

// NewLyapunovWorkspace preallocates scratch storage for n×n Lyap/DLyap
// problems; it grows on demand. A workspace is not safe for concurrent use.
func NewLyapunovWorkspace(n int) *LyapunovWorkspace {
	return &LyapunovWorkspace{
		aData:      make([]float64, n*n),
		wr:         make([]float64, n),
		wi:         make([]float64, n),
		vs:         make([]float64, n*n),
		work:       make([]float64, n*50),
		tmp:        make([]float64, n*n),
		w:          make([]float64, n*n),
		blocks:     make([]int, 0, n),
		blockStart: make([]int, 0, n),
		buf:        make([]float64, n*2),
	}
}

// Lyap solves the continuous Lyapunov equation
//
//	A*X + X*A' + Q = 0
//
// A is n×n, Q is n×n symmetric. Returns X (n×n symmetric). An Inf or NaN
// entry of A or Q returns ErrInvalidArgument, of X ErrOverflow.
//
// A nil error certifies X as the unique solution to relative accuracy
// 1e-10. As MATLAB lyap errors when the solution does not exist or is not
// unique, Lyap returns ErrSingularEquation when the operator X ↦ A·X + X·Aᵀ
// is singular or within K·ε·2‖A‖_F of singular (K = 4(n+5)), the rounding
// of the stored A, both in the stored scaling and in its diagonally
// balanced form D⁻¹·A·D (power-of-two D; the balanced equation is tried
// only when the stored one fails other than in the solver's own diagonal
// test, and its eigenvalue pairs are apart beyond the band in the balanced
// units): its eigenvalues λᵢ + λⱼ = 0 for some
// pair, or nearly so in a sense rounding can reach. A nonsingular but
// ill-conditioned equation is refined from double-double residuals; when
// refinement stalls above 1e-10, Lyap returns its best X with
// ErrAccuracyUnverified.
func Lyap(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	return lyapunovExported("Lyap", A, Q, opts, false)
}

// lyapunovExported is Lyap or DLyap: lyapunovChecked plus the result check.
func lyapunovExported(op string, A, Q *mat.Dense, opts *LyapunovOpts, discrete bool) (*mat.Dense, error) {
	X, err := lyapunovChecked(op, A, Q, opts, discrete)
	if X == nil {
		return nil, err
	}
	if ferr := finiteDense(op, "X", X, ErrOverflow); ferr != nil {
		return nil, ferr
	}
	return X, err
}

// lyapunovChecked validates A and Q and returns the certified X (see Lyap),
// with an error that is nil, wraps ErrAccuracyUnverified (X non-nil) or any
// other (X nil). It skips the finiteness check of X, for Covar, whose
// continuous feedthrough entries are infinite whatever X holds.
func lyapunovChecked(op string, A, Q *mat.Dense, opts *LyapunovOpts, discrete bool) (*mat.Dense, error) {
	if err := requireFiniteDense(op, "A", A); err != nil {
		return nil, err
	}
	if err := requireFiniteDense(op, "Q", Q); err != nil {
		return nil, err
	}
	X, s, err := solveLyapunov(op, A, Q, opts, discrete)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if err = certifyLyapunov(&s, A, Q, X.RawMatrix().Data, nil); err == nil {
		return X, nil
	}
	unv := errors.Is(err, ErrAccuracyUnverified)
	if !unv && !errors.Is(err, ErrSingularEquation) {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if Xb, errb, ok := balancedLyapunov(A, Q, discrete, s.t); ok {
		if errb == nil || (!unv && errors.Is(errb, ErrAccuracyUnverified)) {
			X, err, unv = Xb, errb, errb != nil
		}
	}
	if err == nil {
		return X, nil
	}
	if unv {
		return X, fmt.Errorf("%s: %w", op, err)
	}
	return nil, fmt.Errorf("%s: %w", op, err)
}

// lyapChecked is Lyap without the result check (see lyapunovChecked).
func lyapChecked(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	return lyapunovChecked("Lyap", A, Q, opts, false)
}

// lyap is the uncertified Lyap solve, for callers that verify X themselves.
func lyap(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	X, _, err := solveLyapunov("Lyap", A, Q, opts, false)
	return X, err
}

// solveLyapunov solves A·X + X·Aᵀ + Q = 0 (discrete: A·X·Aᵀ − X + Q = 0) on
// the real Schur form of A, which it returns for the certificate; its only
// singularity tests are those on the Schur diagonal.
func solveLyapunov(op string, A, Q *mat.Dense, opts *LyapunovOpts, discrete bool) (*mat.Dense, schurLyapunov, error) {
	problem, err := newLyapunovProblem(A, Q, opts)
	if err != nil {
		return nil, schurLyapunov{}, err
	}
	n := problem.n
	if n == 0 {
		return nil, schurLyapunov{}, fmt.Errorf("%s: A has no rows: %w", op, ErrDimensionMismatch)
	}
	ws := problem.ws

	s, err := newSchurLyapunov(A, n, discrete, ws)
	if err != nil {
		return nil, schurLyapunov{}, err
	}
	w := reuseSlice(&ws, n*n, func(w *LyapunovWorkspace) *[]float64 { return &w.w })
	qRaw := Q.RawMatrix()
	if err := s.solve(qRaw.Data, qRaw.Stride, w); err != nil {
		return nil, schurLyapunov{}, err
	}
	return mat.NewDense(n, n, w), s, nil
}

// schurLyapunov holds the real Schur form A = U·T·Uᵀ of a Lyapunov equation,
// so equations sharing A, such as refinement corrections, reuse it.
type schurLyapunov struct {
	n        int
	discrete bool
	t, u     []float64
	tmp      []float64
	ws       *LyapunovWorkspace
}

func newSchurLyapunov(A *mat.Dense, n int, discrete bool, ws *LyapunovWorkspace) (schurLyapunov, error) {
	nn := n * n
	aData := reuseSlice(&ws, nn, func(w *LyapunovWorkspace) *[]float64 { return &w.aData })
	aRaw := A.RawMatrix()
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)

	wr := reuseSlice(&ws, n, func(w *LyapunovWorkspace) *[]float64 { return &w.wr })
	wi := reuseSlice(&ws, n, func(w *LyapunovWorkspace) *[]float64 { return &w.wi })
	vs := reuseSlice(&ws, nn, func(w *LyapunovWorkspace) *[]float64 { return &w.vs })

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, aData, n, wr, wi, vs, n, workQuery[:], -1, nil)
	lwork := int(workQuery[0])
	work := reuseSlice(&ws, lwork, func(w *LyapunovWorkspace) *[]float64 { return &w.work })

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, aData, n, wr, wi, vs, n, work, lwork, nil)
	if !ok {
		return schurLyapunov{}, fmt.Errorf("real Schur form of A did not converge: %w", ErrSchurFailed)
	}
	tmp := reuseSlice(&ws, nn, func(w *LyapunovWorkspace) *[]float64 { return &w.tmp })
	return schurLyapunov{n: n, discrete: discrete, t: aData, u: vs, tmp: tmp, ws: ws}, nil
}

// solve writes into w (n×n, stride n) the symmetric X of A·X + X·Aᵀ + Q = 0
// (continuous) or A·X·Aᵀ − X + Q = 0 (discrete), Q of row stride ldq.
func (s *schurLyapunov) solve(q []float64, ldq int, w []float64) error {
	n, vs, tmp := s.n, s.u, s.tmp
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		-1, blas64.General{Rows: n, Cols: n, Data: q, Stride: ldq},
		blas64.General{Rows: n, Cols: n, Data: vs, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n})

	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: vs, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: w, Stride: n})

	var scale float64
	if s.discrete {
		var err error
		if scale, err = solveDiscreteSchurLyap(n, s.t, n, w, n, s.ws); err != nil {
			return err
		}
	} else {
		var ok bool
		scale, ok = impl.Dtrsyl(blas.NoTrans, blas.Trans, 1,
			n, n, s.t, n, s.t, n, w, n)
		if !ok {
			return fmt.Errorf("A and -A share an eigenvalue: %w", ErrSingularEquation)
		}
	}
	if scale != 1 {
		invScale := 1.0 / scale
		for i := range w[:n*n] {
			w[i] *= invScale
		}
	}

	blas64.Gemm(blas.NoTrans, blas.Trans,
		1, blas64.General{Rows: n, Cols: n, Data: w, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: vs, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n})

	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: vs, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: w, Stride: n})

	symmetrize(w, n, n)
	return nil
}

// DLyap solves the discrete Lyapunov equation
//
//	A*X*A' - X + Q = 0
//
// A is n×n, Q is n×n symmetric. Returns X (n×n symmetric). An Inf or NaN
// entry of A or Q returns ErrInvalidArgument, of X ErrOverflow.
//
// The certificate and errors are those of Lyap, for the Stein operator
// X ↦ A·X·Aᵀ − X (singular when λᵢ·λⱼ = 1 for some eigenvalue pair) and
// the scale ‖A‖_F² + 1 in place of 2‖A‖_F, as MATLAB dlyap errors when the
// solution does not exist or is not unique.
func DLyap(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	return lyapunovExported("DLyap", A, Q, opts, true)
}

// dlyapChecked is DLyap without the result check (see lyapunovChecked).
func dlyapChecked(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	return lyapunovChecked("DLyap", A, Q, opts, true)
}

// dlyap is the uncertified DLyap solve, for callers that verify X themselves.
func dlyap(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	X, _, err := solveLyapunov("DLyap", A, Q, opts, true)
	return X, err
}

// solveDiscreteSchurLyap solves T*Y*T' - Y = C in-place (C overwritten with Y).
// T is n×n upper quasi-triangular (real Schur form). Returns scale factor.
// Only the upper triangle of C is used/written; the lower triangle is ignored.
func solveDiscreteSchurLyap(n int, t []float64, ldt int, c []float64, ldc int, ws *LyapunovWorkspace) (float64, error) {
	if n == 0 {
		return 1, nil
	}

	blocks := reuseIntSlice(ws, n, func(w *LyapunovWorkspace) *[]int { return &w.blocks })
	blockStart := reuseIntSlice(ws, n, func(w *LyapunovWorkspace) *[]int { return &w.blockStart })
	i := 0
	for i < n {
		if i+1 < n && t[(i+1)*ldt+i] != 0 {
			blocks = append(blocks, 2)
			blockStart = append(blockStart, i)
			i += 2
		} else {
			blocks = append(blocks, 1)
			blockStart = append(blockStart, i)
			i++
		}
	}
	nblk := len(blocks)

	scale := 1.0
	smin := math.Abs(t[0])
	for i := 1; i < n; i++ {
		if v := math.Abs(t[i*ldt+i]); v > smin {
			smin = v
		}
	}
	smin = math.Max(smin*eps(), math.SmallestNonzeroFloat64)

	// yRead reads Y[i,k] from the upper triangle of c (Y is symmetric).
	yRead := func(i, k int) float64 {
		if i <= k {
			return c[i*ldc+k]
		}
		return c[k*ldc+i]
	}

	var buf []float64
	if ws != nil {
		if len(ws.buf) < n*2 {
			ws.buf = make([]float64, n*2)
		}
		buf = ws.buf[:n*2]
	} else {
		buf = make([]float64, n*2)
	}

	for lb := nblk - 1; lb >= 0; lb-- {
		j1 := blockStart[lb]
		dl := blocks[lb]
		j2 := j1 + dl

		// Step 1: Subtract contribution from solved columns (q >= j2).
		// C[:j2, j1:j2] -= (T[:j2,:] * Y[:, j2:n] * T[j1:j2, j2:n]')[:j2, :]
		// Compute v[i,jj] = sum_{k>=j2} Y[i,k]*T[j1+jj,k], then C[:j2, j1:j2] -= T[:j2,:]*v
		if j2 < n {
			for i := range n {
				for jj := range dl {
					sum := 0.0
					for k := j2; k < n; k++ {
						sum += yRead(i, k) * t[(j1+jj)*ldt+k]
					}
					buf[i*dl+jj] = sum
				}
			}
			for i := range j2 {
				for jj := range dl {
					sum := 0.0
					for p := range n {
						sum += t[i*ldt+p] * buf[p*dl+jj]
					}
					c[i*ldc+j1+jj] -= sum
				}
			}
		}

		// Step 2: Subtract contribution from lower-triangle rows (p >= j2) within current column.
		// These are Y[p, j1:j2] = Y[j1:j2, p]' for p >= j2, known from symmetry.
		// C[:j2, j1:j2] -= T[:j2, j2:n] * Y[j2:n, j1:j2] * T_ll'
		if j2 < n {
			for i := range j2 {
				for jj := range dl {
					sum := 0.0
					for p := j2; p < n; p++ {
						sum += t[i*ldt+p] * yRead(p, j1+jj)
					}
					buf[i*dl+jj] = sum
				}
			}
			for i := range j2 {
				for jj := range dl {
					val := 0.0
					for pp := range dl {
						val += buf[i*dl+pp] * t[(j1+jj)*ldt+j1+pp]
					}
					c[i*ldc+j1+jj] -= val
				}
			}
		}

		// Step 3: Solve diagonal block T_ll * Y_ll * T_ll' - Y_ll = C_ll
		if dl == 1 {
			denom := t[j1*ldt+j1]*t[j1*ldt+j1] - 1
			if math.Abs(denom) <= smin {
				return scale, ErrSingularEquation
			}
			c[j1*ldc+j1] /= denom
		} else {
			s, err := solveDiscrLyap2(
				t[j1*ldt+j1], t[j1*ldt+j1+1],
				t[(j1+1)*ldt+j1], t[(j1+1)*ldt+j1+1],
				c, j1, ldc)
			if err != nil {
				return scale, err
			}
			scale *= s
		}

		// Step 4: Back-substitute off-diagonal rows (k < l) within column block.
		for kb := lb - 1; kb >= 0; kb-- {
			i1 := blockStart[kb]
			dk := blocks[kb]

			// Update: C[i1:i2, j1:j2] -= (sum_{p=kb+1..lb} T[i1:i2, p_rows] * Y[p_rows, j1:j2]) * T_ll'
			var s [4]float64
			for mb := kb + 1; mb <= lb; mb++ {
				m1 := blockStart[mb]
				dm := blocks[mb]
				for ii := range dk {
					for jj := range dl {
						for mm := range dm {
							s[ii*dl+jj] += t[(i1+ii)*ldt+m1+mm] * c[(m1+mm)*ldc+j1+jj]
						}
					}
				}
			}
			for ii := range dk {
				for jj := range dl {
					val := 0.0
					for pp := range dl {
						val += s[ii*dl+pp] * t[(j1+jj)*ldt+j1+pp]
					}
					c[(i1+ii)*ldc+j1+jj] -= val
				}
			}

			sc, err := solveDiscrKron(dk, dl,
				t, i1, ldt,
				t, j1, ldt,
				c, i1, j1, ldc)
			if err != nil {
				return scale, err
			}
			scale *= sc
		}
	}

	// Copy upper triangle to lower for symmetry
	for i := range n {
		for j := i + 1; j < n; j++ {
			c[j*ldc+i] = c[i*ldc+j]
		}
	}

	return scale, nil
}

// solveDiscrLyap2 solves T*Y*T' - Y = C for 2×2 symmetric T,C,Y.
// C is stored in the matrix c at position (r,r) with stride ldc.
// Exploits symmetry: 3 unknowns (y11, y12, y22) → 3×3 system.
func solveDiscrLyap2(t11, t12, t21, t22 float64, c []float64, r, ldc int) (float64, error) {
	// Balance the block by D = diag(1, d), d a power of two near √|t21/t12|:
	// a non-normal block (|t12| ≫ |t21|) otherwise gives a 3×3 system whose
	// scaling, not its conditioning, trips Dgetc2's perturbed-pivot test.
	d := 1.0
	if t12 != 0 && t21 != 0 {
		d = pow2Near(math.Sqrt(math.Abs(t21)) / math.Sqrt(math.Abs(t12)))
		t12, t21 = t12*d, t21/d
	}
	c11 := c[r*ldc+r]
	c12 := c[r*ldc+r+1] / d
	c22 := c[(r+1)*ldc+r+1] / (d * d)

	// Build 3×3 system M*[y11, y12, y22]' = [c11, c12, c22]'
	// From T*Y*T' - Y = C, with Y symmetric (3 unknowns):
	// Row for (1,1): t11*t11*y11 + 2*t11*t12*y12 + t12*t12*y22 - y11 = c11
	// Row for (1,2): t11*t21*y11 + (t11*t22+t12*t21)*y12 + t12*t22*y22 - y12 = c12
	// Row for (2,2): t21*t21*y11 + 2*t21*t22*y12 + t22*t22*y22 - y22 = c22
	m := [9]float64{
		t11*t11 - 1, 2 * t11 * t12, t12 * t12,
		t11 * t21, t11*t22 + t12*t21 - 1, t12 * t22,
		t21 * t21, 2 * t21 * t22, t22*t22 - 1,
	}
	rhs := [3]float64{c11, c12, c22}

	var ipiv, jpiv [3]int
	if impl.Dgetc2(3, m[:], 3, ipiv[:], jpiv[:]) >= 0 {
		return 1, ErrSingularEquation
	}
	scale := impl.Dgesc2(3, m[:], 3, rhs[:], ipiv[:], jpiv[:])

	c[r*ldc+r] = rhs[0]
	c[r*ldc+r+1] = rhs[1] * d
	c[(r+1)*ldc+r] = rhs[1] * d
	c[(r+1)*ldc+r+1] = rhs[2] * d * d

	if scale == 0 {
		return 1, ErrSingularEquation
	}
	return scale, nil
}

// solveDiscrKron solves T_kk * X * T_ll' - X = C_kl for small blocks.
// A block starts at t[aRow, aRow] with size da, B block at t[bRow, bRow] with size db.
// C/X block at c[aRow, bCol] with size da×db.
// Vectorizes to (da*db)×(da*db) system: (T_ll ⊗ T_kk - I)*vec(X) = vec(C).
func solveDiscrKron(da, db int, at []float64, aRow, ldat int, bt []float64, bRow, ldbt int,
	c []float64, cRow, cCol, ldc int) (float64, error) {

	nn := da * db
	if nn == 1 {
		denom := at[aRow*ldat+aRow]*bt[bRow*ldbt+bRow] - 1
		smin := math.Max(math.Abs(at[aRow*ldat+aRow])*eps(), math.SmallestNonzeroFloat64)
		if math.Abs(denom) <= smin {
			return 1, ErrSingularEquation
		}
		c[cRow*ldc+cCol] /= denom
		return 1, nil
	}

	// Extract small A block (da×da) and B block (db×db)
	var aBlk [4]float64
	for i := range da {
		for j := range da {
			aBlk[i*da+j] = at[(aRow+i)*ldat+aRow+j]
		}
	}
	var bBlk [4]float64
	for i := range db {
		for j := range db {
			bBlk[i*db+j] = bt[(bRow+i)*ldbt+bRow+j]
		}
	}
	// Balance each 2×2 block as solveDiscrLyap2 does, T = D·T̃·D⁻¹ with
	// D = diag(1, d): Y = Da⁻¹·X·Db⁻¹ solves T̃_kk·Y·T̃_llᵀ − Y = Da⁻¹·C·Db⁻¹.
	sa, sb := kronBalance(da, aBlk[:]), kronBalance(db, bBlk[:])

	// Build Kronecker product (B ⊗ A) - I of size nn×nn
	// vec(A*X*B') = (B ⊗ A) * vec(X), so (T_ll ⊗ T_kk - I)*vec(X) = vec(C)
	var kron [16]float64
	for i := range db {
		for j := range db {
			for k := range da {
				for l := range da {
					row := i*da + k
					col := j*da + l
					kron[row*nn+col] = bBlk[i*db+j] * aBlk[k*da+l]
				}
			}
		}
	}
	for i := range nn {
		kron[i*nn+i] -= 1
	}

	// Extract RHS
	var rhs [4]float64
	for i := range da {
		for j := range db {
			rhs[j*da+i] = c[(cRow+i)*ldc+cCol+j] / (sa[i] * sb[j])
		}
	}

	var ipiv, jpiv [4]int
	if impl.Dgetc2(nn, kron[:nn*nn], nn, ipiv[:nn], jpiv[:nn]) >= 0 {
		return 1, ErrSingularEquation
	}
	scale := impl.Dgesc2(nn, kron[:nn*nn], nn, rhs[:nn], ipiv[:nn], jpiv[:nn])

	for i := range da {
		for j := range db {
			c[(cRow+i)*ldc+cCol+j] = rhs[j*da+i] * sa[i] * sb[j]
		}
	}

	if scale == 0 {
		return 1, ErrSingularEquation
	}
	return scale, nil
}

// kronBalance balances the row-major 2×2 block blk (size 2; size 1 is
// left alone) in place by D = diag(1, d), d a power of two near
// √|t21/t12|, and returns D's diagonal.
func kronBalance(size int, blk []float64) [2]float64 {
	s := [2]float64{1, 1}
	if size == 2 && blk[1] != 0 && blk[2] != 0 {
		s[1] = pow2Near(math.Sqrt(math.Abs(blk[2])) / math.Sqrt(math.Abs(blk[1])))
		blk[1], blk[2] = blk[1]*s[1], blk[2]/s[1]
	}
	return s
}
