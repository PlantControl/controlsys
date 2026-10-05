package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// ZerosResult holds the invariant zeros and the normal rank of the transfer
// matrix.
type ZerosResult struct {
	// Zeros are the finite invariant zeros; empty when there are none.
	Zeros []complex128
	// Rank is the normal rank of the system pencil [A-sE B; C D] minus n,
	// which equals the normal rank of the transfer matrix (rank(D) for a
	// static gain).
	Rank int
}

// Zeros returns the finite invariant zeros: the points s where the system
// pencil [A-sE B; C D] loses rank below its normal rank (MATLAB tzero). This
// holds for SISO models too, so zeros cancelled by uncontrollable or
// unobservable poles are kept, matching MATLAB zero on ss models; they coincide
// with the transfer function numerator roots only for minimal realizations. As
// in MATLAB zero, internal delays are set to zero (zero-order Padé) and an
// ill-posed zero-delay loop returns ErrAlgebraicLoop; input/output delays
// contribute no finite zeros and are ignored. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.zero.html and
// https://www.mathworks.com/help/control/ref/dynamicsystem.tzero.html.
func (sys *System) Zeros() ([]complex128, error) {
	res, err := sys.zerosDetail("Zeros")
	if err != nil {
		return nil, err
	}
	return res.Zeros, nil
}

// ZerosDetail returns the invariant zeros together with the normal rank of
// the transfer matrix, as MATLAB [z, nrank] = tzero(sys). A static gain has no
// zeros and rank rank(D); a model with no inputs or outputs has rank 0.
// Non-finite model data returns ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.tzero.html.
func (sys *System) ZerosDetail() (*ZerosResult, error) {
	return sys.zerosDetail("ZerosDetail")
}

func (sys *System) zerosDetail(op string) (*ZerosResult, error) {
	if err := requireFiniteSystem(op, sys); err != nil {
		return nil, err
	}
	sys, err := zeroInternalDelays(sys)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	n, m, p := sys.Dims()
	var res *ZerosResult
	switch {
	case m == 0 || p == 0:
		return &ZerosResult{}, nil
	case n == 0:
		e := &mat.Dense{}
		_, _, _, rank := zerosStaircase(e, e, e, sys.D, 0, m, p)
		return &ZerosResult{Rank: rank}, nil
	case sys.IsDescriptor():
		res, err = descriptorZeros(sys)
	default:
		res, err = mimoZeros(sys)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return res, nil
}

// zeroInternalDelays closes the internal-delay loop of sys with every delay
// set to zero, w = z, and drops external delays.
func zeroInternalDelays(sys *System) (*System, error) {
	if !sys.HasDelay() && sys.LFT == nil {
		return sys, nil
	}
	out, err := sys.ZeroDelayApprox()
	if err != nil {
		return nil, err
	}
	out.Delay, out.InputDelay, out.OutputDelay = nil, nil, nil
	return out, nil
}

// descriptorZeros compresses E = U diag(Σ_r, 0) Vᵀ and treats the n-r
// algebraic states as extra inputs and the n-r algebraic equations as extra
// outputs. The resulting order-r explicit system has the same system pencil up
// to row/column permutation, so its invariant zeros are those of sys, and its
// normal rank exceeds that of sys by n-r.
func descriptorZeros(sys *System) (*ZerosResult, error) {
	n, m, p := sys.Dims()
	var svd mat.SVD
	if !svd.Factorize(sys.E, mat.SVDFull) {
		return nil, fmt.Errorf("SVD of E failed: %w", ErrSingularTransform)
	}
	sv := svd.Values(nil)
	tol := float64(n) * eps() * sv[0]
	r := 0
	for r < n && sv[r] > tol {
		r++
	}
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)

	var At, tmp, Bt, Ct mat.Dense
	tmp.Mul(U.T(), sys.A)
	At.Mul(&tmp, &V)
	Bt.Mul(U.T(), sys.B)
	Ct.Mul(sys.C, &V)
	for i := range r {
		s := 1 / sv[i]
		for j := range n {
			At.Set(i, j, At.At(i, j)*s)
		}
		for j := range m {
			Bt.Set(i, j, Bt.At(i, j)*s)
		}
	}

	q := n - r
	mh, ph := m+q, p+q
	at, bt, ct := At.RawMatrix(), Bt.RawMatrix(), Ct.RawMatrix()
	dh := make([]float64, ph*mh)
	copyBlock(dh, mh, 0, 0, at.Data, at.Stride, r, r, q, q)
	copyBlock(dh, mh, 0, q, bt.Data, bt.Stride, r, 0, q, m)
	copyBlock(dh, mh, q, 0, ct.Data, ct.Stride, 0, r, p, q)
	dRaw := sys.D.RawMatrix()
	copyBlock(dh, mh, q, q, dRaw.Data, dRaw.Stride, 0, 0, p, m)
	Dh := mat.NewDense(ph, mh, dh)

	if r == 0 {
		_, _, _, rank := zerosStaircase(&mat.Dense{}, &mat.Dense{}, &mat.Dense{}, Dh, 0, mh, ph)
		return &ZerosResult{Rank: rank - q}, nil
	}
	ah := make([]float64, r*r)
	bh := make([]float64, r*mh)
	ch := make([]float64, ph*r)
	copyBlock(ah, r, 0, 0, at.Data, at.Stride, 0, 0, r, r)
	copyBlock(bh, mh, 0, 0, at.Data, at.Stride, 0, r, r, q)
	copyBlock(bh, mh, 0, q, bt.Data, bt.Stride, 0, 0, r, m)
	copyBlock(ch, r, 0, 0, at.Data, at.Stride, r, 0, q, r)
	copyBlock(ch, r, q, 0, ct.Data, ct.Stride, 0, 0, p, r)
	aug := &System{A: mat.NewDense(r, r, ah), B: mat.NewDense(r, mh, bh), C: mat.NewDense(ph, r, ch), D: Dh, Dt: sys.Dt}
	res, err := mimoZeros(aug)
	if err != nil {
		return nil, err
	}
	res.Rank -= q
	return res, nil
}

func mimoZeros(sys *System) (*ZerosResult, error) {
	n, m, p := sys.Dims()

	if m == p {
		var luD mat.LU
		luD.Factorize(sys.D)
		if !nearZero(luD.Det()) {
			var DinvC mat.Dense
			if err := luD.SolveTo(&DinvC, false, sys.C); err == nil {
				var BDinvC mat.Dense
				BDinvC.Mul(sys.B, &DinvC)
				var M mat.Dense
				M.Sub(sys.A, &BDinvC)
				var eig mat.Eigen
				if !eig.Factorize(&M, mat.EigenNone) {
					return nil, fmt.Errorf("eigenvalues of A-BD⁻¹C did not converge: %w", ErrSchurFailed)
				}
				zeros := eig.Values(nil)
				sortZeros(zeros)
				return &ZerosResult{Zeros: zeros, Rank: m}, nil
			}
		}
	}

	afData, bfData, nu, rank := zerosStaircase(sys.A, sys.B, sys.C, sys.D, n, m, p)
	if nu == 0 {
		return &ZerosResult{Rank: rank}, nil
	}

	alphar := make([]float64, nu)
	alphai := make([]float64, nu)
	beta := make([]float64, nu)

	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nu,
		afData, nu, bfData, nu,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)

	ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nu,
		afData, nu, bfData, nu,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, lwork)
	if !ok {
		return nil, fmt.Errorf("QZ of the reduced pencil did not converge: %w", ErrSchurFailed)
	}

	betaTol := float64(nu) * eps()
	var zeros []complex128
	for j := range nu {
		if math.Abs(beta[j]) <= betaTol {
			continue
		}
		re := alphar[j] / beta[j]
		im := alphai[j] / beta[j]
		if math.Abs(im) < math.Abs(re)*eps()*100 {
			im = 0
		}
		zeros = append(zeros, complex(re, im))
	}
	sortZeros(zeros)
	return &ZerosResult{Zeros: zeros, Rank: rank}, nil
}

func zerosStaircase(A, B, C, D *mat.Dense, n, m, p int) (afOut, bfOut []float64, nu, rank int) {
	if n == 0 && min(m, p) == 0 {
		return nil, nil, 0, 0
	}

	np := n + p
	mn := m + n
	stride := mn
	if stride == 0 {
		stride = 1
	}

	// Build compound matrix [B A; D C], (n+p) × (m+n), row-major
	bf := make([]float64, np*stride)
	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()
	dRaw := D.RawMatrix()

	if m > 0 {
		copyStrided(bf, stride, bRaw.Data, bRaw.Stride, n, m)
	}
	if n > 0 {
		copyBlock(bf, stride, 0, m, aRaw.Data, aRaw.Stride, 0, 0, n, n)
	}
	if m > 0 {
		copyBlock(bf, stride, n, 0, dRaw.Data, dRaw.Stride, 0, 0, p, m)
	}
	if n > 0 {
		copyBlock(bf, stride, n, m, cRaw.Data, cRaw.Stride, 0, 0, p, n)
	}

	// Tolerance
	thresh := math.Sqrt(float64(np*mn)) * eps()
	tol := thresh

	// svlmax = Frobenius norm of compound matrix
	svlmax := 0.0
	for _, v := range bf {
		svlmax += v * v
	}
	svlmax = math.Sqrt(svlmax)

	// Pass 1: reduce D to full row rank
	infz := make([]int, max(n, 1))
	kronl := make([]int, n+1)
	kronr := make([]int, n+1)

	nu, mu, _, ninfz := zerosStaircasePass(n, m, p, p, 0, svlmax, bf, stride, 0, infz, kronl, tol)
	rank = mu

	numu := nu + mu
	if numu == 0 {
		return nil, nil, 0, rank
	}
	mnu := m + nu

	// Pertranspose (anti-transpose): AF[r][c] = BF[NUMU-1-c][MNU-1-r].
	// Fortran DCOPY copies each row of BF into a reversed column of AF.
	afStride := numu
	if afStride == 0 {
		afStride = 1
	}
	af := make([]float64, mnu*afStride)
	for r := range mnu {
		for c := range numu {
			af[r*afStride+c] = bf[(numu-1-c)*stride+(mnu-1-r)]
		}
	}

	nn := nu
	pp := m
	mm := mu

	if mu != m {
		// Pass 2: reduce D to square invertible
		ro2 := pp - mm
		sigma2 := mm
		nu, mu, _, _ = zerosStaircasePass(nn, mm, pp, ro2, sigma2, svlmax, af, afStride, ninfz, infz, kronr, tol)
	}

	if nu == 0 {
		return nil, nil, 0, rank
	}

	// Pencil extraction
	i1 := nu + mu
	bfPencil := make([]float64, nu*i1)
	for i := 0; i < nu; i++ {
		bfPencil[i*i1+mu+i] = 1
	}

	if rank != 0 && mu > 0 {
		// D' block: MU rows × I1 cols, extract for DTZRZF
		dp := make([]float64, mu*i1)
		copyBlock(dp, i1, 0, 0, af, afStride, nu, 0, mu, i1)

		tauRZ := make([]float64, mu)
		work := make([]float64, 1)
		impl.Dtzrzf(mu, i1, dp, i1, tauRZ, work, -1)
		work = make([]float64, max(int(work[0]), 1))
		impl.Dtzrzf(mu, i1, dp, i1, tauRZ, work, len(work))

		l := i1 - mu
		work2 := make([]float64, 1)
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, af, afStride, work2, -1)
		work2 = make([]float64, max(int(work2[0]), 1))
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, af, afStride, work2, len(work2))
		impl.Dormrz(blas.Right, blas.Trans, nu, i1, mu, l, dp, i1, tauRZ, bfPencil, i1, work2, len(work2))
	}

	// Extract Af = af[0:nu, mu:mu+nu] and Bf = bf[0:nu, mu:mu+nu]
	afOut = make([]float64, nu*nu)
	bfOut = make([]float64, nu*nu)
	copyBlock(afOut, nu, 0, 0, af, afStride, 0, mu, nu, nu)
	copyBlock(bfOut, nu, 0, 0, bfPencil, i1, 0, mu, nu, nu)

	return afOut, bfOut, nu, rank
}

func zerosStaircasePass(n, m, p, ro, sigma int, svlmax float64, abcd []float64, stride int,
	ninfz int, infz, kronl []int, tol float64) (nu, mu, nkrol, ninfzOut int) {

	mu = p
	nu = n
	nkrol = 0
	ninfzOut = ninfz

	iz := 0
	ik := 0
	mm1 := m

	work := make([]float64, max(max(m+n, p+n), 1))

	for mu > 0 {
		ro1 := ro
		mnu := m + nu

		if m > 0 {
			// Step a: compress D rows, merge SIGMA triangular cols with RO new rows
			if sigma != 0 {
				irow := nu
				for i1 := 0; i1 < sigma; i1++ {
					colLen := ro + 1
					beta, t := impl.Dlarfg(colLen, abcd[irow*stride+i1], abcd[(irow+1)*stride+i1:], stride)
					abcd[irow*stride+i1] = beta

					if t != 0 && i1+1 < mnu {
						saved := abcd[irow*stride+i1]
						abcd[irow*stride+i1] = 1
						impl.Dlarf(blas.Left, colLen, mnu-i1-1, abcd[irow*stride+i1:], stride, t, abcd[irow*stride+i1+1:], stride, work)
						abcd[irow*stride+i1] = saved
					}
					irow++
				}
				// Zero lower triangular part
				for r := nu + 1; r < nu+ro+sigma; r++ {
					for c := 0; c < sigma && c < r-nu; c++ {
						abcd[r*stride+c] = 0
					}
				}
			}

			// Step b: rank-revealing QR on remaining D block
			if sigma < m {
				i1 := sigma
				irow := nu + sigma
				rankQR, _, jpvtSub, tauSub := colPivotQR(ro1, m-sigma, abcd[irow*stride+i1:], stride, tol, svlmax)

				// Apply column permutation to rows 0:nu+sigma, cols i1:i1+m-sigma
				impl.Dlapmt(true, nu+sigma, m-sigma, abcd[i1:], stride, jpvtSub)

				if rankQR > 0 {
					// Apply Q^T to C submatrix
					impl.Dormqr(blas.Left, blas.Trans, ro1, nu, rankQR,
						abcd[irow*stride+i1:], stride, tauSub[:rankQR],
						abcd[irow*stride+mm1:], stride, work, len(work))

					// Zero lower triangle of QR result
					if ro1 > 1 {
						for r := 1; r < ro1; r++ {
							for c := 0; c < min(r, rankQR); c++ {
								abcd[(irow+r)*stride+i1+c] = 0
							}
						}
					}
					ro1 -= rankQR
				}
			}
		}

		tau := ro1
		sigma = mu - tau

		// Infinite zero determination
		if iz > 0 {
			infz[iz-1] += ro - tau
			ninfzOut += iz * (ro - tau)
		}
		if ro1 == 0 {
			break
		}
		iz++

		if nu <= 0 {
			mu = sigma
			nu = 0
			ro = 0
		} else {
			// Step c: rank-revealing RQ on C2 block
			c2row := nu + sigma
			mntau := min(tau, nu)
			rank2, _, _, tau2 := rowPivotRQ(tau, nu, abcd[c2row*stride+mm1:], stride, tol, svlmax)

			if rank2 > 0 {
				irow2 := c2row + tau - rank2

				// Apply Q^T from RQ to [A;C1] from the right
				impl.Dormr2(blas.Right, blas.Trans, c2row, nu, rank2,
					abcd[irow2*stride+mm1:], stride, tau2[mntau-rank2:],
					abcd[mm1:], stride, work)

				// Apply Q to [B A] from the left
				impl.Dormr2(blas.Left, blas.NoTrans, nu, mnu, rank2,
					abcd[irow2*stride+mm1:], stride, tau2[mntau-rank2:],
					abcd[0:], stride, work)

				// Zero out
				for r := range rank2 {
					for c := 0; c < nu-rank2; c++ {
						abcd[(irow2+r)*stride+mm1+c] = 0
					}
				}
				if rank2 > 1 {
					for r := 1; r < rank2; r++ {
						for c := 0; c < r; c++ {
							abcd[(irow2+r)*stride+mm1+nu-rank2+c] = 0
						}
					}
				}
			}

			ro = rank2
		}

		// Kronecker indices
		kronl[ik] += tau - ro
		nkrol += kronl[ik]
		ik++

		nu -= ro
		mu = sigma + ro
		if ro == 0 {
			break
		}
	}

	return nu, mu, nkrol, ninfzOut
}

func nearZero(x float64) bool {
	return math.Abs(x) < 1e-14
}

func sortZeros(z []complex128) {
	sort.Slice(z, func(i, j int) bool {
		return cmplx.Abs(z[i]) < cmplx.Abs(z[j])
	})
}
