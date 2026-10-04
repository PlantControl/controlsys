package controlsys

import (
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type ZerosResult struct {
	Zeros []complex128
	Rank  int
}

func (sys *System) Zeros() ([]complex128, error) {
	res, err := sys.ZerosDetail()
	if err != nil {
		return nil, err
	}
	return res.Zeros, nil
}

func (sys *System) ZerosDetail() (*ZerosResult, error) {
	n, m, p := sys.Dims()
	if n == 0 || m == 0 || p == 0 {
		return &ZerosResult{}, nil
	}

	if m == 1 && p == 1 {
		return sisoZeros(sys)
	}
	return mimoZeros(sys)
}

func sisoZeros(sys *System) (*ZerosResult, error) {
	delayFree := sys
	if sys.LFT != nil {
		delayFree = sys.Copy()
		delayFree.LFT = nil
	}
	tfr, err := delayFree.rationalTransferFunction(nil)
	if err != nil {
		return nil, err
	}
	zeros, err := Poly(tfr.TF.Num[0][0]).Roots()
	if err != nil {
		return nil, err
	}
	return &ZerosResult{Zeros: zeros, Rank: len(zeros)}, nil
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
					return nil, ErrSingularTransform
				}
				zeros := eig.Values(nil)
				sortZeros(zeros)
				return &ZerosResult{Zeros: zeros, Rank: n}, nil
			}
		}
	}

	afData, bfData, nu, rank, err := zerosStaircase(sys.A, sys.B, sys.C, sys.D, n, m, p)
	if err != nil {
		return nil, err
	}
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
		return nil, ErrSingularTransform
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

func zerosStaircase(A, B, C, D *mat.Dense, n, m, p int) (afOut, bfOut []float64, nu, rank int, err error) {
	if n == 0 && min(m, p) == 0 {
		return nil, nil, 0, 0, nil
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
		return nil, nil, 0, rank, nil
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
		return nil, nil, 0, rank, nil
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

	return afOut, bfOut, nu, rank, nil
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
