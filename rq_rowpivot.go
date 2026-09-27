package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
)

func rowPivotRQ(m, n int, a []float64, lda int, rcond, svlmax float64) (rank int, sval [3]float64, jpvt []int, tau []float64) {
	k := min(m, n)
	jpvt = make([]int, m)
	tau = make([]float64, k)
	if k == 0 {
		return
	}

	tolz := math.Sqrt(eps())

	norms := make([]float64, 2*m)
	for i := range m {
		norms[i] = blas64.Nrm2(blas64.Vector{N: n, Data: a[i*lda:], Inc: 1})
		norms[m+i] = norms[i]
		jpvt[i] = i
	}

	// Condition estimator vectors, mirroring Fortran DWORK layout.
	// ismin/ismax start at offset m-1 (0-indexed) and grow backwards.
	dwork := make([]float64, 2*m)
	ismin := m - 1
	ismax := 2*m - 1
	work := make([]float64, max(m, n))

	var smax, smin, smaxpr, sminpr, c1, c2, s1, s2 float64
	var aii float64

	for rank < k {
		mki := m - rank
		nki := n - rank

		pvt := blas64.Iamax(blas64.Vector{N: mki, Data: norms, Inc: 1})

		if pvt != mki-1 {
			blas64.Swap(
				blas64.Vector{N: n, Data: a[pvt*lda:], Inc: 1},
				blas64.Vector{N: n, Data: a[(mki-1)*lda:], Inc: 1},
			)
			jpvt[pvt], jpvt[mki-1] = jpvt[mki-1], jpvt[pvt]
			norms[pvt] = norms[mki-1]
			norms[m+pvt] = norms[m+mki-1]
		}

		row := mki - 1
		col := nki - 1
		tauIdx := k - rank - 1

		if nki > 1 {
			aii = a[row*lda+col]
			beta, t := impl.Dlarfg(nki, a[row*lda+col], a[row*lda:], 1)
			a[row*lda+col] = beta
			tau[tauIdx] = t
		}

		if rank == 0 {
			smax = math.Abs(a[(m-1)*lda+(n-1)])
			smin = smax
			smaxpr = smax
			sminpr = smin
			c1 = 1
			c2 = 1
		} else {
			for j := 0; j < rank; j++ {
				work[j] = a[row*lda+col+1+j]
			}

			sminpr, s1, c1 = impl.Dlaic1(2, rank, dwork[ismin:ismin+rank], smin, work[:rank], a[row*lda+col])
			smaxpr, s2, c2 = impl.Dlaic1(1, rank, dwork[ismax:ismax+rank], smax, work[:rank], a[row*lda+col])
		}

		if svlmax*rcond <= smaxpr && svlmax*rcond <= sminpr && smaxpr*rcond < sminpr {
			if mki > 1 {
				aii = a[row*lda+col]
				a[row*lda+col] = 1
				impl.Dlarf(blas.Right, row, nki, a[row*lda:], 1, tau[tauIdx], a, lda, work[:row])
				a[row*lda+col] = aii

				for j := range row {
					if norms[j] != 0 {
						temp := math.Abs(a[j*lda+col]) / norms[j]
						temp = math.Max((1+temp)*(1-temp), 0)
						temp2 := temp * (norms[j] / norms[m+j]) * (norms[j] / norms[m+j])
						if temp2 <= tolz {
							norms[j] = blas64.Nrm2(blas64.Vector{N: nki - 1, Data: a[j*lda:], Inc: 1})
							norms[m+j] = norms[j]
						} else {
							norms[j] *= math.Sqrt(temp)
						}
					}
				}
			}

			for i := 0; i < rank; i++ {
				dwork[ismin+i] *= s1
				dwork[ismax+i] *= s2
			}

			if rank > 0 {
				ismin--
				ismax--
			}
			dwork[ismin] = c1
			dwork[ismax] = c2
			smin = sminpr
			smax = smaxpr
			rank++
			continue
		}
		break
	}

	mki := m - rank
	nki := n - rank
	row := mki - 1
	col := nki - 1
	tauIdx := k - rank - 1
	if rank < k && nki > 1 {
		blas64.Scal(-a[row*lda+col]*tau[tauIdx], blas64.Vector{N: col, Data: a[row*lda:], Inc: 1})
		a[row*lda+col] = aii
	}

	sval[0] = smax
	sval[1] = smin
	sval[2] = sminpr
	return
}
