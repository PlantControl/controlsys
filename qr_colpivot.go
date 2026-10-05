package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
)

func colPivotQR(m, n int, a []float64, lda int, rcond, svlmax float64) (rank int, sval [3]float64, jpvt []int, tau []float64) {
	mn := min(m, n)
	if mn == 0 {
		return 0, [3]float64{}, nil, nil
	}

	jpvt = make([]int, n)
	tau = make([]float64, mn)

	tolz := math.Sqrt(eps())

	cnorm := make([]float64, n)
	cnormSave := make([]float64, n)

	colBuf := make([]float64, m)

	for j := range n {
		nrm := blas64.Nrm2(blas64.Vector{N: m, Data: a[j:], Inc: lda})
		cnorm[j] = nrm
		cnormSave[j] = nrm
		jpvt[j] = j
	}

	ismin := make([]float64, n)
	ismax := make([]float64, n)
	work := make([]float64, n)

	var smax, smin, smaxpr, sminpr float64
	var c1, c2, s1, s2 float64
	var aii float64

	rank = 0

	for rank < mn {
		i := rank

		pvt := i + blas64.Iamax(blas64.Vector{N: n - i, Data: cnorm[i:], Inc: 1})

		if pvt != i {
			blas64.Swap(
				blas64.Vector{N: m, Data: a[pvt:], Inc: lda},
				blas64.Vector{N: m, Data: a[i:], Inc: lda},
			)
			jpvt[pvt], jpvt[i] = jpvt[i], jpvt[pvt]
			cnorm[pvt] = cnorm[i]
			cnormSave[pvt] = cnormSave[i]
		}

		if i < m-1 {
			aii = a[i*lda+i]
			for k := 0; k < m-i; k++ {
				colBuf[k] = a[(i+k)*lda+i]
			}
			beta, t := impl.Dlarfg(m-i, colBuf[0], colBuf[1:], 1)
			a[i*lda+i] = beta
			for k := 1; k < m-i; k++ {
				a[(i+k)*lda+i] = colBuf[k]
			}
			tau[i] = t
		} else {
			tau[mn-1] = 0
		}

		if rank == 0 {
			smax = math.Abs(a[0])
			smin = smax
			smaxpr = smax
			sminpr = smin
			c1 = 1
			c2 = 1
		} else {
			for k := 0; k < rank; k++ {
				colBuf[k] = a[k*lda+i]
			}
			sminpr, s1, c1 = impl.Dlaic1(2, rank, ismin[:rank], smin, colBuf[:rank], a[i*lda+i])
			smaxpr, s2, c2 = impl.Dlaic1(1, rank, ismax[:rank], smax, colBuf[:rank], a[i*lda+i])
		}

		if svlmax*rcond <= smaxpr {
			if svlmax*rcond <= sminpr {
				if smaxpr*rcond < sminpr {
					if i < n-1 {
						aii = a[i*lda+i]
						a[i*lda+i] = 1
						for k := 0; k < m-i; k++ {
							colBuf[k] = a[(i+k)*lda+i]
						}
						impl.Dlarf(blas.Left, m-i, n-i-1, colBuf[:m-i], 1, tau[i], a[i*lda+i+1:], lda, work)
						a[i*lda+i] = aii
					}

					for j := i + 1; j < n; j++ {
						if cnorm[j] != 0 {
							temp := math.Abs(a[i*lda+j]) / cnorm[j]
							temp = math.Max((1+temp)*(1-temp), 0)
							temp2 := temp * (cnorm[j] / cnormSave[j]) * (cnorm[j] / cnormSave[j])
							if temp2 <= tolz {
								if m-i-1 > 0 {
									cnorm[j] = blas64.Nrm2(blas64.Vector{N: m - i - 1, Data: a[(i+1)*lda+j:], Inc: lda})
									cnormSave[j] = cnorm[j]
								} else {
									cnorm[j] = 0
									cnormSave[j] = 0
								}
							} else {
								cnorm[j] *= math.Sqrt(temp)
							}
						}
					}

					for k := 0; k < rank; k++ {
						ismin[k] = s1 * ismin[k]
						ismax[k] = s2 * ismax[k]
					}
					ismin[rank] = c1
					ismax[rank] = c2
					smin = sminpr
					smax = smaxpr
					rank++
					continue
				}
			}
		}
		break
	}

	i := rank
	if rank < n {
		if i < m-1 {
			blas64.Scal(-a[i*lda+i]*tau[i], blas64.Vector{N: m - i - 1, Data: a[(i+1)*lda+i:], Inc: lda})
			a[i*lda+i] = aii
		}
	}
	if rank == 0 {
		smin = 0
		sminpr = 0
	}
	sval = [3]float64{smax, smin, sminpr}

	return rank, sval, jpvt, tau
}
