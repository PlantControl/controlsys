package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

type RootLocusResult struct {
	Gains             []float64
	Branches          [][]complex128
	Breakaway         []complex128
	AsymptoteAngles   []float64
	AsymptoteCentroid float64
	DepartureAngles   []float64
	ArrivalAngles     []float64
}

func RootLocus(sys *System, gains []float64) (*RootLocusResult, error) {
	if _, err := newSISOLoopModel(sys, "RootLocus"); err != nil {
		return nil, err
	}
	n, _, _ := sys.Dims()

	dRaw := sys.D.RawMatrix()
	if dRaw.Data[0] != 0 {
		return nil, fmt.Errorf("controlsys: RootLocus requires D=0: %w", ErrDimensionMismatch)
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	zeros, err := sys.Zeros()
	if err != nil {
		return nil, err
	}

	if gains == nil {
		gains = defaultGains()
	} else {
		g := make([]float64, len(gains))
		copy(g, gains)
		gains = g
		sort.Float64s(gains)
	}

	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	bc := make([]float64, n*n)
	for i := range n {
		for j := range n {
			bc[i*n+j] = bRaw.Data[i*bRaw.Stride] * cRaw.Data[j]
		}
	}

	aRaw := sys.A.RawMatrix()
	aFlat := make([]float64, n*n)
	for i := range n {
		copy(aFlat[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}

	allEigs := make([][]complex128, len(gains))
	work := make([]float64, n*n)

	for gi, K := range gains {
		for i := 0; i < n*n; i++ {
			work[i] = aFlat[i] - K*bc[i]
		}
		M := mat.NewDense(n, n, work)
		var eig mat.Eigen
		if !eig.Factorize(M, mat.EigenNone) {
			return nil, ErrSchurFailed
		}
		vals := eig.Values(nil)
		allEigs[gi] = vals
	}

	branches := makeBranches(allEigs, n, len(gains))

	breakaway := computeBreakaway(sys, poles, zeros)

	nPoles := len(poles)
	nZeros := len(zeros)

	var asymAngles []float64
	var asymCentroid float64
	diff := nPoles - nZeros
	if diff > 0 {
		asymAngles = make([]float64, diff)
		for k := range diff {
			asymAngles[k] = float64(2*k+1) * math.Pi / float64(diff)
		}
		sumP := 0.0
		for _, pole := range poles {
			sumP += real(pole)
		}
		sumZ := 0.0
		for _, z := range zeros {
			sumZ += real(z)
		}
		asymCentroid = (sumP - sumZ) / float64(diff)
	}

	departAngles := make([]float64, nPoles)
	for i, pi := range poles {
		sumPoles := 0.0
		for j, pj := range poles {
			if j != i {
				sumPoles += cmplx.Phase(pi - pj)
			}
		}
		sumZeros := 0.0
		for _, zk := range zeros {
			sumZeros += cmplx.Phase(pi - zk)
		}
		departAngles[i] = math.Pi - sumPoles + sumZeros
	}

	arrivalAngles := make([]float64, nZeros)
	for k, zk := range zeros {
		sumZeros := 0.0
		for j, zj := range zeros {
			if j != k {
				sumZeros += cmplx.Phase(zk - zj)
			}
		}
		sumPoles := 0.0
		for _, pi := range poles {
			sumPoles += cmplx.Phase(zk - pi)
		}
		arrivalAngles[k] = math.Pi - sumZeros + sumPoles
	}

	return &RootLocusResult{
		Gains:             gains,
		Branches:          branches,
		Breakaway:         breakaway,
		AsymptoteAngles:   asymAngles,
		AsymptoteCentroid: asymCentroid,
		DepartureAngles:   departAngles,
		ArrivalAngles:     arrivalAngles,
	}, nil
}

func defaultGains() []float64 {
	n := 200
	gains := make([]float64, n+1)
	gains[0] = 0
	for i := range n {
		exp := -6.0 + 12.0*float64(i)/float64(n-1)
		gains[i+1] = math.Pow(10, exp)
	}
	sort.Float64s(gains)
	return gains
}

func makeBranches(allEigs [][]complex128, nStates, nGains int) [][]complex128 {
	if nStates == 0 || nGains == 0 {
		return nil
	}

	ordered := make([][]complex128, nGains)
	ordered[0] = make([]complex128, nStates)
	copy(ordered[0], allEigs[0])

	used := make([]bool, nStates)
	for gi := 1; gi < nGains; gi++ {
		cur := allEigs[gi]
		prev := ordered[gi-1]
		perm := make([]int, nStates)
		for i := range used {
			used[i] = false
		}

		for i := range nStates {
			bestJ := -1
			bestDist := math.Inf(1)
			for j := range nStates {
				if used[j] {
					continue
				}
				d := cmplx.Abs(prev[i] - cur[j])
				if d < bestDist {
					bestDist = d
					bestJ = j
				}
			}
			perm[i] = bestJ
			used[bestJ] = true
		}
		row := make([]complex128, nStates)
		for i := range nStates {
			row[i] = cur[perm[i]]
		}
		ordered[gi] = row
	}

	branches := make([][]complex128, nStates)
	for b := range nStates {
		branches[b] = make([]complex128, nGains)
		for gi := range nGains {
			branches[b][gi] = ordered[gi][b]
		}
	}
	return branches
}

func computeBreakaway(sys *System, poles, zeros []complex128) []complex128 {
	tfr, err := sys.TransferFunction(nil)
	if err != nil {
		return nil
	}
	num := Poly(tfr.TF.Num[0][0])
	den := Poly(tfr.TF.Den[0])

	if len(num) == 0 || len(den) == 0 {
		return nil
	}

	numD := num.Derivative()
	denD := den.Derivative()

	// breakaway: num'*den - num*den' = 0
	poly := numD.Mul(den).Sub(num.Mul(denD))

	roots, err := poly.Roots()
	if err != nil || len(roots) == 0 {
		return nil
	}

	var result []complex128
	for _, r := range roots {
		if math.Abs(imag(r)) > 1e-6*math.Max(1, math.Abs(real(r))) {
			continue
		}
		s := real(r)
		if isOnRealAxisSegment(s, poles, zeros) || isNearRepeatedReal(s, poles) || isNearRepeatedReal(s, zeros) {
			result = append(result, complex(s, 0))
		}
	}
	return result
}

func isOnRealAxisSegment(s float64, poles, zeros []complex128) bool {
	// Count real poles and zeros to the right of s
	count := 0
	for _, p := range poles {
		if math.Abs(imag(p)) < 1e-6 && real(p) > s {
			count++
		}
	}
	for _, z := range zeros {
		if math.Abs(imag(z)) < 1e-6 && real(z) > s {
			count++
		}
	}
	return count%2 == 1
}

func isNearRepeatedReal(s float64, roots []complex128) bool {
	count := 0
	for _, r := range roots {
		if math.Abs(imag(r)) < 1e-6 && math.Abs(real(r)-s) < 1e-4 {
			count++
		}
	}
	return count >= 2
}
