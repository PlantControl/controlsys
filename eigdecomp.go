package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

func decomposeByEigenvalues(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	if sys.HasDelay() {
		return nil, nil, fmt.Errorf("controlsys: decomposition does not support delayed systems; use Pade/AbsorbDelay first")
	}
	policy := newRealizationTransformPolicy(sys)
	n, m, p := sys.Dims()
	if n == 0 {
		cp := sys.Copy()
		gain := policy.zeroOrderZeroFeedthrough()
		return cp, gain, nil
	}

	var eig mat.Eigen
	ok := eig.Factorize(sys.A, mat.EigenRight)
	if !ok {
		return decomposeBySchur(sys, isGroup1)
	}

	vals := eig.Values(nil)

	var idx1, idx2 []int
	assigned := make([]bool, n)
	for i := range n {
		if assigned[i] {
			continue
		}
		if imag(vals[i]) != 0 {
			j := -1
			for k := i + 1; k < n; k++ {
				if !assigned[k] && isConjugate(vals[i], vals[k]) {
					j = k
					break
				}
			}
			if j >= 0 {
				if isGroup1(vals[i]) {
					idx1 = append(idx1, i, j)
				} else {
					idx2 = append(idx2, i, j)
				}
				assigned[i] = true
				assigned[j] = true
				continue
			}
		}
		if isGroup1(vals[i]) {
			idx1 = append(idx1, i)
		} else {
			idx2 = append(idx2, i)
		}
		assigned[i] = true
	}

	sort.Ints(idx1)
	sort.Ints(idx2)

	n1 := len(idx1)
	n2 := len(idx2)

	if n1 == 0 {
		return policy.zeroOrderZeroFeedthrough(), sys.Copy(), nil
	}
	if n2 == 0 {
		return policy.copyWithZeroFeedthrough(), policy.zeroOrderOriginalFeedthrough(), nil
	}

	var vecC mat.CDense
	eig.VectorsTo(&vecC)

	reordered := make([]int, 0, n)
	reordered = append(reordered, idx1...)
	reordered = append(reordered, idx2...)

	V := mat.NewCDense(n, n, nil)
	for j, col := range reordered {
		for i := range n {
			V.Set(i, j, vecC.At(i, col))
		}
	}

	Vinv, err := cinvert(V, n)
	if err != nil {
		return decomposeBySchur(sys, isGroup1)
	}

	condNum := cmatNorm(V, n) * cmatNorm(Vinv, n)
	if condNum > 1e12 {
		return decomposeBySchur(sys, isGroup1)
	}

	At := cmatMul3(Vinv, sys.A, V, n, n, n)
	Bt := cmatMulDense(Vinv, sys.B, n, m)
	Ct := cdenseMulCmat(sys.C, V, p, n)

	A1r := realPart(At, 0, n1, 0, n1)
	B1r := realPart(Bt, 0, n1, 0, m)
	C1r := realPart(Ct, 0, p, 0, n1)

	A2r := realPart(At, n1, n, n1, n)
	B2r := realPart(Bt, n1, n, 0, m)
	C2r := realPart(Ct, 0, p, n1, n)

	sys1, err := policy.resultWithZeroFeedthrough(A1r, B1r, C1r)
	if err != nil {
		return nil, nil, err
	}

	sys2, err := policy.resultWithOriginalFeedthrough(A2r, B2r, C2r)
	if err != nil {
		return nil, nil, err
	}

	return sys1, sys2, nil
}

func decomposeBySchur(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	policy := newRealizationTransformPolicy(sys)
	n, m, p := sys.Dims()

	t := make([]float64, n*n)
	aRaw := sys.A.RawMatrix()
	copyStrided(t, n, aRaw.Data, aRaw.Stride, n, n)

	z := make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	bwork := make([]bool, n)

	workQuery := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, workQuery, -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, work, lwork, bwork)
	if !ok {
		return nil, nil, ErrSchurFailed
	}

	evals := schurEigenvaluesRaw(t, n)

	n1 := 0
	for i := range n {
		if isGroup1(evals[i]) {
			n1++
		}
	}

	if n1 == 0 {
		return policy.zeroOrderZeroFeedthrough(), sys.Copy(), nil
	}
	if n1 == n {
		return policy.copyWithZeroFeedthrough(), policy.zeroOrderOriginalFeedthrough(), nil
	}

	trexcWork := make([]float64, n)
	nDone := 0
	i := 0
	for i < n {
		blockSize := 1
		if i+1 < n && t[(i+1)*n+i] != 0 {
			blockSize = 2
		}

		ev := evals[i]
		if isGroup1(ev) {
			if i != nDone {
				_, _, swapOk := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, i, nDone, trexcWork)
				if !swapOk {
					return nil, nil, ErrSchurFailed
				}
				evals = schurEigenvaluesRaw(t, n)
			}
			if nDone+1 < n && t[(nDone+1)*n+nDone] != 0 {
				nDone += 2
			} else {
				nDone++
			}
			i = nDone
		} else {
			i += blockSize
		}
	}

	n1 = nDone

	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	zGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}

	bt := make([]float64, n*m)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bt})

	ct := make([]float64, p*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		zGen,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: ct})

	A1 := mat.NewDense(n1, n1, nil)
	for i := range n1 {
		for j := range n1 {
			A1.Set(i, j, t[i*n+j])
		}
	}
	B1 := mat.NewDense(n1, m, nil)
	for i := range n1 {
		copy(B1.RawMatrix().Data[i*m:i*m+m], bt[i*m:i*m+m])
	}
	C1 := mat.NewDense(p, n1, nil)
	for i := range p {
		for j := range n1 {
			C1.Set(i, j, ct[i*n+j])
		}
	}

	n2 := n - n1
	A2 := mat.NewDense(n2, n2, nil)
	for i := range n2 {
		for j := range n2 {
			A2.Set(i, j, t[(n1+i)*n+(n1+j)])
		}
	}
	B2 := mat.NewDense(n2, m, nil)
	for i := range n2 {
		copy(B2.RawMatrix().Data[i*m:i*m+m], bt[(n1+i)*m:(n1+i)*m+m])
	}
	C2 := mat.NewDense(p, n2, nil)
	for i := range p {
		for j := range n2 {
			C2.Set(i, j, ct[i*n+(n1+j)])
		}
	}

	sys1, err := policy.resultWithZeroFeedthrough(A1, B1, C1)
	if err != nil {
		return nil, nil, err
	}

	sys2, err := policy.resultWithOriginalFeedthrough(A2, B2, C2)
	if err != nil {
		return nil, nil, err
	}

	return sys1, sys2, nil
}

func schurEigenvaluesRaw(t []float64, n int) []complex128 {
	evals := make([]complex128, n)
	i := 0
	for i < n {
		if i+1 < n && t[(i+1)*n+i] != 0 {
			a := t[i*n+i]
			b := t[i*n+i+1]
			c := t[(i+1)*n+i]
			d := t[(i+1)*n+i+1]
			tr := a + d
			det := a*d - b*c
			disc := tr*tr - 4*det
			if disc < 0 {
				re := tr / 2
				im := math.Sqrt(-disc) / 2
				evals[i] = complex(re, im)
				evals[i+1] = complex(re, -im)
			} else {
				sq := math.Sqrt(disc)
				evals[i] = complex((tr+sq)/2, 0)
				evals[i+1] = complex((tr-sq)/2, 0)
			}
			i += 2
		} else {
			evals[i] = complex(t[i*n+i], 0)
			i++
		}
	}
	return evals
}

func isConjugate(a, b complex128) bool {
	return math.Abs(real(a)-real(b)) < 1e-10*math.Max(1, math.Abs(real(a))) &&
		math.Abs(imag(a)+imag(b)) < 1e-10*math.Max(1, math.Abs(imag(a)))
}

func cinvert(A *mat.CDense, n int) (*mat.CDense, error) {
	augR := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			v := A.At(i, j)
			augR.Set(i, j, real(v))
			augR.Set(i, j+n, -imag(v))
			augR.Set(i+n, j, imag(v))
			augR.Set(i+n, j+n, real(v))
		}
	}

	augI := mat.NewDense(2*n, 2*n, nil)
	for i := range 2 * n {
		augI.Set(i, i, 1)
	}

	var lu mat.LU
	lu.Factorize(augR)
	var sol mat.Dense
	if err := lu.SolveTo(&sol, false, augI); err != nil {
		return nil, fmt.Errorf("controlsys: eigenvector matrix singular: %w", err)
	}

	result := mat.NewCDense(n, n, nil)
	for i := range n {
		for j := range n {
			re := sol.At(i, j)
			im := sol.At(i+n, j)
			result.Set(i, j, complex(re, im))
		}
	}
	return result, nil
}

func cmatNorm(A *mat.CDense, n int) float64 {
	sum := 0.0
	for i := range n {
		for j := range n {
			sum += cmplx.Abs(A.At(i, j)) * cmplx.Abs(A.At(i, j))
		}
	}
	return math.Sqrt(sum)
}

func cmatMul3(Vinv *mat.CDense, Areal *mat.Dense, V *mat.CDense, n1, n2, n3 int) *mat.CDense {
	tmp := mat.NewCDense(n1, n2, nil)
	for i := range n1 {
		for j := range n2 {
			var s complex128
			for k := range n2 {
				s += Vinv.At(i, k) * complex(Areal.At(k, j), 0)
			}
			tmp.Set(i, j, s)
		}
	}
	result := mat.NewCDense(n1, n3, nil)
	for i := range n1 {
		for j := range n3 {
			var s complex128
			for k := range n2 {
				s += tmp.At(i, k) * V.At(k, j)
			}
			result.Set(i, j, s)
		}
	}
	return result
}

func cmatMulDense(Vinv *mat.CDense, B *mat.Dense, n, m int) *mat.CDense {
	result := mat.NewCDense(n, m, nil)
	for i := range n {
		for j := range m {
			var s complex128
			for k := range n {
				s += Vinv.At(i, k) * complex(B.At(k, j), 0)
			}
			result.Set(i, j, s)
		}
	}
	return result
}

func cdenseMulCmat(C *mat.Dense, V *mat.CDense, p, n int) *mat.CDense {
	result := mat.NewCDense(p, n, nil)
	for i := range p {
		for j := range n {
			var s complex128
			for k := range n {
				s += complex(C.At(i, k), 0) * V.At(k, j)
			}
			result.Set(i, j, s)
		}
	}
	return result
}

func realPart(C *mat.CDense, r0, r1, c0, c1 int) *mat.Dense {
	rows := r1 - r0
	cols := c1 - c0
	result := mat.NewDense(rows, cols, nil)
	for i := range rows {
		for j := range cols {
			result.Set(i, j, real(C.At(r0+i, c0+j)))
		}
	}
	return result
}
