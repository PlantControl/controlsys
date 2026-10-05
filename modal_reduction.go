package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type ModalTruncateOptions struct {
	Order       int
	MaxRealPart float64
}

type ModalReductionResult struct {
	Sys        *System
	Order      int
	Method     string
	Kept       []int
	KeptPoles  []complex128
	Basis      *mat.Dense
	Projection *mat.Dense
}

func ModalTruncate(sys *System, opts *ModalTruncateOptions) (*ModalReductionResult, error) {
	if err := requireSystem("ModalTruncate", sys); err != nil {
		return nil, err
	}
	if opts == nil {
		opts = &ModalTruncateOptions{}
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("ModalTruncate"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("ModalTruncate"); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	if opts.Order < 0 || opts.Order > n {
		return nil, fmt.Errorf("ModalTruncate: order %d outside [0,%d]: %w", opts.Order, n, ErrInvalidOrder)
	}
	if n == 0 {
		return &ModalReductionResult{Sys: sys.Copy(), Order: 0, Method: "real-schur-modal-truncate", Basis: &mat.Dense{}, Projection: &mat.Dense{}}, nil
	}

	t, z, err := modalSchur(sys)
	if err != nil {
		return nil, err
	}
	threshold := opts.Order == 0 && opts.MaxRealPart != 0
	if err := orderModalSchur(t, z, n, sys.Dt, opts.MaxRealPart, threshold); err != nil {
		return nil, err
	}
	poles := schurEigenvaluesRaw(t, n)
	order, err := modalReductionOrder(t, poles, n, sys.Dt, opts)
	if err != nil {
		return nil, err
	}
	if order < 1 || order > n {
		return nil, fmt.Errorf("ModalTruncate: order %d outside [1,%d]: %w", order, n, ErrInvalidOrder)
	}
	if splitsSchurBlock(t, n, order) {
		return nil, fmt.Errorf("ModalTruncate: order %d splits a complex-conjugate mode pair: %w", order, ErrInvalidOrder)
	}
	if order < unstableModalCount(poles, sys.Dt) {
		return nil, fmt.Errorf("ModalTruncate: order %d would discard an unstable mode: %w", order, ErrInvalidOrder)
	}
	keptPoles := append([]complex128(nil), poles[:order]...)
	if order == n {
		identity := eyeDense(n)
		return &ModalReductionResult{
			Sys:        sys.Copy(),
			Order:      n,
			Method:     "real-schur-modal-truncate",
			Kept:       rangeInts(n),
			KeptPoles:  keptPoles,
			Basis:      identity,
			Projection: eyeDense(n),
		}, nil
	}

	x, err := separateSchurBlocks(t, n, order, "ModalTruncate")
	if err != nil {
		return nil, err
	}
	bModal, cModal := modalInputOutput(sys, z, n, m, p)
	aReduced := extractModalBlock(t, n, 0, order, 0, order)
	bReduced := newDense(order, m)
	for i := range order {
		for j := range m {
			value := bModal[i*m+j]
			for k := order; k < n; k++ {
				value -= x[i*(n-order)+k-order] * bModal[k*m+j]
			}
			bReduced.Set(i, j, value)
		}
	}
	cReduced := newDense(p, order)
	for i := range p {
		copy(cReduced.RawMatrix().Data[i*order:(i+1)*order], cModal[i*n:i*n+order])
	}
	reduced, err := policy.resultWithOriginalFeedthrough(aReduced, bReduced, cReduced)
	if err != nil {
		return nil, err
	}
	basis, projection := modalProjectionBases(z, x, n, order)
	return &ModalReductionResult{
		Sys:        reduced,
		Order:      order,
		Method:     "real-schur-modal-truncate",
		Kept:       rangeInts(order),
		KeptPoles:  keptPoles,
		Basis:      basis,
		Projection: projection,
	}, nil
}

func modalSchur(sys *System) (t, z []float64, err error) {
	n, _, _ := sys.Dims()
	t = make([]float64, n*n)
	aRaw := sys.A.RawMatrix()
	copyStrided(t, n, aRaw.Data, aRaw.Stride, n, n)
	z = make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	bwork := make([]bool, n)
	query := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, query, -1, bwork)
	work := make([]float64, int(query[0]))
	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, work, len(work), bwork)
	if !ok {
		return nil, nil, ErrSchurFailed
	}
	return t, z, nil
}

func orderModalSchur(t, z []float64, n int, dt, maxRealPart float64, threshold bool) error {
	work := make([]float64, n)
	placed := 0
	for placed < n {
		evals := schurEigenvaluesRaw(t, n)
		best := placed
		for i := placed; i < n; i += schurBlockSize(t, n, i) {
			if modalBlockBefore(evals[i], evals[best], dt, maxRealPart, threshold) {
				best = i
			}
		}
		if best != placed {
			_, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, best, placed, work)
			if !ok {
				return ErrSchurFailed
			}
		}
		placed += schurBlockSize(t, n, placed)
	}
	return nil
}

func modalBlockBefore(candidate, current complex128, dt, maxRealPart float64, threshold bool) bool {
	if threshold {
		candidateSelected := real(candidate) >= maxRealPart || modalPoleUnstable(candidate, dt)
		currentSelected := real(current) >= maxRealPart || modalPoleUnstable(current, dt)
		if candidateSelected != currentSelected {
			return candidateSelected
		}
	}
	candidateScore := real(candidate)
	currentScore := real(current)
	if dt > 0 {
		candidateScore = cmplx.Abs(candidate)
		currentScore = cmplx.Abs(current)
	}
	if candidateScore != currentScore {
		return candidateScore > currentScore
	}
	return math.Abs(imag(candidate)) < math.Abs(imag(current))
}

func modalReductionOrder(t []float64, poles []complex128, n int, dt float64, opts *ModalTruncateOptions) (int, error) {
	if opts.Order > 0 {
		return opts.Order, nil
	}
	if opts.MaxRealPart != 0 {
		order := 0
		for order < n && (real(poles[order]) >= opts.MaxRealPart || modalPoleUnstable(poles[order], dt)) {
			order += schurBlockSize(t, n, order)
		}
		if order == 0 {
			return 0, fmt.Errorf("ModalTruncate: no modes satisfy MaxRealPart %g: %w", opts.MaxRealPart, ErrInvalidOrder)
		}
		return order, nil
	}
	order := n / 2
	if order == 0 {
		order = 1
	}
	if splitsSchurBlock(t, n, order) {
		order++
	}
	unstable := unstableModalCount(poles, dt)
	if order < unstable {
		order = unstable
	}
	return order, nil
}

func separateSchurBlocks(t []float64, n, order int, context string) ([]float64, error) {
	discarded := n - order
	x := make([]float64, order*discarded)
	for i := range order {
		for j := range discarded {
			x[i*discarded+j] = -t[i*n+order+j]
		}
	}
	scale, ok := impl.Dtrsyl(blas.NoTrans, blas.NoTrans, -1, order, discarded, t, n, t[order*n+order:], n, x, discarded)
	if !ok || scale == 0 || scale < 1e-12 {
		return nil, fmt.Errorf("%s: retained and discarded modes cannot be separated reliably: %w", context, ErrSchurFailed)
	}
	if scale != 1 {
		for i := range x {
			x[i] /= scale
		}
	}
	return x, nil
}

func modalInputOutput(sys *System, z []float64, n, m, p int) (bModal, cModal []float64) {
	zGeneral := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}
	bModal = make([]float64, n*m)
	if m > 0 {
		bRaw := sys.B.RawMatrix()
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGeneral,
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bModal})
	}
	cModal = make([]float64, p*n)
	if p == 0 {
		return bModal, cModal
	}
	cRaw := sys.C.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data}, zGeneral,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cModal})
	return bModal, cModal
}

func modalProjectionBases(z, x []float64, n, order int) (*mat.Dense, *mat.Dense) {
	basis := mat.NewDense(n, order, nil)
	for i := range n {
		copy(basis.RawMatrix().Data[i*order:(i+1)*order], z[i*n:i*n+order])
	}
	schurProjection := mat.NewDense(order, n, nil)
	for i := range order {
		schurProjection.Set(i, i, 1)
		for j := order; j < n; j++ {
			schurProjection.Set(i, j, -x[i*(n-order)+j-order])
		}
	}
	projection := mat.NewDense(order, n, nil)
	projection.Mul(schurProjection, mat.NewDense(n, n, z).T())
	return basis, projection
}

func extractModalBlock(data []float64, stride, rowStart, rowEnd, colStart, colEnd int) *mat.Dense {
	rows := rowEnd - rowStart
	cols := colEnd - colStart
	if rows == 0 || cols == 0 {
		return &mat.Dense{}
	}
	out := mat.NewDense(rows, cols, nil)
	for i := range rows {
		copy(out.RawMatrix().Data[i*cols:(i+1)*cols], data[(rowStart+i)*stride+colStart:(rowStart+i)*stride+colEnd])
	}
	return out
}

func schurBlockSize(t []float64, n, start int) int {
	if start+1 < n && t[(start+1)*n+start] != 0 {
		return 2
	}
	return 1
}

func splitsSchurBlock(t []float64, n, order int) bool {
	return order > 0 && order < n && t[order*n+order-1] != 0
}

func unstableModalCount(poles []complex128, dt float64) int {
	count := 0
	for _, pole := range poles {
		if modalPoleUnstable(pole, dt) {
			count++
		}
	}
	return count
}

func modalPoleUnstable(pole complex128, dt float64) bool {
	if dt > 0 {
		return cmplx.Abs(pole) >= 1
	}
	return real(pole) >= 0
}

func rangeInts(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
