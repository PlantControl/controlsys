package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

type energyAnalysisPolicy struct {
	sys     *System
	n, m, p int
}

func newEnergyAnalysisPolicy(sys *System) energyAnalysisPolicy {
	n, m, p := sys.Dims()
	return energyAnalysisPolicy{sys: sys, n: n, m: m, p: p}
}

func (p energyAnalysisPolicy) requireStandard(context string) error {
	return newDescriptorPolicy(p.sys).requireStandard(context)
}

func (p energyAnalysisPolicy) requireStable(errUnstable error) error {
	stable, err := p.sys.IsStable()
	if err != nil {
		return err
	}
	if !stable {
		return errUnstable
	}
	return nil
}

// gramianInputs returns the Lyapunov operands of the typ gramian with B (or
// C) scaled by 2⁻ᵉ, so that B·Bᵀ (or Cᵀ·C) cannot overflow; the gramian of
// sys is the solution times 4ᵉ.
func (p energyAnalysisPolicy) gramianInputs(typ GramType) (Aarg, Q *mat.Dense, e int, err error) {
	aRaw := p.sys.A.RawMatrix()
	aData := make([]float64, p.n*p.n)
	copyStrided(aData, p.n, aRaw.Data, aRaw.Stride, p.n, p.n)

	switch typ {
	case GramControllability:
		Q, e = p.controllabilityEnergy()
		return mat.NewDense(p.n, p.n, aData), Q, e, nil
	case GramObservability:
		Q, e = p.observabilityEnergy()
		return mat.NewDense(p.n, p.n, transposeSquareData(aRaw.Data, aRaw.Stride, p.n)), Q, e, nil
	default:
		return nil, nil, 0, fmt.Errorf("unknown gramian type %d: %w", typ, ErrInvalidArgument)
	}
}

// scaledGramian returns X and e with the typ gramian of sys equal to X·4ᵉ.
func (p energyAnalysisPolicy) scaledGramian(typ GramType) (*mat.Dense, int, error) {
	Aarg, Q, e, err := p.gramianInputs(typ)
	if err != nil {
		return nil, 0, err
	}
	X, err := p.solveLyapunov(Aarg, Q)
	if err != nil {
		return nil, 0, err
	}
	return X, e, nil
}

func (p energyAnalysisPolicy) controllabilityEnergy() (*mat.Dense, int) {
	if p.m == 0 {
		return mat.NewDense(p.n, p.n, nil), 0
	}
	B, e := pow2Scaled(p.sys.B)
	bRaw := B.RawMatrix()
	q := make([]float64, p.n*p.n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: p.n, Cols: p.m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: p.n, Cols: p.m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: p.n, Cols: p.n, Stride: p.n, Data: q})
	symmetrize(q, p.n, p.n)
	return mat.NewDense(p.n, p.n, q), e
}

func (p energyAnalysisPolicy) observabilityEnergy() (*mat.Dense, int) {
	if p.p == 0 {
		return mat.NewDense(p.n, p.n, nil), 0
	}
	C, e := pow2Scaled(p.sys.C)
	cRaw := C.RawMatrix()
	q := make([]float64, p.n*p.n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p.p, Cols: p.n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p.p, Cols: p.n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: p.n, Cols: p.n, Stride: p.n, Data: q})
	symmetrize(q, p.n, p.n)
	return mat.NewDense(p.n, p.n, q), e
}

// ldexpFinite multiplies every entry of data by 2ᵉ in place and reports
// whether all results are finite.
func ldexpFinite(data []float64, e int) bool {
	ok := true
	for i, x := range data {
		data[i] = math.Ldexp(x, e)
		ok = ok && isFinite(data[i])
	}
	return ok
}

// ldexpDense multiplies every entry of M by 2ᵉ in place.
func ldexpDense(M *mat.Dense, e int) {
	if e == 0 {
		return
	}
	raw := M.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j, x := range row {
			row[j] = math.Ldexp(x, e)
		}
	}
}

func (p energyAnalysisPolicy) solveLyapunov(A, Q *mat.Dense) (*mat.Dense, error) {
	if p.sys.IsContinuous() {
		return Lyap(A, Q, nil)
	}
	return DLyap(A, Q, nil)
}

func transposeSquareData(data []float64, stride, n int) []float64 {
	out := make([]float64, n*n)
	for i := range n {
		for j := range n {
			out[i*n+j] = data[j*stride+i]
		}
	}
	return out
}

// finiteDimensionalModel returns sys with discrete internal delays folded
// into the state exactly. Continuous internal delays give infinitely many
// poles and no finite-order Lyapunov form, so they are rejected with
// ErrContinuousInternalDelay.
func finiteDimensionalModel(sys *System, context string) (*System, error) {
	if !sys.HasInternalDelay() {
		return sys, nil
	}
	if sys.IsContinuous() {
		return nil, fmt.Errorf("%s: %w", context, ErrContinuousInternalDelay)
	}
	return absorbInternalDelay(sys)
}

// lftHasDirectFeedthrough reports whether some D12·D22^k·D21 is nonzero, i.e.
// the internal delays carry a non-decaying u→y path.
func lftHasDirectFeedthrough(lft *LFTDelay) bool {
	if lft == nil || nonEmptyDense(lft.D12) == nil || nonEmptyDense(lft.D21) == nil {
		return false
	}
	path := mat.DenseCopyOf(lft.D21)
	for range lft.Tau {
		var g mat.Dense
		g.Mul(lft.D12, path)
		if !allZeroDense(&g) {
			return true
		}
		if lft.D22 == nil {
			return false
		}
		var next mat.Dense
		next.Mul(lft.D22, path)
		path = &next
	}
	return false
}
