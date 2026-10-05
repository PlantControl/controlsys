package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// Ctrb returns the n×(n·m) controllability matrix [B, AB, A²B, …, Aⁿ⁻¹B], as
// MATLAB ctrb(A,B). Nil A or B returns ErrInvalidArgument; A with no states
// returns ErrDimensionMismatch.
func Ctrb(A, B *mat.Dense) (*mat.Dense, error) {
	if A == nil || B == nil {
		return nil, fmt.Errorf("Ctrb: A and B are required: %w", ErrInvalidArgument)
	}
	n, nc := A.Dims()
	if n != nc {
		return nil, fmt.Errorf("Ctrb: A is %d×%d, must be square: %w", n, nc, ErrDimensionMismatch)
	}
	br, m := B.Dims()
	if br != n {
		return nil, fmt.Errorf("Ctrb: B has %d rows, want %d: %w", br, n, ErrDimensionMismatch)
	}
	if n == 0 {
		return nil, fmt.Errorf("Ctrb: system has no states: %w", ErrDimensionMismatch)
	}

	cols := n * m
	data := make([]float64, n*cols)
	bRaw := B.RawMatrix()
	copyStrided(data, cols, bRaw.Data, bRaw.Stride, n, m)

	aGen := rawToGen(A)

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[(k-1)*m:]}
		cur := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[k*m:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aGen, prev, 0, cur)
	}

	return mat.NewDense(n, cols, data), nil
}

// Obsv returns the (n·p)×n observability matrix [C; CA; CA²; …; CAⁿ⁻¹], as
// MATLAB obsv(A,C). Nil A or C returns ErrInvalidArgument; A with no states
// returns ErrDimensionMismatch.
func Obsv(A, C *mat.Dense) (*mat.Dense, error) {
	if A == nil || C == nil {
		return nil, fmt.Errorf("Obsv: A and C are required: %w", ErrInvalidArgument)
	}
	n, nc := A.Dims()
	if n != nc {
		return nil, fmt.Errorf("Obsv: A is %d×%d, must be square: %w", n, nc, ErrDimensionMismatch)
	}
	p, cc := C.Dims()
	if cc != n {
		return nil, fmt.Errorf("Obsv: C has %d columns, want %d: %w", cc, n, ErrDimensionMismatch)
	}
	if n == 0 {
		return nil, fmt.Errorf("Obsv: system has no states: %w", ErrDimensionMismatch)
	}

	rows := n * p
	data := make([]float64, rows*n)
	cRaw := C.RawMatrix()
	copyStrided(data, n, cRaw.Data, cRaw.Stride, p, n)

	aGen := rawToGen(A)

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[(k-1)*p*n:]}
		cur := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[k*p*n:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, prev, aGen, 0, cur)
	}

	return mat.NewDense(rows, n, data), nil
}

// rawToGen returns a blas64.General for m, copying only if stride != cols.
func rawToGen(m *mat.Dense) blas64.General {
	raw := m.RawMatrix()
	if raw.Stride == raw.Cols {
		return blas64.General{Rows: raw.Rows, Cols: raw.Cols, Stride: raw.Stride, Data: raw.Data}
	}
	n := raw.Rows
	data := make([]float64, n*raw.Cols)
	copyStrided(data, raw.Cols, raw.Data, raw.Stride, n, raw.Cols)
	return blas64.General{Rows: n, Cols: raw.Cols, Stride: raw.Cols, Data: data}
}

// StaircaseForm is the result of CtrbF or ObsvF, the outputs
// [Abar,Bbar,Cbar,T,k] of MATLAB ctrbf/obsvf: Abar = T·A·Tᵀ, Bbar = T·B and
// Cbar = C·Tᵀ with T orthogonal. K holds the rank found at each staircase
// step; their sum is the number of controllable (CtrbF) or observable (ObsvF)
// states.
type StaircaseForm struct {
	A, B, C *mat.Dense
	T       *mat.Dense
	K       []int
}

// CtrbF computes the controllability staircase form of (A, B, C), as MATLAB
// [Abar,Bbar,Cbar,T,k] = ctrbf(A,B,C,tol)
// (https://www.mathworks.com/help/control/ref/ctrbf.html):
//
//	Abar = [Anc 0; A21 Ac], Bbar = [0; Bc], Cbar = [Cnc Cc]
//
// with the nc = sum(K) controllable states last. tol is the relative rank
// tolerance of each staircase step; 0 selects n²·eps. Unlike MATLAB's absolute
// default, it scales with the size of each block. Nil, non-finite or
// mismatched matrices, a negative or NaN tol and a model with no states
// return an error.
func CtrbF(A, B, C *mat.Dense, tol float64) (*StaircaseForm, error) {
	n, err := requireStaircaseArgs("CtrbF", A, B, C, tol)
	if err != nil {
		return nil, err
	}
	return controllabilityForm(A, B, C, n, tol), nil
}

// ObsvF computes the observability staircase form of (A, B, C), as MATLAB
// [Abar,Bbar,Cbar,T,k] = obsvf(A,B,C,tol)
// (https://www.mathworks.com/help/control/ref/obsvf.html):
//
//	Abar = [Ano A12; 0 Ao], Bbar = [Bno; Bo], Cbar = [0 Co]
//
// with the no = sum(K) observable states last. It is the dual of CtrbF:
// CtrbF(Aᵀ, Cᵀ, Bᵀ, tol) transposed. tol and the errors are as for CtrbF.
func ObsvF(A, B, C *mat.Dense, tol float64) (*StaircaseForm, error) {
	n, err := requireStaircaseArgs("ObsvF", A, B, C, tol)
	if err != nil {
		return nil, err
	}
	dual := controllabilityForm(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), mat.DenseCopyOf(B.T()), n, tol)
	return &StaircaseForm{
		A: mat.DenseCopyOf(dual.A.T()),
		B: mat.DenseCopyOf(dual.C.T()),
		C: mat.DenseCopyOf(dual.B.T()),
		T: dual.T,
		K: dual.K,
	}, nil
}

func requireStaircaseArgs(op string, A, B, C *mat.Dense, tol float64) (int, error) {
	for _, nm := range []struct {
		name string
		m    *mat.Dense
	}{{"A", A}, {"B", B}, {"C", C}} {
		if err := requireFiniteDense(op, nm.name, nm.m); err != nil {
			return 0, err
		}
	}
	if !(tol >= 0) || math.IsInf(tol, 1) {
		return 0, fmt.Errorf("%s: tol %g must be finite and non-negative: %w", op, tol, ErrInvalidArgument)
	}
	n, nc := A.Dims()
	if n != nc {
		return 0, fmt.Errorf("%s: A is %d×%d, must be square: %w", op, n, nc, ErrDimensionMismatch)
	}
	if br, _ := B.Dims(); br != n {
		return 0, fmt.Errorf("%s: B has %d rows, want %d: %w", op, br, n, ErrDimensionMismatch)
	}
	if _, cc := C.Dims(); cc != n {
		return 0, fmt.Errorf("%s: C has %d columns, want %d: %w", op, cc, n, ErrDimensionMismatch)
	}
	if n == 0 {
		return 0, fmt.Errorf("%s: system has no states: %w", op, ErrDimensionMismatch)
	}
	return n, nil
}

// controllabilityForm runs the staircase with the controllable states first,
// recovering its orthogonal Q by transforming [C; I], then reverses the state
// order to MATLAB's layout with the controllable states last.
func controllabilityForm(A, B, C *mat.Dense, n int, tol float64) *StaircaseForm {
	p, _ := C.Dims()
	cAug := mat.NewDense(p+n, n, nil)
	cAug.Slice(0, p, 0, n).(*mat.Dense).Copy(C)
	for i := range n {
		cAug.Set(p+i, i, 1)
	}
	st := ControllabilityStaircase(A, B, cAug, tol)
	_, m := B.Dims()
	rev := func(i int) int { return n - 1 - i }
	res := &StaircaseForm{
		A: mat.NewDense(n, n, nil),
		B: mat.NewDense(n, m, nil),
		C: mat.NewDense(p, n, nil),
		T: mat.NewDense(n, n, nil),
		K: append([]int{}, st.BlockSizes...),
	}
	for i := range n {
		for j := range n {
			res.A.Set(rev(i), rev(j), st.A.At(i, j))
			res.T.Set(rev(i), j, st.C.At(p+j, i))
		}
		for j := range m {
			res.B.Set(rev(i), j, st.B.At(i, j))
		}
	}
	for i := range p {
		for j := range n {
			res.C.Set(i, rev(j), st.C.At(i, j))
		}
	}
	return res
}

// IsStabilizable reports whether every uncontrollable mode of (A, B) is
// stable: in the open left half-plane when continuous, inside the unit circle
// otherwise. A model with no states is stabilizable. Nil, non-finite or
// mismatched matrices return an error.
func IsStabilizable(A, B *mat.Dense, continuous bool) (bool, error) {
	return isStabilizable("IsStabilizable", A, B, continuous)
}

func isStabilizable(op string, A, B *mat.Dense, continuous bool) (bool, error) {
	if err := requireFiniteDense(op, "A", A); err != nil {
		return false, err
	}
	if err := requireFiniteDense(op, "B", B); err != nil {
		return false, err
	}
	n, nc := A.Dims()
	if n != nc {
		return false, fmt.Errorf("%s: A is %d×%d, must be square: %w", op, n, nc, ErrDimensionMismatch)
	}
	if br, _ := B.Dims(); br != n {
		return false, fmt.Errorf("%s: B has %d rows, want %d: %w", op, br, n, ErrDimensionMismatch)
	}
	if n == 0 {
		return true, nil
	}

	res := ControllabilityStaircase(A, B, nil, 0)
	if res.NCont == n {
		return true, nil
	}

	nc2 := n - res.NCont
	auc := mat.NewDense(nc2, nc2, nil)
	aRaw := res.A.RawMatrix()
	for i := range nc2 {
		for j := range nc2 {
			auc.Set(i, j, aRaw.Data[(res.NCont+i)*aRaw.Stride+(res.NCont+j)])
		}
	}

	var eig mat.Eigen
	if !eig.Factorize(auc, mat.EigenNone) {
		return false, fmt.Errorf("%s: eigenvalues of the uncontrollable part: %w", op, ErrSchurFailed)
	}
	for _, v := range eig.Values(nil) {
		if poleOnOrOutsideStabilityBoundary(v, continuous, poleStabilityTolerance(v)) {
			return false, nil
		}
	}
	return true, nil
}

// IsDetectable reports whether every unobservable mode of (A, C) is stable,
// the dual of IsStabilizable.
func IsDetectable(A, C *mat.Dense, continuous bool) (bool, error) {
	if err := requireFiniteDense("IsDetectable", "A", A); err != nil {
		return false, err
	}
	if err := requireFiniteDense("IsDetectable", "C", C); err != nil {
		return false, err
	}
	n, nc := A.Dims()
	if n != nc {
		return false, fmt.Errorf("IsDetectable: A is %d×%d, must be square: %w", n, nc, ErrDimensionMismatch)
	}
	if _, cc := C.Dims(); cc != n {
		return false, fmt.Errorf("IsDetectable: C has %d columns, want %d: %w", cc, n, ErrDimensionMismatch)
	}
	return isStabilizable("IsDetectable", mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), continuous)
}
