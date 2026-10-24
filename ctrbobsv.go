package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

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
// with the nc = sum(K) controllable states last. A singular value of a
// staircase step counts as rank above tol·max(σ₁(step), ‖A‖_F), so rounding
// noise in a block small against ‖A‖ is not rank. tol = 0 selects n²·eps
// with the floor raised to MATLAB's absolute default 10·n·‖A‖₁·eps.
// Nil, non-finite or
// mismatched matrices, a negative or NaN tol and a model with no states
// return an error.
func CtrbF(A, B, C *mat.Dense, tol float64) (*StaircaseForm, error) {
	n, err := requireStaircaseArgs("CtrbF", A, B, C, tol)
	if err != nil {
		return nil, err
	}
	res, err := controllabilityForm(A, B, C, n, tol)
	if err != nil {
		return nil, fmt.Errorf("CtrbF: %w", err)
	}
	return res, nil
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
	dual, err := controllabilityForm(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), mat.DenseCopyOf(B.T()), n, tol)
	if err != nil {
		return nil, fmt.Errorf("ObsvF: %w", err)
	}
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

// controllabilityForm runs the staircase with the controllable states first
// and reverses the state order to MATLAB's layout with the controllable
// states last.
func controllabilityForm(A, B, C *mat.Dense, n int, tol float64) (*StaircaseForm, error) {
	st, err := controllabilityStaircaseFloor(A, B, C, tol, ctrbfFloor(A, tol), true)
	if err != nil {
		return nil, err
	}
	p, _ := C.Dims()
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
			res.T.Set(rev(i), j, st.T.At(i, j))
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
	return res, nil
}

// IsStabilizable reports whether every uncontrollable mode of (A, B) is
// stable: in the open left half-plane when continuous, inside the unit circle
// otherwise. A mode within rounding of uncontrollable on the boundary (PBH
// rank at the staircase's floor) is not stabilizable, whichever side
// rounding puts its eigenvalue; so is an unstable mode whose PBH rank is at
// the floor when the staircase's coupling could be amplified rounding (a weak
// stair or nearly parallel inputs). A model with no states is stabilizable. Nil, non-finite or
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

	res, err := controllabilityStaircase(A, B, nil, 0, false)
	if err != nil {
		return false, fmt.Errorf("%s: %w", op, err)
	}
	// Coupling below rounding amplified by the staircase (a weak stair,
	// nearly parallel inputs, a dominant unstable mode; ≤ 1e-3 of the
	// bound seen) may be no coupling at all, so PBH decides.
	an := frobNorm(A)
	weak := res.MinCoupling <= max(float64(n)*math.Sqrt(eps())*an, float64(n)*eps()*an*res.Amplification)
	var eig mat.Eigen
	if nc2 := n - res.NCont; nc2 > 0 {
		auc := mat.NewDense(nc2, nc2, nil)
		aRaw := res.A.RawMatrix()
		for i := range nc2 {
			for j := range nc2 {
				auc.Set(i, j, aRaw.Data[(res.NCont+i)*aRaw.Stride+(res.NCont+j)])
			}
		}
		if !eig.Factorize(auc, mat.EigenNone) {
			return false, fmt.Errorf("%s: eigenvalues of the uncontrollable part: %w", op, ErrSchurFailed)
		}
		for _, v := range eig.Values(nil) {
			if poleOnOrOutsideStabilityBoundary(v, continuous, poleStabilityTolerance(v)) {
				return false, nil
			}
		}
	} else if !weak {
		return true, nil
	}
	if !eig.Factorize(A, mat.EigenNone) {
		return false, fmt.Errorf("%s: eigenvalues of A: %w", op, ErrSchurFailed)
	}
	return !boundaryPBHDeficient(A, B, eig.Values(nil), continuous, weak), nil
}

// unstablePBHDeficient reports a computed λ on or outside the stability
// boundary with σmin([A − λI, B]) ≤ floor at some point within δλ of it, δλ
// bounding the error of λ. σmin is 1-Lipschitz in λ, so only σmin(λ) ≤
// floor + δλ needs the search (diskSigmaMin). δλ, needing eigenvectors, is
// computed only when some σmin lies within band of floor.
func unstablePBHDeficient(A, B *mat.Dense, lams []complex128, continuous bool, floor, band float64) bool {
	test := func(lam complex128) bool {
		return imag(lam) >= 0 && poleOnOrOutsideStabilityBoundary(lam, continuous, poleStabilityTolerance(lam))
	}
	need := false
	for _, lam := range lams {
		if !test(lam) {
			continue
		}
		s := pbhSigmaMin(A, B, lam)
		if s <= floor {
			return true
		}
		need = need || s <= floor+band
	}
	if !need {
		return false
	}
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenBoth) {
		return true
	}
	lams = eig.Values(nil)
	slack := eigenvalueErrorBounds(&eig, A, len(lams), band)
	var searched []complex128
	for k, lam := range lams {
		if !test(lam) || pbhSigmaMin(A, B, lam) > floor+slack[k] {
			continue
		}
		if slack[k] <= floor {
			return true
		}
		var mean complex128
		var cnt int
		for _, mu := range lams {
			if cmplx.Abs(mu-lam) <= slack[k] {
				mean += mu
				cnt++
			}
		}
		mean /= complex(float64(cnt), 0)
		if slices.ContainsFunc(searched, func(z complex128) bool { return cmplx.Abs(z-mean) <= slack[k] }) {
			continue
		}
		searched = append(searched, mean)
		if diskSigmaMin(A, B, lam, slack[k], floor, mean) <= floor {
			return true
		}
	}
	return false
}

// diskSigmaMin is the least σmin([A − λI, B]) found by compass search from
// mean over λ within radius of center, stopping once it is at most floor or
// the step is below floor/10 or a quarter of its excess over floor (a local
// minimum well above floor). A defective mode's computed λ (center) errs by
// up to √eps·‖A‖ but its cluster mean is accurate, so a controllable Jordan
// block keeps the σmin it has there rather than the Lipschitz slack.
func diskSigmaMin(A, B *mat.Dense, center complex128, radius, floor float64, mean complex128) float64 {
	at := func(z complex128) float64 {
		if imag(z) < 0 {
			z = cmplx.Conj(z)
		}
		return pbhSigmaMin(A, B, z)
	}
	x, f := mean, at(mean)
	step := radius / 2
	for evals := 0; f > floor && step > max(floor/10, (f-floor)/4) && evals < 400; {
		moved := false
		for _, d := range [4]complex128{1, -1, 1i, -1i} {
			y := x + d*complex(step, 0)
			if cmplx.Abs(y-center) > radius {
				continue
			}
			evals++
			if fy := at(y); fy < f {
				x, f, moved = y, fy, true
				break
			}
		}
		if !moved {
			step /= 2
		}
	}
	return f
}

// eigenvalueErrorBounds is n·eps·‖A‖_F·κ(λ) for each eigenvalue of a
// factorization with both eigenvector sets, κ = ‖x‖‖y‖/|yᴴx|, capped at
// band where a mode is defective to rounding.
func eigenvalueErrorBounds(eig *mat.Eigen, A *mat.Dense, n int, band float64) []float64 {
	an := frobNorm(A)
	var vr, vl mat.CDense
	eig.VectorsTo(&vr)
	eig.LeftVectorsTo(&vl)
	out := make([]float64, n)
	for k := range n {
		var xx, yy float64
		var yx complex128
		for i := range n {
			x, y := vr.At(i, k), vl.At(i, k)
			xx += real(x)*real(x) + imag(x)*imag(x)
			yy += real(y)*real(y) + imag(y)*imag(y)
			yx += cmplx.Conj(y) * x
		}
		out[k] = band
		if d := cmplx.Abs(yx); d > 0 {
			out[k] = min(band, float64(n)*eps()*an*math.Sqrt(xx*yy)/d)
		}
	}
	return out
}

// boundaryPBHDeficient reports a point λ on the stability boundary where
// [A − λI, B] has rank below n: σmin ≤ n²·eps·‖[A, B]‖_F, the staircase's
// rank floor, with B scaled to ‖B‖_F = ‖A‖_F as the staircase's first step
// is relative to σ₁(B), so the verdict is invariant to B's scale (a tiny B
// gives σmin ≤ ‖B‖ at a pole near the boundary). The staircase alone
// misses an uncontrollable boundary mode whose Jordan partner is
// controllable: the pair splits by about √(eps·κ)·‖A‖, so the isolated
// mode lands inside or outside the boundary by rounding (FMA flips it), or
// the Krylov coupling, amplified rounding, exceeds the floor and the mode
// counts as controllable. The real boundary points (z = ±1, s = 0) are
// tested exactly when an eigenvalue is within its neighbour gap or
// band = n·√eps·‖A‖_F of them, so the verdict does not depend on which side
// rounding put a split pair. Off the real axis a split pair's mean,
// accurate only to eps·κ, is projected onto the boundary and σmin
// minimized along it within the pair's gap; pairs farther apart than 16·band
// are distinct modes, not a split, and are left to the staircase. When weak,
// the staircase counted rounding-level coupling as rank, which can hide a
// simple uncontrollable mode anywhere, so every λ on or outside the
// boundary is tested too (unstablePBHDeficient).
func boundaryPBHDeficient(A, B *mat.Dense, lams []complex128, continuous, weak bool) bool {
	n, _ := A.Dims()
	an := frobNorm(A)
	if bn := frobNorm(B); bn > 0 {
		s := an
		if s == 0 {
			s = 1
		}
		B = mat.DenseCopyOf(B)
		B.Scale(s/bn, B)
	}
	floor := float64(n*n) * eps() * math.Hypot(an, frobNorm(B))
	band := float64(n) * math.Sqrt(eps()) * an
	if weak && unstablePBHDeficient(A, B, lams, continuous, floor, band) {
		return true
	}
	gaps := make([]float64, len(lams))
	nbs := make([]int, len(lams))
	for k, lam := range lams {
		gaps[k], nbs[k] = math.Inf(1), -1
		for j, mu := range lams {
			if d := cmplx.Abs(mu - lam); j != k && d < gaps[k] {
				gaps[k], nbs[k] = d, j
			}
		}
	}
	points := []complex128{0}
	if !continuous {
		points = []complex128{1, -1}
	}
	for _, pt := range points {
		for k, lam := range lams {
			if cmplx.Abs(lam-pt) <= max(band, min(gaps[k], 1)) {
				if pbhSigmaMin(A, B, pt) <= floor {
					return true
				}
				break
			}
		}
	}
	for k, lam := range lams {
		j := nbs[k]
		if gaps[k] > 16*band || (j < k && nbs[j] == k) {
			continue
		}
		mean := (lam + lams[j]) / 2
		lb, d := projectToStabilityBoundary(mean, continuous)
		if imag(lb) == 0 || d > gaps[k] {
			continue
		}
		if boundarySigmaMin(A, B, lb, gaps[k], floor, continuous) <= floor {
			return true
		}
	}
	return false
}

// boundarySigmaMin is the least σmin([A − λI, B]) found by golden-section
// search over boundary points λ within radius of lb: the arc e^{iθ} when
// discrete, the imaginary axis when continuous. σmin is 1-Lipschitz in λ,
// so the search stops at an interval of floor/10.
func boundarySigmaMin(A, B *mat.Dense, lb complex128, radius, floor float64, continuous bool) float64 {
	best := pbhSigmaMin(A, B, lb)
	p0 := imag(lb)
	at := func(p float64) complex128 { return complex(0, p) }
	if !continuous {
		p0 = cmplx.Phase(lb)
		at = func(p float64) complex128 { return cmplx.Rect(1, p) }
	}
	const ratio = 0.6180339887498949
	lo, hi := p0-radius, p0+radius
	x1, x2 := hi-ratio*(hi-lo), lo+ratio*(hi-lo)
	f1, f2 := pbhSigmaMin(A, B, at(x1)), pbhSigmaMin(A, B, at(x2))
	for range 80 {
		if hi-lo <= max(floor/10, eps()*math.Abs(p0)) {
			break
		}
		if f1 <= f2 {
			hi, x2, f2 = x2, x1, f1
			x1 = hi - ratio*(hi-lo)
			f1 = pbhSigmaMin(A, B, at(x1))
		} else {
			lo, x1, f1 = x1, x2, f2
			x2 = lo + ratio*(hi-lo)
			f2 = pbhSigmaMin(A, B, at(x2))
		}
	}
	return min(best, f1, f2)
}

// projectToStabilityBoundary returns the point of the stability boundary
// nearest p and its distance from p.
func projectToStabilityBoundary(p complex128, continuous bool) (complex128, float64) {
	if continuous {
		return complex(0, imag(p)), math.Abs(real(p))
	}
	r := cmplx.Abs(p)
	if r == 0 {
		return 1, 1
	}
	return p / complex(r, 0), math.Abs(1 - r)
}

// pbhSigmaMin is σmin([A − λI, B]), through the real embedding when λ is
// complex.
func pbhSigmaMin(A, B *mat.Dense, lam complex128) float64 {
	n, _ := A.Dims()
	_, m := B.Dims()
	aRaw, bRaw := A.RawMatrix(), B.RawMatrix()
	lr, li := real(lam), imag(lam)
	k := 1
	if li != 0 {
		k = 2
	}
	cols := k * (n + m)
	g := mat.NewDense(k*n, cols, nil)
	gRaw := g.RawMatrix()
	for i := range n {
		for j := range n {
			re := aRaw.Data[i*aRaw.Stride+j]
			if i == j {
				re -= lr
			}
			gRaw.Data[i*gRaw.Stride+j] = re
			if k == 2 {
				gRaw.Data[(n+i)*gRaw.Stride+n+m+j] = re
			}
		}
		for j := range m {
			b := bRaw.Data[i*bRaw.Stride+j]
			gRaw.Data[i*gRaw.Stride+n+j] = b
			if k == 2 {
				gRaw.Data[(n+i)*gRaw.Stride+2*n+m+j] = b
			}
		}
		if k == 2 {
			gRaw.Data[i*gRaw.Stride+n+m+i] = li
			gRaw.Data[(n+i)*gRaw.Stride+i] = -li
		}
	}
	var svd mat.SVD
	if !svd.Factorize(g, mat.SVDNone) {
		return 0
	}
	sv := svd.Values(nil)
	return sv[k*n-1]
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
