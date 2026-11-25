package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// errHoldUnverified comes with the ZOH, FOH or Tustin model of a
// descriptor whose DC gain is not verified against DCGain of the caller's
// model (holdGate); C2D and C2DMap return it as ErrAccuracyUnverified.
var errHoldUnverified = errors.New("discretized model not verified")

func holdUnverified(format string, args ...any) error {
	return fmt.Errorf(format+": %w", append(args, errHoldUnverified)...)
}

// holdGated reports a C2D of sys by method that holdGate judges: a
// continuous delay-free-core descriptor (I/O delays allowed, no LFT) by
// ZOH, FOH or Tustin.
func holdGated(sys *System, method C2DMethod) bool {
	switch method {
	case "", C2DMethodZOH, C2DMethodFOH, C2DMethodTustin:
	default:
		return false
	}
	return sys.IsContinuous() && sys.IsDescriptor() && sys.LFT == nil
}

// holdGate is the DC gate of ZOH, FOH and Tustin C2D of a descriptor: the
// model is nil only where, entry by entry, its DC gain ĥ, exact for its
// stored entries, lies within the descriptor bar (dcBar) of g0 = DCGain of
// the caller's delay-free model, plus the rounding ε·S of the result's own
// entries,
//
//	S = |D| + Σ|C||X| + |C(I − Φ)⁻¹|·(|Φ||X| + |Γ|),  X = (I − Φ)⁻¹Γ,
//
// the first-order change of ĥ under ε-relative changes of (Φ, Γ, C, D).
// That allowance counts only where ε·S ≤ √ε·max(|g0|, T0), T0 the caller's
// terms, and I − Φ is not near singular: a poor realization inflates its
// own S. The test is |ĥ − g0| + err(ĥ) + err(g0) ≤ bar, err(g0) = ε·T0 +
// ½ulp by the DCGain contract and err(ĥ) DCGain's bound on the result. An
// entry of g0 that is ±Inf (a pole at DC) is not gated; where other
// entries are, no bar exists and the model is unverified. refused marks a
// fold taken for a stiff split that was not certified: always unverified.
type holdGate struct {
	g0      *mat.Dense
	err     error
	bar     dcBar
	gated   bool
	refused error
}

func newHoldGate(sys *System) *holdGate {
	free := ioDelayFree(sys)
	n, _, _ := free.Dims()
	var solved []dd
	g0, err := (*mat.Dense)(nil), error(nil)
	if r := solveDCRegular(nil, free.A, free.B, free.C, free.D, 0, n > exactDetMaxOrder); r.solveErr == nil && !r.overflow && r.certified {
		g0, solved = r.g, r.x
	} else if g0, err = free.DCGain(); err != nil {
		return &holdGate{err: holdUnverified("DCGain of the model: %v", err)}
	}
	gate := &holdGate{g0: g0}
	for _, v := range g0.RawMatrix().Data {
		if !math.IsInf(v, 0) {
			gate.gated = true
		} else if gate.err == nil {
			gate.err = holdUnverified("DCGain of the model has a pole at DC")
		}
	}
	if !gate.gated {
		gate.err = nil
		return gate
	}
	if gate.err != nil {
		return gate
	}
	gate.bar = newDCBarSolved(free, solved)
	if gate.bar.reason != "" {
		gate.err = holdUnverified("%s", gate.bar.reason)
	}
	return gate
}

// check returns nil where out passes the gate, else errHoldUnverified.
func (g *holdGate) check(out *System) error {
	if g.refused != nil {
		return holdUnverified("%v", g.refused)
	}
	if !g.gated {
		return nil
	}
	if g.err != nil {
		return g.err
	}
	if out.LFT != nil {
		return holdUnverified("DC gain of a result with internal delays (fractional I/O delays) is not checked")
	}
	if err := finiteSystemMatrices("result", out, ErrOverflow); err != nil {
		return holdUnverified("%v", err)
	}
	h, herr, allow, ok := holdDC(out)
	if !ok {
		return holdUnverified("DC gain of the result is not resolved")
	}
	m := g.bar.m
	p, _ := g.g0.Dims()
	for k := range p * m {
		ref, gk := g.g0.At(k/m, k%m), h.At(k/m, k%m)
		need := mddAbs(mddSub(gk, ref)) + herr[k] + 0x1p-52*g.bar.terms[k] + halfUlp(ref)
		a := 0.0
		if allow != nil && allow[k] <= math.Sqrt(0x1p-52)*math.Max(math.Abs(ref), g.bar.terms[k]) {
			a = allow[k]
		}
		if !(need <= math.MaxFloat64) || !g.bar.within(k, need, a) {
			return holdUnverified("DC gain %v of the result (entry %d,%d) is not within the descriptor bar of DCGain %v", gk, k/m, k%m, ref)
		}
	}
	return nil
}

func halfUlp(v float64) float64 {
	a := math.Abs(v)
	return (math.Nextafter(a, math.Inf(1)) - a) / 2
}

// holdDC is the DC gain of the discrete explicit out, by DCGain's
// machinery, with a bound on each entry's error against the gain exact for
// its stored entries, and the allowance ε·S of each entry (holdAllowance);
// ok is false where neither the certified solve nor the exact solve (order
// up to exactDetMaxOrder) decides.
func holdDC(out *System) (h *mat.Dense, herr, allow []float64, ok bool) {
	n, m, p := out.Dims()
	if n == 0 {
		allow = make([]float64, p*m)
		for i := range p {
			for j := range m {
				allow[i*m+j] = 0x1p-52 * math.Abs(out.D.At(i, j))
			}
		}
		return out.D, make([]float64, p*m), allow, true
	}
	r := solveDCRegular(nil, out.A, out.B, out.C, out.D, 1, n > exactDetMaxOrder)
	if r.solveErr == nil && !r.overflow && r.certified {
		herr = make([]float64, p*m)
		for k := range p * m {
			herr[k] = r.tau[k] + halfUlp(r.g.At(k/m, k%m))
		}
		x := make([]float64, n*m)
		for k, v := range r.x {
			x[k] = v.hi
		}
		return r.g, herr, holdAllowanceFrom(out, x, r.lu, r.ipiv, r.cond), true
	}
	if n > exactDetMaxOrder {
		return nil, nil, nil, false
	}
	g, overflow, ok := exactDCGainAtNonsingular(nil, out.A, out.B, out.C, out.D, 1)
	if !ok || overflow {
		return nil, nil, nil, false
	}
	herr = make([]float64, p*m)
	for k := range p * m {
		herr[k] = halfUlp(g.At(k/m, k%m))
	}
	return g, herr, holdAllowance(out), true
}

// holdAllowance is ε·S of each entry of the discrete explicit out (S of
// holdGate), nil where I − Φ is near singular.
func holdAllowance(out *System) []float64 {
	n, m, _ := out.Dims()
	lu := make([]float64, n*n)
	ar := out.A.RawMatrix()
	for i := range n {
		for j := range n {
			lu[i*n+j] = -ar.Data[i*ar.Stride+j]
		}
		lu[i*n+i]++
	}
	ipiv := make([]int, n)
	work := make([]float64, 4*n)
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, lu, n, work)
	if !impl.Dgetrf(n, n, lu, n, ipiv) {
		return nil
	}
	cond := 1 / impl.Dgecon(lapack.MaxColumnSum, n, lu, n, anorm, work, make([]int, n))
	x := make([]float64, n*m)
	br := out.B.RawMatrix()
	for i := range n {
		copy(x[i*m:i*m+m], br.Data[i*br.Stride:i*br.Stride+m])
	}
	impl.Dgetrs(blas.NoTrans, n, m, lu, n, ipiv, x, max(m, 1))
	return holdAllowanceFrom(out, x, lu, ipiv, cond)
}

// holdAllowanceFrom is holdAllowance from X = (I − Φ)⁻¹Γ (row-major) and
// the LU factors of I − Φ with its condition estimate cond.
func holdAllowanceFrom(out *System, x, lu []float64, ipiv []int, cond float64) []float64 {
	if nearSingularCondition(cond) || !finiteSlice(x) {
		return nil
	}
	n, m, p := out.Dims()
	y := make([]float64, n*p)
	cr := out.C.RawMatrix()
	for i := range p {
		for l := range n {
			y[l*p+i] = cr.Data[i*cr.Stride+l]
		}
	}
	impl.Dgetrs(blas.Trans, n, p, lu, n, ipiv, y, max(p, 1))
	ar, br := out.A.RawMatrix(), out.B.RawMatrix()
	w := make([]float64, n)
	allow := make([]float64, p*m)
	for j := range m {
		for l := range n {
			s := math.Abs(br.Data[l*br.Stride+j])
			for k := range n {
				s += math.Abs(ar.Data[l*ar.Stride+k] * x[k*m+j])
			}
			w[l] = s
		}
		for i := range p {
			s := math.Abs(out.D.At(i, j))
			for l := range n {
				s += math.Abs(cr.Data[i*cr.Stride+l]*x[l*m+j]) + math.Abs(y[l*p+i])*w[l]
			}
			allow[i*m+j] = 0x1p-52 * s
		}
	}
	return allow
}
