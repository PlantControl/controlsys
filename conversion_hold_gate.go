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
// continuous descriptor (I/O and internal delays allowed) by ZOH, FOH or
// Tustin.
func holdGated(sys *System, method C2DMethod) bool {
	switch method {
	case "", C2DMethodZOH, C2DMethodFOH, C2DMethodTustin:
	default:
		return false
	}
	return sys.IsContinuous() && sys.IsDescriptor()
}

// holdGate is the DC gate of ZOH, FOH and Tustin C2D of a descriptor: the
// model is nil only where, entry by entry, its DC gain ĥ, exact for its
// stored entries, lies within the descriptor bar (dcBar) of g0 = DCGain of
// the caller's delay-free model, plus the rounding ε·S of the result's own
// entries. With internal delays g0 is the DC gain of the closed DC pencil
// (lftClosedAtDC), certified by its own solve or else DCGain of the
// pencil, and the bar is that pencil's with each delay channel kept as its
// own block (channelBar); where C2, D21 and D22 are zero they are the
// delay-free core's.
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
	ref, barRef, channels := lftClosedAtDC(ioDelayFree(sys))
	n, m, p := ref.Dims()
	if m == 0 || p == 0 {
		return &holdGate{}
	}
	x0, e := 0.0, (*mat.Dense)(nil)
	if ref.IsDiscrete() {
		x0, e = 1, ref.E
	}
	var solved []dd
	var g0 *mat.Dense
	if n > 0 {
		if r := solveDCRegular(e, ref.A, ref.B, ref.C, ref.D, x0, n > exactDetMaxOrder); r.solveErr == nil && !r.overflow && r.certified {
			g0, solved = r.g, r.x
		}
	}
	if g0 == nil {
		var err error
		if g0, err = ref.DCGain(); err != nil {
			return &holdGate{err: holdUnverified("DCGain of the model: %v", err)}
		}
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
	gate.bar = newDCBarSolved(barRef, solved)
	if channels > 0 && gate.bar.reason == "" {
		gate.bar = channelBar(barRef, channels, gate.bar)
	}
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
	if g.err != nil {
		return g.err
	}
	if !g.gated {
		return nil
	}
	if err := finiteSystemMatrices("result", out, ErrOverflow); err != nil {
		return holdUnverified("%v", err)
	}
	if out.LFT != nil {
		out = delaysClosedAtDC(out)
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

// lftClosedAtDC returns the DC pencil of the delay-free sys or of sys with
// its internal delays closed, the identity at s = 0: the delayed signals w
// become algebraic states of 0 = A·x + B2·w + B·u,
// 0 = C2·x + (D22 − I)·w + D21·u, output C·x + D12·w + D·u. pencil holds it
// as the discrete descriptor E′ − A′ at z = 1, A′ = [A B2; C2 D22] and
// E′ = blkdiag(0, I), whose entries are stored ones, so its DC solve takes
// I − D22 without rounding; bar holds it as the continuous descriptor
// A′ − E′, E = blkdiag(E, 0), for its bar (channelBar), with channels the
// N delay channels, its last N states. Where C2, D21 and D22 are zero,
// w ≡ 0 and both are the delay-free core, with no channels.
func lftClosedAtDC(sys *System) (pencil, bar *System, channels int) {
	L := sys.LFT
	if L == nil {
		return sys, sys, 0
	}
	if allZeroDense(L.C2) && allZeroDense(L.D21) && allZeroDense(L.D22) {
		core := *sys
		core.LFT = nil
		return &core, &core, 0
	}
	n, m, p := sys.Dims()
	N := sys.internalDelayCount()
	A, B, C := newDense(n+N, n+N), newDense(n+N, m), newDense(p, n+N)
	E, Ec := newDense(n+N, n+N), newDense(n+N, n+N)
	if n > 0 {
		setBlock(A, 0, 0, sys.A)
		setBlock(A, 0, n, L.B2)
		setBlock(A, n, 0, L.C2)
		setBlock(B, 0, 0, sys.B)
		setBlock(C, 0, 0, sys.C)
		setBlock(Ec, 0, 0, sys.E)
	}
	setBlock(A, n, n, L.D22)
	setBlock(B, n, 0, L.D21)
	setBlock(C, 0, n, L.D12)
	Ac := denseCopy(A)
	for i := range N {
		E.Set(n+i, n+i, 1)
		Ac.Set(n+i, n+i, Ac.At(n+i, n+i)-1)
	}
	return &System{A: A, B: B, C: C, D: sys.D, E: E, Dt: 1}, &System{A: Ac, B: B, C: C, D: sys.D, E: Ec}, N
}

// channelBar is b, newDCBar of the continuous closed DC pencil bar of
// lftClosedAtDC, with the bar of each entry (i, j) taken with each of the
// last N states (delay channels) as its own block: dcBar's
// u·(ĉ·m̂·â·m̂·b̂ + ĉ·m̂·b̂ + |D_ij|) with the (1+N)-vectors ĉ = [‖C_i‖,
// |D12_ik|], b̂ = [‖B_j‖; |D21_kj|] and the matrices â = [‖A‖, ‖B2_·k‖;
// ‖C2_k·‖, |(D22 − I)_kl|], m̂ the same block norms of M = (−Ã)⁻¹,
// Frobenius norms. Rescaling a delay channel (w_k → α·w_k) or the state
// equation, both exact equivalences of the model, acts on them as a
// diagonal similarity that cancels; one pencil norm would not. With N = 0
// it is dcBar's.
func channelBar(bar *System, N int, b dcBar) dcBar {
	q, m, p := bar.Dims()
	n := q - N
	lu := make([]float64, q*q)
	ar := bar.A.RawMatrix()
	for i := range q {
		for j := range q {
			lu[i*q+j] = -ar.Data[i*ar.Stride+j]
		}
	}
	ipiv := make([]int, q)
	if !impl.Dgetrf(q, q, lu, q, ipiv) {
		b.reason = "DC pencil not invertible"
		return b
	}
	inv := make([]float64, q*q)
	for i := range q {
		inv[i*q+i] = 1
	}
	impl.Dgetrs(blas.NoTrans, q, q, lu, q, ipiv, inv, q)
	K := 1 + N
	blockNorms := func(data []float64, stride int) []float64 {
		out := make([]float64, K*K)
		for i := range q {
			bi := 0
			if i >= n {
				bi = 1 + i - n
			}
			for j := range q {
				bj := 0
				if j >= n {
					bj = 1 + j - n
				}
				v := data[i*stride+j]
				out[bi*K+bj] += v * v
			}
		}
		for k := range out {
			out[k] = math.Sqrt(out[k])
		}
		return out
	}
	ah, mh := blockNorms(ar.Data, ar.Stride), blockNorms(inv, q)
	br, cr := bar.B.RawMatrix(), bar.C.RawMatrix()
	ch, bh, r, x := make([]float64, K), make([]float64, K), make([]float64, K), make([]float64, K)
	b.bar, b.lu = make([]float64, p*m), nil
	for i := range p {
		for j := range m {
			clear(ch)
			clear(bh)
			for l := range q {
				bl := 0
				if l >= n {
					bl = 1 + l - n
				}
				ch[bl] += cr.Data[i*cr.Stride+l] * cr.Data[i*cr.Stride+l]
				bh[bl] += br.Data[l*br.Stride+j] * br.Data[l*br.Stride+j]
			}
			for k := range K {
				ch[k], bh[k] = math.Sqrt(ch[k]), math.Sqrt(bh[k])
			}
			for c := range K {
				r[c], x[c] = 0, 0
				for l := range K {
					r[c] += ch[l] * mh[l*K+c]
					x[c] += mh[c*K+l] * bh[l]
				}
			}
			t := b.absD[i*m+j]
			for c := range K {
				t += r[c] * bh[c]
				for l := range K {
					t += r[c] * ah[c*K+l] * x[l]
				}
			}
			v := b.u * t
			if math.IsNaN(v) || math.IsInf(v, 0) {
				b.reason = "DC bar not finite"
			}
			b.bar[i*m+j] = v
		}
	}
	return b
}

// delaysClosedAtDC returns the discrete out with its internal delays, the
// identity at z = 1, closed without rounding: the delayed signals w become
// states of x⁺ = A·x + B2·w + B·u, w⁺ = C2·x + D22·w + D21·u, output
// C·x + D12·w + D·u, whose DC gain, exact for the stored entries, is out's.
func delaysClosedAtDC(out *System) *System {
	n, m, p := out.Dims()
	L := out.LFT
	N := out.internalDelayCount()
	A, B, C := newDense(n+N, n+N), newDense(n+N, m), newDense(p, n+N)
	set := func(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
		if src == nil {
			return
		}
		rows, cols := src.Dims()
		for i := range rows {
			for j := range cols {
				dst.Set(r0+i, c0+j, src.At(i, j))
			}
		}
	}
	set(A, 0, 0, out.A)
	set(A, 0, n, L.B2)
	set(A, n, 0, L.C2)
	set(A, n, n, L.D22)
	set(B, 0, 0, out.B)
	set(B, n, 0, L.D21)
	set(C, 0, 0, out.C)
	set(C, 0, n, L.D12)
	return &System{A: A, B: B, C: C, D: out.D, Dt: out.Dt}
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
