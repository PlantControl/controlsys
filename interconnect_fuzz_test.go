package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Interconnection fuzz targets (Series, Parallel, Append, Feedback, LFT,
// Connect, ConnectByName with SumBlk). A second component is decoded from
// the same tuple (fzICPartner); v[15] is no internal delay here but a
// selector: its integer part picks the partner order, dimensions and
// channels, its fraction the evaluation points, and its sign requests a
// constructed algebraic loop.
//
// Oracle (fzIC): the stacked components G = blkdiag(Gi) close through a
// real coupling u = Q·y + v, H = Sout·(I − G·Q)⁻¹·G·Sin, evaluated exactly
// over big.Rat at complex points from the float64 component data. The
// library result's EvalFr must match within a first-order bound on the
// rounding of its realization (formed from the component data) and of its
// evaluation; points where that bound exceeds 1e-3·|H| are skipped.
// Well-posedness: ErrAlgebraicLoop is accepted iff det(I − D·Q) = 0
// exactly or the float loop matrix has cond·eps ≥ 1e-3; a constructed loop
// whose float-formed loop matrix is exactly singular must return it.
// p×0 stateless results are the documented ErrDimensionMismatch.
//
// Delays: each target reruns on the components with their I/O delays
// (fzDecode's v[12..14]) and checks the result against the exact
// composition of the delayed responses (checkDelayed). Components whose
// pole magnitudes span more than 1e12 are not rerun: internal-delay
// evaluation then returns 0 or Inf (an EDGE defect, not checked here).

func fzICSel(s *fzSpec) (q uint64, frac float64, loop bool) {
	a := fzClamp(math.Abs(s.Vals[15]), 0, 0x1p52)
	return uint64(a), a - math.Floor(a), s.Vals[15] < 0
}

// fzICPartner decodes the second component (order n2, m2 inputs, p2
// outputs) from a fixed permutation of the tuple, in the same domain and
// with the same sample time.
func fzICPartner(t *testing.T, s *fzSpec, n2, m2, p2 int) *fzSpec {
	t.Helper()
	v := s.Vals
	g := 12 - fzClamp(math.Abs(v[6]), 0, 12)
	if v[6] >= 0 {
		g = -g
	}
	w := [fzNVals]float64{
		v[4], v[5], v[0], v[1], v[2], v[3], g, 0.5 - 0.7*v[7],
		v[8], v[9], v[10] + 0.61, v[11] + 1.37, v[13], v[12], v[14], fzNoDelay,
	}
	kind := s.Kind&1 | uint8(m2-1)<<1 | uint8(p2-1)<<3 | uint8(n2)<<5
	return fzDecode(t, kind, w)
}

type fzIC struct {
	comps  []*fzSpec
	M, P   int
	Q      []float64
	mo, po int
	Sin    []float64
	Sout   []float64
}

func newFzIC(comps ...*fzSpec) *fzIC {
	ic := &fzIC{comps: comps}
	for _, c := range comps {
		ic.M += c.M
		ic.P += c.P
	}
	ic.Q = make([]float64, ic.M*ic.P)
	return ic
}

func (ic *fzIC) in(k, j int) int {
	o := 0
	for _, c := range ic.comps[:k] {
		o += c.M
	}
	return o + j
}

func (ic *fzIC) out(k, i int) int {
	o := 0
	for _, c := range ic.comps[:k] {
		o += c.P
	}
	return o + i
}

func (ic *fzIC) states() int {
	n := 0
	for _, c := range ic.comps {
		n += c.N
	}
	return n
}

// link adds g·y[from] to u[to] (stacked indices).
func (ic *fzIC) link(to, from int, g float64) { ic.Q[to*ic.P+from] += g }

// keep selects stacked inputs and outputs as the external channels.
func (ic *fzIC) keep(inputs, outputs []int) {
	ic.mo, ic.po = len(inputs), len(outputs)
	ic.Sin = make([]float64, ic.M*ic.mo)
	for k, j := range inputs {
		ic.Sin[j*ic.mo+k] = 1
	}
	ic.Sout = make([]float64, ic.po*ic.P)
	for k, i := range outputs {
		ic.Sout[k*ic.P+i] = 1
	}
}

func fzICRange(lo, n int) []int {
	r := make([]int, n)
	for k := range r {
		r[k] = lo + k
	}
	return r
}

// dblk returns the stacked feedthrough (P×M).
func (ic *fzIC) dblk() []float64 {
	d := make([]float64, ic.P*ic.M)
	r, c := 0, 0
	for _, s := range ic.comps {
		for i := range s.P {
			for j := range s.M {
				d[(r+i)*ic.M+c+j] = s.D[i*s.M+j]
			}
		}
		r += s.P
		c += s.M
	}
	return d
}

// blkSystem realizes blkdiag of the components directly.
func (ic *fzIC) blkSystem(t *testing.T, delays bool) *System {
	t.Helper()
	n := 0
	for _, s := range ic.comps {
		n += s.N
	}
	a := make([]float64, n*n)
	b := make([]float64, n*ic.M)
	c := make([]float64, ic.P*n)
	no, mo, po := 0, 0, 0
	for _, s := range ic.comps {
		for i := range s.N {
			for j := range s.N {
				a[(no+i)*n+no+j] = s.A[i*s.N+j]
			}
			for j := range s.M {
				b[(no+i)*ic.M+mo+j] = s.B[i*s.M+j]
			}
		}
		for i := range s.P {
			for j := range s.N {
				c[(po+i)*n+no+j] = s.C[i*s.N+j]
			}
		}
		no += s.N
		mo += s.M
		po += s.P
	}
	sys, err := NewFromSlices(n, ic.M, ic.P, a, b, c, ic.dblk(), ic.comps[0].Dt)
	if err != nil {
		t.Fatalf("NewFromSlices(blkdiag): %v", err)
	}
	if !delays {
		return sys
	}
	in := make([]float64, 0, ic.M)
	out := make([]float64, 0, ic.P)
	io := mat.NewDense(ic.P, ic.M, nil)
	r, k := 0, 0
	for _, s := range ic.comps {
		in = append(in, sliceOrZeros(s.In, s.M)...)
		out = append(out, sliceOrZeros(s.Out, s.P)...)
		for i := range s.P {
			for j := range s.M {
				if s.IO != nil {
					io.Set(r+i, k+j, s.IO[i*s.M+j])
				}
			}
		}
		r += s.P
		k += s.M
	}
	if err := sys.SetInputDelay(in); err != nil {
		t.Fatalf("SetInputDelay: %v", err)
	}
	if err := sys.SetOutputDelay(out); err != nil {
		t.Fatalf("SetOutputDelay: %v", err)
	}
	if err := sys.SetDelay(io); err != nil {
		t.Fatalf("SetDelay: %v", err)
	}
	return sys
}

func fzCOf(re, im float64) *fzCRat {
	z := new(fzCRat)
	z.re.Set(fzRat(re))
	z.im.Set(fzRat(im))
	return z
}

func fzCSubMulTo(dst, a, b *fzCRat) {
	p := fzCMul(a, b)
	dst.re.Sub(&dst.re, &p.re)
	dst.im.Sub(&dst.im, &p.im)
}

func fzCMatMul(r, k, c int, a, b []*fzCRat) []*fzCRat {
	out := make([]*fzCRat, r*c)
	for i := range r {
		for j := range c {
			acc := new(fzCRat)
			for l := range k {
				x, y := a[i*k+l], b[l*c+j]
				if x.isZero() || y.isZero() {
					continue
				}
				p := fzCMul(x, y)
				acc.re.Add(&acc.re, &p.re)
				acc.im.Add(&acc.im, &p.im)
			}
			out[i*c+j] = acc
		}
	}
	return out
}

func fzCReal(r, c int, x []float64) []*fzCRat {
	out := make([]*fzCRat, r*c)
	for k := range out {
		out[k] = fzCOf(x[k], 0)
	}
	return out
}

// fzCSolve solves the n×n system a·X = b (b n×k) exactly; ok is false
// when a is singular.
func fzCSolve(n int, a []*fzCRat, k int, b []*fzCRat) ([]*fzCRat, bool) {
	w := n + k
	M := make([]*fzCRat, n*w)
	for i := range n {
		for j := range n {
			z := new(fzCRat)
			z.re.Set(&a[i*n+j].re)
			z.im.Set(&a[i*n+j].im)
			M[i*w+j] = z
		}
		for j := range k {
			z := new(fzCRat)
			z.re.Set(&b[i*k+j].re)
			z.im.Set(&b[i*k+j].im)
			M[i*w+n+j] = z
		}
	}
	for col := range n {
		piv := -1
		for i := col; i < n; i++ {
			if !M[i*w+col].isZero() {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range w {
			M[col*w+j], M[piv*w+j] = M[piv*w+j], M[col*w+j]
		}
		for i := range n {
			if i == col || M[i*w+col].isZero() {
				continue
			}
			f := fzCQuo(M[i*w+col], M[col*w+col])
			for j := col; j < w; j++ {
				if !M[col*w+j].isZero() {
					fzCSubMulTo(M[i*w+j], f, M[col*w+j])
				}
			}
		}
	}
	X := make([]*fzCRat, n*k)
	for i := range n {
		for j := range k {
			X[i*k+j] = fzCQuo(M[i*w+n+j], M[i*w+i])
		}
	}
	return X, true
}

// fzICResp is the exact response C(zI − A)⁻¹B + D of a component.
func fzICResp(s *fzSpec, z complex128) ([]*fzCRat, bool) {
	G := fzCReal(s.P, s.M, s.D)
	if s.N == 0 {
		return G, true
	}
	zI := make([]*fzCRat, s.N*s.N)
	for i := range s.N {
		for j := range s.N {
			e := fzCOf(-s.A[i*s.N+j], 0)
			if i == j {
				e.re.Add(&e.re, fzRat(real(z)))
				e.im.Set(fzRat(imag(z)))
			}
			zI[i*s.N+j] = e
		}
	}
	X, ok := fzCSolve(s.N, zI, s.M, fzCReal(s.N, s.M, s.B))
	if !ok {
		return nil, false
	}
	CX := fzCMatMul(s.P, s.N, s.M, fzCReal(s.P, s.N, s.C), X)
	for k := range G {
		G[k].re.Add(&G[k].re, &CX[k].re)
		G[k].im.Add(&G[k].im, &CX[k].im)
	}
	return G, true
}

// exact returns H(z) rounded to complex128 (po×mo); ok is false when a
// component or the loop is singular at z.
func (ic *fzIC) exact(z complex128) ([]complex128, bool) {
	h, _, ok := ic.exactLoop(z, false)
	return h, ok
}

// exactLoop is exact with the component I/O delays applied when delays
// (factors rounded to complex128), also returning the loop matrix
// I − G·Q rounded (P×P).
func (ic *fzIC) exactLoop(z complex128, delays bool) ([]complex128, []complex128, bool) {
	G := make([]*fzCRat, ic.P*ic.M)
	for k := range G {
		G[k] = new(fzCRat)
	}
	r, c := 0, 0
	for _, s := range ic.comps {
		g, ok := fzICResp(s, z)
		if !ok {
			return nil, nil, false
		}
		for i := range s.P {
			for j := range s.M {
				e := g[i*s.M+j]
				if f := fzICDelayFactor(s, i, j, z); delays && f != 1 {
					e = fzCMul(e, fzCOf(real(f), imag(f)))
				}
				G[(r+i)*ic.M+c+j] = e
			}
		}
		r += s.P
		c += s.M
	}
	L := fzCMatMul(ic.P, ic.M, ic.P, G, fzCReal(ic.M, ic.P, ic.Q))
	for i := range ic.P {
		for j := range ic.P {
			e := L[i*ic.P+j]
			e.re.Neg(&e.re)
			e.im.Neg(&e.im)
			if i == j {
				e.re.Add(&e.re, big.NewRat(1, 1))
			}
		}
	}
	rhs := fzCMatMul(ic.P, ic.M, ic.mo, G, fzCReal(ic.M, ic.mo, ic.Sin))
	T, ok := fzCSolve(ic.P, L, ic.mo, rhs)
	if !ok {
		return nil, nil, false
	}
	H := fzCMatMul(ic.po, ic.P, ic.mo, fzCReal(ic.po, ic.P, ic.Sout), T)
	return fzCRound(H), fzCRound(L), true
}

func fzCRound(x []*fzCRat) []complex128 {
	out := make([]complex128, len(x))
	for k, h := range x {
		re, _ := h.re.Float64()
		im, _ := h.im.Float64()
		out[k] = complex(re, im)
	}
	return out
}

// fzICDelayFactor is the I/O delay factor of entry (i, j) of component c
// at z: e^{−zτ}, or z^{−τ} for discrete models.
func fzICDelayFactor(c *fzSpec, i, j int, z complex128) complex128 {
	var tau float64
	if c.In != nil {
		tau += c.In[j]
	}
	if c.Out != nil {
		tau += c.Out[i]
	}
	if c.IO != nil {
		tau += c.IO[i*c.M+j]
	}
	if tau == 0 {
		return 1
	}
	if c.Discrete {
		f := complex(1, 0)
		for range int(tau) {
			f /= z
		}
		return f
	}
	return cmplx.Exp(-z * complex(tau, 0))
}

// delayed reruns the operation on the components with their I/O delays
// (when any, and no component spans more than 1e12 in pole magnitude) and
// checks it with checkDelayed.
func (ic *fzIC) delayed(t *testing.T, s *fzSpec, op string, frac float64, resDF *System, errDF error, skip bool,
	run func(d []*System) (*System, error)) {
	t.Helper()
	if skip || errDF != nil || !ic.hasDelays() {
		return
	}
	for _, c := range ic.comps {
		if c.NormJ*c.CondT > 1e12 {
			return
		}
	}
	d := make([]*System, len(ic.comps))
	for k, c := range ic.comps {
		d[k] = c.System(t, fzSysOpts{Delays: true})
	}
	res, err := run(d)
	ic.checkDelayed(t, s, op, frac, resDF, res, err)
}

func (ic *fzIC) hasDelays() bool {
	for _, c := range ic.comps {
		if c.In != nil || c.Out != nil || c.IO != nil {
			return true
		}
	}
	return false
}

// checkDelayed compares the operation on the delayed components (res, err)
// with the exact composition of the delayed responses (delay factors
// rounded), at points where the delay-free result resDF is accurate (bound
// ≤ 1e-8·|H|) and the delayed loop has condition number kd ≤ 1e6. The
// tolerance is fzICLFTBound on the result's own internal-delay realization,
// amplified by the static loop condition, plus kd times the rounding of
// the delay factors, both scaled by 1 + |z|·τmax for the rounding of the
// phase z·τ; points where it exceeds 1e-3·|H| are skipped.
func (ic *fzIC) checkDelayed(t *testing.T, s *fzSpec, op string, frac float64, resDF, res *System, err error) {
	t.Helper()
	op += "(delayed)"
	fzContract(t, s, op, res != nil, err)
	if err != nil {
		if _, cond := ic.loopDet(); errors.Is(err, ErrAlgebraicLoop) && cond*0x1p-52 >= 1e-6 {
			return
		}
		t.Fatalf("%s: %v, delay-free succeeded\n%s", op, err, s)
	}
	_, mr, pr := res.Dims()
	if mr != ic.mo || pr != ic.po {
		t.Fatalf("%s: result %d×%d, want %d×%d\n%s", op, pr, mr, ic.po, ic.mo, s)
	}
	if mr == 0 || pr == 0 {
		return
	}
	var tmax float64
	for _, c := range ic.comps {
		for i := range c.P {
			for j := range c.M {
				f := 0.0
				if c.In != nil {
					f += c.In[j]
				}
				if c.Out != nil {
					f += c.Out[i]
				}
				if c.IO != nil {
					f += c.IO[i*c.M+j]
				}
				tmax = math.Max(tmax, f)
			}
		}
	}
	for _, z := range fzICPoints(frac, ic.comps...) {
		if !ic.comps[0].Discrete && math.Abs(real(z))*tmax > 5 {
			continue
		}
		h0, ok := ic.exact(z)
		if !ok || !(ic.bound(resDF, z) <= 1e-8*fzCFrob(h0)) {
			continue
		}
		hd, L, ok := ic.exactLoop(z, true)
		if !ok {
			continue
		}
		Li, ok := fzCInv(ic.P, L)
		if ic.P > 0 && !ok {
			continue
		}
		kd := 1.0
		if ic.P > 0 {
			kd = fzCFrob(L) * fzCFrob(Li)
		}
		if kd > 1e6 {
			continue
		}
		const eps = 0x1p-52
		phase := 1 + cmplx.Abs(z)*tmax
		tol := fzICLFTBound(res, z)*ic.loopCond()*phase + 1e3*eps*kd*phase*(fzCFrob(hd)+fzCFrob(h0))
		if !(tol <= 1e-3*fzCFrob(hd)) {
			continue
		}
		g, err := res.EvalFr(z)
		if err != nil {
			t.Fatalf("%s EvalFr(%v): %v\n%s", op, z, err, s)
		}
		for i := range pr {
			for j := range mr {
				w := hd[i*mr+j]
				if d := cmplx.Abs(g[i][j] - w); !(d <= tol) {
					t.Fatalf("%s: H[%d,%d](%v) = %v, exact %v (|err| %.3g, tol %.3g, delay-free %v)\n%s",
						op, i, j, z, g[i][j], w, d, tol, h0[i*mr+j], s)
				}
			}
		}
	}
}

// loopDet returns det(I − D·Q) exactly and the 2-norm condition number of
// the float-formed loop matrix.
func (ic *fzIC) loopDet() (*big.Rat, float64) {
	D := ic.dblk()
	R := make([]*big.Rat, ic.P*ic.P)
	F := mat.NewDense(max(ic.P, 1), max(ic.P, 1), nil)
	for i := range ic.P {
		for j := range ic.P {
			acc := new(big.Rat)
			var fl float64
			for k := range ic.M {
				if q := ic.Q[k*ic.P+j]; q != 0 {
					acc.Add(acc, new(big.Rat).Mul(fzRat(D[i*ic.M+k]), fzRat(q)))
					fl += D[i*ic.M+k] * q
				}
			}
			acc.Neg(acc)
			fl = -fl
			if i == j {
				acc.Add(acc, big.NewRat(1, 1))
				fl++
			}
			R[i*ic.P+j] = acc
			F.Set(i, j, fl)
		}
	}
	if ic.P == 0 {
		return big.NewRat(1, 1), 1
	}
	_, det := fzRatElim(R, ic.P, ic.P)
	return det, mat.Cond(F, 2)
}

func fzDenseData(m *mat.Dense, r, c int) []float64 {
	out := make([]float64, r*c)
	if m == nil || r == 0 || c == 0 {
		return out
	}
	for i := range r {
		for j := range c {
			out[i*c+j] = m.At(i, j)
		}
	}
	return out
}

func fzCFrob(x []complex128) float64 {
	var s float64
	for _, v := range x {
		s = math.Hypot(s, cmplx.Abs(v))
	}
	return s
}

// fzCInv inverts the n×n complex matrix by Gauss–Jordan with partial
// pivoting (a bound estimate only).
func fzCInv(n int, a []complex128) ([]complex128, bool) {
	M := slices.Clone(a)
	X := make([]complex128, n*n)
	for i := range n {
		X[i*n+i] = 1
	}
	for col := range n {
		piv, best := -1, 0.0
		for i := col; i < n; i++ {
			if v := cmplx.Abs(M[i*n+col]); v > best {
				piv, best = i, v
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range n {
			M[col*n+j], M[piv*n+j] = M[piv*n+j], M[col*n+j]
			X[col*n+j], X[piv*n+j] = X[piv*n+j], X[col*n+j]
		}
		inv := 1 / M[col*n+col]
		for j := range n {
			M[col*n+j] *= inv
			X[col*n+j] *= inv
		}
		for i := range n {
			if i == col || M[i*n+col] == 0 {
				continue
			}
			f := M[i*n+col]
			for j := range n {
				M[i*n+j] -= f * M[col*n+j]
				X[i*n+j] -= f * X[col*n+j]
			}
		}
	}
	for _, v := range X {
		if cmplx.IsNaN(v) || cmplx.IsInf(v) {
			return nil, false
		}
	}
	return X, true
}

func fzCMulF(r, k, c int, a []complex128, b []float64, bFirst bool) []complex128 {
	out := make([]complex128, r*c)
	for i := range r {
		for j := range c {
			var acc complex128
			for l := range k {
				if bFirst {
					acc += complex(b[i*k+l], 0) * a[l*c+j]
				} else {
					acc += a[i*k+l] * complex(b[l*c+j], 0)
				}
			}
			out[i*c+j] = acc
		}
	}
	return out
}

// loopCond is 1 + cond(I − D·Q) (Frobenius), the amplification of
// feedthrough-loop rounding in the formed realization.
func (ic *fzIC) loopCond() float64 {
	if ic.P == 0 {
		return 1
	}
	D := ic.dblk()
	L := make([]complex128, ic.P*ic.P)
	for i := range ic.P {
		L[i*ic.P+i] = 1
		for j := range ic.P {
			for k := range ic.M {
				L[i*ic.P+j] -= complex(D[i*ic.M+k]*ic.Q[k*ic.P+j], 0)
			}
		}
	}
	Li, ok := fzCInv(ic.P, L)
	if !ok {
		return math.Inf(1)
	}
	return 1 + fzCFrob(L)*fzCFrob(Li)
}

// fzICLFTBound bounds the error of evaluating res at z through its own
// realization: the augmented P(z) = [C; C2](zE−A)⁻¹[B B2] + [D D12; D21 D22]
// carries an absolute error eP of a backward-stable solve, which the
// internal-delay closure H = P11 + P12·Δ(I − P22·Δ)⁻¹·P21 amplifies by
// (1 + ‖P12·Δ·M‖)(1 + ‖Δ·M·P21‖), M = (I − P22·Δ)⁻¹; boundary I/O delay
// factors scale it by their largest modulus.
func fzICLFTBound(res *System, z complex128) float64 {
	const eps = 0x1p-52
	n, m, p := res.Dims()
	N := 0
	var tau []float64
	if res.LFT != nil {
		tau = res.LFT.Tau
		N = len(tau)
	}
	mt, pt := m+N, p+N
	A := fzDenseData(res.A, n, n)
	Bt := make([]float64, n*mt)
	Ct := make([]float64, pt*n)
	Dt := make([]float64, pt*mt)
	put := func(dst []float64, w, r0, c0 int, src *mat.Dense, r, c int) {
		x := fzDenseData(src, r, c)
		for i := range r {
			for j := range c {
				dst[(r0+i)*w+c0+j] = x[i*c+j]
			}
		}
	}
	put(Bt, mt, 0, 0, res.B, n, m)
	put(Ct, n, 0, 0, res.C, p, n)
	put(Dt, mt, 0, 0, res.D, p, m)
	if N > 0 {
		put(Bt, mt, 0, m, res.LFT.B2, n, N)
		put(Ct, n, p, 0, res.LFT.C2, N, n)
		put(Dt, mt, 0, m, res.LFT.D12, p, N)
		put(Dt, mt, p, 0, res.LFT.D21, N, m)
		put(Dt, mt, p, m, res.LFT.D22, N, N)
	}
	c := 1e3 * float64(n+mt+pt+1) * eps
	P := make([]complex128, pt*mt)
	for k, d := range Dt {
		P[k] = complex(d, 0)
	}
	eP := c * fzFrob(Dt)
	if n > 0 {
		E := make([]float64, n*n)
		if res.E != nil {
			E = fzDenseData(res.E, n, n)
		} else {
			for i := range n {
				E[i*n+i] = 1
			}
		}
		Z := make([]complex128, n*n)
		for i := range Z {
			Z[i] = z*complex(E[i], 0) - complex(A[i], 0)
		}
		R, ok := fzCInv(n, Z)
		if !ok {
			return math.Inf(1)
		}
		RB := fzCMulF(n, n, mt, R, Bt, false)
		CR := fzCMulF(pt, n, n, R, Ct, true)
		CRB := fzCMulF(pt, n, mt, RB, Ct, true)
		for k := range P {
			P[k] += CRB[k]
		}
		eP += c * (fzCFrob(CR)*(fzFrob(A)+cmplx.Abs(z))*fzCFrob(RB) + fzFrob(Ct)*fzCFrob(RB))
	}
	f := func(tau float64) complex128 {
		if res.Dt > 0 {
			return cmplx.Pow(z, complex(-tau, 0))
		}
		return cmplx.Exp(-z * complex(tau, 0))
	}
	amp := 1.0
	if N > 0 {
		L := make([]complex128, N*N)
		for i := range N {
			for j := range N {
				L[i*N+j] = -P[(p+i)*mt+m+j] * f(tau[j])
			}
			L[i*N+i] += 1
		}
		M, ok := fzCInv(N, L)
		if !ok {
			return math.Inf(1)
		}
		DM := make([]complex128, N*N)
		for i := range N {
			for j := range N {
				DM[i*N+j] = f(tau[i]) * M[i*N+j]
			}
		}
		var a, b float64
		for i := range p {
			for j := range N {
				var acc complex128
				for k := range N {
					acc += P[i*mt+m+k] * DM[k*N+j]
				}
				a = math.Hypot(a, cmplx.Abs(acc))
			}
		}
		for i := range N {
			for j := range m {
				var acc complex128
				for k := range N {
					acc += DM[i*N+k] * P[(p+k)*mt+j]
				}
				b = math.Hypot(b, cmplx.Abs(acc))
			}
		}
		amp = (1 + a) * (1 + b)
	}
	io := 1.0
	for _, d := range slices.Concat(res.InputDelay, res.OutputDelay, fzDenseData(res.Delay, p, m)) {
		io = math.Max(io, cmplx.Abs(f(d)))
	}
	if res.Delay != nil || res.InputDelay != nil && res.OutputDelay != nil {
		io *= io
	}
	return eP * amp * io
}

// bound is a first-order bound on |EvalFr(res, z) − H(z)|: the realization
// formed from the component data, u = Q·E·(C·x + D·Sin·v) with
// E = (I − D·Q)⁻¹, carries entry errors of order eps times the factor
// norms; their effect and that of evaluating res at z go through
// ‖C(zE−A)⁻¹‖ and ‖(zE−A)⁻¹B‖ of the result.
func (ic *fzIC) bound(res *System, z complex128) float64 {
	const eps = 0x1p-52
	nr, mr, pr := res.Dims()
	c := 1e3 * float64(nr+ic.M+ic.P+1) * eps
	var nA, nB, nC float64
	for _, s := range ic.comps {
		nA = math.Hypot(nA, fzFrob(s.A))
		nB = math.Hypot(nB, fzFrob(s.B))
		nC = math.Hypot(nC, fzFrob(s.C))
	}
	D := ic.dblk()
	nD := fzFrob(D)
	nQ, nSin, nSout := fzFrob(ic.Q), fzFrob(ic.Sin), fzFrob(ic.Sout)
	nE, nL := 1.0, 1.0
	if ic.P > 0 {
		L := mat.NewDense(ic.P, ic.P, nil)
		for i := range ic.P {
			L.Set(i, i, 1)
			for j := range ic.P {
				for k := range ic.M {
					L.Set(i, j, L.At(i, j)-D[i*ic.M+k]*ic.Q[k*ic.P+j])
				}
			}
		}
		var Li mat.Dense
		if err := Li.Inverse(L); err != nil {
			return math.Inf(1)
		}
		nE, nL = mat.Norm(&Li, 2), mat.Norm(L, 2)
	}
	k := 1 + nL*nE
	dA := c * (nA + nB*nQ*nE*nC*k)
	dB := c * nB * nSin * (1 + nQ*nE*nD) * k
	dC := c * nSout * nE * nC * k
	dD := c * nSout * nE * nD * nSin * k

	Ar := fzDenseData(res.A, nr, nr)
	Br := fzDenseData(res.B, nr, mr)
	Cr := fzDenseData(res.C, pr, nr)
	Dr := fzDenseData(res.D, pr, mr)
	var nCR, nRB float64
	if nr > 0 {
		Er := make([]float64, nr*nr)
		if res.E != nil {
			Er = fzDenseData(res.E, nr, nr)
		} else {
			for i := range nr {
				Er[i*nr+i] = 1
			}
		}
		Z := make([]complex128, nr*nr)
		for i := range Z {
			Z[i] = z*complex(Er[i], 0) - complex(Ar[i], 0)
		}
		R, ok := fzCInv(nr, Z)
		if !ok {
			return math.Inf(1)
		}
		nCR = fzCFrob(fzCMulF(pr, nr, nr, R, Cr, true))
		nRB = fzCFrob(fzCMulF(nr, nr, mr, R, Br, false))
	}
	eval := c * (nCR*(fzFrob(Ar)+cmplx.Abs(z))*nRB + fzFrob(Cr)*nRB + fzFrob(Dr))
	return nCR*dA*nRB + nCR*dB + dC*nRB + dD + eval
}

// fzICPoints returns evaluation points: off-axis and imaginary-axis points
// at the scale of the intended poles (continuous) or points off the unit
// circle (discrete).
func fzICPoints(frac float64, comps ...*fzSpec) []complex128 {
	if comps[0].Discrete {
		a := 0.4 + 2.2*frac
		return []complex128{
			cmplx.Rect(1.07, a), cmplx.Rect(0.9, 1.3+frac), cmplx.Rect(1.31, 0.2+0.5*frac),
		}
	}
	var lg float64
	k := 0
	for _, c := range comps {
		for _, e := range c.S {
			if a := cmplx.Abs(e); a > 0 {
				lg += math.Log(a)
				k++
			}
		}
	}
	w := 1.0
	if k > 0 {
		w = math.Exp(lg / float64(k))
	}
	w *= math.Pow(10, 2*frac-1)
	return []complex128{complex(0.37*w, w), complex(0, 1.9*w), complex(-0.23*w, 0.71*w)}
}

type fzICCase struct {
	op       string
	states   int // expected order; -1 allows a structurally reduced one
	mustLoop bool
}

// check verifies res/err from the operation against the oracle.
func (ic *fzIC) check(t *testing.T, s *fzSpec, c fzICCase, frac float64, res *System, err error) {
	t.Helper()
	fzContract(t, s, c.op, res != nil, err)
	det, cond := ic.loopDet()
	const eps = 0x1p-52
	if err != nil {
		if errors.Is(err, ErrAlgebraicLoop) {
			if det.Sign() == 0 || cond*eps >= 1e-3 {
				fzSkip(t, c.op+": algebraic loop")
			}
			t.Fatalf("%s: ErrAlgebraicLoop on a well-posed loop (det %s, cond %.3g): %v\n%s", c.op, det.FloatString(6), cond, err, s)
		}
		if errors.Is(err, ErrDimensionMismatch) && (ic.mo == 0) != (ic.po == 0) && ic.states() == 0 {
			fzSkip(t, c.op+": p×0 static gain")
		}
		fzErr(t, s, c.op, err)
	}
	if c.mustLoop {
		t.Fatalf("%s: float loop matrix exactly singular, no ErrAlgebraicLoop\n%s", c.op, s)
	}
	if det.Sign() == 0 {
		fzSkip(t, c.op+": ill-posed within rounding")
	}
	nr, mr, pr := res.Dims()
	if mr != ic.mo || pr != ic.po {
		t.Fatalf("%s: result %d×%d, want %d×%d\n%s", c.op, pr, mr, ic.po, ic.mo, s)
	}
	want := ic.states()
	if c.states >= 0 && nr != c.states || c.states < 0 && nr > want {
		t.Fatalf("%s: %d states, want %d\n%s", c.op, nr, want, s)
	}
	if mr == 0 || pr == 0 {
		return
	}
	checked := 0
	for _, z := range fzICPoints(frac, ic.comps...) {
		h, ok := ic.exact(z)
		if !ok {
			continue
		}
		b := ic.bound(res, z)
		hn := fzCFrob(h)
		if !(b <= 1e-3*hn) && !(hn == 0 && b == 0) {
			continue
		}
		g, err := res.EvalFr(z)
		fzErr(t, s, c.op+" EvalFr", err)
		for i := range pr {
			for j := range mr {
				w := h[i*mr+j]
				if d := cmplx.Abs(g[i][j] - w); !(d <= b+1e3*eps*cmplx.Abs(w)) {
					t.Fatalf("%s: H[%d,%d](%v) = %v, exact %v (|err| %.3g, bound %.3g)\n%s\nres: A=%v\nB=%v\nC=%v\nD=%v",
						c.op, i, j, z, g[i][j], w, d, b, s, fzDenseData(res.A, nr, nr), fzDenseData(res.B, nr, mr),
						fzDenseData(res.C, pr, nr), fzDenseData(res.D, pr, mr))
				}
			}
		}
		checked++
	}
	if checked == 0 {
		fzSkip(t, c.op+": ill-conditioned interconnection")
	}
}

// fzICMask returns the set bits of mask (taken mod 2^n, empty → all) in
// order, reversed when rev.
func fzICMask(mask uint64, n int, rev bool) []int {
	mask &= 1<<n - 1
	if mask == 0 {
		mask = 1<<n - 1
	}
	var out []int
	for k := range n {
		if mask&(1<<k) != 0 {
			out = append(out, k)
		}
	}
	if rev {
		slices.Reverse(out)
	}
	return out
}

// FuzzSeries: Series(sys1, sys2) = G2·G1 with n1+n2 states.
func FuzzSeries(f *testing.F) {
	fzICAddSeeds(f, "Series")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, _ := fzICSel(s)
		s2 := fzICPartner(t, s, int(q%7), s.P, 1+int(q/7%3))
		ic := newFzIC(s, s2)
		for j := range s.P {
			ic.link(ic.in(1, j), ic.out(0, j), 1)
		}
		ic.keep(fzICRange(0, s.M), fzICRange(ic.out(1, 0), s2.P))
		res, err := Series(s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{}))
		ic.delayed(t, s, "Series", frac, res, err, false, func(d []*System) (*System, error) { return Series(d[0], d[1]) })
		ic.check(t, s, fzICCase{op: "Series", states: s.N + s2.N}, frac, res, err)
	})
}

// FuzzParallel: Parallel(sys1, sys2) = G1 + G2 with n1+n2 states.
func FuzzParallel(f *testing.F) {
	fzICAddSeeds(f, "Parallel")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, _ := fzICSel(s)
		s2 := fzICPartner(t, s, int(q%7), s.M, s.P)
		ic := newFzIC(s, s2)
		ic.keep(fzICRange(0, s.M), fzICRange(0, s.P))
		for j := range s.M {
			ic.Sin[ic.in(1, j)*ic.mo+j] = 1
		}
		for i := range s.P {
			ic.Sout[i*ic.P+ic.out(1, i)] = 1
		}
		res, err := Parallel(s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{}))
		ic.delayed(t, s, "Parallel", frac, res, err, false, func(d []*System) (*System, error) { return Parallel(d[0], d[1]) })
		ic.check(t, s, fzICCase{op: "Parallel", states: s.N + s2.N}, frac, res, err)
	})
}

// FuzzAppend: Append(sys1, sys2) = blkdiag(G1, G2).
func FuzzAppend(f *testing.F) {
	fzICAddSeeds(f, "Append")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, _ := fzICSel(s)
		s2 := fzICPartner(t, s, int(q%7), 1+int(q/7%3), 1+int(q/21%3))
		ic := newFzIC(s, s2)
		ic.keep(fzICRange(0, ic.M), fzICRange(0, ic.P))
		res, err := Append(s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{}))
		ic.delayed(t, s, "Append", frac, res, err, false, func(d []*System) (*System, error) { return Append(d[0], d[1]) })
		ic.check(t, s, fzICCase{op: "Append", states: s.N + s2.N}, frac, res, err)
	})
}

// fzICSingular makes the static part of the loop exactly singular in
// float: D2 is zero but for one entry c = 1/(sign·d) against the largest
// entry d of the D1 block the loop sees. It reports whether
// fl(sign·d·c) = 1, i.e. the float-formed I − sign·D1·D2 is singular.
func fzICSingular(d1 func(a, b int) float64, na, nb int, sign float64, set func(b, a int, c float64)) bool {
	ba, bb, best := -1, -1, 0.0
	for a := range na {
		for b := range nb {
			if v := math.Abs(d1(a, b)); v > best {
				ba, bb, best = a, b, v
			}
		}
	}
	if ba < 0 {
		return false
	}
	d := d1(ba, bb)
	c := 1 / (sign * d)
	set(bb, ba, c)
	return sign*(d*c) == 1
}

// FuzzFeedback: Feedback(plant, controller, sign[, channels]) closes
// u[in[k]] = v + sign·y2[k], u2 = y1[out]; a nil controller closes unit
// feedback. A negative selector zeroes D2 but one entry so that
// I − sign·D1·D2 is exactly singular in float (must be ErrAlgebraicLoop).
func FuzzFeedback(f *testing.F) {
	fzICAddSeeds(f, "Feedback")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, loop := fzICSel(s)
		n2 := int(q % 7)
		q /= 7
		sign := 1.0
		if q&1 == 0 {
			sign = -1
		}
		mode := int(q >> 1 % 3)
		q >>= 1
		q /= 3
		plant := s.System(t, fzSysOpts{})
		if mode == 2 && s.M == s.P {
			ic := newFzIC(s)
			for i := range s.M {
				ic.link(i, i, sign)
			}
			ic.keep(fzICRange(0, s.M), fzICRange(0, s.P))
			res, err := Feedback(plant, nil, sign)
			ic.delayed(t, s, "Feedback(unit)", frac, res, err, false, func(d []*System) (*System, error) { return Feedback(d[0], nil, sign) })
			ic.check(t, s, fzICCase{op: "Feedback(unit)", states: s.N}, frac, res, err)
			return
		}
		in, out := fzICRange(0, s.M), fzICRange(0, s.P)
		var ch []FeedbackChannels
		if mode == 1 {
			in = fzICMask(q, s.M, q>>3&1 != 0)
			out = fzICMask(q>>4, s.P, q>>7&1 != 0)
			ch = []FeedbackChannels{{FeedIn: in, FeedOut: out}}
		}
		s2 := fzICPartner(t, s, n2, len(out), len(in))
		must := false
		if loop {
			clear(s2.D)
			must = fzICSingular(func(a, b int) float64 { return s.D[out[a]*s.M+in[b]] }, len(out), len(in), sign,
				func(b, a int, c float64) { s2.D[b*s2.M+a] = c })
		}
		ic := newFzIC(s, s2)
		for k, j := range in {
			ic.link(ic.in(0, j), ic.out(1, k), sign)
		}
		for k, i := range out {
			ic.link(ic.in(1, k), ic.out(0, i), 1)
		}
		ic.keep(fzICRange(0, s.M), fzICRange(0, s.P))
		res, err := Feedback(plant, s2.System(t, fzSysOpts{}), sign, ch...)
		ic.delayed(t, s, "Feedback", frac, res, err, loop, func(d []*System) (*System, error) { return Feedback(d[0], d[1], sign, ch...) })
		ic.check(t, s, fzICCase{op: "Feedback", states: s.N + s2.N, mustLoop: must}, frac, res, err)
	})
}

// FuzzLFT: LFT(sys1, sys2[, {Nu, Ny}]) is the star product: u1[m1−nu+k] =
// y2[k], u2[k] = y1[p1−ny+k]; external [w1; w2] → [z1; z2]. Without fb the
// MATLAB dimension rule picks lower/upper or ErrDimensionMismatch.
func FuzzLFT(f *testing.F) {
	fzICAddSeeds(f, "LFT")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, loop := fzICSel(s)
		n2, m2, p2 := int(q%7), 1+int(q/7%3), 1+int(q/21%3)
		q /= 63
		m1, p1 := s.M, s.P
		var nu, ny int
		var fb []LFTFeedback
		switch {
		case q&1 != 0:
			nu, ny = int(q>>1)%(min(m1, p2)+1), int(q>>3)%(min(p1, m2)+1)
			fb = []LFTFeedback{{Nu: nu, Ny: ny}}
		case m2 < m1 && p2 < p1:
			nu, ny = p2, m2
		case m1 < m2 && p1 < p2:
			nu, ny = m1, p1
		default:
			s2 := fzICPartner(t, s, n2, m2, p2)
			res, err := LFT(s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{}))
			if !errors.Is(err, ErrDimensionMismatch) || res != nil {
				t.Fatalf("LFT %d×%d with %d×%d: got %v, %v; want ErrDimensionMismatch\n%s", p1, m1, p2, m2, res, err, s)
			}
			return
		}
		s2 := fzICPartner(t, s, n2, m2, p2)
		must := false
		if loop && nu > 0 && ny > 0 {
			clear(s2.D)
			must = fzICSingular(func(l, k int) float64 { return s.D[(p1-ny+l)*m1+m1-nu+k] }, ny, nu, 1,
				func(k, l int, c float64) { s2.D[k*m2+l] = c })
		}
		ic := newFzIC(s, s2)
		for k := range nu {
			ic.link(ic.in(0, m1-nu+k), ic.out(1, k), 1)
		}
		for k := range ny {
			ic.link(ic.in(1, k), ic.out(0, p1-ny+k), 1)
		}
		ic.keep(append(fzICRange(0, m1-nu), fzICRange(ic.in(1, ny), m2-ny)...),
			append(fzICRange(0, p1-ny), fzICRange(ic.out(1, nu), p2-nu)...))
		res, err := LFT(s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{}), fb...)
		ic.delayed(t, s, "LFT", frac, res, err, loop, func(d []*System) (*System, error) { return LFT(d[0], d[1], fb...) })
		ic.check(t, s, fzICCase{op: "LFT", states: -1, mustLoop: must}, frac, res, err)
	})
}

// FuzzConnect: Connect on blkdiag(G1, G2) with up to three junction terms
// u_i += ±y_j and selected external channels. A negative selector sets
// D1[0,0] = 1 and the single term u_0 += y_0, an exactly singular loop.
func FuzzConnect(f *testing.F) {
	fzICAddSeeds(f, "Connect")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, loop := fzICSel(s)
		s2 := fzICPartner(t, s, int(q%7), 1+int(q/7%3), 1+int(q/21%3))
		q /= 63
		ic := newFzIC(s, s2)
		var js []Junction
		if loop {
			s.D[0] = 1
			js = []Junction{{Input: 0, Plus: []int{0}}}
			ic.link(0, 0, 1)
		} else {
			for range q % 4 {
				q /= 4
				i, j := int(q%uint64(ic.M)), int(q/7%uint64(ic.P))
				q /= 49
				if q&1 != 0 {
					js = append(js, Junction{Input: i, Minus: []int{j}})
					ic.link(i, j, -1)
				} else {
					js = append(js, Junction{Input: i, Plus: []int{j}})
					ic.link(i, j, 1)
				}
				q >>= 1
			}
		}
		inputs := fzICMask(q, ic.M, q>>6&1 != 0)
		outputs := fzICMask(q>>7, ic.P, q>>13&1 != 0)
		ic.keep(inputs, outputs)
		res, err := Connect(ic.blkSystem(t, false), js, inputs, outputs)
		ic.delayed(t, s, "Connect", frac, res, err, loop, func([]*System) (*System, error) {
			return Connect(ic.blkSystem(t, true), js, inputs, outputs)
		})
		ic.check(t, s, fzICCase{op: "Connect", states: -1, mustLoop: loop}, frac, res, err)
	})
}

func fzICNames(base string, w int) []string {
	if w == 1 {
		return []string{base}
	}
	out := make([]string, w)
	for k := range out {
		out[k] = base + "(" + string(rune('1'+k)) + ")"
	}
	return out
}

var fzSumBlks = []struct {
	expr string
	ins  []string
	coef []float64
}{
	{"e = r - y", []string{"r", "y"}, []float64{1, -1}},
	{"e = r + y", []string{"r", "y"}, []float64{1, 1}},
	{"e = -y + r", []string{"y", "r"}, []float64{-1, 1}},
	{"e = r - 2*y", []string{"r", "y"}, []float64{1, -2}},
	{"e = 0.5*r - y + d", []string{"r", "y", "d"}, []float64{0.5, -1, 1}},
	{"e = r - y - y", []string{"r", "y"}, []float64{1, -2}},
}

// FuzzConnectByName: SumBlk junction e = Σ coef·signal (checked against
// its table), plant G (u → y) and controller K (e → u) joined by name;
// external r (and d) → y (and e).
func FuzzConnectByName(f *testing.F) {
	fzICAddSeeds(f, "ConnectByName")
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		q, frac, _ := fzICSel(s)
		s2 := fzICPartner(t, s, int(q%7), s.P, s.M)
		q /= 7
		sb := fzSumBlks[q%uint64(len(fzSumBlks))]
		q /= uint64(len(fzSumBlks))
		w := s.P
		sum, err := SumBlk(sb.expr, w)
		fzContract(t, s, "SumBlk", sum != nil, err)
		fzErr(t, s, "SumBlk", err)
		ss := &fzSpec{M: w * len(sb.ins), P: w, Dt: s.Dt, Discrete: s.Discrete}
		ss.D = make([]float64, ss.P*ss.M)
		var inNames []string
		for k, name := range sb.ins {
			inNames = append(inNames, fzICNames(name, w)...)
			for i := range w {
				ss.D[i*ss.M+k*w+i] = sb.coef[k]
			}
		}
		if n, m, p := sum.Dims(); n != 0 || m != ss.M || p != ss.P ||
			!slices.Equal(sum.InputName, inNames) || !slices.Equal(sum.OutputName, fzICNames("e", w)) ||
			!slices.Equal(fzDenseData(sum.D, p, m), ss.D) {
			t.Fatalf("SumBlk(%q, %d) = n=%d %v→%v D=%v, want %v→%v D=%v", sb.expr, w, n, sum.InputName, sum.OutputName,
				fzDenseData(sum.D, p, m), inNames, fzICNames("e", w), ss.D)
		}
		sum.Dt = s.Dt
		G, K := s.System(t, fzSysOpts{}), s2.System(t, fzSysOpts{})
		G.InputName, G.OutputName = fzICNames("u", s.M), fzICNames("y", s.P)
		K.InputName, K.OutputName = fzICNames("e", w), fzICNames("u", s.M)
		var conns []Connection
		for _, base := range []string{"y", "e", "u"} {
			for _, nm := range fzICNames(base, map[string]int{"y": s.P, "e": w, "u": s.M}[base]) {
				conns = append(conns, Connection{From: nm, To: nm})
			}
		}
		ic := newFzIC(s, s2, ss)
		for j := range s.M {
			ic.link(ic.in(0, j), ic.out(1, j), 1)
		}
		for j := range w {
			ic.link(ic.in(1, j), ic.out(2, j), 1)
		}
		var extIn, extInNames []string
		var keepIn []int
		for k, name := range sb.ins {
			if name == "y" {
				for i := range w {
					ic.link(ic.in(2, k*w+i), ic.out(0, i), 1)
				}
				continue
			}
			extIn = append(extIn, fzICNames(name, w)...)
			keepIn = append(keepIn, fzICRange(ic.in(2, k*w), w)...)
		}
		extInNames = extIn
		extOut := fzICNames("y", s.P)
		keepOut := fzICRange(0, s.P)
		if q&1 != 0 {
			extOut = append(extOut, fzICNames("e", w)...)
			keepOut = append(keepOut, fzICRange(ic.out(2, 0), w)...)
		}
		ic.keep(keepIn, keepOut)
		res, err := ConnectByName([]*System{G, K, sum}, conns, extInNames, extOut)
		ic.delayed(t, s, "ConnectByName", frac, res, err, false, func(d []*System) (*System, error) {
			d[0].InputName, d[0].OutputName = G.InputName, G.OutputName
			d[1].InputName, d[1].OutputName = K.InputName, K.OutputName
			return ConnectByName([]*System{d[0], d[1], sum}, conns, extInNames, extOut)
		})
		if err == nil && (!slices.Equal(res.InputName, extInNames) || !slices.Equal(res.OutputName, extOut)) {
			t.Fatalf("ConnectByName names %v→%v, want %v→%v\n%s", res.InputName, res.OutputName, extInNames, extOut, s)
		}
		ic.check(t, s, fzICCase{op: "ConnectByName", states: -1}, frac, res, err)
	})
}

// fzICSeeds: SISO/MIMO, continuous/discrete, static partners, mixed
// stable/unstable modes, and constructed algebraic loops (negative Tau).
func fzICSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 0.5, Gain: 6}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0], s.Tau = 2, 1.5, 2.25 }),
		with(func(s *fzSeed) { s.N, s.Damp[0], s.Tau = 1, 1, 0.5 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Tau = 4, 2, 3, [3]float64{0.3, -0.5}, [3]float64{1, -2}, 3.4+7*11
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.Tau = 3, 3, 2, [3]float64{0.2, 1}, 5.7+7*137 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.Damp, s.Tau = true, -1, 2, 2, [3]float64{1.2}, 4.1+7*29
		}),
		with(func(s *fzSeed) { s.N, s.P, s.DScal, s.Tau = 0, 2, 1.7, 0.9+7*5 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Shear, s.Tau = 5, 2, 2, -1, 2.6+7*3011 }),
		with(func(s *fzSeed) { s.N, s.Damp[0], s.Tau = 2, 1.5, -3.25 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Tau = 2, 2, 2, -(1.5 + 7*13) }),
		with(func(s *fzSeed) { s.Discrete, s.LogDt, s.N, s.M, s.P, s.Tau = true, -2, 3, 3, 3, -(6.3 + 7*63) }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.In, s.Out, s.IO, s.Tau = 3, 2, 2, 0.2, -0.5, 0.1, 4.4+7*95 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.In, s.Out, s.IO, s.Tau = true, -1, 2, 2, 0.5, 0.3, 0.7, 1.6+7*7
		}),
		{Discrete: true, P: 2, M: 1, Mag: [3]float64{0, 536, 124}, Damp: [3]float64{0, -66, 93}, Gain: 6, DScal: 90.3,
			LogDt: -31, Shear: 35, Rot: 0.37, Phase: -10, In: -70, Out: -10, IO: 390, Tau: 175.9},
	}
}

// fzICCrashers are fuzz finds at the limits of the bounds: near-DC points
// beside tiny or zero poles and high-frequency delay phases (seeded once
// they passed).
var fzICCrashers = []struct {
	op   string
	kind uint8
	v    [fzNVals]float64
}{
	{"Append", 0x20, [fzNVals]float64{-31, -77.5, 99, -96, -240, -120, -1.4642857142857142, 0.5, -13, -32, 143.11, 103, 30, 31.25, 0.8571428571428571, -54.5}},
	{"Connect", 0x20, [fzNVals]float64{-68, 1, 0, 7.25, -28, 56, 6, 4.9, 7, -10, -285.78000000000003, -231, -79, -1.25, -309.44444444444446, 3480}},
	{"Feedback", 0x4a, [fzNVals]float64{-28, 41, 14, 0, -39, -7, 48, 15, 0, -10, 0.37, -86, -10, 8, 12, -82.5}},
	{"Feedback", 0x4a, [fzNVals]float64{-28, 41, 14, 0, -39, 0, 48, 0.5, 0, -10, 0.37, 0, -10, -10, 12, -82.5}},
	{"LFT", 0x48, [fzNVals]float64{-91, 96, 0, 0, 44, -70, 6, -0.5, -26, -10, 0.37, 0, 0.5, 20.299999999999997, 0.7, 101.2}},
	{"Parallel", 0x3c, [fzNVals]float64{-6.5, 37, 3.3333333333333335, -43, 0, -24, 44, 4, 0, -10, 0.37, 7.428571428571429, -58, -2, 68, 0.08333333333333333}},
	{"Series", 0x20, [fzNVals]float64{-70, 1, 0, 0, 23, 14, 6, 0.5, 0, -10, 97.06166666666667, 66, 27, -10, -10, -36.92857142857143}},
}

func fzICAddSeeds(f *testing.F, op string) {
	for _, s := range fzICSeeds() {
		fzAddSeed(f, s)
	}
	for _, c := range fzICCrashers {
		if c.op != op {
			continue
		}
		v := c.v
		f.Add(c.kind, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])
	}
}
