package controlsys

// Delay-approximation fuzz targets (BAY5BF): FuzzPadeDelay, FuzzPadeAbsorb
// (Pade and continuous AbsorbDelay), FuzzAbsorbDelayDiscrete and
// FuzzDelayLFTRoundTrip (PullDelaysToLFT, GetDelayModel/SetDelayModel,
// ZeroDelayApprox). ThiranDelay up to 40 samples is FuzzThiran's.
//
// Models: fzApxSystem adds to the decoded spec its input, output and I/O
// delays and, when Tau > 0, an internal LFT with N = 1 or 2 delays (Phase
// picks N; τ2 = 1.7·τ1, or τ1+1 samples) and nonzero B2, C2, D12, D21 and
// D22 (entries <= 0.3, D22 scaled by sin(7·Tau field)).
// Tuple fields repurposed: Kind picks the PadeDelay order; Phase the Pade
// order (FuzzPadeAbsorb); In's fractional digits the AbsorbDelay scopes.
//
// Oracles: fzApxEval closes any model's LFT in complex128 Gauss–Jordan on
// [sI−A, −B2; −ΔC2, I−ΔD22] with a first-order rounding bound, Δ a caller
// function of τ (e^{−sτ}, z^{−τ}, the Padé approximant or 1). The Padé
// approximant R_n is evaluated from factorial coefficients in big.Float
// (fzApxPade); TestFzApxOracles pins it to the closed forms and to the
// leading error term (n!)²/((2n)!(2n+1)!)·x^{2n+1}. PadeDelay itself is
// evaluated exactly (fzExactResponse). Continuous Padé of I/O-delay matrices
// is checked against e^{−sT} within K·|R_n(jωT)−e^{−jωT}| per channel
// (ωT <= 2, where that error is monotone), since the library's split of T
// into Padé factors is its own choice.

import (
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"sync"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type fzApxFn func(tau float64) complex128

// fzApxPt is an evaluation point: s = jω (continuous) or z = e^{jθ}.
type fzApxPt struct {
	x     complex128
	theta float64
	disc  bool
	ph    float64 // phase scale ω·Σdelays: rounding of e^{−jωτ}
}

func (pt fzApxPt) exact(tau float64) complex128 {
	if tau == 0 {
		return 1
	}
	if pt.disc {
		return cmplx.Exp(complex(0, -pt.theta*tau))
	}
	return cmplx.Exp(-pt.x * complex(tau, 0))
}

func fzApxOne(float64) complex128 { return 1 }

func (pt fzApxPt) pade(n int) fzApxFn {
	return func(tau float64) complex128 {
		if tau == 0 {
			return 1
		}
		return fzApxPade(n, pt.x*complex(tau, 0))
	}
}

var fzApxPadeCoefs = sync.OnceValue(func() [][]*big.Rat {
	out := make([][]*big.Rat, 11)
	fact := func(k int) *big.Int { return new(big.Int).MulRange(1, int64(max(k, 1))) }
	for n := 1; n <= 10; n++ {
		out[n] = make([]*big.Rat, n+1)
		for k := 0; k <= n; k++ {
			num := new(big.Int).Mul(fact(2*n-k), fact(n))
			den := new(big.Int).Mul(fact(2*n), fact(k))
			den.Mul(den, fact(n-k))
			out[n][k] = new(big.Rat).SetFrac(num, den)
		}
	}
	return out
})

// fzApxPade returns R_n(x) = Σ(−x)^k q_k / Σ x^k q_k, q_k =
// (2n−k)!n!/((2n)!k!(n−k)!), in big.Float.
func fzApxPade(n int, x complex128) complex128 {
	q := fzApxPadeCoefs()[n]
	nf := func() *big.Float { return new(big.Float).SetPrec(fzBigPrec) }
	xr, xi := nf().SetFloat64(real(x)), nf().SetFloat64(imag(x))
	horner := func(sign int) (*big.Float, *big.Float) {
		vr, vi := nf(), nf()
		for k := n; k >= 0; k-- {
			a := nf().Mul(vr, xr)
			a.Sub(a, nf().Mul(vi, xi))
			b := nf().Mul(vr, xi)
			b.Add(b, nf().Mul(vi, xr))
			c := nf().SetRat(q[k])
			if sign < 0 && k%2 == 1 {
				c.Neg(c)
			}
			vr, vi = a.Add(a, c), b
		}
		return vr, vi
	}
	ar, ai := horner(-1)
	br, bi := horner(1)
	d := nf().Mul(br, br)
	d.Add(d, nf().Mul(bi, bi))
	re := nf().Mul(ar, br)
	re.Add(re, nf().Mul(ai, bi))
	im := nf().Mul(ai, br)
	im.Sub(im, nf().Mul(ar, bi))
	r, _ := re.Quo(re, d).Float64()
	i, _ := im.Quo(im, d).Float64()
	return complex(r, i)
}

// fzApxLead is the leading Padé error coefficient (n!)²/((2n)!(2n+1)!).
func fzApxLead(n int) float64 {
	f := func(k int) float64 {
		v, _ := new(big.Float).SetInt(new(big.Int).MulRange(1, int64(max(k, 1)))).Float64()
		return v
	}
	return f(n) * f(n) / (f(2*n) * f(2*n+1))
}

// fzApxErr is |R_n(jx) − e^{−jx}|.
func fzApxErr(n int, x float64) float64 {
	return cmplx.Abs(fzApxPade(n, complex(0, x)) - cmplx.Exp(complex(0, -x)))
}

func fzApxGet(m *mat.Dense, r, c int) (func(i, j int) float64, bool) {
	if r == 0 || c == 0 || m == nil {
		return func(int, int) float64 { return 0 }, true
	}
	if rr, cc := m.Dims(); rr != r || cc != c {
		return nil, false
	}
	raw := m.RawMatrix()
	return func(i, j int) float64 { return raw.Data[i*raw.Stride+j] }, true
}

// fzApxEval returns the p×m response of sys at pt with internal delays
// closed by delta and external (input, output, I/O) delays by ext, and a
// first-order bound on its rounding error; ok is false on an exactly
// singular or malformed system.
func fzApxEval(sys *System, pt fzApxPt, delta, ext fzApxFn) (h []complex128, bound float64, ok bool) {
	const eps = 0x1p-52
	n, m, p := sys.Dims()
	var tau []float64
	var lft LFTDelay
	if sys.LFT != nil {
		tau, lft = sys.LFT.Tau, *sys.LFT
	}
	N := len(tau)
	q := n + N
	A, ok1 := fzApxGet(sys.A, n, n)
	B, ok2 := fzApxGet(sys.B, n, m)
	C, ok3 := fzApxGet(sys.C, p, n)
	D, ok4 := fzApxGet(sys.D, p, m)
	B2, ok5 := fzApxGet(lft.B2, n, N)
	C2, ok6 := fzApxGet(lft.C2, N, n)
	D12, ok7 := fzApxGet(lft.D12, p, N)
	D21, ok8 := fzApxGet(lft.D21, N, m)
	D22, ok9 := fzApxGet(lft.D22, N, N)
	E, ok10 := fzApxGet(sys.E, n, n)
	if !(ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8 && ok9 && ok10) {
		return nil, 0, false
	}
	if sys.E == nil {
		E = func(i, j int) float64 {
			if i == j {
				return 1
			}
			return 0
		}
	}
	dl := make([]complex128, N)
	for k := range N {
		dl[k] = delta(tau[k])
	}
	w := q + m + q
	M := make([]complex128, q*w)
	var nM, nR float64
	for i := range n {
		for j := range n {
			M[i*w+j] = pt.x*complex(E(i, j), 0) - complex(A(i, j), 0)
		}
		for k := range N {
			M[i*w+n+k] = complex(-B2(i, k), 0)
		}
		for c := range m {
			M[i*w+q+c] = complex(B(i, c), 0)
		}
	}
	for k := range N {
		r := (n + k) * w
		for j := range n {
			M[r+j] = -dl[k] * complex(C2(k, j), 0)
		}
		for l := range N {
			M[r+n+l] = -dl[k] * complex(D22(k, l), 0)
		}
		M[r+n+k] += 1
		for c := range m {
			M[r+q+c] = dl[k] * complex(D21(k, c), 0)
		}
	}
	for i := range q {
		for j := range q {
			nM += fzApxSq(M[i*w+j])
		}
		for c := range m {
			nR += fzApxSq(M[i*w+q+c])
		}
		M[i*w+q+m+i] = 1
	}
	for k := range q {
		piv, best := k, cmplx.Abs(M[k*w+k])
		for i := k + 1; i < q; i++ {
			if v := cmplx.Abs(M[i*w+k]); v > best {
				piv, best = i, v
			}
		}
		if best == 0 || math.IsNaN(best) {
			return nil, 0, false
		}
		if piv != k {
			for j := range w {
				M[k*w+j], M[piv*w+j] = M[piv*w+j], M[k*w+j]
			}
		}
		inv := 1 / M[k*w+k]
		for j := k; j < w; j++ {
			M[k*w+j] *= inv
		}
		for i := range q {
			if i == k || M[i*w+k] == 0 {
				continue
			}
			f := M[i*w+k]
			for j := k; j < w; j++ {
				M[i*w+j] -= f * M[k*w+j]
			}
		}
	}
	var nX, nInv, nCf, nD float64
	for i := range q {
		for c := range m {
			nX += fzApxSq(M[i*w+q+c])
		}
		for j := range q {
			nInv += fzApxSq(M[i*w+q+m+j])
		}
	}
	h = make([]complex128, p*m)
	for i := range p {
		for j := range n {
			nCf += C(i, j) * C(i, j)
		}
		for k := range N {
			nCf += D12(i, k) * D12(i, k)
		}
		for c := range m {
			nD += D(i, c) * D(i, c)
			v := complex(D(i, c), 0)
			for j := range n {
				v += complex(C(i, j), 0) * M[j*w+q+c]
			}
			for k := range N {
				v += complex(D12(i, k), 0) * M[(n+k)*w+q+c]
			}
			f := complex(1, 0)
			if len(sys.InputDelay) == m && sys.InputDelay[c] != 0 {
				f *= ext(sys.InputDelay[c])
			}
			if len(sys.OutputDelay) == p && sys.OutputDelay[i] != 0 {
				f *= ext(sys.OutputDelay[i])
			}
			if sys.Delay != nil {
				if d := sys.Delay.At(i, c); d != 0 {
					f *= ext(d)
				}
			}
			h[i*m+c] = v * f
		}
	}
	sq := math.Sqrt
	bound = 16 * float64(q+1) * eps * (sq(nCf)*sq(nInv)*(sq(nM)*sq(nX)+sq(nR)) + sq(nD))
	// Backward error of the routine that built sys: a perturbation of
	// eps·‖[A B B2; C D D12; C2 D21 D22]‖ in any entry.
	data := 0.0
	for _, g := range []struct {
		f    func(i, j int) float64
		r, c int
	}{{A, n, n}, {B, n, m}, {C, p, n}, {D, p, m}, {B2, n, N}, {C2, N, n}, {D12, p, N}, {D21, N, m}, {D22, N, N}} {
		for i := range g.r {
			for j := range g.c {
				data += g.f(i, j) * g.f(i, j)
			}
		}
	}
	bound += 64 * eps * sq(data) * ((sq(nCf)+1)*sq(nInv)*(sq(nX)+1) + sq(nX) + 1)
	for _, v := range h {
		if cmplx.IsNaN(v) || cmplx.IsInf(v) {
			return nil, 0, false
		}
	}
	return h, bound, !math.IsNaN(bound) && !math.IsInf(bound, 0)
}

func fzApxSq(z complex128) float64 { return real(z)*real(z) + imag(z)*imag(z) }

// fzApxStrip returns a copy of sys without external delays.
func fzApxStrip(sys *System) *System {
	cp := sys.Copy()
	cp.InputDelay, cp.OutputDelay, cp.Delay = nil, nil, nil
	return cp
}

// fzApxSystem builds the spec's model with all its external delays and,
// when Tau > 0 and n > 0, a 1- or 2-delay internal LFT.
func fzApxSystem(t *testing.T, s *fzSpec) *System {
	t.Helper()
	sys := s.System(t, fzSysOpts{Delays: true})
	if !(s.Tau > 0) || s.N == 0 {
		return sys
	}
	n, m, p := s.N, s.M, s.P
	N := 1 + int(math.Abs(s.Vals[11])*13)%2
	tau := []float64{s.Tau}
	if N == 2 {
		if s.Discrete {
			tau = append(tau, s.Tau+1)
		} else {
			tau = append(tau, 1.7*s.Tau)
		}
	}
	ph := fzClamp(s.Vals[11], -1e3, 1e3)
	g22 := 0.3 * math.Sin(7*s.Vals[15])
	b2, c2 := make([]float64, n*N), make([]float64, N*n)
	d12, d21, d22 := make([]float64, p*N), make([]float64, N*m), make([]float64, N*N)
	for k := range N {
		for i := range n {
			b2[i*N+k] = s.B[i*m+k%m] * math.Cos(float64(k)+0.4)
			c2[k*n+i] = 0.3 * math.Sin(float64(i+k)+ph)
		}
		for i := range p {
			d12[i*N+k] = 0.3 * math.Cos(float64(i+2*k)+ph)
		}
		for j := range m {
			d21[k*m+j] = 0.3 * math.Cos(float64(2*j+k)+0.5*ph)
		}
		for l := range N {
			d22[k*N+l] = g22 * math.Cos(float64(k-l)+ph)
		}
	}
	err := sys.SetInternalDelay(tau, mat.NewDense(n, N, b2), mat.NewDense(N, n, c2),
		mat.NewDense(p, N, d12), mat.NewDense(N, m, d21), mat.NewDense(N, N, d22))
	if err != nil {
		t.Fatalf("SetInternalDelay: %v\n%s", err, s)
	}
	return sys
}

func fzApxMaxDelay(sys *System) float64 {
	d := 0.0
	for _, v := range slices.Concat(sys.InputDelay, sys.OutputDelay) {
		d = math.Max(d, v)
	}
	if sys.Delay != nil {
		d = math.Max(d, mat.Max(sys.Delay))
	}
	if sys.LFT != nil {
		for _, v := range sys.LFT.Tau {
			d = math.Max(d, v)
		}
	}
	return d
}

// fzApxPoints returns evaluation points: frequencies tied to the delays and
// (offset from) the poles, plus one fuzzed frequency.
func fzApxPoints(s *fzSpec, tmax float64) []fzApxPt {
	var ws []float64
	if s.Discrete {
		ws = []float64{0.05, 0.3, 1.1, 2.0, 2.9}
		for _, e := range s.Eig {
			if th := math.Abs(cmplx.Phase(e)); th > 1e-3 {
				ws = append(ws, math.Min(1.07*th, 3.1))
			}
		}
		ws = append(ws, math.Mod(math.Abs(s.Vals[0]), math.Pi))
	} else {
		ws = []float64{1e-3, 1, 10 * math.Pow(10, fzClamp(s.Vals[0], -3, 3))}
		if tmax > 0 {
			ws = append(ws, 0.1/tmax, 0.7/tmax, 1.9/tmax, 8/tmax)
		}
		for _, e := range s.S {
			if a := cmplx.Abs(e); a > 0 {
				ws = append(ws, 1.07*a)
			}
		}
	}
	slices.Sort(ws)
	ws = slices.Compact(ws)
	tsum := fzApxSum(s.In) + fzApxSum(s.Out) + 2.7*s.Tau
	if s.IO != nil {
		tsum += slices.Max(s.IO)
	}
	out := make([]fzApxPt, 0, len(ws))
	for _, w := range ws {
		if s.Discrete {
			out = append(out, fzApxPt{x: cmplx.Exp(complex(0, w)), theta: w, disc: true, ph: w * tsum})
		} else {
			out = append(out, fzApxPt{x: complex(0, w), ph: w * tsum})
		}
	}
	return out
}

// fzApxCompare fails when got and want differ beyond the two evaluation
// bounds and a relative rounding slack.
func fzApxCompare(t *testing.T, s *fzSpec, op string, pt fzApxPt, got, want []complex128, bg, bw float64) {
	t.Helper()
	const eps = 0x1p-52
	for k := range want {
		tol := 4*(bg+bw) + (256+32*pt.ph)*eps*cmplx.Abs(want[k])
		if d := cmplx.Abs(got[k] - want[k]); !(d <= tol) {
			t.Fatalf("%s at %v: entry %d = %v, want %v (|Δ| %.3g > tol %.3g)\ngot  %v\nwant %v\n%s",
				op, pt.x, k, got[k], want[k], d, tol, got, want, s)
		}
	}
}

// fzApxUnstable counts the exact unstable poles of a delay-free model.
func fzApxUnstable(sys *System) (int, bool) {
	n, _, _ := sys.Dims()
	if n == 0 {
		return 0, true
	}
	cp := fzCharPolyExact(mat.DenseCopyOf(sys.A).RawMatrix().Data, n)
	if sys.IsDiscrete() {
		return fzSchurCohnOutside(cp, 1)
	}
	return fzRouthRight(cp, 0)
}

func fzApxSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 0.5, Gain: 6}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0], s.In = 2, 1.5, -0.3 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.In, s.Out, s.Phase = 3, 2, 2, 1.2, -0.5, -0.8, 0.21 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.IO, s.In = 2, 2, 2, 1.7, 0.1, -1.234 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out, s.Tau, s.Phase = 3, 2, 1, [3]float64{0.2, -0.3, 0}, [3]float64{1.4, 0, 0}, -0.6, -0.9, -0.4, 0.05
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.Tau, s.Phase, s.In = 2, 1, 2, -1.5, -0.2, 0.9, -2.71 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.IO, s.In, s.Out = 0, 2, 3, 0.3, -1, -1.5 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.P, s.Damp[0], s.In, s.Out, s.IO = true, -1, 2, 2, 2, 1.6, 0.3, 0.1, 0.5
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.P, s.Damp[0], s.Tau, s.Phase, s.In = true, -0.5, 3, 1, 2, 1.3, 0.4, 0.61, 0.6
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.P, s.Damp[0], s.Tau, s.Out, s.In = true, -1, 2, 2, 1, -1.8, 0.2, 0.5, 0.31
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Shear, s.Tau, s.IO, s.Out, s.In = 4, 3, 3, [3]float64{0.5, -0.5, 0}, [3]float64{2, 1, 0}, -1, -0.1, 0.2, -0.6, -1.901
		}),
	}
}

// FuzzPadeDelay checks PadeDelay(τ, n) against the factorial-coefficient
// Padé approximant (exact evaluation of the returned realization), its
// exact Hurwitz denominator, and the leading-term error bound for ωτ <= 1.
func FuzzPadeDelay(f *testing.F) {
	for _, sd := range fzApxSeeds() {
		fzAddSeed(f, sd)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		const eps = 0x1p-52
		tau := math.Pow(10, fzClamp(s.Vals[15], -6, 6))
		order := 1 + int(s.Kind)%10
		sys, err := PadeDelay(tau, order)
		fzContract(t, s, "PadeDelay", sys != nil, err)
		fzErr(t, s, "PadeDelay", err)
		n, m, p := sys.Dims()
		if n != order || m != 1 || p != 1 || sys.HasDelay() || !sys.IsContinuous() {
			t.Fatalf("PadeDelay(%g, %d): %d states %d×%d delay=%v dt=%g", tau, order, n, p, m, sys.HasDelay(), sys.Dt)
		}
		if k, ok := fzApxUnstable(sys); !ok || k > 0 {
			t.Fatalf("PadeDelay(%g, %d) not Hurwitz (%d unstable, degenerate %v)", tau, order, k, !ok)
		}
		a := mat.DenseCopyOf(sys.A).RawMatrix().Data
		cp := fzCharPolyExact(a, n)
		cf := make([]float64, len(cp))
		for i, c := range cp {
			cf[i], _ = c.Float64()
		}
		raw := func(d *mat.Dense) []float64 { return mat.DenseCopyOf(d).RawMatrix().Data }
		xs := []float64{1e-3, 0.05, 0.3, 1, 2.5, 7, 30, 300, math.Pow(10, fzClamp(s.Vals[0], -3, 3))}
		for _, x := range xs {
			sp := complex(0, x/tau)
			g, ok := fzExactResponse(n, 1, 1, a, raw(sys.B), raw(sys.C), raw(sys.D), sp)
			if !ok {
				t.Fatalf("PadeDelay(%g, %d): sI−A singular at %v", tau, order, sp)
			}
			mag, den := 0.0, fzBigPolyEval(cf, sp)
			for i, c := range cf {
				mag += math.Abs(c) * math.Pow(cmplx.Abs(sp), float64(len(cf)-1-i))
			}
			cond := mag / cmplx.Abs(den)
			want := fzApxPade(order, complex(0, x))
			tol := 64*float64(n+1)*eps*cond*(1+cmplx.Abs(g[0])) + 16*eps*(1+x)
			if d := cmplx.Abs(g[0] - want); !(d <= tol) {
				t.Fatalf("PadeDelay(%g, %d) at ωτ=%g: %v, want %v (|Δ| %.3g > %.3g)", tau, order, x, g[0], want, d, tol)
			}
			if x <= 1 {
				lead := 1.5 * fzApxLead(order) * math.Pow(x, float64(2*order+1))
				if d := cmplx.Abs(g[0] - cmplx.Exp(complex(0, -x))); !(d <= lead+tol+4*eps) {
					t.Fatalf("PadeDelay(%g, %d) at ωτ=%g: error %.3g above leading-term bound %.3g", tau, order, x, d, lead)
				}
			}
		}
	})
}

type fzApxClasses struct{ in, out, io, internal bool }

var fzApxScopeSets = [][]AbsorbScope{
	nil, {AbsorbInput}, {AbsorbOutput}, {AbsorbIO}, {AbsorbInternal},
	{AbsorbInput, AbsorbOutput}, {AbsorbInternal, AbsorbInput}, {AbsorbAll},
}

// fzApxScopes picks a scope set from In's fractional digits and returns the
// delay classes it absorbs: AbsorbIO also absorbs input and output delays
// when an I/O delay is present (TestAbsorbDelayScopedContinuousPreservesInternalDelay).
func fzApxScopes(s *fzSpec, sys *System) ([]AbsorbScope, fzApxClasses) {
	set := fzApxScopeSets[int(math.Abs(s.Vals[12])*1000)%len(fzApxScopeSets)]
	var c fzApxClasses
	if len(set) == 0 {
		return set, fzApxClasses{true, true, true, true}
	}
	for _, sc := range set {
		switch sc {
		case AbsorbAll:
			c = fzApxClasses{true, true, true, true}
		case AbsorbInput:
			c.in = true
		case AbsorbOutput:
			c.out = true
		case AbsorbInternal:
			c.internal = true
		case AbsorbIO:
			c.io = true
			if sys.Delay != nil && delayMatrixHasNonzero(sys.Delay) {
				c.in, c.out = true, true
			}
		}
	}
	return set, c
}

// FuzzPadeAbsorb checks Pade(order) and continuous AbsorbDelay against
// the model with each absorbed delay replaced by its Padé approximant
// (exactly, when no I/O-delay matrix is absorbed) or within the Padé error
// bound of the exact delay; kept delays stay exact. Without internal delays
// the result is stable exactly when the plant is.
func FuzzPadeAbsorb(f *testing.F) {
	for _, sd := range fzApxSeeds() {
		fzAddSeed(f, sd)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.Discrete {
			fzSkip(t, "pade: discrete spec")
		}
		orig := fzApxSystem(t, s)
		var res *System
		var err error
		var cls fzApxClasses
		order := DefaultPadeOrder
		op := "Pade"
		usePade := int(math.Abs(s.Vals[12])*1e4)%3 == 0
		if usePade {
			order = 1 + int(math.Abs(s.Vals[11])*7)%10
			cls = fzApxClasses{true, true, true, true}
			res, err = orig.Pade(order)
		} else {
			var set []AbsorbScope
			set, cls = fzApxScopes(s, orig)
			op = fmt.Sprintf("AbsorbDelay%v", set)
			res, err = orig.AbsorbDelay(set...)
		}
		fzContract(t, s, op, res != nil, err)
		fzErr(t, s, op, err)
		fzApxCheckClasses(t, s, op, orig, res, cls)
		hasIO := orig.Delay != nil && delayMatrixHasNonzero(orig.Delay)
		nInt := orig.internalDelayCount()
		tmax := fzApxMaxDelay(orig)
		for _, pt := range fzApxPoints(s, tmax) {
			got, bg, ok := fzApxEval(res, pt, pt.exact, pt.exact)
			if !ok {
				continue
			}
			R := pt.pade(order)
			fdelta := fzApxFn(pt.exact)
			if cls.internal {
				fdelta = R
			}
			F, bf, ok := fzApxEval(fzApxStrip(orig), pt, fdelta, nil)
			if !ok {
				continue
			}
			if !(cls.io && hasIO) {
				g := func(on bool) fzApxFn {
					if on {
						return R
					}
					return pt.exact
				}
				want := slices.Clone(F)
				for i := range s.P {
					for j := range s.M {
						v := complex(1, 0)
						if s.In != nil {
							v *= g(cls.in)(s.In[j])
						}
						if s.Out != nil {
							v *= g(cls.out)(s.Out[i])
						}
						if hasIO {
							v *= g(cls.io)(s.IO[i*s.M+j])
						}
						want[i*s.M+j] *= v
					}
				}
				fzApxCompare(t, s, op, pt, got, want, bg, bf)
				continue
			}
			w := imag(pt.x)
			for i := range s.P {
				for j := range s.M {
					T := s.IO[i*s.M+j]
					if s.In != nil {
						T += s.In[j]
					}
					if s.Out != nil {
						T += s.Out[i]
					}
					if w*T > 2 {
						continue
					}
					k := i*s.M + j
					want := F[k] * pt.exact(T)
					tol := 3*fzApxErr(order, w*T)*cmplx.Abs(F[k]) + 4*(bg+bf) + (1e-14+32*0x1p-52*pt.ph)*cmplx.Abs(F[k])
					if d := cmplx.Abs(got[k] - want); !(d <= tol) {
						t.Fatalf("%s at ω=%g, entry (%d,%d), T=%g: %v vs exact delay %v (|Δ| %.3g > Padé bound %.3g)\n%s",
							op, w, i, j, T, got[k], want, d, tol, s)
					}
				}
			}
		}
		if nInt > 0 || res.HasDelay() || !s.Decidable() {
			return
		}
		if nr, _, _ := res.Dims(); nr > 24 {
			return
		}
		k, ok := fzApxUnstable(res)
		if !ok || (k == 0) != (s.Unstable() == 0) {
			t.Fatalf("%s: %d unstable poles (degenerate %v), plant has %d\n%s", op, k, !ok, s.Unstable(), s)
		}
	})
}

// fzApxCheckClasses fails when an absorbed delay class survives in res or
// a kept internal delay is lost.
func fzApxCheckClasses(t *testing.T, s *fzSpec, op string, orig, res *System, c fzApxClasses) {
	t.Helper()
	if res.IsDiscrete() != orig.IsDiscrete() || res.Dt != orig.Dt {
		t.Fatalf("%s: dt %g, want %g\n%s", op, res.Dt, orig.Dt, s)
	}
	if _, m, p := res.Dims(); m != s.M || p != s.P {
		t.Fatalf("%s: %d×%d, want %d×%d\n%s", op, p, m, s.P, s.M, s)
	}
	switch {
	case c.in && delaySliceHasNonzero(res.InputDelay):
		t.Fatalf("%s: input delay %v left\n%s", op, res.InputDelay, s)
	case c.out && delaySliceHasNonzero(res.OutputDelay):
		t.Fatalf("%s: output delay %v left\n%s", op, res.OutputDelay, s)
	case c.io && res.Delay != nil && delayMatrixHasNonzero(res.Delay):
		t.Fatalf("%s: I/O delay %v left\n%s", op, mat.Formatted(res.Delay), s)
	case c.internal && res.HasInternalDelay():
		t.Fatalf("%s: internal delays %v left\n%s", op, res.LFT.Tau, s)
	}
	if !c.io && orig.Delay != nil && delayMatrixHasNonzero(orig.Delay) && (res.Delay == nil || !mat.Equal(res.Delay, orig.Delay)) {
		t.Fatalf("%s: kept I/O delay %v became %v\n%s", op, mat.Formatted(orig.Delay), res.Delay, s)
	}
	if !c.internal && orig.HasInternalDelay() {
		if !res.HasInternalDelay() {
			t.Fatalf("%s dropped the internal delays %v\n%s", op, orig.LFT.Tau, s)
		}
		for _, v := range orig.LFT.Tau {
			if !slices.Contains(res.LFT.Tau, v) {
				t.Fatalf("%s: internal delay %g missing from %v\n%s", op, v, res.LFT.Tau, s)
			}
		}
	}
}

func fzApxSum(v []float64) float64 {
	t := 0.0
	for _, x := range v {
		t += x
	}
	return t
}

// FuzzAbsorbDelayDiscrete checks discrete AbsorbDelay, an exact identity:
// the absorbed model (kept delays evaluated exactly) has the delayed
// model's response, single-class absorption adds exactly one state per
// sample of delay (as MATLAB absorbDelay), and the kept classes survive.
func FuzzAbsorbDelayDiscrete(f *testing.F) {
	for _, sd := range fzApxSeeds() {
		fzAddSeed(f, sd)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if !s.Discrete {
			fzSkip(t, "absorb discrete: continuous spec")
		}
		orig := fzApxSystem(t, s)
		set, cls := fzApxScopes(s, orig)
		op := fmt.Sprintf("AbsorbDelay%v", set)
		res, err := orig.AbsorbDelay(set...)
		fzContract(t, s, op, res != nil, err)
		fzErr(t, s, op, err)
		fzApxCheckClasses(t, s, op, orig, res, cls)
		n0, _, _ := orig.Dims()
		nr, _, _ := res.Dims()
		var add float64
		switch {
		case slices.Equal(set, []AbsorbScope{AbsorbInput}):
			add = fzApxSum(orig.InputDelay)
		case slices.Equal(set, []AbsorbScope{AbsorbOutput}):
			add = fzApxSum(orig.OutputDelay)
		case slices.Equal(set, []AbsorbScope{AbsorbInternal}) && orig.LFT != nil:
			add = fzApxSum(orig.LFT.Tau)
		default:
			add = -1
		}
		if add >= 0 && float64(nr) != float64(n0)+add {
			t.Fatalf("%s: %d states, want %d + %g\n%s", op, nr, n0, add, s)
		}
		if nr > 400 {
			return
		}
		for _, pt := range fzApxPoints(s, 0) {
			want, bw, ok1 := fzApxEval(orig, pt, pt.exact, pt.exact)
			got, bg, ok2 := fzApxEval(res, pt, pt.exact, pt.exact)
			if !ok1 || !ok2 {
				continue
			}
			fzApxCompare(t, s, op, pt, got, want, bg, bw)
		}
	})
}

// FuzzDelayLFTRoundTrip checks PullDelaysToLFT and GetDelayModel →
// SetDelayModel preserve the delayed response and leave no external delay
// (all original internal delays kept), and ZeroDelayApprox equals the model
// with every internal delay closed by Δ = 1, external delays kept.
func FuzzDelayLFTRoundTrip(f *testing.F) {
	for _, sd := range fzApxSeeds() {
		fzAddSeed(f, sd)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		orig := fzApxSystem(t, s)
		lft, err := orig.PullDelaysToLFT()
		fzContract(t, s, "PullDelaysToLFT", lft != nil, err)
		fzErr(t, s, "PullDelaysToLFT", err)
		fzApxCheckClasses(t, s, "PullDelaysToLFT", orig, lft, fzApxClasses{in: true, out: true, io: true})
		if lft.LFT != nil {
			for _, v := range lft.LFT.Tau {
				if !(v > 0) {
					t.Fatalf("PullDelaysToLFT: internal delay %g in %v\n%s", v, lft.LFT.Tau, s)
				}
			}
		}
		H, tau, err := orig.GetDelayModel()
		fzContract(t, s, "GetDelayModel", H != nil && tau != nil, err)
		fzErr(t, s, "GetDelayModel", err)
		if _, mh, ph := H.Dims(); mh != s.M+len(tau) || ph != s.P+len(tau) || H.HasDelay() {
			t.Fatalf("GetDelayModel: H %d×%d with %d delays (delay %v), want %d×%d\n%s", ph, mh, len(tau), H.HasDelay(), s.P+len(tau), s.M+len(tau), s)
		}
		back, err := SetDelayModel(H, tau)
		fzContract(t, s, "SetDelayModel", back != nil, err)
		fzErr(t, s, "SetDelayModel", err)
		zero, err := orig.ZeroDelayApprox()
		fzContract(t, s, "ZeroDelayApprox", zero != nil, err)
		fzErr(t, s, "ZeroDelayApprox", err)
		if zero.HasInternalDelay() {
			t.Fatalf("ZeroDelayApprox left internal delays %v\n%s", zero.LFT.Tau, s)
		}
		zl, err := lft.ZeroDelayApprox()
		fzContract(t, s, "ZeroDelayApprox(PullDelaysToLFT)", zl != nil, err)
		fzErr(t, s, "ZeroDelayApprox(PullDelaysToLFT)", err)
		if zl.HasDelay() {
			t.Fatalf("ZeroDelayApprox(PullDelaysToLFT) has delays\n%s", s)
		}
		for _, pt := range fzApxPoints(s, fzApxMaxDelay(orig)) {
			want, bw, ok := fzApxEval(orig, pt, pt.exact, pt.exact)
			if !ok {
				continue
			}
			for _, c := range []struct {
				op  string
				sys *System
			}{{"PullDelaysToLFT", lft}, {"SetDelayModel(GetDelayModel)", back}} {
				if got, bg, ok := fzApxEval(c.sys, pt, pt.exact, pt.exact); ok {
					fzApxCompare(t, s, c.op, pt, got, want, bg, bw)
				}
			}
			if want, bw, ok := fzApxEval(orig, pt, fzApxOne, pt.exact); ok {
				if got, bg, ok := fzApxEval(zero, pt, pt.exact, pt.exact); ok {
					fzApxCompare(t, s, "ZeroDelayApprox", pt, got, want, bg, bw)
				}
			}
			if want, bw, ok := fzApxEval(orig, pt, fzApxOne, fzApxOne); ok {
				if got, bg, ok := fzApxEval(zl, pt, pt.exact, pt.exact); ok {
					fzApxCompare(t, s, "ZeroDelayApprox(PullDelaysToLFT)", pt, got, want, bg, bw)
				}
			}
		}
	})
}

// TestFzApxOracles pins fzApxPade to the closed forms R_1, R_2, to |R| = 1
// on the imaginary axis, to the leading error term for x <= 1 (ratio in
// [0.5, 1.5]) and to a monotone error on (0, 2], and fzApxEval to a
// hand-closed delay loop.
func TestFzApxOracles(t *testing.T) {
	for _, x := range []complex128{0.3, complex(0, 1.7), complex(-2, 0.5)} {
		r1 := (1 - x/2) / (1 + x/2)
		r2 := (1 - x/2 + x*x/12) / (1 + x/2 + x*x/12)
		if d := cmplx.Abs(fzApxPade(1, x) - r1); d > 1e-15 {
			t.Errorf("R_1(%v) off by %g", x, d)
		}
		if d := cmplx.Abs(fzApxPade(2, x) - r2); d > 1e-15 {
			t.Errorf("R_2(%v) off by %g", x, d)
		}
	}
	for n := 1; n <= 10; n++ {
		prev := 0.0
		for x := 0.01; x <= 2; x *= 1.1 {
			if a := cmplx.Abs(fzApxPade(n, complex(0, x))); math.Abs(a-1) > 1e-15 {
				t.Errorf("|R_%d(j%g)| = %v", n, x, a)
			}
			e := fzApxErr(n, x)
			lead := fzApxLead(n) * math.Pow(x, float64(2*n+1))
			if lead > 1e-11 && x <= 1 && (e/lead < 0.5 || e/lead > 1.5) {
				t.Errorf("n=%d x=%g: error %g, leading term %g", n, x, e, lead)
			}
			if lead > 1e-11 && e < prev {
				t.Errorf("n=%d: error not monotone at x=%g (%g < %g)", n, x, e, prev)
			}
			prev = e
		}
	}
	sys, err := NewFromSlices(1, 1, 1, []float64{-2}, []float64{1}, []float64{3}, []float64{0.5}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.4}, mat.NewDense(1, 1, []float64{0.7}), mat.NewDense(1, 1, []float64{-1.1}),
		mat.NewDense(1, 1, []float64{0.2}), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, []float64{0.25})); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{0.9}); err != nil {
		t.Fatal(err)
	}
	pt := fzApxPt{x: complex(0, 1.3)}
	got, bound, ok := fzApxEval(sys, pt, pt.exact, pt.exact)
	sp, dl := pt.x, pt.exact(0.4)
	wPerX := -1.1 * dl / (1 - 0.25*dl)
	wPerU := 0.3 * dl / (1 - 0.25*dl)
	xu := (1 + 0.7*wPerU) / (sp + 2 - 0.7*wPerX)
	want := (3*xu + 0.2*(wPerX*xu+wPerU) + 0.5) * pt.exact(0.9)
	if !ok || cmplx.Abs(got[0]-want) > 1e-14 || bound > 1e-12 {
		t.Errorf("fzApxEval = %v (bound %g, ok %v), want %v", got, bound, ok, want)
	}
}
