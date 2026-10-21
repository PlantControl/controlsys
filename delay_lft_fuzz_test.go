package controlsys

// Delay-system fuzz targets (PN4IYP): FuzzDelayLFT (IsStable, HinfNorm of
// internal-delay LFT models, commensurate, incommensurate, neutral) and
// FuzzDelayLoop (DiskMargin and the Feedback closed loop of SISO loops
// G·e^{−sτ1} + d2·e^{−sτ2}).
//
// Tuple fields repurposed (fzSeed names):
//   - Tau: base delay h, truncated to 20 mantissa bits so that k·h and the
//     subset sums θ_S are exact (whole samples when discrete).
//   - Out: delay pattern (fzDLPattern): < −3 one delay; [−3, 30) two; >= 30
//     three (y = Out−33). An integer y gives τ2 = k·h, k = 1+((y+3) mod 6)
//     (commensurate); otherwise τ2 = r·h, r = 2^{(y+3)/10} (incommensurate,
//     skipped within 1e-13 of p/q, q <= 64: AQYIAI band). τ3 = k3·τ1 or
//     k3·τ2 by Phase.
//   - IO: D22 scale (FuzzDelayLFT) or d2 (FuzzDelayLoop), off when < −3.
//   - In: input delays for the I/O-delay invariance of IsStable.
//
// Oracles: χ = Σ_S p_S(s)·e^{−sθ_S} exactly over big.Rat (multilinear
// expansion of det [sI−A, −B2; −ΔC2, I−ΔD22] in the rows of Δ); a
// certified argument-principle count of its roots right of Re s = σ
// (fzDLQuasi.countRight: Lipschitz step control, zero-free box from the
// leading difference operator), exact Schur–Cohn on z^Θ·χ(z) when discrete;
// dense sweeps confirmed by exact rational responses for never-under-report;
// the phase-torus sup of the high-frequency LFT for limits at ω = +Inf.

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const (
	fzDLBudget   = 200000
	fzDLMaxDisc  = 30
	fzDLRatioGap = 1e-13
	fzDLMaxTau   = 8
	fzDLMaxOrder = 4
)

// fzDLQuasi is χ(s) = Σ_k p_k(s)·e^{−sθ_k} with distinct θ_k >= 0, every p_k
// padded to degree n (descending): exact coefficients r, their float64
// roundings f and magnitudes a.
type fzDLQuasi struct {
	n     int
	theta []float64
	r     [][]*big.Rat
	f, a  [][]float64
}

// fzDLNewQuasi merges terms of equal θ (bitwise) and drops zero ones.
func fzDLNewQuasi(theta []float64, polys [][]*big.Rat) *fzDLQuasi {
	q := &fzDLQuasi{}
	for _, p := range polys {
		q.n = max(q.n, len(p)-1)
	}
	for i, p := range polys {
		k := slices.Index(q.theta, theta[i])
		if k < 0 {
			k = len(q.theta)
			q.theta = append(q.theta, theta[i])
			z := make([]*big.Rat, q.n+1)
			for j := range z {
				z[j] = new(big.Rat)
			}
			q.r = append(q.r, z)
		}
		off := q.n + 1 - len(p)
		for j, c := range p {
			q.r[k][off+j].Add(q.r[k][off+j], c)
		}
	}
	for k := range slices.Backward(q.theta) {
		if !slices.ContainsFunc(q.r[k], func(c *big.Rat) bool { return c.Sign() != 0 }) {
			q.theta = slices.Delete(q.theta, k, k+1)
			q.r = slices.Delete(q.r, k, k+1)
		}
	}
	for _, p := range q.r {
		f := fzRatFloats(p)
		a := make([]float64, len(f))
		for i, x := range f {
			a[i] = math.Abs(x)
		}
		q.f, q.a = append(q.f, f), append(q.a, a)
	}
	return q
}

// eval returns χ(s) and a bound on its float64 evaluation error.
func (q *fzDLQuasi) eval(s complex128) (complex128, float64) {
	var v complex128
	e, r := 0.0, cmplx.Abs(s)
	for k, th := range q.theta {
		var p complex128
		pa := 0.0
		for i, c := range q.f[k] {
			p = p*s + complex(c, 0)
			pa = pa*r + q.a[k][i]
		}
		if th != 0 {
			p *= cmplx.Exp(complex(-real(s)*th, -imag(s)*th))
			pa *= math.Exp(-real(s) * th)
		}
		v += p
		e += pa * (1 + r*th)
	}
	return v, e * float64(8*(q.n+4)) * 0x1p-52
}

// lip bounds |χ'| on |s| <= R, Re s >= smin.
func (q *fzDLQuasi) lip(R, smin float64) float64 {
	m := 0.0
	for k, th := range q.theta {
		P, dP := 0.0, 0.0
		for _, a := range q.a[k] {
			dP = dP*R + P
			P = P*R + a
		}
		m += math.Exp(-smin*th) * (dP + th*P)
	}
	return m
}

// lead returns μ = |p_0 lead| − Σ_{θ>0} |p_k lead|·e^{−σθ_k}: |χ(s)|/|s|^n
// tends to at least μ on Re s >= σ.
func (q *fzDLQuasi) lead(sigma float64) float64 {
	mu, tot := 0.0, 0.0
	for k, th := range q.theta {
		l := q.a[k][0]
		tot += l
		if th == 0 {
			mu += l
		} else {
			mu -= l * math.Exp(-sigma*th)
		}
	}
	return mu - 1e-12*tot
}

// countRight counts the roots of χ with Re s > σ by the argument principle
// on [σ, B]×[−B, B], B such that |χ(s)| >= |s|^n(μ − tail) > 0 beyond it.
// Each step [z, w] is accepted only when lip·|w−z| + err(z) + err(w) <=
// |χ(z)|/2, so χ has no zero on the segment and its argument moves by the
// principal value of χ(w)/χ(z). ok is false when μ <= 0 (difference operator
// not certified), the contour grazes a root, or the budget runs out.
func (q *fzDLQuasi) countRight(sigma float64, budget int) (int, bool) {
	mu := q.lead(sigma)
	if !(mu > 0) || len(q.theta) == 0 {
		return 0, false
	}
	tail := func(R float64) float64 {
		t := 0.0
		for k, th := range q.theta {
			v := 0.0
			for i := q.n; i >= 1; i-- {
				v = (v + q.a[k][i]) / R
			}
			t += math.Exp(-sigma*th) * v
		}
		return t * (1 + 1e-9)
	}
	B := 1.0
	for !(tail(B) < mu) || B <= 2*math.Abs(sigma) {
		B *= 2
		if B > 0x1p30 {
			return 0, false
		}
	}
	corners := []complex128{complex(sigma, -B), complex(B, -B), complex(B, B), complex(sigma, B), complex(sigma, -B)}
	total, evals := 0.0, 0
	walk := func(a, b complex128) bool {
		L := cmplx.Abs(b - a)
		u := (b - a) / complex(L, 0)
		z := a
		fz, ez := q.eval(z)
		pos, h := 0.0, L/64
		for pos < L {
			if !(cmplx.Abs(fz) > 2*ez) {
				return false
			}
			h = math.Min(2*h, L-pos)
			for {
				evals++
				if evals > budget {
					return false
				}
				w := b
				if pos+h < L {
					w = a + u*complex(pos+h, 0)
				}
				fw, ew := q.eval(w)
				if q.lip(cmplx.Abs(z)+h, math.Min(real(z), real(w)))*h+ez+ew <= 0.5*cmplx.Abs(fz) {
					total += cmplx.Phase(fw / fz)
					pos += h
					z, fz, ez = w, fw, ew
					break
				}
				h /= 2
				if h < 1e-15*math.Max(1, cmplx.Abs(z)) {
					return false
				}
			}
		}
		return true
	}
	for e := range 4 {
		if !walk(corners[e], corners[e+1]) {
			return 0, false
		}
	}
	turns := total / (2 * math.Pi)
	if math.Abs(turns-math.Round(turns)) > 0.1 {
		return 0, false
	}
	return int(math.Round(turns)), true
}

// discrete returns z^Θ·χ(z), Θ = max θ_k, for integer sample delays.
func (q *fzDLQuasi) discrete() []*big.Rat {
	T := int(slices.Max(q.theta))
	p := make([]*big.Rat, q.n+T+1)
	for i := range p {
		p[i] = new(big.Rat)
	}
	for k, th := range q.theta {
		for i, c := range q.r[k] {
			p[i+int(th)].Add(p[i+int(th)], c)
		}
	}
	return fzTrimRat(p)
}

// fzDLDetPoly returns det(s·E − F) exactly, descending of length n+1, for
// the size×size rational F and E = diag(I_n, 0): Newton interpolation at
// s = 0..n.
func fzDLDetPoly(F []*big.Rat, size, n int) []*big.Rat {
	y := make([]*big.Rat, n+1)
	M := make([]*big.Rat, size*size)
	for k := range y {
		for i := range size {
			for j := range size {
				v := new(big.Rat).Neg(F[i*size+j])
				if i == j && i < n {
					v.Add(v, big.NewRat(int64(k), 1))
				}
				M[i*size+j] = v
			}
		}
		_, y[k] = fzRatElim(M, size, size)
	}
	for j := 1; j <= n; j++ {
		for k := n; k >= j; k-- {
			y[k] = new(big.Rat).Quo(new(big.Rat).Sub(y[k], y[k-1]), big.NewRat(int64(j), 1))
		}
	}
	poly := []*big.Rat{new(big.Rat).Set(y[n])}
	for k := n - 1; k >= 0; k-- {
		poly = fzRatPolyMul(poly, []*big.Rat{big.NewRat(1, 1), big.NewRat(-int64(k), 1)})
		poly[len(poly)-1].Add(poly[len(poly)-1], y[k])
	}
	return poly
}

// fzDLModel is an internal-delay LFT: fzSpec's A, B1 = B, C1 = C, D11 = D
// with nd delays τ and blocks B2 (n×nd), C2 (nd×n), D12 (p×nd), D21
// (nd×m), D22 (nd×nd). class/mult give the commensurate classes
// τ_i = mult_i·h_class (dim classes).
type fzDLModel struct {
	s                  *fzSpec
	nd                 int
	tau                []float64
	class, mult        []int
	dim                int
	B2, C2, D12, D21   []float64
	D22                []float64
	d22                float64
	commensurate, diag bool
}

// fzDLTrunc keeps the leading bits of x's mantissa.
func fzDLTrunc(x float64, bits int) float64 {
	fr, e := math.Frexp(x)
	return math.Ldexp(math.Trunc(math.Ldexp(fr, bits)), e-bits)
}

// fzDLNearRational reports a ratio within fzDLRatioGap of p/q, q <= 64.
func fzDLNearRational(r float64) bool {
	for l := 1.0; l <= 64; l++ {
		if x := r * l; math.Abs(x-math.Round(x)) <= fzDLRatioGap*x {
			return true
		}
	}
	return false
}

// fzDLPattern decodes the delays from the base h, the pattern field x and
// the phase field; see the file comment.
func fzDLPattern(t *testing.T, h, x, phase float64, discrete bool) (tau []float64, class, mult []int, dim int) {
	tau, class, mult, dim = []float64{h}, []int{0}, []int{1}, 1
	if x < -3 {
		return
	}
	y := x
	nd := 2
	if x >= 30 {
		nd, y = 3, x-33
	}
	y = fzClamp(y, -3, 27)
	if y == math.Trunc(y) {
		k := 1 + (int(y)+3)%6
		tau, class, mult = append(tau, float64(k)*h), append(class, 0), append(mult, k)
	} else {
		r := math.Exp2((y + 3) / 10)
		if discrete {
			k := math.Max(1, math.Round(r*h))
			tau, class, mult = append(tau, k), append(class, 0), append(mult, int(k))
		} else {
			if fzDLNearRational(r) {
				fzSkip(t, "delay ratio within the near-rational band")
			}
			tau, class, mult, dim = append(tau, fzDLTrunc(r*h, 40)), append(class, 1), append(mult, 1), 2
		}
	}
	if nd == 3 {
		k3 := 1 + int(math.Abs(phase)*7)%4
		base := int(math.Abs(phase)*13) % 2
		tau = append(tau, float64(k3)*tau[base])
		class = append(class, class[base])
		mult = append(mult, k3*mult[base])
	}
	if discrete {
		for i := range mult {
			mult[i] = int(tau[i])
		}
	}
	return
}

func fzDLNewModel(t *testing.T, s *fzSpec) *fzDLModel {
	t.Helper()
	if s.Tau == 0 {
		fzSkip(t, "no internal delay")
	}
	if s.N > fzDLMaxOrder {
		fzSkip(t, "order cap")
	}
	h := fzDLTrunc(s.Tau, 20)
	tau, class, mult, dim := fzDLPattern(t, h, s.Vals[13], s.Vals[11], s.Discrete)
	if !s.Discrete && slices.Max(tau) > fzDLMaxTau {
		fzSkip(t, "delay range cap")
	}
	if s.Discrete {
		sum := 0.0
		for _, x := range tau {
			sum += x
		}
		if sum > fzDLMaxDisc {
			fzSkip(t, "discrete delay order cap")
		}
	}
	n, m, p, nd := s.N, s.M, s.P, len(tau)
	ph := s.Vals[11]
	tex := func(i, j int, a, b float64) float64 { return math.Cos(ph + a*float64(i+1) + b*float64(j+1)) }
	md := &fzDLModel{s: s, nd: nd, tau: tau, class: class, mult: mult, dim: dim, commensurate: dim == 1}
	md.B2 = make([]float64, n*nd)
	md.C2 = make([]float64, nd*n)
	md.D12 = make([]float64, p*nd)
	md.D21 = make([]float64, nd*m)
	md.D22 = make([]float64, nd*nd)
	for i := range n {
		for k := range nd {
			md.B2[i*nd+k] = tex(i, k, 0.9, 2.3)
			md.C2[k*n+i] = 0.5 * math.Sin(ph+1.1*float64(k+1)+0.8*float64(i+1))
		}
	}
	for k := range nd {
		for i := range p {
			md.D12[i*nd+k] = 0.5 * tex(i, k, 1.7, 0.3)
		}
		for j := range m {
			md.D21[k*m+j] = 0.5 * tex(k, j, 0.4, 2.6)
		}
	}
	if v := s.Vals[14]; v >= -3 {
		md.d22 = fzClamp(v, -3, 3) / 2
		for i := range nd {
			for j := range nd {
				md.D22[i*nd+j] = md.d22 * tex(i, j, 1.9, 0.7)
			}
		}
	}
	return md
}

func fzDLDense(r, c int, v []float64) *mat.Dense {
	if r == 0 || c == 0 {
		return nil
	}
	return mat.NewDense(r, c, v)
}

func (md *fzDLModel) system(t *testing.T, inputDelay bool) *System {
	t.Helper()
	s := md.s
	sys, err := NewFromSlices(s.N, s.M, s.P, s.A, s.B, s.C, s.D, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v\n%s", err, s)
	}
	if inputDelay && s.In != nil {
		if err := sys.SetInputDelay(s.In); err != nil {
			t.Fatalf("SetInputDelay: %v\n%s", err, s)
		}
	}
	nd := md.nd
	if err := sys.SetInternalDelay(md.tau, fzDLDense(s.N, nd, md.B2), fzDLDense(nd, s.N, md.C2),
		fzDLDense(s.P, nd, md.D12), fzDLDense(nd, s.M, md.D21), mat.NewDense(nd, nd, md.D22)); err != nil {
		t.Fatalf("SetInternalDelay: %v\n%s", err, md)
	}
	return sys
}

func (md *fzDLModel) String() string {
	return md.s.String() + "\n" + fzDLFmtLFT(md.tau, md.B2, md.C2, md.D12, md.D21, md.D22)
}

func fzDLFmtLFT(tau, b2, c2, d12, d21, d22 []float64) string {
	return "tau=" + fzDLFmt(tau) + " B2=" + fzDLFmt(b2) + " C2=" + fzDLFmt(c2) +
		" D12=" + fzDLFmt(d12) + " D21=" + fzDLFmt(d21) + " D22=" + fzDLFmt(d22)
}

func fzDLFmt(v []float64) string {
	out := "["
	for i, x := range v {
		if i > 0 {
			out += " "
		}
		out += big.NewFloat(x).Text('g', 17)
	}
	return out + "]"
}

// quasi returns χ of the model: p_S = det [sI−A, −B2_S; −C2_S, −D22_SS]
// with θ_S = Σ_{i∈S} τ_i.
func (md *fzDLModel) quasi() *fzDLQuasi {
	s, n, nd := md.s, md.s.N, md.nd
	var theta []float64
	var polys [][]*big.Rat
	for mask := range 1 << nd {
		var S []int
		th := 0.0
		for i := range nd {
			if mask>>i&1 == 1 {
				S = append(S, i)
				th += md.tau[i]
			}
		}
		k := n + len(S)
		F := make([]*big.Rat, k*k)
		for i := range F {
			F[i] = new(big.Rat)
		}
		for i := range n {
			for j := range n {
				F[i*k+j] = fzRat(s.A[i*n+j])
			}
			for c, l := range S {
				F[i*k+n+c] = fzRat(md.B2[i*nd+l])
			}
		}
		for r, l := range S {
			for j := range n {
				F[(n+r)*k+j] = fzRat(md.C2[l*n+j])
			}
			for c, l2 := range S {
				F[(n+r)*k+n+c] = fzRat(md.D22[l*nd+l2])
			}
		}
		theta = append(theta, th)
		polys = append(polys, fzDLDetPoly(F, k, n))
	}
	return fzDLNewQuasi(theta, polys)
}

// fzDLVerdict decides the stability of χ: +1 stable, −1 unstable, 0
// undecided. Continuous: equal certified counts right of ±δ; a commensurate
// difference operator with a root in |z| < 1 is unstable (an RHP root
// chain). Discrete: exact Schur–Cohn on z^Θχ at 1 ± 2⁻²⁰.
func fzDLVerdict(q *fzDLQuasi, discrete bool, delta float64, leadZ []*big.Rat) (int, string) {
	if discrete {
		p := q.discrete()
		lo, ok1 := fzSchurCohnOutside(p, 1+0x1p-20)
		hi, ok2 := fzSchurCohnOutside(p, 1-0x1p-20)
		if !ok1 || !ok2 || lo != hi {
			return 0, "characteristic root within margin of stability boundary"
		}
		if hi == 0 {
			return 1, ""
		}
		return -1, ""
	}
	if q.lead(-delta) <= 0 {
		if leadZ != nil {
			g := fzTrimRat(leadZ)
			if len(g) > 1 {
				out, ok := fzSchurCohnOutside(g, 1-0x1p-20)
				if ok && out < len(g)-1 {
					return -1, ""
				}
			}
		}
		return 0, "difference operator not certifiable"
	}
	lo, ok1 := q.countRight(delta, fzDLBudget)
	hi, ok2 := q.countRight(-delta, fzDLBudget)
	if !ok1 || !ok2 {
		return 0, "argument principle not certified"
	}
	if lo != hi {
		return 0, "characteristic root within margin of stability boundary"
	}
	if hi == 0 {
		return 1, ""
	}
	return -1, ""
}

// leadZ returns the leading difference operator Σ_k lead_k·z^{θ_k/h} as a
// descending polynomial in z = e^{−sh} for commensurate continuous models.
func (md *fzDLModel) leadZ(q *fzDLQuasi) []*big.Rat {
	if !md.commensurate || md.s.Discrete {
		return nil
	}
	h := md.tau[0]
	K := 0
	for _, th := range q.theta {
		K = max(K, int(math.Round(th/h)))
	}
	g := make([]*big.Rat, K+1)
	for i := range g {
		g[i] = new(big.Rat)
	}
	for k, th := range q.theta {
		e := int(math.Round(th / h))
		if float64(e)*h != th {
			return nil
		}
		g[K-e].Add(g[K-e], q.r[k][0])
	}
	return g
}

// delta returns Δ = diag(e^{−jφ_i}) for continuous frequency ω (φ = ωτ) or
// z^{−τ} at z = e^{jωDt}.
func (md *fzDLModel) delta(w float64) []complex128 {
	d := make([]complex128, md.nd)
	for i, t := range md.tau {
		phi := w * t
		if md.s.Discrete {
			phi = math.Mod(w*md.s.Dt, 2*math.Pi) * t
		}
		d[i] = cmplx.Exp(complex(0, -phi))
	}
	return d
}

// response returns the p×m response at x with delay factors d, by Gaussian
// elimination on [xI−A, −B2; −ΔC2, I−ΔD22]·[X; W] = [B1; ΔD21];
// states=false gives the high-frequency LFT D11 + D12(I−ΔD22)⁻¹ΔD21.
func (md *fzDLModel) response(x complex128, d []complex128, states bool) []complex128 {
	s := md.s
	n, m, p, nd := s.N, s.M, s.P, md.nd
	if !states {
		n = 0
	}
	k := n + nd
	cols := k + m
	M := make([]complex128, k*cols)
	for i := range n {
		for j := range n {
			M[i*cols+j] = complex(-s.A[i*s.N+j], 0)
		}
		M[i*cols+i] += x
		for l := range nd {
			M[i*cols+n+l] = complex(-md.B2[i*nd+l], 0)
		}
		for j := range m {
			M[i*cols+k+j] = complex(s.B[i*m+j], 0)
		}
	}
	for l := range nd {
		r := (n + l) * cols
		for j := range n {
			M[r+j] = -d[l] * complex(md.C2[l*s.N+j], 0)
		}
		for l2 := range nd {
			M[r+n+l2] = -d[l] * complex(md.D22[l*nd+l2], 0)
		}
		M[r+n+l]++
		for j := range m {
			M[r+k+j] = d[l] * complex(md.D21[l*m+j], 0)
		}
	}
	if !fzDLSolve(M, k, cols) {
		return nil
	}
	g := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			v := complex(s.D[i*m+j], 0)
			for l := range n {
				v += complex(s.C[i*s.N+l], 0) * M[l*cols+k+j]
			}
			for l := range nd {
				v += complex(md.D12[i*nd+l], 0) * M[(n+l)*cols+k+j]
			}
			g[i*m+j] = v
		}
	}
	return g
}

// fzDLSolve reduces the k×cols augmented M to [I | X] in place with partial
// pivoting; false on an exactly zero pivot.
func fzDLSolve(M []complex128, k, cols int) bool {
	for c := range k {
		piv := c
		for i := c + 1; i < k; i++ {
			if cmplx.Abs(M[i*cols+c]) > cmplx.Abs(M[piv*cols+c]) {
				piv = i
			}
		}
		if M[piv*cols+c] == 0 {
			return false
		}
		for j := range cols {
			M[c*cols+j], M[piv*cols+j] = M[piv*cols+j], M[c*cols+j]
		}
		pv := M[c*cols+c]
		for j := c; j < cols; j++ {
			M[c*cols+j] /= pv
		}
		for i := range k {
			if i == c || M[i*cols+c] == 0 {
				continue
			}
			f := M[i*cols+c]
			for j := c; j < cols; j++ {
				M[i*cols+j] -= f * M[c*cols+j]
			}
		}
	}
	return true
}

func fzDLCRat(z complex128) *fzCRat {
	r := new(fzCRat)
	r.re.Set(fzRat(real(z)))
	r.im.Set(fzRat(imag(z)))
	return r
}

// exact is response evaluated exactly over the rationals at the float64 x
// and delay factors d; ok is false when the system matrix is singular.
func (md *fzDLModel) exact(x complex128, d []complex128) ([]complex128, bool) {
	s := md.s
	n, m, p, nd := s.N, s.M, s.P, md.nd
	k := n + nd
	cols := k + m
	M := make([]*fzCRat, k*cols)
	for i := range M {
		M[i] = new(fzCRat)
	}
	xr := fzDLCRat(x)
	for i := range n {
		for j := range n {
			M[i*cols+j].re.Neg(fzRat(s.A[i*n+j]))
		}
		M[i*cols+i].re.Add(&M[i*cols+i].re, &xr.re)
		M[i*cols+i].im.Set(&xr.im)
		for l := range nd {
			M[i*cols+n+l].re.Neg(fzRat(md.B2[i*nd+l]))
		}
		for j := range m {
			M[i*cols+k+j].re.Set(fzRat(s.B[i*m+j]))
		}
	}
	one := big.NewRat(1, 1)
	for l := range nd {
		r := (n + l) * cols
		dl := fzDLCRat(d[l])
		scaled := func(v float64, neg bool) *fzCRat {
			var c fzCRat
			c.re.Set(fzRat(v))
			z := fzCMul(dl, &c)
			if neg {
				z.re.Neg(&z.re)
				z.im.Neg(&z.im)
			}
			return z
		}
		for j := range n {
			M[r+j] = scaled(md.C2[l*n+j], true)
		}
		for l2 := range nd {
			M[r+n+l2] = scaled(md.D22[l*nd+l2], true)
		}
		M[r+n+l].re.Add(&M[r+n+l].re, one)
		for j := range m {
			M[r+k+j] = scaled(md.D21[l*m+j], false)
		}
	}
	for c := range k {
		piv := -1
		for i := c; i < k; i++ {
			if !M[i*cols+c].isZero() {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range cols {
			M[c*cols+j], M[piv*cols+j] = M[piv*cols+j], M[c*cols+j]
		}
		pv := M[c*cols+c]
		for j := c + 1; j < cols; j++ {
			M[c*cols+j] = fzCQuo(M[c*cols+j], pv)
		}
		M[c*cols+c] = fzDLCRat(1)
		for i := range k {
			if i == c || M[i*cols+c].isZero() {
				continue
			}
			f := M[i*cols+c]
			for j := c + 1; j < cols; j++ {
				t := fzCMul(f, M[c*cols+j])
				M[i*cols+j].re.Sub(&M[i*cols+j].re, &t.re)
				M[i*cols+j].im.Sub(&M[i*cols+j].im, &t.im)
			}
			M[i*cols+c] = new(fzCRat)
		}
	}
	g := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			var acc fzCRat
			acc.re.Set(fzRat(s.D[i*m+j]))
			add := func(c float64, z *fzCRat) {
				cr := fzRat(c)
				acc.re.Add(&acc.re, new(big.Rat).Mul(cr, &z.re))
				acc.im.Add(&acc.im, new(big.Rat).Mul(cr, &z.im))
			}
			for l := range n {
				add(s.C[i*n+l], M[l*cols+k+j])
			}
			for l := range nd {
				add(md.D12[i*nd+l], M[(n+l)*cols+k+j])
			}
			re, _ := acc.re.Float64()
			im, _ := acc.im.Float64()
			g[i*m+j] = complex(re, im)
		}
	}
	return g, true
}

func (md *fzDLModel) point(w float64) complex128 {
	if md.s.Discrete {
		return cmplx.Exp(complex(0, w*md.s.Dt))
	}
	return complex(0, w)
}

// sigmaAt is the exact σmax at ω; ok is false at a singular point or where
// a one-ulp change of the delay factors moves it by more than 1e-9
// relative (the float64 response itself is not resolved there).
func (md *fzDLModel) sigmaAt(w float64) (float64, bool) {
	s := md.s
	d := md.delta(w)
	g, ok := md.exact(md.point(w), d)
	if !ok {
		return 0, false
	}
	v := fzSigmaMax(g, s.P, s.M)
	for i := range d {
		d[i] *= complex(1+0x1p-52, 0)
	}
	g2, ok := md.exact(md.point(w), d)
	if !ok {
		return 0, false
	}
	v2 := fzSigmaMax(g2, s.P, s.M)
	gf := md.response(md.point(w), md.delta(w), true)
	if gf == nil {
		return 0, false
	}
	vf := fzSigmaMax(gf, s.P, s.M)
	if math.Abs(v2-v) > 1e-9*v || math.Abs(vf-v) > 1e-8*v {
		return v, false
	}
	return v, true
}

// fzDLGrid adds to fzFreqGrid a uniform grid of spacing π/(8·τmax) (at most
// 3000 points) up to beyond the pole region.
func fzDLGrid(poles []complex128, dt float64, taus []float64) []float64 {
	g := fzFreqGrid(poles, dt, 60)
	wmax := g[len(g)-1]
	if dt > 0 {
		wmax = math.Pi / dt
	} else {
		tmax := slices.Max(taus)
		wmax = math.Min(math.Max(10*wmax, 200/slices.Min(taus)), 1e6)
		step := math.Max(math.Pi/(8*tmax), wmax/3000)
		for w := step; w < wmax; w += step {
			g = append(g, w)
		}
	}
	for i := range 3000 {
		g = append(g, wmax*float64(i)/3000)
	}
	slices.Sort(g)
	return slices.Compact(g)
}

// torusSup is the sup of f over the class torus [0, 2π)^dim, the phases of
// delay i being mult_i·φ_class(i): a grid plus golden-section coordinate
// refinement of its best points. It also returns the smallest
// σmin(I − ΔD22) seen at the maximiser.
func fzDLTorusSup(dim int, kmax int, f func(phi []float64) float64) (float64, []float64) {
	N := 64 * max(kmax, 1)
	if dim == 2 {
		N = 24 * max(kmax, 4)
	}
	type pt struct {
		v   float64
		phi []float64
	}
	var top []pt
	push := func(phi []float64) {
		v := f(phi)
		top = append(top, pt{v, slices.Clone(phi)})
		slices.SortFunc(top, func(a, b pt) int {
			if a.v > b.v {
				return -1
			}
			if a.v < b.v {
				return 1
			}
			return 0
		})
		if len(top) > 4 {
			top = top[:4]
		}
	}
	step := 2 * math.Pi / float64(N)
	if dim == 1 {
		for i := range N {
			push([]float64{float64(i) * step})
		}
	} else {
		for i := range N {
			for j := range N {
				push([]float64{float64(i) * step, float64(j) * step})
			}
		}
	}
	best, bphi := math.Inf(-1), []float64(nil)
	const gr = 0.6180339887498949
	for _, p := range top {
		phi := slices.Clone(p.phi)
		for range 4 {
			for c := range dim {
				a, b := phi[c]-step, phi[c]+step
				g := func(x float64) float64 {
					q := slices.Clone(phi)
					q[c] = x
					return f(q)
				}
				x1, x2 := b-gr*(b-a), a+gr*(b-a)
				f1, f2 := g(x1), g(x2)
				for range 40 {
					if f1 > f2 {
						b, x2, f2 = x2, x1, f1
						x1 = b - gr*(b-a)
						f1 = g(x1)
					} else {
						a, x1, f1 = x1, x2, f2
						x2 = a + gr*(b-a)
						f2 = g(x2)
					}
				}
				if v := math.Max(f1, f2); v > g(phi[c]) {
					phi[c] = x1
					if f2 > f1 {
						phi[c] = x2
					}
				}
			}
		}
		if v := f(phi); v > best {
			best, bphi = v, phi
		}
	}
	return best, bphi
}

func (md *fzDLModel) torusDelta(phi []float64) []complex128 {
	d := make([]complex128, md.nd)
	for i := range d {
		d[i] = cmplx.Exp(complex(0, -float64(md.mult[i])*phi[md.class[i]]))
	}
	return d
}

func (md *fzDLModel) torusGain(phi []float64) float64 {
	g := md.response(0, md.torusDelta(phi), false)
	if g == nil {
		return math.Inf(1)
	}
	return fzSigmaMax(g, md.s.P, md.s.M)
}

// diffSigmaMin is σmin(I − ΔD22) at the torus point phi.
func (md *fzDLModel) diffSigmaMin(phi []float64) float64 {
	nd := md.nd
	d := md.torusDelta(phi)
	M := make([]complex128, nd*nd)
	for i := range nd {
		for j := range nd {
			M[i*nd+j] = -d[i] * complex(md.D22[i*nd+j], 0)
		}
		M[i*nd+i]++
	}
	sv := fzSingularValues(M, nd, nd)
	return sv[len(sv)-1]
}

// FuzzDelayLFT: for internal-delay LFT models with up to three delays
// (commensurate, incommensurate, neutral D22), IsStable matches the
// certified root count of χ (input delays never change it), HinfNorm is
// +Inf exactly when unstable, and a finite norm is attained at its ω, never
// under-reported by an exact-confirmed dense sweep nor by the phase-torus
// sup of the high-frequency LFT, and at ω = +Inf matches that sup.
func FuzzDelayLFT(f *testing.F) {
	for _, s := range fzDLSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		md := fzDLNewModel(t, s)
		q := md.quasi()
		verdict, why := fzDLVerdict(q, s.Discrete, 1e-6*math.Max(1, s.NormJ), md.leadZ(q))
		if verdict == 0 {
			fzSkip(t, why)
		}
		sys := md.system(t, false)
		st, err := sys.IsStable()
		fzContract(t, s, "IsStable(LFT)", st || err == nil, err)
		stErr := err
		dst, derr := md.system(t, true).IsStable()
		fzContract(t, s, "IsStable(LFT+in)", dst || derr == nil, derr)
		if (stErr == nil) != (derr == nil) || stErr == nil && dst != st {
			t.Fatalf("IsStable with input delays = %v, %v; without %v, %v\n%s", dst, derr, st, stErr, md)
		}
		norm, omega, herr := HinfNorm(sys)
		if herr == nil && (math.IsNaN(norm) || math.IsNaN(omega) || norm < 0) {
			t.Fatalf("HinfNorm = %v at %v\n%s", norm, omega, md)
		}
		if stErr != nil {
			fzErr(t, s, "IsStable(LFT)", stErr, ErrDelayUnsupported)
		}
		if want := verdict > 0; st != want {
			t.Fatalf("IsStable(LFT) = %v, certified χ verdict stable=%v\n%s", st, want, md)
		}
		fzErr(t, s, "HinfNorm(LFT)", herr, ErrDelayUnsupported)
		if verdict < 0 {
			if !math.IsInf(norm, 1) {
				t.Fatalf("HinfNorm(LFT) = %v at %v for a certified unstable χ\n%s", norm, omega, md)
			}
			return
		}
		if math.IsInf(norm, 0) {
			t.Fatalf("HinfNorm(LFT) = %v for a certified stable χ\n%s", norm, md)
		}
		md.checkPeak(t, norm, omega)
	})
}

func (md *fzDLModel) checkPeak(t *testing.T, norm, omega float64) {
	t.Helper()
	s := md.s
	const tol = 1e-6
	if math.IsInf(omega, 1) {
		if s.Discrete {
			t.Fatalf("HinfNorm(LFT) omega = +Inf for a discrete model\n%s", md)
		}
	} else {
		if omega < 0 || s.Discrete && omega > math.Pi/s.Dt*(1+1e-12) {
			t.Fatalf("HinfNorm(LFT) omega = %v outside [0, Nyquist]\n%s", omega, md)
		}
		if v, ok := md.sigmaAt(omega); ok && v < norm*(1-tol) {
			t.Fatalf("HinfNorm(LFT) = %v at ω=%v, but exact σmax there is %v (over-reported)\n%s", norm, omega, v, md)
		}
	}
	wbest, _ := fzSweepMax(func(w float64) float64 {
		g := md.response(md.point(w), md.delta(w), true)
		if g == nil {
			return 0
		}
		return fzSigmaMax(g, s.P, s.M)
	}, fzDLGrid(s.S, s.Dt, md.tau), 3)
	if v, ok := md.sigmaAt(wbest); ok && norm < v*(1-tol) {
		t.Fatalf("HinfNorm(LFT) = %v at ω=%v, but exact σmax at ω=%v is %v (under-reported)\n%s", norm, omega, wbest, v, md)
	}
	if s.Discrete {
		return
	}
	kmax := slices.Max(md.mult)
	sup, phi := fzDLTorusSup(md.dim, kmax, md.torusGain)
	if math.IsInf(sup, 1) {
		fzSkip(t, "difference operator singular on the torus")
	}
	if norm < sup*(1-tol) {
		t.Fatalf("HinfNorm(LFT) = %v at ω=%v below the high-frequency torus sup %v at φ=%v (under-reported)\n%s", norm, omega, sup, phi, md)
	}
	if math.IsInf(omega, 1) && md.diffSigmaMin(phi) > 1e-3 && norm > sup*(1+tol) {
		t.Fatalf("HinfNorm(LFT) = %v at ω=+Inf above the high-frequency torus sup %v\n%s", norm, sup, md)
	}
}

// fzDLLoop is the SISO loop L = G·e^{−sτ1} + d2·e^{−sτ2}, G = D + C(sI−A)⁻¹B
// from channel (0,0) of the spec.
type fzDLLoop struct {
	s         *fzSpec
	b, c      []float64
	d, d2     float64
	tau1      float64
	tau2      float64
	class     []int
	mult      []int
	dim       int
	den, num  []*big.Rat
	denF, nmF []float64
}

func fzDLNewLoop(t *testing.T, s *fzSpec) (*fzDLLoop, *System) {
	t.Helper()
	if s.Discrete {
		fzSkip(t, "FuzzDelayLoop: continuous only")
	}
	if s.Tau == 0 {
		fzSkip(t, "no loop delay")
	}
	if s.N > fzDLMaxOrder {
		fzSkip(t, "order cap")
	}
	n := s.N
	l := &fzDLLoop{s: s, b: make([]float64, n), c: make([]float64, n), d: s.D[0]}
	for i := range n {
		l.b[i], l.c[i] = s.B[i*s.M], s.C[i]
	}
	l.tau1 = fzDLTrunc(s.Tau, 20)
	tau, class, mult, dim := fzDLPattern(t, l.tau1, math.Min(s.Vals[13], 29), 0, false)
	l.tau2 = tau[len(tau)-1]
	if l.tau2 > fzDLMaxTau {
		fzSkip(t, "delay range cap")
	}
	l.class, l.mult, l.dim = []int{0, class[len(tau)-1]}, []int{1, mult[len(tau)-1]}, dim
	if v := s.Vals[14]; v >= -3 {
		l.d2 = fzClamp(v, -3, 3) / 2
	} else {
		l.class, l.mult, l.dim = []int{0, 0}, []int{1, 1}, 1
	}
	G, err := NewFromSlices(n, 1, 1, s.A, l.b, l.c, []float64{l.d}, 0)
	if err != nil {
		t.Fatalf("NewFromSlices: %v\n%s", err, s)
	}
	if err := G.SetInputDelay([]float64{l.tau1}); err != nil {
		t.Fatalf("SetInputDelay: %v\n%s", err, s)
	}
	L := G
	if l.d2 != 0 {
		K, err := NewGain(mat.NewDense(1, 1, []float64{l.d2}), 0)
		if err != nil {
			t.Fatalf("NewGain: %v", err)
		}
		if err := K.SetInputDelay([]float64{l.tau2}); err != nil {
			t.Fatalf("SetInputDelay: %v", err)
		}
		if L, err = Parallel(G, K); err != nil {
			t.Fatalf("Parallel: %v\n%s", err, l)
		}
	}
	l.den = fzCharPolyExact(s.A, n)
	l.num = fzPencilDetRat(n, 1, s.A, l.b, l.c, []float64{l.d})
	l.denF, l.nmF = fzRatFloats(l.den), fzRatFloats(l.num)
	return l, L
}

func (l *fzDLLoop) String() string {
	return l.s.String() + "\nloop: tau1=" + fzDLFmt([]float64{l.tau1}) + " d2=" + fzDLFmt([]float64{l.d2}) + " tau2=" + fzDLFmt([]float64{l.tau2})
}

// quasi returns χ_cl = den + num·e^{−sτ1} + d2·den·e^{−sτ2} = den·(1+L).
func (l *fzDLLoop) quasi() *fzDLQuasi {
	theta := []float64{0, l.tau1}
	polys := [][]*big.Rat{l.den, l.num}
	if l.d2 != 0 {
		p := make([]*big.Rat, len(l.den))
		for i, c := range l.den {
			p[i] = new(big.Rat).Mul(c, fzRat(l.d2))
		}
		theta, polys = append(theta, l.tau2), append(polys, p)
	}
	return fzDLNewQuasi(theta, polys)
}

// at returns L(jω) with exact rational N(jω)/den(jω) and a relative error
// estimate of 1+L.
func (l *fzDLLoop) at(w float64) (complex128, float64, bool) {
	wr := fzRat(w)
	ev := func(p []*big.Rat) *fzCRat {
		var acc fzCRat
		x := new(fzCRat)
		x.im.Set(wr)
		for _, c := range p {
			acc = *fzCMul(&acc, x)
			acc.re.Add(&acc.re, c)
		}
		return &acc
	}
	dn := ev(l.den)
	if dn.isZero() {
		return 0, 0, false
	}
	g := fzCQuo(ev(l.num), dn)
	re, _ := g.re.Float64()
	im, _ := g.im.Float64()
	G := complex(re, im)
	L := G*cmplx.Exp(complex(0, -w*l.tau1)) + complex(l.d2, 0)*cmplx.Exp(complex(0, -w*l.tau2))
	scale := cmplx.Abs(G)*(1+w*l.tau1) + math.Abs(l.d2)*(1+w*l.tau2)
	return L, 4 * 0x1p-52 * scale / cmplx.Abs(1+L), true
}

// atFloat is L(jω) by float64 Horner, used only to locate peaks.
func (l *fzDLLoop) atFloat(w float64) complex128 {
	x := complex(0, w)
	G := polyEval(l.nmF, x) / polyEval(l.denF, x)
	return G*cmplx.Exp(complex(0, -w*l.tau1)) + complex(l.d2, 0)*cmplx.Exp(complex(0, -w*l.tau2))
}

// sens returns |1/(1+L) − c| at ω and whether it is resolved to 1e-9.
func (l *fzDLLoop) sens(w, c float64) (float64, bool) {
	L, rel, ok := l.at(w)
	if !ok {
		return 0, false
	}
	v := cmplx.Abs(1/(1+L) - complex(c, 0))
	return v, rel*cmplx.Abs(1/(1+L)) <= 1e-9*v
}

func (l *fzDLLoop) checkPeak(t *testing.T, name string, peak, w float64, c float64, fz func(float64) (float64, bool)) {
	t.Helper()
	const tol = 1e-6
	if !(w >= 0) {
		t.Fatalf("%s peak frequency %v\n%s", name, w, l)
	}
	if !math.IsInf(w, 1) {
		if v, ok := fz(w); ok && v < peak*(1-tol) {
			t.Fatalf("%s = %v at ω=%v, but exact value there is %v (over-reported)\n%s", name, peak, w, v, l)
		}
	}
	taus := []float64{l.tau1}
	if l.d2 != 0 {
		taus = append(taus, l.tau2)
	}
	wbest, _ := fzSweepMax(func(w float64) float64 {
		L := l.atFloat(w)
		return cmplx.Abs(1/(1+L) - complex(c, 0))
	}, fzDLGrid(l.s.S, 0, taus), 3)
	if v, ok := fz(wbest); ok && peak < v*(1-tol) {
		t.Fatalf("%s = %v at ω=%v, but exact value at ω=%v is %v (under-reported)\n%s", name, peak, w, wbest, v, l)
	}
	minRet := math.Inf(1)
	sup, phi := fzDLTorusSup(l.dim, l.mult[1], func(phi []float64) float64 {
		L := complex(l.d, 0)*cmplx.Exp(complex(0, -phi[l.class[0]])) + complex(l.d2, 0)*cmplx.Exp(complex(0, -float64(l.mult[1])*phi[l.class[1]]))
		minRet = math.Min(minRet, cmplx.Abs(1+L))
		return cmplx.Abs(1/(1+L) - complex(c, 0))
	})
	if peak < sup*(1-tol) {
		t.Fatalf("%s = %v at ω=%v below the high-frequency torus sup %v at φ=%v (under-reported)\n%s", name, peak, w, sup, phi, l)
	}
	if math.IsInf(w, 1) && minRet > 1e-3 && peak > sup*(1+tol) {
		t.Fatalf("%s = %v at ω=+Inf above the high-frequency torus sup %v\n%s", name, peak, sup, l)
	}
}

// FuzzDelayLoop: for continuous SISO loops L = G·e^{−sτ1} + d2·e^{−sτ2}
// (integrators and lightly damped poles included) DiskMargin's verdict and
// Feedback(L)'s IsStable and HinfNorm match the certified root count of
// den·(1+L); a stable loop's Alpha, PeakSensitivity and ‖T‖∞ are attained
// at their frequencies and never beaten by an exact-confirmed dense sweep.
func FuzzDelayLoop(f *testing.F) {
	for _, s := range fzDLLoopSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		l, L := fzDLNewLoop(t, s)
		if math.Abs(1+l.d+l.d2) < 1e-8 || math.Abs(1+l.d) < 1e-8 {
			fzSkip(t, "algebraic loop 1 + L(∞) ≈ 0")
		}
		verdict, why := fzDLVerdict(l.quasi(), false, 1e-6*math.Max(1, s.NormJ), nil)
		if verdict == 0 {
			fzSkip(t, why)
		}
		dm, dmErr := DiskMargin(L)
		fzContract(t, s, "DiskMargin(delay)", dm != nil, dmErr)
		T, err := Feedback(L, nil, -1)
		if err != nil {
			t.Fatalf("Feedback: %v\n%s", err, l)
		}
		st, stErr := T.IsStable()
		fzContract(t, s, "IsStable(T)", st || stErr == nil, stErr)
		norm, omega, hErr := HinfNorm(T)
		if stErr == nil && st != (verdict > 0) {
			t.Fatalf("Feedback(L).IsStable = %v, certified χ_cl verdict stable=%v\n%s", st, verdict > 0, l)
		}
		if hErr == nil && math.IsInf(norm, 1) != (verdict < 0) {
			t.Fatalf("HinfNorm(Feedback(L)) = %v at %v, certified χ_cl verdict stable=%v\n%s", norm, omega, verdict > 0, l)
		}
		if dmErr == nil {
			if verdict < 0 {
				if dm.Alpha != 0 || !math.IsInf(dm.PeakSensitivity, 1) || dm.GainMargin != [2]float64{1, 1} || dm.PhaseMargin != 0 {
					t.Fatalf("DiskMargin %+v for a certified unstable closed loop\n%s", dm, l)
				}
			} else {
				if !(dm.Alpha > 0) || math.IsInf(dm.PeakSensitivity, 0) {
					t.Fatalf("DiskMargin Alpha=%v Ms=%v for a certified stable closed loop\n%s", dm.Alpha, dm.PeakSensitivity, l)
				}
				w, _ := dm.Frequency()
				l.checkPeak(t, "DiskMargin 1/Alpha = ‖S−½‖∞", 1/dm.Alpha, w, 0.5, func(w float64) (float64, bool) { return l.sens(w, 0.5) })
				w, _ = dm.PeakFreq()
				l.checkPeak(t, "DiskMargin PeakSensitivity", dm.PeakSensitivity, w, 0, func(w float64) (float64, bool) { return l.sens(w, 0) })
				a := dm.Alpha
				lo, hi := 0.0, math.Inf(1)
				if !math.IsInf(a, 1) {
					lo = max((2-a)/(2+a), 0)
				}
				if a < 2 {
					hi = (2 + a) / (2 - a)
				}
				pm := 2 * math.Atan(a/2) * 180 / math.Pi
				near := func(x, y float64) bool { return x == y || math.Abs(x-y) <= 1e-9*math.Max(1, math.Abs(y)) }
				if !near(dm.GainMargin[0], lo) || !near(dm.GainMargin[1], hi) || !near(dm.PhaseMargin, pm) {
					t.Fatalf("DiskMargin GM %v PM %v for Alpha %v, want [%v %v] %v\n%s", dm.GainMargin, dm.PhaseMargin, a, lo, hi, pm, l)
				}
			}
		}
		if hErr == nil && verdict > 0 {
			l.checkPeak(t, "HinfNorm(Feedback(L))", norm, omega, 1, func(w float64) (float64, bool) { return l.sens(w, 1) })
		}
		for _, e := range []error{dmErr, stErr, hErr} {
			if e != nil {
				fzErr(t, s, "delay loop", e, ErrDelayUnsupported)
			}
		}
	})
}

func fzDLSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: 0, Gain: 6, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out = 1, 0.2, 1, -0.5, -1 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 2, 0.3, 2, -0.3, -2, 1 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 2, 0, 1.5, -1, 2+1e-9, -1 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 3, 2, 2
			s.Mag, s.Damp = [3]float64{0.5, 0.2}, [3]float64{1.2, 3}
			s.Tau, s.Out, s.IO, s.Phase, s.In = -0.7, 35.5, 0.4, 0.33, -1
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 1, 0, 1, -0.4, -3, 2.9 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 2, -0.2, 2, -0.6, 3, 1.6 }),
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.Mag[1], s.Damp[1] = 4, 0.4, 3, 0.4, 3
			s.Tau, s.Out = -0.1, -2
		}),
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.Tau, s.IO = 0, 0, 0, -0.5, 1.2
			s.M, s.P, s.Out = 2, 1, 1.37
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 2, 0.5, 2, math.Log10(3), -2, 0.8
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -0.5
			s.N, s.M, s.P = 3, 2, 1
			s.Mag, s.Damp = [3]float64{0.2, -0.3}, [3]float64{1.5, 1}
			s.Tau, s.Out, s.IO, s.Phase = 0, 33.4, -1, 1.1
		}),
	}
}

func fzDLLoopSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: -1, Gain: 6, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Gain = 2, 0, 2, 6.5 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO = 2, 0.2, 1, -0.5, -2, 0.6 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO, s.DScal = 1, 0, 1, -0.3, -1, -0.5, 0.3 }),
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.Mag[1] = 3, 0.3, 1.5, -31
			s.Tau, s.Gain = -1.6, 5.5
		}),
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.Mag[1], s.Damp[1] = 4, 0, 4, 0, 4
			s.Tau, s.Gain = -2, 5
		}),
		with(func(s *fzSeed) {
			s.N, s.Mag, s.Damp = 3, [3]float64{-2, -31}, [3]float64{1.3, 1}
			s.Tau = -2.5
		}),
		with(func(s *fzSeed) {
			s.N, s.Mag, s.Damp = 4, [3]float64{0.3, 0.3}, [3]float64{1.5, 1.5}
			s.Tau, s.Out, s.IO = -0.8, 2+1e-9, 0.2
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Out, s.IO, s.Gain = 2, 1, 2.5, -0.27, -2, 0.4, 6.5 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Gain = 2, 0, 1.7, 0, 7 }),
	}
}

// TestFzDLOracles checks the certified counter against analytic and exact
// counts: s + a·e^{−sτ} is stable iff aτ < π/2; delay-free χ matches Routh;
// single-delay χ matches fzQuasiRHP; a neutral case with |d| < 1.
func TestFzDLOracles(t *testing.T) {
	r := func(v ...float64) []*big.Rat { return fzRatPoly(v) }
	cases := []struct {
		name  string
		theta []float64
		polys [][]*big.Rat
		want  int
	}{
		{"s+e^{-s}", []float64{0, 1}, [][]*big.Rat{r(1, 0), r(1)}, 0},
		{"s+e^{-2s}", []float64{0, 2}, [][]*big.Rat{r(1, 0), r(1)}, 2},
		{"(s-1)(s+2)", []float64{0}, [][]*big.Rat{r(1, 1, -2)}, 1},
		{"s^2+0.5s+1+0.3e^{-s}", []float64{0, 1}, [][]*big.Rat{r(1, 0.5, 1), r(0.3)}, 0},
		{"neutral s+1+0.5se^{-s}", []float64{0, 1}, [][]*big.Rat{r(1, 1), r(0.5, 0)}, 0},
		{"neutral s-0.5+0.5se^{-s}", []float64{0, 1}, [][]*big.Rat{r(1, -0.5), r(0.5, 0)}, 1},
		{"two delays", []float64{0, 1, 1.4142135623730951}, [][]*big.Rat{r(1, 2, 1), r(0.5), r(0.4, 0)}, -1},
	}
	for _, c := range cases {
		q := fzDLNewQuasi(c.theta, c.polys)
		got, ok := q.countRight(0, fzDLBudget)
		if !ok {
			t.Fatalf("%s: not certified", c.name)
		}
		want := c.want
		if want < 0 {
			num := fzRatFloats(c.polys[1])
			den := fzRatFloats(c.polys[0])
			if len(c.polys) == 3 {
				want, ok = fzDLDenseCount(func(s complex128) complex128 { v, _ := q.eval(s); return v }, 64)
			} else {
				want, ok = fzQuasiRHP(num, den, c.theta[1])
			}
			if !ok {
				t.Fatalf("%s: reference oracle failed", c.name)
			}
		}
		if got != want {
			t.Errorf("%s: count %d, want %d", c.name, got, want)
		}
	}
	for _, tau := range []float64{0.1, 0.7, 2} {
		den, num := []float64{1, 1.3, 2.2, 0.4}, []float64{0.8, -0.5}
		want, ok := fzQuasiRHP(num, den, tau)
		if !ok {
			continue
		}
		q := fzDLNewQuasi([]float64{0, tau}, [][]*big.Rat{fzRatPoly(den), fzRatPoly(num)})
		if got, ok := q.countRight(0, fzDLBudget); !ok || got != want {
			t.Errorf("τ=%v: count %d (%v), fzQuasiRHP %d", tau, got, ok, want)
		}
	}
	q := fzDLNewQuasi([]float64{0, 2}, [][]*big.Rat{r(1, -0.5), r(0.3)})
	p := q.discrete()
	if got, ok := fzSchurCohnOutside(p, 1); !ok || got != 0 {
		t.Errorf("discrete z^3 − 0.5z^2 + 0.3: outside %d (%v), want 0", got, ok)
	}
}

// fzDLDenseCount is an uncertified dense argument-principle count on
// [0, box]×[−box, box] used only to cross-check countRight.
func fzDLDenseCount(chi func(complex128) complex128, box float64) (int, bool) {
	corners := []complex128{complex(0, -box), complex(box, -box), complex(box, box), complex(0, box), complex(0, -box)}
	total := 0.0
	const n = 200000
	for e := range 4 {
		a, b := corners[e], corners[e+1]
		prev := chi(a)
		for i := 1; i <= n; i++ {
			v := chi(a + (b-a)*complex(float64(i)/n, 0))
			if v == 0 {
				return 0, false
			}
			total += cmplx.Phase(v / prev)
			prev = v
		}
	}
	turns := total / (2 * math.Pi)
	return int(math.Round(turns)), math.Abs(turns-math.Round(turns)) < 1e-6
}
