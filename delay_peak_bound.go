package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// peakCertTol is the relative tolerance of a certified delay peak: the
// supremum over the certified band is at most (1 + peakCertTol) times the
// reported peak, itself an attained value.
const peakCertTol = 1e-9

// descriptorResponse evaluates a continuous model with internal delays as
//
//	G(ω) = D + Ĉ·M(ω)⁻¹·B̃,  M(ω) = M₀ + diag(jω·Iₙ, e^{jωτ₁}, …, e^{jωτ_nd}),
//
// the descriptor form in the states x and delay outputs v, where
// (jω − A)x − B2·v = B·u and Δ(ω)*·v − C2·x − D22·v = D21·u, with Ĉ = [C D12]
// and B̃ = [B; D21]. M is affine in jω and the unit-modulus e^{jωτ}, so
// |dM/dω| ≤ W = diag(Iₙ, τ) entrywise on the diagonal, and det M is the
// characteristic function χ(jω) up to a unit factor; M is regular on the
// axis whenever χ is.
type descriptorResponse struct {
	n, size, p, m int
	m0            []complex128 // size×size
	c             []complex128 // p×size
	b             []complex128 // size×m
	d             []complex128 // p×m
	tau           []float64
	weight        []float64 // diagonal of W
	svd           *complexSVDWorkspace

	lu, inv, tmp []complex128
	x, y, unit   []complex128
	v, u         []complex128
	maxWeight    float64
	piv          []int
	evals        int
}

// newDescriptorResponse builds the descriptor form of sys with its I/O
// delays pulled into the delay channels.
func newDescriptorResponse(sys *System) (*descriptorResponse, error) {
	pulled, err := sys.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	n, m, p := pulled.Dims()
	var tau []float64
	var b2, c2, d12, d21, d22 *mat.Dense
	if lft := pulled.LFT; lft != nil {
		tau, b2, c2, d12, d21, d22 = lft.Tau, lft.B2, lft.C2, lft.D12, lft.D21, lft.D22
	}
	nd := len(tau)
	size := n + nd
	r := &descriptorResponse{
		n: n, size: size, p: p, m: m,
		m0: make([]complex128, size*size), c: make([]complex128, p*size),
		b: make([]complex128, size*m), d: make([]complex128, p*m),
		tau: tau, weight: make([]float64, size),
		lu: make([]complex128, size*size), inv: make([]complex128, size*size),
		tmp: make([]complex128, p*m), piv: make([]int, size),
		x: make([]complex128, p*size), y: make([]complex128, size*m), unit: make([]complex128, size),
		v: make([]complex128, size*m), u: make([]complex128, size*m),
	}
	at := func(x *mat.Dense, i, j int) complex128 {
		if x == nil || x.IsEmpty() {
			return 0
		}
		return complex(x.At(i, j), 0)
	}
	for i := range n {
		for j := range n {
			r.m0[i*size+j] = -at(pulled.A, i, j)
		}
		for k := range nd {
			r.m0[i*size+n+k] = -at(b2, i, k)
			r.m0[(n+k)*size+i] = -at(c2, k, i)
		}
		for j := range m {
			r.b[i*m+j] = at(pulled.B, i, j)
		}
		for j := range p {
			r.c[j*size+i] = at(pulled.C, j, i)
		}
		r.weight[i] = 1
	}
	for k := range nd {
		for l := range nd {
			r.m0[(n+k)*size+n+l] = -at(d22, k, l)
		}
		for j := range m {
			r.b[(n+k)*m+j] = at(d21, k, j)
		}
		for j := range p {
			r.c[j*size+n+k] = at(d12, j, k)
		}
		r.weight[n+k] = tau[k]
	}
	for _, w := range r.weight {
		r.maxWeight = max(r.maxWeight, w)
	}
	for i := range p {
		for j := range m {
			r.d[i*m+j] = at(pulled.D, i, j)
		}
	}
	if p > 2 || m > 2 {
		r.svd = newComplexSVDWorkspace(p, m)
	}
	r.balance()
	return r, nil
}

// balance applies the diagonal similarity M₀ → D⁻¹M₀D, Ĉ → ĈD, B̃ → D⁻¹B̃
// with powers of 2 that equalise the off-diagonal row and column norms of M₀,
// as LAPACK gebal does. G and the diagonal of M(ω) − M₀ are unchanged, while
// the norms of M⁻¹ that enter the bounds, and the rounding errors, shrink.
func (r *descriptorResponse) balance() {
	size := r.size
	for changed := true; changed; {
		changed = false
		for i := range size {
			var c, rn float64
			for j := range size {
				if j != i {
					c += cabs1(r.m0[j*size+i])
					rn += cabs1(r.m0[i*size+j])
				}
			}
			if c == 0 || rn == 0 {
				continue
			}
			sum, f := c+rn, 1.0
			for c < rn/2 {
				f, c = 2*f, 4*c
			}
			for c > 2*rn {
				f, c = f/2, c/4
			}
			if c+rn >= 0.95*sum*f {
				continue
			}
			changed = true
			g := complex(1/f, 0)
			fc := complex(f, 0)
			for j := range size {
				r.m0[i*size+j] *= g
				r.m0[j*size+i] *= fc
			}
			for j := range r.m {
				r.b[i*r.m+j] *= g
			}
			for j := range r.p {
				r.c[j*size+i] *= fc
			}
		}
	}
}

// sensitivity turns the SISO loop L = D + Ĉ·M⁻¹·B̃ into S = 1/(1+L) by
// Sherman–Morrison: with β = 1/(1+D), S = β − β²·Ĉ·(M + β·B̃Ĉ)⁻¹·B̃, whose
// descriptor matrix is singular exactly at the closed-loop roots.
func (r *descriptorResponse) sensitivity() error {
	if r.p != 1 || r.m != 1 {
		return fmt.Errorf("sensitivity of a %d×%d loop: %w", r.p, r.m, ErrDimensionMismatch)
	}
	if r.d[0] == -1 {
		return fmt.Errorf("loop feedthrough −1: %w", errDelayLoopUnsupported)
	}
	beta := 1 / (1 + r.d[0])
	for i := range r.size {
		for j := range r.size {
			r.m0[i*r.size+j] += beta * r.b[i] * r.c[j]
		}
	}
	for j := range r.c {
		r.c[j] *= -beta * beta
	}
	r.d[0] = beta
	r.balance()
	return nil
}

// descriptorSample holds G and dG/dω at ω with the norms that bound
// G″ near ω: Z = M(ω)⁻¹, αW = ‖ĈZW‖, β = ‖ZB̃‖, βW = ‖WZB̃‖, ζ = ‖ZW‖ and
// ζL = ‖WZ‖, all Frobenius norms, which bound the spectral ones.
type descriptorSample struct {
	w                         float64
	g, g1                     []complex128
	g2                        float64 // ‖G″(ω)‖_F
	alphaW, beta, betaW, zeta float64
	zetaL, xi                 float64
	regular                   bool
}

func (r *descriptorResponse) sample(w float64) *descriptorSample {
	r.evals++
	gs := make([]complex128, 2*r.p*r.m)
	s := &descriptorSample{w: w, g: gs[:r.p*r.m], g1: gs[r.p*r.m:]}
	n, size := r.n, r.size
	copy(r.lu, r.m0)
	unit := r.unit
	for i := range size {
		if i < n {
			unit[i] = 1
			r.lu[i*size+i] += complex(0, w)
		} else {
			unit[i] = cmplx.Exp(complex(0, w*r.tau[i-n]))
			r.lu[i*size+i] += unit[i]
		}
	}
	copy(s.g, r.d)
	if size == 0 {
		s.regular = true
		return s
	}
	if !cInvert(r.lu, r.piv, size) {
		return s
	}
	r.inv, r.lu = r.lu, r.inv
	for i := range size {
		for j := range size {
			z := sq(r.inv[i*size+j])
			wi, wj := r.weight[i]*r.weight[i], r.weight[j]*r.weight[j]
			s.zeta += z * wj
			s.zetaL += z * wi
			s.xi += z * wi * wj
		}
	}
	x := r.x
	clear(x)
	for i := range r.p {
		for k := range size {
			if c := r.c[i*size+k]; c != 0 {
				for j := range size {
					x[i*size+j] += c * r.inv[k*size+j]
				}
			}
		}
		for j := range size {
			s.alphaW += sq(x[i*size+j]) * r.weight[j] * r.weight[j]
		}
	}
	y := r.y
	clear(y)
	for i := range size {
		for k := range size {
			if z := r.inv[i*size+k]; z != 0 {
				for j := range r.m {
					y[i*r.m+j] += z * r.b[k*r.m+j]
				}
			}
		}
		for j := range r.m {
			v := sq(y[i*r.m+j])
			s.beta += v
			s.betaW += v * r.weight[i] * r.weight[i]
		}
	}
	for i := range r.p {
		for j := range r.m {
			var g, g1 complex128
			for k := range size {
				g += r.c[i*size+k] * y[k*r.m+j]
				g1 += x[i*size+k] * complex(0, -r.weight[k]) * unit[k] * y[k*r.m+j]
			}
			s.g[i*r.m+j] += g
			s.g1[i*r.m+j] = g1
		}
	}
	s.g2 = r.secondDerivativeNorm(x, y)
	s.alphaW, s.beta, s.betaW = math.Sqrt(s.alphaW), math.Sqrt(s.beta), math.Sqrt(s.betaW)
	s.zeta, s.zetaL, s.xi = math.Sqrt(s.zeta), math.Sqrt(s.zetaL), math.Sqrt(s.xi)
	s.regular = true
	return s
}

// secondDerivativeNorm returns ‖G″‖_F = ‖X(2M′ZM′ − M″)Y‖_F with X = ĈZ,
// Y = ZB̃, M′ = jWU and M″ = −W²U on the delay rows and 0 on the states, U
// the unit-modulus factor of M(ω) − M₀.
func (r *descriptorResponse) secondDerivativeNorm(x, y []complex128) float64 {
	size, m := r.size, r.m
	v, u := r.v, r.u
	for k := range size {
		wu := complex(r.weight[k], 0) * r.unit[k]
		for j := range m {
			v[k*m+j] = wu * y[k*m+j]
		}
	}
	clear(u)
	for i := range size {
		for k := range size {
			if z := r.inv[i*size+k]; z != 0 {
				for j := range m {
					u[i*m+j] += z * v[k*m+j]
				}
			}
		}
	}
	norm := 0.0
	for i := range r.p {
		for j := range m {
			var g2 complex128
			for k := range size {
				d := -2 * complex(r.weight[k], 0) * r.unit[k] * u[k*m+j]
				if k >= r.n {
					d += complex(r.weight[k], 0) * v[k*m+j]
				}
				g2 += x[i*size+k] * d
			}
			norm += sq(g2)
		}
	}
	return math.Sqrt(norm)
}

func sq(z complex128) float64 { return real(z)*real(z) + imag(z)*imag(z) }

// gain is σ_max(G + c·I) for a SISO shift c or c = 0.
func (r *descriptorResponse) gain(g []complex128, c float64) float64 {
	if r.p == 1 && r.m == 1 {
		return cmplx.Abs(g[0] + complex(c, 0))
	}
	if r.svd == nil {
		return sigmaMaxSmall(g, r.p, r.m)
	}
	sv, err := r.svd.maximumFromFlat(g, 0, r.p, r.m)
	if err != nil {
		return math.Inf(1)
	}
	return sv
}

// sigmaMaxSmall is σ_max of a p×m matrix with p, m ≤ 2 from its Gram matrix.
func sigmaMaxSmall(g []complex128, p, m int) float64 {
	var a, d float64
	var c complex128
	if m == 1 {
		for _, v := range g {
			a += sq(v)
		}
		return math.Sqrt(a)
	}
	for i := range p {
		x, y := g[i*m], g[i*m+1]
		a += sq(x)
		d += sq(y)
		c += cmplx.Conj(x) * y
	}
	disc := math.Sqrt((a-d)*(a-d)/4 + sq(c))
	return math.Sqrt((a+d)/2 + disc)
}

// bound returns an upper bound on σ_max(G + c·I) between s.w and s.w + h
// (h may be negative). With E = M(ω) − M(s.w) = |h|·W·D′, ‖D′‖ ≤ 1, the
// Neumann series gives, with q = 1/(1 − |h|ζ) and qL = 1/(1 − |h|ζL),
// ‖ĈZ(ω)W‖ ≤ αW·q, ‖Z(ω)W‖ ≤ ζ·q, ‖Z(ω)B̃‖ ≤ β·q, ‖WZ(ω)B̃‖ ≤ βW·qL and
// ‖WZ(ω)W‖ ≤ ξ·qL. They bound G″ = Ĉ(2ZM′ZM′Z − ZM″Z)B̃ by m2 and
// G‴ = Ĉ(−6ZM′ZM′ZM′Z + 3ZM″ZM′Z + 3ZM′ZM″Z − ZM‴Z)B̃ by m3 on the interval,
// with |M′| ≤ W, |M″| ≤ W², |M‴| ≤ W³ entrywise. G then lies within
// min(m2·h²/2, ‖G″(s.w)‖·h²/2 + m3·|h|³/6) of the tangent
// G(s.w) + t·G′(s.w), whose gain is convex in t and so largest at an end.
func (r *descriptorResponse) bound(s *descriptorSample, h, c float64) float64 {
	ah := math.Abs(h)
	if !s.regular || !(ah*s.zeta < 1) || !(ah*s.zetaL < 1) {
		return math.Inf(1)
	}
	q, qL := 1/(1-ah*s.zeta), 1/(1-ah*s.zetaL)
	aw, z, b, bw := s.alphaW*q, s.zeta*q, s.beta*q, s.betaW*qL
	m2 := aw * (2*z*b + bw)
	m3 := aw * (6*z*z*b + 3*s.xi*qL*b + 3*z*bw + r.maxWeight*bw)
	rem := min(m2*h*h/2, s.g2*h*h/2+m3*ah*ah*ah/6)
	for i := range r.tmp {
		r.tmp[i] = s.g[i] + complex(h, 0)*s.g1[i]
	}
	return max(r.gain(s.g, c), r.gain(r.tmp, c)) + rem
}

// certify raises (peak, wPeak) to within peakCertTol of sup σ_max(G(jω) + c)
// over [ws[0], ws[len(ws)-1]] for every shift c, with ws sorted. Each
// interval is bounded from the Taylor models at its two ends, each covering
// half of it, and split while the bound exceeds (1 + peakCertTol)·peak, at
// a peak frequency inside it, where the Taylor model is flat, or else at its
// midpoint; larger samples raise the peak. The
// intervals start on ws, skipping points while the half-width h keeps
// h·max(ζ, ζL) <= 1/2, so the Neumann factors stay below 2. More than
// budget samples, or an interval that cannot be split, return
// errUnsupported.
func (r *descriptorResponse) certify(ws, shifts, peaks, wPeaks []float64, budget int, errUnsupported error) error {
	start := r.evals
	sample := func(w float64) (*descriptorSample, error) {
		if r.evals-start >= budget {
			return nil, fmt.Errorf("peak certification exceeds %d samples: %w", budget, errUnsupported)
		}
		s := r.sample(w)
		if !s.regular {
			return nil, fmt.Errorf("singular descriptor at ω=%g: %w", w, errUnsupported)
		}
		for k, c := range shifts {
			if v := r.gain(s.g, c); v > peaks[k] {
				peaks[k], wPeaks[k] = v, w
			}
		}
		return s, nil
	}
	type span struct{ a, b *descriptorSample }
	var stack []span
	a, err := sample(ws[0])
	if err != nil {
		return err
	}
	for i := 0; i < len(ws)-1; {
		reach := a.w + 1/max(a.zeta, a.zetaL)
		j := i + 1
		for j+1 < len(ws) && ws[j+1] <= reach {
			j++
		}
		b, err := sample(ws[j])
		if err != nil {
			return err
		}
		stack = append(stack[:0], span{a, b})
		for len(stack) > 0 {
			sp := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			h := (sp.b.w - sp.a.w) / 2
			if h <= 0 || r.certified(sp.a, sp.b, h, shifts, peaks) {
				continue
			}
			mid := sp.a.w + h
			for _, w := range wPeaks {
				if sp.a.w+h/8 < w && w < sp.b.w-h/8 {
					mid = w
					break
				}
			}
			if !(sp.a.w < mid && mid < sp.b.w) {
				return fmt.Errorf("peak not certified near ω=%g: %w", mid, errUnsupported)
			}
			sm, err := sample(mid)
			if err != nil {
				return err
			}
			stack = append(stack, span{sm, sp.b}, span{sp.a, sm})
		}
		a, i = b, j
	}
	return nil
}

func (r *descriptorResponse) certified(a, b *descriptorSample, h float64, shifts, peaks []float64) bool {
	for k, c := range shifts {
		if !(max(r.bound(a, h, c), r.bound(b, -h, c)) <= (1+peakCertTol)*peaks[k]) {
			return false
		}
	}
	return true
}

// cInvert overwrites the row-major n×n a with its inverse by Gauss–Jordan
// elimination with partial pivoting. It reports false at a zero pivot.
func cInvert(a []complex128, piv []int, n int) bool {
	for k := range n {
		p, best := k, cabs1(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(a[i*n+k]); v > best {
				p, best = i, v
			}
		}
		if best == 0 {
			return false
		}
		piv[k] = p
		rk := a[k*n : (k+1)*n]
		if p != k {
			rp := a[p*n : (p+1)*n]
			for j := range rk {
				rk[j], rp[j] = rp[j], rk[j]
			}
		}
		ik := crecip(rk[k])
		rk[k] = 1
		for j := range rk {
			rk[j] *= ik
		}
		for i := range n {
			if i == k {
				continue
			}
			ri := a[i*n : (i+1)*n]
			f := ri[k]
			if f == 0 {
				continue
			}
			ri[k] = 0
			for j := range ri {
				ri[j] -= f * rk[j]
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		if p := piv[k]; p != k {
			for i := range n {
				a[i*n+k], a[i*n+p] = a[i*n+p], a[i*n+k]
			}
		}
	}
	return true
}
