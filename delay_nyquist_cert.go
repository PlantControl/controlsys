package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// loopCertifier gives certified local models on the jω axis of f = 1+L
// (kind 0) and, where kinds() is 2, of S = 1/f (kind 1), whose phase
// varies as much as that of f and which is finite and nonzero exactly where
// f is: model returns g(ω) and g′(ω) for the function g of the kind, and
// remainder bounds |g(ω+t) − g(ω) − t·g′(ω)| for |t| ≤ |h|, +Inf where it
// has none.
type loopCertifier interface {
	kinds() int
	model(kind int, w float64) (loopModel, bool)
	remainder(kind int, m *loopModel, h float64) float64
}

// loopModel is the Taylor model of one kind at one frequency with the
// scalars its remainder needs.
type loopModel struct {
	f, df complex128
	s     descriptorSample // scalar norms only; g and g1 are nil
	// det(I − G) models: the remainder of G = H22Δ is
	// t²(gT2/2 + g1T) + remainder(s, t), and det(I − G(ω+t))/det(I − G(ω))
	// = det(I + X), X = tY + R, ‖R‖_F <= nu·remG.
	nu, normY float64
	gT2, g1T  float64
	a1        complex128 // tr Y
	// normalized loops: the same for the difference operator I − D22Δ
	nu0, normY0, d0T2 float64
	a0                complex128
}

// spanCertPhase bounds the phase variation of 1+L over an accepted span.
// Any bound below π makes the wrapped phase step between its ends exact;
// the margin covers rounding in the sampled phases.
const spanCertPhase = 0.9 * math.Pi

var sinPi8 = math.Sin(math.Pi / 8)

// stadiumPhase bounds the phase variation of f over [ω, ω+h] from its
// model at ω: f lies within rem of the segment from f(ω) to f(ω) + h·f′(ω).
// When rem is below the distance d of the segment from the origin, f has no
// zero there and its phase varies by at most the angle the segment subtends
// plus 2·asin(rem/d); otherwise it returns +Inf.
func stadiumPhase(f, df complex128, h, rem float64) float64 {
	v := complex(h, 0) * df
	p1 := f + v
	rem += 16 * 0x1p-52 * (1 + cmplx.Abs(f) + cmplx.Abs(v))
	t := 0.0
	if vv := real(v)*real(v) + imag(v)*imag(v); vv > 0 {
		t = min(1, max(0, -(real(f)*real(v)+imag(f)*imag(v))/vv))
	}
	d := cmplx.Abs(f + complex(t, 0)*v)
	if !(rem < d) {
		return math.Inf(1)
	}
	return math.Abs(wrapPi(cmplx.Phase(p1)-cmplx.Phase(f))) + 2*math.Asin(rem/d)
}

// descriptorCertifier models 1+L of a SISO loop from the Taylor bounds of
// its descriptor form L = D + Ĉ·M(ω)⁻¹·B̃, regular off the open-loop poles,
// and S = 1/(1+L) from those of its closed-loop descriptor form, regular off
// the closed-loop roots. Near a multiple imaginary-axis pole of L the norm of
// M⁻¹ grows faster than |L|, so the reach of the model of 1+L shrinks below
// the bisection tolerance, while S stays regular there.
type descriptorCertifier struct {
	r   [2]*descriptorResponse
	buf [2]*descriptorSample
}

func newDescriptorCertifier(sys *System) (*descriptorCertifier, error) {
	c := &descriptorCertifier{}
	for k := range c.r {
		r, err := newDescriptorResponse(sys)
		if err != nil {
			return nil, err
		}
		if k == 1 && r.sensitivity() != nil {
			break
		}
		c.r[k], c.buf[k] = r, r.newSample()
	}
	return c, nil
}

func (c *descriptorCertifier) kinds() int {
	if c.r[1] == nil {
		return 1
	}
	return 2
}

func (c *descriptorCertifier) model(kind int, w float64) (loopModel, bool) {
	s := c.r[kind].sampleInto(c.buf[kind], w)
	if !s.regular || !finite(s.g[0]) || !finite(s.g1[0]) {
		return loopModel{}, false
	}
	m := loopModel{f: s.g[0], df: s.g1[0], s: *s}
	if kind == 0 {
		m.f++
	}
	m.s.g, m.s.g1 = nil, nil
	return m, true
}

func (c *descriptorCertifier) remainder(kind int, m *loopModel, h float64) float64 {
	return c.r[kind].remainder(&m.s, h)
}

func finite(z complex128) bool {
	return !cmplx.IsNaN(z) && !cmplx.IsInf(z)
}

// detCertifier models det(I − H22Δ), or with normalized its ratio to
// det(I − D22Δ), from the descriptor Taylor model of the rational H22. With
// N = I − G, G = H22Δ, det N(ω+t)/det N(ω) = det(I + X) for
// X = −N⁻¹(G(ω+t) − G(ω)) = tY + R, Y = −N⁻¹G′, and
// |det(I+X) − 1 − tr X| <= Σ_{k>=2} C(nd,k)(x/nd)^k for x = ‖X‖_* <=
// √nd·‖X‖_F, since det(I+X) − 1 − tr X = Σ_{k>=2} e_k(λ(X)),
// |e_k(λ)| <= C(nd,k)(Σ|λᵢ|/nd)^k by Maclaurin's inequality and
// Σ|λᵢ| <= ‖X‖_*.
type detCertifier struct {
	e          *delayLFT
	h22        *descriptorResponse // nil without states
	buf        *descriptorSample
	normalized bool
	sqrtNd     float64
	g, g1, n   []complex128
	inv        []complex128
	piv        []int
}

func newDetCertifier(e *delayLFT, normalized bool) (*detCertifier, error) {
	nd := e.nd
	c := &detCertifier{
		e: e, normalized: normalized, sqrtNd: math.Sqrt(float64(nd)),
		g: make([]complex128, nd*nd), g1: make([]complex128, nd*nd),
		n: make([]complex128, nd*nd), inv: make([]complex128, nd*nd), piv: make([]int, nd),
	}
	if e.h22sys != nil {
		r, err := newDescriptorResponse(e.h22sys)
		if err != nil {
			return nil, err
		}
		c.h22, c.buf = r, r.newSample()
	}
	return c, nil
}

func (c *detCertifier) kinds() int { return 1 }

func (c *detCertifier) model(_ int, w float64) (loopModel, bool) {
	nd := c.e.nd
	var m loopModel
	if c.h22 != nil {
		s := c.h22.sampleInto(c.buf, w)
		if !s.regular {
			return m, false
		}
		copy(c.g, s.g)
		copy(c.g1, s.g1)
		m.s = *s
		m.s.g, m.s.g1 = nil, nil
	} else {
		copy(c.g, c.e.d22Complex())
		clear(c.g1)
		m.s.regular = true
	}
	tau := c.e.tau
	for i := range nd {
		for k := range nd {
			h, h1 := c.g[i*nd+k], c.g1[i*nd+k]
			t2 := tau[k] * tau[k]
			m.gT2 += sq(h) * t2 * t2
			m.g1T += sq(h1) * t2
		}
	}
	m.gT2, m.g1T = math.Sqrt(m.gT2), math.Sqrt(m.g1T)
	u, ok := c.relative(w, c.g, c.g1, &m.nu, &m.normY, &m.a1)
	if !ok {
		return m, false
	}
	m.f, m.df = u, u*m.a1
	if c.normalized {
		d22 := c.e.d22Complex()
		copy(c.g, d22)
		clear(c.g1)
		for i := range nd {
			for k := range nd {
				t2 := tau[k] * tau[k]
				m.d0T2 += sq(d22[i*nd+k]) * t2 * t2
			}
		}
		m.d0T2 = math.Sqrt(m.d0T2)
		u0, ok := c.relative(w, c.g, c.g1, &m.nu0, &m.normY0, &m.a0)
		if !ok {
			return m, false
		}
		m.f = u / u0
		m.df = m.f * (m.a1 - m.a0)
	}
	if !finite(m.f) || !finite(m.df) {
		return m, false
	}
	return m, true
}

// relative returns det N for N = I − HΔ at ω, with ν = ‖N⁻¹‖_F, ‖Y‖_F and
// tr Y for Y = −N⁻¹G′, G′ = (H′ − jHT)Δ. It overwrites h and h1.
func (c *detCertifier) relative(w float64, h, h1 []complex128, nu, normY *float64, trY *complex128) (complex128, bool) {
	nd := c.e.nd
	for k, t := range c.e.tau {
		d := cmplx.Exp(complex(0, -w*t))
		for i := range nd {
			h1[i*nd+k] = (h1[i*nd+k] - complex(0, t)*h[i*nd+k]) * d
			h[i*nd+k] *= d
		}
	}
	for i := range h {
		c.n[i] = -h[i]
	}
	for i := range nd {
		c.n[i*nd+i]++
	}
	copy(c.inv, c.n)
	if !cLUFactor(c.inv, c.piv, nd) {
		return 0, false
	}
	det := complex(1, 0)
	for k := range nd {
		det *= c.inv[k*nd+k]
		if c.piv[k] != k {
			det = -det
		}
	}
	copy(c.inv, c.n)
	if !cInvert(c.inv, c.piv, nd) {
		return 0, false
	}
	*nu, *normY, *trY = 0, 0, 0
	for i := range nd {
		for j := range nd {
			*nu += sq(c.inv[i*nd+j])
			var y complex128
			for k := range nd {
				y -= c.inv[i*nd+k] * h1[k*nd+j]
			}
			*normY += sq(y)
			if i == j {
				*trY += y
			}
		}
	}
	*nu, *normY = math.Sqrt(*nu), math.Sqrt(*normY)
	return det, true
}

func (c *detCertifier) remainder(_ int, m *loopModel, h float64) float64 {
	ah := math.Abs(h)
	remH := 0.0
	if c.h22 != nil {
		remH = c.h22.remainder(&m.s, h)
	}
	remG := ah*ah*(m.gT2/2+m.g1T) + remH
	r1 := c.relRemainder(ah, m.nu, m.normY, remG)
	if !c.normalized {
		return cmplx.Abs(m.f) * r1
	}
	r0 := c.relRemainder(ah, m.nu0, m.normY0, ah*ah*m.d0T2/2)
	da := cmplx.Abs(m.a1 - m.a0)
	den := 1 - ah*cmplx.Abs(m.a0) - r0
	if !(den > 0) {
		return math.Inf(1)
	}
	return cmplx.Abs(m.f) * (r1 + r0*(1+ah*da) + ah*ah*da*cmplx.Abs(m.a0)) / den
}

// relRemainder bounds |det(I+X) − 1 − t·tr Y| for ‖R‖_F <= ν·remG.
func (c *detCertifier) relRemainder(ah, nu, normY, remG float64) float64 {
	r := nu * remG
	x := c.sqrtNd * (ah*normY + r)
	return c.sqrtNd*r + detHigherTerms(x, c.e.nd)
}

// detHigherTerms returns Σ_{k=2}^{nd} C(nd,k)(x/nd)^k.
func detHigherTerms(x float64, nd int) float64 {
	y := x / float64(nd)
	sum, term := 0.0, float64(nd)*y
	for k := 2; k <= nd; k++ {
		term *= y * float64(nd-k+1) / float64(k)
		sum += term
	}
	return sum
}

// pidLoopCertifier returns the certifier of the loop c·plant, with an ideal
// derivative (Kd ≠ 0, Tf = 0) realized as Kd·s·P = Kd·(CB + CA(sI−A)⁻¹B)
// on a second copy of the delay-free strictly proper plant.
func pidLoopCertifier(plant *System, c *PID2) (*descriptorCertifier, error) {
	loop, err := pidLoopSystem(plant, c)
	if err != nil {
		return nil, err
	}
	return newDescriptorCertifier(loop)
}

func pidLoopSystem(plant *System, c *PID2) (*System, error) {
	if c.Kd != 0 && c.Tf == 0 {
		return pidIdealDerivativeLoop(plant, c)
	}
	var a, b, cc []float64
	d := c.Kp
	if c.Ki != 0 {
		a, b, cc = append(a, 0), append(b, 1), append(cc, c.Ki)
	}
	if c.Kd != 0 {
		a, b, cc = append(a, -1/c.Tf), append(b, 1), append(cc, -c.Kd/(c.Tf*c.Tf))
		d += c.Kd / c.Tf
	}
	n := len(a)
	am := make([]float64, n*n)
	for i := range n {
		am[i*n+i] = a[i]
	}
	ctrl, err := NewFromSlices(n, 1, 1, am, b, cc, []float64{d}, 0)
	if err != nil {
		return nil, err
	}
	return Series(ctrl, plant)
}

func pidIdealDerivativeLoop(plant *System, c *PID2) (*System, error) {
	if plant.HasInternalDelay() {
		return nil, fmt.Errorf("ideal derivative on internal delays: %w", errDelayLoopUnsupported)
	}
	np, _, _ := plant.Dims()
	nz := 0
	if c.Ki != 0 {
		nz = 1
	}
	n := 2*np + nz
	off := np + nz
	a := mat.NewDense(n, n, nil)
	b := mat.NewDense(n, 1, nil)
	cm := mat.NewDense(1, n, nil)
	d := 0.0
	if nz == 1 {
		b.Set(np, 0, 1)
	}
	for i := range np {
		var ca float64
		for k := range np {
			ca += plant.C.At(0, k) * plant.A.At(k, i)
		}
		d += plant.C.At(0, i) * plant.B.At(i, 0)
		for j := range np {
			a.Set(i, j, plant.A.At(i, j))
			a.Set(off+i, off+j, plant.A.At(i, j))
		}
		if nz == 1 {
			a.Set(i, np, c.Ki*plant.B.At(i, 0))
		}
		b.Set(i, 0, c.Kp*plant.B.At(i, 0))
		b.Set(off+i, 0, plant.B.At(i, 0))
		cm.Set(0, i, plant.C.At(0, i))
		cm.Set(0, off+i, c.Kd*ca)
	}
	loop, err := New(a, b, cm, mat.NewDense(1, 1, []float64{c.Kd * d}), 0)
	if err != nil {
		return nil, err
	}
	if err := loop.SetInputDelay([]float64{sisoLoopDelay(plant)}); err != nil {
		return nil, err
	}
	return loop, nil
}
