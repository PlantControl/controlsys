package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// loopCertifier gives certified local models on the jω axis of f = 1+L
// (kind 0) and, where kinds() is 2, of a second function regular where f
// has a pole (kind 1): model returns the model of the kind at ω, and phase
// bounds the phase variation of f over [ω, ω+h] from it, +Inf where it
// cannot.
type loopCertifier interface {
	kinds() int
	model(kind int, w float64) (loopModel, bool)
	phase(kind int, m *loopModel, h float64) float64
}

// loopModel is the Taylor model of one kind at one frequency with the
// scalars its remainder needs.
type loopModel struct {
	f, df complex128
	// f and df carry rounding rho·(unit + |f| + |h·df|): unit 1 where f is
	// formed as 1 + L, 0 for a determinant with relative error rho.
	rho, unit float64
	s         descriptorSample // scalar norms only; g and g1 are nil
	// det(I − G) models: the remainder of G = H22Δ is
	// t²(gT2/2 + g1T) + remainder(s, t), and det(I − G(ω+t))/det(I − G(ω))
	// = det(I + X), X = tY + R, ‖R‖_F <= nu·remG.
	nu, normY float64
	gT2, g1T  float64
	a1        complex128 // tr Y
	// normalized loops: the same for the difference operator I − D22Δ
	nu0, normY0, d0T2 float64
	a0                complex128
	// det M models (detCertifier kind 1): ‖ZW²‖_F, and the model of
	// det(I − D22Δ) with its rounding on normalized loops
	zW2, rho0 float64
	u0        complex128
}

// spanCertPhase bounds the phase variation of 1+L over an accepted span.
// Any bound below π makes the wrapped phase step between its ends exact;
// the margin covers rounding in the sampled phases.
const spanCertPhase = 0.9 * math.Pi

var sinPi8 = math.Sin(math.Pi / 8)

// modelRho is the rounding of a model formed as 1 + L, relative to 1+|f|.
const modelRho = 16 * 0x1p-52

// stadiumPhase bounds the phase variation of f over [ω, ω+h] from its
// model at ω: f lies within rem of the segment from f(ω) to f(ω) + h·f′(ω),
// widened by the model's rounding rho·(unit + |f| + |h·f′|).
// When rem is below the distance d of the segment from the origin, f has no
// zero there and its phase varies by at most the angle the segment subtends
// plus 2·asin(rem/d); otherwise it returns +Inf.
func stadiumPhase(f, df complex128, h, rem, rho, unit float64) float64 {
	v := complex(h, 0) * df
	p1 := f + v
	rem += rho * (unit + cmplx.Abs(f) + cmplx.Abs(v))
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
	m := loopModel{f: s.g[0], df: s.g1[0], s: *s, rho: modelRho, unit: 1}
	m.s.g, m.s.g1 = nil, nil
	if kind == 0 {
		m.f++
		return m, true
	}
	if cmplx.Abs(m.f) < sensitivityFloor {
		c.sensitivityFromLoop(&m, w)
	}
	return m, true
}

// sensitivityFloor is the |S| below which the absolute rounding of the S
// descriptor exceeds 1e-6 relative.
const sensitivityFloor = 1e6 * modelRho

// sensitivityFromLoop replaces the S value of m by 1/(1+L) where |1+L| > 1.
// The closed-loop descriptor forms S as β minus a term near β, so its
// absolute rounding ~eps swamps |S| ~ ω^k/K near a k-fold pole of L at the
// origin, while 1/(1+L) keeps the relative accuracy of the sample of L. Both
// are the same function, so the remainder of the S descriptor still holds.
func (c *descriptorCertifier) sensitivityFromLoop(m *loopModel, w float64) {
	s := c.r[0].sampleInto(c.buf[0], w)
	if !s.regular || !finite(s.g[0]) || !finite(s.g1[0]) {
		return
	}
	f := 1 + s.g[0]
	if !(cmplx.Abs(f) > 1) {
		return
	}
	sv, dsv := 1/f, -s.g1[0]/(f*f)
	if !finite(sv) || !finite(dsv) {
		return
	}
	m.f, m.df, m.rho, m.unit = sv, dsv, 2*modelRho, 0
}

func (c *descriptorCertifier) remainder(kind int, m *loopModel, h float64) float64 {
	return c.r[kind].remainder(&m.s, h)
}

func (c *descriptorCertifier) phase(kind int, m *loopModel, h float64) float64 {
	return stadiumPhase(m.f, m.df, h, c.remainder(kind, m, h), m.rho, m.unit)
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
//
// Near a multiple imaginary-axis pole of H22 ‖(jωI − A)⁻¹‖ grows faster than
// |H22|, so the reach of that model falls below the bisection tolerance.
// Kind 1 then models det M(ω) of the delay descriptor (see
// descriptorResponse) M = M₀ + diag(jωIₙ, e^{jωτ}), which is entire with
// det M = det(jωI − A)·e^{jωΣτ}·det(I − H22Δ) and regular off the
// closed-loop roots: the phase of f varies by at most that of det M, plus
// |h|Στ, plus the angle each eigenvalue of A off the axis subtends from the
// span; an eigenvalue on the axis adds nothing, as the span lies on the axis
// off it.
type detCertifier struct {
	e          *delayLFT
	h22        *descriptorResponse // nil without states
	buf        *descriptorSample
	normalized bool
	g, g1, n   []complex128
	inv        []complex128
	piv        []int

	desc    *descriptorResponse // M₀ of kind 1, see chiDescriptor
	descErr bool
	mlu     []complex128
	mpiv    []int
	offAxis []complex128
	sumTau  float64
}

func newDetCertifier(e *delayLFT, normalized bool) (*detCertifier, error) {
	nd := e.nd
	c := &detCertifier{
		e: e, normalized: normalized,
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

// kinds is 2 with states; the descriptor of kind 1 is built on first use.
func (c *detCertifier) kinds() int {
	if c.h22 == nil {
		return 1
	}
	return 2
}

func (c *detCertifier) chiDescriptor() *descriptorResponse {
	if c.desc != nil || c.descErr {
		return c.desc
	}
	e := c.e
	sys, err := SetDelayModel(e.h22sys, e.tau)
	if err == nil {
		c.desc, err = newDescriptorResponse(sys)
	}
	if err != nil {
		c.descErr = true
		return nil
	}
	size := c.desc.size
	c.mlu, c.mpiv = make([]complex128, size*size), make([]int, size)
	for _, p := range e.poles {
		if classifyPoleStability(p, true, poleStabilityTolerance(p)) != poleClassBoundary {
			c.offAxis = append(c.offAxis, p)
		}
	}
	for _, t := range e.tau {
		c.sumTau += t
	}
	return c.desc
}

func (c *detCertifier) model(kind int, w float64) (loopModel, bool) {
	if kind == 1 {
		return c.chiModel(w)
	}
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
	gF := cFrobenius(c.g)
	u, ok := c.relative(w, c.g, c.g1, &m.nu, &m.normY, &m.a1)
	if !ok {
		return m, false
	}
	m.f, m.df = u, u*m.a1
	m.rho = c.detRho(m.nu, gF)
	if c.normalized {
		if !c.differenceModel(w, &m) {
			return m, false
		}
		m.f = u / m.u0
		m.df = m.f * (m.a1 - m.a0)
		m.rho += m.rho0
	}
	if !finite(m.f) || !finite(m.df) {
		return m, false
	}
	return m, true
}

// differenceModel sets the model of u0 = det(I − D22Δ) in m.
func (c *detCertifier) differenceModel(w float64, m *loopModel) bool {
	nd, tau := c.e.nd, c.e.tau
	d22 := c.e.d22Complex()
	copy(c.g, d22)
	clear(c.g1)
	m.d0T2 = 0
	for i := range nd {
		for k := range nd {
			t2 := tau[k] * tau[k]
			m.d0T2 += sq(d22[i*nd+k]) * t2 * t2
		}
	}
	m.d0T2 = math.Sqrt(m.d0T2)
	g0F := cFrobenius(c.g)
	u0, ok := c.relative(w, c.g, c.g1, &m.nu0, &m.normY0, &m.a0)
	if !ok || !finite(u0) || u0 == 0 {
		return false
	}
	m.u0, m.rho0 = u0, c.detRho(m.nu0, g0F)
	return true
}

// chiModel returns the model of det M at ω: with Z = M⁻¹ and
// M′ = j·diag(1, τe^{jωτ}), det M(ω+t)/det M(ω) = det(I + X),
// X = tZM′ + ZR, where R = M(ω+t) − M(ω) − tM′ is diagonal with
// |R_kk| <= t²τ_k²/2, so ‖ZR‖_F <= ‖ZW²‖_F·t²/2.
func (c *detCertifier) chiModel(w float64) (loopModel, bool) {
	var m loopModel
	r := c.chiDescriptor()
	if r == nil {
		return m, false
	}
	n, size := r.n, r.size
	lu := c.mlu
	copy(lu, r.m0)
	for i := range size {
		if i < n {
			r.unit[i] = complex(0, 1)
			lu[i*size+i] += complex(0, w)
		} else {
			e := cmplx.Exp(complex(0, w*r.tau[i-n]))
			r.unit[i] = complex(0, r.tau[i-n]) * e
			lu[i*size+i] += e
		}
	}
	mF := cFrobenius(lu)
	copy(r.inv, lu)
	if !cLUFactor(lu, c.mpiv, size) {
		return m, false
	}
	det := complex(1, 0)
	for k := range size {
		det *= lu[k*size+k]
		if c.mpiv[k] != k {
			det = -det
		}
	}
	if !cInvert(r.inv, c.mpiv, size) {
		return m, false
	}
	var nu float64
	for i := range size {
		for j := range size {
			z := r.inv[i*size+j]
			az := sq(z)
			nu += az
			wj := r.weight[j] * r.weight[j]
			m.normY += az * wj
			if j >= n {
				m.zW2 += az * wj * wj
			}
			if i == j {
				m.a1 += z * r.unit[j]
			}
		}
	}
	nu, m.normY, m.zW2 = math.Sqrt(nu), math.Sqrt(m.normY), math.Sqrt(m.zW2)
	m.f, m.df = det, det*m.a1
	m.rho = 16 * float64(size) * 0x1p-52 * (1 + nu*mF)
	if c.normalized && !c.differenceModel(w, &m) {
		return m, false
	}
	if !finite(m.f) || !finite(m.df) || m.f == 0 {
		return m, false
	}
	m.s.w, m.s.regular = w, true
	return m, true
}

func (c *detCertifier) phase(kind int, m *loopModel, h float64) float64 {
	if kind == 0 {
		return stadiumPhase(m.f, m.df, h, c.remainder(0, m, h), m.rho, m.unit)
	}
	ah := math.Abs(h)
	rel := relativeDetRemainder(ah, m.normY, m.zW2*ah*ah/2, c.desc.size) + m.rho*ah*m.normY
	p := stadiumPhase(m.f, m.df, h, cmplx.Abs(m.f)*rel, m.rho, 0) + ah*c.sumTau
	if c.normalized {
		rel0 := c.relRemainder(ah, m.nu0, m.normY0, ah*ah*m.d0T2/2) + m.rho0*ah*m.normY0
		p += stadiumPhase(m.u0, m.u0*m.a0, h, cmplx.Abs(m.u0)*rel0, m.rho0, 0)
	}
	lo, hi := m.s.w, m.s.w+h
	if hi < lo {
		lo, hi = hi, lo
	}
	for _, l := range c.offAxis {
		d := cmplx.Abs(l - complex(0, min(hi, max(lo, imag(l)))))
		p += min(math.Pi, ah/d)
		if !(p <= math.Pi) {
			return math.Inf(1)
		}
	}
	return p
}

// relativeDetRemainder bounds |det(I+X) − 1 − t·tr Y| for the dim×dim
// X = tY + R, |t| <= ah, ‖R‖_F <= r.
func relativeDetRemainder(ah, normY, r float64, dim int) float64 {
	s := math.Sqrt(float64(dim))
	return s*r + detHigherTerms(s*(ah*normY+r), dim)
}

// detRho bounds the relative rounding of det N and of tr Y for
// N = I − HΔ, ‖N⁻¹‖_F = nu, ‖H‖_F = hF: forming N and its LU factors is
// backward stable with ‖E‖_F <~ nd·eps·(1+hF), and det(N+E)/det N − 1 ≈
// tr(N⁻¹E) <= nu·‖E‖_F. A near-singular N, not a small |det N|, makes the
// determinant inaccurate.
func (c *detCertifier) detRho(nu, hF float64) float64 {
	return 16 * float64(c.e.nd) * 0x1p-52 * (1 + nu*(1+hF))
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
	r1 := c.relRemainder(ah, m.nu, m.normY, remG) + m.rho*ah*(m.normY+m.normY0)
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
	return relativeDetRemainder(ah, normY, nu*remG, c.e.nd)
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

// singleChannelCertifier returns the certifier of 1 + L = 1 − H22·e^{−sτ}
// for a single delay channel, whose S model the determinant lacks: near a
// multiple pole of H22 on the axis the reach of every model of 1+L falls
// below the bisection tolerance.
func singleChannelCertifier(e *delayLFT) (*descriptorCertifier, error) {
	var c2, d22 mat.Dense
	c2.Scale(-1, e.h22sys.C)
	d22.Scale(-1, e.h22sys.D)
	loop := &System{A: e.h22sys.A, B: e.h22sys.B, C: &c2, D: &d22}
	if err := loop.SetInputDelay([]float64{e.tau[0]}); err != nil {
		return nil, err
	}
	return newDescriptorCertifier(loop)
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

// pidIdealDerivativeLoop realizes (Kp + Ki/s + Kd·s)·P on one copy of the
// plant, x′ = Ax + Bu, w′ = Cx: L = (Kp·C + Kd·CA)x + Ki·w + Kd·CB·u. A
// second plant copy for the derivative would leave hidden modes at the plant
// poles, which make the descriptor matrices of 1+L and S singular there.
func pidIdealDerivativeLoop(plant *System, c *PID2) (*System, error) {
	if plant.HasInternalDelay() {
		return nil, fmt.Errorf("ideal derivative on internal delays: %w", errDelayLoopUnsupported)
	}
	np, _, _ := plant.Dims()
	nz := 0
	if c.Ki != 0 {
		nz = 1
	}
	n := np + nz
	a := mat.NewDense(n, n, nil)
	b := mat.NewDense(n, 1, nil)
	cm := mat.NewDense(1, n, nil)
	d := 0.0
	for i := range np {
		var ca float64
		for k := range np {
			ca += plant.C.At(0, k) * plant.A.At(k, i)
		}
		d += plant.C.At(0, i) * plant.B.At(i, 0)
		for j := range np {
			a.Set(i, j, plant.A.At(i, j))
		}
		if nz == 1 {
			a.Set(np, i, plant.C.At(0, i))
		}
		b.Set(i, 0, plant.B.At(i, 0))
		cm.Set(0, i, c.Kp*plant.C.At(0, i)+c.Kd*ca)
	}
	if nz == 1 {
		cm.Set(0, np, c.Ki)
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
