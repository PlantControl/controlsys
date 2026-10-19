package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzLoopsens: for a MIMO plant P (fzSpec, delay-free) and a static
// controller K (m×p, gain 10^clamp(v[15],−3,3), sign of v[14], texture
// phase v[14]), the four returned sensitivities match (I+PK)⁻¹,
// PK(I+PK)⁻¹, (I+KP)⁻¹ and KP(I+KP)⁻¹ formed from the exact response of P,
// each returned model evaluated exactly, at three frequencies. An ill-posed
// loop (I + D·K singular) must return ErrAlgebraicLoop.
func FuzzLoopsens(f *testing.F) {
	for _, s := range fzLsSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		P := s.System(t, fzSysOpts{})
		m, p := s.M, s.P
		kg := math.Pow(10, fzClamp(s.Vals[15], -3, 3))
		if s.Vals[14] < 0 {
			kg = -kg
		}
		ph := fzClamp(s.Vals[14], -1e3, 1e3)
		K := make([]float64, m*p)
		for i := range m {
			for j := range p {
				K[i*p+j] = kg * math.Cos(ph+1.7*float64(i+1)+0.9*float64(j+1))
			}
		}
		C, err := NewGain(mat.NewDense(m, p, K), s.Dt)
		if err != nil {
			t.Fatalf("NewGain: %v\n%s", err, s)
		}
		res, err := Loopsens(P, C)
		fzContract(t, s, "Loopsens", res != nil, err)
		idk := fzLsIPlus(fzLsMul(fzComplexD(s), fzLsC(K), p, m, p), p)
		condDK := fzLsCond(idk, p)
		if err != nil {
			if errors.Is(err, ErrAlgebraicLoop) && condDK > 1e8 {
				fzSkip(t, "Loopsens: I + D·K numerically singular")
			}
			t.Fatalf("Loopsens: undocumented error %v (cond(I+DK)=%g)\n%s", err, condDK, s)
		}
		if math.IsInf(condDK, 1) {
			t.Fatalf("Loopsens succeeded with I + D·K exactly singular\n%s", s)
		}
		for name, g := range map[string]*System{"So": res.So, "To": res.To, "Si": res.Si, "Ti": res.Ti} {
			_, gm, gp := g.Dims()
			want := p
			if name[1] == 'i' {
				want = m
			}
			if gm != want || gp != want || g.Dt != s.Dt {
				t.Fatalf("Loopsens %s is %d×%d Dt=%v, want %d×%d Dt=%v\n%s", name, gp, gm, g.Dt, want, want, s.Dt, s)
			}
		}
		for _, w := range fzLsFreqs(s) {
			x := fzFreqPoint(s, w)
			pg, ok := fzExactResponse(s.N, m, p, s.A, s.B, s.C, s.D, x)
			if !ok {
				continue
			}
			kc := fzLsC(K)
			qo := fzLsIPlus(fzLsMul(pg, kc, p, m, p), p)
			qi := fzLsIPlus(fzLsMul(kc, pg, m, p, m), m)
			co, ci := fzLsCond(qo, p), fzLsCond(qi, m)
			if co > 1e6 || ci > 1e6 {
				continue
			}
			so, si := fzLsInv(qo, p), fzLsInv(qi, m)
			wants := map[string][]complex128{
				"So": so, "To": fzLsIMinus(so, p),
				"Si": si, "Ti": fzLsIMinus(si, m),
			}
			for name, g := range map[string]*System{"So": res.So, "To": res.To, "Si": res.Si, "Ti": res.Ti} {
				got, tol, ok := fzLsEval(g, x)
				if !ok {
					continue
				}
				tol *= math.Max(co, ci)
				if tol > 1e-3 {
					continue
				}
				want := wants[name]
				scale := 1.0
				for _, v := range want {
					scale = math.Max(scale, cmplx.Abs(v))
				}
				for k := range want {
					if cmplx.Abs(got[k]-want[k]) > tol*scale {
						t.Fatalf("Loopsens %s at ω=%v: got %v, want %v (tol %g)\n%s", name, w, got, want, tol*scale, s)
					}
				}
			}
		}
	})
}

// fzLsEval evaluates the delay-free state-space g exactly at x and returns
// a relative tolerance for the rounding committed in forming its
// realization: 1e-10·(1 + ‖A‖ + ‖B‖‖C‖)² / σmin(xI − A)².
func fzLsEval(g *System, x complex128) ([]complex128, float64, bool) {
	n, m, p := g.Dims()
	if g.IsDescriptor() || g.HasDelay() {
		return nil, 0, false
	}
	raw := func(d *mat.Dense, r, c int) []float64 {
		out := make([]float64, r*c)
		for i := range r {
			for j := range c {
				out[i*c+j] = d.At(i, j)
			}
		}
		return out
	}
	var a, b, c []float64
	if n > 0 {
		a, b, c = raw(g.A, n, n), raw(g.B, n, m), raw(g.C, p, n)
	}
	d := raw(g.D, p, m)
	out, ok := fzExactResponse(n, m, p, a, b, c, d, x)
	if !ok {
		return nil, 0, false
	}
	if n == 0 {
		return out, 1e-10, true
	}
	xa := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			xa[i*n+j] = complex(-a[i*n+j], 0)
		}
		xa[i*n+i] += x
	}
	sv := fzSingularValues(xa, n, n)
	if len(sv) == 0 || sv[n-1] == 0 {
		return nil, 0, false
	}
	scale := 1 + math.Sqrt(fzMgSumSq(a)) + math.Sqrt(fzMgSumSq(b)*fzMgSumSq(c))
	return out, 1e-10 * scale * scale / (sv[n-1] * sv[n-1]), true
}

// fzLsFreqs picks three test frequencies around the pole magnitudes, below
// Nyquist when discrete.
func fzLsFreqs(s *fzSpec) []float64 {
	if s.Discrete {
		ny := math.Pi / s.Dt
		return []float64{0.13 * ny, 0.41 * ny, 0.77 * ny}
	}
	w := 1.0
	for _, e := range s.S {
		if a := cmplx.Abs(e); a > 0 {
			w = a
			break
		}
	}
	return []float64{0.31 * w, 1.7 * w, 9.1 * w}
}

func fzLsC(x []float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex(v, 0)
	}
	return out
}

// fzLsMul returns the r×c product of the r×k a and k×c b.
func fzLsMul(a, b []complex128, r, k, c int) []complex128 {
	out := make([]complex128, r*c)
	for i := range r {
		for j := range c {
			for l := range k {
				out[i*c+j] += a[i*k+l] * b[l*c+j]
			}
		}
	}
	return out
}

func fzLsIPlus(a []complex128, n int) []complex128 {
	out := append([]complex128(nil), a...)
	for i := range n {
		out[i*n+i]++
	}
	return out
}

func fzLsIMinus(a []complex128, n int) []complex128 {
	out := make([]complex128, n*n)
	for i := range out {
		out[i] = -a[i]
	}
	for i := range n {
		out[i*n+i]++
	}
	return out
}

func fzLsCond(a []complex128, n int) float64 {
	sv := fzSingularValues(a, n, n)
	if len(sv) == 0 || sv[n-1] == 0 {
		return math.Inf(1)
	}
	return sv[0] / sv[n-1]
}

// fzLsInv inverts the n×n complex a by Gauss–Jordan with partial pivoting.
func fzLsInv(a []complex128, n int) []complex128 {
	m := append([]complex128(nil), a...)
	inv := make([]complex128, n*n)
	for i := range n {
		inv[i*n+i] = 1
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(m[i*n+k]) > cmplx.Abs(m[piv*n+k]) {
				piv = i
			}
		}
		for j := range n {
			m[k*n+j], m[piv*n+j] = m[piv*n+j], m[k*n+j]
			inv[k*n+j], inv[piv*n+j] = inv[piv*n+j], inv[k*n+j]
		}
		pk := m[k*n+k]
		for j := range n {
			m[k*n+j] /= pk
			inv[k*n+j] /= pk
		}
		for i := range n {
			if i == k {
				continue
			}
			f := m[i*n+k]
			for j := range n {
				m[i*n+j] -= f * m[k*n+j]
				inv[i*n+j] -= f * inv[k*n+j]
			}
		}
	}
	return inv
}

func fzLsSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzHinfNormSeeds(),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 3
			s.Mag, s.Damp = [3]float64{0.2, -0.4}, [3]float64{1.5, 3}
			s.Gain, s.DScal, s.IO, s.Tau, s.Phase = 6.5, 0.7, 0.4, 0.5, 0.2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P = 3, 3, 2
			s.Mag, s.Damp = [3]float64{0.8, 0.1}, [3]float64{2, 0}
			s.Gain, s.DScal, s.IO, s.Tau = 7, -0.5, 1.1, -1
		}),
		// SISO with D·K = −1: ill-posed loop.
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.DScal, s.Phase, s.IO, s.Tau = 2, 0, 1.5, 1, -3.3, math.Pi-2.6, 0
		}),
	)
}
