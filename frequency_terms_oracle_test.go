package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"
)

// frTermsExact is the within-terms oracle of FreqResponse at one point of
// an explicit delay-free model, per entry (i, j), row-major p×m: the
// 320-bit response h = D + CMB, M = (λI − A)⁻¹ with λ = −p/q of pt exactly
// as FreqResponse forms it, the term bound δᵢⱼ = u(|cᵢ||M||A||M||bⱼ| +
// |cᵢ||M||bⱼ| + |Dᵢⱼ|) and the shared determinacy η = u·ρ(|M||A|), u =
// n·eps. ρ is a Collatz–Wielandt upper bound on |M||A| + τ·11ᵀ, so η is
// never understated. sing reports λI − A exactly singular in 320 bits.
func frTermsExact(sys *System, pt frequencyPoint) (h []bandwidthDipC, delta []float64, eta float64, sing bool) {
	n, m, p := sys.Dims()
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthDipPrec) }
	cx := func(v complex128) bandwidthDipC {
		return bandwidthDipC{bandwidthDipF(real(v)), bandwidthDipF(imag(v))}
	}
	lam := bandwidthDipDiv(cx(-pt.p), cx(pt.q))
	abs2 := func(a bandwidthDipC) *big.Float {
		v := nf().Mul(a.re, a.re)
		return v.Add(v, nf().Mul(a.im, a.im))
	}
	w := 2 * n
	k := make([]bandwidthDipC, n*w)
	for i := range n {
		for j := range n {
			k[i*w+j] = bandwidthDipC{bandwidthDipF(-sys.A.At(i, j)), bandwidthDipF(0)}
			k[i*w+n+j] = bandwidthDipC{bandwidthDipF(0), bandwidthDipF(0)}
		}
		k[i*w+i] = bandwidthDipC{nf().Add(k[i*w+i].re, lam.re), nf().Set(lam.im)}
		k[i*w+n+i] = bandwidthDipC{bandwidthDipF(1), bandwidthDipF(0)}
	}
	for c := range n {
		pv := c
		for i := c + 1; i < n; i++ {
			if abs2(k[i*w+c]).Cmp(abs2(k[pv*w+c])) > 0 {
				pv = i
			}
		}
		if abs2(k[pv*w+c]).Sign() == 0 {
			return nil, nil, 0, true
		}
		for j := range w {
			k[c*w+j], k[pv*w+j] = k[pv*w+j], k[c*w+j]
		}
		piv := k[c*w+c]
		for j := range w {
			k[c*w+j] = bandwidthDipDiv(k[c*w+j], piv)
		}
		for i := range n {
			if i == c || abs2(k[i*w+c]).Sign() == 0 {
				continue
			}
			f := k[i*w+c]
			for j := range w {
				k[i*w+j] = bandwidthDipSub(k[i*w+j], bandwidthDipMul(f, k[c*w+j]))
			}
		}
	}
	mm := func(i, j int) bandwidthDipC { return k[i*w+n+j] }
	am := make([]float64, n*n)
	for i := range n {
		for j := range n {
			v, _ := nf().Sqrt(abs2(mm(i, j))).Float64()
			am[i*n+j] = v
		}
	}
	mb := make([]bandwidthDipC, n*m)
	for i := range n {
		for jj := range m {
			acc := bandwidthDipC{bandwidthDipF(0), bandwidthDipF(0)}
			for l := range n {
				t := bandwidthDipMul(mm(i, l), bandwidthDipC{bandwidthDipF(sys.B.At(l, jj)), bandwidthDipF(0)})
				acc.re.Add(acc.re, t.re)
				acc.im.Add(acc.im, t.im)
			}
			mb[i*m+jj] = acc
		}
	}
	h = make([]bandwidthDipC, p*m)
	for ii := range p {
		for jj := range m {
			acc := bandwidthDipC{bandwidthDipF(sys.D.At(ii, jj)), bandwidthDipF(0)}
			for l := range n {
				t := bandwidthDipMul(bandwidthDipC{bandwidthDipF(sys.C.At(ii, l)), bandwidthDipF(0)}, mb[l*m+jj])
				acc.re.Add(acc.re, t.re)
				acc.im.Add(acc.im, t.im)
			}
			h[ii*m+jj] = acc
		}
	}
	u := float64(n) * eps()
	absMb := make([]float64, n*m)
	amb := make([]float64, n*m)
	for i := range n {
		for jj := range m {
			var s float64
			for l := range n {
				s += am[i*n+l] * math.Abs(sys.B.At(l, jj))
			}
			absMb[i*m+jj] = s
		}
	}
	for i := range n {
		for jj := range m {
			var s float64
			for l := range n {
				s += math.Abs(sys.A.At(i, l)) * absMb[l*m+jj]
			}
			amb[i*m+jj] = s
		}
	}
	delta = make([]float64, p*m)
	for ii := range p {
		cm := make([]float64, n)
		for l := range n {
			for j := range n {
				cm[j] += math.Abs(sys.C.At(ii, l)) * am[l*n+j]
			}
		}
		for jj := range m {
			var t1, t2 float64
			for l := range n {
				t1 += cm[l] * amb[l*m+jj]
				t2 += cm[l] * math.Abs(sys.B.At(l, jj))
			}
			delta[ii*m+jj] = u * (t1 + t2 + math.Abs(sys.D.At(ii, jj)))
		}
	}
	pm := make([]float64, n*n)
	pmax := 0.0
	for i := range n {
		for j := range n {
			var s float64
			for l := range n {
				s += am[i*n+l] * math.Abs(sys.A.At(l, j))
			}
			pm[i*n+j] = s
			pmax = max(pmax, s)
		}
	}
	if pmax == 0 {
		return h, delta, 0, false
	}
	tau := 1e-30 * pmax
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = 1
	}
	rho := math.Inf(1)
	for range 200 {
		ymax, r := 0.0, 0.0
		var xs float64
		for j := range n {
			xs += x[j]
		}
		for i := range n {
			var s float64
			for j := range n {
				s += pm[i*n+j] * x[j]
			}
			y[i] = s + tau*xs
			r = max(r, y[i]/x[i])
			ymax = max(ymax, y[i])
		}
		rho = min(rho, r)
		for i := range n {
			x[i] = y[i] / ymax
		}
	}
	return h, delta, u * rho, false
}

// frTermsBound is frTermsExact rounded: h, and per entry the bound
// δ/(1−η), or −1 where η ≥ 1 leaves the point undetermined.
func frTermsBound(sys *System, pt frequencyPoint) (h []complex128, tb []float64, sing bool) {
	hb, delta, eta, sing := frTermsExact(sys, pt)
	if sing {
		return nil, nil, true
	}
	h = make([]complex128, len(hb))
	tb = make([]float64, len(hb))
	for k, v := range hb {
		re, _ := v.re.Float64()
		im, _ := v.im.Float64()
		h[k] = complex(re, im)
		tb[k] = -1
		if eta < 1 {
			tb[k] = delta[k] / (1 - eta)
		}
	}
	return h, tb, false
}

// checkWithinTerms checks resp, FreqResponse of the explicit delay-free sys
// at omega, against frTermsBound: each determined entry within δ/(1−η) of
// h, and each undetermined one (η ≥ 1, where the contract bounds the
// backward error only) finite and, for undeterminedTol > 0, within it
// relative of h. Exactly singular points are skipped.
func checkWithinTerms(t *testing.T, sys *System, omega []float64, resp *FreqResponseMatrix, undeterminedTol float64) {
	t.Helper()
	e := newFrequencyEvaluator(sys)
	_, m, p := sys.Dims()
	for k, w := range omega {
		h, tb, sing := frTermsBound(sys, e.pointAt(w))
		if sing {
			continue
		}
		for i := range p {
			for j := range m {
				v, want, b := resp.At(k, i, j), h[i*m+j], tb[i*m+j]
				d := cmplx.Abs(v - want)
				if b < 0 {
					if cmplx.IsInf(v) || cmplx.IsNaN(v) || undeterminedTol > 0 && !(d <= undeterminedTol*cmplx.Abs(want)) {
						t.Errorf("ω=%v (%d,%d) undetermined: got %v, want %v", w, i, j, v, want)
					}
					continue
				}
				if !(d <= b) {
					t.Errorf("ω=%v (%d,%d): got %v, want %v within %.3g (%.3g×)", w, i, j, v, want, b, d/b)
				}
			}
		}
	}
}
