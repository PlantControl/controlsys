package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// fzFRSweepModel stacks the decoded model into the upper block-triangular
// [A K₁ K₂; 0 2A K₃; 0 0 A/2] (poles s, 2s, s/2; entries formed exactly),
// B = [B; B; B], C = [C C C], large enough for the Hessenberg sweep.
func fzFRSweepModel(s *fzSpec) *fzFRModel {
	n, m, p := s.N, s.M, s.P
	N := 3 * n
	md := &fzFRModel{n: N, m: m, p: p, a: make([]float64, N*N), b: make([]float64, N*m), c: make([]float64, p*N), d: s.D, blk: []int{n, n, n}}
	scale := [3]float64{1, 2, 0.5}
	for bi := range 3 {
		for i := range n {
			r := bi*n + i
			for j := range n {
				md.a[r*N+bi*n+j] = scale[bi] * s.A[i*n+j]
				for bj := bi + 1; bj < 3; bj++ {
					md.a[r*N+bj*n+j] = 0.5 * math.Cos(float64(3*r+5*j+bj)+s.Vals[10])
				}
			}
			copy(md.b[r*m:(r+1)*m], s.B[i*m:(i+1)*m])
		}
		for i := range p {
			copy(md.c[i*N+bi*n:i*N+(bi+1)*n], s.C[i*n:(i+1)*n])
		}
	}
	return md
}

// FuzzFreqResponseSweep: on the stacked model of fzFRSweepModel, a sweep
// long enough that FreqResponse takes the Hessenberg path (where the
// backend's cost model allows), and both engines called directly, match
// the exact response.
func FuzzFreqResponseSweep(f *testing.F) {
	for _, s := range fzFRSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.N < 2 {
			fzSkip(t, "sweep needs n >= 2")
		}
		md := fzFRSweepModel(s)
		for _, x := range md.a {
			if fzBadFloat(x) {
				fzSkip(t, "denormal realization entry")
			}
		}
		sys := md.system(t, s.Dt)
		var ws []float64
		for _, k := range []float64{1, 2, 0.5} {
			for _, w := range fzFRFreqs(s) {
				ws = append(ws, k*w)
			}
		}
		lo, hi := 1e-3, 1e3
		for _, e := range s.S {
			if a := cmplx.Abs(e); a > 0 {
				lo, hi = math.Min(lo, a/10), math.Max(hi, 10*a)
			}
		}
		for i := range 40 {
			ws = append(ws, lo*math.Pow(hi/lo, float64(i)/39))
		}
		if s.Discrete {
			ws = slices.DeleteFunc(ws, func(w float64) bool { return w > math.Pi/s.Dt })
		}
		slices.Sort(ws)
		ws = slices.Compact(ws)
		ev := newFrequencyEvaluator(sys)
		for len(ws) < 400 && ev.useDenseSweep(len(ws)) && !ev.useDenseSweep(400) {
			ws = append(ws, ws...)
		}
		fr, err := sys.FreqResponse(ws)
		fzContract(t, s, "FreqResponse", fr != nil, err)
		fzErr(t, s, "FreqResponse", err)
		dense := newBalancedDense(sys, md.n, md.m, md.p)
		hess := newHessenbergSweep(sys, md.n, md.m, md.p)
		ps := &fzSpec{Discrete: s.Discrete, Dt: s.Dt, P: s.P, M: s.M}
		seen := map[float64]bool{}
		stride := max(1, len(slices.Compact(slices.Clone(ws)))/24)
		for k, w := range ws {
			if seen[w] {
				continue
			}
			seen[w] = true
			if len(seen)%stride != 0 {
				continue
			}
			d := fzFROracle(ps, md, w, false)
			fzFRCheck(t, s, "sweep FreqResponse", w, fzFRAt(fr, k), d.g, d.beta, d.cond, nil)
			for _, eng := range []struct {
				name string
				eval func(frequencyPoint, []complex128) error
			}{{"balancedDense", dense.evalInto}, {"hessenbergSweep", hess.evalInto}} {
				got := make([]complex128, s.P*s.M)
				if err := eng.eval(d.pt, got); err != nil {
					if !errors.Is(err, ErrSingularTransform) || d.cond < fzFRCondPole {
						t.Fatalf("sweep %s at ω=%g: %v at point condition %.3g\n%s", eng.name, w, err, d.cond, s)
					}
					continue
				}
				fzFRCheck(t, s, "sweep "+eng.name, w, got, d.g, d.beta, d.cond, nil)
			}
		}
	})
}

// FuzzFreqResponseInternalDelay: with the internal delay loop of
// fzSpec.System (w = Δ·z, z = c₂ᵀx, ẋ gets b₂w, D blocks zero) the model is
// the explicit one with A_Δ = A + Δ·b₂c₂ᵀ, Δ = e^{−jωτ} (z^{−τ}). The oracle
// solves (xI − A_Δ)X = B exactly with Δ rounded to complex128, so a pole of
// A that the loop moves is no pole here; slack covers Δ's rounding through
// |∂G/∂Δ| = |CᵢRb₂|·|c₂ᵀRBⱼ|. I/O delays multiply as in FuzzFreqResponse.
func FuzzFreqResponseInternalDelay(f *testing.F) {
	for _, s := range fzFRSeeds() {
		s.Tau = math.Log10(0.3)
		fzAddSeed(f, s)
		s.Tau = math.Log10(2)
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		const eps = 0x1p-52
		if s.N == 0 || s.Tau == 0 {
			fzSkip(t, "no internal delay")
		}
		sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
		b2, c2 := fzFRInternalLoop(s)
		ws := fzFRFreqs(s)
		fr, err := sys.FreqResponse(ws)
		fzContract(t, s, "FreqResponse", fr != nil, err)
		if errors.Is(err, ErrSingularTransform) {
			for _, w := range ws {
				_, x := fzFRPoint(s, w)
				if _, cond := fzFRModelOf(s).bound(fzFRRound(x)); cond >= fzFRCondCheck {
					fzSkip(t, "precision limit: internal-delay model with xI − A singular to working precision")
				}
			}
		}
		fzErr(t, s, "FreqResponse", err)
		lftLoss := false
		for k, w := range ws {
			d, known := fzFRInternalOracle(s, b2, c2, w)
			if fzFRCheckKnown(t, s, "internal-delay FreqResponse", w, fzFRAt(fr, k), d.gd, d.beta, d.cond, d.slack, known) {
				lftLoss = true
			}
		}
		if lftLoss {
			fzSkip(t, "known: internal-delay LFT formula loses accuracy at an ill-conditioned open-loop point")
		}
	})
}

// fzFRInternalLoop returns the loop vectors b₂, c₂ of fzSpec.System's
// internal delay.
func fzFRInternalLoop(s *fzSpec) (b2, c2 []float64) {
	b2 = make([]float64, s.N)
	c2 = make([]float64, s.N)
	for i := range s.N {
		b2[i] = s.B[i*s.M]
		c2[i] = 0.5 * math.Sin(float64(i)+s.Vals[11])
	}
	return b2, c2
}

// fzFRInternalOracle is the exact internal-delay response gd at w with its
// bounds beta, slack and cond, and the known LFT allowance.
func fzFRInternalOracle(s *fzSpec, b2, c2 []float64, w float64) (fzFRPointData, []float64) {
	const eps = 0x1p-52
	n, m, p := s.N, s.M, s.P
	pt, x := fzFRPoint(s, w)
	phi := w * s.Tau
	if s.Discrete {
		phi = 2 * s.Tau * cmplx.Phase(pt.p)
	}
	delta := cmplx.Exp(complex(0, -phi))
	dr := fzCOf(real(delta), imag(delta))
	z := fzFRRound(x)
	M := make([]*fzCRat, n*n)
	Z := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			e := fzCOf(-s.A[i*n+j], 0)
			var bc fzCRat
			bc.re.Mul(fzRat(b2[i]), fzRat(c2[j]))
			t := fzCMul(&bc, dr)
			e.re.Sub(&e.re, &t.re)
			e.im.Sub(&e.im, &t.im)
			if i == j {
				e.re.Add(&e.re, &x.re)
				e.im.Add(&e.im, &x.im)
			}
			M[i*n+j] = e
			Z[i*n+j] = complex(-s.A[i*n+j], 0) - delta*complex(b2[i]*c2[j], 0)
		}
		Z[i*n+i] += z
	}
	var gd []complex128
	if X, ok := fzCSolve(n, M, m, fzCReal(n, m, s.B)); ok {
		G := fzCReal(p, m, s.D)
		CX := fzCMatMul(p, n, m, fzCReal(p, n, s.C), X)
		for e := range G {
			G[e].re.Add(&G[e].re, &CX[e].re)
			G[e].im.Add(&G[e].im, &CX[e].im)
		}
		gd = fzCRound(G)
	}
	nZ := fzFrob(s.A) + cmplx.Abs(z) + fzFrob(b2)*fzFrob(c2)
	beta, cond := fzFRBoundZ(n, m, p, Z, nZ, s.B, s.C, s.D)
	slack := make([]float64, p*m)
	if R, ok := fzCInv(n, Z); ok && gd != nil {
		CRb := fzFRCMul(p, n, 1, fzCMulF(p, n, n, R, s.C, true), fzFRComplex(b2))
		cRB := fzCMulF(1, n, m, fzFRCMul(1, n, n, fzFRComplex(c2), R), s.B, false)
		f, phiIO := fzFRDelayPhase(s, pt, w, true)
		for e := range gd {
			sens := cmplx.Abs(CRb[e/m]) * cmplx.Abs(cRB[e%m])
			slack[e] = 8*eps*(2+math.Abs(phi))*sens + 8*eps*(2+phiIO)*cmplx.Abs(gd[e])
			gd[e] *= f[e]
		}
	}
	return fzFRPointData{w: w, pt: pt, gd: gd, beta: beta, slack: slack, cond: cond}, fzFRLFTKnown(s, b2, c2, x, delta, gd)
}

// TestFzFRLFTKnownCapped: an integrator-like pole at 1e-13 makes the
// open-loop terms of the LFT large at ω = 0 while xI − A stays well
// conditioned, so the allowance must stay at rounding level rather than
// scale with the open-loop terms (it once admitted errors of 4·|h|).
func TestFzFRLFTKnownCapped(t *testing.T) {
	s := fzDecode(t, 0x3a, [fzNVals]float64{-20, -99.9, 41, -45, 60.333333333333336, -45, -26.666666666666668, -13.45, 42, -10, -4.3757142857142854, 8.666666666666666, -10, -163, -10, -0.05228787452803376})
	if s.N != 1 || s.A[0] != 1e-13 || s.Tau == 0 {
		t.Fatalf("decoded spec changed:\n%s", s)
	}
	b2, c2 := fzFRInternalLoop(s)
	checked := 0
	for _, w := range fzFRFreqs(s) {
		d, known := fzFRInternalOracle(s, b2, c2, w)
		if d.gd == nil || d.cond >= fzFRCondCheck {
			continue
		}
		for e, h := range d.gd {
			checked++
			if !(known[e] <= 1e-9*cmplx.Abs(h)) {
				t.Errorf("ω=%g [%d]: known allowance %.3g for |h| %.3g", w, e, known[e], cmplx.Abs(h))
			}
		}
	}
	if checked == 0 {
		t.Fatal("no checked points")
	}
}

func fzFRComplex(x []float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex(v, 0)
	}
	return out
}

// fzFRLFTKnown bounds the error of evaluating G11 + G12·Δ·G21/(1 − G22·Δ)
// from a backward-stable open-loop solve of the augmented plant
// [A, [B b₂]; [C; c₂ᵀ], [D 0; 0 0]] (evalFrLFTInto): the open-loop bounds β
// propagate through the loop denominator, capped at the relative accuracy
// k·ε·cond of the open-loop point so cancellation of large open-loop terms
// cannot hide wrong values. Zero where the open-loop point is singular to
// working precision: the library then closes the loop before solving.
func fzFRLFTKnown(s *fzSpec, b2, c2 []float64, x *fzCRat, delta complex128, h []complex128) []float64 {
	const eps = 0x1p-52
	n, m, p := s.N, s.M, s.P
	ma, pa := m+1, p+1
	md := &fzFRModel{n: n, m: ma, p: pa, a: s.A, b: make([]float64, n*ma), c: make([]float64, pa*n), d: make([]float64, pa*ma)}
	for i := range n {
		copy(md.b[i*ma:i*ma+m], s.B[i*m:(i+1)*m])
		md.b[i*ma+m] = b2[i]
	}
	copy(md.c, s.C)
	copy(md.c[p*n:], c2)
	for i := range p {
		copy(md.d[i*ma:i*ma+m], s.D[i*m:(i+1)*m])
	}
	known := make([]float64, p*m)
	G, ok := md.exact(x)
	beta, cond := md.bound(fzFRRound(x))
	if !ok || h == nil || !(cond < fzFRCondCheck) {
		return known
	}
	k := 64 * float64(n+2) * eps * cond
	g22, b22 := G[p*ma+m], beta[p*ma+m]
	den := cmplx.Abs(1 - g22*delta)
	for i := range p {
		for j := range m {
			g12, g21 := cmplx.Abs(G[i*ma+m]), cmplx.Abs(G[p*ma+j])
			lft := 4 * (beta[i*ma+j] + (beta[i*ma+m]*g21+g12*beta[p*ma+j])/den + g12*g21*b22/(den*den))
			known[i*m+j] = min(lft, k*cmplx.Abs(h[i*m+j]))
		}
	}
	return known
}
