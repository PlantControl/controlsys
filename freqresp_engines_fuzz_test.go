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
		for k, w := range ws {
			d := fzFRInternalOracle(s, b2, c2, w)
			fzFRCheck(t, s, "internal-delay FreqResponse", w, fzFRAt(fr, k), d.gd, d.beta, d.cond, d.slack)
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
// bounds beta, slack and cond.
func fzFRInternalOracle(s *fzSpec, b2, c2 []float64, w float64) fzFRPointData {
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
	return fzFRPointData{w: w, pt: pt, gd: gd, beta: beta, slack: slack, cond: cond}
}

func fzFRComplex(x []float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex(v, 0)
	}
	return out
}
