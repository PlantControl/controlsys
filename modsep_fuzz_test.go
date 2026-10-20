package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzModsep: on delay-free models Modsep(sys, cutoff) returns Slow + Fast
// with the original exact response (bound scaled by the separation
// conditioning), Slow.D = 0 and Fast.D = D. Continuous models also split
// as the intended natural frequencies do: Slow has exactly the poles with
// |s| < cutoff (Bauer–Fike radius). cutoff lies between distinct sorted
// intended |s|: index ⌊clamp((v[14]+3)/6,0,1)·(n+1)⌋, geometric mean of the
// neighbours (÷10 / ×10 at the ends). Discrete splits are not checked: the
// library compares |z| with cutoff where MATLAB freqsep uses |ln z|/Ts
// (followup).
func FuzzModsep(f *testing.F) {
	for _, s := range fzBalredSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		const eps = 0x1p-52
		n := s.N
		ws := make([]float64, n)
		for k, p := range s.S {
			ws[k] = cmplx.Abs(p)
		}
		slices.Sort(ws)
		ws = slices.Compact(ws)
		nw := len(ws)
		j := min(int(fzClamp((s.Vals[14]+3)/6, 0, 1)*float64(nw+1)), nw)
		var cutoff float64
		switch {
		case nw == 0:
			cutoff = 1
		case j == 0:
			cutoff = ws[0] / 10
		case j == nw:
			cutoff = ws[nw-1] * 10
		default:
			cutoff = math.Sqrt(ws[j-1] * ws[j])
		}
		if !(cutoff > 0) || math.IsInf(cutoff, 0) {
			fzSkip(t, "cutoff not representable")
		}
		sys := s.System(t, fzSysOpts{})
		res, err := Modsep(sys, cutoff)
		fzContract(t, s, "Modsep", res != nil, err)
		fzErr(t, s, "Modsep", err, ErrSchurFailed)
		if res.Slow == nil || res.Fast == nil {
			t.Fatalf("Modsep: nil part\n%s", s)
		}
		ns, _, _ := res.Slow.Dims()
		nf, _, _ := res.Fast.Dims()
		if ns+nf != n {
			t.Fatalf("Modsep orders %d+%d, want %d\n%s", ns, nf, n, s)
		}
		zero := make([]float64, s.P*s.M)
		if g := fzDenseClone(res.Slow.D); !slices.Equal(g, zero) {
			t.Fatalf("Modsep Slow.D = %v, want 0\n%s", g, s)
		}
		if g := fzDenseClone(res.Fast.D); !slices.Equal(g, s.D) {
			t.Fatalf("Modsep Fast.D = %v, want %v\n%s", g, s.D, s)
		}
		if res.Slow.Dt != s.Dt || res.Fast.Dt != s.Dt {
			t.Fatalf("Modsep changed Dt\n%s", s)
		}
		if n == 0 {
			return
		}
		r := s.EigErr()
		if s.Discrete {
			fzModsepAdditive(t, s, res, math.Inf(1), r)
			return
		}
		var slow, fast []complex128
		for _, e := range s.Eig {
			a := cmplx.Abs(e)
			if math.Abs(a-cutoff) <= 4*(r+1e-10*math.Max(1, a)) {
				fzSkip(t, "pole within error band of the cutoff")
			}
			if a < cutoff {
				slow = append(slow, e)
			} else {
				fast = append(fast, e)
			}
		}
		if ns != len(slow) {
			t.Fatalf("Modsep(cutoff %v) Slow has %d states, %d intended poles have |s| < cutoff\n%s", cutoff, ns, len(slow), s)
		}
		gap := math.Inf(1)
		for _, a := range slow {
			for _, b := range fast {
				gap = math.Min(gap, cmplx.Abs(a-b))
			}
		}
		kappa := 1 + s.CondT*s.CondT*math.Max(s.NormJ, 1)/gap
		rr := r + 1e3*float64(n)*eps*kappa*math.Max(s.NormJ, 1)
		for _, part := range []struct {
			name string
			sys  *System
			want []complex128
		}{{"Slow", res.Slow, slow}, {"Fast", res.Fast, fast}} {
			k, _, _ := part.sys.Dims()
			if k == 0 {
				continue
			}
			var e mat.Eigen
			if !e.Factorize(part.sys.A, mat.EigenNone) {
				fzSkip(t, "oracle eigen failed")
			}
			if _, _, msg := fzMatchEig(e.Values(nil), part.want, rr); msg != "" {
				t.Fatalf("Modsep %s poles: %s\n%s", part.name, msg, s)
			}
		}
		fzModsepAdditive(t, s, res, kappa, r)
	})
}

func fzModsepAdditive(t *testing.T, s *fzSpec, res *ModsepResult, kappa, r float64) {
	t.Helper()
	if math.IsInf(kappa, 0) {
		kappa = 1 + s.CondT*s.CondT*math.Max(s.NormJ, 1)/math.Max(r, 1e-300)
	}
	gs, gf := fzSysMats(res.Slow), fzSysMats(res.Fast)
	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()
	n, m, p := s.N, s.M, s.P
	for _, x := range fzTestPoints(s) {
		bound, ok := fzRespBound(n, s.A, s.B, s.C, s.D, x, 1e4*float64(n+m+p))
		b1, ok1 := fzRespBound(ns, gs[0], gs[1], gs[2], gs[3], x, 1e4*float64(n+m+p))
		b2, ok2 := fzRespBound(nf, gf[0], gf[1], gf[2], gf[3], x, 1e4*float64(n+m+p))
		if !ok || !ok1 || !ok2 {
			continue
		}
		bound = (bound + b1 + b2) * kappa
		g0, ok0 := fzExactResponse(n, m, p, s.A, s.B, s.C, s.D, x)
		g1, okS := fzExactResponse(ns, m, p, gs[0], gs[1], gs[2], gs[3], x)
		g2, okF := fzExactResponse(nf, m, p, gf[0], gf[1], gf[2], gf[3], x)
		if !ok0 || !okS || !okF {
			continue
		}
		for k := range g0 {
			if d := cmplx.Abs(g0[k] - g1[k] - g2[k]); d > bound {
				t.Fatalf("Modsep: Slow+Fast at %v entry %d = %v, want %v (|Δ| %.3g > %.3g)\n%s", x, k, g1[k]+g2[k], g0[k], d, bound, s)
			}
		}
	}
}
