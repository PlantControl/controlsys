package controlsys

import (
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzIsStableInternalDelay: IsStable of a model with one internal delay
// loop (fzSysOpts.Internal, D22 = 0) matches the exact root count of its
// characteristic function χ_A − n·e^{−sτ} (z^τ·χ_A − n discrete), with
// n = c2·adj(λI−A)·b2 = χ_A − χ_{A+b2c2} by the matrix determinant lemma,
// whenever no root lies within a relative 1e-6 of the boundary; I/O delays
// never change the verdict.
func FuzzIsStableInternalDelay(f *testing.F) {
	for _, s := range fzInternalDelaySeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.N == 0 || s.Tau == 0 {
			fzSkip(t, "no internal delay")
		}
		sys := s.System(t, fzSysOpts{Internal: true})
		st, err := sys.IsStable()
		fzContract(t, s, "IsStable(internal)", st || err == nil, err)
		fzErr(t, s, "IsStable(internal)", err, ErrDelayUnsupported)
		dsys := s.System(t, fzSysOpts{Internal: true, Delays: true})
		dst, err := dsys.IsStable()
		fzContract(t, s, "IsStable(internal+I/O)", dst || err == nil, err)
		fzErr(t, s, "IsStable(internal+I/O)", err, ErrDelayUnsupported)
		if dst != st {
			t.Fatalf("IsStable with I/O delays = %v, without %v\n%s", dst, st, s)
		}
		chi, num := fzInternalLoop(sys)
		var lo, hi int
		var ok bool
		if s.Discrete {
			lo, hi, ok = fzInternalCountDiscrete(chi, num, int(s.Tau))
		} else {
			lo, hi, ok = fzInternalCountContinuous(chi, num, s.Tau, 1e-6*math.Max(1, s.NormJ))
		}
		if !ok || lo != hi {
			fzSkip(t, "characteristic root within margin of stability boundary")
		}
		if want := hi == 0; st != want {
			t.Fatalf("IsStable(internal) = %v, want %v (%d exact roots outside)\n%s", st, want, hi, s)
		}
	})
}

// fzInternalLoop returns χ_A and n = χ_A − χ_{A+b2c2} exactly, descending,
// for the single internal loop of sys.
func fzInternalLoop(sys *System) (chi, num []*big.Rat) {
	n, _, _ := sys.Dims()
	A := fzRatPoly(mat.DenseCopyOf(sys.A).RawMatrix().Data)
	b2 := mat.DenseCopyOf(sys.LFT.B2).RawMatrix().Data
	c2 := mat.DenseCopyOf(sys.LFT.C2).RawMatrix().Data
	Abc := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			Abc[i*n+j] = new(big.Rat).Add(A[i*n+j], new(big.Rat).Mul(fzRat(b2[i]), fzRat(c2[j])))
		}
	}
	chi = fzCharPolyRat(A, n)
	chi2 := fzCharPolyRat(Abc, n)
	num = make([]*big.Rat, n+1)
	for i := range num {
		num[i] = new(big.Rat).Sub(chi[i], chi2[i])
	}
	return chi, num
}

// fzInternalCountDiscrete counts roots of z^τ·χ − n exactly outside
// |z| = 1+2⁻²⁰ (lo) and 1−2⁻²⁰ (hi), radii with short dyadic expansions.
func fzInternalCountDiscrete(chi, num []*big.Rat, tau int) (lo, hi int, ok bool) {
	p := make([]*big.Rat, len(chi)+tau)
	for i := range p {
		p[i] = new(big.Rat)
		if i < len(chi) {
			p[i].Set(chi[i])
		}
	}
	off := len(p) - len(num)
	for i, c := range num {
		p[off+i].Sub(p[off+i], c)
	}
	lo, ok1 := fzSchurCohnOutside(p, 1+0x1p-20)
	hi, ok2 := fzSchurCohnOutside(p, 1-0x1p-20)
	return lo, hi, ok1 && ok2
}

// fzInternalCountContinuous counts roots of χ − n·e^{−sτ} with Re s > δ
// (lo) and Re s > −δ (hi) by the argument principle on the shifted
// quasi-polynomials χ(s+σ) − n(s+σ)e^{−στ}·e^{−sτ}.
func fzInternalCountContinuous(chi, num []*big.Rat, tau, delta float64) (lo, hi int, ok bool) {
	count := func(sigma float64) (int, bool) {
		den := fzRatFloats(fzTaylorShift(chi, fzRat(sigma)))
		nu := fzRatFloats(fzTaylorShift(num, fzRat(sigma)))
		k := -math.Exp(-sigma * tau)
		for i := range nu {
			nu[i] *= k
		}
		return fzQuasiRHP(nu, den, tau)
	}
	lo, ok1 := count(delta)
	hi, ok2 := count(-delta)
	return lo, hi, ok1 && ok2
}

func fzRatFloats(c []*big.Rat) []float64 {
	out := make([]float64, len(c))
	for i, x := range c {
		out[i], _ = x.Float64()
	}
	return out
}

func fzInternalDelaySeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Phase = 1, 0.5, 1, -1, 0.7 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Phase = 1, -0.5, 1, 0.5, 1.4 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau, s.Phase = 2, 0, 2.5, 0, 1.2 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau = 2, 0.3, 4, 0.8 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 2, 2
			s.Mag, s.Damp = [3]float64{0, 0.5}, [3]float64{2, 1.5}
			s.Tau, s.In, s.Out, s.IO = -0.5, 0, -1, 0.3
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 3, 1, 2
			s.Mag, s.Damp = [3]float64{-1, 0.2}, [3]float64{-1.5, 1}
			s.Tau, s.Out = -2, 0.4
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.Mag[0], s.Damp[0], s.Tau = 2, 0.5, 2, 0.5
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.5)
			s.N, s.M, s.P = 5, 2, 1
			s.Mag, s.Damp = [3]float64{0, -0.4, 0.2}, [3]float64{3, 2, 1}
			s.Tau, s.In = 1, 0.3
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -2
			s.N, s.M, s.P = 6, 3, 3
			s.Mag, s.Damp = [3]float64{1, 1.5, 0.5}, [3]float64{3, 2.5, 4}
			s.Tau, s.IO = 0.8, 0.2
		}),
	}
}
