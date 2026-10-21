package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// lftFeedthroughCases: L = (b0·s + b1)/(s + a)·e^{-sτ} with |b0| in (0.77, 1).
// PullDelaysToLFT moves b0 into D12·D21 with D = 0, so L(jω) does not settle
// toward D; the first two are the wave-15 repros (A249 flipped on amd64).
func lftFeedthroughCases() []quasiLoop {
	cases := []quasiLoop{
		{name: "repro", num: []float64{0.9370756617374014, 0.008757399675478617}, den: []float64{1, 1.0258522634845244}, tau: 1.5326090365796423},
		{name: "A249", num: []float64{-0.8561202058270045, 0.7368750743551509}, den: []float64{1, 1.0742800376367483}, tau: 0.67025051719364},
		{name: "unstable", num: []float64{0.9, 3}, den: []float64{1, 0.5}, tau: 2},
		{name: "D=-0.99", num: []float64{-0.99, 0.5}, den: []float64{1, 1}, tau: 0.3},
	}
	rng := rand.New(rand.NewPCG(16, 7))
	for range 40 {
		d := 0.77 + 0.22*rng.Float64()
		if rng.IntN(2) == 0 {
			d = -d
		}
		cases = append(cases, quasiLoop{
			name: "sweep",
			num:  []float64{d, 6*rng.Float64() - 3},
			den:  []float64{1, 0.1 + 2.9*rng.Float64()},
			tau:  math.Exp(math.Log(0.05) + rng.Float64()*math.Log(100)),
		})
	}
	return cases
}

func TestNyquistLFTFeedthroughMatchesIODelay(t *testing.T) {
	for _, q := range lftFeedthroughCases() {
		b0, b1, a := q.num[0], q.num[1], q.den[1]
		B := 2*(math.Abs(b1)+math.Abs(a))/(1-math.Abs(b0)) + 1
		want := halfDiskRoots(t, q.chi, q.tau, B)
		sys := q.system(t)
		lft, err := sys.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		if !lft.HasInternalDelay() || lft.D.At(0, 0) != 0 {
			t.Fatalf("%s: LFT form has D = %v, want internal delay with D = 0", q.name, lft.D.At(0, 0))
		}
		for _, s := range []*System{sys, lft} {
			res, err := s.Nyquist(nil, 0)
			if err != nil {
				t.Fatalf("%s %v tau=%v internal=%v: %v", q.name, q.num, q.tau, s.HasInternalDelay(), err)
			}
			if res.RHPPoles != 0 || res.RHPZerosCL != want {
				t.Fatalf("%s %v tau=%v internal=%v: RHPZerosCL %d (RHPPoles %d), argument principle %d", q.name, q.num, q.tau, s.HasInternalDelay(), res.RHPZerosCL, res.RHPPoles, want)
			}
		}
	}
}

// Two internal delays, D = 0: L = (b0·s + b1)/(s + a)·(e^{-s} + e^{-τ₂s}).
func TestNyquistLFTFeedthroughTwoDelays(t *testing.T) {
	const tau2 = 2.718281828459045
	for _, c := range []struct{ b0, b1, a float64 }{{0.45, 0.2, 1}, {-0.47, 0.3, 0.8}, {0.4, 1.5, 0.5}} {
		mk := func(tau float64) *System {
			sys, err := NewFromSlices(1, 1, 1, []float64{-c.a}, []float64{1}, []float64{c.b1 - c.b0*c.a}, []float64{c.b0}, 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := sys.SetInputDelay([]float64{tau}); err != nil {
				t.Fatal(err)
			}
			return sys
		}
		par, err := Parallel(mk(1), mk(tau2))
		if err != nil {
			t.Fatal(err)
		}
		sys, err := par.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		if sys.internalDelayCount() != 2 || sys.D.At(0, 0) != 0 {
			t.Fatalf("%v: %d internal delays, D = %v", c, sys.internalDelayCount(), sys.D.At(0, 0))
		}
		chi := func(s complex128) complex128 {
			return s + complex(c.a, 0) + (complex(c.b0, 0)*s+complex(c.b1, 0))*(cmplx.Exp(-s)+cmplx.Exp(-tau2*s))
		}
		B := 2*(2*math.Abs(c.b1)+c.a)/(1-2*math.Abs(c.b0)) + 1
		want := halfDiskRoots(t, chi, tau2, B)
		res, err := sys.Nyquist(nil, 0)
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		if res.RHPPoles != 0 || res.RHPZerosCL != want {
			t.Fatalf("%v: RHPZerosCL %d (RHPPoles %d), argument principle %d", c, res.RHPZerosCL, res.RHPPoles, want)
		}
	}
}
