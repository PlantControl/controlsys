package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// halfDiskRoots counts the zeros of the closed-form chi inside the right
// half-disk of radius B by the argument principle, bisecting until every
// phase step is below 0.2 rad; tau is the largest delay in chi.
func halfDiskRoots(t *testing.T, chi func(complex128) complex128, tau, B float64) int {
	t.Helper()
	total := 0.0
	var walk func(a, b, fa, fb complex128, depth int)
	walk = func(a, b, fa, fb complex128, depth int) {
		if d := cmplx.Phase(fb / fa); math.Abs(d) <= 0.2 {
			total += d
			return
		}
		if depth > 60 {
			t.Fatalf("oracle does not resolve χ near %v", a)
		}
		m := (a + b) / 2
		fm := chi(m)
		walk(a, m, fa, fm, depth+1)
		walk(m, b, fm, fb, depth+1)
	}
	n := int(8*B*max(tau, 1/B)) + 1000
	path := make([]complex128, 0, 2*n+1)
	for i := range n + 1 {
		path = append(path, complex(0, B*(1-2*float64(i)/float64(n))))
	}
	for i := 1; i <= n; i++ {
		path = append(path, cmplx.Rect(B, -math.Pi/2+math.Pi*float64(i)/float64(n)))
	}
	for i := 1; i < len(path); i++ {
		walk(path[i-1], path[i], chi(path[i-1]), chi(path[i]), 0)
	}
	return int(math.Round(total / (2 * math.Pi)))
}

func TestNyquistDelayWindingsNotAliased(t *testing.T) {
	cases := []struct {
		name string
		q    quasiLoop
		B    float64 // |num/den| < 1 on Re s >= 0, |s| >= B
	}{
		{"k=50", quasiLoop{num: []float64{50}, den: []float64{1, 1}, tau: 1}, 100},
		{"k=100", quasiLoop{num: []float64{100}, den: []float64{1, 1}, tau: 1}, 200},
		{"k=40 tau=3", quasiLoop{num: []float64{40}, den: []float64{1, 1.5, 2}, tau: 3}, 20},
		// D != 0: |L| > 1 out to ω ≈ z/√3, past the base grid end 1e3/τ.
		{"D=0.5 z=100", quasiLoop{num: []float64{0.5, 50}, den: []float64{1, 1}, tau: 1}, 400},
		{"D=0.5 z=4000", quasiLoop{num: []float64{0.5, 2000}, den: []float64{1, 1}, tau: 1}, 16000},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want := halfDiskRoots(t, c.q.chi, c.q.tau, c.B)
			sys := c.q.system(t)
			res, err := sys.Nyquist(nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			if res.RHPPoles != 0 || res.RHPZerosCL != want {
				t.Fatalf("RHPZerosCL %d (RHPPoles %d), argument principle %d", res.RHPZerosCL, res.RHPPoles, want)
			}
			lft, err := sys.PullDelaysToLFT()
			if err != nil {
				t.Fatal(err)
			}
			if !lft.HasInternalDelay() {
				return
			}
			res, err = lft.Nyquist(nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			if res.RHPZerosCL != want {
				t.Fatalf("internal-delay RHPZerosCL %d, argument principle %d", res.RHPZerosCL, want)
			}
		})
	}
}

// TestNyquistInternalDelayEnvelope covers |L| rippling with internal delays:
// L = k/(s+1)·(e^{-s} + e^{-τ₂s}) nearly vanishes wherever (τ₂-1)ω/2 is an odd
// multiple of π/2, so samples of |L| alone can straddle a winding (these
// cases miss 6 roots that way).
func TestNyquistInternalDelayEnvelope(t *testing.T) {
	const tau2 = 4.797797254423184
	for _, k := range []float64{20, 40} {
		mk := func(tau float64) *System {
			sys, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{k}, []float64{0}, 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := sys.SetInputDelay([]float64{tau}); err != nil {
				t.Fatal(err)
			}
			return sys
		}
		sys, err := Parallel(mk(1), mk(tau2))
		if err != nil {
			t.Fatal(err)
		}
		if !sys.HasInternalDelay() {
			t.Fatal("parallel delays not internal")
		}
		chi := func(s complex128) complex128 {
			return s + 1 + complex(k, 0)*(cmplx.Exp(-s)+cmplx.Exp(-tau2*s))
		}
		want := halfDiskRoots(t, chi, tau2, 8*k)
		res, err := sys.Nyquist(nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		if res.RHPPoles != 0 || res.RHPZerosCL != want {
			t.Fatalf("k=%g: RHPZerosCL %d (RHPPoles %d), argument principle %d", k, res.RHPZerosCL, res.RHPPoles, want)
		}
	}
}

// A loop with |L| > 1 out to ω ≈ 5e5 and τ = 20 has millions of closed-loop
// RHP roots; resolving them exceeds the sample budget, which must be reported
// rather than a count from an aliased grid.
func TestNyquistDelayBudget(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{-1.5848931924611134, 0, 0, -1.5848931924611134},
		[]float64{-0.9667981925794609, -0.01238866346289056},
		[]float64{856888.7533689471, 210795.79943077973}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{19.9526231496888}); err != nil {
		t.Fatal(err)
	}
	if _, err := sys.Nyquist(nil, 0); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("err %v, want ErrDelayUnsupported", err)
	}
}

// TestNyquistDelaySmallGain covers small-gain loops with a fast pole: |L| < 1
// on Re s >= 0 (Rouché: no closed-loop RHP roots), but |L| >= 1/2 out to
// ω ≈ p with pτ >= 1e5, so resolving e^{-sτ} there would exceed the budget.
func TestNyquistDelaySmallGain(t *testing.T) {
	cases := []quasiLoop{
		{name: "p=1e6", num: []float64{9e5}, den: []float64{1, 1e6}, tau: 1},
		{name: "p=1e7", num: []float64{9e6}, den: []float64{1, 1e7}, tau: 1},
		{name: "p=1e5 tau=10", num: []float64{9e4}, den: []float64{1, 1e5}, tau: 10},
		{name: "D=0.6", num: []float64{0.6, 9e6}, den: []float64{1, 1e7}, tau: 1},
		{name: "sweep1", num: []float64{414034.1}, den: []float64{1, 427103.1}, tau: 1.58},
		{name: "sweep2", num: []float64{848522.9}, den: []float64{1, 897705.5}, tau: 3.21},
		{name: "sweep3", num: []float64{1.1295e6}, den: []float64{1, 1.3847e6}, tau: 0.60},
	}
	for _, q := range cases {
		t.Run(q.name, func(t *testing.T) {
			if want := halfDiskRoots(t, q.chi, q.tau, 10); want != 0 {
				t.Fatalf("argument principle %d, want 0", want)
			}
			sys := q.system(t)
			lft, err := sys.PullDelaysToLFT()
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range []*System{sys, lft} {
				res, err := s.Nyquist(nil, 0)
				if err != nil {
					t.Fatalf("internal=%v: %v", s.HasInternalDelay(), err)
				}
				if res.RHPPoles != 0 || res.RHPZerosCL != 0 {
					t.Fatalf("internal=%v: RHPZerosCL %d (RHPPoles %d), want 0", s.HasInternalDelay(), res.RHPZerosCL, res.RHPPoles)
				}
			}
		})
	}
}
