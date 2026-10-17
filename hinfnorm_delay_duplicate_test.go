package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

type sisoDelayLoop struct {
	a, b1, b2, c1, c2, d11, d12, d21, d22, h float64
}

func randomSISODelayLoop(rng *rand.Rand) sisoDelayLoop {
	p := sisoDelayLoop{
		a: -1 - 3*rng.Float64(), b1: 0.3 * rng.NormFloat64(), b2: rng.NormFloat64(),
		c1: rng.NormFloat64(), c2: rng.NormFloat64(),
		d11: rng.NormFloat64(), d12: rng.NormFloat64(), d21: rng.NormFloat64(),
		d22: 0.2 + 0.7*rng.Float64(),
	}
	if rng.NormFloat64() < 0 {
		p.d22 = -p.d22
	}
	p.h = 0.1 + rng.Float64()
	return p
}

func (p sisoDelayLoop) siso(t *testing.T) *System {
	t.Helper()
	H, err := NewFromSlices(1, 2, 2, []float64{p.a}, []float64{p.b1, p.b2}, []float64{p.c1, p.c2},
		[]float64{p.d11, p.d12, p.d21, p.d22}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{p.h})
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// duplicate returns the block-diagonal 2x2 of p and p with its second channel
// scaled by k: closed-loop roots are those of p, each doubled.
func (p sisoDelayLoop) duplicate(t *testing.T, k float64) *System {
	t.Helper()
	H, err := NewFromSlices(2, 4, 4, []float64{p.a, 0, 0, p.a},
		[]float64{p.b1, 0, p.b2, 0, 0, k * p.b1, 0, p.b2},
		[]float64{p.c1, 0, 0, p.c1, p.c2, 0, 0, p.c2},
		[]float64{p.d11, 0, p.d12, 0, 0, k * p.d11, 0, p.d12, p.d21, 0, p.d22, 0, 0, k * p.d21, 0, p.d22}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{p.h, p.h})
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestHinfNormDelayDuplicatedLightlyDamped: the loop has a closed-loop root
// pair at −0.0027828 ± 2.8446939j (Newton on χ below), so 1+L of the
// block-diagonal duplicate, det = (1+L₁)², turns by 2π between two grid samples
// that the endpoint phase step alone saw as −36°.
func TestHinfNormDelayDuplicatedLightlyDamped(t *testing.T) {
	p := sisoDelayLoop{a: -1.4666973867074775, b1: 1, b2: -2.2697439281287144, c1: 1, c2: 1,
		d12: 1, d21: 1, d22: -0.44819513831474195, h: 0.8637187571421795}
	chi := func(s complex128) complex128 {
		e := cmplx.Exp(-s * complex(p.h, 0))
		return (s-complex(p.a, 0))*(1-complex(p.d22, 0)*e) - complex(p.b2*p.c2, 0)*e
	}
	s := complex(0, 2.84)
	for range 50 {
		const d = 1e-7
		s -= chi(s) * complex(d, 0) / (chi(s+d) - chi(s))
	}
	if math.Abs(real(s)+0.0027828) > 1e-6 || math.Abs(imag(s)-2.8446939) > 1e-6 {
		t.Fatalf("oracle root %v", s)
	}

	want, wantW, err := HinfNorm(p.siso(t))
	if err != nil || math.IsInf(want, 1) {
		t.Fatalf("SISO HinfNorm = %g, %v", want, err)
	}
	if math.Abs(wantW-imag(s)) > 1e-3 {
		t.Fatalf("SISO peak at %g, root at %g", wantW, imag(s))
	}
	for _, k := range []float64{1, 1 - 1e-9} {
		got, w, err := HinfNorm(p.duplicate(t, k))
		if err != nil {
			t.Fatalf("k=%g: %v", k, err)
		}
		if math.Abs(got-want) > 1e-6*want || math.Abs(w-wantW) > 1e-4 {
			t.Errorf("k=%g: HinfNorm = %g at %g, want %g at %g", k, got, w, want, wantW)
		}
	}
}

// TestHinfNormDelayDuplicateSweep: duplicating a channel doubles every
// closed-loop root, so the duplicate is stable exactly when the channel is
// and has its norm.
func TestHinfNormDelayDuplicateSweep(t *testing.T) {
	const seeds = 300
	unstable := 0
	for seed := range seeds {
		p := randomSISODelayLoop(rand.New(rand.NewSource(int64(seed))))
		want, _, err := HinfNorm(p.siso(t))
		if err != nil {
			continue
		}
		got, _, err := HinfNorm(p.duplicate(t, 1))
		if err != nil {
			t.Errorf("seed %d: duplicate: %v (SISO %g)", seed, err, want)
			continue
		}
		if math.IsInf(want, 1) {
			unstable++
			if !math.IsInf(got, 1) {
				t.Errorf("seed %d: SISO unstable, duplicate %g", seed, got)
			}
			continue
		}
		if math.Abs(got-want) > 1e-6*want {
			t.Errorf("seed %d: duplicate %g, SISO %g", seed, got, want)
		}
	}
	if unstable == 0 || unstable == seeds {
		t.Fatalf("%d of %d unstable; sweep covers one side only", unstable, seeds)
	}
}
