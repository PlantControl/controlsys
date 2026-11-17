package controlsys

import (
	"errors"
	"math"
	"testing"
)

// ZEYADU: at an exactly singular DC pencil of order up to exactDetMaxOrder
// the exact limit decides over the float route past the as-given Laurent
// expansion. An ill-conditioned integer block (unimodular, entries to 1e9)
// has a computed pole near DC that the transfer limit or a deflated
// expansion reads as an integrator, or reads with the wrong Inf sign. Where
// every pole coefficient of the exact limit is exactly 0, or its lowest
// nonzero one exceeds ε of its own terms, DCGain is that limit with a nil
// error, and FreqResponse at DC follows it.
func TestDCGainExactLimitOverFloatRoute(t *testing.T) {
	for _, tc := range []dcGainLiteral{
		// hunt-true 30: exact cancellation, finite; the transfer limit reads +Inf.
		{"cancelled integrator", 5, 1, 1, []float64{0, -2, 2, 2, 1, 0, -1, -456, 0, 1759, 0, 3052, 1391711, 0, -5368468, 0, 0, 3647, -1, 0, 0, 0, 0, 0, -1}, []float64{1, 5734, -17500167, -3646, -3}, []float64{1, 1, -1, 0, 1}, []float64{0}, 0},
		// hunt-false 3849: +Inf; the deflated expansion reads −Inf.
		{"integrator, deflated sign flipped", 3, 1, 1, []float64{0, 2, 2, 0, 442082, -127761251, 0, -989, 285820}, []float64{-5, -1326246, 2967}, []float64{1, -1, 0}, []float64{0}, 0},
		// hunt-false 3309: +Inf; the transfer limit reads a finite gain.
		{"integrator, transfer limit finite", 4, 1, 1, []float64{0, -2, -2, -1, 0, 32966607, 0, -4336, 0, 42039465, -1, -5529, 0, 7603, 0, -1}, []float64{6, 65946222, 84095520, 15209}, []float64{1, 1, 1, 0}, []float64{0}, 0},
	} {
		sys := tc.system(t)
		want := dcGainRationalLimit(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.x0())[0]
		g, err := sys.DCGain()
		if err != nil || g.At(0, 0) != want {
			t.Errorf("%s: DCGain = %v, %v; want %v, nil", tc.name, g, err, want)
			continue
		}
		fr, ferr := sys.FreqResponse([]float64{0})
		if ferr != nil {
			t.Errorf("%s: FreqResponse(0): %v", tc.name, ferr)
			continue
		}
		if h := fr.At(0, 0, 0); h != complex(want, 0) {
			t.Errorf("%s: FreqResponse(0) = %v, want %v", tc.name, h, want)
		}
	}
}

// ZEYADU: where the exact limit's Inf rests on a pole coefficient below ε of
// its own terms and the deflated expansion reads the other Inf, DCGain keeps
// the exact limit with ErrAccuracyUnverified, and so does FreqResponse at DC
// (hunt-false 3404).
func TestDCGainExactLimitBelowRoundingAgainstFloatRoute(t *testing.T) {
	tc := dcGainLiteral{"sub-rounding residue", 3, 1, 1, []float64{0, 1, 1, 0, -3431681, -1915, 0, -991757601, -553436}, []float64{5, -6867192, -1984622074}, []float64{1, 1, -1}, []float64{0}, 0}
	sys := tc.system(t)
	want := dcGainRationalLimit(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.x0())[0]
	if !math.IsInf(want, 1) {
		t.Fatalf("exact limit %v, want +Inf", want)
	}
	g, err := sys.DCGain()
	if !errors.Is(err, ErrAccuracyUnverified) || g == nil || g.At(0, 0) != want {
		t.Errorf("DCGain = %v, %v; want %v with ErrAccuracyUnverified", g, err, want)
	}
	if _, ferr := sys.FreqResponse([]float64{0}); !errors.Is(ferr, ErrAccuracyUnverified) {
		t.Errorf("FreqResponse(0): %v, want ErrAccuracyUnverified", ferr)
	}
}
