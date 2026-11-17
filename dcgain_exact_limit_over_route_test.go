package controlsys

import (
	"errors"
	"math"
	"testing"
)

// ZEYADU: at an exactly singular DC pencil of order up to exactDetMaxOrder
// where the deflated expansion disagrees with the exact limit, a finite
// exact limit decides: the exact solve settles an exactly cancelled pole.
// An ill-conditioned integer block (unimodular, entries to 1e9) has a
// computed pole near DC that the transfer limit or a deflated expansion
// reads as an integrator. DCGain is the exact limit with a nil error, and
// FreqResponse at DC follows it.
//
// Infinite exact limits there stay ErrAccuracyUnverified, their Inf sign
// not robust to rounding of the stored data (INGGF2). Two +Inf rows kept for
// a follow-up on certifying Infs robust to that rounding (dcGainLiteral
// fields after the name):
//
//	hunt-false 3849, deflated expansion reads −Inf:
//	{3, 1, 1, A {0, 2, 2, 0, 442082, -127761251, 0, -989, 285820},
//	 B {-5, -1326246, 2967}, C {1, -1, 0}, D {0}, dt 0}
//	hunt-false 3309, transfer limit reads a finite gain:
//	{4, 1, 1, A {0, -2, -2, -1, 0, 32966607, 0, -4336, 0, 42039465, -1, -5529, 0, 7603, 0, -1},
//	 B {6, 65946222, 84095520, 15209}, C {1, 1, 1, 0}, D {0}, dt 0}
func TestDCGainExactLimitOverFloatRoute(t *testing.T) {
	// hunt-true 30: exact cancellation, finite; the transfer limit reads +Inf.
	tc := dcGainLiteral{"cancelled integrator", 5, 1, 1, []float64{0, -2, 2, 2, 1, 0, -1, -456, 0, 1759, 0, 3052, 1391711, 0, -5368468, 0, 0, 3647, -1, 0, 0, 0, 0, 0, -1}, []float64{1, 5734, -17500167, -3646, -3}, []float64{1, 1, -1, 0, 1}, []float64{0}, 0}
	sys := tc.system(t)
	want := dcGainRationalLimit(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.x0())[0]
	if math.IsInf(want, 0) {
		t.Fatalf("exact limit %v, want finite", want)
	}
	g, err := sys.DCGain()
	if err != nil || g.At(0, 0) != want {
		t.Fatalf("DCGain = %v, %v; want %v, nil", g, err, want)
	}
	fr, ferr := sys.FreqResponse([]float64{0})
	if ferr != nil {
		t.Fatalf("FreqResponse(0): %v", ferr)
	}
	if h := fr.At(0, 0, 0); h != complex(want, 0) {
		t.Errorf("FreqResponse(0) = %v, want %v", h, want)
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
