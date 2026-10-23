package controlsys

import (
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
)

// newUnitRowCompanionCase draws a slow or fast observable companion with
// graded B and p outputs, each eₙᵀ, a unit row at a random state, or dense.
// A unit row eₖᵀ, k < n−1, sees the slow states only through the products
// of the last column, so the observability cut drops them by magnitude.
func newUnitRowCompanionCase(r *rand.Rand) mimoCompanionCase {
	n := 2 + r.IntN(9)
	m := 1 + r.IntN(3)
	p := 2 + r.IntN(2)
	scale := math.Ldexp(1, -20+r.IntN(28))
	den := ratPolyFromRoots(hessenbergSweepRoots(r, n, scale))
	b := make([]float64, n*m)
	for k := range b {
		if r.IntN(3) > 0 {
			b[k] = math.Ldexp(float64(r.IntN(129)-64), r.IntN(41)-25)
		}
	}
	c := make([]float64, p*n)
	for i := range p {
		switch r.IntN(3) {
		case 0:
			c[i*n+n-1] = 1
		case 1:
			c[i*n+r.IntN(n)] = 1
		default:
			for k := range n {
				c[i*n+k] = float64(r.IntN(17)-8) / 4
			}
		}
	}
	return newObservableMIMO(den, b, c, m, p)
}

func checkUnitRowCompanion(t *testing.T, cs mimoCompanionCase, label string) {
	t.Helper()
	tf := convertMIMOCompanion(t, cs, false)
	if e := newMIMOExact(cs).respErr(tf, cs.m); !(e <= 1e-8) {
		t.Errorf("%s: response err %.3g; dens %v", label, e, tf.Den)
	}
	for i, d := range tf.Den {
		if len(d) == cs.n+1 && rhpRoots(d) != rhpRoots(cs.den) {
			t.Errorf("%s: row %d den %v RHP roots %d, want %d", label, i, d, rhpRoots(d), rhpRoots(cs.den))
		}
	}
}

// O57JTC: a slow observable companion whose rows are unit vectors at
// states other than the last, or dense next to them. The observability cut
// dropped the states such a row sees through the tiny last-column
// products, and keepObservable did not restore them: its error floor was
// eps times the largest entry of any row (196: row e₁ of ~1e8 next to a
// row of ~1e41 looked matched to 1e-8 while 18 off), the reference
// missing a small entry near the band edge by 1e-2 in both candidates
// vetoed an exact full row (352; 996 failed with the per-row floor alone),
// and a band point where the full row's den overflows counted as a miss
// (718). Each row's den must be det(sI − A), as MATLAB ss2tf's poly(A),
// and its response exact.
func TestTransferFunctionUnitRowCompanionKeepsStates(t *testing.T) {
	cases := []struct {
		name string
		cs   mimoCompanionCase
	}{
		{"196", newObservableMIMO(
			[]float64{1, 8.130073547363281e-05, 3.0550495466741268e-09, 6.791056161425252e-14, 9.642546828754932e-19, 8.904322543115604e-24, 5.215654656188393e-29, 1.7724939809177077e-34, 2.6762900857425205e-40},
			[]float64{240, -1.1920928955078125e-07, 0, 0, -1.71661376953125e-05, 0, 0, 0.3125, 0.109375, 0, -128, 0, -164, -0.0625, 0.65625, 0},
			[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 2, 2)},
		{"352", newObservableMIMO(
			[]float64{1, 2.6464462280273438e-05, 3.474838194961194e-10, 6.193369385051883e-16, -2.708997119065559e-21, 2.050031477399705e-26, -9.24513449952267e-33},
			[]float64{0, -576, 0.000518798828125, -0.02734375, 3.6875, -0.005859375, -6, 0, 0, -0.00341796875, 0, -0.01318359375},
			[]float64{0.5, -0.25, 1.5, 0.25, -0.5, 2, 1, 0, 0, 0, 0, 0, 1.25, 1.75, 0, 2, -2, 2}, 2, 3)},
		{"718", newObservableMIMO(
			[]float64{1, 4.9233436584472656e-05, 1.1448832992755342e-09, 1.6317086841844547e-14, 1.5552855945360954e-19, 1.0191240444586584e-24, 4.543848990188033e-30, 1.3114700602334192e-35, 2.186236393565597e-41, 1.5710419608824168e-47, 0},
			[]float64{-0.0009765625, 0, 0, 0, 0, 0, 0, 0, -0.03125, 0, 950272, -0.0068359375, 0, 14.75, -0.09375, -4.5, 0.013671875, 0, 0, 25.5, 0, 0, 1.7881393432617188e-06, -58, 10.5, 1.049041748046875e-05, 11, -4.0531158447265625e-06, 0, -31744},
			[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 3, 2)},
		{"996", newObservableMIMO(
			[]float64{1, 8.225440979003906e-06, 3.529976311256178e-11, 5.861467994999758e-17, 0},
			[]float64{-0.00040435791015625, -311296, 0, -2.6702880859375e-05},
			[]float64{0, 0, 0, 1, 0, 1, 0, 0, 0, -2, 1.5, -2}, 1, 3)},
		// w21 #10: C = [eₙᵀ; e₁ᵀ] gave orders 6 and 1 for 7 and 7.
		{"w21-10", newObservableMIMO(
			[]float64{1, 0.2900390625, 0.022520720958709717, 0.002077794633805752, 6.61780904920306e-5, 3.036461748706643e-6, 9.239124132420786e-9, 1.0830891739033179e-11},
			[]float64{-211765717959.0472, -1541821429211.5383, 2956277.822168, -13739.401396682866, -264534.0241872499, -389.58287621735127, 0.08732935319518553},
			[]float64{0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}, 1, 2)},
		// w21 #8: the dense row's den had 1.30e-18 for an exact 1.61e-19.
		{"w21-8", newObservableMIMO(
			[]float64{1, 0.001220703125, 4.949979484081268e-07, 8.685674401931465e-11, 6.572054098846758e-15, 1.6116125310493149e-19},
			[]float64{0.0146484375, 2.613387921948515e-15, 0, -2.1260578828901107e-11, 0, 7.026102893719522e-08, 0, 1.786480377754889e-05, 0, 0.6408260336388681},
			[]float64{0, 0, 0, 0, 1, -1.2654555457593786, -1.473550928819034, -1.6874863195643912, -0.004061948422853678, 0.4335796894003101}, 2, 2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := tc.cs
			tf := convertMIMOCompanion(t, cs, false)
			for i, got := range tf.Den {
				if len(got) != cs.n+1 {
					t.Fatalf("row %d den %v, want %v", i, got, cs.den)
				}
				for k, want := range cs.den {
					if math.Abs(got[k]-want) > 1e-12*math.Abs(want) {
						t.Errorf("row %d den[%d] = %.17g, want %.17g", i, k, got[k], want)
					}
				}
			}
			if e := newMIMOExact(cs).respErr(tf, cs.m); !(e <= 1e-10) {
				t.Errorf("response err %.3g", e)
			}
		})
	}
}

// TestTransferFunctionUnitRowCompanionSweep converts seeded companions of
// newUnitRowCompanionCase against the exact big.Rat entries (17 of 1500
// missed by over 1e-8, most by O(1), before O57JTC).
func TestTransferFunctionUnitRowCompanionSweep(t *testing.T) {
	skipSlowUnderRace(t)
	r := rand.New(rand.NewPCG(1, 5))
	for trial := range 1500 {
		checkUnitRowCompanion(t, newUnitRowCompanionCase(r), "trial "+strconv.Itoa(trial))
	}
}
