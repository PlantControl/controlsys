package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Loops beside poles near DC whose double |L| below the first pencil
// crossing is rounding noise (OXOAL6, A2OX65 census): dpar0 reads 19.17 and
// 0.0156 in turn near 5.8e-13 where the stored model has 0.0128, and the
// below-crossing scan closed on that noise with nil. Its bracket ends now
// carry the sign of the stored model's exact response (exactResponseAt):
// nil and the 320-bit oracle crossings. dpar0 closes inside the dc
// rounding disk two decades below the noise; dover-pcg/5.1 and series1 in
// the decade the double read, past noisy bisection ends; dover-pcg/14.1
// in the band between the disk and the double solve's reach, where the
// double response is unsolved; dover-grid on a flip the exact response
// moves down a decade. dover-pcg/130.1 grazes |L| = 1 near 0.0781, where
// polish read a pair of crossings 1.4e-13 apart on double rounding; the
// exact signs there change once (fine 320-bit scan: one crossing).
func TestAllMarginGainCrossBelowScanCertifiedEnds(t *testing.T) {
	for _, tc := range []struct {
		name    string
		dt      float64
		a, b, c []float64
		oracle  []float64
	}{
		{"dpar0/a=0.1,b=0.7,k=5", 0.1,
			[]float64{0, -0.69999999999999996, 0, 1, 1.7, 0, 0, 0, 1},
			[]float64{-3.5, 5, -5},
			[]float64{0, 1, 1},
			[]float64{7.4014868308343759e-15}},
		{"dover-pcg/5.1", 0.1,
			[]float64{1, 0, 0, 41.047256209870888, 0, -0.015527761157662474, -70.089665533440126, 1, 0.31897826264881401},
			[]float64{1, 0, 0},
			[]float64{41.69468126205291, 0, 1},
			[]float64{4.1256422787808609e-14, 0.16707408199135748}},
		{"dover-pcg/14.1", 0.1,
			[]float64{1, 0, 0, 1.376085305350051, 0, -0.54865854644176604, -1.5272397684151886, 1, 1.499081478213224},
			[]float64{1, 0, 0},
			[]float64{3.0488786139660506, 0, 1},
			[]float64{1.1571457207361749e-14, 0.16373238649783406}},
		{"dover-grid/a=0.7,b=0.99,k=5.1", 0.1,
			[]float64{1, 0, 0, 1.5350000000000001, 0, -0.69299999999999995, -1.5500000000000003, 1, 1.6899999999999999},
			[]float64{1, 0, 0},
			[]float64{5, 0, 1},
			[]float64{3.7007434156454433e-13, 0.0060107292303075814}},
		{"dover-pcg/130.1", 0.1,
			[]float64{1, 0, 0, 29.50531391380034, 0, -0.20485999316463663, -40.260758743346351, 1, 0.91501103577792753},
			[]float64{1, 0, 0},
			[]float64{37.107067510325287, 0, 1},
			[]float64{1.8321125898066527e-14, 0.078119751347199576}},
		{"series1/232.1", 0,
			[]float64{-0, 0, -203.40831141013936, -14.532107196753987},
			[]float64{1, 0},
			[]float64{13.997165631669324, 1},
			[]float64{3.0506121242022132e-15}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(len(tc.b), 1, 1, tc.a, tc.b, tc.c, []float64{0}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			am, err := AllMargin(sys)
			if err != nil {
				t.Fatalf("AllMargin: %v, want oracle %.17g", err, tc.oracle)
			}
			if !sameCrossings(am.GainCrossFreqs, tc.oracle, 1e-8) {
				t.Fatalf("GainCrossFreqs %.17g, want oracle %.17g", am.GainCrossFreqs, tc.oracle)
			}
		})
	}
}

// crossingScan on a double reading that misplaces the sign of
// f = log(w0/ω) at chosen steps: the bracket ends carry the exact sign, so
// a misread step below the crossing is passed, one above it moves the
// bracket up, and a misread final end bisects again on exact values.
func TestCrossingScanCertifiedEnds(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(sys)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		w0     float64
		lo, hi float64
		read   float64
	}{
		{"flip above the crossing", 1e-6, 9e-4, 1.1e-3, 1},
		{"step above misread", 3e-2, 9e-3, 1.1e-2, -1},
		{"final end misread", 3e-2, 3e-2 * (1 - 1e-3), 3e-2, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			truth := func(w float64) float64 { return math.Log(tc.w0 / w) }
			misread := 0
			double := func(w float64) float64 {
				if w > tc.lo && w < tc.hi {
					misread++
					return tc.read
				}
				return truth(w)
			}
			cs := &crossingScan{eval: eval, f: double, tol: 1e-9, exact: func(w float64) (float64, bool) { return truth(w), true }}
			c, ok, _ := cs.scan(1, func(t float64) float64 { return t }, 1)
			if misread == 0 {
				t.Fatal("no step misread")
			}
			if !ok || math.Abs(c-tc.w0) > 1e-12*tc.w0 {
				t.Fatalf("scan = %.17g, %v; want %.17g", c, ok, tc.w0)
			}
		})
	}
}

// paddedDoverGrid is dover-grid a=0.7,b=0.99,k=5.1 (discrete, Dt 0.1) with
// n−3 unobservable states added: the loop, and so its crossings, are the
// order-3 core's. A2OX65 320-bit gcf oracle [3.7007434156454433e-13
// 0.0060107292303075814].
func paddedDoverGrid(t *testing.T, n int) *System {
	t.Helper()
	core := []float64{1, 0, 0, 1.5350000000000001, 0, -0.69299999999999995, -1.5500000000000003, 1, 1.6899999999999999}
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := range 3 {
		copy(a[i*n:i*n+3], core[i*3:i*3+3])
	}
	copy(b, []float64{1, 0, 0})
	copy(c, []float64{5, 0, 1})
	for i := 3; i < n; i++ {
		a[i*n+i], b[i] = 0.5, 1
	}
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Above marginEndMaxOrder, the padding mixed into the core (mixPadding) so
// that no exactly zero block drops it (structuralCore), no exact response
// certifies a bracket end of the scan below the pencil crossing, so
// AllMargin is ErrAccuracyUnverified (plain rejection out of the exact
// scope, OXOAL6 ruling 2; n moved from 17 to 25 by CISC3B ruling 4, and
// 65 added). Main returned the noise-bracketed
// [1.0691800428606525e-12 0.0060107292292177926] with nil at order 17.
func TestAllMarginGainCrossBelowScanOutOfExactScope(t *testing.T) {
	for _, n := range []int{25, 65} {
		if res, err := AllMargin(mixPadding(t, paddedDoverGrid(t, n), 3)); !errors.Is(err, ErrAccuracyUnverified) {
			t.Fatalf("n=%d: AllMargin = %v, %v; want ErrAccuracyUnverified", n, res, err)
		}
	}
}

// Past exactDetMaxOrder, within marginEndMaxOrder, the bracket ends carry
// the exact response's sign: the order-17 padding returns the core's
// crossings with nil (CISC3B). Both lie within the gain search's 1e-9 of
// log|L| = 0 at the oracle: 320-bit log|L| −1.8e-10 at 0.0060107292292,
// 2.6e-10 at the order-3 core's 0.0060107292319.
func TestAllMarginGainCrossBelowScanPaddedWithinMarginCap(t *testing.T) {
	want := []float64{3.7007434156454433e-13, 0.0060107292303075814}
	for _, n := range []int{3, 17} {
		res, err := AllMargin(paddedDoverGrid(t, n))
		if err != nil {
			t.Fatalf("n=%d: AllMargin: %v", n, err)
		}
		got := res.GainCrossFreqs
		if len(got) != len(want) {
			t.Fatalf("n=%d: GainCrossFreqs = %v, want %v", n, got, want)
		}
		for i, w := range want {
			if math.Abs(got[i]-w) > 1e-8*w {
				t.Errorf("n=%d: GainCrossFreqs[%d] = %.17g, want %.17g within 1e-8", n, i, got[i], w)
			}
		}
	}
}

// mixPadding rotates each state past the first n0 of sys with state 0 by a
// fixed Givens rotation: the transfer function within rounding, with no
// exactly zero coupling left for structuralCore to drop.
func mixPadding(t *testing.T, sys *System, n0 int) *System {
	t.Helper()
	n, _, _ := sys.Dims()
	a := mat.DenseCopyOf(sys.A)
	b := mat.DenseCopyOf(sys.B)
	c := mat.DenseCopyOf(sys.C)
	const cs, sn = 0.8, 0.6
	for k := n0; k < n; k++ {
		for j := range n {
			x, y := a.At(0, j), a.At(k, j)
			a.Set(0, j, cs*x-sn*y)
			a.Set(k, j, sn*x+cs*y)
		}
		for i := range n {
			x, y := a.At(i, 0), a.At(i, k)
			a.Set(i, 0, cs*x-sn*y)
			a.Set(i, k, sn*x+cs*y)
		}
		x, y := b.At(0, 0), b.At(k, 0)
		b.Set(0, 0, cs*x-sn*y)
		b.Set(k, 0, sn*x+cs*y)
		x, y = c.At(0, 0), c.At(0, k)
		c.Set(0, 0, cs*x-sn*y)
		c.Set(0, k, sn*x+cs*y)
	}
	out, err := New(a, b, c, mat.DenseCopyOf(sys.D), sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// padUnobservable adds n−n0 stable states that the output does not see to
// the SISO core (a0, b0, c0): the loop is the core's.
func padUnobservable(n0, n int, a0, b0, c0 []float64, dt float64) (a, b, c []float64) {
	a, b, c = make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := range n0 {
		copy(a[i*n:i*n+n0], a0[i*n0:i*n0+n0])
		b[i], c[i] = b0[i], c0[i]
	}
	for k := n0; k < n; k++ {
		if dt == 0 {
			a[k*n+k] = -0.5 - 0.37*float64(k-n0)
		} else {
			a[k*n+k] = 0.05 + 0.04*float64(k-n0)
		}
		b[k] = 1
	}
	return
}

// Loops with a pole at DC whose residue DCGain cannot tell from rounding
// (ErrAccuracyUnverified), padded past exactDetMaxOrder (CISC3B, A2OX65
// census series1/232.1 and dpar0 a=0.1,b=0.7,k=5). Main returned [] with
// nil: its DC end took the double solve's failed reading, the legacy
// residue (0.963 for series1, −3.5e-10 for dpar0), on the side of the level
// the pencil crossings start from, so no scan ran. The DC end is now the
// exact DC limit (a pole: |L| above every level), and up to
// marginEndMaxOrder the scan's ends carry the exact response's sign: the
// 320-bit oracle crossing with nil. Past marginDCMaxOrder no end certifies
// and AllMargin is ErrAccuracyUnverified, never [] with nil.
func TestAllMarginPaddedPoleAtDCResidueBelowRounding(t *testing.T) {
	cases := []struct {
		name    string
		n0      int
		a, b, c []float64
		dt      float64
		want    float64
	}{
		{"series1", 2, []float64{0, 0, -203.40831141013936, -14.532107196753987}, []float64{1, 0}, []float64{13.997165631669324, 1}, 0, 3.0506121242022132e-15},
		{"dpar0", 3, []float64{0, -0.69999999999999996, 0, 1, 1.7, 0, 0, 0, 1}, []float64{-3.5, 5, -5}, []float64{0, 1, 1}, 0.1, 7.401486830834376e-15},
	}
	for _, tc := range cases {
		for _, n := range []int{tc.n0, 17, marginEndMaxOrder, marginDCMaxOrder + 1} {
			a, b, c := padUnobservable(tc.n0, n, tc.a, tc.b, tc.c, tc.dt)
			sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			if n > marginDCMaxOrder {
				sys = mixPadding(t, sys, tc.n0)
			}
			res, err := AllMargin(sys)
			if n > marginDCMaxOrder {
				if !errors.Is(err, ErrAccuracyUnverified) {
					t.Errorf("%s n=%d: AllMargin = %v, %v; want ErrAccuracyUnverified", tc.name, n, res, err)
				}
				continue
			}
			if err != nil {
				t.Fatalf("%s n=%d: AllMargin: %v", tc.name, n, err)
			}
			if got := res.GainCrossFreqs; len(got) != 1 || math.Abs(got[0]-tc.want) > 1e-8*tc.want {
				t.Errorf("%s n=%d: GainCrossFreqs = %.17g, want [%.17g]", tc.name, n, got, tc.want)
			}
		}
	}
}
