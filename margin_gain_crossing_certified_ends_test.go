package controlsys

import (
	"errors"
	"math"
	"testing"
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

// dover-grid a=0.7,b=0.99,k=5.1 with 14 unobservable states added: the
// loop is the same, but at order 17 > exactDetMaxOrder no exact response
// certifies a bracket end of the scan below the pencil crossing, so
// AllMargin is ErrAccuracyUnverified (plain rejection out of the exact
// scope, OXOAL6 ruling 2). Main returned the noise-bracketed
// [1.0691800428606525e-12 0.0060107292292177926] with nil; A2OX65 320-bit
// gcf oracle [3.7007434156454433e-13 0.0060107292303075814].
func TestAllMarginGainCrossBelowScanOutOfExactScope(t *testing.T) {
	core := []float64{1, 0, 0, 1.5350000000000001, 0, -0.69299999999999995, -1.5500000000000003, 1, 1.6899999999999999}
	const n = 17
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
	if res, err := AllMargin(sys); !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("AllMargin = %v, %v; want ErrAccuracyUnverified", res, err)
	}
}
