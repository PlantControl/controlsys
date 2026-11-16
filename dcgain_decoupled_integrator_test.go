package controlsys

import (
	"testing"
)

// An exactly decoupled integrator (zero row and column of A, zero residue)
// beside an ill-conditioned regular block: the gain is the regular part's,
// which a plain float solve misses by up to κ·ε. Before ONYZO7 the route
// trusted the decoupled modes' sure signs for the finite entries too and
// returned the float solve with nil (3.98794 for an exact 4).
func TestDCGainDecoupledIntegratorBesideIllConditionedBlock(t *testing.T) {
	cases := []struct {
		name    string
		n       int
		a, b, c []float64
		want    float64
	}{
		{"exact 4", 3,
			[]float64{0, 0, 0, 0, 5.4764069e+07, -1.01406e+06, 0, -1.825847e+06, 33809},
			[]float64{-0, 1.07500018e+08, -3.584076e+06}, []float64{1, -1, -1}, 4},
		{"exact 2", 3,
			[]float64{0, 0, 0, 0, -6.0265341e+07, -3610, 0, -16694, -1},
			[]float64{-0, -6.0254511e+07, -16691}, []float64{1, 1, 1}, 2},
		{"exact 0, n=3", 3,
			[]float64{0, 0, 0, 0, -1, 6513, 0, 12756, -8.3079829e+07},
			[]float64{-0, 13028, -1.6618517e+08}, []float64{1, 1, 1}, 0},
		{"exact 0, n=4", 4,
			[]float64{0, 0, 0, 0, 0, -1, -0, -0, 0, -1919, -1, -0, 0, -134649, -986, -1},
			[]float64{-0, 0, -2, -1970}, []float64{1, 0, -1, -1}, 0},
	}
	for _, c := range cases {
		sys, err := NewFromSlices(c.n, 1, 1, c.a, c.b, c.c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: DCGain: %v", c.name, err)
		}
		if got := g.At(0, 0); got != c.want {
			t.Errorf("%s: DCGain = %.17g, want %g", c.name, got, c.want)
		}
	}
}

// A pole at DC whose state no channel reaches (structural reduction) leaves
// a nonsingular channel solve certified only within its terms; at order ≤
// exactDetMaxOrder DCGain's nil promises the exact limit, here 0 (ONYZO7).
func TestDCGainStructuralChannelBesideUnreachedIntegratorExactLimit(t *testing.T) {
	sys, err := NewFromSlices(4, 1, 1,
		[]float64{0, 1, 0, 0, 0, -1, -0, -0, 0, 934, -1, -431, 0, -393637, 420, 181019},
		[]float64{-0, 0, -1296, 544317}, []float64{1, 1, -1, 1}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("DCGain: %v", err)
	}
	if got := g.At(0, 0); got != 0 {
		t.Errorf("DCGain = %.17g, want 0", got)
	}
}
