package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestZPKModelRotatedDoubleIntegratorKeepsMarkovGain checks ZPKModel of a
// continuous double integrator in a non-modal basis against a 320-bit
// oracle. eig(A) splits the double pole at 0 to ±8e-11, so a gain fitted at
// probe points inside that split misses C·B = 1.48822 by 45% at every
// frequency; the leading Markov parameter is well conditioned.
func TestZPKModelRotatedDoubleIntegratorKeepsMarkovGain(t *testing.T) {
	A := []float64{0.029758384136693662, 1.0383761448255364, -0.0008528329843091777, -0.029758384136693666}
	B := []float64{0.8004786008717647, 1.1431085773578076}
	C := []float64{1.1316053315232386, 0.5094787509615359}
	sys, err := New(mat.NewDense(2, 2, A), mat.NewDense(2, 1, B), mat.NewDense(1, 2, C), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	z := res.ZPK
	if len(z.Poles[0][0]) != 2 || len(z.Zeros[0][0]) != 1 {
		t.Fatalf("got %d poles and %d zeros, want 2 and 1", len(z.Poles[0][0]), len(z.Zeros[0][0]))
	}
	o := newStepOraclePencil(sys)
	for _, s := range []complex128{0.5i, 2i, 10i, 1 + 1i} {
		got, err := z.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want := matchedOracleTransfer(o, s)
		if rel := cmplx.Abs(got[0][0]-want) / cmplx.Abs(want); !(rel <= 1e-9) {
			t.Errorf("H(%v) = %v, oracle %v (rel %.3g)", s, got[0][0], want, rel)
		}
	}
}
