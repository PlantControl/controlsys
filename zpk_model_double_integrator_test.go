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
	zpkCheckOracleResponse(t, sys, z, []complex128{0.5i, 2i, 10i, 1 + 1i})
}

// TestZPKModelRotatedDiscreteDoubleIntegratorKeepsMarkovGain is the dt = 1
// analogue: eig(A) splits the double pole at z = 1 to 1 ± 3e-9 and a gain
// fitted inside that split misses C·B = 1.82670 by 60%.
func TestZPKModelRotatedDiscreteDoubleIntegratorKeepsMarkovGain(t *testing.T) {
	A := []float64{0.971954641643158, 1.0900815040101024, -0.0007215443271628928, 1.028045358356842}
	B := []float64{1.2626209011558214, 0.8942986198543982}
	C := []float64{0.5731509139865, 1.2334048611777249}
	sys, err := New(mat.NewDense(2, 2, A), mat.NewDense(2, 1, B), mat.NewDense(1, 2, C), mat.NewDense(1, 1, nil), 1)
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
	zpkCheckOracleResponse(t, sys, z, []complex128{cmplx.Rect(1, 0.3), cmplx.Rect(1, 1), cmplx.Rect(1, 2), -0.5})
}

func zpkCheckOracleResponse(t *testing.T, sys *System, z *ZPK, pts []complex128) {
	t.Helper()
	o := newStepOraclePencil(sys)
	for _, s := range pts {
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
