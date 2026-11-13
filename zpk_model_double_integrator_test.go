package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestZPKModelRotatedDoubleIntegratorKeepsMarkovGain checks ZPKModel of a
// continuous double integrator in a non-modal basis against a 320-bit
// oracle. eig(A) splits the double pole at 0 to ±8e-11, so a gain fitted at
// probe points inside that split misses C·B = 1.48822 by 45% at every
// frequency; the leading Markov parameter is well conditioned. The stored
// A is not exactly singular (oracle response at s = 0 −1.1087e20, terms
// 1.1377e20), and s = 0 is a structured point. Where eig(A) splits the
// pole to ±8.2e-11 (arm64), s = 0 is kept: double GEPP pivots exactly
// zero there, double-double resolves the oracle value, and the pencil
// roots, Z(0) = −2.0074e20, miss it by 3.0e7·√ε: ErrAccuracyUnverified.
// Where eig(A) returns the double pole at −1.7e-18 (amd64), s = 0 lies
// within the solve's resolution ρ₀ = 4.6e-16 of it and is dropped, as the
// census drops it; the result verifies, within 0.80·√ε of the terms at the
// kept points (oracle). Either outcome is the contract's.
func TestZPKModelRotatedDoubleIntegratorKeepsMarkovGain(t *testing.T) {
	A := []float64{0.029758384136693662, 1.0383761448255364, -0.0008528329843091777, -0.029758384136693666}
	B := []float64{0.8004786008717647, 1.1431085773578076}
	C := []float64{1.1316053315232386, 0.5094787509615359}
	sys, err := New(mat.NewDense(2, 2, A), mat.NewDense(2, 1, B), mat.NewDense(1, 2, C), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZPKModel(nil)
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("err %v, want nil or ErrAccuracyUnverified", err)
	}
	z := res.ZPK
	if len(z.Poles[0][0]) != 2 || len(z.Zeros[0][0]) != 1 {
		t.Fatalf("got %d poles and %d zeros, want 2 and 1", len(z.Poles[0][0]), len(z.Zeros[0][0]))
	}
	zpkCheckOracleResponse(t, sys, z, []complex128{0.5i, 2i, 10i, 1 + 1i})
	if err != nil {
		return
	}
	// A nil result certifies within terms: at the frozen probe points (the
	// worst, 0.80·√ε, at s = 1e-6j) and at s = 0 wherever the verifier keeps it.
	pts := []complex128{1e-6i, 1e-3i, 0.1i, 0.5i, 2i, 10i, 1 + 1i}
	c, ok := newZPKRowCheck(sys)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	if drop, ok := c.dropped(0); !ok {
		t.Fatal("rounding bounds did not converge")
	} else if !drop {
		pts = append(pts, 0)
	}
	o := newStepOraclePencil(sys)
	tol := math.Sqrt(eps())
	for _, s := range pts {
		got, err := z.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		g, terms := zpkOracleResponseTerms(o, s)
		den := math.Max(math.Max(cmplx.Abs(g), cmplx.Abs(got[0][0])), terms)
		if e := cmplx.Abs(got[0][0]-g) / (tol * den); !(e <= 1) {
			t.Errorf("nil error but Z(%v) is %.3g·√ε of max(|G|, |Z|, terms) off the oracle", s, e)
		}
	}
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
