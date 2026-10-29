package controlsys

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// boundaryUnreachablePlant is rv46Gen("bnd", 1214) (DBOVA4): row 0 of A
// and E are −0.8499…·e₀ᵀ and 0.8499…·e₀ᵀ, so z = −1 is an exact
// generalized eigenvalue with left eigenvector e₀, and the noise input
// reaches it with weight noise0 (B[0]); it is observable from C.
func boundaryUnreachablePlant(noise0 float64) (A, B, C, E *mat.Dense) {
	A = mat.NewDense(4, 4, []float64{-0.8499358840694509, 0, 0, 0, -0.8094930674224757, 0.5078163484939502, -1.0454272121624435, -0.5638273563054963, -0.009638010908428418, 0.09529955293769937, -1.0330498016568526, -0.5418598159636345, 0.9786642536568926, -1.5795529712310534, -1.2148394351964353, -1.8021820762358332})
	B = mat.NewDense(4, 1, []float64{noise0, -0.2823728743940954, -0.03181740442263309, 0.2745401422425885})
	C = mat.NewDense(1, 4, []float64{-0.4925902745983426, 1.2280922984327654, 0.012899056288373623, -0.16793563809978201})
	E = mat.NewDense(4, 4, []float64{0.8499358840694509, 0, 0, 0, 0, 0.9719114000014576, -1.091011444704252, -1.2622375030831572, 0, 0, 1.275756383997863, -0.3543085871904883, 0, 0, 0, 2.403113001630761e-10})
	return A, B, C, E
}

// dualDare returns Kalman's dual Dare data (Aᵀ, Cᵀ, B·Bᵀ, Eᵀ).
func dualDare(A, B, C, E *mat.Dense) (At, Ct, Q, Et *mat.Dense) {
	n, _ := A.Dims()
	Q = mat.NewDense(n, n, nil)
	Q.Mul(B, B.T())
	return mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), Q, mat.DenseCopyOf(E.T())
}

// wantUncertified reports a result that is not Accuracy 1.
func wantUncertified(t *testing.T, what string, res *RiccatiResult, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("%s: %v, want Accuracy 1", what, err)
	} else if res.Accuracy != 1 {
		t.Errorf("%s: Accuracy %d, Eig %v, want 1", what, res.Accuracy, res.Eig)
	}
}

// TestKalman_BoundaryModeUnreachableFromNoise (DBOVA4): the mode at z = −1
// is exactly unreachable from the process noise, so the filter Riccati
// equation has no stabilizing solution (MATLAB kalman/idare). Base
// returned Accuracy 0 with Eig −0.9999999999999999; the 320-bit closed
// loop of its gain has spectral radius 1 (not Schur stable). The float64
// data cannot tell that mode from one an ulp inside, which has a
// solution, so Kalman keeps the solve's X and K at Accuracy 1.
func TestKalman_BoundaryModeUnreachableFromNoise(t *testing.T) {
	A, B, C, E := boundaryUnreachablePlant(0)
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, ns := range []bool{false, true} {
		res, err := Kalman(sys, eyeDense(1), eyeDense(1), nil, &RiccatiOpts{NoScaling: ns})
		wantUncertified(t, fmt.Sprintf("NoScaling %v", ns), res, err)
	}
}

// TestDare_BoundaryModeUnobservableToCost (DBOVA4): Kalman's dual Dare
// solved directly, (Aᵀ, Cᵀ, B·Bᵀ, I, Eᵀ): the z = −1 mode is unobservable
// to the cost Q = B·Bᵀ, and Dare returns Accuracy 1 as Kalman does.
func TestDare_BoundaryModeUnobservableToCost(t *testing.T) {
	At, Ct, Q, Et := dualDare(boundaryUnreachablePlant(0))
	for _, ns := range []bool{false, true} {
		res, err := Dare(At, Ct, Q, eyeDense(1), &RiccatiOpts{E: Et, NoScaling: ns})
		wantUncertified(t, fmt.Sprintf("NoScaling %v", ns), res, err)
	}
}

// TestDare_WeaklyObservableBoundaryModeKeepsSolution (DBOVA4 control):
// with B[0] = 1e-15 the z = −1 mode is observable to the cost, the
// stabilizing solution exists with its closed loop 3e-15 inside the unit
// circle, within the rounding band that runs the boundary test, and Dare
// keeps the certified solution: K matches the 320-bit reference.
func TestDare_WeaklyObservableBoundaryModeKeepsSolution(t *testing.T) {
	At, Ct, Q, Et := dualDare(boundaryUnreachablePlant(1e-15))
	R := eyeDense(1)
	ref := riccatiExactReference(false, At, Ct, Q, R, nil, Et, 320)
	if !ref.OK {
		t.Fatal("320-bit reference did not converge")
	}
	for _, ns := range []bool{false, true} {
		res, err := Dare(At, Ct, Q, R, &RiccatiOpts{E: Et, NoScaling: ns})
		if err != nil {
			t.Errorf("NoScaling %v: %v", ns, err)
			continue
		}
		if res.Accuracy != 0 {
			t.Errorf("NoScaling %v: Accuracy %d, want 0", ns, res.Accuracy)
		}
		if e := relErr(res.K, ref.K); e > 1e-12 {
			t.Errorf("NoScaling %v: K off the 320-bit gain by %.3g", ns, e)
		}
	}
}

// kalmanCrossTerms returns Kalman's dual cross term S = G·(Qn·Hᵀ + Nn) and
// Rbar = Rn + H·Nn + Nnᵀ·Hᵀ + H·Qn·Hᵀ for scalar Qn = Rn = 1, H and Nn.
func kalmanCrossTerms(G *mat.Dense, h, nn float64) (S, Rbar *mat.Dense) {
	r, _ := G.Dims()
	S = mat.NewDense(r, 1, nil)
	S.Scale(h+nn, G)
	return S, mat.NewDense(1, 1, []float64{1 + 2*h*nn + h*h})
}

var correlatedNoiseCases = []struct {
	name  string
	h, nn float64
}{{"Nn=0.3", 0, 0.3}, {"D=0.5", 0.5, 0}}

// TestKalman_BoundaryModeUnreachableWithCorrelatedNoise (DBOVA4): noise
// feedthrough H or cross-covariance Nn gives the dual Dare a cross term
// S = G·(Qn·Hᵀ + Nn) that inherits G's zero row at the z = −1 mode, which
// stays unreachable, so no stabilizing solution exists. Base returned
// Accuracy 0; the 320-bit closed loop of its gain has spectral radius
// 1.0000000000000002. Kalman and the dual Dare return Accuracy 1.
func TestKalman_BoundaryModeUnreachableWithCorrelatedNoise(t *testing.T) {
	A, B, C, E := boundaryUnreachablePlant(0)
	At, Ct, Q, Et := dualDare(A, B, C, E)
	for _, tc := range correlatedNoiseCases {
		sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, []float64{tc.h}), E, 1)
		if err != nil {
			t.Fatal(err)
		}
		var Nn *mat.Dense
		if tc.nn != 0 {
			Nn = mat.NewDense(1, 1, []float64{tc.nn})
		}
		S, Rbar := kalmanCrossTerms(B, tc.h, tc.nn)
		for _, ns := range []bool{false, true} {
			res, err := Kalman(sys, eyeDense(1), eyeDense(1), Nn, &RiccatiOpts{NoScaling: ns})
			wantUncertified(t, fmt.Sprintf("%s NoScaling %v: Kalman", tc.name, ns), res, err)
			res, err = Dare(At, Ct, Q, Rbar, &RiccatiOpts{E: Et, S: S, NoScaling: ns})
			wantUncertified(t, fmt.Sprintf("%s NoScaling %v: dual Dare", tc.name, ns), res, err)
		}
	}
}

// TestDare_WeaklyObservableBoundaryModeWithCrossTermKeepsSolution (DBOVA4
// control): with B[0] = 1e-15 and noise feedthrough H = 0.5 the cost and
// cross term reach the z = −1 mode, and Dare keeps the certified solution
// of the 320-bit reference. (Nn = 0.3 errs here on base too: the closed
// loop sits within rounding of the boundary; so does D = 0.5 under -race.)
func TestDare_WeaklyObservableBoundaryModeWithCrossTermKeepsSolution(t *testing.T) {
	A, B, C, E := boundaryUnreachablePlant(1e-15)
	At, Ct, Q, Et := dualDare(A, B, C, E)
	for _, tc := range correlatedNoiseCases[1:] {
		S, Rbar := kalmanCrossTerms(B, tc.h, tc.nn)
		ref := riccatiExactReference(false, At, Ct, Q, Rbar, S, Et, 320)
		if !ref.OK {
			t.Fatalf("%s: 320-bit reference did not converge", tc.name)
		}
		for _, ns := range []bool{false, true} {
			res, err := Dare(At, Ct, Q, Rbar, &RiccatiOpts{E: Et, S: S, NoScaling: ns})
			if raceEnabled && errors.Is(err, ErrNoStabilizing) {
				t.Logf("%s NoScaling %v: race build rounding: %v, as on base", tc.name, ns, err)
				continue
			}
			if err != nil {
				t.Errorf("%s NoScaling %v: %v", tc.name, ns, err)
				continue
			}
			if res.Accuracy != 0 {
				t.Errorf("%s NoScaling %v: Accuracy %d, want 0", tc.name, ns, res.Accuracy)
			}
			if e := relErr(res.K, ref.K); e > 1e-12 {
				t.Errorf("%s NoScaling %v: K off the 320-bit gain by %.3g", tc.name, ns, e)
			}
		}
	}
}

// TestCare_InBandBoundaryModeKeepsGainUncertified (DBOVA4, zz54b seed 960
// Care): Kalman's dual Care of a plant whose modes ±1.3977i sit 1e-15
// inside the boundary, reached from the noise with weight 1e-15. The
// stabilizing solution exists and the certified K of base matched the
// 320-bit gain, but the closed loop is within the n·eps·‖A‖_F band and
// the mode fails the cost PBH test, so Care keeps base's K bit for bit at
// Accuracy 1.
func TestCare_InBandBoundaryModeKeepsGainUncertified(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0, 1.397732532312216, -0.30946530991220467, -1.397732532312216, 0, 0.9388100738397125, 0, 0, 0.28905150423430187})
	B := mat.NewDense(3, 1, []float64{-0.21113589611533712, 1.7729228433450344, 2.087470777717446})
	Q := mat.NewDense(3, 3, []float64{8.06276950101559e-31, -5.306222105224478e-31, -1.6258568633996824e-15, -5.306222105224478e-31, 3.4920994611623646e-31, 1.0699992883854174e-15, -1.6258568633996824e-15, 1.0699992883854174e-15, 3.2785391420787713})
	R := eyeDense(1)
	baseK := map[string][]uint64{
		"arm64": {0xbcc60f4a328349a2, 0x3cd2d23ba5ce42e1, 0x3fff4558c34f301d},
		"amd64": {0x3cc643e675e55bb0, 0x3ceba50ae73143fc, 0x3fff4558c34f3018},
	}
	ref := riccatiExactReference(true, A, B, Q, R, nil, nil, 320)
	if !ref.OK {
		t.Fatal("320-bit reference did not converge")
	}
	res, err := Care(A, B, Q, R, &RiccatiOpts{NoScaling: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Accuracy != 1 {
		t.Errorf("Accuracy %d, want 1", res.Accuracy)
	}
	if e := relErr(res.K, ref.K); e > 1e-12 {
		t.Errorf("K off the 320-bit gain by %.3g", e)
	}
	if want, ok := baseK[runtime.GOARCH]; ok && !raceEnabled {
		for i, v := range res.K.RawMatrix().Data {
			if math.Float64bits(v) != want[i] {
				t.Errorf("K[%d] = %#x, want base %#x", i, math.Float64bits(v), want[i])
			}
		}
	}
}
