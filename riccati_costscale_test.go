package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// costScaleCase is the lightly coupled plant of finding ML2GAC: ‖A‖ ≈ 1e3
// dwarfs X when the cost is small.
func costScaleCase() (A, B, Q, R, S, E *mat.Dense) {
	A = mat.NewDense(2, 2, []float64{-300, 900, -900, -300})
	B = mat.NewDense(2, 1, []float64{0.45, -0.74})
	Q = mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 0.5})
	R = mat.NewDense(1, 1, []float64{1})
	S = mat.NewDense(2, 1, []float64{0.1, -0.3})
	E = mat.NewDense(2, 2, []float64{1, 0.3, -0.2, 2})
	return
}

func scaled(c float64, m *mat.Dense) *mat.Dense {
	if m == nil {
		return nil
	}
	var d mat.Dense
	d.Scale(c, m)
	return &d
}

// TestRiccati_CostScaleInvariance: (cQ, cR, cS) has the solution cX and the
// same K, so X keeps its relative accuracy for any c (the Schur method was
// accurate only to eps·‖H‖ absolute: 1.6e-6 relative at c = 1e-8).
func TestRiccati_CostScaleInvariance(t *testing.T) {
	A, B, Q, R, S, E := costScaleCase()
	Ad := eulerStep(A, 1e-4)
	for _, tc := range []struct {
		name  string
		solve func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
		a     *mat.Dense
		s, e  bool
	}{
		{"Care", Care, A, false, false},
		{"Care/S", Care, A, true, false},
		{"Care/E", Care, A, false, true},
		{"Care/S/E", Care, A, true, true},
		{"Dare", Dare, Ad, false, false},
		{"Dare/S", Dare, Ad, true, false},
		{"Dare/E", Dare, Ad, false, true},
	} {
		opts := func(c float64) *RiccatiOpts {
			o := &RiccatiOpts{}
			if tc.s {
				o.S = scaled(c, S)
			}
			if tc.e {
				o.E = E
			}
			return o
		}
		ref, err := tc.solve(tc.a, B, Q, R, opts(1))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for _, c := range []float64{1e-12, 1e-8, 1e-4, 1e4, 1e8, 1e12} {
			t.Run(fmt.Sprintf("%s/c=%g", tc.name, c), func(t *testing.T) {
				got, err := tc.solve(tc.a, B, scaled(c, Q), scaled(c, R), opts(c))
				if err != nil {
					t.Fatal(err)
				}
				if e := relErr(got.X, scaled(c, ref.X)); !(e <= 1e-12) {
					t.Errorf("X rel err %g", e)
				}
				if e := relErr(got.K, ref.K); !(e <= 1e-12) {
					t.Errorf("K rel err %g", e)
				}
				assertEigSetNear(t, tc.name, got.Eig, ref.Eig, 1e-9)
			})
		}
	}
}

// newtonCare refines the stabilizing Care solution by Newton–Kleinman from
// the stabilizing gain K0 (Lyapunov solves only).
func newtonCare(t *testing.T, A, B, Q, R, K0 *mat.Dense) *mat.Dense {
	t.Helper()
	K := mat.DenseCopyOf(K0)
	var Rinv mat.Dense
	if err := Rinv.Inverse(R); err != nil {
		t.Fatal(err)
	}
	var X *mat.Dense
	for range 30 {
		var bk, acl, rhs, krk mat.Dense
		bk.Mul(B, K)
		acl.Sub(A, &bk)
		krk.Product(K.T(), R, K)
		rhs.Add(Q, &krk)
		var err error
		if X, err = Lyap(mat.DenseCopyOf(acl.T()), &rhs, nil); err != nil {
			t.Fatal(err)
		}
		K.Product(&Rinv, B.T(), X)
	}
	return X
}

// TestCare_TinyCostAgainstNewton is repro 2 of ML2GAC: Q = CᵀC ≈ 1e-12
// against ‖A‖ ≈ 1e4. The unscaled Schur method returned an indefinite X
// with relative error 1.19; Newton–Kleinman gives the PSD solution.
func TestCare_TinyCostAgainstNewton(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3162.2776601683786, 9486.832980505138, -9486.832980505138, -3162.2776601683786})
	B := mat.NewDense(2, 1, []float64{0.4543874744042692, -0.7367935074839924})
	C := mat.NewDense(1, 2, []float64{9.555985806128393e-07, -5.87782813560387e-07})
	Q, R := gram(C), mat.NewDense(1, 1, []float64{1e-3})
	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := newtonCare(t, A, B, Q, R, mat.NewDense(1, 2, nil))
	if e := relErr(res.X, want); !(e <= 1e-10) {
		t.Errorf("X rel err %g\nX=%v\nwant=%v", e, mat.Formatted(res.X), mat.Formatted(want))
	}
	var eig mat.EigenSym
	if !eig.Factorize(mat.NewSymDense(2, mat.DenseCopyOf(res.X).RawMatrix().Data), false) {
		t.Fatal("EigenSym failed")
	}
	if v := eig.Values(nil); !(v[0] > 0) {
		t.Errorf("X not positive definite: eig %v", v)
	}
	if math.Abs(want.At(0, 0)-1.30637e-16) > 1e-21 {
		t.Errorf("oracle X11 = %g, want 1.30637e-16", want.At(0, 0))
	}
}

// TestCare_UnstableTinyCostKeepsAccuracy: X → the Q = 0 stabilizing
// solution, not ∝ Q, so cost scaling by √(‖Q‖/‖G‖) (SB02MD) would lose 8
// digits; Care keeps the residual at rounding.
func TestCare_UnstableTinyCostKeepsAccuracy(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 2, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	for _, q := range []float64{1e-14, 1e-30} {
		Q := mat.NewDense(2, 2, []float64{q, 0, 0, q})
		res, err := Care(A, B, Q, R, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := newtonCare(t, A, B, Q, R, res.K)
		if e := relErr(res.X, want); !(e <= 1e-13) {
			t.Errorf("q=%g: X rel err %g", q, e)
		}
	}
}

// TestCare_TinyQbarCrossTerm is corpus qbar_cancel_care: Q ≈ 1e-12 with a
// cross term S, ‖X‖ ≈ 5e-15 against ‖A‖ ≈ 10. Without cost scaling Care
// returned an indefinite X (relative error 1). Oracle: Newton with the
// residual in 300-bit big.Float from the 1-ulp-perturbed data (X moves
// 3.3e-9 relative).
func TestCare_TinyQbarCrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3.8749574161655898, 10.471285286135096, -1.5287087138634041, 3.87295741616559})
	B := mat.NewDense(2, 3, []float64{-0.46225445740506277, 0.9988159580766447, -0.5462422338892708, 0.7307801165481209, 0.22030638553843548, -0.9532217576632307})
	Q := mat.NewDense(2, 2, []float64{9.18088552826618e-13, -5.563120977722391e-13, -5.563120977722391e-13, 3.370950973900178e-13})
	R := mat.NewDense(3, 3, []float64{2601.807645907061, 851.7699414343014, -1032.7425080685794, 851.7699414343014, 278.8503273596277, -338.0954641645226, -1032.7425080685794, -338.0954641645226, 409.93041948610824})
	S := mat.NewDense(2, 3, []float64{-4.887421506556899e-05, -1.6000261729492946e-05, 1.939978111982341e-05, 2.9615135736512105e-05, 9.695294795076357e-06, -1.1755220014304417e-05})
	want := mat.NewDense(2, 2, []float64{6.75376352187457e-16, -1.7118468075867222e-15, -1.7118468075867222e-15, 4.6282938821906125e-15})
	res, err := Care(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(res.X, want); !(e <= 3e-8) {
		t.Errorf("X rel err %g\nX=%v", e, mat.Formatted(res.X))
	}
}

// TestDare_CostScaleKeepsBetterPass: R = DᵀD + 1e-3·I (cond 1e8) with
// ‖X‖ ≈ 2e-12 makes the rescale estimate fire, but the balanced pencil is
// less accurate here (2.5e-9); Dare keeps the pass with smaller residual.
// Oracle: scalar Dare in exact rationals, X = a²X/(1+βX) + q, β = BR⁻¹Bᵀ.
func TestDare_CostScaleKeepsBetterPass(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{-157.9967631854184, -135.69600507366528, -91.97183144532306, 159.4467355237148, 152.03721471336473, 120.62436069492874})
	C := mat.NewDense(2, 1, []float64{-8.568887533689471e-07, -9.87479769908865e-07})
	var R mat.Dense
	R.Mul(D.T(), D)
	for i := range 3 {
		R.Set(i, i, R.At(i, i)+1e-3)
	}
	A := mat.NewDense(1, 1, []float64{0.36787944117144233})
	B := mat.NewDense(1, 3, []float64{-0.9667981925794609, 0.70866977429126, 0.25125984258225487})
	res, err := Dare(A, B, gram(C), &R, nil)
	if err != nil {
		t.Fatal(err)
	}
	const want = 1.976921919084928425e-12
	if e := math.Abs(res.X.At(0, 0)-want) / want; !(e <= 1e-10) {
		t.Errorf("X = %.17g, rel err %g", res.X.At(0, 0), e)
	}
}
