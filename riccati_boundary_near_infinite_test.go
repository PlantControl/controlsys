package controlsys

import (
	"errors"
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorDareDoublingBig returns the stabilizing solution of the Dare
// with E by SDA doubling (dareDoublingHB) on the explicit model E⁻¹A, E⁻¹B
// formed in 320 bits: the descriptor X = E⁻ᵀYE⁻¹, Y = EᵀXE, K and the
// spectral radius of E⁻¹(A − BK). ok reports convergence to a positive
// definite Y with a stable closed loop.
func descriptorDareDoublingBig(A, B, Q, R, E *mat.Dense) (X, Y, K *mat.Dense, rho float64, ok bool) {
	e := hbFrom(E)
	a, b, r := hbSolve(e, hbFrom(A)), hbSolve(e, hbFrom(B)), hbFrom(R)
	y, conv := dareDoublingHB(a, b, hbFrom(Q), r, hbMat{}, 120)
	if !conv {
		return nil, nil, nil, 0, false
	}
	btx := hbMul(b.t(), y)
	k := hbSolve(hbAdd(r, hbMul(btx, b), 1), hbMul(btx, a))
	var ed mat.Eigen
	if !ed.Factorize(hbDense(hbAdd(a, hbMul(b, k), -1)), mat.EigenNone) {
		return nil, nil, nil, 0, false
	}
	for _, v := range ed.Values(nil) {
		rho = max(rho, cmplx.Abs(v))
	}
	z := hbSolve(e.t(), y)
	X = hbDense(hbSolve(e.t(), z.t()).t())
	symmetrize(X.RawMatrix().Data, X.RawMatrix().Rows, X.RawMatrix().Cols)
	return X, hbDense(y), hbDense(k), rho, rho < 1 && hbPositiveDefinite(y)
}

// assertBoundaryVerdict checks boundaryUnstabilizable of (A, B, E)
// unscaled and state-scaled as the solvers scale, in both scaling modes.
func assertBoundaryVerdict(t *testing.T, label string, continuous bool, A, B, Q, R, E *mat.Dense, want bool) {
	t.Helper()
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		problem, err := newRiccatiProblem(A, B, Q, R, opts)
		if err != nil {
			t.Fatalf("%s NoScaling %v: %v", label, ns, err)
		}
		if got := problem.boundaryUnstabilizable(continuous); got != want {
			t.Errorf("%s NoScaling %v: boundaryUnstabilizable %v, want %v", label, ns, got, want)
		}
		problem.scale(opts, continuous)
		if got := problem.boundaryUnstabilizable(continuous); got != want {
			t.Errorf("%s NoScaling %v: boundaryUnstabilizable %v once scaled, want %v", label, ns, got, want)
		}
	}
}

// A near-singular E has a finite eigenvalue λ ≈ 1/σ_min(E) of (A, E),
// unstable in both time domains, that B reaches. The PBH test of
// [A − λE, B] there allowed τ‖E‖ of the band for λ's error τ, which
// at |λ| ≈ 4e14 is about 5 against σ_min ≈ 2.6, so Dlqr and Dare reported
// an unstabilizable boundary mode (IR5YOF, about 95% of the discrete rows
// of the wave-40 near-singular-E sweep). The same test of the reversed
// pencil [λ⁻¹A − E, B], rank-equivalent, has σ_min 0.76 against a band of
// 2e-15. The 320-bit doubling solution of the explicit model formed in
// 320 bits stabilizes (ρ ≈ 0.61), so no boundary verdict may stand, and a
// returned value must be that solution or carry Accuracy 1. Here the
// pencil X, ‖X‖ ≈ 4e29, is off by 7% and stays ErrNoStabilizing.
func TestDare_NearInfiniteControllableModeIsNotBoundaryUnstabilizable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0.3431284077914542, -0.7926130901776904})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, -2.5367599785728485, -0.7294612311954138})
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	Q, R := eyeDense(2), eyeDense(2)
	wantX, wantY, wantK, rho, ok := descriptorDareDoublingBig(A, B, Q, R, E)
	if !ok {
		t.Fatalf("doubling oracle: no stabilizing solution (ρ %.3g)", rho)
	}
	assertBoundaryVerdict(t, "Dare", false, A, B, Q, R, E, false)
	checkNearInfiniteResult(t, "Dare", A, B, Q, R, E, wantX, wantK)
	checkNearInfiniteResult(t, "Dlqr", A, B, Q, R, E, wantY, wantK)
}

// checkNearInfiniteResult runs Dare (descriptor X) or Dlqr (EᵀXE) in both
// scaling modes and fails on a boundary verdict or on an Accuracy 0
// result off the oracle by more than newtonTol.
func checkNearInfiniteResult(t *testing.T, api string, A, B, Q, R, E, wantX, wantK *mat.Dense) {
	t.Helper()
	for _, ns := range []bool{false, true} {
		label := fmt.Sprintf("%s NoScaling %v", api, ns)
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		var res *RiccatiResult
		var err error
		if api == "Dare" {
			res, err = Dare(A, B, Q, R, opts)
		} else {
			res, err = Dlqr(A, B, Q, R, opts)
		}
		if errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("%s: %v", label, err)
			continue
		}
		if err != nil || res.Accuracy != 0 {
			continue
		}
		if rel := relErr(res.X, wantX); !(rel <= newtonTol) {
			t.Errorf("%s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", label, rel)
		}
		if rel := relErr(res.K, wantK); !(rel <= newtonTol) {
			t.Errorf("%s: Accuracy 0 with ‖K − K*‖/‖K*‖ = %.3g", label, rel)
		}
	}
}

// A stabilizable plant whose near-infinite mode (σ_min(E) = 1.8e-13) was
// the IR5YOF boundary verdict: Dlqr and Dare now certify the 320-bit
// doubling solution (seed 1 of the wave-40 triangular discrete sweep).
func TestDlqr_NearSingularEStabilizableIsCertified(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{0.8110305471830912, -2.798084983771413, -0.017491933522454567, -0.39791666421793337, -0.8090820362520077, -0.5499181819116858, -0.46262154101836206, -0.8340902193720336, -0.9883516861538988, -2.614357975324414, 0.513330110402648, 2.049045874234565, -0.08641878060699576, -0.5775565235201454, -0.4432567713689527, 0.7056482628882423, -0.597668402517777, 0.7407429714810015, -0.3708537462269688, -0.9877061648126102, 0.6044085481070023, 0.71649527299946, -1.8055876037300738, -1.6494675577286064, -0.016188655813992314})
	B := mat.NewDense(5, 1, []float64{2.0382094016789454, -0.08179111491178975, 1.1355811750839162, 0.4653174896812371, 0.6252205456119206})
	E := mat.NewDense(5, 5, []float64{0.728116116855907, 0.1576742650568862, 0.9679861848999214, -0.28435313557425956, -0.5870769808340399, 0, 1.1384134609187906, 0.3285791757945944, -0.8038111554228949, 0.6030483366743167, 0, 0, 0.8753709080953666, 0.0081188653096545, 0.4543347874916437, 0, 0, 0, 0.8612000048726609, -0.674701915132129, 0, 0, 0, 0, 1.801470805444606e-13})
	Q, R := eyeDense(5), eyeDense(1)
	wantX, wantY, wantK, rho, ok := descriptorDareDoublingBig(A, B, Q, R, E)
	if !ok {
		t.Fatalf("doubling oracle: no stabilizing solution (ρ %.3g)", rho)
	}
	assertBoundaryVerdict(t, "Dare", false, A, B, Q, R, E, false)
	for _, api := range []string{"Dare", "Dlqr"} {
		want := wantX
		if api == "Dlqr" {
			want = wantY
		}
		var res *RiccatiResult
		var err error
		if api == "Dare" {
			res, err = Dare(A, B, Q, R, &RiccatiOpts{E: E})
		} else {
			res, err = Dlqr(A, B, Q, R, &RiccatiOpts{E: E})
		}
		if err != nil {
			t.Errorf("%s: %v", api, err)
			continue
		}
		if res.Accuracy != 0 {
			t.Errorf("%s: Accuracy %d, want 0", api, res.Accuracy)
		}
		if rel := relErr(res.X, want); !(rel <= newtonTol) {
			t.Errorf("%s: ‖X − X*‖/‖X*‖ = %.3g", api, rel)
		}
		if rel := relErr(res.K, wantK); !(rel <= newtonTol) {
			t.Errorf("%s: ‖K − K*‖/‖K*‖ = %.3g", api, rel)
		}
	}
	checkNearInfiniteResult(t, "Dare", A, B, Q, R, E, wantX, wantK)
}

// The continuous counterpart: λ ≈ +1/σ_min(E) is unstable and B reaches
// it. Care and Lqr must not report an unstabilizable boundary mode, and
// an Accuracy 0 result must be the 320-bit Newton solution of the
// explicit model (seed 9 of the wave-40 rotated continuous sweep, an
// Accuracy 1 X here).
func TestCare_NearInfiniteControllableModeIsNotBoundaryUnstabilizable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5407819029319603, -0.1852056177244038, -1.6242211188762974, -0.2850790020671532})
	B := mat.NewDense(2, 1, []float64{-1.7426999725998762, -0.738241857778609})
	E := mat.NewDense(2, 2, []float64{0.18344000710370426, 0.33396687232703354, 0.5510801694129795, 1.0032845260245629})
	Q, R := eyeDense(2), eyeDense(1)
	assertBoundaryVerdict(t, "Care", true, A, B, Q, R, E, false)
	for _, api := range []string{"Care", "Lqr"} {
		for _, ns := range []bool{false, true} {
			label := fmt.Sprintf("%s NoScaling %v", api, ns)
			opts := &RiccatiOpts{E: E, NoScaling: ns}
			var res *RiccatiResult
			var err error
			if api == "Care" {
				res, err = Care(A, B, Q, R, opts)
			} else {
				res, err = Lqr(A, B, Q, R, opts)
			}
			if errors.Is(err, errBoundaryUnstabilizable) {
				t.Errorf("%s: %v", label, err)
				continue
			}
			if err != nil || res.Accuracy != 0 {
				continue
			}
			Y, X, K, conv := nearSingularOracle(true, A, B, Q, R, nil, E, res.K)
			if !conv {
				t.Errorf("%s: Accuracy 0 K from which Newton does not converge", label)
				continue
			}
			want := X
			if api == "Lqr" {
				want = Y
			}
			if rel := relErr(res.X, want); !(rel <= newtonTol) {
				t.Errorf("%s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", label, rel)
			}
			if rel := relErr(res.K, K); !(rel <= newtonTol) {
				t.Errorf("%s: Accuracy 0 with ‖K − K*‖/‖K*‖ = %.3g", label, rel)
			}
		}
	}
}

// The reversed PBH test must not accept a near-infinite mode B cannot
// reach: with A and E upper triangular and B's last row zero, the mode
// a₂₂/e₂₂ ≈ ∓4.5e14 is exactly uncontrollable and unstable, so Dare,
// Dlqr, Care and Lqr report it.
func TestRiccati_NearInfiniteUncontrollableModeIsUnstabilizable(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, 0, 0})
	Q, R := eyeDense(2), eyeDense(2)
	for _, continuous := range []bool{false, true} {
		a22 := -0.7926130901776904
		if continuous {
			a22 = -a22
		}
		A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0, a22})
		label := map[bool]string{false: "Dare", true: "Care"}[continuous]
		assertBoundaryVerdict(t, label, continuous, A, B, Q, R, E, true)
		for _, ns := range []bool{false, true} {
			opts := &RiccatiOpts{E: E, NoScaling: ns}
			var err1, err2 error
			if continuous {
				_, err1 = Care(A, B, Q, R, opts)
				_, err2 = Lqr(A, B, Q, R, opts)
			} else {
				_, err1 = Dare(A, B, Q, R, opts)
				_, err2 = Dlqr(A, B, Q, R, opts)
			}
			for i, err := range []error{err1, err2} {
				if !errors.Is(err, errBoundaryUnstabilizable) {
					t.Errorf("%s (%d) NoScaling %v: %v, want an unstabilizable boundary mode", label, i, ns, err)
				}
			}
		}
	}
}
