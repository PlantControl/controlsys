package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// infiniteModePencil is the Lqi augmentation of nearSingularTriangularE
// seed 962 (n = 2, p = 1, D = 0, LU cond(E) < 1/eps) with its integrator
// row of E, A and B scaled by 2^k (AMDL7W). E's (2,2) entry at rounding
// gives a numerically infinite eigenvalue (|λ| ~ 1e15, β ~ eps‖E‖) whose
// λ-space error τ is O(|λ|); [E, B] has full rank, so the mode is
// controllable.
func infiniteModePencil(continuous bool, k int) (E, A, B *mat.Dense) {
	E = mat.NewDense(3, 3, []float64{
		0.7896157989055387, -0.11304980533880604, 0,
		0, 2.1606796378566445e-16, 0,
		0, 0, 1,
	})
	a33 := 1.0
	if continuous {
		a33 = 0
	}
	A = mat.NewDense(3, 3, []float64{
		0.7932593837230228, 0.023753621702102712, 0,
		1.0320517535387468, -0.015623668007243463, 0,
		0.10363962314518527, -1.0955324850541426, a33,
	})
	B = mat.NewDense(3, 1, []float64{-1.3129471670238144, 1.7935007672090852, 0})
	for _, M := range []*mat.Dense{E, A, B} {
		_, c := M.Dims()
		for j := range c {
			M.Set(2, j, math.Ldexp(M.At(2, j), k))
		}
	}
	return E, A, B
}

// TestBoundaryUnstabilizableNumericallyInfiniteMode checks the PBH
// verdict at a numerically infinite mode (AMDL7W): the base threshold
// c·τ·‖E‖ was O(‖E‖) there and exceeded σmin = 0.25 of the integrator
// row, an ErrNoStabilizing on noise.
func TestBoundaryUnstabilizableNumericallyInfiniteMode(t *testing.T) {
	for _, continuous := range []bool{false, true} {
		for _, k := range []int{-2, 2} {
			E, A, B := infiniteModePencil(continuous, k)
			plantE := mat.DenseCopyOf(E.Slice(0, 2, 0, 2))
			problem, err := newRiccatiProblem(A, B, eyeDense(3), eyeDense(1), &RiccatiOpts{E: E, plantE: plantE, augmentedE: true, NoScaling: true})
			if err != nil {
				t.Fatal(err)
			}
			if problem.boundaryUnstabilizable(continuous) {
				t.Errorf("continuous=%v k=%d: controllable numerically infinite mode judged unstabilizable", continuous, k)
			}
		}
	}
}

// TestDlqrNumericallyInfiniteModeCertified checks that Dlqr, with the
// false verdict lifted, returns the 320-bit oracle solution (AMDL7W:
// base ErrNoStabilizing).
func TestDlqrNumericallyInfiniteModeCertified(t *testing.T) {
	E, A, B := infiniteModePencil(false, -2)
	Q, R := eyeDense(3), eyeDense(1)
	res, err := Dlqr(A, B, Q, R, &RiccatiOpts{E: E})
	if err != nil {
		t.Fatalf("Dlqr: %v", err)
	}
	pr := nearSingularRiccati{false, A, B, Q, R, nil, E, false}
	if eX, eK, ok := pr.honest(t, res.X, res.K); !ok {
		t.Errorf("X off oracle by %.3g, K by %.3g", eX, eK)
	}
}

// TestRiccatiNumericallyInfiniteModeNotUnstabilizable checks that Lqr and
// Dlqr (Lqi's augmented E, integrator rows x1/4 and x4) never report
// ErrNoStabilizing for the controllable numerically infinite mode, and
// that any value they return matches the 320-bit oracle (AMDL7W).
func TestRiccatiNumericallyInfiniteModeNotUnstabilizable(t *testing.T) {
	for _, continuous := range []bool{false, true} {
		for _, k := range []int{-2, 2} {
			E, A, B := infiniteModePencil(continuous, k)
			plantE := mat.DenseCopyOf(E.Slice(0, 2, 0, 2))
			Q, R := eyeDense(3), eyeDense(1)
			opts := &RiccatiOpts{E: E, plantE: plantE, augmentedE: true}
			solve := Dlqr
			if continuous {
				solve = Lqr
			}
			res, err := acceptUnverified(solve(A, B, Q, R, opts))
			switch {
			case errors.Is(err, ErrNoStabilizing):
				t.Errorf("continuous=%v k=%d: %v", continuous, k, err)
			case err == nil:
				pr := nearSingularRiccati{continuous, A, B, Q, R, nil, E, false}
				if eX, eK, ok := pr.honest(t, res.X, res.K); !ok {
					t.Errorf("continuous=%v k=%d: X off oracle by %.3g, K by %.3g", continuous, k, eX, eK)
				}
			case !errors.Is(err, ErrNotConverged):
				t.Errorf("continuous=%v k=%d: want value or ErrNotConverged, got %v", continuous, k, err)
			}
		}
	}
}

// TestLqiUnstabilizableIntegratorsNoStabilizing checks that more
// integrators than inputs (p = 2 > m = 1, nearSingularRotatedE seed 166
// with E and A scaled by 1e-4) stay ErrNoStabilizing: the integrator
// modes are exactly uncontrollable.
func TestLqiUnstabilizableIntegratorsNoStabilizing(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{
		3.4247675728633475e-05, 1.379888360407495e-05,
		-6.871411448372405e-05, -2.76859099937471e-05,
	})
	A := mat.NewDense(2, 2, []float64{
		7.855113096671436e-05, -7.877916084747171e-05,
		-0.00011003692772285951, 4.000398765576812e-05,
	})
	B := mat.NewDense(2, 1, []float64{-0.695558688263809, -0.9950214742016862})
	C := mat.NewDense(2, 2, []float64{
		-0.6023737916575802, 0.3761432355635413,
		-0.3480336707911668, -1.4125721510474998,
	})
	D := mat.NewDense(2, 1, nil)
	for _, dt := range []float64{0, 1} {
		sys, err := NewDescriptor(A, B, C, D, E, dt)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Lqi(sys, eyeDense(4), eyeDense(1), nil); !errors.Is(err, ErrNoStabilizing) {
			t.Errorf("dt=%v: want ErrNoStabilizing, got %v", dt, err)
		}
	}
}

// TestDareUncontrollableNumericallyInfiniteModeNoStabilizing checks that
// a numerically infinite mode (|λ| ~ 5e13, σmin(E) ~ 30·eps) that B does
// not reach stays ErrNoStabilizing (AMDL7W): its λ-space τ exceeds the
// gap to the finite mode at 1.03, so it is judged alone in μ = 1/λ,
// where [E, B] is rank deficient to rounding; the cluster mean λ/2 alone
// passed it.
func TestDareUncontrollableNumericallyInfiniteModeNoStabilizing(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{
		0.20192728579879055, -0.5629227329651477,
		0.5981031119894665, -1.6673617785936576,
	})
	A := mat.NewDense(2, 2, []float64{
		0.3200597790806389, -0.3291579827388065,
		0.5126415008448932, -1.9586007850882343,
	})
	B := mat.NewDense(2, 1, []float64{-0.13577711225476594, -0.4021681025190298})
	Q := mat.NewDense(2, 2, []float64{
		2.004964298539094, 2.58564227305348,
		2.58564227305348, 5.139379636728618,
	})
	R := mat.NewDense(1, 1, []float64{1.0218829583403486})
	if _, err := Dare(A, B, Q, R, &RiccatiOpts{E: E}); !errors.Is(err, ErrNoStabilizing) {
		t.Errorf("want ErrNoStabilizing, got %v", err)
	}
}
