package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// wantBoundaryProved fails t unless err is ErrNoStabilizing alone (neither
// ErrNotStabilizable, ErrNotDetectable nor ErrNotConverged) with a nil
// result: the exact boundary eigenvalue of boundaryEigenvalueProved.
func wantBoundaryProved(t *testing.T, what string, res *RiccatiResult, err error) {
	t.Helper()
	if !errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) || errors.Is(err, ErrNotDetectable) || errors.Is(err, ErrNotConverged) || res != nil {
		t.Errorf("%s: result %v, err %v; want nil with ErrNoStabilizing alone", what, res != nil, err)
	}
}

// wantDualData fails t unless the dual problem Lqe and Kalman solve is
// exactly (a, b, q, r, s) with descriptor e (nil for none).
func wantDualData(t *testing.T, d estimatorDual, a, b, q, r, s, e []float64) {
	t.Helper()
	same := func(m *mat.Dense, want []float64) bool {
		if m == nil {
			return want == nil
		}
		raw := mat.DenseCopyOf(m).RawMatrix().Data
		if len(raw) != len(want) {
			return false
		}
		for i := range raw {
			if raw[i] != want[i] {
				return false
			}
		}
		return true
	}
	if !same(d.At, a) || !same(d.Ct, b) || !same(d.Qbar, q) || !same(d.Rbar, r) || !same(d.opts.S, s) || !same(d.opts.E, e) {
		t.Fatal("dual problem is not the oracle's data")
	}
}

func denseTransposed(r, c int, a []float64) *mat.Dense {
	return mat.DenseCopyOf(mat.NewDense(r, c, a).T())
}

// TestLqr_HamiltonianAxisPairNoStabilizing (JFLGVW): Lqr of the data of
// TestCare_HamiltonianAxisPairNoStabilizing (non-symmetric A, cross term
// S) solves the same Care, whose exact Hamiltonian has the pair ±iω on
// the imaginary axis: degree 8, p = q(s²), q(t) ≈ t⁴ − 44.79t³ − 45.84t²
// + 0.01487t − 7.421e-18, q(0) < 0 and one Sturm root on (−∞, 0). Lqr
// returned ErrNotConverged; it now returns ErrNoStabilizing alone.
func TestLqr_HamiltonianAxisPairNoStabilizing(t *testing.T) {
	const n, m = 4, 3
	a, b, q, r, s := hamiltonianAxisPair()
	if !hamiltonianAxisRoot(t, n, m, a, b, q, r, s) {
		t.Fatal("oracle: no exact Hamiltonian root on the imaginary axis")
	}
	for _, ns := range []bool{false, true} {
		res, err := Lqr(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r),
			&RiccatiOpts{S: mat.NewDense(n, m, s), NoScaling: ns})
		wantBoundaryProved(t, "Lqr", res, err)
	}
}

// TestDlqr_UnitCirclePairNoStabilizing (JFLGVW): Dlqr of the data of
// TestDare_UnitCirclePairNoStabilizing (cross term S) and of Kalman's dual
// Dare of TestDare_BoundaryModeUnobservableToCost (descriptor E, the
// z = −1 mode unobservable to the cost: p(−1) = 0 exactly). Both exact
// symplectic pencils have a root on the unit circle, so no stabilizing
// solution exists. Dlqr returned ErrNotConverged; it now returns
// ErrNoStabilizing alone.
func TestDlqr_UnitCirclePairNoStabilizing(t *testing.T) {
	const n, m = 4, 2
	a, b, q, r, s := dareUnitCirclePair()
	if !symplecticUnitCircleRoot(t, n, m, a, b, q, r, s, nil) {
		t.Fatal("oracle: no exact symplectic root on the unit circle")
	}
	At, Ct, Q, Et := dualDare(boundaryUnreachablePlant(0))
	raw := func(m *mat.Dense) []float64 { return mat.DenseCopyOf(m).RawMatrix().Data }
	if !symplecticUnitCircleRoot(t, 4, 1, raw(At), raw(Ct), raw(Q), []float64{1}, nil, raw(Et)) {
		t.Fatal("oracle: descriptor: no exact symplectic root on the unit circle")
	}
	for _, ns := range []bool{false, true} {
		res, err := Dlqr(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r),
			&RiccatiOpts{S: mat.NewDense(n, m, s), NoScaling: ns})
		wantBoundaryProved(t, "Dlqr", res, err)
		res, err = Dlqr(At, Ct, Q, eyeDense(1), &RiccatiOpts{E: Et, NoScaling: ns})
		wantBoundaryProved(t, "Dlqr descriptor", res, err)
	}
}

// TestLqe_DualHamiltonianAxisPairNoStabilizing (JFLGVW): Lqe of
// (Aᵀ, G = I, Bᵀ, Qn = Q, Rn = R, Nn = S) for the data of
// TestCare_HamiltonianAxisPairNoStabilizing solves the dual Care of
// (A, B, G·Qn·Gᵀ = Q, R) with cross term G·Nn = S, bit for bit that data,
// whose exact Hamiltonian has the pair ±iω on the imaginary axis (proof
// as in TestLqr_HamiltonianAxisPairNoStabilizing). Lqe returned
// ErrNotConverged; it now returns ErrNoStabilizing alone, not wrapped with
// ErrNotDetectable.
func TestLqe_DualHamiltonianAxisPairNoStabilizing(t *testing.T) {
	const n, m = 4, 3
	a, b, q, r, s := hamiltonianAxisPair()
	if !hamiltonianAxisRoot(t, n, m, a, b, q, r, s) {
		t.Fatal("oracle: no exact Hamiltonian root on the imaginary axis")
	}
	At, Bt, Q, R, S := denseTransposed(n, n, a), denseTransposed(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r), mat.NewDense(n, m, s)
	wantDualData(t, newEstimatorDual(At, nil, eyeDense(n), Bt, nil, Q, R, S, nil), a, b, q, r, s, nil)
	for _, ns := range []bool{false, true} {
		res, err := Lqe(At, eyeDense(n), Bt, Q, R, S, &RiccatiOpts{NoScaling: ns})
		wantBoundaryProved(t, "Lqe", res, err)
	}
}

// TestKalman_DualBoundaryEigenvalueNoStabilizing (JFLGVW): Kalman of
// continuous and discrete plants (Aᵀ, G = I, Bᵀ, H = 0) with Qn = Q,
// Rn = R and Nn = S solves the dual Care (Dare) of the data of
// TestCare_HamiltonianAxisPairNoStabilizing
// (TestDare_UnitCirclePairNoStabilizing), bit for bit, whose exact pencil
// has a root on the imaginary axis (unit circle). Kalman returned
// ErrNotConverged; it now returns ErrNoStabilizing alone. The descriptor
// plant with an unreachable z = −1 mode is
// TestKalman_BoundaryModeUnreachableFromNoise.
func TestKalman_DualBoundaryEigenvalueNoStabilizing(t *testing.T) {
	for _, c := range []struct {
		name       string
		continuous bool
		n, m       int
		data       func() (a, b, q, r, s []float64)
	}{
		{"continuous", true, 4, 3, hamiltonianAxisPair},
		{"discrete", false, 4, 2, dareUnitCirclePair},
	} {
		n, m := c.n, c.m
		a, b, q, r, s := c.data()
		if c.continuous && !hamiltonianAxisRoot(t, n, m, a, b, q, r, s) ||
			!c.continuous && !symplecticUnitCircleRoot(t, n, m, a, b, q, r, s, nil) {
			t.Fatalf("%s: oracle: no exact root on the boundary", c.name)
		}
		dt := 0.0
		if !c.continuous {
			dt = 1
		}
		sys, err := New(denseTransposed(n, n, a), eyeDense(n), denseTransposed(n, m, b), mat.NewDense(m, n, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		Q, R, S := mat.NewDense(n, n, q), mat.NewDense(m, m, r), mat.NewDense(n, m, s)
		wantDualData(t, newEstimatorDual(sys.A, sys.E, sys.B, sys.C, sys.D, Q, R, S, nil), a, b, q, r, s, nil)
		for _, ns := range []bool{false, true} {
			res, err := Kalman(sys, Q, R, S, &RiccatiOpts{NoScaling: ns})
			wantBoundaryProved(t, "Kalman "+c.name, res, err)
		}
	}
}
