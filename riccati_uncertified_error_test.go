package controlsys

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// acceptUnverified is the view of an exported Riccati result that tests of
// its value take (ZOJBZX): an ErrAccuracyUnverified partial with a nil
// error, its accuracy 1 readable. It panics on any result that breaks the
// carrier contract, so every such test also checks it.
func acceptUnverified(res *RiccatiResult, err error) (*RiccatiResult, error) {
	if msg := riccatiContractBreach(res, err); msg != "" {
		panic(msg)
	}
	if errors.Is(err, ErrAccuracyUnverified) {
		return res, nil
	}
	return res, err
}

// riccatiContractBreach describes how res, err of an exported Riccati
// operation break the carrier contract, or is empty: res non-nil iff err
// is nil (accuracy 0) or ErrAccuracyUnverified (accuracy 1), and
// ErrNoStabilizing never with ErrNotConverged.
func riccatiContractBreach(res *RiccatiResult, err error) string {
	unverified := errors.Is(err, ErrAccuracyUnverified)
	switch {
	case err == nil && res == nil:
		return "nil result with nil error"
	case err == nil && res.accuracy != 0:
		return fmt.Sprintf("nil error with accuracy %d", res.accuracy)
	case unverified && (res == nil || res.accuracy == 0):
		return fmt.Sprintf("ErrAccuracyUnverified without an uncertified result: %v", err)
	case err != nil && !unverified && res != nil:
		return fmt.Sprintf("result with error %v", err)
	case errors.Is(err, ErrNoStabilizing) && errors.Is(err, ErrNotConverged):
		return fmt.Sprintf("ErrNoStabilizing with ErrNotConverged: %v", err)
	case errors.Is(err, errNoCandidate) || errors.Is(err, errSingularBasis):
		return fmt.Sprintf("unexported sentinel escapes: %v", err)
	}
	return ""
}

// riccatiNoSolution reports an exported Riccati error that was
// ErrNoStabilizing before ZOJBZX: the proved boundary verdict, a
// non-convergence other than ErrSchurFailed, or a singular stable basis.
// Tests that accepted any such failure accept it.
func riccatiNoSolution(err error) bool {
	return errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrSingularEquation) ||
		errors.Is(err, ErrNotConverged) && !errors.Is(err, ErrSchurFailed)
}

// TestRiccatiUnverifiedPartialResult: an X no Newton step certifies
// (MATLAB icare/idare Report 1) comes back from every exported entry
// point, non-nil and bit for bit the core's (descriptor Eig as the exits
// recompute it, descriptorClosedLoopEig), with ErrAccuracyUnverified
// (ZOJBZX): Care's weakBStall and its duals, and Dare's well-conditioned
// E with a far mode (TestDare_WellConditionedEFarModeStaysUncertified).
func TestRiccatiUnverifiedPartialResult(t *testing.T) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	n, m := B.Dims()
	core, err := careCore(A, B, Q, R, nil)
	if err != nil || core.accuracy != 1 {
		t.Fatalf("Care core: err %v, want an uncertified X", err)
	}
	At, Bt := mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T())
	sys, err := New(At, eyeDense(n), Bt, mat.NewDense(m, n, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	dA := mat.NewDense(2, 2, []float64{-456.1125380107185, 212.12282671008353, -6542.374939554918, 3040.8017276406886})
	dB := mat.NewDense(2, 2, []float64{0.1511888949722682, 1.360456698990255, -0.010540390965378899, -0.09484655257220741})
	dE := mat.NewDense(2, 2, []float64{0.35739616083221576, 0.9339528811574985, -0.9339528811574985, 0.357396160832216})
	dOpts := &RiccatiOpts{E: dE}
	dCore, err := dareFor(dA, dB, eyeDense(2), eyeDense(2), dOpts, false)
	if err != nil || dCore.accuracy != 1 {
		t.Fatalf("Dare core: err %v, want an uncertified X", err)
	}
	descriptorClosedLoopEig(false, dA, dB, dOpts, dCore, nil)
	dlqrCore, err := dareFor(dA, dB, eyeDense(2), eyeDense(2), dOpts, true)
	if err == nil {
		dlqrCore, err = explicitRiccatiSolution(dlqrCore, dOpts, nil)
	}
	if err != nil || dlqrCore.accuracy != 1 {
		t.Fatalf("Dlqr core: err %v, want an uncertified X", err)
	}
	descriptorClosedLoopEig(false, dA, dB, dOpts, dlqrCore, nil)
	for _, c := range []struct {
		op      string
		res     func() (*RiccatiResult, error)
		X, K    *mat.Dense
		eig     []complex128
		rcnd    float64
		transpK bool
	}{
		{"Care", func() (*RiccatiResult, error) { return Care(A, B, Q, R, nil) }, core.X, core.K, core.Eig, core.Rcnd, false},
		{"Lqr", func() (*RiccatiResult, error) { return Lqr(A, B, Q, R, nil) }, core.X, core.K, core.Eig, core.Rcnd, false},
		{"Lqe", func() (*RiccatiResult, error) { return Lqe(At, eyeDense(n), Bt, Q, R, nil, nil) }, core.X, core.K, core.Eig, core.Rcnd, true},
		{"Kalman", func() (*RiccatiResult, error) { return Kalman(sys, Q, R, nil, nil) }, core.X, core.K, core.Eig, core.Rcnd, true},
		{"Dare", func() (*RiccatiResult, error) { return Dare(dA, dB, eyeDense(2), eyeDense(2), dOpts) }, dCore.X, dCore.K, dCore.Eig, dCore.Rcnd, false},
		{"Dlqr", func() (*RiccatiResult, error) { return Dlqr(dA, dB, eyeDense(2), eyeDense(2), dOpts) }, dlqrCore.X, dlqrCore.K, dlqrCore.Eig, dlqrCore.Rcnd, false},
	} {
		res, err := c.res()
		if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
			t.Errorf("%s: result %v, err %v; want a partial result with ErrAccuracyUnverified", c.op, res != nil, err)
			continue
		}
		if msg := riccatiContractBreach(res, err); msg != "" {
			t.Errorf("%s: %s", c.op, msg)
		}
		K := c.K
		if c.transpK {
			K = mat.DenseCopyOf(K.T())
		}
		if !bitEqualDense(res.X, c.X) || !bitEqualDense(res.K, K) || !slices.Equal(res.Eig, c.eig) || math.Float64bits(res.Rcnd) != math.Float64bits(c.rcnd) {
			t.Errorf("%s: partial result differs from the core's", c.op)
		}
	}
}

// TestRiccatiProvedNoStabilizing: the PBH boundary verdict is the one
// Riccati error that ErrNoStabilizing reports, wrapped with
// ErrNotStabilizable for Dare and Dlqr and with ErrNotDetectable for
// Kalman, never with ErrNotConverged (ZOJBZX Q2, Q3): an exactly
// uncontrollable unstable near-infinite mode
// (TestDare_NearSingularEUncontrollableModeIsUnstabilizable) and an
// undetectable mode at z = 1 (TestKalman_BoundaryUndetectableNoStabilizing).
func TestRiccatiProvedNoStabilizing(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, 0, 0})
	A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0, -0.7926130901776904})
	check := func(op string, res *RiccatiResult, err, want, not error) {
		t.Helper()
		if !errors.Is(err, ErrNoStabilizing) || !errors.Is(err, want) || errors.Is(err, not) || errors.Is(err, ErrNotConverged) || res != nil {
			t.Errorf("%s: result %v, err %v; want nil with ErrNoStabilizing and %v", op, res != nil, err, want)
		}
		if msg := riccatiContractBreach(res, err); msg != "" {
			t.Errorf("%s: %s", op, msg)
		}
	}
	opts := &RiccatiOpts{E: E}
	res, err := Dare(A, B, eyeDense(2), eyeDense(2), opts)
	check("Dare", res, err, ErrNotStabilizable, ErrNotDetectable)
	res, err = Dlqr(A, B, eyeDense(2), eyeDense(2), opts)
	check("Dlqr", res, err, ErrNotStabilizable, ErrNotDetectable)
	s := syKalmanSpec(fzDecode(t, 0xa7, [fzNVals]float64{-0.3010299956639812, 45.6, -68, 1, 0, -6, -30, 156, 1.2, 23.7824, 135.4, 48, -10, -2, 47, -1.1111}))
	qn, rn, nn := syNoise(s, s.M, s.P)
	var nnD *mat.Dense
	if nn != nil {
		nnD = mqDense(s.M, s.P, nn)
	}
	res, err = Kalman(s.System(t, fzSysOpts{}), mqDense(s.M, s.M, qn), mqDense(s.P, s.P, rn), nnD, nil)
	check("Kalman", res, err, ErrNotDetectable, ErrNotStabilizable)
}

// TestRiccatiNotConvergedIsNotNoStabilizing: a solve that finds no
// verified solution is ErrNotConverged with a nil result, not
// ErrNoStabilizing, where no certificate proves that none exists (ZOJBZX
// Q1): the census near-unit-circle Dare, whose pencil has no eigenvalue on
// the circle (TestDare_NearUnitCircleNotProved), for Dare, Dlqr and
// Kalman on its dual (descriptor Eᵀ), and the Hamiltonian axis pair with
// Q one ulp off symmetric (TestCare_AxisPairInexactSymmetryNotProved) for
// Lqr and Lqe on its dual. The boundary mode unreachable from the noise
// (TestKalman_BoundaryModeUnreachableFromNoise), not proved before,
// is proved since I4J6DP (Dare) and JFLGVW (Dlqr, Kalman).
func TestRiccatiNotConvergedIsNotNoStabilizing(t *testing.T) {
	check := func(op string, res *RiccatiResult, err error) {
		t.Helper()
		if !errors.Is(err, ErrNotConverged) || errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) || errors.Is(err, ErrNotDetectable) || res != nil {
			t.Errorf("%s: result %v, err %v; want nil with ErrNotConverged alone", op, res != nil, err)
		}
		if msg := riccatiContractBreach(res, err); msg != "" {
			t.Errorf("%s: %s", op, msg)
		}
	}
	nA, nB, nQ, nR, nE := dareNearUnitCircle()
	res, err := Dare(nA, nB, nQ, nR, &RiccatiOpts{E: nE})
	check("Dare", res, err)
	res, err = Dlqr(nA, nB, nQ, nR, &RiccatiOpts{E: nE})
	check("Dlqr", res, err)
	sys, err := NewDescriptor(mat.DenseCopyOf(nA.T()), eyeDense(2), mat.DenseCopyOf(nB.T()), mat.NewDense(1, 2, nil), mat.DenseCopyOf(nE.T()), 1)
	if err != nil {
		t.Fatal(err)
	}
	res, err = Kalman(sys, nQ, nR, nil, nil)
	check("Kalman", res, err)
	a, b, q, r, s := hamiltonianAxisPair()
	q[1] = math.Nextafter(q[1], 1)
	A, B, Q, R, S := mat.NewDense(4, 4, a), mat.NewDense(4, 3, b), mat.NewDense(4, 4, q), mat.NewDense(3, 3, r), mat.NewDense(4, 3, s)
	res, err = Lqr(A, B, Q, R, &RiccatiOpts{S: S})
	check("Lqr", res, err)
	res, err = Lqe(mat.DenseCopyOf(A.T()), eyeDense(4), mat.DenseCopyOf(B.T()), Q, R, S, nil)
	check("Lqe", res, err)
}

// TestRiccatiCertifyMapping: certify, the one exit of the exported
// Riccati operations, maps each unexported core sentinel to its exported
// one and keeps the message (ZOJBZX Q3-Q5): errNoCandidate to
// ErrNotConverged, errSingularBasis (icare/idare Report 2) to
// ErrSingularEquation, errBoundaryUnstabilizable to ErrNoStabilizing with
// the operation's sentinel; ErrSchurFailed is an ErrNotConverged with its
// own message, and other errors pass unchanged.
func TestRiccatiCertifyMapping(t *testing.T) {
	partial := &RiccatiResult{X: eyeDense(1), K: eyeDense(1), accuracy: 1}
	for _, c := range []struct {
		name       string
		res        *RiccatiResult
		err        error
		unsolvable error
		want       []error
		not        []error
		msg        string
	}{
		{"not converged", nil, fmt.Errorf("Care: %w", errNoCandidate), ErrNotStabilizable, []error{ErrNotConverged}, []error{ErrNoStabilizing, ErrNotStabilizable, ErrSingularEquation}, "Care: no verified stabilizing solution found: controlsys: iteration did not converge"},
		{"not converged, unscaled verdict", nil, fmt.Errorf("Dare: %w", unscaledBoundaryError{errNoCandidate}), ErrNotStabilizable, []error{ErrNotConverged}, []error{ErrNoStabilizing, ErrNotStabilizable}, "Dare: no verified stabilizing solution found: controlsys: iteration did not converge"},
		{"singular basis", nil, fmt.Errorf("Dare: %w", errSingularBasis), ErrNotStabilizable, []error{ErrSingularEquation}, []error{ErrNotConverged, ErrNoStabilizing}, "Dare: stable subspace basis U11 singular, X not finite: controlsys: matrix equation is singular or nearly singular"},
		{"boundary, control", nil, fmt.Errorf("Care: %w", errBoundaryUnstabilizable), ErrNotStabilizable, []error{ErrNoStabilizing, ErrNotStabilizable}, []error{ErrNotDetectable, ErrNotConverged}, "Care: controlsys: no stabilizing solution exists: a mode on or beyond the stability boundary is unstabilizable"},
		{"boundary, filter", nil, fmt.Errorf("Kalman: Dare: %w", errBoundaryUnstabilizable), ErrNotDetectable, []error{ErrNoStabilizing, ErrNotDetectable}, []error{ErrNotStabilizable, ErrNotConverged}, "Kalman: Dare: controlsys: no stabilizing solution exists: a mode on or beyond the stability boundary is unstabilizable"},
		{"Schur", nil, fmt.Errorf("Care: %w", ErrSchurFailed), ErrNotStabilizable, []error{ErrSchurFailed, ErrNotConverged}, []error{ErrNoStabilizing}, "Care: controlsys: Schur decomposition failed to converge"},
		{"other", nil, ErrSingularR, ErrNotStabilizable, []error{ErrSingularR}, []error{ErrNotConverged}, ErrSingularR.Error()},
		{"unverified", partial, nil, ErrNotDetectable, []error{ErrAccuracyUnverified}, []error{ErrNotConverged, ErrNotDetectable}, "Kalman: solution not certified: controlsys: result accuracy not verified"},
	} {
		op := "Care"
		if c.unsolvable == ErrNotDetectable {
			op = "Kalman"
		}
		res, err := certify(op, c.unsolvable, c.res, c.err)
		for _, w := range c.want {
			if !errors.Is(err, w) {
				t.Errorf("%s: %v is not %v", c.name, err, w)
			}
		}
		for _, w := range c.not {
			if errors.Is(err, w) {
				t.Errorf("%s: %v is %v", c.name, err, w)
			}
		}
		if err == nil || err.Error() != c.msg {
			t.Errorf("%s: message %q, want %q", c.name, err, c.msg)
		}
		if (res != nil) != (c.res != nil) || res != nil && res != c.res {
			t.Errorf("%s: result %v, want %v", c.name, res, c.res)
		}
		if msg := riccatiContractBreach(res, err); msg != "" {
			t.Errorf("%s: %s", c.name, msg)
		}
	}
	certified := &RiccatiResult{X: eyeDense(1), K: eyeDense(1)}
	if res, err := certify("Dare", ErrNotStabilizable, certified, nil); res != certified || err != nil {
		t.Errorf("certified: %v, %v; want the result with a nil error", res, err)
	}
	if res, err := certify("Dare", ErrNotStabilizable, partial, fmt.Errorf("Dare: %w", errNoCandidate)); res != nil || err == nil {
		t.Errorf("error with a result: %v, %v; want a nil result", res, err)
	}
	if ErrSchurFailed.Error() != "controlsys: Schur decomposition failed to converge" {
		t.Errorf("ErrSchurFailed message %q", ErrSchurFailed)
	}
}

// bitEqualDense reports a and b of equal shape and bits.
func bitEqualDense(a, b *mat.Dense) bool {
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	if ra != rb || ca != cb {
		return false
	}
	for i := range ra {
		for j := range ca {
			if math.Float64bits(a.At(i, j)) != math.Float64bits(b.At(i, j)) {
				return false
			}
		}
	}
	return true
}

// TestSchurFailedIsNotConverged: ErrSchurFailed is an ErrNotConverged
// (ZOJBZX Q4) and stays matchable through wrapping.
func TestSchurFailedIsNotConverged(t *testing.T) {
	if !errors.Is(ErrSchurFailed, ErrNotConverged) {
		t.Error("ErrSchurFailed is not ErrNotConverged")
	}
	wrapped := fmt.Errorf("x: %w", ErrSchurFailed)
	if !errors.Is(wrapped, ErrSchurFailed) || !errors.Is(wrapped, ErrNotConverged) {
		t.Errorf("%v: lost ErrSchurFailed or ErrNotConverged", wrapped)
	}
	if errors.Is(ErrNotConverged, ErrSchurFailed) {
		t.Error("ErrNotConverged is ErrSchurFailed")
	}
}

// riccatiContractClass is the outcome class of an exported Riccati
// operation under the ZOJBZX contract.
type riccatiContractClass int

const (
	contractCertified riccatiContractClass = iota
	contractUnverified
	contractNotConverged
	contractProved
	contractBoundaryProved
)

func (c riccatiContractClass) String() string {
	return [...]string{"certified", "unverified", "not converged", "proved", "boundary eigenvalue proved"}[c]
}

// TestRiccatiContractByOperation checks each exported Riccati operation
// in each outcome class it has a fixture for (ZOJBZX):
//   - certified: (res, nil);
//   - uncertified: (res ≠ nil, ErrAccuracyUnverified);
//   - not proved: (nil, ErrNotConverged), never ErrNoStabilizing;
//   - proved: (nil, ErrNoStabilizing) with ErrNotStabilizable for the
//     control side (Care, Dare, Lqr, Dlqr, Lqi, Lqrd) and ErrNotDetectable
//     for the estimator side (Lqe, Kalman, Kalmd), never the other;
//   - boundary eigenvalue proved (Care, Dare since I4J6DP; Lqr, Dlqr, Lqe
//     and Kalman, the last two on their dual, since JFLGVW): (nil,
//     ErrNoStabilizing) with neither ErrNotStabilizable nor ErrNotDetectable.
//
// Fixtures: a stabilizable well-conditioned plant (certified); weakBStall
// and its dual, and the well-conditioned-E Dare
// (TestRiccatiUnverifiedPartialResult; uncertified); the Hamiltonian axis
// pair (TestCare_HamiltonianAxisPairNoStabilizing) and the noise-unreachable
// boundary mode (TestKalman_BoundaryModeUnreachableFromNoise), whose
// boundary eigenvalue is proved; not proved, the axis pair with Q one ulp
// off symmetric (Care, Lqr, Lqe) and the census near-unit-circle Dare
// (TestDare_NearUnitCircleNotProved; Dare, Dlqr, Kalman on its dual);
// an exactly uncontrollable (unobservable) unstable mode, and K72KOR's
// undetectable mode at z = 1 (proved).
//
// Lqi, Lqrd and Kalmd build their Riccati data (the integrator
// augmentation, the Van Loan discretization), so the uncertified and
// not-proved fixtures, which are exact Riccati data, do not carry over and
// no plant is known to reach those classes through them: they are checked
// certified and proved only. Their uncertified and not-proved exits are the
// same certify as Lqr, Dlqr and Kalman's (Lqi and Lqrd return Lqr and
// Dlqr; Kalmd certifies its dareFor result like Kalman), which the rows
// for those cover.
func TestRiccatiContractByOperation(t *testing.T) {
	cA := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	cB := mat.NewDense(2, 1, []float64{0, 1})
	cC := mat.NewDense(1, 2, []float64{1, 0})
	dA := mat.NewDense(2, 2, []float64{1.1, 0.2, 0, 0.5})
	dB := mat.NewDense(2, 1, []float64{1, 1})
	I2, I1 := eyeDense(2), eyeDense(1)
	cSys, err := New(cA, cB, cC, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	dSys, err := New(dA, dB, mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 1)
	if err != nil {
		t.Fatal(err)
	}

	uA := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	uB := mat.NewDense(2, 1, []float64{0, 1})
	uC := mat.NewDense(1, 2, []float64{0, 1})
	uDA := mat.NewDense(2, 2, []float64{2, 0, 0, 0.5})
	ctlSys, err := New(uA, uB, uC, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	obsSys, err := New(uA, I2, uC, mat.NewDense(1, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	k72 := syKalmanSpec(fzDecode(t, 0xa7, [fzNVals]float64{-0.3010299956639812, 45.6, -68, 1, 0, -6, -30, 156, 1.2, 23.7824, 135.4, 48, -10, -2, 47, -1.1111}))
	k72qn, k72rn, k72nn := syNoise(k72, k72.M, k72.P)
	var k72N *mat.Dense
	if k72nn != nil {
		k72N = mqDense(k72.M, k72.P, k72nn)
	}
	k72Sys := k72.System(t, fzSysOpts{})

	wA, wB, wQ, wR, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	wn, wm := wB.Dims()
	wAt, wBt := mat.DenseCopyOf(wA.T()), mat.DenseCopyOf(wB.T())
	wSys, err := New(wAt, eyeDense(wn), wBt, mat.NewDense(wm, wn, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	eA := mat.NewDense(2, 2, []float64{-456.1125380107185, 212.12282671008353, -6542.374939554918, 3040.8017276406886})
	eB := mat.NewDense(2, 2, []float64{0.1511888949722682, 1.360456698990255, -0.010540390965378899, -0.09484655257220741})
	eOpts := &RiccatiOpts{E: mat.NewDense(2, 2, []float64{0.35739616083221576, 0.9339528811574985, -0.9339528811574985, 0.357396160832216})}

	xa, xb, xq, xr, xs := hamiltonianAxisPair()
	xA, xB, xQ, xR, xS := mat.NewDense(4, 4, xa), mat.NewDense(4, 3, xb), mat.NewDense(4, 4, xq), mat.NewDense(3, 3, xr), mat.NewDense(4, 3, xs)
	pA, pB, pC, pE := boundaryUnreachablePlant(0)
	pSys, err := NewDescriptor(pA, pB, pC, mat.NewDense(1, 1, nil), pE, 1)
	if err != nil {
		t.Fatal(err)
	}
	pAt, pCt, pQ, pEt := dualDare(pA, pB, pC, pE)
	xqa := slices.Clone(xq)
	xqa[1] = math.Nextafter(xqa[1], 1)
	xQa := mat.NewDense(4, 4, xqa)
	nA, nB, nQ, nR, nE := dareNearUnitCircle()
	nSys, err := NewDescriptor(mat.DenseCopyOf(nA.T()), I2, mat.DenseCopyOf(nB.T()), mat.NewDense(1, 2, nil), mat.DenseCopyOf(nE.T()), 1)
	if err != nil {
		t.Fatal(err)
	}

	type op = func() (*RiccatiResult, error)
	cases := []struct {
		name  string
		class riccatiContractClass
		run   op
	}{
		{"Care", contractCertified, func() (*RiccatiResult, error) { return Care(cA, cB, I2, I1, nil) }},
		{"Care", contractUnverified, func() (*RiccatiResult, error) { return Care(wA, wB, wQ, wR, nil) }},
		{"Care", contractNotConverged, func() (*RiccatiResult, error) { return Care(xA, xB, xQa, xR, &RiccatiOpts{S: xS}) }},
		{"Care", contractBoundaryProved, func() (*RiccatiResult, error) { return Care(xA, xB, xQ, xR, &RiccatiOpts{S: xS}) }},
		{"Care", contractProved, func() (*RiccatiResult, error) { return Care(uA, uB, I2, I1, nil) }},
		{"Lqr", contractCertified, func() (*RiccatiResult, error) { return Lqr(cA, cB, I2, I1, nil) }},
		{"Lqr", contractUnverified, func() (*RiccatiResult, error) { return Lqr(wA, wB, wQ, wR, nil) }},
		{"Lqr", contractNotConverged, func() (*RiccatiResult, error) { return Lqr(xA, xB, xQa, xR, &RiccatiOpts{S: xS}) }},
		{"Lqr", contractBoundaryProved, func() (*RiccatiResult, error) { return Lqr(xA, xB, xQ, xR, &RiccatiOpts{S: xS}) }},
		{"Lqr", contractProved, func() (*RiccatiResult, error) { return Lqr(uA, uB, I2, I1, nil) }},
		{"Dare", contractCertified, func() (*RiccatiResult, error) { return Dare(dA, dB, I2, I1, nil) }},
		{"Dare", contractUnverified, func() (*RiccatiResult, error) { return Dare(eA, eB, I2, I2, eOpts) }},
		{"Dare", contractNotConverged, func() (*RiccatiResult, error) { return Dare(nA, nB, nQ, nR, &RiccatiOpts{E: nE}) }},
		{"Dare", contractBoundaryProved, func() (*RiccatiResult, error) { return Dare(pAt, pCt, pQ, I1, &RiccatiOpts{E: pEt}) }},
		{"Dare", contractProved, func() (*RiccatiResult, error) { return Dare(uDA, uB, I2, I1, nil) }},
		{"Dlqr", contractCertified, func() (*RiccatiResult, error) { return Dlqr(dA, dB, I2, I1, nil) }},
		{"Dlqr", contractUnverified, func() (*RiccatiResult, error) { return Dlqr(eA, eB, I2, I2, eOpts) }},
		{"Dlqr", contractNotConverged, func() (*RiccatiResult, error) { return Dlqr(nA, nB, nQ, nR, &RiccatiOpts{E: nE}) }},
		{"Dlqr", contractBoundaryProved, func() (*RiccatiResult, error) { return Dlqr(pAt, pCt, pQ, I1, &RiccatiOpts{E: pEt}) }},
		{"Dlqr", contractProved, func() (*RiccatiResult, error) { return Dlqr(uDA, uB, I2, I1, nil) }},
		{"Lqi", contractCertified, func() (*RiccatiResult, error) { return Lqi(cSys, eyeDense(3), I1, nil) }},
		{"Lqi", contractProved, func() (*RiccatiResult, error) { return Lqi(ctlSys, eyeDense(3), I1, nil) }},
		{"Lqrd", contractCertified, func() (*RiccatiResult, error) { return Lqrd(cA, cB, I2, I1, 0.1, nil) }},
		{"Lqrd", contractProved, func() (*RiccatiResult, error) { return Lqrd(uA, uB, I2, I1, 0.1, nil) }},
		{"Lqe", contractCertified, func() (*RiccatiResult, error) { return Lqe(cA, I2, cC, I2, I1, nil, nil) }},
		{"Lqe", contractUnverified, func() (*RiccatiResult, error) { return Lqe(wAt, eyeDense(wn), wBt, wQ, wR, nil, nil) }},
		{"Lqe", contractNotConverged, func() (*RiccatiResult, error) {
			return Lqe(mat.DenseCopyOf(xA.T()), eyeDense(4), mat.DenseCopyOf(xB.T()), xQa, xR, xS, nil)
		}},
		{"Lqe", contractBoundaryProved, func() (*RiccatiResult, error) {
			return Lqe(mat.DenseCopyOf(xA.T()), eyeDense(4), mat.DenseCopyOf(xB.T()), xQ, xR, xS, nil)
		}},
		{"Lqe", contractProved, func() (*RiccatiResult, error) { return Lqe(uA, I2, uC, I2, I1, nil, nil) }},
		{"Kalman", contractCertified, func() (*RiccatiResult, error) { return Kalman(dSys, I1, I1, nil, nil) }},
		{"Kalman", contractUnverified, func() (*RiccatiResult, error) { return Kalman(wSys, wQ, wR, nil, nil) }},
		{"Kalman", contractNotConverged, func() (*RiccatiResult, error) { return Kalman(nSys, nQ, nR, nil, nil) }},
		{"Kalman", contractBoundaryProved, func() (*RiccatiResult, error) { return Kalman(pSys, I1, I1, nil, nil) }},
		{"Kalman", contractProved, func() (*RiccatiResult, error) {
			return Kalman(k72Sys, mqDense(k72.M, k72.M, k72qn), mqDense(k72.P, k72.P, k72rn), k72N, nil)
		}},
		{"Kalmd", contractCertified, func() (*RiccatiResult, error) { return Kalmd(cSys, I1, I1, 0.1, nil) }},
		{"Kalmd", contractProved, func() (*RiccatiResult, error) { return Kalmd(obsSys, I2, I1, 0.1, nil) }},
	}
	estimator := map[string]bool{"Lqe": true, "Kalman": true, "Kalmd": true}
	for _, c := range cases {
		res, err := c.run()
		what := fmt.Sprintf("%s %v", c.name, c.class)
		if msg := riccatiContractBreach(res, err); msg != "" {
			t.Errorf("%s: %s", what, msg)
		}
		switch c.class {
		case contractCertified:
			if err != nil || res == nil {
				t.Errorf("%s: result %v, err %v; want a result with nil", what, res != nil, err)
			}
		case contractUnverified:
			if !errors.Is(err, ErrAccuracyUnverified) || res == nil {
				t.Errorf("%s: result %v, err %v; want a result with ErrAccuracyUnverified", what, res != nil, err)
			}
		case contractNotConverged:
			if !errors.Is(err, ErrNotConverged) || errors.Is(err, ErrNoStabilizing) || res != nil {
				t.Errorf("%s: result %v, err %v; want nil with ErrNotConverged", what, res != nil, err)
			}
		case contractProved:
			want, not := ErrNotStabilizable, ErrNotDetectable
			if estimator[c.name] {
				want, not = not, want
			}
			if !errors.Is(err, ErrNoStabilizing) || !errors.Is(err, want) || errors.Is(err, not) || errors.Is(err, ErrNotConverged) || res != nil {
				t.Errorf("%s: result %v, err %v; want nil with ErrNoStabilizing and %v", what, res != nil, err, want)
			}
		case contractBoundaryProved:
			if !errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) || errors.Is(err, ErrNotDetectable) || errors.Is(err, ErrNotConverged) || res != nil {
				t.Errorf("%s: result %v, err %v; want nil with ErrNoStabilizing alone", what, res != nil, err)
			}
		}
	}
}
