package controlsys

import (
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack/gonum"
	"plantcontrol.org/v1/gonum/mat"
)

// TestDare_DescriptorNearSingularFirstStepNoiseIsNotCertified (D5PKRG):
// the dual Dare of a Kalman filter with E singular values 1.90 and
// 1.76e-12 (cancellation of EᵀXE 2.2e18), from its pencil X 4.3e-8 off
// the 320-bit solution. The float64 Newton step of X is rounding noise
// there: under 1-ulp moves of X it ranged 9e-10 to 1.9e-7, so some moves
// accepted X with K exact while the descriptor refinement stalled, and X
// was certified on K alone. The EᵀXE step fails on all of them; none may
// keep Accuracy 0 off by more than newtonTol. How many moves accepted
// depends on gonum's rounding (5 of 47 arm64, 2 of 47 amd64 before).
func TestDare_DescriptorNearSingularFirstStepNoiseIsNotCertified(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.6426426310425746, 0.4755115130022278, 0.6418460336191358, 1.3029988471948917})
	B := mat.NewDense(2, 1, []float64{2.736418800501581, -1.0193367423216586})
	Q := mat.NewDense(2, 2, []float64{3.434438929316185, -1.1398381928138346, -1.1398381928138346, 0.3782950090353746})
	R := mat.NewDense(1, 1, []float64{0.5664526206691559})
	E := mat.NewDense(2, 2, []float64{0.8862090667958542, 0, -1.68131954178768, 3.767150261254627e-12})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, 320)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	const n = 2
	x0 := []float64{1.81227569417264092e+24, 9.55234928156196823e+23, 9.55234928156196823e+23, 5.03496113148582821e+23}
	idx := [][2]int{{0, 0}, {0, 1}, {1, 1}}
	pats := [][]int{{0, 0, 0}}
	for q := range idx {
		for _, sg := range []int{1, -1} {
			d := make([]int, 3)
			d[q] = sg
			pats = append(pats, d)
		}
	}
	seed := uint64(12345)
	for range 40 {
		d := make([]int, 3)
		for q := range d {
			seed = seed*6364136223846793005 + 1442695040888963407
			d[q] = int(seed>>62)%3 - 1
		}
		pats = append(pats, d)
	}
	demoted := 0
	for _, d := range pats {
		xp := slices.Clone(x0)
		for q, ij := range idx {
			v := xp[ij[0]*n+ij[1]]
			for range max(d[q], -d[q]) {
				v = math.Nextafter(v, math.Copysign(math.Inf(1), float64(d[q])))
			}
			xp[ij[0]*n+ij[1]], xp[ij[1]*n+ij[0]] = v, v
		}
		opts := &RiccatiOpts{E: E, NoScaling: true}
		problem, err := newRiccatiProblem(A, B, Q, R, opts)
		if err != nil {
			t.Fatal(err)
		}
		if err := problem.ws.fits(n, 1); err != nil {
			t.Fatal(err)
		}
		problem.scale(opts, false)
		rc := problem.ws.rChol[:1]
		rc[0] = problem.R.At(0, 0)
		gonum.Implementation{}.Dpotrf(blas.Upper, 1, rc, 1)
		X := mat.NewDense(n, n, slices.Clone(xp))
		K, err := problem.acceptRiccati(false, X, make([]complex128, n), 5.96e-25)
		if err != nil {
			continue
		}
		if problem.gainUncertified {
			demoted++
			continue
		}
		if e := relErr(X, ref.X); e > newtonTol {
			t.Errorf("nudge %v: X certified %.3g off the 320-bit solution (K %.3g)", d, e, relErr(K, ref.K))
		}
	}
	if demoted == 0 {
		t.Errorf("no nudge of %d reached the uncertified branch", len(pats))
	}
}
