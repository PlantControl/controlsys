package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// randomClusteredMatched draws a minimal stable continuous ZPK model, 5-8
// poles with real parts in [-2.1, -0.1] and up to two real zeros, and
// discretizes it by matched C2D at dt 1e-3 or 1e-2, which clusters every
// pole within 2e-2 of z = 1 in a graded (triangular) realization.
func randomClusteredMatched(t *testing.T, seed uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 0x6a09e667f3bcc908))
	n := 5 + rng.IntN(4)
	var poles []complex128
	for len(poles) < n {
		re := -(0.1 + 2*rng.Float64())
		if n-len(poles) >= 2 && rng.IntN(2) == 0 {
			im := 2 * rng.Float64()
			poles = append(poles, complex(re, im), complex(re, -im))
			continue
		}
		poles = append(poles, complex(re, 0))
	}
	var zeros []complex128
	for range rng.IntN(3) {
		zeros = append(zeros, complex(-(0.1+2*rng.Float64()), 0))
	}
	dt := []float64{1e-3, 1e-2}[rng.IntN(2)]
	return clusteredMatchedDiscrete(t, zeros, poles, 1+rng.Float64(), dt)
}

// withHiddenClusteredMode appends to the SISO discrete sys a mode
// e^{-a·dt}, a in [0.1, 2.1], that is exactly uncontrollable (B row zero,
// O(1) couplings into the other states) or exactly unobservable. The
// realization stays graded; an orthogonal mix of it would leave the
// response near z = 1 undetermined by its rounded entries (the bound of
// boundedResponse exceeds the response there).
func withHiddenClusteredMode(sys *System, rng *rand.Rand, uncontrollable bool) (*System, error) {
	n, _, _ := sys.Dims()
	N := n + 1
	A, B, C := mat.NewDense(N, N, nil), mat.NewDense(N, 1, nil), mat.NewDense(1, N, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.A)
	B.Slice(0, n, 0, 1).(*mat.Dense).Copy(sys.B)
	C.Slice(0, 1, 0, n).(*mat.Dense).Copy(sys.C)
	A.Set(n, n, math.Exp(-(0.1+2*rng.Float64())*sys.Dt))
	for j := range n {
		if uncontrollable {
			A.Set(j, n, rng.NormFloat64())
		} else {
			A.Set(n, j, rng.NormFloat64())
		}
	}
	if uncontrollable {
		C.Set(0, n, rng.NormFloat64())
	} else {
		B.Set(n, 0, rng.NormFloat64())
	}
	return New(A, B, C, mat.NewDense(1, 1, nil), sys.Dt)
}

// clusteredOracleMisfit is the largest relative difference between red and
// the 320-bit response of the stored matrices of sys on a log grid of
// θ ∈ [0.1·dt, 2.8], z = e^{jθ}.
func clusteredOracleMisfit(t *testing.T, sys, red *System) float64 {
	t.Helper()
	o := newStepOraclePencil(sys)
	worst := 0.0
	for j := range 25 {
		th := sys.Dt * 0.1 * math.Pow(2.8/(sys.Dt*0.1), float64(j)/24)
		z := cmplx.Rect(1, th)
		want := matchedOracleTransfer(o, z)
		got, err := red.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Max(worst, cmplx.Abs(got[0][0]-want)/cmplx.Abs(want))
	}
	return worst
}

// TestMinimalRealizationClusteredMatchedKeepsGenuineModes checks the
// QMS7PE seeds: minimal matched C2D models whose staircase drops a state
// whose coupling is below ε·‖A‖ but carries an O(1) share of the response
// near z = 1 (seed 1: order 6 of 7, misfit 1.0 against the 320-bit
// oracle). Reduce must keep them.
func TestMinimalRealizationClusteredMatchedKeepsGenuineModes(t *testing.T) {
	for _, seed := range []uint64{1, 13, 41} {
		sys := randomClusteredMatched(t, seed)
		n, _, _ := sys.Dims()
		res, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != n {
			t.Errorf("seed %d: order %d, want %d", seed, res.Order, n)
		}
		if e := clusteredOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
			t.Errorf("seed %d: response misfit %.3g", seed, e)
		}
	}
}

// TestMinimalRealizationClusteredMatchedFamilyInBand checks that every
// reduction of the clustered matched family stays within √ε of the 320-bit
// response of the model (QMS7PE: on base 115 of the 300 models lost
// genuine modes beyond that band, 37 of them with a full-degree transfer
// function).
func TestMinimalRealizationClusteredMatchedFamilyInBand(t *testing.T) {
	for seed := range uint64(300) {
		sys := randomClusteredMatched(t, seed)
		n, _, _ := sys.Dims()
		res, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order == n {
			continue
		}
		if e := clusteredOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
			t.Errorf("seed %d: order %d of %d, response misfit %.3g", seed, res.Order, n, e)
		}
	}
}

// TestMinimalRealizationClusteredMatchedHiddenModeVerified plants an exact
// hidden mode in clustered matched models: Reduce may drop it, and only
// it, when the reduced response is within √ε of the 320-bit response;
// otherwise it keeps the model (QMS7PE: the staircase also cut genuine
// modes of these, up to 9 of them, and the rotated remnant of an exact
// cut misfits by up to O(1) near z = 1).
func TestMinimalRealizationClusteredMatchedHiddenModeVerified(t *testing.T) {
	reduced := 0
	for seed := range uint64(100) {
		base := randomClusteredMatched(t, seed)
		n, _, _ := base.Dims()
		rng := rand.New(rand.NewPCG(seed, 99))
		for _, unc := range []bool{true, false} {
			sys, err := withHiddenClusteredMode(base, rng, unc)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.MinimalRealization()
			if err != nil {
				t.Fatal(err)
			}
			switch res.Order {
			case n + 1:
			case n:
				reduced++
				if e := clusteredOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
					t.Errorf("seed %d unc %v: response misfit %.3g", seed, unc, e)
				}
			default:
				t.Errorf("seed %d unc %v: order %d, want %d or %d", seed, unc, res.Order, n, n+1)
			}
		}
	}
	t.Logf("%d of 200 hidden modes removed", reduced)
}

// TestMinimalRealizationHiddenModeClusteredLiteralsReduce guards the
// ZAR4NK literals, clustered discrete models with one hidden mode that the
// unit-circle band check must still let go.
func TestMinimalRealizationHiddenModeClusteredLiteralsReduce(t *testing.T) {
	for _, tc := range zpkHiddenModeClustered(t) {
		res, err := tc.sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != 5 {
			t.Errorf("%s: order %d, want 5", tc.name, res.Order)
		}
	}
}
