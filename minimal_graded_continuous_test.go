package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// continuousOraclePoints is a log grid of s = jω, ω ∈ [1e-3, 1e6], with
// s = 0 and j·Im λ, j·|λ| for every eigenvalue λ of A.
func continuousOraclePoints(sys *System) []complex128 {
	pts := []complex128{0}
	for k := range 19 {
		pts = append(pts, complex(0, math.Pow(10, -3+0.5*float64(k))))
	}
	if n, _, _ := sys.Dims(); n > 0 {
		var eig mat.Eigen
		if eig.Factorize(sys.A, mat.EigenNone) {
			for _, ev := range eig.Values(nil) {
				pts = append(pts, complex(0, math.Abs(imag(ev))), complex(0, cmplx.Abs(ev)))
			}
		}
	}
	return pts
}

// storedOracleResponse is the 320-bit response of the stored matrices of sys at
// s, row-major p×m; nil at a pole.
func storedOracleResponse(sys *System, s complex128) []complex128 {
	n, m, p := sys.Dims()
	g := make([]complex128, p*m)
	if n == 0 {
		for i := range p {
			for j := range m {
				g[i*m+j] = complex(sys.D.At(i, j), 0)
			}
		}
		return g
	}
	re, im := newStepOraclePencil(sys).transfer(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))})
	if re == nil {
		return nil
	}
	for i := range p {
		for j := range m {
			g[i*m+j] = stepOracleComplex{re[i][j], im[i][j]}.complex128()
		}
	}
	return g
}

// determinedOracleMisfit is the largest difference between the 320-bit
// responses of red and sys, beyond the first-order rounding bound δ_G of
// sys's own entries (boundedResponse), relative to the largest entry of
// G − D, over continuousOraclePoints: the misfit the model's data
// determine.
func determinedOracleMisfit(t *testing.T, sys, red *System) float64 {
	t.Helper()
	_, m, p := sys.Dims()
	pts := continuousOraclePoints(sys)
	bound := newBoundedResponse(sys, pts)
	worst := 0.0
	for k, s := range pts {
		g, r := storedOracleResponse(sys, s), storedOracleResponse(red, s)
		if g == nil || r == nil || !bound.ok[k] {
			continue
		}
		scale := 0.0
		for i := range p {
			for j := range m {
				scale = math.Max(scale, cmplx.Abs(g[i*m+j]-complex(sys.D.At(i, j), 0)))
			}
		}
		if !(scale > 0) {
			continue
		}
		for i, v := range g {
			worst = math.Max(worst, (cmplx.Abs(v-r[i])-bound.data[k*p*m+i])/scale)
		}
	}
	return worst
}

// TestReduceGradedKalmanContinuousInBand checks every continuous model of
// the first 300 draws of the Kalman-structure family graded by a diagonal
// similarity of 10^±8, in every mode, against the 320-bit response of the
// model: the staircase floor ε·‖A‖ judges couplings of graded states that
// carry the response, and without a resolved spectral gap the cut must be
// verified on the imaginary axis (AYT3ND: on base d358023, 220 reductions
// of 81 models misfit beyond √ε, median 0.94).
func TestReduceGradedKalmanContinuousInBand(t *testing.T) {
	modes := []*ReduceOpts{nil, {Mode: ReduceUncontrollable}, {Mode: ReduceUnobservable}}
	reduced := 0
	for id := range 300 {
		kc := newKalmanCaseScaled(rand.New(rand.NewPCG(uint64(id), 0x9e3779b97f4a7c15)), 8)
		if !kc.continuousDt {
			continue
		}
		n, _, _ := kc.sys.Dims()
		for k, opts := range modes {
			res, err := kc.sys.Reduce(opts)
			if err != nil {
				t.Fatalf("id %d mode %d: %v", id, k, err)
			}
			if res.Order == n {
				continue
			}
			reduced++
			if e := determinedOracleMisfit(t, kc.sys, res.Sys); !(e <= math.Sqrt(eps())) {
				t.Errorf("id %d mode %d: order %d of %d, response misfit %.3g", id, k, res.Order, n, e)
			}
		}
	}
	t.Logf("%d reductions", reduced)
}

// TestReduceGradedKalmanContinuousKeepsResponse pins graded Kalman draws
// whose staircase cut on base d358023 lost the response (AYT3ND): d4
// id 1392 (ReduceUncontrollable) dropped to order 1 of 4, misfit 6.1;
// d4 id 470 misfit 0.38, d4 id 1876 0.042, d8 id 0 1.15.
func TestReduceGradedKalmanContinuousKeepsResponse(t *testing.T) {
	for _, tc := range []struct {
		dec  float64
		id   uint64
		opts *ReduceOpts
	}{
		{4, 1392, &ReduceOpts{Mode: ReduceUncontrollable}},
		{4, 470, &ReduceOpts{Mode: ReduceUncontrollable}},
		{4, 1876, nil},
		{8, 0, nil},
	} {
		kc := newKalmanCaseScaled(rand.New(rand.NewPCG(tc.id, 0x9e3779b97f4a7c15)), tc.dec)
		res, err := kc.sys.Reduce(tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		if e := determinedOracleMisfit(t, kc.sys, res.Sys); !(e <= math.Sqrt(eps())) {
			t.Errorf("d%g id %d: order %d, response misfit %.3g", tc.dec, tc.id, res.Order, e)
		}
	}
}

// TestMinimalRealizationNearSingularExplicitKeepsResponse checks explicit
// forms of near-singular descriptor models (nearSingularEDraw, as in the
// NZIRSS census) whose cut on base d358023 misfit the response beyond the
// model's own rounding by 0.06 to 1.5 (AYT3ND): the slow eigenvalues are
// not resolved from zero against ‖A‖ ≈ 1e15, so no resolved gap exists,
// and the staircase dropped the states that carry the response.
func TestMinimalRealizationNearSingularExplicitKeepsResponse(t *testing.T) {
	for _, seed := range []uint64{33, 116, 273, 307, 657, 719, 1197, 1460, 1589, 1710, 1808} {
		rng := rand.New(rand.NewPCG(seed, 7704))
		n := 2 + rng.IntN(5)
		m, p := 1, 1
		if rng.IntN(3) == 0 {
			m, p = 1+rng.IntN(2), 1+rng.IntN(2)
		}
		k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
		desc, _ := nearSingularEDraw(t, rng, n, m, p, k)
		sys, err := desc.ToExplicit()
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if e := determinedOracleMisfit(t, sys, res.Sys); !(e <= math.Sqrt(eps())) {
			t.Errorf("seed %d: order %d, response misfit %.3g", seed, res.Order, e)
		}
	}
}
