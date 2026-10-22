package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// cheapControlCare is a trial of the INY4MG sweep (3CVH7I): n = 5, m = 2,
// A = 0.4·N(0,1), B = N(0,1), Q = MMᵀ·10^U(3,6), R = LLᵀ·10^U(−15,−11).
func cheapControlCare(rng *rand.Rand) (A, B, Q, R *mat.Dense) {
	const n, m = 5, 2
	normal := func(r, c int, s float64) *mat.Dense {
		d := mat.NewDense(r, c, nil)
		for i := range r {
			for j := range c {
				d.Set(i, j, s*rng.NormFloat64())
			}
		}
		return d
	}
	A, B = normal(n, n, 0.4), normal(n, m, 1)
	M := normal(n, n, 1)
	Q = mat.NewDense(n, n, nil)
	Q.Mul(M, M.T())
	Q.Scale(math.Pow(10, 3+3*rng.Float64()), Q)
	L := normal(m, m, 1)
	R = mat.NewDense(m, m, nil)
	R.Mul(L, L.T())
	R.Scale(math.Pow(10, -15+4*rng.Float64()), R)
	return A, B, Q, R
}

// TestCare_CostProxyScaling (3CVH7I): state scaling balanced with B
// ignores R, so under cheap control its T leaves no cost scale that
// deflates the Hamiltonian or the pencil ("modeOnly" on arm64: every
// scale fails) and Care leaned on the NoScaling retry. Balanced with B·U⁻¹ at ‖C‖,
// the scaled mode itself certifies modeOnly, to the X of the unscaled
// mode, and most of the sweep (3000 trials: 1888 certified with B, 2616
// with the proxy); Care, which falls back to B, certifies every trial
// either mode does.
func TestCare_CostProxyScaling(t *testing.T) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, "modeOnly").dense()
	res, err := care(A, B, Q, R, nil)
	if err != nil || res.Accuracy != 0 {
		t.Fatalf("modeOnly scaled: %v, %+v", err, res)
	}
	ref, err := care(A, B, Q, R, &RiccatiOpts{NoScaling: true})
	if err != nil || ref.Accuracy != 0 {
		t.Fatalf("modeOnly unscaled: %v", err)
	}
	var d mat.Dense
	d.Sub(res.X, ref.X)
	if rel := mat.Norm(&d, 2) / mat.Norm(ref.X, 2); !(rel <= 1e-8) {
		t.Errorf("modeOnly: scaled X off the unscaled one by %.3g", rel)
	}

	const trials = 300
	var plain, proxy int
	rng := rand.New(rand.NewPCG(7, 11))
	for i := range trials {
		A, B, Q, R := cheapControlCare(rng)
		rp, ep := care(A, B, Q, R, &RiccatiOpts{plainScaling: true})
		rc, ec := care(A, B, Q, R, nil)
		rn, en := care(A, B, Q, R, &RiccatiOpts{NoScaling: true})
		ok := func(r *RiccatiResult, e error) bool { return e == nil && r.Accuracy == 0 }
		if ok(rp, ep) {
			plain++
		}
		if ok(rc, ec) {
			proxy++
		}
		if ok(rp, ep) || ok(rc, ec) || ok(rn, en) {
			if res, err := Care(A, B, Q, R, nil); !ok(res, err) {
				t.Errorf("trial %d: Care %v, %+v; a mode certifies", i, err, res)
			}
		}
	}
	if proxy < plain+trials/10 {
		t.Errorf("scaled mode certified %d of %d with the proxy, %d with B", proxy, trials, plain)
	}
}
