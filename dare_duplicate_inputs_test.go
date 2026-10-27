package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Exactly duplicated actuators with a cross term S under cheap control:
// Dare eliminates the input B does not see and solves the reduced problem
// to 1e-16, but certified it again in the inputs w with BV₂ ≈ 1e-17
// (rounding of V), whose product with K₂ ≈ R₂₂⁻¹S₂ᵀ ≈ 1e12 moved X off
// by 1e-5 to 9e-5 and K by up to 0.15, at Accuracy 0 (G3NWYW). The
// doubling oracle moves 5e-15 under eps perturbations that keep the
// columns equal.
func TestDare_DuplicateInputColumnsWithCrossTermReturnsReducedX(t *testing.T) {
	cases := []struct{ a, b, q, r, s []float64 }{
		{
			[]float64{-0.6757227466642478, 0.9050780114791821, 0.0026091971648505574, 0.6741314989523022, -0.1256833149483058, -0.3309056483929904, -0.4090654770311977, 0.1556457973232021, 0.9326274524935398},
			[]float64{0.025319988053062792, 0.025319988053062792, 2.0703698540817426, 2.0703698540817426, 0.2889502120515861, 0.2889502120515861},
			[]float64{4.8078068982823006e14, -3.4770363703086207e13, 3.443767432031855e14, -3.4770363703086207e13, 7.514084519221203e13, -2.2397053062616156e14, 3.443767432031855e14, -2.2397053062616156e14, 8.035644361958215e14},
			[]float64{1.0181363287747717e-12, 3.177515149529748e-14, 3.177515149529748e-14, 1.057412778366046e-12},
			[]float64{-0.4541325552660498, 0.12131079694018426, -0.3879073121260799, 0.2497148785398958, -0.33713416769609794, 0.33626019474051794},
		},
		{
			[]float64{-2.1547315640019296, -1.4182022364833529, 1.1263060441812394, 2.7202859247955535},
			[]float64{-1.5553448482603027, -1.5553448482603027, 0.22986259320803504, 0.22986259320803504},
			[]float64{1.2180882275523276e12, 1.2067247620445618e12, 1.2067247620445618e12, 8.755947997281988e12},
			[]float64{1.0490843484968255e-12, -2.1584190522342457e-14, -2.1584190522342457e-14, 1.0162420030766115e-12},
			[]float64{-0.18832243113502833, 0.5490584734559056, -0.19042113490561197, 2.5153205898721467},
		},
	}
	for i, c := range cases {
		n := len(c.b) / 2
		A, B, Q := mat.NewDense(n, n, c.a), mat.NewDense(n, 2, c.b), mat.NewDense(n, n, c.q)
		R, S := mat.NewDense(2, 2, c.r), mat.NewDense(n, 2, c.s)
		want := dareDoublingBig(A, B, Q, R, S)
		wantK := dareGainBig(A, B, R, S, want)
		for _, ns := range []bool{false, true} {
			name := fmt.Sprintf("case %d NoScaling=%v", i, ns)
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: ns})
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			if rel := relErr(res.X, want); res.Accuracy != 0 || !(rel <= 1e-12) {
				t.Errorf("%s: Accuracy %d, ‖X − X*‖/‖X*‖ = %.3g", name, res.Accuracy, rel)
			}
			if rel := relErr(res.K, wantK); !(rel <= 1e-8) {
				t.Errorf("%s: ‖K − K*‖/‖K*‖ = %.3g", name, rel)
			}
			for _, e := range res.Eig {
				if !(cmplx.Abs(e) < 1) {
					t.Errorf("%s: Eig %v not stable", name, e)
				}
			}
		}
	}
}

// nullExact separates an exact null space of B (duplicated columns, m > n)
// from a B whose σ_min is below the rank tolerance only by rounding (one
// column one ulp off): zeroing BV₂ for the latter moved X by up to 10×
// the oracle's eps-perturbation sensitivity on a near-duplicate sweep.
func TestNullExactSeparatesExactFromRoundingNullSpace(t *testing.T) {
	dup := []float64{0.025319988053062792, 2.0703698540817426, 0.2889502120515861}
	ulp := slices.Clone(dup)
	ulp[1] = math.Nextafter(ulp[1], 3)
	cases := []struct {
		name string
		b    *mat.Dense
		want bool
	}{
		{"duplicated", mat.NewDense(3, 2, []float64{dup[0], dup[0], dup[1], dup[1], dup[2], dup[2]}), true},
		{"scaled duplicate", mat.NewDense(3, 2, []float64{dup[0], -2 * dup[0], dup[1], -2 * dup[1], dup[2], -2 * dup[2]}), true},
		{"m > n", mat.NewDense(2, 3, []float64{-0.46943512585815883, 0.78425864045369575, 1.5178474246624851, -1.7749767000838221, 0.51375479371405053, -1.3216979308286421}), true},
		{"one ulp off", mat.NewDense(3, 2, []float64{dup[0], ulp[0], dup[1], ulp[1], dup[2], ulp[2]}), false},
	}
	for _, c := range cases {
		n, m := c.b.Dims()
		var svd mat.SVD
		if !svd.Factorize(c.b, mat.SVDFullV) {
			t.Fatalf("%s: SVD failed", c.name)
		}
		sv := svd.Values(nil)
		r := 0
		for _, s := range sv {
			if s > float64(max(n, m))*eps()*sv[0] {
				r++
			}
		}
		if r >= m {
			t.Fatalf("%s: rank %d, want < %d", c.name, r, m)
		}
		var v, bv mat.Dense
		svd.VTo(&v)
		bv.Mul(c.b, &v)
		if got := nullExact(c.b, &bv, &v, sv, r); got != c.want {
			t.Errorf("%s: nullExact = %v, want %v", c.name, got, c.want)
		}
	}
}
