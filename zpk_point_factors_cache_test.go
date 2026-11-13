package controlsys

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestBoundedResponseKeptFactorsMatchFresh checks that the GEPP factors a
// boundedResponse keeps per point give bit-identical results to factoring
// afresh: the bounded responses, the sharpened bounds (sharpenAt reads the
// kept factors) and W, η and the terms bounds of zpkPointFactors.
func TestBoundedResponseKeptFactorsMatchFresh(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 29))
	same := func(a, b []float64) bool {
		return slices.EqualFunc(a, b, func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) })
	}
	sameC := func(a, b []complex128) bool {
		return slices.EqualFunc(a, b, func(x, y complex128) bool {
			return math.Float64bits(real(x)) == math.Float64bits(real(y)) && math.Float64bits(imag(x)) == math.Float64bits(imag(y))
		})
	}
	for trial := range 40 {
		n, m := 2+rng.IntN(6), 1+rng.IntN(2)
		a := make([]float64, n*n)
		for i := range a {
			a[i] = rng.NormFloat64() * math.Pow(10, 3*rng.NormFloat64())
		}
		b, c := make([]float64, n*m), make([]float64, n)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		for i := range c {
			c[i] = rng.NormFloat64()
		}
		sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(1, n, c), mat.NewDense(1, m, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		var eig mat.Eigen
		if !eig.Factorize(sys.A, mat.EigenNone) {
			t.Fatal("eigen failed")
		}
		pts := eigenAxisProbes(eig.Values(nil), false)
		for range 6 {
			pts = append(pts, complex(0, math.Pow(10, 4*rng.NormFloat64())))
		}
		br := newBalancedRealization(sys, n, m, 1)
		kept := boundedResponseKeep(br, pts, true)
		fresh := boundedResponseKeep(newBalancedRealization(sys, n, m, 1), pts, false)
		if !slices.Equal(kept.ok, fresh.ok) || !sameC(kept.g, fresh.g) || !same(kept.comp, fresh.comp) || !same(kept.ver, fresh.ver) || !same(kept.terms, fresh.terms) {
			t.Fatalf("trial %d: kept-factor bounded response differs from fresh", trial)
		}
		var pk, pf zpkPointFactors
		for k, s := range pts {
			if !kept.ok[k] {
				continue
			}
			pk.factorFrom(kept.ev.factorsAt(k))
			if !pf.factor(&fresh.ev.br, s) {
				t.Fatalf("trial %d point %d: fresh factor met a zero pivot the kept one did not", trial, k)
			}
			if !sameC(pk.w, pf.w) || !same(pk.absLU, pf.absLU) || math.Float64bits(pk.eta) != math.Float64bits(pf.eta) || !slices.Equal(pk.piv, pf.piv) {
				t.Fatalf("trial %d point %d: W, |L||U| or η differ", trial, k)
			}
			if pk.eta < 1 {
				u := kept.ev.u[k*n*m : (k+1)*n*m]
				f := 1 / (1 - pk.eta)
				lo1, hi1 := pk.termsBounds(&kept.ev.br, u, kept.terms[k*m:(k+1)*m], f)
				lo2, hi2 := pf.termsBounds(&fresh.ev.br, u, fresh.terms[k*m:(k+1)*m], f)
				if !same(lo1, lo2) || !same(hi1, hi2) {
					t.Fatalf("trial %d point %d: terms bounds differ", trial, k)
				}
			}
			kept.sharpenAt(k, true)
			fresh.sharpenAt(k, true)
			o := k * m
			if !sameC(kept.g[o:o+m], fresh.g[o:o+m]) || !same(kept.comp[o:o+m], fresh.comp[o:o+m]) || !same(kept.ver[o:o+m], fresh.ver[o:o+m]) || !sameC(kept.ev.u[k*n*m:(k+1)*n*m], fresh.ev.u[k*n*m:(k+1)*n*m]) {
				t.Fatalf("trial %d point %d: sharpened response differs", trial, k)
			}
		}
		kept.release()
		fresh.release()
	}
}
