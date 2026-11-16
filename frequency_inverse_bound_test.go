package controlsys

import (
	"math"
	"math/big"
	"testing"
)

// FKYKOG: beside a pole the inverse tier's working-precision rounding
// bound γ(|λ||Y| + |Â||Y|) alone holds ρ(R̄) above 1/2 while the residual
// I − PY of the GEPP inverse is ε-small; R̄ from the compensated residual
// keeps the tier, and its bounds hold against the exact M.
func TestInverseBoundCompensatedResidualBesidePole(t *testing.T) {
	sys := withinTermsModel(t, 2, []float64{1, 0.5, 0, -2}, []float64{1, 1}, []float64{1, 1}, []float64{0}, 0)
	for _, k := range []float64{4, 16, 64} {
		lam := 1 + k*0x1p-52
		pt := frequencyPoint{p: complex(lam, 0), q: -1}
		bd := newBalancedDense(sys, 2, 1, 1)
		if !bd.factorAt(pt) {
			t.Fatalf("k=%v: factorAt failed", k)
		}
		br := &bd.balancedRealization
		tc := newTermsCert(br)
		ib := newInverseBound(br, tc.absA, pt, func(y []complex128) { cLUSolve(bd.pencil, bd.piv, y, 2, 2) })
		if ib == nil {
			t.Fatalf("k=%v: no inverse bound", k)
		}
		m := exactInverse2(br.a, lam)
		for i, v := range m {
			if !(ib.mLow[i] <= v*(1+0x1p-51)) {
				t.Errorf("k=%v: mLow[%d] = %v above |M| = %v", k, i, ib.mLow[i], v)
			}
			if v > 1 && !(ib.mLow[i] > 0) {
				t.Errorf("k=%v: mLow[%d] = %v, |M| = %v", k, i, ib.mLow[i], v)
			}
		}
		for _, y0 := range [][]float64{{1, 1}, {1, 1e-9}, {1e-9, 1}} {
			y := append([]float64(nil), y0...)
			if !ib.upper(y) {
				t.Fatalf("k=%v: upper failed", k)
			}
			for i := range 2 {
				if want := m[2*i]*y0[0] + m[2*i+1]*y0[1]; !(y[i] >= want*(1-0x1p-51)) {
					t.Errorf("k=%v y=%v: upper[%d] = %v below |M|y = %v", k, y0, i, y[i], want)
				}
			}
		}
	}
}

// exactInverse2 is |(λI − A)⁻¹| of the real 2×2 a, each entry rounded once.
func exactInverse2(a []float64, lam float64) []float64 {
	r := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	l := r(lam)
	p := [4]*big.Rat{
		new(big.Rat).Sub(l, r(a[0])), new(big.Rat).Neg(r(a[1])),
		new(big.Rat).Neg(r(a[2])), new(big.Rat).Sub(l, r(a[3])),
	}
	det := new(big.Rat).Sub(new(big.Rat).Mul(p[0], p[3]), new(big.Rat).Mul(p[1], p[2]))
	adj := [4]*big.Rat{p[3], new(big.Rat).Neg(p[1]), new(big.Rat).Neg(p[2]), p[0]}
	out := make([]float64, 4)
	for i, v := range adj {
		f, _ := new(big.Rat).Quo(v, det).Float64()
		out[i] = math.Abs(f)
	}
	return out
}
