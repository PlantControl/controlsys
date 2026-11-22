package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestModsepIllConditionedDecouplingErrors: base Modsep returned nil on
// splits whose Sylvester decoupling had ‖X‖_F ≈ 1.5e36 (tied/717, PZET7B,
// cutoff 1.3486991523489157e-12, Ts 0.5: parts ≈ 5e20, ‖G−G1−G2‖ ≈ 1e19)
// and ≈ 1e8 (rotZ/880, AST4BK: a near-defective pair at ±0.5i split across
// the cutoff, ‖G−G1−G2‖ ≈ 2e5 SepTol units). Both are rejected now.
func TestModsepIllConditionedDecouplingErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		sys  *System
		cut  float64
	}{
		{"tied/717", sepLit(t, 7, 1, 1, []float64{0.99999999999909051, 1, 0, 0, 0, -0.5, 0, 0, 0.99999999999909051, 0, 0, 0, 0.5, 1, 0, 0, -1, 0, 0, 0, 0.5, -0.5, 0, 0, 1, 0, 0, 1, 0, 1, -1, 0, 0.99999999999909051, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
			[]float64{-1, 0.5, -1, 2, 2, -1, 0.5}, []float64{2, 2, 2, -1, 2, 1, 1}, []float64{-1}, 0.5), 1.3486991523489157e-12},
		{"rotZ/880", sepLit(t, 6, 1, 1, []float64{-3.637978807091713e-12, 0.5, 0, 1, -1, -0.5, 0, -1.4901161193847656e-08, 0, -0.5, -0.5, 0, 0, 1, -2, 0, -0.5, 0, 0, 0.5, 0, -1.4901161193847656e-08, 0.5, 0, 0, 0, 0, 0, -3.814697265625e-06, 0, 0.5, 0, 0, 1, 0, -3.637978807091713e-12},
			[]float64{0.5, -1, 1, 2, 1, -1}, []float64{1, 0, 0.5, 1, 1, 2}, []float64{0}, 0), 0.50000000000000033},
	} {
		res, err := Modsep(tc.sys, tc.cut)
		if !errors.Is(err, ErrSchurFailed) {
			r := -1.0
			if err == nil {
				pts := sepPointsC
				if tc.sys.IsDiscrete() {
					pts = sepPointsD
				}
				r = sepTolRatio(t, tc.sys, tc.sys, res.Slow, res.Fast, pts)
			}
			t.Errorf("%s: Modsep err = %v (sum/SepTol unit %.3g), want ErrSchurFailed", tc.name, err, r)
		}
	}
}

// TestModsepNegligibleCouplingStaysNil: the rotH/95 type (PZET7B ruling
// tests), a decoupling with ‖X‖ ≈ 2²⁰ whose fast mode the input barely
// drives (‖B̃2‖ = 2⁻³⁰): ‖C̃1‖‖X‖‖B̃2‖ is O(‖C‖‖B‖), the split is
// accurate, and the certificate, weighting X by ‖C̃1‖‖B̃2‖, keeps it nil.
// A = [1/4 γ; 0 1] is already in Schur form, so the split is the same on
// every arch.
func TestModsepNegligibleCouplingStaysNil(t *testing.T) {
	sys := sepLit(t, 2, 1, 1, []float64{0.25, 0x1p20, 0, 1}, []float64{1, 0x1p-30}, []float64{1, -0.5}, []float64{0}, 0)
	res, err := Modsep(sys, 0.5)
	if err != nil {
		t.Fatalf("Modsep err = %v, want nil", err)
	}
	if ns, _, _ := res.Slow.Dims(); ns != 1 {
		t.Fatalf("Slow order %d, want 1 (a split, not a trivial copy)", ns)
	}
	if r := sepTolRatio(t, sys, sys, res.Slow, res.Fast, sepPointsC); r > 10.0/8 {
		t.Fatalf("sum/SepTol unit = %.3g, want ≤ 1.25", r)
	}
}

// TestModsepAcceptedSumWithinSepTol: well-separated random non-symmetric
// models, continuous and discrete, split nil with G1 + G2 within SepTol/8 of
// the normwise rounding accuracy of G.
func TestModsepAcceptedSumWithinSepTol(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 17))
	for trial := range 12 {
		n, m, p := 4+trial%5, 1+trial%2, 1+trial%3
		dt := 0.0
		if trial%2 == 1 {
			dt = 0.1
		}
		T := make([]float64, n*n)
		for i := range n {
			v := float64(i+1) * 0.6
			if dt > 0 {
				v = 0.15 + 0.8*float64(i)/float64(n)
			}
			T[i*n+i] = v
			for j := i + 1; j < n; j++ {
				T[i*n+j] = 0.5 * rng.NormFloat64()
			}
		}
		Q := sepRandOrthogonal(rng, n)
		a := make([]float64, n*n)
		for i := range n {
			for j := range n {
				var s float64
				for k := range n {
					for l := range n {
						s += Q[i*n+k] * T[k*n+l] * Q[j*n+l]
					}
				}
				a[i*n+j] = s
			}
		}
		b, c := make([]float64, n*m), make([]float64, p*n)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		for i := range c {
			c[i] = rng.NormFloat64()
		}
		sys := sepLit(t, n, m, p, a, b, c, make([]float64, p*m), dt)
		cut := 0.6 * (float64(n/2) + 0.5)
		pts := sepPointsC
		if dt > 0 {
			cut = math.Abs(math.Log(0.15+0.8*(float64(n/2)-0.5)/float64(n))) / dt
			pts = sepPointsD
		}
		res, err := Modsep(sys, cut)
		if err != nil {
			t.Fatalf("trial %d: Modsep err = %v, want nil", trial, err)
		}
		if r := sepTolRatio(t, sys, sys, res.Slow, res.Fast, pts); r > 10.0/8 {
			t.Fatalf("trial %d: sum/SepTol unit = %.3g, want ≤ 1.25", trial, r)
		}
	}
}

func sepRandOrthogonal(rng *rand.Rand, n int) []float64 {
	g := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(g)
	var q mat.Dense
	qr.QTo(&q)
	return q.RawMatrix().Data
}
