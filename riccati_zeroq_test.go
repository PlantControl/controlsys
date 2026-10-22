package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zeroQOracle returns the stabilizing X of the Q = 0, R = I Riccati
// equation for an E⁻¹A with no stable eigenvalue: X = E⁻ᵀY⁻¹E⁻¹ with
// ÃY + YÃᵀ = B̃B̃ᵀ (continuous) or Y = ÃYÃᵀ − B̃B̃ᵀ (discrete), Ã = E⁻¹A,
// B̃ = E⁻¹B, solved as a Kronecker system.
func zeroQOracle(t *testing.T, continuous bool, A, B, E *mat.Dense) *mat.Dense {
	t.Helper()
	n, _ := A.Dims()
	var Einv mat.Dense
	if E == nil {
		Einv.CloneFrom(eye(n))
	} else if err := Einv.Inverse(E); err != nil {
		t.Fatal(err)
	}
	var At, Bt, BB mat.Dense
	At.Mul(&Einv, A)
	Bt.Mul(&Einv, B)
	BB.Mul(&Bt, Bt.T())
	K := mat.NewDense(n*n, n*n, nil)
	rhs := mat.NewVecDense(n*n, nil)
	for i := range n {
		for j := range n {
			row := i*n + j
			rhs.SetVec(row, BB.At(i, j))
			for k := range n {
				for l := range n {
					col := k*n + l
					var v float64
					if continuous {
						if l == j {
							v += At.At(i, k)
						}
						if k == i {
							v += At.At(j, l)
						}
					} else {
						v = At.At(i, k) * At.At(j, l)
						if k == i && l == j {
							v--
						}
					}
					K.Set(row, col, K.At(row, col)+v)
				}
			}
		}
	}
	var y mat.VecDense
	if err := y.SolveVec(K, rhs); err != nil {
		t.Fatal(err)
	}
	Y := mat.NewDense(n, n, y.RawVector().Data)
	var Yinv, X mat.Dense
	if err := Yinv.Inverse(Y); err != nil {
		t.Fatal(err)
	}
	X.Mul(Einv.T(), &Yinv)
	X.Mul(&X, &Einv)
	return &X
}

func TestRiccati_ZeroQ(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{2, 0.1, 0, 1})
	anti := mat.NewDense(2, 2, []float64{1, 2, -3, 0.5})
	cases := []struct {
		name       string
		continuous bool
		A, B, E    *mat.Dense
		zero       bool
	}{
		{"care scalar unstable", true, mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), nil, false},
		{"care antistable", true, anti, mat.NewDense(2, 2, []float64{1, 0, 0.3, 1}), nil, false},
		{"care antistable E", true, anti, mat.NewDense(2, 1, []float64{1, 0.3}), E, false},
		{"care stable", true, mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, 0.3}), nil, true},
		{"care stable E", true, mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3}), mat.NewDense(2, 1, []float64{1, 0.3}), E, true},
		{"dare antistable", false, anti, mat.NewDense(2, 2, []float64{1, 0, 0.3, 1}), nil, false},
		{"dare antistable E", false, anti, mat.NewDense(2, 1, []float64{1, 0.3}), E, false},
		{"dare stable", false, mat.NewDense(2, 2, []float64{0.5, 2, 0, 0.3}), mat.NewDense(2, 1, []float64{1, 0.3}), nil, true},
		{"dare stable E", false, mat.NewDense(2, 2, []float64{0.5, 2, 0.1, 0.3}), mat.NewDense(2, 1, []float64{1, 0.3}), E, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n, _ := c.A.Dims()
			_, m := c.B.Dims()
			var opts *RiccatiOpts
			if c.E != nil {
				opts = &RiccatiOpts{E: c.E}
			}
			solve := Dare
			if c.continuous {
				solve = Care
			}
			res, err := solve(c.A, c.B, mat.NewDense(n, n, nil), eye(m), opts)
			if err != nil {
				t.Fatal(err)
			}
			if res.Accuracy != 0 {
				t.Fatalf("Accuracy = %d, want 0", res.Accuracy)
			}
			want := mat.NewDense(n, n, nil)
			if !c.zero {
				want = zeroQOracle(t, c.continuous, c.A, c.B, c.E)
			}
			scale := math.Max(1, mat.Norm(want, math.Inf(1)))
			var diff mat.Dense
			diff.Sub(res.X, want)
			if d := mat.Norm(&diff, math.Inf(1)); d > 1e-10*scale {
				t.Fatalf("X = %v, want %v (diff %g)", mat.Formatted(res.X), mat.Formatted(want), d)
			}
		})
	}
}

// TestRiccati_ZeroQMixed checks Q = 0 with stable and unstable modes:
// the stabilizing X keeps the stable eigenvalues and mirrors the unstable
// ones (λ → −λ̄ continuous, λ → 1/λ̄ discrete).
func TestRiccati_ZeroQMixed(t *testing.T) {
	cases := []struct {
		name       string
		continuous bool
		A          *mat.Dense
	}{
		{"care", true, mat.NewDense(2, 2, []float64{1, 2, 0.5, -2})},
		{"dare", false, mat.NewDense(2, 2, []float64{2, 1, 0.3, 0.5})},
	}
	B := mat.NewDense(2, 1, []float64{1, 1})
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var eig mat.Eigen
			if !eig.Factorize(c.A, mat.EigenNone) {
				t.Fatal("eig failed")
			}
			var want []complex128
			for _, l := range eig.Values(nil) {
				switch {
				case c.continuous && real(l) > 0:
					l = -cmplx.Conj(l)
				case !c.continuous && cmplx.Abs(l) > 1:
					l = 1 / cmplx.Conj(l)
				}
				want = append(want, l)
			}
			solve := Dare
			if c.continuous {
				solve = Care
			}
			res, err := solve(c.A, B, mat.NewDense(2, 2, nil), eye(1), nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.Accuracy != 0 {
				t.Fatalf("Accuracy = %d, want 0", res.Accuracy)
			}
			got := append([]complex128(nil), res.Eig...)
			byRe := func(s []complex128) {
				sort.Slice(s, func(i, j int) bool { return real(s[i]) < real(s[j]) })
			}
			byRe(got)
			byRe(want)
			for i := range want {
				if cmplx.Abs(got[i]-want[i]) > 1e-10 {
					t.Fatalf("Eig = %v, want %v", got, want)
				}
			}
			var r float64
			if c.continuous {
				r = careResidual(c.A, B, mat.NewDense(2, 2, nil), eye(1), res.X)
			} else {
				r = dareResidual(c.A, B, mat.NewDense(2, 2, nil), eye(1), res.X)
			}
			if r > 1e-12*math.Max(1, mat.Norm(res.X, 1)) {
				t.Fatalf("residual %g", r)
			}
		})
	}
}
