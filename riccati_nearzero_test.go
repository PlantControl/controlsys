package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dareFixedPoint iterates the Riccati difference equation from X = 0 to
// convergence: an oracle independent of the pencil methods.
func dareFixedPoint(A, B, Q, R *mat.Dense, maxIter int) *mat.Dense {
	n, _ := A.Dims()
	X := mat.NewDense(n, n, nil)
	var xa, xb, g, gk, rb, k, nx, d mat.Dense
	for range maxIter {
		xa.Mul(X, A)
		xb.Mul(X, B)
		nx.Mul(A.T(), &xa)
		g.Mul(A.T(), &xb)
		rb.Mul(B.T(), &xb)
		rb.Add(&rb, R)
		k.Solve(&rb, g.T())
		gk.Mul(&g, &k)
		nx.Sub(&nx, &gk)
		nx.Add(&nx, Q)
		d.Add(&nx, nx.T())
		d.Scale(0.5, &d)
		nx.Sub(&d, X)
		done := mat.Norm(&nx, 1) <= 1e-15*mat.Norm(&d, 1)
		X.Copy(&d)
		if done {
			break
		}
	}
	return X
}

func closedLoopRadius(t *testing.T, A, B, K *mat.Dense) float64 {
	t.Helper()
	var bk, acl mat.Dense
	bk.Mul(B, K)
	acl.Sub(A, &bk)
	return spectralRadius(t, &acl)
}

func TestDareNearZeroEigenvalues(t *testing.T) {
	d := func(v ...float64) *mat.Dense { return mat.NewDense(1, len(v), v) }
	expm := func(ac []float64, dt float64) *mat.Dense {
		var as, ad mat.Dense
		as.Scale(dt, mat.NewDense(2, 2, ac))
		ad.Exp(&as)
		return &ad
	}
	stiff := []float64{-50, 10, 0, -80}
	b2 := mat.NewDense(2, 1, []float64{0.3, 1})
	i2 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	tests := []struct {
		name       string
		A, B, Q, R *mat.Dense
	}{
		{"scalar a=3e-17", d(3e-17), d(-0.92), d(1), d(1)},
		{"scalar a=1e-18 m=2", d(1e-18), d(1, 1), d(1), i2},
		{"scalar a=1e-20 q=1e6", d(1e-20), d(-0.9667981925794609), d(1e6), d(1e-3)},
		{"scalar a=3.72e-44", d(3.72e-44), d(1), d(1), d(1)},
		{"c2d stiff dt=0.3 R=1e-6", expm(stiff, 0.3), b2, i2, d(1e-6)},
		{"c2d stiff dt=0.5", expm(stiff, 0.5), b2, i2, d(1)},
		{"c2d stiff dt=0.5 R=1e-6", expm(stiff, 0.5), b2, i2, d(1e-6)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := Dare(tc.A, tc.B, tc.Q, tc.R, nil)
			if err != nil {
				t.Fatal(err)
			}
			// ‖A‖ ≤ 3e-7, so X = Q + O(‖A‖²) and A−BK = O(‖A‖).
			var diff mat.Dense
			diff.Sub(res.X, tc.Q)
			if e := mat.Norm(&diff, 1) / mat.Norm(tc.Q, 1); e > 1e-12 {
				t.Errorf("‖X−Q‖/‖Q‖ = %.3g\nX = %v", e, mat.Formatted(res.X))
			}
			if rho := closedLoopRadius(t, tc.A, tc.B, res.K); !(rho < 1e-6) {
				t.Errorf("ρ(A−BK) = %.3g", rho)
			}
		})
	}
}

func TestKalmanNearZeroEigenvalue(t *testing.T) {
	one := mat.NewDense(1, 1, []float64{1})
	sys, err := New(mat.NewDense(1, 1, []float64{1.93e-22}), one, one, mat.NewDense(1, 1, nil), 100)
	if err != nil {
		t.Fatal(err)
	}
	w := mat.NewDense(1, 1, []float64{1e-3})
	res, err := Kalman(sys, w, w, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// P = Qn + O(a²).
	if p := res.X.At(0, 0); !(math.Abs(p-1e-3) <= 1e-15) {
		t.Errorf("P = %.17g, want 1e-3", p)
	}
}

func TestDareNearZeroEigenvalueFuzzCases(t *testing.T) {
	tests := []struct {
		name  string
		kind  uint8
		vals  [fzNVals]float64
		normC bool
		prob  func(s *fzSpec) (A, B, Q, R *mat.Dense)
	}{
		{
			// eig 5.1e-8: base returned a gain with ρ(A−BK) = 1.12.
			name:  "Dlqr unstable gain",
			kind:  0xcb,
			vals:  [fzNVals]float64{0, -2.1, -0.2, 75, 0.4, 0.975, 4.666666666666667, 0.3, -0.5, -4.8999999999999995, -1.4, 0.034722222222222224, 3, -133, 1, -10},
			normC: true,
			prob: func(s *fzSpec) (A, B, Q, R *mat.Dense) {
				n, m := s.N, s.M
				q := mqGram(s.C, s.P, n, true)
				for i := range n {
					q[i*n+i] += 1000
				}
				r := mat.NewDense(m, m, nil)
				for i := range m {
					r.Set(i, i, 1e-3)
				}
				return mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(n, n, q), r
			},
		},
		{
			// eig 3.2e-7: base returned ErrNoStabilizing on the Kalman dual.
			name:  "Kalman dual eig 3e-7",
			kind:  0xc5,
			vals:  [fzNVals]float64{97.69897000433602, 13, 0, 1, 0, 54, 0, 0, -0.125, -0.5228787452803376, 0.4, 0, 29, -30, -7, -39},
			normC: true,
			prob: func(s *fzSpec) (A, B, Q, R *mat.Dense) {
				n, m, p := s.N, s.M, s.P
				q := syMul(s.B, mqTranspose(s.B, n, m), n, m, n)
				for i := range q {
					q[i] *= 1000
				}
				return mqDense(n, n, mqTranspose(s.A, n, n)), mqDense(n, p, mqTranspose(s.C, p, n)), mqDense(n, n, q), mat.NewDense(1, 1, []float64{1e-3})
			},
		},
		{
			// eigs near 1e-11 and 1e-18: base returned ErrSingularR.
			name: "Q=CᵀC 4e12",
			kind: 0x7d,
			vals: [fzNVals]float64{0, 10, 0, 0, -17, 0, 82, -13.285714285714286, 0, 11, 0.3, 0, -10, -10, -10, -40},
			prob: func(s *fzSpec) (A, B, Q, R *mat.Dense) {
				n, m, p := s.N, s.M, s.P
				r := mqGram(s.D, p, m, true)
				for i := range m {
					r[i*m+i] += 1e-3
				}
				return mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(n, n, mqGram(s.C, p, n, true)), mqDense(m, m, r)
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := fzDecode(t, tc.kind, tc.vals)
			if tc.normC {
				s = syNormC(s)
			}
			A, B, Q, R := tc.prob(s)
			res, err := Dare(A, B, Q, R, nil)
			if err != nil {
				t.Fatal(err)
			}
			ref := dareFixedPoint(A, B, Q, R, 20000)
			var diff mat.Dense
			diff.Sub(res.X, ref)
			// cond(R) ≈ 1e6 in the third case limits X to about 1e-9.
			if e := mat.Norm(&diff, 1) / mat.Norm(ref, 1); !(e < 1e-7) {
				t.Errorf("‖X−Xref‖/‖Xref‖ = %.3g", e)
			}
			if rho := closedLoopRadius(t, A, B, res.K); !(rho < 1) {
				t.Errorf("ρ(A−BK) = %.6g", rho)
			}
		})
	}
}
