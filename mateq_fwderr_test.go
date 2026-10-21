package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestMqLyapOpMatchesDirect(t *testing.T) {
	const n = 3
	acl := []float64{-1.2, 0.7, 0.3, -0.4, -0.9, 1.1, 0.25, -0.6, -2}
	x := []float64{0.3, -1.1, 0.8, 2.2, 0.5, -0.7, -0.2, 1.4, 0.9}
	A, X := mqDense(n, n, acl), mqDense(n, n, x)
	for _, discrete := range []bool{false, true} {
		var want, t1 mat.Dense
		if discrete {
			t1.Mul(A.T(), X)
			want.Mul(&t1, A)
			want.Sub(&want, X)
		} else {
			t1.Mul(A.T(), X)
			want.Mul(X, A)
			want.Add(&want, &t1)
		}
		var got mat.Dense
		got.Mul(mqDense(n*n, n*n, mqLyapOp(acl, n, discrete)), mqDense(n*n, 1, x))
		w := mqRaw(&want)
		for i, g := range mqRaw(&got) {
			if math.Abs(g-w[i]) > 1e-14 {
				t.Fatalf("discrete=%v: op·vec(X)[%d] = %v, want %v", discrete, i, g, w[i])
			}
		}
	}
}

// TestMqRiccatiForwardErrBoundsOracle: on FuzzRiccati corpus cases whose
// X passes the residual gate (backward stable), the forward-error estimate
// of the PSD gate bounds the error of Care/Dare against a 320-bit Newton
// oracle. It assumes a rounding-level residual, so it does not replace the
// residual gate.
func TestMqRiccatiForwardErrBoundsOracle(t *testing.T) {
	cases := []struct {
		name string
		kind uint8
		v    [fzNVals]float64
	}{
		{"04679c87ed2ffeaf", 0xf1, [fzNVals]float64{75.83972937520639, 6, 3.6132640614817766, 6.3, 0.652072438895929, 6, 0, -847, 483, -1, -75.3, -166.8, -10.714285714285714, -10, -1, -36}},
		{"illcond_r_dare_cross", 0x27, [fzNVals]float64{0, 1.25, 0, -29, -17, 0, 0, -837, -2.8333333333333335, 11, 0.075, 0, -10, -10, -1.6666666666666667, -10}},
		{"tiny_x_care_0x82", 0x82, [fzNVals]float64{-0.03762874945799765, 4.875, 0, 6, -63, 0, 0, 0, 0, -0.05228787452803376, 0.4, -88, -10, 54, -10, -10}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := fzDecode(t, c.kind, c.v)
			n, m, p := s.N, s.M, s.P
			q := mqGram(s.C, p, n, true)
			r := mqGram(s.D, p, m, true)
			rr := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
			for i := range m {
				r[i*m+i] += rr
			}
			var sm []float64
			var S *mat.Dense
			if v := s.Vals[14]; v >= -3 && v <= 0 {
				var cd mat.Dense
				cd.Mul(mqDense(p, n, s.C).T(), mqDense(p, m, s.D))
				sm = mqRaw(&cd)
				S = mqDense(n, m, sm)
			}
			A, B, Q, R := mqDense(n, n, s.A), mqDense(n, m, s.B), mqDense(n, n, q), mqDense(m, m, r)
			solve, oracle := Care, kleinmanCareBig
			if s.Discrete {
				solve, oracle = Dare, hewerDareBig
			}
			res, err := solve(A, B, Q, R, &RiccatiOpts{S: S})
			if err != nil {
				t.Fatal(err)
			}
			x, k := mqRaw(res.X), mqRaw(res.K)
			var bk, acl mat.Dense
			bk.Mul(B, res.K)
			acl.Sub(A, &bk)
			_, _, _, rden := mqRiccatiResiduals(s, q, r, sm, x, k)
			fwd := mqRiccatiForwardErr(s, mqRaw(&acl), x, rden)
			e := relErr(oracle(A, B, Q, R, S, res.K, 10), res.X)
			if !(e <= fwd) || !(fwd <= 1e-4) {
				t.Errorf("oracle rel err %.3g, forward-error estimate %.3g", e, fwd)
			}
		})
	}
}
