package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// checkD2CMatchedAgainstOracle compares matched D2C c of the discrete sys
// with MATLAB d2c(...,'matched') evaluated at stepOraclePrec bits: poles
// log(eig(A))/T, zeros log(z)/T of the finite zeros other than −1, and the
// gain that keeps the DC value H(1).
func checkD2CMatchedAgainstOracle(t *testing.T, name string, sys, c *System) {
	t.Helper()
	dt := sys.Dt
	disc := newStepOraclePencil(sys)
	starts, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	modes, ok := newStepOracleModes(sys, starts)
	if !ok {
		t.Fatalf("%s: oracle poles unresolved", name)
	}
	zstarts, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	var zeros []complex128
	for _, z := range matchedOracleZeros(t, disc, zstarts) {
		if cmplx.Abs(z+1) > 1e-9 {
			zeros = append(zeros, cmplx.Log(z)/complex(dt, 0))
		}
	}
	poles := make([]complex128, len(modes.poles))
	for i, p := range modes.poles {
		poles[i] = cmplx.Log(p) / complex(dt, 0)
	}
	dc := matchedOracleTransfer(disc, 1)

	got, err := c.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(poles) {
		t.Fatalf("%s dt=%g: %d poles, want %d", name, dt, len(got), len(poles))
	}
	for _, l := range poles {
		best := math.Inf(1)
		for _, q := range got {
			best = math.Min(best, cmplx.Abs(q-l))
		}
		if !(best <= 1e-8*cmplx.Abs(l)) {
			t.Errorf("%s dt=%g: no pole near log(λ)/T = %v: nearest %.3g away", name, dt, l, best)
		}
	}
	cont := newStepOraclePencil(c)
	for _, s := range []complex128{0, 0.05i, 0.3i, 1i, 3i, -0.1 + 0.5i, 2 + 1i} {
		want := dc
		for _, z := range zeros {
			want *= 1 - s/z
		}
		for _, p := range poles {
			want /= 1 - s/p
		}
		g := matchedOracleTransfer(cont, s)
		if rel := cmplx.Abs(g-want) / math.Max(cmplx.Abs(want), 1e-12); !(rel <= 1e-7) {
			t.Errorf("%s dt=%g: G(%v) = %v, matched ZPK %v (rel %.3g)", name, dt, s, g, want, rel)
		}
	}
	if stable, err := c.IsStable(); err != nil || !stable {
		t.Errorf("%s dt=%g: IsStable = %v, %v; want stable", name, dt, stable, err)
	}
}

// Matched D2C poles are log(eig(A))/T to rounding, as MATLAB
// d2c(...,'matched'); polynomial roots of the transfer function and a
// companion realization turned this stable cluster unstable, with relative
// pole errors of 5.8 at T = 0.01 and 88 at T = 0.001 (UYCHOO).
func TestD2CMatchedClusteredPolesMapExactly(t *testing.T) {
	cs := matchedClusterDiagonal(t)
	for _, dt := range []float64{0.1, 0.01, 0.001} {
		d := cs.Copy()
		d.Dt = dt
		for i := range 8 {
			d.A.Set(i, i, math.Exp(cs.A.At(i, i)*dt))
		}
		c, err := d.D2C(D2COptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatal(err)
		}
		checkD2CMatchedAgainstOracle(t, "diag cluster", d, c)
		got, _ := c.Poles()
		for k := 1; k <= 8; k++ {
			want := -0.2 * float64(k)
			best := math.Inf(1)
			for _, q := range got {
				best = math.Min(best, cmplx.Abs(q-complex(want, 0)))
			}
			if !(best <= 1e-9*math.Abs(want)) {
				t.Errorf("dt=%g: no pole near %g: nearest %.3g away", dt, want, best)
			}
		}
	}
}

// A non-normal similarity of clustered real poles and pole pairs keeps the
// eig(A) route.
func TestD2CMatchedRotatedClusterFollowsMatchedZPK(t *testing.T) {
	const n = 8
	rng := rand.New(rand.NewPCG(49, 8))
	for _, dt := range []float64{0.01, 0.001} {
		blk := mat.NewDense(n, n, nil)
		for i, p := range []float64{-0.3, -0.5, -0.7, -0.9} {
			blk.Set(i, i, math.Exp(p*dt))
		}
		for k, pole := range []complex128{complex(-0.4, 0.6), complex(-0.8, 1.1)} {
			e := cmplx.Exp(pole * complex(dt, 0))
			i := 4 + 2*k
			blk.Set(i, i, real(e))
			blk.Set(i, i+1, imag(e))
			blk.Set(i+1, i, -imag(e))
			blk.Set(i+1, i+1, real(e))
		}
		T := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				v := 0.3 * rng.NormFloat64()
				if i == j {
					v += 1
				}
				T.Set(i, j, v)
			}
		}
		var Ti, tmp, A mat.Dense
		if err := Ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		tmp.Mul(T, blk)
		A.Mul(&tmp, &Ti)
		bm := mat.NewDense(n, 1, []float64{1, 1, 1, 1, 0, 1, 0, 1})
		cm := mat.NewDense(1, n, []float64{0.4, 0.7, 0.5, 0.9, 0.3, -0.2, 0.6, 0.1})
		var B, C mat.Dense
		B.Mul(T, bm)
		C.Mul(cm, &Ti)
		d, err := New(&A, &B, &C, mat.NewDense(1, 1, nil), dt)
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.D2C(D2COptions{Method: C2DMethodMatched})
		if err != nil {
			t.Fatal(err)
		}
		checkD2CMatchedAgainstOracle(t, "rotated cluster", d, c)
	}
}

// D2C undoes matched C2D: the zeros at −1 that C2D adds up to relative
// degree one are dropped and the clustered poles come back. The transfer
// function of the discrete model cancelled two poles of the dt = 0.001
// cluster (sweep seed 431) and lost one (seed 307), and tzero spreads the
// five- and six-fold zeros at −1 by up to 0.08 (seeds 494, 307; UYCHOO).
func TestD2CMatchedInvertsMatchedC2D(t *testing.T) {
	cases := []struct {
		name         string
		zeros, poles []complex128
		k            float64
		dts          []float64
	}{
		{"real cluster", nil, []complex128{-0.2, -0.4, -0.6, -0.8, -1}, 0.384, []float64{0.01, 0.001}},
		{"near cancellation", []complex128{-0.5001, -1.0347}, []complex128{-0.4566 + 1.814i, -0.4566 - 1.814i, -0.7451 + 1.4472i, -0.7451 - 1.4472i, -1.5689, -1.6005}, 1.6263, []float64{0.001}},
		{"five-fold zero at -1", []complex128{-1.1729}, []complex128{-0.1531, -0.9583, -1.4969 + 1.397i, -1.4969 - 1.397i, -1.6828 + 1.9039i, -1.6828 - 1.9039i, -0.6914}, 1.3863, []float64{0.001}},
		{"six-fold zero at -1", []complex128{-2.1046}, []complex128{-2.0689 + 0.3575i, -2.0689 - 0.3575i, -0.5251 + 1.6007i, -0.5251 - 1.6007i, -1.5949 + 0.2987i, -1.5949 - 0.2987i, -0.8573 + 0.211i, -0.8573 - 0.211i}, 1.7933, []float64{0.01}},
	}
	for _, tc := range cases {
		g, err := NewZPK(tc.zeros, tc.poles, tc.k, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := g.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range tc.dts {
			d, err := res.Sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
			if err != nil {
				t.Fatal(err)
			}
			c, err := d.D2C(D2COptions{Method: C2DMethodMatched})
			if err != nil {
				t.Fatalf("%s dt=%g: %v", tc.name, dt, err)
			}
			got, err := c.Poles()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.poles) {
				t.Errorf("%s dt=%g: %d poles, want %d", tc.name, dt, len(got), len(tc.poles))
			}
			for _, p := range tc.poles {
				best := math.Inf(1)
				for _, q := range got {
					best = math.Min(best, cmplx.Abs(q-p))
				}
				if !(best <= 1e-8*cmplx.Abs(p)) {
					t.Errorf("%s dt=%g: no pole near %v: nearest %.3g away", tc.name, dt, p, best)
				}
			}
			cont := newStepOraclePencil(c)
			for _, s := range []complex128{0, 0.3i, 1i, -0.1 + 0.5i, 3i} {
				want := complex(tc.k, 0)
				for _, z := range tc.zeros {
					want *= s - z
				}
				for _, p := range tc.poles {
					want /= s - p
				}
				if got := matchedOracleTransfer(cont, s); !(cmplx.Abs(got-want) <= 1e-7*cmplx.Abs(want)) {
					t.Errorf("%s dt=%g: G(%v) = %v, want %v", tc.name, dt, s, got, want)
				}
			}
			if stable, err := c.IsStable(); err != nil || !stable {
				t.Errorf("%s dt=%g: IsStable = %v, %v; want stable", tc.name, dt, stable, err)
			}
		}
	}
}
