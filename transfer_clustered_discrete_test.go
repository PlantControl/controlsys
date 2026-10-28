package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// clusteredMatchedDiscrete is matched C2D of a ZPK model at a sample time
// that maps every pole into a cluster within 2e-2 of z = 1.
func clusteredMatchedDiscrete(t *testing.T, zeros, poles []complex128, k, dt float64) *System {
	t.Helper()
	g, err := NewZPK(zeros, poles, k, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := g.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	d, err := res.Sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// MMAAZJ: on these models the observability cut drops modes whose last
// Hessenberg coupling is at rounding level in the Dgebal basis, and the TF
// misses the response near z = 1. The full-order den cannot do better: its
// coefficients, even rounded correctly from the exact poles, have den(1) and
// root moduli dominated by rounding (literal: den(1) 2.3e-17 exact, -1.6e-15
// rounded, roots up to 1.0027; second model roots up to 1.0041), so a
// full-order TF is unstable and no float64 TF meets TF(1) = EvalFr(1). The
// kept TF must stay stable and match the 320-bit response away from the
// cluster.
func TestTransferFunctionClusteredDiscreteStaysStableAndMatchesAwayFromCluster(t *testing.T) {
	cases := []struct {
		name         string
		zeros, poles []complex128
		k, dt        float64
	}{
		{"literal", []complex128{-0.5001, -1.0347},
			[]complex128{-0.4566 + 1.814i, -0.4566 - 1.814i, -0.7451 + 1.4472i, -0.7451 - 1.4472i, -1.5689, -1.6005}, 1.6263, 0.001},
		{"eight poles", []complex128{-2.1046125531510307},
			[]complex128{
				-2.068874567619866 + 0.3575300061345922i, -2.068874567619866 - 0.3575300061345922i,
				-0.5251289361561816 + 1.6007282981397544i, -0.5251289361561816 - 1.6007282981397544i,
				-1.5949182540218703 + 0.298720258881107i, -1.5949182540218703 - 0.298720258881107i,
				-0.8573159275781227 + 0.21103191696387205i, -0.8573159275781227 - 0.21103191696387205i,
			}, 1.7933080296139967, 0.01},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := clusteredMatchedDiscrete(t, tc.zeros, tc.poles, tc.k, tc.dt)
			res, err := d.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			roots, err := Poly(res.TF.Den[0]).Roots()
			if err != nil {
				t.Fatal(err)
			}
			for _, r := range roots {
				if cmplx.Abs(r) >= 1 {
					t.Errorf("den root %v outside the unit circle; model is stable", r)
				}
			}
			o := newStepOraclePencil(d)
			for _, th := range []float64{0.3, 1, 3} {
				z := cmplx.Rect(1, th)
				want := matchedOracleTransfer(o, z)
				got, err := res.TF.Eval(z)
				if err != nil {
					t.Fatal(err)
				}
				if e := cmplx.Abs(got[0][0]-want) / cmplx.Abs(want); !(e <= 1e-6) || math.IsNaN(e) {
					t.Errorf("θ=%g: TF %v, oracle %v, rel err %.3g", th, got[0][0], want, e)
				}
			}
		})
	}
}
