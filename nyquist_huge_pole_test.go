package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// X34YEB: a well-conditioned unstable pole beside a pole of ~1e10 was put on
// the boundary by a band of sqrt(eps) times the spectral radius.
func TestNyquist_RHPPolesBesideHugePole(t *testing.T) {
	similar := func(l1, l2 float64) []float64 {
		// T = [1 2; 0.5 3], A = T·diag(l1, l2)·T⁻¹.
		const det = 1*3 - 2*0.5
		return []float64{
			(3*l1 - 1*l2) / det, (-2*l1 + 2*l2) / det,
			(1.5*l1 - 1.5*l2) / det, (-1*l1 + 3*l2) / det,
		}
	}
	cases := []struct {
		name     string
		a, b, c  []float64
		dt       float64
		outDelay float64
		rhp      int
	}{
		{
			name: "symmetric 1.0295 beside 6.9e10",
			a:    []float64{3.950378993621064e+10, 3.411907393429129e+10, 3.4119073934291298e+10, 2.9468342356123444e+10},
			b:    []float64{0.511703992453149, 0.9647326178866098},
			c:    []float64{-259817.35621375585, 997798.2791785806},
			dt:   0.7943282347242815,
			rhp:  2,
		},
		{
			name:     "nonsymmetric 1.0347 beside 5.4e12, output delay",
			a:        similar(1.0347, 5.4e12),
			b:        []float64{1, -0.5},
			c:        []float64{0.25, 1e-12},
			dt:       1e-4,
			outDelay: 10,
			rhp:      2,
		},
		{
			name: "nonsymmetric 0.9 beside 1e10",
			a:    similar(0.9, 1e10),
			b:    []float64{1, -0.5},
			c:    []float64{0.25, 1e-10},
			dt:   0.1,
			rhp:  1,
		},
	}
	for _, tc := range cases {
		sys, err := New(mat.NewDense(2, 2, tc.a), mat.NewDense(2, 1, tc.b), mat.NewDense(1, 2, tc.c), mat.NewDense(1, 1, nil), tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		if tc.outDelay > 0 {
			if err := sys.SetOutputDelay([]float64{tc.outDelay}); err != nil {
				t.Fatal(err)
			}
		}
		res, err := sys.Nyquist(nil, 0)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if res.RHPPoles != tc.rhp {
			t.Errorf("%s: RHPPoles = %d, want %d", tc.name, res.RHPPoles, tc.rhp)
		}
		z, ok := fzNyqClosedLoopRHP(fzPidPlantOf(t, sys))
		if !ok {
			t.Fatalf("%s: exact closed-loop count undecided", tc.name)
		}
		if res.RHPZerosCL != z {
			t.Errorf("%s: RHPZerosCL = %d (N %d, P %d), closed-loop unstable poles %d", tc.name, res.RHPZerosCL, res.Encirclements, res.RHPPoles, z)
		}
	}
}

// A pole exactly on the boundary beside a huge pole stays on the boundary.
func TestNyquist_BoundaryPoleBesideHugePole(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{1, 1e3, 0, 1e10}), mat.NewDense(2, 1, []float64{0.5, 1}), mat.NewDense(1, 2, []float64{0.1, 1e-10}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.RHPPoles != 1 {
		t.Errorf("RHPPoles = %d, want 1 (z = 1 on the boundary, 1e10 outside)", res.RHPPoles)
	}
}
