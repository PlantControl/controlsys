package controlsys

import (
	"errors"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestHinfSyn_Simple(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	P.InputName = []string{"disturbance1", "disturbance2", "control"}
	P.OutputName = []string{"performance1", "performance2", "measurement"}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	kn, km, kp := res.K.Dims()
	if kn != 2 {
		t.Errorf("controller states: got %d, want 2", kn)
	}
	if km != 1 {
		t.Errorf("controller inputs: got %d, want 1", km)
	}
	if kp != 1 {
		t.Errorf("controller outputs: got %d, want 1", kp)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

	if res.X == nil {
		t.Error("X is nil")
	}
	if res.Y == nil {
		t.Error("Y is nil")
	}
	if res.GammaOpt <= 0 {
		t.Errorf("gamma should be positive, got %v", res.GammaOpt)
	}
	if !stringSlicesEqual(res.K.InputName, []string{"measurement"}) {
		t.Fatalf("controller input names = %v, want [measurement]", res.K.InputName)
	}
	if !stringSlicesEqual(res.K.OutputName, []string{"control"}) {
		t.Fatalf("controller output names = %v, want [control]", res.K.OutputName)
	}
}

func TestHinfSyn_DiscreteError(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, nil)

	P, err := New(A, B, C, D, 0.01)
	if err != nil {
		t.Fatal(err)
	}

	_, err = HinfSyn(P, 1, 1)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("got %v, want ErrWrongDomain", err)
	}
}

func TestHinfSyn_DimensionErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 3, nil)

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name  string
		nmeas int
		ncont int
	}{
		{"nmeas=0", 0, 1},
		{"ncont=0", 1, 0},
		{"nmeas too large", 4, 1},
		{"ncont too large", 1, 4},
		{"nmeas=p", 3, 1},
		{"ncont=m", 1, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := HinfSyn(P, tt.nmeas, tt.ncont)
			if !errors.Is(err, ErrInvalidPartition) {
				t.Errorf("got %v, want ErrInvalidPartition", err)
			}
		})
	}
}

func TestHinfSyn_GammaPositive(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	if res.GammaOpt <= 0 {
		t.Errorf("gamma must be positive, got %v", res.GammaOpt)
	}
	if math.IsInf(res.GammaOpt, 0) || math.IsNaN(res.GammaOpt) {
		t.Errorf("gamma must be finite, got %v", res.GammaOpt)
	}
}

func TestHinfSyn_NonNormalizedD12D21(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	// D12 = [0; 2] (not [0; 1]), D21 = [0.3 0.3] (not [0.1 0.1])
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 2,
		0.3, 0.3, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}
	if res.GammaOpt <= 0 {
		t.Errorf("gamma should be positive, got %v", res.GammaOpt)
	}
	if math.IsInf(res.GammaOpt, 0) || math.IsNaN(res.GammaOpt) {
		t.Errorf("gamma should be finite, got %v", res.GammaOpt)
	}
}

func TestHinfSyn_Stability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) >= -1e-10 {
			t.Errorf("closed-loop pole not in open left half-plane: %v", p)
		}
	}
}

// mixedSensitivityFeedthroughPlant weights S by 1/(s+0.1) and KS by 0.1 for G = (s+2)/(s+1), giving D22 = -1.
func mixedSensitivityFeedthroughPlant(t *testing.T) *System {
	t.Helper()
	P, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, -1, -0.1}),
		mat.NewDense(2, 2, []float64{0, 1, 1, -1}),
		mat.NewDense(3, 2, []float64{0, 1, 0, 0, -1, 0}),
		mat.NewDense(3, 2, []float64{0, 0, 0, 0.1, 1, -1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

func withoutD22(t *testing.T, P *System, nmeas, ncont int) *System {
	t.Helper()
	_, m, p := P.Dims()
	D := mat.DenseCopyOf(P.D)
	for i := p - nmeas; i < p; i++ {
		for j := m - ncont; j < m; j++ {
			D.Set(i, j, 0)
		}
	}
	P0, err := New(P.A, P.B, P.C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	return P0
}

func closedLoop(t *testing.T, P, K *System, nmeas, ncont int) *System {
	t.Helper()
	_, m, p := P.Dims()
	cl, err := LFT(P, K, m-ncont, p-nmeas)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Fatalf("closed loop unstable: poles %v", poles)
	}
	return cl
}

func TestHinfSyn_ClosedLoopMeetsGamma(t *testing.T) {
	plant := func(A, B, C, D []float64, n, m, p int) *System {
		P, err := New(mat.NewDense(n, n, A), mat.NewDense(n, m, B), mat.NewDense(p, n, C), mat.NewDense(p, m, D), 0)
		if err != nil {
			t.Fatal(err)
		}
		return P
	}
	cases := []struct {
		name string
		P    *System
	}{
		{"orthogonal", plant(
			[]float64{0, 1, -2, -3},
			[]float64{1, 0, 0, 0, 0, 1},
			[]float64{1, 0, 0, 0, 1, 0},
			[]float64{0, 0, 0, 0, 0, 1, 0, 1, 0},
			2, 3, 3)},
		{"B1D21 cross term", plant(
			[]float64{0, 1, -2, -3},
			[]float64{1, 0, 0, 0, 1, 1},
			[]float64{1, 0, 0, 0, 1, 0},
			[]float64{0, 0, 0, 0, 0, 1, 0.1, 0.1, 0},
			2, 3, 3)},
		{"D12C1 cross term", plant(
			[]float64{0, 1, -2, -3},
			[]float64{1, 0, 0, 0, 1, 1},
			[]float64{1, 0, 0.5, 1, 1, 0},
			[]float64{0, 0, 0, 0, 0, 1, 0.1, 0.1, 0},
			2, 3, 3)},
		{"mixed sensitivity", withoutD22(t, mixedSensitivityFeedthroughPlant(t), 1, 1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := HinfSyn(tc.P, 1, 1)
			if err != nil {
				t.Fatal(err)
			}
			norm, _, err := HinfNorm(closedLoop(t, tc.P, res.K, 1, 1))
			if err != nil {
				t.Fatal(err)
			}
			if norm > res.GammaOpt*(1+1e-6) {
				t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v", norm, res.GammaOpt)
			}
		})
	}
}
