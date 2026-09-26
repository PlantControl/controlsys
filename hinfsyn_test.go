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

func assertHinfSynD22(t *testing.T, P *System, nmeas, ncont int) {
	t.Helper()
	res, err := HinfSyn(P, nmeas, ncont)
	if err != nil {
		t.Fatal(err)
	}
	cl := closedLoop(t, P, res.K, nmeas, ncont)
	norm, _, err := HinfNorm(cl)
	if err != nil {
		t.Fatal(err)
	}
	if norm > res.GammaOpt*(1+1e-6) {
		t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v", norm, res.GammaOpt)
	}

	ref, err := HinfSyn(withoutD22(t, P, nmeas, ncont), nmeas, ncont)
	if err != nil {
		t.Fatal(err)
	}
	if res.GammaOpt != ref.GammaOpt {
		t.Fatalf("gamma %v, want D22-free gamma %v", res.GammaOpt, ref.GammaOpt)
	}
	refNorm, _, err := HinfNorm(closedLoop(t, withoutD22(t, P, nmeas, ncont), ref.K, nmeas, ncont))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-refNorm) > 1e-8*refNorm {
		t.Fatalf("closed-loop Hinf norm %v, want D22-free norm %v", norm, refNorm)
	}
}

func TestHinfSyn_D22LoopShiftMixedSensitivity(t *testing.T) {
	assertHinfSynD22(t, mixedSensitivityFeedthroughPlant(t), 1, 1)
}

func TestHinfSyn_D22LoopShiftMIMO(t *testing.T) {
	P, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -2, 1,
			0.3, 0, -0.5,
		}),
		mat.NewDense(3, 4, []float64{
			1, 0, 0, 1,
			0, 1, 0, 0.5,
			0, 0, 1, -1,
		}),
		mat.NewDense(4, 3, []float64{
			1, 0, 0,
			0, 0, 0,
			0, 1, 0,
			0, 0, 1,
		}),
		mat.NewDense(4, 4, []float64{
			0, 0, 0, 0,
			0, 0, 0, 1,
			0.2, 0.1, 0, 0.7,
			0, 0.1, 0.3, -0.4,
		}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertHinfSynD22(t, P, 2, 1)
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

func hinfD11Plant(t testing.TB, n, m, p int, A, B, C, D []float64) *System {
	t.Helper()
	P, err := New(mat.NewDense(n, n, A), mat.NewDense(n, m, B), mat.NewDense(p, n, C), mat.NewDense(p, m, D), 0)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// sisoBiproperMixedSensitivityPlant weights S = 1/(1+G) for G = 1/(s+1) by
// the biproper W1 = (0.5s+0.5)/(s+0.005) and KS by 0.1, so D11 = [0.5; 0].
func sisoBiproperMixedSensitivityPlant(t testing.TB) *System {
	return hinfD11Plant(t, 2, 2, 3,
		[]float64{-1, 0, -1, -0.005},
		[]float64{0, 1, 1, 0},
		[]float64{-0.5, 0.4975, 0, 0, -1, 0},
		[]float64{0.5, 0, 0, 0.1, 1, 0})
}

func mimoBiproperMixedSensitivityPlant(t testing.TB) *System {
	return hinfD11Plant(t, 4, 4, 6,
		[]float64{
			-1, 0.5, 0, 0,
			-0.3, -2, 0, 0,
			-1, 0, -0.005, 0,
			-0.4, -1, 0, -0.01,
		},
		[]float64{
			0, 0, 1, 0.2,
			0, 0, 0, 1,
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		[]float64{
			-0.5, 0, 0.4975, 0,
			-0.16, -0.4, 0, 0.2,
			0, 0, 0, 0,
			0, 0, 0, 0,
			-1, 0, 0, 0,
			-0.4, -1, 0, 0,
		},
		[]float64{
			0.5, 0, 0, 0,
			0, 0.4, 0, 0,
			0, 0, 0.1, 0,
			0, 0, 0, 0.1,
			1, 0, 0, 0,
			0, 1, 0, 0,
		})
}

func genericD11Plant(t testing.TB, d22 float64) *System {
	return hinfD11Plant(t, 3, 3, 3,
		[]float64{0.5, 1, 0, -1, -0.3, 0.4, 0.2, 0, -1.2},
		[]float64{1, 0, 0, 0, 0.5, 1, 0.3, 1, 0.5},
		[]float64{1, 0, 0.5, 0, 1, -0.2, 0.7, 0, 1},
		[]float64{0.2, -0.1, 0.3, 0.4, 0.3, 1, 0.5, 1, d22})
}

// Reference gammas are SLICOT SB10AD optima (bisection, gtol 1e-10) whose
// controllers were verified to attain them.
func TestHinfSyn_D11ClosedLoopMeetsOptimalGamma(t *testing.T) {
	cases := []struct {
		name         string
		P            *System
		nmeas, ncont int
		want         float64
	}{
		{"siso biproper W1", sisoBiproperMixedSensitivityPlant(t), 1, 1, 0.5098041291076711},
		{"mimo biproper W1", mimoBiproperMixedSensitivityPlant(t), 2, 2, 0.5109338104884777},
		{"generic", genericD11Plant(t, 0), 1, 1, 3.5096256649205544},
		{"generic D22", genericD11Plant(t, 0.6), 1, 1, 3.5096256649205544},
		{"square D12", hinfD11Plant(t, 2, 3, 2,
			[]float64{-0.5, 1, -2, -1},
			[]float64{1, 0, 0, 0, 1, 1},
			[]float64{1, 0.3, 1, 1},
			[]float64{0.4, 0.2, 1, 0, 1, 0}), 1, 1, 0.7531813936485063},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := HinfSyn(tc.P, tc.nmeas, tc.ncont)
			if err != nil {
				t.Fatal(err)
			}
			norm, _, err := HinfNorm(closedLoop(t, tc.P, res.K, tc.nmeas, tc.ncont))
			if err != nil {
				t.Fatal(err)
			}
			if norm > res.GammaOpt*(1+1e-6) {
				t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v", norm, res.GammaOpt)
			}
			if math.Abs(res.GammaOpt-tc.want) > 1e-5*tc.want {
				t.Fatalf("gamma %v, want optimum %v", res.GammaOpt, tc.want)
			}
		})
	}
}

// D11 lives only in the block reachable by both u and y, so a controller
// feedthrough cancels it and the optimum lies below sigma_max(D11) = 2.
func TestHinfSyn_D11CancelledByFeedthrough(t *testing.T) {
	P := hinfD11Plant(t, 2, 3, 3,
		[]float64{-0.5, 1, -2, -1},
		[]float64{1, 0, 0, 0, 1, 1},
		[]float64{1, 0, 0, 0.5, 1, 1},
		[]float64{0, 0, 0, 0, 2, 1, 0, 1, 0})
	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if res.GammaOpt >= 1 {
		t.Fatalf("gamma %v, want below 1", res.GammaOpt)
	}
	norm, _, err := HinfNorm(closedLoop(t, P, res.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	if norm > res.GammaOpt*(1+1e-6) {
		t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v", norm, res.GammaOpt)
	}
	if norm < res.GammaOpt*(1-1e-3) {
		t.Fatalf("closed-loop Hinf norm %v far below gamma %v; gamma not optimal", norm, res.GammaOpt)
	}
}
