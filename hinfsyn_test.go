package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
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
	cl, err := LFT(P, K, LFTFeedback{Nu: ncont, Ny: nmeas})
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
			if math.Abs(res.GammaOpt-tc.want) > (1e-5+hinfControllerBackoff)*tc.want {
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

// Plant generators ported from process-lab (randomModelDesignPlant and
// randomHinfPlant); they differ only in draw order.
func randomModelDesignTestPlant(t testing.TB, n, m int, seed int64) *System {
	t.Helper()
	random := rand.New(rand.NewSource(seed))
	A, B, C := mat.NewDense(n, n, nil), mat.NewDense(n, m, nil), mat.NewDense(m, n, nil)
	for i := range n {
		for j := range n {
			if i == j {
				A.Set(i, j, -1-float64(i)/2)
			} else {
				A.Set(i, j, 0.1*random.NormFloat64()/math.Sqrt(float64(n)))
			}
		}
	}
	for i := range n {
		for j := range m {
			B.Set(i, j, random.NormFloat64())
		}
	}
	for i := range m {
		for j := range n {
			C.Set(i, j, random.NormFloat64())
		}
	}
	G, err := New(A, B, C, mat.NewDense(m, m, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return G
}

func randomHinfTestPlant(t testing.TB, n, m int, seed int64) *System {
	t.Helper()
	random := rand.New(rand.NewSource(seed))
	A, B, C := mat.NewDense(n, n, nil), mat.NewDense(n, m, nil), mat.NewDense(m, n, nil)
	for i := range n {
		for j := range n {
			if i == j {
				A.Set(i, j, -1-float64(i)/2)
			} else {
				A.Set(i, j, 0.1*random.NormFloat64()/math.Sqrt(float64(n)))
			}
		}
		for j := range m {
			B.Set(i, j, random.NormFloat64())
			C.Set(j, i, random.NormFloat64())
		}
	}
	G, err := New(A, B, C, mat.NewDense(m, m, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return G
}

// mixedSensitivityPlant augments the square, strictly proper G with
// W1 = k1/(s+a1) + d1 and W2 = w2 per channel: w = r, z = [W1 e; W2 u],
// y = e = r - G u.
func mixedSensitivityPlant(t testing.TB, G *System, k1, a1, d1, w2 float64) *System {
	t.Helper()
	ng, m, _ := G.Dims()
	n := ng + m
	A, B := mat.NewDense(n, n, nil), mat.NewDense(n, 2*m, nil)
	C, D := mat.NewDense(3*m, n, nil), mat.NewDense(3*m, 2*m, nil)
	setBlock(A, 0, 0, G.A)
	setBlock(B, 0, m, G.B)
	for ch := range m {
		w := ng + ch
		A.Set(w, w, -a1)
		B.Set(w, ch, 1)
		C.Set(ch, w, k1)
		D.Set(ch, ch, d1)
		D.Set(m+ch, m+ch, w2)
		D.Set(2*m+ch, ch, 1)
		for j := range ng {
			A.Set(w, j, -G.C.At(ch, j))
			C.Set(ch, j, -d1*G.C.At(ch, j))
			C.Set(2*m+ch, j, -G.C.At(ch, j))
		}
	}
	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// The order-50 case also covers HinfNorm, whose batched sweep lost the peak
// of that closed loop (1e-3 excess). Its near-optimal controller leaves a
// closed loop so ill-conditioned that sigma_max carries relative errors of a
// few 1e-6 that vary with build flags (-race), hence the looser tolerance.
func TestHinfSyn_RandomMixedSensitivityMeetsGamma(t *testing.T) {
	cases := []struct {
		name     string
		G        *System
		d1       float64
		minGamma float64
		tol      float64
	}{
		{"rank-deficient 4x4 strict W1", randomModelDesignTestPlant(t, 2, 4, 7), 0, 100, 1e-6},
		{"rank-deficient 4x4 biproper W1", randomModelDesignTestPlant(t, 2, 4, 7), 0.5, 100, 1e-6},
		{"order 50 strict W1", randomHinfTestPlant(t, 50, 2, 7), 0, 0, 1e-5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, m, _ := tc.G.Dims()
			P := mixedSensitivityPlant(t, tc.G, 1, 0.01, tc.d1, 0.1)
			res, err := HinfSyn(P, m, m)
			if err != nil {
				t.Fatal(err)
			}
			norm, _, err := HinfNorm(closedLoop(t, P, res.K, m, m))
			if err != nil {
				t.Fatal(err)
			}
			if norm > res.GammaOpt*(1+tc.tol) {
				t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v", norm, res.GammaOpt)
			}
			if res.GammaOpt < tc.minGamma*(1-1e-6) {
				t.Fatalf("gamma %v below the lower bound %v", res.GammaOpt, tc.minGamma)
			}
		})
	}
}

// Hamiltonian eigenvalues near +-1e-3 and +-1e7: the slow modes must not be
// judged on-axis against the fast ones. The 1e10 spread limits Riccati
// accuracy to about 1e-5, so gamma is checked against SLICOT SB10AD to 1e-4.
func TestHinfSyn_StiffPlant(t *testing.T) {
	for _, tc := range []struct{ d11, want float64 }{
		{0, 0.6324554030356921},
		{0.3, 0.7637279085111218},
	} {
		P := hinfD11Plant(t, 2, 3, 3,
			[]float64{-1e-3, 1e-3, 0, -1e7},
			[]float64{1e-3, 0, 1e-3, 0, 1e7, 1e7},
			[]float64{1, 0, 0, 0, 1, 0},
			[]float64{tc.d11, 0, 0, 0, 0, 1, 0, 1, 0})
		res, err := HinfSyn(P, 1, 1)
		if err != nil {
			t.Fatalf("D11 = %v: %v", tc.d11, err)
		}
		norm, _, err := HinfNorm(closedLoop(t, P, res.K, 1, 1))
		if err != nil {
			t.Fatal(err)
		}
		if norm > res.GammaOpt*(1+1e-4) {
			t.Fatalf("D11 = %v: closed-loop Hinf norm %v exceeds gamma %v", tc.d11, norm, res.GammaOpt)
		}
		if math.Abs(res.GammaOpt-tc.want) > (1e-4+hinfControllerBackoff)*tc.want {
			t.Fatalf("D11 = %v: gamma %v, want optimum %v", tc.d11, res.GammaOpt, tc.want)
		}
	}
}

// Hamiltonian modes -5e-7 +- j are lightly damped but well conditioned, so
// they must not be taken for imaginary-axis eigenvalues. The optimum is 0.
func TestHinfSyn_LightlyDampedModes(t *testing.T) {
	for _, d11 := range []float64{0, 0.3} {
		P := hinfD11Plant(t, 2, 2, 2,
			[]float64{-5e-7, -1, 1, -5e-7},
			[]float64{0, 0, 0, 0},
			[]float64{0, 0, 0, 0},
			[]float64{d11, 1, 1, 0})
		res, err := HinfSyn(P, 1, 1)
		if err != nil {
			t.Fatalf("D11 = %v: %v", d11, err)
		}
		norm, _, err := HinfNorm(closedLoop(t, P, res.K, 1, 1))
		if err != nil {
			t.Fatal(err)
		}
		if norm > max(res.GammaOpt*(1+1e-6), 1e-12) {
			t.Fatalf("D11 = %v: closed-loop Hinf norm %v exceeds gamma %v", d11, norm, res.GammaOpt)
		}
	}
}

func TestHinfBisect_ZeroOptimumTerminates(t *testing.T) {
	calls := 0
	gamma, err := hinfBisect(0, func(float64) bool { calls++; return true })
	if err != nil {
		t.Fatal(err)
	}
	if gamma > hinfGammaAbsTol || calls > 64 {
		t.Fatalf("gamma %v after %d feasibility calls", gamma, calls)
	}
}

// The verified back-off (6WORAF) must leave problems with a positive optimum
// within 2e-4 of it, continuous and Tustin-discretized (which keeps the
// optimum), with ‖T_zw‖∞ ≤ GammaOpt exactly. Optima are SLICOT SB10AD's.
func TestHinfSyn_PositiveOptimumGammaNearOptimal(t *testing.T) {
	cases := []struct {
		name         string
		P            *System
		nmeas, ncont int
		want         float64
	}{
		{"siso biproper W1", sisoBiproperMixedSensitivityPlant(t), 1, 1, 0.5098041291076711},
		{"mimo biproper W1", mimoBiproperMixedSensitivityPlant(t), 2, 2, 0.5109338104884777},
		{"generic D22", genericD11Plant(t, 0.6), 1, 1, 3.5096256649205544},
	}
	for _, tc := range cases {
		Pd, err := tc.P.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
		if err != nil {
			t.Fatal(err)
		}
		for _, P := range []*System{tc.P, Pd} {
			res, err := HinfSyn(P, tc.nmeas, tc.ncont)
			if err != nil {
				t.Fatalf("%s Ts=%g: %v", tc.name, P.Dt, err)
			}
			if res.GammaOpt < tc.want*(1-1e-6) || res.GammaOpt > tc.want*(1+2e-4) {
				t.Errorf("%s Ts=%g: gamma %.12g, want within 2e-4 above optimum %.12g", tc.name, P.Dt, res.GammaOpt, tc.want)
			}
			cl, err := LFT(P, res.K, LFTFeedback{Nu: tc.ncont, Ny: tc.nmeas})
			if err != nil {
				t.Fatal(err)
			}
			norm, _, err := HinfNorm(cl)
			if err != nil {
				t.Fatal(err)
			}
			if norm > res.GammaOpt {
				t.Errorf("%s Ts=%g: ‖T_zw‖∞ = %.12g exceeds gamma %.12g", tc.name, P.Dt, norm, res.GammaOpt)
			}
		}
	}
}

// Mixed sensitivity of 0.01/(s+0.01) with default weights (process-lab
// 3FFQZL): X grows without bound as gamma nears the optimum, so a central
// controller built at the bisection edge overshot its gamma on every arch.
func TestHinfSyn_ControllerMeetsGammaNearSingularEdge(t *testing.T) {
	P, err := New(
		mat.NewDense(2, 2, []float64{-0.01, 0, -0.01, -0.00004320073460981398}),
		mat.NewDense(2, 2, []float64{0, 1, 1, 0}),
		mat.NewDense(3, 2, []float64{-0.005, 0.004298473093676491, 0, 0, -0.01, 0}),
		mat.NewDense(3, 2, []float64{0.5, 0, 0, 0.1, 1, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	norm, _, err := HinfNorm(closedLoop(t, P, res.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	if norm > res.GammaOpt {
		t.Fatalf("closed-loop Hinf norm %v exceeds gamma %v by %.3g", norm, res.GammaOpt, norm/res.GammaOpt-1)
	}
	if math.Abs(res.GammaOpt-0.5074089765548706) > 2*hinfControllerBackoff*0.5074089765548706 {
		t.Fatalf("gamma %v, want near optimum 0.50741", res.GammaOpt)
	}
}

// MCOU37: D11 = 0 with D12 = 0 must be rejected up front, not bisected to 1e12.
func TestHinfSyn_RankDeficientD12D11Zero(t *testing.T) {
	G := mixsynTF(t, []float64{200}, []float64{0.025, 1.0025, 10.1, 1}, 0)
	W1 := mixsynTF(t, []float64{1}, []float64{1, 1}, 0)
	P, err := Augw(G, W1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = HinfSyn(P, 1, 1)
	if !errors.Is(err, ErrInvalidPartition) || errors.Is(err, ErrGammaNotAchievable) {
		t.Fatalf("HinfSyn err = %v, want ErrInvalidPartition", err)
	}
}

func TestHinfSyn_RankDeficientD21(t *testing.T) {
	P, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0.2, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0.3, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}),
		mat.NewDense(2, 2, []float64{0, 1, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	for name, syn := range map[string]func() error{
		"HinfSyn": func() error { _, err := HinfSyn(P, 1, 1); return err },
		"H2Syn":   func() error { _, err := H2Syn(P, 1, 1); return err },
	} {
		if err := syn(); !errors.Is(err, ErrInvalidPartition) || errors.Is(err, ErrGammaNotAchievable) {
			t.Errorf("%s err = %v, want ErrInvalidPartition", name, err)
		}
	}
}

// discreteHinfNormSweep is the peak of σ_max(G(e^{jθ})) over θ ∈ [0, π], from
// a dense uniform and logarithmic grid refined by golden-section search
// around the largest local maxima; independent of the library's evaluators.
func discreteHinfNormSweep(sys *System) float64 {
	gain := func(th float64) float64 { return frSigmaMax(mixsynFR(sys, cmplx.Exp(complex(0, th)))) }
	var ths []float64
	for k := -80; k < 0; k++ {
		ths = append(ths, math.Pi*math.Pow(10, float64(k)/10))
	}
	const N = 4000
	for i := range N + 1 {
		ths = append(ths, math.Pi*float64(i)/N)
	}
	g := make([]float64, len(ths))
	for i, th := range ths {
		g[i] = gain(th)
	}
	best := math.Max(g[0], g[len(g)-1])
	for i := 1; i+1 < len(ths); i++ {
		if g[i] < g[i-1] || g[i] < g[i+1] || g[i] < 0.5*best {
			continue
		}
		lo, hi := math.Min(ths[i-1], ths[i+1]), math.Max(ths[i-1], ths[i+1])
		const r = 0.6180339887498949
		a, b := hi-r*(hi-lo), lo+r*(hi-lo)
		ga, gb := gain(a), gain(b)
		for hi-lo > 1e-13 {
			if ga > gb {
				hi, b, gb = b, a, ga
				a = hi - r*(hi-lo)
				ga = gain(a)
			} else {
				lo, a, ga = a, b, gb
				b = lo + r*(hi-lo)
				gb = gain(b)
			}
		}
		best = math.Max(best, math.Max(g[i], math.Max(ga, gb)))
	}
	return best
}

// handLFT closes P with K (u = K·y) from the block formulas, independent of
// LFT, including D22 through (I − Dk·D22)⁻¹.
func handLFT(t *testing.T, P, K *System, nmeas, ncont int) *System {
	t.Helper()
	n, m, p := P.Dims()
	nk, _, _ := K.Dims()
	m1, p1 := m-ncont, p-nmeas
	blk := func(M *mat.Dense, r0, r1, c0, c1 int) *mat.Dense { return mat.DenseCopyOf(M.Slice(r0, r1, c0, c1)) }
	B1, B2 := blk(P.B, 0, n, 0, m1), blk(P.B, 0, n, m1, m)
	C1, C2 := blk(P.C, 0, p1, 0, n), blk(P.C, p1, p, 0, n)
	D11, D12 := blk(P.D, 0, p1, 0, m1), blk(P.D, 0, p1, m1, m)
	D21, D22 := blk(P.D, p1, p, 0, m1), blk(P.D, p1, p, m1, m)
	var IDD, Delta mat.Dense
	IDD.Mul(K.D, D22)
	IDD.Scale(-1, &IDD)
	for i := range ncont {
		IDD.Set(i, i, IDD.At(i, i)+1)
	}
	if err := Delta.Inverse(&IDD); err != nil {
		t.Fatal(err)
	}
	N := n + nk
	Cu := mat.NewDense(ncont, N, nil)
	var DkC2, DkD21, Du mat.Dense
	DkC2.Mul(K.D, C2)
	Cu.Slice(0, ncont, 0, n).(*mat.Dense).Copy(&DkC2)
	if nk > 0 {
		Cu.Slice(0, ncont, n, N).(*mat.Dense).Copy(K.C)
	}
	Cu.Mul(&Delta, mat.DenseCopyOf(Cu))
	DkD21.Mul(K.D, D21)
	Du.Mul(&Delta, &DkD21)
	Cy := mat.NewDense(nmeas, N, nil)
	Cy.Slice(0, nmeas, 0, n).(*mat.Dense).Copy(C2)
	var tmp mat.Dense
	tmp.Mul(D22, Cu)
	Cy.Add(Cy, &tmp)
	var Dy mat.Dense
	Dy.Mul(D22, &Du)
	Dy.Add(&Dy, D21)

	A := mat.NewDense(N, N, nil)
	B := mat.NewDense(N, m1, nil)
	var t1 mat.Dense
	t1.Mul(B2, Cu)
	A.Slice(0, n, 0, N).(*mat.Dense).Copy(&t1)
	a := A.Slice(0, n, 0, n).(*mat.Dense)
	a.Add(a, P.A)
	var t2 mat.Dense
	t2.Mul(B2, &Du)
	t2.Add(&t2, B1)
	B.Slice(0, n, 0, m1).(*mat.Dense).Copy(&t2)
	if nk > 0 {
		var t3, t4 mat.Dense
		t3.Mul(K.B, Cy)
		A.Slice(n, N, 0, N).(*mat.Dense).Copy(&t3)
		ak := A.Slice(n, N, n, N).(*mat.Dense)
		ak.Add(ak, K.A)
		t4.Mul(K.B, &Dy)
		B.Slice(n, N, 0, m1).(*mat.Dense).Copy(&t4)
	}
	var C, D mat.Dense
	C.Mul(D12, Cu)
	c := C.Slice(0, p1, 0, n).(*mat.Dense)
	c.Add(c, C1)
	D.Mul(D12, &Du)
	D.Add(&D, D11)
	cl, err := New(A, B, &C, &D, P.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return cl
}

func assertSchurStable(t *testing.T, A *mat.Dense) {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		t.Fatal("closed-loop eigenvalues failed")
	}
	for _, ev := range eig.Values(nil) {
		if cmplx.Abs(ev) >= 1-1e-9 {
			t.Fatalf("closed loop not Schur stable: pole %v", ev)
		}
	}
}

// discreteHinfTestPlant is a MIMO plant (3 states, w ∈ R³, u ∈ R, z ∈ R²,
// y ∈ R²) with non-symmetric A and nonzero D11 and D22, discretized by ZOH.
func discreteHinfTestPlant(t *testing.T, d11 float64) *System {
	t.Helper()
	Pc, err := New(
		mat.NewDense(3, 3, []float64{-1, 0.5, 0, 0, -2, 1, 0.3, 0, 0.5}),
		mat.NewDense(3, 4, []float64{1, 0, 0, 1, 0, 1, 0, 0.5, 0, 0, 1, -1}),
		mat.NewDense(4, 3, []float64{1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(4, 4, []float64{
			d11, 0, -0.5 * d11, 0,
			0, 0.5 * d11, 0, 1,
			0.2, 0.1, 0, 0.7,
			0, 0.1, 0.3, -0.4,
		}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	P, err := Pc.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// assertDiscreteHinfSyn checks the discrete design against an independent
// unit-circle norm of the hand-built closed loop, Schur stability, and the
// continuous design on the Tustin d2c plant (the same γ, and K its Tustin
// c2d).
func assertDiscreteHinfSyn(t *testing.T, P *System, nmeas, ncont int) *HinfSynResult {
	t.Helper()
	res, err := HinfSyn(P, nmeas, ncont)
	if err != nil {
		t.Fatal(err)
	}
	if res.K.Dt != P.Dt {
		t.Fatalf("K.Dt = %g, want %g", res.K.Dt, P.Dt)
	}
	cl := handLFT(t, P, res.K, nmeas, ncont)
	assertSchurStable(t, cl.A)
	for _, p := range res.CLPoles {
		if cmplx.Abs(p) >= 1 {
			t.Fatalf("CLPoles %v not inside the unit circle", res.CLPoles)
		}
	}
	norm := discreteHinfNormSweep(cl)
	if norm > res.GammaOpt*(1+1e-9) {
		t.Fatalf("‖CL‖∞ = %.12g exceeds γ = %.12g", norm, res.GammaOpt)
	}
	if res.GammaOpt > norm*(1+2e-4) {
		t.Fatalf("γ = %.12g not near achieved ‖CL‖∞ = %.12g", res.GammaOpt, norm)
	}

	return res
}

func assertTustinAgreement(t *testing.T, P *System, res *HinfSynResult, nmeas, ncont int) {
	t.Helper()
	Pc, err := P.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := HinfSyn(Pc, nmeas, ncont)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.GammaOpt-ref.GammaOpt) > 1e-8*ref.GammaOpt {
		t.Fatalf("γ = %.15g, continuous Tustin-equivalent γ = %.15g", res.GammaOpt, ref.GammaOpt)
	}
	Kd, err := ref.K.C2D(P.Dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	for _, th := range []float64{0, 0.3, 1.2, 2.9} {
		z := cmplx.Exp(complex(0, th))
		assertFRClose(t, fmt.Sprintf("K(e^j%g)", th), mixsynFR(res.K, z), mixsynFR(Kd, z), 1e-8)
	}
}

func TestHinfSyn_DiscreteMIMO(t *testing.T) {
	for _, d11 := range []float64{0, 0.4} {
		t.Run(fmt.Sprintf("D11=%g", d11), func(t *testing.T) {
			P := discreteHinfTestPlant(t, d11)
			res := assertDiscreteHinfSyn(t, P, 2, 1)
			assertTustinAgreement(t, P, res, 2, 1)
		})
	}
}

// twoDisturbancePlant has w ∈ R², u ∈ R, z ∈ R², y ∈ R with the given A and
// sample time.
func twoDisturbancePlant(t *testing.T, A []float64, dt float64) *System {
	t.Helper()
	P, err := New(
		mat.NewDense(2, 2, A),
		mat.NewDense(2, 3, []float64{0.1, 0, 0.5, 0, 0.2, 1}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 0.4, 1, 0.5}),
		mat.NewDense(3, 3, []float64{0, 0, 0, 0, 0, 0.2, 0.3, 0.1, 0}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	return P
}

// An integrator at z = 1 keeps the standard Tustin map.
func TestHinfSyn_DiscreteIntegrator(t *testing.T) {
	P := twoDisturbancePlant(t, []float64{1, 0.1, 0, 0.6}, 0.5)
	res := assertDiscreteHinfSyn(t, P, 1, 1)
	assertTustinAgreement(t, P, res, 1, 1)
}

// A mode at z = −1 has no Tustin equivalent, so the design runs on the
// reflected plant P(−z) and reflects K back.
func TestHinfSyn_DiscretePoleAtMinusOne(t *testing.T) {
	P := twoDisturbancePlant(t, []float64{-1, 0.3, 0, 0.5}, 0.2)
	if _, err := P.D2C(D2COptions{Method: C2DMethodTustin}); err == nil {
		t.Fatal("expected the plain Tustin map to fail at z = −1")
	}
	res := assertDiscreteHinfSyn(t, P, 1, 1)

	Pr := P.Copy()
	Pr.A.Scale(-1, Pr.A)
	Pr.B.Scale(-1, Pr.B)
	ref := assertDiscreteHinfSyn(t, Pr, 1, 1)
	assertTustinAgreement(t, Pr, ref, 1, 1)
	if res.GammaOpt != ref.GammaOpt {
		t.Fatalf("γ = %.15g, reflected-plant γ = %.15g", res.GammaOpt, ref.GammaOpt)
	}
}

// Modes at both z = 1 and z = −1 have no Tustin equivalent under either map;
// HinfSyn first closes a static output feedback that moves them. Checked
// against the unit-circle sweep, against the plain Tustin route on the plant
// with a different static gain closed by hand, and against plants whose z = −1 mode is perturbed into the disk.
func TestHinfSyn_DiscreteModesAtBothPlusMinusOne(t *testing.T) {
	P := twoDisturbancePlant(t, []float64{1, 0.3, 0, -1}, 0.2)
	res := assertDiscreteHinfSyn(t, P, 1, 1)

	// u = d·y + v with D22 = 0 moves both modes (A + B2·d·C2 has eigenvalues
	// ≈ 1.35 and −0.85), so the shifted plant designs through Tustin alone.
	const d = 0.5
	B2, C2 := mat.NewDense(2, 1, []float64{0.5, 1}), mat.NewDense(1, 2, []float64{1, 0.5})
	shift := func(M, L, R *mat.Dense) *mat.Dense {
		var LR mat.Dense
		LR.Mul(L, R)
		LR.Scale(d, &LR)
		LR.Add(M, &LR)
		return &LR
	}
	var B1, C1, D11, D12, D21 = P.B.Slice(0, 2, 0, 2).(*mat.Dense), P.C.Slice(0, 2, 0, 2).(*mat.Dense),
		P.D.Slice(0, 2, 0, 2).(*mat.Dense), P.D.Slice(0, 2, 2, 3).(*mat.Dense), P.D.Slice(2, 3, 0, 2).(*mat.Dense)
	Bs, Cs, Ds := mat.DenseCopyOf(P.B), mat.DenseCopyOf(P.C), mat.DenseCopyOf(P.D)
	Bs.Slice(0, 2, 0, 2).(*mat.Dense).Copy(shift(B1, B2, D21))
	Cs.Slice(0, 2, 0, 2).(*mat.Dense).Copy(shift(C1, D12, C2))
	Ds.Slice(0, 2, 0, 2).(*mat.Dense).Copy(shift(D11, D12, D21))
	Ps, err := New(shift(P.A, B2, C2), Bs, Cs, Ds, P.Dt)
	if err != nil {
		t.Fatal(err)
	}
	ref := assertDiscreteHinfSyn(t, Ps, 1, 1)
	if math.Abs(res.GammaOpt-ref.GammaOpt) > 1e-6*ref.GammaOpt {
		t.Fatalf("γ = %.12g, hand-shifted plant γ = %.12g", res.GammaOpt, ref.GammaOpt)
	}

	// Near-both plants are shifted too: through the Tustin map alone γ was
	// 1.5 at δ = 1e-5 and 4e4 at δ = 1e-7.
	for _, delta := range []float64{1e-3, 1e-5, 1e-7} {
		pert := assertDiscreteHinfSyn(t, twoDisturbancePlant(t, []float64{1, 0.3, 0, -1 + delta}, 0.2), 1, 1)
		if math.Abs(pert.GammaOpt-res.GammaOpt) > 10*delta*res.GammaOpt {
			t.Fatalf("A22 = −1 + %g: γ = %.12g, unperturbed γ = %.12g", delta, pert.GammaOpt, res.GammaOpt)
		}
	}
}

// MIMO variant with D22 ≠ 0, two measurements and a third stable mode.
func TestHinfSyn_DiscreteModesAtBothPlusMinusOneMIMO(t *testing.T) {
	P, err := New(
		mat.NewDense(3, 3, []float64{1, 0.3, 0.1, 0, -1, 0.2, 0, 0, 0.5}),
		mat.NewDense(3, 4, []float64{0.1, 0, 0.2, 0.5, 0, 0.2, 0, 1, 0.3, 0, 0.1, -0.4}),
		mat.NewDense(4, 3, []float64{1, 0, 0.2, 0, 0.4, 0, 1, 0.5, 0, 0, 1, 1}),
		mat.NewDense(4, 4, []float64{
			0.1, 0, 0, 0,
			0, 0, 0, 0.2,
			0.3, 0.1, 0, 0.3,
			0, 0.2, 0.4, -0.5,
		}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertDiscreteHinfSyn(t, P, 2, 1)
}

// In discrete time the rank condition is on P12 and P21 at z = −1, the
// boundary point Tustin sends to s = ∞: D12 = 0 is fine, while
// P12(−1) = 0 is rejected.
func TestHinfSyn_DiscreteRankAtMinusOne(t *testing.T) {
	A := []float64{0.5, 0.2, -0.1, 0.3}
	mk := func(c1 float64) *System {
		P, err := New(
			mat.NewDense(2, 2, A),
			mat.NewDense(2, 3, []float64{1, 0, 1, 0.2, 0.5, 0}),
			mat.NewDense(3, 2, []float64{c1, 1, 0, 0, 0.5, 1}),
			mat.NewDense(3, 3, []float64{0, 0, 0, 0.5, 0, 0, 1, 0.2, 0}),
			1,
		)
		if err != nil {
			t.Fatal(err)
		}
		return P
	}
	P := mk(0.4)
	res := assertDiscreteHinfSyn(t, P, 1, 1)
	assertTustinAgreement(t, P, res, 1, 1)

	// P12(−1) = C1(−I − A)⁻¹B2 with B2 = e1 vanishes for c1 = −v1/v0, v the
	// first column of (−I − A)⁻¹.
	var M mat.Dense
	if err := M.Inverse(mat.NewDense(2, 2, []float64{-1 - A[0], -A[1], -A[2], -1 - A[3]})); err != nil {
		t.Fatal(err)
	}
	if _, err := HinfSyn(mk(-M.At(1, 0)/M.At(0, 0)), 1, 1); !errors.Is(err, ErrInvalidPartition) {
		t.Fatalf("HinfSyn err = %v, want ErrInvalidPartition", err)
	}
}

// Every constant Youla parameter Q with ‖Q‖ < γ must give a stabilizing
// controller with ‖T_zw‖∞ < γ (Zhou, Doyle & Glover, Thm 17.13); this checks
// the non-central M∞ blocks (B̂2, Ĉ2, D̂12, D̂21) against a Hamiltonian
// bisection of the hand-built closed loop, with D1111 nonempty and D22 ≠ 0.
func TestHinfYoulaParameterMeetsGamma(t *testing.T) {
	for _, c := range []struct {
		name         string
		P            *System
		nmeas, ncont int
	}{
		{"generic D11", genericD11Plant(t, 0.3), 1, 1},
		{"MIMO mixed sensitivity", mimoBiproperMixedSensitivityPlant(t), 2, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			gp, err := partitionGeneralizedPlant("test", c.P, c.nmeas, c.ncont)
			if err != nil {
				t.Fatal(err)
			}
			hp, err := newHinfGeneralPlant(gp)
			if err != nil {
				t.Fatal(err)
			}
			res, err := HinfSyn(c.P, c.nmeas, c.ncont)
			if err != nil {
				t.Fatal(err)
			}
			gamma := res.GammaOpt
			X, Y, Rinv, Rtinv, err := hp.riccatis(gamma)
			if err != nil {
				t.Fatal(err)
			}
			M, err := hp.youla(gamma, X, Y, Rinv, Rtinv)
			if err != nil {
				t.Fatal(err)
			}
			for _, frac := range []float64{0.3, 0.9} {
				Q := mat.NewDense(c.ncont, c.nmeas, nil)
				for i := range c.ncont {
					for j := range c.nmeas {
						Q.Set(i, j, []float64{0.6, -0.3, 0.2, 0.5}[(i*c.nmeas+j)%4])
					}
				}
				Q.Scale(frac*gamma/mat.Norm(Q, 2), Q)
				K, err := gp.newController(M.controller(Q))
				if err != nil {
					t.Fatal(err)
				}
				cl := handLFT(t, c.P, K, c.nmeas, c.ncont)
				var eig mat.Eigen
				if !eig.Factorize(cl.A, mat.EigenNone) {
					t.Fatal("closed-loop eigenvalues failed")
				}
				for _, ev := range eig.Values(nil) {
					if real(ev) >= 0 {
						t.Fatalf("‖Q‖ = %gγ: closed loop unstable: pole %v", frac, ev)
					}
				}
				if norm := hinfNormBisection(t, cl.A, cl.B, cl.C, cl.D); norm > gamma*(1+1e-9) {
					t.Fatalf("‖Q‖ = %gγ: ‖CL‖∞ = %.12g exceeds γ = %.12g", frac, norm, gamma)
				}
			}
		})
	}
}
