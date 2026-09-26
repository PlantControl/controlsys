package controlsys

import (
	"errors"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestH2Syn_Simple(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	P.InputName = []string{"disturbance", "control"}
	P.OutputName = []string{"performance1", "performance2", "measurement"}

	res, err := H2Syn(P, 1, 1)
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
	if !stringSlicesEqual(res.K.InputName, []string{"measurement"}) {
		t.Fatalf("controller input names = %v, want [measurement]", res.K.InputName)
	}
	if !stringSlicesEqual(res.K.OutputName, []string{"control"}) {
		t.Fatalf("controller output names = %v, want [control]", res.K.OutputName)
	}
}

func TestH2Syn_Stability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := res.K.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("controller is not stable")
	}

	for _, p := range res.CLPoles {
		if real(p) >= -1e-10 {
			t.Errorf("closed-loop pole not in open left half-plane: %v", p)
		}
	}
}

func TestH2Syn_DiscreteError(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, nil)

	P, err := New(A, B, C, D, 0.01)
	if err != nil {
		t.Fatal(err)
	}

	_, err = H2Syn(P, 1, 1)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("got %v, want ErrWrongDomain", err)
	}
}

func TestH2Syn_DimensionErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, nil)

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
		{"ncont too large", 1, 3},
		{"nmeas=p", 3, 1},
		{"ncont=m", 1, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := H2Syn(P, tt.nmeas, tt.ncont)
			if !errors.Is(err, ErrInvalidPartition) {
				t.Errorf("got %v, want ErrInvalidPartition", err)
			}
		})
	}
}

func TestH2Syn_D11NonZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		0, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = H2Syn(P, 1, 1)
	if !errors.Is(err, ErrNoFiniteH2Norm) {
		t.Errorf("got %v, want ErrNoFiniteH2Norm", err)
	}
}

func TestH2Syn_D22NonZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0.25,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	closedLoop(t, P, res.K, 1, 1)
}

func TestH2Syn_Unstabilizable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = H2Syn(P, 1, 1)
	if !errors.Is(err, ErrNotStabilizable) {
		t.Errorf("got %v, want ErrNotStabilizable", err)
	}
}

func TestH2Syn_SecondOrder(t *testing.T) {
	// Plant: x' = Ax + B1*w + B2*u, z = C1*x + D12*u, y = C2*x + D21*w
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	}) // B1=2×2, B2=2×1
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	}) // C1=2×2, C2=1×2
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0, // D11=0
		0, 0, 1, // D12=[0;1]
		0.1, 0.1, 0, // D21=[0.1 0.1], D22=0
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	// nmeas=1 (1 measurement), ncont=1 (1 control)
	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) > 1e-8 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

}

func TestH2Syn_D22LoopShiftMixedSensitivity(t *testing.T) {
	P := mixedSensitivityFeedthroughPlant(t)
	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	norm, err := H2Norm(closedLoop(t, P, res.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}

	P0 := withoutD22(t, P, 1, 1)
	ref, err := H2Syn(P0, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	refNorm, err := H2Norm(closedLoop(t, P0, ref.K, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-refNorm) > 1e-8*refNorm {
		t.Fatalf("closed-loop H2 norm %v, want D22-free optimum %v", norm, refNorm)
	}
}
