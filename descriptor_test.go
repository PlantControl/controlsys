package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestGeneralizedPoles_Standard(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	expected := []complex128{complex(-1, 0), complex(-2, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestGeneralizedPoles_InfiniteFiltered(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3, 1, 0, -1})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 0})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 {
		t.Fatalf("got %d finite poles, want 1", len(poles))
	}
	if math.Abs(real(poles[0])-(-3)) > 1e-10 {
		t.Errorf("expected pole -3, got %v", poles[0])
	}
}

func TestDescriptorSystem_Poles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	E := mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = E

	if !sys.IsDescriptor() {
		t.Error("expected IsDescriptor() = true")
	}

	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	// E^{-1}*A eigenvalues: [2,0;0,1]^{-1}*[-1,2;0,-3] = [-0.5,1;0,-3]
	expected := []complex128{complex(-0.5, 0), complex(-3, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestDescriptorSystem_Copy(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	cp := sys.Copy()
	if cp.E == nil {
		t.Fatal("Copy did not preserve E")
	}
	cp.E.Set(0, 0, 99)
	if sys.E.At(0, 0) == 99 {
		t.Error("Copy shares E backing array")
	}
}

func TestDescriptorSystem_NonDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if sys.IsDescriptor() {
		t.Error("standard system should not be descriptor")
	}
}

func TestDescriptorSystem_UnsupportedOperationsRejectExplicitly(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	ops := []struct {
		name string
		run  func() error
	}{
		{name: "TransferFunction", run: func() error {
			_, err := sys.TransferFunction(nil)
			return err
		}},
		{name: "Simulate", run: func() error {
			discrete := sys.Copy()
			discrete.Dt = 0.1
			_, err := discrete.Simulate(mat.NewDense(1, 3, nil), nil, nil)
			return err
		}},
		{name: "Inv", run: func() error {
			_, err := Inv(sys)
			return err
		}},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			if err := op.run(); !errors.Is(err, ErrDescriptorUnsupported) {
				t.Fatalf("got %v, want ErrDescriptorUnsupported", err)
			}
		})
	}
}

func TestDescriptorSystem_FrequencyResponseUsesDescriptorPencil(t *testing.T) {
	sys, err := NewDescriptor(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{4}),
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0, 1, 5}
	response, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for k, frequency := range omega {
		want := 12/complex(1, 2*frequency) + 0.5
		if got := response.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12 {
			t.Fatalf("response at %g = %v, want %v", frequency, got, want)
		}
	}
	evaluated, err := sys.EvalFr(1i)
	if err != nil {
		t.Fatalf("EvalFr: %v", err)
	}
	if want := 12/complex(1, 2) + 0.5; cmplx.Abs(evaluated[0][0]-want) > 1e-12 {
		t.Fatalf("EvalFr = %v, want %v", evaluated[0][0], want)
	}
}

func TestDescriptorSystem_FrequencyResponsePivotsNonSymmetricPencil(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{0, -2, 3, -1})
	e := mat.NewDense(2, 2, []float64{0, 1, 1, 0.5})
	b := mat.NewDense(2, 2, []float64{1, -1, 2, 0.5})
	c := mat.NewDense(2, 2, []float64{1, 0.25, -0.5, 2})
	d := mat.NewDense(2, 2, []float64{0.1, 0.2, -0.3, 0.4})
	sys, err := NewDescriptor(a, b, c, d, e, 0)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 1)
	pencil := [4]complex128{
		s*complex(e.At(0, 0), 0) - complex(a.At(0, 0), 0),
		s*complex(e.At(0, 1), 0) - complex(a.At(0, 1), 0),
		s*complex(e.At(1, 0), 0) - complex(a.At(1, 0), 0),
		s*complex(e.At(1, 1), 0) - complex(a.At(1, 1), 0),
	}
	det := pencil[0]*pencil[3] - pencil[1]*pencil[2]
	inv := [4]complex128{pencil[3] / det, -pencil[1] / det, -pencil[2] / det, pencil[0] / det}
	want := make([][]complex128, 2)
	for i := range 2 {
		want[i] = make([]complex128, 2)
		for j := range 2 {
			for k := range 2 {
				for l := range 2 {
					want[i][j] += complex(c.At(i, k), 0) * inv[k*2+l] * complex(b.At(l, j), 0)
				}
			}
			want[i][j] += complex(d.At(i, j), 0)
		}
	}

	got, err := sys.EvalFr(s)
	if err != nil {
		t.Fatalf("EvalFr: %v", err)
	}
	response, err := sys.FreqResponse([]float64{1})
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for i := range 2 {
		for j := range 2 {
			if diff := cmplx.Abs(got[i][j] - want[i][j]); diff > 1e-12 {
				t.Fatalf("EvalFr(%d,%d) diff=%g: got %v, want %v", i, j, diff, got[i][j], want[i][j])
			}
			if diff := cmplx.Abs(response.At(0, i, j) - want[i][j]); diff > 1e-12 {
				t.Fatalf("FreqResponse(%d,%d) diff=%g: got %v, want %v", i, j, diff, response.At(0, i, j), want[i][j])
			}
		}
	}
}

func TestDescriptorSystem_IdentityDescriptorUsesStandardOperations(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	if sys.IsDescriptor() {
		t.Fatal("identity E should be treated as standard form")
	}

	ops := []struct {
		name string
		run  func() error
	}{
		{name: "TransferFunction", run: func() error {
			_, err := sys.TransferFunction(nil)
			return err
		}},
		{name: "FreqResponse", run: func() error {
			_, err := sys.FreqResponse([]float64{1})
			return err
		}},
		{name: "EvalFr", run: func() error {
			_, err := sys.EvalFr(1i)
			return err
		}},
		{name: "Simulate", run: func() error {
			discrete := sys.Copy()
			discrete.Dt = 0.1
			_, err := discrete.Simulate(mat.NewDense(1, 3, nil), nil, nil)
			return err
		}},
		{name: "Inv", run: func() error {
			_, err := Inv(sys)
			return err
		}},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			if err := op.run(); err != nil {
				t.Fatalf("got %v, want nil", err)
			}
		})
	}
}
