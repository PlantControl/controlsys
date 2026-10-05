package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func vecEqual(a, b *mat.VecDense, tol float64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if a.Len() != b.Len() {
		return false
	}
	for i := 0; i < a.Len(); i++ {
		if math.Abs(a.AtVec(i)-b.AtVec(i)) > tol {
			return false
		}
	}
	return true
}

func TestSimulateManualPropagation(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{
		0.9, 0.1,
		0.0, 0.8,
	})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})

	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, 3, []float64{1, 2, 3})
	x0 := mat.NewVecDense(2, []float64{0.5, -0.3})

	r, err := sys.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}

	// Manual step-by-step:
	// k=0: x=[0.5, -0.3], u=1
	//   y = C*x + D*u = 1*0.5 + 0*(-0.3) + 0.5*1 = 1.0
	//   x1 = A*x + B*u = [0.9*0.5+0.1*(-0.3)+1, 0.8*(-0.3)] = [1.42, -0.24]
	x := []float64{0.5, -0.3}
	uu := []float64{1, 2, 3}
	wantY := make([]float64, 3)
	wantX := make([]float64, 2)

	for k := range 3 {
		wantY[k] = 1*x[0] + 0*x[1] + 0.5*uu[k]
		nx0 := 0.9*x[0] + 0.1*x[1] + 1*uu[k]
		nx1 := 0.0*x[0] + 0.8*x[1] + 0*uu[k]
		x[0], x[1] = nx0, nx1
	}
	wantX[0], wantX[1] = x[0], x[1]

	wantYMat := mat.NewDense(1, 3, wantY)
	wantXVec := mat.NewVecDense(2, wantX)

	if !matEqual(r.Y, wantYMat, 1e-12) {
		t.Errorf("Y mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.Y), mat.Formatted(wantYMat))
	}
	if !vecEqual(r.XFinal, wantXVec, 1e-12) {
		t.Errorf("XFinal mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.XFinal), mat.Formatted(wantXVec))
	}
}

func TestSimulateChaining(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0.0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	u1 := mat.NewDense(1, 3, []float64{1, 2, 3})
	u2 := mat.NewDense(1, 2, []float64{4, 5})

	r1, err := sys.Simulate(u1, nil, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := sys.Simulate(u2, r1.XFinal, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}

	uAll := mat.NewDense(1, 5, []float64{1, 2, 3, 4, 5})
	rAll, err := sys.Simulate(uAll, nil, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}

	// r1.Y columns 0..2 should match rAll.Y columns 0..2
	for k := range 3 {
		if math.Abs(r1.Y.At(0, k)-rAll.Y.At(0, k)) > 1e-12 {
			t.Errorf("chaining mismatch at step %d: r1=%f rAll=%f", k, r1.Y.At(0, k), rAll.Y.At(0, k))
		}
	}
	// r2.Y columns 0..1 should match rAll.Y columns 3..4
	for k := range 2 {
		if math.Abs(r2.Y.At(0, k)-rAll.Y.At(0, k+3)) > 1e-12 {
			t.Errorf("chaining mismatch at step %d: r2=%f rAll=%f", k+3, r2.Y.At(0, k), rAll.Y.At(0, k+3))
		}
	}
	if !vecEqual(r2.XFinal, rAll.XFinal, 1e-12) {
		t.Errorf("XFinal mismatch after chaining")
	}
}

func TestSimulatePureFeedthrough(t *testing.T) {
	D := mat.NewDense(2, 1, []float64{3, -1})
	sys, err := New(nil, nil, nil, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, 4, []float64{1, 2, 3, 4})
	r, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	wantY := mat.NewDense(2, 4, []float64{
		3, 6, 9, 12,
		-1, -2, -3, -4,
	})
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("feedthrough Y mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}
	if r.XFinal != nil {
		t.Errorf("expected nil XFinal when not requested, got %v", r.XFinal)
	}
	if _, err := sys.Simulate(u, nil, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("n=0 FinalState err = %v, want ErrDimensionMismatch", err)
	}
}

func TestSimulateNoInputs(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.25})
	C := mat.NewDense(1, 2, []float64{1, 1})

	sys, err := New(A, nil, C, nil, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	x0 := mat.NewVecDense(2, []float64{4, 8})

	if _, err := sys.Simulate(nil, x0, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("zero samples error = %v, want ErrInsufficientData", err)
	}
}

func TestSimulateAutonomousPropagation(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.25})
	B := mat.NewDense(2, 1, nil)
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, nil)

	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	x0 := mat.NewVecDense(2, []float64{4, 8})
	u := mat.NewDense(1, 4, nil)

	r, err := sys.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}

	// y[k] = C * A^k * x0 (D*u = 0, B*u = 0)
	// k=0: y = [1,1]*[4,8]^T = 12
	// k=1: x1 = [2, 2], y = 4
	// k=2: x2 = [1, 0.5], y = 1.5
	// k=3: x3 = [0.5, 0.125], y = 0.625
	wantY := mat.NewDense(1, 4, []float64{12, 4, 1.5, 0.625})
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("autonomous Y mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}

	wantXFinal := mat.NewVecDense(2, []float64{0.25, 0.03125})
	if !vecEqual(r.XFinal, wantXFinal, 1e-12) {
		t.Errorf("XFinal mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.XFinal), mat.Formatted(wantXFinal))
	}
}

func TestSimulateStepsZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	x0 := mat.NewVecDense(2, []float64{5, 3})

	if _, err := sys.Simulate(nil, x0, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("zero samples error = %v, want ErrInsufficientData", err)
	}
	if _, err := sys.Simulate(nil, x0, nil); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("nil u, nil opts error = %v, want ErrInsufficientData", err)
	}
}

func TestSimulateContinuousError(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0) // continuous
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, 5, nil)
	_, err = sys.Simulate(u, nil, nil)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("expected ErrWrongDomain, got %v", err)
	}
}

func TestSimulateWorkspaceReuse(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.5})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, 3, []float64{1, 0, 0})
	ws := mat.NewVecDense(2, nil)
	opts := &SimulateOpts{Workspace: ws}

	r1, err := sys.Simulate(u, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(r1.Y, r2.Y, 1e-12) {
		t.Errorf("workspace reuse gave different results")
	}
}

func TestSimulateWithExternalDelayUsesOutputBuffer(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{0.25}),
		1,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{1}

	u := mat.NewDense(1, 4, []float64{1, 2, 3, 4})
	yBuf := mat.NewDense(1, 4, nil)
	resp, err := sys.Simulate(u, nil, &SimulateOpts{yBuf: yBuf})
	if err != nil {
		t.Fatalf("Simulate with external delay: %v", err)
	}
	if resp.Y != yBuf {
		t.Fatalf("Simulate with external delay did not use provided output buffer")
	}
}

func TestSimulateNilX0(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.9})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, 2, []float64{1, 0})
	r, err := sys.Simulate(u, nil, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}

	// k=0: x=0, y=0, x1=1
	// k=1: x=1, y=1, x2=0.9
	wantY := mat.NewDense(1, 2, []float64{0, 1})
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("nil x0 Y mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}
	wantXF := mat.NewVecDense(1, []float64{0.9})
	if !vecEqual(r.XFinal, wantXF, 1e-12) {
		t.Errorf("nil x0 XFinal mismatch")
	}
}

func TestSimulateDimensionErrors(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, nil),
		1,
	)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		u    *mat.Dense
		x0   *mat.VecDense
		opts *SimulateOpts
	}{
		{
			name: "input rows",
			u:    mat.NewDense(2, 3, nil),
		},
		{
			name: "initial state",
			u:    mat.NewDense(1, 3, nil),
			x0:   mat.NewVecDense(1, nil),
		},
		{
			name: "workspace",
			u:    mat.NewDense(1, 3, nil),
			opts: &SimulateOpts{Workspace: mat.NewVecDense(1, nil)},
		},
		{
			name: "y buffer",
			u:    mat.NewDense(1, 3, nil),
			opts: &SimulateOpts{yBuf: mat.NewDense(2, 3, nil)},
		},
		{
			name: "du buffer",
			u:    mat.NewDense(1, 3, nil),
			opts: &SimulateOpts{duBuf: mat.NewDense(1, 2, nil)},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := sys.Simulate(tc.u, tc.x0, tc.opts); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func TestSimulateValidatesMutatedDelayFields(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, nil),
		1,
	)
	if err != nil {
		t.Fatal(err)
	}
	u := mat.NewDense(1, 3, nil)

	sys.InputDelay = []float64{1, 2}
	if _, err := sys.Simulate(u, nil, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("InputDelay: got %v, want ErrDimensionMismatch", err)
	}

	sys.InputDelay = nil
	sys.OutputDelay = []float64{-1}
	if _, err := sys.Simulate(u, nil, nil); !errors.Is(err, ErrNegativeDelay) {
		t.Fatalf("OutputDelay: got %v, want ErrNegativeDelay", err)
	}

	sys.OutputDelay = nil
	sys.Delay = mat.NewDense(2, 1, nil)
	if _, err := sys.Simulate(u, nil, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("Delay: got %v, want ErrDimensionMismatch", err)
	}
}

func TestSimulate_WithInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	sys.InputDelay = []float64{3}

	ref, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	ref.Delay = mat.NewDense(1, 1, []float64{3})

	u := mat.NewDense(1, 10, nil)
	for i := range 10 {
		u.Set(0, i, 1)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("InputDelay mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_WithOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	sys.OutputDelay = []float64{3}

	ref, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	ref.Delay = mat.NewDense(1, 1, []float64{3})

	u := mat.NewDense(1, 10, nil)
	for i := range 10 {
		u.Set(0, i, 1)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("OutputDelay mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_WithCombinedDelays(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	sys.InputDelay = []float64{2}
	sys.OutputDelay = []float64{1}

	ref, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	ref.Delay = mat.NewDense(1, 1, []float64{3})

	u := mat.NewDense(1, 10, nil)
	for i := range 10 {
		u.Set(0, i, 1)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("CombinedDelays mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_WithMIMOInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	sys.InputDelay = []float64{1, 3}

	ref, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	ref.Delay = mat.NewDense(2, 2, []float64{1, 3, 1, 3})

	u := mat.NewDense(2, 8, nil)
	for j := range 8 {
		u.Set(0, j, 1)
		u.Set(1, j, 0.5)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("MIMOInputDelay mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_WithMIMOOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	sys.OutputDelay = []float64{2, 0}

	ref, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	ref.Delay = mat.NewDense(2, 2, []float64{2, 2, 0, 0})

	u := mat.NewDense(2, 8, nil)
	for j := range 8 {
		u.Set(0, j, 1)
		u.Set(1, j, 0.5)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("MIMOOutputDelay mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_WithAllDelayTypes(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	sys.InputDelay = []float64{1, 0}
	sys.OutputDelay = []float64{0, 2}
	sys.Delay = mat.NewDense(2, 2, []float64{0, 1, 1, 0})

	ref, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		1.0,
	)
	ref.Delay = mat.NewDense(2, 2, []float64{1, 1, 4, 2})

	u := mat.NewDense(2, 10, nil)
	for j := range 10 {
		u.Set(0, j, 1)
		u.Set(1, j, 0.5)
	}

	got, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("AllDelayTypes mismatch\ngot:  %v\nwant: %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_InternalDelay_SISO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0.0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.2})

	refSys, _ := New(A, B, C, D, 1.0)
	refSys.InputDelay = []float64{3}

	lftSys := &System{
		A:  mat.DenseCopyOf(A),
		B:  mat.NewDense(2, 1, nil),
		C:  mat.DenseCopyOf(C),
		D:  mat.NewDense(1, 1, nil),
		Dt: 1.0,
		LFT: &LFTDelay{
			Tau: []float64{3},
			B2:  mat.DenseCopyOf(B),
			C2:  mat.NewDense(1, 2, nil),
			D12: mat.DenseCopyOf(D),
			D21: mat.NewDense(1, 1, []float64{1}),
			D22: mat.NewDense(1, 1, nil),
		},
	}

	u := mat.NewDense(1, 15, nil)
	for i := range 15 {
		u.Set(0, i, float64(i+1)*0.1)
	}

	got, err := lftSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := refSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("InternalDelay SISO mismatch\ngot:  %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_InternalDelay_MultipleDelays(t *testing.T) {
	refSys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		1.0,
	)
	refSys.InputDelay = []float64{2, 5}

	lftSys := &System{
		A:  mat.NewDense(2, 2, []float64{0.5, 0.1, 0.0, 0.8}),
		B:  mat.NewDense(2, 2, nil),
		C:  mat.NewDense(1, 2, []float64{1, 1}),
		D:  mat.NewDense(1, 2, nil),
		Dt: 1.0,
		LFT: &LFTDelay{
			Tau: []float64{2, 5},
			B2:  mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
			C2:  mat.NewDense(2, 2, nil),
			D12: mat.NewDense(1, 2, nil),
			D21: mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
			D22: mat.NewDense(2, 2, nil),
		},
	}

	u := mat.NewDense(2, 12, nil)
	for j := range 12 {
		u.Set(0, j, 1.0)
		u.Set(1, j, 0.5)
	}

	got, err := lftSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := refSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("InternalDelay multiple mismatch\ngot:  %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_InternalDelay_WithX0(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0.0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	refSys, _ := New(A, B, C, D, 1.0)
	refSys.InputDelay = []float64{2}

	lftSys := &System{
		A:  mat.DenseCopyOf(A),
		B:  mat.NewDense(2, 1, nil),
		C:  mat.DenseCopyOf(C),
		D:  mat.NewDense(1, 1, nil),
		Dt: 1.0,
		LFT: &LFTDelay{
			Tau: []float64{2},
			B2:  mat.DenseCopyOf(B),
			C2:  mat.NewDense(1, 2, nil),
			D12: mat.DenseCopyOf(D),
			D21: mat.NewDense(1, 1, []float64{1}),
			D22: mat.NewDense(1, 1, nil),
		},
	}

	x0 := mat.NewVecDense(2, []float64{1.0, -0.5})
	u := mat.NewDense(1, 10, nil)
	for i := range 10 {
		u.Set(0, i, 1.0)
	}

	got, err := lftSys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := refSys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("InternalDelay x0 mismatch\ngot:  %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulate_InternalDelay_D22Nonzero(t *testing.T) {
	lftSys := &System{
		A:  mat.NewDense(1, 1, []float64{0.5}),
		B:  mat.NewDense(1, 1, []float64{1}),
		C:  mat.NewDense(1, 1, []float64{1}),
		D:  mat.NewDense(1, 1, []float64{0}),
		Dt: 1.0,
		LFT: &LFTDelay{
			Tau: []float64{2},
			B2:  mat.NewDense(1, 1, []float64{0.3}),
			C2:  mat.NewDense(1, 1, []float64{0.5}),
			D12: mat.NewDense(1, 1, []float64{0.1}),
			D21: mat.NewDense(1, 1, []float64{0.2}),
			D22: mat.NewDense(1, 1, []float64{0.4}),
		},
	}

	u := mat.NewDense(1, 8, nil)
	for i := range 8 {
		u.Set(0, i, 1.0)
	}

	got, err := lftSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	x := 0.0
	wantY := make([]float64, 8)
	zHist := make([]float64, 8)

	for k := range 8 {
		wk := 0.0
		if k >= 2 {
			wk = zHist[k-2]
		}

		wantY[k] = 1.0*x + 0.0*1.0 + 0.1*wk
		zHist[k] = 0.5*x + 0.2*1.0 + 0.4*wk
		x = 0.5*x + 1.0*1.0 + 0.3*wk
	}

	wantYMat := mat.NewDense(1, 8, wantY)
	if !matEqual(got.Y, wantYMat, 1e-12) {
		t.Errorf("InternalDelay D22 mismatch\ngot:  %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(wantYMat))
	}
}

func TestSimulate_InternalDelay_NonUnitDt(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})

	refSys, _ := New(A, B, C, D, 0.1)
	refSys.InputDelay = []float64{3}

	lftSys := &System{
		A:  mat.NewDense(1, 1, []float64{0.5}),
		B:  mat.NewDense(1, 1, nil),
		C:  mat.NewDense(1, 1, []float64{1}),
		D:  mat.NewDense(1, 1, nil),
		Dt: 0.1,
		LFT: &LFTDelay{
			Tau: []float64{3},
			B2:  mat.NewDense(1, 1, []float64{1}),
			C2:  mat.NewDense(1, 1, nil),
			D12: mat.NewDense(1, 1, nil),
			D21: mat.NewDense(1, 1, []float64{1}),
			D22: mat.NewDense(1, 1, nil),
		},
	}

	u := mat.NewDense(1, 10, nil)
	for i := range 10 {
		u.Set(0, i, 1.0)
	}

	got, err := lftSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := refSys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 1e-12) {
		t.Errorf("InternalDelay Dt!=1 mismatch\ngot:  %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulateDiscreteX0DelaysFreeResponseByOutputDelay(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, 0.7})
	b := mat.NewDense(2, 2, []float64{1, 0.3, 0, 1})
	c := mat.NewDense(2, 2, []float64{1, 0, 0.4, 1})
	d := mat.NewDense(2, 2, []float64{0.2, 0.1, 0, -0.3})
	sys, err := New(a, b, c, d, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{1, 0}
	sys.OutputDelay = []float64{0, 3}
	steps := 12
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(0.7*float64(k)))
		u.Set(1, k, 1)
	}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})
	resp, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	inner := make([][]float64, 2)
	for i := range inner {
		inner[i] = make([]float64, steps)
	}
	x := mat.VecDenseCopyOf(x0)
	for k := range steps {
		ud := mat.NewVecDense(2, nil)
		for j := range 2 {
			if lag := int(sys.InputDelay[j]); k >= lag {
				ud.SetVec(j, u.At(j, k-lag))
			}
		}
		var y, cx, du mat.VecDense
		cx.MulVec(c, x)
		du.MulVec(d, ud)
		y.AddVec(&cx, &du)
		for i := range 2 {
			inner[i][k] = y.AtVec(i)
		}
		var ax, bu mat.VecDense
		ax.MulVec(a, x)
		bu.MulVec(b, ud)
		x.AddVec(&ax, &bu)
	}
	for k := range steps {
		for i := range 2 {
			want := 0.0
			if lag := int(sys.OutputDelay[i]); k >= lag {
				want = inner[i][k-lag]
			}
			if math.Abs(resp.Y.At(i, k)-want) > 1e-12 {
				t.Fatalf("y_%d[%d] = %g, want %g", i, k, resp.Y.At(i, k), want)
			}
		}
	}
}

func TestSimulateInvertibleDescriptorWithInternalDelays(t *testing.T) {
	desc := absorbScopePlant(t, 1, true, true)
	explicit := desc.Copy()
	explicit.E = nil
	var lu mat.LU
	lu.Factorize(desc.E)
	for _, pair := range [][2]*mat.Dense{{explicit.A, desc.A}, {explicit.B, desc.B}, {explicit.LFT.B2, desc.LFT.B2}} {
		if err := lu.SolveTo(pair[0], false, pair[1]); err != nil {
			t.Fatal(err)
		}
	}
	u := mat.NewDense(2, 12, nil)
	for k := range 12 {
		u.Set(0, k, math.Cos(0.4*float64(k)))
		u.Set(1, k, 0.5-0.1*float64(k))
	}
	x0 := mat.NewVecDense(3, []float64{0.3, -1, 0.7})
	got, err := desc.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	want, err := explicit.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(got.Y, want.Y, 1e-12) || !mat.EqualApprox(got.XFinal, want.XFinal, 1e-12) {
		t.Fatalf("Y = %v\nwant %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulateSingularDescriptor(t *testing.T) {
	u := mat.NewDense(2, 15, nil)
	for k := range 15 {
		u.Set(0, k, math.Sin(0.9*float64(k)))
		u.Set(1, k, 1)
	}
	for _, delays := range []bool{false, true} {
		desc, oracle := index1Descriptor(t, 0.1, delays)
		got, err := desc.Simulate(u, nil, &SimulateOpts{Workspace: mat.NewVecDense(3, nil)})
		if err != nil {
			t.Fatalf("delays=%v: %v", delays, err)
		}
		want, err := oracle.Simulate(u, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !mat.EqualApprox(got.Y, want.Y, 1e-11) {
			t.Fatalf("delays=%v Y = %v\nwant %v", delays, mat.Formatted(got.Y), mat.Formatted(want.Y))
		}
		if got.XFinal != nil {
			t.Fatalf("XFinal = %v, want nil when not requested", got.XFinal)
		}
		if _, err := desc.Simulate(u, nil, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrDescriptorInitialState) {
			t.Fatalf("delays=%v FinalState err = %v, want ErrDescriptorInitialState", delays, err)
		}
		if _, err := desc.Simulate(u, mat.NewVecDense(3, []float64{1, 0, 0}), nil); !errors.Is(err, ErrDescriptorInitialState) {
			t.Fatalf("x0 err = %v, want ErrDescriptorInitialState", err)
		}
	}
	improper, _ := index2Descriptor(t, 0.1, true)
	if _, err := improper.Simulate(u, nil, nil); !errors.Is(err, ErrImproperModel) {
		t.Fatalf("non-causal err = %v, want ErrImproperModel", err)
	}
}

func TestSimulateAutonomousSteps(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.5, 0.2, -0.1, -0.3, 0.4, 0.25, 0.1, -0.2, 0.6})
	C := mat.NewDense(2, 3, []float64{1, -0.5, 0.3, 0.2, 1, -1})
	x0 := []float64{1, -2, 0.5}
	const steps = 7
	for _, p := range []int{2, 0} {
		var Cp *mat.Dense
		if p > 0 {
			Cp = C
		}
		sys, err := New(A, nil, Cp, nil, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		r, err := sys.Simulate(nil, mat.NewVecDense(3, append([]float64(nil), x0...)), &SimulateOpts{Steps: steps, FinalState: true})
		if err != nil {
			t.Fatalf("p=%d: %v", p, err)
		}
		x := mat.NewVecDense(3, append([]float64(nil), x0...))
		for k := range steps {
			for i := range p {
				want := mat.Dot(C.RowView(i), x)
				if math.Abs(r.Y.At(i, k)-want) > 1e-12 {
					t.Errorf("p=%d y[%d][%d] = %v, want C·A^%d·x0 = %v", p, i, k, r.Y.At(i, k), k, want)
				}
			}
			var next mat.VecDense
			next.MulVec(A, x)
			x = &next
		}
		if p == 0 && r.Y != nil && !r.Y.IsEmpty() {
			t.Errorf("p=0: Y = %v, want empty", r.Y)
		}
		if !vecEqual(r.XFinal, x, 1e-12) {
			t.Errorf("p=%d XFinal = %v, want A^%d·x0 = %v", p, mat.Formatted(r.XFinal.T()), steps, mat.Formatted(x.T()))
		}
	}
}

func TestSimulateStepsOptionValidation(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, -0.2, 0.3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0.5}), 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sys.Simulate(nil, nil, &SimulateOpts{Steps: 3}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Steps without u on m=1 model: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := sys.Simulate(mat.NewDense(1, 4, nil), nil, &SimulateOpts{Steps: 3}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Steps != u columns: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := sys.Simulate(mat.NewDense(1, 3, nil), nil, &SimulateOpts{Steps: -1}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("negative Steps: err = %v, want ErrInvalidArgument", err)
	}
	r, err := sys.Simulate(mat.NewDense(1, 3, []float64{1, 0, 0}), nil, &SimulateOpts{Steps: 3})
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(1, 3, []float64{0.5, 1, 0.3})
	if !matEqual(r.Y, want, 1e-12) {
		t.Errorf("Y = %v, want %v", mat.Formatted(r.Y), mat.Formatted(want))
	}
}

func TestSimulateNoOutputsPropagatesState(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.2, -0.3, 0.4})
	B := mat.NewDense(2, 1, []float64{1, -0.5})
	sys, err := New(A, B, nil, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	u := mat.NewDense(1, 5, []float64{1, -1, 0.5, 2, 0})
	x0 := mat.NewVecDense(2, []float64{0.3, -0.8})
	r, err := sys.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	x := mat.VecDenseCopyOf(x0)
	for k := range 5 {
		var next mat.VecDense
		next.MulVec(A, x)
		next.AddScaledVec(&next, u.At(0, k), B.ColView(0))
		x = &next
	}
	if r.Y != nil {
		t.Errorf("Y = %v, want nil", r.Y)
	}
	if !vecEqual(r.XFinal, x, 1e-12) {
		t.Errorf("XFinal = %v, want %v", mat.Formatted(r.XFinal.T()), mat.Formatted(x.T()))
	}
}

type delaySimOracle struct {
	A, B, C, D            *mat.Dense
	B2, C2, D12, D21, D22 *mat.Dense
	tau, inLag, outLag    []int
}

// run propagates x_{k+1} = A x + B ud + B2 w with explicit shift registers:
// ud_j(k) = u_j(k-inLag_j), w_l(k) = z_l(k-tau_l), y_i(k) = v_i(k-outLag_i).
func (o delaySimOracle) run(u *mat.Dense, x0 []float64) (*mat.Dense, []float64) {
	n, _ := o.A.Dims()
	m, steps := u.Dims()
	p, _ := o.C.Dims()
	N := len(o.tau)
	inReg := make([][]float64, m)
	for j := range m {
		inReg[j] = make([]float64, o.inLag[j]+1)
	}
	outReg := make([][]float64, p)
	for i := range p {
		outReg[i] = make([]float64, o.outLag[i]+1)
	}
	zReg := make([][]float64, N)
	for l := range N {
		zReg[l] = make([]float64, o.tau[l]+1)
	}
	shift := func(reg []float64, v float64) float64 {
		copy(reg[1:], reg[:len(reg)-1])
		reg[0] = v
		return reg[len(reg)-1]
	}
	x := append([]float64(nil), x0...)
	if x == nil {
		x = make([]float64, n)
	}
	Y := mat.NewDense(p, steps, nil)
	for k := range steps {
		ud := make([]float64, m)
		for j := range m {
			ud[j] = shift(inReg[j], u.At(j, k))
		}
		w := make([]float64, N)
		for l := range N {
			w[l] = zReg[l][len(zReg[l])-1]
		}
		for i := range p {
			v := 0.0
			for a := range n {
				v += o.C.At(i, a) * x[a]
			}
			for j := range m {
				v += o.D.At(i, j) * ud[j]
			}
			for l := range N {
				v += o.D12.At(i, l) * w[l]
			}
			Y.Set(i, k, shift(outReg[i], v))
		}
		z := make([]float64, N)
		for l := range N {
			for a := range n {
				z[l] += o.C2.At(l, a) * x[a]
			}
			for j := range m {
				z[l] += o.D21.At(l, j) * ud[j]
			}
			for r := range N {
				z[l] += o.D22.At(l, r) * w[r]
			}
		}
		next := make([]float64, n)
		for a := range n {
			for b := range n {
				next[a] += o.A.At(a, b) * x[b]
			}
			for j := range m {
				next[a] += o.B.At(a, j) * ud[j]
			}
			for l := range N {
				next[a] += o.B2.At(a, l) * w[l]
			}
		}
		x = next
		for l := range N {
			shift(zReg[l], 0)
			zReg[l][1] = z[l]
		}
	}
	return Y, x
}

func delayXFinalPlant() (A, B, C, D *mat.Dense, u *mat.Dense, x0 []float64) {
	A = mat.NewDense(3, 3, []float64{
		0.5, 0.2, -0.1,
		-0.3, 0.4, 0.25,
		0.1, -0.2, 0.6,
	})
	B = mat.NewDense(3, 2, []float64{1, 0.3, -0.5, 0.8, 0.2, -1})
	C = mat.NewDense(2, 3, []float64{1, 1, 0, 0.4, -0.7, 1.2})
	D = mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.5})
	u = mat.NewDense(2, 8, []float64{
		1, -1, 0.5, 2, 0, -0.7, 1.3, 0.4,
		0.2, 0.9, -1.1, 0.3, 1.5, -0.4, 0, 0.6,
	})
	return A, B, C, D, u, []float64{0.3, -0.8, 0.5}
}

func TestSimulateXFinalInputDelayRepro(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.2, -0.3, 0.4})
	B := mat.NewDense(2, 1, []float64{1, -0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, nil)
	sys, err := New(A, B, C, D, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	u := mat.NewDense(1, 5, []float64{1, -1, 0.5, 2, 0})
	r, err := sys.Simulate(u, mat.NewVecDense(2, []float64{0.3, -0.8}), &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	wantY := mat.NewDense(1, 5, []float64{-0.5, -0.42, -0.248, 0.386, -0.63812})
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("Y = %v, want %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}
	wantX := mat.NewVecDense(2, []float64{0.178723, -0.053391})
	if !vecEqual(r.XFinal, wantX, 1e-12) {
		t.Errorf("XFinal = %v, want %v", mat.Formatted(r.XFinal.T()), mat.Formatted(wantX.T()))
	}
}

func TestSimulateXFinalDelayedInputsMatchShiftRegisters(t *testing.T) {
	A, B, C, D, u, x0 := delayXFinalPlant()
	cases := []struct {
		name          string
		in, out       []float64
		delay         *mat.Dense
		inLag, outLag []int
	}{
		{name: "input", in: []float64{1, 3}, inLag: []int{1, 3}, outLag: []int{0, 0}},
		{name: "output", out: []float64{2, 0}, inLag: []int{0, 0}, outLag: []int{2, 0}},
		{name: "input+output", in: []float64{0, 2}, out: []float64{1, 3}, inLag: []int{0, 2}, outLag: []int{1, 3}},
		{name: "lag beyond horizon", in: []float64{9, 8}, inLag: []int{9, 8}, outLag: []int{0, 0}},
		{name: "decomposable Delay", delay: mat.NewDense(2, 2, []float64{1, 2, 4, 5}), inLag: []int{1, 2}, outLag: []int{0, 3}},
		{name: "Delay+InputDelay", in: []float64{2, 0}, delay: mat.NewDense(2, 2, []float64{0, 1, 2, 3}), inLag: []int{2, 1}, outLag: []int{0, 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewWithDelay(A, B, C, D, tc.delay, 1)
			if err != nil {
				t.Fatal(err)
			}
			if tc.in != nil {
				if err := sys.SetInputDelay(tc.in); err != nil {
					t.Fatal(err)
				}
			}
			if tc.out != nil {
				if err := sys.SetOutputDelay(tc.out); err != nil {
					t.Fatal(err)
				}
			}
			for _, withX0 := range []bool{false, true} {
				var x0v *mat.VecDense
				var ox0 []float64
				if withX0 {
					x0v, ox0 = mat.NewVecDense(3, append([]float64(nil), x0...)), x0
				}
				r, err := sys.Simulate(u, x0v, &SimulateOpts{FinalState: true})
				if err != nil {
					t.Fatal(err)
				}
				wantY, wantX := delaySimOracle{A: A, B: B, C: C, D: D, inLag: tc.inLag, outLag: tc.outLag}.run(u, nil)
				if withX0 {
					freeY, freeX := delaySimOracle{A: A, B: B, C: C, D: D, inLag: []int{0, 0}, outLag: tc.outLag}.run(mat.NewDense(2, 8, nil), ox0)
					wantY.Add(wantY, freeY)
					for i := range wantX {
						wantX[i] += freeX[i]
					}
				}
				if !matEqual(r.Y, wantY, 1e-12) {
					t.Errorf("x0=%v Y = %v, want %v", withX0, mat.Formatted(r.Y), mat.Formatted(wantY))
				}
				if !vecEqual(r.XFinal, mat.NewVecDense(3, wantX), 1e-12) {
					t.Errorf("x0=%v XFinal = %v, want %v", withX0, r.XFinal, wantX)
				}
				plain, err := sys.Simulate(u, x0v, nil)
				if err != nil {
					t.Fatal(err)
				}
				if plain.XFinal != nil || !matEqual(plain.Y, r.Y, 0) {
					t.Errorf("x0=%v unrequested: XFinal = %v, want nil; Y must match", withX0, plain.XFinal)
				}

				lft, err := sys.PullDelaysToLFT()
				if err != nil {
					t.Fatal(err)
				}
				rl, err := lft.Simulate(u, x0v, &SimulateOpts{FinalState: true})
				if err != nil {
					t.Fatal(err)
				}
				if !vecEqual(rl.XFinal, r.XFinal, 1e-12) {
					t.Errorf("x0=%v PullDelaysToLFT XFinal = %v, want %v", withX0, rl.XFinal, r.XFinal)
				}
				if !matEqual(rl.Y, r.Y, 1e-12) {
					t.Errorf("PullDelaysToLFT Y = %v, want %v", mat.Formatted(rl.Y), mat.Formatted(r.Y))
				}
			}
		})
	}
}

func TestSimulateXFinalNondecomposableDelayErrors(t *testing.T) {
	A, B, C, D, u, x0 := delayXFinalPlant()
	for _, internal := range []bool{false, true} {
		sys, err := NewWithDelay(A, B, C, D, mat.NewDense(2, 2, []float64{0, 2, 1, 0}), 1)
		if err != nil {
			t.Fatal(err)
		}
		if internal {
			if err := sys.SetInternalDelay([]float64{2},
				mat.NewDense(3, 1, []float64{0.4, -0.2, 0.1}),
				mat.NewDense(1, 3, []float64{0.3, 0.5, -0.6}),
				mat.NewDense(2, 1, []float64{0.2, -0.1}),
				mat.NewDense(1, 2, []float64{0.7, 0.1}),
				mat.NewDense(1, 1, nil)); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := sys.Simulate(u, mat.NewVecDense(3, x0), nil); !errors.Is(err, ErrDelayUnsupported) {
			t.Errorf("internal=%v: nonzero x0 err = %v, want ErrDelayUnsupported", internal, err)
		}
		r, err := sys.Simulate(u, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if r.Y == nil {
			t.Fatalf("internal=%v: Y = nil", internal)
		}
		if r.XFinal != nil {
			t.Errorf("internal=%v: XFinal = %v, want nil when not requested", internal, r.XFinal)
		}
		_, err = sys.Simulate(u, nil, &SimulateOpts{FinalState: true})
		if !errors.Is(err, ErrDelayUnsupported) {
			t.Errorf("internal=%v: FinalState err = %v, want ErrDelayUnsupported: no input+output split of Delay", internal, err)
		}
	}
}

func TestSimulateXFinalInternalAndIODelaysMatchShiftRegisters(t *testing.T) {
	A, B, C, D, u, x0 := delayXFinalPlant()
	B2 := mat.NewDense(3, 1, []float64{0.4, -0.2, 0.1})
	C2 := mat.NewDense(1, 3, []float64{0.3, 0.5, -0.6})
	D12 := mat.NewDense(2, 1, []float64{0.2, -0.1})
	D21 := mat.NewDense(1, 2, []float64{0.7, 0.1})
	D22 := mat.NewDense(1, 1, []float64{0.25})
	sys, err := New(A, B, C, D, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{1, 3}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{2, 0}); err != nil {
		t.Fatal(err)
	}
	r, err := sys.Simulate(u, mat.NewVecDense(3, append([]float64(nil), x0...)), &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	o := delaySimOracle{A: A, B: B, C: C, D: D, B2: B2, C2: C2, D12: D12, D21: D21, D22: D22,
		tau: []int{2}, inLag: []int{1, 3}, outLag: []int{2, 0}}
	wantY, wantX := o.run(u, x0)
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("Y = %v, want %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}
	if !vecEqual(r.XFinal, mat.NewVecDense(3, wantX), 1e-12) {
		t.Errorf("XFinal = %v, want %v", r.XFinal, wantX)
	}
}

func TestSimulateXFinalDelayedNoOutputAndNoInput(t *testing.T) {
	A, B, C, _, u, x0 := delayXFinalPlant()
	noOut, err := New(A, B, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := noOut.SetInputDelay([]float64{2, 1}); err != nil {
		t.Fatal(err)
	}
	r, err := noOut.Simulate(u, mat.NewVecDense(3, append([]float64(nil), x0...)), &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	o := delaySimOracle{A: A, B: B, C: mat.NewDense(1, 3, nil), D: mat.NewDense(1, 2, nil), inLag: []int{2, 1}, outLag: []int{0}}
	_, wantX := o.run(u, x0)
	if !vecEqual(r.XFinal, mat.NewVecDense(3, wantX), 1e-12) {
		t.Errorf("p=0 XFinal = %v, want %v", r.XFinal, wantX)
	}

	noIn, err := New(A, nil, C, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := noIn.SetOutputDelay([]float64{2, 1}); err != nil {
		t.Fatal(err)
	}
	const steps = 6
	r, err = noIn.Simulate(nil, mat.NewVecDense(3, append([]float64(nil), x0...)), &SimulateOpts{Steps: steps, FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	o = delaySimOracle{A: A, B: mat.NewDense(3, 1, nil), C: C, D: mat.NewDense(2, 1, nil), inLag: []int{0}, outLag: []int{2, 1}}
	wantY, wantX := o.run(mat.NewDense(1, steps, nil), x0)
	if !matEqual(r.Y, wantY, 1e-12) {
		t.Errorf("m=0 Y = %v, want %v", mat.Formatted(r.Y), mat.Formatted(wantY))
	}
	if !vecEqual(r.XFinal, mat.NewVecDense(3, wantX), 1e-12) {
		t.Errorf("m=0 XFinal = %v, want %v", r.XFinal, wantX)
	}
}

// x0DelayLFTRun is a shift-register recurrence for x+ = Ax + Bu + B2w,
// y = Cx + Du + D12w, z = C2x + D21u + D22w, w(k) = z(k-tau), with no I/O
// delays and zero delay-line history.
func x0DelayLFTRun(A, B, C, D, B2, C2, D12, D21, D22 *mat.Dense, tau []int, u *mat.Dense, x0 []float64) *mat.Dense {
	n, _ := A.Dims()
	m, steps := u.Dims()
	p, _ := C.Dims()
	N := len(tau)
	reg := make([][]float64, N)
	for l := range N {
		reg[l] = make([]float64, tau[l])
	}
	x := make([]float64, n)
	copy(x, x0)
	Y := mat.NewDense(p, steps, nil)
	for k := range steps {
		w := make([]float64, N)
		for l := range N {
			w[l] = reg[l][tau[l]-1]
		}
		for i := range p {
			v := 0.0
			for a := range n {
				v += C.At(i, a) * x[a]
			}
			for j := range m {
				v += D.At(i, j) * u.At(j, k)
			}
			for l := range N {
				v += D12.At(i, l) * w[l]
			}
			Y.Set(i, k, v)
		}
		z := make([]float64, N)
		for l := range N {
			for a := range n {
				z[l] += C2.At(l, a) * x[a]
			}
			for j := range m {
				z[l] += D21.At(l, j) * u.At(j, k)
			}
			for r := range N {
				z[l] += D22.At(l, r) * w[r]
			}
		}
		next := make([]float64, n)
		for a := range n {
			for b := range n {
				next[a] += A.At(a, b) * x[b]
			}
			for j := range m {
				next[a] += B.At(a, j) * u.At(j, k)
			}
			for l := range N {
				next[a] += B2.At(a, l) * w[l]
			}
		}
		x = next
		for l := range N {
			copy(reg[l][1:], reg[l][:tau[l]-1])
			reg[l][0] = z[l]
		}
	}
	return Y
}

// x0DelayOracle superposes the free response, delayed by the output share
// outLag of Delay, and each input's forced response, delayed per channel by
// Delay(i,j).
func x0DelayOracle(lft func(u *mat.Dense, x0 []float64) *mat.Dense, delay *mat.Dense, outLag []int, u *mat.Dense, x0 []float64) *mat.Dense {
	m, steps := u.Dims()
	p, _ := delay.Dims()
	Y := mat.NewDense(p, steps, nil)
	free := lft(mat.NewDense(m, steps, nil), x0)
	for i := range p {
		for k := outLag[i]; k < steps; k++ {
			Y.Set(i, k, free.At(i, k-outLag[i]))
		}
	}
	for j := range m {
		uj := mat.NewDense(m, steps, nil)
		for k := range steps {
			uj.Set(j, k, u.At(j, k))
		}
		forced := lft(uj, nil)
		for i := range p {
			d := int(delay.At(i, j))
			for k := d; k < steps; k++ {
				Y.Set(i, k, Y.At(i, k)+forced.At(i, k-d))
			}
		}
	}
	return Y
}

func TestSimulateX0IODelayOutputShare(t *testing.T) {
	A, B, C, D, u, x0s := delayXFinalPlant()
	x0 := mat.NewVecDense(3, x0s)
	delay := mat.NewDense(2, 2, []float64{1, 2, 4, 5})
	outLag := []int{0, 3}
	B2 := mat.NewDense(3, 1, []float64{0.4, -0.2, 0.3})
	C2 := mat.NewDense(1, 3, []float64{0.5, 0.1, -0.6})
	D12 := mat.NewDense(2, 1, []float64{0.2, -0.3})
	D21 := mat.NewDense(1, 2, []float64{0.1, 0.7})
	D22 := mat.NewDense(1, 1, []float64{0.25})

	for _, internal := range []bool{false, true} {
		sys, err := NewWithDelay(A, B, C, D, delay, 1)
		if err != nil {
			t.Fatal(err)
		}
		b2, c2, d12, d21, d22 := mat.NewDense(3, 1, nil), mat.NewDense(1, 3, nil), mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil)
		if internal {
			b2, c2, d12, d21, d22 = B2, C2, D12, D21, D22
			if err := sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22); err != nil {
				t.Fatal(err)
			}
		}
		lft := func(u *mat.Dense, x0 []float64) *mat.Dense {
			return x0DelayLFTRun(A, B, C, D, b2, c2, d12, d21, d22, []int{2}, u, x0)
		}
		want := x0DelayOracle(lft, delay, outLag, u, x0.RawVector().Data)

		got, err := sys.Simulate(u, x0, nil)
		if err != nil {
			t.Fatalf("internal=%v: %v", internal, err)
		}
		if !matEqual(got.Y, want, 1e-12) {
			t.Errorf("internal=%v: Simulate Y =\n%v\nwant\n%v", internal, mat.Formatted(got.Y), mat.Formatted(want))
		}
		merged, err := sys.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		if mn, _, _ := merged.Dims(); mn != 3 {
			t.Fatalf("internal=%v: PullDelaysToLFT states %d, want 3", internal, mn)
		}
		viaLFT, err := merged.Simulate(u, x0, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !matEqual(viaLFT.Y, want, 1e-12) {
			t.Errorf("internal=%v: PullDelaysToLFT Y =\n%v\nwant\n%v", internal, mat.Formatted(viaLFT.Y), mat.Formatted(want))
		}

		tv := make([]float64, 8)
		for k := range tv {
			tv[k] = float64(k)
		}
		lr, err := Lsim(sys, mat.DenseCopyOf(u.T()), tv, x0)
		if err != nil {
			t.Fatal(err)
		}
		if !matEqual(lr.Y, want, 1e-12) {
			t.Errorf("internal=%v: Lsim Y =\n%v\nwant\n%v", internal, mat.Formatted(lr.Y), mat.Formatted(want))
		}
		ir, err := Initial(sys, x0, 7)
		if err != nil {
			t.Fatal(err)
		}
		wantFree := x0DelayOracle(lft, delay, outLag, mat.NewDense(2, 8, nil), x0.RawVector().Data)
		if !matEqual(ir.Y, wantFree, 1e-12) {
			t.Errorf("internal=%v: Initial Y =\n%v\nwant\n%v", internal, mat.Formatted(ir.Y), mat.Formatted(wantFree))
		}
	}
}

func TestSimulateX0NondecomposableIODelay(t *testing.T) {
	A, B, C, D, u, x0s := delayXFinalPlant()
	x0 := mat.NewVecDense(3, x0s)
	sys, err := NewWithDelay(A, B, C, D, mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 1)
	if err != nil {
		t.Fatal(err)
	}
	tv := make([]float64, 8)
	for k := range tv {
		tv[k] = float64(k)
	}
	if _, err := sys.Simulate(u, x0, nil); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("Simulate err = %v, want ErrDelayUnsupported", err)
	}
	if _, err := Initial(sys, x0, 7); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("Initial err = %v, want ErrDelayUnsupported", err)
	}
	if _, err := Lsim(sys, mat.DenseCopyOf(u.T()), tv, x0); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("Lsim err = %v, want ErrDelayUnsupported", err)
	}
	zero := mat.NewVecDense(3, nil)
	got, err := sys.Simulate(u, zero, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(got.Y, want.Y, 0) {
		t.Errorf("zero x0 Y = %v, want %v", mat.Formatted(got.Y), mat.Formatted(want.Y))
	}
}

func TestSimulateNoOutputs(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, 0.3}), mat.NewDense(2, 1, []float64{1, 0.5}), nil, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	u := mat.NewDense(1, 3, []float64{1, -1, 2})
	if _, err := sys.Simulate(u, nil, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("p=0 without FinalState error = %v, want ErrDimensionMismatch", err)
	}
	r, err := sys.Simulate(u, nil, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	x := mat.NewVecDense(2, nil)
	for k := range 3 {
		var next mat.VecDense
		next.MulVec(sys.A, x)
		next.AddScaledVec(&next, u.At(0, k), sys.B.ColView(0))
		x = &next
	}
	if r.Y != nil || !vecEqual(r.XFinal, x, 1e-14) {
		t.Errorf("p=0 FinalState: Y=%v XFinal=%v, want nil, %v", r.Y, r.XFinal, x)
	}
}

func TestSimulateXFinalDoesNotAliasWorkspace(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	ws := mat.NewVecDense(1, nil)
	opts := &SimulateOpts{Workspace: ws, FinalState: true}
	r1, err := sys.Simulate(mat.NewDense(1, 1, []float64{1}), nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if r1.XFinal == ws {
		t.Fatal("XFinal is the caller's Workspace")
	}
	if _, err := sys.Simulate(mat.NewDense(1, 1, []float64{7}), nil, opts); err != nil {
		t.Fatal(err)
	}
	if got := r1.XFinal.AtVec(0); got != 1 {
		t.Errorf("first XFinal changed to %g by second call, want 1", got)
	}
}

func TestSimulateRejectsInvalidModel(t *testing.T) {
	var nilSys *System
	if _, err := nilSys.Simulate(nil, nil, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Simulate = %v, want ErrInvalidArgument", err)
	}
	cont, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = cont.Simulate(mat.NewDense(1, 2, nil), nil, nil)
	if !errors.Is(err, ErrWrongDomain) || !strings.HasPrefix(err.Error(), "Simulate: ") {
		t.Errorf("continuous Simulate = %v, want Simulate: ... ErrWrongDomain", err)
	}
}
