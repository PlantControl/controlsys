package controlsys

import (
	"errors"
	"math"
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

	r, err := sys.Simulate(u, x0, nil)
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

	r1, err := sys.Simulate(u1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := sys.Simulate(u2, r1.XFinal, nil)
	if err != nil {
		t.Fatal(err)
	}

	uAll := mat.NewDense(1, 5, []float64{1, 2, 3, 4, 5})
	rAll, err := sys.Simulate(uAll, nil, nil)
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
		t.Errorf("expected nil XFinal for n=0, got %v", r.XFinal)
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

	// Need to pass u=nil but steps would be 0. For autonomous propagation we need
	// a way to specify steps. With u=nil, steps=0 so we get empty Y.
	// Instead, pass a 0-row input with the desired number of columns.
	// Actually m=0 for this system, so we can't create a 0×steps matrix.
	// The spec says u==nil means zero input → 0 steps.
	// For m=0 systems, we must construct a dummy input to set step count.
	// Let's test with u=nil → steps=0, XFinal=x0.
	r, err := sys.Simulate(nil, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Y != nil {
		t.Errorf("expected nil Y for 0 steps")
	}
	if !vecEqual(r.XFinal, x0, 1e-12) {
		t.Errorf("XFinal should equal x0 when steps=0")
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

	r, err := sys.Simulate(u, x0, nil)
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

	// u with 0 columns → steps=0
	r, err := sys.Simulate(nil, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Y != nil {
		t.Errorf("expected nil Y for steps=0")
	}
	if !vecEqual(r.XFinal, x0, 1e-12) {
		t.Errorf("XFinal should equal x0 for steps=0")
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
	r, err := sys.Simulate(u, nil, nil)
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
	got, err := desc.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := explicit.Simulate(u, x0, nil)
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
			t.Fatalf("XFinal = %v, want nil for singular E", got.XFinal)
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
