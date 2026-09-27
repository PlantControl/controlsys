package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestNewBasic(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("Dims() = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
	if !sys.IsContinuous() {
		t.Error("expected continuous")
	}
}

func TestNewDiscrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		nil, 0.01,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.IsDiscrete() {
		t.Error("expected discrete")
	}
	if sys.Dt != 0.01 {
		t.Errorf("Dt = %v, want 0.01", sys.Dt)
	}
}

func TestNewNilD(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sys.D.At(0, 0) != 0 {
		t.Error("nil D should produce zero matrix")
	}
}

func TestNewDimensionMismatch(t *testing.T) {
	_, err := New(
		mat.NewDense(2, 2, nil),
		mat.NewDense(3, 1, nil),
		nil, nil, 0,
	)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestNewNonSquareA(t *testing.T) {
	_, err := New(mat.NewDense(2, 3, nil), nil, nil, nil, 0)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestNewInvalidSampleTime(t *testing.T) {
	_, err := New(nil, nil, nil, nil, -1)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
}

func TestNewGain(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 0 || m != 3 || p != 2 {
		t.Errorf("Dims() = (%d,%d,%d), want (0,3,2)", n, m, p)
	}
}

func TestNewCopiesInputMatrices(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	A.Set(0, 0, 99)
	B.Set(0, 0, 99)
	C.Set(0, 0, 99)
	D.Set(0, 0, 99)

	if got := sys.A.At(0, 0); got != 0 {
		t.Fatalf("A alias detected: got %v, want 0", got)
	}
	if got := sys.B.At(0, 0); got != 0 {
		t.Fatalf("B alias detected: got %v, want 0", got)
	}
	if got := sys.C.At(0, 0); got != 1 {
		t.Fatalf("C alias detected: got %v, want 1", got)
	}
	if got := sys.D.At(0, 0); got != 0 {
		t.Fatalf("D alias detected: got %v, want 0", got)
	}
}

func TestNewFromSlices(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("Dims() = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
}

func TestNewFromSlicesGain(t *testing.T) {
	sys, err := NewFromSlices(0, 2, 1, nil, nil, nil, []float64{3, 4}, 0)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	if n != 0 || m != 2 || p != 1 {
		t.Errorf("Dims() = (%d,%d,%d), want (0,2,1)", n, m, p)
	}
}

func TestNewGainCopiesInputMatrix(t *testing.T) {
	D := mat.NewDense(1, 2, []float64{3, 4})

	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}

	D.Set(0, 0, 99)
	if got := sys.D.At(0, 0); got != 3 {
		t.Fatalf("D alias detected: got %v, want 3", got)
	}
}

func TestNewFromSlicesGainOnlyCopiesInputSlice(t *testing.T) {
	d := []float64{3, 4}

	sys, err := NewFromSlices(0, 2, 1, nil, nil, nil, d, 0)
	if err != nil {
		t.Fatal(err)
	}

	d[0] = 99
	if got := sys.D.At(0, 0); got != 3 {
		t.Fatalf("D alias detected: got %v, want 3", got)
	}
}

func TestNewFromSlicesDynamicCopiesInputSlices(t *testing.T) {
	a := []float64{0, 1, -2, -3}
	b := []float64{0, 1}
	c := []float64{1, 0}
	d := []float64{5}

	sys, err := NewFromSlices(2, 1, 1, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}

	a[0] = 99
	b[0] = 99
	c[0] = 99
	d[0] = 99

	if got := sys.A.At(0, 0); got != 0 {
		t.Fatalf("A alias detected: got %v, want 0", got)
	}
	if got := sys.B.At(0, 0); got != 0 {
		t.Fatalf("B alias detected: got %v, want 0", got)
	}
	if got := sys.C.At(0, 0); got != 1 {
		t.Fatalf("C alias detected: got %v, want 1", got)
	}
	if got := sys.D.At(0, 0); got != 5 {
		t.Fatalf("D alias detected: got %v, want 5", got)
	}
}

func TestCopy(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	cp := sys.Copy()
	sys.A.Set(0, 0, 999)
	if cp.A.At(0, 0) == 999 {
		t.Error("Copy should be independent")
	}
}

func TestPoles(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("expected 2 poles, got %d", len(poles))
	}
	found := [2]bool{}
	for _, p := range poles {
		if cmplx.Abs(p-(-1)) < 1e-10 {
			found[0] = true
		}
		if cmplx.Abs(p-(-2)) < 1e-10 {
			found[1] = true
		}
	}
	if !found[0] || !found[1] {
		t.Errorf("expected poles -1,-2, got %v", poles)
	}
}

func TestPolesEmpty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 0 {
		t.Errorf("expected no poles for gain system, got %v", poles)
	}
}

func TestIsStableContinuous(t *testing.T) {
	stable, _ := NewFromSlices(2, 1, 1,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	if s, err := stable.IsStable(); err != nil {
		t.Fatal(err)
	} else if !s {
		t.Error("expected stable")
	}

	unstable, _ := NewFromSlices(2, 1, 1,
		[]float64{1, 0, 0, -2},
		[]float64{1, 0},
		[]float64{1, 0},
		[]float64{0},
		0,
	)
	if s, err := unstable.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected unstable")
	}
}

func TestIsStableDiscrete(t *testing.T) {
	stable, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5},
		[]float64{1},
		[]float64{1},
		[]float64{0},
		1,
	)
	if s, err := stable.IsStable(); err != nil {
		t.Fatal(err)
	} else if !s {
		t.Error("expected stable discrete")
	}

	unstable, _ := NewFromSlices(1, 1, 1,
		[]float64{1.5},
		[]float64{1},
		[]float64{1},
		[]float64{0},
		1,
	)
	if s, err := unstable.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected unstable discrete")
	}
}

func TestIsStableGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{100}), 0)
	if s, err := sys.IsStable(); err != nil {
		t.Fatal(err)
	} else if !s {
		t.Error("gain system should be stable (no poles)")
	}
}

func TestDimsConsistency(t *testing.T) {
	sys, _ := NewFromSlices(3, 2, 4,
		make([]float64, 9),
		make([]float64, 6),
		make([]float64, 12),
		make([]float64, 8),
		0,
	)
	n, m, p := sys.Dims()
	if n != 3 || m != 2 || p != 4 {
		t.Errorf("Dims() = (%d,%d,%d), want (3,2,4)", n, m, p)
	}
	ar, ac := sys.A.Dims()
	br, bc := sys.B.Dims()
	cr, cc := sys.C.Dims()
	dr, dc := sys.D.Dims()
	if ar != n || ac != n {
		t.Errorf("A: %d×%d, want %d×%d", ar, ac, n, n)
	}
	if br != n || bc != m {
		t.Errorf("B: %d×%d, want %d×%d", br, bc, n, m)
	}
	if cr != p || cc != n {
		t.Errorf("C: %d×%d, want %d×%d", cr, cc, p, n)
	}
	if dr != p || dc != m {
		t.Errorf("D: %d×%d, want %d×%d", dr, dc, p, m)
	}
}

func TestSystemValidate(t *testing.T) {
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
	if err := sys.Validate(); err != nil {
		t.Fatalf("valid system failed Validate: %v", err)
	}

	sys.E = mat.NewDense(1, 1, nil)
	if err := sys.Validate(); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("E: got %v, want ErrDimensionMismatch", err)
	}

	sys.E = nil
	sys.LFT = &LFTDelay{Tau: []float64{0}}
	if err := sys.Validate(); !errors.Is(err, ErrZeroInternalDelay) {
		t.Fatalf("LFT tau: got %v, want ErrZeroInternalDelay", err)
	}
}

func TestIsStableMarginal(t *testing.T) {
	// Pole on imaginary axis (marginally stable = not stable)
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -1, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		nil,
		0,
	)
	if s, err := sys.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("marginally stable (jω axis poles) should not be stable")
	}
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}
