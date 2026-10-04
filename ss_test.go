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

func TestIsStable_BoundaryPolesFromNonNormalA(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for k := range 200 {
			w := 0.37 + 0.01*float64(k)
			st, err := boundaryOscillator(t, w, dt).IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if st {
				t.Fatalf("dt=%v w=%g: IsStable=true for poles on the stability boundary", dt, w)
			}
		}
	}
}

func TestIsStable_ConsistentWithIsStabilizable(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1e-12})
	sys, err := New(A, mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	st, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	stz, err := IsStabilizable(A, mat.NewDense(1, 1, nil), true)
	if err != nil {
		t.Fatal(err)
	}
	if st != stz {
		t.Errorf("IsStable=%v but IsStabilizable(B=0)=%v for pole -1e-12", st, stz)
	}
}

// discreteLFTDelayFixture returns a 2x2 discrete model with one internal
// delay of 3 samples and its hand-absorbed state-space realization
// x̃ = [x; s1; s2; s3], s1⁺ = z, s2⁺ = s1, s3⁺ = s2, w = s3.
func discreteLFTDelayFixture(t *testing.T, b2 []float64) (lft, hand *System) {
	t.Helper()
	a := []float64{0.5, 0.2, -0.1, 0.7}
	b1 := []float64{1, 0.3, 0, 1}
	c1 := []float64{1, 0, 0.4, 1}
	d11 := []float64{0, 0.1, 0, 0}
	c2 := []float64{0.3, 1}
	d12 := []float64{0.2, 0}
	d21 := []float64{0, 0.1}
	d22 := 0.1
	lft, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 2, b1), mat.NewDense(2, 2, c1), mat.NewDense(2, 2, d11), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := lft.SetInternalDelay([]float64{3}, mat.NewDense(2, 1, b2), mat.NewDense(1, 2, c2),
		mat.NewDense(2, 1, d12), mat.NewDense(1, 2, d21), mat.NewDense(1, 1, []float64{d22})); err != nil {
		t.Fatal(err)
	}
	hand, err = New(
		mat.NewDense(5, 5, []float64{
			a[0], a[1], 0, 0, b2[0],
			a[2], a[3], 0, 0, b2[1],
			c2[0], c2[1], 0, 0, d22,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
		}),
		mat.NewDense(5, 2, []float64{b1[0], b1[1], b1[2], b1[3], d21[0], d21[1], 0, 0, 0, 0}),
		mat.NewDense(2, 5, []float64{c1[0], c1[1], 0, 0, d12[0], c1[2], c1[3], 0, 0, d12[1]}),
		mat.NewDense(2, 2, d11), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	return lft, hand
}

func spectralRadius(t *testing.T, a *mat.Dense) float64 {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(a, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	r := 0.0
	for _, v := range eig.Values(nil) {
		r = math.Max(r, cmplx.Abs(v))
	}
	return r
}

// scalarDDE returns x' = -x + b·x(t-tau) + u, y = x as an internal-delay LFT.
func scalarDDE(t *testing.T, b, tau float64) *System {
	t.Helper()
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{tau}, mat.NewDense(1, 1, []float64{b}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0})); err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestIsStable_DiscreteInternalDelayUsesAbsorbedPoles(t *testing.T) {
	for _, tc := range []struct {
		name string
		b2   []float64
	}{
		{"unstable delay loop", []float64{1, 0.5}},
		{"stable delay loop", []float64{0.1, 0.05}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lft, hand := discreteLFTDelayFixture(t, tc.b2)
			want := spectralRadius(t, hand.A) < 1
			got, err := lft.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("IsStable = %v, want %v (absorbed spectral radius %.6g)", got, want, spectralRadius(t, hand.A))
			}
		})
	}
	lft, hand := discreteLFTDelayFixture(t, []float64{1, 0.5})
	if r := spectralRadius(t, hand.A); r < 1.1 {
		t.Fatalf("fixture spectral radius %.6g, want unstable delay loop", r)
	}
	if r := spectralRadius(t, lft.A); r >= 1 {
		t.Fatalf("fixture A spectral radius %.6g, want stable delay-free part", r)
	}
}

// x' = -x + b·x(t-tau) with b < -1 is stable iff tau < arccos(-1/b)/sqrt(b²-1);
// for b = -2 the critical delay is 1.2092. Neither eig(A) nor the zero-delay
// model (pole -3) decides it, so IsStable must not answer.
func TestIsStable_ContinuousInternalDelayRejected(t *testing.T) {
	for _, tau := range []float64{0.5, 2} {
		sys := scalarDDE(t, -2, tau)
		if _, err := sys.IsStable(); !errors.Is(err, ErrContinuousInternalDelay) {
			t.Fatalf("tau=%g: err = %v, want ErrContinuousInternalDelay", tau, err)
		}
	}
	sys := scalarDDE(t, -2, 2)
	sys.LFT = nil
	if err := sys.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	if stable, err := sys.IsStable(); err != nil || !stable {
		t.Fatalf("input delay only: stable=%v err=%v, want true", stable, err)
	}
}
