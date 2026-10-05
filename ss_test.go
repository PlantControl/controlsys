package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"
	"testing"
	"time"

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

func TestPoles_InternalDelaySetToZero(t *testing.T) {
	singularE := mat.NewDense(3, 3, []float64{1, 2, 0, 0, 1, 1, 1, 3, 1})
	for _, E := range []*mat.Dense{nil, singularE} {
		for _, m := range []int{1, 2} {
			for _, dt := range []float64{0, 0.1} {
				sys, closed := internalDelayZerosFixture(t, dt, m, mat.NewDense(1, 1, []float64{0.25}))
				var want []complex128
				if E != nil {
					sys.E = mat.DenseCopyOf(E)
					ev, err := generalizedPoles(closed.A, E, 3)
					if err != nil {
						t.Fatal(err)
					}
					for _, v := range ev {
						if cmplx.Abs(v) < 1e6 {
							want = append(want, v)
						}
					}
				} else {
					var eig mat.Eigen
					if !eig.Factorize(closed.A, mat.EigenNone) {
						t.Fatal("eigen failed")
					}
					want = eig.Values(nil)
				}
				got, err := sys.Poles()
				if err != nil {
					t.Fatalf("E=%v m=%d dt=%v: %v", E != nil, m, dt, err)
				}
				var finite []complex128
				for _, p := range got {
					if !cmplx.IsInf(p) && !cmplx.IsNaN(p) && cmplx.Abs(p) < 1e6 {
						finite = append(finite, p)
					}
				}
				assertZerosMatch(t, finite, want, 1e-9)

				damp, err := Damp(sys)
				if err != nil {
					t.Fatal(err)
				}
				pz, err := Pzmap(sys)
				if err != nil {
					t.Fatal(err)
				}
				if len(damp) != len(got) || len(pz.Poles) != len(got) {
					t.Fatalf("Damp %d / Pzmap %d poles, want %d", len(damp), len(pz.Poles), len(got))
				}
				for i := range got {
					if damp[i].Pole != got[i] && !(cmplx.IsNaN(got[i]) || cmplx.IsInf(got[i])) {
						t.Errorf("Damp pole %v != Poles %v", damp[i].Pole, got[i])
					}
				}
			}
		}
	}
}

func TestPoles_InternalDelayAlgebraicLoop(t *testing.T) {
	sys, _ := internalDelayZerosFixture(t, 0, 2, mat.NewDense(1, 1, []float64{1}))
	if _, err := sys.Poles(); !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("err %v, want ErrAlgebraicLoop", err)
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

func emptyIOFixture(t *testing.T, n, m, p int, dt float64) *System {
	t.Helper()
	a := make([]float64, n*n)
	for i := range n {
		a[i*n+i] = -1 - float64(i)
		if i+1 < n {
			a[i*n+i+1] = 0.7
			a[(i+1)*n+i] = -0.2
		}
	}
	if dt > 0 {
		for i := range a {
			a[i] *= 0.2
		}
		for i := range n {
			a[i*n+i] += 1
		}
	}
	b := make([]float64, n*m)
	for i := range b {
		b[i] = float64(i%3) + 0.5
	}
	c := make([]float64, p*n)
	for i := range c {
		c[i] = 1 - 0.3*float64(i%4)
	}
	d := make([]float64, p*m)
	for i := range d {
		d[i] = 0.1 * float64(i+1)
	}
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

type emptyIOOp struct {
	name string
	fn   func(sys *System) (any, error)
}

func emptyIOOps() []emptyIOOp {
	w := []float64{0.1, 1, 3}
	tGrid := func(sys *System) []float64 {
		if sys.IsDiscrete() {
			return []float64{0, sys.Dt, 2 * sys.Dt, 3 * sys.Dt}
		}
		return []float64{0, 0.1, 0.2, 0.3}
	}
	x0Of := func(sys *System) *mat.VecDense {
		n, _, _ := sys.Dims()
		if n == 0 {
			return nil
		}
		x0 := mat.NewVecDense(n, nil)
		x0.SetVec(0, 1)
		return x0
	}
	return []emptyIOOp{
		{"Poles", func(s *System) (any, error) { return s.Poles() }},
		{"IsStable", func(s *System) (any, error) { return s.IsStable() }},
		{"Zeros", func(s *System) (any, error) { return s.Zeros() }},
		{"ZerosDetail", func(s *System) (any, error) { return s.ZerosDetail() }},
		{"DCGain", func(s *System) (any, error) { return s.DCGain() }},
		{"FreqResponse", func(s *System) (any, error) { return s.FreqResponse(w) }},
		{"FreqResponsePointwise", func(s *System) (any, error) { return s.FreqResponsePointwise(w) }},
		{"Bode", func(s *System) (any, error) { return s.Bode(w, 0) }},
		{"BodeAuto", func(s *System) (any, error) { return s.Bode(nil, 20) }},
		{"Sigma", func(s *System) (any, error) { return s.Sigma(nil, 20) }},
		{"Nyquist", func(s *System) (any, error) { return s.Nyquist(nil, 20) }},
		{"Nichols", func(s *System) (any, error) { return s.Nichols(nil, 20) }},
		{"EvalFr", func(s *System) (any, error) { return s.EvalFr(1i) }},
		{"FRD", func(s *System) (any, error) { return s.FRD(w) }},
		{"H2Norm", func(s *System) (any, error) { return H2Norm(s) }},
		{"HinfNorm", func(s *System) (any, error) { g, _, err := HinfNorm(s); return g, err }},
		{"NormInf", func(s *System) (any, error) { return Norm(s, math.Inf(1)) }},
		{"HSV", func(s *System) (any, error) { return HSV(s) }},
		{"GramC", func(s *System) (any, error) { return Gram(s, GramControllability) }},
		{"GramO", func(s *System) (any, error) { return Gram(s, GramObservability) }},
		{"Balreal", func(s *System) (any, error) { return Balreal(s) }},
		{"Balred", func(s *System) (any, error) {
			r, _, err := Balred(s, 1, BalredOptions{StateProjection: Truncate})
			return r, err
		}},
		{"BalredSP", func(s *System) (any, error) {
			r, _, err := Balred(s, 1, BalredOptions{StateProjection: MatchDC})
			return r, err
		}},
		{"Modred", func(s *System) (any, error) { return Modred(s, []int{0}, Truncate) }},
		{"ModredMatchDC", func(s *System) (any, error) { return Modred(s, []int{0}, MatchDC) }},
		{"CanonModal", func(s *System) (any, error) { return Canon(s, CanonModal) }},
		{"CanonCompanion", func(s *System) (any, error) { return Canon(s, CanonCompanion) }},
		{"Ssbal", func(s *System) (any, error) { return Ssbal(s) }},
		{"Prescale", func(s *System) (any, error) { return Prescale(s) }},
		{"Stabsep", func(s *System) (any, error) { return Stabsep(s) }},
		{"Modsep", func(s *System) (any, error) { return Modsep(s, 1.5) }},
		{"Sminreal", func(s *System) (any, error) { return Sminreal(s) }},
		{"MinimalRealization", func(s *System) (any, error) { return s.MinimalRealization() }},
		{"Reduce", func(s *System) (any, error) { return s.Reduce(nil) }},
		{"ModalTruncate", func(s *System) (any, error) { return ModalTruncate(s, &ModalTruncateOptions{Order: 1}) }},
		{"Augstate", func(s *System) (any, error) { return Augstate(s) }},
		{"Inv", func(s *System) (any, error) { return Inv(s) }},
		{"Series", func(s *System) (any, error) { return Series(s, s.Copy()) }},
		{"Parallel", func(s *System) (any, error) { return Parallel(s, s.Copy()) }},
		{"Append", func(s *System) (any, error) { return Append(s, s.Copy()) }},
		{"BlkDiag", func(s *System) (any, error) { return BlkDiag(s, s.Copy()) }},
		{"Feedback", func(s *System) (any, error) { return Feedback(s, s.Copy(), -1) }},
		{"FeedbackGain", func(s *System) (any, error) {
			_, m, p := s.Dims()
			k, err := NewGain(newDense(m, p), s.Dt)
			if err != nil {
				return nil, err
			}
			return Feedback(s, k, -1)
		}},
		{"Loopsens", func(s *System) (any, error) { return Loopsens(s, s.Copy()) }},
		{"Connect", func(s *System) (any, error) {
			_, m, _ := s.Dims()
			return Connect(s, newDense(m, 2), nil, nil)
		}},
		{"Margin", func(s *System) (any, error) { return Margin(s) }},
		{"AllMargin", func(s *System) (any, error) { return AllMargin(s) }},
		{"DiskMargin", func(s *System) (any, error) { return DiskMargin(s) }},
		{"Bandwidth", func(s *System) (any, error) { return Bandwidth(s, -3) }},
		{"Damp", func(s *System) (any, error) { return Damp(s) }},
		{"Pzmap", func(s *System) (any, error) { return Pzmap(s) }},
		{"Step", func(s *System) (any, error) { return Step(s, 0) }},
		{"Impulse", func(s *System) (any, error) { return Impulse(s, 0) }},
		{"Initial", func(s *System) (any, error) { return Initial(s, x0Of(s), 0) }},
		{"Lsim", func(s *System) (any, error) {
			_, m, _ := s.Dims()
			var u *mat.Dense
			if m > 0 {
				u = mat.NewDense(4, m, nil)
				u.Set(1, 0, 1)
			}
			return Lsim(s, u, tGrid(s), x0Of(s))
		}},
		{"Simulate", func(s *System) (any, error) {
			_, m, _ := s.Dims()
			var u *mat.Dense
			if m > 0 {
				u = mat.NewDense(m, 4, nil)
			}
			return s.Simulate(u, x0Of(s), nil)
		}},
		{"StepInfoForSystem", func(s *System) (any, error) { return StepInfoForSystem(s, 0, nil) }},
		{"TransferFunction", func(s *System) (any, error) { return s.TransferFunction(nil) }},
		{"ZPKModel", func(s *System) (any, error) { return s.ZPKModel(nil) }},
		{"RootLocus", func(s *System) (any, error) { return RootLocus(s, nil) }},
		{"Passive", func(s *System) (any, error) { return Passive(s, nil) }},
		{"SpectralFactor", func(s *System) (any, error) { return SpectralFactor(s) }},
		{"Discretize", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return s.D2C(D2COptions{Method: C2DMethodTustin})
			}
			return s.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
		}},
		{"DiscretizeFOH", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return s.D2D(0.2, D2DOptions{})
			}
			return s.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
		}},
		{"DiscretizeTustin", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return s.D2C(D2COptions{Method: C2DMethodTustin})
			}
			return s.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
		}},
		{"DiscretizeImpulse", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return nil, nil
			}
			return s.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
		}},
		{"DiscretizeMatched", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return nil, nil
			}
			return s.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
		}},
		{"String", func(s *System) (any, error) { return s.String(), nil }},
		{"IsProper", func(s *System) (any, error) { return s.IsProper() }},
		{"Covar", func(s *System) (any, error) {
			_, m, _ := s.Dims()
			return Covar(s, eyeOrEmptyDense(m))
		}},
		{"SS2SS", func(s *System) (any, error) {
			n, _, _ := s.Dims()
			T := eyeOrEmptyDense(n)
			for i := range n {
				T.Set(i, i, 2)
				if i+1 < n {
					T.Set(i, i+1, 1)
				}
			}
			return SS2SS(s, T)
		}},
		{"Xperm", func(s *System) (any, error) {
			n, _, _ := s.Dims()
			perm := make([]int, n)
			for i := range n {
				perm[i] = n - 1 - i
			}
			return Xperm(s, perm)
		}},
		{"SelectByIndex", func(s *System) (any, error) { return s.SelectByIndex(nil, nil) }},
		{"Pade", func(s *System) (any, error) { return s.Pade(2) }},
		{"AbsorbDelay", func(s *System) (any, error) { return s.AbsorbDelay() }},
		{"ToExplicit", func(s *System) (any, error) { return s.ToExplicit() }},
		{"Estim", func(s *System) (any, error) {
			n, _, p := s.Dims()
			return Estim(s, newDense(n, p))
		}},
		{"Kalmd", func(s *System) (any, error) {
			if s.IsDiscrete() {
				return nil, nil
			}
			_, m, p := s.Dims()
			return Kalmd(s, eyeOrEmptyDense(m), eyeOrEmptyDense(p), 0.1, nil)
		}},
		{"Reg", func(s *System) (any, error) {
			n, m, p := s.Dims()
			return Reg(s, newDense(m, n), newDense(n, p))
		}},
		{"Lqg", func(s *System) (any, error) {
			n, m, p := s.Dims()
			return Lqg(s, eyeOrEmptyDense(n+m), eyeOrEmptyDense(n+p), nil)
		}},
		{"Ctrb", func(s *System) (any, error) { return Ctrb(s.A, s.B) }},
		{"Obsv", func(s *System) (any, error) { return Obsv(s.A, s.C) }},
		{"CtrbF", func(s *System) (any, error) { return CtrbF(s.A, s.B, s.C) }},
		{"ObsvF", func(s *System) (any, error) { return ObsvF(s.A, s.B, s.C) }},
		{"Lqr", func(s *System) (any, error) {
			n, m, _ := s.Dims()
			return Lqr(s.A, s.B, eyeOrEmptyDense(n), eyeOrEmptyDense(m), nil)
		}},
		{"DiskMarginSkew", func(s *System) (any, error) { return DiskMarginSkew(s, 0.5) }},
		{"SampledPassive", func(s *System) (any, error) { return SampledPassive(s, nil) }},
		{"MinimalLFT", func(s *System) (any, error) { return s.MinimalLFT() }},
		{"ZeroDelayApprox", func(s *System) (any, error) { return s.ZeroDelayApprox() }},
		{"PullDelaysToLFT", func(s *System) (any, error) { return s.PullDelaysToLFT() }},
		{"TotalDelay", func(s *System) (any, error) { return s.TotalDelay() }},
		{"Pidtune", func(s *System) (any, error) { return Pidtune(s, PidtunePI) }},
		{"LFT", func(s *System) (any, error) { return LFT(s, s.Copy(), 0, 0) }},
		{"ModelArray", func(s *System) (any, error) { return NewModelArray([]int{1}, []*System{s}) }},
		{"Kalman", func(s *System) (any, error) {
			_, m, p := s.Dims()
			return Kalman(s, eyeOrEmptyDense(m), eyeOrEmptyDense(p), nil)
		}},
	}
}

func emptyIOCheckOutput(out any) error {
	var syss []*System
	switch v := out.(type) {
	case *System:
		syss = append(syss, v)
	case *BalrealResult:
		syss = append(syss, v.Sys)
	case *CanonResult:
		syss = append(syss, v.Sys)
	case *SsbalResult:
		syss = append(syss, v.Sys)
	case *StabsepResult:
		syss = append(syss, v.Stable, v.Unstable)
	case *ModsepResult:
		syss = append(syss, v.Slow, v.Fast)
	case *PrescaleResult:
		syss = append(syss, v.Sys)
	case *ReduceResult:
		syss = append(syss, v.Sys)
	case *ModalReductionResult:
		syss = append(syss, v.Sys)
	case *LoopsensResult:
		syss = append(syss, v.So, v.To, v.Si, v.Ti)
	}
	for _, s := range syss {
		if s == nil {
			return errors.New("nil system in successful result")
		}
		if err := s.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// TestEmptyIOModelsNeverPanic covers MATLAB-valid models with no inputs,
// no outputs or no states: every public op must work or return an error.
func TestEmptyIOModelsNeverPanic(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, n := range []int{0, 1, 3} {
			for _, m := range []int{0, 1, 2} {
				for _, p := range []int{0, 1, 3} {
					if (m > 0 && p > 0) || (n == 0 && m+p > 0) {
						continue
					}
					sys := emptyIOFixture(t, n, m, p, dt)
					if err := sys.Validate(); err != nil {
						t.Fatalf("dt=%g n=%d m=%d p=%d: Validate: %v", dt, n, m, p, err)
					}
					if gn, gm, gp := sys.Dims(); gn != n || gm != m || gp != p {
						t.Fatalf("Dims=(%d,%d,%d), want (%d,%d,%d)", gn, gm, gp, n, m, p)
					}
					for _, op := range emptyIOOps() {
						tag := fmt.Sprintf("dt=%g n=%d m=%d p=%d %s", dt, n, m, p, op.name)
						func() {
							defer func() {
								if r := recover(); r != nil {
									t.Errorf("%s: panic: %v", tag, r)
								}
							}()
							out, err := op.fn(sys.Copy())
							if err == nil {
								if verr := emptyIOCheckOutput(out); verr != nil {
									t.Errorf("%s: invalid output: %v", tag, verr)
								}
							}
						}()
					}
				}
			}
		}
	}
}

// TestEmptyIOInternalDelayModelsNeverPanic repeats the empty-I/O sweep on
// internal-delay models with no inputs or no outputs, built by closing an LFT
// around a delayed Delta with nu=0 or ny=0.
func TestEmptyIOInternalDelayModelsNeverPanic(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, part := range [][2]int{{0, 2}, {2, 0}} {
			M, Delta := lftZeroWidthPlants(t, dt, part[0], part[1])
			Delta.InputDelay = []float64{1, 2}
			if dt == 0 {
				Delta.InputDelay = []float64{0.1, 0.2}
			}
			sys, err := LFT(M, Delta, part[0], part[1])
			if err != nil {
				t.Fatalf("dt=%g nu,ny=%v: LFT: %v", dt, part, err)
			}
			if !sys.HasInternalDelay() {
				t.Fatalf("dt=%g nu,ny=%v: want internal delays", dt, part)
			}
			for _, op := range emptyIOOps() {
				tag := fmt.Sprintf("dt=%g nu,ny=%v %s", dt, part, op.name)
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Errorf("%s: panic: %v", tag, r)
						}
					}()
					out, err := op.fn(sys.Copy())
					if err == nil {
						if verr := emptyIOCheckOutput(out); verr != nil {
							t.Errorf("%s: invalid output: %v", tag, verr)
						}
					}
				}()
			}
		}
	}
}

func autonomousFixture(t *testing.T, dt float64) *System {
	t.Helper()
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	if dt > 0 {
		A = mat.NewDense(2, 2, []float64{0.6, 0.3, -0.2, 0.4})
	}
	sys, err := New(A, nil, mat.NewDense(2, 2, []float64{1, 0.5, -0.3, 2}), nil, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestNewAcceptsEmptyInputOutputBlocks(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	C := mat.NewDense(1, 2, []float64{1, 0.5})
	sys, err := New(A, &mat.Dense{}, C, &mat.Dense{}, 0)
	if err != nil {
		t.Fatalf("New with empty B, D: %v", err)
	}
	if n, m, p := sys.Dims(); n != 2 || m != 0 || p != 1 {
		t.Fatalf("Dims = (%d,%d,%d), want (2,0,1)", n, m, p)
	}
	sys, err = New(A, mat.NewDense(2, 1, []float64{1, 1}), &mat.Dense{}, nil, 0.1)
	if err != nil {
		t.Fatalf("New with empty C: %v", err)
	}
	if n, m, p := sys.Dims(); n != 2 || m != 1 || p != 0 {
		t.Fatalf("Dims = (%d,%d,%d), want (2,1,0)", n, m, p)
	}
}

func TestNewFromSlicesRejectsUnrepresentableEmptyGain(t *testing.T) {
	for _, d := range [][2]int{{0, 2}, {3, 0}} {
		if _, err := NewFromSlices(0, d[0], d[1], nil, nil, nil, nil, 0); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("NewFromSlices(0,%d,%d) err = %v, want ErrDimensionMismatch", d[0], d[1], err)
		}
	}
	if _, err := NewFromSlices(0, 0, 0, nil, nil, nil, nil, 0); err != nil {
		t.Errorf("empty 0x0 gain: %v", err)
	}
}

func TestConstructorsRejectNonFiniteSampleTime(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	for _, dt := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -0.1} {
		if _, err := New(A, B, B, B, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("New dt=%v err = %v", dt, err)
		}
		if _, err := NewGain(B, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewGain dt=%v err = %v", dt, err)
		}
		if _, err := NewFromSlices(0, 1, 1, nil, nil, nil, []float64{1}, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewFromSlices gain dt=%v err = %v", dt, err)
		}
		if _, err := NewDescriptor(A, B, B, B, B, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewDescriptor dt=%v err = %v", dt, err)
		}
		if _, err := Drss(2, 1, 1, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("Drss dt=%v err = %v", dt, err)
		}
	}
}

func TestEmptyIOAnalysisMatchesMATLAB(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		auto := autonomousFixture(t, dt)
		noOut, err := New(auto.A, mat.NewDense(2, 1, []float64{1, -1}), nil, nil, dt)
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"m=0": auto, "p=0": noOut} {
			tag := fmt.Sprintf("dt=%g %s", dt, name)
			if _, err := sys.DCGain(); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("%s DCGain err = %v, want ErrDimensionMismatch", tag, err)
			}
			h2, err := H2Norm(sys)
			if err != nil || h2 != 0 {
				t.Errorf("%s H2Norm = %v, %v; want 0", tag, h2, err)
			}
			hinf, _, err := HinfNorm(sys)
			if err != nil || hinf != 0 {
				t.Errorf("%s HinfNorm = %v, %v; want 0", tag, hinf, err)
			}
			hsv, err := HSV(sys)
			if err != nil || len(hsv) != 2 || hsv[0] != 0 || hsv[1] != 0 {
				t.Errorf("%s HSV = %v, %v; want [0 0]", tag, hsv, err)
			}
			if _, err := Balreal(sys); !errors.Is(err, ErrNotMinimal) {
				t.Errorf("%s Balreal err = %v, want ErrNotMinimal", tag, err)
			}
			if _, err := Bandwidth(sys, -3); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("%s Bandwidth err = %v, want ErrDimensionMismatch", tag, err)
			}
			if _, err := Step(sys, 1); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("%s Step err = %v, want ErrDimensionMismatch", tag, err)
			}
		}

		wc, err := Gram(auto, GramControllability)
		if err != nil || mat.Norm(wc, 1) != 0 {
			t.Errorf("dt=%g Gram(c) of no-input model = %v, %v; want zeros", dt, wc, err)
		}
		wo, err := Gram(auto, GramObservability)
		if err != nil {
			t.Fatal(err)
		}
		var res, ata, ctc mat.Dense
		ctc.Mul(auto.C.T(), auto.C)
		if dt == 0 {
			ata.Mul(auto.A.T(), wo)
			res.Mul(wo, auto.A)
			res.Add(&res, &ata)
		} else {
			ata.Mul(auto.A.T(), wo)
			res.Mul(&ata, auto.A)
			res.Sub(&res, wo)
		}
		res.Add(&res, &ctc)
		if r := mat.Norm(&res, math.Inf(1)); r > 1e-12 {
			t.Errorf("dt=%g Gram(o) Lyapunov residual %g", dt, r)
		}

		P, err := Covar(auto, &mat.Dense{})
		if err != nil || !mat.Equal(P, mat.NewDense(2, 2, nil)) {
			t.Errorf("dt=%g Covar(no inputs) = %v, %v; want 2x2 zeros", dt, P, err)
		}
	}
}

func TestInitialAutonomousModel(t *testing.T) {
	x0 := mat.NewVecDense(2, []float64{1, -0.5})
	for _, dt := range []float64{0, 0.1} {
		sys := autonomousFixture(t, dt)
		initial, err := Initial(sys, x0, 2)
		if err != nil {
			t.Fatalf("dt=%g Initial: %v", dt, err)
		}
		lsim, err := Lsim(sys, nil, []float64{0, 0.1, 0.2, 0.3, 0.4}, x0)
		if err != nil {
			t.Fatalf("dt=%g Lsim: %v", dt, err)
		}
		for _, resp := range []*TimeResponse{initial, lsim} {
			for k, tk := range resp.T {
				var Phi mat.Dense
				if dt == 0 {
					var At mat.Dense
					At.Scale(tk, sys.A)
					Phi.Exp(&At)
				} else {
					Phi.Pow(sys.A, int(math.Round(tk/dt)))
				}
				var x, y mat.VecDense
				x.MulVec(&Phi, x0)
				y.MulVec(sys.C, &x)
				for i := range 2 {
					if d := math.Abs(resp.Y.At(i, k) - y.AtVec(i)); d > 1e-9 {
						t.Fatalf("dt=%g t=%g y%d = %g, want %g", dt, tk, i, resp.Y.At(i, k), y.AtVec(i))
					}
				}
			}
		}
	}
}

func TestEmptyIOStateTransforms(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		auto := autonomousFixture(t, dt)
		T := mat.NewDense(2, 2, []float64{2, 1, 0.5, 3})
		got, err := SS2SS(auto, T)
		if err != nil {
			t.Fatal(err)
		}
		var Ti, wantA, wantC, tmp mat.Dense
		if err := Ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		tmp.Mul(auto.A, &Ti)
		wantA.Mul(T, &tmp)
		wantC.Mul(auto.C, &Ti)
		if !mat.EqualApprox(got.A, &wantA, 1e-12) || !mat.EqualApprox(got.C, &wantC, 1e-12) {
			t.Errorf("dt=%g SS2SS A=%v C=%v, want %v %v", dt, mat.Formatted(got.A), mat.Formatted(got.C), mat.Formatted(&wantA), mat.Formatted(&wantC))
		}
		if n, m, p := got.Dims(); n != 2 || m != 0 || p != 2 {
			t.Errorf("dt=%g SS2SS Dims = (%d,%d,%d)", dt, n, m, p)
		}

		red, err := Modred(auto, []int{1}, MatchDC)
		if err != nil {
			t.Fatal(err)
		}
		a := auto.A
		shift := 0.0
		if dt > 0 {
			shift = 1
		}
		a22 := a.At(1, 1) - shift
		wantAr := a.At(0, 0) - a.At(0, 1)*a.At(1, 0)/a22
		if math.Abs(red.A.At(0, 0)-wantAr) > 1e-12 {
			t.Errorf("dt=%g Modred Ar = %g, want %g", dt, red.A.At(0, 0), wantAr)
		}
		for i := range 2 {
			wantCr := auto.C.At(i, 0) - auto.C.At(i, 1)*a.At(1, 0)/a22
			if math.Abs(red.C.At(i, 0)-wantCr) > 1e-12 {
				t.Errorf("dt=%g Modred Cr[%d] = %g, want %g", dt, i, red.C.At(i, 0), wantCr)
			}
		}
		if n, m, p := red.Dims(); n != 1 || m != 0 || p != 2 {
			t.Errorf("dt=%g Modred Dims = (%d,%d,%d)", dt, n, m, p)
		}

		aug, err := Augstate(auto)
		if err != nil {
			t.Fatal(err)
		}
		if n, m, p := aug.Dims(); n != 2 || m != 0 || p != 4 {
			t.Errorf("dt=%g Augstate Dims = (%d,%d,%d)", dt, n, m, p)
		}
	}
}

func TestEmptyIOInterconnections(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		plant := autonomousFixture(t, dt)
		ctrl, err := New(mat.NewDense(1, 1, []float64{-0.25}), mat.NewDense(1, 2, []float64{1, -2}), nil, nil, dt)
		if err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(plant, ctrl, -1)
		if err != nil {
			t.Fatalf("dt=%g Feedback: %v", dt, err)
		}
		wantA := mat.NewDense(3, 3, nil)
		setBlock(wantA, 0, 0, plant.A)
		setBlock(wantA, 2, 0, mulDense(ctrl.B, plant.C))
		setBlock(wantA, 2, 2, ctrl.A)
		wantC := mat.NewDense(2, 3, nil)
		setBlock(wantC, 0, 0, plant.C)
		if !mat.EqualApprox(cl.A, wantA, 1e-12) || !mat.EqualApprox(cl.C, wantC, 1e-12) {
			t.Errorf("dt=%g Feedback A=%v C=%v", dt, mat.Formatted(cl.A), mat.Formatted(cl.C))
		}
		if n, m, p := cl.Dims(); n != 3 || m != 0 || p != 2 {
			t.Errorf("dt=%g Feedback Dims = (%d,%d,%d)", dt, n, m, p)
		}

		ser, err := Series(plant, ctrl)
		if err != nil {
			t.Fatalf("dt=%g Series: %v", dt, err)
		}
		if n, m, p := ser.Dims(); n != 3 || m != 0 || p != 0 {
			t.Errorf("dt=%g Series Dims = (%d,%d,%d)", dt, n, m, p)
		}
		if !mat.EqualApprox(ser.A, wantA, 1e-12) {
			t.Errorf("dt=%g Series A=%v", dt, mat.Formatted(ser.A))
		}

		par, err := Parallel(plant, plant.Copy())
		if err != nil {
			t.Fatalf("dt=%g Parallel: %v", dt, err)
		}
		var sumC mat.Dense
		sumC.Augment(plant.C, plant.C)
		if !mat.EqualApprox(par.C, &sumC, 1e-12) {
			t.Errorf("dt=%g Parallel C=%v", dt, mat.Formatted(par.C))
		}
	}
}

// TestEmptyIOStatelessResults covers ops that remove every state from a model
// with no inputs or no outputs. MATLAB returns a p×0 or 0×m static gain, which
// cannot be stored, so the op must fail with ErrDimensionMismatch instead of
// returning a 0×0 gain. Models with neither inputs nor outputs still reduce.
func TestEmptyIOStatelessResults(t *testing.T) {
	sysOut := func(s *System, err error) ([]*System, error) { return []*System{s}, err }
	reduceOut := func(r *ReduceResult, err error) ([]*System, error) {
		if err != nil {
			return nil, err
		}
		return []*System{r.Sys}, nil
	}
	ops := []struct {
		name string
		fn   func(s *System) ([]*System, error)
	}{
		{"ModredTruncate", func(s *System) ([]*System, error) { return sysOut(Modred(s, []int{0, 1, 2}, Truncate)) }},
		{"ModredMatchDC", func(s *System) ([]*System, error) {
			return sysOut(Modred(s, []int{2, 0, 1}, MatchDC))
		}},
		{"MinimalRealization", func(s *System) ([]*System, error) { return reduceOut(s.MinimalRealization()) }},
		{"ReduceUncontrollable", func(s *System) ([]*System, error) {
			_, m, _ := s.Dims()
			mode := ReduceUncontrollable
			if m > 0 {
				mode = ReduceUnobservable
			}
			return reduceOut(s.Reduce(&ReduceOpts{Mode: mode}))
		}},
		{"Sminreal", func(s *System) ([]*System, error) { return sysOut(Sminreal(s)) }},
		{"Stabsep", func(s *System) ([]*System, error) {
			r, err := Stabsep(s)
			if err != nil {
				return nil, err
			}
			return []*System{r.Stable, r.Unstable}, nil
		}},
		{"ModsepAllFast", func(s *System) ([]*System, error) {
			r, err := Modsep(s, 1e-6)
			if err != nil {
				return nil, err
			}
			return []*System{r.Slow, r.Fast}, nil
		}},
		{"ModsepAllSlow", func(s *System) ([]*System, error) {
			r, err := Modsep(s, 1e6)
			if err != nil {
				return nil, err
			}
			return []*System{r.Slow, r.Fast}, nil
		}},
	}
	for _, dt := range []float64{0, 0.1} {
		for _, mp := range [][2]int{{0, 2}, {2, 0}, {0, 0}} {
			m, p := mp[0], mp[1]
			sys := emptyIOFixture(t, 3, m, p, dt)
			for _, op := range ops {
				tag := fmt.Sprintf("dt=%g m=%d p=%d %s", dt, m, p, op.name)
				out, err := op.fn(sys.Copy())
				if m == 0 && p == 0 {
					if err != nil {
						t.Errorf("%s: %v", tag, err)
					}
				} else if !errors.Is(err, ErrDimensionMismatch) {
					t.Errorf("%s: err = %v, want ErrDimensionMismatch", tag, err)
				}
				if err != nil {
					continue
				}
				for _, o := range out {
					if _, om, op := o.Dims(); om != m || op != p {
						t.Errorf("%s: result is %dx%d, want %dx%d", tag, op, om, p, m)
					}
				}
			}
		}
	}
}

func TestDescriptorWithoutDynamicStatesKeepsIODims(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 5})
	E := mat.NewDense(2, 2, nil)
	for _, dt := range []float64{0, 0.1} {
		for _, mp := range [][2]int{{0, 2}, {2, 0}} {
			m, p := mp[0], mp[1]
			var B, C *mat.Dense
			if m > 0 {
				B = mat.NewDense(2, m, []float64{1, -0.5, 0.3, 2})
			}
			if p > 0 {
				C = mat.NewDense(p, 2, []float64{1, 0.4, -0.7, 1})
			}
			sys, err := NewDescriptor(A, B, C, nil, E, dt)
			if err != nil {
				t.Fatal(err)
			}
			tag := fmt.Sprintf("dt=%g m=%d p=%d", dt, m, p)
			convert := func() (*System, error) { return sys.D2C(D2COptions{Method: C2DMethodTustin}) }
			if dt == 0 {
				convert = func() (*System, error) { return sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin}) }
			}
			if _, err := convert(); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("%s convert: err = %v, want ErrDimensionMismatch", tag, err)
			}
			if _, err := Stabsep(sys); !errors.Is(err, ErrDimensionMismatch) {
				t.Errorf("%s Stabsep: err = %v, want ErrDimensionMismatch", tag, err)
			}
		}
	}
}

func TestSelectStaticGainRejectsOneSidedEmpty(t *testing.T) {
	g, err := NewGain(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, sel := range [][2][]int{{{}, {1}}, {{0, 2}, {}}} {
		if _, err := g.SelectByIndex(sel[0], sel[1]); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("SelectByIndex(%v,%v): err = %v, want ErrDimensionMismatch", sel[0], sel[1], err)
		}
	}
	r, err := g.SelectByIndex([]int{2}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if _, m, p := r.Dims(); m != 1 || p != 1 || r.D.At(0, 0) != 6 {
		t.Errorf("SelectByIndex([2],[1]) = %dx%d D=%v, want 1x1 [6]", p, m, r.D)
	}
	r, err = g.SelectByIndex([]int{}, []int{})
	if err != nil {
		t.Fatal(err)
	}
	if n, m, p := r.Dims(); n+m+p != 0 {
		t.Errorf("SelectByIndex(nil,nil) dims = (%d,%d,%d), want empty", n, m, p)
	}
}

func TestPolesRejectNonFiniteWithoutHanging(t *testing.T) {
	newSys := func(a01, e01 float64, descriptor bool) *System {
		A := mat.NewDense(3, 3, []float64{-1, a01, 0, 0.5, -3, 1, 0, 2, -4})
		B := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, 1})
		C := mat.NewDense(2, 3, []float64{1, 0, 1, 0, 1, 0})
		D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})
		var E *mat.Dense
		if descriptor {
			E = mat.NewDense(3, 3, []float64{1, e01, 0, 0, 2, 0, 0, 0, 1})
		}
		sys, err := NewDescriptor(A, B, C, D, E, 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	cases := []struct {
		name string
		sys  *System
	}{
		{"A NaN", newSys(math.NaN(), 0, false)},
		{"A +Inf", newSys(math.Inf(1), 0, false)},
		{"descriptor A NaN", newSys(math.NaN(), 0.5, true)},
		{"descriptor E Inf", newSys(2, math.Inf(-1), true)},
	}
	for _, tc := range cases {
		for _, dt := range []float64{0, 0.1} {
			t.Run(fmt.Sprintf("%s dt=%g", tc.name, dt), func(t *testing.T) {
				sys := tc.sys.Copy()
				sys.Dt = dt
				var poleErr, stableErr error
				finishesWithin(t, 5*time.Second, func() {
					_, poleErr = sys.Poles()
					_, stableErr = sys.IsStable()
				})
				if !errors.Is(poleErr, ErrInvalidArgument) || !strings.HasPrefix(poleErr.Error(), "Poles: ") {
					t.Fatalf("Poles: %v", poleErr)
				}
				if !errors.Is(stableErr, ErrInvalidArgument) {
					t.Fatalf("IsStable: %v", stableErr)
				}
			})
		}
	}
}

func TestPolesFiniteUnchanged(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	sys, err := New(A, mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	got := map[float64]bool{}
	for _, p := range poles {
		got[math.Round(real(p)*1e9)/1e9] = imag(p) == 0
	}
	if len(poles) != 2 || !got[-1] || !got[-3] {
		t.Fatalf("poles %v, want {-1,-3}", poles)
	}
}
