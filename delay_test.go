package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestNewWithDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{3})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.HasDelay() {
		t.Error("expected HasDelay true")
	}
	if sys.Delay.At(0, 0) != 3 {
		t.Errorf("delay = %v, want 3", sys.Delay.At(0, 0))
	}
}

func TestNewWithDelayCopiesDelayMatrix(t *testing.T) {
	delay := mat.NewDense(1, 1, []float64{3})
	sys, err := NewWithDelay(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		delay,
		1.0,
	)
	if err != nil {
		t.Fatal(err)
	}

	delay.Set(0, 0, 99)
	if got := sys.Delay.At(0, 0); got != 3 {
		t.Fatalf("delay alias detected: got %v, want 3", got)
	}
}

func TestNewWithDelayNil(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := NewWithDelay(A, B, C, D, nil, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if sys.HasDelay() {
		t.Error("nil delay should not HasDelay")
	}
}

func TestSetDelayDimMismatch(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 1.0)
	err := sys.SetDelay(mat.NewDense(2, 1, []float64{1, 2}))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSetDelayNegative(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	err := sys.SetDelay(mat.NewDense(1, 1, []float64{-1}))
	if !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("expected ErrNegativeDelay, got %v", err)
	}
}

func TestSetDelayCopiesInputMatrix(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	delay := mat.NewDense(1, 1, []float64{2})
	if err := sys.SetDelay(delay); err != nil {
		t.Fatal(err)
	}

	delay.Set(0, 0, 99)
	if got := sys.Delay.At(0, 0); got != 2 {
		t.Fatalf("delay alias detected: got %v, want 2", got)
	}
}

func TestSetDelayFractionalDiscrete(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	err := sys.SetDelay(mat.NewDense(1, 1, []float64{2.5}))
	if !errors.Is(err, ErrFractionalDelay) {
		t.Errorf("expected ErrFractionalDelay, got %v", err)
	}
}

func TestSetDelayContinuousAllowsFractional(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	err := sys.SetDelay(mat.NewDense(1, 1, []float64{0.5}))
	if err != nil {
		t.Errorf("continuous should allow fractional delay, got %v", err)
	}
}

func TestHasDelayZeroMatrix(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	_ = sys.SetDelay(mat.NewDense(1, 1, []float64{0}))
	if sys.HasDelay() {
		t.Error("all-zero delay should return HasDelay=false")
	}
}

func TestCopyPreservesDelay(t *testing.T) {
	sys, _ := NewWithDelay(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{3}),
		1.0,
	)
	cp := sys.Copy()
	if cp.Delay == nil {
		t.Fatal("Copy should preserve delay")
	}
	if cp.Delay.At(0, 0) != 3 {
		t.Errorf("copied delay = %v, want 3", cp.Delay.At(0, 0))
	}
	sys.Delay.Set(0, 0, 99)
	if cp.Delay.At(0, 0) == 99 {
		t.Error("Copy delay should be independent")
	}
}

func TestCopyNilDelayStaysNil(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	cp := sys.Copy()
	if cp.Delay != nil {
		t.Error("nil delay should stay nil after Copy")
	}
}

// Non-symmetric A to catch transposition bugs
func TestSimulateSISOWithDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	delay := mat.NewDense(1, 1, []float64{3})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 10
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	noDelaySys, _ := New(A, B, C, D, 1.0)
	noDelayResp, _ := noDelaySys.Simulate(u, nil, nil)

	// First 3 steps: output = autonomous only (x0=0 → all zero)
	for k := range 3 {
		if math.Abs(resp.Y.At(0, k)) > 1e-12 {
			t.Errorf("step %d: expected 0 (delayed), got %v", k, resp.Y.At(0, k))
		}
	}

	// Steps 3..9: should match undelayed steps 0..6
	for k := 3; k < steps; k++ {
		got := resp.Y.At(0, k)
		want := noDelayResp.Y.At(0, k-3)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v (undelayed step %d)", k, got, want, k-3)
		}
	}
}

func TestSimulateMIMOWithDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.2, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, nil)
	delay := mat.NewDense(2, 2, []float64{
		2, 0,
		0, 3,
	})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 8
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
		u.Set(1, k, 1)
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp.Y == nil {
		t.Fatal("Y is nil")
	}
	r, c := resp.Y.Dims()
	if r != 2 || c != steps {
		t.Fatalf("Y dims = %d×%d, want 2×%d", r, c, steps)
	}
}

func TestSimulateDelayMatchesAbsorb(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{4})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, float64(k+1))
	}

	delayResp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if absorbed.HasDelay() {
		t.Fatal("absorbed system should have no delay")
	}

	absorbResp, err := absorbed.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(delayResp.Y, absorbResp.Y, 1e-10) {
		t.Errorf("Simulate with delay != AbsorbDelay+Simulate\ndelay:    %v\nabsorbed: %v",
			mat.Formatted(delayResp.Y), mat.Formatted(absorbResp.Y))
	}
}

func TestSimulateDelayMatchesAbsorbNonSymA(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-0.1, -0.3, 0.5,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 2,
		0.5, 0.5,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 0, 1,
	})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})
	delay := mat.NewDense(2, 2, []float64{3, 5, 3, 5})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 20
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3))
		u.Set(1, k, math.Cos(float64(k)*0.5))
	}

	delayResp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}

	absorbResp, err := absorbed.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(delayResp.Y, absorbResp.Y, 1e-10) {
		t.Error("Simulate with delay != AbsorbDelay+Simulate (non-sym A)")
	}
}

func TestAbsorbDelayContinuousInput(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 1,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0, 0, 1}, []float64{1, 1}, []float64{0, 0}, 0)
	sys.InputDelay = []float64{0.3, 0.5}

	aug, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("result still has delays")
	}
	n, m, p := aug.Dims()
	if m != 2 || p != 1 {
		t.Errorf("IO dims changed: m=%d p=%d", m, p)
	}
	if n <= 2 {
		t.Errorf("expected augmented states from Padé, got n=%d", n)
	}
}

func TestAbsorbDelayContinuousOutput(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 2,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0}, 0)
	sys.OutputDelay = []float64{0.2, 0.4}

	aug, err := sys.AbsorbDelay(AbsorbOutput)
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("result still has delays")
	}
	n, m, p := aug.Dims()
	if m != 1 || p != 2 {
		t.Errorf("IO dims changed: m=%d p=%d", m, p)
	}
	if n <= 2 {
		t.Errorf("expected augmented states from Padé, got n=%d", n)
	}
}

func TestAbsorbDelayContinuousAll(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 2,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0, 0, 1}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0}, 0)
	sys.InputDelay = []float64{0.3, 0}
	sys.OutputDelay = []float64{0, 0.5}

	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("result still has delays")
	}
	n, m, p := aug.Dims()
	if m != 2 || p != 2 {
		t.Errorf("IO dims changed: m=%d p=%d", m, p)
	}
	if n <= 2 {
		t.Errorf("expected augmented states from Padé, got n=%d", n)
	}
}

func TestAbsorbDelayContinuousDims(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.InputDelay = []float64{0.5}

	aug, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := aug.Dims()
	if n != 1+DefaultPadeOrder {
		t.Errorf("expected n=%d (1 + Padé order %d), got n=%d", 1+DefaultPadeOrder, DefaultPadeOrder, n)
	}
}

func TestAbsorbDelayContinuousIODelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.Delay = mat.NewDense(1, 1, []float64{0.5})

	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("result still has delays")
	}
}

func TestAbsorbDelayNonUniform(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.5})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 1, nil)
	delay := mat.NewDense(2, 1, []float64{3, 5})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)
	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("non-uniform IODelay should be fully absorbed via decomposition")
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("non-uniform IODelay simulation mismatch after absorption")
	}
}

func TestAbsorbDelayDimensions(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 2, nil)
	delay := mat.NewDense(1, 2, []float64{3, 5})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)
	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}

	nAug, mAug, pAug := aug.Dims()
	if nAug != 2+3+5 {
		t.Errorf("augmented n = %d, want %d", nAug, 10)
	}
	if mAug != 2 {
		t.Errorf("augmented m = %d, want 2", mAug)
	}
	if pAug != 1 {
		t.Errorf("augmented p = %d, want 1", pAug)
	}
}

func TestAbsorbDelayNoDelay(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := aug.Dims()
	if n != 1 {
		t.Errorf("no delay absorb should keep n=1, got %d", n)
	}
}

func TestAbsorbDelayZeroDelay(t *testing.T) {
	sys, _ := NewWithDelay(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := aug.Dims()
	if n != 1 {
		t.Errorf("zero delay absorb should keep n=1, got %d", n)
	}
}

func TestTFEvalContinuousDelay(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{1}}},
		Den:   [][]float64{{1, 1}},
		Delay: [][]float64{{0.5}},
		Dt:    0,
	}

	s := complex(0, 2*math.Pi)
	result := mustEval(t, tf, s)

	// H(s) = 1/(s+1) * exp(-0.5*s)
	h0 := complex(1, 0) / (s + 1)
	expected := h0 * cmplx.Exp(-s*complex(0.5, 0))

	if cmplx.Abs(result[0][0]-expected) > 1e-12 {
		t.Errorf("Eval with delay: got %v, want %v", result[0][0], expected)
	}
}

func TestTFEvalDiscreteDelay(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{1}}},
		Den:   [][]float64{{1, -0.5}},
		Delay: [][]float64{{3}},
		Dt:    1.0,
	}

	z := complex(0, 0) + cmplx.Exp(complex(0, math.Pi/4))
	result := mustEval(t, tf, z)

	// H(z) = 1/(z-0.5) * z^{-3}
	h0 := complex(1, 0) / (z - 0.5)
	expected := h0 / (z * z * z)

	if cmplx.Abs(result[0][0]-expected) > 1e-12 {
		t.Errorf("Eval discrete delay: got %v, want %v", result[0][0], expected)
	}
}

func TestTFHasDelay(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 1}},
		Dt:  0,
	}
	if tf.HasDelay() {
		t.Error("nil delay should be false")
	}

	tf.Delay = [][]float64{{0}}
	if tf.HasDelay() {
		t.Error("zero delay should be false")
	}

	tf.Delay = [][]float64{{2}}
	if !tf.HasDelay() {
		t.Error("nonzero delay should be true")
	}
}

func TestTransferFunctionPreservesDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.5})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if tfRes.TF.Delay == nil {
		t.Fatal("TF should have delay")
	}
	if tfRes.TF.Delay[0][0] != 0.5 {
		t.Errorf("TF delay = %v, want 0.5", tfRes.TF.Delay[0][0])
	}
}

func TestStateSpacePreservesDelay(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{1, 0}}},
		Den:   [][]float64{{1, 2, 1}},
		Delay: [][]float64{{0.3}},
		Dt:    0,
	}

	ssRes, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if ssRes.Sys.Delay == nil {
		t.Fatal("SS should have delay")
	}
	if ssRes.Sys.Delay.At(0, 0) != 0.3 {
		t.Errorf("SS delay = %v, want 0.3", ssRes.Sys.Delay.At(0, 0))
	}
}

func TestSSTFSSRoundtripPreservesDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = mat.NewDense(1, 1, []float64{1.5})

	tfRes, _ := sys.TransferFunction(nil)
	ssRes, _ := tfRes.TF.StateSpace()

	if ssRes.Sys.Delay == nil {
		t.Fatal("roundtrip should preserve delay")
	}
	if ssRes.Sys.Delay.At(0, 0) != 1.5 {
		t.Errorf("roundtrip delay = %v, want 1.5", ssRes.Sys.Delay.At(0, 0))
	}
}

func TestDiscretizeConvertsDelay(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.3})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay == nil {
		t.Fatal("discretized system should have delay")
	}
	if disc.Delay.At(0, 0) != 3 {
		t.Errorf("discrete delay = %v, want 3 samples", disc.Delay.At(0, 0))
	}
}

func TestDiscretizeDelayFractionalRounds(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.37})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay.At(0, 0) != 4 {
		t.Fatalf("rounded path delay=%v, want4", disc.Delay)
	}
}

func TestUndiscretizeConvertsDelay(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.9})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{5})

	sys, _ := New(A, B, C, D, 0.1)
	sys.Delay = delay

	cont, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if cont.Delay == nil {
		t.Fatal("undiscretized should have delay")
	}
	if math.Abs(cont.Delay.At(0, 0)-0.5) > 1e-15 {
		t.Errorf("continuous delay = %v, want 0.5", cont.Delay.At(0, 0))
	}
}

func TestDiscretizeZOHConvertsDelay(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-2})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.5})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	disc, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay == nil {
		t.Fatal("ZOH discretized should have delay")
	}
	if disc.Delay.At(0, 0) != 5 {
		t.Errorf("ZOH discrete delay = %v, want 5", disc.Delay.At(0, 0))
	}
}

func TestReducePreservesDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.5})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Sys.Delay == nil {
		t.Fatal("reduced system should preserve delay")
	}
	if res.Sys.Delay.At(0, 0) != 0.5 {
		t.Errorf("reduced delay = %v, want 0.5", res.Sys.Delay.At(0, 0))
	}
}

func TestSimulateNoDelayBackwardCompat(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0.0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	u := mat.NewDense(1, 3, []float64{1, 2, 3})
	x0 := mat.NewVecDense(2, []float64{0.5, -0.3})
	r, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	x := []float64{0.5, -0.3}
	uu := []float64{1, 2, 3}
	wantY := make([]float64, 3)
	for k := range 3 {
		wantY[k] = 1*x[0] + 0*x[1] + 0.5*uu[k]
		nx0 := 0.9*x[0] + 0.1*x[1] + 1*uu[k]
		nx1 := 0.0*x[0] + 0.8*x[1] + 0*uu[k]
		x[0], x[1] = nx0, nx1
	}
	wantYMat := mat.NewDense(1, 3, wantY)
	if !matEqual(r.Y, wantYMat, 1e-12) {
		t.Errorf("backward compat Y mismatch\ngot:  %v\nwant: %v", mat.Formatted(r.Y), mat.Formatted(wantYMat))
	}
}

func TestSimulatePureFeedthroughWithDelay(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{2})
	sys := &System{
		A:     &mat.Dense{},
		B:     &mat.Dense{},
		C:     &mat.Dense{},
		D:     D,
		Delay: mat.NewDense(1, 1, []float64{3}),
		Dt:    1.0,
	}

	u := mat.NewDense(1, 6, []float64{1, 1, 1, 1, 1, 1})
	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for k := range 3 {
		if math.Abs(resp.Y.At(0, k)) > 1e-12 {
			t.Errorf("step %d: expected 0, got %v", k, resp.Y.At(0, k))
		}
	}
	for k := 3; k < 6; k++ {
		if math.Abs(resp.Y.At(0, k)-2) > 1e-12 {
			t.Errorf("step %d: expected 2, got %v", k, resp.Y.At(0, k))
		}
	}
}

func TestSimulateWithDelayAndX0(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.9})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{2})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)
	x0 := mat.NewVecDense(1, []float64{5})

	steps := 6
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	resp, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Autonomous part: y_auto[k] = C * A^k * x0 = 0.9^k * 5
	// Forced part (input j=0, delay=2): shift by 2
	// No-delay SIMO response with x0=0, u=1: y_nd[k]
	noDelaySys, _ := New(A, B, C, D, 1.0)
	uForced := mat.NewDense(1, steps, nil)
	for k := range steps {
		uForced.Set(0, k, 1)
	}
	ndResp, _ := noDelaySys.Simulate(uForced, nil, nil)

	for k := range steps {
		autoY := 5 * math.Pow(0.9, float64(k))
		forcedY := 0.0
		if k >= 2 {
			forcedY = ndResp.Y.At(0, k-2)
		}
		want := autoY + forcedY
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v", k, got, want)
		}
	}
}

// MATLAB-inspired: ss(1,1,1,0,'InputDelay',5) with dt=1
// Equivalent to MATLAB: sys = ss(0.9,1,1,0,1); sys.InputDelay = 5;
func TestMATLABSISOInputDelay(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.9})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{5})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 12
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	resp, _ := sys.Simulate(u, nil, nil)

	for k := range 5 {
		if math.Abs(resp.Y.At(0, k)) > 1e-12 {
			t.Errorf("step %d: expected 0 during delay, got %v", k, resp.Y.At(0, k))
		}
	}

	// After delay: step response of G(z) = 1/(z-0.9)
	// y[5]=0 (x=0 at that point), y[6]=1*1=1 (x was 0, now x=1), etc
	noDelay, _ := New(A, B, C, D, 1.0)
	ndResp, _ := noDelay.Simulate(u, nil, nil)
	for k := 5; k < steps; k++ {
		got := resp.Y.At(0, k)
		want := ndResp.Y.At(0, k-5)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v", k, got, want)
		}
	}
}

// MATLAB-inspired: MIMO system with different I/O delays
// sys = ss(A,B,C,D,1); sys.IODelay = [2 4; 1 3];
func TestMATLABMIMOIODelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0.5, 0.5, 1})
	C := mat.NewDense(2, 2, []float64{1, 0.5, 0.5, 1})
	D := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	delay := mat.NewDense(2, 2, []float64{2, 4, 1, 3})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 10
	u := mat.NewDense(2, steps, nil)
	u.Set(0, 0, 1) // impulse on input 0 only
	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Channel (0,0) has delay=2: D feedthrough appears at k=2
	for k := range 2 {
		if math.Abs(resp.Y.At(0, k)) > 1e-12 {
			t.Errorf("y(0,%d): expected 0, got %v", k, resp.Y.At(0, k))
		}
	}
	if math.Abs(resp.Y.At(0, 2)) < 1e-12 {
		t.Errorf("y(0,2): expected nonzero (D feedthrough delayed by 2)")
	}

	// Channel (1,0) has delay=1, D[1,0]=0: first nonzero at k=2
	for k := range 2 {
		if math.Abs(resp.Y.At(1, k)) > 1e-12 {
			t.Errorf("y(1,%d): expected 0, got %v", k, resp.Y.At(1, k))
		}
	}
	if math.Abs(resp.Y.At(1, 2)) < 1e-12 {
		t.Errorf("y(1,2): expected nonzero (state response delayed by 1)")
	}
}

// MATLAB-inspired: tf with delay - H(s) = exp(-2s)/(s+1)
// Verify frequency response at s=jw
func TestMATLABTFWithDelay(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{1}}},
		Den:   [][]float64{{1, 1}},
		Delay: [][]float64{{2.0}},
		Dt:    0,
	}

	freqs := []float64{0.1, 1.0, 10.0}
	for _, w := range freqs {
		s := complex(0, w)
		H := mustEval(t, tf, s)
		hNoDelay := complex(1, 0) / (s + 1)
		expected := hNoDelay * cmplx.Exp(-s*2)

		if cmplx.Abs(H[0][0]-expected) > 1e-12 {
			t.Errorf("w=%v: got %v, want %v", w, H[0][0], expected)
		}

		// Magnitude should match |1/(jw+1)| (delay doesn't change magnitude)
		gotMag := cmplx.Abs(H[0][0])
		wantMag := cmplx.Abs(hNoDelay)
		if math.Abs(gotMag-wantMag) > 1e-12 {
			t.Errorf("w=%v: magnitude %v != %v (delay shouldn't change magnitude)", w, gotMag, wantMag)
		}
	}
}

// MATLAB-inspired: c2d/d2c delay conversion
// sys = ss(-1,1,1,0); sys.InputDelay = 0.3;
// sysd = c2d(sys, 0.1); → InputDelay should be 3 samples
// sysc = d2c(sysd); → InputDelay should be 0.3s
func TestMATLABC2DD2CDelayRoundtrip(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{0.3})

	sys, _ := New(A, B, C, D, 0)
	sys.Delay = delay

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay.At(0, 0) != 3 {
		t.Fatalf("c2d delay = %v, want 3", disc.Delay.At(0, 0))
	}

	cont, err := disc.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cont.Delay.At(0, 0)-0.3) > 1e-14 {
		t.Errorf("d2c delay = %v, want 0.3", cont.Delay.At(0, 0))
	}
}

// MATLAB-inspired: absorbDelay equivalence
// sys = ss(0.5,1,1,0,1); sys.InputDelay = 3;
// sysAug = absorbDelay(sys);
// Both should give identical step responses
func TestMATLABAbsorbDelayStepResponse(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{3})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 20
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	delayResp, _ := sys.Simulate(u, nil, nil)
	aug, _ := sys.AbsorbDelay()
	augResp, _ := aug.Simulate(u, nil, nil)

	nAug, _, _ := aug.Dims()
	if nAug != 4 {
		t.Fatalf("augmented order = %d, want 4 (1 + 3 delay states)", nAug)
	}

	if !matEqual(delayResp.Y, augResp.Y, 1e-12) {
		t.Error("absorbDelay step response mismatch")
		for k := range steps {
			t.Logf("  k=%d: delay=%v aug=%v", k, delayResp.Y.At(0, k), augResp.Y.At(0, k))
		}
	}
}

// MATLAB-inspired: 2-input system with different input delays
// sys = ss(A,B,C,D,1); sys.InputDelay = [2; 5];
func TestMATLABMultiInputDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 2, []float64{0, 0})
	delay := mat.NewDense(1, 2, []float64{2, 5})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 15
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
		u.Set(1, k, 1)
	}

	resp, _ := sys.Simulate(u, nil, nil)

	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	augResp, _ := aug.Simulate(u, nil, nil)

	nAug, _, _ := aug.Dims()
	if nAug != 2+2+5 {
		t.Errorf("augmented n = %d, want 9", nAug)
	}

	if !matEqual(resp.Y, augResp.Y, 1e-10) {
		t.Error("multi-input delay: Simulate != AbsorbDelay+Simulate")
	}
}

func TestSimulateSIMOWithDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -0.5, 1.2})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 1, []float64{0.2, 0})
	delay := mat.NewDense(2, 1, []float64{3, 5})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, float64(k+1)*0.5)
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	noDelaySys, _ := New(A, B, C, D, 1.0)
	ndResp, _ := noDelaySys.Simulate(u, nil, nil)

	for i := range 2 {
		d := int(delay.At(i, 0))
		for k := range d {
			if math.Abs(resp.Y.At(i, k)) > 1e-12 {
				t.Errorf("output %d step %d: expected 0 during delay, got %v", i, k, resp.Y.At(i, k))
			}
		}
		for k := d; k < steps; k++ {
			got := resp.Y.At(i, k)
			want := ndResp.Y.At(i, k-d)
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("output %d step %d: got %v, want %v", i, k, got, want)
			}
		}
	}

	t.Run("with_x0", func(t *testing.T) {
		x0 := mat.NewVecDense(2, []float64{1, -0.5})
		resp, err := sys.Simulate(u, x0, nil)
		if err != nil {
			t.Fatal(err)
		}

		autoSys, _ := New(A,
			mat.NewDense(2, 1, nil),
			C,
			mat.NewDense(2, 1, nil),
			1.0)
		autoResp, _ := autoSys.Simulate(u, x0, nil)

		simoSys, _ := New(A, B, C, D, 1.0)
		forcedResp, _ := simoSys.Simulate(u, nil, nil)

		outShare := []int{0, 2}
		for i := range 2 {
			d := int(delay.At(i, 0))
			for k := range steps {
				want := 0.0
				if k >= outShare[i] {
					want = autoResp.Y.At(i, k-outShare[i])
				}
				if k >= d {
					want += forcedResp.Y.At(i, k-d)
				}
				got := resp.Y.At(i, k)
				if math.Abs(got-want) > 1e-10 {
					t.Errorf("x0 output %d step %d: got %v, want %v", i, k, got, want)
				}
			}
		}
	})
}

func TestSimulateMISOWithDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -0.3, 0.8})
	B := mat.NewDense(2, 2, []float64{1, 0.5, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0.7})
	D := mat.NewDense(1, 2, []float64{0.3, 0.1})
	delay := mat.NewDense(1, 2, []float64{2, 4})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	steps := 15
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.4)+1)
		u.Set(1, k, math.Cos(float64(k)*0.3))
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	u0 := mat.NewDense(1, steps, nil)
	u1 := mat.NewDense(1, steps, nil)
	for k := range steps {
		u0.Set(0, k, u.At(0, k))
		u1.Set(0, k, u.At(1, k))
	}

	B0 := mat.NewDense(2, 1, []float64{B.At(0, 0), B.At(1, 0)})
	D0 := mat.NewDense(1, 1, []float64{D.At(0, 0)})
	sys0, _ := New(A, B0, C, D0, 1.0)
	resp0, _ := sys0.Simulate(u0, nil, nil)

	B1 := mat.NewDense(2, 1, []float64{B.At(0, 1), B.At(1, 1)})
	D1 := mat.NewDense(1, 1, []float64{D.At(0, 1)})
	sys1, _ := New(A, B1, C, D1, 1.0)
	resp1, _ := sys1.Simulate(u1, nil, nil)

	for k := range steps {
		want := 0.0
		if k >= 2 {
			want += resp0.Y.At(0, k-2)
		}
		if k >= 4 {
			want += resp1.Y.At(0, k-4)
		}
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v", k, got, want)
		}
	}

	absorbed, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	absResp, _ := absorbed.Simulate(u, nil, nil)
	if !matEqual(resp.Y, absResp.Y, 1e-10) {
		t.Error("Simulate with delay != AbsorbDelay+Simulate")
	}
}

func TestSetInputDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 3, 2,
		[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)

	t.Run("valid", func(t *testing.T) {
		err := sys.SetInputDelay([]float64{1, 2, 3})
		if err != nil {
			t.Fatal(err)
		}
		if sys.InputDelay[0] != 1 || sys.InputDelay[1] != 2 || sys.InputDelay[2] != 3 {
			t.Errorf("got %v", sys.InputDelay)
		}
	})

	t.Run("wrong_length", func(t *testing.T) {
		err := sys.SetInputDelay([]float64{1, 2})
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("negative", func(t *testing.T) {
		err := sys.SetInputDelay([]float64{1, -1, 0})
		if !errors.Is(err, ErrNegativeDelay) {
			t.Errorf("expected ErrNegativeDelay, got %v", err)
		}
	})

	t.Run("fractional_discrete", func(t *testing.T) {
		err := sys.SetInputDelay([]float64{1, 2.5, 0})
		if !errors.Is(err, ErrFractionalDelay) {
			t.Errorf("expected ErrFractionalDelay, got %v", err)
		}
	})

	t.Run("fractional_continuous_ok", func(t *testing.T) {
		cSys, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 0)
		err := cSys.SetInputDelay([]float64{0.5, 1.3, 0})
		if err != nil {
			t.Errorf("continuous should allow fractional, got %v", err)
		}
	})

	t.Run("nil_clears", func(t *testing.T) {
		sys2, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)
		_ = sys2.SetInputDelay([]float64{1, 2, 3})
		err := sys2.SetInputDelay(nil)
		if err != nil {
			t.Fatal(err)
		}
		if sys2.InputDelay != nil {
			t.Error("nil should clear InputDelay")
		}
	})
}

func TestSetOutputDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 3, 2,
		[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)

	t.Run("valid", func(t *testing.T) {
		err := sys.SetOutputDelay([]float64{1, 2})
		if err != nil {
			t.Fatal(err)
		}
		if sys.OutputDelay[0] != 1 || sys.OutputDelay[1] != 2 {
			t.Errorf("got %v", sys.OutputDelay)
		}
	})

	t.Run("wrong_length", func(t *testing.T) {
		err := sys.SetOutputDelay([]float64{1, 2, 3})
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("negative", func(t *testing.T) {
		err := sys.SetOutputDelay([]float64{-1, 0})
		if !errors.Is(err, ErrNegativeDelay) {
			t.Errorf("expected ErrNegativeDelay, got %v", err)
		}
	})

	t.Run("fractional_discrete", func(t *testing.T) {
		err := sys.SetOutputDelay([]float64{1.5, 0})
		if !errors.Is(err, ErrFractionalDelay) {
			t.Errorf("expected ErrFractionalDelay, got %v", err)
		}
	})

	t.Run("fractional_continuous_ok", func(t *testing.T) {
		cSys, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 0)
		err := cSys.SetOutputDelay([]float64{0.5, 1.3})
		if err != nil {
			t.Errorf("continuous should allow fractional, got %v", err)
		}
	})

	t.Run("nil_clears", func(t *testing.T) {
		sys2, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)
		_ = sys2.SetOutputDelay([]float64{1, 2})
		err := sys2.SetOutputDelay(nil)
		if err != nil {
			t.Fatal(err)
		}
		if sys2.OutputDelay != nil {
			t.Error("nil should clear OutputDelay")
		}
	})
}

func TestHasDelay_InputOutputDelay(t *testing.T) {
	makeSys := func() *System {
		s, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)
		return s
	}

	t.Run("only_input_delay", func(t *testing.T) {
		s := makeSys()
		_ = s.SetInputDelay([]float64{1, 0, 0})
		if !s.HasDelay() {
			t.Error("expected HasDelay true with InputDelay")
		}
	})

	t.Run("only_output_delay", func(t *testing.T) {
		s := makeSys()
		_ = s.SetOutputDelay([]float64{0, 3})
		if !s.HasDelay() {
			t.Error("expected HasDelay true with OutputDelay")
		}
	})

	t.Run("all_nil", func(t *testing.T) {
		s := makeSys()
		if s.HasDelay() {
			t.Error("expected HasDelay false when all nil")
		}
	})

	t.Run("all_zeros", func(t *testing.T) {
		s := makeSys()
		_ = s.SetInputDelay([]float64{0, 0, 0})
		_ = s.SetOutputDelay([]float64{0, 0})
		if s.HasDelay() {
			t.Error("expected HasDelay false when all zeros")
		}
	})
}

func TestCopy_WithInputOutputDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 3, 2,
		[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)
	_ = sys.SetInputDelay([]float64{1, 2, 3})
	_ = sys.SetOutputDelay([]float64{4, 5})

	cp := sys.Copy()

	t.Run("preserves_values", func(t *testing.T) {
		if len(cp.InputDelay) != 3 || cp.InputDelay[0] != 1 || cp.InputDelay[1] != 2 || cp.InputDelay[2] != 3 {
			t.Errorf("InputDelay not preserved: %v", cp.InputDelay)
		}
		if len(cp.OutputDelay) != 2 || cp.OutputDelay[0] != 4 || cp.OutputDelay[1] != 5 {
			t.Errorf("OutputDelay not preserved: %v", cp.OutputDelay)
		}
	})

	t.Run("deep_copy", func(t *testing.T) {
		cp.InputDelay[0] = 99
		cp.OutputDelay[0] = 99
		if sys.InputDelay[0] == 99 {
			t.Error("modifying copy InputDelay affected original")
		}
		if sys.OutputDelay[0] == 99 {
			t.Error("modifying copy OutputDelay affected original")
		}
	})
}

func TestTotalDelay(t *testing.T) {
	makeSys := func() *System {
		s, _ := NewFromSlices(2, 3, 2,
			[]float64{0, 1, -2, -3}, []float64{1, 0, 0, 0, 1, 0}, []float64{1, 0, 0, 1}, []float64{0, 0, 0, 0, 0, 0}, 1.0)
		return s
	}

	t.Run("all_nil", func(t *testing.T) {
		s := makeSys()
		td := mustTotalDelay(t, s)
		if r, c := td.Dims(); r != 2 || c != 3 || mat.Max(td) != 0 || mat.Min(td) != 0 {
			t.Errorf("delay-free TotalDelay = %v, want 2×3 zeros", mat.Formatted(td))
		}
		empty, _ := NewFromSlices(1, 0, 1, []float64{-1}, nil, []float64{1}, nil, 0)
		if _, err := empty.TotalDelay(); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("no inputs: err = %v, want ErrDimensionMismatch", err)
		}
		var nilSys *System
		if _, err := nilSys.TotalDelay(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("nil: err = %v", err)
		}
	})

	t.Run("only_input_delay", func(t *testing.T) {
		s := makeSys()
		_ = s.SetInputDelay([]float64{1, 0, 2})
		td := mustTotalDelay(t, s)
		if td == nil {
			t.Fatal("expected non-nil")
		}
		r, c := td.Dims()
		if r != 2 || c != 3 {
			t.Fatalf("dims = %d×%d, want 2×3", r, c)
		}
		for i := range 2 {
			for j := range 3 {
				want := s.InputDelay[j]
				if td.At(i, j) != want {
					t.Errorf("td[%d,%d] = %v, want %v", i, j, td.At(i, j), want)
				}
			}
		}
	})

	t.Run("only_output_delay", func(t *testing.T) {
		s := makeSys()
		_ = s.SetOutputDelay([]float64{0, 3})
		td := mustTotalDelay(t, s)
		if td == nil {
			t.Fatal("expected non-nil")
		}
		for i := range 2 {
			for j := range 3 {
				want := s.OutputDelay[i]
				if td.At(i, j) != want {
					t.Errorf("td[%d,%d] = %v, want %v", i, j, td.At(i, j), want)
				}
			}
		}
	})

	t.Run("only_io_delay", func(t *testing.T) {
		s := makeSys()
		ioDelay := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
		_ = s.SetDelay(ioDelay)
		td := mustTotalDelay(t, s)
		if td == nil {
			t.Fatal("expected non-nil")
		}
		for i := range 2 {
			for j := range 3 {
				if td.At(i, j) != ioDelay.At(i, j) {
					t.Errorf("td[%d,%d] = %v, want %v", i, j, td.At(i, j), ioDelay.At(i, j))
				}
			}
		}
	})

	t.Run("combined_matlab_example", func(t *testing.T) {
		s := makeSys()
		_ = s.SetInputDelay([]float64{1, 0, 2})
		_ = s.SetOutputDelay([]float64{0, 3})
		_ = s.SetDelay(mat.NewDense(2, 3, []float64{0, 1, 0, 2, 0, 1}))

		td := mustTotalDelay(t, s)
		if td == nil {
			t.Fatal("expected non-nil")
		}
		want := mat.NewDense(2, 3, []float64{1, 1, 2, 6, 3, 6})
		for i := range 2 {
			for j := range 3 {
				if td.At(i, j) != want.At(i, j) {
					t.Errorf("td[%d,%d] = %v, want %v", i, j, td.At(i, j), want.At(i, j))
				}
			}
		}
	})
}

func TestDecomposeIODelay(t *testing.T) {
	tests := []struct {
		name       string
		ioDelay    [][]float64
		wantInput  []float64
		wantOutput []float64
		wantResid  [][]float64
	}{
		{
			name:       "fully decomposable 2x2",
			ioDelay:    [][]float64{{1, 3}, {2, 4}},
			wantInput:  []float64{1, 3},
			wantOutput: []float64{0, 1},
			wantResid:  [][]float64{{0, 0}, {0, 0}},
		},
		{
			name:       "already zero",
			ioDelay:    [][]float64{{0, 0}, {0, 0}},
			wantInput:  []float64{0, 0},
			wantOutput: []float64{0, 0},
			wantResid:  [][]float64{{0, 0}, {0, 0}},
		},
		{
			name:       "uniform delay",
			ioDelay:    [][]float64{{5, 5}, {5, 5}},
			wantInput:  []float64{5, 5},
			wantOutput: []float64{0, 0},
			wantResid:  [][]float64{{0, 0}, {0, 0}},
		},
		{
			name:       "non-decomposable 2x2",
			ioDelay:    [][]float64{{1, 2}, {3, 1}},
			wantInput:  []float64{1, 1},
			wantOutput: []float64{0, 0},
			wantResid:  [][]float64{{0, 1}, {2, 0}},
		},
		{
			name:       "single row (1x3)",
			ioDelay:    [][]float64{{2, 5, 3}},
			wantInput:  []float64{2, 5, 3},
			wantOutput: []float64{0},
			wantResid:  [][]float64{{0, 0, 0}},
		},
		{
			name:       "single column (3x1)",
			ioDelay:    [][]float64{{2}, {5}, {3}},
			wantInput:  []float64{2},
			wantOutput: []float64{0, 3, 1},
			wantResid:  [][]float64{{0}, {0}, {0}},
		},
		{
			name:       "3x2 asymmetric",
			ioDelay:    [][]float64{{1, 4}, {3, 2}, {2, 5}},
			wantInput:  []float64{1, 2},
			wantOutput: []float64{0, 0, 1},
			wantResid:  [][]float64{{0, 2}, {2, 0}, {0, 2}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := len(tt.ioDelay)
			m := len(tt.ioDelay[0])
			ioData := make([]float64, 0, p*m)
			for _, row := range tt.ioDelay {
				ioData = append(ioData, row...)
			}
			ioMat := mat.NewDense(p, m, ioData)

			gotInput, gotOutput, gotResid := DecomposeIODelay(ioMat)

			for j := range m {
				if math.Abs(gotInput[j]-tt.wantInput[j]) > 1e-14 {
					t.Errorf("inputDelay[%d] = %g, want %g", j, gotInput[j], tt.wantInput[j])
				}
			}
			for i := range p {
				if math.Abs(gotOutput[i]-tt.wantOutput[i]) > 1e-14 {
					t.Errorf("outputDelay[%d] = %g, want %g", i, gotOutput[i], tt.wantOutput[i])
				}
			}
			for i := range p {
				for j := range m {
					if math.Abs(gotResid.At(i, j)-tt.wantResid[i][j]) > 1e-14 {
						t.Errorf("residual[%d,%d] = %g, want %g", i, j, gotResid.At(i, j), tt.wantResid[i][j])
					}
				}
			}
			for i := range p {
				for j := range m {
					got := gotInput[j] + gotOutput[i] + gotResid.At(i, j)
					want := tt.ioDelay[i][j]
					if math.Abs(got-want) > 1e-14 {
						t.Errorf("reconstruction [%d,%d]: %g != %g", i, j, got, want)
					}
				}
			}
		})
	}
}

func simulateWithOutputDelay(sys *System, u *mat.Dense, x0 *mat.VecDense) *mat.Dense {
	n, _, p := sys.Dims()
	_, steps := u.Dims()

	noDelaySys := &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D, Dt: sys.Dt}
	ndResp, _ := noDelaySys.Simulate(u, x0, nil)

	Y := mat.NewDense(p, steps, nil)
	for i := range p {
		d := 0
		if sys.OutputDelay != nil && i < len(sys.OutputDelay) {
			d = int(sys.OutputDelay[i])
		}
		for k := d; k < steps; k++ {
			Y.Set(i, k, ndResp.Y.At(i, k-d))
		}
	}
	_ = n
	return Y
}

func TestAbsorbOutputDelay_SISO(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.9}, []float64{1}, []float64{1}, []float64{0}, 0.1)
	_ = sys.SetOutputDelay([]float64{3})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}
	nAug, mAug, pAug := aug.Dims()
	if nAug != 4 {
		t.Errorf("n = %d, want 4", nAug)
	}
	if mAug != 1 {
		t.Errorf("m = %d, want 1", mAug)
	}
	if pAug != 1 {
		t.Errorf("p = %d, want 1", pAug)
	}
	if aug.OutputDelay != nil {
		t.Error("absorbed system should have nil OutputDelay")
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	wantY := simulateWithOutputDelay(sys, u, nil)
	augResp, err := aug.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(wantY, augResp.Y, 1e-10) {
		t.Errorf("simulation mismatch\nwant: %v\ngot:  %v",
			mat.Formatted(wantY), mat.Formatted(augResp.Y))
	}
}

func TestAbsorbOutputDelay_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 1, []float64{0.2, 0})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetOutputDelay([]float64{2, 4})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}
	nAug, _, _ := aug.Dims()
	if nAug != 2+2+4 {
		t.Errorf("n = %d, want 8", nAug)
	}

	steps := 20
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3)+1)
	}

	wantY := simulateWithOutputDelay(sys, u, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(wantY, augResp.Y, 1e-10) {
		t.Error("MIMO simulation mismatch")
		for k := range steps {
			t.Logf("  k=%d: want=[%v,%v] got=[%v,%v]", k,
				wantY.At(0, k), wantY.At(1, k),
				augResp.Y.At(0, k), augResp.Y.At(1, k))
		}
	}
}

func TestAbsorbOutputDelay_ZeroDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 2,
		[]float64{0.8, 0.1, 0, 0.9}, []float64{1, 0.5}, []float64{1, 0, 0.3, 1}, []float64{0.2, 0}, 0.1)

	t.Run("nil", func(t *testing.T) {
		aug, err := absorbOutputDelay(sys)
		if err != nil {
			t.Fatal(err)
		}
		n, _, _ := aug.Dims()
		if n != 2 {
			t.Errorf("n = %d, want 2", n)
		}
	})

	t.Run("all_zeros", func(t *testing.T) {
		sys2 := sys.Copy()
		_ = sys2.SetOutputDelay([]float64{0, 0})
		aug, err := absorbOutputDelay(sys2)
		if err != nil {
			t.Fatal(err)
		}
		n, _, _ := aug.Dims()
		if n != 2 {
			t.Errorf("n = %d, want 2", n)
		}
		if aug.OutputDelay != nil {
			t.Error("should clear OutputDelay")
		}
	})
}

func TestAbsorbOutputDelay_ContinuousError(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	_, err := absorbOutputDelay(sys)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("expected ErrWrongDomain, got %v", err)
	}
}

func TestAbsorbOutputDelay_MixedDelays(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 1, []float64{0.2, 0.1})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetOutputDelay([]float64{0, 3})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}
	nAug, _, _ := aug.Dims()
	if nAug != 2+3 {
		t.Errorf("n = %d, want 5", nAug)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, float64(k+1)*0.5)
	}
	wantY := simulateWithOutputDelay(sys, u, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(wantY, augResp.Y, 1e-10) {
		t.Error("mixed delay simulation mismatch")
	}
}

func TestAbsorbOutputDelay_GainSystem(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{2.5})
	sys, _ := NewGain(D, 0.1)
	_ = sys.SetOutputDelay([]float64{2})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}
	nAug, _, _ := aug.Dims()
	if nAug != 2 {
		t.Errorf("n = %d, want 2", nAug)
	}

	steps := 8
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	wantY := simulateWithOutputDelay(sys, u, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(wantY, augResp.Y, 1e-10) {
		t.Errorf("gain system mismatch\nwant: %v\ngot:  %v",
			mat.Formatted(wantY), mat.Formatted(augResp.Y))
	}
}

func TestAbsorbOutputDelay_Poles(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{0.9}, []float64{1}, []float64{1}, []float64{0}, 0.1)
	_ = sys.SetOutputDelay([]float64{3})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := aug.Poles()
	if err != nil {
		t.Fatal(err)
	}

	zeroCount := 0
	for _, p := range poles {
		if cmplx.Abs(p) < 1e-10 {
			zeroCount++
		}
	}
	if zeroCount != 3 {
		t.Errorf("expected 3 poles at z=0, got %d (poles: %v)", zeroCount, poles)
	}
}

func TestAbsorbOutputDelay_NonSymA(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-0.1, -0.3, 0.5,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 2,
		0.5, 0.5,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 0, 1,
	})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetOutputDelay([]float64{2, 5})

	aug, err := absorbOutputDelay(sys)
	if err != nil {
		t.Fatal(err)
	}

	steps := 20
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3))
		u.Set(1, k, math.Cos(float64(k)*0.5))
	}

	wantY := simulateWithOutputDelay(sys, u, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(wantY, augResp.Y, 1e-10) {
		t.Error("non-sym A simulation mismatch")
	}
}

func TestAbsorbScope_InputOnly(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetInputDelay([]float64{2, 0})
	_ = sys.SetOutputDelay([]float64{0, 3})

	aug, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}

	if aug.InputDelay != nil {
		t.Error("InputDelay should be cleared")
	}
	if aug.OutputDelay == nil || aug.OutputDelay[0] != 0 || aug.OutputDelay[1] != 3 {
		t.Errorf("OutputDelay should be preserved: %v", aug.OutputDelay)
	}

	nAug, _, _ := aug.Dims()
	if nAug != 2+2 {
		t.Errorf("n = %d, want 4 (2 original + 2 input shift)", nAug)
	}
}

func TestAbsorbScope_OutputOnly(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetInputDelay([]float64{2, 0})
	_ = sys.SetOutputDelay([]float64{0, 3})

	aug, err := sys.AbsorbDelay(AbsorbOutput)
	if err != nil {
		t.Fatal(err)
	}

	if aug.OutputDelay != nil {
		t.Error("OutputDelay should be cleared")
	}
	if aug.InputDelay == nil || aug.InputDelay[0] != 2 || aug.InputDelay[1] != 0 {
		t.Errorf("InputDelay should be preserved: %v", aug.InputDelay)
	}

	nAug, _, _ := aug.Dims()
	if nAug != 2+3 {
		t.Errorf("n = %d, want 5 (2 original + 3 output shift)", nAug)
	}
}

func TestAbsorbScope_IODecompose(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	sys.Delay = mat.NewDense(2, 2, []float64{1, 3, 2, 4})

	aug, err := sys.AbsorbDelay(AbsorbIO)
	if err != nil {
		t.Fatal(err)
	}

	if aug.Delay != nil {
		hasNonzero := false
		r, c := aug.Delay.Dims()
		for i := range r {
			for j := range c {
				if aug.Delay.At(i, j) != 0 {
					hasNonzero = true
				}
			}
		}
		if hasNonzero {
			t.Error("IODelay should be fully decomposed after AbsorbIO")
		}
	}

	nAug, _, _ := aug.Dims()
	if nAug <= 2 {
		t.Errorf("n = %d, should be > 2 after IO absorption", nAug)
	}
}

func TestAbsorbScope_All_Default(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetInputDelay([]float64{2, 0})
	_ = sys.SetOutputDelay([]float64{0, 3})
	sys.Delay = mat.NewDense(2, 2, []float64{1, 3, 2, 4})

	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}

	if aug.HasDelay() {
		t.Error("AbsorbAll should clear all delays")
	}

	steps := 20
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3)+1)
		u.Set(1, k, math.Cos(float64(k)*0.5))
	}

	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("AbsorbAll simulation mismatch")
	}
}

func TestAbsorbScope_All_Explicit(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0.3, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetInputDelay([]float64{2, 0})
	_ = sys.SetOutputDelay([]float64{0, 3})

	aug, err := sys.AbsorbDelay(AbsorbAll)
	if err != nil {
		t.Fatal(err)
	}

	if aug.HasDelay() {
		t.Error("AbsorbAll should clear all delays")
	}

	nAug, _, _ := aug.Dims()
	if nAug != 2+2+3 {
		t.Errorf("n = %d, want 7", nAug)
	}
}

func TestAbsorbScope_InputOnly_Simulation(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0.3})
	D := mat.NewDense(1, 1, []float64{0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetInputDelay([]float64{3})

	aug, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("AbsorbInput simulation mismatch")
	}
}

func TestAbsorbScope_OutputOnly_Simulation(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0.3})
	D := mat.NewDense(1, 1, []float64{0.2})

	sys, _ := New(A, B, C, D, 0.1)
	_ = sys.SetOutputDelay([]float64{4})

	aug, err := sys.AbsorbDelay(AbsorbOutput)
	if err != nil {
		t.Fatal(err)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("AbsorbOutput simulation mismatch")
	}
}

func TestAbsorbScope_ContinuousViaPade(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.SetInputDelay([]float64{0.5})

	aug, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatalf("AbsorbInput on continuous system: %v", err)
	}
	n, _, _ := aug.Dims()
	if n <= 1 {
		t.Errorf("expected augmented states from Padé, got n=%d", n)
	}
	for _, d := range aug.InputDelay {
		if d != 0 {
			t.Errorf("input delay not absorbed: %v", aug.InputDelay)
			break
		}
	}
}

func TestAbsorbScope_NoDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0.8, 0.1, 0, 0.9}, []float64{1, 0.5}, []float64{1, 0.3}, []float64{0.2}, 0.1)

	for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbAll} {
		aug, err := sys.AbsorbDelay(scope)
		if err != nil {
			t.Fatalf("scope %s: %v", scope, err)
		}
		n, _, _ := aug.Dims()
		if n != 2 {
			t.Errorf("scope %s: n = %d, want 2", scope, n)
		}
	}
}

func TestAbsorbScope_BackwardCompat(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0})
	delay := mat.NewDense(1, 1, []float64{4})

	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	aug, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if aug.HasDelay() {
		t.Error("backward compat: should clear delays")
	}

	nAug, _, _ := aug.Dims()
	if nAug != 2+4 {
		t.Errorf("n = %d, want 6", nAug)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, float64(k+1))
	}

	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("backward compat simulation mismatch")
	}
}

func TestAbsorbScope_IOWithResidual(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})

	sys, _ := New(A, B, C, D, 0.1)
	sys.Delay = mat.NewDense(2, 2, []float64{1, 2, 3, 1})

	aug, err := sys.AbsorbDelay(AbsorbIO)
	if err != nil {
		t.Fatal(err)
	}

	if aug.InputDelay != nil {
		t.Error("InputDelay should be nil after AbsorbIO")
	}
	if aug.OutputDelay != nil {
		t.Error("OutputDelay should be nil after AbsorbIO")
	}

	hasResidual := false
	if aug.Delay != nil {
		r, c := aug.Delay.Dims()
		for i := range r {
			for j := range c {
				if aug.Delay.At(i, j) != 0 {
					hasResidual = true
				}
			}
		}
	}
	if !hasResidual {
		t.Log("non-decomposable IO delay has no residual (may be fully absorbed)")
	}

	steps := 20
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3)+1)
		u.Set(1, k, math.Cos(float64(k)*0.5))
	}

	origResp, _ := sys.Simulate(u, nil, nil)
	augResp, _ := aug.Simulate(u, nil, nil)
	if !matEqual(origResp.Y, augResp.Y, 1e-10) {
		t.Error("AbsorbIO simulation mismatch for non-decomposable delay")
	}
}

func TestGetDelayModel_NoInternalDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0.1)

	H, tau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	if len(tau) != 0 {
		t.Errorf("expected empty tau, got %v", tau)
	}
	_, m, p := H.Dims()
	if m != 1 || p != 1 {
		t.Errorf("H dims = (%d, %d), want (1, 1)", m, p)
	}
	if !matEqual(H.A, sys.A, 1e-15) || !matEqual(H.B, sys.B, 1e-15) ||
		!matEqual(H.C, sys.C, 1e-15) || !matEqual(H.D, sys.D, 1e-15) {
		t.Error("H should equal original system when no internal delays")
	}
}

func TestGetDelayModel_SingleDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, 1, []float64{0.1})
	D21 := mat.NewDense(1, 1, []float64{0.6})
	D22 := mat.NewDense(1, 1, []float64{0})
	err := sys.SetInternalDelay([]float64{1.5}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	H, tau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	if len(tau) != 1 || tau[0] != 1.5 {
		t.Errorf("tau = %v, want [1.5]", tau)
	}

	n, m, p := H.Dims()
	if n != 2 {
		t.Errorf("n = %d, want 2", n)
	}
	if m != 2 {
		t.Errorf("m = %d, want 2 (1+1)", m)
	}
	if p != 2 {
		t.Errorf("p = %d, want 2 (1+1)", p)
	}

	if !matEqual(H.A, sys.A, 1e-15) {
		t.Error("H.A != sys.A")
	}
	if H.B.At(0, 0) != 0 || H.B.At(1, 0) != 1 || H.B.At(0, 1) != 0.5 || H.B.At(1, 1) != 0.3 {
		t.Errorf("H.B = %v", mat.Formatted(H.B))
	}
	if H.C.At(0, 0) != 1 || H.C.At(0, 1) != 0 || H.C.At(1, 0) != 0.2 || H.C.At(1, 1) != 0.4 {
		t.Errorf("H.C = %v", mat.Formatted(H.C))
	}
	if H.D.At(0, 0) != 0 || H.D.At(0, 1) != 0.1 || H.D.At(1, 0) != 0.6 || H.D.At(1, 1) != 0 {
		t.Errorf("H.D = %v", mat.Formatted(H.D))
	}
	if H.Dt != 0 {
		t.Errorf("H.Dt = %v, want 0", H.Dt)
	}
}

func TestGetDelayModel_MultipleDelays(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 2,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0.1, 0, 0, 0.2}, 0)

	N := 3
	B2 := mat.NewDense(2, N, []float64{1, 2, 3, 4, 5, 6})
	C2 := mat.NewDense(N, 2, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	D12 := mat.NewDense(2, N, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	D21 := mat.NewDense(N, 2, []float64{0.7, 0.8, 0.9, 1.0, 1.1, 1.2})
	D22 := mat.NewDense(N, N, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0})
	tau := []float64{0.5, 1.0, 2.0}

	err := sys.SetInternalDelay(tau, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	H, gotTau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	if len(gotTau) != 3 {
		t.Fatalf("tau len = %d, want 3", len(gotTau))
	}
	for i, v := range gotTau {
		if v != tau[i] {
			t.Errorf("tau[%d] = %v, want %v", i, v, tau[i])
		}
	}

	n, m, p := H.Dims()
	if n != 2 || m != 5 || p != 5 {
		t.Errorf("H dims = (%d, %d, %d), want (2, 5, 5)", n, m, p)
	}
}

func TestSetDelayModel_Basic(t *testing.T) {
	H, _ := NewFromSlices(2, 3, 3,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0.5, 0.3, 0, 1, 0.4},
		[]float64{1, 0, 0, 1, 0.2, 0.4},
		[]float64{0.1, 0, 0.1, 0, 0.2, 0, 0.6, 0, 0}, 0.1)

	tau := []float64{2}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Errorf("dims = (%d, %d, %d), want (2, 2, 2)", n, m, p)
	}
	if len(sys.LFT.Tau) != 1 || sys.LFT.Tau[0] != 2 {
		t.Errorf("InternalDelay = %v, want [2]", sys.LFT.Tau)
	}
	if sys.Dt != 0.1 {
		t.Errorf("Dt = %v, want 0.1", sys.Dt)
	}

	r, c := sys.LFT.B2.Dims()
	if r != 2 || c != 1 {
		t.Errorf("B2 dims = %d×%d, want 2×1", r, c)
	}
	r, c = sys.LFT.C2.Dims()
	if r != 1 || c != 2 {
		t.Errorf("C2 dims = %d×%d, want 1×2", r, c)
	}
	r, c = sys.LFT.D12.Dims()
	if r != 2 || c != 1 {
		t.Errorf("D12 dims = %d×%d, want 2×1", r, c)
	}
	r, c = sys.LFT.D21.Dims()
	if r != 1 || c != 2 {
		t.Errorf("D21 dims = %d×%d, want 1×2", r, c)
	}
	r, c = sys.LFT.D22.Dims()
	if r != 1 || c != 1 {
		t.Errorf("D22 dims = %d×%d, want 1×1", r, c)
	}
}

func TestGetSetDelayModel_Roundtrip(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 2,
		[]float64{0, 1, -2, -3},
		[]float64{1, 0.5, 0, 1},
		[]float64{1, 0, 0.3, 1},
		[]float64{0.1, 0, 0, 0.2}, 0.5)

	N := 2
	B2 := mat.NewDense(2, N, []float64{0.5, 0.3, 0.2, 0.7})
	C2 := mat.NewDense(N, 2, []float64{0.1, 0.4, 0.6, 0.2})
	D12 := mat.NewDense(2, N, []float64{0.1, 0.2, 0.3, 0.4})
	D21 := mat.NewDense(N, 2, []float64{0.5, 0.6, 0.7, 0.8})
	D22 := mat.NewDense(N, N, []float64{0, 0.1, 0.2, 0})
	tau := []float64{1.0, 3.0}

	err := sys.SetInternalDelay(tau, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	H, gotTau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	rebuilt, err := SetDelayModel(H, gotTau)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(rebuilt.A, sys.A, 1e-15) {
		t.Error("A mismatch")
	}
	if !matEqual(rebuilt.B, sys.B, 1e-15) {
		t.Error("B mismatch")
	}
	if !matEqual(rebuilt.C, sys.C, 1e-15) {
		t.Error("C mismatch")
	}
	if !matEqual(rebuilt.D, sys.D, 1e-15) {
		t.Error("D mismatch")
	}
	if !matEqual(rebuilt.LFT.B2, sys.LFT.B2, 1e-15) {
		t.Error("B2 mismatch")
	}
	if !matEqual(rebuilt.LFT.C2, sys.LFT.C2, 1e-15) {
		t.Error("C2 mismatch")
	}
	if !matEqual(rebuilt.LFT.D12, sys.LFT.D12, 1e-15) {
		t.Error("D12 mismatch")
	}
	if !matEqual(rebuilt.LFT.D21, sys.LFT.D21, 1e-15) {
		t.Error("D21 mismatch")
	}
	if !matEqual(rebuilt.LFT.D22, sys.LFT.D22, 1e-15) {
		t.Error("D22 mismatch")
	}
	if rebuilt.Dt != sys.Dt {
		t.Errorf("Dt = %v, want %v", rebuilt.Dt, sys.Dt)
	}
	for i, v := range rebuilt.LFT.Tau {
		if v != sys.LFT.Tau[i] {
			t.Errorf("InternalDelay[%d] = %v, want %v", i, v, sys.LFT.Tau[i])
		}
	}
}

func TestSetDelayModel_EmptyTau(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0.1)

	result, err := SetDelayModel(sys, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.LFT != nil {
		t.Error("expected nil InternalDelay for empty tau")
	}
	n, m, p := result.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("dims = (%d, %d, %d), want (2, 1, 1)", n, m, p)
	}
}

func TestSetDelayModel_NegativeDelay(t *testing.T) {
	H, _ := NewFromSlices(2, 3, 3,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0.5, 0.3, 0, 1, 0.4},
		[]float64{1, 0, 0, 1, 0.2, 0.4},
		[]float64{0.1, 0, 0.1, 0, 0.2, 0, 0.6, 0, 0}, 0.1)

	_, err := SetDelayModel(H, []float64{-1})
	if !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("expected ErrNegativeDelay, got %v", err)
	}
}

func TestSetDelayModel_DimensionTooSmall(t *testing.T) {
	H, _ := NewFromSlices(2, 1, 1,
		[]float64{0.8, 0.1, 0, 0.9}, []float64{1, 0}, []float64{1, 0}, []float64{0.1}, 0.1)

	_, err := SetDelayModel(H, []float64{1, 2})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestGetDelayModel_DeepCopy(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, 1, []float64{0.1})
	D21 := mat.NewDense(1, 1, []float64{0.6})
	D22 := mat.NewDense(1, 1, []float64{0})
	_ = sys.SetInternalDelay([]float64{1.5}, B2, C2, D12, D21, D22)

	H, tau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}

	tau[0] = 999
	if sys.LFT.Tau[0] == 999 {
		t.Error("modifying returned tau affected original")
	}

	H.A.Set(0, 0, 999)
	if sys.A.At(0, 0) == 999 {
		t.Error("modifying H.A affected original")
	}
}

func TestSetDelayModel_DeepCopy(t *testing.T) {
	H, _ := NewFromSlices(2, 3, 3,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0.5, 0.3, 0, 1, 0.4},
		[]float64{1, 0, 0, 1, 0.2, 0.4},
		[]float64{0.1, 0, 0.1, 0, 0.2, 0, 0.6, 0, 0}, 0.1)

	tau := []float64{2}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}

	tau[0] = 999
	if sys.LFT.Tau[0] == 999 {
		t.Error("modifying input tau affected result")
	}

	H.A.Set(0, 0, 999)
	if sys.A.At(0, 0) == 999 {
		t.Error("modifying input H.A affected result")
	}
}

func TestAbsorbInternalDelay_DiscreteSingle(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	err := sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("absorbed system should have no internal delays")
	}
	nAug, _, _ := absorbed.Dims()
	if nAug != 2+3 {
		t.Errorf("state dim = %d, want %d", nAug, 5)
	}
}

func TestAbsorbInternalDelay_DiscreteMultiple(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	N := 2
	B2 := mat.NewDense(1, N, []float64{0.3, 0.7})
	C2 := mat.NewDense(N, 1, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, nil)
	D21 := mat.NewDense(N, 1, nil)
	D22 := mat.NewDense(N, N, nil)
	err := sys.SetInternalDelay([]float64{2, 4}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("absorbed system should have no internal delays")
	}
	nAug, _, _ := absorbed.Dims()
	wantN := 1 + 2 + 4
	if nAug != wantN {
		t.Errorf("state dim = %d, want %d", nAug, wantN)
	}
}

func TestAbsorbInternalDelay_NoInternalDelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0.9, 0.1, 0, 0.8},
		[]float64{1, 0}, []float64{1, 0}, []float64{0}, 1.0)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := absorbed.Dims()
	if n != 2 {
		t.Errorf("state dim = %d, want 2", n)
	}
}

func TestAbsorbInternalDelay_DiscreteStepResponse(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0.1})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(1, 1, []float64{0.3})
	C2 := mat.NewDense(1, 1, []float64{0.2})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	nAug, m, p := absorbed.Dims()
	if nAug != 4 {
		t.Fatalf("state dim = %d, want 4", nAug)
	}
	if m != 1 || p != 1 {
		t.Fatalf("I/O dims = %d x %d, want 1x1", p, m)
	}

	aRaw := absorbed.A.RawMatrix()
	if math.Abs(aRaw.Data[0*4+0]-0.5) > 1e-12 {
		t.Errorf("A[0,0] = %v, want 0.5", aRaw.Data[0])
	}
	if math.Abs(aRaw.Data[0*4+3]-0.3) > 1e-12 {
		t.Errorf("A[0,3] = %v, want 0.3 (B2)", aRaw.Data[3])
	}
	if math.Abs(aRaw.Data[1*4+0]-0.2) > 1e-12 {
		t.Errorf("A[1,0] = %v, want 0.2 (C2)", aRaw.Data[1*4])
	}
	if math.Abs(aRaw.Data[2*4+1]-1) > 1e-12 {
		t.Errorf("A[2,1] = %v, want 1 (shift)", aRaw.Data[2*4+1])
	}
	if math.Abs(aRaw.Data[3*4+2]-1) > 1e-12 {
		t.Errorf("A[3,2] = %v, want 1 (shift)", aRaw.Data[3*4+2])
	}
}

func TestAbsorbInternalDelay_DiscreteAllScope(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(1, 1, []float64{0.3})
	C2 := mat.NewDense(1, 1, []float64{0.2})
	D12 := mat.NewDense(1, 1, nil)
	D21 := mat.NewDense(1, 1, nil)
	D22 := mat.NewDense(1, 1, nil)
	_ = sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbAll)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("AbsorbAll should remove internal delays")
	}
	nAug, _, _ := absorbed.Dims()
	if nAug != 3 {
		t.Errorf("state dim = %d, want 3", nAug)
	}
}

func TestAbsorbInternalDelay_ContinuousPade(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	B2 := mat.NewDense(1, 1, []float64{0.5})
	C2 := mat.NewDense(1, 1, []float64{0.3})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	_ = sys.SetInternalDelay([]float64{0.5}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("absorbed system should have no internal delays")
	}
	nAug, m, p := absorbed.Dims()
	if nAug != 6 {
		t.Errorf("state dim = %d, want 6", nAug)
	}
	if m != 1 || p != 1 {
		t.Errorf("I/O dims = %d x %d, want 1x1", p, m)
	}
	if !absorbed.IsContinuous() {
		t.Error("absorbed system should be continuous")
	}
}

func TestAbsorbInternalDelay_Roundtrip(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)

	if !sys.HasInternalDelay() {
		t.Fatal("system should have internal delay")
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("absorbed system should have no internal delays")
	}
	if absorbed.LFT != nil {
		t.Error("LFT should be nil")
	}
}

func TestAbsorbInternalDelay_DiscreteD22NonZero(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(1, 1, []float64{0.3})
	C2 := mat.NewDense(1, 1, []float64{0.2})
	D12 := mat.NewDense(1, 1, []float64{0.1})
	D21 := mat.NewDense(1, 1, []float64{0.05})
	D22 := mat.NewDense(1, 1, []float64{0.4})
	_ = sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	nAug, _, _ := absorbed.Dims()
	if nAug != 3 {
		t.Errorf("state dim = %d, want 3", nAug)
	}
	if absorbed.HasInternalDelay() {
		t.Error("should not have internal delays after absorption")
	}

	cRaw := absorbed.C.RawMatrix()
	if math.Abs(cRaw.Data[0*3+2]-0.1) > 1e-12 {
		t.Errorf("C[0,2] = %v, want 0.1 (D12)", cRaw.Data[2])
	}

	bRaw := absorbed.B.RawMatrix()
	if math.Abs(bRaw.Data[1*1+0]-0.05) > 1e-12 {
		t.Errorf("B[1,0] = %v, want 0.05 (D21)", bRaw.Data[1])
	}

	aRaw := absorbed.A.RawMatrix()
	if math.Abs(aRaw.Data[1*3+2]-0.4) > 1e-12 {
		t.Errorf("A[1,2] = %v, want 0.4 (D22)", aRaw.Data[1*3+2])
	}
}

func TestAbsorbInternalDelay_FrequencyResponse(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0.1})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(1, 1, []float64{0.3})
	C2 := mat.NewDense(1, 1, []float64{0.2})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	// DC gain of absorbed system: C*(I-A)^{-1}*B + D
	n, _, _ := absorbed.Dims()
	ImA := mat.NewDense(n, n, nil)
	for i := range n {
		ImA.Set(i, i, 1)
	}
	ImA.Sub(ImA, absorbed.A)
	var luDC mat.LU
	luDC.Factorize(ImA)
	invImA := mat.NewDense(n, n, nil)
	eyeN := mat.NewDense(n, n, nil)
	for i := range n {
		eyeN.Set(i, i, 1)
	}
	_ = luDC.SolveTo(invImA, false, eyeN)
	dcGain := mat.NewDense(1, 1, nil)
	tmp := mat.NewDense(1, n, nil)
	tmp.Mul(absorbed.C, invImA)
	dcGain.Mul(tmp, absorbed.B)
	dcGain.Add(dcGain, absorbed.D)

	// DC gain of original LFT system at z=1:
	// At DC, z^{-d} = 1, so w = z.
	// With D22=0, D12=0, D21=0:
	// z = C2*x = 0.2*x, w = z
	// (I-A)*x = B*u + B2*w => (1-0.5)*x = u + 0.3*0.2*x
	// (0.5 - 0.06)*x = u => x = u/0.44
	// y = C*x + D*u = u/0.44 + 0.1*u
	expectedDC := 1.0/0.44 + 0.1

	absorbedDC := dcGain.At(0, 0)
	if math.Abs(absorbedDC-expectedDC) > 1e-10 {
		t.Errorf("DC gain = %v, want %v", absorbedDC, expectedDC)
	}
}

func TestAbsorbInternalDelay_ContinuousMultiple(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-2})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	N := 2
	B2 := mat.NewDense(1, N, []float64{0.3, 0.5})
	C2 := mat.NewDense(N, 1, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, nil)
	D21 := mat.NewDense(N, 1, nil)
	D22 := mat.NewDense(N, N, nil)
	_ = sys.SetInternalDelay([]float64{0.3, 0.7}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.HasInternalDelay() {
		t.Error("should have no internal delays")
	}
	nAug, _, _ := absorbed.Dims()
	if nAug != 11 {
		t.Errorf("state dim = %d, want 11", nAug)
	}
}

func TestAbsorbInternalDelay_PreservesNilLFTFields(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	B2 := mat.NewDense(1, 1, []float64{0.3})
	C2 := mat.NewDense(1, 1, []float64{0.2})
	D12 := mat.NewDense(1, 1, nil)
	D21 := mat.NewDense(1, 1, nil)
	D22 := mat.NewDense(1, 1, nil)
	_ = sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22)

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	if absorbed.LFT != nil {
		t.Error("LFT should be nil after absorption")
	}
}

func TestSimulateMIMOWithDelayValues(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-0.2, -0.5, 0.6,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		0.3, 0.5,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0.2,
		0.1, 1, 0,
	})
	D := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.3})
	delay := mat.NewDense(2, 2, []float64{
		2, 4,
		1, 3,
	})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	steps := 15

	for jin := range 2 {
		t.Run(fmt.Sprintf("impulse_input_%d", jin), func(t *testing.T) {
			u := mat.NewDense(2, steps, nil)
			u.Set(jin, 0, 1)

			resp, err := sys.Simulate(u, nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			Bj := mat.NewDense(3, 1, nil)
			for i := range 3 {
				Bj.Set(i, 0, B.At(i, jin))
			}
			Dj := mat.NewDense(2, 1, nil)
			for i := range 2 {
				Dj.Set(i, 0, D.At(i, jin))
			}
			uj := mat.NewDense(1, steps, nil)
			uj.Set(0, 0, 1)
			ndSys, _ := New(A, Bj, C, Dj, 1.0)
			ndResp, _ := ndSys.Simulate(uj, nil, nil)

			for i := range 2 {
				d := int(delay.At(i, jin))
				for k := range d {
					if math.Abs(resp.Y.At(i, k)) > 1e-12 {
						t.Errorf("output %d step %d: expected 0 during delay %d, got %v", i, k, d, resp.Y.At(i, k))
					}
				}
				for k := d; k < steps; k++ {
					got := resp.Y.At(i, k)
					want := ndResp.Y.At(i, k-d)
					if math.Abs(got-want) > 1e-10 {
						t.Errorf("output %d step %d: got %v, want %v (delay=%d)", i, k, got, want, d)
					}
				}
			}
		})
	}
}

func TestPullDelaysToLFT_D22CrossCoupling(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	sys.SetInputDelay([]float64{0.3})
	sys.SetOutputDelay([]float64{0.5})

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(lft.LFT.Tau) != 2 {
		t.Fatalf("want 2 internal delays, got %d", len(lft.LFT.Tau))
	}

	t.Run("D22_has_cross_coupling", func(t *testing.T) {
		got := lft.LFT.D22.At(1, 0)
		if got != 2 {
			t.Errorf("D22[1,0] = %v, want 2 (feedthrough routed through both delays)", got)
		}
	})

	t.Run("D12_no_feedthrough_for_delayed_output", func(t *testing.T) {
		got := lft.LFT.D12.At(0, 0)
		if got != 0 {
			t.Errorf("D12[0,0] = %v, want 0 (output has its own delay)", got)
		}
	})

	t.Run("D21_no_feedthrough_for_delayed_input", func(t *testing.T) {
		got := lft.LFT.D21.At(1, 0)
		if got != 0 {
			t.Errorf("D21[1,0] = %v, want 0 (input has its own delay)", got)
		}
	})

	t.Run("D21_input_channel_passthrough", func(t *testing.T) {
		got := lft.LFT.D21.At(0, 0)
		if got != 1 {
			t.Errorf("D21[0,0] = %v, want 1", got)
		}
	})

	t.Run("D12_output_channel_passthrough", func(t *testing.T) {
		got := lft.LFT.D12.At(0, 1)
		if got != 1 {
			t.Errorf("D12[0,1] = %v, want 1", got)
		}
	})

	t.Run("D_zeroed", func(t *testing.T) {
		if lft.D.At(0, 0) != 0 {
			t.Errorf("D[0,0] = %v, want 0", lft.D.At(0, 0))
		}
	})
}

func TestPullDelaysToLFT_D22FreqResponse(t *testing.T) {
	tauIn, tauOut := 0.3, 0.5
	tauTotal := tauIn + tauOut
	Dval := 2.0

	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{Dval}),
		0,
	)
	sys.SetInputDelay([]float64{tauIn})
	sys.SetOutputDelay([]float64{tauOut})

	lft, _ := sys.PullDelaysToLFT()
	respLFT, err := lft.FreqResponse([]float64{0.1, 0.5, 1.0, 2.0, 5.0, 10.0})
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range []float64{0.1, 0.5, 1.0, 2.0, 5.0, 10.0} {
		s := complex(0, w)
		bare := 1/(s+1) + complex(Dval, 0)
		want := bare * cmplx.Exp(-s*complex(tauTotal, 0))
		got := respLFT.At(k, 0, 0)
		relErr := cmplx.Abs(got-want) / math.Max(cmplx.Abs(want), 1e-15)
		if relErr > 1e-10 {
			t.Errorf("w=%v: got %v, want %v, relErr=%v", w, got, want, relErr)
		}
	}
}

func TestPullDelaysToLFT_D22Simulation(t *testing.T) {
	tauIn, tauOut := 2.0, 3.0
	Dt := 1.0

	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.8}),
		Dt,
	)
	sys.SetInputDelay([]float64{tauIn})
	sys.SetOutputDelay([]float64{tauOut})

	ref, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.8}),
		Dt,
	)
	ref.SetDelay(mat.NewDense(1, 1, []float64{tauIn + tauOut}))

	steps := 20
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	respSys, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	respRef, err := ref.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for k := range steps {
		got := respSys.Y.At(0, k)
		want := respRef.Y.At(0, k)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v", k, got, want)
		}
	}
}

func TestPullDelaysToLFT_D22MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 2}),
		mat.NewDense(2, 1, []float64{1, 3}),
		mat.NewDense(2, 2, []float64{
			0.5, 0.7,
			0.3, 0.9,
		}),
		0,
	)
	sys.SetInputDelay([]float64{0.2, 0})
	sys.SetOutputDelay([]float64{0, 0.4})

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(lft.LFT.Tau) != 2 {
		t.Fatalf("want 2 internal delays, got %d", len(lft.LFT.Tau))
	}

	t.Run("D22_cross_row1_col0", func(t *testing.T) {
		got := lft.LFT.D22.At(1, 0)
		if got != 0.3 {
			t.Errorf("D22[1,0] = %v, want 0.3", got)
		}
	})

	t.Run("D12_row0_has_feedthrough", func(t *testing.T) {
		got := lft.LFT.D12.At(0, 0)
		if got != 0.5 {
			t.Errorf("D12[0,0] = %v, want 0.5 (row 0 has no output delay)", got)
		}
	})

	t.Run("D21_col1_has_feedthrough", func(t *testing.T) {
		got := lft.LFT.D21.At(1, 1)
		if got != 0.9 {
			t.Errorf("D21[1,1] = %v, want 0.9 (col 1 has no input delay)", got)
		}
	})

	t.Run("D21_col0_no_feedthrough_for_output_delay_row", func(t *testing.T) {
		got := lft.LFT.D21.At(1, 0)
		if got != 0 {
			t.Errorf("D21[1,0] = %v, want 0 (routed through D22)", got)
		}
	})
}

func TestPullDelaysToLFT_D22Absorption(t *testing.T) {
	tauIn, tauOut := 2.0, 3.0
	Dt := 1.0

	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.8}),
		Dt,
	)
	sys.SetInputDelay([]float64{tauIn})
	sys.SetOutputDelay([]float64{tauOut})

	absorbed, err := sys.AbsorbDelay(AbsorbAll)
	if err != nil {
		t.Fatal(err)
	}

	ref, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.8}),
		Dt,
	)
	ref.SetDelay(mat.NewDense(1, 1, []float64{tauIn + tauOut}))
	refAbsorbed, err := ref.AbsorbDelay(AbsorbAll)
	if err != nil {
		t.Fatal(err)
	}

	steps := 20
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	respAbs, err := absorbed.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	respRef, err := refAbsorbed.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for k := range steps {
		got := respAbs.Y.At(0, k)
		want := respRef.Y.At(0, k)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("step %d: got %v, want %v", k, got, want)
		}
	}
}

func TestPullDelaysToLFT_D22Roundtrip(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	sys.SetInputDelay([]float64{0.3})
	sys.SetOutputDelay([]float64{0.5})

	H, tau, err := sys.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	rebuilt, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 10.0}
	respOrig, _ := sys.FreqResponse(freqs)
	respRebuilt, _ := rebuilt.FreqResponse(freqs)

	for k, w := range freqs {
		got := respRebuilt.At(k, 0, 0)
		want := respOrig.At(k, 0, 0)
		relErr := cmplx.Abs(got-want) / math.Max(cmplx.Abs(want), 1e-15)
		if relErr > 1e-10 {
			t.Errorf("w=%v: got %v, want %v, relErr=%v", w, got, want, relErr)
		}
	}
}

func TestPullDelaysToLFT_D22ZeroFeedthrough(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.SetInputDelay([]float64{0.3})
	sys.SetOutputDelay([]float64{0.5})

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	if lft.LFT.D22.At(1, 0) != 0 {
		t.Errorf("D22[1,0] = %v, want 0 (D=0 means no cross-coupling)", lft.LFT.D22.At(1, 0))
	}
}

func TestSetDelayModel_RejectsZeroTau(t *testing.T) {
	H, _ := NewFromSlices(2, 3, 3,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0.5, 0.3, 0, 1, 0.4},
		[]float64{1, 0, 0, 1, 0.2, 0.4},
		[]float64{0.1, 0, 0.1, 0, 0.2, 0, 0.6, 0, 0}, 0.1)

	_, err := SetDelayModel(H, []float64{0})
	if !errors.Is(err, ErrZeroInternalDelay) {
		t.Errorf("expected ErrZeroInternalDelay, got %v", err)
	}
}

func TestSetInternalDelay_RejectsZeroTau(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)

	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, 1, []float64{0.1})
	D21 := mat.NewDense(1, 1, []float64{0.6})
	D22 := mat.NewDense(1, 1, []float64{0})

	err := sys.SetInternalDelay([]float64{0}, B2, C2, D12, D21, D22)
	if !errors.Is(err, ErrZeroInternalDelay) {
		t.Errorf("expected ErrZeroInternalDelay, got %v", err)
	}
}

func TestSetDelayModel_ValidatesTau(t *testing.T) {
	discrete, _ := NewFromSlices(2, 3, 3,
		[]float64{0.8, 0.1, 0, 0.9},
		[]float64{1, 0.5, 0.3, 0, 1, 0.4},
		[]float64{1, 0, 0, 1, 0.2, 0.4},
		[]float64{0.1, 0, 0.1, 0, 0.2, 0, 0.6, 0, 0}, 0.1)
	continuous := discrete.Copy()
	continuous.Dt = 0
	cases := []struct {
		name string
		H    *System
		tau  float64
		want error
	}{
		{"discrete fractional", discrete, 1.5, ErrFractionalDelay},
		{"discrete NaN", discrete, math.NaN(), ErrInvalidArgument},
		{"continuous Inf", continuous, math.Inf(1), ErrInvalidArgument},
		{"continuous negative", continuous, -0.2, ErrNegativeDelay},
		{"continuous zero", continuous, 0, ErrZeroInternalDelay},
	}
	for _, c := range cases {
		if _, err := SetDelayModel(c.H, []float64{c.tau}); !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
	}
	sys, err := SetDelayModel(continuous, []float64{1.5})
	if err != nil {
		t.Fatal(err)
	}
	if sys.LFT.Tau[0] != 1.5 {
		t.Errorf("continuous tau = %v, want 1.5", sys.LFT.Tau[0])
	}
	if _, err := SetDelayModel(nil, []float64{1}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil H: err = %v", err)
	}
}

func TestSetInternalDelayEmptyTauClears(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0.5}, 0.1)
	if err := sys.SetInternalDelay([]float64{2},
		mat.NewDense(2, 1, []float64{0.5, 0.3}), mat.NewDense(1, 2, []float64{0.1, 0.2}),
		mat.NewDense(1, 1, []float64{0.4}), mat.NewDense(1, 1, []float64{0.6}),
		mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{}, nil, nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if sys.LFT != nil {
		t.Fatalf("empty tau stored LFT %+v", sys.LFT)
	}
	got, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if v := got.D.At(0, 0); v != 0.5 {
		t.Errorf("D = %v, want 0.5", v)
	}
	if err := sys.SetInternalDelay([]float64{1.5}, nil, nil, nil, nil, nil); !errors.Is(err, ErrFractionalDelay) {
		t.Errorf("fractional discrete tau: err = %v", err)
	}
}

func TestZeroDelayApproxEmptyLFTDoesNotPanic(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	sys.LFT = &LFTDelay{Tau: []float64{}}
	if _, err := sys.ZeroDelayApprox(); err != nil {
		t.Fatal(err)
	}
}

func TestZeroDelayApproxD22Zero(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	N := 1
	B2 := mat.NewDense(2, N, []float64{0.5, 0.3})
	C2 := mat.NewDense(N, 2, []float64{0.1, 0.2})
	D12 := mat.NewDense(1, N, []float64{0.4})
	D21 := mat.NewDense(N, 1, []float64{0.6})
	D22 := mat.NewDense(N, N, []float64{0})
	err := sys.SetInternalDelay([]float64{1.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if approx.LFT != nil {
		t.Error("approx should have no InternalDelay")
	}

	wantA := mat.NewDense(2, 2, nil)
	b2c2 := mat.NewDense(2, 2, nil)
	b2c2.Mul(B2, C2)
	wantA.Add(sys.A, b2c2)
	if !mat.EqualApprox(approx.A, wantA, 1e-14) {
		t.Errorf("A_approx mismatch:\ngot  %v\nwant %v", mat.Formatted(approx.A), mat.Formatted(wantA))
	}
}

func TestZeroDelayApproxNonzeroD22(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	N := 1
	B2 := mat.NewDense(2, N, []float64{0.5, 0.3})
	C2 := mat.NewDense(N, 2, []float64{0.1, 0.2})
	D12 := mat.NewDense(1, N, []float64{0.4})
	D21 := mat.NewDense(N, 1, []float64{0.6})
	D22 := mat.NewDense(N, N, []float64{0.5})
	err := sys.SetInternalDelay([]float64{1.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}

	e := 1.0 / (1.0 - 0.5)
	wantA := mat.NewDense(2, 2, nil)
	ec2 := mat.NewDense(1, 2, nil)
	ec2.Scale(e, C2)
	b2ec2 := mat.NewDense(2, 2, nil)
	b2ec2.Mul(B2, ec2)
	wantA.Add(sys.A, b2ec2)
	if !mat.EqualApprox(approx.A, wantA, 1e-14) {
		t.Errorf("A_approx mismatch:\ngot  %v\nwant %v", mat.Formatted(approx.A), mat.Formatted(wantA))
	}

	wantD := 0.0 + 0.4*e*0.6
	if math.Abs(approx.D.At(0, 0)-wantD) > 1e-14 {
		t.Errorf("D_approx = %v, want %v", approx.D.At(0, 0), wantD)
	}
}

func TestZeroDelayApproxSingular(t *testing.T) {
	sys, _ := NewFromSlices(1, 1, 1,
		[]float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	B2 := mat.NewDense(1, 1, []float64{1})
	C2 := mat.NewDense(1, 1, []float64{1})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{1})
	err := sys.SetInternalDelay([]float64{0.5}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	_, err = sys.ZeroDelayApprox()
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("expected ErrAlgebraicLoop, got %v", err)
	}
	var diagnostic *AlgebraicLoopError
	if !errors.As(err, &diagnostic) {
		t.Fatalf("err = %T, want *AlgebraicLoopError", err)
	}
	if !math.IsInf(diagnostic.Condition, 1) {
		t.Errorf("condition = %g, want +Inf", diagnostic.Condition)
	}
}

func TestZeroDelayApproxNoInternal(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)

	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(approx.A, sys.A, 1e-15) {
		t.Error("no internal delay: A should match original")
	}
	if !mat.EqualApprox(approx.D, sys.D, 1e-15) {
		t.Error("no internal delay: D should match original")
	}
}

func TestZeroDelayApproxMIMO(t *testing.T) {
	sys, _ := NewFromSlices(2, 2, 2,
		[]float64{0, 1, -2, -3},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0}, 0)
	N := 2
	B2 := mat.NewDense(2, N, []float64{1, 0, 0, 1})
	C2 := mat.NewDense(N, 2, []float64{0.5, 0, 0, 0.5})
	D12 := mat.NewDense(2, N, []float64{0, 0, 0, 0})
	D21 := mat.NewDense(N, 2, []float64{0, 0, 0, 0})
	D22 := mat.NewDense(N, N, []float64{0, 0, 0, 0})
	err := sys.SetInternalDelay([]float64{1.0, 2.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := approx.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Errorf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}
	if approx.LFT != nil {
		t.Error("approx should have no InternalDelay")
	}
}

func TestZeroDelayApproxPreservesIODelay(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	N := 1
	B2 := mat.NewDense(2, N, []float64{1, 0})
	C2 := mat.NewDense(N, 2, []float64{0, 1})
	D12 := mat.NewDense(1, N, []float64{0})
	D21 := mat.NewDense(N, 1, []float64{0})
	D22 := mat.NewDense(N, N, []float64{0})
	err := sys.SetInternalDelay([]float64{1.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{0.5}
	sys.OutputDelay = []float64{0.3}

	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if len(approx.InputDelay) != 1 || approx.InputDelay[0] != 0.5 {
		t.Errorf("InputDelay = %v, want [0.5]", approx.InputDelay)
	}
	if len(approx.OutputDelay) != 1 || approx.OutputDelay[0] != 0.3 {
		t.Errorf("OutputDelay = %v, want [0.3]", approx.OutputDelay)
	}
}

func TestMixedDelayTypes(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	err := sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("SetInputDelay_with_InternalDelay", func(t *testing.T) {
		err := sys.SetInputDelay([]float64{2})
		if err != nil {
			t.Fatalf("should allow InputDelay with InternalDelay, got %v", err)
		}
	})

	t.Run("SetOutputDelay_with_InternalDelay", func(t *testing.T) {
		err := sys.SetOutputDelay([]float64{1})
		if err != nil {
			t.Fatalf("should allow OutputDelay with InternalDelay, got %v", err)
		}
	})

	t.Run("SetDelay_with_InternalDelay", func(t *testing.T) {
		err := sys.SetDelay(mat.NewDense(1, 1, []float64{2}))
		if err != nil {
			t.Fatalf("should allow IODelay with InternalDelay, got %v", err)
		}
	})

	t.Run("SetInternalDelay_with_IODelay", func(t *testing.T) {
		sys2, _ := New(A, B, C, D, 1.0)
		_ = sys2.SetInputDelay([]float64{3})
		err := sys2.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22)
		if err != nil {
			t.Fatalf("should allow InternalDelay with IODelay, got %v", err)
		}
	})

	t.Run("HasDelay_with_both", func(t *testing.T) {
		if !sys.HasDelay() {
			t.Error("expected HasDelay true with mixed delays")
		}
		if !sys.HasInternalDelay() {
			t.Error("expected HasInternalDelay true")
		}
	})
}

func TestMixedDelayTotalDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	_ = sys.SetInputDelay([]float64{2})
	_ = sys.SetOutputDelay([]float64{1})

	td := mustTotalDelay(t, sys)
	if td == nil {
		t.Fatal("expected non-nil TotalDelay")
	}
	want := 3.0
	if td.At(0, 0) != want {
		t.Errorf("TotalDelay = %v, want %v (InternalDelay excluded)", td.At(0, 0), want)
	}
}

func TestMixedDelayFreqResponse(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	_ = sys.SetInputDelay([]float64{2})

	omega := []float64{0.1, 0.5, 1.0, 2.0}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}

	sysNoIO := sys.Copy()
	sysNoIO.InputDelay = nil
	lftResp, err := sysNoIO.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*sys.Dt))
		got := resp.At(k, 0, 0)
		lftVal := lftResp.At(k, 0, 0)
		wantVal := lftVal / (z * z)
		if cmplx.Abs(got-wantVal) > 1e-10 {
			t.Errorf("omega=%v: got %v, want %v", w, got, wantVal)
		}
	}
}

func TestMixedDelayEvalFr(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	_ = sys.SetOutputDelay([]float64{1})

	s := cmplx.Exp(complex(0, 0.5))
	got, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}

	sysNoIO := sys.Copy()
	sysNoIO.OutputDelay = nil
	wantNoIO, _ := sysNoIO.EvalFr(s)
	wantVal := wantNoIO[0][0] / s
	if cmplx.Abs(got[0][0]-wantVal) > 1e-10 {
		t.Errorf("EvalFr: got %v, want %v", got[0][0], wantVal)
	}
}

func TestMixedDelaySimulate(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	_ = sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)
	_ = sys.SetInputDelay([]float64{2})

	steps := 20
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	merged, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	mergedResp, err := merged.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(resp.Y, mergedResp.Y, 1e-10) {
		t.Errorf("mixed delay Simulate != PullDelaysToLFT+Simulate\nmixed:  %v\nmerged: %v",
			mat.Formatted(resp.Y), mat.Formatted(mergedResp.Y))
	}
}

func TestMixedDelaySimulate_WithX0(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)

	N := 1
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1})
	C2 := mat.NewDense(N, 2, []float64{0.2, 0.4})
	D12 := mat.NewDense(1, N, []float64{0.1})
	D21 := mat.NewDense(N, 1, []float64{0.2})
	D22 := mat.NewDense(N, N, []float64{0})
	_ = sys.SetInternalDelay([]float64{2}, B2, C2, D12, D21, D22)
	_ = sys.SetOutputDelay([]float64{1})

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3))
	}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})

	resp, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	merged, _ := sys.PullDelaysToLFT()
	mergedResp, _ := merged.Simulate(u, x0, nil)

	if !matEqual(resp.Y, mergedResp.Y, 1e-10) {
		t.Errorf("mixed delay Simulate with x0 mismatch\nmixed:  %v\nmerged: %v",
			mat.Formatted(resp.Y), mat.Formatted(mergedResp.Y))
	}
}

func TestMixedDelaySimulate_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.2, 0, 0.9})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})
	sys, _ := New(A, B, C, D, 1.0)

	N := 2
	B2 := mat.NewDense(2, N, []float64{0.3, 0.1, 0.2, 0.4})
	C2 := mat.NewDense(N, 2, []float64{0.1, 0.2, 0.3, 0.1})
	D12 := mat.NewDense(2, N, []float64{0.1, 0, 0, 0.1})
	D21 := mat.NewDense(N, 2, []float64{0.2, 0, 0, 0.3})
	D22 := mat.NewDense(N, N, nil)
	_ = sys.SetInternalDelay([]float64{2, 3}, B2, C2, D12, D21, D22)
	_ = sys.SetInputDelay([]float64{1, 2})
	_ = sys.SetOutputDelay([]float64{1, 0})

	steps := 20
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.3))
		u.Set(1, k, math.Cos(float64(k)*0.5))
	}

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	merged, _ := sys.PullDelaysToLFT()
	mergedResp, _ := merged.Simulate(u, nil, nil)

	if !matEqual(resp.Y, mergedResp.Y, 1e-10) {
		t.Errorf("MIMO mixed delay Simulate mismatch")
	}
}

func TestSimulateWithDelay_MIMOManualReference(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0.82, 0.14, 0.00, 0.05,
		0.07, 0.76, 0.11, 0.00,
		0.03, 0.09, 0.71, 0.12,
		0.00, 0.04, 0.08, 0.68,
	})
	B := mat.NewDense(4, 3, []float64{
		1.0, 0.2, 0.0,
		0.1, 0.9, 0.3,
		0.0, 0.4, 0.8,
		0.2, 0.0, 0.7,
	})
	C := mat.NewDense(3, 4, []float64{
		1.0, 0.1, 0.0, 0.2,
		0.0, 0.8, 0.3, 0.0,
		0.2, 0.0, 0.7, 0.4,
	})
	D := mat.NewDense(3, 3, []float64{
		0.10, 0.05, 0.00,
		0.00, 0.15, 0.07,
		0.03, 0.00, 0.12,
	})
	delay := mat.NewDense(3, 3, []float64{
		0, 2, 1,
		3, 1, 4,
		2, 0, 3,
	})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	steps := 18
	u := mat.NewDense(3, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(0.2*float64(k))+0.1)
		u.Set(1, k, math.Cos(0.15*float64(k))-0.2)
		u.Set(2, k, 0.05*float64(k)-0.3)
	}
	x0 := mat.NewVecDense(4, []float64{0.5, -0.3, 0.2, -0.1})
	if _, err := sys.Simulate(u, x0, nil); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("nonzero x0 with nondecomposable Delay: err = %v, want ErrDelayUnsupported", err)
	}
	x0.Zero()

	got, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	autoSys, err := New(A, mat.NewDense(4, 3, nil), C, mat.NewDense(3, 3, nil), 1.0)
	if err != nil {
		t.Fatal(err)
	}
	autoResp, err := autoSys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	wantY := mat.DenseCopyOf(autoResp.Y)

	for j := range 3 {
		Bj := mat.NewDense(4, 1, nil)
		Dj := mat.NewDense(3, 1, nil)
		uj := mat.NewDense(1, steps, nil)
		for i := range 4 {
			Bj.Set(i, 0, B.At(i, j))
		}
		for i := range 3 {
			Dj.Set(i, 0, D.At(i, j))
		}
		for k := range steps {
			uj.Set(0, k, u.At(j, k))
		}

		simoSys, err := New(A, Bj, C, Dj, 1.0)
		if err != nil {
			t.Fatal(err)
		}
		simoResp, err := simoSys.Simulate(uj, nil, nil)
		if err != nil {
			t.Fatal(err)
		}

		for i := range 3 {
			d := int(delay.At(i, j))
			for k := d; k < steps; k++ {
				wantY.Set(i, k, wantY.At(i, k)+simoResp.Y.At(i, k-d))
			}
		}
	}

	if !matEqual(got.Y, wantY, 1e-10) {
		t.Errorf("delayed MIMO response mismatch\n got: %v\nwant: %v",
			mat.Formatted(got.Y), mat.Formatted(wantY))
	}
	if got.XFinal != nil {
		t.Errorf("XFinal = %v, want nil when not requested", mat.Formatted(got.XFinal))
	}
	if _, err := sys.Simulate(u, x0, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("FinalState err = %v, want ErrDelayUnsupported: Delay has no input+output split", err)
	}
}

func TestMinimalLFTNoInternal(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0)
	result, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}
	if result.HasInternalDelay() {
		t.Error("expected no internal delay")
	}
	n, m, p := result.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("dims = %d,%d,%d want 2,1,1", n, m, p)
	}
}

func TestMinimalLFTZeroGainRemoval(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 1.0)

	B2 := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 0, 1,
	})
	C2 := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		0, 1,
	})
	D12 := mat.NewDense(1, 3, []float64{0.5, 0, 0.3})
	D21 := mat.NewDense(3, 1, []float64{0.1, 0, 0.2})
	D22 := mat.NewDense(3, 3, nil)

	err := sys.SetInternalDelay([]float64{2, 3, 4}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	result, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(result.LFT.Tau) != 2 {
		t.Fatalf("got %d delays, want 2", len(result.LFT.Tau))
	}
	if result.LFT.Tau[0] != 2 || result.LFT.Tau[1] != 4 {
		t.Errorf("delays = %v, want [2,4]", result.LFT.Tau)
	}

	r, c := result.LFT.B2.Dims()
	if r != 2 || c != 2 {
		t.Errorf("B2 dims = %d,%d want 2,2", r, c)
	}
	r, c = result.LFT.C2.Dims()
	if r != 2 || c != 2 {
		t.Errorf("C2 dims = %d,%d want 2,2", r, c)
	}
	r, c = result.LFT.D22.Dims()
	if r != 2 || c != 2 {
		t.Errorf("D22 dims = %d,%d want 2,2", r, c)
	}
}

func TestMinimalLFTAllZeroGain(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 1.0)

	N := 3
	B2 := mat.NewDense(2, N, nil)
	C2 := mat.NewDense(N, 2, nil)
	D12 := mat.NewDense(1, N, nil)
	D21 := mat.NewDense(N, 1, nil)
	D22 := mat.NewDense(N, N, nil)

	err := sys.SetInternalDelay([]float64{1, 2, 3}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	result, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if result.HasInternalDelay() {
		t.Error("expected all internal delays removed")
	}
	if result.LFT != nil {
		t.Error("expected nil LFT")
	}
}

func TestMinimalLFTNoReduction(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{0, 1, -2, -3}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 1.0)

	B2 := mat.NewDense(2, 2, []float64{1, 0.5, 0, 1})
	C2 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D12 := mat.NewDense(1, 2, []float64{0.1, 0.2})
	D21 := mat.NewDense(2, 1, []float64{0.3, 0.4})
	D22 := mat.NewDense(2, 2, nil)

	err := sys.SetInternalDelay([]float64{1, 2}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	result, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(result.LFT.Tau) != 2 {
		t.Fatalf("got %d delays, want 2", len(result.LFT.Tau))
	}
	if !matEqual(result.LFT.B2, sys.LFT.B2, 1e-15) {
		t.Error("B2 changed unexpectedly")
	}
	if !matEqual(result.LFT.C2, sys.LFT.C2, 1e-15) {
		t.Error("C2 changed unexpectedly")
	}
}

func TestMinimalLFTFreqResponse(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 1}, []float64{0}, 0)

	B2 := mat.NewDense(2, 3, []float64{
		1, 0, 0.5,
		0, 0, 0.3,
	})
	C2 := mat.NewDense(3, 2, []float64{
		0.5, 0.1,
		0, 0,
		0.2, 0.4,
	})
	D12 := mat.NewDense(1, 3, []float64{0.1, 0, 0.2})
	D21 := mat.NewDense(3, 1, []float64{0.3, 0, 0.1})
	D22 := mat.NewDense(3, 3, nil)

	err := sys.SetInternalDelay([]float64{0.1, 0.2, 0.3}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.1, 1.0, 10.0, 100.0}
	freqOrig, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(reduced.LFT.Tau) != 2 {
		t.Fatalf("expected 2 delays after reduction, got %d", len(reduced.LFT.Tau))
	}

	freqReduced, err := reduced.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}

	for fi := range omega {
		orig := freqOrig.At(fi, 0, 0)
		red := freqReduced.At(fi, 0, 0)
		if cmplx.Abs(orig-red) > 1e-10 {
			t.Errorf("freq %.1f: orig=%v reduced=%v", omega[fi], orig, red)
		}
	}
}

func TestMinimalLFTMergeDuplicateDelays(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 1}, []float64{0}, 0)

	B2 := mat.NewDense(2, 3, []float64{
		1, 2, 0.5,
		0, 0, 0.3,
	})
	C2 := mat.NewDense(3, 2, []float64{
		0.5, 0.1,
		1.0, 0.2,
		0.2, 0.4,
	})
	D12 := mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	D21 := mat.NewDense(3, 1, []float64{0.3, 0.6, 0.1})
	D22 := mat.NewDense(3, 3, nil)

	err := sys.SetInternalDelay([]float64{1.0, 1.0, 2.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.01, 0.1, 1.0, 10.0, 100.0}
	freqOrig, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(reduced.LFT.Tau) != 2 {
		t.Fatalf("expected 2 delays, got %d: %v", len(reduced.LFT.Tau), reduced.LFT.Tau)
	}

	freqReduced, err := reduced.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}

	for fi := range omega {
		orig := freqOrig.At(fi, 0, 0)
		red := freqReduced.At(fi, 0, 0)
		if cmplx.Abs(orig-red) > 1e-10 {
			t.Errorf("freq %.2f: orig=%v reduced=%v", omega[fi], orig, red)
		}
	}
}

func TestMinimalLFTMergeNonProportionalKept(t *testing.T) {
	sys, _ := NewFromSlices(2, 1, 1,
		[]float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 1}, []float64{0}, 0)

	B2 := mat.NewDense(2, 2, []float64{
		1, 0.5,
		0, 0.3,
	})
	C2 := mat.NewDense(2, 2, []float64{
		0.5, 0.1,
		0.3, 0.7,
	})
	D12 := mat.NewDense(1, 2, []float64{0.1, 0.2})
	D21 := mat.NewDense(2, 1, []float64{0.3, 0.4})
	D22 := mat.NewDense(2, 2, nil)

	err := sys.SetInternalDelay([]float64{1.0, 1.0}, B2, C2, D12, D21, D22)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := sys.MinimalLFT()
	if err != nil {
		t.Fatal(err)
	}

	if len(reduced.LFT.Tau) != 2 {
		t.Fatalf("expected 2 delays (non-proportional), got %d", len(reduced.LFT.Tau))
	}
}

func TestSimulateDelay_AllDelaysExceedSteps(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0.5})
	D := mat.NewDense(1, 1, []float64{2})
	delay := mat.NewDense(1, 1, []float64{10})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	steps := 5
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	x0 := mat.NewVecDense(2, []float64{3, 1})

	resp, err := sys.Simulate(u, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	autoSys, err := New(A, mat.NewDense(2, 1, nil), C, mat.NewDense(1, 1, nil), 1.0)
	if err != nil {
		t.Fatal(err)
	}
	zeroU := mat.NewDense(1, steps, nil)
	autoResp, err := autoSys.Simulate(zeroU, x0, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !matEqual(resp.Y, autoResp.Y, 1e-14) {
		t.Errorf("delay > steps: forced response should be invisible\n got: %v\nwant: %v",
			mat.Formatted(resp.Y), mat.Formatted(autoResp.Y))
	}
}

func TestSimulateDelay_BoundaryDelay(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{3})

	steps := 5
	delay := mat.NewDense(1, 1, []float64{float64(steps - 1)})

	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	u := mat.NewDense(1, steps, nil)
	u.Set(0, 0, 1)

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for k := 0; k < steps-1; k++ {
		if resp.Y.At(0, k) != 0 {
			t.Errorf("Y[0,%d] = %f, want 0", k, resp.Y.At(0, k))
		}
	}
	want := D.At(0, 0) * u.At(0, 0)
	got := resp.Y.At(0, steps-1)
	if math.Abs(got-want) > 1e-14 {
		t.Errorf("Y[0,%d] = %f, want %f", steps-1, got, want)
	}
}

func TestZeroDelayApproxKeepsDescriptorE(t *testing.T) {
	sys, closed := internalDelayZerosFixture(t, 0, 2, mat.NewDense(1, 1, []float64{0.25}))
	E := mat.NewDense(3, 3, []float64{1, 2, 0, 0, 1, 1, 1, 3, 1})
	sys.E = mat.DenseCopyOf(E)
	approx, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if approx.E == nil || !mat.Equal(approx.E, E) || approx.E == sys.E {
		t.Fatalf("E not preserved as a copy: %v", approx.E)
	}
	if !mat.EqualApprox(approx.A, closed.A, 1e-14) {
		t.Errorf("A mismatch")
	}
}

func TestSetDelayModelZeroState(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		H, err := NewGain(mat.NewDense(2, 2, []float64{1, 0.5, 0.2, 0.1}), dt)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, []float64{3})
		if err != nil {
			t.Fatal(err)
		}
		n, m, p := sys.Dims()
		if n != 0 || m != 1 || p != 1 || len(sys.LFT.Tau) != 1 {
			t.Fatalf("dims n=%d m=%d p=%d tau=%v", n, m, p, sys.LFT.Tau)
		}
		assertResponseOracle(t, "zero-state SetDelayModel", sys, func(s complex128) [][]complex128 {
			d := exactDelayFactor(dt)(delayInternal, 3, s)
			return [][]complex128{{1 + 0.5*d*0.2/(1-0.1*d)}}
		})
	}
}

func TestZeroDelayApproxKeepsDescriptorAndZeroState(t *testing.T) {
	noInternal := func(dt float64) delayFactorFunc {
		exact := exactDelayFactor(dt)
		return func(c delayClass, tau float64, s complex128) complex128 {
			if c == delayInternal {
				return 1
			}
			return exact(c, tau, s)
		}
	}
	for _, dt := range []float64{0, 0.1} {
		A := mat.NewDense(2, 2, []float64{-1, 2, 0.5, -3})
		if dt > 0 {
			A = mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, 0.7})
		}
		desc, err := NewDescriptor(A, mat.NewDense(2, 2, []float64{1, 0.3, 0, 1}),
			mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}), mat.NewDense(2, 2, []float64{0.2, 0.1, 0, 0.3}),
			mat.NewDense(2, 2, []float64{2, 0.5, 0.1, 1.5}), dt)
		if err != nil {
			t.Fatal(err)
		}
		desc.InputDelay = []float64{0, 2 * (1 + 9*dt*10)}
		if err := desc.SetInternalDelay([]float64{0.7 * (1 + 9*dt*10)}, mat.NewDense(2, 1, []float64{1, 0.5}), mat.NewDense(1, 2, []float64{0.3, 1}),
			mat.NewDense(2, 1, []float64{0.2, 0}), mat.NewDense(1, 2, []float64{0, 0.1}), mat.NewDense(1, 1, []float64{0.1})); err != nil {
			t.Fatal(err)
		}
		gain, _ := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), dt)
		gain.Delay = mat.NewDense(2, 2, []float64{4 * (1 + 9*dt*10) / 10, 0, 0, 0})
		staticLFT, err := gain.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"descriptor": desc, "zero-state": staticLFT} {
			z, err := sys.ZeroDelayApprox()
			if err != nil {
				t.Fatalf("dt=%v %s: %v", dt, name, err)
			}
			if z.HasInternalDelay() {
				t.Fatalf("dt=%v %s: internal delays remain", dt, name)
			}
			assertResponseOracle(t, fmt.Sprintf("dt=%v %s ZeroDelayApprox", dt, name), z, func(s complex128) [][]complex128 {
				return evalDelaySystem(t, sys, s, noInternal(dt))
			})
		}
	}
}

func TestDelaySettersRejectNonFiniteDelays(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, v := range []float64{math.NaN(), math.Inf(1)} {
			sys := emptyIOFixture(t, 2, 2, 1, dt)
			if err := sys.SetInputDelay([]float64{0, v}); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("dt=%g SetInputDelay(%v) err = %v", dt, v, err)
			}
			if err := sys.SetOutputDelay([]float64{v}); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("dt=%g SetOutputDelay(%v) err = %v", dt, v, err)
			}
			if err := sys.SetDelay(mat.NewDense(1, 2, []float64{v, 0})); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("dt=%g SetDelay(%v) err = %v", dt, v, err)
			}
			if _, err := NewWithDelay(sys.A, sys.B, sys.C, sys.D, mat.NewDense(1, 2, []float64{0, v}), dt); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("dt=%g NewWithDelay(%v) err = %v", dt, v, err)
			}
			if sys.HasDelay() {
				t.Errorf("dt=%g rejected delay %v was stored", dt, v)
			}
		}
	}
}

func TestCopyKeepsStaticInternalDelayModelValid(t *testing.T) {
	g, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	g.OutputDelay = []float64{1}
	k, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := Feedback(g, k, -1)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := fb.Dims(); n != 0 || !fb.HasInternalDelay() {
		t.Fatalf("want static internal-delay model, n=%d internal=%v", n, fb.HasInternalDelay())
	}
	if err := fb.Copy().Validate(); err != nil {
		t.Errorf("Copy().Validate: %v", err)
	}
}

func TestSetInternalDelayStaticGainAcceptsNilEmptyBlocks(t *testing.T) {
	g, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	one := func(v float64) *mat.Dense { return mat.NewDense(1, 1, []float64{v}) }
	if err := g.SetInternalDelay([]float64{1}, nil, nil, one(1), one(1), one(0.2)); err != nil {
		t.Fatalf("SetInternalDelay: %v", err)
	}
	if err := g.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
	if !g.HasInternalDelay() {
		t.Error("HasInternalDelay = false")
	}
	if err := g.SetInternalDelay([]float64{1}, nil, nil, nil, one(1), one(0.2)); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("nil 1x1 D12: err = %v, want ErrDimensionMismatch", err)
	}
}

func TestAbsorbDelayScopeValidationAndUnion(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{0.5, 0.2, -0.1, 0.3},
		[]float64{1, 0.4, 0, 1},
		[]float64{1, 0, 0.3, 1},
		[]float64{0.1, 0, 0, 0.2}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{2, 1}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{0, 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := sys.AbsorbDelay("bogus"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("unknown scope: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := sys.AbsorbDelay(AbsorbInput, "bogus"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("unknown second scope: err = %v, want ErrInvalidArgument", err)
	}
	var nilSys *System
	if _, err := nilSys.AbsorbDelay(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil: err = %v", err)
	}

	got, err := sys.AbsorbDelay(AbsorbOutput, AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	if got.HasDelay() {
		t.Fatalf("input+output union left delays: in=%v out=%v", got.InputDelay, got.OutputDelay)
	}
	if n, _, _ := got.Dims(); n != 2+3+3 {
		t.Errorf("n = %d, want 8", n)
	}
	omega := []float64{0.3, 1.7, 5}
	want, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	have, err := got.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range want.Data {
		if cmplx.Abs(have.Data[k]-want.Data[k]) > 1e-12 {
			t.Errorf("H[%d] = %v, want %v", k, have.Data[k], want.Data[k])
		}
	}

	onlyIn, err := sys.AbsorbDelay(AbsorbInput, AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	if onlyIn.InputDelay != nil || onlyIn.OutputDelay[1] != 3 {
		t.Errorf("repeated input scope: in=%v out=%v", onlyIn.InputDelay, onlyIn.OutputDelay)
	}
}

func TestGetDelayModelContract(t *testing.T) {
	var nilSys *System
	if _, _, err := nilSys.GetDelayModel(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil: err = %v", err)
	}
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{-1, 0.4, -0.3, -2}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, -0.2}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(2, 2, []float64{2, 0.1, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	H, tau, err := sys.GetDelayModel()
	if err != nil || tau == nil || len(tau) != 0 {
		t.Fatalf("delay-free: tau=%v err=%v, want empty non-nil", tau, err)
	}
	if err := sys.SetInputDelay([]float64{0.4}); err != nil {
		t.Fatal(err)
	}
	H, tau, err = sys.GetDelayModel()
	if err != nil || len(tau) != 1 || tau[0] != 0.4 {
		t.Fatalf("tau=%v err=%v", tau, err)
	}
	H.E.Set(0, 0, 99)
	if sys.E.At(0, 0) != 2 {
		t.Error("GetDelayModel H.E aliases sys.E")
	}
}

func mustTotalDelay(t *testing.T, sys *System) *mat.Dense {
	t.Helper()
	td, err := sys.TotalDelay()
	if err != nil {
		t.Fatal(err)
	}
	return td
}
