package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestDescriptorConstructionAccessAndExplicitConversion(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-2, 1, -3, -4})
	B := mat.NewDense(2, 1, []float64{2, -4})
	C := mat.NewDense(1, 2, []float64{1, -1})
	D := mat.NewDense(1, 1, []float64{0.5})
	E := mat.NewDense(2, 2, []float64{2, 0, 0, 4})

	desc, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatalf("NewDescriptor: %v", err)
	}
	if !desc.IsDescriptor() {
		t.Fatal("expected descriptor model")
	}
	gotE := desc.DescriptorE()
	gotE.Set(0, 0, 99)
	if desc.E.At(0, 0) == 99 {
		t.Fatal("DescriptorE returned aliased matrix")
	}

	explicit, err := desc.ToExplicit()
	if err != nil {
		t.Fatalf("ToExplicit: %v", err)
	}
	wantA := mat.NewDense(2, 2, []float64{-1, 0.5, -0.75, -1})
	wantB := mat.NewDense(2, 1, []float64{1, -1})
	if explicit.IsDescriptor() {
		t.Fatal("explicit model still reports descriptor")
	}
	if !matEqual(explicit.A, wantA, 1e-12) || !matEqual(explicit.B, wantB, 1e-12) {
		t.Fatalf("explicit matrices mismatch\nA=%v\nB=%v", mat.Formatted(explicit.A), mat.Formatted(explicit.B))
	}

	singular, err := NewDescriptor(A, B, C, D, mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := singular.ToExplicit(); !errors.Is(err, ErrDescriptorSingular) {
		t.Fatalf("singular ToExplicit err = %v, want ErrDescriptorSingular", err)
	}
}

func TestStateSpaceUtilityWrappersPreserveBehaviorAndMetadata(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.InputName = []string{"left", "right"}
	sys.OutputName = []string{"position"}
	sys.StateName = []string{"fast", "slow"}

	truncated, err := sys.EliminateStates([]int{1}, Truncate)
	if err != nil {
		t.Fatalf("EliminateStates: %v", err)
	}
	if n, _, _ := truncated.Dims(); n != 1 {
		t.Fatalf("truncated order = %d, want 1", n)
	}
	if !sameStrings(truncated.InputName, sys.InputName) || !sameStrings(truncated.OutputName, sys.OutputName) {
		t.Fatalf("metadata lost input=%v output=%v", truncated.InputName, truncated.OutputName)
	}

	T := mat.NewDense(2, 2, []float64{1, 2, 0, 1})
	equivalent, err := sys.StateTransform(T)
	if err != nil {
		t.Fatalf("StateTransform: %v", err)
	}
	omega := []float64{0.1, 1.0, 3.0}
	baseResp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	equivResp, err := equivalent.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		for i := 0; i < baseResp.P; i++ {
			for j := 0; j < baseResp.M; j++ {
				if cmplx.Abs(baseResp.At(k, i, j)-equivResp.At(k, i, j)) > 1e-10 {
					t.Fatalf("transformed response mismatch at (%d,%d,%d)", k, i, j)
				}
			}
		}
	}
}

func TestFixedInputReductionAddsOffsetChannel(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.InputName = []string{"free", "bias"}
	sys.OutputName = []string{"y"}

	reduced, err := sys.FixedInputReduction(map[int]float64{1: 2.5}, "bias_offset")
	if err != nil {
		t.Fatalf("FixedInputReduction: %v", err)
	}
	if _, m, p := reduced.Dims(); m != 2 || p != 1 {
		t.Fatalf("reduced dims = (_, %d, %d), want (_, 2, 1)", m, p)
	}
	if !sameStrings(reduced.InputName, []string{"free", "bias_offset"}) {
		t.Fatalf("input names = %v", reduced.InputName)
	}

	uOriginal := mat.NewDense(4, 2, []float64{
		1, 2.5,
		0.5, 2.5,
		-0.25, 2.5,
		0, 2.5,
	})
	uReduced := mat.NewDense(4, 2, []float64{
		1, 1,
		0.5, 1,
		-0.25, 1,
		0, 1,
	})
	tGrid := []float64{0, 0.1, 0.2, 0.3}
	origResp, err := Lsim(sys, uOriginal, tGrid, nil)
	if err != nil {
		t.Fatal(err)
	}
	redResp, err := Lsim(reduced, uReduced, tGrid, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !matEqual(origResp.Y, redResp.Y, 1e-12) {
		t.Fatalf("fixed-input response mismatch\norig=%v\nred=%v", mat.Formatted(origResp.Y), mat.Formatted(redResp.Y))
	}
}

func TestAugmentInternalDelayOutputs(t *testing.T) {
	sys := utilityTestSystem(t)
	sys.OutputName = []string{"plant_y"}
	if err := sys.SetInternalDelay(
		[]float64{0.2},
		mat.NewDense(2, 1, []float64{0.5, -0.25}),
		mat.NewDense(1, 2, []float64{0.3, -0.4}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 2, []float64{0.2, 0.1}),
		mat.NewDense(1, 1, []float64{0}),
	); err != nil {
		t.Fatalf("SetInternalDelay: %v", err)
	}

	aug, err := sys.AugmentInternalDelayOutputs("delay")
	if err != nil {
		t.Fatalf("AugmentInternalDelayOutputs: %v", err)
	}
	if _, _, p := aug.Dims(); p != 2 {
		t.Fatalf("outputs = %d, want 2", p)
	}
	if !sameStrings(aug.OutputName, []string{"plant_y", "delay1"}) {
		t.Fatalf("output names = %v", aug.OutputName)
	}
	if !aug.HasInternalDelay() || aug.LFT.Tau[0] != 0.2 {
		t.Fatalf("internal delay not preserved: %v", aug.LFT)
	}
	if aug.LFT.C2.At(0, 0) != 0.3 || aug.LFT.D21.At(0, 1) != 0.1 {
		t.Fatal("internal-delay output matrices not preserved")
	}
}

func utilityTestSystem(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, -2, -3}),
		mat.NewDense(2, 2, []float64{1, -1, 0.5, 2}),
		mat.NewDense(1, 2, []float64{2, -0.5}),
		mat.NewDense(1, 2, []float64{0.25, -0.75}),
		0,
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return sys
}
