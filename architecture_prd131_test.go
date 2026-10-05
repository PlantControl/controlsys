package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPRD131RealizationTransformsPreserveMetadataAndSampleTime(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0.7, 0.2,
			-0.1, 0.5,
		}),
		mat.NewDense(2, 1, []float64{1, 0.4}),
		mat.NewDense(1, 2, []float64{0.3, 1}),
		mat.NewDense(1, 1, []float64{0.2}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"valve"}
	sys.OutputName = []string{"level"}

	check := func(name string, got *System) {
		t.Helper()
		if got.Dt != sys.Dt {
			t.Fatalf("%s Dt = %g, want %g", name, got.Dt, sys.Dt)
		}
		if len(got.InputName) != 1 || got.InputName[0] != "valve" {
			t.Fatalf("%s InputName = %v", name, got.InputName)
		}
		if len(got.OutputName) != 1 || got.OutputName[0] != "level" {
			t.Fatalf("%s OutputName = %v", name, got.OutputName)
		}
		assertFreqResponseApprox(t, sys, got, []float64{0.1, 0.7, 1.2}, 1e-8)
	}

	canon, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}
	check("Canon", canon.Sys)

	prescaled, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}
	check("Prescale", prescaled.Sys)

	ssbal, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}
	check("Ssbal", ssbal.Sys)
}

func TestPRD131ReductionAndBalancingPreserveResultConstructionBehavior(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0.5, 0.1,
			0, 0.2,
		}),
		mat.NewDense(2, 1, []float64{1, 0.3}),
		mat.NewDense(1, 2, []float64{0.4, 1}),
		mat.NewDense(1, 1, []float64{0.15}),
		0.2,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"input"}
	sys.OutputName = []string{"output"}

	reduced, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if reduced.Sys.Dt != sys.Dt || reduced.Sys.InputName[0] != "input" || reduced.Sys.OutputName[0] != "output" {
		t.Fatalf("MinimalRealization metadata not preserved: Dt=%g input=%v output=%v", reduced.Sys.Dt, reduced.Sys.InputName, reduced.Sys.OutputName)
	}
	assertFreqResponseApprox(t, sys, reduced.Sys, []float64{0.05, 0.4}, 1e-8)

	balred, _, err := Balred(sys, 1, Truncate)
	if err != nil {
		t.Fatal(err)
	}
	if balred.Dt != sys.Dt || balred.InputName[0] != "input" || balred.OutputName[0] != "output" {
		t.Fatalf("Balred metadata not preserved: Dt=%g input=%v output=%v", balred.Dt, balred.InputName, balred.OutputName)
	}
	if math.Abs(balred.D.At(0, 0)-sys.D.At(0, 0)) > 1e-12 {
		t.Fatalf("Balred feedthrough = %g, want %g", balred.D.At(0, 0), sys.D.At(0, 0))
	}
}

func TestPRD131StabsepUsesConsistentDecompositionResultAssembly(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			-0.4, 0.2,
			0, 0.3,
		}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{0.7, 1}),
		mat.NewDense(1, 1, []float64{2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}

	sep, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}
	if sep.Stable.InputName[0] != "u" || sep.Stable.OutputName[0] != "y" {
		t.Fatalf("stable metadata = %v/%v", sep.Stable.InputName, sep.Stable.OutputName)
	}
	if sep.Unstable.InputName[0] != "u" || sep.Unstable.OutputName[0] != "y" {
		t.Fatalf("unstable metadata = %v/%v", sep.Unstable.InputName, sep.Unstable.OutputName)
	}
	if math.Abs(sep.Stable.D.At(0, 0)-2) > 1e-12 {
		t.Fatalf("stable feedthrough = %g, want 2", sep.Stable.D.At(0, 0))
	}
	if math.Abs(sep.Unstable.D.At(0, 0)) > 1e-12 {
		t.Fatalf("unstable feedthrough = %g, want 0", sep.Unstable.D.At(0, 0))
	}
	sum, err := Parallel(sep.Stable, sep.Unstable)
	if err != nil {
		t.Fatal(err)
	}
	assertFreqResponseApprox(t, sys, sum, []float64{0.1, 1.0}, 1e-8)
}
