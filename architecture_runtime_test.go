package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func runtimeArchitectureMIMO(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{0, 1, -2, -3},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestRuntimeArchitectureSISOLoopAnalysisRejectsMIMOConsistently(t *testing.T) {
	sys := runtimeArchitectureMIMO(t)
	checks := []struct {
		name string
		run  func() error
	}{
		{name: "Margin", run: func() error {
			_, err := Margin(sys)
			return err
		}},
		{name: "AllMargin", run: func() error {
			_, err := AllMargin(sys)
			return err
		}},
		{name: "DiskMargin", run: func() error {
			_, err := DiskMargin(sys)
			return err
		}},
		{name: "Nyquist", run: func() error {
			_, err := sys.Nyquist(nil, 0)
			return err
		}},
		{name: "RootLocus", run: func() error {
			_, err := RootLocus(sys, nil)
			return err
		}},
		{name: "Pidtune", run: func() error {
			_, err := Pidtune(sys, PidtunePI)
			return err
		}},
	}
	for _, tc := range checks {
		if err := tc.run(); !errors.Is(err, ErrNotSISO) {
			t.Fatalf("%s error = %v, want ErrNotSISO", tc.name, err)
		}
	}
}

func TestRuntimeArchitectureSISOLoopAnalysisKeepsGainWorkflow(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if wgFreq(mr) == wgFreq(mr) {
		t.Fatalf("WgFreq = %v, want NaN for static gain without gain crossover", wgFreq(mr))
	}
}

func TestRuntimeArchitectureCovarianceRejectsDescriptorWorkflows(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	checks := []struct {
		name string
		want error
		run  func() error
	}{
		{name: "Covar", want: ErrDescriptorUnsupported, run: func() error {
			_, err := Covar(sys, Qn)
			return err
		}},
		{name: "Kalmd singular E", want: ErrDescriptorSingular, run: func() error {
			singular := sys.Copy()
			singular.E = mat.NewDense(2, 2, []float64{2, 0, 0, 0})
			_, err := Kalmd(singular, Qn, Rn, 0.1, nil)
			return err
		}},
	}
	for _, tc := range checks {
		if err := tc.run(); !errors.Is(err, tc.want) {
			t.Fatalf("%s error = %v, want %v", tc.name, err, tc.want)
		}
	}
}

func TestRuntimeArchitectureTransformationsRejectDescriptorWorkflows(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	checks := []struct {
		name string
		run  func() error
	}{
		{name: "Reduce", run: func() error {
			_, err := sys.Reduce(nil)
			return err
		}},
		{name: "Sminreal", run: func() error {
			_, err := Sminreal(sys)
			return err
		}},
		{name: "Balreal", run: func() error {
			_, err := Balreal(sys)
			return err
		}},
		{name: "Canon", run: func() error {
			_, err := Canon(sys, CanonModal)
			return err
		}},
	}
	for _, tc := range checks {
		if err := tc.run(); !errors.Is(err, ErrDescriptorUnsupported) {
			t.Fatalf("%s error = %v, want ErrDescriptorUnsupported", tc.name, err)
		}
	}
}

func TestRuntimeArchitectureEnergyAnalysisRejectsDescriptorWorkflows(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	checks := []struct {
		name string
		run  func() error
	}{
		{name: "Gram", run: func() error {
			_, err := Gram(sys, GramControllability)
			return err
		}},
		{name: "H2Norm", run: func() error {
			_, err := H2Norm(sys)
			return err
		}},
		{name: "HSV", run: func() error {
			_, err := HSV(sys)
			return err
		}},
		{name: "HinfNorm", run: func() error {
			_, _, err := HinfNorm(sys)
			return err
		}},
	}
	for _, tc := range checks {
		if err := tc.run(); !errors.Is(err, ErrDescriptorUnsupported) {
			t.Fatalf("%s error = %v, want ErrDescriptorUnsupported", tc.name, err)
		}
	}
}
