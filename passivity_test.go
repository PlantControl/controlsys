package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPassivityDenseAndFRD(t *testing.T) {
	passive := makeSISO(-1, 1, 1, 0)
	result, err := Passive(passive, nil)
	if err != nil {
		t.Fatalf("Passive: %v", err)
	}
	if !result.Passive || result.MinHermitianPart < 0 {
		t.Fatalf("expected passive result, got %#v", result)
	}
	if result.Status != PassivitySampled || len(result.Omega) != 121 {
		t.Fatalf("expected disclosed sampled result, got %#v", result)
	}

	nonpassive := makeSISO(-1, 1, -1, 0)
	bad, err := Passive(nonpassive, nil)
	if err != nil {
		t.Fatalf("Passive nonpassive: %v", err)
	}
	if bad.Passive {
		t.Fatalf("expected nonpassive result, got %#v", bad)
	}
	if bad.Status != PassivityViolated {
		t.Fatalf("nonpassive status = %q, want %q", bad.Status, PassivityViolated)
	}

	frd, err := passive.FRD([]float64{0.1, 1, 10})
	if err != nil {
		t.Fatal(err)
	}
	frdResult, err := FRDPassive(frd, nil)
	if err != nil {
		t.Fatalf("FRDPassive: %v", err)
	}
	if !frdResult.Passive {
		t.Fatalf("expected passive FRD result, got %#v", frdResult)
	}
	if got, want := frdResult.Omega, []float64{0.1, 1, 10}; !sameFloatSlice(got, want) {
		t.Fatalf("checked frequencies = %v, want %v", got, want)
	}
}

func TestSampledPassiveDoesNotClaimCertification(t *testing.T) {
	sys := makeSISO(-1, 1, -2, 1)
	result, err := SampledPassive(sys, &PassivityOptions{Omega: []float64{10, 100}})
	if err != nil {
		t.Fatalf("SampledPassive high-frequency grid: %v", err)
	}
	if !result.Passive || result.Status != PassivitySampled {
		t.Fatalf("high-frequency sampled result = %#v, want sampled pass", result)
	}

	result, err = SampledPassive(sys, &PassivityOptions{Omega: []float64{0, 10}})
	if err != nil {
		t.Fatalf("SampledPassive grid including DC: %v", err)
	}
	if result.Passive || result.Status != PassivityViolated || result.Frequency != 0 {
		t.Fatalf("grid including DC result = %#v, want violation at DC", result)
	}
}

func TestSampledPassiveValidatesGridAndUsesDiscreteNyquist(t *testing.T) {
	discrete := makeSISO(0.5, 1, 1, 0)
	discrete.Dt = 0.2
	result, err := SampledPassive(discrete, nil)
	if err != nil {
		t.Fatalf("SampledPassive discrete default: %v", err)
	}
	if got, want := result.Omega[len(result.Omega)-1], 3.141592653589793/0.2; got != want {
		t.Fatalf("discrete grid upper frequency = %g, want Nyquist %g", got, want)
	}
	if _, err := SampledPassive(discrete, &PassivityOptions{Omega: []float64{0, 20}}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("above-Nyquist grid err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := SampledPassive(discrete, &PassivityOptions{Omega: []float64{1, 0}}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("unsorted grid err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := SampledPassive(discrete, &PassivityOptions{Omega: []float64{1, 1}}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("duplicate frequency err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := SampledPassive(discrete, &PassivityOptions{Tol: -1}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("negative tolerance err = %v, want ErrDimensionMismatch", err)
	}
}

func TestFRDPassiveUsesMIMOHermitianEigenvalue(t *testing.T) {
	frd, err := NewFRD([][][]complex128{
		{
			{1, 2},
			{2, 1},
		},
	}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	result, err := FRDPassive(frd, nil)
	if err != nil {
		t.Fatalf("FRDPassive: %v", err)
	}
	if result.Passive {
		t.Fatalf("expected nonpassive MIMO result, got %#v", result)
	}
	if result.MinHermitianPart != -1 {
		t.Fatalf("min Hermitian part = %g, want -1", result.MinHermitianPart)
	}
}

func TestSpectralFactorStaticGainAndUnsupportedCases(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	factor, err := SpectralFactor(sys)
	if err != nil {
		t.Fatalf("SpectralFactor: %v", err)
	}
	if factor.D.At(0, 0) != 2 {
		t.Fatalf("factor D = %g, want 2", factor.D.At(0, 0))
	}

	dynamic := makeSISO(-1, 1, 1, 0)
	if _, err := SpectralFactor(dynamic); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("dynamic spectral factor err = %v, want ErrDimensionMismatch", err)
	}
	delayed := sys.Copy()
	delayed.InputDelay = []float64{1}
	if _, err := SpectralFactor(delayed); !errors.Is(err, ErrDescriptorUnsupported) {
		t.Fatalf("delayed spectral factor err = %v, want ErrDescriptorUnsupported", err)
	}
}

func sameFloatSlice(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
