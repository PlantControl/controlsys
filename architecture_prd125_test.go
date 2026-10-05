package controlsys

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPRD125SampledTimeDomainWorkflowsPreserveOrientation(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0.7, 0.2,
			-0.1, 0.8,
		}),
		mat.NewDense(2, 2, []float64{
			1.0, 0.3,
			0.2, 0.7,
		}),
		mat.NewDense(2, 2, []float64{
			1.0, -0.4,
			0.5, 0.9,
		}),
		mat.NewDense(2, 2, []float64{
			0.1, 0.05,
			-0.02, 0.2,
		}),
		0.2,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.OutputName = []string{"temperature", "pressure"}

	uSamplesByChannels := mat.NewDense(5, 2, []float64{
		1, 0,
		0, 1,
		2, -1,
		0.5, 0.25,
		-1, 2,
	})
	uChannelsBySamples := mat.NewDense(2, 5, []float64{
		1, 0, 2, 0.5, -1,
		0, 1, -1, 0.25, 2,
	})
	tGrid := []float64{0, 0.2, 0.4, 0.6, 0.8}

	sim, err := sys.Simulate(uChannelsBySamples, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	lsim, err := Lsim(sys, uSamplesByChannels, tGrid, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "Lsim output uses the same sampled signal orientation as Simulate", lsim.Y, sim.Y, 1e-12)
	if got, want := lsim.OutputName, sys.OutputName; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("Lsim output names = %v, want %v", got, want)
	}
}

func TestPRD125LsimAllowsZeroInputAutonomousSampledSignal(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0.5, 0.1,
			0, 0.25,
		}),
		nil,
		mat.NewDense(1, 2, []float64{1, 1}),
		nil,
		1.0,
	)
	if err != nil {
		t.Fatal(err)
	}

	tGrid := []float64{0, 1, 2, 3}
	u := mat.NewDense(len(tGrid), 1, nil).Slice(0, len(tGrid), 0, 0).(*mat.Dense)
	x0 := mat.NewVecDense(2, []float64{4, 8})
	resp, err := Lsim(sys, u, tGrid, x0)
	if err != nil {
		t.Fatal(err)
	}

	wantY := mat.NewDense(1, 4, []float64{12, 4.8, 2.1, 0.975})
	assertMatClose(t, "autonomous Lsim output", resp.Y, wantY, 1e-12)
}

func TestPRD125FreqRespEstUsesStridedSampledSignals(t *testing.T) {
	dt := 0.05
	steps := 512
	baseInput := mat.NewDense(2, steps+3, nil)
	baseOutput := mat.NewDense(2, steps+3, nil)
	input := baseInput.Slice(0, 2, 1, steps+1).(*mat.Dense)
	output := baseOutput.Slice(0, 2, 1, steps+1).(*mat.Dense)

	rng := rand.New(rand.NewSource(125))
	for k := range steps {
		u0 := rng.NormFloat64()
		u1 := rng.NormFloat64()
		input.Set(0, k, u0)
		input.Set(1, k, u1)
		output.Set(0, k, 0.6*u0-0.2*u1)
		output.Set(1, k, 0.1*u0+0.8*u1)
	}

	est, err := FreqRespEst(input, output, dt, &FreqRespEstOpts{NFFT: 128})
	if err != nil {
		t.Fatal(err)
	}
	if est.H.P != 2 || est.H.M != 2 {
		t.Fatalf("estimated response dims = %dx%d, want 2x2", est.H.P, est.H.M)
	}
	if coh := est.CoherenceAt(1, 0, 0); math.IsNaN(coh) || coh <= 0 {
		t.Fatalf("coherence from strided sampled signal = %g, want positive finite", coh)
	}
}

func TestPRD125ERAMarkovValidationPreservesPublicErrors(t *testing.T) {
	_, err := ERA([]*mat.Dense{
		mat.NewDense(2, 1, nil),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 1, nil),
	}, 1, 0.1)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("ERA Markov dimension error = %v, want ErrDimensionMismatch", err)
	}

	_, err = ERA([]*mat.Dense{
		mat.NewDense(1, 1, nil),
		mat.NewDense(1, 1, nil),
		mat.NewDense(1, 1, nil),
	}, 1, 0)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Fatalf("ERA sample time error = %v, want ErrInvalidSampleTime", err)
	}
}

func TestPRD125MatrixEquationPublicErrorBehavior(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{
		0, 1,
		-2, -3,
	})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Qnonsym := mat.NewDense(2, 2, []float64{
		1, 2,
		0, 1,
	})
	R := mat.NewDense(1, 1, []float64{1})
	if _, err := Care(A, B, Qnonsym, R, nil); !errors.Is(err, ErrNotSymmetric) {
		t.Fatalf("Care nonsymmetric Q error = %v, want ErrNotSymmetric", err)
	}

	QnotPSD := mat.NewDense(2, 2, []float64{
		1, 0,
		0, -1,
	})
	if _, err := Dare(A, B, QnotPSD, R, nil); !errors.Is(err, ErrNotPSD) {
		t.Fatalf("Dare non-PSD Q error = %v, want ErrNotPSD", err)
	}

	if _, err := Lyap(A, Qnonsym, nil); !errors.Is(err, ErrNotSymmetric) {
		t.Fatalf("Lyap nonsymmetric Q error = %v, want ErrNotSymmetric", err)
	}
	if _, err := DLyap(A, mat.NewDense(1, 1, []float64{1}), nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("DLyap dimension error = %v, want ErrDimensionMismatch", err)
	}
}

func TestPRD125MatrixEquationPublicNumericalBehavior(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0.7, 0.2,
			-0.1, 0.6,
		}),
		mat.NewDense(2, 1, []float64{0.3, 1.0}),
		mat.NewDense(1, 2, []float64{1.0, -0.25}),
		mat.NewDense(1, 1, nil),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	Q := mat.NewDense(2, 2, []float64{1, 0.1, 0.1, 2})
	R := mat.NewDense(1, 1, []float64{1.5})
	if _, err := Dlqr(sys.A, sys.B, Q, R, nil); err != nil {
		t.Fatalf("Dlqr with nonsymmetric A: %v", err)
	}
	if _, err := Kalman(sys, mat.NewDense(1, 1, []float64{0.2}), mat.NewDense(1, 1, []float64{0.4}), nil, nil); err != nil {
		t.Fatalf("Kalman with nonsymmetric A: %v", err)
	}
	if h2, err := H2Norm(sys); err != nil || h2 <= 0 || math.IsNaN(h2) {
		t.Fatalf("H2Norm = %g, %v; want positive finite", h2, err)
	}
	if hsv, err := HSV(sys); err != nil || len(hsv) != 2 {
		t.Fatalf("HSV = %v, %v; want two values", hsv, err)
	}
}
