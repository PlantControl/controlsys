package controlsys

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func ioIdentificationFixture(noise float64, direct bool, delay int, initial []float64) (u, y []float64) {
	rng := rand.New(rand.NewSource(31))
	u = make([]float64, 900)
	y = make([]float64, len(u))
	truth := make([]float64, len(u))
	nk := 1 + delay
	if direct {
		nk = delay
	}
	for k := range u {
		u[k] = rng.NormFloat64()
		get := func(index int) float64 {
			if index >= 0 {
				return truth[index]
			}
			if -index-1 < len(initial) {
				return initial[-index-1]
			}
			return 0
		}
		truth[k] = 1.2*get(k-1) - .32*get(k-2)
		if k >= nk {
			truth[k] += .4 * u[k-nk]
		}
		if k >= nk+1 {
			truth[k] += .1 * u[k-nk-1]
		}
		y[k] = truth[k] + noise*rng.NormFloat64()
	}
	return
}

func TestIdentifyIOStateSpaceNoiseFreeOrdersAndHeldOut(t *testing.T) {
	u, y := ioIdentificationFixture(0, false, 0, []float64{.7, -.2})
	result, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{MinOrder: 1, MaxOrder: 4, InitialCondition: "estimate"})
	if err != nil {
		t.Fatal(err)
	}
	selected := result.Candidates[result.Selected]
	if selected.Order != 2 || selected.ValidationNRMSE > 1e-8 {
		t.Fatalf("order/error: %+v", selected)
	}
	for i, want := range []float64{1, -1.2, .32} {
		if math.Abs(selected.Denominator[i]-want) > 1e-9 {
			t.Fatalf("denominator %v", selected.Denominator)
		}
	}
	response, err := selected.System.FreqResponse([]float64{.2, 1, 10})
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{.2, 1, 10} {
		z := complex(math.Cos(w*.1), math.Sin(w*.1))
		want := (.4*z + .1) / (z*z - 1.2*z + .32)
		got := response.At(i, 0, 0)
		if math.Hypot(real(got-want), imag(got-want)) > 1e-9 {
			t.Fatal("identified SS transfer mismatch")
		}
	}
	if selected.Rank != 4 || len(selected.SingularValues) != 4 || !selected.Stable {
		t.Fatalf("missing diagnostics: %+v", selected)
	}
}

func TestIdentifyIOStateSpaceNoisyMultiModeAndTrainingIsolation(t *testing.T) {
	u, y := ioIdentificationFixture(.04, false, 0, []float64{1, -.2})
	options := IOStateSpaceOptions{Order: 2, InitialCondition: "estimate"}
	selected, selectionErr := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{MinOrder: 1, MaxOrder: 4, InitialCondition: "estimate"})
	if selectionErr != nil {
		t.Fatal(selectionErr)
	}
	if selected.Candidates[selected.Selected].Order != 2 {
		t.Fatalf("noisy order selection: %+v", selected)
	}
	result, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .1, options)
	if err != nil {
		t.Fatal(err)
	}
	fit := result.Candidates[0]
	if fit.ValidationNRMSE > .15 || fit.ValidationNRMSE >= fit.ValidationBaselineNRMSE*.2 {
		t.Fatalf("poor held-out fit %+v", fit)
	}
	changed := append([]float64(nil), y[600:]...)
	for k := range changed {
		changed[k] += 2 * math.Sin(float64(k))
	}
	other, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], changed, .1, options)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range fit.Denominator {
		if v != other.Candidates[0].Denominator[i] {
			t.Fatal("validation changed fitted dynamics")
		}
	}
	for i, v := range fit.InitialHistory {
		if v != other.Candidates[0].InitialHistory[i] {
			t.Fatal("validation changed training initial conditions")
		}
	}
}

func TestIdentifyIOStateSpaceDelayFeedthroughAndValidationInitialization(t *testing.T) {
	for _, direct := range []bool{false, true} {
		u, y := ioIdentificationFixture(0, direct, 2, []float64{.4, .3})
		result, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .2, IOStateSpaceOptions{Order: 2, InputDelay: 2, DirectFeedthrough: direct, InitialCondition: "estimate"})
		if err != nil {
			t.Fatal(err)
		}
		if result.Candidates[0].ValidationNRMSE > 1e-8 {
			t.Fatalf("delay/direct %v: %+v", direct, result.Candidates[0])
		}
	}
	u, y := ioIdentificationFixture(0, false, 0, []float64{.4, .3})
	vu, vy := ioIdentificationFixture(0, false, 0, []float64{3, -2})
	result, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], vu[:200], vy[:200], .1, IOStateSpaceOptions{Order: 2, InitialCondition: "estimate", ValidationInitialCondition: "estimate", InitializationSamples: 20})
	if err != nil {
		t.Fatal(err)
	}
	if result.Candidates[0].ValidationNRMSE > 1e-8 || result.Candidates[0].ValidationInitializationSamples != 20 {
		t.Fatal("separate validation initialization failed")
	}
}

func TestIdentifyIOStateSpaceBoundsExcitationAndCancellation(t *testing.T) {
	u, y := ioIdentificationFixture(0, false, 0, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := IdentifyIOStateSpace(ctx, u[:600], y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	for _, opts := range []IOStateSpaceOptions{{Order: 13}, {InputDelay: 101}, {MaxEvaluations: 2001}, {ValidationInitialCondition: "estimate", InitializationSamples: 899}} {
		if _, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .1, opts); err == nil {
			t.Fatalf("invalid options accepted %+v", opts)
		}
	}
	constant := make([]float64, 600)
	for k := range constant {
		constant[k] = 1
	}
	if _, err := IdentifyIOStateSpace(context.Background(), constant, y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{}); err == nil {
		t.Fatal("unexcited input accepted")
	}
}
