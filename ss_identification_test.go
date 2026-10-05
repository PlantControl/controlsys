package controlsys

import (
	"context"
	"errors"
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
	if _, err := IdentifyIOStateSpace(ctx, u[:600], y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	for _, tc := range []struct {
		opts IOStateSpaceOptions
		want error
	}{
		{IOStateSpaceOptions{Order: 13}, ErrInvalidOrder},
		{IOStateSpaceOptions{InputDelay: 101}, ErrInvalidArgument},
		{IOStateSpaceOptions{MaxEvaluations: 2001}, ErrInvalidArgument},
		{IOStateSpaceOptions{ValidationInitialCondition: "estimate", InitializationSamples: 899}, ErrInvalidArgument},
		{IOStateSpaceOptions{InitialCondition: "guess"}, ErrInvalidArgument},
	} {
		if _, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], .1, tc.opts); !errors.Is(err, tc.want) {
			t.Errorf("options %+v: err = %v, want %v", tc.opts, err, tc.want)
		}
	}
	for _, dt := range []float64{0, math.NaN()} {
		if _, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:600], u[600:], y[600:], dt, IOStateSpaceOptions{}); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("dt=%g: err = %v, want ErrInvalidSampleTime", dt, err)
		}
	}
	if _, err := IdentifyIOStateSpace(context.Background(), u[:10], y[:10], u[600:], y[600:], .1, IOStateSpaceOptions{}); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("short record: err = %v, want ErrInsufficientData", err)
	}
	if _, err := IdentifyIOStateSpace(context.Background(), u[:600], y[:599], u[600:], y[600:], .1, IOStateSpaceOptions{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("length mismatch: err = %v, want ErrDimensionMismatch", err)
	}
	bad := append([]float64(nil), y[:600]...)
	bad[5] = math.Inf(1)
	if _, err := IdentifyIOStateSpace(context.Background(), u[:600], bad, u[600:], y[600:], .1, IOStateSpaceOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Inf sample: err = %v, want ErrInvalidArgument", err)
	}
	constant := make([]float64, 600)
	for k := range constant {
		constant[k] = 1
	}
	if _, err := IdentifyIOStateSpace(context.Background(), constant, y[:600], u[600:], y[600:], .1, IOStateSpaceOptions{}); err == nil {
		t.Fatal("unexcited input accepted")
	}
}

func TestIdentifyIOStateSpaceDelayedExcerptEstimatesPreRecordInputs(t *testing.T) {
	for _, direct := range []bool{false, true} {
		for _, delay := range []int{0, 2} {
			u, y := ioIdentificationFixture(0, direct, delay, []float64{.4, .3})
			const cut = 100
			for _, mode := range []IOInitialCondition{IOInitialContinuation, IOInitialEstimate} {
				options := IOStateSpaceOptions{Order: 2, InputDelay: delay, DirectFeedthrough: direct, ValidationInitialCondition: mode}
				vu, vy := u[600:], y[600:]
				if mode == "estimate" {
					options.InitializationSamples = 20
					vu, vy = u[650:850], y[650:850]
				}
				result, err := IdentifyIOStateSpace(context.Background(), u[cut:600], y[cut:600], vu, vy, .2, options)
				if err != nil {
					t.Fatal(err)
				}
				c := result.Candidates[0]
				if c.TrainingNRMSE > 1e-9 || c.ValidationNRMSE > 1e-8 {
					t.Fatalf("direct=%v delay=%d %s: training NRMSE %g validation NRMSE %g", direct, delay, mode, c.TrainingNRMSE, c.ValidationNRMSE)
				}
				num := []float64{.4, .1}
				if direct {
					num = append(num, 0)
				}
				den := make([]float64, 3+delay)
				den[0], den[1], den[2] = 1, -1.2, .32
				for i, v := range append(append([]float64(nil), num...), den...) {
					got := append(append([]float64(nil), c.Numerator...), c.Denominator...)
					if len(got) != len(num)+len(den) || math.Abs(got[i]-v) > 1e-8 {
						t.Fatalf("direct=%v delay=%d: num=%v den=%v, want %v %v", direct, delay, c.Numerator, c.Denominator, num, den)
					}
				}
				if len(c.InitialHistory) != 2+delay {
					t.Fatalf("direct=%v delay=%d: initial history %v, want %d values", direct, delay, c.InitialHistory, 2+delay)
				}
				// The true trailing feedthrough coefficient is zero, so direct fits cannot pin u[-delay].
				for i := 1; i <= delay && !direct; i++ {
					if got := c.InitialHistory[1+i]; math.Abs(got-u[cut-i]) > 1e-8 {
						t.Fatalf("direct=%v delay=%d: pre-record input u[-%d]=%g, want %g", direct, delay, i, got, u[cut-i])
					}
				}
			}
		}
	}
}

func TestIdentifyIOStateSpaceFailedCandidateCarriesError(t *testing.T) {
	u, y := ioIdentificationFixture(0, false, 0, nil)
	_, err := IdentifyIOStateSpace(context.Background(), u[:25], y[:25], u[600:], y[600:], .1, IOStateSpaceOptions{Order: 2})
	if !errors.Is(err, ErrInsufficientData) {
		t.Fatalf("too few training rows: err = %v, want ErrInsufficientData", err)
	}
	res, err := IdentifyIOStateSpace(context.Background(), u[:200], y[:200], u[600:], y[600:], .1, IOStateSpaceOptions{MinOrder: 1, MaxOrder: 12})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range res.Candidates {
		if c.Err != nil {
			if c.Convergence != IOFailed || c.Failure != c.Err.Error() || c.System != nil || c.ValidationNRMSE != 0 {
				t.Errorf("order %d failed candidate = %+v", c.Order, c)
			}
		} else if c.System == nil || c.Convergence == IOFailed {
			t.Errorf("order %d fitted candidate missing System or marked failed", c.Order)
		}
	}
}
