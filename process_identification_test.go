package controlsys

import (
	"context"
	"errors"
	"math"
	"testing"
)

func processOracleInput(n int) []float64 {
	u := make([]float64, n)
	state := uint64(763)
	level := 0.0
	for i := range u {
		if i%7 == 0 {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			level = float64(int(state%7) - 3)
		}
		u[i] = level
	}
	return u
}

// Independent partial fractions of the continuous process step response;
// superposition of held input changes also resolves fractional delay exactly.
func processOracle(s ProcessStructure, p ProcessParameters, u []float64, dt float64) []float64 {
	step := func(t float64) float64 {
		if t < 0 {
			return 0
		}
		if s.UnderdampedPair {
			w := p.NaturalFrequency
			z := p.Damping
			wd := w * math.Sqrt(1-z*z)
			return p.Gain * (1 - math.Exp(-z*w*t)*(math.Cos(wd*t)+z*w/wd*math.Sin(wd*t)))
		}
		g, dg := 1.0, 0.0
		if s.Integrator {
			g = t
			dg = 1
		}
		for i, tau := range p.TimeConstants {
			coefficient := math.Pow(tau, float64(len(p.TimeConstants)-1))
			for j, other := range p.TimeConstants {
				if j != i {
					coefficient /= tau - other
				}
			}
			e := math.Exp(-t / tau)
			if s.Integrator {
				g -= coefficient * tau * (1 - e)
				dg -= coefficient * e
			} else {
				g -= coefficient * e
				dg += coefficient / tau * e
			}
		}
		return p.Gain * (g + p.ZeroTime*dg)
	}
	y := make([]float64, len(u))
	previous := 0.0
	for j, value := range u {
		change := value - previous
		previous = value
		if change == 0 {
			continue
		}
		for i := j; i < len(y); i++ {
			y[i] += change * step(float64(i-j)*dt-p.Delay)
		}
	}
	return y
}

func TestProcessFitIndependentStructures(t *testing.T) {
	cases := []struct {
		name            string
		s               ProcessStructure
		p               ProcessParameters
		initial, offset bool
	}{
		{"one pole fractional delay", ProcessStructure{Poles: 1, Delay: true}, ProcessParameters{Gain: 2.3, TimeConstants: []float64{1.7}, Delay: .235}, false, false},
		{"two poles", ProcessStructure{Poles: 2}, ProcessParameters{Gain: -1.8, TimeConstants: []float64{.4, 1.4}}, false, false},
		{"three poles", ProcessStructure{Poles: 3}, ProcessParameters{Gain: 2, TimeConstants: []float64{.25, .8, 1.8}}, false, false},
		{"nonminimum zero", ProcessStructure{Poles: 2, Zero: true}, ProcessParameters{Gain: 1.3, TimeConstants: []float64{.4, 1.4}, ZeroTime: -.3}, false, false},
		{"integrator", ProcessStructure{Poles: 1, Integrator: true}, ProcessParameters{Gain: .7, TimeConstants: []float64{.8}}, false, false},
		{"initial state and offset", ProcessStructure{Poles: 2}, ProcessParameters{Gain: 1.3, TimeConstants: []float64{.4, 1.4}}, true, true},
		{"underdamped pair", ProcessStructure{Poles: 2, UnderdampedPair: true}, ProcessParameters{Gain: 1.5, NaturalFrequency: 2, Damping: .35}, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u := processOracleInput(500)
			y := processOracle(tc.s, tc.p, u, .1)
			if tc.initial {
				for i := range y {
					y[i] += .4 * math.Exp(-float64(i)*.1/tc.p.TimeConstants[0])
				}
			}
			if tc.offset {
				for i := range y {
					y[i] += 3
				}
			}
			initial := tc.p
			initial.TimeConstants = append([]float64(nil), tc.p.TimeConstants...)
			for i := range initial.TimeConstants {
				initial.TimeConstants[i] *= 1.15
			}
			if tc.s.Delay {
				initial.Delay += .1
			}
			if tc.s.Zero {
				initial.ZeroTime *= .8
			}
			if tc.s.UnderdampedPair {
				initial.NaturalFrequency *= 1.15
				initial.Damping *= 1.1
			}
			options := ProcessFitOptions{Structure: tc.s, Initial: &initial, EstimateInitialState: tc.initial, EstimateOffset: tc.offset, Starts: 2, Bounds: ProcessBounds{MinTimeConstant: .1, MaxTimeConstant: 4, MinDelay: 0, MaxDelay: .8, MinZero: -1, MaxZero: 1, MinFrequency: .5, MaxFrequency: 5}}
			result, err := FitProcess(context.Background(), ProcessFitData{Input: u, Output: y, SampleTime: .1, TrainingSamples: 350}, options)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("fit %+v error %.3g eval=%d status=%s", result.Parameters, result.ValidationNRMSE, result.Evaluations, result.Termination)
			if result.ValidationNRMSE > .01 || math.Abs(result.Parameters.Gain/tc.p.Gain-1) > .01 {
				t.Fatalf("fit misses quantitative gate: %+v err %g", result.Parameters, result.ValidationNRMSE)
			}
			for i, tau := range tc.p.TimeConstants {
				if math.Abs(result.Parameters.TimeConstants[i]/tau-1) > .01 {
					t.Fatalf("tau %d got %g want %g", i, result.Parameters.TimeConstants[i], tau)
				}
			}
			if tc.s.Delay && math.Abs(result.Parameters.Delay-tc.p.Delay) > .01 {
				t.Fatalf("fractional delay %g want %g", result.Parameters.Delay, tc.p.Delay)
			}
			if tc.s.Zero && math.Abs(result.Parameters.ZeroTime-tc.p.ZeroTime) > .01 {
				t.Fatalf("zero %g want %g", result.Parameters.ZeroTime, tc.p.ZeroTime)
			}
			if tc.s.UnderdampedPair && (math.Abs(result.Parameters.Damping/tc.p.Damping-1) > .01 || math.Abs(result.Parameters.NaturalFrequency/tc.p.NaturalFrequency-1) > .01) {
				t.Fatal("pair parameters missed gate")
			}
		})
	}
}

func TestProcessFitReinitializationNoiseAndValidationIsolation(t *testing.T) {
	s := ProcessStructure{Poles: 1}
	p := ProcessParameters{Gain: 2.3, TimeConstants: []float64{1.7}}
	u := processOracleInput(500)
	y := processOracle(s, p, u, .1)
	for i := range y {
		y[i] += .02 * math.Sin(float64(i)*2.17)
	}
	options := ProcessFitOptions{Structure: s, Starts: 2}
	data := ProcessFitData{Input: u, Output: y, SampleTime: .1, TrainingSamples: 350}
	first, err := FitProcess(context.Background(), data, options)
	if err != nil {
		t.Fatal(err)
	}
	if first.ValidationNRMSE >= first.ValidationMeanNRMSE || first.ValidationNRMSE > .05 {
		t.Fatalf("noisy fit fails baseline: %g vs %g", first.ValidationNRMSE, first.ValidationMeanNRMSE)
	}
	for i := 350; i < len(y); i++ {
		y[i] += 1000
	}
	second, err := FitProcess(context.Background(), data, options)
	if err != nil {
		t.Fatal(err)
	}
	if first.Parameters.Gain != second.Parameters.Gain || first.Parameters.TimeConstants[0] != second.Parameters.TimeConstants[0] || first.Evaluations != second.Evaluations {
		t.Fatal("held-out outputs affected estimation")
	}
}

func TestProcessFitBoundsCancellationAndExcitation(t *testing.T) {
	u := processOracleInput(100)
	s := ProcessStructure{Poles: 1}
	p := ProcessParameters{Gain: 2, TimeConstants: []float64{1}}
	d := ProcessFitData{Input: u, Output: processOracle(s, p, u, .1), SampleTime: .1, TrainingSamples: 70}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := FitProcess(ctx, d, ProcessFitOptions{Structure: s}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	result, err := FitProcess(context.Background(), d, ProcessFitOptions{Structure: s, MaxEvaluations: 20, Starts: 1})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || result.Evaluations > 20 || result.Termination == "converged" {
		t.Fatalf("exhaustion %+v", result)
	}
	for i := range u {
		u[i] = 1
	}
	if _, err = FitProcess(context.Background(), d, ProcessFitOptions{Structure: s}); !errors.Is(err, ErrProcessExcitation) {
		t.Fatalf("excitation: %v", err)
	}
}

func TestProcessHeldInputBasisMatchesAnalyticStep(t *testing.T) {
	p := ProcessParameters{Gain: 1, TimeConstants: []float64{1.7}}
	s, err := processSystem(ProcessStructure{Poles: 1}, p)
	if err != nil {
		t.Fatal(err)
	}
	u := make([]float64, 10)
	for i := range u {
		u[i] = 1
	}
	response, _, err := processResponseBasis(s, u, .1, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	want := 1 - math.Exp(-.1/1.7)
	if math.Abs(response[1]-want) > 1e-10 {
		t.Fatalf("got %g want %g", response[1], want)
	}
}

func TestProcessManualGainAndSeparateValidation(t *testing.T) {
	s := ProcessStructure{Poles: 1}
	p := ProcessParameters{Gain: 2, TimeConstants: []float64{1}}
	u := processOracleInput(200)
	train := processOracle(s, p, u[:140], .1)
	validation := processOracle(s, p, u[140:], .1)
	y := append(train, validation...)
	d := ProcessFitData{Input: u, Output: y, SampleTime: .1, TrainingSamples: 140}
	result, err := EvaluateProcess(context.Background(), d, ProcessFitOptions{Structure: s, ValidationInitialCondition: "zero", ValidationInitializationSamples: 5}, p)
	if err != nil {
		t.Fatal(err)
	}
	if result.ValidationNRMSE > 1e-10 || result.Parameters.Gain != 2 {
		t.Fatalf("manual/zero init result gain %g rmse %g", result.Parameters.Gain, result.ValidationNRMSE)
	}
	p.Gain = 1
	result, err = EvaluateProcess(context.Background(), d, ProcessFitOptions{Structure: s, ValidationInitialCondition: "zero"}, p)
	if err != nil {
		t.Fatal(err)
	}
	if result.Parameters.Gain != 1 || result.ValidationNRMSE < .4 {
		t.Fatal("manual gain was optimized")
	}
}

type processCancelContext struct {
	context.Context
	calls    int
	canceled bool
	done     chan struct{}
}

func (c *processCancelContext) Done() <-chan struct{} { return c.done }
func (c *processCancelContext) Err() error {
	c.calls++
	if c.calls >= 40 && !c.canceled {
		c.canceled = true
		close(c.done)
	}
	if c.canceled {
		return context.Canceled
	}
	return nil
}
func TestProcessFitRetainsBestOnCancellation(t *testing.T) {
	u := processOracleInput(200)
	s := ProcessStructure{Poles: 2}
	p := ProcessParameters{Gain: 2, TimeConstants: []float64{.4, 1.7}}
	d := ProcessFitData{Input: u, Output: processOracle(s, p, u, .1), SampleTime: .1, TrainingSamples: 140}
	ctx := &processCancelContext{Context: context.Background(), done: make(chan struct{})}
	result, err := FitProcess(ctx, d, ProcessFitOptions{Structure: s})
	if !errors.Is(err, context.Canceled) || result == nil || result.Termination != "canceled" || !processFinite(result.ValidationNRMSE) {
		t.Fatalf("canceled fit did not retain valid best: result=%v error=%v", result != nil, err)
	}
}
