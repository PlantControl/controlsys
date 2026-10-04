package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestAllInputResponseMatchesIndependentSimulations(t *testing.T) {
	continuous, err := New(
		mat.NewDense(2, 2, []float64{-1.2, 0.7, -0.4, -2.1}),
		mat.NewDense(2, 3, []float64{1, -0.5, 0.25, 0.3, 1.2, -0.8}),
		mat.NewDense(2, 2, []float64{1.1, -0.2, 0.4, 0.9}),
		mat.NewDense(2, 3, []float64{0.1, -0.3, 0.2, 0.5, 0.25, -0.4}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	discrete, err := New(
		mat.NewDense(2, 2, []float64{0.7, 0.2, -0.1, 0.5}),
		mat.NewDense(2, 3, []float64{1, -0.5, 0.25, 0.3, 1.2, -0.8}),
		mat.NewDense(2, 2, []float64{1.1, -0.2, 0.4, 0.9}),
		mat.NewDense(2, 3, []float64{0.1, -0.3, 0.2, 0.5, 0.25, -0.4}),
		0.2,
	)
	if err != nil {
		t.Fatal(err)
	}
	gain, err := NewGain(mat.NewDense(2, 3, []float64{1, -2, 0.5, 0.25, 3, -0.75}), 0)
	if err != nil {
		t.Fatal(err)
	}
	delayed := discrete.Copy()
	delayed.Delay = mat.NewDense(2, 3, []float64{0, 1, 2, 2, 0, 1})

	tests := []struct {
		name   string
		sys    *System
		final  float64
		kind   standardInputResponse
		public func(*System, float64) (*TimeResponse, error)
	}{
		{name: "ContinuousStep", sys: continuous, final: 1.2, kind: stepResponse, public: Step},
		{name: "ContinuousImpulse", sys: continuous, final: 1.2, kind: impulseResponse, public: Impulse},
		{name: "DiscreteStep", sys: discrete, final: 2, kind: stepResponse, public: Step},
		{name: "DiscreteImpulse", sys: discrete, final: 2, kind: impulseResponse, public: Impulse},
		{name: "GainStep", sys: gain, final: 0.2, kind: stepResponse, public: Step},
		{name: "GainImpulse", sys: gain, final: 0.2, kind: impulseResponse, public: Impulse},
		{name: "DelayedStep", sys: delayed, final: 2, kind: stepResponse, public: Step},
		{name: "DelayedImpulse", sys: delayed, final: 2, kind: impulseResponse, public: Impulse},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.public(test.sys, test.final)
			if err != nil {
				t.Fatal(err)
			}
			want := independentInputResponse(t, test.sys, test.final, test.kind)
			compareTimeResponses(t, got, want, 2e-13)
		})
	}
}

func independentInputResponse(t *testing.T, sys *System, tFinal float64, kind standardInputResponse) *TimeResponse {
	t.Helper()
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, inputs, outputs := plan.sim.Dims()
	Y := mat.NewDense(outputs*inputs, plan.steps, nil)
	if kind == impulseResponse && sys.IsContinuous() {
		n, _, _ := sys.Dims()
		for sample, tk := range plan.t {
			if n == 0 {
				break
			}
			var scaled, expAt, cExp, h mat.Dense
			scaled.Scale(tk, sys.A)
			expAt.Exp(&scaled)
			cExp.Mul(sys.C, &expAt)
			h.Mul(&cExp, sys.B)
			for input := range inputs {
				for output := range outputs {
					Y.Set(input*outputs+output, sample, h.At(output, input))
				}
			}
		}
		return plan.response(Y)
	}
	amplitude := kind.amplitude(plan)
	for input := range inputs {
		u := mat.NewDense(inputs, plan.steps, nil)
		if kind == stepResponse {
			for sample := range plan.steps {
				u.Set(input, sample, amplitude)
			}
		} else {
			u.Set(input, 0, amplitude)
		}
		response, err := plan.sim.Simulate(u, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for output := range outputs {
			for sample := range plan.steps {
				Y.Set(input*outputs+output, sample, response.Y.At(output, sample))
			}
		}
	}
	return plan.response(Y)
}

func compareTimeResponses(t *testing.T, got, want *TimeResponse, tolerance float64) {
	t.Helper()
	if len(got.T) != len(want.T) {
		t.Fatalf("time-grid lengths differ: got %d, want %d", len(got.T), len(want.T))
	}
	for i := range got.T {
		if got.T[i] != want.T[i] {
			t.Fatalf("time %d = %.17g, want %.17g", i, got.T[i], want.T[i])
		}
	}
	gotRows, gotCols := got.Y.Dims()
	wantRows, wantCols := want.Y.Dims()
	if gotRows != wantRows || gotCols != wantCols {
		t.Fatalf("response dimensions = %dx%d, want %dx%d", gotRows, gotCols, wantRows, wantCols)
	}
	for row := range gotRows {
		for col := range gotCols {
			gotValue := got.Y.At(row, col)
			wantValue := want.Y.At(row, col)
			scale := math.Max(1, math.Max(math.Abs(gotValue), math.Abs(wantValue)))
			if math.Abs(gotValue-wantValue) > tolerance*scale {
				t.Fatalf("response(%d,%d) = %.17g, want %.17g", row, col, gotValue, wantValue)
			}
		}
	}
}
