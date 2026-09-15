package controlsys

import (
	"context"
	"math"
	"testing"
)

func TestTunePIDFRDMatchesAnalyticSystem(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	omega := make([]float64, 1001)
	for k := range omega {
		omega[k] = math.Exp(math.Log(.01) + float64(k)*math.Log(10000)/1000)
	}
	response, err := p.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	values := make([][][]complex128, len(omega))
	for k := range values {
		values[k] = [][]complex128{{response.At(k, 0, 0)}}
	}
	frd, err := NewFRD(values, omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	o := PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60, Weights: &PIDTuningWeights{}}
	a, err := TunePID(context.Background(), p, PidtunePI, o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TunePIDFRD(context.Background(), frd, PidtunePI, o)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(a.Controller.Kp-b.Controller.Kp) > 1e-3 || math.Abs(a.Controller.Ki-b.Controller.Ki) > 1e-3 {
		t.Fatalf("system %+v FRD %+v", a.Controller, b.Controller)
	}
	if b.Evidence.Stability != "sampled-frequency-only" || len(b.Evidence.Warnings) < 2 {
		t.Fatal("missing frequency-only/unknown poles evidence")
	}
	if math.Abs(b.Evidence.AchievedPhaseMargin-60) > 3.000001 {
		t.Fatal("target missed")
	}
}

func TestTunePIDFRDRejectsMalformedCoverage(t *testing.T) {
	makeFRD := func() *FRD {
		return &FRD{Omega: []float64{.1, 1, 10}, Response: [][][]complex128{{{1}}, {{.5 - .5i}}, {{.01 - .1i}}}}
	}
	for _, edit := range []func(*FRD){func(f *FRD) { f.Omega[0] = 0 }, func(f *FRD) { f.Omega[1] = .1 }, func(f *FRD) { f.Response[1][0][0] = complex(math.NaN(), 0) }, func(f *FRD) { f.Response[1] = nil }, func(f *FRD) { f.Dt = math.NaN() }} {
		f := makeFRD()
		edit(f)
		if _, err := TunePIDFRD(context.Background(), f, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1}); err == nil {
			t.Fatal("accepted malformed FRD")
		}
	}
	for _, wc := range []float64{.01, .1, 10, 100} {
		if _, err := TunePIDFRD(context.Background(), makeFRD(), PidtunePI, PIDTuningOptions{CrossoverFrequency: wc}); err == nil {
			t.Fatal("accepted out-of-band or endpoint crossover")
		}
	}
}

func TestTunePIDFRDExactDelayedSystemAgreement(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	if err := p.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	omega := make([]float64, 2001)
	for k := range omega {
		omega[k] = math.Exp(math.Log(.01) + float64(k)*math.Log(10000)/2000)
	}
	response, err := p.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	values := make([][][]complex128, len(omega))
	for k := range values {
		values[k] = [][]complex128{{response.At(k, 0, 0)}}
	}
	frd, err := NewFRD(values, omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	known := 0
	o := PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60, UnstablePoles: &known}
	a, err := TunePID(context.Background(), p, PidtunePI, o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TunePIDFRD(context.Background(), frd, PidtunePI, o)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(a.Evidence.AchievedCrossover-b.Evidence.AchievedCrossover) > .05 || math.Abs(a.Evidence.AchievedPhaseMargin-b.Evidence.AchievedPhaseMargin) > 3 {
		t.Fatal("delayed FRD/System disagreement")
	}
	if b.Evidence.Stability != "sampled-frequency-only" {
		t.Fatal("known pole count promoted finite FRD to global certificate")
	}
}
