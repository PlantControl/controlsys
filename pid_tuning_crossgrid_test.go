package controlsys

import (
	"context"
	"math"
	"slices"
	"testing"
	"time"
)

func pidWithin(t *testing.T, d time.Duration, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		f()
	}()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("did not return within %v", d)
	}
}

// A floor grid's crossLow of 0 (or a crossHigh of +Inf) made the widening
// loops run forever; bounds outside (0, ω₀) and (ω_end, ∞) widen nothing.
func TestPIDTuningCrossGridUnsetBounds(t *testing.T) {
	omega := []float64{.01, .1, 1, 10, 100}
	m := complex(-.05, 3)
	base := pidTuningCrossGrid(pidTuningPlant{modes: []complex128{m}}, omega)
	for _, b := range [][2]float64{{0, 0}, {-1, -1}, {math.NaN(), math.NaN()}, {math.Inf(-1), math.Inf(1)}, {1, 1}} {
		p := pidTuningPlant{modes: []complex128{m}, hasCrossFloor: true, crossLow: b[0], crossHigh: b[1]}
		var grid []float64
		pidWithin(t, 10*time.Second, func() { grid = pidTuningCrossGrid(p, omega) })
		if !slices.Equal(grid, base) {
			t.Errorf("bounds %v: grid %v, want %v", b, grid, base)
		}
	}
}

// tunePIDFloor on sampled data with crossLow and crossHigh left 0 hung in
// pidTuningCrossGrid; it must return as with the bounds at the band.
func TestTunePIDFloorUnsetCrossBand(t *testing.T) {
	num, den := []float64{17.66332276886301}, []float64{1, .4166065677430895}
	frd := pidExactFRD(t, num, den, 1e-3, 1e3, 600)
	o := PIDTuningOptions{PhaseMargin: 45, CrossoverFrequency: 1}
	p := pidTuningFRDPlant(frd, nil)
	p.maxSensitivity = pidTuningMaxSensitivity
	var r *PIDTuningResult
	var err error
	pidWithin(t, 60*time.Second, func() { r, err = tunePIDFloor(context.Background(), "test", p, PidtunePD, o) })
	q := p
	q.crossLow, q.crossHigh = p.low, p.high
	ref, refErr := tunePIDFloor(context.Background(), "ref", q, PidtunePD, o)
	if (err == nil) != (refErr == nil) {
		t.Fatalf("err %v, want %v", err, refErr)
	}
	if err != nil {
		t.Fatalf("no design: %v", err)
	}
	k := *r.Controller
	if k != *ref.Controller {
		t.Fatalf("design %+v, want %+v", k, *ref.Controller)
	}
	if re := pidClosedLoopMaxRe(num, den, k); re >= 0 {
		t.Errorf("%+v: closed-loop max Re %g", k, re)
	}
	if peak := pidExactSensitivityPeak(num, den, k, 1e-3, 1e3); peak > pidTuningMaxSensitivity*1.001 {
		t.Errorf("%+v: exact |S| peak %g", k, peak)
	}
}
