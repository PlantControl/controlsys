package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FreqResponsePointwise guarantees that the value at each omega[k] is
// bit-identical to FreqResponse([]float64{omega[k]}), regardless of sweep
// length. These tests pin that guarantee across the sweep-length regimes
// where FreqResponse itself changes evaluation strategy (the direct
// state-space work limit) and across the structurally distinct paths:
// delay-free, I/O delay, discrete, MIMO, internal delay (LFT), and the
// per-point pole limit when the state-space solve fails.

func pointwiseTestGrid(wMin, wMax float64, n int) []float64 {
	return logspace(math.Log10(wMin), math.Log10(wMax), n)
}

func assertBitIdenticalToSinglePoint(t *testing.T, sys *System, omega []float64) {
	t.Helper()

	got, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("FreqResponsePointwise: %v", err)
	}
	if got.NFreq != len(omega) {
		t.Fatalf("NFreq = %d, want %d", got.NFreq, len(omega))
	}

	for k, w := range omega {
		want, err := sys.FreqResponse([]float64{w})
		if err != nil {
			t.Fatalf("FreqResponse([%v]): %v", w, err)
		}
		for i := range got.P {
			for j := range got.M {
				g := got.At(k, i, j)
				e := want.At(0, i, j)
				if math.Float64bits(real(g)) != math.Float64bits(real(e)) ||
					math.Float64bits(imag(g)) != math.Float64bits(imag(e)) {
					t.Errorf("omega[%d]=%v out=%d in=%d: pointwise %v, single-point %v", k, w, i, j, g, e)
				}
			}
		}
	}
}

func TestFreqResponsePointwise_BitIdenticalAcrossSweepLengths(t *testing.T) {
	firstOrder, err := New(
		mat.NewDense(1, 1, []float64{-0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	secondOrderDelay, err := New(
		mat.NewDense(2, 2, []float64{-0.125, 0, 0.125, -1.0 / 3.0}),
		mat.NewDense(2, 1, []float64{0.1875, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := secondOrderDelay.SetInputDelay([]float64{1.5}); err != nil {
		t.Fatal(err)
	}

	gain, err := NewGain(mat.NewDense(1, 1, []float64{3.5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	systems := []struct {
		name string
		sys  *System
	}{
		{"first_order", firstOrder},
		{"second_order_input_delay", secondOrderDelay},
		{"static_gain", gain},
	}

	// 1 and 40 stay within the direct state-space work limit for these
	// orders; 137 and 500 force FreqResponse onto its transfer-function
	// sweep for any n >= 1.
	for _, tc := range systems {
		for _, nPoints := range []int{1, 40, 137, 500} {
			t.Run(fmt.Sprintf("%s/%d_points", tc.name, nPoints), func(t *testing.T) {
				assertBitIdenticalToSinglePoint(t, tc.sys, pointwiseTestGrid(0.001, 10.0, nPoints))
			})
		}
	}
}

func TestFreqResponsePointwise_Discrete(t *testing.T) {
	sysc, err := New(
		mat.NewDense(2, 2, []float64{-0.125, 0, 0.125, -1.0 / 3.0}),
		mat.NewDense(2, 1, []float64{0.1875, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sysd, err := sysc.Discretize(0.5)
	if err != nil {
		t.Fatal(err)
	}
	assertBitIdenticalToSinglePoint(t, sysd, pointwiseTestGrid(0.001, 2.0, 137))
}

func TestFreqResponsePointwise_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-0.5, 0.1, 0, 0, -0.25, 0.2, 0, 0, -1}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0.5, 0.5}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 1}),
		mat.NewDense(2, 2, []float64{0, 0.1, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{0, 2.0}); err != nil {
		t.Fatal(err)
	}
	assertBitIdenticalToSinglePoint(t, sys, pointwiseTestGrid(0.001, 10.0, 137))
}

func TestFreqResponsePointwise_InternalDelay(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-0.5, 0.1, 0, -0.25}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	B2 := mat.NewDense(2, 1, []float64{0.5, 0.3})
	C2 := mat.NewDense(1, 2, []float64{0.2, 0.4})
	zero := mat.NewDense(1, 1, []float64{0})
	if err := sys.SetInternalDelay([]float64{3}, B2, C2, zero, zero, zero); err != nil {
		t.Fatal(err)
	}
	assertBitIdenticalToSinglePoint(t, sys, pointwiseTestGrid(0.001, 10.0, 137))
}

// A pure integrator makes the state-space pencil singular at omega=0, so
// FreqResponse([]float64{0}) falls back to transfer-function evaluation
// for that point while every other point solves directly. The pointwise
// sweep must reproduce that mixed per-point behavior bit-for-bit.
func TestFreqResponsePointwise_PerPointFallback(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	omega := append([]float64{0}, pointwiseTestGrid(0.001, 10.0, 40)...)
	assertBitIdenticalToSinglePoint(t, sys, omega)
}

// The pointwise sweep reuses one workspace (solve buffers and effective
// delay matrices) across omega points, so allocations per sweep must stay
// constant as the sweep grows: only the result buffer scales with
// len(omega), and that is a single allocation. A per-point allocation
// regression (like rebuilding the delay matrix inside the loop) makes the
// long-sweep count scale with n and fails this test.
func TestFreqResponsePointwise_AllocsAmortizeAcrossSweepLengths(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-0.125, 0, 0.125, -1.0 / 3.0}),
		mat.NewDense(2, 1, []float64{0.1875, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{1.5}); err != nil {
		t.Fatal(err)
	}

	allocsAt := func(n int) float64 {
		omega := pointwiseTestGrid(0.001, 10.0, n)
		return testing.AllocsPerRun(20, func() {
			if _, err := sys.FreqResponsePointwise(omega); err != nil {
				t.Fatal(err)
			}
		})
	}

	short := allocsAt(10)
	long := allocsAt(1000)
	if long > short {
		t.Fatalf("allocations scale with sweep length: %v allocs at 10 points, %v at 1000 (want constant)", short, long)
	}
}

func TestFreqResponsePointwise_EmptyOmega(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := sys.FreqResponsePointwise(nil)
	if err != nil {
		t.Fatalf("FreqResponsePointwise(nil): %v", err)
	}
	if resp != nil {
		t.Fatalf("FreqResponsePointwise(nil) = %v, want nil (matching FreqResponse)", resp)
	}
}
