package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func conversionDynamicLFT(t *testing.T) *System {
	t.Helper()
	sys, err := makeTestSystem().Discretize(.1)
	if err != nil {
		t.Fatal(err)
	}
	sys.LFT = &LFTDelay{Tau: []float64{2}, B2: mat.NewDense(2, 1, []float64{.2, .3}), C2: mat.NewDense(1, 2, []float64{.1, .2}), D12: mat.NewDense(1, 1, []float64{.5}), D21: mat.NewDense(1, 1, []float64{.3}), D22: mat.NewDense(1, 1, []float64{.2})}
	return sys
}

func TestSeriesInternalDelayIndependentProduct(t *testing.T) {
	bank, err := buildContinuousDelayBank([]float64{.35}, 1, .1, 3)
	if err != nil {
		t.Fatal(err)
	}
	filter := conversionStateAsDelay(bank)
	plant, err := makeTestSystem().Discretize(.1)
	if err != nil {
		t.Fatal(err)
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{2}), .1)
	if err != nil {
		t.Fatal(err)
	}
	delayed := conversionDynamicLFT(t)
	for _, models := range [][2]*System{{filter, gain}, {gain, filter}, {filter, plant}, {plant, filter}, {delayed, filter}, {filter, delayed}, {delayed, delayed}} {
		series, err := Series(models[0], models[1])
		if err != nil {
			t.Fatal(err)
		}
		for _, theta := range []float64{.1, .5, 1.2, 2.2} {
			z := cmplx.Exp(complex(0, theta))
			h1, err := models[0].EvalFr(z)
			if err != nil {
				t.Fatal(err)
			}
			h2, err := models[1].EvalFr(z)
			if err != nil {
				t.Fatal(err)
			}
			assertPrewarpResponse(t, series, z, [][]complex128{{h2[0][0] * h1[0][0]}})
		}
	}
}

func TestPullExternalDelaysPreservesExistingInternalPorts(t *testing.T) {
	for _, delays := range [][2]float64{{2, 0}, {0, 3}, {2, 3}} {
		sys := conversionDynamicLFT(t)
		sys.InputDelay = []float64{delays[0]}
		sys.OutputDelay = []float64{delays[1]}
		pulled, err := sys.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		for _, theta := range []float64{.1, .5, 1.2, 2.2} {
			z := cmplx.Exp(complex(0, theta))
			want, err := sys.EvalFr(z)
			if err != nil {
				t.Fatal(err)
			}
			assertPrewarpResponse(t, pulled, z, want)
		}
	}
}

func TestAbsorbInternalDelaysUsesSamplesAndPreservesExternal(t *testing.T) {
	sys := conversionDynamicLFT(t)
	sys.InputDelay = []float64{1}
	sys.OutputDelay = []float64{3}
	out, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := out.Dims()
	if n != 4 || out.LFT != nil || out.InputDelay[0] != 1 || out.OutputDelay[0] != 3 {
		t.Fatalf("incorrect absorption: n=%d input=%v output=%v LFT=%v", n, out.InputDelay, out.OutputDelay, out.LFT)
	}
	for _, theta := range []float64{.1, .5, 1.2, 2.2} {
		z := cmplx.Exp(complex(0, theta))
		want, err := sys.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		assertPrewarpResponse(t, out, z, want)
	}
}

func TestGainZOHRetainsIntegerPathDelay(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.Delay = mat.NewDense(1, 1, []float64{.3})
	disc, err := sys.DiscretizeZOH(.1)
	if err != nil {
		t.Fatal(err)
	}
	response, err := disc.Simulate(mat.NewDense(1, 7, []float64{1, 1, 1, 1, 1, 1, 1}), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range 7 {
		want := 2.
		if k < 3 {
			want = 0
		}
		if response.Y.At(0, k) != want {
			t.Fatalf("sample %d got %g want %g", k, response.Y.At(0, k), want)
		}
	}
}
