package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// withinTermsGrid is the half-decade grid 1e-18..1e4, cut below ωDt = 3
// for a discrete model.
func withinTermsGrid(dt float64) []float64 {
	var ws []float64
	for k := -36; k <= 8; k++ {
		w := math.Pow(10, float64(k)/2)
		if dt > 0 && w*dt >= 3 {
			continue
		}
		ws = append(ws, w)
	}
	return ws
}

func withinTermsModel(t *testing.T, n int, a, b, c, d []float64, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, 1, 1, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// FKYKOG: a triangular model with a pole near DC, where the refined double
// solve kept up to 30 times its term bound (relative 1e-13 against 7e-15).
func TestFreqResponseWithinTermsNearDCTriangle(t *testing.T) {
	sys := withinTermsModel(t, 4,
		[]float64{-8.003552344283472e-05, 170.47472752689742, 0, 399.91461726773656, 0, -0.0007860929578062323, 0, 0, -102.38717607474956, 35.779752835667026, -6.9902425608346695e-06, -0.0012280356372853872, 0, -0.03383311775335814, 0, -3.019212338553195e-05},
		[]float64{-0.1907081693174828, -1.2510480777931423, -0.7422581206800616, 0.42872310360926436},
		[]float64{0.37340305214508085, 0.9295613010877934, 1.797540287582489, 0.07324152102881243},
		[]float64{0.594834957573056}, 0)
	ws := withinTermsGrid(0)
	for _, pointwise := range []bool{false, true} {
		resp, err := sys.FreqResponse(ws)
		if pointwise {
			resp, err = sys.FreqResponsePointwise(ws)
		}
		if err != nil {
			t.Fatal(err)
		}
		checkWithinTerms(t, sys, ws, resp, 1e-8)
	}
}

// FKYKOG: an integrator with unit gains one ulp below 1, where the
// response's own rounding takes the whole term bound of a one-state model.
func TestFreqResponseWithinTermsIntegratorOrderOne(t *testing.T) {
	sys := withinTermsModel(t, 1, []float64{0}, []float64{0.9999999999999999}, []float64{0.9999999999999999}, []float64{3.469446951953614e-18}, 0)
	ws := withinTermsGrid(0)
	resp, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	checkWithinTerms(t, sys, ws, resp, 1e-8)
}

// FKYKOG: discrete models with poles in their own rounding disk of z = 1.
// Off DC no point is a pole of the stored model: main returned +Inf, or
// failed with errPoleUnresolved between finite neighbours.
func TestFreqResponseWithinTermsBesideDCCluster(t *testing.T) {
	for _, tc := range []struct {
		name       string
		n          int
		a, b, c, d []float64
	}{
		{"pcg", 4,
			[]float64{0, -0.06519722661984333, 0, 0, 1, 0.5676527376498068, 0, 0, 0, 1, 0, -1, 0, 0, 1, 2},
			[]float64{0.0429818642986183, -0.06585876444955387, 0.04597960716697493, 0},
			[]float64{0, 0, 0, 1}, []float64{0}},
		{"grid", 3,
			[]float64{0, -0.010000000000000002, 0, 1, 0.2, 0, 0, 1, 1},
			[]float64{0.99, -1.8, 1},
			[]float64{0, 0, 1}, []float64{0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := withinTermsModel(t, tc.n, tc.a, tc.b, tc.c, tc.d, 0.1)
			ws := withinTermsGrid(0.1)
			resp, err := sys.FreqResponse(ws)
			if err != nil {
				t.Fatal(err)
			}
			checkWithinTerms(t, sys, ws, resp, 0)
			for k, w := range ws {
				one, err := sys.FreqResponse([]float64{w})
				if err != nil {
					t.Fatalf("ω=%v: %v", w, err)
				}
				if got, want := one.At(0, 0, 0), resp.At(k, 0, 0); cmplx.Abs(got-want) > 1e-12*cmplx.Abs(want) {
					t.Errorf("ω=%v: single point %v, sweep %v", w, got, want)
				}
			}
		})
	}
}

// FKYKOG: FreqResponse at DC is DCGain's gain, bit for bit, where the
// double solve there returned a value 4e-11 relative off.
func TestFreqResponseAtDCIsDCGain(t *testing.T) {
	sys := withinTermsModel(t, 3,
		[]float64{-0.041582607469972409, 0, 0, 1, 0, 0, 0, -0.64790650386012283, -0.64790650386012283},
		[]float64{0.24406981298565866, 0, 0},
		[]float64{0, 1, 1}, []float64{0}, 0)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	resp, err := sys.FreqResponse([]float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resp.At(0, 0, 0), complex(g.At(0, 0), 0); got != want {
		t.Errorf("FreqResponse(0) = %v, DCGain %v", got, want)
	}
	h, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	if h[0][0] != complex(g.At(0, 0), 0) {
		t.Errorf("EvalFr(0) = %v, DCGain %v", h[0][0], g.At(0, 0))
	}
}
