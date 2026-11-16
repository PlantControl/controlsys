package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
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

// FKYKOG: an output cancelling below the GEPP solve's cancellation guard
// (main kept 46 times the term bound, 6e-10 relative, at ω = 3162).
func TestFreqResponseWithinTermsCancellingOutput(t *testing.T) {
	sys := withinTermsModel(t, 4,
		[]float64{0, -1, -2, -1, 0, -1, 0, 4908, 0, 0, -1, 0, 0, 5559, 0, -2.7283573e+07},
		[]float64{1, -2, 1, 11118},
		[]float64{1, -1, -1, 0}, []float64{0}, 0)
	ws := withinTermsGrid(0)
	resp, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	checkWithinTerms(t, sys, ws, resp, 0)
}

// FKYKOG: both delay-free explicit solvers, MIMO, continuous and discrete,
// against the term-bound oracle on the stressed sweep models.
func TestFreqResponseWithinTermsSweepSolvers(t *testing.T) {
	for kind := range sweepModelKinds {
		for _, dt := range []float64{0, 0.1} {
			for _, n := range []int{5, 24} {
				rng := rand.New(rand.NewPCG(uint64(n), uint64(kind)<<1|map[bool]uint64{false: 0, true: 1}[dt > 0]))
				sys, _ := randomSweepRealization(rng, kind, dt, false, n, 2, 2)
				if sys == nil {
					continue
				}
				var ws []float64
				for k, w := range withinTermsGrid(dt) {
					if k%3 == 0 {
						ws = append(ws, w)
					}
				}
				e := newFrequencyEvaluator(sys)
				for _, solver := range []frequencyPointSolver{newBalancedDense(sys, n, 2, 2), newHessenbergSweep(sys, n, 2, 2)} {
					name := fmt.Sprintf("%v/dt=%g/n=%d/%T", kind, dt, n, solver)
					data := make([]complex128, len(ws)*4)
					if err := e.sweepInto(ws, data, solver); err != nil {
						t.Errorf("%s: %v", name, err)
						continue
					}
					t.Run(name, func(t *testing.T) {
						checkWithinTerms(t, sys, ws, newFreqResponseMatrix(data, ws, 2, 2, nil, nil), 0)
					})
				}
			}
		}
	}
}

// FKYKOG: the discrete DC point z = 1 is DCGain's too, at an integrator
// and off one.
func TestFreqResponseAtDCIsDCGainDiscrete(t *testing.T) {
	for _, a22 := range []float64{1, -0.3} {
		sys := withinTermsModel(t, 2, []float64{0.5, 0.25, 0, a22}, []float64{1, 1}, []float64{1, 0.5}, []float64{0.125}, 0.1)
		g, err := sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		want := complex(g.At(0, 0), 0)
		resp, err := sys.FreqResponse([]float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		h, err := sys.EvalFr(1)
		if err != nil {
			t.Fatal(err)
		}
		if got := resp.At(0, 0, 0); got != want || h[0][0] != want {
			t.Errorf("a22=%v: FreqResponse(0) = %v, EvalFr(1) = %v, DCGain %v", a22, got, h[0][0], want)
		}
	}
}

// FKYKOG: an order-22 model beside a DC integrator cluster at ω = 1e-10,
// determined (η = 2.1e-11) but with X far above |c||x̂| (X ≈ 1.5e18 against
// L = 569). Only the approximate-inverse tier's lower bound on X certifies
// it, and that tier took the left residual I − YP of the GEPP inverse,
// which exceeds 1/2 beside the cluster where the right one, I − PY, is
// 4e-11: the point was ErrAccuracyUnverified.
func TestFreqResponseWithinTermsBesideIntegratorCluster(t *testing.T) {
	a := []float64{
		0, -2, -2, 1, 2, -1, 2, 2, -2, 2, -2, -1, 1, -2, 1, 2, -1, 2, -2, 1, 2, 1, 0, -1, 0,
		13, 0, 0, -2, 14, 0, 0, 56, 0, 0, 4, 0, -40, -200, 0, 0, 0, 0, 8, 0, 0, -1, 0, 0, -3,
		-3, -15, -6, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 8, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 4, 0, 0, -1, 7, 0, 0, 28, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0,
		-1, 5, 0, -4, 0, 0, 0, 0, 30, 150, 30, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
		0, 0, 0, -6, -30, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 21, 0, -1, 84, 0, 0, 0, 1, -7,
		-35, 175, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 5, 0, -25, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 200, 1000, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -15, 0, -3, 0, -6, 0, 0, 0, -40, 0, 0, 3,
		-17, 0, 0, -72, 0, 0, -7, 0, 0, 0, 0, 0, 0, 0, -40, 0, 0, 0, 44, 0, 0, 3, -21, 0, 0,
		-84, 0, 0, 0, -1, -224, -1120, -4, 0, 0, 0, 16, 0, 0, 0, -10, 0, 0, -1, 9, 0, 0, 36, 0,
		0, 2, 0, -1, -5, 25, 0, 5, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
		0, 5, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -56, -280, -1, 0, 0, 0, 4,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 5, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, -1,
	}
	b := []float64{
		7, -224, 10, 80, 2, 0, 89, 303, -66, 490, -3, 2009, -15, -173, -2558, 173, -12, -569,
		-4, -8, 5, 2,
	}
	c := []float64{
		1, -1, -1, -1, -1, 1, 0, -1, 1, 1, 0, 0, 1, -1, -1, 0, 0, 0, -1, 1, -1, -1,
	}
	sys := withinTermsModel(t, 22, a, b, c, []float64{0}, 0)
	ws := withinTermsGrid(0)
	resp, err := sys.FreqResponse(ws)
	if err != nil {
		t.Fatal(err)
	}
	checkWithinTerms(t, sys, ws, resp, 0)
}
