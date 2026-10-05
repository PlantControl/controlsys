package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func delayTestPoints(dt float64) []complex128 {
	var pts []complex128
	for _, w := range []float64{0.01, 0.3, 1, 2.5, 7} {
		if dt > 0 {
			pts = append(pts, cmplx.Exp(complex(0, w*dt)))
		} else {
			pts = append(pts, complex(0, w))
		}
	}
	return pts
}

// closedLoopOracle evaluates (I - sign*G*K)^{-1} G from the open-loop
// responses of plant and controller.
func closedLoopOracle(t *testing.T, plant, ctrl *System, sign float64, s complex128) [][]complex128 {
	t.Helper()
	f := exactDelayFactor(plant.Dt)
	G := evalDelaySystem(t, plant, s, f)
	K := evalDelaySystem(t, ctrl, s, f)
	GK := cmul(G, K)
	loop := make([][]complex128, len(GK))
	for i := range GK {
		loop[i] = make([]complex128, len(GK))
		for j := range GK {
			loop[i][j] = -complex(sign, 0) * GK[i][j]
		}
		loop[i][i] += 1
	}
	return csolve(t, loop, G)
}

func assertResponseOracle(t *testing.T, label string, got *System, ref func(s complex128) [][]complex128) {
	t.Helper()
	f := exactDelayFactor(got.Dt)
	worst := 0.0
	for _, s := range delayTestPoints(got.Dt) {
		want := ref(s)
		have := evalDelaySystem(t, got, s, f)
		for i := range want {
			for j := range want[i] {
				worst = math.Max(worst, cmplx.Abs(have[i][j]-want[i][j])/(1+cmplx.Abs(want[i][j])))
			}
		}
	}
	if worst > 1e-9 {
		t.Errorf("%s: response mismatch %.3e", label, worst)
	}
}

func feedbackDelayPlant(t *testing.T, dt float64, kind string) *System {
	t.Helper()
	A := []float64{-1, 2, 0.5, -3}
	if dt > 0 {
		A = []float64{0.5, 0.2, -0.1, 0.7}
	}
	P, err := New(mat.NewDense(2, 2, A), mat.NewDense(2, 2, []float64{1, 0.3, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}), mat.NewDense(2, 2, []float64{0.2, 0.1, 0, 0.3}), dt)
	if err != nil {
		t.Fatal(err)
	}
	sc := 1.0
	if dt > 0 {
		sc = 10
	}
	switch kind {
	case "in":
		P.InputDelay = []float64{0.1 * sc, 0.2 * sc}
	case "out":
		P.OutputDelay = []float64{0, 0.3 * sc}
	case "iod":
		P.Delay = mat.NewDense(2, 2, []float64{0.4 * sc, 0, 0, 0})
	case "mixed":
		P.InputDelay = []float64{0.1 * sc, 0.2 * sc}
		P.OutputDelay = []float64{0, 0.3 * sc}
		P.Delay = mat.NewDense(2, 2, []float64{0.4 * sc, 0, 0, 0})
	case "lft":
		P.InputDelay = []float64{0.2 * sc, 0}
		if err := P.SetInternalDelay([]float64{0.7 * sc}, mat.NewDense(2, 1, []float64{1, 0.5}), mat.NewDense(1, 2, []float64{0.3, 1}),
			mat.NewDense(2, 1, []float64{0.2, 0}), mat.NewDense(1, 2, []float64{0, 0.1}), mat.NewDense(1, 1, []float64{0.1})); err != nil {
			t.Fatal(err)
		}
	}
	return P
}

func feedbackDelayController(t *testing.T, dt float64, kind string) *System {
	t.Helper()
	sc := 1.0
	if dt > 0 {
		sc = 10
	}
	a := -2.0
	if dt > 0 {
		a = 0.3
	}
	K, err := New(mat.NewDense(1, 1, []float64{a}), mat.NewDense(1, 2, []float64{1, 0.5}),
		mat.NewDense(2, 1, []float64{0.3, 0.2}), mat.NewDense(2, 2, []float64{0.5, 0, 0.1, 0.4}), dt)
	if err != nil {
		t.Fatal(err)
	}
	switch kind {
	case "in":
		K.InputDelay = []float64{0.1 * sc, 0}
	case "out":
		K.OutputDelay = []float64{0, 0.2 * sc}
	case "iod":
		K.Delay = mat.NewDense(2, 2, []float64{0, 0.3 * sc, 0, 0})
	case "gain-iod":
		K, _ = NewGain(mat.NewDense(2, 2, []float64{0.5, 0, 0.1, 0.4}), dt)
		K.Delay = mat.NewDense(2, 2, []float64{0, 0, 0.3 * sc, 0})
	}
	return K
}

func TestFeedbackDelaysMatchClosedLoopOracle(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for _, pk := range []string{"none", "in", "out", "iod", "mixed", "lft"} {
			for _, kk := range []string{"none", "in", "out", "iod", "gain-iod"} {
				if pk == "none" && kk == "none" {
					continue
				}
				for _, sign := range []float64{-1, 1} {
					P := feedbackDelayPlant(t, dt, pk)
					K := feedbackDelayController(t, dt, kk)
					label := fmt.Sprintf("dt=%v P=%s K=%s sign=%v", dt, pk, kk, sign)
					cl, err := Feedback(P, K, sign)
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					if cl.InputDelay != nil || cl.OutputDelay != nil || cl.Delay != nil {
						t.Errorf("%s: loop delays left external: in=%v out=%v io=%v", label, cl.InputDelay, cl.OutputDelay, cl.Delay)
					}
					assertResponseOracle(t, label, cl, func(s complex128) [][]complex128 {
						return closedLoopOracle(t, P, K, sign, s)
					})
				}
			}
		}
	}
}

func TestFeedbackInputDelayInsideLoopStep(t *testing.T) {
	plant, _ := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	plant.InputDelay = []float64{3}
	k, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)
	cl, err := Feedback(plant, k, -1)
	if err != nil {
		t.Fatal(err)
	}
	const N = 20
	u := mat.NewDense(1, N, nil)
	for i := range N {
		u.Set(0, i, 1)
	}
	r, err := cl.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	x := 0.0
	v := make([]float64, N)
	for i := range N {
		y := x
		if d := math.Abs(r.Y.At(0, i) - y); d > 1e-12 {
			t.Fatalf("k=%d: y=%v, want %v", i, r.Y.At(0, i), y)
		}
		v[i] = 1 - 0.8*y
		in := 0.0
		if i >= 3 {
			in = v[i-3]
		}
		x = 0.5*x + in
	}
}

func TestFeedbackDelayedStaticGain(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		g, _ := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), dt)
		g.Delay = mat.NewDense(2, 2, []float64{0.4, 0, 0, 0})
		if dt > 0 {
			g.Delay.Set(0, 0, 4)
		}
		for _, K := range []*System{nil, feedbackDelayController(t, dt, "in")} {
			cl, err := Feedback(g, K, -1)
			if err != nil {
				t.Fatalf("dt=%v: %v", dt, err)
			}
			ref := K
			if ref == nil {
				ref, _ = NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), dt)
			}
			assertResponseOracle(t, fmt.Sprintf("dt=%v static K=%v", dt, K != nil), cl, func(s complex128) [][]complex128 {
				return closedLoopOracle(t, g, ref, -1, s)
			})
		}
	}
}

func TestFeedbackDelayedKeepsStateNames(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		P := feedbackChannelPlant(t, dt, true)
		cl, err := Feedback(P, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if want := []string{"p1", "p2", "p3"}; !slices.Equal(cl.StateName, want) {
			t.Errorf("dt=%v: StateName = %v, want %v", dt, cl.StateName, want)
		}
	}
}
