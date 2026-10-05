package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const poleTol = 1e-9

func poleSys(t *testing.T, n, m, p int, a, b, c, d []float64, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// poleDescriptorTwin returns (M·A, M·B, C, D, E=M), which has the same
// transfer matrix as sys. M is upper bidiagonal with dyadic entries so M·A is
// exact and a pole of sys stays an exact pole of the pencil.
func poleDescriptorTwin(t *testing.T, sys *System) *System {
	t.Helper()
	n, _, _ := sys.Dims()
	M := mat.NewDense(n, n, nil)
	for i := range n {
		M.Set(i, i, 2)
		if i+1 < n {
			M.Set(i, i+1, 0.5)
		}
	}
	var A, B mat.Dense
	A.Mul(M, sys.A)
	B.Mul(M, sys.B)
	desc, err := NewDescriptor(&A, &B, sys.C, sys.D, M, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return desc
}

func checkPoleEntry(t *testing.T, label string, got, want complex128) {
	t.Helper()
	if cmplx.IsInf(want) {
		if !cmplx.IsInf(got) {
			t.Errorf("%s: got %v, want Inf", label, got)
		}
		return
	}
	if cmplx.IsNaN(got) || cmplx.Abs(got-want) > poleTol*max(1, cmplx.Abs(want)) {
		t.Errorf("%s: got %v, want %v", label, got, want)
	}
}

// checkPoleResponse checks FreqResponse, FreqResponsePointwise and EvalFr
// against the closed form at every omega.
func checkPoleResponse(t *testing.T, label string, sys *System, omega []float64, oracle func(s complex128) [][]complex128) {
	t.Helper()
	td := newTimeDomain(sys.Dt)
	fr, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s FreqResponse: %v", label, err)
	}
	pw, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("%s FreqResponsePointwise: %v", label, err)
	}
	for k, w := range omega {
		s := td.frequencyVariable(w)
		want := oracle(s)
		got, err := sys.EvalFr(s)
		if err != nil {
			t.Fatalf("%s EvalFr(%v): %v", label, s, err)
		}
		for i := range want {
			for j := range want[i] {
				checkPoleEntry(t, label+" FreqResponse", fr.At(k, i, j), want[i][j])
				checkPoleEntry(t, label+" FreqResponsePointwise", pw.At(k, i, j), want[i][j])
				checkPoleEntry(t, label+" EvalFr", got[i][j], want[i][j])
			}
		}
	}
}

var poleInf = cmplx.Inf()

func isPole(s, p complex128) bool { return cmplx.Abs(s-p) < 1e-12 }

// poleOscLag2x2 is blockdiag(1/(s²+1), 2/((s+1)(s+3)+1)) with D = [.5 0; .2 0];
// the discrete version is blockdiag(1/(z²+1), 0.2/((z-.5)(z-.3)+.02)).
func poleOscLag2x2(t *testing.T, dt float64) (*System, func(s complex128) [][]complex128) {
	lag := []float64{-1, 2, -0.5, -3}
	if dt != 0 {
		lag = []float64{0.5, 0.2, -0.1, 0.3}
	}
	sys := poleSys(t, 4, 2, 2,
		[]float64{0, 1, 0, 0, -1, 0, 0, 0, 0, 0, lag[0], lag[1], 0, 0, lag[2], lag[3]},
		[]float64{0, 0, 1, 0, 0, 0, 0, 1},
		[]float64{1, 0, 0, 0, 0, 0, 1, 0},
		[]float64{0.5, 0, 0.2, 0}, dt)
	return sys, func(s complex128) [][]complex128 {
		g11 := poleInf
		if !isPole(s, 1i) && !isPole(s, -1i) {
			g11 = 1/(s*s+1) + 0.5
		}
		g22 := 2 / ((s+1)*(s+3) + 1)
		if dt != 0 {
			g22 = 0.2 / ((s-0.5)*(s-0.3) + 0.02)
		}
		return [][]complex128{{g11, 0}, {0.2, g22}}
	}
}

func TestFreqResponseAtPoleExplicitAndDescriptor(t *testing.T) {
	cont, contOracle := poleOscLag2x2(t, 0)
	const dt = 0.1
	disc, discOracle := poleOscLag2x2(t, dt)
	contW := []float64{0.5, 1, 2}
	discW := []float64{1, math.Pi / 2 / dt, 25}
	checkPoleResponse(t, "explicit continuous", cont, contW, contOracle)
	checkPoleResponse(t, "descriptor continuous", poleDescriptorTwin(t, cont), contW, contOracle)
	checkPoleResponse(t, "explicit discrete", disc, discW, discOracle)
	arr, err := NewModelArray([]int{2}, []*System{cont, poleDescriptorTwin(t, cont)})
	if err != nil {
		t.Fatal(err)
	}
	ar, err := arr.FreqResponse(contW)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range ar.Responses {
		for k, w := range contW {
			want := contOracle(complex(0, w))
			for i := range want {
				for j := range want[i] {
					checkPoleEntry(t, "ModelArray", r.At(k, i, j), want[i][j])
				}
			}
		}
	}
	checkPoleResponse(t, "descriptor discrete", poleDescriptorTwin(t, disc), discW, discOracle)
}

// TestFreqResponseAtPoleHessenbergSweep drives the pole through the
// Hessenberg sweep (n > 8m+8, nw > 2, A not upper Hessenberg): outputs
// 1/(s²+1) and 1/(s+1)^8 + 1/(s+1)^7 + 0.3 from two interleaved lag chains
// z_k' = -z_k + z_{k-2}.
func TestFreqResponseAtPoleHessenbergSweep(t *testing.T) {
	const chain = 15
	n := 2 + chain
	a := make([]float64, n*n)
	b := make([]float64, n)
	c := make([]float64, 2*n)
	a[1], a[n] = 1, -1
	b[1], b[2], b[3] = 1, 1, 1
	for k := 2; k < n; k++ {
		a[k*n+k] = -1
		if k > 3 {
			a[k*n+k-2] = 1
		}
	}
	c[0], c[2*n-1], c[2*n-2] = 1, 1, 1
	sys := poleSys(t, n, 1, 2, a, b, c, []float64{0, 0.3}, 0)
	if newFrequencyEvaluator(sys).useDenseSweep(4) {
		t.Fatal("expected the Hessenberg sweep")
	}
	oracle := func(s complex128) [][]complex128 {
		g1 := poleInf
		if !isPole(s, 1i) {
			g1 = 1 / (s*s + 1)
		}
		lag := 1 / (s + 1)
		return [][]complex128{{g1}, {cmplx.Pow(lag, 8) + cmplx.Pow(lag, 7) + 0.3}}
	}
	checkPoleResponse(t, "hessenberg", sys, []float64{0.5, 1, 2, 3}, oracle)
}

func TestFreqResponseAtPoleIntegratorMIMO(t *testing.T) {
	// [1/s 1/(s+2); 0 1/(s+2)] + D, continuous and the z = 1 discrete analogue.
	oracle := func(dt float64) func(s complex128) [][]complex128 {
		return func(s complex128) [][]complex128 {
			pole, fast := complex(0, 0), s+2
			if dt != 0 {
				pole, fast = 1, s-0.4
			}
			g11 := poleInf
			if !isPole(s, pole) {
				g11 = 1 / (s - pole)
			}
			return [][]complex128{{g11, 1/fast + 0.1}, {0, 1 / fast}}
		}
	}
	for _, dt := range []float64{0, 0.2} {
		a := []float64{0, 0, 0, -2}
		if dt != 0 {
			a = []float64{1, 0, 0, 0.4}
		}
		sys := poleSys(t, 2, 2, 2, a, []float64{1, 0, 0, 1}, []float64{1, 1, 0, 1}, []float64{0, 0.1, 0, 0}, dt)
		w := []float64{0, 0.7}
		checkPoleResponse(t, "integrator explicit", sys, w, oracle(dt))
		desc := poleDescriptorTwin(t, sys)
		checkPoleResponse(t, "integrator descriptor", desc, w, oracle(dt))
		dcPoint := newTimeDomain(dt).frequencyVariable(0)
		want := oracle(dt)(dcPoint)
		for _, s := range []*System{sys, desc} {
			dc, err := s.DCGain()
			if err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				for j := range 2 {
					checkPoleEntry(t, "integrator DCGain", complex(dc.At(i, j), 0), want[i][j])
				}
			}
		}
	}
}

// poleDelayLoop is x' = A x + B u - b2·w, w = c2·x(t-τ), y = C x + D u.
func poleDelayLoop(t *testing.T, sys *System, b2, c2 []float64, tau float64) *System {
	t.Helper()
	n, m, p := sys.Dims()
	if err := sys.SetInternalDelay([]float64{tau},
		mat.NewDense(n, 1, b2), mat.NewDense(1, n, c2),
		mat.NewDense(p, 1, nil), mat.NewDense(1, m, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestFreqResponseAtPoleInternalDelay(t *testing.T) {
	const tau = 0.3
	delay := func(s complex128) complex128 { return cmplx.Exp(-s * tau) }

	// Process Lab repro: Lag inside the delay loop, 1/(s²+1) outside it.
	repro := poleDelayLoop(t, poleSys(t, 3, 1, 2,
		[]float64{-1, 0, 0, 0, 0, 1, 1, -1, 0}, []float64{1, 0, 0},
		[]float64{1, 0, 0, 0, 1, 0}, []float64{0.1, 0}, 0),
		[]float64{-1, 0, 0}, []float64{1, 0, 0}, tau)
	checkPoleResponse(t, "repro", repro, []float64{0.5, 1, 2}, func(s complex128) [][]complex128 {
		l := 1 / (s + 1 + delay(s))
		g2 := poleInf
		if !isPole(s, 1i) {
			g2 = l / (s*s + 1)
		}
		return [][]complex128{{l + 0.1}, {g2}}
	})

	// Integrator inside the loop: H is singular at s = 0, G is not.
	intLoop := poleDelayLoop(t, poleSys(t, 1, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 0),
		[]float64{-1}, []float64{1}, tau)
	checkPoleResponse(t, "integrator in loop", intLoop, []float64{0, 1}, func(s complex128) [][]complex128 {
		return [][]complex128{{1 / (s + delay(s))}}
	})

	// Oscillator inside the loop: G(j) = 1/e^{-jτ}.
	oscLoop := poleDelayLoop(t, poleSys(t, 2, 1, 1, []float64{0, 1, -1, 0}, []float64{0, 1}, []float64{1, 0}, []float64{0}, 0),
		[]float64{0, -1}, []float64{1, 0}, tau)
	checkPoleResponse(t, "oscillator in loop", oscLoop, []float64{0.5, 1, 2}, func(s complex128) [][]complex128 {
		return [][]complex128{{1 / (s*s + 1 + delay(s))}}
	})

	// Integrator outside the loop: a true pole of G at s = 0.
	outside, err := Series(intLoop, poleSys(t, 1, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 0))
	if err != nil {
		t.Fatal(err)
	}
	checkPoleResponse(t, "integrator outside loop", outside, []float64{0, 1}, func(s complex128) [][]complex128 {
		if s == 0 {
			return [][]complex128{{poleInf}}
		}
		return [][]complex128{{1 / (s * (s + delay(s)))}}
	})

	if dc, err := intLoop.DCGain(); err != nil || math.Abs(dc.At(0, 0)-1) > poleTol {
		t.Errorf("integrator in loop DCGain = %v, %v; want 1", dc, err)
	}
	if dc, err := outside.DCGain(); err != nil || !math.IsInf(dc.At(0, 0), 1) {
		t.Errorf("integrator outside loop DCGain = %v, %v; want +Inf", dc, err)
	}
}

func TestFreqResponseAtPoleDiscreteInternalDelay(t *testing.T) {
	const dt, d = 0.1, 2
	loop := poleDelayLoop(t, poleSys(t, 1, 1, 1, []float64{1}, []float64{1}, []float64{1}, []float64{0}, dt),
		[]float64{-1}, []float64{1}, d)
	loopTF := func(z complex128) complex128 { return 1 / (z - 1 + 1/(z*z)) }
	checkPoleResponse(t, "discrete integrator in loop", loop, []float64{0, 3}, func(z complex128) [][]complex128 {
		return [][]complex128{{loopTF(z)}}
	})
	outside, err := Series(loop, poleSys(t, 1, 1, 1, []float64{1}, []float64{1}, []float64{1}, []float64{0}, dt))
	if err != nil {
		t.Fatal(err)
	}
	checkPoleResponse(t, "discrete integrator outside loop", outside, []float64{0, 3}, func(z complex128) [][]complex128 {
		if isPole(z, 1) {
			return [][]complex128{{poleInf}}
		}
		return [][]complex128{{loopTF(z) / (z - 1)}}
	})
	if dc, err := loop.DCGain(); err != nil || math.Abs(dc.At(0, 0)-1) > poleTol {
		t.Errorf("DCGain = %v, %v; want 1", dc, err)
	}
	if dc, err := outside.DCGain(); err != nil || !math.IsInf(dc.At(0, 0), 1) {
		t.Errorf("DCGain = %v, %v; want +Inf", dc, err)
	}
}

// TestFreqResponseAtPoleDoubleIntegrator covers a defective (Jordan) pole:
// [1/s² 1/(s+2); 0 1/(s+2)] and its z = 1 discrete analogue.
func TestFreqResponseAtPoleDoubleIntegrator(t *testing.T) {
	for _, dt := range []float64{0, 0.2} {
		a := []float64{0, 1, 0, 0, 0, 0, 0, 0, -2}
		pole, fastPole := complex(0, 0), complex(-2, 0)
		if dt != 0 {
			a = []float64{1, 1, 0, 0, 1, 0, 0, 0, 0.4}
			pole, fastPole = 1, 0.4
		}
		sys := poleSys(t, 3, 2, 2, a, []float64{0, 0, 1, 0, 0, 1}, []float64{1, 0, 1, 0, 0, 1}, []float64{0, 0, 0, 0.1}, dt)
		oracle := func(s complex128) [][]complex128 {
			g11 := poleInf
			if !isPole(s, pole) {
				g11 = 1 / ((s - pole) * (s - pole))
			}
			fast := 1 / (s - fastPole)
			return [][]complex128{{g11, fast}, {0, fast + 0.1}}
		}
		w := []float64{0, 0.7}
		checkPoleResponse(t, "double integrator", sys, w, oracle)
		checkPoleResponse(t, "double integrator descriptor", poleDescriptorTwin(t, sys), w, oracle)
	}
}

// TestFreqResponseAtPoleAlgebraicDelayLoop: w = Δ(z) with z = w + u, y = w,
// so G = Δ/(1-Δ) has poles at ω = 2πk/τ where I - D22·Δ is singular.
func TestFreqResponseAtPoleAlgebraicDelayLoop(t *testing.T) {
	sys := poleSys(t, 1, 1, 1, []float64{-1}, []float64{0}, []float64{0}, []float64{0}, 0)
	if err := sys.SetInternalDelay([]float64{1}, mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1})); err != nil {
		t.Fatal(err)
	}
	checkPoleResponse(t, "algebraic delay loop", sys, []float64{0, math.Pi}, func(s complex128) [][]complex128 {
		dl := cmplx.Exp(-s)
		if cmplx.Abs(1-dl) < 1e-12 {
			return [][]complex128{{poleInf}}
		}
		return [][]complex128{{dl / (1 - dl)}}
	})
}

func TestFreqResponseSingularEverywhereErrors(t *testing.T) {
	sys, err := NewDescriptor(mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sys.FreqResponse([]float64{1}); !errors.Is(err, ErrSingularTransform) {
		t.Errorf("FreqResponse err = %v, want ErrSingularTransform", err)
	}
	if _, err := sys.EvalFr(1i); !errors.Is(err, ErrSingularTransform) {
		t.Errorf("EvalFr err = %v, want ErrSingularTransform", err)
	}
}

func TestSigmaAndGainGoalsAtPole(t *testing.T) {
	sys, _ := poleOscLag2x2(t, 0)
	if _, err := sys.Sigma([]float64{1}, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("MIMO Sigma at pole err = %v, want ErrInvalidArgument (σ₂ undetermined)", err)
	}
	three, err := Append(sys, poleSys(t, 1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0))
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []*System{sys, three} {
		goal, err := NewTuningGoal(TuningGoalSpec{Name: "gain", Type: TuningGoalWeightedGain, Max: 10, Omega: []float64{0.5, 1}})
		if err != nil {
			t.Fatal(err)
		}
		res, err := goal.Evaluate(s)
		if err != nil {
			t.Fatal(err)
		}
		if !math.IsInf(res.Value, 1) || res.Pass {
			t.Errorf("max gain over a pole = %+v, want +Inf failing", res)
		}
	}
	intLoop := poleDelayLoop(t, poleSys(t, 1, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 0),
		[]float64{-1}, []float64{1}, 0.3)
	integrating, err := Series(intLoop, poleSys(t, 1, 1, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 0))
	if err != nil {
		t.Fatal(err)
	}
	res, err := mustOK(NewTrackingGoal("track", 0.1)).Evaluate(integrating)
	if err != nil || !math.IsInf(res.Value, 1) {
		t.Errorf("tracking of integrating delayed model = %+v, %v; want +Inf", res, err)
	}
	if bw, err := Bandwidth(integrating, 0); err != nil || bw != 0 {
		t.Errorf("Bandwidth of integrating delayed model = %v, %v; want 0 (infinite DC gain)", bw, err)
	}
}
