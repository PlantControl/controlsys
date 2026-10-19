package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
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
	for idx := range arr.Len() {
		r, err := ar.ResponseFlat(idx)
		if err != nil {
			t.Fatal(err)
		}
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
// Hessenberg sweep (n = 17, m = 1, 104 points, A not upper Hessenberg): outputs
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
	omega := append([]float64{0.5, 1, 2, 3}, logspace(-1, 1, 100)...)
	if newFrequencyEvaluator(sys).useDenseSweep(len(omega)) {
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
	checkPoleResponse(t, "hessenberg", sys, omega, oracle)
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

	// Oscillator inside the loop: H is singular at z = ±j, an exact
	// unit-circle point away from z = 1, where G(j) = 1/(j²+1+j⁻²) = −1.
	osc := poleSys(t, 2, 1, 1, []float64{0, 1, -1, 0}, []float64{0, 1}, []float64{1, 0}, []float64{0}, dt)
	oscTF := func(z complex128) [][]complex128 { return [][]complex128{{1 / (z*z + 1 + 1/(z*z))}} }
	oscW := []float64{1, math.Pi / 2 / dt}
	checkPoleResponse(t, "discrete oscillator in loop", poleDelayLoop(t, osc.Copy(), []float64{0, -1}, []float64{1, 0}, d), oscW, oscTF)
	checkPoleResponse(t, "discrete descriptor oscillator in loop",
		poleDelayLoop(t, poleDescriptorTwin(t, osc), []float64{-0.5, -2}, []float64{1, 0}, d), oscW, oscTF)
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
		s.InputName, s.OutputName = nil, nil
		_, m, p := s.Dims()
		if err := s.SetInputName(expandName("r", m)...); err != nil {
			t.Fatal(err)
		}
		if err := s.SetOutputName(expandName("y", p)...); err != nil {
			t.Fatal(err)
		}
		goal, err := mustOK(NewGainGoal("r", "y", 10)).WithFocus(0.5, 1)
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
	integrating.InputName, integrating.OutputName = []string{"r"}, []string{"y"}
	res, err := mustOK(NewTrackingGoal("r", "y", 1, 0.1, 1)).Evaluate(integrating)
	if err != nil || res.Pass || !(res.Value > 10) {
		t.Errorf("tracking of integrating delayed model = %+v, %v; want large failing value", res, err)
	}
	if bw, err := Bandwidth(integrating, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Bandwidth of integrating delayed model = %v, %v; want ErrInvalidArgument (infinite DC gain)", bw, err)
	}
}

func poleLimitAt(sys *System, w float64) (h complex128, raw, err error) {
	eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
	pt := newTimeDomain(sys.Dt).frequencyPoint(w)
	dst := make([]complex128, 1)
	raw = eval(pt, dst)
	err = evalWithPoleLimit(eval, sys, pt, dst)
	return dst[0], raw, err
}

// ergoOffOriginLoop is the ERG6ZB loop: H22 of Feedback(P, nil, −1) around
// P = num/den·e^{−τs} has the poles {0, 0, 1.34e−7, −0.464} of den.
var ergoOffOriginLoop = quasiLoop{
	num: []float64{1.9188317375876882, 1.5083785090172463, 0.32156836056367194, 0.015658860719559062},
	den: []float64{1, 0.46404121934763415, -6.209067747296065e-08, 0, 0},
	tau: 0.001181949427158492,
}

// TestPoleLimitOffOriginCluster (ERG6ZB): at ω ≈ 4e−7 the double solve of
// H22 is singular and the pole-limit samples, a step 4.6e−6 off, saw the
// origin cluster as a pole at s and returned +Inf, nil, although |s−p| ≥
// |s|/3 for every pole. Off a pole the response is the dd one, for the
// singular-E twin too, so h22Det matches 1+L(jω) of the polynomials. The
// gate of the delay loop's walk still decides: where it denies dd the
// solve error stands. Oracle: the stored realization solved in big.Float.
func TestPoleLimitOffOriginCluster(t *testing.T) {
	q := ergoOffOriginLoop
	p := makePlant(t, q.num, q.den)
	if err := p.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(p, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	e, err := newDelayLFT(cl)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := e.h22sys.Dims()
	a := mat.NewDense(n+1, n+1, nil)
	a.Slice(0, n, 0, n).(*mat.Dense).Copy(e.h22sys.A)
	a.Set(n, n, 1)
	ed := mat.NewDense(n+1, n+1, nil)
	b := mat.NewDense(n+1, 1, nil)
	c := mat.NewDense(1, n+1, nil)
	for i := range n {
		ed.Set(i, i, 1)
		b.Set(i, 0, e.h22sys.B.At(i, 0))
		c.Set(0, i, e.h22sys.C.At(0, i))
	}
	infTwin, err := NewDescriptor(a, b, c, e.h22sys.D, ed, 0)
	if err != nil {
		t.Fatal(err)
	}
	ws := []float64{2e-8, 2e-7, 3.93e-7, 3.9339e-7, 4e-7, 1e-6}
	for _, w := range ws {
		for _, sys := range []*System{e.h22sys, infTwin} {
			h, raw, err := poleLimitAt(sys, w)
			if raw == nil {
				t.Fatalf("ω=%g: double solve succeeded; case no longer exercises the pole limit", w)
			}
			want := frExactResponse(sys, frExactPoint(w, 0))[0]
			if err != nil || cmplx.Abs(h-want) > 1e-12*cmplx.Abs(want) {
				t.Errorf("ω=%g, descriptor %v: evalWithPoleLimit = %v, %v; want %v", w, sys.E != nil, h, err, want)
			}
			fr, err := sys.FreqResponse([]float64{w})
			if err != nil || cmplx.Abs(fr.Data[0]-want) > 1e-12*cmplx.Abs(want) {
				t.Errorf("ω=%g, descriptor %v: FreqResponse = %v, %v; want %v", w, sys.E != nil, fr, err, want)
			}
		}
		got, ok := e.h22Det(w)
		if want := 1 + q.L(w); !ok || cmplx.Abs(got-want) > 1e-9*cmplx.Abs(want) {
			t.Errorf("ω=%g: h22Det = %v, %v; want %v", w, got, ok, want)
		}
	}
	eval := newFrequencyEvaluator(e.h22sys).pointSolver(math.MaxInt).evalInto
	pt := newTimeDomain(0).frequencyPoint(3.93e-7)
	dst := make([]complex128, 1)
	deny := func(float64) bool { return false }
	if err := evalWithPoleLimitGated(eval, e.h22sys, pt, dst, ddGate{allow: deny, w: 3.93e-7}); !errors.Is(err, ErrSingularTransform) {
		t.Errorf("dd denied: evalWithPoleLimitGated = %v, %v; want ErrSingularTransform", dst[0], err)
	}
	l := e.loop(false)
	for _, w := range ws {
		if !l.ddAllowed(w) {
			t.Fatalf("ddAllowed(%g) = false; nearest pole is the boundary integrator", w)
		}
		got, ok := e.h22Det(w)
		if want := 1 + q.L(w); !ok || cmplx.Abs(got-want) > 1e-9*cmplx.Abs(want) {
			t.Errorf("gated ω=%g: h22Det = %v, %v; want %v", w, got, ok, want)
		}
	}
	if st, err := cl.IsStable(); err != nil || !st {
		t.Errorf("IsStable = %v, %v; want true", st, err)
	}
}

// TestPoleLimitOffOriginClusterMIMO (ERG6ZB): a 1×2 model with the cluster of
// TestPoleLimitOffOriginCluster in both channels got +Inf, nil away from any
// pole; the dd fallback covers every channel. Oracle: big.Float solve of the
// stored realization.
func TestPoleLimitOffOriginClusterMIMO(t *testing.T) {
	q := ergoOffOriginLoop
	half := make([]float64, len(q.num))
	for i, v := range q.num {
		half[i] = 0.5 * v
	}
	r, err := (&TransferFunc{Num: [][][]float64{{q.num, half}}, Den: [][]float64{q.den}}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := r.Sys
	ws := []float64{3.93e-7, 3.9339e-7, 4e-7}
	check := func(api string, w float64, got []complex128) {
		t.Helper()
		want := frExactResponse(sys, frExactPoint(w, 0))
		for j := range want {
			if cmplx.Abs(got[j]-want[j]) > 1e-12*cmplx.Abs(want[j]) {
				t.Errorf("%s(%g)[%d] = %v; want %v", api, w, j, got[j], want[j])
			}
		}
	}
	for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
		"FreqResponse":          sys.FreqResponse,
		"FreqResponsePointwise": sys.FreqResponsePointwise,
	} {
		fr, err := f(ws)
		if err != nil {
			t.Fatalf("%s: %v", api, err)
		}
		for k, w := range ws {
			check(api, w, []complex128{fr.At(k, 0, 0), fr.At(k, 0, 1)})
		}
	}
	for _, w := range ws {
		h, err := sys.EvalFr(complex(0, w))
		if err != nil {
			t.Fatalf("EvalFr(%g): %v", w, err)
		}
		check("EvalFr", w, h[0])
	}
}

// TestPoleLimitOriginClusterNoWindow (ERG6ZB): s(s²+2.31s−2.5e−11) has poles
// {0, 1.1e−11, −2.31}. An |s| ≤ 1e−9·scale origin window took the cluster for
// a pole at s, leaving +Inf where the response is finite, and AllMargin of
// the delayed loop read the Inf-to-finite jump at its edge as a phase
// crossover. On the axis no pole is within 1e−3 relative of jω > 0, so every
// such ω the dd solve resolves is finite, and the margins are those of
// L = num/den·e^{−τs}.
func TestPoleLimitOriginClusterNoWindow(t *testing.T) {
	q := quasiLoop{
		num: []float64{0.759396090035918, 0.10333370680260914},
		den: []float64{1, 2.3119684689572186, -2.533779851794843e-11, 0},
		tau: 0.0004936910785846063,
	}
	sys := makePlant(t, q.num, q.den)
	edge := 1e-9 * 2.3119684689681783
	ws := []float64{6.6e-12, 1.0255865427292543e-10, 1.8e-10, 2.3e-9, edge * (1 - 1e-12), edge, edge * (1 + 1e-12)}
	for e := -18.0; e <= -5; e += 0.05 {
		ws = append(ws, math.Pow(10, e))
	}
	if _, raw, _ := poleLimitAt(sys, 1.0255865427292543e-10); raw == nil {
		t.Fatal("double solve succeeded; case no longer exercises the pole limit")
	}
	closeTo := func(got, want complex128) bool { return cmplx.Abs(got-want) <= 1e-10*cmplx.Abs(want) }
	for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
		"FreqResponse":          sys.FreqResponse,
		"FreqResponsePointwise": sys.FreqResponsePointwise,
	} {
		fr, err := f(ws)
		if err != nil {
			t.Fatalf("%s: %v", api, err)
		}
		for k, w := range ws {
			if want := frExactResponse(sys, frExactPoint(w, 0))[0]; !closeTo(fr.At(k, 0, 0), want) {
				t.Errorf("%s(%.17g) = %v; want %v", api, w, fr.At(k, 0, 0), want)
			}
		}
	}
	for _, w := range ws[:7] {
		h, err := sys.EvalFr(complex(0, w))
		if want := frExactResponse(sys, frExactPoint(w, 0))[0]; err != nil || !closeTo(h[0][0], want) {
			t.Errorf("EvalFr(j%.17g) = %v, %v; want %v", w, h, err, want)
		}
	}
	if h, err := sys.EvalFr(0); err != nil || !cmplx.IsInf(h[0][0]) {
		t.Errorf("EvalFr(0) = %v, %v; want +Inf", h, err)
	}

	if err := sys.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	am, err := AllMargin(sys)
	if err != nil {
		t.Fatalf("AllMargin: %v", err)
	}
	if len(am.PhaseCrossFreqs) != 3 || len(am.GainMargins) != 3 || len(am.GainCrossFreqs) != 1 {
		t.Fatalf("AllMargin = %+v; want 3 phase and 1 gain crossovers", am)
	}
	for i, w := range am.PhaseCrossFreqs {
		l := q.L(w)
		if math.Abs(imag(l)) > 1e-7*cmplx.Abs(l) || real(l) >= 0 {
			t.Errorf("phase crossover %v: L = %v; want negative real", w, l)
		}
		if want := -20 * math.Log10(cmplx.Abs(l)); math.Abs(am.GainMargins[i]-want) > 1e-6 {
			t.Errorf("GM at %v = %v dB; want %v", w, am.GainMargins[i], want)
		}
	}
	if wc := am.GainCrossFreqs[0]; math.Abs(cmplx.Abs(q.L(wc))-1) > 1e-9 {
		t.Errorf("gain crossover %v: |L| = %v; want 1", wc, cmplx.Abs(q.L(wc)))
	}
}

// TestPoleLimitAtPoleKeepsInf: where s is a pole of the realization the
// limit stays +Inf: an undamped axis pair at its frequency, and a double
// integrator at ω = 0.
func TestPoleLimitAtPoleKeepsInf(t *testing.T) {
	for _, c := range []struct {
		name string
		den  []float64
		w    float64
	}{
		{"(s²+1)(s+2)", []float64{1, 2, 1, 2}, 1},
		{"s²(s+1)", []float64{1, 1, 0, 0}, 0},
	} {
		sys := makePlant(t, []float64{1, 3, 1}, c.den)
		h, raw, err := poleLimitAt(sys, c.w)
		if raw == nil {
			t.Fatalf("%s: double solve succeeded; case no longer exercises the pole limit", c.name)
		}
		if err != nil || !cmplx.IsInf(h) {
			t.Errorf("%s: evalWithPoleLimit = %v, %v; want +Inf, nil", c.name, h, err)
		}
	}
}

// realSolveResponse returns C(jωI−A)⁻¹B + D of a SISO model through the real
// 2n system [−A, −ωI; ωI, −A][x_r; x_i] = [B; 0].
func realSolveResponse(t *testing.T, sys *System, w float64) complex128 {
	t.Helper()
	n, _, _ := sys.Dims()
	m := mat.NewDense(2*n, 2*n, nil)
	rhs := mat.NewDense(2*n, 1, nil)
	for i := range n {
		for j := range n {
			m.Set(i, j, -sys.A.At(i, j))
			m.Set(n+i, n+j, -sys.A.At(i, j))
		}
		m.Set(i, n+i, -w)
		m.Set(n+i, i, w)
		rhs.Set(i, 0, sys.B.At(i, 0))
	}
	var x mat.Dense
	if err := x.Solve(m, rhs); err != nil {
		t.Fatal(err)
	}
	g := complex(sys.D.At(0, 0), 0)
	for i := range n {
		g += complex(sys.C.At(0, i), 0) * complex(x.At(i, 0), x.At(n+i, 0))
	}
	return g
}

// TestPoleLimitRotatedIntegratorKeepsInf: an integrator in a non-companion
// realization has a computed eigenvalue of order eps·‖A‖, not 0, so at s = 0
// the relative rule |p−s| ≤ 1e−3·|p| never held and the solve error replaced
// +Inf. At the origin a pole within the cluster tolerance is a pole at s.
func TestPoleLimitRotatedIntegratorKeepsInf(t *testing.T) {
	tr := mat.NewDense(3, 3, []float64{1, 2, -0.5, 0.5, -1, 1, 0.3, 0.7, 2})
	var ti mat.Dense
	if err := ti.Inverse(tr); err != nil {
		t.Fatal(err)
	}
	jordan := func(j []float64) *mat.Dense {
		var a mat.Dense
		a.Mul(tr, mat.NewDense(3, 3, j))
		a.Mul(&a, &ti)
		return &a
	}
	cases := []struct {
		name string
		a    *mat.Dense
		b, c []float64
	}{
		{
			"T·diag(0,−1,−2)·T⁻¹",
			mat.NewDense(3, 3, []float64{-1.6248882522520376, -0.9719506377073485, 0.6405952553186043, -0.3477020896209262, -1.4618964782779869, 0.057624747347860814, -0.2992862221443891, -0.6715049195519469, 0.08678473053002438}),
			[]float64{0.9953954176514858, 1.1800868278140295, 0.48194866225006905},
			[]float64{-0.33415372880770317, -0.012405380534607158, 0.6578201082582456},
		},
		{"T·(J₂(0)⊕−1)·T⁻¹", jordan([]float64{0, 1, 0, 0, 0, 0, 0, 0, -1}), []float64{1, -0.4, 0.8}, []float64{0.6, 1.1, -0.9}},
	}
	for _, c := range cases {
		sys, err := New(c.a, mat.NewDense(3, 1, c.b), mat.NewDense(1, 3, c.c), mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, raw, _ := poleLimitAt(sys, 0); raw == nil {
			t.Fatalf("%s: double solve succeeded at ω=0; case no longer exercises the pole limit", c.name)
		}
		want := realSolveResponse(t, sys, 1)
		check := func(api string, h0, h1 complex128) {
			t.Helper()
			if !cmplx.IsInf(h0) {
				t.Errorf("%s %s(0) = %v; want +Inf", c.name, api, h0)
			}
			if cmplx.Abs(h1-want) > 1e-10*cmplx.Abs(want) {
				t.Errorf("%s %s(1) = %v; want %v", c.name, api, h1, want)
			}
		}
		for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
			"FreqResponse":          sys.FreqResponse,
			"FreqResponsePointwise": sys.FreqResponsePointwise,
		} {
			r, err := f([]float64{0, 1})
			if err != nil {
				t.Errorf("%s %s: %v", c.name, api, err)
				continue
			}
			check(api, r.At(0, 0, 0), r.At(1, 0, 0))
		}
		h0, err0 := sys.EvalFr(0)
		h1, err1 := sys.EvalFr(1i)
		if err0 != nil || err1 != nil {
			t.Errorf("%s EvalFr: %v, %v", c.name, err0, err1)
		} else {
			check("EvalFr", h0[0][0], h1[0][0])
		}
		bd, err := sys.Bode([]float64{0, 1}, 0)
		if err != nil {
			t.Errorf("%s Bode: %v", c.name, err)
			continue
		}
		if !math.IsInf(bd.MagDBAt(0, 0, 0), 1) {
			t.Errorf("%s Bode mag(0) = %v dB; want +Inf", c.name, bd.MagDBAt(0, 0, 0))
		}
		if wantDB := 20 * math.Log10(cmplx.Abs(want)); math.Abs(bd.MagDBAt(1, 0, 0)-wantDB) > 1e-8 {
			t.Errorf("%s Bode mag(1) = %v dB; want %v", c.name, bd.MagDBAt(1, 0, 0), wantDB)
		}
	}
}

// TestPoleLimitOffOriginClusterPublicAPIs: s(s²+0.0954s+1.03e−10) has poles
// {0, −1.07e−9, −0.0954}. Near ω ≈ 1e−10 the double solve is singular though
// no pole is near s; the limit there must not fail the call, as it did for
// Bode, FreqResponse and EvalFr, and AllMargin/Margin of the delayed loop on
// its grid point ω = 1.07e−10. Off the pole the response is the dd one, and
// matches the polynomial ratio.
func TestPoleLimitOffOriginClusterPublicAPIs(t *testing.T) {
	q := quasiLoop{
		num: []float64{1.3174888794708708, -0.5268575342818054, -1.1928987103850879},
		den: []float64{1, 0.09543133525017007, 1.0257621949288232e-10, 0},
		tau: 2.895773588604179e-4,
	}
	sys := makePlant(t, q.num, q.den)
	g := func(w float64) complex128 {
		s := complex(0, w)
		return polyEval(q.num, s) / polyEval(q.den, s)
	}
	const w0 = 1.0748693851901617e-10
	if _, raw, _ := poleLimitAt(sys, w0); raw == nil {
		t.Fatalf("double solve succeeded at ω=%g; case no longer exercises the pole limit", w0)
	}
	closeTo := func(got, want complex128, tol float64) bool {
		return cmplx.Abs(got-want) <= tol*cmplx.Abs(want)
	}
	h, err := sys.EvalFr(complex(0, w0))
	if err != nil || !closeTo(h[0][0], g(w0), 1e-9) {
		t.Errorf("EvalFr(jω₀) = %v, %v; want %v", h, err, g(w0))
	}
	for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
		"FreqResponse":          sys.FreqResponse,
		"FreqResponsePointwise": sys.FreqResponsePointwise,
	} {
		r, err := f([]float64{w0, 1})
		if err != nil {
			t.Errorf("%s: %v", api, err)
			continue
		}
		for k, w := range []float64{w0, 1} {
			if got := r.At(k, 0, 0); !closeTo(got, g(w), 1e-9) {
				t.Errorf("%s(%g) = %v; want %v", api, w, got, g(w))
			}
		}
	}
	bd, err := sys.Bode(nil, 200)
	if err != nil {
		t.Fatalf("Bode: %v", err)
	}
	for k, w := range bd.Omega {
		if want := 20 * math.Log10(cmplx.Abs(g(w))); math.Abs(bd.MagDBAt(k, 0, 0)-want) > 1e-8 {
			t.Errorf("Bode mag(%g) = %v dB; want %v", w, bd.MagDBAt(k, 0, 0), want)
		}
	}

	if err := sys.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	am, err := AllMargin(sys)
	if err != nil {
		t.Fatalf("AllMargin: %v", err)
	}
	if len(am.GainCrossFreqs) != 1 || len(am.PhaseCrossFreqs) != 2 || len(am.PhaseMargins) != 1 || len(am.GainMargins) != 2 {
		t.Fatalf("AllMargin = %+v; want 1 gain and 2 phase crossovers", am)
	}
	wc := am.GainCrossFreqs[0]
	pm := 180 + cmplx.Phase(q.L(wc))*180/math.Pi
	if lc := q.L(wc); math.Abs(cmplx.Abs(lc)-1) > 1e-9 || math.Abs(am.PhaseMargins[0]-pm) > 1e-7 {
		t.Errorf("gain crossover %v: |L| = %v, PM = %v; want |L| = 1, PM = %v", wc, cmplx.Abs(lc), am.PhaseMargins[0], pm)
	}
	for i, w := range am.PhaseCrossFreqs {
		l := q.L(w)
		if math.Abs(imag(l)) > 1e-9*cmplx.Abs(l) || real(l) >= 0 {
			t.Errorf("phase crossover %v: L = %v; want negative real", w, l)
		}
		if want := -20 * math.Log10(cmplx.Abs(l)); math.Abs(am.GainMargins[i]-want) > 1e-7 {
			t.Errorf("GM at %v = %v dB; want %v", w, am.GainMargins[i], want)
		}
	}
	m, err := Margin(sys)
	if err != nil {
		t.Fatalf("Margin: %v", err)
	}
	if math.Abs(m.GainMargin-am.GainMargins[0]) > 1e-9 || math.Abs(m.PhaseMargin-am.PhaseMargins[0]) > 1e-9 {
		t.Errorf("Margin = %+v; want GM %v, PM %v", m, am.GainMargins[0], am.PhaseMargins[0])
	}
}

// TestPoleLimitSamplesBesideOriginCluster (D4OV3H): the pole-limit samples
// s+kδ hit the singular double solve beside a cluster at the pole, and the
// solve error replaced +Inf: s²(s+1)(s+1e−9) at ω = 0, and (z−1)³, whose
// computed poles spread by eps^{1/3}, so δ was a fraction of 7e−6. Ungated,
// the samples fall back to dd and, if still singular, δ steps past the
// cluster. At ω = 1e−14 the stored response is finite and dd resolves it. A
// gated caller keeps the double failure. Oracle: big.Float solve of the
// stored realization.
func TestPoleLimitSamplesBesideOriginCluster(t *testing.T) {
	cont := makePlant(t, []float64{1, 3, 1}, []float64{1, 1 + 1e-9, 1e-9, 0, 0})
	disc := makeDiscretePlant(t, []float64{0.2376376495436092, -0.22623825681198792}, []float64{1, -3, 3, -1}, 0.0411986772067474)
	for _, c := range []struct {
		name      string
		sys       *System
		inf, fin  []float64
		finiteTol float64
	}{
		{"s²(s+1)(s+1e-9)", cont, []float64{0}, []float64{1e-14, 1e-10, 1e-3}, 1e-9},
		{"(z-1)³", disc, []float64{0, 1e-9}, []float64{1e-5, 1e-4, 1e-3}, 1e-8},
	} {
		if _, raw, _ := poleLimitAt(c.sys, 0); raw == nil {
			t.Fatalf("%s: double solve succeeded at ω=0; case no longer exercises the pole limit", c.name)
		}
		ws := append(append([]float64{}, c.inf...), c.fin...)
		for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
			"FreqResponse":          c.sys.FreqResponse,
			"FreqResponsePointwise": c.sys.FreqResponsePointwise,
		} {
			fr, err := f(ws)
			if err != nil {
				t.Errorf("%s %s: %v", c.name, api, err)
				continue
			}
			for k, w := range ws {
				got := fr.At(k, 0, 0)
				if k < len(c.inf) {
					if !cmplx.IsInf(got) {
						t.Errorf("%s %s(%g) = %v; want +Inf", c.name, api, w, got)
					}
					continue
				}
				if want := frExactResponse(c.sys, frExactPoint(w, c.sys.Dt))[0]; cmplx.Abs(got-want) > c.finiteTol*cmplx.Abs(want) {
					t.Errorf("%s %s(%g) = %v; want %v", c.name, api, w, got, want)
				}
			}
		}
		z := complex(1, 0)
		if c.sys.IsContinuous() {
			z = 0
		}
		if h, err := c.sys.EvalFr(z); err != nil || !cmplx.IsInf(h[0][0]) {
			t.Errorf("%s EvalFr(%v) = %v, %v; want +Inf", c.name, z, h, err)
		}
	}
	eval := newFrequencyEvaluator(cont).pointSolver(math.MaxInt).evalInto
	pt := newTimeDomain(0).frequencyPoint(0)
	dst := make([]complex128, 1)
	allow := func(float64) bool { return true }
	for _, gate := range []ddGate{{allow: allow}, {limit: true}} {
		if err := evalWithPoleLimitGated(eval, cont, pt, dst, gate); !errors.Is(err, ErrSingularTransform) {
			t.Errorf("gate limit=%v: evalWithPoleLimitGated = %v, %v; want ErrSingularTransform", gate.limit, dst[0], err)
		}
	}
}

// TestPoleLimitRotatedTripleIntegrator (D4OV3H): T·(J₃(0)⊕−1)·T⁻¹ rounded
// has poles −1, −1.8e−6±3.1e−6i and 3.6e−6, the eps^{1/3} spread of the
// triple root, so the stored realization is regular at 0 and its response
// there finite. The double solve fails at ω ≤ 1e−10 and so did the pole
// limit; the response is the dd one. Oracle: big.Float solve.
func TestPoleLimitRotatedTripleIntegrator(t *testing.T) {
	a := mat.NewDense(4, 4, []float64{-0.09724203518782691, -0.6262482168330955, 1.1245839277223013, 0.3649548264384213, 0.19892217467110473, -0.13456966238706608, -0.2670787763512443, -0.25756855286099223, -0.010778253288952289, -0.34569662387066097, 0.32921223648755743, 0.42431447139007766, 0.004453954667934697, 0.4472658107465526, -0.0286891741955936, -1.0974005389126644})
	sys, err := New(a, mat.NewDense(4, 1, []float64{1, -0.4, 0.8, 0.5}), mat.NewDense(1, 4, []float64{0.6, 1.1, -0.9, 0.3}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, raw, _ := poleLimitAt(sys, 0); raw == nil {
		t.Fatal("double solve succeeded at ω=0; case no longer exercises the pole limit")
	}
	ws := []float64{0, 1e-14, 1e-10, 1e-8, 1e-6}
	check := func(api string, w float64, got complex128) {
		t.Helper()
		if want := frExactResponse(sys, frExactPoint(w, 0))[0]; cmplx.Abs(got-want) > 1e-9*cmplx.Abs(want) {
			t.Errorf("%s(%g) = %v; want %v", api, w, got, want)
		}
	}
	for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
		"FreqResponse":          sys.FreqResponse,
		"FreqResponsePointwise": sys.FreqResponsePointwise,
	} {
		fr, err := f(ws)
		if err != nil {
			t.Fatalf("%s: %v", api, err)
		}
		for k, w := range ws {
			check(api, w, fr.At(k, 0, 0))
		}
	}
	for _, w := range ws {
		h, err := sys.EvalFr(complex(0, w))
		if err != nil {
			t.Fatalf("EvalFr(%g): %v", w, err)
		}
		check("EvalFr", w, h[0][0])
	}
}

// TestPoleLimitIntegratorBesideTinyPoleSweep (D4OV3H): s(s²+as+b) with
// |b| ≤ 1e−10 has a pole of order b/a beside the integrator; where the double
// solve is singular off both, FreqResponse, Bode and AllMargin failed for
// most such plants. Oracles: big.Float solve of the stored realization, and
// |L| = 1 at each gain crossover from the polynomials.
func TestPoleLimitIntegratorBesideTinyPoleSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 3))
	ws := make([]float64, 0, 71)
	for e := -13.0; e <= 1; e += 0.2 {
		ws = append(ws, math.Pow(10, e))
	}
	exercised := 0
	for trial := range 40 {
		a := math.Pow(10, -4+3*rng.Float64())
		b := math.Pow(10, -16+6*rng.Float64())
		if rng.IntN(2) == 0 {
			b = -b
		}
		num := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		den := []float64{1, a, b, 0}
		sys := makePlant(t, num, den)
		for _, w := range ws {
			if _, raw, _ := poleLimitAt(sys, w); raw != nil {
				exercised++
				break
			}
		}
		fr, err := sys.FreqResponse(ws)
		if err != nil {
			t.Errorf("trial %d %v/%v: FreqResponse: %v", trial, num, den, err)
			continue
		}
		for k, w := range ws {
			if want := frExactResponse(sys, frExactPoint(w, 0))[0]; cmplx.Abs(fr.At(k, 0, 0)-want) > 1e-9*cmplx.Abs(want) {
				t.Errorf("trial %d %v/%v: G(%g) = %v; want %v", trial, num, den, w, fr.At(k, 0, 0), want)
			}
		}
		if _, err := sys.Bode(nil, 200); err != nil {
			t.Errorf("trial %d %v/%v: Bode: %v", trial, num, den, err)
		}
		am, err := AllMargin(sys)
		if err != nil {
			t.Errorf("trial %d %v/%v: AllMargin: %v", trial, num, den, err)
			continue
		}
		for _, wc := range am.GainCrossFreqs {
			s := complex(0, wc)
			if l := cmplx.Abs(polyEval(num, s) / polyEval(den, s)); math.Abs(l-1) > 1e-8 {
				t.Errorf("trial %d %v/%v: |L(j%g)| = %v; want 1", trial, num, den, wc, l)
			}
		}
	}
	if exercised < 20 {
		t.Errorf("%d of 40 plants exercise the pole limit; want ≥ 20", exercised)
	}
}

// TestPoleLimitLightlyDampedDiscretePair (ND75VC): the discrete pair
// 1±2.9e−6j, |p|−1 ≈ 4e−12, leaves the double solve singular 4e−10 from a
// pole, within 1e−3 relative but 7× the pole limit's step 1e−5·|p−p̄|. Its
// samples straddled the pole and read a residue: FreqResponse returned the
// solve error, then +Inf, where G ≈ 9.3e13 is finite. Oracles: big.Float
// solve of the stored realization, and num(z)/den(z) in big.Float, for both
// columns of a 1×2 model.
func TestPoleLimitLightlyDampedDiscretePair(t *testing.T) {
	num := []float64{1.5841368901875659, -1.4026199636681365}
	den := []float64{1, -2.348244102065107, 1.6964882041386697, -0.3482441020680517}
	r, err := (&TransferFunc{Num: [][][]float64{{num, {-0.5 * num[0], -0.5 * num[1]}}}, Den: [][]float64{den}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := r.Sys
	const w0 = 2.9074070280281846e-06
	rs, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	siso := rs.Sys
	if _, raw, _ := poleLimitAt(siso, w0); raw == nil {
		t.Fatal("double solve succeeded; case no longer exercises the pole limit")
	}
	poly := func(c []float64, z frBig) frBig {
		v := frBigC(0)
		for _, x := range c {
			v = v.mul(z).add(frBigC(complex(x, 0)))
		}
		return v
	}
	ratio := func(z frBig) complex128 { return poly(num, z).quo(poly(den, z)).complex() }
	check := func(api string, w float64, z frBig, got []complex128) {
		t.Helper()
		want := frExactResponse(sys, z)
		if g := ratio(z); cmplx.Abs(g-want[0]) > 1e-12*cmplx.Abs(g) {
			t.Fatalf("ω=%g: realization %v disagrees with num/den %v", w, want[0], g)
		}
		for j := range want {
			if cmplx.IsInf(got[j]) || cmplx.Abs(got[j]-want[j]) > 1e-12*cmplx.Abs(want[j]) {
				t.Errorf("%s(%.17g)[%d] = %v; want %v", api, w, j, got[j], want[j])
			}
		}
	}
	ws := []float64{w0}
	for _, k := range []float64{1e-9, 1e-8, 1e-7, 1e-6} {
		ws = append(ws, w0*(1-k), w0*(1+k))
	}
	for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
		"FreqResponse":          sys.FreqResponse,
		"FreqResponsePointwise": sys.FreqResponsePointwise,
	} {
		fr, err := f(ws)
		if err != nil {
			t.Fatalf("%s: %v", api, err)
		}
		for k, w := range ws {
			check(api, w, frExactPoint(w, 1), []complex128{fr.At(k, 0, 0), fr.At(k, 0, 1)})
		}
	}
	for _, w := range ws {
		fr, err := siso.FreqResponse([]float64{w})
		if err != nil {
			t.Fatalf("SISO FreqResponse(%g): %v", w, err)
		}
		if want := frExactResponse(siso, frExactPoint(w, 1))[0]; cmplx.Abs(fr.At(0, 0, 0)-want) > 1e-12*cmplx.Abs(want) {
			t.Errorf("SISO FreqResponse(%.17g) = %v; want %v", w, fr.At(0, 0, 0), want)
		}
		z := cmplx.Exp(complex(0, w))
		h, err := sys.EvalFr(z)
		if err != nil {
			t.Fatalf("EvalFr(e^{j%g}): %v", w, err)
		}
		check("EvalFr", w, frBigC(z), h[0])
	}
}

// TestPoleLimitUnresolvedPoleTakesDD (ND75VC): next to a lightly damped
// pair a computed pole is uncertain by about eps/|p−p̄|, more than its
// distance to s, yet within the pole limit's step, or, with p̄ inside
// poleClusterTol, the step jumps past the pair. The double solve failed
// there and the pole limit gave +Inf, nil for a finite response that dd
// resolves; the realization's own poles lie off the axis (unit circle).
// Fixed repros, a near-double axis pair at its frequency and a random
// sweep at computed pole frequencies, each in SISO, 1×2 and descriptor
// form. Oracles: the stored realization in big.Float, and num/den for the
// explicit forms (M·A of the descriptor twin rounds).
func TestPoleLimitUnresolvedPoleTakesDD(t *testing.T) {
	type plant struct {
		num, den []float64
		dt       float64
		ws       []float64
	}
	w2 := 1 + 1e-8
	pair := polyMul([]float64{1, 2e-9, 1}, []float64{1, 2e-9 * w2, w2 * w2})
	plants := []plant{
		{[]float64{0.11931901744373974, 0.19831174247945071, 0.037218094309625305}, []float64{1, -2.4315716530191986, 1.8631433060586569, -0.4315716530365083}, 0.1, []float64{2.2779283771334325e-05}},
		{[]float64{1.0114337837321554, -0.38455634485006573, 0.4304747928833453}, []float64{1, -2.4687192781174288, 1.9374385619169319, -0.46871928379944205}, 0.1, []float64{3.394055460334797e-06, 3.3978457972625797e-06}},
		{[]float64{1, 3, 1}, polyMul(pair, []float64{1, 2}), 0, []float64{1}},
	}
	rng := rand.New(rand.NewPCG(7, 0x4d75))
	for range 40 {
		th := math.Pow(10, -7+3*rng.Float64())
		r := 1 - math.Pow(10, -12+4*rng.Float64())
		den := polyMul([]float64{1, -2 * r * math.Cos(th), r * r}, []float64{1, -0.9 * rng.Float64()})
		plants = append(plants, plant{[]float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}, den, 0.1, nil})
	}
	poly := func(c []float64, z frBig) frBig {
		v := frBigC(0)
		for _, x := range c {
			v = v.mul(z).add(frBigC(complex(x, 0)))
		}
		return v
	}
	exercised := 0
	for k, pl := range plants {
		tf := func(num ...[]float64) *System {
			r, err := (&TransferFunc{Num: [][][]float64{num}, Den: [][]float64{pl.den}, Dt: pl.dt}).StateSpace()
			if err != nil {
				t.Fatal(err)
			}
			return r.Sys
		}
		siso := tf(pl.num)
		ws := pl.ws
		if ws == nil {
			poles, err := siso.Poles()
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range poles {
				if imag(p) > 0 {
					ws = append(ws, cmplx.Phase(p)/pl.dt)
				}
			}
		}
		td := newTimeDomain(pl.dt)
		failing := func(sys *System) []float64 {
			_, m, _ := sys.Dims()
			eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
			var fails []float64
			for _, w := range ws {
				if eval(td.frequencyPoint(w), make([]complex128, m)) != nil {
					fails = append(fails, w)
				}
			}
			return fails
		}
		if fails := failing(siso); k < 3 && len(fails) != len(ws) {
			t.Fatalf("plant %d: double solve succeeded at %v; case no longer exercises the pole limit", k, ws)
		} else if len(fails) > 0 {
			exercised++
		}
		half := []float64{0.5 * pl.num[0], -pl.num[1], 2 * pl.num[2]}
		for form, sys := range map[string]*System{"SISO": siso, "1×2": tf(pl.num, half), "descriptor": poleDescriptorTwin(t, siso)} {
			fails := failing(sys)
			if len(fails) == 0 {
				continue
			}
			check := func(api string, w float64, z frBig, got []complex128) {
				t.Helper()
				want := frExactResponse(sys, z)
				if g := poly(pl.num, z).quo(poly(pl.den, z)).complex(); sys.E == nil && cmplx.Abs(g-want[0]) > 1e-9*cmplx.Abs(g) {
					t.Fatalf("plant %d %s ω=%g: realization %v disagrees with num/den %v", k, form, w, want[0], g)
				}
				for j := range want {
					if cmplx.IsInf(got[j]) || cmplx.Abs(got[j]-want[j]) > 1e-9*cmplx.Abs(want[j]) {
						t.Errorf("plant %d %s %s(%.17g)[%d] = %v; want %v", k, form, api, w, j, got[j], want[j])
					}
				}
			}
			_, m, _ := sys.Dims()
			for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
				"FreqResponse":          sys.FreqResponse,
				"FreqResponsePointwise": sys.FreqResponsePointwise,
			} {
				fr, err := f(fails)
				if err != nil {
					t.Errorf("plant %d %s %s: %v", k, form, api, err)
					continue
				}
				for i, w := range fails {
					got := make([]complex128, m)
					for j := range m {
						got[j] = fr.At(i, 0, j)
					}
					check(api, w, frExactPoint(w, pl.dt), got)
				}
			}
			for _, w := range fails {
				s := td.frequencyVariable(w)
				h, err := sys.EvalFr(s)
				if err != nil {
					t.Errorf("plant %d %s EvalFr(%v): %v", k, form, s, err)
					continue
				}
				check("EvalFr", w, frBigC(s), h[0])
			}
		}
	}
	if exercised < 20 {
		t.Errorf("%d of %d plants exercise the pole limit; want ≥ 20", exercised, len(plants))
	}
}

// TestPoleLimitExactOriginPoleFamily (D4OV3H): s^k(s²+as+b), k = 2, 3, with
// |b| ≤ 1e−10 in companion form keeps its k origin poles exact, so at small
// ω > 0 the response is large but finite. The double solve fails there and
// dd rejected its last pivot, the characteristic polynomial, as below 2⁻¹⁰⁰
// of the pencil scale though formed from its own small terms without
// cancellation; FreqResponse returned ErrSingularTransform. Two fixed
// realizations from the report and a random sweep in SISO, 1×2 and
// descriptor form, including E = diag(I, 0) with an infinite pole. Oracle: big.Float solve of the stored realization.
func TestPoleLimitExactOriginPoleFamily(t *testing.T) {
	ws := make([]float64, 0, 81)
	for e := -16.0; e <= 0; e += 0.2 {
		ws = append(ws, math.Pow(10, e))
	}
	ws = append(ws, 1.0054166766090156e-10)
	check := func(label string, sys *System) {
		t.Helper()
		_, m, p := sys.Dims()
		want := make([][]complex128, len(ws))
		for k, w := range ws {
			want[k] = frExactResponse(sys, frExactPoint(w, 0))
		}
		for api, f := range map[string]func([]float64) (*FreqResponseMatrix, error){
			"FreqResponse":          sys.FreqResponse,
			"FreqResponsePointwise": sys.FreqResponsePointwise,
		} {
			fr, err := f(ws)
			if err != nil {
				t.Errorf("%s %s: %v", label, api, err)
				continue
			}
			for k, w := range ws {
				for i := range p {
					for j := range m {
						if got, g := fr.At(k, i, j), want[k][i*m+j]; cmplx.Abs(got-g) > 1e-9*cmplx.Abs(g) {
							t.Errorf("%s %s(%g)[%d,%d] = %v; want %v", label, api, w, i, j, got, g)
						}
					}
				}
			}
		}
	}
	companion := func(a, b []float64) *System {
		n := len(b)
		c := make([]float64, n)
		c[n-1] = 1
		return poleSys(t, n, 1, 1, a, b, c, []float64{0}, 0)
	}
	infTwin := func(sys *System) *System {
		n, m, p := sys.Dims()
		a := mat.NewDense(n+1, n+1, nil)
		a.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.A)
		a.Set(n, n, 1)
		e := mat.NewDense(n+1, n+1, nil)
		b := mat.NewDense(n+1, m, nil)
		c := mat.NewDense(p, n+1, nil)
		b.Slice(0, n, 0, m).(*mat.Dense).Copy(sys.B)
		c.Slice(0, p, 0, n).(*mat.Dense).Copy(sys.C)
		for i := range n {
			e.Set(i, i, 1)
		}
		twin, err := NewDescriptor(a, b, c, sys.D, e, 0)
		if err != nil {
			t.Fatal(err)
		}
		return twin
	}
	for name, sys := range map[string]*System{
		"double": companion(
			[]float64{0, 0, 0, -0, 1, 0, 0, -0, 0, 1, 0, 2.2702318530740966e-10, 0, 0, 1, -0.0006739302865839005},
			[]float64{0.09173182574148675, -0.9583106107971275, -2.6790649922210594, -0.009260826722141069}),
		"triple": companion(
			[]float64{0, 0, 0, 0, -0, 1, 0, 0, 0, -0, 0, 1, 0, 0, -0, 0, 0, 1, 0, -3.949450574200987e-11, 0, 0, 0, 1, -0.0002672106221559042},
			[]float64{-1.7559471126269988, -0.22991682993005003, 0, 0, 0}),
	} {
		check(name, sys)
		check(name+" E=diag(I,0)", infTwin(sys))
	}

	rng := rand.New(rand.NewPCG(4, 0xd40))
	exercised := 0
	for trial := range 30 {
		k := 2 + trial%2
		a := math.Pow(10, -4+3*rng.Float64())
		b := math.Pow(10, -16+6*rng.Float64())
		if rng.IntN(2) == 0 {
			b = -b
		}
		den := append([]float64{1, a, b}, make([]float64, k)...)
		num := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		half := []float64{0.5 * num[0], -num[1], 2 * num[2]}
		r, err := (&TransferFunc{Num: [][][]float64{{num, half}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		siso := makePlant(t, num, den)
		for _, w := range ws {
			if _, raw, _ := poleLimitAt(siso, w); raw != nil {
				exercised++
				break
			}
		}
		label := func(form string) string { return fmt.Sprintf("trial %d %s %v/%v", trial, form, num, den) }
		check(label("SISO"), siso)
		check(label("1×2"), r.Sys)
		check(label("descriptor"), poleDescriptorTwin(t, siso))
		check(label("E=diag(I,0)"), infTwin(r.Sys))
	}
	if exercised < 20 {
		t.Errorf("%d of 30 plants exercise the pole limit; want ≥ 20", exercised)
	}
}
