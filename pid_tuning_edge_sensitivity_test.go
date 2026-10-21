package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// 2FQ6DJ: |S| at ω = 0 and z = −1, which the sampled band stops short of,
// matches the controller's own realization evaluated there.
func TestPIDTuningEdgeSensitivityMatchesController(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 0x2f6d))
	n := 0
	for i := range 400 {
		dt := 0.
		if i%2 == 1 {
			dt = .1
		}
		c := pidRandomControllerTest(rng, dt)
		g := complex(math.Exp(4*(rng.Float64()-.5)), 0)
		if rng.IntN(2) == 0 {
			g = -g
		}
		sys, err := c.System()
		if err != nil {
			continue
		}
		var at, edge complex128
		p := pidTuningPlant{}
		switch {
		case dt == 0 && c.Ki == 0:
			at, p.dcEdge = 0, g
		case dt > 0 && c.Ki == 0 && i%4 == 1:
			at, p.dcEdge = 1, g
		case dt > 0:
			at, p.nyquistEdge = -1, g
		default:
			continue
		}
		h, err := sys.EvalFr(at)
		if err != nil {
			t.Fatal(err)
		}
		edge = h[0][0]
		want := 1 / cmplx.Abs(1+g*edge)
		if !pidFiniteComplex(edge) {
			want = 0
		}
		n++
		got := pidTuningEdgeSensitivity(p, c)
		if math.Abs(got-want) > 1e-9*math.Max(1, want) {
			t.Errorf("%d: %+v G %v at %v: got %g, want %g", i, c, g, at, got, want)
		}
	}
	if n < 200 {
		t.Errorf("only %d cases compared", n)
	}
}

// 2FQ6DJ #18a: on this unstable plant PDF parked its closed-loop real
// root at −3.8e-8, where 1 + Kp·G(0) ≈ 0, with |S(0)| 3.7e5 below the
// band's wc/100 floor. Oracles: companion roots and dense exact |S| down to
// 1e-8 rad/s.
func TestTunePIDDCEdgeSensitivity(t *testing.T) {
	num := []float64{1.2596546753037878, 0.017772352197396}
	den := []float64{1, -20.966911609722604, -19.037466330567696, -6.334020743539023}
	r, err := TunePID(context.Background(), pidtuneBackoffPlant(t, num, den, 0, 0), PidtunePDF, PIDTuningOptions{PhaseMargin: 30})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if re := pidClosedLoopMaxRe(num, den, k); !(re < -1e-4) {
		t.Errorf("%+v: closed-loop root at Re %g", k, re)
	}
	if s := pidExactSensitivityPeak(num, den, k, 1e-8, 1e6); s > 4*1.05 {
		t.Errorf("%+v: |S| %g", k, s)
	}
	if r.Evidence.AchievedPhaseMargin < 27 {
		t.Errorf("%+v: margin %g", k, r.Evidence.AchievedPhaseMargin)
	}
}

// 2FQ6DJ #18b: on sampled data of a stable non-minimum-phase plant, PDF
// put 1 + Kp·G(0) at −9.8e-7 relative to the plant's DC gain, inside the
// gap between the lowest sample and DC, which the Nyquist count took for
// the DC gain: the loop was unstable. No design keeps |S| <= 4, so the
// returned one keeps a warning. Oracle: companion roots of the exact model.
func TestTunePIDFRDDCEdgeStable(t *testing.T) {
	num := []float64{-0.5218348847732306, 0.01176025018024663}
	den := []float64{1, 0.03931173190809955, 0.0050731315734235775}
	r, err := TunePIDFRD(context.Background(), pidExactFRD(t, num, den, 1e-3, 1e2, 100), PidtunePDF, PIDTuningOptions{PhaseMargin: 30})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if re := pidClosedLoopMaxRe(num, den, k); !(re < 0) {
		t.Errorf("%+v: closed-loop root at Re %g", k, re)
	}
	if !pidSensitivityWarned(r) {
		t.Errorf("%+v: |S| %g without warning", k, pidExactSensitivityPeak(num, den, k, 1e-8, 1e3))
	}
}

// 2FQ6DJ #19: an explicit-crossover discrete PDF design with a closed-loop
// root at z = −0.9999974 has |S(z = −1)| 2.6e4, past the 0.99·π/Dt band
// top, and must carry the explicit crossover's |S| warning. Oracle: |S|
// evaluated from the coefficients at z = −1.
func TestTunePIDNyquistEdgeSensitivityWarned(t *testing.T) {
	num := []float64{0.3326997755226579, 0.19928067616655967}
	den := []float64{1, 0.8549741731764755, -0.8818278935884929, -0.8727991041686791, -0.10034717541930334}
	r, err := TunePID(context.Background(), pidtuneBackoffPlant(t, num, den, .1, 0), PidtunePDF, PIDTuningOptions{CrossoverFrequency: 0.021415278880037036, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	g := pidPolyAt(num, -1) / pidPolyAt(den, -1)
	c := k.Kp - k.Ki*k.Dt/2 + k.Kd/(k.Tf-k.Dt/2)
	if s := 1 / cmplx.Abs(1+g*complex(c, 0)); s > 4 && !pidHasWarning(r, "an explicit crossover does not bound it") {
		t.Errorf("%+v: |S(-1)| %g without warning", k, s)
	}
}
