package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// lsimStep is a step of height amp on input j at time at.
type lsimStep struct {
	j       int
	at, amp float64
}

// lsimStepGrid returns the uniform grid of n+1 samples on [0, tf] and the
// ZOH input that is the sum of steps.
func lsimStepGrid(n, m int, tf float64, steps []lsimStep) ([]float64, *mat.Dense) {
	dt := tf / float64(n)
	t := make([]float64, n+1)
	u := mat.NewDense(n+1, m, nil)
	for k := range t {
		t[k] = float64(k) * dt
		for _, s := range steps {
			if t[k] >= s.at-dt/2 {
				u.Set(k, s.j, u.At(k, s.j)+s.amp)
			}
		}
	}
	return t, u
}

// lsimStepsOracle is the oracle response to the sum of steps at t plus the
// free response from x0 (nil: none); ok is false when a shifted time lies
// on the internal delay grid.
func lsimStepsOracle(o lftStepOracle, sys *System, steps []lsimStep, x0 []float64, t float64) ([]float64, bool) {
	y := make([]float64, o.p)
	if x0 != nil {
		w, ok := o.delayedResponse(sys, -1, x0, t)
		if !ok {
			return nil, false
		}
		copy(y, w)
	}
	for _, s := range steps {
		if t-s.at <= 0 {
			continue
		}
		w, ok := o.delayedResponse(sys, s.j, nil, t-s.at)
		if !ok {
			return nil, false
		}
		for i := range y {
			y[i] += s.amp * w[i]
		}
	}
	return y, true
}

// Lsim of the stiff descriptor with an internal delay follows the pencil,
// as Step and Initial do (43H73S): for a unit step, a piecewise-constant
// input on every input, a free response from x0 and both together, every
// sample off the delay grid matches the method-of-steps response of E⁻¹A
// formed at 320 bits, shifted by the I/O delays. The error is the O(dt) of
// the ZOH discretization of the delay channels on the caller's grid (dt =
// 2e-5 here; about 5e-4 at dt = 1e-3). The exact chain of the LU fold E⁻¹A
// was off by 0.03 to 0.33.
func TestLsimInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	skipSlowUnderRace(t)
	const tf, samples = 1.2, 60000
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		o := newLFTStepOracle(t, sys)
		n, m, p := sys.Dims()
		x0 := make([]float64, n)
		for i := range n {
			x0[i] = 0.7 - 0.45*float64(i)
		}
		mixed := []lsimStep{{0, 0, 1}, {0, 0.45, -1.5}}
		if m > 1 {
			mixed = append(mixed, lsimStep{1, 0.25, 0.7})
		}
		for _, c := range []struct {
			name  string
			steps []lsimStep
			x0    []float64
		}{
			{"unit step", []lsimStep{{0, 0, 1}}, nil},
			{"piecewise input", mixed, nil},
			{"free from x0", nil, x0},
			{"piecewise input from x0", mixed, x0},
		} {
			tt, u := lsimStepGrid(samples, m, tf, c.steps)
			var x *mat.VecDense
			if c.x0 != nil {
				x = mat.NewVecDense(n, c.x0)
			}
			r, err := Lsim(sys, u, tt, x)
			if err != nil {
				t.Errorf("%s %s: Lsim: %v", names[k], c.name, err)
				continue
			}
			if len(r.T) != len(tt) {
				t.Fatalf("%s %s: %d samples, want %d", names[k], c.name, len(r.T), len(tt))
			}
			worst := 0.0
			for _, s := range lftOracleSamples(len(tt)) {
				want, ok := lsimStepsOracle(o, sys, c.steps, c.x0, tt[s])
				if !ok {
					continue
				}
				for i := range p {
					worst = math.Max(worst, stepOracleRel(r.Y.At(i, s), want[i]))
				}
			}
			t.Logf("%s %s: worst %.3g", names[k], c.name, worst)
			if worst > lftStepTol {
				t.Errorf("%s %s: Lsim off the pencil by %.3g", names[k], c.name, worst)
			}
		}
	}
}

// Lsim of the stiff descriptor with an internal delay keeps the Step
// samples: a unit step on input j on the Step grid gives Step's column j.
func TestLsimInternalDelayStiffDescriptorMatchesStep(t *testing.T) {
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		st, err := Step(sys, 0.5)
		if err != nil {
			t.Fatalf("%s: Step: %v", names[k], err)
		}
		_, m, p := sys.Dims()
		for j := range m {
			u := mat.NewDense(len(st.T), m, nil)
			for s := range st.T {
				u.Set(s, j, 1)
			}
			r, err := Lsim(sys, u, st.T, nil)
			if err != nil {
				t.Fatalf("%s: Lsim: %v", names[k], err)
			}
			worst := 0.0
			for s := range st.T {
				for i := range p {
					worst = math.Max(worst, stepOracleRel(r.Y.At(i, s), st.Y.At(j*p+i, s)))
				}
			}
			if worst > 1e-9 {
				t.Errorf("%s input %d: Lsim off Step by %.3g", names[k], j, worst)
			}
		}
	}
}
