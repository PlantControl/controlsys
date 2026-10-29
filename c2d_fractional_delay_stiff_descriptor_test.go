package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// C2D (ZOH) of a stiff descriptor with an internal delay of 2.5 or 3.5
// samples holds the delayed signal over the fraction of a sample: the
// frequency response matches G of the 320-bit pencil with the modified
// z-transform on the delay column, closed by the whole samples. A single
// exponential of the stiff A·h in the fractional hold rounded the slow
// modes off by up to 0.03 (RT73OO review, fold seed 28: 0.0134).
func TestC2DZOHFractionalInternalDelayStiffDescriptorFollowsPencil(t *testing.T) {
	const dt = 0.1
	names, systems := stiffInternalDelayCases(t)
	for k, sys := range systems {
		for _, tau := range []float64{0.25, 0.35} {
			s := sys.Copy()
			s.LFT.Tau = []float64{tau}
			_, m, _ := s.Dims()
			frac := make([]float64, m+1)
			frac[m] = tau - math.Floor(tau/dt+1e-9)*dt
			oracle := pencilDiscreteOracle(t, lftAugmentedDescriptor(s), C2DMethodZOH, dt, frac)
			d, err := s.C2D(dt, C2DOptions{})
			if err != nil {
				t.Errorf("%s τ=%g: C2D: %v", names[k], tau, err)
				continue
			}
			if worst := lftOracleGap(t, s, d, oracle, dt); !(worst <= 1e-9) {
				t.Errorf("%s τ=%g: C2D frequency response off the pencil by %.3g", names[k], tau, worst)
			}
		}
	}
}

// C2D (ZOH) of a stiff descriptor with a fractional input delay matches the
// modified z-transform of its 320-bit pencil, times the whole samples of
// the delay; the single exponential of the stiff hold put it off by up to
// 0.0095 (stepFoldLossCases seed 355, τ = 0.25).
func TestC2DZOHFractionalInputDelayStiffDescriptorFollowsPencil(t *testing.T) {
	const dt = 0.1
	checked := 0
	for _, c := range stepFoldLossCases {
		core := c.system(t)
		if !matlabRankFull(t, core.E) {
			continue
		}
		checked++
		for _, tau := range []float64{0.25, 0.35} {
			sys := core.Copy()
			sys.InputDelay = make([]float64, c.m)
			sys.InputDelay[0] = tau
			whole := math.Floor(tau/dt + 1e-9)
			frac := make([]float64, c.m)
			frac[0] = tau - whole*dt
			oracle := pencilDiscreteOracle(t, core, C2DMethodZOH, dt, frac)
			d, err := sys.C2D(dt, C2DOptions{})
			if err != nil {
				t.Errorf("%s τ=%g: C2D: %v", c.name, tau, err)
				continue
			}
			worst := 0.0
			for _, th := range []float64{0.05, 0.5, 2, 3} {
				z := cmplx.Exp(complex(0, th))
				got := discreteLFTResponse(t, d, z)
				for i := range c.p {
					for j := range c.m {
						w := oracle(i, j, z)
						if j == 0 {
							w *= cmplx.Pow(z, complex(-whole, 0))
						}
						g := got[i][j] * discreteIODelayAt(d, i, j, z)
						worst = math.Max(worst, cmplx.Abs(g-w)/math.Max(1, cmplx.Abs(w)))
					}
				}
			}
			if !(worst <= 1e-9) {
				t.Errorf("%s τ=%g: C2D frequency response off the pencil by %.3g", c.name, tau, worst)
			}
		}
	}
	if checked < 3 {
		t.Fatalf("only %d models have full-rank E", checked)
	}
}

// discreteLFTResponse is the response at z of the discrete d with its
// internal delays closed, without its I/O delays.
func discreteLFTResponse(t *testing.T, d *System, z complex128) [][]complex128 {
	t.Helper()
	if d.LFT == nil {
		plain := d.Copy()
		plain.InputDelay, plain.OutputDelay, plain.Delay = nil, nil, nil
		return pencilResponse(t, plain, z)
	}
	_, m, p := d.Dims()
	return lftClose(m, p, lftAugmentedAt(t, d, z), d.LFT.Tau, z)
}
