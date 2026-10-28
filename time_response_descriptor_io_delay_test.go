package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// ioDelayCase puts I/O delays on a copy of a model; tau(i, j) is the total
// delay of channel (i, j) and out(i) the output share, which alone shifts
// the free response.
type ioDelayCase struct {
	name string
	set  func(s *System)
	tau  func(i, j int) float64
	out  func(i int) float64
}

func ioDelayCases(m, p int) []ioDelayCase {
	first := func(k int, v float64) float64 {
		if k == 0 {
			return v
		}
		return 0
	}
	cases := []ioDelayCase{
		{"OutputDelay 0.3",
			func(s *System) { s.OutputDelay = make([]float64, p); s.OutputDelay[0] = 0.3 },
			func(i, _ int) float64 { return first(i, 0.3) },
			func(i int) float64 { return first(i, 0.3) }},
		{"InputDelay 0.2",
			func(s *System) { s.InputDelay = make([]float64, m); s.InputDelay[0] = 0.2 },
			func(_, j int) float64 { return first(j, 0.2) },
			func(int) float64 { return 0 }},
		{"InputDelay 0.1234567 OutputDelay 0.41",
			func(s *System) {
				s.InputDelay, s.OutputDelay = make([]float64, m), make([]float64, p)
				for j := range m {
					s.InputDelay[j] = 0.1234567
				}
				for i := range p {
					s.OutputDelay[i] = 0.41
				}
			},
			func(int, int) float64 { return 0.1234567 + 0.41 },
			func(int) float64 { return 0.41 }},
	}
	if m == 1 && p == 1 {
		// A free response takes the input share of Delay first
		// (splitIODelayForInitialState), as for an explicit model.
		cases = append(cases, ioDelayCase{"Delay 0.15",
			func(s *System) { s.Delay = mat.NewDense(1, 1, []float64{0.15}) },
			func(int, int) float64 { return 0.15 },
			func(int) float64 { return 0 }})
	}
	return cases
}

func (m stepOracleModes) impulse(i, j int, t float64) float64 {
	var y complex128
	for k, l := range m.poles {
		if real(l)*t > -700 {
			y += m.residues[k][i][j] * cmplx.Exp(l*complex(t, 0))
		}
	}
	return real(y)
}

// shiftedAt returns f(t − tau) for t > tau, at0 at t = tau and 0 before;
// ok is false within 1e-9 of tau when at0 is not defined.
func shiftedAt(t, tau float64, at0 *float64, f func(float64) float64) (float64, bool) {
	switch {
	case math.Abs(t-tau) < 1e-9:
		if at0 == nil {
			return 0, false
		}
		return *at0, true
	case t < tau:
		return 0, true
	}
	return f(t - tau), true
}

// MATLAB step/impulse/initial/lsim of a dss with I/O delays is the shifted
// delay-free response. On the full-rank ill-conditioned-E models whose fold
// E⁻¹A loses the slow poles (XWFTRV: seed 2210 was off by 3.33, continuous-k
// seed 1439 by 326 and k 0.7 n 2 seed 2 by 11.1 with any I/O delay), Step and
// Impulse with I/O delays follow the shifted 320-bit pencil response.
func TestStepImpulseIODelayedIllConditionedEFollowShiftedPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		_, m, p := sys.Dims()
		for _, dc := range ioDelayCases(m, p) {
			sd := sys.Copy()
			dc.set(sd)
			st, err := Step(sd, 5)
			if err != nil {
				t.Errorf("%s, %s: Step: %v", names[k], dc.name, err)
				continue
			}
			im, err := Impulse(sd, 5)
			if err != nil {
				t.Errorf("%s, %s: Impulse: %v", names[k], dc.name, err)
				continue
			}
			worstStep, worstImp := 0.0, 0.0
			for s := 1; s < len(st.T); s++ {
				for j := range m {
					for i := range p {
						d := sys.D.At(i, j)
						if want, ok := shiftedAt(st.T[s], dc.tau(i, j), &d, func(x float64) float64 { return modes[k].step(i, j, x) }); ok {
							worstStep = math.Max(worstStep, stepOracleRel(st.Y.At(j*p+i, s), want))
						}
					}
				}
			}
			for s := 1; s < len(im.T); s++ {
				for j := range m {
					for i := range p {
						if want, ok := shiftedAt(im.T[s], dc.tau(i, j), nil, func(x float64) float64 { return modes[k].impulse(i, j, x) }); ok {
							worstImp = math.Max(worstImp, stepOracleRel(im.Y.At(j*p+i, s), want))
						}
					}
				}
			}
			stepTol := 1e-7
			if fractional := dc.tau(0, 0) / st.T[1]; math.Abs(fractional-math.Round(fractional)) > 1e-6 {
				// A delay off the grid takes the fractional-delay ZOH of
				// the stiff explicit parts, which loses the slow mode to
				// ~1e-6 (RR43HG); the fold was off by O(1).
				stepTol = 1e-5
			}
			if worstStep > stepTol {
				t.Errorf("%s, %s: Step off the shifted pencil by %.3g", names[k], dc.name, worstStep)
			}
			if worstImp > 1e-7 {
				t.Errorf("%s, %s: Impulse off the shifted pencil by %.3g", names[k], dc.name, worstImp)
			}
		}
	}
}

// Initial and Lsim from x0 of the same models with I/O delays: the free
// response ignores input delays and is shifted by the output ones (XWFTRV:
// Initial of seed 46 was off by 0.71, k 0.7 n 2 seed 2 by 2.93). Lsim of a
// constant input on a grid the delays fall on adds the shifted forced
// response.
func TestInitialLsimIODelayedIllConditionedEFollowShiftedPencil(t *testing.T) {
	names, systems, modes := stiffDescriptorModels(t)
	for k, sys := range systems {
		n, m, p := sys.Dims()
		x0 := onesVec(n)
		free := freeOracle(t, sys, modes[k], x0)
		for _, dc := range ioDelayCases(m, p) {
			sd := sys.Copy()
			dc.set(sd)
			r, err := Initial(sd, x0, 5)
			if err != nil {
				t.Errorf("%s, %s: Initial: %v", names[k], dc.name, err)
				continue
			}
			worst := 0.0
			for s := 1; s < len(r.T); s++ {
				for i := range p {
					y0 := free(i, 0)
					if want, ok := shiftedAt(r.T[s], dc.out(i), &y0, func(x float64) float64 { return free(i, x) }); ok {
						worst = math.Max(worst, stepOracleRel(r.Y.At(i, s), want))
					}
				}
			}
			if worst > 1e-7 {
				t.Errorf("%s, %s: Initial off the shifted pencil by %.3g", names[k], dc.name, worst)
			}
			const dt = 0.05
			if onGrid := func(v float64) bool { return math.Abs(v/dt-math.Round(v/dt)) < 1e-9 }; !onGrid(dc.tau(0, 0)) || !onGrid(dc.out(0)) {
				continue
			}
			tt := makeTimeVector(101, dt)
			u := mat.NewDense(len(tt), m, nil)
			for s := range tt {
				u.Set(s, 0, 1)
			}
			for _, xx := range []*mat.VecDense{nil, x0} {
				r, err := Lsim(sd, u, tt, xx)
				if err != nil {
					t.Errorf("%s, %s: Lsim (x0 %v): %v", names[k], dc.name, xx != nil, err)
					continue
				}
				worst := 0.0
				for s := 1; s < len(tt); s++ {
					for i := range p {
						d := sys.D.At(i, 0)
						want, ok := shiftedAt(tt[s], dc.tau(i, 0), &d, func(x float64) float64 { return modes[k].step(i, 0, x) })
						if xx != nil {
							y0 := free(i, 0)
							f, okf := shiftedAt(tt[s], dc.out(i), &y0, func(x float64) float64 { return free(i, x) })
							want, ok = want+f, ok && okf
						}
						if ok && math.Abs(tt[s]-dc.tau(i, 0)) > 1e-9 {
							worst = math.Max(worst, stepOracleRel(r.Y.At(i, s), want))
						}
					}
				}
				if worst > 1e-7 {
					t.Errorf("%s, %s: Lsim (x0 %v) off the shifted pencil by %.3g", names[k], dc.name, xx != nil, worst)
				}
			}
		}
	}
}

// A nonzero x0 with a Delay matrix that has no input+output split is still
// rejected on the pencil route, as on the fold route.
func TestInitialLsimIllConditionedEUnsplitDelayMatrixRejected(t *testing.T) {
	names, systems, _ := stiffDescriptorModels(t)
	ran := 0
	for k, sys := range systems {
		n, m, p := sys.Dims()
		if m < 2 {
			continue
		}
		var C, D mat.Dense
		C.Stack(sys.C, sys.C)
		D.Stack(sys.D, sys.D)
		sd, err := NewDescriptor(sys.A, sys.B, &C, &D, sys.E, 0)
		if err != nil {
			t.Fatal(err)
		}
		sd.Delay = mat.NewDense(2*p, m, nil)
		sd.Delay.Set(0, 0, 0.15)
		if _, _, res := decomposeIODelay(sd.Delay); !delayMatrixHasNonzero(res) {
			t.Fatalf("%s: Delay splits into input and output delays", names[k])
		}
		if !foldInaccurateContinuous(sd) {
			t.Fatalf("%s: not on the pencil route", names[k])
		}
		ran++
		x0 := onesVec(n)
		if _, err := Initial(sd, x0, 5); !errors.Is(err, ErrDelayUnsupported) {
			t.Errorf("%s: Initial err = %v, want ErrDelayUnsupported", names[k], err)
		}
		tt := makeTimeVector(11, 0.05)
		if _, err := Lsim(sd, mat.NewDense(len(tt), m, nil), tt, x0); !errors.Is(err, ErrDelayUnsupported) {
			t.Errorf("%s: Lsim err = %v, want ErrDelayUnsupported", names[k], err)
		}
		if _, err := Lsim(sd, mat.NewDense(len(tt), m, nil), tt, nil); err != nil {
			t.Errorf("%s: Lsim zero x0: %v", names[k], err)
		}
	}
	if ran == 0 {
		t.Fatal("no multi-input model")
	}
}
