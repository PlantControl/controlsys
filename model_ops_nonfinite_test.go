package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func modelOpsFinite() *System {
	sys, err := NewFromSlices(2, 1, 1, []float64{-1, 2, -0.5, -3}, []float64{1, 0.5}, []float64{0.3, -1}, []float64{0.2}, 0)
	if err != nil {
		panic(err)
	}
	return sys
}

func modelOpsNonFinite(v float64) *System {
	return &System{
		A: mat.NewDense(2, 2, []float64{-1, 2, -0.5, v}),
		B: mat.NewDense(2, 1, []float64{1, 0.5}),
		C: mat.NewDense(1, 2, []float64{0.3, -1}),
		D: mat.NewDense(1, 1, []float64{0.2}),
	}
}

func modelOpsGain(d float64) *System {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{d}), 0)
	if err != nil {
		panic(err)
	}
	return sys
}

// modelOpsLargeBC has B = C = 2^600 entries, so any B·C product is +Inf.
func modelOpsLargeBC(m, p int) *System {
	big := math.Ldexp(1, 600)
	b := make([]float64, 2*m)
	c := make([]float64, p*2)
	for i := range b {
		b[i] = big
	}
	for i := range c {
		c[i] = big
	}
	sys, err := NewFromSlices(2, m, p, []float64{-1, 2, -0.5, -3}, b, c, nil, 0)
	if err != nil {
		panic(err)
	}
	return sys
}

func modelOpsDelayed(sys *System, tau float64) *System {
	out := sys.Copy()
	out.InputDelay = []float64{tau}
	return out
}

// modelOpsCarry returns s's poisoned A entry when it is non-finite, else v,
// for ops whose input cannot reuse the shared A.
func modelOpsCarry(s *System, v float64) float64 {
	if a := s.A.At(1, 1); !isFinite(a) {
		return a
	}
	return v
}

func TestAllFinite_EntryPositions(t *testing.T) {
	x := make([]float64, 12)
	for i := range x {
		x[i] = float64(i) - 5.5
	}
	if !allFinite(x) {
		t.Fatal("allFinite = false for finite entries")
	}
	for _, i := range []int{0, 7, 8, 11} {
		for _, nf := range nonFiniteValues {
			y := append([]float64(nil), x...)
			y[i] = nf.v
			if allFinite(y) {
				t.Errorf("allFinite = true with %s at %d", nf.name, i)
			}
		}
	}
}

func describeModelOpOutcome(sys *System, err error) string {
	switch {
	case errors.Is(err, ErrOverflow):
		return "ErrOverflow"
	case errors.Is(err, ErrInvalidArgument):
		return "ErrInvalidArgument"
	case err != nil:
		return "other error: " + err.Error()
	case sys == nil:
		return "nil model, nil error"
	case systemMatricesFinite(sys, ErrOverflow) != nil:
		return "non-finite model, nil error"
	default:
		return "finite model, nil error"
	}
}

type modelOpCase struct {
	name string
	run  func(sys *System) (*System, error)
}

func modelOpsUnary() []modelOpCase {
	col := mat.NewDense(2, 1, []float64{0.4, -0.2})
	row := mat.NewDense(1, 2, []float64{0.5, 0.1})
	return []modelOpCase{
		{"Series/first", func(s *System) (*System, error) { return Series(s, modelOpsFinite()) }},
		{"Series/second", func(s *System) (*System, error) { return Series(modelOpsFinite(), s) }},
		{"Parallel", func(s *System) (*System, error) { return Parallel(modelOpsFinite(), s) }},
		{"Feedback/plant", func(s *System) (*System, error) { return Feedback(s, modelOpsGain(1), -1) }},
		{"Feedback/controller", func(s *System) (*System, error) { return Feedback(modelOpsFinite(), s, -1) }},
		{"Append", func(s *System) (*System, error) { return Append(modelOpsFinite(), s) }},
		{"BlkDiag", func(s *System) (*System, error) { return BlkDiag(modelOpsFinite(), s) }},
		{"Connect", func(s *System) (*System, error) {
			return Connect(s, []Junction{{Input: 0, Minus: []int{0}}}, []int{0}, []int{0})
		}},
		{"LFT", func(s *System) (*System, error) {
			P, err := NewFromSlices(2, 2, 2, []float64{-1, 2, -0.5, -3}, []float64{1, 0, 0.5, 1}, []float64{0.3, -1, 1, 0}, nil, 0)
			if err != nil {
				return nil, err
			}
			return LFT(P, s)
		}},
		{"Inv", Inv},
		{"Augstate", Augstate},
		{"Pade", func(s *System) (*System, error) { return modelOpsDelayed(s, 0.1).Pade(2) }},
		{"AbsorbDelay", func(s *System) (*System, error) {
			d := s.Copy()
			d.Dt = 0.1
			d.InputDelay = []float64{2}
			return d.AbsorbDelay()
		}},
		{"Estim", func(s *System) (*System, error) { return Estim(s, col, []int{0}, []int{0}) }},
		{"Reg", func(s *System) (*System, error) { return Reg(s, row, col) }},
		{"Sminreal", Sminreal},
		{"ToExplicit", func(s *System) (*System, error) {
			d := s.Copy()
			d.E = mat.NewDense(2, 2, []float64{2, 0.1, 0, 1})
			return d.ToExplicit()
		}},
		{"C2D", func(s *System) (*System, error) { return s.C2D(0.1, C2DOptions{}) }},
		{"D2C", func(s *System) (*System, error) {
			d := s.Copy()
			d.Dt = 0.1
			d.A = mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, modelOpsCarry(s, 0.4)})
			return d.D2C(D2COptions{})
		}},
		{"Reduce", func(s *System) (*System, error) {
			r, err := s.Reduce(nil)
			if err != nil {
				return nil, err
			}
			return r.Sys, nil
		}},
		{"Balreal", func(s *System) (*System, error) {
			r, err := Balreal(s)
			if err != nil {
				return nil, err
			}
			return r.Sys, nil
		}},
		{"Balred", func(s *System) (*System, error) {
			r, _, err := Balred(s, 1, BalredOptions{})
			return r, err
		}},
		{"EliminateStates", func(s *System) (*System, error) { return s.EliminateStates([]int{1}, MatchDC) }},
		{"D2D", func(s *System) (*System, error) {
			d := s.Copy()
			d.Dt = 0.1
			d.A = mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, modelOpsCarry(s, 0.4)})
			return d.D2D(0.2, D2DOptions{})
		}},
		{"MinimalLFT", func(s *System) (*System, error) { return modelOpsDelayed(s, 0.1).MinimalLFT() }},
		{"AugmentInternalDelayOutputs", func(s *System) (*System, error) {
			return s.AugmentInternalDelayOutputs("d")
		}},
		{"Modred", func(s *System) (*System, error) { return Modred(s, []int{1}, MatchDC) }},
		{"SS2SS", func(s *System) (*System, error) {
			return SS2SS(s, mat.NewDense(2, 2, []float64{1, 0.5, 0, 2}))
		}},
		{"StateTransform", func(s *System) (*System, error) {
			return s.StateTransform(mat.NewDense(2, 2, []float64{1, 0.5, 0, 2}))
		}},
		{"Xperm", func(s *System) (*System, error) { return Xperm(s, []int{1, 0}) }},
		{"SelectByIndex", func(s *System) (*System, error) { return s.SelectByIndex([]int{0}, []int{0}) }},
		{"FixedInputReduction", func(s *System) (*System, error) {
			return s.FixedInputReduction(map[int]float64{0: 1}, "")
		}},
		{"SmithPredictor/controller", func(s *System) (*System, error) {
			return SmithPredictor(s, modelOpsFinite(), 0.1, 2)
		}},
		{"SmithPredictor/model", func(s *System) (*System, error) {
			return SmithPredictor(modelOpsFinite(), s, 0.1, 2)
		}},
		{"ZeroDelayApprox", func(s *System) (*System, error) { return modelOpsDelayed(s, 0.1).ZeroDelayApprox() }},
		{"PullDelaysToLFT", func(s *System) (*System, error) { return modelOpsDelayed(s, 0.1).PullDelaysToLFT() }},
		{"GetDelayModel", func(s *System) (*System, error) {
			H, _, err := modelOpsDelayed(s, 0.1).GetDelayModel()
			return H, err
		}},
		{"SetDelayModel", func(s *System) (*System, error) {
			H, err := Append(modelOpsFinite(), modelOpsGain(1))
			if err != nil {
				return nil, err
			}
			H.A = s.A
			return SetDelayModel(H, []float64{0.1})
		}},
		{"Augw", func(s *System) (*System, error) { return Augw(s, modelOpsGain(1), nil, nil) }},
		{"ConnectByName", func(s *System) (*System, error) {
			a := modelOpsFinite()
			a.InputName, a.OutputName = []string{"u"}, []string{"e"}
			b := s.Copy()
			b.InputName, b.OutputName = []string{"e"}, []string{"y"}
			return ConnectByName([]*System{a, b}, nil, []string{"u"}, []string{"y"})
		}},
		{"SpectralFactor", func(s *System) (*System, error) {
			return SpectralFactor(&System{D: mat.NewDense(1, 1, []float64{modelOpsCarry(s, 2)})})
		}},
		{"NewModelArray", func(s *System) (*System, error) {
			a, err := NewModelArray([]int{1}, []*System{s})
			if err != nil {
				return nil, err
			}
			return a.ModelFlat(0)
		}},
		{"NewPhysicalComponent", func(s *System) (*System, error) {
			_, err := NewPhysicalComponent("c", s, nil)
			return s, err
		}},
		{"FixedBlock", func(s *System) (*System, error) {
			b, err := FixedBlock(s)
			if err != nil {
				return nil, err
			}
			return b.CurrentSystem()
		}},
	}
}

func TestModelOps_NonFiniteInputReturnsErrInvalidArgument(t *testing.T) {
	for _, op := range modelOpsUnary() {
		for _, nf := range nonFiniteValues {
			t.Run(op.name+"/"+nf.name, func(t *testing.T) {
				got := describeModelOpOutcome(op.run(modelOpsNonFinite(nf.v)))
				if got != "ErrInvalidArgument" {
					t.Errorf("outcome %s, want ErrInvalidArgument", got)
				}
			})
		}
	}
}

func TestModelOps_FiniteInputUnchanged(t *testing.T) {
	for _, op := range modelOpsUnary() {
		t.Run(op.name, func(t *testing.T) {
			got := describeModelOpOutcome(op.run(modelOpsFinite()))
			if got != "finite model, nil error" {
				t.Errorf("outcome %s, want finite model", got)
			}
		})
	}
}

// modelOpsStepF jumps from -MaxFloat64 to +MaxFloat64 at x = 0, so its
// central difference there overflows.
func modelOpsStepF(x, _ *mat.VecDense) *mat.VecDense {
	return mat.NewVecDense(1, []float64{math.Copysign(math.MaxFloat64, x.AtVec(0))})
}

func modelOpsZeroH(_, _ *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, []float64{0}) }

func TestModelOps_NonFiniteParameterReturnsErrInvalidArgument(t *testing.T) {
	for _, nf := range nonFiniteValues {
		cases := []struct {
			name string
			run  func() (*System, error)
		}{
			{"Reg/K", func() (*System, error) {
				return Reg(modelOpsFinite(), mat.NewDense(1, 2, []float64{1, nf.v}), mat.NewDense(2, 1, []float64{1, 1}))
			}},
			{"Reg/L", func() (*System, error) {
				return Reg(modelOpsFinite(), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(2, 1, []float64{nf.v, 1}))
			}},
			{"FixedInputReduction/fixed", func() (*System, error) {
				return modelOpsFinite().FixedInputReduction(map[int]float64{0: nf.v}, "")
			}},
			{"Linearize/x0", func() (*System, error) {
				return Linearize(&NonlinearModel{F: modelOpsStepF, H: modelOpsZeroH, N: 1, M: 1, P: 1},
					mat.NewVecDense(1, []float64{nf.v}), mat.NewVecDense(1, []float64{0}))
			}},
			{"Linearize/F", func() (*System, error) {
				F := func(_, _ *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, []float64{nf.v}) }
				return Linearize(&NonlinearModel{F: F, H: modelOpsZeroH, N: 1, M: 1, P: 1},
					mat.NewVecDense(1, []float64{0}), mat.NewVecDense(1, []float64{0}))
			}},
		}
		for _, tc := range cases {
			t.Run(tc.name+"/"+nf.name, func(t *testing.T) {
				if got := describeModelOpOutcome(tc.run()); got != "ErrInvalidArgument" {
					t.Errorf("outcome %s, want ErrInvalidArgument", got)
				}
			})
		}
	}
}

func TestModelOps_FiniteOverflowReturnsErrOverflow(t *testing.T) {
	big := math.Ldexp(1, 600)
	huge := math.Ldexp(1, 1023)
	tiny := math.Ldexp(1, -600)
	cases := []struct {
		name string
		run  func() (*System, error)
	}{
		{"Series/static", func() (*System, error) { return Series(modelOpsGain(big), modelOpsGain(big)) }},
		{"Series/dynamic", func() (*System, error) { return Series(modelOpsLargeBC(1, 1), modelOpsLargeBC(1, 1)) }},
		{"Parallel/static", func() (*System, error) { return Parallel(modelOpsGain(huge), modelOpsGain(huge)) }},
		{"Feedback/dynamic", func() (*System, error) { return Feedback(modelOpsLargeBC(1, 1), modelOpsGain(1), -1) }},
		{"Connect/dynamic", func() (*System, error) {
			return Connect(modelOpsLargeBC(1, 1), []Junction{{Input: 0, Minus: []int{0}}}, []int{0}, []int{0})
		}},
		{"LFT/dynamic", func() (*System, error) { return LFT(modelOpsLargeBC(2, 2), modelOpsGain(1)) }},
		{"Inv/dynamic", func() (*System, error) {
			s := modelOpsLargeBC(1, 1)
			s.D.Set(0, 0, 1)
			return Inv(s)
		}},
		{"Estim", func() (*System, error) {
			return Estim(modelOpsLargeBC(1, 1), mat.NewDense(2, 1, []float64{big, big}), []int{0}, []int{0})
		}},
		{"Reg", func() (*System, error) {
			return Reg(modelOpsLargeBC(1, 1), mat.NewDense(1, 2, []float64{big, big}), mat.NewDense(2, 1, []float64{big, big}))
		}},
		{"SS2SS", func() (*System, error) {
			return SS2SS(modelOpsLargeBC(1, 1), mat.NewDense(2, 2, []float64{tiny, tiny / 2, 0, 2 * tiny}))
		}},
		{"FixedInputReduction", func() (*System, error) {
			return modelOpsLargeBC(1, 1).FixedInputReduction(map[int]float64{0: big}, "")
		}},
		{"C2D", func() (*System, error) {
			s := modelOpsFinite()
			s.A.Scale(-big, s.A)
			return s.C2D(1, C2DOptions{})
		}},
		{"SmithPredictor", func() (*System, error) {
			return SmithPredictor(modelOpsLargeBC(1, 1), modelOpsLargeBC(1, 1), 0.1, 2)
		}},
		{"Modred", func() (*System, error) {
			s, err := NewFromSlices(2, 1, 1, []float64{-1, big, big / 2, -tiny}, []float64{1, 0.5}, []float64{0.3, -1}, nil, 0)
			if err != nil {
				return nil, err
			}
			return Modred(s, []int{1}, MatchDC)
		}},
		{"TransferFunc.StateSpace/static", func() (*System, error) {
			tf := &TransferFunc{Num: [][][]float64{{{big}}}, Den: [][]float64{{tiny}}}
			res, err := tf.StateSpace()
			if err != nil {
				return nil, err
			}
			return res.Sys, nil
		}},
		{"PID.System", func() (*System, error) {
			pid, err := NewPID(1, 0, big, tiny, 0)
			if err != nil {
				return nil, err
			}
			return pid.System()
		}},
		{"Linearize", func() (*System, error) {
			return Linearize(&NonlinearModel{F: modelOpsStepF, H: modelOpsZeroH, N: 1, M: 1, P: 1},
				mat.NewVecDense(1, []float64{0}), mat.NewVecDense(1, []float64{0}))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := describeModelOpOutcome(tc.run())
			if got != "ErrOverflow" {
				t.Errorf("outcome %s, want ErrOverflow", got)
			}
		})
	}
}
