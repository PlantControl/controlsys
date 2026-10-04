package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type aliasFixture struct {
	plant, disc, delayed, discDelayed, residual, internal, desc, gain, siso, sisoK *System
}

func newAliasFixture(t *testing.T) aliasFixture {
	t.Helper()
	named := func(sys *System) *System {
		n, m, p := sys.Dims()
		sys.InputName = make([]string, m)
		for j := range m {
			sys.InputName[j] = fmt.Sprintf("u%d", j)
		}
		sys.OutputName = make([]string, p)
		for i := range p {
			sys.OutputName[i] = fmt.Sprintf("y%d", i)
		}
		if n > 0 {
			sys.StateName = make([]string, n)
			for i := range n {
				sys.StateName[i] = fmt.Sprintf("x%d", i)
			}
		}
		sys.Notes = "fixture"
		return sys
	}
	f := aliasFixture{
		plant:       named(absorbScopePlant(t, 0, false, false)),
		disc:        named(absorbScopePlant(t, 1, false, false)),
		delayed:     named(absorbScopePlant(t, 0, false, false)),
		discDelayed: named(absorbScopePlant(t, 1, false, false)),
		residual:    named(absorbScopePlant(t, 0, false, false)),
		internal:    named(absorbScopePlant(t, 0, true, false)),
		desc:        named(absorbScopePlant(t, 0, false, true)),
	}
	absorbScopeCases[0].apply(t, f.delayed, 0.125)
	absorbScopeCases[0].apply(t, f.discDelayed, 1)
	absorbScopeCases[1].apply(t, f.residual, 0.125)
	gain, err := NewGain(mat.NewDense(2, 2, []float64{1, 0.2, -0.1, 0.8}), 0)
	if err != nil {
		t.Fatal(err)
	}
	f.gain = named(gain)
	siso, err := New(mat.NewDense(2, 2, []float64{-1, 0.4, -0.3, -2}), mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, -0.2}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	f.siso = named(siso)
	sisoK, err := New(mat.NewDense(1, 1, []float64{-0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, []float64{0.8}), 0)
	if err != nil {
		t.Fatal(err)
	}
	f.sisoK = named(sisoK)
	return f
}

type aliasCase struct {
	name string
	in   func(f aliasFixture) []*System
	run  func(in []*System) ([]*System, error)
}

func one(op func(*System) (*System, error)) func([]*System) ([]*System, error) {
	return func(in []*System) ([]*System, error) {
		s, err := op(in[0])
		return []*System{s}, err
	}
}

func two(op func(a, b *System) (*System, error)) func([]*System) ([]*System, error) {
	return func(in []*System) ([]*System, error) {
		s, err := op(in[0], in[1])
		return []*System{s}, err
	}
}

func pick(sel ...func(f aliasFixture) *System) func(f aliasFixture) []*System {
	return func(f aliasFixture) []*System {
		out := make([]*System, len(sel))
		for i, s := range sel {
			out[i] = s(f)
		}
		return out
	}
}

var (
	fxPlant       = func(f aliasFixture) *System { return f.plant }
	fxDisc        = func(f aliasFixture) *System { return f.disc }
	fxDelayed     = func(f aliasFixture) *System { return f.delayed }
	fxDiscDelayed = func(f aliasFixture) *System { return f.discDelayed }
	fxResidual    = func(f aliasFixture) *System { return f.residual }
	fxInternal    = func(f aliasFixture) *System { return f.internal }
	fxDesc        = func(f aliasFixture) *System { return f.desc }
	fxGain        = func(f aliasFixture) *System { return f.gain }
	fxSISO        = func(f aliasFixture) *System { return f.siso }
	fxSISOK       = func(f aliasFixture) *System { return f.sisoK }
)

func aliasCases() []aliasCase {
	cases := []aliasCase{
		{"Copy", pick(fxInternal), one(func(s *System) (*System, error) { return s.Copy(), nil })},
		{"Augstate/delayed", pick(fxDelayed), one(Augstate)},
		{"Augstate/internal", pick(fxInternal), one(Augstate)},
		{"Augstate/gain", pick(fxGain), one(Augstate)},
		{"AbsorbDelay/none", pick(fxPlant), one(func(s *System) (*System, error) { return s.AbsorbDelay() })},
		{"AbsorbDelay/disc", pick(fxDiscDelayed), one(func(s *System) (*System, error) { return s.AbsorbDelay() })},
		{"Pade/none", pick(fxPlant), one(func(s *System) (*System, error) { return s.Pade(2) })},
		{"Pade/delayed", pick(fxDelayed), one(func(s *System) (*System, error) { return s.Pade(2) })},
		{"Pade/internal", pick(fxInternal), one(func(s *System) (*System, error) { return s.Pade(2) })},
		{"PullDelaysToLFT/none", pick(fxPlant), one((*System).PullDelaysToLFT)},
		{"PullDelaysToLFT", pick(fxDelayed), one((*System).PullDelaysToLFT)},
		{"PullDelaysToLFT/residual", pick(fxResidual), one((*System).PullDelaysToLFT)},
		{"Pade/residual", pick(fxResidual), one(func(s *System) (*System, error) { return s.Pade(2) })},
		{"Discretize/residual", pick(fxResidual), one(func(s *System) (*System, error) { return s.Discretize(0.1) })},
		{"Discretize/thiran", pick(fxDelayed), one(func(s *System) (*System, error) {
			return s.DiscretizeWithOpts(0.05, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 2})
		})},
		{"Discretize/thiran/residual", pick(fxResidual), one(func(s *System) (*System, error) {
			return s.DiscretizeWithOpts(0.05, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 2})
		})},
		{"Discretize/internalModel", pick(fxResidual), one(func(s *System) (*System, error) {
			return s.DiscretizeWithOpts(0.05, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
		})},
		{"Discretize/internal", pick(fxInternal), one(func(s *System) (*System, error) { return s.Discretize(0.1) })},
		{"Series/residual", pick(fxResidual, fxPlant), two(Series)},
		{"Feedback/residual", pick(fxResidual, fxGain), two(func(a, b *System) (*System, error) { return Feedback(a, b, -1) })},
		{"ZeroDelayApprox", pick(fxInternal), one((*System).ZeroDelayApprox)},
		{"ZeroDelayApprox/none", pick(fxPlant), one((*System).ZeroDelayApprox)},
		{"MinimalLFT", pick(fxInternal), one((*System).MinimalLFT)},
		{"MinimalLFT/none", pick(fxPlant), one((*System).MinimalLFT)},
		{"AugmentInternalDelayOutputs", pick(fxInternal), one(func(s *System) (*System, error) { return s.AugmentInternalDelayOutputs("d") })},
		{"GetDelayModel", pick(fxInternal), one(func(s *System) (*System, error) { H, _ := s.GetDelayModel(); return H, nil })},
		{"GetDelayModel/none", pick(fxPlant), one(func(s *System) (*System, error) { H, _ := s.GetDelayModel(); return H, nil })},
		{"SetDelayModel", pick(fxPlant), one(func(s *System) (*System, error) {
			return SetDelayModel(s, nil)
		})},
		{"Discretize", pick(fxDelayed), one(func(s *System) (*System, error) { return s.Discretize(0.1) })},
		{"DiscretizeZOH", pick(fxPlant), one(func(s *System) (*System, error) { return s.DiscretizeZOH(0.1) })},
		{"DiscretizeFOH", pick(fxPlant), one(func(s *System) (*System, error) { return s.DiscretizeFOH(0.1) })},
		{"DiscretizeImpulse", pick(fxPlant), one(func(s *System) (*System, error) { return s.DiscretizeImpulse(0.1) })},
		{"DiscretizeMatched", pick(fxSISO), one(func(s *System) (*System, error) { return s.DiscretizeMatched(0.1) })},
		{"DiscretizeTustin", pick(fxPlant), one(func(s *System) (*System, error) {
			return s.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethodTustin})
		})},
		{"D2C", pick(fxDisc), one(func(s *System) (*System, error) { return s.D2C(C2DMethodZOH) })},
		{"D2C/delayed", pick(fxDiscDelayed), one(func(s *System) (*System, error) { return s.D2C(C2DMethodZOH) })},
		{"D2D/same", pick(fxDisc), one(func(s *System) (*System, error) { return s.D2D(1, C2DOptions{}) })},
		{"D2D", pick(fxDisc), one(func(s *System) (*System, error) { return s.D2D(0.5, C2DOptions{}) })},
		{"Undiscretize", pick(fxDisc), one((*System).Undiscretize)},
		{"ToExplicit/desc", pick(fxDesc), one((*System).ToExplicit)},
		{"ToExplicit/plain", pick(fxPlant), one((*System).ToExplicit)},
		{"SelectByIndex/all", pick(fxDelayed), one(func(s *System) (*System, error) { return s.SelectByIndex([]int{0, 1}, []int{0, 1}) })},
		{"SelectByIndex", pick(fxInternal), one(func(s *System) (*System, error) { return s.SelectByIndex([]int{1}, []int{0}) })},
		{"SelectByName", pick(fxDelayed), one(func(s *System) (*System, error) { return s.SelectByName([]string{"u1"}, []string{"y0", "y1"}) })},
		{"Xperm/identity", pick(fxPlant), one(func(s *System) (*System, error) { return Xperm(s, []int{0, 1, 2}) })},
		{"Xperm", pick(fxPlant), one(func(s *System) (*System, error) { return Xperm(s, []int{2, 0, 1}) })},
		{"SS2SS", pick(fxInternal), one(func(s *System) (*System, error) {
			return SS2SS(s, mat.NewDense(3, 3, []float64{1, 0.2, 0, 0, 1, 0.1, 0.3, 0, 1}))
		})},
		{"FixedInputReduction", pick(fxPlant), one(func(s *System) (*System, error) {
			return s.FixedInputReduction(map[int]float64{1: 0.5}, "off")
		})},
		{"FixedInputReduction/none", pick(fxPlant), one(func(s *System) (*System, error) { return s.FixedInputReduction(nil, "off") })},
		{"Modred", pick(fxPlant), one(func(s *System) (*System, error) { return Modred(s, []int{2}, SingularPerturbation) })},
		{"Modred/none", pick(fxPlant), one(func(s *System) (*System, error) { return Modred(s, nil, Truncate) })},
		{"Balred", pick(fxPlant), one(func(s *System) (*System, error) { r, _, err := Balred(s, 2, Truncate); return r, err })},
		{"Balreal", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(Balreal(s)) })},
		{"Canon", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(Canon(s, CanonModal)) })},
		{"Prescale", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(Prescale(s)) })},
		{"Ssbal", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(Ssbal(s)) })},
		{"Reduce", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(s.Reduce(nil)) })},
		{"MinimalRealization", pick(fxPlant), one(func(s *System) (*System, error) { return resultSys(s.MinimalRealization()) })},
		{"ModalTruncate", pick(fxPlant), one(func(s *System) (*System, error) {
			return resultSys(ModalTruncate(s, &ModalTruncateOptions{Order: 2}))
		})},
		{"Sminreal", pick(fxPlant), one(Sminreal)},
		{"Sminreal/gain", pick(fxGain), one(Sminreal)},
		{"Stabsep", pick(fxPlant), func(in []*System) ([]*System, error) {
			r, err := Stabsep(in[0])
			if err != nil {
				return nil, err
			}
			return []*System{r.Stable, r.Unstable}, nil
		}},
		{"Modsep", pick(fxPlant), func(in []*System) ([]*System, error) {
			r, err := Modsep(in[0], 1)
			if err != nil {
				return nil, err
			}
			return []*System{r.Slow, r.Fast}, nil
		}},
		{"Inv", pick(fxPlant), one(Inv)},
		{"Inv/gain", pick(fxGain), one(Inv)},
		{"Estim", pick(fxPlant), one(func(s *System) (*System, error) {
			return Estim(s, mat.NewDense(3, 2, []float64{0.1, 0, 0, 0.2, 0.1, 0.1}))
		})},
		{"Reg", pick(fxPlant), one(func(s *System) (*System, error) {
			return Reg(s, mat.NewDense(2, 3, []float64{0.1, 0, 0.2, 0, 0.3, 0}), mat.NewDense(3, 2, []float64{0.1, 0, 0, 0.2, 0.1, 0.1}))
		})},
		{"Series", pick(fxDelayed, fxGain), two(Series)},
		{"Series/internal", pick(fxInternal, fxPlant), two(Series)},
		{"Series/desc", pick(fxDesc, fxPlant), two(Series)},
		{"Parallel", pick(fxDelayed, fxPlant), two(Parallel)},
		{"Parallel/gain", pick(fxGain, fxGain), two(Parallel)},
		{"Append", pick(fxInternal, fxGain), two(Append)},
		{"Append/gain", pick(fxGain, fxGain), two(Append)},
		{"BlkDiag/one", pick(fxDelayed), one(func(s *System) (*System, error) { return BlkDiag(s) })},
		{"BlkDiag", pick(fxDelayed, fxDesc), two(func(a, b *System) (*System, error) { return BlkDiag(a, b) })},
		{"Feedback", pick(fxPlant, fxGain), two(func(a, b *System) (*System, error) { return Feedback(a, b, -1) })},
		{"Feedback/nil", pick(fxPlant), one(func(s *System) (*System, error) { return Feedback(s, nil, -1) })},
		{"Feedback/delayed", pick(fxDelayed, fxGain), two(func(a, b *System) (*System, error) {
			return Feedback(a, b, -1, WithApproximatedDelays())
		})},
		{"Feedback/internal", pick(fxInternal, fxGain), two(func(a, b *System) (*System, error) { return Feedback(a, b, -1) })},
		{"LFT/nil", pick(fxPlant), one(func(s *System) (*System, error) { return LFT(s, nil, 1, 1) })},
		{"LFT/nil/delayed", pick(fxDelayed), one(func(s *System) (*System, error) { return LFT(s, nil, 1, 1) })},
		{"LFT/nil/internal", pick(fxInternal), one(func(s *System) (*System, error) { return LFT(s, nil, 1, 1) })},
		{"LFT/nil/full", pick(fxPlant), one(func(s *System) (*System, error) { return LFT(s, nil, 2, 2) })},
		{"LFT", pick(fxPlant, fxSISOK), two(func(a, b *System) (*System, error) { return LFT(a, b, 1, 1) })},
		{"LFT/delayed", pick(fxDelayed, fxSISOK), two(func(a, b *System) (*System, error) { return LFT(a, b, 1, 1) })},
		{"Connect", pick(fxPlant), one(func(s *System) (*System, error) {
			return Connect(s, mat.NewDense(2, 2, []float64{0, 0.5, 0, 0}), []int{0, 1}, []int{0, 1})
		})},
		{"Connect/delayed", pick(fxDelayed), one(func(s *System) (*System, error) {
			return Connect(s, mat.NewDense(2, 2, nil), []int{0, 1}, []int{0, 1})
		})},
		{"ConnectByName", pick(fxPlant), one(func(s *System) (*System, error) {
			return ConnectByName([]*System{s}, nil, []string{"u0", "u1"}, []string{"y0", "y1"})
		})},
		{"SmithPredictor", pick(fxSISOK, fxSISO), two(func(c, m *System) (*System, error) { return SmithPredictor(c, m, 0.3, 2) })},
		{"Loopsens", pick(fxSISO, fxSISOK), func(in []*System) ([]*System, error) {
			r, err := Loopsens(in[0], in[1])
			if err != nil {
				return nil, err
			}
			return []*System{r.So, r.To, r.Si, r.Ti}, nil
		}},
	}
	for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbInternal, AbsorbAll} {
		for _, fx := range []struct {
			name string
			sel  func(aliasFixture) *System
		}{{"delayed", fxDelayed}, {"residual", fxResidual}, {"internal", fxInternal}, {"plain", fxPlant}} {
			cases = append(cases, aliasCase{"AbsorbDelay(" + string(scope) + ")/" + fx.name, pick(fx.sel),
				one(func(s *System) (*System, error) { return s.AbsorbDelay(scope) })})
		}
	}
	return cases
}

type sysResult interface {
	*BalrealResult | *CanonResult | *PrescaleResult | *SsbalResult | *ReduceResult | *ModalReductionResult
}

func resultSys[R sysResult](r R, err error) (*System, error) {
	if err != nil {
		return nil, err
	}
	switch v := any(r).(type) {
	case *BalrealResult:
		return v.Sys, nil
	case *CanonResult:
		return v.Sys, nil
	case *PrescaleResult:
		return v.Sys, nil
	case *SsbalResult:
		return v.Sys, nil
	case *ReduceResult:
		return v.Sys, nil
	case *ModalReductionResult:
		return v.Sys, nil
	}
	panic("unreachable")
}

// TestSystemOpsDoNotAlias mutates every field of each op's result and checks
// that neither the inputs nor sibling results change.
func TestSystemOpsDoNotAlias(t *testing.T) {
	for _, c := range aliasCases() {
		t.Run(c.name, func(t *testing.T) {
			in := c.in(newAliasFixture(t))
			snaps := make([]*System, len(in))
			for i, s := range in {
				snaps[i] = s.Copy()
			}
			out, err := c.run(in)
			if err != nil {
				t.Fatal(err)
			}
			for i, r := range out {
				if r == nil {
					continue
				}
				for j, s := range in {
					if r == s {
						t.Fatalf("result %d is input %d", i, j)
					}
				}
				if err := r.Validate(); err != nil {
					t.Fatalf("result %d invalid: %v", i, err)
				}
			}
			outSnaps := make([]*System, len(out))
			for i, r := range out {
				if r != nil {
					outSnaps[i] = r.Copy()
				}
			}
			for i, r := range out {
				if r == nil {
					continue
				}
				mutateSystem(r)
				for j, s := range in {
					if d := systemDiff(snaps[j], s); d != "" {
						t.Fatalf("mutating result %d changed input %d: %s", i, j, d)
					}
				}
				for k, o := range out {
					if k == i || o == nil {
						continue
					}
					if d := systemDiff(outSnaps[k], o); d != "" {
						t.Fatalf("mutating result %d changed result %d: %s", i, k, d)
					}
				}
				outSnaps[i] = r.Copy()
			}
		})
	}
}

// TestSystemConstructorsCopyArguments mutates caller-owned arguments after
// construction and checks the system is unchanged.
func TestSystemConstructorsCopyArguments(t *testing.T) {
	type args struct {
		mats   []*mat.Dense
		slices [][]float64
		names  []string
	}
	fresh := func() args {
		return args{
			mats: []*mat.Dense{
				mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2}),
				mat.NewDense(2, 2, []float64{1, 0.2, 0, 1}),
				mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}),
				mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2}),
				mat.NewDense(2, 2, []float64{1.2, 0.1, 0, 0.9}),
				mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 0.4}),
				mat.NewDense(2, 2, []float64{0.2, 0, 0.1, 0.1}),
				mat.NewDense(2, 2, []float64{0.1, 0.1, 0, 0.2}),
			},
			slices: [][]float64{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}},
			names:  []string{"a", "b"},
		}
	}
	cases := []struct {
		name  string
		build func(a args) (*System, error)
	}{
		{"New", func(a args) (*System, error) { return New(a.mats[0], a.mats[1], a.mats[2], a.mats[3], 0) }},
		{"NewDescriptor", func(a args) (*System, error) {
			return NewDescriptor(a.mats[0], a.mats[1], a.mats[2], a.mats[3], a.mats[4], 0)
		}},
		{"NewWithDelay", func(a args) (*System, error) {
			return NewWithDelay(a.mats[0], a.mats[1], a.mats[2], a.mats[3], a.mats[5], 0)
		}},
		{"NewGain", func(a args) (*System, error) { return NewGain(a.mats[3], 0) }},
		{"NewFromSlices", func(a args) (*System, error) {
			return NewFromSlices(1, 1, 1, a.slices[0][:1], a.slices[1][:1], a.slices[2][:1], a.slices[0][1:], 0)
		}},
		{"Setters", func(a args) (*System, error) {
			s, err := New(mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2}), mat.NewDense(2, 2, []float64{1, 0.2, 0, 1}),
				mat.NewDense(2, 2, []float64{1, 0, 0.4, 1}), mat.NewDense(2, 2, nil), 0)
			if err != nil {
				return nil, err
			}
			if err := s.SetDelay(a.mats[5]); err != nil {
				return nil, err
			}
			if err := s.SetInputDelay(a.slices[0]); err != nil {
				return nil, err
			}
			if err := s.SetOutputDelay(a.slices[1]); err != nil {
				return nil, err
			}
			if err := s.SetInternalDelay(a.slices[2], a.mats[0], a.mats[1], a.mats[6], a.mats[7], a.mats[3]); err != nil {
				return nil, err
			}
			if err := s.SetInputName(a.names...); err != nil {
				return nil, err
			}
			if err := s.SetOutputName(a.names...); err != nil {
				return nil, err
			}
			return s, s.SetStateName(a.names...)
		}},
		{"SetDelayModel", func(a args) (*System, error) {
			H, err := New(a.mats[0], mat.NewDense(2, 3, []float64{1, 0, 0.1, 0, 1, 0.2}),
				mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0.3, 0.1}), mat.NewDense(3, 3, nil), 0)
			if err != nil {
				return nil, err
			}
			return SetDelayModel(H, a.slices[0][:1])
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := fresh()
			sys, err := c.build(a)
			if err != nil {
				t.Fatal(err)
			}
			snap := sys.Copy()
			for _, m := range a.mats {
				fillDense(m)
			}
			for _, s := range a.slices {
				for i := range s {
					s[i] = math.NaN()
				}
			}
			for i := range a.names {
				a.names[i] = "mutated"
			}
			if d := systemDiff(snap, sys); d != "" {
				t.Fatalf("mutating arguments changed system: %s", d)
			}
		})
	}
}

func fillDense(m *mat.Dense) {
	if m == nil {
		return
	}
	data := m.RawMatrix().Data
	for i := range data {
		data[i] = math.NaN()
	}
}

func mutateSystem(s *System) {
	for _, m := range []*mat.Dense{s.A, s.B, s.C, s.D, s.E, s.Delay} {
		fillDense(m)
	}
	for _, d := range [][]float64{s.InputDelay, s.OutputDelay} {
		for i := range d {
			d[i] = math.NaN()
		}
	}
	if s.LFT != nil {
		for i := range s.LFT.Tau {
			s.LFT.Tau[i] = math.NaN()
		}
		for _, m := range []*mat.Dense{s.LFT.B2, s.LFT.C2, s.LFT.D12, s.LFT.D21, s.LFT.D22} {
			fillDense(m)
		}
	}
	for _, names := range [][]string{s.InputName, s.OutputName, s.StateName} {
		for i := range names {
			names[i] = "mutated"
		}
	}
}

func systemDiff(want, got *System) string {
	mats := func(s *System) []*mat.Dense {
		out := []*mat.Dense{s.A, s.B, s.C, s.D, s.E, s.Delay}
		if s.LFT != nil {
			out = append(out, s.LFT.B2, s.LFT.C2, s.LFT.D12, s.LFT.D21, s.LFT.D22)
		}
		return out
	}
	labels := []string{"A", "B", "C", "D", "E", "Delay", "B2", "C2", "D12", "D21", "D22"}
	wm, gm := mats(want), mats(got)
	if len(wm) != len(gm) {
		return "LFT presence"
	}
	for k := range wm {
		if !denseIdentical(wm[k], gm[k]) {
			return labels[k]
		}
	}
	floats := [][2][]float64{{want.InputDelay, got.InputDelay}, {want.OutputDelay, got.OutputDelay}}
	if want.LFT != nil {
		floats = append(floats, [2][]float64{want.LFT.Tau, got.LFT.Tau})
	}
	for k, f := range floats {
		if !floatsIdentical(f[0], f[1]) {
			return []string{"InputDelay", "OutputDelay", "Tau"}[k]
		}
	}
	for k, n := range [][2][]string{{want.InputName, got.InputName}, {want.OutputName, got.OutputName}, {want.StateName, got.StateName}} {
		if fmt.Sprint(n[0]) != fmt.Sprint(n[1]) {
			return []string{"InputName", "OutputName", "StateName"}[k]
		}
	}
	return ""
}

func denseIdentical(a, b *mat.Dense) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := range ar {
		for j := range ac {
			if math.Float64bits(a.At(i, j)) != math.Float64bits(b.At(i, j)) {
				return false
			}
		}
	}
	return true
}

func floatsIdentical(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return true
}
