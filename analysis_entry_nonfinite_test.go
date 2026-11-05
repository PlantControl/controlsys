package controlsys

import (
	"context"
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Analysis entry points reject NaN/Inf caller models and parameters (56CMFW).

func analysisGenPlant() *System {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 2, -0.5, -3},
		[]float64{1, 0.5, 0.5, 1},
		[]float64{0.3, -1, 1, 0.4},
		[]float64{0, 1, 1, 0}, 0)
	if err != nil {
		panic(err)
	}
	return sys
}

func analysisDiscrete() *System {
	sys, err := NewFromSlices(2, 1, 1, []float64{0.5, 0.2, -0.1, 0.4}, []float64{1, 0.5}, []float64{0.3, -1}, []float64{0.2}, 0.1)
	if err != nil {
		panic(err)
	}
	return sys
}

func analysisStaticGain() *System { return modelOpsGain(2) }

func analysisNoFeedthrough() *System {
	sys := modelOpsFinite()
	sys.D.Set(0, 0, 0)
	return sys
}

func analysisEye(n int) *mat.Dense {
	m := mat.NewDense(n, n, nil)
	for i := range n {
		m.Set(i, i, 1)
	}
	return m
}

// analysisPoison sets the last entry of block of a copy of sys to v; block
// "E" makes the model a descriptor with a non-identity E. It reports false
// when sys has no such block.
func analysisPoison(sys *System, block string, v float64) (*System, bool) {
	out := sys.Copy()
	if n, _, _ := out.Dims(); n == 0 && block != "D" {
		return nil, false
	}
	var m *mat.Dense
	switch block {
	case "A":
		m = out.A
	case "B":
		m = out.B
	case "C":
		m = out.C
	case "D":
		m = out.D
	case "E":
		n, _, _ := out.Dims()
		out.E = analysisEye(n)
		out.E.Set(0, n-1, 0.1)
		m = out.E
	}
	r, c := m.Dims()
	m.Set(r-1, c-1, v)
	return out, true
}

func analysisTunableGain() (*TunableGain, error) {
	k, err := newBoundedReal("K", 1, 0.1, 5)
	if err != nil {
		return nil, err
	}
	return tunableGainWith("C", [][]*TunableReal{{k}})
}

type analysisCase struct {
	name string
	base func() *System
	run  func(sys *System) error
}

func analysisErr[T any](_ T, err error) error { return err }

func analysisCases() []analysisCase {
	ctx := context.Background()
	w := []float64{0.1, 1, 10}
	named := func(s *System) *System {
		s.InputName, s.OutputName = []string{"u"}, []string{"y"}
		return s
	}
	return []analysisCase{
		{"Bode", modelOpsFinite, func(s *System) error { return analysisErr(s.Bode(w, 0)) }},
		{"Bode/auto", modelOpsFinite, func(s *System) error { return analysisErr(s.Bode(nil, 20)) }},
		{"Nichols", modelOpsFinite, func(s *System) error { return analysisErr(s.Nichols(w, 0)) }},
		{"Sigma", modelOpsFinite, func(s *System) error { return analysisErr(s.Sigma(w, 0)) }},
		{"Nyquist", modelOpsFinite, func(s *System) error { return analysisErr(s.Nyquist(w, 0)) }},
		{"FreqResponse", modelOpsFinite, func(s *System) error { return analysisErr(s.FreqResponse(w)) }},
		{"FreqResponsePointwise", modelOpsFinite, func(s *System) error { return analysisErr(s.FreqResponsePointwise(w)) }},
		{"EvalFr", modelOpsFinite, func(s *System) error { return analysisErr(s.EvalFr(complex(0, 1))) }},
		{"DefaultFrequencyGrid", modelOpsFinite, func(s *System) error { return analysisErr(s.DefaultFrequencyGrid(20)) }},
		{"FRD", modelOpsFinite, func(s *System) error { return analysisErr(s.FRD(w)) }},
		{"Margin", modelOpsFinite, func(s *System) error { return analysisErr(Margin(s)) }},
		{"AllMargin", modelOpsFinite, func(s *System) error { return analysisErr(AllMargin(s)) }},
		{"Bandwidth", modelOpsFinite, func(s *System) error { return analysisErr(Bandwidth(s, -3)) }},
		{"DiskMargin", modelOpsFinite, func(s *System) error { return analysisErr(DiskMargin(s)) }},
		{"DiskMarginSkew", modelOpsFinite, func(s *System) error { return analysisErr(DiskMarginSkew(s, 0.5)) }},
		{"Step", modelOpsFinite, func(s *System) error { return analysisErr(Step(s, 5)) }},
		{"Step/discrete", analysisDiscrete, func(s *System) error { return analysisErr(Step(s, 1)) }},
		{"Impulse", modelOpsFinite, func(s *System) error { return analysisErr(Impulse(s, 5)) }},
		{"StepInfoForSystem", modelOpsFinite, func(s *System) error { return analysisErr(StepInfoForSystem(s, 5, nil)) }},
		{"Simulate", analysisDiscrete, func(s *System) error {
			return analysisErr(s.Simulate(mat.NewDense(1, 3, []float64{1, 1, 1}), nil, nil))
		}},
		{"Kalman", modelOpsFinite, func(s *System) error {
			return analysisErr(Kalman(s, analysisEye(1), analysisEye(1), nil, nil))
		}},
		{"Kalman/discrete", analysisDiscrete, func(s *System) error {
			return analysisErr(Kalman(s, analysisEye(1), analysisEye(1), nil, nil))
		}},
		{"Kalmd", analysisNoFeedthrough, func(s *System) error {
			return analysisErr(Kalmd(s, analysisEye(1), analysisEye(1), 0.1, nil))
		}},
		{"Lqi", modelOpsFinite, func(s *System) error { return analysisErr(Lqi(s, analysisEye(3), analysisEye(1), nil)) }},
		{"Lqg", modelOpsFinite, func(s *System) error { return analysisErr(Lqg(s, analysisEye(3), analysisEye(3), nil)) }},
		{"H2Syn", analysisGenPlant, func(s *System) error { return analysisErr(H2Syn(s, 1, 1)) }},
		{"HinfSyn", analysisGenPlant, func(s *System) error { return analysisErr(HinfSyn(s, 1, 1)) }},
		{"Mixsyn/G", modelOpsFinite, func(s *System) error {
			return analysisErr(Mixsyn(s, modelOpsGain(0.5), modelOpsGain(0.1), nil))
		}},
		{"Loopsens/P", modelOpsFinite, func(s *System) error { return analysisErr(Loopsens(s, modelOpsGain(1))) }},
		{"Loopsens/C", modelOpsFinite, func(s *System) error { return analysisErr(Loopsens(modelOpsFinite(), s)) }},
		{"Poles", modelOpsFinite, func(s *System) error { return analysisErr(s.Poles()) }},
		{"PolesDetail", modelOpsFinite, func(s *System) error { return analysisErr(s.PolesDetail()) }},
		{"Zeros", modelOpsFinite, func(s *System) error { return analysisErr(s.Zeros()) }},
		{"ZerosDetail", modelOpsFinite, func(s *System) error { return analysisErr(s.ZerosDetail()) }},
		{"IsStable", modelOpsFinite, func(s *System) error { return analysisErr(s.IsStable()) }},
		{"IsProper", modelOpsFinite, func(s *System) error { return analysisErr(s.IsProper()) }},
		{"DCGain", modelOpsFinite, func(s *System) error { return analysisErr(s.DCGain()) }},
		{"Damp", modelOpsFinite, func(s *System) error { return analysisErr(Damp(s)) }},
		{"Pzmap", modelOpsFinite, func(s *System) error { return analysisErr(Pzmap(s)) }},
		{"TotalDelay", modelOpsFinite, func(s *System) error { return analysisErr(s.TotalDelay()) }},
		{"MinimalRealization", modelOpsFinite, func(s *System) error { return analysisErr(s.MinimalRealization()) }},
		{"SelectByName", modelOpsFinite, func(s *System) error {
			return analysisErr(named(s).SelectByName([]string{"u"}, []string{"y"}))
		}},
		{"Canon", modelOpsFinite, func(s *System) error { return analysisErr(Canon(s, CanonModal)) }},
		{"Covar", modelOpsFinite, func(s *System) error { return analysisErr(Covar(s, analysisEye(1))) }},
		{"Gram", modelOpsFinite, func(s *System) error { return analysisErr(Gram(s, GramControllability)) }},
		{"Norm", modelOpsFinite, func(s *System) error { return analysisErr(Norm(s, 2)) }},
		{"H2Norm", modelOpsFinite, func(s *System) error { return analysisErr(H2Norm(s)) }},
		{"HinfNorm", modelOpsFinite, func(s *System) error {
			_, _, err := HinfNorm(s)
			return err
		}},
		{"HSV", modelOpsFinite, func(s *System) error { return analysisErr(HSV(s)) }},
		{"Prescale", modelOpsFinite, func(s *System) error { return analysisErr(Prescale(s)) }},
		{"Ssbal", modelOpsFinite, func(s *System) error { return analysisErr(Ssbal(s)) }},
		{"Stabsep", modelOpsFinite, func(s *System) error { return analysisErr(Stabsep(s)) }},
		{"Modsep", modelOpsFinite, func(s *System) error { return analysisErr(Modsep(s, 1)) }},
		{"ModalTruncate", modelOpsFinite, func(s *System) error {
			return analysisErr(ModalTruncate(s, &ModalTruncateOptions{Order: 1}))
		}},
		{"RootLocus", modelOpsFinite, func(s *System) error { return analysisErr(RootLocus(s, []float64{0, 1})) }},
		{"SampledPassive", modelOpsFinite, func(s *System) error { return analysisErr(SampledPassive(s, nil)) }},
		{"Pidtune", modelOpsFinite, func(s *System) error { return analysisErr(Pidtune(s, PidtunePI, 0, nil)) }},
		{"TunePID", modelOpsFinite, func(s *System) error {
			return analysisErr(TunePID(ctx, s, PidtunePI, PIDTuningOptions{}))
		}},
		{"TransferFunction", modelOpsFinite, func(s *System) error { return analysisErr(s.TransferFunction(nil)) }},
		{"ZPKModel", modelOpsFinite, func(s *System) error { return analysisErr(s.ZPKModel(nil)) }},
		{"TuningGoal.Evaluate", modelOpsFinite, func(s *System) error {
			g, err := NewGainGoal("u", "y", 2)
			if err != nil {
				return err
			}
			return analysisErr(g.Evaluate(named(s)))
		}},
		{"NewLoopShapeGoal", analysisStaticGain, func(s *System) error {
			return analysisErr(NewLoopShapeGoal("u", s, 0))
		}},
		{"NewWeightedGainGoal", analysisStaticGain, func(s *System) error {
			return analysisErr(NewWeightedGainGoal("u", "y", s, nil))
		}},
		{"Looptune", modelOpsFinite, func(s *System) error {
			C0, err := analysisTunableGain()
			if err != nil {
				return err
			}
			return analysisErr(Looptune(ctx, s, C0, []float64{1}, nil, &LooptuneOptions{GridPoints: 5}))
		}},
	}
}

func analysisOutcome(err error) string {
	switch {
	case errors.Is(err, ErrInvalidArgument):
		return "ErrInvalidArgument"
	case errors.Is(err, ErrOverflow):
		return "ErrOverflow"
	case err != nil:
		return "other error: " + err.Error()
	default:
		return "nil error"
	}
}

func TestAnalysisEntry_NonFiniteModelReturnsErrInvalidArgument(t *testing.T) {
	for _, tc := range analysisCases() {
		for _, block := range []string{"A", "B", "C", "D", "E"} {
			for _, nf := range nonFiniteValues {
				sys, ok := analysisPoison(tc.base(), block, nf.v)
				if !ok {
					continue
				}
				t.Run(tc.name+"/"+block+"/"+nf.name, func(t *testing.T) {
					got := analysisOutcome(tc.run(sys))
					if got != "ErrInvalidArgument" {
						t.Errorf("outcome %s, want ErrInvalidArgument", got)
					}
				})
			}
		}
	}
}

func TestAnalysisEntry_FiniteModelUnchanged(t *testing.T) {
	for _, tc := range analysisCases() {
		t.Run(tc.name, func(t *testing.T) {
			if got := analysisOutcome(tc.run(tc.base())); got != "nil error" {
				t.Errorf("outcome %s, want nil error", got)
			}
		})
	}
}

type analysisParamCase struct {
	name string
	run  func() error
}

// analysisParamCases poisons one parameter with v. Frequencies take only NaN:
// ±Inf evaluates the response at infinity (D), as MATLAB evalfr(sys, Inf).
func analysisParamCases(v float64) []analysisParamCase {
	ctx := context.Background()
	s := modelOpsFinite
	w := []float64{0.1, v, 10}
	eye := analysisEye
	poisoned := func(n int) *mat.Dense {
		m := eye(n)
		m.Set(n-1, n-1, v)
		return m
	}
	offDiag := func(n int) *mat.Dense {
		m := eye(n)
		m.Set(0, n-1, v)
		return m
	}
	noFeed := analysisNoFeedthrough
	var cases []analysisParamCase
	if v != v {
		cases = []analysisParamCase{
			{"Bode/omega", func() error { return analysisErr(s().Bode(w, 0)) }},
			{"Nichols/omega", func() error { return analysisErr(s().Nichols(w, 0)) }},
			{"Sigma/omega", func() error { return analysisErr(s().Sigma(w, 0)) }},
			{"FreqResponse/omega", func() error { return analysisErr(s().FreqResponse(w)) }},
			{"FreqResponsePointwise/omega", func() error { return analysisErr(s().FreqResponsePointwise(w)) }},
			{"FRD/omega", func() error { return analysisErr(s().FRD(w)) }},
			{"EvalFr/real", func() error { return analysisErr(s().EvalFr(complex(v, 1))) }},
			{"EvalFr/imag", func() error { return analysisErr(s().EvalFr(complex(0, v))) }},
			{"ModalTruncate/MaxRealPart", func() error {
				return analysisErr(ModalTruncate(s(), &ModalTruncateOptions{MaxRealPart: &v}))
			}},
		}
	}
	return append(cases, []analysisParamCase{
		{"Nyquist/omega", func() error { return analysisErr(s().Nyquist(w, 0)) }},
		{"Bandwidth/dbDrop", func() error { return analysisErr(Bandwidth(s(), v)) }},
		{"DiskMarginSkew/sigma", func() error { return analysisErr(DiskMarginSkew(s(), v)) }},
		{"Step/tFinal", func() error { return analysisErr(Step(s(), v)) }},
		{"Impulse/tFinal", func() error { return analysisErr(Impulse(s(), v)) }},
		{"StepInfoForSystem/tFinal", func() error { return analysisErr(StepInfoForSystem(s(), v, nil)) }},
		{"StepInfoForSystem/SettlingThreshold", func() error {
			return analysisErr(StepInfoForSystem(s(), 5, &StepInfoOptions{SettlingThreshold: v}))
		}},
		{"StepInfoForSystem/RiseTimeLimits", func() error {
			return analysisErr(StepInfoForSystem(s(), 5, &StepInfoOptions{RiseTimeLimits: [2]float64{0.1, v}}))
		}},
		{"StepInfoForSystem/SteadyStateValue", func() error {
			return analysisErr(StepInfoForSystem(s(), 5, &StepInfoOptions{SteadyStateValue: []float64{v}}))
		}},
		{"StepInfoForSystem/InitialValue", func() error {
			return analysisErr(StepInfoForSystem(s(), 5, &StepInfoOptions{InitialValue: []float64{v}}))
		}},
		{"Simulate/u", func() error {
			return analysisErr(analysisDiscrete().Simulate(mat.NewDense(1, 3, []float64{1, v, 1}), nil, nil))
		}},
		{"Simulate/x0", func() error {
			return analysisErr(analysisDiscrete().Simulate(mat.NewDense(1, 3, []float64{1, 1, 1}), mat.NewVecDense(2, []float64{0, v}), nil))
		}},
		{"Lsim/u", func() error {
			return analysisErr(Lsim(s(), mat.NewDense(3, 1, []float64{1, v, 1}), []float64{0, 0.1, 0.2}, nil))
		}},
		{"Lsim/x0", func() error {
			return analysisErr(Lsim(s(), mat.NewDense(3, 1, []float64{1, 1, 1}), []float64{0, 0.1, 0.2}, mat.NewVecDense(2, []float64{v, 0})))
		}},
		{"Initial/x0", func() error { return analysisErr(Initial(s(), mat.NewVecDense(2, []float64{1, v}), 1)) }},
		{"Kalman/Qn", func() error { return analysisErr(Kalman(s(), poisoned(1), eye(1), nil, nil)) }},
		{"Kalman/Rn", func() error { return analysisErr(Kalman(s(), eye(1), poisoned(1), nil, nil)) }},
		{"Kalman/Nn", func() error { return analysisErr(Kalman(s(), eye(1), eye(1), poisoned(1), nil)) }},
		{"Kalmd/Qn", func() error { return analysisErr(Kalmd(noFeed(), poisoned(1), eye(1), 0.1, nil)) }},
		{"Kalmd/Rn", func() error { return analysisErr(Kalmd(noFeed(), eye(1), poisoned(1), 0.1, nil)) }},
		{"Lqi/Q", func() error { return analysisErr(Lqi(s(), poisoned(3), eye(1), nil)) }},
		{"Lqi/Q/offDiagonal", func() error { return analysisErr(Lqi(s(), offDiag(3), eye(1), nil)) }},
		{"Lqi/R", func() error { return analysisErr(Lqi(s(), eye(3), poisoned(1), nil)) }},
		{"Lqg/QXU", func() error { return analysisErr(Lqg(s(), poisoned(3), eye(3), nil)) }},
		{"Lqg/QXU/offDiagonal", func() error { return analysisErr(Lqg(s(), offDiag(3), eye(3), nil)) }},
		{"Lqg/QWV", func() error { return analysisErr(Lqg(s(), eye(3), poisoned(3), nil)) }},
		{"Lqg/QWV/offDiagonal", func() error { return analysisErr(Lqg(s(), eye(3), offDiag(3), nil)) }},
		{"Kalman/Qn/offDiagonal", func() error {
			return analysisErr(Kalman(analysisGenPlant(), offDiag(2), eye(2), nil, nil))
		}},
		{"Lqg/QI", func() error { return analysisErr(Lqg(s(), eye(3), eye(3), &LqgOpts{QI: poisoned(1)})) }},
		{"Covar/W", func() error { return analysisErr(Covar(s(), poisoned(1))) }},
		{"Modsep/cutoff", func() error { return analysisErr(Modsep(s(), v)) }},
		{"RootLocus/gains", func() error { return analysisErr(RootLocus(s(), []float64{0, v})) }},
		{"SampledPassive/Omega", func() error {
			return analysisErr(SampledPassive(s(), &PassivityOptions{Omega: []float64{0, 1, v}}))
		}},
		{"SampledPassive/Tol", func() error { return analysisErr(SampledPassive(s(), &PassivityOptions{Tol: v})) }},
		{"Pidtune/wc", func() error { return analysisErr(Pidtune(s(), PidtunePI, v, nil)) }},
		{"TunePID/CrossoverFrequency", func() error {
			return analysisErr(TunePID(ctx, s(), PidtunePI, PIDTuningOptions{CrossoverFrequency: v}))
		}},
		{"TunePID/PhaseMargin", func() error {
			return analysisErr(TunePID(ctx, s(), PidtunePI, PIDTuningOptions{PhaseMargin: v}))
		}},
		{"Looptune/wc", func() error {
			C0, err := analysisTunableGain()
			if err != nil {
				return err
			}
			return analysisErr(Looptune(ctx, s(), C0, []float64{v}, nil, &LooptuneOptions{GridPoints: 5}))
		}},
		{"NewLoopShapeGoal/crosstol", func() error { return analysisErr(NewLoopShapeGoal("u", modelOpsGain(2), v)) }},
	}...)
}

func TestAnalysisEntry_NonFiniteParameterReturnsErrInvalidArgument(t *testing.T) {
	for _, nf := range nonFiniteValues {
		for _, tc := range analysisParamCases(nf.v) {
			t.Run(tc.name+"/"+nf.name, func(t *testing.T) {
				if got := analysisOutcome(tc.run()); got != "ErrInvalidArgument" {
					t.Errorf("outcome %s, want ErrInvalidArgument", got)
				}
			})
		}
	}
}
