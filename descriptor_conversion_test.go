package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const descriptorConversionTol = 1e-12

func assertSystemMatricesClose(t *testing.T, label string, want, got *System) {
	t.Helper()
	if got.E != nil && !isIdentityDescriptor(got.E) {
		t.Fatalf("%s: result keeps descriptor E", label)
	}
	pairs := [][3]any{{"A", want.A, got.A}, {"B", want.B, got.B}, {"C", want.C, got.C}, {"D", want.D, got.D}}
	if want.LFT != nil || got.LFT != nil {
		if want.LFT == nil || got.LFT == nil {
			t.Fatalf("%s: LFT presence differs", label)
		}
		pairs = append(pairs,
			[3]any{"B2", want.LFT.B2, got.LFT.B2}, [3]any{"C2", want.LFT.C2, got.LFT.C2},
			[3]any{"D12", want.LFT.D12, got.LFT.D12}, [3]any{"D21", want.LFT.D21, got.LFT.D21},
			[3]any{"D22", want.LFT.D22, got.LFT.D22})
	}
	for _, p := range pairs {
		w, g := p[1].(*mat.Dense), p[2].(*mat.Dense)
		if w.IsEmpty() && g.IsEmpty() {
			continue
		}
		if !mat.EqualApprox(w, g, descriptorConversionTol) {
			t.Fatalf("%s: %s differs\nwant %v\ngot  %v", label, p[0], mat.Formatted(w), mat.Formatted(g))
		}
	}
}

func assertDescriptorFreqClose(t *testing.T, label string, want, got *System, omega []float64) {
	t.Helper()
	fw, err := want.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: reference FreqResponse: %v", label, err)
	}
	fg, err := got.FreqResponse(omega)
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	for k := range fw.Data {
		if d := cmplx.Abs(fw.Data[k] - fg.Data[k]); d > descriptorConversionTol {
			t.Fatalf("%s: freq response entry %d differs by %g", label, k, d)
		}
	}
}

func descriptorSISOPlant(t *testing.T, dt float64) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1.5, 0.2, -0.1, -0.1, -0.8, 0.15, 0.05, -0.2, -2})
	E := mat.NewDense(3, 3, []float64{1.5, 0.2, 0, 0, 0.8, -0.1, 0.1, 0, 1.2})
	B := mat.NewDense(3, 1, []float64{1, 0.2, -0.4})
	C := mat.NewDense(1, 3, []float64{1, 0.4, -0.3})
	D := mat.NewDense(1, 1, []float64{1})
	if dt > 0 {
		cont, err := New(A, B, C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		disc, err := cont.DiscretizeZOH(dt)
		if err != nil {
			t.Fatal(err)
		}
		A.Mul(E, disc.A)
		B.Mul(E, disc.B)
	}
	sys, err := NewDescriptor(A, B, C, D, E, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func singularDescriptorPlant(t *testing.T, dt float64) *System {
	t.Helper()
	return descriptorSingular(absorbScopePlant(t, dt, false, true))
}

func descriptorSingular(sys *System) *System {
	sys.E = mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 0})
	return sys
}

func TestToExplicitDescriptorInternalDelay(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		sys := absorbScopePlant(t, dt, true, true)
		want := explicitWithLFT(t, sys)
		got, err := sys.ToExplicit()
		if err != nil {
			t.Fatal(err)
		}
		label := "continuous"
		if dt > 0 {
			label = "discrete"
		}
		assertSystemMatricesClose(t, label, want, got)
		assertDescriptorFreqClose(t, label, want, got, absorbScopeOmega)
		if dt > 0 {
			assertSimulateClose(t, label, want, got)
		}
		if sys.LFT.B2 == got.LFT.B2 {
			t.Fatalf("%s: B2 aliases source", label)
		}
	}
}

type descriptorDelaySetup struct {
	name  string
	apply func(*System)
}

var descriptorDelaySetups = []descriptorDelaySetup{
	{"none", func(*System) {}},
	{"input", func(s *System) { s.InputDelay = []float64{0.0025, 0.002}[:descriptorInputs(s)] }},
	{"output", func(s *System) { s.OutputDelay = []float64{0.003, 0.0013}[:descriptorOutputs(s)] }},
	{"io", func(s *System) {
		d := mat.NewDense(2, 2, []float64{0.002, 0, 0.0015, 0.003})
		s.Delay = mat.DenseCopyOf(d.Slice(0, descriptorOutputs(s), 0, descriptorInputs(s)))
	}},
}

func TestDiscretizeDescriptorMatchesExplicit(t *testing.T) {
	const dt = 0.001
	methods := []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH, C2DMethodImpulse, C2DMethodMatched, C2DMethodLeastSquares}
	for _, internal := range []bool{false, true} {
		for _, setup := range descriptorDelaySetups {
			for _, method := range methods {
				if internal && (method == C2DMethodMatched || method == C2DMethodLeastSquares || method == C2DMethodImpulse) {
					continue
				}
				if method == C2DMethodLeastSquares && setup.name != "none" {
					continue
				}
				label := string(method) + "/" + setup.name
				if internal {
					label += "/internal"
				}
				sys := absorbScopePlant(t, 0, internal, true)
				if method == C2DMethodMatched || method == C2DMethodLeastSquares {
					sys = descriptorSISOPlant(t, 0)
				}
				setup.apply(sys)
				explicit := explicitWithLFTOrPlain(t, sys)
				opts := C2DOptions{Method: method}
				if method == C2DMethodLeastSquares {
					opts.FitOrder = 3
				}
				want, err := explicit.DiscretizeWithOpts(dt, opts)
				if err != nil {
					t.Fatalf("%s: explicit: %v", label, err)
				}
				got, err := sys.DiscretizeWithOpts(dt, opts)
				if err != nil {
					t.Fatalf("%s: descriptor: %v", label, err)
				}
				assertSystemMatricesClose(t, label, want, got)
				assertDescriptorFreqClose(t, label, want, got, []float64{0.1, 3, 40})
			}
		}
	}
}

func explicitWithLFTOrPlain(t *testing.T, sys *System) *System {
	t.Helper()
	if sys.LFT != nil {
		return explicitWithLFT(t, sys)
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	out := sys.Copy()
	out.E = nil
	for _, pair := range [][2]*mat.Dense{{out.A, sys.A}, {out.B, sys.B}} {
		if err := lu.SolveTo(pair[0], false, pair[1]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func TestDiscretizeDescriptorShortcutsMatchExplicit(t *testing.T) {
	const dt = 0.001
	type conv func(*System) (*System, error)
	cases := map[string]conv{
		"Discretize":        func(s *System) (*System, error) { return s.Discretize(dt) },
		"DiscretizeZOH":     func(s *System) (*System, error) { return s.DiscretizeZOH(dt) },
		"DiscretizeFOH":     func(s *System) (*System, error) { return s.DiscretizeFOH(dt) },
		"DiscretizeImpulse": func(s *System) (*System, error) { return s.DiscretizeImpulse(dt) },
		"DiscretizeMatched": func(s *System) (*System, error) { return s.DiscretizeMatched(dt) },
		"DiscretizeLeastSquares": func(s *System) (*System, error) {
			r, err := s.DiscretizeLeastSquares(dt, 3)
			if err != nil {
				return nil, err
			}
			return r.Sys, nil
		},
	}
	for name, f := range cases {
		for _, internal := range []bool{false, true} {
			siso := name == "DiscretizeMatched" || name == "DiscretizeLeastSquares"
			if internal && (siso || name == "DiscretizeImpulse") {
				continue
			}
			sys := absorbScopePlant(t, 0, internal, true)
			if siso {
				sys = descriptorSISOPlant(t, 0)
			}
			want, err := f(explicitWithLFTOrPlain(t, sys))
			if err != nil {
				t.Fatalf("%s: explicit: %v", name, err)
			}
			got, err := f(sys)
			if err != nil {
				t.Fatalf("%s: descriptor: %v", name, err)
			}
			assertSystemMatricesClose(t, name, want, got)
			assertDescriptorFreqClose(t, name, want, got, []float64{0.1, 3, 40})
		}
	}
}

func TestDescriptorContinuousConversionAccuracy(t *testing.T) {
	sys := absorbScopePlant(t, 0, false, true)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		disc, err := sys.DiscretizeWithOpts(0.001, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		fc, err := sys.FreqResponse([]float64{0.1})
		if err != nil {
			t.Fatal(err)
		}
		fd, err := disc.FreqResponse([]float64{0.1})
		if err != nil {
			t.Fatal(err)
		}
		for k := range fc.Data {
			if d := cmplx.Abs(fc.Data[k] - fd.Data[k]); d > 1e-4 {
				t.Fatalf("%s: response at w=0.1 differs from continuous by %g", method, d)
			}
		}
	}
}

func TestD2CDescriptorMatchesExplicit(t *testing.T) {
	for _, internal := range []bool{false, true} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH, C2DMethodMatched} {
			if internal && method != C2DMethodTustin {
				continue
			}
			label := string(method)
			if internal {
				label += "/internal"
			}
			sys := absorbScopePlant(t, 0.1, internal, true)
			sys.A.Scale(0.5, sys.A)
			if method == C2DMethodMatched {
				sys = descriptorSISOPlant(t, 0.1)
			}
			want, err := explicitWithLFTOrPlain(t, sys).D2C(method)
			if err != nil {
				t.Fatalf("%s: explicit: %v", label, err)
			}
			got, err := sys.D2C(method)
			if err != nil {
				t.Fatalf("%s: descriptor: %v", label, err)
			}
			assertSystemMatricesClose(t, label, want, got)
			assertDescriptorFreqClose(t, label, want, got, []float64{0.1, 3})
		}
	}
	sys := absorbScopePlant(t, 0.1, false, true)
	sys.A.Scale(0.5, sys.A)
	explicit := explicitWithLFTOrPlain(t, sys)
	want, err := explicit.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	assertSystemMatricesClose(t, "Undiscretize", want, got)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		want, err := explicit.D2D(0.05, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.D2D(0.05, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		assertSystemMatricesClose(t, "D2D/"+string(method), want, got)
	}
}

func TestConversionSingularDescriptorRejected(t *testing.T) {
	cont := singularDescriptorPlant(t, 0)
	disc := singularDescriptorPlant(t, 0.1)
	calls := map[string]func() (*System, error){
		"Discretize":        func() (*System, error) { return cont.Discretize(0.01) },
		"DiscretizeZOH":     func() (*System, error) { return cont.DiscretizeZOH(0.01) },
		"DiscretizeFOH":     func() (*System, error) { return cont.DiscretizeFOH(0.01) },
		"DiscretizeImpulse": func() (*System, error) { return cont.DiscretizeImpulse(0.01) },
		"DiscretizeMatched": func() (*System, error) { return cont.DiscretizeMatched(0.01) },
		"DiscretizeWithOpts": func() (*System, error) {
			return cont.DiscretizeWithOpts(0.01, C2DOptions{Method: C2DMethodTustin})
		},
		"D2C":          func() (*System, error) { return disc.D2C(C2DMethodZOH) },
		"D2CMatched":   func() (*System, error) { return descriptorSingular(descriptorSISOPlant(t, 0.1)).D2C(C2DMethodMatched) },
		"Undiscretize": func() (*System, error) { return disc.Undiscretize() },
		"D2D":          func() (*System, error) { return disc.D2D(0.05, C2DOptions{}) },
	}
	for name, call := range calls {
		_, err := call()
		if !errors.Is(err, ErrDescriptorSingular) || !errors.Is(err, ErrDescriptorUnsupported) {
			t.Fatalf("%s: err = %v, want ErrDescriptorSingular and ErrDescriptorUnsupported", name, err)
		}
	}
}

func descriptorInputs(s *System) int {
	_, m, _ := s.Dims()
	return m
}

func descriptorOutputs(s *System) int {
	_, _, p := s.Dims()
	return p
}

func TestConversionResultDescriptorStateMap(t *testing.T) {
	for _, internal := range []bool{false, true} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH} {
			label := string(method)
			if internal {
				label += "/internal"
			}
			sys := absorbScopePlant(t, 0, internal, true)
			want, err := explicitWithLFTOrPlain(t, sys).DiscretizeWithResult(0.001, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s: explicit: %v", label, err)
			}
			got, err := sys.DiscretizeWithResult(0.001, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s: descriptor: %v", label, err)
			}
			assertSystemMatricesClose(t, label, want.System, got.System)
			assertStateMapClose(t, label, want.InitialStateMap, got.InitialStateMap)
		}
	}
	sys := absorbScopePlant(t, 0.1, true, true)
	sys.A.Scale(0.5, sys.A)
	want, err := explicitWithLFTOrPlain(t, sys).D2CWithResult(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.D2CWithResult(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertSystemMatricesClose(t, "D2CWithResult", want.System, got.System)
	assertStateMapClose(t, "D2CWithResult", want.InitialStateMap, got.InitialStateMap)
}

func assertStateMapClose(t *testing.T, label string, want, got *mat.Dense) {
	t.Helper()
	if want == nil || got == nil {
		if want != got {
			t.Fatalf("%s: initial-state map presence differs", label)
		}
		return
	}
	if !mat.EqualApprox(want, got, descriptorConversionTol) {
		t.Fatalf("%s: initial-state map differs\nwant %v\ngot  %v", label, mat.Formatted(want), mat.Formatted(got))
	}
}
