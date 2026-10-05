package controlsys

import (
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
		disc, err := cont.C2D(dt, C2DOptions{})
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
		for _, pair := range [][3]any{{"A", sys.A, got.A}, {"B", sys.B, got.B}, {"B2", sys.LFT.B2, got.LFT.B2}} {
			var back mat.Dense
			back.Mul(sys.E, pair[2].(*mat.Dense))
			if !mat.EqualApprox(&back, pair[1].(*mat.Dense), descriptorConversionTol) {
				t.Fatalf("%s: E*%s does not reproduce descriptor matrix", label, pair[0])
			}
		}
		if sys.LFT.B2 == got.LFT.B2 {
			t.Fatalf("%s: B2 aliases source", label)
		}
	}
}

func TestToExplicitDescriptorNoInputs(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
	E := mat.NewDense(2, 2, []float64{2, 0.5, 0, 1})
	sys, err := NewDescriptor(A, nil, mat.NewDense(1, 2, []float64{1, -1}), nil, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{-0.45, 0.65, -0.2, -2})
	if !mat.EqualApprox(got.A, want, descriptorConversionTol) || got.E != nil {
		t.Fatalf("A = %v, want %v", mat.Formatted(got.A), mat.Formatted(want))
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
				want, err := explicit.C2D(dt, opts)
				if err != nil {
					t.Fatalf("%s: explicit: %v", label, err)
				}
				got, err := sys.C2D(dt, opts)
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
		"Discretize":        func(s *System) (*System, error) { return s.C2D(dt, C2DOptions{Method: C2DMethodTustin}) },
		"DiscretizeZOH":     func(s *System) (*System, error) { return s.C2D(dt, C2DOptions{}) },
		"DiscretizeFOH":     func(s *System) (*System, error) { return s.C2D(dt, C2DOptions{Method: C2DMethodFOH}) },
		"DiscretizeImpulse": func(s *System) (*System, error) { return s.C2D(dt, C2DOptions{Method: C2DMethodImpulse}) },
		"DiscretizeMatched": func(s *System) (*System, error) { return s.C2D(dt, C2DOptions{Method: C2DMethodMatched}) },
		"DiscretizeLeastSquares": func(s *System) (*System, error) {
			rSys, _, err := s.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 3})
			if err != nil {
				return nil, err
			}
			return rSys, nil
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
		disc, err := sys.C2D(0.001, C2DOptions{Method: method})
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
			want, err := explicitWithLFTOrPlain(t, sys).D2C(D2COptions{Method: method})
			if err != nil {
				t.Fatalf("%s: explicit: %v", label, err)
			}
			got, err := sys.D2C(D2COptions{Method: method})
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
	want, err := explicit.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertSystemMatricesClose(t, "Undiscretize", want, got)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		want, err := explicit.D2D(0.05, D2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.D2D(0.05, D2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		assertSystemMatricesClose(t, "D2D/"+string(method), want, got)
	}
}

func TestConversionSingularDescriptorReduced(t *testing.T) {
	cont := singularDescriptorPlant(t, 0)
	disc := singularDescriptorPlant(t, 0.1)
	calls := map[string]func() (*System, error){
		"Discretize":        func() (*System, error) { return cont.C2D(0.01, C2DOptions{Method: C2DMethodTustin}) },
		"DiscretizeZOH":     func() (*System, error) { return cont.C2D(0.01, C2DOptions{}) },
		"DiscretizeFOH":     func() (*System, error) { return cont.C2D(0.01, C2DOptions{Method: C2DMethodFOH}) },
		"DiscretizeImpulse": func() (*System, error) { return cont.C2D(0.01, C2DOptions{Method: C2DMethodImpulse}) },
		"DiscretizeMatched": func() (*System, error) {
			return descriptorSingular(descriptorSISOPlant(t, 0)).C2D(0.01, C2DOptions{Method: C2DMethodMatched})
		},
		"DiscretizeWithOpts": func() (*System, error) {
			return cont.C2D(0.01, C2DOptions{Method: C2DMethodTustin})
		},
		"D2C": func() (*System, error) { return disc.D2C(D2COptions{Method: C2DMethodZOH}) },
		"D2CMatched": func() (*System, error) {
			return descriptorSingular(descriptorSISOPlant(t, 0.1)).D2C(D2COptions{Method: C2DMethodMatched})
		},
		"Undiscretize": func() (*System, error) { return disc.D2C(D2COptions{Method: C2DMethodTustin}) },
		"D2D":          func() (*System, error) { return disc.D2D(0.05, D2DOptions{}) },
	}
	for name, call := range calls {
		got, err := call()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if n, _, _ := got.Dims(); n != 2 || got.IsDescriptor() {
			t.Fatalf("%s: order %d descriptor=%v, want explicit order 2", name, n, got.IsDescriptor())
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

func TestConversionMapDescriptorStateMap(t *testing.T) {
	for _, internal := range []bool{false, true} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH} {
			label := string(method)
			if internal {
				label += "/internal"
			}
			sys := absorbScopePlant(t, 0, internal, true)
			want, wantMap, err := explicitWithLFTOrPlain(t, sys).C2DMap(0.001, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s: explicit: %v", label, err)
			}
			got, gotMap, err := sys.C2DMap(0.001, C2DOptions{Method: method})
			if err != nil {
				t.Fatalf("%s: descriptor: %v", label, err)
			}
			assertSystemMatricesClose(t, label, want, got)
			assertStateMapClose(t, label, wantMap, gotMap)
		}
	}
	sys := absorbScopePlant(t, 0.1, true, true)
	sys.A.Scale(0.5, sys.A)
	want, wantMap, err := explicitWithLFTOrPlain(t, sys).D2CMap(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	got, gotMap, err := sys.D2CMap(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertSystemMatricesClose(t, "D2CMap", want, got)
	assertStateMapClose(t, "D2CMap", wantMap, gotMap)
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
