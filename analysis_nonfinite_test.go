package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type nonFiniteVariant struct {
	name   string
	dt     float64
	desc   bool
	lft    bool
	blocks []string
}

var nonFiniteVariants = []nonFiniteVariant{
	{"continuous", 0, false, false, []string{"A", "B", "C", "D"}},
	{"discrete", 0.1, false, false, []string{"A", "B", "C", "D"}},
	{"descriptor", 0, true, false, []string{"E"}},
	{"continuousLFT", 0, false, true, []string{"B2", "C2", "D12", "D21", "D22"}},
	{"discreteLFT", 0.1, false, true, []string{"B2", "C2", "D12", "D21", "D22"}},
}

func nonFiniteModel(t *testing.T, v nonFiniteVariant, block string, val float64) *System {
	t.Helper()
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	if v.dt > 0 {
		A = mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, -0.3})
	}
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0.3})
	D := mat.NewDense(1, 1, []float64{0.1})
	var sys *System
	var err error
	if v.desc {
		sys, err = NewDescriptor(A, B, C, D, mat.NewDense(2, 2, []float64{1, 0.2, 0, 1}), v.dt)
	} else {
		sys, err = New(A, B, C, D, v.dt)
	}
	if err != nil {
		t.Fatal(err)
	}
	if v.lft {
		tau := 0.3
		if v.dt > 0 {
			tau = 2
		}
		if err := sys.SetInternalDelay([]float64{tau},
			mat.NewDense(2, 1, []float64{0.2, 0.1}), mat.NewDense(1, 2, []float64{0.3, -0.2}),
			mat.NewDense(1, 1, []float64{0.1}), mat.NewDense(1, 1, []float64{0.2}),
			mat.NewDense(1, 1, []float64{0.1})); err != nil {
			t.Fatal(err)
		}
	}
	m := map[string]*mat.Dense{"A": sys.A, "B": sys.B, "C": sys.C, "D": sys.D, "E": sys.E}
	if l := sys.LFT; l != nil {
		m["B2"], m["C2"], m["D12"], m["D21"], m["D22"] = l.B2, l.C2, l.D12, l.D21, l.D22
	}
	m[block].Set(0, 0, val)
	return sys
}

func TestAnalysisRejectsNonFiniteModel(t *testing.T) {
	one := mat.NewDense(1, 1, []float64{1})
	err1 := func(_ any, err error) error { return err }
	entries := []struct {
		name string
		call func(*System) error
	}{
		{"Norm2", func(s *System) error { return err1(Norm(s, 2)) }},
		{"NormInf", func(s *System) error { return err1(Norm(s, math.Inf(1))) }},
		{"H2Norm", func(s *System) error { return err1(H2Norm(s)) }},
		{"HinfNorm", func(s *System) error { _, _, err := HinfNorm(s); return err }},
		{"HSV", func(s *System) error { return err1(HSV(s)) }},
		{"Gram", func(s *System) error { return err1(Gram(s, GramObservability)) }},
		{"Poles", func(s *System) error { return err1(s.Poles()) }},
		{"PolesDetail", func(s *System) error { return err1(s.PolesDetail()) }},
		{"IsStable", func(s *System) error { return err1(s.IsStable()) }},
		{"Damp", func(s *System) error { return err1(Damp(s)) }},
		{"Step", func(s *System) error { return err1(Step(s, 1)) }},
		{"Impulse", func(s *System) error { return err1(Impulse(s, 1)) }},
		{"Initial", func(s *System) error { return err1(Initial(s, mat.NewVecDense(2, []float64{1, -0.5}), 1)) }},
		{"Lsim", func(s *System) error {
			dt := 0.1
			if s.Dt > 0 {
				dt = s.Dt
			}
			return err1(Lsim(s, mat.NewDense(3, 1, []float64{1, 1, 1}), []float64{0, dt, 2 * dt}, nil))
		}},
		{"Simulate", func(s *System) error { return err1(s.Simulate(mat.NewDense(1, 3, []float64{1, 1, 1}), nil, nil)) }},
		{"Modred", func(s *System) error { return err1(Modred(s, []int{1}, MatchDC)) }},
		{"TransferFunction", func(s *System) error { return err1(s.TransferFunction(nil)) }},
		{"ZPKModel", func(s *System) error { return err1(s.ZPKModel(nil)) }},
		{"Covar", func(s *System) error { return err1(Covar(s, one)) }},
	}
	for _, e := range entries {
		for _, v := range nonFiniteVariants {
			if e.name == "Simulate" && v.dt == 0 {
				continue
			}
			for _, block := range v.blocks {
				for _, val := range []float64{math.NaN(), math.Inf(1)} {
					sys := nonFiniteModel(t, v, block, val)
					if err := e.call(sys); !errors.Is(err, ErrInvalidArgument) {
						t.Errorf("%s %s %s=%v: err = %v, want ErrInvalidArgument", e.name, v.name, block, val, err)
					}
				}
			}
		}
	}
}

func TestHinfNormRejectsInfFeedthroughWithHugeGains(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0.5, 0.2, 0, -0.3}),
		mat.NewDense(2, 1, []float64{0, math.Ldexp(1, 521)}),
		mat.NewDense(1, 2, []float64{math.Ldexp(1, 520), math.Ldexp(1, 520)}),
		mat.NewDense(1, 1, []float64{math.Inf(1)}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if norm, _, err := HinfNorm(sys); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("HinfNorm = %v, %v; want ErrInvalidArgument", norm, err)
	}
	if norm, err := Norm(sys, math.Inf(1)); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Norm(Inf) = %v, %v; want ErrInvalidArgument", norm, err)
	}
}

func TestCtrbObsvRejectNonFinite(t *testing.T) {
	for _, val := range []float64{math.NaN(), math.Inf(-1)} {
		for _, arg := range []string{"A", "X"} {
			A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
			B := mat.NewDense(2, 1, []float64{1, 0.5})
			C := mat.NewDense(1, 2, []float64{1, 0.3})
			if arg == "A" {
				A.Set(1, 0, val)
			} else {
				B.Set(1, 0, val)
				C.Set(0, 1, val)
			}
			if _, err := Ctrb(A, B); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("Ctrb %s=%v: err = %v, want ErrInvalidArgument", arg, val, err)
			}
			if _, err := Obsv(A, C); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("Obsv %s=%v: err = %v, want ErrInvalidArgument", arg, val, err)
			}
		}
	}
}
