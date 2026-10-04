package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func assertResponsesMatch(t *testing.T, label string, want, got *System, omega []float64) {
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
		if d := cmplx.Abs(fw.Data[k] - fg.Data[k]); d > 1e-12*max(1, cmplx.Abs(fw.Data[k])) {
			t.Fatalf("%s: FreqResponse entry %d differs by %g", label, k, d)
		}
	}
	td := newTimeDomain(want.Dt)
	for _, w := range omega {
		s := td.frequencyVariable(w)
		ew, err := want.EvalFr(s)
		if err != nil {
			t.Fatalf("%s: reference EvalFr: %v", label, err)
		}
		eg, err := got.EvalFr(s)
		if err != nil {
			t.Fatalf("%s: EvalFr: %v", label, err)
		}
		for i := range ew {
			for j := range ew[i] {
				if d := cmplx.Abs(ew[i][j] - eg[i][j]); d > 1e-12*max(1, cmplx.Abs(ew[i][j])) {
					t.Fatalf("%s: EvalFr(%v)[%d][%d] differs by %g", label, s, i, j, d)
				}
			}
		}
	}
}

func TestFreqResponseDescriptorInternalDelayMatchesExplicit(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		for _, ioDelay := range []bool{false, true} {
			sys := absorbScopePlant(t, dt, true, true)
			if ioDelay {
				scale := 0.125
				if dt > 0 {
					scale = 1
				}
				absorbScopeCases[1].apply(t, sys, scale)
			}
			explicit, err := sys.ToExplicit()
			if err != nil {
				t.Fatal(err)
			}
			label := map[bool]string{false: "continuous", true: "discrete"}[dt > 0]
			if ioDelay {
				label += "/io"
			}
			assertResponsesMatch(t, label, explicit, sys, absorbScopeOmega)
		}
	}
}

var singularLFTCases = []struct {
	dt  float64
	tau []float64
}{{0, []float64{0.35}}, {1, []float64{2}}}

// singularLFTPair returns a singular-E internal-delay model and its hand
// reduction. E = diag(1,0) makes x2 algebraic: 0 = 0.3x1 - x2 + 0.4u - 0.3w.
func singularLFTPair(t *testing.T, dt float64, tau []float64) (desc, reduced *System) {
	t.Helper()
	desc, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0.3, -1}),
		mat.NewDense(2, 1, []float64{1, 0.4}),
		mat.NewDense(1, 2, []float64{1, 2}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}),
		dt)
	if err != nil {
		t.Fatal(err)
	}
	if dt > 0 {
		desc.A.Set(0, 0, 0.4)
	}
	err = desc.SetInternalDelay(tau,
		mat.NewDense(2, 1, []float64{0.2, -0.3}),
		mat.NewDense(1, 2, []float64{0.4, -0.6}),
		mat.NewDense(1, 1, []float64{0.05}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.15}))
	if err != nil {
		t.Fatal(err)
	}
	a := desc.A.At(0, 0) + 0.5*0.3
	reduced, err = New(
		mat.NewDense(1, 1, []float64{a}),
		mat.NewDense(1, 1, []float64{1 + 0.5*0.4}),
		mat.NewDense(1, 1, []float64{1 + 2*0.3}),
		mat.NewDense(1, 1, []float64{0.1 + 2*0.4}),
		dt)
	if err != nil {
		t.Fatal(err)
	}
	err = reduced.SetInternalDelay(tau,
		mat.NewDense(1, 1, []float64{0.2 - 0.5*0.3}),
		mat.NewDense(1, 1, []float64{0.4 - 0.6*0.3}),
		mat.NewDense(1, 1, []float64{0.05 - 2*0.3}),
		mat.NewDense(1, 1, []float64{0.1 - 0.6*0.4}),
		mat.NewDense(1, 1, []float64{0.15 + 0.6*0.3}))
	if err != nil {
		t.Fatal(err)
	}
	return desc, reduced
}

func TestFreqResponseSingularDescriptorInternalDelay(t *testing.T) {
	for _, tc := range singularLFTCases {
		desc, reduced := singularLFTPair(t, tc.dt, tc.tau)
		if _, err := desc.ToExplicit(); err == nil {
			t.Fatal("singular E unexpectedly has an explicit form")
		}
		assertResponsesMatch(t, map[bool]string{false: "continuous", true: "discrete"}[tc.dt > 0], reduced, desc, absorbScopeOmega)
	}
}

// At DC every internal delay is unity, so w = z closes algebraically:
// z = (C2x + D21u)/(1-D22). The delay-free closure is an independent oracle.
func TestDCGainInternalDelayAndDescriptor(t *testing.T) {
	for _, tc := range singularLFTCases {
		desc, reduced := singularLFTPair(t, tc.dt, tc.tau)
		l := reduced.LFT
		k := 1 / (1 - l.D22.At(0, 0))
		b2, c2, d12, d21 := l.B2.At(0, 0), l.C2.At(0, 0), l.D12.At(0, 0), l.D21.At(0, 0)
		closed, err := New(
			mat.NewDense(1, 1, []float64{reduced.A.At(0, 0) + b2*k*c2}),
			mat.NewDense(1, 1, []float64{reduced.B.At(0, 0) + b2*k*d21}),
			mat.NewDense(1, 1, []float64{reduced.C.At(0, 0) + d12*k*c2}),
			mat.NewDense(1, 1, []float64{reduced.D.At(0, 0) + d12*k*d21}),
			tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		want, err := closed.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		for name, sys := range map[string]*System{"reduced": reduced, "descriptor": desc} {
			got, err := sys.DCGain()
			if err != nil {
				t.Fatalf("dt=%g %s: %v", tc.dt, name, err)
			}
			if d := got.At(0, 0) - want.At(0, 0); d > 1e-12 || d < -1e-12 {
				t.Fatalf("dt=%g %s: DCGain = %g, want %g", tc.dt, name, got.At(0, 0), want.At(0, 0))
			}
		}
	}

	sys := absorbScopePlant(t, 1, false, true)
	explicit, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	want, err := explicit.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(got, want, 1e-12) {
		t.Fatalf("discrete descriptor DCGain = %v, want %v", mat.Formatted(got), mat.Formatted(want))
	}
}
