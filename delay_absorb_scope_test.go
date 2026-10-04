package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const absorbScopeTol = 1e-12

var absorbScopeOmega = []float64{0.05, 0.4, 1.1, 2.3, 3.0}

func absorbScopePlant(t *testing.T, dt float64, internal, descriptor bool) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{0.5, 0.2, -0.1, -0.1, 0.3, 0.15, 0.05, -0.2, 0.4})
	B := mat.NewDense(3, 2, []float64{1, 0.3, 0.2, 1, -0.4, 0.5})
	C := mat.NewDense(2, 3, []float64{1, 0.4, -0.3, -0.2, 1, 0.6})
	D := mat.NewDense(2, 2, []float64{0.1, -0.3, 0.05, 0.2})
	if dt == 0 {
		A = mat.NewDense(3, 3, []float64{-1.5, 0.2, -0.1, -0.1, -0.8, 0.15, 0.05, -0.2, -2})
	}
	var sys *System
	var err error
	if descriptor {
		E := mat.NewDense(3, 3, []float64{1.5, 0.2, 0, 0, 0.8, -0.1, 0.1, 0, 1.2})
		sys, err = NewDescriptor(A, B, C, D, E, dt)
	} else {
		sys, err = New(A, B, C, D, dt)
	}
	if err != nil {
		t.Fatal(err)
	}
	if internal {
		tau := []float64{2, 1}
		if dt == 0 {
			tau = []float64{0.3, 0.7}
		}
		err = sys.SetInternalDelay(tau,
			mat.NewDense(3, 2, []float64{0.3, -0.2, 0.1, 0.25, -0.15, 0.05}),
			mat.NewDense(2, 3, []float64{0.4, 0.1, -0.2, -0.3, 0.2, 0.1}),
			mat.NewDense(2, 2, []float64{0.2, 0.1, -0.1, 0.3}),
			mat.NewDense(2, 2, []float64{0.1, 0.3, -0.2, 0.15}),
			mat.NewDense(2, 2, []float64{0.2, -0.1, 0.05, 0.1}))
		if err != nil {
			t.Fatal(err)
		}
	}
	return sys
}

type absorbScopeDelays struct {
	name  string
	in    []float64
	out   []float64
	io    []float64
	resid bool
}

var absorbScopeCases = []absorbScopeDelays{
	{name: "decomposable", in: []float64{1, 0}, out: []float64{0, 2}, io: []float64{1, 2, 3, 4}},
	{name: "residual", in: []float64{1, 0}, out: []float64{0, 2}, io: []float64{1, 0, 0, 3}, resid: true},
}

func (c absorbScopeDelays) apply(t *testing.T, sys *System, scale float64) {
	t.Helper()
	sc := func(v []float64) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = x * scale
		}
		return out
	}
	if err := sys.SetInputDelay(sc(c.in)); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay(sc(c.out)); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, sc(c.io))); err != nil {
		t.Fatal(err)
	}
	_, _, residual := DecomposeIODelay(sys.Delay)
	if got := delayMatrixHasNonzero(residual); got != c.resid {
		t.Fatalf("residual split = %v, want %v", got, c.resid)
	}
}

func assertFreqClose(t *testing.T, label string, want, got *System) {
	t.Helper()
	fw, err := want.FreqResponse(absorbScopeOmega)
	if err != nil {
		t.Fatalf("%s: reference FreqResponse: %v", label, err)
	}
	fg, err := got.FreqResponse(absorbScopeOmega)
	if err != nil {
		t.Fatalf("%s: absorbed FreqResponse: %v", label, err)
	}
	for k := range fw.Data {
		if d := cmplx.Abs(fw.Data[k] - fg.Data[k]); d > absorbScopeTol {
			t.Fatalf("%s: freq response entry %d differs by %g", label, k, d)
		}
	}
}

func assertSimulateClose(t *testing.T, label string, want, got *System) {
	t.Helper()
	const steps = 40
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(0.3*float64(k))+1)
		u.Set(1, k, math.Cos(0.7*float64(k)))
	}
	rw, err := want.Simulate(u, nil, nil)
	if err != nil {
		t.Fatalf("%s: reference Simulate: %v", label, err)
	}
	rg, err := got.Simulate(u, nil, nil)
	if err != nil {
		t.Fatalf("%s: absorbed Simulate: %v", label, err)
	}
	if !mat.EqualApprox(rw.Y, rg.Y, absorbScopeTol) {
		t.Fatalf("%s: simulated outputs differ\nwant %v\ngot  %v", label, mat.Formatted(rw.Y), mat.Formatted(rg.Y))
	}
}

func assertAbsorbedScope(t *testing.T, label string, scope AbsorbScope, src, got *System) {
	t.Helper()
	if got.Delay != nil && delayMatrixHasNonzero(got.Delay) && scope != AbsorbInput && scope != AbsorbOutput {
		t.Fatalf("%s: IO delay left after %s", label, scope)
	}
	if scope == AbsorbAll {
		if got.HasInternalDelay() {
			t.Fatalf("%s: internal delay left after AbsorbAll", label)
		}
		return
	}
	if !got.HasInternalDelay() {
		t.Fatalf("%s: %s dropped internal delays", label, scope)
	}
	tau := map[float64]int{}
	for _, v := range got.LFT.Tau {
		tau[v]++
	}
	for _, v := range src.LFT.Tau {
		if tau[v] == 0 {
			t.Fatalf("%s: internal delay %g missing from %v", label, v, got.LFT.Tau)
		}
	}
}

func TestAbsorbDelayScopedDiscretePreservesInternalDelay(t *testing.T) {
	for _, c := range absorbScopeCases {
		for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbAll} {
			label := c.name + "/" + string(scope)
			sys := absorbScopePlant(t, 1, true, false)
			c.apply(t, sys, 1)
			got, err := sys.AbsorbDelay(scope)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			assertAbsorbedScope(t, label, scope, sys, got)
			assertFreqClose(t, label, sys, got)
			assertSimulateClose(t, label, sys, got)
		}
	}
}

func TestAbsorbDelayScopedDiscreteInternalOnlyInputs(t *testing.T) {
	sys := absorbScopePlant(t, 1, true, false)
	if err := sys.SetInputDelay([]float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	got, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	assertAbsorbedScope(t, "input", AbsorbInput, sys, got)
	assertFreqClose(t, "input", sys, got)
	assertSimulateClose(t, "input", sys, got)
}

func TestAbsorbDelayScopedDiscreteDescriptor(t *testing.T) {
	for _, c := range absorbScopeCases {
		for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbAll} {
			label := c.name + "/" + string(scope)
			sys := absorbScopePlant(t, 1, false, true)
			c.apply(t, sys, 1)
			got, err := sys.AbsorbDelay(scope)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			if !got.IsDescriptor() {
				t.Fatalf("%s: descriptor E dropped", label)
			}
			assertFreqClose(t, label, sys, got)
		}
	}
}

func TestAbsorbDelayInputDescriptorEIsBlockIdentity(t *testing.T) {
	sys := absorbScopePlant(t, 1, false, true)
	if err := sys.SetInputDelay([]float64{2, 1}); err != nil {
		t.Fatal(err)
	}
	got, err := sys.AbsorbDelay(AbsorbInput)
	if err != nil {
		t.Fatal(err)
	}
	if got.E == nil {
		t.Fatal("descriptor E dropped")
	}
	want := mat.NewDense(6, 6, nil)
	setBlock(want, 0, 0, sys.E)
	for i := 3; i < 6; i++ {
		want.Set(i, i, 1)
	}
	if !mat.Equal(want, got.E) {
		t.Fatalf("E = %v, want %v", mat.Formatted(got.E), mat.Formatted(want))
	}
}

// explicitWithLFT eliminates E including the internal-delay input matrix B2,
// which ToExplicit leaves untouched.
func explicitWithLFT(t *testing.T, sys *System) *System {
	t.Helper()
	var lu mat.LU
	lu.Factorize(sys.E)
	out := sys.Copy()
	out.E = nil
	for _, pair := range [][2]*mat.Dense{{out.A, sys.A}, {out.B, sys.B}, {out.LFT.B2, sys.LFT.B2}} {
		if err := lu.SolveTo(pair[0], false, pair[1]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func TestAbsorbDelayScopedDiscreteDescriptorInternalDelay(t *testing.T) {
	for _, c := range absorbScopeCases {
		for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO, AbsorbInternal, AbsorbAll} {
			label := c.name + "/" + string(scope)
			sys := absorbScopePlant(t, 1, true, true)
			c.apply(t, sys, 1)
			got, err := sys.AbsorbDelay(scope)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			if !got.IsDescriptor() {
				t.Fatalf("%s: descriptor E dropped", label)
			}
			want := explicitWithLFT(t, sys)
			if !got.HasInternalDelay() {
				got, err = got.ToExplicit()
				if err != nil {
					t.Fatal(err)
				}
			} else {
				got = explicitWithLFT(t, got)
			}
			assertFreqClose(t, label, want, got)
			assertSimulateClose(t, label, want, got)
		}
	}
}

func padeColumnResponse(t *testing.T, delays []float64, omega float64) []complex128 {
	t.Helper()
	out := make([]complex128, len(delays))
	for i, tau := range delays {
		if tau == 0 {
			out[i] = 1
			continue
		}
		pd, err := PadeDelay(tau, DefaultPadeOrder)
		if err != nil {
			t.Fatal(err)
		}
		fr, err := pd.FreqResponse([]float64{omega})
		if err != nil {
			t.Fatal(err)
		}
		out[i] = fr.Data[0]
	}
	return out
}

func TestAbsorbDelayScopedContinuousPreservesInternalDelay(t *testing.T) {
	in := []float64{0.2, 0}
	out := []float64{0, 0.15}
	for _, scope := range []AbsorbScope{AbsorbInput, AbsorbOutput, AbsorbIO} {
		sys := absorbScopePlant(t, 0, true, false)
		if err := sys.SetInputDelay(in); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay(out); err != nil {
			t.Fatal(err)
		}
		if scope == AbsorbIO {
			if err := sys.SetDelay(mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 0.4})); err != nil {
				t.Fatal(err)
			}
		}
		got, err := sys.AbsorbDelay(scope)
		if err != nil {
			t.Fatalf("%s: %v", scope, err)
		}
		assertAbsorbedScope(t, string(scope), scope, sys, got)

		inPade, outPade := make([]float64, 2), make([]float64, 2)
		rest := sys.Copy()
		switch scope {
		case AbsorbInput:
			inPade, rest.InputDelay = in, nil
		case AbsorbOutput:
			outPade, rest.OutputDelay = out, nil
		case AbsorbIO:
			di, do, _ := DecomposeIODelay(sys.Delay)
			for j := range 2 {
				inPade[j] = in[j] + di[j]
				outPade[j] = out[j] + do[j]
			}
			rest.InputDelay, rest.OutputDelay, rest.Delay = nil, nil, nil
		}
		for _, w := range absorbScopeOmega {
			base, err := rest.FreqResponse([]float64{w})
			if err != nil {
				t.Fatal(err)
			}
			fg, err := got.FreqResponse([]float64{w})
			if err != nil {
				t.Fatal(err)
			}
			pin := padeColumnResponse(t, inPade, w)
			pout := padeColumnResponse(t, outPade, w)
			for i := range 2 {
				for j := range 2 {
					want := pout[i] * base.At(0, i, j) * pin[j]
					if d := cmplx.Abs(want - fg.At(0, i, j)); d > absorbScopeTol {
						t.Fatalf("%s: w=%g (%d,%d) differs by %g", scope, w, i, j, d)
					}
				}
			}
		}
	}
}

func TestAbsorbDelayContinuousResidualKeepsInternalDelay(t *testing.T) {
	sys := absorbScopePlant(t, 0, true, false)
	absorbScopeCases[1].apply(t, sys, 0.1)
	got, err := sys.AbsorbDelay(AbsorbIO)
	if err != nil {
		t.Fatal(err)
	}
	assertAbsorbedScope(t, "residual", AbsorbIO, sys, got)
}

func TestAbsorbDelayContinuousDescriptorDelayFreeKeepsE(t *testing.T) {
	sys := absorbScopePlant(t, 0, false, true)
	got, err := sys.AbsorbDelay(AbsorbIO)
	if err != nil || !got.IsDescriptor() {
		t.Fatalf("delay-free descriptor: err=%v descriptor=%v", err, got != nil && got.IsDescriptor())
	}
}
