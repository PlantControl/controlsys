package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestAbsorbDelayContinuousKeepsIODelayMatrix(t *testing.T) {
	in := []float64{0.5, 1}
	out := []float64{0.3, 0}
	iod := []float64{0, 0.2, 0.4, 0}
	gain := func(t *testing.T) *System {
		sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	plants := map[string]func(*testing.T) *System{
		"gain":     gain,
		"dynamic":  func(t *testing.T) *System { return absorbScopePlant(t, 0, false, false) },
		"internal": func(t *testing.T) *System { return absorbScopePlant(t, 0, true, false) },
	}
	scopeSets := [][]AbsorbScope{
		{AbsorbInput},
		{AbsorbOutput},
		{AbsorbInput, AbsorbOutput},
		{AbsorbInternal, AbsorbInput},
		{AbsorbOutput, AbsorbInternal},
	}
	for name, mk := range plants {
		for _, set := range scopeSets {
			sys := mk(t)
			if err := sys.SetInputDelay(in); err != nil {
				t.Fatal(err)
			}
			if err := sys.SetOutputDelay(out); err != nil {
				t.Fatal(err)
			}
			if err := sys.SetDelay(mat.NewDense(2, 2, iod)); err != nil {
				t.Fatal(err)
			}
			label := name + " " + string(set[0])
			if len(set) > 1 {
				label += "+" + string(set[1])
			}
			got, err := sys.AbsorbDelay(set...)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}

			absorbIn, absorbOut, absorbInt := false, false, false
			for _, s := range set {
				absorbIn = absorbIn || s == AbsorbInput
				absorbOut = absorbOut || s == AbsorbOutput
				absorbInt = absorbInt || s == AbsorbInternal
			}
			if got.Delay == nil || !mat.Equal(got.Delay, sys.Delay) {
				t.Fatalf("%s: I/O-delay matrix not kept: %v", label, got.Delay)
			}
			wantInternal := sys.HasInternalDelay() && !absorbInt
			if got.HasInternalDelay() != wantInternal {
				t.Fatalf("%s: HasInternalDelay=%v, want %v (tau %v)", label, got.HasInternalDelay(), wantInternal, got.LFT)
			}
			if wantInternal && len(got.LFT.Tau) != len(sys.LFT.Tau) {
				t.Fatalf("%s: internal delays %v, want %v", label, got.LFT.Tau, sys.LFT.Tau)
			}
			if absorbIn == delaySliceHasNonzero(got.InputDelay) {
				t.Fatalf("%s: input delay %v", label, got.InputDelay)
			}
			if absorbOut == delaySliceHasNonzero(got.OutputDelay) {
				t.Fatalf("%s: output delay %v", label, got.OutputDelay)
			}

			rest := sys.Copy()
			rest.InputDelay, rest.OutputDelay, rest.Delay = nil, nil, nil
			if absorbInt && rest.HasInternalDelay() {
				if rest, err = rest.AbsorbDelay(AbsorbInternal); err != nil {
					t.Fatal(err)
				}
			}
			zero := []float64{0, 0}
			inPade, outPade, inExact, outExact := zero, zero, in, out
			if absorbIn {
				inPade, inExact = in, zero
			}
			if absorbOut {
				outPade, outExact = out, zero
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
						tau := outExact[i] + iod[i*2+j] + inExact[j]
						want := pout[i] * base.At(0, i, j) * pin[j] * cmplx.Exp(complex(0, -w*tau))
						if d := cmplx.Abs(want - fg.At(0, i, j)); d > 1e-11 {
							t.Fatalf("%s: w=%g (%d,%d) differs by %g", label, w, i, j, d)
						}
					}
				}
			}
		}
	}
}

func BenchmarkAbsorbDelayContinuousInputKeepIO(b *testing.B) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 3, 4}), 0)
	if err != nil {
		b.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{0.5, 1}); err != nil {
		b.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, []float64{0, 0.2, 0.4, 0})); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.AbsorbDelay(AbsorbInput); err != nil {
			b.Fatal(err)
		}
	}
}
