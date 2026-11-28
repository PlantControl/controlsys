package controlsys

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// internalDelayGateSystem is a MIMO continuous descriptor (n = 3, m = 2,
// p = 2) with N = 2 internal delays and the LFT blocks given; nil blocks
// are zero.
func internalDelayGateSystem(t *testing.T, c2, d21, d22 *mat.Dense) *System {
	t.Helper()
	sys, err := NewDescriptor(
		mat.NewDense(3, 3, []float64{-2, 0.7, 0.1, 0.3, -1.5, 0.4, -0.2, 0.5, -3}),
		mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1, 0.3, 0.6}),
		mat.NewDense(2, 3, []float64{1, -0.5, 0.2, 0.1, 0.8, -1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0, -0.2}),
		mat.NewDense(3, 3, []float64{1, 0.3, 0, -0.1, 0.9, 0.2, 0.05, 0, 1.2}),
		0)
	if err != nil {
		t.Fatal(err)
	}
	if c2 == nil {
		c2 = mat.NewDense(2, 3, nil)
	}
	if d21 == nil {
		d21 = mat.NewDense(2, 2, nil)
	}
	if d22 == nil {
		d22 = mat.NewDense(2, 2, nil)
	}
	if err := sys.SetInternalDelay([]float64{0.05, 0.12}, mat.NewDense(3, 2, []float64{0.5, -0.2, 0.1, 0.7, -0.3, 0.4}), c2, mat.NewDense(2, 2, []float64{0.3, 0, -0.1, 0.2}), d21, d22); err != nil {
		t.Fatal(err)
	}
	return sys
}

// lftClosedAtDC keeps the delay-free core where C2, D21 and D22 are zero
// (w ≡ 0) and otherwise closes w = z at DC into an (n+N)-state pencil whose
// DC gain is that of the LFT: on a MIMO model with two delays and each of
// C2, D21, D22 nonzero alone or together, it matches DCGain.
func TestLFTClosedAtDCMatchesDCGain(t *testing.T) {
	c2 := mat.NewDense(2, 3, []float64{0.4, -0.1, 0.2, 0, 0.3, -0.5})
	d21 := mat.NewDense(2, 2, []float64{0.2, -0.3, 0.1, 0.4})
	d22 := mat.NewDense(2, 2, []float64{0.1, 0.05, -0.2, 0.3})
	for _, tc := range []struct {
		name          string
		c2, d21, d22  *mat.Dense
		wantDelayFree bool
	}{
		{"w-free", nil, nil, nil, true},
		{"C2", c2, nil, nil, false},
		{"D21", nil, d21, nil, false},
		{"D22", nil, nil, d22, false},
		{"all", c2, d21, d22, false},
	} {
		sys := internalDelayGateSystem(t, tc.c2, tc.d21, tc.d22)
		ref, barRef := lftClosedAtDC(sys)
		n, _, _ := ref.Dims()
		if wantN := map[bool]int{true: 3, false: 5}[tc.wantDelayFree]; ref.LFT != nil || n != wantN {
			t.Errorf("%s: closed model has %d states, LFT %v; want %d", tc.name, n, ref.LFT != nil, wantN)
			continue
		}
		want, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: DCGain: %v", tc.name, err)
		}
		x0, e := 0.0, (*mat.Dense)(nil)
		if ref.IsDiscrete() {
			x0, e = 1, ref.E
		}
		if nb, _, _ := barRef.Dims(); nb != n || barRef.IsDiscrete() {
			t.Errorf("%s: bar model has %d states, discrete %v", tc.name, nb, barRef.IsDiscrete())
		}
		r := solveDCRegular(e, ref.A, ref.B, ref.C, ref.D, x0, false)
		if r.solveErr != nil || !r.certified {
			t.Fatalf("%s: closed DC solve not certified: %v", tc.name, r.solveErr)
		}
		for i := range 2 {
			for j := range 2 {
				if g, w := r.g.At(i, j), want.At(i, j); math.Abs(g-w) > 1e-13*math.Max(1, math.Abs(w)) {
					t.Errorf("%s: closed DC (%d,%d) = %v, DCGain %v", tc.name, i, j, g, w)
				}
			}
		}
		gate := newHoldGate(sys)
		if gate.err != nil || !gate.gated || gate.bar.m != 2 || len(gate.bar.terms) != 4 {
			t.Errorf("%s: gate err %v gated %v", tc.name, gate.err, gate.gated)
		}
	}
}

// C2D of an internal-delay descriptor by ZOH, FOH or Tustin is gated on
// its DC gain (PT5WKA): a nil result's DC, closed at z = 1, is within the
// gate's bar of DCGain, on MIMO models with two delays, with I/O delays
// beside them, and C2DMap's verdict is C2D's.
func TestC2DInternalDelayDescriptorDCGated(t *testing.T) {
	c2 := mat.NewDense(2, 3, []float64{0.4, -0.1, 0.2, 0, 0.3, -0.5})
	d21 := mat.NewDense(2, 2, []float64{0.2, -0.3, 0.1, 0.4})
	d22 := mat.NewDense(2, 2, []float64{0.1, 0.05, -0.2, 0.3})
	io := internalDelayGateSystem(t, c2, d21, d22)
	io.InputDelay = []float64{0.02, 0}
	io.OutputDelay = []float64{0, 0.03}
	for name, sys := range map[string]*System{
		"w-free": internalDelayGateSystem(t, nil, nil, nil),
		"all":    internalDelayGateSystem(t, c2, d21, d22),
		"io":     io,
	} {
		want, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: DCGain: %v", name, err)
		}
		for _, dt := range []float64{1e-3, 0.05} {
			for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
				key := fmt.Sprintf("%s@%g/%s", name, dt, meth)
				d, err := sys.C2D(dt, C2DOptions{Method: meth})
				if err != nil {
					t.Errorf("%s: C2D: %v", key, err)
					continue
				}
				h, _, _, ok := holdDC(delaysClosedAtDC(ioDelayFree(d)))
				if !ok {
					t.Errorf("%s: DC of the result not resolved", key)
					continue
				}
				for k := range 4 {
					if g, w := h.At(k/2, k%2), want.At(k/2, k%2); math.Abs(g-w) > 1e-12*math.Max(1, math.Abs(w)) {
						t.Errorf("%s: DC (%d,%d) = %v, DCGain %v", key, k/2, k%2, g, w)
					}
				}
			}
		}
	}
}

// A loop with I − D22 singular at DC (D22 = 1, w otherwise unexcited) has
// no DC solve of its closed pencil: C2D returns no nil (PT5WKA).
func TestC2DInternalDelayDescriptorSingularLoop(t *testing.T) {
	sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 0.2, 0, -2}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0.5}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, []float64{1, 0.1, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.1}, mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{1})); err != nil {
		t.Fatal(err)
	}
	for _, meth := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		d, err := sys.C2D(0.01, C2DOptions{Method: meth})
		if err == nil {
			t.Errorf("%s: C2D nil on a loop singular at DC", meth)
		}
		if d != nil && !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: model with %v", meth, err)
		}
	}
}

// fallbackGateSystem is the arbiter's constructed internal-delay
// descriptor whose closed DC pencil is near singular (d = 2⁻⁵²): the
// certified solve does not certify and the gate takes DCGain of the pencil
// (PT5WKA ruling F1).
func fallbackGateSystem(t *testing.T, c2 []float64) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(2, 2, []float64{-1, 1, 1, -(1 + 0x1p-52)}), mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 1, nil), mat.NewDense(2, 2, []float64{1, 0, 0, 2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.05}, mat.NewDense(2, 1, []float64{0.3, -0.7}), mat.NewDense(1, 2, c2),
		mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return sys
}

// Where the closed DC pencil's solve does not certify, the gate's g0 is
// DCGain of the pencil, not the uncertified evaluation of the LFT (off by
// 3.2e-4 relative here): with C2 = [2⁻⁵⁰, 0] it is the 320-bit value
// within ε·T0, and C2D's ZOH result is nil; with C2 = 0 (the delay-free
// core) DCGain reports a pole within rounding of DC and C2D is unverified.
func TestC2DInternalDelayDescriptorGateFallback(t *testing.T) {
	const g320, t0 = 7794691662756627.0, 7.79e15
	sys := fallbackGateSystem(t, []float64{0x1p-50, 0})
	gate := newHoldGate(sys)
	if gate.err != nil || gate.g0 == nil {
		t.Fatalf("C2 ≠ 0: gate %v", gate.err)
	}
	if g := gate.g0.At(0, 0); math.Abs(g-g320) > 0x1p-52*t0+1 {
		t.Errorf("C2 ≠ 0: gate g0 = %.17g, 320-bit %.17g", g, g320)
	}
	if _, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodZOH}); err != nil {
		t.Errorf("C2 ≠ 0: C2D ZOH: %v", err)
	}
	core := fallbackGateSystem(t, []float64{0, 0})
	if d, err := core.C2D(0.01, C2DOptions{Method: C2DMethodZOH}); d == nil || !errors.Is(err, ErrAccuracyUnverified) {
		t.Errorf("C2 = 0: C2D ZOH %v, want the model with ErrAccuracyUnverified", err)
	}
}

// The closed DC pencil takes I − D22 without rounding (PT5WKA ruling F2):
// with c = 1 − 2⁻⁵² and D22 = 2⁻⁵⁴, 1 − D22 rounds to 1 in float64, which
// moves g0 = c/(1 − D22 − c) = (1 − 2⁻⁵²)·2⁵⁴/3 by a third.
func TestC2DInternalDelayDescriptorGateExactLoop(t *testing.T) {
	sys, err := NewDescriptor(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	c := 1 - 0x1p-52
	if err := sys.SetInternalDelay([]float64{0.05}, mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{c}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{0x1p-54})); err != nil {
		t.Fatal(err)
	}
	want := c * 0x1p54 / 3
	gate := newHoldGate(sys)
	if gate.g0 == nil || gate.err != nil {
		t.Fatalf("gate: %v", gate.err)
	}
	if g := gate.g0.At(0, 0); math.Abs(g-want) > 1e-12*want {
		t.Errorf("gate g0 = %.17g, want %.17g", g, want)
	}
}

// C2D of the frozen internal-delay census rows (PT5WKA: strad1m-lft/212 on
// the band fold, strad2-lft/256 on the stiff parts, each with C2 = 0 and
// with a random C2) keeps its model and returns ErrAccuracyUnverified
// where the DC gain, closed at z = 1, is over the U-1 bar of the closed DC
// pencil (the core's for C2 = 0), and nil only within it; C2DMap returns
// C2D's model bitwise with C2D's class and its G.
func TestC2DInternalDelayDescriptorDCGateRows(t *testing.T) {
	for _, r := range internalDelayDCGateRows {
		sys, err := NewDescriptor(r.A, r.B, r.C, r.D, r.E, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInternalDelay(r.tau, r.B2, r.C2, r.D12, r.D21, r.D22); err != nil {
			t.Fatal(err)
		}
		for key, want := range r.classes {
			var dt float64
			var meth string
			if _, err := fmt.Sscanf(strings.Replace(key, "/", " ", 1), "%g %s", &dt, &meth); err != nil {
				t.Fatal(err)
			}
			opts := C2DOptions{Method: C2DMethod(meth)}
			name := r.name + "@" + key
			d, err := sys.C2D(dt, opts)
			if d == nil || (err != nil && !errors.Is(err, ErrAccuracyUnverified)) {
				t.Errorf("%s: C2D model %v, err %v", name, d != nil, err)
				continue
			}
			got := byte('N')
			if err != nil {
				got = 'U'
			}
			if want != '?' && got != want {
				t.Errorf("%s: C2D class %c (%v), want %c", name, got, err, want)
			}
			if got == 'N' {
				h, _, allow, ok := holdDC(delaysClosedAtDC(d))
				if !ok {
					t.Errorf("%s: DC of the result not resolved", name)
				} else {
					a := 0.0
					if allow != nil && allow[0] <= math.Sqrt(0x1p-52)*math.Max(math.Abs(r.g0), r.t0) {
						a = allow[0]
					}
					if diff := math.Abs(h.At(0, 0) - r.g0); diff > r.bar0+a {
						t.Errorf("%s: nil with DC %v off the 320-bit %v by %.3g bars", name, h.At(0, 0), r.g0, diff/(r.bar0+a))
					}
				}
			}
			dm, g, merr := sys.C2DMap(dt, opts)
			if dm == nil || g == nil {
				t.Errorf("%s: C2DMap: %v", name, merr)
				continue
			}
			if errClass(dm, merr) != errClass(d, err) || !bitsEqualSystem(dm, d) || !bitsEqualLFT(dm, d) {
				t.Errorf("%s: C2DMap %v, C2D %v, same model %v", name, merr, err, bitsEqualSystem(dm, d) && bitsEqualLFT(dm, d))
			}
		}
	}
}
