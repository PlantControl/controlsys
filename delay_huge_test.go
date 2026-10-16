package controlsys_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

// hugeDelays straddle the int64 range: the largest float64 below 2^63, 2^63
// itself, and values far past it. Each exceeds any simulation horizon.
var hugeDelays = []float64{1e19, math.Nextafter(math.Exp2(63), 0), math.Exp2(63), 1e300}

func hugeDelayFirstOrder(t *testing.T) *controlsys.System {
	t.Helper()
	sys, err := controlsys.New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), 1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func hugeDelayOnes(steps int) *mat.Dense {
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	return u
}

func hugeDelayRequireZero(t *testing.T, label string, y *mat.Dense) {
	t.Helper()
	if y == nil {
		t.Fatalf("%s: nil Y", label)
	}
	r, c := y.Dims()
	for i := range r {
		for k := range c {
			if v := y.At(i, k); v != 0 {
				t.Fatalf("%s: y(%d,%d)=%g, want 0", label, i, k, v)
			}
		}
	}
}

func TestSimulateHugeIODelayYieldsNoContribution(t *testing.T) {
	const steps = 6
	x0 := mat.NewVecDense(1, []float64{1})
	for _, d := range hugeDelays {
		label := fmt.Sprintf("d=%g", d)

		in := hugeDelayFirstOrder(t)
		if err := in.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		resp, err := in.Simulate(hugeDelayOnes(steps), nil, &controlsys.SimulateOpts{FinalState: true})
		if err != nil {
			t.Fatalf("%s input: %v", label, err)
		}
		hugeDelayRequireZero(t, label+" input Y", resp.Y)
		if v := resp.XFinal.AtVec(0); v != 0 {
			t.Fatalf("%s input: XFinal=%g, want 0", label, v)
		}

		out := hugeDelayFirstOrder(t)
		if err := out.SetOutputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		resp, err = out.Simulate(hugeDelayOnes(steps), x0, nil)
		if err != nil {
			t.Fatalf("%s output: %v", label, err)
		}
		hugeDelayRequireZero(t, label+" output Y", resp.Y)

		io, err := controlsys.NewWithDelay(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{d}), 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := io.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		resp, err = io.Simulate(hugeDelayOnes(steps), nil, &controlsys.SimulateOpts{FinalState: true})
		if err != nil {
			t.Fatalf("%s io: %v", label, err)
		}
		hugeDelayRequireZero(t, label+" io Y", resp.Y)
		if v := resp.XFinal.AtVec(0); v != 0 {
			t.Fatalf("%s io: XFinal=%g, want 0", label, v)
		}
	}
}

// TestSimulateIODelaySumPastInt64 splits a delay whose input share and
// InputDelay each fit in int64 but whose sum does not.
func TestSimulateIODelaySumPastInt64(t *testing.T) {
	half := math.Exp2(62)
	sys, err := controlsys.NewWithDelay(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{half}), 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{half}); err != nil {
		t.Fatal(err)
	}
	resp, err := sys.Simulate(hugeDelayOnes(4), nil, &controlsys.SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	hugeDelayRequireZero(t, "Y", resp.Y)
	if v := resp.XFinal.AtVec(0); v != 0 {
		t.Fatalf("XFinal=%g, want 0", v)
	}
}

// TestSimulateHugeInternalDelay: y = u + w, z = u, w = z delayed by d, so
// within any horizon shorter than d the response is y = u.
func TestSimulateHugeInternalDelay(t *testing.T) {
	u := mat.NewDense(1, 4, []float64{1, -2, 3, 0.5})
	for _, d := range hugeDelays {
		H, err := controlsys.NewGain(mat.NewDense(2, 2, []float64{1, 1, 1, 0}), 1)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := controlsys.SetDelayModel(H, []float64{d})
		if err != nil {
			t.Fatal(err)
		}
		for _, withIO := range []bool{false, true} {
			s := sys
			if withIO {
				s = sys.Copy()
				if err := s.SetInputDelay([]float64{d}); err != nil {
					t.Fatal(err)
				}
			}
			resp, err := s.Simulate(u, nil, nil)
			if err != nil {
				t.Fatalf("d=%g io=%v: %v", d, withIO, err)
			}
			for k := range 4 {
				want := u.At(0, k)
				if withIO {
					want = 0
				}
				if got := resp.Y.At(0, k); got != want {
					t.Fatalf("d=%g io=%v: y[%d]=%g, want %g", d, withIO, k, got, want)
				}
			}
		}
	}
}

func TestFreqResponseHugeDiscreteDelayUnitModulus(t *testing.T) {
	omega := []float64{0.3, 1, 2.5}
	for _, d := range hugeDelays {
		label := fmt.Sprintf("d=%g", d)
		gain, err := controlsys.NewGain(mat.NewDense(1, 1, []float64{2}), 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := gain.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		fr, err := gain.FreqResponse(omega)
		if err != nil {
			t.Fatalf("%s io: %v", label, err)
		}
		for k := range omega {
			if got := cmplx.Abs(fr.At(k, 0, 0)); math.Abs(got-2) > 1e-14 {
				t.Fatalf("%s io: |G(%g)|=%g, want 2", label, omega[k], got)
			}
		}

		H, err := controlsys.NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 0.5}), 1)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := controlsys.SetDelayModel(H, []float64{d})
		if err != nil {
			t.Fatal(err)
		}
		fr, err = sys.FreqResponse(omega)
		if err != nil {
			t.Fatalf("%s internal: %v", label, err)
		}
		for k := range omega {
			g := fr.At(k, 0, 0)
			if cmplx.IsNaN(g) || cmplx.IsInf(g) {
				t.Fatalf("%s internal: G(%g)=%v", label, omega[k], g)
			}
			delta := g / (1 + 0.5*g)
			if math.Abs(cmplx.Abs(delta)-1) > 1e-12 {
				t.Fatalf("%s internal: |z^-d|=%g, want 1", label, cmplx.Abs(delta))
			}
		}
	}
}

// TestFreqResponseSmallDiscreteDelayPhase pins z^-d = e^{-jωd} for ordinary
// delays alongside the huge-delay evaluation.
func TestFreqResponseSmallDiscreteDelayPhase(t *testing.T) {
	omega := []float64{0.3, 1, 2.5}
	for _, d := range []float64{1, 3, 17} {
		gain, err := controlsys.NewGain(mat.NewDense(1, 1, []float64{1}), 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := gain.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		H, err := controlsys.NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 0}), 1)
		if err != nil {
			t.Fatal(err)
		}
		internal, err := controlsys.SetDelayModel(H, []float64{d})
		if err != nil {
			t.Fatal(err)
		}
		for _, sys := range []*controlsys.System{gain, internal} {
			fr, err := sys.FreqResponse(omega)
			if err != nil {
				t.Fatal(err)
			}
			for k, w := range omega {
				want := cmplx.Exp(complex(0, -w*d))
				if got := fr.At(k, 0, 0); cmplx.Abs(got-want) > 1e-13 {
					t.Fatalf("d=%g w=%g: G=%v, want %v", d, w, got, want)
				}
			}
		}
	}
}

func TestTransferFuncEvalHugeDiscreteDelay(t *testing.T) {
	for _, d := range hugeDelays {
		tf := &controlsys.TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1}}, Delay: [][]float64{{d}}, Dt: 1}
		for _, tc := range []struct {
			s    complex128
			want complex128
		}{{2, 0}, {1, 1}} {
			h, err := tf.Eval(tc.s)
			if err != nil {
				t.Fatal(err)
			}
			if got := h[0][0]; cmplx.Abs(got-tc.want) > 1e-15 {
				t.Fatalf("d=%g s=%v: H=%v, want %v", d, tc.s, got, tc.want)
			}
		}
	}
	tf := &controlsys.TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1}}, Delay: [][]float64{{3}}, Dt: 1}
	h, err := tf.Eval(2)
	if err != nil {
		t.Fatal(err)
	}
	if got := h[0][0]; cmplx.Abs(got-0.125) > 1e-16 {
		t.Fatalf("d=3 s=2: H=%v, want 0.125", got)
	}
}

func TestContinuousResponseHugeDelay(t *testing.T) {
	for _, d := range []float64{1e19, 1e300} {
		sys, err := controlsys.New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		resp, err := controlsys.Initial(sys, mat.NewVecDense(1, []float64{1}), 3)
		if err != nil {
			t.Fatalf("d=%g Initial: %v", d, err)
		}
		hugeDelayRequireZero(t, fmt.Sprintf("d=%g Initial", d), resp.Y)

		H, err := controlsys.New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{1, 0.5}),
			mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		internal, err := controlsys.SetDelayModel(H, []float64{1})
		if err != nil {
			t.Fatal(err)
		}
		if err := internal.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		step, err := controlsys.Step(internal, 3)
		if err != nil {
			t.Fatalf("d=%g Step: %v", d, err)
		}
		hugeDelayRequireZero(t, fmt.Sprintf("d=%g Step", d), step.Y)
	}
}

// TestNonFiniteDelayFieldsRejected assigns NaN and ±Inf straight to the
// exported delay fields, bypassing the setters.
func TestNonFiniteDelayFieldsRejected(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		for _, field := range []string{"InputDelay", "OutputDelay", "Delay", "Tau"} {
			label := fmt.Sprintf("%s=%g", field, v)
			H, err := controlsys.NewGain(mat.NewDense(2, 2, []float64{1, 1, 1, 0}), 1)
			if err != nil {
				t.Fatal(err)
			}
			sys, err := controlsys.SetDelayModel(H, []float64{2})
			if err != nil {
				t.Fatal(err)
			}
			switch field {
			case "InputDelay":
				sys.InputDelay = []float64{v}
			case "OutputDelay":
				sys.OutputDelay = []float64{v}
			case "Delay":
				sys.Delay = mat.NewDense(1, 1, []float64{v})
			case "Tau":
				sys.LFT.Tau = []float64{v}
			}
			if _, err := sys.Simulate(hugeDelayOnes(3), nil, nil); err == nil {
				t.Errorf("%s: Simulate accepted", label)
			}
			if _, err := sys.FreqResponse([]float64{1}); err == nil {
				t.Errorf("%s: FreqResponse accepted", label)
			}
			if _, err := sys.EvalFr(2); err == nil {
				t.Errorf("%s: EvalFr accepted", label)
			}
		}
		tf := &controlsys.TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1}}, Delay: [][]float64{{v}}, Dt: 1}
		if _, err := tf.Eval(2); err == nil {
			t.Errorf("TransferFunc delay %g: Eval accepted", v)
		}
	}
}
