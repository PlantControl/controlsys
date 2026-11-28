package controlsys

import (
	"errors"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func simulateOverflowPole10(t *testing.T) *System {
	t.Helper()
	sys, err := New(mat.NewDense(1, 1, []float64{10}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 1)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func simulateOnes(m, steps int) *mat.Dense {
	u := mat.NewDense(m, steps, nil)
	for i := range m {
		for k := range steps {
			u.Set(i, k, 1)
		}
	}
	return u
}

func requireSimulateOverflow(t *testing.T, name string, resp *Response, err error) {
	t.Helper()
	if !errors.Is(err, ErrOverflow) {
		t.Fatalf("%s: err = %v, want ErrOverflow", name, err)
	}
	if resp != nil {
		t.Fatalf("%s: response %+v came with %v", name, resp, err)
	}
	if !strings.HasPrefix(err.Error(), "Simulate: ") {
		t.Fatalf("%s: err %q lacks the Simulate prefix", name, err)
	}
}

func requireSimulateFinite(t *testing.T, name string, resp *Response, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: err = %v", name, err)
	}
	if resp.Y != nil && !allFiniteDense(resp.Y) {
		t.Fatalf("%s: non-finite Y with nil", name)
	}
	if resp.XFinal != nil {
		for i := range resp.XFinal.Len() {
			if !isFinite(resp.XFinal.AtVec(i)) {
				t.Fatalf("%s: XFinal[%d] = %g with nil", name, i, resp.XFinal.AtVec(i))
			}
		}
	}
}

// 3MIOII: z = 10 over 400 steps returned 90 Inf samples with nil.
func TestSimulateOverflowUnstableDiscrete(t *testing.T) {
	sys := simulateOverflowPole10(t)
	for _, fs := range []bool{false, true} {
		resp, err := sys.Simulate(simulateOnes(1, 400), nil, &SimulateOpts{FinalState: fs})
		requireSimulateOverflow(t, "400 steps", resp, err)
	}

	resp, err := sys.Simulate(simulateOnes(1, 300), nil, &SimulateOpts{FinalState: true})
	requireSimulateFinite(t, "300 steps", resp, err)
	if got := []float64{resp.Y.At(0, 0), resp.Y.At(0, 1), resp.Y.At(0, 2), resp.Y.At(0, 5)}; got[0] != 0 || got[1] != 1 || got[2] != 11 || got[3] != 11111 {
		t.Errorf("y[0,1,2,5] = %v, want [0 1 11 11111]", got)
	}
}

// 3MIOII: Y stays finite through sample 308 while the state after it is
// 1e309, so only a FinalState request overflows.
func TestSimulateOverflowFinalStateOnly(t *testing.T) {
	sys := simulateOverflowPole10(t)
	u := mat.NewDense(1, 309, nil)
	x0 := mat.NewVecDense(1, []float64{1})

	resp, err := sys.Simulate(u, x0, nil)
	requireSimulateFinite(t, "FinalState off", resp, err)
	if y := resp.Y.At(0, 308); !(y > 1e307) {
		t.Errorf("y[308] = %g, want about 1e308", y)
	}

	resp, err = sys.Simulate(u, x0, &SimulateOpts{FinalState: true})
	requireSimulateOverflow(t, "FinalState on", resp, err)

	resp, err = sys.Simulate(u.Slice(0, 1, 0, 308).(*mat.Dense), x0, &SimulateOpts{FinalState: true})
	requireSimulateFinite(t, "308 steps", resp, err)
	if x := resp.XFinal.AtVec(0); !(x > 1e307) {
		t.Errorf("XFinal = %g, want about 1e308", x)
	}
}

// 3MIOII: a model without outputs has Y nil, so only XFinal can overflow.
func TestSimulateOverflowNoOutputsFinalState(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{10}), mat.NewDense(1, 1, []float64{1}), nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := sys.Simulate(simulateOnes(1, 400), nil, &SimulateOpts{FinalState: true})
	requireSimulateOverflow(t, "no outputs", resp, err)

	resp, err = sys.Simulate(simulateOnes(1, 5), nil, &SimulateOpts{FinalState: true})
	requireSimulateFinite(t, "no outputs, 5 steps", resp, err)
	if x := resp.XFinal.AtVec(0); x != 11111 {
		t.Errorf("XFinal = %g, want 11111", x)
	}
}

// 3MIOII: each simulation route (I/O delay, internal delay, descriptor
// with invertible E, model without inputs) is bounded.
func TestSimulateOverflowRoutes(t *testing.T) {
	delayed := simulateOverflowPole10(t)
	if err := delayed.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	internal := simulateOverflowPole10(t)
	if err := internal.SetInternalDelay([]float64{1},
		mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0})); err != nil {
		t.Fatal(err)
	}
	desc, err := NewDescriptor(mat.NewDense(1, 1, []float64{20}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{2}), 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		sys  *System
	}{{"input delay", delayed}, {"internal delay", internal}, {"descriptor", desc}} {
		resp, err := c.sys.Simulate(simulateOnes(1, 400), nil, nil)
		requireSimulateOverflow(t, c.name, resp, err)
		resp, err = c.sys.Simulate(simulateOnes(1, 20), nil, nil)
		requireSimulateFinite(t, c.name+", 20 steps", resp, err)
	}

	free, err := New(mat.NewDense(1, 1, []float64{10}), nil, mat.NewDense(1, 1, []float64{1}), nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	x0 := mat.NewVecDense(1, []float64{1})
	resp, err := free.Simulate(nil, x0, &SimulateOpts{Steps: 400})
	requireSimulateOverflow(t, "no inputs", resp, err)
	resp, err = free.Simulate(nil, x0, &SimulateOpts{Steps: 3})
	requireSimulateFinite(t, "no inputs, 3 steps", resp, err)
	if got := resp.Y.RawMatrix().Data; got[0] != 1 || got[1] != 10 || got[2] != 100 {
		t.Errorf("Y = %v, want [1 10 100]", got)
	}
}

// 3MIOII: continuous unstable plant (the 5EKNHC A = [-0.005 0.3; 0 3]),
// ZOH-discretized at 0.1 s and simulated over 300 s, passes e^709. The
// continuous model itself stays ErrWrongDomain, and Lsim of it keeps its own
// Lsim-prefixed ErrOverflow.
func TestSimulateOverflowDiscretizedContinuous(t *testing.T) {
	csys, err := New(mat.NewDense(2, 2, []float64{-0.005, 0.3, 0, 3}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := csys.Simulate(simulateOnes(1, 10), nil, nil); !errors.Is(err, ErrWrongDomain) {
		t.Fatalf("continuous Simulate: err = %v, want ErrWrongDomain", err)
	}
	dsys, err := csys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := dsys.Simulate(simulateOnes(1, 3000), nil, nil)
	requireSimulateOverflow(t, "C2D 3000 steps", resp, err)
	resp, err = dsys.Simulate(simulateOnes(1, 50), nil, nil)
	requireSimulateFinite(t, "C2D 50 steps", resp, err)

	tt := make([]float64, 3001)
	for k := range tt {
		tt[k] = 0.1 * float64(k)
	}
	lresp, err := Lsim(csys, simulateOnes(3001, 1), tt, nil)
	if !errors.Is(err, ErrOverflow) || lresp != nil || !strings.HasPrefix(err.Error(), "Lsim: ") {
		t.Fatalf("Lsim: resp %v, err %v, want nil and Lsim: …ErrOverflow", lresp, err)
	}
}
