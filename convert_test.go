package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func makeTestSystem() *System {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	return sys
}

func TestDiscretize_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestDiscretize_Roundtrip(t *testing.T) {
	orig := makeTestSystem()
	dt := 0.01

	disc, err := orig.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.Dt != dt {
		t.Fatalf("Dt = %v, want %v", disc.Dt, dt)
	}

	rec, err := disc.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !rec.IsContinuous() {
		t.Fatal("expected continuous")
	}

	tol := 1e-10
	assertMatClose(t, "A", rec.A, orig.A, tol)
	assertMatClose(t, "B", rec.B, orig.B, tol)
	assertMatClose(t, "C", rec.C, orig.C, tol)
	assertMatClose(t, "D", rec.D, orig.D, tol)
}

func TestDiscretize_EigenvalueMapping(t *testing.T) {
	sys := makeTestSystem()
	contPoles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range contPoles {
		if real(p) >= 0 {
			t.Fatalf("continuous pole %v has non-negative real part", p)
		}
	}

	disc, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	discPoles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range discPoles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("discrete pole %v has |z| >= 1", p)
		}
	}
}

func TestDiscretize_SingularTransform(t *testing.T) {
	dt := 1.0
	beta := 2.0 / dt
	A := mat.NewDense(2, 2, []float64{beta, 0, 0, -1})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	_, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err == nil {
		t.Fatal("expected ErrSingularTransform")
	}
}

func TestDiscretize_N0_GainOnly(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  D,
		Dt: 0,
	}

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "D", disc.D, D, 1e-15)
	n, _, _ := disc.Dims()
	if n != 0 {
		t.Fatalf("expected n=0, got %d", n)
	}
}

func TestDiscretize_N_GT0_M0(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	sys, _ := New(A, nil, nil, nil, 0)

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	n, m, _ := disc.Dims()
	if n != 2 || m != 0 {
		t.Fatalf("expected n=2 m=0, got n=%d m=%d", n, m)
	}

	rec, err := disc.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "A", rec.A, A, 1e-10)
}

func TestUndiscretize_WrongDomain(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestUndiscretize_SingularTransform(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, 0})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0.1)

	_, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err == nil {
		t.Fatal("expected ErrSingularTransform for eigenvalue at -1")
	}
}

func TestZOH_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestZOH_N0_GainOnly(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{5})
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  D,
		Dt: 0,
	}

	disc, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "D", disc.D, D, 1e-15)
}

func TestZOH_ScalarAnalytical(t *testing.T) {
	a := -2.0
	b := 3.0
	dt := 0.1

	A := mat.NewDense(1, 1, []float64{a})
	B := mat.NewDense(1, 1, []float64{b})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	disc, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Analytical: Ad = exp(a*dt), Bd = (exp(a*dt)-1)/a * b
	adExpected := math.Exp(a * dt)
	bdExpected := (math.Exp(a*dt) - 1) / a * b

	assertMatClose(t, "Ad", disc.A, mat.NewDense(1, 1, []float64{adExpected}), 1e-12)
	assertMatClose(t, "Bd", disc.B, mat.NewDense(1, 1, []float64{bdExpected}), 1e-12)
	assertMatClose(t, "C", disc.C, C, 1e-15)
	assertMatClose(t, "D", disc.D, D, 1e-15)
}

func TestZOH_StabilityPreserved(t *testing.T) {
	sys := makeTestSystem()
	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Fatal("continuous system should be stable")
	}

	disc, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	dStable, err := disc.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !dStable {
		t.Fatal("ZOH-discretized system should be stable")
	}

	dPoles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range dPoles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("discrete pole %v has |z| >= 1", p)
		}
	}
}

func TestZOH_2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// For double integrator: expm([0 1; 0 0]*dt) = [1 dt; 0 1]
	// Full block: expm([0 1 0; 0 0 1; 0 0 0]*dt) = [1 dt dt^2/2; 0 1 dt; 0 0 1]
	// So Ad = [1 dt; 0 1], Bd = [dt^2/2; dt]
	AdExpected := mat.NewDense(2, 2, []float64{1, dt, 0, 1})
	BdExpected := mat.NewDense(2, 1, []float64{dt * dt / 2, dt})

	assertMatClose(t, "Ad", disc.A, AdExpected, 1e-12)
	assertMatClose(t, "Bd", disc.B, BdExpected, 1e-12)
}

func TestDiscretizeWithOpts_ZOH_IntegerInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeWithOpts_Tustin_IntegerOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 5 {
		t.Errorf("OutputDelay = %v, want [5]", disc.OutputDelay)
	}
}

func TestDiscretizeWithOpts_FractionalInputDelay_ExactZOH(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if disc.LFT == nil || disc.LFT.Tau[0] != 1 || disc.InputDelay[0] != 3 {
		t.Fatalf("exact delayed ZOH metadata: %+v", disc)
	}
}

func TestDiscretizeWithOpts_FractionalInputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 1 {
		t.Errorf("InputDelay = %v, want integer remainder [1]", disc.InputDelay)
	}
	n, _, _ := disc.Dims()
	if n != 4 {
		t.Errorf("state dim = %d, want 4 (plant + order3 filter)", n)
	}
}

func TestDiscretizeWithOpts_FractionalOutputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 1 {
		t.Errorf("OutputDelay = %v, want integer remainder [1]", disc.OutputDelay)
	}
	n, _, _ := disc.Dims()
	if n != 4 {
		t.Errorf("state dim = %d, want 4 (plant + order3 filter)", n)
	}
}

func TestDiscretizeWithOpts_MIMO_FractionalInputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.35, 0.2}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 2 || disc.InputDelay[0] != 1 || disc.InputDelay[1] != 2 {
		t.Errorf("InputDelay = %v, want integer remainders [1 2]", disc.InputDelay)
	}
	_, m, p := disc.Dims()
	if m != 2 || p != 2 {
		t.Errorf("dims m=%d p=%d, want 2,2", m, p)
	}
}

func TestDiscretizeWithOpts_MixedIntegerFractionalInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.3, 0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	_, m, _ := disc.Dims()
	if m != 2 {
		t.Errorf("m = %d, want 2", m)
	}
}

func TestDiscretizeWithOpts_DefaultMethod(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	disc, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
}

func TestDiscretizeWithOpts_AlreadyDiscrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{})
	if err == nil {
		t.Fatal("expected error for already discrete")
	}
}

func TestDiscretizeWithOpts_FreqResponseWithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	tfDisc, _ := disc.TransferFunction(nil)

	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfDisc.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of reasonable range", w, mag)
		}
	}
}

func TestDiscretizeWithOpts_ZeroDelay_NoThiranNeeded(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no Thiran for zero delay)", n)
	}
}

func TestDiscretize_DoesNotMutateReceiver(t *testing.T) {
	orig := makeTestSystem()
	before := orig.Copy()

	_, err := orig.C2D(0.01, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}

	assertMatClose(t, "A", orig.A, before.A, 0)
	assertMatClose(t, "B", orig.B, before.B, 0)
	assertMatClose(t, "C", orig.C, before.C, 0)
	assertMatClose(t, "D", orig.D, before.D, 0)
}

func TestDiscretizeWithOpts_SISO_FractionalIODelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.35})

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil after Thiran decomposition")
	}

	tfDisc, _ := disc.TransferFunction(nil)
	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfDisc.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of reasonable range", w, mag)
		}
	}
}

func TestDiscretizeWithOpts_MIMO_DecomposableIODelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0.5}),
		mat.NewDense(2, 1, []float64{1, 0.3}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.Delay = mat.NewDense(2, 2, []float64{0.3, 0.5, 0.3, 0.5})

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil for column-uniform decomposable IODelay")
	}
}

func TestDiscretizeWithOpts_SISO_IntegerIODelay_NoThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.3})

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, ThiranOrder: 0})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay == nil {
		t.Fatal("expected discrete Delay to be set")
	}
	if got := disc.Delay.At(0, 0); got != 3 {
		t.Errorf("Delay = %v, want 3", got)
	}
}

func TestDiscretizeWithOpts_MixedIODelay_InputDelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.25})
	sys.InputDelay = []float64{0.15}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil after decomposition+merge")
	}

	tfDisc, _ := disc.TransferFunction(nil)
	z := cmplx.Exp(complex(0, 0.5*dt))
	h := tfDisc.TF.Eval(z)[0][0]
	mag := cmplx.Abs(h)
	if mag > 2.0 || mag < 0 {
		t.Errorf("w=0.5: |H(z)| = %v out of reasonable range", mag)
	}

	_ = math.Abs(0) // keep math import
}

func TestUndiscretizeInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.InputDelay = []float64{3}

	ct, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.InputDelay) != 1 || math.Abs(ct.InputDelay[0]-0.3) > 1e-12 {
		t.Errorf("InputDelay = %v, want [0.3]", ct.InputDelay)
	}
}

func TestUndiscretizeOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.OutputDelay = []float64{5}

	ct, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.OutputDelay) != 1 || math.Abs(ct.OutputDelay[0]-0.5) > 1e-12 {
		t.Errorf("OutputDelay = %v, want [0.5]", ct.OutputDelay)
	}
}

func TestUndiscretizeInternalDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.LFT = &LFTDelay{
		Tau: []float64{5},
		B2:  mat.NewDense(1, 1, []float64{0.2}),
		C2:  mat.NewDense(1, 1, []float64{0.3}),
		D12: mat.NewDense(1, 1, []float64{0.4}),
		D21: mat.NewDense(1, 1, []float64{0.5}),
		D22: mat.NewDense(1, 1, []float64{0}),
	}

	ct, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.LFT.Tau) != 1 || math.Abs(ct.LFT.Tau[0]-0.5) > 1e-12 {
		t.Errorf("InternalDelay = %v, want [0.5]", ct.LFT.Tau)
	}
	for _, w := range []float64{.2, 1, 5} {
		point := complex(0, w)
		z := (20 + point) / (20 - point)
		want, err := conversionAugmentedRational(sys).EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		assertPrewarpResponse(t, conversionAugmentedRational(ct), point, want)
	}
}

func TestUndiscretizeAllDelays(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{2})
	sys.InputDelay = []float64{3}
	sys.OutputDelay = []float64{4}

	ct, err := sys.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(ct.Delay.At(0, 0)-0.2) > 1e-12 {
		t.Errorf("Delay = %v, want 0.2", ct.Delay.At(0, 0))
	}
	if math.Abs(ct.InputDelay[0]-0.3) > 1e-12 {
		t.Errorf("InputDelay = %v, want [0.3]", ct.InputDelay)
	}
	if math.Abs(ct.OutputDelay[0]-0.4) > 1e-12 {
		t.Errorf("OutputDelay = %v, want [0.4]", ct.OutputDelay)
	}
}

func TestRoundtripDelays(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if disc.InputDelay[0] != 3 {
		t.Fatalf("discrete InputDelay = %v, want 3", disc.InputDelay[0])
	}
	if disc.OutputDelay[0] != 5 {
		t.Fatalf("discrete OutputDelay = %v, want 5", disc.OutputDelay[0])
	}

	ct, err := disc.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(ct.InputDelay[0]-0.3) > 1e-10 {
		t.Errorf("roundtrip InputDelay = %v, want 0.3", ct.InputDelay[0])
	}
	if math.Abs(ct.OutputDelay[0]-0.5) > 1e-10 {
		t.Errorf("roundtrip OutputDelay = %v, want 0.5", ct.OutputDelay[0])
	}
}

func TestDiscretizeZOHWithInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeZOHWithOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 5 {
		t.Errorf("OutputDelay = %v, want [5]", disc.OutputDelay)
	}
}

func TestDiscretizeTustinWithInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeTustinFractionalDelayRounds(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.37}

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if disc.InputDelay[0] != 4 {
		t.Fatalf("rounded InputDelay=%v, want [4]", disc.InputDelay)
	}
}

func TestDiscretizeZOHWithInternalDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.LFT = &LFTDelay{
		Tau: []float64{0.2},
		B2:  mat.NewDense(2, 1, []float64{0.5, 0.3}),
		C2:  mat.NewDense(1, 2, []float64{0.1, 0.2}),
		D12: mat.NewDense(1, 1, []float64{0.4}),
		D21: mat.NewDense(1, 1, []float64{0.6}),
		D22: mat.NewDense(1, 1, []float64{0}),
	}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.Dt != dt {
		t.Fatalf("Dt = %v, want %v", disc.Dt, dt)
	}

	if len(disc.LFT.Tau) != 1 || disc.LFT.Tau[0] != 2 {
		t.Errorf("InternalDelay = %v, want [2]", disc.LFT.Tau)
	}
	expectedB2 := mat.NewDense(2, 1, []float64{0.0512036577844453, 0.021304040984362})
	assertMatClose(t, "B2", disc.LFT.B2, expectedB2, 1e-12)
	assertMatClose(t, "C2", disc.LFT.C2, sys.LFT.C2, 1e-15)
	assertMatClose(t, "D12", disc.LFT.D12, sys.LFT.D12, 1e-15)
	assertMatClose(t, "D21", disc.LFT.D21, sys.LFT.D21, 1e-15)
	assertMatClose(t, "D22", disc.LFT.D22, sys.LFT.D22, 1e-15)

	refSys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	ref, _ := refSys.C2D(dt, C2DOptions{})
	assertMatClose(t, "A", disc.A, ref.A, 1e-12)
	assertMatClose(t, "B", disc.B, ref.B, 1e-12)
	assertMatClose(t, "C", disc.C, ref.C, 1e-12)
	assertMatClose(t, "D", disc.D, ref.D, 1e-12)
}

func TestDiscretizeZOHInternalDelayD22UpperTriangular(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.LFT = &LFTDelay{
		Tau: []float64{0.2, 0.3},
		B2:  mat.NewDense(1, 2, []float64{0.5, 0.3}),
		C2:  mat.NewDense(2, 1, []float64{0.1, 0.2}),
		D12: mat.NewDense(1, 2, []float64{0.4, 0.1}),
		D21: mat.NewDense(2, 1, []float64{0.6, 0.7}),
		D22: mat.NewDense(2, 2, []float64{0, 0.5, 0, 0}),
	}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.LFT.Tau) != 2 {
		t.Fatalf("InternalDelay len = %d, want 2", len(disc.LFT.Tau))
	}
	if disc.LFT.Tau[0] != 2 || disc.LFT.Tau[1] != 3 {
		t.Errorf("InternalDelay = %v, want [2 3]", disc.LFT.Tau)
	}
}

func TestDiscretizeZOHInternalDelayD22General(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.LFT = &LFTDelay{
		Tau: []float64{0.2, 0.3},
		B2:  mat.NewDense(1, 2, []float64{0.5, 0.3}),
		C2:  mat.NewDense(2, 1, []float64{0.1, 0.2}),
		D12: mat.NewDense(1, 2, []float64{0.4, 0.1}),
		D21: mat.NewDense(2, 1, []float64{0.6, 0.7}),
		D22: mat.NewDense(2, 2, []float64{0.1, 0.5, 0.3, 0}),
	}

	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "positive-delay D22", disc.LFT.D22, sys.LFT.D22, 1e-12)
}

func TestDiscretizeZOHInternalDelayFreqResp(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.5}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.1
	disc, err := lft.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}

	if disc.LFT == nil || len(disc.LFT.Tau) == 0 {
		t.Fatal("expected InternalDelay on discretized LFT system")
	}

	if _, err := disc.TransferFunction(nil); !errors.Is(err, ErrDelayNotRepresentable) {
		t.Fatalf("TransferFunction of internal-delay model err = %v, want ErrDelayNotRepresentable", err)
	}
	for _, w := range []float64{0.1, 0.5, 1.0, 2.0} {
		z := cmplx.Exp(complex(0, w*dt))
		H, err := disc.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		h := H[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of range", w, mag)
		}
	}
}

func TestC2DDelayModelingInternal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}

	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no state augmentation)", n)
	}
	if len(disc.LFT.Tau) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1", len(disc.LFT.Tau))
	}
	if math.Abs(disc.LFT.Tau[0]-1) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 1", disc.LFT.Tau[0])
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestC2DDelayModelingInternalFreqResp(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}
	dt := 0.1

	discInternal, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
	if err != nil {
		t.Fatal(err)
	}
	discState, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))

		hInt, err := discInternal.EvalFr(z)
		if err != nil {
			t.Fatalf("w=%v: EvalFr internal error: %v", w, err)
		}
		hState, err := discState.EvalFr(z)
		if err != nil {
			t.Fatalf("w=%v: EvalFr state error: %v", w, err)
		}

		magInt := cmplx.Abs(hInt[0][0])
		magState := cmplx.Abs(hState[0][0])
		if math.Abs(magInt-magState)/magState > 0.15 {
			t.Errorf("w=%v: mag mismatch internal=%v state=%v", w, magInt, magState)
		}
	}
}

func TestC2DDelayModelingDefault(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
	if disc.LFT != nil {
		t.Errorf("InternalDelay should be nil for default (state) mode, got %v", disc.LFT.Tau)
	}
}

func TestC2DDelayModelingIntegerDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
	if disc.LFT != nil {
		t.Errorf("InternalDelay should be nil for integer delay, got %v", disc.LFT.Tau)
	}
}

func TestC2DDelayModelingInternalOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no state augmentation)", n)
	}
	if len(disc.LFT.Tau) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1", len(disc.LFT.Tau))
	}
	if math.Abs(disc.LFT.Tau[0]-1) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 1", disc.LFT.Tau[0])
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 3 {
		t.Errorf("OutputDelay = %v, want [3]", disc.OutputDelay)
	}
}

func TestC2DDelayModelingInternalMIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.35, 0.2}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingInternal})
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := disc.Dims()
	if n != 2 {
		t.Errorf("state dim = %d, want 2", n)
	}
	if m != 2 || p != 2 {
		t.Errorf("dims m=%d p=%d, want 2,2", m, p)
	}
	if len(disc.LFT.Tau) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1 (only ch0 has fractional)", len(disc.LFT.Tau))
	}
	if math.Abs(disc.LFT.Tau[0]-1) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 1", disc.LFT.Tau[0])
	}
	if disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay[0] = %v, want 3", disc.InputDelay[0])
	}
	if disc.InputDelay[1] != 2 {
		t.Errorf("InputDelay[1] = %v, want 2", disc.InputDelay[1])
	}
}

// --- DiscretizeImpulse tests ---

func TestImpulse_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestImpulse_InvalidDt(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.C2D(0, C2DOptions{Method: C2DMethodImpulse})
	if err == nil {
		t.Fatal("expected error for dt=0")
	}
	_, err = sys.C2D(-1, C2DOptions{Method: C2DMethodImpulse})
	if err == nil {
		t.Fatal("expected error for dt<0")
	}
}

func TestImpulse_PureGain(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	sys, _ := NewGain(D, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := disc.Dims()
	if n != 0 || m != 3 || p != 2 {
		t.Fatalf("dims = %d,%d,%d, want 0,3,2", n, m, p)
	}
	assertMatClose(t, "D", disc.D, newDense(2, 3), 1e-14)
}

func TestImpulse_Scalar(t *testing.T) {
	a, b, c := -2.0, 3.0, 1.0
	dt := 0.1
	sys, _ := New(
		mat.NewDense(1, 1, []float64{a}),
		mat.NewDense(1, 1, []float64{b}),
		mat.NewDense(1, 1, []float64{c}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}

	adExpected := math.Exp(a * dt)
	bdExpected := dt * adExpected * b
	tol := 1e-12
	if diff := math.Abs(disc.A.At(0, 0) - adExpected); diff > tol {
		t.Errorf("Ad = %v, want %v", disc.A.At(0, 0), adExpected)
	}
	if diff := math.Abs(disc.B.At(0, 0) - bdExpected); diff > tol {
		t.Errorf("Bd = %v, want %v", disc.B.At(0, 0), bdExpected)
	}
	if disc.C.At(0, 0) != c {
		t.Errorf("Cd = %v, want %v", disc.C.At(0, 0), c)
	}
	if math.Abs(disc.D.At(0, 0)-dt*c*b) > 1e-12 {
		t.Errorf("Dd = %v, want %v", disc.D.At(0, 0), dt*c*b)
	}
}

func TestImpulse_ImpulseResponseMatch(t *testing.T) {
	sys := makeTestSystem()
	dt := 0.05
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	Adt := mat.NewDense(n, n, nil)
	Adt.Scale(dt, sys.A)
	var eA mat.Dense
	eA.Exp(Adt)

	AkT := mat.NewDense(n, n, nil)
	for i := range n {
		AkT.Set(i, i, 1)
	}

	AdPow := mat.NewDense(n, n, nil)
	for i := range n {
		AdPow.Set(i, i, 1)
	}

	for k := 1; k <= 20; k++ {
		AkT.Mul(AkT, &eA)

		var contH mat.Dense
		var tmp mat.Dense
		tmp.Mul(sys.C, AkT)
		contH.Mul(&tmp, sys.B)
		expected := dt * contH.At(0, 0)

		var discH mat.Dense
		var tmp2 mat.Dense
		tmp2.Mul(disc.C, AdPow)
		discH.Mul(&tmp2, disc.B)
		got := discH.At(0, 0)

		AdPow.Mul(AdPow, disc.A)

		if diff := math.Abs(got - expected); diff > 1e-10 {
			t.Errorf("k=%d: h_d=%v, want %v (diff %v)", k, got, expected, diff)
		}
	}
}

func TestImpulse_Stability(t *testing.T) {
	sys := makeTestSystem()
	disc, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range poles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("pole[%d] = %v, magnitude %v >= 1", i, p, cmplx.Abs(p))
		}
	}
}

func TestImpulse_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	sys, _ := New(A, B, C, D, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := disc.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Fatalf("dims = %d,%d,%d, want 2,2,2", n, m, p)
	}
}

func TestImpulse_ViaOpts(t *testing.T) {
	sys := makeTestSystem()
	dt := 0.1
	direct, err := sys.C2D(dt, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	viaOpts, err := sys.C2D(dt, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-12
	assertMatClose(t, "A", viaOpts.A, direct.A, tol)
	assertMatClose(t, "B", viaOpts.B, direct.B, tol)
	assertMatClose(t, "C", viaOpts.C, direct.C, tol)
	assertMatClose(t, "D", viaOpts.D, direct.D, tol)
}

// --- DiscretizeFOH tests ---

func TestFOH_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestFOH_InvalidDt(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.C2D(0, C2DOptions{Method: C2DMethodFOH})
	if err == nil {
		t.Fatal("expected error for dt=0")
	}
}

func TestFOH_PureGain(t *testing.T) {
	d := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	sys, _ := NewGain(d, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := disc.Dims()
	if n != 0 || m != 3 || p != 2 {
		t.Fatalf("dims = %d,%d,%d, want 0,3,2", n, m, p)
	}
	assertMatClose(t, "D", disc.D, d, 1e-14)
}

func TestFOH_Scalar(t *testing.T) {
	a, b, c, dt := -2.0, 3.0, 1.0, 0.1
	sys, _ := New(mat.NewDense(1, 1, []float64{a}), mat.NewDense(1, 1, []float64{b}), mat.NewDense(1, 1, []float64{c}), mat.NewDense(1, 1, []float64{0}), 0)
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	ad := math.Exp(a * dt)
	gamma0 := math.Expm1(a*dt) / a * b
	gamma1 := (math.Expm1(a*dt) - a*dt) / (a * a * dt) * b
	assertMatClose(t, "A", disc.A, mat.NewDense(1, 1, []float64{ad}), 1e-12)
	assertMatClose(t, "B", disc.B, mat.NewDense(1, 1, []float64{gamma0 + (ad-1)*gamma1}), 1e-12)
	assertMatClose(t, "C", disc.C, sys.C, 1e-12)
	assertMatClose(t, "D", disc.D, mat.NewDense(1, 1, []float64{c * gamma1}), 1e-12)
}

func TestFOH_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := disc.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = %d,%d,%d, want 2,1,1", n, m, p)
	}

	tol := 1e-10
	if diff := math.Abs(disc.A.At(0, 0) - 1); diff > tol {
		t.Errorf("A[0,0] = %v, want 1", disc.A.At(0, 0))
	}
	if diff := math.Abs(disc.A.At(0, 1) - dt); diff > tol {
		t.Errorf("A[0,1] = %v, want %v", disc.A.At(0, 1), dt)
	}
	if diff := math.Abs(disc.A.At(1, 1) - 1); diff > tol {
		t.Errorf("A[1,1] = %v, want 1", disc.A.At(1, 1))
	}
}

func TestFOH_SingularA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, -1})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := disc.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = %d,%d,%d, want 2,1,1", n, m, p)
	}
}

func TestFOH_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	sys, _ := New(A, B, C, D, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := disc.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Fatalf("dims = %d,%d,%d, want 2,2,2", n, m, p)
	}
}

func TestFOH_Stability(t *testing.T) {
	sys := makeTestSystem()
	disc, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range poles {
		if cmplx.Abs(p) >= 1+1e-10 {
			t.Errorf("pole[%d] = %v, magnitude %v >= 1", i, p, cmplx.Abs(p))
		}
	}
}

func TestFOH_ViaOpts(t *testing.T) {
	sys := makeTestSystem()
	dt := 0.1
	direct, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	viaOpts, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-12
	assertMatClose(t, "A", viaOpts.A, direct.A, tol)
	assertMatClose(t, "B", viaOpts.B, direct.B, tol)
	assertMatClose(t, "C", viaOpts.C, direct.C, tol)
	assertMatClose(t, "D", viaOpts.D, direct.D, tol)
}

// --- DiscretizeMatched tests ---

func TestMatched_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestMatched_InvalidDt(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.C2D(0, C2DOptions{Method: C2DMethodMatched})
	if err == nil {
		t.Fatal("expected error for dt=0")
	}
}

func TestMatched_NotSISO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	sys, _ := New(A, B, C, D, 0)
	_, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
	if err == nil {
		t.Fatal("expected ErrNotSISO")
	}
}

func TestMatched_PureGain(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{5})
	sys, _ := NewGain(D, 0)
	disc, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := disc.Dims()
	if n != 0 {
		t.Fatalf("n = %d, want 0", n)
	}
	if diff := math.Abs(disc.D.At(0, 0) - 5); diff > 1e-14 {
		t.Errorf("D = %v, want 5", disc.D.At(0, 0))
	}
}

func TestMatched_FirstOrder(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}

	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	expectedPole := math.Exp(-dt)
	if len(poles) != 1 {
		t.Fatalf("got %d poles, want 1", len(poles))
	}
	if diff := cmplx.Abs(poles[0] - complex(expectedPole, 0)); diff > 1e-10 {
		t.Errorf("pole = %v, want %v", poles[0], expectedPole)
	}

	tfr, err := disc.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	dcGain := Poly(tfr.TF.Num[0][0]).Eval(complex(1, 0)) / Poly(tfr.TF.Den[0]).Eval(complex(1, 0))
	if diff := math.Abs(real(dcGain) - 1.0); diff > 1e-6 {
		t.Errorf("DC gain = %v, want 1.0", dcGain)
	}
}

func TestMatched_SecondOrderComplex(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -5, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	discPoles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	contPoles, _ := sys.Poles()
	for _, cp := range contPoles {
		expectedZ := cmplx.Exp(cp * complex(dt, 0))
		found := false
		for _, dp := range discPoles {
			if cmplx.Abs(dp-expectedZ) < 1e-6 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected discrete pole near %v (from cont pole %v), not found", expectedZ, cp)
		}
	}
}

func TestMatched_WithZeros(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}

	contZeros, _ := sys.Zeros()
	discZeros, err := disc.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	for _, cz := range contZeros {
		expectedZ := cmplx.Exp(cz * complex(dt, 0))
		found := false
		for _, dz := range discZeros {
			if cmplx.Abs(dz-expectedZ) < 1e-6 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected discrete zero near %v, not found in %v", expectedZ, discZeros)
		}
	}
}

func TestMatched_Integrator(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}

	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	foundUnit := false
	for _, p := range poles {
		if cmplx.Abs(p-1) < 1e-10 {
			foundUnit = true
		}
	}
	if !foundUnit {
		t.Errorf("expected pole at z=1 for integrator, got %v", poles)
	}

	w := 0.5
	s := complex(0, w)
	z := cmplx.Exp(complex(0, w*dt))
	contH := complex(1, 0) / s
	tfr, err := disc.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	discH := Poly(tfr.TF.Num[0][0]).Eval(z) / Poly(tfr.TF.Den[0]).Eval(z)
	ratio := cmplx.Abs(discH) / cmplx.Abs(contH)
	if ratio < 0.5 || ratio > 2.0 {
		t.Errorf("frequency response magnitude ratio at w=%v: %v (too far from 1)", w, ratio)
	}
}

func TestMatched_DCGain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dt := 0.05
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}

	contTF, _ := sys.TransferFunction(nil)
	contDC := real(Poly(contTF.TF.Num[0][0]).Eval(0) / Poly(contTF.TF.Den[0]).Eval(0))

	discTF, _ := disc.TransferFunction(nil)
	discDC := real(Poly(discTF.TF.Num[0][0]).Eval(complex(1, 0)) / Poly(discTF.TF.Den[0]).Eval(complex(1, 0)))

	if diff := math.Abs(discDC - contDC); diff > 1e-6 {
		t.Errorf("DC gain: disc=%v, cont=%v, diff=%v", discDC, contDC, diff)
	}
}

func TestMatched_Stability(t *testing.T) {
	sys := makeTestSystem()
	disc, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range poles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("pole[%d] = %v, magnitude %v >= 1", i, p, cmplx.Abs(p))
		}
	}
}

func TestMatched_ViaOpts(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dt := 0.1
	direct, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	viaOpts, err := sys.C2D(dt, C2DOptions{Method: C2DMethodMatched})
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	assertMatClose(t, "A", viaOpts.A, direct.A, tol)
	assertMatClose(t, "B", viaOpts.B, direct.B, tol)
	assertMatClose(t, "C", viaOpts.C, direct.C, tol)
	assertMatClose(t, "D", viaOpts.D, direct.D, tol)
}

// --- D2D tests ---

func TestD2D_Continuous(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.D2D(0.1, D2DOptions{})
	if err == nil {
		t.Fatal("expected error for continuous system")
	}
}

func TestD2D_InvalidDt(t *testing.T) {
	sys := makeTestSystem()
	disc, _ := sys.C2D(0.1, C2DOptions{})
	_, err := disc.D2D(0, D2DOptions{})
	if err == nil {
		t.Fatal("expected error for newDt=0")
	}
	_, err = disc.D2D(-1, D2DOptions{})
	if err == nil {
		t.Fatal("expected error for newDt<0")
	}
}

func TestD2D_SameDt(t *testing.T) {
	sys := makeTestSystem()
	disc, _ := sys.C2D(0.1, C2DOptions{})
	result, err := disc.D2D(0.1, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-12
	assertMatClose(t, "A", result.A, disc.A, tol)
	assertMatClose(t, "B", result.B, disc.B, tol)
	assertMatClose(t, "C", result.C, disc.C, tol)
	assertMatClose(t, "D", result.D, disc.D, tol)
}

func TestD2D_Downsample(t *testing.T) {
	sys := makeTestSystem()
	disc1, _ := sys.C2D(0.05, C2DOptions{})
	disc2, err := disc1.D2D(0.1, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if disc2.Dt != 0.1 {
		t.Errorf("Dt = %v, want 0.1", disc2.Dt)
	}

	w := 0.3
	z1 := cmplx.Exp(complex(0, w*disc1.Dt))
	z2 := cmplx.Exp(complex(0, w*disc2.Dt))
	tf1, _ := disc1.TransferFunction(nil)
	tf2, _ := disc2.TransferFunction(nil)
	h1 := Poly(tf1.TF.Num[0][0]).Eval(z1) / Poly(tf1.TF.Den[0]).Eval(z1)
	h2 := Poly(tf2.TF.Num[0][0]).Eval(z2) / Poly(tf2.TF.Den[0]).Eval(z2)
	if diff := cmplx.Abs(h1 - h2); diff > 0.05 {
		t.Errorf("freq response mismatch at w=%v: h1=%v, h2=%v, diff=%v", w, h1, h2, diff)
	}
}

func TestD2D_Upsample(t *testing.T) {
	sys := makeTestSystem()
	disc1, _ := sys.C2D(0.2, C2DOptions{})
	disc2, err := disc1.D2D(0.1, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if disc2.Dt != 0.1 {
		t.Errorf("Dt = %v, want 0.1", disc2.Dt)
	}
	n, m, p := disc2.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = %d,%d,%d, want 2,1,1", n, m, p)
	}
}

func TestD2D_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	sys, _ := New(A, B, C, D, 0)
	disc, _ := sys.C2D(0.1, C2DOptions{})
	result, err := disc.D2D(0.2, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Fatalf("dims = %d,%d,%d, want 2,2,2", n, m, p)
	}
}

func TestD2D_DefaultMethod(t *testing.T) {
	sys := makeTestSystem()
	disc, _ := sys.C2D(0.1, C2DOptions{})
	result, err := disc.D2D(0.2, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Dt != 0.2 {
		t.Errorf("Dt = %v, want 0.2", result.Dt)
	}
}

func TestD2D_Names(t *testing.T) {
	sys := makeTestSystem()
	sys.InputName = []string{"force"}
	sys.OutputName = []string{"position"}
	disc, _ := sys.C2D(0.1, C2DOptions{})
	result, err := disc.D2D(0.2, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputName) == 0 || result.InputName[0] != "force" {
		t.Errorf("InputName = %v, want [force]", result.InputName)
	}
	if len(result.OutputName) == 0 || result.OutputName[0] != "position" {
		t.Errorf("OutputName = %v, want [position]", result.OutputName)
	}
}

func makeLFTSystem(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{0.5}
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	return lft
}

func TestDiscretizeTustinWithInternalDelay(t *testing.T) {
	lft := makeLFTSystem(t)

	dt := 0.1
	disc, err := lft.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}

	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.LFT == nil || len(disc.LFT.Tau) == 0 {
		t.Fatal("expected InternalDelay")
	}

	n, m, p := disc.Dims()
	if n < 2 || m < 1 || p < 1 {
		t.Errorf("dims = (%d,%d,%d), want ≥ (2,1,1)", n, m, p)
	}

	stable, err := disc.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("discretized system unstable")
	}
}

func TestDiscretizeFOHWithInternalDelay(t *testing.T) {
	lft := makeLFTSystem(t)

	dt := 0.1
	disc, err := lft.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}

	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.LFT == nil || len(disc.LFT.Tau) == 0 {
		t.Fatal("expected InternalDelay")
	}

	stable, err := disc.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("discretized system unstable")
	}
}

func TestDiscretizeImpulseWithInternalDelay(t *testing.T) {
	lft := makeLFTSystem(t)

	_, err := lft.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
	if !errors.Is(err, ErrFeedbackDelay) {
		t.Fatalf("internal impulse: %v", err)
	}
}

func TestDiscretizeTustinAugmented_FreqResp(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.InputDelay = []float64{0.5}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.1
	discZOH, err := lft.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	discTustin, err := lft.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		hZOH, _ := discZOH.EvalFr(z)
		hTustin, _ := discTustin.EvalFr(z)
		magZOH := cmplx.Abs(hZOH[0][0])
		magTustin := cmplx.Abs(hTustin[0][0])
		if magZOH == 0 || magTustin == 0 {
			continue
		}
		ratio := magTustin / magZOH
		if ratio < 0.5 || ratio > 2.0 {
			t.Errorf("w=%v: Tustin/ZOH mag ratio = %v, expected ~1", w, ratio)
		}
	}
}

func TestDiscretizeFOHAugmented_Stability(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.InputDelay = []float64{0.3}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	disc, err := lft.C2D(0.1, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}

	stable, err := disc.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("FOH-discretized system unstable")
	}
}

func TestDiscretizeImpulseAugmented_SISO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.InputDelay = []float64{0.3}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	_, err = lft.C2D(0.1, C2DOptions{Method: C2DMethodImpulse})
	if !errors.Is(err, ErrFeedbackDelay) {
		t.Fatalf("internal impulse: %v", err)
	}
}

func TestD2C_WrongDomain(t *testing.T) {
	sys := makeTestSystem()
	if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err == nil {
		t.Fatal("expected ErrWrongDomain for continuous input")
	}
	if _, err := sys.D2C(D2COptions{Method: C2DMethodTustin}); err == nil {
		t.Fatal("expected ErrWrongDomain for continuous input")
	}
}

func TestD2C_UnknownMethod(t *testing.T) {
	disc, err := makeTestSystem().C2D(0.05, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disc.D2C(D2COptions{Method: C2DMethod("bogus")}); err == nil {
		t.Fatal("expected error for unknown method")
	}
}

func TestD2C_Tustin_Roundtrip(t *testing.T) {
	orig := makeTestSystem()
	dt := 0.02

	disc, err := orig.C2D(dt, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !rec.IsContinuous() {
		t.Fatal("expected continuous")
	}
	tol := 1e-10
	assertMatClose(t, "A", rec.A, orig.A, tol)
	assertMatClose(t, "B", rec.B, orig.B, tol)
	assertMatClose(t, "C", rec.C, orig.C, tol)
	assertMatClose(t, "D", rec.D, orig.D, tol)
}

func TestD2C_ZOH_Roundtrip(t *testing.T) {
	orig := makeTestSystem()
	dt := 0.05

	disc, err := orig.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if !rec.IsContinuous() {
		t.Fatal("expected continuous")
	}
	tol := 1e-9
	assertMatClose(t, "A", rec.A, orig.A, tol)
	assertMatClose(t, "B", rec.B, orig.B, tol)
	assertMatClose(t, "C", rec.C, orig.C, tol)
	assertMatClose(t, "D", rec.D, orig.D, tol)
}

func TestD2C_ZOH_DefaultMethod(t *testing.T) {
	orig := makeTestSystem()
	dt := 0.05
	disc, err := orig.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	recDefault, err := disc.D2C(D2COptions{Method: C2DMethod("")})
	if err != nil {
		t.Fatal(err)
	}
	recZOH, err := disc.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "A default==zoh", recDefault.A, recZOH.A, 1e-14)
	assertMatClose(t, "B default==zoh", recDefault.B, recZOH.B, 1e-14)
}

func TestD2C_ZOH_NegativeEigenvalue(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-0.5}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	out, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	alpha, omega := math.Log(0.5)/0.1, math.Pi/0.1
	assertMatClose(t, "A", out.A, mat.NewDense(2, 2, []float64{alpha, -omega, omega, alpha}), 1e-10)
	assertMatClose(t, "B", out.B, mat.NewDense(2, 1, []float64{alpha / (-1.5), omega / (-1.5)}), 1e-10)
}

func TestD2C_ZOH_EigenvalueAtOne(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0.1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	out, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "A", out.A, mat.NewDense(1, 1, []float64{0}), 1e-14)
	assertMatClose(t, "B", out.B, mat.NewDense(1, 1, []float64{1}), 1e-14)
}

func TestD2C_ZOH_MIMO_Roundtrip(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0.5, 0,
		0, -2, 1,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 0,
	})
	D := mat.NewDense(2, 2, []float64{
		0, 0,
		0, 0,
	})
	orig, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	dt := 0.1
	disc, err := orig.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-8
	assertMatClose(t, "A", rec.A, orig.A, tol)
	assertMatClose(t, "B", rec.B, orig.B, tol)
}

func TestD2C_ZOH_N0_GainOnly(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	disc := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  D,
		Dt: 0.1,
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if !rec.IsContinuous() {
		t.Fatal("expected continuous")
	}
	assertMatClose(t, "D", rec.D, D, 0)
}

func TestD2C_InputDelayRoundtrip(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	orig, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	orig.InputDelay = []float64{0.2}

	dt := 0.1
	disc, err := orig.C2D(dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 2 {
		t.Fatalf("discrete input delay = %v, want [2]", disc.InputDelay)
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.InputDelay) != 1 || math.Abs(rec.InputDelay[0]-0.2) > 1e-12 {
		t.Fatalf("continuous input delay = %v, want [0.2]", rec.InputDelay)
	}
}

func TestConversionDefaultsAreZOH(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	disc, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ad := math.Exp(-0.2)
	if math.Abs(disc.A.At(0, 0)-ad) > 1e-14 || math.Abs(disc.B.At(0, 0)-3*(1-ad)/2) > 1e-14 {
		t.Fatalf("C2D default: A=%g B=%g, want ZOH %g %g", disc.A.At(0, 0), disc.B.At(0, 0), ad, 3*(1-ad)/2)
	}
	cont, err := disc.D2C(D2COptions{})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cont.A.At(0, 0)+2) > 1e-12 || math.Abs(cont.B.At(0, 0)-3) > 1e-12 {
		t.Fatalf("D2C default: A=%g B=%g, want ZOH inverse -2 3", cont.A.At(0, 0), cont.B.At(0, 0))
	}
	re, err := disc.D2D(0.25, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if want := math.Exp(-0.5); math.Abs(re.A.At(0, 0)-want) > 1e-12 {
		t.Fatalf("D2D default: A=%g, want ZOH %g", re.A.At(0, 0), want)
	}
}

func TestConversionNilAndFitMethod(t *testing.T) {
	var nilSys *System
	if _, err := nilSys.C2D(0.1, C2DOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("C2D: %v", err)
	}
	if _, err := nilSys.D2C(D2COptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("D2C: %v", err)
	}
	if _, err := nilSys.D2D(0.1, D2DOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("D2D: %v", err)
	}
	if _, _, err := nilSys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("C2DFit: %v", err)
	}
	sys := makeTestSystem()
	if _, _, err := sys.C2DFit(0.1, C2DOptions{}); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("C2DFit zoh: %v, want ErrOptionUnsupported", err)
	}
}
