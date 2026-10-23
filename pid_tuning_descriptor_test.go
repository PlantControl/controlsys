package controlsys

import (
	"context"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidDescriptorOf returns dss(E·A, E·B, C, D, E), whose transfer function is
// sys's.
func pidDescriptorOf(t *testing.T, sys *System, E *mat.Dense) *System {
	t.Helper()
	var ea, eb mat.Dense
	ea.Mul(E, sys.A)
	eb.Mul(E, sys.B)
	d, err := NewDescriptor(&ea, &eb, sys.C, sys.D, E, sys.Dt)
	if err != nil {
		t.Fatal(err)
	}
	d.InputDelay = sys.InputDelay
	return d
}

func pidGainsClose(a, b *PID2) bool {
	for _, p := range [][2]float64{{a.Kp, b.Kp}, {a.Ki, b.Ki}, {a.Kd, b.Kd}, {a.Tf, b.Tf}} {
		if math.Abs(p[0]-p[1]) > 1e-8*(1+math.Abs(p[1])) {
			return false
		}
	}
	return true
}

func pidClosedLoopStable(t *testing.T, plant *System, c *PID2) bool {
	t.Helper()
	cs, err := (&PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt}).System()
	if err != nil {
		return false
	}
	loop, err := Feedback(plant, cs, -1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := loop.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	return stable
}

// MATLAB pidtune accepts dss models: a descriptor plant with nonsingular E
// tunes as its explicit model E⁻¹A, E⁻¹B.
func TestTunePIDDescriptorPlant(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -6, -11, -6})
	B := mat.NewDense(3, 1, []float64{0, 0.5, 1})
	C := mat.NewDense(1, 3, []float64{2, 1, 0})
	D := mat.NewDense(1, 1, nil)
	cont, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	delayed := cont.Copy()
	delayed.InputDelay = []float64{0.2}
	disc, err := cont.discretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	E := mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, -1, 0, 0.25, 1.5})
	ctx := context.Background()
	for _, tc := range []struct {
		name string
		sys  *System
	}{{"continuous", cont}, {"delayed", delayed}, {"discrete", disc}} {
		for _, e := range []*mat.Dense{E, eyeDense(3)} {
			desc := pidDescriptorOf(t, tc.sys, e)
			for _, fam := range []PidtuneType{PidtunePI, PidtunePD, PidtunePID, PidtunePIDF} {
				want, err := TunePID(ctx, tc.sys, fam, PIDTuningOptions{})
				if err != nil {
					t.Fatalf("%s %s explicit: %v", tc.name, fam, err)
				}
				got, err := TunePID(ctx, desc, fam, PIDTuningOptions{})
				if err != nil {
					t.Errorf("%s %s E=%v: %v", tc.name, fam, e.RawMatrix().Data, err)
					continue
				}
				if !pidGainsClose(got.Controller, want.Controller) {
					t.Errorf("%s %s: descriptor %+v, explicit %+v", tc.name, fam, *got.Controller, *want.Controller)
				}
				if got.Evidence.Stability != want.Evidence.Stability {
					t.Errorf("%s %s: stability evidence %v, explicit %v", tc.name, fam, got.Evidence.Stability, want.Evidence.Stability)
				}
				if tc.sys.HasDelay() || got.Controller.Tf == 0 && got.Controller.Kd != 0 {
					continue
				}
				if !pidClosedLoopStable(t, tc.sys, got.Controller) {
					t.Errorf("%s %s: %+v destabilizes the explicit plant", tc.name, fam, *got.Controller)
				}
			}
			if tc.sys.HasDelay() {
				continue
			}
			for _, fam := range []PidtuneType{PidtunePD, PidtunePID} {
				want, err := Pidtune(tc.sys, fam, 0, nil)
				if err != nil {
					t.Fatalf("Pidtune %s %s explicit: %v", tc.name, fam, err)
				}
				got, err := Pidtune(desc, fam, 0, nil)
				if err != nil {
					t.Errorf("Pidtune %s %s: %v", tc.name, fam, err)
					continue
				}
				if !pidGainsClose(&PID2{Kp: got.Kp, Ki: got.Ki, Kd: got.Kd, Tf: got.Tf}, &PID2{Kp: want.Kp, Ki: want.Ki, Kd: want.Kd, Tf: want.Tf}) {
					t.Errorf("Pidtune %s %s: descriptor %+v, explicit %+v", tc.name, fam, *got, *want)
				}
			}
		}
	}
}

// dss(A, B, C, D, eye(n)) is ss(A, B, C, D): the exact closed-loop test and
// the integral-zero detection take it.
func TestPIDTuningIdentityDescriptorScope(t *testing.T) {
	plant := makePlant(t, Poly{1, 1e-10}, Poly{1, 2, 1})
	id := plant.Copy()
	id.E = eyeDense(2)
	if _, ok := pidTuningIntegralZero(id, PidtunePI); !ok {
		t.Error("E=I: zero -1e-10 not detected")
	}
	cube := makePlant(t, Poly{1}, Poly{1, 3, 3, 1})
	cube.E = eyeDense(3)
	cs, err := (&PID{Kp: 8 - 1e-6}).System()
	if err != nil {
		t.Fatal(err)
	}
	if stable, ok := pidTuningExactStable(cube, cs); !stable || !ok {
		t.Errorf("E=I: exact test %v,%v, want true,true", stable, ok)
	}
	nonId := pidDescriptorOf(t, makePlant(t, Poly{1}, Poly{1, 3, 3, 1}), mat.NewDense(3, 3, []float64{2, 1, 0, 0, 1, 0, 0, 0, 1}))
	if _, ok := pidTuningExactStable(nonId, cs); ok {
		t.Error("E≠I: exact test claims scope")
	}
}
