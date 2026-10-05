package controlsys

import (
	"math"
	"testing"
)

func TestGenSig_Step(t *testing.T) {
	tt, u, err := GenSig("step", 1.0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if len(tt) != 11 {
		t.Fatalf("len = %d, want 11", len(tt))
	}
	for k, v := range u {
		if v != 1.0 {
			t.Errorf("u[%d] = %f, want 1", k, v)
		}
	}
}

func TestGenSig_Sine(t *testing.T) {
	period := 1.0
	dt := 0.25
	tt, u, err := GenSig("sine", period, dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(tt) != 5 {
		t.Fatalf("len = %d, want 5", len(tt))
	}
	if math.Abs(u[0]) > 1e-12 {
		t.Errorf("u[0] = %f, want 0", u[0])
	}
	if math.Abs(u[1]-1.0) > 1e-12 {
		t.Errorf("u[1] = %f, want 1", u[1])
	}
	if math.Abs(u[2]) > 1e-12 {
		t.Errorf("u[2] = %f, want 0", u[2])
	}
	if math.Abs(u[3]+1.0) > 1e-12 {
		t.Errorf("u[3] = %f, want -1", u[3])
	}
}

func TestGenSig_Square(t *testing.T) {
	tt, u, err := GenSig("square", 1.0, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if len(tt) != 5 {
		t.Fatalf("len = %d, want 5", len(tt))
	}
	// Half-period boundary belongs to the second half;
	// and t = period starts a new period.
	want := []float64{1, 1, -1, -1, 1}
	for k := range u {
		if u[k] != want[k] {
			t.Errorf("u[%d] = %f, want %f", k, u[k], want[k])
		}
	}
}

func TestGenSig_Pulse(t *testing.T) {
	dt := 0.1
	_, u, err := GenSig("pulse", 1.0, dt)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(u[0]-1.0/dt) > 1e-12 {
		t.Errorf("u[0] = %f, want %f", u[0], 1.0/dt)
	}
	for k := 1; k < len(u); k++ {
		if u[k] != 0 {
			t.Errorf("u[%d] = %f, want 0", k, u[k])
		}
	}
}

func TestGenSig_UnknownType(t *testing.T) {
	_, _, err := GenSig("triangle", 1.0, 0.1)
	if err == nil {
		t.Fatal("expected error for unknown type")
	}
}

func TestGenSig_InvalidParams(t *testing.T) {
	_, _, err := GenSig("step", -1.0, 0.1)
	if err == nil {
		t.Fatal("expected error for negative period")
	}
	_, _, err = GenSig("step", 1.0, 0)
	if err == nil {
		t.Fatal("expected error for zero dt")
	}
}

func TestGenSig_IncludesFinalSample(t *testing.T) {
	tt, u, err := GenSig("sine", 0.3, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if len(tt) != 4 || math.Abs(tt[3]-0.3) > 1e-12 {
		t.Fatalf("t = %v, want 4 samples ending at 0.3", tt)
	}
	if math.Abs(u[3]) > 1e-12 {
		t.Errorf("u(period) = %g, want 0", u[3])
	}
	for _, bad := range [][2]float64{{math.NaN(), 0.1}, {math.Inf(1), 0.1}, {1, math.NaN()}, {1, math.Inf(1)}} {
		if _, _, err := GenSig("sine", bad[0], bad[1]); err == nil {
			t.Errorf("GenSig(%g, %g): want error", bad[0], bad[1])
		}
	}
}
