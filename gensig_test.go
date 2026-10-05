package controlsys

import (
	"errors"
	"math"
	"testing"
)

func TestGenSigDefaultsMatchMATLAB(t *testing.T) {
	u, tt, err := GenSig("sine", 2, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(tt) != 5*64+1 || len(u) != len(tt) {
		t.Fatalf("len = %d/%d, want %d samples (Tf = 5*tau, Ts = tau/64)", len(tt), len(u), 5*64+1)
	}
	if math.Abs(tt[len(tt)-1]-10) > 1e-12 || math.Abs(tt[1]-2.0/64) > 1e-15 {
		t.Fatalf("t ends at %g, step %g; want 10 and %g", tt[len(tt)-1], tt[1], 2.0/64)
	}
	for k := range tt {
		if want := math.Sin(math.Pi * tt[k]); math.Abs(u[k]-want) > 1e-12 {
			t.Fatalf("u[%d] = %g, want %g", k, u[k], want)
		}
	}
}

func TestGenSigSquareAndPulse(t *testing.T) {
	u, tt, err := GenSig("square", 1, 2, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	wantSquare := []float64{0, 0, 1, 1, 0, 0, 1, 1, 0}
	if len(u) != len(wantSquare) || len(tt) != len(wantSquare) {
		t.Fatalf("square len = %d, want %d", len(u), len(wantSquare))
	}
	for k, w := range wantSquare {
		if u[k] != w {
			t.Fatalf("square u = %v, want %v", u, wantSquare)
		}
	}
	u, _, err = GenSig("pulse", 1, 3, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range u {
		want := 0.0
		if k == 4 || k == 8 || k == 12 {
			want = 1
		}
		if v != want {
			t.Fatalf("pulse u = %v, want unit samples at t = 1, 2, 3", u)
		}
	}
}

func TestGenSigRejects(t *testing.T) {
	for _, tc := range []struct {
		typ         string
		tau, tf, ts float64
	}{
		{"triangle", 1, 0, 0},
		{"step", 1, 0, 0},
		{"sine", 0, 0, 0},
		{"sine", -1, 0, 0},
		{"sine", math.NaN(), 0, 0},
		{"sine", math.Inf(1), 0, 0},
		{"sine", 1, -1, 0},
		{"sine", 1, math.NaN(), 0},
		{"sine", 1, 0, -0.1},
		{"sine", 1, 0, math.Inf(1)},
	} {
		if _, _, err := GenSig(tc.typ, tc.tau, tc.tf, tc.ts); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("GenSig(%q, %g, %g, %g) err = %v, want ErrInvalidArgument", tc.typ, tc.tau, tc.tf, tc.ts, err)
		}
	}
}
