package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestInvStaticGainSwapsIONames(t *testing.T) {
	g, err := NewGain(mat.NewDense(2, 2, []float64{2, 1, 0, 4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	g.InputName = []string{"u1", "u2"}
	g.OutputName = []string{"y1", "y2"}
	r, err := Inv(g)
	if err != nil {
		t.Fatal(err)
	}
	if !sameStrings(r.InputName, []string{"y1", "y2"}) || !sameStrings(r.OutputName, []string{"u1", "u2"}) {
		t.Fatalf("names in=%v out=%v, want in=[y1 y2] out=[u1 u2]", r.InputName, r.OutputName)
	}
	want := []float64{0.5, -0.125, 0, 0.25}
	for i, w := range want {
		if got := r.D.RawMatrix().Data[i]; !approxEqual(got, w, 1e-14) {
			t.Fatalf("D[%d] = %g, want %g", i, got, w)
		}
	}
}

func TestInvRejects(t *testing.T) {
	delayed := makeSISO(-1, 1, 1, 1)
	delayed.InputDelay = []float64{0.5}
	tests := []struct {
		name string
		sys  *System
		want error
	}{
		{"nil", nil, ErrInvalidArgument},
		{"delay", delayed, ErrDelayUnsupported},
	}
	for _, tc := range tests {
		if _, err := Inv(tc.sys); !errors.Is(err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, err, tc.want)
		}
	}
}
