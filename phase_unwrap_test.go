package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func multiTurnOmega() []float64 {
	omega := make([]float64, 81)
	for k := range omega {
		omega[k] = float64(k) * math.Pi / 10
	}
	return omega
}

func TestPhaseUnwrap_DelayMultipleTurns(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 2, []float64{1, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{1, 2.5}); err != nil {
		t.Fatal(err)
	}
	omega := multiTurnOmega()
	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	nichols, err := sys.Nichols(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	delays := []float64{1, 2.5}
	for name, phase := range map[string]func(int, int, int) float64{
		"System.Bode":    bode.PhaseAt,
		"System.Nichols": nichols.PhaseAt,
		"FRD.Bode":       frd.Bode().PhaseAt,
	} {
		t.Run(name, func(t *testing.T) {
			for j, tau := range delays {
				for k, w := range omega {
					want := -w * tau * 180 / math.Pi
					if got := phase(k, 0, j); math.Abs(got-want) > 1e-9 {
						t.Fatalf("input %d sample %d omega=%g: phase=%g, want %g", j, k, w, got, want)
					}
				}
			}
		})
	}
}

func TestPhaseUnwrap_PositiveRotation(t *testing.T) {
	omega := multiTurnOmega()
	resp := make([][][]complex128, len(omega))
	for k, w := range omega {
		resp[k] = [][]complex128{{cmplx.Exp(complex(0, 2*w))}}
	}
	f, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	bode := f.Bode()
	for k, w := range omega {
		want := 2 * w * 180 / math.Pi
		if got := bode.PhaseAt(k, 0, 0); math.Abs(got-want) > 1e-9 {
			t.Fatalf("sample %d omega=%g: phase=%g, want %g", k, w, got, want)
		}
	}
}

func TestUnwrapBodePhase(t *testing.T) {
	nan := math.NaN()
	tests := []struct {
		name string
		in   []float64
		want []float64
	}{
		{"plus180 boundary kept", []float64{0, 180}, []float64{0, 180}},
		{"minus180 boundary kept", []float64{0, -180}, []float64{0, -180}},
		{"plus540 jump to plus180", []float64{0, 540}, []float64{0, 180}},
		{"minus540 jump to minus180", []float64{0, -540}, []float64{0, -180}},
		{"first point kept", []float64{-200, 150}, []float64{-200, -210}},
		{"two turns down", []float64{-170, 170, 150, 100, 30, -60, -150, 120}, []float64{-170, -190, -210, -260, -330, -420, -510, -600}},
		{"two turns up", []float64{170, -170, -100, 0, 100, 175, -175}, []float64{170, 190, 260, 360, 460, 535, 545}},
		{"nan skipped", []float64{0, -170, nan, 170}, []float64{0, -170, nan, -190}},
		{"leading nan", []float64{nan, 10, -170}, []float64{nan, 10, -170}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := append([]float64(nil), tc.in...)
			unwrapBodePhase(got, 1, 1, len(got))
			for k := range got {
				if math.IsNaN(tc.want[k]) {
					if !math.IsNaN(got[k]) {
						t.Fatalf("got %v, want %v", got, tc.want)
					}
					continue
				}
				if got[k] != tc.want[k] {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			}
		})
	}
}
