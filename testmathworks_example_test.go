package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestMathWorksFeedbackExample(t *testing.T) {
	A_g := mat.NewDense(1, 1, []float64{-10})
	B_g := mat.NewDense(1, 1, []float64{1})
	C_g := mat.NewDense(1, 1, []float64{1})
	D_g := mat.NewDense(1, 1, []float64{0})
	G, _ := New(A_g, B_g, C_g, D_g, 0)
	G.InputDelay = []float64{2.1}

	A_c := mat.NewDense(1, 1, []float64{0})
	B_c := mat.NewDense(1, 1, []float64{1})
	C_c := mat.NewDense(1, 1, []float64{2.3})
	D_c := mat.NewDense(1, 1, []float64{0.5})
	C, _ := New(A_c, B_c, C_c, D_c, 0)

	L, err := Series(C, G)
	if err != nil {
		t.Fatalf("Series error: %v", err)
	}

	T, err := Feedback(L, nil, -1)
	if err != nil {
		t.Fatalf("Feedback error: %v", err)
	}

	if !T.HasInternalDelay() || len(T.LFT.Tau) != 1 {
		t.Errorf("Expected 1 internal delay, got %d", len(T.LFT.Tau))
	} else if T.LFT.Tau[0] != 2.1 {
		t.Errorf("Expected internal delay of 2.1, got %f", T.LFT.Tau[0])
	}
}
