package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// HVXYLI: a descriptor plant is not realizable, so an ideal-derivative loop
// given a transfer reaches the transfer path with neither the exact nor the
// realization test, and an undecided float verdict there fails closed
// instead of standing. TransferFunction refuses descriptors, so the
// transfer is supplied directly.
// E⁻¹A, E⁻¹B are the companion form of 1/(s²+ws+w²), w = 1/64; with
// Kp = w², Kd = w the characteristic s³+2ws²+2w²s+Ki is on the Hurwitz
// boundary at Ki = 4w³, where the pair ±j√2·w moves by Re ≈ ΔKi/(12w²):
// ΔKi = ∓6e-13 puts it at Re ≈ ∓1.5e-10, beyond the pole tolerance but
// within the float error model. Oracle: exact Routh–Hurwitz.
func TestPIDTuningIdealStableDescriptorFailClosed(t *testing.T) {
	const w = 0x1p-6
	e := mat.NewDense(2, 2, []float64{2, 1, 0, 2})
	var a, b mat.Dense
	a.Mul(e, mat.NewDense(2, 2, []float64{0, 1, -w * w, -w}))
	b.Mul(e, mat.NewDense(2, 1, []float64{0, 1}))
	plant, err := NewDescriptor(&a, &b, mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), e, 0)
	if err != nil {
		t.Fatal(err)
	}
	if plant.E == nil {
		t.Fatal("descriptor E dropped")
	}
	transfer := func() (*TransferFuncResult, error) {
		return &TransferFuncResult{TF: &TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, w, w * w}}}}, nil
	}
	for _, tc := range []struct {
		ki            float64
		oracle, close bool
	}{
		{4*w*w*w + 6e-13, false, true},
		{4*w*w*w - 6e-13, true, true},
		{w * w * w, true, false},
		{10 * w * w * w, false, false},
	} {
		k := PID2{Kp: w * w, Ki: tc.ki, Kd: w, B: 1, C: 1}
		if pidtuneExactHurwitz(&PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd}, Poly{1}, Poly{1, w, w * w}) != tc.oracle {
			t.Fatalf("Ki %v: oracle not %v", tc.ki, tc.oracle)
		}
		if _, decided, err := pidTuningIdealFloat(Poly{1, 2 * w, 2 * w * w, tc.ki}, true); err != nil || decided == tc.close {
			t.Fatalf("Ki %v: float decided %v %v", tc.ki, decided, err)
		}
		want := tc.oracle && !tc.close
		if got, err := pidTuningIdealStable(plant, transfer, &k, 0); err != nil || got != want {
			t.Errorf("Ki %v: stable %v %v, want %v", tc.ki, got, err, want)
		}
	}
}

// 3TH3O6: E = I, as dss(A, B, C, D, eye(n)) builds, is a standard model, so
// its ideal-derivative loop takes the exact chain like E = nil. It was held
// unrealizable and failed closed on the transfer path, rejecting the stable
// Ki = 4w³ − 6e-13 loop of TestPIDTuningIdealStableDescriptorFailClosed.
// Oracle: exact Routh–Hurwitz.
func TestPIDTuningIdealStableIdentityDescriptor(t *testing.T) {
	const w = 0x1p-6
	plant, err := NewDescriptor(mat.NewDense(2, 2, []float64{0, 1, -w * w, -w}), mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), eyeDense(2), 0)
	if err != nil {
		t.Fatal(err)
	}
	if plant.E == nil {
		t.Fatal("E dropped")
	}
	transfer := func() (*TransferFuncResult, error) { return plant.rationalTransferFunction(nil) }
	for _, ki := range []float64{4*w*w*w - 6e-13, 4*w*w*w + 6e-13, w * w * w, 10 * w * w * w} {
		k := PID2{Kp: w * w, Ki: ki, Kd: w, B: 1, C: 1}
		want := pidtuneExactHurwitz(&PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd}, Poly{1}, Poly{1, w, w * w})
		if want != (ki < 4*w*w*w) {
			t.Fatalf("Ki %v: oracle %v", ki, want)
		}
		if got, err := pidTuningIdealStable(plant, transfer, &k, 0); err != nil || got != want {
			t.Errorf("Ki %v: stable %v %v, oracle %v", ki, got, err, want)
		}
	}
}
