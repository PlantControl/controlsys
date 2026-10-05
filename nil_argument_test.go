package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestNilArgumentsReturnInvalidArgument(t *testing.T) {
	var nilSys *System
	sys := makeSISO(-1, 1, 1, 0)
	eye1 := mat.NewDense(1, 1, []float64{1})
	tests := []struct {
		name string
		call func() error
	}{
		{"Inv", func() error { _, err := Inv(nilSys); return err }},
		{"Gram", func() error { _, err := Gram(nilSys, GramControllability); return err }},
		{"H2Syn", func() error { _, err := H2Syn(nilSys, 1, 1); return err }},
		{"HinfSyn", func() error { _, err := HinfSyn(nilSys, 1, 1); return err }},
		{"LFT/M", func() error { _, err := LFT(nilSys, sys, 1, 1); return err }},
		{"Linearize", func() error { _, err := Linearize(nil, mat.NewVecDense(1, nil), mat.NewVecDense(1, nil)); return err }},
		{"Loopsens/P", func() error { _, err := Loopsens(nilSys, sys); return err }},
		{"Loopsens/C", func() error { _, err := Loopsens(sys, nilSys); return err }},
		{"Lqg", func() error { _, err := Lqg(nilSys, eye1, eye1, nil); return err }},
		{"Lyap/A", func() error { _, err := Lyap(nil, eye1, nil); return err }},
		{"Lyap/Q", func() error { _, err := Lyap(eye1, nil, nil); return err }},
		{"DLyap/A", func() error { _, err := DLyap(nil, eye1, nil); return err }},
		{"DLyap/Q", func() error { _, err := DLyap(eye1, nil, nil); return err }},
		{"Reduce", func() error { _, err := nilSys.Reduce(nil); return err }},
		{"MinimalRealization", func() error { _, err := nilSys.MinimalRealization(); return err }},
		{"ModalTruncate", func() error { _, err := ModalTruncate(nilSys, nil); return err }},
		{"Modsep", func() error { _, err := Modsep(nilSys, 1); return err }},
		{"SetInputName", func() error { return nilSys.SetInputName("u") }},
		{"SetOutputName", func() error { return nilSys.SetOutputName("y") }},
		{"SetStateName", func() error { return nilSys.SetStateName("x") }},
		{"SelectByIndex", func() error { _, err := nilSys.SelectByIndex([]int{0}, []int{0}); return err }},
		{"SelectByName", func() error { _, err := nilSys.SelectByName([]string{"u"}, []string{"y"}); return err }},
		{"ConnectByName", func() error { _, err := ConnectByName([]*System{nilSys}, nil, nil, nil); return err }},
		{"Norm", func() error { _, err := Norm(nilSys, NormH2); return err }},
		{"H2Norm", func() error { _, err := H2Norm(nilSys); return err }},
		{"HSV", func() error { _, err := HSV(nilSys); return err }},
		{"HinfNorm", func() error { _, _, err := HinfNorm(nilSys); return err }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("panic: %v", r)
					}
				}()
				err = tc.call()
			}()
			if !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("err = %v, want ErrInvalidArgument", err)
			}
		})
	}
}

func TestNonFiniteModelsRejectedBeforeLAPACK(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(1)} {
		sys, err := New(
			mat.NewDense(2, 2, []float64{-1, bad, 0.5, -2}),
			mat.NewDense(2, 2, []float64{1, 0, 0.3, 1}),
			mat.NewDense(2, 2, []float64{1, 0.2, 0, 1}),
			mat.NewDense(2, 2, []float64{0.1, 0, 0, 0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		eye4 := eyeDense(4)
		tests := map[string]func() error{
			"Gram":           func() error { _, err := Gram(sys, GramControllability); return err },
			"Reduce":         func() error { _, err := sys.Reduce(nil); return err },
			"Modsep":         func() error { _, err := Modsep(sys, 1); return err },
			"ModalTruncate":  func() error { _, err := ModalTruncate(sys, nil); return err },
			"Norm":           func() error { _, err := Norm(sys, math.Inf(1)); return err },
			"H2Norm":         func() error { _, err := H2Norm(sys); return err },
			"HSV":            func() error { _, err := HSV(sys); return err },
			"HinfNorm":       func() error { _, _, err := HinfNorm(sys); return err },
			"Lqg":            func() error { _, err := Lqg(sys, eye4, eye4, nil); return err },
			"matLog":         func() error { _, err := matLog(sys.A); return err },
			"matLogSpectral": func() error { _, err := matLogSpectral(sys.A); return err },
		}
		for name, call := range tests {
			if err := call(); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("%s with A[0,1] = %g: err = %v, want ErrInvalidArgument", name, bad, err)
			}
		}
	}
}
