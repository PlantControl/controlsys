package controlsys

import (
	"errors"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestGroup3ErrorsWrapSentinelWithOpPrefix(t *testing.T) {
	siso := makeSISO(-1, 1, 1, 0)
	named := makeSISO(-1, 1, 1, 0)
	named.InputName = []string{"u"}
	named.OutputName = []string{"y"}
	mimo, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	model := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense { return x },
		H: func(x, u *mat.VecDense) *mat.VecDense { return x },
		N: 1, M: 1, P: 1,
	}
	tests := []struct {
		name   string
		prefix string
		want   error
		call   func() error
	}{
		{"Norm type", "Norm: ", ErrInvalidArgument, func() error { _, err := Norm(siso, 3); return err }},
		{"Loopsens dims", "Loopsens: ", ErrDimensionMismatch, func() error { _, err := Loopsens(siso, mimo); return err }},
		{"LFT negative", "LFT: ", ErrInvalidArgument, func() error { _, err := LFT(siso, siso, LFTFeedback{Nu: -1}); return err }},
		{"Linearize nil x0", "Linearize: ", ErrInvalidArgument, func() error { _, err := Linearize(model, nil, mat.NewVecDense(1, nil)); return err }},
		{"SelectByIndex range", "SelectByIndex: ", ErrInvalidArgument, func() error { _, err := siso.SelectByIndex([]int{1}, []int{0}); return err }},
		{"SelectByName missing", "SelectByName: ", ErrSignalNotFound, func() error { _, err := named.SelectByName([]string{"v"}, []string{"y"}); return err }},
		{"ConnectByName missing", "ConnectByName: ", ErrSignalNotFound, func() error {
			_, err := ConnectByName([]*System{named}, []Connection{{From: "z", To: "u"}}, []string{"u"}, []string{"y"})
			return err
		}},
		{"H2Syn D11", "H2Syn: ", ErrNoFiniteH2Norm, func() error { _, err := H2Syn(mimo, 1, 1); return err }},
		{"HinfSyn partition", "HinfSyn: ", ErrInvalidPartition, func() error { _, err := HinfSyn(mimo, 3, 1); return err }},
		{"Lyap dims", "Lyap: ", ErrDimensionMismatch, func() error { _, err := Lyap(mimo.A, mat.NewDense(1, 1, []float64{1}), nil); return err }},
		{"DLyap symmetric", "DLyap: ", ErrNotSymmetric, func() error { _, err := DLyap(mimo.A, mimo.A, nil); return err }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if msg := err.Error(); !strings.HasPrefix(msg, tc.prefix) || strings.Count(msg, "controlsys:") != 1 {
				t.Fatalf("err = %q, want prefix %q and one controlsys: sentinel text", msg, tc.prefix)
			}
		})
	}
}

func TestInvertSmallSingularIsSingularTransform(t *testing.T) {
	if _, err := invertSmall(mat.NewDense(2, 2, []float64{1, 2, 2, 4}), 2); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("err = %v, want ErrSingularTransform", err)
	}
	if _, err := matLog(mat.NewDense(2, 3, nil)); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("matLog err = %v, want ErrDimensionMismatch", err)
	}
}

func TestGeneralizedClosedLoopErrorsUseCallerOp(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	k, err := NewTunableReal("K", 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := k.SetBounds(0, 2); err != nil {
		t.Fatal(err)
	}
	gain, err := NewTunableGain("K", [][]*TunableReal{{k}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	loop, err := NewGeneralizedClosedLoop("cl", plant, gain, "y")
	if err != nil {
		t.Fatal(err)
	}
	for op, call := range map[string]func() error{
		"GeneralizedClosedLoop.AnalysisPoint: ":            func() error { _, err := loop.AnalysisPoint("x"); return err },
		"GeneralizedClosedLoop.OpenLoop: ":                 func() error { _, err := loop.OpenLoop("x"); return err },
		"GeneralizedClosedLoop.ClosedLoop: ":               func() error { _, err := loop.ClosedLoop("x"); return err },
		"GeneralizedClosedLoop.ComplementarySensitivity: ": func() error { _, err := loop.ComplementarySensitivity("x"); return err },
		"GeneralizedClosedLoop.Sensitivity: ":              func() error { _, err := loop.Sensitivity("x"); return err },
	} {
		err := call()
		if !errors.Is(err, ErrSignalNotFound) || !strings.HasPrefix(err.Error(), op) || strings.Count(err.Error(), "GeneralizedClosedLoop.") != 1 {
			t.Errorf("err = %v, want single prefix %q and ErrSignalNotFound", err, op)
		}
	}
}
