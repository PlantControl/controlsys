package controlsys

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type missingMatrixCase struct {
	name string
	sys  *System
}

func missingDCases() []missingMatrixCase {
	var cases []missingMatrixCase
	for _, dname := range []string{"nilD", "emptyD"} {
		for _, dt := range []float64{0, 0.1} {
			for _, delay := range []string{"input", "internal"} {
				for _, n := range []int{1, 2} {
					var A, B, C *mat.Dense
					if n == 1 {
						A = mat.NewDense(1, 1, []float64{-1})
						B = mat.NewDense(1, 1, []float64{1})
						C = mat.NewDense(1, 1, []float64{1})
					} else {
						A = mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
						B = mat.NewDense(2, 1, []float64{1, 0.5})
						C = mat.NewDense(1, 2, []float64{1, -0.4})
					}
					if dt > 0 {
						A.Scale(0.4, A)
						A.Apply(func(i, j int, v float64) float64 {
							if i == j {
								return v + 1
							}
							return v
						}, A)
					}
					sys := &System{A: A, B: B, C: C, Dt: dt}
					if dname == "emptyD" {
						sys.D = &mat.Dense{}
					}
					if delay == "input" {
						sys.InputDelay = []float64{1}
					} else {
						sys.LFT = &LFTDelay{
							Tau: []float64{1},
							B2:  mat.NewDense(n, 1, nil),
							C2:  mat.NewDense(1, n, nil),
							D12: mat.NewDense(1, 1, []float64{0.5}),
							D21: mat.NewDense(1, 1, []float64{1}),
							D22: mat.NewDense(1, 1, nil),
						}
						sys.LFT.C2.Set(0, 0, 0.3)
					}
					cases = append(cases, missingMatrixCase{
						name: fmt.Sprintf("%s/dt=%g/%s/n=%d", dname, dt, delay, n),
						sys:  sys,
					})
				}
			}
		}
	}
	return cases
}

func TestValidateRejectsMissingD(t *testing.T) {
	for _, tc := range missingDCases() {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.sys.Validate(); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("Validate: got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func TestAnalysisRejectsMissingD(t *testing.T) {
	apis := []struct {
		name string
		call func(*System) error
	}{
		{"DiskMargin", func(s *System) error { _, err := DiskMargin(s); return err }},
		{"DiskMarginSkew", func(s *System) error { _, err := DiskMarginSkew(s, 0.5); return err }},
		{"Nyquist", func(s *System) error { _, err := s.Nyquist(nil, 0); return err }},
		{"Margin", func(s *System) error { _, err := Margin(s); return err }},
		{"AllMargin", func(s *System) error { _, err := AllMargin(s); return err }},
		{"Pidtune", func(s *System) error { _, err := Pidtune(s, PidtunePI, 0, nil); return err }},
		{"TunePID", func(s *System) error {
			_, err := TunePID(context.Background(), s, PidtunePI, PIDTuningOptions{})
			return err
		}},
	}
	for _, tc := range missingDCases() {
		for _, api := range apis {
			t.Run(tc.name+"/"+api.name, func(t *testing.T) {
				if err := api.call(tc.sys); !errors.Is(err, ErrDimensionMismatch) {
					t.Fatalf("got %v, want ErrDimensionMismatch", err)
				}
			})
		}
	}
}

func TestValidateRejectsMissingABC(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, -0.4})
	D := mat.NewDense(1, 1, []float64{0.1})
	for _, tc := range []struct {
		name string
		sys  *System
	}{
		{"nilA", &System{B: B, C: C, D: D}},
		{"nilB", &System{A: A, C: C, D: D}},
		{"emptyB", &System{A: A, B: &mat.Dense{}, C: C, D: D}},
		{"nilC", &System{A: A, B: B, D: D}},
		{"emptyC", &System{A: A, B: B, C: &mat.Dense{}, D: D}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.sys.Validate(); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("Validate: got %v, want ErrDimensionMismatch", err)
			}
			if _, err := DiskMargin(tc.sys); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("DiskMargin: got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func TestValidateAcceptsEmptyBlocksOfEmptyShape(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
	for _, tc := range []struct {
		name string
		sys  *System
	}{
		{"noIO", &System{A: A, B: &mat.Dense{}, C: &mat.Dense{}, D: &mat.Dense{}}},
		{"staticGain", &System{A: &mat.Dense{}, B: &mat.Dense{}, C: &mat.Dense{}, D: mat.NewDense(1, 2, []float64{1, 2})}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.sys.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
		})
	}
}

func TestNewFillsMissingDWithZeros(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, -0.4})
	want, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	wantDM, err := DiskMargin(want)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		D    *mat.Dense
	}{{"nilD", nil}, {"emptyD", &mat.Dense{}}} {
		t.Run(tc.name, func(t *testing.T) {
			for _, ctor := range []struct {
				name string
				f    func() (*System, error)
			}{
				{"New", func() (*System, error) { return New(A, B, C, tc.D, 0) }},
				{"newNoCopy", func() (*System, error) { return newNoCopy(A, B, C, tc.D, 0) }},
			} {
				sys, err := ctor.f()
				if err != nil {
					t.Fatalf("%s: %v", ctor.name, err)
				}
				if !mat.Equal(sys.D, want.D) {
					t.Fatalf("%s: D = %v, want 1×1 zero", ctor.name, mat.Formatted(sys.D))
				}
				dm, err := DiskMargin(sys)
				if err != nil {
					t.Fatalf("%s: DiskMargin: %v", ctor.name, err)
				}
				if dm.Alpha != wantDM.Alpha || dm.GainMargin != wantDM.GainMargin || dm.PhaseMargin != wantDM.PhaseMargin {
					t.Fatalf("%s: DiskMargin %+v, want %+v", ctor.name, dm, wantDM)
				}
			}
		})
	}
}
