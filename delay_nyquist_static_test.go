package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// staticInternalDelayLoop is L(s) = 0.2 + 0.3·e^{−s} as a state-free model
// whose B2 and C2 blocks are empty: |1+L| ranges over [0.9, 1.5], extremes at
// ω = π and 0. Discrete delays count samples: with dt = 0.1 the delay is
// z^−10, equal to e^{−jω} on the unit circle.
func staticInternalDelayLoop(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewGain(mat.NewDense(1, 1, []float64{0.2}), dt)
	if err != nil {
		t.Fatal(err)
	}
	tau := 1.0
	if dt > 0 {
		tau = 1 / dt
	}
	if err := sys.SetInternalDelay([]float64{tau}, nil, nil,
		mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0})); err != nil {
		t.Fatal(err)
	}
	return sys
}

// Pins YU3OKE: the rational tail bound reads D12(0,i) and D21(i,0), so the
// state-free internal-delay loop, and its Copy with empty B2/C2, must reach
// it with 1×N and N×1 blocks.
func TestNyquistInternalDelayStaticGain(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		base := staticInternalDelayLoop(t, dt)
		for name, sys := range map[string]*System{"orig": base, "copy": base.Copy()} {
			t.Run(fmt.Sprintf("dt=%g/%s", dt, name), func(t *testing.T) {
				res, err := sys.Nyquist([]float64{0, math.Pi}, 0)
				if err != nil {
					t.Fatal(err)
				}
				want := []complex128{0.5, -0.1}
				if len(res.Omega) != 2 || len(res.Contour) != 2 || len(res.ContourN) != 2 {
					t.Fatalf("lengths Omega %d Contour %d ContourN %d, want 2", len(res.Omega), len(res.Contour), len(res.ContourN))
				}
				for k, w := range want {
					if cmplx.Abs(res.Contour[k]-w) > 1e-12 {
						t.Errorf("Contour[%d] = %v, want %v", k, res.Contour[k], w)
					}
				}
				if res.Encirclements != 0 || res.RHPPoles != 0 || res.RHPZerosCL != 0 {
					t.Errorf("Encirclements %d RHPPoles %d RHPZerosCL %d, want 0 0 0", res.Encirclements, res.RHPPoles, res.RHPZerosCL)
				}
			})
		}
	}
}

func TestDiskMarginInternalDelayStaticGain(t *testing.T) {
	alpha := 18.0 / 11
	for _, dt := range []float64{0, 0.1} {
		base := staticInternalDelayLoop(t, dt)
		for name, sys := range map[string]*System{"orig": base, "copy": base.Copy()} {
			t.Run(fmt.Sprintf("dt=%g/%s", dt, name), func(t *testing.T) {
				dm, err := DiskMargin(sys)
				if err != nil {
					t.Fatal(err)
				}
				near := func(field string, got, want float64) {
					t.Helper()
					if math.Abs(got-want) > 1e-6*max(1, math.Abs(want)) {
						t.Errorf("%s = %.12g, want %.12g", field, got, want)
					}
				}
				near("Alpha", dm.Alpha, alpha)
				near("GainMargin[0]", dm.GainMargin[0], 0.1)
				near("GainMargin[1]", dm.GainMargin[1], 10)
				near("PhaseMargin", dm.PhaseMargin, 2*math.Atan(alpha/2)*180/math.Pi)
				near("PeakSensitivity", dm.PeakSensitivity, 1/0.9)
				w, ok := dm.Frequency()
				if !ok {
					t.Fatal("Frequency not ok for a stable loop")
				}
				near("cos(ω·τ) at disk-margin ω", math.Cos(w), -1)

				skew, err := DiskMarginSkew(sys, 1)
				if err != nil {
					t.Fatal(err)
				}
				near("Alpha(σ=1)", skew.Alpha, 0.9)
				near("PeakSensitivity(σ=1)", skew.PeakSensitivity, 1/0.9)
			})
		}
	}
}

func TestDelayLoopInternalDelayShapeRejected(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		mimo, err := NewGain(mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.1}), dt)
		if err != nil {
			t.Fatal(err)
		}
		if err := mimo.SetInternalDelay([]float64{1}, nil, nil,
			mat.NewDense(2, 1, []float64{0.3, 0}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0})); err != nil {
			t.Fatal(err)
		}
		emptyD12 := &System{D: mat.NewDense(1, 1, []float64{0.2}), Dt: dt,
			LFT: &LFTDelay{Tau: []float64{1}, D12: &mat.Dense{}, D21: mat.NewDense(1, 1, []float64{1}), D22: mat.NewDense(1, 1, nil)}}
		for _, c := range []struct {
			name string
			sys  *System
			want error
		}{{"mimo", mimo, ErrNotSISO}, {"emptyD12", emptyD12, ErrDimensionMismatch}} {
			t.Run(fmt.Sprintf("dt=%g/%s", dt, c.name), func(t *testing.T) {
				if _, err := c.sys.Nyquist(nil, 0); !errors.Is(err, c.want) {
					t.Errorf("Nyquist err = %v, want %v", err, c.want)
				}
				if _, err := DiskMargin(c.sys); !errors.Is(err, c.want) {
					t.Errorf("DiskMargin err = %v, want %v", err, c.want)
				}
			})
		}
	}
}
