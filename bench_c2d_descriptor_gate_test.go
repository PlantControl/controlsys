package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// BenchmarkC2DDescriptorRoutes times C2D of frozen EJDS3U rows on each
// descriptor route: band parts (cond/13), dense fold passing the DC check
// (cond/644 at 0.01), fold failing it with the real Schur retry (cond/644
// at 1, cond/111 at 0.01), the reduced standard form (stfr/45) and the
// refused band split (strad1m/47).
func BenchmarkC2DDescriptorRoutes(b *testing.B) {
	for _, tc := range []struct {
		row    string
		dt     float64
		method C2DMethod
	}{
		{"cond/13", 0.01, C2DMethodZOH},
		{"cond/13", 0.01, C2DMethodTustin},
		{"cond/13", 0.01, C2DMethodFOH},
		{"strad1m/47", 0.01, C2DMethodFOH},
		{"cond/644", 0.01, C2DMethodZOH},
		{"cond/644", 1, C2DMethodZOH},
		{"cond/111", 0.01, C2DMethodZOH},
		{"stfr/45", 0.01, C2DMethodZOH},
		{"strad1m/47", 0.01, C2DMethodZOH},
	} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == tc.row {
				var err error
				if sys, err = r.system(); err != nil {
					b.Fatal(err)
				}
			}
		}
		b.Run(tc.row+"/"+string(tc.method)+"/"+formatBenchDt(tc.dt), func(b *testing.B) {
			for b.Loop() {
				_, _ = sys.C2D(tc.dt, C2DOptions{Method: tc.method})
			}
		})
	}
}

func formatBenchDt(dt float64) string {
	switch dt {
	case 0.01:
		return "dt0.01"
	case 1:
		return "dt1"
	}
	return "dt"
}

// BenchmarkC2DExplicitMethods times ZOH, FOH and Tustin C2D of explicit
// models (E = nil), which the descriptor DC check does not touch.
func BenchmarkC2DExplicitMethods(b *testing.B) {
	for _, n := range []int{4, 8} {
		A := mat.NewDense(n, n, nil)
		B := mat.NewDense(n, 1, nil)
		C := mat.NewDense(1, n, nil)
		for i := range n {
			for j := range n {
				A.Set(i, j, 0.3*math.Sin(float64(3*i+7*j+1)))
			}
			A.Set(i, i, A.At(i, i)-1-float64(i))
			B.Set(i, 0, math.Cos(float64(i)))
			C.Set(0, i, math.Sin(float64(2*i+1)))
		}
		sys, err := New(A, B, C, mat.NewDense(1, 1, []float64{0.1}), 0)
		if err != nil {
			b.Fatal(err)
		}
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
			b.Run(fmt.Sprintf("n%d/%s", n, m), func(b *testing.B) {
				for b.Loop() {
					_, _ = sys.C2D(0.01, C2DOptions{Method: m})
				}
			})
		}
	}
}

// BenchmarkTimeResponseDescriptorRoutes times Step, Lsim, Impulse and
// Initial of frozen EJDS3U rows, which discretize through c2dUnguarded and
// take no DC check.
func BenchmarkTimeResponseDescriptorRoutes(b *testing.B) {
	for _, row := range []string{"cond/13", "cond/644", "stfr/45"} {
		var sys *System
		for _, r := range descriptorDCGateRows {
			if r.name == row {
				var err error
				if sys, err = r.system(); err != nil {
					b.Fatal(err)
				}
			}
		}
		n, _, _ := sys.Dims()
		tt := make([]float64, 201)
		u := mat.NewDense(201, 1, nil)
		for k := range tt {
			tt[k] = 0.01 * float64(k)
			u.Set(k, 0, math.Sin(0.3*float64(k)))
		}
		x0 := mat.NewVecDense(n, nil)
		x0.SetVec(0, 1)
		b.Run(row+"/Step", func(b *testing.B) {
			for b.Loop() {
				_, _ = Step(sys, 2)
			}
		})
		b.Run(row+"/Lsim", func(b *testing.B) {
			for b.Loop() {
				_, _ = Lsim(sys, u, tt, nil)
			}
		})
		b.Run(row+"/Impulse", func(b *testing.B) {
			for b.Loop() {
				_, _ = Impulse(sys, 2)
			}
		})
		b.Run(row+"/Initial", func(b *testing.B) {
			for b.Loop() {
				_, _ = Initial(sys, x0, 2)
			}
		})
	}
}
