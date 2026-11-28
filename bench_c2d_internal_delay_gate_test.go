package controlsys

import (
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// BenchmarkC2DInternalDelayDescriptor times C2D and C2DMap of internal-delay
// descriptors whose DC the hold gate judges: the band fold with C2 = 0 (the
// core's bar) and C2 ≠ 0 (the per-channel bar of the closed pencil), the
// stiff parts with C2 ≠ 0, a MIMO model with two delays, and the same
// model without E (not gated).
func BenchmarkC2DInternalDelayDescriptor(b *testing.B) {
	rows := map[string]*System{}
	for _, r := range internalDelayDCGateRows {
		if strings.HasSuffix(r.name, ":lft/212") || strings.HasSuffix(r.name, ":lftc/212") || strings.HasSuffix(r.name, ":lftc/256") {
			sys, err := NewDescriptor(r.A, r.B, r.C, r.D, r.E, 0)
			if err != nil {
				b.Fatal(err)
			}
			if err := sys.SetInternalDelay(r.tau, r.B2, r.C2, r.D12, r.D21, r.D22); err != nil {
				b.Fatal(err)
			}
			rows[r.name] = sys
		}
	}
	rows["mimo2"] = internalDelayGateSystem(b, mat.NewDense(2, 3, []float64{0.4, -0.1, 0.2, 0, 0.3, -0.5}),
		mat.NewDense(2, 2, []float64{0.2, -0.3, 0.1, 0.4}), mat.NewDense(2, 2, []float64{0.1, 0.05, -0.2, 0.3}))
	ss := rows["mimo2"].Copy()
	ss.E = nil
	rows["mimo2-ss"] = ss
	for _, name := range []string{"strad1m-lft:lft/212", "strad1m-lft:lftc/212", "strad2-lft:lftc/256", "mimo2", "mimo2-ss"} {
		sys := rows[name]
		if sys == nil {
			b.Fatal(name)
		}
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
			opts := C2DOptions{Method: method}
			b.Run(name+"/"+string(method)+"/C2D", func(b *testing.B) {
				for b.Loop() {
					_, _ = sys.C2D(0.01, opts)
				}
			})
			b.Run(name+"/"+string(method)+"/C2DMap", func(b *testing.B) {
				for b.Loop() {
					_, _, _ = sys.C2DMap(0.01, opts)
				}
			})
		}
	}
}

// BenchmarkHoldGateInternalDelay times the DC gate alone (newHoldGate and
// check of C2D's result) on the internal-delay rows of
// BenchmarkC2DInternalDelayDescriptor, and on the same rows' delay-free core
// (LFT dropped) with the core's own C2D result.
func BenchmarkHoldGateInternalDelay(b *testing.B) {
	type row struct {
		name      string
		sys, core *System
	}
	var rows []row
	for _, r := range internalDelayDCGateRows {
		if !strings.HasSuffix(r.name, ":lft/212") && !strings.HasSuffix(r.name, ":lftc/212") && !strings.HasSuffix(r.name, ":lftc/256") {
			continue
		}
		sys, err := NewDescriptor(r.A, r.B, r.C, r.D, r.E, 0)
		if err != nil {
			b.Fatal(err)
		}
		core := sys.Copy()
		if err := sys.SetInternalDelay(r.tau, r.B2, r.C2, r.D12, r.D21, r.D22); err != nil {
			b.Fatal(err)
		}
		rows = append(rows, row{r.name, sys, core})
	}
	mimo := internalDelayGateSystem(b, mat.NewDense(2, 3, []float64{0.4, -0.1, 0.2, 0, 0.3, -0.5}),
		mat.NewDense(2, 2, []float64{0.2, -0.3, 0.1, 0.4}), mat.NewDense(2, 2, []float64{0.1, 0.05, -0.2, 0.3}))
	core := mimo.Copy()
	core.LFT = nil
	rows = append(rows, row{"mimo2", mimo, core})
	for _, r := range rows {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
			for _, v := range []struct {
				name string
				s    *System
			}{{"lft", r.sys}, {"core", r.core}} {
				out, _ := v.s.C2D(0.01, C2DOptions{Method: method})
				if out == nil {
					b.Fatal(r.name, method)
				}
				b.Run(r.name+"/"+string(method)+"/"+v.name, func(b *testing.B) {
					for b.Loop() {
						_ = newHoldGate(v.s).check(out)
					}
				})
			}
		}
	}
}
