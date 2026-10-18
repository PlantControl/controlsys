package controlsys

import (
	"fmt"
	"math"
	"testing"
)

// TestPidtuneZeroDCMirror pins Pidtune(−G) = −Pidtune(G) on stable plants
// with zero DC gain (SJ5QTW): both signs get the design whichever sign the
// high-frequency guess picks first.
func TestPidtuneZeroDCMirror(t *testing.T) {
	for _, c := range []struct {
		num, den  Poly
		dt, delay float64
		fams      []PidtuneType
	}{
		{Poly{0.5, 0}, Poly{1, 2}, 0, 0, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{0.0563, -0.3311, 0}, Poly{1, 7.063, 15.54, 12.54}, 0, 0, []PidtuneType{PidtuneP, PidtunePD, PidtunePDF}},
		{Poly{0.4, -0.4}, Poly{1, -0.6}, 0.1, 0, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{-0.3, 0.3, 0}, Poly{1, -0.9, 0.2}, 0.05, 0, []PidtuneType{PidtuneP, PidtunePD}},
	} {
		for _, f := range c.fams {
			var got [2]*PID
			for i, sg := range []float64{1, -1} {
				num := make(Poly, len(c.num))
				for j := range num {
					num[j] = sg * c.num[j]
				}
				name := fmt.Sprintf("%s %v/%v dt=%g", f, num, c.den, c.dt)
				pid, err := Pidtune(pidtuneBackoffPlant(t, num, c.den, c.dt, c.delay), f, 0, nil)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				got[i] = pid
				if stable, ok := pidtuneOracleStable(t, pid, num, c.den, c.dt, c.delay); !ok || !stable {
					t.Errorf("%s: %+v unstable", name, pid)
				}
				wMax := 1e6
				if c.dt > 0 {
					wMax = math.Pi / c.dt
				}
				if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, c.den, c.dt, c.delay), 1e-6, wMax); ms > 4.5 {
					t.Errorf("%s: %+v |S| peak %.3g", name, pid, ms)
				}
			}
			if got[0] == nil || got[1] == nil {
				continue
			}
			for _, g := range [][2]float64{{got[0].Kp, -got[1].Kp}, {got[0].Ki, -got[1].Ki}, {got[0].Kd, -got[1].Kd}, {got[0].Tf, got[1].Tf}} {
				if math.Abs(g[0]-g[1]) > 1e-9*math.Max(1, math.Abs(g[0])) {
					t.Errorf("%s %v/%v: G gives %+v, −G gives %+v, not mirrored", f, c.num, c.den, got[0], got[1])
					break
				}
			}
		}
	}
}
