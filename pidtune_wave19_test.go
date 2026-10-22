package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// G6APMO: an explicit-wc PD tries MATLAB's ideal Kp + Kd·s first. On
// 1/(s(s−1)) at wc = 3 the 60° margin needs a lead of 150° − atan 3 ≈ 78.4°,
// above the 56.4° a Td/10 filtered PD reaches, so Pidtune returned
// ErrPIDTuningTargetUnattainable. Oracle: |C(j3)| = 1/|G(j3)| = 3√10 at
// that lead gives Kp = 3√10·cos φ, Kd = √10·sin φ, and s² + (Kd−1)s + Kp is
// Hurwitz iff Kd > 1 and Kp > 0.
func TestPidtuneExplicitWcIdealPD(t *testing.T) {
	const wc = 3.0
	num, den := Poly{1}, Poly{1, -1, 0}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	phi := (150 - math.Atan(wc)*180/math.Pi) * math.Pi / 180
	kp, kd := 3*math.Sqrt(10)*math.Cos(phi), math.Sqrt(10)*math.Sin(phi)
	if !(kd > 1 && kp > 0) {
		t.Fatalf("oracle: Kp %g, Kd %g not stabilizing", kp, kd)
	}
	for _, opts := range []*PidtuneOptions{nil, {PhaseMargin: 60}} {
		pid, err := Pidtune(p, PidtunePD, wc, opts)
		if err != nil {
			t.Fatalf("opts %+v: %v", opts, err)
		}
		if pid.Tf != 0 || pid.Ki != 0 || math.Abs(pid.Kp-kp) > 1e-9*kp || math.Abs(pid.Kd-kd) > 1e-9*kd {
			t.Errorf("opts %+v: %+v, want Kp %g Kd %g Tf 0", opts, pid, kp, kd)
		}
		l := pidtuneOracleLoop(pid, num, den, 0, 0)(wc)
		if math.Abs(cmplx.Abs(l)-1) > 1e-9 || math.Abs(180-math.Abs(cmplx.Phase(l))*180/math.Pi-60) > 1e-6 {
			t.Errorf("opts %+v: L(j%g) = %v, want |L| 1 and 60° margin", opts, wc, l)
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
			t.Errorf("opts %+v: %+v unstable", opts, pid)
		}
	}
}
