package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
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

// 6RPB54: on G = C(sI−A)⁻¹B·e^{−0.00316s} with an undamped pair at
// 2.15e-7 rad/s and a zero at +5.0e-8 (N(s) = 0.831s − 4.16e-8), the forced
// sign is −1 and its PI fails; the mirrored PI (Kp 0.867, Ki 0.414) passed
// the Nyquist test, which missed the closed-loop root near +5e-8 seven
// decades below wc 0.83. Pidtune now returns a stable design or
// ErrPIDTuningTargetUnattainable for every family. Oracle: the real-axis
// characteristic P(s) = s·D(s) + (Kp·s + Ki)·N(s)·e^{−sτ} of the old PI
// changes sign on [0, 1] (a real RHP root), and no returned design with
// integral action does.
func TestPidtuneMirroredForcedSignSlowZero(t *testing.T) {
	a := []float64{-8.589e-24, 2.1544346900318827e-07, -2.1544346900318824e-07, -2.903e-23}
	b := []float64{-0.9667981925794609, -0.01238866346289056}
	c := []float64{-0.8568887533689471, -0.21079579943077972}
	const tau = 0.0031622776601683785
	p, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, b), mat.NewDense(1, 2, c), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	den := Poly{1, -(a[0] + a[3]), a[0]*a[3] - a[1]*a[2]}
	num := Poly{c[0]*b[0] + c[1]*b[1], c[0]*(a[1]*b[1]-a[3]*b[0]) + c[1]*(a[2]*b[0]-a[0]*b[1])}
	char := func(kp, ki, s float64) float64 {
		x := complex(s, 0)
		return s*real(den.Eval(x)) + (kp*s+ki)*real(num.Eval(x))*math.Exp(-s*tau)
	}
	if char(0.8673363870904787, 0.41363219216174363, 0)*char(0.8673363870904787, 0.41363219216174363, 1) >= 0 {
		t.Fatal("oracle: the old PI has no real RHP root")
	}
	for _, f := range pidtuneAllTypes {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", f, err)
			}
			continue
		}
		if pid.Ki != 0 && char(pid.Kp, pid.Ki, 0)*char(pid.Kp, pid.Ki, 1) < 0 {
			t.Errorf("%s: %+v has a real RHP closed-loop root", f, pid)
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, tau); ok && !stable {
			t.Errorf("%s: %+v unstable", f, pid)
		}
	}
}
