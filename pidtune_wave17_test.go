package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// AMWRJU: Pidtune's PD is C = Kp(1 + Td·s/(Td/10·s + 1)), a lead of ratio
// 11 whose phase never exceeds asin(10/12) = 56.44°. On 1/(s(s−1)) and
// 1/(s²−0.5s+1) the plant phase tends to −180° only as ω → ∞, so no PD of
// that family keeps 58.5° at a gain crossover; P on 1/((s+5)(s−1)) peaks at
// atan√5 − atan(√5/5) = 41.8°, and the discrete PD on 1/((z−1)(z−1.1)) at
// about 30°. TunePID's PD reaches 57° on the first two only because it is
// ideal (Tf = 0). Pidtune fails closed; PDF, with Tf = 0.1/wc, is robust on
// the continuous two. Oracle: the best smallest margin over a (K, Td) grid of
// loops stable by companion-matrix closed-loop roots stays below 58.5°.
func TestPidtuneFilteredPDUnattainable(t *testing.T) {
	if best := pidtuneFamilyBestMargin(t, Poly{1}, Poly{1, -1}, 0, false); !(best > 58.5) {
		t.Fatalf("oracle: P on 1/(s−1) reaches %.2f°, want ≈ 90°", best)
	}
	if best := pidtuneFamilyBestMargin(t, Poly{1}, Poly{1, 1, 0}, 0, true); !(best > 58.5) {
		t.Fatalf("oracle: PD on 1/(s(s+1)) reaches %.2f°, want > 58.5°", best)
	}
	for _, c := range []struct {
		num, den Poly
		dt       float64
		typ      PidtuneType
		pdf      bool
	}{
		{Poly{1}, Poly{1, -1, 0}, 0, PidtunePD, true},
		{Poly{1}, Poly{1, -0.5, 1}, 0, PidtunePD, true},
		{Poly{1}, Poly{1, -2.1, 1.1}, 0.1, PidtunePD, false},
		{Poly{1}, Poly{1, 4, -5}, 0, PidtuneP, false},
	} {
		name := fmt.Sprintf("%s %v/%v dt=%g", c.typ, c.num, c.den, c.dt)
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		if best := pidtuneFamilyBestMargin(t, c.num, c.den, c.dt, c.typ == PidtunePD); !(best < 58.5) {
			t.Fatalf("%s: oracle reaches %.2f°", name, best)
		}
		if pid, err := Pidtune(p, c.typ, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s: got %+v, %v; want ErrPIDTuningTargetUnattainable", name, pid, err)
		}
		if !c.pdf {
			continue
		}
		pid, err := Pidtune(p, PidtunePDF, 0, nil)
		if err != nil {
			t.Errorf("%s PDF: %v", name, err)
			continue
		}
		pidtuneCheckRobust(t, name+" PDF", pid, c.num, c.den, c.dt, pidTuningMaxSensitivity)
	}
}

// pidtuneFamilyBestMargin returns the largest smallest gain-crossover margin
// 180° − |∠L| over a log grid of K (either sign) and, for pd, Td with
// Tf = Td/10 (forward-Euler derivative in z), among loops whose closed-loop
// roots lie in the stability region.
func pidtuneFamilyBestMargin(t *testing.T, num, den Poly, dt float64, pd bool) float64 {
	t.Helper()
	wmax := 1e3
	if dt > 0 {
		wmax = math.Pi / dt
	}
	var xs []complex128
	for lw := -3.0; lw < math.Log10(wmax); lw += 0.005 {
		w := math.Pow(10, lw)
		x := complex(0, w)
		if dt > 0 {
			x = cmplx.Exp(complex(0, w*dt))
		}
		xs = append(xs, x)
	}
	tds := []float64{0}
	if pd {
		tds = tds[:0]
		for lt := -3.0; lt <= 3; lt += 0.05 {
			tds = append(tds, math.Pow(10, lt))
		}
	}
	best := math.Inf(-1)
	for _, td := range tds {
		nc, dc := Poly{1}, Poly{1}
		if td > 0 {
			tf := td / 10
			nc, dc = Poly{tf + td, 1}, Poly{tf, 1}
			if dt > 0 {
				nc, dc = Poly{tf + td, dt - tf - td}, Poly{tf, dt - tf}
			}
		}
		for _, s := range []float64{1, -1} {
			for lk := -3.0; lk <= 3; lk += 0.02 {
				k := s * math.Pow(10, lk)
				if !pidtuneOracleRootsStable(t, den.Mul(dc).Add(num.Mul(nc).Scale(k)), dt) {
					continue
				}
				pm, prev := math.Inf(1), 0.0
				for i, x := range xs {
					l := complex(k, 0) * num.Eval(x) * nc.Eval(x) / (den.Eval(x) * dc.Eval(x))
					a := cmplx.Abs(l) - 1
					if i > 0 && (a >= 0) != (prev >= 0) {
						pm = math.Min(pm, 180-math.Abs(cmplx.Phase(l))*180/math.Pi)
					}
					prev = a
				}
				if !math.IsInf(pm, 1) {
					best = math.Max(best, pm)
				}
			}
		}
	}
	return best
}

func pidtuneOracleRootsStable(t *testing.T, p Poly, dt float64) bool {
	t.Helper()
	for len(p) > 1 && p[0] == 0 {
		p = p[1:]
	}
	n := len(p) - 1
	m := mat.NewDense(n, n, nil)
	for j := range n {
		m.Set(0, j, -p[j+1]/p[0])
	}
	for i := 1; i < n; i++ {
		m.Set(i, i-1, 1)
	}
	var e mat.Eigen
	if !e.Factorize(m, mat.EigenNone) {
		t.Fatalf("eigen of %v failed", p)
	}
	for _, r := range e.Values(nil) {
		if dt == 0 && real(r) >= 0 || dt > 0 && cmplx.Abs(r) >= 1 {
			return false
		}
	}
	return true
}
