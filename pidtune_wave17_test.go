package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// AMWRJU: MATLAB pidtune's PD is Kp + Kd·s; the Td/10-filtered PD
// C = Kp(1 + Td·s/(Td/10·s + 1)) is a lead of ratio 11 whose phase never
// exceeds asin(10/12) = 56.44°. On 1/(s(s−1)) and 1/(s²−0.5s+1) the plant
// phase tends to −180° only as ω → ∞, so no filtered PD keeps 58.5° at a
// gain crossover and Pidtune returned ErrPIDTuningTargetUnattainable; the
// ideal PD, Pidtune's last fallback, reaches it. P on 1/((s+5)(s−1)) peaks
// at atan√5 − atan(√5/5) = 41.8°, and either PD on 1/((z−1)(z−1.1)) below
// 58.5°, so those fail closed; on 2.15/((z−1.006)(z−1.018)), Dt 0.01, only
// the ideal (backward-Euler) PD does. PDF, with Tf = 0.1/wc, is robust on
// the continuous two. Oracle: the best smallest margin over a (K, Td) grid of
// loops stable by companion-matrix closed-loop roots.
func TestPidtuneFilteredPDUnattainable(t *testing.T) {
	if best := pidtuneFamilyBestMargin(t, Poly{1}, Poly{1, -1}, 0, pidtuneOracleP); !(best > 58.5) {
		t.Fatalf("oracle: P on 1/(s−1) reaches %.2f°, want ≈ 90°", best)
	}
	if best := pidtuneFamilyBestMargin(t, Poly{1}, Poly{1, 1, 0}, 0, pidtuneOracleFilteredPD); !(best > 58.5) {
		t.Fatalf("oracle: PD on 1/(s(s+1)) reaches %.2f°, want > 58.5°", best)
	}
	for _, c := range []struct {
		num, den Poly
		dt       float64
		typ      PidtuneType
		ideal    bool
	}{
		{Poly{1}, Poly{1, -1, 0}, 0, PidtunePD, true},
		{Poly{1}, Poly{1, -0.5, 1}, 0, PidtunePD, true},
		{Poly{1}, Poly{1, -2.1, 1.1}, 0.1, PidtunePD, false},
		{Poly{2.152560471847437}, Poly{1, -2.024624265088923, 1.0247399380515019}, 0.01, PidtunePD, true},
		{Poly{1}, Poly{1, 4, -5}, 0, PidtuneP, false},
	} {
		name := fmt.Sprintf("%s %v/%v dt=%g", c.typ, c.num, c.den, c.dt)
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		fams := []pidtuneOracleFamily{pidtuneOracleP}
		if c.typ == PidtunePD {
			fams = []pidtuneOracleFamily{pidtuneOracleFilteredPD, pidtuneOracleIdealPD}
		}
		for _, f := range fams {
			best := pidtuneFamilyBestMargin(t, c.num, c.den, c.dt, f)
			if want := f == pidtuneOracleIdealPD && c.ideal; best > 58.5 != want {
				t.Fatalf("%s: oracle family %d reaches %.2f°", name, f, best)
			}
		}
		pid, err := Pidtune(p, c.typ, 0, nil)
		if !c.ideal {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: got %+v, %v; want ErrPIDTuningTargetUnattainable", name, pid, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", name, err)
		} else {
			if pid.Tf != 0 {
				t.Errorf("%s: %+v, want ideal PD", name, pid)
			}
			pidtuneCheckRobust(t, name, pid, c.num, c.den, c.dt, pidTuningMaxSensitivity)
		}
		if c.dt > 0 {
			continue
		}
		pid, err = Pidtune(p, PidtunePDF, 0, nil)
		if err != nil {
			t.Errorf("%s PDF: %v", name, err)
			continue
		}
		pidtuneCheckRobust(t, name+" PDF", pid, c.num, c.den, c.dt, pidTuningMaxSensitivity)
	}
}

type pidtuneOracleFamily int

const (
	pidtuneOracleP pidtuneOracleFamily = iota
	pidtuneOracleFilteredPD
	pidtuneOracleIdealPD
)

// pidtuneFamilyBestMargin returns the largest smallest gain-crossover margin
// 180° − |∠L| over a log grid of K (either sign) and, for a PD, Td with
// Tf = Td/10 (forward-Euler derivative in z) or Tf = 0 (backward Euler in
// z), among loops whose closed-loop roots lie in the stability region.
func pidtuneFamilyBestMargin(t *testing.T, num, den Poly, dt float64, fam pidtuneOracleFamily) float64 {
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
	if fam != pidtuneOracleP {
		tds = tds[:0]
		for lt := -3.0; lt <= 3; lt += 0.05 {
			tds = append(tds, math.Pow(10, lt))
		}
	}
	best := math.Inf(-1)
	for _, td := range tds {
		nc, dc := Poly{1}, Poly{1}
		switch {
		case fam == pidtuneOracleFilteredPD && dt == 0:
			tf := td / 10
			nc, dc = Poly{tf + td, 1}, Poly{tf, 1}
		case fam == pidtuneOracleFilteredPD:
			tf := td / 10
			nc, dc = Poly{tf + td, dt - tf - td}, Poly{tf, dt - tf}
		case fam == pidtuneOracleIdealPD && dt == 0:
			nc = Poly{td, 1}
		case fam == pidtuneOracleIdealPD:
			nc, dc = Poly{dt + td, -td}, Poly{dt, 0}
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
