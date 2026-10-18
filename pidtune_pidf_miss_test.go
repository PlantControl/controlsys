package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// pidtuneSignedMargins returns the phase margin 180°+∠L, wrapped to
// (−180°, 180°] as MATLAB margin reports it, at every gain crossover of L on
// [wMin, wMax] (log sweep, bisected), and the sampled peak of 1/|1+L|.
func pidtuneSignedMargins(L func(float64) complex128, wMin, wMax float64) (pms []float64, ms float64) {
	f := func(w float64) float64 { return cmplx.Abs(L(w)) - 1 }
	prevW, prev := wMin, f(wMin)
	for w := wMin * 1.001; w <= wMax; w *= 1.001 {
		ms = math.Max(ms, 1/cmplx.Abs(1+L(w)))
		cur := f(w)
		if prev*cur <= 0 && prev != cur {
			lo, hi := prevW, w
			for range 60 {
				mid := math.Sqrt(lo * hi)
				if f(lo)*f(mid) <= 0 {
					hi = mid
				} else {
					lo = mid
				}
			}
			pm := 180 + cmplx.Phase(L(lo))*180/math.Pi
			if pm > 180 {
				pm -= 360
			}
			pms = append(pms, pm)
		}
		prevW, prev = w, cur
	}
	return pms, ms
}

// OW4BCI: on an integrator with two real poles the Tf = Td/10 PIDF iteration
// cannot reach the 80° lead the 60° margin needs at the automatic crossover
// (a Td/10 filter lags more than the derivative leads), and returned
// Kp 0.102, Kd 1.67, Tf 1.64: PM 2.4° at its only crossover, |S| 27. The
// design at the default filter Tf = 0.1/wc reaches the margin, as TunePID
// does. Oracle: closed-loop roots from num/den and the signed margin at
// every crossover of the directly evaluated loop.
func TestPidtunePIDFMissedFilterPhase(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den Poly
	}{
		{"negative", Poly{-3.8562437633148288}, Poly{1, 2.38218255046348, 0.43053693845881924, 0}},
		{"positive", Poly{6.021719327000207}, Poly{1, 3.3266544288844027, 0.13253243642255236, 0}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		wc0, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		for _, wc := range []float64{0, wc0} {
			pid, err := Pidtune(p, PidtunePIDF, wc, nil)
			if err != nil {
				t.Fatalf("%s wc=%g: %v", c.name, wc, err)
			}
			if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, 0, 0); !ok || !stable {
				t.Errorf("%s wc=%g: %+v unstable", c.name, wc, pid)
			}
			if pid.Tf <= 0 || pid.Tf*wc0 > 1 {
				t.Errorf("%s wc=%g: Tf %g, want a filter faster than the crossover", c.name, wc, pid.Tf)
			}
			pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, 0, 0), wc0*1e-6, wc0*1e4)
			if len(pms) == 0 || ms > pidTuningMaxSensitivity {
				t.Errorf("%s wc=%g: %+v crossovers %v, |S| peak %.3g", c.name, wc, pid, pms, ms)
			}
			for _, pm := range pms {
				if math.Abs(pm-60) > 1 {
					t.Errorf("%s wc=%g: %+v phase margin %.4g° at a crossover, want 60°", c.name, wc, pid, pm)
				}
			}
		}
	}

	// −(…)/(…) at wc = 1 has phase 81° once signed: the 159° lead needed
	// exceeds any same-sign PIDF, and the clamped Td/10 design crosses at
	// wc with signed margin −76°. Its loop is pole-certified stable, so the
	// gate counts the classical 180° − |∠L| = 76° there (FPLHR5) and keeps
	// it: every crossover is ≥ 60° from −1 and |S| ≈ 1.2.
	num := Poly{-0.9872552701278292, -0.6864360868728736, -0.7928397763180903, -0.5894001648613465}
	den := Poly{1, 5.318581187837577, 9.734059318548901, 7.470525385519017, 2.061318122442111}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	pid, err := Pidtune(p, PidtunePIDF, 1, nil)
	if err != nil {
		t.Fatalf("unreachable lead: %v", err)
	}
	if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
		t.Errorf("unreachable lead: %+v unstable", pid)
	}
	pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, 0, 0), 1e-6, 1e4)
	if len(pms) == 0 || ms > pidTuningMaxSensitivity {
		t.Errorf("unreachable lead: %+v crossovers %v, |S| peak %.3g", pid, pms, ms)
	}
	for _, m := range pms {
		if math.Abs(m) < 58.5 {
			t.Errorf("unreachable lead: %+v phase margin %.3g° at a crossover", pid, m)
		}
	}
}
