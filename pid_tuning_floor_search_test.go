package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// pidWindingDeg returns the unwrapped change of arg(1+L(jω)) in degrees over
// a dense log grid on [lo, hi].
func pidWindingDeg(L func(float64) complex128, lo, hi float64) float64 {
	const n = 400000
	step := math.Log(hi/lo) / n
	prev := 1 + L(lo)
	total := 0.
	for k := 1; k <= n; k++ {
		next := 1 + L(lo*math.Exp(float64(k)*step))
		total += cmplx.Phase(next / prev)
		prev = next
	}
	return total * 180 / math.Pi
}

func pidPoly(k float64, factors ...[]float64) []float64 {
	p := []float64{k}
	for _, f := range factors {
		p = polyMulTest(p, f)
	}
	return p
}

func pidPolyAt(p []float64, s complex128) complex128 {
	var v complex128
	for _, c := range p {
		v = v*s + complex(c, 0)
	}
	return v
}

func pidPlantDelayed(t *testing.T, num, den []float64, tau float64) *System {
	t.Helper()
	p := makePlant(t, num, den)
	if tau > 0 {
		if err := p.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

// An all-unstable band does not end the floor: the controller shape changes
// with the target band, so a later band can be stable. Stopping the floor
// on unstable bands returned no design for either plant. An
// independent dense sweep checks the margins, and the change of arg(1+L)
// over (0+, ∞) checks stability: with a stable plant and one integrator,
// 1+L turns from −90° to 0° when it does not encircle the origin.
func TestTunePIDFloorSearchesPastUnstableBands(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		family   PidtuneType
		wc       float64
	}{
		{"-0.48e^-0.215s/((s+10.6)(s²+0.0338s+0.716))", pidPoly(-.48), pidPoly(1, []float64{1, 10.6}, []float64{1, .0338, .716}), .215, PidtunePID, 0},
		{"0.106/(s²+0.27s+45.6)", pidPoly(.106), pidPoly(1, []float64{1, .27, 45.6}), 0, PidtunePIDF, 3.378},
	}
	for _, c := range cases {
		plant := pidPlantDelayed(t, c.num, c.den, c.tau)
		r, err := TunePID(context.Background(), plant, c.family, PIDTuningOptions{CrossoverFrequency: c.wc})
		if err != nil {
			t.Fatalf("%s %s: %v", c.name, c.family, err)
		}
		k := r.Controller
		L := func(w float64) complex128 {
			s := complex(0, w)
			g := pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
			return g * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		}
		margins := pidLoopMargins(L, 1e-4, 1e4)
		if len(margins) == 0 {
			t.Fatalf("%s: no crossover", c.name)
		}
		for _, pm := range margins {
			if pm < 57 {
				t.Errorf("%s: %+v independent margin %g below the 60° floor", c.name, k, pm)
			}
		}
		if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind-90) > 1 {
			t.Errorf("%s: %+v arg(1+L) changes by %g°, want 90° for a stable loop", c.name, k, wind)
		}
	}
}

// −1.09(1−s/1.81)e^{−0.437s}/((s+1.77)(s+4.47)(s+1.08)(s²+0.281s+0.0404))
// at wc = 7.927 has unwrapped phase near −675°: no same-sign PID, whose
// phase lies in [−90°, 90°], brings the loop above −180° there, so every
// floor band is skipped without a search instead of costing ~1 ms per
// Nyquist check across 1,400 evaluations.
func TestTunePIDFloorSkipsBandsBeyondUnwrappedPhase(t *testing.T) {
	num := pidPoly(-1.09, []float64{-1 / 1.81, 1})
	den := pidPoly(1, []float64{1, 1.77}, []float64{1, 4.47}, []float64{1, 1.08}, []float64{1, .281, .0404})
	plant := pidPlantDelayed(t, num, den, .437)
	const wc = 7.927
	_, ev, searches, err := pidTuneCounted(t, plant, PidtunePID, PIDTuningOptions{CrossoverFrequency: wc})
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("err %v", err)
	}
	if searches != 1 || ev.Evaluations > 200 {
		t.Fatalf("%d searches, %d evaluations; want only the ±3° search", searches, ev.Evaluations)
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	want := -(math.Atan(wc/1.77) + math.Atan(wc/4.47) + math.Atan(wc/1.08) + math.Atan2(.281*wc, .0404-wc*wc) + math.Atan(wc/1.81) + .437*wc) * 180 / math.Pi
	if got := pidTuningUnwrappedPhase(eval.at, -1, wc, .437); math.Abs(got-want) > 1e-6 {
		t.Fatalf("unwrapped phase %g, want %g", got, want)
	}
}

// With MaxEvaluations = 5 the ±3° PDF search fails, and each moved
// crossover gets its own five evaluations, as a cap per search: the first
// moved crossover meets PM 60°±3°.
func TestTunePIDMovedCrossoverGetsOwnBudget(t *testing.T) {
	num := pidPoly(-.52, []float64{1 / 1.24, 1})
	den := pidPoly(1, []float64{1, 10.9, 60.8}, []float64{1, .0371}, []float64{1, 1.62, 1.35})
	const tau = .108
	r, err := TunePID(context.Background(), pidPlantDelayed(t, num, den, tau), PidtunePDF, PIDTuningOptions{MaxEvaluations: 5})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	if k.Kp >= 0 || k.Kd >= 0 || k.Ki != 0 || k.Tf <= 0 || r.Evidence.Evaluations <= 5 || r.Evidence.Evaluations > 25 {
		t.Fatalf("%+v, %d evaluations", k, r.Evidence.Evaluations)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		g := pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*tau)
		return g * (complex(k.Kp, 0) + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
	}
	margins := pidLoopMargins(L, 1e-4, 1e4)
	if len(margins) == 0 {
		t.Fatal("no crossover")
	}
	for _, pm := range margins {
		if pm < 57 {
			t.Errorf("%+v independent margin %g below 57°", k, pm)
		}
	}
	if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind) > 1 {
		t.Errorf("%+v arg(1+L) changes by %g°, want 0 for a stable loop", k, wind)
	}
}
