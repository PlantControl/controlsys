package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A discrete plant with |G| < 1 everywhere and gain rising with frequency
// has no gain crossover and an infinite bandwidth, so the crossover is half
// the slowest pole's natural frequency |ln z|/Dt, capped a decade below
// Nyquist; it scales with 1/Dt. Pidtune's designs there are stable by the
// independent oracle.
func TestPlantCrossoverFreqDiscretePoleFallback(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		pole     complex128
	}{
		{Poly{0.1, -0.095}, Poly{1, -0.8}, 0.8},
		{Poly{0.1, -0.09}, Poly{1, 0.5}, -0.5},
	} {
		for _, dt := range []float64{0.1, 0.01} {
			p := pidtuneBackoffPlant(t, c.num, c.den, dt, 0)
			want := math.Min(0.5*cmplx.Abs(cmplx.Log(c.pole))/dt, 0.1*math.Pi/dt)
			wc, err := plantCrossoverFreq(p)
			if err != nil || math.Abs(wc-want) > 1e-9*want {
				t.Fatalf("%v/%v dt=%g: crossover %g, %v; want %g", c.num, c.den, dt, wc, err, want)
			}
			for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
				pid, err := Pidtune(p, f, 0, nil)
				if err != nil {
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
						t.Errorf("%v/%v dt=%g %s: %v", c.num, c.den, dt, f, err)
					}
					continue
				}
				if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, dt, 0); ok && !stable {
					t.Errorf("%v/%v dt=%g %s: %+v unstable", c.num, c.den, dt, f, pid)
				}
			}
		}
	}
}

// High-pass discrete plants (|G| < 1, gain rising to z = −1) take the
// crossover from their slowest pole near z = −1, about a decade below
// Nyquist, where no I design is stable; the first stable one lies more than
// three decades lower, so the back-off reaches six (TFE7I2 review). Oracle:
// closed-loop roots, signed margin at every crossover and |S| of the
// directly evaluated loop; Ki matches the design before TFE7I2.
func TestPidtunePoleFallbackBacksOffSixDecades(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		dt, ki   float64
	}{
		{Poly{-0.04189665061210819, 0.07677735137479592, -0.03510307300396654}, Poly{1, 0.9371514351204928, 0.13383773145948022}, 0.0757765515708101, -23.07},
		{Poly{-0.011302534772155698, 0.019672327226871213, -0.008550308672972825}, Poly{1, 1.3218336193087052, 0.4064581673333586}, 0.6967602637285988, -4.127},
		{Poly{0.02168291613348848, -0.040003288616014994, 0.018372574831365913}, Poly{1, 1.0961478669592766, 0.30038125589712455}, 0.06586027470121651, 56.0},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			t.Fatalf("%v/%v: pole fallback %v, %v", c.num, c.den, fromPole, err)
		}
		pid, err := Pidtune(p, PidtuneI, 0, nil)
		if err != nil {
			t.Errorf("%v/%v: %v", c.num, c.den, err)
			continue
		}
		if r := pid.Ki / c.ki; !(r > 1/1.5 && r < 1.5) {
			t.Errorf("%v/%v: Ki %g, want about %g", c.num, c.den, pid.Ki, c.ki)
		}
		if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
			t.Errorf("%v/%v: %+v unstable", c.num, c.den, pid)
		}
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), 1e-8, math.Pi/c.dt)
		if len(pms) == 0 || ms > 4.5 {
			t.Errorf("%v/%v: %+v crossovers %v, |S| peak %.3g", c.num, c.den, pid, pms, ms)
		}
		for _, m := range pms {
			if m < 58.5 {
				t.Errorf("%v/%v: %+v phase margin %.3g°", c.num, c.den, pid, m)
			}
		}
	}
}

// A crossover from the pole fallback is only a guess, so its first design is
// gated like a back-off step: before, PI on the first plant returned PM
// 0.65° and |S| 96, and on the second |S| 19, where the crossover before
// TFE7I2 gave |S| ≤ 1.3. Over a seeded population of pole-fallback plants
// every returned design is stable and keeps the margin at every crossover
// and |S| ≤ 4.5. Oracle: closed-loop roots from num/den, and |S| and the
// margin of the directly evaluated loop, counted as the distance of ∠L from
// −1 since the roots certify stability.
func TestPidtunePoleFallbackFirstDesignGated(t *testing.T) {
	robust := func(name string, pid *PID, num, den Poly, dt float64) {
		t.Helper()
		if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, 0); !ok || !stable {
			t.Errorf("%s: %+v unstable", name, pid)
			return
		}
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, dt, 0), 1e-9, math.Pi/dt)
		if len(pms) == 0 || ms > 4.5 {
			t.Errorf("%s: %+v crossovers %v, |S| peak %.3g", name, pid, pms, ms)
		}
		for _, m := range pms {
			if math.Abs(m) < 58.5 {
				t.Errorf("%s: %+v phase margin %.3g° at a crossover", name, pid, m)
			}
		}
	}
	for _, c := range []struct {
		num, den Poly
		dt       float64
	}{
		{Poly{0.019904639405814103, -0.05024414943963601, 0.042032260170646436, -0.011657063647966679}, Poly{1, 0.8409752628969186, 0.898983184352559, 0.3360282522140155}, 0.06476480350653667},
		{Poly{-0.5541090033279573, 0.4645622775451982}, Poly{1, -0.5121828860610971, 0.34735631228677943}, 0.05686198745051635},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			t.Fatalf("%v/%v: pole fallback %v, %v", c.num, c.den, fromPole, err)
		}
		name := fmt.Sprintf("PI %v/%v", c.num, c.den)
		pid, err := Pidtune(p, PidtunePI, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		robust(name, pid, c.num, c.den, c.dt)
	}

	fams := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	rng := rand.New(rand.NewPCG(11, 13))
	plants, designs := 0, 0
	for trial := 0; plants < 100 && trial < 100000; trial++ {
		n := 1 + rng.IntN(4)
		dt := 0.01 + 0.5*rng.Float64()
		den := Poly{1}
		for range n {
			den = den.Mul(Poly{1, 0.95 - 1.9*rng.Float64()})
		}
		num := make(Poly, rng.IntN(n)+1)
		for i := range num {
			num[i] = 2*rng.Float64() - 1
		}
		p := pidtuneBackoffPlant(t, num, den, dt, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			continue
		}
		plants++
		for _, f := range fams {
			name := fmt.Sprintf("%s %v/%v dt=%g", f, num, den, dt)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%s: %v", name, err)
				}
				continue
			}
			designs++
			robust(name, pid, num, den, dt)
		}
	}
	if plants < 100 || 2*designs < 7*plants {
		t.Fatalf("%d pole-fallback plants, %d designs", plants, designs)
	}
}

// Continuous high-pass plants k·sᵐ/den(s) with |G| < 1 take the crossover
// from their slowest pole. Lowered six decades, the P, PDF and I designs
// reached |K| ~ 1e6–1e18, where Feedback's E12 = 1 + D2·E21·D1 cancelled to
// rounding noise and an RHP pair passed the pole test, and where the
// realization's split double zero at s = 0 hid the integrator it cancels
// (TFE7I2 review); an integral term on a plant zero at the integrator is
// never asymptotically stable. Oracle: closed-loop roots from num/den;
// over a seeded population every returned design is stable.
func TestPidtunePoleFallbackContinuousStable(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtuneP, Poly{-0.24371488599761515, 0, 0, 0}, Poly{1, 5.3999877579765405, 7.271135289167493, 2.7160209747907498}},
		{PidtunePDF, Poly{0.4680649732178014, 0, 0, 0}, Poly{1, 4.609720722243954, 4.49026010353689, 1.2399444016467684}},
		{PidtuneI, Poly{0.8638233193779494, 0, 0}, Poly{1, 3.6152996784214597, 3.1165143190980205}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			t.Fatalf("%v/%v: pole fallback %v, %v", c.num, c.den, fromPole, err)
		}
		pid, err := Pidtune(p, c.f, 0, nil)
		if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			stable, _ := pidtuneOracleStable(t, pid, c.num, c.den, 0, 0)
			t.Errorf("%s %v/%v: %+v (stable %v), %v; want ErrPIDTuningTargetUnattainable", c.f, c.num, c.den, pid, stable, err)
		}
	}
	for _, f := range []PidtuneType{PidtuneI, PidtunePI, PidtunePIDF} {
		p := pidtuneBackoffPlant(t, Poly{0.5, 0}, Poly{1, 2}, 0, 0)
		if pid, err := Pidtune(p, f, 1, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s on 0.5s/(s+2) at wc=1: %+v, %v; want ErrPIDTuningTargetUnattainable", f, pid, err)
		}
	}

	fams := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	rng := rand.New(rand.NewPCG(5, 9))
	plants, designs := 0, 0
	for trial := 0; plants < 60 && trial < 10000; trial++ {
		n := 1 + rng.IntN(3)
		den := Poly{1}
		for range n {
			den = den.Mul(Poly{1, 0.05 + 4*rng.Float64()})
		}
		num := Poly{2*rng.Float64() - 1}
		for range 1 + rng.IntN(n) {
			num = num.Mul(Poly{1, 0})
		}
		p := pidtuneBackoffPlant(t, num, den, 0, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			continue
		}
		plants++
		for _, f := range fams {
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%s %v/%v: %v", f, num, den, err)
				}
				continue
			}
			designs++
			if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
				t.Errorf("%s %v/%v: %+v unstable", f, num, den, pid)
			}
		}
	}
	if plants < 60 || 2*designs < plants {
		t.Fatalf("%d pole-fallback plants, %d designs", plants, designs)
	}
}

// On these high-pass plants the loop phase at a crossover is positive, so
// the signed margin 180° + ∠L is negative however far L stays from −1. The
// back-off gate read it signed: PD, PID and PIDF on T0/T2 errored at every
// step (FPLHR5), and the PDF least-lead design, crossing at wc itself with
// signed margin ≈ −120°…−160°, was rejected there for every family, so PDF
// errored with "needs phase lag". On a pole-certified loop the gate counts
// 180° − |∠L|. Oracle: closed-loop roots from num/den, and the classical
// margin at every crossover and |S| of the directly evaluated loop.
func TestPidtuneGateClassicalMargin(t *testing.T) {
	t0 := [2]Poly{{-0.04189665061210819, 0.07677735137479592, -0.03510307300396654}, {1, 0.9371514351204928, 0.13383773145948022}}
	t2 := [2]Poly{{0.02168291613348848, -0.040003288616014994, 0.018372574831365913}, {1, 1.0961478669592766, 0.30038125589712455}}
	all := []PidtuneType{PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	for _, c := range []struct {
		num, den Poly
		dt       float64
		fams     []PidtuneType
	}{
		{t0[0], t0[1], 0.0757765515708101, all},
		{t2[0], t2[1], 0.06586027470121651, all},
		{Poly{0.9147921246960318, -0.39502144372826553}, Poly{1, -0.5431690613836653, -0.1892877676898109}, 0.31622446378773106, []PidtuneType{PidtunePDF}},
		{Poly{0.799289620228443, 0}, Poly{1, 1.6124129207235272, 0.6499446539813588}, 0, []PidtuneType{PidtunePDF}},
		{Poly{0.3109408305381509, 0}, Poly{1, 4.675160826318902, 5.1106958875365285}, 0, []PidtuneType{PidtunePDF}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		wMax := 1e4
		if c.dt > 0 {
			wMax = math.Pi / c.dt
		}
		for _, f := range c.fams {
			name := fmt.Sprintf("%s %v/%v dt=%g", f, c.num, c.den, c.dt)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
				t.Errorf("%s: %+v unstable", name, pid)
				continue
			}
			pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), 1e-9, wMax)
			if len(pms) == 0 || ms > pidTuningMaxSensitivity {
				t.Errorf("%s: %+v crossovers %v, |S| peak %.3g", name, pid, pms, ms)
			}
			for _, m := range pms {
				if math.Abs(m) < 58.5 {
					t.Errorf("%s: %+v phase margin %.3g° at a crossover", name, pid, m)
				}
			}
		}
	}
}

// PD on k·s/(s+a) cannot lag at the pole-fallback crossover; its clamped
// design meets |L(wc)| ≈ 0.49 and crosses over well above, 137° from −1.
// The gate read wc as a crossover with margin −92.6°, as at every back-off
// step, so Pidtune errored (TFE7I2 review). Oracle: closed-loop root, and
// the signed margin at every crossover and |S| of the directly evaluated
// loop; gains match the design before TFE7I2.
func TestPidtunePoleFallbackPDHighPass(t *testing.T) {
	for _, c := range []struct {
		k, a       float64
		kp, kd, tf float64
	}{
		{0.10902430967794469, 0.23104521558459049, 1.025063312, 177.3174913, 17.29819897},
		{0.5499305783505386, 3.6597358499148074, 0.2032, 2.219, 1.092},
	} {
		num, den := Poly{c.k, 0}, Poly{1, c.a}
		p := pidtuneBackoffPlant(t, num, den, 0, 0)
		wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
		if err != nil || !fromPole {
			t.Fatalf("%v/%v: pole fallback %v, %v", num, den, fromPole, err)
		}
		pid, err := Pidtune(p, PidtunePD, 0, nil)
		if err != nil {
			t.Errorf("%v/%v: %v", num, den, err)
			continue
		}
		for _, g := range [][2]float64{{pid.Kp, c.kp}, {pid.Kd, c.kd}, {pid.Tf, c.tf}} {
			if math.Abs(g[0]-g[1]) > 1e-3*math.Abs(g[1]) {
				t.Errorf("%v/%v: %+v, want Kp %g Kd %g Tf %g", num, den, pid, c.kp, c.kd, c.tf)
				break
			}
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
			t.Errorf("%v/%v: %+v unstable", num, den, pid)
		}
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, 0, 0), wc*1e-6, wc*1e6)
		if len(pms) == 0 || ms > 1.1 {
			t.Errorf("%v/%v: %+v crossovers %v, |S| peak %.3g", num, den, pid, pms, ms)
		}
		for _, m := range pms {
			if math.Abs(m) < 120 {
				t.Errorf("%v/%v: %+v margin %.3g° at a crossover", num, den, pid, m)
			}
		}
	}
}

// pidtuneCheckRobust checks pid on num/den by the independent oracle:
// closed-loop roots, a crossover with every signed margin at least 58.5°,
// and |S| at most msMax on the directly evaluated loop.
func pidtuneCheckRobust(t *testing.T, name string, pid *PID, num, den Poly, dt, msMax float64) {
	t.Helper()
	if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, 0); !ok || !stable {
		t.Errorf("%s: %+v unstable", name, pid)
	}
	hi := 1e6
	if dt > 0 {
		hi = math.Pi / dt
	}
	pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, dt, 0), 1e-6, hi)
	if len(pms) == 0 || ms > msMax {
		t.Errorf("%s: %+v crossovers %v, |S| %.3g", name, pid, pms, ms)
	}
	for _, m := range pms {
		if math.Abs(m) < 58.5 {
			t.Errorf("%s: %+v margin %.3g°", name, pid, m)
		}
	}
}

// A plant zero strictly inside the stability region, however close to the
// integrator, leaves the closed-loop pole an integral term pairs with it
// strictly stable, and MATLAB pidtune designs there. The reject for a zero
// at the integrator took 1e-6·wc as its radius and caught zeros at −1e-3
// (wc 8669), −1e-7, −1e-5, −1e-8 and z = 1−1e-8 (TFE7I2 review).
func TestPidtuneZeroNearIntegratorDesigns(t *testing.T) {
	all := []PidtuneType{PidtuneI, PidtunePI, PidtunePID, PidtunePIDF}
	for _, c := range []struct {
		num, den Poly
		dt, wc   float64
		fams     []PidtuneType
	}{
		{Poly{0.69687559053705495}.Mul(Poly{1, 1e-3}), Poly{1, 4.403080400780377, 1.6256391362894465}, 0, 0, []PidtuneType{PidtunePI}},
		{Poly{0.11030945862927788}.Mul(Poly{1, 1e-7}), Poly{1, 1.3644905835382877}, 0, 0, []PidtuneType{PidtuneI}},
		{Poly{-0.53434948917407166}.Mul(Poly{1, 1e-5}), Poly{1, 2.5122667807172974, 0.82341538655475466, 0.069695621091596771}, 0, 0, []PidtuneType{PidtunePI}},
		{Poly{2.0875}.Mul(Poly{1, 1e-8}).Mul(Poly{1, 3}), Poly{1, 1}.Mul(Poly{1, 2}).Mul(Poly{1, 0.5}), 0, 0.7335, all},
		{Poly{0.5}.Mul(Poly{1, -(1 - 1e-8)}).Mul(Poly{1, -0.3}), Poly{1, -0.6}.Mul(Poly{1, -0.4}).Mul(Poly{1, 0.2}), 0.1, 0, all},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		for _, f := range c.fams {
			name := fmt.Sprintf("%s %v/%v dt=%g", f, c.num, c.den, c.dt)
			pid, err := Pidtune(p, f, c.wc, nil)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			pidtuneCheckRobust(t, name, pid, c.num, c.den, c.dt, 4.5)
		}
	}
}

// A PD design clamped short of |L(wc)| = 1 crossed nowhere; the gate, which
// drops wc then, checked no crossover and accepted a loop with |L| ≤ 0.18
// everywhere where the design at the automatic crossover keeps a 60° margin
// (TFE7I2 review).
func TestPidtunePDBackoffNeedsCrossover(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		dt       float64
	}{
		{Poly{-0.94546289712921561}, Poly{1, -1.4539218418093962, 0.51962244899907128}, 0.3697375861344023},
		{Poly{0.79755174815541685}, Poly{1, -1.0025664143562638, -0.069075795306954832, 0.16754669270864272, 0.012844162343441125}, 0.2798273375686335},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		name := fmt.Sprintf("%v/%v dt=%g", c.num, c.den, c.dt)
		pid, err := Pidtune(p, PidtunePD, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, c.dt, 3)
	}
}

// A plant with an exact zero at the integrator (s = 0, z = 1) has DC gain
// exactly 0, so Bandwidth errored before the pole fallback and every family
// returned ErrInvalidArgument (SJ5QTW). The crossover is half the slowest
// pole's natural frequency, capped a decade below Nyquist; integral
// families return ErrPIDTuningTargetUnattainable, also when no crossover
// can be inferred (poles only at z = 0). A balanced or transformed
// realization rounds the DC gain to ~1e-17, whose -3 dB "bandwidth" at
// rounding level gave wc ~1e-17 or ~1e15; it takes the same fallback.
// Oracle: crossover from den's roots; closed-loop roots and |S| of the
// directly evaluated loop.
func TestPidtuneExactZeroAtIntegrator(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		dt       float64
		designs  []PidtuneType
	}{
		{Poly{-0.0563, 0.3311, 0}, Poly{1, 7.063, 15.54, 12.54}, 0, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{0.5, 0}, Poly{1, 2}, 0, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{0.3, -0.3}, Poly{1, -0.5}, 0.1, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{0.3, -0.3}, Poly{1, -1.2, 0.35}, 0.1, nil},
	} {
		base := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		roots, err := c.den.Roots()
		if err != nil {
			t.Fatal(err)
		}
		want := math.Inf(1)
		for _, r := range roots {
			w := cmplx.Abs(r)
			if c.dt > 0 {
				w = cmplx.Abs(cmplx.Log(r)) / c.dt
			}
			want = math.Min(want, 0.5*w)
		}
		if c.dt > 0 {
			want = math.Min(want, 0.1*math.Pi/c.dt)
		}
		n, _, _ := base.Dims()
		T := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				T.Set(i, j, 0.3*float64((7*i+3*j)%5)-0.4)
			}
			T.Set(i, i, T.At(i, i)+2)
		}
		transformed, err := SS2SS(base, T)
		if err != nil {
			t.Fatal(err)
		}
		balanced, err := Balreal(base)
		if err != nil {
			t.Fatal(err)
		}
		for k, p := range []*System{base, transformed, balanced.Sys} {
			wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
			if err != nil || !fromPole || math.Abs(wc-want) > 1e-9*want {
				t.Fatalf("%v/%v realization %d: crossover %g (pole %v), %v; want %g", c.num, c.den, k, wc, fromPole, err, want)
			}
			for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
				name := fmt.Sprintf("%s %v/%v dt=%g realization %d", f, c.num, c.den, c.dt, k)
				pid, err := Pidtune(p, f, 0, nil)
				if err != nil {
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) || slices.Contains(c.designs, f) {
						t.Errorf("%s: %v", name, err)
					}
					continue
				}
				if _, hasI, _, _, _ := pidTuningTerms(f); hasI {
					t.Errorf("%s: %+v; want ErrPIDTuningTargetUnattainable", name, pid)
					continue
				}
				if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
					t.Errorf("%s: %+v unstable", name, pid)
				}
				hi := wc * 1e6
				if c.dt > 0 {
					hi = math.Pi / c.dt
				}
				if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), wc*1e-6, hi); ms > 4.05 {
					t.Errorf("%s: %+v |S| peak %.3g", name, pid, ms)
				}
			}
		}
	}

	p := pidtuneBackoffPlant(t, Poly{0.3, -0.3}, Poly{1, 0}, 0.1, 0)
	for _, f := range []PidtuneType{PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
		if pid, err := Pidtune(p, f, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s on 0.3(z-1)/z: %+v, %v; want ErrPIDTuningTargetUnattainable", f, pid, err)
		}
	}
}

// A zero near, not at, the integrator attenuates G(0) so far that its
// -3 dB bandwidth lies on the c/s asymptote, where gains of 1e9..1e18
// placed a closed-loop pole at the zero (TA526O). The crossover is half the
// slowest pole's natural frequency instead, for every realization; integral
// families may fail when the closed-loop pole left near the zero is within
// the stability tolerance. Bandwidths within ten times the fastest root,
// a resonant low-pass included, are kept. Oracle: crossover from den's
// roots; closed-loop roots, gains against 1/|G(j·wc)| and |S| of the
// directly evaluated loop.
func TestPidtuneZeroNearIntegrator(t *testing.T) {
	all := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePID, PidtunePIDF}
	for _, c := range []struct {
		num, den Poly
		designs  []PidtuneType
	}{
		{Poly{0.8296, 0.8296e-10}, Poly{1, 0.999, 0.1257}, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{-0.1376, -0.1376e-10}, Poly{1, 4.684, 1.41}, []PidtuneType{PidtuneP, PidtunePD}},
		{Poly{0.8296, 0.8296e-6}, Poly{1, 0.999, 0.1257}, all},
		{Poly{0.8296, 0.8296e-3}, Poly{1, 0.999, 0.1257}, all},
		{Poly{0.8296, 0.8296e-2}, Poly{1, 0.999, 0.1257}, all},
	} {
		base := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		roots, err := c.den.Roots()
		if err != nil {
			t.Fatal(err)
		}
		want := math.Inf(1)
		for _, r := range roots {
			want = math.Min(want, 0.5*cmplx.Abs(r))
		}
		bound := 1e3 * cmplx.Abs(c.den.Eval(complex(0, want))/c.num.Eval(complex(0, want)))
		n, _, _ := base.Dims()
		T := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				T.Set(i, j, 0.3*float64((7*i+3*j)%5)-0.4)
			}
			T.Set(i, i, T.At(i, i)+2)
		}
		transformed, err := SS2SS(base, T)
		if err != nil {
			t.Fatal(err)
		}
		balanced, err := Balreal(base)
		if err != nil {
			t.Fatal(err)
		}
		for k, p := range []*System{base, transformed, balanced.Sys} {
			wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
			if err != nil || !fromPole || math.Abs(wc-want) > 1e-9*want {
				t.Fatalf("%v/%v realization %d: crossover %g (pole %v), %v; want %g", c.num, c.den, k, wc, fromPole, err, want)
			}
			for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
				name := fmt.Sprintf("%s %v/%v realization %d", f, c.num, c.den, k)
				pid, err := Pidtune(p, f, 0, nil)
				if err != nil {
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) || slices.Contains(c.designs, f) {
						t.Errorf("%s: %v", name, err)
					}
					continue
				}
				if !(math.Abs(pid.Kp) <= bound && math.Abs(pid.Ki) <= bound*wc && math.Abs(pid.Kd) <= bound/wc) {
					t.Errorf("%s: %+v; gains above %g", name, pid, bound)
				}
				if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, 0, 0); !ok || !stable {
					t.Errorf("%s: %+v unstable", name, pid)
				}
				if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, 0, 0), wc*1e-6, wc*1e6); ms > 4.05 {
					t.Errorf("%s: %+v |S| peak %.3g", name, pid, ms)
				}
			}
		}
	}

	for _, c := range []struct{ num, den Poly }{
		{Poly{0.8296, 0.08296}, Poly{1, 0.999, 0.1257}},
		{Poly{0.8296, 0.024888}, Poly{1, 0.999, 0.1257}},
		{Poly{0.001}, Poly{1, 0.01, 1}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		bw, err := Bandwidth(p, -3)
		if err != nil {
			t.Fatal(err)
		}
		if wc, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || fromPole || wc != bw {
			t.Errorf("%v/%v: crossover %g (pole %v), %v; want bandwidth %g", c.num, c.den, wc, fromPole, err, bw)
		}
	}
}
