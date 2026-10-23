package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// IBWZGV: the ungated retry at an automatic CrossGain/Rolloff wc returned
// designs the gate rejected: PID with an ideal derivative on a biproper
// plant (4 crossovers, worst margin 6.9°, |S| 14.5), a PD with no 0 dB
// crossing (|S| 35) and a PI on an unstable plant with PM 1.07° (|S| 54).
// Also at the bandwidth of unstable plants: I with PM 0.001° (|S| 1249), P
// with 1.4° (|S| 40); and PID with an ideal derivative, −10.9° and |S| 5.3.
// The retry now needs |S| ≤ 4 and a crossover, so these return
// ErrPIDTuningTargetUnattainable or a design robust per the oracle:
// closed-loop roots, margins and |S| of the directly evaluated loop.
func TestPidtuneUngatedRetryRobust(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtuneI, Poly{0.27517413517840106, 3.6669924390083567e-05}, Poly{1, 6.391643798250351, -30.351456187775952}},
		{PidtuneP, Poly{3.24364602965127, 0.0021262777284359194}, Poly{1, 13.719866434906919, 22.960516396640905, -141.22246804852216}},
		{PidtunePID, Poly{-1.8326830463596728, -4.165603628320087e-05}, Poly{1, 0.7334709862442443, 0.25518924748370436, 0.03900616993539339}},
		{PidtunePID, Poly{0.05153281511550831, 0.13713526593255349, 0.07056564212558677, 1.7258480953641303e-05}, Poly{1, 1.6146986643307246, 1.9381866559399348, 2.7956778334415695}},
		{PidtunePD, Poly{0.6095365102040942, -0.3200683340297962, -0.1152823784131911}, Poly{1, 5.393739050975377, 9.53892918533701, 5.541668306872566}},
		{PidtunePI, Poly{0.0009165950240917041, 0.00037102544443059034, 0.032239853096689565}, Poly{1, 0.40478666660694174, -13.978966685379245}},
	} {
		name := fmt.Sprintf("%s %v/%v", c.f, c.num, c.den)
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		pid, err := Pidtune(p, c.f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, pidTuningMaxSensitivity)
	}
}

// X7DDPS: when back-off failed on a stable plant, the retry at the
// automatic wc gated only on |S| ≤ 4 and a crossover returned designs short
// of the margin at another crossover, mostly on biproper plants: PID with an
// ideal derivative at −23.7°, −24.2°, −19.6°, −24.5°, its integral-only form
// at 36.8°, PD at −26.3°, −53.8° and −48.9°, and on the last two plants
// (92e7f8d) PID at 17.4° (|S| 3.54) and 19.8° (|S| 3.05). The full gate
// never passed the retry on stable plants, which now return
// ErrPIDTuningTargetUnattainable or a design robust per the oracle.
func TestPidtuneNoFragileRetry(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtunePID, Poly{1.2654745408440329, 1.8998645549384214, 1.938148022288258e-06}, Poly{1, 1.2955994761823357, 1.078580429900587}},
		{PidtunePID, Poly{1.4754169078841284, 1.9475037635085144, 1.86207705614513, 3.085670387747623e-07}, Poly{1, 1.9884706490264086, 0.7760771899540704, 0.09700570403707065}},
		{PidtunePID, Poly{1.6393781502237057, 1.5729744617276629, 1.4422833102656818, 5.005710706969215e-07}, Poly{1, 0.9945214998051222, 0.5848509925742207, 0.15398262905361126}},
		{PidtunePID, Poly{1.7018722825622308, 0.7912405885497165, 1.1656162153765089e-06}, Poly{1, 0.31138563121400886, 0.008977261787331159}},
		{PidtunePID, Poly{0.7441606413430202, -1.4348718019413627, -8.909880207643177e-07}, Poly{1, 0.9530571064061941, 1.7469511168498217}},
		{PidtunePD, Poly{1.869847039800924, 1.9460197960801928, 0.33618542251074346, 0.8662899479933694}, Poly{1, 0.37809636237642263, 1.2251180342422012, 0.10118488017248772}},
		{PidtunePD, Poly{1.4070934529826, 0.01569665533397635, 1.7315628492954016}, Poly{1, 0.13987154850957495, 1.498413308783041}},
		{PidtunePD, Poly{-1.3812490475523809, 0.3049081517775649, -0.679036258551609}, Poly{1, 0.8507156397529578, 0.6749494711240132}},
		{PidtunePID, Poly{0.44274738180524537, 0.8774349592894404, 5.105628467814613e-09}, Poly{1, 0.44104297875501014, 0.6743286112634963}},
		{PidtunePID, Poly{0.298926677490796, 1.2071203877758425, 1.2110010334103394, 1.5931056551339456e-09}, Poly{1, 1.3266358687227324, 0.845646116236277, 0.4893392826128204}},
	} {
		name := fmt.Sprintf("%s %v/%v", c.f, c.num, c.den)
		roots, err := c.den.Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range roots {
			if real(r) >= 0 {
				t.Fatalf("%s: plant pole %v not stable", name, r)
			}
		}
		pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, 0, 0), c.f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, pidTuningMaxSensitivity)
	}
}

// PS6S32: on zero-DC continuous plants the legacy PD rerun, clamped short of
// the margin at its pole-derived wc, passed on |S| alone and returned
// margins of 44.4° (|S| 1.9), 56.8° and 58.4° at its only crossover. Higher
// crossovers keep the margin; lower ones, back-off, lose it. The legacy
// rerun now also keeps pm − 1.5° at every crossover, so these return
// ErrPIDTuningTargetUnattainable or a design robust per the oracle.
func TestPidtuneZeroDCLegacyPDGated(t *testing.T) {
	for _, c := range []struct{ num, den Poly }{
		{Poly{0.4060949555907569, 0, 0}, Poly{1, 1.5349963623779264, 0.5521256684141752}},
		{Poly{0.34666872878320376, 0, 0}, Poly{1, 3.9256001901459285, 3.843722834621563}},
		{Poly{-0.7431680988546034, 0, 0}, Poly{1, 5.3495557727549565, 6.22597598141245, 2.052553524633182}},
	} {
		name := fmt.Sprintf("%v/%v", c.num, c.den)
		pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, 0, 0), PidtunePD, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, pidTuningMaxSensitivity)
	}
}

// FI3ZIB: rounding moves a simple integrator of (z−1)·∏(z−pᵢ), pᵢ near 1,
// beyond pidtuneClusterRadius(2, 1) while the true DC gain stays infinite
// (DCGain resolves the rounded pencil finite, like evalfr); taking it
// as the slowest pole put the crossover ~600x below half the slowest
// generated pole (1.46e-4 for the repro, its integrator 2.9e-6 off z = 1).
// At the corrected crossover the gate's double solve was singular over the
// band, so every back-off step failed on NaN margins and the retry returned
// the first design: P with PM 33.8° on the second plant. Oracles: for a
// nearest pole beyond that radius with every other pole ten times farther,
// half the slowest generated pole's |ln p|/Dt, capped at 0.1π/Dt, to 10%
// when the others lie 100 times farther and 2x below that (rounding also
// scatters the slow poles); every design stable by the exact Schur–Cohn
// test and, on every other plant for time, robust by extended-precision
// Schur–Cohn, margins ≥ 58.5° and |S| from the coefficients. Conjugate
// pairs or comparable distances are scattered clusters this oracle does
// not describe.
func TestPidtuneIsolatedDisplacedIntegrator(t *testing.T) {
	t.Parallel()
	num := Poly{1.353366909367569e-15}
	den := Poly{1, -5.60431550709433, 13.063429504287544, -16.210023582131356, 11.292650985912374, -4.187371704674265, 0.6456303037000314}
	p := pidtuneBackoffPlant(t, num, den, 0.01, 0)
	if wc, err := plantCrossoverFreq(p); err != nil || math.Abs(wc-0.0895) > 0.02*0.0895 {
		t.Errorf("repro crossover %g, %v; want 0.0895", wc, err)
	}
	num = Poly{1.1974308853542987e-16}
	den = Poly{1, -6.184746550403649, 16.242438259051475, -23.437999081676736, 20.022569140290482, -10.09289718715969, 2.766371287397997, -0.31573586749987936}
	p = pidtuneBackoffPlant(t, num, den, 0.01, 0)
	for _, f := range []PidtuneType{PidtuneP, PidtunePI} {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			t.Errorf("gate repro %s: %v", f, err)
			continue
		}
		pidDCSignCheckRobust(t, fmt.Sprintf("gate repro %s", f), pid, num, den, 0.01, 58.5)
	}

	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(5, 8))
	kept, close := 0, 0
	for range 300 {
		dt := 0.01
		num, den := Poly{math.Pow(10, -16+4*rng.Float64())}, Poly{1, -1}
		slow := math.Inf(1)
		for range 5 + rng.IntN(4) {
			a := math.Pow(10, -1+3*rng.Float64())
			slow = math.Min(slow, a)
			den = den.Mul(Poly{1, -math.Exp(-a * dt)})
		}
		for range rng.IntN(3) {
			num = num.Mul(Poly{1, -(rng.Float64()*1.8 - 0.9)})
		}
		p := pidtuneBackoffPlant(t, num, den, dt, 0)
		am, err := AllMargin(p)
		if err != nil || len(am.GainCrossFreqs) > 0 {
			continue
		}
		poles, err := p.Poles()
		if err != nil {
			t.Fatal(err)
		}
		near, second := math.Inf(1), math.Inf(1)
		for _, r := range poles {
			switch d := cmplx.Abs(r - 1); {
			case d < near:
				near, second = d, near
			case d < second:
				second = d
			}
		}
		if near <= pidtuneClusterRadius(2, 1) || second < 10*near {
			continue
		}
		kept++
		name := fmt.Sprintf("%v/%v dt=%g", num, den, dt)
		want := math.Min(0.5*slow, 0.1*math.Pi/dt)
		wc, err := plantCrossoverFreq(p)
		if second >= 100*near {
			close++
			if err != nil || math.Abs(wc-want) > 0.1*want {
				t.Errorf("%s: crossover %g, %v; want %g", name, wc, err, want)
			}
		} else if err != nil || wc < want/2 || wc > 2*want {
			t.Errorf("%s: crossover %g, %v; want %g within 2x", name, wc, err, want)
		}
		for _, f := range pidtuneAllTypes {
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%s %s: %v", name, f, err)
				}
				continue
			}
			if kept%2 == 1 {
				pidDCSignCheckRobust(t, fmt.Sprintf("%s %v", name, f), pid, num, den, dt, 58.5)
			} else if !pidtuneExactSchur(pid, num, den, dt) {
				t.Errorf("%s %s: %+v unstable", name, f, pid)
			}
		}
	}
	if kept < 20 || close < 10 {
		t.Fatalf("sweep kept %d plants, %d with poles 100x farther", kept, close)
	}
}

// Beside a pole cluster at z = 1 the gate's solve snapped ω = 1.9e-5,
// where |G| is 0.32 and no pole lies, to +Inf without an error; the scan
// paired that spike with a crossover of NaN margin and failed the P design
// at the plant's own crossover (wc 6.1e-6, 9.9e-7) on every back-off step.
// Oracle: extended-precision Schur–Cohn, margins and |S| from the
// coefficients (one crossover, PM 90°, |S| 1).
func TestPidtuneGateSnappedSample(t *testing.T) {
	skipSlowUnderRace(t)
	const dt = 0.01
	for _, c := range []struct{ num, den Poly }{
		{Poly{6.110773630519141e-14, 3.4198170787824316e-15}, Poly{1, -5.71016988999306, 13.887907770588715, -18.645468986193702, 14.92072833819654, -7.1153754170879235, 1.8719233030765783, -0.20954511858714697}},
		{Poly{3.875422988056212e-15}, Poly{1, -4.667597493849348, 8.695671657313316, -8.081072724970674, 3.7455208432476796, -0.692522281740973}},
	} {
		name := fmt.Sprintf("%v/%v", c.num, c.den)
		pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, dt, 0), PidtuneP, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		pidDCSignCheckRobust(t, name, pid, c.num, c.den, dt, 58.5)
	}
}

// On plants that are not stable the retry at the automatic wc was gated on
// |S| and a crossover only, returning P with PM 18.1° (|S| 3.2) and PI with
// 15.3° on an unstable plant, and PI with 18.6° (|S| 3.7) on an integrating
// discrete one. It now takes the full gate, so these return
// ErrPIDTuningTargetUnattainable or a design robust per the oracle
// (extended precision for the discrete plant).
func TestPidtuneUnstableRetryKeepsMargin(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
		dt       float64
	}{
		{PidtuneP, Poly{2.645988078729179}, Poly{1, 0.6994623329224331, -1.129423843040615}, 0},
		{PidtunePI, Poly{2.645988078729179}, Poly{1, 0.6994623329224331, -1.129423843040615}, 0},
		{PidtunePI, Poly{1.9331897426073496e-15, -1.0091665836801221e-15}, Poly{1, -6.853180411596403, 20.125372937931182, -32.82904851475664, 32.126073810329295, -18.859953544489116, 6.150097496943598, -0.8593617743619164}, 0.01},
	} {
		name := fmt.Sprintf("%s %v/%v dt=%g", c.f, c.num, c.den, c.dt)
		pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0), c.f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		if c.dt > 0 {
			pidDCSignCheckRobust(t, name, pid, c.num, c.den, c.dt, 58.5)
		} else {
			pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, pidTuningMaxSensitivity)
		}
	}
}

// The non-stable retry's full gate floored the margin at pm − 0.5°, so a P
// at the automatic wc 3.9e-5 of this integrating discrete plant, PM 58.9°
// at its one crossover, |S| 1.07, became unattainable; back-off loses phase
// (56° at 0.9·wc). The retry keeps pm − 1.5°. Oracle: extended-precision
// Schur–Cohn, margins and |S| from the coefficients.
func TestPidtuneUnstableRetryMarginSlack(t *testing.T) {
	const dt = 0.01
	num := Poly{9.086858380437903e-15, -5.791919405607725e-15, 1.4667499867564151e-16}
	den := Poly{1, -6.1841221951263785, 16.283637862682834, -23.64207337573077, 20.414839732703783, -10.466618593329219, 2.9435553564625145, -0.3492187876627648}
	pid, err := Pidtune(pidtuneBackoffPlant(t, num, den, dt, 0), PidtuneP, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	pidDCSignCheckRobust(t, "P", pid, num, den, dt, 58.5)
}
