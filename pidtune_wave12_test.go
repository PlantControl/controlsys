package controlsys

import (
	"errors"
	"fmt"
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
