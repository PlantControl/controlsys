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
