package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

// OQWQKF: every automatic design meets one bar, oracle-stable with 58.5° at
// every gain crossover and |S| ≤ 4, and when back-off fails Pidtune raises
// the crossover before giving up. Back-off lowers wc, losing phase on
// these unstable plants (PID PM 28.3° and 46.8°, PD unstable at wc), and
// the |S|-only rule for P on zero-DC k·sᵐ plants kept 48.5°, 29.5°, 28.9°,
// 29.1° and 33.9° at a crossover; raised crossovers meet the bar. Oracle:
// closed-loop roots, signed margins and |S| of the directly evaluated loop.
func TestPidtuneRaisedCrossoverMeetsBar(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtunePID, Poly{0.08119770312059038, 1.4299696862733304, 1.1509782187662239, 0.74301299939046}, Poly{1, 1.5247143694217347, 0.26500427775406976, 1.1490289607929067}},
		{PidtunePID, Poly{0.012117279206722564, 0.6998831440522459, 1.7138236892014762, 1.8876491780936953}, Poly{1, 0.30102668894257456, 1.2322300903394714, 1.7765490062040799}},
		{PidtunePD, Poly{0.4820870937175388, 1.3079853107003931, 1.9844096182517548, -1.0452039084558389}, Poly{1, 0.8298649253038679, 1.349639061840446, 1.53625974244542}},
		{PidtuneP, Poly{1.4392005301712567, 0, 0}, Poly{1, 8.24762471156669, 20.7899930610465, 15.321468764445184}},
		{PidtuneP, Poly{0.8542595646531987, 0, 0}, Poly{1, 7.023342124760237, 12.926235567617814, 2.066641791908522}},
		{PidtuneP, Poly{-0.4747823914433865, 0, 0}, Poly{1, 1.7019690144285535, 0.19889294875649396}},
		{PidtuneP, Poly{-3.815515468598575, 0, 0}, Poly{1, 7.773322676859117, 15.64159738856245, 2.4039470279016206}},
		{PidtuneP, Poly{0.17174745558754434, 0, 0}, Poly{1, 4.304762793303786, 3.0119125343741975}},
	} {
		name := fmt.Sprintf("%s %v/%v", c.f, c.num, c.den)
		pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, 0, 0), c.f, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, pidTuningMaxSensitivity+0.05)
	}

	num := Poly{6.749592744149843e-16, -6.056872503393394e-16}
	den := Poly{1, -7.9623743329365935, 28.09350817041953, -57.63766640485889, 75.76142091126783, -66.14845014857204, 38.35309958277561, -14.235001137708403, 3.067835464760931, -0.29237210514798584}
	pid, err := Pidtune(pidtuneBackoffPlant(t, num, den, 0.01, 0), PidtunePI, 0, nil)
	pidtuneRobustOrUnattainable(t, "PI discrete", pid, err, num, den, 0.01, 60)

	num, den = Poly{-0.8228803215170728}, Poly{1, -1.1691124906967234}
	pid, err = Pidtune(pidtuneBackoffPlant(t, num, den, 0.23622257780900835, 0), PidtuneP, 0, nil)
	if err != nil {
		t.Fatalf("P discrete: %v", err)
	}
	pidtuneCheckRobust(t, "P discrete", pid, num, den, 0.23622257780900835, pidTuningMaxSensitivity+0.05)
}

// P on an unstable zero-DC k·s²/(s² + a₁s + a₀), a₀ < 0, is stable only for
// 1 + Kp·k < 0 (with a₁ < 0), where |L(j∞)| = |Kp·k| > 1; the |S|-only rule
// returned PM 21.2°, 19.1°, 20.3° and 17.5°. Pidtune now returns
// ErrPIDTuningTargetUnattainable. Oracle: no Kp of either sign on a dense
// log grid gives an oracle-stable loop with 58.5° at every crossover and
// |S| ≤ 4.
func TestPidtuneZeroDCUnstablePUnattainable(t *testing.T) {
	for _, c := range []struct{ num, den Poly }{
		{Poly{0.799670298477293, 0, 0}, Poly{1, -1.5732586048745847, -0.7058774442032956}},
		{Poly{0.058830338909354074, 0, 0}, Poly{1, -0.7250580809031377, -0.2552994670450437}},
		{Poly{-0.12280682141220423, 0, 0}, Poly{1, -0.7736639107226584, -0.2148870741051826}},
		{Poly{-0.74052413307732, 0, 0}, Poly{1, -1.6654877956902752, -1.9024369107095214}},
	} {
		name := fmt.Sprintf("%v/%v", c.num, c.den)
		if pid, err := Pidtune(pidtuneBackoffPlant(t, c.num, c.den, 0, 0), PidtuneP, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s: %+v, %v; want ErrPIDTuningTargetUnattainable", name, pid, err)
		}
		for e := -3.0; e <= 6; e += 0.02 {
			for _, k := range []float64{math.Pow(10, e), -math.Pow(10, e)} {
				c2 := &PID{Kp: k}
				if stable, ok := pidtuneOracleStable(t, c2, c.num, c.den, 0, 0); !ok || !stable {
					continue
				}
				pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(c2, c.num, c.den, 0, 0), 1e-6, 1e6)
				good := len(pms) > 0 && ms <= pidTuningMaxSensitivity
				for _, m := range pms {
					good = good && math.Abs(m) >= 58.5
				}
				if good {
					t.Errorf("%s: Kp %g keeps %v with |S| %.3g", name, k, pms, ms)
				}
			}
		}
	}
}
