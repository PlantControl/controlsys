package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
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
	skipSlowUnderRace(t)
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

// X46RST: on a discrete plant with a pole cluster at z = 1, the gate before
// e3d0a6d/49041a9 read NaN plant samples, rejecting every family with
// "phase margin NaN°" (amd64) or returning PM 0.84° (arm64). Every design
// now keeps 58.5° at every crossover with |S| ≤ 4. Oracle: exact Schur–Cohn
// test (float closed-loop roots cannot resolve PI/PID/PIDF here), signed
// margins and |S| of the loop with the plant evaluated in 256-bit arithmetic:
// float64 Horner on den loses |G| below ω ≈ 1e-4 (I: 307 spurious crossovers,
// |S| 232; true PM 58.7°, |S| 1.36).
func TestPidtuneDisplacedIntegratorMeetsBar(t *testing.T) {
	skipSlowUnderRace(t)
	num := Poly{1.8748015424080903e-16}
	den := Poly{1, -6.656527066165689, 18.974082402962605, -30.020939359334676, 28.473509438245436, -16.187809823253666, 5.1075991528389695, -0.6899147452929796}
	const dt = 0.1
	p := pidtuneBackoffPlant(t, num, den, dt, 0)
	for _, f := range pidtuneAllTypes {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if f == PidtuneI && errors.Is(err, ErrPIDTuningTargetUnattainable) {
				continue
			}
			t.Errorf("%s: %v", f, err)
			continue
		}
		if !pidtuneExactSchur(pid, num, den, dt) {
			t.Errorf("%s: %+v unstable", f, pid)
		}
		numC, denC := pidtuneOracleController(pid, dt)
		L := func(w float64) complex128 {
			z := cmplx.Exp(complex(0, w*dt))
			return numC.Eval(z) / denC.Eval(z) * pidtuneBigEval(num, z) / pidtuneBigEval(den, z)
		}
		pms, ms := pidtuneSignedMargins(L, 1e-8, math.Pi/dt)
		if len(pms) == 0 || ms > pidTuningMaxSensitivity+0.05 {
			t.Errorf("%s: %+v crossovers %v, |S| %.3g", f, pid, pms, ms)
		}
		for _, m := range pms {
			if math.Abs(m) < 58.5 {
				t.Errorf("%s: %+v margin %.3g°", f, pid, m)
			}
		}
	}
}

// pidtuneBigEval returns p(z) by Horner's rule in 256-bit arithmetic,
// exact for the float64 z and coefficients up to the final rounding.
func pidtuneBigEval(p Poly, z complex128) complex128 {
	const prec = 256
	f := func(v float64) *big.Float { return new(big.Float).SetPrec(prec).SetFloat64(v) }
	zr, zi := f(real(z)), f(imag(z))
	re, im, nr, ni, x, c := f(0), f(0), f(0), f(0), f(0), f(0)
	for _, v := range p {
		nr.Mul(re, zr)
		nr.Sub(nr, x.Mul(im, zi))
		ni.Mul(re, zi)
		ni.Add(ni, x.Mul(im, zr))
		nr.Add(nr, c.SetFloat64(v))
		re, nr, im, ni = nr, re, ni, im
	}
	a, _ := re.Float64()
	b, _ := im.Float64()
	return complex(a, b)
}

// X46RST: auto PID on this unstable biproper plant returns Kd = 0 with
// |L(wc)| = 1 at the minimum of |L|, ∠L = −120° there; |L| ≥ 1 elsewhere
// (6.3 at 1e-7, 1.31 plateau, 2.24 at ∞), so a sign-change scan sees no
// crossover. Not spurious: the touch keeps 60° and |S| 3.31. The touch is
// fragile, outside the bar: gains ×0.99 cross at 49.9° and 70.0°, ×0.9 at
// 26.5° with |S| 5.7. Oracle: exact Routh test, |S| and signed margins of
// the directly evaluated loop, and its |L| minimum on a dense log grid.
func TestPidtuneTangentCrossoverMeetsBar(t *testing.T) {
	num := Poly{1.7837475641504483, 0.7946993866758785, 1.4552772148607651, 6.888693123877885e-07}
	den := Poly{1, 0.1506135003142426, 1.3911883292147762, 0.612642264165268}
	pid, err := Pidtune(pidtuneBackoffPlant(t, num, den, 0, 0), PidtunePID, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !pidtuneExactHurwitz(pid, num, den) {
		t.Errorf("%+v unstable", pid)
	}
	L := pidtuneOracleLoop(pid, num, den, 0, 0)
	pms, ms := pidtuneSignedMargins(L, 1e-9, 1e6)
	if ms > pidTuningMaxSensitivity {
		t.Errorf("%+v |S| %.3g", pid, ms)
	}
	for _, m := range pms {
		if math.Abs(m) < 58.5 {
			t.Errorf("%+v margin %.3g°", pid, m)
		}
	}
	best, low := 0.0, math.Inf(1)
	for e := -1.0; e <= 1; e += 1e-5 {
		w := math.Pow(10, e)
		if a := cmplx.Abs(L(w)); a < low {
			best, low = w, a
		}
	}
	m := 180 - math.Abs(cmplx.Phase(L(best)))*180/math.Pi
	if math.Abs(low-1) > 1e-6 || m < 58.5 {
		t.Errorf("%+v: min |L| %.9g at %g, margin %.3g°", pid, low, best, m)
	}
}

// NWFI4Y: DCGain is finite next to an integrator rounded inside z = 1
// (MATLAB dcgain = evalfr at z = 1), so pidTuningDCSign must recognise the
// integrator itself: plain DCGain gave +1 for the first two plants (0.2084
// and 34.9, exact for the stored coefficients; on amd64 the second's
// computed pole rounds outside z = 1, so it is unstable there). A pole
// radius rule (pidtuneRoundedIntegrator) instead gave 0 for the third,
// GK5SUC's plant with finite DC gain 85 and a pole beside z = 1. Oracle:
// exact D(1) of the stored coefficients is within ε/2·Σ|dᵢ|, the most
// rounding each dᵢ to float64 moves it, only for the integrators (third
// plant 4.1ε·Σ|dᵢ|); ordinary plants keep their DC sign.
func TestPidtuneDCSignRecognisesRoundedIntegrator(t *testing.T) {
	const dt = 0.01
	noSample := func(float64) (complex128, bool) { return 0, false }
	atRounding := func(den Poly) bool {
		d1, mag := new(big.Float).SetPrec(256), 0.0
		for _, c := range den {
			d1.Add(d1, new(big.Float).SetPrec(256).SetFloat64(c))
			mag += math.Abs(c)
		}
		v, _ := d1.Float64()
		return math.Abs(v) <= 0x1p-53*mag
	}
	for _, c := range []struct {
		num, den Poly
		want     float64
	}{
		{Poly{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}, Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}, 0},
		{Poly{3.875422988056212e-15}, Poly{1, -4.667597493849348, 8.695671657313316, -8.081072724970674, 3.7455208432476796, -0.692522281740973}, 0},
		{Poly{1.5379663637764417e-12, -3.387450861436736e-13}, Poly{1, -3.9146780141541093, 5.7453934609919255, -3.7467528452104566, 0.9160373983726545}, 1},
		{Poly{1e-3}, Poly{1, -0.999}, 1},
		{Poly{-2, 1}, Poly{1, -1.2, 0.35}, -1},
	} {
		name := fmt.Sprintf("%v/%v", c.num, c.den)
		if atRounding(c.den) != (c.want == 0) {
			t.Fatalf("%s: oracle D(1) at rounding = %v", name, c.want != 0)
		}
		p := pidtuneBackoffPlant(t, c.num, c.den, dt, 0)
		dc, err := p.DCGain()
		if err != nil {
			t.Fatalf("%s: DCGain: %v", name, err)
		}
		if got := pidTuningDCSign(p, noSample, 1e-4); got != c.want {
			t.Errorf("%s: pidTuningDCSign %g, want %g (DCGain %g)", name, got, c.want, dc.At(0, 0))
		}
	}
}
