package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"
)

// pidtuneExactHurwitz reports whether the continuous loop of pid on num/den
// has every closed-loop root in Re s < 0, by the Routh array in exact
// rationals: a closed-loop root near a zero at 1e-7 defeats floating-point
// root finding.
func pidtuneExactHurwitz(pid *PID, num, den Poly) bool {
	r := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	poly := func(p Poly) []*big.Rat {
		out := make([]*big.Rat, len(p))
		for i, v := range p {
			out[i] = r(v)
		}
		return out
	}
	mul := func(a, b []*big.Rat) []*big.Rat {
		out := make([]*big.Rat, len(a)+len(b)-1)
		for i := range out {
			out[i] = new(big.Rat)
		}
		for i, x := range a {
			for j, y := range b {
				out[i+j].Add(out[i+j], new(big.Rat).Mul(x, y))
			}
		}
		return out
	}
	add := func(a, b []*big.Rat) []*big.Rat {
		if len(a) < len(b) {
			a, b = b, a
		}
		out := make([]*big.Rat, len(a))
		for i := range a {
			out[i] = new(big.Rat).Set(a[i])
			if k := i - (len(a) - len(b)); k >= 0 {
				out[i].Add(out[i], b[k])
			}
		}
		return out
	}
	s := []*big.Rat{r(1), r(0)}
	kp, ki, kd := []*big.Rat{r(pid.Kp)}, []*big.Rat{r(pid.Ki)}, []*big.Rat{r(pid.Kd)}
	dd := []*big.Rat{r(pid.Tf), r(1)}
	if pid.Tf == 0 {
		dd = dd[1:]
	}
	numC, denC := kp, []*big.Rat{r(1)}
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = add(add(mul(mul(kp, s), dd), mul(ki, dd)), mul(mul(kd, s), s))
		denC = mul(s, dd)
	case pid.Ki != 0:
		numC, denC = add(mul(kp, s), ki), s
	case pid.Kd != 0:
		numC, denC = add(mul(kp, dd), mul(kd, s)), dd
	}
	c := add(mul(poly(den), denC), mul(poly(num), numC))
	for len(c) > 1 && c[0].Sign() == 0 {
		c = c[1:]
	}
	if c[0].Sign() < 0 {
		for _, v := range c {
			v.Neg(v)
		}
	}
	row := func(k int) []*big.Rat {
		var out []*big.Rat
		for i := k; i < len(c); i += 2 {
			out = append(out, c[i])
		}
		return out
	}
	prev, cur := row(0), row(1)
	for n := len(c) - 1; n > 0; n-- {
		if len(cur) == 0 || cur[0].Sign() <= 0 {
			return false
		}
		next := make([]*big.Rat, max(len(prev)-1, 0))
		for i := range next {
			b := new(big.Rat)
			if i+1 < len(cur) {
				b.Set(cur[i+1])
			}
			v := new(big.Rat).Mul(cur[0], prev[i+1])
			v.Sub(v, new(big.Rat).Mul(prev[0], b))
			next[i] = v.Quo(v, cur[0])
		}
		prev, cur = cur, next
	}
	return true
}

// pidtuneRolloffRobust checks a design by the exact Routh test and a dense
// sweep of the directly evaluated loop on [1e-10, 1e7]: |S| ≤ 4 and, but for
// P and PD, which the pole fallback gates by |S| alone, the classical margin
// 180° − |∠L| ≥ 60° at every crossover (sampling slack only).
func pidtuneRolloffRobust(t *testing.T, name string, f PidtuneType, pid *PID, num, den Poly) {
	t.Helper()
	if !pidtuneExactHurwitz(pid, num, den) {
		t.Errorf("%s: %+v closed loop not Hurwitz", name, pid)
		return
	}
	pm, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, num, den, 0, 0), 1e-10, 1e7, 1.001)
	if ms > 4.05 || f != PidtuneP && f != PidtunePD && pm < 58.5 {
		t.Errorf("%s: %+v phase margin %.3g, |S| %.3g", name, pid, pm, ms)
	}
}

// FPLHR5 review: a bandwidth beyond ten times every pole and zero natural
// frequency that |G| rises less than 1e3 times above, as for mid-band or
// non-minimum-phase zeros, is the crossover, as before TA526O, which took
// the pole fallback for these plants and turned robust designs into errors
// (P "unstable", PDF "needs phase lag", PD |S| 4.02). Oracle: crossover
// against Bandwidth; exact Routh test, dense margins and |S|.
func TestPidtuneRolloffKeepsBandwidth(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		designs  []PidtuneType
	}{
		{Poly{0.9259305881240456, -0.25001768378306766, -14.09549687784304, -4.875339983688054},
			Poly{1, 10.913817523403575, 51.69190733555752, 143.2440252325901, 183.43503385908775},
			[]PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePDF, PidtunePID, PidtunePIDF}},
		{Poly{0.7719873206417287, -1.6763502758609454, -0.7304346890048893},
			Poly{1, 3.240539909594753, 69.92813113932584, 138.60815692432595},
			[]PidtuneType{PidtuneI, PidtunePDF, PidtunePID, PidtunePIDF}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		bw, err := Bandwidth(p, -3)
		if err != nil {
			t.Fatal(err)
		}
		if wc, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || fromPole || wc != bw {
			t.Fatalf("%v/%v: crossover %g (pole %v), %v; want bandwidth %g", c.num, c.den, wc, fromPole, err, bw)
		}
		for _, f := range c.designs {
			name := fmt.Sprintf("%s %v/%v", f, c.num, c.den)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			pidtuneRolloffRobust(t, name, f, pid, c.num, c.den)
		}
	}
}

// FPLHR5 review: zeros at 1e-8..1e-4 of the poles on plants of relative
// degree 2 and 3 put the bandwidth on the rolloff asymptote, where gains
// reach 1e8..1e12. The crossover is the first back-off step from the
// bandwidth below ten times the fastest root, a pole fallback whose back-off
// visits the steps a back-off from the bandwidth would, so designs found
// before TA526O are still found; the half-slowest-pole fallback gave errors
// and, backing off 6 decades, PI Kp −7.1e6 with |S| 340 and PDF Kd 8.3e9
// with PM 0.4°. Oracle: crossover from num/den's roots and Bandwidth;
// exact Routh test, dense margins and |S|, gains within 1e4/max|G|.
func TestPidtuneRolloffAttenuatedFallback(t *testing.T) {
	all := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	for _, c := range []struct {
		num, den Poly
		designs  []PidtuneType
	}{
		{Poly{-0.9723820423024041, -1.1581311208815152e-08}, Poly{1, 4.557094171719956, 7.307037355121531, 6.999193219108127, 2.0176236467017628}, all},
		{Poly{0.8484599182261484, 1.7083072773596847e-07}, Poly{1, 4.443225162574768, 20.071278658927426, 41.44198818240536, 36.85105879735006}, all},
		{Poly{-0.582250944976203, -1.19673211284052e-07}, Poly{1, 5.944612275806493, 19.277112508793728, 50.8630355439298}, all},
		{Poly{1.4030199221250264, 3.3086036514615857e-06}, Poly{1, 8.702209332684196, 19.80306857300817, 3.7622996702063825}, all},
		{Poly{0.7288751830680972, 5.863987566648214e-08}, Poly{1, 6.218349479753987, 22.14963750783292, 48.15474136069117}, all},
		{Poly{0.7283224838588798, -0.28928965480818264, -2.564425437218254e-05}, Poly{1, 3.2935694302816767, 6.247135618618195, 9.097330722097237, 4.247152681589145},
			[]PidtuneType{PidtuneI, PidtunePDF, PidtunePID, PidtunePIDF}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		poles, err := c.den.Roots()
		if err != nil {
			t.Fatal(err)
		}
		zeros, err := c.num.Roots()
		if err != nil {
			t.Fatal(err)
		}
		fast := 0.0
		for _, r := range slices.Concat(poles, zeros) {
			fast = math.Max(fast, cmplx.Abs(r))
		}
		bw, err := Bandwidth(p, -3)
		if err != nil {
			t.Fatal(err)
		}
		want := bw * math.Pow(10, -math.Ceil(20*math.Log10(bw/(10*fast)))/20)
		wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
		if err != nil || !fromPole || math.Abs(wc-want) > 1e-6*want {
			t.Fatalf("%v/%v: crossover %g (pole %v), %v; want %g", c.num, c.den, wc, fromPole, err, want)
		}
		peak := 0.0
		for w := 1e-10; w < 1e7; w *= 1.01 {
			peak = math.Max(peak, cmplx.Abs(c.num.Eval(complex(0, w))/c.den.Eval(complex(0, w))))
		}
		for _, f := range all {
			name := fmt.Sprintf("%s %v/%v", f, c.num, c.den)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) || slices.Contains(c.designs, f) {
					t.Errorf("%s: %v", name, err)
				}
				continue
			}
			if g := 1e4 / peak; !(math.Abs(pid.Kp) <= g && math.Abs(pid.Ki) <= g*wc && math.Abs(pid.Kd) <= g/wc*1e2) {
				t.Errorf("%s: %+v; gains above %g", name, pid, g)
			}
			pidtuneRolloffRobust(t, name, f, pid, c.num, c.den)
		}
	}
}

// FPLHR5 review: the back-off gate's band ends at 1e3 times the automatic
// crossover, and a design backed off below it can cross 0 dB again above
// it: PDF Kp −479, Kd −6.4e5, Tf 2639 crossed at 36.4 rad/s, above the band
// end 21.3, with PM 1.2° and |S| 49.6. Above the band the gate scans up to
// where the loop's Markov tail bound keeps |L| ≤ 3/4. Oracle: exact Routh
// test, dense margins and |S| up to 1e7 rad/s.
func TestPidtuneGateScansAboveBand(t *testing.T) {
	num, den := Poly{-1.8326830463596728, -4.165603628320087e-05}, Poly{1, 0.7334709862442443, 0.25518924748370436, 0.03900616993539339}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	for _, f := range []PidtuneType{PidtunePDF, PidtunePIDF} {
		name := fmt.Sprintf("%s %v/%v", f, num, den)
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		pidtuneRolloffRobust(t, name, f, pid, num, den)
	}
}

// pidtuneExactHurwitz against roots: s² + 3s + 2 + K under P is Hurwitz for
// K > −2; a PI's integrator with Ki < 0 adds a root in Re s > 0.
func TestPidtuneExactHurwitzOracle(t *testing.T) {
	for _, c := range []struct {
		pid  PID
		want bool
	}{
		{PID{Kp: 1}, true}, {PID{Kp: -1.9}, true}, {PID{Kp: -2}, false}, {PID{Kp: -3}, false},
		{PID{Kp: 1, Ki: 1}, true}, {PID{Kp: 1, Ki: -1}, false}, {PID{Kp: 1, Ki: 100}, false},
		{PID{Kp: 1, Kd: 1, Tf: 0.1}, true}, {PID{Kp: 1, Ki: 1, Kd: -4}, false}, {PID{Kp: 1, Ki: 1, Kd: -0.5}, true},
	} {
		if got := pidtuneExactHurwitz(&c.pid, Poly{1}, Poly{1, 3, 2}); got != c.want {
			t.Errorf("%+v: Hurwitz %v, want %v", c.pid, got, c.want)
		}
	}
}
