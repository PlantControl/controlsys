package controlsys

import (
	"context"
	"fmt"
	"math"
	"math/big"
	"strings"
	"testing"
)

const pidDCSignPrec = 400

func pidDCSignBig(p Poly) []*big.Float {
	out := make([]*big.Float, len(p))
	for i, c := range p {
		out[i] = new(big.Float).SetPrec(pidDCSignPrec).SetFloat64(c)
	}
	return out
}

func pidDCSignBigMulAdd(a, b, c, d Poly) []*big.Float {
	ab, cd := pidDCSignBigMul(a, b), pidDCSignBigMul(c, d)
	if len(ab) < len(cd) {
		ab, cd = cd, ab
	}
	off := len(ab) - len(cd)
	for i, v := range cd {
		ab[off+i].Add(ab[off+i], v)
	}
	return ab
}

func pidDCSignBigMul(a, b Poly) []*big.Float {
	x, y := pidDCSignBig(a), pidDCSignBig(b)
	out := pidDCSignBig(make(Poly, len(a)+len(b)-1))
	t := new(big.Float).SetPrec(pidDCSignPrec)
	for i := range x {
		for j := range y {
			out[i+j].Add(out[i+j], t.Mul(x[i], y[j]))
		}
	}
	return out
}

// pidDCSignSchurCohn reports whether every root of p (descending powers)
// lies strictly inside the unit circle, by the Schur–Cohn recursion in
// extended precision.
func pidDCSignSchurCohn(p []*big.Float) bool {
	for len(p) > 1 && p[0].Sign() == 0 {
		p = p[1:]
	}
	for len(p) > 1 {
		n := len(p) - 1
		a, b := p[0], p[n]
		if new(big.Float).Abs(b).Cmp(new(big.Float).Abs(a)) >= 0 {
			return false
		}
		q := make([]*big.Float, n)
		t := new(big.Float).SetPrec(pidDCSignPrec)
		for k := range n {
			q[k] = new(big.Float).SetPrec(pidDCSignPrec).Mul(a, p[k])
			q[k].Sub(q[k], t.Mul(b, p[n-k]))
		}
		p = q
	}
	return true
}

// pidDCSignEval evaluates p at e^{jθ} in extended precision, with z − 1
// formed as −2sin²(θ/2) + j·sin θ so samples beside z = 1 keep their
// relative accuracy.
func pidDCSignEval(p Poly, th float64) complex128 {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(pidDCSignPrec).SetFloat64(x) }
	sh := math.Sin(th / 2)
	zr := nf(1)
	zr.Sub(zr, nf(2*sh*sh))
	zi := nf(math.Sin(th))
	re, im, t1, t2 := nf(0), nf(0), nf(0), nf(0)
	for _, c := range p {
		nr := nf(0).Sub(t1.Mul(re, zr), t2.Mul(im, zi))
		im.Add(t1.Mul(re, zi), t2.Mul(im, zr))
		re.Add(nr, nf(c))
	}
	r, _ := re.Float64()
	i, _ := im.Float64()
	return complex(r, i)
}

// pidDCSignCheckRobust checks pid on the discrete num/den by extended
// precision oracles: Schur–Cohn on the closed-loop polynomial, a gain
// crossover with every signed margin at least minPM, and |S| ≤ 4.
func pidDCSignCheckRobust(t *testing.T, name string, pid *PID, num, den Poly, dt, minPM float64) {
	t.Helper()
	numC, denC := pidtuneOracleController(pid, dt)
	if !pidDCSignSchurCohn(pidDCSignBigMulAdd(den, denC, num, numC)) {
		t.Errorf("%s: %+v unstable", name, pid)
	}
	L := func(w float64) complex128 {
		th := w * dt
		return pidDCSignEval(num, th) * pidDCSignEval(numC, th) / (pidDCSignEval(den, th) * pidDCSignEval(denC, th))
	}
	pms, ms := pidtuneSignedMargins(L, 1e-10, math.Pi/dt)
	if len(pms) == 0 || ms > pidTuningMaxSensitivity {
		t.Errorf("%s: %+v crossovers %v, |S| %.3g", name, pid, pms, ms)
	}
	for _, m := range pms {
		if math.Abs(m) < minPM {
			t.Errorf("%s: %+v margin %.3g°", name, pid, m)
		}
	}
}

// GK5SUC: pidTuningDCSign samples |G| over six decades around wc only to
// test the DC gain for a relative zero. On stable discrete plants with
// poles beside z = 1 the sample at wc/1000 fails, and its sticky eval.err
// made Pidtune and TunePID fatal ("loop response at ω=wc/1000") though the
// DC gain (84.4, 339) is finite and every other sample resolves. Oracles:
// Schur–Cohn on the closed-loop polynomial and margins and |S| of the loop,
// both in extended precision from the coefficients. TunePID's designs keep
// their reported margin at every crossover; its other failures on these
// plants, such as PI and (on amd64) PID at the phase walk's start wc·1e-6,
// lie past pidTuningDCSign.
func TestPidtuneDCSignFailedSample(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	const dt = 0.01
	cases := []struct {
		num, den Poly
		tune     []PidtuneType
	}{
		{Poly{1.5379663637764417e-12, -3.387450861436736e-13}, Poly{1, -3.9146780141541093, 5.7453934609919255, -3.7467528452104566, 0.9160373983726545}, []PidtuneType{PidtuneP, PidtunePD, PidtunePDF, PidtunePIDF}},
		{Poly{1.7295311989236164e-12}, Poly{1, -2.999890083275418, 2.999780168026009, -0.999890084750586}, nil},
	}
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, dt, 0)
		wc, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := eval.try(wc / 1000); ok {
			t.Fatalf("%v/%v: precondition: sample at wc/1000 = %g resolves", c.num, c.den, wc/1000)
		}
		if s := pidTuningDCSign(p, eval.try, wc); s != 1 {
			t.Errorf("%v/%v: DC sign %g, want 1", c.num, c.den, s)
		}
		sample := fmt.Sprintf("ω=%g:", wc/1000)
		for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
			name := fmt.Sprintf("%s %v/%v", f, c.num, c.den)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			pidDCSignCheckRobust(t, name, pid, c.num, c.den, dt, 58.5)
		}
		for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
			if _, err := TunePID(context.Background(), p, f, PIDTuningOptions{}); err != nil && strings.Contains(err.Error(), sample) {
				t.Errorf("TunePID %s %v/%v: %v", f, c.num, c.den, err)
			}
		}
		for _, f := range c.tune {
			name := fmt.Sprintf("TunePID %s %v/%v", f, c.num, c.den)
			res, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			k := res.Controller
			if k.IFormula != 0 {
				t.Fatalf("%s: IFormula %v outside the oracle", name, k.IFormula)
			}
			pidDCSignCheckRobust(t, name, &PID{DFormula: k.DFormula, Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt}, c.num, c.den, dt, res.Evidence.AchievedPhaseMargin-0.5)
		}
	}
}
