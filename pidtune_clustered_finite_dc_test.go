package controlsys

import (
	"context"
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidtuneExactSchurTwoState reports whether the discrete loop of pid on the 2-state
// SISO plant (a, b, c, d), row-major a, has every closed-loop root in
// |z| < 1, by the Schur–Cohn recursion on the characteristic polynomial
// built from the exact coefficients in 2048-bit floats: the plant's
// det(zI−A) has a constant term 1 + 4.6e-18 that a float64 transfer
// function cannot hold.
func pidtuneExactSchurTwoState(pid *PID, a, b, c []float64, d, dt float64) bool {
	r := pidtuneRat
	mul := func(x, y *big.Rat) *big.Rat { return new(big.Rat).Mul(x, y) }
	add := func(x, y *big.Rat) *big.Rat { return new(big.Rat).Add(x, y) }
	sub := func(x, y *big.Rat) *big.Rat { return new(big.Rat).Sub(x, y) }
	a00, a01, a10, a11 := r(a[0]), r(a[1]), r(a[2]), r(a[3])
	den := pidtuneRatPoly(r(1), new(big.Rat).Neg(add(a00, a11)), sub(mul(a00, a11), mul(a01, a10)))
	b0, b1, c0, c1 := r(b[0]), r(b[1]), r(c[0]), r(c[1])
	cb := add(mul(c0, b0), mul(c1, b1))
	k0 := add(add(new(big.Rat).Neg(mul(a11, mul(c0, b0))), mul(c0, mul(a01, b1))), sub(mul(c1, mul(a10, b0)), mul(a00, mul(c1, b1))))
	num := pidtuneRatAdd(pidtuneRatMul(pidtuneRatPoly(r(d)), den), pidtuneRatPoly(cb, k0))

	rdt, one := r(dt), r(1)
	x := pidtuneRatPoly(one, r(-1))
	wi, wd := r(pidFormulaWeight(pid.IFormula)), r(pidFormulaWeight(pid.DFormula))
	iNum := pidtuneRatMul(pidtuneRatPoly(mul(r(pid.Ki), rdt)), pidtuneRatPoly(wi, sub(one, wi)))
	lead := add(r(pid.Tf), mul(wd, rdt))
	dd := pidtuneRatPoly(lead, sub(rdt, lead))
	if dd[0].Sign() == 0 {
		dd = dd[1:]
	}
	kp, kd := pidtuneRatPoly(r(pid.Kp)), pidtuneRatPoly(r(pid.Kd))
	numC, denC := kp, pidtuneRatPoly(one)
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = pidtuneRatAdd(pidtuneRatAdd(pidtuneRatMul(pidtuneRatMul(kp, x), dd), pidtuneRatMul(iNum, dd)), pidtuneRatMul(pidtuneRatMul(kd, x), x))
		denC = pidtuneRatMul(x, dd)
	case pid.Ki != 0:
		numC, denC = pidtuneRatAdd(pidtuneRatMul(kp, x), iNum), x
	case pid.Kd != 0:
		numC, denC = pidtuneRatAdd(pidtuneRatMul(kp, dd), pidtuneRatMul(kd, x)), dd
	}
	ch := pidtuneRatAdd(pidtuneRatMul(den, denC), pidtuneRatMul(num, numC))
	for len(ch) > 1 && ch[0].Sign() == 0 {
		ch = ch[1:]
	}
	const prec = 2048
	f := make([]*big.Float, len(ch))
	for i, v := range ch {
		f[i] = new(big.Float).SetPrec(prec).SetRat(v)
	}
	for n := len(f) - 1; n > 0; n-- {
		k := new(big.Float).SetPrec(prec).Quo(f[n], f[0])
		if new(big.Float).Abs(k).Cmp(big.NewFloat(1)) >= 0 {
			return false
		}
		q := make([]*big.Float, n)
		for i := range q {
			q[i] = new(big.Float).SetPrec(prec).Mul(k, f[n-i])
			q[i].Sub(f[i], q[i])
		}
		f = q
	}
	return true
}

// YRNRQP (w13 #64): a biproper plant with |G| > 1 everywhere whose two
// poles (1 ± 2.15e-9i) and two zeros (1 − 1.47e-9, 1 − 9.8e-10) lie
// inside pidtuneIntegratorCluster's z = 1 radius has a finite DC gain
// (161), so it missed the integrating path, and the pole fallback, whose
// cluster excludes both poles, left nothing: Pidtune and TunePID returned
// ErrInvalidArgument. It now takes the integrating crossover an exact
// integrator gets, 0.1π/Dt here. Oracles: that cap; every Pidtune family
// and TunePID P, I, PI, PID and PIDF design, stable by Schur–Cohn on the
// exact characteristic polynomial.
func TestPidtuneClusteredPolesFiniteDCCrossover(t *testing.T) {
	a := []float64{1.0000000116488188, 1.8099675739185816e-08, -7.753540390871093e-09, 0.9999999883511813}
	b := []float64{0.6979798590772273, -0.5033119781891744}
	c := []float64{1.4839794477710827e-06, -4.772606107217278e-07}
	const d, dt = 520.8700127245612, 0.1
	p, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, b), mat.NewDense(1, 2, c), mat.NewDense(1, 1, []float64{d}), dt)
	if err != nil {
		t.Fatal(err)
	}
	if wc, err := plantCrossoverFreq(p); err != nil || math.Abs(wc-0.1*math.Pi/dt) > 1e-12 {
		t.Errorf("crossover %g, %v; want %g", wc, err, 0.1*math.Pi/dt)
	}
	for _, f := range pidtuneAllTypes {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			t.Errorf("Pidtune %s: %v", f, err)
			continue
		}
		if !pidtuneExactSchurTwoState(pid, a, b, c, d, dt) {
			t.Errorf("Pidtune %s: %+v unstable", f, pid)
		}
	}
	for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
		res, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
		if err != nil {
			t.Errorf("TunePID %s: %v", f, err)
			continue
		}
		k := res.Controller
		pid := &PID{IFormula: k.IFormula, DFormula: k.DFormula, Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt}
		if !pidtuneExactSchurTwoState(pid, a, b, c, d, dt) {
			t.Errorf("TunePID %s: %+v unstable", f, k)
		}
	}
}

// YRNRQP (w13 #10): AllMargin's first gain crossing of this discrete plant
// is at rounding level (3.2e-11 rad/s); Pidtune once errored there and
// must keep designing. Oracle: exact Schur–Cohn.
func TestPidtuneRoundingLevelCrossoverDesigns(t *testing.T) {
	num := Poly{4.905554616319012e-13}
	den := Poly{1, -2.310228431318273, 1.8436939371933398, -0.6052316064965124, 0.07495077124036684, -0.0032273933261639186, 4.2722707242222904e-05}
	p := pidtuneBackoffPlant(t, num, den, 0.1, 0)
	for _, f := range []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePID} {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		if !pidtuneExactSchur(pid, num, den, 0.1) {
			t.Errorf("%s: %+v unstable", f, pid)
		}
	}
}
