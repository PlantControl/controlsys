package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Badly scaled SISO realizations from the wave-18 fuzz campaign (YEAQIK):
// rows of A five orders of magnitude apart and C ~ 1e-6.
var (
	badScaleIntA = []float64{
		1.0579408711784543e+06, -2.7317975836031805e+06, 2.9703941158074806e+06, -1.4978162396691812e+06, -711412.3632025617, 3.7288939714288316e+06,
		729745.356985211, -1.8830733218656706e+06, 2.047076948089578e+06, -1.0324005829670081e+06, -490343.5291307842, 2.5701929041342805e+06,
		271957.5660142623, -701168.8474183332, 762023.6629196389, -384413.00426393596, -182546.48865383747, 956944.1545634485,
		72.3088030933218, -186.42864350567376, 218.9933124005808, -147.23244458326784, -27.476461205089215, 282.62551339179015,
		25.077895261688333, -64.65655349847313, 75.95052215570654, -51.05830140010958, -9.532991206947965, 98.0172709792755,
		0, 0, 0, 0, 0, 0,
	}
	badScaleIntB = []float64{-0.9667981925794609, -0.01238866346289056, 0.9601702866503661, 0.5260775173811046, -0.6787200473200125, 0.8438539587324921}
	badScaleIntC = []float64{-1.7097178377177223e-06, -4.2059291475797306e-07, 1.9816644401936497e-06, -8.607099623200489e-07, -1.4251473383665473e-06, 1.7821804937944417e-06}

	badScaleNotchA = []float64{
		1.092254319223823e+07, -9.433811160314137e+06, 3.89484882389886e+06, -1.3012935693982365e+07, 2.137645536643469e+07,
		1.15294764528245e+07, -9.958030970591629e+06, 4.11126571131265e+06, -1.3736034038881391e+07, 2.2564292488971718e+07,
		-4.964098449758366e+06, 4.287447920878318e+06, -1.7701683193508307e+06, 5.9141186950771585e+06, -9.715171876618419e+06,
		-6.550476018827282, 5.657588194902013, -1.2729162239566958, 9.090270243284742, -11.7530426289037,
		-0.543819527008143, 0.46969211509445113, -0.10567731212869187, 0.6056800490364861, 0.1658576421872085,
	}
	badScaleNotchB = []float64{-0.5238867210621593, 0.6806081625683271, 0.8880104935408425, -0.7410232161870921, 0.4487866359562142}
	badScaleNotchC = []float64{-8.962443382072952e-06, 4.788573971855422e-06, 5.866251371562384e-06, -8.581569701905733e-06, -3.175874676866163e-07}
	badScaleNotchD = 73.66622400519694
)

func badScaleSystem(t *testing.T, a, b, c []float64, d float64) *System {
	t.Helper()
	n := len(b)
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, 1, slices.Clone(b)), mat.NewDense(1, n, slices.Clone(c)), mat.NewDense(1, 1, []float64{d}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// badScaleUlp moves about two thirds of the entries of a by one ulp, up or
// down by trial; trial 0 is a itself.
func badScaleUlp(a []float64, trial int) []float64 {
	a = slices.Clone(a)
	if trial == 0 {
		return a
	}
	for i := range a {
		switch (i*7 + trial) % 3 {
		case 0:
			a[i] = math.Nextafter(a[i], math.Inf(1))
		case 1:
			a[i] = math.Nextafter(a[i], math.Inf(-1))
		}
	}
	return a
}

// badScaleOracle is c(jωI − A)⁻¹b + d by partially pivoted Gaussian
// elimination in 300-bit arithmetic on the coefficients as given.
func badScaleOracle(a, b, c []float64, d, w float64) complex128 {
	type bc struct{ re, im *big.Float }
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	z := func() *big.Float { return new(big.Float).SetPrec(300) }
	sub := func(x, y bc) bc { return bc{z().Sub(x.re, y.re), z().Sub(x.im, y.im)} }
	mul := func(x, y bc) bc {
		return bc{z().Sub(z().Mul(x.re, y.re), z().Mul(x.im, y.im)), z().Add(z().Mul(x.re, y.im), z().Mul(x.im, y.re))}
	}
	div := func(x, y bc) bc {
		den := z().Add(z().Mul(y.re, y.re), z().Mul(y.im, y.im))
		num := mul(x, bc{y.re, z().Neg(y.im)})
		return bc{z().Quo(num.re, den), z().Quo(num.im, den)}
	}
	abs := func(x bc) float64 {
		re, _ := x.re.Float64()
		im, _ := x.im.Float64()
		return math.Hypot(re, im)
	}
	n := len(b)
	m := make([][]bc, n)
	for i := range n {
		m[i] = make([]bc, n+1)
		for j := range n {
			m[i][j] = bc{nf(-a[i*n+j]), nf(0)}
		}
		m[i][i].im = nf(w)
		m[i][n] = bc{nf(b[i]), nf(0)}
	}
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if abs(m[i][k]) > abs(m[p][k]) {
				p = i
			}
		}
		m[k], m[p] = m[p], m[k]
		for i := k + 1; i < n; i++ {
			f := div(m[i][k], m[k][k])
			for j := k; j <= n; j++ {
				m[i][j] = sub(m[i][j], mul(f, m[k][j]))
			}
		}
	}
	x := make([]bc, n)
	for i := n - 1; i >= 0; i-- {
		s := m[i][n]
		for j := i + 1; j < n; j++ {
			s = sub(s, mul(m[i][j], x[j]))
		}
		x[i] = div(s, m[i][i])
	}
	r := bc{nf(d), nf(0)}
	for i := range n {
		r = sub(r, mul(bc{nf(-c[i]), nf(0)}, x[i]))
	}
	re, _ := r.re.Float64()
	im, _ := r.im.Float64()
	return complex(re, im)
}

// badScaleOracleCrossing bisects the oracle's |L| = g on [lo, hi].
func badScaleOracleCrossing(a, b, c []float64, d, g, lo, hi float64) float64 {
	f := func(w float64) float64 { return cmplx.Abs(badScaleOracle(a, b, c, d, w)) - g }
	flo := f(lo)
	for range 60 {
		mid := (lo + hi) / 2
		if fm := f(mid); fm*flo <= 0 {
			hi = mid
		} else {
			lo, flo = mid, fm
		}
	}
	return (lo + hi) / 2
}

// An integrator loop with a slow pole at −7e-8 and fast poles up to −6e4
// crosses |L| = 1 once, at ω ≈ 8.108e-6, robustly under 1-ulp changes of A.
// The pencil lost the crossing in most perturbations, and with no pencil
// root endCrossings had no outer crossing to scan below.
func TestAllMarginBadlyScaledIntegratorCrossover(t *testing.T) {
	for trial := range 6 {
		a := badScaleUlp(badScaleIntA, trial)
		sys := badScaleSystem(t, a, badScaleIntB, badScaleIntC, 0)
		res, err := AllMargin(sys)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		want := badScaleOracleCrossing(a, badScaleIntB, badScaleIntC, 0, 1, 1e-6, 1e-5)
		if len(res.GainCrossFreqs) != 1 || math.Abs(res.GainCrossFreqs[0]-want) > 1e-6*want {
			t.Fatalf("trial %d: gain crossings %v, oracle %.10g", trial, res.GainCrossFreqs, want)
		}
		h := badScaleOracle(a, badScaleIntB, badScaleIntC, 0, want)
		if pm := phaseMarginDeg(h); math.Abs(res.PhaseMargins[0]-pm) > 1e-4 {
			t.Errorf("trial %d: PM %.8g, oracle %.8g", trial, res.PhaseMargins[0], pm)
		}
		m, err := Margin(sys)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if wc, ok := m.GainCrossover(); !ok || wc != res.GainCrossFreqs[0] || m.PhaseMargin != res.PhaseMargins[0] {
			t.Errorf("trial %d: Margin crossover %g %v PM %g", trial, wc, ok, m.PhaseMargin)
		}
	}
}

// Beside an unstable pair 1e-6 ± j1.00000004 and a real pole at 1, |L|
// rises from 73.67 to 167 and dips to 31.8, crossing the −3 dB level 52.15
// at ω ≈ 1.0000012 and 1.0000060 whatever 1-ulp change of A. The double
// solve snapped every ω within 1.2e-3 of the pair to +Inf, and the pencil
// roots fall up to 5e-6 off, both sides of the crossings.
func TestBandwidthBadlyScaledNotch(t *testing.T) {
	g := badScaleNotchD * math.Pow(10, -3.0/20)
	for trial := range 6 {
		a := badScaleUlp(badScaleNotchA, trial)
		sys := badScaleSystem(t, a, badScaleNotchB, badScaleNotchC, badScaleNotchD)
		bw, err := Bandwidth(sys, -3)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		w1 := badScaleOracleCrossing(a, badScaleNotchB, badScaleNotchC, badScaleNotchD, g, 1.000001, 1.0000016)
		w2 := badScaleOracleCrossing(a, badScaleNotchB, badScaleNotchC, badScaleNotchD, g, 1.0000055, 1.0000066)
		if math.Abs(bw-w1) > 1e-9 {
			t.Fatalf("trial %d: bandwidth %.12g, oracle %.12g", trial, bw, w1)
		}
		loop, err := newRationalLoop(sys)
		if err != nil {
			t.Fatal(err)
		}
		ws, err := loop.gainCrossings(g)
		if err != nil {
			t.Fatal(err)
		}
		if len(ws) != 2 || math.Abs(ws[0]-w1) > 1e-9 || math.Abs(ws[1]-w2) > 1e-9 {
			t.Errorf("trial %d: crossings %.12g, oracle [%.12g %.12g]", trial, ws, w1, w2)
		}
	}
}
