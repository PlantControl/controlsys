package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
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

// GAHTRN: beside the notch pair 9.9e-7 ± j1.00000004 the double solve fails
// and sisoEval.at, with its extended fallback, snapped every ω within
// 1e-5·rho ≈ 1.4e-5 of the pair to +Inf: rho, the distance to the real pole
// at 1, skips the pair as within the cluster tolerance 1e-6·‖poles‖ ≈ 0.8.
// The response there is finite, 30–170. Off DC at takes the double-double
// response as FreqResponse does. Oracle: 300-bit solve of the stored data;
// the double-double solve's error is about 2⁻¹⁰⁴·κ with κ ≲ 1e14 here, so
// 1e-12 relative leaves headroom over the measured ~1e-16.
func TestSISOEvalBadlyScaledResonanceFinite(t *testing.T) {
	for trial := range 3 {
		a := badScaleUlp(badScaleNotchA, trial)
		sys := badScaleSystem(t, a, badScaleNotchB, badScaleNotchC, badScaleNotchD)
		e, err := newSISOEval(sys)
		if err != nil {
			t.Fatal(err)
		}
		e.extended = true
		eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
		failed := 0
		const points = 800
		for i := range points + 1 {
			w := 0.99998 + 4e-5*float64(i)/points
			if eval(pointAt(complex(0, w)), []complex128{0}) != nil {
				failed++
			}
			e.err = nil
			h := e.at(w)
			want := badScaleOracle(a, badScaleNotchB, badScaleNotchC, badScaleNotchD, w)
			if e.err != nil || cmplx.Abs(h-want) > 1e-12*cmplx.Abs(want) {
				t.Fatalf("trial %d: at(%.10g) = %v, %v; oracle %v", trial, w, h, e.err, want)
			}
		}
		if failed < points/2 {
			t.Fatalf("trial %d: double solve failed at %d of %d ω; case no longer exercises the pole limit", trial, failed, points+1)
		}
	}
}

// badScaleResonance is a realization T·A₀·T⁻¹ of a pair σ ± j, a real pole
// at 1, and poles near −150 and −1.5e5, with T = diag(10^k)·G₁·G₂ for k in
// [−3, 3], G₁ Gaussian and G₂ Gaussian with rows graded by 10^{-ci/4}: an
// ill-conditioned, non-diagonal similarity that balancing cannot undo.
func badScaleResonance(t *testing.T, rng *rand.Rand, sigma, c float64) *System {
	t.Helper()
	const n = 5
	a0 := mat.NewDense(n, n, []float64{
		sigma, 1, 0, 0, 0,
		-1, sigma, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, -100 * (1 + rng.Float64()), 0,
		0, 0, 0, 0, -1e5 * (1 + rng.Float64()),
	})
	g1, g2 := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g1.Set(i, j, rng.NormFloat64())
			g2.Set(i, j, rng.NormFloat64()*math.Pow(10, -c*float64(i)/4))
		}
	}
	var tm, ti, tmp, a, b, cm mat.Dense
	tm.Mul(g1, g2)
	for i := range n {
		k := math.Pow(10, -3+6*rng.Float64())
		for j := range n {
			tm.Set(i, j, tm.At(i, j)*k)
		}
	}
	if err := ti.Inverse(&tm); err != nil {
		t.Fatal(err)
	}
	tmp.Mul(&tm, a0)
	a.Mul(&tmp, &ti)
	b0, c0 := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		b0.Set(i, 0, rng.NormFloat64())
		c0.Set(0, i, 1e-4*rng.NormFloat64())
	}
	b.Mul(&tm, b0)
	cm.Mul(c0, &ti)
	sys, err := New(&a, &b, &cm, mat.NewDense(1, 1, []float64{10 * rng.NormFloat64()}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// badScaleResonanceFreq is the imaginary part of the computed upper pole
// of the pair, the ω of closest approach.
func badScaleResonanceFreq(t *testing.T, sys *System) float64 {
	t.Helper()
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range poles {
		if imag(p) > 0 {
			return imag(p)
		}
	}
	t.Fatalf("no complex pole in %v", poles)
	return 0
}

// GAHTRN: a seeded family of badly scaled resonant realizations, pair
// damping |σ| ≤ 1e-6, ‖A‖ 1e9–1e15. Across ω within 2e-5 of the pair,
// inside the old snap band 1.4e-5, sisoEval.at must be finite and match
// the oracle. Off the double solve's failures it is the double solve
// itself, accurate to about eps·κ ≲ 1e-10 here; within them the
// double-double response, ~1e-16.
func TestSISOEvalBadlyScaledResonanceFamily(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x6a48, 0x7452))
	exercised := 0
	const cases = 16
	for k := range cases {
		sys := badScaleResonance(t, rng, 1e-6*(2*rng.Float64()-1), 4)
		wp := badScaleResonanceFreq(t, sys)
		e, err := newSISOEval(sys)
		if err != nil {
			t.Fatal(err)
		}
		e.extended = true
		eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
		failed := 0
		for i := -20; i <= 20; i++ {
			w := wp + 1e-6*float64(i)
			if eval(pointAt(complex(0, w)), []complex128{0}) != nil {
				failed++
			}
			e.err = nil
			h := e.at(w)
			want := frExactResponse(sys, frBigC(complex(0, w)))[0]
			if e.err != nil || cmplx.Abs(h-want) > 1e-9*cmplx.Abs(want) {
				t.Errorf("case %d: at(%.12g) = %v, %v; oracle %v", k, w, h, e.err, want)
			}
		}
		if failed > 0 {
			exercised++
		}
	}
	if exercised < cases/2 {
		t.Errorf("%d of %d cases exercise the pole limit; want ≥ %d", exercised, cases, cases/2)
	}
}

// GAHTRN guard: the snap still holds at a genuine pole. sisoEval.at reads
// +Inf at an undamped axis pair, at a double integrator at ω = 0, and next
// to an origin cluster as before; at an undamped pair of a badly scaled
// realization it agrees with FreqResponse's pole limit.
func TestSISOEvalAtPoleSnapsInf(t *testing.T) {
	for _, c := range []struct {
		name string
		den  []float64
		w    float64
	}{
		{"(s²+1)(s+2)", []float64{1, 2, 1, 2}, 1},
		{"s²(s+1)", []float64{1, 1, 0, 0}, 0},
	} {
		sys := makePlant(t, []float64{1, 3, 1}, c.den)
		if _, raw, _ := poleLimitAt(sys, c.w); raw == nil {
			t.Fatalf("%s: double solve succeeded; case no longer exercises the pole limit", c.name)
		}
		for _, ext := range []bool{false, true} {
			e, err := newSISOEval(sys)
			if err != nil {
				t.Fatal(err)
			}
			e.extended = ext
			if h := e.at(c.w); e.err != nil || !cmplx.IsInf(h) {
				t.Errorf("%s extended %v: at(%g) = %v, %v; want +Inf", c.name, ext, c.w, h, e.err)
			}
		}
	}
	rng := rand.New(rand.NewPCG(0x6a48, 0x1f))
	snapped, compared := 0, 0
	for k := range 8 {
		sys := badScaleResonance(t, rng, 0, 4)
		wp := badScaleResonanceFreq(t, sys)
		pt := pointAt(complex(0, wp))
		eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
		if eval(pt, []complex128{0}) == nil {
			continue
		}
		want := []complex128{0}
		if err := evalWithPoleLimit(eval, sys, pt, want); err != nil {
			t.Fatal(err)
		}
		e, err := newSISOEval(sys)
		if err != nil {
			t.Fatal(err)
		}
		e.extended = true
		h := e.at(wp)
		if e.err != nil || cmplx.IsInf(h) != cmplx.IsInf(want[0]) || !cmplx.IsInf(h) && cmplx.Abs(h-want[0]) > 1e-12*cmplx.Abs(want[0]) {
			t.Errorf("case %d: at(%.17g) = %v, %v; FreqResponse pole limit %v", k, wp, h, e.err, want[0])
		}
		compared++
		if cmplx.IsInf(h) {
			snapped++
		}
	}
	if compared < 4 {
		t.Errorf("%d of 8 undamped cases exercise the pole limit; want ≥ 4", compared)
	}
	t.Logf("%d of %d badly scaled undamped pairs snap to +Inf", snapped, compared)
}

// UG5V3W: without the extended fallback sisoEval.at takes the pole limit
// in double, whose samples beside a badly scaled resonance straddle the
// pair and fit a pole that is not there: it read +Inf with no error. Case
// 9 of the GAHTRN family (PCG 0x6a48, 0x7452), frozen as computed on
// arm64. Off DC each reading must be finite and match the oracle, or fail.
func TestSISOEvalAtOffDCPoleLimitUnverified(t *testing.T) {
	a := mat.NewDense(5, 5, []float64{
		4.1244111051501073e+06, -3.3430528969799967e+09, -3.195564281908565e+08, -19190.11090663089, -4.474318187921283e+06,
		-2217.6681604315468, 1.8007139021469527e+06, 171253.9570382416, 9.672745853334929, 2396.9507662035653,
		1892.241213506034, -2.727198425767869e+06, 66794.36006332611, 231.3249404232146, 1235.638981686064,
		-8.987127649113962e+07, 7.308258505731027e+10, 6.92073008922217e+09, 370324.2949348788, 9.684047260627452e+07,
		5.736655166685106e+06, -4.56044793022558e+09, -4.6045938060693806e+08, -44677.21082229327, -6.4696288449062975e+06,
	})
	b := mat.NewDense(5, 1, []float64{-7.114047454508141, 0.001889007727279433, -0.01636711763289042, 124.24391342103425, -7.33632168041491})
	c := mat.NewDense(1, 5, []float64{-0.44998107945742327, 405.3726760087558, 27.520312969649293, -0.006412855848687369, 0.3707311485010806})
	sys, err := New(a, b, c, mat.NewDense(1, 1, []float64{-7.167117819816671}), 0)
	if err != nil {
		t.Fatal(err)
	}
	const wp = 0.9999995290676637
	e, err := newSISOEval(sys)
	if err != nil {
		t.Fatal(err)
	}
	eval := newFrequencyEvaluator(sys).pointSolver(math.MaxInt).evalInto
	failed, rejected := 0, 0
	for i := -20; i <= 20; i++ {
		w := wp + 1e-6*float64(i)
		if eval(pointAt(complex(0, w)), []complex128{0}) != nil {
			failed++
		}
		e.err = nil
		h := e.at(w)
		if e.err != nil {
			if !errors.Is(e.err, ErrAccuracyUnverified) && !errors.Is(e.err, ErrSingularTransform) {
				t.Errorf("at(%.17g): error %v", w, e.err)
			}
			rejected++
			continue
		}
		want := frExactResponse(sys, frBigC(complex(0, w)))[0]
		if cmplx.Abs(h-want) > 1e-9*cmplx.Abs(want) {
			t.Errorf("at(%.17g) = %v, nil; oracle %v", w, h, want)
		}
	}
	if failed == 0 {
		t.Fatal("double solve resolved every ω; case no longer exercises the pole limit")
	}
	t.Logf("double solve failed at %d of 41 ω, at rejected %d", failed, rejected)
}
