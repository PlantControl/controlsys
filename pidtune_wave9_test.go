package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

var pidtuneAllTypes = []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}

func pidtuneRat(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }

func pidtuneRatPoly(p ...*big.Rat) []*big.Rat { return p }

func pidtuneRatMul(a, b []*big.Rat) []*big.Rat {
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

func pidtuneRatAdd(a, b []*big.Rat) []*big.Rat {
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

// pidtuneExactSchur reports whether the discrete loop of pid on num/den
// (in z, highest power first) has every closed-loop root in |z| < 1, by
// the Schur–Cohn recursion on the exact characteristic polynomial in
// 2048-bit floats: an integrator among poles within 1e-3 of z = 1 defeats
// float64 root finding.
func pidtuneExactSchur(pid *PID, num, den Poly, dt float64) bool {
	r := pidtuneRat
	rdt := r(dt)
	poly := func(p Poly) []*big.Rat {
		out := make([]*big.Rat, len(p))
		for i, v := range p {
			out[i] = r(v)
		}
		return out
	}
	x := pidtuneRatPoly(r(1), r(-1))
	wi, wd := r(pidFormulaWeight(pid.IFormula)), r(pidFormulaWeight(pid.DFormula))
	one := r(1)
	iNum := pidtuneRatMul(pidtuneRatPoly(new(big.Rat).Mul(r(pid.Ki), rdt)), pidtuneRatPoly(wi, new(big.Rat).Sub(one, wi)))
	lead := new(big.Rat).Add(r(pid.Tf), new(big.Rat).Mul(wd, rdt))
	dd := pidtuneRatPoly(lead, new(big.Rat).Sub(rdt, lead))
	if dd[0].Sign() == 0 {
		dd = dd[1:]
	}
	kp, kd := pidtuneRatPoly(r(pid.Kp)), pidtuneRatPoly(r(pid.Kd))
	numC, denC := kp, pidtuneRatPoly(r(1))
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = pidtuneRatAdd(pidtuneRatAdd(pidtuneRatMul(pidtuneRatMul(kp, x), dd), pidtuneRatMul(iNum, dd)), pidtuneRatMul(pidtuneRatMul(kd, x), x))
		denC = pidtuneRatMul(x, dd)
	case pid.Ki != 0:
		numC, denC = pidtuneRatAdd(pidtuneRatMul(kp, x), iNum), x
	case pid.Kd != 0:
		numC, denC = pidtuneRatAdd(pidtuneRatMul(kp, dd), pidtuneRatMul(kd, x)), dd
	}
	c := pidtuneRatAdd(pidtuneRatMul(poly(den), denC), pidtuneRatMul(poly(num), numC))
	for len(c) > 1 && c[0].Sign() == 0 {
		c = c[1:]
	}
	const prec = 2048
	f := make([]*big.Float, len(c))
	for i, v := range c {
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

// pidtuneCheckDesigns runs every family at wc = 0: an error must be
// ErrPIDTuningTargetUnattainable, a design stable by pidtuneExactSchur.
// want lists families that must return a design.
func pidtuneCheckDesigns(t *testing.T, name string, p *System, num, den Poly, dt float64, want []PidtuneType) {
	t.Helper()
	for _, f := range pidtuneAllTypes {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s %s: %v", name, f, err)
			}
			for _, w := range want {
				if w == f {
					t.Errorf("%s %s: %v", name, f, err)
				}
			}
			continue
		}
		if !pidtuneExactSchur(pid, num, den, dt) {
			t.Errorf("%s %s: %+v unstable", name, f, pid)
		}
	}
}

// TA526O review: the rolloff fallback also fired for unstable plants with a
// zero near the integrator, whose stabilizing gain the bandwidth gives; the
// capped crossover and its back-off never stabilize them, turning designs
// into errors (P "unstable" at wc 90.7 for case 1). The crossover of an
// unstable plant is its -3 dB bandwidth. I on case 1 and P on case 2 may
// error (IBWZGV): their stable designs there kept PM 0.001° (|S| 1249) and
// 1.4° (|S| 40). Oracle: Bandwidth; exact Routh test of every design.
func TestPidtuneRolloffUnstableKeepsBandwidth(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		want     []PidtuneType
	}{
		{Poly{0.27517413517840106, 3.6669924390083567e-05}, Poly{1, 6.391643798250351, -30.351456187775952}, []PidtuneType{PidtuneP, PidtunePD, PidtunePDF}},
		{Poly{3.24364602965127, 0.0021262777284359194}, Poly{1, 13.719866434906919, 22.960516396640905, -141.22246804852216}, nil},
		{Poly{0.15352054260394388, 2.4370462844322383e-06}, Poly{1, 9.545146152266515, -3.889305115757257}, []PidtuneType{PidtuneP, PidtunePD, PidtunePDF}},
	} {
		pidtuneRolloffUnstableCase(t, c.num, c.den, c.want, true)
	}

	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(7, 9))
	for range 120 {
		z0 := math.Pow(10, -3-5*rng.Float64())
		if rng.IntN(2) == 0 {
			z0 = -z0
		}
		num, den := Poly{1, z0}, Poly{1, -(0.1 + 5*rng.Float64())}
		for range rng.IntN(3) {
			den = den.Mul(Poly{1, 0.2 + 10*rng.Float64()})
		}
		if rng.IntN(3) == 0 {
			num = num.Mul(Poly{1, 0.5 + 5*rng.Float64()})
		}
		if rng.IntN(4) == 0 {
			den = den.Mul(Poly{1, -(0.1 + 3*rng.Float64())})
		}
		num = num.Mul(Poly{math.Pow(10, -2+3*rng.Float64())})
		if len(num) > len(den) {
			continue
		}
		pidtuneRolloffUnstableCase(t, num, den, nil, z0 > 0)
	}
}

// pidtuneRolloffUnstableCase checks the crossover, and with designs every
// design: a real zero in s > 0 with an odd count of real unstable poles
// above it admits no stable stabilizing controller (parity interlacing),
// which Pidtune does not detect, so the sweep skips designs for such zeros.
func pidtuneRolloffUnstableCase(t *testing.T, num, den Poly, want []PidtuneType, designs bool) {
	t.Helper()
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	name := fmt.Sprintf("%v/%v", num, den)
	if am, err := AllMargin(p); err != nil || len(am.GainCrossFreqs) > 0 {
		return
	}
	bw, err := Bandwidth(p, -3)
	if err != nil || !(bw > 0) || math.IsInf(bw, 0) {
		return
	}
	if wc, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || fromPole || wc != bw {
		t.Errorf("%s: crossover %g (pole %v), %v; want bandwidth %g", name, wc, fromPole, err, bw)
	}
	if !designs {
		return
	}
	for _, f := range pidtuneAllTypes {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s %s: %v", name, f, err)
			}
			for _, w := range want {
				if w == f {
					t.Errorf("%s %s: %v", name, f, err)
				}
			}
			continue
		}
		if !pidtuneExactHurwitz(pid, num, den) {
			t.Errorf("%s %s: %+v closed loop not Hurwitz", name, f, pid)
		}
	}
}

// FPMBHI review: an ill-conditioned realization of (z−1)·∏(z−pᵢ), pᵢ near
// 1, moves the integrator pole 1e-10..1e-8 off z = 1, beyond the simple
// cluster radius 100·ε, while DCGain stays infinite; with |G| < 1 the
// displaced integrator was taken as the slowest pole, giving wc ~1e-9 and
// "zero or infinite gain" or "singular complex matrix" errors. Oracle: the
// crossover is half the slowest generated pole's |ln p|/Dt, capped at
// 0.1π/Dt, to 5% (clustered poles round, 3% on amd64); every
// design stable by the exact Schur–Cohn test. Integrators displaced beyond
// pidtuneClusterRadius(2, 1) are TestPidtuneIsolatedDisplacedIntegrator's.
func TestPidtuneDisplacedDiscreteIntegrator(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(5, 8))
	kept, displaced := 0, 0
	for range 800 {
		dt := []float64{0.01, 0.1, 1}[rng.IntN(3)]
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
		dc, err := p.DCGain()
		if err != nil || !math.IsInf(dc.At(0, 0), 0) {
			continue
		}
		poles, err := p.Poles()
		if err != nil {
			t.Fatal(err)
		}
		near := math.Inf(1)
		for _, r := range poles {
			near = math.Min(near, cmplx.Abs(r-1))
		}
		if near > pidtuneClusterRadius(2, 1) {
			continue
		}
		kept++
		if near > 100*0x1p-52 {
			displaced++
		}
		name := fmt.Sprintf("%v/%v dt=%g", num, den, dt)
		want := math.Min(0.5*slow, 0.1*math.Pi/dt)
		wc, err := plantCrossoverFreq(p)
		if err != nil || math.Abs(wc-want) > 5e-2*want {
			t.Errorf("%s: crossover %g, %v; want %g", name, wc, err, want)
			continue
		}
		pidtuneCheckDesigns(t, name, p, num, den, dt, nil)
	}
	if kept < 20 || displaced < 10 {
		t.Fatalf("sweep kept %d plants, %d displaced", kept, displaced)
	}
}

// Wave 11 merge: next to an integrator displaced 1.3e-7 inside z = 1 (case
// 0) FreqResponse and DCGain at DC are finite (0.2084, exact for the
// stored coefficients), so a pidTuningDCSign taking that DC sign sampled 3
// decades below wc, where the loop response fails, and every family
// errored; Pidtune now recognises the integrator itself. Case 1's computed
// pole lies 3.4e-5 outside z = 1 (the exact root of the stored denominator
// is 3.6e-7 inside).
// Oracle: exact Schur–Cohn test of every design.
func TestPidtuneDisplacedIntegratorDCSign(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		dt       float64
		want     []PidtuneType
	}{
		{Poly{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}, Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}, 0.01, []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}},
		{Poly{1.8748015424080903e-16}, Poly{1, -6.656527066165689, 18.974082402962605, -30.020939359334676, 28.473509438245436, -16.187809823253666, 5.1075991528389695, -0.6899147452929796}, 0.1, []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePID, PidtunePIDF}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		pidtuneCheckDesigns(t, fmt.Sprintf("%v/%v", c.num, c.den), p, c.num, c.den, c.dt, c.want)
	}
}

// SJ5QTW/TA526O review: unstable plants with a zero near the integrator keep
// the -3 dB bandwidth crossover, where the PDF lag floor taken when only the
// mirrored sign lagged gave PDF Kp −1.8e9 whose closed loop has a root near
// 1e-10 in s > 0 (constant term −0.63, leading Tf > 0), and stable PDFs
// crossing 0 dB at 1.25e-7 with PM 6.4° and |S| 160, where 6a728b2 errored.
// Case 1's zero at +1.4e-9 lies below its unstable pole 0.213, so no stable
// controller stabilizes it (parity interlacing). Oracle: Bandwidth; exact
// Routh test, dense margins and |S| on [1e-10, 1e7]. P and PD, which never
// take the lag floor, are left out: as in 6a728b2 they return designs the
// pole test misses roots of or with |S| up to 800.
func TestPidtuneUnstableBandwidthLagFloor(t *testing.T) {
	for _, c := range []struct{ num, den Poly }{
		{Poly{-0.12851687378089954, 1.827217025221083e-10}, Poly{1, 1.1881765967408056, -0.29841510512323605}},
		{Poly{-0.011742895229596644, -8.703213959160969e-13}, Poly{1, -1.2883025580770138, -206.68825123865938}},
		{Poly{-0.08096947227011618, 2.849829949486798e-11}, Poly{1, 18.28073944410921, -2.193118184378895}},
		{Poly{-0.08892180697264399, -1.3561740229044294e-13}, Poly{1, 5.04214482512001, -79.10168669801203}},
		{Poly{-0.020532597597922114, 5.863958650611105e-14}, Poly{1, 0.6789049920746436, -0.10457076289624506}},
		{Poly{-0.017989551265351702, 2.0099607239269687e-08}, Poly{1, -15.376997209772385, 7.2747594545148075}},
		{Poly{-2.1243501066617125, -0.3754289703529278, 1.2534782657779512e-05}, Poly{1, -8.883497559326386, -57.23481627365082, 89.10956726077973}},
		{Poly{-0.5496489873828221, -0.3110705188753598, 8.219087877604514e-06}, Poly{1, 3.0604165976957916, -0.9814565587506462, 0.0745769201632338}},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		bw, err := Bandwidth(p, -3)
		if err != nil {
			t.Fatal(err)
		}
		if wc, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || fromPole || wc != bw {
			t.Errorf("%v/%v: crossover %g (pole %v), %v; want bandwidth %g", c.num, c.den, wc, fromPole, err, bw)
		}
		for _, f := range []PidtuneType{PidtuneI, PidtunePI, PidtunePDF, PidtunePID, PidtunePIDF} {
			name := fmt.Sprintf("%s %v/%v", f, c.num, c.den)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%s: %v", name, err)
				}
				continue
			}
			pidtuneRolloffRobust(t, name, f, pid, c.num, c.den)
		}
	}
}

// TA526O review: a stable plant with a zero near the integrator and a
// continuous input delay took the rolloff crossover, where the delay's lag
// leaves a PDF needing phase lag ("unattainable" at wc 28.7 for case 1);
// the delay-capped bandwidth has a robust PDF (6a728b2: PM 60°, |S| 2.88).
// Oracle: Bandwidth capped at (180°−PM)/(2τ); Nyquist winding, dense
// margins and |S| of the directly evaluated delayed loop.
func TestPidtuneRolloffDelayKeepsBandwidth(t *testing.T) {
	check := func(num, den Poly, tau float64, want bool) {
		t.Helper()
		name := fmt.Sprintf("%v/%v τ=%g", num, den, tau)
		p := pidtuneBackoffPlant(t, num, den, 0, tau)
		bw, err := Bandwidth(pidtuneBackoffPlant(t, num, den, 0, 0), -3)
		if err != nil {
			t.Fatal(err)
		}
		wantWc := math.Min(bw, (180-60)*math.Pi/180/(2*tau))
		if wc, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || fromPole || math.Abs(wc-wantWc) > 1e-9*wantWc {
			t.Errorf("%s: crossover %g (pole %v), %v; want %g", name, wc, fromPole, err, wantWc)
		}
		pid, err := Pidtune(p, PidtunePDF, 0, nil)
		if err != nil {
			if want || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			return
		}
		if stable, ok := pidtuneWindingStable(pid, num, den, tau); !ok || !stable {
			t.Errorf("%s: %+v not stable by winding (ok %v)", name, pid, ok)
			return
		}
		if !want {
			return
		}
		pm, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, num, den, 0, tau), 1e-10, 1e7, 1.001)
		if pm < 58.5 || ms > 4.05 {
			t.Errorf("%s: %+v phase margin %.3g, |S| %.3g", name, pid, pm, ms)
		}
	}
	for _, c := range []struct {
		num, den Poly
		tau      float64
	}{
		{Poly{-0.10741411672891396, -3.230296708983251e-06}, Poly{1, 3.663417582210873, 1.7231714426773892}, 0.013688309558579217},
		{Poly{0.015035095437983783, 5.402994672543609e-07}, Poly{1, 0.4181878551368774, 0.03350934017285575}, 0.11529063838884701},
		{Poly{0.42181178954733345, 1.6345035762368298e-09}, Poly{1, 1.283524811348602, 0.21980553487118143}, 0.022145359893947905},
		{Poly{-0.7683877383500903, -1.793530740229162e-10}, Poly{1, 5.039155455719128, 1.2593858413801469}, 0.014311486405501969},
	} {
		check(c.num, c.den, c.tau, true)
	}

	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(11, 3))
	rolloff := 0
	for range 60 {
		z0 := math.Pow(10, -4-6*rng.Float64())
		if rng.IntN(2) == 0 {
			z0 = -z0
		}
		a, b := 0.1+5*rng.Float64(), 0.1+5*rng.Float64()
		k := math.Pow(10, -2+2*rng.Float64())
		if rng.IntN(2) == 0 {
			k = -k
		}
		num, den := Poly{k, k * z0}, Poly{1, a + b, a * b}
		free := pidtuneBackoffPlant(t, num, den, 0, 0)
		bw, err := Bandwidth(free, -3)
		if err != nil || !(bw > 0) || math.IsInf(bw, 0) {
			continue
		}
		if _, ok := pidtuneRolloffCrossover(free, bw); !ok {
			continue
		}
		rolloff++
		check(num, den, math.Pow(10, -2+1.5*rng.Float64()), false)
	}
	if rolloff < 20 {
		t.Fatalf("sweep kept %d rolloff plants", rolloff)
	}
}
