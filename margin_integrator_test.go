package controlsys

import (
	"context"
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

// DSRJ7M: on lead multiple integrators the L − L~ pencil root at 0 lands
// at ω ~ 1e-8 and L(0) is a pole, where the state-space solve is singular;
// the sticky evaluation error aborted AllMargin. MATLAB returns the margins.
// Oracle: −(a·jω+b)/ω² is never real, so no phase crossover, and |L| = 1
// at ω² = (a² + √(a⁴+4b²))/2.
func TestAllMarginLeadDoubleIntegratorPencilRootAtZero(t *testing.T) {
	a, b := -0.032012382534863985, -0.052223340965159556
	res, err := AllMargin(makePlant(t, []float64{a, b}, []float64{1, 0, 0}))
	if err != nil {
		t.Fatal(err)
	}
	wc := math.Sqrt((a*a + math.Sqrt(a*a*a*a+4*b*b)) / 2)
	l := complex(b, a*wc) / complex(-wc*wc, 0)
	pm := 180 + cmplx.Phase(l)*180/math.Pi - 360
	if len(res.PhaseCrossFreqs) != 0 || len(res.GainCrossFreqs) != 1 ||
		math.Abs(res.GainCrossFreqs[0]-wc) > 1e-9*wc || math.Abs(res.PhaseMargins[0]-pm) > 1e-7 {
		t.Fatalf("got %+v, want wc %g pm %g", *res, wc, pm)
	}
}

// Seeded k(s+z)/sⁿ and k(z−c)/(z−1)ⁿ, n = 2, 3: AllMargin succeeds and its
// crossovers match a dense log grid of the ratio, with z−1 = 2j·sin(θ/2)·e^{jθ/2}
// so the discrete oracle keeps its accuracy near z = 1.
func TestAllMarginLeadIntegratorSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for trial := range 400 {
		k := (2*rng.Float64() - 1) * math.Pow(10, 4*rng.Float64()-2)
		den := []float64{1, 0, 0}
		if trial%3 == 0 {
			den = []float64{1, 0, 0, 0}
		}
		var num []float64
		dt := 0.0
		var at func(float64) complex128
		if trial%2 == 0 {
			num = []float64{k, k * math.Pow(10, 4*rng.Float64()-2)}
			at = func(w float64) complex128 { s := complex(0, w); return pidPolyAt(num, s) / pidPolyAt(den, s) }
		} else {
			dt = 0.01 * math.Pow(10, rng.Float64())
			c1 := math.Pow(10, -3*rng.Float64())
			num = []float64{k, -k * (1 - c1)}
			n := len(den) - 1
			if n == 2 {
				den = []float64{1, -2, 1}
			} else {
				den = []float64{1, -3, 3, -1}
			}
			at = func(w float64) complex128 {
				zm1 := complex(0, 2*math.Sin(w*dt/2)) * cmplx.Exp(complex(0, w*dt/2))
				return complex(k, 0) * (zm1 + complex(c1, 0)) / cmplx.Pow(zm1, complex(float64(n), 0))
			}
		}
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		all, err := AllMargin(res.Sys)
		if err != nil {
			t.Errorf("trial %d %v/%v dt %g: %v", trial, num, den, dt, err)
			continue
		}
		lo, hi := 1e-6, 1e6
		if dt > 0 {
			hi = math.Pi / dt * (1 - 1e-9)
		}
		gain, phase := denseMarginCrossings(at, lo, hi)
		if !crossingsMatch(all.GainCrossFreqs, gain) {
			t.Errorf("trial %d %v/%v dt %g: gain crossings %v, dense %v", trial, num, den, dt, all.GainCrossFreqs, gain)
		}
		var inner []float64
		for _, w := range all.PhaseCrossFreqs {
			if dt == 0 || w < hi {
				inner = append(inner, w)
			}
		}
		if !crossingsMatch(inner, phase) {
			t.Errorf("trial %d %v/%v dt %g: phase crossings %v, dense %v", trial, num, den, dt, all.PhaseCrossFreqs, phase)
		}
	}
}

func denseMarginCrossings(at func(float64) complex128, lo, hi float64) (gain, phase []float64) {
	const n = 40000
	step := math.Log(hi/lo) / n
	h0 := at(lo)
	for i := 1; i <= n; i++ {
		w0, w1 := lo*math.Exp(float64(i-1)*step), lo*math.Exp(float64(i)*step)
		h1 := at(w1)
		if (cmplx.Abs(h0)-1)*(cmplx.Abs(h1)-1) < 0 {
			gain = append(gain, math.Sqrt(w0*w1))
		}
		if imag(h0)*imag(h1) < 0 && real(h0)+real(h1) < 0 {
			phase = append(phase, math.Sqrt(w0*w1))
		}
		h0 = h1
	}
	return gain, phase
}

func crossingsMatch(got, dense []float64) bool {
	if len(got) != len(dense) {
		return false
	}
	for i := range got {
		if math.Abs(math.Log(got[i]/dense[i])) > 1e-3 {
			return false
		}
	}
	return true
}

// k/((z−1)²·z^d) has phase −180° − (d+1)θ, so phase crossovers at
// θ = 2πm/(d+1) with |L| = k/(4 sin²(θ/2)); the pole at z = 1 must not drop
// a crossover at twice another, where the phase error is 0 at half of it.
func TestAllMarginDiscreteDoubleIntegratorDelayHarmonics(t *testing.T) {
	const k = 0.01
	for _, dt := range []float64{1, 0.1} {
		for d := 1; d <= 8; d++ {
			for _, asDelay := range []bool{false, true} {
				den := []float64{1, -2, 1}
				if !asDelay {
					den = append(den, make([]float64, d)...)
				}
				r, err := (&TransferFunc{Num: [][][]float64{{{k}}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
				if err != nil {
					t.Fatal(err)
				}
				sys := r.Sys
				if asDelay {
					sys.InputDelay = []float64{float64(d)}
				}
				res, err := AllMargin(sys)
				if err != nil {
					t.Fatalf("dt %g d %d delay %v: %v", dt, d, asDelay, err)
				}
				var ws, gms []float64
				for m := 1; 2*m <= d+1; m++ {
					th := 2 * math.Pi * float64(m) / float64(d+1)
					s := math.Sin(th / 2)
					ws = append(ws, th/dt)
					gms = append(gms, -20*math.Log10(k/(4*s*s)))
				}
				ok := len(res.PhaseCrossFreqs) == len(ws)
				for i := 0; ok && i < len(ws); i++ {
					ok = math.Abs(res.PhaseCrossFreqs[i]-ws[i]) <= 1e-9*ws[i] && math.Abs(res.GainMargins[i]-gms[i]) <= 1e-9
				}
				if !ok {
					t.Errorf("dt %g d %d delay %v: wpc %v gm %v, want %v %v", dt, d, asDelay, res.PhaseCrossFreqs, res.GainMargins, ws, gms)
				}
			}
		}
	}
}

// JPZZRT: for k/(z∓1)ⁿ with tiny k the pencil roots near the pole cluster
// at z = ±1 lose accuracy beyond the polish bracket and the gain crossover
// vanished. Oracle: |e^{jθ} − 1| = 2·sin(θ/2), |e^{jθ} + 1| = 2·cos(θ/2), so
// |L| = 1 at sin(θ/2) = k^{1/n}/2, or θ = π minus that, with angle
// −n(θ+π)/2, or −nθ/2. Next to z = −1 the response itself carries relative
// rounding of order eps/|e^{jθ}+1|ⁿ, which bounds ω to 1e-9·θ and the phase
// margin to 1e-4°.
func TestAllMarginTinyGainPoleCluster(t *testing.T) {
	cases := []struct {
		s float64
		n int
		k float64
	}{
		{-1, 1, 1e-14}, {-1, 2, 3.1623e-8}, {-1, 3, 3.1623e-8}, {-1, 3, 1e-10},
		{-1, 4, 1e-7}, {-1, 4, 3.1623e-8}, {-1, 4, 1e-10},
		{1, 1, 3.1623e-8}, {1, 1, 1e-10}, {1, 1, 1e-14}, {1, 2, 1e-10}, {1, 2, 1e-14},
		{1, 3, 1e-10}, {1, 3, 1e-14}, {1, 4, 1e-10},
	}
	for _, c := range cases {
		den := []float64{1}
		for range c.n {
			next := make([]float64, len(den)+1)
			for i, v := range den {
				next[i] += v
				next[i+1] += c.s * v
			}
			den = next
		}
		nc := complex(float64(c.n), 0)
		half := math.Asin(math.Pow(c.k, 1/float64(c.n)) / 2)
		th, gap := 2*half, 2*half
		h := complex(c.k, 0) / cmplx.Pow(cmplx.Rect(2*math.Sin(half), (th+math.Pi)/2), nc)
		if c.s > 0 {
			th = math.Pi - 2*half
			h = complex(c.k, 0) / cmplx.Pow(cmplx.Rect(2*math.Cos(th/2), th/2), nc)
		}
		pm := phaseMarginDeg(h)
		for _, dt := range []float64{1, 0.1} {
			r, err := (&TransferFunc{Num: [][][]float64{{{c.k}}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
			if err != nil {
				t.Fatal(err)
			}
			res, err := AllMargin(r.Sys)
			if err != nil {
				t.Fatalf("s %g n %d k %g dt %g: %v", c.s, c.n, c.k, dt, err)
			}
			if len(res.GainCrossFreqs) != 1 || math.Abs(res.GainCrossFreqs[0]-th/dt) > (1e-8*gap+1e-9*th)/dt || math.Abs(res.PhaseMargins[0]-pm) > 1e-4 {
				t.Errorf("s %g n %d k %g dt %g: wgc %v pm %v, want %.17g %.10g", c.s, c.n, c.k, dt, res.GainCrossFreqs, res.PhaseMargins, th/dt, pm)
			}
		}
	}
}

// JPZZRT follow-up: next to the cluster at z = 1 the response of
// k·num/(z−1)ⁿ carries relative rounding of order eps/θⁿ, which exceeded the
// fixed tolerance endCrossings used to accept a bisected root, so correctly
// bracketed crossovers were dropped. Oracle: |e^{jθ} − 1| = 2·sin(θ/2), so
// |L(θ)| = k·|num(e^{jθ})|/(2·sin(θ/2))ⁿ is evaluated without cancellation
// and bisected for |L| = 1; ω and phase margin are held to that rounding.
func TestAllMarginTinyGainNearUnitCluster(t *testing.T) {
	cases := []struct {
		num []float64
		n   int
	}{
		{[]float64{1e-10}, 2}, {[]float64{math.Pow(10, -10.5)}, 2}, {[]float64{math.Pow(10, -9.25)}, 2},
		{[]float64{3.01e-12}, 2}, {[]float64{1e-14}, 2}, {[]float64{1.08e-9, 0.31 * 1.08e-9}, 2},
		{[]float64{1e-14}, 3}, {[]float64{1e-13}, 4}, {[]float64{math.Pow(10, -13.25)}, 4},
	}
	for _, c := range cases {
		den := []float64{1}
		for range c.n {
			next := make([]float64, len(den)+1)
			for i, v := range den {
				next[i] += v
				next[i+1] -= v
			}
			den = next
		}
		loop := func(th float64) complex128 {
			z, num := cmplx.Rect(1, th), complex(0, 0)
			for _, v := range c.num {
				num = num*z + complex(v, 0)
			}
			return num / cmplx.Pow(cmplx.Rect(2*math.Sin(th/2), (th+math.Pi)/2), complex(float64(c.n), 0))
		}
		lo, hi := 1e-12, 1e-2
		for range 200 {
			if m := math.Sqrt(lo * hi); cmplx.Abs(loop(m)) > 1 {
				lo = m
			} else {
				hi = m
			}
		}
		th := math.Sqrt(lo * hi)
		noise := 1e-15 / math.Pow(th, float64(c.n))
		pm := phaseMarginDeg(loop(th))
		for _, dt := range []float64{1, 0.1} {
			r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
			if err != nil {
				t.Fatal(err)
			}
			res, err := AllMargin(r.Sys)
			if err != nil {
				t.Fatalf("num %v n %d dt %g: %v", c.num, c.n, dt, err)
			}
			if len(res.GainCrossFreqs) != 1 || math.Abs(res.GainCrossFreqs[0]-th/dt) > max(1e-8, noise)*th/dt || math.Abs(res.PhaseMargins[0]-pm) > max(1e-4, noise*180/math.Pi) {
				t.Errorf("num %v n %d dt %g: wgc %v pm %v, want %.17g %.10g", c.num, c.n, dt, res.GainCrossFreqs, res.PhaseMargins, th/dt, pm)
			}
		}
	}
}

// 5RWUVZ: polish probed brackets up to half the gap to the next candidate,
// which next to the 4-fold pole cluster at z = −1 of k·num/((z+1)⁴·den)
// reached an ω where the solve is singular, and the sticky error failed
// AllMargin. Oracle: e^{jθ}+1 = 2·cos(θ/2)·e^{jθ/2}, so L(θ) is evaluated
// without cancellation and its crossings bisected on a grid graded towards
// Nyquist; the pole limit at Nyquist is no crossover. The response carries
// relative rounding of order eps/(2·cos(θ/2))⁴, which bounds ω and the
// margins. For k < 0 over (z+1)⁴ alone, arg L = π − 2θ meets −π only at
// the pole θ = π, so there is no phase crossover.
func TestAllMarginNyquistPoleCluster(t *testing.T) {
	cases := []struct {
		k      float64
		zs, ps []float64
		dt     float64
	}{
		{-1e-6, nil, nil, 1}, {-5.32259e-11, nil, nil, 1}, {-7.926960706503617e-11, nil, nil, 1},
		{-1.7380762773365073e-06, nil, nil, 1}, {-0.0014485035794807087, nil, nil, 0.79},
		{1e-6, nil, nil, 1}, {2.33286e-6, []float64{0.49718}, nil, 1},
		{2.919066787947241e-06, nil, []float64{-0.13565250527172176}, 0.79},
		{-0.4751559371429119, []float64{0.36215662478514177}, []float64{-0.3392596002410553}, 0.01},
		{8.016558531989637e-08, nil, []float64{0.3435098967102221}, 0.1},
		{-6.214045044554006e-06, []float64{-0.17935431128743506}, []float64{0.22517105094095477}, 0.01},
		{-0.1987764295225583, []float64{-0.19913536571637425}, []float64{0.26106995883500667}, 0.1},
	}
	poly := func(k float64, rs []float64) []float64 {
		p := []float64{k}
		for _, r := range rs {
			next := make([]float64, len(p)+1)
			for i, v := range p {
				next[i] += v
				next[i+1] -= r * v
			}
			p = next
		}
		return p
	}
	bisect := func(a, b float64, f func(float64) float64) float64 {
		fa := f(a)
		for range 200 {
			m := (a + b) / 2
			if m <= a || m >= b {
				break
			}
			if fm := f(m); (fm > 0) == (fa > 0) {
				a, fa = m, fm
			} else {
				b = m
			}
		}
		return (a + b) / 2
	}
	noise := func(th float64) float64 { return 1e-15 / math.Pow(2*math.Cos(th/2), 4) }
	for _, c := range cases {
		loop := func(th float64) complex128 {
			z := cmplx.Rect(1, th)
			h := complex(c.k, 0) / cmplx.Pow(cmplx.Rect(2*math.Cos(th/2), th/2), 4)
			for _, q := range c.zs {
				h *= z - complex(q, 0)
			}
			for _, q := range c.ps {
				h /= z - complex(q, 0)
			}
			return h
		}
		logMag := func(th float64) float64 { return math.Log(cmplx.Abs(loop(th))) }
		phase := func(th float64) float64 { return phaseOffsetDeg(loop(th)) }
		var thg, thp []float64
		const N = 100000
		for i := 2; i < N; i++ {
			a, b := math.Pi*(1-math.Pow(1-float64(i-1)/N, 3)), math.Pi*(1-math.Pow(1-float64(i)/N, 3))
			if (logMag(a) > 0) != (logMag(b) > 0) {
				thg = append(thg, bisect(a, b, logMag))
			}
			if p := phase(b); (p > 0) != (phase(a) > 0) && math.Abs(p) < 90 && b < math.Pi-1e-8 {
				thp = append(thp, bisect(a, b, phase))
			}
		}
		r, err := (&TransferFunc{Num: [][][]float64{{poly(c.k, c.zs)}}, Den: [][]float64{poly(1, append(c.ps, -1, -1, -1, -1))}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("k %g zs %v ps %v dt %g: %v", c.k, c.zs, c.ps, c.dt, err)
		}
		ok := len(res.GainCrossFreqs) == len(thg) && len(res.PhaseCrossFreqs) == len(thp)
		for i := 0; ok && i < len(thg); i++ {
			th, e := thg[i], noise(thg[i])
			ok = math.Abs(res.GainCrossFreqs[i]*c.dt-th) <= 1e-9*th+e*(math.Pi-th) &&
				math.Abs(res.PhaseMargins[i]-phaseMarginDeg(loop(th))) <= 1e-4+e*180/math.Pi
		}
		for i := 0; ok && i < len(thp); i++ {
			th, e := thp[i], noise(thp[i])
			ok = math.Abs(res.PhaseCrossFreqs[i]*c.dt-th) <= 1e-9*th+e &&
				math.Abs(res.GainMargins[i]+20*math.Log10(cmplx.Abs(loop(th)))) <= 1e-6+10*e
		}
		if !ok {
			t.Errorf("k %g zs %v ps %v dt %g: wgc %v pm %v wpc %v gm %v, want θ %v %v", c.k, c.zs, c.ps, c.dt,
				res.GainCrossFreqs, res.PhaseMargins, res.PhaseCrossFreqs, res.GainMargins, thg, thp)
		}
	}
}

// ZQL2JH: k·z^−d with k < 0 is real negative at θ = 2πm/d, |L| = |k|, so
// phase crossovers there with GM = −20·log10|k|. L(0) = k makes the phase
// error 0 at ω = 0, and a crossover at twice another has it 0 at half its
// frequency too; MATLAB allmargin lists every crossover.
func TestAllMarginNegativeGainDelayHarmonics(t *testing.T) {
	for _, k := range []float64{-2, -116.75, -0.3} {
		for _, dt := range []float64{1, 0.1, 0.79} {
			for d := 1; d <= 12; d++ {
				for _, asDelay := range []bool{false, true} {
					den := []float64{1}
					if !asDelay {
						den = append(den, make([]float64, d)...)
					}
					r, err := (&TransferFunc{Num: [][][]float64{{{k}}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
					if err != nil {
						t.Fatal(err)
					}
					sys := r.Sys
					if asDelay {
						sys.InputDelay = []float64{float64(d)}
					}
					res, err := AllMargin(sys)
					if err != nil {
						t.Fatalf("k %g dt %g d %d delay %v: %v", k, dt, d, asDelay, err)
					}
					var ws []float64
					for m := 1; 2*m <= d; m++ {
						ws = append(ws, 2*math.Pi*float64(m)/float64(d)/dt)
					}
					gm := -20 * math.Log10(-k)
					ok := len(res.PhaseCrossFreqs) == len(ws)
					for i := 0; ok && i < len(ws); i++ {
						ok = math.Abs(res.PhaseCrossFreqs[i]-ws[i]) <= 1e-9*ws[i] && math.Abs(res.GainMargins[i]-gm) <= 1e-9
					}
					if !ok {
						t.Errorf("k %g dt %g d %d delay %v: wpc %v gm %v, want %v %g", k, dt, d, asDelay, res.PhaseCrossFreqs, res.GainMargins, ws, gm)
					}
				}
			}
		}
	}
}

// 6WSXWR: rounded poles of z⁴ − 3.927z³ + … at 0.9329, 0.99437 and
// 1 ± 1.26e-6. The L − L~ pencil root next to the pair lands at ω ≈ 1.6e-4,
// and polish probed f through the sticky evaluation at a bracket end where
// the solve is singular, failing AllMargin and with it TunePID and Pidtune's
// crossover search. Oracle: mpmath at 60 digits on the transfer function,
// which gives the crossover, PM and the Nyquist GM below. Its phase
// crossover at ω ≈ 8.27e-6, |L| ≈ 5.5e13, lies where the state-space solve
// is singular in double, as at every ω ≤ 1e-4, so only ω ≥ 1e-3 is checked.
func TestAllMarginRoundedUnitPolePair(t *testing.T) {
	num := []float64{0.5296264797589852, -0.49398687205654906}
	den := []float64{1, -3.927287723431127, 5.782240810121571, -3.7826184499497617, 0.9276653632593171}
	dt := 0.0422252269048428
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	r, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	res, err := AllMargin(r.Sys)
	if err != nil {
		t.Fatalf("AllMargin: %v", err)
	}
	var wpc, gm []float64
	for i, w := range res.PhaseCrossFreqs {
		if w >= 1e-3 {
			wpc, gm = append(wpc, w), append(gm, res.GainMargins[i])
		}
	}
	const wgc, pm, gmNyq = 19.74525666665513, -161.30454669064147, 23.558862928861675
	if len(res.GainCrossFreqs) != 1 || math.Abs(res.GainCrossFreqs[0]-wgc) > 1e-9*wgc || math.Abs(res.PhaseMargins[0]-pm) > 1e-6 ||
		len(wpc) != 1 || math.Abs(wpc[0]-math.Pi/dt) > 1e-9*wpc[0] || math.Abs(gm[0]-gmNyq) > 1e-8 {
		t.Errorf("wgc %v pm %v wpc %v gm %v, want [%v] [%v] [%v] [%v]", res.GainCrossFreqs, res.PhaseMargins,
			res.PhaseCrossFreqs, res.GainMargins, wgc, pm, math.Pi/dt, gmNyq)
	}
	tr, err := TunePID(context.Background(), r.Sys, PidtunePIDF, PIDTuningOptions{})
	pidCheckZPoleResult(t, "TunePID PIDF", num, den, dt, tr, err)
	if p, err := Pidtune(r.Sys, PidtunePIDF, 0, nil); err != nil && !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Errorf("Pidtune PIDF: %v", err)
	} else if err == nil && !pidtuneExactSchurStable(t, p, num, den, dt) {
		t.Errorf("Pidtune PIDF: unstable design %+v", *p)
	}
}

// UFBPWZ: the pencil root next to a pole pair 1 ± jε lands 0.6% off the
// phase crossover below the pair, and the first bracket spanning it reaches
// past the pole on its far end, where the solve fails; polish dropped the
// crossover. Oracle: factored-form evaluation bisected on a log grid locates
// the crossovers; rounding the expanded denominator moves the pair by
// ~eps/|den'| ≈ 1e-10, so each is re-bisected on the float64 coefficients
// evaluated exactly in big.Float. MATLAB allmargin lists every crossover.
func TestAllMarginPhaseCrossBesidePolePair(t *testing.T) {
	cases := []struct {
		k, eps, dt, wpcLow float64
		zs, ps             []float64
	}{
		{-0.02309469977260592, 2.9030126265483175e-05, 1, 2.8582737529213393e-05,
			[]float64{-0.4863575702993606}, []float64{0.9561590173593617, 0.8871366217113326, -0.43121472459012195}},
		{-0.21501793541077313, 6.176538894270818e-05, 0.79, 7.731566170130944e-05,
			[]float64{-0.4405797671906855, -0.6838981765389421}, []float64{0.9703560326623678, 0.8574995610377053, 0.7919093674641045}},
	}
	bisect := func(a, b float64, f func(float64) float64) float64 {
		fa := f(a)
		for range 200 {
			m := (a + b) / 2
			if m <= a || m >= b {
				break
			}
			if fm := f(m); (fm > 0) == (fa > 0) {
				a, fa = m, fm
			} else {
				b = m
			}
		}
		return (a + b) / 2
	}
	for _, c := range cases {
		loop := func(th float64) complex128 {
			z := cmplx.Rect(1, th)
			h := complex(c.k, 0) / ((z - complex(1, c.eps)) * (z - complex(1, -c.eps)))
			for _, q := range c.zs {
				h *= z - complex(q, 0)
			}
			for _, q := range c.ps {
				h /= z - complex(q, 0)
			}
			return h
		}
		phase := func(th float64) float64 { return phaseOffsetDeg(loop(th)) }
		var thp []float64
		const N = 200000
		lo, hi := math.Log(1e-9), math.Log(math.Pi)
		for i := 1; i <= N; i++ {
			a, b := math.Exp(lo+(hi-lo)*float64(i-1)/N), math.Exp(lo+(hi-lo)*float64(i)/N)
			if p := phase(b); (p > 0) != (phase(a) > 0) && math.Abs(p) < 90 {
				thp = append(thp, bisect(a, b, phase))
			}
		}
		if len(thp) == 0 || math.Abs(thp[0]/c.dt-c.wpcLow) > 1e-9*c.wpcLow {
			t.Fatalf("oracle θ %v, want first ω %v", thp, c.wpcLow)
		}
		num, den := []float64{c.k}, []float64{1, -2, 1 + c.eps*c.eps}
		for _, q := range c.zs {
			num = polyMul(num, []float64{1, -q})
		}
		for _, q := range c.ps {
			den = polyMul(den, []float64{1, -q})
		}
		exact := func(th float64) complex128 {
			n, d := bigHorner(num, th), bigHorner(den, th)
			return n / d
		}
		for i, th := range thp {
			thp[i] = bisect(th*(1-1e-4), min(th*(1+1e-4), math.Pi), func(th float64) float64 { return phaseOffsetDeg(exact(th)) })
		}
		loop = exact
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("k %g: %v", c.k, err)
		}
		ok := len(res.PhaseCrossFreqs) == len(thp)
		for i := 0; ok && i < len(thp); i++ {
			th := thp[i]
			ok = math.Abs(res.PhaseCrossFreqs[i]*c.dt-th) <= 1e-7*th &&
				math.Abs(res.GainMargins[i]+20*math.Log10(cmplx.Abs(loop(th)))) <= 1e-5
		}
		if !ok {
			t.Errorf("k %g: wpc %v gm %v, want θ %v", c.k, res.PhaseCrossFreqs, res.GainMargins, thp)
		}
	}
}

// UJSGC2: the pencil root for the crossover below an unstable pair 1 ± jε,
// |p| − 1 ≈ 1e-10, lands past the pair (arm64 1.4888e-5, amd64 1.4700e-5,
// pole 1.4688e-5), where L is real positive; polish never sees it and the
// resonance scan must recover the crossover 1% below the pole. Oracle:
// float64 coefficients evaluated at z = e^{jθ} with cos, sin and Horner in
// 300-bit big.Float, Im(N·conj D) bisected; Re < 0 there.
func TestAllMarginPhaseCrossPastUnstablePolePair(t *testing.T) {
	num := []float64{-2.166935933035251, 1.9807432626284676}
	den := []float64{1, -2.73127553541105, 2.2142827468764708, -0.23473888724605785, -0.2482683242149491}
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	mul := func(a, b *big.Float) *big.Float { return nf(0).Mul(a, b) }
	at := func(p []float64, th float64) (re, im *big.Float) {
		c, s, term := nf(0), nf(0), nf(1)
		for k := range 40 {
			switch k % 4 {
			case 0:
				c.Add(c, term)
			case 1:
				s.Add(s, term)
			case 2:
				c.Sub(c, term)
			default:
				s.Sub(s, term)
			}
			term.Mul(term, nf(th)).Quo(term, nf(float64(k+1)))
		}
		re, im = nf(0), nf(0)
		for _, a := range p {
			re, im = nf(0).Sub(mul(re, c), mul(im, s)), nf(0).Add(mul(re, s), mul(im, c))
			re.Add(re, nf(a))
		}
		return re, im
	}
	loop := func(th float64) (re, im *big.Float, mag float64) {
		nr, ni := at(num, th)
		dr, di := at(den, th)
		re = nf(0).Add(mul(nr, dr), mul(ni, di))
		im = nf(0).Sub(mul(ni, dr), mul(nr, di))
		n2, _ := nf(0).Add(mul(nr, nr), mul(ni, ni)).Float64()
		d2, _ := nf(0).Add(mul(dr, dr), mul(di, di)).Float64()
		return re, im, math.Sqrt(n2 / d2)
	}
	cross := func(a, b float64) (w, gm float64) {
		_, ia, _ := loop(a)
		for range 200 {
			m := (a + b) / 2
			if m <= a || m >= b {
				break
			}
			if _, im, _ := loop(m); im.Sign() == ia.Sign() {
				a = m
			} else {
				b = m
			}
		}
		re, _, mag := loop(a)
		if re.Sign() >= 0 {
			t.Fatalf("oracle: Re L ≥ 0 at %v", a)
		}
		return a, -20 * math.Log10(mag)
	}
	wLow, gmLow := cross(1.44e-5, 1.465e-5)
	wHigh, gmHigh := cross(1.6, 1.75)
	r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	res, err := AllMargin(r.Sys)
	if err != nil {
		t.Fatal(err)
	}
	want, wantGM := []float64{wLow, wHigh}, []float64{gmLow, gmHigh}
	ok := len(res.PhaseCrossFreqs) == len(want)
	for i := 0; ok && i < len(want); i++ {
		ok = math.Abs(res.PhaseCrossFreqs[i]-want[i]) <= 1e-7*want[i] && math.Abs(res.GainMargins[i]-wantGM[i]) <= 1e-5
	}
	if !ok {
		t.Errorf("wpc %v gm %v, want %v %v", res.PhaseCrossFreqs, res.GainMargins, want, wantGM)
	}
}

// The walk past a singular band must not keep a sign change that rounding
// makes beside it: next to (z+1)⁴ the response carries relative rounding
// eps/(2·cos(θ/2))⁴, which near θ ≈ 3.1411 flips the sign of a phase offset
// of −4e-4° (amd64 reported a crossover there, GM −211 dB). Oracle:
// factored form with z+1 = 2·cos(θ/2)·e^{jθ/2}; arg L + π stays negative
// from the crossover near θ = 3.0769 up to the pole at π.
func TestAllMarginNoCrossoverBesideNyquistCluster(t *testing.T) {
	k := -0.0075348300096785325
	zs := []float64{-0.3183318840227378, -0.6312441609901465}
	ps := []float64{-0.22135332000025376, 0.13662162891886176}
	num, den := []float64{k}, polyMul([]float64{1, 2, 1}, []float64{1, 2, 1})
	for _, q := range zs {
		num = polyMul(num, []float64{1, -q})
	}
	for _, q := range ps {
		den = polyMul(den, []float64{1, -q})
	}
	loop := func(th float64) complex128 {
		z := cmplx.Rect(1, th)
		h := complex(k, 0) / cmplx.Pow(cmplx.Rect(2*math.Cos(th/2), th/2), 4)
		for _, q := range zs {
			h *= z - complex(q, 0)
		}
		for _, q := range ps {
			h /= z - complex(q, 0)
		}
		return h
	}
	var thp []float64
	const N = 100000
	for i := 2; i < N; i++ {
		a, b := math.Pi*(1-math.Pow(1-float64(i-1)/N, 3)), math.Pi*(1-math.Pow(1-float64(i)/N, 3))
		pa, pb := phaseOffsetDeg(loop(a)), phaseOffsetDeg(loop(b))
		if (pa > 0) != (pb > 0) && math.Abs(pb) < 90 {
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if pm := phaseOffsetDeg(loop(m)); (pm > 0) == (pa > 0) {
					a, pa = m, pm
				} else {
					b = m
				}
			}
			thp = append(thp, (a+b)/2)
		}
	}
	r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	res, err := AllMargin(r.Sys)
	if err != nil {
		t.Fatal(err)
	}
	ok := len(thp) == 1 && math.Abs(thp[0]-3.0769) < 1e-3 && len(res.PhaseCrossFreqs) == 1
	if ok {
		th := thp[0]
		ok = math.Abs(res.PhaseCrossFreqs[0]-th) <= 1e-9*th && math.Abs(res.GainMargins[0]+20*math.Log10(cmplx.Abs(loop(th)))) <= 1e-6
	}
	if !ok {
		t.Errorf("wpc %v gm %v, want θ %v", res.PhaseCrossFreqs, res.GainMargins, thp)
	}
}

// bigHorner evaluates the real polynomial p (descending) at e^{jθ}, taken
// in float64, rounding only the result.
func bigHorner(p []float64, th float64) complex128 {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	zr, zi := nf(math.Cos(th)), nf(math.Sin(th))
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

// Y3RHVQ: B of k/((z+1)⁴·q(z)) with |k| ≈ 1e-14 sits below QZ's backward
// error relative to A, and the pencil of L − L~ lost phase crossovers far
// from the cluster. Oracle: factored form with z+1 = 2·cos(θ/2)·e^{jθ/2},
// bisected on a log grid; coefficient rounding moves L there by ~1e-9
// relative. Crossovers within 0.05 of Nyquist are left out: rounding of the
// expanded (z+1)⁴ is O(1) relative there. A seeded sweep of |k| in
// [1e-15, 1e-11] with an optional zero and lag follows the repros.
func TestAllMarginPhaseCrossTinyGainNyquistCluster(t *testing.T) {
	cases := []struct {
		k, dt  float64
		zs, ps []float64
	}{
		{-1.6726418877515324e-14, 0.1, nil, []float64{-0.7445460248190218}},
		{1.0113710330406558e-14, 1, nil, []float64{-0.6087189607622736}},
		{-5.6015128843776325e-14, 0.1, []float64{0.8621385615295611}, nil},
		{-1.7579243818241682e-14, 0.79, []float64{0.5819112675552129}, nil},
	}
	rng := rand.New(rand.NewPCG(41, 43))
	dts := []float64{1, 0.1, 0.79, 0.01}
	for i := range 80 {
		c := cases[0]
		c.k, c.dt, c.zs, c.ps = math.Copysign(math.Pow(10, -15+4*rng.Float64()), rng.Float64()-0.5), dts[i%4], nil, nil
		if rng.IntN(2) == 0 {
			c.zs = []float64{1 - 2*rng.Float64()}
		}
		if rng.IntN(2) == 0 {
			c.ps = []float64{0.9 - 1.8*rng.Float64()}
		}
		cases = append(cases, c)
	}
	const thMax = math.Pi - 0.05
	for ci, c := range cases {
		num, den := []float64{c.k}, polyMul([]float64{1, 2, 1}, []float64{1, 2, 1})
		for _, q := range c.zs {
			num = polyMul(num, []float64{1, -q})
		}
		for _, q := range c.ps {
			den = polyMul(den, []float64{1, -q})
		}
		loop := func(th float64) complex128 {
			z := cmplx.Rect(1, th)
			h := complex(c.k, 0) / cmplx.Pow(cmplx.Rect(2*math.Cos(th/2), th/2), 4)
			for _, q := range c.zs {
				h *= z - complex(q, 0)
			}
			for _, q := range c.ps {
				h /= z - complex(q, 0)
			}
			return h
		}
		var thp []float64
		const N = 100000
		lo, hi := math.Log(1e-6), math.Log(thMax)
		for i := 1; i <= N; i++ {
			a, b := math.Exp(lo+(hi-lo)*float64(i-1)/N), math.Exp(lo+(hi-lo)*float64(i)/N)
			pa, pb := phaseOffsetDeg(loop(a)), phaseOffsetDeg(loop(b))
			if (pa > 0) == (pb > 0) || math.Abs(pb) >= 90 {
				continue
			}
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if pm := phaseOffsetDeg(loop(m)); (pm > 0) == (pa > 0) {
					a, pa = m, pm
				} else {
					b = m
				}
			}
			thp = append(thp, (a+b)/2)
		}
		if len(thp) == 0 && ci < 4 {
			t.Fatalf("k %g: oracle found no crossover", c.k)
		}
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("k %g: %v", c.k, err)
		}
		var got, gm []float64
		for i, w := range res.PhaseCrossFreqs {
			if w*c.dt < thMax {
				got, gm = append(got, w*c.dt), append(gm, res.GainMargins[i])
			}
		}
		ok := len(got) == len(thp)
		for i := 0; ok && i < len(thp); i++ {
			th := thp[i]
			ok = math.Abs(got[i]-th) <= 1e-6*th && math.Abs(gm[i]+20*math.Log10(cmplx.Abs(loop(th)))) <= 1e-5
		}
		if !ok {
			t.Errorf("k %g: θpc %v gm %v, want θ %v", c.k, got, gm, thp)
		}
	}
}

// Y3RHVQ: beside the cluster of k/((z+1)⁴·q(z)) with tiny |k|, |L| reaches 1
// where the double solve of zI − A is singular to working precision, and
// gain crossovers were dropped. Oracle: the expanded coefficients evaluated
// in big floats, bisected on a grid graded to Nyquist; the two reviewer
// repros and a seeded sweep of |k| in [1e-14, 3e-13].
func TestAllMarginGainCrossTinyGainNyquistCluster(t *testing.T) {
	type tc struct {
		num, den []float64
		dt       float64
	}
	den4 := polyMul([]float64{1, 2, 1}, []float64{1, 2, 1})
	cases := []tc{
		{[]float64{-1.6726418877515324e-14}, polyMul(den4, []float64{1, 0.7445460248190218}), 0.1},
		{polyMul([]float64{2.4983505154002314e-13}, []float64{1, 0.8685335634364679}), polyMul(den4, []float64{1, 0.6851821799011946}), 0.79},
	}
	rng := rand.New(rand.NewPCG(3, 5))
	dts := []float64{1, 0.1, 0.79, 0.01}
	for i := range 40 {
		k := math.Copysign(math.Pow(10, -14+math.Log10(30)*rng.Float64()), rng.Float64()-0.5)
		c := tc{[]float64{k}, den4, dts[i%4]}
		if rng.IntN(2) == 0 {
			c.num = polyMul(c.num, []float64{1, 1 - 2*rng.Float64()})
		}
		if rng.IntN(2) == 0 {
			c.den = polyMul(c.den, []float64{1, 0.9 - 1.8*rng.Float64()})
		}
		cases = append(cases, c)
	}
	for _, c := range cases {
		loop := func(th float64) complex128 { return bigHorner(c.num, th) / bigHorner(c.den, th) }
		lm := func(th float64) float64 { return math.Log(cmplx.Abs(loop(th))) }
		var want []float64
		const N = 20000
		for i := 1; i < N; i++ {
			a, b := math.Pi*(1-math.Pow(1-float64(i-1)/N, 4)), math.Pi*(1-math.Pow(1-float64(i)/N, 4))
			fa, fb := lm(a), lm(b)
			if (fa > 0) == (fb > 0) {
				continue
			}
			for range 100 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if fm := lm(m); (fm > 0) == (fa > 0) {
					a, fa = m, fm
				} else {
					b = m
				}
			}
			want = append(want, (a+b)/2)
		}
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("%v/%v dt %g: %v", c.num, c.den, c.dt, err)
		}
		ok := len(res.GainCrossFreqs) == len(want)
		for i := 0; ok && i < len(want); i++ {
			th := want[i]
			ok = math.Abs(res.GainCrossFreqs[i]*c.dt-th) <= 1e-7*th &&
				math.Abs(wrapDegrees(res.PhaseMargins[i]-phaseMarginDeg(loop(th)))) <= 1e-4
		}
		if !ok {
			t.Errorf("%v/%v dt %g: wgc %v pm %v, want θ %v", c.num, c.den, c.dt, res.GainCrossFreqs, res.PhaseMargins, want)
		}
	}
}

// Bandwidth of a loop with small DC gain: the gain pencil at γ = |L(0)|·10^(−3/20)
// carried C/γ against B ≪ C/γ, and C·Cᵀ/γ² swamped A, so QZ lost the root
// and Bandwidth returned +Inf. Oracle: analytic a·√(10^0.3 − 1) for k/(s+a);
// factored-form bisection on a log grid for the seeded sweep.
func TestBandwidthSmallDCGain(t *testing.T) {
	a := 0.597392081703245
	for _, k := range []float64{1e-14, 1e-12, 1e-10, 1} {
		r, err := (&TransferFunc{Num: [][][]float64{{{k}}}, Den: [][]float64{{1, a}}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		want := a * math.Sqrt(math.Pow(10, 0.3)-1)
		if bw, err := Bandwidth(r.Sys, -3); err != nil || math.Abs(bw-want) > 1e-9*want {
			t.Errorf("k %g: bw %v err %v, want %v", k, bw, err, want)
		}
	}
	rng := rand.New(rand.NewPCG(11, 13))
	for range 150 {
		k := math.Copysign(math.Pow(10, -14+14*rng.Float64()), rng.Float64()-0.5)
		var zs, ps []float64
		for range 1 + rng.IntN(4) {
			ps = append(ps, -math.Pow(10, -2+4*rng.Float64()))
		}
		if rng.IntN(2) == 0 {
			zs = append(zs, -math.Pow(10, -2+4*rng.Float64()))
		}
		num, den := []float64{k}, []float64{1}
		for _, q := range zs {
			num = polyMul(num, []float64{1, -q})
		}
		for _, q := range ps {
			den = polyMul(den, []float64{1, -q})
		}
		mag := func(w float64) float64 {
			h := complex(k, 0)
			for _, q := range zs {
				h *= complex(-q, w)
			}
			for _, q := range ps {
				h /= complex(-q, w)
			}
			return cmplx.Abs(h)
		}
		lvl := mag(0) * math.Pow(10, -3.0/20)
		above := func(w float64) bool { return mag(w) > lvl }
		want := math.Inf(1)
		lo, hi := math.Log(1e-6), math.Log(1e6)
		const N = 20000
		for i := 1; i <= N; i++ {
			a, b := math.Exp(lo+(hi-lo)*float64(i-1)/N), math.Exp(lo+(hi-lo)*float64(i)/N)
			if above(a) == above(b) {
				continue
			}
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if above(m) == above(a) {
					a = m
				} else {
					b = m
				}
			}
			want = (a + b) / 2
			break
		}
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		if bw, err := Bandwidth(r.Sys, -3); err != nil || math.Abs(bw-want) > 1e-7*want {
			t.Errorf("k %g zs %v ps %v: bw %v err %v, want %v", k, zs, ps, bw, err, want)
		}
	}
}

// UFBPWZ sweep: pole pairs 1 ± jε, ε in [1e-6, 1e-3], with lags and zeros.
// The pencil of L − L~ holds the pair, its reflection and the crossings in a
// cluster of radius ε and lost crossovers near ε for ε ≲ 1e-4.5. Oracle:
// the expanded float64 coefficients evaluated exactly in big.Float, phase
// offset bisected on a log grid, plus θ = π when L is real negative there.
// Realization rounding moves ε² by ~eps, so θ is held to 50·eps/ε² relative
// (1-ulp coefficient perturbations move it by up to ~10·eps/ε²).
func TestAllMarginPhaseCrossPolePairSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 23))
	dts := []float64{1, 0.1, 0.79}
	for it := range 90 {
		k := math.Copysign(math.Pow(10, -3+3*rng.Float64()), rng.Float64()-0.5)
		ep := math.Pow(10, -6+3*rng.Float64())
		num, den := []float64{k}, []float64{1, -2, 1 + ep*ep}
		for range rng.IntN(3) {
			num = polyMul(num, []float64{1, -(1.8*rng.Float64() - 0.9)})
		}
		for range 1 + rng.IntN(3) {
			den = polyMul(den, []float64{1, -(1.9*rng.Float64() - 0.9)})
		}
		dt := dts[it%3]
		ph := func(th float64) float64 { return phaseOffsetDeg(bigHorner(num, th) / bigHorner(den, th)) }
		var want []float64
		const N = 5000
		lo, hi := math.Log(1e-9), math.Log(math.Pi)
		for i := 1; i <= N; i++ {
			a, b := math.Exp(lo+(hi-lo)*float64(i-1)/N), math.Exp(lo+(hi-lo)*float64(i)/N)
			if i == N {
				b = math.Pi
			}
			pa, pb := ph(a), ph(b)
			if (pa > 0) == (pb > 0) || math.Abs(pb) >= 90 || math.Abs(pa) >= 90 {
				continue
			}
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if pm := ph(m); (pm > 0) == (pa > 0) {
					a, pa = m, pm
				} else {
					b = m
				}
			}
			want = append(want, (a+b)/2)
		}
		if math.Abs(ph(math.Pi)) <= 1e-7 && (len(want) == 0 || want[len(want)-1] < math.Pi*(1-1e-9)) {
			want = append(want, math.Pi)
		}
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Errorf("%d: %v", it, err)
			continue
		}
		rel := max(1e-7, 50*eps()/(ep*ep))
		ok := len(res.PhaseCrossFreqs) == len(want)
		for i := 0; ok && i < len(want); i++ {
			ok = math.Abs(res.PhaseCrossFreqs[i]*dt-want[i]) <= rel*want[i]
		}
		if !ok {
			t.Errorf("%d ε %.3g num %v den %v dt %g: wpc %v, want θ %v", it, ep, num, den, dt, res.PhaseCrossFreqs, want)
		}
	}
}

// The resonance scan applies to continuous lightly damped pairs too; it must
// add only crossovers the loop has. Oracle: the expanded coefficients at jω
// in big.Float, bisected on a log grid refined around the pair.
func TestAllMarginPhaseCrossLightlyDampedSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 37))
	for it := range 60 {
		k := math.Copysign(math.Pow(10, -2+4*rng.Float64()), rng.Float64()-0.5)
		w0 := math.Pow(10, -2+4*rng.Float64())
		z := math.Pow(10, -7+4*rng.Float64())
		num, den := []float64{k}, []float64{1, 2 * z * w0, w0 * w0}
		for range rng.IntN(3) {
			num = polyMul(num, []float64{1, math.Copysign(math.Pow(10, -2+4*rng.Float64()), rng.Float64()-0.3)})
		}
		for range 1 + rng.IntN(3) {
			den = polyMul(den, []float64{1, math.Pow(10, -2+4*rng.Float64())})
		}
		ph := func(w float64) float64 { return phaseOffsetDeg(bigHornerJ(num, w) / bigHornerJ(den, w)) }
		var grid []float64
		const N = 5000
		lo, hi := math.Log(1e-5), math.Log(1e5)
		for i := 0; i <= N; i++ {
			grid = append(grid, math.Exp(lo+(hi-lo)*float64(i)/N))
		}
		for i := 0; i <= 1000; i++ {
			s := math.Pow(10, -10+10*float64(i)/1000)
			grid = append(grid, w0*(1-0.9*s), w0*(1+9*s))
		}
		slices.Sort(grid)
		var want []float64
		for i := 1; i < len(grid); i++ {
			a, b := grid[i-1], grid[i]
			pa, pb := ph(a), ph(b)
			if (pa > 0) == (pb > 0) || math.Abs(pb) >= 90 || math.Abs(pa) >= 90 {
				continue
			}
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if pm := ph(m); (pm > 0) == (pa > 0) {
					a, pa = m, pm
				} else {
					b = m
				}
			}
			want = append(want, (a+b)/2)
		}
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Errorf("%d: %v", it, err)
			continue
		}
		var got []float64
		for _, w := range res.PhaseCrossFreqs {
			if w > 1e-5 && w < 1e5 {
				got = append(got, w)
			}
		}
		ok := len(got) == len(want)
		for i := 0; ok && i < len(want); i++ {
			ok = math.Abs(got[i]-want[i]) <= 1e-6*want[i]
		}
		if !ok {
			t.Errorf("%d ζ %.3g ω0 %.3g num %v den %v: wpc %v, want %v", it, z, w0, num, den, got, want)
		}
	}
}

// bigHornerJ evaluates the real polynomial p (descending) at jω exactly,
// rounding only the result.
func bigHornerJ(p []float64, w float64) complex128 {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	zi := nf(w)
	re, im, t1 := nf(0), nf(0), nf(0)
	for _, c := range p {
		nr := nf(0).Neg(t1.Mul(im, zi))
		im.Mul(re, zi)
		re.Add(nr, nf(c))
	}
	r, _ := re.Float64()
	i, _ := im.Float64()
	return complex(r, i)
}
