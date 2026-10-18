package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
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
