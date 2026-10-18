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
