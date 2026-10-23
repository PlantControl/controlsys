package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// delayLimitLoop is L = (d + c/(s−a))·e^{−sτ1} + d2·e^{−sτ2}.
type delayLimitLoop struct {
	a, c, d, tau1, d2, tau2 float64
}

func (l delayLimitLoop) system(t testing.TB) *System {
	t.Helper()
	var G *System
	var err error
	if l.c == 0 {
		G, err = NewGain(mat.NewDense(1, 1, []float64{l.d}), 0)
	} else {
		G, err = NewFromSlices(1, 1, 1, []float64{l.a}, []float64{1}, []float64{l.c}, []float64{l.d}, 0)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := G.SetInputDelay([]float64{l.tau1}); err != nil {
		t.Fatal(err)
	}
	K, err := NewGain(mat.NewDense(1, 1, []float64{l.d2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := K.SetInputDelay([]float64{l.tau2}); err != nil {
		t.Fatal(err)
	}
	L, err := Parallel(G, K)
	if err != nil {
		t.Fatal(err)
	}
	return L
}

func (l delayLimitLoop) at(w float64) complex128 {
	g := complex(l.d, 0)
	if l.c != 0 {
		g += complex(l.c, 0) / complex(-l.a, w)
	}
	return g*cmplx.Exp(complex(0, -w*l.tau1)) + complex(l.d2, 0)*cmplx.Exp(complex(0, -w*l.tau2))
}

// limitAt is the high-frequency limit of L with the phase φ = ω·τ1.
func (l delayLimitLoop) limitAt(phi float64) complex128 {
	return complex(l.d, 0)*cmplx.Exp(complex(0, -phi)) + complex(l.d2, 0)*cmplx.Exp(complex(0, -phi*l.tau2/l.tau1))
}

// delayLimitSup is sup |1/(1+f(x)) + c| over x in [0, hi] by a dense sweep
// refined by golden section.
func delayLimitSup(f func(float64) complex128, c, hi float64, n int) (float64, float64) {
	g := func(x float64) float64 { return cmplx.Abs(1/(1+f(x)) + complex(c, 0)) }
	h := hi / float64(n)
	best, xb := 0.0, 0.0
	for i := 0; i <= n; i++ {
		if v := g(float64(i) * h); v > best {
			best, xb = v, float64(i)*h
		}
	}
	lo, up := math.Max(0, xb-h), xb+h
	r := (math.Sqrt(5) - 1) / 2
	for range 200 {
		x1, x2 := up-r*(up-lo), lo+r*(up-lo)
		if g(x1) > g(x2) {
			up = x2
		} else {
			lo = x1
		}
	}
	if v := g((lo + up) / 2); v > best {
		best, xb = v, (lo+up)/2
	}
	return best, xb
}

// TestDiskMarginDelayedFeedthroughLimitSet checks DiskMargin of delayed loops
// with direct feedthrough whose high-frequency limit set misses −tailLimit,
// where the peak used to be the disk bound at tailLimit reported at +Inf:
// 1.722222 = (1+0.55)/(2·0.45) for odd τ2/τ1 = 3, 5 and 1.5 for
// 0.3 + 0.2e^{−s}. The oracle is a dense sweep of the limit over one phase
// period and, for a rational G, of L(jω) up to where it has settled.
func TestDiskMarginDelayedFeedthroughLimitSet(t *testing.T) {
	const h = 0.50118637084960938
	cases := []struct {
		name   string
		l      delayLimitLoop
		finite bool
	}{
		{"static k=3", delayLimitLoop{d: 0.3, tau1: 0.5, d2: -0.25, tau2: 1.5}, false},
		{"static k=5", delayLimitLoop{d: 0.3, tau1: 0.5, d2: -0.25, tau2: 2.5}, false},
		{"static k=2", delayLimitLoop{d: 0.3, tau1: 0.5, d2: -0.25, tau2: 1}, false},
		{"fuzz PN4IYP", delayLimitLoop{d: 0.29732546154427597, tau1: h, d2: -0.25, tau2: 3 * h}, false},
		{"static -0.3 k=3", delayLimitLoop{d: -0.3, tau1: 1, d2: 0.33, tau2: 3}, false},
		{"static -0.5 k=3", delayLimitLoop{d: -0.5, tau1: 1, d2: 0.1, tau2: 3}, false},
		{"constant plus delay", delayLimitLoop{d: 0.3, tau1: 0, d2: 0.2, tau2: 1}, false},
		{"rational k=3", delayLimitLoop{a: -1, c: 0.1, d: 0.3, tau1: 0.5, d2: -0.25, tau2: 1.5}, true},
		{"rational k=5", delayLimitLoop{a: -1, c: 0.1, d: 0.3, tau1: 0.5, d2: -0.25, tau2: 2.5}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := tc.l
			res, err := DiskMargin(l.system(t))
			if err != nil {
				t.Fatal(err)
			}
			for _, sh := range []struct {
				c        float64
				got, wGo float64
			}{{-0.5, 1 / res.Alpha, res.frequency}, {0, res.PeakSensitivity, res.peakFreq}} {
				var want float64
				if l.tau1 == 0 {
					want, _ = delayLimitSup(func(p float64) complex128 {
						return complex(l.d, 0) + complex(l.d2, 0)*cmplx.Exp(complex(0, -p))
					}, sh.c, 2*math.Pi, 200000)
				} else {
					want, _ = delayLimitSup(l.limitAt, sh.c, 2*math.Pi, 200000)
				}
				if tc.finite {
					fin, wf := delayLimitSup(l.at, sh.c, 400, 2000000)
					if !(fin > want) {
						t.Fatalf("c=%g: oracle finite peak %v at %v not above limit sup %v", sh.c, fin, wf, want)
					}
					want = fin
					if math.IsInf(sh.wGo, 0) || math.Abs(sh.wGo-wf) > 1e-3*wf {
						t.Errorf("c=%g: peak at ω=%v, want %v", sh.c, sh.wGo, wf)
					}
				}
				if math.Abs(sh.got-want) > 1e-6*want {
					t.Errorf("c=%g: ‖S+c‖∞ = %.12g at ω=%v, want %.12g", sh.c, sh.got, sh.wGo, want)
				}
			}
		})
	}
}

// TestDiskMarginStaticDelayLimitSweep checks the peaks of seeded static
// loops d·e^{−s} + d2·e^{−rs}, |d| + |d2| < 0.9, against a dense sweep of
// the periodic response. Where the grid sample already matched the limit
// sup, the delay H∞ peak used to refuse and the disk bound at tailLimit was
// kept at +Inf (2.70 for d = −0.3, d2 = 0.33, r = 3, exact 1.91).
func TestDiskMarginStaticDelayLimitSweep(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(15, 4))
	ratios := []float64{2, 3, 4, 5, 1.5, 2.5}
	for i := range 60 {
		r := ratios[i%len(ratios)]
		d := 1.2*rng.Float64() - 0.6
		d2 := (0.9 - math.Abs(d)) * (2*rng.Float64() - 1)
		l := delayLimitLoop{d: d, tau1: 1, d2: d2, tau2: r}
		res, err := DiskMargin(l.system(t))
		if err != nil {
			t.Errorf("%+v: %v", l, err)
			continue
		}
		for _, sh := range []struct{ c, got float64 }{{-0.5, 1 / res.Alpha}, {0, res.PeakSensitivity}} {
			want, _ := delayLimitSup(l.limitAt, sh.c, 4*math.Pi, 400000)
			if math.Abs(sh.got-want) > 1e-6*want {
				t.Errorf("%+v c=%g: ‖S+c‖∞ = %.12g, want %.12g", l, sh.c, sh.got, want)
			}
		}
	}
}

func BenchmarkDiskMargin_DelayFeedthroughLimitSet(b *testing.B) {
	L := delayLimitLoop{a: -1, c: 0.1, d: 0.3, tau1: 0.5, d2: -0.25, tau2: 1.5}.system(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DiskMargin(L); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiskMargin_DelayFeedthroughIO(b *testing.B) {
	L, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{-0.1}, []float64{0.5}, 0)
	if err != nil {
		b.Fatal(err)
	}
	if err := L.SetInputDelay([]float64{0.2}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DiskMargin(L); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiskMargin_DelayFeedthroughIncommensurate(b *testing.B) {
	L := delayLimitLoop{a: -1, c: 0.1, d: 0.3, tau1: 0.5, d2: -0.25, tau2: 0.5 * math.Sqrt2}.system(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DiskMargin(L); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiskMargin_DelayFeedthroughStaticOdd(b *testing.B) {
	L := delayLimitLoop{d: -0.3, tau1: 1, d2: 0.33, tau2: 3}.system(b)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := DiskMargin(L); err != nil {
			b.Fatal(err)
		}
	}
}
