package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// seriesParallelOracle is G1·(G2·e^{-τ2·s} + 1)·e^{-τ1·s}, the response of
// Parallel(Series(G1, G2), G1) for first-order quasiLoops, at s.
func seriesParallelOracle(q1, q2 quasiLoop, s complex128) complex128 {
	g1 := polyEval(q1.num, s) / polyEval(q1.den, s)
	g2 := polyEval(q2.num, s) / polyEval(q2.den, s)
	return g1 * (g2*cmplx.Exp(-s*complex(q2.tau, 0)) + 1) * cmplx.Exp(-s*complex(q1.tau, 0))
}

func seriesParallel(t *testing.T, q1, q2 quasiLoop, swap bool) *System {
	t.Helper()
	g1 := q1.system(t)
	s, err := Series(g1, q2.system(t))
	if err != nil {
		t.Fatal(err)
	}
	a, b := s, g1
	if swap {
		a, b = b, a
	}
	p, err := Parallel(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !p.HasInternalDelay() {
		t.Fatal("model has no internal delay")
	}
	return p
}

// TestFreqResponseInternalDelayOriginClusterPole: at s = 0 an integrator next
// to poles far below ‖A‖ makes the double solve fail at every pole-limit
// sample; the response is +Inf there, as MATLAB freqresp, not an error.
func TestFreqResponseInternalDelayOriginClusterPole(t *testing.T) {
	cases := []struct {
		name   string
		q1, q2 quasiLoop
	}{
		{"near-integrator-then-integrator",
			quasiLoop{num: []float64{-2.6506832460777727}, den: []float64{1, 2.9599743917931223e-06}, tau: 0.5317069693106294},
			quasiLoop{num: []float64{-0.0629684105573812}, den: []float64{1, 0}, tau: 2.8901588864517946}},
		{"two-integrators",
			quasiLoop{num: []float64{2.968381527212278}, den: []float64{1, 0}, tau: 2.001323961875724},
			quasiLoop{num: []float64{1.948102215865926}, den: []float64{1, 0}, tau: 2.0429465163095806}},
	}
	omega := []float64{0, 1e-9, 1e-7, 1e-3, 1}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sys := seriesParallel(t, c.q1, c.q2, false)
			r, err := sys.FreqResponse(omega)
			if err != nil {
				t.Fatalf("FreqResponse: %v", err)
			}
			if !cmplx.IsInf(r.Data[0]) {
				t.Errorf("G(0) = %v, want Inf", r.Data[0])
			}
			for k, w := range omega[1:] {
				got, want := r.Data[k+1], seriesParallelOracle(c.q1, c.q2, complex(0, w))
				if e := cmplx.Abs(got-want) / cmplx.Abs(want); !(e < 1e-9) {
					t.Errorf("G(j%g) = %v, want %v (rel %g)", w, got, want, e)
				}
			}
			g, err := sys.EvalFr(0)
			if err != nil {
				t.Fatalf("EvalFr(0): %v", err)
			}
			if !cmplx.IsInf(g[0][0]) {
				t.Errorf("EvalFr(0) = %v, want Inf", g[0][0])
			}
		})
	}
}

// TestFreqResponseInternalDelayDiscreteUnitCluster is the discrete
// analogue at z = 1: an integrator beside a pole 1e-7 inside it.
func TestFreqResponseInternalDelayDiscreteUnitCluster(t *testing.T) {
	lag := func(k, a, d float64) *System {
		sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{1}, []float64{k}, []float64{0}, 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInputDelay([]float64{d}); err != nil {
			t.Fatal(err)
		}
		return sys
	}
	g1, g2 := lag(-2.65, 1-1e-7, 2), lag(-0.063, 1, 3)
	s, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := Parallel(s, g1)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.HasInternalDelay() {
		t.Fatal("model has no internal delay")
	}
	omega := []float64{0, 1e-3}
	r, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	if !cmplx.IsInf(r.Data[0]) {
		t.Errorf("G(1) = %v, want Inf", r.Data[0])
	}
	z := cmplx.Exp(complex(0, omega[1]))
	h1, h2 := -2.65/(z-(1-1e-7)), -0.063/(z-1)
	want := h1 * (h2*cmplx.Pow(z, -3) + 1) * cmplx.Pow(z, -2)
	if e := cmplx.Abs(r.Data[1]-want) / cmplx.Abs(want); !(e < 1e-9) {
		t.Errorf("G(e^{j·1e-3}) = %v, want %v (rel %g)", r.Data[1], want, e)
	}
}

// TestFreqResponseInternalDelayOriginClusterFiniteEntry: appending a
// delayed lag keeps the whole pencil singular at s = 0, so the lag's entry is
// extrapolated from the double-double samples; it must keep its finite value.
func TestFreqResponseInternalDelayOriginClusterFiniteEntry(t *testing.T) {
	q1 := quasiLoop{num: []float64{-2.6506832460777727}, den: []float64{1, 2.9599743917931223e-06}, tau: 0.5317069693106294}
	q2 := quasiLoop{num: []float64{-0.0629684105573812}, den: []float64{1, 0}, tau: 2.8901588864517946}
	l1 := quasiLoop{num: []float64{1.7}, den: []float64{1, 0.8}, tau: 0.3}
	l2 := quasiLoop{num: []float64{-0.6}, den: []float64{1, 2.5}, tau: 1.1}
	sys, err := Append(seriesParallel(t, q1, q2, false), seriesParallel(t, l1, l2, false))
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0, 1e-3}
	r, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for k, w := range omega {
		s := complex(0, w)
		g := r.Data[k*4 : (k+1)*4]
		if w == 0 {
			if !cmplx.IsInf(g[0]) {
				t.Errorf("G11(0) = %v, want Inf", g[0])
			}
		} else if want := seriesParallelOracle(q1, q2, s); !(cmplx.Abs(g[0]-want) < 1e-9*cmplx.Abs(want)) {
			t.Errorf("G11(j%g) = %v, want %v", w, g[0], want)
		}
		if g[1] != 0 || g[2] != 0 {
			t.Errorf("off-diagonal at ω=%g = %v, %v, want 0", w, g[1], g[2])
		}
		if want := seriesParallelOracle(l1, l2, s); !(cmplx.Abs(g[3]-want) < 1e-9*cmplx.Abs(want)) {
			t.Errorf("G22(j%g) = %v, want %v", w, g[3], want)
		}
	}
}

// TestFreqResponseInternalDelaySeriesParallelSeeded sweeps first-order
// integrators, near-integrators and lags through Series and Parallel against
// the analytic response.
func TestFreqResponseInternalDelaySeriesParallelSeeded(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	pick := func() quasiLoop {
		var a float64
		switch rng.IntN(3) {
		case 0:
		case 1:
			a = math.Pow(10, -8+4*rng.Float64())
		default:
			a = 0.1 + 3*rng.Float64()
		}
		return quasiLoop{num: []float64{(rng.Float64()*2 - 1) * 3}, den: []float64{1, a}, tau: 3 * rng.Float64()}
	}
	omega := []float64{0, 1e-9, 1e-6, 1e-3, 1}
	for c := range 200 {
		q1, q2 := pick(), pick()
		sys := seriesParallel(t, q1, q2, c%2 == 1)
		r, err := sys.FreqResponse(omega)
		if err != nil {
			t.Errorf("case %d (%+v, %+v): %v", c, q1, q2, err)
			continue
		}
		for k, w := range omega {
			got, want := r.Data[k], seriesParallelOracle(q1, q2, complex(0, w))
			if cmplx.IsInf(want) || cmplx.IsNaN(want) {
				if !cmplx.IsInf(got) {
					t.Errorf("case %d G(j%g) = %v, want Inf", c, w, got)
				}
				continue
			}
			if e := cmplx.Abs(got-want) / cmplx.Abs(want); !(e < 1e-9) {
				t.Errorf("case %d G(j%g) = %v, want %v (rel %g)", c, w, got, want, e)
			}
		}
	}
}
