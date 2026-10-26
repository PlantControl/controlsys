package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// pidFRDDenseMarginAny is the least phase margin of c on num/den, in s or
// at dt in z, from a dense log sweep over [lo, hi], hi capped below
// Nyquist: +Inf without a crossing.
func pidFRDDenseMarginAny(num, den []float64, dt float64, c PID2, lo, hi float64) float64 {
	if dt > 0 {
		hi = math.Min(hi, math.Pi/dt*(1-1e-9))
	}
	L := func(w float64) complex128 {
		x := complex(0, w)
		if dt > 0 {
			x = cmplx.Exp(complex(0, w*dt))
		}
		return pidPolyAt(num, x) / pidPolyAt(den, x) * pidTuningFeedback(c, w)
	}
	least := math.Inf(1)
	const n = 200000
	prev := cmplx.Abs(L(lo)) - 1
	for k := 1; k <= n; k++ {
		w := lo * math.Exp(float64(k)*math.Log(hi/lo)/n)
		v := cmplx.Abs(L(w)) - 1
		if prev*v < 0 || v == 0 {
			least = math.Min(least, 180-math.Abs(cmplx.Phase(L(w)))*180/math.Pi)
		}
		prev = v
	}
	return least
}

// The done-bar on certified TunePIDFRD designs (XTCC7X item 3): over
// fixed seeds of the strict, resonance, sparse and top-cut families, with
// a DC sample, every design returned without a "not certified" warning
// is stable, its reported margin at most 1.5° above the plant's, and its
// |S| peak at most 4·1.05. The sparse family is the strict one on 12-51
// samples; the top-cut one stops its band 3-30 times lower; the
// hidden-pair one 10-63 times below a lightly damped pair. XXVU4O
// (resonance 84, 237), NHV6F5 (strict 189) and QHMGQL (top-cut 402,
// hidden-pair 41) were certified violators;
// a floor on certified designs guards against certifying nothing.
// Oracles: companion roots, exact Schur–Cohn, dense margin and |S|
// sweeps of the exact plant.
func TestTunePIDFRDCertificationBar(t *testing.T) {
	skipSlowUnderRace(t)
	t.Parallel()
	n := 100
	if testing.Short() {
		n = 25
	}
	strict := func(seed uint64) pidFRDStrictCase {
		return pidFRDStrictSweepCase(rand.New(rand.NewPCG(9103, seed)))
	}
	for _, f := range []struct {
		name  string
		cases func(uint64) pidFRDStrictCase
		extra []uint64
		floor int
	}{
		{"strict", strict, []uint64{189}, n / 14},
		{"resonance", func(seed uint64) pidFRDStrictCase {
			return pidFRDResonanceSweepCase(rand.New(rand.NewPCG(2525, seed)))
		}, []uint64{237}, n / 6},
		{"sparse", func(seed uint64) pidFRDStrictCase {
			c := pidFRDStrictSweepCase(rand.New(rand.NewPCG(9106, seed)))
			c.n = 12 + rand.New(rand.NewPCG(9107, seed)).IntN(40)
			return c
		}, nil, n / 15},
		{"top-cut", pidFRDTopCutSweepCase, []uint64{402}, n / 40},
		{"hidden-pair", pidFRDHiddenPairSweepCase, nil, n / 40},
	} {
		t.Run(f.name, func(t *testing.T) {
			t.Parallel()
			seeds := make([]uint64, 0, n+len(f.extra))
			for i := range n {
				seeds = append(seeds, uint64(i))
			}
			certified := 0
			for _, seed := range append(seeds, f.extra...) {
				c := f.cases(seed)
				frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
				r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
				if err != nil || pidHasWarning(r, "not certified") {
					continue
				}
				certified++
				k := *r.Controller
				pm := r.Evidence.AchievedPhaseMargin
				least := pidFRDDenseMarginAny(c.num, c.den, c.dt, k, c.lo*1e-4, c.hi*1e5)
				peak := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo)
				if !pidFRDStrictStable(c.num, c.den, k) || pm > least+1.5 || !(peak <= 4*1.05) {
					t.Errorf("%s %d %v: certified %+v stable %v, achieved margin %g, dense %g, dense |S| %g", f.name, seed, c.family, k, pidFRDStrictStable(c.num, c.den, k), pm, least, peak)
				}
			}
			if certified < f.floor {
				t.Errorf("%s: only %d of %d designs certified", f.name, certified, n+len(f.extra))
			}
		})
	}
}
