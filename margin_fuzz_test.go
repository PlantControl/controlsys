package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzMgMaxDeg caps the oracle degree n + delay samples.
const fzMgMaxDeg = 12

// fzMgSetup builds the SISO loop of channel (0,0) of the spec, with its
// input and output delays when delays is set, and the exact oracle.
func fzMgSetup(t *testing.T, s *fzSpec, delays bool) (*fzMgLoop, *System) {
	t.Helper()
	lo, hi := math.Inf(1), 0.0
	for i, e := range s.Eig {
		if s.Discrete && cmplx.Abs(e) > 1e6 {
			fzSkip(t, "discrete pole beyond 1e6")
		}
		sp := s.S[i]
		if s.Discrete {
			sp = cmplx.Log(e) / complex(s.Dt, 0)
		}
		a := cmplx.Abs(sp)
		if a > 0 && a < 1e-8 {
			fzSkip(t, "pole below 1e-8 rad/s")
		}
		if a > 0 && math.Abs(real(sp)) < 1e-6*a {
			fzSkip(t, "pole damping below 1e-6")
		}
		if a > 0 {
			lo, hi = math.Min(lo, a), math.Max(hi, a)
		}
	}
	if hi > 1e8*lo {
		fzSkip(t, "pole magnitudes spread beyond 1e8")
	}
	n := s.N
	b, c := make([]float64, n), make([]float64, n)
	for i := range n {
		b[i], c[i] = s.B[i*s.M], s.C[i]
	}
	d := s.D[0]
	sys, err := NewFromSlices(n, 1, 1, s.A, b, c, []float64{d}, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v\n%s", err, s)
	}
	var in, out float64
	if delays && s.In != nil {
		in = s.In[0]
		if err := sys.SetInputDelay([]float64{in}); err != nil {
			t.Fatalf("SetInputDelay: %v\n%s", err, s)
		}
	}
	if delays && s.Out != nil {
		out = s.Out[0]
		if err := sys.SetOutputDelay([]float64{out}); err != nil {
			t.Fatalf("SetOutputDelay: %v\n%s", err, s)
		}
	}
	k, tau := 0, 0.0
	if s.Discrete {
		k = int(in + out)
	} else {
		tau = in + out
	}
	if n+k > fzMgMaxDeg {
		fzSkip(t, "oracle degree cap")
	}
	return fzMgNewLoop(n, s.A, b, c, d, s.Dt, k, tau), sys
}

// fzMgFromSystem builds the oracle of a delay-free or discrete-delayed SISO
// state-space model.
func fzMgFromSystem(t *testing.T, sys *System) *fzMgLoop {
	t.Helper()
	n, _, _ := sys.Dims()
	a := make([]float64, n*n)
	b, c := make([]float64, n), make([]float64, n)
	for i := range n {
		for j := range n {
			a[i*n+j] = sys.A.At(i, j)
		}
		b[i], c[i] = sys.B.At(i, 0), sys.C.At(0, i)
	}
	delay := 0.0
	for _, x := range sys.InputDelay {
		delay += x
	}
	for _, x := range sys.OutputDelay {
		delay += x
	}
	k, tau := 0, 0.0
	if sys.IsDiscrete() {
		k = int(delay)
	} else {
		tau = delay
	}
	return fzMgNewLoop(n, a, b, c, sys.D.At(0, 0), sys.Dt, k, tau)
}

func fzMgSound(cs []fzMgCross, w, f, tol float64) bool {
	for _, c := range cs {
		if fzMgMatches(c, w) {
			return true
		}
	}
	return math.Abs(f) <= tol
}

func fzMgFound(c fzMgCross, ws []float64) bool {
	for _, w := range ws {
		if fzMgMatches(c, w) {
			return true
		}
	}
	return false
}

const fzMgDB = 20 / math.Ln10

func (l *fzMgLoop) gmDB(w float64) float64 { return -fzMgDB * l.lnMag(w) }

func (l *fzMgLoop) pmDeg(w float64) float64 { return fzMgWrap(l.phase(w)+math.Pi) * 180 / math.Pi }

func (l *fzMgLoop) gmTol(w, gm float64) float64 {
	return fzMgDB*l.budget(w) + 1e-6*math.Max(1, math.Abs(gm))
}

func (l *fzMgLoop) pmTol(w float64) float64 { return l.budget(w)*180/math.Pi + 1e-6 }

func fzMgDegDiff(a, b float64) float64 {
	return math.Abs(fzMgWrap((a-b)*math.Pi/180)) * 180 / math.Pi
}

// fzMgCheckAllMargin checks soundness and completeness of every crossing
// and the margin at each against the exact loop.
func fzMgCheckAllMargin(t *testing.T, l *fzMgLoop, res *AllMarginResult, ctx string) {
	t.Helper()
	if len(res.GainMargins) != len(res.PhaseCrossFreqs) || len(res.PhaseMargins) != len(res.GainCrossFreqs) {
		t.Fatalf("AllMargin lengths GM %d / Wpc %d, PM %d / Wgc %d\n%s", len(res.GainMargins), len(res.PhaseCrossFreqs), len(res.PhaseMargins), len(res.GainCrossFreqs), ctx)
	}
	for _, ws := range [][]float64{res.GainCrossFreqs, res.PhaseCrossFreqs} {
		for _, w := range ws {
			if !(w > 0) || math.IsInf(w, 1) || w > l.nyq()*(1+1e-12) {
				t.Fatalf("AllMargin crossing ω=%v outside (0, Nyquist]: %+v\n%s", w, res, ctx)
			}
		}
		if !fzMgSorted(ws) {
			t.Fatalf("AllMargin crossings not increasing: %v\n%s", ws, ctx)
		}
	}
	gc, gzero, ok := l.gainCross(1)
	if !ok {
		fzSkip(t, "oracle roots outside float64 range")
	}
	if !gzero {
		for _, i := range fzMgSample(len(res.GainCrossFreqs)) {
			w := res.GainCrossFreqs[i]
			if !fzMgSound(gc, w, l.lnMag(w), l.budget(w)+1e-6) {
				t.Fatalf("AllMargin gain crossover ω=%v is spurious: exact |L|=%v, oracle %+v\n%s", w, math.Exp(l.lnMag(w)), gc, ctx)
			}
			if pm := l.pmDeg(w); fzMgDegDiff(res.PhaseMargins[i], pm) > l.pmTol(w) {
				t.Fatalf("AllMargin PM=%v at ω=%v, exact %v\n%s", res.PhaseMargins[i], w, pm, ctx)
			}
		}
		for _, c := range gc {
			if c.required && !fzMgFound(c, res.GainCrossFreqs) {
				t.Fatalf("AllMargin misses gain crossover ω=%v (oracle %+v), got %v\n%s", c.w, gc, res.GainCrossFreqs, ctx)
			}
		}
	}
	if l.tau > 0 {
		for _, i := range fzMgSample(len(res.PhaseCrossFreqs)) {
			w := res.PhaseCrossFreqs[i]
			if f := l.phaseOff(w); !(math.Abs(f) <= math.Pi/180+l.budget(w)) {
				t.Fatalf("AllMargin phase crossover ω=%v is spurious: exact phase offset %v°\n%s", w, f*180/math.Pi, ctx)
			}
			l.checkGM(t, w, res.GainMargins[i], ctx)
		}
		return
	}
	pc, pzero, ok := l.phaseCross()
	if !ok {
		fzSkip(t, "oracle roots outside float64 range")
	}
	if pzero {
		return
	}
	for _, i := range fzMgSample(len(res.PhaseCrossFreqs)) {
		w := res.PhaseCrossFreqs[i]
		if !fzMgSound(pc, w, l.phaseOff(w), l.budget(w)+1e-6) {
			t.Fatalf("AllMargin phase crossover ω=%v is spurious: exact phase offset %v°, oracle %+v\n%s", w, l.phaseOff(w)*180/math.Pi, pc, ctx)
		}
		l.checkGM(t, w, res.GainMargins[i], ctx)
	}
	for _, c := range pc {
		if c.required && !fzMgFound(c, res.PhaseCrossFreqs) {
			t.Fatalf("AllMargin misses phase crossover ω=%v (GM %v dB; oracle %+v), got %v\n%s", c.w, l.gmDB(c.w), pc, res.PhaseCrossFreqs, ctx)
		}
	}
}

func (l *fzMgLoop) checkGM(t *testing.T, w, got float64, ctx string) {
	t.Helper()
	if gm := l.gmDB(w); !(math.Abs(got-gm) <= l.gmTol(w, gm)) && !(math.IsInf(gm, 0) && got == gm) {
		t.Fatalf("GM=%v dB at ω=%v, exact %v\n%s", got, w, gm, ctx)
	}
}

// FuzzAllMargin: every reported gain/phase crossover of a SISO loop
// (delay-free, discrete delays absorbed, or continuous I/O delay) is a
// crossing of the exact float64 loop, every robust exact crossing is
// reported (phase crossings of continuous delayed loops excepted: the grid
// search is soundness-checked only), and the margins match the exact loop
// at the reported frequencies.
func FuzzAllMargin(f *testing.F) {
	for _, s := range fzMgSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		l, sys := fzMgSetup(t, s, true)
		res, err := AllMargin(sys)
		fzContract(t, s, "AllMargin", res != nil, err)
		fzErr(t, s, "AllMargin", err)
		fzMgCheckAllMargin(t, l, res, s.String())
	})
}

// FuzzMargin: Margin reports a sound crossing, and its |GM| and |PM| are no
// larger than at any robust exact crossing; with no crossing the margin is
// +Inf and the exact loop has no robust crossing of that kind.
func FuzzMargin(f *testing.F) {
	for _, s := range fzMgSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		l, sys := fzMgSetup(t, s, true)
		m, err := Margin(sys)
		fzContract(t, s, "Margin", m != nil, err)
		fzErr(t, s, "Margin", err)
		ctx := s.String()
		gc, gzero, ok := l.gainCross(1)
		if !ok {
			fzSkip(t, "oracle roots outside float64 range")
		}
		if w, has := m.GainCrossover(); has {
			if !gzero && !fzMgSound(gc, w, l.lnMag(w), l.budget(w)+1e-6) {
				t.Fatalf("Margin gain crossover ω=%v is spurious (oracle %+v)\n%s", w, gc, ctx)
			}
			pm := l.pmDeg(w)
			if fzMgDegDiff(m.PhaseMargin, pm) > l.pmTol(w) {
				t.Fatalf("Margin PM=%v at ω=%v, exact %v\n%s", m.PhaseMargin, w, pm, ctx)
			}
			for _, c := range gc {
				if c.required && !fzMgMatches(c, w) && math.Abs(m.PhaseMargin) > math.Abs(l.pmDeg(c.w))+l.pmTol(c.w)+l.pmTol(w)+l.spread(c, l.pmDeg) {
					t.Fatalf("Margin PM=%v at ω=%v, but crossover ω=%v has PM %v\n%s", m.PhaseMargin, w, c.w, l.pmDeg(c.w), ctx)
				}
			}
		} else {
			if !math.IsInf(m.PhaseMargin, 1) {
				t.Fatalf("Margin PM=%v without gain crossover\n%s", m.PhaseMargin, ctx)
			}
			for _, c := range gc {
				if c.required {
					t.Fatalf("Margin reports no gain crossover, oracle has ω=%v\n%s", c.w, ctx)
				}
			}
		}
		var pc []fzMgCross
		pzero := l.tau > 0
		if !pzero {
			if pc, pzero, ok = l.phaseCross(); !ok {
				fzSkip(t, "oracle roots outside float64 range")
			}
		}
		if w, has := m.PhaseCrossover(); has {
			if l.tau > 0 {
				if f := l.phaseOff(w); !(math.Abs(f) <= math.Pi/180+l.budget(w)) {
					t.Fatalf("Margin phase crossover ω=%v is spurious: offset %v°\n%s", w, f*180/math.Pi, ctx)
				}
			} else if !pzero && !fzMgSound(pc, w, l.phaseOff(w), l.budget(w)+1e-6) {
				t.Fatalf("Margin phase crossover ω=%v is spurious (oracle %+v)\n%s", w, pc, ctx)
			}
			l.checkGM(t, w, m.GainMargin, ctx)
			for _, c := range pc {
				if gm := l.gmDB(c.w); c.required && !fzMgMatches(c, w) && math.Abs(m.GainMargin) > math.Abs(gm)+l.gmTol(c.w, gm)+l.gmTol(w, m.GainMargin)+l.spread(c, l.gmDB) {
					t.Fatalf("Margin GM=%v at ω=%v, but crossover ω=%v has GM %v\n%s", m.GainMargin, w, c.w, gm, ctx)
				}
			}
		} else {
			if !math.IsInf(m.GainMargin, 1) {
				t.Fatalf("Margin GM=%v without phase crossover\n%s", m.GainMargin, ctx)
			}
			for _, c := range pc {
				if c.required && !pzero {
					t.Fatalf("Margin reports no phase crossover, oracle has ω=%v\n%s", c.w, ctx)
				}
			}
		}
	})
}

// dcValue returns L at ω = 0 exactly as num/den after cancelling common
// factors s (z − 1), so 0/1 or 1/0 for a zero or infinite DC gain, and its
// relative error budget.
func (l *fzMgLoop) dcValue() (num, den *big.Rat, budget float64) {
	at0 := func(p fzMgPoly) fzMgPoly {
		if !l.discrete {
			return p
		}
		desc := make([]*big.Rat, len(p))
		for i, x := range p {
			desc[len(p)-1-i] = x
		}
		desc = fzTaylorShift(desc, big.NewRat(1, 1))
		out := make(fzMgPoly, len(p))
		for i, x := range desc {
			out[len(p)-1-i] = x
		}
		return out
	}
	low := func(p fzMgPoly) int {
		for k, x := range p {
			if x.Sign() != 0 {
				return k
			}
		}
		return math.MaxInt
	}
	pn, pd := at0(l.num), at0(l.den)
	kn, kd := low(pn), low(pd)
	g, _ := l.rsens(0)
	switch {
	case kn > kd:
		return new(big.Rat), big.NewRat(1, 1), g
	case kn < kd:
		return big.NewRat(1, 1), new(big.Rat), g
	}
	return pn[kn], pd[kd], g
}

// fzMgCheckBandwidth checks Bandwidth(sys, dbDrop) against the first
// robust exact crossing of |L| = |L(0)|·10^(dbDrop/20).
func fzMgCheckBandwidth(t *testing.T, l *fzMgLoop, sys *System, dbDrop float64, multiInt bool, ctx string) {
	t.Helper()
	bw, err := Bandwidth(sys, dbDrop)
	num, den, dcb := l.dcValue()
	if num.Sign() == 0 || den.Sign() == 0 {
		if err == nil && multiInt {
			fzSkip(t, "Bandwidth: exact DC 0/∞ beside an integrator cluster")
		}
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("Bandwidth = %v, %v for DC gain %v/%v, want ErrInvalidArgument\n%s", bw, err, num, den, ctx)
		}
		return
	}
	if dcb > 1e-3 {
		fzSkip(t, "Bandwidth: DC gain ill-determined")
	}
	if err != nil {
		t.Fatalf("Bandwidth: undocumented error %v (exact DC %v)\n%s", err, fzMgRatio(num, den), ctx)
	}
	g := math.Abs(fzMgRatio(num, den)) * math.Pow(10, dbDrop/20)
	if g == 0 || math.IsInf(g, 0) {
		fzSkip(t, "Bandwidth: threshold outside float64 range")
	}
	cs, zero, ok := l.gainCross(g)
	if !ok || zero {
		fzSkip(t, "oracle roots outside float64 range")
	}
	if len(cs) == 0 {
		if !math.IsInf(bw, 1) {
			t.Fatalf("Bandwidth = %v, exact gain never reaches %v\n%s", bw, g, ctx)
		}
		return
	}
	first := cs[0]
	lg := math.Log(g)
	f := func(w float64) float64 { return l.lnMag(w) - lg }
	robust := func(w float64) bool { return l.gainRobust(w, g, dcb+fzMgGainTol) }
	if !l.straddles(&first, f, robust) || !(f(first.lo*(1-first.tol/2)) > 0) {
		fzSkip(t, "Bandwidth: first crossing not robust")
	}
	if fzMgMatches(first, bw) {
		return
	}
	if bw < first.lo && math.Abs(f(bw)) <= l.budget(bw)+dcb+1e-6 {
		return
	}
	t.Fatalf("Bandwidth = %v, exact first crossing of |L| = %v at ω=%v (oracle %+v)\n%s", bw, g, first.w, cs, ctx)
}

// FuzzBandwidth: Bandwidth is the first robust exact crossing of the DC gain
// lowered by dbDrop (v[15]; −3 dB below −3), +Inf when there is none, and
// ErrInvalidArgument for an exact DC gain of 0 or ∞.
func FuzzBandwidth(f *testing.F) {
	for _, s := range fzMgSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		l, sys := fzMgSetup(t, s, true)
		dbDrop := -3.0
		if v := s.Vals[15]; v >= -3 {
			dbDrop = -math.Pow(10, fzClamp(v, -2, 2))
		}
		fzMgCheckBandwidth(t, l, sys, dbDrop, fzMgIntegrators(s) > 1, s.String())
	})
}

// cl returns the closed-loop polynomial D + N, descending.
func (l *fzMgLoop) cl() []*big.Rat {
	p := fzMgAdd(l.den, l.num, 1)
	out := make([]*big.Rat, len(p))
	for i, x := range p {
		out[len(p)-1-i] = x
	}
	return out
}

// sens returns |S − c| at ω exactly rounded, S = D/(D + N); ω = +Inf is the
// continuous high-frequency limit.
func (l *fzMgLoop) sens(w, c float64) float64 {
	var nr, ni, dr, di *big.Rat
	if math.IsInf(w, 1) {
		k := len(l.den) - 1
		nr, ni, dr, di = l.num[k], new(big.Rat), l.den[k], new(big.Rat)
	} else {
		nr, ni, dr, di = l.values(w)
	}
	rc := fzRat(c)
	sr, si := new(big.Rat).Add(dr, nr), new(big.Rat).Add(di, ni)
	tr := new(big.Rat).Sub(dr, new(big.Rat).Mul(rc, sr))
	ti := new(big.Rat).Sub(di, new(big.Rat).Mul(rc, si))
	return math.Sqrt(fzMgRatio(fzMgAbs2(tr, ti), fzMgAbs2(sr, si)))
}

// fzMgPeakTol is the relative accuracy required of the disk-margin peaks:
// HinfNorm returns the ω = ∞ level of a peak that exceeds it by ~1.6e-6
// (reported), so tighter checks only rediscover that.
const fzMgPeakTol = 1e-5

// sensBudget bounds the relative error of S at ω.
func (l *fzMgLoop) sensBudget(w float64) float64 {
	if math.IsInf(w, 1) {
		return fzMgPeakTol
	}
	L := l.loop(w)
	a := cmplx.Abs(L)
	return l.budget(w)*a*(1/cmplx.Abs(1-L)+1/cmplx.Abs(1+L)) + fzMgPeakTol
}

// clRoots returns the closed-loop poles in the s-plane (float, for grids).
func (l *fzMgLoop) clRoots() []complex128 {
	p := l.cl()
	n := len(p) - 1
	if n < 1 {
		return nil
	}
	c := make([]float64, n+1)
	for i, x := range p {
		c[i], _ = x.Float64()
	}
	comp := mat.NewDense(n, n, nil)
	for j := range n {
		comp.Set(0, j, -c[j+1]/c[0])
		if j > 0 {
			comp.Set(j, j-1, 1)
		}
	}
	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		return nil
	}
	out := eig.Values(nil)
	if l.discrete {
		for i, z := range out {
			out[i] = cmplx.Log(z) / complex(l.dt, 0)
		}
	}
	return out
}

func (l *fzMgLoop) checkPeak(t *testing.T, name string, peak, w float64, ok bool, c float64, ctx string) {
	t.Helper()
	if !ok || math.IsNaN(w) || w < 0 || l.discrete && w > l.nyq()*(1+1e-12) {
		t.Fatalf("DiskMargin %s frequency %v (ok %v) for a stable loop\n%s", name, w, ok, ctx)
	}
	if got := l.sens(w, c); got < peak*(1-l.sensBudget(w)) {
		t.Fatalf("DiskMargin %s peak %v at ω=%v, exact value there %v\n%s", name, peak, w, got, ctx)
	}
	f := func(w float64) float64 { return l.sens(w, c) }
	wbest, _ := fzSweepMax(f, fzFreqGrid(l.clRoots(), l.dt, 80), 3)
	if v := f(wbest); v > peak*(1+l.sensBudget(wbest)) {
		t.Fatalf("DiskMargin %s peak %v, exact value at ω=%v is %v\n%s", name, peak, wbest, v, ctx)
	}
	if !l.discrete {
		if v := l.sens(math.Inf(1), c); v > peak*(1+1e-6) {
			t.Fatalf("DiskMargin %s peak %v below its ω→∞ limit %v\n%s", name, peak, v, ctx)
		}
	}
}

// FuzzDiskMargin: on delay-free SISO loops DiskMargin's stability verdict
// matches an exact root count of D + N, and for a stable loop Alpha and
// PeakSensitivity are the exact peaks of |S − ½| and |S| (attained at the
// returned frequencies, not exceeded on a closed-loop-pole-clustered sweep),
// with GainMargin and PhaseMargin the disk intercepts of Alpha.
func FuzzDiskMargin(f *testing.F) {
	for _, s := range fzMgSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		l, sys := fzMgSetup(t, s, false)
		ctx := s.String()
		if math.Abs(1+s.D[0]) < 1e-8 {
			fzSkip(t, "DiskMargin: 1 + D ≈ 0")
		}
		dm, err := DiskMargin(sys)
		fzContract(t, s, "DiskMargin", dm != nil, err)
		fzErr(t, s, "DiskMargin", err)
		cl := l.cl()
		const delta = 1e-5
		var stable, unstable bool
		if l.discrete {
			k1, ok1 := fzSchurCohnOutside(cl, 1-delta)
			k2, ok2 := fzSchurCohnOutside(cl, 1+delta)
			stable, unstable = ok1 && k1 == 0, ok2 && k2 > 0
		} else {
			bound := 1.0
			for _, x := range cl[1:] {
				v, _ := new(big.Rat).Quo(x, cl[0]).Float64()
				bound = math.Max(bound, 1+math.Abs(v))
			}
			k1, ok1 := fzRouthRight(cl, -delta*bound)
			k2, ok2 := fzRouthRight(cl, delta*bound)
			stable, unstable = ok1 && k1 == 0, ok2 && k2 > 0
		}
		switch {
		case unstable:
			if dm.Alpha != 0 || !math.IsInf(dm.PeakSensitivity, 1) {
				t.Fatalf("DiskMargin Alpha=%v Ms=%v for an exactly unstable closed loop\n%s", dm.Alpha, dm.PeakSensitivity, ctx)
			}
			return
		case !stable:
			fzSkip(t, "DiskMargin: closed-loop pole near the stability boundary")
		}
		if !(dm.Alpha > 0) || math.IsInf(dm.PeakSensitivity, 0) {
			t.Fatalf("DiskMargin Alpha=%v Ms=%v for an exactly stable closed loop\n%s", dm.Alpha, dm.PeakSensitivity, ctx)
		}
		w, ok := dm.Frequency()
		l.checkPeak(t, "|S−½|", 1/dm.Alpha, w, ok, 0.5, ctx)
		w, ok = dm.PeakFreq()
		l.checkPeak(t, "|S|", dm.PeakSensitivity, w, ok, 0, ctx)
		a := dm.Alpha
		lo, hi := max((2-a)/(2+a), 0), math.Inf(1)
		if a < 2 {
			hi = (2 + a) / (2 - a)
		}
		pm := 2 * math.Atan(a/2) * 180 / math.Pi
		near := func(x, y float64) bool { return x == y || math.Abs(x-y) <= 1e-9*math.Max(1, math.Abs(y)) }
		if !near(dm.GainMargin[0], lo) || !near(dm.GainMargin[1], hi) || !near(dm.PhaseMargin, pm) {
			t.Fatalf("DiskMargin GM %v PM %v for Alpha %v, want [%v %v] %v\n%s", dm.GainMargin, dm.PhaseMargin, a, lo, hi, pm, ctx)
		}
	})
}

func fzMgSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	nyq := func(dt float64) float64 { return math.Log10(math.Pi / dt) }
	return []fzSeed{
		// ZQL2JH: −2·z⁻⁵ and −2·z⁻⁸ (phase crossovers at multiples of 2π/5, π/4).
		with(func(s *fzSeed) { s.Discrete, s.DScal, s.Phase, s.In = true, -2, -3.3, math.Log10(5) }),
		with(func(s *fzSeed) { s.Discrete, s.DScal, s.Phase, s.In = true, -2, -3.3, math.Log10(8) }),
		// JPZZRT: small gain over a real pole pair next to z = 1.
		with(func(s *fzSeed) { s.Discrete, s.N, s.Mag[0], s.Damp[0], s.Gain = true, 2, -4, 0.5, 1 }),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.Mag[0], s.Damp[0], s.Mag[1], s.Gain = true, 3, -3, 0.8, -3.2, 0.5
		}),
		// 5RWUVZ, Y3RHVQ: four poles next to z = −1, negative small gain.
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.Gain = true, 4, -0.2
			s.Mag, s.Damp = [3]float64{nyq(1), nyq(1)}, [3]float64{7.9, 7.5}
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.Gain = true, -1, 5, -0.1
			s.Mag, s.Damp = [3]float64{nyq(0.1), nyq(0.1), 1.2}, [3]float64{7.9, 7.5, 1}
		}),
		// 6WSXWR: rounded poles next to z = 1, Dt = 0.042.
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.Gain, s.Shear = true, math.Log10(0.0422252269048428), 4, 0.3, -2
			s.Mag, s.Damp = [3]float64{-1, -1.5}, [3]float64{1.5, 0.7}
		}),
		// UFBPWZ, VT3OF7, UJSGC2: lightly damped pair 1 ± jε with real poles.
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.Gain = true, 5, -1.6
			s.Mag, s.Damp = [3]float64{math.Log10(2.903e-5), -1.2, 0.1}, [3]float64{7.9, 1.5, 0}
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.Gain = true, -1, 3, -3
			s.Mag, s.Damp = [3]float64{math.Log10(7.11e-5), 0.5}, [3]float64{7.9, 0}
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.Gain, s.Phase = true, 4, -6.3, 0.6
			s.Mag, s.Damp = [3]float64{math.Log10(1.47e-5), -0.5}, [3]float64{7.5, 1.2}
		}),
		// 42ZG7L: continuous resonance, ζ raised from 1.2e-8 to 1.2e-6.
		with(func(s *fzSeed) {
			s.N, s.Gain, s.Mag[0], s.Damp[0] = 2, -6.7, math.Log10(14.4809428364274), 2-math.Log10(1.2e-6)
		}),
		with(func(s *fzSeed) {
			s.N, s.Gain, s.Mag[0], s.Damp[0], s.Mag[1] = 3, 8.6, 0.01, 2-math.Log10(4e-3), -0.5
		}),
		// 42XIJD: small DC gain over (s+1)(s+2).
		with(func(s *fzSeed) {
			s.N, s.Gain, s.Mag[0], s.Damp[0] = 2, -4, math.Log10(2)/2, 2-math.Log10(1.5/math.Sqrt2)
		}),
		with(func(s *fzSeed) { s.N, s.Gain, s.Mag[0], s.Damp[0] = 3, 0, 0.1, 1.5 }),
		// DSRJ7M: integrator with a lead.
		with(func(s *fzSeed) { s.N, s.Gain, s.Mag[0], s.Damp[0], s.Mag[1] = 3, -5.5, 0.2, 1.8, -40 }),
		// Continuous loops with I/O delay; discrete with delay and feedthrough.
		with(func(s *fzSeed) { s.N, s.Gain, s.Mag[0], s.Damp[0], s.In = 2, 6.5, 0, 1.7, -0.5 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.Gain, s.DScal, s.In = true, -0.5, 3, 6.2, 0.4, math.Log10(3)
			s.Mag, s.Damp = [3]float64{0.3, -0.2}, [3]float64{1.5, 0}
		}),
		with(func(s *fzSeed) {
			s.N, s.Gain, s.DScal, s.Shear, s.Rot, s.Phase = 6, 7, 0.3, -1, 0.8, 0.4
			s.Mag, s.Damp = [3]float64{-1, 0.3, 2}, [3]float64{3, 5, 0.5}
		}),
	}
}

// TestFzMgRepros replays the fixed margin repros (exact coefficients from
// their reports) through the oracle and the AllMargin/Bandwidth checks; the
// oracle must reproduce the crossings the reports document.
func TestFzMgRepros(t *testing.T) {
	cases := []struct {
		name       string
		num, den   []float64
		dt         float64
		delay      float64
		wantG      []float64
		wantP      []float64
		bandwidth  float64
		checkBandW bool
	}{
		{name: "ZQL2JH -2z^-5", num: []float64{-2}, den: []float64{1, 0, 0, 0, 0, 0}, dt: 1, wantP: []float64{1.2566370614359172, 2.5132741228718345}},
		{name: "ZQL2JH -2z^-4 delay", num: []float64{-2}, den: []float64{1}, dt: 1, delay: 4, wantP: []float64{math.Pi / 2, math.Pi}},
		{name: "ZQL2JH -2z^-8", num: []float64{-2}, den: []float64{1, 0, 0, 0, 0, 0, 0, 0, 0}, dt: 1, wantP: []float64{math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4, math.Pi}},
		{name: "JPZZRT k/(z-1)^2", num: []float64{3.1623e-8}, den: []float64{1, -2, 1}, dt: 1, wantG: []float64{1.7783e-4}},
		{name: "JPZZRT k/(z-1)^3", num: []float64{3.1623e-8}, den: []float64{1, -3, 3, -1}, dt: 0.1, wantG: []float64{3.1623e-2}},
		{name: "5RWUVZ", num: []float64{-1e-6}, den: []float64{1, 4, 6, 4, 1}, dt: 1, wantG: []float64{3.10997}},
		{name: "5RWUVZ k>0", num: []float64{1e-6}, den: []float64{1, 4, 6, 4, 1}, dt: 1, wantG: []float64{3.10997}, wantP: []float64{math.Pi / 2}},
		{name: "6WSXWR", num: []float64{0.5296264797589852, -0.49398687205654906}, den: []float64{1, -3.927287723431127, 5.782240810121571, -3.7826184499497617, 0.9276653632593171}, dt: 0.0422252269048428},
		{name: "42ZG7L", num: []float64{-4.998581848916508, 0.7194337044136604}, den: []float64{1, 3.4849223836704207e-07, 209.69770538172128}, wantG: []float64{12.1956, 17.1944}, wantP: []float64{14.480942836427431}},
		{name: "VT3OF7", num: []float64{-0.0013196134305395058, -0.00023536903292695072}, den: []float64{1, -2.3854598907075455, 1.770919781465705, -0.3854598907270551}, dt: 0.1, wantP: []float64{4.707087894e-05, 10 * math.Pi}},
		{name: "UJSGC2", num: []float64{-2.166935933035251, 1.9807432626284676}, den: []float64{1, -2.73127553541105, 2.2142827468764708, -0.23473888724605785, -0.2482683242149491}, dt: 1, wantP: []float64{1.45446e-05, 1.6806}},
		{name: "DSRJ7M", num: []float64{-0.032012382534863985, -0.052223340965159556}, den: []float64{1, 0, 0}},
		{name: "42XIJD 1e-4", num: []float64{1e-4}, den: []float64{1, 3, 2}, bandwidth: 0.8358236303817804, checkBandW: true},
		{name: "42XIJD 1e-12", num: []float64{1e-12}, den: []float64{1, 3, 2}, bandwidth: 0.8358236303817804, checkBandW: true},
		{name: "42XIJD cubic 1e-8", num: []float64{1e-8}, den: []float64{1, 6, 11, 6}, bandwidth: 0.78579, checkBandW: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
			if err != nil {
				t.Fatal(err)
			}
			sys := r.Sys
			if c.delay > 0 {
				if err := sys.SetInputDelay([]float64{c.delay}); err != nil {
					t.Fatal(err)
				}
			}
			l := fzMgFromSystem(t, sys)
			match := func(kind string, want []float64, cs []fzMgCross) {
				for _, w := range want {
					found := false
					for _, x := range cs {
						if x.required && math.Abs(x.w-w) <= 1e-4*w {
							found = true
						}
					}
					if !found {
						t.Fatalf("oracle %s crossings %+v lack required %v", kind, cs, w)
					}
				}
			}
			gc, _, ok := l.gainCross(1)
			if !ok {
				t.Fatal("oracle range")
			}
			match("gain", c.wantG, gc)
			pc, _, ok := l.phaseCross()
			if !ok {
				t.Fatal("oracle range")
			}
			match("phase", c.wantP, pc)
			if c.checkBandW {
				num, den, _ := l.dcValue()
				cs, _, _ := l.gainCross(math.Abs(fzMgRatio(num, den)) * math.Pow(10, -3.0/20))
				if len(cs) == 0 || math.Abs(cs[0].w-c.bandwidth) > 1e-4*c.bandwidth {
					t.Fatalf("oracle bandwidth crossings %+v, want first %v", cs, c.bandwidth)
				}
				fzMgCheckBandwidth(t, l, sys, -3, false, c.name)
				return
			}
			res, err := AllMargin(sys)
			if err != nil {
				t.Fatal(err)
			}
			fzMgCheckAllMargin(t, l, res, c.name)
		})
	}
}

// fzMgSample returns the indices to check of n reported crossings: all of
// them, or the first and last 32 and a stride between when a delayed loop
// reports thousands.
func fzMgSample(n int) []int {
	var out []int
	step := max(1, (n-64)/64)
	for i := range n {
		if i < 32 || i >= n-32 || i%step == 0 {
			out = append(out, i)
		}
	}
	return out
}

// spread is how much f moves across the location uncertainty of c.
func (l *fzMgLoop) spread(c fzMgCross, f func(float64) float64) float64 {
	t := max(c.tol, fzMgMatch)
	v := f(c.w)
	s := 0.0
	for _, w := range []float64{c.lo * (1 - t), c.hi * (1 + t)} {
		if !l.discrete || w <= l.nyq() {
			s = math.Max(s, math.Abs(f(w)-v))
		}
	}
	return s
}

// fzMgIntegrators counts intended poles at s = 0 (z = 1).
func fzMgIntegrators(s *fzSpec) int {
	k := 0
	for _, e := range s.S {
		if e == 0 {
			k++
		}
	}
	return k
}
