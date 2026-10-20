package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Time-domain fuzz targets (Step, Impulse, Initial, Lsim + GenSig, StepInfo).
//
// Oracle: the modal form fzDecode builds, A ≈ T·J·T⁻¹ with J real
// block-diagonal from the intended spectrum. Continuous responses use the
// closed-form e^{Jt} and ∫₀ᵗe^{Jτ}dτ per block (no matrix exponential is
// shared with the library); discrete ones a modal recurrence ξ ← J·ξ + T⁻¹b.
// Delays enter as exact time shifts (fractional continuous input delays
// through the split ZOH step). Tolerances scale with the running maximum of
// the term magnitudes and with condT²·steps (library recurrence) plus
// condT³·‖J‖·t (float64 A versus T·J·T⁻¹); ill-conditioned similarities
// (condT > 1e4) and overflowing trajectories are skipped.

const fzTRMaxCond = 1e4

type fzBlk struct {
	k, sz int
	a, b  float64
}

// fzModal holds T, T⁻¹ and the blocks of J (continuous: a ± ib of s;
// discrete: the entries of J).
type fzModal struct {
	n      int
	T, Ti  *mat.Dense
	blocks []fzBlk
}

func fzModalOf(t *testing.T, s *fzSpec) *fzModal {
	t.Helper()
	n := s.N
	md := &fzModal{n: n}
	if n == 0 {
		return md
	}
	shear := 0.0
	if s.Vals[9] > -9 {
		shear = math.Pow(10, fzClamp(s.Vals[9], -8, 0.5))
	}
	theta := fzClamp(s.Vals[10], -4, 4)
	T := mat.NewDense(n, n, nil)
	for i := range n {
		T.Set(i, i, 1)
		for j := i + 1; j < n; j++ {
			T.Set(i, j, shear)
		}
	}
	for i := n - 2; i >= 0; i-- {
		c, sn := math.Cos(theta*float64(i+1)), math.Sin(theta*float64(i+1))
		for j := range n {
			a, b := T.At(i, j), T.At(i+1, j)
			T.Set(i, j, c*a-sn*b)
			T.Set(i+1, j, sn*a+c*b)
		}
	}
	var Ti mat.Dense
	if err := Ti.Inverse(T); err != nil {
		fzSkip(t, "singular similarity")
	}
	md.T, md.Ti = T, &Ti
	for k := 0; k < n; {
		e := s.S[k]
		if s.Discrete {
			e = s.Eig[k]
		}
		if imag(s.S[k]) != 0 {
			md.blocks = append(md.blocks, fzBlk{k, 2, real(e), math.Abs(imag(e))})
			k += 2
			continue
		}
		md.blocks = append(md.blocks, fzBlk{k, 1, real(e), 0})
		k++
	}
	return md
}

// row returns c·T, col T⁻¹·b.
func (md *fzModal) row(c []float64) []float64 {
	r := make([]float64, md.n)
	for j := range md.n {
		for k := range md.n {
			r[j] += c[k] * md.T.At(k, j)
		}
	}
	return r
}

func (md *fzModal) col(b []float64) []float64 {
	w := make([]float64, md.n)
	for i := range md.n {
		for k := range md.n {
			w[i] += md.Ti.At(i, k) * b[k]
		}
	}
	return w
}

// blkExp returns (Re, Im) of e^{λt}; blkInt of ∫₀ᵗ e^{λτ}dτ.
func blkExp(bl fzBlk, t float64) (float64, float64) {
	if bl.sz == 1 {
		return math.Exp(bl.a * t), 0
	}
	e := math.Exp(bl.a * t)
	return e * math.Cos(bl.b*t), e * math.Sin(bl.b*t)
}

func blkInt(bl fzBlk, t float64) (float64, float64) {
	if bl.sz == 1 {
		if bl.a == 0 {
			return t, 0
		}
		return math.Expm1(bl.a*t) / bl.a, 0
	}
	sh := math.Sin(bl.b * t / 2)
	em1 := complex(math.Expm1(bl.a*t)*math.Cos(bl.b*t)-2*sh*sh, math.Exp(bl.a*t)*math.Sin(bl.b*t))
	q := em1 / complex(bl.a, bl.b)
	return real(q), imag(q)
}

// blkApply returns F·v on the block, F = [[f1, f2], [−f2, f1]] (f1 for a
// scalar block), and the same with absolute values.
func blkApply(bl fzBlk, f1, f2 float64, v []float64, out, mag []float64) {
	k := bl.k
	if bl.sz == 1 {
		out[k] = f1 * v[k]
		if mag != nil {
			mag[k] = math.Abs(f1 * v[k])
		}
		return
	}
	x, y := v[k], v[k+1]
	out[k], out[k+1] = f1*x+f2*y, -f2*x+f1*y
	if mag != nil {
		mag[k] = math.Abs(f1*x) + math.Abs(f2*y)
		mag[k+1] = math.Abs(f2*x) + math.Abs(f1*y)
	}
}

// ctEval returns r·F(t)·w and its magnitude, F = e^{Jt} or ∫₀ᵗe^{Jτ}dτ.
func (md *fzModal) ctEval(integral bool, r, w []float64, t float64) (float64, float64) {
	out := make([]float64, md.n)
	mag := make([]float64, md.n)
	for _, bl := range md.blocks {
		var f1, f2 float64
		if integral {
			f1, f2 = blkInt(bl, t)
		} else {
			f1, f2 = blkExp(bl, t)
		}
		blkApply(bl, f1, f2, w, out, mag)
	}
	var v, m float64
	for k := range md.n {
		v += r[k] * out[k]
		m += math.Abs(r[k]) * mag[k]
	}
	return v, m
}

// dStep advances ξ ← J·ξ (discrete) in place.
func (md *fzModal) dStep(xi []float64) {
	out := make([]float64, md.n)
	for _, bl := range md.blocks {
		blkApply(bl, bl.a, bl.b, xi, out, nil)
	}
	copy(xi, out)
}

func fzDot(a, b []float64) (float64, float64) {
	var v, m float64
	for k := range a {
		v += a[k] * b[k]
		m += math.Abs(a[k] * b[k])
	}
	return v, m
}

// fzTRTol is the relative tolerance at sample k, time tk.
func fzTRTol(s *fzSpec, k int, dt, tk float64) float64 {
	const eps = 0x1p-52
	c := s.CondT
	nj := math.Max(s.NormJ, 1)
	if s.Discrete {
		return 1e-9 + 1e3*eps*c*c*c*float64(k+1)*nj
	}
	return 1e-9 + 1e3*eps*(c*c*float64(k+1)*(1+s.NormJ*dt)+c*c*c*s.NormJ*tk)
}

// fzTRCompare checks got against want sample by sample with tolerance
// tol(k)·max(runmax(mag), max(mag)/100) plus a model-scale floor (roundoff leaking into the
// zero head of a delayed channel); nil want samples (NaN) are not compared.
func fzTRCompare(t *testing.T, s *fzSpec, op string, row int, T, got, want, mag []float64) {
	t.Helper()
	fzTROverflow(t, s, T)
	dt := 0.0
	if len(T) > 1 {
		dt = T[1] - T[0]
	}
	run, top := 0.0, 0.0
	floor := 1e-12*(fzFrob(s.C)*(fzFrob(s.B)+1)*math.Max(dt, 1)+fzFrob(s.D)) + 1e-300
	for _, m := range mag {
		if math.IsInf(m, 0) || math.IsNaN(m) || m > 1e250 {
			fzSkip(t, "oracle trajectory overflows")
		}
		top = math.Max(top, m)
	}
	for k := range got {
		run = math.Max(run, mag[k])
		if math.IsNaN(want[k]) {
			continue
		}
		tol := fzTRTol(s, k, dt, T[k])*math.Max(run, 1e-2*top) + floor
		if !(math.Abs(got[k]-want[k]) <= tol) {
			t.Fatalf("%s row %d sample %d (t=%g): got %v, oracle %v (|Δ|=%.3g tol %.3g, scale %.3g)\n%s",
				op, row, k, T[k], got[k], want[k], math.Abs(got[k]-want[k]), tol, run, s)
		}
	}
}

// fzTROverflow skips horizons over which an unstable mode overflows the
// state: delayed outputs then read Inf·0 = NaN where the exact response is
// still zero or finite.
func fzTROverflow(t *testing.T, s *fzSpec, T []float64) {
	t.Helper()
	grow := 0.0
	for k, e := range s.Eig {
		if s.Discrete {
			grow = math.Max(grow, float64(len(T))*math.Log(cmplx.Abs(e)))
		} else {
			grow = math.Max(grow, real(s.S[k])*T[len(T)-1])
		}
	}
	if grow > 600 {
		fzSkip(t, "unstable state overflows on the horizon")
	}
}

func fzChanDelay(s *fzSpec, i, j int) float64 {
	d := 0.0
	if s.In != nil {
		d += s.In[j]
	}
	if s.Out != nil {
		d += s.Out[i]
	}
	if s.IO != nil {
		d += s.IO[i*s.M+j]
	}
	return d
}

func fzBcol(s *fzSpec, j int) []float64 {
	b := make([]float64, s.N)
	for k := range s.N {
		b[k] = s.B[k*s.M+j]
	}
	return b
}

// fzTRGrid validates the response time grid.
func fzTRGrid(t *testing.T, s *fzSpec, op string, resp *TimeResponse, tFinal float64) {
	t.Helper()
	T := resp.T
	if len(T) < 2 {
		t.Fatalf("%s: %d samples\n%s", op, len(T), s)
	}
	if r, c := resp.Y.Dims(); c != len(T) || r == 0 {
		t.Fatalf("%s: Y is %d×%d for %d samples\n%s", op, r, c, len(T), s)
	}
	if T[0] != 0 {
		t.Fatalf("%s: T[0] = %v\n%s", op, T[0], s)
	}
	dt := T[1]
	for k := 1; k < len(T); k++ {
		if !(math.Abs(T[k]-float64(k)*dt) <= 1e-9*float64(k)*dt) {
			t.Fatalf("%s: non-uniform grid T[%d] = %v, dt %v\n%s", op, k, T[k], dt, s)
		}
	}
	if s.Discrete {
		if dt != s.Dt {
			t.Fatalf("%s: grid step %v, Dt %v\n%s", op, dt, s.Dt, s)
		}
		if tFinal > 0 {
			if want := int(math.Floor(tFinal/s.Dt+1e-9)) + 1; len(T) != want {
				t.Fatalf("%s: %d samples for tFinal %v, want %d\n%s", op, len(T), tFinal, want, s)
			}
		}
		return
	}
	if tFinal > 0 && T[len(T)-1] != tFinal {
		t.Fatalf("%s: last sample %v, tFinal %v\n%s", op, T[len(T)-1], tFinal, s)
	}
}

// fzStdOracle returns the oracle step (impulse=false) or impulse response of
// channel (i, j) on grid T with the spec's I/O delays, and term magnitudes.
// Samples whose delayed time is within roundoff of a jump are NaN.
func fzStdOracle(s *fzSpec, md *fzModal, impulse bool, i, j int, T []float64) (want, mag []float64) {
	K := len(T)
	want, mag = make([]float64, K), make([]float64, K)
	d := s.D[i*s.M+j]
	var r, w []float64
	if s.N > 0 {
		r, w = md.row(s.C[i*s.N:(i+1)*s.N]), md.col(fzBcol(s, j))
	}
	tau := fzChanDelay(s, i, j)
	if s.Discrete {
		k0 := int(math.Round(tau))
		xi := make([]float64, s.N)
		for k := range K {
			kk := k - k0
			switch {
			case kk < 0:
				continue
			case impulse && kk == 0:
				want[k], mag[k] = d, math.Abs(d)
				continue
			}
			if impulse {
				if kk == 1 {
					copy(xi, w)
				} else {
					md.dStep(xi)
				}
			} else if kk > 0 {
				md.dStep(xi)
				for q := range xi {
					xi[q] += w[q]
				}
			}
			v, m := fzDot(r, xi)
			if !impulse {
				v, m = v+d, m+math.Abs(d)
			}
			want[k], mag[k] = v, m
		}
		return want, mag
	}
	dt := T[1]
	jump := math.Abs(d)
	if impulse && s.N > 0 {
		jump, _ = fzDot(r, w)
		jump = math.Abs(jump)
	}
	for k := range K {
		tp := T[k] - tau
		if math.Abs(tp) <= 1e-7*dt && jump != 0 && tau > 0 {
			want[k] = math.NaN()
			continue
		}
		if tp < 0 {
			continue
		}
		var v, m float64
		if s.N > 0 {
			v, m = md.ctEval(!impulse, r, w, tp)
		}
		if !impulse {
			v, m = v+d, m+math.Abs(d)
		}
		want[k], mag[k] = v, m
	}
	return want, mag
}

// fzMetric is an independent StepInfo on one sampled row.
type fzMetric struct {
	over, under, peak, peakT, final float64
	rise, settle                    float64
	risen, settled                  bool
	lastOut                         int
}

func fzStepMetric(T, y []float64, final float64, useFinal bool, lims [2]float64, thr float64) fzMetric {
	K := len(y)
	y0 := 0.0
	if !useFinal {
		final = y[K-1]
	}
	delta := final - y0
	scale := math.Abs(delta)
	if scale == 0 {
		scale = math.Max(math.Abs(final), 1)
	}
	dir := 1.0
	if delta < 0 {
		dir = -1
	}
	m := fzMetric{final: final, peak: y[0], peakT: T[0], lastOut: -1}
	worst := 0.0
	for k := range K {
		if dir*y[k] > dir*m.peak {
			m.peak, m.peakT = y[k], T[k]
		}
		worst = math.Max(worst, dir*(y0-y[k]))
		if math.Abs(y[k]-final) > thr*scale {
			m.lastOut = k
		}
	}
	if b := dir * (m.peak - final); b > 0 && delta != 0 {
		m.over = 100 * b / math.Abs(delta)
	}
	m.under = 100 * worst / scale
	cross := func(lev float64) (float64, bool) {
		if dir*(y[0]-lev) >= 0 {
			return T[0], true
		}
		for k := 1; k < K; k++ {
			if dir*(y[k]-lev) >= 0 {
				if y[k] == y[k-1] {
					return T[k], true
				}
				return T[k-1] + (lev-y[k-1])/(y[k]-y[k-1])*(T[k]-T[k-1]), true
			}
		}
		return 0, false
	}
	if delta != 0 {
		lo, okL := cross(y0 + delta*lims[0])
		hi, okH := cross(y0 + delta*lims[1])
		if okL && okH {
			m.rise, m.risen = hi-lo, true
		}
	}
	switch m.lastOut {
	case -1:
		m.settle, m.settled = T[0], true
	case K - 1:
	default:
		m.settle, m.settled = T[m.lastOut+1], true
	}
	return m
}

func fzClose(a, b float64) bool {
	return a == b || math.Abs(a-b) <= 1e-9*math.Max(math.Abs(a), math.Abs(b))+1e-300
}

// fzCheckStepInfo compares StepInfo on resp against fzStepMetric.
func fzCheckStepInfo(t *testing.T, s *fzSpec, resp *TimeResponse, ss []float64) {
	t.Helper()
	var opts *StepInfoOptions
	if ss != nil {
		opts = &StepInfoOptions{SteadyStateValue: ss}
	}
	info, err := StepInfo(resp, opts)
	fzContract(t, s, "StepInfo", info != nil, err)
	fzErr(t, s, "StepInfo", err)
	rows, _ := resp.Y.Dims()
	if len(info.Metrics) != rows {
		t.Fatalf("StepInfo: %d metrics for %d rows\n%s", len(info.Metrics), rows, s)
	}
	for row := range rows {
		y := mat.Row(nil, row, resp.Y)
		if !fzFinite(y) {
			continue
		}
		final := 0.0
		if ss != nil {
			final = ss[row]
		}
		w := fzStepMetric(resp.T, y, final, ss != nil, [2]float64{0.1, 0.9}, 0.02)
		g := info.Metrics[row]
		rt, rok := g.RiseTime()
		st, sok := g.SettlingTime()
		if !fzClose(g.Overshoot, w.over) || !fzClose(g.Undershoot, w.under) || g.Peak != w.peak ||
			g.PeakTime != w.peakT || g.SteadyStateValue != w.final || rok != w.risen || sok != w.settled ||
			(rok && !fzClose(rt, w.rise)) || (sok && st != w.settle) {
			t.Fatalf("StepInfo row %d = %+v rise=(%v,%v) settle=(%v,%v), direct %+v\n%s", row, g, rt, rok, st, sok, w, s)
		}
	}
}

// FuzzStepImpulse: Step and Impulse (explicit horizon from the Tau field,
// or automatic) sampled exactly against the modal closed form, with I/O
// delays as exact shifts; StepInfo matches a direct computation on the
// returned rows; for decidably stable models StepInfoForSystem uses
// DCGain as steady state, its settling verdict is consistent with the
// oracle trajectory, and the final sample of the automatic horizon is
// within the modal transient bound of the exact DC gain. Decidably
// unstable models make StepInfoForSystem return ErrUnstable.
func FuzzStepImpulse(f *testing.F) {
	for _, s := range fzTRSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.CondT > fzTRMaxCond {
			fzSkip(t, "ill-conditioned similarity")
		}
		md := fzModalOf(t, s)
		sys := s.System(t, fzSysOpts{Delays: true})
		tFinal := s.Tau
		if s.Discrete {
			tFinal *= s.Dt
		}
		for _, impulse := range []bool{false, true} {
			op := "Step"
			resp, err := Step(sys, tFinal)
			if impulse {
				op = "Impulse"
				resp, err = Impulse(sys, tFinal)
			}
			fzContract(t, s, op, resp != nil, err)
			fzErr(t, s, op, err)
			fzTRGrid(t, s, op, resp, tFinal)
			if r, _ := resp.Y.Dims(); r != s.M*s.P {
				t.Fatalf("%s: %d rows, want %d\n%s", op, r, s.M*s.P, s)
			}
			for j := range s.M {
				for i := range s.P {
					row := j*s.P + i
					want, mag := fzStdOracle(s, md, impulse, i, j, resp.T)
					fzTRCompare(t, s, op, row, resp.T, mat.Row(nil, row, resp.Y), want, mag)
				}
			}
			if !impulse {
				fzCheckStepInfo(t, s, resp, nil)
				fzStepInfoSystem(t, s, md, sys, resp, tFinal)
			}
		}
	})
}

func fzStepInfoSystem(t *testing.T, s *fzSpec, md *fzModal, sys *System, resp *TimeResponse, tFinal float64) {
	t.Helper()
	if !s.Decidable() {
		return
	}
	info, err := StepInfoForSystem(sys, tFinal, nil)
	if s.Unstable() > 0 {
		if !errors.Is(err, ErrUnstable) {
			t.Fatalf("StepInfoForSystem on an unstable model: err %v\n%s", err, s)
		}
		return
	}
	fzContract(t, s, "StepInfoForSystem", info != nil, err)
	fzErr(t, s, "StepInfoForSystem", err)
	x0 := 0.0
	if s.Discrete {
		x0 = 1
	}
	var dc []complex128
	if s.N > 0 {
		var ok bool
		if dc, ok = fzExactResponse(s.N, s.M, s.P, s.A, s.B, s.C, s.D, complex(x0, 0)); !ok {
			fzSkip(t, "singular DC solve")
		}
	}
	ss := make([]float64, s.M*s.P)
	for j := range s.M {
		for i := range s.P {
			ss[j*s.P+i] = s.D[i*s.M+j]
			if s.N > 0 {
				ss[j*s.P+i] = real(dc[i*s.M+j])
			}
		}
	}
	gain, err := sys.DCGain()
	fzErr(t, s, "DCGain", err)
	K := len(resp.T)
	for j := range s.M {
		for i := range s.P {
			row := j*s.P + i
			g := info.Metrics[row]
			if g.SteadyStateValue != gain.At(i, j) {
				t.Fatalf("StepInfoForSystem row %d steady state %v, DCGain %v\n%s", row, g.SteadyStateValue, gain.At(i, j), s)
			}
			want, mag := fzStdOracle(s, md, false, i, j, resp.T)
			run := 0.0
			for k := range K {
				run = math.Max(run, mag[k])
			}
			tol := fzTRTol(s, K, resp.T[1], resp.T[K-1])*run + 1e-300
			if math.IsNaN(want[0]) {
				continue
			}
			scale := math.Abs(ss[row])
			if scale <= 10*tol {
				continue
			}
			band := 0.02 * scale
			lastOut := -1
			if st, ok := g.SettlingTime(); ok {
				for k := range K {
					if resp.T[k] < st {
						lastOut = k
					}
				}
			} else {
				lastOut = K - 1
			}
			for k := lastOut + 1; k < K; k++ {
				if !math.IsNaN(want[k]) && math.Abs(want[k]-ss[row]) > band*(1+1e-6)+2*tol {
					t.Fatalf("StepInfoForSystem row %d settled at sample %d but oracle sample %d is %v, outside %v±%v\n%s",
						row, lastOut+1, k, want[k], ss[row], band, s)
				}
			}
			if lastOut >= 0 && !math.IsNaN(want[lastOut]) && math.Abs(want[lastOut]-ss[row]) < band*(1-1e-6)-2*tol {
				t.Fatalf("StepInfoForSystem row %d: sample %d (%v) counted outside the band %v±%v\n%s",
					row, lastOut, want[lastOut], ss[row], band, s)
			}
			if tFinal != 0 || s.N == 0 || s.Discrete {
				continue
			}
			// y(T) − DC = −r·e^{JT}·J⁻¹·w, bounded by the modal magnitudes.
			r, w := md.row(s.C[i*s.N:(i+1)*s.N]), md.col(fzBcol(s, j))
			tEnd := resp.T[K-1] - fzChanDelay(s, i, j)
			bound := 0.0
			for _, bl := range md.blocks {
				e1, e2 := blkExp(bl, tEnd)
				amp := math.Hypot(e1, e2) / math.Hypot(bl.a, bl.b)
				for q := bl.k; q < bl.k+bl.sz; q++ {
					for p := bl.k; p < bl.k+bl.sz; p++ {
						bound += math.Abs(r[q]) * amp * math.Abs(w[p])
					}
				}
			}
			yEnd := resp.Y.At(row, K-1)
			if !(math.Abs(yEnd-ss[row]) <= 1.01*bound+2*tol+1e-12*math.Abs(ss[row])) {
				t.Fatalf("Step row %d final %v, exact DC %v, transient bound %.3g\n%s", row, yEnd, ss[row], bound, s)
			}
		}
	}
}

// FuzzInitial: free response from x0 (texture from the phase field) with
// input and output delays: y_i(t) = C_i·e^{A(t−τ_i)}·x0 for t ≥ τ_i
// (τ_i the output delay, input delays inert), exactly sampled; tFinal from
// the Tau field (0 = automatic).
func FuzzInitial(f *testing.F) {
	for _, s := range fzTRSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.CondT > fzTRMaxCond {
			fzSkip(t, "ill-conditioned similarity")
		}
		md := fzModalOf(t, s)
		sys, err := NewFromSlices(s.N, s.M, s.P, s.A, s.B, s.C, s.D, s.Dt)
		if err != nil {
			t.Fatalf("NewFromSlices: %v\n%s", err, s)
		}
		if s.In != nil {
			if err := sys.SetInputDelay(s.In); err != nil {
				t.Fatal(err)
			}
		}
		if s.Out != nil {
			if err := sys.SetOutputDelay(s.Out); err != nil {
				t.Fatal(err)
			}
		}
		x0 := fzX0(s)
		var xv *mat.VecDense
		if s.N > 0 {
			xv = mat.NewVecDense(s.N, x0)
		} else {
			xv = &mat.VecDense{}
		}
		tFinal := s.Tau
		if s.Discrete {
			tFinal *= s.Dt
		}
		resp, err := Initial(sys, xv, tFinal)
		fzContract(t, s, "Initial", resp != nil, err)
		fzErr(t, s, "Initial", err)
		fzTRGrid(t, s, "Initial", resp, tFinal)
		if r, _ := resp.Y.Dims(); r != s.P {
			t.Fatalf("Initial: %d rows, want %d\n%s", r, s.P, s)
		}
		K := len(resp.T)
		var w []float64
		if s.N > 0 {
			w = md.col(x0)
		}
		for i := range s.P {
			want, mag := make([]float64, K), make([]float64, K)
			tau := 0.0
			if s.Out != nil {
				tau = s.Out[i]
			}
			if s.N > 0 {
				r := md.row(s.C[i*s.N : (i+1)*s.N])
				if s.Discrete {
					k0 := int(math.Round(tau))
					xi := append([]float64(nil), w...)
					for k := k0; k < K; k++ {
						if k > k0 {
							md.dStep(xi)
						}
						want[k], mag[k] = fzDot(r, xi)
					}
				} else {
					dt := resp.T[1]
					for k := range K {
						tp := resp.T[k] - tau
						if tau > 0 && math.Abs(tp) <= 1e-7*dt {
							want[k] = math.NaN()
							continue
						}
						if tp >= 0 {
							want[k], mag[k] = md.ctEval(false, r, w, tp)
						}
					}
				}
			}
			fzTRCompare(t, s, "Initial", i, resp.T, mat.Row(nil, i, resp.Y), want, mag)
		}
	})
}

func fzX0(s *fzSpec) []float64 {
	x := make([]float64, s.N)
	for i := range x {
		x[i] = math.Cos(s.Vals[11]*0.37 + 1.1*float64(i+1))
	}
	return x
}

// fzLsimPlan decodes the Lsim grid and signal from the Tau field v:
// samples K = 2 + ⌊100·(clamp(v,−3,3)+3)⌋; signal ⌊|v|⌋ mod 4 (sine,
// square, pulse, texture) with GenSig period ts·(2 + ⌊7|v|⌋ mod 61); x0 is
// used when ⌊|v|⌋ is odd and the model has no delays. Continuous ts is
// 10^clamp(v[8],−4,2).
type fzLsimPlan struct {
	K      int
	ts     float64
	sig    int
	period float64
	useX0  bool
}

func fzLsimPlanOf(s *fzSpec) fzLsimPlan {
	v := s.Vals[15]
	p := fzLsimPlan{K: 2 + int(100*(fzClamp(v, -3, 3)+3))}
	p.ts = s.Dt
	if !s.Discrete {
		p.ts = math.Pow(10, fzClamp(s.Vals[8], -4, 2))
	}
	iv := int(math.Min(math.Floor(math.Abs(v)), 1e6))
	p.sig = iv % 4
	p.period = p.ts * float64(2+int(math.Min(math.Floor(7*math.Abs(v)), 1e6))%61)
	p.useX0 = iv%2 == 1 && s.In == nil && s.Out == nil && s.IO == nil
	return p
}

var fzSigNames = [...]string{"sine", "square", "pulse"}

// fzGenSigCheck validates GenSig against its definition.
func fzGenSigCheck(t *testing.T, s *fzSpec, typ string, tau, tf, ts float64, u, tt []float64) {
	t.Helper()
	steps := int(math.Floor(tf/ts+1e-9)) + 1
	if len(u) != steps || len(tt) != steps {
		t.Fatalf("GenSig(%s,%g,%g,%g): %d/%d samples, want %d\n%s", typ, tau, tf, ts, len(u), len(tt), steps, s)
	}
	for k := range tt {
		if tt[k] != float64(k)*ts {
			t.Fatalf("GenSig t[%d] = %v, want %v\n%s", k, tt[k], float64(k)*ts, s)
		}
		ph := tt[k] / tau
		frac := ph - math.Floor(ph)
		var want float64
		switch typ {
		case "sine":
			want = math.Sin(2 * math.Pi * tt[k] / tau)
			if math.Abs(u[k]-want) > 1e-12 {
				t.Fatalf("GenSig sine u[%d] = %v, want %v\n%s", k, u[k], want, s)
			}
			continue
		case "square":
			if math.Abs(frac-0.5) < 1e-6 || frac < 1e-6 || frac > 1-1e-6 {
				continue
			}
			if frac > 0.5 {
				want = 1
			}
		case "pulse":
			if k == 0 {
				want = 0
				break
			}
			hi, lo := tt[k]/tau, (tt[k]-ts)/tau
			if math.Abs(hi-math.Round(hi)) < 1e-6 || math.Abs(lo-math.Round(lo)) < 1e-6 {
				continue
			}
			if math.Floor(hi) > math.Floor(lo) {
				want = 1
			}
		}
		if u[k] != want {
			t.Fatalf("GenSig %s u[%d] (t=%v, phase %v) = %v, want %v\n%s", typ, k, tt[k], ph, u[k], want, s)
		}
	}
}

// FuzzLsim: Lsim with a GenSig (or texture) input, I/O delays and x0,
// against the exact ZOH modal recurrence; continuous delays are exact
// shifts with the split step ξ ← e^{J·dt}ξ + e^{J(dt−r)}G(r)β·u_{k−q−1} +
// G(dt−r)β·u_{k−q} for a delay q·dt + r (G(h) = ∫₀ʰe^{Jτ}dτ), output delays
// moved to the input (zero initial state).
func FuzzLsim(f *testing.F) {
	for _, s := range fzTRSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if s.CondT > fzTRMaxCond {
			fzSkip(t, "ill-conditioned similarity")
		}
		md := fzModalOf(t, s)
		pl := fzLsimPlanOf(s)
		sys := s.System(t, fzSysOpts{Delays: true})
		sig := make([]float64, pl.K)
		if pl.sig < 3 {
			typ := fzSigNames[pl.sig]
			tf := float64(pl.K-1) * pl.ts
			u, tt, err := GenSig(typ, pl.period, tf, pl.ts)
			fzContract(t, s, "GenSig", u != nil, err)
			fzErr(t, s, "GenSig", err)
			fzGenSigCheck(t, s, typ, pl.period, tf, pl.ts, u, tt)
			if len(u) < pl.K {
				fzSkip(t, "GenSig grid shorter than plan")
			}
			copy(sig, u)
		} else {
			for k := range sig {
				sig[k] = math.Cos(0.7*float64(k) + s.Vals[11])
			}
		}
		T := make([]float64, pl.K)
		for k := range T {
			T[k] = float64(k) * pl.ts
		}
		U := mat.NewDense(pl.K, s.M, nil)
		uAt := func(j, k int) float64 {
			if k-j < 0 {
				return 0
			}
			return (1 - 0.4*float64(j)) * sig[k-j]
		}
		for k := range pl.K {
			for j := range s.M {
				U.Set(k, j, uAt(j, k))
			}
		}
		var x0 []float64
		var xv *mat.VecDense
		if pl.useX0 && s.N > 0 {
			x0 = fzX0(s)
			xv = mat.NewVecDense(s.N, x0)
		}
		resp, err := Lsim(sys, U, T, xv)
		fzContract(t, s, "Lsim", resp != nil, err)
		fzErr(t, s, "Lsim", err)
		if r, c := resp.Y.Dims(); r != s.P || c != pl.K || len(resp.T) != pl.K {
			t.Fatalf("Lsim: Y %d×%d, T %d, want %d×%d\n%s", r, c, len(resp.T), s.P, pl.K, s)
		}
		dt := pl.ts
		for i := range s.P {
			want, mag := make([]float64, pl.K), make([]float64, pl.K)
			var r []float64
			if s.N > 0 {
				r = md.row(s.C[i*s.N : (i+1)*s.N])
			}
			for j := range s.M {
				var beta []float64
				if s.N > 0 {
					beta = md.col(fzBcol(s, j))
				}
				d := s.D[i*s.M+j]
				tau := fzChanDelay(s, i, j)
				q, rr := 0, 0.0
				if s.Discrete {
					q = int(math.Round(tau))
				} else {
					qf := math.Floor(tau / dt)
					q, rr = int(qf), tau-qf*dt
					if f := rr / dt; f < 1e-6 || f > 1-1e-6 {
						if tau > 0 && f != 0 {
							fzSkip(t, "continuous delay within roundoff of the grid")
						}
					}
				}
				ud := func(k int) float64 {
					if k < 0 {
						return 0
					}
					return uAt(j, k)
				}
				fzLsimChannel(s, md, r, beta, d, q, rr, dt, ud, want, mag)
			}
			if x0 != nil {
				w := md.col(x0)
				xi := append([]float64(nil), w...)
				for k := range pl.K {
					var v, m float64
					if s.Discrete {
						if k > 0 {
							md.dStep(xi)
						}
						v, m = fzDot(r, xi)
					} else {
						v, m = md.ctEval(false, r, w, T[k])
					}
					want[k] += v
					mag[k] += m
				}
			}
			fzTRCompare(t, s, "Lsim", i, T, mat.Row(nil, i, resp.Y), want, mag)
		}
	})
}

// fzLsimChannel adds the zero-state response of input j (delayed by
// q·dt + r) through output row r to want.
func fzLsimChannel(s *fzSpec, md *fzModal, r, beta []float64, d float64, q int, rr, dt float64, ud func(int) float64, want, mag []float64) {
	K := len(want)
	n := s.N
	xi := make([]float64, n)
	tmp := make([]float64, n)
	type blkOps struct{ e, g, gA, eB, g2 [2]float64 }
	ops := make([]blkOps, len(md.blocks))
	if !s.Discrete {
		for bi, bl := range md.blocks {
			var o blkOps
			o.e[0], o.e[1] = blkExp(bl, dt)
			o.g[0], o.g[1] = blkInt(bl, dt)
			if rr > 0 {
				o.gA[0], o.gA[1] = blkInt(bl, rr)
				o.eB[0], o.eB[1] = blkExp(bl, dt-rr)
				o.g2[0], o.g2[1] = blkInt(bl, dt-rr)
			}
			ops[bi] = o
		}
	}
	gb := make([]float64, n)
	ga := make([]float64, n)
	g2 := make([]float64, n)
	if !s.Discrete && n > 0 {
		for bi, bl := range md.blocks {
			blkApply(bl, ops[bi].g[0], ops[bi].g[1], beta, gb, nil)
			if rr > 0 {
				blkApply(bl, ops[bi].gA[0], ops[bi].gA[1], beta, tmp, nil)
				blkApply(bl, ops[bi].eB[0], ops[bi].eB[1], tmp, ga, nil)
				blkApply(bl, ops[bi].g2[0], ops[bi].g2[1], beta, g2, nil)
			}
		}
	}
	for k := range K {
		uNow := ud(k - q)
		if rr > 0 {
			uNow = ud(k - q - 1)
		}
		v, m := 0.0, 0.0
		if n > 0 {
			v, m = fzDot(r, xi)
		}
		want[k] += v + d*uNow
		mag[k] += m + math.Abs(d*uNow)
		if n == 0 {
			continue
		}
		if s.Discrete {
			md.dStep(xi)
			u := ud(k - q)
			for p := range xi {
				xi[p] += beta[p] * u
			}
			continue
		}
		for bi, bl := range md.blocks {
			blkApply(bl, ops[bi].e[0], ops[bi].e[1], xi, tmp, nil)
		}
		if rr > 0 {
			u1, u2 := ud(k-q-1), ud(k-q)
			for p := range xi {
				xi[p] = tmp[p] + ga[p]*u1 + g2[p]*u2
			}
		} else {
			u := ud(k - q)
			for p := range xi {
				xi[p] = tmp[p] + gb[p]*u
			}
		}
	}
}

// fzTRSeeds: SISO first/second order, MIMO with D, an exact integrator,
// unstable modes, discrete with whole-sample delays, continuous
// fractional delays, static gain, and automatic horizons (Tau off).
func fzTRSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: 0.3, Shear: -10, Rot: 0.37, DScal: 0.5, LogDt: -1}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0] = 1, 2.5 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 2, 0.5, 2.7 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Tau = 2, 0.5, 2.7, fzNoDelay }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.Shear = 4, 2, 3, [3]float64{0.3, -0.5}, [3]float64{1, 2}, -1
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Rot, s.Tau = 1, -40, 0, 1.1 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp = 3, [3]float64{0, 0.2}, [3]float64{-2.5, 1} }),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.M, s.P, s.Mag, s.Damp, s.In, s.Out, s.IO, s.Tau = true, 3, 2, 2, [3]float64{0.5, 0.1}, [3]float64{2, 1}, 0.2, 0.5, 0.3, 1.5
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.N, s.Mag, s.Damp, s.Tau = true, 2, [3]float64{0.4}, [3]float64{2.2}, fzNoDelay
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.In, s.Out, s.IO, s.Tau = 2, 2, 2, [3]float64{0.2}, -0.7, -1.1, -0.9, 0.8
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 0, 2, 2, 3 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.Mag, s.Damp, s.Tau, s.LogDt = 5, 3, [3]float64{0.5, 1, -0.5}, [3]float64{2.2, 1.7, 1}, 1.4, -1.5
		}),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.Tau, s.LogDt = 2, [3]float64{0.3}, [3]float64{2.2}, 2.3, -1 }),
		with(func(s *fzSeed) {
			s.N, s.Mag, s.Damp, s.Tau, s.LogDt, s.In = 2, [3]float64{0.3}, [3]float64{2.2}, 2.7, -1.3, -0.8
		}),
	}
}

func TestFzModalOracle(t *testing.T) {
	for _, sd := range fzTRSeeds() {
		k, v := sd.encode()
		s := fzDecode(t, k, v)
		if s.N == 0 {
			continue
		}
		md := fzModalOf(t, s)
		J := mat.NewDense(s.N, s.N, nil)
		for _, bl := range md.blocks {
			J.Set(bl.k, bl.k, bl.a)
			if bl.sz == 2 {
				J.Set(bl.k+1, bl.k+1, bl.a)
				J.Set(bl.k, bl.k+1, bl.b)
				J.Set(bl.k+1, bl.k, -bl.b)
			}
		}
		var A, tmp mat.Dense
		tmp.Mul(md.T, J)
		A.Mul(&tmp, md.Ti)
		for i, a := range s.A {
			if math.Abs(A.RawMatrix().Data[i]-a) > 1e-12*math.Max(1, s.NormJ)*s.CondT {
				t.Fatalf("T·J·T⁻¹ ≠ A at %d: %v vs %v\n%s", i, A.RawMatrix().Data[i], a, s)
			}
		}
		if s.Discrete {
			continue
		}
		var ad mat.Dense
		dt := 0.37
		var sc mat.Dense
		sc.Scale(dt, mat.NewDense(s.N, s.N, s.A))
		ad.Exp(&sc)
		for i := range s.N {
			e := make([]float64, s.N)
			e[i] = 1
			w := md.col(e)
			for r := range s.N {
				c := make([]float64, s.N)
				c[r] = 1
				got, _ := md.ctEval(false, md.row(c), w, dt)
				if math.Abs(got-ad.At(r, i)) > 1e-10 {
					t.Fatalf("modal e^{At}[%d,%d] = %v, expm %v", r, i, got, ad.At(r, i))
				}
			}
		}
	}
	bl := fzBlk{0, 2, -0.3, 2}
	g1, g2 := blkInt(bl, 0.8)
	var r1, r2 float64
	const N = 200000
	for k := range N {
		x := (float64(k) + 0.5) * 0.8 / N
		e1, e2 := blkExp(bl, x)
		r1 += e1 * 0.8 / N
		r2 += e2 * 0.8 / N
	}
	if math.Abs(g1-r1) > 1e-9 || math.Abs(g2-r2) > 1e-9 {
		t.Fatalf("blkInt = (%v,%v), quadrature (%v,%v)", g1, g2, r1, r2)
	}
}

func fzFinite(x []float64) bool {
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// FuzzStepInternalDelay: discrete Step and Impulse with an internal delay
// loop w_k = C2·x_{k−τ} (fzSysOpts.Internal: B2 = B[:,0], D12 = D21 =
// D22 = 0) against the direct buffered recurrence
// x_{k+1} = A·x_k + B·u_k + B2·w_k, y_k = C·x_k + D·u_k, shifted by the
// whole-sample input and output delays. Continuous internal delays have no
// exact oracle here and are skipped.
func FuzzStepInternalDelay(f *testing.F) {
	for _, s := range fzTRSeeds() {
		if s.Discrete {
			s.Tau = 0.5
			fzAddSeed(f, s)
		}
	}
	fzAddSeed(f, fzSeed{Discrete: true, N: 3, M: 2, P: 2, Mag: [3]float64{0.2, 0.5}, Damp: [3]float64{1.5, 2}, DScal: 0.5,
		LogDt: -1, Shear: -10, Rot: 0.37, In: fzNoDelay, Out: 0.3, IO: fzNoDelay, Tau: 0.9})
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		if !s.Discrete {
			fzSkip(t, "continuous internal delay: no exact oracle")
		}
		if s.Tau == 0 || s.N == 0 {
			fzSkip(t, "no internal delay")
		}
		sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
		n, m, p := s.N, s.M, s.P
		tau := int(s.Tau)
		b2, c2 := make([]float64, n), make([]float64, n)
		for i := range n {
			b2[i] = s.B[i*m]
			c2[i] = 0.5 * math.Sin(float64(i)+s.Vals[11])
		}
		for _, impulse := range []bool{false, true} {
			op := "Step(internal)"
			resp, err := Step(sys, 0)
			if impulse {
				op = "Impulse(internal)"
				resp, err = Impulse(sys, 0)
			}
			fzContract(t, s, op, resp != nil, err)
			fzErr(t, s, op, err)
			fzTRGrid(t, s, op, resp, 0)
			K := len(resp.T)
			if r, _ := resp.Y.Dims(); r != m*p {
				t.Fatalf("%s: %d rows, want %d\n%s", op, r, m*p, s)
			}
			for j := range m {
				x, mx := make([]float64, n), make([]float64, n)
				zs, mz := make([]float64, K), make([]float64, K)
				want, mag := make([][]float64, p), make([][]float64, p)
				for i := range p {
					want[i], mag[i] = make([]float64, K), make([]float64, K)
				}
				for k := range K {
					uk := 1.0
					if impulse && k > 0 {
						uk = 0
					}
					for i := range p {
						v, _ := fzDot(s.C[i*n:(i+1)*n], x)
						_, ma := fzDot(fzAbs(s.C[i*n:(i+1)*n]), mx)
						want[i][k], mag[i][k] = v+s.D[i*m+j]*uk, ma+math.Abs(s.D[i*m+j]*uk)
					}
					zs[k], _ = fzDot(c2, x)
					_, mz[k] = fzDot(fzAbs(c2), mx)
					w, mw := 0.0, 0.0
					if k-tau >= 0 {
						w, mw = zs[k-tau], mz[k-tau]
					}
					nx, nm := make([]float64, n), make([]float64, n)
					for r := range n {
						v, _ := fzDot(s.A[r*n:(r+1)*n], x)
						_, mv := fzDot(fzAbs(s.A[r*n:(r+1)*n]), mx)
						nx[r] = v + s.B[r*m+j]*uk + b2[r]*w
						nm[r] = mv + math.Abs(s.B[r*m+j]*uk) + math.Abs(b2[r])*mw
					}
					x, mx = nx, nm
				}
				for i := range p {
					d := fzOutShift(s, i, j)
					sw, sm := make([]float64, K), make([]float64, K)
					for k := d; k < K; k++ {
						sw[k], sm[k] = want[i][k-d], mag[i][k-d]
					}
					row := j*p + i
					fzTRCompare(t, s, op, row, resp.T, mat.Row(nil, row, resp.Y), sw, sm)
				}
			}
		}
	})
}

// fzOutShift is the whole-sample delay of channel (i, j) with an internal
// delay, where the helper drops Delay so only input and output delays act.
func fzOutShift(s *fzSpec, i, j int) int {
	d := 0.0
	if s.In != nil {
		d += s.In[j]
	}
	if s.Out != nil {
		d += s.Out[i]
	}
	return int(math.Round(d))
}

func fzAbs(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = math.Abs(v)
	}
	return y
}
