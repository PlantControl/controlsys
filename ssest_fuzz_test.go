package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// fzIOInput is a deterministic ±1 telegraph signal of length n with hold
// lengths 1..hold, seeded from the spec tuple (no math/rand).
func fzIOInput(s *fzSpec, n, hold int) []float64 {
	x := uint64(s.Kind)<<56 ^ math.Float64bits(s.Vals[11]) ^ math.Float64bits(s.Vals[10])<<1 ^ 0x9e3779b97f4a7c15
	next := func() uint64 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		return x
	}
	u := make([]float64, n)
	level := 1.0
	for k := 0; k < n; {
		run := 1 + int(next()%uint64(max(hold, 1)))
		for i := 0; i < run && k < n; i++ {
			u[k] = level
			k++
		}
		if next()&1 == 0 {
			level = -level
		} else {
			level = -level * 0.5
			if math.Abs(level) < 0.25 {
				level = math.Copysign(1, level)
			}
		}
	}
	return u
}

// fzIOSimulate runs x[k+1] = A x[k] + b u[k-delay], y = c x + d u[k-delay]
// from x0, with zero input before the record.
func fzIOSimulate(n int, a, b, c []float64, d float64, x0, u []float64, delay int) []float64 {
	x := append([]float64(nil), x0...)
	next := make([]float64, n)
	y := make([]float64, len(u))
	for k := range u {
		uk := 0.0
		if k-delay >= 0 {
			uk = u[k-delay]
		}
		v := d * uk
		for j := range n {
			v += c[j] * x[j]
		}
		y[k] = v
		for i := range n {
			w := b[i] * uk
			for j := range n {
				w += a[i*n+j] * x[j]
			}
			next[i] = w
		}
		x, next = next, x
	}
	return y
}

type fzIOCase struct {
	b, c, x0   []float64
	d          float64
	delay      int
	u, y       []float64
	train      int
	estimate   bool
	feed       bool
	markov     []float64
	hankel     float64
	rho        float64
	wellPosed  bool
	identified bool
}

func fzIOBuild(t *testing.T, s *fzSpec) *fzIOCase {
	t.Helper()
	if !s.Discrete || s.N == 0 {
		fzSkip(t, "IO state-space needs a discrete model with states")
	}
	if s.Unstable() > 0 || !s.Decidable() {
		fzSkip(t, "unstable or boundary model (output-error free run diverges)")
	}
	n, m := s.N, s.M
	cs := &fzIOCase{b: make([]float64, n), c: s.C[:n], d: s.D[0], train: 200}
	for i := range n {
		cs.b[i] = s.B[i*m]
	}
	if s.In != nil {
		cs.delay = int(s.In[0]) % 6
	}
	cs.feed = cs.d != 0
	cs.estimate = s.Vals[15] > 0
	cs.x0 = make([]float64, n)
	if cs.estimate {
		for i := range n {
			cs.x0[i] = 2 * math.Sin(1.7*float64(i+1)+s.Vals[11])
		}
	}
	rho := 0.0
	for _, e := range s.Eig {
		rho = math.Max(rho, cmplx.Abs(e))
	}
	mk := fzEraMarkov(n, 1, 1, s.A, cs.b, cs.c, []float64{cs.d}, 2*n+3)
	for _, v := range mk {
		cs.markov = append(cs.markov, v[0])
	}
	sv := fzEraHankelSV(mk, n+1, 1, 1)
	if sv == nil || sv[0] == 0 {
		fzSkip(t, "zero SISO channel")
	}
	cs.hankel = sv[n-1] / sv[0]
	if cs.hankel < 1e-7 {
		fzSkip(t, "SISO channel numerically non-minimal")
	}
	sep := math.Inf(1)
	for i, e := range s.Eig {
		for j := range i {
			sep = math.Min(sep, cmplx.Abs(e-s.Eig[j]))
		}
	}
	cs.rho = rho
	dyn, minz := 0.0, math.Inf(1)
	for _, v := range cs.markov[1:] {
		dyn += math.Abs(v)
	}
	for _, e := range s.Eig {
		minz = math.Min(minz, cmplx.Abs(e))
	}
	cs.wellPosed = dyn >= 1e-3*math.Abs(cs.d) && minz >= 1e-3 && rho <= 0.95 && sep >= 0.05 && cs.hankel >= 1e-4 && s.CondT < 1e3
	cs.u = fzIOInput(s, cs.train+80, 3)
	cs.y = fzIOSimulate(n, s.A, cs.b, cs.c, cs.d, cs.x0, cs.u, cs.delay)
	for _, v := range cs.y {
		if fzBadFloat(v) {
			fzSkip(t, "denormal output")
		}
	}
	return cs
}

// FuzzIdentIOStateSpace: IdentifyIOStateSpace on noiseless SISO data from a
// decoded stable discrete model (input delay, feedthrough, nonzero initial
// state with InitialCondition estimate) at the true order reproduces the
// held-out record (NRMSE <= 1e-9·max(1, 1e-3·cond), 1e-6 on ill-posed
// regressions); its System matches the true Markov parameters and is
// stable when every pole has |z| < 1−1e-4; an order scan 1..n+1 selects
// the lowest order within max(1.01·best, 1e-8), at most n. Documented
// excitation/instability errors skip only on ill-posed data.
func FuzzIdentIOStateSpace(f *testing.F) {
	for _, s := range fzIOSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		cs := fzIOBuild(t, s)
		n := s.N
		ic := IOInitialZero
		if cs.estimate {
			ic = IOInitialEstimate
		}
		opts := IOStateSpaceOptions{Order: n, InputDelay: cs.delay, DirectFeedthrough: cs.feed, InitialCondition: ic}
		tu, ty, vu, vy := cs.u[:cs.train], cs.y[:cs.train], cs.u[cs.train:], cs.y[cs.train:]
		res, err := IdentifyIOStateSpace(context.Background(), tu, ty, vu, vy, s.Dt, opts)
		fzContract(t, s, "IdentifyIOStateSpace", res != nil, err)
		if err != nil && !cs.wellPosed && (errors.Is(err, ErrProcessExcitation) || errors.Is(err, ErrUnstable)) {
			fzSkip(t, "ill-posed regression: documented error")
		}
		fzErr(t, s, "IdentifyIOStateSpace", err)
		if res.Selected != 0 || len(res.Candidates) != 1 {
			t.Fatalf("selected %d of %d candidates\n%s", res.Selected, len(res.Candidates), s)
		}
		c := res.Candidates[0]
		if c.Err != nil || c.System == nil || c.Order != n {
			t.Fatalf("candidate order %d err %v\n%s", c.Order, c.Err, s)
		}
		tol := 1e-9 * math.Max(1, 1e-3*c.Condition)
		if !cs.wellPosed {
			tol = math.Max(tol, 1e-6)
		}
		if !(c.ValidationNRMSE <= tol) || !(c.TrainingNRMSE <= tol) {
			if !cs.wellPosed && c.Convergence != IOConverged {
				fzSkip(t, "ill-posed regression: refinement not converged")
			}
			t.Fatalf("NRMSE train %g validation %g > %g (cond %g, %s, hankel %g, delay %d, feed %v, estimate %v)\n%s",
				c.TrainingNRMSE, c.ValidationNRMSE, tol, c.Condition, c.Convergence, cs.hankel, cs.delay, cs.feed, cs.estimate, s)
		}
		fzIOCheckSystem(t, s, cs, c.System, 1e4*tol)
		if !c.Stable && cs.rho < 1-1e-4 {
			t.Fatalf("identified model of a stable plant reported unstable\n%s", s)
		}
		if n > 3 || cs.hankel < 1e-4 || c.ValidationNRMSE > 1e-9 {
			return
		}
		opts.Order, opts.MinOrder, opts.MaxOrder = 0, 1, n+1
		scan, err := IdentifyIOStateSpace(context.Background(), tu, ty, vu, vy, s.Dt, opts)
		fzContract(t, s, "IdentifyIOStateSpace(scan)", scan != nil, err)
		fzErr(t, s, "IdentifyIOStateSpace(scan)", err)
		if len(scan.Candidates) != n+1 || scan.Selected < 0 {
			t.Fatalf("scan: %d candidates, selected %d\n%s", len(scan.Candidates), scan.Selected, s)
		}
		best := math.Inf(1)
		for _, c := range scan.Candidates {
			if c.Err == nil {
				best = math.Min(best, c.ValidationNRMSE)
			}
		}
		want := -1
		for i, c := range scan.Candidates {
			if c.Err == nil && c.ValidationNRMSE <= math.Max(1.01*best, 1e-8) {
				want = i
				break
			}
		}
		if got := scan.Candidates[scan.Selected].Order; scan.Selected != want || got > n {
			t.Fatalf("scan selected order %d, want index %d, order <= %d (NRMSE by order %v)\n%s", got, want, n, fzIONRMSEs(scan), s)
		}
	})
}

func fzIONRMSEs(r *IOStateSpaceResult) []float64 {
	v := make([]float64, len(r.Candidates))
	for i, c := range r.Candidates {
		v[i] = c.ValidationNRMSE
		if c.Err != nil {
			v[i] = math.NaN()
		}
	}
	return v
}

func fzIOCheckSystem(t *testing.T, s *fzSpec, cs *fzIOCase, sys *System, rel float64) {
	t.Helper()
	rn, rm, rp := sys.Dims()
	if rm != 1 || rp != 1 || sys.Dt != s.Dt {
		t.Fatalf("identified system %d×%d×%d dt=%g\n%s", rn, rm, rp, sys.Dt, s)
	}
	a, b, c, d := fzEraRaw(sys)
	want := make([]float64, len(cs.markov)+cs.delay)
	copy(want[cs.delay:], cs.markov)
	got := fzEraMarkov(rn, 1, 1, a, b, c, d, len(want))
	scale := 0.0
	for _, v := range want {
		scale = math.Max(scale, math.Abs(v))
	}
	for k := range want {
		if !(math.Abs(got[k][0]-want[k]) <= rel*scale) {
			t.Fatalf("identified Markov Y%d = %g, want %g (scale %g, rel %g)\n%s", k, got[k][0], want[k], scale, rel, s)
		}
	}
}

func fzIOSeeds() []fzSeed {
	return []fzSeed{
		{Discrete: true, N: 1, M: 1, P: 1, Mag: [3]float64{0}, Damp: [3]float64{2}, Gain: 6, LogDt: -1, Shear: -10, In: fzNoDelay, Tau: fzNoDelay},
		{Discrete: true, N: 2, M: 1, P: 1, Mag: [3]float64{0.5}, Damp: [3]float64{1.5}, Gain: 6, DScal: 0.7, LogDt: -1, Shear: -1, Rot: 0.4, In: 0.3, Tau: fzNoDelay},
		{Discrete: true, N: 3, M: 1, P: 1, Mag: [3]float64{0.6, 0.2}, Damp: [3]float64{1.2, 3}, Gain: 6, LogDt: -1, Shear: -0.5, Rot: 0.7, Phase: 0.3, In: fzNoDelay, Tau: 0.5},
		{Discrete: true, N: 4, M: 1, P: 1, Mag: [3]float64{0.7, 1.0}, Damp: [3]float64{1.7, 2.2}, Gain: 7, DScal: 0.5, LogDt: -1.2, Shear: -1, Rot: 1.1, Phase: 1.3, In: 0.5, Tau: 0.5},
		{Discrete: true, N: 2, M: 1, P: 1, Mag: [3]float64{0.8}, Damp: [3]float64{0.3}, Gain: 5, LogDt: -1, Shear: -2, Rot: -0.6, Phase: 2.1, In: fzNoDelay, Tau: 0.2},
		{Discrete: true, N: 5, M: 2, P: 1, Mag: [3]float64{0.5, 0.9, 0.3}, Damp: [3]float64{1.4, 2.0, 1.1}, Gain: 6, DScal: -1, LogDt: -1.1, Shear: -1.5, Rot: 0.9, Phase: -0.7, In: fzNoDelay, Tau: fzNoDelay},
	}
}
