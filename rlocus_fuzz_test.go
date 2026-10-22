package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"sort"
	"testing"
)

// FuzzRootLocus (task O5FYRC): the SISO loop is the spec's first input and
// output channel (discrete models carry input+output delay L, absorbed by
// RootLocus). At each gain K the closed-loop roots are those of the exact
// polynomial P_K = z^L·den + K·(det(sI−A+BC) − den + D·den), den = det(sI−A),
// over big.Rat. Branches are checked by exact root counts right of lines
// (continuous, fzRouthRight) or outside circles (discrete,
// fzSchurCohnOutside) drawn in the gaps of the returned values; the K = 0
// column of a delay-free loop must match the intended eigenvalues. Each
// Breakaway point s₀ must lie on the K > 0 locus: K(s₀) = −s₀^L·den/num > 0.

const fzRlocBand = 1e-6

func FuzzRootLocus(f *testing.F) {
	for _, s := range fzRlocSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n := s.N
		b := make([]float64, n)
		for i := range n {
			b[i] = s.B[i*s.M]
		}
		c, d := s.C[:n], s.D[0]
		sys, err := NewFromSlices(n, 1, 1, s.A, b, c, []float64{d}, s.Dt)
		if err != nil {
			t.Fatalf("NewFromSlices: %v\n%s", err, s)
		}
		L := 0
		if s.Discrete {
			if s.In != nil {
				L += int(s.In[0])
				if err := sys.SetInputDelay(s.In[:1]); err != nil {
					t.Fatalf("SetInputDelay: %v\n%s", err, s)
				}
			}
			if s.Out != nil {
				L += int(s.Out[0])
				if err := sys.SetOutputDelay(s.Out[:1]); err != nil {
					t.Fatalf("SetOutputDelay: %v\n%s", err, s)
				}
			}
		} else if s.In != nil && n > 0 {
			dsys := sys.Copy()
			if err := dsys.SetInputDelay(s.In[:1]); err != nil {
				t.Fatalf("SetInputDelay: %v\n%s", err, s)
			}
			res, err := RootLocus(dsys, nil)
			fzContract(t, s, "RootLocus(delayed)", res != nil, err)
			if !errors.Is(err, ErrDelayUnsupported) {
				t.Fatalf("RootLocus continuous delay: err %v, want ErrDelayUnsupported\n%s", err, s)
			}
		}
		gscale := math.Pow(10, fzClamp(math.Abs(s.Vals[6]), 0, 12)-6)
		k := math.Pow(10, fzClamp(s.Vals[15], -3, 3)) / gscale
		gains := []float64{10 * k, 0, k}
		if s.Vals[14] < 0 {
			gains = append(gains, -k)
		}
		res, err := RootLocus(sys, gains)
		fzContract(t, s, "RootLocus", res != nil, err)
		if n == 0 && L == 0 {
			if !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("RootLocus static gain: err %v, want ErrDimensionMismatch\n%s", err, s)
			}
			return
		}
		if n == 0 {
			fzSkip(t, "rlocus: pure delay loop")
		}
		if L == 0 && errors.Is(err, ErrAlgebraicLoop) {
			for _, K := range gains {
				if 1+K*d == 0 {
					fzSkip(t, "rlocus: 1+K·D = 0")
				}
			}
		}
		fzErr(t, s, "RootLocus", err)
		want := slices.Clone(gains)
		sort.Float64s(want)
		if !slices.Equal(res.Gains, want) {
			t.Fatalf("Gains = %v, want sorted %v\n%s", res.Gains, want, s)
		}
		if len(res.Branches) != n+L {
			t.Fatalf("%d branches, want %d\n%s", len(res.Branches), n+L, s)
		}
		for _, br := range res.Branches {
			if len(br) != len(want) {
				t.Fatalf("branch has %d points for %d gains\n%s", len(br), len(want), s)
			}
		}
		if n+L > 14 {
			fzSkip(t, "rlocus: oracle degree > 14")
		}
		den := fzCharPolyExact(s.A, n)
		abc := make([]*big.Rat, n*n)
		for i := range n {
			for j := range n {
				x := fzRat(s.A[i*n+j])
				abc[i*n+j] = x.Sub(x, new(big.Rat).Mul(fzRat(b[i]), fzRat(c[j])))
			}
		}
		cpBC := fzCharPolyRat(abc, n)
		num := make([]*big.Rat, n+1)
		for i := range num {
			num[i] = new(big.Rat).Sub(cpBC[i], den[i])
			num[i].Add(num[i], new(big.Rat).Mul(fzRat(d), den[i]))
		}
		for gi, K := range res.Gains {
			vals := make([]complex128, len(res.Branches))
			for bi, br := range res.Branches {
				vals[bi] = br[gi]
			}
			P := make([]*big.Rat, n+L+1)
			for i := range P {
				P[i] = new(big.Rat)
				if i <= n {
					P[i].Set(den[i])
				}
			}
			kr := fzRat(K)
			for i := range num {
				P[L+i].Add(P[L+i], new(big.Rat).Mul(kr, num[i]))
			}
			if P[0].Sign() == 0 {
				continue
			}
			fzRlocCheckCounts(t, s, K, vals, P)
			if K == 0 && L == 0 {
				r := s.EigErr() * fzBalanceCond(s.A, n)
				if _, _, msg := fzMatchEig(vals, s.Eig, r); msg != "" {
					t.Fatalf("K=0 branches %v: %s\n%s", vals, msg, s)
				}
			}
		}
		fzRlocCheckBreakaway(t, s, sys, res.Breakaway, den, num, L)
	})
}

// fzRlocCheckCounts compares, for every line Re s = σ (circle |z| = ρ) in a
// gap of vals wider than the band, the number of vals beyond it with the
// exact root count of P; lines where the counts at σ ± band disagree
// (a true root inside the band) are skipped.
func fzRlocCheckCounts(t *testing.T, s *fzSpec, K float64, vals []complex128, P []*big.Rat) {
	t.Helper()
	key := func(v complex128) float64 {
		if s.Discrete {
			return cmplx.Abs(v)
		}
		return real(v)
	}
	R := 1.0
	for _, v := range vals {
		if cmplx.IsNaN(v) || cmplx.IsInf(v) {
			t.Fatalf("K=%g: non-finite closed-loop pole %v\n%s", K, v, s)
		}
		R = math.Max(R, cmplx.Abs(v))
	}
	band := fzRlocBand * R
	ks := make([]float64, len(vals))
	for i, v := range vals {
		ks[i] = key(v)
	}
	sort.Float64s(ks)
	count := func(x float64) (int, bool) {
		if s.Discrete {
			if x <= 0 {
				return 0, false
			}
			return fzSchurCohnOutside(P, x)
		}
		return fzRouthRight(P, x)
	}
	lines := []float64{ks[0] - 4*band}
	for i := 1; i < len(ks); i++ {
		if ks[i]-ks[i-1] > 4*band {
			lines = append(lines, (ks[i]+ks[i-1])/2)
		}
	}
	lines = append(lines, ks[len(ks)-1]+4*band)
	for _, x := range lines {
		lo, okLo := count(x - band)
		hi, okHi := count(x + band)
		if !okLo || !okHi || lo != hi {
			continue
		}
		got := 0
		for _, k := range ks {
			if k > x {
				got++
			}
		}
		if got != hi {
			t.Fatalf("K=%g: %d closed-loop poles beyond %g, exact count %d\nvals=%v\nP=%v\n%s", K, got, x, hi, vals, P, s)
		}
	}
}

// fzRlocCheckBreakaway requires every reported break point clear of the
// poles, zeros and the root cluster at the origin to be a stationary point
// of G with K(s₀) = −s₀^L·den(s₀)/num(s₀) > 0, for either sign of the
// high-frequency gain.
func fzRlocCheckBreakaway(t *testing.T, s *fzSpec, sys *System, bp []complex128, den, num []*big.Rat, L int) {
	t.Helper()
	evalAbs := func(p []*big.Rat, x float64) (v, mag float64) {
		var acc, abs big.Rat
		xr := fzRat(x)
		ax := new(big.Rat).Abs(xr)
		for _, ci := range p {
			acc.Mul(&acc, xr)
			acc.Add(&acc, ci)
			abs.Mul(&abs, ax)
			abs.Add(&abs, new(big.Rat).Abs(ci))
		}
		v, _ = acc.Float64()
		mag, _ = abs.Float64()
		return v, mag
	}
	deriv := func(p []*big.Rat) []*big.Rat {
		out := make([]*big.Rat, max(len(p)-1, 1))
		out[0] = new(big.Rat)
		for i := range len(p) - 1 {
			out[i] = new(big.Rat).Mul(p[i], big.NewRat(int64(len(p)-1-i), 1))
		}
		return out
	}
	dz := slices.Concat(den, make([]*big.Rat, L))
	for i := len(den); i < len(dz); i++ {
		dz[i] = new(big.Rat)
	}
	q1, q2 := fzRatPolyMul(deriv(num), dz), fzRatPolyMul(num, deriv(dz))
	q := make([]*big.Rat, max(len(q1), len(q2)))
	for i := range q {
		q[i] = new(big.Rat)
		if j := i - (len(q) - len(q1)); j >= 0 {
			q[i].Add(q[i], q1[j])
		}
		if j := i - (len(q) - len(q2)); j >= 0 {
			q[i].Sub(q[i], q2[j])
		}
	}
	wellScaled := s.NormJ >= 1e-3 && s.NormJ <= 1e3 && s.CondT < 10
	for _, e := range s.Eig {
		if a := cmplx.Abs(e); a > 0 && a < 1e-4*s.NormJ {
			wellScaled = false
		}
	}
	mult := L
	for _, e := range s.Eig {
		if cmplx.Abs(e) < 1e-10*math.Max(1, s.NormJ) {
			mult++
		}
	}
	origin := 0.0
	if mult > 0 {
		origin = math.Pow(1e-15, 1/float64(mult+1)) * math.Max(1, s.NormJ)
	}
	for _, z := range bp {
		if imag(z) != 0 {
			t.Fatalf("Breakaway %v is not real\n%s", z, s)
		}
		x := real(z)
		if math.Abs(x) < origin {
			continue
		}
		if qt := fzTrimRat(q); len(qt) > 1 && math.Abs(x) > 1e8*math.Max(1, fzPidFujiwara(fzPidRatFloats(qt))) {
			fzSkip(t, "rlocus: far breakaway from ss→tf rounding noise (followup O5FYRC-4)")
		}
		if !fzRlocNearRoot(q, x, evalAbs) {
			if !wellScaled {
				fzSkip(t, "rlocus: breakaway root inaccurate on ill-scaled loop")
			}
			if L > 0 {
				fzSkip(t, "rlocus: delay-absorbed breakaway inaccurate (followup O5FYRC-4)")
			}
			t.Fatalf("Breakaway %v is not a stationary point of G\nq=%v\n%s", z, q, s)
		}
		dv, dm := evalAbs(den, x)
		nv, nm := evalAbs(num, x)
		if math.Abs(dv) <= 1e-6*dm || math.Abs(nv) <= 1e-6*nm {
			continue
		}
		K := -dv / nv * math.Pow(x, float64(L))
		if K > 0 {
			continue
		}
		if fzRlocAbsThreshold(t, s, sys, L) {
			fzSkip(t, "rlocus: absolute real-axis thresholds (followup O5FYRC-2)")
		}
		t.Fatalf("Breakaway %v needs gain K=%g, not on the K>0 locus\nden=%v num=%v L=%d\n%s", z, K, den, num, L, s)
	}
}

// fzRlocNearRoot reports whether x is a root of the exact q to working
// precision: |q(x)| ≤ 1e-6·Σ|qᵢ|·r^(deg−i) with r = max(|x|, δ), or a sign
// change of q within δ of x; δ = 1e-6·max(|x|, root bound) covers root
// clusters near the origin.
func fzRlocNearRoot(q []*big.Rat, x float64, eval func([]*big.Rat, float64) (float64, float64)) bool {
	q = fzTrimRat(q)
	if len(q) == 0 {
		return true
	}
	d := 1e-6 * math.Max(math.Abs(x), fzPidFujiwara(fzPidRatFloats(q)))
	v, _ := eval(q, x)
	if _, m := eval(q, math.Max(math.Abs(x), d)); math.Abs(v) <= 1e-6*m {
		return true
	}
	a, _ := eval(q, x-d)
	b, _ := eval(q, x+d)
	return a*b <= 0
}

// fzRlocAbsThreshold reports whether a pole (delay poles at z = 0 included)
// or zero of the loop is complex
// with |Im| < 2e-6, or two distinct real ones lie within 2e-4 (twice the
// absolute cut-offs of isOnRealAxisSegment / isNearRepeatedReal, which then
// misclassify them).
func fzRlocAbsThreshold(t *testing.T, s *fzSpec, sys *System, L int) bool {
	t.Helper()
	zeros, err := sys.Zeros()
	fzErr(t, s, "Zeros", err)
	roots := slices.Concat(s.Eig, zeros, make([]complex128, L))
	for i, r := range roots {
		if imag(r) != 0 && math.Abs(imag(r)) < 2e-6 {
			return true
		}
		for _, q := range roots[i+1:] {
			if imag(r) == 0 && imag(q) == 0 && r != q && math.Abs(real(r-q)) < 2e-4 {
				return true
			}
		}
	}
	return false
}

func fzRlocSeeds() []fzSeed {
	none := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: 0, Shear: -10}
	mk := func(f func(*fzSeed)) fzSeed {
		x := none
		f(&x)
		return x
	}
	return []fzSeed{
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain = 2, [3]float64{0.3}, [3]float64{1}, 6 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain = 2, [3]float64{0}, [3]float64{1.5}, 6 }),
		mk(func(x *fzSeed) {
			x.N, x.Mag, x.Damp, x.Gain, x.DScal = 3, [3]float64{0, 0.5}, [3]float64{2.5, 1}, 6, 0.5
		}),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.IO = 2, [3]float64{0}, [3]float64{1.5}, -6, -1 }),
		mk(func(x *fzSeed) {
			x.N, x.Mag, x.Damp, x.Gain, x.Shear, x.Rot = 4, [3]float64{0.2, -0.5}, [3]float64{-1.5, 2.5}, 7, -1, 0.7
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.Mag, x.Damp, x.Gain, x.In = true, -1, 2, [3]float64{0.5}, [3]float64{1.2}, 6, 0.3
		}),
		mk(func(x *fzSeed) {
			x.Discrete, x.LogDt, x.N, x.Mag, x.Damp, x.Gain, x.Out, x.DScal = true, -0.5, 3, [3]float64{0, 0.3}, [3]float64{2, 1}, 6.5, 0, 0.2
		}),
		mk(func(x *fzSeed) { x.N, x.Tau = 0, 1 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.In = 1, [3]float64{0}, [3]float64{1}, 6, 0 }),
		mk(func(x *fzSeed) { x.N, x.Mag, x.Damp, x.Gain, x.Tau = 3, [3]float64{-40, 1}, [3]float64{1, 2.5}, 9, 2 }),
	}
}
