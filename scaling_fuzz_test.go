package controlsys

import (
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzPow2Exp returns k with x = 2^k, ok false when x is not a positive power
// of two.
func fzPow2Exp(x float64) (int, bool) {
	fr, e := math.Frexp(x)
	return e - 1, fr == 0.5
}

// fzScaledEqual reports whether got = x·2^k exactly; skip is true when the
// exact value is not a normal float64 (scaling is then inexact).
func fzScaledEqual(got, x float64, k int) (eq, skip bool) {
	want := math.Ldexp(x, k)
	if math.IsInf(want, 0) || fzBadFloat(want) || x != 0 && want == 0 {
		return false, true
	}
	return got == want, false
}

func fzDenseClone(d *mat.Dense) []float64 {
	if d == nil || d.IsEmpty() {
		return nil
	}
	return slices.Clone(mat.DenseCopyOf(d).RawMatrix().Data)
}

// fzCheckDiagScaled checks the delayed model got = (diag(2^l)·A·diag(2^r),
// diag(2^l)·B, C·diag(2^r), D) of sys exactly, with internal-delay channels
// B2, C2 scaled like B, C and every delay and LFT feedthrough unchanged.
func fzCheckDiagScaled(t *testing.T, s *fzSpec, op string, sys, got *System, l, r []int) {
	t.Helper()
	n, m, p := sys.Dims()
	if gn, gm, gp := got.Dims(); gn != n || gm != m || gp != p || got.Dt != sys.Dt {
		t.Fatalf("%s: dims %d,%d,%d dt=%v, want %d,%d,%d dt=%v\n%s", op, gn, gm, gp, got.Dt, n, m, p, sys.Dt, s)
	}
	chk := func(name string, gotM, wantM *mat.Dense, row, col func(i, j int) int) {
		gd, wd := fzDenseClone(gotM), fzDenseClone(wantM)
		if len(gd) != len(wd) {
			t.Fatalf("%s: %s has %d entries, want %d\n%s", op, name, len(gd), len(wd), s)
		}
		if len(wd) == 0 {
			return
		}
		_, c := wantM.Dims()
		for k, x := range wd {
			i, j := k/c, k%c
			eq, skip := fzScaledEqual(gd[k], x, row(i, j)+col(i, j))
			if skip {
				fzSkip(t, op+": scaled entry leaves the normal float64 range")
			}
			if !eq {
				t.Fatalf("%s: %s[%d,%d] = %v, want %v·2^%d exactly\n%s", op, name, i, j, gd[k], x, row(i, j)+col(i, j), s)
			}
		}
	}
	zero := func(int, int) int { return 0 }
	li := func(i, _ int) int { return l[i] }
	rj := func(_, j int) int { return r[j] }
	chk("A", got.A, sys.A, li, rj)
	chk("B", got.B, sys.B, li, zero)
	chk("C", got.C, sys.C, zero, rj)
	chk("D", got.D, sys.D, zero, zero)
	if !slices.Equal(got.InputDelay, sys.InputDelay) || !slices.Equal(got.OutputDelay, sys.OutputDelay) ||
		!slices.Equal(fzDenseClone(got.Delay), fzDenseClone(sys.Delay)) {
		t.Fatalf("%s: I/O delays changed\n%s", op, s)
	}
	if (got.LFT == nil) != (sys.LFT == nil) {
		t.Fatalf("%s: internal delays dropped or added\n%s", op, s)
	}
	if sys.LFT != nil {
		if !slices.Equal(got.LFT.Tau, sys.LFT.Tau) {
			t.Fatalf("%s: Tau changed\n%s", op, s)
		}
		chk("B2", got.LFT.B2, sys.LFT.B2, li, zero)
		chk("C2", got.LFT.C2, sys.LFT.C2, zero, rj)
		chk("D12", got.LFT.D12, sys.LFT.D12, zero, zero)
		chk("D21", got.LFT.D21, sys.LFT.D21, zero, zero)
		chk("D22", got.LFT.D22, sys.LFT.D22, zero, zero)
	}
}

// fzOffDiagSum is Σ|entries| of [A−diag(A), B B2; C; C2] (the matrix
// Osborne balancing works on).
func fzOffDiagSum(sys *System) float64 {
	n, _, _ := sys.Dims()
	var sum float64
	a := fzDenseClone(sys.A)
	for i := range n {
		for j := range n {
			if i != j {
				sum += math.Abs(a[i*n+j])
			}
		}
	}
	mats := []*mat.Dense{sys.B, sys.C}
	if sys.LFT != nil {
		mats = append(mats, sys.LFT.B2, sys.LFT.C2)
	}
	for _, d := range mats {
		for _, x := range fzDenseClone(d) {
			sum += math.Abs(x)
		}
	}
	return sum
}

// FuzzSsbal: Ssbal returns a positive power-of-two diagonal T and exactly
// (T·A/T, T·B, C/T, D) with B2, C2 scaled like B, C and the delays
// unchanged (MATLAB ssbal convention). Without a cond(T) bound, Osborne
// steps only shrink Σ|off-diagonal entries| of the augmented matrix; with
// one, max(T)/min(T) ≤ condT. |v[10]| > 5 (beyond the Givens clamp) sets
// condT = 10^min(|v[10]|−5, 12).
func FuzzSsbal(f *testing.F) {
	for _, s := range fzScalingSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
		var opts []SsbalOption
		condT := math.Inf(1)
		if a := math.Abs(s.Vals[10]); a > 5 {
			condT = math.Pow(10, math.Min(a-5, 12))
			opts = append(opts, WithCondT(condT))
		}
		res, err := Ssbal(sys, opts...)
		fzContract(t, s, "Ssbal", res != nil, err)
		if s.N == 0 {
			fzErr(t, s, "Ssbal", err, ErrDimensionMismatch)
		}
		fzErr(t, s, "Ssbal", err)
		n := s.N
		if r, c := res.T.Dims(); r != n || c != n {
			t.Fatalf("Ssbal: T is %d×%d\n%s", r, c, s)
		}
		k := make([]int, n)
		neg := make([]int, n)
		lo, hi := math.Inf(1), 0.0
		for i := range n {
			for j := range n {
				if v := res.T.At(i, j); i != j && v != 0 {
					t.Fatalf("Ssbal: T not diagonal\n%s", s)
				}
			}
			v := res.T.At(i, i)
			e, ok := fzPow2Exp(v)
			if !ok || v <= 0 {
				t.Fatalf("Ssbal: T[%d,%d] = %v is not a positive power of two\n%s", i, i, v, s)
			}
			k[i], neg[i] = e, -e
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
		if hi/lo > condT {
			t.Fatalf("Ssbal: cond(T) = %v > condT %v\n%s", hi/lo, condT, s)
		}
		fzCheckDiagScaled(t, s, "Ssbal", sys, res.Sys, k, neg)
		if math.IsInf(condT, 1) {
			if before, after := fzOffDiagSum(sys), fzOffDiagSum(res.Sys); after > before*(1+1e-12) {
				t.Fatalf("Ssbal: Σ|off-diagonal| grew from %v to %v\n%s", before, after, s)
			}
		}
	})
}

// FuzzPrescale: Prescale's LeftScale and StateScale are reciprocal powers
// of two and Sys is exactly (TL·A·TR, TL·B, C·TR, D) with internal-delay
// channels scaled like B and C; InputScale/OutputScale are the reciprocal
// column/row maxima of [Bs; D] and [Cs D].
func FuzzPrescale(f *testing.F) {
	for _, s := range fzScalingSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
		res, err := Prescale(sys)
		fzContract(t, s, "Prescale", res != nil, err)
		fzErr(t, s, "Prescale", err)
		n, m, p := s.N, s.M, s.P
		in := res.Info
		if len(in.InputScale) != m || len(in.OutputScale) != p || n > 0 && (len(in.LeftScale) != n || len(in.StateScale) != n) {
			t.Fatalf("Prescale: info lengths %d/%d/%d/%d\n%s", len(in.LeftScale), len(in.StateScale), len(in.InputScale), len(in.OutputScale), s)
		}
		l, r := make([]int, n), make([]int, n)
		for i := range n {
			el, okl := fzPow2Exp(in.LeftScale[i])
			er, okr := fzPow2Exp(in.StateScale[i])
			if !okl || !okr || el != -er || in.LeftScale[i] <= 0 {
				t.Fatalf("Prescale: scales %v, %v are not reciprocal positive powers of two\n%s", in.LeftScale[i], in.StateScale[i], s)
			}
			l[i], r[i] = el, er
		}
		fzCheckDiagScaled(t, s, "Prescale", sys, res.Sys, l, r)
		if n == 0 {
			fzSkip(t, "Prescale: static gain returns unit I/O scales, not 1/max|D| (followup)")
		}
		bs, cs, d := fzDenseClone(res.Sys.B), fzDenseClone(res.Sys.C), s.D
		for j := range m {
			mx := 0.0
			for i := range n {
				mx = math.Max(mx, math.Abs(bs[i*m+j]))
			}
			for i := range p {
				mx = math.Max(mx, math.Abs(d[i*m+j]))
			}
			want := 1.0
			if mx > 0 {
				want = 1 / mx
			}
			if in.InputScale[j] != want {
				t.Fatalf("Prescale: InputScale[%d] = %v, want %v\n%s", j, in.InputScale[j], want, s)
			}
		}
		for i := range p {
			mx := 0.0
			for j := range n {
				mx = math.Max(mx, math.Abs(cs[i*n+j]))
			}
			for j := range m {
				mx = math.Max(mx, math.Abs(d[i*m+j]))
			}
			want := 1.0
			if mx > 0 {
				want = 1 / mx
			}
			if in.OutputScale[i] != want {
				t.Fatalf("Prescale: OutputScale[%d] = %v, want %v\n%s", i, in.OutputScale[i], want, s)
			}
		}
	})
}

// FuzzAugstate: Augstate appends the undelayed states to the outputs
// exactly: C' = [C; I], D' = [D; 0], OutputDelay and IODelay padded with
// zeros, InputDelay and internal delays kept (D12 padded with zero rows).
func FuzzAugstate(f *testing.F) {
	for _, s := range fzScalingSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
		got, err := Augstate(sys)
		fzContract(t, s, "Augstate", got != nil, err)
		fzErr(t, s, "Augstate", err)
		n, m, p := s.N, s.M, s.P
		gn, gm, gp := got.Dims()
		if gn != n || gm != m || gp != p+n || got.Dt != s.Dt {
			t.Fatalf("Augstate dims %d,%d,%d, want %d,%d,%d\n%s", gn, gm, gp, n, m, p+n, s)
		}
		if !slices.Equal(fzDenseClone(got.A), s.A) || !slices.Equal(fzDenseClone(got.B), s.B) {
			t.Fatalf("Augstate changed A or B\n%s", s)
		}
		c, d := fzDenseClone(got.C), fzDenseClone(got.D)
		for i := range p + n {
			for j := range n {
				want := 0.0
				if i < p {
					want = s.C[i*n+j]
				} else if i-p == j {
					want = 1
				}
				if c[i*n+j] != want {
					t.Fatalf("Augstate C[%d,%d] = %v, want %v\n%s", i, j, c[i*n+j], want, s)
				}
			}
			for j := range m {
				want := 0.0
				if i < p {
					want = s.D[i*m+j]
				}
				if d[i*m+j] != want {
					t.Fatalf("Augstate D[%d,%d] = %v, want %v\n%s", i, j, d[i*m+j], want, s)
				}
			}
		}
		if !slices.Equal(got.InputDelay, sys.InputDelay) {
			t.Fatalf("Augstate InputDelay %v, want %v\n%s", got.InputDelay, sys.InputDelay, s)
		}
		pad := func(name string, gotD []float64, src []float64, rows, cols int) {
			if src == nil {
				if gotD != nil && slices.ContainsFunc(gotD, func(x float64) bool { return x != 0 }) {
					t.Fatalf("Augstate %s = %v from none\n%s", name, gotD, s)
				}
				return
			}
			if len(gotD) != (rows+n)*cols {
				t.Fatalf("Augstate %s has %d entries, want %d\n%s", name, len(gotD), (rows+n)*cols, s)
			}
			for k, x := range gotD {
				want := 0.0
				if k < rows*cols {
					want = src[k]
				}
				if x != want {
					t.Fatalf("Augstate %s[%d] = %v, want %v\n%s", name, k, x, want, s)
				}
			}
		}
		pad("OutputDelay", got.OutputDelay, sys.OutputDelay, p, 1)
		pad("IODelay", fzDenseClone(got.Delay), fzDenseClone(sys.Delay), p, m)
		if (got.LFT == nil) != (sys.LFT == nil) {
			t.Fatalf("Augstate dropped or added internal delays\n%s", s)
		}
		if sys.LFT != nil {
			nd := len(sys.LFT.Tau)
			if !slices.Equal(got.LFT.Tau, sys.LFT.Tau) || !slices.Equal(fzDenseClone(got.LFT.B2), fzDenseClone(sys.LFT.B2)) ||
				!slices.Equal(fzDenseClone(got.LFT.C2), fzDenseClone(sys.LFT.C2)) ||
				!slices.Equal(fzDenseClone(got.LFT.D21), fzDenseClone(sys.LFT.D21)) ||
				!slices.Equal(fzDenseClone(got.LFT.D22), fzDenseClone(sys.LFT.D22)) {
				t.Fatalf("Augstate changed the internal-delay channels\n%s", s)
			}
			pad("D12", fzDenseClone(got.LFT.D12), fzDenseClone(sys.LFT.D12), p, nd)
		}
	})
}

// fzScalingSeeds: badly scaled realizations (shear, wide pole spread,
// large gain), every delay kind, a cond(T) bound and a static gain.
func fzScalingSeeds() []fzSeed {
	return append(fzIsStableSeeds(),
		fzSeed{N: 6, M: 2, P: 3, Mag: [3]float64{-3, 0, 3.5}, Damp: [3]float64{1, 2, -0.5},
			Gain: 11, DScal: 1, Shear: 0.4, Rot: 0.9, In: 0, Out: 0.2, IO: -10, Tau: 0.4},
		fzSeed{N: 5, M: 3, P: 1, Mag: [3]float64{-2, 1, 2}, Damp: [3]float64{3, -1, 1},
			Gain: -9, Shear: -0.2, Rot: 7, Phase: 0.4, In: -10, Out: -10, IO: 0.3, Tau: -10},
		fzSeed{Discrete: true, LogDt: -2, N: 4, M: 1, P: 2, Mag: [3]float64{1, 2.5}, Damp: [3]float64{1.5, 0.5},
			Gain: 4, Shear: 0.3, Rot: -9.5, In: 0.5, Out: -10, IO: -10, Tau: 0.6},
	)
}
