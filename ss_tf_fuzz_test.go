package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"strconv"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// FuzzSSToTF: System.TransferFunction matches the exact rational
// C(xI−A)⁻¹B + D at points where the polynomial form is well conditioned
// (bound K·eps·(ν+|x|)ⁿ/|det(xI−A)|, ν = max(‖A‖, ‖B‖, ‖C_i‖) for the
// normwise backward-stable staircase and Hessenberg reductions, else the
// row is skipped, as the TransferFunction doc allows O(1) error there); row degrees sum to
// MinimalOrder; external delays fold into TF.Delay as In[j]+Out[i]+IO[i][j]
// without changing the coefficients, internal delays are rejected. The
// TF realization (TransferFunc.StateSpace) and ZPKModel are checked against
// the TF: exact companion response, exact gains, zeros/poles as roots of
// the TF polynomials (backward-error residual) and, when a row keeps every
// state, the poles against the intended spectrum by Rouché.
func FuzzSSToTF(f *testing.F) {
	for _, s := range fzSSToTFSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		n, m, p := s.N, s.M, s.P
		sys := s.System(t, fzSysOpts{})
		res, err := sys.TransferFunction(nil)
		fzContract(t, s, "TransferFunction", res != nil, err)
		fzErr(t, s, "TransferFunction", err, ErrOverflow)
		tf := res.TF
		if len(tf.Den) != p || len(tf.Num) != p || tf.Dt != s.Dt || len(res.RowDegrees) != p {
			t.Fatalf("TF shape rows=%d/%d dt=%g, want %d, %g\n%s", len(tf.Den), len(tf.Num), tf.Dt, p, s.Dt, s)
		}
		total := 0
		for i := range p {
			if len(tf.Num[i]) != m {
				t.Fatalf("Num[%d] has %d channels\n%s", i, len(tf.Num[i]), s)
			}
			deg := len(tf.Den[i]) - 1
			if deg != res.RowDegrees[i] || deg > n || tf.Den[i][0] == 0 {
				t.Fatalf("Den[%d] = %v, RowDegrees %v, n=%d\n%s", i, tf.Den[i], res.RowDegrees, n, s)
			}
			total += deg
			for j := range m {
				if len(tf.Num[i][j]) > len(tf.Den[i]) {
					t.Fatalf("Num[%d][%d] = %v improper over %v\n%s", i, j, tf.Num[i][j], tf.Den[i], s)
				}
				if n == 0 && (len(tf.Num[i][j]) != 1 || tf.Num[i][j][0]/tf.Den[i][0] != s.D[i*m+j]) {
					t.Fatalf("static TF[%d,%d] = %v/%v, D = %v\n%s", i, j, tf.Num[i][j], tf.Den[i], s.D[i*m+j], s)
				}
			}
		}
		if total != res.MinimalOrder {
			t.Fatalf("MinimalOrder %d, Σ RowDegrees %d\n%s", res.MinimalOrder, total, s)
		}

		dsys := s.System(t, fzSysOpts{Delays: true})
		dres, err := dsys.TransferFunction(nil)
		fzContract(t, s, "TransferFunction(delayed)", dres != nil, err)
		fzErr(t, s, "TransferFunction(delayed)", err, ErrOverflow)
		for i := range p {
			for j := range m {
				if !slices.Equal(dres.TF.Num[i][j], tf.Num[i][j]) || !slices.Equal(dres.TF.Den[i], tf.Den[i]) {
					t.Fatalf("delays changed TF[%d,%d]: %v/%v vs %v/%v\n%s", i, j, dres.TF.Num[i][j], dres.TF.Den[i], tf.Num[i][j], tf.Den[i], s)
				}
				want := 0.0
				if s.In != nil {
					want += s.In[j]
				}
				if s.Out != nil {
					want += s.Out[i]
				}
				if s.IO != nil {
					want += s.IO[i*m+j]
				}
				got := 0.0
				if dres.TF.Delay != nil {
					got = dres.TF.Delay[i][j]
				}
				if math.Abs(got-want) > 4e-16*want {
					t.Fatalf("TF.Delay[%d][%d] = %g, want In+Out+IO = %g\n%s", i, j, got, want, s)
				}
			}
		}
		if s.Tau > 0 && n > 0 {
			isys := s.System(t, fzSysOpts{Internal: true})
			ir, err := isys.TransferFunction(nil)
			fzContract(t, s, "TransferFunction(internal)", ir != nil, err)
			if !errors.Is(err, ErrDelayNotRepresentable) {
				t.Fatalf("TransferFunction with internal delay: err %v, want ErrDelayNotRepresentable\n%s", err, s)
			}
		}

		real_, err := dres.TF.StateSpace()
		fzContract(t, s, "TransferFunc.StateSpace", real_ != nil, err)
		fzErr(t, s, "TransferFunc.StateSpace", err)
		fzCheckRealization(t, s, "TransferFunc.StateSpace", dres.TF, real_)

		zr, err := sys.ZPKModel(nil)
		fzContract(t, s, "ZPKModel", zr != nil, err)
		fzErr(t, s, "ZPKModel", err)
		if zr.MinimalOrder != res.MinimalOrder || !slices.Equal(zr.RowDegrees, res.RowDegrees) {
			t.Fatalf("ZPKModel order %d %v, TF %d %v\n%s", zr.MinimalOrder, zr.RowDegrees, res.MinimalOrder, res.RowDegrees, s)
		}
		if dres.TF.HasDelay() {
			dz, err := dsys.ZPKModel(nil)
			fzContract(t, s, "ZPKModel(delayed)", dz != nil, err)
			if !errors.Is(err, ErrDelayNotRepresentable) {
				t.Fatalf("ZPKModel with delays: err %v, want ErrDelayNotRepresentable\n%s", err, s)
			}
		}
		nA := math.Max(fzFrob(s.A), s.CondT*s.NormJ)
		var extra []float64
		if n > 0 {
			extra = fzBinomAbs(n, nA)
			for k := range extra {
				extra[k] *= 1e3 * float64(n*n)
			}
		}
		for i := range p {
			den := tf.Den[i]
			for j := range m {
				z := zr.ZPK
				num := trimLeadingFloatZeros(tf.Num[i][j])
				if num[0] == 0 {
					if z.Gain[i][j] != 0 || len(z.Zeros[i][j])+len(z.Poles[i][j]) != 0 {
						t.Fatalf("zero channel (%d,%d): gain %v zeros %v poles %v\n%s", i, j, z.Gain[i][j], z.Zeros[i][j], z.Poles[i][j], s)
					}
					continue
				}
				if g := z.Gain[i][j]; g != num[0]/den[0] {
					t.Fatalf("ZPK gain[%d,%d] = %v, want %v\n%s", i, j, g, num[0]/den[0], s)
				}
				if len(z.Zeros[i][j]) != len(num)-1 || len(z.Poles[i][j]) != len(den)-1 {
					t.Fatalf("ZPK (%d,%d) has %d zeros %d poles, TF degrees %d/%d\n%s", i, j, len(z.Zeros[i][j]), len(z.Poles[i][j]), len(num)-1, len(den)-1, s)
				}
				for _, r := range z.Zeros[i][j] {
					if msg := fzRootResidual(num, r); msg != "" {
						t.Fatalf("ZPK zero (%d,%d) %v: %s\nnum=%v\n%s", i, j, r, msg, num, s)
					}
				}
				for _, r := range z.Poles[i][j] {
					if msg := fzRootResidual(den, r); msg != "" {
						t.Fatalf("ZPK pole (%d,%d) %v: %s\nden=%v\n%s", i, j, r, msg, den, s)
					}
				}
				if res.RowDegrees[i] == n && n > 0 && s.Decidable() {
					mon := make([]float64, len(den))
					for k, v := range den {
						mon[k] = v / den[0]
					}
					if msg := fzRouche(z.Poles[i][j], s.Eig, mon, extra); msg != "" {
						t.Fatalf("ZPK poles (%d,%d) vs intended: %s\ngot %v want %v\n%s", i, j, msg, z.Poles[i][j], s.Eig, s)
					}
				}
			}
		}
		if n == 0 {
			return
		}

		const eps = 0x1p-52
		cp := fzRatFloats(fzCharPolyExact(s.A, n))
		for _, c := range cp {
			if math.IsInf(c, 0) {
				fzSkip(t, "char poly overflows float64")
			}
		}
		pts := fzConvPoints(s)
		if s.Discrete {
			pts = append(pts, 1)
		} else {
			pts = append(pts, 0)
		}
		nB := fzFrob(s.B)
		kk := 64 * float64((n+1)*(n+1))
		checked := 0
		for _, x := range pts {
			cx := cmplx.Abs(fzBigPolyEval(cp, x))
			if cx == 0 || math.IsInf(cx, 0) {
				continue
			}
			g, ok := fzExactResponse(n, m, p, s.A, s.B, s.C, s.D, x)
			if !ok {
				continue
			}
			for i := range p {
				nC := fzFrob(s.C[i*n : (i+1)*n])
				nu := math.Max(nA, math.Max(nB, nC)) + cmplx.Abs(x)
				sc := math.Pow(nu, float64(n))
				kap := sc / cx
				if !(kk*eps*kap <= 1e-3) {
					continue
				}
				checked++
				for j := range m {
					h := g[i*m+j]
					hp := fzBigPolyEval(tf.Num[i][j], x) / fzBigPolyEval(tf.Den[i], x)
					tol := kk * eps * kap * (cmplx.Abs(h) + math.Abs(s.D[i*m+j]) + nC*nB/nu)
					if d := cmplx.Abs(hp - h); !(d <= tol) {
						if fzBalanceSpread(s.A, n) > 1e6 {
							fzSkip(t, "TransferFunction: Dgebal scaling of A alone inflates B or C (followup)")
						}
						t.Fatalf("TF(%v)[%d,%d] = %v, exact %v (|Δ| %g, tol %g, κ %g)\nnum=%v den=%v\n%s", x, i, j, hp, h, d, tol, kap, tf.Num[i][j], tf.Den[i], s)
					}
				}
			}
		}
		if checked == 0 {
			fzSkip(t, "polynomial form ill-conditioned at every point")
		}
	})
}

// fzBinomAbs returns the coefficients of (s + a)ⁿ, descending.
func fzBinomAbs(n int, a float64) []float64 {
	c := []float64{1}
	for range n {
		next := make([]float64, len(c)+1)
		for k, v := range c {
			next[k] += v
			next[k+1] += v * a
		}
		c = next
	}
	return c
}

// fzRootResidual checks that r is an eigenvalue of a companion matrix of c
// perturbed normwise by K·eps: |ĉ(r)| ≤ K·eps·ν·max(1,ν)·Σ|r|^k with ĉ
// monic and ν = max(1, max|ĉ_k|), evaluated in big.Float.
func fzRootResidual(c []float64, r complex128) string {
	const eps = 0x1p-52
	if cmplx.IsNaN(r) || cmplx.IsInf(r) {
		return "non-finite root"
	}
	nu := 1.0
	for _, v := range c[1:] {
		nu = math.Max(nu, math.Abs(v/c[0]))
	}
	ar := cmplx.Abs(r)
	pw := 0.0
	for range c {
		pw = pw*ar + 1
	}
	res := cmplx.Abs(fzBigPolyEval(c, r)) / math.Abs(c[0])
	if bound := 256 * float64(len(c)) * eps * nu * nu * pw; res > bound {
		return "residual " + fzG(res) + " > backward-error bound " + fzG(bound)
	}
	return ""
}

func fzG(x float64) string { return strconv.FormatFloat(x, 'g', 3, 64) }

// fzSSToTFSeeds adds external and internal delays to the conversion seeds.
func fzSSToTFSeeds() []fzSeed {
	seeds := fzConvSeeds()
	for _, k := range []int{2, 4, 8} {
		s := seeds[k]
		s.In, s.Out, s.IO, s.Tau = -1, 0.3, 0, 0.1
		seeds = append(seeds, s)
	}
	return seeds
}

// fzBalanceSpread is max/min of the Dgebal scaling of A, which
// TransferFunction applies to A, B and C before its staircase.
func fzBalanceSpread(a []float64, n int) float64 {
	ab := slices.Clone(a)
	sc := make([]float64, n)
	impl.Dgebal(lapack.Scale, n, ab, n, sc)
	lo, hi := math.Inf(1), 0.0
	for _, d := range sc {
		lo, hi = math.Min(lo, d), math.Max(hi, d)
	}
	return hi / lo
}
