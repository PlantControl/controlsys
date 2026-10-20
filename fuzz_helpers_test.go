package controlsys

// Shared fuzz infrastructure (epic TP5HGX).
//
// Input: every target uses the canonical tuple (kind uint8, v [fzNVals]float64)
// via fzFuzz, so corpora stay compatible and mutation stays local: each float
// is clamped then mapped (usually through 10^x); no byte-seeded RNG. fzDecode
// turns the tuple into an fzSpec with known (intended) eigenvalues:
//
//	kind bit 0     discrete (z = exp(s·Dt))
//	kind bits 1-2  inputs  m = 1 + min(bits, 2)
//	kind bits 3-4  outputs p = 1 + min(bits, 2)
//	kind bits 5-7  order   n = min(bits, 6)
//	v[2k], v[2k+1] block k (k < 3): natural frequency 10^clamp(v,-13,4)
//	               (v < -30: poles at s = 0) and damping code b:
//	               ζ = 10^clamp(2-|b|, -10, 2), ζ = 0 for |b| > 12.5,
//	               b < 0 mirrors the block into Re s > 0. ζ >= 1 gives two
//	               real poles; an odd order's last block is the single pole
//	               ∓ω.
//	v[6]  gain sign·10^(clamp(|v|,0,12)-6) on C;  v[7] D scale
//	v[8]  Dt = 10^clamp(v,-4,2);  v[9] shear 10^clamp(v,-8,0.5) (v <= -9: 0)
//	v[10] Givens angle;  v[11] B/C/D texture phase
//	v[12..14] input / output / I/O delays, v[15] internal delay τ: fzDelay
//
// A = T·J·T⁻¹ with J normal block-diagonal and T = Givens·(I + shear·U),
// so Bauer–Fike bounds the eigenvalue error of A and of any backward-stable
// eigensolver by fzSpec.EigErr; Decidable reports whether every intended
// pole is clear of that error and of the library's 1e-10·max(1,|p|) boundary
// band (stability.go). Targets must verify against intended values or the
// exact oracles below, never against another library routine.
//
// Policy: failure = wrong value, wrong verdict, unstable design returned,
// panic (fzRecover dumps the spec) or a Go error-contract breach
// (fzContract: nil value with nil error, value with error). Undecidable
// inputs and documented unsupported/unattainable errors call fzSkip (counted
// per reason; fzStats logs counts on seed/replay runs) — never failure. Any
// other error on a decoded finite input is a value->error failure (fzErr).
//
// Oracles: fzCharPolyExact (Faddeev–LeVerrier over big.Rat on the float64
// matrix), fzRouthRight / fzSchurCohnOutside (exact root counts right of a
// line / outside a circle), fzBigPolyEval (big.Float), fzExactResponse
// (exact rational C(sI−A)⁻¹B+D), fzSweepMax (dense grid + golden-section
// refinement, fzFreqGrid clusters samples at the poles), fzQuasiRHP
// (argument principle, wraps sweepOracle; oracleRHPRoots and
// sisoDelayLoop.rectRoots are the dense variants).
//
// Seeds must run sub-second; fzFuzz calls skipSlowUnderRace. Long runs:
// scripts/fuzz.sh (arm64 fuzzing, amd64 replay of new crashers).

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"runtime/debug"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const fzNVals = 16

// fzSeed names the tuple fields for readable f.Add seeds.
type fzSeed struct {
	Discrete    bool
	N, M, P     int
	Mag, Damp   [3]float64
	Gain, DScal float64
	LogDt       float64
	Shear, Rot  float64
	Phase       float64
	In, Out, IO float64
	Tau         float64
}

// fzNoDelay disables a delay field.
const fzNoDelay = -10

func (s fzSeed) encode() (uint8, [fzNVals]float64) {
	k := uint8(0)
	if s.Discrete {
		k |= 1
	}
	k |= uint8(max(s.M-1, 0)&3) << 1
	k |= uint8(max(s.P-1, 0)&3) << 3
	k |= uint8(s.N&7) << 5
	return k, [fzNVals]float64{
		s.Mag[0], s.Damp[0], s.Mag[1], s.Damp[1], s.Mag[2], s.Damp[2],
		s.Gain, s.DScal, s.LogDt, s.Shear, s.Rot, s.Phase, s.In, s.Out, s.IO, s.Tau,
	}
}

func fzAddSeed(f *testing.F, s fzSeed) {
	k, v := s.encode()
	f.Add(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15])
}

// fzFuzz runs body on every decoded input, skipping under -race and
// turning panics into failures that print the spec.
func fzFuzz(f *testing.F, body func(t *testing.T, s *fzSpec)) {
	fzStats(f)
	f.Fuzz(func(t *testing.T, kind uint8, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 float64) {
		skipSlowUnderRace(t)
		s := fzDecode(t, kind, [fzNVals]float64{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15})
		defer fzRecover(t, s)
		body(t, s)
	})
}

func fzRecover(t *testing.T, s *fzSpec) {
	if r := recover(); r != nil {
		t.Fatalf("panic: %v\n%s\n%s", r, s, debug.Stack())
	}
}

var fzSkips sync.Map

func fzSkip(t *testing.T, reason string) {
	t.Helper()
	c, _ := fzSkips.LoadOrStore(reason, new(atomic.Int64))
	c.(*atomic.Int64).Add(1)
	t.Skipf("fuzz skip [%s]", reason)
}

func fzStats(f *testing.F) {
	f.Cleanup(func() {
		var lines []string
		fzSkips.Range(func(k, v any) bool {
			lines = append(lines, fmt.Sprintf("%s=%d", k, v.(*atomic.Int64).Load()))
			return true
		})
		if len(lines) > 0 {
			sort.Strings(lines)
			f.Logf("fuzz skips: %s", strings.Join(lines, " "))
		}
	})
}

// fzErr returns true when err is nil. A documented error is a counted skip;
// any other error fails.
func fzErr(t *testing.T, s *fzSpec, op string, err error, documented ...error) bool {
	t.Helper()
	if err == nil {
		return true
	}
	for _, d := range documented {
		if errors.Is(err, d) {
			fzSkip(t, op+": "+d.Error())
		}
	}
	t.Fatalf("%s: undocumented error %v\n%s", op, err, s)
	return false
}

// fzContract fails on nil value with nil error or value with error.
func fzContract(t *testing.T, s *fzSpec, op string, hasValue bool, err error) {
	t.Helper()
	if err == nil && !hasValue {
		t.Fatalf("%s: nil value with nil error\n%s", op, s)
	}
	if err != nil && hasValue {
		t.Fatalf("%s: value returned with error %v\n%s", op, err, s)
	}
}

func fzBadFloat(x float64) bool {
	return math.IsNaN(x) || math.IsInf(x, 0) || (x != 0 && math.Abs(x) < 0x1p-1022)
}

func fzClamp(x, lo, hi float64) float64 { return math.Min(math.Max(x, lo), hi) }

// fzDelay maps v to 0 for v < -3, else 10^clamp(v,-3,1.3) seconds, rounded
// to whole samples when discrete.
func fzDelay(v float64, discrete bool) float64 {
	if v < -3 {
		return 0
	}
	d := math.Pow(10, fzClamp(v, -3, 1.3))
	if discrete {
		return math.Round(d)
	}
	return d
}

// fzSpec is a decoded input. Eig holds the intended eigenvalues of A (z for
// discrete models) with conjugates adjacent; S the continuous-time ones.
type fzSpec struct {
	Kind     uint8
	Vals     [fzNVals]float64
	Discrete bool
	N, M, P  int
	Dt       float64
	S, Eig   []complex128
	A, B, C  []float64
	D        []float64
	CondT    float64
	NormJ    float64
	In, Out  []float64
	IO       []float64
	Tau      float64
}

func (s *fzSpec) String() string {
	return fmt.Sprintf("kind=%#02x vals=%v\nn=%d m=%d p=%d discrete=%v dt=%g condT=%.3g eig=%v\nA=%v\nB=%v\nC=%v\nD=%v in=%v out=%v io=%v tau=%g",
		s.Kind, s.Vals, s.N, s.M, s.P, s.Discrete, s.Dt, s.CondT, s.Eig, s.A, s.B, s.C, s.D, s.In, s.Out, s.IO, s.Tau)
}

func fzDecode(t *testing.T, kind uint8, v [fzNVals]float64) *fzSpec {
	t.Helper()
	for _, x := range v {
		if fzBadFloat(x) {
			fzSkip(t, "non-finite or denormal input")
		}
	}
	s := &fzSpec{Kind: kind, Vals: v, Discrete: kind&1 != 0}
	s.M = 1 + min(int(kind>>1)&3, 2)
	s.P = 1 + min(int(kind>>3)&3, 2)
	s.N = min(int(kind>>5)&7, 6)
	if s.Discrete {
		s.Dt = math.Pow(10, fzClamp(v[8], -4, 2))
	}
	n, m, p := s.N, s.M, s.P
	J := make([]float64, n*n)
	for k := 0; 2*k < n; k++ {
		w := 0.0
		if v[2*k] >= -30 {
			w = math.Pow(10, fzClamp(v[2*k], -13, 4))
		}
		b := v[2*k+1]
		sign := 1.0
		if b < 0 {
			sign = -1
		}
		i := 2 * k
		if i == n-1 {
			s.S = append(s.S, complex(-sign*w, 0))
			continue
		}
		zeta := 0.0
		if math.Abs(b) <= 12.5 {
			zeta = math.Pow(10, fzClamp(2-math.Abs(b), -10, 2))
		}
		if zeta < 1 {
			re, im := -sign*zeta*w, w*math.Sqrt(1-zeta*zeta)
			s.S = append(s.S, complex(re, im), complex(re, -im))
		} else {
			q := zeta + math.Sqrt(zeta*zeta-1)
			s.S = append(s.S, complex(-sign*w/q, 0), complex(-sign*w*q, 0))
		}
	}
	for k := range n {
		e := s.S[k]
		if s.Discrete {
			if math.Abs(real(e)*s.Dt) > 300 {
				fzSkip(t, "exp(s·Dt) out of range")
			}
			e = cmplx.Exp(e * complex(s.Dt, 0))
			if imag(s.S[k]) == 0 {
				e = complex(real(e), 0)
			}
		}
		s.Eig = append(s.Eig, e)
	}
	for k := 0; k < n; {
		e := s.Eig[k]
		if imag(s.S[k]) != 0 {
			J[k*n+k], J[k*n+k+1] = real(e), math.Abs(imag(e))
			J[(k+1)*n+k], J[(k+1)*n+k+1] = -math.Abs(imag(e)), real(e)
			k += 2
			continue
		}
		J[k*n+k] = real(e)
		k++
	}
	for _, x := range J {
		if fzBadFloat(x) {
			fzSkip(t, "denormal pole")
		}
		s.NormJ = math.Max(s.NormJ, math.Abs(x))
	}
	for _, e := range s.Eig {
		s.NormJ = math.Max(s.NormJ, cmplx.Abs(e))
	}
	shear := 0.0
	if v[9] > -9 {
		shear = math.Pow(10, fzClamp(v[9], -8, 0.5))
	}
	theta := fzClamp(v[10], -4, 4)
	phase := fzClamp(v[11], -1e3, 1e3)
	s.CondT = 1
	if n > 0 {
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
		var Ti, tmp, A mat.Dense
		if err := Ti.Inverse(T); err != nil {
			fzSkip(t, "singular similarity")
		}
		s.CondT = mat.Cond(T, 2)
		tmp.Mul(T, mat.NewDense(n, n, J))
		A.Mul(&tmp, &Ti)
		s.A = slices.Clone(A.RawMatrix().Data)
	}
	g := math.Pow(10, fzClamp(math.Abs(v[6]), 0, 12)-6)
	if v[6] < 0 {
		g = -g
	}
	tex := func(i, j int, a, b float64) float64 { return math.Cos(phase + a*float64(i+1) + b*float64(j+1)) }
	s.B = make([]float64, n*m)
	for i := range n {
		for j := range m {
			s.B[i*m+j] = tex(i, j, 1.3, 2.1)
		}
	}
	s.C = make([]float64, p*n)
	for i := range p {
		for j := range n {
			s.C[i*n+j] = g * tex(i, j, 0.7, 1.9)
		}
	}
	s.D = make([]float64, p*m)
	dsc := fzClamp(v[7], -1e3, 1e3)
	for i := range p {
		for j := range m {
			s.D[i*m+j] = dsc * tex(i, j, 2.9, 0.4)
		}
	}
	for _, x := range slices.Concat(s.A, s.B, s.C, s.D) {
		if fzBadFloat(x) {
			fzSkip(t, "denormal realization entry")
		}
	}
	if d := fzDelay(v[12], s.Discrete); d > 0 {
		s.In = make([]float64, m)
		for j := range m {
			s.In[j] = d * float64(j+1)
		}
	}
	if d := fzDelay(v[13], s.Discrete); d > 0 {
		s.Out = make([]float64, p)
		for i := range p {
			s.Out[i] = d * float64(p-i)
		}
	}
	if d := fzDelay(v[14], s.Discrete); d > 0 {
		s.IO = make([]float64, p*m)
		for k := range s.IO {
			s.IO[k] = d * float64(k%3)
		}
	}
	s.Tau = fzDelay(v[15], s.Discrete)
	return s
}

// EigErr bounds |λ(A) − intended| plus the backward error of a stable
// eigensolver on A, by Bauer–Fike with eigenvector matrix T·Q.
func (s *fzSpec) EigErr() float64 {
	const eps = 0x1p-52
	return 100 * float64(max(s.N, 1)) * eps * math.Pow(s.CondT, 3) * math.Max(s.NormJ, 1)
}

// Margin is the distance of eigenvalue e to the stability boundary.
func (s *fzSpec) Margin(e complex128) float64 {
	if s.Discrete {
		return math.Abs(cmplx.Abs(e) - 1)
	}
	return math.Abs(real(e))
}

// Decidable reports whether every intended eigenvalue is clear of the
// stability boundary by more than the numerical error and the library's
// boundary band.
func (s *fzSpec) Decidable() bool {
	d := s.EigErr()
	for _, e := range s.Eig {
		if s.Margin(e) <= 4*math.Max(d, 1e-10*math.Max(1, cmplx.Abs(e)+d)) {
			return false
		}
	}
	return true
}

// Unstable counts intended eigenvalues outside the stability region.
func (s *fzSpec) Unstable() int {
	k := 0
	for _, e := range s.Eig {
		if s.Discrete && cmplx.Abs(e) >= 1 || !s.Discrete && real(e) >= 0 {
			k++
		}
	}
	return k
}

type fzSysOpts struct {
	Delays   bool
	Internal bool
}

// System builds the model; I/O delays and an internal delay loop are
// added only when requested.
func (s *fzSpec) System(t *testing.T, o fzSysOpts) *System {
	t.Helper()
	sys, err := NewFromSlices(s.N, s.M, s.P, s.A, s.B, s.C, s.D, s.Dt)
	if err != nil {
		t.Fatalf("NewFromSlices: %v\n%s", err, s)
	}
	if o.Delays {
		if s.In != nil {
			if err := sys.SetInputDelay(s.In); err != nil {
				t.Fatalf("SetInputDelay: %v\n%s", err, s)
			}
		}
		if s.Out != nil {
			if err := sys.SetOutputDelay(s.Out); err != nil {
				t.Fatalf("SetOutputDelay: %v\n%s", err, s)
			}
		}
		if s.IO != nil && !(o.Internal && s.Tau > 0) {
			if err := sys.SetDelay(mat.NewDense(s.P, s.M, s.IO)); err != nil {
				t.Fatalf("SetDelay: %v\n%s", err, s)
			}
		}
	}
	if o.Internal && s.Tau > 0 && s.N > 0 {
		b2 := make([]float64, s.N)
		c2 := make([]float64, s.N)
		for i := range s.N {
			b2[i] = s.B[i*s.M]
			c2[i] = 0.5 * math.Sin(float64(i)+s.Vals[11])
		}
		err := sys.SetInternalDelay([]float64{s.Tau}, mat.NewDense(s.N, 1, b2), mat.NewDense(1, s.N, c2),
			mat.NewDense(s.P, 1, nil), mat.NewDense(1, s.M, nil), mat.NewDense(1, 1, nil))
		if err != nil {
			t.Fatalf("SetInternalDelay: %v\n%s", err, s)
		}
	}
	return sys
}

func fzRat(x float64) *big.Rat {
	r := new(big.Rat)
	if r.SetFloat64(x) == nil {
		panic(fmt.Sprintf("fzRat(%v)", x))
	}
	return r
}

// fzCharPolyExact returns det(λI − A) exactly, descending, for the n×n
// row-major float64 matrix a (Faddeev–LeVerrier over big.Rat).
func fzCharPolyExact(a []float64, n int) []*big.Rat {
	return fzCharPolyRat(fzRatPoly(a), n)
}

// fzCharPolyRat is fzCharPolyExact for a rational matrix. It runs
// Faddeev–LeVerrier fraction-free on L·A, L the lcm of the denominators
// (exact /k over the integers), and returns c'_k / L^k; big.Rat
// normalisation per product dominated the cost.
func fzCharPolyRat(A []*big.Rat, n int) []*big.Rat {
	L := big.NewInt(1)
	var g big.Int
	for _, x := range A {
		d := x.Denom()
		g.GCD(nil, nil, L, d)
		L.Mul(L, new(big.Int).Quo(d, &g))
	}
	Ai := make([]*big.Int, n*n)
	for i, x := range A {
		v := new(big.Int).Mul(x.Num(), L)
		Ai[i] = v.Quo(v, x.Denom())
	}
	M := make([]*big.Int, n*n)
	AM := make([]*big.Int, n*n)
	for i := range M {
		M[i], AM[i] = new(big.Int), new(big.Int)
	}
	ck := big.NewInt(1)
	Lk := big.NewInt(1)
	c := make([]*big.Rat, n+1)
	c[0] = big.NewRat(1, 1)
	var t, tr big.Int
	for k := 1; k <= n; k++ {
		for i := range n {
			for j := range n {
				v := AM[i*n+j].SetInt64(0)
				for l := range n {
					v.Add(v, t.Mul(Ai[i*n+l], M[l*n+j]))
				}
				if i == j {
					v.Add(v, ck)
				}
			}
		}
		M, AM = AM, M
		tr.SetInt64(0)
		for i := range n {
			for l := range n {
				tr.Add(&tr, t.Mul(Ai[i*n+l], M[l*n+i]))
			}
		}
		ck = new(big.Int).Quo(&tr, big.NewInt(int64(-k)))
		Lk.Mul(Lk, L)
		c[k] = new(big.Rat).SetFrac(ck, Lk)
	}
	return c
}

func fzRatPoly(c []float64) []*big.Rat {
	out := make([]*big.Rat, len(c))
	for i, x := range c {
		out[i] = fzRat(x)
	}
	return out
}

// fzRouthRight counts roots of the descending polynomial c with Re s > sigma
// exactly. ok is false when the leading coefficient or a Routh pivot is zero
// (a root on the line, a symmetric pair, or a degenerate table).
func fzRouthRight(c []*big.Rat, sigma float64) (count int, ok bool) {
	a := fzTaylorShift(c, fzRat(sigma))
	n := len(a) - 1
	if n < 0 || a[0].Sign() == 0 {
		return 0, false
	}
	w := n/2 + 2
	row := func(off int) []*big.Rat {
		r := make([]*big.Rat, w)
		for j := range w {
			r[j] = new(big.Rat)
			if k := off + 2*j; k <= n {
				r[j].Set(a[k])
			}
		}
		return r
	}
	r0, r1 := row(0), row(1)
	signs := []int{r0[0].Sign()}
	for k := 1; k <= n; k++ {
		if r1[0].Sign() == 0 {
			return 0, false
		}
		signs = append(signs, r1[0].Sign())
		next := make([]*big.Rat, w)
		for j := range w {
			next[j] = new(big.Rat)
			if j+1 < w {
				var x, y big.Rat
				x.Mul(r1[0], r0[j+1])
				y.Mul(r0[0], r1[j+1])
				next[j].Quo(x.Sub(&x, &y), r1[0])
			}
		}
		r0, r1 = r1, next
	}
	for i := 1; i < len(signs); i++ {
		if signs[i] != signs[i-1] {
			count++
		}
	}
	return count, true
}

// fzTaylorShift returns c(s + sigma), descending.
func fzTaylorShift(c []*big.Rat, sigma *big.Rat) []*big.Rat {
	a := make([]*big.Rat, len(c))
	for i := range c {
		a[i] = new(big.Rat).Set(c[i])
	}
	n := len(a) - 1
	var t big.Rat
	for i := range n {
		for j := 1; j <= n-i; j++ {
			a[j].Add(a[j], t.Mul(sigma, a[j-1]))
		}
	}
	return a
}

func fzRatPolyMul(a, b []*big.Rat) []*big.Rat {
	out := make([]*big.Rat, len(a)+len(b)-1)
	for i := range out {
		out[i] = new(big.Rat)
	}
	var t big.Rat
	for i := range a {
		for j := range b {
			out[i+j].Add(out[i+j], t.Mul(a[i], b[j]))
		}
	}
	return out
}

// fzSchurCohnOutside counts roots of the descending polynomial c with
// |z| > rho exactly, by z = rho(1+s)/(1−s) and Routh. ok is false when a
// root lies on |z| = rho (or at z = −rho) or the table is degenerate.
func fzSchurCohnOutside(c []*big.Rat, rho float64) (int, bool) {
	n := len(c) - 1
	r := fzRat(rho)
	plus := []*big.Rat{big.NewRat(1, 1), big.NewRat(1, 1)}
	minus := []*big.Rat{big.NewRat(-1, 1), big.NewRat(1, 1)}
	q := []*big.Rat{new(big.Rat)}
	pw := big.NewRat(1, 1)
	for k := n; k >= 0; k-- {
		term := []*big.Rat{new(big.Rat).Mul(c[k], pw)}
		for range n - k {
			term = fzRatPolyMul(term, plus)
		}
		for range k {
			term = fzRatPolyMul(term, minus)
		}
		if len(q) < len(term) {
			q = append(make([]*big.Rat, len(term)-len(q)), q...)
			for i := range q {
				if q[i] == nil {
					q[i] = new(big.Rat)
				}
			}
		}
		off := len(q) - len(term)
		for i := range term {
			q[off+i].Add(q[off+i], term[i])
		}
		pw = new(big.Rat).Mul(pw, r)
	}
	return fzRouthRight(q, 0)
}

const fzBigPrec = 512

// fzBigPolyEval evaluates the descending float64 polynomial c at s in
// big.Float and rounds to complex128.
func fzBigPolyEval(c []float64, s complex128) complex128 {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(fzBigPrec).SetFloat64(x) }
	sr, si := nf(real(s)), nf(imag(s))
	vr, vi := nf(0), nf(0)
	for _, x := range c {
		a := nf(0).Mul(vr, sr)
		a.Sub(a, nf(0).Mul(vi, si))
		b := nf(0).Mul(vr, si)
		b.Add(b, nf(0).Mul(vi, sr))
		vr, vi = a.Add(a, nf(x)), b
	}
	r, _ := vr.Float64()
	i, _ := vi.Float64()
	return complex(r, i)
}

type fzCRat struct{ re, im big.Rat }

func (z *fzCRat) isZero() bool { return z.re.Sign() == 0 && z.im.Sign() == 0 }

func fzCMul(a, b *fzCRat) *fzCRat {
	var r fzCRat
	var t big.Rat
	r.re.Mul(&a.re, &b.re)
	r.re.Sub(&r.re, t.Mul(&a.im, &b.im))
	r.im.Mul(&a.re, &b.im)
	r.im.Add(&r.im, t.Mul(&a.im, &b.re))
	return &r
}

func fzCQuo(a, b *fzCRat) *fzCRat {
	var d, t big.Rat
	d.Mul(&b.re, &b.re)
	d.Add(&d, t.Mul(&b.im, &b.im))
	var conj fzCRat
	conj.re.Set(&b.re)
	conj.im.Neg(&b.im)
	r := fzCMul(a, &conj)
	r.re.Quo(&r.re, &d)
	r.im.Quo(&r.im, &d)
	return r
}

// fzExactResponse returns C(sI − A)⁻¹B + D exactly for the float64 model,
// rounded to complex128; ok is false when sI − A is exactly singular.
func fzExactResponse(n, m, p int, a, b, c, d []float64, s complex128) ([]complex128, bool) {
	cols := n + m
	M := make([]*fzCRat, n*cols)
	for i := range n {
		for j := range cols {
			z := new(fzCRat)
			if j < n {
				z.re.Neg(fzRat(a[i*n+j]))
				if i == j {
					z.re.Add(&z.re, fzRat(real(s)))
					z.im.Set(fzRat(imag(s)))
				}
			} else {
				z.re.Set(fzRat(b[i*m+j-n]))
			}
			M[i*cols+j] = z
		}
	}
	for k := range n {
		piv := -1
		for i := k; i < n; i++ {
			if !M[i*cols+k].isZero() {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		for j := range cols {
			M[k*cols+j], M[piv*cols+j] = M[piv*cols+j], M[k*cols+j]
		}
		for i := range n {
			if i == k || M[i*cols+k].isZero() {
				continue
			}
			f := fzCQuo(M[i*cols+k], M[k*cols+k])
			for j := k; j < cols; j++ {
				t := fzCMul(f, M[k*cols+j])
				M[i*cols+j].re.Sub(&M[i*cols+j].re, &t.re)
				M[i*cols+j].im.Sub(&M[i*cols+j].im, &t.im)
			}
		}
	}
	out := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			var acc fzCRat
			acc.re.Set(fzRat(d[i*m+j]))
			for k := range n {
				x := fzCQuo(M[k*cols+n+j], M[k*cols+k])
				var t big.Rat
				ck := fzRat(c[i*n+k])
				acc.re.Add(&acc.re, t.Mul(ck, &x.re))
				acc.im.Add(&acc.im, new(big.Rat).Mul(ck, &x.im))
			}
			re, _ := acc.re.Float64()
			im, _ := acc.im.Float64()
			out[i*m+j] = complex(re, im)
		}
	}
	return out, true
}

// fzFreqGrid returns sorted frequencies: a dense log grid around the pole
// magnitudes plus geometric clusters at each pole frequency (continuous-time
// poles s; capped at the Nyquist frequency when dt > 0).
func fzFreqGrid(s []complex128, dt float64, dense int) []float64 {
	wmax := math.Inf(1)
	if dt > 0 {
		wmax = math.Pi / dt
	}
	lo, hi := 1.0, 1.0
	out := []float64{0}
	for _, p := range s {
		if a := cmplx.Abs(p); a > 0 {
			lo, hi = math.Min(lo, a), math.Max(hi, a)
		}
		w, d := math.Abs(imag(p)), math.Abs(real(p))
		out = append(out, w)
		for k := -8; k <= 8; k++ {
			out = append(out, w+float64(k)*d/4)
		}
		for r := 1e-13; r < 0.5; r *= 2 {
			out = append(out, w*(1+r), w*(1-r), w+r*math.Max(1, w))
		}
	}
	lo, hi = lo/1e4, math.Min(hi*1e4, wmax)
	for i := range dense {
		out = append(out, lo*math.Pow(hi/lo, float64(i)/float64(max(dense-1, 1))))
	}
	if dt > 0 {
		out = append(out, wmax)
	}
	out = slices.DeleteFunc(out, func(w float64) bool { return w < 0 || w > wmax || math.IsNaN(w) })
	slices.Sort(out)
	return slices.Compact(out)
}

// fzSweepMax returns sup f over the sorted grid, refining the best `refine`
// local maxima by golden-section search between their neighbours.
func fzSweepMax(f func(float64) float64, grid []float64, refine int) (wbest, fbest float64) {
	vals := make([]float64, len(grid))
	for i, w := range grid {
		vals[i] = f(w)
	}
	var peaks []int
	for i, v := range vals {
		if (i == 0 || v >= vals[i-1]) && (i == len(vals)-1 || v >= vals[i+1]) {
			peaks = append(peaks, i)
		}
	}
	sort.Slice(peaks, func(a, b int) bool { return vals[peaks[a]] > vals[peaks[b]] })
	fbest = math.Inf(-1)
	const r = 0.6180339887498949
	for _, i := range peaks[:min(refine, len(peaks))] {
		if vals[i] > fbest {
			wbest, fbest = grid[i], vals[i]
		}
		a, b := grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]
		c, d := b-r*(b-a), a+r*(b-a)
		fc, fd := f(c), f(d)
		for range 80 {
			if fc > fd {
				b, d, fd = d, c, fc
				c = b - r*(b-a)
				fc = f(c)
			} else {
				a, c, fc = c, d, fd
				d = a + r*(b-a)
				fd = f(d)
			}
		}
		if fc > fbest {
			wbest, fbest = c, fc
		}
		if fd > fbest {
			wbest, fbest = d, fd
		}
	}
	return wbest, fbest
}

// fzQuasiRHP counts the RHP roots of den + num·e^{−τs} by the argument
// principle (sweepOracle); ok is false when the contour grazes a root.
func fzQuasiRHP(num, den []float64, tau float64) (int, bool) {
	return sweepOracle(quasiLoop{num: num, den: den, tau: tau})
}

// fzRequireStable fails when the finite-dimensional sys is exactly unstable
// (exact Routh / Schur–Cohn on its float64 A). A Hurwitz polynomial has a
// positive Routh column, so a degenerate table is not stable either.
func fzRequireStable(t *testing.T, s *fzSpec, op string, sys *System) {
	t.Helper()
	if sys.HasInternalDelay() || sys.IsDescriptor() {
		fzSkip(t, op+": delayed or descriptor design not covered by fzRequireStable")
	}
	n, _, _ := sys.Dims()
	if n == 0 {
		return
	}
	A := mat.DenseCopyOf(sys.A).RawMatrix().Data
	cp := fzCharPolyExact(A, n)
	var k int
	var ok bool
	if sys.IsDiscrete() {
		k, ok = fzSchurCohnOutside(cp, 1)
	} else {
		k, ok = fzRouthRight(cp, 0)
	}
	if !ok || k > 0 {
		t.Fatalf("%s returned a design that is not stable (%d unstable poles, degenerate table %v)\n%s", op, k, !ok, s)
	}
}

func TestFzOracles(t *testing.T) {
	p := fzRatPoly(polyMul(polyMul([]float64{1, 1}, []float64{1, -2}), []float64{1, 2, 5}))
	for _, c := range []struct {
		sigma float64
		want  int
	}{{0, 1}, {-1.5, 4}, {-0.5, 1}, {1, 1}, {3, 0}} {
		if k, ok := fzRouthRight(p, c.sigma); !ok || k != c.want {
			t.Errorf("Routh right of %g = %d,%v, want %d", c.sigma, k, ok, c.want)
		}
	}
	if _, ok := fzRouthRight(fzRatPoly([]float64{1, 0, 1}), 0); ok {
		t.Error("Routh on s²+1 should be degenerate")
	}
	z := fzRatPoly(polyMul(polyMul([]float64{1, -0.5}, []float64{1, 2}), []float64{1, 0, 0.81}))
	for _, c := range []struct {
		rho  float64
		want int
	}{{1, 1}, {0.6, 3}, {0.25, 4}, {3, 0}} {
		if k, ok := fzSchurCohnOutside(z, c.rho); !ok || k != c.want {
			t.Errorf("Schur–Cohn outside %g = %d,%v, want %d", c.rho, k, ok, c.want)
		}
	}
	cp := fzCharPolyExact([]float64{1, 2, 3, 4}, 2)
	for i, w := range []int64{1, -5, -2} {
		if cp[i].Cmp(big.NewRat(w, 1)) != 0 {
			t.Errorf("char poly[%d] = %v, want %d", i, cp[i], w)
		}
	}
	cp = fzCharPolyRat([]*big.Rat{big.NewRat(1, 3), big.NewRat(1, 2), big.NewRat(2, 5), big.NewRat(-1, 1)}, 2)
	for i, w := range []*big.Rat{big.NewRat(1, 1), big.NewRat(2, 3), big.NewRat(-8, 15)} {
		if cp[i].Cmp(w) != 0 {
			t.Errorf("rational char poly[%d] = %v, want %v", i, cp[i], w)
		}
	}
	s := complex(1, 2)
	g, ok := fzExactResponse(1, 1, 1, []float64{-2}, []float64{3}, []float64{5}, []float64{7}, s)
	if want := 15/(s+2) + 7; !ok || cmplx.Abs(g[0]-want) > 1e-15*cmplx.Abs(want) {
		t.Errorf("exact response %v, want %v", g, want)
	}
	x := 1 + 1e-5
	if got, want := fzBigPolyEval([]float64{1, -5, 10, -10, 5, -1}, complex(x, 0)), math.Pow(x-1, 5); math.Abs(real(got)-want) > 1e-12*want {
		t.Errorf("big poly eval %v, want %v", got, want)
	}
	const zeta = 0.01
	pole := complex(-zeta, math.Sqrt(1-zeta*zeta))
	_, peak := fzSweepMax(func(w float64) float64 {
		s := complex(0, w)
		return 1 / cmplx.Abs(s*s+complex(2*zeta, 0)*s+1)
	}, fzFreqGrid([]complex128{pole, cmplx.Conj(pole)}, 0, 200), 3)
	if want := 1 / (2 * zeta * math.Sqrt(1-zeta*zeta)); math.Abs(peak-want) > 1e-10*want {
		t.Errorf("sweep peak %v, want %v", peak, want)
	}
	for _, c := range []struct {
		tau  float64
		want int
	}{{0.1, 0}, {2, 2}} {
		if k, ok := fzQuasiRHP([]float64{2}, []float64{1, 1}, c.tau); !ok || k != c.want {
			t.Errorf("quasi RHP τ=%g = %d,%v, want %d", c.tau, k, ok, c.want)
		}
	}
}

// TestFzDecodeEigErr: gonum eigenvalues of every decoded seed A lie within
// EigErr of the intended ones, and decoding is deterministic.
func TestFzDecodeEigErr(t *testing.T) {
	for _, sd := range fzIsStableSeeds() {
		k, v := sd.encode()
		t.Run("", func(t *testing.T) {
			s := fzDecode(t, k, v)
			if s2 := fzDecode(t, k, v); !slices.Equal(s.A, s2.A) {
				t.Fatal("decode not deterministic")
			}
			if s.N == 0 {
				return
			}
			var e mat.Eigen
			if !e.Factorize(mat.NewDense(s.N, s.N, s.A), mat.EigenNone) {
				t.Fatal("eigen failed")
			}
			used := make([]bool, s.N)
			for _, got := range e.Values(nil) {
				best, bi := math.Inf(1), -1
				for i, w := range s.Eig {
					if d := cmplx.Abs(got - w); !used[i] && d < best {
						best, bi = d, i
					}
				}
				used[bi] = true
				if best > s.EigErr() {
					t.Errorf("eig %v is %g from intended, EigErr %g\n%s", got, best, s.EigErr(), s)
				}
			}
		})
	}
}

// fzPencilRat returns the row-major (n+p)×(n+m) Rosenbrock pencil
// [xI−A, −B; C, D] of the float64 model at rational x.
func fzPencilRat(n, m, p int, a, b, c, d []float64, x *big.Rat) []*big.Rat {
	w := n + m
	P := make([]*big.Rat, (n+p)*w)
	for i := range n {
		for j := range n {
			v := new(big.Rat).Neg(fzRat(a[i*n+j]))
			if i == j {
				v.Add(v, x)
			}
			P[i*w+j] = v
		}
		for j := range m {
			P[i*w+n+j] = new(big.Rat).Neg(fzRat(b[i*m+j]))
		}
	}
	for i := range p {
		for j := range n {
			P[(n+i)*w+j] = fzRat(c[i*n+j])
		}
		for j := range m {
			P[(n+i)*w+n+j] = fzRat(d[i*m+j])
		}
	}
	return P
}

// fzRatElim row-reduces a copy of the r×c rational matrix and returns its
// rank and, when square, its determinant (else 0).
func fzRatElim(a []*big.Rat, r, c int) (rank int, det *big.Rat) {
	M := make([]*big.Rat, len(a))
	for i, x := range a {
		M[i] = new(big.Rat).Set(x)
	}
	det = big.NewRat(1, 1)
	var t big.Rat
	for col := 0; col < c && rank < r; col++ {
		piv := -1
		for i := rank; i < r; i++ {
			if M[i*c+col].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			continue
		}
		if piv != rank {
			for j := range c {
				M[piv*c+j], M[rank*c+j] = M[rank*c+j], M[piv*c+j]
			}
			det.Neg(det)
		}
		pv := M[rank*c+col]
		det.Mul(det, pv)
		for i := rank + 1; i < r; i++ {
			if M[i*c+col].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Quo(M[i*c+col], pv)
			for j := col; j < c; j++ {
				M[i*c+j].Sub(M[i*c+j], t.Mul(f, M[rank*c+j]))
			}
		}
		rank++
	}
	if rank < r || r != c {
		det.SetInt64(0)
	}
	return rank, det
}

// fzPencilDetRat returns det [sI−A, −B; C, D] exactly, descending of
// length n+1, for a square (m = p) model: Newton interpolation of exact
// determinants at s = 0..n. For SISO this is the numerator of
// D + C(sI−A)⁻¹B over det(sI−A), cancelled factors included.
func fzPencilDetRat(n, m int, a, b, c, d []float64) []*big.Rat {
	y := make([]*big.Rat, n+1)
	for k := range y {
		_, y[k] = fzRatElim(fzPencilRat(n, m, m, a, b, c, d, big.NewRat(int64(k), 1)), n+m, n+m)
	}
	for j := 1; j <= n; j++ {
		for k := n; k >= j; k-- {
			y[k] = new(big.Rat).Quo(new(big.Rat).Sub(y[k], y[k-1]), big.NewRat(int64(j), 1))
		}
	}
	poly := []*big.Rat{new(big.Rat).Set(y[n])}
	for k := n - 1; k >= 0; k-- {
		poly = fzRatPolyMul(poly, []*big.Rat{big.NewRat(1, 1), big.NewRat(-int64(k), 1)})
		poly[len(poly)-1].Add(poly[len(poly)-1], y[k])
	}
	return poly
}

// fzTrimRat drops exactly zero leading coefficients.
func fzTrimRat(c []*big.Rat) []*big.Rat {
	for len(c) > 0 && c[0].Sign() == 0 {
		c = c[1:]
	}
	return c
}

// fzLimitAt returns lim_{s→x0} num/den of two exact descending polynomials
// as ±Inf, 0 or the rounded finite ratio; den must not vanish identically.
func fzLimitAt(num, den []*big.Rat, x0 *big.Rat) float64 {
	low := func(c []*big.Rat) (int, *big.Rat) {
		a := fzTaylorShift(c, x0)
		for k, v := range slices.Backward(a) {
			if v.Sign() != 0 {
				return len(a) - 1 - k, v
			}
		}
		return -1, nil
	}
	on, nv := low(num)
	od, dv := low(den)
	switch {
	case on < 0 || on > od:
		return 0
	case on < od:
		return math.Inf(nv.Sign() * dv.Sign())
	}
	r, _ := new(big.Rat).Quo(nv, dv).Float64()
	return r
}

// fzComplexPencil returns the row-major float pencil [zI−A, −βB; γC, γβD].
func fzComplexPencil(s *fzSpec, z complex128, beta, gamma float64) []complex128 {
	n, m, p := s.N, s.M, s.P
	w := n + m
	P := make([]complex128, (n+p)*w)
	for i := range n {
		for j := range n {
			P[i*w+j] = complex(-s.A[i*n+j], 0)
		}
		P[i*w+i] += z
		for j := range m {
			P[i*w+n+j] = complex(-beta*s.B[i*m+j], 0)
		}
	}
	for i := range p {
		for j := range n {
			P[(n+i)*w+j] = complex(gamma*s.C[i*n+j], 0)
		}
		for j := range m {
			P[(n+i)*w+n+j] = complex(gamma*beta*s.D[i*m+j], 0)
		}
	}
	return P
}

// fzSingularValues returns the min(r,c) singular values of the r×c
// complex g, descending, via the real embedding [Re −Im; Im Re].
func fzSingularValues(g []complex128, r, c int) []float64 {
	e := mat.NewDense(2*r, 2*c, nil)
	for i := range r {
		for j := range c {
			re, im := real(g[i*c+j]), imag(g[i*c+j])
			e.Set(i, j, re)
			e.Set(i, c+j, -im)
			e.Set(r+i, j, im)
			e.Set(r+i, c+j, re)
		}
	}
	var svd mat.SVD
	if !svd.Factorize(e, mat.SVDNone) {
		return nil
	}
	v := svd.Values(nil)
	out := make([]float64, min(r, c))
	for k := range out {
		out[k] = v[2*k]
	}
	return out
}

// fzMatchEig checks got against the intended eigenvalues by Bauer–Fike:
// disks of radius r about want merge into connected components, every got
// value lies in one and each component holds as many got as want values.
// comp[i] is the component of got[i]; wantComp[j] that of want[j]. msg is
// empty on success.
func fzMatchEig(got, want []complex128, r float64) (comp, wantComp []int, msg string) {
	wantComp = make([]int, len(want))
	for i := range wantComp {
		wantComp[i] = i
	}
	var find func(int) int
	find = func(i int) int {
		if wantComp[i] != i {
			wantComp[i] = find(wantComp[i])
		}
		return wantComp[i]
	}
	for i := range want {
		for j := i + 1; j < len(want); j++ {
			if cmplx.Abs(want[i]-want[j]) <= 2*r {
				wantComp[find(i)] = find(j)
			}
		}
	}
	count := map[int]int{}
	for j := range want {
		wantComp[j] = find(j)
		count[wantComp[j]]++
	}
	comp = make([]int, len(got))
	for i, g := range got {
		comp[i] = -1
		best := math.Inf(1)
		for j, w := range want {
			if d := cmplx.Abs(g - w); d <= r && d < best {
				best, comp[i] = d, wantComp[j]
			}
		}
		if comp[i] < 0 {
			return nil, nil, fmt.Sprintf("%v is farther than %g from every intended eigenvalue", g, r)
		}
		count[comp[i]]--
	}
	for c, k := range count {
		if k != 0 {
			return nil, nil, fmt.Sprintf("cluster of %v: %d more intended than returned", want[c], k)
		}
	}
	return comp, wantComp, ""
}

func TestFzPencilOracles(t *testing.T) {
	a := []float64{0, 1, -6, -5}
	b := []float64{0, 1}
	c := []float64{3, 1}
	got := fzPencilDetRat(2, 1, a, b, c, []float64{2})
	for i, w := range []int64{2, 11, 15} {
		if got[i].Cmp(big.NewRat(w, 1)) != 0 {
			t.Errorf("pencil det[%d] = %v, want %d", i, got[i], w)
		}
	}
	if r, _ := fzRatElim(fzPencilRat(2, 1, 1, a, b, c, []float64{0}, big.NewRat(-3, 1)), 3, 3); r != 2 {
		t.Errorf("rank at the zero -3 of the D=0 pencil = %d, want 2", r)
	}
	num := fzPencilDetRat(1, 1, []float64{0}, []float64{1}, []float64{-2}, []float64{0})
	if l := fzLimitAt(num, fzCharPolyExact([]float64{0}, 1), new(big.Rat)); !math.IsInf(l, -1) {
		t.Errorf("limit of -2/s = %v, want -Inf", l)
	}
	if l := fzLimitAt(got, fzCharPolyExact(a, 2), new(big.Rat)); math.Abs(l-2.5) > 1e-15 {
		t.Errorf("limit at 0 = %v, want 2.5", l)
	}
	sv := fzSingularValues([]complex128{2i, 0, 0, 3}, 2, 2)
	if math.Abs(sv[0]-3) > 1e-14 || math.Abs(sv[1]-2) > 1e-14 {
		t.Errorf("singular values %v, want [3 2]", sv)
	}
	want := []complex128{1, 1 + 1e-3, 5}
	if _, _, msg := fzMatchEig([]complex128{1.0004, 1.0006, 5}, want, 1e-3); msg != "" {
		t.Error(msg)
	}
	if _, _, msg := fzMatchEig([]complex128{1, 5, 5}, want, 1e-4); msg == "" {
		t.Error("cluster count mismatch not detected")
	}
}
