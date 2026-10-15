package controlsys_test

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sync"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

// sweepRobustModel is a plain-data delay model used to build systems and to
// evaluate independent references.
type sweepRobustModel struct {
	n, m, p, N            int
	A, B, C, D            []float64
	B2, C2, D12, D21, D22 []float64
	tau                   []float64
	in, out               []float64
	io                    []float64
	dt                    float64
}

func (md sweepRobustModel) String() string {
	return fmt.Sprintf("n=%d m=%d p=%d N=%d dt=%g\nA=%v\nB=%v\nC=%v\nD=%v\nB2=%v\nC2=%v\nD12=%v\nD21=%v\nD22=%v\ntau=%v in=%v out=%v io=%v",
		md.n, md.m, md.p, md.N, md.dt, md.A, md.B, md.C, md.D, md.B2, md.C2, md.D12, md.D21, md.D22, md.tau, md.in, md.out, md.io)
}

func sweepRobustDense(r, c int, d []float64) *mat.Dense {
	if r == 0 || c == 0 {
		return nil
	}
	return mat.NewDense(r, c, append([]float64(nil), d...))
}

func sweepRobustBuild(md sweepRobustModel) (*controlsys.System, error) {
	var sys *controlsys.System
	var err error
	if md.n == 0 {
		sys, err = controlsys.NewGain(mat.NewDense(md.p, md.m, append([]float64(nil), md.D...)), md.dt)
	} else {
		sys, err = controlsys.NewFromSlices(md.n, md.m, md.p, md.A, md.B, md.C, md.D, md.dt)
	}
	if err != nil {
		return nil, err
	}
	if md.in != nil {
		if err := sys.SetInputDelay(md.in); err != nil {
			return nil, err
		}
	}
	if md.out != nil {
		if err := sys.SetOutputDelay(md.out); err != nil {
			return nil, err
		}
	}
	if md.io != nil {
		if err := sys.SetDelay(mat.NewDense(md.p, md.m, append([]float64(nil), md.io...))); err != nil {
			return nil, err
		}
	}
	if md.N > 0 {
		err := sys.SetInternalDelay(md.tau,
			sweepRobustDense(md.n, md.N, md.B2), sweepRobustDense(md.N, md.n, md.C2),
			sweepRobustDense(md.p, md.N, md.D12), sweepRobustDense(md.N, md.m, md.D21),
			sweepRobustDense(md.N, md.N, md.D22))
		if err != nil {
			return nil, err
		}
	}
	return sys, nil
}

func sweepRobustMustBuild(t *testing.T, md sweepRobustModel) *controlsys.System {
	t.Helper()
	sys, err := sweepRobustBuild(md)
	if err != nil {
		t.Fatalf("build: %v\n%v", err, md)
	}
	return sys
}

func sweepRobustRandSlice(rng *rand.Rand, k int, scale float64) []float64 {
	s := make([]float64, k)
	for i := range s {
		s[i] = scale * rng.NormFloat64()
	}
	return s
}

// sweepRobustRandA returns a non-symmetric A that is stable or unstable by a
// Gershgorin bound (continuous) or by its infinity norm (discrete).
func sweepRobustRandA(rng *rand.Rand, n int, stable bool, dt float64) []float64 {
	a := sweepRobustRandSlice(rng, n*n, 1)
	if dt > 0 {
		norm := 0.0
		for i := range n {
			row := 0.0
			for j := range n {
				row += math.Abs(a[i*n+j])
			}
			norm = math.Max(norm, row)
		}
		target := 0.85
		if !stable {
			target = 1.08
		}
		for i := range a {
			a[i] *= target / norm
		}
		if !stable {
			for i := range n {
				a[i*n+i] += 0.05 * math.Copysign(1, a[i*n+i])
			}
		}
		return a
	}
	for i := range n {
		row := 0.0
		for j := range n {
			if j != i {
				row += math.Abs(a[i*n+j])
			}
		}
		if stable {
			a[i*n+i] = -(row + 0.5 + rng.Float64())
		} else if i == 0 {
			a[i*n+i] = row + 0.3
		}
	}
	return a
}

func sweepRobustRandDelay(rng *rand.Rand, dt float64, allowZero bool) float64 {
	if dt > 0 {
		lo := 1
		if allowZero {
			lo = 0
		}
		return float64(lo + rng.Intn(4))
	}
	if allowZero && rng.Intn(4) == 0 {
		return 0
	}
	return []float64{0.1, 0.25, 0.5, 0.5, 1.3}[rng.Intn(5)]
}

func sweepRobustRandModel(rng *rand.Rand, dt float64, stable bool, withIO, withInternal bool) sweepRobustModel {
	md := sweepRobustModel{n: 1 + rng.Intn(5), m: 1 + rng.Intn(3), p: 1 + rng.Intn(3), dt: dt}
	n, m, p := md.n, md.m, md.p
	md.A = sweepRobustRandA(rng, n, stable, dt)
	md.B = sweepRobustRandSlice(rng, n*m, 1)
	md.C = sweepRobustRandSlice(rng, p*n, 1)
	md.D = sweepRobustRandSlice(rng, p*m, 0.5)
	if withIO {
		md.in = make([]float64, m)
		for j := range m {
			md.in[j] = sweepRobustRandDelay(rng, dt, true)
		}
		md.out = make([]float64, p)
		for i := range p {
			md.out[i] = sweepRobustRandDelay(rng, dt, true)
		}
	}
	if withInternal {
		md.N = 1 + rng.Intn(3)
		N := md.N
		md.tau = make([]float64, N)
		for k := range N {
			md.tau[k] = sweepRobustRandDelay(rng, dt, false)
		}
		if N > 1 && rng.Intn(2) == 0 {
			md.tau[1] = md.tau[0]
		}
		md.B2 = sweepRobustRandSlice(rng, n*N, 0.3)
		md.C2 = sweepRobustRandSlice(rng, N*n, 0.3)
		md.D12 = sweepRobustRandSlice(rng, p*N, 0.5)
		md.D21 = sweepRobustRandSlice(rng, N*m, 0.5)
		md.D22 = sweepRobustRandSlice(rng, N*N, 0.15)
	}
	return md
}

type sweepRobustCM struct {
	r, c int
	d    []complex128
}

func sweepRobustCMNew(r, c int) sweepRobustCM {
	return sweepRobustCM{r: r, c: c, d: make([]complex128, r*c)}
}

func sweepRobustCMReal(r, c int, d []float64) sweepRobustCM {
	x := sweepRobustCMNew(r, c)
	for i := range r * c {
		x.d[i] = complex(d[i], 0)
	}
	return x
}

func (x sweepRobustCM) at(i, j int) complex128 { return x.d[i*x.c+j] }

func sweepRobustCMMul(a, b sweepRobustCM) sweepRobustCM {
	out := sweepRobustCMNew(a.r, b.c)
	for i := range a.r {
		for k := range a.c {
			aik := a.d[i*a.c+k]
			for j := range b.c {
				out.d[i*b.c+j] += aik * b.d[k*b.c+j]
			}
		}
	}
	return out
}

func sweepRobustCMAdd(a, b sweepRobustCM) sweepRobustCM {
	out := sweepRobustCMNew(a.r, a.c)
	for i := range a.d {
		out.d[i] = a.d[i] + b.d[i]
	}
	return out
}

// sweepRobustCMSolve solves A X = B by Gaussian elimination with partial
// pivoting and returns the smallest pivot magnitude relative to max |A|.
func sweepRobustCMSolve(a, b sweepRobustCM) (sweepRobustCM, float64) {
	n := a.r
	A := append([]complex128(nil), a.d...)
	X := append([]complex128(nil), b.d...)
	nc := b.c
	scale := 0.0
	for _, v := range A {
		scale = math.Max(scale, cmplx.Abs(v))
	}
	if scale == 0 {
		scale = 1
	}
	minPiv := math.Inf(1)
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(A[i*n+k]) > cmplx.Abs(A[piv*n+k]) {
				piv = i
			}
		}
		if piv != k {
			for j := range n {
				A[k*n+j], A[piv*n+j] = A[piv*n+j], A[k*n+j]
			}
			for j := range nc {
				X[k*nc+j], X[piv*nc+j] = X[piv*nc+j], X[k*nc+j]
			}
		}
		minPiv = math.Min(minPiv, cmplx.Abs(A[k*n+k])/scale)
		if A[k*n+k] == 0 {
			return sweepRobustCM{}, 0
		}
		for i := k + 1; i < n; i++ {
			f := A[i*n+k] / A[k*n+k]
			for j := k; j < n; j++ {
				A[i*n+j] -= f * A[k*n+j]
			}
			for j := range nc {
				X[i*nc+j] -= f * X[k*nc+j]
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		for j := range nc {
			s := X[k*nc+j]
			for l := k + 1; l < n; l++ {
				s -= A[k*n+l] * X[l*nc+j]
			}
			X[k*nc+j] = s / A[k*n+k]
		}
	}
	return sweepRobustCM{r: n, c: nc, d: X}, minPiv
}

// sweepRobustRef evaluates the hand-written LFT reference
// P11 + P12 (I - Δ P22)^{-1} Δ P21 with I/O delay factors. ok is false when a
// solve is too close to singular to serve as a reference.
func sweepRobustRef(md sweepRobustModel, w float64) (sweepRobustCM, bool) {
	n, m, p, N := md.n, md.m, md.p, md.N
	s := complex(0, w)
	lag := func(d float64) complex128 {
		if d == 0 {
			return 1
		}
		if md.dt > 0 {
			return cmplx.Exp(complex(0, -w*md.dt*d))
		}
		return cmplx.Exp(complex(0, -w*d))
	}
	if md.dt > 0 {
		s = cmplx.Exp(complex(0, w*md.dt))
	}
	mN, pN := m+N, p+N
	Bf := make([]float64, n*mN)
	Cf := make([]float64, pN*n)
	Df := make([]float64, pN*mN)
	for i := range n {
		for j := range m {
			Bf[i*mN+j] = md.B[i*m+j]
		}
		for j := range N {
			Bf[i*mN+m+j] = md.B2[i*N+j]
		}
	}
	for j := range n {
		for i := range p {
			Cf[i*n+j] = md.C[i*n+j]
		}
		for i := range N {
			Cf[(p+i)*n+j] = md.C2[i*n+j]
		}
	}
	for i := range p {
		for j := range m {
			Df[i*mN+j] = md.D[i*m+j]
		}
		for j := range N {
			Df[i*mN+m+j] = md.D12[i*N+j]
		}
	}
	for i := range N {
		for j := range m {
			Df[(p+i)*mN+j] = md.D21[i*m+j]
		}
		for j := range N {
			Df[(p+i)*mN+m+j] = md.D22[i*N+j]
		}
	}
	P := sweepRobustCMReal(pN, mN, Df)
	ok := true
	if n > 0 {
		sIA := sweepRobustCMNew(n, n)
		for i := range n {
			for j := range n {
				sIA.d[i*n+j] = complex(-md.A[i*n+j], 0)
			}
			sIA.d[i*n+i] += s
		}
		R, piv := sweepRobustCMSolve(sIA, sweepRobustCMReal(n, mN, Bf))
		if piv < 1e-9 {
			return sweepRobustCM{}, false
		}
		P = sweepRobustCMAdd(P, sweepRobustCMMul(sweepRobustCMReal(pN, n, Cf), R))
	}
	G := sweepRobustCMNew(p, m)
	for i := range p {
		for j := range m {
			G.d[i*m+j] = P.at(i, j)
		}
	}
	if N > 0 {
		P12, P21, P22 := sweepRobustCMNew(p, N), sweepRobustCMNew(N, m), sweepRobustCMNew(N, N)
		for i := range p {
			for j := range N {
				P12.d[i*N+j] = P.at(i, m+j)
			}
		}
		for i := range N {
			lg := lag(md.tau[i])
			for j := range m {
				P21.d[i*m+j] = lg * P.at(p+i, j)
			}
			for j := range N {
				P22.d[i*N+j] = -lg * P.at(p+i, m+j)
			}
			P22.d[i*N+i] += 1
		}
		X, piv := sweepRobustCMSolve(P22, P21)
		if piv < 1e-9 {
			return sweepRobustCM{}, false
		}
		G = sweepRobustCMAdd(G, sweepRobustCMMul(P12, X))
	}
	for i := range p {
		for j := range m {
			d := 0.0
			if md.in != nil {
				d += md.in[j]
			}
			if md.out != nil {
				d += md.out[i]
			}
			if md.io != nil {
				d += md.io[i*m+j]
			}
			G.d[i*m+j] *= lag(d)
		}
	}
	return G, ok
}

var sweepRobustOmegas = []float64{0.013, 0.31, 1.7, 4.4, 11}

func sweepRobustCheckFreq(t *testing.T, label string, sys *controlsys.System, md sweepRobustModel, omegas []float64, tol float64) {
	t.Helper()
	fr, err := sys.FreqResponse(omegas)
	if err != nil {
		t.Fatalf("%s FreqResponse: %v\n%v", label, err, md)
	}
	for k, w := range omegas {
		ref, ok := sweepRobustRef(md, w)
		if !ok {
			continue
		}
		for i := range md.p {
			for j := range md.m {
				got, want := fr.At(k, i, j), ref.at(i, j)
				if cmplx.Abs(got-want) > tol*(1+cmplx.Abs(want)) {
					t.Fatalf("%s w=%g (%d,%d): got %v want %v\n%v", label, w, i, j, got, want, md)
				}
			}
		}
	}
}

func sweepRobustCheckFreqPair(t *testing.T, label string, a, b *controlsys.System, omegas []float64, tol float64, ctx fmt.Stringer) {
	t.Helper()
	fa, err := a.FreqResponse(omegas)
	if err != nil {
		t.Fatalf("%s FreqResponse a: %v\n%v", label, err, ctx)
	}
	fb, err := b.FreqResponse(omegas)
	if err != nil {
		t.Fatalf("%s FreqResponse b: %v\n%v", label, err, ctx)
	}
	if fa.P != fb.P || fa.M != fb.M {
		t.Fatalf("%s dims %dx%d vs %dx%d", label, fa.P, fa.M, fb.P, fb.M)
	}
	for k := range omegas {
		for i := range fa.P {
			for j := range fa.M {
				x, y := fa.At(k, i, j), fb.At(k, i, j)
				if cmplx.Abs(x-y) > tol*(1+cmplx.Abs(x)) {
					t.Fatalf("%s w=%g (%d,%d): %v vs %v\n%v", label, omegas[k], i, j, x, y, ctx)
				}
			}
		}
	}
}

// sweepRobustSimRef is a hand-written discrete recursion: delayed inputs,
// internal delay lines w_l(k) = z_l(k - tau_l), then delayed outputs.
// u is m×steps (row-major per channel); the result is p×steps.
func sweepRobustSimRef(md sweepRobustModel, u [][]float64, steps int) [][]float64 {
	n, m, p, N := md.n, md.m, md.p, md.N
	x := make([]float64, n)
	zHist := make([][]float64, N)
	for l := range N {
		zHist[l] = make([]float64, steps)
	}
	yRaw := make([][]float64, p)
	for i := range p {
		yRaw[i] = make([]float64, steps)
	}
	ud := make([]float64, m)
	w := make([]float64, N)
	for k := range steps {
		for j := range m {
			d := 0
			if md.in != nil {
				d = int(md.in[j])
			}
			ud[j] = 0
			if k-d >= 0 {
				ud[j] = u[j][k-d]
			}
		}
		for l := range N {
			d := int(md.tau[l])
			w[l] = 0
			if k-d >= 0 {
				w[l] = zHist[l][k-d]
			}
		}
		for l := range N {
			z := 0.0
			for i := range n {
				z += md.C2[l*n+i] * x[i]
			}
			for j := range m {
				z += md.D21[l*m+j] * ud[j]
			}
			for q := range N {
				z += md.D22[l*N+q] * w[q]
			}
			zHist[l][k] = z
		}
		for i := range p {
			y := 0.0
			for q := range n {
				y += md.C[i*n+q] * x[q]
			}
			for j := range m {
				y += md.D[i*m+j] * ud[j]
			}
			for q := range N {
				y += md.D12[i*N+q] * w[q]
			}
			yRaw[i][k] = y
		}
		xn := make([]float64, n)
		for i := range n {
			v := 0.0
			for q := range n {
				v += md.A[i*n+q] * x[q]
			}
			for j := range m {
				v += md.B[i*m+j] * ud[j]
			}
			for q := range N {
				v += md.B2[i*N+q] * w[q]
			}
			xn[i] = v
		}
		x = xn
	}
	y := make([][]float64, p)
	for i := range p {
		y[i] = make([]float64, steps)
		d := 0
		if md.out != nil {
			d = int(md.out[i])
		}
		for k := d; k < steps; k++ {
			y[i][k] = yRaw[i][k-d]
		}
	}
	return y
}

func sweepRobustRandInput(rng *rand.Rand, m, steps int) ([][]float64, *mat.Dense) {
	u := make([][]float64, m)
	U := mat.NewDense(m, steps, nil)
	for j := range m {
		u[j] = make([]float64, steps)
		for k := range steps {
			u[j][k] = rng.NormFloat64()
			U.Set(j, k, u[j][k])
		}
	}
	return u, U
}

func sweepRobustCheckSim(t *testing.T, label string, Y *mat.Dense, ref [][]float64, tol float64, ctx fmt.Stringer) {
	t.Helper()
	for i := range ref {
		for k := range ref[i] {
			got, want := Y.At(i, k), ref[i][k]
			if math.IsNaN(got) || math.Abs(got-want) > tol*(1+math.Abs(want)) {
				t.Fatalf("%s y[%d][%d]: got %.17g want %.17g\n%v", label, i, k, got, want, ctx)
			}
		}
	}
}

// sweepRobustGuard runs fn in a goroutine, converting panics into a returned
// value and failing when fn exceeds limit.
func sweepRobustGuard(t *testing.T, label string, limit time.Duration, fn func()) (panicked any) {
	t.Helper()
	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		fn()
	}()
	select {
	case r := <-done:
		return r
	case <-time.After(limit):
		t.Fatalf("%s: exceeded %v", label, limit)
		return nil
	}
}

func sweepRobustScalarSys(t *testing.T, a, b, c, d, dt float64) *controlsys.System {
	t.Helper()
	sys, err := controlsys.NewFromSlices(1, 1, 1, []float64{a}, []float64{b}, []float64{c}, []float64{d}, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sweepRobustGain(t *testing.T, k, dt float64) *controlsys.System {
	t.Helper()
	g, err := controlsys.NewGain(mat.NewDense(1, 1, []float64{k}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestSweepRobustStaticDiscreteLFTRecursion (E-A.a): static discrete LFT with
// D21=1, D22=0.5 follows w_k = z_{k-τ}, z_k = u_k + 0.5 w_k in Simulate and Lsim.
func TestSweepRobustStaticDiscreteLFTRecursion(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	for _, tau := range []int{1, 2, 5} {
		for _, v := range []struct{ d11, d12 float64 }{{0, 1}, {1, 0.5}} {
			md := sweepRobustModel{m: 1, p: 1, N: 1, dt: 0.1, D: []float64{v.d11},
				D12: []float64{v.d12}, D21: []float64{1}, D22: []float64{0.5}, tau: []float64{float64(tau)}}
			sys := sweepRobustMustBuild(t, md)
			const steps = 40
			u, U := sweepRobustRandInput(rng, 1, steps)
			z := make([]float64, steps)
			want := make([]float64, steps)
			for k := range steps {
				w := 0.0
				if k >= tau {
					w = z[k-tau]
				}
				z[k] = u[0][k] + 0.5*w
				want[k] = v.d11*u[0][k] + v.d12*w
			}
			resp, err := sys.Simulate(U, nil, nil)
			if err != nil {
				t.Fatalf("tau=%d Simulate: %v", tau, err)
			}
			sweepRobustCheckSim(t, fmt.Sprintf("tau=%d d11=%g Simulate", tau, v.d11), resp.Y, [][]float64{want}, 1e-13, md)
			tm := make([]float64, steps)
			uT := mat.NewDense(steps, 1, nil)
			for k := range steps {
				tm[k] = 0.1 * float64(k)
				uT.Set(k, 0, u[0][k])
			}
			lr, err := controlsys.Lsim(sys, uT, tm, nil)
			if err != nil {
				t.Fatalf("tau=%d Lsim: %v", tau, err)
			}
			sweepRobustCheckSim(t, fmt.Sprintf("tau=%d d11=%g Lsim", tau, v.d11), lr.Y, [][]float64{want}, 1e-13, md)
		}
	}
}

// TestSweepRobustZeroDelayCurrentSample (E-A.b): τ=0 internal delays are
// rejected with ErrZeroInternalDelay; ZeroDelayApprox closes the loop at the
// current sample (static D22=0.5 gives gain 2 at k=0, no one-sample shift).
func TestSweepRobustZeroDelayCurrentSample(t *testing.T) {
	g := sweepRobustGain(t, 0, 1)
	one := mat.NewDense(1, 1, []float64{1})
	if err := g.SetInternalDelay([]float64{0}, nil, nil, one, one, mat.NewDense(1, 1, []float64{0.5})); !errors.Is(err, controlsys.ErrZeroInternalDelay) {
		t.Fatalf("SetInternalDelay tau=0: err=%v, want ErrZeroInternalDelay", err)
	}
	if g.LFT != nil {
		t.Fatalf("failed SetInternalDelay modified LFT")
	}
	H, _ := controlsys.NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 0.5}), 1)
	if _, err := controlsys.SetDelayModel(H, []float64{0}); !errors.Is(err, controlsys.ErrZeroInternalDelay) {
		t.Fatalf("SetDelayModel tau=0: err=%v, want ErrZeroInternalDelay", err)
	}

	md := sweepRobustModel{m: 1, p: 1, N: 1, dt: 1, D: []float64{0}, D12: []float64{1}, D21: []float64{1}, D22: []float64{0.5}, tau: []float64{1}}
	z, err := sweepRobustMustBuild(t, md).ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	if z.HasInternalDelay() {
		t.Fatalf("ZeroDelayApprox kept internal delays")
	}
	resp, err := z.Simulate(mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1}), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range 5 {
		if got := resp.Y.At(0, k); math.Abs(got-2) > 1e-14 {
			t.Fatalf("static ZeroDelayApprox y[%d]=%v, want 2", k, got)
		}
	}

	dyn := sweepRobustModel{n: 1, m: 1, p: 1, N: 1, dt: 1, A: []float64{0.5}, B: []float64{1}, C: []float64{1}, D: []float64{0},
		B2: []float64{0.3}, C2: []float64{0.4}, D12: []float64{0.2}, D21: []float64{1}, D22: []float64{0.5}, tau: []float64{3}}
	zd, err := sweepRobustMustBuild(t, dyn).ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	const steps = 12
	r, err := zd.Simulate(mat.NewDense(1, steps, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	x := 0.0
	for k := range steps {
		w := (0.4*x + 1) / (1 - 0.5)
		y := x + 0.2*w
		if got := r.Y.At(0, k); math.Abs(got-y) > 1e-13 {
			t.Fatalf("dynamic ZeroDelayApprox y[%d]=%.17g want %.17g", k, got, y)
		}
		x = 0.5*x + 1 + 0.3*w
	}
}

// TestSweepRobustFeedbackAlgebraicLoop (E-A): zero-delay static feedback loops
// resolve at the current sample; singular loops return ErrAlgebraicLoop; a
// delayed loop follows its recursion.
func TestSweepRobustFeedbackAlgebraicLoop(t *testing.T) {
	for _, c := range []struct {
		g, h, want float64
		nilH       bool
	}{{1, 0.5, 2, false}, {0.5, 1, 1, true}, {2, -0.25, 4.0 / 3, false}} {
		for _, dt := range []float64{0, 0.1} {
			var h *controlsys.System
			if !c.nilH {
				h = sweepRobustGain(t, c.h, dt)
			}
			cl, err := controlsys.Feedback(sweepRobustGain(t, c.g, dt), h, 1)
			if err != nil {
				t.Fatalf("Feedback(%g,%g,+1) dt=%g: %v", c.g, c.h, dt, err)
			}
			st, err := controlsys.Step(cl, 1)
			if err != nil {
				t.Fatalf("Step: %v", err)
			}
			for k := range st.T {
				if got := st.Y.At(0, k); math.Abs(got-c.want) > 1e-13 {
					t.Fatalf("Feedback(%g,%g,+1) dt=%g Step t=%g: %v want %v", c.g, c.h, dt, st.T[k], got, c.want)
				}
			}
			if dt > 0 {
				r, err := cl.Simulate(mat.NewDense(1, 3, []float64{1, 1, 1}), nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				if got := r.Y.At(0, 0); math.Abs(got-c.want) > 1e-13 {
					t.Fatalf("Simulate y[0]=%v want %v", got, c.want)
				}
			}
		}
	}

	for _, dt := range []float64{0, 0.1} {
		cl, err := controlsys.Feedback(sweepRobustGain(t, 1, dt), nil, 1)
		if !sweepRobustIsLoopError(err) || cl != nil {
			t.Fatalf("Feedback(1,1,+1) dt=%g: sys=%v err=%v, want ErrAlgebraicLoop (ergo %s)", dt, cl, err, sweepRobustErgoFeedbackEr)
		}
		cl, err = controlsys.Feedback(sweepRobustScalarSys(t, -1, 1, 1, 2, dt), sweepRobustGain(t, 0.5, dt), 1)
		if !sweepRobustIsLoopError(err) || cl != nil {
			t.Fatalf("Feedback(D=2,0.5,+1) dt=%g: sys=%v err=%v, want ErrAlgebraicLoop", dt, cl, err)
		}
		I2 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
		g2, _ := controlsys.NewGain(I2, dt)
		h2, _ := controlsys.NewGain(I2, dt)
		cl, err = controlsys.Feedback(g2, h2, 1)
		if !sweepRobustIsLoopError(err) || cl != nil {
			t.Fatalf("Feedback(I2,I2,+1) dt=%g: err=%v, want ErrAlgebraicLoop", dt, err)
		}
	}

	pl := sweepRobustGain(t, 1, 1)
	if err := pl.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	cl, err := controlsys.Feedback(pl, sweepRobustGain(t, 0.5, 1), 1)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	u, U := sweepRobustRandInput(rng, 1, 30)
	r, err := cl.Simulate(U, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float64, 30)
	for k := 1; k < 30; k++ {
		want[k] = u[0][k-1] + 0.5*want[k-1]
	}
	sweepRobustCheckSim(t, "delayed static loop", r.Y, [][]float64{want}, 1e-13, nil)
}

// TestSweepRobustContinuousFeedthroughLoopStep (E-A): continuous Step of a
// unit negative feedback loop around a plant with D=0.5, without and with an
// input delay inside the loop, against closed forms.
func TestSweepRobustContinuousFeedthroughLoopStep(t *testing.T) {
	cl, err := controlsys.Feedback(sweepRobustScalarSys(t, -1, 1, 1, 0.5, 0), nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	st, err := controlsys.Step(cl, 5)
	if err != nil {
		t.Fatal(err)
	}
	for k, tk := range st.T {
		want := 1.0/3 + (2.0/3)*0.4*(1-math.Exp(-5*tk/3))
		if got := st.Y.At(0, k); math.Abs(got-want) > 1e-10 {
			t.Fatalf("no-delay loop t=%g: %v want %v", tk, got, want)
		}
	}

	const L = 0.5
	pl := sweepRobustScalarSys(t, -1, 1, 1, 0.5, 0)
	if err := pl.SetInputDelay([]float64{L}); err != nil {
		t.Fatal(err)
	}
	cl, err = controlsys.Feedback(pl, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	st, err = controlsys.Step(cl, 2*L-1e-9)
	if err != nil {
		t.Fatal(err)
	}
	for k, tk := range st.T {
		var want float64
		switch {
		case tk < L-1e-9:
			want = 0
		case tk > L+1e-9:
			want = 1 - math.Exp(-(tk - L)) + 0.5
		default:
			continue
		}
		if got := st.Y.At(0, k); math.Abs(got-want) > 1e-8 {
			t.Fatalf("delayed loop t=%.17g: %v want %v", tk, got, want)
		}
	}
}

func sweepRobustCases(seed int64, count int, fn func(i int, rng *rand.Rand, dt float64, stable bool)) {
	rng := rand.New(rand.NewSource(seed))
	for i := range count {
		dt := 0.0
		if i%2 == 1 {
			dt = 0.1
		}
		fn(i, rng, dt, i%4 < 2)
	}
}

// TestSweepRobustRandomFreqResponseReference (E-B): FreqResponse of random
// models with I/O, I/O-matrix and internal delays (repeated τ included)
// matches the hand-written LFT reference.
func TestSweepRobustRandomFreqResponseReference(t *testing.T) {
	sweepRobustCases(201, 24, func(i int, rng *rand.Rand, dt float64, stable bool) {
		md := sweepRobustRandModel(rng, dt, stable, i%3 != 2, true)
		if i%3 == 1 {
			md.io = make([]float64, md.p*md.m)
			for k := range md.io {
				md.io[k] = sweepRobustRandDelay(rng, dt, true)
			}
		}
		sys := sweepRobustMustBuild(t, md)
		sweepRobustCheckFreq(t, fmt.Sprintf("case %d", i), sys, md, sweepRobustOmegas, 1e-9)
	})
}

// TestSweepRobustZeroDelayIdentity (E-B): explicit all-zero delays equal the
// delay-free model, and ZeroDelayApprox equals the reference with Δ = I.
func TestSweepRobustZeroDelayIdentity(t *testing.T) {
	sweepRobustCases(202, 20, func(i int, rng *rand.Rand, dt float64, stable bool) {
		md := sweepRobustRandModel(rng, dt, stable, false, false)
		plain := sweepRobustMustBuild(t, md)
		zmd := md
		zmd.in, zmd.out, zmd.io = make([]float64, md.m), make([]float64, md.p), make([]float64, md.p*md.m)
		zero := sweepRobustMustBuild(t, zmd)
		sweepRobustCheckFreqPair(t, fmt.Sprintf("case %d zeros", i), plain, zero, sweepRobustOmegas, 1e-13, md)
		if dt > 0 {
			_, U := sweepRobustRandInput(rng, md.m, 20)
			a, err := plain.Simulate(U, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			b, err := zero.Simulate(U, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !mat.EqualApprox(a.Y, b.Y, 1e-12) {
				t.Fatalf("case %d zero-delay Simulate differs\n%v", i, md)
			}
		}

		imd := sweepRobustRandModel(rng, dt, stable, false, true)
		zda, err := sweepRobustMustBuild(t, imd).ZeroDelayApprox()
		if err != nil {
			t.Fatalf("case %d ZeroDelayApprox: %v\n%v", i, err, imd)
		}
		ref := imd
		ref.tau = make([]float64, imd.N)
		sweepRobustCheckFreq(t, fmt.Sprintf("case %d ZeroDelayApprox", i), zda, ref, sweepRobustOmegas, 1e-9)
	})
}

func sweepRobustPermuteInputs(md sweepRobustModel, perm []int) sweepRobustModel {
	out := md
	n, m, p, N := md.n, md.m, md.p, md.N
	out.B, out.D, out.D21 = make([]float64, n*m), make([]float64, p*m), make([]float64, N*m)
	if md.in != nil {
		out.in = make([]float64, m)
	}
	if md.io != nil {
		out.io = make([]float64, p*m)
	}
	for j, src := range perm {
		for i := range n {
			out.B[i*m+j] = md.B[i*m+src]
		}
		for i := range p {
			out.D[i*m+j] = md.D[i*m+src]
			if md.io != nil {
				out.io[i*m+j] = md.io[i*m+src]
			}
		}
		for i := range N {
			out.D21[i*m+j] = md.D21[i*m+src]
		}
		if md.in != nil {
			out.in[j] = md.in[src]
		}
	}
	return out
}

func sweepRobustPermuteOutputs(md sweepRobustModel, perm []int) sweepRobustModel {
	out := md
	n, m, p, N := md.n, md.m, md.p, md.N
	out.C, out.D, out.D12 = make([]float64, p*n), make([]float64, p*m), make([]float64, p*N)
	if md.out != nil {
		out.out = make([]float64, p)
	}
	if md.io != nil {
		out.io = make([]float64, p*m)
	}
	for i, src := range perm {
		copy(out.C[i*n:(i+1)*n], md.C[src*n:(src+1)*n])
		copy(out.D[i*m:(i+1)*m], md.D[src*m:(src+1)*m])
		copy(out.D12[i*N:(i+1)*N], md.D12[src*N:(src+1)*N])
		if md.io != nil {
			copy(out.io[i*m:(i+1)*m], md.io[src*m:(src+1)*m])
		}
		if md.out != nil {
			out.out[i] = md.out[src]
		}
	}
	return out
}

// TestSweepRobustPermutation (E-B): permuting inputs (outputs) permutes the
// columns (rows) of FreqResponse.
func TestSweepRobustPermutation(t *testing.T) {
	sweepRobustCases(203, 20, func(i int, rng *rand.Rand, dt float64, stable bool) {
		md := sweepRobustRandModel(rng, dt, stable, true, i%2 == 0)
		sys := sweepRobustMustBuild(t, md)
		ip, op := rng.Perm(md.m), rng.Perm(md.p)
		ps := sweepRobustMustBuild(t, sweepRobustPermuteOutputs(sweepRobustPermuteInputs(md, ip), op))
		fa, err := sys.FreqResponse(sweepRobustOmegas)
		if err != nil {
			t.Fatal(err)
		}
		fb, err := ps.FreqResponse(sweepRobustOmegas)
		if err != nil {
			t.Fatal(err)
		}
		for k := range sweepRobustOmegas {
			for r := range md.p {
				for c := range md.m {
					x, y := fb.At(k, r, c), fa.At(k, op[r], ip[c])
					if cmplx.Abs(x-y) > 1e-11*(1+cmplx.Abs(y)) {
						t.Fatalf("case %d perm in=%v out=%v (%d,%d): %v vs %v\n%v", i, ip, op, r, c, x, y, md)
					}
				}
			}
		}
	})
}

// TestSweepRobustSimilarity (E-B): x' = T x leaves the delayed frequency
// response unchanged (B, B2 → T B, T B2; C, C2 → C T⁻¹, C2 T⁻¹).
func TestSweepRobustSimilarity(t *testing.T) {
	sweepRobustCases(204, 20, func(i int, rng *rand.Rand, dt float64, stable bool) {
		md := sweepRobustRandModel(rng, dt, stable, true, true)
		n := md.n
		T := mat.NewDense(n, n, sweepRobustRandSlice(rng, n*n, 0.3))
		for k := range n {
			T.Set(k, k, T.At(k, k)+1.5)
		}
		var Ti mat.Dense
		if err := Ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		tm := md
		mulInto := func(a, b mat.Matrix) []float64 {
			var r mat.Dense
			r.Mul(a, b)
			return append([]float64(nil), r.RawMatrix().Data...)
		}
		var tmp mat.Dense
		tmp.Mul(T, mat.NewDense(n, n, md.A))
		tm.A = mulInto(&tmp, &Ti)
		tm.B = mulInto(T, mat.NewDense(n, md.m, md.B))
		tm.C = mulInto(mat.NewDense(md.p, n, md.C), &Ti)
		tm.B2 = mulInto(T, mat.NewDense(n, md.N, md.B2))
		tm.C2 = mulInto(mat.NewDense(md.N, n, md.C2), &Ti)
		sweepRobustCheckFreqPair(t, fmt.Sprintf("case %d similarity", i), sweepRobustMustBuild(t, md), sweepRobustMustBuild(t, tm), sweepRobustOmegas, 1e-9, md)
	})
}

func sweepRobustRandChain(rng *rand.Rand, dt float64, stable bool, m, p int) sweepRobustModel {
	md := sweepRobustRandModel(rng, dt, stable, true, rng.Intn(2) == 0)
	md.m, md.p = m, p
	md.B = sweepRobustRandSlice(rng, md.n*m, 1)
	md.C = sweepRobustRandSlice(rng, p*md.n, 1)
	md.D = sweepRobustRandSlice(rng, p*m, 0.5)
	md.in, md.out = make([]float64, m), make([]float64, p)
	for j := range m {
		md.in[j] = sweepRobustRandDelay(rng, dt, true)
	}
	for i := range p {
		md.out[i] = sweepRobustRandDelay(rng, dt, true)
	}
	if md.N > 0 {
		md.D12 = sweepRobustRandSlice(rng, p*md.N, 0.5)
		md.D21 = sweepRobustRandSlice(rng, md.N*m, 0.5)
	}
	return md
}

// TestSweepRobustSeriesParallelAlgebra (E-B): Series is associative and
// Parallel commutative for delayed models, and both match products/sums of
// the per-model references.
func TestSweepRobustSeriesParallelAlgebra(t *testing.T) {
	sweepRobustCases(205, 20, func(i int, rng *rand.Rand, dt float64, stable bool) {
		d0, d1, d2, d3 := 1+rng.Intn(3), 1+rng.Intn(3), 1+rng.Intn(3), 1+rng.Intn(3)
		ma, mb, mc := sweepRobustRandChain(rng, dt, stable, d0, d1), sweepRobustRandChain(rng, dt, stable, d1, d2), sweepRobustRandChain(rng, dt, stable, d2, d3)
		a, b, c := sweepRobustMustBuild(t, ma), sweepRobustMustBuild(t, mb), sweepRobustMustBuild(t, mc)
		ctx := sweepRobustStringer(func() string { return fmt.Sprintf("a:\n%v\nb:\n%v\nc:\n%v", ma, mb, mc) })
		ab, err := controlsys.Series(a, b)
		if err != nil {
			t.Fatalf("case %d Series(a,b): %v\n%v", i, err, ctx)
		}
		left, err := controlsys.Series(ab, c)
		if err != nil {
			t.Fatalf("case %d Series(ab,c): %v\n%v", i, err, ctx)
		}
		bc, err := controlsys.Series(b, c)
		if err != nil {
			t.Fatalf("case %d Series(b,c): %v\n%v", i, err, ctx)
		}
		right, err := controlsys.Series(a, bc)
		if err != nil {
			t.Fatalf("case %d Series(a,bc): %v\n%v", i, err, ctx)
		}
		sweepRobustCheckFreqPair(t, fmt.Sprintf("case %d associativity", i), left, right, sweepRobustOmegas, 1e-9, ctx)
		fl, err := left.FreqResponse(sweepRobustOmegas)
		if err != nil {
			t.Fatal(err)
		}
		for k, w := range sweepRobustOmegas {
			ra, oka := sweepRobustRef(ma, w)
			rb, okb := sweepRobustRef(mb, w)
			rc, okc := sweepRobustRef(mc, w)
			if !oka || !okb || !okc {
				continue
			}
			ref := sweepRobustCMMul(rc, sweepRobustCMMul(rb, ra))
			for r := range d3 {
				for q := range d0 {
					if x, y := fl.At(k, r, q), ref.at(r, q); cmplx.Abs(x-y) > 1e-9*(1+cmplx.Abs(y)) {
						t.Fatalf("case %d series vs ref w=%g (%d,%d): %v vs %v\n%v", i, w, r, q, x, y, ctx)
					}
				}
			}
		}

		pa, pb := sweepRobustRandChain(rng, dt, stable, d0, d1), sweepRobustRandChain(rng, dt, stable, d0, d1)
		sa, sb := sweepRobustMustBuild(t, pa), sweepRobustMustBuild(t, pb)
		pctx := sweepRobustStringer(func() string { return fmt.Sprintf("a:\n%v\nb:\n%v", pa, pb) })
		ab, err = controlsys.Parallel(sa, sb)
		if err != nil {
			t.Fatalf("case %d Parallel(a,b): %v\n%v", i, err, pctx)
		}
		ba, err := controlsys.Parallel(sb, sa)
		if err != nil {
			t.Fatalf("case %d Parallel(b,a): %v\n%v", i, err, pctx)
		}
		sweepRobustCheckFreqPair(t, fmt.Sprintf("case %d commutativity", i), ab, ba, sweepRobustOmegas, 1e-10, pctx)
		fp, err := ab.FreqResponse(sweepRobustOmegas)
		if err != nil {
			t.Fatal(err)
		}
		for k, w := range sweepRobustOmegas {
			ra, oka := sweepRobustRef(pa, w)
			rb, okb := sweepRobustRef(pb, w)
			if !oka || !okb {
				continue
			}
			ref := sweepRobustCMAdd(ra, rb)
			for r := range d1 {
				for q := range d0 {
					if x, y := fp.At(k, r, q), ref.at(r, q); cmplx.Abs(x-y) > 1e-9*(1+cmplx.Abs(y)) {
						t.Fatalf("case %d parallel vs ref w=%g (%d,%d): %v vs %v\n%v", i, w, r, q, x, y, pctx)
					}
				}
			}
		}
	})
}

type sweepRobustStringer func() string

func (s sweepRobustStringer) String() string { return s() }

// TestSweepRobustDelayFactorization (E-B): a separable delay d_ij = a_j + b_i
// as a Delay matrix, as InputDelay+OutputDelay, after PullDelaysToLFT and
// after SetDelayModel(GetDelayModel) all match the reference.
func TestSweepRobustDelayFactorization(t *testing.T) {
	sweepRobustCases(206, 20, func(i int, rng *rand.Rand, dt float64, stable bool) {
		md := sweepRobustRandModel(rng, dt, stable, true, false)
		io := md
		io.in, io.out = nil, nil
		io.io = make([]float64, md.p*md.m)
		for r := range md.p {
			for c := range md.m {
				io.io[r*md.m+c] = md.in[c] + md.out[r]
			}
		}
		label := fmt.Sprintf("case %d", i)
		inout := sweepRobustMustBuild(t, md)
		matrix := sweepRobustMustBuild(t, io)
		sweepRobustCheckFreq(t, label+" in/out", inout, md, sweepRobustOmegas, 1e-9)
		sweepRobustCheckFreq(t, label+" matrix", matrix, md, sweepRobustOmegas, 1e-9)
		for name, s := range map[string]*controlsys.System{"in/out": inout, "matrix": matrix} {
			pulled, err := s.PullDelaysToLFT()
			if err != nil {
				t.Fatalf("%s %s PullDelaysToLFT: %v\n%v", label, name, err, md)
			}
			if td, err := pulled.TotalDelay(); err != nil || mat.Norm(td, 1) != 0 {
				t.Fatalf("%s %s PullDelaysToLFT left I/O delays %v (err %v)", label, name, td, err)
			}
			sweepRobustCheckFreq(t, label+" "+name+" pulled", pulled, md, sweepRobustOmegas, 1e-9)
			H, tau, err := s.GetDelayModel()
			if err != nil {
				t.Fatal(err)
			}
			if H.HasDelay() {
				t.Fatalf("%s %s GetDelayModel H has delays", label, name)
			}
			back, err := controlsys.SetDelayModel(H, tau)
			if err != nil {
				t.Fatalf("%s %s SetDelayModel: %v", label, name, err)
			}
			sweepRobustCheckFreq(t, label+" "+name+" SetDelayModel", back, md, sweepRobustOmegas, 1e-9)
		}
	})
}

// TestSweepRobustDiscreteBufferVsAbsorb (E-B): discrete internal-delay
// buffer simulation, AbsorbDelay augmented states and the hand recursion give
// equal Simulate and Lsim output sequences.
func TestSweepRobustDiscreteBufferVsAbsorb(t *testing.T) {
	sweepRobustCases(207, 40, func(i int, rng *rand.Rand, dt float64, stable bool) {
		if dt == 0 {
			return
		}
		md := sweepRobustRandModel(rng, dt, stable, true, true)
		sys := sweepRobustMustBuild(t, md)
		const steps = 60
		u, U := sweepRobustRandInput(rng, md.m, steps)
		ref := sweepRobustSimRef(md, u, steps)
		buf, err := sys.Simulate(U, nil, nil)
		if err != nil {
			t.Fatalf("case %d Simulate: %v\n%v", i, err, md)
		}
		sweepRobustCheckSim(t, fmt.Sprintf("case %d buffer", i), buf.Y, ref, 1e-10, md)
		abs, err := sys.AbsorbDelay()
		if err != nil {
			t.Fatalf("case %d AbsorbDelay: %v\n%v", i, err, md)
		}
		if abs.HasDelay() {
			t.Fatalf("case %d AbsorbDelay kept delays", i)
		}
		ar, err := abs.Simulate(U, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		sweepRobustCheckSim(t, fmt.Sprintf("case %d absorbed", i), ar.Y, ref, 1e-10, md)
		tm := make([]float64, steps)
		uT := mat.NewDense(steps, md.m, nil)
		for k := range steps {
			tm[k] = dt * float64(k)
			for j := range md.m {
				uT.Set(k, j, u[j][k])
			}
		}
		for name, s := range map[string]*controlsys.System{"buffer": sys, "absorbed": abs} {
			lr, err := controlsys.Lsim(s, uT, tm, nil)
			if err != nil {
				t.Fatalf("case %d %s Lsim: %v\n%v", i, name, err, md)
			}
			sweepRobustCheckSim(t, fmt.Sprintf("case %d %s Lsim", i, name), lr.Y, ref, 1e-10, md)
		}
	})
}

// TestSweepRobustExtremeDelayRatio (E-C): τ = 1e-6 and 1e4 in one model, as
// input and as internal delays, match the reference; Step on a short horizon
// finishes and stays finite.
func TestSweepRobustExtremeDelayRatio(t *testing.T) {
	base := sweepRobustModel{n: 2, m: 2, p: 1, A: []float64{-1, 0.4, -0.3, -2}, B: []float64{1, 0, 0.5, 1}, C: []float64{1, -0.7}, D: []float64{0, 0.1}}
	io := base
	io.in = []float64{1e-6, 1e4}
	omegas := []float64{1e-3, 0.7, 1e3}
	sys := sweepRobustMustBuild(t, io)
	sweepRobustCheckFreq(t, "input 1e-6/1e4", sys, io, omegas, 1e-8)
	in := base
	in.N, in.tau = 2, []float64{1e-6, 1e4}
	in.B2, in.C2 = []float64{0.2, 0, 0, 0.1}, []float64{0.3, 0, 0, 0.2}
	in.D12, in.D21, in.D22 = []float64{0.4, 0.3}, []float64{1, 0, 0, 1}, []float64{0.2, 0.1, 0, -0.3}
	isys := sweepRobustMustBuild(t, in)
	sweepRobustCheckFreq(t, "internal 1e-6/1e4", isys, in, omegas, 1e-8)
	for name, s := range map[string]*controlsys.System{"input": sys, "internal": isys} {
		var resp *controlsys.TimeResponse
		var err error
		if r := sweepRobustGuard(t, name+" Step", 20*time.Second, func() { resp, err = controlsys.Step(s, 0.5) }); r != nil {
			t.Fatalf("%s Step panicked: %v", name, r)
		}
		if err != nil {
			t.Logf("%s Step: explicit error %v", name, err)
			continue
		}
		r, c := resp.Y.Dims()
		for a := range r {
			for b := range c {
				if v := resp.Y.At(a, b); math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("%s Step non-finite at (%d,%d)", name, a, b)
				}
			}
		}
	}
}

// TestSweepRobustLargeDiscreteDelay (E-C): discrete delays of 1e5 and 1e9
// samples are simulated with bounded memory or rejected; never panic.
func TestSweepRobustLargeDiscreteDelay(t *testing.T) {
	const steps = 64
	rng := rand.New(rand.NewSource(301))
	u, U := sweepRobustRandInput(rng, 1, steps)
	for _, d := range []float64{1e5, 1e9} {
		for _, kind := range []string{"input", "output"} {
			md := sweepRobustModel{n: 1, m: 1, p: 1, dt: 1, A: []float64{0.5}, B: []float64{1}, C: []float64{1}, D: []float64{1}}
			if kind == "input" {
				md.in = []float64{d}
			} else {
				md.out = []float64{d}
			}
			sys := sweepRobustMustBuild(t, md)
			var resp *controlsys.Response
			var err error
			if r := sweepRobustGuard(t, kind+" Simulate", 10*time.Second, func() { resp, err = sys.Simulate(U, nil, nil) }); r != nil {
				t.Fatalf("%s delay %g Simulate panicked: %v", kind, d, r)
			}
			if err != nil {
				t.Fatalf("%s delay %g Simulate: %v", kind, d, err)
			}
			for k := range steps {
				if v := resp.Y.At(0, k); v != 0 {
					t.Fatalf("%s delay %g y[%d]=%v, want 0", kind, d, k, v)
				}
			}
			if d > 1e5 {
				continue
			}
			sweepRobustCheckFreq(t, fmt.Sprintf("%s delay %g", kind, d), sys, md, []float64{0.3, 1.1}, 1e-9)
		}
	}

	imd := sweepRobustModel{n: 1, m: 1, p: 1, N: 1, dt: 1, A: []float64{0.5}, B: []float64{1}, C: []float64{1}, D: []float64{0.2},
		B2: []float64{0.1}, C2: []float64{0.3}, D12: []float64{0.4}, D21: []float64{1}, D22: []float64{0.5}, tau: []float64{1e5}}
	isys := sweepRobustMustBuild(t, imd)
	var resp *controlsys.Response
	var err error
	if r := sweepRobustGuard(t, "internal 1e5 Simulate", 10*time.Second, func() { resp, err = isys.Simulate(U, nil, nil) }); r != nil {
		t.Fatalf("internal 1e5 Simulate panicked: %v", r)
	}
	if err != nil {
		t.Fatalf("internal 1e5 Simulate: %v", err)
	}
	sweepRobustCheckSim(t, "internal 1e5", resp.Y, sweepRobustSimRef(imd, u, steps), 1e-12, imd)
	sweepRobustCheckFreq(t, "internal 1e5", isys, imd, []float64{0.3, 1.1}, 1e-9)

	for _, tc := range []struct {
		name string
		tau  float64
	}{{"absorb1e9", 1e9}, {"absorb1e5", 1e5}} {
		t.Run(tc.name, func(t *testing.T) {
			huge := sweepRobustModel{n: 1, m: 1, p: 1, dt: 1, A: []float64{0.5}, B: []float64{1}, C: []float64{1}, D: []float64{0}, in: []float64{tc.tau}}
			hs := sweepRobustMustBuild(t, huge)
			var aerr error
			if r := sweepRobustGuard(t, "AbsorbDelay "+tc.name, 10*time.Second, func() { _, aerr = hs.AbsorbDelay() }); r != nil {
				t.Fatalf("AbsorbDelay(InputDelay=%g) panicked: %v", tc.tau, r)
			}
			if !errors.Is(aerr, controlsys.ErrInvalidArgument) {
				t.Fatalf("AbsorbDelay(InputDelay=%g) err = %v, want ErrInvalidArgument size error", tc.tau, aerr)
			}
		})
	}
	t.Run("freq1e9", func(t *testing.T) {
		t.Skipf("known defect, ergo %s: discrete FreqResponse applies a d-sample I/O delay by d complex divisions (seconds per frequency at 1e9); not run", sweepRobustErgoFreqIOLoop)
	})
	t.Run("internal1e9", func(t *testing.T) {
		hmd := imd
		hmd.tau = []float64{1e9}
		hs := sweepRobustMustBuild(t, hmd)
		var hresp *controlsys.Response
		var herr error
		if r := sweepRobustGuard(t, "internal 1e9 Simulate", 10*time.Second, func() { hresp, herr = hs.Simulate(U, nil, nil) }); r != nil {
			t.Fatalf("internal 1e9 Simulate panicked: %v", r)
		}
		if herr != nil {
			t.Fatalf("internal 1e9 Simulate: %v", herr)
		}
		sweepRobustCheckSim(t, "internal 1e9", hresp.Y, sweepRobustSimRef(hmd, u, steps), 1e-12, hmd)

		mmd := sweepRobustModel{n: 1, m: 1, p: 1, N: 2, dt: 1, A: []float64{0.5}, B: []float64{1}, C: []float64{1}, D: []float64{0.2},
			B2: []float64{0.1, -0.2}, C2: []float64{0.3, 0.7}, D12: []float64{0.4, 0.6}, D21: []float64{1, -0.5},
			D22: []float64{0.5, 0.1, -0.3, 0.2}, tau: []float64{3, 1e9}}
		ms := sweepRobustMustBuild(t, mmd)
		if r := sweepRobustGuard(t, "internal {3,1e9} Simulate", 10*time.Second, func() { hresp, herr = ms.Simulate(U, nil, nil) }); r != nil {
			t.Fatalf("internal {3,1e9} Simulate panicked: %v", r)
		}
		if herr != nil {
			t.Fatalf("internal {3,1e9} Simulate: %v", herr)
		}
		sweepRobustCheckSim(t, "internal {3,1e9}", hresp.Y, sweepRobustSimRef(mmd, u, steps), 1e-12, mmd)
	})
}

const (
	sweepRobustErgoFreqIOLoop = "CIC2TD"
	sweepRobustErgoFeedbackEr = "KVUJYJ"
)

func sweepRobustIsLoopError(err error) bool {
	return errors.Is(err, controlsys.ErrAlgebraicLoop) || errors.Is(err, controlsys.ErrSingularTransform)
}

// TestSweepRobustMixedTimeScales (E-C): poles at -1e-4 and -1e4 with input
// and internal delays match the reference across ω ∈ [1e-6, 1e6].
func TestSweepRobustMixedTimeScales(t *testing.T) {
	T := mat.NewDense(2, 2, []float64{1, 0.5, 0.2, 1})
	var Ti, A mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	A.Mul(T, mat.NewDense(2, 2, []float64{-1e-4, 0, 0, -1e4}))
	A.Mul(&A, &Ti)
	md := sweepRobustModel{n: 2, m: 1, p: 1, A: append([]float64(nil), A.RawMatrix().Data...), B: []float64{1, 1e4}, C: []float64{1e-4, 1}, D: []float64{0},
		in: []float64{0.3}, N: 1, tau: []float64{2}, B2: []float64{1e-4, 0.5}, C2: []float64{0.2, 1e-4}, D12: []float64{0.3}, D21: []float64{0.1}, D22: []float64{0.2}}
	sys := sweepRobustMustBuild(t, md)
	omegas := []float64{1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6}
	sweepRobustCheckFreq(t, "mixed scales", sys, md, omegas, 1e-8)
	var resp *controlsys.TimeResponse
	var err error
	if r := sweepRobustGuard(t, "mixed scales Step", 20*time.Second, func() { resp, err = controlsys.Step(sys, 1) }); r != nil {
		t.Fatalf("Step panicked: %v", r)
	}
	if err == nil {
		for k := range resp.T {
			if v := resp.Y.At(0, k); math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("mixed scales Step non-finite at t=%g", resp.T[k])
			}
		}
	}
}

// TestSweepRobustIllConditionedA (E-C): a highly non-normal A (cond ~1e12)
// with delays still matches the reference to a conditioning-aware tolerance.
func TestSweepRobustIllConditionedA(t *testing.T) {
	md := sweepRobustModel{n: 3, m: 1, p: 1, A: []float64{-1, 1e6, 0, 0, -1.001, 1e5, 0, 0, -2}, B: []float64{0, 0, 1}, C: []float64{1, 0, 0}, D: []float64{0},
		in: []float64{0.5}, N: 1, tau: []float64{1.2}, B2: []float64{0, 1e-6, 0}, C2: []float64{1e-6, 0, 0}, D12: []float64{0}, D21: []float64{0}, D22: []float64{0.3}}
	sys := sweepRobustMustBuild(t, md)
	sweepRobustCheckFreq(t, "ill-conditioned", sys, md, []float64{1e-3, 0.5, 1, 7, 100}, 1e-8)
	dmd := md
	dmd.dt, dmd.in, dmd.tau = 0.1, []float64{2}, []float64{3}
	dmd.A = []float64{0.9, 1e5, 0, 0, 0.9001, 1e4, 0, 0, 0.5}
	sweepRobustCheckFreq(t, "ill-conditioned discrete", sweepRobustMustBuild(t, dmd), dmd, []float64{1e-3, 0.5, 1, 7, 30}, 1e-8)
}

// TestSweepRobustInvalidDelaysRejected (E-C): NaN, ±Inf, negative and
// fractional-discrete delays are rejected by every setter and leave the model
// unchanged.
func TestSweepRobustInvalidDelaysRejected(t *testing.T) {
	bad := []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1, -1e-300}
	one := mat.NewDense(1, 1, []float64{1})
	half := mat.NewDense(1, 1, []float64{0.5})
	for _, dt := range []float64{0, 1} {
		vals := bad
		if dt > 0 {
			vals = append(append([]float64(nil), bad...), 0.5, 1e9+0.5)
		}
		for _, v := range vals {
			sys := sweepRobustScalarSys(t, -0.5, 1, 1, 0, dt)
			label := fmt.Sprintf("dt=%g v=%g", dt, v)
			if err := sys.SetInputDelay([]float64{v}); err == nil || sys.InputDelay != nil {
				t.Errorf("%s SetInputDelay accepted (err=%v)", label, err)
			}
			if err := sys.SetOutputDelay([]float64{v}); err == nil || sys.OutputDelay != nil {
				t.Errorf("%s SetOutputDelay accepted (err=%v)", label, err)
			}
			if err := sys.SetDelay(mat.NewDense(1, 1, []float64{v})); err == nil || sys.Delay != nil {
				t.Errorf("%s SetDelay accepted (err=%v)", label, err)
			}
			if err := sys.SetInternalDelay([]float64{v}, one, one, one, one, half); err == nil || sys.LFT != nil {
				t.Errorf("%s SetInternalDelay accepted (err=%v)", label, err)
			}
			if err := sys.SetInternalDelay([]float64{1, v}, mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(2, 1, []float64{1, 1}),
				mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(2, 2, nil)); err == nil || sys.LFT != nil {
				t.Errorf("%s SetInternalDelay second tau accepted (err=%v)", label, err)
			}
			H, _ := controlsys.NewGain(mat.NewDense(2, 2, []float64{0, 1, 1, 0.5}), dt)
			if s, err := controlsys.SetDelayModel(H, []float64{v}); err == nil {
				t.Errorf("%s SetDelayModel accepted: %v", label, s)
			}
			if s, err := controlsys.NewWithDelay(one, one, one, half, mat.NewDense(1, 1, []float64{v}), dt); err == nil {
				t.Errorf("%s NewWithDelay accepted: %v", label, s)
			}
			if fr, err := sys.FreqResponse([]float64{1}); err != nil || cmplx.IsNaN(fr.At(0, 0, 0)) {
				t.Errorf("%s model unusable after rejected delays: %v", label, err)
			}
		}
	}
}

// TestSweepRobustNearSingularD22 (E-C): I-D22 with condition ~1e14 is solved
// accurately or rejected; condition ~1e16 returns ErrAlgebraicLoop.
func TestSweepRobustNearSingularD22(t *testing.T) {
	for _, eps := range []float64{1e-14, 1e-17} {
		d11 := 1 - eps
		delta := 1 - d11
		md := sweepRobustModel{m: 1, p: 1, N: 2, dt: 0.1, D: []float64{0}, D12: []float64{1, 0}, D21: []float64{0, 1},
			D22: []float64{0, -1, 0, d11}, tau: []float64{1, 2}}
		sys := sweepRobustMustBuild(t, md)
		z, err := sys.ZeroDelayApprox()
		label := fmt.Sprintf("eps=%g delta=%g", eps, delta)
		if delta == 0 {
			if !errors.Is(err, controlsys.ErrAlgebraicLoop) {
				t.Fatalf("%s: err=%v, want ErrAlgebraicLoop", label, err)
			}
			continue
		}
		if err != nil {
			if !errors.Is(err, controlsys.ErrAlgebraicLoop) {
				t.Fatalf("%s: unexpected error %v", label, err)
			}
			t.Logf("%s: rejected (%v)", label, err)
			continue
		}
		want := -1 / delta
		got := z.D.At(0, 0)
		if math.IsNaN(got) || math.Abs(got-want) > 1e-10*math.Abs(want) {
			t.Fatalf("%s: D=%.17g want %.17g", label, got, want)
		}
		fr, err := sys.FreqResponse([]float64{0.5})
		if err != nil {
			t.Fatalf("%s FreqResponse: %v", label, err)
		}
		if v := fr.At(0, 0, 0); cmplx.IsNaN(v) || cmplx.IsInf(v) {
			t.Fatalf("%s FreqResponse non-finite %v", label, v)
		}
		sweepRobustCheckFreq(t, label, sys, md, []float64{0.5, 3}, 1e-9)
	}

	for _, eps := range []float64{1e-14, 1e-17} {
		k := 1 - eps
		cl, err := controlsys.Feedback(sweepRobustGain(t, 1, 0), sweepRobustGain(t, k, 0), 1)
		if 1-k == 0 {
			if !sweepRobustIsLoopError(err) {
				t.Fatalf("Feedback k=%v: err=%v want ErrAlgebraicLoop", k, err)
			}
			continue
		}
		if err != nil {
			if !sweepRobustIsLoopError(err) {
				t.Fatalf("Feedback k=%v: %v", k, err)
			}
			continue
		}
		if got, want := cl.D.At(0, 0), 1/(1-k); math.Abs(got-want) > 1e-10*want {
			t.Fatalf("Feedback k=%.17g: D=%.17g want %.17g", k, got, want)
		}
	}
}

// TestSweepRobustSetInternalDelayDimensionMismatch (E-C): wrong LFT block
// shapes return ErrDimensionMismatch, missing required blocks
// ErrInvalidArgument, and the model is left unchanged.
func TestSweepRobustSetInternalDelayDimensionMismatch(t *testing.T) {
	n, m, p, N := 2, 1, 1, 2
	good := func() []*mat.Dense {
		return []*mat.Dense{mat.NewDense(n, N, nil), mat.NewDense(N, n, nil), mat.NewDense(p, N, nil), mat.NewDense(N, m, nil), mat.NewDense(N, N, nil)}
	}
	wrong := []*mat.Dense{mat.NewDense(n, N+1, nil), mat.NewDense(N, n+1, nil), mat.NewDense(p+1, N, nil), mat.NewDense(N+1, m, nil), mat.NewDense(N, N+1, nil)}
	names := []string{"B2", "C2", "D12", "D21", "D22"}
	for k := range 5 {
		sys, _ := controlsys.NewFromSlices(n, m, p, []float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 1}, []float64{0}, 0)
		g := good()
		g[k] = wrong[k]
		if err := sys.SetInternalDelay([]float64{1, 2}, g[0], g[1], g[2], g[3], g[4]); !errors.Is(err, controlsys.ErrDimensionMismatch) || sys.LFT != nil {
			t.Errorf("%s wrong shape: err=%v", names[k], err)
		}
		g = good()
		g[k] = nil
		if err := sys.SetInternalDelay([]float64{1, 2}, g[0], g[1], g[2], g[3], g[4]); !errors.Is(err, controlsys.ErrInvalidArgument) || sys.LFT != nil {
			t.Errorf("%s nil: err=%v", names[k], err)
		}
		g = good()
		if err := sys.SetInternalDelay([]float64{1}, g[0], g[1], g[2], g[3], g[4]); !errors.Is(err, controlsys.ErrDimensionMismatch) || sys.LFT != nil {
			t.Errorf("tau length 1 vs N=2: err=%v", err)
		}
	}
	sys, _ := controlsys.NewFromSlices(n, m, p, []float64{-1, 0.5, 0, -2}, []float64{1, 0}, []float64{1, 1}, []float64{0}, 0)
	if err := sys.SetInputDelay([]float64{1, 2}); !errors.Is(err, controlsys.ErrDimensionMismatch) {
		t.Errorf("SetInputDelay wrong length: %v", err)
	}
	if err := sys.SetOutputDelay([]float64{}); !errors.Is(err, controlsys.ErrDimensionMismatch) {
		t.Errorf("SetOutputDelay wrong length: %v", err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 1, nil)); !errors.Is(err, controlsys.ErrDimensionMismatch) {
		t.Errorf("SetDelay wrong shape: %v", err)
	}
	if _, err := controlsys.SetDelayModel(sys, []float64{1, 2}); !errors.Is(err, controlsys.ErrDimensionMismatch) {
		t.Errorf("SetDelayModel too small: %v", err)
	}
}

// TestSweepRobustConcurrentDelayedUse (E-E): one delayed System shared by
// goroutines calling FreqResponse, Step, Simulate and AbsorbDelay yields the
// serial results and is not mutated.
func TestSweepRobustConcurrentDelayedUse(t *testing.T) {
	cmd := sweepRobustModel{n: 2, m: 1, p: 1, A: []float64{-1, 0.4, -0.3, -2}, B: []float64{1, 0.5}, C: []float64{1, -0.7}, D: []float64{0.1},
		in: []float64{0.2}, N: 1, tau: []float64{0.5}, B2: []float64{0.2, 0.1}, C2: []float64{0.3, 0.2}, D12: []float64{0.4}, D21: []float64{0.5}, D22: []float64{0.2}}
	dmd := cmd
	dmd.dt, dmd.in, dmd.tau = 0.1, []float64{2}, []float64{3}
	dmd.A = []float64{0.5, 0.2, -0.1, 0.3}
	csys, dsys := sweepRobustMustBuild(t, cmd), sweepRobustMustBuild(t, dmd)
	_, U := sweepRobustRandInput(rand.New(rand.NewSource(9)), 1, 50)
	run := func() (fc, fd *controlsys.FreqResponseMatrix, sc, sd *controlsys.TimeResponse, sim *controlsys.Response, ad *controlsys.System, err error) {
		if fc, err = csys.FreqResponse(sweepRobustOmegas); err != nil {
			return
		}
		if fd, err = dsys.FreqResponse(sweepRobustOmegas); err != nil {
			return
		}
		if sc, err = controlsys.Step(csys, 3); err != nil {
			return
		}
		if sd, err = controlsys.Step(dsys, 3); err != nil {
			return
		}
		if sim, err = dsys.Simulate(U, nil, nil); err != nil {
			return
		}
		ad, err = dsys.AbsorbDelay()
		return
	}
	fc0, fd0, sc0, sd0, sim0, ad0, err := run()
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for range 8 {
		wg.Go(func() {
			for range 3 {
				fc, fd, sc, sd, sim, ad, err := run()
				if err != nil {
					errs <- err
					return
				}
				if !sweepRobustEqualComplex(fc.Data, fc0.Data) || !sweepRobustEqualComplex(fd.Data, fd0.Data) ||
					!mat.Equal(sc.Y, sc0.Y) || !mat.Equal(sd.Y, sd0.Y) || !mat.Equal(sim.Y, sim0.Y) || !mat.Equal(ad.A, ad0.A) {
					errs <- fmt.Errorf("concurrent result differs from serial baseline")
					return
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	sweepRobustCheckFreq(t, "after concurrent use", csys, cmd, sweepRobustOmegas, 1e-9)
	sweepRobustCheckFreq(t, "after concurrent use discrete", dsys, dmd, sweepRobustOmegas, 1e-9)
}

func sweepRobustEqualComplex(a, b []complex128) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] && !(cmplx.IsNaN(a[i]) && cmplx.IsNaN(b[i])) {
			return false
		}
	}
	return true
}

// FuzzSweepRobustDelayTopology (E-D): random small models with input, output,
// I/O and internal delays never panic in FreqResponse, Step, Simulate,
// AbsorbDelay, PullDelaysToLFT, GetDelayModel or ZeroDelayApprox.
func FuzzSweepRobustDelayTopology(f *testing.F) {
	f.Add(int64(1), uint8(1), uint8(1), uint8(1), uint8(1), uint8(0xff), false)
	f.Add(int64(2), uint8(0), uint8(2), uint8(2), uint8(2), uint8(0x0f), true)
	f.Add(int64(3), uint8(3), uint8(2), uint8(1), uint8(0), uint8(0x07), true)
	f.Add(int64(4), uint8(2), uint8(1), uint8(2), uint8(2), uint8(0xf0), false)
	f.Add(int64(5), uint8(0), uint8(1), uint8(1), uint8(1), uint8(0x18), true)
	f.Add(int64(1), uint8('H'), uint8(1), uint8('e'), uint8(1), uint8(0xff), false)
	f.Fuzz(func(t *testing.T, seed int64, n8, m8, p8, N8, flags uint8, discrete bool) {
		rng := rand.New(rand.NewSource(seed))
		md := sweepRobustModel{n: int(n8 % 4), m: 1 + int(m8%2), p: 1 + int(p8%2), N: int(N8 % 3)}
		if discrete {
			md.dt = 0.1
		}
		n, m, p, N := md.n, md.m, md.p, md.N
		md.A = sweepRobustRandSlice(rng, n*n, 1)
		md.B = sweepRobustRandSlice(rng, n*m, 1)
		md.C = sweepRobustRandSlice(rng, p*n, 1)
		md.D = sweepRobustRandSlice(rng, p*m, 1)
		if flags&1 != 0 {
			md.in = make([]float64, m)
			for j := range m {
				md.in[j] = sweepRobustRandDelay(rng, md.dt, true)
			}
		}
		if flags&2 != 0 {
			md.out = make([]float64, p)
			for i := range p {
				md.out[i] = sweepRobustRandDelay(rng, md.dt, true)
			}
		}
		if flags&4 != 0 {
			md.io = make([]float64, p*m)
			for k := range md.io {
				md.io[k] = sweepRobustRandDelay(rng, md.dt, true)
			}
		}
		if N > 0 {
			md.tau = make([]float64, N)
			for k := range N {
				md.tau[k] = sweepRobustRandDelay(rng, md.dt, false)
			}
			scale := 0.3
			if flags&8 != 0 {
				scale = 1.5
			}
			md.B2 = sweepRobustRandSlice(rng, n*N, 1)
			md.C2 = sweepRobustRandSlice(rng, N*n, 1)
			md.D12 = sweepRobustRandSlice(rng, p*N, 1)
			md.D21 = sweepRobustRandSlice(rng, N*m, 1)
			md.D22 = sweepRobustRandSlice(rng, N*N, scale)
			if flags&16 != 0 {
				for k := range N {
					md.D22[k*N+k] = 1
				}
			}
		}
		sys, err := sweepRobustBuild(md)
		if err != nil {
			return
		}
		tf := 2.0
		if flags&32 != 0 {
			tf = 0
		}
		_, _ = sys.FreqResponse([]float64{0, 0.5, 3})
		_, _ = controlsys.Step(sys, tf)
		_, _ = sys.PullDelaysToLFT()
		if H, tau, err := sys.GetDelayModel(); err == nil {
			_, _ = controlsys.SetDelayModel(H, tau)
		}
		_, _ = sys.ZeroDelayApprox()
		if abs, err := sys.AbsorbDelay(); err == nil {
			_, _ = abs.FreqResponse([]float64{0.5})
		}
		if discrete {
			_, U := sweepRobustRandInput(rng, m, 10)
			_, _ = sys.Simulate(U, nil, nil)
		}
	})
}
