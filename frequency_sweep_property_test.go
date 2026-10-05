package controlsys

import (
	"encoding/json"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// sweepPointwiseErr returns the worst per-frequency error of FreqResponse
// relative to FreqResponsePointwise, normalised by max|G(jω)|.
func sweepPointwiseErr(t *testing.T, sys *System, omega []float64) (float64, float64) {
	t.Helper()
	got, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	want, err := sys.FreqResponsePointwise(omega)
	if err != nil {
		t.Fatalf("FreqResponsePointwise: %v", err)
	}
	pm := want.P * want.M
	worst, at := 0.0, 0.0
	for k, w := range omega {
		norm, diff := 0.0, 0.0
		for i := k * pm; i < (k+1)*pm; i++ {
			norm = max(norm, cmplx.Abs(want.Data[i]))
			diff = max(diff, cmplx.Abs(got.Data[i]-want.Data[i]))
		}
		if norm == 0 {
			norm = 1
		}
		if e := diff / norm; e > worst || math.IsNaN(e) {
			worst, at = e, w
		}
	}
	return worst, at
}

type sweepModelKind int

const (
	sweepDense sweepModelKind = iota
	sweepBlockTriangular
	sweepBadlyScaled
	sweepNearDecoupled
	sweepCascade
	sweepWeakRows
	sweepModelKinds
)

func (k sweepModelKind) String() string {
	return [...]string{"dense", "blocktri", "scaled", "decoupled", "cascade", "weakrows"}[k]
}

// randomSweepModel draws a stable model whose realisation stresses the sweep:
// reducible structure, wide diagonal similarity scaling, and near-zero
// couplings like those Series leaves behind.
// With descriptor set, the same model is realised as (E, EA, EB, C) for a
// random well-conditioned E, and the scaling is applied to E as well. E is
// diagonal for weakrows so that EA keeps the near-empty rows.
func randomSweepModel(rng *rand.Rand, kind sweepModelKind, dt float64, descriptor bool) (stressed, twin *System) {
	n := 1 + rng.IntN(40)
	m, p := 1+rng.IntN(4), 1+rng.IntN(4)
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, rng.NormFloat64()/math.Sqrt(float64(n)))
		}
	}
	switch kind {
	case sweepBlockTriangular, sweepCascade:
		k := 1 + rng.IntN(n)
		for i := range k {
			for j := k; j < n; j++ {
				A.Set(i, j, 0)
			}
		}
	case sweepNearDecoupled:
		k := 1 + rng.IntN(n)
		for i := range k {
			for j := k; j < n; j++ {
				A.Set(i, j, rng.NormFloat64()*1e-15)
			}
		}
	}
	stabilise(A, n, dt)
	if kind == sweepWeakRows {
		// Slow input-driven modes that Series couples back only through
		// roundoff: a near-empty row against a heavy column. A zero row keeps
		// A[i,i] an eigenvalue and leaves the others unchanged.
		for i := range n {
			if rng.IntN(3) != 0 {
				continue
			}
			tiny := 1e-15 * float64(rng.IntN(2))
			for j := range n {
				if j != i {
					A.Set(i, j, rng.NormFloat64()*tiny)
					A.Set(j, i, A.At(j, i)*1e3)
				}
			}
			slow := math.Pow(10, -1-3*rng.Float64())
			if dt == 0 {
				A.Set(i, i, -slow)
			} else {
				A.Set(i, i, 1-slow)
			}
		}
	}
	B := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	D := mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			C.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64()*float64(rng.IntN(2)))
		}
	}
	var E *mat.Dense
	if descriptor {
		E = mat.NewDense(n, n, nil)
		for i := range n {
			if kind != sweepWeakRows {
				for j := range n {
					E.Set(i, j, 0.3*rng.NormFloat64()/math.Sqrt(float64(n)))
				}
			}
			E.Set(i, i, E.At(i, i)+math.Pow(2, 2*rng.Float64()-1))
		}
		A.Mul(E, mat.DenseCopyOf(A))
		B.Mul(E, mat.DenseCopyOf(B))
	}
	build := func(A, B, C, D, E *mat.Dense) *System {
		var sys *System
		var err error
		if E == nil {
			sys, err = New(A, B, C, D, dt)
		} else {
			sys, err = NewDescriptor(A, B, C, D, E, dt)
		}
		if err != nil {
			panic(err)
		}
		return sys
	}
	var twinE *mat.Dense
	if E != nil {
		twinE = mat.DenseCopyOf(E)
	}
	twin = build(mat.DenseCopyOf(A), mat.DenseCopyOf(B), mat.DenseCopyOf(C), mat.DenseCopyOf(D), twinE)
	if kind == sweepBadlyScaled || kind == sweepCascade {
		decades := float64(1 + rng.IntN(6))
		for i := range n {
			s := math.Pow(10, decades*(2*rng.Float64()-1))
			for j := range n {
				A.Set(i, j, A.At(i, j)*s)
				A.Set(j, i, A.At(j, i)/s)
				if E != nil {
					E.Set(i, j, E.At(i, j)*s)
					E.Set(j, i, E.At(j, i)/s)
				}
			}
			for j := range m {
				B.Set(i, j, B.At(i, j)*s)
			}
			for j := range p {
				C.Set(j, i, C.At(j, i)/s)
			}
		}
	}
	return build(A, B, C, D, E), twin
}

// stabilise shifts A so every eigenvalue sits at least a fixed margin inside
// the stability boundary, keeping the resolvent well conditioned on the axis.
func stabilise(A *mat.Dense, n int, dt float64) {
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		panic("eig")
	}
	if dt == 0 {
		worst := math.Inf(-1)
		for _, v := range eig.Values(nil) {
			worst = max(worst, real(v))
		}
		for i := range n {
			A.Set(i, i, A.At(i, i)-worst-0.2)
		}
		return
	}
	rho := 0.0
	for _, v := range eig.Values(nil) {
		rho = max(rho, cmplx.Abs(v))
	}
	if rho > 0 {
		A.Scale(0.85/rho, A)
	}
}

// kernelErr evaluates solver over omega and returns the worst error against
// oracleResponse(ref) at every stride-th frequency, normalised by max|G| per
// frequency.
func kernelErr(t *testing.T, sys, ref *System, solver frequencyPointSolver, omega []float64, stride int) (float64, float64) {
	t.Helper()
	n, m, p := sys.Dims()
	td := newTimeDomain(sys.Dt)
	got := make([]complex128, p*m)
	worst, at := 0.0, 0.0
	for k := 0; k < len(omega); k += stride {
		s := td.frequencyVariable(omega[k])
		if err := solver.evalInto(s, got); err != nil {
			t.Fatalf("ω=%g: %v", omega[k], err)
		}
		want := oracleResponse(t, ref, s, n, m, p)
		norm, diff := 0.0, 0.0
		for i, v := range want {
			norm = max(norm, cmplx.Abs(v))
			diff = max(diff, cmplx.Abs(got[i]-v))
		}
		if e := diff / norm; !(e <= worst) {
			worst, at = e, omega[k]
		}
	}
	return worst, at
}

// Both explicit solvers are held to a refined-dense oracle on the well-scaled
// twin (the same transfer function), independent of which one the sweep tier
// picks; the user-facing sweep is also held to its own pointwise path.
func TestFreqResponseSweepMatchesPointwiseRandomized(t *testing.T) {
	const tol = 1e-12
	seeds := 400
	if testing.Short() {
		seeds = 60
	}
	failures := 0
	for seed := range seeds {
		for _, dt := range []float64{0, 0.05} {
			kind := sweepModelKind(seed % int(sweepModelKinds))
			rng := rand.New(rand.NewPCG(uint64(seed), uint64(kind)))
			sys, twin := randomSweepModel(rng, kind, dt, false)
			n, m, p := sys.Dims()
			omega := logspace(-3, 3, 60)
			if dt > 0 {
				omega = logspace(-3, math.Log10(math.Pi/dt), 60)
			}
			errHess, wHess := kernelErr(t, sys, twin, newHessenbergSweep(sys, n, m, p), omega, 5)
			errDense, wDense := kernelErr(t, sys, twin, newBalancedDense(sys, n, m, p), omega, 5)
			errPoint, wPoint := sweepPointwiseErr(t, sys, omega)
			if errHess <= tol && errDense <= tol && errPoint <= tol {
				continue
			}
			t.Errorf("seed=%d kind=%v dt=%g n=%d m=%d p=%d: hessenberg %g at ω=%g, dense %g at ω=%g, sweep vs pointwise %g at ω=%g",
				seed, kind, dt, n, m, p, errHess, wHess, errDense, wDense, errPoint, wPoint)
			if failures++; failures >= 10 {
				t.FailNow()
			}
		}
	}
}

// Descriptor sweeps are held to the refined oracle on the well-scaled
// descriptor twin; EvalFr must agree with the sweep bit for bit.
func TestFreqResponseDescriptorSweepRandomized(t *testing.T) {
	const tol = 1e-12
	seeds := 240
	if testing.Short() {
		seeds = 36
	}
	failures := 0
	for seed := range seeds {
		for _, dt := range []float64{0, 0.05} {
			kind := sweepModelKind(seed % int(sweepModelKinds))
			rng := rand.New(rand.NewPCG(uint64(seed), uint64(kind)))
			sys, twin := randomSweepModel(rng, kind, dt, true)
			n, m, p := sys.Dims()
			omega := logspace(-3, 3, 60)
			if dt > 0 {
				omega = logspace(-3, math.Log10(math.Pi/dt), 60)
			}
			got, err := sys.FreqResponse(omega)
			if err != nil {
				t.Errorf("seed=%d kind=%v dt=%g n=%d: %v", seed, kind, dt, n, err)
				if failures++; failures >= 10 {
					t.FailNow()
				}
				continue
			}
			td := newTimeDomain(dt)
			worst, at := 0.0, 0.0
			for k := 0; k < len(omega); k += 5 {
				s := td.frequencyVariable(omega[k])
				want := oracleResponse(t, twin, s, n, m, p)
				g, err := sys.EvalFr(s)
				if err != nil {
					t.Fatalf("seed=%d EvalFr: %v", seed, err)
				}
				norm, diff := 0.0, 0.0
				for i, v := range want {
					norm = max(norm, cmplx.Abs(v))
					diff = max(diff, cmplx.Abs(got.Data[k*p*m+i]-v))
					if g[i/m][i%m] != got.Data[k*p*m+i] {
						t.Fatalf("seed=%d ω=%g: EvalFr %v != sweep %v", seed, omega[k], g[i/m][i%m], got.Data[k*p*m+i])
					}
				}
				if e := diff / norm; !(e <= worst) {
					worst, at = e, omega[k]
				}
			}
			if worst <= tol {
				continue
			}
			t.Errorf("seed=%d kind=%v dt=%g n=%d m=%d p=%d: descriptor sweep %g at ω=%g", seed, kind, dt, n, m, p, worst, at)
			if failures++; failures >= 10 {
				t.FailNow()
			}
		}
	}
}

// loadSeriesHinf loads the 8-state Series(controller, plant) loop from Process
// Lab's H∞ feedthrough design whose sweep lost 1.7e-7 in v1.13.0.
func loadSeriesHinf(t *testing.T) (*System, []float64) {
	t.Helper()
	var d struct {
		A, B, C, D [][]float64
		Omega      []float64
	}
	raw, err := os.ReadFile("testdata/sweep_accuracy_series_hinf.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	dense := func(rows [][]float64) *mat.Dense {
		out := mat.NewDense(len(rows), len(rows[0]), nil)
		for i, r := range rows {
			out.SetRow(i, r)
		}
		return out
	}
	sys, err := New(dense(d.A), dense(d.B), dense(d.C), dense(d.D), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys, d.Omega
}

func TestFreqResponseSweepSeriesHinfLoop(t *testing.T) {
	sys, omega := loadSeriesHinf(t)
	n, m, p := sys.Dims()
	// Both sit in GEPP's componentwise accuracy class: 3.5e-13 (dense) and
	// 6.4e-13 (refined Hessenberg) here, against 1.8e-7 before the fix.
	for _, k := range []struct {
		name   string
		solver frequencyPointSolver
		tol    float64
	}{
		{"hessenberg", newHessenbergSweep(sys, n, m, p), 1e-12},
		{"dense", newBalancedDense(sys, n, m, p), 1e-12},
	} {
		if e, w := kernelErr(t, sys, sys, k.solver, omega, 1); e > k.tol {
			t.Errorf("%s: vs oracle %g at ω=%g", k.name, e, w)
		}
	}

	// G(jω₀) from exact rational (Fraction) GEPP on the float64 data, rounded
	// to float64. Compared normwise: G[1,0] cancels by ~2e4 across states and
	// even dense GEPP is 7.4e-12 off on it entrywise.
	exact := []complex128{
		complex(1.949644754006485, -19.767718992792179), complex(-0.0044800914509772333, 0.084011632661665947),
		complex(0.0069114303337741911, -0.02451652410746212), complex(3.2041001068660249, -28.27801811358389),
	}
	got, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	norm, diff := 0.0, 0.0
	for i, want := range exact {
		norm = max(norm, cmplx.Abs(want))
		diff = max(diff, cmplx.Abs(got.Data[i]-want))
	}
	if diff > 1e-13*norm {
		t.Errorf("G(jω₀) vs exact: normwise rel err %g", diff/norm)
	}
	if e, w := sweepPointwiseErr(t, sys, omega); e > 1e-12 {
		t.Errorf("sweep vs pointwise: rel err %g at ω=%g", e, w)
	}
}

// oracleResponse evaluates G(s) on the unbalanced realization by dense GEPP
// with ten steps of mixed-precision (Dot2 residual) iterative refinement,
// independent of both production solvers; it converges to rounding even
// where GEPP alone is off by 1e-5 (checked against exact rational arithmetic).
func oracleResponse(t *testing.T, sys *System, s complex128, n, m, p int) []complex128 {
	t.Helper()
	a := sys.A.RawMatrix()
	b := sys.B.RawMatrix()
	var e []float64
	var eStride int
	if sys.E != nil {
		raw := sys.E.RawMatrix()
		e, eStride = raw.Data, raw.Stride
	}
	pencil := make([]complex128, n*n)
	x := make([]complex128, n*m)
	d := make([]complex128, n*m)
	fillComplexPencil(pencil, a.Data, a.Stride, e, eStride, s, n)
	copyRealMatrixToComplex(x, b.Data, b.Stride, n, m)
	if err := cSolveInPlace(pencil, x, n, m); err != nil {
		t.Fatal(err)
	}
	for range 10 {
		for i := range n {
			for j := range m {
				var re, im dot2
				re.add(b.Data[i*b.Stride+j])
				if e != nil {
					descriptorResidualRow(&re, &im, a.Data[i*a.Stride:i*a.Stride+n], e[i*eStride:i*eStride+n], x, j, m, s)
					d[i*m+j] = complex(re.value(), im.value())
					continue
				}
				for k := range n {
					v := -a.Data[i*a.Stride+k]
					if k == i {
						dk := s + complex(v, 0)
						re.addProd(-real(dk), real(x[k*m+j]))
						re.addProd(imag(dk), imag(x[k*m+j]))
						im.addProd(-real(dk), imag(x[k*m+j]))
						im.addProd(-imag(dk), real(x[k*m+j]))
						continue
					}
					re.addProd(-v, real(x[k*m+j]))
					im.addProd(-v, imag(x[k*m+j]))
				}
				d[i*m+j] = complex(re.value(), im.value())
			}
		}
		fillComplexPencil(pencil, a.Data, a.Stride, e, eStride, s, n)
		if err := cSolveInPlace(pencil, d, n, m); err != nil {
			t.Fatal(err)
		}
		for i := range x {
			x[i] += d[i]
		}
	}
	c := sys.C.RawMatrix()
	out := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			var re, im dot2
			if sys.D != nil {
				re.add(sys.D.At(i, j))
			}
			for k := range n {
				re.addProd(c.Data[i*c.Stride+k], real(x[k*m+j]))
				im.addProd(c.Data[i*c.Stride+k], imag(x[k*m+j]))
			}
			out[i*m+j] = complex(re.value(), im.value())
		}
	}
	return out
}

// descriptorResidualRow adds -(sE-A)ᵢx[:,j] to (re, im). (Ex)ᵢ is formed
// in double-double before the product with s so that sE-A is never rounded.
func descriptorResidualRow(re, im *dot2, aRow, eRow []float64, x []complex128, j, m int, s complex128) {
	var exRe, exIm dot2
	for k, ek := range eRow {
		xk := x[k*m+j]
		exRe.addProd(ek, real(xk))
		exIm.addProd(ek, imag(xk))
		re.addProd(aRow[k], real(xk))
		im.addProd(aRow[k], imag(xk))
	}
	sr, si := real(s), imag(s)
	for _, v := range [2]float64{exRe.hi, exRe.lo} {
		re.addProd(-sr, v)
		im.addProd(-si, v)
	}
	for _, v := range [2]float64{exIm.hi, exIm.lo} {
		re.addProd(si, v)
		im.addProd(-sr, v)
	}
}

// dot2 accumulates a sum of products in twice the working precision
// (Ogita, Rump & Oishi 2005).
type dot2 struct{ hi, lo float64 }

func (d *dot2) add(v float64) {
	t := d.hi + v
	z := t - d.hi
	d.lo += (d.hi - (t - z)) + (v - z)
	d.hi = t
}

func (d *dot2) addProd(a, b float64) {
	// The conversion forbids fusing a*b into the following add, which
	// would break the error-free transformation (Go fuses on arm64).
	p := float64(a * b)
	d.lo += math.FMA(a, b, -p)
	d.add(p)
}

func (d *dot2) value() float64 { return d.hi + d.lo }

// A package variable defeats constant folding, which would hide fusion.
var dot2FusionProbe = []float64{1 + 0x1p-27, -(1 + 0x1p-26)}

func TestDot2AddProdErrorFree(t *testing.T) {
	a, s := dot2FusionProbe[0], dot2FusionProbe[1]
	var d dot2
	d.add(s)
	d.addProd(a, a)
	if got := d.value(); got != 0x1p-54 {
		t.Fatalf("dot2 value = %g, want %g; a*b fused into the TwoSum add", got, 0x1p-54)
	}
}
