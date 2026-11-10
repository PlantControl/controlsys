package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// roundedStiffIntegrator draws A = T·diag(0, −s₁, …, −stiff)·T⁻¹ for a
// dense Gaussian T, n in [3, 5], slow modes s_k within three decades above
// max(1, 1e-9·stiff), and returns the model with the intended residue
// (C·T e₀)(e₀ᵀ T⁻¹B) of the integrator per entry.
func roundedStiffIntegrator(t *testing.T, seed uint64, stiff float64, m, p int) (*System, [][]float64) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 4517))
	n := 3 + rng.IntN(3)
	T := mat.NewDense(n, n, nil)
	for i := range n * n {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	lo := math.Max(0, math.Log10(stiff)-9)
	J := mat.NewDense(n, n, nil)
	for k := 1; k < n-1; k++ {
		J.Set(k, k, -math.Pow(10, lo+3*rng.Float64()))
	}
	J.Set(n-1, n-1, -stiff)
	var tmp, A mat.Dense
	tmp.Mul(T, J)
	A.Mul(&tmp, &Ti)
	B, C, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range n * m {
		B.RawMatrix().Data[i] = rng.NormFloat64()
	}
	for i := range p * n {
		C.RawMatrix().Data[i] = rng.NormFloat64()
	}
	for i := range p * m {
		D.RawMatrix().Data[i] = rng.NormFloat64()
	}
	sys, err := New(&A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res := make([][]float64, p)
	for i := range p {
		res[i] = make([]float64, m)
		ct := mat.Dot(C.RowView(i), T.ColView(0))
		for j := range m {
			res[i][j] = ct * mat.Dot(Ti.RowView(0), B.ColView(j))
		}
	}
	return sys, res
}

// A dense similarity of diag(0, −1e4, −1e12) rounds the integrator to
// −2.8e-5, which componentwise rounding of A moves by more than its
// modulus. EvalFr(0) is +Inf; on base DCGain read the transfer limit,
// whose coefficient floor 100ε·max|coef| kept the rounded constant 4.1e11
// of the characteristic polynomial against a roundoff of order ε‖A‖³, and
// returned 15797.6 (W45MGH). The exact limit of the float data, 80439.7,
// is R/(−λ) for the rounded λ, which the data do not tell from 0, so the
// Inf comes with ErrAccuracyUnverified (N4GQHT).
func TestDCGainRoundedStiffIntegratorInf(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 2026))
	T := mat.NewDense(3, 3, nil)
	for i := range 9 {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti, tmp, A mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	tmp.Mul(T, mat.NewDense(3, 3, []float64{0, 0, 0, 0, -1e4, 0, 0, 0, -1e12}))
	A.Mul(&tmp, &Ti)
	b := mat.NewDense(3, 1, []float64{0.7, -1.2, 0.4})
	c := mat.NewDense(1, 3, []float64{1.1, 0.3, -0.8})
	sys, err := New(&A, b, c, mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	r := mat.Dot(c.RowView(0), T.ColView(0)) * mat.Dot(Ti.RowView(0), b.ColView(0))
	dc, err := sys.DCGain()
	requireDCGainUnverified(t, "rounded stiff integrator", dc, err)
	if want := math.Inf(signInt(r)); dc.At(0, 0) != want {
		t.Errorf("DCGain = %v, want %v", dc.At(0, 0), want)
	}
	ev, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	if !cmplx.IsInf(ev[0][0]) {
		t.Errorf("EvalFr(0) = %v, want Inf", ev[0][0])
	}
}

// Rounded integrators beside a fast pole of 1e4 to 1e14: every entry is
// the signed Inf of the intended residue, SISO and MIMO, and EvalFr(0)
// reads the pole too. On base DCGain was finite (W45MGH). DCGain flags
// the Inf with ErrAccuracyUnverified, the float model's exact limit being
// finite (N4GQHT); integratingDCGain does not check it.
func TestDCGainRoundedStiffIntegratorFamily(t *testing.T) {
	for seed := range uint64(44) {
		stiff := math.Pow(10, 4+float64(seed%11))
		m, p := 1, 1
		if seed%4 == 3 {
			m, p = 2, 3
		}
		sys, res := roundedStiffIntegrator(t, seed, stiff, m, p)
		label := fmt.Sprintf("seed %d stiff %g", seed, stiff)
		dc, err := sys.DCGain()
		requireDCGainUnverified(t, label, dc, err)
		ig, err := sys.integratingDCGain()
		if err != nil {
			t.Fatalf("%s: integratingDCGain: %v", label, err)
		}
		ev, err := sys.EvalFr(0)
		if err != nil {
			t.Fatalf("%s: EvalFr(0): %v", label, err)
		}
		for i := range p {
			for j := range m {
				want := math.Inf(signInt(res[i][j]))
				if g := dc.At(i, j); g != want {
					t.Errorf("%s: DCGain[%d][%d] = %v, want %v", label, i, j, g, want)
				}
				if g := ig.At(i, j); g != want {
					t.Errorf("%s: integratingDCGain[%d][%d] = %v, want %v", label, i, j, g, want)
				}
				if !cmplx.IsInf(ev[i][j]) {
					t.Errorf("%s: EvalFr(0)[%d][%d] = %v, want Inf", label, i, j, ev[i][j])
				}
			}
		}
	}
}

// The rounded-integrator rule takes no pole the data locate: the resolved
// slow poles of the BLHSVK stiff families, and a triangular slow pole of
// 1e-5 below ε·‖A‖ beside −1e12, keep their finite gain, as does a channel
// the integrator does not reach.
func TestDCGainRoundedIntegratorKeepsResolvedPoles(t *testing.T) {
	for seed := range uint64(40) {
		sys := stiffSlowPoleExplicit(t, seed)
		if _, ok := sys.dcGainByRoundedIntegrator(); ok {
			t.Errorf("stiff seed %d: rounded integrator taken", seed)
		}
	}
	for _, seed := range []uint64{71, 109, 8, 50} {
		sys := stiffFoldTwinExplicit(t, seed)
		if _, ok := sys.dcGainByRoundedIntegrator(); ok {
			t.Errorf("fold twin seed %d: rounded integrator taken", seed)
		}
	}
	for _, a := range [][]float64{{-1e-5, 1, 0, -1e12}, {1e-5, 1, 0, -1e12}, {-1e-3, 3, 0, -1e14}} {
		sys, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, []float64{0.7, -1.2}), mat.NewDense(1, 2, []float64{1.1, 0.3}), mat.NewDense(1, 1, []float64{0.5}), 0)
		if err != nil {
			t.Fatal(err)
		}
		want := bigExplicitDCGain(t, sys)[0][0]
		for name, f := range map[string]func() (*mat.Dense, error){"DCGain": sys.DCGain, "integratingDCGain": sys.integratingDCGain} {
			g, err := f()
			if err != nil {
				t.Fatalf("%v: %s: %v", a, name, err)
			}
			if !(math.Abs(g.At(0, 0)-want) <= 1e-10*math.Abs(want)) {
				t.Errorf("%v: %s = %v, want %v", a, name, g.At(0, 0), want)
			}
		}
	}
	sys, _ := roundedStiffIntegrator(t, 3, 1e10, 1, 1)
	n, _, _ := sys.Dims()
	var eig mat.Eigen
	if !eig.Factorize(sys.A, mat.EigenRight) {
		t.Fatal("eigen")
	}
	var x mat.CDense
	eig.VectorsTo(&x)
	k := 0
	for i, v := range eig.Values(nil) {
		if cmplx.Abs(v) > cmplx.Abs(eig.Values(nil)[k]) {
			k = i
		}
	}
	B := mat.NewDense(n, 2, nil)
	for i := range n {
		B.Set(i, 0, real(x.At(i, k)))
		B.Set(i, 1, sys.B.At(i, 0))
	}
	two, err := New(sys.A, B, sys.C, mat.NewDense(1, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if g, ok := two.dcGainByRoundedIntegrator(); ok && math.IsInf(g.At(0, 0), 0) {
		t.Errorf("fast-mode input: rounded integrator gain %v, want the channel left to the transfer limit", g.At(0, 0))
	}
}
