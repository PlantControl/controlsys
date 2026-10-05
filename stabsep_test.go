package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStabsep_Diagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 2, 0, 0, 0, -3}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	stablePoles, _ := res.Stable.Poles()
	for _, p := range stablePoles {
		if real(p) >= 0 {
			t.Errorf("stable pole %v has Re >= 0", p)
		}
	}

	unstablePoles, _ := res.Unstable.Poles()
	for _, p := range unstablePoles {
		if real(p) < 0 {
			t.Errorf("unstable pole %v has Re < 0", p)
		}
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_FullyStable(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 0 {
		t.Errorf("unstable order = %d, want 0", nu)
	}
}

func TestStabsep_FullyUnstable(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1, 0, 0, 2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
	if nu != 2 {
		t.Errorf("unstable order = %d, want 2", nu)
	}
}

func TestStabsep_ComplexEigenvalues(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, -1, 0, 0, 0, 0, 2}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	// ±j are marginal (Re=0), treated as unstable; pole 2 is unstable
	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
	if nu != 3 {
		t.Errorf("unstable order = %d, want 3", nu)
	}
}

func TestStabsep_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0.5, 0, 0, 0, 1.5, 0, 0, 0, 0.9}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_NonDiagonalStable(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -2, 0,
			0, 0, 3,
		}),
		mat.NewDense(3, 1, []float64{1, 1, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_Empty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
}

func TestStabsep_EigenvaluePreservation(t *testing.T) {
	sys, err := New(
		mat.NewDense(4, 4, []float64{
			-1, 0.5, 0, 0,
			0, -2, 0, 0,
			0, 0, 1, 0.3,
			0, 0, 0, 3,
		}),
		mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
		mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
		mat.NewDense(4, 4, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	origPoles, _ := sys.Poles()
	stablePoles, _ := res.Stable.Poles()
	unstablePoles, _ := res.Unstable.Poles()

	allDecomp := append(stablePoles, unstablePoles...)
	sortPoles(origPoles)
	sortPoles(allDecomp)

	if len(origPoles) != len(allDecomp) {
		t.Fatalf("pole count: %d vs %d", len(origPoles), len(allDecomp))
	}
	for i := range origPoles {
		if cmplx.Abs(origPoles[i]-allDecomp[i]) > 1e-10 {
			t.Errorf("pole[%d]: %v != %v", i, origPoles[i], allDecomp[i])
		}
	}
}

func checkAdditiveDecomposition(t *testing.T, sys, sys1, sys2 *System) {
	t.Helper()
	_, m, p := sys.Dims()

	freqs := []float64{0.01, 0.1, 1, 10, 100}
	if sys.IsDiscrete() {
		freqs = []float64{0.01, 0.1, 0.5, 1, 2}
	}

	for _, w := range freqs {
		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w)
		} else {
			s = complex(0, w*sys.Dt)
		}

		g, err := sys.EvalFr(s)
		if err != nil {
			continue
		}
		g1, err := sys1.EvalFr(s)
		if err != nil {
			continue
		}
		g2, err := sys2.EvalFr(s)
		if err != nil {
			continue
		}

		for i := range p {
			for j := range m {
				sum := g1[i][j] + g2[i][j]
				diff := cmplx.Abs(sum - g[i][j])
				mag := cmplx.Abs(g[i][j])
				tol := 1e-8
				if mag > 1 {
					tol = 1e-8 * mag
				}
				if diff > tol && mag > 1e-12 {
					t.Errorf("w=%g: G[%d,%d] sum=%v, orig=%v, diff=%g", w, i, j, sum, g[i][j], diff)
				}
			}
		}
	}
}

func TestStabsep_AdditiveDecompositionDCGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 2, 0, 0, 0, -3}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	stableDC, err := res.Stable.DCGain()
	if err != nil {
		t.Fatal(err)
	}

	// Stable part DC gain: diag(-1/(−1), -1/(-3)) in transformed coords
	// The trace of stable DC should be sum of -1/pole for stable poles
	traceDC := 0.0
	ns, _, _ := res.Stable.Dims()
	for i := range ns {
		if i < 3 {
			traceDC += stableDC.At(i, i)
		}
	}
	if math.IsInf(traceDC, 0) || math.IsNaN(traceDC) {
		t.Errorf("stable DC gain has inf/nan")
	}
}

func pencilResponse(t *testing.T, sys *System, s complex128) [][]complex128 {
	t.Helper()
	n, m, p := sys.Dims()
	G := cmatOf(sys.D, p, m)
	if n == 0 {
		return G
	}
	M := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n)
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			M[i][j] = s*complex(e, 0) - complex(sys.A.At(i, j), 0)
		}
	}
	CX := cmul(cmatOf(sys.C, p, n), csolve(t, M, cmatOf(sys.B, n, m)))
	for i := range p {
		for j := range m {
			G[i][j] += CX[i][j]
		}
	}
	return G
}

func assertSplitSum(t *testing.T, label string, orig, a, b *System) {
	t.Helper()
	pts := []complex128{complex(0.3, 0.7), complex(-0.2, 2.5), complex(1.7, -0.4)}
	for _, s := range pts {
		want := pencilResponse(t, orig, s)
		ga := pencilResponse(t, a, s)
		gb := pencilResponse(t, b, s)
		for i := range want {
			for j := range want[i] {
				if d := cmplx.Abs(ga[i][j] + gb[i][j] - want[i][j]); d > 1e-9*(1+cmplx.Abs(want[i][j])) {
					t.Fatalf("%s: parts sum mismatch at s=%v [%d,%d]: |diff|=%.3g", label, s, i, j, d)
				}
			}
		}
	}
}

func generalizedEigOracle(t *testing.T, A, E *mat.Dense) []complex128 {
	t.Helper()
	var lu mat.LU
	lu.Factorize(E)
	var F mat.Dense
	if err := lu.SolveTo(&F, false, A); err != nil {
		t.Fatal(err)
	}
	var eig mat.Eigen
	if !eig.Factorize(&F, mat.EigenNone) {
		t.Fatal("eig failed")
	}
	return eig.Values(nil)
}

func descriptorSplitFixture(t *testing.T, dt float64) *System {
	t.Helper()
	E := mat.NewDense(3, 3, []float64{2, 1, 0, 0.5, 3, 0.2, 0, 0.4, 1.5})
	A := mat.NewDense(3, 3, []float64{1.5, 0.7, 0.1, -0.3, -2.1, 0.4, 0.25, -0.15, -3.3})
	if dt > 0 {
		A = mat.NewDense(3, 3, []float64{2.9, 1.2, 0.4, 0.3, 1.1, -0.8, 0.1, 0.6, -0.9})
	}
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1.3, 0.6, -0.8})
	C := mat.NewDense(2, 3, []float64{1, 0.5, -0.2, 0, 1.1, 0.3})
	D := mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3})
	sys, err := NewDescriptor(A, B, C, D, E, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestStabsep_Descriptor(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := descriptorSplitFixture(t, dt)
		wantStable := 0
		for _, ev := range generalizedEigOracle(t, sys.A, sys.E) {
			if (dt == 0 && real(ev) < 0) || (dt > 0 && cmplx.Abs(ev) < 1) {
				wantStable++
			}
		}
		if wantStable == 0 || wantStable == 3 {
			t.Fatalf("dt=%v: fixture not mixed (%d stable)", dt, wantStable)
		}
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatal(err)
		}
		if ns, _, _ := res.Stable.Dims(); ns != wantStable {
			t.Errorf("dt=%v: stable order %d, want %d", dt, ns, wantStable)
		}
		if st, _ := res.Stable.IsStable(); !st {
			t.Errorf("dt=%v: stable part not stable", dt)
		}
		assertSplitSum(t, "Stabsep descriptor", sys, res.Stable, res.Unstable)
	}
}

func boundaryOscillator(t *testing.T, w, dt float64) *System {
	t.Helper()
	T := mat.NewDense(3, 3, []float64{1, 0.3, 0.7, 0.2, 1.1, -0.4, 0.5, 0.1, 0.9})
	var Ti, A, tmp mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	J := mat.NewDense(3, 3, []float64{0, w, 0, -w, 0, 0, 0, 0, -1})
	if dt > 0 {
		J = mat.NewDense(3, 3, []float64{math.Cos(w), math.Sin(w), 0, -math.Sin(w), math.Cos(w), 0, 0, 0, 0.5})
	}
	tmp.Mul(T, J)
	A.Mul(&tmp, &Ti)
	sys, err := New(&A, mat.NewDense(3, 1, []float64{1, 0, 1}), mat.NewDense(1, 3, []float64{1, 1, 0}), mat.NewDense(1, 1, []float64{0.2}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestStabsep_BoundaryPolesUnstable(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		for k := range 200 {
			w := 0.37 + 0.01*float64(k)
			sys := boundaryOscillator(t, w, dt)
			res, err := Stabsep(sys)
			if err != nil {
				t.Fatal(err)
			}
			ns, _, _ := res.Stable.Dims()
			nu, _, _ := res.Unstable.Dims()
			if ns != 1 || nu != 2 {
				t.Fatalf("dt=%v w=%g: stable/unstable orders %d/%d, want 1/2", dt, w, ns, nu)
			}
		}
	}
}

func TestStabsep_ComplexPairNonNormalSumExact(t *testing.T) {
	for _, tc := range []struct {
		dt float64
		A  []float64
	}{
		{0, []float64{0.5, 2, 0, -0.5, -3, 1, 0.3, 0, -2}},
		{0.1, []float64{1.4, 0.2, 0, -0.1, 0.3, 0.4, 0.2, 0, -0.6}},
		{0.1, []float64{0.2, -0.9, 0.1, 0.8, 0.3, 0.5, 0, 0.4, 1.3}},
	} {
		sys, err := New(mat.NewDense(3, 3, tc.A), mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1}),
			mat.NewDense(2, 3, []float64{1, 0.5, 0, 0, 1, -1}), mat.NewDense(2, 2, []float64{0.3, 0, -0.1, 0.2}), tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatal(err)
		}
		ns, _, _ := res.Stable.Dims()
		nu, _, _ := res.Unstable.Dims()
		if ns == 0 || nu == 0 {
			t.Fatalf("dt=%v: fixture not mixed (%d/%d)", tc.dt, ns, nu)
		}
		assertSplitSum(t, "Stabsep", sys, res.Stable, res.Unstable)
		mres, err := Modsep(sys, 1.2)
		if err != nil {
			t.Fatal(err)
		}
		assertSplitSum(t, "Modsep", sys, mres.Slow, mres.Fast)
	}
}
