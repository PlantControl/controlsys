package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func randomOrthogonal(rng *rand.Rand, n int) *mat.Dense {
	g := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(g)
	var q mat.Dense
	qr.QTo(&q)
	return &q
}

// illConditionedEDescriptor returns A = U·F·V, E = U·W·V with F upper
// triangular, W diagonal with last entry k·ε, U and V orthogonal, and the
// structured oracle (F, W, b, c, D) whose response is exact in float64.
func illConditionedEDescriptor(t *testing.T, rng *rand.Rand, n, m, p int, k, dt float64) (sys, oracle *System) {
	t.Helper()
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		F.Set(i, i, rng.NormFloat64()-0.3)
		W.Set(i, i, 0.5+rng.Float64())
		for j := i + 1; j < n; j++ {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	W.Set(n-1, n-1, k*eps())
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64())
		}
	}
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, dt); err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, dt); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

// TestTransferFunctionBadlyScaledDiagonalEStaysExact: a diagonal E spanning
// 1e-6..1e6 has cond(E) far above the stiff threshold, but E⁻¹A by LU is
// exact row scaling. The normwise SVD/QZ route of the ill-conditioned-E
// path would perturb the small entries by ε·‖E‖ (response error ~1e-9), so
// TransferFunction must judge cond(E) after equilibration (QKYC6R).
// Oracle: the pencil response of the same data.
func TestTransferFunctionBadlyScaledDiagonalEStaysExact(t *testing.T) {
	pts := []complex128{0.05i, 0.37i, complex(0.5, 3), 10i, 100i}
	for seed := range uint64(600) {
		rng := rand.New(rand.NewPCG(seed, 77))
		n := 2 + rng.IntN(5)
		A, E := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
		B, C := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		D := mat.NewDense(1, 1, []float64{rng.NormFloat64()})
		for i := range n {
			for j := range n {
				A.Set(i, j, rng.NormFloat64())
			}
			B.Set(i, 0, rng.NormFloat64())
			C.Set(0, i, rng.NormFloat64())
		}
		for i := range n {
			E.Set(i, i, math.Pow(10, -6+12*rng.Float64()))
		}
		sys, err := NewDescriptor(A, B, C, D, E, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		worst := 0.0
		for _, s := range pts {
			want := pencilResponse(t, sys, s)[0][0]
			got, err := res.TF.Eval(s)
			if err != nil {
				t.Fatalf("seed %d: Eval: %v", seed, err)
			}
			worst = math.Max(worst, cmplx.Abs(got[0][0]-want)/math.Max(1, cmplx.Abs(want)))
		}
		if !(worst < 3e-10) {
			t.Errorf("seed %d (n=%d): TF response error %.3g", seed, n, worst)
		}
	}
}

// TestTransferFunctionIllConditionedEMatchesPencil: a full-rank E with
// cond(E) ≈ 1e4..1e15 put the stiff direction's cond(E)·ε error of the LU
// E⁻¹A into every row, and the explicit ss→tf then lost the slow modes:
// TF.Eval was 1.7 relative off (a pole dropped at k = 2) while
// FreqResponse was exact (QKYC6R). Oracle: the response of the structured
// pencil (F, W) and its full pole count.
func TestTransferFunctionIllConditionedEMatchesPencil(t *testing.T) {
	ks := []float64{10, 30, 1e3, 1e5, 1e8, 1e11}
	pts := []complex128{0.05i, 0.37i, complex(0.5, 3), 10i}
	for seed := range uint64(120) {
		rng := rand.New(rand.NewPCG(seed, 33))
		n := 2 + rng.IntN(4)
		m, p := 1, 1
		if seed%4 == 3 {
			m, p = 1+rng.IntN(2), 1+rng.IntN(2)
		}
		dt := 0.0
		if seed%5 == 4 {
			dt = 0.1
		}
		k := ks[rng.IntN(len(ks))]
		sys, oracle := illConditionedEDescriptor(t, rng, n, m, p, k, dt)
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		worst := 0.0
		for _, s := range pts {
			want := pencilResponse(t, oracle, s)
			got, err := res.TF.Eval(s)
			if err != nil {
				t.Fatalf("seed %d: Eval: %v", seed, err)
			}
			for i := range want {
				for j := range want[i] {
					worst = math.Max(worst, cmplx.Abs(got[i][j]-want[i][j])/math.Max(1, cmplx.Abs(want[i][j])))
				}
			}
		}
		if !(worst < 1e-8) {
			t.Errorf("seed %d (n=%d m=%d p=%d k=%g dt=%g): TF response error %.3g", seed, n, m, p, k, dt, worst)
		}
		if m == 1 && p == 1 {
			if deg := len(res.TF.Den[0]) - 1; deg != n {
				t.Errorf("seed %d (n=%d k=%g): denominator degree %d, want %d", seed, n, k, deg, n)
			}
		}
	}
}
