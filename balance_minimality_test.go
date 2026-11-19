package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/bits"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// ratKalmanRank is the exact rank over ℚ of [B AB … Aⁿ⁻¹B] of the stored
// float64 data, by rational Gaussian elimination.
func ratKalmanRank(A, B *mat.Dense) int {
	n, m := B.Dims()
	rat := func(x float64) *big.Rat { return new(big.Rat).SetFloat64(x) }
	cur := make([][]*big.Rat, n)
	for i := range n {
		cur[i] = make([]*big.Rat, m)
		for j := range m {
			cur[i][j] = rat(B.At(i, j))
		}
	}
	K := make([][]*big.Rat, n)
	for range n {
		for i := range n {
			K[i] = append(K[i], cur[i]...)
		}
		next := make([][]*big.Rat, n)
		for i := range n {
			next[i] = make([]*big.Rat, m)
			for j := range m {
				s := new(big.Rat)
				for k := range n {
					if a := A.At(i, k); a != 0 && cur[k][j].Sign() != 0 {
						s.Add(s, new(big.Rat).Mul(rat(a), cur[k][j]))
					}
				}
				next[i][j] = s
			}
		}
		cur = next
	}
	rank := 0
	for c := 0; c < n*m && rank < n; c++ {
		piv := -1
		for i := rank; i < n; i++ {
			if K[i][c].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			continue
		}
		K[rank], K[piv] = K[piv], K[rank]
		for i := rank + 1; i < n; i++ {
			if K[i][c].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Quo(K[i][c], K[rank][c])
			for j := c; j < n*m; j++ {
				K[i][j].Sub(K[i][j], new(big.Rat).Mul(f, K[rank][j]))
			}
		}
		rank++
	}
	return rank
}

func ratMinimal(sys *System) bool {
	n, _, _ := sys.Dims()
	return ratKalmanRank(sys.A, sys.B) == n &&
		ratKalmanRank(mat.DenseCopyOf(sys.A.T()), mat.DenseCopyOf(sys.C.T())) == n
}

// TestBalrealMinimalityHadamardRotation: A = H·T·H/4 with H the 4×4
// Hadamard matrix is exact in float64, so the uncontrollable mode of T
// stays exactly uncontrollable after the rotation. The rounded
// controllability gramian is positive definite, so a Cholesky-based test
// returns nil.
func TestBalrealMinimalityHadamardRotation(t *testing.T) {
	T := []float64{
		-1, 0.5, 0, 0,
		-0.5, -1, 0, 0,
		0, 0, -2, 1,
		0, 0, 0, -0.5,
	}
	H := mat.NewDense(4, 4, []float64{1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1})
	var A mat.Dense
	A.Mul(H, mat.NewDense(4, 4, T))
	A.Mul(&A, H)
	A.Scale(0.25, &A)
	b := mat.NewVecDense(4, []float64{1, -0.5, 0.75, 0})
	var B mat.Dense
	B.Mul(H, b)
	B.Scale(0.5, &B)
	C := mat.NewDense(1, 4, []float64{0.5, 1, -0.25, 2})
	sys, err := New(mat.DenseCopyOf(&A), mat.DenseCopyOf(&B), C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if ratMinimal(sys) {
		t.Fatal("fixture is minimal")
	}
	if _, err := Balreal(sys); !errors.Is(err, ErrNotMinimal) {
		t.Fatalf("Balreal err = %v, want ErrNotMinimal", err)
	}
	if _, _, err := Balred(sys, 2, BalredOptions{}); !errors.Is(err, ErrNotMinimal) {
		t.Fatalf("Balred err = %v, want ErrNotMinimal", err)
	}
}

func TestExactMinimalityKrylov(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 2, 0.5, 0, -3, 1, 0, 0, -0x1p-40})
	for _, tc := range []struct {
		name string
		B    []float64
		want minimality
	}{
		{"minimal", []float64{0, 0, 1}, minimalExact},
		{"uncontrollable tail", []float64{1, 1, 0}, notMinimalExact},
		{"tiny last entry", []float64{1, 1, 0x1p-1000}, minimalExact},
	} {
		t.Run(tc.name, func(t *testing.T) {
			B := mat.NewDense(3, 1, tc.B)
			C := mat.NewDense(1, 3, []float64{1, 0.5, -0.25})
			got := exactMinimality(A, B, C)
			if got != tc.want {
				t.Fatalf("exactMinimality = %d, want %d", got, tc.want)
			}
			want := ratKalmanRank(A, B) == 3 && ratKalmanRank(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T())) == 3
			if (got == minimalExact) != want {
				t.Fatalf("exactMinimality = %d, rational rank minimal = %v", got, want)
			}
		})
	}
}

func balrealHSVWithin(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("HSV %v, want %v", got, want)
	}
	for i := range want {
		if d := math.Abs(got[i] - want[i]); d > tol*want[0] {
			t.Fatalf("HSV[%d] = %v, want %v (|Δ|/σ₁ = %.3g)", i, got[i], want[i], d/want[0])
		}
	}
}

// balrealCensusRows are frozen gonum-3407 census rows (O2EOLR), all with
// non-symmetric A; ref holds Hankel singular values from 320-bit gramians.
var balrealCensusRows = []struct {
	name       string
	n, m, p    int
	a, b, c, d []float64
	dt         float64
	ref        []float64
	want       error
}{
	{
		name: "Hadamard-rotated discrete exactly non-minimal", n: 4, m: 1, p: 1, dt: 0.5,
		a:    []float64{-0.375, -0.0625, 0.0625, 0.125, 0.0625, -0.625, 0.125, 0.1875, 0.0625, 0, -0.25, -0.0625, 0, -0.0625, 0.3125, -0.5},
		b:    []float64{0.5, 2, 1, 1},
		c:    []float64{-1, 0, 0.5, 0.5},
		d:    []float64{0},
		want: ErrNotMinimal,
	},
	{
		name: "permuted exactly non-minimal", n: 8, m: 1, p: 2,
		a: []float64{-0.5, 0.5, 1, 0.5, 1, 1, 0, 0.5, 0, -0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, -1, 0, -2, 0, -0.5, 0, 0, -1, 0, -1, 0, -0.5, -1, 1, -0.5, 0, 0, -0.5, 0.5, 0, -2, 0, 0, 0, 0, 0.5, -0.5, 0, 0, -1, 0, 0, -1, 0.5, 0, 0, 0, 0, -1},
		b: []float64{0, 2, 0.5, 0, -1, 1, 2, 1},
		c: []float64{2, 1, 2, 0.5, 2, 0, 0, -1, -1, 0.5, -1, 1, 0.5, 0.5, -1, 0},
		d: []float64{0, 0}, want: ErrNotMinimal,
	},
	{
		name: "rotated minimal with HSV below rounding", n: 8, m: 1, p: 2,
		a:   []float64{-0.7136615849928617, -0.6726377456902913, 0.08711373578612366, 0.009653546986575992, -0.05687439908800164, 0.2582789929090527, 0.8763306123414072, -1.0956058668426305, 0.2956158342898989, -1.5026484743112671, 0.6185213720676135, -0.30088906856363173, -0.11771605249537166, -0.21664834993418128, -0.1200852355658564, 0.3361581471582128, -0.18126723871772507, 0.679748609663744, -1.3298232494479316, 0.21288916366385233, 0.5659098211122744, 0.39611747403096204, 1.0472562971947537, -0.6073564948418204, 0.28250390666720226, 0.685934328533943, 0.6908569402331854, -1.256084106464082, -1.6772613088602983, 0.8754932609182505, 0.09732227301381653, 0.3218920317124821, 0.06413323774569799, 0.1281029429334822, -0.41245988114942683, -0.05411602022383696, -0.24806785153498653, -0.36581585618783663, 0.07722421538340832, -0.4544561687801383, -0.10864264062702096, 0.17257512531270508, -0.5242250138892433, -0.30779834222411856, 0.35405278289510084, -0.6853358726875095, 0.24613468772239414, 0.2536811409773781, -0.15038599071926578, -0.6545837067889329, 0.33534439170823754, -0.7414670355418297, -0.8415826805830006, 0.4242584106942894, -1.2346855061103321, -0.14198709346324392, 0.2438288275090407, 0.6553016149423757, 0.7690668968909954, -0.8190234184377136, -0.24456467820640007, -0.44208616338466883, -0.5836803867513715, -1.0296933544510298},
		b:   []float64{-1, 1, -1, -1, 1, 0, 0.5, 2},
		c:   []float64{2, 0.5, 1, 0.5, 2, 0, 0.5, 0, 0, -1, 2, 0, 2, 1, -1, 0},
		d:   []float64{2, 2},
		ref: []float64{3.0338888364157954, 1.6564897459186587, 0.6473631602399, 0.37410853489808504, 0.1480408110708804, 0.0129507773649628, 0.00220268018816738, 1.0669559595043593e-19},
	},
	{
		name: "lightly damped pair refined gramian", n: 6, m: 1, p: 1,
		a:   []float64{-1, 1, 0.5, 1, 1, 0.5, 0, -4.76837158203125e-07, 1, 0, 0, 1, 0, -1, -4.76837158203125e-07, -0.5, 0, 0, 0, 0, 0, -0.5, -0.5, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -9.313225746154785e-10},
		b:   []float64{2, 2, 0.5, -1, 0, 2},
		c:   []float64{-1, 2, 0.5, 1, 1, 1},
		d:   []float64{0},
		ref: []float64{2.5634548155293176e+06, 2.5634535030294615e+06, 510.00012124880374, 1.1748640979912757, 0.8007947395333875, 0.06343208356337678},
	},
	{
		name: "near-defective gramian beyond float64", n: 4, m: 1, p: 1,
		a:    []float64{-4.656612873077393e-10, 0, 0, -0.5, 0, -5.820766091346741e-11, -0.5, -0.5, 0, 0.5, -5.820766091346741e-11, 0.5, 0.5, 0, 0, -4.656612873077393e-10},
		b:    []float64{-1, 1, 2, 0.5},
		c:    []float64{-1, 2, 2, 0},
		d:    []float64{0},
		want: ErrSingularEquation,
	},
	{
		name: "rotated minimal with HSV rounding to zero", n: 2, m: 2, p: 2,
		a:   []float64{-1.0000000000000002, 1.071516862846046e-16, 1.071516862846046e-16, -1},
		b:   []float64{0, 0, -1, 1},
		c:   []float64{1, -1, 0.5, 2},
		d:   []float64{-1, -1, -1, 1},
		ref: []float64{1.5811388300841898, 4.2355422973396943e-17},
	},
}

func TestBalrealCensusRows(t *testing.T) {
	for _, tc := range balrealCensusRows {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			if minimal := ratMinimal(sys); minimal != (tc.want != ErrNotMinimal) {
				t.Fatalf("fixture exactly minimal = %v", minimal)
			}
			br, err := Balreal(sys)
			if tc.want != nil {
				if !errors.Is(err, tc.want) || br != nil {
					t.Fatalf("Balreal = %v, %v; want %v", br, err, tc.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("Balreal: %v", err)
			}
			balrealHSVWithin(t, br.HSV, tc.ref, 1e-8)
			red, hsv, err := Balred(sys, (tc.n+1)/2, BalredOptions{})
			if err != nil || red == nil {
				t.Fatalf("Balred: %v", err)
			}
			balrealHSVWithin(t, hsv, tc.ref, 1e-8)
		})
	}
}

func TestExactMinimalityBudgetPaths(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 2, 0.5, 0, -3, 1, 0, 0, -0x1p-40})
	a := newDyadicImage(A, false)
	for _, tc := range []struct {
		name   string
		b      []float64
		budget minimalityBudget
		want   minimality
	}{
		{"certificate", []float64{1, 1, 0}, defaultMinimalityBudget, notMinimalExact},
		{"Hadamard-bound proof", []float64{1, 1, 0}, minimalityBudget{certPrimes: -1, work: 1 << 28}, notMinimalExact},
		{"over work bound", []float64{1, 1, 0}, minimalityBudget{certPrimes: -1, work: 1}, minimalityUndetermined},
		{"minimal within any budget", []float64{0, 0, 1}, minimalityBudget{certPrimes: -1, work: 1}, minimalExact},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newDyadicImage(mat.NewDense(3, 1, tc.b), false)
			if got := exactKrylovFullRank(a, b, tc.budget); got != tc.want {
				t.Fatalf("exactKrylovFullRank = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestBalrealMinimalityLarge: exactly non-minimal models of order 64 with
// dense non-symmetric A, one structurally uncontrollable and one whose
// uncontrollable subspace is a Hadamard rotation of a coordinate subspace;
// the Hadamard-bound proof alone would exceed its work bound.
func TestBalrealMinimalityLarge(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewPCG(60, 7))
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, rng.NormFloat64()/n)
		}
		A.Set(i, i, A.At(i, i)-2)
	}
	B := mat.NewDense(n, 1, nil)
	C := mat.NewDense(1, n, nil)
	for i := range n {
		B.Set(i, 0, rng.NormFloat64())
		C.Set(0, i, rng.NormFloat64())
	}
	for j := range n - 1 {
		A.Set(n-1, j, 0)
	}
	B.Set(n-1, 0, 0)
	structural, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	H := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			H.Set(i, j, float64(1-2*(bits.OnesCount(uint(i&j))&1))/8)
		}
	}
	T := mat.NewDense(n, n, nil)
	Bt := mat.NewDense(n, 1, nil)
	for i := range n {
		T.Set(i, i, -1-float64(i%5)/4)
		if i+1 < n && i != n/2-1 {
			T.Set(i, i+1, 0.5)
		}
		if i < n/2 {
			Bt.Set(i, 0, float64(i%3)-1)
		}
	}
	for i := range n / 2 {
		for j := n / 2; j < n; j++ {
			T.Set(i, j, float64((i+j)%4)/8)
		}
	}
	var Ah, Bh mat.Dense
	Ah.Mul(H, T)
	Ah.Mul(&Ah, H)
	Bh.Mul(H, Bt)
	rotated, err := New(mat.DenseCopyOf(&Ah), mat.DenseCopyOf(&Bh), C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"structural": structural, "Hadamard": rotated} {
		if _, err := Balreal(sys); !errors.Is(err, ErrNotMinimal) {
			t.Errorf("%s: Balreal err = %v, want ErrNotMinimal", name, err)
		}
	}
}
