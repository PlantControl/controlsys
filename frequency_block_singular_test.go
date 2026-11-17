package controlsys

import (
	"errors"
	"math/rand/v2"
	"testing"
)

// cycleLaplacianModel is the order-(k+1) model whose first k states form
// −L, L the Laplacian of a k-cycle (an exact eigenvalue at 0, irreducible),
// and whose last state is a decoupled stable pole.
func cycleLaplacianModel(t *testing.T, k int) *System {
	t.Helper()
	n := k + 1
	a := make([]float64, n*n)
	for i := range k {
		a[i*n+i] = -2
		a[i*n+(i+1)%k] = 1
		a[i*n+(i+k-1)%k] = 1
	}
	a[k*n+k] = -1
	b, c := make([]float64, n), make([]float64, n)
	for i := range n {
		b[i], c[i] = 1, 1
	}
	b[1] = 0.5
	c[2] = -0.25
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// FKYKOG: at ω = 1e-18 beside an exact pole at 0 of an irreducible 16-state
// block, η ≫ 1, no tier certifies and the order-17 model has no exact
// response; deciding that the pencil is not exactly singular must not take
// the fraction-free solve (≈7 ms a point): the modular filter proves the
// block nonsingular.
func TestFreqResponseUndeterminedBesidePoleModularFilter(t *testing.T) {
	sys := cycleLaplacianModel(t, 16)
	const w = 1e-18
	pt := newTimeDomain(0).frequencyPoint(w)
	for _, blk := range [][]int{seqInts(16), {16}} {
		k := len(blk)
		a := make([]float64, k*k)
		for i, bi := range blk {
			for j, bj := range blk {
				a[i*k+j] = sys.A.At(bi, bj)
			}
		}
		ps := &poleSchurData{n: k, m: 1, p: 1, a: a, b: unitVec(k), c: unitVec(k), d: []float64{0}}
		ip, _, _, _, _ := exactPencilSystem(ps, pt)
		if !exactNonsingularModP(ip, 2*k, 2*k+1) {
			t.Errorf("block of order %d: modular filter did not decide it nonsingular", k)
		}
	}
	if blockSingularAt(sys, pt) {
		t.Fatal("blockSingularAt: pencil reported singular")
	}
	_, err := sys.FreqResponse([]float64{w})
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("FreqResponse at ω = %g: err = %v, want ErrAccuracyUnverified", w, err)
	}
}

// At an exact pole (λ = i of a rotation block, λ = 0 of the Laplacian) the
// filter must not claim nonsingular, and blockSingularAt still finds it.
func TestBlockSingularAtExactPoleModularFilter(t *testing.T) {
	sys := cycleLaplacianModel(t, 16)
	pt := newTimeDomain(0).frequencyPoint(0)
	if !blockSingularAt(sys, pt) {
		t.Error("Laplacian block at s = 0: not reported singular")
	}
	n := 18
	a := make([]float64, n*n)
	for i := range n {
		a[i*n+i] = -1
	}
	a[0], a[1], a[n], a[n+1] = 0, 1, -1, 0
	for i := 2; i < n; i++ {
		a[i*n+i-1] = 0.5
	}
	b, c := make([]float64, n), make([]float64, n)
	b[0], c[n-1] = 1, 1
	rot, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	pt = newTimeDomain(0).frequencyPoint(1)
	ps := &poleSchurData{n: 2, m: 1, p: 1, a: []float64{0, 1, -1, 0}, b: unitVec(2), c: unitVec(2), d: []float64{0}}
	ip, _, _, _, _ := exactPencilSystem(ps, pt)
	if exactNonsingularModP(ip, 4, 5) {
		t.Error("rotation block at s = i: modular filter claims nonsingular")
	}
	if !blockSingularAt(rot, pt) {
		t.Error("rotation block at s = i: not reported singular")
	}
}

// The modular filter only ever proves what the fraction-free solve finds:
// on small integer pencils, singular or not, a nonsingular verdict implies
// solveExactAt succeeds, and every singular pencil is left undecided.
func TestExactNonsingularModPAgreesWithBareiss(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 17))
	decided := 0
	for trial := range 300 {
		k := 1 + rng.IntN(6)
		a := make([]float64, k*k)
		for i := range a {
			a[i] = float64(rng.IntN(7) - 3)
		}
		if trial%3 == 0 && k > 1 {
			for j := range k {
				a[(k-1)*k+j] = a[j] - a[k+j]
			}
		}
		var pt frequencyPoint
		switch trial % 4 {
		case 0:
			pt = frequencyPoint{p: 0, q: -1}
		case 1:
			pt = frequencyPoint{p: complex(0, float64(rng.IntN(3))), q: -1}
		default:
			pt = newTimeDomain(0.1).frequencyPoint(rng.Float64() * 30)
		}
		ps := &poleSchurData{n: k, m: 1, p: 1, a: a, b: unitVec(k), c: unitVec(k), d: []float64{0}}
		ip, _, _, _, _ := exactPencilSystem(ps, pt)
		mod := exactNonsingularModP(ip, 2*k, 2*k+1)
		_, _, ok := solveExactAt(ps, pt)
		if mod && !ok {
			t.Fatalf("trial %d: filter nonsingular, Bareiss singular (a = %v, pt = %v)", trial, a, pt)
		}
		if mod {
			decided++
		}
	}
	if decided == 0 {
		t.Fatal("filter decided no pencil")
	}
}

func seqInts(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

func unitVec(n int) []float64 {
	v := make([]float64, n)
	v[0] = 1
	return v
}
