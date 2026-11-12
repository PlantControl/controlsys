package controlsys

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// symplecticUnitCircleRoot reports whether the exact symplectic pencil
// z·[I G; 0 Āᵀ] − [Ā 0; −Q̄ I] of Ā = A − BR⁻¹Sᵀ, G = BR⁻¹Bᵀ,
// Q̄ = Q − SR⁻¹Sᵀ (reduced, unlike the extended pencil Dare's proof uses)
// has a root on the unit circle: at z = −1, or a root of
// P(s) = (1−s)^2n·p((1+s)/(1−s)) on the imaginary axis (evenPolyAxisRoot).
func symplecticUnitCircleRoot(t *testing.T, n, m int, a, b, q, r, s []float64) bool {
	t.Helper()
	A, B, Q, S := fzRatPoly(a), fzRatPoly(b), fzRatPoly(q), fzRatPoly(s)
	Ri := mqRatInv(fzRatPoly(r), m)
	if Ri == nil {
		t.Fatal("R singular in exact arithmetic")
	}
	G := mqRatMul(mqRatMul(B, Ri, n, m, m), mqRatT(B, n, m), n, m, n)
	F := mqRatMul(Ri, mqRatT(S, n, m), m, m, n)
	BF, SF := mqRatMul(B, F, n, m, n), mqRatMul(S, F, n, m, n)
	d := 2 * n
	L, M := make([]*big.Rat, d*d), make([]*big.Rat, d*d)
	for i := range L {
		L[i], M[i] = new(big.Rat), new(big.Rat)
	}
	for i := range n {
		for j := range n {
			M[i*d+j].Sub(A[i*n+j], BF[i*n+j])
			M[(n+i)*d+j].Sub(SF[i*n+j], Q[i*n+j])
			L[i*d+n+j].Set(G[i*n+j])
			L[(n+i)*d+n+j].Sub(A[j*n+i], BF[j*n+i])
		}
		L[i*d+i].SetInt64(1)
		M[(n+i)*d+n+i].SetInt64(1)
	}
	p := fzTrimRat(mqDetPoly(L, M, d))
	for len(p) < d+1 {
		p = append([]*big.Rat{new(big.Rat)}, p...)
	}
	at := new(big.Rat)
	for _, c := range p {
		at.Neg(at).Add(at, c)
	}
	if at.Sign() == 0 {
		return true
	}
	one, minus := big.NewRat(1, 1), big.NewRat(-1, 1)
	P := make([]*big.Rat, d+1)
	for i := range P {
		P[i] = new(big.Rat)
	}
	for i, c := range p {
		f := []*big.Rat{big.NewRat(1, 1)}
		for range d - i {
			f = fzRatPolyMul(f, []*big.Rat{one, one})
		}
		for range i {
			f = fzRatPolyMul(f, []*big.Rat{minus, one})
		}
		for j := range f {
			P[j].Add(P[j], new(big.Rat).Mul(c, f[j]))
		}
	}
	return evenPolyAxisRoot(t, fzTrimRat(P), d)
}

// dareUnitCirclePair is the data of TestDare_UnitCirclePairNoStabilizing
// (n = 4, m = 2, row-major): A a perturbed orthogonal matrix, Q ≈ 1e-10,
// R of cond 1e3 and a cross term S that makes Q − SR⁻¹Sᵀ indefinite.
func dareUnitCirclePair() (a, b, q, r, s []float64) {
	a = []float64{-0.9263546485295784, -0.1853483294438246, 0.1608373356486542, -0.2855889503431461, -0.08514801288556635, -0.744026542866163, -0.36397276373908843, 0.5538033465666263, -0.12360846913446866, 0.253742678843065, -0.9166163757066255, -0.28046202712142854, -0.3454530755508101, 0.5906110732010617, -0.013351370222736478, 0.7301377519682434}
	b = []float64{0.7841184475832665, -0.2625865945204884, -0.04995585738934194, -0.40249439038110113, 0.20207053088084748, -0.03764805666927029, 0.09934364756371772, 0.8069363876012987}
	q = []float64{2.5457805265128857e-10, 8.573415320773726e-11, 2.9058591893372496e-11, -2.1824090531683754e-10, 8.573415320773726e-11, 1.8422616194618447e-10, 2.3301772325331496e-10, -1.1233825255052844e-10, 2.9058591893372496e-11, 2.3301772325331496e-10, 4.0464879406293486e-10, 4.904652578230848e-11, -2.1824090531683754e-10, -1.1233825255052844e-10, 4.904652578230848e-11, 4.130390063962223e-10}
	r = []float64{11444.500746816846, -647.5919051404701, -647.5919051404701, 60.651958634690104}
	s = []float64{-0.020316634030970126, -0.010439231870301454, -0.004113616217047509, -0.002895850578145378, 0.002615743846862104, 0.010696536829168612, -0.027672879671926977, -0.009733827377609645}
	return
}

// TestDare_UnitCirclePairNoStabilizing (I4J6DP): the exact reduced
// symplectic pencil of the float64 data has a root on the unit circle
// (its Cayley image is even with a root of q on (−∞, 0)), so no
// stabilizing solution exists (idare: no stabilizing solution). Dare
// returned ErrNotConverged; it now proves the root on its extended
// pencil and returns ErrNoStabilizing, not wrapped with
// ErrNotStabilizable.
func TestDare_UnitCirclePairNoStabilizing(t *testing.T) {
	const n, m = 4, 2
	a, b, q, r, s := dareUnitCirclePair()
	if !symplecticUnitCircleRoot(t, n, m, a, b, q, r, s) {
		t.Fatal("oracle: no exact symplectic root on the unit circle")
	}
	for _, ns := range []bool{false, true} {
		_, err := Dare(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r),
			&RiccatiOpts{S: mat.NewDense(n, m, s), NoScaling: ns})
		if !errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) || errors.Is(err, ErrNotConverged) {
			t.Errorf("NoScaling %v: %v, want ErrNoStabilizing alone", ns, err)
		}
	}
}

// TestCare_AxisPairInexactSymmetryNotProved (I4J6DP): with Q of
// TestCare_HamiltonianAxisPairNoStabilizing one ulp off symmetric, which
// Care accepts, the Hamiltonian is not exactly Hamiltonian and its axis
// pair is not proved, so Care keeps ErrNotConverged.
func TestCare_AxisPairInexactSymmetryNotProved(t *testing.T) {
	const n, m = 4, 3
	a, b, q, r, s := hamiltonianAxisPair()
	q[1] = math.Nextafter(q[1], 1)
	for _, ns := range []bool{false, true} {
		_, err := Care(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r),
			&RiccatiOpts{S: mat.NewDense(n, m, s), NoScaling: ns})
		if !errors.Is(err, ErrNotConverged) || errors.Is(err, ErrNoStabilizing) {
			t.Errorf("NoScaling %v: %v, want ErrNotConverged", ns, err)
		}
	}
}

// TestRiccatiBoundaryProof_NearAxisNotProved (I4J6DP): a lightly damped
// mode, damping 1e-12 (continuous) or radius 1 − 1e-9 (discrete), puts a
// Hamiltonian (symplectic) pencil eigenvalue within the float64 candidate
// band of the boundary but, in exact arithmetic, off it: the proof must
// fail, and Care and Dare keep their certified X.
func TestRiccatiBoundaryProof_NearAxisNotProved(t *testing.T) {
	const d = 1e-12
	B := mat.NewDense(3, 1, []float64{0, 1e-5, 1})
	Q := mat.NewDense(3, 3, []float64{1e-10, 0, 0, 0, 0, 0, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	for _, c := range []struct {
		continuous bool
		a          []float64
	}{
		{true, []float64{-d, 2, 0.3, -0.5, -d, 0.1, 0, 0, -1}},
		{false, []float64{0.6, 0.8 - 1e-9, 0.1, -0.8, 0.6, 0, 0, 0, 0.5}},
	} {
		A := mat.NewDense(3, 3, c.a)
		L, M := extendedPencil(c.continuous, A, B, Q, R, nil)
		if !boundaryCandidate(c.continuous, L, M, 7) {
			t.Errorf("continuous %v: no float64 candidate, want one", c.continuous)
		}
		if boundaryEigenvalueProved(c.continuous, A, B, Q, R, nil) {
			t.Errorf("continuous %v: boundary eigenvalue proved, want none", c.continuous)
		}
		solve := Dare
		if c.continuous {
			solve = Care
		}
		if _, err := solve(A, B, Q, R, nil); err != nil {
			t.Errorf("continuous %v: %v, want a certified X", c.continuous, err)
		}
	}
}

// TestRiccatiBoundaryProof_NegativeRoot (I4J6DP): hasNegativeRoot, by sign
// change or the integer Sturm count, against polynomials built from known
// roots: repeated (even multiplicity, no sign change) and positive real
// roots, and quadratics with no real root.
func TestRiccatiBoundaryProof_NegativeRoot(t *testing.T) {
	asc := func(roots []int64, quads int, scale int64) []*big.Rat {
		p := []*big.Rat{big.NewRat(scale, 1)}
		mul := func(f []*big.Rat) {
			out := make([]*big.Rat, len(p)+len(f)-1)
			for i := range out {
				out[i] = new(big.Rat)
			}
			for i := range p {
				for j := range f {
					out[i+j].Add(out[i+j], new(big.Rat).Mul(p[i], f[j]))
				}
			}
			p = out
		}
		for _, r := range roots {
			mul([]*big.Rat{big.NewRat(-r, 3), big.NewRat(1, 1)})
		}
		for k := range quads {
			mul([]*big.Rat{big.NewRat(int64(k+2), 1), big.NewRat(int64(k), 7), big.NewRat(1, 1)})
		}
		return p
	}
	for _, c := range []struct {
		roots []int64
		quads int
		scale int64
		want  bool
	}{
		{[]int64{-1}, 0, 1, true},
		{[]int64{-1, -1}, 1, -3, true},
		{[]int64{-1, -1, -2, -2}, 2, 5, true},
		{[]int64{-4, -4, 4}, 1, -1, true},
		{[]int64{5, 6}, 3, 2, false},
		{[]int64{-3, -3, -5, -5, 2, 2}, 2, 7, true},
		{[]int64{1, 2, 2}, 1, -2, false},
		{nil, 3, 1, false},
	} {
		if got := hasNegativeRoot(asc(c.roots, c.quads, c.scale)); got != c.want {
			t.Errorf("roots %v/3, %d quadratics, scale %d: negative root %v, want %v", c.roots, c.quads, c.scale, got, c.want)
		}
	}
}

// TestDare_NearUnitCircleNotProved (I4J6DP; census zw46reg, descriptor
// n = 2): Dare finds no verified solution, and a float64 QZ eigenvalue of
// the extended symplectic pencil lies within the candidate band of the unit
// circle, but its exact pencil has no root on the circle (the reduced
// pencil oracle of the census agrees): the error must stay ErrNotConverged.
func TestDare_NearUnitCircleNotProved(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.7071077945360846, 0.5651357546147349, -0.483579317756526, 0.30502374582034064})
	B := mat.NewDense(2, 1, []float64{-0.006727119980158644, -0.3605160239611313})
	Q := mat.NewDense(2, 2, []float64{0.29901236741459425, -0.33933860216785683, -0.33933860216785683, 1.811958051992169})
	R := mat.NewDense(1, 1, []float64{1.721750259837974})
	E := mat.NewDense(2, 2, []float64{-0.6791155402070246, -0.5629147446817965, 1.9837244002196524, -0.18599666473396131})
	opts := &RiccatiOpts{E: E}
	L, M := extendedPencil(false, A, B, Q, R, opts)
	if !boundaryCandidate(false, L, M, 5) {
		t.Error("no float64 candidate, want one")
	}
	if boundaryEigenvalueProved(false, A, B, Q, R, opts) {
		t.Error("unit-circle eigenvalue proved, want none")
	}
	for _, ns := range []bool{false, true} {
		_, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: ns})
		if !errors.Is(err, ErrNotConverged) || errors.Is(err, ErrNoStabilizing) {
			t.Errorf("NoScaling %v: %v, want ErrNotConverged", ns, err)
		}
	}
}
