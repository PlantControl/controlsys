package controlsys

import (
	"errors"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hamiltonianAxisPair is the data of TestCare_HamiltonianAxisPairNoStabilizing
// (n = 4, m = 3, row-major).
func hamiltonianAxisPair() (a, b, q, r, s []float64) {
	a = []float64{-5.852900801308777e-17, -3.1308312642672526e-17, -1.011821108703951e-17, -0.7487487897980519, -6.2540681277693556e-18, 2.2839678909122157e-17, 9.176669748562204e-19, -0.6466877066541815, 4.428638875100395e-18, 2.9662967492570215e-18, 2.706018035580183e-19, -0.14550003380861354, 0.7487487897980522, 0.6466877066541818, 0.14550003380861357, 2.191876733536412e-17}
	b = []float64{-0.7550087835333313, -0.1848558071733854, 0.941656251861669, -0.8337832088965788, 0.8975031783126406, -0.07241875797759885, 0.3089367201199639, 0.6650179049443035, -0.9804001177205474, 0.9990636303990257, -0.5417201571387211, -0.4520930081696937}
	q = []float64{9.139585954334613e-11, 7.179176350643137e-11, -1.3781491580006885e-10, 1.731648536624537e-11, 7.179176350643137e-11, 2.0796681358957843e-10, -2.0625876568115495e-10, -7.460419955184777e-11, -1.3781491580006885e-10, -2.0625876568115495e-10, 2.7117752983779706e-10, 3.092103335238967e-11, 1.731648536624537e-11, -7.460419955184777e-11, 3.092103335238967e-11, 5.461130459291629e-11}
	r = []float64{3489.2163523210584, 4983.1862300899365, 5690.421572455358, 4983.1862300899365, 7176.559696654014, 8236.910343230698, 5690.421572455358, 8236.910343230698, 9482.973084042356}
	s = []float64{0.024997152419964155, -0.022427896956449166, -0.019217730337032675, 0.0035215060875750624, -0.031617872079749046, 0.004626071390055927, -0.020090238583525835, -0.02162887027003342, 0.025663760287776942, -0.03151551411462858, 0.001479914014287713, 0.03113415656908463}
	return
}

// TestCare_HamiltonianAxisPairNoStabilizing (3XCWZW, seed
// FuzzRiccati/b6a0ca90111879f6; I4J6DP): A with spectrum {0, 0, ±i},
// Q ≈ 1e-10, R of cond 1e7 and a cross term S that makes Q − SR⁻¹Sᵀ
// indefinite. The exact Hamiltonian of the float64 data has characteristic
// polynomial q(s²), q(t) ≈ t⁴ − 44.79t³ − 45.84t² + 0.01487t − 7.421e-18:
// q(0) < 0 and q(−∞) > 0, and its Sturm sequence counts one distinct root
// on (−∞, 0), between −2 and −1, a pair ±iω exactly on the imaginary axis,
// so no stabilizing solution exists (icare: no stabilizing solution). Care
// returned ErrNotConverged; it now proves the pair and returns
// ErrNoStabilizing, not wrapped with ErrNotStabilizable.
func TestCare_HamiltonianAxisPairNoStabilizing(t *testing.T) {
	const n, m = 4, 3
	a, b, q, r, s := hamiltonianAxisPair()

	A, B, Q, S := fzRatPoly(a), fzRatPoly(b), fzRatPoly(q), fzRatPoly(s)
	Ri := mqRatInv(fzRatPoly(r), m)
	if Ri == nil {
		t.Fatal("R singular in exact arithmetic")
	}
	G := mqRatMul(mqRatMul(B, Ri, n, m, m), mqRatT(B, n, m), n, m, n)
	F := mqRatMul(Ri, mqRatT(S, n, m), m, m, n)
	BF, SF := mqRatMul(B, F, n, m, n), mqRatMul(S, F, n, m, n)
	const d = 2 * n
	L, H := make([]*big.Rat, d*d), make([]*big.Rat, d*d)
	for i := range L {
		L[i], H[i] = new(big.Rat), new(big.Rat)
	}
	for i := range n {
		for j := range n {
			ab := new(big.Rat).Sub(A[i*n+j], BF[i*n+j])
			H[i*d+j].Set(ab)
			H[(n+j)*d+n+i].Neg(ab)
			H[i*d+n+j].Neg(G[i*n+j])
			H[(n+i)*d+j].Sub(SF[i*n+j], Q[i*n+j])
		}
		L[i*d+i].SetInt64(1)
		L[(n+i)*d+n+i].SetInt64(1)
	}
	if !evenPolyAxisRoot(t, fzTrimRat(mqDetPoly(L, H, d)), d) {
		t.Fatal("oracle: no exact Hamiltonian root on the imaginary axis")
	}

	for _, ns := range []bool{false, true} {
		_, err := Care(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(n, n, q), mat.NewDense(m, m, r),
			&RiccatiOpts{S: mat.NewDense(n, m, s), NoScaling: ns})
		if !errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrNotStabilizable) || errors.Is(err, ErrNotConverged) {
			t.Errorf("NoScaling %v: %v, want ErrNoStabilizing alone", ns, err)
		}
	}
}

// evenPolyAxisRoot reports whether the descending exact polynomial p of
// degree d is q(s²) with q(0) = 0 or a root of q on (−∞, 0), failing t if
// p has degree other than d or a nonzero odd coefficient.
func evenPolyAxisRoot(t *testing.T, p []*big.Rat, d int) bool {
	t.Helper()
	if len(p) != d+1 {
		t.Fatalf("oracle: degree %d, want %d", len(p)-1, d)
	}
	var q []*big.Rat
	for k, c := range p {
		if (d-k)%2 == 1 {
			if c.Sign() != 0 {
				t.Fatalf("oracle: odd coefficient of s^%d nonzero", d-k)
			}
			continue
		}
		q = append(q, c)
	}
	return q[len(q)-1].Sign() == 0 || fzSturmCount(q) > 0
}
