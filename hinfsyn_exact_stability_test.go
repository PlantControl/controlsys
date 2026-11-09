package controlsys

import (
	"cmp"
	"errors"
	"math/big"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hinfNearlyImproperCase is Mixsyn(G, W1, W2 = 0.01) of fuzz tuple
// kind=0x60 (NFSIWH, w12f #9): G SISO with poles 0.001 ± 1i and 3.16,
// RHP zeros 0.0072 ± 1.0008i and 3.164, and D = −95. The central
// controller's feedthrough nearly inverts G(∞), so its D22 loop shift
// gave K a pole at −1e13 and gains near 1e12; the exact loop of the
// float64 G and K was unstable (poles 0.0054 ± 1.037i) while the rounded
// float64 CL was stable.
func hinfNearlyImproperCase(t *testing.T) (*fzSpec, *System, *System, *System) {
	t.Helper()
	v := [fzNVals]float64{0, -5, 0.5, -2, 0, 0, 0, -100, 0, -10, 0.37, 41, 1, -10, -10, -2}
	s := syNormC(fzDecode(t, 0x60, v))
	aw, cw, dw, rho, _ := syMixWeights(s)
	W1, err := NewFromSlices(1, 1, 1, []float64{aw}, []float64{1}, []float64{cw}, []float64{dw}, s.Dt)
	if err != nil {
		t.Fatal(err)
	}
	W2, err := NewGain(mat.NewDense(1, 1, []float64{rho}), s.Dt)
	if err != nil {
		t.Fatal(err)
	}
	return s, s.System(t, fzSysOpts{}), W1, W2
}

// hinfExactGKUnstable returns the number of closed right-half-plane roots
// of the exact loop u = K·e, e = −G·u, of the float64 G of s and K
// (Routh on the big.Rat characteristic polynomial).
func hinfExactGKUnstable(t *testing.T, s *fzSpec, K *System) int {
	t.Helper()
	nk, a, b, c, d := syCtrlData(t, s, "Mixsyn", K, s.M, s.P)
	G := &syPlant{n: s.N, nu: s.M, ny: s.P, A: s.A, B2: s.B, C2: make([]float64, len(s.C)), D22: make([]float64, len(s.D))}
	for i, x := range s.C {
		G.C2[i] = -x
	}
	for i, x := range s.D {
		G.D22[i] = -x
	}
	loop, ok := syLFT(G, nk, a, b, c, d)
	if !ok {
		t.Fatal("I − Dk·D22 exactly singular")
	}
	k, _, ok, _ := syExactCount(loop.A, loop.N, s.Discrete, 1e-8)
	if !ok {
		t.Fatal("degenerate Routh table")
	}
	return k
}

func hinfSortedPoles(p []complex128) []complex128 {
	p = slices.Clone(p)
	slices.SortFunc(p, func(x, y complex128) int {
		return cmp.Or(cmp.Compare(real(x), real(y)), cmp.Compare(imag(x), imag(y)))
	})
	return p
}

// A nil error must come with a K whose exact loop with the float64 G is
// stable, and CLPoles must be the poles of the returned CL.
func TestMixsyn_NearlyImproperCentralControllerExactLoop(t *testing.T) {
	s, G, W1, W2 := hinfNearlyImproperCase(t)
	res, err := Mixsyn(G, W1, W2, nil)
	if res == nil {
		t.Fatalf("Mixsyn: %v", err)
	}
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("Mixsyn: result with error %v", err)
	}
	if err == nil {
		if k := hinfExactGKUnstable(t, s, res.K); k != 0 {
			t.Errorf("nil error but the exact loop of G and K has %d unstable roots", k)
		}
	}
	poles, perr := res.CL.Poles()
	if perr != nil {
		t.Fatal(perr)
	}
	if got, want := hinfSortedPoles(res.Info.CLPoles), hinfSortedPoles(poles); !slices.Equal(got, want) {
		t.Errorf("CLPoles %v, poles of CL %v", got, want)
	}
}

// The central controller base 2c611af returned for hinfNearlyImproperCase
// (native): its exact loop is unstable, so the certificate must reject it.
func TestHinfSynStabilityCertificate_RejectsFragileController(t *testing.T) {
	s, G, W1, W2 := hinfNearlyImproperCase(t)
	K, err := New(
		mat.NewDense(4, 4, []float64{
			-6.047207962748546e+11, 1.5655296211713892e+12, -1.8230017001013008e+12, -9.17588894528671e+09,
			1.5623015443125244e+12, -4.0445596709913193e+12, 4.709741072077431e+12, 2.3705990529805664e+10,
			-1.8361517050013406e+12, 4.753515838795276e+12, -5.535294470573952e+12, -2.786132746810194e+10,
			-8.429834906564793e-06, 2.1823552742716856e-05, -2.5412728064111434e-05, -0.3162278939291383}),
		mat.NewDense(4, 1, []float64{8.983853109208295e+07, -2.3209864240571094e+08, 2.7278237003201264e+08, 0.5000000012523531}),
		mat.NewDense(1, 4, []float64{1.5985843644927583e+09, -4.1384903412590284e+09, 4.819119885012471e+09, 2.425653737814983e+07}),
		mat.NewDense(1, 1, []float64{-237488.8802164854}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if k := hinfExactGKUnstable(t, s, K); k == 0 {
		t.Fatal("oracle: exact loop stable; the pinned controller is not the fragile one")
	}
	P, err := Augw(G, W1, W2, nil)
	if err != nil {
		t.Fatal(err)
	}
	gp, err := partitionGeneralizedPlant("HinfSyn", P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if stabilityCertified(gp, K) {
		t.Error("certificate accepts a controller whose exact loop is unstable")
	}
}

// CLPoles of a discrete design (Tustin route) are the poles of CL.
func TestHinfSyn_DiscreteCLPolesArePolesOfCL(t *testing.T) {
	P, ny, nu := hinfRegPlant(t, 0xb3, hinfLargeGammaTuple, 1)
	res, err := HinfSyn(P, ny, nu)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := res.CL.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := hinfSortedPoles(res.CLPoles), hinfSortedPoles(poles); !slices.Equal(got, want) {
		t.Errorf("CLPoles %v, poles of CL %v", got, want)
	}
}

func TestLyapunovStableAndExactStable(t *testing.T) {
	rat := func(m *mat.Dense) [][]*big.Rat { return ratMat(m) }
	for _, c := range []struct {
		name     string
		a        []float64
		discrete bool
		lyap     bool
		exact    bool
	}{
		{"Hurwitz non-symmetric", []float64{-1, 3, -2, -4}, false, true, true},
		{"one RHP pole", []float64{0.5, 3, 0, -4}, false, false, false},
		{"imaginary axis pair", []float64{0, 1, -1, 0}, false, false, false},
		// Stable with margin 1e-20 relative to ‖A‖: exact only.
		{"tiny margin", []float64{-1e-20, 1, 0, -1}, false, false, true},
		{"Schur non-symmetric", []float64{0.5, 0.4, -0.3, 0.2}, true, true, true},
		{"pole outside disk", []float64{1.01, 0, 0.3, 0.2}, true, false, false},
		{"pole on circle", []float64{-1, 0, 0.3, 0.2}, true, false, false},
		{"Schur tiny margin", []float64{1 - 0x1p-52, 0, 0.3, 0.2}, true, false, true},
	} {
		A := mat.NewDense(2, 2, c.a)
		if got := lyapunovStable(A, c.discrete); got != c.lyap {
			t.Errorf("%s: lyapunovStable %v, want %v", c.name, got, c.lyap)
		}
		if got := exactStable(rat(A), c.discrete); got != c.exact {
			t.Errorf("%s: exactStable %v, want %v", c.name, got, c.exact)
		}
	}
}

// Beyond the exact path's size gates (rlExactMaxStates, rlExactBitBudget)
// exactStable fails closed: stable matrices are reported uncertified.
func TestExactStableSizeGateFailsClosed(t *testing.T) {
	n := rlExactMaxStates + 1
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1))
	}
	if exactStable(ratMat(A), false) {
		t.Errorf("order %d > rlExactMaxStates: exactStable true, want false", n)
	}
	if !lyapunovStable(A, false) {
		t.Errorf("order %d diagonal Hurwitz: lyapunovStable false", n)
	}
	// Upper triangular, Hurwitz, binary spread 2000 over 3 states:
	// 3·(2001+64) > rlExactBitBudget.
	B := mat.NewDense(3, 3, []float64{-1, 0x1p1000, 0, 0, -0x1p-1000, 0, 0, 0, -1})
	if exactStable(ratMat(B), false) {
		t.Error("over rlExactBitBudget: exactStable true, want false")
	}
	C := mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -0x1p-500, 0, 0, 0, -1})
	if !exactStable(ratMat(C), false) {
		t.Error("within budget: exactStable false on a Hurwitz triangular matrix")
	}
}
