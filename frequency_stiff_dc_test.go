package controlsys

import (
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// stiffFoldTwinExplicit is the explicit form of the triangular twin (F, W)
// of a nearSingularEContinuousK draw: W's last entry is of order ε, so the
// fold puts one pole near 1e15 above slow poles F_ii/W_ii.
func stiffFoldTwinExplicit(t *testing.T, seed uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7704))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	_, twin := nearSingularEDraw(t, rng, n, m, p, k)
	tx, err := twin.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	return tx
}

// stiffSlowPoleExplicit draws an upper-triangular A with slow poles of
// modulus in [0.01, 3] and one pole of modulus in [1e13, 1e16], so the
// slow poles lie below ε·‖A‖ yet A is far from singular componentwise.
func stiffSlowPoleExplicit(t *testing.T, seed uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 2026))
	n := 2 + rng.IntN(4)
	sign := func() float64 {
		if rng.IntN(2) == 0 {
			return -1
		}
		return 1
	}
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, sign()*math.Pow(10, -2+rng.Float64()*2.5))
		for j := i + 1; j < n; j++ {
			A.Set(i, j, rng.NormFloat64())
		}
	}
	A.Set(n-1, n-1, sign()*math.Pow(10, 13+3*rng.Float64()))
	B, C, D := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, []float64{rng.NormFloat64()})
	for i := range n {
		B.Set(i, 0, rng.NormFloat64())
		C.Set(0, i, rng.NormFloat64())
	}
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// bigExplicitDCGain is −CA⁻¹B + D of a continuous explicit model by GEPP
// in 256-bit big.Float.
func bigExplicitDCGain(t *testing.T, sys *System) [][]float64 {
	t.Helper()
	const prec = 256
	n, m, p := sys.Dims()
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(prec).SetFloat64(x) }
	a := make([][]*big.Float, n)
	for i := range n {
		a[i] = make([]*big.Float, n+m)
		for j := range n {
			a[i][j] = nf(sys.A.At(i, j))
		}
		for j := range m {
			a[i][n+j] = nf(sys.B.At(i, j))
		}
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if new(big.Float).Abs(a[i][k]).Cmp(new(big.Float).Abs(a[piv][k])) > 0 {
				piv = i
			}
		}
		a[k], a[piv] = a[piv], a[k]
		if a[k][k].Sign() == 0 {
			t.Fatal("oracle: A singular")
		}
		for i := k + 1; i < n; i++ {
			f := new(big.Float).SetPrec(prec).Quo(a[i][k], a[k][k])
			for j := k; j < n+m; j++ {
				a[i][j].Sub(a[i][j], new(big.Float).SetPrec(prec).Mul(f, a[k][j]))
			}
		}
	}
	x := make([][]*big.Float, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = make([]*big.Float, m)
		for c := range m {
			v := new(big.Float).SetPrec(prec).Set(a[i][n+c])
			for j := i + 1; j < n; j++ {
				v.Sub(v, new(big.Float).SetPrec(prec).Mul(a[i][j], x[j][c]))
			}
			x[i][c] = v.Quo(v, a[i][i])
		}
	}
	g := make([][]float64, p)
	for r := range p {
		g[r] = make([]float64, m)
		for c := range m {
			v := nf(sys.D.At(r, c))
			for i := range n {
				v.Sub(v, new(big.Float).SetPrec(prec).Mul(nf(sys.C.At(r, i)), x[i][c]))
			}
			g[r][c], _ = v.Float64()
		}
	}
	return g
}

// assertDCResponse checks EvalFr(0), FreqResponse at ω = 0 and DCGain
// against want to rel relative error.
func assertDCResponse(t *testing.T, label string, sys *System, want [][]float64, rel float64) {
	t.Helper()
	ev, err := sys.EvalFr(0)
	if err != nil {
		t.Fatalf("%s: EvalFr(0): %v", label, err)
	}
	fr, err := sys.FreqResponse([]float64{0})
	if err != nil {
		t.Fatalf("%s: FreqResponse: %v", label, err)
	}
	dc, err := sys.DCGain()
	if err != nil {
		t.Fatalf("%s: DCGain: %v", label, err)
	}
	for i := range want {
		for j, w := range want[i] {
			tol := rel * math.Max(math.Abs(w), 1e-300)
			if g := ev[i][j]; !(cmplx.Abs(g-complex(w, 0)) <= tol) {
				t.Errorf("%s: EvalFr(0)[%d][%d] = %v, want %v", label, i, j, g, w)
			}
			if g := fr.At(0, i, j); !(cmplx.Abs(g-complex(w, 0)) <= tol) {
				t.Errorf("%s: FreqResponse(0)[%d][%d] = %v, want %v", label, i, j, g, w)
			}
			if g := dc.At(i, j); !(math.Abs(g-w) <= tol) {
				t.Errorf("%s: DCGain[%d][%d] = %v, want %v", label, i, j, g, w)
			}
		}
	}
}

// A stiff explicit model whose slow pole lies below ε·‖A‖ but is located
// by the data has the DC response −CA⁻¹B + D at s = 0, as MATLAB freqresp
// and dcgain. The DC near-pole rule took any pole within 1e-6·‖poles‖ of 0
// for an integrator, so the s = 0 solve took the pole limit, whose samples
// at 1e-5·rho ~ 1e10 read the fast pole: 0.373 for 66.67 (seed 71), and
// DCGain ∓Inf for seeds 71 and 109 (BLHSVK).
func TestEvalFrStiffExplicitDCResolvedSlowPole(t *testing.T) {
	for _, seed := range []uint64{71, 109, 8, 50} {
		sys := stiffFoldTwinExplicit(t, seed)
		assertDCResponse(t, fmt.Sprint("fold twin seed ", seed), sys, bigExplicitDCGain(t, sys), 1e-10)
	}
	for seed := range uint64(40) {
		sys := stiffSlowPoleExplicit(t, seed)
		assertDCResponse(t, fmt.Sprint("stiff seed ", seed), sys, bigExplicitDCGain(t, sys), 1e-10)
	}
}

// A pole at s = 0 keeps the pole limit at DC: an exact integrator beside a
// fast pole near 1e15, and an integrator rounded off 0 by a dense
// similarity, simple or double, alone or beside a fast pole, read +Inf
// where it reaches. On base the stiff ones read 0.5 = D: the pole limit
// stepped 1e-5·rho past the slow poles. DCGain reads the integrator too;
// for the rounded stiff one the transfer limit was finite (W45MGH). Where
// the integrator is rounded off 0, the float model's exact limit is finite,
// so DCGain's Inf comes with ErrAccuracyUnverified (N4GQHT).
func TestEvalFrStiffExplicitDCIntegratorStaysPole(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 2026))
	T := mat.NewDense(3, 3, nil)
	for i := range 9 {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	similar := func(j ...float64) *mat.Dense {
		var tmp, A mat.Dense
		tmp.Mul(T, mat.NewDense(3, 3, j))
		A.Mul(&tmp, &Ti)
		return &A
	}
	b := mat.NewDense(3, 1, []float64{0.7, -1.2, 0.4})
	c := mat.NewDense(1, 3, []float64{1.1, 0.3, -0.8})
	d := mat.NewDense(1, 1, []float64{0.5})
	for _, tc := range []struct {
		name string
		A    *mat.Dense
	}{
		{"exact stiff", mat.NewDense(3, 3, []float64{0, 0.4, -1.3, 0, -0.5, 0.9, 0, 0, -3.8e15})},
		{"rounded dense", similar(0, 0, 0, 0, -1, 0, 0, 0, -3)},
		{"rounded double", similar(0, 1, 0, 0, 0, 0, 0, 0, -2)},
		{"rounded stiff", similar(0, 0, 0, 0, -1e4, 0, 0, 0, -1e12)},
	} {
		sys, err := New(tc.A, b, c, d, 0)
		if err != nil {
			t.Fatal(err)
		}
		ev, err := sys.EvalFr(0)
		if err != nil {
			t.Fatalf("%s: EvalFr(0): %v", tc.name, err)
		}
		fr, err := sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatalf("%s: FreqResponse: %v", tc.name, err)
		}
		dc, err := sys.DCGain()
		if tc.name == "exact stiff" && err != nil {
			t.Fatalf("%s: DCGain: %v", tc.name, err)
		}
		if tc.name != "exact stiff" {
			requireDCGainUnverified(t, tc.name, dc, err)
		}
		if !cmplx.IsInf(ev[0][0]) || !cmplx.IsInf(fr.At(0, 0, 0)) || !math.IsInf(dc.At(0, 0), 0) {
			t.Errorf("%s: EvalFr(0) %v, FreqResponse(0) %v, DCGain %v, want Inf", tc.name, ev[0][0], fr.At(0, 0, 0), dc.At(0, 0))
		}
	}
}
