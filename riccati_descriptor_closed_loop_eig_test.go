package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// assertClosedLoopEigExact checks that eig holds the eigenvalues of the
// exact pencil (A − BK, E) of the returned K, each within 8× the
// first-order backward-error bound n·eps·(‖A − BK‖_F + |λ|‖E‖_F)·κ(λ) of
// the pencil with its rows equilibrated by powers of two (an exact
// equivalence), κ = ‖x‖‖y‖/|yᴴEx|: the roots of the 320-bit characteristic
// polynomial of F = E⁻¹(A − BK), Newton-polished from its float64
// eigenvalues.
func assertClosedLoopEigExact(t *testing.T, name string, A, B, K, E *mat.Dense, eig []complex128) {
	t.Helper()
	Ah := hbAdd(hbFrom(A), hbMul(hbFrom(B), hbFrom(K)), -1)
	Eh := hbFrom(E)
	for i := range Ah.r {
		row := 0.0
		for j := range Ah.c {
			a, _ := Ah.at(i, j).Float64()
			row = math.Hypot(row, math.Hypot(a, E.At(i, j)))
		}
		d := pow2Near(1 / row)
		for j := range Ah.c {
			Ah.at(i, j).SetMantExp(Ah.at(i, j), math.Ilogb(d))
			Eh.at(i, j).SetMantExp(Eh.at(i, j), math.Ilogb(d))
		}
	}
	E = hbDense(Eh)
	Fh := hbSolve(Eh, Ah)
	F := hbDense(Fh)
	n, _ := F.Dims()
	var ed mat.Eigen
	if !ed.Factorize(F, mat.EigenBoth) {
		t.Fatalf("%s: oracle eig failed", name)
	}
	want := ed.Values(nil)
	var vr, vl mat.CDense
	ed.VectorsTo(&vr)
	ed.LeftVectorsTo(&vl)
	var lu mat.LU
	lu.Factorize(E.T())
	an, en := frobNorm(hbDense(Ah)), frobNorm(E)
	bound := make([]float64, n)
	zr, zi := mat.NewVecDense(n, nil), mat.NewVecDense(n, nil)
	var yr, yi mat.VecDense
	for k := range n {
		var xx, zx complex128
		for i := range n {
			x, z := vr.At(i, k), vl.At(i, k)
			xx += x * cmplx.Conj(x)
			zx += cmplx.Conj(z) * x
			zr.SetVec(i, real(z))
			zi.SetVec(i, imag(z))
		}
		if err := lu.SolveVecTo(&yr, false, zr); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := lu.SolveVecTo(&yi, false, zi); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		yy := mat.Dot(&yr, &yr) + mat.Dot(&yi, &yi)
		bound[k] = float64(n) * eps() * (an + cmplx.Abs(want[k])*en) * math.Sqrt(real(xx)*yy) / cmplx.Abs(zx)
	}
	coef := hbCharPoly(Fh)
	for i, w := range want {
		want[i] = hbPolishRoot(coef, w)
	}
	if len(eig) != n {
		t.Fatalf("%s: Eig %v, want %d values", name, eig, n)
	}
	used := make([]bool, n)
	for _, v := range eig {
		bi, bd := -1, math.Inf(1)
		for i, w := range want {
			if d := cmplx.Abs(v - w); !used[i] && d < bd {
				bi, bd = i, d
			}
		}
		if bi < 0 || !(bd <= 8*bound[bi]) {
			t.Errorf("%s: Eig %v, exact closed loop %v (bounds %.3g)", name, eig, want, bound)
			return
		}
		used[bi] = true
	}
}

// hbCharPoly returns the coefficients c₀..cₙ (cₙ = 1) of det(λI − F) by
// Faddeev–LeVerrier.
func hbCharPoly(F hbMat) []*big.Float {
	n := F.r
	c := make([]*big.Float, n+1)
	c[n] = new(big.Float).SetPrec(hbPrec).SetInt64(1)
	M := newHbMat(n, n)
	tr := new(big.Float).SetPrec(hbPrec)
	for k := 1; k <= n; k++ {
		for i := range n {
			M.at(i, i).Add(M.at(i, i), c[n-k+1])
		}
		M = hbMul(F, M)
		tr.SetInt64(0)
		for i := range n {
			tr.Add(tr, M.at(i, i))
		}
		c[n-k] = new(big.Float).SetPrec(hbPrec).Quo(tr, new(big.Float).SetInt64(int64(-k)))
	}
	return c
}

// hbPolishRoot refines the root z of the polynomial coef by Newton's
// method in 320-bit complex arithmetic.
func hbPolishRoot(coef []*big.Float, z complex128) complex128 {
	nf := func(v float64) *big.Float { return new(big.Float).SetPrec(hbPrec).SetFloat64(v) }
	mul := func(ar, ai, br, bi *big.Float) (*big.Float, *big.Float) {
		r := nf(0).Sub(nf(0).Mul(ar, br), nf(0).Mul(ai, bi))
		i := nf(0).Add(nf(0).Mul(ar, bi), nf(0).Mul(ai, br))
		return r, i
	}
	zr, zi := nf(real(z)), nf(imag(z))
	for range 60 {
		pr, pi, dr, di := nf(0), nf(0), nf(0), nf(0)
		for k := len(coef) - 1; k >= 0; k-- {
			dr, di = mul(dr, di, zr, zi)
			dr.Add(dr, pr)
			di.Add(di, pi)
			pr, pi = mul(pr, pi, zr, zi)
			pr.Add(pr, coef[k])
		}
		den := nf(0).Add(nf(0).Mul(dr, dr), nf(0).Mul(di, di))
		if den.Sign() == 0 {
			break
		}
		qr := nf(0).Quo(nf(0).Add(nf(0).Mul(pr, dr), nf(0).Mul(pi, di)), den)
		qi := nf(0).Quo(nf(0).Sub(nf(0).Mul(pi, dr), nf(0).Mul(pr, di)), den)
		zr.Sub(zr, qr)
		zi.Sub(zi, qi)
	}
	r, _ := zr.Float64()
	i, _ := zi.Float64()
	return complex(r, i)
}

func pow2Dense(M *mat.Dense, e int) *mat.Dense {
	o := mat.DenseCopyOf(M)
	raw := o.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j, v := range row {
			row[j] = math.Ldexp(v, e)
		}
	}
	return o
}

// BGKVXR (from AHNSX7): with E, A and B scaled by 1e-15 the Riccati solve
// of the augmented descriptor blkdiag(E, I) returned K exact but Eig
// 2.6e-2 (dt 0) and 1.6e-3 (dt 0.1) off the closed loop of that K, nil
// error: Eig came from E⁻¹(A − BK) of the state-scaled iterate, not
// balanced. Lqi normalizes the plant rows; the augmented pencil is solved
// directly here as well.
func TestLqiDescriptorSmallNormClosedLoopEig(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		plant := obsTestPlant(t, dt, true)
		E, A, B := mat.DenseCopyOf(plant.E), mat.DenseCopyOf(plant.A), mat.DenseCopyOf(plant.B)
		for _, M := range []*mat.Dense{E, A, B} {
			M.Scale(1e-15, M)
		}
		sys, err := NewDescriptor(A, B, plant.C, plant.D, E, dt)
		if err != nil {
			t.Fatal(err)
		}
		Q, R := eyeDense(5), eyeDense(2)
		Aa, Ba := lqiAugmentation(sys)
		Ea := augmentedDescriptor(sys.E, 2)

		res, err := Lqi(sys, Q, R, nil)
		if err != nil {
			t.Fatalf("dt=%v Lqi: %v", dt, err)
		}
		assertClosedLoopEigExact(t, "Lqi", Aa, Ba, res.K, Ea, res.Eig)

		opts := &RiccatiOpts{E: Ea, plantE: sys.E, augmentedE: true}
		solve := Lqr
		if dt > 0 {
			solve = Dlqr
		}
		res, err = solve(Aa, Ba, Q, R, opts)
		if err != nil {
			t.Fatalf("dt=%v augmented: %v", dt, err)
		}
		assertClosedLoopEigExact(t, "augmented", Aa, Ba, res.K, Ea, res.Eig)
	}
}

// E7ZW6G: scaling (E, A, B) by 2³⁰, an exact equivalence, left K correct
// to 1e-11 but moved Eig 5.1e-6 (2e-5 with the integrator rows at 2²⁹ or
// 2³¹) with cond(E) 1e4, nil error.
func TestLqrDescriptorEigInvariantUnderEquationScaling(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 7))
	U, V := randomOrthogonal(rng, 3), randomOrthogonal(rng, 3)
	graded := mulDense(mulDense(U, mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1e-2, 0, 0, 0, 1e-4})), V)
	plant := obsTestPlant(t, 0, true)
	Q, R := eyeDense(5), eyeDense(2)
	var ref []complex128
	for _, c := range [][2]int{{0, 0}, {30, 30}, {30, 29}, {30, 31}, {-30, -30}, {20, 20}} {
		j, k := c[0], c[1]
		sys, err := NewDescriptor(pow2Dense(plant.A, j), pow2Dense(plant.B, j), plant.C, plant.D, pow2Dense(graded, j), 0)
		if err != nil {
			t.Fatal(err)
		}
		Aa, Ba := lqiAugmentation(sys)
		Ea := augmentedDescriptor(sys.E, 2)
		for _, M := range []*mat.Dense{Ea, Aa, Ba} {
			raw := M.RawMatrix()
			for i := 3; i < 5; i++ {
				for q := range raw.Cols {
					raw.Data[i*raw.Stride+q] = math.Ldexp(raw.Data[i*raw.Stride+q], k)
				}
			}
		}
		for _, api := range []string{"Lqr", "Care"} {
			opts := &RiccatiOpts{E: Ea, plantE: sys.E, augmentedE: true}
			solve := Lqr
			if api == "Care" {
				solve = Care
			}
			res, err := solve(Aa, Ba, Q, R, opts)
			if err != nil {
				t.Fatalf("j=%d k=%d %s: %v", j, k, api, err)
			}
			assertClosedLoopEigExact(t, api, Aa, Ba, res.K, Ea, res.Eig)
			if ref == nil {
				ref = res.Eig
			} else if d := eigSetRelErr(res.Eig, ref); !(d <= 1e-9) {
				t.Errorf("j=%d k=%d %s: Eig %v is %.3g from the unscaled %v", j, k, api, res.Eig, d, ref)
			}
		}
	}
}
