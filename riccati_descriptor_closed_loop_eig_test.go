package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// assertClosedLoopEigExact checks that eig holds the eigenvalues of the
// exact pencil (A − BK, E) of the returned K, each within 8× the
// first-order backward-error bound n·eps·(‖A − BK‖_F + |λ|‖E‖_F)·κ(λ) of
// the pencil with its rows equilibrated by powers of two (an exact
// equivalence), κ = ‖x‖‖y‖/|yᴴEx|, for some pairing of eig with the
// eigenvalues: those are the distinct roots of the 320-bit characteristic
// polynomial of F = E⁻¹(A − BK) Newton-converged from the float64
// eigenvalues of F and from eig, x and y from 320-bit inverse iteration.
// Seeds from F alone are not enough: rounding F with an eigenvalue of 1e8
// lost its O(1) ones, which then polished onto duplicates; pairing nearest
// first swapped two eigenvalues within 1e-12 of zero.
func assertClosedLoopEigExact(t *testing.T, name string, A, B, K, E *mat.Dense, eig []complex128) {
	t.Helper()
	ratio, want, bound := closedLoopEigOracleRatio(A, B, K, E, eig)
	if want == nil {
		t.Fatalf("%s: oracle eig failed for Eig %v", name, eig)
	}
	if !(ratio <= 8) {
		t.Errorf("%s: Eig %v, exact closed loop %v (bounds %.3g): %.3g× the bound", name, eig, want, bound, ratio)
	}
}

// closedLoopEigOracleRatio returns the least over pairings of eig with the
// exact eigenvalues of (A − BK, E) of the largest error over its bound, as
// assertClosedLoopEigExact defines them, and those eigenvalues and bounds;
// nil eigenvalues when the oracle fails.
func closedLoopEigOracleRatio(A, B, K, E *mat.Dense, eig []complex128) (float64, []complex128, []float64) {
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
	Fh := hbSolve(Eh, Ah)
	n := Fh.r
	var ed mat.Eigen
	if len(eig) != n || !ed.Factorize(hbDense(Fh), mat.EigenNone) {
		return 0, nil, nil
	}
	coef := hbCharPoly(Fh)
	var want []complex128
	for _, z := range append(ed.Values(nil), eig...) {
		if !isFinite(real(z)) || !isFinite(imag(z)) {
			continue
		}
		r, ok := hbNewtonRoot(coef, z)
		if ok && !slices.ContainsFunc(want, func(w complex128) bool { return cmplx.Abs(w-r) <= 1e-13*cmplx.Abs(r) }) {
			want = append(want, r)
		}
	}
	if len(want) != n {
		return 0, nil, nil
	}
	an, en := frobNorm(hbDense(Ah)), frobNorm(hbDense(Eh))
	Ft, Et := Fh.t(), Eh.t()
	bound := make([]float64, n)
	for k, lam := range want {
		x, z := hbEigenvector(Fh, lam), hbEigenvector(Ft, cmplx.Conj(lam))
		if x == nil || z == nil {
			return 0, nil, nil
		}
		zr, zi := newHbMat(n, 1), newHbMat(n, 1)
		for i := range n {
			zr.d[i].SetFloat64(real(z[i]))
			zi.d[i].SetFloat64(imag(z[i]))
		}
		yr, yi := hbSolve(Et, zr), hbSolve(Et, zi)
		var xx, yy float64
		var zx complex128
		for i := range n {
			a, _ := yr.d[i].Float64()
			b, _ := yi.d[i].Float64()
			xx += real(x[i] * cmplx.Conj(x[i]))
			yy += a*a + b*b
			zx += cmplx.Conj(z[i]) * x[i]
		}
		bound[k] = float64(n) * eps() * (an + cmplx.Abs(lam)*en) * math.Sqrt(xx*yy) / cmplx.Abs(zx)
	}
	cost := make([]float64, n*n)
	for i, v := range eig {
		for j, w := range want {
			cost[i*n+j] = cmplx.Abs(v-w) / bound[j]
		}
	}
	cand := slices.Clone(cost)
	slices.Sort(cand)
	k, _ := slices.BinarySearchFunc(cand, true, func(th float64, _ bool) int {
		if bottleneckMatches(n, cost, th) {
			return 1
		}
		return -1
	})
	if k == len(cand) {
		return math.Inf(1), want, bound
	}
	return cand[k], want, bound
}

// bottleneckMatches reports whether rows and columns of the n×n cost pair
// up perfectly through entries at most th.
func bottleneckMatches(n int, cost []float64, th float64) bool {
	owner := make([]int, n)
	for j := range owner {
		owner[j] = -1
	}
	var try func(i int, seen []bool) bool
	try = func(i int, seen []bool) bool {
		for j := range n {
			if cost[i*n+j] <= th && !seen[j] {
				seen[j] = true
				if owner[j] < 0 || try(owner[j], seen) {
					owner[j] = i
					return true
				}
			}
		}
		return false
	}
	for i := range n {
		if !try(i, make([]bool, n)) {
			return false
		}
	}
	return true
}

// hbEigenvector returns an eigenvector of M for its eigenvalue nearest lam
// by 320-bit inverse iteration on the realified shifted system.
func hbEigenvector(M hbMat, lam complex128) []complex128 {
	n := M.r
	S := newHbMat(2*n, 2*n)
	for i := range n {
		for j := range n {
			S.at(i, j).Set(M.at(i, j))
			S.at(n+i, n+j).Set(M.at(i, j))
		}
		S.at(i, i).Sub(S.at(i, i), big.NewFloat(real(lam)))
		S.at(n+i, n+i).Sub(S.at(n+i, n+i), big.NewFloat(real(lam)))
		S.at(i, n+i).SetFloat64(imag(lam))
		S.at(n+i, i).SetFloat64(-imag(lam))
	}
	v := newHbMat(2*n, 1)
	for i := range v.d {
		v.d[i].SetFloat64(1 + 0.37*float64(i))
	}
	for range 3 {
		v = hbSolve(S, v)
		mx := 0.0
		for _, e := range v.d {
			f, _ := e.Float64()
			mx = math.Max(mx, math.Abs(f))
		}
		if !(mx > 0) || math.IsInf(mx, 0) {
			return nil
		}
		s := new(big.Float).SetPrec(hbPrec).SetFloat64(1 / mx)
		for _, e := range v.d {
			e.Mul(e, s)
		}
	}
	x := make([]complex128, n)
	for i := range n {
		re, _ := v.d[i].Float64()
		im, _ := v.d[n+i].Float64()
		x[i] = complex(re, im)
	}
	return x
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

// hbNewtonRoot refines the root z of the polynomial coef by Newton's
// method in 320-bit complex arithmetic, reporting whether it converged.
func hbNewtonRoot(coef []*big.Float, z complex128) (complex128, bool) {
	nf := func(v float64) *big.Float { return new(big.Float).SetPrec(hbPrec).SetFloat64(v) }
	mul := func(ar, ai, br, bi *big.Float) (*big.Float, *big.Float) {
		r := nf(0).Sub(nf(0).Mul(ar, br), nf(0).Mul(ai, bi))
		i := nf(0).Add(nf(0).Mul(ar, bi), nf(0).Mul(ai, br))
		return r, i
	}
	zr, zi := nf(real(z)), nf(imag(z))
	tiny := nf(1e-140)
	for range 200 {
		pr, pi, dr, di := nf(0), nf(0), nf(0), nf(0)
		for _, c := range slices.Backward(coef) {
			dr, di = mul(dr, di, zr, zi)
			dr.Add(dr, pr)
			di.Add(di, pi)
			pr, pi = mul(pr, pi, zr, zi)
			pr.Add(pr, c)
		}
		den := nf(0).Add(nf(0).Mul(dr, dr), nf(0).Mul(di, di))
		if den.Sign() == 0 {
			break
		}
		qr := nf(0).Quo(nf(0).Add(nf(0).Mul(pr, dr), nf(0).Mul(pi, di)), den)
		qi := nf(0).Quo(nf(0).Sub(nf(0).Mul(pi, dr), nf(0).Mul(pr, di)), den)
		zr.Sub(zr, qr)
		zi.Sub(zi, qi)
		step := nf(0).Add(nf(0).Mul(qr, qr), nf(0).Mul(qi, qi))
		mag := nf(0).Add(nf(0).Add(nf(0).Mul(zr, zr), nf(0).Mul(zi, zi)), nf(1e-200))
		if step.Sign() == 0 || step.Quo(step, mag).Cmp(tiny) < 0 {
			r, _ := zr.Float64()
			i, _ := zi.Float64()
			return complex(r, i), true
		}
	}
	return 0, false
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
