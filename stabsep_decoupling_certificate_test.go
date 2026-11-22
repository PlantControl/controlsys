package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// sepTolUnit is the normwise rounding accuracy of G(s) that Stabsep and
// Modsep's SepTol is relative to: u·(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sI − A)⁻¹, u = 4(n+5)ε, Frobenius norms, for the explicit (or
// E⁻¹-folded) model ref.
func sepTolUnit(t *testing.T, ref *System, s complex128) float64 {
	t.Helper()
	v, ok := sepTolUnitOK(ref, s, math.Inf(1))
	if !ok {
		t.Fatalf("sI − A singular at %v", s)
	}
	return v
}

// sepTolUnitOK is sepTolUnit; ok is false when sI − A is singular or its
// condition exceeds maxCond, beyond which the float inverse may misstate
// the bar.
func sepTolUnitOK(ref *System, s complex128, maxCond float64) (float64, bool) {
	n, _, _ := ref.Dims()
	K := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			v := -ref.A.At(i, j)
			if i == j {
				v += real(s)
				K.Set(i, n+j, -imag(s))
				K.Set(n+i, j, imag(s))
			}
			K.Set(i, j, v)
			K.Set(n+i, n+j, v)
		}
	}
	var inv mat.Dense
	if err := inv.Inverse(K); err != nil {
		return 0, false
	}
	mf := mat.Norm(&inv, 2) / math.Sqrt2
	if mf*mat.Norm(K, 2) > maxCond {
		return 0, false
	}
	fro := func(m *mat.Dense) float64 {
		if r, c := m.Dims(); r == 0 || c == 0 {
			return 0
		}
		return mat.Norm(m, 2)
	}
	a, b, c, d := fro(ref.A), fro(ref.B), fro(ref.C), fro(ref.D)
	return 4 * float64(n+5) * 0x1p-52 * (c*mf*a*mf*b + c*mf*b + d), true
}

// sepEval320 returns C(sE − A)⁻¹B + D in 320 bits (E = I when nil).
func sepEval320(t *testing.T, sys *System, s complex128) []complex128 {
	t.Helper()
	n, m, p := sys.Dims()
	out := make([]complex128, p*m)
	if n == 0 {
		for i := range p {
			for j := range m {
				out[i*m+j] = complex(sys.D.At(i, j), 0)
			}
		}
		return out
	}
	e := func(i, j int) float64 {
		if sys.E != nil {
			return sys.E.At(i, j)
		}
		if i == j {
			return 1
		}
		return 0
	}
	K := newHbMat(2*n, 2*n)
	for i := range n {
		for j := range n {
			re := new(big.Float).SetPrec(hbPrec).SetFloat64(real(s))
			re.Mul(re, new(big.Float).SetPrec(hbPrec).SetFloat64(e(i, j)))
			re.Sub(re, new(big.Float).SetPrec(hbPrec).SetFloat64(sys.A.At(i, j)))
			im := new(big.Float).SetPrec(hbPrec).SetFloat64(imag(s))
			im.Mul(im, new(big.Float).SetPrec(hbPrec).SetFloat64(e(i, j)))
			K.at(i, j).Set(re)
			K.at(n+i, n+j).Set(re)
			K.at(i, n+j).Neg(im)
			K.at(n+i, j).Set(im)
		}
	}
	rhs := newHbMat(2*n, m)
	for i := range n {
		for j := range m {
			rhs.at(i, j).SetFloat64(sys.B.At(i, j))
		}
	}
	x := hbSolve(K, rhs)
	pr := new(big.Float).SetPrec(hbPrec)
	for i := range p {
		for j := range m {
			re := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.D.At(i, j))
			im := new(big.Float).SetPrec(hbPrec)
			for k := range n {
				c := new(big.Float).SetPrec(hbPrec).SetFloat64(sys.C.At(i, k))
				re.Add(re, pr.Mul(c, x.at(k, j)))
				im.Add(im, pr.Mul(c, x.at(n+k, j)))
			}
			r, _ := re.Float64()
			q, _ := im.Float64()
			out[i*m+j] = complex(r, q)
		}
	}
	return out
}

// sepTolRatio returns max over pts of ‖G − G1 − G2‖_F / sepTolUnit, G and
// the parts evaluated in 320 bits; ref is the model the bar is taken on.
func sepTolRatio(t *testing.T, sys, ref, g1, g2 *System, pts []complex128) float64 {
	t.Helper()
	worst := 0.0
	for _, s := range pts {
		g, a, b := sepEval320(t, sys, s), sepEval320(t, g1, s), sepEval320(t, g2, s)
		var d l2
		for k := range g {
			d.addC(g[k] - a[k] - b[k])
		}
		worst = math.Max(worst, d.norm()/sepTolUnit(t, ref, s))
	}
	return worst
}

func sepLit(t *testing.T, n, m, p int, a, b, c, d []float64, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

var sepPointsC = []complex128{0.07i, 0.6i, 3i, complex(0.37, 0.09), complex(0.37, 0.78), complex(0.37, 3.9)}
var sepPointsD = []complex128{complex(0.9553, 0.2955), complex(0.4536, 0.8912), complex(-0.7374, 0.6755), complex(1.3, 0.4)}

// TestStabsepIllConditionedDecouplingErrors: tied/703 (AST4BK, XVYTBG S2;
// arm64 literal, n=7 discrete) split a near-defective cluster at |z| = 1
// with ‖X‖_F ≈ 7e10; base returned nil with ‖G−G1−G2‖ ≈ 6e6 times the
// SepTol unit. The decoupling certificate now rejects it.
func TestStabsepIllConditionedDecouplingErrors(t *testing.T) {
	sys := sepLit(t, 7, 2, 2, []float64{1.443190524382989, -0.050485035806474447, -0.26634193994854477, 0.85828945017646341, 0.64795477158400283, 0.36033784085867349, -0.15052878825209876, 1.1382397478984099, -0.60247796032727485, 1.0152146153729023, -0.29575290259434656, -0.72607595010588533, 0.3604816147716407, 1.165848964048005, -0.048998573615816991, 0.92762804129425758, 0.27556973726771189, -1.1314809441056688, 0.021955870293289607, -0.50252492920474778, -0.26156874993868223, 0.22260189466842425, -0.13566173328675235, -0.19213295264304947, 0.11927520848274449, -0.39221504165141591, -0.39308105473485122, 0.36197393681440709, 0.27131412148984024, -0.21002539679009077, 0.77167478395293376, -0.81549087903040496, -0.01029169434313223, -0.3175450711766804, -0.29425673968802485, 0.27632811597450202, -0.70522663918508066, 0.35905746739864702, 0.13201226335371194, -0.53919357479420549, 0.78777643320963497, 0.015474066324836657, 0.63985060030405638, -0.22795638728506321, -0.23751839695644258, -0.3014547078037671, -0.1796510261929623, -0.2833632135826985, 0.98695775132460106},
		[]float64{2, 0.5, 1, 0.5, 0.5, -1, 2, -1, 1, -1, -1, 0, 0.5, 1},
		[]float64{1, 0.5, 1, 0.5, -1, 1, 0, 0, -1, 1, 0, 2, 1, 0.5},
		[]float64{0, 0, 0, 0}, 0.5)
	res, err := Stabsep(sys)
	if !errors.Is(err, ErrSchurFailed) {
		r := -1.0
		if err == nil {
			r = sepTolRatio(t, sys, sys, res.Stable, res.Unstable, sepPointsD)
		}
		t.Fatalf("Stabsep err = %v (sum/SepTol unit %.3g), want ErrSchurFailed", err, r)
	}
}

// sepTriangularPair is A = Pᵀ·[−1 γ; 0 1]·P (P swaps the states), so the
// Schur form is T = [−1 γ; 0 1] after the swap and the decoupling is X = γ/2
// with weight ‖C̃1‖‖B̃2‖/(‖C‖‖B‖) = 0.8: P2 accepts ‖X‖ up to about 157.
func sepTriangularPair(t *testing.T, gamma float64) *System {
	return sepLit(t, 2, 1, 1, []float64{1, 0, gamma, -1}, []float64{1, 0.5}, []float64{-0.5, 1}, []float64{0}, 0)
}

// TestStabsepDecouplingBandBoundary: the certificate is a data-defined band
// (PZET7B ruling 3): coupling γ = 200 (‖X‖ = 100) is accepted with an
// exact-to-rounding sum; γ = 400 (‖X‖ = 200) is rejected although its split
// is accurate, an in-band honest error.
func TestStabsepDecouplingBandBoundary(t *testing.T) {
	sys := sepTriangularPair(t, 200)
	res, err := Stabsep(sys)
	if err != nil {
		t.Fatalf("Stabsep(γ=200) err = %v, want nil", err)
	}
	if r := sepTolRatio(t, sys, sys, res.Stable, res.Unstable, sepPointsC); r > 10.0/8 {
		t.Fatalf("Stabsep(γ=200) sum/SepTol unit = %.3g, want ≤ 1.25", r)
	}
	if _, err := Stabsep(sepTriangularPair(t, 400)); !errors.Is(err, ErrSchurFailed) {
		t.Fatalf("Stabsep(γ=400) err = %v, want ErrSchurFailed", err)
	}
}

// TestStabsepDecouplingCertificateBoundsExactRounding checks the error model
// behind P1 and P2 on random non-symmetric splits: in 320 bits, the exact
// residual T11·X − X·T22 + T12 of the computed X is within
// ‖R̂‖ + ½(n+1)ε((‖T11‖+‖T22‖)‖X‖ + ‖T12‖), and the rounding of
// B̃1 − X·B̃2 and C̃2 + C̃1·X within ½(n2+1)ε(‖B̃1‖ + ‖X‖‖B̃2‖) and
// ½(n1+1)ε(‖C̃2‖ + ‖C̃1‖‖X‖).
func TestStabsepDecouplingCertificateBoundsExactRounding(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	const e = 0x1p-52
	checked := 0
	for trial := range 40 {
		n, m, p := 3+trial%8, 1+trial%3, 1+trial%2
		a := make([]float64, n*n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		b, c := make([]float64, n*m), make([]float64, p*n)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		for i := range c {
			c[i] = rng.NormFloat64()
		}
		sys := sepLit(t, n, m, p, a, b, c, make([]float64, p*m), 0)
		tt, z, err := modalSchur(sys)
		if err != nil {
			t.Fatal(err)
		}
		n1, err := orderSchurGroupFirst(tt, z, n, func(l complex128) bool { return real(l) < 0 })
		if err != nil || n1 == 0 || n1 == n {
			continue
		}
		x, err := separateSchurBlocks(tt, n, n1, "test")
		if err != nil {
			continue
		}
		n2 := n - n1
		bt, ct := modalInputOutput(sys, z, n, m, p)
		cert := newDecouplingCertificate(sys, tt, x, bt, ct, n1)
		bf := func(v float64) *big.Float { return new(big.Float).SetPrec(hbPrec).SetFloat64(v) }
		fro := func(v []*big.Float) float64 {
			s := new(big.Float).SetPrec(hbPrec)
			for _, q := range v {
				s.Add(s, new(big.Float).SetPrec(hbPrec).Mul(q, q))
			}
			f, _ := s.Sqrt(s).Float64()
			return f
		}
		var rex []*big.Float
		for i := range n1 {
			for j := range n2 {
				s := bf(tt[i*n+n1+j])
				for k := range n1 {
					s.Add(s, new(big.Float).SetPrec(hbPrec).Mul(bf(tt[i*n+k]), bf(x[k*n2+j])))
				}
				for k := range n2 {
					s.Sub(s, new(big.Float).SetPrec(hbPrec).Mul(bf(x[i*n2+k]), bf(tt[(n1+k)*n+n1+j])))
				}
				rex = append(rex, s)
			}
		}
		if got, lim := fro(rex), cert.r+0.5*float64(n+1)*e*((cert.t11+cert.t22)*cert.x+cert.t12); got > lim {
			t.Fatalf("trial %d: exact residual %.3g > bound %.3g", trial, got, lim)
		}
		B1 := extractModalBlock(bt, m, 0, n1, 0, m)
		B2 := extractModalBlock(bt, m, n1, n, 0, m)
		C1 := extractModalBlock(ct, n, 0, p, 0, n1)
		C2 := extractModalBlock(ct, n, 0, p, n1, n)
		nb, nc := mat.DenseCopyOf(B1), mat.DenseCopyOf(C2)
		xGen := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: x}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, xGen, B2.RawMatrix(), 1, nb.RawMatrix())
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, C1.RawMatrix(), xGen, 1, nc.RawMatrix())
		checked++
		var db, dc []*big.Float
		for i := range n1 {
			for j := range m {
				s := bf(B1.At(i, j))
				for k := range n2 {
					s.Sub(s, new(big.Float).SetPrec(hbPrec).Mul(bf(x[i*n2+k]), bf(B2.At(k, j))))
				}
				db = append(db, s.Sub(s, bf(nb.At(i, j))))
			}
		}
		for i := range p {
			for j := range n2 {
				s := bf(C2.At(i, j))
				for k := range n1 {
					s.Add(s, new(big.Float).SetPrec(hbPrec).Mul(bf(C1.At(i, k)), bf(x[k*n2+j])))
				}
				dc = append(dc, s.Sub(s, bf(nc.At(i, j))))
			}
		}
		if got, lim := fro(db), 0.5*float64(n2+1)*e*(cert.b1+cert.x*cert.b2); got > lim {
			t.Fatalf("trial %d: rounding of B̃1 − X·B̃2 %.3g > bound %.3g", trial, got, lim)
		}
		if got, lim := fro(dc), 0.5*float64(n1+1)*e*(cert.c2+cert.c1*cert.x); got > lim {
			t.Fatalf("trial %d: rounding of C̃2 + C̃1·X %.3g > bound %.3g", trial, got, lim)
		}
	}
	if checked < 20 {
		t.Fatalf("only %d of 40 trials split", checked)
	}
}

// sepDyadic is A = Pᵀ·T·P for a fixed state permutation P and T upper
// triangular with dyadic data: diagonal alternating ±(1 + k/8), coupling
// 1/8 on the first two superdiagonals. With near set, the last two
// diagonal entries become −2⁻²⁰ and 2⁻²⁰ coupled by 1, so ‖X‖ ≈ 2¹⁹.
func sepDyadic(t *testing.T, n int, near bool) *System {
	T := make([]float64, n*n)
	for i := range n {
		v := 1 + float64(i%8)/8
		if i%2 == 1 {
			v = -v
		}
		T[i*n+i] = v
		for d := 1; d <= 2 && i+d < n; d++ {
			T[i*n+i+d] = 0.125
		}
	}
	if near {
		T[(n-2)*n+n-2], T[(n-1)*n+n-1] = -0x1p-20, 0x1p-20
		T[(n-2)*n+n-1] = 1
	}
	perm := make([]int, n)
	for i := range n {
		perm[i] = (i*7 + 3) % n
	}
	if n%7 == 0 {
		for i := range n {
			perm[i] = (i*5 + 3) % n
		}
	}
	a := make([]float64, n*n)
	for i := range n {
		for j := range n {
			a[perm[i]*n+perm[j]] = T[i*n+j]
		}
	}
	b, c := make([]float64, n), make([]float64, n)
	for i := range n {
		b[i] = float64(i%3) - 0.5
		c[i] = 1 - float64(i%4)/4
	}
	return sepLit(t, n, 1, 1, a, b, c, []float64{0.25}, 0)
}

// TestStabsepLargeDyadicDecoupling: n = 16, 32, 48 well-separated splits stay
// nil with G1 + G2 within SepTol/8; adding a ±2⁻²⁰ pair coupled across the
// boundary (‖X‖ ≈ 2¹⁹) is rejected.
func TestStabsepLargeDyadicDecoupling(t *testing.T) {
	for _, n := range []int{16, 32, 48} {
		sys := sepDyadic(t, n, false)
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("n=%d: Stabsep err = %v, want nil", n, err)
		}
		if r := sepTolRatio(t, sys, sys, res.Stable, res.Unstable, sepPointsC[:3]); r > 10.0/8 {
			t.Fatalf("n=%d: sum/SepTol unit = %.3g, want ≤ 1.25", n, r)
		}
		if _, err := Stabsep(sepDyadic(t, n, true)); !errors.Is(err, ErrSchurFailed) {
			t.Fatalf("n=%d near pair: Stabsep err = %v, want ErrSchurFailed", n, err)
		}
	}
}

// TestStabsepFoldRouteDecoupling: a well-conditioned nonsingular E is folded
// (E⁻¹A) and gets the same certificate: nil within SepTol of the folded
// model's bar, and the near-pair variant rejected.
func TestStabsepFoldRouteDecoupling(t *testing.T) {
	for _, near := range []bool{false, true} {
		ex := sepDyadic(t, 6, near)
		n := 6
		E := mat.NewDense(n, n, nil)
		for i := range n {
			E.Set(i, i, 2)
			if i+1 < n {
				E.Set(i, i+1, 0.5)
			}
		}
		var A mat.Dense
		A.Mul(E, ex.A)
		var B mat.Dense
		B.Mul(E, ex.B)
		sys, err := NewDescriptor(&A, &B, ex.C, ex.D, E, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Stabsep(sys)
		if near {
			if !errors.Is(err, ErrSchurFailed) {
				t.Fatalf("fold near pair: Stabsep err = %v, want ErrSchurFailed", err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("fold: Stabsep err = %v, want nil", err)
		}
		ref, err := sys.ToExplicit()
		if err != nil {
			t.Fatal(err)
		}
		if r := sepTolRatio(t, sys, ref, res.Stable, res.Unstable, sepPointsC[:3]); r > 10.0/8 {
			t.Fatalf("fold: sum/SepTol unit = %.3g, want ≤ 1.25", r)
		}
	}
}

// TestStabsepDecouplingCertificateTerms pins each term of the accept rule on
// hand-set norms (n = 6, n1 = n2 = 3, unit A, B, C, X = 1, so P2 alone
// accepts): a residual ‖R̂‖ beyond the P1 budget β·U·ε·‖A‖/(‖C̃1‖‖B̃2‖)
// rejects, as do a non-finite X, R̂ or updated block.
func TestStabsepDecouplingCertificateTerms(t *testing.T) {
	const e = 0x1p-52
	base := decouplingCertificate{n: 6, n1: 3, n2: 3, a: 1, b: 1, c: 1, t11: 0.5, t22: 0.5, t12: 0.5, x: 1, b1: 0.5, b2: 0.5, c1: 0.5, c2: 0.5}
	finite := []float64{1, 2}
	if !base.accept(finite, finite) {
		t.Fatal("unit decoupling rejected")
	}
	budget := 9 * 4 * float64(6+5) * e / (0.5 * 0.5)
	for _, tc := range []struct {
		name   string
		r      float64
		accept bool
	}{{"residual at half the P1 budget", budget / 2, true}, {"residual at twice the P1 budget", 2 * budget, false}} {
		d := base
		d.r = tc.r
		if got := d.accept(finite, finite); got != tc.accept {
			t.Errorf("%s: accept = %v, want %v", tc.name, got, tc.accept)
		}
	}
	inf := math.Inf(1)
	for _, tc := range []struct {
		name   string
		d      decouplingCertificate
		b1, c2 []float64
	}{
		{"X", func() decouplingCertificate { d := base; d.x = inf; return d }(), finite, finite},
		{"residual", func() decouplingCertificate { d := base; d.r = math.NaN(); return d }(), finite, finite},
		{"B̃1 − X·B̃2", base, []float64{1, inf}, finite},
		{"C̃2 + C̃1·X", base, finite, []float64{math.NaN()}},
	} {
		if tc.d.accept(tc.b1, tc.c2) {
			t.Errorf("non-finite %s accepted", tc.name)
		}
	}
}
