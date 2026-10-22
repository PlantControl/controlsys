package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// exactTFCoeffs returns the numerator C·adj(sI−A)·B and the monic det(sI−A)
// of the float data in exact rational arithmetic (Faddeev–LeVerrier).
func exactTFCoeffs(a, b, c []float64, n int) (num, den []float64) {
	rat := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	A := make([]*big.Rat, n*n)
	M := make([]*big.Rat, n*n)
	for i := range A {
		A[i], M[i] = rat(a[i]), new(big.Rat)
	}
	for i := range n {
		M[i*n+i].SetInt64(1)
	}
	cmb := func() float64 {
		s := new(big.Rat)
		for i := range n {
			for j := range n {
				t := new(big.Rat).Mul(rat(c[i]), M[i*n+j])
				s.Add(s, t.Mul(t, rat(b[j])))
			}
		}
		f, _ := s.Float64()
		return f
	}
	den = []float64{1}
	num = []float64{cmb()}
	for k := 1; k <= n; k++ {
		AM := make([]*big.Rat, n*n)
		tr := new(big.Rat)
		for i := range n {
			for j := range n {
				s := new(big.Rat)
				for l := range n {
					s.Add(s, new(big.Rat).Mul(A[i*n+l], M[l*n+j]))
				}
				AM[i*n+j] = s
			}
			tr.Add(tr, AM[i*n+i])
		}
		ak := tr.Quo(tr, big.NewRat(int64(-k), 1))
		f, _ := ak.Float64()
		den = append(den, f)
		if k == n {
			break
		}
		for i := range n {
			AM[i*n+i].Add(AM[i*n+i], ak)
		}
		M = AM
		num = append(num, cmb())
	}
	return num, den
}

var fastModeS = []complex128{0.3i, 1 + 2i, 10i, -0.2}

func requireFastModeTF(t *testing.T, name string, sys *System, want func(complex128) complex128, tol float64) *TransferFunc {
	t.Helper()
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	zpk, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatalf("%s: ZPKModel: %v", name, err)
	}
	for _, s := range fastModeS {
		w := want(s)
		g, _ := res.TF.Eval(s)
		z, _ := zpk.ZPK.Eval(s)
		if e := cmplx.Abs(g[0][0]-w) / cmplx.Abs(w); e > tol {
			t.Errorf("%s: TF(%v) rel err %.3g, num %v", name, s, e, res.TF.Num[0][0])
		}
		if e := cmplx.Abs(z[0][0]-w) / cmplx.Abs(w); e > tol {
			t.Errorf("%s: ZPK(%v) rel err %.3g", name, s, e)
		}
	}
	return res.TF
}

// A slow plant fed by a fast actuator, F(s+0.5)/((s+F)·∏(s+k)), in a dense
// orthogonal basis: its Markov parameters C·A^k·B, k ≥ 3, are dominated by
// the rounding of A while the numerator coefficients are resolved.
func TestTransferFunctionFastActuatorDenseBasis(t *testing.T) {
	a := []float64{
		-1196.0113459802285, -2767.5980955753362, 930.99236609671868, 1128.4211480064171, -325.45196053167297, -169.26147476120457,
		-2799.5956378811561, -7184.1185662964799, 2255.5592313259904, 2382.6038834796282, -531.55808587510717, -964.04360808488775,
		946.86027202620176, 2225.2832726613069, -741.47215788363553, -880.0126741133306, 245.39457141283344, 162.48451562796996,
		981.09904465438649, 2474.3687057785055, -786.14356224876474, -852.14550825565311, 200.83350613748078, 301.6315306648375,
		-210.35671222455125, -613.85962633409133, 176.88524504871825, 151.41350333433044, -15.592967905714689, -133.32472266342964,
		-371.09757892625657, -825.17489594667768, 284.85708995486317, 362.57723778518357, -110.8601789353085, -25.659453678290813,
	}
	b := []float64{3361.5397887219083, 8449.3791235735698, -2689.4136553615022, -2925.8141010443533, 695.63261963083619, 1015.2851668123957}
	c := []float64{0.20567693983991436, -0.31193255000543874, -0.074184135220376632, -0.46051371881104997, 0.91505165655557519, -0.23558305407756486}
	sys, err := NewFromSlices(6, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	const F = 1e4
	want := func(s complex128) complex128 {
		v := complex(F, 0) * (s + 0.5) / (s + F)
		for k := 1; k <= 5; k++ {
			v /= s + complex(float64(k), 0)
		}
		return v
	}
	tf := requireFastModeTF(t, "fast actuator", sys, want, 1e-8)

	exNum, exDen := exactTFCoeffs(a, b, c, 6)
	num := tf.Num[0][0]
	if len(num) < 2 {
		t.Fatalf("num %v lost its s coefficient", num)
	}
	for k := 1; k <= 2; k++ {
		got, ex := num[len(num)-k], exNum[len(exNum)-k]
		if math.Abs(got-ex) > 1e-8*math.Abs(ex) {
			t.Errorf("num s^%d coeff %v, exact %v", k-1, got, ex)
		}
	}
	for _, s := range fastModeS {
		ex := Poly(exNum).Eval(s) / Poly(exDen).Eval(s)
		g, _ := tf.Eval(s)
		if e := cmplx.Abs(g[0][0]-ex) / cmplx.Abs(ex); e > 1e-8 {
			t.Errorf("TF(%v) rel err %.3g vs exact rational TF of the data", s, e)
		}
	}
}

// fastModePlant is the 5-state companion of num/∏(s+k), k = 1…5, with a
// state ẋ_f = −F·x_f, either a series actuator F/(s+F) or decoupled from
// input and output, in a seeded random orthogonal basis.
func fastModePlant(t *testing.T, F float64, zeros []float64, series bool) (*System, func(complex128) complex128) {
	t.Helper()
	const np, n = 5, 6
	num, den := Poly{1}, Poly{1}
	for _, z := range zeros {
		num = num.Mul(Poly{1, z})
	}
	for k := 1; k <= np; k++ {
		den = den.Mul(Poly{1, float64(k)})
	}
	a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range np - 1 {
		a.Set(i, i+1, 1)
	}
	for j := range np {
		a.Set(np-1, j, -den[np-j])
	}
	for j := range num {
		c.Set(0, j, num[len(num)-1-j])
	}
	a.Set(np, np, -F)
	if series {
		a.Set(np-1, np, 1)
		b.Set(np, 0, F)
	} else {
		b.Set(np-1, 0, 1)
	}
	rng := rand.New(rand.NewPCG(5, 5))
	g := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(g)
	var q, qa, aq, qb, cq mat.Dense
	qr.QTo(&q)
	qa.Mul(q.T(), a)
	aq.Mul(&qa, &q)
	qb.Mul(q.T(), b)
	cq.Mul(c, &q)
	sys, err := New(&aq, &qb, &cq, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys, func(s complex128) complex128 {
		v := num.Eval(s) / den.Eval(s)
		if series {
			v *= complex(F, 0) / (s + complex(F, 0))
		}
		return v
	}
}

func TestTransferFunctionFastModeDenseBasis(t *testing.T) {
	for _, tc := range []struct {
		name   string
		F      float64
		zeros  []float64
		series bool
	}{
		{"r4 F5e3", 5e3, []float64{0.5}, true},
		{"r4 F1e4", 1e4, []float64{0.5}, true},
		{"r3 F1e5", 1e5, []float64{0.5, 2.5}, true},
		{"r3 F1e6", 1e6, []float64{0.5, 2.5}, true},
		{"r4 hidden F1e5", 1e5, []float64{0.5}, false},
	} {
		sys, want := fastModePlant(t, tc.F, tc.zeros, tc.series)
		requireFastModeTF(t, tc.name, sys, want, 1e-6)
	}
}

// MATLAB roots takes the eigenvalues of compan(p), coefficients in the first
// row; in the last column Dlanv2 cancels on a 2×2 with a tiny leading
// coefficient and loses the small root.
func TestPolyRootsTinyLeadingCoefficient(t *testing.T) {
	for _, tc := range []struct {
		p     Poly
		small float64
	}{
		{Poly{1.55e-17, 1, 0.15444209679158707}, -0.15444209679158707},
		{Poly{-6.56365572663833e-10, -113.19388865955423, 3.383839253286235}, 0.029894186809528187},
	} {
		r, err := tc.p.Roots()
		if err != nil {
			t.Fatal(err)
		}
		if len(r) != 2 {
			t.Fatalf("%v: roots %v", tc.p, r)
		}
		small := r[0]
		if cmplx.Abs(r[1]) < cmplx.Abs(small) {
			small = r[1]
		}
		if cmplx.Abs(small-complex(tc.small, 0)) > 1e-12*math.Abs(tc.small) {
			t.Errorf("%v: small root %v, want %v", tc.p, small, tc.small)
		}
	}
}
