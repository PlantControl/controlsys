package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dcGainExactRat returns X* = (x0·E − A)⁻¹B and G* = D + C·X* of the float
// model in exact arithmetic, E = I for an explicit model and dropping out at
// x0 = 0; ok is false where x0·E − A is exactly singular.
func dcGainExactRat(sys *System) (x, g []*big.Rat, ok bool) {
	n, m, p := sys.Dims()
	x0 := 0.0
	if sys.IsDiscrete() {
		x0 = 1
	}
	w := n + m
	M := make([]*big.Rat, n*w)
	for i := range n {
		for j := range n {
			v := new(big.Rat).Neg(new(big.Rat).SetFloat64(sys.A.At(i, j)))
			switch {
			case x0 != 0 && sys.E != nil:
				v.Add(v, new(big.Rat).SetFloat64(x0*sys.E.At(i, j)))
			case i == j:
				v.Add(v, new(big.Rat).SetFloat64(x0))
			}
			M[i*w+j] = v
		}
		for j := range m {
			M[i*w+n+j] = new(big.Rat).SetFloat64(sys.B.At(i, j))
		}
	}
	for k := range n {
		piv := -1
		for i := k; i < n; i++ {
			if M[i*w+k].Sign() != 0 {
				piv = i
				break
			}
		}
		if piv < 0 {
			return nil, nil, false
		}
		for j := range w {
			M[k*w+j], M[piv*w+j] = M[piv*w+j], M[k*w+j]
		}
		inv := new(big.Rat).Inv(M[k*w+k])
		for j := k; j < w; j++ {
			M[k*w+j].Mul(M[k*w+j], inv)
		}
		for i := range n {
			if i == k || M[i*w+k].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(M[i*w+k])
			for j := k; j < w; j++ {
				M[i*w+j].Sub(M[i*w+j], new(big.Rat).Mul(f, M[k*w+j]))
			}
		}
	}
	x = make([]*big.Rat, n*m)
	for i := range n {
		for j := range m {
			x[i*m+j] = M[i*w+n+j]
		}
	}
	g = make([]*big.Rat, p*m)
	for i := range p {
		for j := range m {
			s := new(big.Rat)
			if sys.D != nil {
				s.SetFloat64(sys.D.At(i, j))
			}
			for l := range n {
				s.Add(s, new(big.Rat).Mul(new(big.Rat).SetFloat64(sys.C.At(i, l)), x[l*m+j]))
			}
			g[i*m+j] = s
		}
	}
	return x, g, true
}

// dcGainOwnRoundingExcess returns the largest |g_ij − G*_ij| − τ_ij − ½ulp(g_ij)
// over the entries, τ_ij = ε·(|D_ij| + Σ_l |C_il||X*_lj|) the rounding of
// the entry's own terms (positive: the nil promise of the regular DC path is
// broken), and ok false where x0·E − A is exactly singular.
func dcGainOwnRoundingExcess(sys *System, g *mat.Dense) (excess float64, ok bool) {
	x, gs, ok := dcGainExactRat(sys)
	if !ok {
		return 0, false
	}
	n, m, p := sys.Dims()
	eps := new(big.Rat).SetFloat64(0x1p-52)
	excess = math.Inf(-1)
	for i := range p {
		for j := range m {
			tau := new(big.Rat)
			if sys.D != nil {
				tau.SetFloat64(math.Abs(sys.D.At(i, j)))
			}
			for l := range n {
				t := new(big.Rat).Mul(new(big.Rat).SetFloat64(sys.C.At(i, l)), x[l*m+j])
				tau.Add(tau, t.Abs(t))
			}
			tau.Mul(tau, eps)
			v := g.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return math.Inf(1), true
			}
			av := math.Abs(v)
			halfUlp := new(big.Rat).SetFloat64((math.Nextafter(av, math.Inf(1)) - av) / 2)
			d := new(big.Rat).Sub(new(big.Rat).SetFloat64(v), gs[i*m+j])
			d.Abs(d)
			d.Sub(d, tau)
			d.Sub(d, halfUlp)
			f, _ := d.Float64()
			if d.Sign() > 0 && f == 0 {
				f = math.SmallestNonzeroFloat64
			}
			excess = max(excess, f)
		}
	}
	return excess, true
}

// dcGainRegularPath reports that DCGain's regular path solves this model:
// the double LU of x0·I − A is nonsingular within gonum's condition tolerance.
func dcGainRegularPath(sys *System) bool {
	n, _, _ := sys.Dims()
	x0 := 0.0
	if sys.IsDiscrete() {
		x0 = 1
	}
	P := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v := -sys.A.At(i, j)
			if i == j {
				v += x0
			}
			P.Set(i, j, v)
		}
	}
	var lu mat.LU
	lu.Factorize(P)
	var X mat.Dense
	return lu.SolveTo(&X, false, sys.B) == nil
}

func dcGainCertTestSystem(t *testing.T, n, m, p int, a, b, c, d []float64, dt float64) *System {
	t.Helper()
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// The regular DC solve of an ill-conditioned but nonsingular pencil used to
// return the double LU's value with a nil error, off the exact gain by
// κ·ε (HUUZRO; INGGF2 census rows F5illCondBlockBesideStiff#1167, κ̂ 7.8e15,
// 7.5e-6 off, and F8rotatedIntegrator#1903, κ̂ 3.1e15, 3% off). A nil error
// now means the exact gain to within the rounding of the entry's own terms.
func TestDCGainRegularPathIllConditionedLiterals(t *testing.T) {
	cases := []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
	}{
		{
			name: "ill-conditioned block beside stiff mode",
			n:    3, m: 1, p: 2,
			a: []float64{
				1.3497740196913073, 0.93067873678338098, 1.2212053307990518,
				0.037718714433019877, 1.1003619580807171, 0.54860351133916163,
				0, 0, -2771.7994679810026,
			},
			b: []float64{1.1113509093747056, 0.23991469126786491, 0.18309386577967279},
			c: []float64{
				-1.1277700615925483, 1.7536387452174069, 1.7619508201267953,
				-1.7339757794992412, 1.2718959111262316, -1.8496943939808626,
			},
			d: []float64{0.63011436284418254, 0.25133262054129979},
		},
		{
			name: "rotated integrator rounded off z = 1",
			n:    2, m: 1, p: 2,
			a: []float64{0.9183286800470255, 0.14494764584955158, 0.14494764584955175, 0.74275155526536929},
			b: []float64{1.2651634973407024, 0.26373911059666044},
			c: []float64{1.4883847101164247, -0.48649579099338913, 0.62595084562522285, 1.375548030570769},
			d: []float64{-0.54979253209163303, 1.8405202357220651},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := dcGainCertTestSystem(t, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, 0.1)
			if !dcGainRegularPath(sys) {
				t.Fatal("model no longer reaches the regular DC path")
			}
			g, err := sys.DCGain()
			if err != nil {
				t.Fatalf("DCGain error %v, want nil", err)
			}
			ex, ok := dcGainOwnRoundingExcess(sys, g)
			if !ok {
				t.Fatal("x0·I − A exactly singular")
			}
			if ex > 0 {
				t.Fatalf("DCGain = %v exceeds the exact gain's own rounding by %g", mat.Formatted(g), ex)
			}
		})
	}
}

// Seeded ill-conditioned models on the regular path (the INGGF2 F5 and F8
// families, HUUZRO): every nil result is the exact gain within the entry's
// own rounding; an error is ErrAccuracyUnverified only beyond order 16.
func TestDCGainRegularPathIllConditionedSample(t *testing.T) {
	r := rand.New(rand.NewPCG(1167, 1903))
	pow := func(lo, hi float64) float64 { return math.Pow(10, lo+r.Float64()*(hi-lo)) }
	randn := func(k int) []float64 {
		v := make([]float64, k)
		for i := range v {
			v[i] = r.NormFloat64()
		}
		return v
	}
	checked, certified := 0, 0
	for k := range 1200 {
		var n int
		disc := r.IntN(2) == 0
		x0 := 0.0
		if disc {
			x0 = 1
		}
		var a []float64
		if k%2 == 0 {
			n = 3 + r.IntN(4)
			a = make([]float64, n*n)
			tc := pow(10, 15.7)
			s1, c1 := math.Sincos(r.Float64() * 6.28)
			s2, c2 := math.Sincos(r.Float64() * 6.28)
			u := [2][2]float64{{c1, -s1}, {s1, c1}}
			v := [2][2]float64{{c2, -s2}, {s2, c2}}
			for i := range 2 {
				for j := range 2 {
					dij := 0.0
					if i == j {
						dij = x0
					}
					a[i*n+j] = dij - (u[i][0]*v[j][0] + u[i][1]/tc*v[j][1])
				}
			}
			for i := 2; i < n; i++ {
				a[i*n+i] = x0 - pow(0, 8)
				for j := i + 1; j < n; j++ {
					a[i*n+j] = r.NormFloat64()
				}
				a[i] = r.NormFloat64()
				a[n+i] = r.NormFloat64()
			}
		} else {
			n = 2 + r.IntN(4)
			a = make([]float64, n*n)
			a[0] = x0
			for i := 1; i < n; i++ {
				a[i*n+i] = x0 - pow(-1, 2)
			}
			for range 2 * n {
				p1, p2 := r.IntN(n), r.IntN(n)
				if p1 == p2 {
					continue
				}
				s, c := math.Sincos(r.Float64() * 6.28)
				for j := range n {
					x, y := a[p1*n+j], a[p2*n+j]
					a[p1*n+j], a[p2*n+j] = c*x-s*y, s*x+c*y
				}
				for i := range n {
					x, y := a[i*n+p1], a[i*n+p2]
					a[i*n+p1], a[i*n+p2] = c*x-s*y, s*x+c*y
				}
			}
		}
		m, p := 1+r.IntN(2), 1+r.IntN(2)
		dt := 0.0
		if disc {
			dt = 0.1
		}
		sys := dcGainCertTestSystem(t, n, m, p, a, randn(n*m), randn(p*n), randn(p*m), dt)
		if !dcGainRegularPath(sys) {
			continue
		}
		checked++
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("model %d: DCGain error %v on the regular path at order %d", k, err, n)
		}
		ex, ok := dcGainOwnRoundingExcess(sys, g)
		if !ok {
			continue
		}
		certified++
		if ex > 0 {
			t.Fatalf("model %d: DCGain = %v exceeds the exact gain's own rounding by %g", k, mat.Formatted(g), ex)
		}
	}
	if checked < 100 || certified < 100 {
		t.Fatalf("only %d models on the regular path, %d checked", checked, certified)
	}
}

// Well-conditioned models keep a nil error on the regular path, within the
// entry's own rounding of the exact gain (HUUZRO: no nil-correct row moves).
func TestDCGainRegularPathWellConditionedStaysNil(t *testing.T) {
	r := rand.New(rand.NewPCG(17026, 962))
	for k := range 200 {
		n, m, p := 1+r.IntN(8), 1+r.IntN(3), 1+r.IntN(3)
		disc := k%2 == 1
		a := make([]float64, n*n)
		for i := range a {
			a[i] = 0.3 * r.NormFloat64() / math.Sqrt(float64(n))
		}
		for i := range n {
			if disc {
				a[i*n+i] *= 0.5
			} else {
				a[i*n+i] -= 1 + r.Float64()
			}
		}
		b, c, d := make([]float64, n*m), make([]float64, p*n), make([]float64, p*m)
		for _, v := range [][]float64{b, c, d} {
			for i := range v {
				v[i] = r.NormFloat64()
			}
		}
		dt := 0.0
		if disc {
			dt = 0.1
		}
		sys := dcGainCertTestSystem(t, n, m, p, a, b, c, d, dt)
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("model %d: DCGain error %v", k, err)
		}
		if ex, ok := dcGainOwnRoundingExcess(sys, g); !ok || ex > 0 {
			t.Fatalf("model %d: DCGain = %v exceeds the exact gain's own rounding by %g (ok %v)", k, mat.Formatted(g), ex, ok)
		}
	}
}

// Beyond the exact check's order, a regular-path gain the certificate cannot
// prove keeps its refined value with ErrAccuracyUnverified (HUUZRO).
func TestDCGainRegularPathUncertifiedLargeOrder(t *testing.T) {
	const n = exactDetMaxOrder + 1
	a := make([]float64, n*n)
	s1, c1 := math.Sincos(0.7)
	s2, c2 := math.Sincos(2.1)
	u := [2][2]float64{{c1, -s1}, {s1, c1}}
	v := [2][2]float64{{c2, -s2}, {s2, c2}}
	for i := range 2 {
		for j := range 2 {
			a[i*n+j] = -(u[i][0]*v[j][0] + u[i][1]*1e-14*v[j][1])
		}
	}
	for i := 2; i < n; i++ {
		a[i*n+i] = -float64(i)
		a[i] = 0.5
	}
	b, c, d := make([]float64, n), make([]float64, n), []float64{0.25}
	for i := range n {
		b[i] = 1 + 0.1*float64(i)
		c[i] = 1 - 0.05*float64(i)
	}
	sys := dcGainCertTestSystem(t, n, 1, 1, a, b, c, d, 0)
	if !dcGainRegularPath(sys) {
		t.Fatal("model no longer reaches the regular DC path")
	}
	g, err := sys.DCGain()
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("DCGain error %v, want ErrAccuracyUnverified", err)
	}
	if g == nil || !allFiniteDense(g) {
		t.Fatalf("DCGain = %v, want a finite refined value", g)
	}
	_, gs, ok := dcGainExactRat(sys)
	if !ok {
		t.Fatal("x0·I − A exactly singular")
	}
	want, _ := gs[0].Float64()
	if got := g.At(0, 0); !(math.Abs(got-want) <= 1e-6*math.Abs(want)) {
		t.Fatalf("DCGain = %v, exact %v", got, want)
	}
}
