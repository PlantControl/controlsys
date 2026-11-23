package controlsys

import (
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// benchDescriptorOrth is a seeded random orthogonal n×n matrix.
func benchDescriptorOrth(rng *rand.Rand, n int) *mat.Dense {
	g := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			g.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(g)
	var q mat.Dense
	qr.QTo(&q)
	return &q
}

// benchDenseSingularE is U·(diag(Q·T·Qᵀ, 1), diag(I, 0))·V with T upper
// triangular (half the modes stable, coupling sigma), B, C dense: a dense
// singular-E descriptor of n+1 states split by the generalized Schur form.
func benchDenseSingularE(seed uint64, n int, sigma float64) *System {
	rng := rand.New(rand.NewPCG(seed, 0xbe7c))
	T := mat.NewDense(n, n, nil)
	for i := range n {
		d := -(0.5 + 1.5*rng.Float64())
		if i%2 == 1 {
			d = 0.5 + 1.5*rng.Float64()
		}
		T.Set(i, i, d)
		for j := i + 1; j < n; j++ {
			T.Set(i, j, sigma*rng.NormFloat64())
		}
	}
	q := benchDescriptorOrth(rng, n)
	var tmp, af mat.Dense
	tmp.Mul(q, T)
	af.Mul(&tmp, q.T())
	N := n + 1
	A := mat.NewDense(N, N, nil)
	E := mat.NewDense(N, N, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(&af)
	A.Set(n, n, 1)
	for i := range n {
		E.Set(i, i, 1)
	}
	B := mat.NewDense(N, 2, nil)
	C := mat.NewDense(2, N, nil)
	for i := range N {
		for j := range 2 {
			B.Set(i, j, rng.NormFloat64())
			C.Set(j, i, rng.NormFloat64())
		}
	}
	u, v := benchDescriptorOrth(rng, N), benchDescriptorOrth(rng, N)
	var a2, e2, b2, c2 mat.Dense
	tmp.Reset()
	tmp.Mul(u, A)
	a2.Mul(&tmp, v)
	tmp.Reset()
	tmp.Mul(u, E)
	e2.Mul(&tmp, v)
	b2.Mul(u, B)
	c2.Mul(C, v)
	sys, err := NewDescriptor(&a2, &b2, &c2, mat.NewDense(2, 2, nil), &e2, 0)
	if err != nil {
		panic(err)
	}
	return sys
}

func benchDescriptorSplit(b *testing.B, sys *System, modsep bool) {
	b.ReportAllocs()
	for b.Loop() {
		var err error
		if modsep {
			_, err = Modsep(sys, 1)
		} else {
			_, err = Stabsep(sys)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStabsep_DescriptorDense_N10(b *testing.B) {
	benchDescriptorSplit(b, benchDenseSingularE(1, 10, 0.3), false)
}

func BenchmarkStabsep_DescriptorDense_N50(b *testing.B) {
	benchDescriptorSplit(b, benchDenseSingularE(2, 50, 0.3), false)
}

func BenchmarkModsep_DescriptorDense_N10(b *testing.B) {
	benchDescriptorSplit(b, benchDenseSingularE(3, 10, 0.3), true)
}

func BenchmarkModsep_DescriptorDense_N50(b *testing.B) {
	benchDescriptorSplit(b, benchDenseSingularE(4, 50, 0.1), true)
}

// BenchmarkStabsep_DescriptorBadlyScaled is the XACVXD pencil (seed 0 of
// TestStabsep_DescriptorRejectedQZReorder).
func BenchmarkStabsep_DescriptorBadlyScaled(b *testing.B) {
	h := []float64{
		11.475236504468711, -4.802980538727953, 0, 0,
		-4.804997870973105e-16, 1.883135785961934e-16, 1, 0,
		1.883135785961934e-16, -7.38023300653887e-17, 0, 1,
		0.0004680437192265164, 0.00021592881520019866, 0, 0,
	}
	j := []float64{
		-0.9260172649172378, 0.37748115799546883, 0.00065672900631252, 0.00161105365515163,
		0, 0, -13.418168006075515, -2.5172613588536494,
		0, 0, 0.597370003150422, -11.258325117032756,
		-3.097492448978087e-05, -2.0823017944789118e-05, -31.53860826496756, -77.36888371806144,
	}
	rng := rand.New(rand.NewPCG(0, 9))
	A, E := mat.NewDense(5, 5, nil), mat.NewDense(5, 5, nil)
	for i := range 4 {
		for k := range 4 {
			A.Set(i, k, h[i*4+k]*(1+1e-12*rng.NormFloat64()))
		}
	}
	for i := range 4 {
		for k := range 4 {
			E.Set(i, k, j[i*4+k]*(1+1e-12*rng.NormFloat64()))
		}
	}
	A.Set(4, 4, 1)
	B, C := mat.NewDense(5, 1, nil), mat.NewDense(1, 5, nil)
	for i := range 5 {
		B.Set(i, 0, 1)
		C.Set(0, i, 1)
	}
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 1)
	if err != nil {
		b.Fatal(err)
	}
	benchDescriptorSplit(b, sys, false)
}

// BenchmarkStabsep_DescriptorSecondTier_N50 is a dense singular-E model
// whose coupling the first tier of the descriptor certificate rejects and
// the ridge tier certifies.
// (seed 7, coupling 0.6 on both arches when written).
func BenchmarkStabsep_DescriptorSecondTier_N50(b *testing.B) {
	benchDescriptorSplit(b, benchDenseSingularE(7, 50, 0.6), false)
}
