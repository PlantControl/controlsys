package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// TestStabsep_DescriptorRejectedQZReorder: Stabsep of the discrete
// descriptor A = blkdiag(H, 1), E = blkdiag(J, 0), with (H, J) the
// compressed 4×4 Dare pencil of gonum ZHGFWA (Netlib DTGEX2 parity)
// perturbed by 1e-12 relative, returned ErrSchurFailed for 118 of 200
// seeds: Dgges rejects the 2×2/2×2 swap although the eigenvalue moduli
// 0.081 and 12.35 are well separated. The oracle is the eigenvalues of
// J⁻¹H and the transfer function of the descriptor. XACVXD.
func TestStabsep_DescriptorRejectedQZReorder(t *testing.T) {
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
	ones := func(r, c int) *mat.Dense {
		m := mat.NewDense(r, c, nil)
		for i := range r {
			for k := range c {
				m.Set(i, k, 1)
			}
		}
		return m
	}
	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, 9))
		A, E := mat.NewDense(5, 5, nil), mat.NewDense(5, 5, nil)
		H, J := mat.NewDense(4, 4, nil), mat.NewDense(4, 4, nil)
		for i := range 4 {
			for k := range 4 {
				H.Set(i, k, h[i*4+k]*(1+1e-12*rng.NormFloat64()))
			}
		}
		for i := range 4 {
			for k := range 4 {
				J.Set(i, k, j[i*4+k]*(1+1e-12*rng.NormFloat64()))
			}
		}
		A.Slice(0, 4, 0, 4).(*mat.Dense).Copy(H)
		E.Slice(0, 4, 0, 4).(*mat.Dense).Copy(J)
		A.Set(4, 4, 1)
		sys, err := NewDescriptor(A, ones(5, 1), ones(1, 5), mat.NewDense(1, 1, nil), E, 1)
		if err != nil {
			t.Fatal(err)
		}

		var want, wantUnstable []complex128
		for _, v := range generalizedEigOracle(t, H, J) {
			if cmplx.Abs(v) < 1 {
				want = append(want, v)
			} else {
				wantUnstable = append(wantUnstable, v)
			}
		}
		if len(want) != 2 {
			t.Fatalf("seed %d: oracle %v", seed, want)
		}

		res, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		poles, err := res.Stable.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, "stable poles", poles, want, 1e-9)
		var finite []complex128
		up, err := res.Unstable.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range up {
			if !cmplx.IsInf(v) && cmplx.Abs(v) < 1e8 {
				finite = append(finite, v)
			}
		}
		assertEigSetNear(t, "unstable poles", finite, wantUnstable, 1e-9)
		assertSplitSum(t, "Stabsep rejected reorder", sys, res.Stable, res.Unstable)
		if t.Failed() {
			t.Fatalf("seed %d", seed)
		}
	}
}

// TestQZRetryConsistent: the reciprocal QZ retry of orderedQZ is kept only
// when its eigenvalues match the first run's one to one, each selected
// alike; a moved eigenvalue, a selection flipped across the boundary
// within qzRetryTol, or a doubled match fails closed.
func TestQZRetryConsistent(t *testing.T) {
	inside := func(ar, ai, b float64) bool { return math.Hypot(ar, ai) < b }
	first := []float64{
		0.5, 0.5, 1 - 1e-10, 3, // α_r
		0.25, -0.25, 0, 0, // α_i
		1, 1, 1, 0, // β: the last is infinite
	}
	cases := []struct {
		name               string
		alphar, alphai, bt []float64
		want               bool
	}{
		{"permuted and rescaled", []float64{6, 1, 1, 2 - 2e-10}, []float64{0, 0.5, -0.5, 0}, []float64{0, 2, 2, 2}, true},
		{"eigenvalue moved", []float64{3, 0.5, 0.5, 0.9}, []float64{0, 0.25, -0.25, 0}, []float64{0, 1, 1, 1}, false},
		{"selection flipped", []float64{3, 0.5, 0.5, 1 + 1e-10}, []float64{0, 0.25, -0.25, 0}, []float64{0, 1, 1, 1}, false},
		{"doubled match", []float64{3, 0.5, 0.5, 1 - 1e-10}, []float64{0, 0.25, 0.25, 0}, []float64{0, 1, 1, 1}, false},
	}
	for _, c := range cases {
		if got := qzRetryConsistent(first, c.alphar, c.alphai, c.bt, inside); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

// TestStableDeflatingQZ_RejectedReorder: the Riccati extended pencil's
// ordered QZ (generalizedStableSubspace) on the compressed 4×4 Dare pencil
// (H, J) of gonum ZHGFWA, perturbed by 1e-12 relative, returned
// ErrSchurFailed whenever Dgges rejected the 2×2/2×2 swap of the
// well-separated moduli 0.081 and 12.35. The retry on (J, H) must give the
// stable deflating subspace: leading eigenvalues those of J⁻¹H inside the
// unit circle, and span(H·Z₁) = span(J·Z₁) for its first two Schur
// vectors Z₁. XACVXD.
func TestStableDeflatingQZ_RejectedReorder(t *testing.T) {
	h0 := []float64{
		11.475236504468711, -4.802980538727953, 0, 0,
		-4.804997870973105e-16, 1.883135785961934e-16, 1, 0,
		1.883135785961934e-16, -7.38023300653887e-17, 0, 1,
		0.0004680437192265164, 0.00021592881520019866, 0, 0,
	}
	j0 := []float64{
		-0.9260172649172378, 0.37748115799546883, 0.00065672900631252, 0.00161105365515163,
		0, 0, -13.418168006075515, -2.5172613588536494,
		0, 0, 0.597370003150422, -11.258325117032756,
		-3.097492448978087e-05, -2.0823017944789118e-05, -31.53860826496756, -77.36888371806144,
	}
	stable := func(ar, ai, b float64) bool { return math.Hypot(ar, ai) < math.Abs(b) }
	const nn = 4
	rejected := 0
	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, 9))
		hs, js := make([]float64, nn*nn), make([]float64, nn*nn)
		for i := range hs {
			hs[i] = h0[i] * (1 + 1e-12*rng.NormFloat64())
			js[i] = j0[i] * (1 + 1e-12*rng.NormFloat64())
		}
		h, j := append([]float64(nil), hs...), append([]float64(nil), js...)
		ar, ai, beta := make([]float64, nn), make([]float64, nn), make([]float64, nn)
		vs, bwork := make([]float64, nn*nn), make([]bool, nn)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
			nn, h, nn, j, nn, ar, ai, beta, nil, 1, vs, nn, work, -1, bwork)
		work = make([]float64, int(work[0]))
		if _, ok := impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
			nn, h, nn, j, nn, ar, ai, beta, nil, 1, vs, nn, work, len(work), bwork); !ok {
			rejected++
		}

		copy(h, hs)
		copy(j, js)
		sdim, ok := stableDeflatingQZ(nn, h, j, hs, js, stable, ar, ai, beta, vs, &work, bwork)
		if !ok || sdim != 2 {
			t.Fatalf("seed %d: sdim %d ok %v", seed, sdim, ok)
		}
		H, J := mat.NewDense(nn, nn, hs), mat.NewDense(nn, nn, js)
		var want []complex128
		for _, v := range generalizedEigOracle(t, H, J) {
			if cmplx.Abs(v) < 1 {
				want = append(want, v)
			}
		}
		got := []complex128{complex(ar[0]/beta[0], ai[0]/beta[0]), complex(ar[1]/beta[1], ai[1]/beta[1])}
		assertEigSetNear(t, "stable eigenvalues", got, want, 1e-9)

		z1 := mat.NewDense(nn, 2, nil)
		for i := range nn {
			z1.Set(i, 0, vs[i*nn])
			z1.Set(i, 1, vs[i*nn+1])
		}
		var hz, jz mat.Dense
		hz.Mul(H, z1)
		jz.Mul(J, z1)
		both := mat.NewDense(nn, 4, nil)
		both.Slice(0, nn, 0, 2).(*mat.Dense).Copy(&hz)
		both.Slice(0, nn, 2, 4).(*mat.Dense).Copy(&jz)
		var svd mat.SVD
		if !svd.Factorize(both, mat.SVDNone) {
			t.Fatal("SVD failed")
		}
		sv := svd.Values(nil)
		if !(sv[2] <= 1e-12*sv[0]) {
			t.Fatalf("seed %d: span(HZ₁) ≠ span(JZ₁): singular values %v", seed, sv)
		}
	}
	if rejected == 0 {
		t.Fatal("no seed exercised a rejected reorder")
	}
}
