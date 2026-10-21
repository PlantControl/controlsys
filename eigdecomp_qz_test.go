package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

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
