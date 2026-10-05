package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func schererPlantA() *mat.Dense {
	return mat.NewDense(3, 3, []float64{
		0, 10, 2,
		-1, 1, 0,
		0, 2, -5,
	})
}

func pvtolAB() (*mat.Dense, *mat.Dense) {
	A := mat.NewDense(6, 6, []float64{
		0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1,
		0, 0, -9.8, -0.0125, 0, 0,
		0, 0, 0, 0, -0.0125, 0,
		0, 0, 0, 0, 0, 0,
	})
	B := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 0,
		0, 0,
		0.25, 0,
		0, 0.25,
		5.2632, 0,
	})
	return A, B
}

func TestH2Syn_SchererExample7(t *testing.T) {
	A := schererPlantA()
	B1 := mat.NewDense(3, 1, []float64{1, 0, 1})
	B2 := mat.NewDense(3, 1, []float64{0, 1, 0})
	C1 := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
	})
	D11 := mat.NewDense(3, 1, []float64{0, 0, 0})
	D12 := mat.NewDense(3, 1, []float64{0, 0, 1})
	C2 := mat.NewDense(1, 3, []float64{0, 1, 0})
	D21 := mat.NewDense(1, 1, []float64{2})
	D22 := mat.NewDense(1, 1, []float64{0})

	Baug := mat.NewDense(3, 2, nil)
	Baug.Augment(B1, B2)

	Caug := mat.NewDense(4, 3, nil)
	Caug.Stack(C1, C2)

	Daug := mat.NewDense(4, 2, nil)
	var d11d12 mat.Dense
	d11d12.Augment(D11, D12)
	var d21d22 mat.Dense
	d21d22.Augment(D21, D22)
	Daug.Stack(&d11d12, &d21d22)

	P, err := New(A, Baug, Caug, Daug, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := H2Syn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

	kn, km, kp := res.K.Dims()
	if kn != 3 {
		t.Errorf("controller states: got %d, want 3", kn)
	}
	if km != 1 {
		t.Errorf("controller inputs: got %d, want 1", km)
	}
	if kp != 1 {
		t.Errorf("controller outputs: got %d, want 1", kp)
	}

	Ak := res.K.A
	Bk := res.K.B
	Ck := res.K.C
	Dk := res.K.D

	n := 3
	nk, _ := Ak.Dims()
	ncl := n + nk

	Acl := mat.NewDense(ncl, ncl, nil)
	Bcl := mat.NewDense(ncl, 1, nil)
	Ccl := mat.NewDense(3, ncl, nil)
	Dcl := mat.NewDense(3, 1, nil)

	var dkc2 mat.Dense
	dkc2.Mul(Dk, C2)
	var b2dkc2 mat.Dense
	b2dkc2.Mul(B2, &dkc2)
	var topLeft mat.Dense
	topLeft.Add(A, &b2dkc2)

	var b2ck mat.Dense
	b2ck.Mul(B2, Ck)

	var bkc2 mat.Dense
	bkc2.Mul(Bk, C2)

	for i := range n {
		for j := range n {
			Acl.Set(i, j, topLeft.At(i, j))
		}
		for j := range nk {
			Acl.Set(i, n+j, b2ck.At(i, j))
		}
	}
	for i := range nk {
		for j := range n {
			Acl.Set(n+i, j, bkc2.At(i, j))
		}
		for j := range nk {
			Acl.Set(n+i, n+j, Ak.At(i, j))
		}
	}

	var dkd21 mat.Dense
	dkd21.Mul(Dk, D21)
	var b2dkd21 mat.Dense
	b2dkd21.Mul(B2, &dkd21)
	var bTop mat.Dense
	bTop.Add(B1, &b2dkd21)
	var bkd21 mat.Dense
	bkd21.Mul(Bk, D21)
	for i := range n {
		Bcl.Set(i, 0, bTop.At(i, 0))
	}
	for i := range nk {
		Bcl.Set(n+i, 0, bkd21.At(i, 0))
	}

	var d12dkc2 mat.Dense
	d12dkc2.Mul(D12, &dkc2)
	var cLeft mat.Dense
	cLeft.Add(C1, &d12dkc2)
	var d12ck mat.Dense
	d12ck.Mul(D12, Ck)
	for i := range 3 {
		for j := range n {
			Ccl.Set(i, j, cLeft.At(i, j))
		}
		for j := range nk {
			Ccl.Set(i, n+j, d12ck.At(i, j))
		}
	}

	var d12dkd21 mat.Dense
	d12dkd21.Mul(D12, &dkd21)
	Dcl.Add(D11, &d12dkd21)

	cl, err := New(Acl, Bcl, Ccl, Dcl, 0)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("closed-loop system is not stable")
	}

	h2, err := H2Norm(cl)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(h2-7.7484) > 0.1 {
		t.Errorf("H2 norm = %v, want ≈ 7.7484", h2)
	}
}

func TestHinfSyn_SchererExample7(t *testing.T) {
	A := schererPlantA()
	B1 := mat.NewDense(3, 1, []float64{1, 0, 1})
	B2 := mat.NewDense(3, 1, []float64{0, 1, 0})
	C1 := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 0, 0,
	})
	D11 := mat.NewDense(2, 1, []float64{0, 0})
	D12 := mat.NewDense(2, 1, []float64{0, 1})
	C2 := mat.NewDense(1, 3, []float64{0, 1, 0})
	D21 := mat.NewDense(1, 1, []float64{2})
	D22 := mat.NewDense(1, 1, []float64{0})

	Baug := mat.NewDense(3, 2, nil)
	Baug.Augment(B1, B2)

	Caug := mat.NewDense(3, 3, nil)
	Caug.Stack(C1, C2)

	Daug := mat.NewDense(3, 2, nil)
	var d11d12 mat.Dense
	d11d12.Augment(D11, D12)
	var d21d22 mat.Dense
	d21d22.Augment(D21, D22)
	Daug.Stack(&d11d12, &d21d22)

	P, err := New(A, Baug, Caug, Daug, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	if res.GammaOpt <= 0 || math.IsInf(res.GammaOpt, 0) || math.IsNaN(res.GammaOpt) {
		t.Errorf("GammaOpt = %v, want positive finite", res.GammaOpt)
	}

	kn, _, _ := res.K.Dims()
	if kn != 3 {
		t.Errorf("controller states: got %d, want 3", kn)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}
}

func TestLqr_PVTOL_6State(t *testing.T) {
	A, B := pvtolAB()

	Q := mat.NewDense(6, 6, nil)
	for i := range 6 {
		Q.Set(i, i, 1)
	}
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	res, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	kr, kc := res.K.Dims()
	if kr != 2 || kc != 6 {
		t.Fatalf("K dims = %dx%d, want 2x6", kr, kc)
	}

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("unstable eigenvalue: %v", e)
		}
	}

	Q2 := mat.NewDense(6, 6, nil)
	Q2.Set(0, 0, 100)
	Q2.Set(1, 1, 10)
	Q2.Set(2, 2, 2*math.Pi/5)
	R2 := mat.NewDense(2, 2, []float64{0.1, 0, 0, 1.0})

	res2, err := Lqr(A, B, Q2, R2, nil)
	if err != nil {
		t.Fatal(err)
	}

	kr2, kc2 := res2.K.Dims()
	if kr2 != 2 || kc2 != 6 {
		t.Fatalf("K2 dims = %dx%d, want 2x6", kr2, kc2)
	}

	for _, e := range res2.Eig {
		if real(e) >= 0 {
			t.Errorf("unstable eigenvalue (design 2): %v", e)
		}
	}
}

func TestKalman_PVTOL_PartialMeasurement(t *testing.T) {
	A, B := pvtolAB()
	C := mat.NewDense(3, 6, []float64{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0,
	})

	Qn := mat.NewDense(2, 2, []float64{1e-2, 0, 0, 1e-2})
	Rn := mat.NewDense(3, 3, []float64{
		2e-4, 0, 1e-5,
		0, 2e-4, 1e-5,
		1e-5, 1e-5, 1e-4,
	})

	res, err := Lqe(A, B, C, Qn, Rn, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	lr, lc := res.K.Dims()
	if lr != 6 || lc != 3 {
		t.Fatalf("L dims = %dx%d, want 6x3", lr, lc)
	}

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("unstable observer eigenvalue: %v", e)
		}
	}
}

func TestLqr_PVTOL_HeavyInputPenalty(t *testing.T) {
	A, B := pvtolAB()

	Q := mat.NewDense(6, 6, nil)
	for i := range 6 {
		Q.Set(i, i, 1)
	}

	R1 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	res1, err := Lqr(A, B, Q, R1, nil)
	if err != nil {
		t.Fatal(err)
	}

	R2 := mat.NewDense(2, 2, []float64{1600, 0, 0, 1600})
	res2, err := Lqr(A, B, Q, R2, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range res2.Eig {
		if real(e) >= 0 {
			t.Errorf("unstable eigenvalue (heavy R): %v", e)
		}
	}

	maxReal1 := math.Inf(-1)
	for _, e := range res1.Eig {
		if real(e) > maxReal1 {
			maxReal1 = real(e)
		}
	}
	maxReal2 := math.Inf(-1)
	for _, e := range res2.Eig {
		if real(e) > maxReal2 {
			maxReal2 = real(e)
		}
	}

	if maxReal2 <= maxReal1 {
		t.Errorf("heavy R should have slower poles: max Re(eig) R=I=%v, R=1600I=%v", maxReal1, maxReal2)
	}
}
