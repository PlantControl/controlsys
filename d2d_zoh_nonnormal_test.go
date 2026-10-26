package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// T3YOES: non-normal n=6 discrete plant with double eigenvalues e^{±i} and a
// near-1 pair. Denman–Beavers stagnates at κ·eps above its 4e-15 stop, which
// made the principal-log path fail although the log exists (MATLAB d2d works).
func nonNormalRepeatedUnitCirclePlant(t *testing.T) *System {
	t.Helper()
	a := mat.NewDense(6, 6, []float64{
		88.5726118909358, 188.27241654160233, 153.35748544717939, 82.10916292149601, -121.92392767317254, -72.90542698921823,
		-15.179760501733082, -32.46232504907121, -25.58323860244785, -15.32527647714353, 21.41105021807897, 12.702651737815602,
		-27.056454634317237, -57.526518951912394, -47.24336082448259, -24.346709234160095, 37.33946780403238, 22.471092528615166,
		-13.40097363853692, -27.742124226420707, -24.44873715062253, -10.991713369417027, 18.952568943796752, 11.108261267432086,
		-2.9095116310712132, -6.02314692085357, -5.30811320307211, -2.7565477704601467, 5.285996575085498, 2.429696301310687,
		-3.284554400074544e-10, -6.799544223328732e-10, -5.992343262852476e-10, -2.1637613336056205e-10, 2.3050115576133167e-10, 1.0000000004221057,
	})
	b := mat.NewDense(6, 2, []float64{
		-0.9667981925794609, 0.70866977429126, -0.01238866346289056, 0.8693974903498248,
		0.9601702866503661, -0.24354415373579286, 0.5260775173811046, -0.9996930420352065,
		-0.6787200473200125, -0.29128928172134555, -0.8891911526253609, 0.8438539587324921,
	})
	c := mat.NewDense(1, 6, []float64{-856888.7533689471, -210795.79943077973, 993184.9187581927, -431376.8449706192, -714265.6520272002, 893206.1115093225})
	d := mat.NewDense(1, 2, []float64{-79.98586136261807, -68.69610256854305})
	sys, err := New(a, b, c, d, 1e-4)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func relFrob(got, want mat.Matrix) float64 {
	var d mat.Dense
	d.Sub(got, want)
	return mat.Norm(&d, 2) / mat.Norm(want, 2)
}

func eigVecCond(t *testing.T, a *mat.Dense) float64 {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(a, mat.EigenRight) {
		t.Fatal("eig failed")
	}
	var v mat.CDense
	eig.VectorsTo(&v)
	n, _ := v.Dims()
	aug := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			x := v.At(i, j)
			aug.Set(i, j, real(x))
			aug.Set(i, n+j, -imag(x))
			aug.Set(n+i, j, imag(x))
			aug.Set(n+i, n+j, real(x))
		}
	}
	var lu mat.LU
	lu.Factorize(aug)
	return lu.Cond()
}

func TestD2DZOHNonNormalRepeatedUnitCirclePoles(t *testing.T) {
	const eps = 0x1p-52
	sys := nonNormalRepeatedUnitCirclePlant(t)
	a, b := sys.A, sys.B
	n, m, _ := sys.Dims()
	kappa := eigVecCond(t, a)
	tol := 4 * float64(n) * kappa * kappa * eps

	for _, dt := range []float64{2e-4, 3e-4, 5e-4} {
		r := int(math.Round(dt / sys.Dt))
		got, err := sys.D2D(dt, D2DOptions{})
		if err != nil {
			t.Fatalf("D2D(%g): %v", dt, err)
		}
		pw, sum := mat.NewDense(n, n, nil), mat.NewDense(n, m, nil)
		for i := range n {
			pw.Set(i, i, 1)
		}
		for range r {
			var t1 mat.Dense
			t1.Mul(pw, b)
			sum.Add(sum, &t1)
			var t2 mat.Dense
			t2.Mul(pw, a)
			pw.Copy(&t2)
		}
		if e := relFrob(got.A, pw); !(e <= tol) {
			t.Errorf("D2D(%g).A vs A^%d: rel err %.3e > %.3e", dt, r, e, tol)
		}
		if e := relFrob(got.B, sum); !(e <= tol) {
			t.Errorf("D2D(%g).B vs ΣA^k·B: rel err %.3e > %.3e", dt, e, tol)
		}
	}

	half, err := sys.D2D(5e-5, D2DOptions{})
	if err != nil {
		t.Fatalf("D2D(5e-5): %v", err)
	}
	var h2, hb, hsum mat.Dense
	h2.Mul(half.A, half.A)
	hb.Mul(half.A, half.B)
	hsum.Add(&hb, half.B)
	if e := relFrob(&h2, a); !(e <= tol) {
		t.Errorf("D2D(dt/2).A² vs A: rel err %.3e > %.3e", e, tol)
	}
	if e := relFrob(&hsum, b); !(e <= tol) {
		t.Errorf("D2D(dt/2) (A_h+I)·B_h vs B: rel err %.3e > %.3e", e, tol)
	}
	var eig mat.Eigen
	if !eig.Factorize(half.A, mat.EigenNone) {
		t.Fatal("eig of D2D(dt/2).A failed")
	}
	for _, v := range eig.Values(nil) {
		if !(real(v) > 0) {
			t.Errorf("D2D(dt/2).A eigenvalue %v not in principal sqrt half-plane", v)
		}
	}
}
