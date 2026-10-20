package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// AULHZC #72: a Hessenberg subdiagonal of A is judged against ‖A‖, not ‖c‖,
// so a C eight orders above ‖A‖ ≈ 1e-13 no longer drops the second state
// and TF(2^k·sys) = 2^k·TF(sys). Oracle: EvalFr of the state-space model.
func TestTransferFunctionOutputScaleInvariant(t *testing.T) {
	a := []float64{-3.162277660168379e-14, 9.486832980505137e-14, -9.486832980505139e-14, -3.162277660168378e-14}
	b := []float64{-0.9667981925794609, -0.01238866346289056}
	z := complex(9.32e-14, 3.62e-14)
	for _, k := range []int{0, -60, -20, 20, 60} {
		f := math.Ldexp(1, k)
		c := []float64{856888.7533689471 * f, 210795.79943077973 * f}
		sys := reduceTestSys(t, 2, 1, 1, a, b, c, 0)
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.MinimalOrder != 2 {
			t.Errorf("C·2^%d: MinimalOrder %d, want 2", k, res.MinimalOrder)
		}
		want, err := sys.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		got, err := res.TF.Eval(z)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(got[0][0] - want[0][0]); d > 1e-8*cmplx.Abs(want[0][0]) {
			t.Errorf("C·2^%d: tf(%v) = %v, ss %v", k, z, got[0][0], want[0][0])
		}
	}
}

// w12 #42: ‖A‖ ≈ 1e155 overflowed the Frobenius norm and the numerator
// products, so 1/(s+a) came back as 0. MATLAB tf(ss) gives 1/(s+a).
func TestTransferFunctionHugeA(t *testing.T) {
	for _, a := range []float64{-1e155, -1e200} {
		sys := reduceTestSys(t, 1, 1, 1, []float64{a}, []float64{1}, []float64{1}, 0)
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.MinimalOrder != 1 {
			t.Fatalf("a=%g: MinimalOrder %d, want 1", a, res.MinimalOrder)
		}
		s := complex(0, -a)
		got, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		want := 1 / (s - complex(a, 0))
		if d := cmplx.Abs(got[0][0] - want); d > 1e-12*cmplx.Abs(want) {
			t.Errorf("a=%g: tf(%v) = %v, want %v", a, s, got[0][0], want)
		}
		r, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if r.Order != 1 {
			t.Errorf("a=%g: minreal order %d, want 1", a, r.Order)
		}
	}
}
