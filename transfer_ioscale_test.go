package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

func polyDiffRel(got, want []float64) float64 {
	k := max(len(got), len(want))
	var diff, norm float64
	for i := range k {
		g, w := 0.0, 0.0
		if j := len(got) - k + i; j >= 0 {
			g = got[j]
		}
		if j := len(want) - k + i; j >= 0 {
			w = want[j]
		}
		diff += (g - w) * (g - w)
		norm += w * w
	}
	return math.Sqrt(diff / norm)
}

func checkTFMatchesSS(t *testing.T, sys *System, res *TransferFuncResult) {
	t.Helper()
	for _, w := range []float64{0.3, 1.1, 2.4} {
		z := cmplx.Exp(complex(0, w))
		if sys.Dt == 0 {
			z = complex(0, w)
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
			t.Errorf("w=%g: TF %v, SS %v", w, got[0][0], want[0][0])
		}
	}
}

// JBKTCQ: ‖B‖ ≈ 1e-15 sits under the staircase floor n²ε‖A‖ unless B is
// I/O-scaled first; the transfer function is linear in B, so MATLAB tf(ss)
// keeps it.
func TestTransferFunctionTinyB(t *testing.T) {
	num := []float64{-8.70643320022765e-16, 2.9186581655941266e-16, 1.3545850066477854e-16}
	den := []float64{1, -5.609886783435264, 13.099973168748164, -16.29841103348229, 11.394314088807377, -4.243853507030152, 0.6578640663921657}
	for _, d := range []float64{0, 1} {
		r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 0.01}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		sys := r.Sys
		sys.D.Set(0, 0, d)
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.MinimalOrder != 6 {
			t.Fatalf("D=%g: MinimalOrder %d, want 6", d, res.MinimalOrder)
		}
		wantNum := Poly(num).Add(Poly(den).Scale(d))
		if e := polyDiffRel(res.TF.Num[0][0], wantNum); e > 1e-10 {
			t.Errorf("D=%g: num %v, want %v (rel %g)", d, res.TF.Num[0][0], wantNum, e)
		}
		if e := polyDiffRel(res.TF.Den[0], den); e > 1e-10 {
			t.Errorf("D=%g: den %v, want %v (rel %g)", d, res.TF.Den[0], den, e)
		}
		checkTFMatchesSS(t, sys, res)
	}
}

// JBKTCQ: Dgebal on this companion form (trailing coefficients down to
// 1e-55) shrank the chain coupling to 7e-18, below the staircase floor, so
// the TF was 0 and the DC gain read 0. den(1) is 4.6e-17 exactly but ~1e-15
// under a one-ulp coefficient perturbation, so any |gain| ≳ 0.25 (Inf
// included) is backward stable; 0 is not.
func TestTransferFunctionCompanionTinyTrailingDen(t *testing.T) {
	num := []float64{2.505546303261118e-16}
	den := []float64{1, -2.5389606785849783, 2.1293764484093325, -0.5907214985880699, 0.0003057287637157701, -1.7299922355352376e-17, 1.1126423085115152e-31, -3.023903148124882e-55}
	sys := pidtuneBackoffPlant(t, num, den, 1, 0)
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 7 {
		t.Fatalf("MinimalOrder %d, want 7", res.MinimalOrder)
	}
	if e := polyDiffRel(res.TF.Num[0][0], num); e > 1e-10 {
		t.Errorf("num %v, want %v", res.TF.Num[0][0], num)
	}
	if e := polyDiffRel(res.TF.Den[0], den); e > 1e-12 {
		t.Errorf("den %v, want %v", res.TF.Den[0], den)
	}
	checkTFMatchesSS(t, sys, res)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if v := g.At(0, 0); !(math.Abs(v) >= 0.1) {
		t.Errorf("DCGain %g, want |gain| ≥ 0.1 or ±Inf", v)
	}
}
