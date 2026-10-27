package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// stepOracleTransferAt is the 320-bit pencil transfer function at the
// float64 point s, rounded to complex128 (nil when sE − A is singular).
func stepOracleTransferAt(o stepOraclePencil, s complex128) [][]complex128 {
	re, im := o.transfer(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))})
	if re == nil {
		return nil
	}
	out := make([][]complex128, len(re))
	for i := range re {
		out[i] = make([]complex128, len(re[i]))
		for j := range re[i] {
			a, _ := re[i][j].Float64()
			b, _ := im[i][j].Float64()
			out[i][j] = complex(a, b)
		}
	}
	return out
}

// A full-rank E with two stiff singular values (1e-15 and 1e-4 beside
// O(1)) puts two stiff poles (seed 8: -7e14 and -8192.39) in one fast
// part of the pencil split. TransferFunction folded that part whole and
// kept -8192.39 only to ε·‖E⁻¹A‖ (-8192.383, 8e-7 relative; 8e-4 response
// error beside it), while Step and C2D split it again (UOFXRA). The pole
// near 1/ε is ill-posed under ε perturbations of E and is not checked.
// Oracle: 320-bit generalized eigenvalues and response of the pencil.
func TestTransferFunctionTwoStiffSingularValuesFollowPencil(t *testing.T) {
	resolved := 0
	for _, seed := range []uint64{8, 12, 26} {
		sys := twoStiffDescriptor(t, seed)
		name := fmt.Sprintf("seed %d", seed)
		pol, err := sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", name, err)
		}
		modes, ok := newStepOracleModes(sys, pol)
		if !ok {
			continue
		}
		resolved++
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", name, err)
		}
		stiffest := 0
		for i, q := range modes.poles {
			if cmplx.Abs(q) > cmplx.Abs(modes.poles[stiffest]) {
				stiffest = i
			}
		}
		o := newStepOraclePencil(sys)
		pts := []complex128{0.05i, 0.37i, complex(0.5, 3), 10i, 1e3i, 1e4i}
		for i := range res.TF.Den {
			roots, err := Poly(res.TF.Den[i]).Roots()
			if err != nil {
				t.Fatalf("%s: Roots: %v", name, err)
			}
			for k, q := range modes.poles {
				if k == stiffest {
					continue
				}
				best := math.Inf(1)
				for _, r := range roots {
					best = math.Min(best, cmplx.Abs(r-q)/cmplx.Abs(q))
				}
				if !(best < 1e-9) {
					t.Errorf("%s row %d: pole %.10g off by %.3g relative", name, i, q, best)
				}
			}
		}
		for k, q := range modes.poles {
			if k != stiffest {
				pts = append(pts, q+complex(0, 1e-3*cmplx.Abs(q)))
			}
		}
		for _, s := range pts {
			want := stepOracleTransferAt(o, s)
			got, err := res.TF.Eval(s)
			if err != nil {
				t.Fatalf("%s: Eval(%v): %v", name, s, err)
			}
			for i := range want {
				for j := range want[i] {
					if e := cmplx.Abs(got[i][j]-want[i][j]) / math.Max(1, cmplx.Abs(want[i][j])); !(e < 1e-8) {
						t.Errorf("%s: G%d%d(%.4g) off by %.3g", name, i, j, s, e)
					}
				}
			}
		}
	}
	if resolved < 2 {
		t.Fatalf("only %d models resolved", resolved)
	}
}

func BenchmarkTransferFunctionTwoStiffSingularValues(b *testing.B) {
	t := &testing.T{}
	sys := twoStiffDescriptor(t, 8)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.TransferFunction(nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStepTwoStiffSingularValues(b *testing.B) {
	t := &testing.T{}
	sys := twoStiffDescriptor(t, 8)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Step(sys, 5); err != nil {
			b.Fatal(err)
		}
	}
}
