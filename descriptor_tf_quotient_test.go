package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// TestTFRowsQuotientOverflow checks that tfRows divides numerator by
// denominator at exactCertPrec bits before rounding (SFF7CU): at s = 1e8j
// s^40 overflows float64, so rounding first gives 0 when only the
// denominator overflows and NaN when both do.
func TestTFRowsQuotientOverflow(t *testing.T) {
	s := complex(0, 1e8)
	pow := func(k int, c float64) []float64 {
		p := make([]float64, k+1)
		p[0] = c
		return p
	}
	for _, tc := range []struct {
		name     string
		num, den []float64
		want     complex128
	}{
		{"denominator only", pow(30, 1e10), pow(40, 1), complex(-1e-70, 0)},
		{"both", pow(41, 1), pow(40, 1), s},
		{"neither", []float64{2, 1}, []float64{1, 0, 3}, (2*s + 1) / (s*s + 3)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tf := &TransferFunc{Num: [][][]float64{{tc.num}}, Den: [][]float64{tc.den}}
			r, mag := make([]complex128, 1), make([]float64, 1)
			if !(tfRows{tf}).at(0, s, r, mag) {
				t.Fatal("row not finite")
			}
			if d := cmplx.Abs(r[0]-tc.want) / cmplx.Abs(tc.want); !(d <= 4*eps()) {
				t.Errorf("tf = %v, want %v (rel %.3g)", r[0], tc.want, d)
			}
			if mag[0] != cmplx.Abs(r[0]) {
				t.Errorf("mag = %v, want |tf| %v", mag[0], cmplx.Abs(r[0]))
			}
		})
	}
	tf := &TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{0, 0}}}
	if (tfRows{tf}).at(0, s, make([]complex128, 1), make([]float64, 1)) {
		t.Error("zero denominator evaluated")
	}
}

// TestTransferFunctionDescriptorFoldMatchesExplicit pins, per the SFF7CU
// ruling, that the E⁻¹ fold of a well-conditioned E is not gated: its TF
// is bit-identical to ToExplicit().TransferFunction, error included.
func TestTransferFunctionDescriptorFoldMatchesExplicit(t *testing.T) {
	cases := zpkDescriptorCases(t)
	for _, name := range []string{"nonsingular E", "discrete two by two"} {
		t.Run(name, func(t *testing.T) {
			sys := cases[name]
			if _, route, _ := sys.transferRationalRoute(nil); route != tfRouteFold {
				t.Fatalf("route %d, want the E⁻¹ fold", route)
			}
			got, err := sys.TransferFunction(nil)
			ex, xerr := sys.ToExplicit()
			if xerr != nil {
				t.Fatal(xerr)
			}
			want, werr := ex.TransferFunction(nil)
			if err != werr && (err == nil || werr == nil || err.Error() != werr.Error()) {
				t.Fatalf("err %v, explicit err %v", err, werr)
			}
			if got.MinimalOrder != want.MinimalOrder || !slices.Equal(got.RowDegrees, want.RowDegrees) {
				t.Errorf("order %d %v, explicit %d %v", got.MinimalOrder, got.RowDegrees, want.MinimalOrder, want.RowDegrees)
			}
			bits := func(x []float64) []uint64 {
				out := make([]uint64, len(x))
				for i, v := range x {
					out[i] = math.Float64bits(v)
				}
				return out
			}
			for i := range want.TF.Den {
				if !slices.Equal(bits(got.TF.Den[i]), bits(want.TF.Den[i])) {
					t.Errorf("row %d den %v, explicit %v", i, got.TF.Den[i], want.TF.Den[i])
				}
				for j := range want.TF.Num[i] {
					if !slices.Equal(bits(got.TF.Num[i][j]), bits(want.TF.Num[i][j])) {
						t.Errorf("(%d, %d) num %v, explicit %v", i, j, got.TF.Num[i][j], want.TF.Num[i][j])
					}
				}
			}
		})
	}
}
