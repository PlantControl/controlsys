package controlsys

import (
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A near-singular E (σ_min 8.6e-15) with a cross term: X and K are exact
// at Accuracy 0, and the exact closed loop eig(A − BK, E) of the returned
// K is 0.576 and −0.412 (320-bit characteristic polynomial of
// E⁻¹(A − BK)), while the float64 QZ of (A − BK, E), with A − BK off by
// eps·|B||K|, gave 1.273 and −1.282 (O2YDYA review, big seed 113; base
// errBoundaryUnstabilizable).
func TestDare_NearSingularEEigIsExactClosedLoop(t *testing.T) {
	c := nearSingularDareCase{
		A: mat.NewDense(2, 2, []float64{11.948512052746423, -21.105884578517625, -8.290676983069437, 23.974770830730687}),
		B: mat.NewDense(2, 1, []float64{0.301695085557923, -1.1687483879060778}),
		Q: mat.NewDense(2, 2, []float64{0.5608101030147544, 0.35734163098495, 0.35734163098495, 5.974414432091204}),
		R: mat.NewDense(1, 1, []float64{2.120685880583526}),
		S: mat.NewDense(2, 1, []float64{-0.09322082961310253, -1.9022641735689931}),
		E: mat.NewDense(2, 2, []float64{1.3452987172462798, -0.9996165691907544, 0, 8.591692187096886e-15}),
	}
	c.check(t, 0, 0)
	for _, ns := range []bool{false, true} {
		res, err := Dare(c.A, c.B, c.Q, c.R, &RiccatiOpts{S: c.S, E: c.E, NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		acl := hbSolve(hbFrom(c.E), hbAdd(hbFrom(c.A), hbMul(hbFrom(c.B), hbFrom(res.K)), -1))
		tr := new(big.Float).SetPrec(hbPrec).Add(acl.at(0, 0), acl.at(1, 1))
		det := new(big.Float).SetPrec(hbPrec).Mul(acl.at(0, 0), acl.at(1, 1))
		det.Sub(det, new(big.Float).SetPrec(hbPrec).Mul(acl.at(0, 1), acl.at(1, 0)))
		trf, _ := tr.Float64()
		detf, _ := det.Float64()
		r := cmplx.Sqrt(complex(trf*trf-4*detf, 0))
		want := []complex128{(complex(trf, 0) + r) / 2, (complex(trf, 0) - r) / 2}
		if len(res.Eig) != 2 {
			t.Fatalf("NoScaling %v: Eig %v", ns, res.Eig)
		}
		got := slices.Clone(res.Eig)
		if real(got[0]) < real(got[1]) {
			got[0], got[1] = got[1], got[0]
		}
		for i := range 2 {
			if d := cmplx.Abs(got[i] - want[i]); !(d <= 1e-8) {
				t.Errorf("NoScaling %v: Eig %v, exact closed loop %v", ns, res.Eig, want)
			}
		}
	}
}
