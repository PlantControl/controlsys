package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestZPKModelResponseBelowTermsVerified checks ZPKModel of a discrete
// matched C2D model with a 7-pole cluster near z = 1 and one
// uncontrollable mode, mixed by an orthogonal similarity (census row
// matched-hid/0-unc-true). Its response lies below √ε of its terms
// Σ|C_l||X_l| at every axis point: the 320-bit oracle has |G| ≤ 2.5e-3·√ε·T
// there, so the transfer function's rank floor leaves row degree 1 and
// gain 1.6e-16, and any Z within √ε of the terms meets the within-terms
// contract. The response is not computable in float64 near the cluster
// (the terms bound reaches the terms), so the row verifies at the points
// away from it; the verifier must return nil, not ErrAccuracyUnverified.
func TestZPKModelResponseBelowTermsVerified(t *testing.T) {
	A := []float64{
		-1220.7787956881002, -1487.5979923369841, -2171.7365153557917, 1578.8099206630468, 1813.6325401656609, 2807.683580097108, -1872.4303332120473, 843.9075206484545,
		-1011.9917224602098, 1327.8091321755833, 764.6764743781586, -400.58329234453964, 60.34121755378857, -28.509918536119716, -125.16601318244409, 866.5698994114902,
		495.4363738174986, -891.8671455316187, -526.1864047135728, 940.064187414977, -600.7842131489964, 1609.5957395835585, -1270.0727781530795, -752.9851097223448,
		-1046.4654844449105, 3294.6611984423967, 2664.347299397963, -2035.8375698994719, -610.1614736561739, -2597.6505129021093, 1636.913468023678, 1202.96402419535,
		-272.2326381906347, -701.1968047322295, -725.6067042326438, 1431.3818979926605, -397.3870871195067, 2929.1854751935725, -2332.645719135864, -282.35872831291806,
		768.5074871282343, -661.0909550581451, -201.09607296498328, 283.1307784451819, -496.0479507517152, 199.56587823329446, -143.22629394275168, -747.8589991900676,
		739.6181241558702, 1514.8249205656487, 1902.8470204300065, -1542.2636015808207, -1228.9647784452648, -2680.80670653177, 1839.9698401236915, -375.3306919339003,
		-647.467895095867, -1215.8057038712639, -1694.6869061992004, 374.1606023149873, 2109.649474931775, 117.85592217409325, 304.60757284695353, 820.8346552933344,
	}
	B := []float64{0.5945040071021007, 0.7739295676845817, -0.9358713676478546, -0.06322393932494313, -0.24216985293349755, 0.045076767689603336, 0.009237811389474937, 0.32708108582097595}
	C := []float64{0.14882132438760087, 0.007196419709805553, 0.13532663196812517, 0.09508666735502475, 0.05106808671112067, 0.19718090024100604, 0.22069180280507045, 0.1224645632029596}
	sys, err := New(mat.NewDense(8, 8, A), mat.NewDense(8, 1, B), mat.NewDense(1, 8, C), mat.NewDense(1, 1, nil), 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.RowDegrees) != 1 || res.RowDegrees[0] != 1 || len(res.ZPK.Poles[0][0]) != 1 {
		t.Fatalf("RowDegrees %v, poles %v, want [1] and one pole", res.RowDegrees, res.ZPK.Poles[0][0])
	}
	o := newStepOraclePencil(sys)
	tol := math.Sqrt(eps())
	for _, th := range []float64{1e-4, 1e-3, 4e-3, 0.02, 0.05, 0.1, 0.5, 1, 2, math.Pi} {
		s := cmplx.Rect(1, th)
		got, err := res.ZPK.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		g, terms := zpkOracleResponseTerms(o, s)
		den := math.Max(math.Max(cmplx.Abs(g), cmplx.Abs(got[0][0])), terms)
		if e := cmplx.Abs(got[0][0]-g) / (tol * den); !(e <= 1) {
			t.Errorf("θ = %g: |Z − G| = %.3g·√ε of max(|G|, |Z|, terms)", th, e)
		}
	}
}

// zpkOracleResponseTerms returns the 320-bit response C(sI−A)⁻¹B of the
// single-input single-output o at s and its terms Σ_l |C_l||X_l|.
func zpkOracleResponseTerms(o stepOraclePencil, s complex128) (complex128, float64) {
	xr, xi := o.solveShifted(stepOracleComplex{stepOracleFloat(real(s)), stepOracleFloat(imag(s))}, o.b)
	r, q, tt := stepOracleNew(), stepOracleNew(), stepOracleNew()
	for k := range o.n {
		r.Add(r, stepOracleNew().Mul(o.c[0][k], xr[k][0]))
		q.Add(q, stepOracleNew().Mul(o.c[0][k], xi[k][0]))
		ax := stepOracleNew().Sqrt(stepOracleNew().Add(stepOracleNew().Mul(xr[k][0], xr[k][0]), stepOracleNew().Mul(xi[k][0], xi[k][0])))
		tt.Add(tt, stepOracleNew().Mul(stepOracleNew().Abs(o.c[0][k]), ax))
	}
	rf, _ := r.Float64()
	qf, _ := q.Float64()
	tf, _ := tt.Float64()
	return complex(rf, qf), tf
}
