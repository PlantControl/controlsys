package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Cheap-control Dare rows (R ~ 1e-12 and 7e-9, Q ~ 1e11, m > n, cross
// term S; rv48Gen seeds 414 and 1676, native literals) whose float64
// solves lost every candidate: R + BᵀXB keeps its smallest eigenvalues
// only to the rounding of the float64 X, so its Cholesky failed at every
// pencil X and Hewer iterate and Dare returned ErrNotConverged natively,
// a value under gonum v0.20.10 or on amd64 (TXU6AT). The extended solve
// (lostCandidateDare; boundaryExtendedDare for the descriptor row, cond(E)
// 3.8e6) returns X and K matching the 320-bit reference, at Accuracy 1:
// K moves by up to 3.7e-5 under eps perturbations of the data, so an
// Accuracy 1 K is held to 1e-4 (the amd64 float64 path returns K 8e-6
// off for 1676). Eig is the extended-precision closed loop rounded once.
func TestDareLostCandidateExtendedSolve(t *testing.T) {
	cases := []struct {
		name             string
		A, B, Q, R, S, E *mat.Dense
	}{
		{
			name: "plain-414",
			A:    mat.NewDense(3, 3, []float64{-0.2251790437173755, -0.48129874627487157, 0.37902882714562086, -0.24190951243465042, 0.14531778185220928, 0.08191536955333552, 0.36581446096955056, 0.4248561892265402, 0.14177912458582959}),
			B:    mat.NewDense(3, 5, []float64{-0.49156955681260417, 0.8430124597488321, -0.1434233562384435, -0.47113092704784254, -0.7927663059661576, 0.9036700124739887, 0.34107457090737514, -0.13252284465820285, 1.0753240485838127, -0.6486547613919175, -1.7004962216260608, -0.5743982519879223, 0.9543225453552299, 0.2305359148815772, -0.3344748444698409}),
			Q:    mat.NewDense(3, 3, []float64{1313365352520.7546, -325324582056.1548, 920303610655.1758, -325324582056.1548, 80583885882.84256, -227961996200.47003, 920303610655.1758, -227961996200.47003, 644876716261.3033}),
			R:    mat.NewDense(5, 5, []float64{1.0693493156981829e-12, 4.186489353531542e-14, -4.964571965180162e-14, -2.5378923590277936e-14, 1.6813823508408398e-14, 4.186489353531542e-14, 1.1240636358230036e-12, -3.1471785134687244e-14, -7.974148651112611e-14, -9.563019317562588e-15, -4.964571965180162e-14, -3.1471785134687244e-14, 1.029675742739064e-12, 1.5535023115711524e-14, -1.988282261302242e-14, -2.5378923590277936e-14, -7.974148651112611e-14, 1.5535023115711524e-14, 1.0514570603279812e-12, 2.1049978562031305e-15, 1.6813823508408398e-14, -9.563019317562588e-15, -1.988282261302242e-14, 2.1049978562031305e-15, 1.0160130153718719e-12}),
			S:    mat.NewDense(3, 5, []float64{0.00006094740506273052, -0.00006417494743315818, -0.000041917901640149205, -0.0002208245396373618, 0.00010308089948889432, -0.000015096857124626063, 0.000015896329160882112, 0.000010383191398782574, 0.00005469890836345346, -0.00002553345151052864, 0.00004270716966275998, -0.00004496877865933217, -0.000029372783556743177, -0.00015473654818095984, 0.00007223102376435926}),
		},
		{
			name: "descriptor-1676",
			A:    mat.NewDense(2, 2, []float64{0.5561197456339678, -0.15378737846060336, -0.16436750614003615, -0.024682211701084037}),
			B:    mat.NewDense(2, 5, []float64{0.38033762261640636, 0.5891130340133576, -0.09377911980354647, 2.5558364313931925, 1.4568859603240722, -0.28866277612480723, -1.0187794683389684, 0.36862037453340313, -0.3233056009853006, 2.655491509937177}),
			Q:    mat.NewDense(2, 2, []float64{190707489946.0168, -296096393283.97144, -296096393283.97144, 459725384360.0676}),
			R:    mat.NewDense(5, 5, []float64{7.163679390184327e-9, 3.532666765028939e-10, 1.3335504456594452e-10, -1.9738198171586303e-10, -5.513264449923645e-11, 3.532666765028939e-10, 7.558627940285399e-9, 2.1499666590007985e-10, -3.9417598975764285e-10, 4.249437134320053e-11, 1.3335504456594452e-10, 2.1499666590007985e-10, 7.1488671973345055e-9, 5.115865143120892e-12, 5.629357466492653e-11, -1.9738198171586303e-10, -3.9417598975764285e-10, 5.115865143120892e-12, 7.300364814153682e-9, -9.831142072984307e-13, -5.513264449923645e-11, 4.249437134320053e-11, 5.629357466492653e-11, -9.831142072984307e-13, 7.168519820939519e-9}),
			S:    mat.NewDense(2, 5, []float64{-0.04003900641729683, -0.03768578944404571, 0.04061451577236974, -0.014011000825019219, -0.07441009008813389, 0.062165389488327207, 0.05851173614418473, -0.06305893721624554, 0.02175376966977688, 0.11553064489112909}),
			E:    mat.NewDense(2, 2, []float64{1.0182414405931177, -0.010989402943873683, 0, 2.7252380040259193e-7}),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref := riccatiExactReference(false, tc.A, tc.B, tc.Q, tc.R, tc.S, tc.E, riccatiExactPrec)
			if !ref.OK {
				t.Fatal("320-bit reference failed")
			}
			for _, ns := range []bool{false, true} {
				res, err := Dare(tc.A, tc.B, tc.Q, tc.R, &RiccatiOpts{S: tc.S, E: tc.E, NoScaling: ns})
				if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
					t.Fatalf("NoScaling=%v: %v", ns, err)
				}
				if d := relErr(res.X, ref.X); d > newtonTol {
					t.Errorf("NoScaling=%v: X %.3g off the 320-bit X", ns, d)
				}
				kTol := gainFloor
				if err != nil {
					kTol = 1e-4
				}
				if d := relErr(res.K, ref.K); d > kTol {
					t.Errorf("NoScaling=%v: K %.3g off the 320-bit K", ns, d)
				}
				for _, v := range res.Eig {
					if !(cmplx.Abs(v) < 1) {
						t.Errorf("NoScaling=%v: closed-loop eigenvalue %v", ns, v)
					}
				}
				if r := lostCandidateRadius(tc.A, tc.B, res.K); tc.E == nil && !(r < 1) {
					t.Errorf("NoScaling=%v: closed-loop radius %.6g", ns, r)
				}
			}
		})
	}
}

// lostCandidateRadius is the float64 spectral radius of A − BK; with a
// near-singular E that of E⁻¹(A − BK) is rounding (riccati_extended.go),
// so the descriptor row is judged on Eig alone.
func lostCandidateRadius(A, B, K *mat.Dense) float64 {
	var bk, acl mat.Dense
	bk.Mul(B, K)
	acl.Sub(A, &bk)
	var eig mat.Eigen
	if !eig.Factorize(&acl, mat.EigenNone) {
		return 2
	}
	r := 0.0
	for _, v := range eig.Values(nil) {
		r = max(r, cmplx.Abs(v))
	}
	return r
}
