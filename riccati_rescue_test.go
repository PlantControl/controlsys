package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hewerDare refines the stabilizing Dare solution by Hewer's iteration from
// the stabilizing gain K0 (Stein solves only), with cross term S (nil for
// none).
func hewerDare(t *testing.T, A, B, Q, R, S, K0 *mat.Dense) *mat.Dense {
	t.Helper()
	K := mat.DenseCopyOf(K0)
	var X *mat.Dense
	for range 40 {
		var bk, acl, qk, sk mat.Dense
		bk.Mul(B, K)
		acl.Sub(A, &bk)
		if r := spectralRadius(t, &acl); !(r < 1) {
			t.Fatalf("Hewer: gain not stabilizing, ρ(A−BK) = %g", r)
		}
		qk.Product(K.T(), R, K)
		qk.Add(&qk, Q)
		if S != nil {
			sk.Mul(S, K)
			qk.Sub(&qk, &sk)
			qk.Sub(&qk, sk.T())
		}
		var qs mat.Dense
		qs.Add(&qk, qk.T())
		qs.Scale(0.5, &qs)
		var err error
		if X, err = DLyap(mat.DenseCopyOf(acl.T()), &qs, nil); err != nil {
			t.Fatal(err)
		}
		var btx, rb, rhs mat.Dense
		btx.Mul(B.T(), X)
		rb.Mul(&btx, B)
		rb.Add(&rb, R)
		rhs.Mul(&btx, A)
		if S != nil {
			rhs.Add(&rhs, S.T())
		}
		var kn mat.Dense
		if err := kn.Solve(&rb, &rhs); err != nil {
			if _, ok := err.(mat.Condition); !ok {
				t.Fatal(err)
			}
		}
		K = &kn
	}
	return X
}

// TestDare_FirstPassFailureRescued is repro 1 of the wave-13 review: Q, R
// ≈ 1e-18 against B ≈ 50 made QZ on the unscaled pencil fail to converge
// (ErrSchurFailed) although the problem is well posed (Q, R scaled by 1e3
// solved). The cost-scaled retry recovers X to Hewer accuracy.
func TestDare_FirstPassFailureRescued(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.28174780751848155, 0.6036631370315961, -0.5456384642950813, -0.28174780751848155})
	B := mat.NewDense(2, 3, []float64{-41.84841226357985, 63.39693822638688, -2.856046081841606, -10.089202798211586, -18.738781535670917, -57.65320508135332})
	Q := mat.NewDense(2, 2, []float64{4.811678921620128e-18, 2.335983846337528e-19, 2.335983846337528e-19, 1.1022920627588624e-19})
	R := mat.NewDense(3, 3, []float64{1.6608616341663586e-18, 5.744877277805154e-19, 3.111894979254119e-18, 5.744877277805154e-19, 8.624437900904661e-18, 4.0470852635315025e-20, 3.111894979254119e-18, 4.0470852635315025e-20, 6.365032256247916e-18})
	for _, noScaling := range []bool{false, true} {
		t.Run(fmt.Sprintf("NoScaling=%v", noScaling), func(t *testing.T) {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{NoScaling: noScaling})
			if err != nil {
				t.Fatal(err)
			}
			if e := relErr(res.X, hewerDare(t, A, B, Q, R, nil, res.K)); !(e <= 1e-12) {
				t.Errorf("X rel err %g against Hewer", e)
			}
		})
	}
}

// TestDare_ScalarExtremeCost is repro 2 of the wave-13 review (n = 1,
// Q ≈ 1.6e20, R ≈ 3e21): the unscaled pencil misplaced an eigenvalue on
// amd64 (ErrNoStabilizing), and under NoScaling returned X = 1.63e20
// (amd64) or 6.89e20 (arm64) without error. X is the root of the scalar
// equation gx² + (1 − a² − qg)x − q = 0, g = bᵀR⁻¹b, in exact arithmetic.
func TestDare_ScalarExtremeCost(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.8246388766958636})
	B := mat.NewDense(1, 2, []float64{-0.11587870306466054, 0.2439079407230551})
	Q := mat.NewDense(1, 1, []float64{1.6316228642470052e20})
	R := mat.NewDense(2, 2, []float64{3.0412381031614555e21, -2.4750910378180346e21, -2.4750910378180346e21, 2.2456003725901646e21})
	const want = 4.6543636239364266e20
	for _, noScaling := range []bool{false, true} {
		t.Run(fmt.Sprintf("NoScaling=%v", noScaling), func(t *testing.T) {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{NoScaling: noScaling})
			if err != nil {
				t.Fatal(err)
			}
			if x := res.X.At(0, 0); !(math.Abs(x-want) <= 1e-12*want) {
				t.Errorf("X = %.17g, want %.17g", x, want)
			}
		})
	}
}

// TestDare_IllConditionedRCrossTerm is corpus illcond_r_dare_cross
// (cond(R) = 1.4e9, Q − SR⁻¹Sᵀ ≈ 5e-22 against Q ≈ 7e-13): the extended
// pencil deflated R with error eps·cond(R), which lost 2e-3 (arm64) to
// 3e-2 (amd64) of X and moved Eig 6e-6 off eig(A − BK). Data rounding
// moves X by about 3e-7; want is the exact solution for the float data.
func TestDare_IllConditionedRCrossTerm(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.99853327742287})
	B := mat.NewDense(1, 3, []float64{-0.9667981925794609, 0.70866977429126, 0.25125984258225487})
	Q := mat.NewDense(1, 1, []float64{7.342583356501883e-13})
	R := mat.NewDense(3, 3, []float64{683136.2493578977, 586713.6639653208, 397661.892696049, 586713.6639653208, 503900.8323657315, 341533.13726788113, 397661.892696049, 341533.13726788113, 231483.810097708})
	S := mat.NewDense(1, 3, []float64{-0.0007082361786449373, -0.0006082708161430631, -0.000412272866434415})
	const want = 1.7658500785417086e-19
	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	if x := res.X.At(0, 0); !(math.Abs(x-want) <= 2e-6*want) {
		t.Errorf("X = %.17g, want %.17g (rel err %.2g)", x, want, math.Abs(x-want)/want)
	}
	var bk mat.Dense
	bk.Mul(B, res.K)
	if acl := A.At(0, 0) - bk.At(0, 0); !(cmplx.Abs(res.Eig[0]-complex(acl, 0)) <= 1e-12) {
		t.Errorf("Eig = %v, A − BK = %.17g", res.Eig, acl)
	}
}

// TestDare_CostScaleInvarianceRescue: plants of the wave-13 cost sweep
// where Dare(A, B, cQ, cR, cS) failed (ErrSchurFailed, ErrNoStabilizing)
// at |log10 c| ≥ 8 on the unscaled pencil. X must be cX₁ and match Hewer.
func TestDare_CostScaleInvarianceRescue(t *testing.T) {
	for _, tc := range []struct {
		name          string
		n, m          int
		a, b, q, r, s []float64
	}{
		{"cross", 2, 2,
			[]float64{1.026669214509211, 0.04044177155695123, -0.23880624061680839, 1.124706212257149},
			[]float64{0.28027548069191427, -0.09997936938185849, 0.25427980404524164, 0.14732778937943633},
			[]float64{9.257871939150997e-18, -4.907838072399855e-18, -4.907838072399855e-18, 2.6017722758764403e-18},
			[]float64{2.2619429844314834e-09, 5.857721037586437e-11, 5.857721037586437e-11, 1.029998257604572e-09},
			[]float64{1.9273942580335994e-14, 2.007369543399835e-14, -1.0217616945098559e-14, -1.0641586679127361e-14}},
		{"n4", 4, 2,
			[]float64{-0.9754874191448087, 1.336931774378182, 0.3488342385604968, 0.8503359688261004, -1.0645501323343096, -1.6968013579606473, 0.07501675142927976, 0.03817750126104429, -0.08757754835869479, -0.31849172809380155, -0.4998328025975689, 0.3482361004839971, -0.06786929096625394, 1.0951834087219083, 0.21465664246608296, -0.26161970552198893},
			[]float64{-1.9373153705681616, 6.368389767608739, 4.858635252537669, 4.228215255963803, 0.5590587775992776, -5.970191560607039, 0.0593679377890042, 0.17051323709837116},
			[]float64{4.630369643739565e-16, -6.307503668716845e-16, 5.957758257558424e-16, 1.647486390506298e-16, -6.307503668716845e-16, 1.6915072646263646e-15, -6.527876683571441e-16, 1.043781668256361e-16, 5.957758257558424e-16, -6.527876683571441e-16, 1.162429455705182e-15, 1.5220360862748926e-16, 1.647486390506298e-16, 1.043781668256361e-16, 1.5220360862748926e-16, 2.295583347526418e-16},
			[]float64{0.005951376898734779, 0.0012344200936146505, 0.0012344200936146505, 0.020129920208935736},
			nil},
	} {
		A, B := mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b)
		Q, R := mat.NewDense(tc.n, tc.n, tc.q), mat.NewDense(tc.m, tc.m, tc.r)
		var S *mat.Dense
		if tc.s != nil {
			S = mat.NewDense(tc.n, tc.m, tc.s)
		}
		ref, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for _, c := range []float64{1e-12, 1e-8, 1e-4, 1e4, 1e8, 1e12} {
			t.Run(fmt.Sprintf("%s/c=%g", tc.name, c), func(t *testing.T) {
				Qc, Rc, Sc := scaled(c, Q), scaled(c, R), scaled(c, S)
				got, err := Dare(A, B, Qc, Rc, &RiccatiOpts{S: Sc})
				if err != nil {
					t.Fatal(err)
				}
				if e := relErr(got.X, scaled(c, ref.X)); !(e <= 1e-10) {
					t.Errorf("X rel err %g against c·X(1)", e)
				}
				if e := relErr(got.X, hewerDare(t, A, B, Qc, Rc, Sc, got.K)); !(e <= 1e-10) {
					t.Errorf("X rel err %g against Hewer", e)
				}
			})
		}
	}
}

// TestDare_NoScalingInaccurateRescued: under NoScaling the unscaled pencil
// returned X with no correct digits and no error (rel err 1 for the cross
// term plant at c ≤ 1e-4, 10 to 9e4 for the n = 2 plant at c ≤ 1e-8);
// a residual above 1e-12 of its terms now retries with cost scaling.
func TestDare_NoScalingInaccurateRescued(t *testing.T) {
	for _, tc := range []struct {
		name          string
		n, m          int
		a, b, q, r, s []float64
		cs            []float64
	}{
		{"cross", 1, 2,
			[]float64{-0.41770152872570837},
			[]float64{-125.82054459896162, -117.5629750252158},
			[]float64{2.2720210405518967e-11},
			[]float64{814.5053775859689, -248.81851916325147, -248.81851916325147, 187.08315058467204},
			[]float64{-0.00013184030272149553, 5.1778514227828224e-05},
			[]float64{1e-4, 1e-8, 1e-12}},
		{"n2", 2, 2,
			[]float64{0.05586567395790124, -0.022978932649404576, -0.0329831762254544, 0.013724345436158403},
			[]float64{-0.43975059735922856, 1.5318187141034458, -0.28898380487617237, 0.09040948177142907},
			[]float64{1.8595480318624465e-10, 5.875102549214723e-12, 5.875102549214723e-12, 1.8561945898874526e-13},
			[]float64{0.5776851433783194, 0.04035942401023552, 0.04035942401023552, 0.24602964169910368},
			nil,
			[]float64{1e-8, 1e-12}},
	} {
		A, B := mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b)
		Q, R := mat.NewDense(tc.n, tc.n, tc.q), mat.NewDense(tc.m, tc.m, tc.r)
		var S *mat.Dense
		if tc.s != nil {
			S = mat.NewDense(tc.n, tc.m, tc.s)
		}
		for _, c := range tc.cs {
			t.Run(fmt.Sprintf("%s/c=%g", tc.name, c), func(t *testing.T) {
				Qc, Rc, Sc := scaled(c, Q), scaled(c, R), scaled(c, S)
				got, err := Dare(A, B, Qc, Rc, &RiccatiOpts{S: Sc, NoScaling: true})
				if err != nil {
					t.Fatal(err)
				}
				if e := relErr(got.X, hewerDare(t, A, B, Qc, Rc, Sc, got.K)); !(e <= 1e-8) {
					t.Errorf("X rel err %g against Hewer", e)
				}
			})
		}
	}
}

// TestCare_FirstPassFailureRescued: Q ≈ 1e-11, R ≈ 1e-13 against ‖A‖ ≈ 1e3
// made the unscaled Hamiltonian's Schur form fail (ErrSchurFailed) while Q
// and R scaled by 1.26e11 solved; plain Care now retries at the cost scale
// of firstSigma, as the pencil path does. X/σ of the scaled problem and
// Newton–Kleinman agree.
func TestCare_FirstPassFailureRescued(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-150.1254541058614, 861.8876560705226, -154.196747075072, -596.3080862186198, 174.50657660928363, -79.14413749733106, 43.468189671386725, 169.51587228776938, -24.147067707965693})
	B := mat.NewDense(3, 1, []float64{-0.032538605969929794, -0.1146638523470365, -0.032744100134825364})
	Q := mat.NewDense(3, 3, []float64{1.6648464612954922e-11, -1.8919379138824958e-11, -5.662071014148134e-12, -1.8919379138824958e-11, 2.1890546288811565e-11, 8.549214239938527e-12, -5.662071014148134e-12, 8.549214239938527e-12, 1.3379022870767825e-11})
	R := mat.NewDense(1, 1, []float64{1.1838247062205125e-13})
	const f = 1.26e11
	var Qs, Rs mat.Dense
	Qs.Scale(f, Q)
	Rs.Scale(f, R)
	ref, err := Care(A, B, &Qs, &Rs, nil)
	if err != nil {
		t.Fatal(err)
	}
	ref.X.Scale(1/f, ref.X)
	want := newtonCare(t, A, B, Q, R, ref.K)
	if e := relErr(ref.X, want); !(e <= 1e-10) {
		t.Fatalf("oracles disagree: rel err %g", e)
	}
	for _, noScaling := range []bool{false, true} {
		t.Run(fmt.Sprintf("NoScaling=%v", noScaling), func(t *testing.T) {
			res, err := Care(A, B, Q, R, &RiccatiOpts{NoScaling: noScaling})
			if err != nil {
				t.Fatal(err)
			}
			if e := relErr(res.X, want); !(e <= 1e-10) {
				t.Errorf("X rel err %g against Newton–Kleinman", e)
			}
			for _, v := range res.Eig {
				if !(real(v) < 0) {
					t.Errorf("closed-loop eigenvalue %v not stable", v)
				}
			}
		})
	}
}

// TestCare_GarbageFirstPassRescued: R ≈ 7.5e-17 against B ≈ 0.1, A
// unstable. The unscaled Schur split returned, without error, an X 1.5e7
// times the true one (Riccati residual ≈ the sum of its terms) and the
// cost-scaled re-solve failed. A first X with residual above 1e-6 of its
// terms is now re-solved at firstSigma and kept only when more accurate.
func TestCare_GarbageFirstPassRescued(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{96.36758671666792, 0.9588728987457393, 62.85981384361142, 21.827959705408684, 44.27011328904998, 62.11146160989517, -67.80386342423793, 38.71178367598355, 81.4545331703082})
	B := mat.NewDense(3, 1, []float64{-0.04160541477814791, -0.17715872321318224, -0.05142743181332761})
	Q := mat.NewDense(3, 3, []float64{4.192814747728362e-13, -1.4573612006267457e-13, -8.411273813530952e-14, -1.4573612006267457e-13, 2.300446620024366e-13, 1.528209332552778e-13, -8.411273813530952e-14, 1.528209332552778e-13, 1.2197388893961272e-13})
	R := mat.NewDense(1, 1, []float64{7.531183056095823e-17})
	const f = 0x1p43
	var Qs, Rs mat.Dense
	Qs.Scale(f, Q)
	Rs.Scale(f, R)
	ref, err := Care(A, B, &Qs, &Rs, nil)
	if err != nil {
		t.Fatal(err)
	}
	ref.X.Scale(1/f, ref.X)
	want := newtonCare(t, A, B, Q, R, ref.K)
	if e := relErr(ref.X, want); !(e <= 1e-10) {
		t.Fatalf("oracles disagree: rel err %g", e)
	}
	for _, noScaling := range []bool{false, true} {
		t.Run(fmt.Sprintf("NoScaling=%v", noScaling), func(t *testing.T) {
			res, err := Care(A, B, Q, R, &RiccatiOpts{NoScaling: noScaling})
			if err != nil {
				t.Fatal(err)
			}
			if e := relErr(res.X, want); !(e <= 1e-10) {
				t.Errorf("X rel err %g against Newton–Kleinman", e)
			}
			for _, v := range res.Eig {
				if !(real(v) < 0) || math.Abs(imag(v)) > 1e3 {
					t.Errorf("closed-loop eigenvalue %v", v)
				}
			}
		})
	}
}
