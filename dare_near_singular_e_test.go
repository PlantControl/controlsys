package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorDareDoublingBig returns the stabilizing solution of the Dare
// with E and cross term S (nil for none) by SDA doubling (dareDoublingHB)
// on the explicit model E⁻¹A, E⁻¹B formed in hbPrec bits, never rounded:
// the descriptor X = E⁻ᵀYE⁻¹, Y = EᵀXE, K and the spectral radius of
// E⁻¹(A − BK). ok reports convergence to a positive definite Y with a
// stable closed loop.
func descriptorDareDoublingBig(A, B, Q, R, S, E *mat.Dense) (X, Y, K *mat.Dense, rho float64, ok bool) {
	e := hbFrom(E)
	a, b, r, s := hbSolve(e, hbFrom(A)), hbSolve(e, hbFrom(B)), hbFrom(R), hbFrom(S)
	y, conv := dareDoublingHB(a, b, hbFrom(Q), r, s, 120)
	if !conv {
		return nil, nil, nil, 0, false
	}
	btx := hbMul(b.t(), y)
	rh := hbMul(btx, a)
	if s.r > 0 {
		rh = hbAdd(rh, s.t(), 1)
	}
	k := hbSolve(hbAdd(r, hbMul(btx, b), 1), rh)
	var ed mat.Eigen
	if !ed.Factorize(hbDense(hbAdd(a, hbMul(b, k), -1)), mat.EigenNone) {
		return nil, nil, nil, 0, false
	}
	for _, v := range ed.Values(nil) {
		rho = max(rho, cmplx.Abs(v))
	}
	z := hbSolve(e.t(), y)
	X = hbDense(hbSolve(e.t(), z.t()).t())
	symmetrize(X.RawMatrix().Data, X.RawMatrix().Rows, X.RawMatrix().Cols)
	return X, hbDense(y), hbDense(k), rho, rho < 1 && hbPositiveDefinite(y)
}

// nearSingularDareCase is a discrete Riccati problem with a near-singular
// E checked against two 320-bit oracles on the explicit model formed
// without rounding: Newton (Hewer) from the returned K and SDA doubling.
type nearSingularDareCase struct {
	A, B, Q, R, S, E *mat.Dense
}

// ulpSpread returns the largest relative move of the Newton oracle's Y, X
// and K under three componentwise 1-ulp perturbations of A, B, E, Q, R
// and S (Q and R kept symmetric).
func (c nearSingularDareCase) ulpSpread(t *testing.T, Y, X, K *mat.Dense) (sY, sX, sK float64) {
	t.Helper()
	rng := rand.New(rand.NewPCG(46, 46))
	perturb := func(a *mat.Dense, sym bool) *mat.Dense {
		if a == nil {
			return nil
		}
		o := mat.DenseCopyOf(a)
		r, cc := o.Dims()
		for i := range r {
			for j := range cc {
				if sym && j < i {
					o.Set(i, j, o.At(j, i))
					continue
				}
				o.Set(i, j, math.Nextafter(o.At(i, j), math.Copysign(math.Inf(1), rng.Float64()-0.5)))
			}
		}
		return o
	}
	for range 3 {
		Yp, Xp, Kp, ok := nearSingularOracle(false, perturb(c.A, false), perturb(c.B, false), perturb(c.Q, true), perturb(c.R, true), perturb(c.S, false), perturb(c.E, false), K)
		if !ok {
			t.Fatalf("Newton oracle of a 1-ulp perturbation does not converge")
		}
		if Y != nil {
			sY = max(sY, relErr(Yp, Y))
		}
		sX, sK = max(sX, relErr(Xp, X)), max(sK, relErr(Kp, K))
	}
	return sY, sX, sK
}

// check runs Dare (descriptor X) and Dlqr (EᵀXE) in both scaling modes and
// requires a value at the given Accuracy (Dare's, Dlqr's) that both
// oracles confirm: within newtonTol at Accuracy 0, which also needs the
// oracle's 1-ulp spread within newtonTol, else within 1e-6.
func (c nearSingularDareCase) check(t *testing.T, accDare, accDlqr int) {
	t.Helper()
	dX, dY, dK, rho, ok := descriptorDareDoublingBig(c.A, c.B, c.Q, c.R, c.S, c.E)
	if !ok {
		t.Fatalf("doubling oracle: no stabilizing solution (ρ %.3g)", rho)
	}
	for _, api := range []string{"Dare", "Dlqr"} {
		for _, ns := range []bool{false, true} {
			label := fmt.Sprintf("%s NoScaling %v", api, ns)
			opts := &RiccatiOpts{S: c.S, E: c.E, NoScaling: ns}
			var res *RiccatiResult
			var err error
			wantAcc := accDare
			if api == "Dare" {
				res, err = Dare(c.A, c.B, c.Q, c.R, opts)
			} else {
				res, err = Dlqr(c.A, c.B, c.Q, c.R, opts)
				wantAcc = accDlqr
			}
			if err != nil {
				t.Errorf("%s: %v", label, err)
				continue
			}
			if res.Accuracy != wantAcc {
				t.Errorf("%s: Accuracy %d, want %d", label, res.Accuracy, wantAcc)
			}
			Y, X, K, conv := nearSingularOracle(false, c.A, c.B, c.Q, c.R, c.S, c.E, res.K)
			if !conv {
				t.Errorf("%s: Newton from the returned K does not converge", label)
				continue
			}
			wantN, wantD := X, dX
			if api == "Dlqr" {
				wantN, wantD = Y, dY
			}
			tol := 1e-6
			if res.Accuracy == 0 {
				tol = newtonTol
				sY, sX, sK := c.ulpSpread(t, Y, X, K)
				s := sX
				if api == "Dlqr" {
					s = sY
				}
				if !(s <= newtonTol && sK <= newtonTol) {
					t.Errorf("%s: Accuracy 0 where 1 ulp moves X by %.3g, K by %.3g", label, s, sK)
				}
			}
			for _, d := range []struct {
				name      string
				got, want *mat.Dense
			}{{"X vs Newton", res.X, wantN}, {"X vs doubling", res.X, wantD}, {"K vs Newton", res.K, K}, {"K vs doubling", res.K, dK}} {
				if rel := relErr(d.got, d.want); !(rel <= tol) {
					t.Errorf("%s: Accuracy %d with %s off by %.3g", label, res.Accuracy, d.name, rel)
				}
			}
			for _, l := range res.Eig {
				if !(cmplx.Abs(l) < 1) {
					t.Errorf("%s: closed-loop eigenvalue %v", label, l)
				}
			}
		}
	}
}

// A stabilizable plant with E of full MATLAB rank, LU cond(E)·eps 0.13,
// whose float64 pencil Dare and Dlqr left with no stabilizing solution
// (errBoundaryUnstabilizable; O2YDYA, IR5YOF repro). ‖X‖ ≈ 4e29, while
// Y = EᵀXE and K are well posed (ρ(E⁻¹(A − BK)) = 0.603); the float64 QZ
// subspace left Y off by 6.5%.
func TestDare_NearSingularEStabilizableIsSolved(t *testing.T) {
	c := nearSingularDareCase{
		A: mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0.3431284077914542, -0.7926130901776904}),
		B: mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, -2.5367599785728485, -0.7294612311954138}),
		E: mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15}),
		Q: eyeDense(2), R: eyeDense(2),
	}
	c.check(t, 0, 0)
}

// The same class at n = 5, σ_min(E) = 1.8e-13 (O2YDYA: seed 1 of the
// wave-40 triangular discrete sweep).
func TestDlqr_NearSingularETriangularIsSolved(t *testing.T) {
	c := nearSingularDareCase{
		A: mat.NewDense(5, 5, []float64{0.8110305471830912, -2.798084983771413, -0.017491933522454567, -0.39791666421793337, -0.8090820362520077, -0.5499181819116858, -0.46262154101836206, -0.8340902193720336, -0.9883516861538988, -2.614357975324414, 0.513330110402648, 2.049045874234565, -0.08641878060699576, -0.5775565235201454, -0.4432567713689527, 0.7056482628882423, -0.597668402517777, 0.7407429714810015, -0.3708537462269688, -0.9877061648126102, 0.6044085481070023, 0.71649527299946, -1.8055876037300738, -1.6494675577286064, -0.016188655813992314}),
		B: mat.NewDense(5, 1, []float64{2.0382094016789454, -0.08179111491178975, 1.1355811750839162, 0.4653174896812371, 0.6252205456119206}),
		E: mat.NewDense(5, 5, []float64{0.728116116855907, 0.1576742650568862, 0.9679861848999214, -0.28435313557425956, -0.5870769808340399, 0, 1.1384134609187906, 0.3285791757945944, -0.8038111554228949, 0.6030483366743167, 0, 0, 0.8753709080953666, 0.0081188653096545, 0.4543347874916437, 0, 0, 0, 0.8612000048726609, -0.674701915132129, 0, 0, 0, 0, 1.801470805444606e-13}),
		Q: eyeDense(5), R: eyeDense(1),
	}
	c.check(t, 0, 0)
}

// The extended-precision solve must not accept a near-infinite mode B
// cannot reach: with A and E upper triangular and B's last row zero, the
// mode a₂₂/e₂₂ ≈ −4.5e14 is exactly uncontrollable and unstable, so Dare
// and Dlqr keep the boundary verdict.
func TestDare_NearSingularEUncontrollableModeIsUnstabilizable(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, 0, 0})
	A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0, -0.7926130901776904})
	Q, R := eyeDense(2), eyeDense(2)
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if _, err := Dare(A, B, Q, R, opts); !errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("Dare NoScaling %v: %v, want an unstabilizable boundary mode", ns, err)
		}
		if _, err := Dlqr(A, B, Q, R, opts); !errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("Dlqr NoScaling %v: %v, want an unstabilizable boundary mode", ns, err)
		}
	}
}

// A well-conditioned E (orthogonal) with a discrete mode at 7e3 weakly
// coupled to B keeps its boundary verdict: the extended solve runs only
// for cond(E)·eps above newtonTol (IR5YOF review 2, zrwGen('big',10876,1),
// where clearing the verdict returned X off by up to 3150% at Accuracy 0).
func TestDare_WellConditionedEKeepsBoundaryVerdict(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-456.1125380107185, 212.12282671008353, -6542.374939554918, 3040.8017276406886})
	B := mat.NewDense(2, 2, []float64{0.1511888949722682, 1.360456698990255, -0.010540390965378899, -0.09484655257220741})
	E := mat.NewDense(2, 2, []float64{0.35739616083221576, 0.9339528811574985, -0.9339528811574985, 0.357396160832216})
	Q, R := eyeDense(2), eyeDense(2)
	for _, ns := range []bool{false, true} {
		if _, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: ns}); !errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("Dare NoScaling %v: %v, want an unstabilizable boundary mode", ns, err)
		}
	}
}

// A rotated near-singular E (cond 5.9e14) with a cross term: X =
// E⁻ᵀYE⁻¹ moves by 4% under 1-ulp perturbations of E, so Dare returns it
// at Accuracy 1, while Y and K, well posed, keep Accuracy 0 in Dlqr
// (O2YDYA sweep ctrl seed 46; base errBoundaryUnstabilizable).
func TestDlqr_NearSingularERotatedCertifiesYNotX(t *testing.T) {
	c := nearSingularDareCase{
		A: mat.NewDense(2, 2, []float64{0.9157029945696862, -0.5986836560374968, 1.314617489524672, 0.18206342374170612}),
		B: mat.NewDense(2, 1, []float64{-0.6180015980622806, -0.30194086085431415}),
		E: mat.NewDense(2, 2, []float64{0.0564948849043164, 0.09424855637151722, -0.359132386009782, -0.599128735017446}),
		Q: mat.NewDense(2, 2, []float64{1.3658176126238195, -0.4667952848832471, -0.4667952848832471, 1.595646110165019}),
		R: mat.NewDense(1, 1, []float64{1.01875062088887}),
		S: mat.NewDense(2, 1, []float64{-0.0828209355705472, 0.10568223311245058}),
	}
	c.check(t, 1, 0)
}

// More inputs than states (m = 3, n = 2) with a cross term: the extended
// solve runs on the given, unreduced problem (O2YDYA sweep ctrl seed 13;
// base errBoundaryUnstabilizable).
func TestDlqr_NearSingularEMoreInputsThanStatesIsSolved(t *testing.T) {
	c := nearSingularDareCase{
		A: mat.NewDense(2, 2, []float64{0.923010477433099, 2.0820636924475, -0.9200537881286719, -1.361812134364599}),
		B: mat.NewDense(2, 3, []float64{-0.7649133317207344, -0.0015390678071096064, 1.2371736568788594, 0.6699795793910823, 0.37405306155207996, 1.3833774115150517}),
		E: mat.NewDense(2, 2, []float64{0.5004371458498397, 2.921975405177471, 0, 8.520053948604121e-13}),
		Q: mat.NewDense(2, 2, []float64{1.8866101723360833, 0.4407050760660495, 0.4407050760660495, 6.019308608388756}),
		R: mat.NewDense(3, 3, []float64{4.7486349226800515, -1.1064545236921663, 1.512245701955511, -1.1064545236921663, 2.42778007790058, 0.6521422583478569, 1.512245701955511, 0.6521422583478569, 3.6215308801989963}),
		S: mat.NewDense(2, 3, []float64{-0.5025960774448042, 0.1337427256625519, -0.18850723325213473, -1.0631702312989222, 0.2829136377269464, -0.3987601331811068}),
	}
	c.check(t, 0, 0)
}

// Kalman of the plant above with C = Bᵀ solves the dual Dare with Eᵀ:
// X = P and Lᵀ must match that Dare's 320-bit solution and gain (base
// errBoundaryUnstabilizable).
func TestKalman_NearSingularEDualIsSolved(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.923010477433099, 2.0820636924475, -0.9200537881286719, -1.361812134364599})
	B := mat.NewDense(2, 3, []float64{-0.7649133317207344, -0.0015390678071096064, 1.2371736568788594, 0.6699795793910823, 0.37405306155207996, 1.3833774115150517})
	E := mat.NewDense(2, 2, []float64{0.5004371458498397, 2.921975405177471, 0, 8.520053948604121e-13})
	Rn := mat.NewDense(3, 3, []float64{4.7486349226800515, -1.1064545236921663, 1.512245701955511, -1.1064545236921663, 2.42778007790058, 0.6521422583478569, 1.512245701955511, 0.6521422583478569, 3.6215308801989963})
	sys, err := NewDescriptor(A, B, mat.DenseCopyOf(B.T()), mat.NewDense(3, 3, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	var gq mat.Dense
	gq.Mul(B, B.T())
	symmetrize(gq.RawMatrix().Data, 2, 2)
	dual := nearSingularDareCase{A: mat.DenseCopyOf(A.T()), B: B, Q: &gq, R: Rn, E: mat.DenseCopyOf(E.T())}
	wantX, _, wantK, rho, ok := descriptorDareDoublingBig(dual.A, dual.B, dual.Q, dual.R, nil, dual.E)
	if !ok {
		t.Fatalf("doubling oracle: no stabilizing solution (ρ %.3g)", rho)
	}
	res, err := Kalman(sys, eyeDense(3), Rn, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	K := mat.DenseCopyOf(res.K.T())
	_, X, Kn, conv := nearSingularOracle(false, dual.A, dual.B, dual.Q, dual.R, nil, dual.E, K)
	if !conv {
		t.Fatal("Newton from the returned gain does not converge")
	}
	tol := 1e-6
	if res.Accuracy == 0 {
		tol = newtonTol
		if _, sX, sK := dual.ulpSpread(t, nil, X, Kn); !(sX <= newtonTol && sK <= newtonTol) {
			t.Errorf("Accuracy 0 where 1 ulp moves X by %.3g, L by %.3g", sX, sK)
		}
	}
	for _, d := range []struct {
		name      string
		got, want *mat.Dense
	}{{"X vs Newton", res.X, X}, {"X vs doubling", res.X, wantX}, {"L vs Newton", K, Kn}, {"L vs doubling", K, wantK}} {
		if rel := relErr(d.got, d.want); !(rel <= tol) {
			t.Errorf("Accuracy %d with %s off by %.3g", res.Accuracy, d.name, rel)
		}
	}
}
