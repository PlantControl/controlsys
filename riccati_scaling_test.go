package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// riccatiScaledRepro returns the Process Lab repro (Z7VJAB) in original
// coordinates and after the state change x' = T·x, T = diag(1e-3, 7, 250).
func riccatiScaledRepro() (A, B, C, As, Bs, Cs, T *mat.Dense) {
	A = mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -1, -3, -3})
	B = mat.NewDense(3, 1, []float64{0, 0, 1})
	C = mat.NewDense(1, 3, []float64{4, 0, 0})
	t := []float64{1e-3, 7, 250}
	T = mat.NewDense(3, 3, nil)
	As, Bs, Cs = mat.NewDense(3, 3, nil), mat.NewDense(3, 1, nil), mat.NewDense(1, 3, nil)
	for i := range 3 {
		T.Set(i, i, t[i])
		for j := range 3 {
			As.Set(i, j, t[i]*A.At(i, j)/t[j])
		}
		Bs.Set(i, 0, t[i]*B.At(i, 0))
		Cs.Set(0, i, C.At(0, i)/t[i])
	}
	return
}

func gram(C *mat.Dense) *mat.Dense {
	var q mat.Dense
	q.Mul(C.T(), C)
	return &q
}

func relErr(got, want mat.Matrix) float64 {
	var d mat.Dense
	d.Sub(got, want)
	return mat.Norm(&d, 1) / mat.Norm(want, 1)
}

// TestRiccati_StateScalingInvariance checks that badly scaled state
// coordinates do not cost accuracy: X' = T⁻ᵀXT⁻¹, K' = KT⁻¹, same poles.
func TestRiccati_StateScalingInvariance(t *testing.T) {
	A, B, C, As, Bs, Cs, T := riccatiScaledRepro()
	R := mat.NewDense(1, 1, []float64{1})
	var Tinv mat.Dense
	if err := Tinv.Inverse(T); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		solve func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
		a, as *mat.Dense
	}{
		{"Lqr", Lqr, A, As},
		{"Dlqr", Dlqr, eulerStep(A, 0.1), nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, as := tc.a, tc.as
			if as == nil {
				as = mat.NewDense(3, 3, nil)
				as.Product(T, a, &Tinv)
			}
			ref, err := tc.solve(a, B, gram(C), R, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tc.solve(as, Bs, gram(Cs), R, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertEigSetNear(t, tc.name, got.Eig, ref.Eig, 1e-12)
			var wantK, wantX mat.Dense
			wantK.Mul(ref.K, &Tinv)
			wantX.Product(&Tinv, ref.X, &Tinv)
			if e := relErr(got.K, &wantK); e > 1e-12 {
				t.Errorf("K rel err %g", e)
			}
			if e := relErr(got.X, &wantX); e > 1e-12 {
				t.Errorf("X rel err %g", e)
			}
		})
	}
}

func TestRiccati_LqeStateScalingInvariance(t *testing.T) {
	A, _, C, As, _, Cs, T := riccatiScaledRepro()
	G := mat.NewDense(3, 1, []float64{0, 0, 1})
	Gs := mat.NewDense(3, 1, nil)
	Gs.Mul(T, G)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1e-2})
	ref, err := Lqe(A, G, C, Qn, Rn, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Lqe(As, Gs, Cs, Qn, Rn, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, "Lqe", got.Eig, ref.Eig, 1e-12)
	var wantL mat.Dense
	wantL.Mul(T, ref.K)
	if e := relErr(got.K, &wantL); e > 1e-12 {
		t.Errorf("L rel err %g", e)
	}
}

func TestRiccati_DescriptorStateScaling(t *testing.T) {
	for _, discrete := range []bool{false, true} {
		A, B, Q, R, S, E := descriptorRiccatiData(discrete)
		n, m := B.Dims()
		d := []float64{1e-3, 7, 250, 0.5, 64}[:n]
		As, Es, Bs, Qs, Ss := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil), mat.NewDense(n, m, nil), mat.NewDense(n, n, nil), mat.NewDense(n, m, nil)
		for i := range n {
			for j := range n {
				As.Set(i, j, d[i]*A.At(i, j)/d[j])
				Es.Set(i, j, d[i]*E.At(i, j)/d[j])
				Qs.Set(i, j, Q.At(i, j)/(d[i]*d[j]))
			}
			for j := range m {
				Bs.Set(i, j, d[i]*B.At(i, j))
				Ss.Set(i, j, S.At(i, j)/d[i])
			}
		}
		solve := Care
		if discrete {
			solve = Dare
		}
		ref, err := solve(A, B, Q, R, &RiccatiOpts{S: S, E: E})
		if err != nil {
			t.Fatal(err)
		}
		got, err := solve(As, Bs, Qs, R, &RiccatiOpts{S: Ss, E: Es})
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, "descriptor", got.Eig, ref.Eig, 1e-10)
		for i := range m {
			for j := range n {
				if w := ref.K.At(i, j) / d[j]; math.Abs(got.K.At(i, j)-w) > 1e-10*mat.Norm(got.K, math.Inf(1)) {
					t.Errorf("discrete=%v K[%d,%d]=%g want %g", discrete, i, j, got.K.At(i, j), w)
				}
			}
		}
	}
}

func TestRiccati_NoScaling(t *testing.T) {
	A, B, Q, R := riccatiTestProblem()
	scaled, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := Care(A, B, Q, R, &RiccatiOpts{NoScaling: true})
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(raw.X, scaled.X); e > 1e-12 {
		t.Errorf("X rel err %g", e)
	}
	if r := careResidual(A, B, Q, R, raw.X); r > 1e-10 {
		t.Errorf("residual %g", r)
	}
}

func eulerStep(A *mat.Dense, h float64) *mat.Dense {
	n, _ := A.Dims()
	Ad := mat.NewDense(n, n, nil)
	Ad.Scale(h, A)
	for i := range n {
		Ad.Set(i, i, Ad.At(i, i)+1)
	}
	return Ad
}

func TestRiccati_NoScalingReachesWrappers(t *testing.T) {
	A, B, C, As, Bs, Cs, _ := riccatiScaledRepro()
	R := mat.NewDense(1, 1, []float64{1})
	ref, err := Lqr(A, B, gram(C), R, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := Lqr(As, Bs, gram(Cs), R, &RiccatiOpts{NoScaling: true})
	if err != nil {
		t.Fatal(err)
	}
	if d := cmplxDist(raw.Eig, ref.Eig); d < 1e-8 {
		t.Errorf("NoScaling Lqr poles match to %g; want the unscaled solver's error", d)
	}
	sys, err := New(As, Bs, Cs, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	one := mat.NewDense(1, 1, []float64{1})
	k, err := Kalman(sys, one, one, nil, &RiccatiOpts{NoScaling: true})
	if err != nil {
		t.Fatal(err)
	}
	ks, err := Kalman(sys, one, one, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if relErr(k.K, ks.K) == 0 {
		t.Error("Kalman ignored NoScaling")
	}
}

func cmplxDist(a, b []complex128) float64 {
	var d float64
	for _, x := range a {
		best := math.Inf(1)
		for _, y := range b {
			best = min(best, cmplxAbs(x-y))
		}
		d = max(d, best)
	}
	return d
}
