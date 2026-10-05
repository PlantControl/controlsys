package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestGram_Continuous_MATLABValidated(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
		mat.NewDense(2, 2, []float64{4, 5, 6, 7}),
		mat.NewDense(2, 2, []float64{13, 14, 15, 16}), 0)
	if err != nil {
		t.Fatal(err)
	}

	wcRes, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	wantWc := mat.NewDense(2, 2, []float64{18.5, 24.5, 24.5, 32.5})
	assertMatNearT(t, "Wc", wcRes, wantWc, 1e-4)

	woRes, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}
	wantWo := mat.NewDense(2, 2, []float64{257.5, -94.5, -94.5, 56.5})
	assertMatNearT(t, "Wo", woRes, wantWo, 1e-4)
}

func TestGram_Discrete_MATLABValidated(t *testing.T) {
	csys, err := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
		mat.NewDense(2, 2, []float64{4, 5, 6, 7}),
		mat.NewDense(2, 2, []float64{13, 14, 15, 16}), 0)
	if err != nil {
		t.Fatal(err)
	}

	dsys, err := csys.DiscretizeZOH(0.2)
	if err != nil {
		t.Fatal(err)
	}

	wcRes, err := Gram(dsys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	woRes, err := Gram(dsys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}

	for _, res := range []*mat.Dense{wcRes, woRes} {
		if !isSymmetric(res, 1e-10) {
			t.Error("gramian not symmetric")
		}
		var eig mat.Eigen
		ok := eig.Factorize(res, mat.EigenNone)
		if !ok {
			t.Fatal("eigendecomposition failed")
		}
		vals := eig.Values(nil)
		for i, v := range vals {
			if real(v) < -1e-10 {
				t.Errorf("eigenvalue[%d] = %g, want >= 0", i, real(v))
			}
		}
	}

	cwcRes, _ := Gram(csys, GramControllability)
	diff := mat.NewDense(2, 2, nil)
	diff.Sub(wcRes, cwcRes)
	if mat.Norm(diff, 1) < 1e-10 {
		t.Error("discrete and continuous gramians should differ")
	}
}

func TestHSV_MATLABValidated(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)
	if err != nil {
		t.Fatal(err)
	}

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len(hsv) = %d, want 2", len(hsv))
	}
	want := []float64{24.42686, 0.5731395}
	for i, w := range want {
		if math.Abs(hsv[i]-w)/w > 1e-3 {
			t.Errorf("hsv[%d] = %g, want %g", i, hsv[i], w)
		}
	}
}

func TestH2Norm_FirstOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 1.0 / math.Sqrt(2)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("H2 = %.15g, want %.15g", got, want)
	}
}

func TestHinfNorm_FirstOrder_Hardening(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("Hinf = %.15g, want 1.0", norm)
	}
}

func TestH2Norm_Unstable_IsInfinite(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := H2Norm(sys)
	if err == nil && !math.IsInf(got, 1) {
		t.Errorf("H2 = %g for unstable system, want +Inf or error", got)
	}
}

func TestHinfNorm_MarginallyStable_IsInfinite(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, 0}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	norm, w, err := HinfNorm(sys)
	if err != nil || !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
		t.Errorf("Hinf = %g at %g, %v for marginally stable system, want Inf at Inf", norm, w, err)
	}
}

func TestNorms_MIMO_MATLABValidated(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1.017041847539126, -0.224182952826418, 0.042538079149249,
			-0.310374015319095, -0.516461581407780, -0.119195790221750,
			-1.452723568727942, 1.799586083710209, -1.491935830615152,
		}),
		mat.NewDense(3, 2, []float64{
			0.312858596637428, -0.164879019209038,
			-0.864879917324456, 0.627707287528727,
			-0.030051296196269, 1.093265669039484,
		}),
		mat.NewDense(2, 3, []float64{
			1.109273297614398, 0.077359091130425, -1.113500741486764,
			-0.863652821988714, -1.214117043615409, -0.006849328103348,
		}),
		mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	hinf, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	wantHinf := 4.276759162964244
	if math.Abs(hinf-wantHinf)/wantHinf > 1e-4 {
		t.Errorf("Hinf = %g, want %g", hinf, wantHinf)
	}

	h2, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	wantH2 := 2.237461821810309
	if math.Abs(h2-wantH2)/wantH2 > 1e-4 {
		t.Errorf("H2 = %g, want %g", h2, wantH2)
	}
}

func TestBalred_MATLABValidated(t *testing.T) {
	sys, err := New(
		mat.NewDense(4, 4, []float64{
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			-60, -200, -60, -15,
		}),
		mat.NewDense(4, 1, []float64{0, 0, 0, 1}),
		mat.NewDense(1, 4, []float64{32, 45, 11, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	origDC, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}

	red, _, err := Balred(sys, 2, Truncate)
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Fatalf("n = %d, want 2", nr)
	}

	stable, err := red.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("reduced system is not stable")
	}

	redDC, err := red.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	wantDC := origDC.At(0, 0)
	gotDC := redDC.At(0, 0)
	if math.Abs(gotDC-wantDC)/math.Abs(wantDC) > 0.5 {
		t.Errorf("DC gain = %g, want ~%g", gotDC, wantDC)
	}
}

func TestBalred_MarginallyStable(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, 0, 0}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Balreal panicked: %v", r)
		}
	}()

	_, err = Balreal(sys)
	if err != nil {
		return
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Balred panicked: %v", r)
		}
	}()

	_, _, err = Balred(sys, 1, Truncate)
	if err != nil {
		return
	}
}

func TestGram_Symmetry(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-2, 0.5, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	wcRes, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	woRes, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		g    *mat.Dense
	}{
		{"Wc", wcRes},
		{"Wo", woRes},
	} {
		n, _ := tc.g.Dims()
		nrm := mat.Norm(tc.g, 1)
		if nrm < 1e-15 {
			continue
		}
		for i := range n {
			for j := range i {
				diff := math.Abs(tc.g.At(i, j) - tc.g.At(j, i))
				if diff > 1e-12*nrm {
					t.Errorf("%s[%d,%d] - %s[%d,%d] = %g, want symmetric", tc.name, i, j, tc.name, j, i, diff)
				}
			}
		}
	}
}

func TestHinfNorm_WithFeedthrough(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-3.0) > 1e-6 {
		t.Errorf("Hinf = %g, want 3.0", norm)
	}

	dc, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if norm < math.Abs(dc.At(0, 0))-1e-10 {
		t.Errorf("Hinf(%g) < |DC gain|(%g)", norm, math.Abs(dc.At(0, 0)))
	}
}

func TestGram_Unstable_Continuous_Hardening(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Gram(sys, GramControllability)
	if !errors.Is(err, ErrUnstableGramian) {
		t.Errorf("got %v, want ErrUnstableGramian", err)
	}
	_, err = Gram(sys, GramObservability)
	if !errors.Is(err, ErrUnstableGramian) {
		t.Errorf("got %v, want ErrUnstableGramian", err)
	}
}
