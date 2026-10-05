package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestBalreal_1x1(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(br.HSV) != 1 {
		t.Fatalf("len(HSV) = %d", len(br.HSV))
	}
	if math.Abs(br.HSV[0]-0.5) > 1e-10 {
		t.Errorf("HSV[0] = %g, want 0.5", br.HSV[0])
	}
}

func TestBalreal_2x2_NonSymA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkTinvT(t, br.TR, br.TL, 2, 1e-8)
	checkEigsPreserved(t, sys, br.Sys, 1e-8)
	checkFreqPreserved(t, sys, br.Sys, 1e-8)
	checkGramiansEqual(t, br.Sys, br.HSV, 1e-6)
}

func TestBalreal_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	checkTinvT(t, br.TR, br.TL, 2, 1e-8)
	checkEigsPreserved(t, sys, br.Sys, 1e-8)
}

func TestBalreal_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Balreal(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestBalreal_Empty(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{1}), 0)
	if _, err := Balreal(sys); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}

func TestBalred_Truncation(t *testing.T) {
	sys := make4thOrderSystem()

	red, hsv, err := Balred(sys, 2, BalredOptions{StateProjection: Truncate})
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Errorf("n = %d, want 2", nr)
	}
	if len(hsv) != 4 {
		t.Fatalf("len(HSV) = %d, want 4", len(hsv))
	}

	checkFreqApprox(t, sys, red, 0.3)
}

func TestBalred_SingularPerturbation(t *testing.T) {
	sys := make4thOrderSystem()

	red, _, err := Balred(sys, 2, BalredOptions{StateProjection: MatchDC})
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Errorf("n = %d, want 2", nr)
	}

	origDC, _ := sys.DCGain()
	redDC, _ := red.DCGain()
	assertMatNearT(t, "DCGain", redDC, origDC, 1e-6)
}

func TestBalred_OrderRange(t *testing.T) {
	sys := make4thOrderSystem()
	n, _, _ := sys.Dims()
	dc, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	red, _, err := Balred(sys, 0, BalredOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if nr, _, _ := red.Dims(); nr != 0 {
		t.Fatalf("order 0 MatchDC: n = %d", nr)
	}
	if !mat.EqualApprox(red.D, dc, 1e-10) {
		t.Errorf("order 0 MatchDC gain = %v, want DC gain %v", mat.Formatted(red.D), mat.Formatted(dc))
	}
	red, _, err = Balred(sys, 0, BalredOptions{StateProjection: Truncate})
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(red.D, sys.D, 0) {
		t.Errorf("order 0 Truncate gain = %v, want D", mat.Formatted(red.D))
	}
	for _, order := range []int{-1, n + 1} {
		if _, hsv, err := Balred(sys, order, BalredOptions{}); !errors.Is(err, ErrInvalidOrder) || hsv != nil {
			t.Errorf("order %d: hsv=%v err=%v, want nil, ErrInvalidOrder", order, hsv, err)
		}
	}
	full, hsv, err := Balred(sys, n, BalredOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkGramiansEqual(t, full, hsv, 1e-6)
}

func TestBalred_InvalidOrder(t *testing.T) {
	sys := make4thOrderSystem()

	_, _, err := Balred(sys, 5, BalredOptions{StateProjection: Truncate})
	if !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("got %v, want ErrInvalidOrder", err)
	}
	_, _, err = Balred(sys, -1, BalredOptions{StateProjection: Truncate})
	if !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("got %v, want ErrInvalidOrder", err)
	}
}

func TestModred_Truncation(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -2, 0.3,
			0, 0, -5,
		}),
		mat.NewDense(3, 1, []float64{1, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	red, err := Modred(sys, []int{2}, Truncate)
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Errorf("n = %d, want 2", nr)
	}
}

func TestModred_SingularPerturbation(t *testing.T) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -2, 0.3,
			0, 0, -5,
		}),
		mat.NewDense(3, 1, []float64{1, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	red, err := Modred(sys, []int{2}, MatchDC)
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Errorf("n = %d, want 2", nr)
	}

	origDC, _ := sys.DCGain()
	redDC, _ := red.DCGain()
	assertMatNearT(t, "DCGain", redDC, origDC, 1e-6)
}

func TestModred_Empty(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	red, err := Modred(sys, nil, Truncate)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := red.Dims()
	if n != 2 {
		t.Errorf("n = %d, want 2 (no elimination)", n)
	}
}

func TestModred_AllEliminated(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0.5}), 0)

	red, err := Modred(sys, []int{0, 1}, Truncate)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := red.Dims()
	if n != 0 {
		t.Errorf("n = %d, want 0", n)
	}
}

func TestModred_InvalidIndex(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Modred(sys, []int{5}, Truncate)
	if err == nil {
		t.Error("expected error for out-of-range index")
	}
}

func make4thOrderSystem() *System {
	sys, _ := New(
		mat.NewDense(4, 4, []float64{
			-1, 0.5, 0.1, 0,
			0, -2, 0.3, 0.1,
			0, 0, -10, 1,
			0, 0, -1, -20,
		}),
		mat.NewDense(4, 1, []float64{1, 0.5, 0.1, 0}),
		mat.NewDense(1, 4, []float64{1, 1, 0.5, 0.1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	return sys
}

func checkTinvT(t *testing.T, T, Tinv *mat.Dense, n int, tol float64) {
	t.Helper()
	prod := mat.NewDense(n, n, nil)
	prod.Mul(Tinv, T)
	for i := range n {
		for j := range n {
			want := 0.0
			if i == j {
				want = 1.0
			}
			got := prod.At(i, j)
			if math.Abs(got-want) > tol {
				t.Errorf("Tinv*T[%d,%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func checkEigsPreserved(t *testing.T, orig, bal *System, tol float64) {
	t.Helper()
	p1, _ := orig.Poles()
	p2, _ := bal.Poles()
	if len(p1) != len(p2) {
		t.Fatalf("pole count %d != %d", len(p1), len(p2))
	}
	sort.Slice(p1, func(i, j int) bool {
		if real(p1[i]) != real(p1[j]) {
			return real(p1[i]) < real(p1[j])
		}
		return imag(p1[i]) < imag(p1[j])
	})
	sort.Slice(p2, func(i, j int) bool {
		if real(p2[i]) != real(p2[j]) {
			return real(p2[i]) < real(p2[j])
		}
		return imag(p2[i]) < imag(p2[j])
	})
	for i := range p1 {
		if cmplx.Abs(p1[i]-p2[i]) > tol {
			t.Errorf("pole[%d]: %v != %v", i, p1[i], p2[i])
		}
	}
}

func checkFreqPreserved(t *testing.T, orig, bal *System, tol float64) {
	t.Helper()
	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		g1, _ := orig.EvalFr(complex(0, w))
		g2, _ := bal.EvalFr(complex(0, w))
		_, m1, p1 := orig.Dims()
		for i := range p1 {
			for j := range m1 {
				diff := cmplx.Abs(g1[i][j] - g2[i][j])
				mag := cmplx.Abs(g1[i][j])
				relTol := tol
				if mag > 1e-10 {
					relTol = tol * mag
				}
				if diff > relTol {
					t.Errorf("w=%g: G[%d,%d] diff=%g > tol=%g", w, i, j, diff, relTol)
				}
			}
		}
	}
}

func checkGramiansEqual(t *testing.T, sys *System, hsv []float64, tol float64) {
	t.Helper()
	wcRes, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatalf("Wc: %v", err)
	}
	woRes, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatalf("Wo: %v", err)
	}
	n, _, _ := sys.Dims()
	for i := range n {
		for j := range n {
			want := 0.0
			if i == j {
				want = hsv[i]
			}
			wcVal := wcRes.At(i, j)
			woVal := woRes.At(i, j)
			if math.Abs(wcVal-want) > tol {
				t.Errorf("Wc[%d,%d] = %g, want %g", i, j, wcVal, want)
			}
			if math.Abs(woVal-want) > tol {
				t.Errorf("Wo[%d,%d] = %g, want %g", i, j, woVal, want)
			}
		}
	}
}

func checkFreqApprox(t *testing.T, orig, red *System, relTol float64) {
	t.Helper()
	for _, w := range []float64{0.01, 0.1, 1, 10} {
		g1, _ := orig.EvalFr(complex(0, w))
		g2, _ := red.EvalFr(complex(0, w))
		_, m1, p1 := orig.Dims()
		for i := range p1 {
			for j := range m1 {
				mag := cmplx.Abs(g1[i][j])
				diff := cmplx.Abs(g1[i][j] - g2[i][j])
				if mag > 1e-10 && diff/mag > relTol {
					t.Errorf("w=%g: G[%d,%d] reldiff=%g > %g", w, i, j, diff/mag, relTol)
				}
			}
		}
	}
}

func makePythonControlSystem() *System {
	sys, _ := New(
		mat.NewDense(4, 4, []float64{
			-15, -7.5, -6.25, -1.875,
			8, 0, 0, 0,
			0, 4, 0, 0,
			0, 0, 1, 0,
		}),
		mat.NewDense(4, 1, []float64{2, 0, 0, 0}),
		mat.NewDense(1, 4, []float64{0.5, 0.6875, 0.7031, 0.5}),
		mat.NewDense(1, 1, []float64{0}), 0)
	return sys
}

func TestBalred_PythonControl_Truncate(t *testing.T) {
	sys := makePythonControlSystem()

	red, hsv, err := Balred(sys, 2, BalredOptions{StateProjection: Truncate})
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Fatalf("n = %d, want 2", nr)
	}

	origDC, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	redDC, err := red.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	truncSum := 0.0
	for i := 2; i < len(hsv); i++ {
		truncSum += hsv[i]
	}
	dcBound := 2 * truncSum
	assertMatNearT(t, "DCGain", redDC, origDC, dcBound)

	negC := mat.NewDense(1, 2, nil)
	negC.Scale(-1, red.C)
	negD := mat.NewDense(1, 1, nil)
	negD.Scale(-1, red.D)
	redNeg, _ := New(denseCopy(red.A), denseCopy(red.B), negC, negD, 0)
	errSys, err := Parallel(sys, redNeg)
	if err != nil {
		t.Fatal(err)
	}
	hinfErr, _, err := HinfNorm(errSys)
	if err != nil {
		t.Fatal(err)
	}
	if hinfErr > dcBound {
		t.Errorf("Hinf error = %g, exceeds 2*sum(truncated HSV) = %g", hinfErr, dcBound)
	}
}

func TestBalred_PythonControl_MatchDC(t *testing.T) {
	sys := makePythonControlSystem()

	red, _, err := Balred(sys, 2, BalredOptions{StateProjection: MatchDC})
	if err != nil {
		t.Fatal(err)
	}
	nr, _, _ := red.Dims()
	if nr != 2 {
		t.Fatalf("n = %d, want 2", nr)
	}

	wantDr := mat.NewDense(1, 1, []float64{-0.08383902})
	assertMatNearT(t, "Dr", red.D, wantDr, 1e-4)

	origDC, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	redDC, err := red.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	assertMatNearT(t, "DCGain", redDC, origDC, 1e-4)

	stable, err := red.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("reduced system is not stable")
	}
}

func spReductionPlant(t *testing.T, dt float64) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-1.2, 0.7, 0.1, -0.3, -2.1, 0.4, 0.25, -0.15, -3.3})
	if dt > 0 {
		A.Scale(0.2, A)
		for i := range 3 {
			A.Set(i, i, A.At(i, i)+0.9)
		}
	}
	sys, err := New(A,
		mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1.3, 0.6, -0.8}),
		mat.NewDense(3, 3, []float64{1, 0.5, -0.2, 0, 1.1, 0.3, 0.7, -0.6, 1}),
		mat.NewDense(3, 2, []float64{0.1, 0, 0, -0.2, 0.05, 0.3}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func dcGainOracle(t *testing.T, sys *System) *mat.Dense {
	t.Helper()
	n, _, _ := sys.Dims()
	M := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v := -sys.A.At(i, j)
			if sys.IsDiscrete() && i == j {
				v++
			}
			M.Set(i, j, v)
		}
	}
	var X, G mat.Dense
	if err := X.Solve(M, sys.B); err != nil {
		t.Fatal(err)
	}
	G.Mul(sys.C, &X)
	G.Add(&G, sys.D)
	return &G
}

func TestSingularPerturbation_DCGainMatches(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := spReductionPlant(t, dt)
		want := dcGainOracle(t, sys)
		red, err := Modred(sys, []int{2}, MatchDC)
		if err != nil {
			t.Fatal(err)
		}
		if got := dcGainOracle(t, red); !matEqual(got, want, 1e-12) {
			t.Errorf("dt=%v Modred SP DC gain =\n%v\nwant\n%v", dt, mat.Formatted(got), mat.Formatted(want))
		}
		bred, _, err := Balred(sys, 2, BalredOptions{StateProjection: MatchDC})
		if err != nil {
			t.Fatal(err)
		}
		if got := dcGainOracle(t, bred); !matEqual(got, want, 1e-10) {
			t.Errorf("dt=%v Balred SP DC gain =\n%v\nwant\n%v", dt, mat.Formatted(got), mat.Formatted(want))
		}
	}
}

func TestBalred_FullOrderIsNoOp(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := spReductionPlant(t, dt)
		for _, method := range []StateProjection{Truncate, MatchDC} {
			red, hsv, err := Balred(sys, 3, BalredOptions{StateProjection: method})
			if err != nil {
				t.Fatalf("dt=%v: %v", dt, err)
			}
			if len(hsv) != 3 {
				t.Errorf("dt=%v: hsv len %d, want 3", dt, len(hsv))
			}
			for _, s := range []complex128{complex(0.3, 0.7), complex(-0.2, 2.5)} {
				got, want := pencilResponse(t, red, s), pencilResponse(t, sys, s)
				for i := range want {
					for j := range want[i] {
						if cmplx.Abs(got[i][j]-want[i][j]) > 1e-12 {
							t.Fatalf("dt=%v: response mismatch at s=%v", dt, s)
						}
					}
				}
			}
		}
	}
}

func TestBalredModredRejectInvalidArgs(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	nanSys := sys.Copy()
	nanSys.A.Set(0, 1, math.NaN())
	for name, call := range map[string]func() error{
		"Balred unknown method": func() error {
			_, _, err := Balred(sys, 1, BalredOptions{StateProjection: StateProjection(42)})
			return err
		},
		"Modred unknown method": func() error { _, err := Modred(sys, []int{1}, StateProjection(-1)); return err },
		"Balreal nil":           func() error { _, err := Balreal(nil); return err },
		"Balred nil":            func() error { _, _, err := Balred(nil, 1, BalredOptions{StateProjection: Truncate}); return err },
		"Modred nil":            func() error { _, err := Modred(nil, []int{1}, Truncate); return err },
		"Balreal NaN":           func() error { _, err := Balreal(nanSys); return err },
		"Balred NaN":            func() error { _, _, err := Balred(nanSys, 1, BalredOptions{StateProjection: Truncate}); return err },
	} {
		if err := call(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%s: err = %v, want ErrInvalidArgument", name, err)
		}
	}
}

func TestBalrealTransformConvention(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 2, 0, -0.5, -3, 1, 0, 0.4, -2}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, 1}),
		mat.NewDense(2, 3, []float64{1, 0, 1, 0, 1, 0}),
		mat.NewDense(2, 2, []float64{0.5, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	var tla, ab, bb, cb, eye mat.Dense
	tla.Mul(br.TL, sys.A)
	ab.Mul(&tla, br.TR)
	bb.Mul(br.TL, sys.B)
	cb.Mul(sys.C, br.TR)
	eye.Mul(br.TL, br.TR)
	if !mat.EqualApprox(&ab, br.Sys.A, 1e-9) || !mat.EqualApprox(&bb, br.Sys.B, 1e-9) || !mat.EqualApprox(&cb, br.Sys.C, 1e-9) {
		t.Errorf("Sys != (TL·A·TR, TL·B, C·TR)")
	}
	if !mat.EqualApprox(&eye, eyeDense(3), 1e-9) {
		t.Errorf("TL·TR = %v, want I", mat.Formatted(&eye))
	}
	tl := mat.DenseCopyOf(br.TL)
	tr := mat.DenseCopyOf(br.TR)
	_ = append(br.HSV, make([]float64, 64)...)
	if !mat.Equal(tl, br.TL) || !mat.Equal(tr, br.TR) {
		t.Error("appending to HSV overwrote TL/TR")
	}
	if cap(br.HSV) != len(br.HSV) {
		t.Errorf("HSV cap %d exceeds len %d", cap(br.HSV), len(br.HSV))
	}
}

func TestModredEliminateAll(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
		if dt > 0 {
			A = mat.NewDense(2, 2, []float64{0.5, 0.2, 0, 0.3})
		}
		sys, err := New(A, mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0.5}), mat.NewDense(1, 1, []float64{0.25}), dt)
		if err != nil {
			t.Fatal(err)
		}
		dc, err := sys.DCGain()
		if err != nil {
			t.Fatal(err)
		}
		red, err := Modred(sys, []int{0, 1}, MatchDC)
		if err != nil {
			t.Fatal(err)
		}
		if !mat.EqualApprox(red.D, dc, 1e-12) {
			t.Errorf("dt=%g MatchDC gain = %v, want %v", dt, mat.Formatted(red.D), mat.Formatted(dc))
		}
		red, err = Modred(sys, []int{1, 0}, Truncate)
		if err != nil {
			t.Fatal(err)
		}
		if red.D.At(0, 0) != 0.25 {
			t.Errorf("dt=%g Truncate gain = %g, want D", dt, red.D.At(0, 0))
		}
		for _, elim := range [][]int{{2}, {-1}, {0, 0}} {
			if _, err := Modred(sys, elim, MatchDC); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("elim %v: err = %v, want ErrInvalidArgument", elim, err)
			}
		}
	}
}
