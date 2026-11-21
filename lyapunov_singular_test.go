package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// lyapKronBig solves A·X + X·Aᵀ + Q = 0 (discrete: A·X·Aᵀ − X + Q = 0) by
// the 320-bit Kronecker system.
func lyapKronBig(A, Q *mat.Dense, discrete bool) *mat.Dense {
	if discrete {
		return dlyapBig(A, Q)
	}
	a, q := hbFrom(A), hbFrom(Q)
	n := a.r
	M := newHbMat(n*n, n*n)
	rhs := newHbMat(n*n, 1)
	for i := range n {
		for j := range n {
			row := i*n + j
			rhs.d[row].Neg(q.at(i, j))
			for k := range n {
				c := M.at(row, k*n+j)
				c.Add(c, a.at(i, k))
				c = M.at(row, i*n+k)
				c.Add(c, a.at(j, k))
			}
		}
	}
	x := hbSolve(M, rhs)
	out := mat.NewDense(n, n, nil)
	for i := range n * n {
		out.RawMatrix().Data[i], _ = x.d[i].Float64()
	}
	return out
}

func lyapSolveFor(discrete bool) func(A, Q *mat.Dense, opts *LyapunovOpts) (*mat.Dense, error) {
	if discrete {
		return DLyap
	}
	return Lyap
}

// lyapTestQs returns Q = B·Bᵀ, B·Bᵀ − Cᵀ·C, 0 and the consistent
// right-hand side −L(S) of a symmetric S, all exact for dyadic A.
func lyapTestQs(A, B, C, S *mat.Dense, discrete bool) map[string]*mat.Dense {
	n, _ := A.Dims()
	var bbt, ctc, indef mat.Dense
	bbt.Mul(B, B.T())
	ctc.Mul(C.T(), C)
	indef.Sub(&bbt, &ctc)
	var as, cons mat.Dense
	as.Mul(A, S)
	if discrete {
		cons.Mul(&as, A.T())
		cons.Sub(S, &cons)
	} else {
		cons.Add(&as, as.T())
		cons.Scale(-1, &cons)
	}
	return map[string]*mat.Dense{"BBt": &bbt, "indefinite": &indef, "zero": mat.NewDense(n, n, nil), "consistent": &cons}
}

// TestDLyapSingularOperatorErrors (ESR456): the dyadic A of census row
// rlrepd/446 has exact eigenvalues 1 (twice) and −0.5 (twice), so its
// Stein operator is exactly singular; DLyap returned nil with ‖X‖ ≈ 1.9e15
// because a rounded Jordan pair splits the Schur diagonal by ~√ε. Every
// right-hand side, consistent or not, has no unique solution.
func TestDLyapSingularOperatorErrors(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0.625, -0.375, 0.875, -0.125,
		-0.125, 0.375, 0.625, 0.125,
		1.125, -0.375, 0.375, -0.125,
		-0.125, 0.875, 0.625, -0.375,
	})
	B := mat.NewDense(4, 1, []float64{0, 1, 0, 1})
	C := mat.NewDense(1, 4, []float64{1, 2, 1, 2})
	S := mat.NewDense(4, 4, []float64{2, 1, 0, -1, 1, 3, 1, 0, 0, 1, 1, 2, -1, 0, 2, 4})
	for name, Q := range lyapTestQs(A, B, C, S, true) {
		X, err := DLyap(A, Q, nil)
		if !errors.Is(err, ErrSingularEquation) || X != nil {
			t.Errorf("%s: DLyap = %v, %v; want nil, ErrSingularEquation", name, X, err)
		}
	}
}

// TestLyapSingularOperatorErrors (ESR456): A = M·D·M⁻¹ with integer unit
// triangular M, so A is exact; D has eigenvalues 1 and −1 (λᵢ + λⱼ = 0),
// or a nilpotent Jordan block, and the continuous operator is singular.
func TestLyapSingularOperatorErrors(t *testing.T) {
	M := mat.NewDense(3, 3, []float64{1, 0, 0, 2, 1, 0, -1, 3, 1})
	var Mi mat.Dense
	if err := Mi.Inverse(M); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		d    []float64
	}{
		{"pair", []float64{1, 0.5, 0, 0, -1, 0.25, 0, 0, -2}},
		{"jordan", []float64{0, 1, 0, 0, 0, 0.5, 0, 0, -3}},
	} {
		var md, A mat.Dense
		md.Mul(M, mat.NewDense(3, 3, tc.d))
		A.Mul(&md, &Mi)
		B := mat.NewDense(3, 1, []float64{1, -1, 2})
		C := mat.NewDense(1, 3, []float64{0.5, 1, -1})
		S := mat.NewDense(3, 3, []float64{2, 1, 0, 1, 3, -1, 0, -1, 1})
		for name, Q := range lyapTestQs(&A, B, C, S, false) {
			X, err := Lyap(&A, Q, nil)
			if !errors.Is(err, ErrSingularEquation) || X != nil {
				t.Errorf("%s/%s: Lyap = %v, %v; want nil, ErrSingularEquation", tc.name, name, X, err)
			}
		}
	}
}

// TestDLyapTiedClusterErrors (ESR456): census row tied/171 (arm64 data), a
// rounded model whose Stein operator is within rounding of singular; DLyap
// returned nil with an X far from the 320-bit solution.
func TestDLyapTiedClusterErrors(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{
		0.41860652923218705, 0.0013529913450435315, -0.25778226835560325, 0.18334441181870753, -1.1371199885494612, -0.22081668708892083,
		1.3587473856658114, -0.12761050358150167, -1.0526426997029537, 0.7538823412425684, 0.7418721893290241, -0.7610527017000616,
		0.10555506089744976, -0.9292427277654646, -0.3662563993784492, 0.3922707807584473, 0.10013613905903195, -0.746947530793947,
		-0.8694933746668019, -0.7113576625892166, 0.4821046562080618, -0.35034564424033077, 0.7142780300360617, 0.4105513211626867,
		-0.4658175872813111, 0.45723919990515444, -0.29861630319346716, -0.037707534410997716, 0.30594173103352246, 0.2866264158096625,
		-0.0023443830758380084, 0.13652613480718112, -0.32979764076210083, 0.5079370697056077, 1.2632269019909301, 0.11966428693275451,
	})
	B := mat.NewDense(6, 2, []float64{-1, 2, 0.5, 0.5, 0, 0.5, 0, 2, 1, 1, 0.5, -1})
	C := mat.NewDense(1, 6, []float64{-1, 1, -1, 2, -1, 2})
	var bbt, indef, ctc mat.Dense
	bbt.Mul(B, B.T())
	ctc.Mul(C.T(), C)
	indef.Sub(&bbt, &ctc)
	for name, Q := range map[string]*mat.Dense{"BBt": &bbt, "indefinite": &indef} {
		X, err := DLyap(A, Q, nil)
		if !errors.Is(err, ErrSingularEquation) || X != nil {
			t.Errorf("%s: DLyap = %v, %v; want nil, ErrSingularEquation", name, X, err)
		}
	}
}

// lyapNearJordan returns S·J·S⁻¹ (rounded) for J a near-Jordan pair at
// −a (continuous) or 1 − a (discrete) beside a well-separated mode.
func lyapNearJordan(t *testing.T, a float64, discrete bool) *mat.Dense {
	t.Helper()
	S := mat.NewDense(3, 3, []float64{1, 0.3, -0.7, 0.2, 1, 0.45, -0.6, 0.1, 1})
	var Si, sj, A mat.Dense
	if err := Si.Inverse(S); err != nil {
		t.Fatal(err)
	}
	J := mat.NewDense(3, 3, []float64{-a, 1, 0, 0, -a, 0, 0, 0, -1})
	if discrete {
		J = mat.NewDense(3, 3, []float64{1 - a, 0.5, 0, 0, 1 - a, 0, 0, 0, 0.5})
	}
	sj.Mul(S, J)
	A.Mul(&sj, &Si)
	return &A
}

// TestLyapIllConditionedRefinedAccurate (ESR456): a near-Jordan pair makes
// the equation ill-conditioned (sep about 3e3·ε·s, outside the band) and
// the plain Schur solve 1e-8 off; Lyap and DLyap refine it to the 320-bit
// solution.
func TestLyapIllConditionedRefinedAccurate(t *testing.T) {
	Q := mat.NewDense(3, 3, []float64{2, 0.5, 0.1, 0.5, 1, -0.3, 0.1, -0.3, 3})
	for _, tc := range []struct {
		name     string
		a        float64
		discrete bool
	}{
		{"continuous", 2e-4, false},
		{"discrete", 1e-4, true},
	} {
		A := lyapNearJordan(t, tc.a, tc.discrete)
		X, err := lyapSolveFor(tc.discrete)(A, Q, nil)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if e := relErr(X, lyapKronBig(A, Q, tc.discrete)); !(e <= 1e-10) {
			t.Errorf("%s: X rel err %g against 320-bit Kronecker solve", tc.name, e)
		}
	}
}

// TestLyapWellConditionedUnrefined (ESR456): when the comparison bound
// certifies the plain solve, Lyap and DLyap return it bit for bit.
func TestLyapWellConditionedUnrefined(t *testing.T) {
	Q := mat.NewDense(3, 3, []float64{2, 0.5, 0.1, 0.5, 1, -0.3, 0.1, -0.3, 3})
	for _, discrete := range []bool{false, true} {
		A := mat.NewDense(3, 3, []float64{-1.2, 0.4, 0.1, -0.3, -0.8, 0.5, 0.2, -0.1, -2})
		if discrete {
			A = mat.NewDense(3, 3, []float64{0.3, 0.2, -0.1, -0.15, 0.5, 0.05, 0.1, 0.2, -0.4})
		}
		want, _, err := solveLyapunov("Lyap", A, Q, nil, discrete)
		if err != nil {
			t.Fatal(err)
		}
		got, err := lyapSolveFor(discrete)(A, Q, nil)
		if err != nil {
			t.Fatalf("discrete=%v: %v", discrete, err)
		}
		if !mat.Equal(got, want) {
			t.Errorf("discrete=%v: certified X differs from the plain solve", discrete)
		}
	}
}

// TestLyapRefinementStallUnverified (ESR456): corrections solved with an
// operator 2.5 times too large contract by 0.6 per step, which refinement
// reads as a stall; it returns ErrAccuracyUnverified with the iterate of
// smallest correction, no worse than the start.
func TestLyapRefinementStallUnverified(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1.2, 0.4, 0.1, -0.3, -0.8, 0.5, 0.2, -0.1, -2})
	Q := mat.NewDense(3, 3, []float64{2, 0.5, 0.1, 0.5, 1, -0.3, 0.1, -0.3, 3})
	X, s, err := solveLyapunov("Lyap", A, Q, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	want := lyapKronBig(A, Q, false)
	x := X.RawMatrix().Data
	for i := range x {
		x[i] *= 1.01
	}
	start := relErr(X, want)
	for i := range s.t {
		s.t[i] *= 2.5
	}
	err = refineLyapunov(&s, A, Q, x, nil, math.Inf(1), ErrSingularEquation)
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("err = %v, want ErrAccuracyUnverified", err)
	}
	if e := relErr(X, want); !(e < start) {
		t.Errorf("returned iterate rel err %g, start %g", e, start)
	}
}

// TestUnverifiedLyapunovSplit (ESR456): consumers keep their value only
// for ErrAccuracyUnverified.
func TestUnverifiedLyapunovSplit(t *testing.T) {
	_, unv := unverifiedLyapunov(fmt.Errorf("x: %w", ErrAccuracyUnverified))
	if unv == nil {
		t.Error("ErrAccuracyUnverified not split off")
	}
	if hard, _ := unverifiedLyapunov(ErrSingularEquation); !errors.Is(hard, ErrSingularEquation) {
		t.Error("ErrSingularEquation not kept as the hard error")
	}
}

// TestGramSingularWithinRoundingErrors (ESR456, user decision U2): the
// stable discrete model with a near-Jordan pair at 1 − 1e-7 has a Stein
// operator within rounding of singular in its stored and balanced forms;
// the gramian consumers return ErrSingularEquation, not a value.
func TestGramSingularWithinRoundingErrors(t *testing.T) {
	A := lyapNearJordan(t, 1e-7, true)
	B := mat.NewDense(3, 1, []float64{1, -1, 0.5})
	C := mat.NewDense(1, 3, []float64{0.5, 1, -1})
	sys, err := New(A, B, C, mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Gram(sys, GramControllability); !errors.Is(err, ErrSingularEquation) {
		t.Errorf("Gram: %v, want ErrSingularEquation", err)
	}
	if _, err := H2Norm(sys); !errors.Is(err, ErrSingularEquation) {
		t.Errorf("H2Norm: %v, want ErrSingularEquation", err)
	}
	if _, err := HSV(sys); !errors.Is(err, ErrSingularEquation) {
		t.Errorf("HSV: %v, want ErrSingularEquation", err)
	}
	if _, err := Covar(sys, mat.NewDense(1, 1, []float64{1})); !errors.Is(err, ErrSingularEquation) {
		t.Errorf("Covar: %v, want ErrSingularEquation", err)
	}
}

// TestLyapBalancedRetryScaledEquation (ESR456): a well-posed equation under
// the exact similarity diag(2³⁰, 1, 2⁻³⁰) is within ε·‖L‖ of singular in
// the stored scaling; the balanced retry certifies X to the 320-bit
// solution in the stored scaling.
func TestLyapBalancedRetryScaledEquation(t *testing.T) {
	d := []float64{0x1p30, 1, 0x1p-30}
	for _, discrete := range []bool{false, true} {
		a := []float64{-1.2, 0.4, 0.1, -0.3, -0.8, 0.5, 0.2, -0.1, -2}
		if discrete {
			a = []float64{0.3, 0.2, -0.1, -0.15, 0.5, 0.05, 0.1, 0.2, -0.4}
		}
		A := mat.NewDense(3, 3, nil)
		Q := mat.NewDense(3, 3, nil)
		q := []float64{2, 0.5, 0.1, 0.5, 1, -0.3, 0.1, -0.3, 3}
		for i := range 3 {
			for j := range 3 {
				A.Set(i, j, d[i]*a[i*3+j]/d[j])
				Q.Set(i, j, d[i]*q[i*3+j]*d[j])
			}
		}
		raw, s, err := solveLyapunov("Lyap", A, Q, nil, discrete)
		if err == nil {
			err = certifyLyapunov(&s, A, Q, raw.RawMatrix().Data, nil)
		}
		if !errors.Is(err, ErrSingularEquation) {
			t.Fatalf("discrete=%v: raw certificate %v, want the band", discrete, err)
		}
		X, err := lyapSolveFor(discrete)(A, Q, nil)
		if err != nil {
			t.Fatalf("discrete=%v: %v", discrete, err)
		}
		if e := relErr(X, lyapKronBig(A, Q, discrete)); !(e <= 1e-10) {
			t.Errorf("discrete=%v: X rel err %g against 320-bit Kronecker solve", discrete, e)
		}
	}
}

// TestBalanceLyapunovIdentity (ESR456): a matrix already balanced, or one
// whose balance is a multiple of I, gives no retry.
func TestBalanceLyapunovIdentity(t *testing.T) {
	a := []float64{-1, 0.5, 0.5, -2}
	if d := balanceLyapunov(a, 2); d != nil {
		t.Errorf("balanced 2×2: D = %v, want nil", d)
	}
}
