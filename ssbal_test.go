package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSsbal_PoorlyConditioned(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1000, 1, 0, 0.001}),
		mat.NewDense(2, 1, []float64{1000, 0.001}),
		mat.NewDense(1, 2, []float64{1, 1000}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)

	for _, w := range []float64{0.01, 0.1, 1, 10} {
		g1, _ := sys.EvalFr(complex(0, w))
		g2, _ := res.Sys.EvalFr(complex(0, w))
		diff := math.Abs(real(g1[0][0]) - real(g2[0][0]))
		relDiff := diff
		if math.Abs(real(g1[0][0])) > 1e-10 {
			relDiff = diff / math.Abs(real(g1[0][0]))
		}
		if relDiff > 1e-6 {
			t.Errorf("w=%g: freq response mismatch rel=%g", w, relDiff)
		}
	}
}

func TestSsbal_AlreadyBalanced(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		d := res.T.At(i, i)
		if math.Abs(d-1) > 1 {
			t.Errorf("T[%d,%d] = %g, expected near 1 for already balanced system", i, i, d)
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestSsbal_Empty(t *testing.T) {
	sys, err := New(nil, nil, nil, mat.NewDense(1, 1, []float64{3}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	dc, _ := res.Sys.DCGain()
	if dc.At(0, 0) != 3 {
		t.Errorf("dcgain = %g, want 3", dc.At(0, 0))
	}
}

func TestSsbal_TransformIsDiagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 100, 0,
			0.01, -2, 50,
			0, 0.02, -3,
		}),
		mat.NewDense(3, 1, []float64{100, 1, 0.01}),
		mat.NewDense(1, 3, []float64{0.01, 1, 100}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		for j := range n {
			if i != j && res.T.At(i, j) != 0 {
				t.Errorf("T[%d,%d] = %g, want 0 (T should be diagonal)", i, j, res.T.At(i, j))
			}
		}
		if res.T.At(i, i) <= 0 {
			t.Errorf("T[%d,%d] = %g, want positive", i, i, res.T.At(i, i))
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)
}

// TB01ID/MATLAB ssbal: T is diagonal and [T·A/T, T·B; C/T, 0] has row and
// column norms (diagonal excluded) within the Osborne stopping band.
func TestSsbal_BalancesRowColumnNorms(t *testing.T) {
	A0 := mat.NewDense(4, 4, []float64{
		-1.2, 0.7, -0.3, 0.5,
		0.4, -2.1, 0.9, -0.2,
		-0.6, 0.3, -0.8, 1.1,
		0.2, -0.5, 0.4, -1.7,
	})
	B0 := mat.NewDense(4, 2, []float64{1, -0.5, 0.3, 2, -1.4, 0.6, 0.8, 0.1})
	C0 := mat.NewDense(2, 4, []float64{0.7, -1.1, 0.4, 0.9, -0.3, 0.5, 1.6, -0.8})
	D := mat.NewDense(2, 2, []float64{0.2, 0, -0.1, 0.4})
	scale := []float64{1e6, 1e-3, 1e2, 1e-5}
	A, B, C := mat.DenseCopyOf(A0), mat.DenseCopyOf(B0), mat.DenseCopyOf(C0)
	for i, s := range scale {
		for j := range 4 {
			A.Set(i, j, A.At(i, j)*s)
			A.Set(j, i, A.At(j, i)/s)
		}
		for j := range 2 {
			B.Set(i, j, B.At(i, j)*s)
			C.Set(j, i, C.At(j, i)/s)
		}
	}
	for _, dt := range []float64{0, 0.1} {
		sys, err := New(A, B, C, D, dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Ssbal(sys)
		if err != nil {
			t.Fatal(err)
		}
		n, m, p := 4, 2, 2
		for i := range n {
			ti := res.T.At(i, i)
			if fr, _ := math.Frexp(ti); fr != 0.5 {
				t.Errorf("dt=%g: T[%d,%d]=%g not a power of two", dt, i, i, ti)
			}
			for j := range n {
				if want := ti * A.At(i, j) / res.T.At(j, j); res.Sys.A.At(i, j) != want {
					t.Errorf("dt=%g: Ab[%d,%d]=%g, want T·A/T = %g", dt, i, j, res.Sys.A.At(i, j), want)
				}
			}
			for j := range m {
				if want := ti * B.At(i, j); res.Sys.B.At(i, j) != want {
					t.Errorf("dt=%g: Bb[%d,%d]=%g, want T·B = %g", dt, i, j, res.Sys.B.At(i, j), want)
				}
			}
			for j := range p {
				if want := C.At(j, i) / ti; res.Sys.C.At(j, i) != want {
					t.Errorf("dt=%g: Cb[%d,%d]=%g, want C/T = %g", dt, j, i, res.Sys.C.At(j, i), want)
				}
			}
		}
		for i := range n {
			var row, col float64
			for j := range n {
				if j != i {
					row += math.Abs(res.Sys.A.At(i, j))
					col += math.Abs(res.Sys.A.At(j, i))
				}
			}
			for j := range m {
				row += math.Abs(res.Sys.B.At(i, j))
			}
			for j := range p {
				col += math.Abs(res.Sys.C.At(j, i))
			}
			if r := max(row/col, col/row); r > 2.5 {
				t.Errorf("dt=%g state %d: row norm %g, col norm %g (ratio %g)", dt, i, row, col, r)
			}
		}
		for _, w := range []float64{0.01, 0.3, 2, 9} {
			s := complex(0, w)
			if dt > 0 {
				s = cmplx.Exp(complex(0, w*dt))
			}
			g0, _ := sys.EvalFr(s)
			g1, _ := res.Sys.EvalFr(s)
			for i := range p {
				for j := range m {
					if d := cmplx.Abs(g0[i][j] - g1[i][j]); d > 1e-12*max(1, cmplx.Abs(g0[i][j])) {
						t.Errorf("dt=%g ω=%g G[%d,%d] differs by %g", dt, w, i, j, d)
					}
				}
			}
		}
	}
}

func ssbalTestModel(t *testing.T, descriptor bool, dt float64) *System {
	t.Helper()
	A := mat.NewDense(4, 4, []float64{
		-1.2, 0.7, -0.3, 0.5,
		0.4, -2.1, 0.9, -0.2,
		-0.6, 0.3, -0.8, 1.1,
		0.2, -0.5, 0.4, -1.7,
	})
	B := mat.NewDense(4, 2, []float64{1, -0.5, 0.3, 2, -1.4, 0.6, 0.8, 0.1})
	C := mat.NewDense(2, 4, []float64{0.7, -1.1, 0.4, 0.9, -0.3, 0.5, 1.6, -0.8})
	D := mat.NewDense(2, 2, []float64{0.2, 0, -0.1, 0.4})
	var E *mat.Dense
	if descriptor {
		E = mat.NewDense(4, 4, []float64{
			2, 0.3, -0.4, 0.1,
			-0.2, 1.5, 0.6, 0,
			0.5, -0.1, 3, 0.7,
			0, 0.4, -0.3, 0.8,
		})
	}
	scale := []float64{1e6, 1e-3, 1e2, 1e-5}
	for i, s := range scale {
		for j := range 4 {
			A.Set(i, j, A.At(i, j)*s)
			A.Set(j, i, A.At(j, i)/s)
			if E != nil {
				E.Set(i, j, E.At(i, j)*s)
				E.Set(j, i, E.At(j, i)/s)
			}
		}
		for j := range 2 {
			B.Set(i, j, B.At(i, j)*s)
			C.Set(j, i, C.At(j, i)/s)
		}
	}
	sys, err := NewDescriptor(A, B, C, D, E, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// checkSsbalResult checks the exact similarity (T·E/T, T·A/T, T·B, C/T, D),
// power-of-two diagonal T with cond(T) ≤ condT and the frequency response.
func checkSsbalResult(t *testing.T, sys *System, res *SsbalResult, condT float64) {
	t.Helper()
	n, m, p := sys.Dims()
	tmin, tmax := math.Inf(1), 0.0
	for i := range n {
		ti := res.T.At(i, i)
		if fr, _ := math.Frexp(ti); fr != 0.5 {
			t.Errorf("T[%d,%d]=%g not a power of two", i, i, ti)
		}
		tmin, tmax = min(tmin, ti), max(tmax, ti)
		for j := range n {
			if j != i && res.T.At(i, j) != 0 {
				t.Errorf("T[%d,%d]=%g, want 0", i, j, res.T.At(i, j))
			}
			tj := res.T.At(j, j)
			if want := ti * sys.A.At(i, j) / tj; res.Sys.A.At(i, j) != want {
				t.Errorf("Ab[%d,%d]=%g, want T·A/T = %g", i, j, res.Sys.A.At(i, j), want)
			}
			if sys.E != nil {
				if want := ti * sys.E.At(i, j) / tj; res.Sys.E.At(i, j) != want {
					t.Errorf("Eb[%d,%d]=%g, want T·E/T = %g", i, j, res.Sys.E.At(i, j), want)
				}
			}
		}
		for j := range m {
			if want := ti * sys.B.At(i, j); res.Sys.B.At(i, j) != want {
				t.Errorf("Bb[%d,%d]=%g, want T·B = %g", i, j, res.Sys.B.At(i, j), want)
			}
		}
		for j := range p {
			if want := sys.C.At(j, i) / ti; res.Sys.C.At(j, i) != want {
				t.Errorf("Cb[%d,%d]=%g, want C/T = %g", j, i, res.Sys.C.At(j, i), want)
			}
		}
	}
	if (sys.E == nil) != (res.Sys.E == nil) {
		t.Errorf("E nil-ness changed: %v -> %v", sys.E == nil, res.Sys.E == nil)
	}
	if c := tmax / tmin; c > condT {
		t.Errorf("cond(T)=%g exceeds condT=%g", c, condT)
	}
	if !mat.Equal(res.Sys.D, sys.D) || res.Sys.Dt != sys.Dt {
		t.Error("D or Dt changed")
	}
	for _, w := range []float64{0.01, 0.3, 2, 9} {
		s := complex(0, w)
		if sys.Dt > 0 {
			s = cmplx.Exp(complex(0, w*sys.Dt))
		}
		g0, err := sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		g1, err := res.Sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range p {
			for j := range m {
				if d := cmplx.Abs(g0[i][j] - g1[i][j]); d > 1e-12*max(1, cmplx.Abs(g0[i][j])) {
					t.Errorf("ω=%g G[%d,%d] differs by %g", w, i, j, d)
				}
			}
		}
	}
}

// MATLAB ssbal scales E like A and balances [|A|+|E| B; C 0]
// (R14 @ss/ssbal.m, shared/controllib/abcbalance.m).
func TestSsbal_Descriptor(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := ssbalTestModel(t, true, dt)
		res, err := Ssbal(sys)
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		checkSsbalResult(t, sys, res, math.Inf(1))
		n, m, p := sys.Dims()
		for i := range n {
			var row, col float64
			for j := range n {
				if j != i {
					row += math.Abs(res.Sys.A.At(i, j)) + math.Abs(res.Sys.E.At(i, j))
					col += math.Abs(res.Sys.A.At(j, i)) + math.Abs(res.Sys.E.At(j, i))
				}
			}
			for j := range m {
				row += math.Abs(res.Sys.B.At(i, j))
			}
			for j := range p {
				col += math.Abs(res.Sys.C.At(j, i))
			}
			if r := max(row/col, col/row); r > 2.5 {
				t.Errorf("dt=%g state %d: row norm %g, col norm %g (ratio %g)", dt, i, row, col, r)
			}
		}
	}
}

func TestSsbal_IdentityEStaysIdentity(t *testing.T) {
	sys := ssbalTestModel(t, false, 0)
	sysE := sys.Copy()
	sysE.E = eye(4)
	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}
	resE, err := Ssbal(sysE)
	if err != nil {
		t.Fatal(err)
	}
	checkSsbalResult(t, sysE, resE, math.Inf(1))
	if !mat.Equal(resE.Sys.E, eye(4)) {
		t.Errorf("E = %v, want identity", mat.Formatted(resE.Sys.E))
	}
	if !mat.Equal(res.T, resE.T) || !mat.Equal(res.Sys.A, resE.Sys.A) {
		t.Error("explicit identity E changed the balancing")
	}
}

// MATLAB ssbal(sys,condT): cond(T) ≤ condT, condT < 1 treated as 1.
func TestSsbal_CondT(t *testing.T) {
	for _, descriptor := range []bool{false, true} {
		for _, dt := range []float64{0, 0.1} {
			sys := ssbalTestModel(t, descriptor, dt)
			free, err := Ssbal(sys)
			if err != nil {
				t.Fatal(err)
			}
			n, _, _ := sys.Dims()
			freeCond := 0.0
			for i := range n {
				for j := range n {
					freeCond = max(freeCond, free.T.At(i, i)/free.T.At(j, j))
				}
			}
			if freeCond < 1e6 {
				t.Fatalf("unbounded cond(T)=%g too small to exercise condT", freeCond)
			}
			for _, condT := range []float64{1e3, 100, 16, 3, 1, 0.5, math.NaN(), math.Inf(1), 1e300} {
				res, err := Ssbal(sys, WithCondT(condT))
				if err != nil {
					t.Fatal(err)
				}
				bound := condT
				if !(bound >= 1) {
					bound = 1
				}
				checkSsbalResult(t, sys, res, bound)
				if bound >= freeCond && !mat.Equal(res.T, free.T) {
					t.Errorf("condT=%g: inactive bound changed T", condT)
				}
				if bound == 1 && !mat.Equal(res.Sys.A, sys.A) {
					t.Errorf("condT=%g: A changed by a scalar T", condT)
				}
			}
		}
	}
}
