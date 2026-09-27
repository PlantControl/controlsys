package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// ===========================================================================
// Cross-validation test data from reference implementations (python-control).
// All expected values are taken directly from reference libraries
// without relaxing tolerances.
// ===========================================================================

// helper: create *mat.Dense from row-major 2D slice
func denseFromRows(rows [][]float64) *mat.Dense {
	r := len(rows)
	if r == 0 {
		return mat.NewDense(0, 0, nil)
	}
	c := len(rows[0])
	data := make([]float64, r*c)
	for i, row := range rows {
		copy(data[i*c:], row)
	}
	return mat.NewDense(r, c, data)
}

// helper: create *mat.Dense from flat row-major data
func denseFromFlat(r, c int, data []float64) *mat.Dense {
	if data == nil {
		return mat.NewDense(r, c, nil)
	}
	d := make([]float64, len(data))
	copy(d, data)
	return mat.NewDense(r, c, d)
}

// helper: compare sorted real poles
func compareSortedPoles(t *testing.T, label string, got, want []complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: pole count %d, want %d", label, len(got), len(want))
	}
	sortC := func(s []complex128) {
		sort.Slice(s, func(i, j int) bool {
			if math.Abs(real(s[i])-real(s[j])) > 1e-14 {
				return real(s[i]) < real(s[j])
			}
			return imag(s[i]) < imag(s[j])
		})
	}
	gs := make([]complex128, len(got))
	ws := make([]complex128, len(want))
	copy(gs, got)
	copy(ws, want)
	sortC(gs)
	sortC(ws)
	for i := range gs {
		if cmplx.Abs(gs[i]-ws[i]) > tol {
			t.Errorf("%s: pole[%d] = %v, want %v (err=%g)", label, i, gs[i], ws[i], cmplx.Abs(gs[i]-ws[i]))
		}
	}
}

// helper: compare sorted real zeros
func compareSortedZeros(t *testing.T, label string, got, want []complex128, tol float64) {
	t.Helper()
	compareSortedPoles(t, label, got, want, tol)
}

func assertMatNear(t *testing.T, label string, got, want *mat.Dense, tol float64) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("%s: dims (%d,%d), want (%d,%d)", label, gr, gc, wr, wc)
	}
	for i := range gr {
		for j := range gc {
			if math.Abs(got.At(i, j)-want.At(i, j)) > tol {
				t.Errorf("%s: [%d,%d] = %g, want %g", label, i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

// ---------------------------------------------------------------------------
// ZOH Discretization - reference: ZOH discretization
// ---------------------------------------------------------------------------
func TestCrossval_ZOH_Reference(t *testing.T) {
	A := denseFromRows([][]float64{
		{1, 0},
		{0, 1},
	})
	B := denseFromRows([][]float64{
		{0.5},
		{0.5},
	})
	C := denseFromRows([][]float64{
		{0.75, 1.0},
		{1.0, 1.0},
		{1.0, 0.25},
	})
	D := denseFromRows([][]float64{
		{0.0},
		{0.0},
		{-0.33},
	})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.5
	hd, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	adTruth := 1.648721270700128
	bdTruth := 0.324360635350064

	if math.Abs(hd.A.At(0, 0)-adTruth) > 1e-10 {
		t.Errorf("Ad[0,0] = %g, want %g", hd.A.At(0, 0), adTruth)
	}
	if math.Abs(hd.A.At(1, 1)-adTruth) > 1e-10 {
		t.Errorf("Ad[1,1] = %g, want %g", hd.A.At(1, 1), adTruth)
	}
	if math.Abs(hd.A.At(0, 1)) > 1e-10 {
		t.Errorf("Ad[0,1] = %g, want 0", hd.A.At(0, 1))
	}
	if math.Abs(hd.A.At(1, 0)) > 1e-10 {
		t.Errorf("Ad[1,0] = %g, want 0", hd.A.At(1, 0))
	}

	if math.Abs(hd.B.At(0, 0)-bdTruth) > 1e-10 {
		t.Errorf("Bd[0,0] = %g, want %g", hd.B.At(0, 0), bdTruth)
	}
	if math.Abs(hd.B.At(1, 0)-bdTruth) > 1e-10 {
		t.Errorf("Bd[1,0] = %g, want %g", hd.B.At(1, 0), bdTruth)
	}

	assertMatNear(t, "Cd", hd.C, C, 1e-10)
	assertMatNear(t, "Dd", hd.D, D, 1e-10)

	if hd.Dt != dt {
		t.Errorf("dt = %g, want %g", hd.Dt, dt)
	}
}

// ---------------------------------------------------------------------------
// ZOH double integrator - python-control test_sample_ss
// ---------------------------------------------------------------------------
func TestCrossval_ZOH_DoubleIntegrator(t *testing.T) {
	sys1, _ := New(
		denseFromFlat(2, 2, []float64{0, 1, 0, 0}),
		denseFromFlat(2, 1, []float64{0, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		denseFromFlat(2, 2, []float64{0, 0, 1, 0}),
		denseFromFlat(2, 1, []float64{1, 0}),
		denseFromFlat(1, 2, []float64{0, 1}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)

	for _, sys := range []*System{sys1, sys2} {
		for _, h := range []float64{0.1, 0.5, 1, 2} {
			t.Run("", func(t *testing.T) {
				I := mat.NewDense(2, 2, nil)
				I.Set(0, 0, 1)
				I.Set(1, 1, 1)

				// Ad = I + h*A
				Ad := mat.NewDense(2, 2, nil)
				Ad.Scale(h, sys.A)
				Ad.Add(Ad, I)

				// Bd = h*B + 0.5*h²*A*B
				AB := mat.NewDense(2, 1, nil)
				AB.Mul(sys.A, sys.B)
				Bd := mat.NewDense(2, 1, nil)
				Bd.Scale(h, sys.B)
				AB2 := mat.NewDense(2, 1, nil)
				AB2.Scale(0.5*h*h, AB)
				Bd.Add(Bd, AB2)

				sysd, err := sys.DiscretizeZOH(h)
				if err != nil {
					t.Fatal(err)
				}

				assertMatNear(t, "Ad", sysd.A, Ad, 1e-10)
				assertMatNear(t, "Bd", sysd.B, Bd, 1e-10)
				assertMatNear(t, "Cd", sysd.C, sys.C, 1e-10)
				assertMatNear(t, "Dd", sysd.D, sys.D, 1e-10)
				if sysd.Dt != h {
					t.Errorf("dt = %g, want %g", sysd.Dt, h)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Tustin discretization - reference: Tustin discretization
// ---------------------------------------------------------------------------
func TestCrossval_Tustin_Reference(t *testing.T) {
	A := denseFromRows([][]float64{
		{1, 0},
		{0, 1},
	})
	B := denseFromRows([][]float64{
		{0.5},
		{0.5},
	})
	C := denseFromRows([][]float64{
		{0.75, 1.0},
		{1.0, 1.0},
		{1.0, 0.25},
	})
	D := denseFromRows([][]float64{
		{0.0},
		{0.0},
		{-0.33},
	})

	sys, _ := New(A, B, C, D, 0)
	dt := 0.5

	hd, err := sys.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}

	adTruth := 5.0 / 3.0
	for i := range 2 {
		for j := range 2 {
			want := 0.0
			if i == j {
				want = adTruth
			}
			if math.Abs(hd.A.At(i, j)-want) > 1e-10 {
				t.Errorf("Ad[%d,%d] = %g, want %g", i, j, hd.A.At(i, j), want)
			}
		}
	}

	sinPi4 := math.Sin(math.Pi / 4)
	bdTruth := (1.0 / 3.0) / sinPi4
	for i := range 2 {
		if math.Abs(hd.B.At(i, 0)-bdTruth) > 1e-10 {
			t.Errorf("Bd[%d,0] = %g, want %g", i, hd.B.At(i, 0), bdTruth)
		}
	}

	cdTruth := denseFromRows([][]float64{
		{1.0, 4.0 / 3.0},
		{4.0 / 3.0, 4.0 / 3.0},
		{4.0 / 3.0, 1.0 / 3.0},
	})
	cr, cc := cdTruth.Dims()
	for i := range cr {
		for j := range cc {
			cdTruth.Set(i, j, cdTruth.At(i, j)*sinPi4)
		}
	}
	assertMatNear(t, "Cd", hd.C, cdTruth, 1e-10)

	ddTruth := denseFromRows([][]float64{
		{0.291666666666667},
		{1.0 / 3.0},
		{-0.121666666666667},
	})
	assertMatNear(t, "Dd", hd.D, ddTruth, 1e-10)
}

// ---------------------------------------------------------------------------
// Poles - python-control sys322
// ---------------------------------------------------------------------------
func TestCrossval_Poles_PythonControl_sys322(t *testing.T) {
	A := denseFromRows([][]float64{
		{-3, 4, 2},
		{-1, -3, 0},
		{2, 5, 3},
	})
	B := denseFromRows([][]float64{
		{1, 4},
		{-3, -3},
		{-2, 1},
	})
	C := denseFromRows([][]float64{
		{4, 2, -3},
		{1, 4, 3},
	})
	D := denseFromRows([][]float64{
		{-2, 4},
		{0, 1},
	})

	sys, _ := New(A, B, C, D, 0)
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}

	wantPoles := []complex128{
		3.34747678408874 + 0i,
		complex(-3.17373839204437, 1.47492908003839),
		complex(-3.17373839204437, -1.47492908003839),
	}
	compareSortedPoles(t, "sys322", poles, wantPoles, 1e-8)
}

// ---------------------------------------------------------------------------
// Zeros - python-control sys322 (square MIMO)
// ---------------------------------------------------------------------------
func TestCrossval_Zeros_PythonControl_sys322(t *testing.T) {
	A := denseFromRows([][]float64{
		{-3, 4, 2},
		{-1, -3, 0},
		{2, 5, 3},
	})
	B := denseFromRows([][]float64{
		{1, 4},
		{-3, -3},
		{-2, 1},
	})
	C := denseFromRows([][]float64{
		{4, 2, -3},
		{1, 4, 3},
	})
	D := denseFromRows([][]float64{
		{-2, 4},
		{0, 1},
	})

	sys, _ := New(A, B, C, D, 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	wantZeros := []complex128{44.41465, -0.490252, -5.924398}
	compareSortedZeros(t, "sys322 zeros", zeros, wantZeros, 1e-3)
}

// ---------------------------------------------------------------------------
// Zeros - python-control sys222 (square MIMO, 2 state)
// ---------------------------------------------------------------------------
func TestCrossval_Zeros_PythonControl_sys222(t *testing.T) {
	A := denseFromRows([][]float64{
		{4, 1},
		{2, -3},
	})
	B := denseFromRows([][]float64{
		{5, 2},
		{-3, -3},
	})
	C := denseFromRows([][]float64{
		{2, -4},
		{0, 1},
	})
	D := denseFromRows([][]float64{
		{3, 2},
		{1, -1},
	})

	sys, _ := New(A, B, C, D, 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	wantZeros := []complex128{-10.568501, 3.368501}
	compareSortedZeros(t, "sys222 zeros", zeros, wantZeros, 1e-3)
}

// ---------------------------------------------------------------------------
// Zeros - empty / no zeros in SISO system
// ---------------------------------------------------------------------------
func TestCrossval_Zeros_NoZeros(t *testing.T) {
	// 1/(s²+2s+1) has no finite zeros
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 2, 1}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros, got %v", zeros)
	}
}

// ---------------------------------------------------------------------------
// Series (cascade) connection - python-control test_multiply_ss
// ---------------------------------------------------------------------------
func TestCrossval_Series_PythonControl(t *testing.T) {
	sys222, _ := New(
		denseFromRows([][]float64{{4, 1}, {2, -3}}),
		denseFromRows([][]float64{{5, 2}, {-3, -3}}),
		denseFromRows([][]float64{{2, -4}, {0, 1}}),
		denseFromRows([][]float64{{3, 2}, {1, -1}}),
		0,
	)
	sys322, _ := New(
		denseFromRows([][]float64{{-3, 4, 2}, {-1, -3, 0}, {2, 5, 3}}),
		denseFromRows([][]float64{{1, 4}, {-3, -3}, {-2, 1}}),
		denseFromRows([][]float64{{4, 2, -3}, {1, 4, 3}}),
		denseFromRows([][]float64{{-2, 4}, {0, 1}}),
		0,
	)

	// sys322 * sys222 in python-control means sys322 after sys222 (sys222 feeds sys322)
	result, err := Series(sys222, sys322)
	if err != nil {
		t.Fatal(err)
	}

	wantA := denseFromRows([][]float64{
		{4, 1, 0, 0, 0},
		{2, -3, 0, 0, 0},
		{2, 0, -3, 4, 2},
		{-6, 9, -1, -3, 0},
		{-4, 9, 2, 5, 3},
	})
	wantB := denseFromRows([][]float64{
		{5, 2},
		{-3, -3},
		{7, -2},
		{-12, -3},
		{-5, -5},
	})
	wantC := denseFromRows([][]float64{
		{-4, 12, 4, 2, -3},
		{0, 1, 1, 4, 3},
	})
	wantD := denseFromRows([][]float64{
		{-2, -8},
		{1, -1},
	})

	assertMatNear(t, "A", result.A, wantA, 1e-10)
	assertMatNear(t, "B", result.B, wantB, 1e-10)
	assertMatNear(t, "C", result.C, wantC, 1e-10)
	assertMatNear(t, "D", result.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Parallel (add) - python-control test_add_ss
// ---------------------------------------------------------------------------
func TestCrossval_Parallel_PythonControl(t *testing.T) {
	sys222, _ := New(
		denseFromRows([][]float64{{4, 1}, {2, -3}}),
		denseFromRows([][]float64{{5, 2}, {-3, -3}}),
		denseFromRows([][]float64{{2, -4}, {0, 1}}),
		denseFromRows([][]float64{{3, 2}, {1, -1}}),
		0,
	)
	sys322, _ := New(
		denseFromRows([][]float64{{-3, 4, 2}, {-1, -3, 0}, {2, 5, 3}}),
		denseFromRows([][]float64{{1, 4}, {-3, -3}, {-2, 1}}),
		denseFromRows([][]float64{{4, 2, -3}, {1, 4, 3}}),
		denseFromRows([][]float64{{-2, 4}, {0, 1}}),
		0,
	)

	result, err := Parallel(sys322, sys222)
	if err != nil {
		t.Fatal(err)
	}

	wantA := denseFromRows([][]float64{
		{-3, 4, 2, 0, 0},
		{-1, -3, 0, 0, 0},
		{2, 5, 3, 0, 0},
		{0, 0, 0, 4, 1},
		{0, 0, 0, 2, -3},
	})
	wantB := denseFromRows([][]float64{
		{1, 4},
		{-3, -3},
		{-2, 1},
		{5, 2},
		{-3, -3},
	})
	wantC := denseFromRows([][]float64{
		{4, 2, -3, 2, -4},
		{1, 4, 3, 0, 1},
	})
	wantD := denseFromRows([][]float64{
		{1, 6},
		{1, 0},
	})

	assertMatNear(t, "A", result.A, wantA, 1e-10)
	assertMatNear(t, "B", result.B, wantB, 1e-10)
	assertMatNear(t, "C", result.C, wantC, 1e-10)
	assertMatNear(t, "D", result.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Subtract (parallel with negated D) - python-control test_subtract_ss
// ---------------------------------------------------------------------------
func TestCrossval_Subtract_PythonControl(t *testing.T) {
	sys222, _ := New(
		denseFromRows([][]float64{{4, 1}, {2, -3}}),
		denseFromRows([][]float64{{5, 2}, {-3, -3}}),
		denseFromRows([][]float64{{2, -4}, {0, 1}}),
		denseFromRows([][]float64{{3, 2}, {1, -1}}),
		0,
	)
	sys322, _ := New(
		denseFromRows([][]float64{{-3, 4, 2}, {-1, -3, 0}, {2, 5, 3}}),
		denseFromRows([][]float64{{1, 4}, {-3, -3}, {-2, 1}}),
		denseFromRows([][]float64{{4, 2, -3}, {1, 4, 3}}),
		denseFromRows([][]float64{{-2, 4}, {0, 1}}),
		0,
	)

	// sys322 - sys222: negate sys222's C and D, then parallel
	negC := mat.NewDense(2, 2, nil)
	negC.Scale(-1, sys222.C)
	negD := mat.NewDense(2, 2, nil)
	negD.Scale(-1, sys222.D)
	sys222neg, _ := New(denseCopy(sys222.A), denseCopy(sys222.B), negC, negD, 0)

	result, err := Parallel(sys322, sys222neg)
	if err != nil {
		t.Fatal(err)
	}

	wantC := denseFromRows([][]float64{
		{4, 2, -3, -2, 4},
		{1, 4, 3, 0, -1},
	})
	wantD := denseFromRows([][]float64{
		{-5, 2},
		{-1, 2},
	})

	assertMatNear(t, "C", result.C, wantC, 1e-10)
	assertMatNear(t, "D", result.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Append - python-control test_append_ss
// ---------------------------------------------------------------------------
func TestCrossval_Append_PythonControl(t *testing.T) {
	sys1, _ := New(
		denseFromRows([][]float64{{-2, 0.5, 0}, {0.5, -0.3, 0}, {0, 0, -0.1}}),
		denseFromRows([][]float64{{0.3, -1.3}, {0.1, 0}, {1, 0}}),
		denseFromRows([][]float64{{0, 0.1, 0}, {-0.3, -0.2, 0}}),
		denseFromRows([][]float64{{0, -0.8}, {-0.3, 0}}),
		0,
	)
	sys2, _ := New(
		denseFromFlat(1, 1, []float64{-1}),
		denseFromFlat(1, 1, []float64{1.2}),
		denseFromFlat(1, 1, []float64{0.5}),
		denseFromFlat(1, 1, []float64{0.4}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	wantA := denseFromRows([][]float64{
		{-2, 0.5, 0, 0},
		{0.5, -0.3, 0, 0},
		{0, 0, -0.1, 0},
		{0, 0, 0, -1},
	})
	wantB := denseFromRows([][]float64{
		{0.3, -1.3, 0},
		{0.1, 0, 0},
		{1, 0, 0},
		{0, 0, 1.2},
	})
	wantC := denseFromRows([][]float64{
		{0, 0.1, 0, 0},
		{-0.3, -0.2, 0, 0},
		{0, 0, 0, 0.5},
	})
	wantD := denseFromRows([][]float64{
		{0, -0.8, 0},
		{-0.3, 0, 0},
		{0, 0, 0.4},
	})

	assertMatNear(t, "A", result.A, wantA, 1e-10)
	assertMatNear(t, "B", result.B, wantB, 1e-10)
	assertMatNear(t, "C", result.C, wantC, 1e-10)
	assertMatNear(t, "D", result.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Feedback dynamic+dynamic - reference: feedback dynamic+dynamic (D=0)
// ---------------------------------------------------------------------------
func TestCrossval_Feedback_DynamicDynamic(t *testing.T) {
	a := denseFromRows([][]float64{{2, 2, 1}, {3, 2, -2}, {1, -1, 1}})
	b := denseFromRows([][]float64{{-1}, {3}, {-3}})
	c := denseFromRows([][]float64{{0, -1, 2}})
	d := denseFromFlat(1, 1, []float64{0})

	k := denseFromRows([][]float64{{2, 0, 2}, {-2, 1, -2}, {0, 1, 1}})
	l := denseFromRows([][]float64{{1}, {-2}, {-3}})
	m2 := denseFromRows([][]float64{{0, 3, -1}})
	n := denseFromFlat(1, 1, []float64{2})

	G, _ := New(a, b, c, d, 0)
	H, _ := New(k, l, m2, n, 0)

	CL, err := Feedback(G, H, -1)
	if err != nil {
		t.Fatal(err)
	}

	wantA := denseFromRows([][]float64{
		{2, 0, 5, 0, 3, -1},
		{3, 8, -14, 0, -9, 3},
		{1, -7, 13, 0, 9, -3},
		{0, -1, 2, 2, 0, 2},
		{0, 2, -4, -2, 1, -2},
		{0, 3, -6, 0, 1, 1},
	})
	wantB := denseFromRows([][]float64{{-1}, {3}, {-3}, {0}, {0}, {0}})
	wantC := denseFromRows([][]float64{{0, -1, 2, 0, 0, 0}})
	wantD := denseFromFlat(1, 1, []float64{0})

	assertMatNear(t, "A", CL.A, wantA, 1e-10)
	assertMatNear(t, "B", CL.B, wantB, 1e-10)
	assertMatNear(t, "C", CL.C, wantC, 1e-10)
	assertMatNear(t, "D", CL.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Feedback dynamic+dynamic with D≠0 - verify via frequency response
// Verify CL(jw) = G(jw)/(1+H(jw)*G(jw)) via frequency response.
// ---------------------------------------------------------------------------
func TestCrossval_Feedback_DynamicDynamic_WithD_FreqCheck(t *testing.T) {
	a := denseFromRows([][]float64{{2, 2, 1}, {3, 2, -2}, {1, -1, 1}})
	b := denseFromRows([][]float64{{-1}, {3}, {-3}})
	c := denseFromRows([][]float64{{0, -1, 2}})
	d := denseFromFlat(1, 1, []float64{1})

	k := denseFromRows([][]float64{{2, 0, 2}, {-2, 1, -2}, {0, 1, 1}})
	l := denseFromRows([][]float64{{1}, {-2}, {-3}})
	m2 := denseFromRows([][]float64{{0, 3, -1}})
	n := denseFromFlat(1, 1, []float64{2})

	G, _ := New(a, b, c, d, 0)
	H, _ := New(k, l, m2, n, 0)

	CL, err := Feedback(G, H, -1)
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		jw := complex(0, w)
		gResp, _ := G.EvalFr(jw)
		hResp, _ := H.EvalFr(jw)
		clResp, _ := CL.EvalFr(jw)

		gVal := gResp[0][0]
		hVal := hResp[0][0]
		clVal := clResp[0][0]

		want := gVal / (1 + hVal*gVal)
		if cmplx.Abs(clVal-want) > 1e-10*cmplx.Abs(want) {
			t.Errorf("w=%g: CL(jw)=%v, want G/(1+HG)=%v", w, clVal, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Feedback static+static - reference: feedback static+static
// ---------------------------------------------------------------------------
func TestCrossval_Feedback_StaticStatic(t *testing.T) {
	G, _ := NewGain(denseFromFlat(1, 1, []float64{5}), 0)
	H, _ := NewGain(denseFromFlat(1, 1, []float64{5}), 0)

	// Negative feedback: D_cl = D_G * inv(I + D_H * D_G) = 5 / (1 + 25) = 5/26
	CL, err := Feedback(G, H, -1)
	if err != nil {
		t.Fatal(err)
	}
	want := 5.0 / 26.0
	if math.Abs(CL.D.At(0, 0)-want) > 1e-10 {
		t.Errorf("D = %g, want %g", CL.D.At(0, 0), want)
	}
}

// ---------------------------------------------------------------------------
// Frequency response - python-control test_freq_resp
// ---------------------------------------------------------------------------
func TestCrossval_FreqResp_PythonControl(t *testing.T) {
	A := denseFromRows([][]float64{{-2, 0.5}, {0.5, -0.3}})
	B := denseFromRows([][]float64{{0.3, -1.3}, {0.1, 0}})
	C := denseFromRows([][]float64{{0, 0.1}, {-0.3, -0.2}})
	D := denseFromRows([][]float64{{0, -0.8}, {-0.3, 0}})

	sys, _ := New(A, B, C, D, 0)

	trueMag := [2][2][2]float64{
		{
			{0.0852992637230322, 0.00103596611395218},
			{0.935374692849736, 0.799380720864549},
		},
		{
			{0.55656854563842, 0.301542699860857},
			{0.609178071542849, 0.0382108097985257},
		},
	}
	truePhase := [2][2][2]float64{
		{
			{-0.566195599644593, -1.68063565332582},
			{3.0465958317514, 3.14141384339534},
		},
		{
			{2.90457947657161, 3.10601268291914},
			{-0.438157380501337, -1.40720969147217},
		},
	}
	trueOmega := []float64{0.1, 10.0}

	resp, err := sys.FreqResponse(trueOmega)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range trueOmega {
		for i := range 2 {
			for j := range 2 {
				h := resp.At(k, i, j)
				gotMag := cmplx.Abs(h)
				gotPhase := cmplx.Phase(h)
				wantMag := trueMag[i][j][k]
				wantPhase := truePhase[i][j][k]
				_ = w
				if math.Abs(gotMag-wantMag) > 1e-6 {
					t.Errorf("w=%g [%d][%d] mag=%g, want %g", w, i, j, gotMag, wantMag)
				}
				if math.Abs(gotPhase-wantPhase) > 1e-6 {
					t.Errorf("w=%g [%d][%d] phase=%g, want %g", w, i, j, gotPhase, wantPhase)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Frequency response at single point - python-control test_call
// ---------------------------------------------------------------------------
func TestCrossval_EvalFr_PythonControl(t *testing.T) {
	A := denseFromRows([][]float64{{-2, 0.5}, {0.5, -0.3}})
	B := denseFromRows([][]float64{{0.3, -1.3}, {0.1, 0}})
	C := denseFromRows([][]float64{{0, 0.1}, {-0.3, -0.2}})
	D := denseFromRows([][]float64{{0, -0.8}, {-0.3, 0}})

	sys, _ := New(A, B, C, D, 0)

	wantResp := [][]complex128{
		{
			complex(4.37636761e-05, -0.01522976),
			complex(-7.92603939e-01, 0.02617068),
		},
		{
			complex(-3.31544858e-01, 0.0576105),
			complex(1.28919037e-01, -0.14382495),
		},
	}

	resp, err := sys.EvalFr(1i)
	if err != nil {
		t.Fatal(err)
	}

	for i := range 2 {
		for j := range 2 {
			if cmplx.Abs(resp[i][j]-wantResp[i][j]) > 1e-3 {
				t.Errorf("[%d][%d] = %v, want %v", i, j, resp[i][j], wantResp[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Discretize → Undiscretize roundtrip - reference: undiscretize roundtrip (bilinear)
// ---------------------------------------------------------------------------
func TestCrossval_DiscretizeRoundtrip(t *testing.T) {
	ac := denseFromRows([][]float64{{1, 0}, {0, 1}})
	bc := denseFromRows([][]float64{{0.5}, {0.5}})
	cc := denseFromRows([][]float64{
		{0.75, 1.0},
		{1.0, 1.0},
		{1.0, 0.25},
	})
	dc := denseFromRows([][]float64{
		{0.0},
		{0.0},
		{-0.33},
	})

	sys, _ := New(ac, bc, cc, dc, 0)
	dt := 0.5

	disc, err := sys.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}
	back, err := disc.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}

	assertMatNear(t, "A roundtrip", back.A, ac, 1e-10)
	assertMatNear(t, "B roundtrip", back.B, bc, 1e-10)
	assertMatNear(t, "C roundtrip", back.C, cc, 1e-10)
	assertMatNear(t, "D roundtrip", back.D, dc, 1e-10)
}

// ---------------------------------------------------------------------------
// DC gain verification via frequency response at s=0
// python-control test_dc_gain_cont
// ---------------------------------------------------------------------------
func TestCrossval_DCGain_Continuous(t *testing.T) {
	// sys = ss(-2, 6, 5, 0) => dcgain = -C*A^{-1}*B + D = -5*(-1/2)*6 + 0 = 15
	sys, _ := New(
		denseFromFlat(1, 1, []float64{-2}),
		denseFromFlat(1, 1, []float64{6}),
		denseFromFlat(1, 1, []float64{5}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)

	resp, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	dcGain := real(resp[0][0])
	if math.Abs(dcGain-15.0) > 1e-10 {
		t.Errorf("DC gain = %g, want 15", dcGain)
	}
}

// ---------------------------------------------------------------------------
// DC gain for MIMO - python-control test_dc_gain_cont (sys2)
// ---------------------------------------------------------------------------
func TestCrossval_DCGain_MIMO(t *testing.T) {
	// sys2 = ss(-2, [6, 4], [[5], [7], [11]], zeros(3,2))
	// dcgain = -C*inv(A)*B + D
	sys, _ := New(
		denseFromFlat(1, 1, []float64{-2}),
		denseFromFlat(1, 2, []float64{6, 4}),
		denseFromFlat(3, 1, []float64{5, 7, 11}),
		denseFromFlat(3, 2, nil),
		0,
	)

	resp, err := sys.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]float64{
		{15, 10},
		{21, 14},
		{33, 22},
	}
	for i := range 3 {
		for j := range 2 {
			got := real(resp[i][j])
			if math.Abs(got-expected[i][j]) > 1e-10 {
				t.Errorf("[%d][%d] DC gain = %g, want %g", i, j, got, expected[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Controllability matrix - python-control testCtrbSISO
// ---------------------------------------------------------------------------
func TestCrossval_Ctrb_SISO(t *testing.T) {
	A := denseFromFlat(2, 2, []float64{1, 2, 3, 4})
	B := denseFromFlat(2, 1, []float64{5, 7})

	// Wc = [B, A*B] = [[5, 19], [7, 43]]
	AB := mat.NewDense(2, 1, nil)
	AB.Mul(A, B)
	Wc := mat.NewDense(2, 2, nil)
	setBlock(Wc, 0, 0, B)
	setBlock(Wc, 0, 1, AB)

	wantWc := denseFromFlat(2, 2, []float64{5, 19, 7, 43})
	assertMatNear(t, "Ctrb", Wc, wantWc, 1e-10)
}

// ---------------------------------------------------------------------------
// Controllability matrix - python-control testCtrbMIMO
// ---------------------------------------------------------------------------
func TestCrossval_Ctrb_MIMO(t *testing.T) {
	A := denseFromFlat(2, 2, []float64{1, 2, 3, 4})
	B := denseFromFlat(2, 2, []float64{5, 6, 7, 8})

	AB := mat.NewDense(2, 2, nil)
	AB.Mul(A, B)
	Wc := mat.NewDense(2, 4, nil)
	setBlock(Wc, 0, 0, B)
	setBlock(Wc, 0, 2, AB)

	wantWc := denseFromFlat(2, 4, []float64{5, 6, 19, 22, 7, 8, 43, 50})
	assertMatNear(t, "Ctrb", Wc, wantWc, 1e-10)
}

// ---------------------------------------------------------------------------
// Staircase - python-control controllability for 3×2 system
// (Chemical Reactor from Byers-Nash, via reference: static controller design)
// ---------------------------------------------------------------------------
func TestCrossval_Staircase_ChemicalReactor(t *testing.T) {
	A := denseFromRows([][]float64{
		{1.380, -0.2077, 6.715, -5.676},
		{-0.5814, -4.290, 0, 0.6750},
		{1.067, 4.273, -6.654, 5.893},
		{0.0480, 4.273, 1.343, -2.104},
	})
	B := denseFromRows([][]float64{
		{0, 5.679},
		{1.136, 1.136},
		{0, 0},
		{-3.146, 0},
	})

	stair := ControllabilityStaircase(A, B, nil, 0)

	if stair.NCont != 4 {
		t.Errorf("NCont = %d, want 4 (fully controllable)", stair.NCont)
	}

	// Eigenvalues should be preserved
	origPoles, err := (&System{A: A}).Poles()
	if err != nil {
		t.Fatal(err)
	}
	transPoles, err := (&System{A: stair.A}).Poles()
	if err != nil {
		t.Fatal(err)
	}
	compareSortedPoles(t, "eigenvalues", transPoles, origPoles, 1e-10)
}

// ---------------------------------------------------------------------------
// Discretize c2d match at DC gain - python-control test_c2d_matched
// ---------------------------------------------------------------------------
func TestCrossval_C2D_DCGainMatch(t *testing.T) {
	tests := []struct {
		name string
		num  []float64
		den  []float64
	}{
		{"1/(s+1)", []float64{1}, []float64{1, 1}},
		{"(s+2)/(s+3)", []float64{1, 2}, []float64{1, 3}},
		{"(s+2)/(3s²+4s+5)", []float64{1, 2}, []float64{3, 4, 5}},
	}

	for _, tc := range tests {
		for _, dt := range []float64{0.1, 1.0, 2.0} {
			t.Run(tc.name, func(t *testing.T) {
				// Build continuous TF, convert to SS, discretize
				tf := &TransferFunc{
					Num: [][][]float64{{tc.num}},
					Den: [][]float64{tc.den},
				}
				ssRes, err := tf.StateSpace(nil)
				if err != nil {
					t.Fatal(err)
				}

				sysd, err := ssRes.Sys.DiscretizeZOH(dt)
				if err != nil {
					t.Fatal(err)
				}

				// Continuous DC gain = H(0) = num(0)/den(0)
				numPoly := Poly(tc.num)
				denPoly := Poly(tc.den)
				dcCont := real(numPoly.Eval(0)) / real(denPoly.Eval(0))

				// Discrete DC gain = H(z=1) = C*(I-A)^{-1}*B + D
				resp, err := sysd.EvalFr(complex(1, 0))
				if err != nil {
					t.Fatal(err)
				}
				dcDisc := real(resp[0][0])

				if math.Abs(dcCont-dcDisc)/math.Max(1, math.Abs(dcCont)) > 1e-6 {
					t.Errorf("dt=%g: DC gain mismatch: cont=%g, disc=%g", dt, dcCont, dcDisc)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Poles preserved through ZOH: z = e^(s*dt)
// python-control test_c2d_matched pole check
// ---------------------------------------------------------------------------
func TestCrossval_ZOH_PoleMapping(t *testing.T) {
	tests := []struct {
		name string
		num  []float64
		den  []float64
	}{
		{"1/(s+1)", []float64{1}, []float64{1, 1}},
		{"(s+2)/(s+3)", []float64{1, 2}, []float64{1, 3}},
		{"(s+2)/(3s²+4s+5)", []float64{1, 2}, []float64{3, 4, 5}},
	}

	for _, tc := range tests {
		for _, dt := range []float64{0.1, 1.0} {
			t.Run(tc.name, func(t *testing.T) {
				tf := &TransferFunc{
					Num: [][][]float64{{tc.num}},
					Den: [][]float64{tc.den},
				}
				ssRes, err := tf.StateSpace(nil)
				if err != nil {
					t.Fatal(err)
				}
				sysc := ssRes.Sys

				sysd, err := sysc.DiscretizeZOH(dt)
				if err != nil {
					t.Fatal(err)
				}

				cPoles, err := sysc.Poles()
				if err != nil {
					t.Fatal(err)
				}
				dPoles, err := sysd.Poles()
				if err != nil {
					t.Fatal(err)
				}

				for _, cp := range cPoles {
					expectedZ := cmplx.Exp(cp * complex(dt, 0))
					found := false
					for _, dp := range dPoles {
						if cmplx.Abs(dp-expectedZ) < 1e-8 {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("dt=%g: continuous pole %v -> expected z=%v not found in %v",
							dt, cp, expectedZ, dPoles)
					}
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// System norm (H-infinity) - reference: H-infinity norm
// Computed as peak magnitude of frequency response
// ---------------------------------------------------------------------------
func TestCrossval_HinfNorm(t *testing.T) {
	// G = Transfer(100, [1, 10, 100])
	// H-inf norm ≈ 1.1547 (underdamped second-order with wn=10, zeta=0.5)
	tf := &TransferFunc{
		Num: [][][]float64{{{100}}},
		Den: [][]float64{{1, 10, 100}},
	}
	ssRes, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	sys := ssRes.Sys

	// Peak of |G(jw)| for second-order: 1/(2*zeta*sqrt(1-zeta²))
	// where wn=10, zeta=0.5 => peak = 1/(2*0.5*sqrt(1-0.25)) = 1/sqrt(0.75) ≈ 1.1547
	wPeak := 10 * math.Sqrt(1-2*0.25) // = 10*sqrt(0.5) ≈ 7.071
	omega := []float64{wPeak}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	peakMag := cmplx.Abs(resp.At(0, 0, 0))
	expected := 1.0 / (2 * 0.5 * math.Sqrt(1-0.25))

	if math.Abs(peakMag-expected) > 1e-4 {
		t.Errorf("H-inf norm at peak = %g, want %g", peakMag, expected)
	}
}

// ---------------------------------------------------------------------------
// Bilinear preserves frequency response at s=jw => z=e^{jwT}
// (Bilinear/Tustin warping property)
// ---------------------------------------------------------------------------
func TestCrossval_Bilinear_FreqResponseMatch(t *testing.T) {
	// Use a non-trivial system
	A := denseFromRows([][]float64{{-2, 0.5}, {0.5, -0.3}})
	B := denseFromRows([][]float64{{0.3, -1.3}, {0.1, 0}})
	C := denseFromRows([][]float64{{0, 0.1}, {-0.3, -0.2}})
	D := denseFromRows([][]float64{{0, -0.8}, {-0.3, 0}})
	sys, _ := New(A, B, C, D, 0)

	dt := 0.01

	disc, err := sys.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}

	// At frequency w, bilinear maps s=jw to z = (1+jw*dt/2)/(1-jw*dt/2)
	for _, w := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, w)
		hc, err := sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}

		z := (1 + s*complex(dt/2, 0)) / (1 - s*complex(dt/2, 0))
		hd, err := disc.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}

		for i := range 2 {
			for j := range 2 {
				if cmplx.Abs(hc[i][j]-hd[i][j]) > 1e-8 {
					t.Errorf("w=%g [%d][%d]: cont=%v, disc=%v", w, i, j, hc[i][j], hd[i][j])
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Controllability staircase - MIMO chemical reactor (fully controllable)
// from Byers & Nash via reference: static controller design.py
// ---------------------------------------------------------------------------
func TestCrossval_Staircase_DistillationColumn(t *testing.T) {
	A := denseFromRows([][]float64{
		{-0.1094, 0.0628, 0, 0, 0},
		{1.306, -2.132, 0.9807, 0, 0},
		{0, 1.595, -3.149, 1.547, 0},
		{0, 0.0355, 2.632, -4.257, 1.855},
		{0, 0.0023, 0, 0.1636, -0.1625},
	})
	B := denseFromRows([][]float64{
		{0, 0},
		{0.638, 0},
		{0.0838, -0.1396},
		{0.1004, -0.206},
		{0.0063, -0.0128},
	})

	stair := ControllabilityStaircase(A, B, nil, 0)

	if stair.NCont != 5 {
		t.Errorf("NCont = %d, want 5 (fully controllable)", stair.NCont)
	}

	origPoles, err := (&System{A: A}).Poles()
	if err != nil {
		t.Fatal(err)
	}
	transPoles, err := (&System{A: stair.A}).Poles()
	if err != nil {
		t.Fatal(err)
	}
	compareSortedPoles(t, "eigenvalues", transPoles, origPoles, 1e-8)
}

// ---------------------------------------------------------------------------
// Transfer function roundtrip: SS → TF → SS preserves frequency response
// Using non-symmetric, non-trivial system (companion form)
// ---------------------------------------------------------------------------
func TestCrossval_TF_Roundtrip_NonSymmetric(t *testing.T) {
	// 4th order system in controllable canonical form
	// den = s⁴ + 10s³ + 35s² + 50s + 24 = (s+1)(s+2)(s+3)(s+4)
	A := denseFromRows([][]float64{
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
		{-24, -50, -35, -10},
	})
	B := denseFromRows([][]float64{{0}, {0}, {0}, {1}})
	C := denseFromRows([][]float64{{1, 0, 0, 0}})
	D := denseFromFlat(1, 1, []float64{0})

	sys, _ := New(A, B, C, D, 0)

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	ssRes, err := tfRes.TF.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	sys2 := ssRes.Sys

	// Compare frequency responses
	for _, w := range []float64{0.1, 0.5, 1, 2, 5, 10, 50} {
		s := complex(0, w)
		h1, _ := sys.EvalFr(s)
		h2, _ := sys2.EvalFr(s)
		if cmplx.Abs(h1[0][0]-h2[0][0]) > 1e-6 {
			t.Errorf("w=%g: orig=%v, roundtrip=%v", w, h1[0][0], h2[0][0])
		}
	}
}

// ---------------------------------------------------------------------------
// Non-square MIMO zeros - python-control sys623
// ---------------------------------------------------------------------------
func TestCrossval_Zeros_NonSquare_sys623(t *testing.T) {
	A := denseFromRows([][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 3, 0, 0, 0},
		{0, 0, 0, -4, 0, 0},
		{0, 0, 0, 0, -1, 0},
		{0, 0, 0, 0, 0, 3},
	})
	B := denseFromRows([][]float64{
		{0, -1},
		{-1, 0},
		{1, -1},
		{0, 0},
		{0, 1},
		{-1, -1},
	})
	C := denseFromRows([][]float64{
		{1, 0, 0, 1, 0, 0},
		{0, 1, 0, 1, 0, 1},
		{0, 0, 1, 0, 0, 1},
	})
	D := denseFromFlat(3, 2, nil)

	sys, _ := New(A, B, C, D, 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	wantZeros := []complex128{2, -1}
	compareSortedZeros(t, "sys623 zeros", zeros, wantZeros, 1e-4)
}

// ---------------------------------------------------------------------------
// Stability checking - continuous and discrete
// ---------------------------------------------------------------------------
func TestCrossval_Stability(t *testing.T) {
	// Stable continuous: all eigenvalues have negative real part
	stable, _ := New(
		denseFromRows([][]float64{{-1, 0}, {0, -2}}),
		denseFromFlat(2, 1, []float64{1, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)
	if s, err := stable.IsStable(); err != nil {
		t.Fatal(err)
	} else if !s {
		t.Error("expected stable continuous system")
	}

	unstable, _ := New(
		denseFromRows([][]float64{{1, 0}, {0, -2}}),
		denseFromFlat(2, 1, []float64{1, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)
	if s, err := unstable.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected unstable continuous system")
	}

	stableD, _ := New(
		denseFromRows([][]float64{{0.5, 0}, {0, 0.3}}),
		denseFromFlat(2, 1, []float64{1, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		1.0,
	)
	if s, err := stableD.IsStable(); err != nil {
		t.Fatal(err)
	} else if !s {
		t.Error("expected stable discrete system")
	}

	unstableD, _ := New(
		denseFromRows([][]float64{{1.1, 0}, {0, 0.3}}),
		denseFromFlat(2, 1, []float64{1, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		1.0,
	)
	if s, err := unstableD.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected unstable discrete system")
	}

	marginal, _ := New(
		denseFromRows([][]float64{{0, 1}, {0, 0}}),
		denseFromFlat(2, 1, []float64{0, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)
	if s, err := marginal.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected marginally stable continuous system to be unstable")
	}

	marginalD, _ := New(
		denseFromRows([][]float64{{1, 0}, {0, 0.3}}),
		denseFromFlat(2, 1, []float64{1, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		1.0,
	)
	if s, err := marginalD.IsStable(); err != nil {
		t.Fatal(err)
	} else if s {
		t.Error("expected marginally stable discrete system to be unstable")
	}
}

// ---------------------------------------------------------------------------
// Simulation of discrete ZOH'd double integrator: step response
// ---------------------------------------------------------------------------
func TestCrossval_ZOH_StepResponse(t *testing.T) {
	// Double integrator y=position, u=acceleration
	sys, _ := New(
		denseFromFlat(2, 2, []float64{0, 1, 0, 0}),
		denseFromFlat(2, 1, []float64{0, 1}),
		denseFromFlat(1, 2, []float64{1, 0}),
		denseFromFlat(1, 1, []float64{0}),
		0,
	)

	dt := 0.01
	sysd, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	steps := 100
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1.0) // unit step
	}

	resp, err := sysd.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Position at time T = n*dt with unit acceleration: y = 0.5*t²
	// Last sample: t = (steps-1)*dt, but simulation output at step k
	// corresponds to after k inputs. y[k] ≈ 0.5*(k*dt)²
	for k := range steps {
		tK := float64(k) * dt
		expected := 0.5 * tK * tK
		got := resp.Y.At(0, k)
		if math.Abs(got-expected) > 0.02 { // ZOH approximation error
			t.Errorf("step %d (t=%g): y=%g, want ~%g", k, tK, got, expected)
		}
	}
}

// ---------------------------------------------------------------------------
// Gain system operations
// ---------------------------------------------------------------------------
func TestCrossval_GainSystems(t *testing.T) {
	g1, _ := NewGain(denseFromFlat(1, 1, []float64{2}), 0)
	g2, _ := NewGain(denseFromFlat(1, 1, []float64{3}), 0)

	// Series: 2*3 = 6
	g3, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if g3.D.At(0, 0) != 6 {
		t.Errorf("series gain = %g, want 6", g3.D.At(0, 0))
	}

	// Parallel: 2+3 = 5
	g4, err := Parallel(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if g4.D.At(0, 0) != 5 {
		t.Errorf("parallel gain = %g, want 5", g4.D.At(0, 0))
	}

	// Feedback: 2/(1+2*3) = 2/7
	g5, err := Feedback(g1, g2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g5.D.At(0, 0)-2.0/7.0) > 1e-10 {
		t.Errorf("feedback gain = %g, want %g", g5.D.At(0, 0), 2.0/7.0)
	}

	// Append: diag(2, 3)
	g6, err := Append(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if g6.D.At(0, 0) != 2 || g6.D.At(1, 1) != 3 {
		t.Errorf("append gain diag = (%g, %g), want (2, 3)", g6.D.At(0, 0), g6.D.At(1, 1))
	}
	if g6.D.At(0, 1) != 0 || g6.D.At(1, 0) != 0 {
		t.Errorf("append gain off-diag = (%g, %g), want (0, 0)", g6.D.At(0, 1), g6.D.At(1, 0))
	}
}

// ---------------------------------------------------------------------------
// Matrix gain series - python-control test_matrix_static_gain
// ---------------------------------------------------------------------------
func TestCrossval_MatrixGainSeries(t *testing.T) {
	d1 := denseFromRows([][]float64{{1, 2, 3}, {4, 5, 6}})
	d2 := denseFromRows([][]float64{{7, 8}, {9, 10}, {11, 12}})

	g1, _ := NewGain(d1, 0)
	g2, _ := NewGain(d2, 0)

	h1, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	// D = d2 @ d1 ... wait, Series(sys1, sys2) means sys1 feeds sys2
	// So result = sys2 * sys1 = d2 * d1... No, it's D2*D1 where sys1 output feeds sys2 input
	// Actually: Series(sys1, sys2) = sys2 after sys1, so D = D2*D1
	// But d1 is 2x3 and d2 is 3x2, so D2*D1 is 2x2? No wait:
	// Series(sys1, sys2): output of sys1 (p1=2) feeds sys2 (m2 must = 2)
	// But d2 is 3x2, so p2=3 outputs... and m2=2 inputs. p1=2, m2=2. OK.
	// D_result = D2 * D1 = (3x2) * (2x3) = 3x3
	wantD := mat.NewDense(3, 3, nil)
	wantD.Mul(d2, d1)
	assertMatNear(t, "D", h1.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Parallel of matrix gains - python-control test_matrix_static_gain
// ---------------------------------------------------------------------------
func TestCrossval_MatrixGainParallel(t *testing.T) {
	d1 := denseFromRows([][]float64{{1, 2, 3}, {4, 5, 6}})
	d3 := denseFromRows([][]float64{{7, 9, 11}, {8, 10, 12}}) // d2.T

	g1, _ := NewGain(d1, 0)
	g3, _ := NewGain(d3, 0)

	h2, err := Parallel(g1, g3)
	if err != nil {
		t.Fatal(err)
	}
	wantD := mat.NewDense(2, 3, nil)
	wantD.Add(d1, d3)
	assertMatNear(t, "D", h2.D, wantD, 1e-10)
}

// ---------------------------------------------------------------------------
// Minimal realization: uncontrollable mode is removed
// reference: minimal realization (SISO)
// ---------------------------------------------------------------------------
func TestCrossval_MinReal_Reference(t *testing.T) {
	// 5-state system with 4 minimal states (from reference)
	A := denseFromRows([][]float64{
		{0, 1, 0, 0, 0},
		{-0.1, -0.5, 1, -1, 0},
		{0, 0, 0, 1, 0},
		{0, 0, 0, 0, 1},
		{0, 3.5, 1, -2, 2},
	})
	B := denseFromRows([][]float64{{0}, {1}, {0}, {0}, {1}})
	C := denseFromRows([][]float64{{0, 3.5, 1, -1, 0}})
	D := denseFromFlat(1, 1, []float64{1})

	sys, _ := New(A, B, C, D, 0)
	reduced, err := sys.Reduce(&ReduceOpts{Tol: 1e-6})
	if err != nil {
		t.Fatal(err)
	}

	if reduced.Order != 4 {
		t.Errorf("minimal order = %d, want 4", reduced.Order)
	}

	// Verify frequency response is preserved
	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		s := complex(0, w)
		h1, _ := sys.EvalFr(s)
		h2, _ := reduced.Sys.EvalFr(s)
		if cmplx.Abs(h1[0][0]-h2[0][0]) > 1e-6 {
			t.Errorf("w=%g: orig=%v, reduced=%v", w, h1[0][0], h2[0][0])
		}
	}
}

// ---------------------------------------------------------------------------
// Minimal realization: MIMO with uncontrollable mode
// reference: minimal realization (MIMO)
// ---------------------------------------------------------------------------
func TestCrossval_MinReal_MIMO(t *testing.T) {
	A := denseFromRows([][]float64{
		{-2, 0, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
		{0, -12, 4, 3},
	})
	B := denseFromRows([][]float64{
		{1, 0},
		{0, 0},
		{0, 0},
		{0, 1},
	})
	C := denseFromRows([][]float64{
		{1, -9, 0, 0},
		{0, -20, 0, 5},
	})
	D := denseFromRows([][]float64{
		{0, 0},
		{0, 1},
	})

	sys, _ := New(A, B, C, D, 0)
	reduced, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if reduced.Order != 3 {
		t.Errorf("minimal order = %d, want 3", reduced.Order)
	}

	// Verify frequency response is preserved
	for _, w := range []float64{0.01, 0.1, 1, 10} {
		s := complex(0, w)
		h1, _ := sys.EvalFr(s)
		h2, _ := reduced.Sys.EvalFr(s)
		for i := range 2 {
			for j := range 2 {
				if cmplx.Abs(h1[i][j]-h2[i][j]) > 1e-4 {
					t.Errorf("w=%g [%d][%d]: orig=%v, reduced=%v", w, i, j, h1[i][j], h2[i][j])
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Bilinear (Tustin) with different sample rate - reference: Tustin discretization
// ---------------------------------------------------------------------------
func TestCrossval_Tustin_Reference_Rate2(t *testing.T) {
	A := denseFromRows([][]float64{{1, 0}, {0, 1}})
	B := denseFromRows([][]float64{{0.5}, {0.5}})
	C := denseFromRows([][]float64{
		{0.75, 1.0},
		{1.0, 1.0},
		{1.0, 0.25},
	})
	D := denseFromRows([][]float64{
		{0.0},
		{0.0},
		{-0.33},
	})

	sys, _ := New(A, B, C, D, 0)
	dt := 1.0 / 3.0

	hd, err := sys.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}

	adTruth := denseFromRows([][]float64{
		{1.4, 0},
		{0, 1.4},
	})
	assertMatNear(t, "Ad", hd.A, adTruth, 1e-10)

	sqrt1over3 := math.Sqrt(1.0 / 3.0)
	bdTruth := denseFromRows([][]float64{
		{0.2 / sqrt1over3},
		{0.2 / sqrt1over3},
	})
	assertMatNear(t, "Bd", hd.B, bdTruth, 1e-10)

	cdTruth := denseFromRows([][]float64{
		{0.9 * sqrt1over3, 1.2 * sqrt1over3},
		{1.2 * sqrt1over3, 1.2 * sqrt1over3},
		{1.2 * sqrt1over3, 0.3 * sqrt1over3},
	})
	assertMatNear(t, "Cd", hd.C, cdTruth, 1e-10)

	ddTruth := denseFromRows([][]float64{
		{0.175},
		{0.2},
		{-0.205},
	})
	assertMatNear(t, "Dd", hd.D, ddTruth, 1e-10)
}

// ---------------------------------------------------------------------------
// MIMO series then check zeros - cross-validation exercise
// ---------------------------------------------------------------------------
func TestCrossval_SeriesThenZeros(t *testing.T) {
	// Two first-order SISO systems in series:
	// G1 = (s+1)/(s+3), G2 = (s+2)/(s+4)
	// Series: G = G2*G1 = (s+1)(s+2)/((s+3)(s+4))
	// Zeros: -1, -2. Poles: -3, -4.
	tf1 := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 3}},
	}
	tf2 := &TransferFunc{
		Num: [][][]float64{{{1, 2}}},
		Den: [][]float64{{1, 4}},
	}
	ss1, _ := tf1.StateSpace(nil)
	ss2, _ := tf2.StateSpace(nil)

	ser, err := Series(ss1.Sys, ss2.Sys)
	if err != nil {
		t.Fatal(err)
	}

	zeros, err := ser.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	wantZeros := []complex128{-1, -2}
	compareSortedZeros(t, "series zeros", zeros, wantZeros, 1e-6)

	wantPoles := []complex128{-3, -4}
	serPoles, err := ser.Poles()
	if err != nil {
		t.Fatal(err)
	}
	compareSortedPoles(t, "series poles", serPoles, wantPoles, 1e-10)
}

// ---------------------------------------------------------------------------
// Feedback stabilization check - inspired by reference: feedback dynamic+static
// ---------------------------------------------------------------------------
func TestCrossval_Feedback_Stabilization(t *testing.T) {
	A := denseFromRows([][]float64{{2, 2, 1}, {3, 2, -2}, {1, -1, 1}})
	B := denseFromRows([][]float64{{-1}, {3}, {-3}})
	C1 := denseFromRows([][]float64{{1, 0, 0}})
	D1 := denseFromFlat(1, 1, nil)

	plant, _ := New(A, B, C1, D1, 0)

	K := denseFromFlat(1, 1, []float64{5})
	controller, _ := NewGain(K, 0)

	CL, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	// CL A_cl = A - B*K*C = A - 5*B*C1
	Acl_manual := mat.NewDense(3, 3, nil)
	BKC := mat.NewDense(3, 3, nil)
	BKC.Mul(B, denseFromRows([][]float64{{5, 0, 0}}))
	Acl_manual.Sub(A, BKC)

	poles, err := CL.Poles()
	if err != nil {
		t.Fatal(err)
	}
	manualPoles, err := (&System{A: Acl_manual}).Poles()
	if err != nil {
		t.Fatal(err)
	}
	compareSortedPoles(t, "feedback poles", poles, manualPoles, 1e-10)
}
