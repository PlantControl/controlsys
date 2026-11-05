package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// XBVOCH: E = 2⁻⁶⁶⁴·I with A = E·M makes E⁻¹B = [0; 2¹¹⁶²] overflow while
// the transfer function C·(sI−M)⁻¹·E⁻¹B stays near 2⁹⁹⁶. The oracle is the
// explicit model under the exact similarity x' = 2⁻⁶⁶⁶·x.
const explicitOverflowShift = 666

var explicitOverflowM = []float64{-1, 0.5, 0, -2}

func explicitOverflowDescriptor(t *testing.T, dt float64) *System {
	t.Helper()
	e := math.Ldexp(1, -664)
	A := mat.NewDense(2, 2, nil)
	A.Scale(e, mat.NewDense(2, 2, explicitOverflowM))
	E := mat.NewDense(2, 2, []float64{e, 0, 0, e})
	B := mat.NewDense(2, 1, []float64{0, math.Ldexp(1, 498)})
	c := math.Ldexp(1, -166)
	C := mat.NewDense(1, 2, []float64{c, c})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func explicitOverflowOracle(t *testing.T, dt float64) *System {
	t.Helper()
	c := math.Ldexp(1, -166+explicitOverflowShift)
	sys, err := New(mat.NewDense(2, 2, explicitOverflowM),
		mat.NewDense(2, 1, []float64{0, math.Ldexp(1, 1162-explicitOverflowShift)}),
		mat.NewDense(1, 2, []float64{c, c}), mat.NewDense(1, 1, nil), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func requireExplicitOverflowFinite(t *testing.T, name string, m *mat.Dense) {
	t.Helper()
	if err := finiteDense("test", name, m, ErrOverflow); err != nil {
		t.Fatal(err)
	}
}

func assertExplicitOverflowEvalFr(t *testing.T, got, want *System, tol float64) {
	t.Helper()
	for _, s := range []complex128{0.1i, 1i, 0.5 + 3i, 10i} {
		g, err := got.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		w, err := want.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(g[0][0]-w[0][0]) / cmplx.Abs(w[0][0]); !(d <= tol) {
			t.Fatalf("s=%v: H=%v, want %v (rel %g)", s, g[0][0], w[0][0], d)
		}
	}
}

func assertExplicitOverflowSamples(t *testing.T, got, want *TimeResponse, tol float64) {
	t.Helper()
	if len(got.T) != len(want.T) {
		t.Fatalf("len(T) = %d, want %d", len(got.T), len(want.T))
	}
	requireExplicitOverflowFinite(t, "Y", got.Y)
	scale := mat.Norm(want.Y, math.Inf(1))
	var d mat.Dense
	d.Sub(got.Y, want.Y)
	if r := mat.Norm(&d, math.Inf(1)) / scale; !(r <= tol) {
		t.Fatalf("response differs from oracle: rel %g", r)
	}
}

func TestToExplicit_OverflowingEinvBIsBalanced(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	got, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	if got.IsDescriptor() {
		t.Fatal("result is still a descriptor model")
	}
	requireExplicitOverflowFinite(t, "A", got.A)
	requireExplicitOverflowFinite(t, "B", got.B)
	requireExplicitOverflowFinite(t, "C", got.C)
	if !mat.Equal(got.A, mat.NewDense(2, 2, explicitOverflowM)) {
		t.Fatalf("A = %v, want M", mat.Formatted(got.A))
	}
	assertExplicitOverflowEvalFr(t, got, explicitOverflowOracle(t, 0), 1e-14)
}

func TestToExplicit_DecimalRepro(t *testing.T) {
	e := 1e-200
	A := mat.NewDense(2, 2, []float64{-e, 0.5 * e, 0, -2 * e})
	E := mat.NewDense(2, 2, []float64{e, 0, 0, e})
	sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{0, 1e150}),
		mat.NewDense(1, 2, []float64{1e-50, 1e-50}), mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	requireExplicitOverflowFinite(t, "B", got.B)
	requireExplicitOverflowFinite(t, "C", got.C)
	h, err := got.EvalFr(0)
	if err != nil {
		t.Fatal(err)
	}
	want := 1e-50 * 1e350 * 0.75
	if d := math.Abs(real(h[0][0])-want) / want; d > 1e-13 {
		t.Fatalf("DC gain %g, want %g", real(h[0][0]), want)
	}
}

func TestToExplicit_OverflowingEinvAReturnsErrOverflow(t *testing.T) {
	e := math.Ldexp(1, -664)
	A := mat.NewDense(2, 2, nil)
	A.Scale(math.Ldexp(1, 500), mat.NewDense(2, 2, explicitOverflowM))
	E := mat.NewDense(2, 2, []float64{e, 0, 0, e})
	sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.ToExplicit()
	if !errors.Is(err, ErrOverflow) || got != nil {
		t.Fatalf("ToExplicit = %v, %v; want nil, ErrOverflow", got, err)
	}
}

func TestStepImpulse_DescriptorWithOverflowingEinvB(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	oracle := explicitOverflowOracle(t, 0)
	for name, f := range map[string]func(*System, float64) (*TimeResponse, error){"Step": Step, "Impulse": Impulse} {
		got, err := f(sys, 5)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want, err := f(oracle, 5)
		if err != nil {
			t.Fatal(err)
		}
		assertExplicitOverflowSamples(t, got, want, 1e-12)
	}
}

func TestInitial_DescriptorWithOverflowingEinvBKeepsStateCoordinates(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	x0 := mat.NewVecDense(2, []float64{1, -3})
	got, err := Initial(sys, x0, 5)
	if err != nil {
		t.Fatal(err)
	}
	x0Oracle := mat.NewVecDense(2, nil)
	x0Oracle.ScaleVec(math.Ldexp(1, -explicitOverflowShift), x0)
	want, err := Initial(explicitOverflowOracle(t, 0), x0Oracle, 5)
	if err != nil {
		t.Fatal(err)
	}
	assertExplicitOverflowSamples(t, got, want, 1e-12)
}

func TestC2D_DescriptorWithOverflowingEinvB(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	got, err := sys.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	requireExplicitOverflowFinite(t, "B", got.B)
	requireExplicitOverflowFinite(t, "C", got.C)
	want, err := explicitOverflowOracle(t, 0).C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertExplicitOverflowEvalFr(t, got, want, 1e-12)

	if _, _, err := sys.C2DMap(0.1, C2DOptions{}); !errors.Is(err, ErrOverflow) {
		t.Fatalf("C2DMap err = %v, want ErrOverflow", err)
	}
}

func TestTransferFunction_DescriptorWithOverflowingEinvB(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0)
	got, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := explicitOverflowOracle(t, 0).TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []complex128{0, 1i, 3i} {
		g, err := got.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		w, err := want.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		if d := cmplx.Abs(g[0][0]-w[0][0]) / cmplx.Abs(w[0][0]); !(d <= 1e-12) {
			t.Fatalf("s=%v: G=%v, want %v", s, g[0][0], w[0][0])
		}
	}
}

func TestSimulate_DescriptorWithOverflowingEinvBFinalState(t *testing.T) {
	sys := explicitOverflowDescriptor(t, 0.1)
	oracle := explicitOverflowOracle(t, 0.1)
	const steps = 20
	u := mat.NewDense(1, steps, nil)
	for j := range steps {
		u.Set(0, j, math.Ldexp(1, -200))
	}
	x0 := mat.NewVecDense(2, []float64{math.Ldexp(1, 900), -math.Ldexp(1, 899)})
	got, err := sys.Simulate(u, x0, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	x0Oracle := mat.NewVecDense(2, nil)
	x0Oracle.ScaleVec(math.Ldexp(1, -explicitOverflowShift), x0)
	want, err := oracle.Simulate(u, x0Oracle, &SimulateOpts{FinalState: true})
	if err != nil {
		t.Fatal(err)
	}
	var d mat.Dense
	d.Sub(got.Y, want.Y)
	if r := mat.Norm(&d, math.Inf(1)) / mat.Norm(want.Y, math.Inf(1)); !(r <= 1e-13) {
		t.Fatalf("Y rel diff %g", r)
	}
	for i := range 2 {
		w := math.Ldexp(want.XFinal.AtVec(i), explicitOverflowShift)
		if g := got.XFinal.AtVec(i); math.Abs(g-w) > 1e-13*math.Abs(w) {
			t.Fatalf("XFinal[%d] = %g, want %g", i, g, w)
		}
	}

	big := mat.NewDense(1, steps, nil)
	for j := range steps {
		big.Set(0, j, 1)
	}
	if res, err := sys.Simulate(big, nil, &SimulateOpts{FinalState: true}); !errors.Is(err, ErrOverflow) || res != nil {
		t.Fatalf("Simulate = %v, %v; want nil, ErrOverflow", res, err)
	}
}

func TestToExplicit_OverflowingEinvBNoOutputs(t *testing.T) {
	e := math.Ldexp(1, -664)
	A := mat.NewDense(2, 2, nil)
	A.Scale(e, mat.NewDense(2, 2, explicitOverflowM))
	E := mat.NewDense(2, 2, []float64{e, 0, 0, e})
	sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{0, math.Ldexp(1, 498)}),
		&mat.Dense{}, &mat.Dense{}, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	requireExplicitOverflowFinite(t, "B", got.B)
}
