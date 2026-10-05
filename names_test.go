package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"reflect"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const nameTol = 1e-10

func TestCopyStringSlice(t *testing.T) {
	if got := copyStringSlice(nil); got != nil {
		t.Errorf("nil input: got %v, want nil", got)
	}

	orig := []string{"a", "b", "c"}
	cp := copyStringSlice(orig)
	if !reflect.DeepEqual(cp, orig) {
		t.Errorf("got %v, want %v", cp, orig)
	}

	cp[0] = "z"
	if orig[0] != "a" {
		t.Error("modifying copy mutated original")
	}
}

func TestConcatStringSlices(t *testing.T) {
	if got := concatStringSlices([][]string{nil, nil}, []int{2, 3}); got != nil {
		t.Errorf("all nil: got %v, want nil", got)
	}

	got := concatStringSlices([][]string{nil, {"x", "y"}}, []int{2, 2})
	want := []string{"", "", "x", "y"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("one nil: got %v, want %v", got, want)
	}

	got = concatStringSlices([][]string{{"a", "b"}, {"c"}}, []int{2, 1})
	want = []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("both non-nil: got %v, want %v", got, want)
	}
}

func TestSelectStringSlice(t *testing.T) {
	if got := selectStringSlice(nil, []int{0, 1}); got != nil {
		t.Errorf("nil input: got %v, want nil", got)
	}

	names := []string{"a", "b", "c", "d"}
	got := selectStringSlice(names, []int{3, 0, 2})
	want := []string{"d", "a", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestLookupSignalIndex(t *testing.T) {
	names := []string{"u", "y", "e"}

	idx, err := lookupSignalIndex(names, "y")
	if err != nil {
		t.Fatal(err)
	}
	if idx != 1 {
		t.Errorf("got %d, want 1", idx)
	}

	_, err = lookupSignalIndex(names, "missing")
	if !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("got err=%v, want ErrSignalNotFound", err)
	}
}

func TestLookupSignalIndices(t *testing.T) {
	names := []string{"a", "b", "c", "d"}

	got, err := lookupSignalIndices(names, []string{"c", "a", "d"})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 0, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}

	_, err = lookupSignalIndices(names, []string{"a", "missing"})
	if !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("got err=%v, want ErrSignalNotFound", err)
	}
}

func TestCopy_Names(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	sys.StateName = []string{"x1", "x2"}
	sys.Notes = "test"

	cp := sys.Copy()
	if !reflect.DeepEqual(cp.InputName, sys.InputName) {
		t.Errorf("InputName mismatch")
	}
	if !reflect.DeepEqual(cp.OutputName, sys.OutputName) {
		t.Errorf("OutputName mismatch")
	}
	if !reflect.DeepEqual(cp.StateName, sys.StateName) {
		t.Errorf("StateName mismatch")
	}
	if cp.Notes != sys.Notes {
		t.Errorf("Notes mismatch")
	}

	cp.InputName[0] = "changed"
	cp.OutputName[0] = "changed"
	cp.StateName[0] = "changed"
	cp.Notes = "changed"

	if sys.InputName[0] != "u" {
		t.Error("InputName not deep copied")
	}
	if sys.OutputName[0] != "y" {
		t.Error("OutputName not deep copied")
	}
	if sys.StateName[0] != "x1" {
		t.Error("StateName not deep copied")
	}
	if sys.Notes != "test" {
		t.Error("Notes not independent")
	}
}

func makeSISO(a, b, c, d float64) *System {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{a}),
		mat.NewDense(1, 1, []float64{b}),
		mat.NewDense(1, 1, []float64{c}),
		mat.NewDense(1, 1, []float64{d}),
		0,
	)
	return sys
}

func TestSeries_Names(t *testing.T) {
	g1 := makeSISO(-1, 1, 1, 0)
	g1.InputName = []string{"u"}
	g1.OutputName = []string{"x"}
	g1.StateName = []string{"s1"}

	g2 := makeSISO(-2, 1, 1, 0)
	g2.InputName = []string{"x"}
	g2.OutputName = []string{"y"}
	g2.StateName = []string{"s2"}

	result, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u"}) {
		t.Errorf("InputName = %v, want [u]", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y"}) {
		t.Errorf("OutputName = %v, want [y]", result.OutputName)
	}
	if !reflect.DeepEqual(result.StateName, []string{"s1", "s2"}) {
		t.Errorf("StateName = %v, want [s1 s2]", result.StateName)
	}
}

func TestParallel_Names(t *testing.T) {
	g1 := makeSISO(-1, 1, 1, 0)
	g1.InputName = []string{"u"}
	g1.OutputName = []string{"y"}
	g1.StateName = []string{"s1"}

	g2 := makeSISO(-2, 1, 1, 0)
	g2.StateName = []string{"s2"}

	result, err := Parallel(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u"}) {
		t.Errorf("InputName = %v, want [u]", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y"}) {
		t.Errorf("OutputName = %v, want [y]", result.OutputName)
	}
	if !reflect.DeepEqual(result.StateName, []string{"s1", "s2"}) {
		t.Errorf("StateName = %v, want [s1 s2]", result.StateName)
	}
}

func TestFeedback_Names(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	plant.InputName = []string{"u"}
	plant.OutputName = []string{"y"}
	plant.StateName = []string{"p1"}

	ctrl := makeSISO(-3, 1, 2, 0)
	ctrl.StateName = []string{"k1"}

	result, err := Feedback(plant, ctrl, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u"}) {
		t.Errorf("InputName = %v, want [u]", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y"}) {
		t.Errorf("OutputName = %v, want [y]", result.OutputName)
	}
	if !reflect.DeepEqual(result.StateName, []string{"p1", "k1"}) {
		t.Errorf("StateName = %v, want [p1 k1]", result.StateName)
	}
}

func TestAppend_Names(t *testing.T) {
	g1 := makeSISO(-1, 1, 1, 0)
	g1.InputName = []string{"u1"}
	g1.OutputName = []string{"y1"}
	g1.StateName = []string{"s1"}

	g2 := makeSISO(-2, 1, 1, 0)
	g2.InputName = []string{"u2"}
	g2.OutputName = []string{"y2"}
	g2.StateName = []string{"s2"}

	result, err := Append(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u1", "u2"}) {
		t.Errorf("InputName = %v", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y1", "y2"}) {
		t.Errorf("OutputName = %v", result.OutputName)
	}
	if !reflect.DeepEqual(result.StateName, []string{"s1", "s2"}) {
		t.Errorf("StateName = %v", result.StateName)
	}
}

func TestBlkDiag_Names(t *testing.T) {
	g1 := makeSISO(-1, 1, 1, 0)
	g1.InputName = []string{"u1"}
	g1.OutputName = []string{"y1"}

	g2 := makeSISO(-2, 1, 1, 0)
	g2.InputName = []string{"u2"}
	g2.OutputName = []string{"y2"}

	g3 := makeSISO(-3, 1, 1, 0)
	g3.InputName = []string{"u3"}
	g3.OutputName = []string{"y3"}

	result, err := BlkDiag(g1, g2, g3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u1", "u2", "u3"}) {
		t.Errorf("InputName = %v", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y1", "y2", "y3"}) {
		t.Errorf("OutputName = %v", result.OutputName)
	}
}

func TestConnect_Names(t *testing.T) {
	g1 := makeSISO(-1, 1, 1, 0)
	g1.InputName = []string{"u1"}
	g1.OutputName = []string{"y1"}

	g2 := makeSISO(-2, 1, 1, 0)
	g2.InputName = []string{"u2"}
	g2.OutputName = []string{"y2"}

	aug, err := Append(g1, g2)
	if err != nil {
		t.Fatal(err)
	}

	Q := mat.NewDense(2, 2, []float64{
		0, 0,
		1, 0,
	})
	result, err := Connect(aug, Q, []int{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"u1"}) {
		t.Errorf("InputName = %v, want [u1]", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"y2"}) {
		t.Errorf("OutputName = %v, want [y2]", result.OutputName)
	}
}

func TestLFT_Names(t *testing.T) {
	M, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	M.InputName = []string{"w", "u_delta"}
	M.OutputName = []string{"z", "y_delta"}

	Delta := makeSISO(-5, 1, 1, 0)

	result, err := LFT(M, Delta)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.InputName, []string{"w"}) {
		t.Errorf("InputName = %v, want [w]", result.InputName)
	}
	if !reflect.DeepEqual(result.OutputName, []string{"z"}) {
		t.Errorf("OutputName = %v, want [z]", result.OutputName)
	}
}

func TestDiscretize_Names(t *testing.T) {
	sys := makeSISO(-1, 1, 1, 0)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	sys.StateName = []string{"x"}
	sys.Notes = "continuous"

	disc, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(disc.InputName, []string{"u"}) {
		t.Errorf("InputName = %v", disc.InputName)
	}
	if !reflect.DeepEqual(disc.OutputName, []string{"y"}) {
		t.Errorf("OutputName = %v", disc.OutputName)
	}
	if !reflect.DeepEqual(disc.StateName, []string{"x"}) {
		t.Errorf("StateName = %v", disc.StateName)
	}
	if disc.Notes != "continuous" {
		t.Errorf("Notes = %q", disc.Notes)
	}
}

func TestSumBlk_Names(t *testing.T) {
	sys, err := SumBlk("e = r - y")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sys.OutputName, []string{"e"}) {
		t.Errorf("OutputName = %v, want [e]", sys.OutputName)
	}
	if !reflect.DeepEqual(sys.InputName, []string{"r", "y"}) {
		t.Errorf("InputName = %v, want [r y]", sys.InputName)
	}

	_, _, p := sys.Dims()
	if p != 1 {
		t.Errorf("outputs = %d, want 1", p)
	}
}

func TestSumBlk_Names_Vector(t *testing.T) {
	sys, err := SumBlk("e = r - y", 2)
	if err != nil {
		t.Fatal(err)
	}
	wantOut := []string{"e(1)", "e(2)"}
	if !reflect.DeepEqual(sys.OutputName, wantOut) {
		t.Errorf("OutputName = %v, want %v", sys.OutputName, wantOut)
	}
	wantIn := []string{"r(1)", "r(2)", "y(1)", "y(2)"}
	if !reflect.DeepEqual(sys.InputName, wantIn) {
		t.Errorf("InputName = %v, want %v", sys.InputName, wantIn)
	}
}

func TestSelectByIndex(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 3, nil),
		0,
	)
	sys.InputName = []string{"u1", "u2", "u3"}
	sys.OutputName = []string{"y1", "y2"}
	sys.StateName = []string{"x1", "x2"}

	sub, err := sys.SelectByIndex([]int{0, 2}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sub.InputName, []string{"u1", "u3"}) {
		t.Errorf("InputName = %v", sub.InputName)
	}
	if !reflect.DeepEqual(sub.OutputName, []string{"y2"}) {
		t.Errorf("OutputName = %v", sub.OutputName)
	}
	if !reflect.DeepEqual(sub.StateName, []string{"x1", "x2"}) {
		t.Errorf("StateName = %v", sub.StateName)
	}
}

func TestSelectByName(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 3, nil),
		0,
	)
	sys.InputName = []string{"u1", "u2", "u3"}
	sys.OutputName = []string{"y1", "y2"}

	sub, err := sys.SelectByName([]string{"u3", "u1"}, []string{"y2"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sub.InputName, []string{"u3", "u1"}) {
		t.Errorf("InputName = %v", sub.InputName)
	}
	if !reflect.DeepEqual(sub.OutputName, []string{"y2"}) {
		t.Errorf("OutputName = %v", sub.OutputName)
	}
}

func TestSelectByName_NotFound(t *testing.T) {
	sys := makeSISO(-1, 1, 1, 0)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}

	_, err := sys.SelectByName([]string{"missing"}, []string{"y"})
	if !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("input err=%v, want ErrSignalNotFound", err)
	}

	_, err = sys.SelectByName([]string{"u"}, []string{"missing"})
	if !errors.Is(err, ErrSignalNotFound) {
		t.Errorf("output err=%v, want ErrSignalNotFound", err)
	}
}

func TestConnectByName_Feedback(t *testing.T) {
	P := makeSISO(-1, 1, 1, 0)
	P.InputName = []string{"u"}
	P.OutputName = []string{"y"}

	K := makeSISO(-3, 1, 2, 0)
	K.InputName = []string{"y"}
	K.OutputName = []string{"v"}

	Sum, err := SumBlk("u = r - v")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ConnectByName(
		[]*System{P, K, Sum},
		[]Connection{
			{From: "y", To: "y"},
			{From: "v", To: "v"},
			{From: "u", To: "u"},
		},
		[]string{"r"},
		[]string{"y"},
	)
	if err != nil {
		t.Fatal(err)
	}

	ref, err := Feedback(P, K, -1)
	if err != nil {
		t.Fatal(err)
	}

	s := 1 + 2i
	hGot := evalTF(result, s)
	hWant := evalTF(ref, s)
	if cmplx.Abs(hGot[0][0]-hWant[0][0]) > nameTol {
		t.Errorf("H(s)=%v, want %v", hGot[0][0], hWant[0][0])
	}
}

func TestConnectByName_AlgebraicLoopDiagnostic(t *testing.T) {
	first, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	first.InputName = []string{"first.input"}
	first.OutputName = []string{"first.output"}
	second, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	second.InputName = []string{"second.input"}
	second.OutputName = []string{"second.output"}

	_, err := ConnectByName(
		[]*System{first, second},
		[]Connection{
			{From: "first.output", To: "second.input"},
			{From: "second.output", To: "first.input"},
		},
		[]string{"first.input"},
		[]string{"first.output"},
	)
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Fatalf("err = %v, want ErrAlgebraicLoop", err)
	}

	var diagnostic *AlgebraicLoopError
	if !errors.As(err, &diagnostic) {
		t.Fatalf("err = %T, want *AlgebraicLoopError", err)
	}
	wantSignals := []string{
		"first.input",
		"second.input",
		"first.output",
		"second.output",
	}
	if !reflect.DeepEqual(diagnostic.Signals, wantSignals) {
		t.Errorf("signals = %v, want %v", diagnostic.Signals, wantSignals)
	}
	if !math.IsInf(diagnostic.Condition, 1) {
		t.Errorf("condition = %g, want +Inf", diagnostic.Condition)
	}
}

func TestConnectByName_NearSingularDiagnostic(t *testing.T) {
	model, _ := NewGain(mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, -1,
		0, -1, -2 * eps(),
	}), 0)
	model.InputName = []string{"stable.input", "critical1.input", "critical2.input"}
	model.OutputName = []string{"stable.output", "critical1.output", "critical2.output"}

	_, err := ConnectByName(
		[]*System{model},
		[]Connection{
			{From: "stable.output", To: "stable.input"},
			{From: "critical1.output", To: "critical1.input"},
			{From: "critical2.output", To: "critical2.input"},
		},
		[]string{"stable.input"},
		[]string{"stable.output"},
	)
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Fatalf("err = %v, want ErrAlgebraicLoop", err)
	}

	var diagnostic *AlgebraicLoopError
	if !errors.As(err, &diagnostic) {
		t.Fatalf("err = %T, want *AlgebraicLoopError", err)
	}
	wantSignals := []string{
		"critical1.input",
		"critical2.input",
		"critical1.output",
		"critical2.output",
	}
	if !reflect.DeepEqual(diagnostic.Signals, wantSignals) {
		t.Errorf("signals = %v, want %v", diagnostic.Signals, wantSignals)
	}
	if math.IsInf(diagnostic.Condition, 0) || diagnostic.Condition*eps() < 1 {
		t.Errorf("condition = %g, want finite rejected condition", diagnostic.Condition)
	}
}

func TestConnectByName_DynamicFeedbackIsWellPosed(t *testing.T) {
	dynamic, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	dynamic.InputName = []string{"dynamic.input"}
	dynamic.OutputName = []string{"dynamic.output"}
	static, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	static.InputName = []string{"static.input"}
	static.OutputName = []string{"static.output"}

	_, err := ConnectByName(
		[]*System{dynamic, static},
		[]Connection{
			{From: "dynamic.output", To: "static.input"},
			{From: "static.output", To: "dynamic.input"},
		},
		[]string{"dynamic.input"},
		[]string{"dynamic.output"},
	)
	if err != nil {
		t.Fatalf("dynamic feedback: %v", err)
	}
}

func TestConnectByName_WithSumBlk(t *testing.T) {
	P := makeSISO(-2, 1, 3, 0)
	P.InputName = []string{"u"}
	P.OutputName = []string{"y"}

	K := makeSISO(-4, 1, 5, 0)
	K.InputName = []string{"y"}
	K.OutputName = []string{"v"}

	Sum, err := SumBlk("u = r - v")
	if err != nil {
		t.Fatal(err)
	}

	result, err := ConnectByName(
		[]*System{P, K, Sum},
		[]Connection{
			{From: "y", To: "y"},
			{From: "v", To: "v"},
			{From: "u", To: "u"},
		},
		[]string{"r"},
		[]string{"y"},
	)
	if err != nil {
		t.Fatal(err)
	}

	ref, err := Feedback(P, K, -1)
	if err != nil {
		t.Fatal(err)
	}

	s := 0.5 + 3i
	hGot := evalTF(result, s)
	hWant := evalTF(ref, s)
	if cmplx.Abs(hGot[0][0]-hWant[0][0]) > nameTol {
		t.Errorf("H(s)=%v, want %v", hGot[0][0], hWant[0][0])
	}
}

func TestConnectByName_FullWorkflow(t *testing.T) {
	P, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	P.InputName = []string{"u"}
	P.OutputName = []string{"y"}

	K, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	K.InputName = []string{"y"}
	K.OutputName = []string{"v"}

	Sum, err := SumBlk("u = r - v")
	if err != nil {
		t.Fatal(err)
	}

	cl, err := ConnectByName(
		[]*System{P, K, Sum},
		[]Connection{
			{From: "y", To: "y"},
			{From: "v", To: "v"},
			{From: "u", To: "u"},
		},
		[]string{"r"},
		[]string{"y"},
	)
	if err != nil {
		t.Fatal(err)
	}

	ref, err := Feedback(P, K, -1)
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range []complex128{0.1 + 0.5i, 1 + 2i, 5 + 0i, 0 + 10i} {
		hCl := evalTF(cl, s)
		hRef := evalTF(ref, s)
		diff := cmplx.Abs(hCl[0][0] - hRef[0][0])
		if diff > nameTol {
			t.Errorf("s=%v: H=%v, want %v (diff=%v)", s, hCl[0][0], hRef[0][0], diff)
		}
	}
}

func TestSetInputName_Validation(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)

	if err := sys.SetInputName("u1", "u2"); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sys.InputName, []string{"u1", "u2"}) {
		t.Errorf("InputName = %v, want [u1 u2]", sys.InputName)
	}

	err := sys.SetInputName("u1", "u2", "u3")
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSetInputName_VectorExpansion(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 3, []float64{0, 0, 0}),
		0,
	)

	if err := sys.SetInputName("ctrl"); err != nil {
		t.Fatal(err)
	}
	want := []string{"ctrl(1)", "ctrl(2)", "ctrl(3)"}
	if !reflect.DeepEqual(sys.InputName, want) {
		t.Errorf("InputName = %v, want %v", sys.InputName, want)
	}
}

func TestSetStateName_VectorExpansion(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	if err := sys.SetStateName("state"); err != nil {
		t.Fatal(err)
	}
	want := []string{"state(1)", "state(2)"}
	if !reflect.DeepEqual(sys.StateName, want) {
		t.Errorf("StateName = %v, want %v", sys.StateName, want)
	}
}

func TestSystem_String(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -5, -2}),
		mat.NewDense(2, 1, []float64{0, 3}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	s := sys.String()
	if !strings.Contains(s, "A =") {
		t.Error("missing A matrix header")
	}
	if !strings.Contains(s, "B =") {
		t.Error("missing B matrix header")
	}
	if !strings.Contains(s, "C =") {
		t.Error("missing C matrix header")
	}
	if !strings.Contains(s, "D =") {
		t.Error("missing D matrix header")
	}
	if !strings.Contains(s, "x1") {
		t.Error("missing auto-generated state label x1")
	}
	if !strings.Contains(s, "u1") {
		t.Error("missing auto-generated input label u1")
	}
	if !strings.Contains(s, "y1") {
		t.Error("missing auto-generated output label y1")
	}
	if !strings.Contains(s, "Continuous-time") {
		t.Error("missing continuous-time footer")
	}
}

func TestSystem_String_Named(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.InputName = []string{"voltage"}
	sys.OutputName = []string{"speed"}
	sys.StateName = []string{"omega"}

	s := sys.String()
	if !strings.Contains(s, "voltage") {
		t.Error("missing custom input name")
	}
	if !strings.Contains(s, "speed") {
		t.Error("missing custom output name")
	}
	if !strings.Contains(s, "omega") {
		t.Error("missing custom state name")
	}
	if !strings.Contains(s, "Discrete-time") {
		t.Error("missing discrete-time footer")
	}
}

func TestTransferFunc_Names(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputName = []string{"force"}
	sys.OutputName = []string{"position"}

	tfResult, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tfResult.TF.InputName, []string{"force"}) {
		t.Errorf("TF InputName = %v, want [force]", tfResult.TF.InputName)
	}
	if !reflect.DeepEqual(tfResult.TF.OutputName, []string{"position"}) {
		t.Errorf("TF OutputName = %v, want [position]", tfResult.TF.OutputName)
	}

	ssResult, err := tfResult.TF.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ssResult.Sys.InputName, []string{"force"}) {
		t.Errorf("SS InputName = %v, want [force]", ssResult.Sys.InputName)
	}
	if !reflect.DeepEqual(ssResult.Sys.OutputName, []string{"position"}) {
		t.Errorf("SS OutputName = %v, want [position]", ssResult.Sys.OutputName)
	}
}

func TestZPK_Names(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}

	zpkResult, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(zpkResult.ZPK.InputName, []string{"u"}) {
		t.Errorf("ZPK InputName = %v, want [u]", zpkResult.ZPK.InputName)
	}
	if !reflect.DeepEqual(zpkResult.ZPK.OutputName, []string{"y"}) {
		t.Errorf("ZPK OutputName = %v, want [y]", zpkResult.ZPK.OutputName)
	}
}

func TestStep_OutputName(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputName = []string{"temp"}

	resp, err := Step(sys, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp.OutputName, []string{"temp"}) {
		t.Errorf("OutputName = %v, want [temp]", resp.OutputName)
	}
}

func TestFreqResponse_Names(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}

	fr, err := sys.FreqResponse([]float64{0.1, 1, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fr.InputName, []string{"u"}) {
		t.Errorf("InputName = %v, want [u]", fr.InputName)
	}
	if !reflect.DeepEqual(fr.OutputName, []string{"y"}) {
		t.Errorf("OutputName = %v, want [y]", fr.OutputName)
	}
}

func TestSelectByIndexKeepsDelayFields(t *testing.T) {
	ins, outs := []int{1, 0}, []int{2, 0}
	for _, dt := range []float64{0, 0.1} {
		for name, mk := range map[string]func(*testing.T, float64) *System{"io": fieldIODelay, "lft": fieldLFT, "descriptor": fieldDescriptor} {
			orig := mk(t, dt)
			sub, err := orig.SelectByIndex(ins, outs)
			if err != nil {
				t.Fatalf("%s/dt=%g: %v", name, dt, err)
			}
			assertFieldResponse(t, fmt.Sprintf("%s/dt=%g", name, dt), sub, func(s complex128) [][]complex128 {
				return fieldSub(fieldOracle(orig, s), outs, ins)
			})
		}
	}
}

func TestSelectByIndexStaticKeepsDelays(t *testing.T) {
	for _, dt := range []float64{0, 1} {
		g, err := NewGain(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}), dt)
		if err != nil {
			t.Fatal(err)
		}
		k := 0.25
		if dt > 0 {
			k = 1
		}
		if err := g.SetDelay(mat.NewDense(2, 3, []float64{k, 0, 2 * k, 0, 3 * k, k})); err != nil {
			t.Fatal(err)
		}
		if err := g.SetInputDelay([]float64{2 * k, 0, k}); err != nil {
			t.Fatal(err)
		}
		if err := g.SetOutputDelay([]float64{k, 4 * k}); err != nil {
			t.Fatal(err)
		}
		ins, outs := []int{2, 1}, []int{1}
		sub, err := g.SelectByIndex(ins, outs)
		if err != nil {
			t.Fatal(err)
		}
		s := fieldPoints(dt)[0]
		want := []complex128{6 * fieldDelay(dt, s, 6*k), 5 * fieldDelay(dt, s, 7*k)}
		got := fieldOracle(sub, s)
		for j := range want {
			if d := cmplx.Abs(got[0][j] - want[j]); d > fieldTol {
				t.Errorf("dt=%g: G[0][%d] differs by %.3g", dt, j, d)
			}
		}
	}
}

func TestSelectByIndexEmptySelection(t *testing.T) {
	sys := fieldIODelay(t, 0)
	sys.InputName = []string{"u1", "u2"}
	sys.OutputName = []string{"y1", "y2", "y3"}
	cases := []struct {
		name         string
		sel          func() (*System, error)
		wantM, wantP int
	}{
		{"none", func() (*System, error) { return sys.SelectByIndex(nil, nil) }, 0, 0},
		{"no inputs", func() (*System, error) { return sys.SelectByIndex([]int{}, []int{1}) }, 0, 1},
		{"no outputs", func() (*System, error) { return sys.SelectByIndex([]int{0}, nil) }, 1, 0},
		{"by name none", func() (*System, error) { return sys.SelectByName(nil, nil) }, 0, 0},
		{"static none", func() (*System, error) {
			g, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
			return g.SelectByIndex(nil, nil)
		}, 0, 0},
	}
	for _, tc := range cases {
		got, err := tc.sel()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if err := got.Validate(); err != nil {
			t.Fatalf("%s: Validate: %v", tc.name, err)
		}
		if _, m, p := got.Dims(); m != tc.wantM || p != tc.wantP {
			t.Errorf("%s: dims m=%d p=%d, want m=%d p=%d", tc.name, m, p, tc.wantM, tc.wantP)
		}
	}
}

func TestConnectByNameRejectsDuplicateConnection(t *testing.T) {
	P := makeSISO(-1, 1, 1, 0)
	P.InputName = []string{"u"}
	P.OutputName = []string{"y"}
	_, err := ConnectByName([]*System{P}, []Connection{{From: "y", To: "u"}, {From: "y", To: "u"}}, []string{"u"}, []string{"y"})
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("err = %v, want ErrInvalidArgument", err)
	}
}
