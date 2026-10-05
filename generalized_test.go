package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestGeneralizedModelCurrentValueAndAnalysisPoint(t *testing.T) {
	k, _ := newBoundedReal("K", 2, 0, 10)
	controller := mustOK(NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0))
	gm, err := NewGeneralizedModel("loop", controller)
	if err != nil {
		t.Fatalf("NewGeneralizedModel: %v", err)
	}
	gm.SetInputName("error")
	gm.SetOutputName("actuator")
	if err := gm.InsertAnalysisPoint("plant_input"); err != nil {
		t.Fatal(err)
	}

	sys, err := gm.CurrentSystem()
	if err != nil {
		t.Fatalf("CurrentSystem: %v", err)
	}
	if sys.D.At(0, 0) != 2 {
		t.Fatalf("current gain = %g, want 2", sys.D.At(0, 0))
	}
	if !sameStrings(sys.InputName, []string{"error"}) || !sameStrings(sys.OutputName, []string{"actuator"}) {
		t.Fatalf("metadata = %v/%v", sys.InputName, sys.OutputName)
	}
	if !gm.HasAnalysisPoint("plant_input") {
		t.Fatal("analysis point not found")
	}
	if _, err := gm.AnalysisPoint("missing"); !errors.Is(err, ErrSignalNotFound) {
		t.Fatalf("missing analysis point err = %v, want ErrSignalNotFound", err)
	}
}

func TestGeneralizedClosedLoopAnalysisHelpers(t *testing.T) {
	plant, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.4, -2, -3}),
		mat.NewDense(2, 1, []float64{1, -0.5}),
		mat.NewDense(1, 2, []float64{2, -1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	k, _ := newBoundedReal("K", 1.5, 0.1, 5)
	controller := mustOK(NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0))
	loop, err := NewGeneralizedClosedLoop("cl", plant, controller, "u")
	if err != nil {
		t.Fatalf("NewGeneralizedClosedLoop: %v", err)
	}

	L, err := loop.OpenLoop("u")
	if err != nil {
		t.Fatalf("OpenLoop: %v", err)
	}
	T, err := loop.ComplementarySensitivity("u")
	if err != nil {
		t.Fatalf("ComplementarySensitivity: %v", err)
	}
	S, err := loop.Sensitivity("u")
	if err != nil {
		t.Fatalf("Sensitivity: %v", err)
	}
	CL, err := loop.ClosedLoop("u")
	if err != nil {
		t.Fatalf("ClosedLoop: %v", err)
	}
	if _, m, p := L.Dims(); m != 1 || p != 1 {
		t.Fatalf("open-loop dims = (_, %d, %d), want SISO", m, p)
	}
	omega := []float64{0.2, 1.0}
	tResp, _ := T.FreqResponse(omega)
	clResp, _ := CL.FreqResponse(omega)
	for i := range omega {
		if cmplx.Abs(tResp.At(i, 0, 0)-clResp.At(i, 0, 0)) > 1e-10 {
			t.Fatalf("closed-loop and complementary sensitivity differ at %g", omega[i])
		}
	}
	sResp, _ := S.FreqResponse(omega)
	for i := range omega {
		sum := sResp.At(i, 0, 0) + tResp.At(i, 0, 0)
		if cmplx.Abs(sum-1) > 1e-8 {
			t.Fatalf("S+T = %v at %g, want 1", sum, omega[i])
		}
	}
}

func TestGeneralizedClosedLoopAnalysisPointsBindDistinctMIMOBreaks(t *testing.T) {
	plant, err := NewGain(mat.NewDense(2, 2, []float64{1, 2, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 3, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	loop, err := NewGeneralizedClosedLoop("mimo", plant, fixedBlockT(t, controller), "plant_output")
	if err != nil {
		t.Fatal(err)
	}
	if err := loop.InsertAnalysisPoint("plant_input", AnalysisPointPlantInput); err != nil {
		t.Fatalf("InsertAnalysisPoint: %v", err)
	}

	outputLoop, err := loop.OpenLoop("plant_output")
	if err != nil {
		t.Fatalf("output OpenLoop: %v", err)
	}
	inputLoop, err := loop.OpenLoop("plant_input")
	if err != nil {
		t.Fatalf("input OpenLoop: %v", err)
	}
	wantOutput := mat.NewDense(2, 2, []float64{7, 2, 3, 1})
	wantInput := mat.NewDense(2, 2, []float64{1, 2, 3, 7})
	if !mat.EqualApprox(outputLoop.D, wantOutput, 1e-14) {
		t.Fatalf("plant-output loop =\n%v\nwant\n%v", mat.Formatted(outputLoop.D), mat.Formatted(wantOutput))
	}
	if !mat.EqualApprox(inputLoop.D, wantInput, 1e-14) {
		t.Fatalf("plant-input loop =\n%v\nwant\n%v", mat.Formatted(inputLoop.D), mat.Formatted(wantInput))
	}

	for _, name := range []string{"plant_output", "plant_input"} {
		sensitivity, err := loop.Sensitivity(name)
		if err != nil {
			t.Fatalf("Sensitivity(%q): %v", name, err)
		}
		complementary, err := loop.ComplementarySensitivity(name)
		if err != nil {
			t.Fatalf("ComplementarySensitivity(%q): %v", name, err)
		}
		sum := mat.NewDense(2, 2, nil)
		sum.Add(sensitivity.D, complementary.D)
		if !mat.EqualApprox(sum, eyeDense(2), 1e-12) {
			t.Fatalf("S+T at %q =\n%v\nwant identity", name, mat.Formatted(sum))
		}
	}
}

func TestGeneralizedClosedLoopRejectsInvalidAnalysisPointLocation(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	loop, err := NewGeneralizedClosedLoop("loop", plant, fixedBlockT(t, plant), "output")
	if err != nil {
		t.Fatal(err)
	}
	if err := loop.InsertAnalysisPoint("bad", AnalysisPointUnspecified); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("InsertAnalysisPoint error = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneralizedModelCurrentSystemRejectsNameCountMismatch(t *testing.T) {
	gm, err := NewGeneralizedModel("g", fixedBlockT(t, makeSISO(-1, 1, 1, 0)))
	if err != nil {
		t.Fatal(err)
	}
	gm.SetInputName("a", "b", "c")
	if _, err := gm.CurrentSystem(); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("input names err = %v, want ErrDimensionMismatch", err)
	}
	gm.SetInputName("a")
	gm.SetOutputName("y1", "y2")
	if _, err := gm.CurrentSystem(); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("output names err = %v, want ErrDimensionMismatch", err)
	}
}

func TestGeneralizedModelInsertAnalysisPointRejects(t *testing.T) {
	var nilModel *GeneralizedModel
	if err := nilModel.InsertAnalysisPoint("u"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil receiver err = %v, want ErrInvalidArgument", err)
	}
	gm, err := NewGeneralizedModel("g", fixedBlockT(t, makeSISO(-1, 1, 1, 0)))
	if err != nil {
		t.Fatal(err)
	}
	if err := gm.InsertAnalysisPoint(""); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty name err = %v, want ErrInvalidArgument", err)
	}
}

func TestGeneralizedBlocksDoNotAliasCallerSystem(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	ctrl := makeSISO(-2, 1, 1, 0)
	cl, err := NewGeneralizedClosedLoop("cl", plant, fixedBlockT(t, ctrl), "y")
	if err != nil {
		t.Fatal(err)
	}
	gm, err := NewGeneralizedModel("g", fixedBlockT(t, ctrl))
	if err != nil {
		t.Fatal(err)
	}
	ctrl.A.Set(0, 0, -50)
	ol, err := cl.OpenLoop("y")
	if err != nil {
		t.Fatal(err)
	}
	if got := ol.A.At(0, 0); got != -2 {
		t.Fatalf("open-loop controller pole = %g, want -2 (caller mutation leaked)", got)
	}
	cs, err := gm.CurrentSystem()
	if err != nil {
		t.Fatal(err)
	}
	if got := cs.A.At(0, 0); got != -2 {
		t.Fatalf("model A = %g, want -2 (caller mutation leaked)", got)
	}
}

func TestGeneralizedConstructorsRejectNil(t *testing.T) {
	var nilSys *System
	if _, err := FixedBlock(nilSys); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("FixedBlock err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewGeneralizedModel("g", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NewGeneralizedModel err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewGeneralizedClosedLoop("cl", makeSISO(-1, 1, 1, 0), nil, "y"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NewGeneralizedClosedLoop err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewGeneralizedClosedLoop("cl", nilSys, fixedBlockT(t, makeSISO(-1, 1, 1, 0)), "y"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NewGeneralizedClosedLoop nil plant err = %v, want ErrInvalidArgument", err)
	}
}

func fixedBlockT(t testing.TB, sys *System) NumericBlock {
	t.Helper()
	b, err := FixedBlock(sys)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestGeneralizedModelNilReceiverSetters(t *testing.T) {
	var g *GeneralizedModel
	if err := g.SetInputName("u"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("SetInputName err = %v, want ErrInvalidArgument", err)
	}
	if err := g.SetOutputName("y"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("SetOutputName err = %v, want ErrInvalidArgument", err)
	}
}
