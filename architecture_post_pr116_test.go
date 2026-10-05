package controlsys

import (
	"errors"
	"math/cmplx"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPostPR116CrossSeamConversionMetadataRationalAndFRD(t *testing.T) {
	cont, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, -3, -4}),
		mat.NewDense(2, 1, []float64{1, -2}),
		mat.NewDense(1, 2, []float64{0.5, -1.5}),
		mat.NewDense(1, 1, []float64{0.25}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	cont.InputDelay = []float64{0.2}
	cont.InputName = []string{"u"}
	cont.OutputName = []string{"y"}
	cont.StateName = []string{"x1", "x2"}

	disc, err := cont.DiscretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(disc.InputDelay, []float64{2}) {
		t.Fatalf("discrete InputDelay = %v, want [2]", disc.InputDelay)
	}
	if !reflect.DeepEqual(disc.InputName, []string{"u"}) || !reflect.DeepEqual(disc.OutputName, []string{"y"}) {
		t.Fatalf("discrete metadata InputName=%v OutputName=%v", disc.InputName, disc.OutputName)
	}

	back, err := disc.D2C(C2DMethodZOH)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back.InputDelay, []float64{0.2}) {
		t.Fatalf("continuous InputDelay = %v, want [0.2]", back.InputDelay)
	}

	tf, err := cont.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tf.TF.Delay, [][]float64{{0.2}}) {
		t.Fatalf("TF Delay = %v, want [[0.2]]", tf.TF.Delay)
	}
	if _, err := tf.TF.ZPK(); !errors.Is(err, ErrDelayNotRepresentable) {
		t.Fatalf("ZPK of delayed TF err = %v, want ErrDelayNotRepresentable", err)
	}
	tf.TF.Delay = nil
	zpk, err := tf.TF.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	roundtripTF, err := zpk.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	s := complex(0.3, 1.1)
	if cmplx.Abs(mustEval(t, tf.TF, s)[0][0]-mustEval(t, roundtripTF, s)[0][0]) > 1e-8 {
		t.Fatalf("TF/ZPK roundtrip mismatch: %v vs %v", mustEval(t, tf.TF, s)[0][0], mustEval(t, roundtripTF, s)[0][0])
	}

	omega := []float64{0.1, 0.5, 1.0}
	plantFRD, err := disc.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	controllerFRD, err := NewFRD([][][]complex128{
		{{0.5}},
		{{0.4 + 0.1i}},
		{{0.3 + 0.2i}},
	}, omega, disc.Dt)
	if err != nil {
		t.Fatal(err)
	}
	controllerFRD.InputName = []string{"y"}
	controllerFRD.OutputName = []string{"u"}

	closedFRD, err := FRDFeedback(plantFRD, controllerFRD, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(closedFRD.InputName, []string{"u"}) {
		t.Fatalf("closed FRD InputName = %v, want [u]", closedFRD.InputName)
	}
	if !reflect.DeepEqual(closedFRD.OutputName, []string{"y"}) {
		t.Fatalf("closed FRD OutputName = %v, want [y]", closedFRD.OutputName)
	}
}
