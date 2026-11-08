package controlsys_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

// A nil error means the result is certified. ErrAccuracyUnverified comes
// with the full result, which a caller may keep as a warning; any other
// error comes with a nil result.
func ExampleCare() {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := controlsys.Care(A, B, Q, R, nil)
	switch {
	case err == nil:
		fmt.Println("certified")
	case errors.Is(err, controlsys.ErrAccuracyUnverified):
		fmt.Println("warning:", err)
	default:
		log.Fatal(err)
	}
	fmt.Printf("K = [%.4f %.4f]\n", res.K.At(0, 0), res.K.At(0, 1))

	// Output:
	// certified
	// K = [0.2361 0.2361]
}

func ExampleSystem_C2DFit() {
	sys, _ := controlsys.New(
		mat.NewDense(2, 2, []float64{0.5, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, nil), 0)

	sd, fit, err := sys.C2DFit(0.1, controlsys.C2DOptions{Method: controlsys.C2DMethodLeastSquares})
	if err != nil && !errors.Is(err, controlsys.ErrAccuracyUnverified) {
		log.Fatal(err)
	}
	fmt.Println("certified:", err == nil)
	fmt.Println("dt:", sd.Dt, "order:", fit.FitOrder)

	// Output:
	// certified: true
	// dt: 0.1 order: 2
}

// An exact DAE has a singular E, so its finite pole count is not certified:
// the poles still come back, with ErrAccuracyUnverified.
func ExampleSystem_PolesDetail() {
	sys, _ := controlsys.NewDescriptor(
		mat.NewDense(2, 2, []float64{-1, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.5}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)

	poles, err := sys.PolesDetail()
	fmt.Println(poles, errors.Is(err, controlsys.ErrAccuracyUnverified))

	// Output:
	// [(-1+0i)] true
}

// TunePID returns an uncertified design in full with ErrAccuracyUnverified;
// Evidence says which check it lacks.
func ExampleTunePID() {
	tf := &controlsys.TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, 3, 2}}}
	ss, _ := tf.StateSpace()

	res, err := controlsys.TunePID(context.Background(), ss.Sys, controlsys.PidtunePI, controlsys.PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil && !errors.Is(err, controlsys.ErrAccuracyUnverified) {
		log.Fatal(err)
	}
	fmt.Println("certified:", err == nil)
	fmt.Println("stability:", res.Evidence.Stability)

	// Output:
	// certified: true
	// stability: closed-loop-poles
}
