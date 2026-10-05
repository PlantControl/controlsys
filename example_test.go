package controlsys_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/controlsys"
	"plantcontrol.org/v1/gonum/mat"
)

func ExampleStep() {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := controlsys.New(A, B, C, D, 0)

	resp, _ := controlsys.Step(sys, 7.0)
	last := len(resp.T) - 1
	fmt.Printf("Steady-state: %.2f\n", resp.Y.At(0, last))

	// Output:
	// Steady-state: 1.00
}

func ExampleSystem_DCGain() {
	A := mat.NewDense(1, 1, []float64{-2})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := controlsys.New(A, B, C, D, 0)

	g, _ := sys.DCGain()
	fmt.Printf("DC gain: %.2f\n", g.At(0, 0))

	// Output:
	// DC gain: 0.50
}

func ExampleDamp() {
	wn := 10.0
	zeta := 0.3
	sigma := -zeta * wn
	A := mat.NewDense(2, 2, []float64{0, 1, -(wn * wn), 2 * sigma})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := controlsys.New(A, B, C, D, 0)

	info, _ := controlsys.Damp(sys)
	fmt.Printf("Wn:   %.1f rad/s\n", info[0].Wn)
	fmt.Printf("Zeta: %.1f\n", info[0].Zeta)

	// Output:
	// Wn:   10.0 rad/s
	// Zeta: 0.3
}

func ExampleNew() {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, _ := controlsys.New(A, B, C, D, 0)

	poles, _ := sys.Poles()
	fmt.Println("Poles:", poles)

	stable, _ := sys.IsStable()
	fmt.Println("Stable:", stable)

	// Output:
	// Poles: [(0+0i) (0+0i)]
	// Stable: false
}

func ExampleFeedback() {
	// Plant: 1/(s+1)
	Ap := mat.NewDense(1, 1, []float64{-1})
	Bp := mat.NewDense(1, 1, []float64{1})
	Cp := mat.NewDense(1, 1, []float64{1})
	Dp := mat.NewDense(1, 1, []float64{0})
	plant, _ := controlsys.New(Ap, Bp, Cp, Dp, 0)

	// Unity negative feedback: T = P/(1+P) = 1/(s+2)
	cl, _ := controlsys.Feedback(plant, nil, -1)

	poles, _ := cl.Poles()
	fmt.Printf("Closed-loop pole: %.1f\n", real(poles[0]))

	// Output:
	// Closed-loop pole: -2.0
}

func ExampleSystem_Discretize() {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := controlsys.New(A, B, C, D, 0)

	sd, _ := sys.Discretize(0.1)
	fmt.Printf("Discrete: dt=%.1f\n", sd.Dt)
	fmt.Printf("A[0,0]: %.4f\n", sd.A.At(0, 0))

	// Output:
	// Discrete: dt=0.1
	// A[0,0]: 0.9048
}

func ExampleSystem_FreqResponse() {
	// First-order lowpass: H(s) = 1/(s+1), |H(j*1)| = 1/sqrt(2)
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := controlsys.New(A, B, C, D, 0)

	omega := []float64{1.0}
	fr, _ := sys.FreqResponse(omega)
	mag := 20 * math.Log10(cmplx.Abs(fr.At(0, 0, 0)))
	fmt.Printf("Gain at w=1: %.2f dB\n", mag)
	fmt.Printf("Expected:    %.2f dB\n", 20*math.Log10(1/math.Sqrt(2)))

	// Output:
	// Gain at w=1: -3.01 dB
	// Expected:    -3.01 dB
}

func ExampleSystem_Simulate() {
	// Discrete integrator: y[k] = y[k-1] + u[k]
	A := mat.NewDense(1, 1, []float64{1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{1})
	sys, _ := controlsys.New(A, B, C, D, 0.1)

	// u is (nInputs x nSteps)
	u := mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	resp, _ := sys.Simulate(u, nil, nil)

	_, cols := resp.Y.Dims()
	for i := range cols {
		fmt.Printf("y[%d] = %.0f\n", i, resp.Y.At(0, i))
	}

	// Output:
	// y[0] = 1
	// y[1] = 2
	// y[2] = 3
	// y[3] = 4
	// y[4] = 5
}

func ExampleSystem_DefaultFrequencyGrid() {
	// H(s) = 1/(s+1): grid spans one decade either side of the pole at 1 rad/s
	sys, _ := controlsys.New(
		mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0,
	)

	omega, _ := sys.DefaultFrequencyGrid(5)
	bode, _ := sys.Bode(nil, 5)
	fmt.Printf("%.4g\n", omega)
	fmt.Println(slices.Equal(omega, bode.Omega))

	// Output:
	// [0.1 0.3162 1 3.162 10]
	// true
}
