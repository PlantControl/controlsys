package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"reflect"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func thiranOracle(tau, dt float64, maxOrder int, z complex128) complex128 {
	samples := tau / dt
	if isIntegerSampleDelay(samples) {
		return cmplx.Pow(z, complex(-math.Round(samples), 0))
	}
	ceiling := math.Ceil(samples)
	order := min(int(ceiling), maxOrder)
	whole := ceiling - float64(order)
	a := thiranCoeffs(samples-whole, order)
	var num, den complex128
	for k, value := range a {
		num += complex(value, 0) * cmplx.Pow(z, complex(-float64(order-k), 0))
		den += complex(value, 0) * cmplx.Pow(z, complex(-float64(k), 0))
	}
	return cmplx.Pow(z, complex(-whole, 0)) * num / den
}

// pathThiranOracle evaluates each discretized channel independently: the
// Tustin image of G at e^{jωdt} is G(j(2/dt)tan(ωdt/2)), and every delay
// segment of the input/output/residual split is an explicit Thiran product.
func pathThiranOracle(t *testing.T, sys *System, dt float64, maxOrder int, omega []float64) [][][]complex128 {
	t.Helper()
	plain := sys.Copy()
	plain.Delay, plain.InputDelay, plain.OutputDelay = nil, nil, nil
	warped := make([]float64, len(omega))
	for k, w := range omega {
		warped[k] = 2 / dt * math.Tan(w*dt/2)
	}
	resp, err := plain.FreqResponse(warped)
	if err != nil {
		t.Fatal(err)
	}
	input, output, residual := decomposeIODelay(sys.Delay)
	_, m, p := sys.Dims()
	out := make([][][]complex128, len(omega))
	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*dt))
		out[k] = make([][]complex128, p)
		for i := range p {
			out[k][i] = make([]complex128, m)
			for j := range m {
				h := resp.At(k, i, j)
				for _, tau := range []float64{input[j], residual.At(i, j), output[i]} {
					h *= thiranOracle(tau, dt, maxOrder, z)
				}
				out[k][i][j] = h
			}
		}
	}
	return out
}

func nondecomposablePathSystem(t *testing.T, p, m int, delay []float64) *System {
	t.Helper()
	a := mat.NewDense(2, 2, []float64{-1, 0.3, 0, -2})
	b := mat.NewDense(2, m, nil)
	c := mat.NewDense(p, 2, nil)
	d := mat.NewDense(p, m, nil)
	for j := range m {
		b.Set(0, j, 1+0.4*float64(j))
		b.Set(1, j, 0.2-0.5*float64(j))
	}
	for i := range p {
		c.Set(i, 0, 1-0.3*float64(i))
		c.Set(i, 1, 0.5+float64(i))
		for j := range m {
			d.Set(i, j, 0.1*float64(i-j))
		}
	}
	sys, err := New(a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.Delay = mat.NewDense(p, m, delay)
	for i := range m {
		sys.InputName = append(sys.InputName, "u"+string(rune('a'+i)))
	}
	for i := range p {
		sys.OutputName = append(sys.OutputName, "y"+string(rune('a'+i)))
	}
	return sys
}

func TestTustinThiranNondecomposablePathDelays(t *testing.T) {
	omega := []float64{0.2, 3, 11, 25}
	dt := 0.1
	for _, tc := range []struct {
		name   string
		p, m   int
		delay  []float64
		order  int
		states map[C2DDelayModeling]int
	}{
		{"issue-2x2", 2, 2, []float64{.01, .02, .03, .07}, 3, map[C2DDelayModeling]int{C2DDelayModelingInternal: 4, C2DDelayModelingState: 8}},
		{"column-grouping-3x2", 3, 2, []float64{.03, 0, .01, 0, .05, 0}, 3, map[C2DDelayModeling]int{C2DDelayModelingInternal: 4, C2DDelayModelingState: 7}},
		{"row-grouping-2x3", 2, 3, []float64{0, 0, 0, .03, 0, .05}, 3, map[C2DDelayModeling]int{C2DDelayModelingInternal: 4, C2DDelayModelingState: 6}},
		{"integer-remainder", 2, 2, []float64{0, 0, 0, .37}, 2, map[C2DDelayModeling]int{C2DDelayModelingInternal: 4, C2DDelayModelingState: 6}},
	} {
		for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			t.Run(tc.name+"/"+string(modeling), func(t *testing.T) {
				sys := nondecomposablePathSystem(t, tc.p, tc.m, tc.delay)
				source := sys.Copy()
				out, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: tc.order, DelayModeling: modeling})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(sys, source) {
					t.Fatal("conversion mutated the source system")
				}
				if n, _, _ := out.Dims(); n != tc.states[modeling] {
					t.Errorf("states = %d, want %d", n, tc.states[modeling])
				}
				if modeling == C2DDelayModelingState && out.HasInternalDelay() {
					t.Errorf("state modeling kept %d internal delays", out.internalDelayCount())
				}
				if !slices.Equal(out.InputName, sys.InputName) || !slices.Equal(out.OutputName, sys.OutputName) {
					t.Errorf("names = %v %v", out.InputName, out.OutputName)
				}
				want := pathThiranOracle(t, sys, dt, tc.order, omega)
				got, err := out.FreqResponse(omega)
				if err != nil {
					t.Fatal(err)
				}
				for k := range omega {
					for i := range tc.p {
						for j := range tc.m {
							if diff := cmplx.Abs(got.At(k, i, j) - want[k][i][j]); diff > 1e-10 {
								t.Fatalf("w=%g H(%d,%d) = %v, oracle %v", omega[k], i, j, got.At(k, i, j), want[k][i][j])
							}
						}
					}
				}
				continuous, err := sys.FreqResponse(omega[:1])
				if err != nil {
					t.Fatal(err)
				}
				for i := range tc.p {
					for j := range tc.m {
						if diff := cmplx.Abs(got.At(0, i, j) - continuous.At(0, i, j)); diff > 5e-3 {
							t.Errorf("low-frequency H(%d,%d) = %v, continuous delayed %v", i, j, got.At(0, i, j), continuous.At(0, i, j))
						}
					}
				}
			})
		}
	}
}

func TestTustinThiranPathDelayIntegerRemainderStaysDelay(t *testing.T) {
	sys := nondecomposablePathSystem(t, 2, 2, []float64{0, 0, 0, .37})
	out, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 2, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if out.Delay == nil || out.Delay.At(1, 1) != 2 || out.Delay.At(0, 0) != 0 {
		t.Fatalf("discrete path delay = %v, want integer remainder 2 samples on (1,1)", out.Delay)
	}
}

func TestTustinThiranDecomposablePathDelayUnchanged(t *testing.T) {
	sys := nondecomposablePathSystem(t, 2, 2, []float64{.01, .03, .02, .04})
	out, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := out.Dims(); n != 2+3 || out.Delay != nil {
		t.Fatalf("decomposable delays: states %d delay %v, want 2 plant + 3 filter states and no path delay", n, out.Delay)
	}
}

func TestTustinThiranPathDelaysWithInternalFeedback(t *testing.T) {
	dt := 0.1
	omega := []float64{0.2, 3, 11}
	for _, tc := range []struct {
		p, m  int
		delay []float64
	}{{2, 2, []float64{.01, .02, .03, .07}}, {3, 2, []float64{.03, 0, .01, 0, .05, 0}}} {
		sys := nondecomposablePathSystem(t, tc.p, tc.m, tc.delay)
		d12, d21 := mat.NewDense(tc.p, 1, nil), mat.NewDense(1, tc.m, nil)
		d12.Set(0, 0, 0.1)
		d21.Set(0, tc.m-1, 0.2)
		if err := sys.SetInternalDelay([]float64{0.25},
			mat.NewDense(2, 1, []float64{0.3, 0.1}), mat.NewDense(1, 2, []float64{0.2, -0.4}),
			d12, d21, mat.NewDense(1, 1, []float64{0.1}),
		); err != nil {
			t.Fatal(err)
		}
		opts := C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}
		out, err := sys.C2D(dt, opts)
		if err != nil {
			t.Fatal(err)
		}
		plain := sys.Copy()
		plain.Delay = nil
		reference, err := plain.C2D(dt, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := out.FreqResponse(omega)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := reference.FreqResponse(omega)
		if err != nil {
			t.Fatal(err)
		}
		input, output, residual := decomposeIODelay(sys.Delay)
		for k, w := range omega {
			z := cmplx.Exp(complex(0, w*dt))
			for i := range tc.p {
				for j := range tc.m {
					want := ref.At(k, i, j)
					for _, tau := range []float64{input[j], residual.At(i, j), output[i]} {
						want *= thiranOracle(tau, dt, 3, z)
					}
					if diff := cmplx.Abs(got.At(k, i, j) - want); diff > 1e-10 {
						t.Fatalf("%dx%d w=%g H(%d,%d) = %v, want %v", tc.p, tc.m, w, i, j, got.At(k, i, j), want)
					}
				}
			}
		}
	}
}

func TestTustinThiranPathDelayResultIsApproximate(t *testing.T) {
	sys := nondecomposablePathSystem(t, 2, 2, []float64{.01, .02, .03, .07})
	opts := C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}
	if _, err := sys.C2D(0.1, opts); err != nil {
		t.Fatal(err)
	}
	if _, _, err := sys.C2DMap(0.1, opts); !errors.Is(err, ErrOptionUnsupported) {
		t.Fatalf("duplicated path states: C2DMap err = %v, want ErrOptionUnsupported", err)
	}
}

func TestTustinThiranPathDelayStaticGain(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 3, []float64{1, -2, 0.5, 3, 0.25, -1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.Delay = mat.NewDense(2, 3, []float64{0, 0, 0, .03, 0, .15})
	omega := []float64{0.2, 7, 21}
	for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
		out, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling})
		if err != nil {
			t.Fatal(err)
		}
		got, err := out.FreqResponse(omega)
		if err != nil {
			t.Fatal(err)
		}
		want := pathThiranOracle(t, sys, 0.1, 3, omega)
		for k := range omega {
			for i := range 2 {
				for j := range 3 {
					if diff := cmplx.Abs(got.At(k, i, j) - want[k][i][j]); diff > 1e-12 {
						t.Fatalf("%s w=%g H(%d,%d) = %v, oracle %v", modeling, omega[k], i, j, got.At(k, i, j), want[k][i][j])
					}
				}
			}
		}
	}
}
