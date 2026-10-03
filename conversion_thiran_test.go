package controlsys

import (
	"math"
	"math/cmplx"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func thiranPublishedResponse(z complex128) complex128 {
	return (((complex(7./1683, 0)*z-complex(9./187, 0))*z+complex(9./17, 0))*z + 1) / (((z+complex(9./17, 0))*z-complex(9./187, 0))*z + complex(7./1683, 0))
}

func TestConversionThiranPublishedCoefficientsAndMaximumOrder(t *testing.T) {
	for _, method := range []C2DMethod{C2DMethodTustin, C2DMethodMatched} {
		for _, tau := range []float64{.02, .24, .54, .2} {
			for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
				sys, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
				sys.InputDelay = []float64{tau}
				original := sys.Copy()
				disc, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: method, ThiranOrder: 3, DelayModeling: modeling})
				if err != nil {
					t.Fatalf("%s delay%g %s: %v", method, tau, modeling, err)
				}
				states, _, _ := disc.Dims()
				order := 3
				if tau == .02 {
					order = 1
				}
				if tau == .2 {
					order = 0
				}
				wantStates := 1
				if modeling == C2DDelayModelingState {
					wantStates += order
				}
				if states != wantStates {
					t.Fatalf("%s delay%g %s states%d want%d", method, tau, modeling, states, wantStates)
				}
				if tau == .54 && (len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3) {
					t.Fatalf("integer remainder: %v", disc.InputDelay)
				}
				for _, theta := range []float64{.03, .3, .9, 1.7, 2.6} {
					z := cmplx.Exp(complex(0, theta))
					delay := thiranPublishedResponse(z)
					switch tau {
					case .02:
						delay = (complex(2./3, 0)*z + 1) / (z + complex(2./3, 0))
					case .54:
						delay *= cmplx.Pow(z, -3)
					case .2:
						delay = cmplx.Pow(z, -2)
					}
					rational := 1 / (complex(20, 0)*(z-1)/(z+1) + 1)
					if method == C2DMethodMatched {
						rational = complex(1-math.Exp(-.1), 0) / (z - complex(math.Exp(-.1), 0))
					}
					assertPrewarpResponse(t, disc, z, [][]complex128{{rational * delay}})
				}
				if !reflect.DeepEqual(sys.InputDelay, original.InputDelay) || !mat.Equal(sys.A, original.A) {
					t.Fatal("mutated source")
				}
			}
		}
	}
}

func thiranMIMOFixture(t *testing.T) *System {
	t.Helper()
	sys, err := New(mat.NewDense(2, 2, []float64{-1, .3, -.2, -2}), mat.NewDense(2, 2, []float64{1, .2, -.1, .8}), mat.NewDense(2, 2, []float64{1, .5, -.4, .9}), mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	err = sys.SetInternalDelay([]float64{.024, .54}, mat.NewDense(2, 2, []float64{.2, .5, -.3, .1}), mat.NewDense(2, 2, []float64{.4, .2, -.1, .3}), mat.NewDense(2, 2, []float64{.1, .2, .3, -.2}), mat.NewDense(2, 2, []float64{.3, .5, .1, -.1}), mat.NewDense(2, 2, []float64{.01, .04, .02, .03}))
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{.02, .2}
	sys.OutputDelay = []float64{.24, 0}
	sys.InputName = []string{"u1", "u2"}
	sys.OutputName = []string{"y1", "y2"}
	sys.StateName = []string{"x1", "x2"}
	return sys
}

func thiranAugmentedReference(s complex128) [4][4]complex128 {
	b := [2][4]float64{{1, .2, .2, .5}, {-.1, .8, -.3, .1}}
	c := [4][2]float64{{1, .5}, {-.4, .9}, {.4, .2}, {-.1, .3}}
	d := [4][4]float64{{0, 0, .1, .2}, {0, 0, .3, -.2}, {.3, .5, .01, .04}, {.1, -.1, .02, .03}}
	determinant := (s+1)*(s+2) + .06
	inverse := [2][2]complex128{{(s + 2) / determinant, .3 / determinant}, {-.2 / determinant, (s + 1) / determinant}}
	var h [4][4]complex128
	for i := range 4 {
		for j := range 4 {
			h[i][j] = complex(d[i][j], 0)
			for k := range 2 {
				for l := range 2 {
					h[i][j] += complex(c[i][k]*b[l][j], 0) * inverse[k][l]
				}
			}
		}
	}
	return h
}

func TestInternalThiranIndependentMIMOResponseAndRepresentations(t *testing.T) {
	for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
		sys := thiranMIMOFixture(t)
		original := sys.Copy()
		disc, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, PrewarpFrequency: 4, DelayModeling: modeling})
		if err != nil {
			t.Fatal(err)
		}
		n, _, _ := disc.Dims()
		if modeling == C2DDelayModelingInternal && n != 2 {
			t.Fatalf("delay mode states%d", n)
		}
		if modeling == C2DDelayModelingState && (n != 10 || disc.LFT == nil || len(disc.LFT.Tau) != 1 || disc.LFT.Tau[0] != 3) {
			t.Fatalf("state mode needs 8 filter states plus n2 and original integer remainder 3: n%d LFT%v", n, disc.LFT)
		}
		if !reflect.DeepEqual(disc.InputName, sys.InputName) || !reflect.DeepEqual(disc.OutputName, sys.OutputName) {
			t.Fatal("lost channel names")
		}
		beta := 4 / math.Tan(.2)
		for _, theta := range []float64{.02, .3, .75, 1.5, 2.5} {
			z := cmplx.Exp(complex(0, theta))
			h := thiranAugmentedReference(complex(beta, 0) * (z - 1) / (z + 1))
			delta := [2]complex128{(complex(19./31, 0)*z + 1) / (z + complex(19./31, 0)), thiranPublishedResponse(z) * cmplx.Pow(z, -3)}
			a, b, c, d := 1-h[2][2]*delta[0], -h[2][3]*delta[1], -h[3][2]*delta[0], 1-h[3][3]*delta[1]
			determinant := a*d - b*c
			inputs := [2]complex128{(complex(2./3, 0)*z + 1) / (z + complex(2./3, 0)), cmplx.Pow(z, -2)}
			outputs := [2]complex128{thiranPublishedResponse(z), 1}
			want := make([][]complex128, 2)
			for i := range 2 {
				want[i] = make([]complex128, 2)
				for j := range 2 {
					q0, q1 := (d*h[2][j]-b*h[3][j])/determinant, (-c*h[2][j]+a*h[3][j])/determinant
					want[i][j] = (h[i][j] + h[i][2]*delta[0]*q0 + h[i][3]*delta[1]*q1) * inputs[j] * outputs[i]
				}
			}
			assertPrewarpResponse(t, disc, z, want)
		}
		if !reflect.DeepEqual(sys.LFT.Tau, original.LFT.Tau) || !mat.Equal(sys.A, original.A) || !mat.Equal(sys.LFT.D22, original.LFT.D22) {
			t.Fatal("mutated internal source")
		}
	}
}

func TestInternalThiranTinyDelayAndSingularInstantaneousLoop(t *testing.T) {
	scalar := func(value float64) *mat.Dense { return mat.NewDense(1, 1, []float64{value}) }
	for _, tau := range []float64{.002, .2, .24, .54} {
		sys, _ := New(scalar(-1), scalar(1), scalar(1), scalar(0), 0)
		if err := sys.SetInternalDelay([]float64{tau}, scalar(.2), scalar(1), scalar(0), scalar(0), scalar(0)); err != nil {
			t.Fatal(err)
		}
		for _, mode := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: mode}); err != nil {
				t.Fatal(err)
			}
		}
	}
	sys, _ := New(scalar(-1), scalar(1), scalar(1), scalar(0), 0)
	if err := sys.SetInternalDelay([]float64{.02}, scalar(0), scalar(0), scalar(0), scalar(0), scalar(1.5)); err != nil {
		t.Fatal(err)
	}
	if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err == nil {
		t.Fatal("accepted singular instantaneous feedback")
	}
}

func BenchmarkConversionThiran(b *testing.B) {
	b.Run("rounded", func(b *testing.B) {
		sys := thiranMIMOFixtureBenchmark()
		b.ReportAllocs()
		for b.Loop() {
			if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin}); err != nil {
				b.Fatal(err)
			}
		}
	})
	for _, mode := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
		b.Run(string(mode), func(b *testing.B) {
			sys := thiranMIMOFixtureBenchmark()
			b.ReportAllocs()
			for b.Loop() {
				if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: mode}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
func thiranMIMOFixtureBenchmark() *System {
	sys, _ := New(mat.NewDense(2, 2, []float64{-1, .3, -.2, -2}), mat.NewDense(2, 2, []float64{1, .2, -.1, .8}), mat.NewDense(2, 2, []float64{1, .5, -.4, .9}), mat.NewDense(2, 2, nil), 0)
	_ = sys.SetInternalDelay([]float64{.024, .54}, mat.NewDense(2, 2, []float64{.2, .5, -.3, .1}), mat.NewDense(2, 2, []float64{.4, .2, -.1, .3}), mat.NewDense(2, 2, []float64{.1, .2, .3, -.2}), mat.NewDense(2, 2, []float64{.3, .5, .1, -.1}), mat.NewDense(2, 2, []float64{.01, .04, .02, .03}))
	return sys
}
