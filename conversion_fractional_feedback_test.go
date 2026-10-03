package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func fractionalFeedbackFixture(t *testing.T, tau []float64) *System {
	t.Helper()
	sys := prewarpModel(t)
	err := sys.SetInternalDelay(tau, mat.NewDense(2, 2, []float64{.2, .1, -.1, .15}), mat.NewDense(2, 2, []float64{.3, -.1, .1, .2}), mat.NewDense(2, 2, []float64{.2, .1, -.1, .1}), mat.NewDense(2, 2, []float64{.1, .2, -.1, .1}), mat.NewDense(2, 2, []float64{.1, .05, -.04, .12}))
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"u", "v"}
	sys.OutputName = []string{"y", "z"}
	sys.StateName = []string{"a", "b"}
	return sys
}

func holdReferenceInput(t, dt, theta float64, foh bool) complex128 {
	sample := t / dt
	k := math.Floor(sample + 1e-12)
	a := sample - k
	if !foh {
		return cmplx.Exp(complex(0, k*theta))
	}
	return complex(1-a, 0)*cmplx.Exp(complex(0, k*theta)) + complex(a, 0)*cmplx.Exp(complex(0, (k+1)*theta))
}

// Analytic impulse kernel integrated against sampled interpolation, independent
// of matrix exponentials, conversion kernels, or the implementation realization.
func holdReferenceChannel(b0, b1, c0, c1, d, tau, dt, theta float64, foh bool) complex128 {
	value := complex(d, 0) * holdReferenceInput(-tau, dt, theta, foh)
	for start := 0.; start < 40; {
		k := math.Floor((-tau-start)/dt + 1e-10)
		end := -tau - k*dt
		if end <= start+1e-10 {
			end += dt
		}
		if end > 40 {
			end = 40
		}
		const parts = 64
		step := (end - start) / parts
		held := holdReferenceInput(-tau-(start+end)/2, dt, theta, false)
		sample := func(r float64) complex128 {
			e1, e3 := math.Exp(-r), math.Exp(-3*r)
			impulse := c0*(e1*b0+(e1-e3)*b1) + c1*e3*b1
			input := held
			if foh {
				input = holdReferenceInput(-tau-r, dt, theta, true)
			}
			return complex(impulse, 0) * input
		}
		integral := sample(start) + sample(end)
		for i := 1; i < parts; i++ {
			weight := 2.
			if i%2 == 1 {
				weight = 4
			}
			integral += complex(weight, 0) * sample(start+float64(i)*step)
		}
		value += complex(step/3, 0) * integral
		start = end
	}
	return value
}

func holdFeedbackReference(sys *System, dt, theta float64, foh bool) [][]complex128 {
	_, m, p := sys.Dims()
	count := len(sys.LFT.Tau)
	b := mat.NewDense(2, m+count, nil)
	c := mat.NewDense(p+count, 2, nil)
	d := mat.NewDense(p+count, m+count, nil)
	setBlock(b, 0, 0, sys.B)
	setBlock(b, 0, m, sys.LFT.B2)
	setBlock(c, 0, 0, sys.C)
	setBlock(c, p, 0, sys.LFT.C2)
	setBlock(d, 0, 0, sys.D)
	setBlock(d, 0, m, sys.LFT.D12)
	setBlock(d, p, 0, sys.LFT.D21)
	setBlock(d, p, m, sys.LFT.D22)
	input, output := make([]float64, m+count), make([]float64, p+count)
	copy(input, sys.InputDelay)
	copy(output, sys.OutputDelay)
	whole := make([]float64, count)
	for k, tau := range sys.LFT.Tau {
		samples := tau / dt
		if isIntegerSampleDelay(samples) {
			whole[k] = math.Round(samples)
		} else {
			whole[k] = math.Floor(samples)
			input[m+k] = tau - whole[k]*dt
		}
	}
	h := make([][]complex128, p+count)
	for i := range h {
		h[i] = make([]complex128, m+count)
		for j := range h[i] {
			tau := input[j] + output[i]
			if i < p && j < m && sys.Delay != nil {
				tau += sys.Delay.At(i, j)
			}
			h[i][j] = holdReferenceChannel(b.At(0, j), b.At(1, j), c.At(i, 0), c.At(i, 1), d.At(i, j), tau, dt, theta, foh)
		}
	}
	z := cmplx.Exp(complex(0, theta))
	k0, k1 := cmplx.Pow(z, -complex(whole[0], 0)), cmplx.Pow(z, -complex(whole[1], 0))
	a, bv, cv, dv := 1-h[p][m]*k0, -h[p][m+1]*k1, -h[p+1][m]*k0, 1-h[p+1][m+1]*k1
	determinant := a*dv - bv*cv
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		for j := range m {
			v0, v1 := (dv*h[p][j]-bv*h[p+1][j])/determinant, (-cv*h[p][j]+a*h[p+1][j])/determinant
			path := 0.
			if sys.Delay != nil {
				path = sys.Delay.At(i, j)
			}
			external := holdReferenceChannel(b.At(0, j), b.At(1, j), c.At(i, 0), c.At(i, 1), d.At(i, j), input[j]+output[i]+path, dt, theta, foh)
			loop0 := holdReferenceChannel(b.At(0, m), b.At(1, m), c.At(i, 0), c.At(i, 1), d.At(i, m), input[m]+output[i]+path, dt, theta, foh)
			loop1 := holdReferenceChannel(b.At(0, m+1), b.At(1, m+1), c.At(i, 0), c.At(i, 1), d.At(i, m+1), input[m+1]+output[i]+path, dt, theta, foh)
			result[i][j] = external + loop0*k0*v0 + loop1*k1*v1
		}
	}
	return result
}

func TestFractionalFeedbackHoldIndependentReference(t *testing.T) {
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		for _, tau := range [][]float64{{.15, .24}, {.04, .08}, {.2, .3}} {
			for _, external := range []string{"none", "fractional", "integer-output"} {
				source := fractionalFeedbackFixture(t, tau)
				if external != "none" {
					source.InputDelay = []float64{.035, .17}
					source.OutputDelay = []float64{.015, .13}
					source.Delay = mat.NewDense(2, 2, []float64{0, .045, .025, 0})
					if external == "integer-output" {
						source.OutputDelay = []float64{.1, .2}
						source.Delay = nil
					}
				}
				before := source.Copy()
				out, err := source.DiscretizeWithResult(.1, C2DOptions{Method: method})
				if err != nil {
					t.Fatalf("method%s tau%v external%v: %v", method, tau, external, err)
				}
				if !out.Approximate {
					t.Fatal("internal hold approximation not reported")
				}
				if external != "none" || !isIntegerSampleDelay(tau[0]/.1) {
					if out.InitialStateMap != nil {
						t.Fatal("fractional histories incorrectly claim state mapping")
					}
				}
				if !reflect.DeepEqual(out.System.InputName, source.InputName) || !reflect.DeepEqual(out.System.OutputName, source.OutputName) {
					t.Fatal("names changed")
				}
				if out.System.LFT != nil {
					for _, value := range out.System.LFT.Tau {
						if value <= 0 || value != math.Round(value) {
							t.Fatalf("noninteger internal samples %v", out.System.LFT.Tau)
						}
					}
				}
				state, err := source.DiscretizeWithOpts(.1, C2DOptions{Method: method, DelayModeling: C2DDelayModelingState})
				if err != nil {
					t.Fatal(err)
				}
				for _, theta := range []float64{.08, .7, 1.8} {
					want := holdFeedbackReference(source, .1, theta, method == C2DMethodFOH)
					got, err := out.System.EvalFr(cmplx.Exp(complex(0, theta)))
					if err != nil {
						t.Fatal(err)
					}
					stateGot, err := state.EvalFr(cmplx.Exp(complex(0, theta)))
					if err != nil {
						t.Fatal(err)
					}
					for i := range want {
						for j := range want[i] {
							if cmplx.Abs(got[i][j]-want[i][j]) > 3e-9*(1+cmplx.Abs(want[i][j])) {
								t.Fatalf("method%s tau%v external%v theta%g H[%d,%d] got%v want%v", method, tau, external, theta, i, j, got[i][j], want[i][j])
							}
							if cmplx.Abs(got[i][j]-stateGot[i][j]) > 1e-10 {
								t.Fatal("delay/state mismatch")
							}
						}
					}
				}
				assertMatClose(t, "immutableA", source.A, before.A, 0)
				assertMatClose(t, "immutableB2", source.LFT.B2, before.LFT.B2, 0)
				assertMatClose(t, "immutableD22", source.LFT.D22, before.LFT.D22, 0)
			}
		}
	}
}

func TestFractionalFeedbackHoldPureGain(t *testing.T) {
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		sys, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
		if err := sys.SetInternalDelay([]float64{.04}, newDense(0, 1), newDense(1, 0), mat.NewDense(1, 1, []float64{.3}), mat.NewDense(1, 1, []float64{.4}), mat.NewDense(1, 1, []float64{.2})); err != nil {
			t.Fatal(err)
		}
		out, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		for _, theta := range []float64{.1, .8, 2.1} {
			z := cmplx.Exp(complex(0, theta))
			shift := 1 / z
			if method == C2DMethodFOH {
				shift = .6 + .4/z
			}
			want := 2 + .12*shift/(1-.2*shift)
			assertPrewarpResponse(t, out, z, [][]complex128{{want}})
		}
	}
}

func TestFractionalFeedbackHoldInvalidDelay(t *testing.T) {
	for _, value := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		sys := fractionalFeedbackFixture(t, []float64{.15, .24})
		sys.LFT.Tau[0] = value
		if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{}); !errors.Is(err, ErrZeroInternalDelay) {
			t.Fatalf("tau%g err%v", value, err)
		}
	}
}

func BenchmarkFractionalFeedbackHold(b *testing.B) {
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		for _, mode := range []string{"integer", "fractional", "integer-output", "mixed-output"} {
			b.Run(string(method)+"/"+mode, func(b *testing.B) {
				sys, _ := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, .5}), mat.NewDense(1, 2, []float64{1, -.2}), mat.NewDense(1, 1, nil), 0)
				tau := []float64{.15, .24}
				if mode == "integer" {
					tau = []float64{.2, .3}
				}
				sys.SetInternalDelay(tau, mat.NewDense(2, 2, []float64{.2, .1, -.1, .15}), mat.NewDense(2, 2, []float64{.3, -.1, .1, .2}), mat.NewDense(1, 2, []float64{.2, .1}), mat.NewDense(2, 1, []float64{.1, .2}), mat.NewDense(2, 2, []float64{.1, .05, -.04, .12}))
				if mode == "integer-output" || mode == "mixed-output" {
					sys.InputDelay = []float64{.035}
					sys.OutputDelay = []float64{.2}
					if mode == "mixed-output" {
						sys.OutputDelay = []float64{.035}
					}
				}
				b.ReportAllocs()
				for range b.N {
					if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: method}); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestFractionalFeedbackHoldIndependentSampledTime(t *testing.T) {
	const dt = .1
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		for _, tau := range []float64{.04, .15, .24, .2} {
			for _, format := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
				sys, _ := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
				sys.SetInternalDelay([]float64{tau}, mat.NewDense(1, 1, []float64{.2}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil))
				out, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: method, DelayModeling: format})
				if err != nil {
					t.Fatal(err)
				}
				u := mat.NewDense(1, 30, nil)
				for k := 1; k < 30; k++ {
					u.Set(0, k, math.Sin(float64(k)*.7)+.2)
				}
				actual, err := out.Simulate(u, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				states := make([]float64, 30)
				for k := range 29 {
					sampled := actual.Y.At(0, k)
					if math.Abs(sampled-states[k]) > 2e-10 {
						t.Fatalf("method%s tau%g format%s sample%d got%g want%g", method, tau, format, k, sampled, states[k])
					}
					known, coefficient := 0., 0.
					rho := math.Mod(tau, dt)
					bounds := []float64{0, dt}
					if rho > 1e-10 && rho < dt-1e-10 {
						bounds = []float64{0, rho, dt}
					}
					value := func(time, midpoint float64) (float64, float64) {
						input := u.At(0, k)
						if method == C2DMethodFOH {
							input += (u.At(0, k+1) - input) * time / dt
						}
						index := math.Floor((float64(k)*dt+midpoint-tau)/dt + 1e-10)
						weight := (float64(k)*dt+time-tau)/dt - index
						past, next := int(index), int(index)+1
						feedback, future := 0., 0.
						if method == C2DMethodZOH {
							if past >= 0 {
								feedback = states[past]
							}
						} else {
							if past >= 0 {
								feedback = (1 - weight) * states[past]
							}
							if next >= 0 {
								if next == k+1 {
									future = weight
								} else {
									feedback += weight * states[next]
								}
							}
						}
						kernel := math.Exp(-(dt - time))
						return kernel * (input + .2*feedback), kernel * .2 * future
					}
					for q := 0; q < len(bounds)-1; q++ {
						start, end := bounds[q], bounds[q+1]
						const parts = 128
						step := (end - start) / parts
						for j := 0; j <= parts; j++ {
							weight := 2.
							if j == 0 || j == parts {
								weight = 1
							} else if j%2 == 1 {
								weight = 4
							}
							v, c := value(start+float64(j)*step, (start+end)/2)
							known += step * weight * v / 3
							coefficient += step * weight * c / 3
						}
					}
					states[k+1] = (math.Exp(-dt)*states[k] + known) / (1 - coefficient)
				}
			}
		}
	}
}
