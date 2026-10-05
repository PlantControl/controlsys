package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func delayTestInput() *mat.Dense {
	data := make([]float64, 2*28)
	for j := range 2 {
		for k := range 28 {
			data[j*28+k] = math.Sin(float64(k)*.7+float64(j)) + .3*float64(j+1)
		}
	}
	data[0] = 0
	data[28] = 0
	return mat.NewDense(2, 28, data)
}

func delayAnalyticImpulse(i, j int, t float64) float64 {
	b := [][]float64{{1, .5}, {2, -1}}
	c := [][]float64{{1, -.4}, {0, 2}}
	e1, e3 := math.Exp(-t), math.Exp(-3*t)
	return c[i][0]*(e1*b[0][j]+(e1-e3)*b[1][j]) + c[i][1]*e3*b[1][j]
}

func delayInputAt(u *mat.Dense, j int, t, dt float64, foh bool) float64 {
	if t < -1e-12 {
		return 0
	}
	if t < 0 {
		t = 0
	}
	_, steps := u.Dims()
	x := t / dt
	k := int(math.Floor(x + 1e-11))
	if k >= steps {
		k = steps - 1
	}
	if !foh || k >= steps-1 {
		return u.At(j, k)
	}
	return u.At(j, k) + (x-float64(k))*(u.At(j, k+1)-u.At(j, k))
}

func delayAnalyticSample(u *mat.Dense, i, k int, dt float64, tau []float64, foh bool) float64 {
	d := [][]float64{{.2, .1}, {-.3, .5}}
	value := 0.
	for j := range 2 {
		limit := float64(k)*dt - tau[j]
		if limit < -1e-12 {
			continue
		}
		if limit < 0 {
			limit = 0
		}
		value += d[i][j] * delayInputAt(u, j, limit, dt, foh)
		for q := 0; float64(q)*dt < limit-1e-14; q++ {
			start, end := float64(q)*dt, math.Min(float64(q+1)*dt, limit)
			const subdivisions = 128
			h := (end - start) / subdivisions
			sample := func(s float64) float64 {
				input := u.At(j, q)
				if foh {
					input += (s - start) / dt * (u.At(j, q+1) - u.At(j, q))
				}
				return delayAnalyticImpulse(i, j, limit-s) * input
			}
			integral := sample(start) + sample(end)
			for v := 1; v < subdivisions; v++ {
				weight := 2.
				if v%2 == 1 {
					weight = 4
				}
				integral += weight * sample(start+float64(v)*h)
			}
			value += h * integral / 3
		}
	}
	return value
}

func TestDelayedZOHIndependentSampledMIMO(t *testing.T) {
	for _, input := range [][]float64{{.035, .075}, {.12, .06}, {.01, .09}} {
		for _, output := range [][]float64{{.015, .02}, {.08, .09}} {
			for _, format := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
				sys := prewarpModel(t)
				sys.InputDelay = input
				sys.OutputDelay = output
				disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodZOH, DelayModeling: format})
				if err != nil {
					t.Fatal(err)
				}
				u := delayTestInput()
				response, err := disc.Simulate(u, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				for i := range 2 {
					for k := range 28 {
						want := delayAnalyticSample(u, i, k, .1, []float64{input[0] + output[i], input[1] + output[i]}, false)
						if math.Abs(response.Y.At(i, k)-want) > 2e-10 {
							t.Fatalf("input=%v output=%v format=%s i=%d k=%d got=%g want=%g", input, output, format, i, k, response.Y.At(i, k), want)
						}
					}
				}
				if format == C2DDelayModelingInternal {
					n, _, _ := disc.Dims()
					if n != 2 {
						t.Fatalf("shared state order=%d", n)
					}
					if disc.LFT != nil {
						for _, tau := range disc.LFT.Tau {
							if tau != 1 && tau != 2 {
								t.Fatalf("internal Tau=%g", tau)
							}
						}
					}
				}
			}
		}
	}
}

func TestDelayedFOHAndPathZOHIndependentSamples(t *testing.T) {
	for _, method := range []C2DMethod{C2DMethodFOH, C2DMethodZOH} {
		for _, format := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			sys := prewarpModel(t)
			sys.InputDelay = []float64{.035, .045}
			sys.OutputDelay = []float64{.02, .04}
			sys.Delay = mat.NewDense(2, 2, []float64{.11, .065, .015, .23})
			disc, err := sys.C2D(.1, C2DOptions{Method: method, DelayModeling: format})
			if err != nil {
				t.Fatal(err)
			}
			u := delayTestInput()
			response, err := disc.Simulate(u, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				for k := range 28 {
					tau := []float64{sys.InputDelay[0] + sys.OutputDelay[i] + sys.Delay.At(i, 0), sys.InputDelay[1] + sys.OutputDelay[i] + sys.Delay.At(i, 1)}
					want := delayAnalyticSample(u, i, k, .1, tau, method == C2DMethodFOH)
					if math.Abs(response.Y.At(i, k)-want) > 2e-10 {
						t.Fatalf("method=%s format=%s i=%d k=%d got=%g want=%g", method, format, i, k, response.Y.At(i, k), want)
					}
				}
			}
		}
	}
}

func TestConversionDelayRoundingAndThiran(t *testing.T) {
	for _, method := range []C2DMethod{C2DMethodTustin, C2DMethodMatched} {
		sys := makeTestSystem()
		sys.InputDelay = []float64{.26}
		sys.OutputDelay = []float64{.14}
		sys.Delay = mat.NewDense(1, 1, []float64{.17})
		disc, err := sys.C2D(.1, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		if disc.InputDelay[0] != 3 || disc.OutputDelay[0] != 1 || disc.Delay.At(0, 0) != 2 {
			t.Fatalf("rounding %s: %+v", method, disc)
		}
	}
	sys := makeTestSystem()
	sys.InputDelay = []float64{.35}
	delay, err := sys.C2D(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	state, err := sys.C2D(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := delay.Dims()
	if n != 2 || delay.LFT == nil {
		t.Fatalf("default delay order=%d LFT=%v", n, delay.LFT)
	}
	for _, theta := range []float64{.1, .5, 1, 2.2} {
		z := cmplx.Exp(complex(0, theta))
		want, err := state.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		assertPrewarpResponse(t, delay, z, want)
	}
}

func TestInternalTustinAllPortsIndependentResponse(t *testing.T) {
	sys := makeTestSystem()
	sys.LFT = &LFTDelay{Tau: []float64{.23, .31}, B2: mat.NewDense(2, 2, []float64{.2, .5, -.3, .1}), C2: mat.NewDense(2, 2, []float64{.4, .2, -.1, .3}), D12: mat.NewDense(1, 2, []float64{.1, .2}), D21: mat.NewDense(2, 1, []float64{.3, .5}), D22: mat.NewDense(2, 2, []float64{.01, .04, .02, .03})}
	dt, w := .1, 4.
	out, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w})
	if err != nil {
		t.Fatal(err)
	}
	beta := w / math.Tan(w*dt/2)
	for _, theta := range []float64{.1, .7, 1.3, 2.4} {
		z := cmplx.Exp(complex(0, theta))
		s := complex(beta, 0) * (z - 1) / (z + 1)
		rational, err := conversionAugmentedRational(sys).EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		d0, d1 := cmplx.Pow(z, -2), cmplx.Pow(z, -3)
		a, b, c, d := 1-rational[1][1]*d0, -rational[1][2]*d1, -rational[2][1]*d0, 1-rational[2][2]*d1
		r0, r1 := rational[1][0], rational[2][0]
		det := a*d - b*c
		q0, q1 := (d*r0-b*r1)/det, (-c*r0+a*r1)/det
		want := rational[0][0] + rational[0][1]*d0*q0 + rational[0][2]*d1*q1
		assertPrewarpResponse(t, out, z, [][]complex128{{want}})
	}
	restored, err := out.D2C(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: w})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "B2", restored.LFT.B2, sys.LFT.B2, 1e-12)
	assertMatClose(t, "C2", restored.LFT.C2, sys.LFT.C2, 1e-12)
	assertMatClose(t, "D22", restored.LFT.D22, sys.LFT.D22, 1e-12)
	if restored.LFT.Tau[0] != .2 || math.Abs(restored.LFT.Tau[1]-.3) > 1e-14 {
		t.Fatalf("delay units=%v", restored.LFT.Tau)
	}
}

func TestInternalTustinEliminatesRoundedZeroDelay(t *testing.T) {
	sys := makeTestSystem()
	sys.LFT = &LFTDelay{Tau: []float64{.01}, B2: mat.NewDense(2, 1, []float64{.2, .3}), C2: mat.NewDense(1, 2, []float64{.1, .2}), D12: mat.NewDense(1, 1, []float64{.5}), D21: mat.NewDense(1, 1, []float64{.3}), D22: mat.NewDense(1, 1, []float64{.2})}
	out, err := sys.C2D(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if out.LFT != nil {
		t.Fatal("zero-delay feedback not eliminated")
	}
	rounded, err := sys.ZeroDelayApprox()
	if err != nil {
		t.Fatal(err)
	}
	for _, theta := range []float64{.2, .7, 1.9} {
		z := cmplx.Exp(complex(0, theta))
		want, err := rounded.EvalFr(20 * (z - 1) / (z + 1))
		if err != nil {
			t.Fatal(err)
		}
		assertPrewarpResponse(t, out, z, want)
	}
}

func TestConversionDelayUnsupportedExplicit(t *testing.T) {
	sys := makeTestSystem()
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse, C2DMethodLeastSquares} {
		if _, err := sys.C2D(.1, C2DOptions{Method: method, ThiranOrder: 1}); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("method=%s error=%v", method, err)
		}
	}
}

func BenchmarkConversionExternalDelay(b *testing.B) {
	for _, kind := range []string{"none", "fractional-io", "residual-path"} {
		b.Run(kind, func(b *testing.B) {
			sys := makeTestSystem()
			if kind != "none" {
				sys.InputDelay = []float64{.035}
				sys.OutputDelay = []float64{.085}
			}
			if kind == "residual-path" {
				sys.Delay = mat.NewDense(1, 1, []float64{.075})
			}
			b.ReportAllocs()
			for range b.N {
				if _, err := sys.C2D(.1, C2DOptions{Method: C2DMethodZOH}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestImpulseFractionalPathDelayIndependentSamples(t *testing.T) {
	sys := prewarpModel(t)
	sys.InputDelay = []float64{.035, .045}
	sys.OutputDelay = []float64{.02, .04}
	sys.Delay = mat.NewDense(2, 2, []float64{.11, .065, .015, .23})
	for _, format := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
		disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodImpulse, DelayModeling: format})
		if err != nil {
			t.Fatal(err)
		}
		for j := range 2 {
			u := mat.NewDense(2, 20, nil)
			u.Set(j, 0, 1)
			response, err := disc.Simulate(u, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				tau := sys.InputDelay[j] + sys.OutputDelay[i] + sys.Delay.At(i, j)
				for k := range 20 {
					time := float64(k)*.1 - tau
					want := 0.
					if time >= 0 {
						want = .1 * delayAnalyticImpulse(i, j, time)
					}
					if math.Abs(response.Y.At(i, k)-want) > 1e-12 {
						t.Fatalf("format=%s i=%d j=%d k=%d got=%g want=%g", format, i, j, k, response.Y.At(i, k), want)
					}
				}
			}
		}
	}
}

func TestFractionalGainZOHIndependentStep(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{.35}
	for _, format := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
		disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodZOH, DelayModeling: format})
		if err != nil {
			t.Fatal(err)
		}
		response, err := disc.Simulate(mat.NewDense(1, 8, []float64{1, 1, 1, 1, 1, 1, 1, 1}), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for k := range 8 {
			want := 2.
			if k < 4 {
				want = 0
			}
			if response.Y.At(0, k) != want {
				t.Fatalf("format=%s k=%d got=%g want=%g", format, k, response.Y.At(0, k), want)
			}
		}
	}
}

func TestImpulseIntegerPathDelayRetained(t *testing.T) {
	sys := prewarpModel(t)
	sys.Delay = mat.NewDense(2, 2, []float64{.2, .3, .1, .4})
	disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	for _, theta := range []float64{.1, .7, 1.9} {
		z := cmplx.Exp(complex(0, theta))
		raw := sys.Copy()
		raw.Delay = nil
		plain, err := raw.C2D(.1, C2DOptions{Method: C2DMethodImpulse})
		if err != nil {
			t.Fatal(err)
		}
		want, err := plain.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 2 {
			for j := range 2 {
				want[i][j] *= cmplx.Pow(z, -complex(math.Round(sys.Delay.At(i, j)/.1), 0))
			}
		}
		assertPrewarpResponse(t, disc, z, want)
	}
}

func TestInternalTustinRetainsExternalPathDelay(t *testing.T) {
	source := conversionDynamicLFT(t)
	cont, err := source.D2C(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	cont.Delay = mat.NewDense(1, 1, []float64{.3})
	out, err := cont.C2D(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if out.Delay == nil || out.Delay.At(0, 0) != 3 {
		t.Fatalf("path metadata=%v", out.Delay)
	}
	for _, theta := range []float64{.1, .7, 1.9} {
		z := cmplx.Exp(complex(0, theta))
		want, err := source.EvalFr(z)
		if err != nil {
			t.Fatal(err)
		}
		want[0][0] /= z * z * z
		assertPrewarpResponse(t, out, z, want)
	}
}
