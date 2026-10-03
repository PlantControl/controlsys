package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestD2DZOHAnalyticRecurrence(t *testing.T) {
	for _, a := range []float64{-.8, 0} {
		for _, period := range []float64{.4, .07, .15} {
			initialDt, b := .1, 1.7
			ad, bd := math.Exp(a*initialDt), b*initialDt
			if a != 0 {
				bd = b * math.Expm1(a*initialDt) / a
			}
			disc, err := New(mat.NewDense(1, 1, []float64{ad}), mat.NewDense(1, 1, []float64{bd}), mat.NewDense(1, 1, []float64{2}), mat.NewDense(1, 1, []float64{-.3}), initialDt)
			if err != nil {
				t.Fatal(err)
			}
			for _, method := range []C2DMethod{"", C2DMethodZOH} {
				out, err := disc.D2D(period, C2DOptions{Method: method})
				if err != nil {
					t.Fatal(err)
				}
				wantA, wantB := math.Exp(a*period), b*period
				if a != 0 {
					wantB = b * math.Expm1(a*period) / a
				}
				if math.Abs(out.A.At(0, 0)-wantA) > 1e-12 || math.Abs(out.B.At(0, 0)-wantB) > 1e-12 || out.C.At(0, 0) != 2 || out.D.At(0, 0) != -.3 {
					t.Fatalf("a=%g Ts=%g: A=%v B=%v", a, period, out.A, out.B)
				}
				if disc.A.At(0, 0) != ad || disc.B.At(0, 0) != bd {
					t.Fatal("source mutated")
				}
			}
		}
	}
}

func TestD2DTustinIndependentFrequency(t *testing.T) {
	sys := prewarpModel(t)
	for _, w := range []float64{0, 6} {
		oldDt := .12
		beta := 2 / oldDt
		if w != 0 {
			beta = w / math.Tan(w*oldDt/2)
		}
		disc, err := sys.DiscretizeWithOpts(oldDt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w})
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{.3, .04, .17} {
			out, err := disc.D2D(dt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w})
			if err != nil {
				t.Fatal(err)
			}
			newBeta := 2 / dt
			if w != 0 {
				newBeta = w / math.Tan(w*dt/2)
			}
			for _, theta := range []float64{.05, .4, 1.2, 2.6} {
				z := cmplx.Exp(complex(0, theta))
				s := complex(newBeta, 0) * (z - 1) / (z + 1)
				assertPrewarpResponse(t, out, z, prewarpAnalytic(s))
				oldZ := (complex(beta, 0) + s) / (complex(beta, 0) - s)
				got, err := disc.EvalFr(oldZ)
				if err != nil {
					t.Fatal(err)
				}
				assertPrewarpResponse(t, out, z, got)
			}
		}
	}
}

func TestD2DValidatesSameRateOptions(t *testing.T) {
	sys := makeTestSystem()
	disc, err := sys.DiscretizeZOH(.1)
	if err != nil {
		t.Fatal(err)
	}
	options := []C2DOptions{{Method: "invalid"}, {Method: C2DMethodFOH}, {Method: C2DMethodMatched}, {Method: C2DMethodImpulse}, {Method: C2DMethodLeastSquares}, {ThiranOrder: -1}, {DelayModeling: "bad"}, {FitOrder: 2}, {Method: C2DMethodTustin, PrewarpFrequency: -1}, {Method: C2DMethodTustin, PrewarpFrequency: math.NaN()}, {Method: C2DMethodTustin, PrewarpFrequency: 40}, {PrewarpFrequency: 1}}
	for _, opts := range options {
		if _, err := disc.D2D(.1, opts); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("options %+v: %v", opts, err)
		}
	}
	for _, dt := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := disc.D2D(dt, C2DOptions{}); !errors.Is(err, ErrInvalidSampleTime) {
			t.Fatalf("dt %g: %v", dt, err)
		}
	}
	if _, err := disc.D2D(.4, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: 10}); !errors.Is(err, ErrInvalidConversionOptions) {
		t.Fatalf("new Nyquist: %v", err)
	}
	disc.Dt = .4
	if _, err := disc.D2D(.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: 10}); !errors.Is(err, ErrInvalidConversionOptions) {
		t.Fatalf("old Nyquist: %v", err)
	}
}

func TestD2DCopiesGainNamesAndDelays(t *testing.T) {
	disc := &System{A: newDense(0, 0), B: newDense(0, 1), C: newDense(1, 0), D: mat.NewDense(1, 1, []float64{3}), Dt: .1, InputDelay: []float64{4}, OutputDelay: []float64{2}, InputName: []string{"u"}, OutputName: []string{"y"}}
	for _, dt := range []float64{.1, .2} {
		out, err := disc.D2D(dt, C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if out.D.At(0, 0) != 3 || out.InputDelay[0] != .4/dt || out.OutputDelay[0] != .2/dt || out.InputName[0] != "u" || out.OutputName[0] != "y" {
			t.Fatalf("metadata: %+v", out)
		}
		out.D.Set(0, 0, 9)
		out.InputDelay[0] = 99
		out.InputName[0] = "changed"
		if disc.D.At(0, 0) != 3 || disc.InputDelay[0] != 4 || disc.InputName[0] != "u" {
			t.Fatal("source mutated")
		}
	}
}
