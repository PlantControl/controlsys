package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func prewarpModel(t *testing.T) *System {
	t.Helper()
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, .5, 2, -1}), mat.NewDense(2, 2, []float64{1, -.4, 0, 2}), mat.NewDense(2, 2, []float64{.2, .1, -.3, .5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func prewarpAnalytic(s complex128) [][]complex128 {
	b := [][]float64{{1, .5}, {2, -1}}
	c := [][]float64{{1, -.4}, {0, 2}}
	d := [][]float64{{.2, .1}, {-.3, .5}}
	h := make([][]complex128, 2)
	for i := range 2 {
		h[i] = make([]complex128, 2)
		for j := range 2 {
			x2 := complex(b[1][j], 0) / (s + 3)
			x1 := (complex(b[0][j], 0) + 2*x2) / (s + 1)
			h[i][j] = complex(c[i][0], 0)*x1 + complex(c[i][1], 0)*x2 + complex(d[i][j], 0)
		}
	}
	return h
}

func assertPrewarpResponse(t *testing.T, sys *System, s complex128, want [][]complex128) {
	t.Helper()
	got, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		for j := range want[i] {
			if cmplx.Abs(got[i][j]-want[i][j]) > 1e-11*(1+cmplx.Abs(want[i][j])) {
				t.Fatalf("response[%d,%d]=%v, want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestPrewarpMIMOIndependentFrequency(t *testing.T) {
	orig := prewarpModel(t)
	snapshot := orig.Copy()
	dt, w := .15, 7.
	opts := C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w}
	disc, err := orig.DiscretizeWithOpts(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	assertPrewarpResponse(t, disc, cmplx.Exp(complex(0, w*dt)), prewarpAnalytic(complex(0, w)))
	beta := w / math.Tan(w*dt/2)
	for _, theta := range []float64{.02, .5, 1.7, 2.8} {
		z := cmplx.Exp(complex(0, theta))
		s := complex(beta, 0) * (z - 1) / (z + 1)
		assertPrewarpResponse(t, disc, z, prewarpAnalytic(s))
	}
	restored, err := disc.D2CWithOpts(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: w})
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct {
		name      string
		got, want *mat.Dense
	}{{"A", restored.A, orig.A}, {"B", restored.B, orig.B}, {"C", restored.C, orig.C}, {"D", restored.D, orig.D}, {"source A", orig.A, snapshot.A}, {"source B", orig.B, snapshot.B}, {"source C", orig.C, snapshot.C}, {"source D", orig.D, snapshot.D}} {
		assertMatClose(t, pair.name, pair.got, pair.want, 1e-12)
	}
}

func TestPrewarpSISOIndependentInverse(t *testing.T) {
	dt, w := .2, 5.
	beta := w / math.Tan(w*dt/2)
	a, b, c, d := -2., 3., 4., .7
	ad := (beta + a) / (beta - a)
	bd := 2 * b / (beta - a)
	cd := beta * c / (beta - a)
	dd := d + c*b/(beta-a)
	disc, err := New(mat.NewDense(1, 1, []float64{ad}), mat.NewDense(1, 1, []float64{bd}), mat.NewDense(1, 1, []float64{cd}), mat.NewDense(1, 1, []float64{dd}), dt)
	if err != nil {
		t.Fatal(err)
	}
	cont, err := disc.D2CWithOpts(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: w})
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct{ got, want float64 }{{cont.A.At(0, 0), a}, {cont.B.At(0, 0), b}, {cont.C.At(0, 0), c}, {cont.D.At(0, 0), d}} {
		if math.Abs(pair.got-pair.want) > 1e-12 {
			t.Fatalf("got %g, want %g", pair.got, pair.want)
		}
	}
}

func TestPrewarpZeroLimitAndDelayUnits(t *testing.T) {
	sys := makeTestSystem()
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	sys.InputDelay = []float64{.4}
	sys.OutputDelay = []float64{.2}
	plain, err := sys.Discretize(.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0, 1e-12, math.SmallestNonzeroFloat64} {
		out, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w})
		if err != nil {
			t.Fatal(err)
		}
		assertMatClose(t, "A", out.A, plain.A, 1e-12)
		assertMatClose(t, "B", out.B, plain.B, 1e-12)
		assertMatClose(t, "C", out.C, plain.C, 1e-12)
		assertMatClose(t, "D", out.D, plain.D, 1e-12)
	}
	out, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: 8})
	if err != nil {
		t.Fatal(err)
	}
	if out.InputDelay[0] != 4 || out.OutputDelay[0] != 2 {
		t.Fatalf("delay units: %v/%v", out.InputDelay, out.OutputDelay)
	}
	cont, err := out.D2CWithOpts(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: 8})
	if err != nil {
		t.Fatal(err)
	}
	if cont.InputDelay[0] != .4 || cont.OutputDelay[0] != .2 || cont.InputName[0] != "u" || cont.OutputName[0] != "y" {
		t.Fatalf("metadata: %+v", cont)
	}
	out.InputName[0] = "changed"
	cont.InputDelay[0] = 3
	if sys.InputName[0] != "u" || sys.InputDelay[0] != .4 {
		t.Fatal("source metadata mutated")
	}
}

func TestPrewarpInvalidOptions(t *testing.T) {
	sys := makeTestSystem()
	disc, err := sys.Discretize(.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{-1, math.NaN(), math.Inf(1), math.Pi / .1, 40} {
		if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w}); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("C2D frequency %g: %v", w, err)
		}
		if _, err := disc.D2CWithOpts(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: w}); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("D2C frequency %g: %v", w, err)
		}
	}
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodMatched} {
		if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: method, PrewarpFrequency: 1}); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("C2D method %s: %v", method, err)
		}
		if _, err := disc.D2CWithOpts(D2COptions{Method: method, PrewarpFrequency: 1}); !errors.Is(err, ErrInvalidConversionOptions) {
			t.Fatalf("D2C method %s: %v", method, err)
		}
	}
	for _, dt := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := sys.Discretize(dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Fatalf("Discretize dt %g: %v", dt, err)
		}
		bad := disc.Copy()
		bad.Dt = dt
		if dt != 0 {
			if _, err := bad.D2CWithOpts(D2COptions{Method: C2DMethodTustin}); !errors.Is(err, ErrInvalidSampleTime) {
				t.Fatalf("D2C dt %g: %v", dt, err)
			}
		}
	}
	bad := disc.Copy()
	bad.A = mat.NewDense(2, 2, []float64{-1, 1, 0, .5})
	if _, err := bad.D2CWithOpts(D2COptions{Method: C2DMethodTustin, PrewarpFrequency: 1}); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("z=-1: %v", err)
	}
}

func BenchmarkPrewarpMIMO(b *testing.B) {
	sys := makeTestSystem()
	for _, w := range []float64{0, 7} {
		b.Run(fmtPrewarpName(w), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if _, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: w}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func fmtPrewarpName(w float64) string {
	if w == 0 {
		return "plain"
	}
	return "prewarp"
}
