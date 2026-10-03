package controlsys

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"plantcontrol.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestD2CZOHDefectiveAndNearUnit(t *testing.T) {
	for _, pole := range []float64{1, 1 + 1e-12, 0.5, 2} {
		ad := mat.NewDense(3, 3, []float64{pole, 0.2, -0.1, 0, pole, 0.3, 0, 0, pole})
		bd := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, -1})
		sys, _ := New(ad, bd, mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 1}), mat.NewDense(2, 2, []float64{0.2, 0, 0, -0.1}), 0.25)
		before := sys.Copy()
		sys.StateName = []string{"x", "y", "z"}
		out, err := sys.D2C(C2DMethodZOH)
		if err != nil {
			t.Fatalf("pole=%g: %v", pole, err)
		}
		nilpotent := mat.NewDense(3, 3, []float64{0, 0.2, -0.1, 0, 0, 0.3, 0, 0, 0})
		var square mat.Dense
		square.Mul(nilpotent, nilpotent)
		expected := mat.NewDense(3, 3, nil)
		expected.Scale(1/pole, nilpotent)
		square.Scale(-0.5/(pole*pole), &square)
		expected.Add(expected, &square)
		for i := range 3 {
			expected.Set(i, i, math.Log(pole))
		}
		expected.Scale(4, expected)
		assertMatClose(t, "analytic Jordan logarithm", out.A, expected, 2e-11)
		back, err := out.DiscretizeZOH(sys.Dt)
		if err != nil {
			t.Fatal(err)
		}
		assertMatClose(t, "Ad", back.A, sys.A, 2e-11)
		assertMatClose(t, "Bd", back.B, sys.B, 2e-11)
		assertMatClose(t, "immutable A", sys.A, before.A, 0)
		assertMatClose(t, "immutable B", sys.B, before.B, 0)
		if !reflect.DeepEqual(out.StateName, sys.StateName) {
			t.Fatalf("state names lost")
		}
	}
}

func TestD2CZOHNegativeMixedDefectiveResponses(t *testing.T) {
	a := mat.NewDense(4, 4, []float64{-0.4, 0.3, 0.2, -0.1, 0, -0.4, 0.1, 0, 0, 0, 0.6, 0.15, 0, 0, 0, 0.9})
	b := mat.NewDense(4, 2, []float64{1, 0, 0, 1, 1, -1, 0.3, 0.2})
	c := mat.NewDense(2, 4, []float64{1, 0.2, 0, 1, 0, 1, -1, 0.2})
	d := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})
	sys, _ := New(a, b, c, d, 0.2)
	sys.InputName = []string{"u1", "u2"}
	sys.OutputName = []string{"y1", "y2"}
	sys.StateName = []string{"x1", "x2", "x3", "x4"}
	sys.InputDelay = []float64{2, 0}
	out, err := sys.D2C(C2DMethodZOH)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := out.Dims(); n != 6 {
		t.Fatalf("extension order %d", n)
	}
	if !reflect.DeepEqual(out.InputName, sys.InputName) || !reflect.DeepEqual(out.OutputName, sys.OutputName) {
		t.Fatal("I/O names lost")
	}
	if out.InputDelay[0] != 0.4 {
		t.Fatalf("delay=%v", out.InputDelay)
	}
	disc, err := out.DiscretizeZOH(0.2)
	if err != nil {
		t.Fatal(err)
	}
	xd := []float64{0.5, -0.3, 0.2, 1}
	xc := append(append([]float64(nil), xd...), make([]float64, 2)...)
	for k := range 20 {
		u := []float64{math.Sin(float64(k)), float64(k%3) - 1}
		for i := range 2 {
			yd, yc := 0.0, 0.0
			for j := range 4 {
				yd += sys.C.At(i, j) * xd[j]
			}
			for j := range 6 {
				yc += disc.C.At(i, j) * xc[j]
			}
			if math.Abs(yd-yc) > 2e-9 {
				t.Fatalf("sample %d output %d: %g vs %g", k, i, yd, yc)
			}
		}
		xd = holdTestStep(sys.A, sys.B, xd, u)
		xc = holdTestStep(disc.A, disc.B, xc, u)
	}
}

func TestD2CZeroPoleRejected(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		if _, err := sys.D2C(method); !errors.Is(err, ErrSingularTransform) {
			t.Fatalf("%s error=%v", method, err)
		}
	}
}

func TestModifiedFOHPiecewiseLinearMIMO(t *testing.T) {
	a := mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 0, 0.3, -0.2, -1.7})
	b := mat.NewDense(3, 2, []float64{0, 0.2, 1, 0, 0.3, 1})
	c := mat.NewDense(2, 3, []float64{1, 0.2, 0, 0, 1, 0.5})
	d := mat.NewDense(2, 2, []float64{0.2, -0.1, 0, 0.3})
	sys, _ := New(a, b, c, d, 0)
	sys.StateName = []string{"x", "v", "lag"}
	sys.InputName = []string{"force", "bias"}
	sys.OutputName = []string{"position", "rate"}
	before := sys.Copy()
	dt := 0.15
	disc, err := sys.DiscretizeFOH(dt)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := disc.D2C(C2DMethodFOH)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "Ac", rec.A, sys.A, 2e-11)
	assertMatClose(t, "Bc", rec.B, sys.B, 2e-11)
	assertMatClose(t, "Cc", rec.C, sys.C, 2e-11)
	assertMatClose(t, "Dc", rec.D, sys.D, 2e-11)
	assertMatClose(t, "source A", sys.A, before.A, 0)
	assertMatClose(t, "source D", sys.D, before.D, 0)
	if !reflect.DeepEqual(rec.StateName, sys.StateName) || !reflect.DeepEqual(rec.InputName, sys.InputName) {
		t.Fatal("names lost")
	}
	samples := [][]float64{{0.4, -0.2}, {1, 0.5}, {-0.5, 0.7}, {0, -0.3}, {2, 0.1}}
	xc := []float64{0.3, -0.2, 0.5}
	gamma1 := holdTestGamma1(sys, dt)
	xd := append([]float64(nil), xc...)
	for i := range 3 {
		for j := range 2 {
			xd[i] -= gamma1.At(i, j) * samples[0][j]
		}
	}
	for k, u := range samples {
		for i := range 2 {
			yc, yd := 0.0, 0.0
			for j := range 3 {
				yc += sys.C.At(i, j) * xc[j]
				yd += disc.C.At(i, j) * xd[j]
			}
			for j := range 2 {
				yc += sys.D.At(i, j) * u[j]
				yd += disc.D.At(i, j) * u[j]
			}
			if math.Abs(yc-yd) > 2e-10 {
				t.Fatalf("sample=%d output=%d: %g vs %g", k, i, yc, yd)
			}
		}
		if k+1 == len(samples) {
			break
		}
		xd = holdTestStep(disc.A, disc.B, xd, u)
		xc = holdTestRK4(sys, xc, u, samples[k+1], dt)
	}
}

func holdTestStep(a, b *mat.Dense, x, u []float64) []float64 {
	n, _ := a.Dims()
	out := make([]float64, n)
	for i := range n {
		for j, v := range x {
			out[i] += a.At(i, j) * v
		}
		for j, v := range u {
			out[i] += b.At(i, j) * v
		}
	}
	return out
}

func holdTestRK4(sys *System, x, u0, u1 []float64, dt float64) []float64 {
	out := append([]float64(nil), x...)
	n := len(x)
	steps := 400
	h := dt / float64(steps)
	derivative := func(state []float64, t float64) []float64 {
		value := make([]float64, n)
		for i := range n {
			for j, v := range state {
				value[i] += sys.A.At(i, j) * v
			}
			for j, v := range u0 {
				value[i] += sys.B.At(i, j) * (v + (u1[j]-v)*t/dt)
			}
		}
		return value
	}
	for k := range steps {
		t := float64(k) * h
		k1 := derivative(out, t)
		tmp := make([]float64, n)
		for i := range n {
			tmp[i] = out[i] + h*k1[i]/2
		}
		k2 := derivative(tmp, t+h/2)
		for i := range n {
			tmp[i] = out[i] + h*k2[i]/2
		}
		k3 := derivative(tmp, t+h/2)
		for i := range n {
			tmp[i] = out[i] + h*k3[i]
		}
		k4 := derivative(tmp, t+h)
		for i := range n {
			out[i] += h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
		}
	}
	return out
}

func holdTestGamma1(sys *System, dt float64) *mat.Dense {
	n, m, _ := sys.Dims()
	result := mat.NewDense(n, m, nil)
	for j := range m {
		u0, u1 := make([]float64, m), make([]float64, m)
		u1[j] = 1
		column := holdTestRK4(sys, make([]float64, n), u0, u1, dt)
		for i, v := range column {
			result.Set(i, j, v)
		}
	}
	return result
}

func TestMatLogRepeatedComplexModes(t *testing.T) {
	a := mat.NewDense(4, 4, []float64{0.8, -0.6, 0.1, 0, 0.6, 0.8, 0, 0.1, 0, 0, 0.8, -0.6, 0, 0, 0.6, 0.8})
	logarithm, err := matLog(a)
	if err != nil {
		t.Fatal(err)
	}
	var exponential mat.Dense
	exponential.Exp(logarithm)
	assertMatClose(t, "complex Jordan exponential", &exponential, a, 2e-12)
}

func TestModifiedFOHSciPyReference(t *testing.T) {
	data, err := os.ReadFile("testdata/conversion_foh_scipy.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Cases []struct {
			Name                       string
			N, M, P                    int
			Dt                         float64
			A, B, C, D, Ad, Bd, Cd, Dd []float64
		}
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, tc := range reference.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			sys, err := New(mat.NewDense(tc.N, tc.N, tc.A), mat.NewDense(tc.N, tc.M, tc.B), mat.NewDense(tc.P, tc.N, tc.C), mat.NewDense(tc.P, tc.M, tc.D), 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := sys.DiscretizeFOH(tc.Dt)
			if err != nil {
				t.Fatal(err)
			}
			assertMatClose(t, "Ad", got.A, mat.NewDense(tc.N, tc.N, tc.Ad), 2e-12)
			assertMatClose(t, "Bd", got.B, mat.NewDense(tc.N, tc.M, tc.Bd), 2e-12)
			assertMatClose(t, "Cd", got.C, mat.NewDense(tc.P, tc.N, tc.Cd), 2e-12)
			assertMatClose(t, "Dd", got.D, mat.NewDense(tc.P, tc.M, tc.Dd), 2e-12)
			fixture, _ := New(mat.NewDense(tc.N, tc.N, tc.Ad), mat.NewDense(tc.N, tc.M, tc.Bd), mat.NewDense(tc.P, tc.N, tc.Cd), mat.NewDense(tc.P, tc.M, tc.Dd), tc.Dt)
			restored, err := fixture.D2C(C2DMethodFOH)
			if err != nil {
				t.Fatal(err)
			}
			assertMatClose(t, "Ac", restored.A, sys.A, 2e-11)
			assertMatClose(t, "Bc", restored.B, sys.B, 2e-11)
			assertMatClose(t, "Cc", restored.C, sys.C, 2e-11)
			assertMatClose(t, "Dc", restored.D, sys.D, 2e-11)
		})
	}
}

func TestFOHIntegratorFeedthroughAndReverseLimitations(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), 0)
	disc, err := sys.DiscretizeFOH(0.2)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "integrator A", disc.A, mat.NewDense(1, 1, []float64{1}), 1e-14)
	assertMatClose(t, "integrator B", disc.B, mat.NewDense(1, 1, []float64{0.4}), 1e-14)
	assertMatClose(t, "integrator D", disc.D, mat.NewDense(1, 1, []float64{4.6}), 1e-14)
	rec, err := disc.D2C(C2DMethodFOH)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "integrator Bc", rec.B, sys.B, 1e-13)
	assertMatClose(t, "integrator Dc", rec.D, sys.D, 1e-13)
	disc.A.Set(0, 0, -0.5)
	if _, err := disc.D2C(C2DMethodFOH); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("negative pole error=%v", err)
	}
}

func BenchmarkModifiedFOHForwardN20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	sys.Dt = 0
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.DiscretizeFOH(0.01); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModifiedFOHReverseN20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.D2C(C2DMethodFOH); err != nil {
			b.Fatal(err)
		}
	}
}
