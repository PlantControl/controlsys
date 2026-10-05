package controlsys

import (
	"encoding/json"
	"errors"
	"math"
	"math/cmplx"
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
		out, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
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
		back, err := out.C2D(sys.Dt, C2DOptions{})
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
	out, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
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
	disc, err := out.C2D(0.2, C2DOptions{})
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
		if _, err := sys.D2C(D2COptions{Method: method}); !errors.Is(err, ErrSingularTransform) {
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
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	rec, err := disc.D2C(D2COptions{Method: C2DMethodFOH})
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
			got, err := sys.C2D(tc.Dt, C2DOptions{Method: C2DMethodFOH})
			if err != nil {
				t.Fatal(err)
			}
			assertMatClose(t, "Ad", got.A, mat.NewDense(tc.N, tc.N, tc.Ad), 2e-12)
			assertMatClose(t, "Bd", got.B, mat.NewDense(tc.N, tc.M, tc.Bd), 2e-12)
			assertMatClose(t, "Cd", got.C, mat.NewDense(tc.P, tc.N, tc.Cd), 2e-12)
			assertMatClose(t, "Dd", got.D, mat.NewDense(tc.P, tc.M, tc.Dd), 2e-12)
			fixture, _ := New(mat.NewDense(tc.N, tc.N, tc.Ad), mat.NewDense(tc.N, tc.M, tc.Bd), mat.NewDense(tc.P, tc.N, tc.Cd), mat.NewDense(tc.P, tc.M, tc.Dd), tc.Dt)
			restored, err := fixture.D2C(D2COptions{Method: C2DMethodFOH})
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
	disc, err := sys.C2D(0.2, C2DOptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "integrator A", disc.A, mat.NewDense(1, 1, []float64{1}), 1e-14)
	assertMatClose(t, "integrator B", disc.B, mat.NewDense(1, 1, []float64{0.4}), 1e-14)
	assertMatClose(t, "integrator D", disc.D, mat.NewDense(1, 1, []float64{4.6}), 1e-14)
	rec, err := disc.D2C(D2COptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "integrator Bc", rec.B, sys.B, 1e-13)
	assertMatClose(t, "integrator Dc", rec.D, sys.D, 1e-13)
	disc.A.Set(0, 0, -0.5)
	if _, err := disc.D2C(D2COptions{Method: C2DMethodFOH}); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("negative pole error=%v", err)
	}
}

func BenchmarkModifiedFOHForwardN20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	sys.Dt = 0
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodFOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkModifiedFOHReverseN20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodFOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func internalDelayDiscretePlant(t *testing.T, a []float64) *System {
	t.Helper()
	sys, err := New(mat.NewDense(3, 3, a),
		mat.NewDense(3, 2, []float64{1, 0.3, -0.2, 1, 0.5, -0.4}),
		mat.NewDense(2, 3, []float64{1, -0.3, 0.7, 0.4, 1, -0.2}),
		mat.NewDense(2, 2, []float64{0.2, 0.1, -0.05, -0.3}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	err = sys.SetInternalDelay([]float64{2, 3},
		mat.NewDense(3, 2, []float64{0.3, 0, -0.2, 0.4, 0.6, 0.1}),
		mat.NewDense(2, 3, []float64{0.2, -0.4, 0.5, 0, 0.3, -0.1}),
		mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.15}),
		mat.NewDense(2, 2, []float64{0.25, 0, 0.1, -0.2}),
		mat.NewDense(2, 2, []float64{0.05, 0.1, 0, -0.08}))
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func assertDelayResponseClose(t *testing.T, label string, got, want *System, points []complex128, tol float64) {
	t.Helper()
	for _, s := range points {
		g := evalDelaySystem(t, got, s, exactDelayFactor(got.Dt))
		w := evalDelaySystem(t, want, s, exactDelayFactor(want.Dt))
		for i := range w {
			for j := range w[i] {
				if cmplx.Abs(g[i][j]-w[i][j]) > tol*math.Max(1, cmplx.Abs(w[i][j])) {
					t.Fatalf("%s: G(%v)[%d][%d] = %v, want %v", label, s, i, j, g[i][j], w[i][j])
				}
			}
		}
		lib, err := got.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range g {
			for j := range g[i] {
				if cmplx.Abs(lib[i][j]-g[i][j]) > tol*math.Max(1, cmplx.Abs(g[i][j])) {
					t.Fatalf("%s: EvalFr(%v)[%d][%d] = %v, closure %v", label, s, i, j, lib[i][j], g[i][j])
				}
			}
		}
	}
}

func unitCirclePoints(dt float64) []complex128 {
	var out []complex128
	for _, w := range []float64{0.3, 2, 9, 25} {
		out = append(out, cmplx.Exp(complex(0, w*dt)))
	}
	return out
}

func TestD2CHoldInternalDelayRoundTrip(t *testing.T) {
	sys := internalDelayDiscretePlant(t, []float64{0.6, 0.2, -0.1, 0.05, 0.8, 0.3, 0, -0.1, 0.5})
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		cont, err := sys.D2C(D2COptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		if math.Abs(cont.LFT.Tau[0]-0.2) > 1e-15 || math.Abs(cont.LFT.Tau[1]-0.3) > 1e-15 {
			t.Fatalf("%s: tau = %v, want [0.2 0.3]", method, cont.LFT.Tau)
		}
		back, err := cont.C2D(sys.Dt, C2DOptions{Method: method})
		if err != nil {
			t.Fatalf("%s c2d: %v", method, err)
		}
		if !reflect.DeepEqual(back.LFT.Tau, sys.LFT.Tau) {
			t.Fatalf("%s: tau back = %v", method, back.LFT.Tau)
		}
		for name, pair := range map[string][2]*mat.Dense{
			"A": {back.A, sys.A}, "B": {back.B, sys.B}, "C": {back.C, sys.C}, "D": {back.D, sys.D},
			"B2": {back.LFT.B2, sys.LFT.B2}, "C2": {back.LFT.C2, sys.LFT.C2},
			"D12": {back.LFT.D12, sys.LFT.D12}, "D21": {back.LFT.D21, sys.LFT.D21}, "D22": {back.LFT.D22, sys.LFT.D22},
		} {
			assertMatClose(t, string(method)+" "+name, pair[0], pair[1], 1e-12)
		}
		assertDelayResponseClose(t, string(method)+" round trip", back, sys, unitCirclePoints(sys.Dt), 1e-11)
	}
}

func TestD2CZOHInternalDelayMatchesHoldExponential(t *testing.T) {
	sys := internalDelayDiscretePlant(t, []float64{0.6, 0.2, -0.1, 0.05, 0.8, 0.3, 0, -0.1, 0.5})
	cont, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	const n, q = 3, 4
	m := mat.NewDense(n+q, n+q, nil)
	m.Slice(0, n, 0, n).(*mat.Dense).Scale(sys.Dt, cont.A)
	m.Slice(0, n, n, n+2).(*mat.Dense).Scale(sys.Dt, cont.B)
	m.Slice(0, n, n+2, n+q).(*mat.Dense).Scale(sys.Dt, cont.LFT.B2)
	var e mat.Dense
	e.Exp(m)
	assertMatClose(t, "exp A", mat.DenseCopyOf(e.Slice(0, n, 0, n)), sys.A, 1e-13)
	assertMatClose(t, "exp B", mat.DenseCopyOf(e.Slice(0, n, n, n+2)), sys.B, 1e-13)
	assertMatClose(t, "exp B2", mat.DenseCopyOf(e.Slice(0, n, n+2, n+q)), sys.LFT.B2, 1e-13)
	for name, pair := range map[string][2]*mat.Dense{
		"C": {cont.C, sys.C}, "D": {cont.D, sys.D}, "C2": {cont.LFT.C2, sys.LFT.C2},
		"D12": {cont.LFT.D12, sys.LFT.D12}, "D21": {cont.LFT.D21, sys.LFT.D21}, "D22": {cont.LFT.D22, sys.LFT.D22},
	} {
		assertMatClose(t, name, pair[0], pair[1], 0)
	}
	assertDelayResponseClose(t, "continuous closure", cont, cont, []complex128{0.4i, 3i, 0.5 + 12i}, 1e-11)
}

func TestD2CZOHInternalDelayNegativePole(t *testing.T) {
	sys := internalDelayDiscretePlant(t, []float64{-0.4, 0.2, -0.1, 0.05, 0.8, 0.3, 0, -0.1, 0.5})
	cont, err := sys.D2C(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := cont.Dims(); n <= 3 {
		t.Fatalf("negative real pole should add extension states, got n=%d", n)
	}
	back, err := cont.C2D(sys.Dt, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertDelayResponseClose(t, "negative pole round trip", back, sys, unitCirclePoints(sys.Dt), 1e-10)
}

func TestD2DInternalDelay(t *testing.T) {
	sys := internalDelayDiscretePlant(t, []float64{0.6, 0.2, -0.1, 0.05, 0.8, 0.3, 0, -0.1, 0.5})
	fast, err := sys.D2D(0.05, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fast.LFT.Tau, []float64{4, 6}) {
		t.Fatalf("tau = %v, want [4 6]", fast.LFT.Tau)
	}
	back, err := fast.D2D(0.1, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertDelayResponseClose(t, "D2D 0.1→0.05→0.1", back, sys, unitCirclePoints(sys.Dt), 1e-10)
}

func TestD2CInternalDelayDescriptor(t *testing.T) {
	desc, oracle := index1Descriptor(t, 0.1, true)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		got, err := desc.D2C(D2COptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		want, err := oracle.D2C(D2COptions{Method: method})
		if err != nil {
			t.Fatalf("%s oracle: %v", method, err)
		}
		assertDelayResponseClose(t, "descriptor "+string(method), got, want, []complex128{0.4i, 3i, 0.5 + 12i}, 1e-10)
		back, err := got.C2D(0.1, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		assertDelayResponseClose(t, "descriptor round trip "+string(method), back, oracle, unitCirclePoints(0.1), 1e-10)
	}
	got, err := desc.D2D(0.05, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want, err := oracle.D2D(0.05, D2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	assertDelayResponseClose(t, "descriptor D2D", got, want, unitCirclePoints(0.05), 1e-10)
}

func TestD2CMapInternalDelayHold(t *testing.T) {
	sys := internalDelayDiscretePlant(t, []float64{0.6, 0.2, -0.1, 0.05, 0.8, 0.3, 0, -0.1, 0.5})
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		inverseSys, inverseMap, err := sys.D2CMap(D2COptions{Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		want := mat.NewDense(3, 7, nil)
		for i := range 3 {
			want.Set(i, i, 1)
		}
		if method == C2DMethodFOH {
			want.Slice(0, 3, 3, 7).(*mat.Dense).Copy(holdTestGamma1(conversionAugmentedRational(inverseSys), sys.Dt))
		}
		assertMatClose(t, string(method)+" inverse state map", inverseMap, want, 1e-9)
		_, forwardMap, err := inverseSys.C2DMap(sys.Dt, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		var sum mat.Dense
		sum.Add(forwardMap.Slice(0, 3, 3, 7), inverseMap.Slice(0, 3, 3, 7))
		assertMatClose(t, string(method)+" forward∘inverse input columns", &sum, mat.NewDense(3, 4, nil), 1e-12)
	}
}
