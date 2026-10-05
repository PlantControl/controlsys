package controlsys

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func negativeOrderFixture(t *testing.T, a *mat.Dense, dt float64) *System {
	t.Helper()
	n, _ := a.Dims()
	b, c, d := mat.NewDense(n, 2, nil), mat.NewDense(2, n, nil), mat.NewDense(2, 2, []float64{.2, -.1, .3, .05})
	for i := range n {
		b.Set(i, 0, float64(i%3+1)/4)
		b.Set(i, 1, float64(2-i%4)/5)
		c.Set(0, i, float64(i%2+1)/3)
		c.Set(1, i, float64(1-i%3)/4)
	}
	sys, err := New(a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	sys.StateName = make([]string, n)
	for i := range n {
		sys.StateName[i] = string(rune('a' + i))
	}
	sys.InputName = []string{"u", "v"}
	sys.OutputName = []string{"y", "z"}
	return sys
}

func verifyNegativeOrderSamples(t *testing.T, source *System, added int) {
	t.Helper()
	before := source.Copy()
	out, G, err := source.D2CMap(D2COptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := source.Dims()
	total, _, _ := out.Dims()
	if total != n+added {
		t.Fatalf("order=%d, want%d", total, n+added)
	}
	if !reflect.DeepEqual(out.InputName, source.InputName) || !reflect.DeepEqual(out.OutputName, source.OutputName) || !reflect.DeepEqual(out.StateName[:n], source.StateName) {
		t.Fatal("source names changed")
	}
	xd := make([]float64, n)
	for i := range n {
		xd[i] = float64(i+1) / 7
	}
	xc := applyInitialMap(t, G, xd, []float64{.3, -.2})
	for i := range n {
		if xc[i] != xd[i] {
			t.Fatal("original states not retained first")
		}
	}
	for _, v := range xc[n:] {
		if v != 0 {
			t.Fatal("aliases need zero initial value")
		}
	}
	for k := range 14 {
		u := make([]float64, m)
		for j := range m {
			u[j] = math.Sin(float64(k+2*j)) + .1*float64(j+1)
		}
		for i := range p {
			yd, yc := 0., 0.
			for j := range n {
				yd += source.C.At(i, j) * xd[j]
			}
			for j := range total {
				yc += out.C.At(i, j) * xc[j]
			}
			for j := range m {
				yd += source.D.At(i, j) * u[j]
				yc += out.D.At(i, j) * u[j]
			}
			if math.Abs(yd-yc) > 2e-8*(1+math.Abs(yd)) {
				t.Fatalf("sample=%d output=%d got%g want%g", k, i, yc, yd)
			}
		}
		xd = holdTestStep(source.A, source.B, xd, u)
		xc = holdTestRK4(out, xc, u, u, source.Dt)
		for i := range n {
			if math.Abs(xc[i]-xd[i]) > 2e-8*(1+math.Abs(xd[i])) {
				t.Fatalf("sample=%d state=%d got%g want%g", k+1, i, xc[i], xd[i])
			}
		}
		for i, v := range xc[n:] {
			if math.Abs(v) > 2e-8 {
				t.Fatalf("sample=%d alias=%d got%g", k+1, i, v)
			}
		}
	}
	assertMatClose(t, "immutableA", source.A, before.A, 0)
	assertMatClose(t, "immutableB", source.B, before.B, 0)
	assertMatClose(t, "immutableC", source.C, before.C, 0)
	assertMatClose(t, "immutableD", source.D, before.D, 0)
}

func TestNegativeZOHAddsOnlyNegativeModes(t *testing.T) {
	for _, fixture := range []struct {
		a     *mat.Dense
		added int
	}{
		{mat.NewDense(2, 2, []float64{.8, 0, 0, -.8}), 1},
		{mat.NewDense(4, 4, []float64{.8, .2, .1, -.3, 0, .8, .15, .1, 0, 0, .6, .25, 0, 0, 0, -.7}), 1},
		{mat.NewDense(4, 4, []float64{-.6, .15, .1, -.2, 0, -.6, .25, .2, 0, 0, .7, .15, 0, 0, 0, .7}), 2},
		{mat.NewDense(3, 3, []float64{-.5, .1, -.2, 0, -.5, .2, 0, 0, -.5}), 3},
	} {
		verifyNegativeOrderSamples(t, negativeOrderFixture(t, fixture.a, .2), fixture.added)
	}
}

func nearCutNegativeFixture(t *testing.T, gap float64) *System {
	t.Helper()
	co, si := .8*math.Cos(gap), .8*math.Sin(gap)
	a := mat.NewDense(6, 6, []float64{.7, .15, .2, 0, .1, -.05, 0, .7, .1, .2, 0, .1, 0, 0, -.6, .1, .15, -.1, 0, 0, 0, -.6, .1, .2, 0, 0, 0, 0, -co, si, 0, 0, 0, 0, -si, -co})
	return negativeOrderFixture(t, a, .2)
}

func TestNegativeZOHDoesNotDuplicateExistingNearCutConjugates(t *testing.T) {
	for _, gap := range []float64{.2, .02, .002} {
		t.Run(fmtNegativeGap(gap), func(t *testing.T) { verifyNegativeOrderSamples(t, nearCutNegativeFixture(t, gap), 2) })
	}
}

func fmtNegativeGap(gap float64) string {
	if gap == .2 {
		return "gap0.2"
	}
	if gap == .02 {
		return "gap0.02"
	}
	return "gap0.002"
}

func TestNegativeZOHRejectsUnresolvedBranchCut(t *testing.T) {
	source := nearCutNegativeFixture(t, 1e-14)
	before := source.Copy()
	if _, err := source.D2C(D2COptions{Method: C2DMethodZOH}); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("nearbranch error=%v", err)
	}
	assertMatClose(t, "immutableA", source.A, before.A, 0)
}

func BenchmarkNegativeZOHCompressedExtension(b *testing.B) {
	for _, a := range []*mat.Dense{mat.NewDense(2, 2, []float64{.8, .2, 0, -.8}), mat.NewDense(4, 4, []float64{-.6, .15, .1, -.2, 0, -.6, .25, .2, 0, 0, .7, .15, 0, 0, 0, .7})} {
		n, _ := a.Dims()
		name := "n2negative1"
		if n == 4 {
			name = "n4negative2"
		}
		b.Run(name, func(b *testing.B) {
			input, output := make([]float64, n), make([]float64, n)
			for i := range n {
				input[i] = float64(i+1) / 4
				output[i] = float64(n-i) / 3
			}
			sys, _ := New(a, mat.NewDense(n, 1, input), mat.NewDense(1, n, output), mat.NewDense(1, 1, nil), .2)
			b.ReportAllocs()
			for range b.N {
				if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestNegativeZOHNonorthogonalDenseCoordinates(t *testing.T) {
	s := mat.NewDense(4, 4, []float64{1, .2, -.1, .3, .1, 1, .25, -.1, -.2, .1, 1, .1, .15, -.1, .1, 1})
	var inverse, left, a mat.Dense
	if err := inverse.Inverse(s); err != nil {
		t.Fatal(err)
	}
	base := mat.NewDense(4, 4, []float64{.7, .2, .1, -.1, 0, .5, .1, .2, 0, 0, -.6, .25, 0, 0, 0, -.8})
	left.Mul(s, base)
	a.Mul(&left, &inverse)
	for _, dt := range []float64{.02, .2, 2, 2e10} {
		verifyNegativeOrderSamples(t, negativeOrderFixture(t, &a, dt), 2)
	}
}

func TestNegativeZOHRejectsOverflowBeforeSubspaceSolve(t *testing.T) {
	source := negativeOrderFixture(t, mat.NewDense(2, 2, []float64{.8, .2, 0, -.8}), math.SmallestNonzeroFloat64)
	if _, err := source.D2C(D2COptions{Method: C2DMethodZOH}); !errors.Is(err, ErrOverflow) {
		t.Fatalf("error=%v", err)
	}
}
