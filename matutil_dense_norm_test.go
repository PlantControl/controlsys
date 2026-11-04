package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func unscaledFrobenius(m *mat.Dense) float64 {
	raw := m.RawMatrix()
	sum := 0.0
	for i := 0; i < raw.Rows; i++ {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for _, v := range row {
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}

func TestDenseNormMatchesUnscaledOnOrdinaryInputs(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {5, 5}, {12, 7}} {
		for _, mag := range []float64{1e-140, 1e-20, 1, 1e20, 1e150} {
			m := mat.NewDense(sz[0], sz[1], nil)
			for i := range sz[0] {
				for j := range sz[1] {
					m.Set(i, j, mag*rng.NormFloat64())
				}
			}
			if got, want := denseNorm(m), unscaledFrobenius(m); got != want {
				t.Errorf("%v·%g: denseNorm = %v, unscaled = %v", sz, mag, got, want)
			}
			sub := m.Slice(0, sz[0], 0, (sz[1]+1)/2).(*mat.Dense)
			if got, want := denseNorm(sub), unscaledFrobenius(sub); got != want {
				t.Errorf("%v·%g strided: denseNorm = %v, unscaled = %v", sz, mag, got, want)
			}
		}
	}
}

func TestDenseNormExtremeMagnitudes(t *testing.T) {
	base := mat.NewDense(2, 3, []float64{3, -4, 12, 0, 2, -1})
	baseNorm := math.Sqrt(174)
	for _, e := range []int{-1074 + 4, -1060, -1000, -600, -520, 515, 520, 600, 1000, 1019} {
		m := mat.NewDense(2, 3, nil)
		m.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, e) }, base)
		got := math.Ldexp(denseNorm(m), -e)
		tol := 4 * eps() * baseNorm
		if e < -1022 {
			tol = baseNorm * math.Ldexp(1, -1074-e) * 8
		}
		if math.Abs(got-baseNorm) > tol || math.IsInf(got, 0) || got == 0 {
			t.Errorf("2^%d scale: denseNorm·2^%d = %v, want %v", e, -e, got, baseNorm)
		}
	}

	ones := mat.NewDense(3, 4, nil)
	ones.Apply(func(_, _ int, _ float64) float64 { return 1e200 }, ones)
	if got, want := denseNorm(ones), 1e200*math.Sqrt(12); math.Abs(got-want) > 4*eps()*want {
		t.Errorf("all 1e200 3×4: got %v, want %v", got, want)
	}
	if got := denseNorm(mat.NewDense(1, 2, []float64{math.MaxFloat64, math.MaxFloat64})); !math.IsInf(got, 1) {
		t.Errorf("norm above MaxFloat64: got %v, want +Inf", got)
	}
}

func TestDenseNormNonFiniteAndZero(t *testing.T) {
	cases := []struct {
		data []float64
		want float64
	}{
		{[]float64{0, 0, 0, 0}, 0},
		{[]float64{1, math.Inf(-1), 2, 3}, math.Inf(1)},
		{[]float64{1e300, math.NaN(), 2, 3}, math.NaN()},
		{[]float64{math.Inf(1), math.NaN(), 2, 3}, math.NaN()},
		{[]float64{math.SmallestNonzeroFloat64, 0, 0, 0}, math.SmallestNonzeroFloat64},
	}
	for _, c := range cases {
		got := denseNorm(mat.NewDense(2, 2, c.data))
		if got != c.want && !(math.IsNaN(got) && math.IsNaN(c.want)) {
			t.Errorf("%v: got %v, want %v", c.data, got, c.want)
		}
	}
}
