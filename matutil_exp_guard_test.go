package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestExpGuardedNonFiniteNormFillsNaN(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	for name, a := range map[string]*mat.Dense{
		"sum overflow":        mat.NewDense(2, 2, []float64{1e308, 1e308, -1e308, 1e308}),
		"Inf entry":           mat.NewDense(2, 2, []float64{1, 2, inf, 3}),
		"-Inf entry":          mat.NewDense(2, 2, []float64{1, -inf, 0, 3}),
		"NaN in small column": mat.NewDense(2, 2, []float64{nan, 1e3, 0, 1e3}),
		"NaN in large column": mat.NewDense(2, 2, []float64{1e3, nan, 1e3, 0}),
		"three by three":      mat.NewDense(3, 3, []float64{1, 2, 3, 0, 1e308, 4, 0, 1e308, 5}),
	} {
		t.Run(name, func(t *testing.T) {
			r, c := a.Dims()
			for label, dst := range map[string]*mat.Dense{"empty": {}, "sized": mat.NewDense(r, c, nil)} {
				returnsWithinDeadline(t, name+" "+label, func() { expGuarded(dst, a) })
				gotR, gotC := dst.Dims()
				if gotR != r || gotC != c {
					t.Fatalf("%s: dims %d,%d, want %d,%d", label, gotR, gotC, r, c)
				}
				for i := range r {
					for j := range c {
						if v := dst.At(i, j); !math.IsNaN(v) {
							t.Fatalf("%s: [%d,%d] = %g, want NaN", label, i, j, v)
						}
					}
				}
			}
		})
	}
}

// expGuarded relies on this: the gonum bump that drops it scans entries for NaN.
func TestDenseOneNormPropagatesNaNAtAnyPosition(t *testing.T) {
	for _, n := range []int{1, 2, 3, 5, 8} {
		for _, fill := range []float64{0, 1, -1e3, 1e300} {
			for pos := range n * n {
				data := make([]float64, n*n)
				for i := range data {
					data[i] = fill * float64(i%3-1)
				}
				data[pos] = math.NaN()
				if norm := mat.Norm(mat.NewDense(n, n, data), 1); !math.IsNaN(norm) {
					t.Errorf("n=%d fill=%g pos=%d: 1-norm = %g, want NaN", n, fill, pos, norm)
				}
			}
		}
	}
}

func TestExpGuardedNaNAtAnyPositionFillsNaN(t *testing.T) {
	for _, n := range []int{2, 3, 5} {
		for _, fill := range []float64{0, 1, -1e3, 1e300} {
			for pos := range n * n {
				data := make([]float64, n*n)
				for i := range data {
					data[i] = fill * float64(i%3-1)
				}
				data[pos] = math.NaN()
				a := mat.NewDense(n, n, data)
				var dst mat.Dense
				returnsWithinDeadline(t, "expGuarded", func() { expGuarded(&dst, a) })
				for _, v := range dst.RawMatrix().Data {
					if !math.IsNaN(v) {
						t.Errorf("n=%d fill=%g pos=%d: entry %g, want NaN", n, fill, pos, v)
						break
					}
				}
			}
		}
	}
}

func TestExpGuardedFiniteNormMatchesExpBits(t *testing.T) {
	for name, a := range map[string]*mat.Dense{
		"small":           mat.NewDense(2, 2, []float64{.001, .002, 0, -.003}),
		"scaled":          mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		"large stable":    mat.NewDense(3, 3, []float64{-700, 1, 0, 0, -3, 2, 1, 0, -5}),
		"one by one":      mat.NewDense(1, 1, []float64{-1e308}),
		"one by one -Inf": mat.NewDense(1, 1, []float64{math.Inf(-1)}),
		"one by one +Inf": mat.NewDense(1, 1, []float64{math.Inf(1)}),
		"norm near top":   mat.NewDense(2, 2, []float64{-8e307, 1, 8e307, -1}),
		"overflowed exp":  mat.NewDense(2, 2, []float64{1e4, 1, 0, -2}),
	} {
		t.Run(name, func(t *testing.T) {
			var got, want mat.Dense
			expGuarded(&got, a)
			want.Exp(a)
			r, c := a.Dims()
			for i := range r {
				for j := range c {
					g, w := got.At(i, j), want.At(i, j)
					if math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("[%d,%d] = %x, want %x", i, j, g, w)
					}
				}
			}
		})
	}
}

func BenchmarkExpGuarded(b *testing.B) {
	for _, tc := range []struct {
		name string
		n    int
	}{{"N2", 2}, {"N5", 5}, {"N20", 20}} {
		a := mat.NewDense(tc.n, tc.n, nil)
		for i := range tc.n {
			for j := range tc.n {
				a.Set(i, j, float64((i*7+j*3)%5)-2)
			}
			a.Set(i, i, -3)
		}
		b.Run(tc.name, func(b *testing.B) {
			var dst mat.Dense
			b.ReportAllocs()
			for b.Loop() {
				expGuarded(&dst, a)
			}
		})
	}
}
