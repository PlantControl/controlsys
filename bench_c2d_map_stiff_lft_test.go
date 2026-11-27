package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// c2dMapLFTSystem is the frozen row name with one internal delay of 0.05 s
// whose delay channel is excited (C2 ≠ 0) and fed back (B2 ≠ 0).
func c2dMapLFTSystem(t testing.TB, name string) *System {
	t.Helper()
	s := c2dMapRowSystem(t, name).Copy()
	n, _, _ := s.Dims()
	b2, c2 := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		b2.Set(i, 0, []float64{0.3, -0.2, 0.5, 0.1, -0.4}[i%5])
		c2.Set(0, i, []float64{0.25, 0.5, -0.4, 0.2, -0.1}[i%5])
	}
	if err := s.SetInternalDelay([]float64{0.05}, b2, c2, mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	return s
}

// BenchmarkC2DMapStiffLFT times C2D and C2DMap of stiff internal-delay
// descriptors on the stiff LFT split: C2DMap adds the riders [E A] through
// the split, the shares and the Schur vectors of the parts.
func BenchmarkC2DMapStiffLFT(b *testing.B) {
	for _, row := range []string{"cluster", "stfr/399"} {
		sys := c2dMapLFTSystem(b, row)
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
			opts := C2DOptions{Method: method}
			b.Run(row+"/"+string(method)+"/C2D", func(b *testing.B) {
				for b.Loop() {
					_, _ = sys.C2D(0.01, opts)
				}
			})
			b.Run(row+"/"+string(method)+"/C2DMap", func(b *testing.B) {
				for b.Loop() {
					_, _, _ = sys.C2DMap(0.01, opts)
				}
			})
		}
	}
}
