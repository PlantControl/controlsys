package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func BenchmarkZZLyapCert4(b *testing.B) {
	A := mat.NewDense(4, 4, []float64{-1.5, 0.2, 0.1, -0.3, 0.1, -1.2, 0.4, 0.2, -0.2, 0.3, -1.7, 0.1, 0.05, -0.1, 0.2, -1.4})
	B := mat.NewDense(4, 1, []float64{1, -0.5, 0.3, 2})
	var Q mat.Dense
	Q.Mul(B, B.T())
	b.ReportAllocs()
	for range b.N {
		Lyap(A, &Q, nil)
	}
}

func BenchmarkZZLyapPlain4(b *testing.B) {
	A := mat.NewDense(4, 4, []float64{-1.5, 0.2, 0.1, -0.3, 0.1, -1.2, 0.4, 0.2, -0.2, 0.3, -1.7, 0.1, 0.05, -0.1, 0.2, -1.4})
	B := mat.NewDense(4, 1, []float64{1, -0.5, 0.3, 2})
	var Q mat.Dense
	Q.Mul(B, B.T())
	b.ReportAllocs()
	for range b.N {
		lyap(A, &Q, nil)
	}
}
