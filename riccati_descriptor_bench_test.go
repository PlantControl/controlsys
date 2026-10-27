package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func benchDescriptorE(n int) *mat.Dense {
	E := mat.NewDense(n, n, nil)
	for i := range n {
		E.Set(i, i, 1)
		for j := range n {
			E.Set(i, j, E.At(i, j)+0.05*float64((i*7+j*3)%5-2))
		}
	}
	return E
}

func benchRiccatiDescriptor(b *testing.B, continuous bool, n, m int) {
	A := benchStableA(n)
	if !continuous {
		A = benchDiscreteA(n)
	}
	B := benchB(n, m)
	Q := benchSymPD(n)
	R := eyeDense(m)
	opts := &RiccatiOpts{E: benchDescriptorE(n), Workspace: NewRiccatiWorkspace(n, m)}
	solve := Dare
	if continuous {
		solve = Care
	}
	b.ResetTimer()
	for range b.N {
		if _, err := solve(A, B, Q, R, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCareDescriptor_N10_M2(b *testing.B) { benchRiccatiDescriptor(b, true, 10, 2) }
func BenchmarkCareDescriptor_N50_M5(b *testing.B) { benchRiccatiDescriptor(b, true, 50, 5) }
func BenchmarkDareDescriptor_N10_M2(b *testing.B) { benchRiccatiDescriptor(b, false, 10, 2) }
func BenchmarkDareDescriptor_N50_M5(b *testing.B) { benchRiccatiDescriptor(b, false, 50, 5) }

// BenchmarkKalmanIllConditionedE times the Kalman filter of a plant whose
// E of condition 1e13 makes its gain cancel, so that it is refined.
func BenchmarkKalmanIllConditionedE(b *testing.B) {
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{0.19887662554059538, -0.14856103312523139, -1.0558847194781948, -0.83053605748214521}),
		mat.NewDense(2, 1, []float64{-1.5154567560715049, 1.7416748591014379}),
		mat.NewDense(1, 2, []float64{0.97228848391683376, -0.39447132160607812}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{0.99465977130926597, -1.5289113538341979, 0, 1.2867483309244827e-13}), 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := Kalman(sys, eyeDense(1), eyeDense(1), nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}
