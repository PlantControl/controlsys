package controlsys

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func BenchmarkSimulateWithDelay_SISO(b *testing.B) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0.5})
	delay := mat.NewDense(1, 1, []float64{5})
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 100
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkSimulateWithDelay_MIMO(b *testing.B) {
	n, m, p := 10, 4, 6
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, 0.9-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < n && i < m; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p && i < n; i++ {
		C.Set(i, i, 1)
	}
	D := mat.NewDense(p, m, nil)

	delay := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			delay.Set(i, j, float64(2+i+j))
		}
	}
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 200
	u := mat.NewDense(m, steps, nil)
	for j := range m {
		for k := range steps {
			u.Set(j, k, math.Sin(float64(k)*0.1+float64(j)))
		}
	}
	x0 := mat.NewVecDense(n, nil)
	for i := range n {
		x0.SetVec(i, float64(i+1)*0.1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkBilinearDiscretize(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 3, nil)
	for i := range 3 {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, n-1, 1)
	D := mat.NewDense(2, 3, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Discretize(0.01)
	}
}

func BenchmarkTransferFunction_MIMO(b *testing.B) {
	n := 15
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	m, p := 3, 4
	B := mat.NewDense(n, m, nil)
	for i := range m {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := range p {
		C.Set(i, i%n, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.TransferFunction(nil)
	}
}

func BenchmarkReduce(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.2)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 2, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 1, 1)
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Reduce(nil)
	}
}

func BenchmarkAbsorbDelay(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := range 5 {
		A.Set(i, i, 0.8-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(5, 3, nil)
	for i := range 3 {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(2, 5, nil)
	C.Set(0, 0, 1)
	C.Set(1, 4, 1)
	D := mat.NewDense(2, 3, nil)
	delay := mat.NewDense(2, 3, []float64{3, 5, 3, 5, 3, 5})
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.AbsorbDelay()
	}
}

func BenchmarkDiscretizeZOH(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	m := 5
	B := mat.NewDense(n, m, nil)
	for i := range m {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(3, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 5, 1)
	C.Set(2, n-1, 1)
	D := mat.NewDense(3, m, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.DiscretizeZOH(0.01)
	}
}

func BenchmarkDenseNorm(b *testing.B) {
	m := mat.NewDense(50, 50, nil)
	for i := range 50 {
		for j := range 50 {
			m.Set(i, j, float64(i*50+j)*0.01)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		denseNorm(m)
	}
}

func BenchmarkSeries(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 4, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Series(sys1, sys2)
	}
}

func BenchmarkParallel(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 3, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Parallel(sys1, sys2)
	}
}

func BenchmarkFeedback(b *testing.B) {
	plant := benchSys(10, 3, 3)
	ctrl := benchSys(5, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Feedback(plant, ctrl, -1)
	}
}

func BenchmarkFeedbackLFT(b *testing.B) {
	plant := benchSys(10, 3, 3)
	_ = plant.SetOutputDelay([]float64{0.5, 1.0, 0.3})
	ctrl := benchSys(5, 3, 3)
	_ = ctrl.SetInputDelay([]float64{0.2, 0.4, 0.6})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Feedback(plant, ctrl, -1)
	}
}

func BenchmarkAppend(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Append(sys1, sys2)
	}
}

func BenchmarkFreqResponse(b *testing.B) {
	sys := benchSys(10, 2, 3)
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

func BenchmarkFreqResponse_ShortSweep(b *testing.B) {
	sys := benchSys(10, 2, 3)
	omega := logspace(-2, 2, 8)
	b.ResetTimer()
	for b.Loop() {
		sys.FreqResponse(omega)
	}
}

func BenchmarkFrequencySweepKernels(b *testing.B) {
	tests := []struct {
		name        string
		n, m, p, nw int
	}{
		{name: "N4_W2", n: 4, m: 2, p: 2, nw: 2},
		{name: "N4_W8", n: 4, m: 2, p: 2, nw: 8},
		{name: "N4_W32", n: 4, m: 2, p: 2, nw: 32},
		{name: "N4_W100", n: 4, m: 2, p: 2, nw: 100},
		{name: "N10_W2", n: 10, m: 2, p: 3, nw: 2},
		{name: "N10_W8", n: 10, m: 2, p: 3, nw: 8},
		{name: "N10_W32", n: 10, m: 2, p: 3, nw: 32},
		{name: "N10_W100", n: 10, m: 2, p: 3, nw: 100},
	}
	for _, test := range tests {
		sys := benchSys(test.n, test.m, test.p)
		evaluator := newFrequencyEvaluator(sys)
		omega := logspace(-2, 2, test.nw)
		size := test.nw * test.p * test.m
		b.Run(test.name+"/StateSpace", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.evalStateSpaceSweepInto(omega, data); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(test.name+"/TransferFunction", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.evalTransferFunctionSweepInto(omega, data); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBode(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Bode(nil, 200)
	}
}

func BenchmarkZeros_SISO(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := range 5 {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(5, 1, []float64{1, 0, 0, 0, 0})
	C := mat.NewDense(1, 5, []float64{0, 0, 0, 0, 1})
	D := mat.NewDense(1, 1, []float64{0.1})
	sys, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkZeros_MIMO(b *testing.B) {
	sys := benchSys(10, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkZeros_NonSquare(b *testing.B) {
	n, m, p := 15, 5, 8
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := range m {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := range p {
		C.Set(i, i%n, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkControllabilityStaircase(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.2)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 3, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	B.Set(2, 2, 1)
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ControllabilityStaircase(A, B, C, 0)
	}
}

func BenchmarkSimulateNoDelay(b *testing.B) {
	sys := benchSys(20, 3, 4)
	sys.Dt = 0.01
	steps := 500
	u := mat.NewDense(3, steps, nil)
	for j := range 3 {
		for k := range steps {
			u.Set(j, k, math.Sin(float64(k)*0.1+float64(j)))
		}
	}
	x0 := mat.NewVecDense(20, nil)
	for i := range 20 {
		x0.SetVec(i, float64(i)*0.05)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkThiranDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ThiranDelay(0.35, 3, 0.1)
	}
}

func BenchmarkPadeDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PadeDelay(0.5, 5)
	}
}

func BenchmarkDecomposeIODelay(b *testing.B) {
	delay := mat.NewDense(4, 3, []float64{
		3, 5, 2,
		4, 6, 3,
		3, 5, 2,
		5, 7, 4,
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DecomposeIODelay(delay)
	}
}

func BenchmarkPullDelaysToLFT(b *testing.B) {
	sys := benchSys(5, 3, 2)
	sys.InputDelay = []float64{0.3, 0.5, 0.1}
	sys.OutputDelay = []float64{0.2, 0.4}
	sys.Delay = mat.NewDense(2, 3, []float64{1, 2, 1, 2, 3, 2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.PullDelaysToLFT()
	}
}

func BenchmarkGetDelayModel(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.GetDelayModel()
	}
}

func BenchmarkSetDelayModel(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	H, tau := sys.GetDelayModel()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SetDelayModel(H, tau)
	}
}

func BenchmarkAbsorbInternalDelay(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := range 5 {
		A.Set(i, i, 0.8-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(5, 2, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	C := mat.NewDense(2, 5, nil)
	C.Set(0, 0, 1)
	C.Set(1, 4, 1)
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 1.0)
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.AbsorbDelay(AbsorbInternal)
	}
}

func BenchmarkAbsorbInternalDelayContinuous(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := range 5 {
		A.Set(i, i, -1-float64(i)*0.2)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(5, 2, []float64{1, 0, 0, 1, 0.2, 0, 0, 0.1, 0.3, 0})
	C := mat.NewDense(2, 5, []float64{1, 0, 0.2, 0, 0, 0, 0, 0.1, 0, 1})
	sys, _ := New(A, B, C, mat.NewDense(2, 2, nil), 0)
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	sys.SetInternalDelay([]float64{0.3, 0.7}, B2, C2, mat.NewDense(2, 2, nil), mat.NewDense(2, 2, nil), mat.NewDense(2, 2, []float64{0, 0.1, 0.2, 0}))
	b.ReportAllocs()
	for b.Loop() {
		sys.AbsorbDelay(AbsorbInternal)
	}
}

func BenchmarkDiscretizeWithOpts_Thiran(b *testing.B) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiscretizeWithOpts_IODelayThiran(b *testing.B) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.35})
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiscretizeWithOpts_PathThiran(b *testing.B) {
	for _, size := range []struct{ n, io int }{{4, 2}, {8, 4}} {
		sys := benchSys(size.n, size.io, size.io)
		sys.Delay = mat.NewDense(size.io, size.io, nil)
		for i := range size.io {
			sys.Delay.Set(i, i, 0.035+0.02*float64(i))
		}
		for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			b.Run(fmt.Sprintf("n=%d/io=%d/%s", size.n, size.io, modeling), func(b *testing.B) {
				opts := C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling}
				b.ReportAllocs()
				for b.Loop() {
					if _, err := sys.DiscretizeWithOpts(0.1, opts); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkFeedbackApproximatedDelays(b *testing.B) {
	plant := benchSys(10, 3, 3)
	plant.Dt = 1.0
	ctrl := benchSys(5, 3, 3)
	ctrl.Dt = 1.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Feedback(plant, ctrl, -1, WithApproximatedDelays())
	}
}

func BenchmarkSimulateInternalDelay(b *testing.B) {
	A := mat.NewDense(3, 3, []float64{0.9, 0.1, 0, 0, 0.8, 0.1, 0, 0, 0.7})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)
	B2 := mat.NewDense(3, 1, []float64{0.5, 0.3, 0.1})
	C2 := mat.NewDense(1, 3, []float64{0.2, 0.4, 0})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)

	steps := 200
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.1))
	}
	x0 := mat.NewVecDense(3, []float64{1, 0, 0})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkFreqResponseWithDelay(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

func BenchmarkPadeSystem(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Pade(3)
	}
}

func BenchmarkZeroDelayApprox(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.ZeroDelayApprox()
	}
}

func BenchmarkIsStrictlyUpperTriangular(b *testing.B) {
	n := 20
	m := mat.NewDense(n, n, nil)
	for i := range n {
		for j := i + 1; j < n; j++ {
			m.Set(i, j, float64(i*n+j)*0.01)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isStrictlyUpperTriangular(m)
	}
}

func BenchmarkFreqResponseLFT(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

// DC motor (MATLAB standard example, L=0 simplification)
// States: [theta, omega], Input: voltage, Output: angle
func BenchmarkSimulate_DCMotor(b *testing.B) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, 0, -10.01}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	disc, _ := sys.DiscretizeZOH(0.001)
	steps := 1000
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		disc.Simulate(u, nil, nil)
	}
}

// Boeing 747 lateral-directional (Franklin, Powell & Emami-Naeini)
// States: [beta, yaw rate, roll rate, roll angle]
func BenchmarkFreqResponse_B747Lateral(b *testing.B) {
	sys, _ := New(
		mat.NewDense(4, 4, []float64{
			-0.0558, -0.9968, 0.0802, 0.0415,
			0.598, -0.115, -0.0318, 0,
			-3.05, 0.388, -0.4650, 0,
			0, 0.0805, 1, 0,
		}),
		mat.NewDense(4, 2, []float64{
			0.00729, 0, -0.475, 0.00775,
			0.153, 0.143, 0, 0,
		}),
		mat.NewDense(2, 4, []float64{
			0, 1, 0, 0,
			1, 0, 0, 0,
		}),
		mat.NewDense(2, 2, nil),
		0,
	)
	omega := logspace(-3, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

// Mass-spring-damper chain (4 masses, k=10, c=0.5, m=1)
// 8 states, SISO, from Skogestad & Postlethwaite
func BenchmarkSimulate_MassSpringDamper(b *testing.B) {
	sys, _ := New(
		mat.NewDense(8, 8, []float64{
			0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 1,
			-20, 10, 0, 0, -1, 0.5, 0, 0,
			10, -20, 10, 0, 0.5, -1, 0.5, 0,
			0, 10, -20, 10, 0, 0.5, -1, 0.5,
			0, 0, 10, -10, 0, 0, 0.5, -0.5,
		}),
		mat.NewDense(8, 1, []float64{0, 0, 0, 0, 1, 0, 0, 0}),
		mat.NewDense(1, 8, []float64{0, 0, 0, 1, 0, 0, 0, 0}),
		mat.NewDense(1, 1, nil),
		0,
	)
	disc, _ := sys.DiscretizeZOH(0.01)
	steps := 500
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.05))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		disc.Simulate(u, nil, nil)
	}
}

// Boeing 747 longitudinal (Ogata) - discretize + simulate workflow
func BenchmarkDiscretizeAndSimulate_B747Longitudinal(b *testing.B) {
	sys, _ := New(
		mat.NewDense(4, 4, []float64{
			-0.003, 0.039, 0, -0.322,
			-0.065, -0.319, 7.74, 0,
			0.0201, -0.101, -0.429, 0,
			0, 0, 1, 0,
		}),
		mat.NewDense(4, 2, []float64{
			0.01, 1, -0.18, -0.04,
			-1.16, 0.598, 0, 0,
		}),
		mat.NewDense(2, 4, []float64{
			1, 0, 0, 0,
			0, 0, 0, 1,
		}),
		mat.NewDense(2, 2, nil),
		0,
	)
	disc, _ := sys.DiscretizeZOH(0.05)
	steps := 200
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		if k < 50 {
			u.Set(0, k, -0.01)
		}
	}
	x0 := mat.NewVecDense(4, []float64{0, 0, 0, 0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		disc.Simulate(u, x0, nil)
	}
}

// Large system simulation (50 states) - scalability test
func BenchmarkSimulate_Large(b *testing.B) {
	n, m, p := 50, 5, 5
	sys := benchSys(n, m, p)
	sys.Dt = 0.01
	steps := 1000
	u := mat.NewDense(m, steps, nil)
	for j := range m {
		for k := range steps {
			u.Set(j, k, math.Sin(float64(k)*0.02+float64(j)))
		}
	}
	x0 := mat.NewVecDense(n, nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

// Feedback + simulate pipeline (typical design workflow)
func BenchmarkFeedbackAndSimulate(b *testing.B) {
	plant, _ := New(
		mat.NewDense(4, 4, []float64{
			-0.0558, -0.9968, 0.0802, 0.0415,
			0.598, -0.115, -0.0318, 0,
			-3.05, 0.388, -0.4650, 0,
			0, 0.0805, 1, 0,
		}),
		mat.NewDense(4, 2, []float64{
			0.00729, 0, -0.475, 0.00775,
			0.153, 0.143, 0, 0,
		}),
		mat.NewDense(2, 4, []float64{
			0, 1, 0, 0,
			1, 0, 0, 0,
		}),
		mat.NewDense(2, 2, nil),
		0,
	)
	ctrl, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{-0.5, 0, 0, -1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	cl, _ := Feedback(plant, ctrl, -1)
	disc, _ := cl.DiscretizeZOH(0.05)
	steps := 200
	u := mat.NewDense(2, steps, nil)
	for k := range 50 {
		u.Set(0, k, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		disc.Simulate(u, nil, nil)
	}
}

// Bode of large MIMO system (scalability)
func BenchmarkBode_LargeMIMO(b *testing.B) {
	sys := benchSys(30, 5, 5)
	omega := logspace(-2, 3, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

// Lyapunov & Riccati benchmarks

func benchStableA(n int) *mat.Dense {
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 0.2)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	return A
}

func benchDiscreteA(n int) *mat.Dense {
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, 0.5+0.3*float64(i)/float64(n))
		if i > 0 {
			A.Set(i, i-1, 0.05)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.03)
		}
	}
	return A
}

func benchSymPD(n int) *mat.Dense {
	Q := mat.NewDense(n, n, nil)
	for i := range n {
		Q.Set(i, i, 2+0.1*float64(i))
		if i > 0 {
			Q.Set(i, i-1, 0.1)
			Q.Set(i-1, i, 0.1)
		}
	}
	return Q
}

func benchB(n, m int) *mat.Dense {
	B := mat.NewDense(n, m, nil)
	for i := range min(n, m) {
		B.Set(i, i, 1)
	}
	return B
}

func benchLyap(b *testing.B, n int) {
	A := benchStableA(n)
	Q := benchSymPD(n)
	b.ResetTimer()
	for range b.N {
		Lyap(A, Q, nil)
	}
}

func BenchmarkLyap_N10(b *testing.B)  { benchLyap(b, 10) }
func BenchmarkLyap_N50(b *testing.B)  { benchLyap(b, 50) }
func BenchmarkLyap_N100(b *testing.B) { benchLyap(b, 100) }

func benchDLyap(b *testing.B, n int) {
	A := benchDiscreteA(n)
	Q := benchSymPD(n)
	b.ResetTimer()
	for range b.N {
		DLyap(A, Q, nil)
	}
}

func BenchmarkDLyap_N10(b *testing.B)  { benchDLyap(b, 10) }
func BenchmarkDLyap_N50(b *testing.B)  { benchDLyap(b, 50) }
func BenchmarkDLyap_N100(b *testing.B) { benchDLyap(b, 100) }

func benchCare(b *testing.B, n, m int) {
	A := benchStableA(n)
	B := benchB(n, m)
	Q := benchSymPD(n)
	R := mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1)
	}
	ws := NewRiccatiWorkspace(n, m)
	opts := &RiccatiOpts{Workspace: ws}
	b.ResetTimer()
	for range b.N {
		Care(A, B, Q, R, opts)
	}
}

func BenchmarkCare_N10_M2(b *testing.B)  { benchCare(b, 10, 2) }
func BenchmarkCare_N50_M5(b *testing.B)  { benchCare(b, 50, 5) }
func BenchmarkCare_N100_M5(b *testing.B) { benchCare(b, 100, 5) }

func benchDare(b *testing.B, n, m int) {
	A := benchDiscreteA(n)
	B := benchB(n, m)
	Q := benchSymPD(n)
	R := mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1)
	}
	ws := NewRiccatiWorkspace(n, m)
	opts := &RiccatiOpts{Workspace: ws}
	b.ResetTimer()
	for range b.N {
		Dare(A, B, Q, R, opts)
	}
}

func BenchmarkDare_N10_M2(b *testing.B)  { benchDare(b, 10, 2) }
func BenchmarkDare_N50_M5(b *testing.B)  { benchDare(b, 50, 5) }
func BenchmarkDare_N100_M5(b *testing.B) { benchDare(b, 100, 5) }

func BenchmarkGram(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		Gram(sys, GramControllability)
	}
}

func BenchmarkCtrb(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		Ctrb(sys.A, sys.B)
	}
}

func BenchmarkH2Norm(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		H2Norm(sys)
	}
}

func BenchmarkHSV(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		HSV(sys)
	}
}

func BenchmarkHinfNorm(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		HinfNorm(sys)
	}
}

func BenchmarkH2Syn_Simple(b *testing.B) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -1})
	B := mat.NewDense(2, 2, []float64{0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 2, []float64{
		0, 0,
		0, 1,
		0.1, 0,
	})
	P, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for b.Loop() {
		H2Syn(P, 1, 1)
	}
}

func BenchmarkHinfSyn_Simple(b *testing.B) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})
	P, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for b.Loop() {
		HinfSyn(P, 1, 1)
	}
}

func BenchmarkHinfSyn_D11(b *testing.B) {
	for _, bc := range []struct {
		name         string
		P            *System
		nmeas, ncont int
	}{
		{"siso", sisoBiproperMixedSensitivityPlant(b), 1, 1},
		{"mimo", mimoBiproperMixedSensitivityPlant(b), 2, 2},
	} {
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				if _, err := HinfSyn(bc.P, bc.nmeas, bc.ncont); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkBalreal(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		Balreal(sys)
	}
}

func BenchmarkBalred(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		Balred(sys, 5, Truncate)
	}
}

// --- Controller & Observer Benchmarks ---

func benchPlace(b *testing.B, n, m int) {
	A := benchStableA(n)
	B := benchB(n, m)
	poles := make([]complex128, n)
	for i := range n {
		poles[i] = complex(-float64(i+1)*2, 0)
	}
	b.ResetTimer()
	for range b.N {
		Place(A, B, poles)
	}
}

func BenchmarkPlace_N10_M2(b *testing.B)  { benchPlace(b, 10, 2) }
func BenchmarkPlace_N50_M5(b *testing.B)  { benchPlace(b, 50, 5) }
func BenchmarkPlace_N100_M5(b *testing.B) { benchPlace(b, 100, 5) }

func benchAcker(b *testing.B, n int) {
	A := benchStableA(n)
	bCol := mat.NewDense(n, 1, nil)
	bCol.Set(0, 0, 1)
	poles := make([]complex128, n)
	for i := range n {
		poles[i] = complex(-float64(i+1)*2, 0)
	}
	b.ResetTimer()
	for range b.N {
		Acker(A, bCol, poles)
	}
}

func BenchmarkAcker_N5(b *testing.B)  { benchAcker(b, 5) }
func BenchmarkAcker_N10(b *testing.B) { benchAcker(b, 10) }

func benchLqr(b *testing.B, n, m int) {
	A := benchStableA(n)
	B := benchB(n, m)
	Q := benchSymPD(n)
	R := mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1)
	}
	ws := NewRiccatiWorkspace(n, m)
	opts := &RiccatiOpts{Workspace: ws}
	b.ResetTimer()
	for range b.N {
		Lqr(A, B, Q, R, opts)
	}
}

func BenchmarkLqr_N10_M2(b *testing.B)  { benchLqr(b, 10, 2) }
func BenchmarkLqr_N50_M5(b *testing.B)  { benchLqr(b, 50, 5) }
func BenchmarkLqr_N100_M5(b *testing.B) { benchLqr(b, 100, 5) }

func benchKalman(b *testing.B, n, m, p int) {
	sys := benchSys(n, m, p)
	Qn := mat.NewDense(m, m, nil)
	for i := range m {
		Qn.Set(i, i, 1)
	}
	Rn := mat.NewDense(p, p, nil)
	for i := range p {
		Rn.Set(i, i, 1)
	}
	ws := NewRiccatiWorkspace(n, p)
	opts := &RiccatiOpts{Workspace: ws}
	b.ResetTimer()
	for range b.N {
		Kalman(sys, Qn, Rn, opts)
	}
}

func BenchmarkKalman_N10_M2_P3(b *testing.B)  { benchKalman(b, 10, 2, 3) }
func BenchmarkKalman_N50_M5_P5(b *testing.B)  { benchKalman(b, 50, 5, 5) }
func BenchmarkKalman_N100_M5_P5(b *testing.B) { benchKalman(b, 100, 5, 5) }

func benchEstim(b *testing.B, n, m, p int) {
	sys := benchSys(n, m, p)
	L := mat.NewDense(n, p, nil)
	for i := range min(n, p) {
		L.Set(i, i, 1)
	}
	b.ResetTimer()
	for range b.N {
		Estim(sys, L)
	}
}

func BenchmarkEstim_N10_M2_P3(b *testing.B)  { benchEstim(b, 10, 2, 3) }
func BenchmarkEstim_N50_M5_P5(b *testing.B)  { benchEstim(b, 50, 5, 5) }
func BenchmarkEstim_N100_M5_P5(b *testing.B) { benchEstim(b, 100, 5, 5) }

func benchReg(b *testing.B, n, m, p int) {
	sys := benchSys(n, m, p)
	K := mat.NewDense(m, n, nil)
	L := mat.NewDense(n, p, nil)
	for i := range min(m, n) {
		K.Set(i, i, 1)
	}
	for i := range min(n, p) {
		L.Set(i, i, 1)
	}
	b.ResetTimer()
	for range b.N {
		Reg(sys, K, L)
	}
}

func BenchmarkReg_N10_M2_P3(b *testing.B)  { benchReg(b, 10, 2, 3) }
func BenchmarkReg_N50_M5_P5(b *testing.B)  { benchReg(b, 50, 5, 5) }
func BenchmarkReg_N100_M5_P5(b *testing.B) { benchReg(b, 100, 5, 5) }

func BenchmarkPolyRoots_N10(b *testing.B) {
	p := Poly{1, -1}
	for i := 2; i <= 10; i++ {
		p = p.Mul(Poly{1, float64(-i)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Roots()
	}
}

func BenchmarkZPKEval_SISO(b *testing.B) {
	z, _ := NewZPK([]complex128{-1, -2, -3}, []complex128{-4, -5, -6, -7}, 2.0, 0)
	s := complex(0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.Eval(s)
	}
}

func BenchmarkZPKFreqResponse_SISO_100(b *testing.B) {
	z, _ := NewZPK([]complex128{-1, -2}, []complex128{-3, -4, -5}, 2.0, 0)
	omega := make([]float64, 100)
	for i := range omega {
		omega[i] = 0.01 * math.Pow(10, 4*float64(i)/99)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.FreqResponse(omega)
	}
}

func BenchmarkZPKToTF_SISO_N10(b *testing.B) {
	zeros := make([]complex128, 9)
	poles := make([]complex128, 10)
	for i := range zeros {
		zeros[i] = complex(float64(-i-1), 0)
	}
	for i := range poles {
		poles[i] = complex(float64(-i-11), 0)
	}
	z, _ := NewZPK(zeros, poles, 1.0, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.TransferFunction()
	}
}

func BenchmarkTFToZPK_SISO_N10(b *testing.B) {
	zeros := make([]complex128, 9)
	poles := make([]complex128, 10)
	for i := range zeros {
		zeros[i] = complex(float64(-i-1), 0)
	}
	for i := range poles {
		poles[i] = complex(float64(-i-11), 0)
	}
	z, _ := NewZPK(zeros, poles, 1.0, 0)
	tf, _ := z.TransferFunction()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tf.ZPK()
	}
}

func BenchmarkSSToZPK_SISO(b *testing.B) {
	sys := benchSys(5, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.ZPKModel(nil)
	}
}

func BenchmarkZPKToSS_SISO(b *testing.B) {
	z, _ := NewZPK([]complex128{-1, -2}, []complex128{-3, -4, -5}, 2.0, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.StateSpace(nil)
	}
}

func BenchmarkPolyFromComplexRoots_N20(b *testing.B) {
	roots := make([]complex128, 20)
	for i := range roots {
		roots[i] = complex(float64(-i-1), 0)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		polyFromComplexRoots(roots)
	}
}

func BenchmarkSigma_SISO(b *testing.B) {
	sys := benchSys(4, 1, 1)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Sigma(omega, 0)
	}
}

func BenchmarkSigma_MIMO(b *testing.B) {
	sys := benchSys(4, 2, 2)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Sigma(omega, 0)
	}
}

func BenchmarkNichols(b *testing.B) {
	sys := benchSys(4, 1, 1)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Nichols(omega, 0)
	}
}

func BenchmarkNyquist(b *testing.B) {
	sys := benchSys(4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Nyquist(nil, 500)
	}
}

func BenchmarkFreqRespEst_SISO(b *testing.B) {
	dt := 0.01
	N := 4096
	uData := make([]float64, N)
	yData := make([]float64, N)
	for i := range N {
		uData[i] = math.Sin(float64(i) * 0.1)
		yData[i] = 0.5 * uData[i]
	}
	u := mat.NewDense(1, N, uData)
	y := mat.NewDense(1, N, yData)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FreqRespEst(u, y, dt, nil)
	}
}

func BenchmarkFreqRespEst_MIMO(b *testing.B) {
	dt := 0.01
	N := 4096
	uData := make([]float64, 2*N)
	yData := make([]float64, 2*N)
	for i := range 2 * N {
		uData[i] = math.Sin(float64(i) * 0.1)
		yData[i] = 0.3 * uData[i]
	}
	u := mat.NewDense(2, N, uData)
	y := mat.NewDense(2, N, yData)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FreqRespEst(u, y, dt, nil)
	}
}

func benchSys(n, m, p int) *System {
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < min(n, m); i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < min(p, n); i++ {
		C.Set(i, i, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)
	return sys
}

func BenchmarkBlkDiag(b *testing.B) {
	s1 := benchSys(10, 3, 4)
	s2 := benchSys(8, 2, 3)
	s3 := benchSys(6, 4, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BlkDiag(s1, s2, s3)
	}
}

func BenchmarkConnect(b *testing.B) {
	s1 := benchSys(10, 3, 4)
	s2 := benchSys(8, 4, 3)
	aug, _ := BlkDiag(s1, s2)
	_, m, p := aug.Dims()
	Q := mat.NewDense(m, p, nil)
	for i := 0; i < min(4, min(m, p)); i++ {
		Q.Set(3+i, i, 1)
	}
	inputs := []int{0, 1, 2}
	outputs := []int{4, 5, 6}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Connect(aug, Q, inputs, outputs)
	}
}

func BenchmarkLFT(b *testing.B) {
	M := benchSys(10, 6, 6)
	Delta := benchSys(5, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LFT(M, Delta, 3, 3)
	}
}

func BenchmarkLFT_Large(b *testing.B) {
	M := benchSys(20, 12, 12)
	Delta := benchSys(10, 6, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LFT(M, Delta, 6, 6)
	}
}

func benchD2CSystem(n, m int, dt float64) *System {
	A := mat.NewDense(n, n, nil)
	for i := range n {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 0.3)
		}
	}
	B := mat.NewDense(n, m, nil)
	for j := 0; j < m && j < n; j++ {
		B.Set(j, j, 1)
	}
	C := mat.NewDense(1, n, nil)
	C.Set(0, 0, 1)
	D := mat.NewDense(1, m, nil)
	cont, _ := New(A, B, C, D, 0)
	disc, _ := cont.DiscretizeZOH(dt)
	return disc
}

func BenchmarkD2C_ZOH_N2(b *testing.B) {
	sys := benchD2CSystem(2, 1, 0.05)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.D2C(C2DMethodZOH)
	}
}

func BenchmarkD2C_ZOH_N5(b *testing.B) {
	sys := benchD2CSystem(5, 2, 0.05)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.D2C(C2DMethodZOH)
	}
}

func BenchmarkD2C_ZOH_N20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.D2C(C2DMethodZOH)
	}
}

func BenchmarkD2C_ZOH_N50(b *testing.B) {
	sys := benchD2CSystem(50, 10, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.D2C(C2DMethodZOH)
	}
}

func BenchmarkD2C_Tustin_N20(b *testing.B) {
	sys := benchD2CSystem(20, 5, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.D2C(C2DMethodTustin)
	}
}

func BenchmarkMatLog_N20(b *testing.B) {
	sys := benchD2CSystem(20, 1, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matLog(sys.A)
	}
}

func BenchmarkMatLog_N50(b *testing.B) {
	sys := benchD2CSystem(50, 1, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matLog(sys.A)
	}
}

func BenchmarkLFTExtract(b *testing.B) {
	for _, n := range []int{10, 50} {
		M := benchSys(n, 6, 6)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := LFT(M, nil, 3, 3); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAugstateDelayed(b *testing.B) {
	for _, n := range []int{10, 50} {
		sys := benchSys(n, 4, 4)
		delay := mat.NewDense(4, 4, nil)
		for i := range 4 {
			delay.Set(i, (i+1)%4, 0.1*float64(i+1))
		}
		if err := sys.SetDelay(delay); err != nil {
			b.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{0.1, 0, 0.2, 0}); err != nil {
			b.Fatal(err)
		}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := Augstate(sys); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
