package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func BenchmarkSimulateWithDelay_SISO(b *testing.B) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0.5})
	delay := mat.NewDense(1, 1, []float64{5})
	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		b.Fatal(err)
	}

	steps := 100
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
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
	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		b.Fatal(err)
	}

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
		if _, err := sys.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodTustin}); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.TransferFunction(nil); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Reduce(nil); err != nil {
			b.Fatal(err)
		}
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
	sys, err := NewWithDelay(A, B, C, D, delay, 1.0)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.AbsorbDelay(); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.C2D(0.01, C2DOptions{}); err != nil {
			b.Fatal(err)
		}
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
	sys1 := benchSys(b, 10, 3, 4)
	sys2 := benchSys(b, 8, 4, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Series(sys1, sys2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParallel(b *testing.B) {
	sys1 := benchSys(b, 10, 3, 4)
	sys2 := benchSys(b, 8, 3, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Parallel(sys1, sys2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFeedback(b *testing.B) {
	plant := benchSys(b, 10, 3, 3)
	ctrl := benchSys(b, 5, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Feedback(plant, ctrl, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFeedbackLFT(b *testing.B) {
	plant := benchSys(b, 10, 3, 3)
	if err := plant.SetOutputDelay([]float64{0.5, 1.0, 0.3}); err != nil {
		b.Fatal(err)
	}
	ctrl := benchSys(b, 5, 3, 3)
	if err := ctrl.SetInputDelay([]float64{0.2, 0.4, 0.6}); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Feedback(plant, ctrl, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAppend(b *testing.B) {
	sys1 := benchSys(b, 10, 3, 4)
	sys2 := benchSys(b, 8, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Append(sys1, sys2); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFreqResponse(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFreqResponse_ShortSweep(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	omega := logspace(-2, 2, 8)
	b.ResetTimer()
	for b.Loop() {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFrequencySweepKernels(b *testing.B) {
	tests := []struct {
		n, m, p, nw int
	}{
		{4, 2, 2, 8}, {4, 2, 2, 100},
		{8, 1, 1, 100}, {8, 2, 2, 100},
		{12, 1, 1, 100}, {12, 2, 3, 100}, {12, 4, 4, 100},
		{18, 1, 1, 100}, {18, 2, 2, 100}, {18, 4, 4, 100},
		{30, 1, 1, 100}, {30, 5, 5, 100},
		{48, 1, 1, 100}, {48, 2, 2, 100}, {64, 1, 1, 100}, {64, 4, 4, 100}, {100, 2, 2, 100},
	}
	for _, test := range tests {
		sys := benchDenseSys(b, test.n, test.m, test.p)
		evaluator := newFrequencyEvaluator(sys)
		omega := logspace(-2, 2, test.nw)
		size := test.nw * test.p * test.m
		name := fmt.Sprintf("N%d_M%d_W%d", test.n, test.m, test.nw)
		b.Run(name+"/Dense", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.sweepInto(omega, data, newBalancedDense(sys, test.n, test.m, test.p)); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/Hessenberg", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.sweepInto(omega, data, newHessenbergSweep(sys, test.n, test.m, test.p)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkFrequencyDispatch times both delay-free sweep kernels around the
// useDenseSweep state threshold n = 8m+8 across grid lengths, for coupled and
// already upper-Hessenberg A. path=auto is the public FreqResponse call and
// includes validation and dispatch. Compare paths with benchstat -col /path.
func BenchmarkFrequencyDispatch(b *testing.B) {
	type dispatchCase struct {
		n, m, nw int
		hess     bool
	}
	var cases []dispatchCase
	for _, m := range []int{1, 2, 4} {
		c := 8*m + 8
		for _, n := range []int{c / 2, c, c + 4, 3 * c / 2, 2 * c} {
			for _, nw := range []int{1, 2, 3, 5, 10, 20, 50, 200} {
				cases = append(cases, dispatchCase{n: n, m: m, nw: nw})
			}
		}
		for _, n := range []int{c, 2 * c} {
			for _, nw := range []int{3, 20, 200} {
				cases = append(cases, dispatchCase{n: n, m: m, nw: nw, hess: true})
			}
		}
	}
	for _, tc := range cases {
		sys := benchDenseSys(b, tc.n, tc.m, tc.m)
		shape := "full"
		if tc.hess {
			shape = "hessenberg"
			raw := sys.A.RawMatrix()
			for i := 2; i < tc.n; i++ {
				clear(raw.Data[i*raw.Stride : i*raw.Stride+i-1])
			}
		}
		evaluator := newFrequencyEvaluator(sys)
		omega := logspace(-2, 2, tc.nw)
		size := tc.nw * tc.m * tc.m
		name := fmt.Sprintf("n=%d/m=%d/w=%d/a=%s", tc.n, tc.m, tc.nw, shape)
		b.Run(name+"/path=dense", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.sweepInto(omega, data, newBalancedDense(sys, tc.n, tc.m, tc.m)); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/path=hessenberg", func(b *testing.B) {
			for b.Loop() {
				data := make([]complex128, size)
				if err := evaluator.sweepInto(omega, data, newHessenbergSweep(sys, tc.n, tc.m, tc.m)); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(name+"/path=auto", func(b *testing.B) {
			for b.Loop() {
				if _, err := sys.FreqResponse(omega); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchDenseSys is a fully coupled stable model; banded benchSys lets GEPP
// skip zero multipliers and understates the dense per-point cost.
func benchDenseSys(tb testing.TB, n, m, p int) *System {
	tb.Helper()
	rng := rand.New(rand.NewPCG(uint64(n), uint64(m*16+p)))
	fill := func(r, c int, scale float64) *mat.Dense {
		out := mat.NewDense(r, c, nil)
		for i := range r {
			for j := range c {
				out.Set(i, j, rng.NormFloat64()*scale)
			}
		}
		return out
	}
	A := fill(n, n, 1/math.Sqrt(float64(n)))
	for i := range n {
		A.Set(i, i, A.At(i, i)-2)
	}
	sys, err := New(A, fill(n, m, 1), fill(p, n, 1), mat.NewDense(p, m, nil), 0)
	if err != nil {
		tb.Fatal(err)
	}
	return sys
}

func BenchmarkBode(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Bode(nil, 200); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Zeros(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZeros_MIMO(b *testing.B) {
	sys := benchSys(b, 10, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Zeros(); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Zeros(); err != nil {
			b.Fatal(err)
		}
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
		mustStaircase(b, A, B, C)
	}
}

func BenchmarkSimulateNoDelay(b *testing.B) {
	sys := benchSys(b, 20, 3, 4)
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
		if _, err := sys.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkThiranDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := thiranDelay(0.35, 3, 0.1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPadeDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PadeDelay(0.5, 5); err != nil {
			b.Fatal(err)
		}
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
		decomposeIODelay(delay)
	}
}

func BenchmarkPullDelaysToLFT(b *testing.B) {
	sys := benchSys(b, 5, 3, 2)
	sys.InputDelay = []float64{0.3, 0.5, 0.1}
	sys.OutputDelay = []float64{0.2, 0.4}
	sys.Delay = mat.NewDense(2, 3, []float64{1, 2, 1, 2, 3, 2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.PullDelaysToLFT(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGetDelayModel(b *testing.B) {
	sys := benchSys(b, 5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := sys.GetDelayModel(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSetDelayModel(b *testing.B) {
	sys := benchSys(b, 5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	H, tau, err := sys.GetDelayModel()
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SetDelayModel(H, tau); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		b.Fatal(err)
	}
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	if err := sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.AbsorbDelay(AbsorbInternal); err != nil {
			b.Fatal(err)
		}
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
	sys, err := New(A, B, C, mat.NewDense(2, 2, nil), 0)
	if err != nil {
		b.Fatal(err)
	}
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	if err := sys.SetInternalDelay([]float64{0.3, 0.7}, B2, C2, mat.NewDense(2, 2, nil), mat.NewDense(2, 2, nil), mat.NewDense(2, 2, []float64{0, 0.1, 0.2, 0})); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.AbsorbDelay(AbsorbInternal); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiscretizeWithOpts_Thiran(b *testing.B) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	sys.InputDelay = []float64{0.35}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiscretizeWithOpts_IODelayThiran(b *testing.B) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	sys.Delay = mat.NewDense(1, 1, []float64{0.35})
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDiscretizeWithOpts_PathThiran(b *testing.B) {
	for _, size := range []struct{ n, io int }{{4, 2}, {8, 4}} {
		sys := benchSys(b, size.n, size.io, size.io)
		sys.Delay = mat.NewDense(size.io, size.io, nil)
		for i := range size.io {
			sys.Delay.Set(i, i, 0.035+0.02*float64(i))
		}
		for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			b.Run(fmt.Sprintf("n=%d/io=%d/%s", size.n, size.io, modeling), func(b *testing.B) {
				opts := C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling}
				b.ReportAllocs()
				for b.Loop() {
					if _, err := sys.C2D(0.1, opts); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkFeedbackAbsorbDelay(b *testing.B) {
	plant := benchSys(b, 10, 3, 3)
	plant.Dt = 1.0
	ctrl := benchSys(b, 5, 3, 3)
	ctrl.Dt = 1.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cl, err := Feedback(plant, ctrl, -1)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := cl.AbsorbDelay(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSimulateInternalDelay(b *testing.B) {
	A := mat.NewDense(3, 3, []float64{0.9, 0.1, 0, 0, 0.8, 0.1, 0, 0, 0.7})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, err := New(A, B, C, D, 1.0)
	if err != nil {
		b.Fatal(err)
	}
	B2 := mat.NewDense(3, 1, []float64{0.5, 0.3, 0.1})
	C2 := mat.NewDense(1, 3, []float64{0.2, 0.4, 0})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	if err := sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22); err != nil {
		b.Fatal(err)
	}

	steps := 200
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.1))
	}
	x0 := mat.NewVecDense(3, []float64{1, 0, 0})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFreqResponseWithDelay(b *testing.B) {
	sys := benchSys(b, 5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPadeSystem(b *testing.B) {
	sys := benchSys(b, 5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Pade(3); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZeroDelayApprox(b *testing.B) {
	sys := benchSys(b, 5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	if err := sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.ZeroDelayApprox(); err != nil {
			b.Fatal(err)
		}
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
	sys := benchSys(b, 5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	if err := sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22); err != nil {
		b.Fatal(err)
	}
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

// DC motor (MATLAB standard example, L=0 simplification)
// States: [theta, omega], Input: voltage, Output: angle
func BenchmarkSimulate_DCMotor(b *testing.B) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, 0, -10.01}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	disc, err := sys.C2D(0.001, C2DOptions{})
	if err != nil {
		b.Fatal(err)
	}
	steps := 1000
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := disc.Simulate(u, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Boeing 747 lateral-directional (Franklin, Powell & Emami-Naeini)
// States: [beta, yaw rate, roll rate, roll angle]
func BenchmarkFreqResponse_B747Lateral(b *testing.B) {
	sys, err := New(
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
	if err != nil {
		b.Fatal(err)
	}
	omega := logspace(-3, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

// Mass-spring-damper chain (4 masses, k=10, c=0.5, m=1)
// 8 states, SISO, from Skogestad & Postlethwaite
func BenchmarkSimulate_MassSpringDamper(b *testing.B) {
	sys, err := New(
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
	if err != nil {
		b.Fatal(err)
	}
	disc, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		b.Fatal(err)
	}
	steps := 500
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(float64(k)*0.05))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := disc.Simulate(u, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Boeing 747 longitudinal (Ogata) - discretize + simulate workflow
func BenchmarkDiscretizeAndSimulate_B747Longitudinal(b *testing.B) {
	sys, err := New(
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
	if err != nil {
		b.Fatal(err)
	}
	disc, err := sys.C2D(0.05, C2DOptions{})
	if err != nil {
		b.Fatal(err)
	}
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
		if _, err := disc.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Large system simulation (50 states) - scalability test
func BenchmarkSimulate_Large(b *testing.B) {
	n, m, p := 50, 5, 5
	sys := benchSys(b, n, m, p)
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
		if _, err := sys.Simulate(u, x0, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Feedback + simulate pipeline (typical design workflow)
func BenchmarkFeedbackAndSimulate(b *testing.B) {
	plant, err := New(
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
	if err != nil {
		b.Fatal(err)
	}
	ctrl, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{-0.5, 0, 0, -1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	cl, err := Feedback(plant, ctrl, -1)
	if err != nil {
		b.Fatal(err)
	}
	disc, err := cl.C2D(0.05, C2DOptions{})
	if err != nil {
		b.Fatal(err)
	}
	steps := 200
	u := mat.NewDense(2, steps, nil)
	for k := range 50 {
		u.Set(0, k, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := disc.Simulate(u, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// Bode of large MIMO system (scalability)
func BenchmarkBode_LargeMIMO(b *testing.B) {
	sys := benchSys(b, 30, 5, 5)
	omega := logspace(-2, 3, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
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
		if _, err := Lyap(A, Q, nil); err != nil {
			b.Fatal(err)
		}
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
		if _, err := DLyap(A, Q, nil); err != nil {
			b.Fatal(err)
		}
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
		if _, err := Care(A, B, Q, R, opts); err != nil {
			b.Fatal(err)
		}
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
		if _, err := Dare(A, B, Q, R, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDare_N10_M2(b *testing.B)  { benchDare(b, 10, 2) }
func BenchmarkDare_N50_M5(b *testing.B)  { benchDare(b, 50, 5) }
func BenchmarkDare_N100_M5(b *testing.B) { benchDare(b, 100, 5) }

func BenchmarkGram(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, err := Gram(sys, GramControllability); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCtrb(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, err := Ctrb(sys.A, sys.B); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkH2Norm(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, err := H2Norm(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHSV(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, err := HSV(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHinfNorm(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := HinfNorm(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHinfNorm_MixedSensitivityLoop(b *testing.B) {
	for _, order := range []int{10, 20, 30} {
		cl := hinfMixedSensitivityLoop(b, order, 7)
		n, _, _ := cl.Dims()
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := HinfNorm(cl); err != nil {
					b.Fatal(err)
				}
			}
		})
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
	P, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := H2Syn(P, 1, 1); err != nil {
			b.Fatal(err)
		}
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
	P, err := New(A, B, C, D, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := HinfSyn(P, 1, 1); err != nil {
			b.Fatal(err)
		}
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
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, err := Balreal(sys); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBalred(b *testing.B) {
	sys := benchSys(b, 10, 2, 3)
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := Balred(sys, 5, BalredOptions{StateProjection: Truncate}); err != nil {
			b.Fatal(err)
		}
	}
}

// --- Controller & Observer Benchmarks ---

type placeFixture func(n, m int) (A, B *mat.Dense, poles []complex128)

func benchPlace(b *testing.B, fixture placeFixture, n, m int) {
	A, B, poles := fixture(n, m)
	b.ResetTimer()
	for range b.N {
		if _, err := Place(A, B, poles); err != nil {
			b.Fatal(err)
		}
	}
}

func placeChainFixture(n, m int) (A, B *mat.Dense, poles []complex128) {
	poles = make([]complex128, n)
	for i := range n {
		poles[i] = complex(-float64(i+1)*2, 0)
	}
	return benchStableA(n), benchB(n, m), poles
}

// placeSpreadFixture drives every state through a dense B. With B only on the
// first states of the benchStableA chain, far-state controllability decays
// like 0.2^(n-m) and is lost in double precision for large n.
func placeSpreadFixture(n, m int) (A, B *mat.Dense, poles []complex128) {
	rng := newPlaceRNG(5)
	B = mat.NewDense(n, m, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng())
		}
	}
	poles = make([]complex128, n)
	for i := range n {
		poles[i] = complex(-float64(i+1)*0.3-1, 0)
	}
	return benchStableA(n), B, poles
}

func BenchmarkPlace_N10_M2(b *testing.B)  { benchPlace(b, placeChainFixture, 10, 2) }
func BenchmarkPlace_N50_M5(b *testing.B)  { benchPlace(b, placeSpreadFixture, 50, 5) }
func BenchmarkPlace_N100_M5(b *testing.B) { benchPlace(b, placeSpreadFixture, 100, 5) }

func TestPlaceBenchFixturesAssignPoles(t *testing.T) {
	for _, tc := range []struct {
		name    string
		fixture placeFixture
		n, m    int
	}{
		{"chain/N10_M2", placeChainFixture, 10, 2},
		{"spread/N50_M5", placeSpreadFixture, 50, 5},
		{"spread/N100_M5", placeSpreadFixture, 100, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			A, B, poles := tc.fixture(tc.n, tc.m)
			K, err := Place(A, B, poles)
			if err != nil {
				t.Fatal(err)
			}
			var cl mat.Dense
			cl.Mul(B, K)
			cl.Sub(A, &cl)
			var eig mat.Eigen
			if !eig.Factorize(&cl, mat.EigenNone) {
				t.Fatal("closed-loop eigendecomposition failed")
			}
			for _, v := range eig.Values(nil) {
				best := math.Inf(1)
				for _, p := range poles {
					best = math.Min(best, cmplx.Abs(v-p)/cmplx.Abs(p))
				}
				if best > 1e-4 {
					t.Fatalf("closed-loop eigenvalue %v is %g from nearest target pole", v, best)
				}
			}
		})
	}
}

func benchPlaceRandom(b *testing.B, n, m int) {
	rng := newPlaceRNG(3)
	a := make([]float64, n*n)
	for i := range a {
		a[i] = rng()
	}
	bd := make([]float64, n*m)
	for i := range bd {
		bd[i] = rng()
	}
	A, B := mat.NewDense(n, n, a), mat.NewDense(n, m, bd)
	poles := make([]complex128, 0, n)
	for len(poles)+2 <= n {
		re := -0.5 - float64(len(poles))*0.1
		poles = append(poles, complex(re, 0.5), complex(re, -0.5))
	}
	if len(poles) < n {
		poles = append(poles, -1)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := Place(A, B, poles); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPlace_Random_N10_M3(b *testing.B) { benchPlaceRandom(b, 10, 3) }
func BenchmarkPlace_Random_N30_M4(b *testing.B) { benchPlaceRandom(b, 30, 4) }

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
		if _, err := Acker(A, bCol, poles); err != nil {
			b.Fatal(err)
		}
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
		if _, err := Lqr(A, B, Q, R, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLqr_N10_M2(b *testing.B)  { benchLqr(b, 10, 2) }
func BenchmarkLqr_N50_M5(b *testing.B)  { benchLqr(b, 50, 5) }
func BenchmarkLqr_N100_M5(b *testing.B) { benchLqr(b, 100, 5) }

func benchKalman(b *testing.B, n, m, p int) {
	sys := benchSys(b, n, m, p)
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
		if _, err := Kalman(sys, Qn, Rn, nil, opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkKalman_N10_M2_P3(b *testing.B)  { benchKalman(b, 10, 2, 3) }
func BenchmarkKalman_N50_M5_P5(b *testing.B)  { benchKalman(b, 50, 5, 5) }
func BenchmarkKalman_N100_M5_P5(b *testing.B) { benchKalman(b, 100, 5, 5) }

func benchEstim(b *testing.B, n, m, p int) {
	sys := benchSys(b, n, m, p)
	L := mat.NewDense(n, p, nil)
	for i := range min(n, p) {
		L.Set(i, i, 1)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := estimAll(sys, L); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEstim_N10_M2_P3(b *testing.B)  { benchEstim(b, 10, 2, 3) }
func BenchmarkEstim_N50_M5_P5(b *testing.B)  { benchEstim(b, 50, 5, 5) }
func BenchmarkEstim_N100_M5_P5(b *testing.B) { benchEstim(b, 100, 5, 5) }

func benchReg(b *testing.B, n, m, p int) {
	sys := benchSys(b, n, m, p)
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
		if _, err := Reg(sys, K, L); err != nil {
			b.Fatal(err)
		}
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
		if _, err := p.Roots(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZPKEval_SISO(b *testing.B) {
	z, err := NewZPK([]complex128{-1, -2, -3}, []complex128{-4, -5, -6, -7}, 2.0, 0)
	if err != nil {
		b.Fatal(err)
	}
	s := complex(0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := z.Eval(s); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZPKFreqResponse_SISO_100(b *testing.B) {
	z, err := NewZPK([]complex128{-1, -2}, []complex128{-3, -4, -5}, 2.0, 0)
	if err != nil {
		b.Fatal(err)
	}
	omega := make([]float64, 100)
	for i := range omega {
		omega[i] = 0.01 * math.Pow(10, 4*float64(i)/99)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := z.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZPKFreqResponseDiscrete_SISO_100(b *testing.B) {
	z, err := NewZPK([]complex128{0.5, -0.2}, []complex128{0.9, 0.6 + 0.3i, 0.6 - 0.3i}, 2.0, 0.1)
	if err != nil {
		b.Fatal(err)
	}
	omega := make([]float64, 100)
	for i := range omega {
		omega[i] = 0.01 * math.Pow(10, 3.4*float64(i)/99)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := z.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
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
	z, err := NewZPK(zeros, poles, 1.0, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := z.TransferFunction(); err != nil {
			b.Fatal(err)
		}
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
	z, err := NewZPK(zeros, poles, 1.0, 0)
	if err != nil {
		b.Fatal(err)
	}
	tf, err := z.TransferFunction()
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tf.ZPK(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSSToZPK_SISO(b *testing.B) {
	sys := benchSys(b, 5, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.ZPKModel(nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkZPKToSS_SISO(b *testing.B) {
	z, err := NewZPK([]complex128{-1, -2}, []complex128{-3, -4, -5}, 2.0, 0)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := z.StateSpace(); err != nil {
			b.Fatal(err)
		}
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
	sys := benchSys(b, 4, 1, 1)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Sigma(omega, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSigma_MIMO(b *testing.B) {
	sys := benchSys(b, 4, 2, 2)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Sigma(omega, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNichols(b *testing.B) {
	sys := benchSys(b, 4, 1, 1)
	omega := logspace(-1, 2, 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Nichols(omega, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNyquist(b *testing.B) {
	sys := benchSys(b, 4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.Nyquist(nil, 500); err != nil {
			b.Fatal(err)
		}
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
		if _, err := FreqRespEst(u, y, dt, nil); err != nil {
			b.Fatal(err)
		}
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
		if _, err := FreqRespEst(u, y, dt, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func benchSys(tb testing.TB, n, m, p int) *System {
	tb.Helper()
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
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		tb.Fatal(err)
	}
	return sys
}

func BenchmarkBlkDiag(b *testing.B) {
	s1 := benchSys(b, 10, 3, 4)
	s2 := benchSys(b, 8, 2, 3)
	s3 := benchSys(b, 6, 4, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := BlkDiag(s1, s2, s3); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConnect(b *testing.B) {
	s1 := benchSys(b, 10, 3, 4)
	s2 := benchSys(b, 8, 4, 3)
	aug, err := BlkDiag(s1, s2)
	if err != nil {
		b.Fatal(err)
	}
	_, m, p := aug.Dims()
	Q := mat.NewDense(m, p, nil)
	for i := 0; i < min(4, min(m, p)); i++ {
		Q.Set(3+i, i, 1)
	}
	inputs := []int{0, 1, 2}
	outputs := []int{4, 5, 6}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := connectGain("Connect", aug, Q, inputs, outputs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLFT(b *testing.B) {
	M := benchSys(b, 10, 6, 6)
	Delta := benchSys(b, 5, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LFT(M, Delta, LFTFeedback{Nu: 3, Ny: 3}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLFT_Large(b *testing.B) {
	M := benchSys(b, 20, 12, 12)
	Delta := benchSys(b, 10, 6, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LFT(M, Delta, LFTFeedback{Nu: 6, Ny: 6}); err != nil {
			b.Fatal(err)
		}
	}
}

func benchD2CSystem(tb testing.TB, n, m int, dt float64) *System {
	tb.Helper()
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
	cont, err := New(A, B, C, D, 0)
	if err != nil {
		tb.Fatal(err)
	}
	disc, err := cont.C2D(dt, C2DOptions{})
	if err != nil {
		tb.Fatal(err)
	}
	return disc
}

func BenchmarkD2C_ZOH_N2(b *testing.B) {
	sys := benchD2CSystem(b, 2, 1, 0.05)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkD2C_ZOH_N5(b *testing.B) {
	sys := benchD2CSystem(b, 5, 2, 0.05)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkD2C_ZOH_N20(b *testing.B) {
	sys := benchD2CSystem(b, 20, 5, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkD2C_ZOH_N50(b *testing.B) {
	sys := benchD2CSystem(b, 50, 10, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodZOH}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkD2C_Tustin_N20(b *testing.B) {
	sys := benchD2CSystem(b, 20, 5, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := sys.D2C(D2COptions{Method: C2DMethodTustin}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatLog_N20(b *testing.B) {
	sys := benchD2CSystem(b, 20, 1, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := matLog(sys.A); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatLog_N50(b *testing.B) {
	sys := benchD2CSystem(b, 50, 1, 0.01)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := matLog(sys.A); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLFTExtract(b *testing.B) {
	for _, n := range []int{10, 50} {
		M := benchSys(b, n, 6, 6)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := lftCloseExternal(M, nil, 3, 3); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAugstateDelayed(b *testing.B) {
	for _, n := range []int{10, 50} {
		sys := benchSys(b, n, 4, 4)
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

func BenchmarkEvalFr_N4(b *testing.B) {
	sys := benchSys(b, 4, 2, 2)
	for b.Loop() {
		if _, err := sys.EvalFr(1i); err != nil {
			b.Fatal(err)
		}
	}
}
