package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type residualCase struct {
	name  string
	sys   *System
	wantN int
}

func residualPlant(t *testing.T, n, m, p int, dt float64) *System {
	t.Helper()
	a := make([]float64, n*n)
	for i := range n {
		for j := range n {
			a[i*n+j] = 0.07 * float64((i*3+j*5)%7-3)
		}
		a[i*n+i] += 0.55 - 0.1*float64(i)
	}
	b := make([]float64, n*m)
	for k := range b {
		b[k] = 0.3 + 0.17*float64((k*5)%7)
	}
	c := make([]float64, p*n)
	for k := range c {
		c[k] = 0.4 - 0.13*float64((k*3)%5)
	}
	d := make([]float64, p*m)
	for k := range d {
		d[k] = 0.11 * float64(k%3+1)
	}
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func withDelays(t *testing.T, sys *System, in, out []float64, io *mat.Dense) *System {
	t.Helper()
	cp := sys.Copy()
	if in != nil {
		if err := cp.SetInputDelay(in); err != nil {
			t.Fatal(err)
		}
	}
	if out != nil {
		if err := cp.SetOutputDelay(out); err != nil {
			t.Fatal(err)
		}
	}
	if io != nil {
		if err := cp.SetDelay(io); err != nil {
			t.Fatal(err)
		}
	}
	return cp
}

// shiftedOracle simulates the delay-free realization per input and shifts each
// (i,j) channel by its total delay, sharing no code with the delay paths.
func shiftedOracle(t *testing.T, sys *System, j int, u []float64) []float64 {
	t.Helper()
	free := sys.Copy()
	free.Delay, free.InputDelay, free.OutputDelay = nil, nil, nil
	_, m, p := free.Dims()
	steps := len(u)
	uj := mat.NewDense(m, steps, nil)
	uj.SetRow(j, u)
	resp, err := free.Simulate(uj, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	total := sys.TotalDelay()
	y := make([]float64, p*steps)
	for i := range p {
		shift := int(math.Round(total.At(i, j)))
		for k := shift; k < steps; k++ {
			y[i*steps+k] = resp.Y.At(i, k-shift)
		}
	}
	return y
}

func assertMatchesOracle(t *testing.T, label string, got, ref *System) {
	t.Helper()
	_, m, p := ref.Dims()
	const steps = 40
	step := make([]float64, steps)
	impulse := make([]float64, steps)
	for k := range step {
		step[k] = 1
	}
	impulse[0] = 1
	for _, sig := range []struct {
		name string
		u    []float64
	}{{"step", step}, {"impulse", impulse}} {
		for j := range m {
			want := shiftedOracle(t, ref, j, sig.u)
			u := mat.NewDense(m, steps, nil)
			u.SetRow(j, sig.u)
			resp, err := got.Simulate(u, nil, nil)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			for i := range p {
				for k := range steps {
					if d := math.Abs(resp.Y.At(i, k) - want[i*steps+k]); d > 1e-12 {
						t.Fatalf("%s %s u%d->y%d k=%d: got %v want %v", label, sig.name, j, i, k, resp.Y.At(i, k), want[i*steps+k])
					}
				}
			}
		}
	}
}

func residualCases(t *testing.T) []residualCase {
	t.Helper()
	const n = 3
	return []residualCase{
		{
			name:  "2x2 [[4,0],[0,0]]",
			sys:   withDelays(t, residualPlant(t, n, 2, 2, 1), nil, nil, mat.NewDense(2, 2, []float64{4, 0, 0, 0})),
			wantN: 2*n + 4,
		},
		{
			name:  "2x3 rows cheaper",
			sys:   withDelays(t, residualPlant(t, n, 3, 2, 1), nil, nil, mat.NewDense(2, 3, []float64{2, 0, 0, 0, 1, 3})),
			wantN: 2*n + 6,
		},
		{
			name:  "3x2 columns cheaper",
			sys:   withDelays(t, residualPlant(t, n, 2, 3, 1), nil, nil, mat.NewDense(3, 2, []float64{2, 0, 0, 1, 0, 3})),
			wantN: 2*n + 6,
		},
		{
			name:  "mixed input output io",
			sys:   withDelays(t, residualPlant(t, n, 2, 2, 1), []float64{1, 0}, []float64{0, 2}, mat.NewDense(2, 2, []float64{3, 0, 0, 0})),
			wantN: 2*n + 6,
		},
		{
			name:  "static gain",
			sys:   withDelays(t, residualPlant(t, 0, 2, 2, 1), nil, nil, mat.NewDense(2, 2, []float64{4, 0, 0, 0})),
			wantN: 4,
		},
	}
}

func TestAbsorbDelayResidualIODelayExact(t *testing.T) {
	for _, tc := range residualCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			assertMatchesOracle(t, "original", tc.sys, tc.sys)
			for _, scope := range []AbsorbScope{AbsorbAll, AbsorbIO} {
				abs, err := tc.sys.AbsorbDelay(scope)
				if err != nil {
					t.Fatal(err)
				}
				if abs.HasDelay() {
					t.Fatalf("%s: absorbed system still has delay", scope)
				}
				if n, _, _ := abs.Dims(); n != tc.wantN {
					t.Fatalf("%s: states %d, want %d", scope, n, tc.wantN)
				}
				assertMatchesOracle(t, string(scope), abs, tc.sys)
			}
		})
	}
}

func TestPullDelaysToLFTResidualIODelayExact(t *testing.T) {
	for _, tc := range residualCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			lft, err := tc.sys.PullDelaysToLFT()
			if err != nil {
				t.Fatal(err)
			}
			if lft.Delay != nil || lft.InputDelay != nil || lft.OutputDelay != nil {
				t.Fatal("external delays left after PullDelaysToLFT")
			}
			assertMatchesOracle(t, "lft", lft, tc.sys)
			abs, err := lft.AbsorbDelay()
			if err != nil {
				t.Fatal(err)
			}
			if abs.HasDelay() {
				t.Fatal("absorbed LFT still has delay")
			}
			assertMatchesOracle(t, "lft absorbed", abs, tc.sys)
		})
	}
}

func TestAbsorbDelayDecomposableStateCountUnchanged(t *testing.T) {
	const n = 3
	sys := withDelays(t, residualPlant(t, n, 2, 2, 1), []float64{1, 0}, nil, mat.NewDense(2, 2, []float64{3, 1, 4, 2}))
	abs, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if got, _, _ := abs.Dims(); got != n+4+1+1 {
		t.Fatalf("states %d, want %d", got, n+6)
	}
	assertMatchesOracle(t, "decomposable", abs, sys)

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	if got, _, _ := lft.Dims(); got != n {
		t.Fatalf("LFT states %d, want %d", got, n)
	}
}

func TestPullDelaysToLFTResidualWithInternalDelay(t *testing.T) {
	const n = 2
	sys := residualPlant(t, n, 2, 2, 1)
	if err := sys.SetInternalDelay([]float64{2},
		mat.NewDense(n, 1, []float64{0.2, -0.1}),
		mat.NewDense(1, n, []float64{0.3, 0.25}),
		mat.NewDense(2, 1, []float64{0.15, -0.2}),
		mat.NewDense(1, 2, []float64{0.4, 0.1}),
		mat.NewDense(1, 1, []float64{0.05}),
	); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetDelay(mat.NewDense(2, 2, []float64{4, 0, 0, 0})); err != nil {
		t.Fatal(err)
	}

	ref, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}
	if ref.HasInternalDelay() || !ref.HasDelay() {
		t.Fatal("internal-only absorption changed external delay")
	}

	assertMatchesOracle(t, "simulate", sys, ref)
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	assertMatchesOracle(t, "lft", lft, ref)
	abs, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	if abs.HasDelay() {
		t.Fatal("absorbed system still has delay")
	}
	assertMatchesOracle(t, "absorbed", abs, ref)
}

func TestPullDelaysToLFTResidualContinuous(t *testing.T) {
	sys := withDelays(t, residualPlant(t, 3, 2, 2, 0), []float64{0.2, 0}, nil, mat.NewDense(2, 2, []float64{0.7, 0, 0, 0.3}))
	if _, _, res := DecomposeIODelay(sys.TotalDelay()); !delayMatrixHasNonzero(res) {
		t.Fatal("case must be non-decomposable")
	}
	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0.01, 0.3, 1, 4, 20}
	want, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	got, err := lft.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range want.Data {
		if d := cmplx.Abs(got.Data[k] - want.Data[k]); d > 1e-12 {
			t.Fatalf("freq index %d: got %v want %v", k, got.Data[k], want.Data[k])
		}
	}

	for _, scope := range []AbsorbScope{AbsorbAll, AbsorbIO} {
		abs, err := sys.AbsorbDelay(scope)
		if err != nil {
			t.Fatal(err)
		}
		if abs.HasDelay() {
			t.Fatalf("%s: continuous absorbed system still has delay", scope)
		}
		dc, err := abs.FreqResponse([]float64{0})
		if err != nil {
			t.Fatal(err)
		}
		dcWant, err := sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatal(err)
		}
		for k := range dcWant.Data {
			if d := cmplx.Abs(dc.Data[k] - dcWant.Data[k]); d > 1e-12 {
				t.Fatalf("%s: DC gain %d: got %v want %v", scope, k, dc.Data[k], dcWant.Data[k])
			}
		}
	}

	if _, err := absorbInputDelay(sys); !errors.Is(err, ErrWrongDomain) {
		t.Fatalf("continuous absorbInputDelay: %v", err)
	}
}
