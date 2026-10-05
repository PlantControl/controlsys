package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func impulseMarkov(t *testing.T, sys *System, steps int) []*mat.Dense {
	t.Helper()
	_, m, p := sys.Dims()
	markov := make([]*mat.Dense, steps)

	for ch := range m {
		uData := make([]float64, m*steps)
		uData[ch*steps] = 1.0
		u := mat.NewDense(m, steps, uData)
		resp, err := sys.Simulate(u, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		yRaw := resp.Y.RawMatrix()
		for k := range steps {
			if markov[k] == nil {
				markov[k] = mat.NewDense(p, m, nil)
			}
			raw := markov[k].RawMatrix()
			for i := range p {
				raw.Data[i*raw.Stride+ch] = yRaw.Data[i*yRaw.Stride+k]
			}
		}
	}
	return markov
}

func sortedPoles(sys *System) ([]complex128, error) {
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	sort.Slice(poles, func(i, j int) bool {
		ai, aj := cmplx.Abs(poles[i]), cmplx.Abs(poles[j])
		if math.Abs(ai-aj) > 1e-12 {
			return ai < aj
		}
		return real(poles[i]) < real(poles[j])
	})
	return poles, nil
}

func polesMatch(a, b []complex128, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	used := make([]bool, len(b))
	for _, pa := range a {
		found := false
		for j, pb := range b {
			if !used[j] && cmplx.Abs(pa-pb) < tol {
				used[j] = true
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func TestERA_SISO_RoundTrip(t *testing.T) {
	dt := 0.01
	A := mat.NewDense(2, 2, []float64{
		0.9, 0.1,
		-0.2, 0.85,
	})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}

	markov := impulseMarkov(t, sys, 100)
	result, err := ERA(markov, 2, dt)
	if err != nil {
		t.Fatal(err)
	}

	origPoles, _ := sortedPoles(sys)
	eraPoles, _ := sortedPoles(result.Sys)

	if !polesMatch(origPoles, eraPoles, 0.01) {
		t.Errorf("poles mismatch:\n  orig: %v\n  era:  %v", origPoles, eraPoles)
	}

	if len(result.HSV) == 0 {
		t.Error("HSV should not be empty")
	}
}

func TestERA_MIMO_RoundTrip(t *testing.T) {
	dt := 0.1
	A := mat.NewDense(3, 3, []float64{
		0.8, 0.1, 0.0,
		-0.1, 0.7, 0.05,
		0.0, -0.05, 0.9,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		0.5, 0.3,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0.5,
		0, 1, 0,
	})
	D := mat.NewDense(2, 2, nil)

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}

	markov := impulseMarkov(t, sys, 100)
	result, err := ERA(markov, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	origPoles, _ := sortedPoles(sys)
	eraPoles, _ := sortedPoles(result.Sys)

	if !polesMatch(origPoles, eraPoles, 0.01) {
		t.Errorf("poles mismatch:\n  orig: %v\n  era:  %v", origPoles, eraPoles)
	}
}

func TestERA_OrderSelection(t *testing.T) {
	dt := 0.05
	A := mat.NewDense(4, 4, []float64{
		0.95, 0.0, 0.0, 0.0,
		0.0, 0.90, 0.0, 0.0,
		0.0, 0.0, 0.3, 0.0,
		0.0, 0.0, 0.0, 0.2,
	})
	B := mat.NewDense(4, 1, []float64{1, 1, 0.01, 0.01})
	C := mat.NewDense(1, 4, []float64{1, 1, 1, 1})
	D := mat.NewDense(1, 1, nil)

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}

	markov := impulseMarkov(t, sys, 200)
	result, err := ERA(markov, 2, dt)
	if err != nil {
		t.Fatal(err)
	}

	if len(result.HSV) < 4 {
		t.Fatalf("expected at least 4 HSV, got %d", len(result.HSV))
	}
	gap := result.HSV[1] / result.HSV[2]
	if gap < 5 {
		t.Errorf("expected clear HSV gap, ratio σ2/σ3 = %.2f", gap)
	}

	markovFull := impulseMarkov(t, sys, 200)
	resultFull, err := ERA(markovFull, 4, dt)
	if err != nil {
		t.Fatal(err)
	}

	uTest := mat.NewDense(1, 50, nil)
	uRaw := uTest.RawMatrix()
	for k := range 50 {
		uRaw.Data[k] = math.Sin(float64(k) * 0.2)
	}

	respOrig, _ := sys.Simulate(uTest, nil, nil)
	respReduced, _ := result.Sys.Simulate(uTest, nil, nil)
	respFull, _ := resultFull.Sys.Simulate(uTest, nil, nil)

	errReduced := matDiffNorm(respOrig.Y, respReduced.Y)
	errFull := matDiffNorm(respOrig.Y, respFull.Y)

	if errFull > 0.1 {
		t.Errorf("full-order ERA error too large: %g", errFull)
	}
	if errReduced > 5*errFull && errReduced > 0.1 {
		t.Errorf("reduced ERA error unexpectedly large: reduced=%g full=%g", errReduced, errFull)
	}
}

func matDiffNorm(a, b *mat.Dense) float64 {
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	rows := min(rb, ra)
	cols := min(cb, ca)
	sum := 0.0
	for i := range rows {
		for j := range cols {
			d := a.At(i, j) - b.At(i, j)
			sum += d * d
		}
	}
	return math.Sqrt(sum)
}

func TestERA_Validation(t *testing.T) {
	m1 := mat.NewDense(2, 1, nil)
	m2 := mat.NewDense(2, 1, nil)
	m3 := mat.NewDense(2, 1, nil)
	m4 := mat.NewDense(2, 1, nil)
	m5 := mat.NewDense(2, 1, nil)
	good := []*mat.Dense{m1, m2, m3, m4, m5}

	tests := []struct {
		name   string
		markov []*mat.Dense
		order  int
		dt     float64
	}{
		{"empty markov", nil, 1, 1.0},
		{"order zero", good, 0, 1.0},
		{"order negative", good, -1, 1.0},
		{"dt zero", good, 1, 0},
		{"dt negative", good, 1, -1},
		{"nil markov entry", []*mat.Dense{m1, nil, m3, m4, m5}, 1, 1.0},
		{"order too large", good, 10, 1.0},
		{"inconsistent dims", []*mat.Dense{
			mat.NewDense(2, 1, nil),
			mat.NewDense(3, 1, nil),
			mat.NewDense(2, 1, nil),
			mat.NewDense(2, 1, nil),
			mat.NewDense(2, 1, nil),
		}, 1, 1.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ERA(tc.markov, tc.order, tc.dt)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestERA_RejectsNilMarkovEntry(t *testing.T) {
	_, err := ERA([]*mat.Dense{
		mat.NewDense(1, 1, nil),
		nil,
		mat.NewDense(1, 1, nil),
	}, 1, 1)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("got %v, want ErrInvalidArgument", err)
	}
}

func TestERA_OrderAboveHankelRank(t *testing.T) {
	markov := []*mat.Dense{
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	}
	if _, err := ERA(markov, 1, 1); !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("zero Hankel: err = %v, want ErrInvalidOrder", err)
	}
	// h[k] = 0.5^(k-1), k >= 1: rank-1 Hankel, exact model 1/(z-0.5).
	geo := make([]*mat.Dense, 7)
	geo[0] = mat.NewDense(1, 1, []float64{0.25})
	for k := 1; k < len(geo); k++ {
		geo[k] = mat.NewDense(1, 1, []float64{math.Pow(0.5, float64(k-1))})
	}
	if _, err := ERA(geo, 2, 1); !errors.Is(err, ErrInvalidOrder) {
		t.Errorf("order 2 on rank-1 Hankel: err = %v, want ErrInvalidOrder", err)
	}
	res, err := ERA(geo, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if a := res.Sys.A.At(0, 0); math.Abs(a-0.5) > 1e-12 {
		t.Errorf("A = %g, want 0.5", a)
	}
	if cb := res.Sys.C.At(0, 0) * res.Sys.B.At(0, 0); math.Abs(cb-1) > 1e-12 {
		t.Errorf("CB = %g, want 1", cb)
	}
	if d := res.Sys.D.At(0, 0); d != 0.25 {
		t.Errorf("D = %g, want 0.25", d)
	}
}

func TestERA_RejectsNonFiniteInput(t *testing.T) {
	ok := []*mat.Dense{
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}),
	}
	for _, dt := range []float64{math.NaN(), math.Inf(1), 0, -1} {
		if _, err := ERA(ok, 1, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("dt=%g: err = %v, want ErrInvalidSampleTime", dt, err)
		}
	}
	bad := []*mat.Dense{ok[0], mat.NewDense(1, 1, []float64{math.NaN()}), ok[2]}
	if _, err := ERA(bad, 1, 1); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NaN markov: err = %v, want ErrInvalidArgument", err)
	}
}

func TestERA_MinValidOddLength(t *testing.T) {
	dt := 0.05
	A := mat.NewDense(2, 2, []float64{
		0.85, 0.15,
		-0.1, 0.8,
	})
	B := mat.NewDense(2, 1, []float64{1, 0.2})
	C := mat.NewDense(1, 2, []float64{1, -0.3})
	D := mat.NewDense(1, 1, nil)

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}

	markov := impulseMarkov(t, sys, 5)
	result, err := ERA(markov, 2, dt)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || result.Sys == nil {
		t.Fatal("expected identified system")
	}
}

func TestERA_StableSystem(t *testing.T) {
	dt := 0.1
	A := mat.NewDense(3, 3, []float64{
		0.5, 0.1, 0.0,
		-0.1, 0.4, 0.1,
		0.0, -0.1, 0.6,
	})
	B := mat.NewDense(3, 1, []float64{1, 0, 0.5})
	C := mat.NewDense(1, 3, []float64{1, 0, 1})
	D := mat.NewDense(1, 1, nil)

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		t.Fatal(err)
	}

	origStable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !origStable {
		t.Fatal("original system should be stable")
	}

	markov := impulseMarkov(t, sys, 100)
	result, err := ERA(markov, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	eraStable, err := result.Sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !eraStable {
		poles, _ := result.Sys.Poles()
		t.Errorf("ERA system should be stable, poles: %v", poles)
	}
}

func TestERA_EvenLengthMIMOReproducesMarkov(t *testing.T) {
	a := []float64{0.7, 0.25, -0.15, 0.6}
	b := []float64{1, 0.4, -0.5, 0.8}
	c := []float64{1, -0.3, 0.2, 0.9}
	d := []float64{0.1, -0.2, 0.3, 0.05}
	truthMarkov := func(n int) []*mat.Dense {
		out := []*mat.Dense{mat.NewDense(2, 2, d)}
		ak := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
		for len(out) < n {
			var cb, cab mat.Dense
			cb.Mul(mat.NewDense(2, 2, c), ak)
			cab.Mul(&cb, mat.NewDense(2, 2, b))
			out = append(out, &cab)
			var next mat.Dense
			next.Mul(mat.NewDense(2, 2, a), ak)
			ak = &next
		}
		return out
	}
	want := truthMarkov(20)
	for _, L := range []int{6, 7, 8} {
		res, err := ERA(truthMarkov(L), 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		got := impulseMarkov(t, res.Sys, 20)
		for k := range want {
			if !mat.EqualApprox(got[k], want[k], 1e-9) {
				t.Fatalf("L=%d: Markov %d = %v, want %v", L, k, mat.Formatted(got[k]), mat.Formatted(want[k]))
			}
		}
	}
}
