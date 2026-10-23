package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func mustPanicAt(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: no panic", name)
		}
	}()
	fn()
}

func TestFreqResponseMatrixAtBounds(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100, 1000}
	siso := make([]complex128, len(omega))
	for k := range siso {
		siso[k] = complex(float64(k+1), -float64(k))
	}
	h := newFreqResponseMatrix(siso, omega, 1, 1, nil, nil)
	for k := range omega {
		if got := h.At(k, 0, 0); got != siso[k] {
			t.Errorf("SISO At(%d,0,0)=%v want %v", k, got, siso[k])
		}
	}
	for k := 1; k < len(omega); k++ {
		mustPanicAt(t, fmt.Sprintf("SISO At(0,0,%d)", k), func() { h.At(0, 0, k) })
		mustPanicAt(t, fmt.Sprintf("SISO At(0,%d,0)", k), func() { h.At(0, k, 0) })
	}

	const p, m = 2, 3
	data := make([]complex128, len(omega)*p*m)
	for idx := range data {
		data[idx] = complex(float64(idx), 1)
	}
	g := newFreqResponseMatrix(data, omega, p, m, nil, nil)
	for k := range omega {
		for i := range p {
			for j := range m {
				want := data[k*p*m+i*m+j]
				if got := g.At(k, i, j); got != want {
					t.Errorf("At(%d,%d,%d)=%v want %v", k, i, j, got, want)
				}
			}
		}
	}
	bad := [][3]int{
		{0, 0, m}, {0, p, 0}, {1, -1, 0}, {1, 0, -1}, {-1, p - 1, m - 1},
		{len(omega), 0, 0}, {0, 1, m}, {2, 0, 2 * m},
	}
	for _, b := range bad {
		mustPanicAt(t, fmt.Sprintf("2x3 At%v", b), func() { g.At(b[0], b[1], b[2]) })
	}

	row := make([]complex128, len(omega)*2)
	for idx := range row {
		row[idx] = complex(float64(idx), 0)
	}
	r := newFreqResponseMatrix(row, omega, 1, 2, nil, nil)
	mustPanicAt(t, "1x2 At(0,1,0) swapped", func() { r.At(0, 1, 0) })
	mustPanicAt(t, "1x2 At(0,0,2)", func() { r.At(0, 0, 2) })
}

func TestBodeResultAtBounds(t *testing.T) {
	omega := []float64{1, 10, 100}
	const p, m = 2, 2
	data := make([]complex128, len(omega)*p*m)
	for idx := range data {
		data[idx] = complex(float64(idx+1), 0)
	}
	b := bodeResultFromResponse(omega, data, p, m, nil, nil)
	if got := b.MagDBAt(2, 1, 1); got != b.magDB[2*p*m+m+1] {
		t.Errorf("MagDBAt(2,1,1)=%v", got)
	}
	for _, c := range [][3]int{{0, 0, m}, {0, p, 0}, {1, -1, 0}, {1, 0, -1}, {len(omega), 0, 0}, {-1, 0, 0}} {
		mustPanicAt(t, fmt.Sprintf("MagDBAt%v", c), func() { b.MagDBAt(c[0], c[1], c[2]) })
		mustPanicAt(t, fmt.Sprintf("PhaseAt%v", c), func() { b.PhaseAt(c[0], c[1], c[2]) })
	}
}

func TestSigmaResultAtBounds(t *testing.T) {
	omega := []float64{1, 10, 100}
	sv := []float64{6, 5, 4, 3, 2, 1}
	s := &SigmaResult{Omega: omega, sv: sv, nSV: 2}
	if got := s.At(1, 1); got != 3 {
		t.Errorf("At(1,1)=%v want 3", got)
	}
	for _, c := range [][2]int{{0, 2}, {1, -1}, {len(omega), 0}, {-1, 1}} {
		mustPanicAt(t, fmt.Sprintf("Sigma At%v", c), func() { s.At(c[0], c[1]) })
	}
}

func TestNicholsResultAtBounds(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100, 1000}
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	nr, err := sys.Nichols(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := nr.MagDBAt(1, 0, 0), -10*math.Log10(2); math.Abs(got-want) > 1e-12 {
		t.Errorf("MagDBAt(1,0,0)=%v want %v", got, want)
	}
	if got := nr.PhaseAt(1, 0, 0); math.Abs(got+45) > 1e-12 {
		t.Errorf("PhaseAt(1,0,0)=%v want -45", got)
	}
	for _, c := range [][3]int{{0, 0, 1}, {0, 1, 0}, {-1, 0, 1}, {1, -1, 0}, {1, 0, -1}, {len(omega), 0, 0}, {-1, 0, 0}} {
		mustPanicAt(t, fmt.Sprintf("Nichols MagDBAt%v", c), func() { nr.MagDBAt(c[0], c[1], c[2]) })
		mustPanicAt(t, fmt.Sprintf("Nichols PhaseAt%v", c), func() { nr.PhaseAt(c[0], c[1], c[2]) })
	}

	mimo, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{1, 2}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 2, []float64{0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	nm, err := mimo.Nichols(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := nm.MagDBAt(1, 0, 1), 20*math.Log10(math.Sqrt2); math.Abs(got-want) > 1e-12 {
		t.Errorf("1x2 MagDBAt(1,0,1)=%v want %v", got, want)
	}
	mustPanicAt(t, "1x2 Nichols MagDBAt(0,1,0) swapped", func() { nm.MagDBAt(0, 1, 0) })
	mustPanicAt(t, "1x2 Nichols PhaseAt(0,0,2)", func() { nm.PhaseAt(0, 0, 2) })
}

func TestFreqRespEstCoherenceAtBounds(t *testing.T) {
	const steps = 1024
	rng := rand.New(rand.NewPCG(7, 11))
	u := mat.NewDense(2, steps, nil)
	y := mat.NewDense(1, steps, nil)
	for k := range steps {
		u0, u1 := rng.NormFloat64(), rng.NormFloat64()
		u.Set(0, k, u0)
		u.Set(1, k, u1)
		y.Set(0, k, 0.6*u0-0.2*u1)
	}
	est, err := FreqRespEst(u, y, 0.1, &FreqRespEstOpts{NFFT: 64})
	if err != nil {
		t.Fatal(err)
	}
	coh, ok := est.Coherence()
	if !ok {
		t.Fatal("H1 estimator: no coherence")
	}
	nf := len(est.Omega)
	for k := range nf {
		for j := range 2 {
			got, ok := est.CoherenceAt(k, 0, j)
			if !ok || got != coh[k*2+j] {
				t.Errorf("CoherenceAt(%d,0,%d)=%v,%v want %v", k, j, got, ok, coh[k*2+j])
			}
		}
	}
	for _, c := range [][3]int{{0, 1, 0}, {0, 0, 2}, {1, -1, 0}, {1, 0, -1}, {-1, 0, 1}, {nf, 0, 0}} {
		mustPanicAt(t, fmt.Sprintf("CoherenceAt%v", c), func() { est.CoherenceAt(c[0], c[1], c[2]) })
	}

	fft, err := FreqRespEst(u, y, 0.1, &FreqRespEstOpts{NFFT: 64, Method: FreqRespEstFFT})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := fft.CoherenceAt(0, 0, 0); ok {
		t.Error("FFT estimator CoherenceAt ok=true")
	}
}
