package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

var yauykmNum, yauykmDen = []float64{-0.06393833756692992, -0.25774269274426936, -0.3895910244582764, -0.26170732884042885, -0.06592065955657661},
	[]float64{1, -1.5650719454733542, 0.46383714179126334, 0.157021399010823, -0.06822678671037938, 0.037484569160006115}

func yauykmLoop(t *testing.T) (*System, *rationalLoop, func(float64) float64) {
	t.Helper()
	r, err := (&TransferFunc{Num: [][][]float64{{yauykmNum}}, Den: [][]float64{yauykmDen}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	loop, err := newRationalLoop(r.Sys)
	if err != nil {
		t.Fatal(err)
	}
	return r.Sys, loop, func(w float64) float64 { return phaseOffsetDeg(loop.eval.at(w)) }
}

// YAUYKM: a wide polish bracket below the crossover at Nyquist spanned a
// ±180° wrap; refinement landed on the wrap and dropped the candidate π,
// whose phase was within tol. Oracle: the coefficients in big floats, L(−1)
// real negative with |num(−1)| far above coefficient noise.
func TestPolishKeepsCandidateBesideWrap(t *testing.T) {
	nyq := bigHorner(yauykmNum, math.Pi) / bigHorner(yauykmDen, math.Pi)
	var scale float64
	for _, v := range yauykmNum {
		scale += math.Abs(v)
	}
	if !(real(nyq) < 0) || math.Abs(imag(nyq)) > 1e-9*math.Abs(real(nyq)) || cmplx.Abs(bigHorner(yauykmNum, math.Pi)) <= 1e-13*scale {
		t.Fatalf("oracle L(−1) = %v", nyq)
	}
	if p := phaseOffsetDeg(bigHorner(yauykmNum, 3.11) / bigHorner(yauykmDen, 3.11)); p < 90 {
		t.Fatalf("oracle phase offset %g° at 3.11, want past a wrap", p)
	}
	sys, loop, f := yauykmLoop(t)
	if got := loop.polish([]float64{1.6891706744270651, math.Pi}, f, 1e-7, true); !slices.Contains(got, math.Pi) {
		t.Errorf("polish = %v, want π kept", got)
	}
	res, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	n := len(res.PhaseCrossFreqs)
	if n == 0 || res.PhaseCrossFreqs[n-1] != math.Pi || math.Abs(res.GainMargins[n-1]+20*math.Log10(cmplx.Abs(nyq))) > 1e-5 {
		t.Errorf("θpc %v gm %v, want π last with gm %g", res.PhaseCrossFreqs, res.GainMargins, -20*math.Log10(cmplx.Abs(nyq)))
	}
}

// YAUYKM: an interior candidate with |f| ≤ tol and no sign change within
// 1e-4 relative, a tangency, was dropped when the 1e-2 bracket spanned a
// ±180° wrap. Synthetic f: a phase offset touching 0 at w = 1 that wraps
// at 0.992.
func TestPolishInteriorCandidateBesideWrap(t *testing.T) {
	_, loop, _ := yauykmLoop(t)
	f := func(w float64) float64 {
		s := 0.02
		if w < 1 {
			s = 0.008
		}
		return math.Remainder(-180*(w-1)*(w-1)/(s*s), 360)
	}
	if got := loop.polish([]float64{1}, f, 1e-7, true); !slices.Equal(got, []float64{1}) {
		t.Errorf("polish = %v, want [1]", got)
	}
}

// YAUYKM: the crossover at Nyquist was not added when the last one below
// lay within 1e-6 relative of π. L(−1) < 0 as above; the crossing below is
// a stand-in, as no loop at hand places one there.
func TestNyquistPhaseCrossingsCloseBelow(t *testing.T) {
	_, loop, f := yauykmLoop(t)
	in := []float64{1.6891706744270651, math.Pi * (1 - 5e-7)}
	if got := loop.nyquistPhaseCrossings(in, f, 1e-7); !slices.Equal(got, append(slices.Clone(in), math.Pi)) {
		t.Errorf("nyquistPhaseCrossings = %v, want π appended", got)
	}
}
