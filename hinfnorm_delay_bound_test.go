package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPermanentOfIPlus(t *testing.T) {
	if got := permanentOfIPlus([]float64{0, 1, 0, 0}, 2); got != 1 {
		t.Fatalf("nilpotent per(I+P) = %v, want 1", got)
	}
	if got := permanentOfIPlus([]float64{1, 1, 1, 1}, 2); got != 5 {
		t.Fatalf("per([2 1; 1 2]) = %v, want 5", got)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	const n = 4
	lu := make([]complex128, n*n)
	piv := make([]int, n)
	for range 200 {
		p := make([]float64, n*n)
		for i := range p {
			p[i] = 0.6 * rng.Float64()
			lu[i] = -cmplx.Rect(p[i]*rng.Float64(), 2*math.Pi*rng.Float64())
		}
		for i := range n {
			lu[i*n+i]++
		}
		det := complex(0, 0)
		if cLUFactor(lu, piv, n) {
			det = 1
			for k := range n {
				det *= lu[k*n+k]
				if piv[k] != k {
					det = -det
				}
			}
		}
		if got, bound := cmplx.Abs(det-1), permanentOfIPlus(p, n)-1; got > bound*(1+1e-12) {
			t.Fatalf("|det(I−M)−1| = %v exceeds per(I+P)−1 = %v", got, bound)
		}
	}
}

func TestNonnegResolventRejectsUnstableBlock(t *testing.T) {
	if _, ok := nonnegResolvent([]float64{1 - 1e-14, 0, 0, 2}, 2); ok {
		t.Fatal("accepted P with ρ(P) = 2")
	}
	q, ok := nonnegResolvent([]float64{0, 1, 0, 0}, 2)
	if !ok || q[0] != 1 || q[1] != 1 || q[2] != 0 || q[3] != 1 {
		t.Fatalf("(I − P)⁻¹ = %v, %v; want [1 1 0 1]", q, ok)
	}
}

func TestDelayTailBelowNormA(t *testing.T) {
	sys := internalDelayModel(t, 1, []float64{-3}, []float64{1, 0}, []float64{0, 1}, []float64{0, 0, 0, 0}, []float64{0.5, 0.7})
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.tail(1); !math.IsInf(got, 1) {
		t.Fatalf("tail(1) = %v for ‖A‖ = 3, want +Inf", got)
	}
}

func TestSpectralNormIsInducedTwoNorm(t *testing.T) {
	for _, tc := range []struct {
		name string
		m    *mat.Dense
		want float64
	}{
		{"scaled rotation", mat.NewDense(2, 2, []float64{-1, 1, -1, -1}), math.Sqrt2},
		{"identity", mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}), 1},
		{"rectangular", mat.NewDense(2, 3, []float64{3, 0, 0, 0, 4, 0}), 4},
		{"row", mat.NewDense(1, 2, []float64{3, 4}), 5},
	} {
		got := spectralNorm(tc.m)
		if got < tc.want || got > tc.want*(1+1e-14) {
			t.Errorf("%s: spectralNorm = %.17g, want upper bound within 1e-14 of %.17g", tc.name, got, tc.want)
		}
	}
}

func TestDelayLFTNormsAreSpectral(t *testing.T) {
	sys := internalDelayModel(t, 2, []float64{-1, 1, -1, -1}, []float64{1, 0, 0, 1}, []float64{1, 0, 0, 1},
		[]float64{0.3, 0, 0, 0.3}, []float64{0.5, 0.7})
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(e.normA-math.Sqrt2) > 1e-14 {
		t.Fatalf("normA = %v, want ‖A‖₂ = √2", e.normA)
	}
	if math.Abs(e.d22-0.3) > 1e-14 {
		t.Fatalf("d22 = %v, want ‖0.3·I‖₂ = 0.3", e.d22)
	}
	if e.entrywise {
		t.Fatal("entrywise tail selected although (1+‖D22‖₂)² − 1 = 0.69 < 1")
	}
}

func TestRationalTailBoundUsesSpectralNormA(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1, []float64{-1, 1, -1, -1}, []float64{1, 0}, []float64{1, 0}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := newRationalTailBound(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.normA-math.Sqrt2) > 1e-14 {
		t.Fatalf("normA = %v, want ‖A‖₂ = √2", r.normA)
	}
}
