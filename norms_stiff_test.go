package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestHinfNormStiffGridPeak: poles from 3e-3 to 2e6 leave the QZ crossings
// near the ω ≈ 3.85e-3 peak outside the near-axis band, so the probe above
// the nearest grid sample (ω ≈ 3.99e-3) certified it, 2e-4 low. Oracle: the
// exact rational σmax at the true peak (FuzzLinfNorm XQYP43 crasher).
func TestHinfNormStiffGridPeak(t *testing.T) {
	a := []float64{
		-57751.6844611474, -77242.25023051786, -325533.0775847391, -52.45185936386116, -97.91675858913155, -2.4454275848528733,
		-77242.25023051785, -103450.19494714614, -435774.24500990094, -70.21458336350534, -131.0760856149504, -3.2735670593666404,
		-325533.077584739, -435774.245009901, -1.8390454525126743e+06, 25.92591417391822, 48.39831247473328, 1.2087263721935815,
		-52.451859363856464, -70.21458336349903, 25.92591417394483, -33.78554109258265, -63.051984020418004, -1.5746953149282903,
		-97.91675858913152, -131.07608561495033, 48.39831247473345, -63.051984020418, -118.49319309157873, 7.006594077797351,
		-2.4454275848534626, -3.2735670593674286, 1.2087263721902621, -1.5746953149282912, 7.006594077797348, -125.63327673783729,
	}
	b := []float64{-0.9416258104001715, -0.5762765558621612, 0.6333192030862999, 0.9151008458042379, -0.1437423944342892, -0.9920026900739702}
	c := []float64{-4.1453098325610847e-07, -7.271528468448447e-07, 8.846928410561699e-07, 1.5512891606026626e-07, -9.849959612184214e-07, 4.817489192689455e-07}
	d := []float64{0}
	sys, err := New(mat.NewDense(6, 6, a), mat.NewDense(6, 1, b), mat.NewDense(1, 6, c), mat.NewDense(1, 1, d), 0)
	if err != nil {
		t.Fatal(err)
	}
	g, ok := fzExactResponse(6, 1, 1, a, b, c, d, complex(0, 0.0038509944524435513))
	if !ok {
		t.Fatal("exact response failed")
	}
	want := fzSigmaMax(g, 1, 1)
	h, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if h < want*(1-1e-9) || h > want*(1+1e-6) {
		t.Errorf("HinfNorm = %v at ω=%v, exact peak ≥ %v (rel %.3g)", h, w, want, (want-h)/want)
	}
	l, _, err := linfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(l-h) > 1e-12*h {
		t.Errorf("linfNorm = %v, HinfNorm = %v", l, h)
	}
}
