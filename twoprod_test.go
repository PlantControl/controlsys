package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestDekkerProductMatchesFMA (ESR456): without hardware FMA the residual's
// TwoProduct uses Dekker's split, which must agree with math.FMA bit for bit
// over the range it is used in.
func TestDekkerProductMatchesFMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range 200000 {
		a := math.Ldexp(rng.NormFloat64(), rng.IntN(400)-200)
		b := math.Ldexp(rng.NormFloat64(), rng.IntN(400)-200)
		h := float64(a * b)
		want := math.FMA(a, b, -h)
		if got := dekkerProductLo(a, b, h); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("case %d: a=%v b=%v: Dekker %v, FMA %v", i, a, b, got, want)
		}
	}
}
