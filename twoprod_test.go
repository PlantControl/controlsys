package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestDekkerProductMatchesFMA (ESR456): off arm64 the residual's TwoProduct
// is Dekker's split, with math.FMA for operands ≥ 2⁹⁹⁵ and products whose
// error underflows; twoProductLo must equal math.FMA bit for bit across
// those ranges (and dekkerProductLo inside its own).
func TestDekkerProductMatchesFMA(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	exps := [][2]int{{-200, 200}, {990, 1020}, {-1074, -900}, {-560, -480}}
	for _, r := range exps {
		for i := range 50000 {
			a := math.Ldexp(rng.NormFloat64(), r[0]+rng.IntN(r[1]-r[0]))
			b := math.Ldexp(rng.NormFloat64(), -rng.IntN(40)+20)
			if r[0] < 0 && r[1] < -400 {
				b = math.Ldexp(rng.NormFloat64(), -rng.IntN(100))
			}
			h := float64(a * b)
			if math.IsInf(h, 0) {
				continue
			}
			want := math.FMA(a, b, -h)
			if got := twoProductLo(a, b, h); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("range %v case %d: a=%v b=%v: twoProductLo %v, FMA %v", r, i, a, b, got, want)
			}
			if math.Abs(a) < 0x1p995 && math.Abs(b) < 0x1p995 && math.Abs(h) >= 0x1p-969 {
				if got := dekkerProductLo(a, b, h); math.Float64bits(got) != math.Float64bits(want) {
					t.Fatalf("range %v case %d: a=%v b=%v: Dekker %v, FMA %v", r, i, a, b, got, want)
				}
			}
		}
	}
}
