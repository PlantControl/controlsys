package controlsys

import "testing"

// BenchmarkLsimDescriptorReduction: Lsim and Step of the singular-E rows
// of reductionDescriptorCases, which take the Weierstrass reduction.
func BenchmarkLsimDescriptorReduction(b *testing.B) {
	tt, u := benchBandLsimInput()
	for _, k := range reductionDescriptorCases {
		sys, err := NewDescriptor(k.a, k.b, k.c, k.d, k.e, 0)
		if err != nil {
			b.Fatal(err)
		}
		b.Run("Lsim/"+k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Lsim(sys, u, tt, nil)
			}
		})
		b.Run("Step/"+k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Step(sys, 0)
			}
		})
	}
}
