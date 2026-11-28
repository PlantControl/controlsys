package controlsys

import "testing"

// BenchmarkLsimDescriptorReduction: Lsim and Step of the singular-E rows
// of reductionDescriptorCases, which take the Weierstrass reduction, with
// the TransferFunction verdict memoized across calls (warm).
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

// BenchmarkLsimDescriptorReductionCold: as BenchmarkLsimDescriptorReduction
// on a fresh model per call with the verdict memo off, so every call pays
// the full verification; Check is reductionUnverified alone.
func BenchmarkLsimDescriptorReductionCold(b *testing.B) {
	tt, u := benchBandLsimInput()
	saved := tfVerdictMemo
	tfVerdictMemo = false
	b.Cleanup(func() { tfVerdictMemo = saved })
	for _, k := range reductionDescriptorCases {
		fresh := func(b *testing.B) *System {
			sys, err := NewDescriptor(k.a, k.b, k.c, k.d, k.e, 0)
			if err != nil {
				b.Fatal(err)
			}
			return sys
		}
		b.Run("Lsim/"+k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Lsim(fresh(b), u, tt, nil)
			}
		})
		b.Run("Step/"+k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Step(fresh(b), 0)
			}
		})
		b.Run("Check/"+k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = fresh(b).reductionUnverified()
			}
		})
	}
}

// BenchmarkReductionUnverifiedCheck: reductionUnverified alone with the
// verdict memoized (warm), the cost Lsim adds per call.
func BenchmarkReductionUnverifiedCheck(b *testing.B) {
	for _, k := range reductionDescriptorCases {
		sys, err := NewDescriptor(k.a, k.b, k.c, k.d, k.e, 0)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(k.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = sys.reductionUnverified()
			}
		})
	}
}
