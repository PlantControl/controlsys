package controlsys

import (
	"math"
	"testing"
)

// TestFreqResponseInternalDelayMovedPole: at an open-loop plant pole that
// the delay loop moves, G11 + G12·Δ·(I − G22Δ)⁻¹·G21 cancels open-loop
// terms far larger than the response, so FreqResponse must close the loop
// before solving. Each model is checked against the exact rational solve of
// (xI − A − Δ·b₂c₂ᵀ)X = B (fzFRInternalOracle) with no LFT allowance.
func TestFreqResponseInternalDelayMovedPole(t *testing.T) {
	seeds := fzFRSeeds()
	withTau := func(i int, tau float64) *fzSpec {
		s := seeds[i]
		s.Tau = math.Log10(tau)
		k, v := s.encode()
		return fzDecode(t, k, v)
	}
	cases := []struct {
		name string
		spec func() *fzSpec
	}{
		{"discrete unit-circle pair", func() *fzSpec { return withTau(7, 2) }},
		{"continuous undamped pair", func() *fzSpec { return withTau(2, 0.3) }},
		{"continuous near-integrator at DC", func() *fzSpec {
			return fzDecode(t, 0x3a, [fzNVals]float64{-20, -99.9, 41, -45, 60.333333333333336, -45, -26.666666666666668, -13.45, 42, -10, -4.3757142857142854, 8.666666666666666, -10, -163, -10, -0.05228787452803376})
		}},
		{"discrete pair near z=1, input delay", func() *fzSpec {
			return fzDecode(t, 0x41, [fzNVals]float64{-4, 85, 0, 33, 0, 0, -32, 0, -2, -2.5, 0.037, -13, 36, -13, -60, -0.2614393726401688})
		}},
		{"discrete pair near z=1, SISO", func() *fzSpec {
			return fzDecode(t, 0x41, [fzNVals]float64{-3, 13, 0, 0, 77, 0, 0, 0, -2, -10, 0.37, -77, -10, -300, 72, -0.1307196863200844})
		}},
		{"continuous undamped n=3", func() *fzSpec {
			return fzDecode(t, 0x72, [fzNVals]float64{-0.6786402423930551, 13, -0.8522801975522336, 11.64528661442717, -40, 9.730159100767608, -5.7124287797345685, 0.8515788906023601, -1.8699414019192915, -0.9009541575928393, 0.6904845230869261, -1.8615858804687453, 0.37193006986806204, -10, -10, -0.9291681166465303})
		}},
		{"discrete pairs near z=1, n=6", func() *fzSpec {
			return fzDecode(t, 0xd7, [fzNVals]float64{-0.8603088727097932, 4.928966109060965, -0.5305909486010396, 13, -1.7603256926868887, 13, 5.5794598346733535, -0.5028651286615842, -1.6756909102435067, -9.522762649906602, -1.2934801749704727, -2.8222193382401946, 0.1393351192955018, -10, -0.4596958443663803, 0.02128779065627474})
		}},
		{"continuous resonance under a large D", func() *fzSpec {
			return fzDecode(t, 0x60, [fzNVals]float64{0, 68, 0, 29, 0, 0, -1.2142857142857142, 30, 0, -10, 0.37, 47, -10, -10, -70, 0.3010299956639812})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := tc.spec()
			if s.N == 0 || s.Tau == 0 {
				t.Fatalf("decoded spec has no internal delay:\n%s", s)
			}
			sys := s.System(t, fzSysOpts{Delays: true, Internal: true})
			b2, c2 := fzFRInternalLoop(s)
			ws := fzFRFreqs(s)
			fr, err := sys.FreqResponse(ws)
			if err != nil {
				t.Fatalf("FreqResponse: %v\n%s", err, s)
			}
			for k, w := range ws {
				d := fzFRInternalOracle(s, b2, c2, w)
				fzFRCheck(t, s, "internal-delay FreqResponse", w, fzFRAt(fr, k), d.gd, d.beta, d.cond, d.slack)
			}
		})
	}
}
