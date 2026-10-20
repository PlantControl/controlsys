package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
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

// TestFreqResponseInternalDelayLFTCancelIllConditionedLoop: at DC of a
// near double integrator whose rank-one delay loop leaves a closed-loop pole
// near zero, the LFT cancels terms 844 times |G|, so FreqResponse closes the
// loop; the frozen pencil, formed with B2·K·C2 rounded, has condition 2e7
// and must refine against the factored loop to beat the LFT. The oracle is
// the exact rational DC gain D − C·A⁻¹·B of the closed loop, K = (I − D22)⁻¹.
func TestFreqResponseInternalDelayLFTCancelIllConditionedLoop(t *testing.T) {
	a := []float64{9.665897930709342e-10, 3.5454187419198177e-09, -2.8732459853766166e-09, -9.993271397365217e-09}
	b1 := []float64{-1.0099020293585714, 0.8018895307913069}
	c1 := []float64{0.513357336126502, -0.7071007175520982}
	d11 := []float64{0.7341173575892297}
	b2 := []float64{-1.1137638255969695, 0.5261995246483373, 0.5268159819473245, -0.2439597733455373}
	c2 := []float64{0.040375852407838364, -0.1558650336068202, 0, 0}
	d21 := []float64{0.2819466415364478, 0}
	d22 := []float64{0, 0, -0.2710131635019522, 0}
	tau := []float64{0.2163537165520414, 0.010908457936042327}
	rat := func(v []float64) []*big.Rat {
		out := make([]*big.Rat, len(v))
		for i, x := range v {
			out[i] = new(big.Rat).SetFloat64(x)
		}
		return out
	}
	add := func(x, y []*big.Rat) []*big.Rat {
		out := make([]*big.Rat, len(x))
		for i := range x {
			out[i] = new(big.Rat).Add(x[i], y[i])
		}
		return out
	}
	for _, scale := range []float64{1, 1e2} {
		as := make([]float64, len(a))
		for i, v := range a {
			as[i] = v * scale
		}
		sys, err := New(mat.NewDense(2, 2, as), mat.NewDense(2, 1, b1), mat.NewDense(1, 2, c1), mat.NewDense(1, 1, d11), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInternalDelay(tau, mat.NewDense(2, 2, b2), mat.NewDense(2, 2, c2), mat.NewDense(1, 2, nil),
			mat.NewDense(2, 1, d21), mat.NewDense(2, 2, d22)); err != nil {
			t.Fatal(err)
		}
		fr, err := sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatal(err)
		}
		imd := rat([]float64{1, 0, 0, 1})
		for i, v := range rat(d22) {
			imd[i].Sub(imd[i], v)
		}
		k := fzRatInv(imd, 2)
		b2k := fzRatMul(rat(b2), k, 2, 2, 2)
		acl := add(rat(as), fzRatMul(b2k, rat(c2), 2, 2, 2))
		bcl := add(rat(b1), fzRatMul(b2k, rat(d21), 2, 2, 1))
		x := fzRatMul(fzRatInv(acl, 2), bcl, 2, 2, 1)
		g := new(big.Rat).Sub(rat(d11)[0], fzRatMul(rat(c1), x, 1, 2, 1)[0])
		want, _ := g.Float64()
		got := fr.At(0, 0, 0)
		if rel := cmplx.Abs(got-complex(want, 0)) / math.Abs(want); rel > 1e-12 {
			t.Errorf("A×%g: G(0) = %.17g, want %.17g (rel err %.3g)", scale, got, want, rel)
		}
	}
}

// BenchmarkFreqResponseLFTCancel: in P(s)·(1 − e^{−sτ}), P = 1/(s+1), the
// LFT sums ±P to a response ωτ smaller, so every point with ωτ < 1/8 takes
// the closed-loop fallback, its worst case.
func BenchmarkFreqResponseLFTCancel(b *testing.B) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -1}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, -1}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		b.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.01}, mat.NewDense(2, 1, []float64{0, 1}), mat.NewDense(1, 2, nil),
		mat.NewDense(1, 1, nil), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil)); err != nil {
		b.Fatal(err)
	}
	omega := logspace(-3, 3, 200)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.FreqResponse(omega); err != nil {
			b.Fatal(err)
		}
	}
}
