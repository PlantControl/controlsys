package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hinfRegPlant is syRegPlant of the fuzz tuple with z (C1, D12) scaled by k.
func hinfRegPlant(t *testing.T, kind uint8, v [fzNVals]float64, k float64) (*System, int, int) {
	t.Helper()
	s := syNormC(fzDecode(t, kind, v))
	P, _, _ := syRegPlant(s)
	for i := range P.C1 {
		P.C1[i] *= k
	}
	for i := range P.D12 {
		P.D12[i] *= k
	}
	return P.System(t, s), P.ny, P.nu
}

// hinfH2CandidateNorm is ‖T_zw‖∞ of P with the H2Syn controller, an upper
// bound on the optimal gamma.
func hinfH2CandidateNorm(t *testing.T, P *System, ny, nu int) float64 {
	t.Helper()
	h2, err := H2Syn(P, ny, nu)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := LFT(P, h2.K, LFTFeedback{Nu: nu, Ny: ny})
	if err != nil {
		t.Fatal(err)
	}
	h, _, err := HinfNorm(cl)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

var hinfLargeGammaTuple = [fzNVals]float64{0.2, -1.5, 0.5, 2, -0.3, 1, 0, 0, -0.25, -7, 1.3, 69, 2, 0, 0.5, -10}

// Scaling z by k scales the optimal gamma by exactly k, so the bisection
// edge of the continuous (Tustin) plant must scale with it (JK62NS w19:
// GammaOpt grew up to 257x above that oracle; at k = 2^30 the edge was
// garbage or no gamma below 1e12 was feasible).
func TestHinfSyn_ZScaledEdgeInvariant(t *testing.T) {
	var edge1 float64
	for _, e := range []int{0, 10, 20, 30, 40} {
		k := math.Ldexp(1, e)
		Pd, ny, nu := hinfRegPlant(t, 0xb3, hinfLargeGammaTuple, k)
		Pc, err := Pd.undiscretizeTustin(0)
		if err != nil {
			t.Fatal(err)
		}
		d, err := hinfSynScaled("HinfSyn", Pc, ny, nu, "D12", "D21")
		if err != nil {
			t.Errorf("k = 2^%d: %v", e, err)
			continue
		}
		if e == 0 {
			edge1 = d.gammaEdge
			continue
		}
		if got := d.gammaEdge / k; !(math.Abs(got-edge1) <= 4e-6*edge1) {
			t.Errorf("k = 2^%d: gammaEdge/k = %.10g, want %.10g", e, got, edge1)
		}
	}
}

// GammaOpt may not exceed the H∞ norm of the stabilizing H2Syn controller
// (JK62NS: w12f/w13 #46 at gamma ~5e7 and 2.8e7, w13 #45 with a pole at
// z = 1; GammaOpt was 2.1%, 0.6% and 0.6% above it). Every z scale of the
// first plant must meet the same bound.
func TestHinfSyn_LargeGammaBelowH2Candidate(t *testing.T) {
	for _, tc := range []struct {
		name string
		kind uint8
		v    [fzNVals]float64
		k    float64
	}{
		{"discrete gamma 5e7", 0xb3, hinfLargeGammaTuple, 1},
		{"discrete z scaled 2^-20", 0xb3, hinfLargeGammaTuple, math.Ldexp(1, -20)},
		{"discrete z scaled 2^20", 0xb3, hinfLargeGammaTuple, math.Ldexp(1, 20)},
		{"discrete gamma 2.7e7", 0xb3, [fzNVals]float64{0.2, -1.5, 0.5, 2, -0.3, 1, 0, 44, -0.25, -1, 3.9000000000000004, 69, 0.4, 73, 0.5, -10}, 1},
		{"discrete pole at z = 1", 0x63, [fzNVals]float64{0, -3.5, -41, -39, -66, 0, -49, 0, 100, -10, 0.37, -70, -10, -1.6666666666666667, 39, -10}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			P, ny, nu := hinfRegPlant(t, tc.kind, tc.v, tc.k)
			res, err := HinfSyn(P, ny, nu)
			if err != nil {
				t.Fatal(err)
			}
			if !(res.Gamma <= res.GammaOpt) {
				t.Errorf("Gamma %.10g > GammaOpt %.10g", res.Gamma, res.GammaOpt)
			}
			if hc := hinfH2CandidateNorm(t, P, ny, nu); !(res.GammaOpt <= hc*(1+1e-3)) {
				t.Errorf("GammaOpt %.10g exceeds the H2Syn controller's ‖T_zw‖∞ %.10g", res.GammaOpt, hc)
			}
		})
	}
}

// D11 = 0 with a z row of size ~2e9 (JK62NS, folded RRJ7IL): C1ᵀC1 −
// S1ᵀR1⁻¹S1 cancelled to noise and X lost accuracy, so HinfSyn found no
// feasible gamma below 1.1e12 although the plant with z scaled by 2^-30
// gives GammaOpt ≈ 8.01, i.e. γ ≈ 8.60e9.
func TestHinfSyn_LargeZScaleD11ZeroFeasible(t *testing.T) {
	plant := func(k float64) *System {
		P, err := NewFromSlices(2, 2, 2,
			[]float64{3.092683724586688, -2.1211278495544508, -0.20852140972795583, -0.5976516957873039},
			[]float64{0.11541948546058889, 1.4768917021909143, 0.006536386266187466, 0.38559007704570214},
			[]float64{2029497287.921752 * k, -1181754578.1963654 * k, -0.536720302095411, -1.2196548309743167},
			[]float64{0, 1009862803.8697219 * k, 1.300772763408314, 0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		return P
	}
	k := math.Ldexp(1, -30)
	small, err := HinfSyn(plant(k), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	res, err := HinfSyn(plant(1), 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := small.GammaOpt / k
	if !(math.Abs(res.GammaOpt-want) <= 1e-4*want) {
		t.Errorf("GammaOpt %.10g, want %.10g (z-scaled oracle)", res.GammaOpt, want)
	}
	if !(res.Gamma <= res.GammaOpt) {
		t.Errorf("Gamma %.10g > GammaOpt %.10g", res.Gamma, res.GammaOpt)
	}
}

// At a coupling-limited optimum the central controller's norm lands a
// little above the gamma it is built for, at random; doubling the margin
// until one met it left GammaOpt at 469.5576, 0.32% above the edge
// 468.07 (JK62NS w13 #51), although a controller built at 468.82 gives
// ‖T_zw‖∞ = 468.8201.
func TestHinfSyn_CoupledOptimumBackoff(t *testing.T) {
	G, err := NewFromSlices(3, 2, 2,
		[]float64{-0.5298, 0.7059, -0.5789, 0.3543, 1.8340, -0.3794, 3.3560, 0.7230, 0.6144},
		[]float64{-0.9096, -1.1678, -0.8803, 0.8286, 1.3798, 0.7272},
		[]float64{1.2588, -0.7490, 1.3930, -1.2942, 0.5468, -1.2652},
		[]float64{0.8069, 0.9490, -2.9791, 0.0258}, 0)
	if err != nil {
		t.Fatal(err)
	}
	W1, err := NewFromSlices(1, 1, 1, []float64{-0.49606}, []float64{1}, []float64{114.444}, []float64{0.5}, 0)
	if err != nil {
		t.Fatal(err)
	}
	W2, err := NewGain(mat.NewDense(1, 1, []float64{0.008474}), 0)
	if err != nil {
		t.Fatal(err)
	}
	r, err := Mixsyn(G, W1, W2, nil)
	if err != nil {
		t.Fatal(err)
	}
	const edge = 468.0599
	if g := r.Info.GammaOpt; !(g <= edge*(1+5e-4)) {
		t.Errorf("GammaOpt %.10g, want within 5e-4 of the edge %.10g", g, edge)
	}
	if !(r.Gamma <= r.Info.GammaOpt) {
		t.Errorf("Gamma %.10g > GammaOpt %.10g", r.Gamma, r.Info.GammaOpt)
	}
}
