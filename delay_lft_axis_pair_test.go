package controlsys

import (
	"math"
	"slices"
	"testing"
)

// Task 4PHRFF: A with eigenvalues ±εj, ε in [1e-14, 1e-8], beside three
// equal internal delays. addAxisPole indented the pair at the origin as one
// pole, so the two-decade growth of |L| read as a root of χ: IsStable was
// false and HinfNorm +Inf although χ(0) = 0.2669 and no root lies right of
// −0.1. Oracles: certified argument-principle counts of the exact χ
// (fzDLQuasi.countRight), independent of the Nyquist path.

var axisPairVals = [fzNVals]float64{-27, 41.5, 324, 0, 6, -59, 0.6, 0, 0, -48, 0.37, 80, -1.1111111111111112, 110.000000001, -1, -1}

// axisPairModel is the FuzzDelayLFT input kind=0x40 axisPairVals as printed
// by fzDLModel.String, with A replaced.
func axisPairModel(a []float64) *fzDLModel {
	tau := 0.099999904632568359
	s := &fzSpec{
		Kind: 0x40, Vals: axisPairVals, N: 2, M: 1, P: 1, A: a,
		B: []float64{-0.1472572140881154, -0.9924448300725411},
		C: []float64{2.4162738596887947e-06, -3.775198247851565e-06},
		D: []float64{0}, S: []complex128{1e-13i, -1e-13i}, Tau: 0.1,
	}
	return &fzDLModel{
		s: s, nd: 3, tau: []float64{tau, tau, tau},
		class: []int{0, 0, 0}, mult: []int{1, 1, 1}, dim: 1, commensurate: true, d22: -0.5,
		B2:  []float64{0.052181610003732448, -0.77945662764586887, 0.98648491125242443, -0.74982310530161977, 0.0062019865204627073, 0.74155863549584022},
		C2:  []float64{0.10842718695648369, 0.42568487861383936, 0.48418223055009274, 0.42683757034690623, 0.33081917672492778, -0.038461145837691363},
		D12: []float64{0.4748388489412716, 0.40734816158954668, 0.30347027614817856},
		D21: []float64{0.12477005898666906, -0.073628607044057701, -0.26040293496877087},
		D22: []float64{-0.30347027614817856, 0.023888242733598245, 0.34001174779366944, 0.47414346275185476, 0.46488691130676729, 0.23698678141574556, -0.0031009932602384591, -0.32447441912734959, -0.49324245562621105},
	}
}

func axisPairRotation(eps float64) []float64 { return []float64{0, eps, -eps, 0} }

// axisPairRoots returns the certified count of roots of χ right of each σ.
func axisPairRoots(t *testing.T, md *fzDLModel, sigmas ...float64) []int {
	t.Helper()
	q := md.quasi()
	out := make([]int, len(sigmas))
	for i, sg := range sigmas {
		n, ok := q.countRight(sg, fzDLBudget)
		if !ok {
			t.Fatalf("countRight(%g) not certified\n%s", sg, md)
		}
		out[i] = n
	}
	return out
}

func axisPairCheckStable(t *testing.T, md *fzDLModel) {
	t.Helper()
	if got := axisPairRoots(t, md, 1e-6, 0, -1e-6, -1e-3, -0.1); slices.Max(got) != 0 {
		t.Fatalf("oracle counts %v, want all 0\n%s", got, md)
	}
	sys := md.system(t, false)
	st, err := sys.IsStable()
	if err != nil || !st {
		t.Fatalf("IsStable = %v, %v; certified χ is stable\n%s", st, err, md)
	}
	norm, omega, err := HinfNorm(sys)
	if err != nil || math.IsInf(norm, 0) || math.IsNaN(norm) {
		t.Fatalf("HinfNorm = %v at %v, %v; certified χ is stable\n%s", norm, omega, err, md)
	}
	md.checkPeak(t, norm, omega)
}

func TestIsStable_InternalDelayLFTOriginAxisPair(t *testing.T) {
	cases := []struct {
		name string
		a    []float64
	}{
		{"eps=1e-14", axisPairRotation(1e-14)},
		{"eps=1e-13", axisPairRotation(1e-13)},
		{"eps=1e-12", axisPairRotation(1e-12)},
		{"eps=1e-10", axisPairRotation(1e-10)},
		{"eps=1e-8", axisPairRotation(1e-8)},
		{"eps=1e-7", axisPairRotation(1e-7)},
		{"eps=1e-6", axisPairRotation(1e-6)},
		{"A=0", []float64{0, 0, 0, 0}},
		{"nilpotent", []float64{0, 1e-13, 0, 0}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { axisPairCheckStable(t, axisPairModel(c.a)) })
	}
}

// TestIsStable_InternalDelayLFTOriginAxisPairDecoded rebuilds the model
// through the fuzz decoder, sweeping ε = 10^v[0] (the decoder clamps it at
// 1e-13), and pins the literal model to the decoded one.
func TestIsStable_InternalDelayLFTOriginAxisPairDecoded(t *testing.T) {
	for _, v0 := range []float64{-27, -12, -10, -8, -7, -6} {
		v := axisPairVals
		v[0] = v0
		md := fzDLNewModel(t, fzDecode(t, 0x40, v))
		if v0 == -27 {
			lit := axisPairModel(md.s.A)
			if !slices.Equal(md.tau, lit.tau) || !slices.Equal(md.B2, lit.B2) || !slices.Equal(md.C2, lit.C2) ||
				!slices.Equal(md.D12, lit.D12) || !slices.Equal(md.D21, lit.D21) || !slices.Equal(md.D22, lit.D22) ||
				!slices.Equal(md.s.B, lit.s.B) || !slices.Equal(md.s.C, lit.s.C) {
				t.Fatalf("decoded model differs from the literal one\n%s\n%s", md, lit)
			}
		}
		q := md.quasi()
		if verdict, why := fzDLVerdict(q, false, 1e-6*math.Max(1, md.s.NormJ), md.leadZ(q)); verdict != 1 {
			t.Fatalf("v0=%g: oracle verdict %d (%s), want stable\n%s", v0, verdict, why, md)
		}
		axisPairCheckStable(t, md)
	}
}

// TestIsStable_InternalDelayLFTOriginAxisPairUnstable keeps the axis pair and
// changes the delay loop so that χ has roots right of the axis: C2 row 0
// times −0.5 gives χ(0) ≈ −0.003, one real root in (1e-6, 1e-2); B2 times −1
// gives two. The fix must not bias the origin indentation toward stable.
func TestIsStable_InternalDelayLFTOriginAxisPairUnstable(t *testing.T) {
	type tweak struct {
		name  string
		f     func(md *fzDLModel)
		roots int
		near  bool
	}
	tweaks := []tweak{
		{"C2row0*-0.5", func(md *fzDLModel) { md.C2[0] *= -0.5; md.C2[1] *= -0.5 }, 1, true},
		{"C2row1*0", func(md *fzDLModel) { md.C2[2], md.C2[3] = 0, 0 }, 1, true},
		{"B2*-1", func(md *fzDLModel) {
			for i := range md.B2 {
				md.B2[i] = -md.B2[i]
			}
		}, 2, false},
	}
	for _, eps := range []float64{1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 0} {
		for _, tw := range tweaks {
			md := axisPairModel(axisPairRotation(eps))
			tw.f(md)
			got := axisPairRoots(t, md, 1e-6, -1e-6, 1e-2)
			if got[0] != tw.roots || got[1] != tw.roots {
				t.Fatalf("eps=%g %s: oracle counts %v, want %d right of ±1e-6\n%s", eps, tw.name, got, tw.roots, md)
			}
			if tw.near && got[2] != 0 {
				t.Fatalf("eps=%g %s: oracle %d roots right of 1e-2, want 0\n%s", eps, tw.name, got[2], md)
			}
			sys := md.system(t, false)
			if st, err := sys.IsStable(); err != nil || st {
				t.Fatalf("eps=%g %s: IsStable = %v, %v; χ has %d RHP roots\n%s", eps, tw.name, st, err, tw.roots, md)
			}
			if norm, omega, err := HinfNorm(sys); err != nil || !math.IsInf(norm, 1) {
				t.Fatalf("eps=%g %s: HinfNorm = %v at %v, %v; want +Inf\n%s", eps, tw.name, norm, omega, err, md)
			}
		}
	}
}

// TestHinfNorm_InternalDelayLFTOriginAxisPair: the norm is finite, attained
// at its ω and not under-reported by the dense exact sweep or the torus sup
// (checkPeak, relative 1e-6), and matches the dense sweep peak to 1e-6:
// the peak at ω ≈ 0.52 is interior and smooth, where golden-section
// refinement of the exact σmax converges far below that.
func TestHinfNorm_InternalDelayLFTOriginAxisPair(t *testing.T) {
	for _, eps := range []float64{1e-14, 1e-13, 1e-10, 1e-8, 1e-6} {
		md := axisPairModel(axisPairRotation(eps))
		norm, omega, err := HinfNorm(md.system(t, false))
		if err != nil || math.IsInf(norm, 0) {
			t.Fatalf("eps=%g: HinfNorm = %v at %v, %v\n%s", eps, norm, omega, err, md)
		}
		md.checkPeak(t, norm, omega)
		wbest, _ := fzSweepMax(func(w float64) float64 {
			g := md.response(md.point(w), md.delta(w), true)
			if g == nil {
				return 0
			}
			return fzSigmaMax(g, 1, 1)
		}, fzDLGrid(md.s.S, 0, md.tau), 3)
		peak, ok := md.sigmaAt(wbest)
		if !ok || math.Abs(norm-peak) > 1e-6*peak {
			t.Fatalf("eps=%g: HinfNorm = %v at %v, exact σmax %v (%v) at the dense sweep peak ω=%v\n%s", eps, norm, omega, peak, ok, wbest, md)
		}
		t.Logf("eps=%g: HinfNorm = %.15g at ω=%.15g; sweep peak %.15g at ω=%.15g", eps, norm, omega, peak, wbest)
	}
}
