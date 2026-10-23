package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCommensurateClasses(t *testing.T) {
	r2, r3 := math.Sqrt2, math.Sqrt(3)
	for _, tc := range []struct {
		tau     []float64
		grp, k  []int
		wantDim int
	}{
		{[]float64{0.7}, []int{0}, []int{1}, 1},
		{[]float64{0.7, 1.4}, []int{0, 0}, []int{1, 2}, 1},
		{[]float64{1, 1}, []int{0, 0}, []int{1, 1}, 1},
		{[]float64{1, 2, r2}, []int{0, 0, 1}, []int{1, 2, 1}, 2},
		{[]float64{r2, 1, 2}, []int{0, 1, 1}, []int{1, 1, 2}, 2},
		{[]float64{1, r2, 2, 2 * r2, 3}, []int{0, 1, 0, 1, 0}, []int{1, 1, 2, 2, 3}, 2},
		{[]float64{1, r2, r3}, []int{0, 1, 2}, []int{1, 1, 1}, 3},
		{[]float64{1, r2, 1}, []int{0, 1, 0}, []int{1, 1, 1}, 2},
		// fl(3·0.1) ≠ 0.3 ≠ 3·0.1: commensurate up to rounding.
		{[]float64{0.1, r2, 0.3}, []int{0, 1, 0}, []int{1, 1, 3}, 2},
		{[]float64{0.1, 0.1 * r2, 0.3, 0.3 * r2}, []int{0, 1, 0, 1}, []int{1, 1, 3, 3}, 2},
		// Commensurate but past maxLiftedOrder: separate classes.
		{[]float64{1, 64}, []int{0, 1}, []int{1, 1}, 2},
	} {
		grp, k, dim := commensurateClasses(tc.tau)
		if !slices.Equal(grp, tc.grp) || !slices.Equal(k, tc.k) || dim != tc.wantDim {
			t.Errorf("commensurateClasses(%v) = %v, %v, %d; want %v, %v, %d", tc.tau, grp, k, dim, tc.grp, tc.k, tc.wantDim)
		}
		for i := range tc.tau {
			for j := range tc.tau {
				if grp[i] == grp[j] && !nearProductsEqual(tc.tau[i], float64(k[j]), tc.tau[j], float64(k[i])) {
					t.Errorf("%v: τ_%d/τ_%d != %d/%d", tc.tau, i, j, k[i], k[j])
				}
			}
		}
	}
}

// partlyCommensurateLoop returns delayResonanceLoop with τ = (1, 2, 3, √2).
func partlyCommensurateLoop(t *testing.T, kr float64, c, d []float64) (*System, func(float64) float64) {
	t.Helper()
	return delayResonanceLoop(t, []float64{1, 2, 3, math.Sqrt2}, kr, c, d)
}

// delayResonanceLoop returns G(s) = r(s) + Σ c_i·e^{−sτ_i}/(1 − d_i·e^{−sτ_i})
// with r(s) = kr/(s² + 2ζω0·s + ω0²), as a delay LFT with diagonal
// D22 = diag(d), and |G(jω)| in closed form.
func delayResonanceLoop(t *testing.T, tau []float64, kr float64, c, d []float64) (*System, func(float64) float64) {
	t.Helper()
	nd := len(tau)
	n := nd + 1
	const w0, zeta = 1.3, 0.05
	dd := make([]float64, n*n)
	for i := range nd {
		dd[i+1] = c[i]
		dd[n*(i+1)] = 1
		dd[n*(i+1)+i+1] = d[i]
	}
	b := make([]float64, 2*n)
	b[n] = kr
	cm := make([]float64, 2*n)
	cm[0] = 1
	H, err := NewFromSlices(2, n, n,
		[]float64{0, 1, -w0 * w0, -2 * zeta * w0},
		b, cm, dd, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	gain := func(w float64) float64 {
		s := complex(0, w)
		v := complex(kr, 0) / (s*s + complex(2*zeta*w0, 0)*s + complex(w0*w0, 0))
		for i := range tau {
			z := cmplx.Exp(-s * complex(tau[i], 0))
			v += complex(c[i], 0) * z / (1 - complex(d[i], 0)*z)
		}
		return cmplx.Abs(v)
	}
	return sys, gain
}

// TestHinfNormPartlyCommensurateDelays checks τ = (1, 2, 3, √2): the limit
// G∞ lives on a 2-torus, one phase for {1, 2, 3} and one for √2. For
// c = (1, 1, −1, 1), d = 0, sup |G∞| = max|1 + z − z²| + 1 = √5 + 1, below
// the 4 of one phase per delay, so a resonance peaking between the two is
// certified as the norm.
func TestHinfNormPartlyCommensurateDelays(t *testing.T) {
	skipSlowUnderRace(t)
	for _, tc := range []struct {
		name   string
		kr     float64
		c, d   []float64
		hfWant float64
	}{
		{"no-loop", 0.05, []float64{1, 1, -1, 1}, []float64{0, 0, 0, 0}, 1 + math.Sqrt(5)},
		{"no-loop-k0.1", 0.1, []float64{1, 1, -1, 1}, []float64{0, 0, 0, 0}, 1 + math.Sqrt(5)},
		{"neutral", 0.15, []float64{1, 0.6, -0.8, 0.5}, []float64{0.2, 0, -0.3, 0.4}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, gain := partlyCommensurateLoop(t, tc.kr, tc.c, tc.d)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			hf := newHighFrequencyGain(e)
			if hf.dim != 2 || !slices.Equal(hf.grp, []int{0, 0, 0, 1}) || !slices.Equal(hf.k, []float64{1, 2, 3, 1}) || hf.kmax != 3 {
				t.Fatalf("dim = %d, grp = %v, k = %v, kmax = %g; want 2, [0 0 0 1], [1 2 3 1], 3", hf.dim, hf.grp, hf.k, hf.kmax)
			}
			gInf := func(p, q float64) float64 {
				var v complex128
				for i, ph := range []float64{p, 2 * p, 3 * p, q} {
					z := cmplx.Exp(complex(0, -ph))
					v += complex(tc.c[i], 0) * z / (1 - complex(tc.d[i], 0)*z)
				}
				return cmplx.Abs(v)
			}
			const k = 1 << 11
			hi := 0.0
			for i := range k {
				for j := range k {
					hi = max(hi, gInf(2*math.Pi*float64(i)/k, 2*math.Pi*float64(j)/k))
				}
			}
			if tc.hfWant > 0 && math.Abs(hi-tc.hfWant) > 1e-5 {
				t.Fatalf("oracle sup |G∞| = %.12g, want %.12g", hi, tc.hfWant)
			}
			if est := hf.estimate(); est > hi*(1+1e-9) || est < hi*(1-1e-2) {
				t.Fatalf("estimate = %.12g, oracle %.12g", est, hi)
			}
			if !hf.below(hi*(1+1e-3), 1<<21) {
				t.Fatal("below(sup·(1+1e-3)) = false")
			}
			if hf.below(hi*(1-1e-3), 1<<21) {
				t.Fatal("below(sup·(1−1e-3)) = true")
			}
			want, wWant := oraclePeak(gain, 60, 1e-3)
			if !(want > hi*(1+1e-2)) {
				t.Fatalf("finite peak %.6g must exceed sup |G∞| %.6g", want, hi)
			}
			if sumC := math.Abs(tc.c[0]) + math.Abs(tc.c[1]) + math.Abs(tc.c[2]) + math.Abs(tc.c[3]); tc.hfWant > 0 && !(want < sumC) {
				t.Fatalf("finite peak %.6g must lie below the one-phase-per-delay bound %g", want, sumC)
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-4 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}

// TestHinfNormOnePhasePeakAtInfinity checks dim = 1 with nd = 2, equal and
// commensurate delays, where the peak is attained only as ω → ∞:
// G(s) = w(s)·G∞(Δ(s)) with w(s) = (s+1)/(s+2), |w(jω)| < 1 → 1, so
// σ_max(G(jω)) < sup_φ σ_max(G∞(φ)) at every finite ω.
func TestHinfNormOnePhasePeakAtInfinity(t *testing.T) {
	d11 := []float64{0.2, -0.1, 0.3, 0.1}
	d12 := []float64{1, -0.5, 0.4, 0.8}
	d21 := []float64{0.7, 0.3, -0.6, 1}
	for _, tc := range []struct {
		name string
		tau  []float64
		k    [2]float64
		d22  []float64
	}{
		{"equal", []float64{0.7, 0.7}, [2]float64{1, 1}, []float64{0.3, 0.2, -0.1, 0.4}},
		{"commensurate", []float64{0.7, 1.4}, [2]float64{1, 2}, []float64{0.3, 0.2, -0.1, 0.4}},
		// Not strongly stable: only the lifted Stein bound certifies it.
		{"commensurate-stein", []float64{0.7, 1.4}, [2]float64{1, 2}, []float64{0.54, 0.6, -0.6, -0.54}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hd := make([]float64, 16)
			for i := range 2 {
				for j := range 2 {
					hd[i*4+j] = d11[i*2+j]
					hd[i*4+2+j] = d12[i*2+j]
					hd[(i+2)*4+j] = d21[i*2+j]
					hd[(i+2)*4+2+j] = tc.d22[i*2+j]
				}
			}
			H, err := NewGain(mat.NewDense(4, 4, hd), 0)
			if err != nil {
				t.Fatal(err)
			}
			net, err := SetDelayModel(H, tc.tau)
			if err != nil {
				t.Fatal(err)
			}
			W, err := NewFromSlices(2, 2, 2, []float64{-2, 0, 0, -2}, []float64{1, 0, 0, 1}, []float64{-1, 0, 0, -1}, []float64{1, 0, 0, 1}, 0)
			if err != nil {
				t.Fatal(err)
			}
			sys, err := Series(W, net)
			if err != nil {
				t.Fatal(err)
			}
			c := func(x float64) complex128 { return complex(x, 0) }
			gInfAt := func(e1, e2 complex128) [2][2]complex128 {
				m := [4]complex128{1 - c(tc.d22[0])*e1, -c(tc.d22[1]) * e2, -c(tc.d22[2]) * e1, 1 - c(tc.d22[3])*e2}
				det := m[0]*m[3] - m[1]*m[2]
				var g [2][2]complex128
				for j := range 2 {
					v0 := (m[3]*c(d21[j]) - m[1]*c(d21[2+j])) / det
					v1 := (m[0]*c(d21[2+j]) - m[2]*c(d21[j])) / det
					for i := range 2 {
						g[i][j] = c(d11[i*2+j]) + c(d12[i*2])*e1*v0 + c(d12[i*2+1])*e2*v1
					}
				}
				return g
			}
			gInf := func(phi float64) float64 {
				return oracleSigmaMax2(gInfAt(cmplx.Exp(complex(0, -tc.k[0]*phi)), cmplx.Exp(complex(0, -tc.k[1]*phi))))
			}
			gain := func(w float64) float64 {
				s := complex(0, w)
				g := gInfAt(cmplx.Exp(-s*c(tc.tau[0])), cmplx.Exp(-s*c(tc.tau[1])))
				return cmplx.Abs((s+1)/(s+2)) * oracleSigmaMax2(g)
			}
			const k = 1 << 18
			hi, phiStar := 0.0, 0.0
			for i := range k {
				if v := gInf(2 * math.Pi * float64(i) / k); v > hi {
					hi, phiStar = v, 2*math.Pi*float64(i)/k
				}
			}
			const phi = 0.6180339887498949
			a, b := phiStar-2*math.Pi/k, phiStar+2*math.Pi/k
			for b-a > 1e-13 {
				x1, x2 := b-phi*(b-a), a+phi*(b-a)
				if gInf(x1) > gInf(x2) {
					b = x2
				} else {
					a = x1
				}
			}
			hi = max(hi, gInf((a+b)/2))
			phiStar = (a + b) / 2

			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if !e.directFeedthrough || e.nd != 2 {
				t.Fatalf("want delayed feedthrough with nd = 2, nd = %d", e.nd)
			}
			if hf := newHighFrequencyGain(e); hf.dim != 1 {
				t.Fatalf("dim = %d, want 1", hf.dim)
			}
			fin, _ := oraclePeak(gain, 80, 1e-3)
			if !(fin < hi) {
				t.Fatalf("finite peak %.12g must stay below sup %.12g", fin, hi)
			}
			h := tc.tau[0] / tc.k[0]
			prev := 0.0
			for _, n := range []float64{10, 100, 1000, 10000} {
				v := gain((phiStar + 2*math.Pi*n) / h)
				if !(v < hi) || !(v > prev) {
					t.Fatalf("gain at period %g = %.12g, want increasing below %.12g", n, v, hi)
				}
				prev = v
			}
			if hi-prev > 1e-6*hi {
				t.Fatalf("gain at large ω = %.12g, not approaching %.12g", prev, hi)
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-hi) > 1e-9*hi || !math.IsInf(w, 1) {
				t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, hi)
			}
		})
	}
}

// TestDelayModelsRejectZeroInternalDelay pins that a zero internal delay
// never reaches the delay LFT: validate (via requireSystem) rejects it in
// HinfNorm, Norm and IsStable before HasInternalDelay, which counts only
// nonzero delays, routes the model there.
func TestDelayModelsRejectZeroInternalDelay(t *testing.T) {
	for _, tau := range [][]float64{{0, 1}, {math.Copysign(0, -1), 1}, {1, 0}} {
		sys := &System{
			A: mat.NewDense(1, 1, []float64{-1}), B: mat.NewDense(1, 1, []float64{1}),
			C: mat.NewDense(1, 1, []float64{1}), D: mat.NewDense(1, 1, []float64{0.1}),
			LFT: &LFTDelay{
				Tau: tau,
				B2:  mat.NewDense(1, 2, []float64{0.5, 0.2}), C2: mat.NewDense(2, 1, []float64{1, -1}),
				D12: mat.NewDense(1, 2, []float64{1, 0.5}), D21: mat.NewDense(2, 1, []float64{1, 1}),
				D22: mat.NewDense(2, 2, []float64{0.3, 0.1, -0.2, 0.4}),
			},
		}
		if !sys.HasInternalDelay() {
			t.Fatalf("%v: HasInternalDelay = false", tau)
		}
		if _, _, err := HinfNorm(sys); !errors.Is(err, ErrZeroInternalDelay) {
			t.Errorf("%v: HinfNorm err = %v, want ErrZeroInternalDelay", tau, err)
		}
		if _, err := Norm(sys, math.Inf(1)); !errors.Is(err, ErrZeroInternalDelay) {
			t.Errorf("%v: Norm err = %v, want ErrZeroInternalDelay", tau, err)
		}
		if _, err := sys.IsStable(); !errors.Is(err, ErrZeroInternalDelay) {
			t.Errorf("%v: IsStable err = %v, want ErrZeroInternalDelay", tau, err)
		}
	}
}

// TestHinfNormLargeMultipleClass checks τ = (1, K, √2) with K up to 63: the
// class {1, K} moves K times faster along its phase, so cells are scaled by
// K_c rather than the whole torus by k_max, and a coarse first estimate is
// raised by samples that exceed a trial level. sup |G∞| = 2.3/0.7, with
// c = (1, 0.3, 1), d = 0.3, below the finite peak at DC or the resonance.
func TestHinfNormLargeMultipleClass(t *testing.T) {
	c, d := []float64{1, 0.3, 1}, []float64{0.3, 0.3, 0.3}
	const sup = 2.3 / 0.7
	for _, tc := range []struct {
		k, kr, want, wWant float64
	}{
		{20, 0.1, 0.1/1.69 + sup, 0},
		{40, 0.1, 0.1/1.69 + sup, 0},
		{40, 0.2, 0.2/1.69 + sup, 0},
		{63, 0.1, 0.1/1.69 + sup, 0},
		{63, 0.25, 3.604434473, 1.31165},
		{63, 0.3, 3.893320594, 1.31121},
	} {
		sys, gain := delayResonanceLoop(t, []float64{1, tc.k, math.Sqrt2}, tc.kr, c, d)
		e, err := newDelayLFT(sys)
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		if hf.dim != 2 || !slices.Equal(hf.scale, []float64{tc.k, 1}) {
			t.Fatalf("K=%g: dim = %d, scale = %v; want 2, [%g 1]", tc.k, hf.dim, hf.scale, tc.k)
		}
		if !hf.below(sup*1.01, 1<<21) || hf.samples > 1<<17 {
			t.Errorf("K=%g: below(sup·1.01) needs %d samples", tc.k, hf.samples)
		}
		if hf.below(sup*0.999, 1<<21) {
			t.Errorf("K=%g: below(sup·0.999) = true", tc.k)
		}
		if lvl, ok := newHighFrequencyGain(e).levelBelow(0, sup*1.02); !ok || !(lvl > sup && lvl < sup*1.02) {
			t.Errorf("K=%g: levelBelow(0, sup·1.02) = %g, %v", tc.k, lvl, ok)
		}
		got, w, err := HinfNorm(sys)
		if err != nil {
			t.Fatalf("K=%g kr=%g: HinfNorm: %v", tc.k, tc.kr, err)
		}
		if math.Abs(got-tc.want) > 1e-9*tc.want || math.Abs(w-tc.wWant) > 1e-5 {
			t.Errorf("K=%g kr=%g: HinfNorm = (%.12g, %.8g), want (%.12g, %.8g)", tc.k, tc.kr, got, w, tc.want, tc.wWant)
		}
		if g := gain(w); math.Abs(g-got) > 1e-9*got {
			t.Errorf("K=%g kr=%g: |G(j%g)| = %.12g, HinfNorm %.12g", tc.k, tc.kr, w, g, got)
		}
		if peak, _ := oraclePeak(gain, 20, 1e-3); peak > got*(1+1e-9) {
			t.Errorf("K=%g kr=%g: oracle peak %.12g above HinfNorm %.12g", tc.k, tc.kr, peak, got)
		}
	}
}

// TestHinfNormManyLargeMultipleClasses checks that the grid stays near 4096
// cells when several classes hold a large multiple: τ = h_c·(1, 63) for
// h = (1, √2/2, √3/2), the last two rounded to 30 bits so 63·h_c is exact.
func TestHinfNormManyLargeMultipleClasses(t *testing.T) {
	skipSlowUnderRace(t)
	h1 := math.Ldexp(math.Round(math.Ldexp(math.Sqrt2/2, 30)), -30)
	h2 := math.Ldexp(math.Round(math.Ldexp(math.Sqrt(3)/2, 30)), -30)
	tau := []float64{1, 63, h1, 63 * h1, h2, 63 * h2}
	c := []float64{1, 0.3, 1, 0.3, 1, 0.3}
	d := []float64{0.3, 0.3, 0.3, 0.3, 0.3, 0.3}
	sys, gain := delayResonanceLoop(t, tau, 1, c, d)
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	hf := newHighFrequencyGain(e)
	if hf.dim != 3 || !slices.Equal(hf.scale, []float64{63, 63, 63}) {
		t.Fatalf("dim = %d, scale = %v; want 3, [63 63 63]", hf.dim, hf.scale)
	}
	if cells, _ := hf.torusGrid(); len(cells)/hf.dim > 2*4096 {
		t.Fatalf("torusGrid has %d cells", len(cells)/hf.dim)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	want, wWant := oraclePeak(gain, 20, 1e-3)
	if !(want > 3.9/0.7) || math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-5 {
		t.Errorf("HinfNorm = (%.12g, %.8g), oracle (%.12g, %.8g)", got, w, want, wWant)
	}
}

// TestHighFrequencyGainDistinctDelays checks the fallback torus of one
// phase per distinct delay: a superset of the class torus, so it certifies
// only levels above the class sup, and nil when the two coincide.
func TestHighFrequencyGainDistinctDelays(t *testing.T) {
	c, d := []float64{1, 0.3, 1}, []float64{0.3, 0.3, 0.3}
	sys, _ := delayResonanceLoop(t, []float64{1, 8, 1}, 0.1, c, d)
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	dd := newHighFrequencyGain(e).distinctDelays()
	if dd == nil || dd.dim != 2 || dd.k != nil || !slices.Equal(dd.grp, []int{0, 1, 0}) {
		t.Fatalf("distinctDelays = %+v; want dim 2, grp [0 1 0], k nil", dd)
	}
	const sup = 2.3 / 0.7
	if !dd.below(sup*1.01, 1<<21) || dd.below(sup*0.999, 1<<21) {
		t.Error("distinct torus does not bracket sup |G∞|")
	}
	sys, _ = delayResonanceLoop(t, []float64{1, math.Sqrt2, math.Sqrt(3)}, 0.1, c, d)
	if e, err = newDelayLFT(sys); err != nil {
		t.Fatal(err)
	}
	if dd := newHighFrequencyGain(e).distinctDelays(); dd != nil {
		t.Errorf("distinctDelays of rationally independent delays = %+v, want nil", dd)
	}
}
