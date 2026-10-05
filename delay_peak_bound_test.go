package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// nonNormalResonance returns ss(T·A0·T⁻¹, T·B0, C0·T⁻¹) for the modal
// realization A0 = [−ζwr wd; −wd −ζwr] of wr²/(s² + 2ζ·wr·s + wr²) and
// T = [1 k; 0 1], k = 5, whose resolvent norm exceeds 1/dist(jω, eig A) by
// about k², while its entries stay O(k·wr) so that its response is
// computable to about 1e-11 at the resonance.
func nonNormalResonance(t testing.TB, wr, zeta float64) (a, b, c []float64) {
	t.Helper()
	const k = 5
	wd := wr * math.Sqrt(1-zeta*zeta)
	A0 := mat.NewDense(2, 2, []float64{-zeta * wr, wd, -wd, -zeta * wr})
	T := mat.NewDense(2, 2, []float64{1, k, 0, 1})
	Ti := mat.NewDense(2, 2, []float64{1, -k, 0, 1})
	var A, tmp mat.Dense
	tmp.Mul(T, A0)
	A.Mul(&tmp, Ti)
	bw := wr * wr / wd
	return A.RawMatrix().Data, []float64{k * bw, bw}, []float64{1, -k}
}

func resonanceResponse(w, wr, zeta float64) complex128 {
	return complex(wr*wr, 0) / complex(wr*wr-w*w, 2*zeta*wr*w)
}

// A resonance of half-width ζ·wr between samples 10 apart is certified to
// its closed-form peak 1/(2ζ√(1−ζ²)), through a non-normal realization and
// an output delay pulled into the descriptor.
func TestDescriptorCertifyNarrowResonanceBetweenSamples(t *testing.T) {
	const wr = 37.3
	ws := make([]float64, 11)
	for i := range ws {
		ws[i] = 10 * float64(i)
	}
	for _, zeta := range []float64{1e-2, 1e-3, 1e-4} {
		a, b, c := nonNormalResonance(t, wr, zeta)
		sys, err := NewFromSlices(2, 1, 1, a, b, c, []float64{0.05}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{0.3}); err != nil {
			t.Fatal(err)
		}
		r, err := newDescriptorResponse(sys)
		if err != nil {
			t.Fatal(err)
		}
		g := func(w float64) float64 {
			return cmplx.Abs(0.05 + resonanceResponse(w, wr, zeta))
		}
		want, _ := oracleNarrowPeak(g, wr, zeta)
		sampled := 0.0
		for _, w := range ws {
			sampled = max(sampled, g(w))
		}
		if sampled > 0.2*want {
			t.Fatalf("ζ=%g: grid samples reach %g of the peak %g", zeta, sampled, want)
		}
		ps, wps := []float64{0}, []float64{0}
		if err := r.certify(ws, []float64{0}, ps, wps, 1<<20, errDelayPeakUnsupported); err != nil {
			t.Fatalf("ζ=%g: %v", zeta, err)
		}
		if math.Abs(ps[0]-want) > 1e-9*want {
			t.Errorf("ζ=%g: certified %.15g at %g, oracle %.15g", zeta, ps[0], wps[0], want)
		}
		if at := g(wps[0]); math.Abs(at-ps[0]) > 1e-9*ps[0] {
			t.Errorf("ζ=%g: oracle %.15g at returned ω, certified %.15g", zeta, at, ps[0])
		}
	}
}

// oracleNarrowPeak returns sup g near a resonance at wr of damping ζ from a
// sweep of ±50 half-widths at 1/400 of a half-width, refined by golden
// section, and the coarse sweep of [0, 4·wr].
func oracleNarrowPeak(g func(float64) float64, wr, zeta float64) (float64, float64) {
	best, bw := 0.0, 0.0
	scan := func(lo, hi float64, n int) {
		for i := range n + 1 {
			w := lo + (hi-lo)*float64(i)/float64(n)
			if v := g(w); v > best {
				best, bw = v, w
			}
		}
	}
	scan(0, 4*wr, 40000)
	hw := zeta * wr
	scan(wr-50*hw, wr+50*hw, 40000)
	step := hw / 400
	a, b := bw-step, bw+step
	const phi = 0.6180339887498949
	for b-a > 1e-15*b {
		x1, x2 := b-phi*(b-a), a+phi*(b-a)
		if g(x1) > g(x2) {
			b = x2
		} else {
			a = x1
		}
	}
	if v := g((a + b) / 2); v > best {
		best, bw = v, (a+b)/2
	}
	return best, bw
}

// mimoResonances is K1·diag(r1, r2)·K2·diag(e^{-0.2s}, e^{-0.5s}) + D with
// two non-normal resonances, ζ = 1e-4 at 23.1 and ζ = 3e-4 at 61.7.
func mimoResonances(t testing.TB) (*System, func(float64) [2][2]complex128) {
	t.Helper()
	const w1, z1, w2, z2 = 23.1, 1e-4, 61.7, 3e-4
	k1 := []float64{1, 0.4, -0.3, 0.8}
	k2 := []float64{0.7, -0.2, 0.5, 1.1}
	d := []float64{0.1, 0, 0.05, -0.2}
	a1, b1, c1 := nonNormalResonance(t, w1, z1)
	a2, b2, c2 := nonNormalResonance(t, w2, z2)
	A := mat.NewDense(4, 4, nil)
	A.Slice(0, 2, 0, 2).(*mat.Dense).Copy(mat.NewDense(2, 2, a1))
	A.Slice(2, 4, 2, 4).(*mat.Dense).Copy(mat.NewDense(2, 2, a2))
	Bd := mat.NewDense(4, 2, []float64{b1[0], 0, b1[1], 0, 0, b2[0], 0, b2[1]})
	Cd := mat.NewDense(2, 4, []float64{c1[0], c1[1], 0, 0, 0, 0, c2[0], c2[1]})
	var B, C mat.Dense
	B.Mul(Bd, mat.NewDense(2, 2, k2))
	C.Mul(mat.NewDense(2, 2, k1), Cd)
	sys, err := New(A, &B, &C, mat.NewDense(2, 2, d), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{0.2, 0.5}); err != nil {
		t.Fatal(err)
	}
	resp := func(w float64) [2][2]complex128 {
		r := [2]complex128{resonanceResponse(w, w1, z1), resonanceResponse(w, w2, z2)}
		e := [2]complex128{cmplx.Exp(complex(0, -0.2*w)), cmplx.Exp(complex(0, -0.5*w))}
		var g [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					g[i][j] += complex(k1[i*2+k], 0) * r[k] * complex(k2[k*2+j], 0)
				}
				g[i][j] = (g[i][j] + complex(d[i*2+j], 0)) * e[j]
			}
		}
		return g
	}
	return sys, resp
}

func TestDescriptorCertifyMIMOResonances(t *testing.T) {
	sys, resp := mimoResonances(t)
	r, err := newDescriptorResponse(sys)
	if err != nil {
		t.Fatal(err)
	}
	g := func(w float64) float64 { return oracleSigmaMax2(resp(w)) }
	p1, _ := oracleNarrowPeak(g, 23.1, 1e-4)
	p2, _ := oracleNarrowPeak(g, 61.7, 3e-4)
	want := max(p1, p2)
	ws := []float64{0, 7, 30, 50, 100}
	ps, wps := []float64{0}, []float64{0}
	if err := r.certify(ws, []float64{0}, ps, wps, 1<<20, errDelayPeakUnsupported); err != nil {
		t.Fatal(err)
	}
	if math.Abs(ps[0]-want) > 1e-9*want {
		t.Errorf("certified %.15g at %g, oracle %.15g", ps[0], wps[0], want)
	}
	if at := g(wps[0]); math.Abs(at-ps[0]) > 1e-9*ps[0] {
		t.Errorf("oracle %.15g at returned ω, certified %.15g", at, ps[0])
	}
}

// The Taylor bound from a sample must dominate the gain on the whole
// half-interval it covers, for non-normal A, MIMO input delays and a delay
// feedback cycle.
func TestDescriptorBoundDominatesDenseGain(t *testing.T) {
	mimo, resp := mimoResonances(t)
	dde := scalarDDE(t, -2, 1)
	a, b, c := nonNormalResonance(t, 5, 1e-3)
	res, err := NewFromSlices(2, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := res.SetOutputDelay([]float64{1.5}); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		sys  *System
		g    func(float64) float64
		wMax float64
	}{
		{"mimo", mimo, func(w float64) float64 { return oracleSigmaMax2(resp(w)) }, 80},
		{"dde", dde, func(w float64) float64 {
			s := complex(0, w)
			return cmplx.Abs(1 / (s + 1 + 2*cmplx.Exp(-s)))
		}, 20},
		{"resonance", res, func(w float64) float64 { return cmplx.Abs(resonanceResponse(w, 5, 1e-3)) }, 10},
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, tc := range cases {
		r, err := newDescriptorResponse(tc.sys)
		if err != nil {
			t.Fatal(err)
		}
		checked := 0
		for range 2000 {
			w := tc.wMax * rng.Float64()
			s := r.sample(w)
			h := math.Pow(10, -4*rng.Float64()) / max(s.zeta, s.zetaL)
			if rng.IntN(2) == 0 {
				h = -h
			}
			ub := r.bound(s, h, 0)
			if math.IsInf(ub, 1) {
				continue
			}
			checked++
			for k := range 201 {
				x := w + h*float64(k)/200
				if v := tc.g(x); v > ub*(1+1e-10) {
					t.Fatalf("%s: gain %.15g at ω=%g exceeds bound %.15g from ω=%g, h=%g", tc.name, v, x, ub, w, h)
				}
			}
		}
		if checked < 1000 {
			t.Fatalf("%s: only %d finite bounds", tc.name, checked)
		}
	}
}

// lightResonanceLoop is L = k·e^{-τs}·wr²/(s² + 2ζ·wr·s + wr²) in a
// non-normal realization, as an I/O delay or an internal delay, with
// |L| ≤ 0.6 so that S = 1/(1+L) is stable with a peak of width ζ·wr.
func lightResonanceLoop(t testing.TB, zeta float64, internal bool) (*System, func(float64) complex128) {
	t.Helper()
	const wr, tau = 37.3, 0.21
	k := 1.2 * zeta
	a, b, c := nonNormalResonance(t, wr, zeta)
	for i := range c {
		c[i] *= k
	}
	sys, err := NewFromSlices(2, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if internal {
		c2 := mat.NewDense(1, 2, c)
		sys.C = mat.NewDense(1, 2, nil)
		err = sys.SetInternalDelay([]float64{tau}, mat.NewDense(2, 1, nil), c2,
			mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil))
	} else {
		err = sys.SetOutputDelay([]float64{tau})
	}
	if err != nil {
		t.Fatal(err)
	}
	return sys, func(w float64) complex128 {
		return complex(k, 0) * resonanceResponse(w, wr, zeta) * cmplx.Exp(complex(0, -tau*w))
	}
}

func TestDiskMarginDelayedNarrowResonanceCertified(t *testing.T) {
	for _, zeta := range []float64{1e-3, 1e-4} {
		for _, internal := range []bool{false, true} {
			t.Run(fmt.Sprintf("zeta=%g/internal=%v", zeta, internal), func(t *testing.T) {
				L, l := lightResonanceLoop(t, zeta, internal)
				loop, err := delayLoopFromSystem(L, "delay loop")
				if err != nil {
					t.Fatal(err)
				}
				shifts := []float64{0, -0.5}
				stable, peaks, err := loop.sensitivityPeaks(shifts...)
				if err != nil || !stable {
					t.Fatalf("stable=%v err=%v", stable, err)
				}
				for i, c := range shifts {
					g := func(w float64) float64 { return cmplx.Abs(1/(1+l(w)) + complex(c, 0)) }
					want, _ := oracleNarrowPeak(g, 37.3, zeta)
					if math.Abs(peaks[i].peak-want) > 1e-9*want {
						t.Errorf("shift %g: peak %.15g at %g, oracle %.15g", c, peaks[i].peak, peaks[i].w, want)
					}
				}
			})
		}
	}
}

func TestHinfNormInternalDelayNarrowResonanceCertified(t *testing.T) {
	for _, zeta := range []float64{1e-3, 1e-4} {
		L, l := lightResonanceLoop(t, zeta, false)
		one, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
		if err != nil {
			t.Fatal(err)
		}
		S, err := Feedback(one, L, -1)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := oracleNarrowPeak(func(w float64) float64 { return cmplx.Abs(1 / (1 + l(w))) }, 37.3, zeta)
		got, w, err := HinfNorm(S)
		if err != nil {
			t.Fatalf("ζ=%g: %v", zeta, err)
		}
		if math.Abs(got-want) > 1e-9*want {
			t.Errorf("ζ=%g: HinfNorm %.15g at %g, oracle %.15g", zeta, got, w, want)
		}
	}
}

// G′ and ‖G″‖ of a sample match central differences of the closed form.
func TestDescriptorSampleDerivatives(t *testing.T) {
	dde := scalarDDE(t, -2, 1)
	g := func(w float64) complex128 {
		s := complex(0, w)
		return 1 / (s + 1 + 2*cmplx.Exp(-s))
	}
	r, err := newDescriptorResponse(dde)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0, 0.7, 2.8669, 11.3} {
		s := r.sample(w)
		const h = 1e-4
		d1 := (g(w+h) - g(w-h)) / (2 * h)
		d2 := cmplx.Abs((g(w+h) - 2*g(w) + g(w-h)) / (h * h))
		if cmplx.Abs(s.g1[0]-d1) > 1e-6*cmplx.Abs(d1) || math.Abs(s.g2-d2) > 1e-5*d2 {
			t.Errorf("ω=%g: G′ %v, ‖G″‖ %g; differences %v, %g", w, s.g1[0], s.g2, d1, d2)
		}
	}
}

// A non-decomposable I/O delay matrix on a MIMO internal-delay model goes
// through the residual split of PullDelaysToLFT; the descriptor must still
// evaluate G ∘ e^{−jωτ_ij}.
func TestDescriptorResidualIODelayMIMO(t *testing.T) {
	S, resp := mimoDelayedSensitivity(t, 1)
	tau := []float64{0.1, 0.2, 0.3, 0.1}
	if err := S.SetDelay(mat.NewDense(2, 2, tau)); err != nil {
		t.Fatal(err)
	}
	delayed := func(w float64) [2][2]complex128 {
		g := resp(w)
		for i := range 2 {
			for j := range 2 {
				g[i][j] *= cmplx.Exp(complex(0, -w*tau[i*2+j]))
			}
		}
		return g
	}
	r, err := newDescriptorResponse(S)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0, 0.4, 3.1, 17} {
		got, want := r.sample(w).g, delayed(w)
		for i := range 2 {
			for j := range 2 {
				if d := cmplx.Abs(got[i*2+j] - want[i][j]); d > 1e-12*max(1, cmplx.Abs(want[i][j])) {
					t.Errorf("ω=%g: G[%d][%d] = %v, oracle %v", w, i, j, got[i*2+j], want[i][j])
				}
			}
		}
	}
	want, wantW := oraclePeak(func(w float64) float64 { return oracleSigmaMax2(delayed(w)) }, 300, 1e-3)
	assertPeak(t, "residual io delay", S, want, wantW, 1e-9)
}

// refDescriptorCertify is descriptorResponse.certify before its samples
// were pooled: every sample is freshly allocated.
func refDescriptorCertify(r *descriptorResponse, ws, shifts, peaks, wPeaks []float64, budget int) error {
	start := r.evals
	sample := func(w float64) (*descriptorSample, error) {
		if r.evals-start >= budget {
			return nil, errDelayPeakUnsupported
		}
		s := r.sample(w)
		if !s.regular {
			return nil, errDelayPeakUnsupported
		}
		for k, c := range shifts {
			if v := r.gain(s.g, c); v > peaks[k] {
				peaks[k], wPeaks[k] = v, w
			}
		}
		return s, nil
	}
	type span struct{ a, b *descriptorSample }
	var stack []span
	a, err := sample(ws[0])
	if err != nil {
		return err
	}
	for i := 0; i < len(ws)-1; {
		reach := a.w + 1/max(a.zeta, a.zetaL)
		j := i + 1
		for j+1 < len(ws) && ws[j+1] <= reach {
			j++
		}
		b, err := sample(ws[j])
		if err != nil {
			return err
		}
		stack = append(stack[:0], span{a, b})
		for len(stack) > 0 {
			sp := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			h := (sp.b.w - sp.a.w) / 2
			if h <= 0 || r.certified(sp.a, sp.b, h, shifts, peaks) {
				continue
			}
			mid := sp.a.w + h
			for _, w := range wPeaks {
				if sp.a.w+h/8 < w && w < sp.b.w-h/8 {
					mid = w
					break
				}
			}
			if !(sp.a.w < mid && mid < sp.b.w) {
				return errDelayPeakUnsupported
			}
			sm, err := sample(mid)
			if err != nil {
				return err
			}
			stack = append(stack, span{sm, sp.b}, span{sp.a, sm})
		}
		a, i = b, j
	}
	return nil
}

// TestDescriptorCertifyPoolMatchesFresh checks that pooled samples, reused
// across calls, reproduce the freshly allocated certification bit for bit.
func TestDescriptorCertifyPoolMatchesFresh(t *testing.T) {
	mimo1, _ := mimoDelayedSensitivity(t, 1)
	mimo2, _ := mimoDelayedSensitivity(t, 1.8)
	_, siso := delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, 0.8)
	for _, tc := range []struct {
		name   string
		sys    *System
		shifts []float64
	}{
		{"mimo", mimo1, []float64{0}},
		{"mimo-gain1.8", mimo2, []float64{0}},
		{"mimo3", mimo3DelayedSensitivity(t), []float64{0}},
		{"siso", siso, []float64{0, -0.5, 0.3}},
		{"dde", scalarDDE(t, -2, 1), []float64{0, 0.2}},
	} {
		_, l, res, err := delayPeakSetup(tc.sys)
		if err != nil {
			t.Fatal(err)
		}
		ws := append([]float64{0}, res.w...)
		ref, err := newDescriptorResponse(tc.sys)
		if err != nil {
			t.Fatal(err)
		}
		r, err := newDescriptorResponse(tc.sys)
		if err != nil {
			t.Fatal(err)
		}
		k := len(tc.shifts)
		want, wantW := make([]float64, k), make([]float64, k)
		if err := refDescriptorCertify(ref, ws, tc.shifts, want, wantW, l.pointBudget()); err != nil {
			t.Fatalf("%s: reference: %v", tc.name, err)
		}
		for pass := range 2 {
			got, gotW := make([]float64, k), make([]float64, k)
			r.evals = 0
			if err := r.certify(ws, tc.shifts, got, gotW, l.pointBudget(), errDelayPeakUnsupported); err != nil {
				t.Fatalf("%s pass %d: %v", tc.name, pass, err)
			}
			for i := range k {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) || math.Float64bits(gotW[i]) != math.Float64bits(wantW[i]) {
					t.Errorf("%s pass %d shift %g: (%.17g, %.17g), want (%.17g, %.17g)",
						tc.name, pass, tc.shifts[i], got[i], gotW[i], want[i], wantW[i])
				}
			}
			if r.evals != ref.evals {
				t.Errorf("%s pass %d: %d samples, want %d", tc.name, pass, r.evals, ref.evals)
			}
		}
	}
}
