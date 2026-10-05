package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// delayedSensitivity returns L = k·P·e^{-τs} with P = ss(a, b, c, d) and
// S = 1/(1+L), whose delay sits inside the feedback loop.
func delayedSensitivity(t *testing.T, n int, a, b, c, d []float64, tau, k float64) (L, S *System) {
	t.Helper()
	plant, err := NewFromSlices(n, 1, 1, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := plant.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{k}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if L, err = Series(plant, gain); err != nil {
		t.Fatal(err)
	}
	one, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if S, err = Feedback(one, L, -1); err != nil {
		t.Fatal(err)
	}
	if !S.HasInternalDelay() {
		t.Fatal("sensitivity has no internal delay")
	}
	return L, S
}

// oraclePeak returns sup_ω g(ω) over [0, wMax] from a uniform grid of
// spacing dw plus a log grid below 1, each local maximum refined by
// golden-section search to 1e-14 in ω. g is evaluated in closed form, with no
// library calls.
func oraclePeak(g func(float64) float64, wMax, dw float64) (peak, wPeak float64) {
	var ws []float64
	for w := 1e-4; w < 1; w *= 1.01 {
		ws = append(ws, w)
	}
	for w := 0.0; w <= wMax; w += dw {
		ws = append(ws, w)
	}
	vals := make([]float64, len(ws))
	for i, w := range ws {
		vals[i] = g(w)
	}
	peak, wPeak = math.Inf(-1), 0
	for i := range ws {
		if vals[i] > peak {
			peak, wPeak = vals[i], ws[i]
		}
	}
	for i := 1; i+1 < len(ws); i++ {
		if !(vals[i] >= vals[i-1] && vals[i] >= vals[i+1]) || vals[i] < 0.5*peak {
			continue
		}
		a, b := min(ws[i-1], ws[i+1]), max(ws[i-1], ws[i+1])
		const phi = 0.6180339887498949
		for b-a > 1e-14*max(1, b) {
			x1, x2 := b-phi*(b-a), a+phi*(b-a)
			if g(x1) > g(x2) {
				b = x2
			} else {
				a = x1
			}
		}
		if v := g((a + b) / 2); v > peak {
			peak, wPeak = v, (a+b)/2
		}
	}
	return peak, wPeak
}

func firstOrderDelayedS(k, d, tau float64) func(float64) float64 {
	return func(w float64) float64 {
		s := complex(0, w)
		L := complex(k, 0) * (complex(d, 0) + 1/(s+1)) * cmplx.Exp(-s*complex(tau, 0))
		return cmplx.Abs(1 / (1 + L))
	}
}

func assertPeak(t *testing.T, name string, sys *System, want, wantW, tol float64) {
	t.Helper()
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("%s: HinfNorm: %v", name, err)
	}
	if math.Abs(got-want) > tol*want {
		t.Fatalf("%s: HinfNorm = %.15g, want %.15g", name, got, want)
	}
	if !math.IsInf(wantW, 1) && math.Abs(w-wantW) > 1e-4*max(1, wantW) {
		t.Fatalf("%s: ω = %.12g, want %.12g", name, w, wantW)
	}
	if math.IsInf(wantW, 1) && !math.IsInf(w, 1) {
		t.Fatalf("%s: ω = %g, want +Inf", name, w)
	}
	linf, err := Norm(sys, math.Inf(1))
	if err != nil {
		t.Fatalf("%s: Norm(Inf): %v", name, err)
	}
	if linf != got {
		t.Fatalf("%s: Norm(Inf) = %.15g, HinfNorm %.15g", name, linf, got)
	}
}

func TestHinfNormContinuousInternalDelayStable(t *testing.T) {
	// The ergo U77XJ4 repro: S = 1/(1 + 0.8·e^{-0.5s}/(s+1)).
	L, S := delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, 0.8)
	want, wantW := oraclePeak(firstOrderDelayedS(0.8, 0, 0.5), 200, 1e-3)
	assertPeak(t, "repro", S, want, wantW, 1e-9)
	dm, err := DiskMarginSkew(L, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dm.PeakSensitivity-want) > 1e-9*want {
		t.Fatalf("DiskMarginSkew peak sensitivity %.15g, oracle %.15g", dm.PeakSensitivity, want)
	}

	// Close to the stability boundary: a sharp resonance that only the
	// refined grid resolves.
	wc := bisectRoot(func(w float64) float64 { return math.Atan(w) + 0.5*w - math.Pi }, 1, 10)
	kc := math.Hypot(1, wc)
	k := kc * (1 - 1e-4)
	_, S = delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, k)
	g := firstOrderDelayedS(k, 0, 0.5)
	want, wantW = oraclePeak(g, 2*wc, 1e-5)
	assertPeak(t, "near-marginal", S, want, wantW, 1e-7)
	if want < 1e3 {
		t.Fatalf("near-marginal oracle peak %g, want a resonance", want)
	}

	// Neutral type, |D22| = 0.3 < 1, with its peak at finite frequency above
	// the high-frequency limit 1/0.7.
	_, S = delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0.3}, 0.5, 1)
	want, wantW = oraclePeak(firstOrderDelayedS(1, 0.3, 0.5), 400, 1e-3)
	if want <= 1/0.7 {
		t.Fatalf("neutral oracle peak %g not above the high-frequency limit", want)
	}
	assertPeak(t, "neutral", S, want, wantW, 1e-9)

	// Integrator in the loop: A has an eigenvalue at 0 cancelled in χ.
	_, S = delayedSensitivity(t, 1, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 0.5, 1)
	want, wantW = oraclePeak(func(w float64) float64 {
		s := complex(0, w)
		return cmplx.Abs(s / (s + cmplx.Exp(-0.5*s)))
	}, 200, 1e-3)
	assertPeak(t, "integrator", S, want, wantW, 1e-9)

	// Retarded DDE x' = -x - 2x(t-1), stable for τ < 2π/(3√3).
	dde := scalarDDE(t, -2, 1)
	want, wantW = oraclePeak(func(w float64) float64 {
		s := complex(0, w)
		return cmplx.Abs(1 / (s + 1 + 2*cmplx.Exp(-s)))
	}, 200, 1e-3)
	assertPeak(t, "dde", dde, want, wantW, 1e-9)
}

func TestHinfNormContinuousInternalDelayPeakAtInfinity(t *testing.T) {
	// S = 1/(1 + (0.5 − 0.4/(s+1))e^{-s}): |L(jω)| < 0.5 at every finite ω
	// and tends to 0.5, so |S| approaches its supremum 2 only as ω → ∞.
	_, S := delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{-0.4}, []float64{0.5}, 1, 1)
	got, w, err := HinfNorm(S)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-2) > 1e-9 || !math.IsInf(w, 1) {
		t.Fatalf("HinfNorm = (%.15g, %g), want (2, +Inf)", got, w)
	}

	// A pure delay network, n = 0: 1/(1+0.5e^{-s}) peaks at 2 at ω = π.
	one, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	half, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := half.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	G, err := Feedback(one, half, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !G.HasInternalDelay() {
		t.Fatal("pure delay network has no internal delay")
	}
	got, w, err = HinfNorm(G)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-2) > 1e-9 || math.Abs(math.Remainder(w-math.Pi, 2*math.Pi)) > 1e-5 {
		t.Fatalf("HinfNorm = (%.15g, %g), want (2, π mod 2π)", got, w)
	}
}

func TestHinfNormContinuousInternalDelayUnstable(t *testing.T) {
	wc := bisectRoot(func(w float64) float64 { return math.Atan(w) + 0.5*w - math.Pi }, 1, 10)
	kc := math.Hypot(1, wc)
	for name, sys := range map[string]*System{
		"dde":      scalarDDE(t, -2, 2),
		"loop":     sensitivityOf(delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, 5)),
		"marginal": sensitivityOf(delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, kc)),
		"hidden":   sensitivityOf(delayedSensitivity(t, 2, []float64{0, 0, 0, -1}, []float64{0, 1}, []float64{1, 1}, []float64{0}, 0.5, 1)),
	} {
		norm, w, err := HinfNorm(sys)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !math.IsInf(norm, 1) || !math.IsInf(w, 1) {
			t.Fatalf("%s: HinfNorm = (%g, %g), want (+Inf, +Inf)", name, norm, w)
		}
		if _, err := Norm(sys, math.Inf(1)); !errors.Is(err, ErrDelayUnsupported) {
			t.Fatalf("%s: Norm(Inf) err = %v, want ErrDelayUnsupported", name, err)
		}
	}
}

func sensitivityOf(_, s *System) *System { return s }

func TestHinfNormContinuousInternalDelayUndecidable(t *testing.T) {
	// |D22| = 1.2: the difference operator is unstable (neutral type).
	_, S := delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{1.2}, 0.5, 1)
	if _, _, err := HinfNorm(S); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("HinfNorm err = %v, want ErrDelayUnsupported", err)
	}
	if _, err := Norm(S, math.Inf(1)); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("Norm(Inf) err = %v, want ErrDelayUnsupported", err)
	}
}

// mimoDelayedSensitivity returns S = (I + K·P·diag(e^{-τ_i s}))⁻¹ for a 2×2
// plant with non-symmetric A and D ≠ 0, and its closed-form response.
func mimoDelayedSensitivity(t *testing.T) (*System, func(float64) [2][2]complex128) {
	t.Helper()
	a := []float64{-1, 0.5, -0.3, -2}
	b := []float64{1, 0.2, 0, 1}
	c := []float64{1, 0, 0.4, 1}
	d := []float64{0.1, 0, 0, 0.05}
	k := []float64{0.5, 0.1, -0.2, 0.4}
	tau := []float64{0.3, 0.7}
	plant, err := NewFromSlices(2, 2, 2, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := plant.SetInputDelay(tau); err != nil {
		t.Fatal(err)
	}
	K, err := NewGain(mat.NewDense(2, 2, k), 0)
	if err != nil {
		t.Fatal(err)
	}
	L, err := Series(plant, K)
	if err != nil {
		t.Fatal(err)
	}
	eye, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	S, err := Feedback(eye, L, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !S.HasInternalDelay() {
		t.Fatal("MIMO sensitivity has no internal delay")
	}
	inv2 := func(m [2][2]complex128) [2][2]complex128 {
		det := m[0][0]*m[1][1] - m[0][1]*m[1][0]
		return [2][2]complex128{{m[1][1] / det, -m[0][1] / det}, {-m[1][0] / det, m[0][0] / det}}
	}
	mul := func(x, y [2][2]complex128) [2][2]complex128 {
		var r [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				r[i][j] = x[i][0]*y[0][j] + x[i][1]*y[1][j]
			}
		}
		return r
	}
	resp := func(w float64) [2][2]complex128 {
		s := complex(0, w)
		res := inv2([2][2]complex128{{s - complex(a[0], 0), complex(-a[1], 0)}, {complex(-a[2], 0), s - complex(a[3], 0)}})
		cm := [2][2]complex128{{complex(c[0], 0), complex(c[1], 0)}, {complex(c[2], 0), complex(c[3], 0)}}
		bm := [2][2]complex128{{complex(b[0], 0), complex(b[1], 0)}, {complex(b[2], 0), complex(b[3], 0)}}
		P := mul(mul(cm, res), bm)
		for i := range 2 {
			for j := range 2 {
				P[i][j] += complex(d[i*2+j], 0)
			}
		}
		delay := [2][2]complex128{{cmplx.Exp(-s * complex(tau[0], 0)), 0}, {0, cmplx.Exp(-s * complex(tau[1], 0))}}
		km := [2][2]complex128{{complex(k[0], 0), complex(k[1], 0)}, {complex(k[2], 0), complex(k[3], 0)}}
		l := mul(km, mul(P, delay))
		l[0][0]++
		l[1][1]++
		return inv2(l)
	}
	return S, resp
}

func oracleSigmaMax2(m [2][2]complex128) float64 {
	var fro float64
	for i := range 2 {
		for j := range 2 {
			fro += real(m[i][j])*real(m[i][j]) + imag(m[i][j])*imag(m[i][j])
		}
	}
	det := cmplx.Abs(m[0][0]*m[1][1] - m[0][1]*m[1][0])
	return math.Sqrt((fro + math.Sqrt(max(fro*fro-4*det*det, 0))) / 2)
}

func TestHinfNormContinuousInternalDelayMIMO(t *testing.T) {
	S, resp := mimoDelayedSensitivity(t)
	want, wantW := oraclePeak(func(w float64) float64 { return oracleSigmaMax2(resp(w)) }, 300, 1e-3)
	assertPeak(t, "mimo", S, want, wantW, 1e-9)
}

// Pade(n) replaces each delay by a rational approximant whose phase error
// vanishes as n grows, so its H∞ norm converges to the exact-delay one.
func TestHinfNormContinuousInternalDelayPadeConverges(t *testing.T) {
	_, siso := delayedSensitivity(t, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0.5, 0.8)
	mimo, _ := mimoDelayedSensitivity(t)
	for name, sys := range map[string]*System{"siso": siso, "mimo": mimo} {
		exact, _, err := HinfNorm(sys)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		prev := math.Inf(1)
		for _, n := range []int{1, 2, 4, 6} {
			approx, err := sys.Pade(n)
			if err != nil {
				t.Fatal(err)
			}
			got, _, err := HinfNorm(approx)
			if err != nil {
				t.Fatalf("%s Pade(%d): %v", name, n, err)
			}
			diff := math.Abs(got - exact)
			if diff > prev*1.01 {
				t.Fatalf("%s Pade(%d): |%.12g − %.12g| = %g grew from %g", name, n, got, exact, diff, prev)
			}
			prev = diff
		}
		if prev > 1e-6*exact {
			t.Fatalf("%s: Pade(6) error %g, want < 1e-6 relative", name, prev)
		}
	}
}
