package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type expansionCase struct {
	name          string
	d22, d12, d21 []float64
	h             float64
}

var (
	// sup |G∞| is attained off φ = 0, π, where a = Re⟨g0, g1⟩ ≠ 0: the
	// finite peaks exceed it by about |a|/(lim·ω) and the highest lies
	// several periods past the grid end.
	expansionFinite = []expansionCase{
		{"near-90",
			[]float64{0.7649551391810011, -0.4053354642068646, -0.5986288790758891, -0.735255719141454},
			[]float64{-0.14804120333685122, 0.8436249853687099}, []float64{-0.40213426719214174, -0.5844303786493328},
			0.3686756259842685},
		{"near-410",
			[]float64{-0.47675042569597786, -0.7643529185038457, 0.6634435446928469, -0.4697636633641407},
			[]float64{-0.477317332593425, 0.9680775762045155}, []float64{0.9231522571301278, 0.678449994147619},
			0.11037823685884307},
	}
	// sup |G∞| is attained at φ = π with κ < 0: every passage stays below.
	expansionLimit = expansionCase{"limit",
		[]float64{-0.6008003587828393, 0.2739929683353646, -0.6159222999747516, -0.37189479027245975},
		[]float64{-0.6331344630555211, 0.728526179491567}, []float64{0.24127665933031084, 0.7586790465022555},
		0.4851105119841035}
)

func (c expansionCase) build(t testing.TB) (*System, func(float64) float64, float64) {
	t.Helper()
	sys, gain, gInf := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, 2 * c.h}, [2]float64{1, 2})
	const k = 1 << 18
	lim := 0.0
	for i := range k {
		lim = max(lim, gInf(2*math.Pi*float64(i)/k))
	}
	return sys, gain, lim
}

// passageMax returns the largest |G| over the passage of φ = ωh through phi
// nearest wc, on a dense grid refined by golden-section search.
func passageMax(gain func(float64) float64, h, phi, wc float64) (float64, float64) {
	k := math.Round((wc*h - phi) / (2 * math.Pi))
	w0, dw := (phi+2*math.Pi*k)/h, 0.5/h
	best, bw := 0.0, 0.0
	for j := -400; j <= 400; j++ {
		w := w0 + dw*float64(j)/400
		if g := gain(w); g > best {
			best, bw = g, w
		}
	}
	bw, best = goldenMax(bw-dw/400, bw+dw/400, bw, best, gain)
	return best, bw
}

// checkExpansionOrder checks the expansion of sys against gain at equal
// phases φ = ωh, where the remainder scales as ω^−(N+1).
func checkExpansionOrder(t *testing.T, sys *System, gain func(float64) float64, h float64) {
	t.Helper()
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	if x == nil {
		t.Fatal("no expansion")
	}
	var errs, ws []float64
	for _, wc := range []float64{50, 400} {
		w := (1.234 + 2*math.Pi*math.Round(wc*h/(2*math.Pi))) / h
		err := math.Abs(gain(w) - x.model(w*h, 1/w))
		first := math.Abs(gain(w) - x.model(w*h, 0))
		if !(first > 1e3*err) {
			t.Fatalf("ω = %g: |G − model| = %g, |G − G∞| = %g", w, err, first)
		}
		errs, ws = append(errs, err), append(ws, w)
	}
	want := math.Pow(ws[1]/ws[0], expansionOrder+1)
	if r := errs[0] / errs[1]; !(r > want/2 && r < 2*want) {
		t.Fatalf("remainder ratio %g (%g, %g), want about %g (O(1/ω^(N+1)))", r, errs[0], errs[1], want)
	}
}

func TestHighFrequencyExpansionMatchesResponse(t *testing.T) {
	for _, c := range append([]expansionCase{expansionLimit}, expansionFinite...) {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, _ := c.build(t)
			checkExpansionOrder(t, sys, gain, c.h)
		})
	}
}

// TestHighFrequencyExpansionLoopDynamics checks the expansion of a 1×2 gain
// whose delay loop has dynamics (C2, B2 ≠ 0, so every H22,k ≠ 0) against
// the response solved directly from the LFT equations
//
//	(jωI − A)·x − B2·Δ·z = B1,  −C2·x + (I − D22·Δ)·z = D21,  G = C1·x + D11 + D12·Δ·z.
func TestHighFrequencyExpansionLoopDynamics(t *testing.T) {
	const n, nu, nd = 2, 2, 2
	a := []float64{-1, 2, -0.7, -3}
	b1 := []float64{1, 0.2, 0.5, -1}
	b2 := []float64{0.3, -0.4, 0.6, 1}
	c1 := []float64{1, -0.4}
	c2 := []float64{0.5, 0.2, -0.3, 0.8}
	d11 := []float64{0.1, 0.05}
	d12 := []float64{-0.6, 0.7}
	d21 := []float64{0.25, 0.75, -0.5, 0.3}
	d22 := []float64{-0.4, 0.3, -0.35, -0.2}
	h := 0.37
	B := make([]float64, 0, n*(nu+nd))
	for i := range n {
		B = append(append(B, b1[i*nu:(i+1)*nu]...), b2[i*nd:(i+1)*nd]...)
	}
	C := append(append([]float64{}, c1...), c2...)
	D := append(append([]float64{}, d11...), d12...)
	for i := range nd {
		D = append(append(D, d21[i*nu:(i+1)*nu]...), d22[i*nd:(i+1)*nd]...)
	}
	H, err := NewFromSlices(n, nu+nd, 1+nd, a, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{h, 2 * h})
	if err != nil {
		t.Fatal(err)
	}
	gain := func(w float64) float64 {
		s := complex(0, w)
		dl := []complex128{cmplx.Exp(-s * complex(h, 0)), cmplx.Exp(-s * complex(2*h, 0))}
		const k = n + nd
		var norm2 float64
		for j := range nu {
			var m [k][k + 1]complex128
			for r := range n {
				for c := range n {
					m[r][c] = complex(-a[r*n+c], 0)
				}
				m[r][r] += s
				for c := range nd {
					m[r][n+c] = complex(-b2[r*nd+c], 0) * dl[c]
				}
				m[r][k] = complex(b1[r*nu+j], 0)
			}
			for r := range nd {
				for c := range n {
					m[n+r][c] = complex(-c2[r*n+c], 0)
				}
				for c := range nd {
					m[n+r][n+c] = complex(-d22[r*nd+c], 0) * dl[c]
				}
				m[n+r][n+r]++
				m[n+r][k] = complex(d21[r*nu+j], 0)
			}
			for c := range k {
				piv := c
				for r := c + 1; r < k; r++ {
					if cmplx.Abs(m[r][c]) > cmplx.Abs(m[piv][c]) {
						piv = r
					}
				}
				m[c], m[piv] = m[piv], m[c]
				for r := range k {
					if r != c {
						f := m[r][c] / m[c][c]
						for q := c; q <= k; q++ {
							m[r][q] -= f * m[c][q]
						}
					}
				}
			}
			g := complex(d11[j], 0)
			for c := range n {
				g += complex(c1[c], 0) * m[c][k] / m[c][c]
			}
			for c := range nd {
				g += complex(d12[c], 0) * dl[c] * m[n+c][k] / m[n+c][n+c]
			}
			norm2 += real(g)*real(g) + imag(g)*imag(g)
		}
		return math.Sqrt(norm2)
	}
	checkExpansionOrder(t, sys, gain, h)
}

func TestHinfNormFinitePeakPastLimitScan(t *testing.T) {
	for _, c := range expansionFinite {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, lim := c.build(t)
			want, wWant := oraclePeak(gain, 600, 1e-3)
			if !(want > lim*(1+1e-5)) {
				t.Fatalf("oracle peak %.12g must exceed sup |G∞| %.12g", want, lim)
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}

func TestHinfNormLimitBelowEveryPassage(t *testing.T) {
	c := expansionLimit
	sys, gain, lim := c.build(t)
	var prev float64
	for _, wc := range []float64{1e3, 1e4, 1e5} {
		m, w := passageMax(gain, c.h, math.Pi, wc)
		excess := (m - lim) * w * w
		if !(excess < 0) {
			t.Fatalf("passage at %g peaks at %.15g above sup |G∞| %.15g", w, m, lim)
		}
		if wc > 1e3 && math.Abs(excess-prev) > 0.02*math.Abs(prev) {
			t.Fatalf("(max − lim)·ω² = %g at %g, %g a decade earlier: not in the 1/ω² regime", excess, w, prev)
		}
		prev = excess
	}
	if want, _ := oraclePeak(gain, 3000, 2e-3); !(want < lim) {
		t.Fatalf("oracle peak %.12g reaches sup |G∞| %.12g", want, lim)
	}
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	finite, W, ok := x.limitVerdict(lim, 1e-10, 10, 1e6)
	if !ok || finite || !(W < 1e4) {
		t.Fatalf("limitVerdict = %v, %g, %v; want +Inf certified from a finite horizon", finite, W, ok)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-9*lim {
		t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
	}
}

// limitAtInfCases are delay models whose peak is the ω → ∞ limit, decided by
// limitVerdict: "limit" and "flat" with k = 2 ("flat" with κ small, so the
// expansion must reach far), "k20" with k = 20 and "far" with k = 3.
var limitAtInfCases = []struct {
	expansionCase
	k float64
}{
	{expansionLimit, 2},
	{expansionCase{"flat",
		[]float64{0.5888561770149723, -0.10453534772349099, 0.26009762791254687, -0.4187778122618928},
		[]float64{-1.3972940735741266, 0.09637692295337041}, []float64{-1.601893769323554, 0.40406439147179873},
		0.3547077050852388}, 2},
	{expansionCase{"k20",
		[]float64{-0.6267854986926139, -0.22753476407416448, 0.7604859326403655, 0.3445225407726179},
		[]float64{2.0994008736286496, -0.7771379653940502}, []float64{2.8964548762636153, -1.852398922038503},
		0.23582769791420977}, 20},
	{phaseScanLimit, 3},
}

// phaseScanLimit, with k = 3, has its +Inf horizon W of limitVerdict about
// 8.5 times the dense scan end, so phaseScan covers most of the tail.
var phaseScanLimit = expansionCase{"far",
	[]float64{-0.4379881585197284, 0.39331592038980734, -0.08540018365047253, -0.200674608663282},
	[]float64{-0.14302132366907738, -2.4255577468517746}, []float64{2.4202689080449487, 0.08789889153068065},
	0.3411510888275282}

// TestModelMaxMatchesFullScan checks the pruned modelMax against the model
// evaluated directly at every sample with every local maximum refined.
func TestModelMaxMatchesFullScan(t *testing.T) {
	for _, c := range limitAtInfCases {
		t.Run(c.name, func(t *testing.T) {
			sys, _, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, c.k * c.h}, [2]float64{1, c.k})
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
			n := 1024 * int(c.k)
			step := 2 * math.Pi / float64(n)
			for _, tt := range []float64{0.2, 0.05, 1e-2, 1e-4, 1e-8} {
				f := func(phi float64) float64 { return x.model(phi, tt) }
				v := make([]float64, n)
				for i := range v {
					v[i] = f(float64(i) * step)
				}
				want := 0.0
				for i := range v {
					if v[i] >= v[(i+n-1)%n] && v[i] >= v[(i+1)%n] {
						_, g := goldenMax(float64(i-1)*step, float64(i+1)*step, float64(i)*step, v[i], f)
						want = max(want, g)
					}
				}
				if got := x.modelMax(tt); math.Abs(got-want) > 1e-12*want {
					t.Errorf("t = %g: modelMax = %.16g, full scan %.16g", tt, got, want)
				}
			}
		})
	}
}

// slightPeakCases have k = 4 and a finite peak above sup |G∞| by only
// about 1e-6 relative, far past the grid end, so tailGain alone closes the
// tail only near ω ~ 1/(peak − lim).
var slightPeakCases = []expansionCase{
	{"A230",
		[]float64{-0.4679143293904018, 0.5568820700329411, 0.5970290865070236, 0.04088492140928639},
		[]float64{-2.7256683166786164, 4.53724253238323}, []float64{-4.626184319536489, -0.5701210155921768},
		0.38635252096584244},
	{"A26",
		[]float64{-0.3076283919048543, -0.5018015133454836, 0.47779093311037, -0.14453698005791601},
		[]float64{-2.335077036398033, 1.74556187352366}, []float64{-4.571548904310475, 4.6882244033680855},
		0.19446812828709992},
}

func TestHinfNormSlightFinitePeakPastLimit(t *testing.T) {
	for _, c := range slightPeakCases {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, 4 * c.h}, [2]float64{1, 4})
			want, wWant := oraclePeak(gain, 1500, 1e-3)
			for _, wc := range []float64{3e3, 1e4, 1e5} {
				if m, w := passageMax(gain, c.h, math.Mod(wWant*c.h, 2*math.Pi), wc); !(m < want) {
					t.Fatalf("passage at %g peaks at %.15g above the oracle peak %.15g", w, m, want)
				}
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}

// TestTailBelowBoundsResponse checks that where tailBelow certifies a level
// past ω = R, a dense sweep of the response on [R, 4R] stays below it, and
// that the certificate is reached within a few times the peak frequency.
func TestTailBelowBoundsResponse(t *testing.T) {
	c := slightPeakCases[0]
	sys, gain, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, 4 * c.h}, [2]float64{1, 4})
	want, wWant := oraclePeak(gain, 1500, 1e-3)
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	level := want * (1 + 1e-10)
	R := 100.0
	for !x.tailBelow(level, 1/R) {
		if R *= 2; R > 64*wWant {
			t.Fatalf("no certificate up to ω = %g, peak at %g", R, wWant)
		}
	}
	if x.tailBelow(want*(1-1e-9), 2/wWant) {
		t.Fatalf("tailBelow certifies %.15g past ω = %g, below the peak %.15g at %g", want*(1-1e-9), wWant/2, want, wWant)
	}
	for w := R; w <= 4*R; w += 1e-3 {
		if g := gain(w); !(g <= level) {
			t.Fatalf("|G(j%g)| = %.15g above the certified %.15g past %g", w, g, level, R)
		}
	}
}

// matrixGain returns U·diag(G, d)·V for the delay model G of c, with u, v
// the row-major U, V, and σ_max of its response in closed form from G(jω).
func (c expansionCase) matrixGain(t *testing.T, d float64, u, v []float64) (*System, func(float64) float64) {
	t.Helper()
	sys, resp, _ := feedthroughDifferenceResponse(t, c.d22, c.d12, c.d21, []float64{c.h, 2 * c.h}, [2]float64{1, 2})
	gain := func(r, c int, d []float64) *System {
		g, err := NewGain(mat.NewDense(r, c, d), 0)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	ap, err := Append(sys, gain(1, 1, []float64{d}))
	if err != nil {
		t.Fatal(err)
	}
	if ap, err = Series(gain(2, 2, v), ap); err != nil {
		t.Fatal(err)
	}
	if ap, err = Series(ap, gain(2, 2, u)); err != nil {
		t.Fatal(err)
	}
	sigma := func(w float64) float64 {
		g := [2]complex128{resp(w), complex(d, 0)}
		var m [4]complex128
		var f float64
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					m[i*2+j] += complex(u[i*2+k]*v[k*2+j], 0) * g[k]
				}
				f += real(m[i*2+j])*real(m[i*2+j]) + imag(m[i*2+j])*imag(m[i*2+j])
			}
		}
		det := cmplx.Abs(m[0]*m[3] - m[1]*m[2])
		return math.Sqrt((f + math.Sqrt(max(f*f-4*det*det, 0))) / 2)
	}
	return ap, sigma
}

// TestHinfNormMatrixFinitePeakPastLimit checks 2×2 gains built from the
// expansionFinite cases, block diagonal and coupled by constant U, V, whose
// finite peak lies past the grid end above sup σ_max(G∞).
func TestHinfNormMatrixFinitePeakPastLimit(t *testing.T) {
	for _, c := range expansionFinite {
		for _, m := range []struct {
			name string
			d    float64
			u, v []float64
		}{
			{"diag", 0.01, []float64{1, 0, 0, 1}, []float64{1, 0, 0, 1}},
			{"coupled", 0.3, []float64{1, 0.4, -0.3, 0.8}, []float64{0.7, -0.2, 0.5, 1.1}},
		} {
			t.Run(c.name+"/"+m.name, func(t *testing.T) {
				sys, sigma := c.matrixGain(t, m.d, m.u, m.v)
				checkExpansionOrder(t, sys, sigma, c.h)
				want, wWant := oraclePeak(sigma, 600, 1e-3)
				if far, _ := passageMax(sigma, c.h, math.Mod(wWant*c.h, 2*math.Pi), 1e5); !(want > far*(1+1e-6)) {
					t.Fatalf("oracle peak %.12g not above the passage at 1e5, %.12g", want, far)
				}
				got, w, err := HinfNorm(sys)
				if err != nil {
					t.Fatalf("HinfNorm: %v", err)
				}
				if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
					t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
				}
			})
		}
	}
}

// TestHinfNormSharpResonanceLimit has one delay with D22 near 1, so sup |G∞|
// at φ = 0 is a resonance of width about 1 − D22 in φ, where c and a′²/α
// nearly cancel in κ. The passage maxima of the closed-form response rise
// toward the limit from below, so the peak is the limit at ω = +Inf.
func TestHinfNormSharpResonanceLimit(t *testing.T) {
	for _, d22 := range []float64{0.999, 0.9993457031381653, 0.99999} {
		t.Run(fmt.Sprint(d22), func(t *testing.T) { checkSharpResonanceLimit(t, d22) })
	}
}

// sharpResonanceTau is the delay of sharpResonanceModel.
const sharpResonanceTau = 0.6705489857287816

// sharpResonanceModel returns the 1×2 model of
// TestHinfNormSharpResonanceLimit, its gain and |G∞(φ)| in closed form.
func sharpResonanceModel(t *testing.T, d22 float64) (*System, func(float64) float64, func(float64) float64) {
	t.Helper()
	const a, tau = -1.8797942063939619, sharpResonanceTau
	b1 := []float64{-0.02704717913348858, -0.1805137473767608}
	const b2, c1, c2 = 0.20395346902869235, 0.5803594342648284, -0.5430833291216368
	d11 := []float64{0.7650979509126943, 0.3709492250386599}
	const d12 = 0.6232424759105828
	d21 := []float64{0.5963224441233179, -0.7001397032844845}
	H, err := NewFromSlices(1, 3, 2, []float64{a}, []float64{b1[0], b1[1], b2}, []float64{c1, c2},
		[]float64{d11[0], d11[1], d12, d21[0], d21[1], d22}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{tau})
	if err != nil {
		t.Fatal(err)
	}
	row := func(r, z complex128) float64 {
		den := 1 - z*(complex(c2, 0)*r*complex(b2, 0)+complex(d22, 0))
		var n2 float64
		for i := range 2 {
			v := complex(c2, 0)*r*complex(b1[i], 0) + complex(d21[i], 0)
			g := complex(c1, 0)*r*complex(b1[i], 0) + complex(d11[i], 0) +
				(complex(c1, 0)*r*complex(b2, 0)+complex(d12, 0))*z*v/den
			n2 += real(g)*real(g) + imag(g)*imag(g)
		}
		return math.Sqrt(n2)
	}
	gain := func(w float64) float64 {
		return row(1/(complex(0, w)-complex(a, 0)), cmplx.Exp(complex(0, -w*tau)))
	}
	gInf := func(phi float64) float64 { return row(0, cmplx.Exp(complex(0, -phi))) }
	return sys, gain, gInf
}

func checkSharpResonanceLimit(t *testing.T, d22 float64) {
	const tau = sharpResonanceTau
	sys, gain, gInf := sharpResonanceModel(t, d22)
	const k = 1 << 16
	lim, at := 0.0, 0.0
	for i := range k {
		if g := gInf(2 * math.Pi * float64(i) / k); g > lim {
			lim, at = g, 2*math.Pi*float64(i)/k
		}
	}
	at, lim = goldenMax(at-2*math.Pi/k, at+2*math.Pi/k, at, lim, gInf)
	if math.Abs(math.Remainder(at, 2*math.Pi)) > 1e-6 {
		t.Fatalf("sup |G∞| at φ = %g, want 0", at)
	}
	prev := 0.0
	for _, kp := range []float64{1e2, 1e3, 1e4, 1e5} {
		w0, dw := 2*math.Pi*kp/tau, 20*(1-d22)/tau
		best, bw := 0.0, w0
		for j := -2000; j <= 2000; j++ {
			w := w0 + dw*float64(j)/2000
			if g := gain(w); g > best {
				best, bw = g, w
			}
		}
		_, best = goldenMax(bw-dw/2000, bw+dw/2000, bw, best, gain)
		if !(best < lim && best > prev) {
			t.Fatalf("passage %g peaks at %.15g, previous %.15g, sup |G∞| %.15g", kp, best, prev, lim)
		}
		prev = best
	}
	if peak, _ := oraclePeak(gain, 2000, 1e-3); !(peak < lim) {
		t.Fatalf("oracle peak %.15g reaches sup |G∞| %.15g", peak, lim)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-9*lim {
		t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
	}
}

func BenchmarkHinfNorm_DelayLimitAtInf(b *testing.B) {
	for _, c := range limitAtInfCases {
		sys, _, _ := feedthroughDifferenceLoop(b, c.d22, c.d12, c.d21, []float64{c.h, c.k * c.h}, [2]float64{1, c.k})
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, w, err := HinfNorm(sys); err != nil || !math.IsInf(w, 1) {
					b.Fatal(w, err)
				}
			}
		})
	}
}
