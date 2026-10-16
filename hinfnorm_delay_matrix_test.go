package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// lftCase is a 2×2 gain with n states and delays τ = (h, k·h) on two
// internal channels:
//
//	ẋ = A·x + B1·u + B2·w,  z = C2·x + D21·u + D22·w,  y = C1·x + D11·u + D12·w,  w = Δz.
type lftCase struct {
	name                   string
	n                      int
	k, h                   float64
	a, b1, b2, c1, c2      []float64
	d11, d12, d21, d22     []float64
	wantW, wantMax, wantLo float64 // peak frequency, oracle scan end and step
}

func (c lftCase) system(t *testing.T) *System {
	t.Helper()
	n := c.n
	B := make([]float64, 0, n*4)
	for i := range n {
		B = append(append(B, c.b1[i*2:i*2+2]...), c.b2[i*2:i*2+2]...)
	}
	C := append(append([]float64{}, c.c1...), c.c2...)
	D := make([]float64, 0, 16)
	for i := range 2 {
		D = append(append(D, c.d11[i*2:i*2+2]...), c.d12[i*2:i*2+2]...)
	}
	for i := range 2 {
		D = append(append(D, c.d21[i*2:i*2+2]...), c.d22[i*2:i*2+2]...)
	}
	H, err := NewFromSlices(n, 4, 4, c.a, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{c.h, c.k * c.h})
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sigma2x2(m [4]complex128) float64 {
	var f float64
	for _, v := range m {
		f += real(v)*real(v) + imag(v)*imag(v)
	}
	det := cmplx.Abs(m[0]*m[3] - m[1]*m[2])
	return math.Sqrt((f + math.Sqrt(max(f*f-4*det*det, 0))) / 2)
}

// sigma returns σ_max(G(jω)), solving (jωI − A)·x − B2·Δ·z = B1,
// −C2·x + (I − D22·Δ)·z = D21 and G = C1·x + D11 + D12·Δ·z by Gauss–Jordan
// elimination.
func (c lftCase) sigma(w float64) float64 {
	n := c.n
	k := n + 2
	s := complex(0, w)
	dl := [2]complex128{cmplx.Exp(-s * complex(c.h, 0)), cmplx.Exp(-s * complex(c.k*c.h, 0))}
	var m [5][7]complex128
	for r := range n {
		for q := range n {
			m[r][q] = complex(-c.a[r*n+q], 0)
		}
		m[r][r] += s
		for q := range 2 {
			m[r][n+q] = complex(-c.b2[r*2+q], 0) * dl[q]
			m[r][k+q] = complex(c.b1[r*2+q], 0)
		}
	}
	for r := range 2 {
		for q := range n {
			m[n+r][q] = complex(-c.c2[r*n+q], 0)
		}
		for q := range 2 {
			m[n+r][n+q] = complex(-c.d22[r*2+q], 0) * dl[q]
			m[n+r][k+q] = complex(c.d21[r*2+q], 0)
		}
		m[n+r][n+r]++
	}
	for col := range k {
		piv := col
		for r := col + 1; r < k; r++ {
			if cmplx.Abs(m[r][col]) > cmplx.Abs(m[piv][col]) {
				piv = r
			}
		}
		m[col], m[piv] = m[piv], m[col]
		for r := range k {
			if r != col {
				f := m[r][col] / m[col][col]
				for q := col; q < k+2; q++ {
					m[r][q] -= f * m[col][q]
				}
			}
		}
	}
	var g [4]complex128
	for i := range 2 {
		for j := range 2 {
			v := complex(c.d11[i*2+j], 0)
			for q := range n {
				v += complex(c.c1[i*n+q], 0) * m[q][k+j] / m[q][q]
			}
			for q := range 2 {
				v += complex(c.d12[i*2+q], 0) * dl[q] * m[n+q][k+j] / m[n+q][n+q]
			}
			g[i*2+j] = v
		}
	}
	return sigma2x2(g)
}

// gInf returns σ_max(D11 + D12·Δ·(I − D22·Δ)⁻¹·D21) at phase φ = ωh.
func (c lftCase) gInf(phi float64) float64 {
	dl := [2]complex128{cmplx.Exp(complex(0, -phi)), cmplx.Exp(complex(0, -c.k*phi))}
	m := [4]complex128{1 - complex(c.d22[0], 0)*dl[0], -complex(c.d22[1], 0) * dl[1], -complex(c.d22[2], 0) * dl[0], 1 - complex(c.d22[3], 0)*dl[1]}
	det := m[0]*m[3] - m[1]*m[2]
	var g [4]complex128
	for j := range 2 {
		d0, d1 := complex(c.d21[j], 0), complex(c.d21[2+j], 0)
		z0 := (m[3]*d0 - m[1]*d1) / det
		z1 := (m[0]*d1 - m[2]*d0) / det
		for i := range 2 {
			g[i*2+j] = complex(c.d11[i*2+j], 0) + complex(c.d12[i*2], 0)*dl[0]*z0 + complex(c.d12[i*2+1], 0)*dl[1]*z1
		}
	}
	return sigma2x2(g)
}

// circleMax returns the largest of f at 2^18 points of [0, 2π), local
// maxima refined by golden-section search.
func circleMax(f func(float64) float64) float64 {
	const k = 1 << 18
	step := 2 * math.Pi / k
	v := make([]float64, k)
	for i := range v {
		v[i] = f(float64(i) * step)
	}
	best := 0.0
	for i := range v {
		if v[i] >= v[(i+k-1)%k] && v[i] >= v[(i+1)%k] {
			_, g := goldenMax(float64(i-1)*step, float64(i+1)*step, float64(i)*step, v[i], f)
			best = max(best, g)
		}
	}
	return best
}

// slightMatrixPeakCases are random 2×2 gains whose finite peak lies past the
// grid end, above sup σ_max(G∞) by only 1e-5 and 2e-6 relative. Before the
// matrix interval bound of the expansion, the tail closed only by tailGain,
// past the delay grid budget.
var slightMatrixPeakCases = []lftCase{
	{name: "n2", n: 2, k: 4, h: 0.2241773747134388,
		a:       []float64{-1.5845892485192312, -0.9615870491313611, -1.1318167767410126, -1.343676827593916},
		b1:      []float64{-0.681635576432122, -0.40186008623576935, 0.14213611957992178, -0.8622242146341708},
		b2:      []float64{-0.18645061318726863, 0.6951787890364394, 0.5245311302435186, -0.376596722921179},
		c1:      []float64{0.7029619860465697, 0.8456275880463768, 0.20029413816215663, -0.981052991556323},
		c2:      []float64{0.07146241964121236, 0.0969600080306037, 0.10935389935192999, 0.46607869997963286},
		d11:     []float64{0.03627999427517786, 0.348065606031262, 0.017937559795325783, -0.6148262858369582},
		d12:     []float64{-0.7109130804395933, -0.5175506431048342, 0.44723754708547214, -2.5351068144429405},
		d21:     []float64{0.3214847551513057, -0.7380003800566692, -0.6534508374438419, 0.938245204224665},
		d22:     []float64{-0.23792789853753246, -0.04701868373003064, -0.2048688554557003, -0.3597517777586704},
		wantMax: 400, wantLo: 0.004},
	{name: "n3", n: 3, k: 4, h: 0.5146770192946389,
		a: []float64{-1.258915989758148, 0.5116422392099202, 1.3376112907287436, -1.3232851086399502,
			-0.9777586608458906, -0.1117225194419712, 0.16881257840806452, -0.07018775350670026, -3.188738021155623},
		b1:      []float64{0.6469635595888751, -0.48604354127282906, 0.090885754909029, -0.15279096378177026, 0.7310027740619334, 0.7170506763126799},
		b2:      []float64{-0.13880595304247462, 0.8695298839160099, 0.6428527288682585, 0.33618979066007926, -0.8109729826642819, 0.3117199769922243},
		c1:      []float64{0.939083008689087, 0.6355555162253137, -0.3173663422744898, -0.12441147888173236, -0.39230477637317485, -0.8749391530134372},
		c2:      []float64{-0.03900840570964031, -0.0388382223004963, 0.3082973736690484, -0.3962178331635787, 0.6280587869043533, 0.5573372750240412},
		d11:     []float64{0.895768915375138, -0.5145591164024097, -0.33340285318247936, 0.5441696176009658},
		d12:     []float64{-0.8626277916296774, 0.5510784380679018, -1.5370618131476224, 0.7991605764194553},
		d21:     []float64{-0.4059149136733134, -1.436424420790968, 1.1854886172906816, -0.6461480283764229},
		d22:     []float64{-0.20170457247937107, 0.39612501892478447, 0.4628055852488583, -0.11220955278579366},
		wantMax: 1200, wantLo: 0.002},
}

func TestHinfNormMatrixSlightPeakPastLimit(t *testing.T) {
	for _, c := range slightMatrixPeakCases {
		t.Run(c.name, func(t *testing.T) {
			sys := c.system(t)
			lim := circleMax(c.gInf)
			want, wWant := oraclePeak(c.sigma, c.wantMax, c.wantLo)
			if !(want > lim*(1+1e-6)) || !(wWant > 200) {
				t.Fatalf("oracle peak %.12g at %g, want above sup σ_max(G∞) %.12g past the grid", want, wWant, lim)
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

// TestHinfNormSlightPeakFarPastGrid checks a SISO gain, k = 10, whose
// finite peak above sup |G∞| first shows near ω = 1500 but keeps rising to
// about 2971, so tailBelow cannot certify the tail at the first sample
// above the limit and the scan must go on.
func TestHinfNormSlightPeakFarPastGrid(t *testing.T) {
	d22 := []float64{-0.655396979015757, -0.7202641867565934, -0.21916713365340643, -0.42845141241946794}
	d12 := []float64{-0.9751179032961488, 2.8522588646674873}
	d21 := []float64{-2.865621575643168, 2.572234902629602}
	h := 0.4791651479878558
	sys, gain, gInf := feedthroughDifferenceLoop(t, d22, d12, d21, []float64{h, 10 * h}, [2]float64{1, 10})
	lim := circleMax(gInf)
	want, wWant := oraclePeak(gain, 4000, 0.002)
	if !(want > lim*(1+1e-8)) || !(wWant > 2000) {
		t.Fatalf("oracle peak %.12g at %g, want above sup |G∞| %.12g far out", want, wWant, lim)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
		t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
	}
}

func matrixExpansion(t *testing.T, c lftCase) *highFrequencyExpansion {
	t.Helper()
	e, err := newDelayLFT(c.system(t))
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	if x == nil || x.vector {
		t.Fatal("no matrix expansion")
	}
	return x
}

// TestIntervalSigmaBoundsModel checks that intervalSigma over [t1, t2] is the
// model at t1 = t2 and bounds it at every t of a fine grid otherwise, by
// little more than the largest of those.
func TestIntervalSigmaBoundsModel(t *testing.T) {
	x := matrixExpansion(t, slightMatrixPeakCases[1])
	g := make([]complex128, (expansionOrder+1)*len(x.g[0]))
	var n [expansionOrder + 1]float64
	for _, phi := range []float64{0.3, 1.7, 2.9, 4.4} {
		x.at(phi)
		x.flatten(g, &n)
		for _, t2 := range []float64{0.2, 0.05, 1e-3} {
			t1 := t2 / 2
			x.prepare(t1, t2)
			b := x.intervalSigma(g, &n, t1, t2, &x.gap)
			top := 0.0
			for i := range 1001 {
				tt := t1 + (t2-t1)*float64(i)/1000
				m := x.model(phi, tt)
				top = max(top, m)
				if !(m <= b) {
					t.Fatalf("φ = %g, t = %g: model %.16g above the bound %.16g over [%g, %g]", phi, tt, m, b, t1, t2)
				}
			}
			slack := 1e-12
			for k := 2; k <= expansionOrder; k++ {
				slack += x.gap[k] * n[k]
			}
			if b-top > slack {
				t.Errorf("φ = %g, [%g, %g]: bound %.16g exceeds the model maximum %.16g by more than the secant gaps %g", phi, t1, t2, b, top, slack)
			}
			x.at(phi)
			x.prepare(t2, t2)
			if e := x.intervalSigma(g, &n, t2, t2, &x.gap); e != x.model(phi, t2) {
				t.Fatalf("φ = %g: intervalSigma(%g, %g) = %.16g, model %.16g", phi, t2, t2, e, x.model(phi, t2))
			}
			x.at(phi)
		}
	}
}

// TestMatrixModelMaxMatchesFullScan checks the pruned modelMax of a matrix
// gain against the model at every sample, local maxima refined, and
// modelMaxOver against the largest modelMax inside its interval.
func TestMatrixModelMaxMatchesFullScan(t *testing.T) {
	x := matrixExpansion(t, slightMatrixPeakCases[0])
	n := 1024 * int(x.hf.kmax)
	step := 2 * math.Pi / float64(n)
	for _, tt := range []float64{0.05, 1e-2, 1e-4} {
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
		over := x.modelMaxOver(tt/2, tt, math.Inf(1))
		for _, s := range []float64{0.5, 0.75, 1} {
			if m := x.modelMax(s * tt); !(m <= over) {
				t.Errorf("modelMax(%g) = %.16g above modelMaxOver(%g, %g) = %.16g", s*tt, m, tt/2, tt, over)
			}
		}
	}
}

// TestPhaseMaskBoundsResponse checks, for a SISO gain whose peak is the
// ω → ∞ limit with a horizon W of limitVerdict far past the dense scan,
// that phaseMask leaves out most phases of an octave of ω, that the
// response stays below the level at every ω of the octave whose phase it
// leaves out, and that HinfNorm reports the limit at +Inf.
func TestPhaseMaskBoundsResponse(t *testing.T) {
	c := phaseScanLimit
	h := c.h
	sys, gain, gInf := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{h, 3 * h}, [2]float64{1, 3})
	lim := circleMax(gInf)
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	finite, W, ok := x.limitVerdict(lim, 1e-10, 20, 1e6)
	if !ok || finite || !(W > 400) {
		t.Fatalf("limitVerdict = %v, %g, %v; want +Inf from a horizon past 400", finite, W, ok)
	}
	level := lim * (1 + 1e-10)
	for _, w1 := range []float64{100, 200} {
		w2 := 2 * w1
		r := x.radius(1 / w1)
		mask := x.phaseMask(1/w2, 1/w1, level-x.remainder(e.tailGain(0, r), r/w1))
		kept := 0
		for _, m := range mask {
			if m {
				kept++
			}
		}
		if !(kept < len(mask)/2) {
			t.Fatalf("octave [%g, %g]: phaseMask keeps %d of %d phases", w1, w2, kept, len(mask))
		}
		for w := w1; w <= w2; w += 1e-3 {
			if !mask[x.sampleAt(w*h)] {
				if g := gain(w); !(g <= level) {
					t.Fatalf("|G(j%g)| = %.15g above %.15g at a phase phaseMask leaves out", w, g, level)
				}
			}
		}
	}
	if want, _ := oraclePeak(gain, 3000, 2e-3); !(want < lim) {
		t.Fatalf("oracle peak %.12g reaches sup |G∞| %.12g", want, lim)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-9*lim {
		t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
	}
}
