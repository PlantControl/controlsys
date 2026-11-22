package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Frozen census literals (JV4E3K); ref holds 320-bit Hankel singular values.
type balanceCertCase struct {
	name       string
	n, m, p    int
	a, b, c, d []float64
	dt         float64
	ref        []float64
}

var (
	// rlrepd/703 arm64: rotated 0.5·I, HSV [σ₁, 5e-32, 0]; base ‖TL·TR−I‖ = 103.
	certRlrepd703Arm = balanceCertCase{
		name: "rotated scalar discrete arm64", n: 3, m: 1, p: 1, dt: 0.5,
		a:   []float64{-0.5, -2.1050085580217597e-17, -1.0157594003912376e-17, -2.1050085580217597e-17, -0.5000000000000001, -6.173738966931341e-17, -1.0157594003912376e-17, -6.173738966931341e-17, -0.5000000000000001},
		b:   []float64{0, 1, 0},
		c:   []float64{2, 0.5, 1},
		d:   []float64{0},
		ref: []float64{0.666666666666667, 5.367278321793786e-32, 0},
	}
	// rlrepd/703 amd64 literal.
	certRlrepd703Amd = balanceCertCase{
		name: "rotated scalar discrete amd64", n: 3, m: 1, p: 1, dt: 0.5,
		a:   []float64{-0.49999999999999994, 0, -2.0816681711721685e-17, 0, -0.5000000000000001, -4.85722573273506e-17, -2.0816681711721685e-17, -4.85722573273506e-17, -0.5000000000000001},
		b:   []float64{0, 1, 0},
		c:   []float64{2, 0.5, 1},
		d:   []float64{0},
		ref: []float64{0.6666666666666669, 3.5950692295228446e-33, 9.070568121743171e-64},
	}
	// rotQ/60: σ₂ rounds to 0, base ‖TL·TR−I‖ = 2e15.
	certRotQ60 = balanceCertCase{
		name: "rotated MIMO HSV rounding to zero", n: 2, m: 2, p: 2,
		a:   []float64{-1.0000000000000002, 1.071516862846046e-16, 1.071516862846046e-16, -1},
		b:   []float64{0, 0, -1, 1},
		c:   []float64{1, -1, 0.5, 2},
		d:   []float64{-1, -1, -1, 1},
		ref: []float64{1.5811388300841898, 4.2355422973396943e-17},
	}
	// Non-symmetric near-scalar A with distinct rounding-level couplings.
	certNonSymScalar = balanceCertCase{
		name: "non-symmetric near-scalar", n: 3, m: 1, p: 1,
		a: []float64{-0.5, 1.3535241512296794e-17, -9.772295458339984e-18, -2.3097258622778364e-18, -0.5, -6.080205957556177e-18, 1.2236835328237762e-17, -2.611635645334779e-17, -0.5},
		b: []float64{1, -1, 0.5},
		c: []float64{0.5, 1, 0},
		d: []float64{0},
	}
	// rotZ/855 arm64: lightly damped pairs; exact ‖TL·TR−I‖ = 1.05e-5 ≤ τ = 2⁻¹⁰.
	certRotZ855 = balanceCertCase{
		name: "lightly damped pairs", n: 6, m: 1, p: 1,
		a:   []float64{-0.5, 1, 0.5, -0.5, 0, 0, -1, -0.5, 0, -0.5, 1, 0.5, 0, 0, -2.9802322387695312e-08, 0.5, -0.5, 1, 0, 0, -0.5, -2.9802322387695312e-08, -1, -1, 0, 0, 0, 0, -2.9802322387695312e-08, 0.5, 0, 0, 0, 0, -0.5, -2.9802322387695312e-08},
		b:   []float64{1, -1, -1, 1, 1, 2},
		c:   []float64{2, 1, 1, 1, -1, 2},
		d:   []float64{0},
		ref: []float64{1.238240386503125e+15, 1.238240346464547e+15, 2.1244848502971612e+14, 2.1244844499113584e+14, 8.477804015171927, 3.327805802715246},
	}
	// rotZ/697 arm64: exact ‖TL·TR−I‖ = 7.2e-6.
	certRotZ697 = balanceCertCase{
		name: "near-integrator MIMO", n: 6, m: 2, p: 1,
		a:   []float64{-2.3283064365386963e-10, -2, -0.5, 0, 0, -1, 2, -2.3283064365386963e-10, 1, 0.5, -1, -0.5, 0, 0, -1.1641532182693481e-10, 0, -0.5, 1, 0, 0, 0, -2, -1, 1, 0, 0, 0.5, 0, -1.1641532182693481e-10, -1, 0, 0, 0, 0, 0, -9.313225746154785e-10},
		b:   []float64{0, -1, -1, 0, -1, 2, 0, 0.5, -1, 2, 2, 0.5},
		c:   []float64{-1, 1, 2, 1, 1, 0},
		d:   []float64{0, 0},
		ref: []float64{3.524713440178828e+10, 3.524713438859081e+10, 4.565499413206939e+09, 3.9501508435421066e+09, 3.950150843165614e+09, 0.3355270405505417},
	}
	// rotZ/526 arm64: exact ‖TL·TR−I‖ = 2.3e-8.
	certRotZ526 = balanceCertCase{
		name: "near-integrator 2x2", n: 5, m: 2, p: 2,
		a:   []float64{-3.814697265625e-06, 0, 0, 0, 0, 1, -1, -2, 0, 0, 0.5, 2, -1, 0, 0, 0.5, 0, 0, -4.656612873077393e-10, -1, 0, 0, 0, 1, -4.656612873077393e-10},
		b:   []float64{2, -1, 1, 0.5, 2, 2, 0, 1, 1, 0.5},
		c:   []float64{0.5, 0.5, -1, -1, 0, -1, 2, 2, 0, 0.5},
		d:   []float64{0, 0, 0, 0},
		ref: []float64{8.488666364925267e+08, 8.488666360315495e+08, 73271.25187978732, 2.1397965767162974, 1.9841561679405209},
	}
	// rlrepd/746 arm64: exact ‖TL·TR−I‖ = 0.0208, float64 product 2.7e7.
	certRlrepd746 = balanceCertCase{
		name: "tiny-gain rotated discrete", n: 2, m: 1, p: 1, dt: 0.5,
		a: []float64{0.24999999999999994, -5.5927857926893454e-18, -5.5927857926893454e-18, 0.24999999999999994},
		b: []float64{-1, 0.5},
		c: []float64{0.5, 1},
		d: []float64{0},
	}
	// rlrep/515 amd64: σ₄, σ₅ below n·ε·σ₁ (320-bit 3e-32, 1e-32).
	certRlrep515 = balanceCertCase{
		name: "sub-floor tail pair", n: 5, m: 1, p: 1,
		a:   []float64{-0.8963685406536798, 0.7510368352478014, 0.724778328579311, 0.7502751714527868, -0.35011398845919717, -1.0695554318122542, -0.1917550427317112, 0.11250994199987713, 0.5375694864905084, -1.1991125616444513, -0.9187788923176139, -0.05029123965489701, -0.5105429189905807, 0.38565517204536826, 0.4478583115163069, 0.2141269673713682, 0.4881356226163699, -0.02948287999069682, -0.8130599443621043, 0.40789702223623336, -0.18535483290323337, 0.6127021009610332, -0.6487999471489236, -0.07474543430334535, -0.5882735532619243},
		b:   []float64{1, 0, 2, 0.5, 2},
		c:   []float64{1, -1, 0.5, 0.5, 0.5},
		d:   []float64{0},
		ref: []float64{3.0957351359294565, 1.5614689384347356, 0.08718460538026614, 3.1495676761082397e-32, 1.296561631962041e-32},
	}
	// rlrepd/179 arm64: σ₃, σ₄ are rounding noise (320-bit 9e-32, 4e-32).
	certRlrepd179 = balanceCertCase{
		name: "rotated discrete noise pair", n: 4, m: 1, p: 1, dt: 0.5,
		a:   []float64{0.24999999999999994, 0.2035506980756449, -0.4536614223607592, -0.052520730902478135, -0.2035506980756449, 0.25000000000000006, 0.05252073090247818, -0.4536614223607592, 0.4536614223607592, -0.05252073090247826, 0.25, -0.2035506980756449, 0.05252073090247822, 0.4536614223607593, 0.20355069807564483, 0.25},
		b:   []float64{1, -1, -1, -1},
		c:   []float64{2, 1, 0.5, 1},
		d:   []float64{0},
		ref: []float64{0.843085846998223, 0.6276682775190372, 9.107748948585846e-32, 3.891549238765644e-32},
	}
)

func (tc balanceCertCase) sys(t *testing.T) *System {
	t.Helper()
	sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// exactCertNorm returns ‖TL·TR − I‖_F evaluated in 256-bit arithmetic.
func exactCertNorm(tl, tr *mat.Dense) float64 {
	n, _ := tl.Dims()
	const prec = 256
	sum := new(big.Float).SetPrec(prec)
	for i := range n {
		for j := range n {
			s := new(big.Float).SetPrec(prec)
			if i == j {
				s.SetInt64(-1)
			}
			for k := range n {
				p := new(big.Float).SetPrec(prec).SetFloat64(tl.At(i, k))
				p.Mul(p, new(big.Float).SetPrec(prec).SetFloat64(tr.At(k, j)))
				s.Add(s, p)
			}
			sum.Add(sum, s.Mul(s, s))
		}
	}
	v, _ := sum.Sqrt(sum).Float64()
	return v
}

func certTau(hsv []float64) float64 {
	n := len(hsv)
	if !(hsv[n-1] > 0) {
		return 0x1p-10
	}
	return math.Min(0x1p-10, math.Max(1e-8, 4*float64(n)*0x1p-52*hsv[0]/hsv[n-1]))
}

// maxResponseGap returns max over ω of max |G(jω) − Gr(jω)| (z = e^{jωDt}).
func maxResponseGap(t *testing.T, g, gr *System) float64 {
	gap, _ := responseGap(t, g, gr)
	return gap
}

// responseGap returns maxResponseGap and max over ω of max |G(jω)|.
func responseGap(t *testing.T, g, gr *System) (gap, top float64) {
	t.Helper()
	om := []float64{0.01, 0.1, 0.4, 1, 2.5, 6}
	if g.Dt > 0 {
		om = []float64{0.01, 0.1, 0.4, 1, 2.5, 5}
	}
	a, err := g.FreqResponse(om)
	if err != nil {
		t.Fatal(err)
	}
	b, err := gr.FreqResponse(om)
	if err != nil {
		t.Fatal(err)
	}
	_, m, p := g.Dims()
	for k := range om {
		for i := range p {
			for j := range m {
				gap = math.Max(gap, cmplx.Abs(a.At(k, i, j)-b.At(k, i, j)))
				top = math.Max(top, cmplx.Abs(a.At(k, i, j)))
			}
		}
	}
	return gap, top
}

// exactResponseGap is maxResponseGap with both responses of the stored data
// evaluated exactly.
func exactResponseGap(t *testing.T, g, gr *System) float64 {
	t.Helper()
	gap := 0.0
	for _, w := range []float64{0.01, 0.1, 0.4, 1, 2.5} {
		s := complex(0, w)
		if g.Dt > 0 {
			s = cmplx.Exp(complex(0, w*g.Dt))
		}
		ga, ok1 := exactStoredResponse(g, s)
		gb, ok2 := exactStoredResponse(gr, s)
		if !ok1 || !ok2 {
			t.Fatalf("exact response failed at %v", s)
		}
		for i := range ga {
			gap = math.Max(gap, cmplx.Abs(ga[i]-gb[i]))
		}
	}
	return gap
}

func exactStoredResponse(sys *System, s complex128) ([]complex128, bool) {
	n, m, p := sys.Dims()
	raw := func(x *mat.Dense) []float64 { return mat.DenseCopyOf(x).RawMatrix().Data }
	return fzExactResponse(n, m, p, raw(sys.A), raw(sys.B), raw(sys.C), raw(sys.D), s)
}

func requireStable(t *testing.T, name string, sys *System) {
	t.Helper()
	st, err := sys.IsStable()
	if err != nil || !st {
		t.Fatalf("%s: stable = %v, %v", name, st, err)
	}
}

// JV4E3K: an exactly minimal model whose HSV fall below the rounding of σ₁
// has no float64 balancing with TL·TR = I; Balreal must not return one.
func TestBalreal_NumericallyNonMinimalErrors(t *testing.T) {
	for _, tc := range []balanceCertCase{certRlrepd703Arm, certRlrepd703Amd, certRotQ60, certNonSymScalar} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			if !ratMinimal(sys) {
				t.Fatal("fixture not exactly minimal")
			}
			br, err := Balreal(sys)
			if !errors.Is(err, ErrSingularTransform) || br != nil {
				t.Fatalf("Balreal = %v, %v; want ErrSingularTransform", br, err)
			}
		})
	}
}

// JV4E3K: transformations within τ_n of TL·TR = I stay nil, including the
// lightly damped pairs fixture (rotZ/855) whose float64 balancing is off by
// 1e-5 but whose HSV are far above the rounding of σ₁.
func TestBalreal_CertifiedTransformation(t *testing.T) {
	for _, tc := range []balanceCertCase{certRotZ855, certRotZ697, certRotZ526} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			br, err := Balreal(sys)
			if err != nil {
				t.Fatalf("Balreal: %v", err)
			}
			f, tau := exactCertNorm(br.TL, br.TR), certTau(br.HSV)
			if !(8*f <= tau) {
				t.Fatalf("exact ‖TL·TR−I‖ = %.3g, τ = %.3g: under 8× margin", f, tau)
			}
			balrealHSVWithin(t, br.HSV, tc.ref, 1e-12)
			_, top := responseGap(t, sys, br.Sys)
			if gap := exactResponseGap(t, sys, br.Sys); gap > 100*f*top+1e-12*tc.ref[0] {
				t.Fatalf("|G − G_bal| = %.3g, max |G| = %.3g, ‖TL·TR−I‖ = %.3g", gap, top, f)
			}
			requireStable(t, "Balreal", br.Sys)
		})
	}
}

// JV4E3K: the compensated TL·TR − I matches 256-bit arithmetic where the
// float64 product is off by 9 orders of magnitude (rlrepd/746).
func TestBalanceCert_CompensatedProduct(t *testing.T) {
	sys := certRlrepd746.sys(t)
	br, err := balanceUncertified(sys)
	if err != nil {
		t.Fatal(err)
	}
	c := newBalanceCert(br)
	c.compute()
	got := math.Sqrt(c.prefix[2])
	want := exactCertNorm(br.TL, br.TR)
	if math.Abs(got-want) > 1e-6*want {
		t.Fatalf("compensated ‖TL·TR−I‖ = %.17g, exact %.17g", got, want)
	}
	var p mat.Dense
	p.Mul(br.TL, br.TR)
	p.Set(0, 0, p.At(0, 0)-1)
	p.Set(1, 1, p.At(1, 1)-1)
	if fl := mat.Norm(&p, 2); !(fl > 1e3*want) {
		t.Logf("float64 product no longer fails here (%.3g vs exact %.3g)", fl, want)
	}
	if want < 1e-3 || want > 1 {
		t.Fatalf("exact ‖TL·TR−I‖ = %.3g, fixture expects ~0.02", want)
	}
}

// JV4E3K: Balred reduces the certified head (rlrepd/703) at orders it
// certifies, within 2·tail in both projections, and names the certified
// order above it.
func TestBalred_CertifiedHeadOrder(t *testing.T) {
	for _, tc := range []balanceCertCase{certRlrepd703Arm, certRlrepd703Amd} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			for _, proj := range []StateProjection{MatchDC, Truncate} {
				red, hsv, err := Balred(sys, 1, BalredOptions{StateProjection: proj})
				if err != nil {
					t.Fatalf("Balred(1, %v): %v", proj, err)
				}
				tail := 2 * (tc.ref[1] + tc.ref[2])
				if gap := maxResponseGap(t, sys, red); gap > tail+1e-13*hsv[0] {
					t.Fatalf("Balred(1, %v): |G − Gr| = %.3g > 2·tail %.3g", proj, gap, tail)
				}
				requireStable(t, "Balred", red)
			}
			_, _, err := Balred(sys, 2, BalredOptions{})
			if !errors.Is(err, ErrSingularTransform) || !strings.Contains(err.Error(), "certified order 1") {
				t.Fatalf("Balred(2) err = %v; want ErrSingularTransform naming certified order 1", err)
			}
			if _, _, err := Balred(sys, 3, BalredOptions{}); !errors.Is(err, ErrSingularTransform) {
				t.Fatalf("Balred(3) err = %v; want ErrSingularTransform as Balreal", err)
			}
		})
	}
}

// JV4E3K: Balred never keeps part of a tail below n·ε·σ₁ (rlrep/515).
func TestBalred_NoSplitSubFloorTail(t *testing.T) {
	sys := certRlrep515.sys(t)
	if _, _, err := Balred(sys, 4, BalredOptions{}); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("Balred(4) err = %v; want ErrSingularTransform", err)
	}
	for _, proj := range []StateProjection{MatchDC, Truncate} {
		red, _, err := Balred(sys, 3, BalredOptions{StateProjection: proj})
		if err != nil {
			t.Fatalf("Balred(3, %v): %v", proj, err)
		}
		requireStable(t, "Balred(3)", red)
		if gap := maxResponseGap(t, sys, red); gap > 1e-12*certRlrep515.ref[0] {
			t.Fatalf("Balred(3, %v): |G − Gr| = %.3g", proj, gap)
		}
	}
}

// JV4E3K: Balred certifies the noise-free head of rlrepd/179 and rejects the
// full order. σ₃ is rounding noise just above n·ε·σ₁, so order 3 may be
// certified or refused depending on the platform's rounding.
func TestBalred_NoiseTailCertifiedOrder(t *testing.T) {
	sys := certRlrepd179.sys(t)
	for k := 0; k <= 3; k++ {
		for _, proj := range []StateProjection{MatchDC, Truncate} {
			red, _, err := Balred(sys, k, BalredOptions{StateProjection: proj})
			if k == 3 && errors.Is(err, ErrSingularTransform) {
				continue
			}
			if err != nil {
				t.Fatalf("Balred(%d, %v): %v", k, proj, err)
			}
			requireStable(t, "Balred", red)
			tail := 0.0
			for i := k; i < 4; i++ {
				tail += certRlrepd179.ref[i]
			}
			if gap := maxResponseGap(t, sys, red); gap > 2*tail+1e-12*certRlrepd179.ref[0] {
				t.Fatalf("Balred(%d, %v): |G − Gr| = %.3g > 2·tail %.3g", k, proj, gap, 2*tail)
			}
		}
	}
	if _, _, err := Balred(sys, 4, BalredOptions{}); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("Balred(4) err = %v; want ErrSingularTransform", err)
	}
}

// JV4E3K: on a certified, numerically minimal model Balred is bit-identical
// to Modred of Balreal, and Balred(n) is Balreal.
func TestBalred_CertifiedPathUnchanged(t *testing.T) {
	sys, err := NewFromSlices(4, 2, 2,
		[]float64{-1, 0.5, 0, 0.25, -0.5, -2, 1, 0, 0, -0.75, -3, 0.5, 0.25, 0, -1, -4},
		[]float64{1, 0, 0.5, 1, -1, 0.5, 0, 2},
		[]float64{1, 0, -0.5, 2, 0, 1, 1, -1},
		[]float64{0, 0.5, 0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	br, err := Balreal(sys)
	if err != nil {
		t.Fatal(err)
	}
	full, hsv, err := Balred(sys, 4, BalredOptions{})
	if err != nil || !mat.Equal(full.A, br.Sys.A) || !mat.Equal(full.B, br.Sys.B) || !mat.Equal(full.C, br.Sys.C) || hsv[3] != br.HSV[3] {
		t.Fatalf("Balred(n) differs from Balreal: %v", err)
	}
	for _, proj := range []StateProjection{MatchDC, Truncate} {
		for k := 0; k < 4; k++ {
			red, _, err := Balred(sys, k, BalredOptions{StateProjection: proj})
			if err != nil {
				t.Fatal(err)
			}
			var elim []int
			for i := k; i < 4; i++ {
				elim = append(elim, i)
			}
			want, err := Modred(br.Sys, elim, proj)
			if err != nil {
				t.Fatal(err)
			}
			if !mat.Equal(red.D, want.D) || k > 0 && (!mat.Equal(red.A, want.A) || !mat.Equal(red.B, want.B) || !mat.Equal(red.C, want.C)) {
				t.Fatalf("Balred(%d, %v) not bit-identical to Modred(Balreal)", k, proj)
			}
		}
	}
}

// JV4E3K: Balred(n) returns Balreal's error when the transformation fails.
func TestBalred_FullOrderIsBalreal(t *testing.T) {
	for _, tc := range []balanceCertCase{certRlrepd703Arm, certRotQ60, certRotZ855} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys(t)
			br, err1 := Balreal(sys)
			red, _, err2 := Balred(sys, tc.n, BalredOptions{})
			if errors.Is(err1, ErrSingularTransform) != errors.Is(err2, ErrSingularTransform) || (err1 == nil) != (err2 == nil) {
				t.Fatalf("Balreal err %v, Balred(n) err %v", err1, err2)
			}
			if err1 == nil && !mat.Equal(red.A, br.Sys.A) {
				t.Fatal("Balred(n) differs from Balreal")
			}
		})
	}
}

// balanceCertHolds checks a nil Balreal against the exact certificate.
func balanceCertHolds(t *testing.T, br *BalrealResult, what string) {
	t.Helper()
	if f, tau := exactCertNorm(br.TL, br.TR), certTau(br.HSV); f > tau {
		t.Fatalf("%s: nil with exact ‖TL·TR−I‖ = %.3g > τ = %.3g", what, f, tau)
	}
	requireStable(t, what, br.Sys)
}
