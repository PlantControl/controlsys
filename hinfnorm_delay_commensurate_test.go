package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCommensurateMultiples(t *testing.T) {
	for _, tc := range []struct {
		tau  []float64
		want []int
	}{
		{[]float64{0.7}, []int{1}},
		{[]float64{0, 0}, []int{1, 1}},
		{[]float64{0.7, 1.4}, []int{1, 2}},
		{[]float64{1, 2, 1}, []int{1, 2, 1}},
		{[]float64{0.75, 0.5}, []int{3, 2}},
		{[]float64{0.25, 0.75, 1.25}, []int{1, 3, 5}},
		{[]float64{0.1, 0.2}, []int{1, 2}},
		// 0.8 and 0.3 are multiples of 0.5 and 0.1 only up to rounding.
		{[]float64{0.5, 0.8}, []int{5, 8}},
		{[]float64{0.1, 0.3}, []int{1, 3}},
		{[]float64{0.3, 0.1}, []int{3, 1}},
		{[]float64{0.2, 0.3}, []int{2, 3}},
		{[]float64{0.1, 0.7, 0.3}, []int{1, 7, 3}},
		{[]float64{1, 3 * (1 + 64*0x1p-52)}, nil},
		{[]float64{1, 3 * (1 + 1e-12)}, nil},
		{[]float64{1, math.Sqrt2}, nil},
		{[]float64{0, 1}, nil},
		{[]float64{1, 65}, nil},
		{[]float64{1, 64}, nil},
		{[]float64{1, 63}, []int{1, 63}},
		{[]float64{0x1p-100, 0x1p100}, nil},
	} {
		if got := commensurateMultiples(tc.tau); !slices.Equal(got, tc.want) {
			t.Errorf("commensurateMultiples(%v) = %v, want %v", tc.tau, got, tc.want)
		}
	}
}

// reciprocalRoots returns the roots of z³·p(1/z) = z³ − a·z² − d·z + det D22
// for p(z) = det(I − D22·diag(z, z²)) of a 2×2 D22 = [a b; c d], from its
// companion matrix: the reciprocals of the zeros of p.
func reciprocalRoots(d22 []float64) []complex128 {
	a, b, c, d := d22[0], d22[1], d22[2], d22[3]
	comp := mat.NewDense(3, 3, []float64{a, d, -(a*d - b*c), 1, 0, 0, 0, 1, 0})
	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		return nil
	}
	return eig.Values(nil)
}

func maxAbs(v []complex128) float64 {
	m := 0.0
	for _, x := range v {
		m = max(m, cmplx.Abs(x))
	}
	return m
}

// notStronglyStable reports ρ(D22·diag(1, −1)) > 1 for a 2×2 D22.
func notStronglyStable(d22 []float64) bool {
	var eig mat.Eigen
	theta := mat.NewDense(2, 2, []float64{d22[0], -d22[1], d22[2], -d22[3]})
	return eig.Factorize(theta, mat.EigenNone) && maxAbs(eig.Values(nil)) > 1
}

// TestNeutralCommensurateDelays also takes τ_2 one ulp off 2·τ_1, a
// multiple up to rounding, which is the same commensurate model.
func TestNeutralCommensurateDelays(t *testing.T) {
	for _, tc := range []struct {
		name   string
		d22    []float64
		stable bool
		tau    []float64
	}{
		// det(I − D22·diag(z, z²)) = 1 − 0.54z + 0.54z² + 0.0684z³ has all
		// zeros outside the unit disk.
		{"stable", []float64{0.54, 0.6, -0.6, -0.54}, true, []float64{0.7, 1.4}},
		{"unstable", []float64{0.6, 1, -1, -0.9}, false, []float64{0.7, 1.4}},
		{"stable-rounded", []float64{0.54, 0.6, -0.6, -0.54}, true, []float64{0.7, math.Nextafter(1.4, 2)}},
		{"unstable-rounded", []float64{0.6, 1, -1, -0.9}, false, []float64{0.7, math.Nextafter(1.4, 2)}},
	} {
		tau := tc.tau
		t.Run(tc.name, func(t *testing.T) {
			if got := maxAbs(reciprocalRoots(tc.d22)) < 1; got != tc.stable {
				t.Fatalf("companion reference stable = %v, want %v", got, tc.stable)
			}
			if !notStronglyStable(tc.d22) {
				t.Fatal("case must not be strongly stable")
			}
			sys, gain := constantDifferenceLoop(t, tc.d22, tau)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if l := e.tail(math.Inf(1)); l < 1 {
				t.Fatalf("tail(∞) = %g; the case must defeat the norm and permanent bounds", l)
			}
			if d := e.difference(); !math.IsInf(d.polydisk, 1) || math.IsInf(d.rhp, 1) == tc.stable {
				t.Fatalf("bounds %+v", *d)
			}
			got, err := sys.IsStable()
			if err != nil || got != tc.stable {
				t.Fatalf("IsStable = %v, %v; want %v", got, err, tc.stable)
			}
			n, _, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if !tc.stable {
				if !math.IsInf(n, 1) {
					t.Fatalf("HinfNorm = %g, want +Inf", n)
				}
				return
			}
			want, _ := oraclePeak(gain, 80, 1e-3)
			if math.Abs(n-want) > 1e-8*want {
				t.Fatalf("HinfNorm = %.12g, want %.12g", n, want)
			}
		})
	}
}

// feedthroughDifferenceLoop returns G = D11 + C1(sI−A)⁻¹B1 +
// (D12 + C1(sI−A)⁻¹B2)·Δ(I − D22Δ)⁻¹·D21 with C2 = 0, so H22 = D22; gain
// evaluates |G(jω)| and gInf |G∞(φ)| with Δ_i = e^{−j·k_i·φ} in closed form.
func feedthroughDifferenceLoop(t testing.TB, d22, d12, d21, tau []float64, k [2]float64) (*System, func(float64) float64, func(float64) float64) {
	t.Helper()
	sys, resp, gInf := feedthroughDifferenceResponse(t, d22, d12, d21, tau, k)
	return sys, func(w float64) float64 { return cmplx.Abs(resp(w)) }, gInf
}

// feedthroughDifferenceResponse is feedthroughDifferenceLoop with the
// response G(jω) itself.
func feedthroughDifferenceResponse(t testing.TB, d22, d12, d21, tau []float64, k [2]float64) (*System, func(float64) complex128, func(float64) float64) {
	t.Helper()
	const d11 = 0.1
	a := []float64{-1, 2, -0.7, -3}
	b1 := []float64{1, 0.5}
	b2 := []float64{0.3, 0, 0, 1}
	c1 := []float64{1, -0.4}
	H, err := NewFromSlices(2, 3, 3,
		a,
		[]float64{b1[0], b2[0], b2[1], b1[1], b2[2], b2[3]},
		[]float64{c1[0], c1[1], 0, 0, 0, 0},
		[]float64{d11, d12[0], d12[1], d21[0], d22[0], d22[1], d21[1], d22[2], d22[3]}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	c := func(x float64) complex128 { return complex(x, 0) }
	loop := func(e1, e2 complex128) (complex128, complex128) {
		m := [4]complex128{1 - c(d22[0])*e1, -c(d22[1]) * e2, -c(d22[2]) * e1, 1 - c(d22[3])*e2}
		det := m[0]*m[3] - m[1]*m[2]
		v0 := (m[3]*c(d21[0]) - m[1]*c(d21[1])) / det
		v1 := (m[0]*c(d21[1]) - m[2]*c(d21[0])) / det
		return e1 * v0, e2 * v1
	}
	resp := func(w float64) complex128 {
		s := complex(0, w)
		v0, v1 := loop(cmplx.Exp(-s*c(tau[0])), cmplx.Exp(-s*c(tau[1])))
		u0 := c(b1[0]) + c(b2[0])*v0 + c(b2[1])*v1
		u1 := c(b1[1]) + c(b2[2])*v0 + c(b2[3])*v1
		r := [4]complex128{s - c(a[0]), -c(a[1]), -c(a[2]), s - c(a[3])}
		rd := r[0]*r[3] - r[1]*r[2]
		x0 := (r[3]*u0 - r[1]*u1) / rd
		x1 := (r[0]*u1 - r[2]*u0) / rd
		return c(d11) + c(c1[0])*x0 + c(c1[1])*x1 + c(d12[0])*v0 + c(d12[1])*v1
	}
	gInf := func(phi float64) float64 {
		v0, v1 := loop(cmplx.Exp(complex(0, -k[0]*phi)), cmplx.Exp(complex(0, -k[1]*phi)))
		return cmplx.Abs(c(d11) + c(d12[0])*v0 + c(d12[1])*v1)
	}
	return sys, resp, gInf
}

func TestNeutralOnePhaseFeedthrough(t *testing.T) {
	for _, tc := range []struct {
		name string
		d22  []float64
		tau  []float64
		k    [2]float64
	}{
		// Equal delays: ρ(D22) = √0.19, so the Stein bound alone certifies
		// (I − zD22)⁻¹ on |z| <= 1.
		{"equal-stein", []float64{0.9, 1, -1, -0.9}, []float64{0.7, 0.7}, [2]float64{1, 1}},
		{"commensurate", []float64{0.54, 0.6, -0.6, -0.54}, []float64{0.7, 1.4}, [2]float64{1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !notStronglyStable(tc.d22) {
				t.Fatal("case must not be strongly stable")
			}
			sys, gain, gInf := feedthroughDifferenceLoop(t, tc.d22, []float64{1, -1}, []float64{1, -1}, tc.tau, tc.k)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if !e.directFeedthrough {
				t.Fatal("case must have a delayed feedthrough")
			}
			if d := e.difference(); !math.IsInf(d.polydisk, 1) || math.IsInf(d.rhp, 1) {
				t.Fatalf("bounds %+v, want rhp", *d)
			}
			const k = 1 << 18
			hi := 0.0
			for i := range k {
				hi = max(hi, gInf(2*math.Pi*float64(i)/k))
			}
			hf := newHighFrequencyGain(e)
			if hf.dim != 1 || math.IsInf(hf.lip, 1) {
				t.Fatalf("dim = %d, lip = %g; want one phase, finite", hf.dim, hf.lip)
			}
			if est := hf.estimate(); math.Abs(est-hi) > 1e-6*hi {
				t.Fatalf("estimate = %.12g, want %.12g", est, hi)
			}
			if !hf.below(hi*(1+1e-3), 1<<21) {
				t.Fatal("below(sup·(1+1e-3)) = false")
			}
			if hf.below(hi*(1-1e-4), 1<<21) {
				t.Fatal("below(sup·(1−1e-4)) = true")
			}
			if got, err := sys.IsStable(); err != nil || !got {
				t.Fatalf("IsStable = %v, %v; want true", got, err)
			}
			n, _, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			want, _ := oraclePeak(gain, 80, 1e-3)
			want = max(want, hi)
			if math.Abs(n-want) > 1e-8*want {
				t.Fatalf("HinfNorm = %.12g, want %.12g", n, want)
			}
		})
	}
}
