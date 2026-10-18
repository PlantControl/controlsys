package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestDelayNyquistSixfoldNearAxisRoot: each loop has a closed-loop root
// within 4e-3 of the axis (Newton on χ) and no RHP root (rectRoots), so its
// 6-fold replica has det(I − H22Δ) = (1+L₁)⁶ between 1e-13 and 1e-20 near
// the root. An absolute 1e-12 floor on |1+L| declared that an axis root and
// the replica unstable; below 1e-16 the sample 1 + (det − 1) is rounding
// alone, so 1+L must be formed as det directly.
func TestDelayNyquistSixfoldNearAxisRoot(t *testing.T) {
	for _, c := range []struct {
		name string
		p    sisoDelayLoop
		root complex128
	}{
		{"pcg0-1916", sisoDelayLoop{a: -0.9152577921210363, b1: -0.16048805105406225, b2: 1.5909595959213585,
			c1: 1.6081167881502552, c2: 0.69971586452038, d11: -0.7450155505260558, d12: 1.9780468068929176,
			d21: -0.4756021924018433, d22: -0.22565532827439366, h: 1.5595406155900269}, -0.0032411667},
		{"pcg3-1012", sisoDelayLoop{a: -1.4993630528685542, b1: 0.36156773258447944, b2: 4.479218753198917,
			c1: 1.601528002694379, c2: 0.18144874769082264, d11: 2.2225489924899557, d12: 0.9541332640591507,
			d21: -0.5205014041808376, d22: 0.45389118438545073, h: 0.7027695697025248}, -0.0038018375},
		{"pcg3-1530", sisoDelayLoop{a: -0.4434243465876844, b1: 0.18119827899774696, b2: 0.7413873232125012,
			c1: -1.2555390938558657, c2: -1.4544758552057484, d11: 0.24405386530661488, d12: -0.7104932753569169,
			d21: 0.49674482991137303, d22: -0.30658129709177573, h: 1.865063162626451}, complex(-0.000736124, 1.1897054629)},
		{"pcg0-1367", sisoDelayLoop{a: -1.1318666079298307, b1: -0.059632552063999054, b2: 0.3032475516276158,
			c1: 2.133887770613745, c2: 1.7072144392853401, d11: -1.1261998123108443, d12: 0.5865244472019743,
			d21: -0.3925027528156678, d22: 0.542198657365, h: 1.3494881236923186}, -0.000232507},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := c.p
			chi := func(s complex128) complex128 {
				e := cmplx.Exp(-s * complex(p.h, 0))
				return (s-complex(p.a, 0))*(1-complex(p.d22, 0)*e) - complex(p.b2*p.c2, 0)*e
			}
			s := complex(0, imag(c.root))
			for range 60 {
				const d = 1e-7
				s -= chi(s) * complex(d, 0) / (chi(s+d) - chi(s))
			}
			if cmplx.Abs(s-c.root) > 1e-9 {
				t.Fatalf("oracle root %v, want %v", s, c.root)
			}
			if n := p.rectRoots(); n != 0 {
				t.Fatalf("oracle %d RHP roots", n)
			}
			want, _, err := HinfNorm(p.siso(t))
			if err != nil || math.IsInf(want, 1) {
				t.Fatalf("SISO HinfNorm = %g, %v", want, err)
			}
			sys := p.replicate(t, 1, 1, 1, 1, 1, 1)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			_, res, err := e.stability()
			if err != nil || !res.stable || res.axisRoot || res.roots != 0 {
				t.Fatalf("stability: stable %v, roots %d, axis root %v at %g, %v", res.stable, res.roots, res.axisRoot, res.axisW, err)
			}
			got, _, err := HinfNorm(sys)
			if err != nil || math.Abs(got-want) > 1e-6*want {
				t.Errorf("HinfNorm = %g, %v; SISO %g", got, err, want)
			}
		})
	}
}

// TestDelayNyquistReplicatedAxisRoot: χ = (s−a)(1−d22·e^{−sh}) − b2c2·e^{−sh}
// vanishes at s = 2j (a = −1, d22 = 0, b2c2 = −√5, h = (π − atan 2)/2) and
// at s = 0 (b2c2 = −a(1−d22)); every replica keeps that axis root and must
// be reported unstable, not resolved by the direct determinant.
func TestDelayNyquistReplicatedAxisRoot(t *testing.T) {
	for _, c := range []struct {
		p sisoDelayLoop
		w float64
	}{
		{sisoDelayLoop{a: -1, b1: 0.2, b2: -math.Sqrt(5), c1: 0.7, c2: 1, d11: 0.3, d12: 1, d21: 1, h: (math.Pi - math.Atan(2)) / 2}, 2},
		{sisoDelayLoop{a: -1.3, b1: 0.2, b2: 1.3 * 0.6, c1: 0.7, c2: 1, d11: 0.3, d12: 1, d21: 1, d22: 0.4, h: 0.7}, 0},
	} {
		p := c.p
		e := cmplx.Exp(complex(0, -c.w*p.h))
		if chi := (complex(-p.a, c.w))*(1-complex(p.d22, 0)*e) - complex(p.b2*p.c2, 0)*e; cmplx.Abs(chi) > 1e-14 {
			t.Fatalf("ω=%g: χ(jω) = %v", c.w, chi)
		}
		for _, r := range []int{1, 2, 6} {
			ks := make([]float64, r)
			for i := range ks {
				ks[i] = 1
			}
			sys := p.replicate(t, ks...)
			lft, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			_, res, err := lft.stability()
			if err != nil || res.stable || !res.axisRoot || math.Abs(res.axisW-c.w) > 1e-2 {
				t.Errorf("ω=%g r=%d: stable %v, axis root %v at %g, %v", c.w, r, res.stable, res.axisRoot, res.axisW, err)
			}
			if got, _, err := HinfNorm(sys); err != nil || !math.IsInf(got, 1) {
				t.Errorf("ω=%g r=%d: HinfNorm = %g, %v", c.w, r, got, err)
			}
		}
	}
}
