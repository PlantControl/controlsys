package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestDelayNyquistSixfoldNearAxisRoot: each loop has a real closed-loop root
// within 4e-3 of the axis (Newton on χ) and no RHP root (rectRoots), so its
// 6-fold replica has det(I − H22Δ) = (1+L₁)⁶ of order 1e-13 near ω = 0. An
// absolute 1e-12 floor on |1+L| declared that an axis root and the replica
// unstable.
func TestDelayNyquistSixfoldNearAxisRoot(t *testing.T) {
	for _, c := range []struct {
		name string
		p    sisoDelayLoop
		root float64
	}{
		{"pcg0-1916", sisoDelayLoop{a: -0.9152577921210363, b1: -0.16048805105406225, b2: 1.5909595959213585,
			c1: 1.6081167881502552, c2: 0.69971586452038, d11: -0.7450155505260558, d12: 1.9780468068929176,
			d21: -0.4756021924018433, d22: -0.22565532827439366, h: 1.5595406155900269}, -0.0032411667},
		{"pcg3-1012", sisoDelayLoop{a: -1.4993630528685542, b1: 0.36156773258447944, b2: 4.479218753198917,
			c1: 1.601528002694379, c2: 0.18144874769082264, d11: 2.2225489924899557, d12: 0.9541332640591507,
			d21: -0.5205014041808376, d22: 0.45389118438545073, h: 0.7027695697025248}, -0.0038018375},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := c.p
			chi := func(s complex128) complex128 {
				e := cmplx.Exp(-s * complex(p.h, 0))
				return (s-complex(p.a, 0))*(1-complex(p.d22, 0)*e) - complex(p.b2*p.c2, 0)*e
			}
			s := complex(0, 0)
			for range 60 {
				const d = 1e-7
				s -= chi(s) * complex(d, 0) / (chi(s+d) - chi(s))
			}
			if math.Abs(real(s)-c.root) > 1e-9 || imag(s) != 0 {
				t.Fatalf("oracle root %v, want %g", s, c.root)
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
