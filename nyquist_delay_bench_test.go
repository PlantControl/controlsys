package controlsys

import "testing"

func BenchmarkNyquistDelay(b *testing.B) {
	mk := func(a, c, d, tau float64, internal bool) *System {
		sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{1}, []float64{c}, []float64{d}, 0)
		if err != nil {
			b.Fatal(err)
		}
		if err := sys.SetInputDelay([]float64{tau}); err != nil {
			b.Fatal(err)
		}
		if internal {
			if sys, err = sys.PullDelaysToLFT(); err != nil {
				b.Fatal(err)
			}
		}
		return sys
	}
	for _, c := range []struct {
		name string
		sys  *System
	}{
		{"io/k=0.5", mk(-1, 0.5, 0, 1, false)},
		{"io/k=50", mk(-1, 50, 0, 1, false)},
		{"io/D=0.5", mk(-1, 0.5*99, 0.5, 1, false)},
		{"io/fast-small-gain", mk(-1e6, 9e5, 0, 1, false)},
		{"lft/k=0.5", mk(-1, 0.5, 0, 1, true)},
		{"lft/k=50", mk(-1, 50, 0, 1, true)},
		{"lft/fast", mk(-1e3, 500, 0, 1, true)},
	} {
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				if _, err := c.sys.Nyquist(nil, 0); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
